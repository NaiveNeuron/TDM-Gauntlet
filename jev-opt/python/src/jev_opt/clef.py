"""Local Clef / Clef-flash backend for the jev-opt executor (drop-in for `JevClient`).

Selected by `JEV_OPT_BACKEND=clef|clef-flash` (`jev.make_client`). The executor's requests are sent as
they are: no per-question cache tier and no question chunking, because Clef decides all questions of a
request jointly (question ids, order and co-questions are part of its input).

- Inference runs in the repository's gauntlet environment (`uv run --extra clef python -m
  gauntlet.clef_serve`), which reuses `gauntlet.clef.ClefClient`: pinned release and revision, official
  encoder, BF16 model arithmetic, FP32 softmax, the complete-input guard (no truncation) and the official
  length-bucketed batching. This package needs no torch.
- Cache: one JSON file per request under `cache_dir`, keyed by SHA-256 of (pinned model, encoding and
  numerical settings, guard, request body in its original key order). Device, GPU and library versions are
  recorded with each response (`configurations/<sha>.json`), not keyed, so a cache made on one machine
  replays on another. Without a device the client is cache-only and a miss raises `ClefCacheMiss`.
- Concurrent `decide` calls from the executor's threads are gathered into windows of up to `window`
  unique requests (the gauntlet's 64-request windows) and answered by one server call.
- Answers are the release's public four-decimal SystemOne answers, as in the frozen zero-shot audit; the
  unrounded FP32 probabilities stay in each response's `audit`. Local compute is unpriced: `cost` is 0
  and GPU seconds are counted in `fresh_inference_seconds`.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .jev import Decision, ResponseLedger, Usage

REPO = Path(os.environ.get("JEV_OPT_CLEF_REPO") or Path(__file__).resolve().parents[4])
# Pinned releases: identical to gauntlet.clef.MODELS (the server's handshake is checked against them).
PINS = {
    "clef": "Cloudflare/clef@2f3de3dd85f379784083b0814d997ab627200f0c",
    "clef-flash": "Cloudflare/clef-flash@17f0b0ad64efb65d273590632833508766b2aae6",
}
# Settings that determine an answer (with the request); the server's configuration must match them.
ENCODING = {
    "backend": "clef-local-frozen",
    "client_encoding_version": 1,
    "dtype": "bfloat16",
    "probability_dtype": "float32",
    "attention_implementation": "sdpa",
    "truncation_policy": "reject_complete_input_overflow_before_forward",
    "rounding_policy": "official systemone_answer: round(value, 4); raw probabilities retained",
}
WINDOW = 64
LINGER_S = 0.05


class ClefCacheMiss(RuntimeError):
    """A request is not in the Clef cache and the client has no device (cache-only replay)."""


class ClefRejected(RuntimeError):
    """The pinned encoder rejected a request (e.g. complete input over the guard); nothing was truncated."""


class ClefServerError(RuntimeError):
    """The inference server failed or exited."""


class Server(Protocol):
    model: str
    configuration: dict[str, Any]

    def compute(self, items: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]: ...

    def close(self) -> None: ...


def cache_key(model: str, max_length: int, body: dict[str, Any]) -> str:
    """Request hash: pinned model, encoding/numerical settings, guard and the request in key order."""
    canonical = json.dumps({"model": model, "encoding": ENCODING, "max_length": max_length, "request": body},
                           ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def server_command(backend: str, device: str, max_length: int, scratch_dir: Path) -> list[str]:
    """The inference server in the repository's `clef` environment (pinned torch/transformers/kernels)."""
    return ["uv", "run", "--project", str(REPO), "--extra", "clef", "python", "-m", "gauntlet.clef_serve",
            "--model", backend, "--device", device, "--max-length", str(max_length), "--scratch-dir", str(scratch_dir)]


class ClefServer:
    """`gauntlet.clef_serve` subprocess (stderr is inherited)."""

    def __init__(self, backend: str, device: str, max_length: int, scratch_dir: Path, command: list[str] | None = None):
        cmd = command or server_command(backend, device, max_length, scratch_dir)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8", cwd=REPO)
        hello = self._read()
        if not hello.get("ready"):
            raise ClefServerError(f"unexpected handshake {str(hello)[:500]}")
        self.model: str = hello["model"]
        self.configuration: dict[str, Any] = hello["configuration"]

    def _read(self) -> dict[str, Any]:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            raise ClefServerError(f"inference server exited (status {self.proc.wait()})")
        return json.loads(line)

    def compute(self, items: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps({"bodies": items}, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        return self._read()["results"]

    def close(self) -> None:
        if self.proc.stdin is not None and not self.proc.stdin.closed:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


@dataclass
class _Pending:
    body: dict[str, Any]
    future: Future
    payload: dict[str, Any] | None = None  # what the server receives, if not the cached request body itself


class ClefBackend:
    """Clef client with the `JevClient` interface used by `execute` (decide, concurrency, usage, ledger).

    The cache, window gathering and server handling are shared with `jev_opt.logprob.LogprobBackend` through the
    class attributes below and `_default_server`."""

    local = True
    pins = PINS
    encoding = ENCODING
    rejected_error: type[RuntimeError] = ClefRejected
    cache_miss_error: type[RuntimeError] = ClefCacheMiss

    def __init__(self, backend: str, cache_dir: Path, device: str | None = None, max_length: int = 32000,
                 window: int = WINDOW, linger_s: float = LINGER_S, server_factory: Callable[[], Server] | None = None):
        if backend not in self.pins:
            raise ValueError(f"unknown backend {backend!r}; expected one of {sorted(self.pins)}")
        self.backend = backend
        self.model = self.pins[backend]
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.max_length = max_length
        self.window = window
        self.concurrency = window  # executor threads: one window of concurrent requests
        self.linger_s = linger_s
        self.usage = Usage()
        self.snapshots: set[str] = set()
        self.ledger: ResponseLedger | None = None
        self.fresh_inference_seconds = 0.0
        self._factory = server_factory or (None if device is None else (lambda: self._default_server(device)))
        self._server: Server | None = None
        self._config_sha: str | None = None
        self._cv = threading.Condition()
        self._queue: OrderedDict[str, _Pending] = OrderedDict()
        self._inflight: dict[str, Future] = {}
        self._thread: threading.Thread | None = None
        self._closed = False
        self._failure: Exception | None = None
        self._usage_lock = threading.Lock()

    def _default_server(self, device: str) -> Server:
        return ClefServer(self.backend, device, self.max_length, self.cache_dir / "scratch")

    # ---- cache ----------------------------------------------------------------------------------

    def key(self, body: dict[str, Any]) -> str:
        return cache_key(self.model, self.max_length, body)

    def _path(self, digest: str) -> Path:
        return self.cache_dir / digest[:2] / f"{digest}.json"

    def _write(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False))
        tmp.replace(path)

    def cached(self, body: dict[str, Any]) -> dict[str, Any] | None:
        path = self._path(self.key(body))
        return json.loads(path.read_text()) if path.exists() else None

    # ---- decide -------------------------------------------------------------------------------

    def decide(self, state: Any, questions: dict[str, Any], context: dict[str, Any] | None = None) -> Decision:
        body = {"model": self.model, "state": state, "questions": questions}
        digest = self.key(body)
        path = self._path(digest)
        fresh = False
        if path.exists():
            record = json.loads(path.read_text())
        else:
            if self._factory is None:
                raise self.cache_miss_error(f"{self.backend}: request {digest[:16]} is not cached and no device is configured")
            record, fresh = self._submit(digest, body)
        response = record["response"]
        missing = [q for q in questions if q not in response["answers"]]
        if missing:
            raise ClefServerError(f"response {digest[:16]} lacks answers for {missing}")
        if self.ledger is not None:
            self.ledger.record(digest, body, record, context)
        tokens = int((response.get("usage") or {}).get("input_tokens") or 0)
        seconds = float((response.get("audit") or {}).get("inference_seconds") or 0.0)
        with self._usage_lock:
            self.usage.calls += 1
            self.usage.input_tokens += tokens
            self.snapshots.add(str(response.get("model", self.model)))
            if fresh:
                self.usage.fresh_calls += 1
                self.usage.fresh_input_tokens += tokens
                self.fresh_inference_seconds += seconds
        return Decision(answers={q: response["answers"][q] for q in questions}, snapshot=str(response.get("model", self.model)),
                        input_tokens=tokens, cost=0.0, latency_s=seconds, cached=not fresh)

    def _submit(self, digest: str, body: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Queue a cache miss for the next window; (record, fresh) once answered. Identical concurrent
        requests are computed once (only the first caller counts it as fresh)."""
        future, owner = self._enqueue(digest, body)
        return future.result(), owner

    def _enqueue(self, digest: str, body: dict[str, Any], payload: dict[str, Any] | None = None) -> tuple[Future, bool]:
        """Queue without waiting: (future of the record, whether this caller queued it)."""
        with self._cv:
            if self._closed:
                raise ClefServerError("client is closed")
            future = self._inflight.get(digest)
            owner = future is None
            if owner:
                future = Future()
                self._inflight[digest] = future
                self._queue[digest] = _Pending(body, future, payload)
                if self._thread is None:
                    self._thread = threading.Thread(target=self._loop, name=f"{self.backend}-dispatch", daemon=True)
                    self._thread.start()
                self._cv.notify_all()
        return future, owner

    def _take_window(self) -> list[tuple[str, _Pending]] | None:
        with self._cv:
            while not self._queue and not self._closed:
                self._cv.wait()
            if not self._queue:
                return None
            while len(self._queue) < self.window:  # gather until the window is full or arrivals stop
                n = len(self._queue)
                self._cv.wait(timeout=self.linger_s)
                if len(self._queue) == n:
                    break
            items = []
            while self._queue and len(items) < self.window:
                items.append(self._queue.popitem(last=False))
            return items

    def _ensure_server(self) -> Server:
        """Start the server on the first miss and check its pin and settings; a failed start is final."""
        if self._failure is not None:
            raise ClefServerError(f"inference server unavailable: {self._failure}")
        if self._server is None:
            assert self._factory is not None
            try:
                server = self._factory()
            except Exception as exc:
                self._failure = exc
                raise
            conf = server.configuration
            drift = {k: (conf.get(k), v) for k, v in self.encoding.items() if conf.get(k) != v}
            if conf.get("max_length") != self.max_length:
                drift["max_length"] = (conf.get("max_length"), self.max_length)
            if server.model != self.model:
                drift["model"] = (server.model, self.model)
            if drift:
                server.close()
                self._failure = ClefServerError(f"server pin/configuration differs from the cache key settings: {drift}")
                raise self._failure
            blob = json.dumps(conf, sort_keys=True, ensure_ascii=False)
            self._config_sha = hashlib.sha256(blob.encode("utf-8")).hexdigest()
            path = self.cache_dir / "configurations" / f"{self._config_sha}.json"
            if not path.exists():
                self._write(path, conf)
            self._server = server
        return self._server

    def _loop(self) -> None:
        while True:
            items = self._take_window()
            if items is None:
                return
            try:
                server = self._ensure_server()
                results = server.compute([(digest, p.body if p.payload is None else p.payload) for digest, p in items])
                by_key = {r["key"]: r for r in results}
                for digest, pending in items:
                    r = by_key.get(digest)
                    if r is None:
                        raise ClefServerError(f"server returned no result for {digest[:16]}")
                    if "error" in r:
                        exc: BaseException = self.rejected_error(f"{r.get('type', 'error')}: {r['error']}")
                        pending.future.set_exception(exc)
                        continue
                    record = {"request_hash": digest, "request": pending.body, "response": r["response"],
                              "latency_s": (r["response"].get("audit") or {}).get("inference_seconds"),
                              "configuration": self._config_sha,
                              "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                    self._write(self._path(digest), record)
                    pending.future.set_result(record)
            except BaseException as exc:  # noqa: BLE001 - delivered to every waiting caller of the window
                for _, pending in items:
                    if not pending.future.done():
                        pending.future.set_exception(exc if isinstance(exc, Exception) else ClefServerError(repr(exc)))
                if not isinstance(exc, Exception):
                    raise
            finally:
                with self._cv:
                    for digest, _ in items:
                        self._inflight.pop(digest, None)

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join()
        if self._server is not None:
            self._server.close()
            self._server = None


def make_clef_client(backend: str) -> ClefBackend:
    """Client from the environment: `JEV_OPT_DEVICE` (absent: cache-only), `JEV_OPT_CLEF_CACHE`
    (default `data/cache/<backend>-jev-opt` in the repository), `JEV_OPT_CLEF_MAX_LENGTH` (default 32000)."""
    cache = Path(os.environ.get("JEV_OPT_CLEF_CACHE") or REPO / "data" / "cache" / f"{backend}-jev-opt")
    return ClefBackend(backend, cache, device=os.environ.get("JEV_OPT_DEVICE") or None,
                       max_length=int(os.environ.get("JEV_OPT_CLEF_MAX_LENGTH") or 32000))
