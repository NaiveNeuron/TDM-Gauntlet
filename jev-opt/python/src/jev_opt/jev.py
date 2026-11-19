"""Jev Decisions API client (architecture §9): disk cache, pacing, retries, usage.

Mirrors the gauntlet client (src/gauntlet/jev.py): SHA-256 content-addressed cache,
<= max_rpm pacing, Retry-After / OpenRouter X-RateLimit-Reset handling for 429 and
retries on Cloudflare 52x. Synchronous httpx + threads: GEPA calls the adapter
synchronously, and batching is concurrent single-state requests.
The cache has two tiers: exact requests, and per-(state, question) pointers into them, so a
question already answered for a state is never re-bought when questions are re-packed.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
RETRY_STATUS = {408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529}
LEDGER = "jev_responses"
SHARD_RECORDS = 5000
MAX_QUESTIONS = 64  # questions per request when unanswered questions of a large batch are sent in chunks
# input_tokens ~ 308 + 0.26 * state chars + 0.22 * question chars (JSON), fitted on 13,319 recorded
# jev-1.13 requests; used only for requests assembled from per-question cache entries.
EST_BASE, EST_STATE, EST_QUESTION = 308.0, 0.26, 0.22
DEFAULT_PRICE = 4.2e-8  # $ per input token (jev-1.13, 2026-09); updated from every fresh response
SOURCE_LRU = 1024  # recorded requests kept in memory for per-question lookups


def decisions_url() -> str:
    return os.environ.get("JEV_OPT_DECISIONS_URL", DECISIONS_URL)


def load_api_key(search_from: list[Path], ignore: str | None = None) -> str:
    """OPENROUTER_API_KEY from env or the nearest `.env` above any of `search_from`. Never logged.
    An environment value equal to `ignore` (a placeholder) does not count."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key and key != ignore:
        return key
    for start in search_from:
        for folder in [start.resolve(), *start.resolve().parents]:
            env = folder / ".env"
            if env.is_file():
                for line in env.read_text().splitlines():
                    name, _, value = line.partition("=")
                    if name.strip().removeprefix("export ").strip() == "OPENROUTER_API_KEY" and value.strip():
                        return value.strip().strip("\"'")
    raise RuntimeError("OPENROUTER_API_KEY not set in environment or any parent .env")


def cache_key(url: str, model: str, state: Any, questions: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"url": url, "model": model, "state": state, "questions": questions},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def question_key(url: str, model: str, state: Any, question: dict[str, Any]) -> str:
    """Cache key of one question's answer: (endpoint, model, state, question content). The question
    id is not part of it: Jev answers every question of a request independently of the others."""
    canonical = json.dumps({"url": url, "model": model, "state": state, "question": question}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def estimate_tokens(state: Any, questions: dict[str, Any]) -> int:
    """Input tokens of a request that was never sent as such (assembled from per-question cache
    entries): linear model fitted on 13,319 recorded jev-1.13 requests, state billed once."""
    state_chars = len(json.dumps(state, ensure_ascii=False))
    question_chars = sum(len(json.dumps(q, ensure_ascii=False)) for q in questions.values())
    return int(round(EST_BASE + EST_STATE * state_chars + EST_QUESTION * question_chars))


class JevError(RuntimeError):
    def __init__(self, status: int, payload: str):
        super().__init__(f"Jev request failed with HTTP {status}: {payload[:500]}")
        self.status = status
        self.payload = payload


@dataclass(frozen=True)
class Decision:
    answers: dict[str, Any]
    snapshot: str  # served dated model snapshot
    input_tokens: int
    cost: float
    latency_s: float
    cached: bool
    estimated: bool = False  # usage estimated (answers assembled from several recorded requests)


@dataclass
class Usage:
    calls: int = 0
    fresh_calls: int = 0
    input_tokens: int = 0
    fresh_cost: float = 0.0
    fresh_input_tokens: int = 0


class JevClient:
    def __init__(
        self,
        cache_dir: Path,
        api_key: str | None,
        model: str = "typesafe/jev-1.13",
        max_rpm: int = 1800,
        concurrency: int = 16,
        max_retries: int = 20,
        timeout_s: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.url = decisions_url()
        self.model = model
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self._api_key = api_key
        self._sleep = sleep
        self._interval = 60.0 / max_rpm
        self._next_slot = 0.0
        self._slot_lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}
        self._key_locks_guard = threading.Lock()
        self._usage_lock = threading.Lock()
        self.usage = Usage()
        self.snapshots: set[str] = set()
        self.ledger: ResponseLedger | None = None  # set per run: every response is also recorded there
        self.max_questions = MAX_QUESTIONS
        self.price = DEFAULT_PRICE
        self._sources: OrderedDict[str, tuple[dict[str, Any], dict[str, Any]]] = OrderedDict()
        self._sources_lock = threading.Lock()
        self._http = httpx.Client(transport=transport, timeout=timeout_s, limits=httpx.Limits(max_connections=concurrency))

    def close(self) -> None:
        self._http.close()

    def _path(self, digest: str) -> Path:
        return self.cache_dir / digest[:2] / f"{digest}.json"

    def _qpath(self, qkey: str) -> Path:
        return self.cache_dir / "q" / qkey[:2] / f"{qkey}.json"

    def _body_path(self, digest: str) -> Path:
        return self.cache_dir / "req" / digest[:2] / f"{digest}.json"

    @staticmethod
    def _write(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False))
        tmp.replace(path)

    def _lock(self, digest: str) -> threading.Lock:
        with self._key_locks_guard:
            return self._key_locks.setdefault(digest, threading.Lock())

    def _index(self, digest: str, body: dict[str, Any]) -> None:
        """Per-question pointers to a recorded request (first answer wins) + its request body."""
        if not self._body_path(digest).exists():
            self._write(self._body_path(digest), body)
        for qid, question in body["questions"].items():
            path = self._qpath(question_key(self.url, self.model, body["state"], question))
            if not path.exists():
                self._write(path, {"request_hash": digest, "qid": qid})

    def _fetch(self, state: Any, questions: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any], bool]:
        """(request hash, body, record, fresh) of exactly this request: the request-level cache, else
        one API call. Concurrent identical requests are fetched once."""
        digest = cache_key(self.url, self.model, state, questions)
        body = {"model": self.model, "state": state, "questions": questions}
        with self._lock(digest):
            path = self._path(digest)
            fresh = not path.exists()
            if fresh:
                t0 = time.monotonic()
                response = self._post(body)
                record = {"request_hash": digest, "latency_s": round(time.monotonic() - t0, 4), "response": response}
                self._write(path, record)
            else:
                record = json.loads(path.read_text())
        self._index(digest, body)
        return digest, body, record, fresh

    def _lookup(self, state: Any, questions: dict[str, Any]) -> tuple[dict[str, Any], dict[str, tuple[dict[str, Any], dict[str, Any]]]]:
        """Answers already recorded for (state, question) in any earlier request, and those requests."""
        answers: dict[str, Any] = {}
        sources: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for qid, question in questions.items():
            path = self._qpath(question_key(self.url, self.model, state, question))
            if not path.exists():
                continue
            pointer = json.loads(path.read_text())
            source = pointer["request_hash"]
            if source not in sources:
                sources[source] = self._source(source)
            answer = sources[source][1]["response"]["answers"].get(pointer["qid"])
            if answer is not None:
                answers[qid] = answer
        return answers, sources

    def _source(self, digest: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """(request body, record) of a recorded request; small LRU (packed records are re-read often)."""
        with self._sources_lock:
            hit = self._sources.get(digest)
            if hit is not None:
                self._sources.move_to_end(digest)
                return hit
        value = (json.loads(self._body_path(digest).read_text()), json.loads(self._path(digest).read_text()))
        with self._sources_lock:
            self._sources[digest] = value
            while len(self._sources) > SOURCE_LRU:
                self._sources.popitem(last=False)
        return value

    def decide(self, state: Any, questions: dict[str, Any], context: dict[str, Any] | None = None) -> Decision:
        """Cached-or-fresh decision (§9). Two cache tiers: the exact request, else each question's
        recorded answer for this state from any earlier request; only unanswered questions are
        sent (in chunks of `max_questions`). Usage is exact for a request recorded as such and
        estimated (`estimate_tokens`) for one assembled from several recorded requests."""
        digest = cache_key(self.url, self.model, state, questions)
        fresh: list[dict[str, Any]] = []
        if self._path(digest).exists() or len(questions) <= self.max_questions and not self._any_indexed(state, questions):
            _, body, record, was_fresh = self._fetch(state, questions)
            answers, sources = dict(record["response"]["answers"]), {digest: (body, record)}
            fresh += [record] if was_fresh else []
        else:
            answers, sources = self._lookup(state, questions)
            missing = [q for q in questions if q not in answers]
            for start in range(0, len(missing), self.max_questions):
                chunk = {q: questions[q] for q in missing[start : start + self.max_questions]}
                h, body, record, was_fresh = self._fetch(state, chunk)
                answers.update({q: record["response"]["answers"][q] for q in chunk})
                sources[h] = (body, record)
                fresh += [record] if was_fresh else []
        if self.ledger is not None:
            for h, (body, record) in sources.items():
                self.ledger.record(h, body, record, context)
        exact = list(sources) == [digest]
        responses = [r["response"] for _, r in sources.values()]
        if exact:
            usage = responses[0].get("usage") or {}
            tokens, cost = int(usage.get("input_tokens") or 0), float(usage.get("cost") or 0.0)
        else:
            tokens = estimate_tokens(state, questions)
            cost = tokens * self.price
        decision = Decision(
            answers={q: answers[q] for q in questions},
            snapshot=str(responses[0].get("model", self.model)) if responses else self.model,
            input_tokens=tokens,
            cost=cost,
            latency_s=max((float(r.get("latency_s") or 0.0) for _, r in sources.values()), default=0.0),
            cached=not fresh,
            estimated=not exact,
        )
        with self._usage_lock:
            self.usage.calls += 1
            self.usage.input_tokens += decision.input_tokens
            self.snapshots.update(str(r.get("model", self.model)) for r in responses)
            for record in fresh:
                usage = record["response"].get("usage") or {}
                self.usage.fresh_calls += 1
                self.usage.fresh_cost += float(usage.get("cost") or 0.0)
                self.usage.fresh_input_tokens += int(usage.get("input_tokens") or 0)
                if usage.get("input_tokens"):
                    self.price = float(usage.get("cost") or 0.0) / int(usage["input_tokens"])
        return decision

    def _any_indexed(self, state: Any, questions: dict[str, Any]) -> bool:
        return any(self._qpath(question_key(self.url, self.model, state, q)).exists() for q in questions.values())

    def _pace(self) -> None:
        with self._slot_lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self._interval
        if wait > 0:
            self._sleep(wait)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        if not self._api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for uncached Jev requests")
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        for attempt in range(self.max_retries + 1):
            self._pace()
            try:
                resp = self._http.post(self.url, json=body, headers=headers)
            except (httpx.TransportError, httpx.TimeoutException):
                if attempt == self.max_retries:
                    raise
                self._sleep(_backoff(attempt))
                continue
            if resp.status_code == 200:
                payload = resp.json()
                if "answers" in payload:
                    return payload
                if attempt == self.max_retries:  # 200 with an error envelope: retryable provider error
                    raise JevError(200, resp.text)
            elif resp.status_code not in RETRY_STATUS or attempt == self.max_retries:
                raise JevError(resp.status_code, resp.text)
            self._sleep(_retry_delay(resp, attempt))
        raise AssertionError("unreachable")


def _retry_delay(resp: httpx.Response, attempt: int) -> float:
    """Prefer server hints: Retry-After header, then OpenRouter's X-RateLimit-Reset (epoch ms)."""
    retry_after = resp.headers.get("retry-after")
    if retry_after and retry_after.replace(".", "", 1).isdigit():
        return float(retry_after)
    try:
        reset_ms = float(resp.json()["error"]["metadata"]["headers"]["X-RateLimit-Reset"])
    except (ValueError, KeyError, TypeError):
        return _backoff(attempt)
    wait = reset_ms / 1000.0 - time.time()
    return min(max(wait, 0.0), 65.0) + random.random()


def _backoff(attempt: int) -> float:
    return min(60.0, 2.0**attempt) * (0.5 + random.random() / 2)


class ResponseLedger:
    """Verbatim decision-model responses a run obtained (fresh and cache hits), one record per request hash.

    Records are appended to `<name>.pending.jsonl` as they arrive (crash-safe); `finalize` merges them
    with existing shards into gzip shards of `shard_records` rows sorted by request hash, written
    byte-deterministically (gzip mtime 0). `name`: `jev_responses` (Jev), `clef_responses` (Clef backends).
    """

    def __init__(self, run_dir: Path, shard_records: int = SHARD_RECORDS, name: str = LEDGER):
        self.run_dir = Path(run_dir)
        self.shard_records = shard_records
        self.name = name
        self.pending = self.run_dir / f"{name}.pending.jsonl"
        self._lock = threading.Lock()
        self._seen = {row["request_hash"] for row in self._rows()}

    def shard_name(self, index: int) -> str:
        return f"{self.name}.jsonl.gz" if index == 0 else f"{self.name}.{index:05d}.jsonl.gz"

    def shards(self) -> list[Path]:
        out: list[Path] = []
        while (path := self.run_dir / self.shard_name(len(out))).exists():
            out.append(path)
        return out

    def _rows(self) -> Iterator[dict[str, Any]]:
        for path in self.shards():
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                yield from (json.loads(line) for line in fh if line.strip())
        if self.pending.exists():
            for line in self.pending.read_text(encoding="utf-8").splitlines():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:  # torn last line after a crash
                    continue

    def record(self, digest: str, request: dict[str, Any], cached: dict[str, Any], context: dict[str, Any] | None) -> None:
        with self._lock:
            if digest in self._seen:
                return
            self._seen.add(digest)
            row = {"request_hash": digest, **{k: v for k, v in (context or {}).items() if v is not None}}
            row.update(request=request, response=cached["response"], latency_s=cached.get("latency_s"))
            with open(self.pending, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def finalize(self) -> dict[str, Any]:
        """Rewrite all records as sorted shards; returns `{shards, records}` for run.json."""
        with self._lock:
            rows: dict[str, dict[str, Any]] = {}
            for row in self._rows():
                rows.setdefault(row["request_hash"], row)
            ordered = [rows[k] for k in sorted(rows)]
            old = self.shards()
            names = []
            for index, start in enumerate(range(0, len(ordered), self.shard_records)):
                names.append(self.shard_name(index))
                data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered[start : start + self.shard_records])
                tmp = self.run_dir / f"{names[-1]}.tmp"
                with open(tmp, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                    gz.write(data.encode("utf-8"))
                tmp.replace(self.run_dir / names[-1])
            for path in old[len(names) :]:
                path.unlink()
            self.pending.unlink(missing_ok=True)
            return {"shards": names, "records": len(ordered)}


def make_client(cache_dir: Path, model: str, max_rpm: int, concurrency: int, search_from: list[Path]) -> JevClient:
    """Production client; `JEV_OPT_MOCK_JEV=1` swaps in the offline keyword-overlap simulator,
    `JEV_OPT_BACKEND=clef|clef-flash` the local Clef backend (`clef.make_clef_client`),
    `JEV_OPT_BACKEND=qwen-base` Clef's base model through option log-probabilities
    (`logprob.make_logprob_client`) and `JEV_OPT_BACKEND=luna|msd1` GPT-6 Luna or Microsoft-Decision-1 through the
    Decisions endpoint (`luna.make_luna_client`: exact-request cache `<cache_dir>/<backend>`, the model's spend
    ledger). For these backends `model` and the rate settings do not apply; for the local ones neither does `cache_dir`."""
    backend = os.environ.get("JEV_OPT_BACKEND", "jev")
    if backend in ("luna", "msd1"):
        from .luna import SPECS, make_luna_client

        return make_luna_client(cache_dir, search_from, spec=SPECS[backend])
    if backend != "jev":
        from .logprob import PINS as LOGPROB_PINS
        from .logprob import make_logprob_client

        if backend in LOGPROB_PINS:
            return make_logprob_client(backend)  # type: ignore[return-value]  # same interface as JevClient
        from .clef import make_clef_client

        return make_clef_client(backend)  # type: ignore[return-value]  # same interface as JevClient
    if os.environ.get("JEV_OPT_MOCK_JEV") == "1":
        from .mockjev import transport

        return JevClient(cache_dir / "mock", "mock-key", model, max_rpm=10**9, concurrency=concurrency, transport=transport())
    try:
        key = load_api_key(search_from)
    except RuntimeError:
        key = None  # cache-only mode; a cache miss raises a clear error
    return JevClient(cache_dir, key, model, max_rpm=max_rpm, concurrency=concurrency)
