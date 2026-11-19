"""Frozen, local Clef inference using the pinned official release implementation.

Heavy imports and model loading are lazy. Public answers retain the release's
four-decimal SystemOne conversion; unrounded FP32-softmax values are also saved
in the response audit. Inputs exceeding the audit context are rejected, never
silently truncated. Native forwards are serialized within each device process.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import multiprocessing
import multiprocessing.connection
import platform
import sys
import threading
import time
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from .task import REPO

MODEL_ID = "Cloudflare/clef"
REVISION = "2f3de3dd85f379784083b0814d997ab627200f0c"
MODEL = f"{MODEL_ID}@{REVISION}"
FLASH_MODEL_ID = "Cloudflare/clef-flash"
FLASH_REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"
FLASH_MODEL = f"{FLASH_MODEL_ID}@{FLASH_REVISION}"
MODELS = {"clef": MODEL, "clef-flash": FLASH_MODEL}
HF_CACHE = REPO / "data" / "cache" / "huggingface"
_LOG = logging.getLogger(__name__)
# A process keeps the same frozen model across sequential task clients. This
# lock also prevents concurrent forwards from multiplying GPU memory use.
_RUNTIME_LOCK = threading.RLock()
_RUNTIMES: dict[str, tuple[Any, Any, Any]] = {}


class ClefContextError(ValueError):
    """The complete state and schema do not fit the comparison context."""


class _TokenCounter:
    """Observe the official encoder's final (state) tokenizer call."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.state_tokens = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = self.tokenizer(*args, **kwargs)
        self.state_tokens = len(result.input_ids)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _release_module(path: Path, revision: str) -> Any:
    # Dataclasses require their module to be registered, but no global sys.path
    # change or unpinned joint_schema_model import is necessary.
    name = f"gauntlet._clef_release_{revision}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path / "joint_schema_model.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the pinned Clef release from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


@dataclass
class ClefClient:
    cache_dir: Path
    device: str = "cuda:0"
    max_length: int = 32000
    model: str = MODEL
    local = True
    batch_size = 16
    batch_padded_token_limit = 16384

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.model not in MODELS.values():
            raise ValueError("Frozen audits require a pinned Clef or Clef-flash release")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_id(self) -> str:
        return self.model.split("@", 1)[0]

    @property
    def revision(self) -> str:
        return self.model.split("@", 1)[1]

    @cached_property
    def _configuration(self) -> dict[str, Any]:
        import torch

        device = torch.device(self.device)
        gpu = None
        if device.type == "cuda":
            index = device.index if device.index is not None else torch.cuda.current_device()
            gpu = {
                "name": torch.cuda.get_device_name(index),
                "compute_capability": list(torch.cuda.get_device_capability(index)),
            }
        return {
            "backend": "clef-local-frozen",
            "client_encoding_version": 1,
            "model_id": self.model_id,
            "revision": self.revision,
            "dtype": "bfloat16",
            "probability_dtype": "float32",
            "max_length": self.max_length,
            "max_state_tokens": None,
            "truncation_policy": "reject_complete_input_overflow_before_forward",
            "encoding": "official encode_record; complete state and schema",
            "attention_implementation": "sdpa",
            "batch_size": self.batch_size,
            "batch_padded_token_limit": self.batch_padded_token_limit,
            "batching": "length-bucketed within bounded request windows",
            "device": str(device),
            "gpu": gpu,
            "python": platform.python_version(),
            "libraries": {
                name: _version(name)
                for name in (
                    "torch", "transformers", "huggingface-hub", "safetensors",
                    "tokenizers", "accelerate", "pillow", "torchvision",
                    "flash-linear-attention", "fla-core", "triton", "causal-conv1d", "flash-attn",
                )
            },
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "numerical_configuration": {
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "sdpa_flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
                "sdpa_math_enabled": torch.backends.cuda.math_sdp_enabled(),
                "sdpa_mem_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "sdpa_cudnn_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
            },
            "rounding_policy": "official systemone_answer: round(value, 4); raw probabilities retained",
            "monetary_cost": "not measured",
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._configuration))

    def _digest(self, body: dict[str, Any]) -> str:
        # Preserve mapping order: the official encoder iterates schema fields in
        # insertion order, so sorted-key JSON alone is not a safe encoding key.
        canonical = json.dumps(
            {"configuration": self._configuration, "request": body},
            ensure_ascii=False, separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _cache_path(self, digest: str) -> Path:
        return self.cache_dir / digest[:2] / f"{digest}.json"

    def cached(self, body: dict[str, Any]) -> dict[str, Any] | None:
        path = self._cache_path(self._digest(body))
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        record["cache_hit"] = True
        return record

    def _runtime(self) -> tuple[Any, Any, Any]:
        import torch
        from huggingface_hub import snapshot_download

        # Context is an encoder setting, not a model-load setting. Different
        # context clients still reuse the same frozen weights on the device.
        configuration = dict(self._configuration)
        for setting in ("max_length", "batch_size", "batch_padded_token_limit", "batching"):
            del configuration[setting]
        key = json.dumps(configuration, sort_keys=True)
        if key not in _RUNTIMES:
            HF_CACHE.mkdir(parents=True, exist_ok=True)
            path = Path(snapshot_download(self.model_id, revision=self.revision, cache_dir=str(HF_CACHE)))
            release = _release_module(path, self.revision)
            model, processor = release.load_release_model(
                path, device=self.device, dtype=torch.bfloat16,
                attn_implementation="sdpa", local_files_only=True, cache_dir=str(HF_CACHE),
            )
            _RUNTIMES[key] = (model, processor, release)
        return _RUNTIMES[key]

    def _encode(self, body: dict[str, Any], processor: Any, release: Any) -> tuple[Any, int]:
        questions = body.get("questions")
        if not isinstance(body.get("model"), str) or "state" not in body:
            raise ValueError("model and state are required")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("at least one question is required")
        for question_id, question in questions.items():
            if question.get("type") not in release.QUESTION_TYPES:
                raise ValueError(f"{question_id}: type must be noul, choice, or score")
            if question["type"] != "noul" and not question.get("criteria"):
                raise ValueError(f"{question_id}: criteria must not be empty")
        counter = _TokenCounter(processor.tokenizer)
        # The pinned encoder otherwise silently clips state tokens. Encode the
        # complete input once, then enforce the comparison bound before tensors
        # are allocated on the GPU. sys.maxsize is not a model context setting.
        encoded = release.encode_record(
            counter, body, max_length=sys.maxsize, processor=processor,
        )
        input_tokens = len(encoded.input_ids)
        overflow = max(0, input_tokens - self.max_length)
        if overflow:
            state_tokens_kept = max(0, counter.state_tokens - overflow)
            _LOG.error(
                "Clef rejected context overflow: input_tokens=%d max_length=%d "
                "state_tokens=%d encoded_state_tokens_at_limit=%d",
                input_tokens, self.max_length, counter.state_tokens, state_tokens_kept,
            )
            raise ClefContextError(
                f"Complete Clef input requires {input_tokens} tokens (limit {self.max_length}); "
                f"state has {counter.state_tokens} tokens, but at most {state_tokens_kept} "
                "would fit. Truncation is forbidden for frozen comparisons."
            )
        return encoded, counter.state_tokens

    def _forward(
        self, records: list[tuple[dict[str, Any], Any, int]],
        model: Any, processor: Any, release: Any,
    ) -> list[dict[str, Any]]:
        import torch

        device = next(model.parameters()).device
        batch = release.collate_records(
            [record[1] for record in records], processor.tokenizer.pad_token_id, device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            logits = model(batch)
            all_probabilities = [
                {
                    question.question_id: dict(zip(
                        question.option_ids, question_logits.float().softmax(-1).tolist(),
                    ))
                    for question, question_logits in zip(encoded.questions, logits[index])
                }
                for index, (_, encoded, _) in enumerate(records)
            ]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - start
        responses = []
        for (body, encoded, state_tokens), probabilities in zip(records, all_probabilities):
            answers = {
                question.question_id: release.systemone_answer(
                    body["questions"][question.question_id], probabilities[question.question_id],
                )
                for question in encoded.questions
            }
            responses.append({
                "model": self.model,
                "answers": answers,
                "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0},
                "audit": {
                    "inference_seconds": inference_seconds / len(records),
                    "batch_inference_seconds": inference_seconds,
                    "batch_size": len(records),
                    "batch_padded_tokens": len(records) * batch["input_ids"].shape[1],
                    "truncated": False,
                    "state_tokens": state_tokens,
                    "encoded_state_tokens": state_tokens,
                    "input_tokens": len(encoded.input_ids),
                    "raw_probabilities": probabilities,
                    "rounding_policy": self._configuration["rounding_policy"],
                },
            })
        return responses

    def compute_batches(self, bodies: dict[str, dict[str, Any]]) -> Iterator[list[tuple[str, dict[str, Any]]]]:
        """Encode complete inputs, then forward length-bucketed batches; yield (key, response) per batch.

        No cache: callers own storage (the gauntlet's request cache, jev-opt's Clef backend)."""
        model, processor, release = self._runtime()
        pending = [
            (key, body, *self._encode(body, processor, release))
            for key, body in bodies.items()
        ]
        pending.sort(key=lambda record: len(record[2].input_ids))
        first = 0
        while first < len(pending):
            end = first + 1
            while (
                end < len(pending)
                and end - first < self.batch_size
                and len(pending[end][2].input_ids) * (end - first + 1)
                <= self.batch_padded_token_limit
            ):
                end += 1
            group = pending[first:end]
            responses = self._forward(
                [(body, encoded, tokens) for _, body, encoded, tokens in group],
                model, processor, release,
            )
            yield [(key, response) for (key, _, _, _), response in zip(group, responses)]
            first = end

    async def decide(self, http: Any, body: dict[str, Any]) -> dict[str, Any]:
        """Return a Jev-compatible record; the HTTP client is intentionally unused."""
        return (await self.decide_many(http, [body]))[0]

    async def decide_many(self, http: Any, bodies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Reuse cached/duplicate requests and batch complete inputs without clipping."""
        digests = [self._digest(body) for body in bodies]
        records: dict[str, dict[str, Any]] = {}
        missing = {}
        with _RUNTIME_LOCK:
            for digest, body in zip(digests, bodies):
                if digest in records or digest in missing:
                    continue
                path = self._cache_path(digest)
                if path.exists():
                    record = json.loads(path.read_text())
                    record["cache_hit"] = True
                    records[digest] = record
                else:
                    missing[digest] = body
            if missing:
                for group in self.compute_batches(missing):
                    for digest, response in group:
                        record = {"request_hash": digest, "response": response, "cache_hit": False}
                        records[digest] = record
                        path = self._cache_path(digest)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        tmp = path.with_suffix(".tmp")
                        tmp.write_text(json.dumps(record, ensure_ascii=False))
                        tmp.replace(path)
            seen = set()
            ordered = []
            for digest in digests:
                record = records[digest]
                ordered.append(dict(record, cache_hit=True) if digest in seen else record)
                seen.add(digest)
            return ordered


class ClefWorkerError(RuntimeError):
    """A spawned native Clef worker failed or exited unexpectedly."""


def _clef_worker(
    connection: Any, cache_dir: Path, device: str, max_length: int, model: str,
    client_factory: Any,
) -> None:
    """Keep the native runtime in its own process across request rounds."""
    window = None
    label = "setup"
    try:
        client = client_factory(cache_dir, device=device, max_length=max_length, model=model)
        connection.send(("ready", client.metadata))
        with asyncio.Runner() as runner:
            while True:
                try:
                    window, label, bodies = connection.recv()
                except EOFError:
                    return
                records = runner.run(client.decide_many(None, bodies))
                connection.send(("result", window, records))
    except BaseException as error:
        try:
            connection.send(("error", {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "context_error": isinstance(error, ClefContextError),
                "window": window,
                "label": label,
            }))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


@dataclass
class _ClefWorker:
    device: str
    process: Any
    connection: Any
    cache_client: ClefClient | None = None


class ClefMultiGpuClient:
    """Schedule original 64-request windows over reusable spawned native clients.

    Cache identities belong to the native device configurations, never to this
    aggregate client. Only cache misses run on GPUs; one window is outstanding
    per device, and whichever device finishes first claims the next window.
    Call close() when the audit ends, including on cancellation. Startup and
    inference failures close all workers and make the client unusable.
    """

    local = True

    def __init__(
        self, cache_dir: Path, devices: list[str] | tuple[str, ...],
        max_length: int = 32000, model: str = MODEL,
    ) -> None:
        if len(devices) < 2 or len(set(devices)) != len(devices):
            raise ValueError("Multi-GPU Clef requires at least two distinct devices")
        # Reuse native validation without evaluating device configuration or
        # importing torch in the scheduling process.
        ClefClient(cache_dir, device=devices[0], max_length=max_length, model=model)
        self.cache_dir = Path(cache_dir)
        self.devices = tuple(devices)
        self.max_length = max_length
        self.model = model
        self._workers: list[_ClefWorker] = []
        self._metadata: dict[str, Any] | None = None
        self._closed = False
        self._operation_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._close_lock = threading.Lock()
        # Tests can supply a picklable native-client subclass before startup.
        # Production always uses the unchanged native encoder/forward/cache.
        self._client_factory = ClefClient

    def _ensure_open(self) -> None:
        with self._state_lock:
            if self._closed:
                raise ClefWorkerError("Multi-GPU Clef client is closed")

    def _receive(self) -> tuple[_ClefWorker, tuple[Any, ...]]:
        """Wait for messages AND process death, including from idle workers."""
        self._ensure_open()
        try:
            ready = multiprocessing.connection.wait([
                item
                for worker in self._workers
                for item in (worker.connection, worker.process.sentinel)
            ])
            # A failed device must not be starved by a fast device's results.
            for worker in sorted(self._workers, key=lambda worker: worker.process.sentinel not in ready):
                exited = worker.process.sentinel in ready
                if exited:
                    worker.process.join()
                if worker.connection in ready:
                    try:
                        message = worker.connection.recv()
                    except (EOFError, OSError) as error:
                        if exited:
                            raise ClefWorkerError(
                                f"Clef worker {worker.device} exited unexpectedly "
                                f"(exit code {worker.process.exitcode})"
                            ) from error
                        raise ClefWorkerError(
                            f"Clef worker {worker.device} closed its pipe unexpectedly"
                        ) from error
                    if message[0] == "error":
                        detail = message[1]
                        message_text = (
                            f"Clef worker {worker.device}, {detail['label']}, "
                            f"window {detail['window']}: {detail['type']}: "
                            f"{detail['message']}\n{detail['traceback']}"
                        )
                        if detail["context_error"]:
                            raise ClefContextError(message_text)
                        raise ClefWorkerError(message_text)
                    if not exited:
                        return worker, message
                if exited:
                    raise ClefWorkerError(
                        f"Clef worker {worker.device} exited unexpectedly "
                        f"(exit code {worker.process.exitcode})"
                    )
        except ClefContextError:
            raise
        except (OSError, ValueError) as error:
            # close() can interrupt this wait from the event-loop thread.
            self._ensure_open()
            raise ClefWorkerError("Clef worker communication failed") from error
        raise ClefWorkerError("Clef worker wait returned no message")

    def _start(self) -> None:
        self._ensure_open()
        if self._metadata is not None:
            for worker in self._workers:
                if not worker.process.is_alive():
                    raise ClefWorkerError(f"Clef worker {worker.device} is no longer running")
            return
        context = multiprocessing.get_context("spawn")
        for device in self.devices:
            parent, child = context.Pipe()
            process = context.Process(
                target=_clef_worker,
                args=(child, self.cache_dir, device, self.max_length, self.model, self._client_factory),
                name=f"clef-{device}",
            )
            try:
                with self._state_lock:
                    if self._closed:
                        raise ClefWorkerError("Multi-GPU Clef client is closed")
                    self._workers.append(_ClefWorker(device, process, parent))
                    process.start()
            finally:
                child.close()
                if process.pid is None:
                    parent.close()
        configurations = {}
        while len(configurations) < len(self._workers):
            worker, message = self._receive()
            if message[0] != "ready" or worker.device in configurations:
                raise ClefWorkerError(f"Unexpected Clef startup message from {worker.device}")
            configurations[worker.device] = message[1]
        common = None
        metadata_devices = []
        canonical_devices = set()
        for worker in self._workers:
            configuration = configurations[worker.device]
            canonical_device = configuration["device"]
            if canonical_device in canonical_devices:
                raise ClefWorkerError(f"Clef devices resolve to the same device: {canonical_device}")
            canonical_devices.add(canonical_device)
            shared = {key: value for key, value in configuration.items() if key not in ("device", "gpu")}
            if common is not None and common != shared:
                raise ClefWorkerError("Clef devices disagree on frozen non-device configuration")
            common = shared
            metadata_devices.append(configuration)
            # Use the original digest and cache reader with the worker's exact
            # configuration. This does not import torch or rewrite saved files.
            worker.cache_client = ClefClient(
                self.cache_dir, device=worker.device, max_length=self.max_length, model=self.model,
            )
            worker.cache_client.__dict__["_configuration"] = configuration
            worker.device = canonical_device
        with self._state_lock:
            if self._closed:
                raise ClefWorkerError("Multi-GPU Clef client is closed")
            self._metadata = dict(common, devices=metadata_devices)

    @property
    def metadata(self) -> dict[str, Any]:
        with self._operation_lock:
            try:
                self._start()
                return json.loads(json.dumps(self._metadata))
            except BaseException:
                self.close()
                raise

    async def decide_windows(
        self, bodies: list[dict[str, Any]], label: str,
    ) -> list[dict[str, Any]]:
        """Return ordered native records with actual execution-device provenance."""
        try:
            return await asyncio.to_thread(self._decide_windows, bodies, label)
        except asyncio.CancelledError:
            self.close()
            raise

    def _decide_windows(
        self, bodies: list[dict[str, Any]], label: str,
    ) -> list[dict[str, Any]]:
        with self._operation_lock:
            try:
                self._ensure_open()
                if not bodies:
                    return []
                self._start()
                # The first native device digest is only a deduplication key.
                # The native digest preserves all request mapping order.
                first_client = self._workers[0].cache_client
                keys = [first_client._digest(body) for body in bodies]
                unique = {}
                for index, (key, body) in enumerate(zip(keys, bodies)):
                    if key not in unique:
                        unique[key] = (index, body)
                records = {}
                windows = {}
                # Finish lookup on every device before any missing body runs.
                for key, (index, body) in unique.items():
                    for worker in self._workers:
                        record = worker.cache_client.cached(body)
                        if record is not None:
                            records[key] = dict(record, execution_device=worker.device)
                            break
                    else:
                        windows.setdefault(index // 64, []).append((key, body))
                remaining = iter(windows.items())
                pending = {}

                def dispatch(worker: _ClefWorker) -> None:
                    item = next(remaining, None)
                    if item is None:
                        return
                    window, requests = item
                    pending[worker.device] = (window, requests)
                    worker.connection.send((window, label, [body for _, body in requests]))

                for worker in self._workers:
                    dispatch(worker)
                while pending:
                    worker, message = self._receive()
                    expected = pending.pop(worker.device, None)
                    if (
                        message[0] != "result" or expected is None
                        or message[1] != expected[0] or len(message[2]) != len(expected[1])
                    ):
                        raise ClefWorkerError(f"Unexpected Clef result from {worker.device}")
                    for (key, _), record in zip(expected[1], message[2]):
                        records[key] = dict(record, execution_device=worker.device)
                    dispatch(worker)
                seen = set()
                ordered = []
                for key in keys:
                    record = records[key]
                    ordered.append(dict(record, cache_hit=True) if key in seen else record)
                    seen.add(key)
                return ordered
            except BaseException:
                self.close()
                raise

    def close(self) -> None:
        """Interrupt pending work, terminate and reap every owned process."""
        # Never acquire the operation lock: a pending forward/read must not
        # prevent cancellation or shutdown from killing its worker.
        with self._close_lock:
            with self._state_lock:
                if self._closed:
                    return
                self._closed = True
                workers = tuple(self._workers)
            for worker in workers:
                if worker.process.pid is not None and worker.process.is_alive():
                    worker.process.terminate()
            deadline = time.monotonic() + 5
            for worker in workers:
                if worker.process.pid is not None:
                    worker.process.join(max(0, deadline - time.monotonic()))
            for worker in workers:
                if worker.process.pid is not None and worker.process.is_alive():
                    worker.process.kill()
            # SIGKILL also closes the pipe, unblocking any in-progress recv.
            for worker in workers:
                if worker.process.pid is not None:
                    worker.process.join()
                worker.connection.close()
                worker.process.close()
