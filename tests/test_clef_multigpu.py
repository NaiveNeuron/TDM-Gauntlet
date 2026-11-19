"""Hardware-free native cache/batching tests with real spawned worker processes."""

import asyncio
import json
import math
import multiprocessing
import os
import time
from collections import Counter
from functools import cached_property
from types import SimpleNamespace

import pytest

from gauntlet.clef import (
    MODEL,
    ClefClient,
    ClefContextError,
    ClefMultiGpuClient,
    ClefWorkerError,
)


def _probabilities(body):
    identifier = body["state"]["id"]
    result = {}
    for question_id, question in body["questions"].items():
        weights = {
            option: math.exp(math.sin((identifier + 1) * (index + 1)) + (identifier % 7) * index / 10)
            for index, option in enumerate(question["criteria"])
        }
        total = sum(weights.values())
        result[question_id] = {option: value / total for option, value in weights.items()}
    return result


class _ControlledClient(ClefClient):
    """Use native digests, cache writes, duplicate handling and length buckets.

    Only hardware configuration/encoding/forward are replaced. CPU inference
    produces numerical probabilities and logs actual batches from each process;
    gates make asymmetric completion and cleanup failures deterministic.
    """

    @cached_property
    def _configuration(self):
        if (self.cache_dir / f"fail-setup-{self.device}").exists():
            raise RuntimeError("controlled setup failure")
        configuration = {
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
            "device": self.device,
            "gpu": {"name": f"controlled-{self.device}", "compute_capability": [0, 0]},
            "python": "controlled",
            "libraries": {"controlled-client": "test"},
            "cuda_runtime": "controlled",
            "cudnn_version": 0,
            "numerical_configuration": {"controlled": True},
            "rounding_policy": "official systemone_answer: round(value, 4); raw probabilities retained",
            "monetary_cost": "not measured",
        }
        if (self.cache_dir / f"mismatch-{self.device}").exists():
            configuration["max_length"] += 1
        return configuration

    def _runtime(self):
        return None, None, None

    def _encode(self, body, processor, release):
        tokens = body["state"]["input_tokens"]
        if tokens > self.max_length:
            raise ClefContextError(f"Complete controlled input requires {tokens} tokens; truncation forbidden")
        return SimpleNamespace(input_ids=range(tokens)), tokens

    def _forward(self, records, model, processor, release):
        event = {
            "ids": [body["state"]["id"] for body, _, _ in records],
            "tokens": [len(encoded.input_ids) for _, encoded, _ in records],
            "pid": os.getpid(),
        }
        with (self.cache_dir / f"events-{self.device}.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        controls = {body["state"].get("control") for body, _, _ in records}
        if "hard-exit" in controls:
            os._exit(29)
        if "error" in controls:
            raise RuntimeError("controlled forward failure")
        if "hold" in controls:
            time.sleep(60)
        if "release-slow-window" in controls:
            (self.cache_dir / "release-slow-window").write_text("ready")
        if "slow-first-window" in controls:
            deadline = time.monotonic() + 10
            while not (self.cache_dir / "release-slow-window").exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError("faster worker never claimed the third original window")
                time.sleep(0.005)
        responses = []
        padded_tokens = len(records) * max(len(encoded.input_ids) for _, encoded, _ in records)
        for body, encoded, state_tokens in records:
            probabilities = _probabilities(body)
            responses.append({
                "model": self.model,
                "answers": {question: max(options, key=options.get) for question, options in probabilities.items()},
                "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0},
                "audit": {
                    "raw_probabilities": probabilities,
                    "inference_seconds": 0.01,
                    "batch_inference_seconds": 0.01 * len(records),
                    "batch_size": len(records),
                    "batch_padded_tokens": padded_tokens,
                    "state_tokens": state_tokens,
                    "encoded_state_tokens": state_tokens,
                    "input_tokens": len(encoded.input_ids),
                    "truncated": False,
                    "rounding_policy": self._configuration["rounding_policy"],
                },
            })
        return responses


def _body(identifier, *, control=None, tokens=None, reverse=False):
    criteria = {"allow": "permitted", "deny": "forbidden"}
    if reverse:
        criteria = dict(reversed(list(criteria.items())))
    body = {
        "model": MODEL,
        "state": {"id": identifier, "input_tokens": tokens or 4 + identifier % 5},
        "questions": {"decision": {"type": "choice", "criteria": criteria}},
    }
    if control is not None:
        body["state"]["control"] = control
    return body


def _client(tmp_path):
    client = ClefMultiGpuClient(tmp_path / "cache", ["cuda:0", "cuda:1"])
    client._client_factory = _ControlledClient
    return client


def _events(client, device):
    path = client.cache_dir / f"events-{device}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _assert_reaped(pids):
    assert not set(pids).intersection(process.pid for process in multiprocessing.active_children())


def test_mixed_device_caches_keep_native_hashes_bytes_and_compute_flags(tmp_path):
    client = _client(tmp_path)
    cached_one, cached_zero, missing = _body(1), _body(2), _body(3)
    native_one = _ControlledClient(client.cache_dir, device="cuda:1")
    native_zero = _ControlledClient(client.cache_dir, device="cuda:0")
    seed_one = asyncio.run(native_one.decide_many(None, [cached_one]))[0]
    seed_zero = asyncio.run(native_zero.decide_many(None, [cached_zero]))[0]
    saved = {
        native_one._cache_path(seed_one["request_hash"]): native_one._cache_path(seed_one["request_hash"]).read_bytes(),
        native_zero._cache_path(seed_zero["request_hash"]): native_zero._cache_path(seed_zero["request_hash"]).read_bytes(),
    }
    for device in client.devices:
        (client.cache_dir / f"events-{device}.jsonl").unlink()
    try:
        metadata = client.metadata
        pids = [worker.process.pid for worker in client._workers]
        records = asyncio.run(client.decide_windows(
            [cached_one, cached_zero, missing, missing, cached_one], "mixed-cache",
        ))
        assert [record["cache_hit"] for record in records] == [True, True, False, True, True]
        assert [record["execution_device"] for record in records] == ["cuda:1", "cuda:0", "cuda:0", "cuda:0", "cuda:1"]
        assert records[0]["request_hash"] == seed_one["request_hash"]
        assert records[1]["request_hash"] == seed_zero["request_hash"]
        assert records[2]["request_hash"] == native_zero._digest(missing)
        assert not native_zero._cache_path(native_zero._digest(cached_one)).exists()
        assert not native_one._cache_path(native_one._digest(cached_zero)).exists()
        assert not native_one._cache_path(native_one._digest(missing)).exists()
        assert {path: path.read_bytes() for path in saved} == saved
        assert _events(client, "cuda:1") == []
        assert [event["ids"] for event in _events(client, "cuda:0")] == [[3]]
        assert _events(client, "cuda:0")[0]["pid"] in pids
        assert len(set(pids)) == 2 and os.getpid() not in pids
        assert "device" not in metadata and "gpu" not in metadata
        assert metadata["devices"] == [native_zero.metadata, native_one.metadata]
        assert metadata["model_id"] == native_zero.model_id
        assert metadata["revision"] == native_zero.revision
    finally:
        client.close()
    _assert_reaped(pids)


def test_fast_worker_claims_next_original_window_and_rounds_reuse_processes(tmp_path):
    client = _client(tmp_path)
    bodies = [_body(index) for index in range(192)]
    bodies[0] = _body(0, control="slow-first-window")
    bodies[128] = _body(128, control="release-slow-window")
    # A duplicate across original windows is owned only by its first occurrence.
    bodies[70] = bodies[0]
    bodies[140] = bodies[4]
    try:
        client.metadata
        pids = [worker.process.pid for worker in client._workers]
        records = asyncio.run(client.decide_windows(bodies, "asymmetric"))
        assert len(records) == len(bodies)
        for body, record in zip(bodies, records):
            assert record["response"]["audit"]["raw_probabilities"] == _probabilities(body)
            assert record["response"]["usage"]["input_tokens"] == body["state"]["input_tokens"]
            native = _ControlledClient(client.cache_dir, device=record["execution_device"])
            assert record["request_hash"] == native._digest(body)
        assert all(record["execution_device"] == "cuda:0" for record in records[:64])
        assert all(record["execution_device"] == "cuda:1" for index, record in enumerate(records[64:], 64) if index not in (70, 140))
        assert [index for index, record in enumerate(records) if record["cache_hit"]] == [70, 140]
        computed = Counter()
        for device in client.devices:
            for event in _events(client, device):
                computed.update(event["ids"])
                assert len({identifier // 64 for identifier in event["ids"]}) == 1
                assert len(event["ids"]) <= 16
                assert len(event["ids"]) * max(event["tokens"]) <= 16384
        assert computed == Counter(body["state"]["id"] for index, body in enumerate(bodies) if index not in (70, 140))
        next_body = _body(193)
        next_records = asyncio.run(client.decide_windows([next_body, bodies[128], next_body], "next-round"))
        assert [worker.process.pid for worker in client._workers] == pids
        assert [record["cache_hit"] for record in next_records] == [False, True, True]
        assert next_records[1]["execution_device"] == "cuda:1"
        assert next_records[1]["request_hash"] == records[128]["request_hash"]
    finally:
        client.close()
    _assert_reaped(pids)


def test_mapping_order_is_not_deduplicated_or_rewritten(tmp_path):
    client = _client(tmp_path)
    normal, reversed_schema = _body(9), _body(9, reverse=True)
    try:
        records = asyncio.run(client.decide_windows([normal, reversed_schema, normal], "schema-order"))
        assert records[0]["request_hash"] != records[1]["request_hash"]
        assert [record["cache_hit"] for record in records] == [False, False, True]
        assert records[0]["response"]["audit"]["raw_probabilities"] == _probabilities(normal)
        assert records[1]["response"]["audit"]["raw_probabilities"] == _probabilities(reversed_schema)
        assert records[0]["response"]["audit"]["raw_probabilities"] != records[1]["response"]["audit"]["raw_probabilities"]
        assert Counter(identifier for event in _events(client, "cuda:0") for identifier in event["ids"]) == {9: 2}
    finally:
        client.close()


@pytest.mark.parametrize("failure", ["error", "context", "hard-exit"])
def test_worker_failure_propagates_and_reaps_other_busy_worker(tmp_path, failure):
    client = _client(tmp_path)
    bodies = [_body(index) for index in range(65)]
    bodies[0] = _body(0, tokens=32001) if failure == "context" else _body(0, control=failure)
    bodies[64] = _body(64, control="hold")
    try:
        client.metadata
        pids = [worker.process.pid for worker in client._workers]
        error_type = ClefContextError if failure == "context" else ClefWorkerError
        with pytest.raises(error_type):
            asyncio.run(client.decide_windows(bodies, "failure-round"))
        _assert_reaped(pids)
        with pytest.raises(ClefWorkerError):
            _ = client.metadata
    finally:
        client.close()
        client.close()


@pytest.mark.parametrize("flag", ["fail-setup-cuda:1", "mismatch-cuda:1"])
def test_startup_failure_cleans_up_all_started_processes(tmp_path, flag):
    client = _client(tmp_path)
    baseline = {process.pid for process in multiprocessing.active_children()}
    (client.cache_dir / flag).write_text("enabled")
    with pytest.raises(ClefWorkerError):
        _ = client.metadata
    assert {process.pid for process in multiprocessing.active_children()} <= baseline
    client.close()


def test_async_cancellation_interrupts_pending_forward_and_reaps_workers(tmp_path):
    client = _client(tmp_path)

    async def scenario():
        client.metadata
        pids = [worker.process.pid for worker in client._workers]
        operation = asyncio.create_task(client.decide_windows([_body(0, control="hold")], "cancel-round"))
        deadline = time.monotonic() + 10
        try:
            while not _events(client, "cuda:0"):
                if time.monotonic() >= deadline:
                    pytest.fail("controlled forward never started")
                await asyncio.sleep(0.005)
            operation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(operation, timeout=10)
            _assert_reaped(pids)
        finally:
            client.close()
            if not operation.done():
                operation.cancel()
                await asyncio.gather(operation, return_exceptions=True)

    asyncio.run(scenario())
