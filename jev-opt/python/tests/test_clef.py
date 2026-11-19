"""Clef backend for the executor (mocked inference server; no torch, no GPU)."""

from __future__ import annotations

import json
import threading

import pytest

from jev_opt.clef import ENCODING, PINS, ClefBackend, ClefCacheMiss, ClefRejected, ClefServerError, cache_key
from jev_opt.execute import run_examples
from jev_opt.ir import load_decider
from jev_opt.jev import ResponseLedger, make_client
from jev_opt.task import Example

MODEL = PINS["clef-flash"]
Q = {"tone": {"type": "choice", "instructions": "Tone?", "criteria": {"pos": "positive", "neg": "negative"}}}


class FakeServer:
    """Answers like the release (four-decimal public answers + raw audit) and logs every window."""

    def __init__(self, model: str = MODEL, max_length: int = 32000, reject: str | None = None, gate: threading.Event | None = None):
        self.model = model
        self.configuration = {**ENCODING, "max_length": max_length, "device": "cuda:9", "gpu": {"name": "fake"}}
        self.windows: list[list[str]] = []
        self.reject = reject
        self.gate = gate
        self.closed = False

    def compute(self, items):
        if self.gate is not None:
            self.gate.wait(5)
        self.windows.append([key for key, _ in items])
        out = []
        for key, body in items:
            if self.reject is not None and self.reject in json.dumps(body["state"]):
                out.append({"key": key, "error": "input requires 40000 tokens (limit 32000)", "type": "ClefContextError"})
                continue
            answers = {}
            for qid, q in body["questions"].items():
                p = 0.25 + 0.5 * (len(json.dumps(body["state"])) % 2)  # deterministic, state-dependent
                if q["type"] == "noul":
                    answers[qid] = {"type": "noul", "noul": round(p, 4)}
                else:
                    opts = list(q["criteria"])
                    probs = {o: (p if i == 0 else (1 - p) / (len(opts) - 1)) for i, o in enumerate(opts)}
                    answers[qid] = {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs}
            out.append({"key": key, "response": {"model": self.model, "answers": answers, "usage": {"input_tokens": 100, "output_tokens": 0},
                                                 "audit": {"inference_seconds": 0.5, "raw_probabilities": {}}}})
        return out

    def close(self):
        self.closed = True


def backend(tmp_path, server: FakeServer | None, **kw) -> ClefBackend:
    return ClefBackend("clef-flash", tmp_path / "cache", device=None if server is None else "cuda:9",
                       server_factory=None if server is None else (lambda: server), **kw)


def test_key_is_order_sensitive_and_pins_model_and_guard() -> None:
    body = {"model": MODEL, "state": {"a": 1, "b": 2}, "questions": Q}
    swapped = {"model": MODEL, "state": {"b": 2, "a": 1}, "questions": Q}
    assert cache_key(MODEL, 32000, body) == cache_key(MODEL, 32000, json.loads(json.dumps(body)))
    assert cache_key(MODEL, 32000, body) != cache_key(MODEL, 32000, swapped)  # Clef's encoder reads keys in order
    assert cache_key(MODEL, 32000, body) != cache_key(MODEL, 65536, body)
    assert cache_key(MODEL, 32000, body) != cache_key(PINS["clef"], 32000, body)


def test_concurrent_misses_share_windows_then_replay_from_cache_only(tmp_path) -> None:
    gate = threading.Event()
    server = FakeServer(gate=gate)
    client = backend(tmp_path, server, window=8, linger_s=0.2)
    states = [{"text": "x" * i} for i in range(6)] + [{"text": "x" * 2}]  # one duplicate in flight
    results: dict[int, object] = {}

    def call(i: int) -> None:
        results[i] = client.decide(states[i], Q, {"split": "val"})

    threads = [threading.Thread(target=call, args=(i,)) for i in range(len(states))]
    for t in threads:
        t.start()
    gate.set()
    for t in threads:
        t.join()
    client.close()
    assert server.closed
    assert sum(len(w) for w in server.windows) == 6 and len(server.windows) <= 2  # the duplicate is computed once
    assert client.usage.calls == 7 and client.usage.fresh_calls == 6
    assert client.fresh_inference_seconds == pytest.approx(3.0)
    assert {results[2].answers["tone"]["choice"]} == {results[6].answers["tone"]["choice"]}

    replay = backend(tmp_path, None)  # no device: cache-only
    for i, state in enumerate(states):
        d = replay.decide(state, Q)
        assert d.cached and d.answers == results[i].answers and d.cost == 0.0
    assert replay.usage.fresh_calls == 0
    with pytest.raises(ClefCacheMiss):
        replay.decide({"text": "never asked"}, Q)


def test_request_is_sent_whole_and_recorded_in_the_ledger(tmp_path) -> None:
    server = FakeServer()
    client = backend(tmp_path, server)
    many = {f"q{i}": {"type": "noul", "instructions": f"is it {i}?"} for i in range(70)}  # Jev would chunk at 64
    (tmp_path / "run").mkdir()
    client.ledger = ResponseLedger(tmp_path / "run", name="clef_responses")
    d = client.decide({"text": "hello"}, many, {"split": "train", "condition": "c"})
    client.close()
    assert len(server.windows) == 1 and len(d.answers) == 70
    rec = json.loads(next((tmp_path / "cache").glob("??/*.json")).read_text())
    assert list(rec["request"]["questions"]) == list(many) and rec["request"]["model"] == MODEL
    conf = json.loads((tmp_path / "cache" / "configurations" / f"{rec['configuration']}.json").read_text())
    assert conf["device"] == "cuda:9"
    info = client.ledger.finalize()
    assert info["shards"] == ["clef_responses.jsonl.gz"] and info["records"] == 1


def test_encoder_rejection_fails_only_its_example(tmp_path) -> None:
    program = load_decider(
        "version: 1\nlabels: [pos, neg]\ninputs: {text: ''}\nroot: tone\nnodes:\n  tone:\n    kind: choice\n"
        "    instructions: Tone?\n    state: [text]\n    options:\n      pos: {criterion: positive}\n      neg: {criterion: negative}\n")
    examples = [Example(f"e{i}", {"text": t}, "pos", "val") for i, t in enumerate(["good", "TOO LONG", "fine"])]
    client = backend(tmp_path, FakeServer(reject="TOO LONG"))
    res = run_examples(program, examples, client)
    client.close()
    assert [r.error is None for r in res] == [True, False, True]
    assert res[1].error.startswith("ClefRejected") and res[1].pred is None
    assert len(list((tmp_path / "cache").glob("??/*.json"))) == 2  # rejections are not cached
    with pytest.raises(ClefRejected):
        b = backend(tmp_path, FakeServer(reject="TOO LONG"))
        try:
            b.decide({"text": "TOO LONG"}, Q)
        finally:
            b.close()


def test_server_with_other_pin_or_guard_is_refused_for_good(tmp_path) -> None:
    starts = []

    def factory():
        starts.append(1)
        return FakeServer(max_length=65536)

    client = ClefBackend("clef-flash", tmp_path / "c", device="cuda:9", server_factory=factory)
    for _ in range(2):
        with pytest.raises(ClefServerError, match="max_length"):
            client.decide({"text": "a"}, Q)
    client.close()
    assert len(starts) == 1
    other = ClefBackend("clef-flash", tmp_path / "d", device="cuda:9", server_factory=lambda: FakeServer(model=PINS["clef"]))
    with pytest.raises(ClefServerError, match="model"):
        other.decide({"text": "a"}, Q)
    other.close()


def test_make_client_selects_the_backend_from_the_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("JEV_OPT_BACKEND", "clef")
    monkeypatch.setenv("JEV_OPT_CLEF_CACHE", str(tmp_path / "clef-cache"))
    monkeypatch.delenv("JEV_OPT_DEVICE", raising=False)
    client = make_client(tmp_path / ".jev-cache", "typesafe/jev-1.13", 1800, 16, [tmp_path])
    assert isinstance(client, ClefBackend) and client.model == PINS["clef"] and client.device is None
    assert client.cache_dir == tmp_path / "clef-cache" and client.concurrency == 64
    with pytest.raises(ClefCacheMiss):
        client.decide({"text": "a"}, Q)
    monkeypatch.setenv("JEV_OPT_BACKEND", "gpt")
    with pytest.raises(ValueError):
        make_client(tmp_path / ".jev-cache", "m", 1, 1, [tmp_path])
