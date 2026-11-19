"""Base-model log-prob backend for the executor (mocked scoring server; no torch, no GPU)."""

from __future__ import annotations

import gzip
import json
import math

import pytest

from jev_opt.clef import ClefServerError
from jev_opt.execute import run_examples
from jev_opt.ir import load_decider
from jev_opt.jev import ResponseLedger, make_client
from jev_opt.logprob import (PINS, SCORING, LogprobBackend, LogprobCacheMiss, LogprobRejected, cache_key, render_unit,
                             unit_body)
from jev_opt.task import Example

MODEL = PINS["qwen-base"]
CHOICE = {"type": "choice", "instructions": "Tone?", "criteria": {"pos": "positive", "neg": "negative", "mix": ""}}
NOUL = {"type": "noul", "instructions": "Is it sarcastic?"}
SCORE = {"type": "score", "instructions": "How formal?", "criteria": ["casual", "neutral", "formal"]}


class FakeServer:
    """Scores like the real server: per-unit log-likelihoods (state- and option-dependent), softmax probabilities."""

    def __init__(self, model: str = MODEL, max_length: int = 32000, scoring: dict | None = None, guard: int | None = None):
        self.model = model
        self.configuration = {**(scoring or SCORING), "max_length": max_length, "device": "cuda:9", "gpu": {"name": "fake"}}
        self.windows: list[list[dict]] = []
        self.guard = guard

    def compute(self, items):
        self.windows.append([payload for _, payload in items])
        out = []
        for key, payload in items:
            if self.guard is not None and len(payload["prompt"]) > self.guard:
                out.append({"key": key, "error": "complete prompt requires 99999 tokens", "type": "LogprobContextError"})
                continue
            lp = [-((len(payload["prompt"]) * (i + 3)) % 7) / 2.0 for i in range(len(payload["continuations"]))]
            z = sum(math.exp(x) for x in lp)
            out.append({"key": key, "response": {"model": self.model, "probabilities": [math.exp(x) / z for x in lp], "logprobs": lp,
                                                 "option_mass": z, "usage": {"input_tokens": len(payload["prompt"]) // 4},
                                                 "audit": {"inference_seconds": 0.25}}})
        return out

    def close(self):
        pass


def backend(tmp_path, server: FakeServer | None, **kw) -> LogprobBackend:
    return LogprobBackend("qwen-base", tmp_path / "cache", device=None if server is None else "cuda:9",
                          server_factory=None if server is None else (lambda: server), **kw)


def test_prompt_follows_the_registered_template() -> None:
    prompt, cont = render_unit({"text": "so fun", "a": 1}, CHOICE)
    assert prompt == ("<|im_start|>system\nRead the state and answer the question about it. Reply with exactly one of the allowed "
                      "answers and nothing else.<|im_end|>\n<|im_start|>user\nSTATE:\n{\"a\":1,\"text\":\"so fun\"}\n\nQUESTION:\nTone?\n\n"
                      "ALLOWED ANSWERS:\nA. pos: positive\nB. neg: negative\nC. mix\nReply with the letter of one allowed answer.<|im_end|>\n"
                      "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert cont == ["A", "B", "C"]
    noul, cont = render_unit("plain text", NOUL)
    assert noul.split("QUESTION:\n", 1)[1].startswith("Is it sarcastic?\n\nReply with Yes or No.<|im_end|>") and cont == ["Yes", "No"]
    score, cont = render_unit("x", SCORE)
    assert "LEVELS (lowest to highest):\n1. casual\n2. neutral\n3. formal\nReply with the number of one level." in score
    assert cont == ["1", "2", "3"]


def test_units_are_scored_alone_without_question_ids_and_answers_have_the_release_format(tmp_path) -> None:
    server = FakeServer()
    client = backend(tmp_path, server)
    d = client.decide({"text": "hello"}, {"tone": CHOICE, "sarc": NOUL, "formal": SCORE, "tone_again": CHOICE})
    assert [len(w) for w in server.windows] == [3]  # tone and tone_again are one unit
    probs = {p["continuations"][0]: p for p in server.windows[0]}
    assert set(probs) == {"A", "Yes", "1"}
    resp = {x["key"]: x["response"] for x in server.compute([(k, p) for k, p in zip("abc", server.windows[0])])}
    ch, no, sc = (resp[k]["probabilities"] for k in "abc")
    assert d.answers["tone"] == d.answers["tone_again"]
    assert d.answers["tone"]["probabilities"] == {k: round(v, 4) for k, v in zip(["pos", "neg", "mix"], ch)}
    assert d.answers["tone"]["choice"] == ["pos", "neg", "mix"][max(range(3), key=lambda i: (ch[i], -i))]
    assert d.answers["sarc"] == {"type": "noul", "noul": round(no[0], 4)}
    assert d.answers["formal"]["score"] == round(sum(i * p for i, p in enumerate(sc)), 4)
    assert d.answers["formal"]["probabilities"] == {str(i): round(p, 4) for i, p in enumerate(sc)}
    assert client.usage.calls == 3 and client.usage.fresh_calls == 3 and client.fresh_inference_seconds == 0.75
    client.close()
    other = backend(tmp_path, None)  # same unit under another id and request: a cache hit, no device needed
    assert other.decide({"text": "hello"}, {"anything": NOUL}).answers["anything"] == d.answers["sarc"]
    with pytest.raises(LogprobCacheMiss):
        other.decide({"text": "unseen"}, {"anything": NOUL})


def test_key_pins_model_guard_state_and_question_order() -> None:
    u = unit_body(MODEL, {"a": 1, "b": 2}, CHOICE)
    swapped = unit_body(MODEL, {"a": 1, "b": 2}, {**CHOICE, "criteria": dict(reversed(list(CHOICE["criteria"].items())))})
    keys = {cache_key(MODEL, 32000, u), cache_key(MODEL, 16000, u), cache_key("other@rev", 32000, u),
            cache_key(MODEL, 32000, swapped), cache_key(MODEL, 32000, unit_body(MODEL, {"b": 2, "a": 1}, CHOICE))}
    assert len(keys) == 5


def test_rejected_unit_fails_only_its_example_and_cache_replay_is_identical(tmp_path) -> None:
    prog = load_decider("""
version: 1
labels: [pos, neg]
inputs: {text: ''}
root: tone
nodes:
  tone:
    kind: choice
    instructions: Tone?
    state: [text]
    options:
      pos: {criterion: positive}
      neg: {criterion: negative}
""")
    examples = [Example(str(i), {"text": "x" * (5 + i if i != 3 else 5000)}, "pos", "val") for i in range(6)]
    live = backend(tmp_path, FakeServer(guard=2000))
    (tmp_path / "run").mkdir()
    led = ResponseLedger(tmp_path / "run", name="lp_responses")
    live.ledger = led
    first = run_examples(prog, examples, live)
    live.close()
    led.finalize()
    assert [r.error is not None for r in first] == [False, False, False, True, False, False]
    assert first[3].error.startswith("LogprobRejected")
    replay = run_examples(prog, examples[:3] + examples[4:], backend(tmp_path, None))
    assert [(r.id, r.pred, r.dist) for r in replay] == [(r.id, r.pred, r.dist) for r in first if r.error is None]
    rows = [json.loads(line) for line in gzip.open(tmp_path / "run" / "lp_responses.jsonl.gz", "rt")]
    assert len(rows) == 5 and all(r["request"]["template"] == "qwen-base-logprob-v1" and "question" in r["request"] for r in rows)


def test_question_outside_the_template_is_rejected_before_any_inference(tmp_path) -> None:
    server = FakeServer()
    client = backend(tmp_path, server)
    wide = {"type": "choice", "instructions": "Pick", "criteria": {f"o{i}": "" for i in range(27)}}
    with pytest.raises(LogprobRejected, match="27 options"):
        client.decide({"t": 1}, {"q": wide})
    with pytest.raises(LogprobRejected, match="10 levels"):
        client.decide({"t": 1}, {"q": {**SCORE, "criteria": [str(i) for i in range(10)]}})
    assert server.windows == []
    client.close()


def test_server_with_other_pin_settings_or_guard_is_refused(tmp_path) -> None:
    for server in (FakeServer(model="Qwen/Qwen3.8-27B@other"), FakeServer(max_length=16000),
                   FakeServer(scoring={**SCORING, "dtype": "float16"})):
        client = LogprobBackend("qwen-base", tmp_path / "c", device="cuda:9", server_factory=lambda s=server: s)
        with pytest.raises(ClefServerError, match="differs"):
            client.decide({"t": 1}, {"q": NOUL})
        client.close()


def test_make_client_selects_the_base_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("JEV_OPT_BACKEND", "qwen-base")
    monkeypatch.setenv("JEV_OPT_LOGPROB_CACHE", str(tmp_path / "lp"))
    monkeypatch.delenv("JEV_OPT_DEVICE", raising=False)
    client = make_client(tmp_path / ".jev-cache", "typesafe/jev-1.13", 1800, 16, [tmp_path])
    assert isinstance(client, LogprobBackend) and client.model == MODEL and client.device is None
    assert client.cache_dir == tmp_path / "lp"
