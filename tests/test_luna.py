"""GPT-6 Luna backend against a local mock Decisions server: answers per question type,
refusals, rejected requests, snapshot pin, cache without re-billing, budget cap, 429 retry, runner accounting."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from gauntlet import luna, runner
from gauntlet.task import Item, Request, Task, read_jsonl

QUESTIONS = {
    "intent": {"type": "choice", "instructions": "Which edit intent?", "criteria": {"Claim": "a claim", "Fact": "evidence", "Other": "other"}},
    "factual": {"type": "noul", "instructions": "Is it factual?", "criteria": {"true": "yes", "false": "no"}},
    "level": {"type": "score", "instructions": "How severe?", "criteria": ["low", "mid", "high"]},
}
ANSWERS = {
    "intent": {"type": "choice", "choice": "Fact", "probabilities": {"Claim": 0.0, "Fact": 1.0, "Other": 0.0}, "confidence": 1.0},
    "factual": {"type": "noul", "noul": 0.25},
    "level": {"type": "score", "score": 1.06, "legend": {"0": "low", "1": "mid", "2": "high"},
              "probabilities": {"0": 0.04, "1": 0.86, "2": 0.1}, "confidence": 0.79},
}


class MockDecisions:
    """Local HTTP Decisions endpoint: answers from `answers`, or the queued (status, payload, headers)."""

    def __init__(self) -> None:
        self.bodies: list[dict] = []
        self.queue: list[tuple[int, object, dict]] = []
        self.answers = dict(ANSWERS)
        self.model = luna.SNAPSHOT
        self.provider = "OpenAI"
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                mock.bodies.append(body)
                if mock.queue:
                    status, payload, headers = mock.queue.pop(0)
                else:
                    status, headers = 200, {}
                    payload = {"model": mock.model, "provider": mock.provider, "id": f"gen-{len(mock.bodies)}",
                               "answers": {q: mock.answers[q] for q in body["questions"] if q in mock.answers},
                               "usage": {"input_tokens": 1000, "output_tokens": 0, "cost": 1e-4}}
                data = json.dumps(payload).encode() if not isinstance(payload, str) else payload.encode()
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/api/alpha/decisions"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def server():
    mock = MockDecisions()
    yield mock
    mock.server.shutdown()


def client(tmp_path, server, total_cap=250.0, run_cap=None):
    spend = luna.SpendLedger(tmp_path / "spend.jsonl", run="zero-shot", total_cap=total_cap, run_cap=run_cap)
    return luna.LunaClient(tmp_path / "cache", api_key="test-key", url=server.url, requests_per_minute=10**6, spend=spend)


def decide(c, questions=QUESTIONS, state="Export fails in Safari."):
    body = {"model": c.model, "state": state, "questions": questions}

    async def go():
        async with httpx.AsyncClient() as http:
            return await c.decide(http, body)
    return asyncio.run(go())


def test_valid_answers_of_each_type_pass_through_and_the_body_is_jevs(tmp_path, server):
    rec = decide(client(tmp_path, server))
    assert server.bodies == [{"model": "openai/gpt-6-luna-decisions", "state": "Export fails in Safari.", "questions": QUESTIONS}]
    assert rec["response"]["answers"] == ANSWERS
    assert "refused" not in rec["response"]
    audit = luna.answer_audit(rec["response"])
    assert audit["n_answers"] == 3 and audit["n_refused_answers"] == 0
    assert audit["n_saturated_answers"] == 1  # the choice at probability exactly 1


@pytest.mark.parametrize("bad", [
    None,  # missing
    {"type": "refusal"},
    {"type": "noul", "noul": 0.9},  # wrong type
    {"type": "choice", "choice": "Opinion", "probabilities": {"Opinion": 1.0}},  # not a criterion
    {"type": "choice", "choice": "Fact"},  # no probabilities
])
def test_refused_or_invalid_answers_become_the_uniform_answer(tmp_path, server, bad):
    server.answers = {**ANSWERS, "intent": bad} if bad is not None else {k: v for k, v in ANSWERS.items() if k != "intent"}
    resp = decide(client(tmp_path, server))["response"]
    assert resp["refused"] == ["intent"] and resp["raw_answers"] == {"intent": bad}
    third = pytest.approx(1 / 3)
    assert resp["answers"]["intent"] == {"type": "choice", "choice": "Claim", "probabilities": {"Claim": third, "Fact": third, "Other": third}, "confidence": 0.0}
    assert resp["answers"]["factual"] == ANSWERS["factual"]
    assert luna.answer_audit(resp)["n_refused_answers"] == 1


def test_uniform_answers_for_noul_and_score():
    assert luna.uniform_answer(QUESTIONS["factual"]) == {"type": "noul", "noul": 0.5}
    level = luna.uniform_answer(QUESTIONS["level"])
    assert level["score"] == 1.0 and level["probabilities"] == {"0": pytest.approx(1 / 3), "1": pytest.approx(1 / 3), "2": pytest.approx(1 / 3)}
    assert level["legend"] == {"0": "low", "1": "mid", "2": "high"}


def test_cache_hit_does_not_rebill(tmp_path, server):
    c = client(tmp_path, server)
    first = decide(c)
    second = decide(client(tmp_path, server))  # a new process: cache on disk, ledger re-read
    assert first == second and len(server.bodies) == 1
    lines = (tmp_path / "spend.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["cost"] == 1e-4 and json.loads(lines[0])["run"] == "zero-shot"
    cached = json.loads(next((tmp_path / "cache").rglob("*.json")).read_text())
    assert cached["response"]["answers"] == ANSWERS  # verbatim API response kept


def test_budget_cap_stops_before_sending(tmp_path, server):
    (tmp_path / "spend.jsonl").write_text(json.dumps({"run": "rq3", "cost": 250 - 1e-6}) + "\n")  # headroom < the estimate
    with pytest.raises(luna.BudgetExceeded, match="total cap"):
        decide(client(tmp_path, server))
    assert server.bodies == []
    with pytest.raises(luna.BudgetExceeded, match="zero-shot cap"):
        decide(client(tmp_path, server, total_cap=1000.0, run_cap=0.00001))
    assert server.bodies == []
    decide(client(tmp_path, server, total_cap=1000.0))  # the other run's spend does not count against this run's cap
    assert len(server.bodies) == 1


def test_429_is_retried_after_retry_after(tmp_path, server):
    server.queue = [(429, {"error": {"code": 429, "message": "Rate limit exceeded"}}, {"Retry-After": "0.3"})]
    t0 = time.monotonic()
    rec = decide(client(tmp_path, server))
    assert time.monotonic() - t0 >= 0.3 and len(server.bodies) == 2
    assert rec["response"]["answers"] == ANSWERS


def test_rejected_request_is_cached_never_resent_and_answered_uniformly(tmp_path, server):
    server.queue = [(400, {"error": {"code": 400, "message": "Invalid request parameters"}}, {})]
    first = decide(client(tmp_path, server))["response"]
    again = decide(client(tmp_path, server))["response"]
    assert first == again and len(server.bodies) == 1
    assert first["model"] == luna.REJECTED_MODEL and first["rejected"]["status"] == 400
    assert first["refused"] == list(QUESTIONS) and first["answers"]["factual"] == {"type": "noul", "noul": 0.5}
    assert (tmp_path / "spend.jsonl").read_text() == ""  # nothing billed


def test_off_pin_snapshot_stops_and_is_not_cached(tmp_path, server):
    server.model = "openai/gpt-6-luna-decisions-20261120"
    with pytest.raises(luna.SnapshotMismatch):
        decide(client(tmp_path, server))
    assert [p.parent.name for p in (tmp_path / "cache").rglob("*.json")] == ["offpin"]
    assert len((tmp_path / "spend.jsonl").read_text().splitlines()) == 1  # billed, so logged
    server.model = luna.SNAPSHOT
    decide(client(tmp_path, server))
    assert len(server.bodies) == 2


REFUSAL_502 = (502, {"error": {"message": 'OpenAI refused to answer question "factual"', "code": 502}}, {"Retry-After": "0"})


def test_refusal_502_retried_then_answered(tmp_path, server):
    server.queue = [REFUSAL_502, REFUSAL_502]
    rec = decide(client(tmp_path, server))
    assert len(server.bodies) == 3 and rec["response"]["answers"] == ANSWERS


def test_persistent_refusal_502_refuses_the_whole_request_once(tmp_path, server):
    server.queue = [REFUSAL_502] * 3
    first = decide(client(tmp_path, server))["response"]
    again = decide(client(tmp_path, server))["response"]
    assert first == again and len(server.bodies) == 3  # cached, never re-sent
    assert first["model"] == luna.REFUSED_MODEL and first["refused_request"]["question"] == "factual"
    assert first["refused"] == list(QUESTIONS) and first["answers"]["factual"] == {"type": "noul", "noul": 0.5}
    audit = luna.answer_audit(first)
    assert (audit["n_refused_requests"], audit["n_refused_answers"], audit["n_saturated_answers"]) == (1, 3, 0)
    assert (tmp_path / "spend.jsonl").read_text() == ""  # not billed


def test_other_502_errors_are_not_refusals(tmp_path, server):
    server.queue = [(502, {"error": {"message": "Provider returned error", "code": 502}}, {"Retry-After": "0"})] * 3
    rec = decide(client(tmp_path, server))
    assert len(server.bodies) == 4 and "refused" not in rec["response"]


class TwoTypeTask(Task):
    name = "luna-tiny"
    prompt_file = "unused.json"

    def items(self):
        return [Item(str(i), "Fact") for i in range(4)]

    def requests(self, item):
        yield Request(item.id, "main", f"text {item.id}", {"intent": QUESTIONS["intent"], "factual": QUESTIONS["factual"]})

    def predict(self, item, answers):
        probs = answers["main"]["intent"]["probabilities"]
        return max(probs, key=probs.get)

    def score(self, items, predictions):
        return {"accuracy": sum(predictions[i.id] == i.gold for i in items) / len(items)}

    def prompt_sha256(self):
        return "0" * 64


def test_runner_records_normalized_responses_and_answer_audit(tmp_path, server, monkeypatch):
    monkeypatch.setattr(runner, "_git_head", lambda: "test-head")
    server.answers = {**ANSWERS, "factual": {"type": "refusal"}}
    c = client(tmp_path, server)
    result = runner.run_task(TwoTypeTask(), client=c, run_root=tmp_path / "runs")
    manifest = result["manifest"]
    assert manifest["model_requested"] == luna.MODEL and manifest["model_snapshots"] == {luna.SNAPSHOT: 4}
    assert manifest["answer_audit"] == {"n_answers": 8, "n_refused_answers": 4, "n_refused_requests": 0, "n_rejected_requests": 0,
                                        "n_requests": 4, "n_saturated_answers": 4}
    assert manifest["cost_usd"] == pytest.approx(4e-4) and result["scores"]["accuracy"] == 1.0
    rows = read_jsonl(tmp_path / "runs" / "luna-tiny" / "responses.jsonl.gz")
    assert all(r["response"]["refused"] == ["factual"] for r in rows)
    replay = runner.run_task(TwoTypeTask(), client=client(tmp_path, server), run_root=tmp_path / "runs")
    assert replay["scores"] == result["scores"] and len(server.bodies) == 4


def test_one_client_serves_consecutive_event_loops(tmp_path, server):
    spend = luna.SpendLedger(tmp_path / "spend.jsonl", run="zero-shot")
    # concurrency 1: the second request waits, which binds the semaphore to the running loop
    c = luna.LunaClient(tmp_path / "cache", api_key="test-key", url=server.url, requests_per_minute=10**6, spend=spend, concurrency=1)
    for state in ("a", "b", "c"):
        body = {"model": c.model, "state": state, "questions": QUESTIONS}

        async def go(body=body):
            async with httpx.AsyncClient() as http:
                return await asyncio.gather(c.decide(http, body), c.decide(http, {**body, "state": body["state"] * 2}))
        asyncio.run(go())
    assert len(server.bodies) == 6


def test_empty_200_body_is_retried(tmp_path, server):
    server.queue = [(200, "", {})]
    rec = decide(client(tmp_path, server))
    assert len(server.bodies) == 2 and rec["response"]["answers"] == ANSWERS


# ------------------------------------------------------------------------------------ Microsoft-Decision-1


def msd1_client(tmp_path, server):
    spend = luna.SpendLedger(tmp_path / "msd1_spend.jsonl", run="zero-shot", spec=luna.MSD1)
    return luna.LunaClient(tmp_path / "msd1_cache", api_key="test-key", url=server.url, requests_per_minute=10**6,
                           spec=luna.MSD1, spend=spend)


def test_msd1_spec_pins_model_provider_price_and_caps(tmp_path, server):
    server.model, server.provider = luna.MSD1.snapshot, "Azure"
    c = msd1_client(tmp_path, server)
    rec = decide(c)
    assert server.bodies[0]["model"] == "microsoft/microsoft-decision-1" and rec["response"]["answers"] == ANSWERS
    assert (c.spend.total_cap, c.spend.run_cap) == (150.0, 130.0)
    assert luna.SpendLedger(tmp_path / "x.jsonl", run="rq3", spec=luna.MSD1).run_cap == 20.0
    assert luna.estimate_cost({"a": "b" * 99}, luna.MSD1.price) == pytest.approx(54 * 0.042e-6)  # 108 bytes / 2 tokens
    assert luna.MSD1.spend_ledger.as_posix().endswith("runs/msd1/spend.jsonl")


def test_msd1_rejects_luna_snapshot_and_luna_rejects_msd1(tmp_path, server):
    with pytest.raises(luna.SnapshotMismatch, match="MSD1 settings"):
        decide(msd1_client(tmp_path, server))  # the mock answers as Luna / OpenAI
    server.model, server.provider = luna.MSD1.snapshot, "Azure"
    with pytest.raises(luna.SnapshotMismatch, match="Luna settings"):
        decide(client(tmp_path, server))


def test_msd1_refusal_of_any_provider_is_a_refused_request(tmp_path, server):
    server.model, server.provider = luna.MSD1.snapshot, "Azure"
    refusal = (502, {"error": {"message": 'Azure refused to answer question "factual"', "code": 502}}, {"Retry-After": "0"})
    server.queue = [refusal] * 3
    first = decide(msd1_client(tmp_path, server))["response"]
    assert first["model"] == luna.REFUSED_MODEL and first["refused_request"]["question"] == "factual"
    assert len(server.bodies) == 3 and (tmp_path / "msd1_spend.jsonl").read_text() == ""
    server.model, server.provider = luna.SNAPSHOT, "OpenAI"
    server.queue = [refusal] * 3  # Luna's rule stays OpenAI-only: a 502 from another provider is retried
    rec = decide(client(tmp_path, server), state="other")
    assert len(server.bodies) == 7 and "refused" not in rec["response"]


def test_msd1_budget_counts_only_its_own_ledger(tmp_path, server):
    server.model, server.provider = luna.MSD1.snapshot, "Azure"
    (tmp_path / "msd1_spend.jsonl").write_text(json.dumps({"run": "rq3", "cost": 150 - 1e-9}) + "\n")
    with pytest.raises(luna.BudgetExceeded, match="MSD1 total cap"):
        decide(msd1_client(tmp_path, server))
    assert server.bodies == []
    (tmp_path / "msd1_spend.jsonl").write_text("")
    (tmp_path / "spend.jsonl").write_text(json.dumps({"run": "zero-shot", "cost": 250.0}) + "\n")  # Luna at its cap
    decide(msd1_client(tmp_path, server))
    assert len(server.bodies) == 1
