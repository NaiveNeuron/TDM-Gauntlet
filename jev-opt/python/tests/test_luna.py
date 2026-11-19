"""GPT-6 Luna backend (jev_opt.luna) against a local mock Decisions server (no network beyond 127.0.0.1)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from jev_opt.jev import JevClient, make_client
from jev_opt.luna import (MODEL, PROVIDER, SNAPSHOT, BudgetExceeded, LunaClient, LunaRejected, SnapshotMismatch,
                          SpendLedger)

CHOICE = {"type": "choice", "instructions": "Tone?", "criteria": {"pos": "positive", "neg": "negative", "neu": "neutral"}}
NOUL = {"type": "noul", "instructions": "Mentions a refund.", "criteria": {"true": "yes", "false": "no"}}
SCORE = {"type": "score", "instructions": "Fluency?", "criteria": ["poor", "fair", "good"]}


def answer(q: dict[str, Any]) -> dict[str, Any]:
    if q["type"] == "noul":
        return {"type": "noul", "noul": 0.8}
    if q["type"] == "choice":
        keys = list(q["criteria"])
        probs = {k: (0.7 if i == 0 else round(0.3 / (len(keys) - 1), 2)) for i, k in enumerate(keys)}
        return {"type": "choice", "choice": keys[0], "probabilities": probs, "confidence": 0.79}
    n = len(q["criteria"])
    probs = {str(i): (0.8 if i == n - 1 else round(0.2 / (n - 1), 2)) for i in range(n)}
    return {"type": "score", "score": 1.7, "legend": {str(i): c for i, c in enumerate(q["criteria"])}, "probabilities": probs, "confidence": 0.79}


class LunaServer:
    """Real HTTP server on 127.0.0.1: Jev-shaped answers from the pinned snapshot; scripted replies first."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.raw: list[bytes] = []
        self.script: list[tuple[int, dict[str, str], Any]] = []  # (status, headers, JSON payload or text)
        self.answers: dict[str, Any] | None = None  # verbatim `answers` override for the next 200
        self.model = SNAPSHOT
        self.provider = PROVIDER
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                body = json.loads(raw)
                server.raw.append(raw)
                server.bodies.append(body)
                if server.script:
                    status, headers, payload = server.script.pop(0)
                else:
                    answers = server.answers if server.answers is not None else {k: answer(q) for k, q in body["questions"].items()}
                    status, headers, payload = 200, {}, {"id": f"gen-{len(server.bodies)}", "model": server.model, "provider": server.provider,
                                                         "answers": answers, "usage": {"input_tokens": 120, "output_tokens": 0, "cost": 1.2e-05}}
                data = (json.dumps(payload) if not isinstance(payload, str) else payload).encode()
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/api/alpha/decisions"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch):
    s = LunaServer()
    monkeypatch.setenv("JEV_OPT_DECISIONS_URL", s.url)
    yield s
    s.close()


def luna(tmp_path, sleeps: list[float] | None = None, **spend_kw: Any) -> LunaClient:
    spend = SpendLedger(tmp_path / "spend.jsonl", run="rq3", **spend_kw)
    return LunaClient(tmp_path / "cache", "test-key", spend, max_rpm=10**9, sleep=(sleeps.append if sleeps is not None else lambda s: None))


def ledger_lines(tmp_path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in (tmp_path / "spend.jsonl").read_text().splitlines()]


def test_question_types_round_trip_and_request_is_sent_verbatim(tmp_path, server) -> None:
    client = luna(tmp_path)
    state = {"text": "The refund never arrived.", "meta": {"lang": "en"}}
    questions = {"tone": CHOICE, "refund": NOUL, "fluency": SCORE}
    d = client.decide(state, questions)
    assert server.bodies == [{"model": MODEL, "state": state, "questions": questions}]
    assert list(server.bodies[0]) == ["model", "state", "questions"] and list(server.bodies[0]["questions"]) == ["tone", "refund", "fluency"]
    assert d.answers == {q: answer(spec) for q, spec in questions.items()}
    assert (d.snapshot, d.input_tokens, d.cost, d.cached) == (SNAPSHOT, 120, 1.2e-05, False)
    assert (client.answers_seen, client.refused_answers, client.saturated_answers) == (3, 0, 0)
    [row] = ledger_lines(tmp_path)
    assert row["run"] == "rq3" and row["model"] == SNAPSHOT and row["provider"] == PROVIDER and row["cost"] == 1.2e-05
    assert set(row) == {"time", "run", "request_hash", "model", "provider", "input_tokens", "cost", "id"}


def test_refused_and_invalid_answers_become_uniform_and_are_counted(tmp_path, server) -> None:
    questions = {"a": CHOICE, "b": NOUL, "c": SCORE, "d": CHOICE, "e": CHOICE, "f": NOUL}
    server.answers = {
        "a": {"type": "refusal", "refusal": "cannot help"},
        # "b" missing
        "c": {"type": "choice", "choice": "0", "probabilities": {"0": 1.0}},  # wrong type
        "d": {"type": "choice", "choice": "other", "probabilities": {"pos": 0.5, "other": 0.5}},  # unknown option
        "e": {"type": "choice", "choice": "neg", "probabilities": {"pos": 0.0, "neg": 1.0, "neu": 0.0}},  # valid, saturated
        "f": {"type": "noul", "noul": 0.3},
    }
    client = luna(tmp_path)
    d = client.decide({"text": "x"}, questions)
    third = pytest.approx(1 / 3)
    assert d.answers["a"] == {"type": "choice", "choice": "pos", "probabilities": {"pos": third, "neg": third, "neu": third}, "confidence": 0.0}
    assert d.answers["b"] == {"type": "noul", "noul": 0.5}
    assert d.answers["c"] == {"type": "score", "score": 1.0, "legend": {"0": "poor", "1": "fair", "2": "good"},
                              "probabilities": {"0": third, "1": third, "2": third}, "confidence": 0.0}
    assert d.answers["d"]["probabilities"] == {"pos": third, "neg": third, "neu": third}
    assert d.answers["e"] == server.answers["e"] and d.answers["f"] == server.answers["f"]
    assert (client.answers_seen, client.refused_answers, client.saturated_answers) == (6, 4, 1)
    # the cache keeps the verbatim response; substitution happens on read, identically
    again = client.decide({"text": "x"}, questions)
    assert again.answers == d.answers and again.cached and len(server.bodies) == 1
    assert client.refused_answers == 8
    [cached] = [p for p in (tmp_path / "cache").rglob("*.json")]
    assert json.loads(cached.read_text())["response"]["answers"] == server.answers


def test_cache_hit_is_not_sent_or_billed_again(tmp_path, server) -> None:
    client = luna(tmp_path)
    first = client.decide({"text": "x"}, {"q": NOUL})
    lines = (tmp_path / "spend.jsonl").read_text()
    second = luna(tmp_path).decide({"text": "x"}, {"q": NOUL})  # new process, same cache and ledger
    assert len(server.bodies) == 1 and (tmp_path / "spend.jsonl").read_text() == lines
    assert second.answers == first.answers and second.cached and (second.input_tokens, second.cost) == (120, 1.2e-05)
    assert client.usage.fresh_calls == 1 and client.usage.fresh_cost == 1.2e-05


def test_budget_caps_stop_before_sending(tmp_path, server) -> None:
    with pytest.raises(BudgetExceeded, match="rq3 cap"):
        luna(tmp_path, run_cap=1e-6).decide({"text": "x"}, {"q": NOUL})
    # spend of the zero-shot sweep in the shared ledger counts against the total cap
    (tmp_path / "spend.jsonl").write_text(json.dumps({"time": "t", "run": "zero-shot", "request_hash": "h", "model": SNAPSHOT,
                                                       "provider": PROVIDER, "input_tokens": 1, "cost": 249.999999, "id": "g"}) + "\n")
    with pytest.raises(BudgetExceeded, match="total cap"):
        luna(tmp_path).decide({"text": "x"}, {"q": NOUL})
    assert server.bodies == [] and not list((tmp_path / "cache").rglob("*.json"))


def test_rate_limit_waits_for_retry_after(tmp_path, server) -> None:
    server.script = [(429, {"Retry-After": "7"}, {"error": {"message": "rate limited"}})]
    sleeps: list[float] = []
    d = luna(tmp_path, sleeps=sleeps).decide({"text": "x"}, {"q": NOUL})
    assert d.answers["q"] == answer(NOUL) and len(server.bodies) == 2 and 7.0 in sleeps
    assert len(ledger_lines(tmp_path)) == 1


def test_rejection_is_cached_and_never_resent(tmp_path, server) -> None:
    server.script = [(413, {}, "request too large")]
    client = luna(tmp_path)
    for _ in range(2):
        with pytest.raises(LunaRejected) as err:
            client.decide({"text": "x"}, {"q": NOUL})
        assert type(err.value).__name__ == "LunaRejected" and "HTTP 413" in str(err.value)
    assert len(server.bodies) == 1 and client.rejected_requests == 2 and client.usage.fresh_calls == 1
    assert (tmp_path / "spend.jsonl").read_text() == ""
    with pytest.raises(LunaRejected):  # a new process reads the cached rejection
        luna(tmp_path).decide({"text": "x"}, {"q": NOUL})
    assert len(server.bodies) == 1


REFUSAL = (502, {}, {"error": {"message": 'OpenAI refused to answer question "tone"', "code": 502}})


def test_whole_request_refusal_is_retried_then_answers_are_used(tmp_path, server) -> None:
    server.script = [REFUSAL, REFUSAL]
    client = luna(tmp_path)
    d = client.decide({"text": "x"}, {"tone": CHOICE, "refund": NOUL})
    assert len(server.bodies) == 3 and d.answers == {"tone": answer(CHOICE), "refund": answer(NOUL)} and not d.cached
    assert (client.refused_requests, client.refused_answers) == (0, 0) and len(ledger_lines(tmp_path)) == 1


def test_persistent_refusal_gives_uniform_answers_cached_and_unbilled(tmp_path, server) -> None:
    server.script = [REFUSAL, REFUSAL, REFUSAL]
    client = luna(tmp_path)
    questions = {"tone": CHOICE, "refund": NOUL}
    first = client.decide({"text": "x"}, questions)
    assert len(server.bodies) == 3
    assert first.answers["refund"] == {"type": "noul", "noul": 0.5} and first.answers["tone"]["confidence"] == 0.0
    assert first.snapshot == "refused" and (first.input_tokens, first.cost) == (0, 0.0)
    second = luna(tmp_path).decide({"text": "x"}, questions)  # new process: cached refusal, not re-sent
    assert second.answers == first.answers and second.cached and len(server.bodies) == 3
    assert (client.refused_requests, client.refused_answers, client.answers_seen, client.usage.fresh_calls) == (1, 2, 2, 1)
    assert (tmp_path / "spend.jsonl").read_text() == ""
    [cached] = list((tmp_path / "cache").rglob("*.json"))
    assert json.loads(cached.read_text())["refused_request"]["question"] == "tone"


def test_off_pin_snapshot_stops_is_billed_and_not_cached(tmp_path, server) -> None:
    server.model = "openai/gpt-6-luna-decisions-20261101"
    client = luna(tmp_path)
    for n in (1, 2):
        with pytest.raises(SnapshotMismatch):
            client.decide({"text": "x"}, {"q": NOUL})
        assert len(server.bodies) == n
    assert len(ledger_lines(tmp_path)) == 2
    quarantined = list((tmp_path / "cache" / "offpin").glob("*.json"))
    assert len(quarantined) == 1 and json.loads(quarantined[0].read_text())["response"]["model"] == server.model
    assert [p for p in (tmp_path / "cache").rglob("*.json") if "offpin" not in p.parts] == []


def test_framing_features_patch_does_not_disable_luna(tmp_path, server, monkeypatch) -> None:
    def no_network(self, body):  # what framing_features installs on JevClient
        raise RuntimeError("analysis must be served from the Jev cache")

    monkeypatch.setattr(JevClient, "_post", no_network)
    d = luna(tmp_path).decide({"text": "x"}, {"q": NOUL})
    assert not d.cached and len(server.bodies) == 1


def test_exact_requests_only_no_answer_assembly(tmp_path, server) -> None:
    client = luna(tmp_path)
    client.max_questions = 1  # unused: no chunking
    state = {"text": "x"}
    client.decide(state, {"a": NOUL, "b": CHOICE})
    client.decide(state, {"a": NOUL})
    assert [list(b["questions"]) for b in server.bodies] == [["a", "b"], ["a"]]
    assert not (tmp_path / "cache" / "q").exists()


def test_make_client_luna_ignores_placeholder_key(tmp_path, server, monkeypatch) -> None:
    monkeypatch.setenv("JEV_OPT_BACKEND", "luna")
    monkeypatch.setenv("JEV_OPT_LUNA_SPEND", str(tmp_path / "spend.jsonl"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-cache-only")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=from-file\n")
    client = make_client(tmp_path / "jev-cache", "typesafe/jev-1.13", 1800, 4, [tmp_path])
    assert isinstance(client, LunaClient) and client.model == MODEL and client._api_key == "from-file"
    assert client.cache_dir == tmp_path / "jev-cache" / "luna" and client.concurrency == 16 and client.url == server.url


def test_empty_200_body_is_retried(tmp_path, server) -> None:
    server.script = [(200, {}, "")]
    d = luna(tmp_path).decide({"text": "x"}, {"q": NOUL})
    assert d.answers["q"] == answer(NOUL) and len(server.bodies) == 2 and len(ledger_lines(tmp_path)) == 1


def test_make_client_msd1_pins_microsoft_decision_1_with_its_own_cache_and_ledger(tmp_path, server, monkeypatch) -> None:
    from jev_opt.luna import MSD1

    monkeypatch.setenv("JEV_OPT_BACKEND", "msd1")
    monkeypatch.setenv("JEV_OPT_MSD1_SPEND", str(tmp_path / "msd1_spend.jsonl"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "real-key")
    client = make_client(tmp_path / "jev-cache", "typesafe/jev-1.13", 1800, 4, [tmp_path])
    assert isinstance(client, LunaClient) and client.model == "microsoft/microsoft-decision-1"
    assert client.cache_dir == tmp_path / "jev-cache" / "msd1" and client.spend.path == tmp_path / "msd1_spend.jsonl"
    assert (client.spend.total_cap, client.spend.run_cap) == (150.0, 20.0)
    with pytest.raises(SnapshotMismatch, match="MSD1 settings"):  # Luna's snapshot answering
        client.decide({"text": "x"}, {"q": NOUL})
    server.model, server.provider = MSD1.snapshot, "Azure"
    server.script = [(502, {"Retry-After": "0"}, {"error": {"message": 'Azure refused to answer question "q"', "code": 502}})] * 3
    d = client.decide({"text": "y"}, {"q": NOUL})
    assert d.snapshot == "refused" and d.answers["q"] == {"type": "noul", "noul": 0.5} and client.refused_requests == 1
    d = client.decide({"text": "z"}, {"q": NOUL})
    assert d.snapshot == MSD1.snapshot and server.bodies[-1]["model"] == "microsoft/microsoft-decision-1"
    assert [r["model"] for r in json.loads("[" + ",".join((tmp_path / "msd1_spend.jsonl").read_text().splitlines()) + "]")] == [
        SNAPSHOT, MSD1.snapshot]  # off-pin response billed and logged, refusal not billed
