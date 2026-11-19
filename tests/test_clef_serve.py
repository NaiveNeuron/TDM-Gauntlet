"""jev-opt's Clef server: window answers, per-body encoder rejections, handshake (no hardware)."""

import io
import json

from gauntlet.clef import MODEL
from gauntlet.clef_serve import answer, serve

from test_clef_multigpu import _ControlledClient, _probabilities


def _body(identifier, tokens=10):
    return {"model": MODEL, "state": {"id": identifier, "input_tokens": tokens},
            "questions": {"q": {"type": "choice", "instructions": "pick", "criteria": {"a": "A", "b": "B"}}}}


def test_window_answers_every_body_in_order_and_isolates_rejections(tmp_path):
    client = _ControlledClient(tmp_path, device="cuda:0", max_length=100)
    bodies = [("k0", _body(0)), ("k1", _body(1, tokens=500)), ("k2", _body(2))]
    out = answer(client, bodies)
    assert [r["key"] for r in out] == ["k0", "k1", "k2"]
    assert out[1]["type"] == "ClefContextError" and "truncation forbidden" in out[1]["error"]
    for result, (_, body) in zip((out[0], out[2]), (bodies[0], bodies[2])):
        assert result["response"]["audit"]["raw_probabilities"] == _probabilities(body)
    assert not any(tmp_path.rglob("*.json")), "the server keeps no cache"


def test_serve_handshake_then_one_result_line_per_window(tmp_path):
    client = _ControlledClient(tmp_path, device="cuda:0", max_length=100)
    inp = io.StringIO(json.dumps({"bodies": [["a", _body(3)], ["b", _body(4)]]}) + "\n\n")
    out = io.StringIO()
    serve(client, inp, out)
    hello, reply = [json.loads(line) for line in out.getvalue().splitlines()]
    assert hello["ready"] and hello["model"] == MODEL and hello["configuration"]["max_length"] == 100
    assert [r["key"] for r in reply["results"]] == ["a", "b"] and all("response" in r for r in reply["results"])
