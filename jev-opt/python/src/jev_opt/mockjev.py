"""Offline Decisions-API simulator (tests, `JEV_OPT_MOCK_JEV=1` demos). Never used for real runs.

Honors the real response shape. An option's evidence is the keyword overlap between its
criterion text and the state, so better-worded criteria measurably raise accuracy.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import httpx

STOP = frozenset(
    "the and for are but not you your with this that from have has was were will would can could "
    "should into about than then them they their there these those also just "
    "only very more most some such any all each other one two its it's our out off over under "
    "state text message label classify exactly following given".split()
)


def tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in STOP}


def _state_text(state: Any) -> str:
    if isinstance(state, dict):
        return " ".join(str(v) for v in state.values())
    return str(state)


def _softmax(scores: dict[str, float], beta: float = 1.5) -> dict[str, float]:
    m = max(scores.values())
    exp = {k: math.exp(beta * (v - m)) for k, v in scores.items()}
    z = sum(exp.values())
    return {k: round(v / z, 4) for k, v in exp.items()}


def answer(state: Any, question: dict[str, Any]) -> dict[str, Any]:
    words = tokens(_state_text(state))
    kind = question["type"]
    if kind == "choice":
        probs = _softmax({o: float(len(words & tokens(d))) for o, d in question["criteria"].items()})
        best = max(probs, key=lambda o: probs[o])
        return {"type": "choice", "choice": best, "probabilities": probs, "confidence": probs[best]}
    if kind == "noul":
        # Only the question's last paragraph is the claim; shared preambles carry no evidence.
        claim = question["instructions"].strip().split("\n\n")[-1]
        overlap = len(words & tokens(claim))
        return {"type": "noul", "noul": round(1 / (1 + math.exp(-(1.5 * overlap - 1.0))), 4)}
    if kind == "score":
        levels = question["criteria"]
        probs = _softmax({str(i): float(len(words & tokens(d))) for i, d in enumerate(levels)})
        return {
            "type": "score",
            "score": round(sum(int(i) * p for i, p in probs.items()), 4),
            "probabilities": probs,
            "legend": {str(i): d for i, d in enumerate(levels)},
        }
    raise ValueError(f"unknown question type {kind!r}")


def respond(body: dict[str, Any]) -> dict[str, Any]:
    tokens_in = len(json.dumps(body)) // 4
    return {
        "model": f"{body['model']}-mock",
        "answers": {qid: answer(body["state"], q) for qid, q in body["questions"].items()},
        "usage": {"input_tokens": tokens_in, "output_tokens": 8, "cost": round(tokens_in * 4e-8, 10)},
        "provider": "mock",
    }


def transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=respond(json.loads(request.content)))

    return httpx.MockTransport(handler)
