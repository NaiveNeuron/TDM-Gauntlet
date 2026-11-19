"""Clef's base model through option log-probabilities, for the jev-opt executor (drop-in for `JevClient`).

Selected by `JEV_OPT_BACKEND=qwen-base` (`jev.make_client`; H100 round 2, Experiment 1).

- Every question of a request is answered on its own. The scoring unit is (state, question specification); the
  question id is not rendered and not keyed. `render_unit` builds the fixed prompt (template
  `qwen-base-logprob-v1`) and the options' one-token continuations: Choice letters A-Z in the program's option
  order, Noul Yes/No, Score numbers 1-9.
- Inference runs in the repository's gauntlet environment (`uv run --extra clef python -m
  gauntlet.logprob_serve`): one forward pass per unit, FP32 log-softmax of each continuation after the prompt,
  probabilities = softmax over the options' log-likelihoods. This package needs no torch.
- Cache: one JSON file per unit under `cache_dir`, keyed by SHA-256 of (pinned model, template id and SHA-256,
  scoring settings, guard, unit in key order). Device, GPU and library versions are recorded with each response
  (`configurations/<sha>.json`), not keyed. Without a device the client is cache-only and a miss raises
  `LogprobCacheMiss`. Window gathering, server handshake checks and the cache files are `ClefBackend`'s.
- Answers have the release's public SystemOne format and four-decimal rounding, as Clef's; the server's raw
  probabilities, full-vocabulary option log-probabilities and option mass stay in each cached response.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .clef import LINGER_S, REPO, WINDOW, ClefBackend, ClefServer, ClefServerError, Server
from .jev import Decision

PINS = {"qwen-base": "Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"}
# Server-side settings that determine an answer; identical to gauntlet.logprob_serve.SCORING (checked at handshake).
SCORING = {
    "backend": "qwen-base-logprob",
    "dtype": "bfloat16",
    "logit_dtype": "float32",
    "attention_implementation": "sdpa",
    "padding": "right",
    "option_scoring": "single-token continuation: FP32 log-softmax over the vocabulary at the position after the prompt; "
                      "probabilities = FP32 softmax over the options' log-likelihoods",
    "truncation_policy": "reject_complete_input_overflow_before_forward",
}
TEMPLATE_ID = "qwen-base-logprob-v1"
TEMPLATE: dict[str, Any] = {
    "system": "Read the state and answer the question about it. Reply with exactly one of the allowed answers and nothing else.",
    "prefix": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\nSTATE:\n",
    "question": "\n\nQUESTION:\n",
    "block": "\n\n",
    "suffix": "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "choice_header": "ALLOWED ANSWERS:\n",
    "choice_line": "{letter}. {key}: {criterion}\n",
    "choice_line_bare": "{letter}. {key}\n",
    "choice_reply": "Reply with the letter of one allowed answer.",
    "letters": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "noul_header": "ALLOWED ANSWERS:\n",
    "noul_lines": {"true": "Yes: {criterion}\n", "false": "No: {criterion}\n"},
    "noul_reply": "Reply with Yes or No.",
    "noul_answers": ["Yes", "No"],
    "score_header": "LEVELS (lowest to highest):\n",
    "score_line": "{number}. {level}\n",
    "score_reply": "Reply with the number of one level.",
    "numbers": "123456789",
}
TEMPLATE_SHA256 = hashlib.sha256(json.dumps(TEMPLATE, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
# Every continuation the template can ask for (the server checks each is one token after the prompt).
ALPHABET = [*TEMPLATE["letters"], *TEMPLATE["numbers"], *TEMPLATE["noul_answers"]]


class LogprobCacheMiss(RuntimeError):
    """A unit is not in the cache and the client has no device (cache-only replay)."""


class LogprobRejected(RuntimeError):
    """A unit outside the template (e.g. > 26 options) or a prompt over the guard; nothing was truncated."""


class TemplateError(ValueError):
    """The question cannot be rendered by the fixed template."""


def render(value: Any) -> str:
    """Clef's encoder rendering: strings as they are, everything else as compact JSON with sorted keys."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def render_unit(state: Any, question: dict[str, Any]) -> tuple[str, list[str]]:
    """(prompt, continuations) of one unit; continuation i is option i (Choice: option order, Noul: true then false,
    Score: level index order)."""
    T = TEMPLATE
    kind = question.get("type")
    instructions = question.get("instructions")
    if instructions is None or instructions == "":
        raise TemplateError("empty instructions")
    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise TemplateError("choice criteria must be a non-empty mapping")
        if len(criteria) > len(T["letters"]):
            raise TemplateError(f"{len(criteria)} options exceed the {len(T['letters'])} letters")
        lines = []
        for letter, (key, criterion) in zip(T["letters"], criteria.items()):
            if criterion is None or criterion == "":
                lines.append(T["choice_line_bare"].format(letter=letter, key=key))
            else:
                lines.append(T["choice_line"].format(letter=letter, key=key, criterion=render(criterion)))
        block = T["choice_header"] + "".join(lines) + T["choice_reply"]
        continuations = list(T["letters"][: len(criteria)])
    elif kind == "noul":
        if criteria:
            if not isinstance(criteria, dict) or set(criteria) - {"true", "false"}:
                raise TemplateError("noul criteria must map true/false")
            block = T["noul_header"] + "".join(T["noul_lines"][k].format(criterion=render(criteria[k])) for k in ("true", "false") if k in criteria)
            block += T["noul_reply"]
        else:
            block = T["noul_reply"]
        continuations = list(T["noul_answers"])
    elif kind == "score":
        if not isinstance(criteria, list) or not criteria:
            raise TemplateError("score criteria must be a non-empty list of levels")
        if len(criteria) > len(T["numbers"]):
            raise TemplateError(f"{len(criteria)} levels exceed the {len(T['numbers'])} one-digit numbers")
        block = T["score_header"] + "".join(T["score_line"].format(number=n, level=render(c)) for n, c in zip(T["numbers"], criteria))
        block += T["score_reply"]
        continuations = list(T["numbers"][: len(criteria)])
    else:
        raise TemplateError(f"unknown question type {kind!r}")
    prompt = (T["prefix"].format(system=T["system"]) + render(state) + T["question"] + render(instructions) + T["block"]
              + block + T["suffix"])
    return prompt, continuations


def unit_body(model: str, state: Any, question: dict[str, Any]) -> dict[str, Any]:
    return {"model": model, "template": TEMPLATE_ID, "state": state, "question": question}


def cache_key(model: str, max_length: int, unit: dict[str, Any]) -> str:
    """Unit hash: pinned model, template id and SHA-256, scoring settings, guard and the unit in key order."""
    canonical = json.dumps({"model": model, "template": TEMPLATE_ID, "template_sha256": TEMPLATE_SHA256, "scoring": SCORING,
                            "max_length": max_length, "unit": {"state": unit["state"], "question": unit["question"]}},
                           ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def answer(question: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """The public SystemOne answer from the server's option probabilities (Clef's `systemone_answer` format)."""
    p = [float(x) for x in response["probabilities"]]
    kind = question["type"]
    if kind == "noul":
        return {"type": "noul", "noul": round(p[0], 4)}
    if kind == "choice":
        keys = [str(k) for k in question["criteria"]]
        probs = dict(zip(keys, p, strict=True))
        choice = max(keys, key=probs.__getitem__)
        return {"type": "choice", "choice": choice, "confidence": round(probs[choice], 4),
                "probabilities": {k: round(v, 4) for k, v in probs.items()}}
    levels = [str(i) for i in range(len(question["criteria"]))]
    return {"type": "score", "score": round(sum(i * x for i, x in enumerate(p)), 4), "confidence": round(max(p), 4),
            "legend": dict(zip(levels, question["criteria"])), "probabilities": {lv: round(x, 4) for lv, x in zip(levels, p, strict=True)}}


def server_command(backend: str, device: str, max_length: int) -> list[str]:
    """The scoring server in the repository's `clef` environment (pinned torch/transformers/kernels)."""
    return ["uv", "run", "--project", str(REPO), "--extra", "clef", "python", "-m", "gauntlet.logprob_serve",
            "--model", backend, "--device", device, "--max-length", str(max_length)]


class LogprobBackend(ClefBackend):
    """Base-model client with the `JevClient` interface used by `execute` (decide, concurrency, usage, ledger).
    Usage counts units: `calls` / `fresh_calls` are scored units, tokens are prompt tokens."""

    pins = PINS
    encoding = SCORING
    rejected_error = LogprobRejected
    cache_miss_error = LogprobCacheMiss

    def __init__(self, backend: str, cache_dir: Path, device: str | None = None, max_length: int = 32000,
                 window: int = WINDOW, linger_s: float = LINGER_S, server_factory: Any = None):
        super().__init__(backend, cache_dir, device, max_length, window, linger_s, server_factory)

    def _default_server(self, device: str) -> Server:
        return ClefServer(self.backend, device, self.max_length, self.cache_dir / "scratch",
                          command=server_command(self.backend, device, self.max_length))

    def key(self, body: dict[str, Any]) -> str:
        return cache_key(self.model, self.max_length, body)

    def decide(self, state: Any, questions: dict[str, Any], context: dict[str, Any] | None = None) -> Decision:
        units = []
        for qid, spec in questions.items():
            try:
                prompt, continuations = render_unit(state, spec)
            except TemplateError as exc:
                raise LogprobRejected(f"TemplateError: {qid}: {exc}") from None
            body = unit_body(self.model, state, spec)
            units.append((qid, spec, body, self.key(body), {"prompt": prompt, "continuations": continuations}))
        records: dict[str, dict[str, Any]] = {}
        waiting = []
        for _, _, body, digest, payload in units:
            if digest in records or any(d == digest for d, _, _ in waiting):
                continue  # the same specification under two ids is one unit
            path = self._path(digest)
            if path.exists():
                records[digest] = json.loads(path.read_text())
            elif self._factory is None:
                raise LogprobCacheMiss(f"{self.backend}: unit {digest[:16]} is not cached and no device is configured")
            else:
                waiting.append((digest, *self._enqueue(digest, body, payload)))
        fresh = set()
        for digest, future, owner in waiting:
            records[digest] = future.result()
            if owner:
                fresh.add(digest)
        answers: dict[str, Any] = {}
        for qid, spec, body, digest, payload in units:
            response = records[digest]["response"]
            if len(response["probabilities"]) != len(payload["continuations"]):
                raise ClefServerError(f"unit {digest[:16]}: {len(response['probabilities'])} probabilities for "
                                      f"{len(payload['continuations'])} options")
            answers[qid] = answer(spec, response)
            if self.ledger is not None:
                self.ledger.record(digest, body, records[digest], context)
        tokens = seconds = 0.0
        with self._usage_lock:
            for digest, record in records.items():
                response = record["response"]
                n = int((response.get("usage") or {}).get("input_tokens") or 0)
                s = float((response.get("audit") or {}).get("inference_seconds") or 0.0)
                tokens, seconds = tokens + n, seconds + s
                self.usage.calls += 1
                self.usage.input_tokens += n
                self.snapshots.add(str(response.get("model", self.model)))
                if digest in fresh:
                    self.usage.fresh_calls += 1
                    self.usage.fresh_input_tokens += n
                    self.fresh_inference_seconds += s
        return Decision(answers=answers, snapshot=self.model, input_tokens=int(tokens), cost=0.0, latency_s=seconds, cached=not fresh)


def make_logprob_client(backend: str) -> LogprobBackend:
    """Client from the environment: `JEV_OPT_DEVICE` (absent: cache-only), `JEV_OPT_LOGPROB_CACHE` (default
    `data/cache/<backend>-jev-opt` in the repository), `JEV_OPT_LOGPROB_MAX_LENGTH` (default 32000)."""
    cache = Path(os.environ.get("JEV_OPT_LOGPROB_CACHE") or REPO / "data" / "cache" / f"{backend}-jev-opt")
    return LogprobBackend(backend, cache, device=os.environ.get("JEV_OPT_DEVICE") or None,
                          max_length=int(os.environ.get("JEV_OPT_LOGPROB_MAX_LENGTH") or 32000))
