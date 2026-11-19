"""Decision-model backends for the jev-opt executor: GPT-6 Luna (RQ3 part of its pre-registered study) and
Microsoft-Decision-1. A `Spec` fixes the model id, pinned snapshot, provider, price, caps
and spend ledger; `LUNA` is the default.

`JevClient` with the model's id on OpenRouter's Decisions endpoint (same request format and key as Jev). The
executor's requests are sent unchanged and cached as exact requests only: no per-question cache tier and no
question chunking (whether these models answer co-questions independently is not documented). On top of `JevClient`:

- snapshot/provider pin: an off-pin response stops the run; it is billed and logged in the spend ledger but
  quarantined (`<cache>/offpin/`), never cached;
- refused or invalid answers are replaced, on read, by the uniform no-information answer and counted (the cache
  keeps the verbatim response, so the substitution is reproducible and never re-billed);
- rejected requests (HTTP 400/413/422) are cached as rejections and raise `LunaRejected` (the driver scores the
  example as failed); a cached rejection is never re-sent;
- a hard budget: the model's file-locked spend ledger `runs/<model>/spend.jsonl`, shared with `gauntlet.luna`
  (same line format and locking), is checked before every uncached request against the total cap and the run
  label's cap.

The answer rules and the ledger mirror `gauntlet.luna` (this package cannot import the gauntlet).
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import httpx

from .jev import RETRY_STATUS, Decision, JevClient, JevError, _backoff, _retry_delay, cache_key, load_api_key

REJECT_STATUS = {400, 413, 422}
REJECTED_MODEL = "rejected"  # `model` of a rejection as recorded in a run's response ledger
# OpenRouter reports a refusal as HTTP 502 for the whole request; tried REFUSAL_ATTEMPTS times in all, then a
# refused request (uniform answers, cached, not billed; rule added on 2026-10-09, before any Luna score was computed)
REFUSAL_ATTEMPTS = 3
REFUSED_MODEL = "refused"  # `model` of a refused request's response view
RUN = "rq3"
OFFLINE_KEY = "offline-cache-only"  # framing_features' placeholder key: never a real key
REPO = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class Spec:
    """One decision model on the Decisions endpoint and its pre-registered pin, price and budget
    (`gauntlet.luna.Spec`)."""

    name: str  # JEV_OPT_BACKEND value, cache subdirectory, runs/<name>
    label: str
    model: str
    snapshot: str
    provider: str
    price: float  # $ per input token; no output charge
    total_cap: float
    run_caps: dict[str, float]
    protocol: str
    refusal_re: re.Pattern[str]

    @property
    def spend_ledger(self) -> Path:
        return REPO / "runs" / self.name / "spend.jsonl"


LUNA = Spec("luna", "Luna", "openai/gpt-6-luna-decisions", "openai/gpt-6-luna-decisions-20261006", "OpenAI", 1e-7,
            250.0, {"zero-shot": 235.0, "rq3": 35.0}, "Luna settings",
            re.compile(r'^OpenAI refused to answer question "(.*)"$', re.S))
MSD1 = Spec("msd1", "MSD1", "microsoft/microsoft-decision-1", "microsoft/microsoft-decision-1-20261009", "Azure", 0.042e-6,
            150.0, {"zero-shot": 130.0, "rq3": 20.0}, "MSD1 settings",
            re.compile(r'^.+ refused to answer question "(.*)"$', re.S))
SPECS = {spec.name: spec for spec in (LUNA, MSD1)}

# Luna's values, read by the Luna analyses.
MODEL, SNAPSHOT, PROVIDER, PRICE = LUNA.model, LUNA.snapshot, LUNA.provider, LUNA.price
TOTAL_CAP_USD, RUN_CAPS_USD, SPEND_LEDGER = LUNA.total_cap, LUNA.run_caps, LUNA.spend_ledger


class BudgetExceeded(RuntimeError):
    pass


class SnapshotMismatch(RuntimeError):
    pass


class LunaRejected(RuntimeError):
    """The endpoint rejected the request itself (HTTP 400/413/422); never truncated, split or re-sent."""


class LunaCacheMiss(RuntimeError):
    """A request is not cached and the client has no key (cache-only replay)."""


def estimate_cost(body: dict[str, Any], price: float = LUNA.price) -> float:
    """Conservative pre-request estimate: UTF-8 bytes / 2 tokens at the input price."""
    return len(json.dumps(body, ensure_ascii=False).encode("utf-8")) / 2 * price


# ------------------------------------------------------------------------------------------------ answers


def _criteria_keys(question: dict[str, Any]) -> list[str]:
    criteria = question.get("criteria")
    if isinstance(criteria, dict):
        return [str(k) for k in criteria]
    if isinstance(criteria, list):
        return [str(i) for i in range(len(criteria))]
    return []


def uniform_answer(question: dict[str, Any]) -> dict[str, Any]:
    """The no-information answer that replaces a refused or invalid one."""
    kind = question.get("type")
    if kind == "noul":
        return {"type": "noul", "noul": 0.5}
    keys = _criteria_keys(question)
    probs = {k: 1.0 / len(keys) for k in keys}
    if kind == "choice":
        return {"type": "choice", "choice": keys[0], "probabilities": probs, "confidence": 0.0}
    if kind == "score":
        legend = {str(i): c for i, c in enumerate(question["criteria"])}
        return {"type": "score", "score": (len(keys) - 1) / 2, "legend": legend, "probabilities": probs, "confidence": 0.0}
    raise ValueError(f"unknown question type {kind!r}")


def _number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def valid_answer(question: dict[str, Any], answer: Any) -> bool:
    if not isinstance(answer, dict) or answer.get("type") != question.get("type"):
        return False
    if question["type"] == "noul":
        return _number(answer.get("noul"))
    probs = answer.get("probabilities")
    if not isinstance(probs, dict) or not probs or not all(_number(v) for v in probs.values()):
        return False
    return set(probs) <= set(_criteria_keys(question))


def saturated(answer: Any) -> bool:
    """Largest probability exactly 1 (choice, score) or noul exactly 0 or 1."""
    if not isinstance(answer, dict):
        return False
    if answer.get("type") == "noul":
        return answer.get("noul") in (0, 1)
    probs = answer.get("probabilities")
    return isinstance(probs, dict) and bool(probs) and max(probs.values()) == 1


def normalize(questions: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """Response with refused/invalid answers replaced by the uniform answer, listed in `refused`."""
    answers = response.get("answers") if isinstance(response.get("answers"), dict) else {}
    out = dict(response)
    out["answers"], refused, raw = {}, [], {}
    for qid, question in questions.items():
        answer = answers.get(qid)
        if valid_answer(question, answer):
            out["answers"][qid] = answer
        else:
            out["answers"][qid] = uniform_answer(question)
            refused.append(qid)
            raw[qid] = answer
    if refused:
        out["refused"], out["raw_answers"] = refused, raw
    return out


def check_pin(response: dict[str, Any], spec: Spec = LUNA) -> None:
    if response.get("model") != spec.snapshot or response.get("provider") != spec.provider:
        raise SnapshotMismatch(f"response from {response.get('model')!r} / {response.get('provider')!r}, "
                               f"pinned {spec.snapshot!r} / {spec.provider!r}; stopped ({spec.protocol})")


# ------------------------------------------------------------------------------------------------ budget


class SpendLedger:
    """Billed responses of one model, one JSON line each, shared by processes through an exclusive file lock
    (`gauntlet.luna.SpendLedger`: same file, line format and locking).

    `reserve` checks spend so far + this process's in-flight estimates + the request's estimate against the
    total cap and the run label's cap; `settle` replaces the reservation by the billed cost.
    """

    def __init__(self, path: Path | None = None, run: str = RUN, total_cap: float | None = None,
                 run_cap: float | None = None, spec: Spec = LUNA):
        self.path = Path(path) if path is not None else spec.spend_ledger
        self.run = run
        self.label = spec.label
        self.total_cap = spec.total_cap if total_cap is None else total_cap
        self.run_cap = spec.run_caps.get(run, self.total_cap) if run_cap is None else run_cap
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = threading.Lock()
        self._offset = 0
        self.total = 0.0
        self.run_total = 0.0
        self.inflight = 0.0

    def _refresh(self, fh: Any) -> None:
        fh.seek(self._offset)
        data = fh.read()
        end = data.rfind("\n") + 1
        for line in data[:end].splitlines():
            if line.strip():
                row = json.loads(line)
                self.total += float(row["cost"])
                self.run_total += float(row["cost"]) if row.get("run") == self.run else 0.0
        self._offset += len(data[:end].encode("utf-8"))

    def _locked(self, fn: Any) -> Any:
        with self._lock, open(self.path, "r+", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                self._refresh(fh)
                return fn(fh)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def reserve(self, estimate: float) -> None:
        def check(_: Any) -> None:
            if self.total + self.inflight + estimate > self.total_cap:
                raise BudgetExceeded(f"{self.label} total cap ${self.total_cap:.2f}: spent ${self.total:.4f}, in flight "
                                     f"${self.inflight:.4f}, next request ≤ ${estimate:.4f}")
            if self.run_total + self.inflight + estimate > self.run_cap:
                raise BudgetExceeded(f"{self.label} {self.run} cap ${self.run_cap:.2f}: spent ${self.run_total:.4f}, in flight "
                                     f"${self.inflight:.4f}, next request ≤ ${estimate:.4f}")
            self.inflight += estimate
        self._locked(check)

    def release(self, estimate: float) -> None:
        with self._lock:
            self.inflight -= estimate

    def settle(self, estimate: float, row: dict[str, Any]) -> None:
        line = json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "run": self.run, **row}, ensure_ascii=False)

        def append(fh: Any) -> None:
            fh.seek(0, os.SEEK_END)
            fh.write(line + "\n")
            fh.flush()
            self.inflight -= estimate
            self.total += float(row["cost"])
            self.run_total += float(row["cost"])
            self._offset = fh.tell()
        self._locked(append)


# ------------------------------------------------------------------------------------------------ client


class _Rejection(LunaRejected):
    """A fresh rejection, carrying the cached rejection record to `decide`."""

    def __init__(self, record: dict[str, Any]):
        super().__init__(_rejection_message(record))
        self.record = record


class _Refusal(RuntimeError):
    """A fresh refused request, carrying the cached refusal record to `decide` (never escapes `decide`)."""

    def __init__(self, record: dict[str, Any]):
        super().__init__(record["refused_request"]["question"])
        self.record = record


def _rejection_message(record: dict[str, Any]) -> str:
    rej = record["rejected"]
    return f"request {record['request_hash'][:16]} rejected with HTTP {rej['status']}: {str(rej['payload'])[:300]}"


def refused_question(resp: httpx.Response, spec: Spec = LUNA) -> str | None:
    """The question named by OpenRouter's whole-request refusal (HTTP 502), else None."""
    if resp.status_code != 502:
        return None
    try:
        message = resp.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return None
    match = spec.refusal_re.match(message) if isinstance(message, str) else None
    return match.group(1) if match else None


def _json_or_none(resp: httpx.Response) -> Any:
    """A 200 body; None when it is not JSON (OpenRouter returned empty 200 bodies on 2026-10-09), retried."""
    try:
        return resp.json()
    except ValueError:
        return None


def refused_response(questions: dict[str, Any], refusal: dict[str, Any]) -> dict[str, Any]:
    """Response view of a refused request: every question gets the uniform answer and counts as refused."""
    return {"model": REFUSED_MODEL, "answers": {q: uniform_answer(spec) for q, spec in questions.items()},
            "usage": {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}, "refused": list(questions), "refused_request": refusal}


class LunaClient(JevClient):
    """Exact-request client of a decision model (`spec`, default Luna; model id `spec.model`) with the `JevClient`
    interface used by `execute` (decide, concurrency, usage,
    ledger). Answer counts for the driver: `answers_seen`, `refused_answers`, `saturated_answers` (over answered,
    not refused questions), `refused_requests` (whole-request refusals: all their answers refused) and
    `rejected_requests`, per `decide` call (cache hits included, like `usage.calls`)."""

    def __init__(self, cache_dir: Path, api_key: str | None, spend: SpendLedger, spec: Spec = LUNA, max_rpm: int = 600,
                 concurrency: int = 16, **kw: Any):
        super().__init__(cache_dir, api_key, spec.model, max_rpm=max_rpm, concurrency=concurrency, **kw)
        self.spec = spec
        self.spend = spend
        self.answers_seen = 0
        self.refused_answers = 0
        self.saturated_answers = 0
        self.rejected_requests = 0
        self.refused_requests = 0

    def _index(self, digest: str, body: dict[str, Any]) -> None:
        """Exact requests only: no per-question pointers."""

    def decide(self, state: Any, questions: dict[str, Any], context: dict[str, Any] | None = None) -> Decision:
        """This exact request from the cache, else one API call; answers normalized (refusals → uniform answer)."""
        try:
            digest, body, record, fresh = self._fetch(state, questions)
        except _Rejection as exc:  # sent now and rejected (not billed)
            self._rejected({"model": self.model, "state": state, "questions": questions}, exc.record, context, fresh=True)
        except _Refusal as exc:  # sent now and refused REFUSAL_ATTEMPTS times (not billed)
            digest, body, record, fresh = exc.record["request_hash"], {"model": self.model, "state": state, "questions": questions}, exc.record, True
        if "rejected" in record:
            self._rejected(body, record, context, fresh=False)
        refused_request = "refused_request" in record
        if refused_request:
            norm = response = refused_response(questions, record["refused_request"])
            if self.ledger is not None:
                self.ledger.record(digest, body, {"response": response}, context)
        else:
            response = record["response"]
            check_pin(response, self.spec)
            norm = normalize(questions, response)
            if self.ledger is not None:
                self.ledger.record(digest, body, record, context)
        usage = response.get("usage") or {}
        tokens, cost = int(usage.get("input_tokens") or 0), float(usage.get("cost") or 0.0)
        refused = norm.get("refused", [])
        with self._usage_lock:
            self.usage.calls += 1
            self.usage.input_tokens += tokens
            if not refused_request:
                self.snapshots.add(str(response.get("model")))
            if fresh:
                self.usage.fresh_calls += 1
                self.usage.fresh_cost += cost
                self.usage.fresh_input_tokens += tokens
            self.answers_seen += len(questions)
            self.refused_answers += len(refused)
            self.refused_requests += refused_request
            self.saturated_answers += sum(saturated(a) for q, a in norm["answers"].items() if q not in refused)
        return Decision(answers=norm["answers"], snapshot=str(response.get("model")), input_tokens=tokens, cost=cost,
                        latency_s=float(record.get("latency_s") or 0.0), cached=not fresh)

    def _rejected(self, body: dict[str, Any], record: dict[str, Any], context: dict[str, Any] | None, fresh: bool) -> NoReturn:
        if self.ledger is not None:
            self.ledger.record(record["request_hash"], body, {"response": {"model": REJECTED_MODEL, "rejected": record["rejected"]}}, context)
        with self._usage_lock:
            self.usage.calls += 1
            self.usage.fresh_calls += fresh
            self.rejected_requests += 1
        raise LunaRejected(_rejection_message(record))

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """One billed request (called by `_fetch` on a cache miss; overrides any patched `JevClient._post`):
        budget reservation, pin check, rejection or refusal record, verbatim payload."""
        digest = cache_key(self.url, self.model, body["state"], body["questions"])
        if not self._api_key:
            raise LunaCacheMiss(f"request {digest[:16]} is not cached and no OPENROUTER_API_KEY is configured (cache-only)")
        estimate = estimate_cost(body, self.spec.price)
        self.spend.reserve(estimate)
        try:
            status, payload, question = self._send(body)
        except BaseException:
            self.spend.release(estimate)
            raise
        if status != 200:
            self.spend.release(estimate)
            if question is not None:
                record = {"request_hash": digest, "refused_request": {"status": status, "question": question, "payload": payload[:4000]}}
                self._write(self._path(digest), record)
                raise _Refusal(record)
            record = {"request_hash": digest, "rejected": {"status": status, "payload": payload[:4000]}}
            self._write(self._path(digest), record)
            raise _Rejection(record)
        usage = payload.get("usage") or {}
        self.spend.settle(estimate, {"request_hash": digest, "model": payload.get("model"), "provider": payload.get("provider"),
                                     "input_tokens": int(usage.get("input_tokens") or 0), "cost": float(usage.get("cost") or 0.0),
                                     "id": payload.get("id")})
        try:
            check_pin(payload, self.spec)
        except SnapshotMismatch:
            self._write(self.cache_dir / "offpin" / f"{digest}.json", {"request_hash": digest, "request": body, "response": payload})
            raise
        return payload

    def _send(self, body: dict[str, Any]) -> tuple[int, Any, str | None]:
        """(200, payload, None), (rejection status, response text, None) or, after REFUSAL_ATTEMPTS whole-request
        refusals, (502, response text, refused question); retries 429/5xx/52x and 200 error envelopes."""
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        refusals = 0
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
                payload = _json_or_none(resp)
                if isinstance(payload, dict) and "answers" in payload:
                    return 200, payload, None
                if attempt == self.max_retries:  # 200 with an error envelope or a non-JSON body: retryable
                    raise JevError(200, resp.text)
            elif resp.status_code in REJECT_STATUS:
                return resp.status_code, resp.text, None
            else:
                question = refused_question(resp, self.spec)
                if question is not None:
                    refusals += 1
                    if refusals == REFUSAL_ATTEMPTS:
                        return resp.status_code, resp.text, question
                if resp.status_code not in RETRY_STATUS or attempt == self.max_retries:
                    raise JevError(resp.status_code, resp.text)
            self._sleep(_retry_delay(resp, attempt))
        raise AssertionError("unreachable")


def luna_api_key(search_from: list[Path]) -> str | None:
    """OPENROUTER_API_KEY from the environment unless it is framing_features' placeholder, else the nearest `.env`;
    None (cache-only) if there is none. Never logged."""
    try:
        return load_api_key(search_from, ignore=OFFLINE_KEY)
    except RuntimeError:
        return None


def make_luna_client(cache_dir: Path, search_from: list[Path], offline: bool = False, spec: Spec = LUNA) -> LunaClient:
    """Client of `spec`'s model from the environment: cache `JEV_OPT_<NAME>_CACHE` (default `<cache_dir>/<name>`,
    e.g. `JEV_OPT_LUNA_CACHE`, `<cache_dir>/luna`), spend ledger `JEV_OPT_<NAME>_SPEND` (default
    `runs/<name>/spend.jsonl`, run label `rq3`), endpoint `JEV_OPT_DECISIONS_URL`.
    `offline`: no key, so a cache miss raises `LunaCacheMiss` (replay)."""
    env = f"JEV_OPT_{spec.name.upper()}"
    cache = Path(os.environ.get(f"{env}_CACHE") or Path(cache_dir) / spec.name)
    spend = SpendLedger(Path(os.environ.get(f"{env}_SPEND") or spec.spend_ledger), run=RUN, spec=spec)
    return LunaClient(cache, None if offline else luna_api_key(search_from), spend, spec)
