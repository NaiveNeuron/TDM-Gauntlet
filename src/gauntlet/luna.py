"""Decision models through OpenRouter's Decisions endpoint: GPT-6 Luna Decisions and Microsoft-Decision-1.
A `Spec` fixes a model's id, pinned snapshot, provider, price, caps, ledger and cache; `LUNA` is the default
everywhere.

Same request format and key as Jev (`gauntlet.jev`); the model id is the only change to a frozen request.
Every response is cached verbatim under its request hash (the body includes the model, so Jev and these models
never collide). On top of the Jev client:

- snapshot/provider pin: an off-pin response stops the run; it is billed and logged but quarantined, never cached;
- refused or invalid answers are replaced, on read, by the uniform no-information answer and listed
  (`response.refused`, verbatim answers in `response.raw_answers`);
- rejected requests (HTTP 400/413/422) are cached as rejections, never truncated, split or re-sent;
- a hard budget: a file-locked spend ledger per model shared by its processes (zero-shot sweep and RQ3) is checked
  before each uncached request against the total cap and the run label's cap.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .jev import DECISIONS_URL, RETRY_STATUS, JevClient, _backoff, _retry_delay, request_hash
from .task import REPO

REJECT_STATUS = {400, 413, 422}
REJECTED_MODEL = "rejected"  # snapshot label of a rejected request in run manifests
REFUSED_MODEL = "refused"  # snapshot label of a request refused as a whole (OpenRouter 502) in run manifests
REFUSAL_ATTEMPTS = 3  # the refusal is accepted once three attempts in a row return it


@dataclass(frozen=True)
class Spec:
    """One decision model on the Decisions endpoint and its pre-registered pin, price and budget."""

    name: str  # runs/<name>, data/cache/<name>, `gauntlet run --backend <name>`
    label: str
    model: str
    snapshot: str
    provider: str
    price: float  # $ per input token; no output charge
    total_cap: float
    run_caps: dict[str, float]
    protocol: str
    # OpenRouter returns a refusal as HTTP 502 for the whole request (observed for Luna 2026-10-09).
    refusal_re: re.Pattern[str]
    excluded: tuple[str, ...] = ("relemb/wikiwd",)  # by cost (Luna: about $250 at Jev's token count, its whole budget)

    @property
    def spend_ledger(self) -> Path:
        return REPO / "runs" / self.name / "spend.jsonl"

    @property
    def cache(self) -> Path:
        return REPO / "data" / "cache" / self.name


LUNA = Spec("luna", "Luna", "openai/gpt-6-luna-decisions", "openai/gpt-6-luna-decisions-20261006", "OpenAI", 1e-7,
            250.0, {"zero-shot": 235.0, "rq3": 35.0}, "Luna settings",
            re.compile(r'^OpenAI refused to answer question "(.*)"$', re.S))
MSD1 = Spec("msd1", "MSD1", "microsoft/microsoft-decision-1", "microsoft/microsoft-decision-1-20261009", "Azure", 0.042e-6,
            150.0, {"zero-shot": 130.0, "rq3": 20.0}, "MSD1 settings",
            re.compile(r'^.+ refused to answer question "(.*)"$', re.S))
SPECS = {spec.name: spec for spec in (LUNA, MSD1)}

# Luna's values, read by the Luna analyses.
MODEL, SNAPSHOT, PROVIDER, PRICE = LUNA.model, LUNA.snapshot, LUNA.provider, LUNA.price
TOTAL_CAP_USD, RUN_CAPS_USD = LUNA.total_cap, LUNA.run_caps
SPEND_LEDGER, CACHE = LUNA.spend_ledger, LUNA.cache
LUNA_EXCLUDED = LUNA.excluded


class BudgetExceeded(RuntimeError):
    pass


class SnapshotMismatch(RuntimeError):
    pass


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


def rejected_response(questions: dict[str, Any], rejection: dict[str, Any]) -> dict[str, Any]:
    return {"model": REJECTED_MODEL, "answers": {q: uniform_answer(spec) for q, spec in questions.items()},
            "usage": {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}, "rejected": rejection,
            "refused": list(questions)}


def refused_response(questions: dict[str, Any], refusal: dict[str, Any]) -> dict[str, Any]:
    """A request OpenRouter answers with `502 <provider> refused to answer question "<name>"`: no answer at all,
    so every question gets the uniform answer."""
    return {"model": REFUSED_MODEL, "answers": {q: uniform_answer(spec) for q, spec in questions.items()},
            "usage": {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}, "refused_request": refusal,
            "refused": list(questions)}


def refusal_question(resp: httpx.Response, spec: Spec = LUNA) -> str | None:
    """The question name of OpenRouter's refusal error, else None."""
    if resp.status_code != 502:
        return None
    try:
        message = resp.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return None
    match = spec.refusal_re.match(message) if isinstance(message, str) else None
    return match.group(1) if match else None


def answer_audit(response: dict[str, Any]) -> Counter[str]:
    """Counts for a run manifest from one (normalized) response."""
    c: Counter[str] = Counter()
    c["n_requests"] += 1
    c["n_rejected_requests"] += "rejected" in response
    c["n_refused_requests"] += "refused_request" in response
    c["n_answers"] += len(response["answers"])
    c["n_refused_answers"] += len(response.get("refused", ()))
    if "rejected" not in response:
        c["n_saturated_answers"] += sum(saturated(a) for q, a in response["answers"].items() if q not in response.get("refused", ()))
    return c


def check_pin(response: dict[str, Any], spec: Spec = LUNA) -> None:
    if response.get("model") != spec.snapshot or response.get("provider") != spec.provider:
        raise SnapshotMismatch(f"response from {response.get('model')!r} / {response.get('provider')!r}, "
                               f"pinned {spec.snapshot!r} / {spec.provider!r}; stopped ({spec.protocol})")


# ------------------------------------------------------------------------------------------------ budget


class SpendLedger:
    """Billed responses of one model, one JSON line each, shared by processes through an exclusive file lock.

    `reserve` checks spend so far + this process's in-flight estimates + the request's estimate against the
    total cap and the run label's cap; `settle` replaces the reservation by the billed cost.
    """

    def __init__(self, path: Path | None = None, run: str = "zero-shot", total_cap: float | None = None,
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


@dataclass
class LunaClient(JevClient):
    """`JevClient` with a decision model's pin, refusal and rejection handling and its spend ledger (`spec`,
    default Luna; the model id is `spec.model`).

    `decide` returns `{request_hash, response}` with the normalized response (what the run records and scores);
    the cache keeps the verbatim API response.
    """

    model: str = MODEL
    requests_per_minute: int = 1200
    url: str = DECISIONS_URL
    spec: Spec = LUNA
    spend: SpendLedger | None = None
    answer_audit = staticmethod(answer_audit)

    def __post_init__(self) -> None:
        self.model = self.spec.model
        if self.spend is None:
            self.spend = SpendLedger(spec=self.spec)
        super().__post_init__()
        self.fresh_cost = 0.0
        self.fresh_requests = 0
        self._loop: asyncio.AbstractEventLoop | None = None

    def _bind_loop(self) -> None:
        """One client serves several `run_task` calls (e.g. biorc800/dev, then biorc800), each with its own
        `asyncio.run` loop; the semaphore and pacing lock must belong to the running loop."""
        loop = asyncio.get_running_loop()
        if loop is not self._loop:
            self._loop = loop
            self._sem = asyncio.Semaphore(self.concurrency)
            self._slot_lock = asyncio.Lock()
            self._next_slot = 0.0

    def _view(self, body: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        if "rejected" in record:
            return {"request_hash": record["request_hash"], "response": rejected_response(body["questions"], record["rejected"])}
        if "refused_request" in record:
            return {"request_hash": record["request_hash"], "response": refused_response(body["questions"], record["refused_request"])}
        check_pin(record["response"], self.spec)
        return {"request_hash": record["request_hash"], "response": normalize(body["questions"], record["response"])}

    async def decide(self, http: httpx.AsyncClient, body: dict[str, Any]) -> dict[str, Any]:
        digest = request_hash(body)
        path = self._cache_path(digest)
        if path.exists():
            return self._view(body, json.loads(path.read_text()))
        estimate = estimate_cost(body, self.spec.price)
        self._bind_loop()
        async with self._sem:  # reserve per request actually being sent, not per queued coroutine
            if path.exists():  # an identical request finished while this one waited
                return self._view(body, json.loads(path.read_text()))
            self.spend.reserve(estimate)
            try:
                status, payload = await self._post(http, body)
            except BaseException:
                self.spend.release(estimate)
                raise
        if status == 200:
            usage = payload.get("usage") or {}
            cost = float(usage.get("cost") or 0.0)
            self.spend.settle(estimate, {"request_hash": digest, "model": payload.get("model"), "provider": payload.get("provider"),
                                         "input_tokens": int(usage.get("input_tokens") or 0), "cost": cost, "id": payload.get("id")})
            self.fresh_cost += cost
            self.fresh_requests += 1
            record: dict[str, Any] = {"request_hash": digest, "response": payload}
            try:
                check_pin(payload, self.spec)
            except SnapshotMismatch:
                self._write(self.cache_dir / "offpin" / f"{digest}.json", record)
                raise
        elif status == "refused":
            self.spend.release(estimate)
            record = {"request_hash": digest, "refused_request": payload}
        else:
            self.spend.release(estimate)
            record = {"request_hash": digest, "rejected": {"status": status, "payload": payload[:4000]}}
        self._write(path, record)
        return self._view(body, record)

    @staticmethod
    def _write(path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False))
        tmp.replace(path)

    async def _post(self, http: httpx.AsyncClient, body: dict[str, Any]) -> tuple[int | str, Any]:
        """(200, payload), (rejection status, response text) or ("refused", {status, question, payload}) after
        REFUSAL_ATTEMPTS refusals; retries 429/5xx/52x and 200 error envelopes."""
        from .jev import JevError

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        refusals = 0
        for attempt in range(self.max_retries + 1):
            await self._pace()
            try:
                resp = await http.post(self.url, json=body, headers=headers, timeout=self.timeout_s)
            except (httpx.TransportError, httpx.TimeoutException):
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(_backoff(attempt))
                continue
            if resp.status_code == 200:
                try:
                    payload = resp.json()
                except ValueError:  # empty/non-JSON 200 body (seen 2026-10-09): retried like an error envelope
                    payload = None
                if isinstance(payload, dict) and "answers" in payload:
                    return 200, payload
                if attempt == self.max_retries:
                    raise JevError(200, resp.text)
            elif resp.status_code in REJECT_STATUS:
                return resp.status_code, resp.text
            elif (question := refusal_question(resp, self.spec)) is not None:
                refusals += 1
                if refusals == REFUSAL_ATTEMPTS:
                    return "refused", {"status": resp.status_code, "question": question, "payload": resp.text[:4000]}
                if attempt == self.max_retries:
                    raise JevError(resp.status_code, resp.text)
            elif resp.status_code not in RETRY_STATUS or attempt == self.max_retries:
                raise JevError(resp.status_code, resp.text)
            await asyncio.sleep(_retry_delay(resp, attempt))
        raise AssertionError("unreachable")


def write_spend_summary(spec: Spec = LUNA, ledger: Path | None = None, out: Path | None = None) -> dict[str, Any]:
    """Digest of the spend ledger: totals per run label, caps, off-pin count and
    the ledger's SHA-256. Every billed response's usage is also in the run directories."""
    import hashlib

    runs: dict[str, dict[str, Any]] = {}
    digest = hashlib.sha256()
    off_pin = lines = 0
    ledger = spec.spend_ledger if ledger is None else ledger
    out = ledger.with_name("spend_summary.json") if out is None else out
    with open(ledger, "rb") as fh:
        for raw in fh:
            digest.update(raw)
            if not raw.strip():
                continue
            lines += 1
            row = json.loads(raw)
            r = runs.setdefault(row.get("run", "?"), {"responses": 0, "input_tokens": 0, "cost_usd": 0.0, "first": row.get("time"), "last": None})
            r["responses"] += 1
            r["input_tokens"] += int(row.get("input_tokens") or 0)
            r["cost_usd"] += float(row["cost"])
            r["last"] = row.get("time")
            off_pin += "model" in row and (row["model"] != spec.snapshot or row.get("provider") != spec.provider)
    for r in runs.values():
        r["cost_usd"] = round(r["cost_usd"], 6)
    summary = {"ledger": str(ledger.relative_to(REPO)) if ledger.is_relative_to(REPO) else str(ledger), "ledger_lines": lines,
               "ledger_sha256": digest.hexdigest(), "total_cost_usd": round(sum(r["cost_usd"] for r in runs.values()), 6),
               "total_cap_usd": spec.total_cap, "run_caps_usd": spec.run_caps, "off_pin_responses": off_pin, "runs": runs}
    out.write_text(json.dumps(summary, indent=1) + "\n")
    return summary


PROJECTION_REFERENCE = LUNA  # whose recorded tokens on the same requests are scaled
MULTI_QUESTION_MIN = 10  # answers per request for a cohort to count as multi-question


def project_cohort(spec: Spec, task: str, run: str = "zero-shot") -> dict[str, Any]:
    """Projection check before a large multi-question cohort (for MSD1: the four TTP cohorts and MADE, run last):
    the reference model's recorded input tokens on the cohort × the largest `spec`/reference token ratio over the
    cohorts `spec` has completed with ≥ MULTI_QUESTION_MIN answers per request (else over all completed cohorts) ×
    `spec.price`, added to the run label's spend so far, against the run label's cap. No request is made."""
    runs, ref_runs = REPO / "runs" / spec.name, REPO / "runs" / PROJECTION_REFERENCE.name
    ratios: dict[str, list[tuple[str, float]]] = {"multi": [], "all": []}
    for scores in sorted(runs.rglob("scores.json")):
        rel = scores.parent.relative_to(runs)
        ref = ref_runs / rel / "manifest.json"
        if "_smoke" in rel.parts or not ref.exists():
            continue
        m, r = json.loads((scores.parent / "manifest.json").read_text()), json.loads(ref.read_text())
        if not r.get("input_tokens"):
            continue
        ratio = m["input_tokens"] / r["input_tokens"]
        ratios["all"].append((rel.as_posix(), ratio))
        audit = m.get("answer_audit") or {}
        if audit.get("n_requests") and audit["n_answers"] / audit["n_requests"] >= MULTI_QUESTION_MIN:
            ratios["multi"].append((rel.as_posix(), ratio))
    basis = "multi" if ratios["multi"] else "all"
    if not ratios[basis]:
        raise SystemExit(f"no completed {spec.label} cohort to project from")
    worst = max(ratios[basis], key=lambda x: x[1])
    ref_tokens = json.loads((ref_runs / task / "manifest.json").read_text())["input_tokens"]
    projected = ref_tokens * worst[1] * spec.price
    ledger = SpendLedger(run=run, spec=spec)
    ledger._locked(lambda _: None)
    out = {"task": task, "reference": PROJECTION_REFERENCE.name, "reference_input_tokens": ref_tokens, "ratio_basis": basis,
           "ratio": round(worst[1], 4), "ratio_from": worst[0], "projected_cost_usd": round(projected, 4),
           "run": run, "run_spent_usd": round(ledger.run_total, 4), "run_cap_usd": ledger.run_cap,
           "total_spent_usd": round(ledger.total, 4), "total_cap_usd": ledger.total_cap}
    out["within_caps"] = (ledger.run_total + projected <= ledger.run_cap) and (ledger.total + projected <= ledger.total_cap)
    return out


if __name__ == "__main__":
    import sys

    if sys.argv[1:2] == ["project"] and len(sys.argv) == 4 and sys.argv[2] in SPECS:
        result = project_cohort(SPECS[sys.argv[2]], sys.argv[3])
        print(json.dumps(result))
        raise SystemExit(0 if result["within_caps"] else 3)
    if len(sys.argv) not in (2, 3) or sys.argv[1] != "summary" or sys.argv[2:3] and sys.argv[2] not in SPECS:
        raise SystemExit(f"usage: python -m gauntlet.luna summary [{'|'.join(SPECS)}]  (default luna)\n"
                         f"       python -m gauntlet.luna project <{'|'.join(SPECS)}> <task>")
    print(json.dumps(write_spend_summary(SPECS[sys.argv[2] if len(sys.argv) == 3 else "luna"]), indent=1))
