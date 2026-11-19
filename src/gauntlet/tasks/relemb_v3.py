"""Wiki-WD v3: shortlist rerank (round 3) and per-property log-biases over Jev's answers.

Rounds 1-2 are exactly v2 (prompts/relemb_wikiwd_v3.json copies every v2 field, checked here),
so round-1 requests are byte-identical to the v2 run and served from its cached answers.

Round 3 (new) asks, per entity pair, the v2 question again over a shortlist: the k
highest-probability properties of each round-1 half (probability > 0, ties in PID order),
option names only as in v2, in PID order, plus the v2 "none of the listed relations" option,
read as P0. Pairs with an empty shortlist are P0.

Decision modes (data/frozen/relemb/wikiwd_v3_params.json, fit on the v3 dev sample of the
release's validation file; never on test):
- "rerank":      round-3 argmax of log(max(p, floor)) + b[label] (b[P0] for the none option);
                 exact ties go to the earlier option (P0 first, then PID order). Pairs outside the
                 round-3 policy ("all": every pair with a shortlist; "any_real": only pairs where
                 a round-1 half winner is a property) are decided by round 1: P0 when both halves
                 pick "none".
- "round1_bias": no new requests: each half's winner by log(max(p, floor)) + b[label] (b[P0]
                 for the half's none option); both none -> P0, one -> it, two -> the higher
                 biased score (ties -> half A).
Round 2 (the v2 final) is requested only by the dev task, to score the v2 baseline.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Any

from ..task import PROMPTS, Item, Request, Task
from .relemb import DATA, Criteria, QSpec, WikiWDTask, _question_answers, argmax_label, build_requests, plan_chunks

PARAMS_PATH = DATA / "wikiwd_v3_params.json"
DEV_KS = (2, 3, 5)
V2_FIELDS = ("task", "question", "labels", "negative_label", "none_option", "halves")


class WikiWDRerank(WikiWDTask):
    prompt_file = "relemb_wikiwd_v3.json"

    def __init__(self, name: str, data_file: str, ks: Sequence[int], with_round2: bool, params: dict[str, Any] | None) -> None:
        self.name = name
        self.data_file = data_file
        self.ks = tuple(ks)
        self.with_round2 = with_round2
        self.params = params
        v2 = json.loads((PROMPTS / self.spec["_round1_prompt"]).read_text())
        if any(self.spec[f] != v2[f] for f in V2_FIELDS):
            raise ValueError("prompts/relemb_wikiwd_v3.json must repeat every v2 field verbatim (cached rounds 1-2)")
        if params is not None:
            if params["mode"] not in ("rerank", "round1_bias") or (params["mode"] == "rerank" and params["k"] not in self.ks):
                raise ValueError(f"bad v3 params {params}")

    @cached_property
    def none_key(self) -> str:
        return self.spec["none_option"]["key"]

    @cached_property
    def pid_order(self) -> dict[str, int]:
        return {lab: i for i, lab in enumerate(self.inv.labels)}

    def _half_probs(self, q: dict[str, Any], half: Sequence[str]) -> dict[str, float]:
        """Round-1 half answer as {label: p}, the none option under the negative label."""
        probs = q["probabilities"]
        out = {self.inv.negative: float(probs.get(self.none_key, 0.0))}
        out.update({lab: float(probs.get(self.inv.key[lab], 0.0)) for lab in half})
        return out

    def _round1(self, n_pairs: int, answers: dict[str, dict[str, Any]]) -> list[tuple[dict[str, float], dict[str, float]]]:
        r1 = _question_answers(answers, "r1.")
        a, b = self.halves
        return [(self._half_probs(r1[f"p{j}a"], a), self._half_probs(r1[f"p{j}b"], b)) for j in range(n_pairs)]

    def shortlist(self, half_a: dict[str, float], half_b: dict[str, float], k: int) -> list[str]:
        out: set[str] = set()
        for probs in (half_a, half_b):
            pos = [lab for lab, p in probs.items() if lab != self.inv.negative and p > 0]
            out.update(sorted(pos, key=lambda lab: (-probs[lab], self.pid_order[lab]))[:k])
        return sorted(out, key=self.pid_order.__getitem__)

    def _in_policy(self, half_a: dict[str, float], half_b: dict[str, float]) -> bool:
        policy = self.params["policy"] if self.params else "all"
        if policy == "all":
            return True
        winners = [argmax_label(h, [self.inv.negative] + [lab for lab in h if lab != self.inv.negative]) for h in (half_a, half_b)]
        return any(w != self.inv.negative for w in winners)

    def _round3_groups(self, item: Item, halves: list[tuple[dict[str, float], dict[str, float]]], k: int) -> list[list[QSpec]]:
        instrs = self._pair_instructions(item)[0]
        groups = []
        for j, (ha, hb) in enumerate(halves):
            short = self.shortlist(ha, hb, k)
            if short and self._in_policy(ha, hb):
                crit = Criteria.of({self.none_key: None, **self.inv.criteria(short)})
                groups.append([QSpec(f"p{j}", instrs[j], crit)])
        return groups

    def _round3_ks(self) -> tuple[int, ...]:
        if self.params is None:
            return self.ks
        return (self.params["k"],) if self.params["mode"] == "rerank" else ()

    def next_requests(self, item: Item, answers: dict[str, dict[str, Any]]) -> list[Request]:
        state, _ = self._state(item)
        groups1 = self._round1_groups(item)
        if any(f"r1.{c}" not in answers for c in range(len(plan_chunks(state, groups1)))):
            return build_requests(item.id, "r1.", state, groups1, skip=answers)
        out: list[Request] = []
        if self.with_round2:
            groups2 = self._round2_groups(item, self._winners(len(groups1), answers))
            out += build_requests(item.id, "r2.", state, groups2, skip=answers) if groups2 else []
        halves = self._round1(len(groups1), answers)
        for k in self._round3_ks():
            groups3 = self._round3_groups(item, halves, k)
            out += build_requests(item.id, f"r3k{k}.", state, groups3, skip=answers) if groups3 else []
        return out

    def _round3(self, answers: dict[str, dict[str, Any]], k: int) -> dict[int, dict[str, float]]:
        """Round-3 answers per pair index as {label: p}, the none option under the negative label."""
        out = {}
        for qid, q in _question_answers(answers, f"r3k{k}.").items():
            probs = q["probabilities"]
            out[int(qid[1:])] = {(self.inv.negative if key == self.none_key else self.inv.label_of[key]): float(p) for key, p in probs.items()}
        return out

    def _biased_argmax(self, probs: dict[str, float]) -> tuple[str, float]:
        """Argmax of log(max(p, floor)) + b over the options with p > 0 (an option Jev gives 0 is
        never chosen); P0 first, then PID order, on exact ties; no positive option -> P0."""
        floor, bias = self.params["floor"], self.params["bias"]
        best, best_s = self.inv.negative, -math.inf
        for lab in sorted((lab for lab, p in probs.items() if p > 0), key=self.pid_order.__getitem__):
            s = math.log(max(probs[lab], floor)) + bias.get(lab, 0.0)
            if s > best_s:
                best, best_s = lab, s
        return best, best_s

    def _decide(self, j: int, ha: dict[str, float], hb: dict[str, float], r3: dict[int, dict[str, float]] | None) -> str:
        neg = self.inv.negative
        if self.params["mode"] == "rerank":  # pairs outside the round-3 policy / with no shortlist -> P0
            return self._biased_argmax(r3[j])[0] if j in r3 else neg
        (wa, sa), (wb, sb) = self._biased_argmax(ha), self._biased_argmax(hb)
        if wa == neg or wb == neg:
            return wb if wa == neg else wa  # both none -> neg
        return wa if sa >= sb else wb

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        pairs, edge_pair = self._pairs(item)
        if self.params is None:  # dev task: v2 decision plus every feature the fit needs
            v2 = super().predict(item, answers)
            halves = self._round1(len(pairs), answers)
            nz = lambda d: {lab: p for lab, p in d.items() if p > 0}  # noqa: E731
            r2 = _question_answers(answers, "r2.")
            return {
                **v2,
                "edge_pair": edge_pair,
                "r1": [[nz(ha), nz(hb)] for ha, hb in halves],
                "r2": {qid: q["probabilities"] for qid, q in r2.items()},
                "r3": {str(k): {str(j): nz(p) for j, p in self._round3(answers, k).items()} for k in self.ks},
            }
        halves = self._round1(len(pairs), answers)
        r3 = self._round3(answers, self.params["k"]) if self.params["mode"] == "rerank" else None
        pair_label = [self._decide(j, ha, hb, r3) for j, (ha, hb) in enumerate(halves)]
        winners = self._winners(len(pairs), answers)
        return {"labels": [pair_label[j] for j in edge_pair], "round1": [[wa, wb] for wa, wb in winners],
                "n_round3": len(r3 or {})}

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        out = super().score(items, predictions)
        out["v3_params"] = None if self.params is None else {k: v for k, v in self.params.items() if k != "bias"}
        if self.params is not None:
            out["v3_params_sha256"] = hashlib.sha256(PARAMS_PATH.read_bytes()).hexdigest()
            out["n_round3_pairs"] = sum(predictions[it.id]["n_round3"] for it in items)
        return out


def _test_task() -> Task:
    params = json.loads(PARAMS_PATH.read_text())
    ks = (params["k"],) if params["mode"] == "rerank" else ()
    task = WikiWDRerank("relemb_v3/wikiwd", "wikiwd_test.jsonl.gz", ks, False, params)
    task.reuses_responses_of = ("relemb/wikiwd",)  # round-1 responses are stored under runs/relemb/wikiwd
    return task


TASKS: dict[str, Callable[[], Task]] = {
    "dev_v3/wikiwd_v3": lambda: WikiWDRerank("dev_v3/wikiwd_v3", "wikiwd_dev_v3.jsonl.gz", DEV_KS, True, None),
    **({"relemb_v3/wikiwd": _test_task} if PARAMS_PATH.exists() else {}),
}
