"""MADE v3: recall-oriented routing decoupled from per-code acceptance.

v2 routes and accepts with one a-priori rule (a code is asked iff a parent has P > 0.5, and
accepted iff its own P > 0.5). v3 separates the two, as the paper's own protocol does for
acceptance ("per-label thresholds tuned on validation", data/manifests/made.json deviations):

- Routing: first the v2 traversal, unchanged (byte-identical requests `r{round}c{chunk}`, so on
  the test set it is served from the v2 answers), then extension rounds `x{round}c{chunk}` that ask
  every not-yet-asked code with an asked parent of P > tau_route, until no new code is reachable.
  The asked set equals a direct traversal at tau_route (Jev answers questions independently of
  how they are batched); only the extension questions are new.
- Acceptance: code c is accepted iff P(c) > t_c, with t_c fit per code on validation (codes with
  too few validation positives share a per-level threshold); the predicted set is closed over
  the deepest accepted codes' own IMDRF paths exactly as in v2.

tau_route and the thresholds are fit on the v3 dev sample of the released validation split
(scripts/fit_v3.py) and frozen in data/frozen/made/v3_params.json before any v3 test run.
`dev_v3/made_v3` routes at DEV_TAU_ROUTE (the grid's lowest value, so every grid point's asked
set is a subset of what it asks) and predicts with the v2 rule restricted to the v2-asked codes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from functools import cached_property
from typing import Any

import numpy as np

from ..task import Item, Request, Task, read_jsonl
from .made import FROZEN_DIR, HF_DIR, MADE, THRESHOLD, binary_entropy, row_sha256

PARAMS_PATH = FROZEN_DIR / "v3_params.json"
DEV_TAU_ROUTE = 0.1
DEV_MANIFEST = FROZEN_DIR / "dev_v3_manifest.jsonl"
VAL_FILE = HF_DIR / "val.jsonl"


def level(code: str) -> int:
    """IMDRF level from the code string: A01 -> 1, A0101 -> 2, A010101 -> 3."""
    return (len(code) - 1) // 2


def decode(p: dict[str, float], accept: Callable[[str], float], ancestors: dict[str, list[str]], descendants: dict[str, set[str]]) -> tuple[set[str], set[str]]:
    """(accepted codes, predicted set): accepted = P > threshold; the predicted set adds the codes
    on the deepest accepted codes' own paths (v2's closure rule)."""
    accepted = {c for c, x in p.items() if x > accept(c)}
    labels = set(accepted)
    for c in accepted:
        if not descendants[c] & accepted:
            labels.update(ancestors[c])
    return accepted, labels


class MADEv3(MADE):
    def __init__(self, name: str, split: str, tau_route: float, params: dict[str, Any] | None) -> None:
        super().__init__()
        self.name = name
        self.split = split
        self.tau_route = tau_route
        self.params = params

    def items(self) -> list[Item]:
        if self.split == "test":
            return super().items()
        by_idx = {row["idx"]: row for row in read_jsonl(VAL_FILE)}
        items = []
        for entry in read_jsonl(DEV_MANIFEST):
            row = by_idx.get(entry["idx"])
            if row is None or row_sha256(row["idx"], row["text"], row["labels"]) != entry["sha256"]:
                raise ValueError(f"MADE validation row idx={entry['idx']} missing or changed since the v3 dev freeze")
            items.append(Item(id=str(row["idx"]), gold=sorted(row["labels"]), fields={"text": row["text"]}))
        return items

    @cached_property
    def children(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {c: [] for c in self.vocab}
        for c in self.vocab:
            for par in self._parents[c]:
                out[par].append(c)
        return out

    def _walk(self, item: Item, answers: dict[str, dict[str, Any]]) -> tuple[dict[str, float], set[str], list[list[str]], list[Request]]:
        """(P per asked code, codes the v2 traversal asked, codes per round, missing requests)."""
        asked, rounds, missing = self._traverse(item, answers)
        v2_asked = set(asked)
        if missing:
            return asked, v2_asked, rounds, missing
        r = 0
        while True:
            frontier = [c for c in self.vocab if c not in asked and any(asked.get(par, 0.0) > self.tau_route for par in self._parents[c])]
            if not frontier:
                return asked, v2_asked, rounds, []
            r += 1
            chunks = self._chunks(frontier)
            keys = [f"x{r}c{i}" for i in range(len(chunks))]
            todo = [(k, ch) for k, ch in zip(keys, chunks) if k not in answers]
            if todo:
                state, _ = self._state(item)
                questions = self._prompt["questions"]
                reqs = [Request(item_id=item.id, key=k, state=state, questions={c: questions[c] for c in ch}) for k, ch in todo]
                return asked, v2_asked, rounds, reqs
            for key, chunk in zip(keys, chunks):
                for code in chunk:
                    asked[code] = float(answers[key][code]["noul"])
            rounds.append(frontier)

    def next_requests(self, item: Item, answers: dict[str, dict[str, Any]]) -> list[Request]:
        return self._walk(item, answers)[3]

    def threshold(self, code: str) -> float:
        return self.params["thresholds"].get(code, self.params["default_threshold"][str(level(code))])

    def _output(self, p: dict[str, float], accept: Callable[[str], float], extra: dict[str, Any]) -> dict[str, Any]:
        accepted, labels = decode(p, accept, self._ancestors, self._descendants)
        scores = {c: p[c] for c in accepted}
        for c in accepted:  # confidence of a path code = its deepest accepted descendant's P, as in v2
            if not self._descendants[c] & accepted:
                for anc in self._ancestors[c]:
                    scores[anc] = max(scores.get(anc, p.get(anc, 0.0)), p[c])
        entropy = {c: binary_entropy(scores[c]) for c in labels}
        return {
            "labels": sorted(labels),
            "scores": {c: scores[c] for c in sorted(labels)},
            "uncertainties": {c: entropy[c] for c in sorted(labels)},
            "uncertainty": float(np.mean(list(entropy.values()))) if labels else None,
            "uncertainty_sum_scored": float(sum(binary_entropy(x) for x in p.values())),
            **extra,
        }

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        asked, v2_asked, rounds, missing = self._walk(item, answers)
        if missing:
            raise ValueError(f"item {item.id}: answers missing for {[r.key for r in missing]}")
        n_requests = sum(len(self._chunks(r)) for r in rounds)
        extra = {"n_asked": len(asked), "n_asked_v2": len(v2_asked), "n_rounds": len(rounds), "n_requests": n_requests}
        if self.params is None:  # dev: the v2 rule on the v2-asked codes, plus every nonzero P for the fit
            out = self._output({c: asked[c] for c in v2_asked}, lambda c: THRESHOLD, extra)
            out["p"] = {c: x for c, x in sorted(asked.items()) if x > 0}
            return out
        return self._output(asked, self.threshold, extra)

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        out = super().score(items, predictions)
        out["tau_route"] = self.tau_route
        out["asked_mean_v2_traversal"] = float(np.mean([predictions[it.id]["n_asked_v2"] for it in items]))
        if self.params is not None:
            out["v3_params_sha256"] = hashlib.sha256(PARAMS_PATH.read_bytes()).hexdigest()
        return out


def _test_task() -> Task:
    params = json.loads(PARAMS_PATH.read_text())
    task = MADEv3("made_v3", "test", params["tau_route"], params)
    task.reuses_responses_of = ("made",)  # v2-traversal responses are stored under runs/made
    return task


def asked_at(p: dict[str, float], tau: float, roots: Iterable[str], children: dict[str, list[str]]) -> set[str]:
    """Codes a traversal routing at `tau` asks, given every P it could need (nonzero Ps of a
    run routed at a lower tau; unlisted codes have P = 0 and are never expanded)."""
    seen = set(roots)
    stack = list(seen)
    while stack:
        c = stack.pop()
        if p.get(c, 0.0) > tau:
            for ch in children[c]:
                if ch not in seen:
                    seen.add(ch)
                    stack.append(ch)
    return seen


TASKS: dict[str, Callable[[], Task]] = {
    "dev_v3/made_v3": lambda: MADEv3("dev_v3/made_v3", "dev_v3", DEV_TAU_ROUTE, None),
    **({"made_v3": _test_task} if PARAMS_PATH.exists() else {}),
}
