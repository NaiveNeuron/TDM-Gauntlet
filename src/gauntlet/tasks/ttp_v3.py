"""Security TTP mapping v3: Jev's own round-1 top-k reranked by one relative Choice.

Round 1 is the frozen v1 scan (one noul per candidate over all 737 profiles). prompts/ttp_v3.json
repeats prompts/ttp.json's question template verbatim (checked here), so round-1 requests are
byte-identical to the v1 run and, on the test sets, are served from its cached answers.

Round 2 asks one Choice per text over the round-1 top-k (ranked by P(yes), ties at the cut in
ATT&CK ID order); options are "<ID> <name>" in ATT&CK ID order with a short description
(prompts/ttp_v3.json `rerank_options`). The final ranking reorders the top-k by

    s = (1 - w) * p_noul + w * p_choice

with ties broken by the round-2 answer (higher p_choice), then p_noul, then ATT&CK ID (v1 broke
every exact tie by ATT&CK ID; 6-11% of v1 texts lost the gold only to that rule). Candidates
outside the top-k keep the round-1 order below them, so the output is still a ranking of the
whole corpus, scored exactly as v1 (App. D.1 metrics). k and w are fit on the v3 dev sample
(released TRAM / Procedures / Derived-procedures dev splits only; Expert train/dev belong to the
`expert_full` test set and are never used) and frozen in data/frozen/ttp/v3_params.json before
any v3 test run.

`dev_v3/ttp_v3` asks round 2 for every k in DEV_KS (one request per k) and ranks with round 1
only; scripts/fit_v3.py fits (k, w) from its predictions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from typing import Any

from ..task import FROZEN, PROMPTS, Item, Request, Task
from . import ttp

PARAMS_PATH = FROZEN / "ttp" / "v3_params.json"
DEV_KS = (3, 5, 10)
QID = "rerank"


def rerank_order(top: Sequence[str], p1: dict[str, float], p2: dict[str, float], w: float, order: dict[str, int]) -> list[str]:
    """The top-k candidates by s = (1-w)*p1 + w*p2; ties -> higher p2, higher p1, ATT&CK ID."""
    return sorted(top, key=lambda t: (-((1 - w) * p1[t] + w * p2.get(t, 0.0)), -p2.get(t, 0.0), -p1[t], order[t]))


class TTPRerank(ttp.TTPMapping):
    prompt_file = "ttp_v3.json"

    def __init__(self, name: str, splits: list[tuple[str, str | None]], ks: Sequence[int], params: dict[str, Any] | None) -> None:
        super().__init__(name, splits)
        v3 = self.prompt()
        v1 = json.loads((PROMPTS / v3["_round1_prompt"]).read_text())
        if v3["question_template"] != v1["question_template"] or v3["_candidates_sha256"] != v1["_candidates_sha256"]:
            raise ValueError("prompts/ttp_v3.json round 1 must equal prompts/ttp.json (cached v1 answers)")
        self.rerank = v3["rerank"]
        self.options = v3["rerank_options"]
        self.key_to_id = {o["key"]: tid for tid, o in self.options.items()}
        self.ks = tuple(ks)
        self.params = params
        if params is not None and (params["k"] not in self.ks or not 0.0 <= params["w"] <= 1.0):
            raise ValueError(f"bad v3 params {params}")

    def _round1(self, answers: dict[str, dict[str, Any]]) -> dict[str, float] | None:
        p: dict[str, float] = {}
        for j, chunk in enumerate(self.chunks):
            got = answers.get(f"candidates-{j:02d}")
            if got is None:
                return None
            for qid in chunk:
                p[qid] = float(got[qid]["noul"])
        return p

    def _ranked(self, p1: dict[str, float]) -> list[str]:
        return sorted(p1, key=lambda t: (-p1[t], self.order[t]))

    def _rerank_request(self, item: Item, top: Sequence[str], k: int) -> Request:
        state, _ = self._state(item)
        crit = {self.options[t]["key"]: self.options[t]["description"] for t in sorted(top, key=self.order.__getitem__)}
        q = {"type": self.rerank["type"], "instructions": self.rerank["instructions"], "criteria": crit}
        return Request(item_id=item.id, key=f"rerank-k{k}", state=state, questions={QID: q})

    def next_requests(self, item: Item, answers: dict[str, dict[str, Any]]) -> list[Request]:
        p1 = self._round1(answers)
        if p1 is None:
            return [r for r in self.requests(item) if r.key not in answers]
        ranked = self._ranked(p1)
        return [self._rerank_request(item, ranked[:k], k) for k in self.ks if f"rerank-k{k}" not in answers]

    def _round2(self, answers: dict[str, dict[str, Any]], k: int) -> dict[str, float]:
        probs = answers[f"rerank-k{k}"][QID]["probabilities"]
        return {self.key_to_id[key]: float(p) for key, p in probs.items()}

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        p1 = self._round1(answers)
        if p1 is None:
            raise ValueError(f"{item.id}: round-1 answers missing")
        ranked = self._ranked(p1)
        p2s = {k: self._round2(answers, k) for k in self.ks}
        if self.params is not None:
            k, w = self.params["k"], self.params["w"]
            ranked = rerank_order(ranked[:k], p1, p2s[k], w, self.order) + ranked[k:]
        return {
            "ranking": ranked[: ttp.KEEP],
            "p": [p1[t] for t in ranked[: ttp.KEEP]],
            "technique_ranking": ttp.technique_ranking(ranked)[: ttp.TECH_KEEP],
            "rerank": {str(k): {t: p2s[k][t] for t in sorted(p2s[k], key=self.order.__getitem__)} for k in self.ks},
        }

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        out = super().score(items, predictions)
        out["v3_params"] = self.params
        if self.params is not None:
            out["v3_params_sha256"] = hashlib.sha256(PARAMS_PATH.read_bytes()).hexdigest()
        return out


def _test_factory(name: str) -> Callable[[], Task]:
    def make() -> Task:
        params = json.loads(PARAMS_PATH.read_text())
        task = TTPRerank(f"ttp_v3/{name.split('/', 1)[1]}", ttp.SPLITS[name], (params["k"],), {"k": params["k"], "w": params["w"]})
        task.reuses_responses_of = (name,)  # round-1 responses are stored under runs/ttp/<subset>
        return task

    return make


TASKS: dict[str, Callable[[], Task]] = {
    "dev_v3/ttp_v3": lambda: TTPRerank("dev_v3/ttp_v3", [("dev_v3", None)], DEV_KS, None),
    **({f"ttp_v3/{n.split('/', 1)[1]}": _test_factory(n) for n in ttp.SPLITS} if PARAMS_PATH.exists() else {}),
}
