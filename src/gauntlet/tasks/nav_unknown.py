"""Navigating the Unknown (Sali & Toraman, Findings EMNLP 2025): intent + near-OOD.

Input: one user utterance. Output: one of the in-distribution (ID) intents or
"fallback" (OOD). Half of each dataset's intents are held out as OOD (paper
Section 4.1: "We split the labels 50%-50% as ID and OOD"); OOD utterances belong to
the same domain but to unseen intents. Replaces the paper's GPT-4o intent
classifier/OOD detector; the matched setting is its zero-shot tool-calling method
(intent names only + a "fallback" tool, Appendix A.2/A.3), which Jev's choice
primitive replaces.

English public datasets only: CLINC150 (1,500 = 150 intents x 10) and
BANKING77 (770 = 77 x 10); DSTC Finance omitted. The paper releases no code, split or
seeds; `scripts/prepare_nav_unknown.py` freezes the first 10 test utterances per
intent and three ID/OOD label partitions (seeds 0, 1, 2) mirroring the paper's
"average of 3 runs". One request per utterance carries one choice question per
partition (answered independently over the same state).

Metrics (paper Section 3.1, Tables 2-3): ALL macro-F1 over ID intents + OOD class,
ID macro-F1 over the ID intents, OOD F1 and OOD recall; each is the mean over the
three partitions. Primary: ALL macro-F1. The paper does not define the averages; its
numbers are not reproducible from these definitions alone (see `all_weighted_f1`,
`id_macro_f1_id_rows` for the plausible alternatives, reported as sub-metrics).
"""

from __future__ import annotations

import json
from functools import cached_property
from typing import Any, Iterator

from .. import metrics
from ..task import FROZEN, Item, Request, Task, read_jsonl

KEY = "intent"
OOD = "fallback"
# Frozen truncation rule: utterances are cut to their first MAX_CHARS characters.
# The longest frozen utterance is 266 characters, so nothing is truncated.
MAX_CHARS = 20_000


def _clip(text: str) -> tuple[str, bool]:
    return (text[:MAX_CHARS], True) if len(text) > MAX_CHARS else (text, False)


def _qid(seed: str) -> str:
    return f"partition_{seed}"


class NavUnknown(Task):
    prompt_file = "nav_unknown.json"

    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.name = f"nav_unknown/{dataset}"

    @cached_property
    def partitions(self) -> dict[str, dict[str, list[str]]]:
        """seed -> {"id": sorted ID intents, "ood": sorted OOD intents}."""
        return json.loads((FROZEN / "nav_unknown" / "splits.json").read_text())[self.dataset]

    @cached_property
    def _prompt(self) -> dict[str, Any]:
        return self.prompt()

    def labels(self, seed: str) -> list[str]:
        """Choice labels for one partition; fixed order is also the argmax tie-break order."""
        return [*self.partitions[seed]["id"], OOD]

    def gold(self, item: Item, seed: str) -> str:
        return item.gold if item.gold in self.partitions[seed]["id"] else OOD

    def items(self) -> list[Item]:
        rows = read_jsonl(FROZEN / "nav_unknown" / f"{self.dataset}.jsonl")
        return [Item(id=r["id"], gold=r["intent"], fields={"text": r["text"]}) for r in rows]

    def _question(self, seed: str) -> dict[str, Any]:
        p = self._prompt
        names = p["intent_criteria"][self.dataset]
        criteria = {intent: names[intent] for intent in self.partitions[seed]["id"]}
        criteria[OOD] = p["fallback_criterion"]
        return {"type": "choice", "instructions": p["instructions"], "criteria": criteria}

    def requests(self, item: Item) -> Iterator[Request]:
        text, _ = _clip(item.fields["text"])
        questions = {_qid(seed): self._question(seed) for seed in self.partitions}
        yield Request(item_id=item.id, key=KEY, state={"utterance": text}, questions=questions)

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        out = {}
        for seed in self.partitions:
            probs = answers[KEY][_qid(seed)]["probabilities"]
            labels = self.labels(seed)
            p = {lab: float(probs.get(lab, 0.0)) for lab in labels}
            # argmax; exact ties go to the earlier label (sorted ID intents, then fallback).
            label = max(labels, key=lambda lab: (p[lab], -labels.index(lab)))
            out[seed] = {"label": label, "p_ood": p[OOD]}
        return out

    def _score_partition(self, items: list[Item], predictions: dict[str, Any], seed: str) -> dict[str, Any]:
        known = self.partitions[seed]["id"]
        labels = self.labels(seed)
        gold = [self.gold(it, seed) for it in items]
        pred = [predictions[it.id][seed]["label"] for it in items]
        p_ood = [predictions[it.id][seed]["p_ood"] for it in items]
        id_rows = [i for i, g in enumerate(gold) if g != OOD]
        n_ood = len(gold) - len(id_rows)
        ood_hits = sum(g == OOD and p == OOD for g, p in zip(gold, pred))
        n_pred_ood = pred.count(OOD)
        return {
            "all_macro_f1": metrics.macro_f1(gold, pred, labels=labels),
            "id_macro_f1": metrics.macro_f1(gold, pred, labels=known),
            "ood_f1": metrics.per_class_f1(gold, pred, labels=[OOD])[OOD],
            "ood_recall": ood_hits / n_ood,
            "ood_precision": ood_hits / n_pred_ood if n_pred_ood else 0.0,
            # Alternatives for the paper's undefined averages:
            "all_weighted_f1": metrics.weighted_f1(gold, pred, labels=labels),
            "id_macro_f1_id_rows": metrics.macro_f1(
                [gold[i] for i in id_rows], [pred[i] for i in id_rows], labels=known
            ),
            "accuracy": metrics.accuracy(gold, pred),
            # Supplementary (not in the paper): OOD ranking quality of P(fallback).
            "ood_auroc": metrics.roc_auc([int(g == OOD) for g in gold], p_ood),
            "n_id_intents": len(known),
            "n_ood_intents": len(self.partitions[seed]["ood"]),
            "n_id": len(id_rows),
            "n_ood": n_ood,
            "n_pred_ood": n_pred_ood,
        }

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        per = {seed: self._score_partition(items, predictions, seed) for seed in self.partitions}
        keys = ("all_macro_f1", "id_macro_f1", "ood_f1", "ood_recall", "ood_precision",
                "all_weighted_f1", "id_macro_f1_id_rows", "accuracy", "ood_auroc")
        result: dict[str, Any] = {"n": len(items)}
        result.update({k: metrics.mean([per[s][k] for s in per]) for k in keys})
        result["per_partition"] = per
        result["n_truncated"] = sum(_clip(it.fields["text"])[1] for it in items)
        return result


TASKS = {
    "nav_unknown/clinc150": lambda: NavUnknown("clinc150"),
    "nav_unknown/banking77": lambda: NavUnknown("banking77"),
}
