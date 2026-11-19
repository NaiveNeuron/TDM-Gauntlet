"""INLI (Havaldar et al., ACL 2025): four-way implied-entailment NLI.

Input: one premise and one hypothesis. Gold: the release column the hypothesis
comes from, one of implied_entailment / explicit_entailment / neutral /
contradiction. Replaces the paper's fine-tuned T5-XXL four-way classifier.
Metrics as in paper Table 6: overall accuracy over all test pairs (primary) and
implied-entailment accuracy, i.e. accuracy on the pairs whose gold label is
implied_entailment (recomputing the paper's released few-shot LLM outputs,
`Resources/Few-shot LLM Results.xlsx`, reproduces its Table 6 numbers this way).
Test set: the released `INLI Data/test.csv`, 1,000 premises x 4 hypotheses =
4,000 pairs (CC BY-SA 4.0, frozen as JSONL by scripts/prepare_inli.py).

One request per pair, as in the paper's protocol: showing a premise's four
hypotheses together would reveal the one-of-each-label group structure.
"""

from __future__ import annotations

from typing import Any, Iterator

from sklearn.metrics import confusion_matrix

from .. import metrics
from ..task import FROZEN, Item, Request, Task, read_jsonl

# Release column order; also the argmax tie-break order.
LABELS = ("implied_entailment", "explicit_entailment", "neutral", "contradiction")
ENTAILMENT = ("implied_entailment", "explicit_entailment")
SOURCES = ("circa", "ludwig", "normbank", "socialchem")
KEY = "relation"
# Frozen truncation rule: premise and hypothesis are each cut to their first
# MAX_FIELD_CHARS characters (~9K tokens each, inside the 32K context). The
# longest released test text is 436 chars, so nothing is truncated.
MAX_FIELD_CHARS = 36_000


def _clip(text: str) -> tuple[str, bool]:
    return (text[:MAX_FIELD_CHARS], True) if len(text) > MAX_FIELD_CHARS else (text, False)


def _three_way(label: str) -> str:
    return "entailment" if label in ENTAILMENT else label


def _label_accuracy(gold: list[str], pred: list[str], label: str) -> float:
    """Accuracy on the pairs whose gold label is `label` (= recall of `label`)."""
    hits = [p == label for g, p in zip(gold, pred) if g == label]
    return sum(hits) / len(hits) if hits else float("nan")


class INLI(Task):
    name = "inli"
    prompt_file = "inli.json"

    def items(self) -> list[Item]:
        rows = read_jsonl(FROZEN / "inli" / "test.jsonl")
        return [
            Item(
                id=r["id"],
                gold=r["gold"],
                fields={
                    "group": r["group"],
                    "source_dataset": r["source_dataset"],
                    "premise": r["premise"],
                    "hypothesis": r["hypothesis"],
                },
            )
            for r in rows
        ]

    def _state(self, item: Item) -> tuple[dict[str, str], bool]:
        premise, cut_p = _clip(item.fields["premise"])
        hypothesis, cut_h = _clip(item.fields["hypothesis"])
        return {"premise": premise, "hypothesis": hypothesis}, cut_p or cut_h

    def requests(self, item: Item) -> Iterator[Request]:
        state, _ = self._state(item)
        yield Request(item_id=item.id, key=KEY, state=state, questions={KEY: self.prompt()["question"]})

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        probs = answers[KEY][KEY]["probabilities"]
        p = {label: float(probs.get(label, 0.0)) for label in LABELS}
        # argmax; exact ties go to the earlier label in LABELS.
        label = max(LABELS, key=lambda lab: (p[lab], -LABELS.index(lab)))
        return {"label": label, "probabilities": p}

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        gold = [it.gold for it in items]
        pred = [predictions[it.id]["label"] for it in items]
        by_source = {
            src: metrics.accuracy(
                [g for g, it in zip(gold, items) if it.fields["source_dataset"] == src],
                [p for p, it in zip(pred, items) if it.fields["source_dataset"] == src],
            )
            for src in SOURCES
            if any(it.fields["source_dataset"] == src for it in items)
        }
        groups: dict[int, bool] = {}
        for g, p, it in zip(gold, pred, items):
            groups[it.fields["group"]] = groups.get(it.fields["group"], True) and g == p
        cm = confusion_matrix(gold, pred, labels=list(LABELS))
        return {
            "n": len(items),
            "accuracy": metrics.accuracy(gold, pred),
            "implied_entailment_accuracy": _label_accuracy(gold, pred, "implied_entailment"),
            "per_label_accuracy": {lab: _label_accuracy(gold, pred, lab) for lab in LABELS},
            # Supplementary (not in Table 6).
            "macro_f1": metrics.macro_f1(gold, pred, labels=list(LABELS)),
            "accuracy_by_source_dataset": by_source,
            "accuracy_3way": metrics.accuracy([_three_way(g) for g in gold], [_three_way(p) for p in pred]),
            "premise_groups_all_correct": sum(groups.values()) / len(groups) if groups else float("nan"),
            "confusion_matrix": {"labels": list(LABELS), "rows_gold_cols_pred": cm.tolist()},
            "n_gold": {lab: gold.count(lab) for lab in LABELS},
            "n_pred": {lab: pred.count(lab) for lab in LABELS},
            "n_truncated": sum(self._state(it)[1] for it in items),
        }


TASKS = {"inli": INLI}
