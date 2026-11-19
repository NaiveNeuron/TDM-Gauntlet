"""Scientific edit intent classification (Ruan, Kuznetsov & Gurevych, EMNLP 2024).

Input: one sentence-level edit from a Re3-Sci scientific paper revision, i.e. the
old sentence and the new sentence (old empty for additions, new empty for
deletions) -- the same text-only input the paper's classifiers get. Gold: the
human Re3-Sci edit intent, one of Grammar, Clarity, Fact/Evidence, Claim, Other.
Replaces the paper's best fine-tuned LLM classifiers (SeqC, Llama2-13B /
Llama3-8B-Instruct). Primary metrics (paper Table 2): accuracy and macro-F1 over
the five labels on the released test split, 2,312 edits.

Additions/deletions are asked with a three-label question (Fact/Evidence, Claim,
Other): Re3 (ACL 2024, Section 5) states they are never surface language edits,
and no released train/val addition/deletion is Grammar or Clarity.
"""

from __future__ import annotations

from typing import Any, Iterator

from .. import metrics
from ..task import FROZEN, Item, Request, Task, read_jsonl

# Authors' label2id order (sorted label names); also the argmax tie-break order.
LABELS = ("Claim", "Clarity", "Fact/Evidence", "Grammar", "Other")
ADD_DELETE_LABELS = ("Claim", "Fact/Evidence", "Other")
KEY = "edit_intent"
# Frozen truncation rule: each of old/new sentence is cut to its first
# MAX_FIELD_CHARS characters (~9K tokens each, well inside the 32K context).
# The longest released test sentence is 1,667 chars, so nothing is truncated.
MAX_FIELD_CHARS = 36_000


def _clip(text: str) -> tuple[str, bool]:
    return (text[:MAX_FIELD_CHARS], True) if len(text) > MAX_FIELD_CHARS else (text, False)


def _question_id(action: str) -> str:
    return "modify" if action == "Modify" else "add_delete"


class EditIntent(Task):
    name = "edit_intent"
    prompt_file = "edit_intent.json"

    def items(self) -> list[Item]:
        rows = read_jsonl(FROZEN / "edit_intent" / "test.jsonl")
        return [
            Item(
                id=r["id"],
                gold=r["gold"],
                fields={"old_text": r["old_text"], "new_text": r["new_text"], "edit_action": r["edit_action"]},
            )
            for r in rows
        ]

    def _state(self, item: Item) -> tuple[dict[str, str], bool]:
        old, cut_old = _clip(item.fields["old_text"])
        new, cut_new = _clip(item.fields["new_text"])
        return {"edit_action": item.fields["edit_action"], "old_text": old, "new_text": new}, cut_old or cut_new

    def requests(self, item: Item) -> Iterator[Request]:
        state, _ = self._state(item)
        question = self.prompt()[_question_id(item.fields["edit_action"])]
        yield Request(item_id=item.id, key=KEY, state=state, questions={KEY: question})

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        allowed = LABELS if item.fields["edit_action"] == "Modify" else ADD_DELETE_LABELS
        probs = answers[KEY][KEY]["probabilities"]
        p = {label: float(probs.get(label, 0.0)) for label in allowed}
        # argmax; exact ties go to the earlier label in LABELS order.
        label = max(allowed, key=lambda lab: (p[lab], -LABELS.index(lab)))
        return {"label": label, "probabilities": p}

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        gold = [it.gold for it in items]
        pred = [predictions[it.id]["label"] for it in items]
        labels = list(LABELS)
        out: dict[str, Any] = {
            "n": len(items),
            "accuracy": metrics.accuracy(gold, pred),
            "macro_f1": metrics.macro_f1(gold, pred, labels=labels),
            "per_class_f1": metrics.per_class_f1(gold, pred, labels=labels),
            "n_gold": {lab: gold.count(lab) for lab in LABELS},
            "n_pred": {lab: pred.count(lab) for lab in LABELS},
        }
        # Supplementary breakdown as in Re3 (ACL 2024) Table 10: revision pairs vs
        # additions/deletions, macro-F1 over the labels possible in each subset.
        by_action: dict[str, Any] = {}
        for subset, actions, subset_labels in (
            ("modify", {"Modify"}, labels),
            ("add_delete", {"Add", "Delete"}, list(ADD_DELETE_LABELS)),
        ):
            idx = [i for i, it in enumerate(items) if it.fields["edit_action"] in actions]
            g = [gold[i] for i in idx]
            p = [pred[i] for i in idx]
            by_action[subset] = {
                "n": len(idx),
                "accuracy": metrics.accuracy(g, p) if idx else None,
                "macro_f1": metrics.macro_f1(g, p, labels=subset_labels) if idx else None,
            }
        out["by_action"] = by_action
        out["n_truncated"] = sum(self._state(it)[1] for it in items)
        return out


TASKS = {"edit_intent": EditIntent}
