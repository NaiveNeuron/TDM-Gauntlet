"""ConTempo (Niu et al., Findings ACL 2024): event-pair temporal relation classification.

Two tasks with separate label spaces (paper Table 1):
  contempo/matres   MATRES Platinum test, 837 pairs / 20 docs: BEFORE, AFTER, EQUAL, VAGUE
  contempo/tbdense  TimeBank-Dense test, 1,427 event-event pairs / 9 docs:
                    BEFORE, AFTER, INCLUDES, IS_INCLUDED, SIMULTANEOUS, VAGUE

Input: the sentence (or two consecutive sentences) containing the event pair, with the
two event triggers marked <e1>...</e1> and <e2>...</e2>; the gold label is the relation
of event 1 to event 2 in the released annotation's order. Replaces the fine-tuned
RoBERTa-large ConTempo classifier (+ unified framework).

Primary metric (paper Section 4.1 and Table 3): micro-F1 in which "the VAGUE labels are
computed into the F1s for both MATRES and TBD but are regarded as negative labels",
i.e. micro-averaged P/R/F1 over the non-VAGUE labels (a VAGUE prediction is never a
true positive; a VAGUE gold is never a positive to recall).

Licence: source texts are copyrighted news (TimeBank via LDC; TempEval-3 Platinum) and
no redistribution licence is stated, so the repository holds only ID + SHA-256 manifests in
data/frozen/contempo/. `scripts/prepare_contempo.py` rebuilds the rows into
data/raw/contempo/built/ and `items()` verifies every row hash.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterator

from .. import metrics
from ..task import FROZEN, REPO, Item, Request, Task, read_jsonl

BUILT_DIR = REPO / "data" / "raw" / "contempo" / "built"
FROZEN_DIR = FROZEN / "contempo"
KEY = "relation"
NEGATIVE = "VAGUE"
# Fixed label orders; also the argmax tie-break order (earlier label wins).
MATRES_LABELS = ("BEFORE", "AFTER", "EQUAL", "VAGUE")
TBDENSE_LABELS = ("BEFORE", "AFTER", "INCLUDES", "IS_INCLUDED", "SIMULTANEOUS", "VAGUE")
# Frozen truncation rule: the context is cut to its first MAX_CONTEXT_CHARS characters.
# Contexts are at most two sentences (longest built context: 607 chars), so none is cut.
MAX_CONTEXT_CHARS = 60_000
ROW_FIELDS = ("id", "doc_id", "event1", "event2", "context", "sentence_distance", "gold")


def row_sha256(row: dict[str, Any]) -> str:
    payload = {k: row[k] for k in ROW_FIELDS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class ConTempo(Task):
    subset: str
    labels: tuple[str, ...]
    split = "test"

    def __init__(self, split: str = "test") -> None:
        # Non-test splits (v3 development samples of the released train/dev data) are
        # separate tasks under `<split>/contempo_<subset>`; the question and decoding are
        # the frozen test task's.
        if split != "test":
            self.split = split
            self.name = f"{split}/contempo_{self.subset}"

    def items(self) -> list[Item]:
        manifest = read_jsonl(FROZEN_DIR / f"{self.subset}_{self.split}_ids.jsonl")
        built = BUILT_DIR / f"{self.subset}_{self.split}.jsonl"
        if not built.exists():
            raise FileNotFoundError(
                f"{built} missing: download the raw sources listed in data/manifests/contempo.json "
                "and run `uv run --with nltk==3.9.1 python scripts/prepare_contempo.py`"
                + ("" if self.split == "test" else " and `uv run scripts/prepare_v3_dev.py`")
            )
        by_id = {r["id"]: r for r in read_jsonl(built)}
        items = []
        for entry in manifest:
            row = by_id.get(entry["id"])
            if row is None or row_sha256(row) != entry["sha256"]:
                raise ValueError(f"{self.name}: row {entry['id']!r} missing or changed since the freeze (sha256 mismatch)")
            if row["gold"] not in self.labels:
                raise ValueError(f"{self.name}: row {row['id']!r} has out-of-inventory gold {row['gold']!r}")
            items.append(
                Item(
                    id=row["id"],
                    gold=row["gold"],
                    fields={"context": row["context"], "event1": row["event1"], "event2": row["event2"]},
                )
            )
        return items

    def _state(self, item: Item) -> tuple[dict[str, str], bool]:
        context = item.fields["context"]
        cut = len(context) > MAX_CONTEXT_CHARS
        return {
            "context": context[:MAX_CONTEXT_CHARS],
            "event_1": item.fields["event1"],
            "event_2": item.fields["event2"],
        }, cut

    def requests(self, item: Item) -> Iterator[Request]:
        state, _ = self._state(item)
        yield Request(item_id=item.id, key=KEY, state=state, questions={KEY: self.prompt()["question"]})

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        probs = answers[KEY][KEY]["probabilities"]
        p = {label: float(probs.get(label, 0.0)) for label in self.labels}
        label = max(self.labels, key=lambda lab: (p[lab], -self.labels.index(lab)))
        return {"label": label, "probabilities": p}

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        gold = [it.gold for it in items]
        pred = [predictions[it.id]["label"] for it in items]
        positive = [lab for lab in self.labels if lab != NEGATIVE]
        tp = sum(g == p and g != NEGATIVE for g, p in zip(gold, pred))
        n_pred_pos = sum(p != NEGATIVE for p in pred)
        n_gold_pos = sum(g != NEGATIVE for g in gold)
        return {
            "n": len(items),
            # Primary (paper Table 3): micro-F1 with VAGUE as the negative label.
            "micro_f1": metrics.micro_f1(gold, pred, labels=positive),
            "micro_precision": tp / n_pred_pos if n_pred_pos else 0.0,
            "micro_recall": tp / n_gold_pos if n_gold_pos else 0.0,
            # Supplementary: all-label accuracy (= micro-F1 with VAGUE counted as a class).
            "accuracy": metrics.accuracy(gold, pred),
            "macro_f1_all_labels": metrics.macro_f1(gold, pred, labels=list(self.labels)),
            "per_class_f1": metrics.per_class_f1(gold, pred, labels=list(self.labels)),
            "n_gold": {lab: gold.count(lab) for lab in self.labels},
            "n_pred": {lab: pred.count(lab) for lab in self.labels},
            "confusion_gold_by_pred": {
                g: {p: sum(1 for a, b in zip(gold, pred) if a == g and b == p) for p in self.labels} for g in self.labels
            },
            "n_truncated": sum(self._state(it)[1] for it in items),
        }


class Matres(ConTempo):
    name = "contempo/matres"
    prompt_file = "contempo_matres.json"
    subset = "matres"
    labels = MATRES_LABELS


class TBDense(ConTempo):
    name = "contempo/tbdense"
    prompt_file = "contempo_tbdense.json"
    subset = "tbdense"
    labels = TBDENSE_LABELS


TASKS = {
    "contempo/matres": Matres,
    "contempo/tbdense": TBDense,
    "dev_v3/contempo_matres": lambda: Matres("dev_v3"),
    "dev_v3/contempo_tbdense": lambda: TBDense("dev_v3"),
}
