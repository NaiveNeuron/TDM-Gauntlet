"""Freeze the Re3-Sci edit-intent test split used by Ruan et al. (EMNLP 2024).

Source: https://github.com/UKPLab/emnlp2024-llm-classifier (commit
a714eb12bd20846bf373ad3c2800972e79658e71), file
data/Re3-Sci/tasks/edit_intent_classification/test.csv. This is the exact
train/val/test split the paper's fine-tuning scripts load
(tasks/task_data_loader.py). Re3-Sci annotations: CC-BY-NC 4.0.

Download first (writes data/raw/edit_intent/):
    git clone https://github.com/UKPLab/emnlp2024-llm-classifier.git \
        data/raw/edit_intent/emnlp2024-llm-classifier
    git -C data/raw/edit_intent/emnlp2024-llm-classifier checkout a714eb12bd20846bf373ad3c2800972e79658e71
Then:
    uv run python scripts/prepare_edit_intent.py

Rows are kept verbatim and in file order. Like the authors' loader
(`load_dataset("csv", keep_default_na=False)`), empty cells stay empty strings:
`text_tgt` (old sentence) is empty for additions, `text_src` (new sentence) is
empty for deletions. `edit_action` is derived from which side is empty.
Item id = "re3sci-test-<edit_index>" (edit_index is unique in the release).
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = (
    REPO / "data" / "raw" / "edit_intent" / "emnlp2024-llm-classifier"
    / "data" / "Re3-Sci" / "tasks" / "edit_intent_classification" / "test.csv"
)
OUT = REPO / "data" / "frozen" / "edit_intent" / "test.jsonl"
RAW_SHA256 = "e784711574d82b73425396f42a636997371326c22e2e4f69fbf3ba00a73b471b"
EXPECTED_N = 2312
LABELS = {"Grammar", "Clarity", "Fact/Evidence", "Claim", "Other"}


def edit_action(old: str, new: str) -> str:
    if old and new:
        return "Modify"
    if new:
        return "Add"
    if old:
        return "Delete"
    raise ValueError("edit with both sides empty")


def main() -> None:
    digest = hashlib.sha256(RAW.read_bytes()).hexdigest()
    if digest != RAW_SHA256:
        raise SystemExit(f"unexpected raw SHA-256 {digest}")
    rows = []
    with RAW.open(newline="", encoding="utf-8") as fh:
        for src in csv.DictReader(fh):
            if src["label"] not in LABELS:
                raise SystemExit(f"edit {src['edit_index']}: unknown label {src['label']!r}")
            old, new = src["text_tgt"], src["text_src"]
            rows.append(
                {
                    "id": f"re3sci-test-{int(src['edit_index'])}",
                    "edit_index": int(src["edit_index"]),
                    "pair_id": src["id"],
                    "doc_name": src["doc_name"],
                    "node_ix_old": src["node_ix_tgt"],
                    "node_ix_new": src["node_ix_src"],
                    "old_text": old,
                    "new_text": new,
                    "edit_action": edit_action(old, new),
                    "gold": src["label"],
                }
            )
    if len(rows) != EXPECTED_N:
        raise SystemExit(f"expected {EXPECTED_N} rows, got {len(rows)}")
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("duplicate item ids")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {OUT.relative_to(REPO)}")
    print("labels", dict(Counter(r["gold"] for r in rows)))
    print("actions", dict(Counter(r["edit_action"] for r in rows)))
    print("sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
