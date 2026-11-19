# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "numpy", "scikit-learn", "scipy"]
# ///
"""jev-opt splits for Re3-Sci edit intent (gauntlet task `edit_intent`). No Jev calls.

train/val: the paper's own `train.csv` / `val.csv` (UKPLab/emnlp2024-llm-classifier,
data/Re3-Sci/tasks/edit_intent_classification), read like scripts/prepare_edit_intent.py
(empty cells stay empty; edit_action from which side is empty), samples stratified by
(edit action, label) in natural proportions. Scientific English throughout, no language
screen (none is applied to the test set either). test = `edit_intent` items() (2,312).

State = {edit_action, old_text, new_text}, as the gauntlet sends it; the seed's switch
routes on edit_action to the Modify (5 labels) or Add/Delete (3 labels) question.

    uv run jev-opt/experiments/edit_intent/prepare.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import REPO, gauntlet_task, stratified, test_rows, write_splits  # noqa: E402

import prepare_edit_intent as pe  # noqa: E402  (gauntlet's freeze script, scripts/ on sys.path)

from gauntlet.task import Item  # noqa: E402

RAW = REPO / "data" / "raw" / "edit_intent" / "emnlp2024-llm-classifier" / "data" / "Re3-Sci" / "tasks" / "edit_intent_classification"


def pool(task, split: str) -> list[dict]:
    rows = []
    with (RAW / f"{split}.csv").open(newline="", encoding="utf-8") as fh:
        for src in csv.DictReader(fh):
            assert src["label"] in pe.LABELS, src["label"]
            old, new = src["text_tgt"], src["text_src"]
            item = Item(id="", gold=src["label"], fields={"old_text": old, "new_text": new, "edit_action": pe.edit_action(old, new)})
            state = task._state(item)[0]
            rows.append({"id": f"re3sci-{split}-{int(src['edit_index'])}", **state, "label": src["label"], "stratum": f"{state['edit_action'] != 'Modify'}|{src['label']}"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=500)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    task = gauntlet_task("edit_intent")
    train_pool, val_pool = pool(task, "train"), pool(task, "val")

    def clean(rows: list[dict]) -> list[dict]:
        return [{k: v for k, v in r.items() if k != "stratum"} for r in rows]

    test = test_rows(task, "edit_intent", lambda it, st: {"id": it.id, **st, "label": it.gold})
    write_splits(
        HERE / "data",
        {
            "train": clean(stratified(train_pool, args.train, args.seed, key="stratum")),
            "val": clean(stratified(val_pool, args.val, args.seed, key="stratum")),
            "test": test,
        },
        {
            "source": "Re3-Sci edit_intent_classification train.csv/val.csv (UKPLab/emnlp2024-llm-classifier @ a714eb1)",
            "pools": {"train": len(train_pool), "val": len(val_pool)},
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()
