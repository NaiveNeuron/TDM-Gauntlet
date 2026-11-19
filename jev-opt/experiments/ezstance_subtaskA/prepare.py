# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "numpy", "scikit-learn", "scipy"]
# ///
"""jev-opt splits for EZ-STANCE Subtask A (gauntlet task `ezstance/subtaskA`). No Jev calls.

train/val: the release's Subtask A `raw_train_all_onecol.csv` / `raw_val_all_onecol.csv`
(noun_phrase/ and claim/ files, which fix the target type), stratified by
(target type, stance) in their natural proportions, after an automatic English screen
(`_common.looks_english`; the test set's screen was adjudicated flag by flag). test =
`ezstance/subtaskA` items() (7,666 pairs after the frozen non-English exclusions).

Rows: {tweet, target} is the Jev state; `target_type` only drives the seed's switch.

    uv run jev-opt/experiments/ezstance_subtaskA/prepare.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import gauntlet_task, looks_english, stratified, test_rows, write_splits  # noqa: E402

from gauntlet.tasks.ezstance import _raw_rows  # noqa: E402  (SHA-256-verified release zip)

TYPES = {"noun_phrase": "np", "claim": "claim"}


def pool(split: str) -> tuple[list[dict], int]:
    rows, dropped = [], 0
    for ttype, abbr in TYPES.items():
        for i, raw in enumerate(_raw_rows(f"subtaskA/{ttype}/raw_{split}_all_onecol.csv")):
            if not looks_english(raw["Text"]):
                dropped += 1
                continue
            rows.append({
                "id": f"A-{split}-{abbr}-{i:05d}",
                "tweet": raw["Text"],
                "target": raw["Target 1"],
                "target_type": ttype,
                "label": raw["Stance 1"],
                "stratum": f"{ttype}|{raw['Stance 1']}",
            })
    return rows, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=500)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train_pool, train_dropped = pool("train")
    val_pool, val_dropped = pool("val")

    def clean(rows: list[dict]) -> list[dict]:
        return [{k: v for k, v in r.items() if k != "stratum"} for r in rows]

    test = test_rows(
        gauntlet_task("ezstance/subtaskA"),
        "ezstance/subtaskA",
        lambda it, st: {"id": it.id, **st, "target_type": it.fields["target_type"], "label": it.gold},
    )
    write_splits(
        HERE / "data",
        {
            "train": clean(stratified(train_pool, args.train, args.seed, key="stratum")),
            "val": clean(stratified(val_pool, args.val, args.seed, key="stratum")),
            "test": test,
        },
        {
            "source": "EZ-STANCE release zip (data/raw/ezstance/ezstance_data.zip), Subtask A train/val CSVs",
            "pools": {"train": len(train_pool), "val": len(val_pool), "dropped_non_english": {"train": train_dropped, "val": val_dropped}},
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()
