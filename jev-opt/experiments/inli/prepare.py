# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "numpy", "scikit-learn", "scipy"]
# ///
"""jev-opt splits for INLI (gauntlet task `inli`). No Jev calls.

train/val: the release's `INLI Data/train.csv` (8,000 premises) and `val.csv` (1,000
premises), google-deepmind/inli @ e953ca2, each premise expanded into its four
premise-hypothesis pairs exactly like scripts/prepare_inli.py does for test. Whole premises
are sampled (so every split stays exactly balanced over the four labels): val = 250
premises (1,000 pairs), train = 125 premises (500 pairs). English release, no language
screen (as for test). test = `inli` items() (4,000 pairs).

    uv run jev-opt/experiments/inli/prepare.py
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import REPO, gauntlet_task, test_rows, write_splits  # noqa: E402

from gauntlet.task import Item  # noqa: E402
from gauntlet.tasks.inli import LABELS  # noqa: E402

RAW = REPO / "data" / "raw" / "inli" / "repo" / "INLI Data"


def pairs(task, split: str, premises: int, seed: int) -> tuple[list[dict], int]:
    with (RAW / f"{split}.csv").open(encoding="utf-8", newline="") as fh:
        src = [r for r in csv.DictReader(fh) if r["premise"].strip() and all(r[lab].strip() for lab in LABELS)]
    picked = sorted(random.Random(seed).sample(range(len(src)), premises))
    rows = []
    for i in picked:
        r = src[i]
        for label in LABELS:
            item = Item(id="", gold=label, fields={"premise": r["premise"], "hypothesis": r[label]})
            rows.append({"id": f"inli-{split}-{int(r['']):04d}-{label}", **task._state(item)[0], "label": label})
    random.Random(seed).shuffle(rows)
    return rows, len(src)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-premises", type=int, default=125)
    ap.add_argument("--val-premises", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    task = gauntlet_task("inli")
    train, n_train = pairs(task, "train", args.train_premises, args.seed)
    val, n_val = pairs(task, "val", args.val_premises, args.seed)
    test = test_rows(task, "inli", lambda it, st: {"id": it.id, **st, "label": it.gold})
    write_splits(
        HERE / "data",
        {"train": train, "val": val, "test": test},
        {
            "source": "INLI release (google-deepmind/inli, CC BY-SA 4.0), INLI Data/train.csv and val.csv",
            "pools": {"train_premises": n_train, "val_premises": n_val},
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()
