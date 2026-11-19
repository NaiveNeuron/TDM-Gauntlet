# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "numpy", "scikit-learn", "scipy"]
# ///
"""jev-opt splits for DREsS_New (gauntlet task `dress_new`), one experiment per rubric. No Jev calls.

The gauntlet freezes the 20% test part of `train_test_split(range(n), test_size=0.2,
random_state=22)` (paper Appendix A: 6:2:2, seed 22). The remaining 80% is split the
same conventional way, `train_test_split(rest, test_size=0.25, random_state=22)`, into
train (60%) and val (20%); the same frozen row rules apply (empty essay / missing rubric
score dropped, >5% non-Latin letters dropped) plus rows with an off-grid score (0.5). val = the whole reconstructed val split;
train = a seeded random sample. test = `dress_new` items() (401 essays).

Writes data/<rubric>/{train,val,test}.jsonl for rubric in content, organization, language;
labels are the score strings "1.0" ... "5.0". Same essays in every rubric's splits.

    uv run jev-opt/experiments/dress_new/prepare.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import gauntlet_task, test_rows, write_splits  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from gauntlet.task import Item  # noqa: E402
from gauntlet.tasks.dress import (  # noqa: E402
    LEVELS,
    MAX_NON_LATIN_SHARE,
    RUBRICS,
    SPLIT_SEED,
    find_raw_file,
    load_raw,
    non_latin_share,
    test_row_indices,
)


def usable(row: dict) -> bool:
    return bool(row["essay"].strip()) and all(row[k] in LEVELS for k in RUBRICS) and non_latin_share(row["prompt"] + " " + row["essay"]) <= MAX_NON_LATIN_SHARE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    task = gauntlet_task("dress_new")
    raw = load_raw(find_raw_file())
    test_idx = set(test_row_indices(len(raw)))
    rest = np.array([i for i in range(len(raw)) if i not in test_idx])
    train_idx, val_idx = train_test_split(rest, test_size=0.25, random_state=SPLIT_SEED)
    train_pool = [raw[i] for i in sorted(int(i) for i in train_idx) if usable(raw[i])]
    val_rows = [raw[i] for i in sorted(int(i) for i in val_idx) if usable(raw[i])]
    train_rows = random.Random(args.seed).sample(train_pool, min(args.train, len(train_pool)))
    items = {it.id: it for it in task.items()}

    def row(r: dict, rubric: str) -> dict:
        state = task._state(Item(id=r["id"], gold=None, fields={"prompt": r["prompt"], "essay": r["essay"]}))[0]
        return {"id": f"dress-{r['id']}", **state, "label": f"{r[rubric]:.1f}"}

    for rubric in RUBRICS:
        test = test_rows(task, "dress_new", lambda it, st, rubric=rubric: {"id": f"dress-{it.id}", **st, "label": f"{it.gold[rubric]:.1f}"})
        write_splits(
            HERE / "data" / rubric,
            {"train": [row(r, rubric) for r in train_rows], "val": [row(r, rubric) for r in val_rows], "test": test},
            {
                "source": "DREsS_New (consented download, data/raw/dress_new); 6:2:2 seed-22 reconstruction",
                "rubric": rubric,
                "pools": {"train_split": len(train_pool), "val_split": len(val_rows), "test_items": len(items)},
                "seed": args.seed,
            },
        )


if __name__ == "__main__":
    main()
