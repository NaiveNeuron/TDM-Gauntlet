# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas==2.2.3", "scikit-learn==1.5.2", "numpy==2.1.3", "httpx", "scipy"]
# ///
"""jev-opt splits for SMARTER implicit hate (gauntlet task `smarter/implicit_hate`). No Jev calls.

train/val: the train (50%) and val (20%) parts of the authors' own `split_train_test`
(seed 123, Table 4 ratios 50:20:30) over Latent Hatred Stage-2 posts, `other` dropped and
Appendix B preprocessing applied -- exactly the split whose 30% test part the gauntlet
test set is sampled from (scripts/prepare_smarter.py). Label-proportional samples, an
automatic English screen (`_common.looks_english`), and any post whose ID or text is in
the test set removed. test = `smarter/implicit_hate` items() (900 posts, 150 per class).

    uv run jev-opt/experiments/smarter_implicit_hate/prepare.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import gauntlet_task, looks_english, stratified, test_rows, write_splits  # noqa: E402

import prepare_smarter as ps  # noqa: E402  (gauntlet's freeze script, scripts/ on sys.path)

from gauntlet.task import Item  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=500)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    task = gauntlet_task("smarter/implicit_hate")
    test = test_rows(task, "smarter/implicit_hate", lambda it, st: {"id": it.id, **st, "label": it.gold})
    test_uids = {it.id.removeprefix("implicit_hate-") for it in task.items()}
    test_texts = {r["post"] for r in test}

    df, _ = ps.latent_frame("implicit_hate")
    ratios, _ = ps.CONFIG["implicit_hate"]
    train_df, val_df, _ = ps.split_train_test(df, ratios, seed=ps.SPLIT_SEED)
    dropped: dict[str, dict[str, int]] = {}

    def pool(frame, split: str) -> list[dict]:
        rows, seen = [], set()
        stats = {"non_english": 0, "in_test": 0, "duplicate_uid": 0}
        for r in frame.itertuples():
            if r.uid in test_uids or r.text in test_texts:
                stats["in_test"] += 1
            elif r.uid in seen:
                stats["duplicate_uid"] += 1
            elif not looks_english(r.text):
                stats["non_english"] += 1
            else:
                seen.add(r.uid)
                rows.append({"id": f"implicit_hate-{split}-{r.uid}", "post": task._state(Item(id="", gold=r.label, fields={"post": r.text}))[0]["post"], "label": r.label})
        dropped[split] = stats
        return rows

    train_pool, val_pool = pool(train_df, "train"), pool(val_df, "val")
    write_splits(
        HERE / "data",
        {"train": stratified(train_pool, args.train, args.seed), "val": stratified(val_pool, args.val, args.seed), "test": test},
        {
            "source": "Latent Hatred Stage-2 (data/raw/smarter/implicit-hate-corpus.zip), SMARTER split_train_test seed 123",
            "pools": {"train": len(train_pool), "val": len(val_pool), "dropped": dropped},
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()
