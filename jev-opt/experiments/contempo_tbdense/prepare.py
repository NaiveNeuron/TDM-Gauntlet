# /// script
# requires-python = ">=3.11"
# dependencies = ["nltk==3.9.1", "httpx", "numpy", "scikit-learn", "scipy"]
# ///
"""jev-opt splits for ConTempo TB-Dense (gauntlet task `contempo/tbdense`). No Jev calls.

train/val: the standard TB-Dense train (22 docs) and dev (5 docs) document lists
(rujunhan/EMNLP-2019 data/tbd/*_docs.txt), event-event pairs built with the gauntlet's
own builder (scripts/prepare_contempo.py: same TimeML source, sentence splitter, context
window and trigger marking). val = the whole dev split; train = a label-stratified sample
of the train split. test = `contempo/tbdense` items() (1,427 pairs, 9 docs).

    uv run jev-opt/experiments/contempo_tbdense/prepare.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import REPO, gauntlet_task, stratified, test_rows, write_splits  # noqa: E402

import prepare_contempo as pc  # noqa: E402  (gauntlet's builder, scripts/ on sys.path)

DOC_LISTS = REPO / "data" / "raw" / "contempo" / "EMNLP-2019" / "data" / "tbd"


def split_rows(doc_ids: set[str], split: str) -> list[dict]:
    docs: dict[str, pc.Doc] = {}
    rows = []
    for line in pc.TBDENSE_FULL.read_text().splitlines():
        if not line.strip():
            continue
        doc_id, id1, id2, code = line.split("\t")
        if doc_id not in doc_ids or not (id1.startswith("e") and id2.startswith("e")):
            continue
        if doc_id not in docs:
            docs[doc_id] = pc.load_doc(pc.TIMEBANK_DIR / f"{doc_id}.tml")
        doc = docs[doc_id]
        if id1 not in doc.events or id2 not in doc.events:  # same TimeBank-event restriction as the test set
            continue
        ev1, ev2 = doc.events[id1], doc.events[id2]
        context, _ = pc.build_context(doc, ev1, ev2)
        rows.append({"id": f"{split}:{doc_id}:{id1}:{id2}", "context": context, "event_1": ev1.text, "event_2": ev2.text, "label": pc.TBDENSE_CODES[code]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=500)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    lists = {s: set((DOC_LISTS / f"{s}_docs.txt").read_text().split()) for s in ("train", "dev", "test")}
    assert lists["test"] == set(pc.TBDENSE_TEST_DOCS) and not (lists["train"] | lists["dev"]) & lists["test"]
    train_pool, dev_pool = split_rows(lists["train"], "train"), split_rows(lists["dev"], "dev")
    test = test_rows(
        gauntlet_task("contempo/tbdense"),
        "contempo/tbdense",
        lambda it, st: {"id": it.id, **st, "label": it.gold},
    )
    write_splits(
        HERE / "data",
        {"train": stratified(train_pool, args.train, args.seed), "val": stratified(dev_pool, args.val, args.seed), "test": test},
        {
            "source": "TimeBank-Dense (Cassidy et al. 2014) via data/raw/contempo; standard train/dev/test doc lists",
            "pools": {"train_split_pairs": len(train_pool), "dev_split_pairs": len(dev_pool)},
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()
