# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Build jev-opt splits for LifeTox (Kim et al., NAACL 2024) from HF `mbkim/LifeTox`.

train/val: disjoint label-stratified samples of LifeTox_train (natural 77/23 safe/unsafe mix);
test: a label-stratified sample of LifeTox_test (released 1,500/1,500). No Jev calls.

    uv run jev-opt/examples/lifetox/prepare.py                      # download from Hugging Face
    uv run jev-opt/examples/lifetox/prepare.py --raw-dir data/raw/lifetox   # reuse local copies
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.request
from pathlib import Path

HF = "https://huggingface.co/datasets/mbkim/LifeTox/resolve/main/{name}"
HERE = Path(__file__).resolve().parent


def raw_rows(name: str, raw_dir: Path | None, cache: Path) -> tuple[list[dict], str]:
    path = (raw_dir / name) if raw_dir else cache / name
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        print(f"downloading {HF.format(name=name)}")
        urllib.request.urlretrieve(HF.format(name=name), path)
    data = path.read_bytes()
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()], hashlib.sha256(data).hexdigest()


def label_of(row: dict) -> str:
    return "safe" if str(row["is_safe"]).strip().lower() == "true" else "unsafe"


def stratified(indices_by_label: dict[str, list[int]], n: int, rng: random.Random, proportional: bool) -> list[int]:
    total = sum(len(v) for v in indices_by_label.values())
    labels = sorted(indices_by_label)
    if proportional:
        quota = {lab: round(n * len(indices_by_label[lab]) / total) for lab in labels}
    else:
        quota = {lab: n // len(labels) for lab in labels}
    quota[labels[0]] += n - sum(quota.values())
    picked: list[int] = []
    for lab in labels:
        pool = indices_by_label[lab]
        take = rng.sample(pool, quota[lab])
        taken = set(take)
        indices_by_label[lab] = [i for i in pool if i not in taken]
        picked += take
    rng.shuffle(picked)
    return picked


def by_label(rows: list[dict]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        if str(row.get("query", "")).strip() and str(row.get("response", "")).strip():
            out.setdefault(label_of(row), []).append(i)
    return out


def write(path: Path, rows: list[dict], picked: list[int], split: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open("w", encoding="utf-8") as fh:
        for i in picked:
            row = rows[i]
            lab = label_of(row)
            counts[lab] = counts.get(lab, 0) + 1
            fh.write(json.dumps({"id": f"lifetox-{split}-{i}", "question": row["query"], "answer": row["response"], "label": lab}, ensure_ascii=False) + "\n")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, help="directory holding LifeTox_train.jsonl and LifeTox_test.jsonl")
    ap.add_argument("--out", type=Path, default=HERE / "data")
    ap.add_argument("--train", type=int, default=200)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    train_rows, train_sha = raw_rows("LifeTox_train.jsonl", args.raw_dir, args.out / "raw")
    test_rows, test_sha = raw_rows("LifeTox_test.jsonl", args.raw_dir, args.out / "raw")
    pool = by_label(train_rows)
    train_idx = stratified(pool, args.train, rng, proportional=True)
    val_idx = stratified(pool, args.val, rng, proportional=True)  # drawn from what train left: disjoint
    test_idx = stratified(by_label(test_rows), args.test, rng, proportional=True)
    manifest = {
        "source": "https://huggingface.co/datasets/mbkim/LifeTox",
        "raw_sha256": {"LifeTox_train.jsonl": train_sha, "LifeTox_test.jsonl": test_sha},
        "seed": args.seed,
        "counts": {
            "train": write(args.out / "train.jsonl", train_rows, train_idx, "train"),
            "val": write(args.out / "val.jsonl", train_rows, val_idx, "train"),
            "test": write(args.out / "test.jsonl", test_rows, test_idx, "test"),
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["counts"]))


if __name__ == "__main__":
    main()
