"""Freeze the DREsS_New test split as an ID + SHA-256 manifest.

DREsS is gated behind a consent form whose terms forbid re-sharing, so the frozen
files hold only essay ids, file row indices and row hashes; `items()` in
src/gauntlet/tasks/dress.py rebuilds the essays and scores from the raw file.

Download first (manual, by a person who accepts the terms): submit the consent
form linked from https://haneul-yoo.github.io/dress/ and put the DREsS_New file
(e.g. DREsS_New.tsv) under data/raw/dress_new/. Then:
    uv run python scripts/prepare_dress_new.py

Frozen rules (see the module docstring):
  * split: sklearn train_test_split(range(n), test_size=0.2, random_state=22) over
    all rows in file order (paper Appendix A: 6:2:2, seed 22);
  * test rows with an empty essay or a missing rubric score are dropped;
  * English-only: test rows whose prompt+essay letters are >5% non-Latin script
    are dropped; rows with any non-Latin letter are listed for inspection.
Writes data/frozen/dress_new/test_ids.jsonl and data/frozen/dress_new/stats.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from gauntlet.task import write_jsonl
from gauntlet.tasks.dress import (
    FROZEN_DIR,
    LEVELS,
    MAX_NON_LATIN_SHARE,
    RAW_DIR,
    RUBRICS,
    find_raw_file,
    load_raw,
    non_latin_share,
    row_sha256,
    test_row_indices,
)

PAPER_N = 2279


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=FROZEN_DIR)
    args = parser.parse_args()

    raw = find_raw_file(args.raw_dir)
    rows = load_raw(raw)
    test = [rows[i] for i in test_row_indices(len(rows))]

    dropped_missing = [r["id"] for r in test if not r["essay"].strip() or any(r[k] is None for k in RUBRICS)]
    shares = {r["id"]: non_latin_share(r["prompt"] + " " + r["essay"]) for r in test}
    flagged = {i: round(s, 4) for i, s in shares.items() if s > 0}
    dropped_non_english = [i for i, s in shares.items() if s > MAX_NON_LATIN_SHARE and i not in dropped_missing]
    drop = set(dropped_missing) | set(dropped_non_english)
    kept = [r for r in test if r["id"] not in drop]

    off_grid = sorted({r[k] for r in rows for k in RUBRICS if r[k] is not None and r[k] not in LEVELS})
    total_mismatch = [
        r["id"]
        for r in rows
        if r["total"] is not None and all(r[k] is not None for k in RUBRICS) and abs(r["total"] - sum(r[k] for k in RUBRICS)) > 1e-9
    ]
    stats = {
        "raw_file": str(raw.relative_to(args.raw_dir)),
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "n_rows": len(rows),
        "n_rows_matches_paper_2279": len(rows) == PAPER_N,
        "n_test_split": len(test),
        "n_test_frozen": len(kept),
        "dropped_missing": dropped_missing,
        "dropped_non_english": dropped_non_english,
        "non_latin_flagged_for_inspection": flagged,
        "off_grid_scores_all_rows": off_grid,
        "total_column_mismatch_ids": total_mismatch,
        "n_prompts_test": len({r["prompt"] for r in kept}),
        "gold_distribution_test": {k: dict(sorted(Counter(r[k] for r in kept).items())) for k in RUBRICS},
    }
    write_jsonl(args.out_dir / "test_ids.jsonl", [{"id": r["id"], "row_index": r["row_index"], "sha256": row_sha256(r)} for r in kept])
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "non_latin_flagged_for_inspection"}, indent=2))
    print("test_ids.jsonl sha256", hashlib.sha256((args.out_dir / "test_ids.jsonl").read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
