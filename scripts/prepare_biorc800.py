"""Freeze the BIORC800 (Findings of EMNLP 2024) test and dev splits; fit dev thresholds.

The release (github.com/ScienceNLP-Lab/LLM-SSC, data_llm/biorc800/) has no licence,
so rows are NOT redistributed.

    uv run python scripts/prepare_biorc800.py            # freeze
    uv run python scripts/prepare_biorc800.py fit-thresholds [runs/biorc800/dev/predictions.jsonl]

`freeze` downloads test.jsonl and dev.jsonl at the pinned commit to
data/raw/biorc800/release/ (skipped if present), checks their SHA-256, and writes
data/frozen/biorc800/<split>/manifest.jsonl: one line per abstract,
{"id": abstract_id, "row", "n_sentences", "sha256"} (see gauntlet.tasks.biorc800.row_sha256).
Deterministic.

`fit-thresholds` reads a completed `gauntlet run biorc800/dev` and writes
data/frozen/biorc800/dev_thresholds.json with the paper's per-label dynamic thresholds
(release find_best_thresholds). Must be run BEFORE the test run.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gauntlet.task import read_jsonl, write_jsonl  # noqa: E402
from gauntlet.tasks.biorc800 import (  # noqa: E402
    COMMIT,
    LABELS,
    SPLITS,
    THRESHOLDS_PATH,
    fit_thresholds,
    frozen_manifest_path,
    raw_path,
    raw_rows,
    raw_url,
    row_sha256,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> None:
    for split, spec in SPLITS.items():
        path = raw_path(split)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading {raw_url(split)}", file=sys.stderr)
            with urllib.request.urlopen(raw_url(split)) as resp:
                path.write_bytes(resp.read())
        got = sha256_file(path)
        if got != spec["sha256"]:
            sys.exit(f"{path}: sha256 {got} != expected {spec['sha256']}")
        rows = raw_rows(split)
        if len(rows) != spec["n"]:
            sys.exit(f"{path}: {len(rows)} abstracts, expected {spec['n']}")
        manifest = [
            {"id": row["abstract_id"], "row": i, "n_sentences": len(row["sentences"]), "sha256": row_sha256(row)}
            for i, row in enumerate(rows)
        ]
        write_jsonl(frozen_manifest_path(split), manifest)
        n_sent = sum(m["n_sentences"] for m in manifest)
        print(f"{split}: {len(manifest)} abstracts, {n_sent} sentences -> {frozen_manifest_path(split).relative_to(REPO)}", file=sys.stderr)


def fit(predictions_path: Path) -> None:
    rows = read_jsonl(predictions_path)
    expected = {m["id"] for m in read_jsonl(frozen_manifest_path("dev"))}
    if {r["item_id"] for r in rows} != expected:
        sys.exit(f"{predictions_path}: item ids do not match the frozen dev manifest (smoke run?)")
    probs = [p for r in rows for p in r["prediction"]["probs"]]
    gold = [labels for r in rows for labels in r["gold"]]
    thresholds = fit_thresholds(probs, gold)
    out = {
        "_source": "Per-label thresholds fit on the released BIORC800 dev split (159 abstracts) with the paper's "
        "dynamic thresholding (Section 3.3; LLM-SSC@" + COMMIT[:7] + " find_best_thresholds: grid 0.0..0.9 step 0.1, "
        "per-label binary F1, first maximum wins, 0 -> default 0.4). Never fit on test.",
        "dev_predictions": str(predictions_path.relative_to(REPO) if predictions_path.is_relative_to(REPO) else predictions_path),
        "dev_predictions_sha256": sha256_file(predictions_path),
        "n_dev_sentences": len(gold),
        "thresholds": {label: thresholds[label] for label in LABELS},
    }
    THRESHOLDS_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["thresholds"]), "->", THRESHOLDS_PATH, file=sys.stderr)


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] == "freeze":
        freeze()
    elif args[0] == "fit-thresholds":
        fit(Path(args[1]).resolve() if len(args) > 1 else REPO / "runs" / "biorc800" / "dev" / "predictions.jsonl")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
