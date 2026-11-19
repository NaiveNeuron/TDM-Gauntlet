"""Freeze the MISMATCHED scientific NLI test set (Shaik et al., Findings ACL 2025).

Source: https://github.com/fshaik8/MisMatched (commit
8d5037227c9d680a2282e2bdc41ec66ef382910e) README dataset link, Google Drive file
15GSYcakchdRmKlNoWIigGbCoI7eQLAUR (zip with dev.tsv, test.tsv). Licence: CC BY-SA 4.0
(README "License" section), so the test rows are redistributed as JSONL with
attribution under the same licence.

    uv run python scripts/prepare_mismatched.py

downloads the zip into data/raw/mismatched/ if missing, verifies its SHA-256 and the
test.tsv SHA-256, and writes data/frozen/mismatched/test.jsonl.

test.tsv is parsed with csv.DictReader(delimiter="\\t") (standard double-quote
quoting), which is field-for-field identical to the authors' loader
`pd.read_csv(path, sep="\\t")` (MisMatched_Zero_Shot.py). Rows are kept verbatim and
in file order. Item id = "mismatched-<sample_ID_F>" (sample_ID_F is unique; sample_ID
repeats across domains). The release's label "entailing" is renamed to the paper's
class name "entailment"; the released string is kept in `label_release`.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "mismatched"
ZIP = RAW / "mismatched.zip"
ZIP_URL = "https://drive.usercontent.google.com/download?id=15GSYcakchdRmKlNoWIigGbCoI7eQLAUR&export=download&confirm=t"
ZIP_SHA256 = "82c558869b851506b3e85bd5cee86f2f78088babca0c006098db47b9b690494f"
TEST_MEMBER = "test.tsv"
TEST_SHA256 = "7b3e5e92e2e97018e7f7848c80c887da46ca7c0ae5ab8f274aa54c27cfb07366"
OUT = REPO / "data" / "frozen" / "mismatched" / "test.jsonl"
EXPECTED_N = 2400
LABEL_MAP = {"entailing": "entailment", "reasoning": "reasoning", "contrasting": "contrasting", "neutral": "neutral"}
DOMAINS = ("Psychology", "Engineering", "Public Health")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if not ZIP.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(ZIP_URL) as resp:
            ZIP.write_bytes(resp.read())
    if (digest := sha256(ZIP.read_bytes())) != ZIP_SHA256:
        raise SystemExit(f"unexpected zip SHA-256 {digest}")
    with zipfile.ZipFile(ZIP) as zf:
        raw = zf.read(TEST_MEMBER)
    if (digest := sha256(raw)) != TEST_SHA256:
        raise SystemExit(f"unexpected {TEST_MEMBER} SHA-256 {digest}")

    rows = []
    for src in csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""), delimiter="\t"):
        if src["label"] not in LABEL_MAP:
            raise SystemExit(f"{src['sample_ID_F']}: unknown label {src['label']!r}")
        if src["Domain"] not in DOMAINS:
            raise SystemExit(f"{src['sample_ID_F']}: unknown domain {src['Domain']!r}")
        if not src["sentence1"] or not src["sentence2"]:
            raise SystemExit(f"{src['sample_ID_F']}: empty sentence")
        rows.append(
            {
                "id": f"mismatched-{src['sample_ID_F']}",
                "sample_ID_F": src["sample_ID_F"],
                "sample_ID": src["sample_ID"],
                "domain": src["Domain"],
                "paper_id": src["paper_id"],
                "sentence1": src["sentence1"],
                "sentence2": src["sentence2"],
                "label_release": src["label"],
                "gold": LABEL_MAP[src["label"]],
            }
        )
    if len(rows) != EXPECTED_N:
        raise SystemExit(f"expected {EXPECTED_N} rows, got {len(rows)}")
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("duplicate item ids")
    cells = Counter((r["domain"], r["gold"]) for r in rows)
    if set(cells.values()) != {200} or len(cells) != 12:
        raise SystemExit(f"expected 200 per domain x label, got {dict(cells)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {OUT.relative_to(REPO)}")
    print("domains", dict(Counter(r["domain"] for r in rows)))
    print("labels", dict(Counter(r["gold"] for r in rows)))
    print("sha256", sha256(OUT.read_bytes()))


if __name__ == "__main__":
    main()
