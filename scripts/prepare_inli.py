"""Freeze the INLI English test set from the official release.

Source: https://github.com/google-deepmind/inli (CC BY-SA 4.0), file
`INLI Data/test.csv` at commit e953ca29a6e404d5203401b66ff51fee88fa1ec6.
Each CSV row is one premise with its four hypotheses (columns
implied_entailment, explicit_entailment, neutral, contradiction); 1,000 premises
-> 4,000 premise-hypothesis pairs, which is the paper's Table 6 test set.

    uv run python scripts/prepare_inli.py

Downloads the pinned CSV to data/raw/inli/test.csv if absent, checks its SHA-256,
and writes data/frozen/inli/test.jsonl: one row per pair, premise groups in file
order, hypotheses in column order. Texts are kept verbatim (including one trailing
space in row 865). Item id = inli-test-<row:04d>-<label>; the id never reaches Jev.
The frozen file is a redistribution of CC BY-SA 4.0 material and carries that licence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMMIT = "e953ca29a6e404d5203401b66ff51fee88fa1ec6"
URL = f"https://raw.githubusercontent.com/google-deepmind/inli/{COMMIT}/" + urllib.parse.quote("INLI Data/test.csv")
RAW = REPO / "data" / "raw" / "inli" / "test.csv"
RAW_SHA256 = "28153a6f601ccaacf70e1f091c35548a7d72cf196b9462e0fe44ca1062131cf2"
OUT = REPO / "data" / "frozen" / "inli" / "test.jsonl"
LABELS = ("implied_entailment", "explicit_entailment", "neutral", "contradiction")
EXPECTED_PREMISES = 1000
EXPECTED_SOURCES = {"circa": 436, "socialchem": 256, "normbank": 254, "ludwig": 54}


def main() -> None:
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, RAW)
    digest = hashlib.sha256(RAW.read_bytes()).hexdigest()
    if digest != RAW_SHA256:
        raise SystemExit(f"unexpected raw SHA-256 {digest}")
    with RAW.open(encoding="utf-8", newline="") as fh:
        src_rows = list(csv.DictReader(fh))
    if len(src_rows) != EXPECTED_PREMISES:
        raise SystemExit(f"expected {EXPECTED_PREMISES} premises, got {len(src_rows)}")
    sources = Counter(r["dataset"] for r in src_rows)
    if dict(sources) != EXPECTED_SOURCES:
        raise SystemExit(f"unexpected source-dataset counts {dict(sources)}")
    rows = []
    for i, src in enumerate(src_rows):
        if int(src[""]) != i:
            raise SystemExit(f"row {i}: index column is {src['']!r}")
        for label in LABELS:
            if not src[label].strip() or not src["premise"].strip():
                raise SystemExit(f"row {i}: empty field")
            rows.append(
                {
                    "id": f"inli-test-{i:04d}-{label}",
                    "group": i,
                    "source_dataset": src["dataset"],
                    "premise": src["premise"],
                    "hypothesis": src[label],
                    "gold": label,
                }
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} pairs ({len(src_rows)} premises) to {OUT.relative_to(REPO)}")
    print("sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
