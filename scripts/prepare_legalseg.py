"""Freeze the LegalSeg test set (Nigam et al., Findings NAACL 2025).

Source: the authors' Hugging Face release. The canonical per-sentence copy,
`L-NLProc/LegalSeg_CSV` (test.csv), is gated. The same test split, in the same
order, is published ungated (Apache-2.0) by the authors as the Index/Text/Label
columns of `L-NLProc/LegalSeg_RhetoricLLaMA_Predictions`
(RhetoricLLaMA_predictions.csv); the extra `llama_p` column (that model's raw
generations) is ignored. This script:

  1. downloads that CSV (pinned revision) to data/raw/legalseg/ if absent and
     checks its SHA-256;
  2. checks the split against paper Table 2 (712 documents, 149,881 sentences,
     exact per-label test counts) and checks the gold sequence against the gold
     column of the authors' ToInLegalBERT prediction file (same order);
  3. writes data/frozen/legalseg/test.jsonl: one row per document, in release
     order, with id, doc_index, n_sentences, gold (one digit per sentence,
     paper label id: None-0 Facts-1 Issue-2 AoP-3 AoR-4 Reasoning-5
     Decision-6) and sha256 over [texts, labels]. No text is redistributed.

    uv run --with huggingface_hub python scripts/prepare_legalseg.py

English check (the benchmark's English-only rule): every sentence is ASCII-only (no non-Latin
script); langid.py 1.1.6 flagged 852 of the 137,922 sentences with >= 4 words
and a romanised-Hindi keyword scan covered all sentences; every flag was
judged from the text alone (see data/manifests/legalseg.json).
The six sentences in EXCLUDED (transliterated Hindi quoted speech) are written
to excluded_non_english.jsonl (doc, position, text SHA-256, reason) and dropped
by the task adapter.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gauntlet.task import write_jsonl  # noqa: E402
from gauntlet.tasks.legalseg import LABELS, RAW_CSV, RAW_CSV_SHA256, doc_sha256, read_raw  # noqa: E402

RELEASE = ("L-NLProc/LegalSeg_RhetoricLLaMA_Predictions", "686d8082bdadc4f08cf21ff068b561ac38a88cc4", "RhetoricLLaMA_predictions.csv")
CHECK = ("L-NLProc/LegalSeg_ToInLegalBert_Predictions", "9c68cfbd231c3bbf46cd851825c8b36a85204469", "ToInlegalbertpredictions.csv")
CHECK_CSV = RAW_CSV.parent / "ToInlegalbertpredictions.csv"
CHECK_SHA256 = "9459d0232675ad2bccbaa7724ddd91c2fc390ce5de62262feec0fafbadf27b12"
OUT = REPO / "data" / "frozen" / "legalseg"

# Paper Table 2, test column.
EXPECTED_DOCS = 712
EXPECTED_SENTENCES = 149_881
EXPECTED_LABELS = {
    "Facts": 24_909,
    "Issue": 1_843,
    "Arguments of Petitioner": 14_520,
    "Arguments of Respondent": 9_579,
    "Reasoning": 36_689,
    "Decision": 3_841,
    "None": 58_500,
}
# Frozen, adjudicated non-English sentences: (doc Index, 0-based position, reason).
EXCLUDED = [
    ("7017", 11, "romanised Hindi clause (quoted confession) inside the sentence"),
    ("7017", 54, "ends in romanised Hindi clause (quoted confession)"),
    ("7017", 55, "romanised Hindi sentence (quoted confession)"),
    ("7017", 56, "romanised Hindi sentence (quoted confession)"),
    ("7017", 57, "romanised Hindi clause (quoted statement) inside the sentence"),
    ("7049", 617, "romanised Hindi sentence (quoted speech)"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(repo_rev_file: tuple[str, str, str], dest: Path, sha256: str) -> None:
    if not dest.exists():
        from huggingface_hub import hf_hub_download

        repo, rev, name = repo_rev_file
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(hf_hub_download(repo, name, repo_type="dataset", revision=rev), dest)
    got = sha256_file(dest)
    if got != sha256:
        raise SystemExit(f"{dest}: SHA-256 {got} != frozen {sha256}")


def main() -> None:
    fetch(RELEASE, RAW_CSV, RAW_CSV_SHA256)
    fetch(CHECK, CHECK_CSV, CHECK_SHA256)
    docs = read_raw(RAW_CSV)

    # Documents are contiguous runs in the release (read_raw would merge a split run).
    with RAW_CSV.open(encoding="utf-8", newline="") as fh:
        csv.field_size_limit(1 << 30)
        index = [row["Index"] for row in csv.DictReader(fh)]
    runs = 1 + sum(a != b for a, b in zip(index, index[1:]))
    assert runs == len(docs) == EXPECTED_DOCS, (runs, len(docs))

    labels = [lab for _, labs in docs.values() for lab in labs]
    assert len(labels) == EXPECTED_SENTENCES, len(labels)
    assert Counter(labels) == Counter(EXPECTED_LABELS), Counter(labels)
    assert set(labels) <= set(LABELS)

    with CHECK_CSV.open(encoding="utf-8", newline="") as fh:
        check_gold = [LABELS[int(row["true"])] for row in csv.DictReader(fh)]
    assert check_gold == labels, "gold differs from the ToInLegalBERT prediction file's gold column"

    rows = []
    for doc_index, (texts, labs) in docs.items():
        rows.append(
            {
                "id": f"doc{doc_index}",
                "doc_index": doc_index,
                "n_sentences": len(texts),
                "gold": "".join(str(LABELS.index(lab)) for lab in labs),
                "sha256": doc_sha256(texts, labs),
            }
        )
    write_jsonl(OUT / "test.jsonl", rows)
    excluded = [
        {
            "doc_index": doc_index,
            "sentence": pos,
            "gold": docs[doc_index][1][pos],
            "sha256": hashlib.sha256(docs[doc_index][0][pos].encode("utf-8")).hexdigest(),
            "reason": reason,
        }
        for doc_index, pos, reason in EXCLUDED
    ]
    write_jsonl(OUT / "excluded_non_english.jsonl", excluded)
    print(f"excluded {len(excluded)} non-English sentences: {Counter(x['gold'] for x in excluded)}")
    print(f"wrote {len(rows)} documents / {len(labels)} sentences to {OUT / 'test.jsonl'}")
    print(json.dumps(Counter(labels), indent=1))


if __name__ == "__main__":
    main()
