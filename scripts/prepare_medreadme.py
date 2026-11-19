"""Freeze the MedReadMe sentence-readability test split as an ID + SHA-256 manifest.

Downloads dataset/readability.csv from chaojiang06/medreadme at a pinned commit
into data/raw/medreadme/ (verified by SHA-256), selects the paper's test split
(split == "test", 1,140 rows), runs the English screen and writes
data/frozen/medreadme/test_manifest.jsonl (id, row_index, sha256) plus
data/frozen/medreadme/stats.json. The release has no licence, so sentences and
ratings are not stored; gauntlet.tasks.medreadme.items() rebuilds them from
the raw CSV and checks every row hash.

    uv run --with langid==1.1.6 python scripts/prepare_medreadme.py

English screen: langid 1.1.6 (all languages, normalised probabilities) on every
test sentence plus a census of non-ASCII characters and a non-Latin-letter check.
Every flag was inspected; rows are dropped only if listed in
NON_ENGLISH_IDS (a frozen list; empty for this release).
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
import unicodedata
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gauntlet.task import write_jsonl  # noqa: E402
from gauntlet.tasks.medreadme import (  # noqa: E402
    CSV_SHA256,
    CSV_URL,
    FROZEN_DIR,
    RAW_CSV,
    SOURCE_GROUP,
    item_id,
    load_raw,
    row_sha256,
    sha256_file,
)

PAPER_TEST_N = 1140
# Non-English test rows to drop (none: all langid flags are English
# headings or fragments with abbreviations).
NON_ENGLISH_IDS: frozenset[str] = frozenset()


def fetch() -> None:
    if RAW_CSV.exists() and sha256_file(RAW_CSV) == CSV_SHA256:
        return
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(CSV_URL, RAW_CSV)
    got = sha256_file(RAW_CSV)
    if got != CSV_SHA256:
        raise SystemExit(f"{RAW_CSV}: sha256 {got} != pinned {CSV_SHA256}")


def main() -> None:
    from langid.langid import LanguageIdentifier, model

    fetch()
    rows = load_raw(RAW_CSV)
    assert [int(r[""]) for r in rows] == list(range(len(rows))), "row ids are not 0..n-1 in file order"
    test = [r for r in rows if r["split"] == "test"]
    assert len(test) == PAPER_TEST_N, len(test)
    unknown = {r["Source"] for r in rows} - set(SOURCE_GROUP)
    assert not unknown, unknown

    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    langid_flags, non_latin = {}, {}
    non_ascii = collections.Counter()
    for r in test:
        lang, prob = identifier.classify(r["Sentence"])
        if lang != "en":
            langid_flags[item_id(r)] = {"lang": lang, "prob": round(prob, 3)}
        letters = [c for c in r["Sentence"] if c.isalpha()]
        foreign = [c for c in letters if "LATIN" not in unicodedata.name(c, "")]
        if foreign:
            non_latin[item_id(r)] = "".join(foreign)
        non_ascii.update(c for c in r["Sentence"] if ord(c) > 127)

    kept = [r for r in test if item_id(r) not in NON_ENGLISH_IDS]
    write_jsonl(
        FROZEN_DIR / "test_manifest.jsonl",
        [{"id": item_id(r), "row_index": int(r[""]), "sha256": row_sha256(r)} for r in kept],
    )
    split_n = collections.Counter(r["split"] for r in rows)
    stats = {
        "raw_csv_sha256": CSV_SHA256,
        "n_rows": len(rows),
        "split_counts": dict(sorted(split_n.items())),
        "n_test_split": len(test),
        "n_test_frozen": len(kept),
        "dropped_non_english": sorted(NON_ENGLISH_IDS),
        "langid_non_en_flags_checked_english": langid_flags,
        "non_latin_letters": non_latin,
        "non_ascii_chars": dict(non_ascii.most_common()),
        "test_n_by_source": dict(sorted(collections.Counter(r["Source"] for r in kept).items())),
        "test_n_by_group": dict(collections.Counter(SOURCE_GROUP[r["Source"]] for r in kept)),
        "test_n_by_side": dict(sorted(collections.Counter(r["Side"] for r in kept).items())),
        "test_duplicate_sentences": sum(c - 1 for c in collections.Counter(r["Sentence"] for r in kept).values() if c > 1),
        "test_sentences_also_in_train_or_dev": len({r["Sentence"] for r in kept} & {r["Sentence"] for r in rows if r["split"] != "test"}),
        "test_max_sentence_chars": max(len(r["Sentence"]) for r in kept),
    }
    (FROZEN_DIR / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k not in {"langid_non_en_flags_checked_english", "non_ascii_chars"}}, indent=2))
    print("test_manifest.jsonl sha256", hashlib.sha256((FROZEN_DIR / "test_manifest.jsonl").read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
