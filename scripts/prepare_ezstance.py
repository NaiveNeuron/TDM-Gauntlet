"""Freeze the EZ-STANCE test sets (Zhao & Caragea, ACL 2024).

Source: the dataset zip linked from https://github.com/chenyez/EZ-STANCE
(README, commit b53dc64ce29622dd845c933972fdfbc17a0210ba), Google Drive file
1WxysM2VwjcZkHnCDMe7P9bwA8VW7IU01. The release carries no licence, so no tweet
text is redistributed: the frozen files hold row coordinates, gold labels and a
per-row SHA-256, and the task adapter rebuilds text/target from the zip.

    uv run python scripts/prepare_ezstance.py

downloads the zip to data/raw/ezstance/ezstance_data.zip if absent (via
`uvx gdown`), checks its SHA-256, and writes data/frozen/ezstance/:
  subtaskA.jsonl   target-based test set (noun_phrase + claim files)
  subtaskB.jsonl   the eight leave-one-domain-out test sets
  excluded_non_english.jsonl  test rows dropped by the frozen language list

Each test set is read from its noun_phrase/ and claim/ files, which fix the
target type; the release's mixed/ file is checked to be exactly their union
(multiset of (Text, Target 1, Stance 1)). Only rows with `In Use` == 1 are
used by the authors' loader for Subtask B; every released test row has it.
Rows whose `Text` hash is in data/frozen/ezstance/non_english_texts.json (the
frozen, adjudicated language screen) are excluded.
Item id = "<A|B>-[<domain abbr>-]<np|claim>-<0-based row in its CSV>".
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW_ZIP = REPO / "data" / "raw" / "ezstance" / "ezstance_data.zip"
RAW_ZIP_SHA256 = "4683d124c8c8dde1b1282c3f534efa6c0b92e22335a8821249ef1d63cb26455b"
GDRIVE_ID = "1WxysM2VwjcZkHnCDMe7P9bwA8VW7IU01"
OUT = REPO / "data" / "frozen" / "ezstance"
LABELS = {"AGAINST", "FAVOR", "NONE"}
TYPES = {"noun_phrase": "np", "claim": "claim"}
DOMAINS = {
    "covid19_domain": "CE",
    "world_event_domain": "WE",
    "education_and_culture_domain": "EdC",
    "consumption_and_entertainment_domain": "EnC",
    "sports_domain": "S",
    "rights_domain": "R",
    "environmental_protection_domain": "EP",
    "politic": "P",
}
# Paper Table 5 (Subtask A) and Table 13 (Subtask B): test N per target type.
EXPECTED = {
    ("subtaskA", None): (2663, 5135),
    ("subtaskB", "covid19_domain"): (2639, 3819),
    ("subtaskB", "world_event_domain"): (2286, 3252),
    ("subtaskB", "education_and_culture_domain"): (2109, 3078),
    ("subtaskB", "consumption_and_entertainment_domain"): (2702, 4053),
    ("subtaskB", "sports_domain"): (1807, 2661),
    ("subtaskB", "rights_domain"): (2468, 3405),
    ("subtaskB", "environmental_protection_domain"): (2071, 2772),
    ("subtaskB", "politic"): (2890, 3999),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def row_sha256(text: str, target: str, gold: str) -> str:
    canonical = json.dumps([text, target, gold], ensure_ascii=False, separators=(",", ":"))
    return sha256_bytes(canonical.encode("utf-8"))


def read_member(zf: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    with zf.open(member) as fh:
        return list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", newline="")))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    if not RAW_ZIP.exists():
        RAW_ZIP.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["uvx", "gdown", GDRIVE_ID, "-O", str(RAW_ZIP)], check=True)
    digest = sha256_bytes(RAW_ZIP.read_bytes())
    assert digest == RAW_ZIP_SHA256, f"zip SHA-256 {digest} != {RAW_ZIP_SHA256}"

    screen = json.loads((OUT / "non_english_texts.json").read_text())
    non_english = {r["text_sha256"] for r in screen["texts"]}
    matched_texts: set[str] = set()

    splits: dict[str, list[dict]] = {"subtaskA": [], "subtaskB": []}
    excluded: list[dict] = []
    with zipfile.ZipFile(RAW_ZIP) as zf:
        for (split, domain), expected in EXPECTED.items():
            base = split if domain is None else None
            counts = []
            union: Counter = Counter()
            for target_type, abbr in TYPES.items():
                member = (
                    f"subtaskA/{target_type}/raw_test_all_onecol.csv"
                    if base
                    else f"subtaskB/{target_type}/{domain}/raw_test_all_onecol.csv"
                )
                rows = read_member(zf, member)
                counts.append(len(rows))
                prefix = "A" if base else f"B-{DOMAINS[domain]}"
                for i, r in enumerate(rows):
                    text, target, gold = r["Text"], r["Target 1"], r["Stance 1"]
                    assert gold in LABELS and text.strip() and target.strip(), (member, i)
                    if domain is not None:
                        assert r["In Use"] == "1" and r["Domain"] == domain, (member, i)
                    union[(text, target, gold)] += 1
                    rec = {
                        "id": f"{prefix}-{abbr}-{i:05d}",
                        "member": member,
                        "row": i,
                        "target_type": target_type,
                        **({"domain": domain} if domain else {}),
                        "gold": gold,
                        "sha256": row_sha256(text, target, gold),
                    }
                    text_hash = sha256_bytes(text.encode("utf-8"))
                    if text_hash in non_english:
                        matched_texts.add(text_hash)
                        excluded.append({"split": split, **rec, "text_sha256": text_hash})
                    else:
                        splits[split].append(rec)
            assert tuple(counts) == expected, (split, domain, counts, expected)
            mixed = (
                "subtaskA/mixed/raw_test_all_onecol.csv"
                if base
                else f"subtaskB/mixed/{domain}/raw_test_all_onecol.csv"
            )
            mixed_rows = Counter((r["Text"], r["Target 1"], r["Stance 1"]) for r in read_member(zf, mixed))
            assert mixed_rows == union, f"{mixed} is not the union of its noun_phrase and claim files"

    assert matched_texts == non_english, "language list holds hashes absent from the test files"
    OUT.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        write_jsonl(OUT / f"{split}.jsonl", rows)
    write_jsonl(OUT / "excluded_non_english.jsonl", excluded)

    for split, rows in splits.items():
        path = OUT / f"{split}.jsonl"
        by = Counter((r.get("domain", "-"), r["target_type"]) for r in rows)
        ex = Counter((r.get("domain", "-"), r["target_type"]) for r in excluded if r["split"] == split)
        print(split, len(rows), "excluded", sum(ex.values()), sha256_bytes(path.read_bytes()))
        for key in sorted(set(by) | set(ex)):
            print("  ", key, "kept", by[key], "excluded", ex[key])
        print("   labels", dict(Counter(r["gold"] for r in rows)))
    print("excluded_non_english.jsonl", sha256_bytes((OUT / "excluded_non_english.jsonl").read_bytes()))


if __name__ == "__main__":
    main()
