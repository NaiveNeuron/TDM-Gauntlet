"""Coverage check A, step 1: re-derive the corpus frame of the paper selection and draw the uniform sample.

Reads the ACL Anthology bulk export ``anthology+abstracts.bib.gz`` (download it from
https://aclanthology.org/anthology+abstracts.bib.gz into data/raw/coverage/), applies the selection's
venue/year filter (main-conference long/short/main and Findings volumes of ACL, EMNLP, NAACL and
EACL, 2024 through the 2026-09-26 cutoff) and writes

    data/coverage/frame.tsv      one row per frame record: anthology id, year, volume, entry type
    data/coverage/sample.jsonl   the uniform random sample (seed below) with title and abstract
    data/coverage/frame.json     export hash, parse and per-year counts, seed, sample size

Usage: uv run python scripts/coverage_frame.py
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "data/raw/coverage/anthology+abstracts.bib.gz"
OUT = ROOT / "data/coverage"

SEED = 20260929
N_SAMPLE = 300
# Export hash and counts of the corpus used to select the 24 papers (scripts/appendix/selection.json).
SELECTION_EXPORT_SHA = "dd753c17fe34a46d8bc9e3076d59a90335589b86e3ff226c841e99205a7373ad"
SELECTION_YEARS = {"2024": 5436, "2025": 7501, "2026": 5264}
SELECTION_RECORDS = 131152

VOLUME = re.compile(
    r"^(acl-long|acl-short|eacl-long|eacl-short|naacl-long|naacl-short|emnlp-main"
    r"|findings-(?:acl|emnlp|naacl|eacl))$"
)
ANTH_ID = re.compile(r"^(2024|2025|2026)\.([a-z-]+)\.(\d+)$")


def field(entry: str, name: str) -> str | None:
    """Value of a BibTeX field, quoted (`"..."`, ending at `",` + next field or the closing brace)
    or brace-delimited (`{...}`, matched by depth)."""
    m = re.search(rf"^\s{{4}}{name} = ", entry, re.M)
    if not m:
        return None
    i = m.end()
    if entry[i] == "{":
        depth = 0
        for j in range(i, len(entry)):
            depth += {"{": 1, "}": -1}.get(entry[j], 0)
            if depth == 0:
                return entry[i + 1 : j]
        return None
    q = re.compile(r'"(.*?)"(?:,\n\s{4}\w+ = |\n\})', re.S).match(entry, i)
    return q.group(1) if q else None


def clean(s: str | None) -> str:
    if s is None:
        return ""
    s = s.replace("{--}", "–").replace("{---}", "—").replace("{''}", "”").replace("{``}", "“")
    s = re.sub(r"\\(&|%|\$|_|#)", r"\1", s)
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    raw = EXPORT.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    text = gzip.decompress(raw).decode("utf-8")
    entries = [e for e in re.split(r"\n(?=@)", text) if e.startswith("@")]

    frame = []
    for e in entries:
        url = field(e, "url")
        if not url:
            continue
        aid = url.rstrip("/").rsplit("/", 1)[-1]
        m = ANTH_ID.match(aid)
        if not m or not VOLUME.match(m.group(2)):
            continue
        frame.append(
            {
                "id": aid,
                "year": m.group(1),
                "volume": m.group(2),
                "entry_type": re.match(r"@(\w+)", e).group(1),
                "title": clean(field(e, "title")),
                "abstract": clean(field(e, "abstract")),
                "url": f"https://aclanthology.org/{aid}/",
                "pdf": f"https://aclanthology.org/{aid}.pdf",
            }
        )
    frame.sort(key=lambda r: r["id"])
    years = collections.Counter(r["year"] for r in frame)

    sample = random.Random(SEED).sample(frame, N_SAMPLE)
    for i, r in enumerate(sample):
        r["sample_index"] = i

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "frame.tsv").open("w") as f:
        f.write("id\tyear\tvolume\tentry_type\n")
        for r in frame:
            f.write(f"{r['id']}\t{r['year']}\t{r['volume']}\t{r['entry_type']}\n")
    with (OUT / "sample.jsonl").open("w") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "export_url": "https://aclanthology.org/anthology+abstracts.bib.gz",
        "export_sha256": sha,
        "spec_export_sha256": SELECTION_EXPORT_SHA,
        "export_matches_spec": sha == SELECTION_EXPORT_SHA,
        "parsed_records": len(entries),
        "spec_parsed_records": SELECTION_RECORDS,
        "frame_records": len(frame),
        "frame_by_year": dict(sorted(years.items())),
        "spec_frame_by_year": SELECTION_YEARS,
        "frame_by_year_matches_spec": dict(years) == SELECTION_YEARS,
        "frame_by_entry_type": dict(collections.Counter(r["entry_type"] for r in frame)),
        "frame_missing_abstract": sum(1 for r in frame if not r["abstract"] and r["entry_type"] != "proceedings"),
        "seed": SEED,
        "sample_size": N_SAMPLE,
        "sampler": "random.Random(seed).sample(frame sorted by anthology id, n)",
        "sample_by_year": dict(sorted(collections.Counter(r["year"] for r in sample).items())),
    }
    (OUT / "frame.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
