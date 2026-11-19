"""Freeze the SciPrompt English test sets (arXiv, S2ORC, SDPRA 2021, Emerging NLP).

Source: the authors' data release linked from github.com/zhiwenyou103/SciPrompt@cc6db91
README ("Our datasets can be downloaded here"), Google Drive file
1w5IxtfayNPlrAE6I_vp2zKhx8ntDMRMv (sciprompt_data.zip, sha256 2b0a53d7...).

Download first (writes data/raw/sciprompt/sciprompt_data.zip):
    uv run --with gdown gdown 1w5IxtfayNPlrAE6I_vp2zKhx8ntDMRMv -O data/raw/sciprompt/sciprompt_data.zip
Then:
    uv run --with transformers==4.38.1 --with tokenizers==0.15.2 python scripts/prepare_sciprompt.py

Frozen rules, applied in release row order:
1. The paper's short-abstract filter (Appendix B; identical code in all three dataset
   scripts): drop rows whose SciBERT (`allenai/scibert_scivocab_uncased`, BertTokenizer,
   transformers 4.38.1 as in the repo's requirements.txt) encoding incl. [CLS]/[SEP] has
   fewer than 30 ids. Drops 10 SDPRA rows (2,800 -> 2,790 = paper Appendix B) and none elsewhere.
2. The benchmark's English-only rule: drop the reviewed NON_ENGLISH rows below.

Output (data/frozen/sciprompt/):
  <dataset>.jsonl  id, row (0-based release row), gold, sha256(abstract, gold) and, for the
                   redistributable SDPRA (CC BY 4.0) and Emerging NLP (Apache-2.0) sets,
                   the abstract itself.
  excluded.jsonl   dataset, row, id, gold, reason (short_abstract | non_english), note.
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gauntlet.task import write_jsonl  # noqa: E402
from gauntlet.tasks.sciprompt import (  # noqa: E402
    DATASETS,
    FROZEN_DIR,
    RAW_ZIP,
    RAW_ZIP_SHA256,
    read_release,
    row_sha256,
)

SCIBERT = "allenai/scibert_scivocab_uncased"
SCIBERT_REVISION = "24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1"
MIN_TOKENS = 30
REDISTRIBUTABLE = {"sdpra", "emerging_nlp"}
RELEASE_N = {"arxiv": 5300, "s2orc": 5700, "sdpra": 2800, "emerging_nlp": 420}
# Rows the paper's filter must drop (verified, not assumed: the script recomputes them).
EXPECTED_SHORT = {"arxiv": [], "s2orc": [], "sdpra": [840, 1111, 1125, 1544, 1683, 1725, 2183, 2283, 2530, 2791], "emerging_nlp": []}

# Reviewed non-English rows (release row -> note). Candidates came from langid 1.1.6
# on whole abstracts and on every >=8-word sentence (non-en, p > 0.99), a foreign
# function-word count, a non-Latin-script count, a foreign "abstract/keywords" heading
# scan and a diacritic-density scan; every candidate was read. Rule (as EZ-STANCE):
# exclude an abstract containing any non-English sentence, clause or multi-word phrase
# (parallel translations, foreign keyword lines); keep English abstracts whose only
# non-English material is names, cited titles, or single glossed terms in any script.
NON_ENGLISH: dict[str, dict[int, str]] = {
    "arxiv": {
        632: "English + French translation", 1585: "English + French translation",
        2928: "Spanish + English translation", 3691: "English + Polish translation",
        4005: "English + German translation",
    },
    "sdpra": {2482: "English + French translation"},
    "emerging_nlp": {},
    "s2orc": {
        108: "Japanese", 1063: "English + French abstract", 1178: "Russian + English abstract",
        1238: "Turkish + English abstract", 1401: "Indonesian keyword line", 1465: "English + mis-encoded Thai",
        1709: "Japanese", 1863: "English + Finnish abstract", 1937: "English + Spanish abstract",
        2426: "Lithuanian + English abstract", 2736: "Spanish keyword line", 2799: "Spanish + English abstract",
        2920: "English + Chinese abstract", 2966: "Spanish", 3609: "Turkish + English abstract",
        3823: "English + Indonesian abstract", 4231: "Swedish", 4300: "Catalan + English abstract",
        4311: "Spanish + English abstract", 4312: "Spanish + English abstract", 4334: "Spanish + Portuguese",
        4339: "Spanish", 4350: "English + Italian abstract", 4397: "Spanish + French + English abstract",
        4462: "English + Thai abstract", 4681: "English + French abstract", 4838: "Chinese keyword line",
        4911: "Spanish + English abstract", 4939: "English + Ukrainian abstract", 4973: "English + Ukrainian abstract",
        5052: "Romanian + English abstract", 5109: "Polish + English abstract", 5292: "Polish + English abstract",
        5484: "Spanish + English abstract", 5525: "English + Turkish abstract", 5579: "Indonesian keyword line",
        5592: "French + English abstract", 5613: "Russian", 5662: "English + French summary",
        5691: "Portuguese preprint notice",
    },
}


def main() -> None:
    from transformers import BertTokenizer

    digest = hashlib.sha256(RAW_ZIP.read_bytes()).hexdigest()
    if digest != RAW_ZIP_SHA256:
        raise SystemExit(f"unexpected raw SHA-256 {digest}")
    tok = BertTokenizer.from_pretrained(SCIBERT, revision=SCIBERT_REVISION)
    excluded = []
    with zipfile.ZipFile(RAW_ZIP) as zf:
        for ds in DATASETS:
            release = read_release(zf, ds)
            if len(release) != RELEASE_N[ds]:
                raise SystemExit(f"{ds}: expected {RELEASE_N[ds]} release rows, got {len(release)}")
            short = [i for i, (text, _) in enumerate(release) if len(tok(text)["input_ids"]) < MIN_TOKENS]
            if short != EXPECTED_SHORT[ds]:
                raise SystemExit(f"{ds}: short-abstract filter drops {short}, expected {EXPECTED_SHORT[ds]}")
            rows = []
            for i, (text, gold) in enumerate(release):
                rid = f"sciprompt-{ds}-test-{i:04d}"
                if i in short or i in NON_ENGLISH[ds]:
                    reason = "short_abstract" if i in short else "non_english"
                    note = f"< {MIN_TOKENS} SciBERT ids (paper filter)" if i in short else NON_ENGLISH[ds][i]
                    excluded.append({"dataset": ds, "row": i, "id": rid, "gold": gold, "reason": reason, "note": note})
                    continue
                row = {"id": rid, "row": i, "gold": gold, "sha256": row_sha256(text, gold)}
                if ds in REDISTRIBUTABLE:
                    row["abstract"] = text
                rows.append(row)
            path = FROZEN_DIR / f"{ds}.jsonl"
            write_jsonl(path, rows)
            counts = Counter(r["gold"] for r in rows)
            print(f"{ds}: {len(rows)} rows ({len(short)} short, {len(NON_ENGLISH[ds])} non-English dropped); "
                  f"{len(counts)} labels, per-label min/max {min(counts.values())}/{max(counts.values())}; "
                  f"sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")
    path = FROZEN_DIR / "excluded.jsonl"
    write_jsonl(path, excluded)
    print(f"excluded: {len(excluded)} rows; sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
