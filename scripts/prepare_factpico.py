# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy>=1.11", "langid==1.1.6"]
# ///
"""Freeze the FactPICO evaluation set (Joseph et al., ACL 2024).

Source: the data zip linked from https://github.com/lilywchen/FactPICO (data/README.md,
repo commit 2e16993a000aedb15cb348b7bcd61070d26bab14): public Box shared link
https://utexas.box.com/s/mpe5idxrqrzs1wcakphng7xfi7h4g83j -> FactPICO.zip.
Annotations are CC-BY-4.0 (paper Appendix B); abstracts come from Evidence Inference 2.0
(PubMed Open Access subset). The frozen JSONL therefore holds the full rows.

    uv run scripts/prepare_factpico.py

Downloads the zip into data/raw/factpico/ when absent, checks its SHA-256, and writes
data/frozen/factpico/test.jsonl: all 345 summaries (115 abstracts x {Alpaca, GPT-4,
Llama-2}) in release order, with
  - abstract / summary: release text with HTML entities decoded (the release escapes
    '<' and '>' as '&lt;'/'&gt;'; no raw '<' or '>' occurs anywhere),
  - result_spans: the Evidence Inference 2.0 result spans the experts rated against
    the summary (evidence_inference_rationales.csv, release order),
  - gold: expert ratings (Population, Intervention, Comparator, Outcome, Results,
    Avg. PICO-R) exactly as released (0 = N/A; doubly annotated items are averages),
  - baselines: the paper's per-summary metric scores (Table 4 systems), kept so the
    comparison can be recomputed with the same code.
It also re-derives the paper's Table 4 correlations from the release as a protocol check
and runs the English-only verification.
"""

from __future__ import annotations

import csv
import hashlib
import html
import http.cookiejar
import io
import json
import sys
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW_DIR = REPO / "data" / "raw" / "factpico"
ZIP = RAW_DIR / "FactPICO.zip"
OUT_DIR = REPO / "data" / "frozen" / "factpico"
OUT = OUT_DIR / "test.jsonl"
STATS = OUT_DIR / "stats.json"

BOX_SHARED = "https://utexas.box.com/s/mpe5idxrqrzs1wcakphng7xfi7h4g83j"
BOX_DOWNLOAD = (
    "https://utexas.app.box.com/index.php?rm=box_download_shared_file"
    "&shared_name=mpe5idxrqrzs1wcakphng7xfi7h4g83j&file_id=f_1447492862089"
)
ZIP_SHA256 = "ec260d7c69db9537f819fbdf728c997520e55de56b9f03b2980b017c91b9d4f4"
MEMBER_SHA256 = {
    "data/all_evaluations.csv": "1035640d11dbe5fd28ad13385785638fca3c2f0632ba5e94480ed319b90992cd",
    "data/evidence_inference_rationales.csv": "b8ab3c4a3c363277da8aad2006e9aee2058db802ad324b0840479fb8f8d4a0bf",
}
EXPECTED_N = 345
MODELS = {"ALPACA": "alpaca", "GPT-4": "gpt4", "LLAMA-2": "llama2"}
GOLD_COLUMNS = {
    "population": "Population",
    "intervention": "Intervention",
    "comparator": "Comparator",
    "outcome": "Outcome",
    "results": "Results",
    "avg": "Avg. PICO-R",
}
BASELINES = ("QAFactEval", "QuestEval", "AlignScore", "DAE", "GPT-4", "Llama-2", "Alpaca", "Mistral", "Extract")
# Paper Table 4, "Avg." rows (Kendall tau-b / Spearman rho of each system's overall score
# vs Avg. PICO-R): the protocol check below must reproduce these to 3 decimals.
TABLE4_AVG = {
    "QAFactEval": (0.289, 0.406), "QuestEval": (0.290, 0.412), "AlignScore": (0.244, 0.348),
    "DAE": (0.152, 0.219), "GPT-4": (0.475, 0.619), "Llama-2": (0.055, 0.080),
    "Alpaca": (0.081, 0.115), "Mistral": (0.047, 0.065), "Extract": (0.474, 0.633),
}


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    opener.open(BOX_SHARED).read()  # sets the Box session cookie for the shared link
    ZIP.write_bytes(opener.open(BOX_DOWNLOAD).read())


def read_member(zf: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    raw = zf.read(name)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != MEMBER_SHA256[name]:
        raise SystemExit(f"{name}: unexpected SHA-256 {digest}")
    csv.field_size_limit(sys.maxsize)
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))


def non_latin_share(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    other = sum("LATIN" not in unicodedata.name(ch, "") for ch in letters)
    return other / len(letters) if letters else 0.0


def main() -> None:
    if not ZIP.exists():
        download()
    digest = hashlib.sha256(ZIP.read_bytes()).hexdigest()
    if digest != ZIP_SHA256:
        raise SystemExit(f"unexpected FactPICO.zip SHA-256 {digest}")
    with zipfile.ZipFile(ZIP) as zf:
        evals = read_member(zf, "data/all_evaluations.csv")
        spans_rows = read_member(zf, "data/evidence_inference_rationales.csv")

    spans: dict[tuple[str, str], list[str]] = {}
    for r in spans_rows:
        spans.setdefault((r["abstract"], r["generation"]), []).append(r["results_span"])

    abstract_index: dict[str, int] = {}
    rows = []
    for r in evals:
        a = abstract_index.setdefault(r["Abstract"], len(abstract_index))
        key = (r["Abstract"], r["generation"])
        if key not in spans:
            raise SystemExit(f"no evidence-inference spans for abstract {a} / {r['model_type']}")
        rows.append(
            {
                "id": f"factpico-{a:03d}-{MODELS[r['model_type']]}",
                "abstract_id": a,
                "summarizer": r["model_type"],
                "abstract": html.unescape(r["Abstract"]),
                "summary": html.unescape(r["generation"]),
                "result_spans": [html.unescape(s) for s in spans[key]],
                "gold": {k: float(r[c]) for k, c in GOLD_COLUMNS.items()},
                "baselines": {b: float(r[b]) for b in BASELINES},
            }
        )
    if len(rows) != EXPECTED_N or len(abstract_index) != 115 or len(spans) != EXPECTED_N:
        raise SystemExit(f"expected 345 summaries / 115 abstracts, got {len(rows)} / {len(abstract_index)}")
    if len({r["id"] for r in rows}) != EXPECTED_N:
        raise SystemExit("duplicate item ids")
    for r in rows:
        g = r["gold"]
        mean5 = sum(g[k] for k in ("population", "intervention", "comparator", "outcome", "results")) / 5
        if abs(mean5 - g["avg"]) > 1e-9:
            raise SystemExit(f"{r['id']}: Avg. PICO-R is not the mean of the five ratings")
        if any("&lt;" in t or "&gt;" in t for t in (r["abstract"], r["summary"], *r["result_spans"])):
            raise SystemExit(f"{r['id']}: HTML entity left after decoding")

    # Protocol check: Table 4 Avg. rows reproduce from the release (all 345, N/A kept as 0).
    from scipy.stats import kendalltau, spearmanr

    human = [r["gold"]["avg"] for r in rows]
    table4 = {}
    for b, (tau, rho) in TABLE4_AVG.items():
        sys_scores = [r["baselines"][b] for r in rows]
        got = (round(float(kendalltau(human, sys_scores, variant="b")[0]), 3), round(float(spearmanr(human, sys_scores)[0]), 3))
        table4[b] = got
        if got != (tau, rho):
            raise SystemExit(f"Table 4 protocol check failed for {b}: {got} vs paper {(tau, rho)}")

    # English-only verification (the benchmark's English-only rule: every row sent to Jev must be English).
    import langid

    texts = {("abstract", r["abstract_id"]): r["abstract"] for r in rows}
    texts.update({("summary", r["id"]): r["summary"] for r in rows})
    for r in rows:
        for s in r["result_spans"]:
            texts.setdefault(("span", s), s)
    flags = []
    for (kind, key), text in texts.items():
        lang, _ = langid.classify(text)
        share = non_latin_share(text)
        if lang != "en" or share > 0.02:
            flags.append({"kind": kind, "langid": lang, "non_latin_share": round(share, 4), "text": text[:200]})

    write_rows(rows)
    stats = {
        "n": len(rows),
        "n_abstracts": len(abstract_index),
        "per_summarizer": {m: sum(r["summarizer"] == m for r in rows) for m in MODELS},
        "n_result_spans": sum(len(r["result_spans"]) for r in rows),
        "spans_per_summary": {str(k): sum(len(r["result_spans"]) == k for r in rows) for k in range(1, 6)},
        "gold_na_ratings": {k: sum(r["gold"][k] == 0 for r in rows) for k in ("population", "intervention", "comparator", "outcome")},
        "max_chars": {"abstract": max(len(r["abstract"]) for r in rows), "summary": max(len(r["summary"]) for r in rows),
                      "span": max(len(s) for r in rows for s in r["result_spans"])},
        "table4_avg_reproduced": {b: {"tau_b": t, "rho": p} for b, (t, p) in table4.items()},
        "english_check": {"texts_checked": len(texts), "method": "langid 1.1.6 + non-Latin letter share > 2%", "flags": flags},
        "frozen_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
    }
    STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "english_check"}, indent=2))
    print(f"english flags: {len(flags)} of {len(texts)}")
    for f in flags:
        print("  ", f)


def write_rows(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
