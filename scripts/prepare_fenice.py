# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "duckdb==1.4.1",
#   "beautifulsoup4==4.13.5",
#   "langid==1.1.6",
#   "numpy<2",
#   "torch",
#   "transformers==4.38.2",
#   "sentencepiece==0.2.0",
#   "spacy==3.7.4",
#   "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl",
# ]
# ///
"""Deterministic rebuild of the FENICE long-form (story) factuality test set.

Release: Babelscape/story-summeval (HF dataset, CC BY-NC-SA 4.0), the "annotations
for long-form summarization evaluation (Section 4.4 in the paper)" linked from
github.com/Babelscape/FENICE. It ships summaries + binary factuality labels with an
explicit val/test split (40/279 rows), but no story texts: rows point to a Gutenberg
id or a Wikisource title. This script

  1. downloads the release at a pinned revision;
  2. rebuilds every story ("narrative") text from its public source
     (manu/project_gutenberg, wikimedia/wikisource 20231201.en, gutenberg.org,
     en.wikisource.org for sub-pages the dump lacks), applying Echoes-from-Alexandria's
     Gutenberg header/footer cleaning and a frozen story-section rule for the two
     narratives that are one story inside a collection;
  3. extracts summary claims with the paper's released T5 claim extractor
     (Babelscape/t5-base-summarization-claim-extractor, FENICE's own generation
     settings; the GPT-3.5 claims of FENICE_GPT_claims are not released);
  4. runs the English checks (langid + Latin-script share) over summaries and stories;
  5. writes data/frozen/fenice/{test,val}.jsonl (summary, label, claims, narrative
     key) and data/frozen/fenice/narratives.json (per-document SHA-256; story texts
     stay in data/raw and are hash-checked by the task adapter).

Usage:  uv run scripts/prepare_fenice.py [--skip-claims]
(--skip-claims reuses frozen claims and only re-verifies sources.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "fenice"
FROZEN = REPO / "data" / "frozen" / "fenice"
NARRATIVE_DIR = RAW / "narratives"

UA = {"User-Agent": "TDM-Gauntlet/0.1 (research reproduction; FENICE story-summeval)"}

HF_DATASET = "Babelscape/story-summeval"
HF_REVISION = "ce741d568615de57a77934d171b301d20ba4a135"
HF_FILES = {"test": "data/test.jsonl", "val": "data/val.jsonl"}

GUTENBERG_HF = "manu/project_gutenberg"
GUTENBERG_HF_REVISION = "164853d214065df26a630ee1ab91a0c39e461caf"
WIKISOURCE_HF = "wikimedia/wikisource"
WIKISOURCE_HF_CONFIG = "20231201.en"

CLAIM_MODEL = "Babelscape/t5-base-summarization-claim-extractor"
CLAIM_MODEL_REVISION = "94775fb1c8dc2c3ef1bfec413f9f961e6ba5a1c8"

LONDON_VOLUME = "A London Life, The Patagonia, The Liar, Mrs. Temperly"
AMERICAN_ISSUE = (
    "A London Life, The Patagonia, The Liar, Mrs. Temperly "
    "(1 volume, American issue, London & New York: Macmillan & Co., 1889)"
)

# Narrative key -> documents. A document is one text; a narrative with several
# documents is verified against each (claim support = max over documents/chunks).
#   gutenberg_hf: row of manu/project_gutenberg 'en' (the release card's instruction)
#   gutenberg_org: www.gutenberg.org plain text (ids absent from manu/project_gutenberg)
#   wikisource_hf: row of wikimedia/wikisource 20231201.en by title (card's instruction)
#   wikisource_page: rendered en.wikisource.org page (sub-pages absent from the dump)
# `section`: (start heading, end heading or None) -- story extracted from a collection.
NARRATIVES: dict[str, dict] = {
    "gutenberg:1354": {"title": "Chronicles of Avonlea (L. M. Montgomery)", "docs": [{"kind": "gutenberg_hf", "id": "1354-0"}]},
    "gutenberg:1472": {"title": "In a German Pension (Katherine Mansfield)", "docs": [{"kind": "gutenberg_org", "id": "1472"}]},
    "gutenberg:2165": {"title": "The Lifted Veil (George Eliot)", "docs": [{"kind": "gutenberg_org", "id": "2165"}]},
    "gutenberg:267": {"title": "The Touchstone (Edith Wharton)", "docs": [{"kind": "gutenberg_hf", "id": "267-0"}]},
    "gutenberg:2814": {
        "title": "The Dead (James Joyce, in Dubliners)",
        "docs": [{"kind": "gutenberg_hf", "id": "2814-0", "section": ("THE DEAD", None)}],
    },
    "gutenberg:29042": {"title": "A Tangled Tale (Lewis Carroll)", "docs": [{"kind": "gutenberg_hf", "id": "29042-8"}]},
    "gutenberg:32759": {"title": "Red Nails (Robert E. Howard)", "docs": [{"kind": "gutenberg_hf", "id": "32759-0"}]},
    "gutenberg:640": {
        "title": "Thumbelina (in The Yellow Fairy Book, ed. Andrew Lang)",
        "docs": [{"kind": "gutenberg_hf", "id": "640-0", "section": ("THUMBELINA", "THE NIGHTINGALE")}],
    },
    "gutenberg:68377": {"title": "Let's Get Together (Isaac Asimov)", "docs": [{"kind": "gutenberg_org", "id": "68377"}]},
    "gutenberg:697": {"title": "The Light Princess (George MacDonald)", "docs": [{"kind": "gutenberg_org", "id": "697"}]},
    "gutenberg:8164": {"title": "My Man Jeeves (P. G. Wodehouse)", "docs": [{"kind": "gutenberg_org", "id": "8164"}]},
    "wikisource:Loss of Breath": {"title": "Loss of Breath (Edgar Allan Poe)", "docs": [{"kind": "wikisource_hf", "title": "Loss of Breath"}]},
    "wikisource:Lot No. 249": {"title": "Lot No. 249 (Arthur Conan Doyle)", "docs": [{"kind": "wikisource_hf", "title": "Lot No. 249"}]},
    "wikisource:The Heathen": {"title": "The Heathen (Jack London)", "docs": [{"kind": "wikisource_hf", "title": "The Heathen"}]},
    "wikisource:The Tree (Lovecraft)": {"title": "The Tree (H. P. Lovecraft)", "docs": [{"kind": "wikisource_hf", "title": "The Tree (Lovecraft)"}]},
    "wikisource:The Union Buries Its Dead": {
        "title": "The Union Buries Its Dead (Henry Lawson)",
        "docs": [{"kind": "wikisource_hf", "title": "The Union Buries Its Dead"}],
    },
    # The dump's "A London Life, The Patagonia, ..." row is a 1,225-char versions page.
    # The volume holds two annotated narratives: The Liar (all test rows: Oliver Lyon,
    # Capadose; ref = Echo-XSum 'The Liar') and A London Life (all val rows: Laura Wing;
    # ref = Echo-XSum 'A London Life'). The only existing Wikisource transcription of
    # this volume is the American issue; each story is its chapter sub-pages in order.
    "wikisource:The Liar": {
        "title": "The Liar (Henry James)",
        "docs": [{"kind": "wikisource_page", "pages": [f"{AMERICAN_ISSUE}/The Liar/Chapter {i}" for i in range(1, 4)]}],
    },
    "wikisource:A London Life": {
        "title": "A London Life (Henry James)",
        "docs": [{"kind": "wikisource_page", "pages": [f"{AMERICAN_ISSUE}/A London Life/Chapter {i}" for i in range(1, 14)]}],
    },
    # The dump's "Thumbelina" row is a 905-char versions page listing 7 English
    # translations; the Yellow Fairy Book one is gutenberg:640. The 73 test rows are
    # 6 x 12 system summaries + 1 reference over the other 6 translations, and the
    # release does not say which translation a row summarises (ids repeat 6 times).
    # Frozen rule: verify against all 6 candidate translations (one document each).
    "wikisource:Thumbelina": {
        "title": "Thumbelina (H. C. Andersen; one of six English Wikisource translations)",
        "docs": [
            {"kind": "wikisource_page", "pages": ["Little Ellie and Other Tales/Little Ellie"]},
            {"kind": "wikisource_page", "pages": ["Hans Andersen's Fairy Tales/Little Tiny"]},
            {"kind": "wikisource_page", "pages": ["Fairy tales and stories (Andersen, Tegner)/Thumbeline"]},
            {"kind": "wikisource_page", "pages": ["Hans Andersen's fairy tales (Robinson)/Tommelise"]},
            {"kind": "wikisource_page", "pages": ["Fairy Tales and Other Stories (Andersen, Craigie)/Thumbelina"]},
            {"kind": "wikisource_page", "pages": ["Fairy tales from Hans Christian Andersen (Walker)/Thumbelisa"]},
        ],
    },
}


def narrative_key(row: dict, split: str) -> str:
    """Frozen row -> narrative rule (source + text_id; the London volume by split)."""
    if row["source"] == "wikisource" and row["text_id"] == LONDON_VOLUME:
        return "wikisource:The Liar" if split == "test" else "wikisource:A London Life"
    return f"{row['source']}:{row['text_id']}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def fetch(url: str) -> bytes:
    """GET with backoff on HTTP 429/503 (Wikimedia rate limits)."""
    import time
    import urllib.error

    for attempt in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as resp:
                data = resp.read()
            time.sleep(1.0)
            return data
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 503) or attempt == 7:
                raise
            time.sleep(float(exc.headers.get("Retry-After") or 0) or 5.0 * 2**attempt)
    raise AssertionError("unreachable")


def cached(path: Path, url: str) -> bytes:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fetch(url))
    return path.read_bytes()


# --- 1. release -------------------------------------------------------------------


def load_release() -> dict[str, list[dict]]:
    out = {}
    for split, name in HF_FILES.items():
        url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{HF_REVISION}/{name}"
        data = cached(RAW / "release" / name, url)
        out[split] = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    return out


# --- 2. narrative texts --------------------------------------------------------------


def gutenberg_clean(text: str) -> str:
    """Echoes-from-Alexandria clean_text (dataset_generation/src/main/clean_books.py):
    drop [Illustration...]/[Footnote...] notes, keep the text between the Project
    Gutenberg START and END markers."""
    for pattern in ("Illustration", "Footnote"):
        text = re.sub("[\n\r]+", "@#@", text)
        text = re.sub(f"\\[{pattern}:?.*?]", "", text)
        text = re.sub("@#@", "\n", text)
    start = [m.end() for m in re.finditer(r"\*{3}\s?START OF.*[\r\n]?.*\*{3}[\n\r]+", text)]
    end = [m.start() for m in re.finditer(r"\*{3}\s?END OF.*[\r\n]?.*\*{3}[\n\r]+", text)]
    new = text[start[0]:] if start else None
    if end:
        new = new[: end[0]] if new is not None else text[: end[0]]
    return new if new is not None else text


def normalise(text: str) -> str:
    """NFC, LF newlines, strip BOM and trailing spaces, collapse 3+ blank lines."""
    text = unicodedata.normalize("NFC", text.replace("\ufeff", "")).replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def extract_section(text: str, start: str, end: str | None) -> str:
    """The story from its all-caps heading line up to the next story's heading line."""
    heads = [m.start() for m in re.finditer(rf"(?m)^{re.escape(start)}$", text)]
    if len(heads) != 1:
        raise ValueError(f"section heading {start!r}: {len(heads)} matches")
    body = text[heads[0]:]
    if end is not None:
        stops = [m.start() for m in re.finditer(rf"(?m)^{re.escape(end)}$", body)]
        if not stops:
            raise ValueError(f"end heading {end!r} not found")
        body = body[: stops[0]]
    return body


def gutenberg_hf_rows(ids: list[str]) -> dict[str, str]:
    import duckdb

    cache = RAW / "gutenberg_hf" / "rows.json"
    if cache.exists():
        rows = json.loads(cache.read_text())
    else:
        tree = json.loads(fetch(f"https://huggingface.co/api/datasets/{GUTENBERG_HF}/tree/{GUTENBERG_HF_REVISION}/data"))
        urls = [
            f"https://huggingface.co/datasets/{GUTENBERG_HF}/resolve/{GUTENBERG_HF_REVISION}/{x['path']}"
            for x in tree
            if "/en-" in x["path"]
        ]
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs;")
        found = con.execute(
            "SELECT id, filename, text FROM read_parquet(?, filename=true) WHERE id IN (SELECT unnest(?))",
            [urls, ids],
        ).fetchall()
        rows = {}
        for gid, filename, text in found:
            # The shards hold some ids twice (29042-8 twice, byte-identical).
            if gid in rows and rows[gid]["text"] != text:
                raise ValueError(f"{GUTENBERG_HF}: id {gid} occurs with different texts")
            rows[gid] = {"file": filename.rsplit("/", 1)[1], "text": text}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows, ensure_ascii=False))
    missing = set(ids) - set(rows)
    if missing:
        raise ValueError(f"{GUTENBERG_HF}: ids not found: {sorted(missing)}")
    return {gid: rows[gid]["text"] for gid in ids}


def wikisource_hf_rows(titles: list[str]) -> dict[str, str]:
    import duckdb

    cache = RAW / "wikisource_hf" / "rows.json"
    if cache.exists():
        rows = json.loads(cache.read_text())
    else:
        tree = json.loads(fetch(f"https://huggingface.co/api/datasets/{WIKISOURCE_HF}/tree/main/{WIKISOURCE_HF_CONFIG}"))
        urls = [f"https://huggingface.co/datasets/{WIKISOURCE_HF}/resolve/main/{x['path']}" for x in tree if x["path"].endswith(".parquet")]
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs;")
        found = con.execute(
            "SELECT id, url, title, text FROM read_parquet(?) WHERE title IN (SELECT unnest(?))", [urls, titles]
        ).fetchall()
        rows = {}
        for wid, url, title, text in found:
            if title in rows:
                raise ValueError(f"{WIKISOURCE_HF}: title {title!r} is not unique")
            rows[title] = {"id": wid, "url": url, "text": text}
        rows["_parquet"] = {x["path"]: x["lfs"]["oid"] for x in tree if x["path"].endswith(".parquet")}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows, ensure_ascii=False))
    missing = set(titles) - set(rows)
    if missing:
        raise ValueError(f"{WIKISOURCE_HF}: titles not found: {sorted(missing)}")
    return {t: rows[t]["text"] for t in titles}


def wikisource_page(page: str) -> tuple[str, int]:
    """Rendered page text (transclusions expanded), header/navigation/page numbers removed."""
    from bs4 import BeautifulSoup

    safe = re.sub(r"[^\w.-]+", "_", page)
    path = RAW / "wikisource_pages" / f"{safe}.json"
    params = urllib.parse.urlencode({"action": "parse", "format": "json", "formatversion": "2", "prop": "text|revid", "page": page})
    data = json.loads(cached(path, f"https://en.wikisource.org/w/api.php?{params}"))["parse"]
    soup = BeautifulSoup(data["text"], "html.parser")
    for node in soup.select('[style*="display:none"]'):
        node.decompose()
    for br in soup.find_all("br"):
        br.replace_with(" ")
    for sel in (
        "style", "script", ".ws-noexport", ".wst-header", "#headertemplate", ".ws-pagenum", ".pagenum",
        "figure", ".mw-editsection", "sup.reference", ".mw-references-wrap", ".references", "table.wst-toc",
    ):
        for node in soup.select(sel):
            node.decompose()
    blocks = []
    for node in soup.find_all(["p", "h2", "h3", "h4", "dd", "li", "div"], recursive=True):
        if node.name == "div" and node.find(["p", "div"]) is not None:
            continue
        text = " ".join(node.get_text("").split())
        if text:
            blocks.append(text)
    # Dedupe consecutive repeats created by nested block elements.
    out = [b for i, b in enumerate(blocks) if i == 0 or b != blocks[i - 1]]
    return "\n\n".join(out), int(data["revid"])


def build_documents() -> dict[str, list[dict]]:
    hf_gut = gutenberg_hf_rows(sorted({d["id"] for n in NARRATIVES.values() for d in n["docs"] if d["kind"] == "gutenberg_hf"}))
    hf_ws = wikisource_hf_rows(sorted({d["title"] for n in NARRATIVES.values() for d in n["docs"] if d["kind"] == "wikisource_hf"}))
    out: dict[str, list[dict]] = {}
    for key, spec in NARRATIVES.items():
        docs = []
        for i, d in enumerate(spec["docs"]):
            prov: dict = {"kind": d["kind"]}
            if d["kind"] == "gutenberg_hf":
                text = gutenberg_clean(hf_gut[d["id"]])
                prov.update(dataset=GUTENBERG_HF, revision=GUTENBERG_HF_REVISION, split="en", id=d["id"])
            elif d["kind"] == "gutenberg_org":
                url = f"https://www.gutenberg.org/cache/epub/{d['id']}/pg{d['id']}.txt"
                raw = cached(RAW / "gutenberg_org" / f"pg{d['id']}.txt", url)
                text = gutenberg_clean(raw.decode("utf-8"))
                prov.update(url=url, raw_sha256=sha256_bytes(raw))
            elif d["kind"] == "wikisource_hf":
                text = hf_ws[d["title"]]
                prov.update(dataset=WIKISOURCE_HF, config=WIKISOURCE_HF_CONFIG, title=d["title"])
            elif d["kind"] == "wikisource_page":
                rendered = [wikisource_page(page) for page in d["pages"]]
                text = "\n\n".join(t for t, _ in rendered)
                prov["pages"] = [
                    {"page": page, "revid": revid, "url": "https://en.wikisource.org/w/index.php?" + urllib.parse.urlencode({"title": page, "oldid": revid})}
                    for page, (_, revid) in zip(d["pages"], rendered)
                ]
            else:
                raise ValueError(d["kind"])
            if "section" in d:
                text = extract_section(text, *d["section"])
                prov["section"] = list(d["section"])
            text = normalise(text)
            doc_id = f"{i}"
            label = (d["pages"][0].rsplit("/Chapter", 1)[0] if "pages" in d else None) or d.get("title") or d.get("id")
            path = NARRATIVE_DIR / re.sub(r"[^\w.-]+", "_", key) / f"{doc_id}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            docs.append({"doc_id": doc_id, "label": label, "path": str(path.relative_to(REPO)), "chars": len(text), "sha256": sha256_text(text), "provenance": prov})
        out[key] = docs
    return out


# --- 3. claims -----------------------------------------------------------------------


def extract_claims(summaries: list[str]) -> dict[str, list[str]]:
    """FENICE ClaimExtractor.process_batch with the model's generation_config (beam 5,
    max_length 250), one summary per batch (no padding effects), fp32 (CUDA if
    available, as FENICE's default device); decoded output split into sentences with
    spaCy en_core_web_sm and de-duplicated. The claims are frozen in the output files."""
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import spacy
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    nlp = spacy.load("en_core_web_sm")
    tok = T5Tokenizer.from_pretrained(CLAIM_MODEL, revision=CLAIM_MODEL_REVISION)
    model = T5ForConditionalGeneration.from_pretrained(CLAIM_MODEL, revision=CLAIM_MODEL_REVISION).eval().to(device)
    out = {}
    unique = sorted(set(summaries))
    for i, summary in enumerate(unique):
        enc = tok.batch_encode_plus([summary], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**enc)
        decoded = tok.batch_decode(gen, skip_special_tokens=True)[0]
        sents = [s.text for s in nlp(decoded).sents]
        seen: set[str] = set()
        out[summary] = [s for s in sents if not (s in seen or seen.add(s))]
        if (i + 1) % 50 == 0:
            print(f"claims {i + 1}/{len(unique)}", file=sys.stderr)
    return out


# --- 4. English checks -------------------------------------------------------------


def latin_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    return sum("LATIN" in unicodedata.name(c, "") for c in letters) / len(letters)


def english_report(rows: list[dict], documents: dict[str, list[dict]]) -> dict:
    import langid

    langid.set_languages(None)
    summary_flags = []
    for r in rows:
        lang, _ = langid.classify(r["summary"])
        share = latin_share(r["summary"])
        if lang != "en" or share < 0.95:
            summary_flags.append({"id": r["id"], "langid": lang, "latin_share": round(share, 3), "summary": r["summary"]})
    doc_stats = {}
    for key, docs in documents.items():
        for d in docs:
            text = (REPO / d["path"]).read_text(encoding="utf-8")
            paras = [p for p in text.split("\n\n") if len(p) >= 200]
            non_en = [p for p in paras if langid.classify(p)[0] != "en"]
            doc_stats[f"{key}#{d['doc_id']}"] = {
                "langid_whole": langid.classify(text[:200_000])[0],
                "latin_share": round(latin_share(text), 4),
                "paragraphs_ge200": len(paras),
                "paragraphs_non_en": len(non_en),
                "non_en_examples": [p[:160] for p in non_en[:5]],
            }
    return {"summary_flags": summary_flags, "documents": doc_stats}


# --- 5. freeze -----------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-claims", action="store_true", help="reuse claims from the frozen files")
    args = ap.parse_args()

    release = load_release()
    documents = build_documents()

    all_rows = [(split, i, r) for split in ("test", "val") for i, r in enumerate(release[split])]
    for split, _, r in all_rows:
        key = narrative_key(r, split)
        if key not in NARRATIVES:
            raise ValueError(f"row {r['id']} ({split}) maps to unknown narrative {key!r}")
    used = {narrative_key(r, s) for s, _, r in all_rows}
    if used != set(NARRATIVES):
        raise ValueError(f"unused narratives: {set(NARRATIVES) - used}")

    if args.skip_claims:
        claims = {}
        for split in ("test", "val"):
            for row in (json.loads(x) for x in (FROZEN / f"{split}.jsonl").read_text().splitlines() if x.strip()):
                claims[row["summary"]] = row["claims"]
    else:
        claims = extract_claims([r["summary"] for _, _, r in all_rows])

    FROZEN.mkdir(parents=True, exist_ok=True)
    for split in ("test", "val"):
        with (FROZEN / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for i, r in enumerate(release[split]):
                row = {
                    "id": f"{split}-{i:03d}",
                    "release_line": i,
                    "release_id": r["id"],
                    "system_id": r["system_id"],
                    "source": r["source"],
                    "text_id": r["text_id"],
                    "narrative": narrative_key(r, split),
                    "summary": r["summary"],
                    "claims": claims[r["summary"]],
                    "factual": int(r["factual"]),
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    narratives = {
        key: {"title": NARRATIVES[key]["title"], "documents": docs}
        for key, docs in documents.items()
    }
    (FROZEN / "narratives.json").write_text(json.dumps(narratives, indent=2, ensure_ascii=False) + "\n")
    report = english_report([dict(r, id=f"{s}-{i:03d}") for s, i, r in all_rows], documents)
    (RAW / "english_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    release_hashes = {name: sha256_bytes((RAW / "release" / name).read_bytes()) for name in HF_FILES.values()}
    print(json.dumps({"release_sha256": release_hashes, "n": {s: len(release[s]) for s in release}}, indent=2))
    for key, docs in documents.items():
        print(key, [(d["label"][:40], d["chars"]) for d in docs])
    print("summary flags:", len(report["summary_flags"]))


if __name__ == "__main__":
    main()
