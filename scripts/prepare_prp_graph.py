"""Rebuild the frozen PRP-Graph TREC-COVID test set (task `prp_graph/trec_covid`).

Reproduces the first stage of PRP-Graph (Luo et al., ACL 2024; code
https://github.com/Memelank/PRP-Graph, `retriever/pyserini_retriever.py` and
`rankers/rank_flan.py`): pyserini BM25 over the prebuilt BEIR index
`beir-v1.0.0-trec-covid.flat`, top-100 per test topic, passages formatted as
"Title: {title} Content: {text}" (whitespace-collapsed) and truncated to the
paper's 128 Flan-T5 tokens (queries: 32 tokens).

This script runs OUTSIDE the project environment, in an isolated venv that
pins the paper's toolchain (pyserini 0.22.1 needs Java 11, available system-wide):

    uv venv --python 3.10 data/raw/prp_graph/venv
    uv pip install --python data/raw/prp_graph/venv/bin/python pyserini==0.22.1 langid==1.1.6
    uv pip install --python data/raw/prp_graph/venv/bin/python faiss-cpu==1.7.4 torch==2.1.0 \
        --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
    uv pip install --python data/raw/prp_graph/venv/bin/python "numpy<2" transformers==4.33.1
    # raw inputs (URLs and SHA-256 in data/manifests/prp_graph.json):
    #   data/raw/prp_graph/trec-covid.zip (BEIR), unzipped to data/raw/prp_graph/trec-covid/
    #   data/raw/prp_graph/home/.cache/anserini/topics-and-qrels/{topics,qrels} (castorini/eval)
    #   data/raw/prp_graph/retrieve_results_covid_top100.jsonl (rank_llm cache, cross-check only)
    JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
        data/raw/prp_graph/venv/bin/python scripts/prepare_prp_graph.py

Outputs
    data/raw/prp_graph/bm25/retrieve_results_covid.json   PRP-Graph retriever JSON (with texts)
    data/raw/prp_graph/bm25/trec_results_covid.txt        PRP-Graph BM25 run file
    data/frozen/prp_graph/trec_covid/queries.jsonl        50 topics: query, BM25 top-100, qrels
    data/frozen/prp_graph/trec_covid/passages.jsonl       per-doc truncation offset + SHA-256 (no text)
    data/raw/prp_graph/prepare_summary.json               verification results for the manifest

Document texts are NOT frozen in the repo (CORD-19 licence: text-and-data-mining
use only, no redistribution grant); the adapter rebuilds them from the BEIR
corpus.jsonl and checks every passage against the frozen SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "prp_graph"
FROZEN = REPO / "data" / "frozen" / "prp_graph" / "trec_covid"

os.environ.setdefault("PYSERINI_CACHE", str(RAW / "pyserini_cache"))
os.environ.setdefault("HF_HOME", str(RAW / "hf"))

import jnius_config  # noqa: E402

# Anserini 0.22.1 looks up BEIR topics/qrels under ${user.home}/.cache/anserini/topics-and-qrels
# and its cloud fallback (castorini/anserini-tools) no longer exists; the pinned files live there.
jnius_config.add_options(f"-Duser.home={RAW / 'home'}")

INDEX = "beir-v1.0.0-trec-covid.flat"
TOPICS = "beir-v1.0.0-trec-covid-test"
HITS = 100
QUERY_TOKENS = 32
PASSAGE_TOKENS = 128
T5_MODEL = "google/flan-t5-large"
T5_REVISION = "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
QRELS_FILE = RAW / "home" / ".cache" / "anserini" / "topics-and-qrels" / "qrels.beir-v1.0.0-trec-covid.test.txt"
BEIR_DIR = RAW / "trec-covid"
RANK_LLM_CACHE = RAW / "retrieve_results_covid_top100.jsonl"

# English-only rule (every row sent to Jev must be English): candidate passages that contain non-English text
# (non-English title, translated title variant, non-English abstract, or non-Latin-script journal
# name) are removed from the candidate lists before any request is built. Found by a segment-level
# langid screen plus non-Latin-script and accented-word screens (`language_screen`), every hit then
# read; English text that merely contains foreign proper names or loanwords (e.g.
# "Sociedad Mexicana de Oncología", "ménage à trois", "Göttingen minipigs") is kept.
NON_ENGLISH: dict[str, str] = {
    "1b7viy5o": "Spanish title and abstract",
    "2vns1wdb": "Spanish title + English title variant",
    "31fnbch7": "Dutch title + English title variant",
    "41ycjpt9": "English + Spanish + Portuguese title variants",
    "42wlbcbm": "German title",
    "5ps6pw6c": "Spanish title",
    "62je92vo": "Spanish title + English title variant",
    "74xuvaz6": "French title + English title variant",
    "7h7v6abe": "French title + English title variant",
    "aezp1isw": "English + Chinese title variants",
    "bgehvxo7": "German title",
    "c7weqc03": "French title + English title variant",
    "dpe9ld92": "Polish title + English title variant",
    "e3clp17r": "Dutch title + English title variant",
    "e4crbz0c": "German title",
    "eqewn6j4": "Chinese journal name in title",
    "gbvyuem5": "Chinese journal name in title",
    "gokevnzw": "English + Croatian title variants",
    "i7scw9mu": "English + Russian title variants",
    "jz1z7fcx": "Spanish title (twice) + English title variant; passage has no abstract text",
    "k4a11aij": "French title",
    "kyvggk83": "Chinese journal name in title",
    "lmo0ry4k": "French title and abstract",
    "mi0pmyo4": "Italian title + English title variant",
    "n6jgr1kx": "English + Spanish title variants",
    "of7nk5ox": "Spanish title (twice) + English title variant",
    "oweh98z2": "Spanish title",
    "p2i0ha2u": "English + French title variants",
    "p7ldfrbv": "Spanish title",
    "pb3r74uq": "German title",
    "ph3z1ha4": "English + Chinese title variants",
    "puc13jf1": "English + Spanish title variants",
    "rq5nqm92": "Spanish title",
    "s5f1olt4": "Chinese journal name in title",
    "sk5eugor": "French title + English title variant",
    "thx2xtsl": "Spanish title + English title variant",
    "tp2tm0vn": "Portuguese title (twice) + English title variant",
    "u7dgeqoh": "French title + English title variant",
    "vu0cevon": "French title + English title variant",
    "vudufbi0": "French title",
    "xp7fxceo": "Spanish title + English title variant",
    "xw244riy": "Swedish title + English title variant",
    "xxnasobj": "English + Turkish title variants",
    "yjmada6s": "German title + English title variant",
    "ynias4ga": "Spanish title",
    "zzljrkbf": "Chinese journal name in title",
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def format_content(title: str, text: str) -> str:
    """`PyseriniRetriever._retrieve_query` formatting (PRP-Graph retriever/pyserini_retriever.py)."""
    return " ".join(("Title: " + title + " " + "Content: " + text).split())


def retrieve() -> tuple[list[dict], dict]:
    """Replicates `PyseriniRetriever(dataset='covid', BM25).retrieve_and_store()`."""
    from pyserini.index import IndexReader
    from pyserini.search import get_qrels, get_topics
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher.from_prebuilt_index(INDEX)
    searcher.set_bm25()
    topics = get_topics(TOPICS)
    qrels = get_qrels(TOPICS)
    reader = IndexReader.from_prebuilt_index(INDEX)
    results = []
    for qid in topics:
        if qid not in qrels:
            continue
        query = topics[qid]["title"]
        hits = searcher.search(query, k=HITS)
        entry = {"query": query, "hits": []}
        for rank, hit in enumerate(hits, start=1):
            raw = json.loads(reader.doc(hit.docid).raw())
            if "title" in raw:
                content = format_content(raw["title"], raw["text"])
            else:
                content = " ".join(raw["contents"].split())
            entry["hits"].append(
                {"content": content, "qid": qid, "docid": hit.docid, "rank": rank, "score": float(hit.score)}
            )
        results.append(entry)
    out = RAW / "bm25"
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieve_results_covid.json").write_text(json.dumps(results, indent=2))
    with (out / "trec_results_covid.txt").open("w") as fh:
        for entry in results:
            for hit in entry["hits"]:
                fh.write(f"{hit['qid']} Q0 {hit['docid']} {hit['rank']} {hit['score']} rank\n")
    return results, qrels


def trec_eval_ndcg10(run_file: Path) -> float:
    """Paper's evaluation: `python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 <qrels> <run>`."""
    out = subprocess.run(
        [sys.executable, "-m", "pyserini.eval.trec_eval", "-c", "-m", "ndcg_cut.10", str(QRELS_FILE), str(run_file)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "ndcg_cut_10" and parts[1] == "all":
            return float(parts[2])
    raise RuntimeError(f"trec_eval output without ndcg_cut_10: {out}")


def token_prefix(tokenizer, text: str, n_tokens: int) -> tuple[str, dict]:
    """Original-text prefix covering the first `n_tokens` Flan-T5 tokens.

    The paper truncates with `tokenizer.convert_tokens_to_string(tokenizer.tokenize(text)[:n])`
    (rank_flan.py). Decoding re-normalises text (NFKC) and turns out-of-vocabulary characters
    into " ⁇ "; we keep the same token budget but cut the ORIGINAL string at the end offset of
    token n, so Jev sees the same span without decode artefacts.
    """
    tokens = tokenizer.tokenize(text)
    paper = tokenizer.convert_tokens_to_string(tokens[:n_tokens])
    if len(tokens) <= n_tokens:
        return text, {"truncated": False, "n_tokens": len(tokens), "paper_string_equal": paper == text}
    proto = tokenizer.sp_model.encode(text, return_type="proto")  # piece offsets are UTF-8 byte offsets
    pieces = [p.piece for p in proto.pieces]
    if pieces != tokens:
        raise RuntimeError("sentencepiece pieces differ from T5Tokenizer.tokenize")
    end_byte = proto.pieces[n_tokens - 1].end
    prefix = text.encode("utf-8")[:end_byte].decode("utf-8").rstrip()
    retok = tokenizer.tokenize(prefix)
    return prefix, {
        "truncated": True,
        "n_tokens": len(tokens),
        "prefix_retokenizes_to_same": retok == tokens[:n_tokens],
        "paper_string_equal": paper == prefix,
    }


def language_screen(langid_identifier, text: str) -> dict:
    """Automated screen feeding the review behind NON_ENGLISH (not itself a filter)."""
    import re
    import unicodedata

    m = re.match(r"Title: (.*?) Content: (.*)", text, re.S)
    title, body = (m.group(1), m.group(2)) if m else (text.removeprefix("Title: "), "")
    segments = [s for s in re.split(r"/ |\[|\]", title) if len(s.split()) >= 2]
    segments += [s for s in re.split(r"(?<=[.!?])\s+", body) if len(s.split()) >= 5]
    flagged = []
    for seg in segments:
        lang, prob = langid_identifier.classify(seg)
        if lang != "en":
            flagged.append([lang, round(float(prob), 3), seg[:120]])
    scripts = sorted(
        {unicodedata.name(c, "?").split()[0] for c in text if c.isalpha() and ord(c) > 0x24F} - {"GREEK", "LATIN"}
    )
    accented = sorted(
        {w for w in re.findall(r"\w+", text) if any(c.isalpha() and 0x7F < ord(c) <= 0x24F for c in w)}
    )
    return {"non_en_segments": flagged, "non_latin_scripts": scripts, "accented_words": accented[:10]}


def main() -> None:
    from langid.langid import LanguageIdentifier, model
    from transformers import T5Tokenizer

    results, qrels = retrieve()
    run_file = RAW / "bm25" / "trec_results_covid.txt"
    bm25_ndcg10 = trec_eval_ndcg10(run_file)

    tokenizer = T5Tokenizer.from_pretrained(T5_MODEL, revision=T5_REVISION)

    # BEIR corpus/queries/qrels (the adapter rebuilds passage text from corpus.jsonl).
    needed = {h["docid"] for e in results for h in e["hits"]}
    corpus = {}
    with (BEIR_DIR / "corpus.jsonl").open() as fh:
        for line in fh:
            row = json.loads(line)
            if row["_id"] in needed:
                corpus[row["_id"]] = row
    beir_queries = {}
    with (BEIR_DIR / "queries.jsonl").open() as fh:
        for line in fh:
            row = json.loads(line)
            beir_queries[row["_id"]] = row["text"]
    beir_qrels: dict[str, dict[str, int]] = {}
    with (BEIR_DIR / "qrels" / "test.tsv").open() as fh:
        next(fh)
        for line in fh:
            q, d, s = line.rstrip("\n").split("\t")
            beir_qrels.setdefault(q, {})[d] = int(s)

    content_mismatch = sorted(
        {
            h["docid"]
            for e in results
            for h in e["hits"]
            if format_content(corpus[h["docid"]]["title"], corpus[h["docid"]]["text"]) != h["content"]
        }
    )
    query_mismatch = [str(e["hits"][0]["qid"]) for e in results if beir_queries[str(e["hits"][0]["qid"])] != e["query"]]
    qrels_frozen = {str(q): {str(d): int(v) for d, v in docs.items()} for q, docs in qrels.items()}
    qrels_diff = {
        q: {
            "only_pyserini": sorted(set(qrels_frozen.get(q, {})) - set(beir_qrels.get(q, {}))),
            "only_beir": sorted(set(beir_qrels.get(q, {})) - set(qrels_frozen.get(q, {}))),
            "label_diff": sorted(d for d in set(qrels_frozen.get(q, {})) & set(beir_qrels.get(q, {})) if qrels_frozen[q][d] != beir_qrels[q][d]),
        }
        for q in sorted(set(qrels_frozen) | set(beir_qrels), key=int)
    }
    qrels_diff = {q: v for q, v in qrels_diff.items() if any(v.values())}

    # Cross-check against castorini/rank_llm_data cached BM25 top-100 (same retriever lineage).
    rank_llm = {}
    if RANK_LLM_CACHE.exists():
        with RANK_LLM_CACHE.open() as fh:
            for line in fh:
                row = json.loads(line)
                rank_llm[str(row["query"]["qid"])] = [c["docid"] for c in row["candidates"]]
    rank_llm_same_order = sum(rank_llm.get(str(e["hits"][0]["qid"])) == [h["docid"] for h in e["hits"]] for e in results)
    rank_llm_same_set = sum(set(rank_llm.get(str(e["hits"][0]["qid"]), [])) == {h["docid"] for h in e["hits"]} for e in results)

    # Truncation + language screen (screen output is reviewed -> NON_ENGLISH).
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    passages = {}
    trunc_stats: Counter[str] = Counter()
    screen = {}
    for entry in results:
        for hit in entry["hits"]:
            docid = hit["docid"]
            if docid in passages:
                continue
            passage, info = token_prefix(tokenizer, hit["content"], PASSAGE_TOKENS)
            flags = language_screen(identifier, passage)
            if any(flags.values()):
                screen[docid] = flags
            passages[docid] = {
                "docid": docid,
                "content_chars": len(hit["content"]),
                "content_sha256": sha256_text(hit["content"]),
                "passage_chars": len(passage),
                "passage_sha256": sha256_text(passage),
                "passage_tokens_total": info["n_tokens"],
                "truncated": info["truncated"],
                "_paper_string_equal": info["paper_string_equal"],
                "_prefix_retokenizes_to_same": info.get("prefix_retokenizes_to_same", True),
            }
    unknown = sorted(set(NON_ENGLISH) - set(passages))
    if unknown:
        raise RuntimeError(f"NON_ENGLISH lists docs outside the candidate pool: {unknown}")
    kept = {d: p for d, p in passages.items() if d not in NON_ENGLISH}
    for p in kept.values():
        trunc_stats["docs"] += 1
        trunc_stats["truncated_at_128_tokens"] += p["truncated"]
        trunc_stats["paper_decoded_string_differs"] += not p.pop("_paper_string_equal")
        trunc_stats["prefix_retokenize_differs_at_boundary"] += not p.pop("_prefix_retokenizes_to_same")

    queries = []
    query_trunc = []
    query_screen = {}
    for entry in results:
        qid = str(entry["hits"][0]["qid"])
        query_input, info = token_prefix(tokenizer, entry["query"], QUERY_TOKENS)
        if info["truncated"]:
            query_trunc.append({"qid": qid, "query": entry["query"], "query_input": query_input})
        lang, prob = identifier.classify(entry["query"])
        if lang != "en":
            query_screen[qid] = [lang, round(float(prob), 3), entry["query"]]
        queries.append(
            {
                "qid": qid,
                "query": entry["query"],
                "query_input": query_input,
                "candidates": [
                    {"docid": h["docid"], "rank": h["rank"], "bm25_score": h["score"]}
                    for h in entry["hits"]
                    if h["docid"] not in NON_ENGLISH
                ],
                "excluded_non_english": [
                    {"docid": h["docid"], "rank": h["rank"], "bm25_score": h["score"]}
                    for h in entry["hits"]
                    if h["docid"] in NON_ENGLISH
                ],
                "qrels": qrels_frozen[qid],
            }
        )

    FROZEN.mkdir(parents=True, exist_ok=True)
    with (FROZEN / "queries.jsonl").open("w") as fh:
        for row in queries:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (FROZEN / "passages.jsonl").open("w") as fh:
        for docid in sorted(kept):
            fh.write(json.dumps(kept[docid], ensure_ascii=False) + "\n")

    # BM25 on the English-only candidate lists (the reference Jev is compared against).
    english_run = RAW / "bm25" / "trec_results_covid.english.txt"
    with english_run.open("w") as fh:
        for row in queries:
            for c in row["candidates"]:
                fh.write(f"{row['qid']} Q0 {c['docid']} {c['rank']} {c['bm25_score']} rank\n")
    bm25_english_ndcg10 = trec_eval_ndcg10(english_run)

    excluded_pairs = [(q["qid"], e) for q in queries for e in q["excluded_non_english"]]
    summary = {
        "n_queries": len(queries),
        "candidates_per_query_before_filter": sorted(Counter(len(e["hits"]) for e in results).items()),
        "candidates_per_query_after_filter": sorted(Counter(len(q["candidates"]) for q in queries).items()),
        "unique_docs_before_filter": len(passages),
        "unique_docs_after_filter": len(kept),
        "non_english_docs_excluded": len(NON_ENGLISH),
        "non_english_query_doc_pairs_excluded": len(excluded_pairs),
        "queries_with_exclusions": len({q for q, _ in excluded_pairs}),
        "excluded_pairs_by_qrel": dict(
            Counter(str(qrels_frozen[q].get(e["docid"], "unjudged")) for q, e in excluded_pairs)
        ),
        "excluded_pairs_in_bm25_top10": [
            [q, e["docid"], e["rank"], qrels_frozen[q].get(e["docid"])] for q, e in excluded_pairs if e["rank"] <= 10
        ],
        "bm25_ndcg_cut_10_trec_eval_full_top100": bm25_ndcg10,
        "bm25_ndcg_cut_10_trec_eval_english_candidates": bm25_english_ndcg10,
        "qrels_judgments": sum(len(v) for v in qrels_frozen.values()),
        "qrels_label_counts": dict(Counter(v for d in qrels_frozen.values() for v in d.values())),
        "qrels_diff_vs_beir_test_tsv": qrels_diff,
        "doc_content_mismatch_vs_beir_corpus": content_mismatch,
        "query_mismatch_vs_beir_queries": query_mismatch,
        "rank_llm_cache_same_order_queries": rank_llm_same_order,
        "rank_llm_cache_same_set_queries": rank_llm_same_set,
        "truncation": dict(trunc_stats),
        "queries_truncated_at_32_tokens": query_trunc,
        "query_langid_non_en": query_screen,
        "language_screen_flagged_docs": len(screen),
        "language_screen_flagged_kept": sorted(set(screen) - set(NON_ENGLISH)),
        "language_screen": screen,
        "query_order": [q["qid"] for q in queries],
        "sha256": {
            "queries.jsonl": sha256_file(FROZEN / "queries.jsonl"),
            "passages.jsonl": sha256_file(FROZEN / "passages.jsonl"),
            "trec_results_covid.txt": sha256_file(run_file),
            "trec_results_covid.english.txt": sha256_file(english_run),
            "retrieve_results_covid.json": sha256_file(RAW / "bm25" / "retrieve_results_covid.json"),
        },
    }
    (RAW / "prepare_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"language_screen", "query_order"}}, indent=2))


if __name__ == "__main__":
    main()
