# /// script
# requires-python = ">=3.11"
# dependencies = ["nltk==3.9.1", "pandas==2.2.3", "scikit-learn==1.5.2", "numpy==2.1.3", "httpx", "scipy"]
# ///
"""Freeze the v3 development samples (train/dev/validation data only; never test).

Every v3 configuration (TTP rerank, Wiki-WD rerank + biases, MADE routing + per-code
thresholds, fitted heads) is fit on these samples and nothing else. Each sample is drawn
from the paper's own released train/dev/validation split, deterministically (seed 0), is
capped at ~3,000 items (to keep Jev cost low), and drops any item
whose text also occurs in the corresponding frozen test set. No Jev calls.

Outputs (texts are stored only where the licence allows, otherwise ID + SHA-256 rows
that the task adapters rebuild from data/raw/ with hash checks, as for the test sets):

  data/frozen/ttp/dev_v3.jsonl                 1,000 texts: 334 TRAM dev, 333 Procedures dev,
                                               333 Derived-procedures dev (CC BY 4.0, text kept)
  data/frozen/contempo/{matres,tbdense}_dev_v3_ids.jsonl  (+ data/raw/contempo/built/*_dev_v3.jsonl)
      TB-Dense: fit = 2,000 label-proportional pairs of the 22 standard train docs;
                dev = all pairs of the 5 standard dev docs.
      MATRES:   MATRES has no dev split; its train docs (TimeBank + AQUAINT) are split by
                document (seed 0, 20% of docs -> dev): fit = 2,000 label-proportional pairs of
                the fit docs, dev = up to 1,000 label-proportional pairs of the dev docs.
  data/frozen/smarter/{subset}_dev_v3.jsonl    fit = class-balanced K per class from the authors'
      train part, dev = class-balanced K per class from their val part (split_train_test seed
      123, the split whose test part the frozen test was sampled from), i.e. exactly the
      test's own k_shot_sample design (K = 400 / 400 / 150) applied to train and val.
  data/frozen/relemb/wikiwd_dev_v3.jsonl.gz    3,000 sentences of the release's validation file
      (semantic-graphs-filtered-validation.02_06.json), same edge filtering as the test freeze.
  data/frozen/made/dev_v3_manifest.jsonl       3,000 reports of the released validation split
      (val.jsonl, Jan-Jun 2024), idx + row SHA-256.
  data/frozen/v3_dev_manifest.json             counts, sources, SHA-256 of every output.

    uv run scripts/prepare_v3_dev.py
"""

from __future__ import annotations

import gzip
import hashlib
import html
import json
import random
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

import prepare_contempo as pc  # noqa: E402
import prepare_smarter as ps  # noqa: E402
import prepare_ttp as pt  # noqa: E402

from gauntlet.task import FROZEN, read_jsonl, write_jsonl  # noqa: E402
from gauntlet.tasks import contempo as contempo_task  # noqa: E402
from gauntlet.tasks import made as made_task  # noqa: E402
from gauntlet.tasks import relemb as relemb_task  # noqa: E402
from gauntlet.tasks import smarter as smarter_task  # noqa: E402

SEED = 0
SPLIT = "dev_v3"
MADE_VAL = made_task.HF_DIR / "val.jsonl"
MADE_VAL_SHA256 = "4857b2a5411c80d0c5ff1fb031574171516b7ce03eee2849ed2098f1b95c5752"  # HF Git-LFS SHA-256 @ HF_REVISION
WIKI_VAL_MEMBER = "WikipediaWikidataDistantSupervisionAnnotations.v1.0/enwiki-20160501/semantic-graphs-filtered-validation.02_06.json"
TBD_DOC_LISTS = REPO / "data" / "raw" / "contempo" / "EMNLP-2019" / "data" / "tbd"
MATRES_TRAIN = {"timebank": pc.RAW / "MATRES" / "timebank.txt", "aquaint": pc.RAW / "MATRES" / "aquaint.txt"}
TBAQ = pc.RAW / "te3_official" / "TBAQ-cleaned"
TBAQ_DIRS = {"timebank": TBAQ / "TimeBank", "aquaint": TBAQ / "AQUAINT"}

# Automatic English screen for train/dev text (the test screens were adjudicated flag by flag).
_FOREIGN = set(
    "der die das und nicht ist ich sie wir ein eine mit auf für auch "
    "el los las una por para con como pero más está muy porque "
    "les des une est pas pour avec dans sur mais très qui "
    "het een niet zijn maar voor wij ook "
    "não uma com para mas você muito "
    "il gli della che non sono per una anche questo ".split()
)


def looks_english(text: str) -> bool:
    """False when >5% of letters are non-ASCII or the text has 2+ distinct foreign function words."""
    letters = [c for c in text if c.isalpha()]
    if letters and sum(not c.isascii() for c in letters) / len(letters) > 0.05:
        return False
    return len(set(re.findall(r"[^\W\d_]+", text.lower())) & _FOREIGN) < 2


def stratified(rows: list[dict], n: int, seed: int, key: str) -> list[dict]:
    """Deterministic label-proportional sample of `n` rows (all rows, shuffled, when n >= len)."""
    rng = random.Random(seed)
    if n >= len(rows):
        out = list(rows)
        rng.shuffle(out)
        return out
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r[key], []).append(r)
    labels = sorted(by)
    quota = {lab: max(1, round(n * len(by[lab]) / len(rows))) for lab in labels}
    quota[max(labels, key=lambda lab: len(by[lab]))] += n - sum(quota.values())
    picked = [r for lab in labels for r in rng.sample(by[lab], min(quota[lab], len(by[lab])))]
    rng.shuffle(picked)
    return picked


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------
# TTP


def ttp() -> dict:
    cand_ids = {c["id"] for c in read_jsonl(FROZEN / "ttp" / "candidates.jsonl")}
    test_texts = {r["text"] for f in ("tram_test", "procedure_test", "derived_procedure_test", "expert_train", "expert_dev", "expert_test")
                  for r in read_jsonl(FROZEN / "ttp" / f"{f}.jsonl")}
    # Expert train/dev are part of the `ttp/expert_full` test set, so they are never used here.
    sources = {"tram": ("tram/tram_dev.tsv", "ttp-tram-dev", 604, 334),
               "procedures": ("procedures/procedure_dev.tsv", "ttp-procedure-dev", 1469, 333),
               "derived_procedures": ("derived_procedures/derived_procedure_dev.tsv", "ttp-derived-dev", 488, 333)}
    rows, stats = [], {}
    for subset, (rel, prefix, expected, n) in sources.items():
        pt.SPLITS[f"_{subset}_dev"] = (rel, prefix, expected)
        pool = pt.build_split(f"_{subset}_dev", cand_ids)
        kept = [r for r in pool if r["text"] not in test_texts]
        pick = random.Random(SEED).sample(kept, n)
        rows += [{**r, "subset": subset} for r in sorted(pick, key=lambda r: r["id"])]
        stats[subset] = {"release_file": rel, "sha256": sha256_file(pt.HF / rel), "pool": len(pool),
                         "dropped_text_in_test": len(pool) - len(kept), "sampled": n}
    path = FROZEN / "ttp" / f"{SPLIT}.jsonl"
    write_jsonl(path, rows)
    return {"file": str(path.relative_to(REPO)), "sha256": sha256_file(path), "n": len(rows), "sources": stats}


# --------------------------------------------------------------------------------------------
# ConTempo


def _contempo_row(doc: pc.Doc, ev1: pc.Event, ev2: pc.Event, rid: str, gold: str) -> dict:
    context, dist = pc.build_context(doc, ev1, ev2)
    return {"id": rid, "doc_id": doc.doc_id, "event1": ev1.text, "event2": ev2.text, "context": context,
            "sentence_distance": dist, "gold": gold}


def _freeze_contempo(subset: str, rows: list[dict], extra: dict) -> dict:
    built = contempo_task.BUILT_DIR / f"{subset}_{SPLIT}.jsonl"
    write_jsonl(built, rows)
    ids = FROZEN / "contempo" / f"{subset}_{SPLIT}_ids.jsonl"
    write_jsonl(ids, [{"id": r["id"], "sha256": contempo_task.row_sha256(r)} for r in rows])
    parts = Counter(r["id"].split(":")[0] for r in rows)
    return {"file": str(ids.relative_to(REPO)), "sha256": sha256_file(ids), "n": len(rows), "parts": dict(parts),
            "labels": {p: dict(Counter(r["gold"] for r in rows if r["id"].startswith(p + ":"))) for p in parts}, **extra}


def tbdense() -> dict:
    lists = {s: set((TBD_DOC_LISTS / f"{s}_docs.txt").read_text().split()) for s in ("train", "dev", "test")}
    assert lists["test"] == set(pc.TBDENSE_TEST_DOCS) and not (lists["train"] | lists["dev"]) & lists["test"]
    docs: dict[str, pc.Doc] = {}
    pools: dict[str, list[dict]] = {"fit": [], "dev": []}
    for line in pc.TBDENSE_FULL.read_text().splitlines():
        if not line.strip():
            continue
        doc_id, id1, id2, code = line.split("\t")
        part = "fit" if doc_id in lists["train"] else "dev" if doc_id in lists["dev"] else None
        if part is None or not (id1.startswith("e") and id2.startswith("e")):
            continue
        if doc_id not in docs:
            docs[doc_id] = pc.load_doc(pc.TIMEBANK_DIR / f"{doc_id}.tml")
        doc = docs[doc_id]
        if id1 not in doc.events or id2 not in doc.events:  # same TimeBank-event restriction as the test set
            continue
        pools[part].append(_contempo_row(doc, doc.events[id1], doc.events[id2], f"{part}:{doc_id}:{id1}:{id2}", pc.TBDENSE_CODES[code]))
    fit = stratified(pools["fit"], 2000, SEED, "gold")
    rows = sorted(fit, key=lambda r: r["id"]) + pools["dev"]
    return _freeze_contempo("tbdense", rows, {"pools": {k: len(v) for k, v in pools.items()},
                                              "source": "TB-Dense standard train (22) / dev (5) doc lists, rujunhan/EMNLP-2019"})


def matres() -> dict:
    by_doc: dict[str, list[tuple[str, list[str]]]] = {}
    for corpus, path in MATRES_TRAIN.items():
        for line in path.read_text().splitlines():
            if line.strip():
                parts = line.split("\t")
                by_doc.setdefault(parts[0], []).append((corpus, parts))
    doc_ids = sorted(by_doc)
    random.Random(SEED).shuffle(doc_ids)
    n_dev = round(0.2 * len(doc_ids))
    dev_docs = set(doc_ids[:n_dev])
    pools: dict[str, list[dict]] = {"fit": [], "dev": []}
    skipped = Counter()
    for doc_id in sorted(by_doc):
        corpus = by_doc[doc_id][0][0]
        doc = pc.load_doc(TBAQ_DIRS[corpus] / f"{doc_id}.tml")
        part = "dev" if doc_id in dev_docs else "fit"
        for _, (d, verb1, verb2, eiid1, eiid2, label) in by_doc[doc_id]:
            try:
                ev1 = doc.events[doc.eiid_to_eid[f"ei{eiid1}"]]
                ev2 = doc.events[doc.eiid_to_eid[f"ei{eiid2}"]]
            except KeyError:
                skipped["eiid not in TimeML"] += 1
                continue
            if (ev1.text, ev2.text) != (verb1, verb2):
                skipped["verb mismatch"] += 1
                continue
            try:
                pools[part].append(_contempo_row(doc, ev1, ev2, f"{part}:{doc_id}:ei{eiid1}:ei{eiid2}", label))
            except ValueError:
                skipped["event spans a sentence boundary / overlap"] += 1
    fit = stratified(pools["fit"], 2000, SEED, "gold")
    dev = stratified(pools["dev"], 1000, SEED, "gold")
    rows = sorted(fit, key=lambda r: r["id"]) + sorted(dev, key=lambda r: r["id"])
    return _freeze_contempo("matres", rows, {"pools": {k: len(v) for k, v in pools.items()}, "docs": len(doc_ids),
                                             "dev_docs": n_dev, "skipped": dict(skipped),
                                             "source": "MATRES timebank.txt + aquaint.txt (train), TBAQ-cleaned TimeML text"})


# --------------------------------------------------------------------------------------------
# SMARTER


def smarter(subset: str) -> dict:
    test = read_jsonl(FROZEN / "smarter" / f"{subset}.jsonl")
    task = smarter_task.Smarter(subset)
    test_items = task.items()
    test_texts = {it.fields["post"] for it in test_items}
    test_uids = {r["uid"] for r in test}
    df, _ = ps.hatexplain_frame() if subset == "hatexplain" else ps.latent_frame(subset)
    ratios, k = ps.CONFIG[subset]
    train_df, val_df, _ = ps.split_train_test(df, ratios, seed=ps.SPLIT_SEED)
    rows, stats = [], {}
    for part, frame in (("fit", train_df), ("dev", val_df)):
        mask = [(u not in test_uids and t not in test_texts and looks_english(t)) for u, t in zip(frame["uid"], frame["text"])]
        clean = frame[mask]
        sample = ps.k_shot_sample(clean, group_var="label", key_var="uid", k=k, random_state=SEED)
        stats[part] = {"pool": len(frame), "dropped": len(frame) - len(clean), "sampled": len(sample),
                       "labels": dict(Counter(sample["label"]))}
        for r in sample.itertuples():
            row = {"id": f"{subset}-{part}-{r.uid}", "part": part, "gold": r.label, "sha256": smarter_task.row_sha256(r.text, r.label)}
            if subset == "hatexplain":
                row["text"] = r.text
            else:
                row.update(uid=r.uid, row=int(r.row))
            rows.append(row)
    path = FROZEN / "smarter" / f"{subset}_{SPLIT}.jsonl"
    write_jsonl(path, rows)
    return {"file": str(path.relative_to(REPO)), "sha256": sha256_file(path), "n": len(rows), "k_per_class": k, "parts": stats}


# --------------------------------------------------------------------------------------------
# Wiki-WD


def wikiwd(n: int = 3000) -> dict:
    test_texts = {" ".join(r["tokens"]) for r in read_jsonl(relemb_task.DATA / "wikiwd_test.jsonl.gz")}
    labels = {lab["label"] for lab in json.loads((REPO / "prompts" / "relemb_wikiwd_v2.json").read_text())["labels"]}
    with zipfile.ZipFile(relemb_task.RAW / "WikipediaWikidataDistantSupervisionAnnotations.v1.0.zip") as zf, zf.open(WIKI_VAL_MEMBER) as fh:
        data = json.load(fh)
    pool, dropped = [], Counter()
    for i, sent in enumerate(data):
        edges = [[e["left"], e["right"], e["kbID"]] for e in sent["edgeSet"] if e["left"] and e["right"]]
        if not edges:
            dropped["no markable edge"] += 1
        elif any(e[2] not in labels for e in edges):
            dropped["label outside the inventory"] += 1
        elif " ".join(sent["tokens"]) in test_texts:
            dropped["sentence in test"] += 1
        elif not looks_english(html.unescape(" ".join(sent["tokens"]))):
            dropped["non-English (automatic screen)"] += 1
        else:
            pool.append({"id": f"wikiwd-val-{i}", "tokens": sent["tokens"], "edges": edges})
    rows = sorted(random.Random(SEED).sample(pool, n), key=lambda r: int(r["id"].rsplit("-", 1)[1]))
    path = relemb_task.DATA / f"wikiwd_{SPLIT}.jsonl.gz"
    write_jsonl(path, rows)
    gold = Counter(e[2] for r in rows for e in r["edges"])
    return {"file": str(path.relative_to(REPO)), "sha256": sha256_file(path), "n_sentences": len(rows), "n_edges": sum(gold.values()),
            "n_p0": gold["P0"], "n_labels": len(gold), "release_member": WIKI_VAL_MEMBER, "release_sentences": len(data),
            "pool": len(pool), "dropped": dict(dropped)}


# --------------------------------------------------------------------------------------------
# MADE


def made(n: int = 3000) -> dict:
    if sha256_file(MADE_VAL) != MADE_VAL_SHA256:
        raise SystemExit(f"{MADE_VAL}: SHA-256 differs from the pinned HF Git-LFS hash")
    test_texts = {r["text"] for r in made_task.load_raw_test()}
    rows = read_jsonl(MADE_VAL)
    pool = [r for r in rows if r["text"] not in test_texts and looks_english(r["text"])]
    pick = sorted(random.Random(SEED).sample(pool, n), key=lambda r: r["idx"])
    path = FROZEN / "made" / f"{SPLIT}_manifest.jsonl"
    write_jsonl(path, [{"idx": r["idx"], "sha256": made_task.row_sha256(r["idx"], r["text"], r["labels"])} for r in pick])
    return {"file": str(path.relative_to(REPO)), "sha256": sha256_file(path), "n": len(pick), "release_file": "val.jsonl",
            "release_sha256": MADE_VAL_SHA256, "release_rows": len(rows), "pool": len(pool),
            "gold_labels_mean": sum(len(r["labels"]) for r in pick) / len(pick)}


def main() -> None:
    manifest = {
        "seed": SEED,
        "rule": "train/dev/validation data only; <=~3,000 items per task; items whose text is in the frozen test set dropped",
        "ttp": ttp(),
        "contempo/tbdense": tbdense(),
        "contempo/matres": matres(),
        **{f"smarter/{s}": smarter(s) for s in smarter_task.SUBSETS},
        "relemb/wikiwd": wikiwd(),
        "made": made(),
    }
    out = FROZEN / "v3_dev_manifest.json"
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
