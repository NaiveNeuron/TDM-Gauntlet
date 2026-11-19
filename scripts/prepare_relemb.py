"""Freeze the English relation-classification test sets of "Entity or Relation Embeddings?"
(Mtumbuka & Schockaert, Findings EMNLP 2024) and build their Jev prompts.

    uv run python scripts/prepare_relemb.py

Downloads use `uvx gdown` (Google Drive) and urllib (pinned GitHub raw files).

Sources (all pinned; hashes in data/manifests/relemb.json):
- Wiki-WD and NYT-FB: the RECON release (https://github.com/ansonb/RECON README, commit b157930),
  Google Drive files 1mmKLh6a78GVNizBoCGhs5ZMYJX2g-DIU (WikipediaWikidataDistantSupervision
  Annotations.v1.0.zip, Sorokin & Gurevych 2017, CC BY-SA 4.0) and 1VhY3i3SoLS3XZvtsk_BmpZgN5cjITlij
  (NYT.zip, Riedel et al. 2010 NYT-Freebase). Same files the paper's loaders read (their
  `wikidata/test.json` = semantic-graphs-filtered-held-out.02_06.json; `nyt/test.json` =
  dataset_triples_test.json, {"data": [...]}).
- Label names (v2 prompts, names only): the paper's inventories (fmtumbuka/RelationEmbeddings@66d3c9a
  experiment/__init__.py) and the dataset's 2016 Wikidata property labels (UKPLab/emnlp2017-relation-
  extraction@77bfd94 resources/properties-with-labels.txt).

Outputs:
  data/frozen/relemb/wikiwd_test.jsonl.gz   id, tokens, edges [[head positions, tail positions, PID]]
  data/frozen/relemb/nytfb_test.jsonl.gz    id, raw row indices, SHA-256 (text not redistributed)
  data/frozen/relemb/excluded.jsonl         dropped rows with reasons
  prompts/relemb_{wikiwd,nytfb}_v2.json      written only if absent, else verified
"""

from __future__ import annotations

import gzip
import hashlib
import html
import json
import re
import subprocess
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "relemb"
OUT = REPO / "data" / "frozen" / "relemb"
PROMPTS = REPO / "prompts"
sys.path.insert(0, str(REPO / "src"))

UA = {"User-Agent": "TDM-Gauntlet/0.1 (research audit; relemb)"}
GDRIVE = {
    "WikipediaWikidataDistantSupervisionAnnotations.v1.0.zip": (
        "1mmKLh6a78GVNizBoCGhs5ZMYJX2g-DIU", "31300761e83cc2b53f03ada2e1323d503b4e4febe9fd88481acdab4a30368902"),
    "NYT.zip": ("1VhY3i3SoLS3XZvtsk_BmpZgN5cjITlij", "7e0816de89f69545653e37fcd4deb59364a053ee9fbe4cd1fc181086cc6643c3"),
}
WIKI_MEMBER = "WikipediaWikidataDistantSupervisionAnnotations.v1.0/enwiki-20160501/semantic-graphs-filtered-held-out.02_06.json"
NYT_MEMBER = "NYT/dataset_triples_test.json"
PINNED_FILES = {
    "sorokin_properties-with-labels.txt": (
        "https://raw.githubusercontent.com/UKPLab/emnlp2017-relation-extraction/77bfd9424912d5357dad8cc31100dd2d59075080/resources/properties-with-labels.txt", "633b33291d66ddc1de47c6a92b8abdfcd6c7db7ddca8b415800b82841d148a65"),
    "paper_experiment_init.py": (
        "https://raw.githubusercontent.com/fmtumbuka/RelationEmbeddings/66d3c9a7fb4e72501514a0e239fe5451b3003d90/src/main/python/experiment/__init__.py", "9e9215fbe1338ca99128112d99777d0c95af33130d750bd1aee50f9745179ffe"),
}

PAPER_TASK = (
    "We consider the standard sentence-level relation extraction setting, where we are given a sentence "
    "in which two entities are highlighted, which we refer to as the head entity and tail entity. The goal "
    "is to predict which relationship holds between these two entities, given a pre-defined set of "
    "candidate relation labels."
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()


def ensure_downloads() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, (gid, sha) in GDRIVE.items():
        path = RAW / name
        if not path.exists():
            subprocess.run(["uvx", "gdown==5.2.0", gid, "-O", str(path)], check=True)
        if sha256_file(path) != sha:
            raise SystemExit(f"{path}: SHA-256 differs from the pinned release")
    for name, (url, sha) in PINNED_FILES.items():
        path = RAW / name
        if not path.exists():
            path.write_bytes(fetch(url))
        if sha256_file(path) != sha:
            raise SystemExit(f"{path}: SHA-256 differs from the pinned file")


# ---------------------------------------------------------------------------------- prompts (v2)
#
# The paper's classifier sees no label text: labels are class indices of the inventory lists
# (datapipeline/relational_dataset.py: `relation_type_idx = self._relation_types.index(...)`) fed to an
# MLP softmax (paper Section 3.2); the Limitations section states "we do not attempt to model the
# relation labels". v2 prompts therefore give Jev only the label NAMES (choice criteria with null
# descriptions, as TypeSafe's Choice docs allow): the dataset's own 2016 Wikidata property labels, the
# Freebase relation paths. The task statement goes once per request in the
# state; each question carries only its marked sentence.


def paper_inventories() -> dict[str, list[str]]:
    """Label lists from the paper's code (experiment/__init__.py)."""
    import ast

    src = (RAW / "paper_experiment_init.py").read_text()
    out = {}
    for name in ("WIKIDATA_RELATION_TYPES", "NYT_RELATION_TYPES"):
        out[name] = ast.literal_eval(re.search(rf"^{name} = (\[.*?\])", src, re.S | re.M).group(1))
    return out


def sorokin_labels() -> dict[str, str]:
    out = {}
    for line in (RAW / "sorokin_properties-with-labels.txt").read_text().splitlines():
        pid, label = line.split("\t", 1)
        out[pid] = label
    return out


QUESTION = (
    "Which relation holds from the head entity to the tail entity in this sentence?\n"
    "{marked_sentence}\nHead entity: {head}\nTail entity: {tail}"
)
MARKING = "In each question the head entity is marked <e1>...</e1> and the tail entity <e2>...</e2>."
SOURCE_COMMON = (
    "v2 (names only). Task text: Mtumbuka & Schockaert (2024) Section 3 (p. 6005), quoted. The paper's classifier uses "
    "no label text (labels are class indices: fmtumbuka/RelationEmbeddings@66d3c9a datapipeline/relational_dataset.py "
    "`relation_type_idx = self._relation_types.index(...)`, MLP softmax per Section 3.2; Limitations: 'we do not attempt "
    "to model the relation labels'), so options are label names with null descriptions. "
)


def wikiwd_prompt(inv: list[str]) -> dict:
    pids = sorted((p for p in inv if p != "P0"), key=lambda p: int(p[1:]))
    names = sorokin_labels()
    labels = [{"label": "P0", "key": "no relation"}] + [{"label": pid, "key": names[pid]} for pid in pids]
    half = (len(pids) + 1) // 2
    return {
        "_source": SOURCE_COMMON + (
            "Inventory: the paper's WIKIDATA_RELATION_TYPES (351 Wikidata properties + P0). Option names: the dataset's "
            "2016 property labels (UKPLab/emnlp2017-relation-extraction@77bfd94 resources/properties-with-labels.txt; all "
            "351 distinct); P0 = 'no relation' (Sorokin & Gurevych 2017 Section 3: pairs without a KB relation are "
            "negative instances). >255 labels: round 1 asks two fixed halves (numeric PID order, 176/175) each with the "
            "none option; round 2 asks between the two half winners (src/gauntlet/tasks/relemb.py)."
        ),
        "task": PAPER_TASK + (
            " The relation labels are Wikidata properties: the head entity is the subject item and the tail entity is the "
            "value of the property. Some questions list only part of the relation inventory; choose \"none of the listed "
            "relations\" when none of the listed relations holds from the head entity to the tail entity. "
        ) + MARKING,
        "question": QUESTION,
        "none_option": {"key": "none of the listed relations"},
        "negative_label": "P0",
        "halves": [pids[:half], pids[half:]],
        "labels": labels,
    }


def nytfb_prompt(inv: list[str]) -> dict:
    order = ["NA"] + [lab for lab in inv if lab != "NA"]
    labels = [{"label": "NA", "key": "NA (no relation)"}] + [{"label": lab, "key": lab} for lab in order[1:]]
    return {
        "_source": SOURCE_COMMON + (
            "Inventory: the paper's NYT_RELATION_TYPES (56 incl. NA). Option names: the Freebase relation paths verbatim; "
            "NA shown as 'NA (no relation)'."
        ),
        "task": PAPER_TASK + (
            " The relation labels are Freebase relations (/domain/type/property): the head entity is the subject and the "
            "tail entity is the value. Choose NA (no relation) when none of the listed relations holds from the head entity "
            "to the tail entity. "
        ) + MARKING,
        "question": QUESTION,
        "negative_label": "NA",
        "labels": labels,
    }


def write_or_verify_prompt(name: str, prompt: dict) -> None:
    path = PROMPTS / name
    text = json.dumps(prompt, indent=2, ensure_ascii=False) + "\n"
    if not path.exists():
        path.write_text(text)
        print(f"wrote {path}")
    elif path.read_text() != text:
        print(f"WARNING: {path} differs from the rebuilt prompt; frozen file kept")


# ---------------------------------------------------------------------------------- freezing


def write_jsonl(path: Path, rows: list[dict]) -> None:
    from gauntlet.task import write_jsonl as _w

    _w(path, rows)


def canonical_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_exclusions() -> dict[str, list]:
    return json.loads((OUT / "non_english.json").read_text())


def freeze_wikiwd(inv: list[str], excluded: list[dict]) -> None:
    with zipfile.ZipFile(RAW / "WikipediaWikidataDistantSupervisionAnnotations.v1.0.zip") as zf, zf.open(WIKI_MEMBER) as fh:
        data = json.load(fh)
    assert len(data) == 360334, len(data)
    non_en = {r["index"]: r["sentence_sha256"] for r in load_exclusions()["wikiwd"]}
    labels = set(inv)
    rows, n_edges = [], 0
    for i, sent in enumerate(data):
        edges = []
        for e in sent["edgeSet"]:
            n_edges += 1
            if not e["left"] or not e["right"]:
                excluded.append({"dataset": "wikiwd", "sentence": i, "edge": [e["left"], e["right"], e["kbID"]], "reason": "empty entity span (unmarkable; the paper's loader also skips it)"})
                continue
            assert e["kbID"] in labels, e["kbID"]
            edges.append([e["left"], e["right"], e["kbID"]])
        if i in non_en:
            text = html.unescape(" ".join(sent["tokens"]))
            assert hashlib.sha256(text.encode("utf-8")).hexdigest() == non_en[i], i
            excluded.append({"dataset": "wikiwd", "sentence": i, "n_edges": len(edges), "reason": "non-English (frozen language screen)"})
            continue
        if edges:
            rows.append({"id": f"wikiwd-{i}", "tokens": sent["tokens"], "edges": edges})
    assert n_edges == 740963, n_edges
    write_jsonl(OUT / "wikiwd_test.jsonl.gz", rows)
    gold = Counter(e[2] for r in rows for e in r["edges"])
    print(f"wikiwd: {len(rows)} sentences, {sum(gold.values())} edges ({gold['P0']} P0), {len(gold)} labels")


def freeze_nytfb(inv: list[str], excluded: list[dict]) -> None:
    with zipfile.ZipFile(RAW / "NYT.zip") as zf, zf.open(NYT_MEMBER) as fh:
        data = json.load(fh)["data"]
    assert len(data) == 172448, len(data)
    non_en = {r["sentence_sha256"] for r in load_exclusions()["nytfb"]}
    labels = set(inv)
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(data):
        if row["rel"] not in labels:
            excluded.append({"dataset": "nytfb", "row": i, "rel": row["rel"], "reason": "label outside the paper's 56-label inventory (the paper's loader skips it)"})
            continue
        key = hashlib.sha256(row["sent"].encode("utf-8")).hexdigest()
        if key in non_en:
            excluded.append({"dataset": "nytfb", "row": i, "reason": "non-English (frozen language screen)"})
            continue
        groups.setdefault(row["sent"], []).append(i)
    from gauntlet.tasks.relemb import nyt_row_sha256, nyt_spans

    rows = []
    for k, (sent, idx) in enumerate(groups.items()):
        for i in idx:
            nyt_spans(sent, data[i]["sub"], data[i]["obj"])  # every mention must be locatable
        rows.append({"id": f"nytfb-{k}", "rows": idx, "sha256": canonical_sha256([nyt_row_sha256(data[i]) for i in idx])})
    write_jsonl(OUT / "nytfb_test.jsonl.gz", rows)
    gold = Counter(data[i]["rel"] for r in rows for i in r["rows"])
    print(f"nytfb: {len(rows)} sentences, {sum(gold.values())} rows ({gold['NA']} NA), {len(gold)} labels")


def main() -> None:
    ensure_downloads()
    OUT.mkdir(parents=True, exist_ok=True)
    inv = paper_inventories()
    write_or_verify_prompt("relemb_wikiwd_v2.json", wikiwd_prompt(inv["WIKIDATA_RELATION_TYPES"]))
    write_or_verify_prompt("relemb_nytfb_v2.json", nytfb_prompt(inv["NYT_RELATION_TYPES"]))
    excluded: list[dict] = []
    freeze_wikiwd(inv["WIKIDATA_RELATION_TYPES"], excluded)
    freeze_nytfb(inv["NYT_RELATION_TYPES"], excluded)
    write_jsonl(OUT / "excluded.jsonl", excluded)
    print(Counter((e["dataset"], e["reason"]) for e in excluded))


if __name__ == "__main__":
    main()
