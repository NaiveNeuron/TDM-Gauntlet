"""Freeze the MADE (ACL 2026) label taxonomy, Jev questions and English test set.

Three stages, run in this order, all deterministic:

  uv run --with openpyxl python scripts/prepare_made.py taxonomy
      Public inputs only. Downloads (if absent) and hash-checks
        * head_medium_tail.json from raunak-agarwal/made-benchmark@9bff3ec (the
          1,154-code vocabulary and its training-frequency buckets; MIT), and
        * the IMDRF Adverse Event Terminology Release 2025 workbook ("Annexes A-G
          consolidated Release2025.xlsx", Wayback copy of imdrf.org; the release the
          paper accessed on 19-11-2025),
      then writes prompts/made.json (v1: one noul question per code, label text verbatim
      from Annexes A/E; a transitive poly-hierarchy) and data/frozen/made/label_buckets.json.

  uv run --with huggingface_hub --with openpyxl python scripts/prepare_made.py hierarchy
      Needs the gated Hugging Face release (a person must accept its access
      conditions at https://huggingface.co/datasets/ragarwal/MADE-Multilabel-Benchmark
      with the Hugging Face account whose token is cached locally). Downloads the
      release at the pinned revision to data/raw/made/hf/ and writes prompts/made_v2.json:
      the v1 questions unchanged, with the hierarchy taken from the release's
      code_to_parent_codes.json (the map the gold labels were up-propagated with): each
      code's ancestors are the codes on its own listed IMDRF paths (not transitive).

  uv run --with huggingface_hub --with langid==1.1.6 python scripts/prepare_made.py test
      Checks test.jsonl against its Git-LFS SHA-256, checks the vocabulary against
      prompts/made_v2.json, compares the release's terminology and ancestor map with
      it (differences are reported, not silently adopted), reports how the gold sets
      relate to the hierarchy, and runs the English screen:
        * without data/frozen/made/non_english.json it writes
          data/frozen/made/language_screen.json and stops: every flagged report
          must be read and the adjudicated non-English idx list recorded
          ({"idx": [...], "method": "..."}; an empty list is valid);
        * with it, it writes data/frozen/made/test_manifest.jsonl (idx + row SHA-256,
          file order, non-English rows removed), excluded_non_english.jsonl and
          stats.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gauntlet.task import write_jsonl
from gauntlet.tasks.made import (
    FROZEN_DIR,
    HF_DIR,
    HF_REPO,
    HF_REVISION,
    HF_URL,
    MAX_REPORT_CHARS,
    PAPER_TEST_N,
    RAW_DIR,
    TEST_FILE_SHA256,
    load_raw_test,
    row_sha256,
    sha256_file,
)

PROMPT_V1_PATH = REPO / "prompts" / "made.json"  # taxonomy stage output (frozen v1)
PROMPT_PATH = REPO / "prompts" / "made_v2.json"  # hierarchy stage output, used by the adapter
RELEASE_HIERARCHY_SHA256 = "50bd69e02333962a0bf2539a66138fdf976eac830d7077d59e57385c1d30fe1a"

GITHUB_COMMIT = "9bff3ec3b6adc3a3d56556fd00b37c3d3cef0186"
BUCKETS_URL = f"https://raw.githubusercontent.com/raunak-agarwal/made-benchmark/{GITHUB_COMMIT}/head_medium_tail.json"
BUCKETS_SHA256 = "3b7fdbc593ecdfda9d674519d4678513da571f4a49e47168f6495771e61fa5a9"
BUCKET_SIZES = {"head": 144, "medium": 481, "tail": 348, "extreme_tail": 181}  # paper Table 2 header

IMDRF_FILE = "imdrf_annexes_a-g_release2025.xlsx"
IMDRF_ORIGINAL_URL = "https://www.imdrf.org/sites/default/files/2025-03/Annexes%20A-G%20consolidated%20Release2025.xlsx"
IMDRF_URL = f"http://web.archive.org/web/20250316184945id_/{IMDRF_ORIGINAL_URL}"
IMDRF_SHA256 = "dcb6f3a8a57ea5fede88c6a3620b0e8131bf10d0dc1637e261589e7032a6d2ca"
ANNEX_TITLES = {"A": "Medical Device Problem", "E": "Health Effects - Clinical Signs and Symptoms or Conditions"}
CODE_RE = re.compile(r"^[AE]\d{2}(?:\d{2}){0,2}$")

STATE_KEY = "report"
# Jev question per code. Paper App. A.7 / prompts/system_prompt_k10.txt: "classifying a
# medical device adverse event report into one or more categories according to the FDA
# taxonomy", "Assign all the labels that are relevant to the report", label lines
# "{code}: {term} - {definition}" (prompts/README.md), and the hierarchy rule "when a child
# label ... is selected, the parent label ... must also be selected".
INSTRUCTIONS = (
    "Should the IMDRF adverse event code {code} be assigned to the medical device adverse event report in "
    "`report` (FDA taxonomy)? Assign it only if it is relevant to the report.{note} {code} ({path}): {term} - "
    "{definition}"
)
NON_LEAF_NOTE = " The code also applies whenever a more specific code under it applies."

HF_FILES = ("test.jsonl", "labels.json", "imdrf_terminology.json", "code_to_parent_codes.json")


def download(url: str, path: Path, sha256: str) -> Path:
    if not path.exists():
        print(f"downloading {url}", file=sys.stderr)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        req = urllib.request.Request(url, headers={"User-Agent": "TDM-Gauntlet/prepare_made"})
        with urllib.request.urlopen(req, timeout=300) as resp, tmp.open("wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        tmp.rename(path)
    got = sha256_file(path)
    if got != sha256:
        sys.exit(f"{path}: sha256 {got} != expected {sha256}")
    return path


# ---------------------------------------------------------------------------
# Stage 1: taxonomy + questions
# ---------------------------------------------------------------------------


def read_annex_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Annex A and E rows keyed by code (poly-hierarchical codes occur once per path)."""
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True)
    rows_by_code: dict[str, list[dict[str, Any]]] = {}
    for annex in ANNEX_TITLES:
        rows = list(wb[annex].iter_rows(values_only=True))
        assert any(str(c).strip() == f"Annex Name: Annex {annex}" for c in rows[0] if c), rows[0]
        assert "Release Number: 2025" in str(rows[2][0]), rows[2]
        header_at = next(i for i, r in enumerate(rows) if r[0] == "Level 1 Term")
        col = {h: i for i, h in enumerate(rows[header_at]) if h}
        for r in rows[header_at + 1 :]:
            code = r[col["Code"]]
            if not code:
                continue
            term = next(r[i] for i in range(3) if r[i])
            rows_by_code.setdefault(code, []).append(
                {
                    "term": str(term).strip(),
                    "definition": str(r[col["Definition"]]).strip(),
                    "status": r[col["Status"]],
                    "path": str(r[col["CodeHierarchy"]]).split("|"),
                }
            )
    return rows_by_code


def build_taxonomy(vocab: list[str], annex: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    in_vocab = set(vocab)
    tax: dict[str, dict[str, Any]] = {}
    for code in vocab:
        rows = annex[code]
        texts = {(r["term"], r["definition"]) for r in rows}
        assert len(texts) == 1, (code, texts)
        parents, on_paths = set(), set()
        for r in rows:
            assert r["path"][-1] == code, (code, r["path"])
            prev = [c for c in r["path"][:-1] if c in in_vocab]
            if prev:
                parents.add(prev[-1])  # nearest in-vocabulary predecessor on this path
            on_paths.update(prev)
        term, definition = texts.pop()
        prefix_path = [code[:k] for k in range(3, len(code) + 1, 2)]
        paths = [r["path"] for r in rows]
        tax[code] = {
            "term": term,
            "definition": definition,
            "status_release2025": sorted({str(r["status"]) for r in rows if r["status"]}),
            "parents": sorted(parents),
            # Path shown in the question: the code-prefix path when IMDRF lists it, else the first listed path.
            "display_path": prefix_path if prefix_path in paths else paths[0],
            "_on_paths": on_paths,
        }
    # Ancestors: transitive closure over every IMDRF path (a poly-hierarchy for Annex E).
    changed = True
    while changed:
        changed = False
        for code in vocab:
            closure = set(tax[code]["_on_paths"])
            for anc in list(closure):
                closure |= tax[anc]["_on_paths"]
            if closure != tax[code]["_on_paths"]:
                tax[code]["_on_paths"] = closure
                changed = True
    for code in vocab:
        tax[code]["ancestors"] = sorted(tax[code].pop("_on_paths"))
        assert code not in tax[code]["ancestors"], code
    return tax


def display_path(code: str, tax: dict[str, dict[str, Any]], annex: dict[str, list[dict[str, Any]]]) -> str:
    """Annex title followed by the terms of the code's ancestors on its displayed IMDRF path."""
    terms = [tax[c]["term"] if c in tax else annex[c][0]["term"] for c in tax[code]["display_path"][:-1]]
    return " > ".join([ANNEX_TITLES[code[0]], *terms])


def dump_one_entry_per_line(obj: dict[str, Any]) -> str:
    """JSON with top-level keys indented and each entry of a nested dict on its own line."""

    def enc(v: Any) -> str:
        return json.dumps(v, ensure_ascii=False)

    parts = []
    for key, value in obj.items():
        if isinstance(value, dict) and value and all(isinstance(v, dict) for v in value.values()):
            inner = ",\n".join(f"  {enc(k)}: {enc(v)}" for k, v in value.items())
            parts.append(f" {enc(key)}: {{\n{inner}\n }}")
        else:
            parts.append(f" {enc(key)}: {enc(value)}")
    return "{\n" + ",\n".join(parts) + "\n}\n"


def stage_taxonomy() -> None:
    buckets_path = download(BUCKETS_URL, RAW_DIR / "head_medium_tail.json", BUCKETS_SHA256)
    imdrf_path = download(IMDRF_URL, RAW_DIR / IMDRF_FILE, IMDRF_SHA256)
    buckets = json.loads(buckets_path.read_text())
    assert {k: len(v) for k, v in buckets.items()} == BUCKET_SIZES, {k: len(v) for k, v in buckets.items()}
    vocab = sorted(c for v in buckets.values() for c in v)
    assert len(vocab) == len(set(vocab)) == 1154 and all(CODE_RE.match(c) for c in vocab)

    annex = read_annex_rows(imdrf_path)
    missing = [c for c in vocab if c not in annex]
    assert not missing, f"codes absent from IMDRF Release 2025 Annexes A/E: {missing}"
    tax = build_taxonomy(vocab, annex)
    has_child = {p for c in vocab for p in tax[c]["parents"]}
    roots = [c for c in vocab if not tax[c]["parents"]]
    assert all(len(c) == 3 for c in roots) and len(roots) == 51, roots
    no_prefix_parent = [c for c in vocab if len(c) > 3 and c[:-2] not in tax[c]["parents"]]

    questions = {}
    for code in vocab:
        t = tax[code]
        text = INSTRUCTIONS.format(
            code=code,
            note=NON_LEAF_NOTE if code in has_child else "",
            path=display_path(code, tax, annex),
            term=t["term"],
            definition=t["definition"],
        )
        questions[code] = {"type": "noul", "instructions": text}

    prompt = {
        "_source": (
            "Agarwal et al., MADE, ACL 2026, https://aclanthology.org/2026.acl-long.2148/. Task framing and rules "
            "from the paper's prompting setup (App. A.7.1-A.7.2, pp. 46325-46328; identical text in "
            f"raunak-agarwal/made-benchmark@{GITHUB_COMMIT[:7]} prompts/system_prompt_k10.txt and "
            "user_prompt_template.txt): 'classifying a medical device adverse event report into one or more "
            "categories according to the FDA taxonomy', 'Assign all the labels that are relevant to the report', "
            "'Labels that start with \"A\" are Medical Device Problems. Labels that start with \"E\" are Health "
            "Effects - Clinical Signs and Symptoms or Conditions', and the rule that a selected child label implies "
            "its parent and grandparent. Each label is shown in the paper's label-line format '{code}: {term} - "
            "{definition}' (prompts/README.md), term and definition verbatim from IMDRF Adverse Event Terminology "
            "Release 2025, Annex A (Medical Device Problem) and Annex E (Health Effects - Clinical Signs and "
            "Symptoms or Conditions) - the paper's label source ('official IMDRF annexes', Section 2; IMDRF "
            "accessed 19-11-2025). The label vocabulary (1,154 codes) is the union of head_medium_tail.json."
        ),
        "_primitive": (
            "noul per IMDRF code (multi-label). Hierarchical traversal (see src/gauntlet/tasks/made.py): round 1 "
            "asks the 51 top-level codes; each later round asks the unasked codes with a parent accepted (P > 0.5) "
            "earlier, all of a round's questions in one request over the same state."
        ),
        "_additions": (
            "Not in the paper's prompt: the primary IMDRF hierarchy path in parentheses (annex title and the terms "
            "of the code's ancestors on its code-prefix path, or its first listed IMDRF path if IMDRF does not list the "
            "prefix path), and for codes with children the sentence 'The code also applies whenever "
            "a more specific code under it applies.' (the paper's up-propagation rule, Section 2). No few-shot "
            "examples (the paper's prompting baselines use 10 kNN-retrieved training reports; Jev is zero-shot)."
        ),
        "_label_source": {
            "file": IMDRF_FILE,
            "url": IMDRF_URL,
            "original_url": IMDRF_ORIGINAL_URL,
            "sha256": IMDRF_SHA256,
            "vocabulary": BUCKETS_URL,
            "vocabulary_sha256": BUCKETS_SHA256,
        },
        "state_key": STATE_KEY,
        "instructions_template": INSTRUCTIONS,
        "non_leaf_note": NON_LEAF_NOTE,
        "questions": questions,
        "taxonomy": {
            code: {k: tax[code][k] for k in ("term", "definition", "parents", "ancestors", "display_path", "status_release2025")}
            for code in vocab
        },
    }
    PROMPT_V1_PATH.write_text(dump_one_entry_per_line(prompt))
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    (FROZEN_DIR / "label_buckets.json").write_text(
        json.dumps(
            {
                "_source": f"{BUCKETS_URL} (sha256 {BUCKETS_SHA256}); training-set frequency buckets, "
                "head >1%, medium 0.1-1%, tail 0.01-0.1%, extreme tail <0.01% (paper Section 4)",
                "buckets": {k: sorted(v) for k, v in buckets.items()},
            },
            indent=1,
        )
        + "\n"
    )
    sizes = [len(json.dumps(q, ensure_ascii=False)) for q in questions.values()]
    print(
        json.dumps(
            {
                "codes": len(vocab),
                "roots": len(roots),
                "with_children": len(has_child),
                "multi_parent": sum(len(tax[c]["parents"]) > 1 for c in vocab),
                "not_under_code_prefix_parent": no_prefix_parent,
                "root_question_chars": sum(len(json.dumps(questions[c], ensure_ascii=False)) for c in roots),
                "question_chars_mean": sum(sizes) / len(sizes),
                "question_chars_max": max(sizes),
                "prompt_sha256": sha256_file(PROMPT_V1_PATH),
            },
            indent=1,
        )
    )


# ---------------------------------------------------------------------------
# Stage 2: gated test set
# ---------------------------------------------------------------------------


def fetch_release() -> None:
    if all((HF_DIR / f).exists() for f in HF_FILES):
        return
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    for name in HF_FILES:
        try:
            hf_hub_download(HF_REPO, name, repo_type="dataset", revision=HF_REVISION, local_dir=HF_DIR)
        except GatedRepoError:
            sys.exit(
                f"BLOCKED: {HF_REPO} is gated. A person must log in at {HF_URL} with the Hugging Face account "
                "whose token is cached here (`hf auth login`), accept the access conditions ('agree to share "
                "your contact information'), then re-run this stage. Alternatively place "
                f"{', '.join(HF_FILES)} from revision {HF_REVISION} in {HF_DIR}."
            )


def flatten_codes(obj: Any) -> set[str]:
    if isinstance(obj, str):
        return {c for c in obj.split("|") if CODE_RE.match(c.strip())}
    if isinstance(obj, dict):
        return set().union(*(flatten_codes(v) for v in obj.values())) if obj else set()
    if isinstance(obj, (list, tuple)):
        return set().union(*(flatten_codes(v) for v in obj)) if obj else set()
    return set()


def terminology_entry(entry: Any) -> tuple[str | None, str | None]:
    if not isinstance(entry, dict):
        return None, None
    low = {str(k).lower(): v for k, v in entry.items()}
    term = next((low[k] for k in ("term", "name", "label", "title") if k in low), None)
    definition = next((low[k] for k in ("definition", "description", "desc") if k in low), None)
    return (str(term).strip() if term is not None else None, str(definition).strip() if definition is not None else None)


def compare_release(prompt: dict[str, Any]) -> dict[str, Any]:
    """Release vocabulary must equal the frozen one; terminology/hierarchy differences are reported."""
    vocab = list(prompt["questions"])
    tax = prompt["taxonomy"]
    labels = json.loads((HF_DIR / "labels.json").read_text())
    release_vocab = set(labels) if isinstance(labels, (list, dict)) else set()
    if release_vocab != set(vocab):
        sys.exit(
            f"labels.json vocabulary differs from {PROMPT_PATH.name}: {len(release_vocab - set(vocab))} extra, "
            f"{len(set(vocab) - release_vocab)} missing"
        )

    terminology = json.loads((HF_DIR / "imdrf_terminology.json").read_text())
    if isinstance(terminology, list):
        terminology = {e.get("code"): e for e in terminology if isinstance(e, dict)}
    term_diff, def_diff, unparsed = [], [], []
    for code in vocab:
        term, definition = terminology_entry(terminology.get(code))
        if term is None and definition is None:
            unparsed.append(code)
            continue
        if term is not None and term != tax[code]["term"]:
            term_diff.append({"code": code, "release": term, "frozen": tax[code]["term"]})
        if definition is not None and definition != tax[code]["definition"]:
            def_diff.append({"code": code, "release": definition, "frozen": tax[code]["definition"]})

    parents_map = json.loads((HF_DIR / "code_to_parent_codes.json").read_text())
    anc_diff = []
    for code in vocab:
        release_anc = (flatten_codes(parents_map.get(code, [])) - {code}) & set(vocab)
        frozen_anc = set(tax[code]["ancestors"])
        if release_anc != frozen_anc:
            anc_diff.append(
                {"code": code, "release_only": sorted(release_anc - frozen_anc), "frozen_only": sorted(frozen_anc - release_anc)}
            )
    return {
        "vocabulary_equal": True,
        "terminology_unparsed_codes": unparsed,
        "term_differences": term_diff,
        "definition_differences": def_diff,
        "ancestor_differences": anc_diff,
    }


def release_paths(parents_map: dict[str, Any], code: str) -> list[list[str]]:
    paths = parents_map[code]
    assert isinstance(paths, list) and paths and all(isinstance(p, list) and p and p[-1] == code for p in paths), code
    return [list(p) for p in paths]


def stage_hierarchy() -> None:
    fetch_release()
    path = HF_DIR / "code_to_parent_codes.json"
    got = sha256_file(path)
    if got != RELEASE_HIERARCHY_SHA256:
        sys.exit(f"{path}: sha256 {got} != pinned {RELEASE_HIERARCHY_SHA256} (HF revision {HF_REVISION})")
    parents_map = json.loads(path.read_text())
    v1 = json.loads(PROMPT_V1_PATH.read_text())
    vocab = list(v1["questions"])
    in_vocab = set(vocab)
    annex = read_annex_rows(download(IMDRF_URL, RAW_DIR / IMDRF_FILE, IMDRF_SHA256))

    taxonomy: dict[str, dict[str, Any]] = {}
    workbook_path_diffs = []
    for code in vocab:
        paths = release_paths(parents_map, code)
        if sorted(map(tuple, paths)) != sorted(tuple(r["path"]) for r in annex[code]):
            workbook_path_diffs.append(code)
        parents, ancestors = set(), set()
        for p in paths:
            prev = [c for c in p[:-1] if c in in_vocab]
            if prev:
                parents.add(prev[-1])  # nearest in-vocabulary predecessor on this path
            ancestors.update(prev)
        old = v1["taxonomy"][code]
        taxonomy[code] = {
            "term": old["term"],
            "definition": old["definition"],
            "paths": paths,
            "parents": sorted(parents),
            "ancestors": sorted(ancestors),
            "display_path": old["display_path"],
            "status_release2025": old["status_release2025"],
        }
    roots = [c for c in vocab if not taxonomy[c]["parents"]]
    assert len(roots) == 51 and all(len(c) == 3 for c in roots), roots

    prompt = {k: v for k, v in v1.items() if k not in ("questions", "taxonomy")}
    prompt["_version"] = (
        "v2 of prompts/made.json, written before any Jev run on MADE. Questions are identical to v1. Only the "
        "hierarchy changed: v1 closed ancestors transitively over every IMDRF path, so a code also inherited the "
        "other parents of its parents (E120203-E120206 got E03 via E1202, E233701 got E05 via E2337). The gold "
        "labels were up-propagated with the release's code_to_parent_codes.json, where each code's ancestors are "
        "the codes on its own listed paths; v2 uses exactly that map. Parent sets (routing) are unchanged."
    )
    prompt["_hierarchy_source"] = {
        "file": "code_to_parent_codes.json",
        "hf_repo": HF_REPO,
        "hf_revision": HF_REVISION,
        "sha256": RELEASE_HIERARCHY_SHA256,
        "paths_equal_imdrf_release2025_workbook": not workbook_path_diffs,
    }
    prompt["questions"] = v1["questions"]
    prompt["taxonomy"] = taxonomy
    PROMPT_PATH.write_text(dump_one_entry_per_line(prompt))
    v1_tax = v1["taxonomy"]
    print(
        json.dumps(
            {
                "codes": len(vocab),
                "roots": len(roots),
                "codes_with_multiple_paths": sum(len(taxonomy[c]["paths"]) > 1 for c in vocab),
                "paths_differing_from_imdrf_workbook": workbook_path_diffs,
                "parents_changed_vs_v1": [c for c in vocab if taxonomy[c]["parents"] != v1_tax[c]["parents"]],
                "ancestors_changed_vs_v1": {
                    c: sorted(set(v1_tax[c]["ancestors"]) - set(taxonomy[c]["ancestors"]))
                    for c in vocab
                    if taxonomy[c]["ancestors"] != v1_tax[c]["ancestors"]
                },
                "prompt_sha256": sha256_file(PROMPT_PATH),
            },
            indent=1,
        )
    )


def gold_closure_report(rows: list[dict[str, Any]], taxonomy: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """How the gold sets relate to the hierarchy.

    per_label: a gold label whose own-path ancestors are not all gold. deepest: the gold set
    differs from the union of its deepest codes (no gold descendant) and their own-path
    ancestors, the rule by which the release up-propagated coder-assigned codes.
    """
    ancestors = {c: set(t["ancestors"]) for c, t in taxonomy.items()}
    descendants: dict[str, set[str]] = {c: set() for c in taxonomy}
    for c, anc in ancestors.items():
        for a in anc:
            descendants[a].add(c)
    per_label, deepest_diff = [], []
    causes: Counter[str] = Counter()
    for r in rows:
        gold = {label for label in r["labels"] if label in taxonomy}
        missing = {(label, a) for label in gold for a in ancestors[label] - gold}
        if missing:
            per_label.append(r["idx"])
            causes.update(f"{label} -> {a}" for label, a in missing)
        closure = set(gold)
        for d in (c for c in gold if not descendants[c] & gold):
            closure |= ancestors[d]
        if closure != gold:
            deepest_diff.append({"idx": r["idx"], "extra": sorted(closure - gold)})
    return {
        "not_closed_per_label": len(per_label),
        "not_closed_per_label_causes": dict(causes.most_common()),
        "not_closed_per_label_examples": per_label[:20],
        "differs_from_deepest_code_closure": len(deepest_diff),
        "differs_from_deepest_code_closure_rows": deepest_diff[:20],
    }


def description(text: str) -> str:
    return text.split("Description of Event:", 1)[-1]


def language_screen(identifier, text: str) -> dict[str, Any]:
    """Automated screen feeding the review of flagged reports (not itself a filter). Thresholds fixed a priori."""
    body = description(text)
    whole_lang, whole_p = identifier.classify(body)
    segments = [s for s in re.split(r"(?<=[.!?])\s+", body) if len(s.split()) >= 8]
    flagged = []
    for seg in segments:
        lang, p = identifier.classify(seg)
        if lang != "en" and p >= 0.99:
            flagged.append([lang, round(float(p), 3), seg[:160]])
    scripts = sorted({unicodedata.name(c, "?").split()[0] for c in text if c.isalpha() and ord(c) > 0x24F} - {"LATIN"})
    return {
        "description_lang": [whole_lang, round(float(whole_p), 3)],
        "non_en_segments": flagged,
        "non_latin_scripts": scripts,
        "flag": whole_lang != "en" or bool(flagged) or bool(scripts),
    }


def stage_test() -> None:
    fetch_release()
    rows = load_raw_test(HF_DIR.parent)  # checks the pinned LFS SHA-256
    if not PROMPT_PATH.exists():
        sys.exit(f"{PROMPT_PATH} is missing: run the `hierarchy` stage first.")
    prompt = json.loads(PROMPT_PATH.read_text())
    vocab = set(prompt["questions"])
    idxs = [r["idx"] for r in rows]
    assert len(idxs) == len(set(idxs)), "duplicate idx in test.jsonl"
    out_of_vocab = Counter(label for r in rows for label in r["labels"] if label not in vocab)
    release_check = compare_release(prompt)

    from langid.langid import LanguageIdentifier, model

    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    screen = {r["idx"]: language_screen(identifier, r["text"]) for r in rows}
    flagged = {idx: s for idx, s in screen.items() if s["flag"]}
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    (FROZEN_DIR / "language_screen.json").write_text(
        json.dumps(
            {
                "method": "langid 1.1.6 (norm_probs) on the event description; flag if the whole description is "
                "non-en, or any sentence of >= 8 words is non-en with p >= 0.99, or any non-Latin letter occurs. "
                "Every flagged report is then read; only the adjudicated list in non_english.json "
                "removes rows.",
                "n_rows": len(rows),
                "n_flagged": len(flagged),
                "flagged": {str(k): v for k, v in flagged.items()},
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n"
    )
    verdict_path = FROZEN_DIR / "non_english.json"
    if not verdict_path.exists():
        sys.exit(
            f"{len(flagged)} of {len(rows)} reports flagged -> {FROZEN_DIR / 'language_screen.json'}. "
            f"Read each flagged report in {HF_DIR / 'test.jsonl'}, then write {verdict_path} as "
            '{"idx": [<non-English idx>], "method": "<who/how>"} (empty list if none) and re-run.'
        )
    verdict = json.loads(verdict_path.read_text())
    drop = set(verdict["idx"])
    assert drop <= set(idxs), "non_english.json lists idx absent from test.jsonl"

    manifest = [{"idx": r["idx"], "sha256": row_sha256(r["idx"], r["text"], r["labels"])} for r in rows if r["idx"] not in drop]
    write_jsonl(FROZEN_DIR / "test_manifest.jsonl", manifest)
    write_jsonl(
        FROZEN_DIR / "excluded_non_english.jsonl",
        [{"idx": r["idx"], "sha256": row_sha256(r["idx"], r["text"], r["labels"])} for r in rows if r["idx"] in drop],
    )
    lengths = sorted(len(r["text"]) for r in rows if r["idx"] not in drop)
    stats = {
        "hf_repo": HF_REPO,
        "hf_revision": HF_REVISION,
        "test_jsonl_sha256": TEST_FILE_SHA256,
        "raw_sha256": {f: sha256_file(HF_DIR / f) for f in HF_FILES},
        "n_release": len(rows),
        "n_paper": PAPER_TEST_N,
        "n_excluded_non_english": len(drop),
        "n_frozen": len(manifest),
        "labels_per_report_mean": sum(len(r["labels"]) for r in rows) / len(rows),
        "gold_labels_out_of_vocab": dict(out_of_vocab),
        "gold_vs_hierarchy": gold_closure_report(rows, prompt["taxonomy"]),
        "n_labels_with_test_positives": len({l for r in rows for l in r["labels"] if l in vocab}),
        "text_chars": {
            "mean": sum(lengths) / len(lengths),
            "max": lengths[-1],
            f"over_{MAX_REPORT_CHARS}_truncated": sum(n > MAX_REPORT_CHARS for n in lengths),
        },
        "release_vs_frozen_prompt": release_check,
        "language_screen_flagged": len(flagged),
    }
    (FROZEN_DIR / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "release_vs_frozen_prompt"}, indent=1))
    print(
        json.dumps(
            {k: len(v) if isinstance(v, list) else v for k, v in release_check.items()},
            indent=1,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=("taxonomy", "hierarchy", "test"))
    args = parser.parse_args()
    {"taxonomy": stage_taxonomy, "hierarchy": stage_hierarchy, "test": stage_test}[args.stage]()


if __name__ == "__main__":
    main()
