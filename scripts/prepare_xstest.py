"""Fetch the XSTest (NAACL 2024) release and freeze the response-judge test set.

Source: https://github.com/paul-rottger/xstest at commit d7bb5bd (2025-02-24).
Prompts are CC-BY-4.0; the model completions are "subject to the original licenses
specified by Meta, Mistral and OpenAI" (README), so completion text is NOT
redistributed. This script:

  1. downloads the prompt file, the 5 human-annotated completion CSVs and the 10
     released automated-evaluation CSVs into data/raw/xstest/<repo path> (skipped
     if present) and checks every SHA-256;
  2. checks release consistency (450 ids per model in prompt order, prompt types
     match xstest_prompts.csv, "contrast_" <=> unsafe, auto-eval files carry the
     same prompt/completion/final_label);
  3. writes data/frozen/xstest/manifest.jsonl: one line per completion,
     {"id", "model", "row", "sha256"} (see gauntlet.tasks.xstest.row_sha256);
  4. recomputes paper Table 2 for manual, StrMatch and GPT-4 evaluation from the
     release, asserts it equals the printed table, and writes the two automated
     judges' agreement with the human labels plus human inter-annotator agreement
     to data/frozen/xstest/reference_evaluators.json.

Deterministic; re-running rewrites identical files.

    uv run python scripts/prepare_xstest.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sklearn.metrics import cohen_kappa_score  # noqa: E402

from gauntlet.task import Item, write_jsonl  # noqa: E402
from gauntlet.tasks.xstest import (  # noqa: E402
    COMPLIANCE,
    LABELS,
    MANIFEST,
    MODELS,
    PAPER_TABLE2,
    RAW,
    RAW_BASE_URL,
    REFERENCE,
    SAFETY,
    XSTest,
    completions_path,
    evaluator_metrics,
    read_csv,
    row_sha256,
    safety_of,
)

RAW_FILES = {
    "xstest_prompts.csv": "11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a",
    "model_completions/xstest_v2_completions_gpt4.csv": "71af323af8ecdec0efae23992317721441d2cfedddedeef3248f842f4e15c9ca",
    "model_completions/xstest_v2_completions_llama2new.csv": "6365cc55ce0a593c7ab8785853d02784be027d97199d4077912b4313cd010099",
    "model_completions/xstest_v2_completions_llama2orig.csv": "cbc52550e8ba8197a3e3d97a5af078d27bae4766737cd62df80270126a87ffc6",
    "model_completions/xstest_v2_completions_mistralguard.csv": "0ac03817b65e019a8f5b4b3eed805086f9f7fafe538ac9ebd156f80210af6903",
    "model_completions/xstest_v2_completions_mistralinstruct.csv": "6a146b65a5b96ec8249729bc2db459697175125998d7cbbcc88f7688ecf31dad",
    "evaluation/automated_evaluation_labels/xstest_v2_completions_gpt4_gpteval.csv": "ba3b1e79ace763998ecdb48bbec70e3a4d763d8402cbb740f58ed6d275fc324a",
    "evaluation/automated_evaluation_labels/xstest_v2_completions_gpt4_streval.csv": "5cc07ac42f62009d4ff33d9b5dc96c329d43ae8076db89712f7b56637368c5f7",
    "evaluation/automated_evaluation_labels/xstest_v2_completions_llama2new_gpteval.csv": "14cb194968a29e7c5d0c2e280657f2ff14ea52f173ca12bc786387abd5185a70",
    "evaluation/automated_evaluation_labels/xstest_v2_completions_llama2new_streval.csv": "605183b95c8859673ff23898456a1fea24fbbd5993b9033dc7c12159227e3555",
    "evaluation/automated_evaluation_labels/xstest_v2_completions_llama2orig_gpteval.csv": "a918c4d0c7355da5c80ce73dd7bc79b20422148b265851afe1e53c08e472ce48",
    "evaluation/automated_evaluation_labels/xstest_v2_completions_llama2orig_streval.csv": "538f284005f829af76fd48cbe82334ab1c593b8193b62421866cc56929cc9881",
    "evaluation/automated_evaluation_labels/xstest_v2_completions_mistralguard_gpteval.csv": "e3e0abcef20d95f3ea607e3db08104edb4e2593e38ce8fd517e69e06cc35260e",
    "evaluation/automated_evaluation_labels/xstest_v2_completions_mistralguard_streval.csv": "0b9d2a5df933194e05620825dbaee0ac8dc121843e34477610f0e1e8eeb42033",
    "evaluation/automated_evaluation_labels/xstest_v2_completions_mistralinstruct_gpteval.csv": "68adba2f51206f6ef30d3f5d755ef5e3f550b738b6f269220fca4067eb513794",
    "evaluation/automated_evaluation_labels/xstest_v2_completions_mistralinstruct_streval.csv": "a1ad24b837a13380a88f9748b39663add74cd992d5841db782271490dcbb41d3",
}
PROMPTS_PER_MODEL = 450
# Paper footnote 10 (p. 5382): GPT-4 refused to classify 11 MistrI responses;
# "We counted these cases as full compliance."
EXPECTED_GPT4_NON_LABELS = {"mistralinstruct": 11}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch() -> None:
    for relpath, expected in RAW_FILES.items():
        path = RAW / relpath
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading {relpath}", file=sys.stderr)
            tmp = path.with_suffix(".part")
            with urllib.request.urlopen(RAW_BASE_URL + relpath) as resp, tmp.open("wb") as fh:
                fh.write(resp.read())
            tmp.rename(path)
        got = sha256_file(path)
        if got != expected:
            sys.exit(f"{path}: sha256 {got} != expected {expected}")


def check(cond: bool, msg: str) -> None:
    if not cond:
        sys.exit(f"release check failed: {msg}")


def auto_labels(model: str, kind: str, rows: list[dict[str, str]]) -> list[str]:
    """Released automated labels, aligned to `rows`; GPT-4 non-answers -> full compliance."""
    column = {"gpteval": "gpt4_label", "streval": "strmatch_label"}[kind]
    auto = read_csv(f"evaluation/automated_evaluation_labels/xstest_v2_completions_{model}_{kind}.csv")
    check(len(auto) == len(rows), f"{model} {kind}: {len(auto)} rows")
    labels = []
    for a, r in zip(auto, rows):
        for field in ("id", "type", "prompt", "completion", "final_label"):
            check(a[field] == r[field], f"{model} {kind} {r['id']}: {field} differs from model_completions")
        labels.append(a[column])
    non_labels = sum(lab not in LABELS for lab in labels)
    check(non_labels == (EXPECTED_GPT4_NON_LABELS.get(model, 0) if kind == "gpteval" else 0), f"{model} {kind}: {non_labels} non-label outputs")
    return [lab if lab in LABELS else COMPLIANCE for lab in labels]


def main() -> None:
    fetch()
    prompts = {f"v2-{r['id']}": r for r in read_csv("xstest_prompts.csv")}
    check(len(prompts) == PROMPTS_PER_MODEL, "prompt count")
    check(sum(r["label"] == "safe" for r in prompts.values()) == 250, "250 safe prompts")

    manifest = []
    pred_gpt4: list[str] = []
    pred_str: list[str] = []
    ann1: list[str] = []
    ann2: list[str] = []
    prompt_text_diffs: set[str] = set()
    for model in MODELS:
        rows = read_csv(completions_path(model))
        check([r["id"] for r in rows] == list(prompts), f"{model}: ids not v2-1..v2-450 in order")
        for i, r in enumerate(rows):
            p = prompts[r["id"]]
            check(r["type"] == p["type"] and safety_of(r["type"]) == p["label"], f"{model} {r['id']}: type/label")
            check(r["final_label"] in LABELS and r["completion"].strip() != "", f"{model} {r['id']}: label/completion")
            check((r["agreement"] == "TRUE") == (r["annotation_1"] == r["annotation_2"]), f"{model} {r['id']}: agreement flag")
            if r["prompt"] != p["prompt"]:
                prompt_text_diffs.add(r["id"])
            manifest.append({"id": f"xstest-{model}-{r['id']}", "model": model, "row": i, "sha256": row_sha256(model, r)})
            ann1.append(r["annotation_1"])
            ann2.append(r["annotation_2"])
        pred_gpt4 += auto_labels(model, "gpteval", rows)
        pred_str += auto_labels(model, "streval", rows)
    # v2-195: the completion files hold the prompt the models actually answered; xstest_prompts.csv was
    # later edited (one possessive word changed).
    check(prompt_text_diffs == {"v2-195"}, f"unexpected prompt-text differences {sorted(prompt_text_diffs)}")

    write_jsonl(MANIFEST, manifest)
    items = XSTest().items()  # rebuild through the adapter: verifies every row hash
    check(len(items) == len(MODELS) * PROMPTS_PER_MODEL, "item count")

    evaluators = {"gpt4_evaluator": evaluator_metrics(items, pred_gpt4), "strmatch": evaluator_metrics(items, pred_str)}
    human = evaluator_metrics(items, [it.gold for it in items])
    for name, table_key, ev in (("annotation", "annotation", None), ("gpt4_evaluator", "gpt4_evaluator", evaluators["gpt4_evaluator"]), ("strmatch", "strmatch", evaluators["strmatch"])):
        for safety in SAFETY:
            for model in MODELS:
                cell = human["table2_refusal_rates_percent"][safety][model]
                got = cell["human"] if ev is None else ev["table2_refusal_rates_percent"][safety][model]["evaluator"]
                want = PAPER_TABLE2[table_key][safety][model]
                check(
                    abs(got["full"] - want[0]) < 0.05 and abs(got["partial"] - want[1]) < 0.05,
                    f"Table 2 {name}/{safety}/{model}: {got} vs paper {want}",
                )

    agree = sum(a == b for a, b in zip(ann1, ann2))
    reference = {
        "_source": "Derived from paul-rottger/xstest@d7bb5bd evaluation/automated_evaluation_labels/ "
        "(gpt4_label, strmatch_label) against model_completions/ final_label, by scripts/prepare_xstest.py. "
        "The paper prints only refusal rates (Table 2, reproduced exactly); agreement metrics are ours. "
        "GPT-4 non-answers (11 MistrI rows) counted as full compliance, as in paper footnote 10.",
        "evaluators": evaluators,
        "human_inter_annotator": {
            "n": len(ann1),
            "raw_agreement": agree / len(ann1),
            "cohen_kappa": float(cohen_kappa_score(ann1, ann2, labels=list(LABELS))),
            "paper_section_4_2_agreement_percent": {"llama2orig": 96.4, "llama2new": 95.8, "mistralinstruct": 97.6, "mistralguard": 93.8, "gpt4": 98.4},
        },
    }
    REFERENCE.write_text(json.dumps(reference, indent=2, ensure_ascii=False) + "\n")
    print(f"manifest: {len(manifest)} rows -> {MANIFEST.relative_to(REPO)}", file=sys.stderr)
    for name, ev in evaluators.items():
        print(f"{name}: acc {ev['accuracy']:.4f} macro-F1 {ev['macro_f1']:.4f} binary acc {ev['binary_refusal']['accuracy']:.4f}", file=sys.stderr)
    print(f"human inter-annotator agreement {agree / len(ann1):.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
