"""RQ1 statistics for the 46 zero-shot test sets (offline; no Jev or LLM calls).

Inputs: the saved zero-shot predictions (runs/<task>/predictions.jsonl[.gz]), the frozen test sets
(data/frozen), results/paper_numbers.json (published values, each with table/page source), the Table 2/11
generator scripts/paper_tables.py (task-specific reference, applied LLM baseline, point-estimate groups) and,
for reference only, results/framing_audit_ci.json (the paper's current interval counts). The metric code and
per-set resampling specs are scripts/framing_audit_ci.py (`build`), which patches the network to raise.

Definitions
(a) Comparator type of each set's task-specific reference (the value paper_tables.rows() compares against):
    trained    = trained/fine-tuned on the paper's own labelled data (incl. few-shot fine-tuning, prompt tuning,
                 reward models fit on the paper's pairs, conformal sets on a fine-tuned classifier);
    untrained  = a component with no training on the task's labels (zero-shot LLM judge or pipeline, zero-shot
                 Flan-T5 pairwise ranking, a pretrained NLI metric whose only fit is one threshold on 40 val items);
    no_labels  = trained without human labels (PiNose: probe trained on offline self-consistency pseudo-labels).
    Encoded per set in COMPARATOR with a source note (paper_numbers.json method row, the appendix task section
    paper/generated/app_tasks.tex).
(b) Jev vs the task-specific reference v (fixed published value). theta_hat = Jev's saved score.
    Bootstrap: B = 10,000 resamples of the units framing_audit_ci.build() defines (items, queries, documents,
    texts, ...), numpy default_rng(0) per set and variant. Clustered bootstrap where several units share one
    source unit (CLUSTER: XSTest prompt x 5 models, FactPICO abstract x 3 summarizers, FENICE narrative,
    ConTempo document, INLI premise, EZ-STANCE tweet (B: within domain strata), LifeTox query, Edit-intent
    document, MisMatched paper, DREsS writing prompt): whole clusters are drawn with replacement (within strata
    where the spec stratifies) and the metric is recomputed on the pooled members. SE = bootstrap SD.
    Two-sample widening (WIDEN): where the published value is computed on a different item set from ours
    (reconstructed split, >1% of the paper's items missing, or a different item subset/protocol), the published
    value is treated as an estimate with the same sampling variance as Jev's on our set, so
    SE_eff = sqrt(2) * SE; otherwise SE_eff = SE (published value fixed).
    Tests (normal approximation with SE_eff): difference p = 2 * Phi(-|theta_hat - v| / SE_eff); equivalence
    (TOST, margin delta) p = max(1 - Phi((theta_hat - v + delta)/SE_eff), Phi((theta_hat - v - delta)/SE_eff)).
    SE_eff = 0 (CICC C150/ACID/ATIS: every resample gives 0.0) gives p = 0 / 1 by the sign.
    Holm within the family of the 46 difference tests, and separately within the 46 equivalence tests.
    Class: separated above/below if Holm-adjusted difference p < .05 (sign of theta_hat - v); else parity if
    Holm-adjusted TOST p < .05; else inconclusive. Primary delta = 1 point (0.01 on the 0-1 scale): the
    published references are printed at 0.1-1 point resolution (11 of the 46 at two decimals, i.e. 1 point),
    the paper's grouping tolerance is 0.5 point, and the 5-point margin of the current audit is
    wide relative to the printed resolution. Sensitivity: delta in {0.5, 1, 2, 5} points; no clustering; no
    widening; widening on all 46; no Holm (unadjusted 95% / 90% normal intervals).
    Counts per set, per paper (bibkey) and by comparator type.
(c) LLM baseline. Prompted-LLM rows of each paper are encoded in PROMPTED (paper_numbers.json keys, shots,
    source). Rule actually applied in scripts/paper_tables.py (DATASETS column 5), verified here: the best
    zero-shot prompted LLM where the paper reports one, else the best few-shot one (INLI 8-shot, MedReadMe
    5-shot); exception: MADE uses GPT-4.1 (43.0) although every MADE prompted row is kNN few-shot and the best
    is GPT-5 (thinking) 54.0. Alternative rule: best reported prompted LLM, few-shot included. Fine-tuned LLM
    rows are excluded (ConTempo fine-tuned GPT-3.5, LegalSeg RhetoricLLaMA, MisMatched Llama-2 LoRA).
    Point-estimate groups exactly as paper_tables.rows(): replaces if jev >= method - 0.005; between if
    jev >= llm - 0.005 (no LLM: jev/method >= 0.8); else below. "Jev >= LLM": strict (jev >= llm) and with
    the 0.5-point tolerance (jev >= llm - 0.005), over the sets with an LLM baseline. Interval classes vs the
    LLM use (b)'s primary settings with Holm within the sets that have an LLM value.

Run from the repo root: uv run python scripts/offline_rq1.py   (`--B=N` for a dry run that writes nothing)
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import framing_audit_ci as F  # noqa: E402  (patches httpx / JevClient to raise)
import paper_tables  # noqa: E402
from gauntlet.tasks import REGISTRY  # noqa: E402

OUT = ROOT / "results" / "offline_v4" / "rq1.json"
TEX = ROOT / "paper" / "generated"
B = 10_000
SEED = 0
ALPHA = 0.05
DELTA = 0.01  # primary equivalence margin (1 point)
DELTAS = (0.005, 0.01, 0.02, 0.05)
TOL = 0.005  # paper_tables grouping tolerance

# ------------------------------------------------------------------ (a) comparator type per set
_PN = "results/paper_numbers.json"
COMPARATOR: dict[str, tuple[str, str]] = {
    **{f"pinose/{s}": ("no_labels", "PiNose probe on Llama2-7B hidden states, trained on offline self-consistency "
                       "pseudo-labels, no human annotation (paper title; app_tasks.tex PiNose)") for s in
       ("true_false", "nq", "triviaqa", "webq")},
    "prp_graph/trec_covid": ("untrained", "PRP-Graph-40 = zero-shot Flan-T5-L pairwise prompting + graph aggregation "
                             f"({_PN} method row; app_tasks.tex PRP-Graph)"),
    "xstest": ("untrained", f"GPT-4 zero-shot response judge (App. D), same value as the LLM column ({_PN})"),
    "factpico": ("untrained", f"GPT-4 + PICO-R Extract, a prompted pipeline ({_PN} method row)"),
    "fenice": ("untrained", "FENICE: pretrained NLI metric over T5-extracted claims, one threshold tuned on 40 val "
               f"items ({_PN}; app_tasks.tex FENICE)"),
    "biorc800": ("trained", f"LLM-SSC with WeighCon, fine-tuned on the paper's training abstracts ({_PN})"),
    "sciprompt/arxiv": ("trained", "SciPromptSoft 50-shot prompt tuning (paper_tables.TASK_SPECIFIC)"),
    "sciprompt/s2orc": ("trained", "SciPrompt 50-shot prompt tuning (paper_tables.TASK_SPECIFIC)"),
    "sciprompt/sdpra": ("trained", "full-set fine-tuning (paper_tables.TASK_SPECIFIC)"),
    "sciprompt/emerging_nlp": ("trained", "SciPrompt 5-shot prompt tuning (paper_tables.TASK_SPECIFIC)"),
    **{f"scalar_constructs/{c}/{m}": ("trained", "DeBERTa-v3-large reward model fit on the paper's pairs (Table 3, 5-fold "
                                      "mean; paper_tables.TASK_SPECIFIC for pairwise)")
       for c in ("immigration_fear", "ad_negativity", "grandstanding") for m in ("pairwise", "pointwise")},
    "lifetox": ("trained", f"RoBERTa-LifeTox 350M fine-tuned on LifeTox ({_PN})"),
    "edit_intent": ("trained", f"SeqC Llama3-8B-Instruct fine-tuned on Re3-Sci ({_PN})"),
    "ezstance/subtaskA": ("trained", f"BART-MNLI-ep fine-tuned on EZ-STANCE ({_PN})"),
    "ezstance/subtaskB": ("trained", f"BART-MNLI-ep fine-tuned on EZ-STANCE ({_PN})"),
    "contempo/matres": ("trained", f"ConTempo + Unified Framework, fine-tuned ({_PN})"),
    "contempo/tbdense": ("trained", f"ConTempo + Unified Framework, fine-tuned ({_PN})"),
    "mismatched": ("trained", f"SciBERT fine-tuned on MSciNLI+ (out-of-domain test) ({_PN})"),
    "inli": ("trained", f"T5-XXL fine-tuned on INLI ({_PN})"),
    "medreadme": ("trained", f"RoBERTa-large fine-tuned on MedReadMe ({_PN})"),
    "nav_unknown/clinc150": ("trained", "Hybrid (balanced): uses training data (app_tasks.tex Navigating the Unknown)"),
    "nav_unknown/banking77": ("trained", "Hybrid (balanced): uses training data (app_tasks.tex Navigating the Unknown)"),
    "made": ("trained", f"Discriminative FT Llama-3.1-8B-Base ({_PN})"),
    "smarter/hatexplain": ("trained", f"SMARTER Llama DPO on 256 labelled examples ({_PN})"),
    "smarter/latent_hate": ("trained", f"SMARTER Llama DPO on 256 labelled examples ({_PN})"),
    "smarter/implicit_hate": ("trained", f"SMARTER Llama DPO on 256 labelled examples ({_PN})"),
    "dress_new": ("trained", f"SFT Llama 3.1 8B on DREsS ({_PN})"),
    **{f"ttp/{s}": ("trained", f"Asymmetric NCE matcher trained on TRAM/Procedures+ ({_PN})")
       for s in ("tram", "procedures_plus", "expert")},
    "legalseg": ("trained", f"Hier_BiLSTM CRF trained on LegalSeg ({_PN})"),
    "relemb/wikiwd": ("trained", f"EnCore+Mask relation classifier trained on Wiki-WD ({_PN})"),
    "relemb/nytfb": ("trained", f"EnCore+Mask relation classifier trained on NYT-FB ({_PN})"),
    **{f"cicc/{s}": ("trained", f"conformal sets on fine-tuned BERT probabilities ({_PN})")
       for s in ("mtod", "hwu64", "c150", "acid", "atis")},
}
TYPES = ("trained", "untrained", "no_labels")
TYPE_TEX = {"trained": "trained", "untrained": "untrained", "no_labels": "no labels"}

# ------------------------------------------------------------------ (b) widening rule per set
WIDEN: dict[str, str] = {
    "smarter/hatexplain": "test files unreleased; split reconstructed with the authors' code",
    "smarter/latent_hate": "test files unreleased; split reconstructed with the authors' code",
    "smarter/implicit_hate": "test files unreleased; split reconstructed with the authors' code",
    "nav_unknown/clinc150": "no split/seeds released; test subset and ID/OOD partitions reconstructed",
    "nav_unknown/banking77": "no split/seeds released; test subset and ID/OOD partitions reconstructed",
    "dress_new": "6:2:2 seed-22 split unreleased (reconstructed); 55 empty essays dropped (N 401 vs 456)",
    "fenice": "public release has 319 of the paper's 338 pairs; scored on the released 279-item test file",
    "ezstance/subtaskA": "English-only: 7,666 of the paper's 7,798 (1.7% fewer)",
    "ezstance/subtaskB": "English-only: 45,274 of the paper's 46,011 (1.6% fewer)",
    **{f"scalar_constructs/{c}/{m}": "RECONSTRUCTION: release-notebook item set, not the paper's V_test; pairwise rows "
       "compare fold-0 with the reward model's 5-fold mean" for c in ("immigration_fear", "ad_negativity", "grandstanding")
       for m in ("pairwise", "pointwise")},
}
SAME_NOTE = ("same items as the published value (released test split; <1% non-English rows dropped where noted: "
             "SciPrompt <=0.7%, PRP-Graph 0.98% of candidate pairs, LegalSeg 6 sentences, Wiki-WD 0.08%)")


# ------------------------------------------------------------------ cluster keys
def _frozen(rel: str, field: str) -> dict[str, str]:
    rows = [json.loads(line) for line in (ROOT / "data" / "frozen" / rel).read_text().splitlines() if line.strip()]
    return {r["id"]: str(r[field]) for r in rows}


def cluster_keys(name: str, items: list) -> tuple[str, list[str]] | None:
    """(cluster unit, key per bootstrap unit in framing_audit_ci.build() order) or None (units independent)."""
    if name == "xstest":
        return "prompt", [it.fields["prompt_id"] for it in items]
    if name == "factpico":
        abstract = _frozen("factpico/test.jsonl", "abstract_id")
        return "abstract", [abstract[it.id] for it in items]
    if name == "fenice":
        return "narrative", [it.fields["narrative"] for it in items if it.fields["split"] == "test"]
    if name.startswith("contempo/"):
        return "document", [it.id.split(":")[0] for it in items]
    if name == "inli":
        return "premise", [it.fields["group"] for it in items]
    if name == "ezstance/subtaskA":
        return "tweet", [it.fields["text"] for it in items]
    if name == "ezstance/subtaskB":
        return "tweet", [f"{it.fields['domain']}\x00{it.fields['text']}" for it in items]
    if name == "lifetox":
        return "query", [it.fields["query"] for it in items]
    if name == "edit_intent":
        doc = _frozen("edit_intent/test.jsonl", "doc_name")
        return "document", [doc[it.id] for it in items]
    if name == "mismatched":
        paper = _frozen("mismatched/test.jsonl", "paper_id")
        return "paper", [paper[it.id] for it in items]
    if name == "dress_new":
        return "prompt", [it.fields["prompt"] for it in items]
    return None


class ClusterDraw:
    """Draw whole clusters with replacement (within strata) and expand to unit indices."""

    def __init__(self, keys: list[str], strata: list[np.ndarray] | None):
        ids = {k: i for i, k in enumerate(dict.fromkeys(keys))}
        cid = np.array([ids[k] for k in keys])
        self.order = np.argsort(cid, kind="stable")
        self.sizes = np.bincount(cid)
        self.starts = np.r_[0, np.cumsum(self.sizes)[:-1]]
        self.n_clusters = len(ids)
        if strata is None:
            self.groups = [np.arange(self.n_clusters)]
        else:
            unit_stratum = np.empty(len(keys), dtype=np.int64)
            for s, members in enumerate(strata):
                unit_stratum[members] = s
            first = unit_stratum[self.order[self.starts]]
            if not all((unit_stratum[cid == c] == first[c]).all() for c in range(self.n_clusters)):
                raise ValueError("a cluster spans several strata")
            self.groups = [np.flatnonzero(first == s) for s in range(len(strata))]

    def __call__(self, rng: np.random.Generator) -> np.ndarray:
        ch = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in self.groups])
        sz = self.sizes[ch]
        offs = np.repeat(self.starts[ch] - (np.cumsum(sz) - sz), sz)
        return np.sort(self.order[offs + np.arange(int(sz.sum()))])


# ------------------------------------------------------------------ (c) prompted LLM rows (paper_numbers.json keys)
# (key, shots, note); every key's value and source are in results/paper_numbers.json[task].
PROMPTED: dict[str, list[tuple[str, str]]] = {
    "xstest": [("GPT-4 zero-shot response judge (App. D)", "0")],
    "factpico": [("GPT-4 zero-shot", "0"), ("Alpaca zero-shot", "0"), ("Llama-2 zero-shot", "0"), ("Mistral zero-shot", "0")],
    "biorc800": [("LLM-SSC ICL 0-shot Gemma-2b", "0"), ("LLM-SSC ICL 5-shot Gemma-2b", "few")],
    **{f"sciprompt/{s}": [(k, "0") for k in ("Llama3-70B zero-shot", "ChatGPT zero-shot", "Llama2-70B zero-shot")]
       for s in ("arxiv", "s2orc", "sdpra")},
    "sciprompt/emerging_nlp": [("Llama2-70B zero-shot", "0")],
    "scalar_constructs/immigration_fear/pairwise": [
        ("Qwen-2.5-72B pairwise 0-shot", "0"), ("Llama-3.3-70B pairwise 0-shot", "0"), ("Qwen-2.5-32B pairwise 0-shot", "0"),
        ("Qwen-2.5-7B pairwise 0-shot", "0"), ("Llama-3.1-8B pairwise 0-shot", "0"),
        ("Qwen-2.5-32B pairwise 5-shot", "few"), ("Llama-3.3-70B pairwise 5-shot", "few")],
    "scalar_constructs/ad_negativity/pairwise": [
        ("Qwen-2.5-72B pairwise 0-shot", "0"), ("Llama-3.3-70B pairwise 0-shot", "0"), ("Qwen-2.5-32B pairwise 0-shot", "0"),
        ("Qwen-2.5-7B pairwise 0-shot", "0"), ("Llama-3.1-8B pairwise 0-shot", "0"), ("Qwen-2.5-72B pairwise 5-shot", "few")],
    "scalar_constructs/grandstanding/pairwise": [
        ("Qwen-2.5-72B pairwise 0-shot", "0"), ("Llama-3.3-70B pairwise 0-shot", "0"), ("Qwen-2.5-32B pairwise 0-shot", "0"),
        ("Qwen-2.5-7B pairwise 0-shot", "0"), ("Llama-3.1-8B pairwise 0-shot", "0"), ("Qwen-2.5-32B pairwise 5-shot", "few")],
    **{f"scalar_constructs/{c}/pointwise": [
        ("Qwen-2.5-72B pointwise 0-shot", "0"), ("Llama-3.3-70B pointwise 0-shot", "0"), ("Qwen-2.5-32B pointwise 0-shot", "0"),
        ("Qwen-2.5-7B pointwise 0-shot", "0"), ("Llama-3.1-8B pointwise 0-shot", "0"), ("Qwen-2.5-72B pointwise 5-shot", "few")]
       + ([("Llama-3.3-70B pointwise 5-shot", "few")] if c == "grandstanding" else [])
       for c in ("immigration_fear", "ad_negativity", "grandstanding")},
    "lifetox": [("GPT-3.5 text-davinci-003 zero-shot", "0"), ("Llama-2-Chat 13B", "0"), ("Llama-2-Chat 7B", "0")],
    # Ruan et al. 2024 Table 2(a): GPT-4 zero-shot and ICL+CoT; Llama2-70B (Re3, Ruan et al. 2024a) is ICL+CoT
    # prompting (Re3 Sec. 6.1 "ICL demonstration selection ... CoT"), not fine-tuning.
    "edit_intent": [("GPT-4-turbo zero-shot", "0"), ("GPT-4 ICL+CoT", "few"), ("Llama2-70B instruction-tuned (Re3)", "few")],
    "ezstance/subtaskA": [("ChatGPT gpt-3.5-turbo-0301", "0"), ("LLaMA-2-13B-chat", "0")],
    "ezstance/subtaskB": [("ChatGPT gpt-3.5-turbo-0301", "0")],
    **{f"contempo/{s}": [("ChatGPT-4 (CoT)", "0"), ("ChatGPT-4", "0"), ("ChatGPT-3.5 (CoT)", "0"), ("ChatGPT-3.5", "0")]
       for s in ("matres", "tbdense")},
    "mismatched": [("Gemini-1.5-Pro zero-shot", "0"), ("Phi-3 zero-shot", "0"), ("GPT-4o zero-shot", "0"),
                   ("Gemini-1.5-Pro few-shot MSciNLI+", "few"), ("GPT-4o few-shot MSciNLI+", "few"),
                   ("Phi-3 few-shot MSciNLI", "few")],
    "inli": [(k, "few") for k in ("Gemini-Pro 8-shot", "GPT-4 8-shot", "GPT-4o 8-shot", "Mistral-Large 8-shot",
                                   "Gemini-Flash 8-shot", "Claude-3-Sonnet 8-shot")],
    "medreadme": [("GPT-4 (gpt-4-0613) 5-shot", "few"), ("Llama-3.1-8B-Instruct 5-shot", "few")],
    "nav_unknown/clinc150": [("GPT-4o zero-shot tool calling", "0"), ("Gemini-1.5-Flash zero-shot", "0"),
                             ("Llama-3.1-70B zero-shot", "0"), ("GPT-4o few-shot 45 kNN", "few"), ("GPT-4o few-shot 10", "few")],
    "nav_unknown/banking77": [("GPT-4o zero-shot tool calling", "0"), ("GPT-4o few-shot 10 kNN", "few"),
                              ("GPT-4o few-shot 10", "few")],
    # MADE's prompted baselines all see kNN-retrieved training reports (app_tasks.tex MADE, "Deviations").
    "made": [(k, "few") for k in ("GPT-5 (thinking)", "Qwen3-235B-A22B-Thinking", "Qwen3-235B-A22B-Instruct", "GPT-4.1",
                                   "Llama-3.1-70B-Instruct", "Qwen3-4B-Instruct", "Llama-3.1-8B-Instruct")],
    "smarter/hatexplain": [("GPT-5-chat zero-shot", "0"), ("GPT-4.1 zero-shot", "0"), ("Qwen-32B zero-shot", "0"),
                           ("GPT-4o-mini zero-shot", "0"), ("GPT-5-chat 16-shot ICL mean", "few")],
    "smarter/latent_hate": [("GPT-4.1 zero-shot", "0"), ("GPT-4o-mini zero-shot", "0"), ("GPT-5-chat zero-shot", "0"),
                            ("Qwen-32B zero-shot", "0"), ("GPT-4.1 16-shot ICL mean", "few")],
    "smarter/implicit_hate": [("GPT-5-chat zero-shot", "0"), ("GPT-4.1 zero-shot", "0"), ("Qwen-32B zero-shot", "0"),
                              ("GPT-4o-mini zero-shot", "0"), ("Qwen-32B 16-shot ICL mean", "few")],
    "dress_new": [("gpt-4o (A) zero-shot ICL", "0"), ("gpt-4o (D) feedback generation", "0"),
                  ("gpt-4o (C) rubric explanation", "0"), ("gpt-4o (B) five-shot ICL", "few")],
    "legalseg": [("LLaMA-2 (unquantized)", "0")],
}
EXCLUDED_LLM_ROWS = {"contempo/*": "Fine-tuned GPT-3.5", "legalseg": "RhetoricLLaMA (instruction fine-tuned)",
                     "mismatched": "Llama-2-13B LoRA on SciNLI"}


# ------------------------------------------------------------------ statistics helpers
def phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


def p_diff(d: float, se: float) -> float:
    if se == 0:
        return 0.0 if d != 0 else 1.0
    return 2 * phi(-abs(d) / se)


def p_tost(d: float, se: float, delta: float) -> float:
    if se == 0:
        return 0.0 if abs(d) < delta else 1.0
    return max(1 - phi((d + delta) / se), phi((d - delta) / se))


def holm(p: list[float]) -> list[float]:
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    out, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        out[i] = run
    return out


def classify(rows: list[dict], ref: str, se_key: str, delta: float, adjust: bool) -> list[str]:
    pd_ = [p_diff(r["jev"] - r[ref], r[se_key]) for r in rows]
    pt = [p_tost(r["jev"] - r[ref], r[se_key], delta) for r in rows]
    if adjust:
        pd_, pt = holm(pd_), holm(pt)
    out = []
    for r, a, b in zip(rows, pd_, pt):
        if a < ALPHA:
            out.append("above" if r["jev"] > r[ref] else "below")
        elif b < ALPHA:
            out.append("parity")
        else:
            out.append("inconclusive")
    return out


CLASSES = ("above", "parity", "below", "inconclusive")


def counts(cls: list[str]) -> dict[str, int]:
    c = Counter(cls)
    return {k: c.get(k, 0) for k in CLASSES}


def fmt_counts(c: dict[str, int]) -> str:
    return " / ".join(str(c[k]) for k in CLASSES)


# ------------------------------------------------------------------ bootstrap
def bootstrap(row: dict) -> dict:
    name = row["task"]
    task = REGISTRY[name]()
    items = task.items()
    preds = F.load_predictions(name)
    spec = F.build(name, task, items, preds)
    full = spec.stat(np.arange(spec.n_units))
    if abs(full - row["jev"]) > 1e-9:
        raise AssertionError(f"{name}: full-sample statistic {full} != saved {row['jev']}")
    rng = np.random.default_rng(SEED)
    item_vals = np.array([spec.stat(F.draw(rng, spec, sort=spec.ordered)) for _ in range(B)])
    out = {"unit": spec.unit, "n_units": spec.n_units, "method": spec.method,
           "se_unit": float(np.nanstd(item_vals, ddof=1)),
           "ci95_unit_pct": [float(x) for x in np.nanpercentile(item_vals, [2.5, 97.5])],
           "nan_resamples": int(np.isnan(item_vals).sum())}
    ck = cluster_keys(name, items)
    if ck is None:
        out.update(cluster=None, n_clusters=spec.n_units, se_cluster=out["se_unit"], ci95_cluster_pct=out["ci95_unit_pct"])
        return out
    unit, keys = ck
    if len(keys) != spec.n_units:
        raise AssertionError(f"{name}: {len(keys)} cluster keys for {spec.n_units} units")
    if spec.ordered:
        raise AssertionError(f"{name}: clustered draws assume an order-free statistic")
    cd = ClusterDraw(keys, spec.strata)
    rng = np.random.default_rng(SEED)
    vals = np.array([spec.stat(cd(rng)) for _ in range(B)])
    out.update(cluster=unit, n_clusters=cd.n_clusters, max_cluster=int(cd.sizes.max()),
               se_cluster=float(np.nanstd(vals, ddof=1)),
               ci95_cluster_pct=[float(x) for x in np.nanpercentile(vals, [2.5, 97.5])],
               nan_resamples_cluster=int(np.isnan(vals).sum()))
    return out


# ------------------------------------------------------------------ (c) LLM rules
def llm_values(rows: list[dict]) -> dict[str, dict]:
    """Per set: applied value (paper_tables), rule-derived zero-shot-else-few-shot value, best prompted value."""
    out = {}
    for r in rows:
        t = r["task"]
        cands = PROMPTED.get(t, [])
        if r["llm"] is None:
            if cands:
                raise AssertionError(f"{t}: prompted rows encoded but paper_tables has no LLM value")
            continue
        pn = paper_tables.PAPER[t]["paper"]
        vals = [(k, s, float(pn[k])) for k, s in cands]
        zs = [v for v in vals if v[1] == "0"]
        rule = max(zs or vals, key=lambda v: v[2])
        best = max(vals, key=lambda v: v[2])
        applied = [v for v in vals if abs(v[2] - r["llm"]) <= 5e-4 + 1e-12]
        if not applied:
            raise AssertionError(f"{t}: applied LLM {r['llm']} not among encoded prompted rows")
        out[t] = {"applied": r["llm"], "applied_row": applied[0][0], "applied_shots": applied[0][1],
                  "rule_row": rule[0], "rule_value": rule[2], "rule_matches_applied": abs(rule[2] - r["llm"]) <= 5e-4 + 1e-12,
                  "best_row": best[0], "best_value": best[2], "best_shots": best[1],
                  "source": paper_tables.PAPER[t]["source"]}
    return out


def group(jev: float, method: float, llm: float | None) -> str:
    """paper_tables.rows() grouping."""
    if jev >= method - TOL:
        return "replaces"
    if llm is not None and jev >= llm - TOL:
        return "between"
    if llm is None and jev / method >= 0.8:
        return "between"
    return "below"


GROUPS = ("replaces", "between", "below")


# ------------------------------------------------------------------ output helpers
def pt(x: float) -> str:
    return f"{100 * x:.1f}"


def sgn(x: float) -> str:
    return f"$+${x:.1f}" if x >= 0 else f"$-${-x:.1f}"


CLS_TEX = {"above": "above", "below": "below", "parity": "parity", "inconclusive": "inconcl."}

PAPER_LABEL = {
    "zhang-etal-2024-transferable": "PiNose", "luo-etal-2024-prp": "PRP-Graph", "rottger-etal-2024-xstest": "XSTest",
    "joseph-etal-2024-factpico": "FactPICO", "scire-etal-2024-fenice": "FENICE", "lan-etal-2024-multi": "BIORC800",
    "you-etal-2024-sciprompt": "SciPrompt", "licht-etal-2025-measuring": "Scalar Constructs",
    "kim-etal-2024-lifetox": "LifeTox", "ruan-etal-2024-large": "Edit intent", "zhao-caragea-2024-ez": "EZ-STANCE",
    "niu-etal-2024-contempo": "ConTempo", "shaik-etal-2025-mismatched": "MisMatched",
    "havaldar-etal-2025-entailed": "INLI", "jiang-xu-2024-medreadme": "MedReadMe",
    "sali-toraman-2025-navigating": "Navigating the Unknown", "agarwal-etal-2026-made": "MADE",
    "nghiem-etal-2026-smarter": "SMARTER", "yoo-etal-2025-dress": "DREsS", "nguyen-etal-2024-noise": "TTP",
    "nigam-etal-2025-legalseg": "LegalSeg", "mtumbuka-schockaert-2024-entity": "RelEmb",
    "hengst-etal-2024-conformal": "CICC",
}


def main() -> None:
    global B
    for a in sys.argv[1:]:
        if a.startswith("--B="):
            B = int(a.split("=", 1)[1])
    dry = B != 10_000
    t0 = time.time()
    rows = [r for r in paper_tables.rows() if r["paper"] is not None]
    if len(rows) != 46:
        raise AssertionError(f"expected 46 comparable sets, got {len(rows)}")
    if set(COMPARATOR) != {r["task"] for r in rows}:
        raise AssertionError("COMPARATOR does not cover exactly the 46 sets")
    if not set(WIDEN) <= {r["task"] for r in rows}:
        raise AssertionError("WIDEN names an unknown set")

    sets = []
    for r in rows:
        t1 = time.time()
        bs = bootstrap(r)
        widen = r["task"] in WIDEN
        rec = {"task": r["task"], "name": r["name"], "paper_key": r["key"], "paper_label": PAPER_LABEL[r["key"]],
               "metric": paper_tables.PAPER[r["task"]]["metric"], "jev": r["jev"], "ref": r["paper"], "llm": r["llm"],
               "group_paper_tables": r["cat"], "comparator": COMPARATOR[r["task"]][0],
               "comparator_note": COMPARATOR[r["task"]][1], "widen": widen,
               "widen_note": WIDEN.get(r["task"], SAME_NOTE), **bs}
        f = math.sqrt(2) if widen else 1.0
        rec["se_eff"] = rec["se_cluster"] * f  # primary: clustered + rule-based widening
        rec["se_unit_nowiden"] = rec["se_unit"]
        rec["se_cluster_nowiden"] = rec["se_cluster"]
        rec["se_unit_widen"] = rec["se_unit"] * f
        rec["se_cluster_widen_all"] = rec["se_cluster"] * math.sqrt(2)
        d = rec["jev"] - rec["ref"]
        rec["delta"] = d
        rec["ci95_delta"] = [d - 1.959964 * rec["se_eff"], d + 1.959964 * rec["se_eff"]]
        rec["p_diff"] = p_diff(d, rec["se_eff"])
        rec["p_tost"] = p_tost(d, rec["se_eff"], DELTA)
        sets.append(rec)
        print(f"{r['task']:45s} {bs['unit']:8s} n={bs['n_units']:6d} clusters={bs['n_clusters']:6d} "
              f"se_u={bs['se_unit']:.4f} se_c={bs['se_cluster']:.4f} widen={int(widen)} d={d:+.4f} "
              f"{time.time() - t1:6.1f}s", flush=True)

    # ---- (b) classes: primary and sensitivity
    prim = classify(sets, "ref", "se_eff", DELTA, True)
    p_holm = holm([s["p_diff"] for s in sets])
    pt_holm = holm([s["p_tost"] for s in sets])
    for s, c, a, b in zip(sets, prim, p_holm, pt_holm):
        s["class"], s["p_diff_holm"], s["p_tost_holm"] = c, a, b
    variants = {
        "normal, unit bootstrap, no widening, no Holm": classify(sets, "ref", "se_unit_nowiden", DELTA, False),
        "+ cluster bootstrap": classify(sets, "ref", "se_cluster_nowiden", DELTA, False),
        "+ widening (rule)": classify(sets, "ref", "se_eff", DELTA, False),
        "+ Holm (primary)": prim,
        "primary, no clustering": classify(sets, "ref", "se_unit_widen", DELTA, True),
        "primary, widening on all 46": classify(sets, "ref", "se_cluster_widen_all", DELTA, True),
        **{f"primary, delta = {100 * dl:g} pt": classify(sets, "ref", "se_eff", dl, True) for dl in DELTAS if dl != DELTA},
        **{f"no Holm, delta = {100 * dl:g} pt": classify(sets, "ref", "se_eff", dl, False) for dl in DELTAS if dl != DELTA},
    }
    audit = json.loads((ROOT / "results" / "framing_audit_ci.json").read_text())
    audit_counts = {"above": audit["counts"]["vs_paper"]["above"], "parity": audit["counts"]["vs_paper"]["parity"],
                    "below": audit["counts"]["vs_paper"]["below"], "inconclusive": audit["counts"]["vs_paper"]["inconclusive"]}
    var_counts = {"current audit (percentile, unit, fixed reference, 5-pt margin, no Holm)": audit_counts,
                  **{k: counts(v) for k, v in variants.items()}}

    by_type = {t: counts([s["class"] for s in sets if s["comparator"] == t]) for t in TYPES}
    n_type = {t: sum(s["comparator"] == t for s in sets) for t in TYPES}
    groups_by_type = {t: {g: sum(s["comparator"] == t and s["group_paper_tables"] == g for s in sets) for g in GROUPS}
                      for t in TYPES}
    papers: dict[str, dict] = {}
    for s in sets:
        p = papers.setdefault(s["paper_key"], {"label": s["paper_label"], "sets": 0, "types": set(), **{k: 0 for k in CLASSES}})
        p["sets"] += 1
        p[s["class"]] += 1
        p["types"].add(s["comparator"])
    for p in papers.values():
        p["types"] = sorted(p["types"])
    papers_any_above = sum(p["above"] > 0 for p in papers.values())
    papers_all_below = sum(p["below"] == p["sets"] for p in papers.values())
    papers_any_below = sum(p["below"] > 0 for p in papers.values())

    # ---- (c) LLM baseline rules
    llm = llm_values(rows)
    exceptions = sorted(t for t, v in llm.items() if not v["rule_matches_applied"])
    if exceptions != ["made"]:
        raise AssertionError(f"applied-rule exceptions changed: {exceptions}")
    rule_desc = {"zs": "applied in scripts/paper_tables.py (zero-shot where reported, else few-shot; MADE GPT-4.1)",
                 "best": "best reported prompted LLM, few-shot included"}
    by_set = {s["task"]: s for s in sets}
    llm_out: dict[str, dict] = {}
    for rule, key in (("zs", "applied"), ("best", "best_value")):
        g, ge_strict, ge_tol = [], 0, 0
        for s in sets:
            v = llm[s["task"]][key] if s["task"] in llm else None
            g.append(group(s["jev"], s["ref"], v))
            if v is not None:
                ge_strict += s["jev"] >= v
                ge_tol += s["jev"] >= v - TOL
        cnt = {k: g.count(k) for k in GROUPS}
        with_llm = [dict(by_set[t], llm_rule=llm[t][key]) for t in by_set if t in llm]
        icls = classify(with_llm, "llm_rule", "se_eff", DELTA, True)
        for s, gg in zip(sets, g):
            s[f"group_{rule}"] = gg
        for s, c in zip(with_llm, icls):
            by_set[s["task"]][f"class_vs_llm_{rule}"] = c
        llm_out[rule] = {"description": rule_desc[rule], "groups": cnt, "jev_ge_llm_strict": int(ge_strict),
                         "jev_ge_llm_tol": int(ge_tol), "n_with_llm": len(with_llm), "interval_vs_llm": counts(icls)}
    if llm_out["zs"]["groups"] != {k: sum(s["group_paper_tables"] == k for s in sets) for k in GROUPS}:
        raise AssertionError("applied-rule groups do not reproduce paper_tables.rows()")
    moved = [s["task"] for s in sets if s["group_zs"] != s["group_best"]]
    changed_llm = [t for t, v in llm.items() if abs(v["best_value"] - v["applied"]) > 5e-4 + 1e-12]
    # Sensitivity: best rule without the Re3 Llama2-70B row (reported from Ruan et al. 2024a, not rerun by the paper)
    alt_edit = max(float(paper_tables.PAPER["edit_intent"]["paper"][k]) for k, _ in PROMPTED["edit_intent"]
                   if k != "Llama2-70B instruction-tuned (Re3)")
    alt_groups = Counter(group(s["jev"], s["ref"], alt_edit if s["task"] == "edit_intent" else
                               (llm[s["task"]]["best_value"] if s["task"] in llm else None)) for s in sets)
    llm_out["best_without_re3_row"] = {k: alt_groups.get(k, 0) for k in GROUPS}

    # ---- macros
    pc = counts(prim)
    unadj = counts(variants["+ widening (rule)"])
    macros = {
        "ovRqOneNSets": str(len(sets)), "ovRqOneNPapers": str(len(papers)), "ovRqOneB": f"{B:,}",
        "ovRqOneTolPts": f"{100 * DELTA:g}",
        "ovRqOneSepAbove": str(pc["above"]), "ovRqOneParity": str(pc["parity"]), "ovRqOneSepBelow": str(pc["below"]),
        "ovRqOneInconcl": str(pc["inconclusive"]),
        "ovRqOneUnadjAbove": str(unadj["above"]), "ovRqOneUnadjParity": str(unadj["parity"]),
        "ovRqOneUnadjBelow": str(unadj["below"]), "ovRqOneUnadjInconcl": str(unadj["inconclusive"]),
        "ovRqOneNClustered": str(sum(s["cluster"] is not None for s in sets)),
        "ovRqOneNWidened": str(sum(s["widen"] for s in sets)),
        "ovRqOneCountsAudit": fmt_counts(audit_counts),
        "ovRqOneCountsNormal": fmt_counts(counts(variants["normal, unit bootstrap, no widening, no Holm"])),
        "ovRqOneCountsCluster": fmt_counts(counts(variants["+ cluster bootstrap"])),
        "ovRqOneCountsWiden": fmt_counts(unadj),
        "ovRqOneCountsHolm": fmt_counts(pc),
        "ovRqOneCountsWidenAll": fmt_counts(counts(variants["primary, widening on all 46"])),
        "ovRqOneCountsTolHalf": fmt_counts(counts(variants["primary, delta = 0.5 pt"])),
        "ovRqOneCountsTolTwo": fmt_counts(counts(variants["primary, delta = 2 pt"])),
        "ovRqOneCountsTolFive": fmt_counts(counts(variants["primary, delta = 5 pt"])),
        "ovRqOnePapersAnyAbove": str(papers_any_above), "ovRqOnePapersAllBelow": str(papers_all_below),
        "ovRqOnePapersAnyBelow": str(papers_any_below),
        "ovRqOneLlmRuleMovedSets": str(len(moved)), "ovRqOneLlmChangedSets": str(len(changed_llm)),
        "ovRqOneNLlm": str(llm_out["zs"]["n_with_llm"]),
    }
    tname = {"trained": "Trained", "untrained": "Untrained", "no_labels": "NoLabels"}
    for t in TYPES:
        macros[f"ovRqOneN{tname[t]}"] = str(n_type[t])
        for k, mk in (("above", "Above"), ("parity", "Parity"), ("below", "Below"), ("inconclusive", "Inconcl")):
            macros[f"ovRqOne{mk}{tname[t]}"] = str(by_type[t][k])
        for g, mk in (("replaces", "Match"), ("between", "Between"), ("below", "PtBelow")):
            macros[f"ovRqOne{mk}{tname[t]}"] = str(groups_by_type[t][g])
    for rule, mk in (("zs", "Zs"), ("best", "Best")):
        o = llm_out[rule]
        macros[f"ovRqOneLlm{mk}Replaces"] = str(o["groups"]["replaces"])
        macros[f"ovRqOneLlm{mk}Between"] = str(o["groups"]["between"])
        macros[f"ovRqOneLlm{mk}Below"] = str(o["groups"]["below"])
        macros[f"ovRqOneJevGeLlm{mk}Strict"] = str(o["jev_ge_llm_strict"])
        macros[f"ovRqOneJevGeLlm{mk}Tol"] = str(o["jev_ge_llm_tol"])
        macros[f"ovRqOneLlm{mk}Sep"] = fmt_counts(o["interval_vs_llm"])
    nore = llm_out["best_without_re3_row"]
    macros["ovRqOneLlmBestNoReReplaces"] = str(nore["replaces"])
    macros["ovRqOneLlmBestNoReBetween"] = str(nore["between"])
    macros["ovRqOneLlmBestNoReBelow"] = str(nore["below"])
    if not all(k.startswith("ov") and k.isalpha() for k in macros):
        raise AssertionError("bad macro name")

    # ---- tex tables (written at the end unless dry run)
    texs: dict[str, str] = {}
    tex_sets = ["% Generated by scripts/offline_rq1.py. Do not edit.", "\\begin{tabular}{lllrrrrrrl}", "\\toprule",
                "Set & Comparator & Unit (clusters) & $\\sqrt2$ & Jev & Ref. & $\\Delta$ & 95\\% CI of $\\Delta$ & "
                "Holm $p$ & Class \\\\", "\\midrule"]
    for s in sorted(sets, key=lambda s: (CLASSES.index(s["class"]), -s["delta"])):
        unit = s["cluster"] or s["unit"]
        ncl = f"{s['n_clusters']:,}"
        lo, hi = (100 * x for x in s["ci95_delta"])
        p = s["p_diff_holm"]
        ptxt = "$<$.001" if p < 0.001 else f"{p:.3f}"
        tex_sets.append(f"{s['name']} & {TYPE_TEX[s['comparator']]} & {unit} ({ncl}) & {'yes' if s['widen'] else ''} & "
                        f"{pt(s['jev'])} & {pt(s['ref'])} & {sgn(100 * s['delta'])} & [{sgn(lo)}, {sgn(hi)}] & {ptxt} & "
                        f"{CLS_TEX[s['class']]} \\\\")
    tex_sets += ["\\bottomrule", "\\end{tabular}", ""]
    texs["offline_rq1_sets.tex"] = "\n".join(tex_sets)

    tex_p = ["% Generated by scripts/offline_rq1.py. Do not edit.", "\\begin{tabular}{llrrrrr}", "\\toprule",
             "Paper & Comparator & Sets & Above & Parity & Below & Inconcl. \\\\", "\\midrule"]
    for key, p in sorted(papers.items(), key=lambda kv: (-kv[1]["above"], kv[1]["label"])):
        tex_p.append(f"{p['label']}~\\citep{{{key}}} & {', '.join(TYPE_TEX[t] for t in p['types'])} & {p['sets']} & "
                     f"{p['above']} & {p['parity']} & {p['below']} & {p['inconclusive']} \\\\")
    tex_p += ["\\midrule"]
    for t in TYPES:
        c = by_type[t]
        tex_p.append(f"\\textit{{All {TYPE_TEX[t]}}} & & {n_type[t]} & {c['above']} & {c['parity']} & {c['below']} & "
                     f"{c['inconclusive']} \\\\")
    tex_p.append(f"\\textit{{Total}} & & {len(sets)} & {pc['above']} & {pc['parity']} & {pc['below']} & {pc['inconclusive']} \\\\")
    tex_p += ["\\bottomrule", "\\end{tabular}", ""]
    texs["offline_rq1_papers.tex"] = "\n".join(tex_p)

    tex_s = ["% Generated by scripts/offline_rq1.py. Do not edit.", "\\begin{tabular}{lrrrr}", "\\toprule",
             "Analysis choice & Above & Parity & Below & Inconcl. \\\\", "\\midrule"]
    for k, c in var_counts.items():
        lab = k.replace("delta", "$\\delta$").replace("+ ", "+\\,")
        tex_s.append(f"{lab} & {c['above']} & {c['parity']} & {c['below']} & {c['inconclusive']} \\\\")
    tex_s += ["\\bottomrule", "\\end{tabular}", ""]
    texs["offline_rq1_sensitivity.tex"] = "\n".join(tex_s)

    gt = {"replaces": "match", "between": "betw.", "below": "below"}
    tex_l = ["% Generated by scripts/offline_rq1.py. Do not edit.", "\\begin{tabular}{lrrlrlll}", "\\toprule",
             "Set & Jev & Applied LLM & Group & Best prompted LLM & Row & Group & vs best (CI) \\\\", "\\midrule"]
    for s in sets:
        if s["task"] not in llm:
            continue
        v = llm[s["task"]]
        shots = "" if v["applied_shots"] == "0" else " (few)"
        bshots = "0-shot" if v["best_shots"] == "0" else "few-shot"
        row = v["best_row"].replace("_", "\\_")
        mark = "$\\rightarrow$ " if s["group_zs"] != s["group_best"] else ""
        tex_l.append(f"{s['name']} & {pt(s['jev'])} & {pt(v['applied'])}{shots} & {gt[s['group_zs']]} & {pt(v['best_value'])} & "
                     f"{row} ({bshots}) & {mark}{gt[s['group_best']]} & {CLS_TEX[s['class_vs_llm_best']]} \\\\")
    tex_l += ["\\midrule",
              f"\\multicolumn{{8}}{{l}}{{All 46 sets, applied rule: {llm_out['zs']['groups']['replaces']} / "
              f"{llm_out['zs']['groups']['between']} / {llm_out['zs']['groups']['below']}; best prompted: "
              f"{llm_out['best']['groups']['replaces']} / {llm_out['best']['groups']['between']} / "
              f"{llm_out['best']['groups']['below']} (match / between / below)}} \\\\",
              "\\bottomrule", "\\end{tabular}", ""]
    texs["offline_rq1_llm.tex"] = "\n".join(tex_l)

    # ---- markdown
    md = [f"## RQ1 statistics (rq1)", "",
          f"Source: `scripts/offline_rq1.py` over the saved zero-shot predictions of the {len(sets)} comparable sets "
          f"({len(papers)} papers), metric code and resampling specs of `scripts/framing_audit_ci.py`, published values "
          f"from `results/paper_numbers.json` via `scripts/paper_tables.py`. B = {B:,}, seed {SEED}.", "",
          "**Rules.** Clustered bootstrap where units share a source unit (prompt, abstract, narrative, document, premise, "
          "tweet, query, paper, writing prompt); SE = bootstrap SD. Two-sample widening: where the published value is on a "
          "different item set (reconstructed split, >1% of items missing, different item subset/protocol) the published "
          "value is given the same sampling variance as Jev's, SE x sqrt(2). Normal-approximation difference test and TOST "
          f"at delta = {100 * DELTA:g} point; Holm within the 46 difference tests and within the 46 equivalence tests. "
          "Separated if Holm p < .05, else parity if Holm TOST p < .05, else inconclusive.", "",
          f"**Primary (vs task-specific method): above {pc['above']}, parity {pc['parity']}, below {pc['below']}, "
          f"inconclusive {pc['inconclusive']}.** Clustered: {macros['ovRqOneNClustered']} sets; widened: "
          f"{macros['ovRqOneNWidened']} sets.", "",
          "| Analysis choice | above / parity / below / inconcl. |", "|---|---|",
          *[f"| {k} | {fmt_counts(c)} |" for k, c in var_counts.items()], "",
          "**By comparator type** (primary classes; point-estimate groups of Table 11 in brackets: match / between / below):", "",
          "| Type | Sets | above / parity / below / inconcl. | match / between / below |", "|---|---|---|---|",
          *[f"| {t} | {n_type[t]} | {fmt_counts(by_type[t])} | {groups_by_type[t]['replaces']} / "
            f"{groups_by_type[t]['between']} / {groups_by_type[t]['below']} |" for t in TYPES], "",
          f"Per paper: {papers_any_above} of {len(papers)} papers have at least one set separated above; "
          f"{papers_all_below} have every set separated below; {papers_any_below} have at least one set separated below.", "",
          "| Paper | Comparator | Sets | above / parity / below / inconcl. |", "|---|---|---|---|",
          *[f"| {p['label']} | {', '.join(p['types'])} | {p['sets']} | {fmt_counts(p)} |"
            for _, p in sorted(papers.items(), key=lambda kv: (-kv[1]["above"], kv[1]["label"]))], "",
          "| Set | Comparator | Unit (clusters) | widened | Jev | Ref | Δ | 95% CI Δ | Holm p | class |",
          "|---|---|---|---|---|---|---|---|---|---|",
          *[f"| {s['name'].replace(chr(92), '')} | {s['comparator']} | {s['cluster'] or s['unit']} ({s['n_clusters']}) | "
            f"{'yes' if s['widen'] else ''} | {pt(s['jev'])} | {pt(s['ref'])} | {100 * s['delta']:+.1f} | "
            f"[{100 * s['ci95_delta'][0]:+.1f}, {100 * s['ci95_delta'][1]:+.1f}] | {s['p_diff_holm']:.3g} | {s['class']} |"
            for s in sets], "",
          "**LLM baseline (c).** Rule applied in `scripts/paper_tables.py`, verified against the encoded prompted rows: the "
          "best zero-shot prompted LLM where the paper reports one, otherwise the best few-shot one (INLI 8-shot, MedReadMe "
          "5-shot); the one exception is MADE, which uses GPT-4.1 (43.0) although all MADE prompted rows are kNN few-shot "
          "and the best is GPT-5 (thinking) 54.0. Fine-tuned LLM rows are never used.", "",
          "| Rule | match / between / below (46) | Jev >= LLM strict | Jev >= LLM (0.5 pt tol.) | interval vs LLM, Holm (above / parity / below / inconcl.) |",
          "|---|---|---|---|---|",
          *[f"| {llm_out[r]['description']} | {llm_out[r]['groups']['replaces']} / {llm_out[r]['groups']['between']} / "
            f"{llm_out[r]['groups']['below']} | {llm_out[r]['jev_ge_llm_strict']} of {llm_out[r]['n_with_llm']} | "
            f"{llm_out[r]['jev_ge_llm_tol']} of {llm_out[r]['n_with_llm']} | {fmt_counts(llm_out[r]['interval_vs_llm'])} |"
            for r in ("zs", "best")], "",
          f"The best-prompted rule changes the LLM value on {len(changed_llm)} sets and moves {len(moved)} sets between groups: "
          + ", ".join(f"{by_set[t]['name'].replace(chr(92), '')} ({pt(by_set[t]['jev'])} vs {pt(llm[t]['best_value'])}, "
                      f"{llm[t]['best_row']})" for t in moved) + ". "
          f"Without the Edit-intent Llama2-70B ICL+CoT row (69.0, reported by Ruan et al. from their earlier Re3 paper) the "
          f"best rule gives {llm_out['best_without_re3_row']['replaces']} / {llm_out['best_without_re3_row']['between']} / "
          f"{llm_out['best_without_re3_row']['below']}.", ""]

    out = {"macros": macros, "md": "\n".join(md) + "\n",
           "settings": {"B": B, "seed": SEED, "alpha": ALPHA, "delta": DELTA, "deltas": list(DELTAS), "group_tol": TOL},
           "sets": sets, "variant_counts": var_counts, "by_type": by_type, "n_type": n_type, "groups_by_type": groups_by_type,
           "papers": papers, "llm": {"per_set": llm, "rules": llm_out, "moved": moved, "changed": changed_llm,
                                     "excluded_rows": EXCLUDED_LLM_ROWS}}
    print(json.dumps({k: macros[k] for k in macros}, indent=0))
    if dry:
        print(f"dry run (B={B}), nothing written; {time.time() - t0:.0f}s")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=str) + "\n")
    TEX.mkdir(parents=True, exist_ok=True)
    for fname, text in texs.items():
        (TEX / fname).write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)} and paper/generated/offline_rq1_*.tex in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
