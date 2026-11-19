"""Cause table for the zero-shot shortfall sets -> results/CAUSES.md, results/causes.json.

An LLM drafted the rows below (ASSIGN); an author checked every row (AUTHOR_VERDICTS). The author's verdict is
final: the output keeps the draft alongside (draft_primary, draft_secondary, draft_confidence, author_verdict), a
changed row gets confidence "author", and the file reports the agreement between draft and author on the primary
cause (raw and Cohen's kappa, bootstrap over rows, B = 10,000, numpy default_rng(0)).

Scope. The shortfall sets are the rows of scripts/paper_tables.rows() whose outcome is not "replaces" (Jev does not
match or beat the task-specific method within 0.5 points): outcomes "between" and "below". Gap to method = Jev minus
the task-specific number, in points (x100), as in the paper's tables.

Assignment rule (fixed before any row was assigned; the LLM drafted each row by reading the evidence below, without
calling a model on the data). Each shortfall set gets a primary cause, optionally a secondary cause, a one-line evidence note pointing
to a file and number, and a confidence (high / medium / low).

  (a) conv   -- unstated annotation conventions. Evidence types, any of:
                - an optimized program states a convention that recovers accuracy (results/FRAMING.md,
                  "Conventions stated by the winning programs", final verdict yes, with the task's test gain);
                - labels whose meaning depends on dataset guidelines beyond the paper's definitions (appendix per-paper
                  notes, the paper's own description of its annotation);
                - systematic confusion between labels the guideline distinguishes only by convention (confusion matrix
                  from the saved zero-shot predictions in runs/<task>/predictions.jsonl).
  (b) large  -- very large or opaque label space: 100 or more options (the body's large-K group), or codes whose names do not carry their
                meaning (label counts from the task's inventory).
  (c) coarse -- probabilities too coarse for the paper's downstream method (conformal sets, thresholds on rare
                classes): the decision is often right but the paper's metric needs probability mass Jev rounds to 0
                (results/CONFIDENCE.md, the saved probability vectors).
  (d) other  -- other or unclear; the note states what (e.g. a continuous gold scale the paper's model regresses on,
                a gap within the method's own run-to-run noise, no item-level evidence either way). "Unclear" is an
                allowed answer and is not forced into (a)-(c).

Secondary cause: given only when a second cause has its own evidence of the same kinds. Confidence: high = direct
evidence of the stated kind on this set (an optimized program's convention with a test gain, a label count above 100,
a metric that collapses for a probability reason); medium = evidence on a sibling set of the same paper, or a
confusion pattern consistent with the cause without a program confirming it; low = the cause is the most plausible
reading but the evidence does not single it out.

Per-paper counts: a paper counts toward a cause if at least one of its shortfall sets has that primary cause.

Changes after the first assignment (each marked in its row): CICC ATIS, primary coarse -> conv (secondary coarse,
confidence high -> medium), because Clef's unrounded probabilities give the gold intent no zeros yet reach only
Single 0.9 at the same top-1 accuracy (results/offline_v4/clef.json), so the binding cause is the top-1 accuracy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import paper_tables as pt  # noqa: E402

CAUSES = {"conv": "unstated annotation conventions", "large": "very large or opaque label space",
          "coarse": "probabilities too coarse for the paper's method", "other": "other or unclear"}
CONFIDENCE = ("high", "medium", "low")
STATUS = "drafted by an LLM, checked by an author"
B = 10_000
# Author changes that also change the secondary cause (the new primary was the draft's secondary).
SECONDARY_ON_CHANGE = {"cicc/acid": None}


def kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in CAUSES)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agreement(draft: list[str], final: list[str]) -> dict:
    import numpy as np
    rng = np.random.default_rng(0)
    n = len(draft)
    ks = []
    for idx in rng.integers(0, n, size=(B, n)):
        ks.append(kappa([draft[i] for i in idx], [final[i] for i in idx]))
    lo, hi = np.percentile(ks, [2.5, 97.5])
    return {"n": n, "agree": sum(x == y for x, y in zip(draft, final)), "kappa": kappa(draft, final),
            "kappa_ci": [float(lo), float(hi)], "B": B, "seed": 0}

FRAMING = "results/FRAMING.md"
SEEDS = "results/SEEDS.md"
HEADS = "results/HEADS.md"
APP = "paper/generated/app_tasks.tex"
SCALAR_ORACLE = "data/manifests/scalar_constructs.json"


def pred(task: str) -> str:
    return f"runs/{task}/predictions.jsonl"


def score(task: str) -> str:
    return f"runs/{task}/scores.json"


# task -> (primary, secondary or None, confidence, evidence, sources). Assigned by reading the cited evidence.
ASSIGN: dict[str, tuple[str, str | None, str, str, list[str]]] = {
    "sciprompt/s2orc": (
        "conv", None, "low",
        "Options are field names only; Engineering is predicted 733 times against 298 gold (Materials Science, "
        "Computer Science, Environmental Science -> Engineering 89/88/85; Biology -> Medicine 78): the corpus's "
        "field tags, not the names, set the boundaries.",
        [pred("sciprompt/s2orc"), APP]),
    "scalar_constructs/immigration_fear/pairwise": (
        "other", None, "low",
        "Unclear. Gold is a Bradley-Terry fit to crowd comparisons; Jev agrees with 75.3% of test-pair choices "
        "against 83.1% for the gold scores themselves; a perfect judge reaches rho 99.4, so no protocol cap.",
        [score("scalar_constructs/immigration_fear/pairwise"), SCALAR_ORACLE]),
    "scalar_constructs/ad_negativity/pairwise": (
        "other", None, "low",
        "Unclear. Jev agrees with 80.5% of test-pair choices against 85.3% for the gold scores; a perfect judge "
        "reaches rho 92.8; no item-level evidence names a cause.",
        [score("scalar_constructs/ad_negativity/pairwise"), SCALAR_ORACLE]),
    "scalar_constructs/grandstanding/pairwise": (
        "other", None, "high",
        "Protocol ceiling: the BT fit sees only fold-0 test pairs, so a perfect pairwise judge reaches rho 63.2, "
        "below the reward model's 78.0 (Jev 58.6).",
        [SCALAR_ORACLE, APP]),
    "scalar_constructs/immigration_fear/pointwise": (
        "other", None, "low",
        "Unclear. Continuous BT gold with no item-level correctness; the paper's own zero-shot LLM is also "
        "below the reward model (81.0 against 91.0).",
        ["results/CONFIDENCE.md", "results/paper_numbers.json"]),
    "scalar_constructs/ad_negativity/pointwise": (
        "other", None, "low",
        "Unclear. A 1.6-point gap (95% CI 87.5-90.4 against 91.0); the paper's zero-shot LLM equals the "
        "reward model; no item-level correctness to locate the gap.",
        [FRAMING, "results/CONFIDENCE.md"]),
    "scalar_constructs/grandstanding/pointwise": (
        "other", None, "low",
        "Unclear. Continuous BT gold with no item-level correctness; Jev is above the paper's zero-shot LLM "
        "(70.0 against 60.0).",
        [FRAMING, "results/CONFIDENCE.md"]),
    "lifetox": (
        "conv", "other", "low",
        "523 of 1,500 unsafe answers chosen safe (112 the other way) while ROC-AUC is 0.90: the unsafe boundary, "
        "not the ranking, misses. Secondary: 2,957 of 3,000 test queries also occur in train, which favours the "
        "fine-tuned model.",
        [pred("lifetox"), score("lifetox"), APP]),
    "edit_intent": (
        "conv", None, "high",
        "All 3 optimized programs state a convention (headings and fragments go to Other; Grammar stays Grammar "
        "when a reference number changes) and gain +6.5 to +8.5 on test; zero-shot Fact/Evidence -> Claim 231.",
        [FRAMING, SEEDS, pred("edit_intent")]),
    "ezstance/subtaskA": (
        "conv", None, "high",
        "All 3 programs state a convention (claim targets with a flipped detail are AGAINST; NONE unless an "
        "attitude phrase can be cited) and gain +15.1 to +17.0 on test; zero-shot NONE recall 0.38 "
        "(NONE -> FAVOR 1,091).",
        [FRAMING, SEEDS, pred("ezstance/subtaskA")]),
    "ezstance/subtaskB": (
        "conv", None, "medium",
        "Same pattern as subtask A, which the programs fix: NONE recall 0.34 (NONE -> FAVOR 6,264, -> AGAINST "
        "3,632). Not optimized itself.",
        [pred("ezstance/subtaskB"), FRAMING]),
    "contempo/matres": (
        "conv", None, "medium",
        "VAGUE and EQUAL get F1 0.24 and 0.22 (BEFORE -> VAGUE 63, BEFORE -> EQUAL 39); a logistic regression on "
        "the answers (20 parameters, fit on dev) gains +6.8 on test (72.3 -> 79.1).",
        [pred("contempo/matres"), HEADS]),
    "contempo/tbdense": (
        "conv", None, "high",
        "Programs of seeds 0 and 1 narrow SIMULTANEOUS to the same occurrence and gain +8.0 and +6.8 on test; "
        "zero-shot VAGUE recall 0.30 (311 predicted, 638 gold), SIMULTANEOUS predicted 150 times against 22 gold.",
        [FRAMING, SEEDS, pred("contempo/tbdense")]),
    "mismatched": (
        "conv", None, "medium",
        "The three relation labels come from a deleted linking phrase; Jev predicts neutral 874 times against 600 "
        "gold (reasoning/entailment/contrasting -> neutral 145/134/131).",
        [pred("mismatched"), APP]),
    "inli": (
        "conv", None, "medium",
        "Errors sit on the dataset's implied/explicit distinction: implied -> neutral 227, explicit -> implied 194, "
        "implied -> explicit 136. Programs gain +4.1 to +5.4 on test but state no convention beyond the paper's "
        "definitions (0 of 3).",
        [pred("inli"), FRAMING, SEEDS]),
    "medreadme": (
        "other", None, "low",
        "Unclear. Continuous rating (mean of two annotators) with no item-level correctness; Jev is level with "
        "the paper's 5-shot LLM (71.4 against 72.1).",
        [FRAMING, "results/CONFIDENCE.md"]),
    "nav_unknown/clinc150": (
        "conv", "other", "low",
        "599 of 756 errors put a held-out intent into a named in-scope one; intents are names only, so their "
        "scope is set by the dataset's utterances, which the method sees. Secondary: reconstructed partitions "
        "and an unstated ALL-F1 formula.",
        [pred("nav_unknown/clinc150"), "data/frozen/nav_unknown/splits.json", APP]),
    "nav_unknown/banking77": (
        "conv", "other", "low",
        "505 of 713 errors put a held-out intent into a named in-scope one (fallback recall 0.57). Secondary: "
        "reconstructed partitions and an unstated ALL-F1 formula.",
        [pred("nav_unknown/banking77"), score("nav_unknown/banking77"), APP]),
    "made": (
        "large", "coarse", "high",
        "1,154 IMDRF codes decided top-down. Secondary: at a fixed threshold Jev over-predicts codes, while the "
        "method tunes a threshold per (often rare) code.",
        [APP, score("made")]),
    "smarter/hatexplain": (
        "conv", "other", "medium",
        "Offensive -> Hate 213 of 400 (Hate predicted 686 times, 395 gold); a calibration with two per-class biases, "
        "fit on dev, gains "
        "+5.5 on test. Secondary: half the posts are 2/1 annotator splits (accuracy 64.2 unanimous, 50.6 split).",
        [pred("smarter/hatexplain"), HEADS, "results/confidence.json"]),
    "smarter/latent_hate": (
        "conv", None, "medium",
        "Implicit -> Explicit Hate 153 (Implicit recall 0.39), a distinction set by the corpus guidelines; a "
        "logistic regression on the answers, fit on dev, gains only +1.4.",
        [pred("smarter/latent_hate"), HEADS]),
    "smarter/implicit_hate": (
        "conv", None, "high",
        "2 of 3 programs route retweets, nationalist hashtags and leaders to Incitement and gain +18.5 and +16.6 "
        "on test; zero-shot Incitement F1 0.20.",
        [FRAMING, SEEDS, pred("smarter/implicit_hate")]),
    "dress_new": (
        "conv", "other", "high",
        "6 of 9 rubric programs state the grader's leniency or score distribution; per-rubric QWK gains +17.6 to "
        "+36.2 on test. Secondary: reconstructed split.",
        [FRAMING, SEEDS, APP]),
    "ttp/tram": (
        "large", None, "high",
        "737 ATT&CK technique profiles ranked per text; confidence near chance on the opaque sets (AUROC 0.54).",
        [APP, "results/CONFIDENCE.md"]),
    "ttp/procedures_plus": (
        "large", None, "high",
        "737 ATT&CK technique profiles ranked per text; confidence near chance on the opaque sets (AUROC 0.54).",
        [APP, "results/CONFIDENCE.md"]),
    "ttp/expert": (
        "large", "other", "high",
        "737 ATT&CK technique profiles ranked per text. Secondary: gold was made by reviewing the paper's own "
        "matcher's top 20.",
        [APP]),
    "legalseg": (
        "conv", "other", "medium",
        "None recall 0.16 (12,851 predicted, 58,499 gold; None -> Reasoning 32,166): which sentences are None is "
        "an annotation convention. Secondary: the 77.0 does not reproduce; the reproducible baseline is 62.0.",
        [pred("legalseg"), APP]),
    "relemb/wikiwd": (
        "large", None, "high",
        "352 relations given by name only, above the 255-option limit (asked in two halves).",
        [APP, score("relemb/wikiwd")]),
    "relemb/nytfb": (
        "conv", None, "medium",
        "Distant supervision (96% NA): Jev's argmax names a relation on 36,220 rows against 6,440 gold positives "
        "(precision 9.9, recall 55.8); gold follows the KB alignment, not what the sentence says.",
        [score("relemb/nytfb"), APP]),
    "cicc/mtod": (
        "other", None, "medium",
        "Top-1 accuracy 98.2 is below the 99% coverage target, so some sets need a second intent (qhat 0.83); "
        "gold P=0 on only 0.3% of calibration items, so rounding does not bind.",
        [score("cicc/mtod"), APP]),
    "cicc/hwu64": (
        "coarse", None, "medium",
        "Gold P=0 on 4.5% of calibration items at alpha 5%, so qhat is 0.99 and every intent with P >= 0.01 "
        "enters the set: Single 47.9 at top-1 81.5.",
        [score("cicc/hwu64"), APP]),
    "cicc/c150": (
        "coarse", None, "high",
        "Gold P=0 on 1.8% of calibration items at alpha 1%: qhat = 1, every set is all 150 intents, Single 0 at "
        "top-1 92.6.",
        [score("cicc/c150"), APP]),
    "cicc/acid": (
        "coarse", "large", "high",
        "Gold P=0 on 2.4% of calibration items at alpha 2%: qhat = 1, Single 0 at top-1 85.5. Secondary: 175 "
        "abbreviated codes (INFO_CL_FNOL -> INFO_CL_FILE_CLAIM 38).",
        [score("cicc/acid"), pred("cicc/acid"), APP]),
    "cicc/atis": (
        "conv", "coarse", "medium",
        "Top-1 66.4: queries labelled flight go to flight+airline (118) and flight_time (92), labels the dataset "
        "separates by its own labelling practice. Secondary: gold P=0 on 3.5% of calibration items at alpha 1%: "
        "qhat = 1, Single 0. Clef, with four-decimal probabilities and no gold zeros, has the same top-1 (66.3) and "
        "Single 0.9, so finer probabilities would not close the gap.",
        [score("cicc/atis"), pred("cicc/atis"), "results/offline_v4/clef.json", APP]),
}


# The author's check of every drafted row, final: task -> (check item, draft primary, draft secondary, verdict,
# new primary cause when the verdict is "change").
AUTHOR_VERDICTS: dict[str, tuple[str, str, str | None, str, str | None]] = {
    "sciprompt/s2orc": ("A10", "conv", None, "change", "other"),
    "scalar_constructs/immigration_fear/pairwise": ("A8", "other", None, "ok", None),
    "scalar_constructs/ad_negativity/pairwise": ("A5", "other", None, "ok", None),
    "scalar_constructs/grandstanding/pairwise": ("A29", "other", None, "ok", None),
    "scalar_constructs/immigration_fear/pointwise": ("A9", "other", None, "ok", None),
    "scalar_constructs/ad_negativity/pointwise": ("A6", "other", None, "ok", None),
    "scalar_constructs/grandstanding/pointwise": ("A7", "other", None, "ok", None),
    "lifetox": ("A3", "conv", "other", "ok", None),
    "edit_intent": ("A26", "conv", None, "ok", None),
    "ezstance/subtaskA": ("A25", "conv", None, "ok", None),
    "ezstance/subtaskB": ("A14", "conv", None, "ok", None),
    "contempo/matres": ("A17", "conv", None, "ok", None),
    "contempo/tbdense": ("A30", "conv", None, "ok", None),
    "mismatched": ("A18", "conv", None, "change", "other"),
    "inli": ("A15", "conv", None, "ok", None),
    "medreadme": ("A4", "other", None, "ok", None),
    "nav_unknown/clinc150": ("A2", "conv", "other", "ok", None),
    "nav_unknown/banking77": ("A1", "conv", "other", "ok", None),
    "made": ("A27", "large", "coarse", "ok", None),
    "smarter/hatexplain": ("A20", "conv", "other", "ok", None),
    "smarter/latent_hate": ("A21", "conv", None, "ok", None),
    "smarter/implicit_hate": ("A28", "conv", None, "ok", None),
    "dress_new": ("A24", "conv", "other", "ok", None),
    "ttp/tram": ("A33", "large", None, "ok", None),
    "ttp/procedures_plus": ("A32", "large", None, "ok", None),
    "ttp/expert": ("A31", "large", "other", "change", "conv"),
    "legalseg": ("A16", "conv", "other", "ok", None),
    "relemb/wikiwd": ("A34", "large", None, "change", "conv"),
    "relemb/nytfb": ("A19", "conv", None, "change", "other"),
    "cicc/mtod": ("A13", "other", None, "ok", None),
    "cicc/hwu64": ("A12", "coarse", None, "change", "conv"),
    "cicc/c150": ("A23", "coarse", None, "ok", None),
    "cicc/acid": ("A22", "coarse", "large", "change", "large"),
    "cicc/atis": ("A11", "conv", "coarse", "ok", None),
}


def build() -> dict:
    rows = [r for r in pt.rows() if r["cat"] in ("between", "below")]
    tasks = [r["task"] for r in rows]
    if set(tasks) != set(ASSIGN):
        raise SystemExit(f"assignments do not match the shortfall sets: missing {sorted(set(tasks) - set(ASSIGN))}, "
                         f"extra {sorted(set(ASSIGN) - set(tasks))}")
    if set(tasks) != set(AUTHOR_VERDICTS):
        raise SystemExit("the author verdicts do not cover exactly the shortfall sets")
    sets = []
    for r in rows:
        primary, secondary, conf, evidence, sources = ASSIGN[r["task"]]
        assert primary in CAUSES and (secondary is None or secondary in CAUSES) and secondary != primary
        assert conf in CONFIDENCE
        item, draft_primary, draft_secondary, verdict, value = AUTHOR_VERDICTS[r["task"]]
        v = {"item": item, "verdict": verdict, "value": value}
        if (draft_primary, draft_secondary) != (primary, secondary):
            raise SystemExit(f"{r['task']}: the checked draft differs from ASSIGN")
        final, final_sec, final_conf = primary, secondary, conf
        if v["verdict"] == "change":
            final, final_conf = v["value"], "author"
            final_sec = SECONDARY_ON_CHANGE.get(r["task"], secondary if secondary != final else None)
        sets.append({"set": r["task"], "name": r["name"], "paper": r["key"], "outcome": r["cat"], "metric": r["metric"],
                     "jev": round(100 * r["jev"], 1), "method": round(100 * r["paper"], 1), "gap": pt.delta(r),
                     "primary": final, "secondary": final_sec, "confidence": final_conf, "evidence": evidence,
                     "sources": sources, "status": STATUS, "review_item": v["item"],
                     "author_verdict": v["verdict"] if v["verdict"] != "change" else f"change: {v['value']}",
                     "draft_primary": primary, "draft_secondary": secondary, "draft_confidence": conf})
    papers = sorted({s["paper"] for s in sets})
    counts = {c: sum(s["primary"] == c for s in sets) for c in CAUSES}
    paper_counts = {c: len({s["paper"] for s in sets if s["primary"] == c}) for c in CAUSES}
    by_conf = {c: {k: sum(s["primary"] == c and s["confidence"] == k for s in sets) for k in CONFIDENCE} for c in CAUSES}
    secondary = {c: sum(s["secondary"] == c for s in sets) for c in CAUSES}
    draft_counts = {c: sum(s["draft_primary"] == c for s in sets) for c in CAUSES}
    agree = agreement([s["draft_primary"] for s in sets], [s["primary"] for s in sets])
    return {"generated_by": "scripts/cause_table.py", "status": STATUS, "causes": CAUSES, "n_sets": len(sets),
            "n_papers": len(papers), "counts": counts, "paper_counts": paper_counts, "by_confidence": by_conf,
            "secondary_counts": secondary, "draft_counts": draft_counts, "agreement": agree,
            "n_changed": sum(s["confidence"] == "author" for s in sets), "sets": sets}


def markdown(d: dict) -> str:
    a = d["agreement"]
    out = ["# Causes of the zero-shot shortfalls", "",
           f"**Status: {d['status']}.** Generated by `scripts/cause_table.py`; the assignment rule is that script's "
           "docstring. An LLM drafted every row; an author checked every row and "
           f"changed {d['n_changed']} of {d['n_sets']} primary causes. Agreement of draft and author on the primary "
           f"cause: {a['agree']} of {a['n']}, Cohen's kappa {a['kappa']:.2f} (95% bootstrap interval over rows "
           f"{a['kappa_ci'][0]:.2f} to {a['kappa_ci'][1]:.2f}). Confidence: the draft's rating on confirmed rows, "
           "`author` on changed rows.", "",
           f"Shortfall sets = the {d['n_sets']} comparable test sets (from {d['n_papers']} papers) where Jev does not match "
           "or beat the task-specific method (outcome between or below). Gap = Jev minus the method, in points.", "",
           "| Cause | Sets (primary) | high / medium / low | Papers | Sets as secondary |", "|---|---:|---|---:|---:|"]
    for c, title in CAUSES.items():
        bc = d["by_confidence"][c]
        out.append(f"| ({'abcd'[list(CAUSES).index(c)]}) {title} | {d['counts'][c]} | {bc['high']} / {bc['medium']} / "
                   f"{bc['low']} | {d['paper_counts'][c]} | {d['secondary_counts'][c]} |")
    out += ["", "A paper counts toward a cause if at least one of its shortfall sets has that primary cause.", "",
            "| Set | Paper | Outcome | Jev | Method | Gap | Primary | Secondary | Confidence | Draft primary | Author | Evidence | Sources |",
            "|---|---|---|---:|---:|---:|---|---|---|---|---|---|---|"]
    for s in d["sets"]:
        name = s["name"].replace("\\ ", " ")
        out.append(f"| {name} (`{s['set']}`) | {s['paper']} | {s['outcome']} | {s['jev']:.1f} | {s['method']:.1f} | "
                   f"{s['gap']:+.1f} | {s['primary']} | {s['secondary'] or '–'} | {s['confidence']} | {s['draft_primary']} | "
                   f"{s['author_verdict']} | {s['evidence']} | "
                   + ", ".join(f"`{p}`" for p in s["sources"]) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    d = build()
    (ROOT / "results" / "causes.json").write_text(json.dumps(d, indent=1) + "\n")
    (ROOT / "results" / "CAUSES.md").write_text(markdown(d))
    print(json.dumps({k: d[k] for k in ("n_sets", "n_papers", "counts", "paper_counts", "by_confidence", "secondary_counts",
                                        "draft_counts", "agreement", "n_changed")}))


if __name__ == "__main__":
    main()
