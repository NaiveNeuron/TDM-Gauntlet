"""Offline checks of the zero-shot Jev / Clef contrast, and one RQ1 interval (no model calls, no GPU).

Inputs: results/decision_models_comparison.json (completed Clef / Clef-flash cohorts), the saved predictions
and raw responses under runs/ (Jev) and runs/clef*/ (Clef, Clef-flash), the frozen test sets, scripts/paper_tables.py
(Table full values and RQ1 groups), results/causes.json (primary causes) and scripts/framing_audit_ci.py (per-set
metric and resampling specs; it patches the network to raise).

(1) CICC rounding counterfactual. For each compared CICC set (MTOD, HWU64, CLINC150, ACID, ATIS) and model, the raw
    returned probabilities are scored with the repository's CICC.predict / CICC.score as returned and rounded to two
    decimals, as Jev returns them (CICC.predict renormalises). "Gold zero" is the share of calibration items whose
    gold intent gets probability 0 after rounding to two decimals. Conformal sets admit every intent (q-hat = 1) when
    that share exceeds alpha.
(2) Agreement on the Table-full test sets with a completed run, using each set's Table-full value (the paper's
    protocol where Table full uses it). RQ1 groups as paper_tables.rows() (match within 0.5 points of the method;
    between: at least the paper's LLM, or 0.8 of the method without one; else below). Spearman correlation of the
    two models' scores relative to the paper's method. Median of Clef - Jev in the primary metric (the macro
    \\clMedian). Paper-clustered intervals: papers (bibkeys) resampled with replacement, their sets kept as they are,
    B = 10,000, numpy default_rng(0); 95% percentile interval.
(3) Paired test per set: the units of framing_audit_ci.build() resampled once per replicate and both models scored
    on the same draw (clusters as in offline_rq1.cluster_keys), B = 10,000, default_rng(0) per set. Difference
    p = 2 Phi(-|d| / SE) with SE the bootstrap SD of the paired difference; Holm over the sets. Sets whose metric
    the specs cannot recompute from Clef's predictions are listed, not tested.
(4) RQ1: paper-clustered interval of the Spearman correlation between the paper's prompted LLM and Jev, both
    relative to the paper's method, over the sets with an LLM baseline (the macro \\bodyRhoLLM). Both ratios share
    the paper's score as denominator, which by itself induces a correlation (Pearson 1897). Shared-denominator null:
    Jev's scores permuted across the sets with each set's denominator kept, B = 10,000, default_rng(0); one-sided
    p = share of permutations with rho at least the observed one. Partial: Spearman correlation of the residuals of
    log LLM score and log Jev score, each regressed (least squares) on the log paper score.
(5) Paired separations split by intent-detection sets (CICC and Navigating the Unknown, whose data Clef's model card
    reports on), since the paper's contamination paragraph names them. The CICC specs hold the conformal threshold
    q-hat fixed at its calibration-split value and resample test utterances only.

Run from the repo root: uv run python scripts/offline_clef.py   (`--B=N` for a dry run that writes nothing;
`--macros` rewrites paper/generated/offline_clef.tex from the saved results/offline_v4/clef.json)
"""

from __future__ import annotations

import gzip
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import framing_audit_ci as F  # noqa: E402  (patches httpx / JevClient to raise)
import offline_rq1  # noqa: E402
import paper_tables  # noqa: E402
from gauntlet.task import read_jsonl  # noqa: E402
from gauntlet.tasks import REGISTRY  # noqa: E402
from gauntlet.tasks.cicc import CICC  # noqa: E402

OUT = ROOT / "results" / "offline_v4" / "clef.json"
TEX = ROOT / "paper" / "generated" / "offline_clef.tex"
SRC = ROOT / "results" / "decision_models_comparison.json"
B = 10_000
SEED = 0
ALPHA = 0.05
TOL = 0.01  # "within a point", as scripts/paper_clef.py
MODELS = {"jev": "", "clef": "clef/", "clef-flash": "clef-flash/"}  # run-root prefix under runs/
CICC_SETS = ("mtod", "hwu64", "c150", "acid", "atis")
INTENT = ("cicc/", "nav_unknown/")  # intent-detection sets (BANKING77, CLINC150, HWU64, ACID, ATIS, MTOD)


# ------------------------------------------------------------------ helpers
def spearman(x: list[float], y: list[float]) -> float:
    def ranks(v: list[float]) -> np.ndarray:
        a = np.asarray(v, float)
        r = np.empty(len(a))
        r[a.argsort(kind="stable")] = np.arange(len(a))
        for val in np.unique(a):  # average ties
            r[a == val] = r[a == val].mean()
        return r
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def paper_ci(rows: list[dict], stat, b: int) -> dict:
    """Papers resampled with replacement; the sets of a drawn paper kept as they are."""
    papers = sorted({r["paper"] for r in rows})
    of = {p: [r for r in rows if r["paper"] == p] for p in papers}
    rng = np.random.default_rng(SEED)
    vals = np.array([stat([r for i in rng.integers(0, len(papers), len(papers)) for r in of[papers[i]]])
                     for _ in range(b)], float)
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"point": stat(rows), "lo": float(lo), "hi": float(hi), "n_papers": len(papers), "n_finite": len(vals)}


def group(v: float, paper: float, llm: float | None) -> str:
    if v >= paper - 0.005:
        return "replaces"
    if llm is not None and v >= llm - 0.005:
        return "between"
    if llm is None and v / paper >= 0.8:
        return "between"
    return "below"


def table_value(task: str, scores: dict) -> float:
    return F.value_from(task, scores)


# ------------------------------------------------------------------ (1) CICC rounding
def cicc_rounding() -> dict:
    out = {}
    for ds in CICC_SETS:
        t = CICC(ds)
        items = t.items()
        cal = [it for it in items if "calibration" in it.fields["roles"]]
        idx = {lab: i for i, lab in enumerate(t.labels)}
        out[ds] = {"alpha": t.alpha, "n_calibration": len(cal)}
        for model, prefix in MODELS.items():
            run = ROOT / "runs" / f"{prefix}cicc" / ds
            answers = {}
            for shard in sorted(run.glob("responses*.jsonl.gz")):
                with gzip.open(shard, "rt") as fh:
                    for line in fh:
                        r = json.loads(line)
                        answers[r["item_id"]] = r["response"]["answers"]
            if set(answers) != {it.id for it in items}:
                raise SystemExit(f"{model} cicc/{ds}: responses do not cover the frozen cohort")
            res = {}
            for label, decimals in (("returned", None), ("two_decimals", 2)):
                preds = {}
                for it in items:
                    a = json.loads(json.dumps(answers[it.id]))
                    if decimals is not None:
                        probs = a["intent"].get("probabilities") or {}
                        a["intent"]["probabilities"] = {k: round(float(v), decimals) for k, v in probs.items()}
                    preds[it.id] = t.predict(it, {"intent": a})
                s = t.score(items, preds)
                zero = sum(preds[it.id]["probs"][idx[it.gold]] == 0 for it in cal) / len(cal)
                res[label] = {"qhat": s["qhat"], "single": s["single"], "mean_set_size": s["mean_set_size"],
                              "accuracy": s["accuracy"], "gold_zero_calibration": zero}
            saved = json.loads((run / "scores.json").read_text())
            if abs(res["returned"]["single"] - saved["single"]) > 1e-12 or abs(res["returned"]["qhat"] - saved["qhat"]) > 1e-12:
                raise SystemExit(f"{model} cicc/{ds}: rescoring differs from the saved scores.json")
            out[ds][model] = res
    return out


# ------------------------------------------------------------------ (2) agreement, (3) paired tests
def completed_sets(d: dict, model: str) -> list[dict]:
    rows = {(r["task"], r["split"]): r for r in d[model]["rows"]}
    ref = {r["task"]: r for r in paper_tables.rows()}
    out = []
    for task, key, *_ in paper_tables.DATASETS:
        r = rows[(task, "test")]
        if r["status"] != "completed":
            continue
        R = ref[task]
        jv = table_value(task, r["jev"]["scores"])
        cv = table_value(task, r["clef"]["scores"])
        if abs(jv - R["jev"]) > 1e-9:
            raise SystemExit(f"{task}: Jev value in the comparison differs from Table full")
        out.append({"task": task, "paper": key, "jev": jv, "clef": cv, "method": R["paper"], "llm": R["llm"],
                    "jev_group": R["cat"], "clef_group": group(cv, R["paper"], R["llm"]),
                    "jev_rel": jv / R["paper"], "clef_rel": cv / R["paper"], "delta": r["delta"], "clef_path": r["clef"]["path"]})
    return out


def agreement(sets: list[dict], b: int) -> dict:
    cross = Counter(f"{s['jev_group']}->{s['clef_group']}" for s in sets)
    return {
        "n_sets": len(sets),
        "jev_groups": dict(Counter(s["jev_group"] for s in sets)),
        "clef_groups": dict(Counter(s["clef_group"] for s in sets)),
        "crosstab": dict(cross),
        "same_group": sum(s["jev_group"] == s["clef_group"] for s in sets),
        "jev_matches": sum(s["jev_group"] == "replaces" for s in sets),
        "matches_shared": sum(s["jev_group"] == s["clef_group"] == "replaces" for s in sets),
        "spearman_rel": paper_ci(sets, lambda xs: spearman([x["jev_rel"] for x in xs], [x["clef_rel"] for x in xs]), b),
        "median_delta": paper_ci(sets, lambda xs: float(np.median([x["delta"] for x in xs])), b),
        "median_delta_sets": statistics.median(s["delta"] for s in sets),
    }


def paired(sets: list[dict], b: int) -> dict:
    rows, skipped = [], []
    for s in sets:
        name = s["task"]
        task = REGISTRY[name]()
        items = task.items()
        pj = F.load_predictions(name)
        path = ROOT / "runs" / s["clef_path"]
        path = path / "predictions.jsonl" if (path / "predictions.jsonl").exists() else path / "predictions.jsonl.gz"
        pc = {r["item_id"]: r["prediction"] for r in read_jsonl(path)}
        try:
            sj = F.build(name, task, items, pj)
            sc = F.build(name, task, items, pc)
        except Exception as e:  # noqa: BLE001  (a spec that cannot take Clef's predictions is listed, not tested)
            skipped.append({"task": name, "reason": f"{type(e).__name__}: {e}"})
            continue
        full = np.arange(sj.n_units)
        if sj.n_units != sc.n_units or abs(sj.stat(full) - s["jev"]) > 1e-9 or abs(sc.stat(full) - s["clef"]) > 1e-9:
            skipped.append({"task": name, "reason": "resampling spec does not reproduce the saved values"})
            continue
        ck = offline_rq1.cluster_keys(name, items)
        rng = np.random.default_rng(SEED)
        if ck is None:
            draws = (F.draw(rng, sj, sort=sj.ordered) for _ in range(b))
        else:
            if sj.ordered:
                raise SystemExit(f"{name}: clustered draws assume an order-free statistic")
            cd = offline_rq1.ClusterDraw(ck[1], sj.strata)
            draws = (cd(rng) for _ in range(b))
        diffs = np.array([sc.stat(i) - sj.stat(i) for i in draws], float)
        d = s["clef"] - s["jev"]
        se = float(np.nanstd(diffs, ddof=1))
        rows.append({"task": name, "unit": ck[0] if ck else sj.unit, "diff": d, "se": se,
                     "ci95": [float(x) for x in np.nanpercentile(diffs, [2.5, 97.5])], "p": offline_rq1.p_diff(d, se)})
        print(f"  {name:45s} d={100 * d:+6.1f} se={100 * se:5.2f} p={rows[-1]['p']:.4f}", flush=True)
    for r, ph in zip(rows, offline_rq1.holm([r["p"] for r in rows])):
        r["p_holm"] = ph
        r["separated"] = ("higher" if r["diff"] > 0 else "lower") if ph < ALPHA else None
    intent = [r for r in rows if r["task"].startswith(INTENT)]
    other = [r for r in rows if not r["task"].startswith(INTENT)]
    split = {name: {"n_tested": len(rs), "n_higher": sum(r["separated"] == "higher" for r in rs),
                    "n_lower": sum(r["separated"] == "lower" for r in rs)} for name, rs in (("intent", intent), ("other", other))}
    return {"n_tested": len(rows), "n_higher": sum(r["separated"] == "higher" for r in rows),
            "n_lower": sum(r["separated"] == "lower" for r in rows), "by_intent": split, "rows": rows, "skipped": skipped}


# ------------------------------------------------------------------ (4) RQ1 rho interval
def rho_llm(b: int) -> dict:
    key = {t: k for t, k, *_ in paper_tables.DATASETS}
    src = [r for r in paper_tables.rows() if r["llm"] is not None and r["paper"] is not None]
    rows = [{"paper": key[r["task"]], "llm_rel": r["llm"] / r["paper"], "jev_rel": r["rel"]} for r in src]
    out = paper_ci(rows, lambda xs: spearman([x["llm_rel"] for x in xs], [x["jev_rel"] for x in xs]), b)
    out["n_sets"] = len(rows)
    llm = np.array([r["llm"] for r in src], float)
    jev = np.array([r["jev"] for r in src], float)
    paper = np.array([r["paper"] for r in src], float)
    rng = np.random.default_rng(SEED)
    null = np.array([spearman(list(llm / paper), list(jev[rng.permutation(len(jev))] / paper)) for _ in range(b)])
    out["shared_denominator_null"] = {"mean": float(null.mean()), "q95": float(np.percentile(null, 95)),
                                      "p_one_sided": float((null >= out["point"]).mean()), "B": b}
    if min(llm.min(), jev.min(), paper.min()) <= 0:
        raise SystemExit("rho_llm partial: a score is not positive; the log regression needs another control")

    def resid(y: np.ndarray) -> np.ndarray:
        x = np.c_[np.ones(len(paper)), np.log(paper)]
        return y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    out["partial_log_paper"] = spearman(list(resid(np.log(llm))), list(resid(np.log(jev))))
    return out


# ------------------------------------------------------------------ macros
def num(x: float, nd: int = 1, sign: bool = False) -> str:
    x = round(x, nd) + 0.0  # no "-0.0"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$")


def macros(res: dict) -> str:
    m: dict[str, str] = {}
    # The text states that the conformal threshold admits every intent exactly where the gold-zero share exceeds alpha.
    for ds, c in res["cicc"].items():
        for model in MODELS:
            for k, v in c[model].items():
                if (v["qhat"] >= 1.0) != (v["gold_zero_calibration"] > c["alpha"]):
                    raise SystemExit(f"cicc/{ds} {model} {k}: full sets do not follow gold-zero share > alpha; update the text")
    for ds, g in (("c150", "Clinc"), ("acid", "Acid"), ("atis", "Atis"), ("hwu64", "Hwu"), ("mtod", "Mtod")):
        c = res["cicc"][ds]
        m[f"ocCicc{g}Alpha"] = num(100 * c["alpha"], 0)
        m[f"ocCicc{g}ClefRoundSingle"] = num(100 * c["clef"]["two_decimals"]["single"], 1)
        m[f"ocCicc{g}ClefSingle"] = num(100 * c["clef"]["returned"]["single"], 1)  # four decimals, as returned
        for model, inf in (("jev", "Jev"), ("clef", "Clef")):
            m[f"ocCicc{g}{inf}Zero"] = num(100 * c[model]["two_decimals"]["gold_zero_calibration"], 1)
            m[f"ocCicc{g}{inf}Acc"] = num(100 * c[model]["returned"]["accuracy"], 1)
    full = [100 * c["jev"]["two_decimals"]["gold_zero_calibration"] for c in res["cicc"].values()
            if c["jev"]["two_decimals"]["qhat"] >= 1.0]
    m["ocCiccJevZeroMin"] = num(min(full), 1)  # over the sets where Jev's threshold admits every intent
    m["ocCiccJevZeroMax"] = num(max(full), 1)
    a = res["agreement"]["clef"]
    if a["matches_shared"] != a["jev_matches"]:
        raise SystemExit("Clef does not match on every set where Jev does; update the text")
    m["ocNSets"] = str(a["n_sets"])
    m["ocSameGroup"] = str(a["same_group"])
    m["ocJevMatches"] = str(a["jev_matches"])
    m["ocRho"] = num(a["spearman_rel"]["point"], 2)
    m["ocRhoLo"] = num(a["spearman_rel"]["lo"], 2)
    m["ocRhoHi"] = num(a["spearman_rel"]["hi"], 2)
    m["ocMedLo"] = num(100 * a["median_delta"]["lo"], 1, sign=True)
    m["ocMedHi"] = num(100 * a["median_delta"]["hi"], 1, sign=True)
    m["ocNPapers"] = str(a["median_delta"]["n_papers"])
    f = res["agreement"]["clef-flash"]
    m["ocFlashSameGroup"] = str(f["same_group"])
    m["ocFlashRho"] = num(f["spearman_rel"]["point"], 2)
    m["ocFlashMedLo"] = num(100 * f["median_delta"]["lo"], 1, sign=True)
    m["ocFlashMedHi"] = num(100 * f["median_delta"]["hi"], 1, sign=True)
    p = res["paired"]
    m["ocPairNTested"] = str(p["n_tested"])
    m["ocPairNHigher"] = str(p["n_higher"])
    m["ocPairNLower"] = str(p["n_lower"])
    m["ocPairNSep"] = str(p["n_higher"] + p["n_lower"])
    for name, inf in (("intent", "Intent"), ("other", "Other")):
        s = p["by_intent"][name]
        m[f"ocPair{inf}NTested"] = str(s["n_tested"])
        m[f"ocPair{inf}NHigher"] = str(s["n_higher"])
        m[f"ocPair{inf}NLower"] = str(s["n_lower"])
    r = res["rho_llm"]
    m["ocRhoLlmLo"] = num(r["lo"], 2)
    m["ocRhoLlmHi"] = num(r["hi"], 2)
    nl = r["shared_denominator_null"]
    m["ocRhoLlmNullMean"] = num(nl["mean"], 2)
    m["ocRhoLlmNullHi"] = num(nl["q95"], 2)
    m["ocRhoLlmNullP"] = num(nl["p_one_sided"], 2)
    m["ocRhoLlmPartial"] = num(r["partial_log_paper"], 2)
    header = "% Generated by scripts/offline_clef.py from results/offline_v4/clef.json. Do not edit."
    return "\n".join([header] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]) + "\n"


def write_macros() -> None:
    """Rewrite the macros from the saved results/offline_v4/clef.json (no recomputation)."""
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text(macros(json.loads(OUT.read_text())))


def main() -> None:
    if "--macros" in sys.argv[1:]:
        write_macros()
        return
    b = next((int(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("--B=")), B)
    t0 = time.time()
    d = json.loads(SRC.read_text())
    res: dict = {"B": b, "seed": SEED}
    print("CICC rounding", flush=True)
    res["cicc"] = cicc_rounding()
    res["agreement"] = {m: agreement(completed_sets(d, m), b) for m in ("clef", "clef-flash")}
    print("paired tests (Clef vs Jev)", flush=True)
    res["paired"] = paired(completed_sets(d, "clef"), b)
    res["rho_llm"] = rho_llm(b)
    res["seconds"] = round(time.time() - t0)
    print({k: v for k, v in res.items() if k not in ("paired", "cicc")})
    print({k: v for k, v in res["paired"].items() if k != "rows"})
    if b != B:
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    write_macros()


if __name__ == "__main__":
    main()
