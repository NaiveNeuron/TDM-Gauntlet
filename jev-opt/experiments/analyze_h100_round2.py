"""H100 round 2 (pre-registered) -> results/H100_ROUND2.{md,json}, paper/generated/h100_round2.tex
(macros \\hrBase… for Experiment 1, \\hrNative… for Experiment 2) and paper/generated/app_h100_round2.tex (Table
tab:h100). No inference: reads the h2-qwen-base-<set>-{cal,opt}
and h2-clef-<set>-{native,native-lr} run directories, the Clef RQ3 run directories (cl-clef-<set>-{cal,opt}), Clef's
frozen zero-shot run, results/frozen.json, results/CLEF_RQ3.json and results/gepa_v3_test.json. An experiment without
finished sets is reported as pending.

  uv run --project jev-opt/python python jev-opt/experiments/analyze_h100_round2.py
  ... analyze_h100_round2.py --report-only   # rewrite the .tex outputs from results/H100_ROUND2.json (the .md
                                             # carries hand-written execution notes and is left alone)
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_clef_rq3 import B, SMALL, TITLE, TOL, boot_p, holm, jev_rows  # noqa: E402
from clef_rq3 import RUNS, SETS, cal_dir, frozen_rows, opt_dir, run_config  # noqa: E402
from h100_round2 import CFG8, EXPS, step_dir  # noqa: E402
from resample import cluster_index, jackknife_t, resamples  # noqa: E402

from jev_opt import metrics  # noqa: E402

ROOT = HERE.parents[1]
OUT_JSON = ROOT / "results" / "H100_ROUND2.json"
OUT_MD = ROOT / "results" / "H100_ROUND2.md"
OUT_TEX = ROOT / "paper" / "generated" / "h100_round2.tex"
OUT_APP = ROOT / "paper" / "generated" / "app_h100_round2.tex"
GROUPS = {"cfg8": CFG8, "frozen5": [s for s in SETS if s not in CFG8]}
# (name, system, reference): Δ = system − reference
BASE_CONTRASTS = [("cal-zero", "cal", "zero"), ("opt-zero", "opt", "zero"), ("opt-cal", "opt", "cal"),
                  ("zero-clef", "zero", "clef_zero"), ("cal-clef", "cal", "clef_cal"), ("opt-clef", "opt", "clef_opt")]
NATIVE_CONTRASTS = [("n0-zero", "n0", "zero"), ("nlr-zero", "nlr", "zero"), ("opt-nlr", "opt", "nlr"), ("nlr-n0", "nlr", "n0")]


def finished(exp: str, stem: str) -> bool:
    return all((step_dir(exp, stem, s) / "test_consumed.json").exists() for s in EXPS[exp]["steps"])


def result(path: Path) -> dict[str, Any]:
    return json.loads((path / "result.json").read_text())


def paired(stem: str, ids: list[str], gold: list[str], score: Any, preds: dict[str, list[str]],
           pairs: list[tuple[str, str, str]]) -> tuple[dict[str, float], dict[str, Any]]:
    """Point scores and paired contrasts (cluster bootstrap in RQ1 units, B resamples from seed 0; jackknife-t p on
    clustered sets with fewer than SMALL clusters), as analyze_clef_rq3.set_row."""
    G = np.array(gold, dtype=object)
    P = {k: np.array(v, dtype=object) for k, v in preds.items()}
    point = {k: score(gold, v) for k, v in preds.items()}
    inv, unit = cluster_index(stem, ids)
    m = int(inv.max()) + 1
    draws = {k: np.empty(B) for k in preds}
    for b, ix in enumerate(resamples(inv, B, 0)):
        g = list(G[ix])
        for k in preds:
            draws[k][b] = score(g, list(P[k][ix]))
    out = {}
    for name, a, z in pairs:
        d = draws[a] - draws[z]
        c = {"delta": point[a] - point[z], "ci": [float(x) for x in np.percentile(d, [2.5, 97.5])], "p_boot": boot_p(d)}
        if unit != "item" and m < SMALL:
            stat = lambda ix, a=a, z=z: score([gold[i] for i in ix], [preds[a][i] for i in ix]) - score([gold[i] for i in ix], [preds[z][i] for i in ix])  # noqa: E731
            lo, hi = jackknife_t(stat, inv)
            se = (hi - lo) / (2 * student_t.ppf(0.975, m - 1))
            c["jackknife_ci"] = [lo, hi]
            c["p_jackknife"] = float(2 * student_t.sf(abs(c["delta"]) / se, m - 1)) if se > 0 else 1.0
        c["p"] = c.get("p_jackknife", c["p_boot"])
        c["draws"] = d
        out[name] = c
    return point, {"contrasts": out, "cluster_unit": unit, "n_clusters": m}


def aligned(ids: list[str], other_ids: list[str], other: list[Any], what: str) -> list[Any]:
    pos = {i: k for k, i in enumerate(other_ids)}
    assert sorted(pos) == sorted(ids), what
    return [other[pos[i]] for i in ids]


def clef_reference(stem: str, ids: list[str]) -> dict[str, list[str]]:
    """Clef's RQ3 predictions on `ids`: frozen zero-shot, the same seed program decoded like the base model (seed
    zero-shot where answered, else the byte-identical frozen requests), calibrated, optimized."""
    cal, opt = result(cal_dir("clef", stem)), result(opt_dir("clef", stem))
    rows = frozen_rows("clef", stem)
    frozen = [rows[i]["pred"] for i in ids]
    seed = aligned(ids, cal["test"]["ids"], cal["test"]["seed_pred"], stem) if cal["test"]["seed_pred"] is not None else frozen
    return {"clef_frozen": frozen, "clef_zero": seed, "clef_cal": aligned(ids, cal["test"]["ids"], cal["test"]["calibrated_pred"], stem),
            "clef_opt": aligned(ids, opt["test"]["ids"], opt["test"]["pred"], stem)}


def base_row(stem: str, jev: dict[str, Any], clef_rows: dict[str, Any]) -> dict[str, Any] | None:
    if not finished("base", stem):
        return None
    cal, opt = result(step_dir("base", stem, "cal")), result(step_dir("base", stem, "opt"))
    cfg = run_config(RUNS / SETS[stem][0])
    labels = list(cfg.labels)
    score = lambda g, p: metrics.primary(cfg.primary_metric, g, p, labels, cfg.negative_label)  # noqa: E731
    ids, gold = cal["test"]["ids"], cal["test"]["gold"]
    preds = {"zero": cal["test"]["zero_pred"], "cal": cal["test"]["calibrated_pred"],
             "opt": aligned(ids, opt["test"]["ids"], opt["test"]["pred"], stem), **clef_reference(stem, ids)}
    point, rs = paired(stem, ids, gold, score, preds, BASE_CONTRASTS)
    zero, paper = point["zero"], jev["paper"]
    gap = (lambda x: None if paper - zero <= TOL else (x - zero) / (paper - zero))
    c = clef_rows[stem]
    use = [v for r in (cal, opt) for v in r["usage"].values()]
    return {"set": stem, "title": TITLE[stem][0], "metric": cfg.primary_metric, "n_test": len(ids), **{k: rs[k] for k in ("cluster_unit", "n_clusters")},
            "zero": zero, "calibrated": point["cal"], "optimized": point["opt"],
            "frozen_rule_zero": cal["test"]["frozen_rule_zero_primary"],
            "val": {"seed": cal["val"]["seed_primary"], "calibrated": cal["val"]["calibrated_primary"], **opt["selection"]["val"]},
            "chosen": opt["selection"]["chosen"], "jev_chosen": opt["selection"]["jev_chosen"], "clef_chosen": c["chosen"],
            "clef": {"frozen_zero": point["clef_frozen"], "zero": point["clef_zero"], "calibrated": point["clef_cal"], "optimized": point["clef_opt"],
                     "gap_cal": c["gap_cal"], "gap_opt": c["gap_opt"]},
            "jev": {"zero": jev["zero"], "calibrated": jev["calibrated"], "optimized": jev["optimized"], "gap_cal": jev["gap_cal"], "gap_opt": jev["gap_opt"]},
            "paper": paper, "llm": jev["llm"], "gap_cal": gap(point["cal"]), "gap_opt": gap(point["opt"]),
            "head_zero": point["clef_zero"] - zero, "head_opt": point["clef_opt"] - point["opt"],
            "rejected_examples": len(cal["rejected"]) + len(opt["rejected"]),
            "fresh_inference_s": sum(v.get("fresh_inference_s", 0.0) for v in use), "fresh_input_tokens": sum(v.get("fresh_input_tokens", 0) for v in use),
            "fresh_units": sum(v.get("fresh_calls", 0) for v in use), "contrasts": rs["contrasts"]}


def native_row(stem: str, jev: dict[str, Any], jev_native: dict[str, float]) -> dict[str, Any] | None:
    if not finished("native", stem):
        return None
    n0, nlr = result(step_dir("native", stem, "native")), result(step_dir("native", stem, "native-lr"))
    cfg = run_config(RUNS / SETS[stem][0])
    labels = list(cfg.labels)
    score = lambda g, p: metrics.primary(cfg.primary_metric, g, p, labels, cfg.negative_label)  # noqa: E731
    ids, gold = n0["test"]["ids"], n0["test"]["gold"]
    ref = clef_reference(stem, ids)
    preds = {"n0": n0["test"]["pred"], "nlr": aligned(ids, nlr["test"]["ids"], nlr["test"]["pred"], stem), "zero": ref["clef_frozen"], "opt": ref["clef_opt"]}
    point, rs = paired(stem, ids, gold, score, preds, NATIVE_CONTRASTS)
    gain = point["opt"] - point["zero"]
    use = [v for r in (n0, nlr) for v in r["usage"].values()]
    return {"set": stem, "title": TITLE[stem][0], "metric": cfg.primary_metric, "n_test": len(ids), **{k: rs[k] for k in ("cluster_unit", "n_clusters")},
            "zero": point["zero"], "native_zero": point["n0"], "native_lr": point["nlr"], "optimized": point["opt"],
            "val": {"native_zero": n0["val"]["primary"], "native_lr": nlr["val"]["primary"]}, "paper": jev["paper"],
            "share_without_search": (point["nlr"] - point["zero"]) / gain if gain >= TOL else None,
            "jev_native_zero": jev_native.get(stem), "jev_zero": jev["zero"],
            "rejected_examples": len(n0["rejected"]) + len(nlr["rejected"]),
            "fresh_inference_s": sum(v.get("fresh_inference_s", 0.0) for v in use), "fresh_input_tokens": sum(v.get("fresh_input_tokens", 0) for v in use),
            "contrasts": rs["contrasts"]}


def summarize(rows: list[dict[str, Any]], contrasts: list[tuple[str, str, str]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_sets": len(rows)}
    for name, *_ in contrasts:
        for p, r in zip(holm([r["contrasts"][name]["p"] for r in rows]), rows):
            r["contrasts"][name]["p_holm"] = p
        pooled = np.mean([r["contrasts"][name]["draws"] for r in rows], axis=0)
        cs = [r["contrasts"][name] for r in rows]
        out[name] = {"n": len(rows), "pooled_delta": statistics.mean(c["delta"] for c in cs),
                     "pooled_ci": [float(x) for x in np.percentile(pooled, [2.5, 97.5])], "pooled_p": boot_p(pooled),
                     "n_positive": sum(c["delta"] > 0 for c in cs), "median_delta": statistics.median(c["delta"] for c in cs),
                     "n_ci_above_zero": sum(c["ci"][0] > 0 for c in cs), "n_ci_below_zero": sum(c["ci"][1] < 0 for c in cs),
                     "n_holm_positive": sum(c["p_holm"] < 0.05 and c["delta"] > 0 for c in cs),
                     "n_holm_negative": sum(c["p_holm"] < 0.05 and c["delta"] < 0 for c in cs)}
    out["fresh_inference_s"] = sum(r["fresh_inference_s"] for r in rows)
    out["fresh_input_tokens"] = sum(r["fresh_input_tokens"] for r in rows)
    out["rejected_examples"] = sum(r["rejected_examples"] for r in rows)
    return out


def medians(rows: list[dict[str, Any]], key: Any) -> dict[str, float | None]:
    out = {}
    for g, stems in {"all": list(SETS), **GROUPS}.items():
        v = [x for r in rows if r["set"] in stems and (x := key(r)) is not None]
        out[g] = statistics.median(v) if v else None
    return out


def pct(x: float | None, signed: bool = False) -> str:
    return "–" if x is None else (f"{100 * x:+.1f}" if signed else f"{100 * x:.1f}")


def ipct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.0f}"


def ci(c: dict[str, Any]) -> str:
    return f"{pct(c['delta'], True)} [{pct(c['ci'][0], True)}, {pct(c['ci'][1], True)}] p_Holm {c['p_holm']:.3f}"


def bullet(label: str, x: dict[str, Any]) -> str:
    return (f"- {label}: pooled {pct(x['pooled_delta'], True)} [{pct(x['pooled_ci'][0], True)}, {pct(x['pooled_ci'][1], True)}] over "
            f"{x['n']} sets; median {pct(x['median_delta'], True)}; Δ > 0 on {x['n_positive']}; Holm-significant positive "
            f"{x['n_holm_positive']}, negative {x['n_holm_negative']}.")


def write_md(rep: dict[str, Any]) -> None:
    L = ["# H100 round 2: interface vs backbone, and the optimizer's prior", "",
         "The study was pre-registered. Generated by `jev-opt/experiments/analyze_h100_round2.py`.",
         "Scores ×100 on each paper's primary metric over the gauntlet test items. Δ: paired cluster bootstrap (RQ1 units,",
         "2,000 resamples), 95% CI, Holm over the sets (jackknife-t p on TB-Dense and MATRES).", ""]
    b = rep["base"]
    L += [f"## Experiment 1: Clef's base model through option log-probabilities (`{b['pin']}`, template `{b['template']}`)", "",
          f"Status: {b['status']}.", ""]
    if b["sets"]:
        L += ["Base model (B): zero-shot Z = the paper's question, argmax / expected level of its option probabilities; calibrated C;",
              "optimized O = Jev's selected questions with the readout refitted on B's answers. Clef: the Clef RQ3 run; its seed",
              "zero-shot is the paper's question sent through the optimizer's request path, decoded the same way as B's. It differs",
              "from Clef's frozen zero-shot run (`results/CLEF_RQ3.md`, paper tables) where the two requests differ: "
              + "; ".join(f"{r['title']} {pct(r['clef']['zero'])} vs {pct(r['clef']['frozen_zero'])}" for r in b["sets"]
                          if abs(r["clef"]["zero"] - r["clef"]["frozen_zero"]) > 1e-9) + " (seed vs frozen).", "",
              "| Set | B zero-shot | B calibrated | B optimized | Δ opt. vs B zero [CI] | Clef seed zero-shot | Clef calibrated | Clef optimized | B − Clef optimized [CI] | Jev pipeline | Paper | Gap cal. / opt. (B) |",
              "|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---|"]
        for r in b["sets"]:
            c = r["contrasts"]
            L.append(f"| {r['title']} | {pct(r['zero'])} | {pct(r['calibrated'])} | {pct(r['optimized'])} | {ci(c['opt-zero'])} | "
                     f"{pct(r['clef']['zero'])} | {pct(r['clef']['calibrated'])} | {pct(r['clef']['optimized'])} | {ci(c['opt-clef'])} | "
                     f"{pct(r['jev']['optimized'])} | {pct(r['paper'])} | {ipct(r['gap_cal'])}% / {ipct(r['gap_opt'])}% |")
        s = b["summary"]
        L += ["", bullet("B calibrated − B zero-shot", s["cal-zero"]), bullet("B optimized − B zero-shot", s["opt-zero"]),
              bullet("B optimized − B calibrated", s["opt-cal"]), bullet("B zero-shot − Clef zero-shot", s["zero-clef"]),
              bullet("B calibrated − Clef calibrated", s["cal-clef"]), bullet("B optimized − Clef optimized", s["opt-clef"]),
              f"- Head's advantage (Clef − B): median {pct(b['head']['median_zero'], True)} zero-shot, {pct(b['head']['median_opt'], True)} "
              f"after the refitted readout; smaller after the readout on {b['head']['n_smaller_after']} of {len(b['sets'])} sets.",
              f"- Median gap closed (B): calibrated {ipct(b['gap']['cal']['all'])}%, optimized {ipct(b['gap']['opt']['all'])}% "
              f"(8 configurations {ipct(b['gap']['opt']['cfg8'])}%, 5 frozen sets {ipct(b['gap']['opt']['frozen5'])}%); Clef optimized "
              f"{ipct(b['gap']['clef_opt']['all'])}%, Jev pipeline {ipct(b['gap']['jev_opt']['all'])}%.",
              f"- Selection equals Clef's on {b['n_chosen_as_clef']} sets and Jev's on {b['n_chosen_as_jev']}. Rejected examples: {s['rejected_examples']}.",
              f"- DREsS under the gauntlet's frozen rule (argmax level), B zero-shot: "
              + ", ".join(f"{r['title']} {pct(r['frozen_rule_zero'])}" for r in b["sets"] if r["frozen_rule_zero"] is not None) + ".",
              f"- Compute: {s['fresh_input_tokens']:,} new input tokens, {s['fresh_inference_s'] / 3600:.1f} GPU-hours of new inference.", ""]
    n = rep["native"]
    L += [f"## Experiment 2: Clef on the jev-native questions an LLM wrote from the task definition (`{n['pin']}`)", "",
          f"Status: {n['status']}.", ""]
    if n["sets"]:
        L += ["N0 = the jev-native seed as written; N+LR = with the paper's readout refitted on Clef's fit-slice answers; Z and O = Clef's",
              "frozen zero-shot and optimized questions (Clef RQ3). Share = (N+LR − Z) / (O − Z).", "",
              "| Set | Clef zero-shot | N0 | N+LR | Clef optimized | Δ N+LR vs zero [CI] | Δ opt. vs N+LR [CI] | Share | Jev N0 |",
              "|---|---:|---:|---:|---:|---|---|---:|---:|"]
        for r in n["sets"]:
            c = r["contrasts"]
            L.append(f"| {r['title']} | {pct(r['zero'])} | {pct(r['native_zero'])} | {pct(r['native_lr'])} | {pct(r['optimized'])} | "
                     f"{ci(c['nlr-zero'])} | {ci(c['opt-nlr'])} | {ipct(r['share_without_search'])}% | {pct(r['jev_native_zero'])} |")
        s = n["summary"]
        L += ["", bullet("N0 − Clef zero-shot", s["n0-zero"]), bullet("N+LR − Clef zero-shot", s["nlr-zero"]),
              bullet("Clef optimized − N+LR", s["opt-nlr"]), bullet("N+LR − N0", s["nlr-n0"]),
              f"- Median share of the optimized gain without the search: {ipct(n['share']['all'])}% (8 configurations "
              f"{ipct(n['share']['cfg8'])}%, 5 frozen sets {ipct(n['share']['frozen5'])}%). Rejected examples: {s['rejected_examples']}.",
              f"- Compute: {s['fresh_input_tokens']:,} new input tokens, {s['fresh_inference_s'] / 3600:.1f} GPU-hours of new inference.", ""]
    OUT_MD.write_text("\n".join(L) + "\n")


def write_tex(rep: dict[str, Any]) -> None:
    m: list[tuple[str, str]] = []
    for exp, p, contrasts in (("base", "hrBase", BASE_CONTRASTS), ("native", "hrNative", NATIVE_CONTRASTS)):
        e = rep[exp]
        rows = e["sets"]
        m.append((f"{p}Status", "finished" if len(rows) == len(SETS) else ("pending" if not rows else f"{len(rows)} of {len(SETS)} sets")))
        if not rows:
            continue
        m.append((f"{p}Nsets", str(len(rows))))
        for r in rows:
            t = TITLE[r["set"]][1]
            if exp == "base":
                m += [(f"{p}Zero{t}", pct(r["zero"])), (f"{p}Cal{t}", pct(r["calibrated"])), (f"{p}Opt{t}", pct(r["optimized"])),
                      (f"{p}DoptClef{t}", pct(r["contrasts"]["opt-clef"]["delta"], True))]
            else:
                m += [(f"{p}Nzero{t}", pct(r["native_zero"])), (f"{p}Nlr{t}", pct(r["native_lr"])),
                      (f"{p}DoptNlr{t}", pct(r["contrasts"]["opt-nlr"]["delta"], True))]
        for name, *_ in contrasts:
            # TeX control words are letters only: the contrast "n0-zero" becomes "NzeroZero", "nlr-n0" "NlrNzero".
            key = "".join(w.capitalize() for w in name.replace("n0", "nzero").split("-"))
            x = e["summary"][name]
            m += [(f"{p}Pool{key}", pct(x["pooled_delta"], True)), (f"{p}Pool{key}Lo", pct(x["pooled_ci"][0], True)),
                  (f"{p}Pool{key}Hi", pct(x["pooled_ci"][1], True)), (f"{p}Med{key}", pct(x["median_delta"], True)),
                  (f"{p}Npos{key}", str(x["n_positive"])), (f"{p}Nholm{key}", str(x["n_holm_positive"])), (f"{p}Nneg{key}", str(x["n_holm_negative"]))]
        m += [(f"{p}GpuHours", f"{e['summary']['fresh_inference_s'] / 3600:.1f}"), (f"{p}InputTokensM", f"{e['summary']['fresh_input_tokens'] / 1e6:.0f}")]
        if exp == "base":
            m += [(f"{p}GapCal", ipct(e["gap"]["cal"]["all"])), (f"{p}GapOpt", ipct(e["gap"]["opt"]["all"])),
                  (f"{p}GapOptCfg", ipct(e["gap"]["opt"]["cfg8"])), (f"{p}GapOptFrozen", ipct(e["gap"]["opt"]["frozen5"])),
                  (f"{p}HeadZero", pct(e["head"]["median_zero"], True)), (f"{p}HeadOpt", pct(e["head"]["median_opt"], True)),
                  (f"{p}HeadSmaller", str(e["head"]["n_smaller_after"]))]
        else:
            m += [(f"{p}Share", ipct(e["share"]["all"])), (f"{p}ShareCfg", ipct(e["share"]["cfg8"])), (f"{p}ShareFrozen", ipct(e["share"]["frozen5"]))]
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("% generated by jev-opt/experiments/analyze_h100_round2.py; do not edit\n"
                       + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m))


def write_app(rep: dict[str, Any]) -> None:
    """Appendix table: both experiments per set, with Clef's and Jev's pipelines and the paper's method."""
    b = {r["set"]: r for r in rep["base"]["sets"]}
    n = {r["set"]: r for r in rep["native"]["sets"]}
    if not b or not n:
        return

    def sg(x: float) -> str:
        return pct(x, True).replace("-", "$-$")

    def cell(c: dict[str, Any]) -> str:
        jk = "p_jackknife" in c
        mark = ("$^{\\ast\\dagger}$" if jk else "$^\\ast$") if c["p_holm"] < 0.05 else ("$^\\dagger$" if jk else "")
        return f"{sg(c['delta'])}{mark} {{\\tiny({sg(c['ci'][0])}, {sg(c['ci'][1])})}}"

    out = ["% generated by jev-opt/experiments/analyze_h100_round2.py; do not edit",
           "\\begin{table*}[t]", "\\centering\\scriptsize", "\\setlength{\\tabcolsep}{3pt}", "\\resizebox{\\textwidth}{!}{%",
           "\\begin{tabular}{@{}lrrrlrrrlrrr@{}}", "\\toprule",
           "& \\multicolumn{4}{c}{Qwen3.8-27B, option log-probabilities} & \\multicolumn{4}{c}{Clef} & & & \\\\",
           "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
           "Set & Zero & Cal. & Opt. & Opt.\\ $-$ Clef opt.\\ (95\\% CI) & Zero & Rephr. & + LR & Opt.\\ $-$ (Rephr.\\ + LR) (95\\% CI) "
           "& Share & \\jev{} & Paper \\\\", "\\midrule"]
    for i, stem in enumerate(s for s in SETS if s in b):
        if i == len(CFG8):
            out.append("\\midrule")
        r, q = b[stem], n[stem]
        out.append(f"{r['title']} & {pct(r['zero'])} & {pct(r['calibrated'])} & {pct(r['optimized'])} & {cell(r['contrasts']['opt-clef'])} & "
                   f"{pct(q['zero'])} & {pct(q['native_zero'])} & {pct(q['native_lr'])} & {cell(q['contrasts']['opt-nlr'])} & "
                   f"{ipct(q['share_without_search'])}\\% & {pct(r['jev']['optimized'])} & {pct(r['paper'])} \\\\")
    bs, ns = rep["base"]["summary"], rep["native"]["summary"]
    out += ["\\bottomrule", "\\end{tabular}}",
            "\\caption{Clef's base model, and Clef on the rephrased question, on the RQ3 sets (test, $\\times$100; the "
            "\\optNTasks{} configurations, then the \\fzNewNSets{} rule-selected sets). Left: Qwen3.8-27B read through the "
            "probabilities of the option tokens. Zero: the paper's question; Cal.: per-class bias and temperature; Opt.: "
            "\\jev{}'s seed-0 optimized questions, logistic regression refitted on its answers; Clef opt.: the same step on "
            "Clef (Table~\\ref{tab:clef-rq3-clef}). Right: Clef. Zero: its frozen zero-shot run, as in Tables~\\ref{tab:clef} "
            "and~\\ref{tab:clef-rq3-clef} (the paper's question sent through the optimizer's request path scores "
            + ", ".join(f"{pct(r['clef']['zero'])} on {r['title']}" for r in rep["base"]["sets"]
                        if abs(r["clef"]["zero"] - r["clef"]["frozen_zero"]) > 1e-9)
            + ", where the two requests differ); Rephr.: the rephrased "
            "question (\\S\\ref{sec:fqs}), written by an LLM from the task definition, the best of three chosen on \\jev{}'s "
            "training labels, as written; + LR: with the logistic regression refitted on Clef's fit-slice answers; Opt.: "
            "Clef's optimized step. Share: (Rephr.\\ + LR $-$ Zero) / (Opt.\\ $-$ Zero). \\jev{}: \\jev{}'s seed-0 pipeline. "
            "$\\Delta$: paired cluster bootstrap with RQ1's units; $^\\ast$: Holm $p<0.05$ over the 13 sets; $^\\dagger$: "
            "jackknife-t $p$ (fewer than 30 documents). "
            f"Pooled: Opt.\\ $-$ Clef opt.\\ {sg(bs['opt-clef']['pooled_delta'])}, Clef: Opt.\\ $-$ (Rephr.\\ + LR) "
            f"{sg(ns['opt-nlr']['pooled_delta'])}.}}",
            "\\label{tab:h100}", "\\end{table*}", ""]
    OUT_APP.parent.mkdir(parents=True, exist_ok=True)
    OUT_APP.write_text("\n".join(out) + "\n")


def strip(rows: list[dict[str, Any]]) -> None:
    for r in rows:
        for c in r["contrasts"].values():
            c.pop("draws", None)


def main() -> int:
    if "--report-only" in sys.argv[1:]:
        rep = json.loads(OUT_JSON.read_text())
        write_tex(rep)
        write_app(rep)
        return 0
    jev = jev_rows()
    clef_rows = {r["set"]: r for r in json.loads((ROOT / "results" / "CLEF_RQ3.json").read_text())["clef"]["sets"]}
    gepa = json.loads((ROOT / "results" / "gepa_v3_test.json").read_text())["tasks"]
    jev_native = {s: gepa[s]["rows"]["Jev-native zero-shot (v3 N)"]["test"] for s in CFG8 if s in gepa}
    rep: dict[str, Any] = {"bootstrap": {"B": B, "seed": 0}}

    rows = [r for s in SETS if (r := base_row(s, jev[s], clef_rows)) is not None]
    base: dict[str, Any] = {"pin": EXPS["base"]["pin"], "template": "qwen-base-logprob-v1", "sets": rows,
                            "status": "finished" if len(rows) == len(SETS) else "pending: " + ", ".join(s for s in SETS if s not in {r["set"] for r in rows})}
    if rows:
        base["summary"] = summarize(rows, BASE_CONTRASTS)
        base["gap"] = {"cal": medians(rows, lambda r: r["gap_cal"]), "opt": medians(rows, lambda r: r["gap_opt"]),
                       "clef_opt": medians(rows, lambda r: r["clef"]["gap_opt"]), "jev_opt": medians(rows, lambda r: r["jev"]["gap_opt"])}
        base["head"] = {"median_zero": statistics.median(r["head_zero"] for r in rows), "median_opt": statistics.median(r["head_opt"] for r in rows),
                        "n_smaller_after": sum(r["head_opt"] < r["head_zero"] for r in rows)}
        base["n_chosen_as_clef"] = sum(r["chosen"] == r["clef_chosen"] for r in rows)
        base["n_chosen_as_jev"] = sum(r["chosen"] == r["jev_chosen"] for r in rows)
    rows2 = [r for s in SETS if (r := native_row(s, jev[s], jev_native)) is not None]
    native: dict[str, Any] = {"pin": EXPS["native"]["pin"], "sets": rows2,
                              "status": "finished" if len(rows2) == len(SETS) else "pending: " + ", ".join(s for s in SETS if s not in {r["set"] for r in rows2})}
    if rows2:
        native["summary"] = summarize(rows2, NATIVE_CONTRASTS)
        native["share"] = medians(rows2, lambda r: r["share_without_search"])
    strip(rows)
    strip(rows2)
    rep["base"], rep["native"] = base, native
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rep, indent=1) + "\n")
    write_md(rep)
    write_tex(rep)
    write_app(rep)
    print(f"base: {len(rows)} sets finished; native: {len(rows2)} sets finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
