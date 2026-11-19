# /// script
# requires-python = ">=3.11"
# ///
"""Appendix tables of the confidence analyses: paper/generated/app_confidence.tex from results/confidence.json.

Offline; reads only the saved results file (written by scripts/confidence_zeroshot.py and
jev-opt/experiments/confidence_readout.py; definitions in results/CONFIDENCE.md). Body macros (\\cf...) and the
figure come from scripts/paper_fig_confidence.py. Usage: uv run scripts/paper_appendix_confidence.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "results" / "confidence.json").read_text())
OUT = ROOT / "paper" / "generated" / "app_confidence.tex"

CAUSES = [("matched", "Matched"), ("conventions", "(i) Conventions"), ("opaque", "(ii) Opaque label spaces"),
          ("coarse", "(iii) Coarse probabilities"), ("unassigned", "Failing, no cause named")]
CAUSE_SHORT = {"matched": "matched", "conventions": "(i)", "opaque": "(ii)", "coarse": "(iii)", "unassigned": "--"}
OUTCOMES = [("replaces", "Matches"), ("between", "Between"), ("below", "Below both")]
OUTCOME_SHORT = {"replaces": "match", "between": "betw.", "below": "below"}
RQ3_STATS = [("ece", "ECE", 3), ("brier", "Brier", 3), ("auroc", "AUROC", 2)]


def missing(x: float | None) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def pct(x: float | None, d: int = 0) -> str:
    return "--" if missing(x) else f"{100 * x:.{d}f}"


def num(x: float | None, d: int = 2) -> str:
    return "--" if missing(x) else f"{x:.{d}f}"


def signed(x: float, d: int) -> str:
    return f"{x:+.{d}f}".replace("-", "$-$")


def ci(c: list[float] | None, d: int = 2, scale: float = 1.0) -> str:
    if c is None or any(missing(v) for v in c):
        return ""
    return f"[{c[0] * scale:.{d}f}, {c[1] * scale:.{d}f}]".replace("-", "$-$")


def p_holm(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def thousands(n: int) -> str:
    return f"{n:,}"


def table(env: str, colspec: str, header: str, rows: list[str], caption: str, label: str, size: str = "\\scriptsize") -> str:
    return (f"\\begin{{{env}}}[!htbp]\n\\centering{size}\n\\setlength{{\\tabcolsep}}{{3pt}}\n"
            f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n" + "\n".join(rows)
            + f"\n\\bottomrule\n\\end{{tabular}}\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{{env}}}\n")


def macros() -> str:
    b = RES["bootstrap"]
    ex = RES["excluded"]
    items = "".join(f"\\item {v['name']} ({OUTCOME_SHORT[v['outcome']]}, cause {CAUSE_SHORT[v['cause']]}): {v['reason']}.\n"
                    for v in ex.values())
    return (f"\\newcommand{{\\cfAppBootB}}{{{thousands(b['B'])}}}\n"
            f"\\newcommand{{\\cfAppMinBand}}{{{RES['min_band_units']}}}\n"
            f"\\newcommand{{\\cfAppHalvings}}{{{RES['halvings']}}}\n"
            f"\\newcommand{{\\cfAppTopBand}}{{{RES['bands'][-1]}}}\n"
            f"\\newcommand{{\\cfAppBands}}{{{', '.join(RES['bands'])}}}\n"
            f"\\newcommand{{\\cfAppExcluded}}{{\\begin{{itemize}}\\setlength{{\\itemsep}}{{0pt}}\\setlength{{\\parsep}}{{0pt}}\n"
            f"{items}\\end{{itemize}}}}\n")


def groups_tex() -> str:
    rq = RES["rq2"]
    rows = []
    for block, spec in (("by_cause", CAUSES), ("by_outcome", OUTCOMES)):
        if rows:
            rows.append("\\midrule")
        for key, name in spec:
            g = rq[block][key]
            bands = " & ".join(pct(v) for v in g["band_error_rate"])
            pap = ci(g["auroc_mean_ci_papers"]) or "--"
            rows.append(f"{name} & {g['n_sets']} & {g['n_papers']} & {pct(g['accuracy_mean'])} & {num(g['auroc_mean'])} "
                        f"{ci(g['auroc_mean_ci'])} & {pap} & {num(g['ece_mean'], 3)} ({g['ece_n_sets']}) & {bands} & "
                        f"{num(g['augrc_mean'], 3)} \\\\")
    names = {"conventions_minus_matched": "(i) $-$ matched", "conventions_minus_unassigned": "(i) $-$ no cause",
             "below_minus_replaces": "below both $-$ matches", "conventions_minus_opaque": "(i) $-$ (ii)",
             "conventions_minus_coarse": "(i) $-$ (iii)"}
    stats = (("auroc", "AUROC", 2, 1), ("ece", "ECE", 3, 1), ("top_rate", "top-band error", 0, 100), ("augrc", "AUGRC", 3, 1))

    def diff(d: dict, nd: int, sc: float, key: str) -> str:
        out = f"{signed(sc * d['delta'], nd)} {ci(d[key], nd, sc)}"
        return out + (f" ($p$ {p_holm(d['p_holm'])})" if "p_holm" in d else "")

    tests = ["{}: {}".format(names[k], ", ".join(f"{lab} {diff(v[s], nd, sc, 'ci_papers')}" for s, lab, nd, sc in stats))
             for k, v in rq["tests"].items()]
    desc = ["{}: {}".format(names[k], ", ".join(f"{lab} {diff(v[s], nd, sc, 'ci_fixed_sets')}" for s, lab, nd, sc in stats))
            for k, v in rq["descriptive_contrasts"].items()]
    b = RES["bands"]
    header = (f"Group & Sets & Papers & Acc. & AUROC [CI] & Paper CI & ECE (sets) & "
              f"\\multicolumn{{4}}{{c}}{{error rate by confidence band}} & AUGRC \\\\\n & & & & & & & {' & '.join(b)} & ")
    cap = ("Confidence against correctness on the \\cfNSets{} zero-shot test sets, by the failure cause named in "
           "\\S\\ref{sec:failures} (top) and by outcome (bottom). Sets are weighted equally. Accuracy and error rates "
           "$\\times$100. \\emph{AUROC [CI]}: interval over items of these sets. \\emph{Paper CI}: papers resampled, only "
           "for groups from at least three papers (--: one or two papers, descriptive only). \\emph{ECE}: equal-mass, over "
           "the sets whose decisions have a probability (count in parentheses). \\emph{AUGRC}: area under the generalized "
           "risk-coverage curve (lower is better). Tested differences (point, paper-level 95\\% CI, Holm-corrected $p$ over "
           "\\cfNTestsHolm{} tests): " + "; ".join(tests) + ". Descriptive differences (point, interval over items): "
           + "; ".join(desc) + ".")
    return table("table*", "lrrrlllrrrrr", header, rows, cap, "tab:conf-groups")


def sets_tex() -> str:
    order = {k: i for i, (k, _) in enumerate(CAUSES)}
    sets = sorted(RES["sets"].values(), key=lambda s: (order[s["cause"]], -s["auroc"]))
    rows = []
    prev = None
    for s in sets:
        if prev is not None and s["cause"] != prev:
            rows.append("\\midrule")
        prev = s["cause"]
        top_n = s["bands"][-1]["n"]
        top = "--" if missing(s["top_band_error_rate"]) else f"{pct(s['top_band_error_rate'])} ({thousands(top_n)})"
        ece = "--" if missing(s["ece"]) else f"{num(s['ece'], 3)} {ci(s['ece_ci'], 3)}"
        rows.append(f"{s['name']} & {OUTCOME_SHORT[s['outcome']]} & {CAUSE_SHORT[s['cause']]} & {thousands(s['n_units'])} & "
                    f"{pct(s['accuracy'])} & {num(s['auroc'])} {ci(s['auroc_ci'])} & {ece} & {top} & "
                    f"{num(s['augrc'], 3)} {ci(s['augrc_ci'], 3)} \\\\")
    header = "Set & Outcome & Cause & Units & Acc. & AUROC [CI] & ECE [CI] & Top-band err.\\ ($n$) & AUGRC [CI]"
    cap = ("Confidence against correctness per zero-shot test set, grouped by failure cause (\\S\\ref{sec:failures}, "
           "--: failing, no cause named) and sorted by AUROC. \\emph{ECE}: equal-mass, of the probability of the predicted "
           "option (--: the decision combines several answers). \\emph{Top-band err.}: error rate among units with "
           "confidence in " + RES["bands"][-1] + " ($n$ units, --: fewer than \\cfAppMinBand). \\emph{AUGRC}: area under "
           "the generalized risk-coverage curve. Accuracy and error rates $\\times$100. Units are the set's decisions "
           "(sentences, pairs, candidates). Intervals resample the item that contains them.")
    return table("table*", "lllllllll", header, rows, cap, "tab:conf-sets")


def selective_tex() -> str:
    rq = RES["rq1"]
    rows = []
    for k in rq["oracle_reach_full"] + rq["oracle_reach_partial"] + rq["oracle_never"]:
        s = RES["sets"][k]
        lt, nv = s["ltt"], s["naive_cross_fitted"]
        rows.append(f"{s['name']} & {OUTCOME_SHORT[s['outcome']]} & {pct(s['target_accuracy'], 1)} & {pct(s['acc_cov100'], 1)} & "
                    f"{pct(lt['mean_coverage'])} & {pct(lt['mean_kept_accuracy'], 1)} & {pct(lt['share_reaching'])} & "
                    f"{pct(nv['mean_coverage'])} & {pct(nv['share_reaching'])} & {num(s['max_coverage_at_target'])} \\\\")
        if k in (rq["oracle_reach_full"][-1], rq["oracle_reach_partial"][-1]):
            rows.append("\\midrule")
    dev = RES["sets"]["inli"]["dev_fitted"]
    header = ("Set & Outcome & Paper & \\jev{} & \\multicolumn{3}{c}{Learn-then-Test} & \\multicolumn{2}{c}{naive} & Oracle \\\\\n"
              " & & & & Cov. & Acc. & Reach & Cov. & Reach & $c^*$")
    cap = ("Selective prediction on the \\cfSelNComparable{} sets whose paper reports an accuracy-type number over the "
           "same units (accuracy or P@1). \\emph{Paper}: the paper's method. \\emph{\\jev{}}: accuracy on all units. Over "
           f"{RES['halvings']} random halvings of the items \\citep{{rao2026jev}}, a confidence threshold is chosen on one "
           "half and applied to the other: \\emph{Learn-then-Test} keeps the lowest threshold at which the half's kept "
           f"accuracy is above the paper's number with confidence {1 - RES['ltt_delta']:.2f}, and \\emph{{naive}} keeps the lowest "
           "threshold whose point accuracy reaches it. \\emph{Cov.}: mean share of units kept. \\emph{Acc.}: their accuracy. "
           "\\emph{Reach}: share of halvings whose kept accuracy reaches the paper's number. \\emph{Oracle $c^*$}: the "
           "largest share of units, most confident first, that reaches it on the whole test set, an optimistic bound. "
           "INLI with thresholds fit on its validation split: Learn-then-Test keeps "
           f"{pct(dev['ltt']['coverage'])}\\% of test items at {pct(dev['ltt']['kept_accuracy'], 1)}, naive "
           f"{pct(dev['naive']['coverage'])}\\% at {pct(dev['naive']['kept_accuracy'], 1)}. Values $\\times$100. Across all "
           "\\cfNSets{} sets, keeping the most confident 80\\% (50\\%) adds a median \\cfSelGainEighty{} "
           "(\\cfSelGainFifty) accuracy points.")
    return table("table*", "llrrrrrrrr", header, rows, cap, "tab:conf-selective")


def readout_tex() -> str:
    r = RES["rq3"]
    rows = []

    def three(z: float, ts: float, ro: float, nd: int) -> str:
        return f"{num(z, nd)} / {num(ts, nd)} / {num(ro, nd)}"

    def cells(z: dict, ts: dict, ro: dict, dt: dict) -> str:
        return (f"{three(z['ece'], ts['ece'], ro['ece'], 3)} & {signed(dt['ece']['delta'], 3)} {ci(dt['ece']['ci'], 3)} & "
                f"{three(z['brier'], ts['brier'], ro['brier'], 3)} & {signed(dt['brier']['delta'], 3)} "
                f"{ci(dt['brier']['ci'], 3)} & {three(z['auroc'], ts['auroc'], ro['auroc'], 2)} & "
                f"{signed(dt['auroc']['delta'], 2)} {ci(dt['auroc']['ci'])}")

    for t in r["tasks"].values():
        z, ts, ro = t["zero_shot"], t["temperature_scaled"], t["readout_seed_mean"]
        rows.append(f"{t['title']} & {thousands(t['n_test'])} & {pct(z['accuracy'])} / {pct(ro['accuracy'])} & "
                    f"{cells(z, ts, ro, t['delta_readout_vs_temperature'])} \\\\")
    s = r["summary_8_tasks"]
    keys = ("ece", "brier", "auroc")
    zs = {k: s[k]["zero_shot"] for k in keys}
    tss = {k: s[k]["temperature_scaled"] for k in keys}
    rs = {k: s[k]["readout"] for k in keys}
    dts = {k: {"delta": s[k]["delta_vs_temperature"], "ci": s[k]["ci_vs_temperature"]} for k in keys}
    rows.append("\\midrule")
    rows.append(f"Mean & & {pct(s['accuracy']['zero_shot'])} / {pct(s['accuracy']['readout'])} & {cells(zs, tss, rs, dts)} \\\\")
    header = "Task & $n$ & Acc. & ECE & $\\Delta$ECE [CI] & Brier & $\\Delta$Brier [CI] & AUROC & $\\Delta$AUROC [CI]"
    n_seeds = {len(t["seeds"]) for t in r["tasks"].values()}
    assert len(n_seeds) == 1, n_seeds
    e = s["ece"]
    cap = (f"Calibration on test ({len(r['tasks'])} tasks): the paper's question zero-shot / the same answers with one "
           "temperature fit on the fit-slice items of the optimized questions / the logistic regression on the answers of the optimized questions (mean over "
           f"{n_seeds.pop()} optimizer seeds). Accuracy: zero-shot / optimized. Confidence is the probability of the predicted "
           "label, and ECE is equal-mass. $\\Delta$: optimized $-$ temperature-scaled, paired bootstrap 95\\% CI. Against the "
           f"zero-shot question the logistic regression lowers ECE by {num(-e['delta'], 3)}. A temperature and class biases fit "
           f"together reach ECE {num(e['bias_temperature'], 3)}.")
    return table("table*", "lrlllllll", header, rows, cap, "tab:conf-readout")


def rerun_tex() -> str:
    r = RES["rerun"]
    rows = [f"{{{b['band']}}} & {thousands(b['n'])} & {b['flips']} & {pct(b['flip_rate'], 1)} \\\\" for b in r["bands"]]
    rows.append("\\midrule")
    rows.append(f"All & {thousands(r['n'])} & {r['flips']} & {pct(r['flip_rate'], 1)} \\\\")
    tasks = ", ".join(f"{RES['sets'][k]['name'] if k in RES['sets'] else k} {v['flips']}/{thousands(v['n'])}"
                      for k, v in r["by_task"].items())
    cap = (f"Label flips among {thousands(r['n'])} byte-identical zero-shot test requests sent again, by the first "
           f"run's confidence. Flip rate in \\%. {r['flips_conf_lt_06']} of the {r['flips']} flips "
           f"({pct(r['share_flips_conf_lt_06'], 1)}\\%) had confidence below 0.6, which "
           f"{pct(r['share_requests_conf_lt_06'], 1)}\\% of requests have. Low confidence predicts a flip with AUROC "
           f"{num(r['auroc_low_conf_for_flip'])}. Flips per task: {tasks}.")
    return table("table", "lrrr", "First-run confidence & Requests & Flips & Rate", rows, cap, "tab:conf-rerun",
                 size="\\footnotesize")


def main() -> None:
    parts = ["% Generated by scripts/paper_appendix_confidence.py from results/confidence.json. Do not edit.",
             macros(), groups_tex(), sets_tex(), selective_tex(), readout_tex(), rerun_tex()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
