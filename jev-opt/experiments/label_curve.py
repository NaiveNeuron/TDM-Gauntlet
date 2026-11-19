"""Label curve of the RQ3 pipeline with the questions fixed (pre-registered) -> results/LABEL_CURVE.{json,md},
<tex-dir>/label_curve.tex (macros \\lc...), <tex-dir>/label_curve_table.tex (tabular).

Offline and cache-only: Jev's answers come from `runs/jev-opt/.jev-cache` with posting disabled (importing
`framing_features` via `clef_rq3`; a cache miss stops the run), test answers are only re-read. Per RQ3 set (Jev's
seed-0 run of `clef_rq3.SETS`) and budget n in {50, 100, 250, 400}, R = 10 label-stratified subsamples of the
400-item fit slice (n = 400: the fit slice itself), shared by three steps:
  cal   per-class bias + temperature (`fit_label_bias`) on the paper's question, applied to the frozen zero-shot
        test distributions (as `analyze_frozen.calibration` / `cal_refit_spread.jev_inputs`);
  opt   Jev's chosen representative (P or P+head, `test_consumed.json`), readouts reset and refitted with
        `fit_program` exactly as `framing_features.refit`, re-executed on test from the cache;
  reph  the rephrased question (`top-zero-jev-native.yaml`) with `with_readouts`, fitted and applied the same way
        (8 configurations: the frozen sets have no cached test answers for it).
At n = 400 the calibrated and optimized test scores must equal results/frozen.json.

  uv run --project jev-opt/python python jev-opt/experiments/label_curve.py [--tex-dir DIR] [--report-only] [--jobs N]
  uv run --project jev-opt/python python jev-opt/experiments/label_curve.py --cell <stem>
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clef_rq3 as C  # noqa: E402  (imports framing_features, which disables Jev posting)
import framing_features as ff  # noqa: E402
from analyze_frozen import gauntlet_zero  # noqa: E402
from cal_refit_spread import CFG8, TITLE  # noqa: E402
from confidence_readout import zero_shot as jev_zero_shot  # noqa: E402

from jev_opt.adapter import cv_spec  # noqa: E402
from jev_opt.execute import combine  # noqa: E402
from jev_opt.fit import fit_label_bias, fit_program, with_readouts  # noqa: E402
from jev_opt.ir import load_decider  # noqa: E402
from jev_opt.jev import make_client  # noqa: E402

ROOT = HERE.parents[1]
CELLS = ROOT / "results" / "label_curve"
OUT_JSON = ROOT / "results" / "LABEL_CURVE.json"
OUT_MD = ROOT / "results" / "LABEL_CURVE.md"
TEX_DIR = ROOT / "paper" / "generated"
BUDGETS = (50, 100, 250, 400)
FULL = 400
R = 10
SEED = 0
NEAR = 0.01  # "within 1 point": mean at the budget >= full-budget score - 1 point
STEPS = ("cal", "reph", "opt")
WORD = {50: "Fifty", 100: "Hundred", 250: "TwoFifty", 400: "Full"}
STEP_TAG = {"cal": "Cal", "reph": "Reph", "opt": "Opt"}


def subsample(gold: list[str], labels: list[str], n: int, r: int) -> list[int]:
    """Indices into the fit slice (ascending): label-stratified, proportional by largest remainder, >= 1 per class."""
    if n >= len(gold):
        return list(range(len(gold)))
    rng = np.random.default_rng([SEED, n, r])
    classes = [lab for lab in labels if lab in set(gold)]
    members = {lab: [i for i, g in enumerate(gold) if g == lab] for lab in classes}
    quota = {lab: n * len(members[lab]) / len(gold) for lab in classes}
    alloc = {lab: int(np.floor(q)) for lab, q in quota.items()}
    for lab in sorted(classes, key=lambda c: (-(quota[c] - alloc[c]), classes.index(c)))[: n - sum(alloc.values())]:
        alloc[lab] += 1
    for lab in classes:
        if alloc[lab] == 0:
            alloc[lab] = 1
            big = max(classes, key=lambda c: (alloc[c], -classes.index(c)))
            alloc[big] -= 1
    assert sum(alloc.values()) == n and all(alloc[c] <= len(members[c]) for c in classes)
    out: list[int] = []
    for lab in classes:
        out += [int(i) for i in rng.choice(members[lab], size=alloc[lab], replace=False)]
    return sorted(out)


def fit_and_score(env: C.Env, blank: Any, pairs: list, idx: list[int], gold: list[str]) -> float:
    """framing_features.refit on the subsample's exhaustive fit-slice answers, then the fitted program on test."""
    opt = env.config.optimizer
    fitted, _ = fit_program(blank, [pairs[i] for i in idx], env.config.primary_metric, opt.calibrate,
                            rounds=2 if opt.fit_parameters else 0, negative=env.config.negative_label,
                            class_bias=opt.class_bias, cv=cv_spec(opt))
    fitted = fitted.check(env.config)
    res = env.run(fitted, env.test, "test")
    return env.primary(gold, [r.pred for r in res])


def cell(stem: str) -> dict[str, Any]:
    os.environ.pop("JEV_OPT_BACKEND", None)
    os.environ.pop("JEV_OPT_DEVICE", None)
    frozen = {r["stem"]: r for r in (lambda d: d["new"] + d["orig"])(json.loads((ROOT / "results" / "frozen.json").read_text()))}[stem]
    src = C.RUNS / C.SETS[stem][0]
    cfg = C.run_config(src)
    client = make_client(C.RUNS / ".jev-cache", cfg.jev.model, cfg.jev.max_rpm, cfg.jev.concurrency, [ROOT])
    try:
        env = C.Env(src, client, test=True)
        labels = env.labels
        fit_gold = [e.label for e in env.fit]
        test_ids, gold = [e.id for e in env.test], [e.label for e in env.test]
        consumed = json.loads((src / "test_consumed.json").read_text())
        rep = consumed["representative"]

        # calibration inputs: the paper's question on the fit slice, the frozen zero-shot test distributions
        seed_fit = env.run(load_decider((src / "seed.yaml").read_text()), env.fit, "train")
        seed_d = [C.floored(r.dist, labels) for r in seed_fit]
        zs = jev_zero_shot(stem, labels) if stem in CFG8 else gauntlet_zero(C.SETS[stem][1], labels)
        z = [zs[i] for i in test_ids]
        assert [x["gold"] for x in z] == gold, f"{stem}: zero-shot gold differs from the test split"
        test_d = [dict(zip(labels, x["dist"])) for x in z]

        def cal_score(idx: list[int]) -> float:
            bias, temp, _ = fit_label_bias(labels, [[seed_d[i]] for i in idx], [fit_gold[i] for i in idx],
                                           cfg.primary_metric, cfg.negative_label)
            return env.primary(gold, [combine(labels, [C.floored(d, labels)], bias, temp)[1] for d in test_d])

        # optimized: Jev's chosen representative, readouts reset
        opt_prog = load_decider((src / f"{rep}.yaml").read_text())
        opt_blank = ff.unfitted(opt_prog, drop_aux=False)
        opt_pairs = list(zip(env.fit, env.run(opt_blank, env.fit, "train", exhaustive=True)))
        # rephrased question + readout: test answers exist only where v3-<set>-N was tested
        n_run = C.RUNS / f"v3-{stem}-N"
        reph_blank = reph_pairs = None
        reph_zero = None
        if (n_run / "test_consumed.json").exists():
            nat_text = (src / "top-zero-jev-native.yaml").read_text()
            assert load_decider(nat_text).digest() == load_decider((n_run / "seed.yaml").read_text()).digest(), stem
            reph_blank = with_readouts(load_decider(nat_text), cfg.primary_metric)
            reph_pairs = list(zip(env.fit, env.run(reph_blank, env.fit, "train", exhaustive=True)))
            reph_zero = json.loads((n_run / "test_consumed.json").read_text())["metrics"]["primary"]

        budgets: dict[str, Any] = {}
        for n in BUDGETS:
            row: dict[str, Any] = {s: [] for s in STEPS} | {"failures": {s: 0 for s in STEPS}, "class_counts": []}
            for r in range(1 if n >= FULL else R):
                idx = subsample(fit_gold, labels, n, r)
                row["class_counts"].append({lab: sum(fit_gold[i] == lab for i in idx) for lab in labels})
                row["cal"].append(cal_score(idx))
                for step, blank, pairs in (("opt", opt_blank, opt_pairs), ("reph", reph_blank, reph_pairs)):
                    if blank is None:
                        continue
                    try:
                        row[step].append(fit_and_score(env, blank, pairs, idx, gold))
                    except ValueError as exc:  # e.g. a readout whose subsample holds a single class
                        row["failures"][step] += 1
                        row.setdefault("errors", []).append(f"{step} r={r}: {str(exc)[:200]}")
                print(f"{stem} n={n} r={r} cal={row['cal'][-1]:.4f} opt={row['opt'][-1] if row['opt'] else None} "
                      f"reph={row['reph'][-1] if row['reph'] else None}", flush=True)
            budgets[str(n)] = row
        full = budgets[str(FULL)]
        if abs(full["cal"][0] - frozen["calibrated"]) > 1e-9 or abs(full["opt"][0] - frozen["optimized"]) > 1e-9:
            raise SystemExit(f"{stem}: full-budget replay differs: cal {full['cal'][0]} vs {frozen['calibrated']}, "
                             f"opt {full['opt'][0]} vs {frozen['optimized']}")
        return {"set": stem, "run": src.name, "representative": rep, "metric": cfg.primary_metric, "labels": labels,
                "n_fit": len(env.fit), "n_test": len(gold), "demos": opt_prog.demos is not None,
                "zero": frozen["zero"], "paper": frozen["paper"], "recorded": {"cal": frozen["calibrated"], "opt": frozen["optimized"]},
                "reph_zero_shot_test": reph_zero, "fresh_calls": client.usage.fresh_calls, "budgets": budgets}
    finally:
        client.close()


def compute(jobs: int) -> None:
    CELLS.mkdir(parents=True, exist_ok=True)
    todo = [s for s in C.SETS if not (CELLS / f"{s}.json").exists()]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    failed: list[str] = []
    while todo:
        batch, todo = todo[:jobs], todo[jobs:]
        procs = {s: subprocess.Popen([sys.executable, __file__, "--cell", s], env=env) for s in batch}
        failed += [s for s, p in procs.items() if p.wait() != 0]
    if failed:
        raise SystemExit(f"cells failed: {failed}")


def gap(x: float, zero: float, paper: float) -> float:
    return (x - zero) / (paper - zero)


def summarize() -> dict[str, Any]:
    bb = json.loads((ROOT / "results" / "BUDGET_BASELINE.json").read_text())
    cells = {s: json.loads((CELLS / f"{s}.json").read_text()) for s in C.SETS}
    sets: dict[str, Any] = {}
    for s, c in cells.items():
        b = bb["sets"][s]
        row: dict[str, Any] = {"title": TITLE[s], "metric": c["metric"], "representative": c["representative"], "demos": c["demos"],
                               "zero": c["zero"], "paper": c["paper"], "bb_zero": b["zero"], "bb_paper": b["paper"],
                               "reph_zero_shot_test": c["reph_zero_shot_test"], "encoder_best": b["best"],
                               "encoder_test": b["best_test_mean"], "encoder_gap": gap(b["best_test_mean"], c["zero"], c["paper"]),
                               "steps": {}}
        for step in STEPS:
            full = c["budgets"][str(FULL)][step]
            if not full:
                continue
            per = {}
            for n in BUDGETS:
                v = c["budgets"][str(n)][step]
                fails = c["budgets"][str(n)]["failures"][step]
                if not v:  # every subsample's fit raised: no score at this budget
                    per[str(n)] = {"scores": [], "mean": None, "sd": None, "n_ok": 0, "failures": fails, "gap": None,
                                   "gap_bb": None, "drop": None, "near_full": False}
                    continue
                m = statistics.mean(v)
                per[str(n)] = {"scores": v, "mean": m, "sd": statistics.stdev(v) if len(v) > 1 else 0.0, "n_ok": len(v),
                               "failures": fails, "gap": gap(m, c["zero"], c["paper"]),
                               "gap_bb": gap(m, b["zero"], b["paper"]), "drop": full[0] - m, "near_full": m >= full[0] - NEAR}
            row["steps"][step] = per
        sets[s] = row
    groups = {"all": list(C.SETS), "cfg8": CFG8, "no_demos": [s for s in C.SETS if not sets[s]["demos"]]}
    summ: dict[str, Any] = {}
    for step in STEPS:
        for g, members in groups.items():
            have = [s for s in members if step in sets[s]["steps"]]
            if not have:
                continue
            key = f"{step}_{g}"
            summ[key] = {"n_sets": len(have), "sets": have}
            for n in BUDGETS:
                rows = [sets[s]["steps"][step][str(n)] for s in have]
                ok = [r for r in rows if r["gap"] is not None]
                summ[key][str(n)] = {"n_scored": len(ok), "unscored": [s for s, r in zip(have, rows) if r["gap"] is None],
                                     "median_gap": statistics.median(r["gap"] for r in ok),
                                     # sensitivity: a set without a fit falls back to zero-shot (gap 0)
                                     "median_gap_fallback_zero": statistics.median(0.0 if r["gap"] is None else r["gap"] for r in rows),
                                     "median_gap_bb": statistics.median(r["gap_bb"] for r in ok),
                                     "median_drop": statistics.median(r["drop"] for r in ok),
                                     "n_near_full": sum(r["near_full"] for r in rows),
                                     "failures": sum(r["failures"] for r in rows)}
    summ["encoder"] = {"budget": 500, "median_gap": statistics.median(sets[s]["encoder_gap"] for s in C.SETS),
                       "median_gap_bb": bb["summary"]["median_gap_share_best"]}
    summ["fresh_calls"] = sum(c["fresh_calls"] for c in cells.values())
    return {"generated_by": "jev-opt/experiments/label_curve.py",
            "budgets": list(BUDGETS), "R": R, "seed": SEED, "near_points": 100 * NEAR, "sets": sets, "summary": summ}


def pct(x: float) -> str:
    return f"{100 * x:.0f}"


def pt(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def write_md(rep: dict[str, Any]) -> None:
    S, M = rep["sets"], rep["summary"]
    L = ["# Label curve of the RQ3 pipeline, questions fixed", "",
         f"Offline, cache-only (fresh Jev calls: {M['fresh_calls']}). Test primary metric ×100; budgets = labels from the "
         f"400-item fit slice; n < 400: mean over R = {R} label-stratified subsamples (`default_rng([0, n, r])`), same "
         "subsample for every step. Cal = per-class bias + temperature on the paper's question; Reph = rephrased question "
         "(`top-zero-jev-native`) + readout (8 configurations); Opt = Jev's chosen P / P+head (seed 0) with readouts refitted. "
         "Gap closed = (score − zero-shot) / (paper − zero-shot), `results/frozen.json` reference; near = sets whose mean "
         "is at least the 400-label score − 1 point.", "",
         "## Median gap closed (%) and sets within 1 point of the 400-label score", "",
         "| Labels | Cal (13) | Reph (8) | Opt (13) | Cal (8) | Opt (8) | Opt, no demos (" + str(M["opt_no_demos"]["n_sets"]) + ") | near Cal / Reph / Opt |",
         "|---|---|---|---|---|---|---|---|"]
    for n in BUDGETS:
        k = str(n)
        L.append(f"| {n} | {pct(M['cal_all'][k]['median_gap'])} | {pct(M['reph_cfg8'][k]['median_gap'])} | {pct(M['opt_all'][k]['median_gap'])} | "
                 f"{pct(M['cal_cfg8'][k]['median_gap'])} | {pct(M['opt_cfg8'][k]['median_gap'])} | {pct(M['opt_no_demos'][k]['median_gap'])} | "
                 f"{M['cal_all'][k]['n_near_full']}/13, {M['reph_cfg8'][k]['n_near_full']}/8, {M['opt_all'][k]['n_near_full']}/13 |")
    L += [f"| 500, same-label encoder (best by val) | | | {pct(M['encoder']['median_gap'])} | | | | |", "",
          f"Encoder: BUDGET_BASELINE.json best family, seed mean, gap against the frozen.json reference; with "
          f"BUDGET_BASELINE's own reference {pct(M['encoder']['median_gap_bb'])}% (`\\bbGapShareBest`).", "",
          "Median drop from the 400-label score (points): " + "; ".join(
              f"{STEP_TAG[st]} " + ", ".join(f"{n}: {pt(M[f'{st}_{g}'][str(n)]['median_drop'])}" for n in BUDGETS[:-1])
              for st, g in (("cal", "all"), ("reph", "cfg8"), ("opt", "all"))) + ".", "",
          "Fit failures (subsamples excluded): " + ", ".join(f"{st} {sum(M[f'{st}_{g}'][str(n)]['failures'] for n in BUDGETS)}"
                                                          for st, g in (("cal", "all"), ("reph", "cfg8"), ("opt", "all"))) + ". "
          "Sets without any successful fit at a budget (excluded from that median, counted as not near): " + "; ".join(
              f"{STEP_TAG[st]} {n}: {', '.join(S[s]['title'] for s in M[f'{st}_{g}'][str(n)]['unscored'])} (median with gap 0 for them: "
              f"{pct(M[f'{st}_{g}'][str(n)]['median_gap_fallback_zero'])}%)"
              for st, g in (("cal", "all"), ("reph", "cfg8"), ("opt", "all")) for n in BUDGETS if M[f'{st}_{g}'][str(n)]['unscored'])
          + ".", "",
          "Sensitivity, zero-shot/paper of BUDGET_BASELINE.json: " + "; ".join(
              f"{STEP_TAG[st]} " + ", ".join(f"{n}: {pct(M[f'{st}_{g}'][str(n)]['median_gap_bb'])}%" for n in BUDGETS)
              for st, g in (("cal", "all"), ("reph", "cfg8"), ("opt", "all"))) + ".", "",
          "## Per set (mean ± sd over subsamples, ×100)", "",
          "| Set | Metric | Rep. | Demos | Zero | Paper | Step | " + " | ".join(str(n) for n in BUDGETS) + " | Gap 50 / 400 |",
          "|---|---|---|---|---|---|---|" + "---|" * len(BUDGETS) + "---|"]
    for s, r in S.items():
        for step in STEPS:
            if step not in r["steps"]:
                continue
            p = r["steps"][step]
            cells = " | ".join(f"{pt(p[str(n)]['mean'])}" + (f" ± {pt(p[str(n)]['sd'])}" if n < FULL else "")
                               + (f" ({p[str(n)]['failures']} failed)" if p[str(n)]['failures'] else "") for n in BUDGETS)
            L.append(f"| {r['title']} | {r['metric']} | {r['representative']} | {'yes' if r['demos'] else 'no'} | {pt(r['zero'])} | "
                     f"{pt(r['paper'])} | {STEP_TAG[step]} | {cells} | {pt(p['50']['gap'])}% / {pt(p[str(FULL)]['gap'])}% |")
    L += ["", "Rephrased question zero-shot test (v3-<set>-N, recorded): " + ", ".join(
        f"{S[s]['title']} {pt(S[s]['reph_zero_shot_test'])}" for s in CFG8) + ".", ""]
    OUT_MD.write_text("\n".join(L))


def write_tex(rep: dict[str, Any], tex_dir: Path) -> None:
    M = rep["summary"]
    m: list[tuple[str, str]] = [("lcNSets", str(M["opt_all"]["n_sets"])), ("lcNSetsReph", str(M["reph_cfg8"]["n_sets"])),
                                ("lcNSetsNoDemo", str(M["opt_no_demos"]["n_sets"])), ("lcR", str(R)), ("lcFullBudget", str(FULL)),
                                ("lcEncBudget", str(M["encoder"]["budget"])), ("lcEncGap", pct(M["encoder"]["median_gap"]))]
    for step, g in (("cal", "all"), ("reph", "cfg8"), ("opt", "all")):
        t = STEP_TAG[step]
        for n in BUDGETS:
            x = M[f"{step}_{g}"][str(n)]
            m += [(f"lc{t}Gap{WORD[n]}", pct(x["median_gap"]))]
            if n < FULL:
                m += [(f"lc{t}Near{WORD[n]}", str(x["n_near_full"])), (f"lc{t}Drop{WORD[n]}", pt(x["median_drop"]))]
            if x["unscored"]:
                m += [(f"lc{t}NScored{WORD[n]}", str(x["n_scored"])),
                      (f"lc{t}GapZeroFallback{WORD[n]}", pct(x["median_gap_fallback_zero"]))]
    for step in ("cal", "opt"):
        for n in BUDGETS:
            m += [(f"lc{STEP_TAG[step]}GapCfg{WORD[n]}", pct(M[f"{step}_cfg8"][str(n)]["median_gap"]))]
    for n in BUDGETS:
        m += [(f"lcOptGapNoDemo{WORD[n]}", pct(M["opt_no_demos"][str(n)]["median_gap"]))]
    m += [("lcFailures", str(sum(M[f"{st}_{g}"][str(n)]["failures"] for st, g in (("cal", "all"), ("reph", "cfg8"), ("opt", "all"))
                                 for n in BUDGETS)))]
    tex_dir.mkdir(parents=True, exist_ok=True)
    (tex_dir / "label_curve.tex").write_text("% generated by jev-opt/experiments/label_curve.py; do not edit\n"
                                             + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m))
    L = ["% generated by jev-opt/experiments/label_curve.py; do not edit",
         "\\begin{tabular}{@{}ccccc@{}}", "\\toprule",
         "& \\multicolumn{4}{c}{Median gap closed (\\%)} \\\\", "\\cmidrule(l){2-5}",
         "& & \\multicolumn{1}{c}{Rephrased} & \\multicolumn{2}{c}{Optimized q.\\ + LR} \\\\", "\\cmidrule(l){4-5}",
         "\\multicolumn{1}{c}{\\multirow{-3}{*}[1ex]{Labels}} & \\multicolumn{1}{c}{\\multirow{-2}{*}[0.5ex]{Calibration}} & \\multicolumn{1}{c}{q.\\ + LR$^\\dagger$} & \\multicolumn{1}{c}{all} & \\multicolumn{1}{c}{no demos$^\\S$} \\\\", "\\midrule"]
    cols = (("cal", "all"), ("reph", "cfg8"), ("opt", "all"), ("opt", "no_demos"))
    unscored = sorted({(rep["sets"][s]["title"], n) for n in BUDGETS for st, g in cols
                       for s in M[f"{st}_{g}"][str(n)].get("unscored", [])})
    notes = [f"without {t}, where no logistic regression could be fitted with {n} labels" for t, n in unscored]

    def cell_tex(st: str, g: str, n: int) -> str:
        x = M[f"{st}_{g}"][str(n)]
        return pct(x["median_gap"]).replace("-", "$-$") + ("$^\\ddagger$" if x.get("unscored") else "")

    for n in BUDGETS:
        L.append(f"{n} & " + " & ".join(cell_tex(st, g, n) for st, g in cols) + " \\\\")
    L += ["\\midrule",
          f"\\multicolumn{{3}}{{@{{}}l}}{{Same-label encoder, {M['encoder']['budget']} labels}} & {pct(M['encoder']['median_gap'])} & \\\\",
          "\\bottomrule", "\\end{tabular}"]
    note = (f"$^\\dagger$~{M['reph_cfg8']['n_sets']} of the {M['opt_all']['n_sets']} configurations; $^\\S$~the "
            f"{M['opt_no_demos']['n_sets']} sets without labelled demonstrations"
            + "".join(f"; $^\\ddagger$~{t}" for t in notes) + ".")
    (tex_dir / "label_curve_table.tex").write_text("\n".join(L) + "\n")
    with (tex_dir / "label_curve.tex").open("a") as fh:  # caption text for the table's marks
        fh.write(f"\\newcommand{{\\lcTableNote}}{{{note}}}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cell", help="compute one set (subprocess)")
    ap.add_argument("--report-only", action="store_true", help="summarize existing cells")
    ap.add_argument("--jobs", type=int, default=13)
    ap.add_argument("--tex-dir", type=Path, default=TEX_DIR)
    a = ap.parse_args()
    if a.cell:
        r = cell(a.cell)
        CELLS.mkdir(parents=True, exist_ok=True)
        (CELLS / f"{a.cell}.json").write_text(json.dumps(r, indent=1) + "\n")
        return 0
    if not a.report_only:
        compute(a.jobs)
    rep = summarize()
    OUT_JSON.write_text(json.dumps(rep, indent=1) + "\n")
    write_md(rep)
    write_tex(rep, a.tex_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
