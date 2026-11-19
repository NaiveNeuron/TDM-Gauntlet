"""Seed variance of the headline configuration (pre-registered) -> results/seeds.json, results/SEEDS.md.

  --select  validation only: per task and seed, top-single vs top-stacked by the pre-registered rule
            (P+head replaces P iff its val primary metric is >= 1.0 point higher); writes
            results/seeds.json["selection"], which `run_seeds.sh test` reads. Opens no test file.
  (default) also reads each chosen representative's test score recorded at consumption
            (`test_consumed.json`) and writes results/SEEDS.md.

Seed 0 = the v3 runs `v3-<task>-P` (third optimization study); seeds 1, 2 = `v3-<task>-P-s<seed>`. Val scores are
recomputed from `val_predictions.json` exactly as analyze_v3.py does. No Jev or LLM call is made.

Run from the repo root:
  uv run --project jev-opt/python python jev-opt/experiments/analyze_seeds.py [--select]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from analyze_v2 import TASKS, jev_cost, omp_cost, run_config, score_fn  # noqa: E402
from analyze_v3 import fresh  # noqa: E402

ROOT = Path.cwd()
RUNS = ROOT / "runs" / "jev-opt"
OUT_JSON = ROOT / "results" / "seeds.json"
OUT_MD = ROOT / "results" / "SEEDS.md"
SEEDS = (0, 1, 2)
MARGIN = 1.0
REPS = {"top-single": "P: population search + readout", "top-stacked": "P+head: stacked readout"}


def run_dir(stem: str, seed: int) -> Path:
    return RUNS / (f"v3-{stem}-P" if seed == 0 else f"v3-{stem}-P-s{seed}")


def finished(d: Path) -> bool:
    try:
        return json.loads((d / "run.json").read_text()).get("state") == "finished"
    except FileNotFoundError:
        return False


def val_scores(d: Path) -> dict[str, float]:
    data = json.loads((d / "val_predictions.json").read_text())
    score = score_fn(run_config(d))
    ids = sorted(data["gold"])
    gold = [data["gold"][i] for i in ids]
    return {rep: score(gold, [data["representatives"][rep][i] for i in ids]) for rep in REPS if rep in data["representatives"]}


def choose(val: dict[str, float]) -> str:
    chosen = "top-single"
    if "top-stacked" in val and 100 * (val["top-stacked"] - val[chosen]) >= MARGIN:
        chosen = "top-stacked"
    return chosen


def candidate(d: Path, rep: str) -> str:
    return json.loads((d / "pareto.json").read_text())["representatives"][rep].get("candidate", "")


def select(prev: dict[str, Any]) -> dict[str, Any]:
    sel: dict[str, Any] = {}
    for stem, _, _ in TASKS:
        sel[stem] = {}
        for seed in SEEDS:
            d = run_dir(stem, seed)
            if not finished(d):
                sel[stem][str(seed)] = {"run": d.name, "status": "not finished"}
                continue
            val = val_scores(d)
            rep = choose(val)
            old = prev.get(stem, {}).get(str(seed), {})
            if old.get("representative") not in (None, rep):
                raise SystemExit(f"{d.name}: selection changed from {old['representative']} to {rep}; the rule is fixed")
            f_cost, f_calls, f_tok = fresh(d)
            sel[stem][str(seed)] = {
                "run": d.name, "status": "finished", "val": val, "representative": rep, "candidate": candidate(d, rep),
                # recorded once, before the test evaluation: the difference afterwards is the test spend
                "jev_used_before_test": old.get("jev_used_before_test", jev_cost(d)) if seed else None,
                "jev_fresh": f_cost, "jev_fresh_requests": f_calls, "jev_fresh_input_tokens": f_tok, "omp": omp_cost(d),
            }
    return sel


def sd(xs: list[float]) -> float | None:
    return statistics.stdev(xs) if len(xs) > 1 else None


def pm(xs: list[float]) -> str:
    if not xs:
        return ""
    s = sd(xs)
    return f"{100 * statistics.mean(xs):.1f} ± {100 * s:.1f}" if s is not None else f"{100 * xs[0]:.1f}"


def vs_zero(stem: str, per: dict[str, Any]) -> dict[str, Any]:
    """Paired bootstrap of each seed's test score against the v1 zero-shot run (the audit question): every run
    re-executed offline from the Jev cache, one set of resamples per task (2,000; seed 0), drawn in the unit the RQ1
    statistics use for the set (resample.py: document, premise, tweet or writing prompt, else item), 95% percentile
    interval of the difference. With fewer than 30 clusters, also a leave-one-cluster-out jackknife-t interval.
    Adds `delta_vs_zero`, `ci_vs_zero`, `cluster_unit`, `n_clusters` (and `ci_vs_zero_jackknife`) to each seed."""
    import numpy as np
    from analyze_test_v3 import B, DISCREPANCIES, evaluate, primary_on  # offline only: posting disabled on import
    from resample import cluster_index, jackknife_t, resamples

    zero = evaluate(RUNS / f"study-{stem}-zero")
    runs = {seed: evaluate(RUNS / p["run"]) for seed, p in per.items()}
    if DISCREPANCIES:
        raise SystemExit(f"re-executed scores differ from the recorded ones: {DISCREPANCIES}")
    n = len(zero["gold"])
    inv, unit = cluster_index(stem, zero["ids"])
    m = int(inv.max()) + 1
    boots = resamples(inv, B, 0)
    base = np.array([primary_on(b, zero) for b in boots])
    for seed, r in runs.items():
        assert r["ids"] == zero["ids"] and r["gold"] == zero["gold"], f"{r['rep']}: test set differs from the zero-shot run"
        lo, hi = np.percentile(np.array([primary_on(b, r) for b in boots]) - base, [2.5, 97.5])
        per[seed] |= {"delta_vs_zero": r["primary"] - zero["primary"], "ci_vs_zero": [float(lo), float(hi)],
                      "cluster_unit": unit, "n_clusters": m}
        if unit != "item" and m < 30:
            per[seed]["ci_vs_zero_jackknife"] = list(jackknife_t(lambda ix: primary_on(ix, r) - primary_on(ix, zero), inv))
    return {"run": f"study-{stem}-zero", "test": zero["primary"], "n": n, "cluster_unit": unit, "n_clusters": m}


def report(sel: dict[str, Any]) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    L = ["# Seed variance of the headline configuration", "",
         "The study was pre-registered. Generated by `jev-opt/experiments/analyze_seeds.py`.",
         "Headline configuration = population search with a fitted readout (`jev-population-opt`, v3 task files), rerun",
         "with optimizer seeds 1 and 2 (seed 0 = the pre-registered v3 runs). The seed changes the 400-example train fit",
         "slice, the readout CV folds and the search's sampling. Per run, val picks P (`top-single`) unless P+head",
         "(`top-stacked`) is ≥ 1.0 point higher; that one representative is evaluated once on test. Values ×100;",
         "mean ± sample sd (n − 1) over the seeds.", "",
         "| Task | Metric | Seed | Chosen | Val P | Val P+head | Val chosen | Test |", "|---|---|---:|---|---:|---:|---:|---:|"]
    summary_rows = ["| Task | Metric | n seeds | Val mean ± sd | Test mean ± sd | Test range |", "|---|---|---:|---:|---:|---:|"]
    cost_rows = ["| Run | Jev fresh $ (val search) | fresh requests | OMP $ | Jev $ test |", "|---|---:|---:|---:|---:|"]
    totals = {"jev_fresh": 0.0, "omp": 0.0, "jev_test": 0.0}
    ci_rows = ["| Task | Seed | Test | Δ audit question (95% CI) | Jev calls/ex | input tok/ex | Jev $ test |", "|---|---:|---:|---|---:|---:|---:|"]
    for stem, title, metric in TASKS:
        per: dict[str, Any] = {}
        for seed in SEEDS:
            s = sel[stem][str(seed)]
            d = run_dir(stem, seed)
            if s["status"] != "finished":
                per[str(seed)] = {"run": s["run"], "status": s["status"]}
                L.append(f"| {title} | {metric} | {seed} | failed: {s['status']} | | | | |")
                continue
            rep = s["representative"]
            tc_path = d / "test_consumed.json"
            test = test_cost = tc = None
            if tc_path.exists():
                tc = json.loads(tc_path.read_text())
                if tc["representative"] != rep:
                    raise SystemExit(f"{d.name}: test consumed by {tc['representative']}, selection is {rep}")
                test = float(tc["metrics"]["primary"])
                if seed:
                    test_cost = jev_cost(d) - s["jev_used_before_test"]
            rec = (tc or {}).get("metrics", {})
            per[str(seed)] = {"run": d.name, "representative": rep, "candidate": s["candidate"], "val": s["val"][rep],
                              "val_all": s["val"], "test": test, "jev_fresh": s["jev_fresh"], "omp": s["omp"], "jev_test": test_cost,
                              "calls": rec.get("mean_calls"), "tokens": rec.get("mean_input_tokens"), "test_cost": rec.get("total_cost")}
            v = s["val"]
            stacked = f"{100 * v['top-stacked']:.1f}" if "top-stacked" in v else ""
            L.append(f"| {title} | {metric} | {seed} | {REPS[rep].split(':')[0]} ({s['candidate']}) | {100 * v['top-single']:.1f} | "
                     f"{stacked} | {100 * v[rep]:.1f} | {'' if test is None else f'{100 * test:.1f}'} |")
            if seed:
                cost_rows.append(f"| `{d.name}` | {s['jev_fresh']:.2f} | {s['jev_fresh_requests']:,} | {s['omp']:.2f} | {'' if test_cost is None else f'{test_cost:.2f}'} |")
                totals["jev_fresh"] += s["jev_fresh"]
                totals["omp"] += s["omp"]
                totals["jev_test"] += test_cost or 0.0
        zero = vs_zero(stem, per) if all(p.get("test") is not None for p in per.values()) else None
        if zero is not None:
            for seed in SEEDS:
                p = per[str(seed)]
                jk = p.get("ci_vs_zero_jackknife")
                jk_txt = f"; jackknife-t {100 * jk[0]:+.1f}…{100 * jk[1]:+.1f}" if jk else ""
                ci_rows.append(f"| {title} | {seed} | {100 * p['test']:.1f} | {100 * p['delta_vs_zero']:+.1f} "
                               f"({100 * p['ci_vs_zero'][0]:+.1f}…{100 * p['ci_vs_zero'][1]:+.1f}{jk_txt}) | {p['calls']:.2f} | {p['tokens']:.0f} | {p['test_cost']:.2f} |")
        vals = [p["val"] for p in per.values() if "val" in p]
        tests = [p["test"] for p in per.values() if p.get("test") is not None]
        tasks[stem] = {"title": title, "metric": metric, "seeds": per,
                       "val_mean": statistics.mean(vals) if vals else None, "val_sd": sd(vals),
                       "test_mean": statistics.mean(tests) if tests else None, "test_sd": sd(tests), "n_test": len(tests),
                       "zero_shot_run": None if zero is None else zero["run"], "zero_shot_test": None if zero is None else zero["test"],
                       "n_test_examples": None if zero is None else zero["n"]}
        rng = f"{100 * min(tests):.1f}–{100 * max(tests):.1f}" if tests else ""
        summary_rows.append(f"| {title} | {metric} | {len(tests)} | {pm(vals)} | {pm(tests)} | {rng} |")
    L += ["", "## Mean ± sd over seeds", "", *summary_rows, "",
          "## Test vs the audit question (v1 zero-shot run on the same test examples)", "",
          "Δ = test minus the audit question's test score; 95% percentile CI of the paired bootstrap in the RQ1 resampling unit",
          "(TB-Dense, edit intent: document; INLI: premise; EZ-STANCE: tweet; DREsS: writing prompt; 2,000 resamples, seed 0; the same resamples and method as `analyze_test_v3.py`, so seed 0 reproduces",
          "`results/gepa_v3_test.json`). Both runs are re-executed offline from the Jev cache and checked against the",
          "scores recorded at consumption. Calls, tokens and $ are per test evaluation (`test_consumed.json`).", "",
          *ci_rows, "",
          "## Cost of the seed-1/seed-2 runs", "",
          "Jev fresh: billed during the validation search (the shared cache serves requests already made by seed 0). "
          "OMP: proposer LLM spend. Jev test: the test requests added to the run's response ledger by its one test "
          "evaluation (the chosen program's requests on the test split).", "",
          *cost_rows,
          f"| **total** | **{totals['jev_fresh']:.2f}** | | **{totals['omp']:.2f}** | **{totals['jev_test']:.2f}** |", ""]
    OUT_MD.write_text("\n".join(L))
    return {"tasks": tasks, "cost_seeds_1_2": totals}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    args = ap.parse_args()
    prev = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    sel = select(prev.get("selection", {}))
    out: dict[str, Any] = {"margin_points": MARGIN, "selection": sel}
    if not args.select:
        out |= report(sel)
    OUT_JSON.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({stem: {seed: (s.get("representative"), round(100 * s["val"][s["representative"]], 1) if "val" in s else s["status"]) for seed, s in v.items()} for stem, v in sel.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
