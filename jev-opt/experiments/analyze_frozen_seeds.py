"""Seed variance of the frozen configuration on the 5 eligible sets (pre-registered)
-> results/frozen_seeds.json, results/FROZEN_SEEDS.md, paper/generated/frozen_seeds.tex (fzs... macros).

  --select  validation only: per set and seed 1/2, top-single vs top-stacked by the seed-variance rule (P+head
            replaces P iff its val primary metric is >= 1.0 point higher); writes results/frozen_seeds.json
            ["selection"], which `run_frozen_seeds.sh test` reads. Opens no test file. A recorded selection never changes.
  (default) also reads each chosen representative's test score recorded at consumption (`test_consumed.json`) and
            writes the report. Seed 0 = `fz-<set>-P` and its saved result in results/frozen.json.

Gap closed = (3-seed mean test − zero-shot) / (paper method − zero-shot), with results/frozen.json's zero-shot and
paper values. Pooled 13 configurations: the 8 original configurations' 3-seed means (results/seeds.json) against
results/frozen.json's values for them, plus the 5 sets. No Jev or LLM call is made. Run from the repo root:
  uv run --project jev-opt/python python jev-opt/experiments/analyze_frozen_seeds.py [--select] [--out DIR]
(--out DIR replaces paper/generated for the macro file.)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_frozen import NEW, TOL, gap, paper_tables  # noqa: E402
from analyze_seeds import candidate, choose, finished, val_scores  # noqa: E402
from analyze_v2 import jev_cost, omp_cost  # noqa: E402
from analyze_v3 import fresh  # noqa: E402

ROOT = Path.cwd()
RUNS = ROOT / "runs" / "jev-opt"
OUT_JSON = ROOT / "results" / "frozen_seeds.json"
OUT_MD = ROOT / "results" / "FROZEN_SEEDS.md"
TEX_DIR = ROOT / "paper" / "generated"
SEEDS = (0, 1, 2)
NEW_SEEDS = (1, 2)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def run_dir(stem: str, seed: int) -> Path:
    return RUNS / (f"fz-{stem}-P" if seed == 0 else f"fz-{stem}-P-s{seed}")


def select(prev: dict[str, Any]) -> dict[str, Any]:
    sel: dict[str, Any] = {}
    for stem in NEW:
        sel[stem] = {}
        for seed in NEW_SEEDS:
            d = run_dir(stem, seed)
            if not finished(d):
                sel[stem][str(seed)] = {"run": d.name, "status": "not finished" if d.exists() else "not started"}
                continue
            val = val_scores(d)
            rep = choose(val)
            old = prev.get(stem, {}).get(str(seed), {})
            if old.get("representative") not in (None, rep):
                raise SystemExit(f"{d.name}: selection changed from {old['representative']} to {rep}; the rule is fixed")
            f_cost, f_calls, f_tok = fresh(d)
            sel[stem][str(seed)] = {
                "run": d.name, "status": "finished", "val": val, "representative": rep, "candidate": candidate(d, rep),
                # recorded once, before the test evaluation: the ledger's growth afterwards is the test spend
                "jev_used_before_test": old.get("jev_used_before_test", jev_cost(d)),
                "jev_fresh": f_cost, "jev_fresh_requests": f_calls, "jev_fresh_input_tokens": f_tok, "omp": omp_cost(d),
            }
    return sel


def sd(xs: list[float]) -> float | None:
    return statistics.stdev(xs) if len(xs) > 1 else None


def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """analyze_frozen.summarize for one system score `key`: configurations averaged within a set (DREsS), sets within a paper."""
    by_set: dict[str, list[float]] = {}
    set_paper: dict[str, str] = {}
    reach_m: dict[str, list[bool]] = {}
    reach_l: dict[str, list[bool]] = {}
    for r in rows:
        by_set.setdefault(r["set"], []).append(gap(r[key], r["zero"], r["paper"]))
        set_paper[r["set"]] = r["paper_id"]
        reach_m.setdefault(r["set"], []).append(r[key] >= r["paper"] - TOL)
        if r["llm"] is not None:
            reach_l.setdefault(r["set"], []).append(r[key] >= r["llm"] - TOL)
    set_gap = {s: statistics.mean(v) for s, v in by_set.items()}
    by_paper: dict[str, list[float]] = {}
    for s, g in set_gap.items():
        by_paper.setdefault(set_paper[s], []).append(g)
    paper_gap = {p: statistics.mean(v) for p, v in by_paper.items()}
    return {"n_configs": len(rows), "median_gap_per_set": statistics.median(set_gap.values()),
            "median_gap_per_paper": statistics.median(paper_gap.values()), "set_gap": set_gap, "paper_gap": paper_gap,
            "n_sets": len(set_gap), "n_papers": len(paper_gap), "n_reach_method": sum(all(v) for v in reach_m.values()),
            "n_reach_llm": sum(all(v) for v in reach_l.values()), "n_with_llm": len(reach_l)}


def report(sel: dict[str, Any]) -> dict[str, Any]:
    fz = load_json(ROOT / "results" / "frozen.json")
    fz_new = {r["stem"]: r for r in fz["new"]}
    fz_orig = {r["stem"]: r for r in fz["orig"]}
    seeds_v3 = load_json(ROOT / "results" / "seeds.json")["tasks"]
    pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
    rows, failures = [], []
    L = ["# Seed variance of the frozen configuration on the 5 eligible sets", "",
         "The study was pre-registered. Generated by `jev-opt/experiments/analyze_frozen_seeds.py`.",
         "The frozen configuration (population search over questions + fitted readout) rerun with optimizer",
         "seeds 1 and 2 (seed 0 = `fz-<set>-P`). Per run, val picks P (`top-single`) unless P+head (`top-stacked`) is ≥ 1.0 point",
         "higher; that one representative is evaluated once on test. Values ×100; mean ± sample sd (n − 1). Gap closed =",
         "(score − zero-shot) / (paper method − zero-shot), zero-shot and paper values as in `results/FROZEN.md`.", "",
         "| Set | Metric | Seed | Chosen (candidate) | Val P | Val P+head | Test |", "|---|---|---:|---|---:|---:|---:|"]
    summ = ["| Set | Zero-shot | Paper method | Seed 0 | Seed 1 | Seed 2 | Mean ± sd | Gap closed seed 0 | Gap closed mean |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    cost_rows = ["| Run | Jev fresh $ (val search) | fresh requests | OMP $ | Jev $ test |", "|---|---:|---:|---:|---:|"]
    tot = {"jev_fresh": 0.0, "omp": 0.0, "jev_test": 0.0}
    for stem, (task, paper_id, macro) in NEW.items():
        title = paper_tables.CONFIG_LABEL[stem]
        f = fz_new[stem]
        per: dict[str, Any] = {"0": {"run": f["run"], "representative": f["representative"], "candidate": f["candidate"],
                                     "val": f["val"][f["representative"]], "val_all": f["val"], "test": f["optimized"]}}
        tc0 = json.loads((run_dir(stem, 0) / "test_consumed.json").read_text())
        assert abs(tc0["metrics"]["primary"] - f["optimized"]) < 1e-9, f"{stem}: seed-0 test differs from frozen.json"
        for seed in NEW_SEEDS:
            s = sel[stem][str(seed)]
            d = run_dir(stem, seed)
            tc_path = d / "test_consumed.json"
            if s["status"] != "finished" or not tc_path.exists():
                failures.append({"run": d.name, "status": s["status"], "test": tc_path.exists()})
                per[str(seed)] = {"run": d.name, "status": s["status"], "test": None}
                continue
            tc = json.loads(tc_path.read_text())
            if tc["representative"] != s["representative"]:
                raise SystemExit(f"{d.name}: test consumed by {tc['representative']}, selection is {s['representative']}")
            jt = jev_cost(d) - s["jev_used_before_test"]
            per[str(seed)] = {"run": d.name, "representative": s["representative"], "candidate": s["candidate"],
                              "val": s["val"][s["representative"]], "val_all": s["val"], "test": float(tc["metrics"]["primary"]),
                              "jev_fresh": s["jev_fresh"], "omp": s["omp"], "jev_test": jt}
            cost_rows.append(f"| `{d.name}` | {s['jev_fresh']:.2f} | {s['jev_fresh_requests']:,} | {s['omp']:.2f} | {jt:.2f} |")
            tot["jev_fresh"] += s["jev_fresh"]
            tot["omp"] += s["omp"]
            tot["jev_test"] += jt
        for seed in SEEDS:
            p = per[str(seed)]
            if p.get("test") is None:
                L.append(f"| {title} | {f['metric']} | {seed} | failed: {p['status']} | | | |")
                continue
            v = p["val_all"]
            stacked = pct(v["top-stacked"]) if "top-stacked" in v else ""
            L.append(f"| {title} | {f['metric']} | {seed} | {p['representative']} ({p['candidate']}) | {pct(v['top-single'])} | {stacked} | {pct(p['test'])} |")
        tests = [per[str(s)]["test"] for s in SEEDS if per[str(s)].get("test") is not None]
        mean = statistics.mean(tests)
        row = {"stem": stem, "title": title, "macro": macro, "set": task, "paper_id": paper_id, "metric": f["metric"],
               "zero": f["zero"], "paper": f["paper"], "llm": f["llm"], "seeds": per, "n_seeds": len(tests),
               "single": f["optimized"], "mean": mean, "sd": sd(tests),
               "gap_single": gap(f["optimized"], f["zero"], f["paper"]), "gap_mean": gap(mean, f["zero"], f["paper"])}
        rows.append(row)
        cells = [pct(per[str(s)]["test"]) if per[str(s)].get("test") is not None else "failed" for s in SEEDS]
        sdt = f" ± {100 * row['sd']:.1f}" if row["sd"] is not None else ""
        summ.append(f"| {title} | {pct(f['zero'])} | {pct(f['paper'])} | {' | '.join(cells)} | {pct(mean)}{sdt} (n = {len(tests)}) | "
                    f"{100 * row['gap_single']:.0f}% | {100 * row['gap_mean']:.0f}% |")
    orig = []
    for slug, r in fz_orig.items():
        t = seeds_v3[slug]
        s0 = t["seeds"]["0"]["test"]
        assert abs(s0 - r["optimized"]) < 1e-9, f"{slug}: seeds.json seed-0 test differs from frozen.json"
        orig.append({"stem": slug, "set": r["set"], "paper_id": r["paper_id"], "zero": r["zero"], "paper": r["paper"],
                     "llm": r["llm"], "single": r["optimized"], "mean": t["test_mean"], "n_seeds": t["n_test"]})
    summary = {}
    for grp, rs in (("new", rows), ("orig", orig), ("combined", rows + orig)):
        summary[grp] = {"single": summarize(rs, "single"), "mean": summarize(rs, "mean")}
    L += ["", "## Per set: test scores and gap closed", "", *summ, "", "## Summary (optimized questions; single run = seed 0 vs 3-seed mean)", "",
          "| Configurations | | Single run (seed 0) | 3-seed mean |", "|---|---|---:|---:|"]
    labels = {"new": "5 eligible sets", "orig": "8 original configurations (6 sets)", "combined": "all 13 configurations (11 sets)"}
    for grp, s in summary.items():
        a, b = s["single"], s["mean"]
        L += [f"| {labels[grp]} | median gap closed per set | {100 * a['median_gap_per_set']:.0f}% | {100 * b['median_gap_per_set']:.0f}% |",
              f"| | median gap closed per paper | {100 * a['median_gap_per_paper']:.0f}% | {100 * b['median_gap_per_paper']:.0f}% |",
              f"| | sets reaching the paper's method | {a['n_reach_method']} / {a['n_sets']} | {b['n_reach_method']} / {b['n_sets']} |",
              f"| | sets reaching the paper's LLM | {a['n_reach_llm']} / {a['n_with_llm']} | {b['n_reach_llm']} / {b['n_with_llm']} |"]
    L += ["", "Original configurations: 3-seed means of `results/SEEDS.md`, gap against `results/FROZEN.md`'s zero-shot and paper values.", "",
          "## Cost of the seed-1/seed-2 runs", "",
          "Jev fresh: billed during the validation search (the shared cache serves requests seed 0 already made). OMP: optimizer",
          "LLM spend. Jev test: requests added to the run's response ledger by its one test evaluation.", "",
          *cost_rows, f"| **total** | **{tot['jev_fresh']:.2f}** | | **{tot['omp']:.2f}** | **{tot['jev_test']:.2f}** |", ""]
    if failures:
        L += ["## Failures", "", *[f"- `{x['run']}`: {x['status']} (test recorded: {x['test']})" for x in failures], ""]
    OUT_MD.write_text("\n".join(L))
    tot["total"] = sum(tot.values())
    return {"sets": rows, "orig": orig, "summary": summary, "failures": failures, "cost_seeds_1_2": tot}


def tex(rep: dict[str, Any], out_dir: Path) -> None:
    one = lambda x: f"{100 * x:.1f}"  # noqa: E731
    word = {0: "Zero", 1: "One", 2: "Two"}
    m: list[tuple[str, str]] = []
    for r in rep["sets"]:
        k = r["macro"]
        for s in SEEDS:
            t = r["seeds"][str(s)].get("test")
            m.append((f"fzs{k}Seed{word[s]}", "--" if t is None else one(t)))
        m += [(f"fzs{k}Mean", one(r["mean"])), (f"fzs{k}Sd", "--" if r["sd"] is None else one(r["sd"])),
              (f"fzs{k}GapMean", f"{100 * r['gap_mean']:.0f}"), (f"fzs{k}NSeeds", str(r["n_seeds"]))]
    for grp, name in (("new", "New"), ("orig", "Orig"), ("combined", "All")):
        for kind, kn in (("single", "Single"), ("mean", "Mean")):
            t = rep["summary"][grp][kind]
            m += [(f"fzs{name}MedGapSet{kn}", f"{100 * t['median_gap_per_set']:.0f}"),
                  (f"fzs{name}MedGapPaper{kn}", f"{100 * t['median_gap_per_paper']:.0f}"),
                  (f"fzs{name}ReachMethod{kn}", str(t["n_reach_method"])), (f"fzs{name}ReachLlm{kn}", str(t["n_reach_llm"]))]
    c = rep["cost_seeds_1_2"]
    m += [("fzsCostJevSearch", f"{c['jev_fresh']:.2f}"), ("fzsCostOmp", f"{c['omp']:.2f}"), ("fzsCostJevTest", f"{c['jev_test']:.2f}"),
          ("fzsCostTotal", f"{c['total']:.2f}")]
    head = "% Generated by jev-opt/experiments/analyze_frozen_seeds.py; do not edit.\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frozen_seeds.tex").write_text(head + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--out", type=Path, default=TEX_DIR, help="directory for frozen_seeds.tex")
    args = ap.parse_args()
    prev = load_json(OUT_JSON)
    out: dict[str, Any] = {"margin_points": 1.0,
                           "selection": select(prev.get("selection", {}))}
    if not args.select:
        rep = report(out["selection"])
        out |= rep
        tex(rep, args.out)
    OUT_JSON.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({stem: {seed: (s.get("representative"), round(100 * s["val"][s["representative"]], 1) if "val" in s else s["status"])
                             for seed, s in v.items()} for stem, v in out["selection"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
