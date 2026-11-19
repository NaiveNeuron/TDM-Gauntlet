"""Frozen-configuration round (final optimizer settings) -> results/frozen.json, results/FROZEN.md, paper/generated/frozen.tex.

  --select : after the fz-<set>-P validation runs: P (top-single) vs P+head (top-stacked) by the 1.0-point
             val margin (the seed-variance rule), plus run costs, written to results/frozen.json["selection"].
             Never reads a test split. A selection already recorded can never change.
  (default): the full report, once every chosen representative has been evaluated on test.

Everything is computed offline: Jev answers come from the runs' caches (posting disabled by framing_features,
so a cache miss is an error and no new test query can happen) and from the gauntlet's saved zero-shot
responses. Run from the repository root (Python 3.11):
  uv run --project jev-opt/python python jev-opt/experiments/analyze_frozen.py [--select]
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_seeds import choose, finished, val_scores  # noqa: E402
from analyze_v2 import jev_cost, omp_cost  # noqa: E402
from analyze_v3 import fresh  # noqa: E402

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
import paper_tables  # noqa: E402  (labels of the sets: CONFIG_LABEL)

RUNS = ROOT / "runs" / "jev-opt"
OUT_JSON = ROOT / "results" / "frozen.json"
OUT_MD = ROOT / "results" / "FROZEN.md"
OUT_TEX = ROOT / "paper" / "generated" / "frozen.tex"
OUT_APP = ROOT / "paper" / "generated" / "app_frozen.tex"
B = 2000
TOL = 0.005  # the grouping's 0.5-point tolerance
HEADER = ["| Set | Metric | n | Zero-shot | Calibrated | Δ cal. (CI) | Optimized | Δ opt. (CI) | Chosen (val) | Paper method | Paper LLM | Gap closed cal. / opt. |",
          "|---|---|---:|---:|---:|---|---:|---|---|---:|---:|---|"]

# New sets: stem -> (gauntlet task, paper, macro name); titles are paper_tables.CONFIG_LABEL
NEW = {
    "contempo-matres": ("contempo/matres", "contempo", "Matres"),
    "lifetox": ("lifetox", "lifetox", "Lifetox"),
    "sciprompt-s2orc": ("sciprompt/s2orc", "sciprompt", "Storc"),
    "smarter-hatexplain": ("smarter/hatexplain", "smarter", "Hatexplain"),
    "smarter-latent-hate": ("smarter/latent_hate", "smarter", "Latent"),
}
# The 8 original configurations (third pre-registration, seed 0): slug -> (gauntlet task, paper, set)
ORIG = {
    "contempo-tbdense": ("contempo/tbdense", "contempo", "contempo/tbdense"),
    "ezstance-subtaskA": ("ezstance/subtaskA", "ezstance", "ezstance/subtaskA"),
    "smarter-implicit-hate": ("smarter/implicit_hate", "smarter", "smarter/implicit_hate"),
    "edit-intent": ("edit_intent", "edit_intent", "edit_intent"),
    "inli": ("inli", "inli", "inli"),
    "dress-new-content": ("dress_new", "dress", "dress_new"),
    "dress-new-organization": ("dress_new", "dress", "dress_new"),
    "dress-new-language": ("dress_new", "dress", "dress_new"),
}


NEW_DIRS = {"contempo-matres": "contempo_matres", "lifetox": "lifetox", "sciprompt-s2orc": "sciprompt_s2orc",
            "smarter-hatexplain": "smarter_hatexplain", "smarter-latent-hate": "smarter_latent_hate"}


def seed_writer_cost(path: Path) -> float:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sum(float(r.get("cost_total") or 0.0) for r in rows if r.get("source") == "omp")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


# --------------------------------------------------------------------------------------------
# selection (val only)


def select(prev: dict[str, Any]) -> dict[str, Any]:
    sel: dict[str, Any] = {}
    for stem in NEW:
        d = RUNS / f"fz-{stem}-P"
        a = RUNS / f"fz-{stem}-A"
        if not finished(d):
            sel[stem] = {"run": d.name, "status": "not finished" if d.exists() else "not started"}
            continue
        val = val_scores(d)
        rep = choose(val)
        old = prev.get(stem, {})
        if old.get("representative") not in (None, rep):
            raise SystemExit(f"{d.name}: selection changed from {old['representative']} to {rep}; the rule is fixed")
        cand = json.loads((d / "pareto.json").read_text())["representatives"][rep].get("candidate", "")
        f_cost, f_calls, f_tok = fresh(d)
        fa = fresh(a) if finished(a) else (None, None, None)
        sel[stem] = {
            "run": d.name, "status": "finished", "val": val, "representative": rep, "candidate": cand,
            # recorded once, before the test evaluation: the ledger's growth afterwards is the test spend
            "jev_used_before_test": old.get("jev_used_before_test", jev_cost(d)),
            "jev_fresh": f_cost, "jev_fresh_requests": f_calls, "jev_fresh_input_tokens": f_tok, "omp": omp_cost(d),
            "a_run": a.name, "a_jev_fresh": fa[0], "a_jev_fresh_requests": fa[1], "a_omp": omp_cost(a) if a.exists() else None,
        }
    return sel


# --------------------------------------------------------------------------------------------
# report


def gauntlet_zero(task: str, labels: list[str]) -> dict[str, dict[str, Any]]:
    """item id -> recorded zero-shot prediction and full answer distribution (single-question sets)."""
    from confidence_readout import normalise

    dists: dict[str, dict[str, float]] = {}
    for shard in sorted((ROOT / "runs" / task).glob("responses*.jsonl.gz")):
        for line in gzip.open(shard, "rt"):
            r = json.loads(line)
            ans = r["response"]["answers"]
            assert len(ans) == 1
            dists[r["item_id"]] = next(iter(ans.values()))["probabilities"]
    out = {}
    for line in (ROOT / "runs" / task / "predictions.jsonl").read_text().splitlines():
        r = json.loads(line)
        p = dists[r["item_id"]]
        assert set(p) <= set(labels), (task, set(p) - set(labels))
        out[r["item_id"]] = {"pred": r["prediction"]["label"], "gold": r["gold"], "dist": normalise(p, labels)}
    return out


def calibration(a_dir: Path, labels: list[str]) -> dict[str, Any]:
    """Per-class bias + temperature on the paper's question, fit on the fit slice (confidence_readout's bias+TS)."""
    import framing_features as ff
    from confidence_readout import floored

    from jev_opt.adapter import train_roles
    from jev_opt.execute import combine
    from jev_opt.fit import fit_label_bias
    from jev_opt.ir import load_decider

    env = ff.Env(a_dir)
    try:
        assert list(env.config.labels) == labels
        fit, _ = train_roles(env.train, env.config.optimizer)
        seed = load_decider((a_dir / "seed.yaml").read_text())
        res = env.run(seed, fit, "train")
        dists = [floored(r.dist, labels) for r in res]
        gold = [r.gold for r in res]
        bias, temp, rep = fit_label_bias(labels, [[d] for d in dists], gold, env.config.primary_metric, env.config.negative_label)
        vres = env.run(seed, env.val, "val")
        vpred = [combine(labels, [floored(r.dist, labels)], bias, temp)[1] for r in vres]
        vgold = [r.gold for r in vres]
        return {"n_fit": len(res), "bias": bias, "temperature": temp, "fit_primary_before_after": [rep["before"], rep["after"]],
                "val_zero": env.primary(vgold, [r.pred for r in vres]), "val_calibrated": env.primary(vgold, vpred),
                "fit_ids": [e.id for e in fit]}
    finally:
        env.close()


def apply_cal(zero: list[dict[str, Any]], labels: list[str], cal: dict[str, Any]) -> list[str]:
    from confidence_readout import floored

    from jev_opt.execute import combine

    return [combine(labels, [floored(dict(zip(labels, x["dist"])), labels)], cal["bias"], cal["temperature"])[1] for x in zero]


def boot_delta(score, gold: list[str], base: list[str], other: list[str], boots: np.ndarray) -> list[float]:
    d = np.array([score([gold[i] for i in b], [other[i] for i in b]) - score([gold[i] for i in b], [base[i] for i in b]) for b in boots])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return [float(lo), float(hi)]


def gap(x: float | None, zero: float, paper: float) -> float | None:
    return None if x is None else (x - zero) / (paper - zero)


def config_row(stem: str, run_dir: Path, a_dir: Path, task: str, title: str, paper_name: str, set_id: str,
               paper: float, llm: float | None, group: str) -> dict[str, Any]:
    from analyze_test_v3 import DISCREPANCIES, evaluate
    from confidence_readout import zero_shot

    from jev_opt import metrics

    opt = evaluate(run_dir)
    if DISCREPANCIES:
        raise SystemExit(f"re-executed test scores differ from the recorded ones: {DISCREPANCIES}")
    cfg = opt["config"]
    labels = list(cfg.labels)
    score = lambda g, p: metrics.primary(cfg.primary_metric, g, p, labels, cfg.negative_label)  # noqa: E731
    zs = zero_shot(stem, labels) if group == "orig" else gauntlet_zero(task, labels)
    ids, gold = opt["ids"], opt["gold"]
    z = [zs[i] for i in ids]
    assert [x["gold"] for x in z] == gold, f"{stem}: zero-shot gold differs from the jev-opt test split"
    zpred = [x["pred"] for x in z]
    zero = score(gold, zpred)
    cal = calibration(a_dir, labels)
    cpred = apply_cal(z, labels, cal)
    calv = score(gold, cpred)
    from resample import cluster_index, resamples

    inv, unit = cluster_index(stem, ids)  # RQ1's resampling unit for the set
    boots = resamples(inv, B, 0)
    row = {
        "stem": stem, "title": title, "paper_id": paper_name, "set": set_id, "group": group, "metric": cfg.primary_metric,
        "cluster_unit": unit, "n_clusters": int(inv.max()) + 1,
        "n_test": len(ids), "run": run_dir.name, "representative": opt["rep"], "candidate": opt["candidate"],
        "zero": zero, "calibrated": calv, "optimized": opt["primary"], "paper": paper, "llm": llm,
        "delta_cal": calv - zero, "ci_cal": boot_delta(score, gold, zpred, cpred, boots),
        "delta_opt": opt["primary"] - zero, "ci_opt": boot_delta(score, gold, zpred, opt["pred"], boots),
        "gap_cal": gap(calv, zero, paper), "gap_opt": gap(opt["primary"], zero, paper),
        "reach_method_cal": calv >= paper - TOL, "reach_method_opt": opt["primary"] >= paper - TOL,
        "reach_llm_cal": None if llm is None else calv >= llm - TOL, "reach_llm_opt": None if llm is None else opt["primary"] >= llm - TOL,
        "calibration": {k: v for k, v in cal.items() if k != "fit_ids"}, "test_cost": opt["cost"],
        "test_calls_per_example": opt["calls"], "test_input_tokens_per_example": opt["tokens"],
    }
    # Optimized score against the paper's method with RQ1's inference: clustered bootstrap SE of the score,
    # x sqrt(2) where RQ1 widens the set (published value on other items); normal 95% interval of the difference.
    widen = json.loads((ROOT / "results" / "offline_v4" / "rq1.json").read_text())
    widen = {s["task"]: bool(s["widen"]) for s in widen["sets"]}.get(task, False)
    se = float(np.std([score([gold[i] for i in b], [opt["pred"][i] for i in b]) for b in boots], ddof=1))
    se_eff = se * (2 ** 0.5 if widen else 1.0)
    d = opt["primary"] - paper
    row["vs_paper_opt"] = {"delta": d, "se": se, "se_eff": se_eff, "widen": widen, "ci": [d - 1.959964 * se_eff, d + 1.959964 * se_eff]}
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_configs": len(rows)}
    for sys_ in ("cal", "opt"):
        by_set: dict[str, list[float]] = {}
        set_paper: dict[str, str] = {}
        reach_m: dict[str, list[bool]] = {}
        reach_l: dict[str, list[bool]] = {}
        for r in rows:
            by_set.setdefault(r["set"], []).append(r[f"gap_{sys_}"])
            set_paper[r["set"]] = r["paper_id"]
            reach_m.setdefault(r["set"], []).append(r[f"reach_method_{sys_}"])
            if r[f"reach_llm_{sys_}"] is not None:
                reach_l.setdefault(r["set"], []).append(r[f"reach_llm_{sys_}"])
        set_gap = {s: statistics.mean(v) for s, v in by_set.items()}
        by_paper: dict[str, list[float]] = {}
        for s, g in set_gap.items():
            by_paper.setdefault(set_paper[s], []).append(g)
        paper_gap = {p: statistics.mean(v) for p, v in by_paper.items()}
        out[sys_] = {
            "median_gap_per_set": statistics.median(set_gap.values()), "median_gap_per_paper": statistics.median(paper_gap.values()),
            "set_gap": set_gap, "paper_gap": paper_gap, "n_sets": len(set_gap), "n_papers": len(paper_gap),
            # a set counts as reaching when all of its configurations do (DREsS: the three rubrics)
            "n_reach_method": sum(all(v) for v in reach_m.values()), "n_reach_llm": sum(all(v) for v in reach_l.values()),
            "n_with_llm": len(reach_l),
            "n_configs_reach_method": sum(r[f"reach_method_{sys_}"] for r in rows),
            "n_configs_ci_above_zero": sum(r[f"ci_{sys_}"][0] > 0 for r in rows),
        }
    return out


def pct(x: float | None, signed: bool = False) -> str:
    return "–" if x is None else (f"{100 * x:+.1f}" if signed else f"{100 * x:.1f}")


def report(sel: dict[str, Any]) -> dict[str, Any]:
    from analyze_test_v3 import PAPER as V3_PAPER

    ptab = {r["task"]: r for r in paper_tables.rows()}
    rows, failures = [], []
    for stem, (task, paper_id, _) in NEW.items():
        title = paper_tables.CONFIG_LABEL[stem]
        s = sel.get(stem, {})
        d = RUNS / f"fz-{stem}-P"
        if s.get("status") != "finished" or not (d / "test_consumed.json").exists():
            failures.append({"stem": stem, "status": s.get("status", "missing"), "test": (d / "test_consumed.json").exists()})
            continue
        tc = json.loads((d / "test_consumed.json").read_text())
        if tc["representative"] != s["representative"]:
            raise SystemExit(f"{d.name}: test consumed by {tc['representative']}, selection is {s['representative']}")
        pt = ptab[task]
        r = config_row(stem, d, RUNS / f"fz-{stem}-A", task, title, paper_id, task, pt["paper"], pt["llm"], "new")
        if abs(r["zero"] - pt["jev"]) > 1e-3:
            raise SystemExit(f"{stem}: recomputed zero-shot {r['zero']} != the audit's {pt['jev']}")
        r |= {"val": s["val"], "jev_fresh_search": s["jev_fresh"], "omp": s["omp"], "a_jev_fresh": s["a_jev_fresh"],
              "jev_test": jev_cost(d) - s["jev_used_before_test"],
              "omp_seeds": seed_writer_cost(HERE / NEW_DIRS[stem] / "seeds_v3" / "usage.jsonl")}
        rows.append(r)
    orig = []
    seeds = load_json(ROOT / "results" / "seeds.json")["selection"]
    for slug, (task, paper_id, set_id) in ORIG.items():
        title = paper_tables.CONFIG_LABEL[slug]
        paper, llm = V3_PAPER[slug]
        s0 = seeds[slug]["0"]
        r = config_row(slug, RUNS / s0["run"], RUNS / f"v3-{slug}-A", task, title, paper_id, set_id, paper, llm, "orig")
        orig.append(r)
    new_sum = summarize(rows) if rows else None
    orig_sum = summarize(orig)
    all_sum = summarize(rows + orig) if rows else None

    L = ["# Frozen configuration on every eligible shortfall set", "",
         "The study was pre-registered. Generated by `jev-opt/experiments/analyze_frozen.py`",
         "(offline: every score is recomputed from the runs' Jev caches and the gauntlet's zero-shot responses). Values ×100,",
         "each paper's primary metric. Zero-shot = the gauntlet's zero-shot test result (the audit table); calibrated = the",
         "paper's question + per-class bias + temperature fit on the 400-example fit slice; optimized = the v3 configuration",
         "(population search over questions + fitted readout, seed 0), the representative chosen on val and evaluated once on",
         "test. Δ: paired bootstrap 95% CI, clusters of the RQ1 statistics resampled (MATRES: document; LifeTox: query; else item; 2,000 resamples, seed 0). Gap closed = (system − zero-shot) /",
         "(paper method − zero-shot). Reaches = within 0.5 points of the paper's number or above.", "",
         "## New sets (all eligible)", "", *HEADER]

    def line(r: dict[str, Any], chosen: str) -> str:
        return (f"| {r['title']} | {r['metric']} | {r['n_test']:,} | {pct(r['zero'])} | {pct(r['calibrated'])} | "
                f"{pct(r['delta_cal'], True)} ({pct(r['ci_cal'][0], True)}…{pct(r['ci_cal'][1], True)}) | {pct(r['optimized'])} | "
                f"{pct(r['delta_opt'], True)} ({pct(r['ci_opt'][0], True)}…{pct(r['ci_opt'][1], True)}) | {chosen} | "
                f"{pct(r['paper'])} | {pct(r['llm'])} | {pct(r['gap_cal'])}% / {pct(r['gap_opt'])}% |")

    for r in rows:
        v = r["val"]
        L.append(line(r, f"{'P+head' if r['representative'] == 'top-stacked' else 'P'} ({pct(v[r['representative']])})"))
    for f in failures:
        L.append(f"| {paper_tables.CONFIG_LABEL[f['stem']]} | | | | | | failed: {f['status']}{'' if f['test'] else ', no test'} | | | | | |")
    L += ["", "## Original 8 configurations (chosen for large gaps; third pre-registration, seed 0, recomputed against the gauntlet zero-shot)", "",
          *HEADER]
    for r in orig:
        L.append(line(r, "P+head" if r["representative"] == "top-stacked" else "P"))

    def summ(title: str, s: dict[str, Any] | None) -> list[str]:
        if s is None:
            return []
        out = [f"### {title}", "", "| | Calibrated | Optimized |", "|---|---:|---:|"]
        c, o = s["cal"], s["opt"]
        out += [f"| sets / papers | {c['n_sets']} / {c['n_papers']} | {o['n_sets']} / {o['n_papers']} |",
                f"| median gap closed per set | {pct(c['median_gap_per_set'])}% | {pct(o['median_gap_per_set'])}% |",
                f"| median gap closed per paper | {pct(c['median_gap_per_paper'])}% | {pct(o['median_gap_per_paper'])}% |",
                f"| sets reaching the paper's method | {c['n_reach_method']} / {c['n_sets']} | {o['n_reach_method']} / {o['n_sets']} |",
                f"| sets reaching the paper's LLM | {c['n_reach_llm']} / {c['n_with_llm']} | {o['n_reach_llm']} / {o['n_with_llm']} |",
                f"| configurations with Δ CI above 0 | {c['n_configs_ci_above_zero']} / {s['n_configs']} | {o['n_configs_ci_above_zero']} / {s['n_configs']} |", ""]
        return out

    L += ["", "## Summary (RQ3)", "", "A set's gap closed is the mean over its configurations (DREsS: three rubrics); a paper's is",
          "the mean over its sets. A set reaches a reference when all its configurations do.", ""]
    L += summ("New sets: all eligible", new_sum) + summ("Original sets: chosen for large gaps", orig_sum) + summ("Combined: original + all eligible", all_sum)
    cost_rows = ["| Run | Jev fresh $ (A run) | Jev fresh $ (P search) | OMP $ (search) | OMP $ (seed writer) | Jev $ test |",
                 "|---|---:|---:|---:|---:|---:|"]
    tot = {"a": 0.0, "p": 0.0, "omp": 0.0, "omp_seeds": 0.0, "test": 0.0}
    for r in rows:
        cost_rows.append(f"| `fz-{r['stem']}` | {r['a_jev_fresh'] or 0:.2f} | {r['jev_fresh_search']:.2f} | {r['omp']:.2f} | {r['omp_seeds']:.2f} | {r['jev_test']:.2f} |")
        tot["a"] += r["a_jev_fresh"] or 0
        tot["p"] += r["jev_fresh_search"]
        tot["omp"] += r["omp"]
        tot["omp_seeds"] += r["omp_seeds"]
        tot["test"] += r["jev_test"]
    cost_rows.append(f"| **total** | **{tot['a']:.2f}** | **{tot['p']:.2f}** | **{tot['omp']:.2f}** | **{tot['omp_seeds']:.2f}** | **{tot['test']:.2f}** |")
    L += ["## Costs", "", "Jev fresh = billed requests (the shared cache serves earlier requests for free); OMP = proposer and seed-writer",
          "LLM spend recorded by the run; Jev test = the one test evaluation's billed requests. The calibration step adds no",
          "Jev call beyond the A run (its test predictions reuse the gauntlet's zero-shot responses).", "", *cost_rows, ""]
    if failures:
        L += ["## Failures", "", *[f"- `fz-{f['stem']}`: {f['status']}" for f in failures], ""]
    OUT_MD.write_text("\n".join(L))
    return {"new": rows, "orig": orig, "failures": failures, "summary": {"new": new_sum, "orig": orig_sum, "combined": all_sum}, "costs": tot}


def tex(rep: dict[str, Any]) -> None:
    m: list[tuple[str, str]] = []
    one = lambda x: f"{100 * x:.1f}"  # noqa: E731
    m.append(("fzNSets", str(len(NEW))))
    m.append(("fzNFinished", str(len(rep["new"]))))
    for r in rep["new"]:
        k = NEW[r["stem"]][2]
        for key, name in (("zero", "Zero"), ("calibrated", "Cal"), ("optimized", "Opt"), ("paper", "Paper")):
            m.append((f"fz{k}{name}", one(r[key])))
        if r["llm"] is not None:
            m.append((f"fz{k}Llm", one(r["llm"])))
        m.append((f"fz{k}GapCal", f"{100 * r['gap_cal']:.0f}"))
        m.append((f"fz{k}GapOpt", f"{100 * r['gap_opt']:.0f}"))
        vp = r["vs_paper_opt"]
        sg = lambda x: f"{100 * x:+.1f}".replace("-", "$-$")  # noqa: E731
        m += [(f"fz{k}VsPaper", sg(vp["delta"])), (f"fz{k}VsPaperLo", sg(vp["ci"][0])), (f"fz{k}VsPaperHi", sg(vp["ci"][1]))]
    # The body's calibration paragraph quotes the calibration step on TB-Dense (seed-0 block of the appendix table).
    tb = next(r for r in rep["orig"] if r["stem"] == "contempo-tbdense")
    m += [("fzTbDenseZero", one(tb["zero"])), ("fzTbDenseCal", one(tb["calibrated"]))]
    # The combined summary mixes the original configurations' seed-0 scores with the new sets' single runs; the
    # paper reports the 8 original configurations as 3-seed means, so it gets no macro (frozen.json keeps it).
    for grp, name in (("new", "New"), ("orig", "Orig")):
        s = rep["summary"][grp]
        if s is None:
            continue
        for sys_, sn in (("cal", "Cal"), ("opt", "Opt")):
            t = s[sys_]
            m += [(f"fz{name}MedGapSet{sn}", f"{100 * t['median_gap_per_set']:.0f}"), (f"fz{name}MedGapPaper{sn}", f"{100 * t['median_gap_per_paper']:.0f}"),
                  (f"fz{name}ReachMethod{sn}", str(t["n_reach_method"])), (f"fz{name}ReachLlm{sn}", str(t["n_reach_llm"])),
                  (f"fz{name}NImproved{sn}", str(t["n_configs_ci_above_zero"]))]
        m += [(f"fz{name}NSets", str(s["opt"]["n_sets"])), (f"fz{name}NPapers", str(s["opt"]["n_papers"])),
              (f"fz{name}NWithLlm", str(s["opt"]["n_with_llm"])), (f"fz{name}NConfigs", str(s["n_configs"]))]
    m += [("fzCostJevSearch", f"{rep['costs']['a'] + rep['costs']['p']:.2f}"), ("fzCostOmp", f"{rep['costs']['omp'] + rep['costs']['omp_seeds']:.2f}"),
          ("fzCostJevTest", f"{rep['costs']['test']:.2f}")]
    head = "% Generated by jev-opt/experiments/analyze_frozen.py; do not edit.\n"
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(head + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m))
    OUT_APP.write_text(app_table(rep))


def app_table(rep: dict[str, Any]) -> str:
    """Appendix table: every frozen-configuration set, plus the original configurations' seed-0 runs."""
    one = lambda x: f"{100 * x:.1f}"  # noqa: E731
    ci = lambda d, c: f"{100 * d:+.1f} [{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]"  # noqa: E731
    llm = lambda r: "--" if r["llm"] is None else one(r["llm"])  # noqa: E731
    gaps = lambda r: f"{100 * r['gap_cal']:.0f} / {100 * r['gap_opt']:.0f}"  # noqa: E731
    metric = {"micro_f1": "micro-F1", "macro_f1": "macro-F1", "accuracy": "acc.", "qwk": "QWK", "balanced_accuracy": "bal.\\ acc."}
    L = ["% Generated by jev-opt/experiments/analyze_frozen.py; do not edit.",
         "\\begin{tabular}{@{}lccccccccccccc@{}}", "\\toprule",
         "Set & " + " & ".join(f"\\multicolumn{{1}}{{c}}{{{h}}}" for h in (
             "Metric", "$n$", "Zero", "Cal.", "$\\Delta$ cal. [95\\% CI]", "Opt.", "$\\Delta$ opt. [95\\% CI]", "Paper", "LLM",
             "Gap \\% cal./opt.", "\\jev{} \\$", "LLM \\$", "Test \\$")) + " \\\\", "\\midrule",
         "\\multicolumn{14}{@{}l}{\\emph{All eligible shortfall sets, frozen configuration, one run (seed 0)}} \\\\"]

    def row(r: dict[str, Any], costs: str) -> str:
        n = f"{r['n_test']:,}".replace(",", "{,}")
        return (f"{paper_tables.CONFIG_LABEL[r['stem']]} & {metric.get(r['metric'], r['metric'])} & {n} & {one(r['zero'])} & {one(r['calibrated'])} & "
                f"{ci(r['delta_cal'], r['ci_cal'])} & {one(r['optimized'])} & {ci(r['delta_opt'], r['ci_opt'])} & {one(r['paper'])} & "
                f"{llm(r)} & {gaps(r)} & {costs} \\\\")

    for r in rep["new"]:
        L.append(row(r, f"{(r['a_jev_fresh'] or 0) + r['jev_fresh_search']:.2f} & {r['omp'] + r['omp_seeds']:.2f} & {r['jev_test']:.2f}"))
    L += ["\\midrule", "\\multicolumn{14}{@{}l}{\\emph{Configurations chosen for large gaps, seed-0 run only (the body reports 3-seed means)}} \\\\"]
    L += [row(r, " & & ") for r in rep["orig"]]
    L += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--tex", action="store_true", help="only rewrite the macros and appendix table from results/frozen.json")
    a = ap.parse_args()
    out = load_json(OUT_JSON)
    if a.tex:
        tex(out)
        return 0
    out["selection"] = select(out.get("selection", {}))
    if not a.select:
        rep = report(out["selection"])
        out |= {k: rep[k] for k in ("new", "orig", "failures", "summary", "costs")}
        tex(rep)
    OUT_JSON.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["selection"], indent=1)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
