"""Calibration vs label budget -> results/offline_v4/calib_budget.json, paper/generated/offline_calib_budget_*.tex.

STRICTLY OFFLINE: no Jev or LLM call. Fit-slice answers come from the jev-opt cache with posting disabled
(framing_features patches JevClient: a cache miss raises) or from saved dev runs; test answers from the
saved zero-shot runs (runs/<task>/).

DEFINITIONS (fixed before results were computed)
================================================
Sets: every zero-shot test set whose *unchanged* question (same prompt) has answers on a fit slice disjoint
from test, with full option-probability vectors:
  - the 8 jev-opt study tasks: fit pool = the 400-item fit slice of runs/jev-opt/v3-<task>-A (train_roles; the
    same slice the readouts were fit on), answers of that run's seed program (the paper's question) from the
    jev-opt cache, exactly as jev-opt/experiments/confidence_readout.py calibrator_controls;
  - runs/dev_v3/{contempo_matres, smarter_hatexplain, smarter_latent_hate}: fit pool = the saved dev-split
    zero-shot run (its prompt_sha256 asserted equal to the test run's; item ids asserted disjoint from test).
  Other sets have no fit-slice answers of the unchanged question with probability vectors (dev_v3 made/ttp/wikiwd
  and biorc800/dev are not single-label option distributions) and are not analysed.
Probability vector = Jev's option (DREsS: level) probabilities renormalised over the task's labels; for the
calibrators every vector is floored at 1e-4 and renormalised (jev_opt BIAS_FLOOR), as in confidence_readout.py.
Methods, each fitted on n in {25, 50, 100, 200, 400} fit items drawn without replacement from the fit pool
(5 draws per n; numpy SeedSequence(20261001) spawned per set in SETS order, then per budget, then per draw; when
n equals the pool size, here n=400 on the jev-opt tasks, all draws are the full pool and sd is 0):
  (1) temperature only: T = jev_opt.fit.fit_temperature (fit log loss, T in [0.05, 20]); the zero-shot
      prediction is kept, only the probabilities change;
  (2) per-class bias + temperature: jev_opt.fit.fit_label_bias with the task's primary metric (and negative
      label), applied with jev_opt.execute.combine (its argmax is the prediction).
References: uncalibrated zero-shot (no fit); optimized system = the selected feature-question programs'
logistic/ridge readouts fit on the 400-item fit slice (seeds 0/1/2, results/confidence.json["rq3"], computed by
confidence_readout.py with the same statistic definitions), mean +- sd over the 3 seeds.
Test statistics: ECE equal-mass, 15 bins on P(predicted label) (scripts/confidence_zeroshot.py SetStats; ties
never split) with the perfect-calibration noise floor (mean ECE over 1,000 label draws from the method's own
probabilities); Brier = half the squared error of the full probability vector, in [0, 1]; accuracy.
Per set: mean +- sd over the 5 draws (floor: mean over draws). Across sets: median of the per-set means.
Check: the n=400 jev-opt-task fits reproduce confidence.json's temperature_scaled / bias_temperature statistics,
and the uncalibrated reference its zero_shot ECE/Brier (asserted to 1e-9).

Run from the repository root (Python 3.11):
  uv run --project jev-opt/python python jev-opt/experiments/offline_calib_budget.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import confidence_readout as cr  # noqa: E402  (imports framing_features: a cache miss raises)
import framing_features as ff  # noqa: E402

from jev_opt.adapter import train_roles  # noqa: E402
from jev_opt.fit import fit_label_bias, fit_temperature  # noqa: E402
from jev_opt.ir import load_decider  # noqa: E402

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
from confidence_zeroshot import SetStats  # noqa: E402

RUNS = ROOT / "runs"
OUT = ROOT / "results" / "offline_v4" / "calib_budget.json"
TEX = ROOT / "paper" / "generated"
BUDGETS = (25, 50, 100, 200, 400)
DRAWS = 5
SEED = 20261001
# dev_v3 sets: (dev run, test run, title, primary metric, negative label). Metrics as the paper scores the set
# (asserted present in the test run's scores.json).
DEV_SETS = [("contempo_matres", "contempo/matres", "ConTempo MATRES", "micro_f1", "VAGUE"),
            ("smarter_hatexplain", "smarter/hatexplain", "SMARTER HateXplain", "macro_f1", None),
            ("smarter_latent_hate", "smarter/latent_hate", "SMARTER Latent Hate", "macro_f1", None)]


def stats(gold: np.ndarray, pred: np.ndarray, dist: np.ndarray) -> dict[str, Any]:
    n = len(gold)
    correct = (gold == pred).astype(float)
    conf = dist[np.arange(n), pred]
    brier = 0.5 * ((dist - np.eye(dist.shape[1])[gold]) ** 2).sum(axis=1)
    st = SetStats(conf, correct, conf, brier)
    one = np.ones(n)
    return {"ece": st.ece(one), "brier": st.brier(one), "accuracy": float(correct.mean()), "_st": st}


def jevopt_set(slug: str, title: str, selection: dict) -> dict[str, Any]:
    sel0 = selection[slug]["0"]
    labels = list(cr._load_run_config(RUNS / "jev-opt" / sel0["run"]).labels)
    a_dir = RUNS / "jev-opt" / f"v3-{slug}-A"
    env_p, env_a = ff.Env(RUNS / "jev-opt" / sel0["run"]), ff.Env(a_dir)
    try:
        fit_p, _ = train_roles(env_p.train, env_p.config.optimizer)
        fit_a, _ = train_roles(env_a.train, env_a.config.optimizer)
        assert [e.id for e in fit_a] == [e.id for e in fit_p], slug
        assert list(env_a.config.labels) == labels
        res = env_a.run(load_decider((a_dir / "seed.yaml").read_text()), fit_a, "train")
        fit_dist = np.array([[max(float(r.dist.get(lab, 0.0)), 0.0) for lab in labels] for r in res])
        fit_gold = [r.gold for r in res]
        test_ids = [e.id for e in env_p.test]
        metric, neg = env_a.config.primary_metric, env_a.config.negative_label
    finally:
        env_a.close()
        env_p.close()
    zs = cr.zero_shot(slug, labels)
    assert not set(e.id for e in fit_a) & set(test_ids)
    z = [zs[i] for i in test_ids]
    return {"slug": slug, "title": title, "source": f"jev-opt cache, {a_dir.name} seed program on train_roles fit slice",
            "labels": labels, "metric": metric, "negative": neg, "fit_dist": fit_dist, "fit_gold": fit_gold,
            "test_dist": np.stack([x["dist"] for x in z]), "test_pred": [x["pred"] for x in z],
            "test_gold": [x["gold"] for x in z]}


def choice_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()]


def dev_set(dev: str, test: str, title: str, metric: str, neg: str | None) -> dict[str, Any]:
    md, mt = (json.loads((RUNS / p / "manifest.json").read_text()) for p in (f"dev_v3/{dev}", test))
    assert md["prompt_sha256"] == mt["prompt_sha256"], dev
    assert metric in json.loads((RUNS / test / "scores.json").read_text())
    fit, tst = choice_rows(RUNS / "dev_v3" / dev / "predictions.jsonl"), choice_rows(RUNS / test / "predictions.jsonl")
    assert not {r["item_id"] for r in fit} & {r["item_id"] for r in tst}
    labels = list(fit[0]["prediction"]["probabilities"])
    for r in fit + tst:
        assert set(r["prediction"]["probabilities"]) <= set(labels) and r["gold"] in labels
    vec = lambda rows: np.stack([cr.normalise(r["prediction"]["probabilities"], labels) for r in rows])  # noqa: E731
    return {"slug": dev, "title": title, "source": f"runs/dev_v3/{dev} (same prompt as runs/{test})", "labels": labels,
            "metric": metric, "negative": neg, "fit_dist": vec(fit), "fit_gold": [r["gold"] for r in fit],
            "test_dist": vec(tst), "test_pred": [r["prediction"]["label"] for r in tst], "test_gold": [r["gold"] for r in tst]}


def fit_apply(s: dict[str, Any], idx: np.ndarray, gold: np.ndarray, test_f: np.ndarray, zpred: np.ndarray) -> dict[str, dict]:
    labels = s["labels"]
    dists = [cr.floored(dict(zip(labels, row)), labels) for row in s["fit_dist"][idx]]
    fg = [s["fit_gold"][i] for i in idx]
    temp = fit_temperature(labels, dists, fg)
    bias, temp_b, _ = fit_label_bias(labels, [[d] for d in dists], fg, s["metric"], s["negative"])
    test_d = [cr.floored(dict(zip(labels, row)), labels) for row in test_f]

    def apply(b: dict[str, float], t: float) -> tuple[np.ndarray, np.ndarray]:
        # jev_opt.execute.combine per item (as confidence_readout.apply_control): log-prob biases, argmax (ties:
        # earliest label), p ** (1/T) renormalised.
        out = [cr.combine(labels, [d], b, t) for d in test_d]
        return np.array([[d[lab] for lab in labels] for d, _ in out]), np.array([labels.index(p) for _, p in out])
    tdist, _ = apply({}, temp)
    bdist, bpred = apply(bias, temp_b)
    return {"ts": stats(gold, zpred, tdist) | {"param": temp},
            "bts": stats(gold, bpred, bdist) | {"param": {"bias": bias, "temperature": temp_b}}}


def fmt(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}"


def main() -> int:
    selection = json.loads((ROOT / "results" / "seeds.json").read_text())["selection"]
    rq3 = json.loads((ROOT / "results" / "confidence.json").read_text())["rq3"]["tasks"]
    sets = [jevopt_set(slug, title, selection) for slug, title in ff.TASKS] + [dev_set(*d) for d in DEV_SETS]
    streams = np.random.SeedSequence(SEED).spawn(len(sets))
    out_sets: dict[str, Any] = {}
    for s, ss in zip(sets, streams):
        labels = s["labels"]
        li = {lab: i for i, lab in enumerate(labels)}
        gold = np.array([li[g] for g in s["test_gold"]])
        zpred = np.array([li[p] for p in s["test_pred"]])
        test_f = s["test_dist"]
        pool = len(s["fit_gold"])
        b_ss = ss.spawn(len(BUDGETS) + 1)
        frng = np.random.default_rng(b_ss[-1])
        un = stats(gold, zpred, s["test_dist"])
        row: dict[str, Any] = {"title": s["title"], "source": s["source"], "n_test": len(gold), "n_fit_pool": pool,
                               "metric": s["metric"], "negative_label": s["negative"],
                               "uncalibrated": {k: un[k] for k in ("ece", "brier", "accuracy")}
                               | {"ece_floor": un["_st"].floor(frng)["floor_mean"]}, "budgets": {}}
        for n, bss in zip(BUDGETS, b_ss[:-1]):
            if n > pool:
                continue
            per: dict[str, list] = {"ts": [], "bts": []}
            for dss in bss.spawn(DRAWS):
                idx = np.sort(np.random.default_rng(dss).choice(pool, n, replace=False))
                for k, v in fit_apply(s, idx, gold, test_f, zpred).items():
                    per[k].append({"ece": v["ece"], "brier": v["brier"], "accuracy": v["accuracy"],
                                   "ece_floor": v["_st"].floor(frng)["floor_mean"], "param": v["param"]})
            row["budgets"][str(n)] = {k: {m: {"mean": float(np.mean([d[m] for d in v])), "sd": float(np.std([d[m] for d in v], ddof=1))}
                                          for m in ("ece", "brier", "accuracy")}
                                      | {"ece_floor": float(np.mean([d["ece_floor"] for d in v])), "draws": v} for k, v in per.items()}
        if s["slug"] in rq3:
            t = rq3[s["slug"]]
            for m in ("ece", "brier"):
                assert abs(t["zero_shot"][m] - un[m]) < 1e-9, (s["slug"], m)
            if pool == 400:
                b4 = row["budgets"]["400"]
                for k, ref in (("ts", "temperature_scaled"), ("bts", "bias_temperature")):
                    for m in ("ece", "brier", "accuracy"):
                        assert abs(b4[k][m]["mean"] - t[ref][m]) < 1e-9, (s["slug"], k, m, b4[k][m]["mean"], t[ref][m])
            seeds = [t["seeds"][str(x)]["readout"] for x in ff.SEEDS]
            row["readout_400"] = {m: {"mean": float(np.mean([r[m] for r in seeds])), "sd": float(np.std([r[m] for r in seeds], ddof=1))}
                                  for m in ("ece", "brier", "accuracy")} | {
                "ece_floor": float(np.mean([f["floor_mean"] for f in t["ece_floor"]["readout_seeds"]])),
                "source": "results/confidence.json rq3 tasks.<task>.seeds (confidence_readout.py)"}
        out_sets[s["slug"]] = row
        b = row["budgets"]
        print(f"{s['slug']}: uncal ece {un['ece']:.3f} | TS ece " + " ".join(f"{n}:{b[n]['ts']['ece']['mean']:.3f}" for n in b)
              + " | BTS ece " + " ".join(f"{n}:{b[n]['bts']['ece']['mean']:.3f}" for n in b), flush=True)

    # Medians across sets of per-set means.
    def med(keyfn, names: list[str]) -> float:
        return float(np.median([keyfn(out_sets[x]) for x in names]))
    all_sets, jo = list(out_sets), [slug for slug, _ in ff.TASKS]
    medians: dict[str, Any] = {}
    for group, names in (("all", all_sets), ("jevopt8", jo)):
        g: dict[str, Any] = {"n_sets": len(names)}
        for m in ("ece", "brier", "accuracy", "ece_floor"):
            g[f"uncalibrated_{m}"] = med(lambda r: r["uncalibrated"][m], names)
            for n in BUDGETS:
                for k in ("ts", "bts"):
                    g[f"{k}_{n}_{m}"] = med(lambda r: r["budgets"][str(n)][k][m] if m == "ece_floor" else r["budgets"][str(n)][k][m]["mean"], names)
            if group == "jevopt8":
                g[f"readout_400_{m}"] = med(lambda r: r["readout_400"][m] if m == "ece_floor" else r["readout_400"][m]["mean"], names)
        medians[group] = g
    A, J = medians["all"], medians["jevopt8"]

    # Count of sets where calibration at 25 labels already lowers ECE / Brier below uncalibrated.
    cnt = lambda k, n, m: sum(out_sets[x]["budgets"][str(n)][k][m]["mean"] < out_sets[x]["uncalibrated"][m] for x in all_sets)  # noqa: E731
    readout_beats_ts_brier = sum(out_sets[x]["readout_400"]["brier"]["mean"] < out_sets[x]["budgets"]["400"]["ts"]["brier"]["mean"] for x in jo)
    readout_beats_ts_ece = sum(out_sets[x]["readout_400"]["ece"]["mean"] < out_sets[x]["budgets"]["400"]["ts"]["ece"]["mean"] for x in jo)
    macros = {"ovCbNSets": str(len(all_sets)), "ovCbNOpt": str(len(jo)), "ovCbDraws": str(DRAWS),
              "ovCbEceUncal": fmt(A["uncalibrated_ece"]), "ovCbBrierUncal": fmt(A["uncalibrated_brier"]),
              "ovCbEceFloorUncal": fmt(A["uncalibrated_ece_floor"]),
              "ovCbTsEceTwentyFiveSetsLower": str(cnt("ts", 25, "ece")), "ovCbTsBrierTwentyFiveSetsLower": str(cnt("ts", 25, "brier")),
              "ovCbOptEceUncal": fmt(J["uncalibrated_ece"]), "ovCbOptBrierUncal": fmt(J["uncalibrated_brier"]),
              "ovCbOptReadoutEce": fmt(J["readout_400_ece"]), "ovCbOptReadoutBrier": fmt(J["readout_400_brier"]),
              "ovCbOptReadoutAcc": fmt(100 * J["readout_400_accuracy"], 1),
              "ovCbOptReadoutBeatsTsBrier": str(readout_beats_ts_brier), "ovCbOptReadoutBeatsTsEce": str(readout_beats_ts_ece)}
    words = {25: "TwentyFive", 50: "Fifty", 100: "Hundred", 200: "TwoHundred", 400: "FourHundred"}
    for n in BUDGETS:
        for k, kw in (("ts", "Ts"), ("bts", "Bts")):
            macros[f"ovCb{kw}Ece{words[n]}"] = fmt(A[f"{k}_{n}_ece"])
            macros[f"ovCb{kw}Brier{words[n]}"] = fmt(A[f"{k}_{n}_brier"])
            macros[f"ovCb{kw}Acc{words[n]}"] = fmt(100 * A[f"{k}_{n}_accuracy"], 1)
            macros[f"ovCbOpt{kw}Ece{words[n]}"] = fmt(J[f"{k}_{n}_ece"])
            macros[f"ovCbOpt{kw}Brier{words[n]}"] = fmt(J[f"{k}_{n}_brier"])
    macros["ovCbAccUncal"] = fmt(100 * A["uncalibrated_accuracy"], 1)

    # Tables.
    def pm(d: dict) -> str:
        return f"{d['mean']:.3f}" if d["sd"] < 5e-4 else f"{d['mean']:.3f}$\\pm${d['sd']:.3f}"
    lines = ["% Generated by jev-opt/experiments/offline_calib_budget.py. Do not edit.", "\\begin{tabular}{lrrrrrr}", "\\toprule",
             "& \\multicolumn{3}{c}{All %d sets} & \\multicolumn{3}{c}{%d optimized tasks} \\\\" % (len(all_sets), len(jo)),
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}", "Method, labels & ECE & Brier & Acc. & ECE & Brier & Acc. \\\\", "\\midrule"]
    r3 = lambda g, p: f"{fmt(g[p + '_ece'])} & {fmt(g[p + '_brier'])} & {fmt(100 * g[p + '_accuracy'], 1)}"  # noqa: E731
    lines.append(f"Zero-shot, uncalibrated & {r3(A, 'uncalibrated')} & {r3(J, 'uncalibrated')} \\\\")
    for k, name in (("ts", "Temperature"), ("bts", "Bias + temperature")):
        lines.append("\\midrule")
        for n in BUDGETS:
            lines.append(f"{name}, {n} & {r3(A, f'{k}_{n}')} & {r3(J, f'{k}_{n}')} \\\\")
    lines += ["\\midrule", f"Optimized readout, 400 & -- & -- & -- & {r3(J, 'readout_400')} \\\\", "\\bottomrule", "\\end{tabular}"]
    TEX.mkdir(parents=True, exist_ok=True)
    (TEX / "offline_calib_budget_medians.tex").write_text("\n".join(lines) + "\n")

    lines = ["% Generated by jev-opt/experiments/offline_calib_budget.py. Do not edit.", "\\begin{tabular}{lrrrrrrr}", "\\toprule",
             "& \\multicolumn{2}{c}{Uncalibrated} & \\multicolumn{2}{c}{Temp., 25} & \\multicolumn{2}{c}{Bias + temp., 25} & Readout \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-8}",
             "Set & ECE (floor) & Brier & ECE & Brier & ECE & Brier & Brier, 400 \\\\", "\\midrule"]
    for x, r in out_sets.items():
        b = r["budgets"]["25"]
        ro = pm(r["readout_400"]["brier"]) if "readout_400" in r else "--"
        lines.append(f"{r['title']} & {fmt(r['uncalibrated']['ece'])} ({fmt(r['uncalibrated']['ece_floor'])}) & {fmt(r['uncalibrated']['brier'])} & "
                     f"{pm(b['ts']['ece'])} & {pm(b['ts']['brier'])} & {pm(b['bts']['ece'])} & {pm(b['bts']['brier'])} & {ro} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TEX / "offline_calib_budget_sets.tex").write_text("\n".join(lines) + "\n")

    md = [f"## Calibration vs label budget\n",
          f"Sets with fit-slice answers of the unchanged question: {len(all_sets)} ({len(jo)} jev-opt tasks, 400-item fit slice; "
          f"{len(DEV_SETS)} dev_v3 sets). {DRAWS} seeded draws per budget; medians across sets of per-set means. "
          "ECE equal-mass 15 bins on P(pred); Brier half squared error of the full vector.\n",
          "| Method | labels | ECE (all) | Brier (all) | Acc (all) | ECE (8 opt) | Brier (8 opt) | Acc (8 opt) |", "|---|---|---|---|---|---|---|---|"]
    rmd = lambda g, p: f"{fmt(g[p + '_ece'])} | {fmt(g[p + '_brier'])} | {fmt(100 * g[p + '_accuracy'], 1)}"  # noqa: E731
    md.append(f"| uncalibrated | 0 | {rmd(A, 'uncalibrated')} | {rmd(J, 'uncalibrated')} |")
    for k, name in (("ts", "temperature"), ("bts", "bias+temperature")):
        for n in BUDGETS:
            md.append(f"| {name} | {n} | {rmd(A, f'{k}_{n}')} | {rmd(J, f'{k}_{n}')} |")
    md.append(f"| optimized readout | 400 | – | – | – | {rmd(J, 'readout_400')} |")
    md.append(f"\nUncalibrated median ECE noise floor {fmt(A['uncalibrated_ece_floor'])}. Temperature at 25 labels lowers ECE on "
              f"{cnt('ts', 25, 'ece')}/{len(all_sets)} sets and Brier on {cnt('ts', 25, 'brier')}/{len(all_sets)}. "
              f"Readout (400) beats temperature (400) on Brier in {readout_beats_ts_brier}/{len(jo)} and on ECE in {readout_beats_ts_ece}/{len(jo)} tasks.\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"generated_by": "jev-opt/experiments/offline_calib_budget.py", "budgets": BUDGETS, "draws": DRAWS,
                               "seed": SEED, "macros": macros, "md": "\n".join(md), "medians": medians, "sets": out_sets}, indent=1) + "\n")
    print("\n".join(md))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
