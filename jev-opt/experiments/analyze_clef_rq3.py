"""RQ3 optimization step on Clef / Clef-flash answers (pre-registered) -> results/CLEF_RQ3.{md,json},
paper/generated/clef_rq3.tex (macros \\clFlash… / \\clClef… / \\clJev…) and paper/generated/app_clef_rq3.tex
(appendix tables). No inference: reads the cl-<model>-<set>-{cal,opt} run directories, the models' frozen
zero-shot runs, results/frozen.json, results/seeds.json, results/gepa_v3_test.json and the saved test
predictions of results/budget_baseline/ (Jev's pipeline and the same-label trained baselines). Jev's zero-shot and
calibrated test predictions are rebuilt offline from the saved zero-shot responses and the calibration recorded
in results/frozen.json (checked against its scores). A model without finished sets is reported as pending.

Post-hoc additions (after the test results, not in the pre-registered analysis plan): gap medians by group (the 8
configurations, the 5 frozen sets, the 8 without DREsS) next to Jev's seed-0 programs; the optimized step against
Jev's seed-0 pipeline and Jev's calibration step, paired with the same resamples; the optimized step against the
same-label best trained baseline with budget_report.py's resampling, small-cluster rule and Holm correction; the gap
closed against a common zero (Jev's frozen zero-shot) and the own-zero gap on the sets outside DREsS.

  uv run --project jev-opt/python python jev-opt/experiments/analyze_clef_rq3.py [--report-only]
(--report-only rewrites the .md and .tex outputs from results/CLEF_RQ3.json without recomputing.)
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_frozen import apply_cal, gauntlet_zero  # noqa: E402
from budget_report import Scorer  # noqa: E402
from clef_rq3 import PINS, RUNS, SETS, cal_dir, frozen_rows, opt_dir, run_config  # noqa: E402
from confidence_readout import zero_shot as jev_zero_shot  # noqa: E402
from offline_rq3 import boot_p as bb_boot_p  # noqa: E402
from resample import cluster_index, jackknife_t, resamples  # noqa: E402

from jev_opt import metrics  # noqa: E402

ROOT = HERE.parents[1]
OUT_JSON = ROOT / "results" / "CLEF_RQ3.json"
OUT_MD = ROOT / "results" / "CLEF_RQ3.md"
OUT_TEX = ROOT / "paper" / "generated" / "clef_rq3.tex"
OUT_APP = ROOT / "paper" / "generated" / "app_clef_rq3.tex"
BB_DIR = ROOT / "results" / "budget_baseline"
B = 2000
B_ENC = 10_000  # budget_report.py's resample count and chunking for the trained-baseline comparison
CHUNK = 500
TOL = 0.005
SMALL = 30  # clustered sets with fewer test clusters use the jackknife-t p (the paper's rule)
EXCLUDED_TBD_TEST = "40.2"  # test micro-F1 (x100) of the excluded RTX 4090 Clef-flash TB-Dense calibration step
MODELS = {"clef-flash": ("Clef-flash", "Flash"), "clef": ("Clef", "Clef")}
CFG8 = list(SETS)[:8]  # the 8 configurations (results/seeds.json); the rest are the 5 frozen sets
GROUPS = {"cfg8": CFG8, "frozen5": list(SETS)[8:], "cfg8_no_dress": [s for s in CFG8 if not s.startswith("dress-")],
          "no_tbdense": [s for s in SETS if s != "contempo-tbdense"]}
FAM_NAME = {"emb": "embeddings + LR", "setfit": "SetFit", "ft": "fine-tuned encoder"}
TITLE = {
    "contempo-tbdense": ("TB-Dense", "Tbdense"), "ezstance-subtaskA": ("EZ-STANCE A", "Ezstance"),
    "smarter-implicit-hate": ("Implicit Hate", "Implicit"), "edit-intent": ("Edit intent", "Edit"), "inli": ("INLI", "Inli"),
    "dress-new-content": ("DREsS content", "DressC"), "dress-new-organization": ("DREsS organization", "DressO"),
    "dress-new-language": ("DREsS language", "DressL"), "contempo-matres": ("MATRES", "Matres"), "lifetox": ("LifeTox", "Lifetox"),
    "sciprompt-s2orc": ("SciPrompt S2ORC", "Storc"), "smarter-hatexplain": ("HateXplain", "Hatexplain"),
    "smarter-latent-hate": ("Latent Hatred", "Latent"),
}
CONTRASTS = [("cal-zero", "cal", "zero"), ("opt-zero", "opt", "zero"), ("opt-cal", "opt", "cal"), ("cal-seed", "cal", "seed"),
             ("opt-jev", "opt", "jev"), ("jevcal-jevzero", "jevcal", "jevzero")]
_JEV_CACHE: dict[str, dict[str, list[str]]] = {}


def boot_p(d: np.ndarray) -> float:
    return float(min(1.0, 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1))))


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [0.0] * len(ps)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def jev_rows() -> dict[str, dict[str, Any]]:
    frozen = json.loads((ROOT / "results" / "frozen.json").read_text())
    return {r["stem"]: r for r in frozen["new"] + frozen["orig"]}


def jev_preds(stem: str, ids: list[str], gold: list[str], labels: list[str], score, jev: dict[str, Any],
              jtp: dict[str, Any]) -> dict[str, list[str]]:
    """Jev's seed-0 pipeline (the transferred program), zero-shot and calibrated test predictions, in `ids` order."""
    if stem not in _JEV_CACHE:
        J = jtp[stem]
        pos = {i: k for k, i in enumerate(J["ids"])}
        assert sorted(pos) == sorted(ids) and [J["gold"][pos[i]] for i in ids] == gold, stem
        pipe = [J["runs"][SETS[stem][0]]["pred"][pos[i]] for i in ids]
        zs = jev_zero_shot(stem, labels) if stem in CFG8 else gauntlet_zero(SETS[stem][1], labels)
        z = [zs[i] for i in ids]
        assert [x["gold"] for x in z] == gold, stem
        out = {"jev": pipe, "jevzero": [x["pred"] for x in z], "jevcal": apply_cal(z, labels, jev["calibration"])}
        for k, ref in (("jev", "optimized"), ("jevzero", "zero"), ("jevcal", "calibrated")):
            assert abs(score(gold, out[k]) - jev[ref]) < 1e-9, (stem, k, score(gold, out[k]), jev[ref])
        _JEV_CACHE[stem] = out
    return _JEV_CACHE[stem]


def vs_encoder(stem: str, cfg, ids: list[str], gold: list[str], pred: list[str], bb: dict[str, Any], J: dict[str, Any]) -> dict[str, Any]:
    """Optimized step − the best same-label trained baseline (budget_report.py: its best family on val, its test seeds
    resampled for the seed mean, the same cluster resampling with B_ENC draws from default_rng(0), jackknife-t p on
    clustered sets with fewer than 30 clusters)."""
    fam = bb["best"]
    rec = json.loads((BB_DIR / fam / f"{stem}.json").read_text())
    pos = {i: k for k, i in enumerate(ids)}
    jids, jgold = J["ids"], J["gold"]
    p = [pred[pos[i]] for i in jids]
    d = SimpleNamespace(metric=cfg.primary_metric, labels=list(cfg.labels), neg=cfg.negative_label)
    sc = Scorer(d, jgold)
    obs_c = metrics.primary(d.metric, jgold, p, d.labels, d.neg)
    ones = np.ones((1, len(jids)))
    oh_c, oh_f = sc.onehot(p), [sc.onehot(x) for x in rec["test_pred"]]
    assert abs(sc.score(oh_c, ones)[0] - obs_c) < 1e-9
    for oh, t in zip(oh_f, rec["test"]):
        assert abs(sc.score(oh, ones)[0] - t) < 1e-9, (stem, fam)
    inv, unit = cluster_index(stem, jids)
    m = int(inv.max()) + 1
    rng = np.random.default_rng(0)
    SC, SF = np.empty(B_ENC), np.empty((len(oh_f), B_ENC))
    for lo in range(0, B_ENC, CHUNK):
        b = min(CHUNK, B_ENC - lo)
        draw = rng.integers(0, m, size=(b, m))
        w = np.bincount((draw + m * np.arange(b)[:, None]).ravel(), minlength=b * m).reshape(b, m)[:, inv].astype(float)
        SC[lo:lo + b] = sc.score(oh_c, w)
        for i, oh in enumerate(oh_f):
            SF[i, lo:lo + b] = sc.score(oh, w)
    pick = rng.integers(0, len(oh_f), size=(B_ENC, len(oh_f)))
    D = SC - SF[pick, np.arange(B_ENC)[:, None]].mean(1)
    base = statistics.mean(rec["test"])
    out = {"family": fam, "baseline_test_mean": base, "diff": obs_c - base,
           "ci": [float(x) for x in np.percentile(D, [2.5, 97.5])], "p_boot": bb_boot_p(D), "draws": D}
    if unit != "item" and m < SMALL:
        def stat(ix):
            g = [jgold[i] for i in ix]
            return (metrics.primary(d.metric, g, [p[i] for i in ix], d.labels, d.neg)
                    - statistics.mean(metrics.primary(d.metric, g, [x[i] for i in ix], d.labels, d.neg) for x in rec["test_pred"]))
        lo, hi = jackknife_t(stat, inv)
        se = (hi - lo) / (2 * student_t.ppf(0.975, m - 1))
        out["jackknife_ci"] = [lo, hi]
        out["p_jackknife"] = float(2 * student_t.sf(abs(out["diff"]) / se, m - 1))
    out["p"] = out.get("p_jackknife", out["p_boot"])
    return out


def set_row(model: str, stem: str, jev: dict[str, Any], jtp: dict[str, Any], bb: dict[str, Any]) -> dict[str, Any] | None:
    cal_p, opt_p = cal_dir(model, stem) / "result.json", opt_dir(model, stem) / "result.json"
    if not ((cal_dir(model, stem) / "test_consumed.json").exists() and (opt_dir(model, stem) / "test_consumed.json").exists()):
        return None
    cal, opt = json.loads(cal_p.read_text()), json.loads(opt_p.read_text())
    cfg = run_config(RUNS / SETS[stem][0])
    labels = list(cfg.labels)
    score = lambda g, p: metrics.primary(cfg.primary_metric, g, p, labels, cfg.negative_label)  # noqa: E731
    ids, gold = cal["test"]["ids"], cal["test"]["gold"]
    assert opt["test"]["ids"] == ids and opt["test"]["gold"] == gold
    rows = frozen_rows(model, stem)
    preds = {"zero": [rows[i]["pred"] for i in ids], "cal": cal["test"]["calibrated_pred"], "opt": opt["test"]["pred"]}
    if cal["test"]["seed_pred"] is not None:
        preds["seed"] = cal["test"]["seed_pred"]
    preds.update(jev_preds(stem, ids, gold, labels, score, jev, jtp))
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
    contrasts = {}
    for name, a, z in CONTRASTS:
        if a not in preds or z not in preds:
            continue
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
        contrasts[name] = c
    zero, paper = point["zero"], jev["paper"]
    gap = (lambda x: None if paper - zero <= TOL else (x - zero) / (paper - zero))
    sel = opt["selection"]
    rej = len(cal["rejected"]) + len(opt["rejected"])
    usage = {"cal": cal["usage"], "opt": opt["usage"]}
    fresh_s = sum(v.get("fresh_inference_s", 0.0) for u in usage.values() for v in u.values())
    tokens = sum(v.get("fresh_input_tokens", 0) for u in usage.values() for v in u.values())
    return {"set": stem, "title": TITLE[stem][0], "metric": cfg.primary_metric, "n_test": len(ids), "cluster_unit": unit, "n_clusters": m,
            "zero": zero, "seed_zero": point.get("seed"), "calibrated": point["cal"], "optimized": point["opt"],
            "val": {"seed": cal["val"]["seed_primary"], "calibrated": cal["val"]["calibrated_primary"], **sel["val"]},
            "chosen": sel["chosen"], "jev_chosen": sel["jev_chosen"], "test_distributions_cal": cal["test_distributions"],
            "paper": paper, "llm": jev["llm"], "jev": {"zero": jev["zero"], "calibrated": jev["calibrated"], "optimized": jev["optimized"],
                                                      "gap_cal": jev["gap_cal"], "gap_opt": jev["gap_opt"]},
            "gap_cal": gap(point["cal"]), "gap_opt": gap(point["opt"]),
            "reach_method_cal": point["cal"] >= paper - TOL, "reach_method_opt": point["opt"] >= paper - TOL,
            "reach_llm_opt": None if jev["llm"] is None else point["opt"] >= jev["llm"] - TOL,
            "opt_minus_jev": point["opt"] - jev["optimized"], "rejected_examples": rej,
            "fresh_inference_s": fresh_s, "fresh_input_tokens": tokens, "usage": usage, "contrasts": contrasts,
            "vs_encoder": vs_encoder(stem, cfg, ids, gold, preds["opt"], bb[stem], jtp[stem])}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_sets": len(rows)}
    for name, *_ in CONTRASTS:
        have = [r for r in rows if name in r["contrasts"]]
        if not have:
            continue
        for p, r in zip(holm([r["contrasts"][name]["p"] for r in have]), have):
            r["contrasts"][name]["p_holm"] = p
        pooled = np.mean([r["contrasts"][name]["draws"] for r in have], axis=0)
        out[name] = {"n": len(have), "pooled_delta": statistics.mean(r["contrasts"][name]["delta"] for r in have),
                     "pooled_ci": [float(x) for x in np.percentile(pooled, [2.5, 97.5])], "pooled_p": boot_p(pooled),
                     "n_ci_above_zero": sum(r["contrasts"][name]["ci"][0] > 0 for r in have),
                     "n_ci_below_zero": sum(r["contrasts"][name]["ci"][1] < 0 for r in have),
                     "n_holm_positive": sum(r["contrasts"][name]["p_holm"] < 0.05 and r["contrasts"][name]["delta"] > 0 for r in have),
                     "n_holm_negative": sum(r["contrasts"][name]["p_holm"] < 0.05 and r["contrasts"][name]["delta"] < 0 for r in have),
                     "n_positive": sum(r["contrasts"][name]["delta"] > 0 for r in have),
                     "median_delta": statistics.median(r["contrasts"][name]["delta"] for r in have)}
    for k in ("gap_cal", "gap_opt"):
        v = [r[k] for r in rows if r[k] is not None]
        out[f"median_{k}"] = statistics.median(v) if v else None
        out[f"n_{k}"] = len(v)
        for g, stems in GROUPS.items():
            v = [r[k] for r in rows if r["set"] in stems and r[k] is not None]
            out[f"median_{k}_{g}"] = statistics.median(v) if v else None
    out["over_100"] = [r["title"] for r in rows if r["gap_opt"] is not None and r["gap_opt"] > 1]
    enc = [r["vs_encoder"] for r in rows]
    for p, e in zip(holm([e["p"] for e in enc]), enc):
        e["p_holm"] = p
        e["reading"] = "model" if p < 0.05 and e["diff"] > 0 else "baseline" if p < 0.05 and e["diff"] < 0 else "unclear"
    noT = [r["vs_encoder"] for r in rows if r["set"] in GROUPS["no_tbdense"]]
    pooled = np.mean([e["draws"] for e in noT], axis=0)
    out["vs_encoder"] = {
        "n": len(enc), "n_model_ahead": sum(e["reading"] == "model" for e in enc), "n_baseline_ahead": sum(e["reading"] == "baseline" for e in enc),
        "n_unclear": sum(e["reading"] == "unclear" for e in enc), "median_diff": statistics.median(e["diff"] for e in enc),
        "no_tbdense": {"n": len(noT), "n_model_ahead": sum(e["reading"] == "model" for e in noT),
                       "n_baseline_ahead": sum(e["reading"] == "baseline" for e in noT), "n_unclear": sum(e["reading"] == "unclear" for e in noT),
                       "n_positive": sum(e["diff"] > 0 for e in noT), "median_diff": statistics.median(e["diff"] for e in noT),
                       "pooled_diff": statistics.mean(e["diff"] for e in noT), "pooled_ci": [float(x) for x in np.percentile(pooled, [2.5, 97.5])]},
        "tbdense": next(({k: v for k, v in r["vs_encoder"].items() if k != "draws"} for r in rows if r["set"] == "contempo-tbdense"), None)}
    out["n_reach_method_opt"] = sum(r["reach_method_opt"] for r in rows)
    out["n_chosen_as_jev"] = sum(r["chosen"] == r["jev_chosen"] for r in rows)
    out["median_opt_minus_jev"] = statistics.median(r["opt_minus_jev"] for r in rows) if rows else None
    out["n_opt_above_jev"] = sum(r["opt_minus_jev"] > 0 for r in rows)
    out["fresh_inference_s"] = sum(r["fresh_inference_s"] for r in rows)
    out["fresh_input_tokens"] = sum(r["fresh_input_tokens"] for r in rows)
    return out


def pct(x: float | None, signed: bool = False) -> str:
    return "–" if x is None else (f"{100 * x:+.1f}" if signed else f"{100 * x:.1f}")


def ipct(x: float) -> str:
    return f"{100 * x:.0f}"


def ci(c: dict[str, Any], key: str = "delta") -> str:
    return f"{pct(c[key], True)} [{pct(c['ci'][0], True)}, {pct(c['ci'][1], True)}] p_Holm {c['p_holm']:.3f}"


def jev_md(rep: dict[str, Any]) -> list[str]:
    J = rep["jev"]
    g0, g3, gf = J["median_gap_opt_seed0"], J["median_gap_opt_mean3"], J["median_gap_opt_frozen_zero"]
    L = ["## Comparison with Jev (post hoc)", "",
         "Median gap closed by the optimized step. The models run Jev's optimizer-seed-0 programs. Jev seed 0: the 8",
         "configurations as in Table 3 (zero = the paper's question through the optimizer's request path, `results/seeds.json`,",
         "paper = `results/gepa_v3_test.json`), the 5 frozen sets from `results/frozen.json`. Jev 3-seed mean: the paper's",
         "headline for the 8. The models' zero is their frozen zero-shot audit (for DREsS without Jev's Score cutoffs).", "",
         "| System | 8 configurations | 8 without DREsS | 5 frozen sets | All 13 |", "|---|---:|---:|---:|---:|"]
    for model, (name, _) in MODELS.items():
        s = rep[model]["summary"]
        L.append(f"| {name} (seed-0 programs) | {pct(s['median_gap_opt_cfg8'])}% | {pct(s['median_gap_opt_cfg8_no_dress'])}% | "
                 f"{pct(s['median_gap_opt_frozen5'])}% | {pct(s['median_gap_opt'])}% |")
    L += [f"| Jev seed 0 | {pct(g0['cfg8'])}% | {pct(g0['cfg8_no_dress'])}% | {pct(g0['frozen5'])}% | {pct(g0['all'])}% |",
          f"| Jev seed 0, zero = frozen audit | {pct(gf['cfg8'])}% | {pct(gf['cfg8_no_dress'])}% | {pct(gf['frozen5'])}% | {pct(gf['all'])}% |",
          f"| Jev 3-seed mean (paper) | {pct(g3['cfg8'])}% | {pct(g3['cfg8_no_dress'])}% | – | – |", ""]
    for model, (name, _) in MODELS.items():
        over = rep[model]["summary"]["over_100"]
        L.append(f"- {name}: gap share above 100% on {', '.join(over) if over else 'no set'} (the model's optimized score passes the "
                 "paper's method; on DREsS the paper's model is on the original splits and Jev's DREsS tasks are rebuilt).")
    L += ["", "Calibration step (bias + temperature on the paper's question; paired cluster bootstrap, Holm over 13 sets, "
          "jackknife-t p on TB-Dense and MATRES):", "",
          "| Model | Δ > 0 | Holm-significant + / − | Median Δ | Mean Δ | Median gap closed |", "|---|---:|---:|---:|---:|---:|"]
    if "cal" in J:
        c = J["cal"]
        L.append(f"| Jev | {c['n_positive']} | {c['n_holm_positive']} / {c['n_holm_negative']} | {pct(c['median_delta'], True)} | "
                 f"{pct(c['pooled_delta'], True)} | {pct(J['median_gap_cal_frozen_zero']['all'])}% |")
    for model, (name, _) in MODELS.items():
        x, s = rep[model]["summary"]["cal-zero"], rep[model]["summary"]
        L.append(f"| {name} | {x['n_positive']} | {x['n_holm_positive']} / {x['n_holm_negative']} | {pct(x['median_delta'], True)} | "
                 f"{pct(x['pooled_delta'], True)} | {pct(s['median_gap_cal'])}% |")
    if "cal" in J:
        L += ["", "| Set | Jev zero-shot | Jev calibrated | Δ [CI] |", "|---|---:|---:|---|"]
        rows = {r["set"]: r for r in rep["clef"]["sets"]}
        for st, c in J["cal"]["sets"].items():
            L.append(f"| {rows[st]['title']} | {pct(rows[st]['jev']['zero'])} | {pct(rows[st]['jev']['calibrated'])} | {ci(c)} |")
    return L + [""]


def compare_md(name: str, rows: list[dict[str, Any]], s: dict[str, Any]) -> list[str]:
    x, e = s["opt-jev"], s["vs_encoder"]
    nt = e["no_tbdense"]
    L = [f"### {name}: optimized step against Jev's seed-0 pipeline and the same-label trained baseline (post hoc)", "",
         "Jev pipeline: Jev's seed-0 program (the program the model answers), paired with the 2,000 resamples above. Trained: the",
         "best of the three same-label baselines on validation (`results/BUDGET_BASELINE.json`), mean over its 3 seeds, compared as",
         "in `budget_report.py` (10,000 cluster resamples from seed 0, baseline seeds resampled, jackknife-t p on TB-Dense and",
         "MATRES, Holm over 13 sets).", "",
         "| Set | Optimized | Jev pipeline | Opt. − Jev [CI] | Trained | Opt. − trained [CI] | Reading |", "|---|---:|---:|---|---|---|---|"]
    for r in rows:
        v = r["vs_encoder"]
        jk = f" (jackknife {pct(v['jackknife_ci'][0], True)}…{pct(v['jackknife_ci'][1], True)})" if "jackknife_ci" in v else ""
        L.append(f"| {r['title']} | {pct(r['optimized'])} | {pct(r['jev']['optimized'])} | {ci(r['contrasts']['opt-jev'])} | "
                 f"{pct(v['baseline_test_mean'])} ({FAM_NAME[v['family']]}) | {ci(v, 'diff')}{jk} | {v['reading']} |")
    L += ["", f"- Optimized − Jev pipeline: pooled {pct(x['pooled_delta'], True)} [{pct(x['pooled_ci'][0], True)}, {pct(x['pooled_ci'][1], True)}]; "
          f"above Jev on {x['n_positive']} of {x['n']}; Holm-significant higher {x['n_holm_positive']}, lower {x['n_holm_negative']}; "
          f"median {pct(x['median_delta'], True)}.",
          f"- Optimized − trained: clearly ahead {e['n_model_ahead']}, behind {e['n_baseline_ahead']}, unclear {e['n_unclear']} of {e['n']} "
          f"(median {pct(e['median_diff'], True)}). Without TB-Dense: ahead {nt['n_model_ahead']}, behind {nt['n_baseline_ahead']}, "
          f"unclear {nt['n_unclear']} of {nt['n']}; above in point estimate on {nt['n_positive']}; median {pct(nt['median_diff'], True)}; "
          f"mean {pct(nt['pooled_diff'], True)} [{pct(nt['pooled_ci'][0], True)}, {pct(nt['pooled_ci'][1], True)}].", ""]
    return L


def write_md(rep: dict[str, Any]) -> None:
    L = ["# RQ3 optimization step on Clef and Clef-flash answers", "",
         "The study was pre-registered. Generated by `jev-opt/experiments/analyze_clef_rq3.py`.",
         "Scores ×100 on each paper's primary metric over the gauntlet test items. Zero-shot = the model's frozen",
         "zero-shot audit; seed zero-shot = the paper's question sent as the jev-opt seed program (where its request",
         "differs from the frozen one). Calibrated = bias + temperature fitted on the model's fit-slice answers.",
         "Optimized = Jev's selected questions answered by the model, readout refitted on its answers, P vs P+head on its val.",
         "Δ vs the model's zero-shot: paired cluster bootstrap (RQ1 units, 2,000 resamples), 95% CI, Holm over the sets",
         "(jackknife-t p on TB-Dense and MATRES). Gap = share of the zero-shot-to-paper gap closed.", ""]
    for model, (name, _) in MODELS.items():
        rows, s = rep[model]["sets"], rep[model]["summary"]
        L += [f"## {name} (`{PINS[model]}`)", ""]
        if not rows:
            L += [f"Pending: no finished set ({rep[model]['status']}).", ""]
            continue
        L += ["| Set | Zero-shot | Seed z.-s. | Calibrated | Δ cal. [CI] | Optimized | Δ opt. [CI] | Chosen (val) | Paper | Jev pipeline | Gap cal. / opt. |",
              "|---|---:|---:|---:|---|---:|---|---|---:|---:|---|"]
        for r in rows:
            c = r["contrasts"]
            L.append(f"| {r['title']} | {pct(r['zero'])} | {pct(r['seed_zero'])} | {pct(r['calibrated'])} | {ci(c['cal-zero'])} | "
                     f"{pct(r['optimized'])} | {ci(c['opt-zero'])} | {r['chosen']} (Jev: {r['jev_chosen']}) | {pct(r['paper'])} | "
                     f"{pct(r['jev']['optimized'])} | {pct(r['gap_cal'])}% / {pct(r['gap_opt'])}% |")
        L += [""]
        for name_, label in (("cal-zero", "Calibrated − zero-shot"), ("opt-zero", "Optimized − zero-shot"), ("opt-cal", "Optimized − calibrated"),
                             ("cal-seed", "Calibrated − seed zero-shot")):
            if name_ in s:
                x = s[name_]
                L.append(f"- {label}: pooled {pct(x['pooled_delta'], True)} [{pct(x['pooled_ci'][0], True)}, {pct(x['pooled_ci'][1], True)}] "
                         f"over {x['n']} sets; Δ > 0 on {x['n_positive']}; CI above 0 on {x['n_ci_above_zero']}, below 0 on {x['n_ci_below_zero']}; "
                         f"Holm-significant positive {x['n_holm_positive']}, negative {x['n_holm_negative']}.")
        L += [f"- Median gap closed: calibrated {pct(s['median_gap_cal'])}%, optimized {pct(s['median_gap_opt'])}% "
              f"({s['n_gap_opt']} sets with a gap); optimized on the 8 configurations {pct(s['median_gap_opt_cfg8'])}% "
              f"({pct(s['median_gap_opt_cfg8_no_dress'])}% without DREsS), on the 5 frozen sets {pct(s['median_gap_opt_frozen5'])}%.",
              f"- Optimized reaches the paper's method on {s['n_reach_method_opt']} of {s['n_sets']} sets; selection equals Jev's on "
              f"{s['n_chosen_as_jev']}; median optimized − Jev pipeline {pct(s['median_opt_minus_jev'], True)}.",
              f"- Compute: {s['fresh_input_tokens']:,} new input tokens, {s['fresh_inference_s'] / 3600:.1f} GPU-hours of new inference.", ""]
        L += ["| Set | Val seed / cal. / P / P+head | Rejected examples | New tokens | GPU s |", "|---|---|---:|---:|---:|"]
        for r in rows:
            v = r["val"]
            L.append(f"| {r['title']} | {pct(v['seed'])} / {pct(v['calibrated'])} / {pct(v.get('top-single'))} / {pct(v.get('top-stacked'))} | "
                     f"{r['rejected_examples']} | {r['fresh_input_tokens']:,} | {r['fresh_inference_s']:.0f} |")
        L += [""]
        L += compare_md(name, rows, s)
    if "jev" in rep:
        L += jev_md(rep)
    OUT_MD.write_text("\n".join(L) + "\n")


def write_tex(rep: dict[str, Any]) -> None:
    m: list[tuple[str, str]] = []
    for model, (_, tag) in MODELS.items():
        rows, s = rep[model]["sets"], rep[model]["summary"]
        p = f"cl{tag}"
        m.append((f"{p}Status", "finished" if len(rows) == len(SETS) else ("pending" if not rows else f"{len(rows)} of {len(SETS)} sets")))
        if not rows:
            continue
        m.append((f"{p}Nsets", str(len(rows))))
        for r in rows:
            t = TITLE[r["set"]][1]
            m += [(f"{p}Zero{t}", pct(r["zero"])), (f"{p}Cal{t}", pct(r["calibrated"])), (f"{p}Opt{t}", pct(r["optimized"])),
                  (f"{p}Dcal{t}", pct(r["contrasts"]["cal-zero"]["delta"], True)), (f"{p}Dopt{t}", pct(r["contrasts"]["opt-zero"]["delta"], True))]
        for name, key in (("cal-zero", "Cal"), ("opt-zero", "Opt"), ("opt-cal", "OptCal"), ("opt-jev", "OptJev")):
            x = s[name]
            m += [(f"{p}Pool{key}", pct(x["pooled_delta"], True)), (f"{p}Pool{key}Lo", pct(x["pooled_ci"][0], True)),
                  (f"{p}Pool{key}Hi", pct(x["pooled_ci"][1], True)), (f"{p}Npos{key}", str(x["n_positive"])),
                  (f"{p}Nholm{key}", str(x["n_holm_positive"])), (f"{p}Nneg{key}", str(x["n_holm_negative"])),
                  (f"{p}Med{key}", pct(x["median_delta"], True))]
        e, nt = s["vs_encoder"], s["vs_encoder"]["no_tbdense"]
        m += [(f"{p}MedGapCal", pct(s["median_gap_cal"])), (f"{p}MedGapOpt", pct(s["median_gap_opt"])),
              (f"{p}GapAll", ipct(s["median_gap_opt"])), (f"{p}GapCalAll", ipct(s["median_gap_cal"])),
              (f"{p}GapCfg", ipct(s["median_gap_opt_cfg8"])), (f"{p}GapCfgNoDress", ipct(s["median_gap_opt_cfg8_no_dress"])),
              (f"{p}GapFrozen", ipct(s["median_gap_opt_frozen5"])), (f"{p}NoverHundred", str(len(s["over_100"]))),
              (f"{p}NreachOpt", str(s["n_reach_method_opt"])), (f"{p}MedOptMinusJev", pct(s["median_opt_minus_jev"], True)),
              (f"{p}NsameChoice", str(s["n_chosen_as_jev"])),
              (f"{p}EncAhead", str(e["n_model_ahead"])), (f"{p}EncBehind", str(e["n_baseline_ahead"])), (f"{p}EncUnclear", str(e["n_unclear"])),
              (f"{p}EncMedDiff", pct(e["median_diff"], True)), (f"{p}EncAheadNoTbd", str(nt["n_model_ahead"])),
              (f"{p}EncBehindNoTbd", str(nt["n_baseline_ahead"])), (f"{p}EncUnclearNoTbd", str(nt["n_unclear"])),
              (f"{p}EncNposNoTbd", str(nt["n_positive"])), (f"{p}EncMedDiffNoTbd", pct(nt["median_diff"], True)),
              (f"{p}EncPoolNoTbd", pct(nt["pooled_diff"], True)), (f"{p}EncPoolNoTbdLo", pct(nt["pooled_ci"][0], True)),
              (f"{p}EncPoolNoTbdHi", pct(nt["pooled_ci"][1], True)), (f"{p}EncTbdDiff", pct(e["tbdense"]["diff"], True)),
              (f"{p}InputTokensM", f"{s['fresh_input_tokens'] / 1e6:.0f}"),
              (f"{p}GpuHours", f"{s['fresh_inference_s'] / 3600:.1f}")]
        # post hoc: gap closed against a common zero (Jev's frozen zero-shot) for every model, and the models' own-zero
        # gap on the sets outside DREsS (whose zero-shot scores differ most between the models)
        common = [(r["optimized"] - r["jev"]["zero"]) / (r["paper"] - r["jev"]["zero"]) for r in rows]
        own_nd = [r["gap_opt"] for r in rows if not r["set"].startswith("dress-") and r["gap_opt"] is not None]
        m += [(f"{p}GapCommon", ipct(statistics.median(common))), (f"{p}GapNoDressAll", ipct(statistics.median(own_nd)))]
    if "jev" in rep:
        J = rep["jev"]
        g0 = J["median_gap_opt_seed0"]
        m += [("clJevGapCfg", ipct(g0["cfg8"])), ("clJevGapCfgNoDress", ipct(g0["cfg8_no_dress"])), ("clJevGapFrozen", ipct(g0["frozen5"])),
              ("clJevGapAll", ipct(g0["all"])), ("clJevGapCalAll", ipct(J["median_gap_cal_frozen_zero"]["all"])),
              ("clJevGapCommon", ipct(J["median_gap_opt_frozen_zero"]["all"])),
              ("clJevGapNoDressAll", ipct(statistics.median(v for k, v in J["gap_opt_frozen_zero"].items() if not k.startswith("dress-"))))]
        if "cal" in J:
            c = J["cal"]
            m += [("clJevNholmCal", str(c["n_holm_positive"])), ("clJevNnegCal", str(c["n_holm_negative"])),
                  ("clJevNposCal", str(c["n_positive"])), ("clJevMedCal", pct(c["median_delta"], True)),
                  ("clJevPoolCal", pct(c["pooled_delta"], True))]
    # Jev's own comparison with the trained baselines (budget_report.py), in digits for the appendix table
    bbs = json.loads((ROOT / "results" / "BUDGET_BASELINE.json").read_text())["summary"]
    m += [("clJevEncAhead", str(bbs["n_pipeline_sig"])), ("clJevEncBehind", str(bbs["n_baseline_sig"])),
          ("clJevEncUnclear", str(len(SETS) - bbs["n_pipeline_sig"] - bbs["n_baseline_sig"])),
          ("clJevEncMedDiff", pct(bbs["median_diff"], True)), ("clJevEncPoolNoTbd", pct(bbs["pooled"]["no_tbdense"]["diff"], True))]
    # The excluded RTX 4090 Clef-flash TB-Dense calibration step (stopped and excluded from the results): its test score
    # was computed and seen.
    m.append(("clFlashExcludedTbdTest", EXCLUDED_TBD_TEST))
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("% generated by jev-opt/experiments/analyze_clef_rq3.py; do not edit\n"
                       + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m))


def app_table(name: str, tag: str, rows: list[dict[str, Any]], s: dict[str, Any]) -> list[str]:
    """One appendix table: every set for one model (scores, contrasts with 95% CIs, Jev, trained baseline)."""
    def sg(x: float) -> str:
        return pct(x, True).replace("-", "$-$")

    def cell(c: dict[str, Any], key: str = "delta", jk: bool = False) -> str:
        mark = ("$^{\\ast\\dagger}$" if jk else "$^\\ast$") if c["p_holm"] < 0.05 else ("$^\\dagger$" if jk else "")
        return f"{sg(c[key])}{mark} {{\\tiny({sg(c['ci'][0])}, {sg(c['ci'][1])})}}"

    out = ["\\begin{table*}[t]", "\\centering\\scriptsize", "\\setlength{\\tabcolsep}{3pt}", "\\resizebox{\\textwidth}{!}{%",
           "\\begin{tabular}{@{}lrrlrlrlrlrl@{}}", "\\toprule",
           f"& \\multicolumn{{9}}{{c}}{{{name}}} & \\multicolumn{{2}}{{c}}{{Same labels, trained}} \\\\",
           "\\cmidrule(lr){2-10}\\cmidrule(l){11-12}",
           "Set & Zero & Cal. & $\\Delta$ cal. (95\\% CI) & Opt. & $\\Delta$ opt. (95\\% CI) & Gap & Rep. & \\jev{} & Opt.\\ $-$ \\jev{} (95\\% CI) "
           "& Best & Opt.\\ $-$ trained (95\\% CI) \\\\", "\\midrule"]
    for i, r in enumerate(rows):
        if i == len(CFG8):
            out.append("\\midrule")
        c, v = r["contrasts"], r["vs_encoder"]
        jk = "p_jackknife" in c["opt-zero"]
        rep_ = {"top-single": "P", "top-stacked": "P+h"}[r["chosen"]]
        if r["chosen"] != r["jev_chosen"]:
            rep_ += "$^\\S$"
        out.append(f"{r['title']} & {pct(r['zero'])} & {pct(r['calibrated'])} & {cell(c['cal-zero'], jk=jk)} & {pct(r['optimized'])} & "
                   f"{cell(c['opt-zero'], jk=jk)} & {ipct(r['gap_opt'])}\\% & {rep_} & {pct(r['jev']['optimized'])} & "
                   f"{cell(c['opt-jev'], jk=jk)} & {pct(v['baseline_test_mean'])} & {cell(v, 'diff', jk='p_jackknife' in v)} \\\\")
    out += ["\\bottomrule", "\\end{tabular}}",
            f"\\caption{{{name} on the RQ3 sets (test, $\\times$100; the \\optNTasks{{}} configurations, then the "
            f"\\fzNewNSets{{}} rule-selected sets). Zero: {name}'s frozen zero-shot run. Cal.: per-class bias and temperature on "
            f"the paper's question, fit on {name}'s answers. Opt.: \\jev{{}}'s seed-0 optimized questions answered by {name}, "
            "logistic regression refitted on its answers, representative (Rep.: P single, P+h stacked; $^\\S$: not \\jev{}'s "
            "choice) selected on its validation answers. Gap: share of the zero-shot-to-paper gap closed by Opt. "
            "\\jev{}: \\jev{}'s seed-0 pipeline score. Best: the best same-label trained baseline on validation, mean of 3 seeds. "
            "$\\Delta$: paired cluster bootstrap with RQ1's units; $^\\ast$: Holm $p<0.05$ over the 13 sets; $^\\dagger$: "
            "jackknife-t $p$ (fewer than 30 documents). "
            f"Pooled: $\\Delta$ cal.\\ {sg(s['cal-zero']['pooled_delta'])}, $\\Delta$ opt.\\ {sg(s['opt-zero']['pooled_delta'])}, "
            f"Opt.\\ $-$ \\jev{{}} {sg(s['opt-jev']['pooled_delta'])}.}}",
            f"\\label{{tab:{tag}}}", "\\end{table*}", ""]
    return out


def write_app(rep: dict[str, Any]) -> None:
    """Appendix tables for Clef and Clef-flash."""
    out = ["% generated by jev-opt/experiments/analyze_clef_rq3.py; do not edit"]
    for model, tag in (("clef", "clef"), ("clef-flash", "flash")):
        if rep[model]["sets"]:
            out += app_table(MODELS[model][0], f"clef-rq3-{tag}", rep[model]["sets"], rep[model]["summary"])
    OUT_APP.write_text("\n".join(out) + "\n")


def jev_reference(jev: dict[str, dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Jev's own gap medians in the paper's conventions, and its calibration step paired like the models' (from `rows`)."""
    seeds = json.loads((ROOT / "results" / "seeds.json").read_text())["tasks"]
    gepa = json.loads((ROOT / "results" / "gepa_v3_test.json").read_text())["tasks"]
    med = statistics.median
    # 8 configurations as in Table 3: zero = the paper's question through the optimizer's request path, paper = Table 3's
    g0 = {s: (seeds[s]["seeds"]["0"]["test"] - seeds[s]["zero_shot_test"]) / (gepa[s]["paper"] - seeds[s]["zero_shot_test"]) for s in CFG8}
    g3 = {s: (seeds[s]["test_mean"] - seeds[s]["zero_shot_test"]) / (gepa[s]["paper"] - seeds[s]["zero_shot_test"]) for s in CFG8}
    gf = {s: jev[s]["gap_opt"] for s in SETS}  # results/frozen.json: zero = the frozen zero-shot audit, seed-0 program
    seed0 = {**g0, **{s: gf[s] for s in GROUPS["frozen5"]}}
    out = {"gap_opt_seed0": seed0, "gap_opt_mean3": g3, "gap_opt_frozen_zero": gf,
           "median_gap_opt_seed0": {g: med(seed0[s] for s in GROUPS[g]) for g in ("cfg8", "cfg8_no_dress", "frozen5")}
           | {"all": med(seed0.values())},
           "median_gap_opt_mean3": {g: med(g3[s] for s in GROUPS[g]) for g in ("cfg8", "cfg8_no_dress")},
           "median_gap_opt_frozen_zero": {g: med(gf[s] for s in GROUPS[g]) for g in ("cfg8", "cfg8_no_dress", "frozen5")}
           | {"all": med(gf.values())},
           "median_gap_cal_frozen_zero": {g: med(jev[s]["gap_cal"] for s in GROUPS[g]) for g in ("cfg8", "frozen5")}
           | {"all": med(jev[s]["gap_cal"] for s in SETS)}}
    if len(rows) == len(SETS):
        cs = [r["contrasts"]["jevcal-jevzero"] for r in rows]
        out["cal"] = {"sets": {r["set"]: dict(c) for r, c in zip(rows, cs)},
                      "n_holm_positive": sum(c["p_holm"] < 0.05 and c["delta"] > 0 for c in cs),
                      "n_holm_negative": sum(c["p_holm"] < 0.05 and c["delta"] < 0 for c in cs),
                      "n_ci_above_zero": sum(c["ci"][0] > 0 for c in cs), "n_positive": sum(c["delta"] > 0 for c in cs),
                      "median_delta": med(c["delta"] for c in cs), "pooled_delta": statistics.mean(c["delta"] for c in cs)}
    return out


def main() -> int:
    if "--report-only" in sys.argv:  # rewrite the .md and .tex outputs from the saved results/CLEF_RQ3.json
        rep = json.loads(OUT_JSON.read_text())
        write_md(rep)
        write_tex(rep)
        write_app(rep)
        return 0
    jev = jev_rows()
    jtp = json.loads((BB_DIR / "jev_test_preds.json").read_text())
    bb = json.loads((ROOT / "results" / "BUDGET_BASELINE.json").read_text())["sets"]
    rep: dict[str, Any] = {"bootstrap": {"B": B, "seed": 0},
                           "bootstrap_vs_encoder": {"B": B_ENC, "seed": 0, "as": "jev-opt/experiments/budget_report.py"}}
    for model in MODELS:
        rows = [r for stem in SETS if (r := set_row(model, stem, jev[stem], jtp, bb)) is not None]
        missing = [s for s in SETS if s not in {r["set"] for r in rows}]
        summary = summarize(rows) if rows else {}
        for r in rows:
            for c in r["contrasts"].values():
                c.pop("draws", None)
            r["vs_encoder"].pop("draws", None)
        rep[model] = {"pin": PINS[model], "status": "finished" if not missing else f"pending: {', '.join(missing)}",
                      "sets": rows, "summary": summary}
        print(f"{model}: {len(rows)} sets finished", flush=True)
    rep["jev"] = jev_reference(jev, rep["clef"]["sets"])
    OUT_JSON.write_text(json.dumps(rep, indent=1) + "\n")
    write_md(rep)
    write_tex(rep)
    write_app(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
