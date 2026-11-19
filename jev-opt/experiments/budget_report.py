"""Analysis of the same-label-budget baselines (the pre-registered analysis).

Reads results/budget_baseline/{jev_test_preds.json, emb/, setfit/, ft/} and writes results/BUDGET_BASELINE.{json,md},
paper/generated/budget_baseline.tex (\\bb... macros) and paper/generated/app_budget_baseline.tex (appendix tables).
No model is trained and no Jev call is made.

  uv run --project jev-opt/python python jev-opt/experiments/budget_report.py [--out DIR]   # DIR replaces paper/generated
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).parent))
from budget_baseline import OUT, OUT_JSON, OUT_MD, ROOT, SETS, TEX, Data  # noqa: E402
from jev_opt import metrics  # noqa: E402
from offline_rq3 import boot_p, holm, metric_from_counts  # noqa: E402
from resample import cluster_index, jackknife_t  # noqa: E402

if "--out" in sys.argv:
    TEX = Path(sys.argv[sys.argv.index("--out") + 1])

B = 10_000
CHUNK = 500
FAMILIES = [("emb", "B1", "Embeddings + LR"), ("setfit", "B2", "SetFit"), ("ft", "B3", "Fine-tuned encoder")]
FAM_NAME = {"emb": "embeddings + linear model", "setfit": "SetFit", "ft": "fine-tuned encoder"}
MODEL_SHORT = {"FacebookAI/roberta-large": "RoBERTa-large", "microsoft/deberta-v3-large": "DeBERTa-v3-large",
               "BAAI/bge-small-en-v1.5": "bge-small", "BAAI/bge-large-en-v1.5": "bge-large", "BAAI/bge-base-en-v1.5": "bge-base"}


def mean(xs):
    return statistics.mean(xs)


def sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def f1(x: float) -> str:
    return f"{100 * x:.1f}"


def sgn(x: float) -> str:
    s = f"{100 * x:+.1f}"
    return "0.0" if s in ("+0.0", "-0.0") else s


def fp(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


class Scorer:
    """Metric on bootstrap-weighted confusion counts; same conventions as jev_opt.metrics.primary."""

    def __init__(self, d: Data, gold: list[str]):
        self.name, self.k = d.metric, len(d.labels)
        self.neg = d.labels.index(d.neg) if d.neg else None
        self.lid = {lab: i for i, lab in enumerate(d.labels)}
        self.gold = gold
        self.g = np.array([self.lid[x] for x in gold])

    def onehot(self, pred: list[str | None]) -> np.ndarray:
        k = self.k
        if self.name == "qwk":  # failed predictions count as the farthest level from gold
            p = np.array([self.lid[x] if x in self.lid else (0 if gi >= k / 2 else k - 1) for gi, x in zip(self.g, pred)])
        else:
            p = np.array([self.lid.get(x, k) if x is not None else k for x in pred])
        oh = np.zeros((len(p), k * (k + 1)))
        oh[np.arange(len(p)), self.g * (k + 1) + p] = 1.0
        return oh

    def score(self, oh: np.ndarray, w: np.ndarray) -> np.ndarray:
        cm = (w @ oh).reshape(len(w), self.k, self.k + 1)
        return metric_from_counts(cm, self.name, self.k, self.neg)


def main() -> int:  # noqa: C901
    jev = json.loads((OUT / "jev_test_preds.json").read_text())
    seeds = json.loads((ROOT / "results" / "seeds.json").read_text())["tasks"]
    gepa = json.loads((ROOT / "results" / "gepa_v3_test.json").read_text())["tasks"]
    frozen = {r["stem"]: r for r in json.loads((ROOT / "results" / "frozen.json").read_text())["new"]}

    T: dict = {}
    D_best: dict[str, np.ndarray] = {}
    D_fam: dict[str, dict[str, np.ndarray]] = {f: {} for f, *_ in FAMILIES}
    gpu_seconds = {f: 0.0 for f, *_ in FAMILIES}
    # Bootstrap streams of the same-label comparison: one cluster stream per task,
    # shared by the three DREsS rubrics (same essays), independent across tasks; one seed-resampling stream per set.
    task_of = {slug: ("dress" if slug.startswith("dress-new-") else slug) for slug, *_ in SETS}
    task_index = {t: k for k, t in enumerate(dict.fromkeys(task_of.values()))}
    task_inv: dict[str, np.ndarray] = {}
    for set_index, (slug, short, tag, group) in enumerate(SETS):
        d = Data(slug)
        J = jev[slug]
        gold, ids = J["gold"], J["ids"]
        sc = Scorer(d, gold)
        jev_runs = list(J["runs"])
        jev_pred = [J["runs"][r]["pred"] for r in jev_runs]
        jev_test = [J["runs"][r]["primary"] for r in jev_runs]
        fams = {}
        for f, *_ in FAMILIES:
            rec = json.loads((OUT / f / f"{slug}.json").read_text())
            gpu_seconds[f] += rec["seconds"]
            for p, t in zip(rec["test_pred"], rec["test"]):  # stored scores = metrics.primary on stored predictions
                assert abs(metrics.primary(d.metric, gold, p, d.labels, d.neg) - t) < 1e-12, (slug, f)
            fams[f] = rec
        if group == "opt":
            Z, P = seeds[slug]["zero_shot_test"], gepa[slug]["paper"]
        else:
            Z, P = frozen[slug]["zero"], frozen[slug]["paper"]
        best = max((f for f, *_ in FAMILIES), key=lambda f: mean(fams[f]["val"]))  # first family wins ties

        # one set of cluster resamples per set, shared by every system
        inv, unit = cluster_index(slug, ids)
        m = int(inv.max()) + 1
        task = task_of[slug]
        if task in task_inv:  # rubrics of one task must share clusters for the aligned draws to mean the same essays
            assert np.array_equal(task_inv[task], inv), slug
        task_inv[task] = inv
        rng = np.random.default_rng([0, task_index[task]])  # cluster draws
        rng_seed = np.random.default_rng([1, set_index])  # seed resampling
        oh_j = [sc.onehot(p) for p in jev_pred]
        oh_f = {f: [sc.onehot(p) for p in fams[f]["test_pred"]] for f in fams}
        ones = np.ones((1, len(inv)))
        for oh, t in zip(oh_j, jev_test):
            assert abs(sc.score(oh, ones)[0] - t) < 1e-9, slug
        for f in fams:
            for oh, t in zip(oh_f[f], fams[f]["test"]):
                assert abs(sc.score(oh, ones)[0] - t) < 1e-9, (slug, f)
        SJ = np.empty((len(oh_j), B))
        SF = {f: np.empty((len(oh_f[f]), B)) for f in fams}
        for lo in range(0, B, CHUNK):
            b = min(CHUNK, B - lo)
            draw = rng.integers(0, m, size=(b, m))
            w = np.bincount((draw + m * np.arange(b)[:, None]).ravel(), minlength=b * m).reshape(b, m)[:, inv].astype(float)
            for i, oh in enumerate(oh_j):
                SJ[i, lo:lo + b] = sc.score(oh, w)
            for f in fams:
                for i, oh in enumerate(oh_f[f]):
                    SF[f][i, lo:lo + b] = sc.score(oh, w)

        def seed_mean(S: np.ndarray) -> np.ndarray:
            n = S.shape[0]
            pick = rng_seed.integers(0, n, size=(B, n))
            return S[pick, np.arange(B)[:, None]].mean(1)

        jm = seed_mean(SJ)
        comps = {}
        for f in fams:
            D = jm - seed_mean(SF[f])
            obs = mean(jev_test) - mean(fams[f]["test"])
            comps[f] = {"diff": obs, "ci": [float(x) for x in np.percentile(D, [2.5, 97.5])], "p": boot_p(D)}
            D_fam[f][slug] = D
        D_best[slug] = jm - seed_mean(SF[best])
        best_cmp = dict(comps[best])
        best_cmp["ci90"] = [float(x) for x in np.percentile(D_best[slug], [5, 95])]  # TOST interval (pre-registration amendment)
        if unit != "item" and m < 30:
            def stat(ix, f=best):
                pj = mean(metrics.primary(d.metric, [gold[i] for i in ix], [p[i] for i in ix], d.labels, d.neg) for p in jev_pred)
                pb = mean(metrics.primary(d.metric, [gold[i] for i in ix], [p[i] for i in ix], d.labels, d.neg)
                          for p in fams[f]["test_pred"])
                return pj - pb
            jlo, jhi = jackknife_t(stat, inv)
            best_cmp["jackknife_ci"] = [jlo, jhi]
            # two-sided jackknife-t p of the observed D (the interval is centred on it): se from the interval half-width
            se = (jhi - jlo) / (2 * student_t.ppf(0.975, m - 1))
            best_cmp["p_jackknife"] = float(2 * student_t.sf(abs(best_cmp["diff"]) / se, m - 1))
            best_cmp["jackknife_ci90"] = [best_cmp["diff"] - student_t.ppf(0.95, m - 1) * se,
                                          best_cmp["diff"] + student_t.ppf(0.95, m - 1) * se]
            best_cmp["jackknife_se"], best_cmp["jackknife_df"] = float(se), m - 1

        def fam_row(f):
            r = fams[f]
            return {"chosen": r["chosen"], "val": r["val"], "val_mean": mean(r["val"]), "test": r["test"],
                    "test_mean": mean(r["test"]), "test_sd": sd(r["test"]), "vs_jev": comps[f],
                    "gap_share": (mean(r["test"]) - Z) / (P - Z), "paper_minus": P - mean(r["test"])}

        T[slug] = {"short": short, "tag": tag, "group": group, "metric": d.metric, "n_test": len(gold),
                   "cluster_unit": unit, "n_clusters": m, "zero": Z, "paper": P,
                   "jev": {"runs": jev_runs, "test": jev_test, "mean": mean(jev_test), "sd": sd(jev_test),
                           "gap_share": (mean(jev_test) - Z) / (P - Z)},
                   "families": {f: fam_row(f) for f, *_ in FAMILIES}, "best": best, "best_vs_jev": best_cmp,
                   "best_test_mean": mean(fams[best]["test"]), "best_test_sd": sd(fams[best]["test"]),
                   "best_paper_minus": P - mean(fams[best]["test"]),
                   "best_gap_share": (mean(fams[best]["test"]) - Z) / (P - Z),
                   "ft_grid": {"n_runs": len(fams["ft"]["grid"]),
                               "n_val_below_0.1": sum(g["val"] < 0.1 for g in fams["ft"]["grid"]),
                               "n_stopped_best_epoch_1": sum(g["best_epoch"] == 1 and g["epochs_run"] < 20
                                                             for g in fams["ft"]["grid"])}}
        print(f"{slug}: jev {f1(mean(jev_test))} best {best} {f1(mean(fams[best]['test']))} D {sgn(best_cmp['diff'])} "
              f"[{sgn(best_cmp['ci'][0])}, {sgn(best_cmp['ci'][1])}] p {best_cmp['p']:.4f}", flush=True)

    slugs = [s for s, *_ in SETS]
    for p, s in zip(holm([T[s]["best_vs_jev"]["p"] for s in slugs]), slugs):
        T[s]["best_vs_jev"]["p_holm"] = p
    # small-cluster rule: the jackknife-t p replaces the bootstrap p on clustered sets with fewer than 30 clusters
    for p, s in zip(holm([T[s]["best_vs_jev"].get("p_jackknife", T[s]["best_vs_jev"]["p"]) for s in slugs]), slugs):
        T[s]["best_vs_jev"]["p_holm_small"] = p
    for f, *_ in FAMILIES:
        for p, s in zip(holm([T[s]["families"][f]["vs_jev"]["p"] for s in slugs]), slugs):
            T[s]["families"][f]["vs_jev"]["p_holm"] = p

    def pooled(Dd: dict[str, np.ndarray], keys: list[str], obs: list[float]) -> dict:
        P_ = np.mean([Dd[k] for k in keys], axis=0)
        return {"diff": mean(obs), "ci": [float(x) for x in np.percentile(P_, [2.5, 97.5])], "p": boot_p(P_), "n_sets": len(keys)}

    groups = {"all": slugs, "opt": [s for s in slugs if T[s]["group"] == "opt"], "frozen": [s for s in slugs if T[s]["group"] == "frozen"],
              "no_tbdense": [s for s in slugs if s != "contempo-tbdense"]}
    pool = {g: pooled(D_best, ks, [T[s]["best_vs_jev"]["diff"] for s in ks]) for g, ks in groups.items()}
    pool_fam = {f: pooled(D_fam[f], slugs, [T[s]["families"][f]["vs_jev"]["diff"] for s in slugs]) for f, *_ in FAMILIES}

    # Equivalence (pre-registration amendment of 2026-10-09): TOST at alpha 0.05 = 90% interval inside (-delta, +delta).
    def tost_pooled(keys: list[str], delta: float) -> dict:
        P_ = np.mean([D_best[k] for k in keys], axis=0)
        lo, hi = (float(x) for x in np.percentile(P_, [5, 95]))
        p = max(float((P_ <= -delta).mean()), float((P_ >= delta).mean()), 1 / len(P_))
        return {"delta": delta, "n_sets": len(keys), "diff": mean([T[s]["best_vs_jev"]["diff"] for s in keys]),
                "ci90": [lo, hi], "p_tost": p, "equivalent": bool(lo > -delta and hi < delta)}

    def tost_set(s: str, delta: float) -> dict:
        c = T[s]["best_vs_jev"]
        if "jackknife_ci90" in c:
            lo, hi = c["jackknife_ci90"]
            se, df = c["jackknife_se"], c["jackknife_df"]
            p = max(float(student_t.sf((c["diff"] + delta) / se, df)), float(student_t.sf((delta - c["diff"]) / se, df)))
            how = "jackknife-t"
        else:
            lo, hi = c["ci90"]
            p = max(float((D_best[s] <= -delta).mean()), float((D_best[s] >= delta).mean()), 1 / len(D_best[s]))
            how = "bootstrap"
        lo, hi = float(lo), float(hi)
        return {"delta": delta, "ci90": [lo, hi], "interval": how, "p_tost": p, "equivalent": bool(lo > -delta and hi < delta)}

    equivalence = {"pooled": {f"{g}_{d}": tost_pooled(groups[g], d / 100) for g in ("all", "no_tbdense") for d in (2, 3)},
                   "per_set_3": {s: tost_set(s, 0.03) for s in slugs}}
    equivalence["n_sets_equivalent_3"] = sum(v["equivalent"] for v in equivalence["per_set_3"].values())

    def read(key: str) -> dict[str, str]:
        return {s: ("pipeline" if T[s]["best_vs_jev"][key] < 0.05 and T[s]["best_vs_jev"]["diff"] > 0 else
                    "baseline" if T[s]["best_vs_jev"][key] < 0.05 and T[s]["best_vs_jev"]["diff"] < 0 else "unclear")
                for s in slugs}
    reading_boot = read("p_holm")  # the pre-registered reading rule (bootstrap p everywhere)
    reading = read("p_holm_small")  # reported counts: jackknife-t p on the sets with fewer than 30 clusters
    summary = {
        "pooled": pool, "pooled_by_family": pool_fam, "equivalence": equivalence, "reading": reading,
        "reading_bootstrap": reading_boot,
        "n_pipeline_sig": sum(r == "pipeline" for r in reading.values()),
        "n_baseline_sig": sum(r == "baseline" for r in reading.values()),
        "n_pipeline_sig_bootstrap": sum(r == "pipeline" for r in reading_boot.values()),
        "n_baseline_sig_bootstrap": sum(r == "baseline" for r in reading_boot.values()),
        "n_pos": sum(T[s]["best_vs_jev"]["diff"] > 0 for s in slugs),
        "median_diff": statistics.median(T[s]["best_vs_jev"]["diff"] for s in slugs),
        "best_family_counts": {f: sum(T[s]["best"] == f for s in slugs) for f, *_ in FAMILIES},
        "median_gap_share_best": statistics.median(T[s]["best_gap_share"] for s in slugs),
        "median_gap_share_jev": statistics.median(T[s]["jev"]["gap_share"] for s in slugs),
        "median_gap_share_by_group": {g: {"best": statistics.median(T[s]["best_gap_share"] for s in groups[g]),
                                          "jev": statistics.median(T[s]["jev"]["gap_share"] for s in groups[g])}
                                      for g in ("opt", "frozen")},
        "median_paper_minus_best": statistics.median(T[s]["best_paper_minus"] for s in slugs),
        "n_best_reach_paper": sum(T[s]["best_paper_minus"] <= 0 for s in slugs),
        "best_reach_paper_sets": [s for s in slugs if T[s]["best_paper_minus"] <= 0],
        "n_jev_reach_paper": sum(T[s]["paper"] - T[s]["jev"]["mean"] <= 0 for s in slugs),
        "gpu_hours": {f: v / 3600 for f, v in gpu_seconds.items()},
    }
    OUT_JSON.write_text(json.dumps({"sets": T, "summary": summary}, indent=1) + "\n")
    write_md(T, summary, slugs)
    write_tex(T, summary, slugs)
    print(json.dumps({k: v for k, v in summary.items() if k != "reading"}, indent=1))
    return 0


def cfg_short(f: str, c: dict) -> str:
    if f == "emb":
        enc = MODEL_SHORT.get(c["encoder"], c["encoder"])
        return f"{enc}, ridge α={c['alpha']:g}" if c["model"] == "ridge" else \
            f"{enc}, LR C={c['C']:g}{', balanced' if c['class_weight'] else ''}"
    if f == "setfit":
        return MODEL_SHORT.get(c["body"], c["body"])
    return f"{MODEL_SHORT.get(c['model'], c['model'])}, lr {c['lr']:g}"


def write_md(T: dict, S: dict, slugs: list[str]) -> None:
    L = ["# Same-label-budget trained baselines", "",
         "Test primary metric ×100. Jev = the decision-model pipeline's optimized questions (mean ± sd over 3 optimizer "
         "seeds on all 13 sets). Baselines trained on the same 500 train labels, "
         "selected on the same validation split; B2/B3 mean ± sd over 3 seeds. Best = family with the highest "
         "validation score. D = Jev − best baseline; hierarchical cluster bootstrap (B = 10,000; RQ1 units; Jev and "
         "baseline seeds resampled), 95% percentile CI, Holm over 13 sets. Holm p and Reading use the jackknife-t p "
         "in place of the bootstrap p on clustered sets with fewer than 30 test clusters (TB-Dense, MATRES); Holm p (boot) "
         "is the pre-registered bootstrap-only rule. Gap share = (score − zero-shot) / (paper − zero-shot).", "",
         "| Set | Metric | Jev | B1 emb | B2 SetFit | B3 fine-tuned | Best (val) | D [95% CI] | Holm p | Holm p (boot) | Reading | Paper | Paper − best |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in slugs:
        t = T[s]
        F = t["families"]
        c = t["best_vs_jev"]
        jk = (f" (jackknife {sgn(c['jackknife_ci'][0])}…{sgn(c['jackknife_ci'][1])}, p {fp(c['p_jackknife'])})"
              if "jackknife_ci" in c else "")
        js = f" ± {f1(t['jev']['sd'])}" if len(t["jev"]["test"]) > 1 else " (1 run)"
        L.append(f"| {t['short']} | {t['metric']} | {f1(t['jev']['mean'])}{js} | "
                 f"{f1(F['emb']['test_mean'])} | {f1(F['setfit']['test_mean'])} ± {f1(F['setfit']['test_sd'])} | "
                 f"{f1(F['ft']['test_mean'])} ± {f1(F['ft']['test_sd'])} | {t['best']} | {sgn(c['diff'])} "
                 f"[{sgn(c['ci'][0])}, {sgn(c['ci'][1])}]{jk} | {fp(c['p_holm_small'])} | {fp(c['p_holm'])} | {S['reading'][s]} | {f1(t['paper'])} | "
                 f"{sgn(t['best_paper_minus'])} |")
    P = S["pooled"]
    L += ["", "## Pooled (Jev − best baseline)", ""]
    for g, lab in (("all", "13 sets"), ("opt", "8 configurations"), ("frozen", "5 frozen sets"), ("no_tbdense", "12 sets without TB-Dense")):
        L.append(f"- {lab}: {sgn(P[g]['diff'])} [{sgn(P[g]['ci'][0])}, {sgn(P[g]['ci'][1])}], p {fp(P[g]['p'])}")
    L += [f"- Jev − each family over 13 sets: " + "; ".join(
        f"{f} {sgn(v['diff'])} [{sgn(v['ci'][0])}, {sgn(v['ci'][1])}]" for f, v in S["pooled_by_family"].items()),
          f"- D > 0 on {S['n_pos']} of 13 sets, median D {sgn(S['median_diff'])}; Holm-significant for the pipeline on "
          f"{S['n_pipeline_sig']}, for the baseline on {S['n_baseline_sig']} (bootstrap-only rule: {S['n_pipeline_sig_bootstrap']} "
          f"and {S['n_baseline_sig_bootstrap']}).",
          f"- Best family by validation: {S['best_family_counts']}.",
          f"- Median gap share (gap from this file's `zero`, which on the 8 study configurations is the paper's question "
          f"as in results/seeds.json, with Score cutoffs fit on training data for DREsS; the paper's single yardstick, gap "
          f"from prompt-only Jev, is results/ANSWER_TABLE.md): best baseline {100 * S['median_gap_share_best']:.0f}%, Jev {100 * S['median_gap_share_jev']:.0f}% "
          f"(8 configurations {100 * S['median_gap_share_by_group']['opt']['best']:.0f}% vs "
          f"{100 * S['median_gap_share_by_group']['opt']['jev']:.0f}%; 5 frozen sets "
          f"{100 * S['median_gap_share_by_group']['frozen']['best']:.0f}% vs {100 * S['median_gap_share_by_group']['frozen']['jev']:.0f}%); "
          f"median paper − best baseline {sgn(S['median_paper_minus_best'])} points; best baseline reaches the paper on "
          f"{S['n_best_reach_paper']} sets ({', '.join(T[s]['short'] for s in S['best_reach_paper_sets'])}), Jev on "
          f"{S['n_jev_reach_paper']}.",
          "- B3 grid runs with validation < 0.1 / stopped early with best epoch 1 (of 18): " + "; ".join(
              f"{T[s]['short']} {T[s]['ft_grid']['n_val_below_0.1']}/{T[s]['ft_grid']['n_stopped_best_epoch_1']}" for s in slugs) + ".",
          f"- Compute (wall-clock on one RTX 4090 per job, includes validation every epoch): " + ", ".join(
              f"{f} {h:.1f} h" for f, h in S["gpu_hours"].items()) + f"; total {sum(S['gpu_hours'].values()):.1f} GPU-hours.",
          "", "## Equivalence (TOST, α = 0.05; pre-registration amendment of 2026-10-09)", "",
          "Equivalent within ±δ iff the 90% interval of D lies inside (−δ, +δ). Pooled: bootstrap 90% percentile interval of "
          "the mean D. Per set: bootstrap 90% interval, jackknife-t 90% interval on TB-Dense and MATRES.", ""]
    E = S["equivalence"]
    for k, v in E["pooled"].items():
        g, d = k.rsplit("_", 1)
        L.append(f"- Pooled, {'13 sets' if g == 'all' else '12 sets without TB-Dense'}, δ = {d} points: D {sgn(v['diff'])}, "
                 f"90% CI [{sgn(v['ci90'][0])}, {sgn(v['ci90'][1])}], p_TOST {fp(v['p_tost'])}: "
                 f"{'equivalent' if v['equivalent'] else 'equivalence not shown'}.")
    L.append(f"- Per set, δ = 3 points: equivalent on {E['n_sets_equivalent_3']} of {len(slugs)}: " + "; ".join(
        f"{T[s]['short']} [{sgn(v['ci90'][0])}, {sgn(v['ci90'][1])}]{' eq.' if v['equivalent'] else ''}"
        for s, v in E["per_set_3"].items()) + ".")
    L += ["", "## Per family: chosen configuration, validation, Jev − family", "",
          "| Set | Family | Chosen | Val | Test | Jev − family [95% CI] | Holm p (13 sets) |", "|---|---|---|---|---|---|---|"]
    for s in slugs:
        for f, *_ in FAMILIES:
            r = T[s]["families"][f]
            c = r["vs_jev"]
            val = f"{f1(r['val_mean'])} ± {f1(sd(r['val']))}" if len(r["val"]) > 1 else f1(r["val_mean"])
            test = f"{f1(r['test_mean'])} ± {f1(r['test_sd'])}" if len(r["test"]) > 1 else f1(r["test_mean"])
            L.append(f"| {T[s]['short']} | {f} | {cfg_short(f, r['chosen'])} | {val} | {test} | {sgn(c['diff'])} "
                     f"[{sgn(c['ci'][0])}, {sgn(c['ci'][1])}] | {fp(c['p_holm'])} |")
    OUT_MD.write_text("\n".join(L) + "\n")


def write_tex(T: dict, S: dict, slugs: list[str]) -> None:
    mac: dict[str, str] = {}
    for s in slugs:
        t, tag, c = T[s], T[s]["tag"], T[s]["best_vs_jev"]
        mac[f"bb{tag}Best"] = f1(t["best_test_mean"])
        mac[f"bb{tag}BestSd"] = f1(t["best_test_sd"])
        mac[f"bb{tag}BestFam"] = FAM_NAME[t["best"]]
        mac[f"bb{tag}Jev"] = f1(t["jev"]["mean"])
        mac[f"bb{tag}Diff"] = sgn(c["diff"])
        mac[f"bb{tag}AbsDiff"] = f1(abs(c["diff"]))
        mac[f"bb{tag}Lo"] = sgn(c["ci"][0])
        mac[f"bb{tag}Hi"] = sgn(c["ci"][1])
        mac[f"bb{tag}NClusters"] = f"{t['n_clusters']:,}"
        if "jackknife_ci" in c:
            mac[f"bb{tag}JkLo"] = sgn(c["jackknife_ci"][0])
            mac[f"bb{tag}JkHi"] = sgn(c["jackknife_ci"][1])
            mac[f"bb{tag}PJk"] = fp(c["p_jackknife"])
        mac[f"bb{tag}PHolm"] = fp(c["p_holm_small"])
        mac[f"bb{tag}PHolmBoot"] = fp(c["p_holm"])
        mac[f"bb{tag}Zero"] = f1(t["zero"])
        mac[f"bb{tag}Paper"] = f1(t["paper"])
        for f, *_ in FAMILIES:
            mac[f"bb{tag}{f.capitalize()}"] = f1(t["families"][f]["test_mean"])
    P = S["pooled"]
    for g, lab in (("all", "All"), ("opt", "Opt"), ("frozen", "Frozen"), ("no_tbdense", "NoTbd")):
        mac[f"bbPooled{lab}Diff"] = sgn(P[g]["diff"])
        mac[f"bbPooled{lab}Abs"] = f1(abs(P[g]["diff"]))
        mac[f"bbPooled{lab}Lo"] = sgn(P[g]["ci"][0])
        mac[f"bbPooled{lab}Hi"] = sgn(P[g]["ci"][1])
    for f, *_ in FAMILIES:
        v = S["pooled_by_family"][f]
        mac[f"bbPooled{f.capitalize()}Diff"] = sgn(v["diff"])
        mac[f"bbPooled{f.capitalize()}Abs"] = f1(abs(v["diff"]))
        mac[f"bbPooled{f.capitalize()}Lo"] = sgn(v["ci"][0])
        mac[f"bbPooled{f.capitalize()}Hi"] = sgn(v["ci"][1])
        mac[f"bbNBest{f.capitalize()}"] = str(S["best_family_counts"][f])
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen"]
    mac["bbNSets"] = str(len(slugs))
    mac["bbNPos"] = str(S["n_pos"])
    mac["bbNNeg"] = str(sum(T[s]["best_vs_jev"]["diff"] < 0 for s in slugs))
    mac["bbMedianDiff"] = sgn(S["median_diff"])
    mac["bbNPipelineSig"] = words[S["n_pipeline_sig"]]
    mac["bbNBaselineSig"] = words[S["n_baseline_sig"]]
    mac["bbNUnclear"] = words[len(slugs) - S["n_pipeline_sig"] - S["n_baseline_sig"]]
    mac["bbNPipelineSigBoot"] = words[S["n_pipeline_sig_bootstrap"]]
    mac["bbNBaselineSigBoot"] = words[S["n_baseline_sig_bootstrap"]]
    mac["bbGapShareBestOpt"] = f"{100 * S['median_gap_share_by_group']['opt']['best']:.0f}"
    mac["bbGapShareJevOpt"] = f"{100 * S['median_gap_share_by_group']['opt']['jev']:.0f}"
    mac["bbGapShareBestFrozen"] = f"{100 * S['median_gap_share_by_group']['frozen']['best']:.0f}"
    mac["bbGapShareJevFrozen"] = f"{100 * S['median_gap_share_by_group']['frozen']['jev']:.0f}"
    tbd = T["contempo-tbdense"]
    mac["bbTbdNGrid"] = str(tbd["ft_grid"]["n_runs"])
    mac["bbTbdNGridCollapsed"] = str(tbd["ft_grid"]["n_val_below_0.1"])
    mac["bbTbdNGridStopped"] = str(tbd["ft_grid"]["n_stopped_best_epoch_1"])
    mac["bbTbdFtMin"] = f1(min(tbd["families"]["ft"]["test"]))
    mac["bbTbdFtMax"] = f1(max(tbd["families"]["ft"]["test"]))
    mac["bbGapShareBest"] = f"{100 * S['median_gap_share_best']:.0f}"
    mac["bbGapShareJev"] = f"{100 * S['median_gap_share_jev']:.0f}"
    mac["bbPaperMinusBest"] = f1(S["median_paper_minus_best"])
    mac["bbNBestReachPaper"] = words[S["n_best_reach_paper"]]
    mac["bbBestReachPaperSets"] = ", ".join(T[s]["short"] for s in S["best_reach_paper_sets"])
    mac["bbGpuHours"] = f"{sum(S['gpu_hours'].values()):.1f}"
    E = S["equivalence"]
    for g, lab in (("all", "All"), ("no_tbdense", "NoTbd")):
        v2, v3 = E["pooled"][f"{g}_2"], E["pooled"][f"{g}_3"]
        mac[f"bbEq{lab}Lo"], mac[f"bbEq{lab}Hi"] = sgn(v2["ci90"][0]), sgn(v2["ci90"][1])
        for d, v in (("Two", v2), ("Three", v3)):
            mac[f"bbEq{lab}{d}P"] = fp(v["p_tost"])
            mac[f"bbEq{lab}{d}Shown"] = "yes" if v["equivalent"] else "no"
    mac["bbNEqThree"] = words[E["n_sets_equivalent_3"]]
    eq_sets = [T[s]["short"] for s in slugs if E["per_set_3"][s]["equivalent"]]
    mac["bbEqThreeSets"] = ", ".join(eq_sets) if eq_sets else "none"
    for f, *_ in FAMILIES:
        mac[f"bbGpuHours{f.capitalize()}"] = f"{S['gpu_hours'][f]:.1f}"
    dress = [-T[s]["best_vs_jev"]["diff"] for s in slugs if s.startswith("dress-")]  # best baseline's lead on DREsS
    mac["bbDressLeadMin"] = f1(min(dress))
    mac["bbDressLeadMax"] = f1(max(dress))
    sig_pipe = [T[s]["short"] for s in slugs if S["reading"][s] == "pipeline"]
    sig_base = [T[s]["short"] for s in slugs if S["reading"][s] == "baseline"]
    mac["bbPipelineSigSets"] = ", ".join(sig_pipe) if sig_pipe else "none"
    mac["bbBaselineSigSets"] = ", ".join(sig_base) if sig_base else "none"
    sig_pipe_boot = [T[s]["short"] for s in slugs if S["reading_bootstrap"][s] == "pipeline" and S["reading"][s] != "pipeline"]
    mac["bbPipelineSigBootOnlySets"] = ", ".join(sig_pipe_boot) if sig_pipe_boot else "none"
    lines = ["% Generated by jev-opt/experiments/budget_report.py. Do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()]
    TEX.mkdir(parents=True, exist_ok=True)
    (TEX / "budget_baseline.tex").write_text("\n".join(lines) + "\n")

    # appendix tables
    R = ["% Generated by jev-opt/experiments/budget_report.py. Do not edit.",
         "\\begin{tabular}{@{}lrrrrlrrrr@{}}", "\\toprule",
         "Set & \\jev{} & B1 & B2 & B3 & Best & $D$ [95\\% CI] & Holm $p$ & 90\\% CI & Paper \\\\", "\\midrule"]
    for i, s in enumerate(slugs):
        t, c, F = T[s], T[s]["best_vs_jev"], T[s]["families"]
        ddag = "$^\\ddagger$"  # equivalent within ±3 points (TOST)
        dag = "$^\\dagger$"
        if i == 8:
            R.append("\\midrule")
        jsd = f"{{\\tiny$\\pm${f1(t['jev']['sd'])}}}" if len(t["jev"]["test"]) > 1 else ""
        R.append(f"{t['short']} & {f1(t['jev']['mean'])}{jsd} & {f1(F['emb']['test_mean'])} & "
                 f"{f1(F['setfit']['test_mean'])}{{\\tiny$\\pm${f1(F['setfit']['test_sd'])}}} & "
                 f"{f1(F['ft']['test_mean'])}{{\\tiny$\\pm${f1(F['ft']['test_sd'])}}} & B{1 + [f for f, *_ in FAMILIES].index(t['best'])} & "
                 f"${sgn(c['diff'])}$ [${sgn(c['ci'][0])}$, ${sgn(c['ci'][1])}$] & {fp(c['p_holm_small']).replace('<', '$<$')}"
                 f"{dag if 'p_jackknife' in c else ''} & "
                 f"[${sgn(E['per_set_3'][s]['ci90'][0])}$, ${sgn(E['per_set_3'][s]['ci90'][1])}$]"
                 f"{ddag if E['per_set_3'][s]['equivalent'] else ''} & "
                 f"{f1(t['paper'])} \\\\")
    R += ["\\midrule",
          f"Pooled (13) & & & & & & ${sgn(S['pooled']['all']['diff'])}$ [${sgn(S['pooled']['all']['ci'][0])}$, "
          f"${sgn(S['pooled']['all']['ci'][1])}$] & & [${sgn(E['pooled']['all_2']['ci90'][0])}$, "
          f"${sgn(E['pooled']['all_2']['ci90'][1])}$] & \\\\",
          f"Pooled (12, no TB-Dense) & & & & & & ${sgn(S['pooled']['no_tbdense']['diff'])}$ "
          f"[${sgn(S['pooled']['no_tbdense']['ci'][0])}$, ${sgn(S['pooled']['no_tbdense']['ci'][1])}$] & & "
          f"[${sgn(E['pooled']['no_tbdense_2']['ci90'][0])}$, ${sgn(E['pooled']['no_tbdense_2']['ci90'][1])}$] & \\\\",
          "\\bottomrule", "\\end{tabular}"]
    C = ["% Generated by jev-opt/experiments/budget_report.py. Do not edit.",
         "\\begin{tabular}{@{}llrr@{}}", "\\toprule", "Set & Chosen configuration (B1 / B2 / B3) & Val (B1/B2/B3) & Clusters \\\\",
         "\\midrule"]
    for s in slugs:
        t, F = T[s], T[s]["families"]
        cfgs = " / ".join(cfg_short(f, F[f]["chosen"]) for f, *_ in FAMILIES).replace("α", "$\\alpha$")
        vals = "/".join(f1(F[f]["val_mean"]) for f, *_ in FAMILIES)
        C.append(f"{t['short']} & {cfgs} & {vals} & {t['cluster_unit']} ({t['n_clusters']:,}) \\\\")
    C += ["\\bottomrule", "\\end{tabular}"]
    (TEX / "app_budget_baseline.tex").write_text("\n".join(R) + "\n")
    (TEX / "app_budget_baseline_cfg.tex").write_text("\n".join(C) + "\n")


if __name__ == "__main__":
    sys.exit(main())
