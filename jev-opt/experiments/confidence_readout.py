"""RQ3 confidence: the paper's question zero-shot vs the feature-question-search readout -> results/confidence.json["rq3"].

For the 8 study tasks and optimizer seeds 0/1/2 (the programs chosen by the seed-variance rule, results/seeds.json
["selection"]), on the TEST split, offline: answers from the Jev cache with posting disabled (a cache miss
raises), no Jev or LLM call, no test split consumed (the test predictions were recorded when each run consumed
its test split).

DEFINITIONS, revision 2 (fixed before the revision-2 results were computed).
Revision 2 follows Rao & Callison-Burch (2026, arXiv:2609.29769v2): bands with fewer than 20 items are omitted;
calibration uses a probability (never Jev's confidence field, which is reported only as a labelled sensitivity).
==============================================================================================================
Zero-shot = the paper's question in the zero-shot evaluation (runs/<gauntlet task>/, recorded predictions and
responses), on the same test items (matched by id; gold labels asserted equal). Its probability vector = Jev's
option probabilities (Choice) or level probabilities (DREsS Score, level i -> label 1.0 + 0.5 i), renormalised
to sum 1 over the task's labels; its prediction = the recorded one (argmax).
Readout = the selected program's fitted readout. Its probabilities are rebuilt offline: every readout is reset
to unfitted and refit on the run's train fit slice exactly as the run fit it (fit_program with the run's
optimizer settings, answers from the cache; jev-opt/experiments/framing_features.refit), then run on test.
Before any probability is used we assert (a) the stored program re-executed on test reproduces the
recorded test score and confusion matrix (test_consumed.json) and (b) the refit program's test predictions
equal the stored program's on 100% of items.
Probability = the probability the method assigns to its predicted label (zero-shot: the argmax option's
probability; lr readouts: the argmax class's; DREsS ridge readouts, whose decision is the band of the
regression output and not always the distribution's argmax: the probability of the predicted level under the
readout's normal(y, sigma) spread). It is used both to calibrate (ECE, Brier) and, for both methods alike, to
rank (AUROC, bands). Correct = prediction equals gold (DREsS: the exact level). For DREsS we also report the
mean absolute level error (in levels of 0.5 points) per band and the AUROC for "within one level". Sensitivity
row (labelled; zero-shot only): Jev's `confidence` field as the ranking confidence (its ECE row applies the ECE
formula to the field, which is not a probability).
Statistics (per method): ECE (15 equal-width bins, confidence as above), Brier (half the squared error of the
renormalised probability vector over the task's labels, in [0, 1]; jev_opt.metrics.example_brier), AUROC of
confidence for correctness (ties 1/2), error rate per band [0,.25), [.25,.5), [.5,.75), [.75,1] (a band with
fewer than 20 items is omitted: zero-shot per task, readout pooled over 3 seeds when under 60, top-band error
rate undefined when the top band holds under 20 items) and the share of errors in the top band. Differences
readout minus zero-shot with a paired percentile bootstrap (2,000
resamples of test items, the same resample for both methods and all seeds of a task; seed 0, numpy
SeedSequence(0) spawned into one stream per task in TASKS order). Seed means: the statistic averaged over the
3 seeds; the 8-task summary averages tasks with equal weight (tasks resampled independently).

Revision 3 (fixed before the revision-3 results were computed) follows prior work and adds:
  - ECE is equal-mass (15 bins cut at the probability's n/15 quantiles, moved to tie boundaries; ties never split),
    with a per-method noise floor (1,000 label draws from the method's own probabilities; floor = mean, band = 95th
    percentile); equal-width ECE is a sensitivity column. Brier as above; DREsS also RPS over the 9 ordered levels.
  - Generic-calibrator controls, fitted on the same 400 fit-slice train items the readout was fit on (the v3 "A"
    runs' fit slice, asserted identical to the P runs'), with the paper's question's answers from the cache
    (jev-opt seed program of runs/jev-opt/v3-<task>-A); every distribution is floored at 1e-4 (jev_opt's BIAS_FLOOR /
    FEATURE_FLOOR, since Jev returns exact zeros) and renormalised: (a) temperature scaling, T = jev_opt.fit.
    fit_temperature (train log loss of p ** (1/T), T in [0.05, 20]); sensitivity: T minimising the fit-slice Brier
    score instead (log loss is dominated by gold labels Jev gave ~0); applied to the zero-shot test distribution with
    the recorded zero-shot prediction kept; (b) per-class log-prob biases + temperature, jev_opt.fit.fit_label_bias
    with the task's primary metric, applied with jev_opt.execute.combine (its argmax is the prediction). Reported
    with the same statistics and paired bootstrap; the question is whether the readout beats a generic calibrator.
  - Across tasks: two-sided exact sign test over the 8 tasks (readout seed mean vs zero-shot, vs temperature scaling)
    next to the item bootstrap.

Run from the repository root (Python 3.11):
  uv run --project jev-opt/python python jev-opt/experiments/confidence_readout.py
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent))
import framing_features as ff  # noqa: E402  (patches JevClient: a cache miss raises)

from jev_opt.adapter import train_roles  # noqa: E402
from jev_opt.execute import combine  # noqa: E402
from jev_opt.fit import fit_label_bias, fit_temperature  # noqa: E402
from jev_opt.ir import load_decider  # noqa: E402
from jev_opt.runs import _load_run_config  # noqa: E402

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
from confidence_zeroshot import BAND_NAMES, BANDS, ECE_BINS, FLOOR_DRAWS, MIN_BAND, SetStats, ci  # noqa: E402

RUNS = ROOT / "runs"
OUT = ROOT / "results" / "confidence.json"
B = 2000
GAUNTLET = {"contempo-tbdense": ("contempo/tbdense", None), "ezstance-subtaskA": ("ezstance/subtaskA", None),
            "smarter-implicit-hate": ("smarter/implicit_hate", None), "edit-intent": ("edit_intent", None),
            "inli": ("inli", None), "dress-new-content": ("dress_new", "content"),
            "dress-new-organization": ("dress_new", "organization"), "dress-new-language": ("dress_new", "language")}
STATS = ("ece", "brier", "auroc", "confident_error_share", "top_band_error_rate")


def normalise(p: dict[str, float], labels: list[str]) -> np.ndarray:
    v = np.array([max(float(p.get(lab, 0.0)), 0.0) for lab in labels])
    return v / v.sum() if v.sum() > 0 else np.full(len(labels), 1.0 / len(labels))


def zero_shot(slug: str, labels: list[str]) -> dict[str, dict[str, Any]]:
    task, rubric = GAUNTLET[slug]
    field: dict[str, float] = {}
    probs: dict[str, dict[str, float]] = {}
    for shard in sorted((RUNS / task).glob("responses*.jsonl.gz")):
        for line in gzip.open(shard, "rt"):
            r = json.loads(line)
            ans = r["response"]["answers"]
            a = ans[rubric] if rubric else next(iter(ans.values()))
            if rubric:
                assert len(ans) == 3
            else:
                assert len(ans) == 1
            field[r["item_id"]] = float(a["confidence"])
            if rubric:
                probs[r["item_id"]] = {f"{1.0 + 0.5 * int(i):.1f}": float(v) for i, v in a["probabilities"].items()}
    out = {}
    for line in (RUNS / task / "predictions.jsonl").read_text().splitlines():
        r = json.loads(line)
        iid = r["item_id"]
        if rubric:
            pred, gold, p = f"{float(r['prediction'][rubric]):.1f}", f"{float(r['gold'][rubric]):.1f}", probs[iid]
            key = f"dress-{iid}"
        else:
            pred, gold, p = r["prediction"]["label"], r["gold"], r["prediction"]["probabilities"]
            key = iid
        assert set(p) <= set(labels), (slug, set(p) - set(labels))
        out[key] = {"pred": pred, "gold": gold, "dist": normalise(p, labels), "field": field[iid]}
    return out


class Method:
    """Per-item arrays of one method on the test items."""

    def __init__(self, gold: list[str], pred: list[str], dist: np.ndarray, labels: list[str], conf: np.ndarray | None = None):
        idx = {lab: i for i, lab in enumerate(labels)}
        g = np.array([idx[x] for x in gold])
        y = np.array([idx[x] for x in pred])
        self.correct = (g == y).astype(float)
        self.conf = dist[np.arange(len(g)), y] if conf is None else conf
        onehot = np.eye(len(labels))[g]
        self.brier = 0.5 * ((dist - onehot) ** 2).sum(axis=1)
        cum = np.cumsum(dist, axis=1)[:, :-1]
        obs = (np.arange(len(labels) - 1)[None, :] >= g[:, None]).astype(float)
        self.rps = ((cum - obs) ** 2).sum(axis=1) / (len(labels) - 1)
        self.abs_err = np.abs(g - y).astype(float)
        p = np.clip(self.conf, 0.0, 1.0)
        self.st = SetStats(self.conf, self.correct, p, self.brier)
        self.st_adj = SetStats(self.conf, (self.abs_err <= 1).astype(float), p)
        self.top_ok = int((self.st.band == 3).sum()) >= MIN_BAND

    def stats(self, w: np.ndarray, ordinal: bool) -> dict[str, float]:
        nb, eb = self.st.bands(w)
        out = {"ece": self.st.ece(w), "ece_equal_width": self.st.ece_width(w), "brier": self.st.brier(w), "auroc": self.st.auroc(w),
               "confident_error_share": float(eb[3] / eb.sum()) if eb.sum() else float("nan"),
               "top_band_error_rate": float(eb[3] / nb[3]) if self.top_ok and nb[3] else float("nan"),
               "accuracy": float((w * self.correct).sum() / w.sum()), "mean_conf": float((w * self.conf).sum() / w.sum())}
        if ordinal:
            out["auroc_within_one_level"] = self.st_adj.auroc(w)
            out["rps"] = float((w * self.rps).sum() / w.sum())
        return out

    def floor(self, rng: np.random.Generator) -> dict[str, float]:
        return self.st.floor(rng)

    def bands(self, ordinal: bool) -> list[dict[str, Any]]:
        out = []
        for b in range(4):
            m = self.st.band == b
            row = {"band": BAND_NAMES[b], "n": int(m.sum()), "errors": int((1 - self.correct[m]).sum()),
                   "error_rate": float(1 - self.correct[m].mean()) if m.sum() >= MIN_BAND else None}
            if ordinal:
                row["mean_abs_level_error"] = float(self.abs_err[m].mean()) if m.sum() >= MIN_BAND else None
            out.append(row)
        return out


def run_seed(slug: str, sel: dict[str, Any], zs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    d = RUNS / "jev-opt" / sel["run"]
    consumed = json.loads((d / "test_consumed.json").read_text())
    rep = sel["representative"]
    assert consumed["representative"] == rep
    program = load_decider((d / f"{rep}.yaml").read_text())
    env = ff.Env(d)
    try:
        labels = list(env.config.labels)
        res = env.run(program, env.test, "test")
        gold, pred = [r.gold for r in res], [r.pred for r in res]
        score = env.primary(gold, pred)
        if abs(score - consumed["metrics"]["primary"]) > 1e-9:
            raise SystemExit(f"{d.name}: re-executed test {score} != recorded {consumed['metrics']['primary']}")
        conf_rec = consumed["metrics"]["confusion"]
        cl = conf_rec["labels"]
        mat = [[sum(g == a and p == b for g, p in zip(gold, pred)) for b in cl] for a in cl]
        if mat != conf_rec["matrix"]:
            raise SystemExit(f"{d.name}: confusion matrix differs from the recorded one")
        fitted = ff.refit(env, ff.unfitted(program, drop_aux=False))
        ref = env.run(fitted, env.test, "test")
        agree = sum(a.pred == b.pred for a, b in zip(ref, res)) / len(res)
        if agree != 1.0:
            raise SystemExit(f"{d.name}: refit test predictions agree on {agree:.4f} of items, not 100%")
        dist = np.array([[r.dist.get(lab, 0.0) for lab in labels] for r in ref])
        dist_stored = np.array([[r.dist.get(lab, 0.0) for lab in labels] for r in res])
        ids = [r.id for r in res]
        z = [zs[i] for i in ids]
        assert [x["gold"] for x in z] == gold, f"{d.name}: zero-shot gold differs"
        return {"run": d.name, "representative": rep, "candidate": consumed["candidate"], "ids": ids, "gold": gold,
                "pred": pred, "dist": dist, "labels": labels, "primary": score,
                "checks": {"recorded_primary": consumed["metrics"]["primary"], "confusion_matches": True,
                           "refit_test_agree": agree,
                           "max_abs_prob_diff_refit_vs_stored": float(np.abs(dist - dist_stored).max())}}
    finally:
        env.close()


def calibrator_controls(slug: str, p_run: str, labels: list[str]) -> dict[str, Any]:
    """Temperature scaling and per-class bias + temperature on the paper's question, fit on the readout's fit slice."""
    a_dir = RUNS / "jev-opt" / f"v3-{slug}-A"
    env_p = ff.Env(RUNS / "jev-opt" / p_run)
    env_a = ff.Env(a_dir)
    try:
        fit_p, _ = train_roles(env_p.train, env_p.config.optimizer)
        fit_a, _ = train_roles(env_a.train, env_a.config.optimizer)
        assert [e.id for e in fit_a] == [e.id for e in fit_p], f"{slug}: A and P fit slices differ"
        assert list(env_a.config.labels) == labels
        seed = load_decider((a_dir / "seed.yaml").read_text())
        res = env_a.run(seed, fit_a, "train")
        dists = [floored(r.dist, labels) for r in res]
        gold = [r.gold for r in res]
        temp = fit_temperature(labels, dists, gold)
        bias, temp_b, rep = fit_label_bias(labels, [[d] for d in dists], gold, env_a.config.primary_metric, env_a.config.negative_label)
        return {"n_fit": len(res), "fit_accuracy": sum(r.pred == r.gold for r in res) / len(res), "temperature": temp,
                "temperature_brier": brier_temperature(labels, dists, gold),
                "bias": bias, "bias_temperature": temp_b, "bias_fit_primary_before_after": [rep["before"], rep["after"]]}
    finally:
        env_a.close()
        env_p.close()


FLOOR = 1e-4  # jev_opt's BIAS_FLOOR / FEATURE_FLOOR


def floored(d: dict[str, float], labels: list[str]) -> dict[str, float]:
    v = {lab: max(float(d.get(lab, 0.0)), FLOOR) for lab in labels}
    s = sum(v.values())
    return {lab: x / s for lab, x in v.items()}


def brier_temperature(labels: list[str], dists: list[dict[str, float]], gold: list[str]) -> float:
    p = np.array([[d[lab] for lab in labels] for d in dists])
    onehot = np.eye(len(labels))[[labels.index(g) for g in gold]]

    def loss(t: float) -> float:
        q = p ** (1.0 / t)
        q /= q.sum(axis=1, keepdims=True)
        return float(0.5 * ((q - onehot) ** 2).sum(axis=1).mean())
    return round(float(minimize_scalar(loss, bounds=(0.05, 20.0), method="bounded").x), 3)


def apply_control(z: list[dict[str, Any]], labels: list[str], bias: dict[str, float], temp: float, keep_pred: bool) -> tuple[list[str], np.ndarray]:
    preds, dists = [], []
    for x in z:
        dist, pred = combine(labels, [floored(dict(zip(labels, x["dist"])), labels)], bias, temp)
        preds.append(x["pred"] if keep_pred else pred)
        dists.append([dist[lab] for lab in labels])
    return preds, np.array(dists)


def main() -> int:
    selection = json.loads((ROOT / "results" / "seeds.json").read_text())["selection"]
    streams = np.random.SeedSequence(0).spawn(len(ff.TASKS))
    tasks_out: dict[str, Any] = {}
    task_boot: dict[str, dict[str, np.ndarray]] = {}
    task_point: dict[str, dict[str, float]] = {}
    for (slug, title), ss in zip(ff.TASKS, streams):
        boot_ss, floor_ss = ss.spawn(2)
        ordinal = slug.startswith("dress")
        sels = [selection[slug][str(s)] for s in ff.SEEDS]
        zs = zero_shot(slug, list(_load_run_config(RUNS / "jev-opt" / sels[0]["run"]).labels))
        runs = [run_seed(slug, sel, zs) for sel in sels]
        ids, labels, gold = runs[0]["ids"], runs[0]["labels"], runs[0]["gold"]
        assert all(r["ids"] == ids for r in runs)
        z = [zs[i] for i in ids]
        ctrl = calibrator_controls(slug, sels[0]["run"], labels)
        zm = Method(gold, [x["pred"] for x in z], np.stack([x["dist"] for x in z]), labels)
        zf = Method(gold, [x["pred"] for x in z], np.stack([x["dist"] for x in z]), labels, conf=np.array([x["field"] for x in z]))
        tm = Method(gold, *apply_control(z, labels, {}, ctrl["temperature"], True), labels)
        bm = Method(gold, *apply_control(z, labels, ctrl["bias"], ctrl["bias_temperature"], False), labels)
        sm = Method(gold, *apply_control(z, labels, {}, ctrl["temperature_brier"], True), labels)
        rms = [Method(gold, r["pred"], r["dist"], labels) for r in runs]
        n = len(ids)
        one = np.ones(n)
        rng = np.random.default_rng(boot_ss)
        boots: dict[str, list[dict[str, float]]] = {"z": [], "zf": [], "ts": [], "bts": [], "tsb": []}
        boot_r: list[list[dict[str, float]]] = [[] for _ in rms]
        for _ in range(B):
            w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
            for key, m in (("z", zm), ("zf", zf), ("ts", tm), ("bts", bm), ("tsb", sm)):
                boots[key].append(m.stats(w, ordinal))
            for k, m in enumerate(rms):
                boot_r[k].append(m.stats(w, ordinal))
        arr = lambda rows, k: np.array([row[k] for row in rows], float)  # noqa: E731
        zp, zfp, tp, bp = (m.stats(one, ordinal) for m in (zm, zf, tm, bm))
        keys = list(zp)
        frng = np.random.default_rng(floor_ss)
        floors = {"zero_shot": zm.floor(frng), "temperature_scaled": tm.floor(frng), "bias_temperature": bm.floor(frng),
                  "readout_seeds": [m.floor(frng) for m in rms]}
        seeds_out = {}
        for k, (r, m) in enumerate(zip(runs, rms)):
            rp = m.stats(one, ordinal)
            seeds_out[str(ff.SEEDS[k])] = {
                "run": r["run"], "candidate": r["candidate"], "primary": r["primary"], "checks": r["checks"],
                "readout": rp, "readout_bands": m.bands(ordinal),
                "delta": {s: {"delta": rp[s] - zp[s], "ci": ci(arr(boot_r[k], s) - arr(boots["z"], s))} for s in keys},
            }
        mean_r = {s: float(np.mean([seeds_out[str(x)]["readout"][s] for x in ff.SEEDS])) for s in keys}
        mean_r_boot = {s: np.mean([arr(boot_r[k], s) for k in range(len(rms))], axis=0) for s in keys}
        # Seed-pooled error by band (each seed weighted equally; same items).
        pooled_bands = []
        for b in range(4):
            nb = sum(int((m.st.band == b).sum()) for m in rms)
            eb = sum(int((1 - m.correct[m.st.band == b]).sum()) for m in rms)
            ok = nb >= len(rms) * MIN_BAND  # mean per seed >= 20
            row = {"band": BAND_NAMES[b], "n": nb, "errors": eb, "error_rate": eb / nb if ok else None}
            if ordinal:
                row["mean_abs_level_error"] = (float(sum(m.abs_err[m.st.band == b].sum() for m in rms) / nb) if ok else None)
            pooled_bands.append(row)
        delta = lambda a, ab, b, bb: {s: {"delta": a[s] - b[s], "ci": ci(ab(s) - bb(s))} for s in keys}  # noqa: E731
        zb = lambda s: arr(boots["z"], s)  # noqa: E731
        tb = lambda s: arr(boots["ts"], s)  # noqa: E731
        btb = lambda s: arr(boots["bts"], s)  # noqa: E731
        rb = lambda s: mean_r_boot[s]  # noqa: E731
        tasks_out[slug] = {
            "title": title, "n_test": n, "ordinal": ordinal, "gauntlet_task": GAUNTLET[slug][0],
            "zero_shot": zp, "zero_shot_bands": zm.bands(ordinal),
            "zero_shot_jev_confidence_field": zfp, "zero_shot_jev_confidence_field_bands": zf.bands(ordinal),
            "controls": ctrl, "temperature_scaled": tp, "temperature_scaled_bands": tm.bands(ordinal),
            "bias_temperature": bp, "bias_temperature_bands": bm.bands(ordinal), "ece_floor": floors,
            "seeds": seeds_out, "readout_seed_mean": mean_r, "readout_bands_pooled_seeds": pooled_bands,
            "delta_seed_mean": delta(mean_r, rb, zp, zb),
            "delta_readout_vs_temperature": delta(mean_r, rb, tp, tb),
            "delta_readout_vs_bias_temperature": delta(mean_r, rb, bp, btb),
            "delta_temperature_vs_zero_shot": delta(tp, tb, zp, zb),
            "temperature_brier_scaled": sm.stats(one, ordinal),
            "delta_readout_vs_temperature_brier": delta(mean_r, rb, sm.stats(one, ordinal), lambda s: arr(boots["tsb"], s)),
            "delta_seed_mean_vs_field": {s: {"delta": mean_r[s] - zfp[s], "ci": ci(mean_r_boot[s] - arr(boots["zf"], s))}
                                         for s in ("ece", "auroc", "confident_error_share", "top_band_error_rate")},
            "zero_shot_ci": {s: ci(zb(s)) for s in keys},
            "readout_seed_mean_ci": {s: ci(mean_r_boot[s]) for s in keys},
            "temperature_scaled_ci": {s: ci(tb(s)) for s in keys},
        }
        task_boot[slug] = ({f"z_{s}": zb(s) for s in keys} | {f"r_{s}": mean_r_boot[s] for s in keys}
                           | {f"t_{s}": tb(s) for s in keys} | {f"b_{s}": btb(s) for s in keys})
        task_point[slug] = ({f"z_{s}": zp[s] for s in keys} | {f"r_{s}": mean_r[s] for s in keys}
                            | {f"t_{s}": tp[s] for s in keys} | {f"b_{s}": bp[s] for s in keys})
        print(f"{slug}: ece zero {zp['ece']:.3f} TS {tp['ece']:.3f} bias+TS {bp['ece']:.3f} readout {mean_r['ece']:.3f} | "
              f"brier {zp['brier']:.3f}/{tp['brier']:.3f}/{bp['brier']:.3f}/{mean_r['brier']:.3f} | auroc "
              f"{zp['auroc']:.3f}/{tp['auroc']:.3f}/{bp['auroc']:.3f}/{mean_r['auroc']:.3f} | T={ctrl['temperature']} "
              f"refit {[r['checks']['refit_test_agree'] for r in runs]}", flush=True)

    summary = {}
    for s in ("ece", "ece_equal_width", "brier", "auroc", "confident_error_share", "top_band_error_rate", "accuracy", "mean_conf"):
        # Tasks where every method defines the statistic (a readout that puts no item in the top band has no
        # top-band error rate).
        ts = [t for t in task_point if not any(np.isnan(task_point[t][f"{m}_{s}"]) for m in "zrtb")]
        mean = {m: float(np.mean([task_point[t][f"{m}_{s}"] for t in ts])) for m in "zrtb"}
        mb = {m: np.nanmean(np.stack([task_boot[t][f"{m}_{s}"] for t in ts]), axis=0) for m in "zrtb"}

        def sign(a: str, b: str) -> float:
            lower = sum(task_point[t][f"{a}_{s}"] < task_point[t][f"{b}_{s}"] for t in ts)
            differ = sum(task_point[t][f"{a}_{s}"] != task_point[t][f"{b}_{s}"] for t in ts)
            return float(binomtest(lower, differ).pvalue) if differ else 1.0
        summary[s] = {"zero_shot": mean["z"], "readout": mean["r"], "temperature_scaled": mean["t"], "bias_temperature": mean["b"],
                      "delta": mean["r"] - mean["z"], "ci": ci(mb["r"] - mb["z"]),
                      "delta_vs_temperature": mean["r"] - mean["t"], "ci_vs_temperature": ci(mb["r"] - mb["t"]),
                      "delta_vs_bias_temperature": mean["r"] - mean["b"], "ci_vs_bias_temperature": ci(mb["r"] - mb["b"]),
                      "delta_temperature_vs_zero": mean["t"] - mean["z"], "ci_temperature_vs_zero": ci(mb["t"] - mb["z"]),
                      "tasks": ts,
                      "n_tasks_readout_lower": sum(task_point[t][f"r_{s}"] < task_point[t][f"z_{s}"] for t in ts),
                      "n_tasks_readout_lower_than_temperature": sum(task_point[t][f"r_{s}"] < task_point[t][f"t_{s}"] for t in ts),
                      "n_tasks_readout_lower_than_bias_temperature": sum(task_point[t][f"r_{s}"] < task_point[t][f"b_{s}"] for t in ts),
                      "sign_p_readout_vs_zero": sign("r", "z"), "sign_p_readout_vs_temperature": sign("r", "t")}
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out["rq3"] = {
        "generated_by": "jev-opt/experiments/confidence_readout.py",
        "definitions_revision": 3,
        "bootstrap": {"B": B, "seed": 0, "paired": True, "unit": "test item"},
        "bands": list(BAND_NAMES), "band_edges": list(BANDS), "ece_bins": ECE_BINS, "ece_estimator": "equal-mass, ties kept together",
        "floor_draws": FLOOR_DRAWS,
        "tasks": tasks_out, "summary_8_tasks": summary,
    }
    OUT.write_text(json.dumps(out, indent=1, allow_nan=True) + "\n")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
