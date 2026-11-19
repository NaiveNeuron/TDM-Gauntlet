"""RQ3 offline reanalysis (slug `rq3`): gap closed, seed-aware paired comparison with GEPA + demonstrations +
calibration, and the full label budget, for the 8 optimization-study task configurations.

STRICTLY OFFLINE. Every test-evaluated representative is re-executed from its run's Jev cache through
`analyze_test_v3.evaluate` (importing it replaces `JevClient._post` by a function that raises `CacheMiss`, so a
cache miss is an error and no request can be made); each recomputed score must equal the score recorded at test
consumption. kNN demonstrations use the on-disk embedding cache (no LLM). No Jev or LLM call is made.

Systems (all on the frozen zero-shot test set of each task, same items, same order):
  zero-shot    = the paper's question, `study-<task>-zero` (`results/gepa_v3_test.json` row "v1 zero-shot").
  optimized    = feature-question search, optimizer seeds 0/1/2 (`results/seeds.json`; seed 0 = `v3-<task>-P`,
                 seeds 1/2 = `v3-<task>-P-s<seed>`), each the one representative tested under the seed-variance rule.
  GEPA+demos+cal = the pre-registered v2 choice (`v2-<task>-CD`, second optimization study): GEPA with kNN demonstrations,
                 class calibration and ensembles. Only ONE optimizer seed was ever test-evaluated (the jev-opt
                 ledger allows one test consumption per run; the v2 seed-1/2 runs were val-only), so the baseline
                 enters every comparison as a single fixed seed.
  paper model  = the paper's task-specific (fine-tuned) method, `results/paper_numbers.json` (`method` value;
                 DREsS: per-rubric content/organization/language entry of `paper_per_rubric_content_org_lang`).

(a) Gap closed, per set: G = (S - Z) / (P - Z), with S the system's test score, Z the zero-shot test score and P
    the paper model. Optimized: per seed G_s, and the 3-seed mean G = (mean_s S_s - Z)/(P - Z) = mean_s G_s.
    Summaries: median over the 8 sets; per paper (sets of one paper averaged first: DREsS content/organization/
    language -> one DREsS value; 6 papers), median over papers; median over the 5 non-DREsS sets. Same for
    GEPA+demos+cal (one seed).

(b) Paired comparison, per task: D = mean over seeds s of [M(opt_s) - M(GEPA+demos+cal)] on the test set, M = the
    task's primary metric (micro-F1 with VAGUE negative, macro-F1, accuracy or QWK, as `jev_opt.metrics.primary`,
    recomputed from bootstrap-weighted confusion matrices and checked against it). Two-level (hierarchical)
    bootstrap, B = 10,000, numpy default_rng(0) per task: each replicate resamples test clusters with replacement
    in the RQ1 statistics' unit (resample.py: TB-Dense and edit intent: document; INLI: premise; EZ-STANCE: tweet;
    DREsS: writing prompt; the
    same resample for every system, so the comparison is paired) and, independently, 3 optimizer seeds with
    replacement from {0, 1, 2}; the replicate statistic is the mean over the drawn seeds of M(opt_s) - M(base).
    95% percentile interval; two-sided bootstrap p = min(1, 2 min(P*(D <= 0), P*(D >= 0))), floored at 1/B; Holm
    over the 8 tasks. Pooled: the per-replicate mean of D over the 8 tasks (each task resampled independently),
    and the paper-weighted mean (DREsS's three rubrics averaged first, 6 papers). Also reported: per-seed paired
    differences with cluster-only bootstrap intervals (same resamples), and for the clustered tasks the same
    hierarchical bootstrap with single items as the resampling unit (sensitivity; TB-Dense has few test documents).
    The baseline's seed variance cannot be represented (one tested seed).

(c) Label budget per task and method, counted from each run's config (`config.yaml`), the split files
    (`load_split`), `jev_opt.adapter.train_roles` (fit slice / GEPA reflection pool), the chosen program files
    (readout fit n) and the demonstrations actually retrieved on test (DemoPool.select instrumented during the
    offline re-execution). Roles: fit slice (numbers/readouts fit, CV scoring in the search), GEPA reflection
    pool, kNN demonstration pool, validation (search and/or selection). Test labels are used only for the one
    reported score per run.

(d) Optimized seed mean against the paper's model, with RQ1's inference (scripts/offline_rq1.py): SE = SD of the
    seed mean over the replicates of (b) (clusters and seeds resampled); SE x sqrt(2) where RQ1 widens the set
    because the published value is on other items (SMARTER, DREsS, EZ-STANCE); normal 95% interval of the
    difference, difference test and TOST at 1 point, Holm over the 8 configurations per family; class above /
    below / parity / inconclusive as in RQ1.

Run from the repository root (Python 3.11):
  uv run --project jev-opt/python python jev-opt/experiments/offline_rq3.py
Writes results/offline_v4/rq3.json and paper/generated/offline_rq3_{gap,paired,budget}.tex.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from analyze_test_v3 import DISCREPANCIES, evaluate  # noqa: E402  (import disables Jev posting: cache-only)
from jev_opt import metrics  # noqa: E402
from jev_opt.adapter import train_roles  # noqa: E402
from jev_opt.demos import DemoPool  # noqa: E402
from jev_opt.hygiene import allow_test_split, load_split  # noqa: E402
from resample import cluster_index  # noqa: E402
from jev_opt.runs import _load_run_config  # noqa: E402

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
import paper_tables  # noqa: E402  (labels of the configurations: CONFIG_LABEL)
RUNS = ROOT / "runs" / "jev-opt"
OUT = ROOT / "results" / "offline_v4" / "rq3.json"
TEX = ROOT / "paper" / "generated"
B = 10_000
CHUNK = 500
SEEDS = ("0", "1", "2")

# slug, macro tag, paper id (sets of one paper are aggregated first), paper_numbers key, DREsS rubric index
TASKS = [
    ("contempo-tbdense", "Tbd", "ConTempo", "contempo/tbdense", None),
    ("ezstance-subtaskA", "Ez", "EZ-STANCE", "ezstance/subtaskA", None),
    ("smarter-implicit-hate", "Sm", "SMARTER", "smarter/implicit_hate", None),
    ("edit-intent", "Ed", "Re3-Sci", "edit_intent", None),
    ("inli", "Inli", "INLI", "inli", None),
    ("dress-new-content", "DrC", "DREsS", "dress_new", 0),
    ("dress-new-organization", "DrO", "DREsS", "dress_new", 1),
    ("dress-new-language", "DrL", "DREsS", "dress_new", 2),
]
BASELINES = {"v1 text-opt": "study-{}-text", "v1 program-opt": "study-{}-program", "tailored": "tailored-{}"}

# ---- instrument kNN demonstration retrieval (records which TRAIN labels are shown on test) -------------------
DEMO_LOG: list[str] = []
_select = DemoPool.select


def _logged_select(self, spec, ex):
    out = _select(self, spec, ex)
    if ex.split == "test":
        DEMO_LOG.extend(d.id for d in out)
    return out


DemoPool.select = _logged_select


def run_eval(name: str) -> dict:
    DEMO_LOG.clear()
    r = evaluate(RUNS / name)
    r["demo_ids"] = sorted(set(DEMO_LOG))
    r["demo_shown"] = len(DEMO_LOG)
    r["run"] = name
    return r


# ---- metrics from bootstrap-weighted confusion matrices ---------------------------------------------------------
def codes(r: dict) -> tuple[np.ndarray, int]:
    c = r["config"]
    labels = list(c.labels)
    k = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    g = np.array([idx[x] for x in r["gold"]])
    if c.primary_metric == "qwk":  # failed predictions count as the farthest level from gold (metrics.primary)
        p = np.array([idx[p] if p in idx else (0 if idx[gl] >= k / 2 else k - 1) for gl, p in zip(r["gold"], r["pred"])])
    else:  # error column k
        p = np.array([idx.get(p, k) if p is not None else k for p in r["pred"]])
    return g * (k + 1) + p, k


def metric_from_counts(cm: np.ndarray, name: str, k: int, neg: int | None) -> np.ndarray:
    """cm: (b, k, k+1) weighted confusion counts (gold rows, predicted columns, last column = failed)."""
    diag = np.einsum("bii->bi", cm[:, :, :k])
    gold_n = cm.sum(axis=2)
    pred_n = cm[:, :, :k].sum(axis=1)
    if name == "accuracy":
        return diag.sum(1) / cm.sum((1, 2))
    if name == "macro_f1":
        den = gold_n + pred_n
        return np.where(den > 0, 2 * diag / np.where(den > 0, den, 1), 0.0).mean(1)
    if name == "micro_f1":
        pos = np.array([i != neg for i in range(k)])
        tp, den = diag[:, pos].sum(1), gold_n[:, pos].sum(1) + pred_n[:, pos].sum(1)
        return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    if name == "qwk":
        o = cm[:, :, :k]
        n = o.sum((1, 2))
        e = np.einsum("bi,bj->bij", o.sum(2), o.sum(1)) / n[:, None, None]
        w = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2
        return 1 - (o * w).sum((1, 2)) / (e * w).sum((1, 2))
    raise ValueError(name)


class Scorer:
    def __init__(self, systems: dict[str, dict]):
        any_r = next(iter(systems.values()))
        c = any_r["config"]
        self.name = c.primary_metric
        self.k = len(c.labels)
        self.neg = list(c.labels).index(c.negative_label) if c.negative_label else None
        self.onehot = {}
        for key, r in systems.items():
            code, _ = codes(r)
            oh = np.zeros((len(code), self.k * (self.k + 1)))
            oh[np.arange(len(code)), code] = 1.0
            self.onehot[key] = oh

    def score(self, key: str, w: np.ndarray) -> np.ndarray:
        cm = (w @ self.onehot[key]).reshape(len(w), self.k, self.k + 1)
        return metric_from_counts(cm, self.name, self.k, self.neg)


def clusters(slug: str, test_examples: list) -> tuple[np.ndarray, str]:
    """RQ1's resampling unit for the set (resample.py)."""
    return cluster_index(slug, [ex.id for ex in test_examples])


DELTA = 0.01  # RQ1's equivalence margin (1 point)
# RQ1's two-sample widening flag per set (scripts/offline_rq1.py WIDEN; the published value is on other items)
WIDEN_RQ1 = {s["task"]: bool(s["widen"]) for s in json.loads((ROOT / "results" / "offline_v4" / "rq1.json").read_text())["sets"]}


def phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


def boot_p(d: np.ndarray) -> float:
    return float(max(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())), 1 / len(d)))


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=ps.__getitem__)
    out, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = run
    return out


# ---- formatting -----------------------------------------------------------------------------------------------
def pts(x: float) -> str:
    s = f"{100 * x:+.1f}"
    return "0.0" if s in ("+0.0", "-0.0") else s


def pct(x: float) -> str:
    return f"{100 * x:.0f}"


def fp(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


# ---- (c) label budget helpers ---------------------------------------------------------------------------------
def readout_fit_n(program_yaml: str) -> list[int]:
    out: list[int] = []

    def walk(o):
        if isinstance(o, dict):
            for key, v in o.items():
                if key == "readout" and isinstance(v, dict) and isinstance(v.get("fit"), dict) and "n" in v["fit"]:
                    out.append(int(v["fit"]["n"]))
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(yaml.safe_load(program_yaml))
    return out


def split_sizes(run: str) -> tuple:
    cfg = _load_run_config(RUNS / run)
    train, val = load_split(cfg, "train"), load_split(cfg, "val")
    fit, refl = train_roles(train, cfg.optimizer)
    return cfg, train, val, fit, refl


def manifest_pool(cfg) -> str:
    man = Path(cfg.data.train).parent / "manifest.json"
    pools = json.loads(man.read_text())["pools"]
    for key, unit in (("train", ""), ("train_split_pairs", " pairs"), ("train_split", ""), ("train_premises", " premises")):
        if key in pools:
            return f"{pools[key]:,}{unit}"
    raise KeyError(man)


def main() -> int:
    paper_json = json.loads((ROOT / "results" / "paper_numbers.json").read_text())
    seeds_json = json.loads((ROOT / "results" / "seeds.json").read_text())
    gepa = json.loads((ROOT / "results" / "gepa_v3_test.json").read_text())["tasks"]
    raw: dict = {"tasks": {}}
    rows_gap, rows_paired, rows_budget = [], [], []
    diffs_task: dict[str, np.ndarray] = {}

    for t_i, (slug, tag, paper_id, pkey, rub) in enumerate(TASKS):
        short = paper_tables.CONFIG_LABEL[slug]
        pj = paper_json[pkey]
        paper = pj["paper_per_rubric_content_org_lang"][pj["method"]][rub] if rub is not None else pj["paper"][pj["method"]]
        assert abs(paper - gepa[slug]["paper"]) < 1e-12, (slug, paper, gepa[slug]["paper"])
        seed_runs = {s: seeds_json["tasks"][slug]["seeds"][s]["run"] for s in SEEDS}
        sysruns = {"zero": f"study-{slug}-zero", "v2": f"v2-{slug}-CD", **{f"opt{s}": r for s, r in seed_runs.items()},
                   **{k: v.format(slug) for k, v in BASELINES.items()}}
        res = {k: run_eval(r) for k, r in sysruns.items()}
        if DISCREPANCIES:
            raise SystemExit(f"re-executed scores differ from recorded ones: {DISCREPANCIES}")
        z = res["zero"]
        for k, r in res.items():
            assert r["ids"] == z["ids"] and r["gold"] == z["gold"], f"{k}: test set differs"
        # recomputed scores = saved ones
        for s in SEEDS:
            assert abs(res[f"opt{s}"]["primary"] - seeds_json["tasks"][slug]["seeds"][s]["test"]) < 1e-12
        assert abs(res["v2"]["primary"] - gepa[slug]["rows"]["v2 chosen"]["test"]) < 1e-12
        assert abs(z["primary"] - gepa[slug]["rows"]["v1 zero-shot"]["test"]) < 1e-12

        # (a) gap closed
        Z = z["primary"]
        opt = [res[f"opt{s}"]["primary"] for s in SEEDS]
        g = lambda x: (x - Z) / (paper - Z)  # noqa: E731
        gap = {"zero": Z, "paper": paper, "paper_method": pj["method"], "opt_test": opt, "opt_mean": statistics.mean(opt),
               "gap_seed": [g(x) for x in opt], "gap_mean": g(statistics.mean(opt)), "v2_test": res["v2"]["primary"],
               "gap_v2": g(res["v2"]["primary"]),
               "gap_other_baselines": {k: g(res[k]["primary"]) for k in BASELINES}}

        # (b) hierarchical bootstrap
        cfg_test = _load_run_config(RUNS / sysruns["zero"])
        with allow_test_split():
            test_ex = load_split(cfg_test, "test")
        assert [e.id for e in test_ex] == z["ids"]
        inv, unit = clusters(slug, test_ex)
        m = int(inv.max()) + 1
        keys = ["v2", *(f"opt{s}" for s in SEEDS)]
        sc = Scorer({k: res[k] for k in keys})
        ones = np.ones((1, len(inv)))
        for k in keys:  # vectorized metric == jev_opt.metrics.primary on the observed test set
            assert abs(sc.score(k, ones)[0] - res[k]["primary"]) < 1e-9, (slug, k)

        def hier_boot(inv: np.ndarray, rng: np.random.Generator, check: bool) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
            m = int(inv.max()) + 1
            D, per_seed, S = np.empty(B), {s: np.empty(B) for s in SEEDS}, np.empty(B)
            for lo in range(0, B, CHUNK):
                b = min(CHUNK, B - lo)
                draw = rng.integers(0, m, size=(b, m))
                cw = np.bincount((draw + m * np.arange(b)[:, None]).ravel(), minlength=b * m).reshape(b, m)
                w = cw[:, inv].astype(float)
                s_draw = rng.integers(0, len(SEEDS), size=(b, len(SEEDS)))
                base = sc.score("v2", w)
                M = np.stack([sc.score(f"opt{s}", w) for s in SEEDS])  # (3, b)
                for j, s in enumerate(SEEDS):
                    per_seed[s][lo:lo + b] = M[j] - base
                S[lo:lo + b] = M[s_draw, np.arange(b)[:, None]].mean(1)
                D[lo:lo + b] = S[lo:lo + b] - base
                if check and lo == 0:  # spot-check the weighted metric against metrics.primary on expanded resamples
                    c = res["v2"]["config"]
                    for row in range(3):
                        ix = np.repeat(np.arange(len(inv)), w[row].astype(int))
                        ref = metrics.primary(c.primary_metric, [z["gold"][i] for i in ix], [res["v2"]["pred"][i] for i in ix],
                                              c.labels, c.negative_label)
                        assert abs(ref - base[row]) < 1e-9, (slug, ref, base[row])
            return D, per_seed, S

        D, per_seed, S = hier_boot(inv, np.random.default_rng(0), True)
        diffs_task[slug] = D
        obs = statistics.mean(opt) - res["v2"]["primary"]
        lo95, hi95 = np.percentile(D, [2.5, 97.5])
        paired = {"diff": obs, "ci": [float(lo95), float(hi95)], "p": boot_p(D), "cluster_unit": unit, "n_clusters": m,
                  "n_items": len(inv), "baseline_seeds": 1, "opt_seeds": len(SEEDS),
                  "per_seed": {s: {"diff": opt[j] - res["v2"]["primary"], "ci": [float(x) for x in np.percentile(per_seed[s], [2.5, 97.5])],
                                   "p": boot_p(per_seed[s])} for j, s in enumerate(SEEDS)}}
        if unit != "item":  # sensitivity: the same hierarchical bootstrap with items as the resampling unit
            Di, _, _ = hier_boot(np.arange(len(inv)), np.random.default_rng(0), False)
            paired["item_level"] = {"ci": [float(x) for x in np.percentile(Di, [2.5, 97.5])], "p": boot_p(Di)}
        # (d) optimized seed mean against the paper's model with RQ1's inference: SE = SD of the seed mean over the
        # same cluster x seed resamples; x sqrt(2) where RQ1 widens the set (published value on other items).
        widen = WIDEN_RQ1[pkey]
        se = float(np.std(S, ddof=1))
        se_eff = se * (2 ** 0.5 if widen else 1.0)
        d_paper = statistics.mean(opt) - paper
        reach = {"delta": d_paper, "se": se, "se_eff": se_eff, "widen": widen,
                 "ci": [d_paper - 1.959964 * se_eff, d_paper + 1.959964 * se_eff],
                 "p_diff": 2 * phi(-abs(d_paper) / se_eff),
                 "p_tost": max(1 - phi((d_paper + DELTA) / se_eff), phi((d_paper - DELTA) / se_eff))}

        # (c) label budget
        bud: dict = {}
        cfg0, train, val, fit0, _ = split_sizes(seed_runs["0"])
        fits = {s: {e.id for e in train_roles(train, _load_run_config(RUNS / r).optimizer)[0]} for s, r in seed_runs.items()}
        for s, r in seed_runs.items():
            c = _load_run_config(RUNS / r)
            assert len(load_split(c, "train")) == len(train) and len(load_split(c, "val")) == len(val)
        opt_runs = {}
        for s, r in seed_runs.items():
            rd = RUNS / r
            rep = json.loads((rd / "test_consumed.json").read_text())["representative"]
            cands = [json.loads(p.read_text()) for p in (rd / "jev-population-opt" / "candidates").glob("*.json")]
            pop = json.loads((rd / "jev-population-opt" / "population.json").read_text())
            c = _load_run_config(rd)
            opt_runs[s] = {"run": r, "representative": rep, "readout_fit_n": readout_fit_n((rd / f"{rep}.yaml").read_text()),
                           "uses_demos": res[f"opt{s}"]["demo_shown"] > 0, "demo_train_ids_on_test": len(res[f"opt{s}"]["demo_ids"]),
                           "val_selection_pool": {"single": sum("single" in x.get("tags", []) for x in cands),
                                                  "stacked": sum("stacked" in x.get("tags", []) for x in cands)},
                           "rounds": pop["rounds_done"], "proposer_worst_examples_max": c.optimizer.population.worst_examples * pop["rounds_done"],
                           "round0_demo_variant": c.optimizer.population.demos is not None}
        assert len({len(f) for f in fits.values()}) == 1 and len(fit0) == len(fits["0"])
        bud["opt"] = {"train": len(train), "val": len(val), "fit_per_seed": len(fit0), "readout_folds": cfg0.optimizer.readout_folds,
                      "fit_union_3_seeds": len(set().union(*fits.values())), "demo_pool": len(train),
                      "total_distinct": len(train) + len(val), "runs": opt_runs}
        cfg2, train2, val2, fit2, refl2 = split_sizes(sysruns["v2"])
        assert {e.id for e in train2} == {e.id for e in train} and len(val2) == len(val)
        v2_rep = json.loads((RUNS / sysruns["v2"] / "test_consumed.json").read_text())["representative"]
        v2_runs = sorted(p.name for p in RUNS.glob(f"v2-{slug}-*") if p.is_dir())
        v2_cands = sum(len(list((RUNS / r).glob("*/candidates/*.json"))) for r in v2_runs)
        bud["v2"] = {"train": len(train2), "val": len(val2), "fit": len(fit2), "reflection_pool": len(refl2),
                     "fit_reflection_disjoint": not ({e.id for e in fit2} & {e.id for e in refl2}),
                     "demo_pool": len(train2) if cfg2.seed_demos is not None else 0, "representative": v2_rep,
                     "uses_demos": res["v2"]["demo_shown"] > 0, "demo_train_ids_on_test": len(res["v2"]["demo_ids"]),
                     "val_progressive_stages": list(cfg2.optimizer.progressive), "val_scored_candidates_all_v2_runs": v2_cands,
                     "v2_runs": v2_runs, "total_distinct": len({e.id for e in train2}) + len(val2)}
        for k, r in BASELINES.items():
            ck, trk, vk, fk, rk = split_sizes(r.format(slug))
            bud[k] = {"train": len(trk), "val": len(vk), "fit": len(fk), "reflection_pool": len(rk),
                      "demo_pool": len(trk) if ck.seed_demos is not None else 0, "uses_demos": res[k]["demo_shown"] > 0,
                      "val_progressive_stages": list(ck.optimizer.progressive), "total_distinct": len(trk) + len(vk)}
        cz, trz, vz, fz, _ = split_sizes(sysruns["zero"])
        bud["zero"] = {"fit": len(fz), "note": "temperature (confidence only); DREsS: Score cutoffs" if rub is not None else "temperature (confidence only)"}
        bud["paper_train_pool"] = manifest_pool(cfg0)
        raw["tasks"][slug] = {"short": short, "paper_id": paper_id, "gap": gap, "paired": paired, "reach": reach, "budget": bud}
        rows_gap.append((slug, short, tag, paper_id, gap))
        rows_paired.append((slug, short, tag, paired))
        rows_budget.append((slug, short, bud))
        print(f"{slug}: gap {pct(gap['gap_mean'])}% (v2 {pct(gap['gap_v2'])}%), D {pts(obs)} [{pts(lo95)}, {pts(hi95)}] p={paired['p']:.4f}", flush=True)

    # ---- summaries ------------------------------------------------------------------------------------------
    T = raw["tasks"]
    ph = holm([T[s]["paired"]["p"] for s, *_ in TASKS])
    for (s, *_), p in zip(TASKS, ph):
        T[s]["paired"]["p_holm"] = p
    # RQ1's classes for the seed mean against the paper's model (Holm over the 8 configurations, per family)
    for key in ("p_diff", "p_tost"):
        for (s, *_), p in zip(TASKS, holm([T[s]["reach"][key] for s, *_ in TASKS])):
            T[s]["reach"][key + "_holm"] = p
    for s, *_ in TASKS:
        r = T[s]["reach"]
        r["class"] = (("above" if r["delta"] > 0 else "below") if r["p_diff_holm"] < 0.05
                      else "parity" if r["p_tost_holm"] < 0.05 else "inconclusive")

    def summarize(key) -> dict:
        vals = {s: key(T[s]["gap"]) for s, *_ in TASKS}
        papers: dict[str, list[float]] = {}
        for s, _, pid, *_ in TASKS:
            papers.setdefault(pid, []).append(vals[s])
        per_paper = {p: statistics.mean(v) for p, v in papers.items()}
        no_dress = [v for (s, *_), v in zip(TASKS, vals.values()) if not s.startswith("dress")]
        return {"per_set": vals, "median_sets": statistics.median(vals.values()), "min": min(vals.values()), "max": max(vals.values()),
                "per_paper": per_paper, "median_papers": statistics.median(per_paper.values()), "median_no_dress": statistics.median(no_dress),
                "n_sets": len(vals), "n_papers": len(per_paper), "n_no_dress": len(no_dress)}

    summ = {"opt_mean": summarize(lambda g: g["gap_mean"]),
            **{f"opt_seed{s}": summarize(lambda g, j=j: g["gap_seed"][j]) for j, s in enumerate(SEEDS)},
            "v2": summarize(lambda g: g["gap_v2"]),
            **{k: summarize(lambda g, k=k: g["gap_other_baselines"][k]) for k in BASELINES}}
    raw["gap_summary"] = summ
    pooled = np.mean([diffs_task[s] for s, *_ in TASKS], axis=0)
    papers_order = list(dict.fromkeys(pid for _, _, _, pid, *_ in TASKS))
    by_paper = np.mean([np.mean([diffs_task[s] for s, _, _, pid, *_ in TASKS if pid == p], axis=0) for p in papers_order], axis=0)
    obs_pooled = statistics.mean(T[s]["paired"]["diff"] for s, *_ in TASKS)
    obs_paper = statistics.mean(statistics.mean(T[s]["paired"]["diff"] for s, _, _, pid, *_ in TASKS if pid == p) for p in papers_order)
    raw["paired_pooled"] = {
        "sets": {"diff": obs_pooled, "ci": [float(x) for x in np.percentile(pooled, [2.5, 97.5])], "p": boot_p(pooled), "n": len(TASKS)},
        "papers": {"diff": obs_paper, "ci": [float(x) for x in np.percentile(by_paper, [2.5, 97.5])], "p": boot_p(by_paper), "n": len(papers_order)},
    }
    sig_pos = [s for s, *_ in TASKS if T[s]["paired"]["p_holm"] < 0.05 and T[s]["paired"]["diff"] > 0]
    sig_neg = [s for s, *_ in TASKS if T[s]["paired"]["p_holm"] < 0.05 and T[s]["paired"]["diff"] < 0]
    ci_pos = [s for s, *_ in TASKS if T[s]["paired"]["ci"][0] > 0]
    ci_neg = [s for s, *_ in TASKS if T[s]["paired"]["ci"][1] < 0]

    # ---- macros ---------------------------------------------------------------------------------------------
    mac: dict[str, str] = {}
    so, sv = summ["opt_mean"], summ["v2"]
    mac["ovRqThreeGapMedian"] = pct(so["median_sets"])
    mac["ovRqThreeGapMin"] = pct(so["min"])
    mac["ovRqThreeGapMax"] = pct(so["max"])
    mac["ovRqThreeGapMedianPaper"] = pct(so["median_papers"])
    mac["ovRqThreeGapMedianNoDress"] = pct(so["median_no_dress"])
    for j, word in enumerate(("Zero", "One", "Two")):
        mac[f"ovRqThreeGapMedianSeed{word}"] = pct(summ[f"opt_seed{j}"]["median_sets"])
    seed_meds = [summ[f"opt_seed{s}"]["median_sets"] for s in SEEDS]
    mac["ovRqThreeGapMedianSeedMin"], mac["ovRqThreeGapMedianSeedMax"] = pct(min(seed_meds)), pct(max(seed_meds))
    mac["ovRqThreeVtwoGapMedian"] = pct(sv["median_sets"])
    mac["ovRqThreeVtwoGapMedianPaper"] = pct(sv["median_papers"])
    mac["ovRqThreeVtwoGapMedianNoDress"] = pct(sv["median_no_dress"])
    st = summ["tailored"]  # Jev-adapted GEPA, one seed
    mac["ovRqThreeTailoredGapMedian"] = pct(st["median_sets"])
    mac["ovRqThreeTailoredGapMedianNoDress"] = pct(st["median_no_dress"])
    mac["ovRqThreeNPapers"] = str(so["n_papers"])
    mac["ovRqThreeBoot"] = f"{B:,}"
    pp, pa = raw["paired_pooled"]["sets"], raw["paired_pooled"]["papers"]
    mac["ovRqThreePooledDiff"] = pts(pp["diff"])
    mac["ovRqThreePooledLo"], mac["ovRqThreePooledHi"] = pts(pp["ci"][0]), pts(pp["ci"][1])
    mac["ovRqThreePooledP"] = fp(pp["p"])
    mac["ovRqThreePaperPooledDiff"] = pts(pa["diff"])
    mac["ovRqThreePaperPooledLo"], mac["ovRqThreePaperPooledHi"] = pts(pa["ci"][0]), pts(pa["ci"][1])
    mac["ovRqThreePaperPooledP"] = fp(pa["p"])
    mac["ovRqThreeNHolmPos"], mac["ovRqThreeNHolmNeg"] = str(len(sig_pos)), str(len(sig_neg))
    mac["ovRqThreeNCiPos"], mac["ovRqThreeNCiNeg"] = str(len(ci_pos)), str(len(ci_neg))
    mac["ovRqThreeNCiZero"] = str(len(TASKS) - len(ci_pos) - len(ci_neg))
    for s, _, tag, *_ in TASKS:
        P, G = T[s]["paired"], T[s]["gap"]
        mac[f"ovRqThree{tag}Diff"] = pts(P["diff"])
        mac[f"ovRqThree{tag}Lo"], mac[f"ovRqThree{tag}Hi"] = pts(P["ci"][0]), pts(P["ci"][1])
        mac[f"ovRqThree{tag}PHolm"] = fp(P["p_holm"])
        mac[f"ovRqThree{tag}Gap"] = pct(G["gap_mean"])
        mac[f"ovRqThree{tag}VtwoGap"] = pct(G["gap_v2"])
        if "item_level" in P:
            mac[f"ovRqThree{tag}ItemLo"], mac[f"ovRqThree{tag}ItemHi"] = pts(P["item_level"]["ci"][0]), pts(P["item_level"]["ci"][1])
        R = T[s]["reach"]
        mac[f"ovRqThree{tag}ReachDiff"] = pts(R["delta"])
        mac[f"ovRqThree{tag}ReachLo"], mac[f"ovRqThree{tag}ReachHi"] = pts(R["ci"][0]), pts(R["ci"][1])
        mac[f"ovRqThree{tag}ReachClass"] = R["class"]
    classes = [T[s]["reach"]["class"] for s, *_ in TASKS]
    for c in ("above", "below", "parity", "inconclusive"):
        mac[f"ovRqThreeReachN{c.capitalize()}"] = str(classes.count(c))
    mac["ovRqThreeTbdNDocs"] = str(T["contempo-tbdense"]["paired"]["n_clusters"])
    sm_ref = paper_json["smarter/implicit_hate"]
    mac["ovRqThreeSmRefTrainN"] = sm_ref["method"].rsplit("-", 1)[1]  # "SMARTER Llama_DPO-256": DPO on 256 examples
    mac["ovRqThreeSmFullPaper"] = f"{100 * sm_ref['paper']['Llama_Full']:.1f}"
    trains = {T[s]["budget"]["opt"]["train"] for s, *_ in TASKS}
    fits = {T[s]["budget"]["opt"]["fit_per_seed"] for s, *_ in TASKS}
    vals = [T[s]["budget"]["opt"]["val"] for s, *_ in TASKS]
    refl = {T[s]["budget"]["v2"]["reflection_pool"] for s, *_ in TASKS}
    v1fit = {T[s]["budget"][k]["fit"] for s, *_ in TASKS for k in ("v1 text-opt", "v1 program-opt")}
    v1refl = {T[s]["budget"][k]["reflection_pool"] for s, *_ in TASKS for k in ("v1 text-opt", "v1 program-opt")}
    tfit = {T[s]["budget"]["tailored"]["fit"] for s, *_ in TASKS}
    trefl = {T[s]["budget"]["tailored"]["reflection_pool"] for s, *_ in TASKS}
    zfit = {T[s]["budget"]["zero"]["fit"] for s, *_ in TASKS}
    stages = {tuple(T[s]["budget"]["v2"]["val_progressive_stages"]) for s, *_ in TASKS}
    worst = {r["proposer_worst_examples_max"] for s, *_ in TASKS for r in T[s]["budget"]["opt"]["runs"].values()}
    rounds = {r["rounds"] for s, *_ in TASKS for r in T[s]["budget"]["opt"]["runs"].values()}
    assert all(len(x) == 1 for x in (trains, fits, refl, v1fit, v1refl, tfit, trefl, zfit, stages)), "budget differs by task"
    n_train, n_fit, n_refl = trains.pop(), fits.pop(), refl.pop()
    mac["ovRqThreeTrainN"] = f"{n_train:,}"
    mac["ovRqThreeFitN"] = f"{n_fit:,}"
    mac["ovRqThreeReflN"] = f"{n_refl:,}"
    mac["ovRqThreeValMin"], mac["ovRqThreeValMax"] = f"{min(vals):,}", f"{max(vals):,}"
    tots = [T[s]["budget"]["opt"]["total_distinct"] for s, *_ in TASKS]
    assert tots == [T[s]["budget"]["v2"]["total_distinct"] for s, *_ in TASKS]
    mac["ovRqThreeLabelsMin"], mac["ovRqThreeLabelsMax"] = f"{min(tots):,}", f"{max(tots):,}"
    mac["ovRqThreeLabelsMedian"] = f"{statistics.median(tots):,.0f}"
    n_demo_runs = sum(r["uses_demos"] for s, *_ in TASKS for r in T[s]["budget"]["opt"]["runs"].values())
    n_v2_demo = sum(T[s]["budget"]["v2"]["uses_demos"] for s, *_ in TASKS)
    mac["ovRqThreeNOptDemoRuns"] = str(n_demo_runs)
    mac["ovRqThreeNOptRuns"] = str(len(TASKS) * len(SEEDS))
    mac["ovRqThreeNVtwoDemo"] = str(n_v2_demo)
    union = {T[s]["budget"]["opt"]["fit_union_3_seeds"] for s, *_ in TASKS}
    mac["ovRqThreeFitUnionMin"], mac["ovRqThreeFitUnionMax"] = f"{min(union):,}", f"{max(union):,}"
    n_worst = max(worst)
    rounds_txt = "–".join(str(x) for x in sorted({min(rounds), max(rounds)}))
    stage_txt = " → ".join([*(str(x) for x in stages.pop()), "full val"])
    opt_runs_all = [r for s, *_ in TASKS for r in T[s]["budget"]["opt"]["runs"].values()]
    n_r0demo = sum(r["round0_demo_variant"] for r in opt_runs_all)
    max_single = max(r["val_selection_pool"]["single"] for r in opt_runs_all)
    max_stacked = max(r["val_selection_pool"]["stacked"] for r in opt_runs_all)
    n_folds = T[TASKS[0][0]]["budget"]["opt"]["readout_folds"]
    n_disjoint = sum(T[s]["budget"]["v2"]["fit_reflection_disjoint"] for s, *_ in TASKS)
    v2c = [T[s]["budget"]["v2"]["val_scored_candidates_all_v2_runs"] for s, *_ in TASKS]
    v2_cands_min, v2_cands_max = min(v2c), max(v2c)

    # ---- markdown ---------------------------------------------------------------------------------------------
    L = ["## RQ3: gap closed, seed-aware comparison with GEPA + demos + calibration, label budget (rq3)", "",
         "Source: `jev-opt/experiments/offline_rq3.py`; every test-evaluated representative re-executed offline from its "
         "Jev cache (cache miss = error) and checked against the recorded score. Optimized = feature-question search, "
         "optimizer seeds 0/1/2 (`results/seeds.json`); GEPA+demos+cal = the pre-registered v2 choice (second optimization study), "
         "**one seed only** (its seed-1/2 runs were never test-evaluated); zero-shot = the paper's question; paper = the "
         "task-specific method (`results/paper_numbers.json`).", "",
         "### (a) Gap closed  G = (score − zero-shot) / (paper − zero-shot)", "",
         "| Set | Zero-shot | Paper | Opt seeds 0/1/2 | Opt mean | G seed 0/1/2 (%) | G mean (%) | GEPA+demos+cal | G (%) |",
         "|---|---:|---:|---|---:|---|---:|---:|---:|"]
    for s, short, _, _, G in rows_gap:
        L.append(f"| {short} | {100 * G['zero']:.1f} | {100 * G['paper']:.1f} | {' / '.join(f'{100 * x:.1f}' for x in G['opt_test'])} | "
                 f"{100 * G['opt_mean']:.1f} | {' / '.join(pct(x) for x in G['gap_seed'])} | {pct(G['gap_mean'])} | "
                 f"{100 * G['v2_test']:.1f} | {pct(G['gap_v2'])} |")
    L += ["", "| Summary of G (%) | Opt mean | Opt seed 0 | seed 1 | seed 2 | GEPA+demos+cal | GEPA text | GEPA prog. | Jev-adapted GEPA |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    cols = ["opt_mean", "opt_seed0", "opt_seed1", "opt_seed2", "v2", "v1 text-opt", "v1 program-opt", "tailored"]
    for lab, key in ((f"median over {so['n_sets']} sets", "median_sets"), (f"median over {so['n_papers']} papers (DREsS rubrics averaged first)", "median_papers"),
                     (f"median over {so['n_no_dress']} non-DREsS sets", "median_no_dress"), ("min", "min"), ("max", "max")):
        L.append(f"| {lab} | " + " | ".join(pct(summ[c][key]) for c in cols) + " |")
    L += ["", f"Per paper (opt mean / GEPA+demos+cal): " + "; ".join(f"{p} {pct(so['per_paper'][p])} / {pct(sv['per_paper'][p])}" for p in so["per_paper"]) + ".", "",
          "### (b) Optimized − GEPA+demos+cal on test (points ×100)", "",
          f"D = mean over the 3 optimizer seeds of the paired test difference (baseline: 1 seed). Hierarchical bootstrap, "
          f"B = {B:,}: test clusters resampled with replacement (shared by all systems) and seeds resampled with "
          "replacement; 95% percentile CI; two-sided bootstrap p, Holm over 8 tasks. With 3 seeds the seed level of the "
          "bootstrap takes only 10 distinct multisets, so the interval reflects seed spread coarsely. The baseline's own seed "
          "variance is not represented (one tested seed), so these intervals understate the uncertainty of a method-level "
          "comparison. Item-level CI: the same bootstrap with single test items as the resampling unit (sensitivity for the "
          f"clustered tasks; TB-Dense's test set has only {T['contempo-tbdense']['paired']['n_clusters']} documents).", "",
          "| Task | Cluster (n) | D | 95% CI | p | p (Holm) | Item-level CI | seed 0 Δ (CI) | seed 1 Δ (CI) | seed 2 Δ (CI) |",
          "|---|---|---:|---|---:|---:|---|---|---|---|"]
    for s, short, _, P in rows_paired:
        ps = " | ".join(f"{pts(P['per_seed'][k]['diff'])} ({pts(P['per_seed'][k]['ci'][0])}…{pts(P['per_seed'][k]['ci'][1])})" for k in SEEDS)
        il = f"{pts(P['item_level']['ci'][0])}…{pts(P['item_level']['ci'][1])}" if "item_level" in P else "(same)"
        L.append(f"| {short} | {P['cluster_unit']} ({P['n_clusters']:,}) | {pts(P['diff'])} | {pts(P['ci'][0])}…{pts(P['ci'][1])} | "
                 f"{fp(P['p'])} | {fp(P['p_holm'])} | {il} | {ps} |")
    L += [f"| **Pooled, 8 sets (equal weight)** | | **{pts(pp['diff'])}** | {pts(pp['ci'][0])}…{pts(pp['ci'][1])} | {fp(pp['p'])} | | | | | |",
          f"| **Pooled, 6 papers (DREsS averaged)** | | **{pts(pa['diff'])}** | {pts(pa['ci'][0])}…{pts(pa['ci'][1])} | {fp(pa['p'])} | | | | | |", "",
          f"Holm-significant (p < 0.05): {len(sig_pos)} task(s) favour the optimized questions ({', '.join(T[s]['short'] for s in sig_pos) or 'none'}), "
          f"{len(sig_neg)} favour GEPA+demos+cal ({', '.join(T[s]['short'] for s in sig_neg) or 'none'}). CI above 0 on {len(ci_pos)}, below 0 on {len(ci_neg)}.", "",
          "### (c) Label budget per task (labelled examples by role)", "",
          f"Both methods draw on the same labelled data: the task's {n_train}-example train file and the full validation split "
          f"({mac['ovRqThreeLabelsMin']}–{mac['ovRqThreeLabelsMax']} labelled examples per task); neither reads test labels except "
          "for the one reported score. They differ in role:", "",
          f"- **Optimized (per seed):** fit slice = {n_fit} train (stratified, seed-specific): readout fit, {n_folds}-fold CV that "
          f"scores every question variant, stacking by CV, and the per-class summary and worst out-of-fold fit examples shown to "
          f"the proposer (≤ {n_worst} over {rounds_txt} rounds); kNN demonstration pool = all {n_train} train "
          f"(round 0 scores the paper question with demonstrations in {n_r0demo} of {len(TASKS) * len(SEEDS)} runs, so the other "
          f"{n_train - n_fit} train examples are eligible as demonstrations during the search; {n_demo_runs} of {len(TASKS) * len(SEEDS)} "
          f"tested programs keep demonstrations at test time); validation = full val split, used only to pick among ≤ {max_single} "
          f"single programs and ≤ {max_stacked} stacks and for the 1-point P / P+head rule (seed 0: also the A-vs-P choice of "
          f"the third study's protocol). Union of the 3 seeds' fit slices: {mac['ovRqThreeFitUnionMin']}–{mac['ovRqThreeFitUnionMax']} of {n_train}.",
          f"- **GEPA+demos+cal:** fit slice = {n_fit} train (class biases + temperature, DREsS Score cutoffs; refit for every "
          f"candidate; ensemble biases); GEPA reflection pool = {n_refl} other train examples (disjoint from the fit slice in "
          f"{n_disjoint} of {len(TASKS)} tasks); kNN demonstration pool = all {n_train} train (8 class-balanced per input; kept by "
          f"{n_v2_demo} of {len(TASKS)} chosen programs); validation = full val split used inside the search (GEPA Pareto scoring, "
          f"progressive stages {stage_txt}), for ensemble membership (top-k by val) and for the configuration choice among A–E and "
          f"the C/D seed runs ({v2_cands_min:,}–{v2_cands_max:,} val-scored candidates per task over all v2 runs).",
          f"- **GEPA text / program (v1):** {v1fit.pop()}-example fit (temperature/cutoffs), reflection minibatches from {v1refl.pop()} "
          f"train, val inside the search. **Jev-adapted GEPA:** {tfit.pop()} fit (class biases + temperature), reflection/acceptance "
          f"minibatches from {trefl.pop()} train, val inside the search.",
          f"- **Zero-shot (paper's question):** {zfit.pop()} train labels for the temperature (confidence only; for DREsS also the Score cutoffs).", "",
          "| Task | Train file | Fit / seed | Fit ∪ 3 seeds | Demo pool | Opt demos (seeds) | Demo ids on test (opt s0/s1/s2) | GEPA+d+c fit / refl. / demo ids on test | Val | Total distinct labels | Paper train pool |",
          "|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|"]
    for s, short, bd in rows_budget:
        o, v = bd["opt"], bd["v2"]
        ids = " / ".join(str(o["runs"][k]["demo_train_ids_on_test"]) for k in SEEDS)
        nd = sum(o["runs"][k]["uses_demos"] for k in SEEDS)
        L.append(f"| {short} | {o['train']} | {o['fit_per_seed']} | {o['fit_union_3_seeds']} | {o['demo_pool']} | {nd}/3 | {ids} | "
                 f"{v['fit']} / {v['reflection_pool']} / {v['demo_train_ids_on_test']} | {o['val']:,} | {o['total_distinct']:,} | {bd['paper_train_pool']} |")
    L += ["", "Demo ids on test = distinct train examples whose labels were shown as demonstrations to the tested program over the whole "
          "test set (counted by instrumenting the demonstration pool during the offline re-execution). Paper train pool = the "
          "training data from which each paper's task-specific model learned, as recorded in the task's data manifest (INLI: premises).", ""]
    md = "\n".join(L)

    # ---- LaTeX tables -------------------------------------------------------------------------------------------
    TEX.mkdir(parents=True, exist_ok=True)
    g_lines = ["\\begin{tabular}{lrrrrrrr}", "\\toprule",
               "& & & \\multicolumn{3}{c}{Feature-question search} & \\multicolumn{2}{c}{GEPA+demo.+cal.} \\\\",
               "\\cmidrule(lr){4-6}\\cmidrule(l){7-8}",
               "Set & Zero-shot & Paper & Mean & Gap closed (\\%) & Seeds 0/1/2 (\\%) & Score & Gap closed (\\%) \\\\", "\\midrule"]
    for s, short, _, _, G in rows_gap:
        g_lines.append(f"{short} & {100 * G['zero']:.1f} & {100 * G['paper']:.1f} & {100 * G['opt_mean']:.1f} & {pct(G['gap_mean'])} & "
                       f"{'/'.join(pct(x) for x in G['gap_seed'])} & {100 * G['v2_test']:.1f} & {pct(G['gap_v2'])} \\\\")
    g_lines += ["\\midrule",
                f"Median, {so['n_sets']} sets & & & & {pct(so['median_sets'])} & {'/'.join(pct(x) for x in seed_meds)} & & {pct(sv['median_sets'])} \\\\",
                f"Median, {so['n_papers']} papers & & & & {pct(so['median_papers'])} & {'/'.join(pct(summ[f'opt_seed{k}']['median_papers']) for k in SEEDS)} & & {pct(sv['median_papers'])} \\\\",
                f"Median, without DREsS & & & & {pct(so['median_no_dress'])} & {'/'.join(pct(summ[f'opt_seed{k}']['median_no_dress']) for k in SEEDS)} & & {pct(sv['median_no_dress'])} \\\\",
                "\\bottomrule", "\\end{tabular}", ""]
    (TEX / "offline_rq3_gap.tex").write_text("\n".join(g_lines))
    p_lines = ["\\begin{tabular}{lccccccc}", "\\toprule",
               "Task & " + " & ".join(f"\\multicolumn{{1}}{{c}}{{{h}}}" for h in ("Cluster ($n$)", "$\\Delta$", "95\\% CI", "$p$", "$p_{\\text{Holm}}$", "Item-level CI", "Per seed $\\Delta$ (0/1/2)")) + " \\\\", "\\midrule"]
    for s, short, _, P in rows_paired:
        il = f"$[{pts(P['item_level']['ci'][0])}, {pts(P['item_level']['ci'][1])}]$" if "item_level" in P else "--"
        p_lines.append(f"{short} & {P['cluster_unit']} ({P['n_clusters']:,}) & ${pts(P['diff'])}$ & $[{pts(P['ci'][0])}, {pts(P['ci'][1])}]$ & "
                       f"{fp(P['p']).replace('<', '$<$')} & {fp(P['p_holm']).replace('<', '$<$')} & {il} & "
                       f"${'/'.join(pts(P['per_seed'][k]['diff']) for k in SEEDS)}$ \\\\")
    p_lines += ["\\midrule",
                f"Pooled, 8 sets & & ${pts(pp['diff'])}$ & $[{pts(pp['ci'][0])}, {pts(pp['ci'][1])}]$ & {fp(pp['p']).replace('<', '$<$')} & & & \\\\",
                f"Pooled, 6 papers & & ${pts(pa['diff'])}$ & $[{pts(pa['ci'][0])}, {pts(pa['ci'][1])}]$ & {fp(pa['p']).replace('<', '$<$')} & & & \\\\",
                "\\bottomrule", "\\end{tabular}", ""]
    (TEX / "offline_rq3_paired.tex").write_text("\n".join(p_lines))
    b_lines = ["\\begin{tabular}{lrrrrrrrr}", "\\toprule",
               "& & \\multicolumn{3}{c}{Feature-question search} & \\multicolumn{2}{c}{GEPA+demo.+cal.} & & \\\\",
               "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}",
               "Task & Train & Fit/seed & Demos (seeds) & Demo ids & Fit/refl. & Demo ids & Val & Paper train \\\\", "\\midrule"]
    for s, short, bd in rows_budget:
        o, v = bd["opt"], bd["v2"]
        b_lines.append(f"{short} & {o['train']} & {o['fit_per_seed']} & {sum(o['runs'][k]['uses_demos'] for k in SEEDS)}/3 & "
                       f"{'/'.join(str(o['runs'][k]['demo_train_ids_on_test']) for k in SEEDS)} & {v['fit']}/{v['reflection_pool']} & "
                       f"{v['demo_train_ids_on_test']} & {o['val']:,} & {bd['paper_train_pool']} \\\\")
    b_lines += ["\\bottomrule", "\\end{tabular}", ""]
    (TEX / "offline_rq3_budget.tex").write_text("\n".join(b_lines))

    mac = {k: "$-$" + v[1:] if v.startswith("-") else v for k, v in mac.items()}  # text-mode minus sign
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"macros": mac, "md": md, **raw}, indent=1, default=float) + "\n")
    print(md)
    print(json.dumps(mac, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
