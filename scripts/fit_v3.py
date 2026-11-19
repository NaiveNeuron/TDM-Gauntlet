"""Fit and freeze the v3 configurations from the v3 dev runs (train/dev/validation data only).

Reads runs/dev_v3/<task>/predictions.jsonl(.gz); never reads any test output. For each lever it
(1) estimates the dev effect on held-out dev items (fitting on one part, scoring on another), (2) applies the
pre-stated selection rule (a configuration is registered for one test evaluation iff the paired-bootstrap
95% CI of its dev effect against the frozen zero-shot decision lies above 0; within a lever the cheapest
configuration within noise of the best is taken), and (3) freezes the selected configuration, refit on the
whole dev sample:

  heads   contempo/{matres,tbdense}, smarter/{hatexplain,latent_hate,implicit_hate}:
          candidates zero-shot argmax / per-class log-biases / L2 multinomial LR (C grid), selected
          by 5-fold CV on the fit part (grouped by document for ConTempo), scored once on the dev
          part -> data/frozen/heads/<task>.json (refit on fit + dev)
  ttp     (k, w) of the round-2 rerank, chosen on the 1,000-text dev sample (2-fold cross-fit
          estimate reported) -> data/frozen/ttp/v3_params.json
  wikiwd  mode (rerank k / policy, or round-1 biases), per-property log-biases; 2-fold x 5
          cross-fit by sentence -> data/frozen/relemb/wikiwd_v3_params.json
  made    tau_route and per-code thresholds; 2-fold x 3 cross-fit by report
          -> data/frozen/made/v3_params.json

Every lever's summary goes to results/v3_dev/<lever>.json; `report` renders results/v3_dev.md.

    uv run python scripts/fit_v3.py heads|ttp|wikiwd|made|report [--no-freeze]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gauntlet.fitted_head import Head, fit_bias, fit_lr, head_path  # noqa: E402
from gauntlet.task import read_jsonl  # noqa: E402

RUNS = REPO / "runs" / "dev_v3"
OUT = REPO / "results" / "v3_dev"
B = 2000  # bootstrap resamples
SEED = 0
FLOOR = 0.005  # half of Jev's two-decimal probability resolution


def predictions(task: str) -> list[dict[str, Any]]:
    d = RUNS / task
    path = d / "predictions.jsonl.gz" if (d / "predictions.jsonl.gz").exists() else d / "predictions.jsonl"
    return read_jsonl(path)


def manifest(task: str) -> dict[str, Any]:
    return json.loads((RUNS / task / "manifest.json").read_text())


def save(lever: str, summary: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{lever}.json").write_text(json.dumps(summary, indent=1) + "\n")


def boot_weights(n: int, seed: int = SEED, b: int = B) -> np.ndarray:
    """Multinomial resampling weights (b x n): row r = how often each item is drawn."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=b).astype(float)


def basic_interval(point: float, boot: np.ndarray) -> tuple[float, float]:
    """Basic (reverse-percentile) bootstrap 95% interval: 2*point - q97.5, 2*point - q2.5. Unlike the
    percentile interval it corrects the resampling bias of macro-F1 over rare labels (a resample
    that drops a rare code's only positive scores it 0), which otherwise puts MADE's point
    estimate outside its own percentile interval; for the other metrics the two agree closely."""
    return 2 * point - float(np.quantile(boot, 0.975)), 2 * point - float(np.quantile(boot, 0.025))


def paired_ci(counts_new: np.ndarray, counts_base: np.ndarray, metric: Callable[[np.ndarray], np.ndarray]) -> dict[str, float]:
    """Paired bootstrap over items of metric(summed counts): counts_* are (n x m) per-item counts;
    `metric` maps (r x m) summed counts to r values."""
    W = boot_weights(len(counts_new))
    d = metric(W @ counts_new) - metric(W @ counts_base)
    point = float(metric(counts_new.sum(0, keepdims=True))[0] - metric(counts_base.sum(0, keepdims=True))[0])
    lo, hi = basic_interval(point, d)
    return {"delta": point, "lo": lo, "hi": hi, "pct_lo": float(np.quantile(d, 0.025)), "pct_hi": float(np.quantile(d, 0.975))}


def value_ci(counts: np.ndarray, metric: Callable[[np.ndarray], np.ndarray]) -> dict[str, float]:
    W = boot_weights(len(counts), seed=SEED + 1)
    point = float(metric(counts.sum(0, keepdims=True))[0])
    boot = metric(W @ counts)
    lo, hi = basic_interval(point, boot)
    return {"value": point, "lo": lo, "hi": hi, "pct_lo": float(np.quantile(boot, 0.025)), "pct_hi": float(np.quantile(boot, 0.975))}


# --------------------------------------------------------------------------------------------
# Single-label metrics as functions of per-item counts


def micro_counts(y: np.ndarray, pred: np.ndarray, neg: int) -> np.ndarray:
    """Per item (tp, predicted positive, gold positive) with `neg` the negative label."""
    return np.stack([(pred == y) & (y != neg), pred != neg, y != neg], axis=1).astype(float)


def micro_metric(S: np.ndarray) -> np.ndarray:
    tp, npred, ngold = S[:, 0], S[:, 1], S[:, 2]
    return np.where(npred + ngold > 0, 2 * tp / np.maximum(npred + ngold, 1e-12), 0.0)


def macro_counts(y: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    """Per item one-hot (tp_1..k, pred_1..k, gold_1..k)."""
    eye = np.eye(k)
    return np.concatenate([eye[y] * (pred == y)[:, None], eye[pred], eye[y]], axis=1)


def macro_metric(k: int) -> Callable[[np.ndarray], np.ndarray]:
    def f(S: np.ndarray) -> np.ndarray:
        tp, npred, ngold = S[:, :k], S[:, k : 2 * k], S[:, 2 * k :]
        return np.where(npred + ngold > 0, 2 * tp / np.maximum(npred + ngold, 1e-12), 0.0).mean(axis=1)

    return f


# --------------------------------------------------------------------------------------------
# Heads


HEAD_TASKS = {
    # task: (dev run, metric kind, negative label for micro-F1, CV group key)
    "contempo/matres": ("contempo_matres", "micro", "VAGUE", "doc"),
    "contempo/tbdense": ("contempo_tbdense", "micro", "VAGUE", "doc"),
    "smarter/hatexplain": ("smarter_hatexplain", "macro", None, None),
    "smarter/latent_hate": ("smarter_latent_hate", "macro", None, None),
    "smarter/implicit_hate": ("smarter_implicit_hate", "macro", None, None),
}
LR_CS = (0.01, 0.1, 1.0, 10.0)


def head_data(task: str) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    run, kind, _, _ = HEAD_TASKS[task]
    rows = predictions(run)
    labels = list(rows[0]["prediction"]["probabilities"])
    P = np.array([[r["prediction"]["probabilities"][lab] for lab in labels] for r in rows])
    y = np.array([labels.index(r["gold"]) for r in rows])
    if task.startswith("contempo"):
        part = np.array([r["item_id"].split(":")[0] for r in rows])
        group = np.array([r["item_id"].split(":")[1] for r in rows])
    else:
        part = np.array([r["item_id"].split("-")[1] for r in rows])
        group = np.arange(len(rows)).astype(str)
    assert set(part) == {"fit", "dev"}
    zero = np.array([labels.index(r["prediction"]["label"]) for r in rows])
    assert (zero == np.argmax(P, axis=1)).all(), "zero-shot label must be the first-max argmax"
    return labels, P, y, part, group


def head_candidates(labels: list[str], metric: Callable[[np.ndarray, np.ndarray], float], balanced: bool) -> dict[str, Callable]:
    cands: dict[str, Callable] = {"zero_shot": lambda P, y: Head("bias", tuple(labels), FLOOR, bias=np.zeros(len(labels)))}
    cands["bias"] = lambda P, y: fit_bias(P, y, labels, metric, FLOOR)
    for C in LR_CS:
        cands[f"lr_C{C:g}"] = lambda P, y, C=C: fit_lr(P, y, labels, C, balanced, FLOOR)
    return cands


def group_folds(group: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    uniq = np.array(sorted(set(group)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % n_folds for i, g in enumerate(uniq)}
    f = np.array([fold_of[g] for g in group])
    return [f == i for i in range(n_folds)]


def fit_heads(freeze: bool) -> None:
    summary: dict[str, Any] = {}
    for task, (run, kind, neg, _) in HEAD_TASKS.items():
        labels, P, y, part, group = head_data(task)
        k = len(labels)
        if kind == "micro":
            ni = labels.index(neg)
            counts = lambda y_, p_: micro_counts(y_, p_, ni)  # noqa: E731
            agg = micro_metric
        else:
            counts = lambda y_, p_: macro_counts(y_, p_, k)  # noqa: E731
            agg = macro_metric(k)
        metric = lambda y_, p_: float(agg(counts(y_, p_).sum(0, keepdims=True))[0])  # noqa: E731
        cands = head_candidates(labels, metric, balanced=(kind == "macro"))
        fit, dev = part == "fit", part == "dev"
        # 5-fold CV on the fit part (document-grouped for ConTempo; pooled out-of-fold predictions)
        cv: dict[str, float] = {}
        folds = group_folds(group[fit], 5, SEED)
        Pf, yf = P[fit], y[fit]
        for name, make in cands.items():
            oof = np.zeros(len(yf), dtype=int)
            for te in folds:
                oof[te] = make(Pf[~te], yf[~te]).predict(Pf[te])
            cv[name] = metric(yf, oof)
        chosen = max(cv, key=lambda n: (cv[n], n == "zero_shot", n == "bias"))
        head = cands[chosen](Pf, yf)
        pred_dev = head.predict(P[dev])
        zero_dev = np.argmax(P[dev], axis=1)
        dev_new, dev_zero = counts(y[dev], pred_dev), counts(y[dev], zero_dev)
        ci = paired_ci(dev_new, dev_zero, agg)
        # Also every candidate's dev score (fit on the fit part), for the record
        cand_dev = {name: metric(y[dev], make(Pf, yf).predict(P[dev])) for name, make in cands.items()}
        selected = chosen != "zero_shot" and ci["lo"] > 0
        final = cands[chosen](P, y) if selected else None
        summary[task] = {
            "metric": "micro_f1 (VAGUE negative)" if kind == "micro" else "macro_f1",
            "n_fit": int(fit.sum()), "n_dev": int(dev.sum()), "labels": labels,
            "cv_fit": cv, "chosen": chosen, "dev_candidates": cand_dev,
            "dev_zero_shot": value_ci(dev_zero, agg), "dev_head": value_ci(dev_new, agg), "dev_delta": ci,
            "selected_for_test": selected,
            "dev_pred_counts": {"zero_shot": dict(Counter(labels[i] for i in zero_dev)), "head": dict(Counter(labels[i] for i in pred_dev))},
            "dev_run": {"task": f"dev_v3/{run}", **{k_: manifest(run)[k_] for k_ in ("n_items", "n_requests", "input_tokens", "cost_usd")}},
        }
        if selected and freeze:
            path = head_path(task)
            path.parent.mkdir(parents=True, exist_ok=True)
            spec = {
                "_task": task, "_condition": "Jev + fitted head (not zero-shot)",
                "_fit": f"{chosen} refit on the whole v3 dev sample (fit + dev parts, n={len(y)}) of runs/dev_v3/{run}; "
                "selected by 5-fold CV on the fit part, dev-part delta CI above 0 (results/v3_dev/heads.json)",
                "_features": f"log(max(p, {FLOOR})) of the frozen zero-shot Choice probabilities, label order as below",
                "head": final.to_json(),
            }
            path.write_text(json.dumps(spec, indent=1) + "\n")
            summary[task]["frozen"] = str(path.relative_to(REPO))
        print(task, chosen, {k_: round(v, 4) for k_, v in cv.items()}, "dev", round(ci["delta"], 4), (round(ci["lo"], 4), round(ci["hi"], 4)))
    save("heads", summary)


# --------------------------------------------------------------------------------------------
# TTP


TTP_WS = tuple(round(x, 2) for x in np.arange(0.0, 1.0001, 0.05))


def ttp_rank(row: dict[str, Any], k: int | None, w: float, order: dict[str, int]) -> list[str]:
    from gauntlet.tasks.ttp_v3 import rerank_order

    pred = row["prediction"]
    ranked, p1 = pred["ranking"], dict(zip(pred["ranking"], pred["p"]))
    if k is None:
        return ranked
    return rerank_order(ranked[:k], p1, pred["rerank"][str(k)], w, order) + ranked[k:]


def fit_ttp(freeze: bool) -> None:
    from gauntlet.tasks import ttp as ttp_task
    from gauntlet.tasks.ttp_v3 import DEV_KS, PARAMS_PATH

    rows = predictions("ttp_v3")
    order = {c["id"]: i for i, c in enumerate(read_jsonl(ttp_task.DATA / "candidates.jsonl"))}
    gold = [set(r["gold"]) for r in rows]
    n = len(rows)
    hit1 = lambda ranks: np.array([[float(rk[0] in g)] for rk, g in zip(ranks, gold)])  # noqa: E731
    mean = lambda S: S[:, 0] / n  # noqa: E731  (summed hits / n; bootstrap weights sum to n)
    v1 = hit1([ttp_rank(r, None, 0.0, order) for r in rows])
    grid = {(k, w): hit1([ttp_rank(r, k, w, order) for r in rows]) for k in DEV_KS for w in TTP_WS}
    p1 = {kw: float(v.mean()) for kw, v in grid.items()}
    best_k = {k: max(TTP_WS, key=lambda w: (p1[(k, w)], -w)) for k in DEV_KS}
    best = max(((k, best_k[k]) for k in DEV_KS), key=lambda kw: (p1[kw], -kw[0]))
    # Smallest k whose best-w P@1 is within noise of the overall best (paired 95% CI of the gap covers 0)
    chosen = best
    for k in sorted(DEV_KS):
        gap = paired_ci(grid[best], grid[(k, best_k[k])], mean)
        if gap["lo"] <= 0:
            chosen = (k, best_k[k])
            break
    # 2-fold x 5 cross-fit of the (k, w) choice (k fixed to the chosen k; w chosen on the other half)
    k_c = chosen[0]
    cross = []
    for rep in range(5):
        rng = np.random.default_rng(SEED + rep)
        half = rng.permutation(n) < n // 2
        pred = np.zeros(n)
        for te in (half, ~half):
            w_fit = max(TTP_WS, key=lambda w: (grid[(k_c, w)][~te].mean(), -w))
            pred[te] = grid[(k_c, w_fit)][te, 0]
        cross.append(float(pred.mean()))
    subsets = [r_["subset"] for r_ in read_jsonl(ttp_task.DATA / "dev_v3.jsonl")]
    ids = [r_["id"] for r_ in read_jsonl(ttp_task.DATA / "dev_v3.jsonl")]
    assert ids == [r["item_id"] for r in rows]
    by_subset = {s: {"v1": float(v1[[x == s for x in subsets]].mean()),
                     "tie_fix_only": float(grid[(k_c, 0.0)][[x == s for x in subsets]].mean()),
                     "v3": float(grid[chosen][[x == s for x in subsets]].mean())} for s in sorted(set(subsets))}
    full = {}
    for name, ranks in (("v1", [ttp_rank(r, None, 0.0, order) for r in rows]), ("v3", [ttp_rank(r, *chosen, order) for r in rows])):
        full[name] = ttp_task.ranking_metrics(ranks, gold)
    ci = paired_ci(grid[chosen], v1, mean)
    selected = ci["lo"] > 0
    summary = {
        "metric": "P@1 (pooled over the 1,000 dev texts: 334 TRAM / 333 Procedures / 333 Derived-procedures dev)",
        "n": n, "grid_p_at_1": {f"k{k}_w{w}": p1[(k, w)] for k, w in grid}, "best_w_per_k": {str(k): best_k[k] for k in DEV_KS},
        "best": {"k": best[0], "w": best[1], "p_at_1": p1[best]}, "chosen": {"k": chosen[0], "w": chosen[1], "p_at_1": p1[chosen]},
        "dev_v1": value_ci(v1, mean), "dev_tie_fix_only": value_ci(grid[(k_c, 0.0)], mean), "dev_v3": value_ci(grid[chosen], mean),
        "dev_delta": ci, "cross_fit_p_at_1": cross, "by_subset": by_subset, "full_metrics": full,
        "selected_for_test": selected,
        "dev_run": {"task": "dev_v3/ttp_v3", **{k_: manifest("ttp_v3")[k_] for k_ in ("n_items", "n_requests", "input_tokens", "cost_usd")}},
    }
    if selected and freeze:
        PARAMS_PATH.write_text(json.dumps({"k": chosen[0], "w": chosen[1], "_fit": "scripts/fit_v3.py ttp on runs/dev_v3/ttp_v3 "
                                           "(results/v3_dev/ttp.json): smallest k within noise of the best P@1, w = argmax P@1 for that k"}, indent=1) + "\n")
        summary["frozen"] = str(PARAMS_PATH.relative_to(REPO))
    save("ttp", summary)
    print(json.dumps({k_: summary[k_] for k_ in ("best", "chosen", "dev_v1", "dev_v3", "dev_delta", "cross_fit_p_at_1", "by_subset")}, indent=1))


# --------------------------------------------------------------------------------------------
# Wiki-WD


class WikiEdges:
    """Per-edge option sets (log-probabilities) for one decision variant, with fast bias refits."""

    def __init__(self, gold: list[str], sent: np.ndarray, options: list[dict[str, float]], neg: str, order: dict[str, int]):
        self.gold, self.sent, self.neg, self.order = gold, sent, neg, order
        self.opts = [sorted(o, key=order.__getitem__) for o in options]  # P0 first, then PID order
        self.logp = [np.log(np.maximum([o[lab] for lab in labs], FLOOR)) if labs else np.zeros(0) for o, labs in zip(options, self.opts)]
        self.by_label: dict[str, list[int]] = defaultdict(list)
        for i, labs in enumerate(self.opts):
            for lab in labs:
                self.by_label[lab].append(i)

    def decide(self, i: int, bias: dict[str, float]) -> str:
        labs = self.opts[i]
        if not labs:
            return self.neg
        s = self.logp[i] + np.array([bias.get(lab, 0.0) for lab in labs])
        return labs[int(np.argmax(s))]

    def counts(self, pred: list[str], idx: Sequence[int] | None = None) -> np.ndarray:
        idx = range(len(pred)) if idx is None else idx
        return np.array([[float(pred[i] == self.gold[i] and self.gold[i] != self.neg), float(pred[i] != self.neg),
                          float(self.gold[i] != self.neg)] for i in idx])

    def fit_bias(self, idx: np.ndarray, min_support: int = 5, grid: Sequence[float] = tuple(np.arange(-3.0, 3.01, 0.5)), rounds: int = 3) -> dict[str, float]:
        """Coordinate ascent on micro-F1 over the edges `idx` (labels with >= min_support gold or
        option occurrences there; strict gains only, ties keep the value closest to the current)."""
        inside = np.zeros(len(self.gold), bool)
        inside[idx] = True
        bias: dict[str, float] = {}
        pred = [self.decide(i, bias) if inside[i] else self.neg for i in range(len(self.gold))]
        tp = sum(inside[i] and pred[i] == self.gold[i] != self.neg for i in range(len(pred)))
        npred = sum(inside[i] and pred[i] != self.neg for i in range(len(pred)))
        ngold = sum(inside[i] and self.gold[i] != self.neg for i in range(len(pred)))
        f1 = lambda a, b_: 2 * a / (b_ + ngold) if b_ + ngold else 0.0  # noqa: E731
        gold_support = Counter(self.gold[i] for i in idx)
        labels = [lab for lab, es in self.by_label.items() if max(gold_support[lab], sum(inside[es])) >= min_support]
        labels.sort(key=self.order.__getitem__)
        for _ in range(rounds):
            moved = False
            for lab in labels:
                es = [i for i in self.by_label[lab] if inside[i]]
                cur = bias.get(lab, 0.0)
                best = f1(tp, npred)
                for g in sorted(grid, key=lambda g: abs(g - cur)):
                    trial = {**bias, lab: g}
                    new = {i: self.decide(i, trial) for i in es}
                    dtp = sum((new[i] == self.gold[i] != self.neg) - (pred[i] == self.gold[i] != self.neg) for i in es)
                    dnp = sum((new[i] != self.neg) - (pred[i] != self.neg) for i in es)
                    if f1(tp + dtp, npred + dnp) > best + 1e-12:
                        best, cur, moved = f1(tp + dtp, npred + dnp), g, True
                        best_new, best_d = new, (dtp, dnp)
                if cur != bias.get(lab, 0.0):
                    bias[lab] = cur
                    for i, v in best_new.items():
                        pred[i] = v
                    tp, npred = tp + best_d[0], npred + best_d[1]
            if not moved:
                break
        return {lab: float(b) for lab, b in bias.items() if b != 0.0}


def wikiwd_variants(rows: list[dict[str, Any]], neg: str, order: dict[str, int]) -> tuple[list[str], np.ndarray, list[str], dict[str, WikiEdges]]:
    from gauntlet.tasks.relemb_v3 import DEV_KS

    gold, sent, v2 = [], [], []
    per_edge_r1: list[tuple[dict, dict]] = []
    per_edge_r3: dict[int, list[dict | None]] = {k: [] for k in DEV_KS}
    real_winner: list[bool] = []
    for s, r in enumerate(rows):
        pred = r["prediction"]
        for e, j in enumerate(pred["edge_pair"]):
            gold.append(r["gold"][e])
            sent.append(s)
            v2.append(pred["labels"][e])
            ha, hb = pred["r1"][j]
            per_edge_r1.append((ha, hb))
            real_winner.append(any(w is not None for w in pred["round1"][j]))
            for k in DEV_KS:
                per_edge_r3[k].append(pred["r3"][str(k)].get(str(j)))
    sent_arr = np.array(sent)
    variants: dict[str, WikiEdges] = {}
    for k in DEV_KS:
        for policy in ("all", "any_real"):
            opts = [(o if (o is not None and (policy == "all" or rw)) else {}) for o, rw in zip(per_edge_r3[k], real_winner)]
            variants[f"rerank_k{k}_{policy}"] = WikiEdges(gold, sent_arr, opts, neg, order)
    return gold, sent_arr, v2, variants | {"round1": WikiRound1(gold, sent_arr, per_edge_r1, neg, order)}


class WikiRound1(WikiEdges):
    """Round-1-only decision: each half's biased winner; two real winners -> higher biased score."""

    def __init__(self, gold: list[str], sent: np.ndarray, halves: list[tuple[dict, dict]], neg: str, order: dict[str, int]):
        self.halves = [(WikiEdges([], sent, [ha], neg, order), WikiEdges([], sent, [hb], neg, order)) for ha, hb in halves]
        opts = [{**{neg: 0.0}, **ha, **hb} for ha, hb in halves]  # label sets only (for by_label)
        super().__init__(gold, sent, opts, neg, order)

    def decide(self, i: int, bias: dict[str, float]) -> str:
        best = []
        for h in self.halves[i]:
            labs = h.opts[0]
            if not labs:
                best.append((self.neg, -math.inf))
                continue
            s = h.logp[0] + np.array([bias.get(lab, 0.0) for lab in labs])
            j = int(np.argmax(s))
            best.append((labs[j], float(s[j])))
        (wa, sa), (wb, sb) = best
        if wa == self.neg or wb == self.neg:
            return wb if wa == self.neg else wa
        return wa if sa >= sb else wb


def fit_wikiwd(freeze: bool) -> None:
    from gauntlet.tasks.relemb_v3 import PARAMS_PATH

    rows = predictions("wikiwd_v3")
    prompt = json.loads((REPO / "prompts" / "relemb_wikiwd_v3.json").read_text())
    labels = [lab["label"] for lab in prompt["labels"]]
    neg = prompt["negative_label"]
    order = {lab: i for i, lab in enumerate(labels)}
    gold, sent, v2, variants = wikiwd_variants(rows, neg, order)
    n_sent = len(rows)
    per_sent = lambda C: np.array([C[sent == s].sum(0) for s in range(n_sent)])  # noqa: E731
    v2_counts = per_sent(variants["round1"].counts(v2))
    results: dict[str, Any] = {"v2": value_ci(v2_counts, micro_metric)}
    cross_counts: dict[str, np.ndarray] = {}
    for name, var in variants.items():
        zero = [var.decide(i, {}) for i in range(len(gold))]
        reps = []
        for rep in range(5):
            rng = np.random.default_rng(SEED + rep)
            half_s = rng.permutation(n_sent) < n_sent // 2
            half = half_s[sent]
            pred = [None] * len(gold)
            for te in (half, ~half):
                bias = var.fit_bias(np.where(~te)[0])
                for i in np.where(te)[0]:
                    pred[i] = var.decide(i, bias)
            reps.append(per_sent(var.counts(pred)))
        cross_counts[name] = np.mean(reps, axis=0)
        results[name] = {"no_bias": value_ci(per_sent(var.counts(zero)), micro_metric),
                         "cross_fit_bias": value_ci(cross_counts[name], micro_metric),
                         "delta_vs_v2": paired_ci(cross_counts[name], v2_counts, micro_metric),
                         "no_bias_delta_vs_v2": paired_ci(per_sent(var.counts(zero)), v2_counts, micro_metric)}
        print(name, round(results[name]["no_bias"]["value"], 4), round(results[name]["cross_fit_bias"]["value"], 4), results[name]["delta_vs_v2"])
    # Selection: best cross-fit F1; then the cheapest variant within noise of it (round1 has no new
    # test requests; then any_real before all; then smaller k).
    best = max(variants, key=lambda v: results[v]["cross_fit_bias"]["value"])
    cost_rank = lambda v: (0, 0, 0) if v == "round1" else (1, int(v.endswith("_all")), int(v.split("_k")[1].split("_")[0]))  # noqa: E731
    chosen = best
    for v in sorted(variants, key=cost_rank):
        if paired_ci(cross_counts[best], cross_counts[v], micro_metric)["lo"] <= 0:
            chosen = v
            break
    selected = results[chosen]["delta_vs_v2"]["lo"] > 0
    var = variants[chosen]
    bias = var.fit_bias(np.arange(len(gold)))
    summary = {"metric": "micro-F1, P0 excluded (dev: 3,000 validation sentences)", "n_sentences": n_sent, "n_edges": len(gold),
               "results": results, "best": best, "chosen": chosen, "selected_for_test": selected, "n_biases": len(bias),
               "biases": dict(sorted(bias.items(), key=lambda kv: -abs(kv[1]))),
               "dev_run": {"task": "dev_v3/wikiwd_v3", **{k_: manifest("wikiwd_v3")[k_] for k_ in ("n_items", "n_requests", "input_tokens", "cost_usd")}}}
    if selected and freeze:
        params = {"mode": "round1_bias"} if chosen == "round1" else {
            "mode": "rerank", "k": int(chosen.split("_k")[1].split("_")[0]), "policy": "all" if chosen.endswith("_all") else "any_real"}
        params |= {"floor": FLOOR, "bias": bias, "_fit": f"scripts/fit_v3.py wikiwd on runs/dev_v3/wikiwd_v3 (results/v3_dev/wikiwd.json); "
                   f"variant {chosen}, per-property log-biases refit on all {len(gold)} dev edges"}
        PARAMS_PATH.write_text(json.dumps(params, indent=1) + "\n")
        summary["frozen"] = str(PARAMS_PATH.relative_to(REPO))
    save("wikiwd", summary)
    print("best", best, "chosen", chosen, "selected", selected)


# --------------------------------------------------------------------------------------------
# MADE


MADE_TAUS = (0.5, 0.4, 0.3, 0.2, 0.15, 0.1)
MADE_GRID = tuple(round(x, 2) for x in np.arange(0.05, 0.951, 0.05))
MIN_POS = 10


class MadeFit:
    def __init__(self) -> None:
        from gauntlet.tasks.made_v3 import MADEv3, level

        self.task = MADEv3("fit", "dev_v3", 0.1, None)
        self.vocab = self.task.vocab
        self.col = {c: i for i, c in enumerate(self.vocab)}
        self.level = np.array([level(c) for c in self.vocab])
        self.anc = self.task._ancestors
        self.desc = self.task._descendants
        self.roots = [c for c in self.vocab if not self.task._parents[c]]
        rows = predictions("made_v3")
        self.gold = np.zeros((len(rows), len(self.vocab)), bool)
        for i, r in enumerate(rows):
            for c in r["gold"]:
                self.gold[i, self.col[c]] = True
        self.v2 = np.zeros_like(self.gold)
        for i, r in enumerate(rows):
            for c in r["prediction"]["labels"]:
                self.v2[i, self.col[c]] = True
        self.p_rows = [r["prediction"]["p"] for r in rows]
        self.n_asked_v2 = np.array([r["prediction"]["n_asked_v2"] for r in rows])

    def P_at(self, tau: float) -> tuple[np.ndarray, np.ndarray]:
        from gauntlet.tasks.made_v3 import asked_at

        P = np.zeros(self.gold.shape)
        n_asked = np.zeros(len(self.p_rows))
        for i, p in enumerate(self.p_rows):
            asked = asked_at(p, tau, self.roots, self.task.children)
            n_asked[i] = len(asked)
            for c in asked:
                P[i, self.col[c]] = p.get(c, 0.0)
        return P, n_asked

    def decode(self, P: np.ndarray, thr: np.ndarray) -> np.ndarray:
        """Predicted sets: accepted = P > thr (per code), closed over deepest accepted codes' own paths."""
        acc = P > thr[None, :]
        pred = acc.copy()
        for i in range(len(P)):
            codes = [self.vocab[j] for j in np.flatnonzero(acc[i])]
            s = set(codes)
            for c in codes:
                if not self.desc[c] & s:
                    for a in self.anc[c]:
                        pred[i, self.col[a]] = True
        return pred

    def fit_thresholds(self, P: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Bottom-up by level: code c's final prediction = (P_c > t_c) or pulled by a deepest
        accepted lower-level code; lower levels are fixed when level l is fit (exact)."""
        thr = np.full(len(self.vocab), 0.5)
        acc = np.zeros(P.shape, bool)
        for lev in sorted(set(self.level), reverse=True):
            cols = np.flatnonzero(self.level == lev)
            # pulled[i, c] for codes of this level, from the deepest accepted codes of lower levels
            pulled = np.zeros(P.shape, bool)
            for i in range(len(P)):
                s = {self.vocab[j] for j in np.flatnonzero(acc[i])}
                for c in s:
                    if not self.desc[c] & s:
                        for a in self.anc[c]:
                            pulled[i, self.col[a]] = True
            low = []
            for j in cols:
                npos = int(Y[:, j].sum())
                if npos < MIN_POS:
                    low.append(j)
                    continue
                thr[j] = self._best_t(P[:, j], Y[:, j], pulled[:, j])
            if low:  # shared per-level threshold for low-support codes: maximize their summed F1
                sums = {t: sum(self._f1((P[:, j] > t) | pulled[:, j], Y[:, j]) for j in low) for t in MADE_GRID}
                t_low = max(MADE_GRID, key=lambda t: (sums[t], -abs(t - 0.5)))
                thr[low] = t_low
            acc[:, cols] = P[:, cols] > thr[None, cols]
        return thr

    @staticmethod
    def _f1(pred: np.ndarray, y: np.ndarray) -> float:
        tp = float((pred & y).sum())
        d = float(pred.sum() + y.sum())
        return 2 * tp / d if tp > 0 else 0.0

    def _best_t(self, p: np.ndarray, y: np.ndarray, pulled: np.ndarray) -> float:
        return max(MADE_GRID, key=lambda t: (self._f1((p > t) | pulled, y), -abs(t - 0.5)))

    def counts(self, pred: np.ndarray) -> np.ndarray:
        """Per report (tp_c, pred_c, gold_c) over the vocabulary: 3 x 1,154 columns."""
        return np.concatenate([(pred & self.gold), pred, self.gold], axis=1).astype(float)

    def metric(self, S: np.ndarray) -> np.ndarray:
        k = len(self.vocab)
        tp, npred, ngold = S[:, :k], S[:, k : 2 * k], S[:, 2 * k :]
        return np.where(tp > 0, 2 * tp / np.maximum(npred + ngold, 1e-12), 0.0).mean(axis=1)


def fit_made(freeze: bool) -> None:
    from gauntlet.tasks.made_v3 import PARAMS_PATH

    m = MadeFit()
    n = len(m.gold)
    v2_counts = m.counts(m.v2)
    results: dict[str, Any] = {"v2": value_ci(v2_counts, m.metric), "v2_asked_mean": float(m.n_asked_v2.mean())}
    cross: dict[float, np.ndarray] = {}
    for tau in MADE_TAUS:
        P, n_asked = m.P_at(tau)
        reps = []
        for rep in range(3):
            half = np.random.default_rng(SEED + rep).permutation(n) < n // 2
            pred = np.zeros_like(m.gold)
            for te in (half, ~half):
                thr = m.fit_thresholds(P[~te], m.gold[~te])
                pred[te] = m.decode(P[te], thr)
            reps.append(m.counts(pred))
        cross[tau] = np.mean(reps, axis=0)
        fixed = m.counts(m.decode(P, np.full(len(m.vocab), 0.5)))
        results[f"tau_{tau}"] = {"asked_mean": float(n_asked.mean()), "accept_0.5": value_ci(fixed, m.metric),
                                 "cross_fit_thresholds": value_ci(cross[tau], m.metric),
                                 "delta_vs_v2": paired_ci(cross[tau], v2_counts, m.metric)}
        print(tau, results[f"tau_{tau}"]["asked_mean"], round(results[f"tau_{tau}"]["cross_fit_thresholds"]["value"], 4), results[f"tau_{tau}"]["delta_vs_v2"])
    best = max(MADE_TAUS, key=lambda t: results[f"tau_{t}"]["cross_fit_thresholds"]["value"])
    chosen = best
    for tau in sorted(MADE_TAUS, reverse=True):  # highest tau (fewest new questions) within noise of the best
        if paired_ci(cross[best], cross[tau], m.metric)["lo"] <= 0:
            chosen = tau
            break
    selected = results[f"tau_{chosen}"]["delta_vs_v2"]["lo"] > 0
    P, _ = m.P_at(chosen)
    thr = m.fit_thresholds(P, m.gold)
    npos = m.gold.sum(0)
    per_code = {c: float(thr[j]) for j, c in enumerate(m.vocab) if npos[j] >= MIN_POS}
    defaults = {str(lev): float(Counter(thr[(m.level == lev) & (npos < MIN_POS)]).most_common(1)[0][0]) if ((m.level == lev) & (npos < MIN_POS)).any() else 0.5
                for lev in sorted(set(m.level))}
    summary = {"metric": "macro-F1 over the 1,154 codes (dev: 3,000 validation reports; codes without dev positives count 0)",
               "n": n, "n_codes_with_dev_positives": int((npos > 0).sum()), "results": results, "best_tau": best, "chosen_tau": chosen,
               "selected_for_test": selected, "n_per_code_thresholds": len(per_code), "default_threshold": defaults,
               "dev_run": {"task": "dev_v3/made_v3", **{k_: manifest("made_v3")[k_] for k_ in ("n_items", "n_requests", "input_tokens", "cost_usd")}}}
    if selected and freeze:
        PARAMS_PATH.write_text(json.dumps({"tau_route": chosen, "thresholds": per_code, "default_threshold": defaults,
                                           "_fit": f"scripts/fit_v3.py made on runs/dev_v3/made_v3 (results/v3_dev/made.json): tau_route = highest "
                                           f"grid value within noise of the best cross-fit macro-F1; per-code thresholds (codes with >= {MIN_POS} "
                                           "dev positives) and per-level shared thresholds, fit bottom-up on all 3,000 dev reports"}, indent=1) + "\n")
        summary["frozen"] = str(PARAMS_PATH.relative_to(REPO))
    save("made", summary)
    print("best", best, "chosen", chosen, "selected", selected)


# --------------------------------------------------------------------------------------------
# Report


def responses(task: str):
    for shard in sorted((RUNS / task).glob("responses*.jsonl.gz")):
        yield from read_jsonl(shard)


def key_cost(task: str, prefix: Callable[[str], bool]) -> tuple[float, int, int]:
    """(USD, input tokens, requests) of the dev run's responses whose key matches."""
    usd = tok = n = 0
    for r in responses(task):
        if prefix(r["key"]):
            usd += r["response"]["usage"]["cost"]
            tok += r["response"]["usage"]["input_tokens"]
            n += 1
    return usd, tok, n


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def ci_str(d: dict[str, float], key: str = "value") -> str:
    return f"{pct(d[key])} [{pct(d['lo'])}, {pct(d['hi'])}]"


def delta_str(d: dict[str, float]) -> str:
    return f"{100 * d['delta']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]"


TEST_N = {"ttp_texts": 725 + 2288 + 697, "wikiwd_edges": 739_977, "made_reports": 10_288}


def render_report() -> None:
    S = {p.stem: json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))}
    L = ["# v3 development results (train/dev/validation data only)", "",
         "Generated by `uv run python scripts/fit_v3.py report` from `results/v3_dev/*.json`. No test data was read.",
         "Design and dev samples: scripts/fit_v3.py and scripts/prepare_v3_dev.py. Values ×100.",
         "\"Zero-shot\" = the frozen v1/v2 decision on the same dev items; the v3 column is a held-out estimate",
         "(heads: fit part → dev part; TTP / Wiki-WD / MADE: 2-fold cross-fitting, mean over repeats). Δ = paired",
         "bootstrap 95% CI over dev items (2,000 resamples). Registered = Δ CI above 0 (the pre-stated rule).", "",
         "| Test task (if registered) | Dev sample | Metric | Zero-shot | v3 / + head | Δ [95% CI] | Configuration | Registered |",
         "|---|---|---|---:|---:|---:|---|---|"]
    dev_cost: list[tuple[str, dict]] = []
    if "ttp" in S:
        t = S["ttp"]
        cf = float(np.mean(t["cross_fit_p_at_1"]))
        L.append(f"| `ttp_v3/{{tram,procedures_plus,expert,expert_full}}` | 1,000 TTP dev texts | P@1 | {ci_str(t['dev_v1'])} | "
                 f"{pct(cf)} (cross-fit; in-sample {pct(t['dev_v3']['value'])}) | {delta_str(t['dev_delta'])} | "
                 f"k={t['chosen']['k']}, w={t['chosen']['w']} | {'yes' if t['selected_for_test'] else 'no'} |")
        dev_cost.append(("dev_v3/ttp_v3", t["dev_run"]))
    if "wikiwd" in S:
        w = S["wikiwd"]
        ch = w["results"][w["chosen"]]
        L.append(f"| `relemb_v3/wikiwd` | 3,000 validation sentences ({w['n_edges']:,} edges) | micro-F1 | {ci_str(w['results']['v2'])} | "
                 f"{ci_str(ch['cross_fit_bias'])} | {delta_str(ch['delta_vs_v2'])} | {w['chosen']} + {w['n_biases']} biases | "
                 f"{'yes' if w['selected_for_test'] else 'no'} |")
        dev_cost.append(("dev_v3/wikiwd_v3", w["dev_run"]))
    if "made" in S:
        m = S["made"]
        ch = m["results"][f"tau_{m['chosen_tau']}"]
        d = ch["delta_vs_v2"]
        L.append(f"| `made_v3` | 3,000 validation reports | macro-F1 (1,154 codes) | {pct(m['results']['v2']['value'])} | "
                 f"{pct(ch['cross_fit_thresholds']['value'])} | {delta_str(d)}; percentile [{100 * d['pct_lo']:+.1f}, {100 * d['pct_hi']:+.1f}] | τ_route={m['chosen_tau']}, "
                 f"{m['n_per_code_thresholds']} per-code thresholds | {'yes' if m['selected_for_test'] else 'no'} |")
        dev_cost.append(("dev_v3/made_v3", m["dev_run"]))
    for task, h in S.get("heads", {}).items():
        L.append(f"| `{task}+head` | fit {h['n_fit']:,} / dev {h['n_dev']:,} | {h['metric']} | {ci_str(h['dev_zero_shot'])} | "
                 f"{ci_str(h['dev_head'])} | {delta_str(h['dev_delta'])} | {h['chosen']} | {'yes' if h['selected_for_test'] else 'no'} |")
        dev_cost.append((h["dev_run"]["task"], h["dev_run"]))
    if "ttp" in S:
        t = S["ttp"]
        L += ["", "## TTP rerank", "", "P@1 on the 1,000 dev texts by k (best w per k); w = 0 is the tie-break fix alone.", "",
              "| k | w=0 (tie fix) | best w | P@1 at best w |", "|---:|---:|---:|---:|"]
        for k, wb in t["best_w_per_k"].items():
            L.append(f"| {k} | {pct(t['grid_p_at_1'][f'k{k}_w0.0'])} | {wb} | {pct(t['grid_p_at_1'][f'k{k}_w{wb}'])} |")
        L += ["", f"v1 (round 1 only, ID tie-break): {pct(t['dev_v1']['value'])}. 2-fold cross-fit P@1 of the w choice at "
              f"k={t['chosen']['k']}: {', '.join(pct(x) for x in t['cross_fit_p_at_1'])}.", "",
              "| dev subset | v1 | tie fix only | v3 |", "|---|---:|---:|---:|"]
        for s, v in t["by_subset"].items():
            L.append(f"| {s} | {pct(v['v1'])} | {pct(v['tie_fix_only'])} | {pct(v['v3'])} |")
        fm = t["full_metrics"]
        L += ["", "| metric (dev) | v1 | v3 |", "|---|---:|---:|"] + [f"| {k} | {pct(fm['v1'][k])} | {pct(fm['v3'][k])} |" for k in ("p@1", "r@1", "f1@3", "mrr@3", "f1@5")]
    if "wikiwd" in S:
        w = S["wikiwd"]
        L += ["", "## Wiki-WD", "", f"v2 decision on the dev edges: {ci_str(w['results']['v2'])}.", "",
              "| variant | no biases | cross-fit biases | Δ vs v2 (cross-fit) |", "|---|---:|---:|---:|"]
        for name, r in w["results"].items():
            if name != "v2":
                L.append(f"| {name} | {pct(r['no_bias']['value'])} | {pct(r['cross_fit_bias']['value'])} | {delta_str(r['delta_vs_v2'])} |")
        top = list(w["biases"].items())[:12]
        L += ["", f"Best: {w['best']}; chosen (cheapest within noise): {w['chosen']}. Largest frozen biases: "
              + ", ".join(f"{lab} {b:+.1f}" for lab, b in top) + "."]
    if "made" in S:
        m = S["made"]
        L += ["", "## MADE", "", f"v2 decision (τ = 0.5 routing and acceptance) on the dev reports: {pct(m['results']['v2']['value'])}; "
              f"codes with dev positives: {m['n_codes_with_dev_positives']}. Report-level bootstrap intervals are unreliable for "
              "macro-F1 over 1,154 mostly rare codes (resamples drop rare codes' only positives), so values are shown without "
              "intervals and each Δ with both the basic and the percentile interval.", "",
              "| τ_route | codes asked / report | accept P > 0.5 | cross-fit per-code thresholds | Δ vs v2, basic | percentile |", "|---:|---:|---:|---:|---:|---:|"]
        for tau in MADE_TAUS:
            r = m["results"][f"tau_{tau}"]
            d = r["delta_vs_v2"]
            L.append(f"| {tau} | {r['asked_mean']:.0f} | {pct(r['accept_0.5']['value'])} | {pct(r['cross_fit_thresholds']['value'])} | "
                     f"{delta_str(d)} | [{100 * d['pct_lo']:+.1f}, {100 * d['pct_hi']:+.1f}] |")
        L += ["", f"Best τ: {m['best_tau']}; chosen (highest within noise): {m['chosen_tau']}. Per-level shared thresholds: {m['default_threshold']}."]
    if "heads" in S:
        L += ["", "## Heads", "", "5-fold CV on the fit part (selection), and each candidate fit on the fit part scored on the dev part (record only).", "",
              "| task | CV: zero-shot | CV: bias | CV: best LR | chosen | dev: chosen | dev predicted-label counts (zero-shot → head) |", "|---|---:|---:|---:|---|---:|---|"]
        for task, h in S["heads"].items():
            lr = max((k for k in h["cv_fit"] if k.startswith("lr")), key=h["cv_fit"].get)
            counts = "; ".join(f"{lab} {h['dev_pred_counts']['zero_shot'].get(lab, 0)}→{h['dev_pred_counts']['head'].get(lab, 0)}" for lab in h["labels"])
            L.append(f"| {task} | {pct(h['cv_fit']['zero_shot'])} | {pct(h['cv_fit']['bias'])} | {pct(h['cv_fit'][lr])} ({lr}) | {h['chosen']} | "
                     f"{pct(h['dev_candidates'][h['chosen']])} | {counts} |")
    L += ["", "## Jev cost of the dev runs", "", "| dev run | items | requests | input tokens | USD |", "|---|---:|---:|---:|---:|"]
    total = 0.0
    for name, d in dev_cost:
        L.append(f"| `{name}` | {d['n_items']:,} | {d['n_requests']:,} | {d['input_tokens']:,} | {d['cost_usd']:.2f} |")
        total += d["cost_usd"]
    L += [f"| **total** | | | | **{total:.2f}** |", ""]
    proj = []
    if S.get("ttp", {}).get("selected_for_test"):
        k = S["ttp"]["chosen"]["k"]
        usd, tok, n = key_cost("ttp_v3", lambda key: key == f"rerank-k{k}")
        proj.append(f"TTP round 2 (k={k}): {tok / n:,.0f} tokens per text on dev → ≈ ${usd / n * TEST_N['ttp_texts']:.2f} for the "
                    f"{TEST_N['ttp_texts']:,} distinct test texts (round 1 served from the v1 cache)")
    if S.get("wikiwd", {}).get("selected_for_test") and S["wikiwd"]["chosen"] != "round1":
        k = int(S["wikiwd"]["chosen"].split("_k")[1].split("_")[0])
        usd, tok, n = key_cost("wikiwd_v3", lambda key: key.startswith(f"r3k{k}."))
        per_edge = usd / S["wikiwd"]["n_edges"]
        proj.append(f"Wiki-WD round 3 (k={k}, all shortlisted pairs): ≈ ${per_edge * TEST_N['wikiwd_edges']:.2f} upper bound for "
                    f"{TEST_N['wikiwd_edges']:,} test edges (rounds 1-2 cached; policy any_real asks fewer)")
    if S.get("made", {}).get("selected_for_test"):
        m = S["made"]
        usd, tok, n = key_cost("made_v3", lambda key: key.startswith("x"))
        extra_dev = m["results"]["tau_0.1"]["asked_mean"] - m["results"]["v2_asked_mean"]
        extra_ch = m["results"][f"tau_{m['chosen_tau']}"]["asked_mean"] - m["results"]["v2_asked_mean"]
        per_report = usd / m["n"] * (extra_ch / extra_dev if extra_dev else 0.0)
        proj.append(f"MADE extension rounds (τ_route={m['chosen_tau']}): ≈ ${per_report * TEST_N['made_reports']:.2f} for "
                    f"{TEST_N['made_reports']:,} test reports (v2 rounds cached)")
    if proj:
        L += ["## Projected new Jev cost of the registered test runs", ""] + [f"- {p}" for p in proj] + [
            "- `+head` tasks: $0 (identical requests, served from the zero-shot cache)", ""]
    (REPO / "results" / "v3_dev.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("lever", choices=["heads", "ttp", "wikiwd", "made", "report"])
    ap.add_argument("--no-freeze", action="store_true")
    args = ap.parse_args()
    if args.lever == "report":
        render_report()
        return
    {"heads": fit_heads, "ttp": fit_ttp, "wikiwd": fit_wikiwd, "made": fit_made}[args.lever](not args.no_freeze)


if __name__ == "__main__":
    main()
