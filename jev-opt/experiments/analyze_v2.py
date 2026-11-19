"""Validation-only analysis of the v2 dev study (the second optimization study) -> results/GEPA_v2_val.md.

Reads each run's `val_predictions.json` (per-example val predictions of every representative,
written at the end of `optimize`), recomputes the task's primary metric, and reports a 95%
percentile bootstrap CI over val examples (2,000 resamples, seed 0) plus the paired bootstrap
difference to configuration A. Costs: Jev = sum of `usage.cost` over the run's unique Jev
requests (its response ledger); OMP = host-reported reflection/structural cost (`usage.jsonl`).
No Jev or LLM call is made and no test file is opened.

Run from the repo root:  uv run --project jev-opt/python python jev-opt/experiments/analyze_v2.py
"""

from __future__ import annotations

import gzip
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from jev_opt import metrics

ROOT = Path.cwd()
RUNS = ROOT / "runs" / "jev-opt"
OUT = ROOT / "results" / "GEPA_v2_val.md"
B = 2000
TASKS = [
    ("contempo-tbdense", "ConTempo TB-Dense", "micro-F1 (VAGUE neg.)"),
    ("ezstance-subtaskA", "EZ-STANCE Subtask A", "macro-F1"),
    ("smarter-implicit-hate", "SMARTER Implicit Hate", "macro-F1"),
    ("edit-intent", "Edit intent (Re3-Sci)", "macro-F1"),
    ("inli", "INLI", "accuracy"),
    ("dress-new-content", "DREsS content", "QWK"),
    ("dress-new-organization", "DREsS organization", "QWK"),
    ("dress-new-language", "DREsS language", "QWK"),
]


def configs(stem: str) -> list[tuple[str, str, str]]:
    """(config label, run id, representative) in table order; missing runs are skipped later."""
    out = [
        ("A: seed + calibration", f"v2-{stem}-A", "top-quality"),
        ("B: + kNN demos", f"v2-{stem}-B", "top-quality"),
        ("C: text-opt", f"v2-{stem}-CD", "top-jev-text-opt"),
        ("D: program-opt", f"v2-{stem}-CD", "top-jev-program-opt"),
        ("E: ensemble-3 (C+D)", f"v2-{stem}-CD", "ensemble-3"),
        ("E: ensemble-5 (C+D)", f"v2-{stem}-CD", "ensemble-5"),
    ]
    for tag in ("text", "program"):
        for seed in (1, 2):
            run = f"v2-{stem}-{tag}-s{seed}"
            out.append((f"{'C' if tag == 'text' else 'D'} seed {seed}", run, "top-quality"))
            out.append((f"{'C' if tag == 'text' else 'D'} seed {seed} ensemble-3", run, "ensemble-3"))
    return out


def run_config(run_dir: Path) -> dict[str, Any]:
    return yaml.safe_load((run_dir / "config.yaml").read_text())["task"]


def jev_cost(run_dir: Path) -> float:
    total = 0.0
    for name in json.loads((run_dir / "run.json").read_text()).get("jev_responses", {}).get("shards", []):
        with gzip.open(run_dir / name, "rt", encoding="utf-8") as fh:
            for line in fh:
                total += float((json.loads(line)["response"].get("usage") or {}).get("cost") or 0.0)
    return total


def omp_cost(run_dir: Path) -> float:
    path = run_dir / "usage.jsonl"
    if not path.exists():
        return 0.0
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sum(float(r.get("cost_total") or 0.0) for r in rows if r.get("source") == "omp")


def hours(run_dir: Path) -> float | None:
    state = json.loads((run_dir / "run.json").read_text())
    try:
        return (datetime.fromisoformat(state["finished_at"]) - datetime.fromisoformat(state["started_at"])).total_seconds() / 3600
    except (KeyError, TypeError, ValueError):
        return None


def score_fn(cfg: dict[str, Any]):
    labels, metric, negative = cfg["labels"], cfg["primary_metric"], cfg.get("negative_label")
    return lambda gold, pred: metrics.primary(metric, gold, pred, labels, negative)


def confusion_metric(cfg: dict[str, Any]):
    """Metric from (B, L+1, L+1) confusion counts (row gold, column pred; last column = failed)."""
    labels, metric, negative = cfg["labels"], cfg["primary_metric"], cfg.get("negative_label")
    L = len(labels)

    def f(c: np.ndarray) -> np.ndarray:
        c = c.astype(float)
        n = c.sum(axis=(1, 2))
        tp = np.stack([c[:, i, i] for i in range(L)], axis=1)
        pred_tot = c[:, :, :L].sum(axis=1)
        gold_tot = c[:, :L, :].sum(axis=2)
        if metric == "accuracy":
            return tp.sum(axis=1) / n
        if metric == "macro_f1":
            denom = pred_tot + gold_tot
            return np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0).mean(axis=1)
        if metric == "micro_f1":
            keep = [i for i, lab in enumerate(labels) if lab != negative]
            t, pp, gg = tp[:, keep].sum(1), pred_tot[:, keep].sum(1), gold_tot[:, keep].sum(1)
            return np.where(pp + gg > 0, 2 * t / np.where(pp + gg > 0, pp + gg, 1), 0.0)
        if metric == "qwk":
            idx = np.arange(L)
            w = (idx[:, None] - idx[None, :]) ** 2 / (L - 1) ** 2
            obs = c[:, :L, :L].copy()
            # failed predictions count as the farthest level from gold (as metrics.primary)
            for g in range(L):
                far = 0 if g >= L / 2 else L - 1
                obs[:, g, far] += c[:, g, L]
            tot = obs.sum(axis=(1, 2))
            rows, cols = obs.sum(axis=2), obs.sum(axis=1)
            exp = rows[:, :, None] * cols[:, None, :] / tot[:, None, None]
            num = (w * obs).sum(axis=(1, 2))
            den = (w * exp).sum(axis=(1, 2))
            return 1 - num / np.where(den > 0, den, 1)
        raise ValueError(metric)

    return f


def bootstrap(gold: list[str], preds: list[list[str | None]], cfg: dict[str, Any], seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(point (P,), resampled (B, P)) metric values; resamples shared (paired) across prediction lists."""
    labels = cfg["labels"]
    L, n = len(labels), len(gold)
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)])  # (B, n)
    gi = np.array([labels.index(g) for g in gold])
    f = confusion_metric(cfg)
    score = score_fn(cfg)
    points, boots = [], []
    for p in preds:
        pi = np.array([labels.index(x) if x in labels else L for x in p])
        onehot = np.zeros((n, (L + 1) * (L + 1)))
        onehot[np.arange(n), gi * (L + 1) + pi] = 1
        full = f(onehot.sum(axis=0).reshape(1, L + 1, L + 1))[0]
        exact = score(gold, p)
        assert abs(full - exact) < 1e-9, (full, exact)  # confusion arithmetic == jev_opt.metrics
        points.append(exact)
        boots.append(f((counts @ onehot).reshape(B, L + 1, L + 1)))
    return np.array(points), np.stack(boots, axis=1)


def main() -> int:
    lines = [
        "# v2 dev study: validation results",
        "",
        "The test round is pre-registered in the second study's protocol. Generated by",
        "`jev-opt/experiments/analyze_v2.py` from each run's `val_predictions.json`. **Validation split only; no",
        "test data was read.** Values ×100; 95% percentile bootstrap CI over val examples (2,000 resamples,",
        "seed 0); Δ A = paired bootstrap difference to configuration A on the same resamples.",
        "GEPA candidates and ensemble members are selected on this same val split, so C/D/E values are",
        "optimistically biased (selection on val); A and B involve no val-based selection.",
        "",
        "Configurations: A = gauntlet seed prompt + Choice class-bias calibration + temperature (fit on 400",
        "train examples, no LLM); B = A + 8 nearest train examples (class-balanced, bge-small-en-v1.5) in the",
        "Jev state; C = GEPA text-opt from B (≤1,500 chars per text slot, reflection pool 100 train examples",
        "disjoint from the fit slice); D = GEPA program-opt, same regularization (demo operators enabled);",
        "E = ensemble of the top-3/5 C+D candidates by val (mean label distribution + per-label biases fit",
        "on train). Seeds 1/2 repeat the better of C/D with a different optimizer seed.",
        "",
    ]
    costs: list[str] = []
    summary: dict[str, Any] = {}
    for stem, title, metric_name in TASKS:
        rows = []
        for label, run_id, rep in configs(stem):
            run_dir = RUNS / run_id
            vp = run_dir / "val_predictions.json"
            if not vp.exists():
                continue
            data = json.loads(vp.read_text())
            if rep not in data["representatives"]:
                continue
            pareto = json.loads((run_dir / "pareto.json").read_text())["representatives"][rep]
            rows.append((label, run_id, rep, data, pareto))
        if not rows:
            continue
        cfg = run_config(RUNS / rows[0][1])
        gold_ids = list(rows[0][3]["gold"])
        gold = [rows[0][3]["gold"][i] for i in gold_ids]
        preds = []
        for _, _, rep, data, _ in rows:
            assert list(data["gold"]) == gold_ids or set(data["gold"]) == set(gold_ids), "val sets differ"
            preds.append([data["representatives"][rep][i] for i in gold_ids])
        point, boot = bootstrap(gold, preds, cfg)
        a_idx = next((j for j, r in enumerate(rows) if r[0].startswith("A:")), None)
        lines += [f"## {title} ({metric_name}, val n={len(gold)})", "", "| Config | Run / representative | Val (95% CI) | Δ A (95% CI) | Jev calls/ex |", "|---|---|---|---|---:|"]
        summary[stem] = {}
        for j, (label, run_id, rep, data, pareto) in enumerate(rows):
            lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
            delta = ""
            if a_idx is not None and j != a_idx:
                d = boot[:, j] - boot[:, a_idx]
                dlo, dhi = np.percentile(d, [2.5, 97.5])
                delta = f"{100 * (point[j] - point[a_idx]):+.1f} ({100 * dlo:+.1f}…{100 * dhi:+.1f})"
            calls = pareto["val"].get("mean_calls")
            member = f" = {pareto['candidate']}" if pareto.get("candidate") and pareto["candidate"] != rep else ""
            lines.append(f"| {label} | `{run_id}` {rep}{member} | **{100 * point[j]:.1f}** ({100 * lo:.1f}…{100 * hi:.1f}) | {delta} | {calls:.2f} |")
            summary[stem][label] = {"run": run_id, "representative": rep, "val": point[j], "ci": [lo, hi], "mean_calls": calls}
        lines.append("")
        for run_id in dict.fromkeys(r[1] for r in rows):
            rd = RUNS / run_id
            h = hours(rd)
            costs.append(f"| {title} | `{run_id}` | ${jev_cost(rd):.2f} | ${omp_cost(rd):.2f} | {h:.2f} |" if h is not None else f"| {title} | `{run_id}` | ${jev_cost(rd):.2f} | ${omp_cost(rd):.2f} | |")
    lines += ["## Cost per run", "", "Jev: sum of the run's unique Jev requests (fresh or cache hits; a request shared by several runs is counted in each). OMP: reflection + structural LLM spend reported by the host.", "", "| Task | Run | Jev $ | OMP $ | wall h |", "|---|---|---:|---:|---:|", *costs, ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    (RUNS / "v2-dev-summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
