"""Validation-only analysis of the v3 dev study (the third optimization study) -> results/GEPA_v3_val.md.

Reads each run's `val_predictions.json`, recomputes the primary metric, and reports a 95% percentile
bootstrap CI over val examples (2,000 resamples, seed 0) plus paired bootstrap differences to v2's
configuration A (seed + class bias; reference, `--v2-runs`) and to v3's A (seed + readout head).
Applies the pre-registered selection rule (`select`). Costs: Jev used = sum of `usage.cost`
over the run's unique Jev requests (its response ledger; requests reused from earlier runs count);
Jev fresh = spend billed during this run (worker usage, per condition); tokens = fresh input tokens;
OMP = host-reported proposer spend (`usage.jsonl`). No Jev or LLM call is made and no test file is
opened.

Run from the repo root:
  uv run --project jev-opt/python python jev-opt/experiments/analyze_v3.py [--v2-runs <runs/jev-opt of the v2 study>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_v2 import TASKS, bootstrap, hours, jev_cost, omp_cost, run_config, score_fn  # noqa: E402

ROOT = Path.cwd()
RUNS = ROOT / "runs" / "jev-opt"
OUT = ROOT / "results" / "GEPA_v3_val.md"
MARGIN = 1.0  # points (x100): a more complex configuration must beat the chosen simpler one by this much on val
CONFIGS = [  # (label, run template, representative, is v2 reference, same-stage paper row for the second delta)
    ("v2-A: seed + class bias (reference)", "v2-{stem}-A", "top-quality", True, None),
    ("zero: paper seed", "v3-{stem}-P", "top-zero-paper", False, None),
    ("zero: jev-native seed", "v3-{stem}-N", "top-quality", False, "zero: paper seed"),
    ("zero: decomposed seed", "v3-{stem}-P", "top-zero-decomposed", False, "zero: paper seed"),
    ("A: seed + readout head", "v3-{stem}-A", "top-quality", False, None),
    ("head: paper seed (± demos)", "v3-{stem}-P", "top-head-paper", False, "A: seed + readout head"),
    ("head: jev-native seed", "v3-{stem}-P", "top-head-jev-native", False, "A: seed + readout head"),
    ("head: decomposed seed", "v3-{stem}-P", "top-head-decomposed", False, "A: seed + readout head"),
    ("search from paper", "v3-{stem}-P", "top-search-paper", False, None),
    ("search from jev-native", "v3-{stem}-P", "top-search-jev-native", False, "search from paper"),
    ("search from decomposed", "v3-{stem}-P", "top-search-decomposed", False, "search from paper"),
    ("P: population-opt", "v3-{stem}-P", "top-single", False, None),
    ("P+head: stacked readout", "v3-{stem}-P", "top-stacked", False, None),
]
FALLBACK = {"zero: jev-native seed": ("v3-{stem}-P", "top-zero-jev-native")}  # before the N run exists
CHOICES = ["A: seed + readout head", "P: population-opt", "P+head: stacked readout"]  # in order of complexity


def fresh(run_dir: Path) -> tuple[float, int, int]:
    """(fresh Jev $, fresh requests, fresh input tokens) billed by the run's conditions."""
    state = json.loads((run_dir / "run.json").read_text())
    cost = calls = tokens = 0
    for info in state.get("conditions", {}).values():
        s = info.get("summary", {}).get("jev_fresh_session", {})
        cost += float(s.get("cost") or 0.0)
        calls += int(s.get("calls") or 0)
        tokens += int(s.get("input_tokens") or 0)
    return cost, calls, tokens


def seed_table(run_dir: Path, gold_ids: list[str], gold: list[str], score) -> list[str]:
    """One row per seed of a population run: zero-shot train/val and train CV with a readout."""
    cand_dir = run_dir / "jev-population-opt" / "candidates"
    state_path = run_dir / "jev-population-opt" / "population.json"
    if not cand_dir.exists() or not state_path.exists():
        return []
    cv = {v["origin"]: v["cv"] for v in json.loads(state_path.read_text())["variants"] if v["round"] == 0}
    out = []
    for path in sorted(cand_dir.glob("*.json"), key=lambda p: int(p.stem)):
        meta = json.loads(path.read_text())
        if "zero" not in meta.get("tags", []):
            continue
        preds = run_zero_preds(run_dir, meta, gold_ids)
        val = score(gold, preds) if preds else None
        val_txt = "" if val is None else f"{100 * val:.1f}"
        seed_cv = cv.get(meta["seed"])
        cv_txt = "" if seed_cv is None else f"{100 * seed_cv:.1f}"
        out.append(f"| {meta['seed']} | {meta['seed_kind']} | {100 * meta['train_zero']:.1f} | {val_txt} | {cv_txt} |")
    return out


def run_zero_preds(run_dir: Path, meta: dict[str, Any], gold_ids: list[str]) -> list[str | None] | None:
    """Val predictions of a zero-shot seed candidate, from its representative `top-zero-<seed>`."""
    data = json.loads((run_dir / "val_predictions.json").read_text())["representatives"].get(f"top-zero-{meta['seed']}")
    return [data[i] for i in gold_ids] if data else None


def select(point: dict[str, float]) -> str | None:
    """Pre-registered rule: start from A; a more complex configuration replaces the current choice
    only if its val primary metric is at least MARGIN points higher."""
    chosen = None
    for label in CHOICES:
        if label not in point:
            continue
        if chosen is None or 100 * (point[label] - point[chosen]) >= MARGIN:
            chosen = label
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-runs", type=Path, default=RUNS)
    args = parser.parse_args()
    lines = [
        "# v3 dev study: validation results",
        "",
        "The test round is pre-registered in the third study's protocol. Generated by",
        "`jev-opt/experiments/analyze_v3.py` from each run's `val_predictions.json`. **Validation split only; no",
        "test data was read.** Values ×100; 95% percentile bootstrap CI over val examples (2,000 resamples,",
        "seed 0); Δ = paired bootstrap difference on the same resamples.",
        "",
        "Every v3 configuration is a **Jev + fitted head** system (readout parameters fit on the 400-example",
        "train fit slice), not zero-shot. A = the gauntlet seed prompt with a readout head (L2 logistic",
        "regression over the Choice log-probabilities; ridge + cutoffs over the Score level distribution for",
        "DREsS; hyperparameters by 5-fold CV, grouped by document for TB-Dense and by premise for INLI).",
        "Seeds: the paper seed plus 3 `jev-native` rewrites and 3 `decomposed` programs written by the omp model",
        "from the task definition only (`/jev-seeds`, written before scoring). zero rows: a seed exactly as",
        "written (no fitted number; the kind's seed with the best zero-shot train score, so no val selection).",
        "head rows: the kind's best seed by train CV + readout head. P = jev-population-opt from the union of",
        "all seeds: 4 rounds × 20 OMP-proposed question variants (≤1,500 chars per field, ≤2 aux questions,",
        "optional kNN demonstrations) asked in packed requests and ranked by train CV; search rows: the best",
        "program by train CV descending from one seed kind; val picks `top-single` among the 4 best programs by",
        "train CV; P+head = stacks of those programs' variants under one readout, forward-selected by train CV",
        "= `top-stacked`. P and P+head are optimistically biased by that val selection (≤ 4 and ≤ 3",
        "candidates); zero, A, head and search rows involve no val-based choice. v2-A (seed + class-bias",
        "calibration, no head) is the reference.",
        "",
    ]
    costs: list[str] = []
    summary: dict[str, Any] = {}
    chosen_rows: list[str] = []
    for stem, title, metric_name in TASKS:
        rows = []
        for label, template, rep, ref, _ in CONFIGS:
            for tpl, rp in [(template, rep), FALLBACK.get(label, (None, None))]:
                if tpl is None:
                    continue
                run_dir = (args.v2_runs if ref else RUNS) / tpl.format(stem=stem)
                vp = run_dir / "val_predictions.json"
                if vp.exists() and rp in (data := json.loads(vp.read_text()))["representatives"]:
                    pareto = json.loads((run_dir / "pareto.json").read_text())["representatives"][rp]
                    rows.append((label, run_dir, rp, data, pareto, ref))
                    break
        if not [r for r in rows if not r[5]]:
            continue
        cfg = run_config(rows[0][1])
        score = score_fn(cfg)
        gold_ids = sorted(rows[0][3]["gold"])
        gold = [rows[0][3]["gold"][i] for i in gold_ids]
        preds = []
        for _, _, rep, data, _, _ in rows:
            assert set(data["gold"]) == set(gold_ids), "val sets differ"
            preds.append([data["representatives"][rep][i] for i in gold_ids])
        boot = bootstrap(gold, preds, score)
        point = [score(gold, p) for p in preds]
        idx = {r[0]: j for j, r in enumerate(rows)}
        ref_j = idx.get(CONFIGS[0][0])
        same_stage = {c[0]: c[4] for c in CONFIGS}
        lines += [f"## {title} ({metric_name}, val n={len(gold)})", "",
                  "| Config | Run / representative | Val (95% CI) | Δ v2-A (95% CI) | Δ paper, same stage (95% CI) | train CV | Jev calls/ex | input tok/ex |",
                  "|---|---|---|---|---|---:|---:|---:|"]
        summary[stem] = {}
        for j, (label, run_dir, rep, data, pareto, ref) in enumerate(rows):
            lo, hi = np.percentile(boot[:, j], [2.5, 97.5])

            def delta(k: int | None) -> str:
                if k is None or k == j:
                    return ""
                d = boot[:, j] - boot[:, k]
                dlo, dhi = np.percentile(d, [2.5, 97.5])
                return f"{100 * (point[j] - point[k]):+.1f} ({100 * dlo:+.1f}…{100 * dhi:+.1f})"

            cand = pareto.get("candidate", "")
            meta_path = run_dir / cand.split("/")[0] / "candidates" / f"{cand.split('/')[-1]}.json" if "/" in cand else None
            train_cv = json.loads(meta_path.read_text()).get("train_cv") if meta_path and meta_path.exists() else None
            v = pareto["val"]
            member = f" = {cand}" if cand and cand != rep else ""
            cv_txt = f"{100 * train_cv:.1f}" if isinstance(train_cv, (int, float)) else ""
            lines.append(f"| {label} | `{run_dir.name}` {rep}{member} | **{100 * point[j]:.1f}** ({100 * lo:.1f}…{100 * hi:.1f}) | {delta(ref_j)} | {delta(idx.get(same_stage.get(label) or ''))} | {cv_txt} | {v.get('mean_calls', 0):.2f} | {v.get('mean_input_tokens', 0):.0f} |")
            summary[stem][label] = {"run": run_dir.name, "representative": rep, "candidate": cand, "val": point[j], "ci": [lo, hi], "train_cv": train_cv, "mean_calls": v.get("mean_calls"), "mean_input_tokens": v.get("mean_input_tokens")}
        seeds_rows = seed_table(RUNS / f"v3-{stem}-P", gold_ids, gold, score)
        if seeds_rows:
            lines += ["", "Every seed zero-shot (as written, no fitted number); the kind's row above is its seed with the best zero-shot train score:", "",
                      "| Seed | Kind | zero-shot train | zero-shot val | train CV with head |", "|---|---|---:|---:|---:|", *seeds_rows]
        choice = select({r[0]: point[j] for j, r in enumerate(rows) if not r[5] and r[0] in CHOICES})
        summary[stem]["chosen"] = choice
        if choice:
            c = summary[stem][choice]
            chosen_rows.append(f"| {title} | {choice} | `{c['run']}` | `{c['representative']}` ({c['candidate']}) | {100 * c['val']:.1f} |")
        lines.append("")
        for run_dir in dict.fromkeys(r[1] for r in rows if not r[5]):
            h = hours(run_dir)
            f_cost, f_calls, f_tok = fresh(run_dir)
            costs.append(f"| {title} | `{run_dir.name}` | ${jev_cost(run_dir):.2f} | ${f_cost:.2f} | {f_calls:,} | {f_tok / 1e6:.2f} | ${omp_cost(run_dir):.2f} | {'' if h is None else f'{h:.2f}'} |")
    lines += ["## Selected configuration per task (pre-registered rule)", "",
              f"Start from A; P, then P+head, replaces the current choice only if its val primary metric is ≥ {MARGIN:.1f} point higher.", "",
              "| Task | Config | Run | Representative | Val |", "|---|---|---|---|---:|", *chosen_rows, ""]
    lines += ["## Cost per run", "",
              "Jev used: sum of the run's unique Jev requests (fresh or served from the shared cache; requests reused from v1/v2 runs count). "
              "Jev fresh: billed during this run (requests, M input tokens). OMP: proposer LLM spend reported by the host.", "",
              "| Task | Run | Jev used $ | Jev fresh $ | fresh requests | fresh M tokens | OMP $ | wall h |", "|---|---|---:|---:|---:|---:|---:|---:|", *costs, ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    (RUNS / "v3-dev-summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
