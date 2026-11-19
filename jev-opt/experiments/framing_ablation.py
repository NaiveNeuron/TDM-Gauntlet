"""Ablation and cost numbers for the paper's framing -> results/framing_ablation.json.

  ablation   per study task, validation only: A = the paper's question + fitted readout (`v3-<task>-A`, top-quality)
             vs P = feature-question search (`v3-<task>-P`, top-single: rewritten questions + fitted readout; seed 0,
             the pre-registered v3 run), both from `val_predictions.json`, primary metric recomputed as
             analyze_v3.py does; paired bootstrap of P - A over val examples (2,000 resamples, seed 0, 95% percentile
             CI). Checked against runs/jev-opt/v3-dev-summary.json.
  readout    test, from saved results: readout alone (the audit question's zero-shot answers + fitted readout,
             results/heads.json) vs the method (3-seed test mean and sd, results/seeds.json), where both exist.
  cost       per task, optimization only (no test): Jev used = Jev-reported cost of the distinct train/val requests in
             the run's response ledger (requests served from the shared cache count, as in tab:opt-cost),
             optimizer = omp spend in usage.jsonl; feature-question search (`v3-<task>-P`, and seeds 1/2) vs GEPA text
             search (`study-<task>-text`) and GEPA program search (`study-<task>-program`); wall hours from run.json.

No Jev or LLM call is made and no test file is opened.
Run from the repo root:  uv run --project jev-opt/python python jev-opt/experiments/framing_ablation.py
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_v2 import TASKS, hours, omp_cost, run_config, score_fn  # noqa: E402

ROOT = Path.cwd()
RUNS = ROOT / "runs" / "jev-opt"
OUT = ROOT / "results" / "framing_ablation.json"
B = 2000
HEADS = {"contempo-tbdense": "contempo/tbdense", "smarter-implicit-hate": "smarter/implicit_hate"}


def val_preds(run: str, rep: str) -> tuple[list[str], dict[str, str], list[str]]:
    d = RUNS / run
    data = json.loads((d / "val_predictions.json").read_text())
    ids = sorted(data["gold"])
    return ids, data["gold"], [data["representatives"][rep][i] for i in ids]


def ablation() -> dict[str, Any]:
    summary = json.loads((RUNS / "v3-dev-summary.json").read_text())
    out = {}
    for stem, title, metric in TASKS:
        score = score_fn(run_config(RUNS / f"v3-{stem}-P"))
        ids_a, gold_a, a = val_preds(f"v3-{stem}-A", "top-quality")
        ids_p, gold_p, p = val_preds(f"v3-{stem}-P", "top-single")
        assert ids_a == ids_p and gold_a == gold_p, f"{stem}: A and P val sets differ"
        gold = [gold_p[i] for i in ids_p]
        n = len(gold)
        va, vp = score(gold, a), score(gold, p)
        for key, v in (("A: seed + readout head", va), ("P: population-opt", vp)):
            assert abs(summary[stem][key]["val"] - v) < 1e-9, f"{stem} {key}: {v} vs summary {summary[stem][key]['val']}"
        boots = np.random.default_rng(0).integers(0, n, size=(B, n))
        d = [score([gold[i] for i in ix], [p[i] for i in ix]) - score([gold[i] for i in ix], [a[i] for i in ix]) for ix in boots]
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[stem] = {"title": title, "metric": metric, "n_val": n, "A": va, "P": vp, "delta": vp - va, "ci": [float(lo), float(hi)],
                     "A_run": f"v3-{stem}-A", "P_run": f"v3-{stem}-P", "P_candidate": summary[stem]["P: population-opt"]["candidate"]}
    return out


def readout_alone() -> dict[str, Any]:
    heads = json.loads((ROOT / "results" / "heads.json").read_text())
    seeds = json.loads((ROOT / "results" / "seeds.json").read_text())["tasks"]
    out = {}
    for stem, key in HEADS.items():
        h = heads[key]
        s = seeds[stem]
        out[stem] = {"readout_alone": h["test_head"]["value"], "readout_alone_zero_shot": h["test_zero_shot"]["value"],
                     "readout_alone_params": h["n_params"], "readout_alone_kind": h["chosen"],
                     "method_mean": s["test_mean"], "method_sd": s["test_sd"], "method_n": s["n_test"],
                     "method_seeds": [s["seeds"][k]["test"] for k in sorted(s["seeds"])]}
    return out


def jev_used(run: str) -> float:
    """Distinct train/val requests of the run's response ledger at Jev's reported cost."""
    d = RUNS / run
    total, seen = 0.0, set()
    for shard in json.loads((d / "run.json").read_text())["jev_responses"]["shards"]:
        with gzip.open(d / shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                split = r.get("split")
                if split is None:
                    raise ValueError(f"{run}: ledger record without split")
                if split == "test" or r["request_hash"] in seen:
                    continue
                seen.add(r["request_hash"])
                total += float((r["response"].get("usage") or {}).get("cost") or 0.0)
    return total


def cost() -> dict[str, Any]:
    out = {}
    for stem, title, _ in TASKS:
        runs = {"fqs": f"v3-{stem}-P", "fqs_s1": f"v3-{stem}-P-s1", "fqs_s2": f"v3-{stem}-P-s2",
                "gepa_text": f"study-{stem}-text", "gepa_program": f"study-{stem}-program"}
        row: dict[str, Any] = {"title": title}
        for k, run in runs.items():
            j, o = jev_used(run), omp_cost(RUNS / run)
            row[k] = {"run": run, "jev_used": j, "omp": o, "total": j + o, "wall_h": hours(RUNS / run)}
        row["fqs_seed_mean_total"] = float(np.mean([row[k]["total"] for k in ("fqs", "fqs_s1", "fqs_s2")]))
        out[stem] = row
        print(stem, {k: round(row[k]["total"], 2) for k in runs}, flush=True)
    return out


def main() -> int:
    out = {"generated_by": "jev-opt/experiments/framing_ablation.py", "bootstrap": {"B": B, "seed": 0},
           "ablation": ablation(), "readout_alone": readout_alone(), "cost": cost()}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
