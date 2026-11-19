"""Report the registered `<task>+head` test runs -> results/HEADS.md, results/heads.json.

For each task: the dev evidence (results/v3_dev/heads.json), the saved zero-shot test row
(runs/<task>/), the single `+head` test run (runs/<task>+head/), and a paired bootstrap 95% CI of the
test Δ over test items (2,000 resamples, seed 0; basic interval, as the dev rule). No Jev call.

    uv run python scripts/report_heads.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from fit_v3 import HEAD_TASKS, macro_counts, macro_metric, micro_counts, micro_metric, paired_ci, value_ci  # noqa: E402
from gauntlet.task import read_jsonl  # noqa: E402

RUNS = REPO / "runs"
NAMES = {"contempo/matres": "ConTempo MATRES", "contempo/tbdense": "ConTempo TB-Dense", "smarter/hatexplain": "SMARTER HateXplain",
         "smarter/latent_hate": "SMARTER Latent Hate", "smarter/implicit_hate": "SMARTER Implicit Hate"}
HEAD_DESC = {"bias": "per-class log-biases", "lr": "L2 logistic regression"}


def preds(d: Path) -> dict[str, dict[str, Any]]:
    path = d / "predictions.jsonl.gz" if (d / "predictions.jsonl.gz").exists() else d / "predictions.jsonl"
    return {r["item_id"]: r for r in read_jsonl(path)}


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def main() -> None:
    dev = json.loads((REPO / "results" / "v3_dev" / "heads.json").read_text())
    out: dict[str, Any] = {}
    rows, cost_rows = [], []
    for task, (_, kind, neg, _) in HEAD_TASKS.items():
        hd = RUNS / f"{task}+head"
        if not (hd / "scores.json").exists():
            rows.append(f"| {NAMES[task]} | not run | | | | | | |")
            continue
        zero, head = preds(RUNS / task), preds(hd)
        assert zero.keys() == head.keys(), f"{task}: item sets differ"
        ids = sorted(zero)
        labels = dev[task]["labels"]
        k = len(labels)
        # items whose zero-shot prediction failed carry no label; the task's own scorer counts them as wrong
        idx = lambda lab: labels.index(lab) if lab in labels else -1  # noqa: E731
        y = np.array([labels.index(zero[i]["gold"]) for i in ids])
        pz = np.array([idx((zero[i]["prediction"] or {}).get("label")) for i in ids])
        ph = np.array([idx((head[i]["prediction"] or {}).get("label")) for i in ids])
        assert (pz >= 0).all() and (ph >= 0).all(), f"{task}: failed predictions need explicit handling"
        if kind == "micro":
            ni = labels.index(neg)
            cz, ch, agg = micro_counts(y, pz, ni), micro_counts(y, ph, ni), micro_metric
        else:
            cz, ch, agg = macro_counts(y, pz, k), macro_counts(y, ph, k), macro_metric(k)
        ci = paired_ci(ch, cz, agg)
        vz, vh = value_ci(cz, agg), value_ci(ch, agg)
        sz, sh = json.loads((RUNS / task / "scores.json").read_text()), json.loads((hd / "scores.json").read_text())
        key = "micro_f1" if kind == "micro" else "macro_f1"
        assert abs(sz[key] - vz["value"]) < 1e-9 and abs(sh[key] - vh["value"]) < 1e-9, f"{task}: recomputed metric differs from scores.json"
        man = json.loads((hd / "manifest.json").read_text())
        spec = json.loads((REPO / "data" / "frozen" / "heads" / f"{task}.json").read_text())["head"]
        # fitted numbers; per-class biases keep the first label's at 0 (k - 1 free)
        n_params = k - 1 if spec["kind"] == "bias" else sum(np.asarray(spec[f]).size for f in ("coef", "intercept"))
        d = dev[task]
        out[task] = {"metric": d["metric"], "n_test": len(ids), "head": spec["kind"], "chosen": d["chosen"], "n_params": int(n_params),
                     "dev_zero_shot": d["dev_zero_shot"]["value"], "dev_head": d["dev_head"]["value"], "dev_delta": d["dev_delta"],
                     "test_zero_shot": vz, "test_head": vh, "test_delta": ci, "n_changed": sh.get("n_changed_from_zero_shot"),
                     "test_cost_usd_new": man.get("cost_usd_excluding_reused"), "test_requests": man.get("n_requests"),
                     "dev_cost_usd": d["dev_run"]["cost_usd"], "head_file_sha256": sh.get("head", {}).get("sha256")}
        rows.append(f"| {NAMES[task]} | {'micro-F1 (VAGUE neg.)' if kind == 'micro' else 'macro-F1'} | {HEAD_DESC[spec['kind']]} ({d['chosen']}; {n_params} params) | "
                    f"{pct(d['dev_zero_shot']['value'])} → {pct(d['dev_head']['value'])} ({100 * d['dev_delta']['delta']:+.1f} [{100 * d['dev_delta']['lo']:+.1f}, {100 * d['dev_delta']['hi']:+.1f}]) | "
                    f"{pct(vz['value'])} | {pct(vh['value'])} | {100 * ci['delta']:+.1f} [{100 * ci['lo']:+.1f}, {100 * ci['hi']:+.1f}] | {sh.get('n_changed_from_zero_shot', 0):,} / {len(ids):,} |")
        cost_rows.append(f"| {NAMES[task]} | `dev_v3/{HEAD_TASKS[task][0]}` | {d['dev_run']['n_items']:,} | {d['dev_run']['cost_usd']:.2f} | "
                         f"{man.get('n_requests', 0):,} | {man.get('cost_usd_excluding_reused', 0.0):.2f} |")
    dev_total = sum(dev[t]["dev_run"]["cost_usd"] for t in dev)
    test_total = sum(v["test_cost_usd_new"] or 0.0 for v in out.values())
    L = ["# Fitted readout heads for the zero-shot audit: test results", "",
         "The dev rule and the frozen heads were fixed before any test run. Generated by",
         "`scripts/report_heads.py`. Condition: **Jev + fitted head** (a small readout fit on released train/dev data over the",
         "frozen audit prompt's Choice probabilities; not zero-shot), reported next to the zero-shot audit row of the same test set.",
         "Each `+head` task was run on test once. Values ×100; Δ = paired bootstrap 95% CI over test items (2,000 resamples,",
         "seed 0, basic interval); dev Δ = held-out dev part (the registration rule).", "",
         "| Task | Metric | Head | Dev: zero-shot → head (Δ [95% CI]) | Test zero-shot | Test + head | Test Δ [95% CI] | Test labels changed |",
         "|---|---|---|---|---:|---:|---:|---:|", *rows, "",
         "## Cost", "",
         "Dev: the frozen audit prompt on the dev sample (Jev). Test: new Jev spend of the `+head` run (its requests are the",
         "zero-shot run's, served from the cache).", "",
         "| Task | Dev run | dev items | dev $ | test requests | test $ new |", "|---|---|---:|---:|---:|---:|", *cost_rows,
         f"| **total** | | | **{dev_total:.2f}** | | **{test_total:.2f}** |", "",
         "## Notes", "",
         "- All five heads passed the dev rule (Δ CI above 0); none was changed after a test score was seen.",
         "- The SMARTER fit, dev and test samples are all class-balanced; ConTempo samples are not (fit, dev and test alike; "
         "read from the zero-shot test runs' label counts, no `+head` score involved).",
         "- TB-Dense's dev Δ lower bound was +0.2, the weakest registration.", ""]
    (REPO / "results" / "HEADS.md").write_text("\n".join(L))
    (REPO / "results" / "heads.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
