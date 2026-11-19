# /// script
# dependencies = ["numpy", "pyyaml"]
# ///
"""Analyze the first optimization study (GEPA text and program search; its settings were fixed before any run)
and write results/GEPA.md.

For every finished run it re-executes the program that consumed the test split (recorded in
test_consumed.json) on that same test set, served entirely from the Jev cache. A cache miss is
an error: JevClient._post is replaced with a function that raises, so no paid call and no
second test query can happen. The re-executed primary metric must equal
the one recorded at consumption time. Paired bootstrap CIs (2,000 resamples, seed 0) compare
text-opt and program-opt against zero on the same test examples.

Run from the repo root:  uv run --project jev-opt/python python jev-opt/experiments/analyze.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENROUTER_API_KEY", "offline-cache-only")  # never used: posting is disabled below

import numpy as np  # noqa: E402

# Must match the worker's Python: 3.12 changed float sum() (compensated summation), which moves
# Score expected levels that fall exactly on a cutoff (found via 2 DREsS runs under 3.13).
assert sys.version_info[:2] == (3, 11), f"run under Python 3.11 (worker's version), not {sys.version}"

from jev_opt import metrics  # noqa: E402
from jev_opt.execute import run_examples  # noqa: E402
from jev_opt.hygiene import allow_test_split, load_split  # noqa: E402
from jev_opt.ir import Program  # noqa: E402
from jev_opt.jev import JevClient, make_client  # noqa: E402


class CacheMiss(RuntimeError):
    pass


def _no_network(self, body):  # the cache key includes the URL, so keep the real URL and forbid posting
    raise CacheMiss("analysis must be served from the Jev cache; a request was not cached")


JevClient._post = _no_network
from jev_opt.runs import _load_run_config  # noqa: E402

ROOT = Path.cwd()
RUNS = ROOT / "runs" / "jev-opt"
OUT = ROOT / "results" / "GEPA.md"
DISCREPANCIES: list[tuple[str, float, float]] = []
TASKS = [
    ("contempo-tbdense", "ConTempo TB-Dense", "micro-F1 (VAGUE negative)", {"paper": 0.6856, "zero-shot LLM": 0.6758}),
    ("ezstance-subtaskA", "EZ-STANCE Subtask A", "macro-F1", {"paper": 0.812, "zero-shot LLM": 0.499}),
    ("smarter-implicit-hate", "SMARTER Implicit Hate", "macro-F1", {"paper": 0.60, "zero-shot LLM": 0.58}),
    ("edit-intent", "Edit intent (Re3-Sci)", "macro-F1", {"paper": 0.843, "zero-shot LLM": 0.37}),
    ("inli", "INLI", "accuracy", {"paper": 0.924, "zero-shot LLM": 0.770}),
    ("dress-new-content", "DREsS content", "QWK", {"paper": 0.631, "zero-shot LLM": 0.31}),
    ("dress-new-organization", "DREsS organization", "QWK", {"paper": 0.771, "zero-shot LLM": 0.322}),
    ("dress-new-language", "DREsS language", "QWK", {"paper": 0.589, "zero-shot LLM": 0.231}),
]
CONDS = ("zero", "text", "program")
B = 2000


def evaluate(run_dir: Path):
    consumed = json.loads((run_dir / "test_consumed.json").read_text())
    config = _load_run_config(run_dir)
    program = Program.from_yaml((run_dir / f"{consumed['representative']}.yaml").read_text())
    with allow_test_split():
        test = load_split(config, "test")
    client = make_client(RUNS / ".jev-cache", config.jev.model, config.jev.max_rpm, config.jev.concurrency, [config.base_dir, ROOT])
    try:
        results = run_examples(program, test, client, context={"split": "test", "condition": "analysis"})
    finally:
        client.close()
    gold = [r.gold for r in results]
    pred = [r.pred for r in results]
    m = metrics.compute(gold, pred, [r.dist for r in results], config.labels, config.primary_metric, config.negative_label)
    recorded = consumed["metrics"]["primary"]
    if abs(m["primary"] - recorded) > 1e-9:
        # Reported, never hidden: the table uses the reproducible re-executed value.
        DISCREPANCIES.append((run_dir.name, recorded, m["primary"]))
    return {"gold": gold, "pred": pred, "m": m, "config": config, "candidate": consumed["candidate"], "calls": m.get("mean_calls")}


def primary_on(idx, r, config):
    g = [r["gold"][i] for i in idx]
    p = [r["pred"][i] for i in idx]
    return metrics.primary(config.primary_metric, g, p, config.labels, config.negative_label)


def main() -> int:
    rng = np.random.default_rng(0)
    rows, missing = [], []
    for slug, title, metric_name, ref in TASKS:
        res = {}
        for c in CONDS:
            d = RUNS / f"study-{slug}-{c}"
            if (d / "test_consumed.json").exists():
                res[c] = evaluate(d)
            else:
                missing.append(d.name)
        if "zero" not in res:
            continue
        cfg = res["zero"]["config"]
        n = len(res["zero"]["gold"])
        assert all(res[c]["gold"] == res["zero"]["gold"] for c in res), f"{slug}: test gold differs across runs"
        boots = rng.integers(0, n, size=(B, n))
        z = res["zero"]["m"]["primary"]
        entry = {"slug": slug, "title": title, "metric": metric_name, "n": n, "zero": z, "ref": ref, "conds": {}}
        for c in ("text", "program"):
            if c not in res:
                continue
            v = res[c]["m"]["primary"]
            deltas = np.array([primary_on(b, res[c], cfg) - primary_on(b, res["zero"], cfg) for b in boots])
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            entry["conds"][c] = {"value": v, "delta": v - z, "ci": (float(lo), float(hi)), "candidate": res[c]["candidate"],
                                 "brier": res[c]["m"]["brier"], "ece": res[c]["m"]["ece"]}
        entry["zero_brier"], entry["zero_ece"] = res["zero"]["m"]["brier"], res["zero"]["m"]["ece"]
        rows.append(entry)
    write(rows, missing)
    (OUT.parent / "gepa.json").write_text(json.dumps({"rows": rows, "missing": missing, "discrepancies": DISCREPANCIES}, indent=2) + "\n")
    print(OUT.read_text())
    return 0


def fmt(x: float) -> str:
    return f"{100 * x:.1f}"


def write(rows, missing) -> None:
    lines = [
        "# GEPA optimization study: results",
        "",
        "First optimization study; its settings were fixed before any run. Data and splits:",
        "`jev-opt/experiments/README.md`. Generated by `jev-opt/experiments/analyze.py`, which",
        "re-executes each test-evaluated program from the Jev cache (offline) and compares with the",
        "score recorded at test time (discrepancies are listed below). Test = the gauntlet's frozen English test set; each run's test",
        "split was consumed exactly once, by its `top-quality` candidate (highest validation score).",
        "",
        "Δ = test score minus the zero-shot seed (same prompt as the gauntlet), with a paired",
        "bootstrap 95% CI over test examples (2,000 resamples). Values ×100.",
        "",
        "| Task | Metric | N | Zero | Text-opt (Δ, 95% CI) | Program-opt (Δ, 95% CI) | Paper method | Zero-shot LLM in paper |",
        "|---|---|---:|---:|---|---|---:|---:|",
    ]
    for e in rows:
        cells = []
        for c in ("text", "program"):
            x = e["conds"].get(c)
            cells.append("—" if x is None else f"**{fmt(x['value'])}** ({x['delta'] * 100:+.1f}; {x['ci'][0] * 100:+.1f}…{x['ci'][1] * 100:+.1f})")
        lines.append(f"| {e['title']} | {e['metric']} | {e['n']:,} | {fmt(e['zero'])} | {cells[0]} | {cells[1]} | {fmt(e['ref']['paper'])} | {fmt(e['ref']['zero-shot LLM'])} |")
    lines += ["", "## Calibration (test, recomputed with the fixed metric)", "",
              "| Task | Zero Brier / ECE | Text-opt Brier / ECE | Program-opt Brier / ECE |", "|---|---|---|---|"]
    for e in rows:
        cal = [f"{e['zero_brier']:.3f} / {e['zero_ece']:.3f}"]
        for c in ("text", "program"):
            x = e["conds"].get(c)
            cal.append("—" if x is None else f"{x['brier']:.3f} / {x['ece']:.3f}")
        lines.append(f"| {e['title']} | " + " | ".join(cal) + " |")
    if missing:
        lines += ["", f"Runs without a test evaluation yet: {', '.join(missing)}."]
    if DISCREPANCIES:
        lines += ["", "**Reproduction discrepancies** (re-executed from the cached Jev responses vs the score recorded when the",
                  "test split was consumed; same program hash, same test file hash, identical cached responses. Unexplained;",
                  "the table uses the re-executed value):", ""]
        lines += [f"- `{n}`: recorded {100 * a:.2f}, re-executed {100 * b:.2f}" for n, a, b in DISCREPANCIES]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
