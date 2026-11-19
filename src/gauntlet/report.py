"""Aggregate runs/<task>/{scores,manifest}.json with paper-reported numbers.

`results/paper_numbers.json` maps task name -> {"metric": key in scores.json,
"paper": {system label: value}, "source": "table/page citation"}.
Writes results/summary.md (a table) and prints it.
"""

from __future__ import annotations

import json
from pathlib import Path

from .task import REPO

RUNS = REPO / "runs"
RESULTS = REPO / "results"


def _fmt(x: object) -> str:
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def build() -> str:
    paper = json.loads((RESULTS / "paper_numbers.json").read_text()) if (RESULTS / "paper_numbers.json").exists() else {}
    lines = [
        "| Task | N | Metric | Jev | Jev, paper's protocol | Paper method | Paper value | Best paper baseline | Input tokens | Cost (USD) | Snapshot |",
        "|---|---:|---|---:|---:|---|---:|---|---:|---:|---|",
    ]
    for scores_path in sorted(RUNS.glob("**/scores.json")):
        # Zero-shot audit runs only: smoke tests and the optimization study's runs (dev_v3, <task>_v3,
        # jev-opt) are reported in results/GEPA*.md; delta_stance was dropped from the study (its official
        # test split was never released).
        top = scores_path.relative_to(RUNS).parts[0]
        if top in ("clef", "clef-flash", "luna", "msd1", "_smoke", "dev_v3", "jev-opt", "delta_stance") or top.endswith("_v3"):
            continue
        task_dir = scores_path.parent
        name = str(task_dir.relative_to(RUNS))
        scores = json.loads(scores_path.read_text())
        manifest = json.loads((task_dir / "manifest.json").read_text())
        ref = paper.get(name, {})
        metric = ref.get("metric", next(iter(k for k in scores if k != "n"), "?"))
        systems = ref.get("paper", {})
        method = ref.get("method", "")
        method_val = systems.get(method, "")
        baselines = {k: v for k, v in systems.items() if k != method}
        best = max(baselines.items(), key=lambda kv: kv[1]) if baselines else ("", "")
        # Optional like-for-like score when the paper's protocol differs from our primary
        # (dotted path into scores.json, e.g. a test-CV interpolation the paper used).
        protocol = ""
        if "jev_paper_protocol" in ref:
            node: object = scores
            for part in ref["jev_paper_protocol"]["path"].split("."):
                node = node[part]  # type: ignore[index]
            protocol = f"{_fmt(node)} ({ref['jev_paper_protocol']['note']})"
        lines.append(
            f"| {name} | {manifest['n_items']} | {metric} | {_fmt(scores.get(metric, 'n/a'))} | {protocol} | {method} | "
            f"{_fmt(method_val)} | {best[0]} {_fmt(best[1])} | {manifest['input_tokens']:,} | "
            f"{manifest['cost_usd']:.4f} | {', '.join(manifest['model_snapshots'])} |"
        )
    table = "\n".join(lines) + "\n"
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.md").write_text(table)
    return table


if __name__ == "__main__":
    print(build())
