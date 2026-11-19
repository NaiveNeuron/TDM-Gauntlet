"""Write the jev-tailored task files (Jev-adapted GEPA) from the v3 task files: same data, splits, metric,
paper seed, 400-example fit slice, CV grouping and jev-native seeds; only the optimizer block differs.

  task_tailored<suffix>.yaml        main run: starts paper + best jev-native seed, T1-T4 + T6
  task_tailored_noT1<suffix>.yaml   ablation: start paper only, no rulebook (half the OMP budget, Jev <= $1)
  task_tailored_noT2<suffix>.yaml   ablation: start paper only, GEPA's default 3-example acceptance

OMP budget per task = the reflection spend of the task's v1 program-opt run (runs/jev-opt/study-<stem>-program,
run.json), the cheaper v1 optimizing run on 7 of 8 tasks. Run from the repo root:
  uv run --project jev-opt/python python jev-opt/experiments/make_tailored_tasks.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parent
RUNS = EXP.parents[1] / "runs" / "jev-opt"
TASKS = [  # experiment dir, run-name stem, task-file suffix
    ("contempo_tbdense", "contempo-tbdense", ""),
    ("ezstance_subtaskA", "ezstance-subtaskA", ""),
    ("smarter_implicit_hate", "smarter-implicit-hate", ""),
    ("edit_intent", "edit-intent", ""),
    ("inli", "inli", ""),
    ("dress_new", "dress-new-content", "_content"),
    ("dress_new", "dress-new-organization", "_organization"),
    ("dress_new", "dress-new-language", "_language"),
]
HEAD = """# jev-tailored dev study (Jev-adapted GEPA), written by make_tailored_tasks.py from task_v3{suffix}.yaml:
# same data, splits, metric, paper seed (the gauntlet prompt), 400-example stratified train fit slice, CV grouping
# and jev-native seeds (seeds_v3{suffix}/, written from the task definition only). {what}
# OMP budget (max_reflection_cost) = the reflection spend of v1 study-{stem}-program (${v1:.4f}){share}.
"""
WHAT = {
    "": "GEPA text search per start (paper; the jev-native seed with the best zero-shot train score) with the Jev rulebook (T1), acceptance on 64 stratified train examples by mean log P(gold) with a 2-point metric guard (T2), joint rewrites of the most confused, least separable option pair (T3), minimal pairs (T4), candidates scored after out-of-fold class biases / cutoffs (T6). GEPA merge off (merges would bypass T2).",
    "_noT1": "Ablation: start paper only, no rulebook (T1 off); everything else as task_tailored{suffix}.yaml.",
    "_noT2": "Ablation: start paper only, GEPA's default acceptance on 3-example minibatches (T2 off); everything else as task_tailored{suffix}.yaml.",
}


def main() -> int:
    for exp, stem, suffix in TASKS:
        v3 = yaml.safe_load((EXP / exp / f"task_v3{suffix}.yaml").read_text())
        v1 = float(json.loads((RUNS / f"study-{stem}-program" / "run.json").read_text())["conditions"]["jev-program-opt"]["summary"]["reflection_cost"])
        opt = v3["optimizer"]
        native = [ref for ref in opt["population"]["seeds"] if ref["kind"] == "jev-native"]
        for pop_only in ("population", "fit_holdout", "reflection_pool"):
            opt.pop(pop_only, None)
        opt.update(readout=False, class_bias=True, calibrate=True, fit_parameters=True, use_merge=False, max_metric_calls=40000)
        for variant in ("", "_noT1", "_noT2"):
            task = json.loads(json.dumps(v3))
            o = task["optimizer"]
            tailored = {"starts": ["paper", "jev-native"], "seeds": native, "rulebook": True, "continuous_accept": True, "accept_n": 64, "accept_tol": 0.02,
                        "confusion_target": True, "minimal_pairs": 4, "max_examples": 8, "max_jev_cost": 2.0, "head": True}
            budget, share = round(v1, 2), ""
            if variant:
                tailored.update(starts=["paper"], max_jev_cost=1.0, rulebook=variant != "_noT1", continuous_accept=variant != "_noT2")
                budget, share = round(v1 / 2, 2), "; ablations: half of it, the paper start's share of the main run"
            o["max_reflection_cost"] = budget
            o["tailored"] = tailored
            text = HEAD.format(suffix=suffix, stem=stem, v1=v1, share=share, what=WHAT[variant].format(suffix=suffix)).replace("\n# ", " ").removeprefix("# ")
            head = "".join(f"# {line}\n" for line in textwrap.wrap(text.strip(), 110))
            out = EXP / exp / f"task_tailored{variant}{suffix}.yaml"
            out.write_text(head + yaml.safe_dump(task, sort_keys=False, allow_unicode=True, width=110))
            print(out.relative_to(EXP), budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
