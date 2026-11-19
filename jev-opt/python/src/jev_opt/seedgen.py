"""Jev-native seed programs written by the OMP model from the task definition only (§5.2).

The paper seeds are faithful copies of LLM-style prompts; population search is anchored to them.
`generate` asks the optimizer model (through OMP, like every optimizer LLM call) for alternative
seed programs written from the label definitions and the TypeSafe authoring guidance below:

- `jev-native`: the paper's topology and primitives, every text rewritten for Jev;
- `decomposed`: the writer may pick other primitives / topology per node (one-vs-rest Nouls, a
  coarse-to-fine hierarchy, a Score for ordinal tasks, branch Nouls).

The writer sees the paper seed's definitional text (task and label definitions, rubric wording) and
the state field names, never a data example or a gold label, so every generated seed is itself a
zero-shot program. Seeds are validated against the IR and hygiene, written as complete programs,
and written before any of them is scored.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic import ValidationError

from .hygiene import ProgramGuard
from .ir import ChoiceNode, OneVsRestNode, Program, ProgramError, ScoreNode, StageNode, SwitchNode
from .task import TaskConfig
from .transforms import parse_proposal, trim_slots

LM = Callable[[str | list[dict[str, Any]]], str]
KINDS = ("jev-native", "decomposed")
NODE_ID = re.compile(r"^[a-z][a-z0-9_]{0,40}$")

# Distilled (2026-09-27) from docs.typesafe.ai (how-to-build-with-system-one, primitives, Score
# "Writing good levels", model-jaggedness/jev-1.13), Kiln's "Auto-optimizing Jev" report and
# T. Kellogg's Jev optimization notes.
GUIDANCE = """How Jev reads a question (TypeSafe guidance for jev-1.13):
- Jev answers each question independently against the state: a Choice returns a probability for every option (relative), a Noul the probability that one statement is true (absolute), a Score a probability per level. Questions in one request never see each other.
- Jev reads literally. State the exact condition in the instructions; put boundary cases in the criteria. Scoping words, negations and implied conditions are taken at face value. Avoid double negatives and indirection (a property of a property, multi-hop conditions); name the state field a question is about, in backticks.
- Keep instructions short and literal: one question, not a procedure. Jev does not reliably follow sequences of steps, and long background dumps, worked procedures and translated guidelines did not help or hurt in published Jev optimizations.
- What helps most: criteria that say what SEPARATES each option from its nearest neighbour (what it covers, what belongs to the other option instead). Instructions and criteria must agree.
- Ask atomic questions: one judgment per question. A broad judgment hides several; split it into narrow questions and combine them in code (routing, hierarchy).
- Score levels: describe observable situations, not degrees ("minor errors that never block understanding", not "fairly good"). Every level must have its own description; the model sees each level on its own, without its number or neighbours, so "better than the previous level" and bare numbers mean nothing. One dimension per Score.
- The evidence a criterion refers to must be present in the state and named there; never ask Jev to count, do arithmetic or compare dates.
- Choice: fine up to many options, but each option needs its own contrastive criterion. Noul: a single true/false statement."""

SEED_IR_GUIDE = """Program nodes (YAML; the numbers below are fixed, nothing is fitted on data for these programs):
- choice: {instructions, state: [fields], options: {key: {criterion, to?: {label: L} | {node: id}}}}: argmax routing; `to` defaults to the label named like the option.
- one_vs_rest: {instructions, state, options: {label: {criterion}}}: one Noul per option (the criterion is a statement judged true or false); the label of the most probable Noul wins.
- score: {instructions, state, levels: [2..10 level descriptions, lowest first], outcomes: [label targets, lowest first]}: with one outcome per level the expected level is rounded; otherwise the level range is split evenly into the outcomes.
- branch: {condition (a true/false statement), state, if_true: target, if_false: target}: routes on p >= 0.5.
Rules: every label must be reachable; node ids use lowercase letters, digits and underscores (never '__'); `state` lists input fields only; state fields are named in the question text with backticks. Never include example texts or ids."""

SEED_TEMPLATE = """You are writing decision programs for Jev, a calibrated classifier (TypeSafe System One), from a task definition only.

""" + GUIDANCE + """

TASK DEFINITION (the paper's wording: the only source you may use; there are no data examples)
<task_info>

The paper's current program (definitions to preserve; its style is LLM-prompt style, not Jev style):
```
<curr_param>
```

""" + SEED_IR_GUIDE + """

<kind_rules>

Write <n> programs that differ clearly from each other (different wording strategies<kind_diff>). Keep every definition's meaning; do not invent example inputs. Every text field must be at most <limit> characters.

Reply with one YAML list inside a single ``` block, one item per program:
```
- rationale: <one line: the strategy of this program>
  nodes:
<node_skeleton>
```"""

KIND_RULES = {
    "jev-native": (
        "KIND jev-native: for each node id below, write ONE question of the same kind as in the paper's program (same option keys "
        "and targets, or the same number of Score levels and outcomes), rewriting every text for Jev by the guidance: short literal "
        "instructions, contrastive criteria, every Score level described by observable rules. No other nodes."
    ),
    "decomposed": (
        "KIND decomposed: for each node id below, you may choose another primitive or topology that fits the label definitions: "
        "`one_vs_rest` (one noul per label), a coarse-to-fine hierarchy (a `choice` over groups whose options route to finer "
        "`choice` nodes), a `score` over ordered situations (ordinal tasks only), or `branch` nouls routing to labels or nodes. "
        "Routing is by argmax (choice), p >= 0.5 (branch), argmax of the noul probabilities (one_vs_rest) and the expected level "
        "(score). Extra node ids start with the entry node's id and an underscore. Every label must stay reachable. No `stage` "
        "or `switch` nodes."
    ),
}


def slot_entries(seed: Program) -> list[str]:
    """Question nodes the code-side routing (switches) leads to: the nodes a seed writer rewrites."""
    from .population import find_slots

    return find_slots(seed)


def task_info(task: TaskConfig, seed: Program, slots: list[str]) -> str:
    desc = task.label_descriptions
    lines = [task.description] if task.description else []
    lines.append("Labels and their definitions:")
    lines += [f"- {lab}: {desc.get(lab, lab)}" for lab in task.labels]
    fields = [f"`{f}`" + (f" ({d})" if d else "") for f, d in task.inputs.items()]
    lines.append(f"State fields: {', '.join(fields)} (their meaning is described in the paper's instructions).")
    switches = [(nid, n) for nid, n in seed.nodes.items() if isinstance(n, SwitchNode)]
    for nid, n in switches:
        cases = "; ".join(f"`{n.field}` = {v!r} -> node `{t.node or t.label}`" for v, t in n.cases.items())
        lines.append(f"Code routes every input before any question ({nid}): {cases}. Keep this routing; write the question node(s) for each target.")
    if task.ordinal:
        lines.append(f"The labels are ordered: {' < '.join(task.labels)}.")
    return "\n".join(lines)


def prompt(task: TaskConfig, seed: Program, kind: str, n: int) -> str | list[dict[str, Any]]:
    slots = slot_entries(seed)
    skeleton = "\n".join(f"    {s}: {{kind: ..., ...}}   # entry node for `{s}` (labels: {', '.join(seed.outcomes_under(s))})" for s in slots)
    if kind == "decomposed":
        skeleton += f"\n    {slots[0]}_<name>: {{...}}   # optional extra nodes"
    subs = {
        "<task_info>": task_info(task, seed, slots), "<n>": str(n), "<limit>": str(task.optimizer.max_component_chars or 1500),
        "<node_skeleton>": skeleton, "<kind_diff>": " or topologies" if kind == "decomposed" else "",
    }
    subs.update({"<kind_rules>": KIND_RULES[kind], "<curr_param>": seed.to_yaml()})
    text = SEED_TEMPLATE
    for key, value in subs.items():
        text = text.replace(key, value)
    return text


def build(task: TaskConfig, seed: Program, kind: str, item: Any, guard: ProgramGuard) -> Program:
    """A complete, validated seed program from one reply item (paper switches kept, no fitted numbers)."""
    if not isinstance(item, dict) or not isinstance(item.get("nodes"), dict):
        raise ValueError("each program needs a `nodes` mapping")
    slots = slot_entries(seed)
    missing = [s for s in slots if s not in item["nodes"]]
    if missing:
        raise ValueError(f"entry nodes {missing} are missing")
    nodes = {nid: n for nid, n in seed.nodes.items() if nid not in slots}
    for nid, spec in item["nodes"].items():
        nid = str(nid)
        if nid in nodes or not NODE_ID.match(nid) or "__" in nid:
            raise ValueError(f"node id {nid!r} is invalid or collides with a routing node")
        if not isinstance(spec, dict):
            raise ValueError(f"node {nid} must be a mapping")
        spec = {k: v for k, v in spec.items() if k not in ("biases", "readout", "cutoffs", "thresholds")}
        paper = seed.nodes.get(nid) if nid in slots else None
        if isinstance(paper, ScoreNode) and "outcomes" not in spec and len(spec.get("levels") or []) == len(paper.outcomes):
            spec["outcomes"] = [t.model_dump(exclude_none=True) for t in paper.outcomes]  # levels map 1:1 to the paper's outcomes
        if kind == "jev-native" and isinstance(paper, (ChoiceNode, OneVsRestNode)) and isinstance(spec.get("options"), dict):
            for key, opt in spec["options"].items():  # keep the paper's option targets when omitted
                if isinstance(opt, dict) and "to" not in opt and key in paper.options and paper.options[key].to is not None:
                    opt["to"] = paper.options[key].to.model_dump(exclude_none=True)
        nodes[nid] = spec
    data = seed.model_dump(mode="json", by_alias=True, exclude_none=True)
    data.update(nodes=nodes, shared_state=True, temperature=1.0)
    data.pop("demos", None)
    program = Program.model_validate(data)
    for node in program.nodes.values():  # zero-shot Score decoding: round the expected level when levels map 1:1 to outcomes
        if isinstance(node, ScoreNode) and len(node.levels) == len(node.outcomes) > 1:
            node.cutoffs = [i + 0.5 for i in range(len(node.levels) - 1)]
    limit = task.optimizer.max_component_chars
    if limit:
        program = trim_slots(program, limit, set())
    program.check(task)
    if kind == "jev-native":
        for s in slots:
            old, new = seed.nodes[s], program.nodes[s]
            if type(old) is not type(new) or set(program.nodes) - set(seed.nodes):
                raise ValueError(f"jev-native keeps the paper's kinds and nodes ({s} is {old.kind} in the paper)")
            if isinstance(old, (ChoiceNode, OneVsRestNode)) and (list(old.options) != list(new.options) or any(old.options[k].target(k) != new.options[k].target(k) for k in old.options)):
                raise ValueError(f"jev-native keeps the option keys and targets of {s}")
            if isinstance(old, ScoreNode) and (len(old.levels) != len(new.levels) or old.outcomes != new.outcomes):
                raise ValueError(f"jev-native keeps the levels count and outcomes of {s}")
    if any(isinstance(n, StageNode) for n in program.nodes.values()):
        raise ValueError("stage nodes are not allowed in seeds")
    problems = guard.check_program(program, set())
    if problems:
        raise ValueError("; ".join(problems))
    return program


def generate(task: TaskConfig, seed: Program, kind: str, n: int, lm: LM, guard: ProgramGuard) -> tuple[list[tuple[Program, str]], list[str]]:
    """`n` validated seed programs of `kind` (one OMP call, one retry listing the rejections)."""
    from .population import _yaml_items

    request = prompt(task, seed, kind, n)
    out: list[tuple[Program, str]] = []
    rejected: list[str] = []
    for attempt in range(2):
        text, _ = parse_proposal(lm(request))
        items, problems = _yaml_items(text)
        rejected += problems
        for i, item in enumerate(items):
            try:
                program = build(task, seed, kind, item, guard)
            except (ProgramError, ValidationError, ValueError, TypeError, KeyError) as exc:
                rejected.append(f"attempt {attempt + 1} program {i + 1}: {str(exc)[:300]}")
                continue
            if all(program.digest() != p.digest() for p, _ in out):
                out.append((program, str(item.get("rationale") or "").strip()[:300]))
        if len(out) >= n or not isinstance(request, str):
            break
        request = f"{request}\n\nSome programs were rejected: {'; '.join(rejected)[:1500]}\nReply again with {n - len(out)} corrected programs."
    return out[:n], rejected


def seed_file(program: Program, kind: str, index: int, rationale: str, model: str) -> str:
    header = f"# {kind} seed {index} (jev-opt /jev-seeds; writer {model}; zero-shot: written from the task definition only)\n# rationale: {rationale}\n"
    return header + program.to_yaml()


def run_generation(params: dict[str, Any], notify: Callable[[str, dict[str, Any]], None], lm_factory: Callable[[str], Any]) -> dict[str, Any]:
    """Worker method `generate_seeds`: `n` seeds per kind for a task, written to `out_dir` as
    `<kind>-<i>.yaml` plus `generation.json` (and the host's `usage.jsonl`). No Jev call."""
    from .hygiene import load_split
    from .task import load_task, seed_program

    config = load_task(params["task_path"])
    out = Path(params["out_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    kinds = list(params.get("kinds") or KINDS)
    unknown = [k for k in kinds if k not in KINDS]
    if unknown:
        raise ValueError(f"unknown seed kinds {unknown}; known: {list(KINDS)}")
    n = int(params.get("n") or 3)
    models = config.optimizer.model_dump(include={"model", "reflection_model", "structural_model", "thinking"})
    notify("progress", {"run_id": out.name, "run_dir": str(out), "event": "run_started", "optimizer": models, "message": f"writing {n} seeds per kind {kinds}"})
    lm = lm_factory(str(out)).with_role("structural", max_tokens=config.optimizer.population.max_tokens)
    seed = seed_program(config).check(config)
    guard = ProgramGuard(load_split(config, "train") + load_split(config, "val"), seed.to_yaml(), config.optimizer.max_program_chars)
    report: dict[str, Any] = {"task": config.name, "task_path": str(Path(params["task_path"]).resolve()), "kinds": {}, "writer_input": "task/label definitions and the paper seed program; no data examples, no gold labels"}
    for kind in kinds:
        programs, rejected = generate(config, seed, kind, n, lm, guard)
        files = []
        for i, (program, rationale) in enumerate(programs, 1):
            model = "/".join(sorted(lm.models_seen())) or "omp"
            path = out / f"{kind}-{i}.yaml"
            path.write_text(seed_file(program, kind, i, rationale, model))
            files.append({"file": path.name, "rationale": rationale, "digest": program.digest(), "nodes": {nid: n.kind for nid, n in program.nodes.items()}})
        report["kinds"][kind] = {"files": files, "rejected": rejected}
        notify("progress", {"run_id": out.name, "run_dir": str(out), "event": "iteration", "message": f"{kind}: {len(files)} seeds ({len(rejected)} rejected)"})
    report["optimizer_models_seen"] = sorted(lm.models_seen())
    report["optimizer_cost"] = round(lm.total_cost, 6)
    (out / "generation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report
