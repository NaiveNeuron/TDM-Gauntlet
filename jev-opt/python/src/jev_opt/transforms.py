"""Structural operators on programs (architecture §6) and the structural proposal prompt.

Operators are pure `Program -> Program` functions raising `TransformError` when not
applicable; callers validate the result with `Program.check(task)`. LLM prompts are
rendered and parsed by GEPA's own `InstructionProposalSignature`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from gepa.strategies.instruction_proposal import InstructionProposalSignature

from .ir import (
    QUESTION_KINDS,
    Aggregation,
    BranchNode,
    ChoiceNode,
    Demos,
    OneVsRestNode,
    Option,
    Program,
    ProgramError,
    ScoreNode,
    Target,
    apply_slots,
    text_slots,
)
from .task import TaskConfig


class TransformError(ValueError):
    pass


def _copy(p: Program) -> Program:
    return p.model_copy(deep=True)


def _node(p: Program, nid: str, kind: type) -> Any:
    node = p.nodes.get(nid)
    if not isinstance(node, kind):
        raise TransformError(f"{nid} is not a {kind.__name__}")
    return node


def _label_options(node: ChoiceNode | OneVsRestNode) -> dict[str, str]:
    """option key -> label, requiring every option to target a label."""
    out = {}
    for key, opt in node.options.items():
        t = opt.target(key)
        if t.label is None:
            raise TransformError("all options must target labels")
        out[key] = t.label
    return out


def choice_to_one_vs_rest(p: Program, nid: str) -> Program:
    node = _node(p, nid, ChoiceNode)
    _label_options(node)
    q = _copy(p)
    q.nodes[nid] = OneVsRestNode(kind="one_vs_rest", instructions=node.instructions, state=list(node.state), options=node.model_copy(deep=True).options)
    return q


def one_vs_rest_to_choice(p: Program, nid: str) -> Program:
    node = _node(p, nid, OneVsRestNode)
    options = node.model_copy(deep=True).options
    abstain = node.aggregation.abstain
    if abstain and abstain not in _label_options(node).values():
        options[abstain] = Option(criterion="None of the other options applies.")
    if len(options) < 2:
        raise TransformError("a Choice needs >= 2 options")
    q = _copy(p)
    q.nodes[nid] = ChoiceNode(kind="choice", instructions=node.instructions, state=list(node.state), options=options)
    return q


def choice_to_score(p: Program, nid: str, task: TaskConfig) -> Program:
    if not task.ordinal:
        raise TransformError("choice_to_score requires an ordinal task")
    node = _node(p, nid, ChoiceNode)
    labels = _label_options(node)
    order = sorted(node.options, key=lambda k: task.labels.index(labels[k]) if labels[k] in task.labels else 10**6)
    if not 2 <= len(order) <= 10:
        raise TransformError("Score supports 2..10 levels")
    q = _copy(p)
    q.nodes[nid] = ScoreNode(
        kind="score",
        instructions=node.instructions,
        state=list(node.state),
        levels=[node.options[k].criterion for k in order],
        outcomes=[Target(label=labels[k]) for k in order],
        cutoffs=[i + 0.5 for i in range(len(order) - 1)],
    )
    return q


def score_to_choice(p: Program, nid: str) -> Program:
    node = _node(p, nid, ScoreNode)
    if len(node.levels) != len(node.outcomes) or any(t.label is None for t in node.outcomes):
        raise TransformError("score_to_choice needs one label outcome per level")
    q = _copy(p)
    options = {t.label: Option(criterion=level) for level, t in zip(node.levels, node.outcomes)}
    q.nodes[nid] = ChoiceNode(kind="choice", instructions=node.instructions, state=list(node.state), options=options)
    return q


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "group"


def hierarchize(p: Program, nid: str, groups: dict[str, list[str]]) -> Program:
    """Flat Choice -> coarse Choice over groups -> fine Choices (singleton groups route directly)."""
    node = _node(p, nid, ChoiceNode)
    labels = _label_options(node)
    by_label = {lab: key for key, lab in labels.items()}
    if sorted(lab for g in groups.values() for lab in g) != sorted(labels.values()) or len(groups) < 2:
        raise TransformError("groups must partition the node's labels into >= 2 groups")
    if all(len(g) == 1 for g in groups.values()):
        raise TransformError("hierarchize needs at least one multi-label group")
    q = _copy(p)
    coarse: dict[str, Option] = {}
    for group, members in groups.items():
        keys = [by_label[lab] for lab in members]
        if len(keys) == 1:
            coarse[keys[0]] = node.options[keys[0]].model_copy(deep=True)
            continue
        child = f"{nid}_{_slug(group)}"
        if child in q.nodes:
            raise TransformError(f"node id {child} already exists")
        q.nodes[child] = ChoiceNode(kind="choice", instructions=node.instructions, state=list(node.state), options={k: node.options[k].model_copy(deep=True) for k in keys})
        joined = " | ".join(node.options[k].criterion for k in keys)
        coarse[group] = Option(criterion=f"{group}: one of — {joined}", to=Target(node=child))
    q.nodes[nid] = ChoiceNode(kind="choice", instructions=node.instructions, state=list(node.state), options=coarse)
    return q


def flatten(p: Program, nid: str) -> Program:
    """Inverse of `hierarchize`: inline child Choices that only this node references."""
    node = _node(p, nid, ChoiceNode)
    refs = {n: sum(n in p.children(o) for o in p.nodes) + (p.root == n) for n in p.nodes}
    q = _copy(p)
    options: dict[str, Option] = {}
    inlined = False
    for key, opt in node.options.items():
        t = opt.target(key)
        child = p.nodes.get(t.node) if t.node else None
        if isinstance(child, ChoiceNode) and refs[t.node] == 1 and all(o.target(k).label for k, o in child.options.items()):
            for ck, copt in child.options.items():
                if ck in options:
                    raise TransformError(f"option {ck} would collide")
                options[ck] = copt.model_copy(deep=True)
            del q.nodes[t.node]
            inlined = True
        else:
            options[key] = opt.model_copy(deep=True)
    if not inlined:
        raise TransformError("nothing to flatten")
    q.nodes[nid] = ChoiceNode(kind="choice", instructions=node.instructions, state=list(node.state), options=options)
    return q


def toggle_shared_state(p: Program) -> Program:
    q = _copy(p)
    q.shared_state = not p.shared_state
    return q


def set_abstain(p: Program, nid: str, label: str | None) -> Program:
    """Threshold aggregation with `label` as the fallback; its own Noul (if any) is dropped."""
    node = _node(p, nid, OneVsRestNode)
    q = _copy(p)
    new = q.nodes[nid]
    if label is not None:
        new.options = {k: o for k, o in new.options.items() if o.target(k).label != label}
        if not new.options:
            raise TransformError("abstain label cannot be the only option")
    new.aggregation = Aggregation(rule="threshold", abstain=label)
    if node.aggregation == new.aggregation and len(node.options) == len(new.options):
        raise TransformError("no change")
    return q


def set_aggregation(p: Program, nid: str, rule: str) -> Program:
    node = _node(p, nid, OneVsRestNode)
    if node.aggregation.rule == rule:
        raise TransformError("no change")
    q = _copy(p)
    q.nodes[nid].aggregation = Aggregation(rule=rule, abstain=node.aggregation.abstain if rule == "threshold" else None)
    return q


def narrow_state(p: Program, nid: str, field: str) -> Program:
    node = p.nodes.get(nid)
    if not isinstance(node, QUESTION_KINDS) or field not in node.state or len(node.state) < 2:
        raise TransformError("cannot narrow")
    q = _copy(p)
    q.nodes[nid].state = [f for f in node.state if f != field]
    return q


def widen_state(p: Program, nid: str, field: str) -> Program:
    node = p.nodes.get(nid)
    if not isinstance(node, QUESTION_KINDS) or field in node.state:
        raise TransformError("cannot widen")
    q = _copy(p)
    q.nodes[nid].state = [*node.state, field]
    return q


def set_demos(p: Program, demos: Demos | None) -> Program:
    if p.demos == demos:
        raise TransformError("no change")
    q = _copy(p)
    q.demos = demos.model_copy(deep=True) if demos is not None else None
    return q


def demo_moves(p: Program) -> list[tuple[str, Demos | None, str]]:
    """(name, new demos spec, rationale) for the deterministic demonstration operators."""
    d = p.demos
    if d is None:
        return [("demos_knn:8:balanced", Demos(mode="knn", k=8, balanced=True), "add the 8 nearest train examples (class-balanced) to the state")]
    out: list[tuple[str, Demos | None, str]] = [("demos_off", None, "drop demonstrations")]
    if d.mode == "knn":
        for k in sorted({max(1, d.k // 2), min(32, d.k * 2)} - {d.k}):
            out.append((f"demos_k:{k}", d.model_copy(update={"k": k}), f"show {k} nearest train examples instead of {d.k}"))
        out.append((f"demos_balanced:{not d.balanced}", d.model_copy(update={"balanced": not d.balanced}), "toggle class balance of the nearest train examples"))
    return out


@dataclass(frozen=True)
class Move:
    name: str
    program: Program
    rationale: str


def moves(p: Program, task: TaskConfig) -> list[Move]:
    """Every applicable deterministic operator, validated, in a fixed order."""
    candidates: list[tuple[str, Any, str]] = []
    for nid, node in p.nodes.items():
        if isinstance(node, ChoiceNode):
            candidates.append((f"choice_to_one_vs_rest:{nid}", lambda n=nid: choice_to_one_vs_rest(p, n), f"ask one Noul per label at {nid} and aggregate"))
            if task.label_groups:
                candidates.append((f"hierarchize:{nid}", lambda n=nid: hierarchize(p, n, task.label_groups), f"route {nid} coarse-to-fine via task.label_groups"))
            candidates.append((f"flatten:{nid}", lambda n=nid: flatten(p, n), f"merge fine choices under {nid} into one Choice"))
            if task.ordinal:
                candidates.append((f"choice_to_score:{nid}", lambda n=nid: choice_to_score(p, n, task), f"treat {nid} as an ordinal Score with fitted cutoffs"))
        elif isinstance(node, OneVsRestNode):
            candidates.append((f"one_vs_rest_to_choice:{nid}", lambda n=nid: one_vs_rest_to_choice(p, n), f"replace the Nouls at {nid} with one Choice"))
            other = "threshold" if node.aggregation.rule == "argmax_normalized" else "argmax_normalized"
            candidates.append((f"set_aggregation:{nid}:{other}", lambda n=nid, r=other: set_aggregation(p, n, r), f"aggregate Nouls at {nid} by {other} (fitted)"))
            for key in list(node.options)[:3]:
                label = node.options[key].target(key).label
                candidates.append((f"set_abstain:{nid}:{label}", lambda n=nid, lab=label: set_abstain(p, n, lab), f"make {label} the fallback when no other Noul at {nid} fires"))
        elif isinstance(node, ScoreNode):
            candidates.append((f"score_to_choice:{nid}", lambda n=nid: score_to_choice(p, n), f"replace the Score at {nid} by a Choice"))
        if isinstance(node, (ChoiceNode, OneVsRestNode, ScoreNode, BranchNode)):
            for f in node.state:
                candidates.append((f"narrow_state:{nid}:{f}", lambda n=nid, x=f: narrow_state(p, n, x), f"drop state field {f} at {nid}"))
            for f in p.inputs:
                candidates.append((f"widen_state:{nid}:{f}", lambda n=nid, x=f: widen_state(p, n, x), f"add state field {f} at {nid}"))
    if sum(isinstance(n, QUESTION_KINDS) for n in p.nodes.values()) > 1:
        candidates.append(("toggle_shared_state", lambda: toggle_shared_state(p), "toggle batching of same-state questions into one request"))
    for name, spec, why in demo_moves(p):
        candidates.append((name, lambda s=spec: set_demos(p, s), why))
    out: list[Move] = []
    for name, make, why in candidates:
        try:
            out.append(Move(name, make().check(task), why))
        except (TransformError, ProgramError, ValueError):
            continue
    return out


# ---- LLM proposal prompts (rendered/parsed by GEPA's InstructionProposalSignature) ----

IR_GUIDE = """Program schema (YAML): version: 1; labels; inputs (state field -> description); shared_state (bool);
temperature (fitted, leave as is); root (node id); nodes: {id: node}. Node kinds:
- choice: {instructions, state: [fields], options: {key: {criterion, to?: {label: L} | {node: id}}}} (Jev Choice; argmax routing; `to` defaults to the label named like the option)
- one_vs_rest: {instructions, state, options: {label: {criterion}}, aggregation: {rule: argmax_normalized|threshold, thresholds: {}, abstain?: label}} (one Jev Noul per option; criterion is a statement judged true/false)
- score: {instructions, state, levels: [2..10 descriptions, ordered], outcomes: [targets, ordered bands], cutoffs?: [...]} (Jev Score)
- branch: {condition (a true/false statement), state, if_true: target, if_false: target, threshold: 0.5} (Jev Noul)
- switch: {field, cases: {value: target}, default?: target} (no Jev call: routes on the exact value of a state field)
- stage: {decision: node id, as: new_field, next: node id} (evaluates `decision`, whose option targets are free-form values
  like {label: yes}, writes the decided value into state field `as`, then runs `next` with that field available)
Rules: every task label must be reachable; outcomes outside stage decisions must be task labels; node ids must not contain '__';
state fields must be inputs or fields derived by an upstream stage; numeric thresholds/cutoffs/biases are fit automatically.
Optional top-level `demos` ({mode: knn, k, balanced} or {mode: static, ids}) adds labelled TRAIN examples to Jev's state as
`examples`; leave it unchanged (dedicated operators edit it).
Never copy example text or ids into the program: describe patterns instead."""

STRUCTURAL_TEMPLATE = """You are improving a Jev decision program: a small graph of calibrated classifier questions (Choice, Noul, Score) that maps an input state to one task label.

Current program:
```
<curr_param>
```

""" + IR_GUIDE + """

Evidence from TRAIN examples (overall metrics, confusions, failures):
```
<side_info>
```

Propose ONE structural change likely to fix the dominant errors, for example: decompose into a stage that first extracts an intermediate property into derived state, route confusable labels through a coarse-to-fine hierarchy, add a branch that isolates a failure mode, or switch a Choice to one-vs-rest Nouls. You may also improve wording.

Reply with one line `Rationale: <why this helps>` and then the complete new program YAML inside ``` blocks."""

TEXT_EDIT_TEMPLATE = """I provided an assistant (Jev, a calibrated classifier) with the following decision program, written in YAML:
```
<curr_param>
```

""" + IR_GUIDE + """

The following are summaries and examples of inputs, the program's decisions, and feedback:
```
<side_info>
```

Rewrite only the text fields (instructions, criterion, levels, condition, input descriptions) to fix the errors. Keep ids, kinds, options, targets, state lists and numbers unchanged. Include niche domain facts inferred from the feedback.

Reply with one line `Rationale: <what you changed and why>` and then the complete program YAML inside ``` blocks."""


def length_rule(limit: int) -> str:
    return f"\n\nLength limit: every text field (instructions, criterion, level, condition, input description) must be at most {limit} characters. Prefer concise, general rules over lists of special cases."


def slot_length_problems(p: Program, limit: int, allowed: set[str]) -> list[str]:
    """Text slots longer than `limit`, except values in `allowed` (e.g. verbatim seed text)."""
    return [f"text field {name} has {len(v)} chars (limit {limit})" for name, v in text_slots(p).items() if len(v) > limit and v not in allowed]


def trim_text(text: str, limit: int) -> str:
    """Cut to at most `limit` chars, at the last sentence end when one exists in the second half."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind(".\n"), cut.rfind("; "))
    return cut[: end + 1].rstrip() if end >= limit // 2 else cut.rstrip()


def trim_slots(p: Program, limit: int, allowed: set[str]) -> Program:
    long = {name: trim_text(v, limit) for name, v in text_slots(p).items() if len(v) > limit and v not in allowed}
    return apply_slots(p, long) if long else p


def render_prompt(template: str, program_yaml: str, records: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    return InstructionProposalSignature.prompt_renderer(
        {"current_instruction_doc": program_yaml, "dataset_with_feedback": records, "prompt_template": template}
    )


def parse_proposal(lm_out: str) -> tuple[str, str]:
    """(program YAML, one-line rationale) from an LLM reply."""
    text = InstructionProposalSignature.output_extractor(lm_out)["new_instruction"]
    match = re.search(r"^\s*rationale\s*:\s*(.+)$", lm_out, flags=re.I | re.M)
    return text, (match.group(1).strip()[:300] if match else "")


def structure_signature(p: Program) -> str:
    """Program with all text slots blanked: equal signatures = same topology and numbers."""
    return apply_slots(p, {k: "" for k in text_slots(p)}).to_json()
