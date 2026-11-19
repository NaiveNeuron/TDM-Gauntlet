"""Jev decision-program IR (architecture §4): Pydantic models, YAML/JSON round-trip, validation.

A program is data: nodes map an example's state to an outcome. In the main tree
(reachable from `root` without passing through a stage's `decision`) outcomes are
task labels; inside a stage's decision subtree outcomes are free-form values that
the stage writes into a derived state field.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Literal, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_CHOICE_OPTIONS = 255  # Jev Choice limit
SCORE_LEVELS = (2, 10)  # Jev Score limits (inclusive)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Target(_Strict):
    label: str | None = None
    node: str | None = None

    @model_validator(mode="after")
    def _one(self) -> Target:
        if (self.label is None) == (self.node is None):
            raise ValueError("target needs exactly one of `label` or `node`")
        return self


class Option(_Strict):
    criterion: str
    to: Target | None = None  # None: routes to the label named like the option

    def target(self, key: str) -> Target:
        return self.to or Target(label=key)


class Aggregation(_Strict):
    rule: Literal["argmax_normalized", "threshold"] = "argmax_normalized"
    # threshold rule: per-option probability thresholds; argmax_normalized: per-option logit biases
    thresholds: dict[str, float] = Field(default_factory=dict)
    abstain: str | None = None


class Readout(_Strict):
    """Fitted decision head of a question node (§4.2, §7): maps the answers of the question nodes in
    `views` (the node itself first) to one of `classes`, replacing the node's argmax/routing.

    `lr`: multinomial logistic regression, `softmax(W x + b)` over `classes`. `ridge`: linear
    regression onto the ordinal index of `classes` (task label order), mapped to a class by
    `cutoffs`. `x` concatenates per-view features (`execute.view_features`); `W`/`b` already
    include the feature standardization. Every number is fitted on train, never text; an
    unfitted readout (`W` empty) only collects answers.
    """

    kind: Literal["lr", "ridge"]
    views: list[str]
    classes: list[str]
    W: list[list[float]] = Field(default_factory=list)  # lr: one row per class; ridge: one row
    b: list[float] = Field(default_factory=list)
    cutoffs: list[float] | None = None  # ridge: len(classes)-1 ascending on [0, len(classes)-1]
    sigma: float = 1.0  # ridge: residual sd (class-index units) spreading the label distribution
    fit: dict[str, Any] = Field(default_factory=dict)  # record of the CV choice (C/alpha, class weights, folds, score)

    @property
    def fitted(self) -> bool:
        return bool(self.W)


class ChoiceNode(_Strict):
    kind: Literal["choice"]
    instructions: str
    state: list[str]
    options: dict[str, Option]
    biases: dict[str, float] | None = None  # fitted per-option log-prob biases (§7 class calibration); never text
    readout: Readout | None = None  # fitted decision head (§4.2)


class OneVsRestNode(_Strict):
    kind: Literal["one_vs_rest"]
    instructions: str
    state: list[str]
    options: dict[str, Option]  # one Noul per option; targets must be labels
    aggregation: Aggregation = Field(default_factory=Aggregation)
    readout: Readout | None = None


class ScoreNode(_Strict):
    kind: Literal["score"]
    instructions: str
    state: list[str]
    levels: list[str]
    outcomes: list[Target]  # ordered bands over the expected level index
    cutoffs: list[float] | None = None  # len(outcomes)-1 ascending; None = even split
    readout: Readout | None = None


class BranchNode(_Strict):
    kind: Literal["branch"]
    condition: str
    state: list[str]
    if_true: Target
    if_false: Target
    threshold: float = 0.5
    readout: Readout | None = None  # e.g. a decomposed seed whose root Noul heads a question bag (§5.2)


class SwitchNode(_Strict):
    """Deterministic routing on the verbatim value of one state field; issues no Jev question."""

    kind: Literal["switch"]
    field: str
    cases: dict[str, Target]
    default: Target | None = None  # used for values not in `cases`; None: such a value is an error


class StageNode(_Strict):
    kind: Literal["stage"]
    decision: str
    as_: str = Field(alias="as")
    next: str


Node = Annotated[ChoiceNode | OneVsRestNode | ScoreNode | BranchNode | SwitchNode | StageNode, Field(discriminator="kind")]
QUESTION_KINDS = (ChoiceNode, OneVsRestNode, ScoreNode, BranchNode)
DEMOS_FIELD = "examples"  # reserved state field carrying demonstrations (§4.1)
MAX_DEMOS = 64


class Demos(_Strict):
    """Labelled TRAIN examples added to the Jev state of question nodes as `examples` (§4.1).

    `static`: fixed train example ids; `knn`: the `k` nearest train examples to the input under a
    local embedding model, optionally class-balanced. The pool is always the train split.
    """

    mode: Literal["static", "knn"]
    ids: list[str] | None = None  # static only
    k: int = 8  # knn only
    balanced: bool = False  # knn only: round-robin over labels, nearest first within each label
    nodes: list[str] | None = None  # question nodes that receive demos; None = every question node
    max_field_chars: int | None = None  # clip each demo field value (None = verbatim)

    @model_validator(mode="after")
    def _shape(self) -> Demos:
        if self.mode == "static":
            if not self.ids or len(self.ids) > MAX_DEMOS or len(set(self.ids)) != len(self.ids):
                raise ValueError(f"static demos need 1..{MAX_DEMOS} unique ids")
        elif self.ids:
            raise ValueError("knn demos take no ids")
        if not 1 <= self.k <= MAX_DEMOS:
            raise ValueError(f"demos.k must be in 1..{MAX_DEMOS}")
        if self.max_field_chars is not None and self.max_field_chars < 50:
            raise ValueError("demos.max_field_chars must be >= 50")
        return self



class TaskLike(Protocol):
    labels: list[str]
    inputs: dict[str, str]
    ordinal: bool


class ProgramError(ValueError):
    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


class Program(_Strict):
    version: Literal[1] = 1
    labels: list[str]
    inputs: dict[str, str]
    shared_state: bool = False
    temperature: float = 1.0  # fitted on train (§7); never a text component
    root: str
    nodes: dict[str, Node]
    demos: Demos | None = None  # demonstrations in state (§4.1)

    def demo_nodes(self) -> set[str]:
        """Question nodes that receive demonstrations."""
        if self.demos is None:
            return set()
        questions = {nid for nid, n in self.nodes.items() if isinstance(n, QUESTION_KINDS)}
        return questions if self.demos.nodes is None else questions & set(self.demos.nodes)

    # ---- serialization -------------------------------------------------
    def to_data(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)

    def to_yaml(self) -> str:
        return yaml.dump(self.to_data(), Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100)

    def to_json(self) -> str:
        return json.dumps(self.to_data(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_yaml(cls, text: str) -> Program:
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ProgramError(["program YAML must be a mapping"])
        return cls.model_validate(data)

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]

    # ---- graph helpers --------------------------------------------------
    def children(self, node_id: str) -> list[str]:
        """Routing children (excludes a stage's decision subtree)."""
        node = self.nodes[node_id]
        if isinstance(node, StageNode):
            return [node.next]
        return [t.node for t in self.targets(node_id) if t.node is not None]

    def targets(self, node_id: str) -> list[Target]:
        node = self.nodes[node_id]
        readout = getattr(node, "readout", None)
        if readout is not None:  # a readout node is a leaf decision over its classes (§4.2)
            return [Target(label=c) for c in readout.classes]
        if isinstance(node, (ChoiceNode, OneVsRestNode)):
            return [opt.target(key) for key, opt in node.options.items()]
        if isinstance(node, ScoreNode):
            return list(node.outcomes)
        if isinstance(node, BranchNode):
            return [node.if_true, node.if_false]
        if isinstance(node, SwitchNode):
            return [*node.cases.values(), *([node.default] if node.default else [])]
        return []

    def outcomes_under(self, node_id: str) -> list[str]:
        """Distinct outcomes reachable from a node (static; used to spread un-evaluated mass)."""
        seen: list[str] = []
        stack, visited = [node_id], set()
        while stack:
            nid = stack.pop()
            if nid in visited:
                continue
            visited.add(nid)
            node = self.nodes[nid]
            if isinstance(node, OneVsRestNode) and node.aggregation.abstain and node.aggregation.abstain not in seen:
                seen.append(node.aggregation.abstain)
            for t in self.targets(nid):
                if t.label is not None and t.label not in seen:
                    seen.append(t.label)
            stack.extend(reversed(self.children(nid)))
        return seen

    def view_nodes(self) -> set[str]:
        """Question nodes asked only as readout features (views of another node's readout)."""
        out: set[str] = set()
        for nid, node in self.nodes.items():
            readout = getattr(node, "readout", None)
            if readout is not None:
                out.update(v for v in readout.views if v != nid)
        return out

    def decision_nodes(self) -> set[str]:
        """Nodes inside some stage's decision subtree."""
        out: set[str] = set()
        for node in self.nodes.values():
            if isinstance(node, StageNode):
                stack = [node.decision]
                while stack:
                    nid = stack.pop()
                    if nid in out or nid not in self.nodes:
                        continue
                    out.add(nid)
                    stack.extend(self.children(nid))
                    sub = self.nodes[nid]
                    if isinstance(sub, StageNode):
                        stack.append(sub.decision)
        return out

    def text_chars(self) -> int:
        return len(self.to_yaml())

    # ---- validation (§4) -----------------------------------------------
    def check(self, task: TaskLike) -> Program:
        problems: list[str] = []
        if list(self.labels) != list(task.labels):
            problems.append(f"labels {self.labels} must equal task labels {task.labels}")
        if set(self.inputs) != set(task.inputs):
            problems.append(f"inputs {sorted(self.inputs)} must equal task inputs {sorted(task.inputs)}")
        if self.root not in self.nodes:
            problems.append(f"root {self.root!r} is not a node")
            raise ProgramError(problems)
        if not 0.05 <= self.temperature <= 20:
            problems.append("temperature must be in [0.05, 20]")
        if self.demos is not None:
            if DEMOS_FIELD in self.inputs:
                problems.append(f"input field {DEMOS_FIELD!r} is reserved for demonstrations")
            unknown = [n for n in self.demos.nodes or [] if not isinstance(self.nodes.get(n), QUESTION_KINDS)]
            if unknown:
                problems.append(f"demos.nodes {unknown} are not question nodes")
        for nid in self.nodes:
            problems += [f"{nid}: {p}" for p in _node_problems(self, nid, task)]
        if problems:
            raise ProgramError(problems)

        decision = self.decision_nodes()
        main: set[str] = set()
        # DFS over routing + stage edges: acyclicity, reachability, available state fields.
        derived_ok: dict[str, set[str]] = {}

        def visit(nid: str, avail: frozenset[str], path: tuple[str, ...], in_decision: bool) -> None:
            if nid in path:
                problems.append(f"cycle through {' -> '.join(path + (nid,))}")
                return
            if not in_decision:
                main.add(nid)
            node = self.nodes[nid]
            prev = derived_ok.get(nid)
            derived_ok[nid] = set(avail) if prev is None else prev & avail
            if not isinstance(node, StageNode):
                missing = [f for f in ([node.field] if isinstance(node, SwitchNode) else node.state) if f not in avail]
                if missing:
                    problems.append(f"{nid}: state fields {missing} are neither inputs nor derived upstream")
            path = path + (nid,)
            if isinstance(node, StageNode):
                visit(node.decision, avail, path, True)
                visit(node.next, avail | {node.as_}, path, in_decision)
            else:
                for child in self.children(nid):
                    visit(child, avail, path, in_decision)
                readout = getattr(node, "readout", None)
                for view in readout.views[1:] if readout is not None else []:
                    visit(view, avail, path, True)  # feature questions: free-form outcomes, never routed

        visit(self.root, frozenset(self.inputs), (), False)
        unreachable = sorted(set(self.nodes) - set(derived_ok))
        if unreachable:
            problems.append(f"unreachable nodes {unreachable}")
        both = sorted(main & decision)
        if both:
            problems.append(f"nodes {both} are used both as stage decisions and in the main tree")
        routed_views = sorted(self.view_nodes() & (main | decision))
        if routed_views:
            problems.append(f"readout views {routed_views} must not also be routed to")
        for nid in sorted(main):
            outs = [t.label for t in self.targets(nid) if t.label is not None]
            node = self.nodes[nid]
            if isinstance(node, OneVsRestNode) and node.aggregation.abstain:
                outs.append(node.aggregation.abstain)
            bad = [o for o in outs if o not in task.labels]
            if bad:
                problems.append(f"{nid}: outcomes {bad} are not task labels")
        if not problems:
            missing_labels = [lab for lab in task.labels if lab not in self.outcomes_under(self.root)]
            if missing_labels:
                problems.append(f"labels {missing_labels} are unreachable")
        if problems:
            raise ProgramError(problems)
        return self


def _node_problems(p: Program, nid: str, task: TaskLike) -> list[str]:
    node = p.nodes[nid]
    out: list[str] = []
    if "__" in nid:
        out.append("node ids must not contain '__' (reserved for Noul question ids)")
    for t in p.targets(nid):
        if t.node is not None and t.node not in p.nodes:
            out.append(f"target node {t.node!r} does not exist")
    if isinstance(node, StageNode):
        for ref in (node.decision, node.next):
            if ref not in p.nodes:
                out.append(f"stage references missing node {ref!r}")
        if node.as_ in task.inputs:
            out.append(f"derived field {node.as_!r} shadows an input")
        return out
    if isinstance(node, SwitchNode):
        if not node.cases:
            out.append("switch needs at least one case")
        return out
    if not node.state:
        out.append("state must list at least one field")
    if len(set(node.state)) != len(node.state):
        out.append("state fields must be unique")
    if isinstance(node, ChoiceNode):
        if not 2 <= len(node.options) <= MAX_CHOICE_OPTIONS:
            out.append(f"Choice needs 2..{MAX_CHOICE_OPTIONS} options")
        if node.biases and any(k not in node.options for k in node.biases):
            out.append("biases must be keyed by option")
        if node.biases and any(not -10.0 <= v <= 10.0 for v in node.biases.values()):
            out.append("biases must be in [-10, 10]")
    elif isinstance(node, OneVsRestNode):
        if not node.options:
            out.append("one_vs_rest needs at least one option")
        if any(opt.to is not None and opt.to.node is not None for opt in node.options.values()):
            out.append("one_vs_rest options must target labels")
        agg = node.aggregation
        if any(k not in node.options for k in agg.thresholds):
            out.append("thresholds must be keyed by option")
        if agg.rule == "threshold" and any(not 0.0 <= v <= 1.0 for v in agg.thresholds.values()):
            out.append("thresholds must be in [0, 1]")
        if len(node.options) == 1 and not agg.abstain:
            out.append("a single-option one_vs_rest needs an abstain outcome")
    elif isinstance(node, ScoreNode):
        lo, hi = SCORE_LEVELS
        if not lo <= len(node.levels) <= hi:
            out.append(f"Score needs {lo}..{hi} levels")
        if not node.outcomes:
            out.append("score needs at least one outcome")
        if node.cutoffs is not None:
            if len(node.cutoffs) != len(node.outcomes) - 1:
                out.append("cutoffs must have len(outcomes)-1 entries")
            if node.cutoffs != sorted(node.cutoffs):
                out.append("cutoffs must be ascending")
            if any(not 0 <= c <= len(node.levels) - 1 for c in node.cutoffs):
                out.append("cutoffs must lie within the level index range")
    elif isinstance(node, BranchNode):
        if not 0.0 <= node.threshold <= 1.0:
            out.append("threshold must be in [0, 1]")
    readout = getattr(node, "readout", None)
    if readout is not None:
        out += _readout_problems(p, nid, readout, task)
    if nid in p.view_nodes():
        if getattr(node, "readout", None) is not None:
            out.append("a readout view cannot have its own readout")
        if isinstance(node, BranchNode) and (node.if_true.node or node.if_false.node) or isinstance(node, (ChoiceNode, OneVsRestNode)) and any(o.to is not None and o.to.node for o in node.options.values()):
            out.append("a readout view must not route to nodes")
    return out


def _readout_problems(p: Program, nid: str, r: Readout, task: TaskLike) -> list[str]:
    out: list[str] = []
    node = p.nodes[nid]
    if not r.views or r.views[0] != nid:
        out.append("readout.views must start with the node itself")
    if len(set(r.views)) != len(r.views):
        out.append("readout.views must be unique")
    bad = [v for v in r.views if not isinstance(p.nodes.get(v), QUESTION_KINDS)]
    if bad:
        out.append(f"readout.views {bad} are not question nodes")
    if not r.classes or len(set(r.classes)) != len(r.classes):
        out.append("readout.classes must be non-empty and unique")
    if r.kind == "ridge":
        if len(r.classes) < 2:
            out.append("a ridge readout needs >= 2 ordered classes")
        if [c for c in task.labels if c in r.classes] != list(r.classes):
            out.append("ridge readout classes must follow the task label order")
    if isinstance(node, ChoiceNode) and node.biases:
        out.append("biases and readout are exclusive")
    if isinstance(node, (ChoiceNode, OneVsRestNode)) and any(o.to is not None and o.to.node for o in node.options.values()):
        out.append("a readout node must not route to nodes (it decides among readout.classes)")
    if r.fitted:
        rows = len(r.classes) if r.kind == "lr" else 1
        if len(r.W) != rows or len(r.b) != rows or len({len(w) for w in r.W}) != 1:
            out.append(f"readout W/b must have {rows} rows of equal length")
        if r.kind == "ridge":
            cuts = r.cutoffs or []
            if len(cuts) != len(r.classes) - 1 or cuts != sorted(cuts) or any(not 0 <= c <= len(r.classes) - 1 for c in cuts):
                out.append("ridge readout needs len(classes)-1 ascending cutoffs within the class index range")
    if not r.sigma > 0:
        out.append("readout.sigma must be positive")
    return out


def _slot_setters(p: Program) -> dict[str, tuple[Any, str, Any]]:
    """Text slot name -> (container, attribute-or-key, current value); order is stable."""
    out: dict[str, tuple[Any, str, Any]] = {f"inputs.{f}": (p.inputs, f, d) for f, d in p.inputs.items()}
    for nid, node in p.nodes.items():
        if isinstance(node, (ChoiceNode, OneVsRestNode, ScoreNode)):
            out[f"nodes.{nid}.instructions"] = (node, "instructions", node.instructions)
        if isinstance(node, (ChoiceNode, OneVsRestNode)):
            for key, opt in node.options.items():
                out[f"nodes.{nid}.options.{key}.criterion"] = (opt, "criterion", opt.criterion)
        if isinstance(node, ScoreNode):
            for i, level in enumerate(node.levels):
                out[f"nodes.{nid}.levels.{i}"] = (node.levels, i, level)
        if isinstance(node, BranchNode):
            out[f"nodes.{nid}.condition"] = (node, "condition", node.condition)
    return out


def text_slots(p: Program) -> dict[str, str]:
    """GEPA components for jev-text-opt (§5): every free-text field of the program."""
    return {name: value for name, (_, _, value) in _slot_setters(p).items()}


def apply_slots(p: Program, values: dict[str, str]) -> Program:
    """Copy of `p` with text slots replaced; topology is untouched."""
    q = p.model_copy(deep=True)
    setters = _slot_setters(q)
    unknown = sorted(set(values) - set(setters))
    if unknown:
        raise ProgramError([f"unknown text slots {unknown}"])
    for name, value in values.items():
        container, key, _ = setters[name]
        if isinstance(container, (dict, list)):
            container[key] = value
        else:
            setattr(container, key, value)
    return q


def default_cutoffs(node: ScoreNode) -> list[float]:
    """Even split of the level index range [0, L-1] into len(outcomes) bands."""
    span = len(node.levels) - 1
    k = len(node.outcomes)
    return [span * (i + 1) / k for i in range(k - 1)]


class Ensemble(_Strict):
    """Representative `ensemble-k` (§10): mean of the members' label distributions, then fitted
    per-label log-prob biases and temperature, argmax. Members are complete (fitted) programs."""

    kind: Literal["ensemble"] = "ensemble"
    labels: list[str]
    members: list[Program]
    label_bias: dict[str, float] = Field(default_factory=dict)
    temperature: float = 1.0
    # ordinal tasks: mean of the members' predicted label indices, banded by these cutoffs
    # (argmax of a mean distribution ignores the members' fitted Score cutoffs)
    cutoffs: list[float] | None = None

    @model_validator(mode="after")
    def _members(self) -> Ensemble:
        if len(self.members) < 2:
            raise ValueError("an ensemble needs >= 2 members")
        if any(list(m.labels) != list(self.labels) for m in self.members):
            raise ValueError("ensemble members must share the ensemble's labels")
        if any(k not in self.labels for k in self.label_bias):
            raise ValueError("label_bias must be keyed by label")
        if self.cutoffs is not None and (len(self.cutoffs) != len(self.labels) - 1 or self.cutoffs != sorted(self.cutoffs)):
            raise ValueError("ensemble cutoffs must be len(labels)-1 ascending values")
        return self

    def to_yaml(self) -> str:
        data = self.model_dump(mode="json", by_alias=True, exclude_none=True)
        return yaml.dump(data, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100)

    def digest(self) -> str:
        blob = json.dumps(self.model_dump(mode="json", by_alias=True, exclude_none=True), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_decider(text: str) -> Program | Ensemble:
    """A representative file: a Program, or an Ensemble (`kind: ensemble`)."""
    data = yaml.safe_load(text)
    if isinstance(data, dict) and data.get("kind") == "ensemble":
        return Ensemble.model_validate(data)
    return Program.from_yaml(text)


class _Dumper(yaml.SafeDumper):
    pass


def _str_presenter(dumper: yaml.SafeDumper, data: str) -> yaml.Node:
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_Dumper.add_representer(str, _str_presenter)
