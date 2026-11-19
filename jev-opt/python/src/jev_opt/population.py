"""`jev-population-opt` (architecture §5.1): batched population search over question variants.

The program's question nodes that are reached only through deterministic routing (`switch`) are
"slots". Per round, one OMP call proposes `population.size` radically diverse variants of one
slot. A variant is a small bag of questions: a main question over the slot's labels (Choice,
one-vs-rest Nouls, or Score for ordinal tasks, any state field selection, optionally with
demonstrations) plus up to `max_aux` feature questions, decided by a readout (§4.2) fitted on the
answers. All variants of a round are asked for every TRAIN fit example in one packed request per
example and state: Jev answers each question of a request independently of the others, and the
per-(state, question) cache (§9) means no answer is ever bought twice. Each variant is scored by
the cross-validated readout on the fit examples (out-of-fold predictions); since slots partition
the examples, every combination of per-slot variants is scored exactly offline from those
predictions. The proposer then sees the elite, per-class errors and the worst out-of-fold
examples. At the end the best programs by train CV, and forward-selected stacks of their
variants under one readout (views), are scored on val; val selects the representatives.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml
from pydantic import TypeAdapter, ValidationError

from .demos import DemoPool
from .execute import run_examples
from .fit import CVSpec, fit_readout, readout_design, readout_kind
from .hygiene import ProgramGuard
from .ir import QUESTION_KINDS, BranchNode, ChoiceNode, Demos, Node, OneVsRestNode, Option, Program, ProgramError, Readout, ScoreNode, SwitchNode, Target
from .jev import JevClient
from .metrics import primary
from .task import Example, TaskConfig
from .transforms import parse_proposal, render_prompt, trim_text

LM = Callable[[str | list[dict[str, Any]]], str]
FIELD_CLIP = 600
MAX_COMBOS = 20000  # exhaustive combination scoring up to this many; coordinate ascent beyond
AUX_ID = re.compile(r"^[a-z][a-z0-9_]{0,23}$")

POPULATION_TEMPLATE = """You are designing questions for Jev, a calibrated classifier (TypeSafe System One). Jev reads a JSON state and answers typed questions about it: `choice` (one option out of several; returns a probability for every option), `noul` (the probability that a statement is true) and `score` (a probability for each of 2-10 ordered levels). Every question in a request is answered independently of the others. The answers of a variant's questions feed a linear head fitted on labelled TRAIN data, so a question helps when its probabilities separate the labels, even if its top answer is not the label.

TASK
<task_info>

The current best variants of node `<slot>` (train cross-validated <metric>, higher is better):
```
<curr_param>
```

Evidence from TRAIN examples under the best variant (out-of-fold: per-class errors, confusions, worst examples):
```
<side_info>
```

Write <size> NEW variants of node `<slot>` that are RADICALLY DIFFERENT from each other and from the variants above. Spread them over, and combine:
- reframed instructions: annotation-guideline style, a decision procedure, contrastive definitions of the confused labels, the question an expert annotator asks, abstract prototypes of each label;
- rewritten or renamed option criteria (the option keys stay the label names);
- other primitives for the main question: `choice` over the labels, `one_vs_rest` (one noul per label; each criterion is a statement that is true for that label)<score_note>;
- other `state` field selections (fields: <fields>);
- up to <max_aux> auxiliary questions (`aux`) measuring cues that separate the confused labels: `branch` (a true/false statement, as `condition`), `choice` over your own options, or `score` over your own levels. Their answers only feed the head, so name observable properties of the input, not labels.<demo_note>
At most 2 variants may be light edits of a variant above. Every text field must be at most <limit> characters; prefer compact wording. Never copy example text or ids: describe patterns.

Reply with one YAML list inside a single ``` block and nothing else in code blocks; one item per variant:
```
- rationale: <one line: what is different about this variant>
  main:
    kind: choice              # or one_vs_rest<score_kind>
    instructions: <text>
    state: [<fields>]
    options:                  # choice / one_vs_rest: exactly these keys: <labels>
      <label>: {criterion: <text>}
    # score instead: levels: [<lowest>, ..., <highest>]  (2-10 ordered descriptions)
  aux:                        # optional, at most <max_aux>; ids: lowercase letters, digits, underscores
    <id>: {kind: branch, condition: <statement>, state: [<fields>]}
  demos: false               # optional
```"""


@dataclass
class Variant:
    """One slot variant: questions (`main` + aux ids) and its cross-validated evidence."""

    slot: str
    nodes: dict[str, Node]
    demos: bool
    rationale: str
    round: int
    cv: float | None = None
    oof: dict[str, tuple[str, dict[str, float]]] = field(default_factory=dict, repr=False)
    fit: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    origin: str = ""  # seed name ("paper", "paper+demos", "jev-native-2", ...) or "r<round>" for proposals
    root: str = "paper"  # seed kind this variant descends from (proposals: the kind of the elite leader shown)

    @property
    def key(self) -> str:
        blob = json.dumps({"nodes": {k: n.model_dump(mode="json", by_alias=True, exclude_none=True) for k, n in self.nodes.items()}, "demos": self.demos}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def spec(self) -> dict[str, Any]:
        out: dict[str, Any] = {"rationale": self.rationale, "main": node_spec(self.nodes["main"])}
        aux = {k: node_spec(n) for k, n in self.nodes.items() if k != "main"}
        if aux:
            out["aux"] = aux
        if self.demos:
            out["demos"] = True
        return out

    def to_record(self) -> dict[str, Any]:
        return {"slot": self.slot, "key": self.key, "round": self.round, "origin": self.origin, "root": self.root, "rationale": self.rationale, "demos": self.demos, "cv": self.cv,
                "fit": self.fit, "error": self.error, "nodes": {k: n.model_dump(mode="json", by_alias=True, exclude_none=True) for k, n in self.nodes.items()}}


def node_spec(node: Node) -> dict[str, Any]:
    """LLM-facing form of a question node (no fitted numbers, no routing)."""
    if isinstance(node, (ChoiceNode, OneVsRestNode)):
        return {"kind": node.kind, "instructions": node.instructions, "state": list(node.state), "options": {k: {"criterion": o.criterion} for k, o in node.options.items()}}
    if isinstance(node, ScoreNode):
        return {"kind": "score", "instructions": node.instructions, "state": list(node.state), "levels": list(node.levels)}
    if isinstance(node, BranchNode):
        return {"kind": "branch", "condition": node.condition, "state": list(node.state)}
    raise TypeError(node.kind)


def _node_from_record(data: dict[str, Any]) -> Node:
    return TypeAdapter(Node).validate_python(data)


def find_slots(program: Program) -> list[str]:
    """Question nodes reached from the root through switch nodes only (each a leaf decision)."""
    out: list[str] = []
    stack, seen = [program.root], set()
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        node = program.nodes[nid]
        if isinstance(node, SwitchNode):
            stack.extend(reversed([t.node for t in program.targets(nid) if t.node is not None]))
        elif isinstance(node, QUESTION_KINDS) and not isinstance(node, BranchNode) and all(t.label is not None for t in program.targets(nid)):
            out.append(nid)
        else:
            raise ValueError(f"population search needs switches over leaf choice/one_vs_rest/score nodes; {nid} ({node.kind}) is not one")
    return out


def route(program: Program, fields: dict[str, str]) -> str | None:
    """The slot an example reaches (None: the switches decide a label directly)."""
    nid = program.root
    while isinstance(program.nodes[nid], SwitchNode):
        node = program.nodes[nid]
        target = node.cases.get(fields[node.field], node.default)
        if target is None or target.node is None:
            return None
        nid = target.node
    return nid


def install(base: Program, slot: str, members: list[Variant], kind: str, classes: list[str], demo_spec: Demos | None) -> Program:
    """`base` with node `slot` replaced by `members[0]`'s main question under one unfitted readout
    whose views are the questions of every member (more than one member = a stacked readout)."""
    q = base.model_copy(deep=True)
    demo_nodes = base.demo_nodes() - {slot}
    views: list[str] = []
    for j, v in enumerate(members):
        prefix = slot if j == 0 else f"{slot}_x{j}"
        for local, node in sorted(v.nodes.items(), key=lambda kv: kv[0] != "main"):
            nid = prefix if local == "main" else f"{prefix}_a_{local}"
            if nid != slot and nid in q.nodes:
                raise ValueError(f"variant node id {nid} collides with an existing node")
            q.nodes[nid] = node.model_copy(deep=True)
            views.append(nid)
            if v.demos:
                demo_nodes.add(nid)
    q.nodes[slot].readout = Readout(kind=kind, views=views, classes=list(classes))
    spec = base.demos or demo_spec
    q.demos = spec.model_copy(update={"nodes": sorted(demo_nodes)}) if demo_nodes and spec is not None else None
    return q


class PopulationSearch:
    def __init__(self, task: TaskConfig, seed: Program, client: JevClient, fit_set: list[Example], guard: ProgramGuard, train_ids: set[str], pool: DemoPool | None, lm: LM | None, condition: str, notify: Callable[[str], None] | None = None, extra_seeds: list[tuple[str, str, Program]] | None = None):
        self.task = task
        self.cfg = task.optimizer.population
        self.base = seed.model_copy(deep=True)
        self.client = client
        self.fit_set = fit_set
        self.guard = guard
        self.train_ids = train_ids
        self.pool = pool
        self.lm = lm
        self.condition = condition
        self.notify = notify or (lambda message: None)
        self.slots = find_slots(seed)
        self.kind = readout_kind(task.primary_metric)
        self.classes = {s: [lab for lab in seed.labels if lab in seed.outcomes_under(s)] for s in self.slots}
        self.cv: CVSpec = CVSpec(folds=task.optimizer.readout_folds, seed=task.optimizer.seed, group=task.optimizer.readout_group)
        self.routes = {ex.id: route(seed, ex.fields) for ex in fit_set}
        self.variants: dict[str, list[Variant]] = {s: [] for s in self.slots}
        self.proposals: list[dict[str, Any]] = []
        self.extra_seeds = list(extra_seeds or [])  # (name, kind, program): zero-shot seeds written from the task definition
        self.seed_problems: list[str] = []
        self.seed_texts = {v for p in [seed, *(q for _, _, q in self.extra_seeds)] for n in p.nodes.values() for v in _texts(n)}

    # ---- variants ------------------------------------------------------
    def seed_variants(self) -> list[Variant]:
        """Round 0: the paper seed's questions (with and without demonstrations) and every extra
        seed's question subtree per slot (§5.2), each a variant decided by a readout."""
        out = []
        for s in self.slots:
            node = _plain(self.base.nodes[s])
            out.append(Variant(s, {"main": node}, False, "paper seed question", 0, origin="paper", root="paper"))
            if self.cfg.demos is not None:
                out.append(Variant(s, {"main": node.model_copy(deep=True)}, True, "paper seed question with demonstrations", 0, origin="paper+demos", root="paper"))
        for name, kind, program in self.extra_seeds:
            for s in self.slots:
                ex = next((e for e in self.fit_set if self.routes.get(e.id) == s), None)
                entry = route(program, ex.fields) if ex is not None else None
                if entry is None:
                    continue
                try:
                    v = Variant(s, fragment(program, entry), False, f"{name} seed questions", 0, origin=name, root=kind)
                    why = self.validate(v)
                except ValueError as exc:
                    why = str(exc)
                if why:  # still evaluated zero-shot; it just cannot start the search
                    self.seed_problems.append(f"{name} -> {s}: {why}"[:300])
                    continue
                out.append(v)
        return out

    def program(self, slot: str, members: list[Variant], base: Program | None = None) -> Program:
        return install(base or self.base, slot, members, self.kind, self.classes[slot], self.cfg.demos)

    def combo_program(self, choice: dict[str, list[Variant]]) -> Program:
        """Program with every slot replaced by its (possibly stacked) variants."""
        program = self.base
        for s in self.slots:
            program = self.program(s, choice[s], program)
        return program.check(self.task)

    def validate(self, v: Variant) -> str | None:
        try:
            program = self.program(v.slot, [v]).check(self.task)
        except (ProgramError, ValidationError, ValueError) as exc:
            return str(exc)[:300]
        problems = self.guard.check_program(program, self.train_ids)
        return "; ".join(problems)[:300] if problems else None

    def parse(self, text: str, slot: str, round_: int) -> tuple[list[Variant], list[str]]:
        items, problems = _yaml_items(text)
        if not items:
            return [], problems or ["reply must be a YAML list of variants"]
        known = {v.key for v in self.variants[slot]}
        out: list[Variant] = []
        for i, item in enumerate(items):
            try:
                v = self._variant(item, slot, round_)
                why = self.validate(v)
                if why:
                    raise ValueError(why)
                if v.key in known:
                    raise ValueError("duplicate of an earlier variant")
                known.add(v.key)
                out.append(v)
            except (ValidationError, ValueError, KeyError, TypeError, AttributeError) as exc:
                problems.append(f"variant {i + 1}: {str(exc)[:200]}")
        return out, problems

    def _variant(self, item: Any, slot: str, round_: int) -> Variant:
        if not isinstance(item, dict) or not isinstance(item.get("main"), dict):
            raise ValueError("each variant needs a `main` question mapping")
        trims: list[str] = []
        default_state = list(self.base.nodes[slot].state)
        nodes: dict[str, Node] = {"main": self._node(item["main"], self.classes[slot], default_state, trims)}
        aux = item.get("aux") or {}
        if not isinstance(aux, dict):
            raise ValueError("`aux` must be a mapping of id -> question")
        if len(aux) > self.cfg.max_aux:
            raise ValueError(f"at most {self.cfg.max_aux} aux questions")
        for local, spec in aux.items():
            local = str(local)
            if not AUX_ID.match(local) or "__" in local or local == "main":
                raise ValueError(f"aux id {local!r} must match {AUX_ID.pattern} without '__'")
            nodes[local] = self._node(spec, None, default_state, trims)
        demos = bool(item.get("demos", False))
        if demos and self.cfg.demos is None and self.base.demos is None:
            raise ValueError("demonstrations are not available for this task")
        rationale = str(item.get("rationale") or "").strip()[:300] + (f" (trimmed: {', '.join(trims)})" if trims else "")
        return Variant(slot, nodes, demos, rationale or "LLM variant", round_, origin=f"r{round_}")

    def _node(self, spec: Any, classes: list[str] | None, default_state: list[str], trims: list[str]) -> Node:
        if not isinstance(spec, dict):
            raise ValueError("a question must be a mapping")
        limit = self.task.optimizer.max_component_chars
        kind = spec.get("kind")
        state = spec.get("state") or default_state
        if not isinstance(state, list) or not all(isinstance(f, str) and f in self.task.inputs for f in state):
            raise ValueError(f"state must list input fields from {sorted(self.task.inputs)}")

        def text(value: Any, name: str) -> str:
            value = str(value or "").strip()
            if not value:
                raise ValueError(f"{name} is empty")
            if limit and len(value) > limit and value not in self.seed_texts:
                trims.append(name)
                value = trim_text(value, limit)
            return value

        if kind in ("choice", "one_vs_rest") and (classes is not None or kind == "choice"):
            raw = spec.get("options")
            if not isinstance(raw, dict) or len(raw) < 2:
                raise ValueError(f"{kind} needs >= 2 options")
            options = {str(k): Option(criterion=text(o.get("criterion") if isinstance(o, dict) else o, f"options.{k}")) for k, o in raw.items()}
            if classes is not None:
                if set(options) != set(classes):
                    raise ValueError(f"main options must be exactly the labels {classes}")
                options = {c: options[c] for c in classes}
            cls = ChoiceNode if kind == "choice" else OneVsRestNode
            return cls(kind=kind, instructions=text(spec.get("instructions"), "instructions"), state=list(state), options=options)
        if kind == "score" and (classes is None or self.task.ordinal):
            levels = [text(level, f"levels.{i}") for i, level in enumerate(spec.get("levels") or [])]
            return ScoreNode(kind="score", instructions=text(spec.get("instructions"), "instructions"), state=list(state), levels=levels, outcomes=[Target(label=(classes or ["score"])[0])])
        if kind == "branch" and classes is None:
            condition = spec.get("condition") or spec.get("statement") or spec.get("instructions")
            return BranchNode(kind="branch", condition=text(condition, "condition"), state=list(state), if_true=Target(label="yes"), if_false=Target(label="no"))
        allowed = "choice, one_vs_rest" + (", score" if self.task.ordinal else "") if classes is not None else "branch, choice, score"
        raise ValueError(f"kind {kind!r} is not allowed here (allowed: {allowed})")

    # ---- evaluation --------------------------------------------------------
    def score(self, slot: str, variants: list[Variant], examples: list[Example], context: dict[str, Any]) -> None:
        """Ask every variant's questions for `examples` (one packed request per example and state),
        then fit each variant's readout with cross-validation from the cached answers."""
        if not variants:
            return
        reached = [ex for ex in examples if route(self.base, ex.fields) == slot]
        run_examples(self.program(slot, variants), reached, self.client, exhaustive=True, context=context, pool=self.pool)
        for v in variants:
            try:
                program = self.program(slot, [v])
                raw = run_examples(program, reached, self.client, exhaustive=True, context=context, pool=self.pool)
                x, kept = readout_design(program, slot, list(zip(reached, raw)))
                _, report, oof = fit_readout(program.nodes[slot].readout, x, kept, self.task.primary_metric, self.task.labels, self.task.negative_label, self.cv)
                v.cv, v.oof, v.fit, v.error = report["cv"], oof, {k: report.get(k) for k in ("C", "class_weight", "alpha", "folds", "n")}, None
            except (ValueError, KeyError) as exc:
                v.cv, v.oof, v.error = None, {}, f"{type(exc).__name__}: {exc}"[:300]

    def stacked_oof(self, slot: str, members: list[Variant]) -> tuple[float, dict[str, tuple[str, dict[str, float]]]]:
        """Train CV of one readout over several variants' questions (answers already cached)."""
        reached = [ex for ex in self.fit_set if self.routes.get(ex.id) == slot]
        program = self.program(slot, members)
        raw = run_examples(program, reached, self.client, exhaustive=True, context={"split": "train", "condition": self.condition}, pool=self.pool)
        x, kept = readout_design(program, slot, list(zip(reached, raw)))
        _, report, oof = fit_readout(program.nodes[slot].readout, x, kept, self.task.primary_metric, self.task.labels, self.task.negative_label, self.cv)
        return report["cv"], oof

    # ---- combinations ---------------------------------------------------------
    def combo_score(self, oofs: dict[str, dict[str, tuple[str, dict[str, float]]]]) -> float:
        """Primary metric of the whole program on the fit examples from per-slot out-of-fold predictions."""
        gold, pred = [], []
        for ex in self.fit_set:
            slot = self.routes[ex.id]
            gold.append(ex.label)
            if slot is None:
                pred.append(route_label(self.base, ex.fields))
            else:
                hit = oofs[slot].get(ex.id)
                pred.append(hit[0] if hit else None)
        return primary(self.task.primary_metric, gold, pred, self.task.labels, self.task.negative_label)

    def ranked_combos(self, limit: int | None = None, keep: Callable[[Variant], bool] | None = None) -> list[tuple[float, dict[str, Variant]]]:
        """Combinations of per-slot variants (those passing `keep`) by program-level train CV
        (exhaustive when small)."""
        pools = {s: [v for v in self.variants[s] if v.cv is not None and (keep is None or keep(v))] for s in self.slots}
        if any(not p for p in pools.values()):
            return []
        sizes = [len(pools[s]) for s in self.slots]
        if int(np.prod(sizes)) <= MAX_COMBOS:
            combos = [dict(zip(self.slots, c)) for c in itertools.product(*(pools[s] for s in self.slots))]
        else:  # coordinate ascent from the per-slot best, then every single-slot deviation from it
            best = {s: max(pools[s], key=lambda v: v.cv) for s in self.slots}
            for _ in range(3):
                for s in self.slots:
                    best[s] = max(pools[s], key=lambda v, s=s: self.combo_score({t: (v if t == s else best[t]).oof for t in self.slots}))
            combos = [best] + [{**best, s: v} for s in self.slots for v in pools[s] if v is not best[s]]
        scored = [(self.combo_score({s: c[s].oof for s in self.slots}), c) for c in combos]
        scored.sort(key=lambda sc: -sc[0])  # stable: earlier variants (seed first) win ties
        return scored[:limit] if limit else scored

    # ---- proposer ------------------------------------------------------------
    def elite(self, slot: str, best: dict[str, Variant]) -> list[tuple[float, Variant]]:
        pool = [v for v in self.variants[slot] if v.cv is not None]
        scored = [(self.combo_score({s: (v if s == slot else best[s]).oof for s in self.slots}), v) for v in pool]
        scored.sort(key=lambda sv: -sv[0])
        return scored[: self.cfg.elite]

    def evidence(self, slot: str, variant: Variant) -> list[dict[str, Any]]:
        """Reflection records (TRAIN fit examples only): per-class summary + worst out-of-fold examples."""
        by_id = {ex.id: ex for ex in self.fit_set}
        rows = [(by_id[i], pred, dist) for i, (pred, dist) in variant.oof.items() if i in by_id]
        labels = self.classes[slot]
        gold = [ex.label for ex, _, _ in rows]
        pred = [p for _, p, _ in rows]
        m = _per_class(gold, pred, labels)
        confusions = Counter((g, p) for g, p in zip(gold, pred) if g != p).most_common(5)
        summary = {
            "Inputs": f"Out-of-fold predictions of the best variant on {len(rows)} TRAIN examples reaching `{slot}`",
            "Generated Outputs": f"{self.task.primary_metric}={variant.cv}; per class: " + "; ".join(f"{lab}: P={v[0]:.2f} R={v[1]:.2f} F1={v[2]:.2f} (n={v[3]})" for lab, v in m.items()),
            "Feedback": "Largest confusions (gold -> predicted): " + ("; ".join(f"{g} -> {p}: {n}" for (g, p), n in confusions) or "none"),
        }
        loss = sorted(rows, key=lambda r: r[2].get(r[0].label, 0.0))
        worst: list[tuple[Example, str, dict[str, float]]] = []
        by_gold: dict[str, list[tuple[Example, str, dict[str, float]]]] = {}
        for r in loss:
            if r[1] != r[0].label:
                by_gold.setdefault(r[0].label, []).append(r)
        for depth in range(self.cfg.worst_examples):
            for group in by_gold.values():
                if depth < len(group) and len(worst) < self.cfg.worst_examples:
                    worst.append(group[depth])
        used = {f for n in variant.nodes.values() for f in n.state} | set(self.base.nodes[slot].state)
        fields = [f for f in self.task.inputs if f in used]
        records = [summary]
        for ex, p, dist in worst:
            records.append({
                "Inputs": {f: _clip(ex.fields[f]) for f in fields},
                "Generated Outputs": {"predicted": p, "probabilities": {k: round(v, 3) for k, v in dist.items()}},
                "Feedback": f"Incorrect: the gold label is {ex.label} (p={dist.get(ex.label, 0.0):.2f}); predicted {p} (p={dist.get(p, 0.0):.2f}).",
            })
        return records

    def task_info(self, slot: str) -> str:
        desc = self.task.label_descriptions
        lines = [self.task.description] if self.task.description else []
        lines.append(f"Node `{slot}` decides among the labels: " + "; ".join(f"{lab} ({desc[lab]})" if desc.get(lab) else lab for lab in self.classes[slot]))
        fields = [f"{f}" + (f" ({d})" if d else "") for f, d in self.task.inputs.items()]
        lines.append(f"State fields: {', '.join(fields)}. The primary metric is {self.task.primary_metric}" + (f" with {self.task.negative_label} as the negative label (predicting it never earns credit)" if self.task.negative_label else "") + ".")
        return "\n".join(lines)

    def prompt(self, slot: str, best: dict[str, Variant]) -> str:
        elite = self.elite(slot, best)
        doc = yaml.safe_dump([{"cv": round(score, 4), **v.spec()} for score, v in elite], sort_keys=False, allow_unicode=True, width=100)
        records = self.evidence(slot, best[slot])
        limit = self.task.optimizer.max_component_chars or 1500
        demo_note = " Set `demos: true` to add the most similar labelled TRAIN examples to the state of all its questions (use it in at most 5 variants)." if self.cfg.demos is not None or self.base.demos is not None else ""
        subs = {
            "<task_info>": self.task_info(slot), "<slot>": slot, "<metric>": self.task.primary_metric, "<size>": str(self.cfg.size), "<fields>": ", ".join(self.task.inputs),
            "<max_aux>": str(self.cfg.max_aux), "<limit>": str(limit), "<labels>": ", ".join(self.classes[slot]), "<demo_note>": demo_note,
            "<score_note>": ", `score` (ordered levels from the lowest to the highest label; the head maps the level probabilities to the label)" if self.task.ordinal else "",
            "<score_kind>": " or score" if self.task.ordinal else "",
        }
        template = POPULATION_TEMPLATE
        for k, v in subs.items():
            template = template.replace(k, v)
        return render_prompt(template, doc, records)

    def propose(self, slot: str, round_: int, best: dict[str, Variant]) -> list[Variant]:
        if self.lm is None:
            return []
        prompt = self.prompt(slot, best)
        leader = self.elite(slot, best)[0][1].root
        variants: list[Variant] = []
        problems: list[str] = []
        for attempt in range(2):  # one retry with the rejection reasons when nothing valid came back
            reply = self.lm(prompt)
            text, _ = parse_proposal(reply)
            variants, problems = self.parse(text, slot, round_)
            self.proposals.append({"round": round_, "slot": slot, "attempt": attempt, "valid": len(variants), "rejected": problems[:20]})
            if variants or not isinstance(prompt, str):
                break
            prompt = f"{prompt}\n\nYour previous reply was rejected: {'; '.join(problems)[:1500]}\nReply again with a corrected YAML list."
        for v in variants:
            v.root = leader
        return variants[: self.cfg.size]

    # ---- final candidates ------------------------------------------------------
    def stacks(self, allowed: dict[str, list[Variant]], start: dict[str, Variant]) -> dict[str, list[Variant]]:
        """Per slot, forward selection of variants (from `allowed`) under one readout by train CV:
        always a second view, further views while the stacked CV improves."""
        out: dict[str, list[Variant]] = {}
        for s in self.slots:
            members = [start[s]]
            current = start[s].cv if start[s].cv is not None else float("-inf")
            while len(members) < self.cfg.stack_max:
                trials = []
                for v in allowed[s]:
                    if any(v is m for m in members):
                        continue
                    try:
                        trials.append((self.stacked_oof(s, [*members, v])[0], v))
                    except (ValueError, KeyError):
                        continue
                if not trials:
                    break
                score, v = max(trials, key=lambda t: t[0])
                if score <= current + 1e-4 and len(members) > 1:  # the first addition is always made
                    break
                members.append(v)
                current = score
            out[s] = members
        return out


def _yaml_items(text: str) -> tuple[list[Any], list[str]]:
    """Top-level list items of a YAML reply, parsed one by one so that one malformed or truncated
    item (e.g. at the reply's token limit) does not lose the others."""
    try:
        items = yaml.safe_load(text)
        if isinstance(items, list):
            return items, []
    except yaml.YAMLError:
        pass
    chunks = re.split(r"(?m)^(?=- )", text)
    out: list[Any] = []
    problems: list[str] = []
    for i, chunk in enumerate(c for c in chunks if c.strip().startswith("- ")):
        try:
            parsed = yaml.safe_load(chunk)
            out.extend(parsed if isinstance(parsed, list) else [])
        except yaml.YAMLError as exc:
            problems.append(f"item {i + 1} is not valid YAML: {str(exc)[:150]}")
    return out, problems


def _plain(node: Node) -> Node:
    """Copy of a question node without fitted numbers or node routing (a readout decides instead)."""
    node = node.model_copy(deep=True)
    for attr in ("readout", "biases"):
        if hasattr(node, attr):
            setattr(node, attr, None)
    if isinstance(node, (ChoiceNode, OneVsRestNode)):
        node.options = {k: o.model_copy(update={"to": None}) if o.to is not None and o.to.node else o for k, o in node.options.items()}
    if isinstance(node, BranchNode):
        node.if_true = Target(label="yes") if node.if_true.node else node.if_true
        node.if_false = Target(label="no") if node.if_false.node else node.if_false
    return node


def fragment(program: Program, entry: str) -> dict[str, Node]:
    """A variant's questions from a seed program: `entry` (as `main`) and every question node it
    routes to (as aux, ids sanitized), routing replaced by the readout over all their answers."""
    order: list[str] = []
    queue = [entry]
    while queue:
        nid = queue.pop(0)
        if nid in order:
            continue
        if not isinstance(program.nodes[nid], QUESTION_KINDS):
            raise ValueError(f"{nid} ({program.nodes[nid].kind}) cannot be part of a variant")
        order.append(nid)
        queue.extend(program.children(nid))
    out: dict[str, Node] = {}
    for nid in order:
        local = "main" if nid == entry else re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]", "_", nid.lower())).strip("_")[:24] or "q"
        if not local[0].isalpha():
            local = f"q{local}"[:24]
        while local in out:
            local = f"{local[:21]}_{len(out)}"
        out[local] = _plain(program.nodes[nid])
    return out


def route_label(program: Program, fields: dict[str, str]) -> str | None:
    """Label decided by switches alone (examples that reach no slot)."""
    nid = program.root
    while isinstance(program.nodes[nid], SwitchNode):
        node = program.nodes[nid]
        target = node.cases.get(fields[node.field], node.default)
        if target is None:
            return None
        if target.label is not None:
            return target.label
        nid = target.node
    return None


def _per_class(gold: list[str], pred: list[str], labels: list[str]) -> dict[str, tuple[float, float, float, int]]:
    out = {}
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        n_pred = sum(p == lab for p in pred)
        n_gold = sum(g == lab for g in gold)
        prec = tp / n_pred if n_pred else 0.0
        rec = tp / n_gold if n_gold else 0.0
        out[lab] = (prec, rec, 2 * prec * rec / (prec + rec) if prec + rec else 0.0, n_gold)
    return out


def _texts(node: Any) -> list[str]:
    out = [getattr(node, a) for a in ("instructions", "condition") if isinstance(getattr(node, a, None), str)]
    out += [o.criterion for o in getattr(node, "options", {}).values()]
    out += list(getattr(node, "levels", []) or [])
    return out


def _clip(value: str) -> str:
    return value if len(value) <= FIELD_CLIP else value[:FIELD_CLIP] + " …[truncated]"


def load_variants(records: list[dict[str, Any]]) -> list[Variant]:
    """Variants persisted in `population.json` (evidence is recomputed from cached answers)."""
    return [Variant(r["slot"], {k: _node_from_record(n) for k, n in r["nodes"].items()}, bool(r["demos"]), r["rationale"], int(r["round"]), origin=r.get("origin", ""), root=r.get("root", "paper")) for r in records]
