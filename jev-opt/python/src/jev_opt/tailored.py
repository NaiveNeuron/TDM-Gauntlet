"""`jev-tailored` (architecture §5.3): GEPA adapted to the Jev decision model.

GEPA still owns the search (Pareto parent selection, val scheduling through `ProgressivePolicy`,
stopping). This module supplies GEPA's hooks:

- T1 rulebook: `RULEBOOK` is the fixed background block of every proposal prompt.
- T2 continuous acceptance: `StratifiedMinibatchSampler` (GEPA `BatchSampler`) draws `accept_n` TRAIN
  examples per iteration (class-balanced for macro metrics); `ContinuousAcceptance` (GEPA
  `AcceptanceCriterion`) accepts a child iff its mean log P(gold) (floored at 1e-4) on that minibatch
  improves and the minibatch primary metric drops by at most `accept_tol`.
- T3 confusion targeting: `TailoredAdapter.select_components` (GEPA `ReflectionComponentSelector`) picks the
  Choice node and option pair with the largest error mass x (1 - pairwise AUC) on the fit slice and has
  both criteria (and the instructions) rewritten jointly; other programs: round-robin over question nodes.
- T4 minimal pairs: similar TRAIN inputs (embedding kNN, the demonstrations' model and cache) whose gold
  answers are the two targeted options join the reflective dataset with Jev's probabilities.
- T6 wording vs bias: a candidate's numbers (Choice biases, cutoffs, thresholds, temperature) are fit on the
  train fit slice (§7); fit-slice examples are scored out of fold (the readout CV folds), other examples with
  the full fit, so acceptance and val selection see the score after bias. The raw score is recorded too.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import yaml
from gepa import EvaluationBatch
from gepa.core.state import GEPAState
from gepa.proposer.base import CandidateProposal
from pydantic import ValidationError

from . import metrics
from .adapter import FIELD_CLIP, JevAdapter, Recorder, _usage, candidate_key, cv_spec
from .demos import DemoPool
from .execute import ExampleResult, MissingAnswer, _Run, redecode, run_examples
from .fit import fit_program
from .hygiene import ProgramGuard
from .ir import BranchNode, ChoiceNode, OneVsRestNode, Program, ProgramError, QUESTION_KINDS, ScoreNode, apply_slots, text_slots
from .jev import JevClient
from .task import Example, TaskConfig
from .transforms import parse_proposal, render_prompt, trim_text

LOGP_FLOOR = 1e-4
DEFAULT_LIMIT = 1500  # chars per rewritten field when the task sets no `max_component_chars`
TOP_PAIRS = 3  # repeated visits of one parent rotate through its 3 highest-priority pairs
LM = Callable[[str | list[dict[str, Any]]], str]

# T1. Sources (read 2026-09-27): TypeSafe docs https://docs.typesafe.ai (how-to-build-with-system-one;
# model-jaggedness/jev-1.13: literal reading, indirection, counting, dates, numeric comparison and long
# irrelevant state are weak spots; a Choice is relative, a Noul absolute); Kiln, "Auto-optimizing Jev"
# https://kiln.tech/blog/auto_optimizing_jev_with_autoresearch (ordered procedures, worked examples and extra
# context did not help or hurt; describing what separates the options did); T. Kellogg's Jev notes
# https://gist.github.com/tkellogg/b23ab4ad90584251edb8151d0b149922 (Score levels as observable rules; the
# state must contain the evidence the levels reference).
RULEBOOK = """Rules for writing Jev questions (TypeSafe guidance for jev-1.13; every text you write must follow them):
1. Short, literal questions. Jev reads every word at face value. Say exactly what is asked, in plain words. No step-by-step procedures, checklists to work through, role play or motivational text: Jev does not follow sequences of steps.
2. Contrast, not description. Each option's criterion states what separates it from its nearest neighbour option: what it covers, and which similar cases belong to the neighbour instead. Instructions and criteria must agree.
3. Score levels are observable rules. Every level describes a self-contained, observable situation ("minor errors that never block understanding"), not a degree word ("fairly good"). Jev sees each level on its own, without its number or its neighbours.
4. Evidence in state. Refer only to evidence present in the state, naming the state field in backticks. Never ask about anything the state does not contain.
5. No indirection. Ask about the text itself, not about what a reader, annotator or model would think of it, and not about a property of a property.
6. Jev's weak spots: counting, dates and calendar arithmetic, numeric comparison, and long irrelevant state. Never make a decision depend on them; name the observable cue instead.
7. Keep the annotation conventions. The gold labels follow the dataset's guidelines: label definitions, boundary cases, tie-breaking and default rules (e.g. when the vague, none or neutral label applies). Rewrite such conventions in Jev style; never delete one the labels depend on."""

TAILORED_TEMPLATE = """I am improving the wording of one question of a Jev decision program. Jev (TypeSafe System One, `typesafe/jev-1.13`) is a calibrated classifier, not a chat model: it answers a question about the state (the example's input fields) with a probability for every option (Choice), for "true" (Noul) or for every level (Score); nothing is generated. After every rewrite, a per-option bias and a temperature are fitted again on training data, so a rewrite pays off only if Jev's probabilities separate the gold answers better.

<rulebook>Task: <task>

Labels (the dataset's definitions):
<labels>

The question to rewrite (node `<node>`<where>):
```
<curr_param>
```

<focus>

Evidence (training examples only, with Jev's answers to the current question):
<side_info>

Rewrite these fields: <fields>. Keep the meaning of every label; do not add, remove or rename options or levels; never copy example texts into the question. Every field must be at most <limit> characters.
Reply with one line `rationale: <what you changed and why>`, then one YAML mapping in a single fenced code block from field name to its new text (omit a field to keep its current text; use `|` block scalars). The current texts:
```
<skeleton>
```"""


# ---- T2: minibatches and acceptance ------------------------------------------------------

def stratified_sample(labels: Sequence[str], n: int, balanced: bool, rng: random.Random) -> list[int]:
    """Indexes of `n` items stratified by label: equal counts per label as far as the labels allow
    (`balanced`, for macro metrics), else label-proportional (largest remainders)."""
    groups: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        groups.setdefault(lab, []).append(i)
    n = min(n, len(labels))
    quota: dict[str, int] = {}
    if balanced:
        left = n
        ordered = sorted(groups, key=lambda lab: (len(groups[lab]), lab))
        for j, lab in enumerate(ordered):
            quota[lab] = min(len(groups[lab]), left // (len(ordered) - j))
            left -= quota[lab]
    else:
        exact = {lab: n * len(idx) / len(labels) for lab, idx in groups.items()}
        quota = {lab: int(v) for lab, v in exact.items()}
        for lab in sorted(groups, key=lambda lab: (-(exact[lab] - quota[lab]), lab))[: n - sum(quota.values())]:
            quota[lab] += 1
    picked = [i for lab in sorted(groups) for i in rng.sample(groups[lab], quota[lab])]
    rng.shuffle(picked)
    return picked


class StratifiedMinibatchSampler:
    """GEPA `BatchSampler`: a fresh stratified minibatch of `n` trainset ids per iteration, deterministic
    in (seed, iteration, call), so a resumed search draws the same minibatches."""

    def __init__(self, examples: Sequence[Example], n: int, balanced: bool, seed: int):
        self.labels = [ex.label for ex in examples]
        self.n, self.balanced, self.seed = n, balanced, seed
        self._iteration: int | None = None
        self._calls = 0

    def next_minibatch_ids(self, loader: Any, state: GEPAState) -> list[int]:
        if state.i == self._iteration:
            self._calls += 1
        else:
            self._iteration, self._calls = state.i, 0
        ids = list(loader.all_ids())
        rng = random.Random(f"{self.seed}:{state.i}:{self._calls}")
        return [ids[j] for j in stratified_sample([self.labels[i] for i in ids], self.n, self.balanced, rng)]


def mean_log_gold(outputs: Sequence[Mapping[str, Any]], key: str = "p_gold") -> float:
    """Mean log P(gold) with probabilities floored at 1e-4 (a failed example counts as the floor)."""
    return float(np.mean([math.log(max(float(o.get(key) or 0.0), LOGP_FLOOR)) for o in outputs])) if outputs else float("-inf")


def acceptance(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]], labels: list[str], metric: str, negative: str | None, tol: float) -> dict[str, Any]:
    """T2 rule on one minibatch: accept iff mean log P(gold) improves and the primary metric drops by at
    most `tol`. An invalid child (any example failed as an invalid program) is rejected."""
    primary = lambda outs: metrics.primary(metric, [o["gold"] for o in outs], [o["pred"] for o in outs], labels, negative)  # noqa: E731
    lb, la = mean_log_gold(before), mean_log_gold(after)
    pb, pa = primary(before), primary(after)
    invalid = any(str(o.get("error") or "").startswith("invalid program") for o in after)
    ok = not invalid and la > lb + 1e-12 and pa >= pb - tol - 1e-12
    return {"accept": ok, "logp_before": round(lb, 6), "logp_after": round(la, 6), "primary_before": round(pb, 6), "primary_after": round(pa, 6), "invalid": invalid, "n": len(after)}


class ContinuousAcceptance:
    """GEPA `AcceptanceCriterion` for T2 (see `acceptance`); verdicts are recorded as evaluation rows."""

    def __init__(self, task: TaskConfig, tol: float, record: Recorder, condition: str, start: str):
        self.labels, self.metric, self.negative = task.labels, task.primary_metric, task.negative_label
        self.tol, self.record, self.condition, self.start = tol, record, condition, start
        self._verdicts: dict[int, dict[str, Any]] = {}

    def _verdict(self, proposal: CandidateProposal) -> dict[str, Any]:
        key = id(proposal)
        if key not in self._verdicts:
            v = acceptance(proposal.eval_before.outputs, proposal.eval_after.outputs, self.labels, self.metric, self.negative, self.tol)
            self._verdicts[key] = v
            self.record({"condition": self.condition, "start": self.start, "candidate": candidate_key(proposal.candidate), "parent": list(proposal.parent_program_ids),
                         "split": "train", "stage": "accept", **v})
        return self._verdicts[key]

    def should_accept(self, proposal: CandidateProposal, state: GEPAState) -> bool:
        return bool(self._verdict(proposal)["accept"])

    def reject_reason(self, proposal: CandidateProposal, state: GEPAState) -> str:
        v = self._verdict(proposal)
        return (f"rejected on {v['n']} train examples: mean log P(gold) {v['logp_before']:.4f} -> {v['logp_after']:.4f}, "
                f"{self.metric} {v['primary_before']:.4f} -> {v['primary_after']:.4f} (tolerance {self.tol})" + (" (invalid program)" if v["invalid"] else ""))


class JevCostStopper:
    """GEPA stopper: fresh Jev spend of this search reached its cap (checked between iterations)."""

    def __init__(self, client: JevClient, cap: float | None, on_check: Callable[[], None] | None = None):
        self.client, self.cap, self.on_check = client, cap, on_check
        self.start = client.usage.fresh_cost

    @property
    def spent(self) -> float:
        return self.client.usage.fresh_cost - self.start

    def __call__(self, gepa_state: GEPAState) -> bool:
        if self.on_check is not None:
            self.on_check()
        return self.cap is not None and self.spent >= self.cap


# ---- T3: confusion targeting -------------------------------------------------------------

@dataclass(frozen=True)
class PairTarget:
    node: str
    a: str  # option keys of `node`
    b: str
    mass: float  # weighted share of fit-slice examples confused between a and b (either direction)
    auc: float  # AUC of p_a / (p_a + p_b) separating gold-a from gold-b examples at the node (0.5 = no signal)
    n_ab: int  # gold a, decided b
    n_ba: int  # gold b, decided a
    support: tuple[int, int]  # examples at the node whose gold option is a / b

    @property
    def priority(self) -> float:
        return self.mass * (1.0 - self.auc)

    def describe(self) -> str:
        return f"{self.node}: {self.a} vs {self.b} (error share {100 * self.mass:.1f}%, pairwise AUC {self.auc:.2f})"


def gold_option(program: Program, nid: str, gold: str) -> str | None:
    """The option of Choice `nid` whose target covers `gold` (None if none or several do)."""
    node = program.nodes[nid]
    hits = []
    for key, opt in node.options.items():
        target = opt.target(key)
        covered = {target.label} if target.label is not None else set(program.outcomes_under(target.node))
        if gold in covered:
            hits.append(key)
    return hits[0] if len(hits) == 1 else None


def pair_auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """P(score of a random positive > a random negative), ties count half; 0.5 without both classes."""
    if not pos or not neg:
        return 0.5
    p, q = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    diff = p[:, None] - q[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def _pair_score(probs: Mapping[str, float], a: str, b: str) -> float:
    pa, pb = float(probs.get(a, 0.0)), float(probs.get(b, 0.0))
    return pa / (pa + pb) if pa + pb > 0 else 0.5


def pair_targets(program: Program, results: Sequence[ExampleResult], weights: Mapping[str, float] | None = None) -> list[PairTarget]:
    """Option pairs of every Choice node (outside stage decisions and readout views), highest error mass
    x (1 - pairwise AUC) first. `results` are decoded runs of `program` (traces hold each visited Choice's
    probabilities and decision); `weights` = per-example quality weights (class balance for macro metrics)."""
    weights = weights or {}
    ok = [r for r in results if r.error is None]
    total = sum(weights.get(r.id, 1.0) for r in ok) or 1.0
    skip = program.decision_nodes() | program.view_nodes()
    out: list[PairTarget] = []
    for nid, node in program.nodes.items():
        if not isinstance(node, ChoiceNode) or nid in skip:
            continue
        rows = []  # (gold option, decided option, probabilities, weight)
        for r in ok:
            t = r.trace.get(nid)
            if not t or "choice" not in t:
                continue
            g = gold_option(program, nid, r.gold)
            if g is not None:
                rows.append((g, t["choice"], t["probabilities"], weights.get(r.id, 1.0)))
        keys = list(node.options)
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                ab = [w for g, d, _, w in rows if g == a and d == b]
                ba = [w for g, d, _, w in rows if g == b and d == a]
                if not ab and not ba:
                    continue
                pos = [_pair_score(p, a, b) for g, _, p, _ in rows if g == a]
                neg = [_pair_score(p, a, b) for g, _, p, _ in rows if g == b]
                out.append(PairTarget(nid, a, b, (sum(ab) + sum(ba)) / total, pair_auc(pos, neg), len(ab), len(ba), (len(pos), len(neg))))
    return sorted(out, key=lambda t: (-t.priority, t.node, t.a, t.b))


def slot_groups(program: Program) -> list[tuple[str, list[str]]]:
    """Round-robin units: each question node's text slots (in node order), then the input descriptions."""
    slots = list(text_slots(program))
    groups = [(nid, [s for s in slots if s.startswith(f"nodes.{nid}.")]) for nid, n in program.nodes.items() if isinstance(n, QUESTION_KINDS)]
    inputs = [s for s in slots if s.startswith("inputs.")]
    return [g for g in groups if g[1]] + ([("inputs", inputs)] if inputs else [])


# ---- T4: minimal pairs -------------------------------------------------------------------

def minimal_pairs(program: Program, target: PairTarget, results: Sequence[ExampleResult], pool: DemoPool, k: int) -> list[tuple[ExampleResult, ExampleResult, float]]:
    """Up to `k` disjoint pairs (gold option a, gold option b) of TRAIN examples at the target node, most
    similar inputs first (cosine of the pool's embeddings); a pair where Jev confused either member gets a
    small bonus. Anything outside the train pool is dropped before ranking."""
    train_ids = pool.ids()
    side: dict[str, list[ExampleResult]] = {target.a: [], target.b: []}
    for r in results:
        if r.error is None and r.id in train_ids and target.node in r.trace:
            g = gold_option(program, target.node, r.gold)
            if g in side:
                side[g].append(r)
    A, B = side[target.a], side[target.b]
    if not A or not B or k <= 0:
        return []
    vecs = pool.vectors([r.id for r in A + B])
    sims = np.array([vecs[r.id] for r in A]) @ np.array([vecs[r.id] for r in B]).T
    confused = lambda r, other: r.trace[target.node].get("choice") == other  # noqa: E731
    ranked = sorted(((float(sims[i, j]) + 0.05 * (confused(A[i], target.b) + confused(B[j], target.a)), i, j) for i in range(len(A)) for j in range(len(B))), reverse=True)
    used_a: set[int] = set()
    used_b: set[int] = set()
    out: list[tuple[ExampleResult, ExampleResult, float]] = []
    for _, i, j in ranked:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        out.append((A[i], B[j], float(sims[i, j])))
        if len(out) == k:
            break
    assert all(x.id in train_ids and y.id in train_ids for x, y, _ in out), "minimal pair outside the train pool"
    return out


# ---- the adapter --------------------------------------------------------------------------

def _p_gold(r: ExampleResult) -> float:
    return 0.0 if r.error is not None or r.pred is None else float(r.dist.get(r.gold, 0.0))


def _node_view(program: Program, nid: str) -> dict[str, Any]:
    """The texts and routing of one question node as the proposer sees them (no fitted numbers)."""
    node = program.nodes[nid]
    out: dict[str, Any] = {"kind": node.kind}
    if isinstance(node, BranchNode):
        out["condition"] = node.condition
    else:
        out["instructions"] = node.instructions
    out["state"] = list(node.state)
    if isinstance(node, (ChoiceNode, OneVsRestNode)):
        opts: dict[str, Any] = {}
        for key, opt in node.options.items():
            t = opt.target(key)
            opts[key] = {"criterion": opt.criterion, "leads to": f"label {t.label}" if t.label is not None else f"node {t.node} (labels {', '.join(program.outcomes_under(t.node))})"}
        out["options"] = opts
    if isinstance(node, ScoreNode):
        out["levels"] = list(node.levels)
        out["outcomes (lowest first)"] = [t.label or f"node {t.node}" for t in node.outcomes]
    return out


class TailoredAdapter(JevAdapter):
    """`JevAdapter` (text slots, topology frozen) with the T2-T6 hooks of `jev-tailored` for one start."""

    def __init__(self, *, task: TaskConfig, seed: Program, client: JevClient, train: list[Example], guard: ProgramGuard, condition: str,
                 record: Recorder | None = None, reflection_lm: LM | None = None, pool: DemoPool | None = None, start: str = "paper"):
        super().__init__(task=task, seed=seed, client=client, train=train, guard=guard, mode="text", condition=condition, record=record, reflection_lm=reflection_lm, pool=pool)
        self.cfg = task.optimizer.tailored
        self.start = start
        self.trainset = list(train)  # minibatches come from all of TRAIN; fit-slice examples are scored out of fold (T6)
        self.text_proposer = self._propose_tailored
        self._weights = {ex.id: ex.weight for ex in train}
        self._oof: dict[str, dict[str, ExampleResult]] = {}
        self._raw: dict[str, Program] = {}
        self._target: PairTarget | None = None
        self._group: str | None = None

    # ---- T6: fit on the fit slice, score out of fold -----------------------------------
    def prepare(self, candidate: dict[str, str]) -> tuple[Program | None, list[str], list[ExampleResult]]:
        key = candidate_key(candidate)
        if key in self._prepared:
            return self._prepared[key]
        out = super().prepare(candidate)
        if out[0] is not None:
            self._raw[key] = self.to_program(candidate).check(self.task)
            if key in self._fit_pairs:
                self._oof[key] = self._out_of_fold(key, out[2])
        return out

    def _out_of_fold(self, key: str, full: list[ExampleResult]) -> dict[str, ExampleResult]:
        """Fit-slice results decoded with numbers fit on the other folds (`optimizer.readout_folds`,
        grouped by `readout_group`); a fold whose fit fails falls back to the full fit."""
        unfitted, pairs = self._fit_pairs[key]
        opt = self.task.optimizer
        full_by_id = {r.id: r for r in full}
        out = {ex.id: r for ex, r in pairs if r.error is not None}
        ok = [(ex, r) for ex, r in pairs if r.error is None]
        cv = cv_spec(opt)
        splits = cv.splits([ex.label for ex, _ in ok], [ex for ex, _ in ok])
        if not splits:
            return {**out, **full_by_id}
        for tr, te in splits:
            try:
                fitted, _ = fit_program(unfitted, [ok[i] for i in tr], self.task.primary_metric, opt.calibrate, rounds=2 if opt.fit_parameters else 0,
                                        negative=self.task.negative_label, class_bias=opt.class_bias, cv=cv)
            except ValueError:
                fitted = None
            for i in te:
                ex, r = ok[i]
                out[ex.id] = redecode(fitted, ex, r.memo, r.demos) if fitted is not None else full_by_id[ex.id]
        return out

    def _raw_results(self, key: str, batch: list[Example], results: list[ExampleResult]) -> list[ExampleResult]:
        """The candidate as written (no fitted number) on the same answers; new calls only where the raw
        routing leaves the recorded path."""
        raw = self._raw[key]
        out: list[ExampleResult | None] = []
        missing: list[int] = []
        for i, (ex, r) in enumerate(zip(batch, results)):
            if r.error is not None:
                out.append(r)
                continue
            try:
                out.append(redecode(raw, ex, r.memo, r.demos))
            except MissingAnswer:
                out.append(None)
                missing.append(i)
        if missing:
            ran = run_examples(raw, [batch[i] for i in missing], self.client, context={"split": ",".join(sorted({batch[i].split for i in missing})), "condition": self.condition}, pool=self.pool)
            for i, r in zip(missing, ran):
                out[i] = r
        return [r for r in out if r is not None]

    def evaluate(self, batch: list[Example], candidate: dict[str, str], capture_traces: bool = False) -> EvaluationBatch:
        program, _, _ = self.prepare(candidate)
        if program is None or self.task.evaluation.command:
            return super().evaluate(batch, candidate, capture_traces)
        key = candidate_key(candidate)
        oof = self._oof.get(key, {})
        live = [ex for ex in batch if ex.id not in oof]
        context = {"split": ",".join(sorted({ex.split for ex in live})), "condition": self.condition}
        ran = dict(zip((ex.id for ex in live), run_examples(program, live, self.client, context=context, pool=self.pool))) if live else {}
        results = [oof[ex.id] if ex.id in oof else ran[ex.id] for ex in batch]
        raws = self._raw_results(key, batch, results)
        labels, metric, negative = self.task.labels, self.task.primary_metric, self.task.negative_label
        scores, raw_scores, objectives, outputs = [], [], [], []
        for ex, r, raw in zip(batch, results, raws):
            quality = metrics.example_quality(metric, ex.label, r.pred, labels)
            scores.append(quality * ex.weight)
            raw_scores.append(metrics.example_quality(metric, ex.label, raw.pred, labels) * ex.weight)
            objectives.append(self._objectives(program, ex, r, quality))
            outputs.append({"id": ex.id, "pred": r.pred, "gold": ex.label, "dist": {k: round(v, 4) for k, v in r.dist.items()}, "p_gold": _p_gold(r),
                            "calls": r.calls, "error": r.error, "pred_raw": raw.pred, "p_gold_raw": _p_gold(raw)})
        gold = [ex.label for ex in batch]
        self.record({
            "condition": self.condition, "start": self.start, "candidate": key, "split": ",".join(sorted({ex.split for ex in batch})), "stage": f"{len(batch)}", "n": len(batch),
            "quality": float(np.mean(scores)) if scores else 0.0, "quality_raw": float(np.mean(raw_scores)) if raw_scores else 0.0,
            "primary": metrics.primary(metric, gold, [r.pred for r in results], labels, negative),
            "primary_raw": metrics.primary(metric, gold, [r.pred for r in raws], labels, negative),
            "logp": mean_log_gold(outputs), "logp_raw": mean_log_gold(outputs, "p_gold_raw"), "out_of_fold": len(batch) - len(live), "valid": True,
            **_usage(list(ran.values())),
        })
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=results if capture_traces else None, objective_scores=objectives)

    # ---- T3: component selection ---------------------------------------------------------
    def fit_view(self, candidate: dict[str, str]) -> list[ExampleResult]:
        """The candidate's fit-slice results after bias, out of fold where available."""
        key = candidate_key(candidate)
        _, _, full = self.prepare(candidate)
        return list(self._oof[key].values()) if key in self._oof else full

    def select_components(self, state: GEPAState, trajectories: list[Any], subsample_scores: list[float], candidate_idx: int, candidate: dict[str, str]) -> list[str]:
        """GEPA `ReflectionComponentSelector`: the highest-priority confused Choice pair of the parent (its two
        criteria + the node's instructions); repeated visits of one parent rotate through its top pairs.
        Without a pair (Score/Noul-only programs, no confusion, or T3 off): round-robin over question nodes."""
        key = candidate_key(candidate)
        visit = self._visits.get(key, 0)
        self._visits[key] = visit + 1
        self._target, self._group = None, None
        program = self._raw.get(key) or self.to_program(candidate)
        if self.cfg.confusion_target:
            targets = [t for t in pair_targets(program, self.fit_view(candidate), self._weights) if t.priority > 0][:TOP_PAIRS]
            if targets:
                t = targets[visit % len(targets)]
                self._target = t
                names = [f"nodes.{t.node}.options.{t.a}.criterion", f"nodes.{t.node}.options.{t.b}.criterion", f"nodes.{t.node}.instructions"]
                return [n for n in names if n in candidate]
        groups = slot_groups(program)
        self._group, names = groups[visit % len(groups)]
        return names

    # ---- reflective dataset (T3 focus + T4 minimal pairs) ----------------------------------
    def make_reflective_dataset(self, candidate: dict[str, str], eval_batch: EvaluationBatch, components_to_update: list[str]) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        key = candidate_key(candidate)
        _, problems, _ = self.prepare(candidate)
        view = self.fit_view(candidate)
        traces = [t for t in (eval_batch.trajectories or []) if t.id in self._train_by_id]  # TRAIN only (§8)
        exhaustive = [r for _, r in self._fit_pairs[key][1]] if key in self._fit_pairs else traces  # real call counts for the cost line
        records: list[dict[str, Any]] = [self._summary_record(view, exhaustive, problems)]
        t = self._target
        if t is not None:
            records.append(self._target_record(t))
        raw_prog = self._raw.get(key)
        for r in self._pick_examples(traces, t, raw_prog):
            records.append(self._tailored_example(r, raw_prog, t))
        if t is not None and self.cfg.minimal_pairs and self.pool is not None and raw_prog is not None:
            for x, y, sim in minimal_pairs(raw_prog, t, view, self.pool, self.cfg.minimal_pairs):
                records.append(self._pair_record(x, y, sim, t, raw_prog))
        return {comp: records for comp in components_to_update}

    def _target_record(self, t: PairTarget) -> dict[str, Any]:
        return {
            "Inputs": f"Targeted confusion at node `{t.node}`: option `{t.a}` vs option `{t.b}`",
            "Generated Outputs": (f"On the train fit slice after the fitted per-option bias: gold {t.a} decided {t.b}: {t.n_ab}; gold {t.b} decided {t.a}: {t.n_ba} "
                                  f"({100 * t.mass:.1f}% of the weighted examples). Pairwise AUC of p({t.a}) / (p({t.a}) + p({t.b})) between gold-{t.a} "
                                  f"({t.support[0]}) and gold-{t.b} ({t.support[1]}) examples: {t.auc:.2f} (0.5 = Jev's probabilities do not separate them; 1.0 = perfectly)."),
            "Feedback": (f"A bias cannot fix what the probabilities do not separate. Rewrite the criteria of `{t.a}` and `{t.b}` together so that the "
                         f"examples below fall on the right side, and adjust the instructions only if they conflict with the criteria."),
        }

    def _pick_examples(self, traces: list[ExampleResult], t: PairTarget | None, program: Program | None) -> list[ExampleResult]:
        n = self.cfg.max_examples
        if t is None or program is None:
            wrong = [r for r in traces if not r.correct]
            right = [r for r in traces if r.correct]
            k = min(len(wrong), max(n - n // 4, n - len(right)))
            return wrong[:k] + right[: n - k]
        gold_opt = {r.id: gold_option(program, t.node, r.gold) if t.node in r.trace else None for r in traces}
        in_pair = [r for r in traces if gold_opt[r.id] in (t.a, t.b)]
        confused = [r for r in in_pair if r.trace[t.node].get("choice") in (t.a, t.b) and r.trace[t.node].get("choice") != gold_opt[r.id]]
        seen = {r.id for r in confused}
        right = [r for r in in_pair if r.correct and r.id not in seen]
        k = min(len(confused), max(n - n // 4, n - len(right)))
        return confused[:k] + right[: n - k]

    def _answers(self, r: ExampleResult, program: Program | None, nid: str | None) -> str:
        """Jev's own answers (before the fitted bias) at `nid`, or at every question node on the decided path."""
        if program is None or r.error is not None:
            return ""
        ex = self._train_by_id[r.id]
        try:
            raw = redecode(program, ex, r.memo, r.demos)
        except MissingAnswer:
            return ""
        run = _Run(program, ex, None, False, dict(r.memo), None, r.demos)

        def one(node_id: str) -> str:
            t, node = raw.trace.get(node_id, {}), program.nodes[node_id]
            if isinstance(node, ScoreNode):
                try:
                    probs = run.fetch(node_id, {**ex.fields, **raw.derived})[node_id].get("probabilities", {})
                except MissingAnswer:
                    return f"expected level {t.get('expected_level', 0.0):.2f}"
                levels = " · ".join(f"level {i}: {float(probs.get(str(i), 0.0)):.2f}" for i in range(len(node.levels)))
                return f"{levels}; expected level {t.get('expected_level', 0.0):.2f}"
            if "probabilities" in t:
                return " · ".join(f"{k} {v:.2f}" for k, v in t["probabilities"].items())
            if isinstance(t.get("p_true"), dict):
                return "P(true): " + " · ".join(f"{k} {v:.2f}" for k, v in t["p_true"].items())
            return f"P(true) {t['p_true']:.2f}" if "p_true" in t else ""

        if nid is not None:
            return one(nid) if nid in raw.trace else ""
        return "; ".join(f"`{n}`: {one(n)}" for n in raw.trace if isinstance(program.nodes[n], QUESTION_KINDS))

    def _tailored_example(self, r: ExampleResult, program: Program | None, t: PairTarget | None) -> dict[str, Any]:
        ex = self._train_by_id[r.id]
        inputs = {k: (v if len(v) <= FIELD_CLIP else v[:FIELD_CLIP] + " …[truncated]") for k, v in ex.fields.items()}
        generated: dict[str, Any] = {"Jev's answers (before the fitted bias)": self._answers(r, program, t.node if t else None) or "n/a",
                                     "final label (after the fitted bias)": r.pred}
        if r.error:
            feedback = f"Failed: {r.error}"
        elif r.correct:
            feedback = f"Correct ({r.gold})."
        else:
            feedback = f"Incorrect: the gold label is {r.gold}; decided {r.pred}."
        return {"Inputs": inputs, "Generated Outputs": generated, "Feedback": feedback}

    def _pair_record(self, x: ExampleResult, y: ExampleResult, sim: float, t: PairTarget, program: Program) -> dict[str, Any]:
        clip = lambda v: v if len(v) <= FIELD_CLIP else v[:FIELD_CLIP] + " …[truncated]"  # noqa: E731
        fx, fy = self._train_by_id[x.id].fields, self._train_by_id[y.id].fields

        def status(r: ExampleResult, want: str) -> str:
            decided = r.trace[t.node].get("choice")
            return "right" if decided == want else f"wrong (decided {decided})"

        return {
            "Inputs": {f"input A (gold option {t.a})": {k: clip(v) for k, v in fx.items()}, f"input B (gold option {t.b})": {k: clip(v) for k, v in fy.items()}},
            "Generated Outputs": {"A: Jev's answers": self._answers(x, program, t.node) or "n/a", "B: Jev's answers": self._answers(y, program, t.node) or "n/a"},
            "Feedback": (f"Minimal pair (input similarity {sim:.2f}): similar inputs, different gold options. Jev was {status(x, t.a)} on A and "
                         f"{status(y, t.b)} on B. The criteria of `{t.a}` and `{t.b}` must name the difference between A and B."),
        }

    # ---- T1: proposal ---------------------------------------------------------------------
    def _render(self, program: Program, components: list[str], records: list[Mapping[str, Any]], candidate: dict[str, str]) -> str:
        t, group = self._target, self._group
        limit = self.task.optimizer.max_component_chars or DEFAULT_LIMIT
        nid = t.node if t is not None else (group if group in program.nodes else None)
        if nid is not None:
            current = yaml.safe_dump(_node_view(program, nid), sort_keys=False, allow_unicode=True, width=100)
            node = program.nodes[nid]
            where = f", {node.kind}, state fields {', '.join(node.state)}"
        else:
            current = yaml.safe_dump({"inputs (state field descriptions)": dict(program.inputs)}, sort_keys=False, allow_unicode=True, width=100)
            where = ": the descriptions of the state fields, shown to Jev before every question"
        if t is not None:
            focus = (f"Focus: Jev confuses option `{t.a}` with option `{t.b}` at this node, and its probabilities barely separate the two "
                     f"(pairwise AUC {t.auc:.2f}), so the fitted bias cannot fix it. Make the two criteria separate these cases.")
        else:
            focus = "Focus: reduce the errors below while keeping the question correct on the examples it already gets right."
        desc = self.task.label_descriptions
        labels = "\n".join(f"- {lab}: {desc[lab]}" if desc.get(lab) else f"- {lab}" for lab in self.task.labels)
        skeleton = yaml.safe_dump({c: candidate[c] for c in components}, sort_keys=False, allow_unicode=True, width=100, default_style=None)
        template = (TAILORED_TEMPLATE.replace("<rulebook>", RULEBOOK + "\n\n" if self.cfg.rulebook else "")
                    .replace("<task>", self.task.description or self.task.name).replace("<labels>", labels).replace("<node>", nid or "inputs")
                    .replace("<where>", where).replace("<focus>", focus).replace("<fields>", ", ".join(f"`{c}`" for c in components))
                    .replace("<limit>", str(limit)).replace("<skeleton>", skeleton.rstrip()))
        prompt = render_prompt(template, current.rstrip(), list(records))
        assert isinstance(prompt, str)
        return prompt

    def _parse(self, reply: str, components: list[str], candidate: dict[str, str], limit: int, trim: bool) -> tuple[dict[str, str], str, list[str]]:
        text, rationale = parse_proposal(reply)
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            return {}, rationale, [f"the reply is not a YAML mapping: {str(exc)[:200]}"]
        if not isinstance(data, dict):
            return {}, rationale, ["the reply must be one YAML mapping from field name to new text"]
        data.pop("rationale", None)
        problems = [f"unknown field {k!r} (editable: {', '.join(components)})" for k in data if k not in components]
        new: dict[str, str] = {}
        for k in components:
            if k not in data:
                continue
            v = str(data[k]).strip() if data[k] is not None else ""
            if not v:
                problems.append(f"field {k} is empty")
            elif len(v) > limit:
                if trim:
                    new[k] = trim_text(v, limit)
                else:
                    problems.append(f"field {k} has {len(v)} characters (limit {limit})")
            else:
                new[k] = v
        if not problems:
            try:
                program = apply_slots(self.seed, {**candidate, **new}).check(self.task)
                problems = self.guard.check_program(program, self._train_ids)
            except (ProgramError, ValidationError, ValueError) as exc:
                problems = [str(exc)[:400]]
        return new, rationale, problems

    def _propose_tailored(self, candidate: dict[str, str], reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]], components_to_update: list[str]) -> dict[str, str]:
        """GEPA `custom_candidate_proposer`: one OMP call rewrites the selected fields jointly under the
        rulebook (T1); one retry with the rejection reasons, over-long fields of the retry are trimmed."""
        assert self.reflection_lm is not None
        key = candidate_key(candidate)
        program = self._raw.get(key) or self.to_program(candidate)
        limit = self.task.optimizer.max_component_chars or DEFAULT_LIMIT
        prompt = self._render(program, components_to_update, list(reflective_dataset.get(components_to_update[0], [])), candidate)
        new: dict[str, str] = {}
        rationale = ""
        for attempt in range(2):
            new, rationale, problems = self._parse(self.reflection_lm(prompt), components_to_update, candidate, limit, trim=attempt == 1)
            if not problems:
                break
            new = {}
            prompt = f"{prompt}\n\nYour previous reply was rejected: {'; '.join(problems)[:600]}\nReply again with a corrected YAML mapping."
        changed = {k: v for k, v in new.items() if v != candidate[k]}
        if changed and candidate_key({**candidate, **changed}) in self._prepared:
            return {}  # an existing candidate: GEPA skips the iteration instead of re-adding it
        if changed:
            what = self._target.describe() if self._target is not None else f"round-robin {self._group}"
            self.rationales[candidate_key({**candidate, **changed})] = f"tailored rewrite of {', '.join(sorted(changed))} [{what}]: {rationale}"[:700]
        return changed
