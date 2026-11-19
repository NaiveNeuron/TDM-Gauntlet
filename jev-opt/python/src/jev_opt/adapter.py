"""GEPA adapter for Jev programs (architecture §5, §7, §9, §10).

`JevAdapter` implements GEPA's `GEPAAdapter` protocol; `ProgressivePolicy` implements
GEPA's `EvaluationPolicy`. GEPA owns search, sampling, acceptance, merge and stopping.
"""

from __future__ import annotations

import hashlib
import json
import random
import shlex
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from gepa import EvaluationBatch
from gepa.core.state import GEPAState
from gepa.strategies.eval_policy import EvaluationPolicy
from gepa.strategies.instruction_proposal import InstructionProposalSignature
from pydantic import ValidationError

from . import metrics
from .demos import DemoPool
from .execute import ExampleResult, MissingAnswer, redecode, run_examples
from .fit import CVSpec, fit_program, parameters, readout_nodes, with_readouts
from .hygiene import ProgramGuard
from .ir import Demos, Program, ProgramError, QUESTION_KINDS, apply_slots, text_slots
from .jev import JevClient
from .task import Example, OptimizerConfig, TaskConfig
from .transforms import (
    STRUCTURAL_TEMPLATE,
    TEXT_EDIT_TEMPLATE,
    Move,
    TransformError,
    length_rule,
    moves,
    parse_proposal,
    render_prompt,
    set_demos,
    slot_length_problems,
    structure_signature,
    trim_slots,
    trim_text,
)

PROGRAM_COMPONENT = "program"
FIELD_CLIP = 600
MAX_EXTRA_FAILURES = 3

LM = Callable[[str | list[dict[str, Any]]], str]
Recorder = Callable[[dict[str, Any]], None]


def candidate_key(candidate: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(candidate, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def stratified_order(examples: list[Example], seed: int) -> list[int]:
    """Deterministic index order whose every prefix is ~label-proportional (nested subsets)."""
    rng = random.Random(seed)
    groups: dict[str, list[int]] = {}
    for i, ex in enumerate(examples):
        groups.setdefault(ex.label, []).append(i)
    for idxs in groups.values():
        rng.shuffle(idxs)
    taken = dict.fromkeys(groups, 0)
    order: list[int] = []
    while len(order) < len(examples):
        lab = min((g for g in groups if taken[g] < len(groups[g])), key=lambda g: (taken[g] / len(groups[g]), g))
        order.append(groups[lab][taken[lab]])
        taken[lab] += 1
    return order


def train_roles(train: list[Example], opt: OptimizerConfig) -> tuple[list[Example], list[Example]]:
    """(fit set, GEPA trainset) from the train split, both stratified prefixes of one fixed order.

    Fit set: the first `fit_examples`. GEPA trainset (reflection minibatches): all of train, or with
    `fit_holdout` the examples after the fit set; capped at `reflection_pool`."""
    order = stratified_order(train, opt.seed)
    fit = [train[i] for i in order[: opt.fit_examples]]
    rest = order[opt.fit_examples :] if opt.fit_holdout else order
    if not rest:
        raise ValueError("fit_holdout leaves no train examples for reflection; lower fit_examples")
    pool = rest[: opt.reflection_pool] if opt.reflection_pool else rest
    return fit, [train[i] for i in sorted(pool)]


def cv_spec(opt: OptimizerConfig) -> CVSpec:
    """Readout cross-validation (§7) from the optimizer config."""
    return CVSpec(folds=opt.readout_folds, seed=opt.seed, group=opt.readout_group)


def parse_program(text: str) -> Program:
    data = yaml.safe_load(text)
    if isinstance(data, dict):
        data.pop("rationale", None)
    return Program.model_validate(data)


def _clip(value: str) -> str:
    return value if len(value) <= FIELD_CLIP else value[:FIELD_CLIP] + " …[truncated]"


class JevAdapter:
    def __init__(
        self,
        *,
        task: TaskConfig,
        seed: Program,
        client: JevClient,
        train: list[Example],
        guard: ProgramGuard,
        mode: Literal["text", "program"],
        condition: str,
        record: Recorder | None = None,
        reflection_lm: LM | None = None,
        structural_lm: LM | None = None,
        pool: DemoPool | None = None,
    ):
        self.task = task
        self.seed = seed
        self.client = client
        self._train_by_id = {ex.id: ex for ex in train}
        self.fit_set, self.trainset = train_roles(train, task.optimizer)
        self.pool = pool
        self._train_ids = {ex.id for ex in train}
        self._seed_texts = set(text_slots(seed).values())
        self.guard = guard
        self.mode = mode
        self.condition = condition
        self.record = record or (lambda row: None)
        self.reflection_lm = reflection_lm
        self.structural_lm = structural_lm
        self.propose_new_texts = self._propose_program if mode == "program" else None
        # text mode with a length budget: GEPA's `custom_candidate_proposer` hook (§5)
        self.text_proposer = self._propose_text if mode == "text" and task.optimizer.max_component_chars else None
        self._prepared: dict[str, tuple[Program | None, list[str], list[ExampleResult]]] = {}
        self._fit_pairs: dict[str, tuple[Program, list[tuple[Example, ExampleResult]]]] = {}  # candidate -> (unfitted program, exhaustive fit-slice run)
        self.rationales: dict[str, str] = {}
        self._visits: dict[str, int] = {}

    # ---- candidate <-> program -----------------------------------------
    def seed_candidate(self) -> dict[str, str]:
        if self.mode == "program":
            return {PROGRAM_COMPONENT: self.seed.to_yaml()}
        return text_slots(self.seed)

    def to_program(self, candidate: dict[str, str]) -> Program:
        if self.mode == "program":
            return parse_program(candidate[PROGRAM_COMPONENT])
        return apply_slots(self.seed, candidate)

    def prepare(self, candidate: dict[str, str]) -> tuple[Program | None, list[str], list[ExampleResult]]:
        """Validate, hygiene-check and numerically fit a candidate once (keyed by candidate hash)."""
        key = candidate_key(candidate)
        if key in self._prepared:
            return self._prepared[key]
        opt = self.task.optimizer
        try:
            program = self.to_program(candidate).check(self.task)
            if opt.readout:
                program = with_readouts(program, self.task.primary_metric).check(self.task)
            problems = self.guard.check_program(program, self._train_ids)
        except (ProgramError, ValidationError, yaml.YAMLError, ValueError) as exc:
            program, problems = None, [str(exc)[:800]]
        fit_results: list[ExampleResult] = []
        if program is not None and not problems:
            if self.fit_set and not self.task.evaluation.command and (opt.calibrate or (opt.fit_parameters and parameters(program, opt.class_bias)) or readout_nodes(program)):
                raw = run_examples(program, self.fit_set, self.client, exhaustive=True, context={"split": "train", "condition": self.condition}, pool=self.pool)
                pairs = list(zip(self.fit_set, raw))
                self._fit_pairs[key] = (program, pairs)
                try:
                    program, report = fit_program(program, pairs, self.task.primary_metric, opt.calibrate, rounds=2 if opt.fit_parameters else 0, negative=self.task.negative_label, class_bias=opt.class_bias, cv=cv_spec(opt))
                except ValueError as exc:  # e.g. a readout whose fit examples hold a single class
                    self._prepared[key] = (None, [f"fitting failed: {exc}"[:800]], [])
                    return self._prepared[key]
                fit_results = [_redecode_or_raw(program, ex, r) for ex, r in pairs]
                self.record({"condition": self.condition, "candidate": key, "split": "train", "stage": "fit", "n": len(pairs), "fit": report, **_usage(raw)})
        else:
            program = None
        self._prepared[key] = (program, problems, fit_results)
        return self._prepared[key]

    def fitted_program(self, candidate: dict[str, str]) -> Program | None:
        return self.prepare(candidate)[0]

    # ---- GEPAAdapter.evaluate --------------------------------------------
    def evaluate(self, batch: list[Example], candidate: dict[str, str], capture_traces: bool = False) -> EvaluationBatch:
        program, problems, _ = self.prepare(candidate)
        labels = self.task.labels
        if program is None:
            results = [ExampleResult(ex.id, ex.label, None, {lab: 1 / len(labels) for lab in labels}, error="invalid program: " + "; ".join(problems)) for ex in batch]
        elif self.task.evaluation.command:
            results = self._black_box(program, batch)
        else:
            context = {"split": ",".join(sorted({ex.split for ex in batch})), "condition": self.condition}
            results = run_examples(program, batch, self.client, context=context, pool=self.pool)
        metric = self.task.primary_metric
        scores, objectives, outputs = [], [], []
        for ex, r in zip(batch, results):
            if r.external_score is not None:
                quality = r.external_score
            else:
                quality = metrics.example_quality(metric, ex.label, r.pred, labels)
            scores.append(quality * ex.weight)
            objectives.append(self._objectives(program, ex, r, quality))
            outputs.append({"id": ex.id, "pred": r.pred, "gold": ex.label, "dist": {k: round(v, 4) for k, v in r.dist.items()}, "calls": r.calls, "error": r.error})
        splits = sorted({ex.split for ex in batch})
        self.record({
            "condition": self.condition,
            "candidate": candidate_key(candidate),
            "split": ",".join(splits),
            "stage": f"{len(batch)}",
            "n": len(batch),
            "quality": float(np.mean(scores)) if scores else 0.0,
            "valid": program is not None,
            "problems": problems[:3],
            **_usage(results),
        })
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=results if capture_traces else None, objective_scores=objectives)

    def _objectives(self, program: Program | None, ex: Example, r: ExampleResult, quality: float) -> dict[str, float]:
        calls = r.calls or (_static_calls(program) if program is not None else 1)
        out = {
            "quality": quality * ex.weight,
            "calibration": quality if r.external_score is not None else (1.0 - metrics.example_brier(ex.label, r.dist, self.task.labels) if r.pred is not None else 0.0),
            "cheap": 1.0 / max(calls, 1) if program is not None else 0.0,
            "tokens": 1.0 / (1.0 + r.input_tokens / 1000.0) if program is not None else 0.0,
        }
        return {k: v for k, v in out.items() if k in self.task.objectives}

    def _black_box(self, program: Program, batch: list[Example]) -> list[ExampleResult]:
        """Task-supplied evaluator: `command` gets {program} and {examples} paths, prints {score, per_example?}."""
        with tempfile.TemporaryDirectory() as tmp:
            prog_path, ex_path = Path(tmp) / "program.yaml", Path(tmp) / "examples.jsonl"
            prog_path.write_text(program.to_yaml())
            ex_path.write_text("".join(json.dumps({"id": e.id, **e.fields, "label": e.label}) + "\n" for e in batch))
            cmd = self.task.evaluation.command.format(program=shlex.quote(str(prog_path)), examples=shlex.quote(str(ex_path)))
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.task.base_dir, timeout=3600)
        if proc.returncode != 0:
            err = f"evaluator exited {proc.returncode}: {proc.stderr[-300:]}"
            return [ExampleResult(e.id, e.label, None, {}, error=err) for e in batch]
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        per = payload.get("per_example") or {}
        return [ExampleResult(e.id, e.label, None, {}, external_score=float(per.get(e.id, payload["score"]))) for e in batch]

    # ---- GEPAAdapter.make_reflective_dataset --------------------------
    def make_reflective_dataset(self, candidate: dict[str, str], eval_batch: EvaluationBatch, components_to_update: list[str]) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        _, problems, fit_results = self.prepare(candidate)
        traces = [t for t in (eval_batch.trajectories or []) if t.id in self._train_by_id]  # TRAIN only (§8)
        records: list[dict[str, Any]] = [self._summary_record(fit_results or traces, traces, problems)]
        seen = {t.id for t in traces}
        records += [self._example_record(t) for t in traces]
        # held-out fit examples (fit_holdout) reach reflection only through the aggregate summary
        extra = [] if self.task.optimizer.fit_holdout else [r for r in fit_results if not r.correct and r.id not in seen][:MAX_EXTRA_FAILURES]
        records += [self._example_record(r) for r in extra]
        return {comp: records for comp in components_to_update}

    def _summary_record(self, sample: list[ExampleResult], live: list[ExampleResult], problems: list[str]) -> dict[str, Any]:
        labels = self.task.labels
        if problems:
            return {"Inputs": "Program validation", "Generated Outputs": "rejected", "Feedback": "The program is invalid: " + "; ".join(problems)}
        if not sample:
            return {"Inputs": "Program-level summary", "Generated Outputs": "no train outputs", "Feedback": "No train examples were evaluated."}
        m = metrics.compute([r.gold for r in sample], [r.pred for r in sample], [r.dist for r in sample], labels, self.task.primary_metric, self.task.negative_label)
        per_class = "; ".join(f"{lab}: P={v['precision']:.2f} R={v['recall']:.2f} F1={v['f1']:.2f} (n={v['support']})" for lab, v in m["per_class"].items())
        clusters = Counter((r.gold, r.pred) for r in sample if not r.correct).most_common(3)
        cluster_txt = "; ".join(f"gold {g} -> predicted {p}: {n}" for (g, p), n in clusters) or "none"
        conf = [max(r.dist.values()) for r in sample if r.dist]
        cost_src = live or sample
        outputs = (
            f"{self.task.primary_metric}={m['primary']:.3f}, accuracy={m['accuracy']:.3f}, macro_f1={m['macro_f1']:.3f} "
            f"on {m['n']} train examples. Per class: {per_class}."
        )
        matrix = m["confusion"]["matrix"]
        confusion = "; ".join(f"gold {lab}: " + ", ".join(f"{p}={matrix[i][j]}" for j, p in enumerate(labels)) for i, lab in enumerate(labels))
        feedback = (
            f"Confusion (gold -> predicted counts): {confusion}. Largest error clusters: {cluster_txt}. Calibration: mean top-label confidence {np.mean(conf):.2f} vs accuracy "
            f"{m['accuracy']:.2f}, Brier {m['brier']:.3f}, ECE {m['ece']:.3f}. Cost: {np.mean([r.calls for r in cost_src]):.2f} Jev calls, "
            f"{np.mean([r.input_tokens for r in cost_src]):.0f} input tokens, {np.mean([r.latency_s for r in cost_src]):.2f}s per example."
        )
        return {"Inputs": f"Program-level summary over {m['n']} train examples", "Generated Outputs": outputs, "Feedback": feedback}

    def _example_record(self, r: ExampleResult) -> dict[str, Any]:
        ex = self._train_by_id.get(r.id)
        inputs = {k: _clip(v) for k, v in ex.fields.items()} if ex else {"id": r.id}
        generated: dict[str, Any] = {"predicted": r.pred, "probabilities": {k: round(v, 3) for k, v in r.dist.items()}}
        if r.derived:
            generated["stage outputs"] = dict(r.derived)
        if r.trace:
            generated["decisions"] = r.trace
        if r.error:
            feedback = f"Failed: {r.error}"
        elif r.correct:
            feedback = f"Correct ({r.gold}, p={r.dist.get(r.gold, 0):.2f})."
        else:
            feedback = f"Incorrect: the gold label is {r.gold}; predicted {r.pred} (p={r.dist.get(r.pred or '', 0):.2f}, gold p={r.dist.get(r.gold, 0):.2f})."
        return {"Inputs": inputs, "Generated Outputs": generated, "Feedback": feedback}

    # ---- jev-program-opt proposer (§5; documented deviation §12.1) ----
    def _propose_program(self, candidate: dict[str, str], reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]], components_to_update: list[str]) -> dict[str, str]:
        parent = self.fitted_program(candidate)
        if parent is None:
            parent = self.seed.model_copy(deep=True)
        records = list(reflective_dataset.get(PROGRAM_COMPONENT, []))
        key = candidate_key(candidate)
        visit = self._visits.get(key, 0)
        self._visits[key] = visit + 1
        det = moves(parent, self.task) + self._static_demo_moves(parent, candidate)
        menu: list[Any] = ["text", "structural", *det]
        for step in range(len(menu)):
            choice = menu[(visit + step) % len(menu)]
            if choice == "text":
                proposal = self._llm_proposal(TEXT_EDIT_TEMPLATE, self.reflection_lm, parent, records, keep_structure=True)
            elif choice == "structural":
                proposal = self._llm_proposal(STRUCTURAL_TEMPLATE, self.structural_lm, parent, records, keep_structure=False)
            else:
                proposal = (choice.program, f"{choice.name}: {choice.rationale}")
            if proposal is None:
                continue
            program, rationale = proposal
            new = {PROGRAM_COMPONENT: program.to_yaml()}
            new_key = candidate_key(new)
            if new_key == key or new_key in self._prepared:
                continue
            self.rationales[new_key] = rationale or (f"{choice} proposal" if isinstance(choice, str) else choice.name)
            return new
        return dict(candidate)

    def _static_demo_moves(self, parent: Program, candidate: dict[str, str]) -> list[Move]:
        """Data-aware demo operators: static ids from misclassified TRAIN fit examples (§4.1)."""
        failures = [r for r in self.prepare(candidate)[2] if not r.correct and r.id in self._train_ids]
        d = parent.demos
        specs: list[tuple[str, Demos, str]] = []
        if d is None or d.mode == "knn":
            by_gold: dict[str, list[str]] = {}
            for r in failures:
                by_gold.setdefault(r.gold, []).append(r.id)
            ids = [i for depth in range(8) for group in by_gold.values() if depth < len(group) for i in [group[depth]]][:8]
            if ids:
                specs.append(("demos_static:failures", Demos(mode="static", ids=ids), "show misclassified train examples (class-balanced) as fixed demonstrations"))
        else:
            new = next((r.id for r in failures if r.id not in d.ids), None)
            if new is not None and len(d.ids) < 16:
                specs.append((f"demos_add:{new}", d.model_copy(update={"ids": [*d.ids, new]}), "add a misclassified train example to the demonstrations"))
            if len(d.ids) > 1:
                specs.append((f"demos_drop:{d.ids[0]}", d.model_copy(update={"ids": d.ids[1:]}), "drop the oldest demonstration"))
        out: list[Move] = []
        for name, spec, why in specs:
            try:
                out.append(Move(name, set_demos(parent, spec).check(self.task), why))
            except (TransformError, ProgramError, ValueError):
                continue
        return out

    def _llm_proposal(self, template: str, lm: LM | None, parent: Program, records: list[Mapping[str, Any]], keep_structure: bool) -> tuple[Program, str] | None:
        if lm is None:
            return None
        limit = self.task.optimizer.max_component_chars
        if limit:
            template += length_rule(limit)
        prompt = render_prompt(template, parent.to_yaml(), list(records))
        for attempt in range(2):  # one retry with the validation error (§5)
            reply = lm(prompt)
            text, rationale = parse_proposal(reply)
            try:
                program = parse_program(text).check(self.task)
                if keep_structure and structure_signature(program) != structure_signature(parent):
                    raise ProgramError(["text edits must keep ids, kinds, options, targets, state lists and numbers unchanged"])
                if limit:
                    allowed = self._seed_texts | set(text_slots(parent).values())
                    if attempt == 1:  # last attempt: trim over-long fields instead of rejecting
                        program = trim_slots(program, limit, allowed)
                    too_long = slot_length_problems(program, limit, allowed)
                    if too_long:
                        raise ProgramError(too_long[:5])
                problems = self.guard.check_program(program, self._train_ids)
                if problems:
                    raise ProgramError(problems)
                return program, rationale
            except (ProgramError, ValidationError, yaml.YAMLError, ValueError, TypeError) as exc:
                if not isinstance(prompt, str):
                    return None
                prompt = f"{prompt}\n\nYour previous reply was rejected: {str(exc)[:600]}\nReply again with a corrected program."
        return None

    # ---- jev-text-opt length-constrained proposer (GEPA custom_candidate_proposer) ----
    def _propose_text(self, candidate: dict[str, str], reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]], components_to_update: list[str]) -> dict[str, str]:
        """GEPA's default instruction proposal plus a hard per-slot length budget: the limit is
        stated in the prompt; an over-long reply is retried once, then trimmed."""
        limit = self.task.optimizer.max_component_chars
        assert limit and self.reflection_lm is not None
        out: dict[str, str] = {}
        notes: list[str] = []
        for name in components_to_update:
            current = candidate[name]
            template = InstructionProposalSignature.default_prompt_template + (
                f"\n\nHard limit: the new instruction must be at most {limit} characters (the current one has {len(current)}). "
                "Prefer concise, general rules over lists of special cases."
            )
            prompt = InstructionProposalSignature.prompt_renderer({"current_instruction_doc": current, "dataset_with_feedback": list(reflective_dataset.get(name, [])), "prompt_template": template})
            new = InstructionProposalSignature.output_extractor(self.reflection_lm(prompt))["new_instruction"]
            if len(new) > limit and isinstance(prompt, str):
                retry = f"{prompt}\n\nYour previous instruction had {len(new)} characters, over the {limit}-character limit. Rewrite it within the limit."
                new = InstructionProposalSignature.output_extractor(self.reflection_lm(retry))["new_instruction"]
            if len(new) > limit:
                notes.append(f"{name} trimmed from {len(new)} chars")
                new = trim_text(new, limit)
            out[name] = new
        changed = ", ".join(components_to_update) or "no slot"
        self.rationales[candidate_key({**candidate, **out})] = f"length-constrained reflective edit of {changed}" + (f" ({'; '.join(notes)})" if notes else "")
        return out

    # ---- GEPA adapter-state persistence (resume) -------------------------
    def get_adapter_state(self) -> dict[str, Any]:
        return {"rationales": dict(self.rationales), "visits": dict(self._visits)}

    def set_adapter_state(self, state: dict[str, Any]) -> None:
        self.rationales.update(state.get("rationales", {}))
        self._visits.update(state.get("visits", {}))


def _redecode_or_raw(program: Program, ex: Example, r: ExampleResult) -> ExampleResult:
    """Fitted re-decoding of a fit-set result; the raw result if it failed or needs an unrecorded answer."""
    if r.error is not None:
        return r
    try:
        return redecode(program, ex, r.memo, r.demos)
    except MissingAnswer:
        return r


def _static_calls(program: Program) -> int:
    nodes = [n for n in program.nodes.values() if isinstance(n, QUESTION_KINDS)]
    if program.shared_state:
        return len({tuple(n.state) for n in nodes})
    return len(nodes)


def _usage(results: list[ExampleResult]) -> dict[str, Any]:
    return {
        "calls": sum(r.calls for r in results),
        "input_tokens": sum(r.input_tokens for r in results),
        "cost": round(sum(r.cost for r in results), 8),
        "latency_s": round(sum(r.latency_s for r in results), 3),
        "errors": sum(r.error is not None for r in results),
    }


class ProgressivePolicy(EvaluationPolicy):
    """GEPA `EvaluationPolicy`: nested stratified val stages (e.g. 32 -> 128 -> full).

    A new candidate is scored on the first stage; it is promoted to the next stage only
    while its mean stays within the current best's 95% bootstrap interval on the same ids.
    Promotion runs inside `get_valset_score`, which GEPA 0.1.4 calls right after adding
    the candidate; extra evaluations are charged to GEPA's metric-call budget.
    """

    def __init__(self, valset: list[Example], stages: list[int], evaluate: Callable[[list[Example], dict[str, str]], EvaluationBatch], seed: int = 0, resamples: int = 1000):
        self.valset = valset
        self.order = stratified_order(valset, seed)
        n = len(valset)
        self.stages = sorted({s for s in stages if 0 < s < n}) + [n]
        self.evaluate = evaluate
        self.rng = np.random.default_rng(seed)
        self.resamples = resamples
        self._pending: int | None = None
        self.promotions: list[dict[str, Any]] = []
        self.scores: dict[int, dict[int, float]] = {}  # per-candidate val scores after promotion

    def get_eval_batch(self, loader: Any, state: GEPAState, target_program_idx: int | None = None) -> list[int]:
        self._pending = len(state.program_candidates)  # the candidate GEPA is about to add
        return self.order[: self.stages[0]]

    def get_best_program(self, state: GEPAState) -> int:
        n = len(self.valset)
        best, best_score = 0, float("-inf")
        for idx, scores in enumerate(state.prog_candidate_val_subscores):
            if len(scores) == n:
                mean = sum(scores.values()) / n
                if mean > best_score:
                    best, best_score = idx, mean
        return best

    def get_valset_score(self, program_idx: int, state: GEPAState) -> float:
        if program_idx == self._pending:
            self._pending = None
            self._promote(program_idx, state)
            self.scores[program_idx] = dict(state.prog_candidate_val_subscores[program_idx])
        scores = state.prog_candidate_val_subscores[program_idx]
        return sum(scores.values()) / len(scores) if scores else float("-inf")

    def _within_best(self, cand: dict[int, float], best: dict[int, float], ids: list[int]) -> bool:
        best_vals = np.array([best[i] for i in ids])
        means = best_vals[self.rng.integers(0, len(ids), size=(self.resamples, len(ids)))].mean(axis=1)
        return float(np.mean([cand[i] for i in ids])) >= float(np.percentile(means, 2.5))

    def _promote(self, idx: int, state: GEPAState) -> None:
        scores = state.prog_candidate_val_subscores[idx]
        best_idx = self.get_best_program(state)
        if best_idx == idx:
            return
        best = state.prog_candidate_val_subscores[best_idx]
        for stage in self.stages[1:]:
            ids = list(scores)
            if not self._within_best(scores, best, ids):
                self.promotions.append({"candidate": idx, "stopped_at": len(ids), "best": best_idx})
                return
            extra = [i for i in self.order[:stage] if i not in scores]
            batch = self.evaluate([self.valset[i] for i in extra], state.program_candidates[idx])
            state.increment_evals(len(extra))
            old_n = len(scores)
            agg = dict(state.prog_candidate_objective_scores[idx])
            for j, vid in enumerate(extra):
                scores[vid] = batch.scores[j]
                state._update_pareto_front_for_val_id(vid, batch.scores[j], idx, batch.outputs[j], None, state.i + 1)
            if batch.objective_scores:
                for obj in agg:
                    new_sum = sum(o[obj] for o in batch.objective_scores)
                    agg[obj] = (agg[obj] * old_n + new_sum) / (old_n + len(extra))
                state.prog_candidate_objective_scores[idx] = agg
                _rebuild_objective_front(state)
        self.promotions.append({"candidate": idx, "stopped_at": len(scores), "best": best_idx})


def _rebuild_objective_front(state: GEPAState) -> None:
    front: dict[str, float] = {}
    members: dict[str, set[int]] = {}
    for idx, agg in enumerate(state.prog_candidate_objective_scores):
        for obj, value in agg.items():
            if value > front.get(obj, float("-inf")):
                front[obj], members[obj] = value, {idx}
            elif value == front[obj]:
                members[obj].add(idx)
    state.objective_pareto_front = front
    state.program_at_pareto_front_objectives = members
