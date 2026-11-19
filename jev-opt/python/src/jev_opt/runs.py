"""Run directories, conditions, representatives and reports (architecture §5, §10, §11).

GEPA runs each optimizing condition (`<condition>/gepa/` is its run_dir, so GEPA owns
checkpoint/resume); this module only wires inputs, records outputs and summarizes.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gepa
import numpy as np
import yaml

from . import __version__, metrics
from .adapter import PROGRAM_COMPONENT, JevAdapter, ProgressivePolicy, candidate_key, cv_spec, train_roles
from .demos import DemoPool, Embedder, SentenceEmbedder
from .execute import ExampleResult, combine, run_decider, run_examples
from .fit import fit_label_bias, fit_ordinal_cutoffs, fit_program, fit_temperature, readout_nodes, with_readouts
from .hygiene import ProgramGuard, SplitAccessError, consumed, fingerprint, load_split, record_consumption, sha256_file, split_report, allow_test_split
from .ir import Ensemble, Program, apply_slots, load_decider, text_slots
from .jev import JevClient, ResponseLedger, make_client
from .population import PopulationSearch, Variant, load_variants, route
from .reflection import OMPReflectionLM
from .tailored import ContinuousAcceptance, JevCostStopper, StratifiedMinibatchSampler, TailoredAdapter
from .task import CONDITIONS, DEFAULT_CONDITIONS, Example, TaskConfig, load_task, seed_program, task_from_data

Notify = Callable[[str, dict[str, Any]], None]
LMFactory = Callable[[str], OMPReflectionLM | None]  # run_dir -> reflection-role callable
REPRESENTATIVES = ("best-quality", "best-cheap", "best-calibrated", "top-quality")
FORBIDDEN_MODULES = ("litellm", "openai", "anthropic", "dspy")
BOOTSTRAP = 2000


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _append(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    line = json.dumps({**row, "at": now()}, ensure_ascii=False, default=str)
    with lock, open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


class FileLogger:
    """GEPA LoggerProtocol writing to run_log.txt only (GEPA's Logger would tee process stdout)."""

    def __init__(self, path: Path):
        self.path = path

    def log(self, message: str) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(f"{message}\n")


@dataclass
class Run:
    run_dir: Path
    config: TaskConfig
    notify: Notify
    lm_factory: LMFactory
    lock: threading.Lock = field(default_factory=threading.Lock)
    cancelled: threading.Event = field(default_factory=threading.Event)
    active_gepa_dir: Path | None = None
    lms: list[OMPReflectionLM] = field(default_factory=list)

    @property
    def run_id(self) -> str:
        return self.run_dir.name

    def record_eval(self, row: dict[str, Any]) -> None:
        _append(self.run_dir / "evaluations.jsonl", row, self.lock)
        if row.get("calls"):
            usage = {k: row.get(k) for k in ("condition", "calls", "input_tokens", "cost")}
            _append(self.run_dir / "usage.jsonl", {"source": "jev", **usage, "split": row.get("split"), "stage": row.get("stage")}, self.lock)

    def progress(self, event: str, message: str, condition: str | None = None, **extra: Any) -> None:
        self.notify("progress", {"run_id": self.run_id, "run_dir": str(self.run_dir), "condition": condition, "event": event, "message": message, **extra})

    def cancel(self) -> None:
        self.cancelled.set()
        if self.active_gepa_dir is not None:
            self.active_gepa_dir.mkdir(parents=True, exist_ok=True)
            (self.active_gepa_dir / "gepa.stop").write_text(now())


ACTIVE: dict[str, Run] = {}


def cancel_all() -> list[str]:
    for run in list(ACTIVE.values()):
        run.cancel()
    return list(ACTIVE)


# ---- run state ----------------------------------------------------------------

def _read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.exists() else default


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    tmp.replace(path)


def _update_run(run_dir: Path, **changes: Any) -> dict[str, Any]:
    state = _read_json(run_dir / "run.json", {})
    for key, value in changes.items():
        if key == "conditions":
            state.setdefault("conditions", {}).update(value)
        else:
            state[key] = value
    _write_json(run_dir / "run.json", state)
    return state


def _git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except OSError:
        return None


def environment(config: TaskConfig, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    packages = {}
    for name in ("gepa", "pydantic", "pyyaml", "httpx", "numpy", "scipy", "scikit-learn"):
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:
            packages[name] = None
    return {
        "jev_opt": __version__,
        "jev_opt_git_sha": _git_sha(),
        "gepa": packages["gepa"],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "jev_model": config.jev.model,
        "optimizer": config.optimizer.model_dump(include={"model", "reflection_model", "structural_model", "thinking"}),
        "seeds": {"optimizer": config.optimizer.seed},
        **(extra or {}),
    }


# ---- optimize -----------------------------------------------------------------

def make_pool(config: TaskConfig, train: list[Example], runs_root: Path, embedder: Embedder | None = None) -> DemoPool:
    """Demonstration pool over the TRAIN split; the embedder loads its model only if knn demos run."""
    if embedder is None:
        emb = config.embedding
        cache = Path(emb.cache_dir) if emb.cache_dir else runs_root / ".embed-cache"
        embedder = SentenceEmbedder(cache, emb.model, emb.revision)
    return DemoPool(train, embedder)


def optimize(params: dict[str, Any], notify: Notify, lm_factory: LMFactory, omp_version: str | None = None, client: JevClient | None = None, embedder: Embedder | None = None) -> dict[str, Any]:
    runs_root = Path(params["runs_root"]).resolve()
    resume = bool(params.get("resume"))
    if resume:
        run_dir = runs_root / params["run_id"]
        if not (run_dir / "config.yaml").exists():
            raise FileNotFoundError(f"no run at {run_dir}")
        record = consumed(run_dir)
        if record:
            raise SplitAccessError(f"run {run_dir.name} consumed its test split at {record['consumed_at']}; resume is refused")
        config = task_from_data(yaml.safe_load((run_dir / "config.yaml").read_text())["task"], run_dir)
        seed = Program.from_yaml((run_dir / "seed.yaml").read_text())
        conditions = _read_json(run_dir / "run.json", {}).get("requested", list(DEFAULT_CONDITIONS))
    else:
        config = load_task(params["task_path"])
        if params.get("seed") is not None:  # extra seeds of one configuration: same task file, new run id
            config.optimizer.seed = int(params["seed"])
        run_id = params.get("run_id") or f"{config.name}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        run_dir = runs_root / run_id
        if run_dir.exists():
            raise FileExistsError(f"run {run_id} exists; pass resume: true to continue it")
        run_dir.mkdir(parents=True)
        seed = seed_program(config).check(config)
        conditions = list(params.get("conditions") or DEFAULT_CONDITIONS)
        unknown = [c for c in conditions if c not in CONDITIONS]
        if unknown:
            raise ValueError(f"unknown conditions {unknown}")
        (run_dir / "seed.yaml").write_text(seed.to_yaml())
        resolved = {"task": config.resolved(), "conditions": conditions, "test_fingerprint": fingerprint(config.split_path("test"))}
        (run_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))
        _update_run(run_dir, run_id=run_id, task=config.name, requested=conditions, started_at=now(), conditions={c: {"status": "pending"} for c in conditions})
        _write_json(run_dir / "environment.json", environment(config, {"omp": omp_version}))

    run = Run(run_dir, config, notify, lm_factory)
    ACTIVE[str(run_dir)] = run
    _update_run(run_dir, state="running", pid=os.getpid(), resumed_at=now() if resume else None)
    run.progress("run_started", f"{'resumed' if resume else 'started'} {run.run_id}", optimizer=config.optimizer.model_dump(include={"model", "reflection_model", "structural_model", "thinking"}))
    own_client = client is None
    ledger = ResponseLedger(run_dir)
    try:
        train, val = load_split(config, "train"), load_split(config, "val")
        split_info = split_report(train, val)
        cache_dir = Path(config.jev.cache_dir) if config.jev.cache_dir else run_dir.parent / ".jev-cache"
        client = client or make_client(cache_dir, config.jev.model, config.jev.max_rpm, config.jev.concurrency, [config.base_dir, Path.cwd()])
        client.ledger = ledger
        guard = ProgramGuard(train + val, seed.to_yaml(), config.optimizer.max_program_chars)
        pool = make_pool(config, train, runs_root, embedder)
        summaries: dict[str, Any] = {}
        for condition in conditions:
            status = _read_json(run_dir / "run.json", {}).get("conditions", {}).get(condition, {})
            if status.get("status") == "finished":
                summaries[condition] = status.get("summary", {})
                continue
            if run.cancelled.is_set():
                break
            _update_run(run_dir, conditions={condition: {"status": "running"}})
            run.progress("condition_started", f"{condition}: started", condition)
            before = (client.usage.fresh_calls, client.usage.fresh_input_tokens, client.usage.fresh_cost)
            summary = run_condition(run, condition, seed, train, val, client, guard, pool)
            summary["jev_fresh_session"] = {"calls": client.usage.fresh_calls - before[0], "input_tokens": client.usage.fresh_input_tokens - before[1], "cost": round(client.usage.fresh_cost - before[2], 6)}
            state = "cancelled" if run.cancelled.is_set() else "finished"
            _update_run(run_dir, conditions={condition: {"status": state, "summary": summary}})
            summaries[condition] = summary
            run.progress("condition_finished", f"{condition}: {state}; best val {summary.get('best_val_quality')}", condition, summary=summary)
        reps = write_representatives(run_dir, config, val, client, pool, train)
        final = "cancelled" if run.cancelled.is_set() else "finished"
        env = _read_json(run_dir / "environment.json", {})
        env.update({
            "jev_snapshots": sorted(client.snapshots | set(env.get("jev_snapshots", []))),
            "optimizer_models_seen": sorted({m for lm in run.lms for m in lm.models_seen()} | set(env.get("optimizer_models_seen", []))),
            "forbidden_modules_loaded": sorted(m for m in FORBIDDEN_MODULES if m in sys.modules),
            "splits": split_info,
        })
        _write_json(run_dir / "environment.json", env)
        _update_run(run_dir, state=final, finished_at=now())
        write_report(run_dir)
        run.progress("run_finished", f"{run.run_id}: {final}; best-quality {reps.get('best-quality', {}).get('candidate')}")
        return {"run_id": run.run_id, "run_dir": str(run_dir), "state": final, "conditions": summaries, "representatives": reps}
    except BaseException as exc:
        _update_run(run_dir, state="failed", error=f"{type(exc).__name__}: {exc}")
        run.progress("error", f"{run.run_id}: {type(exc).__name__}: {exc}")
        raise
    finally:
        ACTIVE.pop(str(run_dir), None)
        if client is not None:
            client.ledger = None
            if own_client:
                client.close()
        _update_run(run_dir, jev_responses=ledger.finalize())


class ProgressCallback:
    """GEPA callbacks -> candidate files + host `progress` notifications.

    `offset` shifts GEPA's candidate indexes (and parents) when one condition holds several GEPA
    searches (`jev-tailored` starts); `extra` is merged into every candidate's metadata."""

    def __init__(self, run: Run, condition: str, adapter: JevAdapter, policy: ProgressivePolicy, cand_dir: Path, seed_candidate: dict[str, str], offset: int = 0, extra: dict[str, Any] | None = None):
        self.run, self.condition, self.adapter, self.policy, self.cand_dir = run, condition, adapter, policy, cand_dir
        self.seed_candidate = seed_candidate
        self.offset, self.extra = offset, extra
        self.candidates: dict[int, dict[str, str]] = {}

    def on_valset_evaluated(self, event: dict[str, Any]) -> None:
        idx = event["candidate_idx"]
        self.candidates[idx] = event["candidate"]
        parents = list(event["parent_ids"]) or [None]
        parent_cands = [self.candidates[p] for p in parents if p in self.candidates]
        # Scores after any progressive promotion (GEPA's event carries the first-stage scores only).
        scores = self.policy.scores.get(idx, dict(event["scores_by_val_id"]))
        shifted = [p + self.offset if p is not None else None for p in parents]
        _write_candidate(self.cand_dir, idx + self.offset, self.adapter, event["candidate"], shifted, scores, None, event["total_valset_size"], self.seed_candidate, parent_cands, extra=self.extra)
        tag = " (new best)" if event["is_best_program"] and idx else ""
        self.run.progress(
            "new_best" if tag else "iteration",
            f"{self.condition}: candidate {idx + self.offset} val {event['average_score']:.3f} on {len(scores)}/{event['total_valset_size']}{tag}",
            self.condition,
            iteration=event["iteration"],
            best=idx + self.offset if tag else None,
        )

    def on_error(self, event: dict[str, Any]) -> None:
        self.run.progress("error", f"{self.condition}: {event['exception']}", self.condition)


def _rationale(adapter: JevAdapter, candidate: dict[str, str], parents: list[int | None], seed_candidate: dict[str, str], parent_cands: list[dict[str, str]] | None) -> str:
    key = candidate_key(candidate)
    if key in adapter.rationales:
        return adapter.rationales[key]
    if candidate == seed_candidate:
        return "seed"
    if len([p for p in parents if p is not None]) > 1:
        return f"GEPA merge of candidates {parents}"
    if adapter.mode == "text" and parent_cands:
        changed = sorted(k for k in candidate if candidate[k] != parent_cands[0].get(k))
        return f"GEPA reflective edit of {', '.join(changed) or 'no slot'}"
    return "GEPA reflective proposal"


def _write_candidate(cand_dir: Path, idx: int, adapter: JevAdapter, candidate: dict[str, str], parents: list[int | None], val_scores: dict[Any, float], objectives: dict[str, float] | None, val_n: int, seed_candidate: dict[str, str], parent_cands: list[dict[str, str]] | None = None, extra: dict[str, Any] | None = None) -> None:
    program = adapter.fitted_program(candidate)
    key = candidate_key(candidate)
    meta = _read_json(cand_dir / f"{idx}.json", {})
    rationale = _rationale(adapter, candidate, parents, seed_candidate, parent_cands)
    meta.update({
        "index": idx,
        "key": key,
        "parents": parents,
        "valid": program is not None,
        "problems": adapter.prepare(candidate)[1],
        "rationale": rationale,
        "val_mean_quality": float(np.mean(list(val_scores.values()))) if val_scores else None,
        "coverage": len(val_scores),
        "val_n": val_n,
        "val_scores": {str(k): v for k, v in val_scores.items()},
        "chars": len(program.to_yaml()) if program else None,
    })
    if objectives is not None:
        meta["objectives"] = objectives
    meta.update(extra or {})
    (cand_dir / f"{idx}.yaml").write_text(f"# rationale: {rationale}\n" + (program.to_yaml() if program else candidate.get("program", json.dumps(candidate, indent=2))))
    _write_json(cand_dir / f"{idx}.json", meta)


def run_condition(run: Run, condition: str, seed: Program, train: list[Example], val: list[Example], client: JevClient, guard: ProgramGuard, pool: DemoPool | None = None) -> dict[str, Any]:
    config = run.config
    opt = config.optimizer
    cond_dir = run.run_dir / condition
    cand_dir = cond_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    mode = "program" if condition == "jev-program-opt" else "text"
    reflection = run.lm_factory(str(run.run_dir))
    structural = reflection.with_role("structural") if reflection is not None and mode == "program" else None
    if reflection is not None:
        run.lms.append(reflection)
    adapter = JevAdapter(task=config, seed=seed, client=client, train=train, guard=guard, mode=mode, condition=condition, record=run.record_eval, reflection_lm=reflection, structural_lm=structural, pool=pool)
    seed_candidate = adapter.seed_candidate()

    if condition == "jev-population-opt":
        return run_population(run, condition, seed, train, val, client, guard, pool, cand_dir, reflection)
    if condition == "jev-tailored":
        return run_tailored(run, condition, seed, train, val, client, guard, pool, cand_dir, reflection)
    if condition == "jev-zero":
        batch = adapter.evaluate(val, seed_candidate)
        scores = {i: s for i, s in enumerate(batch.scores)}
        objectives = {k: float(np.mean([o[k] for o in batch.objective_scores])) for k in batch.objective_scores[0]} if batch.objective_scores else {}
        _write_candidate(cand_dir, 0, adapter, seed_candidate, [None], scores, objectives, len(val), seed_candidate)
        return {"candidates": 1, "best_candidate": 0, "best_val_quality": float(np.mean(batch.scores)), "seed_val_quality": float(np.mean(batch.scores)), "metric_calls": len(val)}

    gepa_dir = cond_dir / "gepa"
    gepa_dir.mkdir(parents=True, exist_ok=True)
    (gepa_dir / "gepa.stop").unlink(missing_ok=True)  # stale stop file from a cancelled attempt
    run.active_gepa_dir = gepa_dir
    policy = ProgressivePolicy(val, opt.progressive, lambda b, c: adapter.evaluate(b, c), seed=opt.seed)
    uniform = all(ex.weight == 1.0 for ex in train + val)
    try:
        result = gepa.optimize(
            seed_candidate=seed_candidate,
            trainset=adapter.trainset,  # reflection pool (§8): train only, optionally capped / fit-held-out
            valset=val,
            adapter=adapter,
            custom_candidate_proposer=adapter.text_proposer,  # text mode with max_component_chars; else GEPA default
            reflection_lm=reflection,  # program mode: proposals go through adapter.propose_new_texts; kept for the cost stopper
            candidate_selection_strategy="pareto",
            frontier_type="hybrid",
            skip_perfect_score=uniform,
            reflection_minibatch_size=opt.reflection_minibatch_size,
            module_selector="round_robin",
            use_merge=opt.use_merge,
            max_metric_calls=opt.max_metric_calls,
            max_reflection_cost=opt.max_reflection_cost,
            logger=FileLogger(gepa_dir / "run_log.txt"),
            run_dir=str(gepa_dir),
            callbacks=[ProgressCallback(run, condition, adapter, policy, cand_dir, seed_candidate)],
            track_best_outputs=False,
            display_progress_bar=False,
            cache_evaluation=False,  # §9: 0.1.4 cache keys collide across train/val ids
            seed=opt.seed,
            raise_on_exception=True,
            val_evaluation_policy=policy,
        )
    finally:
        run.active_gepa_dir = None
    for idx, cand in enumerate(result.candidates):
        parents = list(result.parents[idx])
        parent_cands = [result.candidates[p] for p in parents if p is not None]
        objectives = (result.val_aggregate_subscores or [{}] * len(result.candidates))[idx]
        _write_candidate(cand_dir, idx, adapter, cand, parents, result.val_subscores[idx], objectives, len(val), seed_candidate, parent_cands)
    best = _best_full(result, len(val))
    _write_json(cond_dir / "promotions.json", policy.promotions)
    return {
        "candidates": result.num_candidates,
        "best_candidate": best,
        "best_val_quality": result.val_aggregate_scores[best],
        "seed_val_quality": result.val_aggregate_scores[0],
        "metric_calls": result.total_metric_calls,
        "full_val_evals": sum(len(s) == len(val) for s in result.val_subscores),
        "reflection_calls": len(reflection.calls) if reflection is not None else 0,
        "reflection_cost": round(reflection.total_cost, 6) if reflection is not None else 0.0,
    }


def _best_full(result: Any, n: int) -> int:
    full = [i for i, s in enumerate(result.val_subscores) if len(s) == n]
    return max(full, key=lambda i: result.val_aggregate_scores[i]) if full else result.best_idx


def run_tailored(run: Run, condition: str, seed: Program, train: list[Example], val: list[Example], client: JevClient, guard: ProgramGuard, pool: DemoPool | None, cand_dir: Path, reflection: OMPReflectionLM | None) -> dict[str, Any]:
    """`jev-tailored` (§5.3; deviation §12.23): one GEPA search per start (`tailored.starts`, in order) with
    the hooks of `jev_opt.tailored`. The run's OMP budget (`max_reflection_cost`) and fresh Jev cap
    (`tailored.max_jev_cost`) are split evenly over the starts still to run. Afterwards every fully evaluated
    candidate is also scored exactly as written (tags `tailored-zero`, `tailored-zero-<start>`) and, with
    `tailored.head`, each start's best fitted candidate by val with a readout head (`tailored-head`).
    Resumable: `starts.json` records finished starts; an unfinished start resumes its GEPA checkpoint."""
    if reflection is None:
        raise RuntimeError("jev-tailored needs the OMP reflection model")
    config = run.config
    opt = config.optimizer
    cfg = opt.tailored
    metric, labels, negative = config.primary_metric, config.labels, config.negative_label
    cond_dir = cand_dir.parent
    seeds_dir = cond_dir / "seeds"
    seeds_dir.mkdir(exist_ok=True)
    state_path = cond_dir / "starts.json"
    saved = _read_json(state_path, {})
    starts: dict[str, dict[str, Any]] = saved.setdefault("starts", {})
    fresh0 = client.usage.fresh_cost - float(saved.get("jev_fresh_cost", 0.0))
    fit_set, _ = train_roles(train, opt)
    ctx = {"split": "train", "condition": condition}
    uniform = all(ex.weight == 1.0 for ex in train + val)
    primary = lambda res: metrics.primary(metric, [r.gold for r in res], [r.pred for r in res], labels, negative)  # noqa: E731

    def persist() -> None:
        saved["jev_fresh_cost"] = round(client.usage.fresh_cost - fresh0, 6)
        _write_json(state_path, saved)

    def seed_of(start: str, info: dict[str, Any]) -> tuple[str, Program]:
        snap = seeds_dir / f"{start}.yaml"  # snapshot: resume does not depend on the seed files
        if snap.exists():
            return info.get("seed", start), Program.from_yaml(snap.read_text()).check(config)
        if start == "paper":
            name, program = "paper", seed
        else:  # the kind's seed with the best zero-shot score on the train fit slice (as written)
            scored: list[tuple[float, str, Program]] = []
            for ref in (r for r in cfg.seeds if r.kind == start):
                stem = Path(ref.path).stem
                p = Program.from_yaml((config.base_dir / ref.path).read_text()).check(config)
                scored.append((primary(run_examples(p, fit_set, client, context=ctx, pool=pool)), stem if stem.startswith(ref.kind) else f"{ref.kind}-{stem}", p))
            info["train_zero"] = {n: s for s, n, _ in scored}
            _, name, program = max(scored, key=lambda x: x[0])  # ties: the first listed
        info["seed"] = name
        snap.write_text(program.to_yaml())
        persist()
        return name, program

    for j, start in enumerate(cfg.starts):
        info = starts.setdefault(start, {})
        if info.get("status") == "finished":
            continue
        if run.cancelled.is_set():
            break
        name, program = seed_of(start, info)
        offset = sum(len(starts[s].get("indices", [])) for s in cfg.starts[:j])
        if "jev_cap" not in info:  # this start's share of what is left, fixed when it first begins
            remaining = len(cfg.starts) - j
            left_jev = None if cfg.max_jev_cost is None else max(cfg.max_jev_cost - (client.usage.fresh_cost - fresh0), 0.0)
            done_omp = sum(float(starts[s].get("omp_cost", 0.0)) for s in cfg.starts[:j])
            info["jev_cap"] = None if left_jev is None else round(left_jev / remaining, 6)
            info["omp_cap"] = None if opt.max_reflection_cost is None else round(max(opt.max_reflection_cost - done_omp, 0.0) / remaining, 6)
            persist()
        prior_jev, prior_omp = float(info.get("jev_cost", 0.0)), float(info.get("omp_cost", 0.0))
        lm = OMPReflectionLM(reflection.rpc, "reflection", reflection.run_dir, reflection.max_tokens, reflection.timeout_s)  # own ledger: per-start cost stopper
        run.lms.append(lm)
        adapter = TailoredAdapter(task=config, seed=program, client=client, train=train, guard=guard, condition=condition, record=run.record_eval, reflection_lm=lm, pool=pool, start=start)
        seed_candidate = adapter.seed_candidate()
        gepa_dir = cond_dir / f"gepa-{start}"
        gepa_dir.mkdir(parents=True, exist_ok=True)
        (gepa_dir / "gepa.stop").unlink(missing_ok=True)
        run.active_gepa_dir = gepa_dir
        policy = ProgressivePolicy(val, opt.progressive, lambda b, c, a=adapter: a.evaluate(b, c), seed=opt.seed)

        def on_check(info: dict[str, Any] = info, lm: OMPReflectionLM = lm, prior_jev: float = prior_jev, prior_omp: float = prior_omp) -> None:
            info["jev_cost"] = round(prior_jev + stopper.spent, 6)
            info["omp_cost"] = round(prior_omp + lm.total_cost, 6)
            persist()

        stopper = JevCostStopper(client, None if info["jev_cap"] is None else info["jev_cap"] - prior_jev, on_check)
        if cfg.continuous_accept:  # T2
            hooks: dict[str, Any] = {"batch_sampler": StratifiedMinibatchSampler(adapter.trainset, cfg.accept_n, metric in ("macro_f1", "balanced_accuracy"), opt.seed),
                                     "reflection_minibatch_size": None, "acceptance_criterion": ContinuousAcceptance(config, cfg.accept_tol, run.record_eval, condition, start)}
        else:  # ablation: GEPA's default sampler and acceptance
            hooks = {"reflection_minibatch_size": opt.reflection_minibatch_size}
        extra = {"start": start, "seed": name, "tags": ["tailored", f"tailored-{start}"]}
        run.progress("iteration", f"{condition}: start {start} (seed {name}); OMP cap {info['omp_cap']}, Jev cap {info['jev_cap']}", condition)
        try:
            result = gepa.optimize(
                seed_candidate=seed_candidate,
                trainset=adapter.trainset,
                valset=val,
                adapter=adapter,
                custom_candidate_proposer=adapter.text_proposer,  # T1: rulebook prompt, joint rewrite of the selected fields
                reflection_lm=lm,  # for GEPA's max_reflection_cost stopper
                candidate_selection_strategy="pareto",
                frontier_type="hybrid",
                skip_perfect_score=uniform,
                module_selector=adapter.select_components,  # T3
                use_merge=opt.use_merge,
                max_metric_calls=opt.max_metric_calls,
                max_reflection_cost=None if info["omp_cap"] is None else max(info["omp_cap"] - prior_omp, 0.0),
                stop_callbacks=[stopper],
                logger=FileLogger(gepa_dir / "run_log.txt"),
                run_dir=str(gepa_dir),
                callbacks=[ProgressCallback(run, condition, adapter, policy, cand_dir, seed_candidate, offset, extra)],
                track_best_outputs=False,
                display_progress_bar=False,
                cache_evaluation=False,  # §9
                seed=opt.seed,
                raise_on_exception=True,
                val_evaluation_policy=policy,
                **hooks,
            )
        finally:
            run.active_gepa_dir = None
            on_check()
        indices = []
        for idx, cand in enumerate(result.candidates):
            parents = [p + offset if p is not None else None for p in result.parents[idx]]
            parent_cands = [result.candidates[p] for p in result.parents[idx] if p is not None]
            objectives = (result.val_aggregate_subscores or [{}] * len(result.candidates))[idx]
            tags = extra["tags"] + ([f"tailored-seedfit-{start}"] if idx == 0 else [])  # the start's seed with its fitted numbers
            _write_candidate(cand_dir, offset + idx, adapter, cand, parents, result.val_subscores[idx], objectives, len(val), seed_candidate, parent_cands, extra={**extra, "tags": tags})
            indices.append(offset + idx)
        best = _best_full(result, len(val))
        _write_json(cond_dir / f"promotions-{start}.json", policy.promotions)
        info.update(indices=indices, summary={
            "seed": name, "candidates": result.num_candidates, "best_candidate": offset + best, "best_val_quality": result.val_aggregate_scores[best],
            "seed_val_quality": result.val_aggregate_scores[0], "metric_calls": result.total_metric_calls,
            "full_val_evals": sum(len(s) == len(val) for s in result.val_subscores), "reflection_calls": len(lm.calls),
            "reflection_cost": info["omp_cost"], "jev_fresh_cost": info["jev_cost"],
        })
        if run.cancelled.is_set():
            persist()
            break
        info["status"] = "finished"
        persist()

    summary: dict[str, Any] = {
        "starts": {s: {k: v for k, v in starts.get(s, {}).items() if k != "indices"} for s in cfg.starts},
        "reflection_cost": round(sum(float(starts.get(s, {}).get("omp_cost", 0.0)) for s in cfg.starts), 6),
        "reflection_calls": sum(int(starts.get(s, {}).get("summary", {}).get("reflection_calls", 0)) for s in cfg.starts),
        "metric_calls": sum(int(starts.get(s, {}).get("summary", {}).get("metric_calls", 0)) for s in cfg.starts),
        "seed_val_quality": starts.get(cfg.starts[0], {}).get("summary", {}).get("seed_val_quality"),
    }
    if run.cancelled.is_set() or any(starts.get(s, {}).get("status") != "finished" for s in cfg.starts):
        return {**summary, "jev_fresh_cost": round(client.usage.fresh_cost - fresh0, 6)}

    # ---- as written (tailored-zero) and with a readout head (tailored-head); val answers are cached
    metas = [json.loads(p.read_text()) for p in sorted(cand_dir.glob("*.json"), key=lambda p: int(p.stem))]
    fitted = [m for m in metas if "tailored" in m.get("tags", []) and m.get("valid") and m.get("coverage") == len(val)]
    idx = max((m["index"] for m in metas if "tailored" in m.get("tags", [])), default=-1) + 1
    val_ctx = {"split": "val", "condition": condition}
    start_seeds = {s: Program.from_yaml((seeds_dir / f"{s}.yaml").read_text()).check(config) for s in cfg.starts}
    best_of: dict[str, tuple[float, dict[str, Any], Program]] = {}
    for m in fitted:
        start = m["start"]
        program = Program.from_yaml((cand_dir / f"{m['index']}.yaml").read_text())
        raw = apply_slots(start_seeds[start], text_slots(program)).check(config)
        score = primary(run_examples(program, val, client, context=val_ctx, pool=pool))
        if start not in best_of or score > best_of[start][0]:
            best_of[start] = (score, m, raw)
        why = f"{condition}/{m['index']} as written, no fitted number ({m['rationale']})"
        tags = ["tailored-zero", f"tailored-zero-{start}"] + ([f"tailored-seed-{start}"] if m["index"] == starts[start]["indices"][0] else [])  # + the seed as written
        _write_plain_candidate(cand_dir, idx, raw, run_examples(raw, val, client, context=val_ctx, pool=pool), val, config, why[:700],
                               {"tags": tags, "start": start, "seed": m.get("seed"), "of": m["index"]})
        idx += 1
    for start, (_, m, raw) in best_of.items() if cfg.head else []:
        headed = with_readouts(raw, metric).check(config)
        if not readout_nodes(headed):
            continue
        pairs = list(zip(fit_set, run_examples(headed, fit_set, client, exhaustive=True, context=ctx, pool=pool)))
        try:
            program, report = fit_program(headed, pairs, metric, opt.calibrate, rounds=2 if opt.fit_parameters else 0, negative=negative, cv=cv_spec(opt))
        except ValueError as exc:  # e.g. a readout whose fit examples hold one class
            run.progress("iteration", f"{condition}: no readout head for {condition}/{m['index']}: {exc}", condition)
            continue
        why = f"{condition}/{m['index']} (best fitted candidate of start {start} by val) with a readout head fit on the train fit slice"
        _write_plain_candidate(cand_dir, idx, program, run_examples(program, val, client, context=val_ctx, pool=pool), val, config, why,
                               {"tags": ["tailored-head", f"tailored-head-{start}"], "start": start, "seed": m.get("seed"), "of": m["index"], "fit": report})
        idx += 1
    best = max(fitted, key=lambda m: m["val_mean_quality"]) if fitted else None
    return {**summary, "candidates": idx, "best_candidate": best["index"] if best else None, "best_val_quality": best["val_mean_quality"] if best else None,
            "jev_fresh_cost": round(client.usage.fresh_cost - fresh0, 6)}


def run_population(run: Run, condition: str, seed: Program, train: list[Example], val: list[Example], client: JevClient, guard: ProgramGuard, pool: DemoPool | None, cand_dir: Path, reflection: OMPReflectionLM | None) -> dict[str, Any]:
    """`jev-population-opt` (§5.1-5.2; deviation §12.19): round 0 = the paper seed and every extra
    seed (`population.seeds`), proposal rounds scored by train CV, then val candidates: each seed
    zero-shot (tags `zero-<seed>`, `zero-<kind>` for the kind's seed chosen by its zero-shot train
    score), each kind's best seed + readout (`head-<kind>`) and best descendant (`search-<kind>`),
    the best programs by train CV (`single`) and stacks of their variants (`stacked`).
    Resumable: finished rounds are reloaded from `population.json` (no LLM calls; the variants'
    answers come from the Jev cache)."""
    config = run.config
    opt = config.optimizer
    cfg = opt.population
    metric, labels, negative = config.primary_metric, config.labels, config.negative_label
    cond_dir = cand_dir.parent
    state_path = cond_dir / "population.json"
    fit_set, _ = train_roles(train, opt)
    extra: list[tuple[str, str, Program]] = []
    (cond_dir / "seeds").mkdir(exist_ok=True)
    for ref in cfg.seeds:  # snapshot the extra seeds into the run (resume does not depend on the files)
        name = Path(ref.path).stem if Path(ref.path).stem.startswith(ref.kind) else f"{ref.kind}-{Path(ref.path).stem}"
        snap = cond_dir / "seeds" / f"{name}.yaml"
        if not snap.exists():
            snap.write_text((config.base_dir / ref.path).read_text())
        extra.append((name, ref.kind, Program.from_yaml(snap.read_text()).check(config)))
    lm = reflection.with_role("structural", max_tokens=cfg.max_tokens) if reflection is not None else None
    search = PopulationSearch(config, seed, client, fit_set, guard, {ex.id for ex in train}, pool, lm, condition, extra_seeds=extra)
    saved = _read_json(state_path, {})
    fresh0 = client.usage.fresh_cost - float(saved.get("jev_fresh_cost", 0.0))
    ctx = {"split": "train", "condition": condition}
    for v in load_variants(saved.get("variants", [])) or search.seed_variants():
        search.variants[v.slot].append(v)
    papers = lambda vs: [v for v in vs if v.origin in ("paper", "paper+demos")]  # noqa: E731
    for s in search.slots:  # requests recorded as such by earlier runs: index them before packing
        for v in papers(search.variants[s]):
            run_examples(search.program(s, [v]), [ex for ex in fit_set if search.routes[ex.id] == s], client, exhaustive=True, context=ctx, pool=pool)
        search.score(s, search.variants[s], fit_set, ctx)
    search.proposals = list(saved.get("proposals", []))

    def persist(rounds_done: int) -> None:
        _write_json(state_path, {
            "slots": search.slots, "rounds_done": rounds_done, "jev_fresh_cost": round(client.usage.fresh_cost - fresh0, 6),
            "variants": [v.to_record() for s in search.slots for v in search.variants[s]], "proposals": search.proposals,
        })

    rounds_done = int(saved.get("rounds_done", 0))
    persist(rounds_done)
    for r in range(rounds_done + 1, cfg.rounds + 1):
        spent = client.usage.fresh_cost - fresh0
        if run.cancelled.is_set() or (cfg.max_jev_cost is not None and spent >= cfg.max_jev_cost):
            run.progress("iteration", f"{condition}: stopping before round {r} (fresh Jev spend ${spent:.2f})", condition)
            break
        ranked = search.ranked_combos(1)
        slot = search.slots[(r - 1) % len(search.slots)]
        before = (client.usage.fresh_calls, client.usage.fresh_input_tokens, client.usage.fresh_cost)
        new = search.propose(slot, r, ranked[0][1]) if ranked else []
        search.score(slot, new, fit_set, ctx)
        search.variants[slot] += new
        rounds_done = r
        persist(r)
        best_cv = search.ranked_combos(1)[0][0] if search.ranked_combos(1) else None
        run.record_eval({"condition": condition, "candidate": f"round-{r}", "split": "train", "stage": f"round-{r}", "n": len(fit_set), "slot": slot, "variants": len(new),
                         "scored": sum(v.cv is not None for v in new), "best_train_cv": best_cv, "fresh_calls": client.usage.fresh_calls - before[0],
                         "fresh_input_tokens": client.usage.fresh_input_tokens - before[1], "fresh_cost": round(client.usage.fresh_cost - before[2], 6)})
        run.progress("iteration", f"{condition}: round {r} ({slot}): {len(new)} variants; best train CV {best_cv}", condition, iteration=r)

    # ---- val candidates (train decides what reaches val; val picks the representatives)
    kinds = list(dict.fromkeys(["paper", *(k for _, k, _ in extra)]))
    entries: dict[tuple, dict[str, Any]] = {}  # combo identity -> {choice, tags, train_cv}

    def add(choice: dict[str, list[Variant]], tag: str, train_cv: float | None) -> None:
        key = tuple(tuple(id(v) for v in choice[s]) for s in search.slots)
        entry = entries.setdefault(key, {"choice": choice, "tags": [], "train_cv": train_cv})
        entry["tags"].append(tag)

    ranked = search.ranked_combos()
    for score, c in ranked[: cfg.val_candidates]:
        add({s: [c[s]] for s in search.slots}, "single", score)
    for kind in kinds:
        for tag, keep in ((f"head-{kind}", lambda v, k=kind: v.round == 0 and v.root == k), (f"search-{kind}", lambda v, k=kind: v.root == k)):
            best = search.ranked_combos(1, keep=keep)
            if best:
                add({s: [best[0][1][s]] for s in search.slots}, tag, best[0][0])
    allowed = {s: list({id(c[s]): c[s] for _, c in ranked[: cfg.val_candidates]}.values()) for s in search.slots}
    if ranked:
        stacks = search.stacks(allowed, ranked[0][1])
        for j in range(2, max(len(m) for m in stacks.values()) + 1):
            choice = {s: stacks[s][:j] for s in search.slots}
            add(choice, "stacked", search.combo_score({s: search.stacked_oof(s, choice[s])[1] if len(choice[s]) > 1 else choice[s][0].oof for s in search.slots}))
    val_ctx = {"split": "val", "condition": condition}
    seeds_all = [("paper", "paper", seed), *extra]
    zero_val: dict[str, list[ExampleResult]] = {"paper": run_examples(seed, val, client, context=val_ctx, pool=pool)}  # recorded requests: indexed
    for s in search.slots:  # val answers of every candidate and seed question: one packed request per example and state
        needed = {id(v): v for e in entries.values() for v in e["choice"][s]}
        needed.update({id(v): v for v in search.variants[s] if v.round == 0 and v.origin not in ("paper", "paper+demos")})
        reached = [ex for ex in val if route(seed, ex.fields) == s]
        for v in papers(needed.values()):
            run_examples(search.program(s, [v]), reached, client, exhaustive=True, context=val_ctx, pool=pool)
        rest = [v for v in needed.values() if v.origin not in ("paper", "paper+demos")]
        if rest:
            run_examples(search.program(s, rest), reached, client, exhaustive=True, context=val_ctx, pool=pool)
    train_zero: dict[str, float] = {}
    for name, _, program in seeds_all:  # zero-shot: each seed exactly as written (no fitted number)
        res = run_examples(program, fit_set, client, context=ctx, pool=pool)
        train_zero[name] = metrics.primary(metric, [r.gold for r in res], [r.pred for r in res], labels, negative)
        if name not in zero_val:
            zero_val[name] = run_examples(program, val, client, context=val_ctx, pool=pool)
    chosen_zero = {kind: max((n for n, k, _ in seeds_all if k == kind), key=lambda n: train_zero[n]) for kind in kinds}

    adapter = JevAdapter(task=config, seed=seed, client=client, train=train, guard=guard, mode="program", condition=condition, record=run.record_eval, pool=pool)
    seed_candidate = {PROGRAM_COMPONENT: seed.to_yaml()}
    idx = 0
    for name, kind, program in seeds_all:
        tags = [f"zero-{name}", "zero"] + ([f"zero-{kind}"] if chosen_zero[kind] == name else [])
        why = f"{name} seed, zero-shot (as written; zero-shot train {metric} {train_zero[name]:.4f})"
        _write_plain_candidate(cand_dir, idx, program, zero_val[name], val, config, why, {"tags": tags, "train_zero": train_zero[name], "seed": name, "seed_kind": kind})
        idx += 1
    best_idx, best_q = 0, float("-inf")
    for entry in entries.values():
        choice, tags, train_cv = entry["choice"], entry["tags"], entry["train_cv"]
        program = search.combo_program(choice)
        cand = {PROGRAM_COMPONENT: program.to_yaml()}
        parts = "; ".join(f"{s}: " + " + ".join(f"[{v.key} {v.origin}/{v.root}] {v.rationale}" for v in choice[s]) for s in search.slots)
        adapter.rationales[candidate_key(cand)] = f"population {'+'.join(tags)} (train CV {train_cv:.4f}): {parts}"[:700] if train_cv is not None else f"population {'+'.join(tags)}: {parts}"[:700]
        batch = adapter.evaluate(val, cand)
        scores = dict(enumerate(batch.scores))
        objectives = {k: float(np.mean([o[k] for o in batch.objective_scores])) for k in batch.objective_scores[0]} if batch.objective_scores else {}
        meta = {"tags": tags, "train_cv": train_cv, "variants": {s: [{"key": v.key, "origin": v.origin, "root": v.root, "round": v.round} for v in choice[s]] for s in search.slots}}
        _write_candidate(cand_dir, idx, adapter, cand, [None], scores, objectives, len(val), seed_candidate, extra=meta)
        if float(np.mean(batch.scores)) > best_q:
            best_idx, best_q = idx, float(np.mean(batch.scores))
        idx += 1
    persist(rounds_done)
    return {
        "candidates": idx,
        "best_candidate": best_idx,
        "best_val_quality": best_q,
        "seed_val_quality": float(np.mean([metrics.example_quality(metric, r.gold, r.pred, labels) * ex.weight for ex, r in zip(val, zero_val["paper"])])),
        "rounds": rounds_done,
        "variants": {s: len(search.variants[s]) for s in search.slots},
        "seeds": {name: {"kind": kind, "train_zero": train_zero[name], "chosen_for_kind": chosen_zero[kind] == name} for name, kind, _ in seeds_all},
        "train_cv_best": ranked[0][0] if ranked else None,
        "proposer_calls": len(search.proposals),
        "reflection_calls": len(reflection.calls) if reflection is not None else 0,
        "reflection_cost": round(reflection.total_cost, 6) if reflection is not None else 0.0,
        "jev_fresh_cost": round(client.usage.fresh_cost - fresh0, 6),
        "seed_problems": search.seed_problems,
    }


def _write_plain_candidate(cand_dir: Path, idx: int, program: Program, results: list[ExampleResult], val: list[Example], config: TaskConfig, rationale: str, extra: dict[str, Any]) -> None:
    """A candidate evaluated exactly as written (no fitting), e.g. a zero-shot seed (§5.2)."""
    scores = {i: metrics.example_quality(config.primary_metric, ex.label, r.pred, config.labels) * ex.weight for i, (ex, r) in enumerate(zip(val, results))}
    meta = {"index": idx, "key": program.digest(), "parents": [None], "valid": True, "problems": [], "rationale": rationale,
            "val_mean_quality": float(np.mean(list(scores.values()))), "coverage": len(scores), "val_n": len(val),
            "val_scores": {str(k): v for k, v in scores.items()}, "chars": len(program.to_yaml()), **extra}
    (cand_dir / f"{idx}.yaml").write_text(f"# rationale: {rationale}\n" + program.to_yaml())
    _write_json(cand_dir / f"{idx}.json", meta)


# ---- representatives (§10) ----------------------------------------------------

def _candidates(run_dir: Path) -> list[dict[str, Any]]:
    out = []
    for cond in CONDITIONS:
        for path in sorted((run_dir / cond / "candidates").glob("*.json"), key=lambda p: int(p.stem)):
            meta = json.loads(path.read_text())
            meta["name"] = f"{cond}/{meta['index']}"
            meta["condition"] = cond
            out.append(meta)
    return out


def _paired_worse(best: dict[str, float], other: dict[str, float], rng: np.random.Generator) -> bool:
    """True if `other` is significantly worse than `best` (paired bootstrap, 95%)."""
    ids = sorted(best)
    diff = np.array([best[i] - other[i] for i in ids])
    if not diff.any():
        return False
    means = diff[rng.integers(0, len(ids), size=(BOOTSTRAP, len(ids)))].mean(axis=1)
    return float(np.percentile(means, 2.5)) > 0


def _stats(results: list[ExampleResult], config: TaskConfig) -> dict[str, Any]:
    m = metrics.compute([r.gold for r in results], [r.pred for r in results], [r.dist for r in results], config.labels, config.primary_metric, config.negative_label)
    m.update({
        "mean_calls": float(np.mean([r.calls for r in results])),
        "mean_input_tokens": float(np.mean([r.input_tokens for r in results])),
        "mean_cost": float(np.mean([r.cost for r in results])),
    })
    return m


def val_stats(program: Program | Ensemble, val: list[Example], client: JevClient, config: TaskConfig, condition: str, pool: DemoPool | None = None) -> tuple[dict[str, Any], list[ExampleResult]]:
    """Full val metrics for a representative (Jev cache hits: every call was made during search)."""
    results = run_decider(program, val, client, context={"split": "val", "condition": condition}, pool=pool)
    return _stats(results, config), results


def build_ensemble(members: list[Program], config: TaskConfig, train: list[Example], client: JevClient, pool: DemoPool | None) -> tuple[Ensemble, dict[str, Any]]:
    """Ensemble of fitted programs; per-label biases + temperature fit on the TRAIN fit set (§7)."""
    fit_set, _ = train_roles(train, config.optimizer)
    per = [run_examples(m, fit_set, client, context={"split": "train", "condition": "ensemble"}, pool=pool) for m in members]
    rows = [[res[i] for res in per] for i in range(len(fit_set))]
    ok = [(ex, rs) for ex, rs in zip(fit_set, rows) if all(r.error is None for r in rs)]
    gold = [ex.label for ex, _ in ok]
    if config.ordinal:  # band the mean predicted level: keeps the members' fitted Score cutoffs in play
        cutoffs, report = fit_ordinal_cutoffs(config.labels, [[r.pred for r in rs] for _, rs in ok], gold, config.primary_metric)
        mean = [combine(config.labels, [r.dist for r in rs], None, 1.0)[0] for _, rs in ok]
        temperature = fit_temperature(config.labels, mean, gold) if gold else 1.0
        return Ensemble(labels=list(config.labels), members=members, temperature=temperature, cutoffs=cutoffs), report
    bias, temperature, report = fit_label_bias(config.labels, [[r.dist for r in rs] for _, rs in ok], gold, config.primary_metric, config.negative_label)
    return Ensemble(labels=list(config.labels), members=members, label_bias=bias, temperature=temperature), report


def write_representatives(run_dir: Path, config: TaskConfig, val: list[Example], client: JevClient, pool: DemoPool | None = None, train: list[Example] | None = None) -> dict[str, Any]:
    cands = [c for c in _candidates(run_dir) if c.get("valid") and c.get("coverage") == len(val)]
    if not cands:
        return {}
    rng = np.random.default_rng(config.optimizer.seed)
    stats: dict[str, dict[str, Any]] = {}
    preds: dict[str, dict[str, str | None]] = {}
    for c in cands:
        program = Program.from_yaml((run_dir / c["condition"] / "candidates" / f"{c['index']}.yaml").read_text())
        stats[c["name"]], results = val_stats(program, val, client, config, c["condition"], pool)
        preds[c["name"]] = {r.id: r.pred for r in results}
        c["program"] = program
    quality = lambda c: c["val_mean_quality"]  # noqa: E731
    top = max(cands, key=quality)
    tied = [c for c in cands if not _paired_worse(top["val_scores"], c["val_scores"], rng)]
    simplest = min(tied, key=lambda c: (stats[c["name"]]["mean_calls"], c["chars"], -quality(c)))
    calib = max(cands, key=lambda c: (1 - stats[c["name"]]["brier"], quality(c)))
    fewest = min(stats[c["name"]]["mean_calls"] for c in cands)
    cheap = max((c for c in cands if stats[c["name"]]["mean_calls"] <= fewest + 1e-9), key=quality)
    # top-quality: highest val primary metric, no simplicity preference (for pre-registered
    # "how good is the optimizer's best?" test evaluations).
    by_primary = lambda c: (stats[c["name"]]["primary"], quality(c))  # noqa: E731
    top_primary = max(cands, key=by_primary)
    chosen = {"best-quality": simplest, "best-cheap": cheap, "best-calibrated": calib, "top-quality": top_primary}
    conditions = sorted({c["condition"] for c in cands}, key=CONDITIONS.index)
    if len(conditions) > 1:  # per-condition top-quality, so one run can hold several pre-registrable configurations
        chosen.update({f"top-{cond}": max((c for c in cands if c["condition"] == cond), key=by_primary) for cond in conditions})
    for tag in sorted({t for c in cands for t in c.get("tags", [])}):  # population candidates (§5.1-5.2): top-single, top-stacked, top-zero-<kind>, …
        chosen[f"top-{tag}"] = max((c for c in cands if tag in c.get("tags", [])), key=by_primary)
    reps: dict[str, Any] = {}
    rep_preds: dict[str, dict[str, str | None]] = {}
    for name, c in chosen.items():
        (run_dir / f"{name}.yaml").write_text(f"# {name}: {c['name']} — {c['rationale']}\n" + c["program"].to_yaml())
        reps[name] = {"candidate": c["name"], "rationale": c["rationale"], "val": _brief(stats[c["name"]]), "val_mean_quality": quality(c)}
        rep_preds[name] = preds[c["name"]]
    reps["best-quality"]["top_candidate"] = top["name"]
    reps["best-quality"]["indistinguishable"] = [c["name"] for c in tied]

    ranked: list[dict[str, Any]] = []
    for c in sorted(cands, key=by_primary, reverse=True):  # distinct programs, best val primary first
        if all(c["program"].digest() != r["program"].digest() for r in ranked):
            ranked.append(c)
    for k in config.optimizer.ensemble_k:
        if len(ranked) < k or train is None:
            continue
        name = f"ensemble-{k}"
        members = ranked[:k]
        ensemble, report = build_ensemble([m["program"] for m in members], config, train, client, pool)
        ens_stats, results = val_stats(ensemble, val, client, config, name, pool)
        why = f"mean label distribution of the top-{k} val candidates {[m['name'] for m in members]}; label biases fit on train"
        (run_dir / f"{name}.yaml").write_text(f"# {name}: {why}\n" + ensemble.to_yaml())
        reps[name] = {"candidate": name, "members": [m["name"] for m in members], "rationale": why, "val": _brief(ens_stats), "val_mean_quality": None, "fit": report}
        rep_preds[name] = {r.id: r.pred for r in results}

    points = {c["name"]: (quality(c), 1 - stats[c["name"]]["brier"], -stats[c["name"]]["mean_calls"], -stats[c["name"]]["mean_input_tokens"]) for c in cands}
    front = [n for n, p in points.items() if not any(all(q >= x for q, x in zip(o, p)) and o != p for o in points.values())]
    pareto = {
        "objectives": ["quality", "calibration (1-brier)", "-calls", "-input_tokens"],
        "representatives": reps,
        "front": [{"candidate": n, "val": _brief(stats[n])} for n in front],
        "candidates": {n: _brief(stats[n]) for n in stats},
    }
    _write_json(run_dir / "pareto.json", pareto)
    _write_json(run_dir / "val_predictions.json", {"gold": {ex.id: ex.label for ex in val}, "representatives": rep_preds})
    return reps


def rewrite_representatives(run_dir: Path) -> dict[str, Any]:
    """Recompute representatives (incl. ensembles) of a finished, test-unconsumed run from the Jev
    cache only (a cache miss raises: no Jev request is made), e.g. after a decoder fix."""
    rd = Path(run_dir)
    if consumed(rd):
        raise SplitAccessError(f"{rd.name} consumed its test split; its representatives are frozen")
    if _read_json(rd / "run.json", {}).get("state") != "finished":
        raise RuntimeError(f"run {rd.name} is not finished")
    config = _load_run_config(rd)
    train, val = load_split(config, "train"), load_split(config, "val")
    cache_dir = Path(config.jev.cache_dir) if config.jev.cache_dir else rd.parent / ".jev-cache"
    client = JevClient(cache_dir, None, config.jev.model, concurrency=config.jev.concurrency)
    ledger = ResponseLedger(rd)
    client.ledger = ledger
    try:
        reps = write_representatives(rd, config, val, client, make_pool(config, train, rd.parent), train)
    finally:
        client.ledger = None
        client.close()
        _update_run(rd, jev_responses=ledger.finalize())
    write_report(rd)
    return reps


def _brief(m: dict[str, Any]) -> dict[str, Any]:
    keys = ("n", "primary", "accuracy", "macro_f1", "balanced_accuracy", "log_loss", "brier", "ece", "mean_calls", "mean_input_tokens", "mean_cost")
    return {k: round(m[k], 5) if isinstance(m.get(k), float) else m.get(k) for k in keys if k in m}


# ---- report (§11) ----------------------------------------------------------------

def write_report(run_dir: Path) -> str:
    state = _read_json(run_dir / "run.json", {})
    pareto = _read_json(run_dir / "pareto.json", {})
    env = _read_json(run_dir / "environment.json", {})
    test = consumed(run_dir)
    lines = [f"# jev-opt run `{run_dir.name}`", "", f"- task: {state.get('task')}; state: {state.get('state')}; started {state.get('started_at')}, finished {state.get('finished_at')}",
             f"- Jev model: {env.get('jev_model')} (served: {', '.join(env.get('jev_snapshots', [])) or 'n/a'}); optimizer: {env.get('optimizer')}; models seen: {env.get('optimizer_models_seen')}",
             f"- versions: jev-opt {env.get('jev_opt')} ({env.get('jev_opt_git_sha')}), gepa {env.get('gepa')}, python {env.get('python')}, omp {env.get('omp')}", "",
             "## Conditions (validation split)", "", "| condition | status | candidates | seed val quality | best val quality | metric calls | reflection cost |", "|---|---|---|---|---|---|---|"]
    for cond, info in state.get("conditions", {}).items():
        s = info.get("summary", {})
        lines.append(f"| {cond} | {info.get('status')} | {s.get('candidates', '')} | {_fmt(s.get('seed_val_quality'))} | {_fmt(s.get('best_val_quality'))} | {s.get('metric_calls', '')} | {s.get('reflection_cost', '')} |")
    lines += ["", "## Representatives (validation split)", "", "| representative | candidate | primary | accuracy | macro F1 | Brier | ECE | calls/ex | input tok/ex | rationale |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, rep in pareto.get("representatives", {}).items():
        v = rep["val"]
        lines.append(f"| {name} | {rep['candidate']} | {_fmt(v.get('primary'))} | {_fmt(v.get('accuracy'))} | {_fmt(v.get('macro_f1'))} | {_fmt(v.get('brier'))} | {_fmt(v.get('ece'))} | {_fmt(v.get('mean_calls'))} | {_fmt(v.get('mean_input_tokens'), 0)} | {rep['rationale']} |")
    if "best-quality" in pareto.get("representatives", {}):
        bq = pareto["representatives"]["best-quality"]
        lines += ["", f"best-quality prefers the simplest of {len(bq.get('indistinguishable', []))} candidates statistically indistinguishable (paired bootstrap, 95%) from the top candidate {bq.get('top_candidate')}."]
    lines += ["", f"Pareto front (quality, calibration, calls, tokens): {', '.join(p['candidate'] for p in pareto.get('front', []))}", "", "## Test split", ""]
    if test:
        m = test["metrics"]
        lines.append(f"Consumed at {test['consumed_at']} by `{test['representative']}` ({test['candidate']}): primary {_fmt(m.get('primary'))}, accuracy {_fmt(m.get('accuracy'))}, macro F1 {_fmt(m.get('macro_f1'))}, Brier {_fmt(m.get('brier'))}, ECE {_fmt(m.get('ece'))} on n={m.get('n')}. Further optimization of this run is refused.")
    else:
        lines.append("Not consumed. Run `/jev-evaluate-final` once to score the chosen representative on test.")
    text = "\n".join(lines) + "\n"
    (run_dir / "report.md").write_text(text)
    return text


def _fmt(x: Any, digits: int = 3) -> str:
    return f"{x:.{digits}f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else ("" if x is None else str(x))


# ---- status / show / pareto / evaluate_final -----------------------------------------

def _load_run_config(run_dir: Path) -> TaskConfig:
    return task_from_data(yaml.safe_load((run_dir / "config.yaml").read_text())["task"], run_dir)


def status(runs_root: str, run_id: str | None = None) -> dict[str, Any]:
    root = Path(runs_root)
    dirs = [root / run_id] if run_id else sorted((p for p in root.glob("*") if (p / "run.json").exists()), key=lambda p: p.stat().st_mtime, reverse=True)
    runs = []
    for d in dirs:
        state = _read_json(d / "run.json", {})
        if not state:
            continue
        if state.get("state") == "running" and str(d) not in ACTIVE and not _alive(state.get("pid")):
            state["state"] = "interrupted"  # worker died mid-run; resume continues from GEPA's checkpoint
        runs.append({"run_id": d.name, "run_dir": str(d), **{k: state.get(k) for k in ("task", "state", "started_at", "finished_at", "error")},
                     "conditions": {c: {"status": v.get("status"), "best_val_quality": v.get("summary", {}).get("best_val_quality")} for c, v in state.get("conditions", {}).items()},
                     "test_consumed": bool(consumed(d))})
    return {"runs": runs}


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def show(run_dir: str, candidate: str) -> dict[str, Any]:
    rd = Path(run_dir)
    reps = _read_json(rd / "pareto.json", {}).get("representatives", {})
    if candidate in REPRESENTATIVES or candidate in reps:
        rep = reps.get(candidate)
        if rep is None:
            raise FileNotFoundError(f"{candidate} not computed for {rd.name}")
        return {"program_yaml": (rd / f"{candidate}.yaml").read_text(), "scores": rep["val"], "rationale": rep["rationale"], "candidate": rep["candidate"]}
    cond, _, idx = candidate.partition("/")
    if cond not in CONDITIONS or not idx.isdigit():
        raise ValueError(f"candidate must be a representative ({', '.join(reps) or ', '.join(REPRESENTATIVES)}) or <condition>/<index>")
    meta = _read_json(rd / cond / "candidates" / f"{idx}.json")
    if meta is None:
        raise FileNotFoundError(f"no candidate {candidate}")
    scores = {k: meta.get(k) for k in ("val_mean_quality", "coverage", "val_n", "objectives", "valid", "problems")}
    return {"program_yaml": (rd / cond / "candidates" / f"{idx}.yaml").read_text(), "scores": scores, "rationale": meta.get("rationale"), "candidate": candidate}


def pareto(run_dir: str) -> dict[str, Any]:
    data = _read_json(Path(run_dir) / "pareto.json")
    if data is None:
        raise FileNotFoundError("pareto.json not written yet (run not finished)")
    return {"representatives": data["representatives"], "front": data["front"]}


def evaluate_final(run_dir: str, representative: str | None = None, client: JevClient | None = None) -> dict[str, Any]:
    rd = Path(run_dir)
    state = _read_json(rd / "run.json", {})
    if state.get("state") != "finished" or str(rd) in ACTIVE:
        raise RuntimeError(f"run {rd.name} is not finished (state {state.get('state')})")
    record = consumed(rd)
    if record:
        raise SplitAccessError(f"test split already consumed at {record['consumed_at']} by {record['representative']}")
    rep_name = representative or "best-quality"
    reps = _read_json(rd / "pareto.json", {}).get("representatives", {})
    if rep_name not in REPRESENTATIVES and rep_name not in reps:
        raise ValueError(f"representative must be one of {sorted(set(REPRESENTATIVES) | set(reps))}")
    rep = reps.get(rep_name)
    if rep is None:
        raise FileNotFoundError(f"{rep_name} not available")
    config = _load_run_config(rd)
    program = load_decider((rd / f"{rep_name}.yaml").read_text())
    test_path = config.split_path("test")
    stored_fp = yaml.safe_load((rd / "config.yaml").read_text()).get("test_fingerprint", {})
    now_fp = fingerprint(test_path)
    with allow_test_split():
        test = load_split(config, "test")
    train, val = load_split(config, "train"), load_split(config, "val")
    own = client is None
    cache_dir = Path(config.jev.cache_dir) if config.jev.cache_dir else rd.parent / ".jev-cache"
    client = client or make_client(cache_dir, config.jev.model, config.jev.max_rpm, config.jev.concurrency, [config.base_dir, Path.cwd()])
    ledger = ResponseLedger(rd)
    client.ledger = ledger
    try:
        pool = make_pool(config, train, rd.parent)
        results = run_decider(program, test, client, context={"split": "test", "condition": rep["candidate"].split("/")[0]}, pool=pool)
    finally:
        client.ledger = None
        if own:
            client.close()
        _update_run(rd, jev_responses=ledger.finalize())
    m = metrics.compute([r.gold for r in results], [r.pred for r in results], [r.dist for r in results], config.labels, config.primary_metric, config.negative_label)
    m.update({
        "mean_calls": float(np.mean([r.calls for r in results])),
        "mean_input_tokens": float(np.mean([r.input_tokens for r in results])),
        "total_cost": float(sum(r.cost for r in results)),
        "hygiene": {
            "test_ids_in_train_or_val": len({e.id for e in test} & {e.id for e in train + val}),
            "test_file_changed_since_optimize": (stored_fp.get("size"), stored_fp.get("mtime_ns")) != (now_fp["size"], now_fp["mtime_ns"]),
            "demo_ids_outside_train": len({d for r in results for d in r.demo_ids} - {e.id for e in train}),
        },
    })
    saved = record_consumption(rd, {"representative": rep_name, "candidate": rep["candidate"], "program_hash": program.digest(), "test_path": str(test_path), "test_sha256": sha256_file(test_path), "metrics": m})
    write_report(rd)
    return {"metrics": m, "consumed_at": saved["consumed_at"], "representative": rep_name, "candidate": rep["candidate"]}
