"""GEPA wiring: optimization improves val, reflection sees train only, progressive policy, resume."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from gepa import EvaluationBatch
from gepa.core.state import GEPAState, ValsetEvaluation
from conftest import read_ledger
from mockhost import InProcessHost

from jev_opt import runs
from jev_opt.adapter import ProgressivePolicy
from jev_opt.reflection import OMPReflectionLM
from jev_opt.task import Example, load_task


def optimize(minimal: Path, tmp_path: Path, client, host: InProcessHost, **params):
    events: list[dict] = []
    params = {"task_path": str(minimal), "runs_root": str(tmp_path / "runs"), "run_id": "r", **params}
    result = runs.optimize(params, lambda m, p: events.append(p), lambda rd: OMPReflectionLM(host, run_dir=rd), client=client)
    return result, events


def test_optimization_improves_val_and_reflection_never_sees_val(minimal: Path, tmp_path: Path, client) -> None:
    host = InProcessHost()
    result, events = optimize(minimal, tmp_path, client, host)
    conds = result["conditions"]
    assert conds["jev-zero"]["best_val_quality"] == pytest.approx(1 / 3)
    for name in ("jev-text-opt", "jev-program-opt"):
        assert conds[name]["seed_val_quality"] == pytest.approx(1 / 3)
        assert conds[name]["best_val_quality"] > 0.9
    assert {r["role"] for r in host.requests} == {"reflection", "structural"}
    assert all(r["run_dir"] == str(tmp_path / "runs" / "r") for r in host.requests)
    config = load_task(minimal)
    val_texts = [json.loads(line)["message"] for line in config.split_path("val").read_text().splitlines()]
    train_texts = [json.loads(line)["message"] for line in config.split_path("train").read_text().splitlines()]
    prompts = "\n".join(str(r["prompt"]) for r in host.requests)
    assert not [t for t in val_texts if t in prompts]  # §8: val never reaches reflection
    assert [t for t in train_texts if t in prompts]  # train failures do
    assert "Program-level summary over 15 train examples" in prompts
    run_dir = tmp_path / "runs" / "r"
    env = json.loads((run_dir / "environment.json").read_text())
    assert env["forbidden_modules_loaded"] == [] and env["optimizer_models_seen"] == ["mock/mock-reflector"]
    assert not [m for m in ("litellm", "openai", "anthropic", "dspy") if m in sys.modules]
    for name in ("best-quality", "best-cheap", "best-calibrated"):
        assert (run_dir / f"{name}.yaml").exists()
    assert (run_dir / "jev-text-opt" / "gepa" / "gepa_state.bin").exists()
    assert "| jev-text-opt | finished |" in (run_dir / "report.md").read_text()
    assert any(e["event"] == "new_best" for e in events) and events[0]["event"] == "run_started"
    rows = [json.loads(line) for line in (run_dir / "usage.jsonl").read_text().splitlines()]
    assert rows and all(r["source"] == "jev" for r in rows)


def _state(n_val: int, seed_scores: list[float]) -> GEPAState:
    evaluation = ValsetEvaluation(
        outputs_by_val_id={i: None for i in range(n_val)},
        scores_by_val_id=dict(enumerate(seed_scores)),
        objective_scores_by_val_id={i: {"quality": s} for i, s in enumerate(seed_scores)},
    )
    return GEPAState({"p": "seed"}, evaluation, frontier_type="hybrid")


def _add(state: GEPAState, policy: ProgressivePolicy, candidate: dict[str, str], score_of) -> int:
    ids = policy.get_eval_batch(None, state)
    evaluation = ValsetEvaluation({i: None for i in ids}, {i: score_of(i) for i in ids}, {i: {"quality": score_of(i)} for i in ids})
    idx = state.update_state_with_new_program([0], candidate, evaluation, None, state.total_num_evals)
    policy.get_valset_score(idx, state)
    return idx


def test_progressive_policy_promotes_only_competitive_candidates() -> None:
    val = [Example(str(i), {"m": ""}, "ab"[i % 2], "val") for i in range(40)]
    seed_score = lambda i: 1.0 if i % 4 else 0.0  # noqa: E731  (0.75 mean)
    calls: list[int] = []

    def evaluate(batch: list[Example], candidate: dict[str, str]) -> EvaluationBatch:
        calls.append(len(batch))
        s = [1.0 if candidate["p"] == "good" else 0.0 for _ in batch]
        return EvaluationBatch(outputs=[None] * len(batch), scores=s, objective_scores=[{"quality": x} for x in s])

    policy = ProgressivePolicy(val, [8, 20], evaluate, seed=0)
    assert policy.stages == [8, 20, 40]
    first = policy.order[:8]
    assert sorted(val[i].label for i in first) == ["a"] * 4 + ["b"] * 4  # stratified stage subset
    state = _state(40, [seed_score(i) for i in range(40)])
    state.total_num_evals = 40

    bad = _add(state, policy, {"p": "bad"}, lambda i: 0.0)
    assert len(state.prog_candidate_val_subscores[bad]) == 8 and calls == []
    good = _add(state, policy, {"p": "good"}, lambda i: 1.0)
    assert len(state.prog_candidate_val_subscores[good]) == 40 and calls == [12, 20]
    assert state.total_num_evals == 40 + 32  # promotions are charged to GEPA's budget
    assert policy.get_best_program(state) == good  # only fully evaluated candidates can be best
    assert state.prog_candidate_objective_scores[good]["quality"] == 1.0
    assert good in state.program_at_pareto_front_objectives["quality"]


def test_cancel_then_resume_continues_the_gepa_checkpoint(minimal: Path, tmp_path: Path, client) -> None:
    host = InProcessHost()
    events: list[dict] = []

    def notify(method: str, params: dict) -> None:
        events.append(params)
        if params["event"] in ("iteration", "new_best") and "candidate 1 " in params["message"]:
            runs.cancel_all()

    params = {"task_path": str(minimal), "runs_root": str(tmp_path / "runs"), "run_id": "r", "conditions": ["jev-text-opt"]}
    first = runs.optimize(params, notify, lambda rd: OMPReflectionLM(host, run_dir=rd), client=client)
    run_dir = tmp_path / "runs" / "r"
    gepa_dir = run_dir / "jev-text-opt" / "gepa"
    assert first["state"] == "cancelled" and (gepa_dir / "gepa.stop").exists()
    saved = GEPAState.load(str(gepa_dir))
    before = (saved.i, saved.total_num_evals, [dict(c) for c in saved.program_candidates])
    assert len(before[2]) == 2 and before[1] < 150
    cancelled_hashes = {r["request_hash"] for r in read_ledger(run_dir)}
    assert cancelled_hashes  # a cancelled run still finalizes its Jev response ledger

    second = runs.optimize({"runs_root": str(tmp_path / "runs"), "run_id": "r", "resume": True}, lambda m, p: None, lambda rd: OMPReflectionLM(host, run_dir=rd), client=client)
    assert second["state"] == "finished"
    resumed = GEPAState.load(str(gepa_dir))
    assert resumed.program_candidates[:2] == before[2]  # checkpointed candidates kept
    assert resumed.i > before[0] and resumed.total_num_evals >= 150
    assert second["conditions"]["jev-text-opt"]["best_val_quality"] > 0.9
    resumed_rows = read_ledger(run_dir)
    assert cancelled_hashes < {r["request_hash"] for r in resumed_rows} and len(resumed_rows) == len({r["request_hash"] for r in resumed_rows})
    with pytest.raises(FileExistsError):
        runs.optimize(params, lambda m, p: None, lambda rd: None, client=client)
