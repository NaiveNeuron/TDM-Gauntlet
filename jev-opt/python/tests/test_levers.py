"""v2 levers: demonstrations in state, Choice class-bias calibration, length-limited proposals, ensembles."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from mockhost import InProcessHost

from jev_opt import runs
from jev_opt.adapter import JevAdapter, train_roles
from jev_opt.demos import DemoError, DemoPool
from jev_opt.execute import DEMOS_NOTE, redecode, run_ensemble, run_examples
from jev_opt.fit import fit_program
from jev_opt.hygiene import ProgramGuard, SplitAccessError, load_split
from jev_opt.ir import ChoiceNode, Demos, Ensemble, Option, Program, ProgramError, ScoreNode, Target, load_decider
from jev_opt.jev import Decision
from jev_opt.reflection import OMPReflectionLM
from jev_opt.task import Example, OptimizerConfig, load_task


class WordEmbedder:
    """Deterministic bag-of-words hashing embedder (stands in for the sentence-transformers model)."""

    name = "test-words"

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower().split(":", 1)[-1]):
                out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


def ex(i: str, text: str, label: str, split: str = "train") -> Example:
    return Example(i, {"text": text}, label, split)


TRAIN = [
    ex("t-cat1", "cat purrs softly", "pet"),
    ex("t-cat2", "cat sleeps softly", "pet"),
    ex("t-dog1", "dog barks loudly", "pet"),
    ex("t-car1", "car engine roars loudly", "vehicle"),
    ex("t-car2", "car engine starts", "vehicle"),
    ex("t-bus1", "bus engine idles", "vehicle"),
]


def choice_program(**extra: Any) -> Program:
    node = ChoiceNode(kind="choice", instructions="Classify the text.", state=["text"], options={"pet": Option(criterion="an animal"), "vehicle": Option(criterion="a machine")})
    return Program(labels=["pet", "vehicle"], inputs={"text": ""}, root="decide", nodes={"decide": node}, **extra)


class StubClient:
    """JevClient stand-in: a Choice answer from a function of (state, question); records requests."""

    concurrency = 4

    def __init__(self, answer):
        self.answer = answer
        self.requests: list[tuple[dict, dict]] = []

    def decide(self, state: dict, questions: dict, context: dict | None = None) -> Decision:
        self.requests.append((state, questions))
        answers = {qid: {"type": q["type"], "probabilities": self.answer(state, q)} for qid, q in questions.items()}
        return Decision(answers, "stub", 10, 0.0, 0.0, True)


# ---- demonstrations ----------------------------------------------------------------

def test_knn_demos_are_nearest_train_examples_never_self_and_balanced_when_asked() -> None:
    pool = DemoPool(TRAIN, WordEmbedder())
    query = ex("v1", "cat purrs loudly", "pet", "val")
    nearest = pool.select(Demos(mode="knn", k=2), query)
    assert [d.id for d in nearest] == ["t-cat1", "t-cat2"]
    balanced = pool.select(Demos(mode="knn", k=2, balanced=True), query)
    assert [d.label for d in balanced] == ["pet", "vehicle"] and balanced[0].id == "t-cat1"
    # a train example is never its own demonstration (leave-one-out)
    assert "t-cat1" not in [d.id for d in pool.select(Demos(mode="knn", k=5), TRAIN[0])]
    assert "t-cat1" not in [d.id for d in pool.select(Demos(mode="static", ids=["t-cat1", "t-car1"]), TRAIN[0])]


def test_demo_pool_is_train_only_and_static_ids_must_be_train_ids(tmp_path: Path) -> None:
    with pytest.raises(DemoError):
        DemoPool([*TRAIN, ex("v9", "val row", "pet", "val")])
    pool = DemoPool(TRAIN)
    with pytest.raises(DemoError, match="not train example ids"):
        pool.select(Demos(mode="static", ids=["v-123"]), ex("v1", "x", "pet", "val"))
    val = [ex("v-val-1", "a val sentence about trucks", "vehicle", "val")]
    guard = ProgramGuard(TRAIN + val, "", None)
    ok = choice_program(demos=Demos(mode="static", ids=["t-cat1", "t-car1"]))
    assert guard.check_program(ok, pool.ids()) == []  # train ids inside `demos` are not an id leak
    bad = choice_program(demos=Demos(mode="static", ids=["t-cat1", "v-val-1"]))
    assert any("not train example ids" in p for p in guard.check_program(bad, pool.ids()))
    assert any("mentions example id" in p for p in guard.problems(bad.to_yaml()))  # ids anywhere else still are


def test_demos_ir_validation() -> None:
    with pytest.raises(ValueError):
        Demos(mode="static")
    with pytest.raises(ValueError):
        Demos(mode="knn", ids=["t-cat1"])
    task = type("T", (), {"labels": ["pet", "vehicle"], "inputs": {"text": ""}, "ordinal": False})()
    with pytest.raises(ProgramError, match="not question nodes"):
        choice_program(demos=Demos(mode="knn", nodes=["nope"])).check(task)
    with pytest.raises(ProgramError, match="biases must be keyed by option"):
        choice_program().model_copy(update={"nodes": {"decide": ChoiceNode(kind="choice", instructions="x", state=["text"], options={"pet": Option(criterion="a"), "vehicle": Option(criterion="b")}, biases={"cat": 1.0})}}).check(task)
    roundtrip = Program.from_yaml(choice_program(demos=Demos(mode="knn", k=3, balanced=True)).to_yaml())
    assert roundtrip.demos == Demos(mode="knn", k=3, balanced=True)
    assert "biases" not in choice_program().to_yaml() and "demos" not in choice_program().to_yaml()  # v1 programs serialize unchanged


def test_demos_reach_jev_state_with_train_labels_and_redecode_offline() -> None:
    client = StubClient(lambda state, q: {"pet": 0.7, "vehicle": 0.3})
    program = choice_program(demos=Demos(mode="knn", k=2))
    query = ex("v1", "car engine noise", "vehicle", "val")
    [result] = run_examples(program, [query], client, pool=DemoPool(TRAIN, WordEmbedder()))
    state, questions = client.requests[0]
    assert state["text"] == "car engine noise"
    assert state["examples"] == [{"text": "car engine starts", "label": "vehicle"}, {"text": "car engine roars loudly", "label": "vehicle"}]
    assert questions["decide"]["instructions"].startswith(DEMOS_NOTE)
    assert result.demo_ids == ["t-car2", "t-car1"]
    again = redecode(program, query, result.memo, result.demos)
    assert (again.pred, again.dist) == (result.pred, result.dist)
    with pytest.raises(ValueError, match="no demo pool"):
        run_examples(program, [query], client)


# ---- class-bias calibration -------------------------------------------------------

def test_class_bias_fit_fixes_systematic_underprediction() -> None:
    # Jev leans "pet": vehicles get only p=0.4, so every example is decoded as pet.
    client = StubClient(lambda state, q: {"pet": 0.9, "vehicle": 0.1} if "cat" in state["text"] or "dog" in state["text"] else {"pet": 0.6, "vehicle": 0.4})
    program = choice_program()
    raw = run_examples(program, TRAIN, client, exhaustive=True)
    assert [r.pred for r in raw].count("vehicle") == 0
    plain, _ = fit_program(program, list(zip(TRAIN, raw)), "macro_f1", calibrate=False)
    assert plain.nodes["decide"].biases is None  # class_bias off: v1 behaviour
    fitted, report = fit_program(program, list(zip(TRAIN, raw)), "macro_f1", calibrate=True, class_bias=True)
    assert report["before"] < 0.5 and report["after"] == 1.0
    val = [ex("v-car", "car wheel", "vehicle", "val"), ex("v-dog", "dog runs", "pet", "val")]
    assert [r.pred for r in run_examples(fitted, val, client)] == ["vehicle", "pet"]
    assert Program.from_yaml(fitted.to_yaml()).nodes["decide"].biases == fitted.nodes["decide"].biases


def test_train_roles_hold_out_fit_examples_and_cap_the_reflection_pool() -> None:
    train = [ex(f"t{i}", f"row {i}", "ab"[i % 2]) for i in range(40)]
    fit, pool = train_roles(train, OptimizerConfig(fit_examples=10, fit_holdout=True, reflection_pool=12))
    assert len(fit) == 10 and len(pool) == 12 and not {e.id for e in fit} & {e.id for e in pool}
    assert sorted(e.label for e in pool) == ["a"] * 6 + ["b"] * 6  # stratified
    fit2, pool2 = train_roles(train, OptimizerConfig(fit_examples=10, reflection_pool=12))
    assert fit2 == fit and {e.id for e in fit} & {e.id for e in pool2}  # without holdout the sets overlap


# ---- length-limited proposals -------------------------------------------------------

class LongLM:
    """Reflection LM whose first reply is too long; records prompts."""

    def __init__(self, replies: list[str]):
        self.replies = replies
        self.prompts: list[str] = []

    def __call__(self, prompt: Any) -> str:
        self.prompts.append(prompt)
        return self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]


def _adapter(minimal: Path, mode: str, lm: Any, client: Any, **opt: Any) -> JevAdapter:
    task = load_task(minimal)
    for key, value in opt.items():
        setattr(task.optimizer, key, value)
    train = load_split(task, "train")
    from jev_opt.task import seed_program

    seed = seed_program(task).check(task)
    guard = ProgramGuard(train, seed.to_yaml(), None)
    return JevAdapter(task=task, seed=seed, client=client, train=train, guard=guard, mode=mode, condition=f"jev-{mode}-opt", reflection_lm=lm, structural_lm=lm)


def test_text_proposer_states_the_limit_retries_then_trims(minimal: Path, client) -> None:
    long = "Complaint: " + "the customer is unhappy about a broken item. " * 20
    lm = LongLM([f"```\n{long}\n```", f"```\n{long}\n```"])
    adapter = _adapter(minimal, "text", lm, client, max_component_chars=200)
    slot = "nodes.decide.options.complaint.criterion"
    cand = adapter.seed_candidate()
    new = adapter.text_proposer(cand, {slot: [{"Inputs": "x", "Generated Outputs": "y", "Feedback": "z"}]}, [slot])
    assert len(new[slot]) <= 200 and new[slot].endswith(".")
    assert "at most 200 characters" in lm.prompts[0] and "over the 200-character limit" in lm.prompts[1]
    short = LongLM(["```\nComplaint: the customer reports a defect.\n```"])
    adapter2 = _adapter(minimal, "text", short, client, max_component_chars=200)
    assert adapter2.text_proposer(cand, {slot: []}, [slot]) == {slot: "Complaint: the customer reports a defect."} and len(short.prompts) == 1
    assert _adapter(minimal, "text", lm, client).text_proposer is None  # no limit: GEPA's default proposer


def test_program_proposals_over_the_limit_are_rejected_then_trimmed(minimal: Path, client) -> None:
    adapter = _adapter(minimal, "program", None, client, max_component_chars=200)
    parent = adapter.seed
    data = parent.to_data()
    data["nodes"]["decide"]["options"]["praise"]["criterion"] = "Praise: the customer is happy. " * 20
    reply = f"Rationale: longer\n```yaml\n{yaml.safe_dump(data, sort_keys=False)}```"
    lm = LongLM([reply, reply])
    program, _ = adapter._llm_proposal("Improve:\n```\n<curr_param>\n```\n<side_info>", lm, parent, [], keep_structure=True)
    assert "at most 200 characters" in lm.prompts[0] and "has 620 chars (limit 200)" in lm.prompts[1]
    assert len(program.nodes["decide"].options["praise"].criterion) <= 200


# ---- ensembles and end-to-end ---------------------------------------------------------

def test_ensemble_averages_member_distributions_then_applies_label_bias() -> None:
    a, b = choice_program(), choice_program()
    b.nodes["decide"].instructions = "Other wording."
    client = StubClient(lambda state, q: {"pet": 0.8, "vehicle": 0.2} if q["instructions"] == "Classify the text." else {"pet": 0.3, "vehicle": 0.7})
    ens = Ensemble(labels=["pet", "vehicle"], members=[a, b])
    [r] = run_ensemble(ens, [ex("v1", "anything", "pet", "val")], client)
    assert r.pred == "pet" and r.dist["pet"] == pytest.approx(0.55) and r.calls == 2
    biased = ens.model_copy(update={"label_bias": {"vehicle": 0.5}})
    assert run_ensemble(biased, [ex("v1", "anything", "pet", "val")], client)[0].pred == "vehicle"
    assert load_decider(biased.to_yaml()) == biased
    with pytest.raises(ValueError):
        Ensemble(labels=["pet", "vehicle"], members=[a])


def test_optimize_with_demos_calibration_and_ensembles_then_evaluate_final(minimal: Path, tmp_path: Path, client, fake_jev) -> None:
    task = yaml.safe_load(minimal.read_text())
    task["seed_demos"] = {"mode": "knn", "k": 3, "balanced": True}
    task["optimizer"].update(class_bias=True, fit_examples=8, fit_holdout=True, reflection_pool=6, max_component_chars=300, ensemble_k=[2])
    path = minimal.parent / "task_v2.yaml"
    path.write_text(yaml.safe_dump(task))
    host = InProcessHost()
    params = {"task_path": str(path), "runs_root": str(tmp_path / "runs"), "run_id": "v2", "conditions": ["jev-text-opt", "jev-program-opt"], "seed": 3}
    result = runs.optimize(params, lambda m, p: None, lambda rd: OMPReflectionLM(host, run_dir=rd), client=client, embedder=WordEmbedder())
    run_dir = tmp_path / "runs" / "v2"
    assert yaml.safe_load((run_dir / "config.yaml").read_text())["task"]["optimizer"]["seed"] == 3
    reps = result["representatives"]
    assert {"top-jev-text-opt", "top-jev-program-opt", "ensemble-2"} <= set(reps)
    assert len(reps["ensemble-2"]["members"]) == 2 and reps["ensemble-2"]["val"]["n"] == 15
    # every Jev request carried train-only demonstrations; val/test texts never appear as demos
    config = load_task(path)
    train = {json.loads(line)["message"] for line in config.split_path("train").read_text().splitlines()}
    demo_texts = {d["message"] for body in fake_jev.bodies for d in body["state"].get("examples", [])}
    assert demo_texts and demo_texts <= train
    # held-out fit examples and examples outside the reflection pool never reach reflection
    config.optimizer.seed = 3  # the run's --seed override also drives the stratified train roles
    fit, pool = train_roles(load_split(config, "train"), config.optimizer)
    prompts = "\n".join(str(r["prompt"]) for r in host.requests)
    assert not [e for e in fit if e.fields["message"] in prompts]
    assert [e for e in pool if e.fields["message"] in prompts]
    assert all(len(v) <= 300 for p in (run_dir / "jev-text-opt" / "candidates").glob("*.yaml") for v in _texts(Program.from_yaml(p.read_text())))
    preds = json.loads((run_dir / "val_predictions.json").read_text())
    assert set(preds["representatives"]["ensemble-2"]) == set(preds["gold"])
    out = runs.evaluate_final(str(run_dir), "ensemble-2", client=client)
    assert out["metrics"]["n"] == 9 and out["metrics"]["hygiene"]["demo_ids_outside_train"] == 0
    with pytest.raises(SplitAccessError):
        runs.evaluate_final(str(run_dir), "top-quality", client=client)


def _texts(program: Program) -> list[str]:
    from jev_opt.ir import text_slots

    seed_texts = {"Classify the customer message as a complaint, praise or question."}
    return [v for v in text_slots(program).values() if v not in seed_texts]


def test_exhaustive_score_run_records_bands_reachable_only_through_the_expected_level() -> None:
    # Band 1 (-> node `mid`) is reached by no integer level under cutoffs [0.4, 0.6]; refitting the
    # cutoffs to [0.1, 0.6] routes expected level 0.2 there, so re-decoding needs `mid`'s answer.
    score = ScoreNode(kind="score", instructions="Rate.", state=["text"], levels=["low", "mid", "high"], outcomes=[Target(label="pet"), Target(node="mid"), Target(label="vehicle")], cutoffs=[0.4, 0.6])
    mid = ChoiceNode(kind="choice", instructions="Pick.", state=["text"], options={"pet": Option(criterion="a"), "vehicle": Option(criterion="b")})
    program = Program(labels=["pet", "vehicle"], inputs={"text": ""}, root="rate", nodes={"rate": score, "mid": mid})
    client = StubClient(lambda state, q: {"0": 0.8, "1": 0.2, "2": 0.0} if q["type"] == "score" else {"pet": 0.1, "vehicle": 0.9})
    [raw] = run_examples(program, [TRAIN[0]], client, exhaustive=True)
    refit = program.model_copy(deep=True)
    refit.nodes["rate"].cutoffs = [0.1, 0.6]
    assert redecode(refit, TRAIN[0], raw.memo, raw.demos).pred == "vehicle"


def test_ordinal_ensemble_bands_the_mean_member_level_with_fitted_cutoffs() -> None:
    from jev_opt.execute import ordinal_vote
    from jev_opt.fit import fit_ordinal_cutoffs

    labels = ["1", "2", "3", "4"]
    # members agree on an order but systematically under-score by one level (the mean distribution's
    # argmax would inherit that); the fitted cutoffs map mean level k to label k + 1
    member_preds = [["1", "1"], ["1", "2"], ["2", "2"], ["2", "3"], ["3", "3"]]
    gold = ["2", "3", "3", "4", "4"]
    cutoffs, report = fit_ordinal_cutoffs(labels, member_preds, gold, "qwk")
    assert report["after"] > report["before"] and cutoffs == sorted(cutoffs)
    assert [ordinal_vote(labels, p, cutoffs) for p in member_preds] == gold
    with pytest.raises(ValueError, match="cutoffs"):
        Ensemble(labels=["pet", "vehicle"], members=[choice_program(), choice_program()], cutoffs=[0.5, 0.7])
