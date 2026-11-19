"""Execution semantics, structural operators and numerical fitting against the mock Jev."""

from __future__ import annotations

import httpx
import pytest

from jev_opt import metrics
from jev_opt.adapter import JevAdapter
from jev_opt.execute import run_examples
from jev_opt.fit import fit_program
from jev_opt.hygiene import ProgramGuard
from jev_opt.ir import Program, ProgramError
from jev_opt.task import Example, TaskConfig
from jev_opt.transforms import (
    TransformError,
    choice_to_one_vs_rest,
    choice_to_score,
    flatten,
    hierarchize,
    moves,
    one_vs_rest_to_choice,
    score_to_choice,
    set_abstain,
    toggle_shared_state,
)

KEYWORDS = {
    "complaint": "broken damaged cracked refund late terrible awful worst disappointed",
    "praise": "love great excellent amazing perfect happy delighted thanks wonderful fantastic",
    "question": "how when where does which",
}
LABELS = list(KEYWORDS)
TASK = TaskConfig.model_validate({
    "name": "support",
    "labels": LABELS,
    "label_groups": {"feedback": ["complaint", "praise"], "question": ["question"]},
    "inputs": {"message": ""},
    "data": {"train": "a", "val": "b", "test": "c"},
    "primary_metric": "macro_f1",
    "optimizer": {"fit_examples": 30},
})
TEXTS = {
    "complaint": ["It arrived broken.", "Terrible and late.", "Cracked screen, refund please.", "Awful, the worst.", "Damaged box, disappointed."],
    "praise": ["I love it.", "Great and perfect.", "Excellent, thanks!", "Amazing, I am happy.", "Wonderful and fantastic."],
    "question": ["How do I start?", "When will it ship?", "Where is the manual?", "Does it float?", "Which cable?"],
}


def examples(split: str = "train") -> list[Example]:
    return [Example(f"{split}-{lab}-{i}", {"message": t}, lab, split) for lab, texts in TEXTS.items() for i, t in enumerate(texts)]


def keyword_choice() -> Program:
    return Program.model_validate({
        "labels": LABELS,
        "inputs": {"message": ""},
        "root": "decide",
        "nodes": {"decide": {"kind": "choice", "instructions": "Route the message.", "state": ["message"], "options": {lab: {"criterion": KEYWORDS[lab]} for lab in LABELS}}},
    }).check(TASK)


def accuracy(results) -> float:
    return sum(r.correct for r in results) / len(results)


def test_choice_and_one_vs_rest_round_trip_and_agree(client, fake_jev) -> None:
    choice = keyword_choice()
    ovr = choice_to_one_vs_rest(choice, "decide").check(TASK)
    assert one_vs_rest_to_choice(ovr, "decide") == choice
    exs = examples("val")
    assert accuracy(run_examples(choice, exs, client)) == 1.0
    before = len(fake_jev.bodies)
    ovr_results = run_examples(ovr, exs, client)
    assert accuracy(ovr_results) == 1.0
    new = fake_jev.bodies[before:]
    assert len(new) == len(exs)  # the K Nouls of one node share one request
    assert all(sorted(b["questions"]) == [f"decide__{lab}" for lab in sorted(LABELS)] for b in new)
    assert {q["type"] for b in new for q in b["questions"].values()} == {"noul"}


def test_hierarchize_routes_coarse_to_fine_and_flatten_inverts(client) -> None:
    flat = keyword_choice()
    tree = hierarchize(flat, "decide", TASK.label_groups).check(TASK)
    assert set(tree.nodes) == {"decide", "decide_feedback"}
    assert tree.nodes["decide"].options["feedback"].to.node == "decide_feedback"
    assert flatten(tree, "decide").nodes["decide"].options == flat.nodes["decide"].options
    results = {r.gold: r for r in run_examples(tree, examples("val"), client)}
    assert accuracy(results.values()) == 1.0
    assert results["praise"].calls == 2 and set(results["praise"].trace) == {"decide", "decide_feedback"}
    assert results["question"].calls == 1  # singleton group routes straight to the label
    assert abs(sum(results["question"].dist.values()) - 1) < 1e-9


def test_ordinal_score_conversion_requires_ordinal_task() -> None:
    with pytest.raises(TransformError):
        choice_to_score(keyword_choice(), "decide", TASK)
    ordinal = TASK.model_copy(update={"ordinal": True})
    score = choice_to_score(keyword_choice(), "decide", ordinal).check(ordinal)
    assert score.nodes["decide"].levels == [KEYWORDS[lab] for lab in LABELS]
    assert score_to_choice(score, "decide") == keyword_choice()


def test_stage_writes_derived_state_for_downstream_questions(client, fake_jev) -> None:
    program = Program.model_validate({
        "labels": LABELS,
        "inputs": {"message": ""},
        "root": "gate",
        "nodes": {
            "gate": {"kind": "stage", "decision": "asks", "as": "form", "next": "decide"},
            "asks": {"kind": "choice", "instructions": "Form?", "state": ["message"], "options": {
                "interrogative": {"criterion": KEYWORDS["question"], "to": {"label": "interrogative"}},
                "statement": {"criterion": KEYWORDS["complaint"] + " " + KEYWORDS["praise"], "to": {"label": "statement"}}}},
            "decide": {"kind": "choice", "instructions": "Route.", "state": ["message", "form"], "options": {
                "complaint": {"criterion": KEYWORDS["complaint"] + " statement"},
                "praise": {"criterion": KEYWORDS["praise"] + " statement"},
                "question": {"criterion": KEYWORDS["question"] + " interrogative"}}},
        },
    }).check(TASK)
    results = run_examples(program, examples("val"), client)
    assert accuracy(results) == 1.0
    q = next(r for r in results if r.gold == "question")
    assert q.derived == {"form": "interrogative"} and q.calls == 2
    assert {"message": "How do I start?", "form": "interrogative"} in [b["state"] for b in fake_jev.bodies]


def test_shared_state_batches_same_state_questions_into_one_request(client) -> None:
    program = Program.model_validate({
        "labels": LABELS,
        "inputs": {"message": ""},
        "root": "is_question",
        "nodes": {
            "is_question": {"kind": "branch", "condition": KEYWORDS["question"], "state": ["message"], "if_true": {"label": "question"}, "if_false": {"node": "tone"}},
            "tone": {"kind": "choice", "instructions": "Tone?", "state": ["message"], "options": {lab: {"criterion": KEYWORDS[lab]} for lab in ("complaint", "praise")}},
        },
    }).check(TASK)
    lazy = run_examples(program, examples("val"), client)
    shared = run_examples(toggle_shared_state(program), examples("val"), client)
    assert [r.pred for r in lazy] == [r.pred for r in shared]
    assert accuracy(shared) == 1.0
    assert sum(r.calls for r in lazy) == 10 * 2 + 5 and sum(r.calls for r in shared) == 15


def test_jev_failure_is_a_per_example_low_score(tmp_path, fake_jev, client) -> None:
    fake_jev.failures = [httpx.Response(400, json={"error": {"message": "bad question"}})]
    [result] = run_examples(keyword_choice(), examples("val")[:1], client)
    assert result.pred is None and "HTTP 400" in result.error
    assert result.dist == {lab: pytest.approx(1 / 3) for lab in LABELS}


def test_set_abstain_falls_back_when_no_noul_fires(client) -> None:
    ovr = choice_to_one_vs_rest(keyword_choice(), "decide")
    abstaining = set_abstain(ovr, "decide", "question").check(TASK)
    assert set(abstaining.nodes["decide"].options) == {"complaint", "praise"}
    results = {r.gold: r for r in run_examples(abstaining, examples("val"), client)}
    assert results["question"].pred == "question" and results["praise"].pred == "praise"


def test_every_deterministic_move_is_valid_and_distinct() -> None:
    found = moves(keyword_choice(), TASK)
    names = [m.name for m in found]
    assert len(names) == len(set(names))
    assert {"choice_to_one_vs_rest:decide", "hierarchize:decide"} <= set(names)
    assert all(m.program != keyword_choice() and m.rationale for m in found)


def biased_task_and_program() -> tuple[TaskConfig, Program, list[Example], list[Example]]:
    """Every message shares `order`, so the unfitted 0.5 Noul threshold flags everything."""
    task = TaskConfig.model_validate({"name": "b", "labels": ["urgent", "routine"], "inputs": {"message": ""}, "data": {"train": "a", "val": "b", "test": "c"}, "primary_metric": "macro_f1", "optimizer": {"fit_examples": 20}})
    program = Program.model_validate({
        "labels": ["urgent", "routine"], "inputs": {"message": ""}, "root": "u",
        "nodes": {"u": {"kind": "one_vs_rest", "instructions": "Is it urgent?", "state": ["message"],
                        "options": {"urgent": {"criterion": "order immediately"}}, "aggregation": {"rule": "threshold", "abstain": "routine"}}},
    }).check(task)

    def make(split: str) -> list[Example]:
        return [Example(f"{split}{i}", {"message": f"order {i} immediately"}, "urgent", split) for i in range(5)] + [
            Example(f"{split}r{i}", {"message": f"order number {i}"}, "routine", split) for i in range(5)]

    return task, program, make("train"), make("val")


def test_threshold_fitting_on_train_improves_val_without_new_calls(client, fake_jev) -> None:
    task, program, train, val = biased_task_and_program()
    before = metrics.primary("macro_f1", [e.label for e in val], [r.pred for r in run_examples(program, val, client)], task.labels)
    raw = run_examples(program, train, client, exhaustive=True)
    calls = len(fake_jev.bodies)
    fitted, report = fit_program(program, list(zip(train, raw)), "macro_f1", calibrate=True)
    assert len(fake_jev.bodies) == calls  # fitting re-decodes recorded answers only
    assert 0.62 < fitted.nodes["u"].aggregation.thresholds["urgent"] <= 0.88
    after = metrics.primary("macro_f1", [e.label for e in val], [r.pred for r in run_examples(fitted, val, client)], task.labels)
    assert before < 0.5 and after == 1.0 and report["after"] == 1.0
    assert fitted.temperature < 1.0  # correct but under-confident on train -> sharpen


def test_adapter_scores_candidates_with_fitted_parameters(client) -> None:
    task, program, train, val = biased_task_and_program()
    adapter = JevAdapter(task=task, seed=program, client=client, train=train, guard=ProgramGuard(train + val, program.to_yaml(), None), mode="program", condition="t")
    batch = adapter.evaluate(val, adapter.seed_candidate())
    assert sum(batch.scores) == pytest.approx(len(val))  # weights are class-balanced (mean 1)
    assert set(batch.objective_scores[0]) == {"quality", "calibration", "cheap", "tokens"}


def test_micro_f1_negative_label_is_never_a_positive() -> None:
    # TB-Dense scoring: VAGUE correct predictions earn nothing, VAGUE gold is never recalled.
    gold = ["A", "B", "V", "V"]
    pred = ["A", "V", "V", "B"]
    # positives: tp=1 (A); predicted positives=2 (A, B); gold positives=2 (A, B) -> P=R=F1=0.5
    assert metrics.primary("micro_f1", gold, pred, ["A", "B", "V"], negative="V") == pytest.approx(0.5)
    # Correct VAGUE agreement does not count: all-label micro-F1 would be 0.75 here.
    gold, pred = ["A", "V", "V", "V"], ["B", "V", "V", "V"]
    assert metrics.primary("micro_f1", gold, pred, ["A", "B", "V"]) == pytest.approx(0.75)
    assert metrics.primary("micro_f1", gold, pred, ["A", "B", "V"], negative="V") == 0.0
    base = {"name": "t", "labels": ["A", "B", "V"], "inputs": {"x": ""}, "data": {"train": "a", "val": "b", "test": "c"}}
    with pytest.raises(ValueError):
        TaskConfig.model_validate({**base, "primary_metric": "macro_f1", "negative_label": "V"})
    with pytest.raises(ValueError):
        TaskConfig.model_validate({**base, "primary_metric": "micro_f1", "negative_label": "Z"})


def test_switch_routes_on_field_value_without_a_jev_call(client, fake_jev) -> None:
    task = TASK.model_copy(update={"inputs": {"message": "", "channel": ""}})
    program = Program.model_validate({
        "labels": LABELS,
        "inputs": {"message": "", "channel": ""},
        "root": "by_channel",
        "nodes": {
            "by_channel": {"kind": "switch", "field": "channel", "cases": {"chat": {"node": "decide"}, "faq": {"label": "question"}}},
            "decide": {"kind": "choice", "instructions": "Route.", "state": ["message"], "options": {lab: {"criterion": KEYWORDS[lab]} for lab in LABELS}},
        },
    }).check(task)
    chat = [Example(e.id, {**e.fields, "channel": "chat"}, e.label, e.split) for e in examples("val")]
    faq = Example("faq-0", {"message": "I love it.", "channel": "faq"}, "question", "val")
    unknown = Example("x-0", {"message": "I love it.", "channel": "mail"}, "praise", "val")
    before = len(fake_jev.bodies)
    results = run_examples(program, [*chat, faq, unknown], client)
    assert accuracy(results[:-2]) == 1.0 and all(r.calls == 1 for r in results[:-2])
    assert results[-2].pred == "question" and results[-2].calls == 0
    assert results[-1].pred is None and "no case" in results[-1].error
    assert len(fake_jev.bodies) - before == len(chat)
    assert all(b["state"] == {"message": b["state"]["message"]} for b in fake_jev.bodies[before:])  # routing field not sent
    with pytest.raises(ProgramError):  # a switch on a field that is not in the state
        Program.model_validate({**program.to_data(), "nodes": {**program.to_data()["nodes"], "by_channel": {"kind": "switch", "field": "nope", "cases": {"x": {"label": "praise"}}}}}).check(task)
