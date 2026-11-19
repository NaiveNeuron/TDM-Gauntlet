from __future__ import annotations

import pytest

from jev_opt.ir import Program, ProgramError, apply_slots, text_slots
from jev_opt.task import TaskConfig

TASK = TaskConfig.model_validate({
    "name": "t",
    "labels": ["low", "mid", "high"],
    "ordinal": True,
    "inputs": {"question": "the user question", "answer": "the reply"},
    "data": {"train": "a", "val": "b", "test": "c"},
})

FULL = """
version: 1
labels: [low, mid, high]
inputs: {question: the user question, answer: the reply}
shared_state: true
root: gate
nodes:
  gate:
    kind: stage
    decision: topic
    as: topic
    next: route
  topic:
    kind: choice
    instructions: "What is the topic?\\nPick one."
    state: [question]
    options:
      health: {criterion: medical or wellness, to: {label: health}}
      other: {criterion: anything else, to: {label: other}}
  route:
    kind: branch
    condition: The topic is health and the answer gives risky advice.
    state: [topic, answer]
    if_true: {node: severity}
    if_false: {node: votes}
    threshold: 0.4
  severity:
    kind: score
    instructions: How harmful is the answer?
    state: [question, answer]
    levels: [harmless, somewhat, very]
    outcomes: [{label: low}, {label: mid}, {label: high}]
    cutoffs: [0.5, 1.5]
  votes:
    kind: one_vs_rest
    instructions: Judge the answer.
    state: [question, answer]
    options:
      low: {criterion: The answer is fine.}
      high: {criterion: The answer is dangerous.}
    aggregation: {rule: threshold, thresholds: {high: 0.6}, abstain: mid}
"""


def test_round_trip_yaml_and_json_is_lossless() -> None:
    p = Program.from_yaml(FULL).check(TASK)
    again = Program.from_yaml(p.to_yaml())
    assert again == p
    assert Program.model_validate_json(p.to_json()) == p
    assert "\n" in again.nodes["topic"].instructions  # block scalars survive


def test_decision_subtree_outcomes_are_free_values_but_main_tree_needs_labels() -> None:
    p = Program.from_yaml(FULL).check(TASK)
    assert p.decision_nodes() == {"topic"}
    assert set(p.outcomes_under(p.root)) == {"low", "mid", "high"}


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        (lambda d: d["nodes"]["route"].update(if_false={"node": "gate"}), "cycle"),
        (lambda d: d["nodes"].update(orphan={"kind": "branch", "condition": "x", "state": ["answer"], "if_true": {"label": "low"}, "if_false": {"label": "high"}}), "unreachable"),
        (lambda d: d["nodes"]["votes"]["aggregation"].update(abstain="unknown"), "not task labels"),
        (lambda d: d["nodes"]["severity"].update(state=["question", "nope"]), "neither inputs nor derived"),
        (lambda d: d["nodes"]["topic"].update(state=["topic"]), "neither inputs nor derived"),
        (lambda d: d["nodes"]["severity"].update(levels=["x"], cutoffs=None), "Score needs 2..10"),
        (lambda d: d["nodes"]["severity"].update(cutoffs=[1.5, 0.5]), "ascending"),
        (lambda d: d["nodes"]["votes"].update(options={"low": {"criterion": "fine"}}, aggregation={"rule": "threshold", "abstain": "mid"}) or d["nodes"]["severity"].update(outcomes=[{"label": "low"}, {"label": "mid"}, {"label": "mid"}]), r"labels \['high'\] are unreachable"),
        (lambda d: d["nodes"]["topic"]["options"].update({f"o{i}": {"criterion": "c", "to": {"label": "x"}} for i in range(260)}), "2..255"),
        (lambda d: d.update(labels=["low", "high", "mid"]), "must equal task labels"),
    ],
)
def test_check_rejects_invalid_programs(patch, message: str) -> None:
    import yaml

    data = yaml.safe_load(FULL)
    patch(data)
    with pytest.raises((ProgramError, ValueError), match=message):
        Program.model_validate(data).check(TASK)


def test_text_slots_cover_every_free_text_and_apply_keeps_topology() -> None:
    p = Program.from_yaml(FULL).check(TASK)
    slots = text_slots(p)
    assert {"inputs.answer", "nodes.topic.instructions", "nodes.topic.options.health.criterion", "nodes.route.condition", "nodes.severity.levels.2", "nodes.votes.options.high.criterion"} <= set(slots)
    edited = apply_slots(p, {"nodes.severity.levels.2": "extremely harmful", "nodes.route.condition": "Risky?"})
    assert edited.nodes["severity"].levels[2] == "extremely harmful"
    assert edited.nodes["route"].condition == "Risky?"
    assert p.nodes["route"].condition != "Risky?"  # original untouched
    with pytest.raises(ProgramError):
        apply_slots(p, {"nodes.gate.instructions": "stages have no text"})


def test_calibration_metrics_independent_of_label_order():
    """log loss / Brier must not depend on the task's label order (sklearn sorts labels)."""
    from jev_opt import metrics

    gold = ["b", "a", "b"]
    dists = [{"b": 0.9, "a": 0.1}, {"b": 0.2, "a": 0.8}, {"b": 0.7, "a": 0.3}]
    pred = ["b", "a", "b"]
    m1 = metrics.compute(gold, pred, dists, ["b", "a"], "accuracy")
    m2 = metrics.compute(gold, pred, dists, ["a", "b"], "accuracy")
    assert m1["log_loss"] == m2["log_loss"] and m1["brier"] == m2["brier"]
    per_example = sum(metrics.example_brier(g, d, ["a", "b"]) for g, d in zip(gold, dists)) / 3
    assert abs(m1["brier"] - per_example) < 1e-9
