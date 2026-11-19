"""jev-tailored (§5.3): continuous acceptance, confusion targeting, minimal-pair hygiene, end to end."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from mockhost import InProcessHost
from test_levers import WordEmbedder

from jev_opt import runs, seedgen
from jev_opt.demos import DemoPool
from jev_opt.execute import ExampleResult
from jev_opt.ir import ChoiceNode, Option, Program, Target
from jev_opt.reflection import OMPReflectionLM
from jev_opt.tailored import RULEBOOK, PairTarget, acceptance, minimal_pairs, pair_targets
from jev_opt.task import Example, load_task

LABELS = ["a", "b", "c"]


def out(gold: str, pred: str | None, p_gold: float, error: str | None = None) -> dict:
    return {"gold": gold, "pred": pred, "p_gold": p_gold, "error": error}


def test_acceptance_needs_higher_likelihood_and_bounds_the_metric_drop() -> None:
    before = [out("a", "a", 0.6), out("b", "b", 0.6), out("c", "a", 0.2), out("a", "a", 0.6)]
    better = [out("a", "a", 0.9), out("b", "b", 0.9), out("c", "a", 0.4), out("a", "a", 0.9)]
    assert acceptance(before, better, LABELS, "accuracy", None, 0.0)["accept"]
    assert not acceptance(before, before, LABELS, "accuracy", None, 0.0)["accept"]  # equal likelihood: no
    # likelihood up, but one more error: accuracy 0.75 -> 0.5 is within a 0.25 tolerance, not within 0.2
    flipped = [out("a", "a", 0.95), out("b", "c", 0.45), out("c", "a", 0.45), out("a", "a", 0.95)]
    assert acceptance(before, flipped, LABELS, "accuracy", None, 0.25)["accept"]
    assert not acceptance(before, flipped, LABELS, "accuracy", None, 0.2)["accept"]
    # a probability of 0 is floored at 1e-4: one hard miss is a finite penalty the others can outweigh
    hard_miss = [out("a", "a", 0.99), out("b", "b", 0.99), out("c", "a", 0.0), out("a", "a", 0.99)]
    v = acceptance(before, hard_miss, LABELS, "accuracy", None, 0.0)
    assert v["logp_after"] > -2.4 and not v["accept"]
    # an invalid child (uniform output) never passes, however overconfident the parent was
    confident_wrong = [out("a", "b", 0.0), out("b", "a", 0.0), out("c", "a", 0.0), out("a", "b", 0.0)]
    invalid = [out(g, None, 1 / 3, "invalid program: bad") for g in "abca"]
    assert not acceptance(confident_wrong, invalid, LABELS, "accuracy", None, 1.0)["accept"]


def _choice() -> Program:
    node = ChoiceNode(kind="choice", instructions="Which?", state=["text"],
                      options={"a": Option(criterion="A"), "b": Option(criterion="B"), "bc": Option(criterion="B or C", to=Target(node="fine"))})
    fine = ChoiceNode(kind="choice", instructions="Fine?", state=["text"], options={"b2": Option(criterion="b", to=Target(label="b")), "c": Option(criterion="C")})
    return Program(labels=LABELS, inputs={"text": ""}, root="top", nodes={"top": node, "fine": fine})


def res(i: str, gold: str, probs: dict[str, float]) -> ExampleResult:
    choice = max(probs, key=probs.get)
    return ExampleResult(i, gold, None, {}, trace={"top": {"choice": choice, "probabilities": probs}})


def test_confusion_target_prefers_the_pair_jev_cannot_separate() -> None:
    program = _choice()
    rows = []
    # a vs b: 2 errors, but p_a/(p_a+p_b) never separates the classes (AUC 0.5)
    for i in range(6):
        rows.append(res(f"ab{i}", "a" if i % 2 else "b", {"a": 0.4, "b": 0.6, "bc": 0.0} if i < 2 else {"a": 0.5, "b": 0.5, "bc": 0.0}))
    # a vs c (routed through option `bc`): 4 errors, perfectly separable by a bias (AUC 1.0)
    for i in range(8):
        gold = "a" if i < 4 else "c"
        rows.append(res(f"ac{i}", gold, {"a": 0.45, "b": 0.0, "bc": 0.55} if gold == "a" else {"a": 0.3, "b": 0.0, "bc": 0.7}))
    targets = pair_targets(program, rows)
    top = targets[0]
    assert (top.node, top.a, top.b) == ("top", "a", "b") and top.auc == 0.5
    abc = next(t for t in targets if (t.a, t.b) == ("a", "bc"))
    assert abc.auc == 1.0 and abc.mass > top.mass and abc.priority == 0.0  # gold c counts as option `bc` (hierarchy)


def test_minimal_pairs_come_from_train_only_most_similar_first() -> None:
    train = [Example("t1", {"text": "cat purrs softly on the mat"}, "a", "train"), Example("t2", {"text": "cat purrs softly on the rug"}, "b", "train"),
             Example("t3", {"text": "car engine roars"}, "a", "train"), Example("t4", {"text": "car engine starts"}, "b", "train"),
             Example("t5", {"text": "unrelated words here"}, "b", "train")]
    pool = DemoPool(train, WordEmbedder())
    program = Program(labels=["a", "b"], inputs={"text": ""}, root="top",
                      nodes={"top": ChoiceNode(kind="choice", instructions="?", state=["text"], options={"a": Option(criterion="A"), "b": Option(criterion="B")})})
    results = [res(ex.id, ex.label, {"a": 0.5, "b": 0.5}) for ex in train]
    results.append(res("v1", "b", {"a": 0.9, "b": 0.1}))  # a val example (outside the train pool): never paired
    target = PairTarget("top", "a", "b", 0.5, 0.5, 1, 1, (2, 3))
    pairs = minimal_pairs(program, target, results, pool, 4)
    assert [(x.id, y.id) for x, y, _ in pairs] == [("t1", "t2"), ("t3", "t4")]  # disjoint, nearest first, gold a then b
    assert all(r.id != "v1" for x, y, _ in pairs for r in (x, y))


def test_tailored_run_selects_on_val_and_reflection_sees_train_only(minimal: Path, tmp_path: Path, client) -> None:
    host = InProcessHost()
    seedgen.run_generation({"task_path": str(minimal), "out_dir": str(minimal.parent / "seeds_v3"), "n": 2, "kinds": ["jev-native"]}, lambda m, p: None, lambda rd: OMPReflectionLM(host, run_dir=rd))
    task = yaml.safe_load(minimal.read_text())
    task["optimizer"].update(class_bias=True, readout_folds=3, max_reflection_cost=0.01, max_metric_calls=400, progressive=[6],
                             tailored={"starts": ["paper", "jev-native"], "accept_n": 8, "minimal_pairs": 2,
                                       "seeds": [{"kind": "jev-native", "path": f"seeds_v3/jev-native-{i}.yaml"} for i in (1, 2)]})
    path = minimal.parent / "task_tailored.yaml"
    path.write_text(yaml.safe_dump(task))
    host.requests.clear()
    params = {"task_path": str(path), "runs_root": str(tmp_path / "runs"), "run_id": "t", "conditions": ["jev-tailored"]}
    result = runs.optimize(params, lambda m, p: None, lambda rd: OMPReflectionLM(host, run_dir=rd), client=client, embedder=WordEmbedder())
    run_dir = tmp_path / "runs" / "t"
    reps = result["representatives"]
    assert {"top-tailored", "top-tailored-zero", "top-tailored-paper", "top-tailored-zero-paper", "top-tailored-jev-native"} <= set(reps)
    assert reps["top-tailored-paper"]["val"]["primary"] > 0.9  # the rewrite fixed the paper seed (val 0.33 as written)
    zero = Program.from_yaml((run_dir / "top-tailored-zero-paper.yaml").read_text())
    assert zero.temperature == 1.0 and all(getattr(n, "biases", None) is None for n in zero.nodes.values())  # as written
    starts = json.loads((run_dir / "jev-tailored" / "starts.json").read_text())["starts"]
    assert starts["jev-native"]["seed"] in starts["jev-native"]["train_zero"] and all(s["status"] == "finished" for s in starts.values())
    assert sum(s["omp_cost"] for s in starts.values()) <= 0.01 + 0.001  # the run's OMP budget, split over the starts
    rows = [json.loads(line) for line in (run_dir / "evaluations.jsonl").read_text().splitlines()]
    accepts = [r for r in rows if r["stage"] == "accept"]
    assert accepts and all(r["n"] == 8 for r in accepts) and any(r["accept"] for r in accepts)
    config = load_task(path)
    val_texts = [json.loads(line)["message"] for line in config.split_path("val").read_text().splitlines()]
    prompts = "\n".join(str(r["prompt"]) for r in host.requests)
    assert RULEBOOK in prompts and not [t for t in val_texts if t in prompts]
