"""v3 levers: readout heads (§4.2), per-(state, question) Jev cache (§9), jev-population-opt (§5.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from mockhost import InProcessHost
from test_levers import StubClient, WordEmbedder, choice_program, ex

from jev_opt import runs, seedgen
from jev_opt.execute import run_examples
from jev_opt.fit import CVSpec, fit_program, with_readouts
from jev_opt.hygiene import SplitAccessError
from jev_opt.ir import BranchNode, Program, ProgramError, Readout, ScoreNode, Target
from jev_opt.jev import JevClient
from jev_opt.reflection import OMPReflectionLM
from jev_opt.task import load_task

# ---- readout heads -------------------------------------------------------------------


def _pets_and_cars(n: int = 20) -> list:
    return [ex(f"p{i}", f"cat {i}", "pet") for i in range(n)] + [ex(f"v{i}", f"car {i}", "vehicle") for i in range(n)]


def test_lr_readout_learns_the_decision_argmax_misses_and_round_trips() -> None:
    # Jev leans "pet" on every input; only the size of p(vehicle) separates the classes.
    client = StubClient(lambda state, q: {"pet": 0.9, "vehicle": 0.1} if "cat" in state["text"] else {"pet": 0.6, "vehicle": 0.4})
    program = with_readouts(choice_program(), "macro_f1")
    assert program.nodes["decide"].readout == Readout(kind="lr", views=["decide"], classes=["pet", "vehicle"])
    train = _pets_and_cars()
    raw = run_examples(program, train, client, exhaustive=True)
    fitted, report = fit_program(program, list(zip(train, raw)), "macro_f1", calibrate=True, cv=CVSpec(folds=5))
    assert report["readouts"]["decide"]["cv"] == 1.0 and report["after"] == 1.0
    val = [ex("vc", "car x", "vehicle", "val"), ex("vd", "cat y", "pet", "val")]
    assert [r.pred for r in run_examples(fitted, val, client)] == ["vehicle", "pet"]
    assert Program.from_yaml(fitted.to_yaml()) == fitted


def test_grouped_cv_keeps_groups_out_of_their_own_folds() -> None:
    examples = [ex(f"train:doc{i % 6}:e{i}", "t", "ab"[i % 2]) for i in range(60)]
    folds = CVSpec(folds=5, group=r"^train:([^:]+):").splits([e.label for e in examples], examples)
    for tr, te in folds:
        assert not {examples[i].id.split(":")[1] for i in tr} & {examples[i].id.split(":")[1] for i in te}


def test_ridge_readout_decompresses_an_ordinal_score() -> None:
    # Jev's expected level barely moves (levels 3-4 of 0-8) while gold spans the whole scale.
    labels = [str(i) for i in range(5)]
    level = {"0": 3, "1": 3, "2": 4, "3": 4, "4": 5}
    node = ScoreNode(kind="score", instructions="Rate the essay.", state=["text"], levels=[f"level {i}" for i in range(9)], outcomes=[Target(label=lab) for lab in labels])
    program = Program(labels=labels, inputs={"text": ""}, root="rate", nodes={"rate": node})
    train = [ex(f"t{g}-{i}", f"grade {g} {i}", g) for g in labels for i in range(12)]

    def answer(state: dict, q: dict) -> dict:
        g = state["text"].split()[1]
        lev, spread = level[g], 0.6 if g in "13" else 0.9
        return {str(i): (spread if i == lev else (1 - spread) if i == lev + 1 else 0.0) for i in range(9)}

    client = StubClient(answer)
    plain = run_examples(program, train, client)
    fitted, report = fit_program(with_readouts(program, "qwk"), list(zip(train, run_examples(with_readouts(program, "qwk"), train, client, exhaustive=True))), "qwk", calibrate=False, cv=CVSpec())
    assert fitted.nodes["rate"].readout.kind == "ridge" and len(fitted.nodes["rate"].readout.cutoffs) == 4
    assert sum(r.pred == r.gold for r in plain) < 20
    assert all(r.pred == r.gold for r in run_examples(fitted, train, client))


def test_readout_views_are_feature_questions_never_routed() -> None:
    program = with_readouts(choice_program(), "macro_f1")
    program.nodes["cue"] = BranchNode(kind="branch", condition="It purrs.", state=["text"], if_true=Target(label="yes"), if_false=Target(label="no"))
    program.nodes["decide"].readout.views.append("cue")
    task = load_task(Path(__file__).resolve().parents[2] / "examples" / "minimal" / "task.yaml").model_copy(update={"labels": ["pet", "vehicle"], "inputs": {"text": ""}})
    program.check(task)  # free-form yes/no outcomes are fine on a view
    client = StubClient(lambda state, q: {"pet": 0.5, "vehicle": 0.5})
    client.decide = _with_noul(client.decide)
    run_examples(program, [ex("a", "cat", "pet")], client)
    assert len(client.requests) == 1 and set(client.requests[0][1]) == {"decide", "cue"}  # one request for the family
    bad = program.model_copy(deep=True)
    bad.nodes["decide"].readout.views = ["cue", "decide"]
    with pytest.raises(ProgramError, match="start with the node itself"):
        bad.check(task)


def _with_noul(decide):
    def wrapped(state, questions, context=None):
        d = decide(state, questions, context)
        for qid, q in questions.items():
            if q["type"] == "noul":
                d.answers[qid] = {"type": "noul", "noul": 0.7}
        return d

    return wrapped


# ---- per-(state, question) cache ---------------------------------------------------------

def _q(text: str) -> dict:
    return {"type": "noul", "instructions": text}


def test_repacking_never_rebuys_an_answer_and_usage_is_estimated(tmp_path: Path, fake_jev, client: JevClient) -> None:
    state = {"message": "refund please"}
    first = client.decide(state, {"a": _q("A refund is requested."), "b": _q("The item is broken.")})
    assert not first.estimated and len(fake_jev.bodies) == 1
    second = client.decide(state, {"b2": _q("The item is broken."), "c": _q("The customer is polite.")})
    assert [set(b["questions"]) for b in fake_jev.bodies] == [{"a", "b"}, {"c"}]  # only the unanswered question was sent
    assert second.answers["b2"] == first.answers["b"] and second.estimated and not second.cached
    third = client.decide(state, {"x": _q("A refund is requested.")})
    assert len(fake_jev.bodies) == 2 and third.cached and third.answers["x"] == first.answers["a"] and third.input_tokens > 0
    assert client.usage.fresh_calls == 2
    client.decide({"message": "other"}, {"a": _q("A refund is requested.")})  # the state is part of the key
    assert len(fake_jev.bodies) == 3


def test_large_batches_are_sent_in_chunks(client: JevClient, fake_jev) -> None:
    client.max_questions = 3
    decision = client.decide({"m": "x"}, {f"q{i}": _q(f"statement {i}") for i in range(7)})
    assert [len(b["questions"]) for b in fake_jev.bodies] == [3, 3, 1] and len(decision.answers) == 7


# ---- jev-population-opt ---------------------------------------------------------------------

def test_seed_generation_writes_validated_programs_from_definitions_only(minimal: Path, tmp_path: Path) -> None:
    host = InProcessHost()
    out = tmp_path / "seeds"
    report = seedgen.run_generation({"task_path": str(minimal), "out_dir": str(out), "n": 2}, lambda m, p: None, lambda rd: OMPReflectionLM(host, run_dir=rd))
    assert [f["file"] for f in report["kinds"]["jev-native"]["files"]] == ["jev-native-1.yaml", "jev-native-2.yaml"]
    decomposed = report["kinds"]["decomposed"]
    assert [f["file"] for f in decomposed["files"]] == ["decomposed-1.yaml", "decomposed-2.yaml"]
    assert any("unreachable" in r for r in decomposed["rejected"])  # the program that drops a label
    task = load_task(minimal)
    hierarchy = Program.from_yaml((out / "decomposed-2.yaml").read_text()).check(task)
    assert set(hierarchy.nodes) == {"decide", "decide_tone"} and hierarchy.shared_state
    train_texts = [json.loads(line)["message"] for line in task.split_path("train").read_text().splitlines()]
    prompts = "\n".join(str(r["prompt"]) for r in host.requests)
    assert not [t for t in train_texts if t in prompts]  # the writer never sees data
    assert all(r["role"] == "structural" and r["run_dir"] == str(out) for r in host.requests)


def test_population_opt_packs_variants_scores_by_train_cv_and_selects_on_val(minimal: Path, tmp_path: Path, client: JevClient, fake_jev) -> None:
    host = InProcessHost()
    seedgen.run_generation({"task_path": str(minimal), "out_dir": str(minimal.parent / "seeds_v3"), "n": 2}, lambda m, p: None, lambda rd: OMPReflectionLM(host, run_dir=rd))
    task = yaml.safe_load(minimal.read_text())
    seeds = [{"kind": kind, "path": f"seeds_v3/{kind}-{i}.yaml"} for kind in ("jev-native", "decomposed") for i in (1, 2)]
    task["optimizer"].update(fit_examples=15, readout=True, readout_folds=3, max_component_chars=300, max_program_chars=20000,
                             population={"size": 5, "rounds": 2, "val_candidates": 3, "stack_max": 3, "demos": {"mode": "knn", "k": 3, "balanced": True}, "seeds": seeds})
    path = minimal.parent / "task_v3.yaml"
    path.write_text(yaml.safe_dump(task))
    host.requests.clear()
    params = {"task_path": str(path), "runs_root": str(tmp_path / "runs"), "run_id": "p", "conditions": ["jev-population-opt"]}
    result = runs.optimize(params, lambda m, p: None, lambda rd: OMPReflectionLM(host, run_dir=rd), client=client, embedder=WordEmbedder())
    run_dir = tmp_path / "runs" / "p"
    reps = result["representatives"]
    assert {"top-single", "top-stacked", "top-quality", "top-zero-paper", "top-zero-jev-native", "top-zero-decomposed", "top-head-decomposed", "top-search-paper"} <= set(reps)
    summary = result["conditions"]["jev-population-opt"]
    assert summary["seeds"]["paper"]["chosen_for_kind"] and sum(s["chosen_for_kind"] for s in summary["seeds"].values()) == 3
    zero = Program.from_yaml((run_dir / "top-zero-decomposed.yaml").read_text())
    assert all(getattr(n, "readout", None) is None for n in zero.nodes.values()) and zero.temperature == 1.0  # zero-shot: as written
    state = json.loads((run_dir / "jev-population-opt" / "population.json").read_text())
    roots = {v["origin"]: v["root"] for v in state["variants"]}
    assert roots["decomposed-2"] == "decomposed" and len(next(v for v in state["variants"] if v["origin"] == "decomposed-2")["nodes"]) == 2  # hierarchy -> one bag
    assert state["rounds_done"] == 2 and len(host.requests) >= 2
    assert any("main options must be exactly the labels" in r for p in state["proposals"] for r in p["rejected"])
    # every (state, question) pair was bought once, although variants were re-packed many times
    pairs = [(json.dumps(b["state"], sort_keys=True), json.dumps(q, sort_keys=True)) for b in fake_jev.bodies for q in b["questions"].values()]
    assert len(pairs) == len(set(pairs))
    assert max(len(b["questions"]) for b in fake_jev.bodies) >= 4  # a round's variants share one request
    # the proposer saw train examples only
    config = load_task(path)
    val_texts = [json.loads(line)["message"] for line in config.split_path("val").read_text().splitlines()]
    prompts = "\n".join(str(r["prompt"]) for r in host.requests)
    assert not [t for t in val_texts if t in prompts]
    stacked = Program.from_yaml((run_dir / "top-stacked.yaml").read_text())
    root = stacked.nodes[stacked.root]
    assert len(root.readout.views) >= 2 and root.readout.fitted
    out = runs.evaluate_final(str(run_dir), "top-stacked", client=client)
    assert out["metrics"]["n"] == 9 and out["metrics"]["hygiene"]["demo_ids_outside_train"] == 0
    with pytest.raises(SplitAccessError):
        runs.evaluate_final(str(run_dir), "top-single", client=client)
