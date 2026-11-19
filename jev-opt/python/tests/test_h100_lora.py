"""H100 LoRA driver (jev-opt/experiments/h100_lora.py) and analysis, end to end on the CPU: a tiny random
Qwen3_5ForConditionalGeneration (tests/lora_tiny.py, real Qwen3.8 tokenizer) trained by `gauntlet.lora_train` in the
repository's root environment (`.venv`, `lora` extra), on the first examples of real RQ3 splits.

Needs the licensed split data under jev-opt/experiments/<set>/data/, the tokenizer files of the pinned snapshot and
the root environment with PEFT; skipped otherwise. Outputs go to a temporary directory; GPUs are hidden.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "jev-opt" / "experiments"
ROOT_PY = REPO / ".venv" / "bin" / "python"
STEM = "smarter-latent-hate"


def _ready() -> str | None:
    if not (EXP / "smarter_latent_hate" / "data" / "train.jsonl").exists():
        return "licensed split data not present"
    if not (REPO / "data" / "cache" / "huggingface" / "models--Qwen--Qwen3.8-27B" / "snapshots").exists():
        return "Qwen3.8 tokenizer files not present"
    if not ROOT_PY.exists() or subprocess.run([str(ROOT_PY), "-c", "import peft"], capture_output=True).returncode != 0:
        return "root environment without the lora extra"
    return None


pytestmark = pytest.mark.skipif(_ready() is not None, reason=str(_ready()))


@pytest.fixture(scope="module")
def HL(tmp_path_factory):
    """The driver pointed at a temporary output tree, the root environment's trainer and a tiny model; Jev posting
    (disabled by the driver's imports) is restored afterwards."""
    from jev_opt.jev import JevClient

    post = JevClient._post
    saved = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")}
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4")
    sys.path.insert(0, str(EXP))
    import h100_lora

    base = tmp_path_factory.mktemp("lora")
    tiny = base / "tiny"
    subprocess.run([str(ROOT_PY), str(REPO / "tests" / "lora_tiny.py"), str(tiny)], check=True, capture_output=True)
    keep = {k: getattr(h100_lora, k) for k in ("OUT", "SCRATCH", "TRAINER", "MODEL_PATH", "LIMIT", "GRID")}
    h100_lora.OUT, h100_lora.SCRATCH = base / "runs", base / "scratch"
    h100_lora.TRAINER = [str(ROOT_PY), "-m", "gauntlet.lora_train"]
    h100_lora.MODEL_PATH, h100_lora.LIMIT = str(tiny), 24
    h100_lora.GRID = {"lrs": (1e-3, 2e-3), "epochs": 2, "seeds": (0, 1)}
    yield h100_lora
    for k, v in keep.items():
        setattr(h100_lora, k, v)
    JevClient._post = post
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def smoke_fixture(HL, seeds):
    HL.dump_json(HL.smoke_dir(HL.SMOKE_SET) / "smoke.json", {"projection": {"seeds": seeds}})


def test_targets_decode_to_the_gold_label_through_the_seed_program(HL):
    for stem in ("contempo-tbdense", "dress-new-content", "edit-intent", "ezstance-subtaskA"):
        _, cfg, _, seed = HL.source(stem)
        units, excluded, n = HL.train_units(stem)
        assert n == 24 and len(units) + len(excluded) == 24
        train = {e.id: e for e in HL.limited(HL.load_split(cfg, "train"))}
        for u in units:
            ex = train[u["id"]]
            assert u["prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
            assert all(len(c) == 1 or c in ("Yes", "No") for c in u["continuations"])
            assert HL.run_decider(seed, [ex], HL.OneHot(u["target"]))[0].pred == ex.label
            if stem == "dress-new-content":  # the gold level's number
                assert u["continuations"][u["target"]] == str(list(cfg.labels).index(ex.label) + 1)
            if stem == "contempo-tbdense":  # the letter of the gold option
                assert list(seed.nodes[seed.root].options)[u["target"]] == ex.label


def test_an_example_no_option_can_produce_has_no_target(HL):
    _, cfg, _, seed = HL.source("edit-intent")
    add = next(e for e in HL.load_split(cfg, "train") if e.fields.get("edit_action") in ("Add", "Delete"))
    bad = dataclasses.replace(add, label="Grammar")  # the Add/Delete question offers Fact/Evidence, Claim, Other
    state, spec = HL.the_unit(seed, bad)
    assert HL.target_of(seed, bad, len(HL.render(state, spec)["continuations"])) is None


def test_run_requires_the_smoke_report(HL):
    with pytest.raises(SystemExit, match="smoke test first"):
        HL.cmd_run("cpu", STEM)


def test_end_to_end_training_selection_test_once_resume_and_replay(HL, monkeypatch):
    smoke_fixture(HL, [0, 1])
    HL.cmd_run("cpu", STEM)
    sd = HL.set_dir(STEM)
    data = json.loads((sd / "data.json").read_text())
    assert data["train"]["n_units"] == 24 and data["limit"] == 24
    runs = [HL.run_name(lr, s) for lr in HL.GRID["lrs"] for s in (0, 1)]
    for run in runs:
        assert (sd / run / "done.json").exists() and (sd / run / "val-epoch-2.jsonl.gz").exists()
    sel = json.loads((sd / "selection.json").read_text())
    curves = json.loads((sd / "val.json").read_text())["runs"]
    want = HL.choose(curves, (0, 1))
    assert sel["chosen_lr"] == want["chosen_lr"] and [t["epoch"] for t in sel["tested"]] == [t["epoch"] for t in want["tested"]]
    ledger = (HL.OUT / "test_ledger.jsonl").read_text().splitlines()
    assert sorted(json.loads(x)["run_id"] for x in ledger) == [f"lora-{STEM}-s0", f"lora-{STEM}-s1"]
    for t in sel["tested"]:
        td = HL.test_dir(STEM, t["seed"])
        res = json.loads((td / "result.json").read_text())
        rec = json.loads((td / "test_consumed.json").read_text())
        assert len(res["test"]["pred"]) == 24 and res["adapter_sha256"] == rec["adapter_sha256"] == t["adapter_sha256"]
        assert res["scoring"]["reload_max_diff"] <= 0.01 and rec["metrics"]
    kept = sorted(p.relative_to(HL.SCRATCH / STEM / "adapters").as_posix() for p in (HL.SCRATCH / STEM / "adapters").glob("*/epoch-*"))
    assert kept == sorted(f"{t['run']}/epoch-{t['epoch']}" for t in sel["tested"])  # untested adapters deleted

    # resume: a finished set makes no trainer call, re-decodes the recorded responses and records nothing new
    def no_trainer(*a, **k):
        raise AssertionError("trainer called on a finished set")

    monkeypatch.setattr(HL, "trainer", no_trainer)
    HL.cmd_run("cpu", STEM)
    HL.cmd_replay(STEM)
    assert (HL.OUT / "test_ledger.jsonl").read_text().splitlines() == ledger
    monkeypatch.undo()

    # a second test evaluation is refused by the ledger; a changed recorded prediction stops the rerun
    from jev_opt.hygiene import SplitAccessError, record_consumption

    td = HL.test_dir(STEM, 0)
    with pytest.raises(SplitAccessError):
        record_consumption(td, {"representative": "again"})
    original = (td / "result.json").read_text()
    res = json.loads(original)
    res["test"]["pred"][0] = "something else"
    (td / "result.json").write_text(json.dumps(res))
    with pytest.raises(SystemExit, match="differs"):
        HL.test_step(STEM, sel, None)

    # an interrupted training run restarts from its beginning (CPU: the same selection comes back)
    untested = next(r for r in runs if r not in {t["run"] for t in sel["tested"]})
    (sd / untested / "done.json").unlink()
    (sd / untested / "partial.marker").write_text("x")
    (td / "result.json").write_text(original)
    HL.cmd_run("cpu", STEM)
    assert (sd / untested / "done.json").exists() and not (sd / untested / "partial.marker").exists()
    assert json.loads((sd / "selection.json").read_text()) == sel


def test_plan_reports_trainer_environment_and_missing_weights_only(HL):
    plan = HL.cmd_plan(None, None, [STEM])
    assert plan["trainer"]["peft"] == "0.21.2" and plan["smoke"] == [0, 1]
    assert all("pinned base-model files missing" in p for p in plan["problems"])  # this machine has no weights


def test_analysis_pending_without_finished_sets(HL, tmp_path, monkeypatch):
    import analyze_h100_lora as AL

    monkeypatch.setattr(HL, "OUT", tmp_path / "empty")
    for k in ("OUT_JSON", "OUT_MD", "OUT_TEX"):
        monkeypatch.setattr(AL, k, tmp_path / getattr(AL, k).name)
    AL.main()
    rep = json.loads((tmp_path / "H100_LORA.json").read_text())
    assert rep["status"].startswith("pending") and rep["sets"] == []
    assert "\\newcommand{\\loraStatus}{pending}" in (tmp_path / "h100_lora.tex").read_text()
    assert "Status: pending" in (tmp_path / "H100_LORA.md").read_text()


def test_paired_bootstrap_with_resampled_seeds(HL):
    import analyze_h100_lora as AL

    gold = ["a", "b"] * 30
    ids = [f"x{i}" for i in range(60)]
    good, bad = list(gold), ["a"] * 60
    score = lambda g, p: sum(x == y for x, y in zip(g, p)) / len(g)  # noqa: E731
    systems = {k: [good] for k in ("zero", "cal", "opt", "jev", "emb", "enc")}
    systems.update(lora=[good, bad, good], setfit=[bad, bad, bad], ft=[good, good, good])
    point, rs = AL.paired("smarter-hatexplain", ids, gold, score, systems, {"lora": 1, "setfit": 2, "ft": 2, "emb": 2, "enc": 2})
    assert point["lora"] == pytest.approx((1 + 0.5 + 1) / 3)
    c = rs["contrasts"]
    assert c["lora-opt"]["delta"] == pytest.approx(point["lora"] - 1.0)
    assert c["lora-ft"]["ci"][0] < 0 and c["lora-setfit"]["delta"] == pytest.approx(point["lora"] - 0.5)
    assert c["lora-opt"]["ci"][1] <= 0  # every seed resample of LoRA averages at most the perfect score


def test_analysis_of_a_finished_set_on_recorded_systems(HL, tmp_path, monkeypatch):
    """A synthetic finished set (LoRA seed 0 = Exp. 1's optimized predictions, seed 1 = its zero-shot ones) runs the
    full analysis against the recorded Exp. 1, trained-baseline and Jev predictions."""
    import analyze_h100_lora as AL

    monkeypatch.setattr(HL, "OUT", tmp_path / "runs")
    for k in ("OUT_JSON", "OUT_MD", "OUT_TEX"):
        monkeypatch.setattr(AL, k, tmp_path / getattr(AL, k).name)
    cal = json.loads((HL.C.RUNS / f"h2-qwen-base-{STEM}-cal" / "result.json").read_text())
    opt = json.loads((HL.C.RUNS / f"h2-qwen-base-{STEM}-opt" / "result.json").read_text())
    ids, gold = cal["test"]["ids"], cal["test"]["gold"]
    preds = {0: [dict(zip(opt["test"]["ids"], opt["test"]["pred"]))[i] for i in ids], 1: cal["test"]["zero_pred"]}
    sd = HL.set_dir(STEM)
    tested = [{"seed": s, "run": HL.run_name(HL.GRID["lrs"][0], s), "epoch": 1, "val_primary": 0.5, "adapter_sha256": "x"} for s in (0, 1)]
    HL.dump_json(sd / "selection.json", {"seeds": [0, 1], "chosen_lr": HL.GRID["lrs"][0], "mean_best_val": {"0.001": 0.5, "0.002": 0.4},
                                         "tested": tested})
    HL.dump_json(sd / "val.json", {"runs": {}})
    HL.dump_json(sd / "data.json", {"train": {"excluded": []}})
    for lr in HL.GRID["lrs"]:
        for s in (0, 1):
            HL.dump_json(sd / HL.run_name(lr, s) / "train_log.json", {"provenance": {"load_seconds": 60.0},
                         "epoch_summaries": [{"train_seconds": 600.0, "val_scoring": {"wall_seconds": 60.0}, "peak_memory_gib": 61.5}] * 2})
    for s in (0, 1):
        td = HL.test_dir(STEM, s)
        HL.dump_json(td / "result.json", {"test": {"ids": ids, "gold": gold, "pred": preds[s], "primary": 0.0, "frozen_rule_primary": None},
                                          "rejected": []})
        HL.dump_json(td / "test_consumed.json", {"representative": "lora"})
    AL.main()
    rep = json.loads((tmp_path / "H100_LORA.json").read_text())
    (r,) = rep["sets"]
    assert rep["status"].startswith("pending") and r["set"] == STEM and r["optimized"] == pytest.approx(opt["test"]["primary"])
    assert r["lora"] == pytest.approx((opt["test"]["primary"] + cal["test"]["zero_primary"]) / 2)
    assert r["contrasts"]["lora-opt"]["delta"] == pytest.approx(r["lora"] - r["optimized"])
    assert r["contrasts"]["lora-zero"]["delta"] == pytest.approx(r["lora"] - r["zero"])
    assert r["encoder"]["best"] in AL.FAMILIES and r["peak_memory_gib"] == 61.5
    tex = (tmp_path / "h100_lora.tex").read_text()
    assert "\\loraScoreLatent" in tex and "\\loraPoolOpt}" in tex and "\\loraNholmEnc" in tex
    assert "| Latent Hatred |" in (tmp_path / "H100_LORA.md").read_text()
