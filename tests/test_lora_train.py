"""gauntlet.lora_train on a tiny random Qwen3_5ForConditionalGeneration on the CPU (real Qwen3.8 tokenizer, no weights).

Run with the GPUs hidden (`CUDA_VISIBLE_DEVICES=`): with a visible GPU the linear-attention kernels of
flash-linear-attention are chosen and refuse CPU tensors (as for tests/test_logprob_serve.py)."""

import gzip
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
torch.set_num_threads(4)  # tiny matrices: more threads only add contention
import lora_tiny  # noqa: E402

from gauntlet import lora_train as L  # noqa: E402
from gauntlet.logprob_serve import PROMPT_END, ContinuationError  # noqa: E402

pytestmark = pytest.mark.skipif(not lora_tiny.available(), reason="Qwen3.8 tokenizer files not in data/cache/huggingface")


def prompt(state: str, n: int) -> str:
    lines = "".join(f"{'ABCDEFGH'[i]}. opt{i}: criterion {i}\n" for i in range(n))
    return (f"<|im_start|>system\nRead the state and answer the question about it.<|im_end|>\n<|im_start|>user\nSTATE:\n{state}"
            f"\n\nQUESTION:\nWhich?\n\nALLOWED ANSWERS:\n{lines}Reply with the letter of one allowed answer.<|im_end|>\n" + PROMPT_END)


def units(n: int, k: int = 3, offset: int = 0) -> list[dict]:
    return [{"key": f"u{offset + i}", "prompt": prompt(f"item {offset + i} " + "word " * (i % 5), k),
             "continuations": list("ABC"[:k]), "target": (offset + i) % k} for i in range(n)]


def write_units(path, rows):
    L.write_rows(path, rows)
    return str(path)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return str(lora_tiny.build(tmp_path_factory.mktemp("tiny") / "model"))


def test_targets_are_single_tokens_and_out_of_range_targets_stop(tiny):
    lm, tok, _ = L.load_base("cpu", tiny)
    sc = L.scorer_for(L.text_runtime(lm, tok), "cpu")
    enc = L.encode_targets(sc, units(4))
    for (ids, target), u in zip(enc, units(4)):
        assert ids == tok(u["prompt"], add_special_tokens=False).input_ids
        assert target == tok(u["continuations"][u["target"]], add_special_tokens=False).input_ids[0]
    with pytest.raises(ValueError, match="outside"):
        L.encode_targets(sc, [{**units(1)[0], "target": 3}])
    with pytest.raises(ContinuationError):
        L.encode_targets(sc, [{**units(1)[0], "continuations": ["A", "AB C"]}])


def test_untrained_adapter_scores_exactly_like_the_base_model(tiny):
    lm, tok, _ = L.load_base("cpu", tiny)
    rt = L.text_runtime(lm, tok)
    base_rows, _ = L.score_units(lm, L.scorer_for(rt, "cpu"), units(6))
    model = L.attach_lora(lm, 0)
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    lora_rows, use = L.score_units(model, L.scorer_for(L.text_runtime(lm, tok), "cpu"), units(6))
    assert use["units"] == 6 and use["rejected"] == 0
    assert L.max_prob_diff(base_rows, lora_rows) == 0.0  # LoRA B starts at zero


def test_two_training_steps_follow_the_schedule_and_change_the_answers(tiny):
    lm, tok, _ = L.load_base("cpu", tiny)
    model = L.attach_lora(lm, 0)
    rt = L.text_runtime(lm, tok)
    sc = L.scorer_for(rt, "cpu")
    before, _ = L.score_units(model, sc, units(4))
    log = L.train(model, rt, sc, units(32), lr=1e-2, seed=0, epochs=1, device="cpu")
    assert log["total_steps"] == 2 and log["warmup_steps"] == 1 and len(log["steps"]) == 2
    assert [s["lr"] for s in log["steps"]] == [0.0, 1e-2]  # warm-up from 0, then linear decay
    assert [s["units"] for s in log["steps"]] == [16, 16]
    assert all(s["loss"] > 0 for s in log["steps"])
    assert log["trainable_parameters"] == sum(p.numel() for p in model.parameters() if p.requires_grad)
    after, _ = L.score_units(model, sc, units(4))
    assert L.max_prob_diff(before, after) > 0


def test_same_seed_same_initialisation_and_order(tiny):
    outs = []
    for _ in range(2):
        lm, tok, _ = L.load_base("cpu", tiny)
        model = L.attach_lora(lm, 1)
        rt = L.text_runtime(lm, tok)
        log = L.train(model, rt, L.scorer_for(rt, "cpu"), units(20), lr=1e-3, seed=1, epochs=1, device="cpu")
        outs.append([s["loss"] for s in log["steps"]])
    assert outs[0] == pytest.approx(outs[1], rel=1e-5)


def test_cli_train_resume_refusal_score_once_and_reload_check(tiny, tmp_path):
    tr = write_units(tmp_path / "train.jsonl.gz", units(20))
    va = write_units(tmp_path / "val.jsonl.gz", [{k: v for k, v in u.items() if k != "target"} for u in units(10, offset=100)])
    te = write_units(tmp_path / "test.jsonl.gz", [{k: v for k, v in u.items() if k != "target"} for u in units(7, offset=200)])
    out, ad = tmp_path / "run", tmp_path / "adapters"
    args = ["train", "--units", tr, "--val-units", va, "--lr", "1e-3", "--seed", "0", "--epochs", "2", "--out", str(out),
            "--adapters", str(ad), "--device", "cpu", "--model-path", tiny]
    L.main(args)
    done = json.loads((out / "done.json").read_text())
    log = json.loads((out / "train_log.json").read_text())
    assert set(done["adapter_sha256"]) == {"1", "2"} and log["total_steps"] == 4
    assert log["provenance"]["model"] == f"local:{tiny}" and log["configuration"]["libraries"]["peft"]
    for e in (1, 2):
        rows = [json.loads(x) for x in gzip.open(out / f"val-epoch-{e}.jsonl.gz", "rt")]
        assert [r["key"] for r in rows] == [f"u{100 + i}" for i in range(10)] and all("response" in r for r in rows)
        assert (ad / f"epoch-{e}" / "adapter_model.safetensors").exists()
    with pytest.raises(SystemExit, match="finished already"):
        L.main(args)

    job = {"units": te, "jobs": [{"name": "s0", "adapter": str(ad / "epoch-2"), "out": str(tmp_path / "test-s0.jsonl.gz"),
                                  "check": {"units": va, "responses": str(out / "val-epoch-2.jsonl.gz"), "n": 4}},
                                 {"name": "s0e1", "adapter": str(ad / "epoch-1"), "out": str(tmp_path / "test-s0e1.jsonl.gz"),
                                  "check": {"units": va, "responses": str(out / "val-epoch-1.jsonl.gz"), "n": 4}}]}
    (tmp_path / "job.json").write_text(json.dumps(job))
    score = ["score", "--job", str(tmp_path / "job.json"), "--report", str(tmp_path / "report.json"), "--device", "cpu", "--model-path", tiny]
    L.main(score)
    rep = json.loads((tmp_path / "report.json").read_text())
    assert all(j["reload_max_diff"] < 1e-3 for j in rep["jobs"])  # the check's batches differ from the val scoring's (BF16)
    assert rep["jobs"][0]["adapter_sha256"] == done["adapter_sha256"]["2"]
    a = [json.loads(x) for x in gzip.open(tmp_path / "test-s0.jsonl.gz", "rt")]
    b = [json.loads(x) for x in gzip.open(tmp_path / "test-s0e1.jsonl.gz", "rt")]
    assert len(a) == 7 and L.max_prob_diff(a, b) > 0  # each job used its own adapter
    with pytest.raises(L.AlreadyScored):
        L.main(score)

    # a reload check against responses the adapter did not produce stops before any test unit is scored
    bad = {"units": te, "jobs": [{**job["jobs"][0], "out": str(tmp_path / "test-bad.jsonl.gz"),
                                  "check": {"units": va, "responses": str(out / "val-epoch-1.jsonl.gz"), "n": 10}}]}
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    if L.max_prob_diff([json.loads(x) for x in gzip.open(out / "val-epoch-1.jsonl.gz", "rt")],
                       [json.loads(x) for x in gzip.open(out / "val-epoch-2.jsonl.gz", "rt")]) > L.RELOAD_TOL:
        with pytest.raises(SystemExit, match="reloaded adapter differs"):
            L.main(["score", "--job", str(tmp_path / "bad.json"), "--report", str(tmp_path / "r2.json"), "--device", "cpu", "--model-path", tiny])
        assert not (tmp_path / "test-bad.jsonl.gz").exists()


def test_cli_smoke_report(tiny, tmp_path):
    lm, tok, _ = L.load_base("cpu", tiny)
    ref_units = units(5)
    ref, _ = L.score_units(lm, L.scorer_for(L.text_runtime(lm, tok), "cpu"), ref_units)
    tr = write_units(tmp_path / "train.jsonl.gz", units(40))
    rf = write_units(tmp_path / "ref.jsonl.gz", ref)
    L.main(["smoke", "--units", tr, "--reference", rf, "--out", str(tmp_path / "smoke.json"), "--scratch", str(tmp_path),
            "--device", "cpu", "--model-path", tiny])
    rep = json.loads((tmp_path / "smoke.json").read_text())
    assert rep["untrained_vs_base"]["max_abs_prob_diff"] == 0.0 and rep["reload"]["ok"]
    assert len(rep["training"]["steps"]) == 2 and rep["training"]["tokens_per_second"] > 0
    assert not (tmp_path / "smoke-adapter").exists()
