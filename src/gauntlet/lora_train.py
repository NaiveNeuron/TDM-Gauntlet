"""LoRA fine-tuning of Clef's base model on option-token targets, and option scoring with the adapter
(driver `jev-opt/experiments/h100_lora.py`).

Runs in the repository's environment with the `lora` extra (the `clef` pins plus PEFT):

    uv run --extra lora python -m gauntlet.lora_train train --units train.jsonl.gz --val-units val.jsonl.gz \
        --lr 1e-4 --seed 0 --out runs/jev-opt/lora-<set>/lr1e-04-s0 --adapters data/cache/lora-qwen-base/<set>/lr1e-04-s0 \
        --device cuda:0
    ... score --job job.json --device cuda:0      # test scoring with saved adapters (refuses existing outputs)
    ... smoke --units train.jsonl.gz --reference ref.jsonl.gz --out smoke.json --scratch DIR --device cuda:0
    ... check                                       # versions, pinned snapshot, static backward checks; loads no model
    ... preflight --device cuda:0 --out pre.json    # compiler, driver, TileLang backward (status 4 on failure)

Units files are gzipped JSON lines `{"key", "prompt", "continuations", "target"?}` written by the driver (prompts of
template qwen-base-logprob-v1). Training: the pinned Qwen3.8-27B in BF16 (frozen), LoRA rank 16 / alpha 32 /
dropout 0.05 on the text decoder's projections (FP32 adapter weights), next-token cross-entropy over the full
vocabulary (FP32 log-softmax) on the target option's single token at the position after the prompt, micro-batch 1,
16 micro-batches per step, AdamW, linear warm-up (10%) and decay, clipping 1.0, gradient checkpointing. After each
epoch the adapter is saved and the validation units are scored with `gauntlet.logprob_serve.LogprobScorer` (the
base model's option log-probability scoring code, unchanged) on the model with the adapter attached.

Every command first points CUDA_HOME/CUDA_PATH/PATH at the pinned standalone CUDA 12.8.1 compiler
(`lora_preflight`), which FLA's TileLang backward of the gated delta rule needs on Hopper.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import lora_preflight as PF
from .logprob_serve import HF_CACHE, MODEL, LogprobScorer, Runtime, _version

# Pre-registered settings.
TARGET_MODULES = (r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
                  r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate_proj|up_proj|down_proj))")
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none", "target_modules": TARGET_MODULES}
TRAIN = {"micro_batch": 1, "grad_accum": 16, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0, "warmup_ratio": 0.1,
         "schedule": "linear warm-up then linear decay to 0 (transformers.get_linear_schedule_with_warmup)",
         "max_grad_norm": 1.0, "gradient_checkpointing": "non-reentrant, every decoder layer",
         "loss": "next-token cross-entropy, FP32 log-softmax over the full vocabulary, target option token only",
         "precision": "BF16 frozen base, FP32 LoRA weights and AdamW states, no autocast"}
WINDOW = 64  # units per scoring window (the jev-opt client's window for the base model's log-prob scoring)
GUARD = 32000
RELOAD_TOL = 0.01
SMOKE_REFERENCE_TOL = 0.05
LIBRARIES = ("torch", "transformers", "peft", "accelerate", "safetensors", "tokenizers", "huggingface-hub",
             "flash-linear-attention", "fla-core", "triton", "causal-conv1d", "tilelang")


class AlreadyScored(RuntimeError):
    """An output of `score` exists already: a test split is scored once."""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def say(msg: str) -> None:
    print(f"{now()} {msg}", file=sys.stderr, flush=True)


def read_units(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Gzipped JSON lines, written atomically (a partial file never has the final name)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.GzipFile(tmp, "wb", mtime=0) as raw:
        raw.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8"))
    tmp.replace(path)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(path)


def sha256_dir(path: Path) -> str:
    """SHA-256 over a saved adapter's files (name and content, sorted by name)."""
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(f.relative_to(path).as_posix().encode() + b"\0")
        h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()


def libraries() -> dict[str, str | None]:
    return {**{name: _version(name) for name in LIBRARIES}, "CUDA_HOME": os.environ.get("CUDA_HOME")}


# --------------------------------------------------------------------------------------------
# model


def parse_devices(device: str) -> list[str]:
    """`cuda:0`, `cpu`, or the pre-registered OOM fallback `cuda:0+cuda:1` (layers split over both GPUs)."""
    return device.split("+")


def load_base(device: str, model_path: str | None = None) -> tuple[Any, Any, dict[str, Any]]:
    """(model, tokenizer, provenance): the pinned release from data/cache/huggingface only, or a local directory
    (tests: a tiny random model of the same class)."""
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    if model_path is None:
        from huggingface_hub import snapshot_download

        model_id, revision = MODEL.split("@", 1)
        path = Path(snapshot_download(model_id, revision=revision, cache_dir=str(HF_CACHE), local_files_only=True))
        source = MODEL
    else:
        path, source = Path(model_path), f"local:{model_path}"
    devices = parse_devices(device)
    if len(devices) == 1:
        placement: dict[str, Any] = {"device_map": {"": devices[0]}}
    else:
        placement = {"device_map": "balanced", "max_memory": {int(d.split(":")[1]): "76GiB" for d in devices}}
    t0 = time.monotonic()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    lm = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                         local_files_only=True, **placement)
    lm.config.use_cache = False
    for p in lm.parameters():
        p.requires_grad_(False)
    return lm, tok, {"model": source, "device": device, "load_seconds": round(time.monotonic() - t0, 3)}


def attach_lora(lm: Any, seed: int) -> Any:
    """LoRA on the text decoder's projections; the seed fixes the initialization (B = 0, A Kaiming-uniform)."""
    import torch
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    model = get_peft_model(lm, LoraConfig(**LORA))
    for name, p in model.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            raise RuntimeError(f"trainable parameter {name} is {p.dtype}, expected float32")
    return model


def text_runtime(lm: Any, tok: Any) -> Runtime:
    """The scorer's runtime on the (adapter-carrying) model: text decoder, output projection, tokenizer."""
    text = lm.model.language_model if hasattr(lm.model, "language_model") else lm.model
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    return Runtime(text, lm.get_output_embeddings(), lambda s: tok(s, add_special_tokens=False).input_ids, int(pad))


def scorer_for(rt: Runtime, device: str) -> LogprobScorer:
    sc = LogprobScorer(device=parse_devices(device)[0], max_length=GUARD, model=MODEL, runtime_factory=lambda: rt)
    sc.runtime()  # every template answer must be one token after the prompt (ContinuationError otherwise)
    return sc


def score_units(model: Any, scorer: LogprobScorer, units: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """The base model's log-prob scoring of each unit (windows of 64, length buckets inside); rows
    `{"key", "response"}` or `{"key", "error", "type"}` in unit order."""
    model.eval()
    rows: list[dict[str, Any]] = []
    t0 = time.monotonic()
    for i in range(0, len(units), WINDOW):
        rows += scorer.compute([(u["key"], {"prompt": u["prompt"], "continuations": u["continuations"]}) for u in units[i:i + WINDOW]])
    tokens = sum(r["response"]["usage"]["input_tokens"] for r in rows if "response" in r)
    return rows, {"units": len(rows), "rejected": sum("error" in r for r in rows), "input_tokens": tokens,
                  "wall_seconds": round(time.monotonic() - t0, 3),
                  "inference_seconds": round(sum(r["response"]["audit"]["inference_seconds"] for r in rows if "response" in r), 3)}


def max_prob_diff(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> float:
    """Largest absolute difference of option probabilities between two scorings of the same units."""
    worst = 0.0
    for x, y in zip(a, b, strict=True):
        if x["key"] != y["key"] or ("response" in x) != ("response" in y):
            return math.inf
        if "response" in x:
            px, py = x["response"]["probabilities"], y["response"]["probabilities"]
            if len(px) != len(py):
                return math.inf
            worst = max(worst, *(abs(float(u) - float(v)) for u, v in zip(px, py)))
    return worst


def _sync(device: str) -> None:
    import torch

    for d in parse_devices(device):
        if d.startswith("cuda"):
            torch.cuda.synchronize(d)


def _peak_gib(device: str) -> float | None:
    import torch

    devs = [d for d in parse_devices(device) if d.startswith("cuda")]
    return round(sum(torch.cuda.max_memory_allocated(d) for d in devs) / 2**30, 3) if devs else None


# --------------------------------------------------------------------------------------------
# training


def encode_targets(scorer: LogprobScorer, units: list[dict[str, Any]]) -> list[tuple[list[int], int]]:
    """(prompt ids, target token id) per training unit; a multi-token answer or a prompt over the guard stops."""
    out = []
    for u in units:
        ids, cont = scorer.encode({"prompt": u["prompt"], "continuations": u["continuations"]})
        t = u["target"]
        if not isinstance(t, int) or not 0 <= t < len(cont):
            raise ValueError(f"unit {u['key'][:16]}: target {t!r} outside its {len(cont)} options")
        out.append((ids, cont[t]))
    return out


def unit_loss(rt: Runtime, ids: list[int], target: int) -> Any:
    """−log p(target | prompt): FP32 log-softmax over the vocabulary at the last prompt position."""
    import torch

    dev = rt.text_model.get_input_embeddings().weight.device
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    hidden = rt.text_model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, return_dict=True).last_hidden_state
    head_dev = next(rt.head.parameters()).device
    logp = torch.log_softmax(rt.head(hidden[0, -1].to(head_dev)).float(), dim=-1)
    return -logp[target]


def train(model: Any, rt: Runtime, scorer: LogprobScorer, units: list[dict[str, Any]], *, lr: float, seed: int, epochs: int,
          device: str, out: Path | None = None, adapters: Path | None = None, val_units: list[dict[str, Any]] | None = None,
          max_steps: int | None = None) -> dict[str, Any]:
    """The pre-registered training loop. After each epoch (if `adapters`/`out` are given): adapter saved to
    `adapters/epoch-<e>/`, val scored to `out/val-epoch-<e>.jsonl.gz`. `max_steps` (smoke only) stops early."""
    import torch
    from transformers import get_linear_schedule_with_warmup

    data = encode_targets(scorer, units)
    accum = TRAIN["grad_accum"]
    per_epoch = math.ceil(len(data) / accum)
    total = epochs * per_epoch
    warm = math.ceil(TRAIN["warmup_ratio"] * total)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=tuple(TRAIN["betas"]), eps=TRAIN["eps"], weight_decay=TRAIN["weight_decay"])
    sched = get_linear_schedule_with_warmup(opt, warm, total)
    model.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    torch.manual_seed(seed)  # dropout stream
    log: dict[str, Any] = {"lr": lr, "seed": seed, "epochs": epochs, "n_units": len(data), "steps_per_epoch": per_epoch,
                           "total_steps": total, "warmup_steps": warm, "trainable_parameters": sum(p.numel() for p in params),
                           "steps": [], "epoch_summaries": []}
    step = 0
    for epoch in range(1, epochs + 1):
        order = list(range(len(data)))
        random.Random(1000 * seed + epoch).shuffle(order)
        model.train()
        _sync(device)
        t_epoch, tokens_epoch = time.monotonic(), 0
        for s in range(per_epoch):
            batch = order[s * accum:(s + 1) * accum]
            _sync(device)
            t0 = time.monotonic()
            total_loss = 0.0
            for i in batch:
                ids, target = data[i]
                loss = unit_loss(rt, ids, target) / len(batch)
                loss.backward()
                total_loss += float(loss.detach())
                tokens_epoch += len(ids)
            if not math.isfinite(total_loss):
                raise FloatingPointError(f"non-finite loss {total_loss} at step {step + 1}")
            norm = float(torch.nn.utils.clip_grad_norm_(params, TRAIN["max_grad_norm"]))
            used_lr = sched.get_last_lr()[0]
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            _sync(device)
            step += 1
            log["steps"].append({"step": step, "epoch": epoch, "loss": round(total_loss, 6), "grad_norm": round(norm, 6),
                                 "lr": used_lr, "units": len(batch), "tokens": sum(len(data[i][0]) for i in batch),
                                 "seconds": round(time.monotonic() - t0, 3)})
            if max_steps is not None and step >= max_steps:
                break
        _sync(device)
        summary: dict[str, Any] = {"epoch": epoch, "train_seconds": round(time.monotonic() - t_epoch, 3), "train_tokens": tokens_epoch,
                                   "mean_loss": round(sum(x["loss"] for x in log["steps"] if x["epoch"] == epoch)
                                                      / max(1, sum(x["epoch"] == epoch for x in log["steps"])), 6),
                                   "peak_memory_gib": _peak_gib(device)}
        if adapters is not None:
            dest = adapters / f"epoch-{epoch}"
            if dest.exists():
                shutil.rmtree(dest)
            model.save_pretrained(dest)
            summary["adapter_sha256"] = sha256_dir(dest)
        if val_units is not None and out is not None:
            rows, use = score_units(model, scorer, val_units)
            write_rows(out / f"val-epoch-{epoch}.jsonl.gz", rows)
            summary["val_scoring"] = use
        log["epoch_summaries"].append(summary)
        say(f"epoch {epoch}: mean loss {summary['mean_loss']}, {summary['train_seconds']:.0f} s, peak {summary['peak_memory_gib']} GiB")
        if max_steps is not None and step >= max_steps:
            break
    model.eval()
    return log


def cmd_train(a: argparse.Namespace) -> None:
    out, adapters = Path(a.out), Path(a.adapters)
    if (out / "done.json").exists():
        raise SystemExit(f"{out}: finished already (done.json)")
    units, val_units = read_units(Path(a.units)), read_units(Path(a.val_units))
    lm, tok, prov = load_base(a.device, a.model_path)
    model = attach_lora(lm, a.seed)
    rt = text_runtime(lm, tok)
    sc = scorer_for(rt, a.device)
    t0 = time.monotonic()
    log = train(model, rt, sc, units, lr=a.lr, seed=a.seed, epochs=a.epochs, device=a.device, out=out, adapters=adapters,
                val_units=val_units)
    log.update(provenance=prov, lora=LORA, train=TRAIN, configuration={**sc.configuration, "libraries": libraries()},
               wall_seconds=round(time.monotonic() - t0, 3), finished_at=now())
    dump_json(out / "train_log.json", log)
    dump_json(out / "done.json", {"finished_at": log["finished_at"], "epochs": a.epochs,
                                  "adapter_sha256": {s["epoch"]: s["adapter_sha256"] for s in log["epoch_summaries"]}})


# --------------------------------------------------------------------------------------------
# scoring with saved adapters (test)


def cmd_score(a: argparse.Namespace) -> None:
    """Job file: {"units": path, "jobs": [{"name", "adapter", "out", "check": {"units", "responses", "n"}}]}. Every
    output must be new; before scoring, each adapter rescores the original validation windows containing
    the first n check units and must reproduce those n recorded responses within RELOAD_TOL."""
    from peft import PeftModel

    job = json.loads(Path(a.job).read_text())
    for j in job["jobs"]:
        if Path(j["out"]).exists():
            raise AlreadyScored(f"{j['out']} exists: a test split is scored once")
    units = read_units(Path(job["units"]))
    lm, tok, prov = load_base(a.device, a.model_path)
    model = None
    report: dict[str, Any] = {"provenance": prov, "jobs": []}
    for k, j in enumerate(job["jobs"]):
        name = f"job{k}"
        if model is None:
            model = PeftModel.from_pretrained(lm, j["adapter"], adapter_name=name, is_trainable=False)
        else:
            model.load_adapter(j["adapter"], adapter_name=name, is_trainable=False)
        model.set_adapter(name)
        sc = scorer_for(text_runtime(lm, tok), a.device)
        chk = j["check"]
        ref = read_units(Path(chk["responses"]))
        cunits = {u["key"]: u for u in read_units(Path(chk["units"]))}
        context_n = ((chk["n"] + WINDOW - 1) // WINDOW) * WINDOW
        again, _ = score_units(model, sc, [cunits[r["key"]] for r in ref[:context_n]])
        diff = max_prob_diff(ref[:chk["n"]], again[:chk["n"]])
        if diff > RELOAD_TOL:
            raise SystemExit(f"{j['name']}: reloaded adapter differs from the recorded val responses (max |dp| {diff:.4g})")
        rows, use = score_units(model, sc, units)
        if Path(j["out"]).exists():
            raise AlreadyScored(f"{j['out']} appeared during scoring")
        write_rows(Path(j["out"]), rows)
        report["jobs"].append({"name": j["name"], "adapter_sha256": sha256_dir(Path(j["adapter"])), "reload_max_diff": diff,
                               "scoring": use, "peak_memory_gib": _peak_gib(a.device)})
        say(f"{j['name']}: scored {use['units']} units ({use['rejected']} rejected) in {use['wall_seconds']:.0f} s")
    report["configuration"] = {**sc.configuration, "libraries": libraries()}
    dump_json(Path(a.report), report)


# --------------------------------------------------------------------------------------------
# smoke test and environment check


def cmd_smoke(a: argparse.Namespace) -> None:
    """(1) untrained adapter vs the base model's recorded responses; (2) 2 steps of 16 units, lr 1e-4, seed 0;
    (3) save, reload, rescore 16 units; the report carries the measured training throughput."""
    units, ref = read_units(Path(a.units)), read_units(Path(a.reference))
    scratch = Path(a.scratch)
    lm, tok, prov = load_base(a.device, a.model_path)
    model = attach_lora(lm, 0)
    rt = text_runtime(lm, tok)
    sc = scorer_for(rt, a.device)
    by_key = {u["key"]: u for u in units}
    base_rows, base_use = score_units(model, sc, [by_key[r["key"]] for r in ref])
    base_diff = max_prob_diff(ref, base_rows)
    log = train(model, rt, sc, units[: 2 * TRAIN["grad_accum"]], lr=1e-4, seed=0, epochs=1, device=a.device, max_steps=2)
    probe = units[:16]
    live, _ = score_units(model, sc, probe)
    dest = scratch / "smoke-adapter"
    if dest.exists():
        shutil.rmtree(dest)
    model.save_pretrained(dest)
    model.load_adapter(dest, adapter_name="reloaded", is_trainable=False)
    model.set_adapter("reloaded")
    again, _ = score_units(model, sc, probe)
    reload_diff = max_prob_diff(live, again)
    steps = log["steps"]
    secs, toks = sum(s["seconds"] for s in steps), sum(s["tokens"] for s in steps)
    report = {"provenance": prov, "n_reference_units": len(ref), "untrained_vs_base": {"max_abs_prob_diff": base_diff, "tol": SMOKE_REFERENCE_TOL,
                                                                                         "ok": base_diff <= SMOKE_REFERENCE_TOL, "scoring": base_use},
              "training": {"steps": steps, "seconds": secs, "tokens": toks, "tokens_per_second": toks / secs if secs else None,
                           "trainable_parameters": log["trainable_parameters"], "peak_memory_gib": _peak_gib(a.device)},
              "reload": {"max_abs_prob_diff": reload_diff, "tol": RELOAD_TOL, "ok": reload_diff <= RELOAD_TOL},
              "lora": LORA, "train": TRAIN, "configuration": {**sc.configuration, "libraries": libraries()}, "computed_at": now()}
    dump_json(Path(a.out), report)
    shutil.rmtree(dest)
    if not (report["untrained_vs_base"]["ok"] and report["reload"]["ok"]):
        raise SystemExit(f"smoke failed: untrained vs base {base_diff:.4g}, reload {reload_diff:.4g}")


def cmd_check(a: argparse.Namespace) -> None:
    """Versions, the pinned snapshot's presence and static backward checks; loads no model and
    touches no GPU."""
    import peft  # noqa: F401

    model_id, revision = MODEL.split("@", 1)
    snap = HF_CACHE / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    backward = PF.static_report()
    print(json.dumps({"libraries": libraries(), "python": sys.version.split()[0], "model": MODEL,
                      "snapshot": str(snap), "snapshot_present": (snap / "config.json").exists(), "lora": LORA, "train": TRAIN,
                      "backward": backward, "backward_problems": PF.static_problems(backward)}))


def cmd_preflight(a: argparse.Namespace) -> None:
    """Preflight on one GPU: static checks, sm_90a compile, driver, backend choice and the TileLang
    backward against the PyTorch reference. Writes the report; exits with status 4 on any problem."""
    static = PF.static_report()
    problems = PF.static_problems(static)
    comp = PF.compile_check()
    problems += PF.compile_problems(comp)
    gpu = PF.gpu_report(a.device) if not problems else None
    if gpu is not None:
        problems += PF.gpu_problems(gpu)
    report = {"device": a.device, "static": static, "compile": comp, "gpu": gpu, "problems": problems, "tol": PF.TOL,
              "libraries": libraries(), "computed_at": now()}
    dump_json(Path(a.out), report)
    print(json.dumps({"device": a.device, "errors": (gpu or {}).get("errors"), "problems": problems}))
    if problems:
        say("preflight failed: " + "; ".join(problems))
        raise SystemExit(4)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--units", required=True)
    t.add_argument("--val-units", required=True)
    t.add_argument("--lr", type=float, required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--out", required=True)
    t.add_argument("--adapters", required=True)
    s = sub.add_parser("score")
    s.add_argument("--job", required=True)
    s.add_argument("--report", required=True)
    m = sub.add_parser("smoke")
    m.add_argument("--units", required=True)
    m.add_argument("--reference", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--scratch", required=True)
    sub.add_parser("check")
    f = sub.add_parser("preflight")
    f.add_argument("--device", required=True)
    f.add_argument("--out", required=True)
    for p in (t, s, m):
        p.add_argument("--device", required=True)
        p.add_argument("--model-path", help="tests only: a local model directory instead of the pinned snapshot")
    a = ap.parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    PF.pin_cuda_compiler()  # before FLA or TileLang resolve the pinned compiler/cache
    {"train": cmd_train, "score": cmd_score, "smoke": cmd_smoke, "check": cmd_check, "preflight": cmd_preflight}[a.command](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
