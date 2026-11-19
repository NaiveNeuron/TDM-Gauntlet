"""H100 LoRA (pre-registered): Qwen3.8-27B fine-tuned with LoRA on the pipeline's 500 train labels.

Per RQ3 set (the 13 sets of clef_rq3.py):

  runs/jev-opt/lora-<stem>/                 data.json (train targets, exclusions, unit digests), one directory per
                                            training run lr<lr>-s<seed>/ (train_log.json, done.json, val-epoch-<e>.jsonl.gz:
                                            the val units scored after each epoch), val.json (decoded val curves),
                                            selection.json (best epoch per run, learning rate by mean val over seeds),
                                            test-scoring-<n>.json (trainer reports of the test scoring)
  runs/jev-opt/lora-<stem>-s<seed>/         the chosen learning rate's seed run at its best epoch on test, once:
                                            test_responses.jsonl.gz, result.json, test_consumed.json
  data/cache/lora-qwen-base/<stem>/         units files and adapters (scratch; untested adapters are deleted
                                            once the set's tests are recorded)

The units are the seed program's (state, question) units rendered with template qwen-base-logprob-v1
(`jev_opt.logprob.render_unit`), as in H100 round 2 Experiment 1; the target is the option that, answered with
certainty, makes the seed program predict the gold label. Training and scoring run in the repository's `lora`
environment (`gauntlet.lora_train`); val and test answers are decoded here by the executor exactly as Experiment 1's
zero-shot (`jev_opt.logprob.answer` + `run_decider`).

Subcommands (repository root, jev-opt environment):
  run      --device cuda:N --set <stem>     one set: training runs (resumable), val decoding, selection, test once
  smoke    --device cuda:N [--set <stem>]   untrained adapter vs Experiment 1's responses, 2 steps, reload check,
                                            throughput -> projected GPU hours -> number of seeds (3 if <= 80, else 1)
  replay   [--set <stem>]                   re-decode the recorded val and test responses (no device)
  plan     [--device ...] [--slot gpuN]     dry run: inputs, weights, trainer environment, targets, state; no model
  workload [--set <stem>]                   prompt tokens of the train/val/test units (base tokenizer)
  assign | estimate                         two-GPU assignment from the workload; expected / measured GPU hours
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

assert sys.version_info[:2] == (3, 11), f"run under Python 3.11 (worker's version), not {sys.version}"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clef_rq3 as C  # noqa: E402  (sets, run_config, recorder, helpers)
import h100_round2 as H2  # noqa: E402  (unit identity, weights, frozen rule, tokenizer)

from jev_opt import metrics  # noqa: E402
from jev_opt.execute import ExampleResult, run_decider  # noqa: E402
from jev_opt.hygiene import allow_test_split, consumed, load_split, record_consumption, sha256_file  # noqa: E402
from jev_opt.ir import load_decider  # noqa: E402
from jev_opt.jev import Decision  # noqa: E402
from jev_opt.logprob import PINS, TEMPLATE_ID, TEMPLATE_SHA256, LogprobRejected, TemplateError, answer, render_unit  # noqa: E402

ROOT = HERE.parents[1]
SETS = C.SETS
CFG8 = list(SETS)[:8]
PIN = PINS["qwen-base"]
# Pre-registered.
GRID: dict[str, Any] = {"lrs": (1e-4, 2e-4), "epochs": 3, "seeds": (0, 1, 2)}
SEED_RULE_GPU_HOURS = 80.0
SMOKE_SET = "smarter-latent-hate"
SMOKE_REFERENCE_UNITS = 32
RELOAD_CHECK_UNITS = 16
LOAD_MINUTES = 3.0  # model loading per training run and per test-scoring job (estimate)
TRAIN_SLOWDOWN = 4.0  # training tokens/s = Exp. 1 inference tokens/s / 4 (estimate; the smoke test measures it)
WORKLOAD = HERE / "h100_lora_workload.json"
ASSIGNMENT = HERE / "h100_lora_assignment.json"
TRAINER_VERSIONS = {"torch": "2.11.0+cu128", "transformers": "5.10.2", "peft": "0.21.2", "accelerate": "1.15.0",
                    "flash-linear-attention": "0.5.2", "triton": "3.6.0", "tilelang": "0.1.12"}

# Output locations and trainer (tests point these elsewhere; the pre-registered values are the defaults).
OUT = C.RUNS
SCRATCH = ROOT / "data" / "cache" / "lora-qwen-base"
# The gauntlet environment as synced once with `--extra clef --extra lora` (never re-synced by a run).
TRAINER: list[str] = ["uv", "run", "--no-sync", "--project", str(ROOT), "--extra", "lora", "python", "-m", "gauntlet.lora_train"]
MODEL_PATH: str | None = None  # tests only: a tiny local model instead of the pinned snapshot
LIMIT: int | None = None  # tests only: first n examples of each split

now, dump_json, load_json = C.now, C.dump_json, C.load_json


# --------------------------------------------------------------------------------------------
# paths and logging


def set_dir(stem: str) -> Path:
    return OUT / f"lora-{stem}"


def test_dir(stem: str, seed: int) -> Path:
    return OUT / f"lora-{stem}-s{seed}"


def smoke_dir(stem: str) -> Path:
    return OUT / f"lora-smoke-{stem}"


def run_name(lr: float, seed: int) -> str:
    return f"lr{lr:.0e}-s{seed}"


def units_path(stem: str, split: str) -> Path:
    return SCRATCH / stem / "units" / f"{split}.jsonl.gz"


def adapter_dir(stem: str, run: str, epoch: int | None = None) -> Path:
    d = SCRATCH / stem / "adapters" / run
    return d if epoch is None else d / f"epoch-{epoch}"


def log(msg: str) -> None:
    line = f"{now()} {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "h100_lora.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.GzipFile(tmp, "wb", mtime=0) as raw:
        raw.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8"))
    tmp.replace(path)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------------
# units and targets


class OneHot:
    """Client stand-in answering every question with probability 1 on option `i` (Exp. 1's answer format)."""

    concurrency = 1

    def __init__(self, i: int):
        self.i = i

    def decide(self, state: Any, questions: dict[str, Any], context: dict[str, Any] | None = None) -> Decision:
        ans = {}
        for q, spec in questions.items():
            n = len(render_unit(state, spec)[1])
            ans[q] = answer(spec, {"probabilities": [1.0 if k == self.i else 0.0 for k in range(n)]})
        return Decision(ans, "one-hot", 0, 0.0, 0.0, True)


class Table:
    """Client answering from scored unit rows (`{"key", "response"}` or `{"key", "error", "type"}`), keyed by the
    unit identity of H100 round 2; a rejected unit raises LogprobRejected (a failed example), a missing one stops."""

    concurrency = 16

    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = {r["key"]: r for r in rows}

    def decide(self, state: Any, questions: dict[str, Any], context: dict[str, Any] | None = None) -> Decision:
        ans = {}
        for q, spec in questions.items():
            key = H2.unit_identity(state, spec)
            row = self.rows.get(key)
            if row is None:
                raise RuntimeError(f"unit {key[:16]} has no scored response")
            if "error" in row:
                raise LogprobRejected(f"{row.get('type')}: {row['error'][:300]}")
            ans[q] = answer(spec, row["response"])
        return Decision(ans, PIN, 0, 0.0, 0.0, True)


def source(stem: str) -> tuple[Path, Any, str, Any]:
    src = C.RUNS / SETS[stem][0]
    text = (src / "seed.yaml").read_text()
    return src, C.run_config(src), text, load_decider(text)


def limited(rows: list[Any]) -> list[Any]:
    return rows if LIMIT is None else rows[:LIMIT]


def the_unit(seed: Any, ex: Any) -> tuple[Any, dict[str, Any]]:
    rec = C._Recorder()
    run_decider(seed, [ex], rec)  # type: ignore[arg-type]
    if len(rec.bodies) != 1 or len(rec.bodies[0][1]) != 1:
        raise SystemExit(f"example {ex.id}: the seed program asks {sum(len(q) for _, q in rec.bodies)} questions (expected 1)")
    state, qs = rec.bodies[0]
    return state, next(iter(qs.values()))


def render(state: Any, spec: dict[str, Any]) -> dict[str, Any]:
    try:
        prompt, cont = render_unit(state, spec)
    except TemplateError as exc:
        raise SystemExit(f"the seed question does not fit template {TEMPLATE_ID}: {exc}") from None
    return {"key": H2.unit_identity(state, spec), "prompt": prompt, "continuations": cont}


def target_of(seed: Any, ex: Any, n_options: int) -> int | None:
    """The option that, answered with certainty, makes the seed program predict the gold label (None: no option)."""
    hits = [i for i in range(n_options) if run_decider(seed, [ex], OneHot(i))[0].pred == ex.label]  # type: ignore[arg-type]
    if len(hits) > 1:
        raise SystemExit(f"example {ex.id}: options {hits} all decode to the gold label {ex.label!r}")
    return hits[0] if hits else None


def train_units(stem: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """(training units with targets, excluded examples, number of train examples)."""
    _, cfg, _, seed = source(stem)
    train = limited(load_split(cfg, "train"))
    units, excluded = [], []
    for ex in train:
        u = render(*the_unit(seed, ex))
        t = target_of(seed, ex, len(u["continuations"]))
        if t is None:
            excluded.append({"id": ex.id, "gold": ex.label})
        else:
            units.append({"id": ex.id, **u, "target": t})
    return units, excluded, len(train)


def split_units(seed: Any, examples: list[Any]) -> list[dict[str, Any]]:
    """Unique units of a split in first-seen order."""
    out, seen = [], set()
    for ex in examples:
        u = render(*the_unit(seed, ex))
        if u["key"] not in seen:
            seen.add(u["key"])
            out.append(u)
    return out


def prepare(stem: str) -> dict[str, Any]:
    """Units files (scratch) and data.json (kept). A rerun recomputes both and stops if data.json differs."""
    src, cfg, _, seed = source(stem)
    units, excluded, n_train = train_units(stem)
    val = limited(load_split(cfg, "val"))
    vunits = split_units(seed, val)
    counts: dict[str, int] = {}
    for u in units:
        counts[u["continuations"][u["target"]]] = counts.get(u["continuations"][u["target"]], 0) + 1
    data = {"set": stem, "source_run": src.name, "model": PIN, "template": {"id": TEMPLATE_ID, "sha256": TEMPLATE_SHA256},
            "seed_program_hash": seed.digest(), "labels": list(cfg.labels), "primary_metric": cfg.primary_metric,
            "limit": LIMIT,
            "train": {"n_examples": n_train, "n_units": len(units), "excluded": excluded, "target_tokens": dict(sorted(counts.items())),
                      "units_sha256": digest([[u["id"], u["key"], u["target"]] for u in units])},
            "val": {"n_examples": len(val), "n_units": len(vunits), "units_sha256": digest([u["key"] for u in vunits])}}
    path = set_dir(stem) / "data.json"
    old = load_json(path)
    if old is not None and old != data:
        raise SystemExit(f"{path}: recomputed training data differ from the recorded ones")
    dump_json(path, data)
    write_rows(units_path(stem, "train"), units)
    write_rows(units_path(stem, "val"), vunits)
    return data


# --------------------------------------------------------------------------------------------
# trainer subprocess


def trainer(args: list[str], capture: bool = False) -> str:
    cmd = [*TRAINER, *args, *(["--model-path", MODEL_PATH] if MODEL_PATH and args[0] != "check" else [])]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "h100_lora.log.stderr", "a", encoding="utf-8") as err:
        err.write(f"{now()} $ {shlex.join(cmd)}\n")
        err.flush()
        p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE if capture else err, stderr=err, text=True,
                           env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
    if p.returncode != 0:
        raise SystemExit(f"trainer {args[0]} failed with status {p.returncode} (see {OUT / 'h100_lora.log.stderr'})")
    return p.stdout if capture else ""


def train_run(stem: str, lr: float, seed: int, device: str) -> None:
    run = run_name(lr, seed)
    out = set_dir(stem) / run
    if (out / "done.json").exists():
        return
    for d in (out, adapter_dir(stem, run)):  # an interrupted run restarts from its beginning
        if d.exists():
            shutil.rmtree(d)
    log(f"{stem} {run} training (device {device})")
    trainer(["train", "--units", str(units_path(stem, "train")), "--val-units", str(units_path(stem, "val")), "--lr", repr(lr),
             "--seed", str(seed), "--epochs", str(GRID["epochs"]), "--out", str(out), "--adapters", str(adapter_dir(stem, run)),
             "--device", device])
    if not (out / "done.json").exists():
        raise SystemExit(f"{out}: trainer finished without done.json")


# --------------------------------------------------------------------------------------------
# decoding, selection, test


def decode(seed: Any, examples: list[Any], rows: list[dict[str, Any]], what: str) -> tuple[list[ExampleResult], list[dict[str, str]]]:
    res = run_decider(seed, examples, Table(rows))  # type: ignore[arg-type]
    bad = [r for r in res if r.error is not None]
    infra = [r for r in bad if not r.error.startswith("LogprobRejected")]
    if infra:
        raise SystemExit(f"{what}: {len(infra)}/{len(res)} examples failed, e.g. {infra[0].error[:400]}")
    return res, [{"id": r.id, "error": r.error[:300]} for r in bad]


def primary(cfg: Any, gold: list[str], pred: list[str | None]) -> float:
    return metrics.primary(cfg.primary_metric, gold, pred, list(cfg.labels), cfg.negative_label)


def val_curves(stem: str, seeds: tuple[int, ...]) -> dict[str, Any]:
    """Val primary after each epoch of every run (executor decoding of the scored val units)."""
    _, cfg, _, seed = source(stem)
    val = limited(load_split(cfg, "val"))
    gold = [e.label for e in val]
    out: dict[str, Any] = {}
    for lr in GRID["lrs"]:
        for s in seeds:
            run = run_name(lr, s)
            curve, rej = [], []
            for e in range(1, GRID["epochs"] + 1):
                res, bad = decode(seed, val, read_rows(set_dir(stem) / run / f"val-epoch-{e}.jsonl.gz"), f"{stem} {run} val epoch {e}")
                curve.append(primary(cfg, gold, [r.pred for r in res]))
                rej.append(len(bad))
            out[run] = {"lr": lr, "seed": s, "val_primary": curve, "rejected": rej}
    return out


def choose(curves: dict[str, Any], seeds: tuple[int, ...]) -> dict[str, Any]:
    """Best epoch per run (ties: earlier); learning rate by the mean over seeds of the best-epoch val (ties: lower)."""
    best = {run: 1 + max(range(len(c["val_primary"])), key=lambda e, c=c: (c["val_primary"][e], -e)) for run, c in curves.items()}
    mean = {lr: statistics.mean(curves[run_name(lr, s)]["val_primary"][best[run_name(lr, s)] - 1] for s in seeds) for lr in GRID["lrs"]}
    lr = max(GRID["lrs"], key=lambda x: (mean[x], -x))
    return {"best_epoch": best, "mean_best_val": {repr(k): v for k, v in mean.items()}, "chosen_lr": lr,
            "tested": [{"seed": s, "run": run_name(lr, s), "epoch": best[run_name(lr, s)],
                        "val_primary": curves[run_name(lr, s)]["val_primary"][best[run_name(lr, s)] - 1]} for s in seeds]}


def select(stem: str, seeds: tuple[int, ...]) -> dict[str, Any]:
    curves = val_curves(stem, seeds)
    dump_json(set_dir(stem) / "val.json", {"set": stem, "runs": curves})
    sel = {"rule": "best val epoch per run (ties: earlier); learning rate by mean best-epoch val over seeds (ties: lower)",
           "seeds": list(seeds), "grid": {"lrs": list(GRID["lrs"]), "epochs": GRID["epochs"]}, **choose(curves, seeds)}
    for t in sel["tested"]:
        done = json.loads((set_dir(stem) / t["run"] / "done.json").read_text())
        t["adapter_sha256"] = done["adapter_sha256"][str(t["epoch"])]
    path = set_dir(stem) / "selection.json"
    old = load_json(path)
    if old is not None:
        if {k: v for k, v in old.items() if k != "computed_at"} != sel:
            raise SystemExit(f"{path}: recomputed selection differs from the recorded one")
        return old
    sel["computed_at"] = now()
    dump_json(path, sel)
    return sel


def test_result(stem: str, t: dict[str, Any], cfg: Any, seed: Any, test: list[Any], rows: list[dict[str, Any]],
                scoring: dict[str, Any] | None) -> dict[str, Any]:
    res, bad = decode(seed, test, rows, f"{stem} test seed {t['seed']}")
    labels = list(cfg.labels)
    gold, pred = [e.label for e in test], [r.pred for r in res]
    assert [r.id for r in res] == [e.id for e in test]
    frozen = H2.score_frozen_rule(seed, res, labels)
    return {"set": stem, "seed": t["seed"], "lr": float(t["run"].split("-s")[0][2:]), "epoch": t["epoch"], "run": t["run"],
            "model": PIN, "adapter_sha256": t["adapter_sha256"], "template": {"id": TEMPLATE_ID, "sha256": TEMPLATE_SHA256},
            "source_run": C.RUNS.joinpath(SETS[stem][0]).name, "labels": labels, "primary_metric": cfg.primary_metric,
            "val_primary": t["val_primary"],
            "test": {"ids": [e.id for e in test], "gold": gold, "pred": pred, "primary": primary(cfg, gold, pred),
                     "frozen_rule_pred": frozen, "frozen_rule_primary": None if frozen is None else primary(cfg, gold, frozen)},
            "rejected": bad, "scoring": scoring, "dists": [r.dist for r in res]}


def scoring_entry(stem: str, name: str) -> dict[str, Any] | None:
    for p in sorted(set_dir(stem).glob("test-scoring-*.json")):
        for j in json.loads(p.read_text())["jobs"]:
            if j["name"] == name:
                return {**j, "report": p.name}
    return None


def test_step(stem: str, sel: dict[str, Any], device: str | None) -> list[dict[str, Any]]:
    """Each tested seed run on test, once: score (if not yet scored), decode, record; a recorded one is replayed."""
    _, cfg, _, seed = source(stem)
    with allow_test_split():
        test = limited(load_split(cfg, "test"))
    jobs = []
    for t in sel["tested"]:
        td = test_dir(stem, t["seed"])
        if not consumed(td) and not (td / "test_responses.jsonl.gz").exists():
            jobs.append(t)
    if jobs:
        if device is None:
            raise SystemExit(f"{stem}: test responses missing and no device")
        write_rows(units_path(stem, "test"), split_units(seed, test))
        n = len(list(set_dir(stem).glob("test-scoring-*.json"))) + 1
        job = {"units": str(units_path(stem, "test")), "jobs": [
            {"name": f"s{t['seed']}", "adapter": str(adapter_dir(stem, t["run"], t["epoch"])),
             "out": str(test_dir(stem, t["seed"]) / "test_responses.jsonl.gz"),
             "check": {"units": str(units_path(stem, "val")), "responses": str(set_dir(stem) / t["run"] / f"val-epoch-{t['epoch']}.jsonl.gz"),
                       "n": RELOAD_CHECK_UNITS}} for t in jobs]}
        for t in jobs:
            test_dir(stem, t["seed"]).mkdir(parents=True, exist_ok=True)
        jpath = SCRATCH / stem / f"test-job-{n}.json"
        dump_json(jpath, job)
        log(f"{stem} test scoring: seeds {[t['seed'] for t in jobs]} (device {device})")
        trainer(["score", "--job", str(jpath), "--report", str(set_dir(stem) / f"test-scoring-{n}.json"), "--device", device])
    out = []
    for t in sel["tested"]:
        td = test_dir(stem, t["seed"])
        rows = read_rows(td / "test_responses.jsonl.gz")
        r = test_result(stem, t, cfg, seed, test, rows, scoring_entry(stem, f"s{t['seed']}"))
        dists = r.pop("dists")
        rec = load_json(td / "result.json")
        if consumed(td):
            C.check_replay(td, "LoRA test predictions", rec["test"], r["test"])
            out.append(rec)
            continue
        r["computed_at"] = now()
        dump_json(td / "result.json", r)
        path = cfg.split_path("test")
        record_consumption(td, {"representative": "lora", "candidate": "h100-lora", "model": PIN, "adapter_sha256": t["adapter_sha256"],
                                "program_hash": seed.digest(), "test_path": str(path), "test_sha256": sha256_file(path),
                                "metrics": metrics.compute(r["test"]["gold"], r["test"]["pred"], dists, list(cfg.labels), cfg.primary_metric,
                                                           cfg.negative_label),
                                "step": "lora-test", "lr": r["lr"], "epoch": t["epoch"], "seed": t["seed"]})
        out.append(r)
    return out


def prune(stem: str, sel: dict[str, Any]) -> None:
    """Delete the adapters other than the tested ones once every tested seed is recorded."""
    if not all(consumed(test_dir(stem, t["seed"])) for t in sel["tested"]):
        return
    keep = {adapter_dir(stem, t["run"], t["epoch"]) for t in sel["tested"]}
    root = SCRATCH / stem / "adapters"
    gone = [d for d in sorted(root.glob("*/epoch-*")) if d not in keep] if root.exists() else []
    for d in gone:
        shutil.rmtree(d)
    if gone:
        log(f"{stem} deleted {len(gone)} untested adapters; kept {sorted(str(k.relative_to(SCRATCH)) for k in keep)}")


# --------------------------------------------------------------------------------------------
# seeds (smoke rule) and commands


def seed_plan() -> tuple[int, ...]:
    rep = load_json(smoke_dir(SMOKE_SET) / "smoke.json")
    if rep is None:
        raise SystemExit(f"{smoke_dir(SMOKE_SET) / 'smoke.json'} missing: run the smoke test first (it fixes the number of seeds)")
    return tuple(rep["projection"]["seeds"])


def cmd_run(device: str, stem: str) -> None:
    seeds = seed_plan()
    t0 = time.monotonic()
    data = prepare(stem)
    log(f"{stem} data: {data['train']['n_units']} training units ({len(data['train']['excluded'])} without a target), "
        f"{data['val']['n_units']} val units; seeds {list(seeds)}")
    for lr in GRID["lrs"]:
        for s in seeds:
            train_run(stem, lr, s, device)
    sel = select(stem, seeds)
    log(f"{stem} selection: lr {sel['chosen_lr']}, epochs {[t['epoch'] for t in sel['tested']]}, mean val {sel['mean_best_val']}")
    res = test_step(stem, sel, device)
    prune(stem, sel)
    log(f"{stem} test: {[round(r['test']['primary'], 4) for r in res]} (seeds {[r['seed'] for r in res]}); "
        f"{time.monotonic() - t0:.0f} s")


def cmd_replay(stem: str) -> None:
    sel = load_json(set_dir(stem) / "selection.json")
    if sel is None:
        raise SystemExit(f"{stem}: no selection recorded")
    recorded = json.loads((set_dir(stem) / "val.json").read_text())["runs"]
    if val_curves(stem, tuple(sel["seeds"])) != recorded:
        raise SystemExit(f"{stem}: re-decoded val curves differ from val.json")
    select(stem, tuple(sel["seeds"]))
    if not all(consumed(test_dir(stem, t["seed"])) for t in sel["tested"]):
        raise SystemExit(f"{stem}: not every tested seed is recorded")
    test_step(stem, sel, None)
    log(f"{stem} replay identical")


def exp1_reference(stem: str) -> dict[str, dict[str, Any]]:
    """unit identity -> Experiment 1's stored response (h2-qwen-base-<stem>-cal ledger: fit, val and test units)."""
    out: dict[str, dict[str, Any]] = {}
    for p in sorted((C.RUNS / f"h2-qwen-base-{stem}-cal").glob("lp_responses*.jsonl.gz")):
        for row in read_rows(p):
            req = row["request"]
            out[H2.unit_identity(req["state"], req["question"])] = row["response"]
    return out


def throughput() -> dict[str, float]:
    """Experiment 1's measured base-model input tokens per inference second, per set (results/H100_ROUND2.json)."""
    rows = json.loads((ROOT / "results" / "H100_ROUND2.json").read_text())["base"]["sets"]
    return {r["set"]: r["fresh_input_tokens"] / r["fresh_inference_s"] for r in rows}


def expected_hours(stem: str, work: dict[str, Any], n_seeds: int, train_tps: float | None = None) -> dict[str, float]:
    tput = throughput()[stem]
    tps = train_tps or tput / TRAIN_SLOWDOWN
    w, runs, E = work[stem], len(GRID["lrs"]) * n_seeds, GRID["epochs"]
    h = {"train": runs * E * w["train_tokens"] / tps / 3600, "val": runs * E * w["val_tokens"] / tput / 3600,
         "test": n_seeds * w["test_tokens"] / tput / 3600, "load": (runs + 1) * LOAD_MINUTES / 60}
    h["total"] = sum(h.values())
    return h


def cmd_smoke(device: str, stem: str) -> None:
    out = smoke_dir(stem)
    units, _, _ = train_units(stem)
    ref = exp1_reference(stem)
    pairs = [{"key": u["key"], "response": ref[u["key"]]} for u in units if u["key"] in ref][:SMOKE_REFERENCE_UNITS]
    if not pairs:
        raise SystemExit(f"{stem}: no training unit has an Experiment 1 response")
    write_rows(units_path(stem, "train"), units)
    rpath = SCRATCH / stem / "smoke" / "reference.jsonl.gz"
    write_rows(rpath, pairs)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    trainer(["smoke", "--units", str(units_path(stem, "train")), "--reference", str(rpath), "--out", str(out / "trainer.json"),
             "--scratch", str(SCRATCH / stem / "smoke"), "--device", device])
    rep = json.loads((out / "trainer.json").read_text())
    tps = rep["training"]["tokens_per_second"]
    work = json.loads(WORKLOAD.read_text())
    per = {s: expected_hours(s, work, len(GRID["seeds"]), tps)["total"] for s in SETS}
    total = sum(per.values())
    seeds = list(GRID["seeds"]) if total <= SEED_RULE_GPU_HOURS else [GRID["seeds"][0]]
    report = {"set": stem, "device": device, "wall_s_incl_load": round(time.monotonic() - t0, 3),
              "untrained_vs_exp1": rep["untrained_vs_base"], "reload": rep["reload"],
              "training": {k: rep["training"][k] for k in ("seconds", "tokens", "tokens_per_second", "trainable_parameters", "peak_memory_gib")},
              "projection": {"train_tokens_per_second": tps, "per_set_gpu_hours_3_seeds": per, "total_gpu_hours_3_seeds": total,
                             "threshold": SEED_RULE_GPU_HOURS, "seeds": seeds},
              "computed_at": now()}
    dump_json(out / "smoke.json", report)
    log(f"smoke {stem}: untrained vs Exp. 1 max |dp| {rep['untrained_vs_base']['max_abs_prob_diff']:.4g}, reload {rep['reload']['max_abs_prob_diff']:.4g}, "
        f"{tps:.0f} training tokens/s, peak {rep['training']['peak_memory_gib']} GiB; projected {total:.1f} GPU-h for 3 seeds -> seeds {seeds}")


# --------------------------------------------------------------------------------------------
# workload, assignment, estimate, plan


def set_workload(stem: str, tok: Any) -> dict[str, Any]:
    _, cfg, _, seed = source(stem)
    units, excluded, n = train_units(stem)
    with allow_test_split():
        test = load_split(cfg, "test")
    out: dict[str, Any] = {"n_train": n, "n_train_units": len(units), "n_excluded": len(excluded)}
    for name, us in (("train", units), ("val", split_units(seed, load_split(cfg, "val"))), ("test", split_units(seed, test))):
        lens = [len(e.ids) for e in tok.encode_batch([u["prompt"] for u in us], add_special_tokens=False)]
        out.update({f"{name}_units": len(lens), f"{name}_tokens": sum(lens), f"{name}_max_tokens": max(lens)})
    return out


def cmd_workload(stems: list[str]) -> None:
    tok = H2._tokenizer()
    data = load_json(WORKLOAD) or {}
    for stem in stems:
        data[stem] = set_workload(stem, tok)
        print(f"{stem}: {json.dumps(data[stem])}", flush=True)
        dump_json(WORKLOAD, data)


def cmd_assign() -> None:
    """Sets by expected GPU hours (3 seeds), largest first, each to the GPU with the smaller running total (ties: gpu0)."""
    work = json.loads(WORKLOAD.read_text())
    est = {s: expected_hours(s, work, len(GRID["seeds"])) for s in SETS}
    load, slots = {"gpu0": 0.0, "gpu1": 0.0}, {"gpu0": [], "gpu1": []}  # type: ignore[var-annotated]
    for stem in sorted(SETS, key=lambda s: (-est[s]["total"], s)):
        g = min(load, key=lambda x: (load[x], x))
        slots[g].append(stem)
        load[g] += est[stem]["total"]
    dump_json(ASSIGNMENT, {"rule": cmd_assign.__doc__, "slots": slots, "expected_gpu_hours": load,
                           "per_set": {s: {k: round(v, 3) for k, v in est[s].items()} for s in SETS}})
    print(json.dumps({"slots": slots, "expected_gpu_hours": load}, indent=1))


def measured_hours(stem: str) -> float | None:
    sel = load_json(set_dir(stem) / "selection.json")
    if sel is None or not all(consumed(test_dir(stem, t["seed"])) for t in sel["tested"]):
        return None
    s = 0.0
    for lr in GRID["lrs"]:
        for seed in sel["seeds"]:
            lg = json.loads((set_dir(stem) / run_name(lr, seed) / "train_log.json").read_text())
            s += lg["provenance"]["load_seconds"] + sum(e["train_seconds"] + e["val_scoring"]["wall_seconds"] for e in lg["epoch_summaries"])
    for p in set_dir(stem).glob("test-scoring-*.json"):
        r = json.loads(p.read_text())
        s += r["provenance"]["load_seconds"] + sum(j["scoring"]["wall_seconds"] for j in r["jobs"])
    return s / 3600


def estimate_table() -> tuple[str, dict[str, float]]:
    assign = json.loads(ASSIGNMENT.read_text())
    work = json.loads(WORKLOAD.read_text())
    lines = ["| GPU | Set | Train units | Train tokens | Val tokens | Test tokens | GPU h |", "|---|---|---:|---:|---:|---:|---:|"]
    total = {"gpu0": 0.0, "gpu1": 0.0}
    for g in ("gpu0", "gpu1"):
        for stem in assign["slots"][g]:
            w = work[stem]
            m = measured_hours(stem)
            hours, tag = (m, "measured") if m is not None else (expected_hours(stem, work, len(GRID["seeds"]))["total"], "est.")
            total[g] += hours
            lines.append(f"| {g} | {stem} | {w['n_train_units']} | {w['train_tokens'] / 1e6:.2f} M | {w['val_tokens'] / 1e6:.2f} M | "
                         f"{w['test_tokens'] / 1e6:.2f} M | {hours:.2f} ({tag}) |")
    for g in ("gpu0", "gpu1"):
        lines.append(f"| {g} | **total** | | | | | **{total[g]:.1f}** |")
    return "\n".join(lines), total


def cmd_estimate() -> None:
    """Expected GPU hours (no inference): 6 runs × 3 epochs of the train tokens at Exp. 1's measured per-set inference
    throughput / 4, val scoring 3× per run and test once per seed at that throughput, 3 min loading per run and test job."""
    table, total = estimate_table()
    print(table)
    print(f"\nTotal {sum(total.values()):.1f} GPU-hours. Throughput source: results/H100_ROUND2.json (base model, per set).")


def cmd_plan(device: str | None, slot: str | None, stems: list[str]) -> dict[str, Any]:
    """Dry run: assignment, weights, trainer environment, inputs, training targets and state; loads no model."""
    problems: list[str] = []
    if slot is not None:
        slots = json.loads(ASSIGNMENT.read_text())["slots"]
        if sorted(s for v in slots.values() for s in v) != sorted(SETS):
            problems.append("assignment does not cover each set exactly once")
        stems = slots[slot]
    wp, weights = H2.weights_problems("base")
    problems += wp
    env: dict[str, Any] = {}
    try:
        env = json.loads(trainer(["check"], capture=True))
        for k, v in TRAINER_VERSIONS.items():
            if env["libraries"].get(k) != v:
                problems.append(f"trainer environment: {k} {env['libraries'].get(k)} (expected {v})")
        if not env["snapshot_present"]:
            problems.append(f"trainer: pinned snapshot missing at {env['snapshot']}")
        problems += [f"trainer backward preflight: {p}" for p in env["backward_problems"]]
    except SystemExit as exc:
        problems.append(f"trainer environment check failed: {exc}")
    smoke = load_json(smoke_dir(SMOKE_SET) / "smoke.json")
    rows = []
    for stem in stems:
        units, excluded, n = train_units(stem)
        if n != 500 and LIMIT is None:
            problems.append(f"{stem}: {n} train rows (expected 500)")
        sel = load_json(set_dir(stem) / "selection.json")
        done = sum((set_dir(stem) / run_name(lr, s) / "done.json").exists() for lr in GRID["lrs"] for s in GRID["seeds"])
        state = {"training_runs_done": done, "selection": sel is not None,
                 "tests_recorded": [t["seed"] for t in (sel or {}).get("tested", []) if consumed(test_dir(stem, t["seed"]))]}
        rows.append({"set": stem, "train_units": len(units), "excluded": len(excluded), "state": state})
    plan = {"model": PIN, "device": device, "slot": slot, "weights": weights, "trainer": env.get("libraries"),
            "smoke": None if smoke is None else smoke["projection"]["seeds"], "sets": rows, "problems": problems, "checked_at": now()}
    print(json.dumps({k: v for k, v in plan.items() if k != "sets"}, indent=1))
    for r in rows:
        print(f"{r['set']:24s} train units {r['train_units']} (excluded {r['excluded']}); {r['state']}")
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["run", "smoke", "replay", "plan", "workload", "assign", "estimate"])
    ap.add_argument("--device", help="cuda:N (or the OOM fallback cuda:0+cuda:1)")
    ap.add_argument("--slot", choices=["gpu0", "gpu1"])
    ap.add_argument("--set", action="append", dest="sets", choices=list(SETS))
    ap.add_argument("--out", type=Path, help="plan: also write the plan as JSON here")
    a = ap.parse_args()
    stems = a.sets or list(SETS)
    if a.command == "workload":
        cmd_workload(stems)
    elif a.command == "assign":
        cmd_assign()
    elif a.command == "estimate":
        cmd_estimate()
    elif a.command == "plan":
        plan = cmd_plan(a.device, a.slot, stems)
        if a.out:
            dump_json(a.out, plan)
        return 1 if plan["problems"] else 0
    elif a.command == "replay":
        for s in stems:
            cmd_replay(s)
    elif a.command == "smoke":
        if not a.device:
            ap.error("smoke needs --device")
        cmd_smoke(a.device, (a.sets or [SMOKE_SET])[0])
    else:
        if not a.device or len(stems) != 1:
            ap.error("run needs --device and exactly one --set")
        cmd_run(a.device, stems[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
