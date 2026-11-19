"""H100 round 2 (pre-registered): interface vs backbone, and the optimizer's prior.

Per RQ3 set (the 13 sets of clef_rq3.py) and experiment:

  base    Clef's base model Qwen3.8-27B through option log-probabilities (`jev_opt.logprob`, JEV_OPT_BACKEND=qwen-base)
    cal     runs/jev-opt/h2-qwen-base-<stem>-cal/  the paper's question (seed.yaml) on the fit slice, val and test; zero-shot
            = its test predictions; bias + temperature fitted on the fit slice and applied to val and test (one test
            evaluation holds both)
    opt     runs/jev-opt/h2-qwen-base-<stem>-opt/  clef_rq3.step_opt unchanged: P and P+head readouts refitted on the base
            model's fit answers, 1.0-point rule on its val, the chosen program answered on test once
  native  Clef (`jev_opt.clef`) on the label-free jev-native seed (`top-zero-jev-native.yaml` of the source run)
    native     runs/jev-opt/h2-clef-<stem>-native/     as written, on val and test
    native-lr  runs/jev-opt/h2-clef-<stem>-native-lr/  the paper's readout head (`fit.with_readouts`) refitted on Clef's
               exhaustive fit answers (`framing_features.refit`), on val and test

Subcommands (repository root, jev-opt environment):
  run      --exp base|native --device cuda:N --set <stem>   both steps of one set (resumable from the cache)
  smoke    --exp ... --device ... --set <stem> --n 20       the experiment's programs on the first n fit items, then a
                                                            cache-only replay compared answer by answer
  replay   --exp ... [--set <stem>]                         cache-only re-execution of finished sets (no device)
  plan     --exp ... [--device ...] [--slot gpu0|gpu1]      dry run: inputs, pins, weights, caches, template limits,
                                                            frozen-unit identity; loads no model, makes no request
  workload [--set <stem>]                                   planned units/tokens (base) and requests/characters (native)
  assign                                                    the pre-registered two-GPU assignment from the workload
  estimate                                                  expected GPU hours per set and GPU (measured once run)
  frozen-units                                              (preparation machine; PYTHONPATH=src) digests of the frozen
                                                            gauntlet test requests' units, checked against the seed program
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import statistics
import sys
import time
from pathlib import Path
from typing import Any

assert sys.version_info[:2] == (3, 11), f"run under Python 3.11 (worker's version), not {sys.version}"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clef_rq3 as C  # noqa: E402  (Env, steps and helpers of the Clef RQ3 driver)
import framing_features as ff  # noqa: E402  (its import disables Jev posting)

from jev_opt import metrics  # noqa: E402
from jev_opt.adapter import train_roles  # noqa: E402
from jev_opt.clef import PINS as CLEF_PINS  # noqa: E402
from jev_opt.clef import ClefBackend  # noqa: E402
from jev_opt.clef import server_command as clef_server_command  # noqa: E402
from jev_opt.execute import ExampleResult, combine, questions_for, run_decider, run_examples  # noqa: E402
from jev_opt.fit import fit_label_bias, with_readouts  # noqa: E402
from jev_opt.hygiene import allow_test_split, consumed, load_split  # noqa: E402
from jev_opt.ir import load_decider  # noqa: E402
from jev_opt.jev import ResponseLedger, make_client  # noqa: E402
from jev_opt.logprob import PINS as LP_PINS  # noqa: E402
from jev_opt.logprob import TEMPLATE_ID, TEMPLATE_SHA256, LogprobBackend, TemplateError, render_unit  # noqa: E402
from jev_opt.logprob import server_command as lp_server_command  # noqa: E402

ROOT = HERE.parents[1]
RUNS = ROOT / "runs" / "jev-opt"
SETS = C.SETS
CFG8 = list(SETS)[:8]
REPS = C.REPS
EXPS: dict[str, dict[str, Any]] = {
    "base": {"model": "qwen-base", "pin": LP_PINS["qwen-base"], "ledger": "lp_responses", "steps": ("cal", "opt")},
    "native": {"model": "clef", "pin": CLEF_PINS["clef"], "ledger": "clef_responses", "steps": ("native", "native-lr")},
}
NATIVE = "top-zero-jev-native.yaml"
WORKLOAD = HERE / "h100_round2_workload.json"
ASSIGNMENT = HERE / "h100_round2_assignment.json"
FROZEN_UNITS = HERE / "h100_round2_frozen_units.json"
WEIGHTS = HERE / "h100_round2_qwen_weights.sha256"
HF = ROOT / "data" / "cache" / "huggingface"
REJECTED = ("ClefRejected", "LogprobRejected")

now, dump_json, load_json, floored = C.now, C.dump_json, C.load_json, C.floored
counters, delta, consume, check_replay, with_ledger = C.counters, C.delta, C.consume, C.check_replay, C.with_ledger


def step_dir(exp: str, stem: str, step: str) -> Path:
    return RUNS / f"h2-{EXPS[exp]['model']}-{stem}-{step}"


def log(exp: str, msg: str) -> None:
    line = f"{now()} {msg}"
    print(line, flush=True)
    with open(RUNS / f"h100_round2-{exp}.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def snapshot(pin: str) -> Path:
    model_id, rev = pin.split("@")
    return HF / f"models--{model_id.replace('/', '--')}" / "snapshots" / rev


class Env(C.Env):
    """clef_rq3.Env with the experiment's ledger name; encoder, guard and template rejections count as failed
    examples, every other failure (cache miss, server error) aborts the set."""

    def __init__(self, run_dir: Path, client: Any, test: bool, ledger_name: str):
        super().__init__(run_dir, client, test)
        self.ledger_name = ledger_name
        self.candidate = "h100-round2"  # clef_rq3.consume's test-ledger label

    def run(self, program: Any, examples: list[Any], split: str, exhaustive: bool = False) -> list[ExampleResult]:
        ctx = {"split": split, "condition": "h100-round2"}
        if exhaustive:
            res = run_examples(program, examples, self.client, exhaustive=True, context=ctx, pool=self.pool)
        else:
            res = run_decider(program, examples, self.client, context=ctx, pool=self.pool)
        bad = [r for r in res if r.error is not None]
        infra = [r for r in bad if not r.error.startswith(REJECTED)]
        if infra:
            raise SystemExit(f"{self.dir.name} {split}: {len(infra)}/{len(res)} examples failed, e.g. {infra[0].error[:400]}")
        self.rejected += [{"split": split, "id": r.id, "error": r.error[:300]} for r in bad]
        return res


def client_for(exp: str, device: str | None) -> Any:
    """The experiment's backend through the executor's client factory (JEV_OPT_BACKEND / JEV_OPT_DEVICE)."""
    os.environ["JEV_OPT_BACKEND"] = EXPS[exp]["model"]
    if device:
        os.environ["JEV_OPT_DEVICE"] = device
    else:
        os.environ.pop("JEV_OPT_DEVICE", None)
    client = make_client(RUNS / ".jev-cache", "", 0, 0, [ROOT])
    assert isinstance(client, LogprobBackend if exp == "base" else ClefBackend) and client.model == EXPS[exp]["pin"]
    return client


def score_frozen_rule(seed: Any, res: list[ExampleResult], labels: list[str]) -> list[str] | None:
    """DREsS: the gauntlet's frozen rule on the seed's Score answer (argmax level, ties to the lower level)."""
    nid = seed.root
    if seed.nodes[nid].kind != "score" or len(seed.nodes[nid].levels) != len(labels):
        return None
    out: list[str | None] = []
    for r in res:
        keys = [k for k in r.memo if k.split("|", 1)[0] == nid]
        if len(keys) != 1:
            out.append(None)
            continue
        raw = r.memo[keys[0]][nid].get("probabilities", {})
        p = [float(raw.get(str(i), 0.0)) for i in range(len(labels))]
        out.append(labels[max(range(len(p)), key=lambda i: (p[i], -i))])
    return out


# --------------------------------------------------------------------------------------------
# Experiment 1: base model


def step_base_cal(env: Env, src: Path, out: Path) -> dict[str, Any]:
    labels = env.labels
    seed_text = (src / "seed.yaml").read_text()
    seed = load_decider(seed_text)
    rec = load_json(out / "result.json") or {}
    out.mkdir(parents=True, exist_ok=True)
    (out / "seed.yaml").write_text(seed_text)
    t: dict[str, Any] = {}

    def answer() -> tuple[list, list, list]:
        c0 = counters(env.client)
        fit_res = env.run(seed, env.fit, "train")
        c1 = counters(env.client)
        val_res = env.run(seed, env.val, "val")
        c2 = counters(env.client)
        test_res = env.run(seed, env.test, "test")
        t.update(fit=delta(c0, c1), val=delta(c1, c2), test=delta(c2, counters(env.client)))
        return fit_res, val_res, test_res

    fit_res, val_res, test_res = with_ledger(env, out, answer)
    dists = [floored(r.dist, labels) for r in fit_res]
    bias, temp, rep = fit_label_bias(labels, [[d] for d in dists], [r.gold for r in fit_res], env.config.primary_metric,
                                     env.config.negative_label)

    def cal(res: list[ExampleResult]) -> tuple[list[dict[str, float]], list[str]]:
        both = [combine(labels, [floored(r.dist, labels)], bias, temp) for r in res]
        return [b[0] for b in both], [b[1] for b in both]

    val_gold = [r.gold for r in val_res]
    test_ids, test_gold = [e.id for e in env.test], [e.label for e in env.test]
    assert [r.id for r in test_res] == test_ids
    zero = [r.pred for r in test_res]
    cal_dists, cal_pred = cal(test_res)
    frozen_rule = score_frozen_rule(seed, test_res, labels)
    result = {
        "set": stem_of(src), "model": env.client.model, "template": {"id": TEMPLATE_ID, "sha256": TEMPLATE_SHA256}, "source_run": src.name,
        "labels": labels, "primary_metric": env.config.primary_metric, "fit_ids": [e.id for e in env.fit],
        "calibration": {"bias": bias, "temperature": temp, "fit_primary_before_after": [rep["before"], rep["after"]], "n_fit": len(fit_res)},
        "fit": {"gold": [r.gold for r in fit_res], "seed_pred": [r.pred for r in fit_res]},
        "val": {"ids": [r.id for r in val_res], "gold": val_gold, "seed_pred": [r.pred for r in val_res], "calibrated_pred": cal(val_res)[1],
                "seed_primary": env.primary(val_gold, [r.pred for r in val_res]), "calibrated_primary": env.primary(val_gold, cal(val_res)[1])},
        "test": {"ids": test_ids, "gold": test_gold, "zero_pred": zero, "zero_primary": env.primary(test_gold, zero),
                 "calibrated_pred": cal_pred, "calibrated_primary": env.primary(test_gold, cal_pred),
                 "frozen_rule_zero_pred": frozen_rule,
                 "frozen_rule_zero_primary": None if frozen_rule is None else env.primary(test_gold, frozen_rule)},
        "rejected": env.rejected, "usage": rec.get("usage") or t, "computed_at": rec.get("computed_at") or now(),
    }
    if consumed(out):
        check_replay(out, "zero-shot and calibrated test predictions", rec["test"], result["test"])
    dump_json(out / "result.json", result)
    if not consumed(out):
        c = env.config
        zero_m = metrics.compute(test_gold, zero, [r.dist for r in test_res], list(c.labels), c.primary_metric, c.negative_label)
        consume(out, "calibrated", seed_text, env, test_gold, cal_pred, cal_dists, {"step": "cal", "zero_shot_metrics": zero_m})
    return result


# --------------------------------------------------------------------------------------------
# Experiment 2: Clef on the jev-native seed


def native_text(src: Path) -> str:
    return (src / NATIVE).read_text()


def step_native(env: Env, src: Path, out: Path) -> dict[str, Any]:
    text = native_text(src)
    prog = load_decider(text)
    rec = load_json(out / "result.json") or {}
    out.mkdir(parents=True, exist_ok=True)
    (out / "native.yaml").write_text(text)
    t: dict[str, Any] = {}
    holder: dict[str, Any] = {}

    def answer() -> None:
        c0 = counters(env.client)
        holder["val"] = env.run(prog, env.val, "val")
        c1 = counters(env.client)
        holder["test"] = env.run(prog, env.test, "test")
        t.update(val=delta(c0, c1), test=delta(c1, counters(env.client)))

    with_ledger(env, out, answer)
    vres, tres = holder["val"], holder["test"]
    val_gold, test_gold = [r.gold for r in vres], [e.label for e in env.test]
    test = {"ids": [r.id for r in tres], "gold": test_gold, "pred": [r.pred for r in tres], "primary": env.primary(test_gold, [r.pred for r in tres])}
    result = {"set": stem_of(src), "model": env.client.model, "source_run": src.name, "program": NATIVE, "program_hash": prog.digest(),
              "val": {"ids": [r.id for r in vres], "pred": [r.pred for r in vres], "primary": env.primary(val_gold, [r.pred for r in vres])},
              "test": test, "rejected": env.rejected, "usage": rec.get("usage") or t, "computed_at": rec.get("computed_at") or now()}
    if consumed(out):
        check_replay(out, "native zero-shot test predictions", rec["test"], test)
    dump_json(out / "result.json", result)
    if not consumed(out):
        consume(out, "native-zero", text, env, test_gold, test["pred"], [r.dist for r in tres], {"step": "native"})
    return result


def step_native_lr(env: Env, src: Path, out: Path) -> dict[str, Any]:
    blank = with_readouts(load_decider(native_text(src)), env.config.primary_metric)
    rec = load_json(out / "result.json") or {}
    out.mkdir(parents=True, exist_ok=True)
    (out / "native-lr.unfitted.yaml").write_text(blank.to_yaml())
    t: dict[str, Any] = {}
    holder: dict[str, Any] = {}

    def answer() -> None:
        c0 = counters(env.client)
        try:
            fit = ff.refit(env, blank)  # type: ignore[arg-type]  # Env duck-types ff.Env
        except ValueError as exc:
            raise SystemExit(f"{out.name}: the readout could not be refitted: {str(exc)[:500]}") from None
        c1 = counters(env.client)
        (out / "native-lr.yaml").write_text(fit.to_yaml())
        holder["fit"] = fit
        holder["val"] = env.run(fit, env.val, "val")
        c2 = counters(env.client)
        holder["test"] = env.run(fit, env.test, "test")
        t.update(fit=delta(c0, c1), val=delta(c1, c2), test=delta(c2, counters(env.client)))

    with_ledger(env, out, answer)
    fit, vres, tres = holder["fit"], holder["val"], holder["test"]
    val_gold, test_gold = [r.gold for r in vres], [e.label for e in env.test]
    test = {"ids": [r.id for r in tres], "gold": test_gold, "pred": [r.pred for r in tres], "primary": env.primary(test_gold, [r.pred for r in tres])}
    result = {"set": stem_of(src), "model": env.client.model, "source_run": src.name, "program": NATIVE, "program_hash": fit.digest(),
              "readout_fit": {s: fit.nodes[s].readout.fit for s in ff.readout_slots(fit)},
              "val": {"ids": [r.id for r in vres], "pred": [r.pred for r in vres], "primary": env.primary(val_gold, [r.pred for r in vres])},
              "test": test, "rejected": env.rejected, "usage": rec.get("usage") or t, "computed_at": rec.get("computed_at") or now()}
    if consumed(out):
        check_replay(out, "native refitted test predictions", rec["test"], test)
    dump_json(out / "result.json", result)
    if not consumed(out):
        consume(out, "native-lr", fit.to_yaml(), env, test_gold, test["pred"], [r.dist for r in tres], {"step": "native-lr"})
    return result


def stem_of(src: Path) -> str:
    return next(s for s, v in SETS.items() if v[0] == src.name)


# --------------------------------------------------------------------------------------------
# commands


def steps(exp: str, stem: str) -> list[tuple[str, Any, Path]]:
    if exp == "base":
        return [("cal", lambda env, src, out: step_base_cal(env, src, out), step_dir(exp, stem, "cal")),
                ("opt", lambda env, src, out: C.step_opt(EXPS[exp]["model"], stem, env, src, out), step_dir(exp, stem, "opt"))]
    return [("native", step_native, step_dir(exp, stem, "native")), ("native-lr", step_native_lr, step_dir(exp, stem, "native-lr"))]


def summary(exp: str, step: str, r: dict[str, Any]) -> str:
    if step == "cal":
        return (f"val {r['val']['seed_primary']:.4f} -> {r['val']['calibrated_primary']:.4f}; test zero-shot {r['test']['zero_primary']:.4f}, "
                f"calibrated {r['test']['calibrated_primary']:.4f}")
    if step == "opt":
        return f"val {r['selection']['val']} chosen {r['selection']['chosen']}, test {r['test']['primary']:.4f}"
    return f"val {r['val']['primary']:.4f}, test {r['test']['primary']:.4f}"


def cmd_run(exp: str, device: str | None, stem: str) -> None:
    src = RUNS / SETS[stem][0]
    client = client_for(exp, device)
    t0 = time.monotonic()
    try:
        env = Env(src, client, test=True, ledger_name=EXPS[exp]["ledger"])
        for step, fn, out in steps(exp, stem):
            log(exp, f"{stem} {step} start (device {device or 'cache-only'})")
            r = fn(env, src, out)
            log(exp, f"{stem} {step} done: {summary(exp, step, r)}; usage {json.dumps(r['usage'])}")
            env.rejected = []
    finally:
        client.close()
    log(exp, f"{stem} finished in {time.monotonic() - t0:.0f} s; fresh calls {client.usage.fresh_calls}, "
             f"fresh inference {client.fresh_inference_seconds:.0f} s")


def cmd_replay(exp: str, stem: str) -> None:
    """Cache-only re-execution (no device): every recorded val/test prediction must come back."""
    src = RUNS / SETS[stem][0]
    client = client_for(exp, None)
    try:
        env = Env(src, client, test=True, ledger_name=EXPS[exp]["ledger"])
        for step, fn, out in steps(exp, stem):
            before = json.loads((out / "result.json").read_text())
            after = fn(env, src, out)  # check_replay compares the test predictions; val compared here
            key = "representatives" if step == "opt" else "val"
            if json.dumps(before[key], sort_keys=True) != json.dumps(after[key], sort_keys=True):
                raise SystemExit(f"{out.name}: replayed {key} differs from the recorded one")
            log(exp, f"{stem} {step} replay identical (fresh calls {client.usage.fresh_calls})")
            env.rejected = []
    finally:
        client.close()


def smoke_programs(exp: str, src: Path, metric: str) -> dict[str, tuple[Any, bool]]:
    if exp == "base":
        progs = {"seed": (load_decider((src / "seed.yaml").read_text()), False)}
        progs.update({n: (ff.unfitted(load_decider((src / f"{n}.yaml").read_text()), drop_aux=False), True) for n in REPS})
        return progs
    native = load_decider(native_text(src))
    return {"native": (native, False), "native-lr": (with_readouts(native, metric), True)}


def cmd_smoke(exp: str, device: str, stem: str, n: int) -> None:
    src = RUNS / SETS[stem][0]
    out = step_dir(exp, "smoke", stem)  # runs/jev-opt/h2-<model>-smoke-<stem>

    def answers(client: Any) -> tuple[dict[str, Any], dict[str, float]]:
        env = Env(src, client, test=False, ledger_name=EXPS[exp]["ledger"])
        got: dict[str, Any] = {}
        c0 = counters(client)
        for name, (prog, exhaustive) in smoke_programs(exp, src, env.config.primary_metric).items():
            res = env.run(prog, env.fit[:n], "train", exhaustive=exhaustive)
            got[name] = [{"id": r.id, "memo": r.memo, "dist": r.dist, "pred": r.pred, "error": r.error} for r in res]
        return got, {**delta(c0, counters(client)), "rejected": len(env.rejected)}

    live = client_for(exp, device)
    out.mkdir(parents=True, exist_ok=True)
    led = ResponseLedger(out, name=EXPS[exp]["ledger"])
    live.ledger = led
    t0 = time.monotonic()
    try:
        first, use1 = answers(live)
    finally:
        live.ledger = None
        shards = led.finalize()
        live.close()
    wall1 = time.monotonic() - t0
    replay = client_for(exp, None)
    try:
        second, use2 = answers(replay)
    finally:
        replay.close()
    same = json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    report: dict[str, Any] = {"set": stem, "experiment": exp, "model": EXPS[exp]["pin"], "device": device, "n_fit_items": n,
                              "programs": list(first), "live": {**use1, "wall_s_incl_load": round(wall1, 3)}, "replay": use2,
                              "replay_identical": same, "ledger": shards, "computed_at": now()}
    conf = live._config_sha and json.loads((live.cache_dir / "configurations" / f"{live._config_sha}.json").read_text())
    report["server_configuration"] = conf
    if exp == "base":
        import gzip

        mass = [row["response"]["option_mass"] for name in shards["shards"] for row in map(json.loads, gzip.open(out / name, "rt", encoding="utf-8"))]
        report["option_mass"] = {"n_units": len(mass), "median": statistics.median(mass) if mass else None, "min": min(mass) if mass else None,
                                 "share_below_0.5": sum(m < 0.5 for m in mass) / len(mass) if mass else None}
    dump_json(out / "smoke.json", report)
    log(exp, f"smoke {stem} n={n}: live {use1}, wall {wall1:.1f} s; replay {use2}; identical {same}"
             + (f"; option mass {report['option_mass']}" if exp == "base" else ""))
    if not same:
        raise SystemExit("cache replay differs from the live answers")


# --------------------------------------------------------------------------------------------
# unit identity of the frozen gauntlet requests (Experiment 1)


def unit_identity(state: Any, question: dict[str, Any]) -> str:
    """The base model's scoring unit without model/template: state and question specification, key order kept."""
    return hashlib.sha256(json.dumps({"state": state, "question": question}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def units_digest(lines: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()


def seed_test_units(stem: str) -> tuple[int, set[str]]:
    src = RUNS / SETS[stem][0]
    cfg = C.run_config(src)
    with allow_test_split():
        test = load_split(cfg, "test")
    seed = load_decider((src / "seed.yaml").read_text())
    lines: set[str] = set()
    for ex in test:
        rec = C._Recorder()
        run_decider(seed, [ex], rec)  # type: ignore[arg-type]
        for state, questions in rec.bodies:
            lines.update(f"{ex.id}\t{unit_identity(state, q)}" for q in questions.values())
    return len(test), lines


def cmd_frozen_units() -> None:
    """Preparation machine only (gauntlet tasks and raw data; `PYTHONPATH=src`): the frozen gauntlet test requests'
    units per item (DREsS: the rubric's own question of the joint request), digested, and compared with the seed
    program's test units."""
    from gauntlet.tasks import REGISTRY

    tasks: dict[str, Any] = {}
    out: dict[str, Any] = {}
    for stem, (_, task_name, rubric) in SETS.items():
        task = tasks.setdefault(task_name, REGISTRY[task_name]())
        frozen: set[str] = set()
        items = task.items()
        for item in items:
            iid = f"dress-{item.id}" if rubric else item.id
            for req in task.next_requests(item, {}):
                frozen.update(f"{iid}\t{unit_identity(req.state, q)}" for qid, q in req.questions.items() if rubric is None or qid == rubric)
        n, seed = seed_test_units(stem)
        out[stem] = {"gauntlet_task": task_name, "rubric": rubric, "n_items": len(items), "n_units": len(frozen),
                     "digest": units_digest(frozen), "seed_program_identical": frozen == seed}
        print(f"{stem}: {len(items)} items, {len(frozen)} units, seed program identical {frozen == seed} (seed: {n} items)", flush=True)
    dump_json(FROZEN_UNITS, {"rule": cmd_frozen_units.__doc__.split("\n\n")[0].strip().replace("\n   ", ""), "sets": out,
                             "computed_at": now()})


# --------------------------------------------------------------------------------------------
# workload, assignment, estimate


def _tokenizer() -> Any:
    from tokenizers import Tokenizer

    path = snapshot(LP_PINS["qwen-base"]) / "tokenizer.json"
    if not path.exists():
        raise SystemExit(f"{path} missing: the base model's tokenizer is needed to count tokens")
    return Tokenizer.from_file(str(path))


def set_workload(stem: str, tok: Any) -> dict[str, Any]:
    """Planned work from the executor with uniform stand-in answers (no model). base: unique units per step and their
    prompt tokens (template, base tokenizer); native: unique Clef requests and serialized characters per step."""
    src = RUNS / SETS[stem][0]
    cfg = C.run_config(src)
    train, val = load_split(cfg, "train"), load_split(cfg, "val")
    with allow_test_split():
        test = load_split(cfg, "test")
    fit, _ = train_roles(train, cfg.optimizer)
    pool = C.demo_pool(cfg, train)

    def bodies(prog: Any, examples: list[Any], exhaustive: bool) -> list[tuple[Any, dict[str, Any]]]:
        rec = C._Recorder()
        rec.concurrency = 16  # type: ignore[misc]
        if exhaustive:
            run_examples(prog, examples, rec, exhaustive=True, pool=pool)  # type: ignore[arg-type]
        else:
            run_decider(prog, examples, rec, pool=pool)  # type: ignore[arg-type]
        return rec.bodies

    base: dict[str, Any] = {}
    seen: set[str] = set()

    def count_units(name: str, prog: Any, examples: list[Any], exhaustive: bool) -> None:
        prompts = []
        for state, questions in bodies(prog, examples, exhaustive):
            for q in questions.values():
                h = unit_identity(state, q)
                if h not in seen:
                    seen.add(h)
                    prompts.append(render_unit(state, q)[0])
        lens = [len(e.ids) for e in tok.encode_batch(prompts, add_special_tokens=False)] if prompts else []
        base[name] = {"units": len(lens), "tokens": sum(lens), "max_tokens": max(lens, default=0)}

    seed = load_decider((src / "seed.yaml").read_text())
    for split, ex in (("fit", fit), ("val", val), ("test", test)):
        count_units(f"seed.{split}", seed, ex, False)
    for name in REPS:
        count_units(f"{name}.fit", ff.unfitted(load_decider((src / f"{name}.yaml").read_text()), drop_aux=False), fit, True)
        count_units(f"{name}.val", load_decider((src / f"{name}.yaml").read_text()), val, False)
    for name in REPS:
        count_units(f"{name}.test", load_decider((src / f"{name}.yaml").read_text()), test, False)

    native: dict[str, Any] = {}
    seen_req: set[str] = set()
    prog = load_decider(native_text(src))
    for name, p, ex, exh in (("native.val", prog, val, False), ("native.test", prog, test, False),
                             ("native-lr.fit", with_readouts(prog, cfg.primary_metric), fit, True)):
        n = chars = 0
        for state, questions in bodies(p, ex, exh):
            blob = json.dumps({"state": state, "questions": questions}, ensure_ascii=False)
            h = hashlib.sha256(blob.encode("utf-8")).hexdigest()
            if h not in seen_req:
                seen_req.add(h)
                n, chars = n + 1, chars + len(blob)
        native[name] = {"requests": n, "chars": chars}
    return {"n": {"fit": len(fit), "val": len(val), "test": len(test)}, "base": base, "native": native}


def cmd_workload(stems: list[str]) -> None:
    tok = _tokenizer()
    data = load_json(WORKLOAD) or {}
    for stem in stems:
        t0 = time.monotonic()
        data[stem] = set_workload(stem, tok)
        b = data[stem]["base"]
        print(f"{stem}: base {sum(v['units'] for v in b.values())} units / {sum(v['tokens'] for v in b.values()) / 1e6:.1f} M tokens "
              f"(max {max(v['max_tokens'] for v in b.values())}); native {json.dumps(data[stem]['native'])} ({time.monotonic() - t0:.0f} s)", flush=True)
        dump_json(WORKLOAD, data)


def clef_chars_per_token() -> float:
    """Clef's characters per input token on the 8 sets whose frozen requests equal the seed program's (the Clef RQ3
    estimate's rule)."""
    work = json.loads(C.WORKLOAD.read_text())
    same = [s for s in SETS if s not in C.SEED_TEST_SETS]
    toks = sum(json.loads((ROOT / "runs" / "clef" / SETS[s][1] / "manifest.json").read_text())["input_tokens"] for s in same)
    return sum(work[s]["steps"]["seed.test"]["chars"] for s in same) / toks


def planned(exp: str, stem: str, work: dict[str, Any], cpt: float) -> dict[str, int]:
    """Planned calls and input tokens of a set (base: Jev's chosen representative stands in for the not yet chosen one)."""
    if exp == "base":
        rep = json.loads((RUNS / SETS[stem][0] / "test_consumed.json").read_text())["representative"]
        keys = ["seed.fit", "seed.val", "seed.test", *(f"{r}.{s}" for r in REPS for s in ("fit", "val")), f"{rep}.test"]
        steps_ = work[stem]["base"]
        return {"calls": sum(steps_[k]["units"] for k in keys), "tokens": sum(steps_[k]["tokens"] for k in keys)}
    steps_ = work[stem]["native"]
    return {"calls": sum(v["requests"] for v in steps_.values()), "tokens": round(sum(v["chars"] for v in steps_.values()) / cpt)}


def cmd_assign() -> None:
    """Within each experiment, sets by planned input tokens, largest first, each to the GPU with the smaller running
    total (ties: gpu0). Each GPU runs its base slot, then its native slot."""
    work = json.loads(WORKLOAD.read_text())
    cpt = clef_chars_per_token()
    out: dict[str, Any] = {"rule": cmd_assign.__doc__.split("\n\n")[0].strip().replace("\n   ", ""), "clef_chars_per_token": cpt}
    for exp in EXPS:
        est = {s: planned(exp, s, work, cpt) for s in SETS}
        load = {"gpu0": 0, "gpu1": 0}
        slots: dict[str, list[str]] = {"gpu0": [], "gpu1": []}
        for stem in sorted(SETS, key=lambda s: (-est[s]["tokens"], s)):
            g = min(load, key=lambda x: (load[x], x))
            slots[g].append(stem)
            load[g] += est[stem]["tokens"]
        out[exp] = {"slots": slots, "planned_tokens": load, "per_set": est}
    dump_json(ASSIGNMENT, out)
    print(json.dumps({e: {k: out[e][k] for k in ("slots", "planned_tokens")} for e in EXPS}, indent=1))


def clef_throughput() -> dict[str, float]:
    """Clef's measured input tokens per inference second on each set in the Clef RQ3 run on these H100s."""
    rows = {r["set"]: r for r in json.loads((ROOT / "results" / "CLEF_RQ3.json").read_text())["clef"]["sets"]}
    return {s: rows[s]["fresh_input_tokens"] / rows[s]["fresh_inference_s"] for s in SETS}


def measured(exp: str, stem: str) -> tuple[float, float] | None:
    done = [load_json(step_dir(exp, stem, s) / "result.json") for s in EXPS[exp]["steps"]]
    if not all(done) or not all(consumed(step_dir(exp, stem, s)) for s in EXPS[exp]["steps"]):
        return None
    use = [v for r in done for v in r["usage"].values()]
    return sum(v["fresh_input_tokens"] for v in use), sum(v["fresh_inference_s"] for v in use) / 3600


def estimate_table() -> tuple[str, dict[str, float]]:
    assign = json.loads(ASSIGNMENT.read_text())
    tput = clef_throughput()
    lines = ["| GPU | Experiment | Set | Calls (planned) | Input tokens | GPU h |", "|---|---|---|---:|---:|---:|"]
    total = {f"{g}/{e}": 0.0 for g in ("gpu0", "gpu1") for e in EXPS}
    for g in ("gpu0", "gpu1"):
        for exp in EXPS:
            for stem in assign[exp]["slots"][g]:
                est = assign[exp]["per_set"][stem]
                tokens, hours, tag = est["tokens"], est["tokens"] / tput[stem] / 3600, "est."
                if (m := measured(exp, stem)) is not None:
                    (tokens, hours), tag = m, "measured"
                total[f"{g}/{exp}"] += hours
                lines.append(f"| {g} | {exp} | {stem} | {est['calls']:,} | {tokens / 1e6:.1f} M | {hours:.2f} ({tag}) |")
    for g in ("gpu0", "gpu1"):
        lines.append(f"| {g} | **total** | base {total[f'{g}/base']:.1f} h + native {total[f'{g}/native']:.1f} h | | | "
                     f"**{total[f'{g}/base'] + total[f'{g}/native']:.1f}** |")
    return "\n".join(lines), total


def cmd_estimate() -> None:
    """Expected GPU hours (no inference): planned input tokens / Clef's measured throughput on the same set in the
    Clef RQ3 run (same backbone size and batching); base tokens are exact prompt tokens of the template, native tokens
    are planned Clef request characters / Clef's characters per token. Measured values replace the estimate once a set
    has run."""
    table, _ = estimate_table()
    print(table)
    print(f"\nThroughput source: results/CLEF_RQ3.json (Clef, per set). Clef characters per token: {clef_chars_per_token():.2f}. "
          "Each set also loads its model once (about 1-2 min).")


# --------------------------------------------------------------------------------------------
# dry run


def template_problems(stem: str, prog_names: list[str]) -> list[str]:
    """Every question of the programs fits the template (choice <= 26 options, score <= 9 levels, instructions)."""
    src = RUNS / SETS[stem][0]
    out = []
    for name in prog_names:
        prog = load_decider((src / name).read_text())
        for nid in prog.nodes:
            try:
                qs = questions_for(prog, nid)
            except TypeError:
                continue  # routing nodes ask nothing
            for qid, q in qs.items():
                try:
                    render_unit("", q)
                except TemplateError as exc:
                    out.append(f"{stem}/{name}/{qid}: {exc}")
    return out


def weights_problems(exp: str) -> tuple[list[str], dict[str, Any]]:
    pin = EXPS[exp]["pin"]
    snap = snapshot(pin)
    problems: list[str] = []
    if exp == "base":
        listed = [line.split(None, 1)[1].strip() for line in WEIGHTS.read_text().splitlines() if line.strip()]
        missing = [p for p in listed if not (ROOT / p).exists() or (ROOT / p).stat().st_size == 0]
        if missing:
            problems.append(f"{len(missing)} of {len(listed)} pinned base-model files missing (e.g. {missing[0]}); download the "
                            f"pinned revision and run sha256sum -c {WEIGHTS.relative_to(ROOT)}")
        return problems, {"snapshot": str(snap.relative_to(ROOT)), "files_listed": len(listed), "files_present": len(listed) - len(missing)}
    weights = sorted(p.name for p in snap.glob("*.safetensors")) if snap.exists() else []
    if not weights or not (snap / "joint_schema_model.py").exists():
        problems.append(f"pinned weights/release code missing at {snap}")
    return problems, {"snapshot": str(snap.relative_to(ROOT)), "safetensors": len(weights)}


def cmd_plan(exp: str, device: str | None, slot: str | None, stems: list[str]) -> dict[str, Any]:
    """Dry run: validates the assignment, inputs, pins, weights, cache path, template limits and the frozen-unit
    identity; loads no model and makes no request."""
    model = EXPS[exp]["model"]
    cache_env = "JEV_OPT_LOGPROB_CACHE" if exp == "base" else "JEV_OPT_CLEF_CACHE"
    cache = Path(os.environ.get(cache_env) or ROOT / "data" / "cache" / f"{model}-jev-opt")
    problems: list[str] = []
    if slot is not None:
        slots = json.loads(ASSIGNMENT.read_text())[exp]["slots"]
        every = sorted(s for v in slots.values() for s in v)
        if every != sorted(SETS):
            problems.append(f"assignment does not cover each set exactly once: {every}")
        stems = slots[slot]
    wp, weights = weights_problems(exp)
    problems += wp
    try:
        cache.mkdir(parents=True, exist_ok=True)
        probe = cache / f".plan-probe-{os.getpid()}"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        problems.append(f"cache {cache} not writable: {exc}")
    work = load_json(WORKLOAD) or {}
    frozen = (load_json(FROZEN_UNITS) or {}).get("sets", {})
    cpt = clef_chars_per_token()
    rows = []
    for stem in stems:
        src = RUNS / SETS[stem][0]
        cfg = C.run_config(src)  # data paths mapped into this repository; refuses a changed test file
        with allow_test_split():
            test = load_split(cfg, "test")
        val = load_split(cfg, "val")
        fit, _ = train_roles(load_split(cfg, "train"), cfg.optimizer)
        row: dict[str, Any] = {"set": stem, "source_run": src.name, "fit": len(fit), "val": len(val), "test": len(test)}
        if exp == "base":
            names = ["seed.yaml", *(f"{r}.yaml" for r in REPS)]
            problems += template_problems(stem, names)
            try:
                n_emb: set[Path] = set()
                C.embedding_paths(stem, n_emb)
                row["demo_embeddings_cached"] = len(n_emb)
            except C.EmbeddingMiss as exc:
                problems.append(f"{stem}: {exc}")
            n, seed_units = seed_test_units(stem)
            want = frozen.get(stem)
            same = want is not None and want["digest"] == units_digest(seed_units) and want["seed_program_identical"]
            if not same:
                problems.append(f"{stem}: seed-program test units differ from the frozen gauntlet units ({FROZEN_UNITS.name})")
            row["frozen_units_identical"] = same
        else:
            native = load_decider(native_text(src))
            if native.demos is not None:
                problems.append(f"{stem}: the native program has demonstrations (not expected)")
            info = json.loads((src / "run.json").read_text())["conditions"]["jev-population-opt"]["summary"]["seeds"]
            chosen = next(k for k, v in info.items() if v["kind"] == "jev-native" and v["chosen_for_kind"])
            row["native_seed"] = chosen
            nrun = RUNS / SETS[stem][0].replace("-P", "-N")
            if stem in CFG8:
                n_seed = load_decider((nrun / "seed.yaml").read_text())
                if n_seed.digest() != native.digest():
                    problems.append(f"{stem}: {NATIVE} differs from {nrun.name}/seed.yaml (the paper's jev-native-zero program)")
                row["equals_native_zero_row"] = n_seed.digest() == native.digest()
        state = {s: ("test consumed" if consumed(step_dir(exp, stem, s)) else "pending") for s in EXPS[exp]["steps"]}
        row.update(outputs=[step_dir(exp, stem, s).name for s in EXPS[exp]["steps"]], state=state,
                   planned=planned(exp, stem, work, cpt) if stem in work else None)
        rows.append(row)
    cmd = None
    if device:
        cmd = lp_server_command(model, device, 32000) if exp == "base" else clef_server_command(model, device, 32000, cache / "scratch")
    plan = {"experiment": exp, "model": EXPS[exp]["pin"], "device": device, "slot": slot, "weights": weights,
            "template": {"id": TEMPLATE_ID, "sha256": TEMPLATE_SHA256} if exp == "base" else None,
            "cache": str(cache.relative_to(ROOT)) if cache.is_relative_to(ROOT) else str(cache),
            "server_command": shlex.join(cmd) if cmd else None, "sets": rows, "problems": problems, "checked_at": now()}
    print(json.dumps({k: v for k, v in plan.items() if k != "sets"}, indent=1))
    for r in rows:
        extra = (f"frozen units identical {r['frozen_units_identical']}" if exp == "base"
                 else f"native seed {r['native_seed']}" + (f", = jev-native-zero row {r['equals_native_zero_row']}" if "equals_native_zero_row" in r else ""))
        print(f"{r['set']:24s} fit {r['fit']} val {r['val']} test {r['test']}; {extra}; planned {r['planned']}; {r['state']}")
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["run", "smoke", "replay", "plan", "workload", "assign", "estimate", "frozen-units"])
    ap.add_argument("--exp", choices=list(EXPS))
    ap.add_argument("--device")
    ap.add_argument("--slot", choices=["gpu0", "gpu1"], help="plan: the sets of this slot of h100_round2_assignment.json")
    ap.add_argument("--set", action="append", dest="sets", choices=list(SETS))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", type=Path, help="plan: also write the plan as JSON here")
    a = ap.parse_args()
    stems = a.sets or list(SETS)
    if a.command == "workload":
        cmd_workload(stems)
        return 0
    if a.command == "assign":
        cmd_assign()
        return 0
    if a.command == "estimate":
        cmd_estimate()
        return 0
    if a.command == "frozen-units":
        cmd_frozen_units()
        return 0
    if a.exp is None:
        ap.error("--exp is required")
    if a.command == "plan":
        plan = cmd_plan(a.exp, a.device, a.slot, stems)
        if a.out:
            dump_json(a.out, plan)
        return 1 if plan["problems"] else 0
    if a.command == "replay":
        for s in stems:
            cmd_replay(a.exp, s)
        return 0
    if not a.device or len(stems) != 1:
        ap.error(f"{a.command} needs --device and exactly one --set")
    if a.command == "run":
        cmd_run(a.exp, a.device, stems[0])
    else:
        cmd_smoke(a.exp, a.device, stems[0], a.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
