"""Optimization step (RQ3: optimization with training labels) on GPT-6 Luna answers (pre-registered; the Clef RQ3 design with
Luna in Clef's place); with `--model msd1` the same on Microsoft-Decision-1 answers (cl-msd1-* step directories,
runs/msd1/<task> frozen runs, cache .jev-cache/msd1, ledger runs/msd1/spend.jsonl, log msd1_rq3.log).

Per RQ3 set (the 13 sets of clef_rq3.py), clef_rq3's steps, unchanged and imported:

  cal  runs/jev-opt/cl-luna-<stem>-cal/  the paper's question answered by Luna on the fit slice and val; bias +
       temperature fitted on the fit slice, applied to val and test. Test distributions: Luna's frozen zero-shot
       responses (runs/luna/<task>) where every seed-program test request equals the frozen one (gauntlet request
       hash recomputed with Luna's model id), else the seed program answered on test (seed zero-shot).
  opt  runs/jev-opt/cl-luna-<stem>-opt/  P and P+head, readouts reset and refitted on Luna's fit-slice answers,
       both on val; P+head replaces P iff >= 1.0 point higher; selection.json before the one test evaluation.

Backend jev_opt.luna (JEV_OPT_BACKEND=luna): exact-request cache runs/jev-opt/.jev-cache/luna/, spend ledger
runs/luna/spend.jsonl (run label rq3, cap $35; total cap $250), verbatim responses per run directory
(luna_responses*.jsonl.gz). Refused answers -> uniform answer, counted per step in result.json `usage` (with
fresh requests, input tokens and cost); rejected requests (HTTP 400/413/422) -> failed examples, scored wrong;
any other failure (budget cap, off-pin snapshot, API error, cache miss in a replay) aborts the set, which resumes
from the cache.

Subcommands (repository root, jev-opt environment):
  run     --set <stem>          both steps of one set (needs Luna's frozen zero-shot run runs/luna/<task>)
  smoke   --set <stem> --n 20   seed, top-single, top-stacked (readouts unfitted, exhaustive) on the first n fit
                                items, then a cache-only replay compared answer by answer
  replay  [--set <stem>]        cache-only re-execution of finished sets (no key, no request); every recorded
                                val/test prediction must come back
  plan    [--set <stem>]        dry run: inputs, identity rule (where the frozen run exists), embeddings, cache,
                                key, spend so far, planned requests and projected cost; makes no request
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

assert sys.version_info[:2] == (3, 11), f"run under Python 3.11 (worker's version), not {sys.version}"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clef_rq3 as C  # noqa: E402  (Env, steps and helpers of the Clef RQ3 driver)
import framing_features as ff  # noqa: E402  (its import disables Jev posting; LunaClient posts through its own _post)

from jev_opt.adapter import train_roles  # noqa: E402
from jev_opt.execute import ExampleResult, run_decider, run_examples  # noqa: E402
from jev_opt.hygiene import allow_test_split, consumed, load_split  # noqa: E402
from jev_opt.ir import load_decider  # noqa: E402
from jev_opt.jev import ResponseLedger, make_client  # noqa: E402
from jev_opt.luna import LUNA, REFUSED_MODEL, REJECTED_MODEL, RUN, SPECS, LunaClient, luna_api_key, make_luna_client  # noqa: E402

ROOT = C.ROOT
SETS = C.SETS
REPS = C.REPS
OUT = C.RUNS  # step directories, test ledger and log
CHARS_PER_TOKEN = 2.68  # Jev's characters per input token on the 8 identical-request sets (pre-registered)
REJECTED = "LunaRejected"

now, dump_json, load_json, delta = C.now, C.dump_json, C.load_json, C.delta


def configure(name: str) -> None:
    """The decision model of this run (`--model`, default luna): its pin, price, caps and the names derived from it.
    NAME is the model name of the cl-<model>-<stem>-{cal,opt} directories and of runs/<model>/<task>."""
    global SPEC, MODEL, SNAPSHOT, PRICE, NAME, LOG, FROZEN, LEDGER_NAME, CANDIDATE
    SPEC = SPECS[name]
    MODEL, SNAPSHOT, PRICE, NAME = SPEC.model, SPEC.snapshot, SPEC.price, SPEC.name
    LOG = OUT / f"{NAME}_rq3.log"
    FROZEN = ROOT / "runs" / NAME  # the model's frozen zero-shot runs (part 1 of its pre-registered study)
    LEDGER_NAME = f"{NAME}_responses"
    CANDIDATE = f"{NAME}-rq3"
    # clef_rq3's steps read these module globals (the steps themselves are unchanged)
    C.PINS = {**C.PINS, NAME: MODEL}  # step_cal records PINS[model]
    Env.ledger_name, Env.candidate = LEDGER_NAME, CANDIDATE


def request_hash(body: dict[str, Any]) -> str:
    """gauntlet.jev.request_hash: the hash under which the frozen zero-shot run recorded each request."""
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def seed_identity(model: str, stem: str, seed_prog: Any, test: list[Any]) -> dict[str, Any]:
    """Is every seed-program test request identical to Luna's frozen zero-shot request? (gauntlet request hash of
    {model, state, questions} with Luna's model id; no backend configuration, no inference)."""
    assert model == NAME, model
    rows = C.frozen_rows(model, stem)
    same, first_diff = 0, None
    for ex in test:
        rec = C._Recorder()
        run_decider(seed_prog, [ex], rec)  # type: ignore[arg-type]
        if len(rec.bodies) != 1:
            raise SystemExit(f"{stem}: seed program issues {len(rec.bodies)} requests per example")
        state, questions = rec.bodies[0]
        if request_hash({"model": MODEL, "state": state, "questions": questions}) in rows[ex.id]["hashes"]:
            same += 1
        elif first_diff is None:
            first_diff = ex.id
    return {"n_test": len(test), "identical": same, "all_identical": same == len(test), "first_difference": first_diff}


def counters(client: Any) -> dict[str, float]:
    """clef_rq3.counters with Luna's answer counts and billed cost (no local inference time)."""
    u = client.usage
    return {"wall_s": time.monotonic(), "calls": u.calls, "fresh_calls": u.fresh_calls, "input_tokens": u.input_tokens,
            "fresh_input_tokens": u.fresh_input_tokens, "fresh_cost": u.fresh_cost, "answers": client.answers_seen,
            "refused_answers": client.refused_answers, "saturated_answers": client.saturated_answers,
            "rejected_requests": client.rejected_requests, "refused_requests": client.refused_requests}


C.seed_identity = seed_identity
C.counters = counters


class Env(C.Env):
    """clef_rq3.Env with the model's ledger name and test-ledger label (set by `configure`); rejected requests count
    as failed examples, every other failure (budget cap, off-pin snapshot, API error, cache miss) aborts the set."""

    ledger_name = "luna_responses"
    candidate = "luna-rq3"

    def run(self, program: Any, examples: list[Any], split: str, exhaustive: bool = False) -> list[ExampleResult]:
        ctx = {"split": split, "condition": CANDIDATE}
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


configure(LUNA.name)


def cal_dir(stem: str) -> Path:
    return OUT / f"cl-{NAME}-{stem}-cal"


def opt_dir(stem: str) -> Path:
    return OUT / f"cl-{NAME}-{stem}-opt"


def log(msg: str) -> None:
    line = f"{now()} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def client_for(live: bool) -> LunaClient:
    """Live: the executor's client factory (JEV_OPT_BACKEND=luna). Replay: the same cache without a key."""
    os.environ["JEV_OPT_BACKEND"] = NAME
    cache = C.RUNS / ".jev-cache"
    client = make_client(cache, "", 0, 0, [ROOT]) if live else make_luna_client(cache, [ROOT], offline=True, spec=SPEC)
    assert isinstance(client, LunaClient) and client.model == MODEL, client
    return client


def spend_path() -> Path:
    return Path(os.environ.get(f"JEV_OPT_{NAME.upper()}_SPEND") or SPEC.spend_ledger)


def spend_totals() -> dict[str, float]:
    """Spend so far in the shared ledger (all Luna runs, and this run label)."""
    total = run_total = 0.0
    path = spend_path()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                total += float(row["cost"])
                run_total += float(row["cost"]) if row.get("run") == RUN else 0.0
    return {"total_usd": round(total, 6), "rq3_usd": round(run_total, 6), "total_cap_usd": SPEC.total_cap, "rq3_cap_usd": SPEC.run_caps[RUN]}


def frozen_problems(stem: str) -> list[str]:
    """The set's frozen zero-shot Luna run: finished (scores.json) and on the pinned snapshot only."""
    d = FROZEN / SETS[stem][1]
    if not (d / "scores.json").exists():
        return [f"{stem}: frozen zero-shot run {d.relative_to(ROOT) if d.is_relative_to(ROOT) else d} not finished (no scores.json)"]
    snaps = set(json.loads((d / "manifest.json").read_text()).get("model_snapshots") or {})
    off = snaps - {SNAPSHOT, REJECTED_MODEL, REFUSED_MODEL}
    return [f"{stem}: frozen run has off-pin snapshots {sorted(off)}"] if off else []


def usage_line(client: LunaClient) -> str:
    u = client.usage
    return (f"fresh requests {u.fresh_calls}, fresh input tokens {u.fresh_input_tokens:,}, fresh cost ${u.fresh_cost:.4f}; "
            f"answers {client.answers_seen:,}, refused {client.refused_answers:,}, saturated {client.saturated_answers:,}, "
            f"refused requests {client.refused_requests}, rejected requests {client.rejected_requests}")


# --------------------------------------------------------------------------------------------
# commands


def held(stem: str) -> bool:
    """The set's gauntlet task is on the model's hold list runs/<model>/HOLD (no request may be sent for it)."""
    path = ROOT / "runs" / NAME / "HOLD"
    names = {line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")} if path.exists() else set()
    return SETS[stem][1] in names


def cmd_run(stem: str) -> None:
    if held(stem):
        log(f"{stem} HELD (runs/{NAME}/HOLD): not run, no request")
        raise SystemExit(4)
    problems = frozen_problems(stem)
    if problems:
        raise SystemExit("; ".join(problems))
    src = C.RUNS / SETS[stem][0]
    client = client_for(live=True)
    t0 = time.monotonic()
    try:
        env = Env(src, client, test=True)
        log(f"{stem} cal start")
        c = C.step_cal(NAME, stem, env, src, cal_dir(stem))
        log(f"{stem} cal done: val {c['val']['seed_primary']:.4f} -> {c['val']['calibrated_primary']:.4f}, "
            f"test calibrated {c['test']['calibrated_primary']:.4f} ({c['test_distributions']}); rejected examples "
            f"{len(c['rejected'])}; usage {json.dumps(c['usage'])}")
        env.rejected = []
        log(f"{stem} opt start")
        o = C.step_opt(NAME, stem, env, src, opt_dir(stem))
        log(f"{stem} opt done: val {o['selection']['val']} chosen {o['selection']['chosen']}, test {o['test']['primary']:.4f}; "
            f"rejected examples {len(o['rejected'])}; usage {json.dumps(o['usage'])}")
    finally:
        client.close()
    log(f"{stem} finished in {time.monotonic() - t0:.0f} s; {usage_line(client)}; spend ledger {json.dumps(spend_totals())}")


def cmd_replay(stem: str) -> None:
    """Cache-only re-execution: every recorded val/test prediction must come back, with no request."""
    src = C.RUNS / SETS[stem][0]
    client = client_for(live=False)
    try:
        env = Env(src, client, test=True)
        for step, fn, out in (("cal", C.step_cal, cal_dir(stem)), ("opt", C.step_opt, opt_dir(stem))):
            before = (out / "result.json").read_bytes()
            fn(NAME, stem, env, src, out)  # check_replay compares the test predictions; the whole record compared here
            if (out / "result.json").read_bytes() != before:
                raise SystemExit(f"{out.name}: replayed result.json differs from the recorded one")
            log(f"{stem} {step} replay identical (fresh requests {client.usage.fresh_calls})")
            env.rejected = []
    finally:
        client.close()
    assert client.usage.fresh_calls == 0


def cmd_smoke(stem: str, n: int) -> None:
    src = C.RUNS / SETS[stem][0]
    out = OUT / f"cl-{NAME}-smoke-{stem}"
    progs = {"seed": load_decider((src / "seed.yaml").read_text())}
    progs.update({name: ff.unfitted(load_decider((src / f"{name}.yaml").read_text()), drop_aux=False) for name in REPS})

    def answers(client: Any) -> tuple[dict[str, Any], dict[str, float]]:
        env = Env(src, client, test=False)
        got: dict[str, Any] = {}
        c0 = counters(client)
        for name, prog in progs.items():
            res = env.run(prog, env.fit[:n], "train", exhaustive=name != "seed")
            got[name] = [{"id": r.id, "memo": r.memo, "dist": r.dist, "pred": r.pred, "error": r.error} for r in res]
        return got, {**delta(c0, counters(client)), "rejected_examples": len(env.rejected)}

    live = client_for(live=True)
    out.mkdir(parents=True, exist_ok=True)
    led = ResponseLedger(out, name=LEDGER_NAME)
    live.ledger = led
    try:
        first, use1 = answers(live)
    finally:
        live.ledger = None
        shards = led.finalize()
        live.close()
    replay = client_for(live=False)
    try:
        second, use2 = answers(replay)
    finally:
        replay.close()
    same = json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    report = {"set": stem, "model": MODEL, "snapshot": SNAPSHOT, "snapshots_seen": sorted(live.snapshots), "n_fit_items": n,
              "programs": list(progs), "live": use1, "replay": use2, "replay_identical": same, "ledger": shards,
              "spend_ledger": spend_totals(), "computed_at": now()}
    dump_json(out / "smoke.json", report)
    log(f"smoke {stem} n={n}: live {use1}; replay {use2}; identical {same}")
    if not same or use2["fresh_calls"]:
        raise SystemExit("cache replay differs from the live answers")


def cmd_plan(stems: list[str]) -> dict[str, Any]:
    """Dry run: inputs, identity (where the frozen run exists), embeddings, cache, key, spend; no request."""
    problems: list[str] = []
    cache = Path(os.environ.get(f"JEV_OPT_{NAME.upper()}_CACHE") or C.RUNS / ".jev-cache" / NAME)
    try:
        cache.mkdir(parents=True, exist_ok=True)
        probe = cache / f".plan-probe-{os.getpid()}"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        problems.append(f"cache {cache} not writable: {exc}")
    spend = spend_totals()
    if spend["rq3_usd"] >= spend["rq3_cap_usd"] or spend["total_usd"] >= spend["total_cap_usd"]:
        problems.append(f"spend ledger at a cap: {spend}")
    work = load_json(C.WORKLOAD) or {}
    rows = []
    for stem in stems:
        src = C.RUNS / SETS[stem][0]
        cfg = C.run_config(src)
        with allow_test_split():
            test = load_split(cfg, "test")
        val = load_split(cfg, "val")
        fit, _ = train_roles(load_split(cfg, "train"), cfg.optimizer)
        for name in ("seed", *REPS):
            load_decider((src / f"{name}.yaml").read_text())
        jev_rep = json.loads((src / "test_consumed.json").read_text())["representative"]
        n_emb: set[Path] = set()
        try:
            C.embedding_paths(stem, n_emb)
        except C.EmbeddingMiss as exc:
            problems.append(f"{stem}: {exc}")
        pending = frozen_problems(stem)
        if pending:
            ident = None
            seed_test = stem in C.SEED_TEST_SETS
            identity = f"pending ({pending[0].split(': ', 1)[1]}); expected {'different' if seed_test else 'identical'}"
        else:
            ident = seed_identity(NAME, stem, load_decider((src / "seed.yaml").read_text()), test)
            seed_test = not ident["all_identical"]
            identity = f"{ident['identical']}/{ident['n_test']}"
            if (stem in C.SEED_TEST_SETS) == ident["all_identical"]:
                problems.append(f"{stem}: identity {identity} contradicts SEED_TEST_SETS")
        est = C.planned(work[stem], seed_test, jev_rep) if stem in work else None
        if est is None:
            problems.append(f"{stem}: not in {C.WORKLOAD.name}")
        else:
            est["tokens"] = round(est["chars"] / CHARS_PER_TOKEN)
            est["cost_usd"] = round(est["tokens"] * PRICE, 4)
        state = {s: ("test consumed" if consumed(d(stem)) else "pending") for s, d in (("cal", cal_dir), ("opt", opt_dir))}
        rows.append({"set": stem, "source_run": src.name, "frozen_run": f"runs/{NAME}/{SETS[stem][1]}", "fit": len(fit), "val": len(val),
                     "test": len(test), "seed_test_requests_identical": identity,
                     "calibration_test_distributions": "frozen zero-shot" if not seed_test else "seed program on test",
                     "demo_embeddings_cached": len(n_emb), "outputs": [cal_dir(stem).name, opt_dir(stem).name], "state": state,
                     "planned": est})
    planned = [r["planned"] for r in rows if r["planned"]]
    total = {k: sum(p[k] for p in planned) for k in ("requests", "chars", "tokens")}
    total["cost_usd"] = round(total["tokens"] * PRICE, 2)
    key = luna_api_key([ROOT])
    plan = {"model": MODEL, "snapshot": SNAPSHOT, "endpoint": os.environ.get("JEV_OPT_DECISIONS_URL") or "https://openrouter.ai/api/alpha/decisions",
            "key": "found" if key else "missing (cache-only)", "cache": str(cache.relative_to(ROOT)) if cache.is_relative_to(ROOT) else str(cache),
            "spend_ledger": {"path": str(spend_path()), **spend}, "chars_per_token": CHARS_PER_TOKEN, "price_per_token": PRICE,
            "planned_total": total, "sets": rows, "problems": problems, "checked_at": now()}
    print(json.dumps({k: v for k, v in plan.items() if k != "sets"}, indent=1))
    for r in rows:
        p = r["planned"] or {}
        print(f"{r['set']:24s} fit {r['fit']} val {r['val']} test {r['test']}; seed requests identical {r['seed_test_requests_identical']} "
              f"-> {r['calibration_test_distributions']}; planned {p.get('requests', 0):,} requests, {p.get('tokens', 0) / 1e6:.1f} M tokens, "
              f"${p.get('cost_usd', 0):.2f}; {r['state']}")
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["run", "smoke", "replay", "plan"])
    ap.add_argument("--set", action="append", dest="sets", choices=list(SETS))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--model", choices=list(SPECS), default=LUNA.name, help="decision model (default luna)")
    ap.add_argument("--out", type=Path, help="plan: also write the plan as JSON here")
    a = ap.parse_args()
    configure(a.model)
    stems = a.sets or list(SETS)
    if a.command == "plan":
        plan = cmd_plan(stems)
        if a.out:
            dump_json(a.out, plan)
        return 1 if plan["problems"] else 0
    if a.command == "replay":
        for s in stems:
            cmd_replay(s)
        return 0
    if len(stems) != 1:
        ap.error(f"{a.command} needs exactly one --set")
    if a.command == "run":
        cmd_run(stems[0])
    else:
        cmd_smoke(stems[0], a.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
