"""Optimization step (RQ3: optimization with training labels) on Clef / Clef-flash answers (pre-registered).

Per RQ3 set (8 configurations of results/seeds.json, seed 0; 5 sets of results/frozen.json) and model:

  cal  runs/jev-opt/cl-<model>-<stem>-cal/  the paper's question (the run's seed.yaml) answered by the model on the
       400-example fit slice and on val; per-class bias + temperature (`fit_label_bias`, distributions floored at
       1e-4) fitted on the fit slice and applied to val and test. Test distributions: the model's frozen zero-shot
       responses when every test request of the seed program is byte-identical to the frozen request (no new
       call), else the seed program answered on the test split (also the "seed zero-shot" reference).
  opt  runs/jev-opt/cl-<model>-<stem>-opt/  the frozen top-single and top-stacked programs of the Jev run (questions,
       auxiliary questions, stacked variants, demonstrations unchanged), readouts reset and refitted on the
       model's fit-slice answers with the original fit code (framing_features.refit = fit_program with the run's
       CV), both answered on val; P+head replaces P iff its val primary is >= 1.0 point higher
       (analyze_seeds.choose); only the chosen program is answered on test, once.

Subcommands (run from the repository root with the jev-opt environment):
  run       --model clef|clef-flash --device cuda:N --set <stem>   both steps of one set (resumable from the cache)
  smoke     --model ... --device ... --set <stem> --n 20           seed/top-single/top-stacked on the first n fit
                                                                   items, then a cache-only replay compared answer by answer
  replay    --model ... --set <stem>                               cache-only re-execution of a finished set; checks
                                                                   every recorded prediction (no device, no model load)
  plan      --model ... [--device ...]                             dry run: inputs, identity rule, outputs, server
                                                                   command; loads no model and makes no call
  replicate-jev [--set <stem>]                                     the same driver on Jev's cache (posting disabled):
                                                                   must reproduce the recorded Jev val/test predictions
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shlex
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

assert sys.version_info[:2] == (3, 11), f"run under Python 3.11 (worker's version), not {sys.version}"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import framing_features as ff  # noqa: E402  (unfitted, refit; its import disables Jev posting)
from analyze_seeds import MARGIN, choose  # noqa: E402

from jev_opt import metrics  # noqa: E402
from jev_opt.adapter import train_roles  # noqa: E402
from jev_opt.clef import PINS, ClefBackend, server_command  # noqa: E402
from jev_opt.execute import ExampleResult, combine, run_decider, run_examples  # noqa: E402
from jev_opt.fit import fit_label_bias  # noqa: E402
from jev_opt.hygiene import allow_test_split, consumed, load_split, record_consumption, sha256_file  # noqa: E402
from jev_opt.ir import load_decider  # noqa: E402
from jev_opt.jev import ResponseLedger, make_client  # noqa: E402
from jev_opt.demos import SentenceEmbedder  # noqa: E402
from jev_opt.runs import _load_run_config  # noqa: E402

import numpy as np  # noqa: E402

ROOT = HERE.parents[1]
RUNS = ROOT / "runs" / "jev-opt"
LEDGER_NAME = "clef_responses"
FLOOR = 1e-4  # jev_opt's BIAS_FLOOR (confidence_readout.FLOOR)
REPS = ("top-single", "top-stacked")
# stem -> (Jev run with the frozen programs, gauntlet task of the frozen zero-shot audit, DREsS rubric)
SETS: dict[str, tuple[str, str, str | None]] = {
    "contempo-tbdense": ("v3-contempo-tbdense-P", "contempo/tbdense", None),
    "ezstance-subtaskA": ("v3-ezstance-subtaskA-P", "ezstance/subtaskA", None),
    "smarter-implicit-hate": ("v3-smarter-implicit-hate-P", "smarter/implicit_hate", None),
    "edit-intent": ("v3-edit-intent-P", "edit_intent", None),
    "inli": ("v3-inli-P", "inli", None),
    "dress-new-content": ("v3-dress-new-content-P", "dress_new", "content"),
    "dress-new-organization": ("v3-dress-new-organization-P", "dress_new", "organization"),
    "dress-new-language": ("v3-dress-new-language-P", "dress_new", "language"),
    "contempo-matres": ("fz-contempo-matres-P", "contempo/matres", None),
    "lifetox": ("fz-lifetox-P", "lifetox", None),
    "sciprompt-s2orc": ("fz-sciprompt-s2orc-P", "sciprompt/s2orc", None),
    "smarter-hatexplain": ("fz-smarter-hatexplain-P", "smarter/hatexplain", None),
    "smarter-latent-hate": ("fz-smarter-latent-hate-P", "smarter/latent_hate", None),
}


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log(model: str, msg: str) -> None:
    line = f"{now()} {msg}"
    print(line, flush=True)
    with open(RUNS / f"clef_rq3-{model}.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def floored(d: dict[str, float] | None, labels: list[str]) -> dict[str, float]:
    v = {lab: max(float((d or {}).get(lab, 0.0)), FLOOR) for lab in labels}
    s = sum(v.values())
    return {lab: x / s for lab, x in v.items()}


def normalise(p: dict[str, float], labels: list[str]) -> dict[str, float]:
    """confidence_readout.normalise: the frozen answer's probabilities over the task labels."""
    v = {lab: max(float(p.get(lab, 0.0)), 0.0) for lab in labels}
    s = sum(v.values())
    return {lab: x / s for lab, x in v.items()} if s > 0 else {lab: 1.0 / len(labels) for lab in labels}


def cal_dir(model: str, stem: str) -> Path:
    return RUNS / f"cl-{model}-{stem}-cal"


def opt_dir(model: str, stem: str) -> Path:
    return RUNS / f"cl-{model}-{stem}-opt"


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


# --------------------------------------------------------------------------------------------
# inputs: data paths and demonstration embeddings independent of the machine


EMBED_ARCHIVE = HERE / "clef_rq3_embed_cache.tar.gz"


class EmbeddingMiss(RuntimeError):
    """A demonstration embedding is not in the cache (recomputing it could change the kNN demonstrations)."""


class CachedEmbedder(SentenceEmbedder):
    """The kNN demonstrations' embeddings from `runs/jev-opt/.embed-cache` only: the vectors the Jev runs used, which
    the jev-opt runs with kNN demonstrations write there. `clef_rq3.py embed-archive` packs them into
    `clef_rq3_embed_cache.tar.gz` to carry them to another machine. A miss raises instead of re-embedding."""

    def __init__(self, cache_dir: Path, model: str, revision: str, record: set[Path] | None = None):
        super().__init__(cache_dir, model, revision)
        self.record = record

    def encode(self, texts: list[str]) -> np.ndarray:
        paths = [self._path(t) for t in texts]
        if self.record is not None:
            self.record.update(paths)
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise EmbeddingMiss(f"{len(missing)} of {len(paths)} demonstration embeddings missing under {self.cache_dir} "
                                f"(the jev-opt runs with kNN demonstrations write them there; on another machine, extract the "
                                f"archive written by `clef_rq3.py embed-archive`, {EMBED_ARCHIVE.relative_to(ROOT)}, "
                                f"into runs/jev-opt/.embed-cache)")
        return np.stack([np.load(p) for p in paths]) if paths else np.zeros((0, 0), dtype=np.float32)


def run_config(src: Path) -> Any:
    """The Jev run's config with its data paths mapped into this repository (the runs recorded absolute paths of
    the machine that produced them); the test file must match the run's recorded test hash."""
    cfg = _load_run_config(src)
    for split in ("train", "val", "test"):
        rel = str(getattr(cfg.data, split)).split("/jev-opt/experiments/", 1)[1]
        setattr(cfg.data, split, str(ROOT / "jev-opt" / "experiments" / rel))
    want = json.loads((src / "test_consumed.json").read_text())["test_sha256"]
    if sha256_file(cfg.split_path("test")) != want:
        raise SystemExit(f"{src.name}: {cfg.split_path('test')} differs from the test file the Jev run consumed")
    return cfg


def demo_pool(cfg: Any, train: list[Any], record: set[Path] | None = None) -> Any:
    from jev_opt.demos import DemoPool

    emb = cfg.embedding
    return DemoPool(train, CachedEmbedder(RUNS / ".embed-cache", emb.model, emb.revision, record))


# --------------------------------------------------------------------------------------------
# frozen zero-shot audit of the model


def frozen_rows(model: str, stem: str) -> dict[str, dict[str, Any]]:
    """jev-opt test id -> {request_hashes, pred, gold, dist (over the set's labels)} of the frozen zero-shot run."""
    _, task, rubric = SETS[stem]
    d = ROOT / "runs" / model / task
    answers: dict[str, dict[str, Any]] = {}
    hashes: dict[str, set[str]] = {}
    for shard in sorted(d.glob("responses*.jsonl.gz")):
        for line in gzip.open(shard, "rt", encoding="utf-8"):
            r = json.loads(line)
            hashes.setdefault(r["item_id"], set()).add(r["request_hash"])
            ans = r["response"]["answers"]
            assert len(ans) == (3 if rubric else 1), (stem, r["item_id"], list(ans))
            answers[r["item_id"]] = ans[rubric] if rubric else next(iter(ans.values()))
    out = {}
    for line in (d / "predictions.jsonl").read_text().splitlines():
        r = json.loads(line)
        iid = r["item_id"]
        if rubric:
            key = f"dress-{iid}"
            pred, gold = f"{float(r['prediction'][rubric]):.1f}", f"{float(r['gold'][rubric]):.1f}"
            probs = {f"{1.0 + 0.5 * int(i):.1f}": float(v) for i, v in answers[iid]["probabilities"].items()}
        else:
            key, pred, gold, probs = iid, r["prediction"]["label"], r["gold"], answers[iid]["probabilities"]
        out[key] = {"hashes": hashes[iid], "pred": pred, "gold": gold, "probs": probs}
    return out


def frozen_configurations(model: str, stem: str) -> list[dict[str, Any]]:
    meta = json.loads((ROOT / "runs" / model / SETS[stem][1] / "manifest.json").read_text())["backend_metadata"]
    return [meta, *(meta.get("devices") or {}).values()]


class _Recorder:
    """Client stand-in that records the seed program's request (one per example) and answers uniformly."""

    concurrency = 1

    def __init__(self) -> None:
        self.bodies: list[tuple[Any, dict[str, Any]]] = []

    def decide(self, state: Any, questions: dict[str, Any], context: dict[str, Any] | None = None) -> Any:
        from jev_opt.jev import Decision

        self.bodies.append((state, questions))
        ans: dict[str, Any] = {}
        for q, spec in questions.items():
            if spec["type"] == "noul":
                ans[q] = {"noul": 0.5}
            else:
                keys = list(spec["criteria"]) if isinstance(spec["criteria"], dict) else [str(i) for i in range(len(spec["criteria"]))]
                ans[q] = {"probabilities": {k: 1.0 / len(keys) for k in keys}}
        return Decision(ans, "recorder", 0, 0.0, 0.0, True)


def seed_identity(model: str, stem: str, seed_prog: Any, test: list[Any]) -> dict[str, Any]:
    """Is every seed-program test request byte-identical to the model's frozen zero-shot request? (request hash
    recomputed under the frozen run's recorded configuration; no inference)."""
    rows = frozen_rows(model, stem)
    confs = frozen_configurations(model, stem)
    pin = f"{confs[0]['model_id']}@{confs[0]['revision']}"
    assert pin == PINS[model], (pin, PINS[model])
    same, first_diff = 0, None
    for ex in test:
        rec = _Recorder()
        run_decider(seed_prog, [ex], rec)  # type: ignore[arg-type]
        if len(rec.bodies) != 1:
            raise SystemExit(f"{stem}: seed program issues {len(rec.bodies)} requests per example")
        state, questions = rec.bodies[0]
        body = {"model": pin, "state": state, "questions": questions}
        digests = {hashlib.sha256(json.dumps({"configuration": c, "request": body}, ensure_ascii=False,
                                             separators=(",", ":")).encode("utf-8")).hexdigest() for c in confs}
        if digests & rows[ex.id]["hashes"]:
            same += 1
        elif first_diff is None:
            first_diff = ex.id
    return {"n_test": len(test), "identical": same, "all_identical": same == len(test), "first_difference": first_diff}


# --------------------------------------------------------------------------------------------
# execution environment


class Env:
    """Config, splits, client and demo pool of one Jev run; `run` aborts on any failure that is not an
    encoder rejection (cache miss, server error), so a set never finishes with silently failed examples."""

    def __init__(self, run_dir: Path, client: Any, test: bool):
        self.dir = run_dir
        self.config = run_config(run_dir)
        self.train = load_split(self.config, "train")
        self.val = load_split(self.config, "val")
        self.test: list[Any] = []
        if test:
            with allow_test_split():
                self.test = load_split(self.config, "test")
        self.client = client
        self.pool = demo_pool(self.config, self.train)
        self.fit, _ = train_roles(self.train, self.config.optimizer)
        self.labels = list(self.config.labels)
        self.rejected: list[dict[str, str]] = []

    def run(self, program: Any, examples: list[Any], split: str, exhaustive: bool = False) -> list[ExampleResult]:
        ctx = {"split": split, "condition": "clef-rq3"}
        if exhaustive:
            res = run_examples(program, examples, self.client, exhaustive=True, context=ctx, pool=self.pool)
        else:
            res = run_decider(program, examples, self.client, context=ctx, pool=self.pool)
        bad = [r for r in res if r.error is not None]
        infra = [r for r in bad if not r.error.startswith("ClefRejected")]
        if infra:
            raise SystemExit(f"{self.dir.name} {split}: {len(infra)}/{len(res)} examples failed, e.g. {infra[0].error[:400]}")
        self.rejected += [{"split": split, "id": r.id, "error": r.error[:300]} for r in bad]
        return res

    def primary(self, gold: list[str], pred: list[str | None]) -> float:
        c = self.config
        return metrics.primary(c.primary_metric, gold, pred, c.labels, c.negative_label)


def counters(client: Any) -> dict[str, float]:
    u = client.usage
    return {"wall_s": time.monotonic(), "calls": u.calls, "fresh_calls": u.fresh_calls, "input_tokens": u.input_tokens,
            "fresh_input_tokens": u.fresh_input_tokens, "fresh_inference_s": getattr(client, "fresh_inference_seconds", 0.0)}


def delta(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {k: round(b[k] - a[k], 6) for k in a}


def with_ledger(env: Env, run_dir: Path, fn: Any) -> Any:
    run_dir.mkdir(parents=True, exist_ok=True)
    led = ResponseLedger(run_dir, name=getattr(env, "ledger_name", LEDGER_NAME))
    env.client.ledger = led
    try:
        return fn()
    finally:
        env.client.ledger = None
        led.finalize()


def consume(run_dir: Path, name: str, program_text: str, env: Env, gold: list[str], pred: list[str | None],
            dists: list[dict[str, float]], extra: dict[str, Any]) -> dict[str, Any]:
    """The single test evaluation of a step (test_consumed.json + runs/jev-opt/test_ledger.jsonl)."""
    c = env.config
    m = metrics.compute(gold, pred, dists, list(c.labels), c.primary_metric, c.negative_label)
    path = c.split_path("test")
    return record_consumption(run_dir, {"representative": name, "candidate": getattr(env, "candidate", "clef-rq3"), "model": env.client.model,
                                        "program_hash": load_decider(program_text).digest() if program_text else None,
                                        "test_path": str(path), "test_sha256": sha256_file(path), "metrics": m, **extra})


def check_replay(run_dir: Path, what: str, recorded: dict[str, Any], got: dict[str, Any]) -> None:
    if recorded != got:
        diff = [k for k in recorded if recorded.get(k) != got.get(k)][:5]
        raise SystemExit(f"{run_dir.name}: re-executed {what} differs from the recorded one (e.g. {diff})")


# --------------------------------------------------------------------------------------------
# steps


def step_cal(model: str, stem: str, env: Env, src: Path, out: Path) -> dict[str, Any]:
    labels = env.labels
    seed_text = (src / "seed.yaml").read_text()
    seed = load_decider(seed_text)
    rec = load_json(out / "result.json") or {}
    out.mkdir(parents=True, exist_ok=True)
    (out / "seed.yaml").write_text(seed_text)
    t: dict[str, Any] = {}

    def answer() -> tuple[list, list, list | None]:
        c0 = counters(env.client)
        fit_res = env.run(seed, env.fit, "train")
        c1 = counters(env.client)
        val_res = env.run(seed, env.val, "val")
        c2 = counters(env.client)
        t.update(fit=delta(c0, c1), val=delta(c1, c2))
        test_res = None
        if not ident["all_identical"]:
            test_res = env.run(seed, env.test, "test")
            t["test"] = delta(c2, counters(env.client))
        return fit_res, val_res, test_res

    ident = seed_identity(model, stem, seed, env.test)
    fit_res, val_res, test_res = with_ledger(env, out, answer)
    dists = [floored(r.dist, labels) for r in fit_res]
    bias, temp, rep = fit_label_bias(labels, [[d] for d in dists], [r.gold for r in fit_res], env.config.primary_metric,
                                     env.config.negative_label)
    cal = lambda ds: [combine(labels, [floored(d, labels)], bias, temp)[1] for d in ds]  # noqa: E731
    val_gold = [r.gold for r in val_res]
    val_cal = cal([r.dist for r in val_res])
    test_ids = [e.id for e in env.test]
    test_gold = [e.label for e in env.test]
    if test_res is None:  # frozen zero-shot distributions: byte-identical requests, no new call
        rows = frozen_rows(model, stem)
        assert [rows[i]["gold"] for i in test_ids] == test_gold, f"{stem}: frozen gold differs from the jev-opt test split"
        test_dists = [normalise(rows[i]["probs"], labels) for i in test_ids]
        test_zero = [rows[i]["pred"] for i in test_ids]
        source = f"frozen zero-shot responses runs/{model}/{SETS[stem][1]} (byte-identical requests)"
    else:
        assert [r.id for r in test_res] == test_ids
        test_dists = [r.dist for r in test_res]
        test_zero = [r.pred for r in test_res]
        source = "seed program answered on the test split (requests differ from the frozen audit)"
    test_cal = cal(test_dists)
    result = {
        "set": stem, "model": PINS[model], "source_run": src.name, "labels": labels, "primary_metric": env.config.primary_metric,
        "fit_ids": [e.id for e in env.fit], "seed_identity": ident, "test_distributions": source,
        "calibration": {"bias": bias, "temperature": temp, "fit_primary_before_after": [rep["before"], rep["after"]], "n_fit": len(fit_res)},
        "fit": {"gold": [r.gold for r in fit_res], "seed_pred": [r.pred for r in fit_res]},
        "val": {"ids": [r.id for r in val_res], "gold": val_gold, "seed_pred": [r.pred for r in val_res], "calibrated_pred": val_cal,
                "seed_primary": env.primary(val_gold, [r.pred for r in val_res]), "calibrated_primary": env.primary(val_gold, val_cal)},
        "test": {"ids": test_ids, "gold": test_gold, "calibrated_pred": test_cal, "calibrated_primary": env.primary(test_gold, test_cal),
                 "seed_pred": None if test_res is None else test_zero,
                 "seed_primary": None if test_res is None else env.primary(test_gold, test_zero)},
        "rejected": env.rejected, "usage": rec.get("usage") or t, "computed_at": rec.get("computed_at") or now(),
    }
    if consumed(out):
        check_replay(out, "calibrated test predictions", rec["test"], result["test"])
    dump_json(out / "result.json", result)
    if not consumed(out):
        consume(out, "calibrated", seed_text, env, test_gold, test_cal, [combine(labels, [floored(d, labels)], bias, temp)[0] for d in test_dists],
                {"step": "cal", "test_distributions": source})
    return result


def step_opt(model: str, stem: str, env: Env, src: Path, out: Path) -> dict[str, Any]:
    rec = load_json(out / "result.json") or {}
    out.mkdir(parents=True, exist_ok=True)
    jev_test = json.loads((src / "test_consumed.json").read_text())
    reps: dict[str, Any] = {}
    fitted: dict[str, Any] = {}
    t: dict[str, Any] = {}

    def answer_val() -> None:
        for name in REPS:
            prog = load_decider((src / f"{name}.yaml").read_text())
            blank = ff.unfitted(prog, drop_aux=False)
            (out / f"{name}.unfitted.yaml").write_text(blank.to_yaml())
            c0 = counters(env.client)
            try:
                fit = ff.refit(env, blank)  # type: ignore[arg-type]  # Env duck-types ff.Env
            except ValueError as exc:  # e.g. a readout whose fit answers hold a single class
                reps[name] = {"fit_error": str(exc)[:500]}
                t[f"{name}.fit"] = delta(c0, counters(env.client))
                continue
            c1 = counters(env.client)
            (out / f"{name}.yaml").write_text(fit.to_yaml())
            fitted[name] = fit
            vres = env.run(fit, env.val, "val")
            t[f"{name}.fit"], t[f"{name}.val"] = delta(c0, c1), delta(c1, counters(env.client))
            reps[name] = {"readout_fit": {s: fit.nodes[s].readout.fit for s in ff.readout_slots(fit)},
                          "program_hash": fit.digest(), "val_pred": {r.id: r.pred for r in vres}}

    with_ledger(env, out, answer_val)
    val_gold = {e.id: e.label for e in env.val}
    ids = sorted(val_gold)
    val = {n: env.primary([val_gold[i] for i in ids], [r["val_pred"][i] for i in ids]) for n, r in reps.items() if "val_pred" in r}
    if "top-single" not in val:
        raise SystemExit(f"{stem}: top-single could not be refitted: {reps['top-single']}")
    chosen = choose(val)
    selection = {"val": val, "margin_points": MARGIN, "chosen": chosen, "jev_chosen": jev_test["representative"]}
    prev = load_json(out / "selection.json")
    if prev is not None and prev["chosen"] != chosen:
        raise SystemExit(f"{out.name}: selection changed from {prev['chosen']} to {chosen}; the rule is fixed")
    if prev is None:
        dump_json(out / "selection.json", {**selection, "selected_at": now()})  # written before any test answer

    test_gold = [e.label for e in env.test]
    holder: dict[str, Any] = {}

    def answer_test() -> None:
        c0 = counters(env.client)
        holder["res"] = env.run(fitted[chosen], env.test, "test")
        t["test"] = delta(c0, counters(env.client))

    with_ledger(env, out, answer_test)
    res = holder["res"]
    test = {"ids": [r.id for r in res], "gold": test_gold, "pred": [r.pred for r in res], "primary": env.primary(test_gold, [r.pred for r in res])}
    if consumed(out):
        check_replay(out, "optimized test predictions", rec["test"], test)
    result = {"set": stem, "model": env.client.model, "source_run": src.name, "jev_test_representative": jev_test["representative"],
              "representatives": reps, "selection": selection, "test": test, "rejected": env.rejected,
              "usage": rec.get("usage") or t, "computed_at": rec.get("computed_at") or now()}
    dump_json(out / "result.json", result)
    if not consumed(out):
        consume(out, chosen, fitted[chosen].to_yaml(), env, test_gold, test["pred"], [r.dist for r in res], {"step": "opt"})
    return result


def client_for(model: str, device: str | None) -> ClefBackend:
    """The configured Clef backend through the executor's client factory (JEV_OPT_BACKEND / JEV_OPT_DEVICE)."""
    os.environ["JEV_OPT_BACKEND"] = model
    if device:
        os.environ["JEV_OPT_DEVICE"] = device
    else:
        os.environ.pop("JEV_OPT_DEVICE", None)
    client = make_client(RUNS / ".jev-cache", "", 0, 0, [ROOT])
    assert isinstance(client, ClefBackend)
    return client


def cmd_run(model: str, device: str | None, stem: str) -> None:
    src = RUNS / SETS[stem][0]
    client = client_for(model, device)
    t0 = time.monotonic()
    try:
        env = Env(src, client, test=True)
        log(model, f"{stem} cal start (device {device or 'cache-only'})")
        c = step_cal(model, stem, env, src, cal_dir(model, stem))
        log(model, f"{stem} cal done: val {c['val']['seed_primary']:.4f} -> {c['val']['calibrated_primary']:.4f}, "
                   f"test calibrated {c['test']['calibrated_primary']:.4f}; usage {json.dumps(c['usage'])}")
        env.rejected = []
        log(model, f"{stem} opt start")
        o = step_opt(model, stem, env, src, opt_dir(model, stem))
        log(model, f"{stem} opt done: val {o['selection']['val']} chosen {o['selection']['chosen']}, test {o['test']['primary']:.4f}; "
                   f"usage {json.dumps(o['usage'])}")
    finally:
        client.close()
    log(model, f"{stem} finished in {time.monotonic() - t0:.0f} s; fresh requests {client.usage.fresh_calls}, "
               f"fresh inference {client.fresh_inference_seconds:.0f} s")


def cmd_replay(model: str, stem: str) -> None:
    """Cache-only re-execution (no device): every recorded val/test prediction must come back."""
    src = RUNS / SETS[stem][0]
    client = client_for(model, None)
    try:
        env = Env(src, client, test=True)
        for step, fn, out in (("cal", step_cal, cal_dir(model, stem)), ("opt", step_opt, opt_dir(model, stem))):
            before = json.loads((out / "result.json").read_text())
            after = fn(model, stem, env, src, out)  # check_replay compares test predictions; val compared here
            keys = ("val",) if step == "cal" else ("representatives",)
            for k in keys:
                if json.dumps(before[k], sort_keys=True) != json.dumps(after[k], sort_keys=True):
                    raise SystemExit(f"{out.name}: replayed {k} differs from the recorded one")
            log(model, f"{stem} {step} replay identical (fresh requests {client.usage.fresh_calls})")
            env.rejected = []
    finally:
        client.close()


def cmd_smoke(model: str, device: str, stem: str, n: int) -> None:
    src = RUNS / SETS[stem][0]
    out = RUNS / f"cl-{model}-smoke-{stem}"
    progs = {"seed": load_decider((src / "seed.yaml").read_text())}
    progs.update({name: ff.unfitted(load_decider((src / f"{name}.yaml").read_text()), drop_aux=False) for name in REPS})

    def answers(client: Any) -> tuple[dict[str, Any], dict[str, float]]:
        env = Env(src, client, test=False)
        got: dict[str, Any] = {}
        c0 = counters(client)
        for name, prog in progs.items():
            res = env.run(prog, env.fit[:n], "train", exhaustive=name != "seed")
            got[name] = [{"id": r.id, "memo": r.memo, "dist": r.dist, "pred": r.pred, "error": r.error} for r in res]
        return got, {**delta(c0, counters(client)), "rejected": len(env.rejected)}

    live = client_for(model, device)
    out.mkdir(parents=True, exist_ok=True)
    led = ResponseLedger(out, name=LEDGER_NAME)
    live.ledger = led
    t0 = time.monotonic()
    try:
        first, use1 = answers(live)
    finally:
        live.ledger = None
        led.finalize()
        live.close()
    wall1 = time.monotonic() - t0
    replay = client_for(model, None)
    try:
        second, use2 = answers(replay)
    finally:
        replay.close()
    same = json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    report = {"set": stem, "model": PINS[model], "device": device, "n_fit_items": n, "programs": list(progs),
              "live": {**use1, "wall_s_incl_load": round(wall1, 3)}, "replay": use2, "replay_identical": same,
              "server_configuration": live._config_sha and json.loads((live.cache_dir / "configurations" / f"{live._config_sha}.json").read_text()),
              "computed_at": now()}
    dump_json(out / "smoke.json", report)
    log(model, f"smoke {stem} n={n}: live {use1}, wall {wall1:.1f} s; replay {use2}; identical {same}")
    if not same:
        raise SystemExit("cache replay differs from the live answers")


WORKLOAD = HERE / "clef_rq3_workload.json"
ASSIGNMENT = HERE / "clef_rq3_assignment.json"


def set_workload(stem: str) -> dict[str, Any]:
    """Unique requests and serialized request characters per step, from the executor with uniform stand-in
    answers (no model; exhaustive fit runs as in the refit). Model-independent: both models get the same bodies."""
    src = RUNS / SETS[stem][0]
    cfg = run_config(src)
    train, val = load_split(cfg, "train"), load_split(cfg, "val")
    with allow_test_split():
        test = load_split(cfg, "test")
    fit, _ = train_roles(train, cfg.optimizer)
    pool = demo_pool(cfg, train)
    seen: set[str] = set()
    out: dict[str, Any] = {"n": {"fit": len(fit), "val": len(val), "test": len(test)}, "steps": {}}

    def count(name: str, prog: Any, examples: list[Any], exhaustive: bool) -> None:
        rec = _Recorder()
        rec.concurrency = 16  # type: ignore[misc]
        if exhaustive:
            run_examples(prog, examples, rec, exhaustive=True, pool=pool)  # type: ignore[arg-type]
        else:
            run_decider(prog, examples, rec, pool=pool)  # type: ignore[arg-type]
        n = chars = 0
        for state, questions in rec.bodies:
            blob = json.dumps({"state": state, "questions": questions}, ensure_ascii=False)
            h = hashlib.sha256(blob.encode("utf-8")).hexdigest()
            if h not in seen:
                seen.add(h)
                n, chars = n + 1, chars + len(blob)
        out["steps"][name] = {"requests": n, "chars": chars}

    seed = load_decider((src / "seed.yaml").read_text())
    count("seed.fit", seed, fit, False)
    count("seed.val", seed, val, False)
    count("seed.test", seed, test, False)
    for name in REPS:
        prog = ff.unfitted(load_decider((src / f"{name}.yaml").read_text()), drop_aux=False)
        count(f"{name}.fit", prog, fit, True)
        count(f"{name}.val", load_decider((src / f"{name}.yaml").read_text()), val, False)
    for name in REPS:
        count(f"{name}.test", load_decider((src / f"{name}.yaml").read_text()), test, False)
    return out


def planned(work: dict[str, Any], seed_test: bool, test_rep: str) -> dict[str, int]:
    """Requests/characters a set's run is expected to send (the chosen representative is not known
    before validation: `test_rep` stands in for it)."""
    keys = ["seed.fit", "seed.val", *(["seed.test"] if seed_test else []), *(f"{r}.{s}" for r in REPS for s in ("fit", "val")), f"{test_rep}.test"]
    return {k: sum(work["steps"][x][k] for x in keys) for k in ("requests", "chars")}


def cmd_workload(stems: list[str]) -> None:
    data = load_json(WORKLOAD) or {}
    for stem in stems:
        t0 = time.monotonic()
        data[stem] = set_workload(stem)
        print(f"{stem}: {json.dumps(data[stem]['steps'])} ({time.monotonic() - t0:.0f} s)", flush=True)
        dump_json(WORKLOAD, data)


def embedding_paths(stem: str, record: set[Path]) -> None:
    """Prefetch every kNN demonstration embedding the set's programs can need (pool + fit/val/test queries)."""
    src = RUNS / SETS[stem][0]
    cfg = run_config(src)
    train, val = load_split(cfg, "train"), load_split(cfg, "val")
    with allow_test_split():
        test = load_split(cfg, "test")
    fit, _ = train_roles(train, cfg.optimizer)
    pool = demo_pool(cfg, train, record)
    for name in ("seed", *REPS):
        prog = load_decider((src / f"{name}.yaml").read_text())
        pool.prefetch(prog.demos, fit + val + test)


def cmd_embed_archive() -> None:
    """Deterministic archive of the demonstration embeddings the 13 sets need (relative to .embed-cache)."""
    import gzip
    import io
    import tarfile

    record: set[Path] = set()
    for stem in SETS:
        embedding_paths(stem, record)
    base = RUNS / ".embed-cache"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(record):
            info = tar.gettarinfo(str(path), arcname=str(path.relative_to(base)))
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            with open(path, "rb") as fh:
                tar.addfile(info, fh)
    with open(EMBED_ARCHIVE, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(buf.getvalue())
    print(f"{len(record)} embeddings -> {EMBED_ARCHIVE.relative_to(ROOT)} ({EMBED_ARCHIVE.stat().st_size / 1e6:.1f} MB)")


def cmd_assign() -> None:
    """Two-GPU split for the H100 machine: sets by planned characters (seed answered on test where the frozen
    requests differ; the Jev-chosen representative stands in for the chosen one), largest first, each to the
    GPU with the smaller running total (ties: gpu0)."""
    work = json.loads(WORKLOAD.read_text())
    est = {}
    for stem in SETS:
        jev_rep = json.loads((RUNS / SETS[stem][0] / "test_consumed.json").read_text())["representative"]
        ident = load_json(cal_dir("clef", stem) / "result.json")
        seed_test = stem in SEED_TEST_SETS if ident is None else not ident["seed_identity"]["all_identical"]
        est[stem] = planned(work[stem], seed_test, jev_rep)
    load = {"gpu0": 0, "gpu1": 0}
    slots: dict[str, list[str]] = {"gpu0": [], "gpu1": []}
    for stem in sorted(SETS, key=lambda s: (-est[s]["chars"], s)):
        slot = min(load, key=lambda g: (load[g], g))
        slots[slot].append(stem)
        load[slot] += est[stem]["chars"]
    dump_json(ASSIGNMENT, {"rule": cmd_assign.__doc__.split("\n\n")[0].strip().replace("\n   ", ""), "slots": slots,
                           "planned_chars": load, "per_set": est})
    print(json.dumps({"slots": slots, "planned_chars": load}, indent=1))


# Sets whose seed-program test requests differ from both models' frozen requests (question id or a joint
# three-rubric request; `seed_identity`, checked again by `plan` and at run time).
SEED_TEST_SETS = {"ezstance-subtaskA", "edit-intent", "dress-new-content", "dress-new-organization", "dress-new-language"}


def cmd_plan(model: str, device: str | None, slot: str | None, stems: list[str]) -> dict[str, Any]:
    """Dry run: validates the assignment, inputs, pins, weights and cache path; loads no model, makes no call."""
    pin_id, rev = PINS[model].split("@")
    snap = ROOT / "data" / "cache" / "huggingface" / f"models--{pin_id.replace('/', '--')}" / "snapshots" / rev
    cache = Path(os.environ.get("JEV_OPT_CLEF_CACHE") or ROOT / "data" / "cache" / f"{model}-jev-opt")
    problems: list[str] = []
    if slot is not None:
        slots = json.loads(ASSIGNMENT.read_text())["slots"]
        every = sorted(s for v in slots.values() for s in v)
        if every != sorted(SETS):
            problems.append(f"assignment does not cover each set exactly once: {every}")
        stems = slots[slot]
    weights = sorted(p.name for p in snap.glob("*.safetensors")) if snap.exists() else []
    if not weights or not (snap / "joint_schema_model.py").exists():
        problems.append(f"pinned weights/release code missing at {snap}")
    try:
        cache.mkdir(parents=True, exist_ok=True)
        probe = cache / f".plan-probe-{os.getpid()}"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        problems.append(f"cache {cache} not writable: {exc}")
    work = load_json(WORKLOAD) or {}
    rows = []
    for stem in stems:
        src = RUNS / SETS[stem][0]
        cfg = run_config(src)
        with allow_test_split():
            test = load_split(cfg, "test")
        val = load_split(cfg, "val")
        fit, _ = train_roles(load_split(cfg, "train"), cfg.optimizer)
        for name in ("seed", *REPS):
            load_decider((src / f"{name}.yaml").read_text())
        jev_rep = json.loads((src / "test_consumed.json").read_text())["representative"]
        try:
            n_emb: set[Path] = set()
            embedding_paths(stem, n_emb)
        except EmbeddingMiss as exc:
            problems.append(f"{stem}: {exc}")
        ident = seed_identity(model, stem, load_decider((src / "seed.yaml").read_text()), test)
        if (stem in SEED_TEST_SETS) == ident["all_identical"]:
            problems.append(f"{stem}: identity {ident['identical']}/{ident['n_test']} contradicts SEED_TEST_SETS")
        state = {s: ("test consumed" if consumed(d(model, stem)) else "pending") for s, d in (("cal", cal_dir), ("opt", opt_dir))}
        est = planned(work[stem], not ident["all_identical"], jev_rep) if stem in work else None
        rows.append({"set": stem, "source_run": src.name, "fit": len(fit), "val": len(val), "test": len(test),
                     "seed_test_requests_identical": f"{ident['identical']}/{ident['n_test']}",
                     "calibration_test_distributions": "frozen zero-shot" if ident["all_identical"] else "seed program on test",
                     "demo_embeddings_cached": len(n_emb),
                     "outputs": [cal_dir(model, stem).name, opt_dir(model, stem).name], "state": state, "planned": est})
    plan = {"model": PINS[model], "device": device, "slot": slot, "weights": {"snapshot": str(snap.relative_to(ROOT)), "safetensors": len(weights)},
            "cache": str(cache.relative_to(ROOT)) if cache.is_relative_to(ROOT) else str(cache),
            "server_command": shlex.join(server_command(model, device, 32000, cache / "scratch")) if device else None,
            "sets": rows, "problems": problems, "checked_at": now()}
    print(json.dumps({k: v for k, v in plan.items() if k != "sets"}, indent=1))
    for r in rows:
        print(f"{r['set']:24s} fit {r['fit']} val {r['val']} test {r['test']}; seed requests identical {r['seed_test_requests_identical']} "
              f"-> {r['calibration_test_distributions']}; planned {r['planned']}; {r['state']}")
    return plan


def cmd_replicate_jev(stems: list[str]) -> None:
    """The driver's steps on Jev's own cache (posting disabled by framing_features): refit val predictions,
    selection and test score must equal the recorded ones; calibrated val must equal results/frozen.json."""
    frozen = json.loads((ROOT / "results" / "frozen.json").read_text())
    rows = {r["stem"]: r for r in frozen["new"] + frozen["orig"]}
    out: dict[str, Any] = {}
    for stem in stems:
        src = RUNS / SETS[stem][0]
        cfg = run_config(src)
        client = make_client(RUNS / ".jev-cache", cfg.jev.model, cfg.jev.max_rpm, cfg.jev.concurrency, [ROOT])
        try:
            env = Env(src, client, test=True)
            seed = load_decider((src / "seed.yaml").read_text())
            fit_res = env.run(seed, env.fit, "train")
            labels = env.labels
            bias, temp, _ = fit_label_bias(labels, [[floored(r.dist, labels)] for r in fit_res], [r.gold for r in fit_res],
                                           cfg.primary_metric, cfg.negative_label)
            vres = env.run(seed, env.val, "val")
            vcal = env.primary([r.gold for r in vres], [combine(labels, [floored(r.dist, labels)], bias, temp)[1] for r in vres])
            recorded_val = json.loads((src / "val_predictions.json").read_text())["representatives"]
            agree, val, fitted = {}, {}, {}
            for name in REPS:
                fit = ff.refit(env, ff.unfitted(load_decider((src / f"{name}.yaml").read_text()), drop_aux=False))  # type: ignore[arg-type]
                fitted[name] = fit
                vr = env.run(fit, env.val, "val")
                agree[name] = sum(recorded_val[name][r.id] == r.pred for r in vr) / len(vr)
                val[name] = env.primary([r.gold for r in vr], [r.pred for r in vr])
            chosen = choose(val)
            consumed_rec = json.loads((src / "test_consumed.json").read_text())
            tres = env.run(fitted[chosen], env.test, "test")
            test = env.primary([r.gold for r in tres], [r.pred for r in tres])
            out[stem] = {"calibrated_val": vcal, "recorded_calibrated_val": rows[stem]["calibration"]["val_calibrated"],
                         "val_agreement": agree, "chosen": chosen, "recorded_chosen": consumed_rec["representative"],
                         "test": test, "recorded_test": consumed_rec["metrics"]["primary"], "fresh_calls": client.usage.fresh_calls}
            ok = (abs(vcal - out[stem]["recorded_calibrated_val"]) < 1e-12 and min(agree.values()) == 1.0
                  and chosen == consumed_rec["representative"] and abs(test - consumed_rec["metrics"]["primary"]) < 1e-12)
            out[stem]["reproduced"] = ok
            print(f"{stem}: reproduced {ok} {json.dumps(out[stem])}", flush=True)
        finally:
            client.close()
    path = ROOT / "results" / "clef_rq3" / "jev_replication.json"
    prev = load_json(path) or {}
    dump_json(path, {**prev, **out})


def cmd_estimate() -> None:
    """Markdown table of expected run time per set on one H100 (no inference). Tokens = planned request
    characters / the model's characters per token on the 8 sets whose frozen zero-shot requests equal the seed
    program's test requests (`seed.test` characters vs the frozen manifests' input tokens); hours = tokens / the
    native throughput (input tokens per inference second) of the model's frozen run of the same set on the H100
    machine. Measured values replace the estimate once a set has run."""
    assign = json.loads(ASSIGNMENT.read_text())
    work = json.loads(WORKLOAD.read_text())
    lines = ["| GPU | Set | Requests (planned) | Clef: input tokens, GPU h | Clef-flash: input tokens, GPU h |", "|---|---|---:|---|---|"]
    total = {(g, m): 0.0 for g in ("gpu0", "gpu1") for m in ("clef", "clef-flash")}
    manifest = {(m, s): json.loads((ROOT / "runs" / m / SETS[s][1] / "manifest.json").read_text()) for m in PINS for s in SETS}
    same = [s for s in SETS if s not in SEED_TEST_SETS]
    cpt = {m: sum(work[s]["steps"]["seed.test"]["chars"] for s in same) / sum(manifest[(m, s)]["input_tokens"] for s in same) for m in PINS}
    for g in ("gpu0", "gpu1"):
        for stem in assign["slots"][g]:
            est = assign["per_set"][stem]
            cells = []
            for model in ("clef", "clef-flash"):
                man = manifest[(model, stem)]
                tokens = est["chars"] / cpt[model]
                hours, tag = tokens / (man["input_tokens"] / man["local_compute"]["inference_seconds"]) / 3600, "est."
                done = [load_json(d(model, stem) / "result.json") for d in (cal_dir, opt_dir)]
                if all(done) and all(consumed(d(model, stem)) for d in (cal_dir, opt_dir)):
                    use = [v for r in done for v in r["usage"].values()]
                    tokens, hours, tag = sum(v["fresh_input_tokens"] for v in use), sum(v["fresh_inference_s"] for v in use) / 3600, "measured"
                total[(g, model)] += hours
                cells.append(f"{tokens / 1e6:.1f} M, {hours:.1f} h ({tag})")
            lines.append(f"| {g} | {stem} | {est['requests']:,} | {cells[0]} | {cells[1]} |")
    for g in ("gpu0", "gpu1"):
        lines.append(f"| {g} | **total** | | {total[(g, 'clef')]:.1f} h | {total[(g, 'clef-flash')]:.1f} h |")
    lines.append(f"\nCharacters per token: Clef {cpt['clef']:.2f}, Clef-flash {cpt['clef-flash']:.2f}.")
    print("\n".join(lines))

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["run", "smoke", "replay", "plan", "workload", "assign", "estimate", "embed-archive", "replicate-jev"])
    ap.add_argument("--model", choices=sorted(PINS))
    ap.add_argument("--device")
    ap.add_argument("--slot", choices=["gpu0", "gpu1"], help="plan: the sets of this slot of clef_rq3_assignment.json")
    ap.add_argument("--set", action="append", dest="sets", choices=list(SETS))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", type=Path, help="plan: also write the plan as JSON here")
    a = ap.parse_args()
    stems = a.sets or list(SETS)
    if a.command == "replicate-jev":
        cmd_replicate_jev(stems)
        return 0
    if a.command == "workload":
        cmd_workload(stems)
        return 0
    if a.command == "assign":
        cmd_assign()
        return 0
    if a.command == "estimate":
        cmd_estimate()
        return 0
    if a.command == "embed-archive":
        cmd_embed_archive()
        return 0
    if a.model is None:
        ap.error("--model is required")
    if a.command == "plan":
        plan = cmd_plan(a.model, a.device, a.slot, stems)
        if a.out:
            dump_json(a.out, plan)
        return 1 if plan["problems"] else 0
    if a.command == "replay":
        for s in stems:
            cmd_replay(a.model, s)
        return 0
    if not a.device or len(stems) != 1:
        ap.error(f"{a.command} needs --device and exactly one --set")
    if a.command == "run":
        cmd_run(a.model, a.device, stems[0])
    else:
        cmd_smoke(a.model, a.device, stems[0], a.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
