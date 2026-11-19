"""Convention ablation (pre-registered) -> results/CONVENTIONS_ABLATION.{md,json}, paper/generated/conventions.tex.

  --run     per program of results/conventions_spans.json (in file order; resumable):
              A  the chosen program, readout refit on the run's fit slice (framing_features.refit); val/test from the
                 cache only (a fresh Jev call in A is an error); refit check against the recorded predictions
              B  A minus the convention spans, C  A minus equal-length non-convention text: same refit, val, test
              D  the paper's question + the spans (zero-shot) on fit slice, val, test; E = D + calibration step
              Z  the paper's question (zero-shot) on the same path; F = Z + calibration step
            New run directories runs/jev-opt/cv-<task>-s<seed>-{B,C,D} and cv-<task>-Z hold the edited program, the Jev
            response ledger and a test consumption record; per-item predictions go to results/conventions_ablation/.
            Run under /tmp/jev-run.lock.
  --report  statistics (no Jev call): per-program paired bootstrap, hierarchical pooled bootstrap, Holm, sign tests.

Run from the repository root:
  flock /tmp/jev-run.lock uv run --project jev-opt/python \\
      python jev-opt/experiments/conventions_ablation.py --run
"""

from __future__ import annotations

import argparse
import os
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

assert sys.version_info[:2] == (3, 11), f"run under Python 3.11 (worker's version), not {sys.version}"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from jev_opt.jev import JevClient, ResponseLedger  # noqa: E402

_POST = JevClient._post
_KEY_WAS_SET = "OPENROUTER_API_KEY" in os.environ
import framing_features as ff  # noqa: E402  (Env, refit, unfitted; its import disables posting)

JevClient._post = _POST  # this round makes fresh calls; cell A checks that it made none
if not _KEY_WAS_SET and os.environ.get("OPENROUTER_API_KEY") == "offline-cache-only":
    del os.environ["OPENROUTER_API_KEY"]  # framing_features' placeholder; the real key comes from .env via load_api_key

from jev_opt import metrics  # noqa: E402
from jev_opt.adapter import train_roles  # noqa: E402
from jev_opt.execute import combine  # noqa: E402
from jev_opt.fit import fit_label_bias  # noqa: E402
from jev_opt.hygiene import record_consumption, sha256_file  # noqa: E402
from jev_opt.ir import load_decider  # noqa: E402

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
import paper_tables  # noqa: E402  (labels of the configurations: CONFIG_LABEL)
RUNS = ROOT / "runs" / "jev-opt"
SPANS = ROOT / "results" / "conventions_spans.json"
CELLS = ROOT / "results" / "conventions_ablation"
OUT_JSON = ROOT / "results" / "CONVENTIONS_ABLATION.json"
OUT_MD = ROOT / "results" / "CONVENTIONS_ABLATION.md"
OUT_TEX = ROOT / "paper" / "generated" / "conventions.tex"
OUT_APP = ROOT / "paper" / "generated" / "app_conventions.tex"
OUT_APP_SPANS = ROOT / "paper" / "generated" / "app_conventions_spans.tex"
LOG = RUNS / "conventions.log"
B = 2000
FLOOR = 1e-4  # confidence_readout.FLOOR (jev_opt's BIAS_FLOOR)

PAPER = {
    "contempo-tbdense": "contempo", "contempo-matres": "contempo", "ezstance-subtaskA": "ezstance",
    "smarter-implicit-hate": "smarter", "smarter-hatexplain": "smarter", "edit-intent": "edit_intent",
    "dress-new-content": "dress", "dress-new-organization": "dress", "dress-new-language": "dress",
    "lifetox": "lifetox", "sciprompt-s2orc": "sciprompt",
}
CONTRASTS = [  # (name, minuend, subtrahend, family)
    ("A-B", "A", "B", "deletion"), ("C-B", "C", "B", "deletion"),
    ("D-Z", "D", "Z", "insertion"), ("E-F", "E", "F", "insertion"),
]
DESCRIPTIVE = [("A-C", "A", "C"), ("E-Z", "E", "Z"), ("F-Z", "F", "Z"), ("A-Z", "A", "Z")]
# Cells of the first run that were superseded (pre-fallback B/C, whose D/Z answers the final run then read from the
# cache): TB-Dense s0 and s1 and EZ-STANCE s0, plus EZ-STANCE s1's B, which ran before its C was rejected. Their fresh
# Jev spend, summed from those cells' result files and the run log:
SUPERSEDED_COST = {"fresh_cost": 0.6105185940000009, "fresh_calls": 27437}


def log(msg: str) -> None:
    line = f"{datetime.now(UTC).isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def floored(d: dict[str, float] | None, labels: list[str]) -> dict[str, float]:
    v = {lab: max(float((d or {}).get(lab, 0.0)), FLOOR) for lab in labels}
    s = sum(v.values())
    return {lab: x / s for lab, x in v.items()}


def dump(program: dict) -> str:
    return yaml.safe_dump(program, sort_keys=False, allow_unicode=True, width=10**6)


def usage(env: ff.Env) -> dict[str, float]:
    u = env.client.usage
    return {"fresh_calls": u.fresh_calls, "fresh_cost": u.fresh_cost, "fresh_input_tokens": u.fresh_input_tokens, "calls": u.calls}


def diff(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {k: b[k] - a[k] for k in a}




def consume(cell_dir: Path, name: str, program_text: str, env: ff.Env, results: list) -> dict[str, Any] | None:
    """Test consumption record for a new evaluation (test_consumed.json + the shared test_ledger.jsonl)."""
    if (cell_dir / "test_consumed.json").exists():  # resumed after a crash: the record stays the first one
        log(f"{cell_dir.name}: test already recorded at {json.loads((cell_dir / 'test_consumed.json').read_text())['consumed_at']}")
        return None
    c = env.config
    m = metrics.compute([r.gold for r in results], [r.pred for r in results], [r.dist or {} for r in results],
                        list(c.labels), c.primary_metric, c.negative_label)
    m["total_cost"] = float(sum(r.cost for r in results))
    path = c.split_path("test")
    return record_consumption(cell_dir, {"representative": name, "candidate": "conventions-ablation",
                                         "program_hash": load_decider(program_text).digest(), "test_path": str(path),
                                         "test_sha256": sha256_file(path), "metrics": m})


def fill_empty(edited: dict, original: dict) -> tuple[dict, list[str]]:
    """The pre-registered empty-field fallback: a text that B or C emptied becomes its option key (option
    texts), the node's option keys joined by " / " (instructions) or the node id (Branch conditions)."""
    q = json.loads(json.dumps(edited))
    filled = []
    for nid, node in q["nodes"].items():
        orig = original["nodes"][nid]
        if "instructions" in node and not node["instructions"].strip() and orig["instructions"].strip():
            node["instructions"] = " / ".join(node.get("options") or node.get("levels"))
            filled.append(f"{nid}.instructions")
        if "condition" in node and not node["condition"].strip() and orig["condition"].strip():
            node["condition"] = nid
            filled.append(f"{nid}.condition")
        for k, opt in (node.get("options") or {}).items():
            if not opt["criterion"].strip() and orig["options"][k]["criterion"].strip():
                opt["criterion"] = k
                filled.append(f"{nid}.options.{k}")
    return q, filled


def cell_path(name: str) -> Path:
    """Run directory of a new evaluation; `<name>-r2` when a first attempt recorded a test pass in which every
    request failed (no answer obtained), which is kept as a record of the failed attempt."""
    tc = RUNS / name / "test_consumed.json"
    if tc.exists():
        m = json.loads(tc.read_text())["metrics"]
        if m.get("errors", 0) == m["n"]:
            return RUNS / f"{name}-r2"
    return RUNS / name


def checked_env(d: Path) -> ff.Env:
    """ff.Env whose runs fail loudly when more than 1% of the examples fail (e.g. a rejected request)."""
    env = ff.Env(d)
    plain = env.run

    def run(program, examples, split, exhaustive=False):
        res = plain(program, examples, split, exhaustive)
        bad = [r for r in res if r.error is not None]
        if len(bad) > 0.01 * len(res):
            raise SystemExit(f"{d.name} {split}: {len(bad)}/{len(res)} examples failed, e.g. {bad[0].error[:400]}")
        return res

    env.run = run
    return env


def ledger_on(env: ff.Env, cell_dir: Path) -> ResponseLedger:
    cell_dir.mkdir(parents=True, exist_ok=True)
    led = ResponseLedger(cell_dir)
    env.client.ledger = led
    return led


def ledger_off(env: ff.Env, led: ResponseLedger) -> None:
    env.client.ledger = None
    led.finalize()


def calibrate(env: ff.Env, fit_res: list, labels: list[str]) -> dict[str, Any]:
    dists = [floored(r.dist, labels) for r in fit_res]
    gold = [r.gold for r in fit_res]
    bias, temp, rep = fit_label_bias(labels, [[d] for d in dists], gold, env.config.primary_metric, env.config.negative_label)
    return {"bias": bias, "temperature": temp, "fit_primary_before_after": [rep["before"], rep["after"]], "n_fit": len(fit_res)}


def apply_cal(results: list, labels: list[str], cal: dict[str, Any]) -> dict[str, str | None]:
    return {r.id: (None if r.error else combine(labels, [floored(r.dist, labels)], cal["bias"], cal["temperature"])[1]) for r in results}


def preds(results: list) -> dict[str, str | None]:
    return {r.id: r.pred for r in results}


def errors(results: list) -> int:
    return sum(r.error is not None for r in results)


# --------------------------------------------------------------------------------------------
# run


def run_refit_cell(env: ff.Env, cell: str, prog_dict: dict, cell_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"cell": cell, "run": cell_dir.name, "usage": {}}
    led = ledger_on(env, cell_dir)
    try:
        (cell_dir / "program_unfitted.yaml").write_text(dump(prog_dict))
        u0 = usage(env)
        fitted = ff.refit(env, ff.unfitted(load_decider(dump(prog_dict)), drop_aux=False))
        u1 = usage(env)
        (cell_dir / "program.yaml").write_text(fitted.to_yaml())
        val = env.run(fitted, env.val, "val")
        u2 = usage(env)
        test = env.run(fitted, env.test, "test")
        u3 = usage(env)
        consume(cell_dir, cell, fitted.to_yaml(), env, test)
    finally:
        ledger_off(env, led)
    out["usage"] = {"fit": diff(u0, u1), "val": diff(u1, u2), "test": diff(u2, u3)}
    out["readout_fit"] = {s: fitted.nodes[s].readout.fit for s in ff.readout_slots(fitted)}
    out["val"], out["test"] = preds(val), preds(test)
    out["errors"] = {"val": errors(val), "test": errors(test)}
    return out


def run_program(p: dict, cache: dict[str, Any]) -> Path:
    key = f"{p['task']}-s{p['seed']}"
    out_path = CELLS / f"{key}.json"
    if out_path.exists():
        return out_path
    d = RUNS / p["run"]
    env = checked_env(d)
    labels = list(env.config.labels)
    fit, _ = train_roles(env.train, env.config.optimizer)
    rec: dict[str, Any] = {
        "program": p["program"], "task": p["task"], "seed": p["seed"], "run": p["run"], "representative": p["representative"],
        "primary_metric": env.config.primary_metric, "labels": labels, "negative_label": env.config.negative_label,
        "val_ids": [e.id for e in env.val], "test_ids": [e.id for e in env.test],
        "val_gold": [e.label for e in env.val], "test_gold": [e.label for e in env.test],
        "fit_ids": [e.id for e in fit], "cells": {},
    }
    t0 = time.monotonic()
    try:
        # A: refit from the cache only
        u0 = usage(env)
        consumed = json.loads((d / "test_consumed.json").read_text())
        assert consumed["representative"] == p["representative"], (d.name, consumed["representative"])
        prog = load_decider((d / f"{p['representative']}.yaml").read_text())
        orig_test = env.run(prog, env.test, "test")
        got = env.primary([r.gold for r in orig_test], [r.pred for r in orig_test])
        if abs(got - consumed["metrics"]["primary"]) > 1e-9:
            raise SystemExit(f"{d.name}: re-executed test {got} != recorded {consumed['metrics']['primary']}")
        vp = json.loads((d / "val_predictions.json").read_text())["representatives"][p["representative"]]
        full = ff.refit(env, ff.unfitted(prog, drop_aux=False))
        a_val, a_test = env.run(full, env.val, "val"), env.run(full, env.test, "test")
        if usage(env)["fresh_calls"] != u0["fresh_calls"]:
            raise SystemExit(f"{d.name}: cell A made fresh Jev calls")
        rec["cells"]["A"] = {
            "cell": "A", "val": preds(a_val), "test": preds(a_test), "recorded_test_primary": consumed["metrics"]["primary"],
            "refit_check": {"val_agree": float(np.mean([vp[r.id] == r.pred for r in a_val])),
                            "test_agree": float(np.mean([o.pred == r.pred for o, r in zip(orig_test, a_test)]))},
            "errors": {"val": errors(a_val), "test": errors(a_test)},
        }
        log(f"{key} A refit check {rec['cells']['A']['refit_check']}")

        # B, C: edited program, same refit. Jev rejects an empty Noul question (HTTP 400 "Noul question must have
        # criteria or instructions"), so the pre-registered empty-field fallback applies to every emptied field of every
        # B/C program; such cells run under `cv-<key>-<cell>-fb` (an earlier plain run stays as a record).
        original = yaml.safe_load((d / f"{p['representative']}.yaml").read_text())
        for cell in ("B", "C"):
            prog_cell, filled = fill_empty(p["programs"][cell], original)
            cell_dir = cell_path(f"cv-{key}-{cell}" + ("-fb" if filled else ""))
            rec["cells"][cell] = run_refit_cell(env, cell, prog_cell, cell_dir)
            rec["cells"][cell]["fallback_filled"] = filled
            log(f"{key} {cell} done; usage {rec['cells'][cell]['usage']}")

        # Z: the paper's question on the same path (val/test once per configuration; fit slice per program)
        zkey = p["task"]
        zdir = cell_path(f"cv-{zkey}-Z")
        seed_text = (d / "seed.yaml").read_text()
        led = ledger_on(env, zdir)
        try:
            (zdir / "program.yaml").write_text(seed_text)
            seed_prog = load_decider(seed_text)
            uz0 = usage(env)
            z_fit = env.run(seed_prog, fit, "train")
            uz1 = usage(env)
            zc = cache.get(zkey)
            if zc is None and (CELLS / f"{zkey}-Z.json").exists():
                zc = json.loads((CELLS / f"{zkey}-Z.json").read_text())
            if zc is None:
                z_val, z_test = env.run(seed_prog, env.val, "val"), env.run(seed_prog, env.test, "test")
                uz2 = usage(env)
                consume(zdir, "Z", seed_text, env, z_test)
                zc = {"val_results": z_val, "test_results": z_test, "usage": {"val_test": diff(uz1, uz2)}}
            elif "val_results" not in zc:  # resumed: rebuild from the cache (requests are recorded)
                zc["val_results"], zc["test_results"] = env.run(seed_prog, env.val, "val"), env.run(seed_prog, env.test, "test")
        finally:
            ledger_off(env, led)
        cache[zkey] = zc
        if not (CELLS / f"{zkey}-Z.json").exists():
            (CELLS / f"{zkey}-Z.json").write_text(json.dumps({"task": zkey, "usage": zc["usage"],
                                                              "val": preds(zc["val_results"]), "test": preds(zc["test_results"])}) + "\n")
        fcal = calibrate(env, z_fit, labels)
        rec["cells"]["Z"] = {"cell": "Z", "run": zdir.name, "val": preds(zc["val_results"]), "test": preds(zc["test_results"]),
                             "usage": {"fit": diff(uz0, uz1)}, "errors": {"val": errors(zc["val_results"]), "test": errors(zc["test_results"])}}
        rec["cells"]["F"] = {"cell": "F", "calibration": fcal, "val": apply_cal(zc["val_results"], labels, fcal),
                             "test": apply_cal(zc["test_results"], labels, fcal)}

        # D, E: the paper's question + the spans
        ddir = cell_path(f"cv-{key}-D")
        led = ledger_on(env, ddir)
        try:
            dtext = dump(p["programs"]["D"])
            (ddir / "program.yaml").write_text(dtext)
            dprog = load_decider(dtext)
            ud0 = usage(env)
            d_fit = env.run(dprog, fit, "train")
            ud1 = usage(env)
            d_val = env.run(dprog, env.val, "val")
            ud2 = usage(env)
            d_test = env.run(dprog, env.test, "test")
            ud3 = usage(env)
            consume(ddir, "D", dtext, env, d_test)
        finally:
            ledger_off(env, led)
        ecal = calibrate(env, d_fit, labels)
        rec["cells"]["D"] = {"cell": "D", "run": ddir.name, "val": preds(d_val), "test": preds(d_test),
                             "usage": {"fit": diff(ud0, ud1), "val": diff(ud1, ud2), "test": diff(ud2, ud3)},
                             "errors": {"fit": errors(d_fit), "val": errors(d_val), "test": errors(d_test)}}
        rec["cells"]["E"] = {"cell": "E", "calibration": ecal, "val": apply_cal(d_val, labels, ecal), "test": apply_cal(d_test, labels, ecal)}
        log(f"{key} D done; usage {rec['cells']['D']['usage']}")
    finally:
        env.close()
    rec["wall_s"] = time.monotonic() - t0
    CELLS.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec) + "\n")
    return out_path


def run_all(only: list[str] | None) -> None:
    spans = json.loads(SPANS.read_text())
    CELLS.mkdir(parents=True, exist_ok=True)
    cache: dict[str, Any] = {}
    for p in spans["programs"]:
        key = f"{p['task']}-s{p['seed']}"
        if only and key not in only:
            continue
        done = (CELLS / f"{key}.json").exists()
        log(f"{key} start" + (" (done)" if done else ""))
        path = run_program(p, cache)
        if not done:
            log(f"{key} cells A-F written to {path}")


# --------------------------------------------------------------------------------------------
# report


class Scorer:
    """Vectorized primary metric (jev_opt.metrics.primary) over many bootstrap resamples."""

    def __init__(self, metric: str, labels: list[str], negative: str | None, gold: list[str]):
        self.metric, self.labels, self.negative = metric, labels, negative
        self.K = len(labels)
        self.idx = {lab: i for i, lab in enumerate(labels)}
        self.g = np.array([self.idx[x] for x in gold])

    def codes(self, pred: list[str | None]) -> np.ndarray:
        K = self.K
        if self.metric == "qwk":  # failed predictions count as the farthest level from gold
            return np.array([self.idx[p] if p in self.idx else (0 if g >= K / 2 else K - 1) for g, p in zip(self.g, pred)])
        return np.array([self.idx.get(p, K) if p is not None else K for p in pred])

    def scores(self, p: np.ndarray, boots: np.ndarray | list[np.ndarray] | None = None) -> np.ndarray:
        """`boots`: rows of item indices (cluster resamples may differ in length)."""
        K, W = self.K, self.K + 1
        g = self.g
        if boots is None:
            boots = np.arange(len(g))[None, :]
        nb = len(boots)
        rows = np.repeat(np.arange(nb), [len(b) for b in boots])
        ix = np.concatenate(list(boots))
        flat = rows * K * W + g[ix] * W + p[ix]
        cm = np.bincount(flat, minlength=nb * K * W).reshape(nb, K, W).astype(float)
        n = cm.sum((1, 2))
        tp = cm[:, np.arange(K), np.arange(K)]
        row = cm.sum(axis=2)
        col = cm[:, :, :K].sum(axis=1)
        m = self.metric
        if m == "accuracy":
            return tp.sum(1) / n
        if m == "balanced_accuracy":
            present = np.isin(np.arange(K), np.unique(g))
            rec = np.divide(tp, row, out=np.zeros_like(tp), where=row > 0)
            return rec[:, present].mean(1)
        if m == "macro_f1":
            den = 2 * tp + (col - tp) + (row - tp)
            f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
            return f1.mean(1)
        if m == "micro_f1":
            pos = np.array([lab != self.negative for lab in self.labels])
            t, c, r = tp[:, pos].sum(1), col[:, pos].sum(1), row[:, pos].sum(1)
            den = 2 * t + (c - t) + (r - t)
            return np.divide(2 * t, den, out=np.zeros_like(t), where=den > 0)
        if m == "qwk":
            O = cm[:, :, :K]
            w = (np.subtract.outer(np.arange(K), np.arange(K)) ** 2) / (K - 1) ** 2
            E = np.einsum("bi,bj->bij", row, col) / n[:, None, None]
            num, den = (w * O).sum((1, 2)), (w * E).sum((1, 2))
            return 1 - np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        raise ValueError(m)


def boot_p(d: np.ndarray) -> float:
    return float(min(1.0, 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1))))


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [0.0] * len(ps)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def sign_test(xs: list[float]) -> dict[str, Any]:
    from scipy.stats import binomtest

    pos, neg = sum(x > 0 for x in xs), sum(x < 0 for x in xs)
    n = pos + neg
    return {"positive": pos, "negative": neg, "ties": len(xs) - n, "p": float(binomtest(pos, n, 0.5).pvalue) if n else 1.0}


def program_stats(rec: dict[str, Any], split: str) -> dict[str, Any]:
    ids, gold = rec[f"{split}_ids"], rec[f"{split}_gold"]
    sc = Scorer(rec["primary_metric"], rec["labels"], rec["negative_label"], gold)
    if split == "test":  # RQ1's resampling unit for the set (resample.py); validation items are resampled singly
        from resample import cluster_index, resamples

        boots = resamples(cluster_index(rec["task"], ids)[0], B, 0)
    else:
        boots = np.random.default_rng(0).integers(0, len(ids), size=(B, len(ids)))
    codes, point, reps = {}, {}, {}
    for cell, c in rec["cells"].items():
        pr = [c[split][i] for i in ids]
        codes[cell] = sc.codes(pr)
        point[cell] = float(sc.scores(codes[cell])[0])
        ref = metrics.primary(rec["primary_metric"], gold, pr, rec["labels"], rec["negative_label"])
        assert abs(point[cell] - ref) < 1e-9, (rec["program"], split, cell, point[cell], ref)
        reps[cell] = sc.scores(codes[cell], boots)
    out: dict[str, Any] = {"n": len(ids), "score": point, "contrasts": {}, "replicates": {}}
    for name, a, b, *_ in CONTRASTS + DESCRIPTIVE:
        dd = reps[a] - reps[b]
        lo, hi = np.percentile(dd, [2.5, 97.5])
        out["contrasts"][name] = {"delta": point[a] - point[b], "ci": [float(lo), float(hi)], "p": boot_p(dd)}
        out["replicates"][name] = dd
    return out


def hierarchical(progs: list[dict[str, Any]], name: str, split: str) -> dict[str, Any]:
    """Nested mean papers -> configurations -> seeds; bootstrap resamples each level, then an item replicate."""
    tree: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for r in progs:
        tree.setdefault(PAPER[r["task"]], {}).setdefault(r["task"], []).append(r)

    def nested(f) -> float:
        return float(np.mean([np.mean([np.mean([f(r) for r in seeds]) for seeds in confs.values()]) for confs in tree.values()]))

    point = nested(lambda r: r["stats"][split]["contrasts"][name]["delta"])
    rng = np.random.default_rng(0)
    papers = list(tree)
    reps = np.empty(B)
    for b in range(B):
        pv = []
        for pi in rng.integers(0, len(papers), len(papers)):
            confs = list(tree[papers[pi]].values())
            cv = []
            for ci in rng.integers(0, len(confs), len(confs)):
                seeds = confs[ci]
                sv = [seeds[si]["stats"][split]["replicates"][name][rng.integers(0, B)] for si in rng.integers(0, len(seeds), len(seeds))]
                cv.append(np.mean(sv))
            pv.append(np.mean(cv))
        reps[b] = np.mean(pv)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    by_conf = {t: float(np.mean([r["stats"][split]["contrasts"][name]["delta"] for r in rs]))
               for confs in tree.values() for t, rs in confs.items()}
    return {"delta": point, "ci": [float(lo), float(hi)], "p": boot_p(reps), "by_configuration": by_conf,
            "sign_test": sign_test(list(by_conf.values()))}


def superseded_cost() -> dict[str, float]:
    """Fresh Jev spend of first-run cells whose results were superseded by the empty-field fallback (recorded)."""
    return dict(SUPERSEDED_COST)


# Empty-field (pre-fallback) B/C runs with a test score (consumption record); TB-Dense s0's first B/C attempt failed
# on every request (placeholder key) and reran as -r2; EZ-STANCE s1's C was rejected by Jev, so it has no score.
PRE_FALLBACK = {
    "contempo-tbdense/s0": {"B": "cv-contempo-tbdense-s0-B-r2", "C": "cv-contempo-tbdense-s0-C-r2"},
    "contempo-tbdense/s1": {"B": "cv-contempo-tbdense-s1-B", "C": "cv-contempo-tbdense-s1-C"},
    "ezstance-subtaskA/s0": {"B": "cv-ezstance-subtaskA-s0-B", "C": "cv-ezstance-subtaskA-s0-C"},
    "ezstance-subtaskA/s1": {"B": "cv-ezstance-subtaskA-s1-B"},
}


def pre_fallback(progs: list[dict[str, Any]]) -> dict[str, Any]:
    """Sensitivity: the first-run B/C cells that sent emptied fields as empty strings (before the empty-field
    fallback), against the final run's A. Scores are the saved consumption records (`test_consumed.json`) of
    those runs; the fallback cells are checked against their own records. Point estimates only."""
    final = {r["program"]: r for r in progs}
    rows: dict[str, Any] = {}
    for prog, runs in PRE_FALLBACK.items():
        r = final[prog]
        sc = r["stats"]["test"]["score"]
        row = {"program": prog, "runs": runs, "score": {"A": sc["A"]}, "contrasts": {}}
        for cell, run in runs.items():
            tc = json.loads((RUNS / run / "test_consumed.json").read_text())
            assert tc["metrics"]["n"] == len(r["test_ids"]) and not tc["metrics"].get("errors"), run
            fb = json.loads((RUNS / r["cells"][cell]["run"] / "test_consumed.json").read_text())["metrics"]["primary"]
            assert abs(fb - sc[cell]) < 1e-9, (prog, cell, fb, sc[cell])
            row["score"][f"{cell}_empty"], row["score"][f"{cell}_fallback"] = tc["metrics"]["primary"], sc[cell]
        for v in ("empty", "fallback"):
            s = row["score"]
            if f"B_{v}" in s:
                row["contrasts"][f"A-B {v}"] = s["A"] - s[f"B_{v}"]
            if f"B_{v}" in s and f"C_{v}" in s:
                row["contrasts"][f"C-B {v}"] = s[f"C_{v}"] - s[f"B_{v}"]
        rows[prog] = row

    def nested(get) -> float:  # the pooled point estimate with the empty-field cells substituted where they exist
        tree: dict[str, dict[str, list[float]]] = {}
        for r in progs:
            tree.setdefault(PAPER[r["task"]], {}).setdefault(r["task"], []).append(get(r))
        return float(np.mean([np.mean([np.mean(v) for v in c.values()]) for c in tree.values()]))

    def sub(r: dict[str, Any], name: str) -> float:
        alt = rows.get(r["program"], {}).get("contrasts", {}).get(f"{name} empty")
        return alt if alt is not None else r["stats"]["test"]["contrasts"][name]["delta"]

    pooled = {name: {"fallback": nested(lambda r, n=name: r["stats"]["test"]["contrasts"][n]["delta"]),
                     "empty_where_run": nested(lambda r, n=name: sub(r, n))} for name in ("A-B", "C-B")}
    return {"programs": rows, "pooled_point": pooled}


def report() -> dict[str, Any]:
    spans = json.loads(SPANS.read_text())
    frozen = json.loads((ROOT / "results" / "frozen.json").read_text())
    fz_cal = {r["stem"]: r for r in frozen["new"] + frozen["orig"]}
    progs = []
    for p in spans["programs"]:
        rec = json.loads((CELLS / f"{p['task']}-s{p['seed']}.json").read_text())
        rec["stats"] = {s: program_stats(rec, s) for s in ("test", "val")}
        rec["n_spans"] = len(p["spans"])
        rec["span_chars"], rec["c_chars"] = p["span_chars"], p["control_C"]["chars"]
        progs.append(rec)

    pooled: dict[str, Any] = {}
    for split in ("test", "val"):
        pooled[split] = {name: hierarchical(progs, name, split) for name, *_ in CONTRASTS + DESCRIPTIVE}
        for fam in ("deletion", "insertion"):
            names = [n for n, _, _, f in CONTRASTS if f == fam]
            for n, adj in zip(names, holm([pooled[split][n]["p"] for n in names])):
                pooled[split][n]["p_holm"] = adj
        for name, *_ in CONTRASTS:
            adj = holm([r["stats"][split]["contrasts"][name]["p"] for r in progs])
            for r, a in zip(progs, adj):
                r["stats"][split]["contrasts"][name]["p_holm"] = a

    # F check against the calibration step of results/frozen.json (seed-0 programs; same fit slice)
    f_check = {}
    for r in progs:
        if r["seed"] == 0 and r["task"] in fz_cal:
            f_check[r["program"]] = {"recomputed": r["stats"]["test"]["score"]["F"], "frozen_json": fz_cal[r["task"]]["calibrated"],
                                     "zero_recomputed": r["stats"]["test"]["score"]["Z"], "zero_frozen_json": fz_cal[r["task"]]["zero"]}

    cost = {"jev_fresh": 0.0, "jev_fresh_requests": 0, "by_cell": {}}
    seen_z = set()
    for r in progs:
        for cell, c in r["cells"].items():
            for split, u in (c.get("usage") or {}).items():
                if not isinstance(u, dict):
                    continue
                for k in ("fresh_cost", "fresh_calls"):
                    if k in u:
                        cost["by_cell"].setdefault(cell, {"fresh_cost": 0.0, "fresh_calls": 0})[k] += u[k]
        if r["task"] not in seen_z:
            seen_z.add(r["task"])
            z = json.loads((CELLS / f"{r['task']}-Z.json").read_text())
            for u in z["usage"].values():
                cost["by_cell"].setdefault("Z", {"fresh_cost": 0.0, "fresh_calls": 0})["fresh_cost"] += u["fresh_cost"]
                cost["by_cell"]["Z"]["fresh_calls"] += u["fresh_calls"]
    cost["jev_fresh"] = sum(v["fresh_cost"] for v in cost["by_cell"].values())
    cost["jev_fresh_requests"] = sum(v["fresh_calls"] for v in cost["by_cell"].values())
    cost["optimizer"] = 0.0
    cost["superseded"] = superseded_cost()

    out = {
        "generated_by": "jev-opt/experiments/conventions_ablation.py",
        "bootstrap": {"B": B, "seed": 0}, "n_programs": len(progs), "n_configurations": len({r["task"] for r in progs}),
        "n_papers": len({PAPER[r["task"]] for r in progs}), "n_spans": spans["n_spans"],
        "pooled": pooled, "f_check": f_check, "cost": cost,
        "programs": [{k: v for k, v in r.items() if k not in ("cells", "val_ids", "test_ids", "val_gold", "test_gold", "fit_ids")}
                     | {"stats": {s: {k: v for k, v in st.items() if k != "replicates"} for s, st in r["stats"].items()},
                        "refit_check": r["cells"]["A"]["refit_check"],
                        "errors": {c: v.get("errors") for c, v in r["cells"].items() if v.get("errors")},
                        "calibration": {c: r["cells"][c]["calibration"] for c in ("E", "F")}}
                     for r in progs],
        "pre_fallback": pre_fallback(progs),
    }
    OUT_JSON.write_text(json.dumps(out, indent=1) + "\n")
    write_md(out)
    write_tex(out, spans)
    return out


def pct(x: float, signed: bool = False) -> str:
    return f"{100 * x:+.1f}" if signed else f"{100 * x:.1f}"


def pfmt(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def held(c: dict[str, Any]) -> str:
    return "yes" if c["ci"][0] > 0 else ("opposite" if c["ci"][1] < 0 else "no")


def write_md(out: dict[str, Any]) -> None:
    L = ["# Convention ablation", "",
         "Pre-registered (fixed with the spans before any Jev call). Generated by",
         "`jev-opt/experiments/conventions_ablation.py --report`. Values ×100, each set's primary metric. Test is primary, validation",
         "secondary. **Every test split here was used before** (zero-shot audit, v3/seed and frozen rounds); nothing is selected from",
         "these numbers. Spans: `results/conventions_spans.json`.", "",
         f"{out['n_programs']} programs, {out['n_configurations']} configurations, {out['n_papers']} papers, {out['n_spans']} spans.", "",
         "Cells: A program (readout refit on its fit slice); B A minus the convention spans (refit); C A minus equal-length",
         "non-convention text (refit); D paper's question + spans (zero-shot); E D + calibration step; F paper's question +",
         "calibration step; Z paper's question zero-shot (same path as D).", "",
         "## Pooled contrasts", "",
         "Nested mean papers → configurations → seeds; hierarchical bootstrap 95% CI (2,000 replicates, seed 0); Holm within each",
         "family; sign test over configurations (per-configuration mean over seeds). Held = CI excludes 0 in the expected direction (> 0).", "",
         "| Split | Contrast | Family | Δ | 95% CI | p | Holm p | Configurations + / − | Sign-test p | Held |",
         "|---|---|---|---:|---|---:|---:|---|---:|---|"]
    for split in ("test", "val"):
        for name, _, _, fam in CONTRASTS:
            c = out["pooled"][split][name]
            st = c["sign_test"]
            L.append(f"| {split} | {name} | {fam} | {pct(c['delta'], True)} | {pct(c['ci'][0], True)}…{pct(c['ci'][1], True)} | "
                     f"{pfmt(c['p'])} | {pfmt(c['p_holm'])} | {st['positive']} / {st['negative']} | {pfmt(st['p'])} | {held(c)} |")
    L += ["", "Descriptive (test): " + "; ".join(
        f"{n} {pct(out['pooled']['test'][n]['delta'], True)} ({pct(out['pooled']['test'][n]['ci'][0], True)}…{pct(out['pooled']['test'][n]['ci'][1], True)})"
        for n, *_ in DESCRIPTIVE) + ".", ""]
    L += ["## Per program (test)", "",
          "Paired bootstrap 95% CI on test, RQ1 resampling unit (TB-Dense, MATRES, edit intent: document; EZ-STANCE: tweet; LifeTox: query; DREsS: writing prompt; else item; 2,000 resamples, seed 0); Holm p within each contrast over the 20 programs.", "",
          "| Program | n | Spans (chars) | C chars | A | B | C | D | E | F | Z | A−B (CI, Holm p) | C−B (CI, Holm p) | D−Z (CI, Holm p) | E−F (CI, Holm p) | Refit check val/test |",
          "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|"]
    for r in out["programs"]:
        s = r["stats"]["test"]
        sc = s["score"]

        def cc(n: str) -> str:
            c = s["contrasts"][n]
            return f"{pct(c['delta'], True)} ({pct(c['ci'][0], True)}…{pct(c['ci'][1], True)}, {pfmt(c['p_holm'])})"

        rc = r["refit_check"]
        L.append(f"| {paper_tables.CONFIG_LABEL[r['task']]} s{r['seed']} | {s['n']:,} | {r['n_spans']} ({r['span_chars']}) | {r['c_chars']} | "
                 + " | ".join(pct(sc[k]) for k in "ABCDEFZ") + f" | {cc('A-B')} | {cc('C-B')} | {cc('D-Z')} | {cc('E-F')} | "
                 f"{100 * rc['val_agree']:.1f}% / {100 * rc['test_agree']:.1f}% |")
    L += ["", "## Per program (validation)", "", "| Program | n | A | B | C | D | E | F | Z | A−B | C−B | D−Z | E−F |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|"]
    for r in out["programs"]:
        s = r["stats"]["val"]
        L.append(f"| {paper_tables.CONFIG_LABEL[r['task']]} s{r['seed']} | {s['n']:,} | " + " | ".join(pct(s["score"][k]) for k in "ABCDEFZ") + " | "
                 + " | ".join(f"{pct(s['contrasts'][n]['delta'], True)} ({pct(s['contrasts'][n]['ci'][0], True)}…{pct(s['contrasts'][n]['ci'][1], True)})"
                              for n in ("A-B", "C-B", "D-Z", "E-F")) + " |")
    pf = out["pre_fallback"]
    L += ["", "## Sensitivity: empty-field B/C runs (before the fallback; test, point estimates)", "",
          "The first runs sent emptied fields as empty strings; after Jev rejected one request (EZ-STANCE s1 C) the fallback",
          "(option name / option names / node name) was applied to every program. Scores of the empty-field runs are their",
          "consumption records. The fallback reinserts option names into the deleted fields.", "",
          "| Program | A | B empty | B fallback | C empty | C fallback | A−B empty | A−B fallback | C−B empty | C−B fallback |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, r in pf["programs"].items():
        s, c = r["score"], r["contrasts"]
        g = lambda d, key, signed=False: pct(d[key], signed) if key in d else "–"  # noqa: E731
        L.append(f"| {k} | {pct(s['A'])} | {g(s, 'B_empty')} | {g(s, 'B_fallback')} | {g(s, 'C_empty')} | {g(s, 'C_fallback')} | "
                 f"{g(c, 'A-B empty', True)} | {g(c, 'A-B fallback', True)} | {g(c, 'C-B empty', True)} | {g(c, 'C-B fallback', True)} |")
    L.append(f"\nPooled point estimate with the empty-field cells substituted where they exist: A−B "
             f"{pct(pf['pooled_point']['A-B']['empty_where_run'], True)} (fallback {pct(pf['pooled_point']['A-B']['fallback'], True)}), "
             f"C−B {pct(pf['pooled_point']['C-B']['empty_where_run'], True)} (fallback {pct(pf['pooled_point']['C-B']['fallback'], True)}).")
    L += ["", "## Calibration-step check (F vs results/frozen.json, seed-0 programs, test)", "",
          "| Program | Z recomputed | Zero-shot (frozen.json) | F recomputed | Calibrated (frozen.json) |", "|---|---:|---:|---:|---:|"]
    for k, v in out["f_check"].items():
        L.append(f"| {k} | {pct(v['zero_recomputed'])} | {pct(v['zero_frozen_json'])} | {pct(v['recomputed'])} | {pct(v['frozen_json'])} |")
    c = out["cost"]
    L += ["", "## Cost", "", f"Jev (fresh requests at Jev's reported cost): ${c['jev_fresh']:.2f} over {c['jev_fresh_requests']:,} fresh requests; "
          "by cell: " + ", ".join(f"{k} ${v['fresh_cost']:.2f} ({v['fresh_calls']:,})" for k, v in sorted(c["by_cell"].items()))
          + f". Superseded first-run cells (pre-fallback B/C and the D/Z answers the final run then read from the cache): "
          f"${c['superseded']['fresh_cost']:.2f} over {c['superseded']['fresh_calls']:,} fresh requests; total "
          f"${c['jev_fresh'] + c['superseded']['fresh_cost']:.2f}. The very first attempt (tbdense s0 B/C/D) sent a placeholder API key, "
          "every request failed and no answer was obtained ($0); its records are kept. Optimizer (OMP): $0 (no call). "
          "The rubric judge for the 5 frozen programs made 5 omp calls before the pre-registration.", ""]
    OUT_MD.write_text("\n".join(L))


def write_tex(out: dict[str, Any], spans: dict[str, Any]) -> None:
    def num(x: float, signed: bool = False) -> str:
        return (f"{100 * x:+.1f}" if signed else f"{100 * x:.1f}").replace("-", "$-$")

    def ptex(p: float) -> str:
        return "<0.001" if p < 0.001 else f"{p:.3f}"

    names = {"A-B": "Del", "C-B": "Ctrl", "D-Z": "Ins", "E-F": "Cal"}
    T = ["% generated by jev-opt/experiments/conventions_ablation.py; do not edit",
         f"\\newcommand{{\\cvPrograms}}{{{out['n_programs']}}}", f"\\newcommand{{\\cvConfigs}}{{{out['n_configurations']}}}",
         f"\\newcommand{{\\cvPapers}}{{{out['n_papers']}}}", f"\\newcommand{{\\cvSpans}}{{{out['n_spans']}}}",
         f"\\newcommand{{\\cvFrozenPos}}{{{sum(p['group'] == 'frozen' for p in spans['programs'])}}}",
         f"\\newcommand{{\\cvSeedPos}}{{{sum(p['group'] == 'seeds' for p in spans['programs'])}}}",
         f"\\newcommand{{\\cvJevCost}}{{\\${out['cost']['jev_fresh']:.2f}}}",
         *(f"\\newcommand{{\\cvMatres{k}}}{{{v}}}" for p in out["programs"] if p["task"] == "contempo-matres"
           for k, v in (("SpanChars", p["span_chars"]), ("CChars", p["c_chars"])))]
    for split, sfx in (("test", ""), ("val", "Val")):
        for n, tag in names.items():
            c = out["pooled"][split][n]
            st = c["sign_test"]
            T += [f"\\newcommand{{\\cv{tag}{sfx}}}{{{num(c['delta'], True)}}}",
                  f"\\newcommand{{\\cv{tag}{sfx}Abs}}{{{num(abs(c['delta']))}}}",
                  f"\\newcommand{{\\cv{tag}{sfx}Lo}}{{{num(c['ci'][0], True)}}}",
                  f"\\newcommand{{\\cv{tag}{sfx}Hi}}{{{num(c['ci'][1], True)}}}",
                  f"\\newcommand{{\\cv{tag}{sfx}P}}{{{ptex(c['p_holm'])}}}",
                  f"\\newcommand{{\\cv{tag}{sfx}Pos}}{{{st['positive']}}}", f"\\newcommand{{\\cv{tag}{sfx}Neg}}{{{st['negative']}}}",
                  f"\\newcommand{{\\cv{tag}{sfx}SignP}}{{{ptex(st['p'])}}}"]
    for n, *_ in DESCRIPTIVE:
        c = out["pooled"]["test"][n]
        tag = {"A-C": "AC", "E-Z": "EZ", "F-Z": "FZ", "A-Z": "AZ"}[n]
        T += [f"\\newcommand{{\\cv{tag}}}{{{num(c['delta'], True)}}}", f"\\newcommand{{\\cv{tag}Lo}}{{{num(c['ci'][0], True)}}}",
              f"\\newcommand{{\\cv{tag}Hi}}{{{num(c['ci'][1], True)}}}", f"\\newcommand{{\\cv{tag}Abs}}{{{num(abs(c['delta']))}}}"]
    for n, tag in names.items():
        conf = out["pooled"]["test"][n]["by_configuration"].values()
        T += [f"\\newcommand{{\\cv{tag}ConfMin}}{{{num(min(conf), True)}}}", f"\\newcommand{{\\cv{tag}ConfMax}}{{{num(max(conf), True)}}}",
              f"\\newcommand{{\\cv{tag}ConfUnderOne}}{{{sum(0 < x < 0.01 for x in conf)}}}"]
    T.append(f"\\newcommand{{\\cvJevCostAll}}{{\\${out['cost']['jev_fresh'] + out['cost']['superseded']['fresh_cost']:.2f}}}")
    for n, tag in names.items():
        k = sum(r["stats"]["test"]["contrasts"][n]["p_holm"] < 0.05 and r["stats"]["test"]["contrasts"][n]["delta"] > 0 for r in out["programs"])
        kn = sum(r["stats"]["test"]["contrasts"][n]["p_holm"] < 0.05 and r["stats"]["test"]["contrasts"][n]["delta"] < 0 for r in out["programs"])
        T += [f"\\newcommand{{\\cv{tag}SigPos}}{{{k}}}", f"\\newcommand{{\\cv{tag}SigNeg}}{{{kn}}}"]
    pf, pp = out["pre_fallback"]["programs"], out["pre_fallback"]["pooled_point"]
    ez, tb = pf["ezstance-subtaskA/s0"], pf["contempo-tbdense/s0"]
    T += [f"\\newcommand{{\\cvPreN}}{{{len(pf)}}}",
          f"\\newcommand{{\\cvPreDel}}{{{num(pp['A-B']['empty_where_run'], True)}}}",
          f"\\newcommand{{\\cvPreCtrl}}{{{num(pp['C-B']['empty_where_run'], True)}}}",
          f"\\newcommand{{\\cvPreEzB}}{{{num(ez['score']['B_empty'])}}}", f"\\newcommand{{\\cvPreEzBFb}}{{{num(ez['score']['B_fallback'])}}}",
          f"\\newcommand{{\\cvPreEzDel}}{{{num(ez['contrasts']['A-B empty'], True)}}}",
          f"\\newcommand{{\\cvPreEzDelFb}}{{{num(ez['contrasts']['A-B fallback'], True)}}}",
          f"\\newcommand{{\\cvPreTbdC}}{{{num(tb['score']['C_empty'])}}}", f"\\newcommand{{\\cvPreTbdCFb}}{{{num(tb['score']['C_fallback'])}}}",
          f"\\newcommand{{\\cvPreTbdDel}}{{{num(tb['contrasts']['A-B empty'], True)}}}",
          f"\\newcommand{{\\cvPreTbdDelFb}}{{{num(tb['contrasts']['A-B fallback'], True)}}}"]
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(T) + "\n")

    A = ["% generated by jev-opt/experiments/conventions_ablation.py; do not edit",
         "\\begin{tabular}{@{}l c c c c c c c c c c c c c c@{}}", "\\toprule",
         "Program & " + " & ".join(f"\\multicolumn{{1}}{{c}}{{{h}}}" for h in ("$n$", "Spans", "A", "B", "C", "D", "E", "F", "Z", "A$-$B", "C$-$B", "D$-$Z", "E$-$F", "Refit")) + " \\\\", "\\midrule"]

    def ci(c: dict[str, Any]) -> str:
        star = "$^{*}$" if c["p_holm"] < 0.05 else ""
        return f"{num(c['delta'], True)}{star}"

    for r in out["programs"]:
        s = r["stats"]["test"]
        A.append(f"{paper_tables.CONFIG_LABEL[r['task']]} s{r['seed']} & {s['n']:,} & {r['n_spans']} & ".replace(",", "{,}")
                 + " & ".join(num(s["score"][k]) for k in "ABCDEFZ") + " & "
                 + " & ".join(ci(s["contrasts"][n]) for n in ("A-B", "C-B", "D-Z", "E-F"))
                 + f" & {100 * r['refit_check']['test_agree']:.0f}\\% \\\\")
    A += ["\\midrule"]
    for split, lab in (("test", "Pooled, test"), ("val", "Pooled, validation")):
        pl = out["pooled"][split]
        A.append(f"\\multicolumn{{10}}{{@{{}}l}}{{{lab}: $\\Delta$ [95\\% CI]}} & " + " & ".join(
            f"{num(pl[n]['delta'], True)} [{num(pl[n]['ci'][0], True)}, {num(pl[n]['ci'][1], True)}]"
            for n in ("A-B", "C-B", "D-Z", "E-F")) + " & \\\\")
        A.append(f"\\multicolumn{{10}}{{@{{}}l}}{{{lab}: Holm $p$}} & " + " & ".join(
            ptex(pl[n]["p_holm"]) for n in ("A-B", "C-B", "D-Z", "E-F")) + " & \\\\")
    A.append("\\multicolumn{10}{@{}l}{Configurations with $\\Delta>0$ / $<0$ (sign-test $p$), test} & " + " & ".join(
        f"{out['pooled']['test'][n]['sign_test']['positive']} / {out['pooled']['test'][n]['sign_test']['negative']} ({ptex(out['pooled']['test'][n]['sign_test']['p'])})"
        for n in ("A-B", "C-B", "D-Z", "E-F")) + " & \\\\")
    pp = out["pre_fallback"]["pooled_point"]
    A.append(f"\\multicolumn{{10}}{{@{{}}l}}{{Pooled, test, empty fields on the {len(out['pre_fallback']['programs'])} programs run so: $\\Delta$}} & "
             f"{num(pp['A-B']['empty_where_run'], True)} & {num(pp['C-B']['empty_where_run'], True)} & & & \\\\")
    A += ["\\bottomrule", "\\end{tabular}"]
    OUT_APP.write_text("\n".join(A) + "\n")

    def esc(t: str) -> str:
        table = {"\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#", "_": "\\_", "{": "\\{", "}": "\\}",
                 "~": "\\textasciitilde{}", "^": "\\textasciicircum{}", "<": "\\textless{}", ">": "\\textgreater{}",
                 "`": "\\textasciigrave{}", '"': "\\textquotedbl{}"}
        return "".join(table.get(ch, ch) for ch in t)

    def field(f: str) -> str:
        return f.split(":", 1)[1] if f.startswith("option:") else ("level " + f.split(":", 1)[1] if f.startswith("level:") else f)

    S = ["% generated by jev-opt/experiments/conventions_ablation.py; do not edit",
         "% one paragraph per program: node, field (option name or instructions), the span deleted in B;",
         "% a dagger marks spans that D does not append (they only point back to a rule stated elsewhere)",
         "{\\footnotesize"]
    dagger = "$^{\\dagger}$"
    for p in spans["programs"]:
        items = [f"\\texttt{{{esc(s['node'])}}}/{esc(field(s['field']))}: ``{esc(s['span'].strip())}''{'' if s['append_in_D'] else dagger}"
                 for s in p["spans"]]
        S.append(f"\\noindent\\textbf{{{paper_tables.CONFIG_LABEL[p['task']]} s{p['seed']}}} ({len(items)}; C deletes {p['control_C']['chars']} of "
                 f"{p['span_chars']} characters). " + "; ".join(items) + ".\\par\\smallskip")
    S += ["}"]
    OUT_APP_SPANS.write_text("\n".join(S) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if a.run:
        run_all(a.only)
    if a.report:
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
