"""Questions as features: how the headline method's winning programs use Jev's answers -> results/framing_features.json.

For each of the 8 study tasks and optimizer seeds 0/1/2 (the programs chosen by the seed-variance rule, listed in
results/seeds.json["selection"]), offline from the Jev cache (posting disabled: a cache miss raises, no Jev call):

  (a) evidence share on test: the share of test items where the method's (readout's) prediction is correct while
      Jev's top answer on the program's main question is not the gold label. Main question = the question node
      that carries the readout for the item's slot; its top answer = argmax option (Choice; option -> label),
      argmax Noul probability (one-vs-rest), argmax level (Score; level i -> label i when the levels are the
      labels, else the level index rescaled to the label index and rounded), none (a Noul main question that
      names no label: its top answer is never the gold label). Also against the audit question: the v1
      zero-shot run's test prediction (`study-<task>-zero`, the paper's own question, argmax decoding).
  (b) program anatomy: auxiliary questions (`<slot>_a_*` views), a main-question type differing from the paper
      question's (seed.yaml), one-vs-rest Noul questions among the views, stacked variants (`<slot>_x*` views).
  (c) drop-aux ablation for programs with auxiliary questions: the aux nodes removed from every readout's views,
      the readout refit on the run's train fit slice exactly as the run fit it (fit_program with the run's
      optimizer settings, answers from the cache), scored on val and test; paired bootstrap of the difference
      (2,000 resamples, seed 0, 95% percentile CI). The refit procedure is validated by refitting the full
      program the same way and checking its val and test predictions against the recorded ones. Post hoc: no
      choice is made from these numbers.

Run from the repository root (Python 3.11):
  uv run --project jev-opt/python python jev-opt/experiments/framing_features.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENROUTER_API_KEY", "offline-cache-only")  # never used: posting is disabled below

import numpy as np  # noqa: E402

assert sys.version_info[:2] == (3, 11), f"run under Python 3.11 (worker's version), not {sys.version}"

from jev_opt import metrics  # noqa: E402
from jev_opt.adapter import cv_spec, train_roles  # noqa: E402
from jev_opt.execute import ExampleResult, _normalize, one_vs_rest_decode, run_decider, run_examples  # noqa: E402
from jev_opt.fit import fit_program  # noqa: E402
from jev_opt.hygiene import allow_test_split, load_split  # noqa: E402
from jev_opt.ir import BranchNode, ChoiceNode, OneVsRestNode, Program, Readout, ScoreNode, load_decider  # noqa: E402
from jev_opt.jev import JevClient, make_client  # noqa: E402
from jev_opt.runs import _load_run_config, make_pool  # noqa: E402


class CacheMiss(RuntimeError):
    pass


def _no_network(self, body):
    raise CacheMiss("analysis must be served from the Jev cache; a request was not cached")


JevClient._post = _no_network

ROOT = Path.cwd()
RUNS = ROOT / "runs" / "jev-opt"
OUT = ROOT / "results" / "framing_features.json"
SEEDS = (0, 1, 2)
B = 2000
TASKS = [
    ("contempo-tbdense", "ConTempo TB-Dense"), ("ezstance-subtaskA", "EZ-STANCE Subtask A"),
    ("smarter-implicit-hate", "SMARTER Implicit Hate"), ("edit-intent", "Edit intent (Re3-Sci)"), ("inli", "INLI"),
    ("dress-new-content", "DREsS content"), ("dress-new-organization", "DREsS organization"),
    ("dress-new-language", "DREsS language"),
]


class Env:
    """Config, splits, cache-only client and demo pool of one run directory."""

    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.config = _load_run_config(run_dir)
        self.train = load_split(self.config, "train")
        self.val = load_split(self.config, "val")
        with allow_test_split():
            self.test = load_split(self.config, "test")
        c = self.config
        cache_dir = Path(c.jev.cache_dir) if c.jev.cache_dir else run_dir.parent / ".jev-cache"
        self.client = make_client(cache_dir, c.jev.model, c.jev.max_rpm, c.jev.concurrency, [c.base_dir, run_dir.parents[2]])
        self.pool = make_pool(c, self.train, run_dir.parent)

    def run(self, program: Program, examples: list, split: str, exhaustive: bool = False) -> list[ExampleResult]:
        ctx = {"split": split, "condition": "analysis"}
        if exhaustive:
            return run_examples(program, examples, self.client, exhaustive=True, context=ctx, pool=self.pool)
        return run_decider(program, examples, self.client, context=ctx, pool=self.pool)

    def primary(self, gold: list[str], pred: list[str | None]) -> float:
        c = self.config
        return metrics.primary(c.primary_metric, gold, pred, c.labels, c.negative_label)

    def close(self) -> None:
        self.client.close()


def readout_slots(p: Program) -> list[str]:
    return [nid for nid, n in p.nodes.items() if getattr(n, "readout", None) is not None]


def main_top(p: Program, nid: str, memo: dict[str, Any]) -> str | None:
    """Jev's top answer on main question `nid` as a task label (None: the question names no label)."""
    node = p.nodes[nid]
    keys = [k for k in memo if k.split("|", 1)[0] == nid]
    if len(keys) != 1:
        raise ValueError(f"{nid}: expected one recorded state, got {len(keys)}")
    ans = memo[keys[0]]
    if isinstance(node, ChoiceNode):
        probs = _normalize({o: float(ans[nid].get("probabilities", {}).get(o, 0.0)) for o in node.options})
        best = max(probs, key=lambda o: probs[o])  # ties: earliest option (as the executor)
        return node.options[best].target(best).label
    if isinstance(node, OneVsRestNode):
        return one_vs_rest_decode(node, {o: float(ans[f"{nid}__{o}"]["noul"]) for o in node.options})[1]
    if isinstance(node, ScoreNode):
        raw = ans[nid].get("probabilities", {})
        L = len(node.levels)
        probs = [float(raw.get(str(i), 0.0)) for i in range(L)]
        i = int(np.argmax(probs))
        classes = node.readout.classes
        return classes[i] if L == len(classes) else classes[int(round(i * (len(classes) - 1) / (L - 1)))]
    if isinstance(node, BranchNode):
        return None
    raise TypeError(f"{nid}: {node.kind}")


def reached_slot(p: Program, r: ExampleResult) -> str:
    hit = [nid for nid in readout_slots(p) if "readout" in r.trace.get(nid, {})]
    if len(hit) != 1:
        raise ValueError(f"{r.id}: readout slots reached {hit}")
    return hit[0]


def anatomy(p: Program, seed: Program) -> dict[str, Any]:
    slots = readout_slots(p)
    views = {s: list(p.nodes[s].readout.views) for s in slots}
    allv = [v for vs in views.values() for v in vs]
    aux = [v for v in allv if "_a_" in v]
    return {
        "slots": {s: {"main_kind": p.nodes[s].kind, "paper_kind": seed.nodes[s].kind, "views": views[s],
                      "view_kinds": [p.nodes[v].kind for v in views[s]]} for s in slots},
        "aux": bool(aux), "n_aux": len(aux),
        "type_differs": any(p.nodes[s].kind != seed.nodes[s].kind for s in slots),
        "one_vs_rest": any(p.nodes[v].kind == "one_vs_rest" for v in allv),
        "stacked": any("_x" in v.split("_a_")[0][len(s):] for s in slots for v in views[s]),
        "n_questions": len(allv),
        "demos": p.demos is not None,
    }


def unfitted(p: Program, drop_aux: bool) -> Program:
    """Copy with every readout reset to unfitted (and, with drop_aux, the auxiliary questions removed)."""
    q = p.model_copy(deep=True)
    for s in readout_slots(q):
        r = q.nodes[s].readout
        views = [v for v in r.views if not (drop_aux and "_a_" in v)]
        q.nodes[s].readout = Readout(kind=r.kind, views=views, classes=list(r.classes))
    if drop_aux:
        for nid in [n for n in q.nodes if "_a_" in n]:
            del q.nodes[nid]
        if q.demos is not None:
            nodes = [n for n in q.demos.nodes if n in q.nodes]
            q.demos = q.demos.model_copy(update={"nodes": nodes}) if nodes else None
    q.temperature = 1.0
    return q


def refit(env: Env, program: Program) -> Program:
    opt = env.config.optimizer
    fit_set, _ = train_roles(env.train, opt)
    raw = env.run(program, fit_set, "train", exhaustive=True)
    fitted, _ = fit_program(program, list(zip(fit_set, raw)), env.config.primary_metric, opt.calibrate,
                            rounds=2 if opt.fit_parameters else 0, negative=env.config.negative_label,
                            class_bias=opt.class_bias, cv=cv_spec(opt))
    return fitted.check(env.config)


def paired(env: Env, gold: list[str], a: list, b: list) -> dict[str, Any]:
    """Score of b minus a with a paired percentile bootstrap (B resamples, seed 0)."""
    n = len(gold)
    boots = np.random.default_rng(0).integers(0, n, size=(B, n))
    d = [env.primary([gold[i] for i in ix], [b[i] for i in ix]) - env.primary([gold[i] for i in ix], [a[i] for i in ix]) for ix in boots]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"a": env.primary(gold, a), "b": env.primary(gold, b), "delta": env.primary(gold, b) - env.primary(gold, a), "ci": [float(lo), float(hi)]}


def audit_predictions(slug: str) -> dict[str, str | None]:
    d = RUNS / f"study-{slug}-zero"
    env = Env(d)
    try:
        consumed = json.loads((d / "test_consumed.json").read_text())
        program = load_decider((d / f"{consumed['representative']}.yaml").read_text())
        res = env.run(program, env.test, "test")
        got = env.primary([r.gold for r in res], [r.pred for r in res])
        if abs(got - consumed["metrics"]["primary"]) > 1e-9:
            raise SystemExit(f"{d.name}: re-executed {got} != recorded {consumed['metrics']['primary']}")
        return {r.id: r.pred for r in res}
    finally:
        env.close()


def analyse_run(slug: str, seed: int, sel: dict[str, Any], audit: dict[str, str | None]) -> dict[str, Any]:
    d = RUNS / sel["run"]
    consumed = json.loads((d / "test_consumed.json").read_text())
    rep = sel["representative"]
    assert consumed["representative"] == rep, f"{d.name}: test consumed by {consumed['representative']}, selection {rep}"
    program = load_decider((d / f"{rep}.yaml").read_text())
    seed_prog = load_decider((d / "seed.yaml").read_text())
    env = Env(d)
    try:
        res = env.run(program, env.test, "test")
        gold, pred = [r.gold for r in res], [r.pred for r in res]
        test_score = env.primary(gold, pred)
        if abs(test_score - consumed["metrics"]["primary"]) > 1e-9:
            raise SystemExit(f"{d.name}: re-executed test {test_score} != recorded {consumed['metrics']['primary']}")
        tops = [main_top(program, reached_slot(program, r), r.memo) for r in res]
        n = len(res)
        ok = [p == g for p, g in zip(pred, gold)]
        top_ok = [t == g for t, g in zip(tops, gold)]
        aud_ok = [audit[r.id] == r.gold for r in res]
        out: dict[str, Any] = {
            "run": d.name, "representative": rep, "candidate": consumed["candidate"], "n_test": n, "test": test_score,
            "acc_method": sum(ok) / n, "acc_main_top": sum(top_ok) / n, "acc_audit": sum(aud_ok) / n,
            "evidence_share": sum(o and not t for o, t in zip(ok, top_ok)) / n,
            "lost_share": sum(t and not o for o, t in zip(ok, top_ok)) / n,
            "changed_share": sum(p != t for p, t in zip(pred, tops)) / n,
            "evidence_share_vs_audit": sum(o and not a for o, a in zip(ok, aud_ok)) / n,
            "lost_share_vs_audit": sum(a and not o for o, a in zip(ok, aud_ok)) / n,
            "main_names_no_label": any(isinstance(program.nodes[s], BranchNode) for s in readout_slots(program)),
            "anatomy": anatomy(program, seed_prog),
        }
        if out["anatomy"]["aux"]:
            vp = json.loads((d / "val_predictions.json").read_text())
            val_gold = [ex.label for ex in env.val]
            rec_val = [vp["representatives"][rep][ex.id] for ex in env.val]
            full = refit(env, unfitted(program, drop_aux=False))
            full_val = [r.pred for r in env.run(full, env.val, "val")]
            full_test = [r.pred for r in env.run(full, env.test, "test")]
            drop = refit(env, unfitted(program, drop_aux=True))
            drop_val = [r.pred for r in env.run(drop, env.val, "val")]
            drop_test = [r.pred for r in env.run(drop, env.test, "test")]
            out["drop_aux"] = {
                "refit_check": {"val_agree": sum(a == b for a, b in zip(full_val, rec_val)) / len(rec_val),
                                "test_agree": sum(a == b for a, b in zip(full_test, pred)) / n},
                "val": paired(env, val_gold, rec_val, drop_val),
                "test": paired(env, gold, pred, drop_test),
                "dropped_nodes": [nid for nid in program.nodes if "_a_" in nid],
            }
        return out
    finally:
        env.close()


def main() -> int:
    selection = json.loads((ROOT / "results" / "seeds.json").read_text())["selection"]
    out: dict[str, Any] = {"generated_by": "jev-opt/experiments/framing_features.py", "bootstrap": {"B": B, "seed": 0}, "tasks": {}}
    for slug, title in TASKS:
        audit = audit_predictions(slug)
        runs = {}
        for seed in SEEDS:
            runs[str(seed)] = analyse_run(slug, seed, selection[slug][str(seed)], audit)
            r = runs[str(seed)]
            print(f"{slug} s{seed}: test {100 * r['test']:.1f} evidence {100 * r['evidence_share']:.1f}% "
                  f"(vs audit {100 * r['evidence_share_vs_audit']:.1f}%) aux={r['anatomy']['aux']}"
                  + (f" drop-aux val {100 * r['drop_aux']['val']['delta']:+.1f} test {100 * r['drop_aux']['test']['delta']:+.1f}"
                     f" refit agree {r['drop_aux']['refit_check']}" if "drop_aux" in r else ""), flush=True)
        mean = lambda k: float(np.mean([runs[s][k] for s in runs]))  # noqa: E731
        out["tasks"][slug] = {"title": title, "seeds": runs, "evidence_share_mean": mean("evidence_share"),
                              "evidence_share_vs_audit_mean": mean("evidence_share_vs_audit"), "lost_share_mean": mean("lost_share")}
    progs = [r for t in out["tasks"].values() for r in t["seeds"].values()]
    out["anatomy_counts"] = {k: sum(r["anatomy"][k] for r in progs) for k in ("aux", "type_differs", "one_vs_rest", "stacked", "demos")}
    out["anatomy_counts"]["total"] = len(progs)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
