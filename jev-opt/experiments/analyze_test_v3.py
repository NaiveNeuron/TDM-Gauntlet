# /// script
# dependencies = ["numpy", "pyyaml"]
# ///
"""Test-round analysis across v1/v2/v3/tailored and write results/GEPA_v3_test.md.

Re-executes, offline from each run's own Jev cache (posting disabled, so a cache miss is an error and no
second test query can happen), the representative that consumed each run's test split, checks the
recomputed primary metric against the value recorded at consumption, and reports paired bootstrap 95%
CIs (2,000 resamples, seed 0, drawn in the RQ1 resampling unit of the set; resample.py) against the v1
zero-shot run on the same test examples.

Run from the repository root (Python 3.11):
  uv run --project jev-opt/python python jev-opt/experiments/analyze_test_v3.py
Run directories are read from this repository's runs/jev-opt by default. --v3-root and --tailored-root name
other directories that hold those runs, because each root holds the Jev response cache and embedding cache
its runs used.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENROUTER_API_KEY", "offline-cache-only")  # never used: posting is disabled below

import numpy as np  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
from resample import cluster_index, resamples  # noqa: E402

assert sys.version_info[:2] == (3, 11), f"run under Python 3.11 (worker's version), not {sys.version}"

from jev_opt import metrics  # noqa: E402
from jev_opt.execute import run_decider  # noqa: E402
from jev_opt.hygiene import allow_test_split, load_split  # noqa: E402
from jev_opt.ir import load_decider  # noqa: E402
from jev_opt.jev import JevClient, make_client  # noqa: E402


class CacheMiss(RuntimeError):
    pass


def _no_network(self, body):
    raise CacheMiss("analysis must be served from the Jev cache; a request was not cached")


JevClient._post = _no_network
from jev_opt.runs import _load_run_config, make_pool  # noqa: E402

ROOT = Path.cwd()
OUT = ROOT / "results" / "GEPA_v3_test.md"
B = 2000
TASKS = [  # slug, title, metric, paper_numbers key
    ("contempo-tbdense", "ConTempo TB-Dense", "micro-F1 (VAGUE neg.)", "contempo/tbdense"),
    ("ezstance-subtaskA", "EZ-STANCE Subtask A", "macro-F1", "ezstance/subtaskA"),
    ("smarter-implicit-hate", "SMARTER Implicit Hate", "macro-F1", "smarter/implicit_hate"),
    ("edit-intent", "Edit intent (Re3-Sci)", "macro-F1", "edit_intent"),
    ("inli", "INLI", "accuracy", "inli"),
    ("dress-new-content", "DREsS content", "QWK", "dress_new"),
    ("dress-new-organization", "DREsS organization", "QWK", "dress_new"),
    ("dress-new-language", "DREsS language", "QWK", "dress_new"),
]
# Paper numbers as used in results/GEPA.md (analyze.py): task-specific method, zero-shot LLM baseline.
PAPER = {
    "contempo-tbdense": (0.6856, 0.6758), "ezstance-subtaskA": (0.812, 0.499), "smarter-implicit-hate": (0.60, 0.58),
    "edit-intent": (0.843, 0.37), "inli": (0.924, 0.770), "dress-new-content": (0.631, 0.31),
    "dress-new-organization": (0.771, 0.322), "dress-new-language": (0.589, 0.231),
}
DISCREPANCIES: list[tuple[str, float, float]] = []


def rows_for(slug: str, roots: dict[str, Path]) -> list[tuple[str, Path]]:
    m, v3, t = roots["main"] / "runs/jev-opt", roots["v3"] / "runs/jev-opt", roots["tailored"] / "runs/jev-opt"
    return [
        ("v1 zero-shot", m / f"study-{slug}-zero"),
        ("v1 text-opt", m / f"study-{slug}-text"),
        ("v1 program-opt", m / f"study-{slug}-program"),
        ("v2 chosen", m / f"v2-{slug}-CD"),
        ("**v3 chosen (headline)**", v3 / f"v3-{slug}-P"),
        ("Jev-native zero-shot (v3 N)", v3 / f"v3-{slug}-N"),
        ("tailored", t / f"tailored-{slug}"),
    ]


def evaluate(run_dir: Path) -> dict:
    consumed = json.loads((run_dir / "test_consumed.json").read_text())
    config = _load_run_config(run_dir)
    program = load_decider((run_dir / f"{consumed['representative']}.yaml").read_text())
    with allow_test_split():
        test = load_split(config, "test")
    train = load_split(config, "train")
    cache_dir = Path(config.jev.cache_dir) if config.jev.cache_dir else run_dir.parent / ".jev-cache"
    client = make_client(cache_dir, config.jev.model, config.jev.max_rpm, config.jev.concurrency, [config.base_dir, run_dir.parents[2]])
    try:
        pool = make_pool(config, train, run_dir.parent)
        results = run_decider(program, test, client, context={"split": "test", "condition": "analysis"}, pool=pool)
    finally:
        client.close()
    gold, pred = [r.gold for r in results], [r.pred for r in results]
    m = metrics.compute(gold, pred, [r.dist for r in results], config.labels, config.primary_metric, config.negative_label)
    rec = consumed["metrics"]
    if abs(m["primary"] - rec["primary"]) > 1e-9:
        DISCREPANCIES.append((run_dir.name, rec["primary"], m["primary"]))
    return {"ids": [e.id for e in test], "gold": gold, "pred": pred, "primary": m["primary"], "config": config,
            "rep": consumed["representative"], "candidate": consumed["candidate"], "recorded": rec["primary"],
            "calls": rec.get("mean_calls"), "tokens": rec.get("mean_input_tokens"), "cost": rec.get("total_cost"), "n": rec["n"]}


def primary_on(idx, r) -> float:
    c = r["config"]
    return metrics.primary(c.primary_metric, [r["gold"][i] for i in idx], [r["pred"][i] for i in idx], c.labels, c.negative_label)


def fmt(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def gauntlet_score(task: str) -> tuple[float | None, dict]:
    p = ROOT / "runs" / task / "scores.json"
    if not p.exists():
        return None, {}
    s = json.loads(p.read_text())
    man = json.loads((ROOT / "runs" / task / "manifest.json").read_text()) if (ROOT / "runs" / task / "manifest.json").exists() else {}
    return s, man


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3-root", default=str(ROOT))
    ap.add_argument("--tailored-root", default=str(ROOT))
    a = ap.parse_args()
    roots = {"main": ROOT, "v3": Path(a.v3_root), "tailored": Path(a.tailored_root)}
    paper_json = json.loads((ROOT / "results" / "paper_numbers.json").read_text())
    out: dict = {"tasks": {}}
    lines = [
        "# v3 test round: results",
        "",
        "Pre-registered: third pre-registration (headline), second pre-registration (v2 chosen), Jev-adapted",
        "GEPA (tailored), candidate reranking and fitted thresholds (lever C). Generated by",
        "`jev-opt/experiments/analyze_test_v3.py`, which re-executes every test-evaluated representative offline",
        "from its run's Jev cache (posting disabled) and checks the recomputed score against the one recorded at",
        "consumption (discrepancies listed below). Every run's test split was consumed exactly once.",
        "",
        "**Headline = v3 chosen** (the third pre-registration's choice, val only). All other rows are",
        "ablations/references. v1 text/program are both shown (v1 had no pre-registered single choice). Values ×100;",
        "Δ v1 zero / Δ v2 chosen = paired bootstrap difference to that run on the same test examples",
        "(2,000 resamples, seed 0, 95% percentile CI). Jev $ / tokens = test evaluation only (`test_consumed.json`).",
        "`tailored-zero` (`top-tailored-zero`) was not evaluated: the jev-opt ledger allows one test consumption",
        "per run (`evaluate_final` refuses a second representative of a consumed run), and `top-tailored` was",
        "consumed first.",
        "",
    ]
    summary = []
    for slug, title, metric_name, pkey in TASKS:
        res = {}
        for label, d in rows_for(slug, roots):
            if (d / "test_consumed.json").exists():
                res[label] = (d, evaluate(d))
        base = res.get("v1 zero-shot")
        v2 = res.get("v2 chosen")
        paper, llm = PAPER[slug]
        pj = paper_json.get(pkey, {})
        lines += [f"## {title} ({metric_name}, test n={base[1]['n'] if base else '?'})", "",
                  "| Row | Run / representative | Test | Δ v1 zero (95% CI) | Δ v2 chosen (95% CI) | Jev calls/ex | input tok/ex | Jev $ test |",
                  "|---|---|---:|---|---|---:|---:|---:|"]
        entry = {}
        inv, _unit = cluster_index(slug, base[1]["ids"])  # RQ1's resampling unit for the set (resample.py)
        boots = resamples(inv, B, 0)
        per_boot = {label: np.array([primary_on(b, r) for b in boots]) for label, (_, r) in res.items()}

        def delta(label: str, ref: str, r: dict, rref: dict) -> str:
            if ref not in res or label == ref:
                return ""
            lo, hi = np.percentile(per_boot[label] - per_boot[ref], [2.5, 97.5])
            entry.setdefault(label, {})[f"ci_vs_{ref}"] = [float(lo), float(hi)]
            return f"{100 * (r['primary'] - rref['primary']):+.1f} ({100 * lo:+.1f}…{100 * hi:+.1f})"

        for label, (d, r) in res.items():
            assert r["ids"] == base[1]["ids"] and r["gold"] == base[1]["gold"], f"{d.name}: test set differs from v1"
            d1 = delta(label, "v1 zero-shot", r, base[1])
            d2 = delta(label, "v2 chosen", r, v2[1]) if v2 else ""
            entry.setdefault(label, {}).update({"run": d.name, "rep": r["rep"], "candidate": r["candidate"], "test": r["primary"],
                                                "calls": r["calls"], "tokens": r["tokens"], "cost": r["cost"]})
            lines.append(f"| {label} | `{d.name}` {r['rep']} = {r['candidate']} | **{fmt(r['primary'])}** | {d1} | {d2} | "
                         f"{r['calls']:.2f} | {r['tokens']:.0f} | ${r['cost']:.2f} |")
        lines += [f"| paper task-specific ({pj.get('method', 'paper')}) | results/paper_numbers.json | {fmt(paper)} | | | | | |",
                  f"| paper zero-shot LLM baseline | (as in `results/GEPA.md`) | {fmt(llm)} | | | | | |", ""]
        out["tasks"][slug] = {"rows": entry, "paper": paper, "paper_llm": llm}
        best = max(res.items(), key=lambda kv: kv[1][1]["primary"])
        h = res["**v3 chosen (headline)**"][1]["primary"]
        summary.append(f"| {title} | {fmt(h)} | {fmt(best[1][1]['primary'])} ({best[0].strip('*')}) | {fmt(paper)} | {100 * (h - paper):+.1f} | {fmt(llm)} |")
    lines += ["## Summary: headline vs paper", "",
              "Best row = highest test score among all rows above (post hoc, not a selection).", "",
              "| Task | v3 headline | Best row (post hoc) | Paper task-specific | Headline − paper | Paper zero-shot LLM |",
              "|---|---:|---:|---:|---:|---:|", *summary, ""]
    # Lever C
    lines += ["## Lever C (gauntlet; candidate reranking and fitted thresholds)", "",
              "| Task | Condition | v3 test | v1/v2 test (earlier run) | Paper method | New Jev $ (excl. reused) |", "|---|---|---:|---:|---:|---:|"]
    prev = {"ttp_v3/tram": ("ttp/tram", 0.440), "ttp_v3/procedures_plus": ("ttp/procedures_plus", 0.459),
            "ttp_v3/expert": ("ttp/expert", 0.548), "ttp_v3/expert_full": ("ttp/expert_full", 0.390),
            "relemb_v3/wikiwd": ("relemb/wikiwd", 0.541), "made_v3": ("made", 0.329)}
    metric_key = {"ttp": "p@1", "relemb": "f1", "made": "macro_f1"}
    out["lever_c"] = {}
    for task, (ptask, prev_v) in prev.items():
        s, man = gauntlet_score(task)
        mk = metric_key[ptask.split("/")[0]]
        v = None
        if s:
            v = s.get(mk, s.get("metrics", {}).get(mk) if isinstance(s.get("metrics"), dict) else None)
        pj = paper_json.get(ptask, {})
        pv = pj.get("paper", {}).get(pj.get("method")) if pj else None
        cost = man.get("cost_usd_excluding_reused")
        out["lever_c"][task] = {"test": v, "prev": prev_v, "paper": pv, "cost": cost}
        lines.append(f"| `{task}` | {mk} | **{fmt(v)}** | {fmt(prev_v)} | {fmt(pv)} ({pj.get('method', '')}) | "
                     f"{'–' if cost is None else f'${cost:.2f}'} |")
    # LifeTox pilot
    lt = ROOT / "runs/jev-opt/lifetox-real-1"
    lines += ["", "## LifeTox pilot (`lifetox-real-1`, not pre-registered as a study; a single test run)", ""]
    if (lt / "test_consumed.json").exists():
        r = evaluate(lt)
        pj = paper_json["lifetox"]["paper"]
        out["lifetox"] = {"test": r["primary"], "rep": r["rep"], "candidate": r["candidate"], "cost": r["cost"]}
        lines += ["| Row | Test macro-F1 | Jev calls/ex | Jev $ test |", "|---|---:|---:|---:|",
                  f"| Jev `{r['rep']}` = {r['candidate']} | **{fmt(r['primary'])}** | {r['calls']:.2f} | ${r['cost']:.2f} |",
                  f"| paper RoBERTa-LifeTox 350M | {fmt(pj['RoBERTa-LifeTox 350M'])} | | |",
                  f"| paper GPT-3.5 text-davinci-003 zero-shot | {fmt(pj['GPT-3.5 text-davinci-003 zero-shot'])} | | |"]
    lines += ["", "## Re-execution discrepancies", ""]
    lines += [f"- `{n}`: recorded {fmt(a)}, re-executed {fmt(b)}" for n, a, b in DISCREPANCIES] or ["None: every re-executed score equals the recorded one."]
    OUT.write_text("\n".join(lines) + "\n")
    (OUT.parent / "gepa_v3_test.json").write_text(json.dumps({**out, "discrepancies": DISCREPANCIES}, indent=2) + "\n")
    print(OUT.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
