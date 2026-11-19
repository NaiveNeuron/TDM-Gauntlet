"""Optimization-study appendix blocks (app:opt), registered with paper_appendix.

Every value is computed here from the run directories and result files; scripts/appendix/opt.tex holds prose only.

Rounds and their sources:

  v1        runs/jev-opt/study-<task>-{zero,text,program}/ (24 runs) + results/gepa.json (analyze.py)
  v2        runs/jev-opt/v2-*/ + runs/jev-opt/v2-dev-summary.json (analyze_v2.py)
  v3        runs/jev-opt/v3-<task>-{A,P,N}/ + v3-dev-summary.json (analyze_v3.py) + results/GEPA_v3_val.md
            (per-seed zero-shot table) + jev-opt/experiments/<task>/seeds_v3*/
  tailored  runs/jev-opt/tailored-<task>/ + tailored-dev-summary.json (analyze_tailored.py)
  test      results/gepa_v3_test.json (analyze_test_v3.py: every tested row, paired-bootstrap CIs, re-execution),
            runs/jev-opt/test_ledger.jsonl, each run's test_consumed.json
  lever C   results/v3_dev/*.json (scripts/fit_v3.py), data/frozen/{ttp,relemb,made}/*_params.json,
            runs/{ttp_v3,relemb_v3,made_v3}/**/{scores,manifest}.json, runs/dev_v3/*/manifest.json
  seeds     runs/jev-opt/v3-<task>-P-s{1,2}/ + results/seeds.json (analyze_seeds.py); seed 0 is
            the v3 P run, cross-checked against results/gepa_v3_test.json and v3-dev-summary.json
  heads     results/heads.json (scripts/report_heads.py), results/v3_dev/heads.json (scripts/fit_v3.py heads),
            data/frozen/heads/**, data/frozen/v3_dev_manifest.json, runs/{contempo,smarter}/<set>{,+head}/,
            runs/dev_v3/<sample>/manifest.json
  LifeTox   runs/jev-opt/lifetox-real-1/ (pilot; test consumed once, at the authors' request)

Values from the git history of the repository in which the study was run, which is not public, are read from
scripts/appendix/opt_history.json: code constants (fit grids, hygiene limits, population limits) and architecture
tables at the jev-opt commit recorded in environment.json by the runs that used them, so later code changes cannot
leak into the description, and the commit times of the protocols and of the records that fixed each choice.

Cost definitions (tab:opt-cost*):
  Opt. calls / $   usage.jsonl rows with source=omp (the host's ledger of optimizer LLM calls).
  Jev fresh $      run.json conditions.*.summary.jev_fresh_session (spend billed during the run); recorded
                   from v3 on, so "--" for v1/v2.
  Jev used $       Jev-reported cost of the distinct train/val requests in the run's response ledger
                   (requests served from the shared cache count); v2 values are read from the cost table of
                   results/GEPA_v2_val.md (same definition, analyze_v2.py), because scanning the v2 ledgers
                   (2.7 GB) would dominate the generator's run time; v1/v3/tailored are scanned here and the
                   v3/tailored values are checked against their saved reports.
  Test $           test_consumed.json metrics.total_cost.
"""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from datetime import datetime, timezone
from functools import cache
from typing import Any

import yaml

from paper_appendix import ROOT, ascii_only, esc, load_json, load_jsonl, register

STUDY = "runs/jev-opt"
CONDS = ("zero", "text", "program")
COND_TEX = {"zero": "Zero", "text": "Text", "program": "Program"}
CONDITION_NAME = {"zero": "jev-zero", "text": "jev-text-opt", "program": "jev-program-opt"}
SHORT = {"contempo-tbdense": "TB-Dense", "ezstance-subtaskA": "EZ-STANCE A", "smarter-implicit-hate": "SMARTER",
         "edit-intent": "Edit intent", "inli": "INLI", "dress-new-content": "DREsS cont.",
         "dress-new-organization": "DREsS org.", "dress-new-language": "DREsS lang."}
METRIC = {"contempo-tbdense": "micro-F1 (VAGUE neg.)", "ezstance-subtaskA": "macro-F1", "smarter-implicit-hate": "macro-F1",
          "edit-intent": "macro-F1", "inli": "accuracy", "dress-new-content": "QWK", "dress-new-organization": "QWK",
          "dress-new-language": "QWK"}
EXP = {"contempo-tbdense": ("contempo_tbdense", ""), "ezstance-subtaskA": ("ezstance_subtaskA", ""),
       "smarter-implicit-hate": ("smarter_implicit_hate", ""), "edit-intent": ("edit_intent", ""), "inli": ("inli", ""),
       "dress-new-content": ("dress_new", "_content"), "dress-new-organization": ("dress_new", "_organization"),
       "dress-new-language": ("dress_new", "_language")}
PAPER_KEY = {"contempo-tbdense": "contempo/tbdense", "ezstance-subtaskA": "ezstance/subtaskA",
             "smarter-implicit-hate": "smarter/implicit_hate", "edit-intent": "edit_intent", "inli": "inli",
             "dress-new-content": "dress_new", "dress-new-organization": "dress_new", "dress-new-language": "dress_new"}
GROUP_NAME = {"contempo-tbdense": "document", "inli": "premise"}  # readout CV groups (asserted against the configs)
METRIC_NAME = {"macro_f1": "macro-F1", "micro_f1": "micro-F1", "accuracy": "accuracy", "qwk": "QWK"}
REPS = ("best-quality", "best-cheap", "best-calibrated", "top-quality")
ARCH = "jev-opt/docs/architecture.md"
TEST_JSON = "results/gepa_v3_test.json"
# (key in gepa_v3_test.json, display label, round)
TEST_ROWS = [("v1 zero-shot", "Paper's question", "v1"), ("v1 text-opt", "GEPA text search", "v1"),
             ("v1 program-opt", "GEPA program search", "v1"), ("v2 chosen", "GEPA + demos/calib. (selected)", "v2"),
             ("**v3 chosen (headline)**", "Optimized questions", "v3"),
             ("Jev-native zero-shot (v3 N)", "Rephrased question", "v3"), ("tailored", "Jev-adapted GEPA", "tailored")]
# Display names of the study's configuration families (internal round keys never reach the paper).
ROUND_TEX = {"v1": "GEPA text/program search", "v2": "GEPA + demos/calib.", "v3": "Optimized questions",
             "tailored": "Jev-adapted GEPA", "LifeTox pilot": "LifeTox pilot", "v3 seeds": "Seed generation",
             "v3 P seeds": "Optimized questions, seeds 1--2", "lever C": "Reranking + thresholds",
             "heads": "Classifier on zero-shot answers"}
# Seed reruns of the headline configuration: v3-<task>-P-s<seed>; seed 0 is v3-<task>-P.
SEED_RERUN = re.compile(r"^v3-(.+)-P-s(\d+)$")


def is_seed_rerun(run_id: str) -> bool:
    return SEED_RERUN.match(run_id) is not None


# Short column heads of tab:opt-valtest (defined in its caption).
TEST_SHORT = {"v1 zero-shot": "Paper's q.", "v1 text-opt": "GEPA text", "v1 program-opt": "GEPA prog.",
              "v2 chosen": "GEPA+d/c", "**v3 chosen (headline)**": "Opt.", "Jev-native zero-shot (v3 N)": "Rephrased q.",
              "tailored": "Jev-adapt."}
V2_CONFIGS = ["A: seed + calibration", "B: + kNN demos", "C: text-opt", "D: program-opt", "E: ensemble-3 (C+D)",
              "E: ensemble-5 (C+D)"]
V2_TEX = {"A: seed + calibration": "A: seed + class bias", "B: + kNN demos": "B: A + kNN demos", "C: text-opt": "C: text-opt",
          "D: program-opt": "D: program-opt", "E: ensemble-3 (C+D)": "E: ensemble-3 (C+D)",
          "E: ensemble-5 (C+D)": "E: ensemble-5 (C+D)"}
V3_CONFIGS = ["v2-A: seed + class bias (reference)", "zero: paper seed", "zero: jev-native seed", "zero: decomposed seed",
              "A: seed + readout head", "head: paper seed (± demos)", "head: jev-native seed", "head: decomposed seed",
              "search from paper", "search from jev-native", "search from decomposed", "P: population-opt",
              "P+head: stacked readout"]
V3_CHOICES = ["A: seed + readout head", "P: population-opt", "P+head: stacked readout"]


def ascii_md(s: str) -> str:
    return s.replace("±", "+/-")


# --------------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------------
def pct(v: float) -> str:
    return f"{100 * v:.1f}"


def spct(v: float) -> str:
    return f"{100 * v:+.1f}"


def ci_txt(lo: float, hi: float, signed: bool = False) -> str:
    f = spct if signed else pct
    return f"[{f(lo)}, {f(hi)}]"


def dollars(v: float) -> str:
    return f"\\${v:,.0f}"


def usd(v: float | None, nd: int = 2) -> str:
    return "--" if v is None else f"{v:,.{nd}f}"


def num(v: int | float) -> str:
    return f"{v:,}" if isinstance(v, int) else f"{v:,.0f}"


def tt(s: str) -> str:
    return f"\\texttt{{{esc(s)}}}"


HISTORY = "scripts/appendix/opt_history.json"


@cache
def history() -> dict[str, Any]:
    """Values from the study repository's git history (see the module docstring and the file's _about)."""
    return load_json(HISTORY)


def code_fact(sha: str, rel: str, name: str) -> Any:
    """`name` in `rel` at jev-opt commit `sha`: a constant's source text or a value read from the file."""
    try:
        return history()["code"][sha][rel][name]
    except KeyError:
        raise KeyError(f"{name} of {rel}@{sha[:9]} not in {HISTORY}") from None


def commit(name: str) -> tuple[str, datetime]:
    """(commit, commit time) of a protocol or record, by its name in HISTORY."""
    h, t = history()["commits"][name]
    return h, parse_time(t)


def code_time(sha: str) -> str:
    """Commit time (UTC) of the jev-opt source at commit `sha`, for the text: the commit itself is not public."""
    return f"{utc(parse_time(history()['code_times'][sha]))} UTC"


def parse_time(s: str) -> datetime:
    t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def utc(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")


def center(colspec: str, header: str, body: list[str], size: str = "\\scriptsize", pre: str = "") -> str:
    return "\n".join([f"\\begin{{center}}{size}{pre}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header,
                      "\\midrule", *body, "\\bottomrule", "\\end{tabular}", "\\end{center}"]) + "\n"


def float_table(colspec: str, header: str, body: list[str], caption: str, label: str, star: bool = True,
                size: str = "\\scriptsize", pre: str = "\\setlength{\\tabcolsep}{3pt}", pos: str = "!tp") -> str:
    env = "table*" if star else "table"
    return "\n".join([f"\\begin{{{env}}}[{pos}]", f"\\centering{size}{pre}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
                      header, "\\midrule", *body, "\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}",
                      f"\\label{{{label}}}", f"\\end{{{env}}}"]) + "\n"


def two_line_cells(cells: list[tuple[float | None, list[float] | None, bool]]) -> tuple[str, str]:
    """Value row and a \\tiny CI row for a row of (value, [lo, hi], bold) cells."""
    top, bot = [], []
    for v, ci, bold in cells:
        if v is None:
            top.append("--")
            bot.append("")
            continue
        s = pct(v)
        top.append(f"\\textbf{{{s}}}" if bold else s)
        bot.append(f"{{\\tiny {pct(ci[0])}--{pct(ci[1])}}}" if ci else "")
    return " & ".join(top), " & ".join(bot)


def task_header(first: str = "Configuration") -> str:
    return f"{first} & " + " & ".join(esc(SHORT[s]) for s in tasks()) + " \\\\"


# --------------------------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------------------------
@cache
def tasks() -> list[str]:
    ts = list(load_json(TEST_JSON)["tasks"])
    if set(ts) != set(SHORT):
        raise KeyError(f"task set changed: {ts}")
    return ts


def run_dir(slug: str, cond: str) -> str:
    return f"{STUDY}/study-{slug}-{cond}"


@cache
def rjson(run: str, name: str) -> Any:
    p = ROOT / STUDY / run / name
    return json.loads(p.read_text()) if p.exists() else None


@cache
def rconfig(run: str) -> dict[str, Any]:
    return yaml.safe_load((ROOT / STUDY / run / "config.yaml").read_text())["task"]


@cache
def omp_rows(run: str) -> list[dict[str, Any]]:
    return [u for u in load_jsonl(f"{STUDY}/{run}/usage.jsonl") if u["source"] == "omp"]


def omp_calls(run: str) -> int:
    return len(omp_rows(run))


def omp_cost(run: str) -> float:
    return sum(float(u["cost_total"]) for u in omp_rows(run))


def wall_h(run: str) -> float:
    s = rjson(run, "run.json")
    return (parse_time(s["finished_at"]) - parse_time(s["started_at"])).total_seconds() / 3600


def fresh_cost(run: str) -> float | None:
    conds = rjson(run, "run.json")["conditions"].values()
    vals = [c.get("summary", {}).get("jev_fresh_session") for c in conds]
    if any(v is None for v in vals):
        return None
    return sum(float(v["cost"]) for v in vals)


def test_cost(run: str) -> float | None:
    t = rjson(run, "test_consumed.json")
    return None if t is None else float(t["metrics"]["total_cost"])


_HEAD = re.compile(r'^\{"request_hash": "([0-9a-f]+)", "split": "(\w+)"')
_COST = re.compile(r'"cost": ([-0-9.eE+]+|null)')


@cache
def ledger(run: str) -> tuple[tuple[str, str, float], ...]:
    """(request hash, split, Jev-reported cost) for every record of the run's response ledger."""
    out = []
    d = ROOT / STUDY / run
    for shard in rjson(run, "run.json")["jev_responses"]["shards"]:
        with gzip.open(d / shard, "rt", encoding="utf-8") as f:
            for line in f:
                m = _HEAD.match(line)
                i = line.rfind('"usage": {')
                c = _COST.search(line, i) if i >= 0 else None
                if not m or not c:
                    r = json.loads(line)
                    out.append((r["request_hash"], r.get("split", "?"), float((r["response"].get("usage") or {}).get("cost") or 0.0)))
                    continue
                out.append((m.group(1), m.group(2), 0.0 if c.group(1) == "null" else float(c.group(1))))
    return tuple(out)


def jev_used(run: str) -> float | None:
    """Distinct train/val requests of the run's ledger at Jev's reported cost (ledgers are per-run deduplicated);
    None for the pilot's ledger, which predates the split field."""
    recs = ledger(run)
    if any(s == "?" for _, s, _ in recs):
        return None
    return sum(c for _, s, c in recs if s != "test")


@cache
def md_costs(rel: str) -> dict[str, dict[str, float]]:
    """Cost-per-run table of an analysis report: run -> {jev_used, ...} (columns named from the header)."""
    out: dict[str, dict[str, float]] = {}
    lines = (ROOT / rel).read_text().split("## Cost per run", 1)[1].splitlines()
    head = None
    for ln in lines:
        if not ln.startswith("|") or ln.startswith("|---"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if head is None:
            head = cells
            continue
        run = cells[1].strip("`")
        out[run] = {h: float(c.replace("$", "").replace(",", "")) for h, c in zip(head[2:], cells[2:]) if c}
    return out


def v2_jev_used(run: str) -> float:
    return md_costs("results/GEPA_v2_val.md")[run]["Jev $"]


# --------------------------------------------------------------------------------------------
# v1 (kept from the study appendix; tables now report volume, costs are in tab:opt-cost)
# --------------------------------------------------------------------------------------------
def gepa_rows() -> list[dict[str, Any]]:
    rows = load_json("results/gepa.json")["rows"]
    missing = [r["slug"] for r in rows if r["slug"] not in SHORT]
    if missing:
        raise KeyError(f"no short name for {missing}")
    return rows


@cache
def study_commit() -> str:
    """Repository commit recorded (environment.json) by the last-started v1 study run."""
    starts = {(load_json(f"{run_dir(r['slug'], c)}/run.json")["started_at"], r["slug"], c) for r in gepa_rows() for c in CONDS}
    _, slug, c = max(starts)
    return load_json(f"{run_dir(slug, c)}/environment.json")["jev_opt_git_sha"]


@cache
def round_commit(prefix: str) -> str:
    """The single jev-opt commit recorded by every run of a round (v3-, tailored-)."""
    shas = {rjson(r, "environment.json")["jev_opt_git_sha"] for r in round_runs(prefix)}
    if len(shas) != 1:
        raise ValueError(f"{prefix}: runs recorded several jev-opt commits {shas}")
    return shas.pop()


@cache
def round_runs(prefix: str) -> tuple[str, ...]:
    return tuple(sorted(p.name for p in (ROOT / STUDY).iterdir() if p.is_dir() and p.name.startswith(prefix)
                        and (p / "run.json").exists() and not is_seed_rerun(p.name)))


@cache
def seed_reruns() -> tuple[str, ...]:
    """Seed-1/2 reruns of the headline configuration (v3-<task>-P-s<seed>)."""
    return tuple(sorted(p.name for p in (ROOT / STUDY).iterdir() if p.is_dir() and is_seed_rerun(p.name)
                        and (p / "run.json").exists()))


@cache
def run(slug: str, cond: str) -> dict[str, Any]:
    """Everything the v1 tables need about one study run."""
    d = run_dir(slug, cond)
    state = load_json(f"{d}/run.json")
    (name, cstate), = state["conditions"].items()
    if name != CONDITION_NAME[cond] or cstate["status"] != "finished" or state.get("resumed_at") is not None:
        raise ValueError(f"{d}: unexpected run state {name} {cstate['status']} resumed={state.get('resumed_at')}")
    usage = load_jsonl(f"{d}/usage.jsonl")
    omp = [u for u in usage if u["source"] == "omp"]
    jev = [u for u in usage if u["source"] == "jev"]
    if {u["split"] for u in jev} - {"train", "val"}:
        raise ValueError(f"{d}: Jev usage rows outside train/val")
    if {u["role"] for u in omp} - {"reflection", "structural"}:
        raise ValueError(f"{d}: unknown optimizer role")
    config = yaml.safe_load((ROOT / d / "config.yaml").read_text())
    cand_dir = ROOT / d / CONDITION_NAME[cond] / "candidates"
    cands = sorted((json.loads(p.read_text()) for p in cand_dir.glob("*.json")), key=lambda c: c["index"])
    t0, t1 = (datetime.fromisoformat(state[k]) for k in ("started_at", "finished_at"))
    return {
        "summary": cstate["summary"], "config": config, "env": load_json(f"{d}/environment.json"),
        "pareto": load_json(f"{d}/pareto.json"), "test": load_json(f"{d}/test_consumed.json"), "cands": cands,
        "refl_n": sum(u["role"] == "reflection" for u in omp), "struct_n": sum(u["role"] == "structural" for u in omp),
        "refl_cost": sum(u["cost_total"] for u in omp if u["role"] == "reflection"),
        "opt_cost": sum(u["cost_total"] for u in omp),
        "jev_train": sum(u["calls"] for u in jev if u["split"] == "train"),
        "jev_val": sum(u["calls"] for u in jev if u["split"] == "val"),
        "jev_tokens": sum(u["input_tokens"] for u in jev), "jev_cost": sum(u["cost"] for u in jev),
        "wall_min": (t1 - t0).total_seconds() / 60,
    }


def all_runs() -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
    return [(row, c, run(row["slug"], c)) for row in gepa_rows() for c in CONDS]


@cache
def ledger_union() -> dict[str, Any]:
    """Distinct Jev requests across the 24 v1 runs' response ledgers (request hash -> cost, first split)."""
    seen: dict[str, tuple[str, float]] = {}
    per_run = 0
    for row, c, _ in all_runs():
        for h, split, cost in ledger(f"study-{row['slug']}-{c}"):
            per_run += 1
            seen.setdefault(h, (split, cost))
    return {"n": len(seen), "cost": sum(c for _, c in seen.values()), "per_run": per_run,
            "test_n": sum(s == "test" for s, _ in seen.values())}


@cache
def totals() -> dict[str, float]:
    runs = [r for _, _, r in all_runs()]
    t = {k: sum(r[k] for r in runs) for k in ("refl_n", "struct_n", "refl_cost", "opt_cost", "jev_train", "jev_val",
                                              "jev_tokens", "jev_cost", "wall_min")}
    t["test_calls"] = sum(r["test"]["metrics"]["n"] * r["test"]["metrics"]["mean_calls"] for r in runs)
    t["test_cost"] = sum(r["test"]["metrics"]["total_cost"] for r in runs)
    t["metric_calls"] = sum(r["summary"]["metric_calls"] for r in runs)
    t["cands"] = sum(r["summary"]["candidates"] for r in runs)
    return t


def macro_values() -> dict[str, str]:
    """Study-wide totals over the 24 v1 runs (macros for generated/numbers.tex; see the v1 cost definitions)."""
    t = totals()
    return {"gepaReflCalls": f"{t['refl_n']:,}", "gepaStructCalls": f"{t['struct_n']:,}",
            "gepaOptCalls": f"{t['refl_n'] + t['struct_n']:,}", "gepaReflCost": dollars(t["refl_cost"]),
            "gepaOptCost": dollars(t["opt_cost"]), "gepaJevCalls": f"{t['jev_train'] + t['jev_val']:,}",
            "gepaJevCost": dollars(t["jev_cost"])}


@register("opt_macros")
def opt_macros() -> str:
    return "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macro_values().items())


def top_val(slug: str, cond: str) -> float:
    return run(slug, cond)["pareto"]["representatives"]["top-quality"]["val"]["primary"]


# --------------------------------------------------------------------------------------------
# Inline values
# --------------------------------------------------------------------------------------------
INLINE: dict[str, Any] = {}


def inline(key: str):
    def deco(fn):
        INLINE[key] = cache(fn)
        register(f"opt.{key}")(INLINE[key])
        return fn
    return deco


def v1_same(key: str) -> Any:
    vals = {json.dumps(r["config"]["task"]["optimizer"][key]) for _, _, r in all_runs()}
    if len(vals) != 1:
        raise ValueError(f"optimizer.{key} differs across v1 runs: {vals}")
    return json.loads(vals.pop())


def round_same(prefix: str, key: str, runs: list[str] | None = None) -> Any:
    rs = runs if runs is not None else list(round_runs(prefix))
    vals = {json.dumps(rconfig(r)["optimizer"].get(key)) for r in rs}
    if len(vals) != 1:
        raise ValueError(f"optimizer.{key} differs across {prefix} runs: {vals}")
    return json.loads(vals.pop())


def v3_p_runs() -> list[str]:
    return [f"v3-{s}-P" for s in tasks()]


@cache
def pop() -> dict[str, Any]:
    """optimizer.population of the v3 P runs, identical across tasks except the per-task seed paths."""
    vals = {json.dumps({k: v for k, v in rconfig(r)["optimizer"]["population"].items() if k != "seeds"}) for r in v3_p_runs()}
    if len(vals) != 1:
        raise ValueError(f"population settings differ across v3 P runs: {vals}")
    return json.loads(vals.pop())


for _k, _f in {
    "nRuns": lambda: str(len(all_runs())),
    "nTasks": lambda: str(len(tasks())),
    "metricCalls": lambda: f"{totals()['metric_calls']:,}",
    "candidates": lambda: f"{totals()['cands']:,}",
    "gepaOptCalls": lambda: macro_values()["gepaOptCalls"],
    "gepaReflCalls": lambda: macro_values()["gepaReflCalls"],
    "gepaStructCalls": lambda: macro_values()["gepaStructCalls"],
    "gepaOptCost": lambda: macro_values()["gepaOptCost"],
    "gepaReflCost": lambda: macro_values()["gepaReflCost"],
    "gepaJevCalls": lambda: macro_values()["gepaJevCalls"],
    "gepaJevCost": lambda: macro_values()["gepaJevCost"],
    "jevTokensM": lambda: f"{totals()['jev_tokens'] / 1e6:,.0f}",
    "testCalls": lambda: f"{totals()['test_calls']:,.0f}",
    "testCost": lambda: f"\\${totals()['test_cost']:,.2f}",
    "wallHours": lambda: f"{totals()['wall_min'] / 60:,.1f}",
    "distinctReq": lambda: f"{ledger_union()['n']:,}",
    "distinctCost": lambda: f"\\${ledger_union()['cost']:,.2f}",
    "ledgerRecords": lambda: f"{ledger_union()['per_run']:,}",
    "ledgerTestRecords": lambda: f"{ledger_union()['test_n']:,}",
    "studyCodeTime": lambda: code_time(study_commit()),
    "progressive": lambda: " and ".join(f"{x:,}" for x in v1_same("progressive")),
    "v1FitExamples": lambda: f"{v1_same('fit_examples'):,}",
    "minibatch": lambda: f"{v1_same('reflection_minibatch_size'):,}",
    "optimizerSpec": lambda: tt(v1_same("model")),
    "forbiddenLoaded": lambda: str(sum(len(rjson(r, "environment.json")["forbidden_modules_loaded"])
                                       for r in all_run_names())),
    "discrepanciesV1": lambda: str(len(load_json("results/gepa.json")["discrepancies"])),
    "discrepancies": lambda: str(len(load_json(TEST_JSON)["discrepancies"])),
    "nTested": lambda: str(sum(len(v["rows"]) for v in load_json(TEST_JSON)["tasks"].values())),
    "nTestRows": lambda: str(len(TEST_ROWS)),
    "fitExamples": lambda: f"{round_same('v3-', 'fit_examples'):,}",
    "reflPool": lambda: f"{round_same('v2-', 'reflection_pool', [r for r in round_runs('v2-') if rconfig(r)['optimizer'].get('reflection_pool')]):,}",
    "cvFolds": lambda: str(round_same("v3-", "readout_folds")),
    "maxComponentChars": lambda: f"{round_same('v3-', 'max_component_chars'):,}",
    "popSize": lambda: str(pop()["size"]),
    "popRounds": lambda: str(pop()["rounds"]),
    "popElite": lambda: str(pop()["elite"]),
    "popAux": lambda: str(pop()["max_aux"]),
    "popValCands": lambda: str(pop()["val_candidates"]),
    "popStackMax": lambda: str(pop()["stack_max"]),
    "popWorst": lambda: str(pop()["worst_examples"]),
    "popJevCap": lambda: f"\\${pop()['max_jev_cost']:.0f}",
    "popJevCapPlain": lambda: f"${pop()['max_jev_cost']:.0f}",
    "demoK": lambda: str(pop()["demos"]["k"]),
    "embedModel": lambda: tt(embedding()["model"]),
    "embedRevision": lambda: tt(embedding()["revision"][:8]),
    "maxCombos": lambda: f"{code_const(round_commit('v3-'), 'jev-opt/python/src/jev_opt/population.py', 'MAX_COMBOS'):,}",
    "leakChars": lambda: str(code_const(round_commit("v3-"), "jev-opt/python/src/jev_opt/hygiene.py", "LEAK_CHARS")),
    "sizeRule": lambda: size_rule(),
    "classBiasGrid": lambda: grid_txt("CLASS_BIAS_GRID"),
    "biasGrid": lambda: grid_txt("BIAS_GRID"),
    "probGrid": lambda: grid_txt("PROB_GRID"),
    "lrC": lambda: set_txt("LR_C"),
    "ridgeAlpha": lambda: set_txt("RIDGE_ALPHA"),
    "tempBounds": lambda: temp_bounds(),
    "logpFloor": lambda: code_fact(round_commit("tailored-"), "jev-opt/python/src/jev_opt/tailored.py", "LOGP_FLOOR"),
    "acceptN": lambda: str(tailored_same("accept_n")),
    "acceptTol": lambda: f"{100 * tailored_same('accept_tol'):.0f}",
    "minimalPairs": lambda: str(tailored_same("minimal_pairs")),
    "tailoredJevCap": lambda: f"\\${tailored_same('max_jev_cost'):.0f}",
    "tailoredStarts": lambda: " and ".join({"paper": "the paper's question", "jev-native": "a rephrased question"}[s]
                                           for s in tailored_same("starts")),
    "v3Margin": lambda: v3_margin(),
    "v3CodeTime": lambda: code_time(round_commit("v3-")),
    "tailoredCodeTime": lambda: code_time(round_commit("tailored-")),
    "seedCost": lambda: f"\\${sum(g['optimizer_cost'] for g in seed_generations().values()):.2f}",
    "seedCalls": lambda: f"{sum(n for n in seed_calls().values()):,}",
    "seedsPerKind": lambda: seeds_per_kind(),
}.items():
    INLINE[_k] = cache(_f)
    register(f"opt.{_k}")(INLINE[_k])


def all_run_names() -> list[str]:
    """Optimization-study runs. fz-* runs (final optimizer settings) are reported by analyze_frozen.py."""
    return [p.name for p in sorted((ROOT / STUDY).iterdir())
            if p.is_dir() and (p / "run.json").exists() and not p.name.startswith("fz-")]


@cache
def embedding() -> dict[str, Any]:
    vals = {json.dumps(rconfig(r).get("embedding")) for r in round_runs("v3-") + round_runs("v2-")}
    if len(vals) != 1:
        raise ValueError(f"embedding differs: {vals}")
    return json.loads(vals.pop())


def code_const(sha: str, rel: str, name: str) -> Any:
    import ast
    return ast.literal_eval(code_fact(sha, rel, name))


FIT = "jev-opt/python/src/jev_opt/fit.py"


def grid_txt(name: str) -> str:
    raw = code_fact(round_commit("v3-"), FIT, name)
    m = re.search(r"np\.arange\(([-0-9.]+), ([-0-9.]+), ([0-9.]+)\)", raw)
    if not m:
        raise ValueError(f"{name}: unexpected grid {raw}")
    lo, hi, step = (float(x) for x in m.groups())
    hi = round(lo + step * int((hi - lo) / step + 1e-9), 2)
    return f"${lo:g}, {lo + step:g}, \\dots, {hi:g}$"


def set_txt(name: str) -> str:
    vals = code_const(round_commit("v3-"), FIT, name)
    if vals != code_const(round_commit("tailored-"), FIT, name):
        raise ValueError(f"{name} changed between v3 and tailored")
    return "$\\{" + ", ".join(f"{v:g}" for v in vals) + "\\}$"


def temp_bounds() -> str:
    lo, hi = code_fact(round_commit("v3-"), FIT, "temperature bounds")
    return f"$[{float(lo):g}, {float(hi):g}]$"


def size_rule() -> str:
    low, factor = code_fact(study_commit(), "jev-opt/python/src/jev_opt/hygiene.py", "size cap")
    return f"{factor}$\\times$ the seed's size (at least {low:,} characters)"


def tailored_same(key: str) -> Any:
    vals = {json.dumps(rconfig(r)["optimizer"]["tailored"][key]) for r in round_runs("tailored-")}
    if len(vals) != 1:
        raise ValueError(f"tailored.{key} differs: {vals}")
    return json.loads(vals.pop())


def v3_margin() -> str:
    m = re.search(r"^MARGIN = ([0-9.]+)", (ROOT / "jev-opt/experiments/analyze_v3.py").read_text(), re.M)
    return f"{float(m.group(1)):.1f}"


def seed_dirs() -> dict[str, str]:
    return {s: f"jev-opt/experiments/{EXP[s][0]}/seeds_v3{EXP[s][1]}" for s in tasks()}


@cache
def seed_generations() -> dict[str, dict[str, Any]]:
    return {s: load_json(f"{d}/generation.json") for s, d in seed_dirs().items()}


@cache
def seed_calls() -> dict[str, int]:
    return {s: sum(u["source"] == "omp" for u in load_jsonl(f"{d}/usage.jsonl")) for s, d in seed_dirs().items()}


def seeds_per_kind() -> str:
    ns = {len(k["files"]) for g in seed_generations().values() for k in g["kinds"].values()}
    if len(ns) != 1:
        raise ValueError(f"seed counts differ: {ns}")
    return str(ns.pop())


# --------------------------------------------------------------------------------------------
# Data, settings, ledger
# --------------------------------------------------------------------------------------------
def test_n(slug: str) -> int:
    ns = {rjson(r, "test_consumed.json")["metrics"]["n"] for r in tested_runs(slug)}
    if len(ns) != 1:
        raise ValueError(f"{slug}: test n differs {ns}")
    return ns.pop()


def tested_runs(slug: str) -> list[str]:
    return [load_json(TEST_JSON)["tasks"][slug]["rows"][k]["run"] for k, _, _ in TEST_ROWS]


@register("opt_data")
def opt_data() -> str:
    body = []
    for s in tasks():
        cfg = rconfig(f"v3-{s}-P")
        env = rjson(f"v3-{s}-P", "environment.json")["splits"]
        group = cfg["optimizer"].get("readout_group")
        if bool(group) != (s in GROUP_NAME):
            raise ValueError(f"{s}: readout grouping {group!r} does not match GROUP_NAME")
        budgets = {rconfig(f"study-{s}-{c}")["optimizer"]["max_metric_calls"] for c in ("text", "program")}
        if len(budgets) != 1:
            raise ValueError(f"{s}: v1 budget differs")
        body.append(" & ".join([esc(SHORT[s]), esc(METRIC[s]), str(len(cfg["labels"])), f"{env['n_train']:,}",
                                f"{env['n_val']:,}", f"{test_n(s):,}", f"{cfg['optimizer']['fit_examples']:,}",
                                GROUP_NAME.get(s, "--"), f"{budgets.pop():,}"]) + " \\\\")
    head = "Task & Primary metric & Labels & Train & Val & Test & Fit slice & CV groups & GEPA budget \\\\"
    return float_table("llrrrrrlr", head, body,
                       "Task configurations of the optimization study, identical in all rounds. \\emph{Fit slice}: stratified "
                       "train prefix on which numbers and logistic regressions are fit (second and third pre-registration and Jev-adapted GEPA). "
                       "\\emph{CV groups}: unit that no cross-validation fold of the logistic regression splits. \\emph{GEPA budget}: metric calls "
                       "per optimizing GEPA run (GEPA text/program search, and GEPA with demonstrations and calibration).",
                       "tab:opt-data", pre="\\setlength{\\tabcolsep}{4pt}")


SETTING_ROWS = [("fit_examples", "Fit slice (train examples)"), ("fit_holdout", "Fit slice held out of reflection"),
                ("reflection_pool", "Reflection pool (train examples)"), ("reflection_minibatch_size", "Reflection minibatch"),
                ("progressive", "Progressive val subsets"), ("use_merge", "GEPA merge"), ("class_bias", "Choice class biases"),
                ("readout", "LR on answers"), ("readout_folds", "CV folds on the fit slice"), ("max_component_chars", "Chars per text field"),
                ("max_program_chars", "Program size cap (chars)"), ("seed", "Optimizer seed")]


def fmt_setting(v: Any) -> str:
    if v is None:
        return "--"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, list):
        return "/".join(f"{x:,}" for x in v) if v else "--"
    return f"{v:,}" if isinstance(v, int) else esc(v)


# The question optimization (`jev-population-opt`) is not a GEPA run: it never reads GEPA's reflection pool, reflection
# minibatch, progressive validation or merge, and its proposer sees the out-of-fold errors of the fit slice itself
# (jev_opt.population.PopulationSearch.evidence), whatever its configuration says. Its Choice questions sit inside
# readout views, which jev_opt.fit.parameters never gives class biases, so `class_bias: true` has no effect either.
POPULATION_CELL = {"fit_holdout": "no", "reflection_pool": "--", "reflection_minibatch_size": "--", "progressive": "--",
                   "use_merge": "--", "class_bias": "--"}


def setting(name: str, run: str, key: str) -> Any:
    opt = rconfig(run)["optimizer"]
    if name == "tailored" and key == "reflection_minibatch_size" and opt["tailored"]["continuous_accept"]:
        return opt["tailored"]["accept_n"]  # Jev-adapted GEPA draws its own minibatch of accept_n train examples (T2)
    return opt.get(key)


@register("opt_settings")
def opt_settings() -> str:
    groups = [("v1", [f"study-{s}-{c}" for s in tasks() for c in ("text", "program")]),
              ("v2", [r for r in round_runs("v2-") if not r.endswith(("-A", "-B"))]),
              ("v3 P", v3_p_runs() + list(seed_reruns())), ("tailored", list(round_runs("tailored-")))]
    body = []
    for key, label in SETTING_ROWS:
        cells = []
        for name, runs in groups:
            if name == "v3 P" and key in POPULATION_CELL:
                cells.append(POPULATION_CELL[key])
                continue
            vals = {json.dumps(setting(name, r, key)) for r in runs}
            if key == "seed":
                cells.append("/".join(str(v) for v in sorted(json.loads(v) for v in vals)))
                continue
            if len(vals) != 1:
                raise ValueError(f"{name}: optimizer.{key} differs {vals}")
            v = json.loads(vals.pop())
            cells.append("default" if key == "max_program_chars" and v is None else fmt_setting(v))
        body.append(f"{label} & " + " & ".join(cells) + " \\\\")
    head = "Setting & \\multicolumn{1}{c}{GEPA} & \\multicolumn{1}{c}{GEPA+d/c C/D} & \\multicolumn{1}{c}{Opt. P} & \\multicolumn{1}{c}{Jev-adapt.} \\\\"
    return float_table("@{}lcccc@{}", head, body,
                       "Optimizer settings of each method, read from the run configurations. GEPA+d/c is GEPA with "
                       "demonstrations and calibration, and Opt.\\ P is our optimized questions. -- marks a setting "
                       "the method does not use: Opt.\\ P is not a GEPA run, and its reflection sees the fit slice. "
                       "The default size cap is 8 times the seed program's size, at least 4,000 characters. GEPA+d/c "
                       "ran seeds 1 and 2 on validation only, to choose its configuration.", "tab:opt-settings",
                       star=False, pre="\\setlength{\\tabcolsep}{2pt}")


def round_of(run_id: str) -> str:
    if is_seed_rerun(run_id):
        return "v3 P seeds"
    for p, name in (("study-", "v1"), ("v2-", "v2"), ("v3-", "v3"), ("tailored-", "tailored"), ("lifetox-", "LifeTox pilot")):
        if run_id.startswith(p):
            return name
    raise ValueError(run_id)


@cache
def test_ledger() -> list[dict[str, Any]]:
    """Test evaluations of this study's rounds; the shared ledger also holds later experiments (fz-, cv-, cl-, h2-, lora-)."""
    return [x for x in load_jsonl(f"{STUDY}/test_ledger.jsonl")
            if x["run_id"].startswith(("study-", "v2-", "v3-", "tailored-", "lifetox-"))]


@register("opt_ledger")
def opt_ledger() -> str:
    rows = test_ledger()
    if len({x["run_id"] for x in rows}) != len(rows):
        raise ValueError("a run appears twice in the test ledger")
    body = []
    for rnd in ("v1", "v2", "v3", "v3 P seeds", "tailored", "LifeTox pilot"):
        sel = [x for x in rows if round_of(x["run_id"]) == rnd]
        reps = Counter(x["representative"] for x in sel)
        t = sorted(parse_time(x["consumed_at"]) for x in sel)
        body.append(" & ".join([ROUND_TEX[rnd], str(len(sel)), ", ".join(f"{rep_tex(k)}~{v}" for k, v in sorted(reps.items())),
                                f"{utc(t[0])}", f"{utc(t[-1])}"]) + " \\\\")
    body.append("\\midrule")
    body.append(f"Total & {len(rows)} & & & \\\\")
    head = "Configuration & Rows & Representatives consumed & First (UTC) & Last (UTC) \\\\"
    return float_table("lrp{6.9cm}ll", head, body,
                       "The one-test-use ledger: one row per test consumption, and no run appears twice. The optimized questions' "
                       "rows include the rephrased-question runs. Its seed reruns (seeds 1 and 2) are a separate row.",
                       "tab:opt-ledger")


def rep_tex(rep: str) -> str:
    """Representative name as printed; the Jev-adapted GEPA representative is named by what it is."""
    return "best fitted" if rep == "top-tailored" else tt(rep)


@inline("ledgerRows")
def _ledger_rows() -> str:
    return str(len(test_ledger()))


@inline("v3LedgerSplit")
def _v3_split() -> str:
    sel = [x for x in test_ledger() if round_of(x["run_id"]) == "v3"]
    p = sum(x["run_id"].endswith("-P") for x in sel)
    n = sum(x["run_id"].endswith("-N") for x in sel)
    if p + n != len(sel):
        raise ValueError("v3 ledger rows outside P/N runs")
    return f"{p} optimized-questions and {n} rephrased-question"


@inline("seedLedgerSplit")
def _seed_split() -> str:
    sel = [x["run_id"] for x in test_ledger() if round_of(x["run_id"]) == "v3 P seeds"]
    per = Counter(SEED_RERUN.match(r).group(1) for r in sel)
    if set(per) != set(tasks()) or len(set(per.values())) != 1 or sorted(sel) != sorted(seed_reruns()):
        raise ValueError(f"seed-rerun ledger rows are not one per seed rerun and task: {per}")
    return str(len(sel))


@inline("testHashesPerTask")
def _hashes() -> str:
    counts = {s: len({x["test_sha256"] for x in test_ledger() if x["run_id"].split("-", 1)[1].startswith(s + "-")
                      or x["run_id"].split("-", 1)[1] == s}) for s in tasks()}
    if set(counts.values()) != {1}:
        raise ValueError(f"test SHA-256 not unique per task: {counts}")
    return "one"


@inline("rowsPerTask")
def _rows_per_task() -> str:
    counts = {s: sum(1 for x in test_ledger() if s in x["run_id"]) for s in tasks()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"ledger rows per task differ {counts}")
    return str(counts[tasks()[0]])


@inline("hygieneTestInTrainVal")
def _hyg1() -> str:
    return str(sum(rjson(x["run_id"], "test_consumed.json")["metrics"]["hygiene"]["test_ids_in_train_or_val"]
                   for x in test_ledger()))


@inline("hygieneTestChanged")
def _hyg2() -> str:
    return str(sum(bool(rjson(x["run_id"], "test_consumed.json")["metrics"]["hygiene"]["test_file_changed_since_optimize"])
                   for x in test_ledger()))


@inline("demoOutside")
def _hyg3() -> str:
    vals = [rjson(x["run_id"], "test_consumed.json")["metrics"]["hygiene"].get("demo_ids_outside_train")
            for x in test_ledger()]
    return str(sum(v for v in vals if v is not None))


@inline("tailoredZeroNote")
def _tz() -> str:
    text = (ROOT / "results/GEPA_v3_test.md").read_text()
    if "`tailored-zero` (`top-tailored-zero`) was not evaluated" not in text:
        raise ValueError("GEPA_v3_test.md no longer records the tailored-zero note")
    reps = {x["representative"] for x in test_ledger() if x["run_id"].startswith("tailored-")}
    if reps != {"top-tailored"}:
        raise ValueError(f"tailored runs consumed {reps}")
    return "its best fitted candidate"


# --------------------------------------------------------------------------------------------
# IR, operators, ownership (architecture.md tables at the v1 study and v3 commits, checked against the code when
# they were recorded in HISTORY)
# --------------------------------------------------------------------------------------------
def md_cell(s: str) -> str:
    parts = re.split(r"`([^`]*)`", s.replace("\\|", "|"))
    return "".join(f"\\texttt{{{esc(p)}}}" if i % 2 else esc(p.replace("**", "")) for i, p in enumerate(parts))


def operator_names() -> list[str]:
    """Deterministic structural operators of architecture.md section 6, each a function of transforms.py."""
    return code_fact(study_commit(), ARCH, "structural operators")


_DET_RE: re.Pattern | None = None


def is_deterministic(rationale: str | None) -> bool:
    global _DET_RE
    if _DET_RE is None:
        _DET_RE = re.compile(r"^(" + "|".join(operator_names()) + r"):")
    return bool(rationale and _DET_RE.match(rationale))


@register("opt_ownership")
def opt_ownership() -> str:
    groups: dict[str, list[str]] = {}
    for cap, owner, *_ in code_fact(study_commit(), ARCH, "## 1. Ownership"):
        groups.setdefault(owner, []).append(cap)
    body = [f"\\raggedright {md_cell(o)} & \\raggedright {md_cell('; '.join(caps))} \\tabularnewline" for o, caps in groups.items()]
    return center("@{}p{2.0cm}p{5.0cm}@{}", "Owner & Capabilities \\\\", body, pre="")


@register("opt_ir")
def opt_ir() -> str:
    body = []
    for kind, prim, sem in code_fact(round_commit("v3-"), ARCH, "## 4. Jev Decision Program IR"):
        k = kind.strip("`")
        body.append(f"\\raggedright\\texttt{{{esc(k)}}} & \\raggedright {md_cell(prim)} & \\raggedright {md_cell(sem)} \\tabularnewline")
    return float_table("@{}p{1.3cm}p{1.0cm}p{4.6cm}@{}", "Kind & Jev & Semantics \\\\", body,
                       "Node kinds of the \\jevopt{} program IR (\\texttt{jev-opt/\\allowbreak docs/\\allowbreak architecture.md} "
                       "\\S4, checked against \\texttt{ir.py}).", "tab:opt-ir", star=False, pre="")


@register("opt_operators")
def opt_operators() -> str:
    return ", ".join(f"\\texttt{{{esc(n)}}}" for n in operator_names())


# --------------------------------------------------------------------------------------------
# v1 tables
# --------------------------------------------------------------------------------------------
@register("opt_budget")
def opt_budget() -> str:
    head = ("Task & Cond. & Metric & Cand. & Full & Refl. & Struct. & \\multicolumn{2}{c}{Jev calls (opt.)} & Jev in \\\\\n"
            " & & calls & & val & calls & calls & train & val & Mtok \\\\")
    body: list[str] = []
    for i, row in enumerate(gepa_rows()):
        if i:
            body.append("\\midrule")
        for j, c in enumerate(CONDS):
            r = run(row["slug"], c)
            s = r["summary"]
            opt = c != "zero"
            cells = [esc(SHORT[row["slug"]]) if j == 0 else "", COND_TEX[c], f"{s['metric_calls']:,}", f"{s['candidates']:,}",
                     f"{s['full_val_evals']:,}" if opt else "--", f"{r['refl_n']:,}" if opt else "--",
                     f"{r['struct_n']:,}" if c == "program" else "--", f"{r['jev_train']:,}", f"{r['jev_val']:,}",
                     f"{r['jev_tokens'] / 1e6:.1f}"]
            body.append(" & ".join(cells) + " \\\\")
    body.append("\\midrule")
    runs = all_runs()
    for c in CONDS + ("all",):
        sel = [r for _, cc, r in runs if c in ("all", cc)]
        s = lambda k: sum(r[k] for r in sel)  # noqa: E731
        opt = c != "zero"
        cells = ["\\textit{Total}" if c == "all" else f"\\textit{{Sum {COND_TEX[c].lower()}}}", "",
                 f"{sum(r['summary']['metric_calls'] for r in sel):,}", f"{sum(r['summary']['candidates'] for r in sel):,}",
                 f"{sum(r['summary'].get('full_val_evals', 0) for r in sel):,}" if opt else "--",
                 f"{s('refl_n'):,}" if opt else "--", f"{s('struct_n'):,}" if c in ("program", "all") else "--",
                 f"{s('jev_train'):,}", f"{s('jev_val'):,}", f"{s('jev_tokens') / 1e6:.1f}"]
        body.append(" & ".join(cells) + " \\\\")
    return float_table("llrrrrrrrr", head, body,
                       "Search volume of the " + INLINE["nRuns"]() + " runs of the first pre-registration (costs: Table~\\ref{tab:opt-cost}). \\emph{Metric "
                       "calls}: GEPA's budget counter (example evaluations). \\emph{Cand.}: candidates incl.\\ the seed. \\emph{Full "
                       "val}: candidates promoted to the full validation split. \\emph{Refl.}/\\emph{Struct.}: optimizer LLM calls by "
                       "role. \\emph{Jev calls (opt.)} and \\emph{Jev in Mtok}: Jev decisions executed during optimization "
                       "(cache-served repeats included) and their input tokens.", "tab:opt-budget",
                       pre="\\setlength{\\tabcolsep}{4pt}")


@register("opt_calibration")
def opt_calibration() -> str:
    body = []
    for row in gepa_rows():
        cells = [esc(SHORT[row["slug"]]), f"{row['zero_brier']:.3f}", f"{row['zero_ece']:.3f}"]
        for c in ("text", "program"):
            cells += [f"{row['conds'][c]['brier']:.3f}", f"{row['conds'][c]['ece']:.3f}"]
        body.append(" & ".join(cells) + " \\\\")
    head = ("& \\multicolumn{2}{c}{Zero} & \\multicolumn{2}{c}{Text} & \\multicolumn{2}{c}{Program} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
            "Task & Brier & ECE & Brier & ECE & Brier & ECE \\\\")
    return center("lrrrrrr", head, body, pre="\\setlength{\\tabcolsep}{3pt}")


def cand_index(name: str) -> str:
    return name.rsplit("/", 1)[1]


@register("opt_representatives")
def opt_representatives() -> str:
    body = []
    for row in gepa_rows():
        for j, c in enumerate(("text", "program")):
            r = run(row["slug"], c)
            reps = r["pareto"]["representatives"]
            bq = reps["best-quality"]
            cells = [esc(SHORT[row["slug"]]) if j == 0 else "", COND_TEX[c][:4] + ("." if c == "program" else ""),
                     str(r["summary"]["best_candidate"]), f"{cand_index(bq['candidate'])} ({len(bq['indistinguishable'])})",
                     *(cand_index(reps[k]["candidate"]) for k in REPS[1:]), str(len(r["pareto"]["front"]))]
            body.append(" & ".join(cells) + " \\\\")
    head = "Task & Cond. & GEPA & Best-q. & Cheap & Calib. & Top-q. & Front \\\\"
    return center("llrrrrrr", head, body, size="\\scriptsize", pre="\\setlength{\\tabcolsep}{3pt}")


def body_chars(rel: str) -> int:
    return len("".join(ln for ln in (ROOT / rel).read_text().splitlines(keepends=True) if not ln.startswith("#")))


def kinds_of(p: dict[str, Any]) -> str:
    nodes = p["nodes"] if "nodes" in p else {}
    k = Counter(n["kind"] for n in nodes.values())
    order = ("switch", "stage", "branch", "choice", "one_vs_rest", "score")
    return ", ".join((f"{k[x]}$\\times${esc(x)}" if k[x] > 1 else esc(x)) for x in order if k[x])


def kinds(rel: str) -> str:
    return kinds_of(yaml.safe_load((ROOT / rel).read_text()))


def shape(p: dict[str, Any]) -> tuple:
    return tuple(sorted((v["kind"], tuple(sorted((v.get("options") or {}).keys()))) for v in p["nodes"].values()))


@register("opt_programs_v1")
def opt_programs_v1() -> str:
    body = []
    for row in gepa_rows():
        slug = row["slug"]
        d = run_dir(slug, "program")
        r = run(slug, "program")
        seed_shape = shape(yaml.safe_load((ROOT / d / "seed.yaml").read_text()))
        cyaml = sorted((ROOT / d / "jev-program-opt" / "candidates").glob("*.yaml"))
        changed = sum(shape(yaml.safe_load(p.read_text())) != seed_shape for p in cyaml)
        det = sum(is_deterministic(c.get("rationale")) for c in r["cands"])
        top_changed = shape(yaml.safe_load((ROOT / d / "top-quality.yaml").read_text())) != seed_shape
        cells = [esc(SHORT[slug]), kinds(f"{d}/seed.yaml"), (kinds(f"{d}/top-quality.yaml") if top_changed else "same"),
                 f"{changed}/{len(cyaml)}", str(det), f"{r['test']['metrics']['mean_calls']:.2f}",
                 f"{body_chars(f'{d}/seed.yaml'):,}", f"{body_chars(run_dir(slug, 'text') + '/top-quality.yaml'):,}",
                 f"{body_chars(f'{d}/top-quality.yaml'):,}"]
        body.append(" & ".join(cells) + " \\\\")
    head = ("& \\multicolumn{2}{c}{Program-opt structure} & Struct./ & Det. & Calls/ & \\multicolumn{3}{c}{Program chars} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){7-9}\n"
            "Task & Seed & Tested & cand. & ops & ex. & Seed & Text & Prog. \\\\")
    return float_table("lllrrrrrr", head, body,
                       "Programs of GEPA text/program search. \\emph{Seed}/\\emph{Tested}: IR node kinds of the seed and of the program-opt candidate "
                       "evaluated on test (\\emph{same}: seed topology). \\emph{Struct./cand.}: program-opt candidates whose "
                       "topology differs from the seed. \\emph{Det.\\ ops}: candidates produced by a deterministic operator. "
                       "\\emph{Calls/ex.}: Jev requests per test example. \\emph{Program chars}: YAML size of the seed and of the "
                       "tested text-opt and program-opt candidates.", "tab:opt-programs-v1", pre="\\setlength{\\tabcolsep}{4pt}")


# --------------------------------------------------------------------------------------------
# v2
# --------------------------------------------------------------------------------------------
@cache
def v2s() -> dict[str, Any]:
    return load_json(f"{STUDY}/v2-dev-summary.json")


def v2_seed_labels(slug: str) -> tuple[str, list[str]]:
    keys = [k for k in v2s()[slug] if re.match(r"^[CD] seed \d", k)]
    cond = {k[0] for k in keys}
    if len(cond) != 1:
        raise ValueError(f"{slug}: seed runs of several conditions")
    return cond.pop(), keys


# Selection rule of GEPA + demos/calib., fixed before its first run: highest validation primary metric; ties within
# V2_TIE points (x100) go to fewer Jev calls per example, then to the earlier row.
V2_TIE = 0.1
# Its chosen configurations as recorded on 2026-09-29 from the validation results, before any test use:
# run -> representative.
V2_REGISTERED = {"v2-contempo-tbdense-CD": "top-jev-text-opt", "v2-ezstance-subtaskA-CD": "top-jev-program-opt",
                 "v2-smarter-implicit-hate-CD": "top-jev-program-opt", "v2-edit-intent-CD": "top-jev-program-opt",
                 "v2-inli-CD": "ensemble-3", "v2-dress-new-content-CD": "ensemble-5",
                 "v2-dress-new-organization-CD": "ensemble-3", "v2-dress-new-language-CD": "ensemble-3"}
# Chosen configurations of question optimization (v3) as recorded on 2026-09-28 from the validation results, before
# any test use: run -> (representative, candidate).
V3_REGISTERED = {"v3-contempo-tbdense-P": ("top-single", "jev-population-opt/10"),
                 "v3-ezstance-subtaskA-P": ("top-single", "jev-population-opt/8"),
                 "v3-smarter-implicit-hate-P": ("top-stacked", "jev-population-opt/14"),
                 "v3-edit-intent-P": ("top-stacked", "jev-population-opt/14"),
                 "v3-inli-P": ("top-single", "jev-population-opt/9"),
                 "v3-dress-new-content-P": ("top-single", "jev-population-opt/8"),
                 "v3-dress-new-organization-P": ("top-single", "jev-population-opt/9"),
                 "v3-dress-new-language-P": ("top-single", "jev-population-opt/8")}
V3_MAX_STACKS = 3  # P+head: validation picks among at most this many stacked readouts
TAILORED_MARGIN = "1.0"  # Jev-adapted GEPA beats a configuration on validation only if at least this many points higher


def v2_tie() -> float:
    return V2_TIE


def v2_rule(slug: str) -> str:
    """Highest val; ties within v2_tie() points -> fewer Jev calls/ex, then earlier row."""
    rows = list(v2s()[slug].items())
    best = max(v["val"] for _, v in rows)
    tied = [(i, k, v) for i, (k, v) in enumerate(rows) if 100 * (best - v["val"]) <= v2_tie() + 1e-9]
    return min(tied, key=lambda t: (round(t[2]["mean_calls"], 2), t[0]))[1]


def v2_protocol_chosen() -> dict[str, str]:
    return V2_REGISTERED


@cache
def v2_chosen() -> dict[str, str]:
    out = {}
    for s in tasks():
        k = v2_rule(s)
        e = v2s()[s][k]
        if v2_protocol_chosen().get(e["run"]) != e["representative"]:
            raise ValueError(f"{s}: rule gives {e['run']} {e['representative']}, the recorded choice differs")
        t = load_json(TEST_JSON)["tasks"][s]["rows"]["v2 chosen"]
        if (t["run"], t["rep"]) != (e["run"], e["representative"]):
            raise ValueError(f"{s}: tested v2 row is not the chosen configuration")
        out[s] = k
    return out


@register("opt_val_v2")
def opt_val_v2() -> str:
    body = []
    for cfg in V2_CONFIGS:
        top, bot = two_line_cells([(v2s()[s][cfg]["val"], v2s()[s][cfg]["ci"], v2_chosen()[s] == cfg) for s in tasks()])
        body += [f"{esc(V2_TEX[cfg])} & {top} \\\\", f" & {bot} \\\\[1pt]"]
    body.append("\\midrule")
    cells = []
    for s in tasks():
        k = v2_chosen()[s]
        cells.append(esc(k.split(":")[0]) if ":" in k else esc(k))
    body.append("Chosen & " + " & ".join(cells) + " \\\\")
    return float_table("l" + "r" * len(tasks()), task_header(), body,
                       "GEPA with demonstrations and calibration: validation primary metric ($\\times$100) with 95\\% percentile "
                       "bootstrap CIs (" + boot() + " resamples over validation examples). Bold: the configuration chosen by "
                       "the pre-registered rule, which also considers the seed runs (Table~\\ref{tab:opt-seeds}). C, D and E are "
                       "selected on this split and optimistically biased. A and B involve no validation choice.",
                       "tab:opt-val-v2")


@register("opt_v2_chosen")
def opt_v2_chosen() -> str:
    body = []
    for s in tasks():
        k = v2_chosen()[s]
        e = v2s()[s][k]
        cand = rjson(e["run"], "pareto.json")["representatives"][e["representative"]]["candidate"]
        body.append(" & ".join([esc(SHORT[s]), esc(k.split(":")[0]), tt(e["representative"]) + (f" ({esc(cand_index(cand))})" if "/" in cand else ""),
                                pct(e["val"]), f"{e['mean_calls']:.2f}"]) + " \\\\")
    return center("lllrr", "Task & Config. & Representative & Val & Calls \\\\", body,
                  pre="\\setlength{\\tabcolsep}{2.5pt}")


@inline("v2ProtocolDate")
def _v2pd() -> str:
    return v2_prereg_sentence()


def v2_prereg_sentence() -> str:
    first_run = min(parse_time(rjson(r, "run.json")["started_at"]) for r in round_runs("v2-"))
    _, when = commit("second protocol")
    first_test = min(parse_time(x["consumed_at"]) for x in test_ledger() if x["run_id"].startswith("v2-"))
    _, fwhen = commit("second protocol, choice fixed")
    if not fwhen < first_test:
        raise ValueError("v2 chosen table filled after the first v2 test")
    rel = "after" if when > first_run else "before"
    return (f"The second protocol was first recorded at {utc(when)} UTC, {rel} its first validation run started "
            f"({utc(first_run)} UTC). Its chosen-configuration table was filled in at {utc(fwhen)} UTC, "
            f"before its first test consumption ({utc(first_test)} UTC)")


# --------------------------------------------------------------------------------------------
# v3
# --------------------------------------------------------------------------------------------
@cache
def v3s() -> dict[str, Any]:
    return load_json(f"{STUDY}/v3-dev-summary.json")


def v3_rule(slug: str) -> str:
    margin = float(v3_margin())
    chosen = None
    for label in V3_CHOICES:
        if label not in v3s()[slug]:
            continue
        if chosen is None or 100 * (v3s()[slug][label]["val"] - v3s()[slug][chosen]["val"]) >= margin:
            chosen = label
    return chosen


@cache
def v3_chosen() -> dict[str, str]:
    out = {}
    for s in tasks():
        k = v3_rule(s)
        if k != v3s()[s]["chosen"]:
            raise ValueError(f"{s}: rule {k} vs summary {v3s()[s]['chosen']}")
        e = v3s()[s][k]
        if V3_REGISTERED.get(e["run"]) != (e["representative"], e["candidate"]):
            raise ValueError(f"{s}: chosen row not among the recorded choices")
        t = load_json(TEST_JSON)["tasks"][s]["rows"]["**v3 chosen (headline)**"]
        if (t["run"], t["rep"], t["candidate"]) != (e["run"], e["representative"], e["candidate"]):
            raise ValueError(f"{s}: tested v3 row is not the chosen configuration")
        out[s] = k
    return out


V3_TEX = {"v2-A: seed + class bias (reference)": "ref.: paper's q. + class bias", "zero: paper seed": "zero: paper's question",
          "zero: jev-native seed": "zero: rephrased (N)", "zero: decomposed seed": "zero: decomposed seed",
          "A: seed + readout head": "A: paper's q. + head", "head: paper seed (± demos)": "head: paper's q. ($\\pm$ demos)",
          "head: jev-native seed": "head: rephrased seed", "head: decomposed seed": "head: decomposed seed",
          "search from paper": "search from paper's q.", "search from jev-native": "search from rephrased",
          "search from decomposed": "search from decomposed", "P: population-opt": "P: optimized questions",
          "P+head: stacked readout": "P+head: stacked LR"}


@register("opt_val_v3")
def opt_val_v3() -> str:
    body = []
    for cfg in V3_CONFIGS:
        top, bot = two_line_cells([(v3s()[s][cfg]["val"], v3s()[s][cfg]["ci"], v3_chosen()[s] == cfg) for s in tasks()])
        body += [f"{V3_TEX[cfg] if '$' in V3_TEX[cfg] else esc(V3_TEX[cfg])} & {top} \\\\", f" & {bot} \\\\[1pt]"]
        if cfg in ("v2-A: seed + class bias (reference)", "zero: decomposed seed", "head: decomposed seed", "search from decomposed"):
            body.append("\\midrule")
    return float_table("l" + "r" * len(tasks()), task_header(), body,
                       "Third pre-registration (question optimization): validation primary metric "
                       "($\\times$100) with 95\\% percentile bootstrap CIs. \\emph{ref.}: configuration A of GEPA with "
                       "demonstrations and calibration. \\emph{paper's q.}: the paper's question. \\emph{zero}: a seed exactly as written (the kind's seed "
                       "with the best zero-shot train score). \\emph{head}: the kind's best seed by train CV with a logistic regression on its answers. "
                       "\\emph{search}: the best program by train CV descending from that kind. P and P+head are chosen on this "
                       "split among $\\leq$" + INLINE["popValCands"]() + " and $\\leq$" + INLINE["stackCands"]() + " candidates (optimistic). Bold: "
                       "chosen by the pre-registered rule. All rows except \\emph{zero} are ``Jev + classifier on the answers'' systems.",
                       "tab:opt-val")


@register("opt_v3_chosen")
def opt_v3_chosen() -> str:
    body = []
    for s in tasks():
        k = v3_chosen()[s]
        e = v3s()[s][k]
        body.append(" & ".join([esc(SHORT[s]), esc(k.split(":")[0]), tt(e["representative"]) + f" ({cand_index(e['candidate'])})",
                                pct(v3s()[s]["A: seed + readout head"]["val"]), pct(v3s()[s]["P: population-opt"]["val"]),
                                pct(v3s()[s]["P+head: stacked readout"]["val"])]) + " \\\\")
    return center("lllrrr", "Task & Chosen & Representative & A & P & P+head \\\\", body,
                  pre="\\setlength{\\tabcolsep}{2.5pt}")


@cache
def v3_seed_table() -> dict[str, list[list[str]]]:
    """Per-seed zero-shot rows of results/GEPA_v3_val.md (analyze_v3.py; val needs per-example predictions)."""
    text = (ROOT / "results/GEPA_v3_val.md").read_text()
    out: dict[str, list[list[str]]] = {}
    titles = {"contempo-tbdense": "ConTempo TB-Dense", "ezstance-subtaskA": "EZ-STANCE Subtask A",
              "smarter-implicit-hate": "SMARTER Implicit Hate", "edit-intent": "Edit intent (Re3-Sci)", "inli": "INLI",
              "dress-new-content": "DREsS content", "dress-new-organization": "DREsS organization",
              "dress-new-language": "DREsS language"}
    for s in tasks():
        sec = text.split(f"## {titles[s]} (", 1)[1].split("\n## ", 1)[0]
        seed = sec.split("| Seed | Kind |", 1)[1]
        rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in seed.splitlines()[2:] if ln.startswith("|")]
        # cross-check the zero-shot train score against the candidate metadata
        cand_dir = ROOT / STUDY / f"v3-{s}-P" / "jev-population-opt" / "candidates"
        train = {m["seed"]: m["train_zero"] for m in (json.loads(p.read_text()) for p in cand_dir.glob("*.json"))
                 if "zero" in m.get("tags", [])}
        for r in rows:
            if f"{100 * train[r[0]]:.1f}" != r[2]:
                raise ValueError(f"{s}/{r[0]}: GEPA_v3_val.md train {r[2]} != candidate {train[r[0]]}")
        out[s] = rows
    return out


@register("opt_v3_seeds")
def opt_v3_seeds() -> str:
    seeds = [r[0] for r in v3_seed_table()[tasks()[0]]]
    panels = []
    for half in (tasks()[:4], tasks()[4:]):
        for s in half:
            if [r[0] for r in v3_seed_table()[s]] != seeds:
                raise ValueError(f"{s}: seed names differ")
        head = ("& " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{esc(SHORT[s])}}}" for s in half) + " \\\\\n"
                + "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(half))) + "\n"
                + "Seed & " + " & ".join("train & val & CV" for _ in half) + " \\\\")
        body = []
        for i, name in enumerate(seeds):
            cells = ["paper's q." if name == "paper" else esc(name.replace("jev-native", "rephrased"))]
            for s in half:
                r = v3_seed_table()[s][i]
                cells += [r[2], r[3] or "--", r[4] or "--"]
            body.append(" & ".join(cells) + " \\\\")
        panels.append("\n".join([f"\\begin{{tabular}}{{l{'rrr' * len(half)}}}", "\\toprule", head, "\\midrule", *body,
                                 "\\bottomrule", "\\end{tabular}"]))
    return ("\\begin{table*}[!tp]\n\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n" + "\n\\par\\medskip\n".join(panels)
            + "\n\\caption{Every seed program of question optimization run exactly as written (zero-shot, no fitted number): "
              "primary metric ($\\times$100) on the train fit slice and on validation, and the train cross-validation score "
              "of the same seed with a logistic regression on its answers (train scores checked against the candidate metadata). \\emph{paper's q.}\\ is "
              "the paper's question, \\emph{rephrased-$i$} the rephrased questions, \\emph{decomposed-$i$} the decomposed "
              "seeds. The rephrased row of Table~\\ref{tab:opt-val} is the rephrased question with the best train score.}\n\\label{tab:opt-v3-seeds}\n\\end{table*}\n")


@register("opt_v3_native_seed")
def opt_v3_native_seed() -> str:
    """Which Jev-native seed the N run used, per task (best zero-shot train score; asserted)."""
    parts = []
    for s in tasks():
        rows = [r for r in v3_seed_table()[s] if r[1] == "jev-native"]
        best = max(rows, key=lambda r: float(r[2]))
        parts.append(f"{esc(SHORT[s])} {esc(best[0].rsplit('-', 1)[1])}")
    return ", ".join(parts)


# --------------------------------------------------------------------------------------------
# tailored
# --------------------------------------------------------------------------------------------
@cache
def ts() -> dict[str, Any]:
    return load_json(f"{STUDY}/tailored-dev-summary.json")


def ts_key(slug: str, prefix: str) -> str:
    ks = [k for k in ts()[slug] if k.startswith(prefix)]
    if len(ks) != 1:
        raise KeyError(f"{slug}: {prefix} -> {ks}")
    return ks[0]


TAILORED_ROWS = [("v2-A: seed + class bias", "paper's q. + class bias (ref.)"), ("v2 best", "GEPA+d/c (selected)"),
                 ("v3 A: seed + readout head", "paper's q. + LR on answers"), ("v3 chosen", "optimized questions"),
                 (None, None),
                 ("tailored", "Jev-adapted GEPA (fitted)"), ("tailored-zero", "Jev-adapted GEPA (as written)"),
                 ("tailored-head", "Jev-adapted GEPA + head"), (None, None),
                 ("tailored, paper start", "best fitted, paper's-q. start"), ("tailored, jev-native start", "best fitted, rephrased start"),
                 ("tailored-zero, paper start", "best as written, paper's-q. start"),
                 ("tailored-zero, jev-native start", "best as written, rephrased start"), (None, None),
                 ("seed as written: paper", "as written: paper's q."), ("seed as written: jev-native", "as written: rephrased"),
                 ("seed fitted: paper", "fitted: paper's q."), ("seed fitted: jev-native", "fitted: rephrased")]
V1_VAL_TEX = {"v1 zero": "paper's q.", "v1 text-opt": "GEPA text", "v1 program-opt": "GEPA prog."}


@register("opt_val_tailored")
def opt_val_tailored() -> str:
    body = []
    for v1 in ("v1 zero", "v1 text-opt", "v1 program-opt"):
        body.append(f"{esc(V1_VAL_TEX[v1])} (point) & " + " & ".join(pct(ts()[s]["v1"][v1]) for s in tasks()) + " \\\\")
    for key, label in TAILORED_ROWS:
        if key is None:
            body.append("\\midrule")
            continue
        es = []
        for s in tasks():
            k = key if key in ts()[s] else ts_key(s, key + " (")
            es.append((ts()[s][k]["val"], ts()[s][k]["ci"], False))
        top, bot = two_line_cells(es)
        body += [f"{esc(label)} & {top} \\\\", f" & {bot} \\\\[1pt]"]
    return float_table("l" + "r" * len(tasks()), task_header("Row"), body,
                       "Jev-adapted GEPA: validation results ($\\times$100, 95\\% percentile bootstrap CIs). Reference rows: "
                       "\\emph{paper's q.}\\ = the paper's question, \\emph{GEPA text/prog.}\\ = GEPA text/program search (point values, "
                       "these runs stored no per-example validation predictions), \\emph{GEPA+d/c} = GEPA with demonstrations and "
                       "calibration. The three "
                       "Jev-adapted GEPA rows are chosen on this split (optimistic). \\emph{start}: the program the search "
                       "started from (the paper's question or a rephrased question). Seed rows involve no choice.",
                       "tab:opt-val-tailored")


START_TEX = {"paper": "paper's q.", "jev-native": "rephrased"}


@register("opt_tailored_budget")
def opt_tailored_budget() -> str:
    body = []
    for s in tasks():
        r = f"tailored-{s}"
        cfg = rconfig(r)["optimizer"]
        starts = json.loads((ROOT / STUDY / r / "jev-tailored" / "starts.json").read_text())["starts"]
        omp = sum(v["omp_cost"] for v in starts.values())
        if abs(omp - omp_cost(r)) > 0.01:
            raise ValueError(f"{r}: starts.json OMP {omp} vs usage {omp_cost(r)}")
        jev = sum(v["jev_cost"] for v in starts.values())
        v1p = run(s, "program")["opt_cost"]
        nat = starts["jev-native"]["seed"]
        chosen = ts()[s]["tailored"]["candidate"]
        idx = int(cand_index(chosen))
        start = next(k for k, v in starts.items() if idx in v["indices"])
        body.append(" & ".join([esc(SHORT[s]), f"{v1p:.2f}", f"{cfg['max_reflection_cost']:.2f}", f"{omp:.2f}",
                                f"{cfg['tailored']['max_jev_cost']:.2f}", f"{jev:.2f}",
                                f"{len(starts['paper']['indices'])}/{len(starts['jev-native']['indices'])}",
                                esc(nat.rsplit('-', 1)[1]), START_TEX[start]]) + " \\\\")
    head = ("& GEPA prog. & \\multicolumn{2}{c}{OMP \\$} & \\multicolumn{2}{c}{Fresh Jev \\$} & Cand. & Rephr. & Tested \\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
            "Task & OMP \\$ & cap & spent & cap & spent & A/N & seed & start \\\\")
    return float_table("lrrrrrrll", head, body,
                       "Jev-adapted GEPA budgets (run configuration and per-start records). OMP cap = the task's "
                       "GEPA program search spend rounded to cents (column 2 for comparison). Caps are checked between GEPA "
                       "iterations, so spend may exceed them by one iteration. \\emph{Cand.\\ A/N}: candidates of the "
                       "paper's-question and rephrased starts. \\emph{Rephr.\\ seed}: the rephrased question with the best "
                       "zero-shot train score. \\emph{Tested start}: start of the tested (best fitted) candidate.",
                       "tab:opt-tailored-budget", star=False, pre="\\setlength{\\tabcolsep}{2pt}")


@inline("tailoredTestTimes")
def _ttt() -> str:
    rows = history()["tailored_tests"]  # [commit, time] that added each tailored-<task>/test_consumed.json
    if len(rows) != len(tasks()):
        raise ValueError("tailored test records missing")
    times = sorted(parse_time(t) for _, t in rows)
    return f"between {utc(times[0])} and {utc(times[-1])} UTC"


# --------------------------------------------------------------------------------------------
# Test round
# --------------------------------------------------------------------------------------------
@cache
def test_json() -> dict[str, Any]:
    return load_json(TEST_JSON)


def program_of(run_id: str, rep: str) -> dict[str, Any]:
    return yaml.safe_load((ROOT / STUDY / run_id / f"{rep}.yaml").read_text())


def default_cutoffs(n: int) -> list[float]:
    return [i + 0.5 for i in range(n)]


def system_kind(p: dict[str, Any]) -> str:
    """H = readout head, F = other fitted numbers that change predictions, Z = none (temperature ignored)."""
    if p.get("kind") == "ensemble":
        return "F" if p.get("label_bias") or p.get("cutoffs") else "Z"
    nodes = p["nodes"].values()
    if any(n.get("readout") for n in nodes):
        return "H"
    for n in nodes:
        if n.get("biases") or (n.get("aggregation") or {}).get("thresholds"):
            return "F"
        if n.get("cutoffs") is not None and [round(c, 2) for c in n["cutoffs"]] != default_cutoffs(len(n["cutoffs"])):
            return "F"
        if n.get("kind") == "branch" and n.get("threshold", 0.5) != 0.5:
            return "F"
    return "Z"


def row_val(slug: str, key: str) -> float | None:
    r = test_json()["tasks"][slug]["rows"][key]
    if key.startswith("v1"):
        c = {"v1 zero-shot": "zero", "v1 text-opt": "text", "v1 program-opt": "program"}[key]
        return top_val(slug, c)
    if key == "v2 chosen":
        e = v2s()[slug][v2_chosen()[slug]]
    elif key.startswith("**v3"):
        e = v3s()[slug][v3_chosen()[slug]]
    elif key.startswith("Jev-native"):
        e = v3s()[slug]["zero: jev-native seed"]
    else:
        e = ts()[slug]["tailored"]
    if (e["run"], e["representative"]) != (r["run"], r["rep"]):
        raise ValueError(f"{slug}/{key}: val entry {e['run']} {e['representative']} is not the tested one")
    return e["val"]


def rep_txt(r: dict[str, Any]) -> str:
    c = r["candidate"]
    return rep_tex(r["rep"]) + (f" ({esc(cand_index(c))})" if "/" in c else "")


def delta_cell(r: dict[str, Any], base: dict[str, Any] | None, key: str) -> str:
    if base is None or key not in r:
        return ""
    lo, hi = r[key]
    return f"{spct(r['test'] - base['test'])} {{\\tiny[{spct(lo)}, {spct(hi)}]}}"


@register("opt_test_all")
def opt_test_all() -> str:
    body = []
    pj = load_json("results/paper_numbers.json")
    for i, s in enumerate(tasks()):
        t = test_json()["tasks"][s]
        rows = t["rows"]
        if list(rows) != [k for k, _, _ in TEST_ROWS]:
            raise ValueError(f"{s}: unexpected test rows {list(rows)}")
        method = pj[PAPER_KEY[s]]["method"]
        if i:
            body.append("\\midrule")
        body.append(f"\\multicolumn{{10}}{{l}}{{\\textbf{{{esc(SHORT[s])}}} ({esc(METRIC[s])}, test $n$={test_n(s):,}); "
                    f"paper task-specific {pct(t['paper'])} ({esc(method)}); paper zero-shot LLM {pct(t['paper_llm'])}}} \\\\")
        v1z, v2 = rows["v1 zero-shot"], rows["v2 chosen"]
        best = max(r["test"] for r in rows.values())
        for key, label, _ in TEST_ROWS:
            r = rows[key]
            sysk = system_kind(program_of(r["run"], r["rep"]))
            val = row_val(s, key)
            test = pct(r["test"])
            cells = [esc(label), sysk, rep_txt(r), "--" if val is None else pct(val),
                     f"\\textbf{{{test}}}" if r["test"] == best else test,
                     "" if key == "v1 zero-shot" else delta_cell(r, v1z, "ci_vs_v1 zero-shot"),
                     "" if key == "v2 chosen" else delta_cell(r, v2, "ci_vs_v2 chosen"),
                     f"{r['calls']:.2f}", f"{r['tokens']:,.0f}", f"{r['cost']:.2f}"]
            body.append(" & ".join(cells) + " \\\\")
    head = ("Row & Sys. & Representative & Val & Test & $\\Delta$ paper's q.\\ [95\\% CI] & $\\Delta$ GEPA+d/c [95\\% CI] & "
            "Calls/ex & Tok/ex & Test \\$ \\\\")
    return ("\\begin{table*}[p]\n\\centering\\scriptsize\\setlength{\\tabcolsep}{2.6pt}\n\\begin{tabular}{lllrrllrrr}\n"
            + "\\toprule\n" + head + "\n\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{Every test evaluation of the optimization study, "
            "values $\\times$100. Each row is a different run whose test split was consumed exactly once. \\emph{Sys.}: Z = "
            "zero-shot (no fitted number that changes predictions), F = Jev + fitted numbers (class biases, Score cutoffs, "
            "thresholds, ensemble label biases), H = Jev + logistic regression on the answers, derived from the tested program. \\emph{Val}: "
            "validation primary metric of the same representative (optimistic where it was selected on validation). "
            "$\\Delta$: paired bootstrap difference to the paper's question (\\emph{paper's q.}) or to GEPA with demonstrations "
            "and calibration, selected configuration (\\emph{GEPA+d/c}), on the same test examples (" + boot()
            + " resamples, seed 0, percentile interval). Calls, tokens and \\$ are per test example / for the test evaluation. "
            "Bold: highest test score per task (post hoc, not a selection).}\n\\label{tab:opt-test-all}\n\\end{table*}\n")


@register("opt_valtest")
def opt_valtest() -> str:
    body = []
    diffs: dict[str, list[float]] = {k: [] for k, _, _ in TEST_ROWS}
    for s in tasks():
        cells = [esc(SHORT[s])]
        for key, _, _ in TEST_ROWS:
            r = test_json()["tasks"][s]["rows"][key]
            v = row_val(s, key)
            cells += [pct(v), pct(r["test"])]
            diffs[key].append(v - r["test"])
        body.append(" & ".join(cells) + " \\\\")
    body.append("\\midrule")
    body.append("Mean val$-$test & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{spct(sum(d) / len(d))}}}"
                                                   for d in diffs.values()) + " \\\\")
    short = TEST_SHORT
    head = ("& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{short[k]}}}" for k, _, _ in TEST_ROWS) + " \\\\\n"
            + "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(TEST_ROWS))) + "\n"
            + "Task & " + " & ".join("val & test" for _ in TEST_ROWS) + " \\\\")
    return float_table("l" + "rr" * len(TEST_ROWS), head, body,
                       "Validation vs.\\ test primary metric ($\\times$100) of every tested representative. \\emph{Paper's q.}: "
                       "the paper's question. \\emph{GEPA text/prog.}: GEPA text/program search. \\emph{GEPA+d/c}: GEPA with "
                       "demonstrations and calibration (selected). \\emph{Opt.}: the optimized questions. "
                       "\\emph{Rephrased q.}: rephrased question. \\emph{Jev-adapt.}: Jev-adapted GEPA. Columns chosen on "
                       "validation (all but paper's q.\\ and rephrased q.) are optimistically biased by the selection.", "tab:opt-valtest", pre="\\setlength{\\tabcolsep}{2.4pt}")


def mean_gap(key: str) -> float:
    return sum(row_val(s, key) - test_json()["tasks"][s]["rows"][key]["test"] for s in tasks()) / len(tasks())


@inline("v3OptimismMean")
def _v3opt() -> str:
    return f"{100 * mean_gap('**v3 chosen (headline)**'):.1f}"


@inline("nativeGapMean")
def _natgap() -> str:
    return f"{100 * mean_gap('Jev-native zero-shot (v3 N)'):+.1f}"


@inline("v3VsNative")
def _v3vn() -> str:
    d = mean_gap("**v3 chosen (headline)**") - mean_gap("Jev-native zero-shot (v3 N)")
    if f"{100 * abs(d):.1f}" == "0.0":
        return "about the same as"
    return f"{100 * abs(d):.1f} points {'larger' if d > 0 else 'smaller'} than"


@inline("v3TestAboveVal")
def _v3tav() -> str:
    gaps = {s: row_val(s, "**v3 chosen (headline)**") - test_json()["tasks"][s]["rows"]["**v3 chosen (headline)**"]["test"]
            for s in tasks()}
    below = sorted((g, s) for s, g in gaps.items() if g < 0)
    if not below:
        raise ValueError("no task where v3 test exceeds validation; update the text")
    return (f"{len(below)} of {len(gaps)} tasks, by up to {100 * -below[0][0]:.1f} points "
            f"({esc(SHORT[below[0][1]])})")


@inline("dressLangV1")
def _dlv1() -> str:
    s = "dress-new-language"
    rows = test_json()["tasks"][s]["rows"]
    zv, pv = row_val(s, "v1 zero-shot"), row_val(s, "v1 program-opt")
    zt, pt_ = rows["v1 zero-shot"]["test"], rows["v1 program-opt"]["test"]
    lo, hi = rows["v1 program-opt"]["ci_vs_v1 zero-shot"]
    if not (pv > zv and pt_ < zt and hi < 0):
        raise ValueError("DREsS language v1 program-opt no longer a validation gain and significant test loss")
    return (f"validation {pct(zv)} $\\rightarrow$ {pct(pv)}, test {pct(zt)} $\\rightarrow$ {pct(pt_)}, "
            f"$\\Delta$ {ci_txt(lo, hi, True)}")


@inline("tbAllOptimistic")
def _tbo() -> str:
    s = "contempo-tbdense"
    n = sum(row_val(s, k) > test_json()["tasks"][s]["rows"][k]["test"] for k, _, _ in TEST_ROWS)
    if n != len(TEST_ROWS):
        raise ValueError("TB-Dense validation no longer exceeds test in every row")
    return f"all {n}"


@inline("v3OptimismMax")
def _v3max() -> str:
    gaps = {s: row_val(s, "**v3 chosen (headline)**") - test_json()["tasks"][s]["rows"]["**v3 chosen (headline)**"]["test"]
            for s in tasks()}
    s = max(gaps, key=gaps.get)
    return f"{100 * gaps[s]:.1f} points ({esc(SHORT[s])})"


@register("opt_pilot")
def opt_pilot() -> str:
    """LifeTox pilot (runs/jev-opt/lifetox-real-1): validation per condition and its single test evaluation."""
    d = "lifetox-real-1"
    state = rjson(d, "run.json")
    tc = rjson(d, "test_consumed.json")
    lt = test_json()["lifetox"]
    if tc is None or [x["run_id"] for x in test_ledger()].count(d) != 1:
        raise ValueError("LifeTox pilot: expected exactly one test consumption")
    if (tc["representative"], tc["candidate"], tc["metrics"]["primary"]) != (lt["rep"], lt["candidate"], lt["test"]):
        raise ValueError("LifeTox pilot: test_consumed.json and gepa_v3_test.json disagree")
    pareto = rjson(d, "pareto.json")
    body = []
    for name, c in state["conditions"].items():
        s = c["summary"]
        body.append(f"\\texttt{{{esc(name)}}} & {s['candidates']:,} & {s['metric_calls']:,} & {pct(s['seed_val_quality'])} & "
                    f"{pct(s['best_val_quality'])} \\\\")
    env = rjson(d, "environment.json")
    cfg = rconfig(d)
    pj = load_json("results/paper_numbers.json")["lifetox"]
    audit = load_json("runs/lifetox/scores.json")
    bq = pareto["representatives"]["best-quality"]
    table = center("lrrrr", "Condition & Cand. & Metric calls & Seed val & Best val \\\\", body, pre="\\setlength{\\tabcolsep}{4pt}")
    paper = ", ".join(f"{esc(k)} {pct(v)}" for k, v in list(pj["paper"].items())[:2])
    note = (f"{{\\footnotesize One run with all three conditions (train {env['splits']['n_train']:,}, val "
            f"{env['splits']['n_val']:,}). GEPA per-example validation score ($\\times$100). {omp_calls(d):,} optimizer LLM calls, "
            f"\\${omp_cost(d):.2f}. Tested once: \\texttt{{{esc(tc['representative'])}}} = {tt(tc['candidate'])} (validation "
            f"{METRIC_NAME[cfg['primary_metric']]} {pct(bq['val']['primary'])}), test {METRIC_NAME[cfg['primary_metric']]} "
            f"\\textbf{{{pct(tc['metrics']['primary'])}}} on $n$={tc['metrics']['n']:,}, {tc['metrics']['mean_calls']:.2f} Jev calls "
            f"per example, \\${tc['metrics']['total_cost']:.2f}. For reference (not the same sample): paper, full test set: "
            f"{paper}. The zero-shot evaluation's Jev run on its {audit['n']:,}-example test set: {pct(audit['macro_f1'])}.\\par}}\n")
    return table + note


@inline("lifetoxTestN")
def _ltn() -> str:
    return f"{rjson('lifetox-real-1', 'test_consumed.json')['metrics']['n']:,}"


@inline("lifetoxRep")
def _ltr() -> str:
    return tt(rjson("lifetox-real-1", "test_consumed.json")["representative"])


# --------------------------------------------------------------------------------------------
# Seed variance of the headline configuration (results/seeds.json)
# --------------------------------------------------------------------------------------------
SEEDS_JSON = "results/seeds.json"
SEED_REP = {"top-single": "P", "top-stacked": "P+head"}
HEADLINE = "**v3 chosen (headline)**"


def seed_run_id(slug: str, seed: int) -> str:
    return f"v3-{slug}-P" if seed == 0 else f"v3-{slug}-P-s{seed}"


def seed_rule(val: dict[str, float]) -> str:
    """Seed-rerun rule: start from top-single (P); top-stacked (P+head) replaces it iff >= margin points higher."""
    return "top-stacked" if 100 * (val["top-stacked"] - val["top-single"]) >= float(v3_margin()) else "top-single"


@cache
def seeds() -> dict[str, Any]:
    """results/seeds.json, checked against every run's files, the ledger, the seed-0 validation summary and
    results/gepa_v3_test.json, and the pre-registered selection rule."""
    import statistics
    d = load_json(SEEDS_JSON)
    if abs(d["margin_points"] - float(v3_margin())) > 1e-12:
        raise ValueError("seeds.json margin differs from analyze_v3.py")
    if set(d["tasks"]) != set(tasks()):
        raise ValueError(f"seeds.json tasks {list(d['tasks'])}")
    ledger_rep = {x["run_id"]: x["representative"] for x in test_ledger()}
    reruns = set()
    for s in tasks():
        t = d["tasks"][s]
        head = test_json()["tasks"][s]["rows"][HEADLINE]
        zero = test_json()["tasks"][s]["rows"]["v1 zero-shot"]
        if sorted(t["seeds"]) != ["0", "1", "2"]:
            raise ValueError(f"{s}: seeds {sorted(t['seeds'])}")
        c0 = rconfig(seed_run_id(s, 0))
        for k, e in t["seeds"].items():
            seed, r = int(k), e["run"]
            if r != seed_run_id(s, seed):
                raise ValueError(f"{s}/{seed}: run {r}")
            c = rconfig(r)
            if c["optimizer"]["seed"] != seed or rjson(r, "environment.json")["seeds"]["optimizer"] != seed:
                raise ValueError(f"{r}: optimizer seed is not {seed}")
            if {**c, "optimizer": {**c["optimizer"], "seed": 0}} != c0:
                raise ValueError(f"{r}: configuration differs from seed 0 beyond the optimizer seed")
            if e["representative"] != seed_rule(e["val_all"]) or e["val"] != e["val_all"][e["representative"]]:
                raise ValueError(f"{r}: chosen representative is not the pre-registered rule's")
            tc = rjson(r, "test_consumed.json")
            if (tc["representative"], tc["candidate"], tc["metrics"]["primary"]) != (e["representative"], e["candidate"], e["test"]):
                raise ValueError(f"{r}: seeds.json disagrees with test_consumed.json")
            if ledger_rep.get(r) != e["representative"]:
                raise ValueError(f"{r}: test ledger row missing or for another representative")
            if abs(omp_cost(r) - e["omp"]) > 1e-6 or abs((fresh_cost(r) or -1.0) - e["jev_fresh"]) > 1e-6:
                raise ValueError(f"{r}: seeds.json costs disagree with usage.jsonl / run.json")
            if abs(e.get("delta_vs_zero", e["test"] - zero["test"]) - (e["test"] - zero["test"])) > 1e-12:
                raise ValueError(f"{r}: delta_vs_zero is not test minus the audit question's test score")
            if seed == 0:
                if (head["run"], head["rep"], head["candidate"], head["test"]) != (r, e["representative"], e["candidate"], e["test"]):
                    raise ValueError(f"{s}: seed 0 is not the tested headline row of {TEST_JSON}")
                if (abs(e["val_all"]["top-single"] - v3s()[s]["P: population-opt"]["val"]) > 1e-9
                        or abs(e["val_all"]["top-stacked"] - v3s()[s]["P+head: stacked readout"]["val"]) > 1e-9):
                    raise ValueError(f"{s}: seed-0 validation differs from v3-dev-summary.json")
                if SEED_REP[e["representative"]] != v3_chosen()[s].split(":")[0]:
                    raise ValueError(f"{s}: seed rule does not reproduce the registered seed-0 choice")
                ci = e.get("ci_vs_zero")
                if ci is not None and any(abs(a - b) > 1e-9 for a, b in zip(ci, head["ci_vs_v1 zero-shot"])):
                    raise ValueError(f"{s}: seed-0 CI differs from {TEST_JSON}")
            else:
                reruns.add(r)
                if abs(e["jev_test"] - test_cost(r)) > 0.01:
                    raise ValueError(f"{r}: ledger test cost {e['jev_test']} vs test_consumed {test_cost(r)}")
        vals = [e["val"] for e in t["seeds"].values()]
        tests = [e["test"] for e in t["seeds"].values()]
        for key, got in (("val_mean", statistics.mean(vals)), ("val_sd", statistics.stdev(vals)),
                         ("test_mean", statistics.mean(tests)), ("test_sd", statistics.stdev(tests))):
            if abs(t[key] - got) > 1e-12:
                raise ValueError(f"{s}: {key} {t[key]} vs recomputed {got}")
        if t["n_test"] != 3 or ("zero_shot_test" in t and t["zero_shot_test"] != zero["test"]):
            raise ValueError(f"{s}: n_test or audit-question test score differs")
    if reruns != set(seed_reruns()):
        raise ValueError(f"seed reruns on disk {seed_reruns()} vs seeds.json {sorted(reruns)}")
    cost = d["cost_seeds_1_2"]
    if abs(cost["omp"] - sum(omp_cost(r) for r in reruns)) > 1e-6 or abs(cost["jev_fresh"] - sum(fresh_cost(r) for r in reruns)) > 1e-6:
        raise ValueError("seeds.json cost totals disagree with the runs")
    return d


def seed_entries(s: str) -> list[tuple[int, dict[str, Any]]]:
    return sorted((int(k), e) for k, e in seeds()["tasks"][s]["seeds"].items())


def seeds_have_ci() -> bool:
    has = {("ci_vs_zero" in e) for s in tasks() for _, e in seed_entries(s)}
    if len(has) != 1:
        raise ValueError("seeds.json has CIs for some runs only")
    return has.pop()


@register("opt_seeds_headline")
def opt_seeds_headline() -> str:
    body = []
    ci = seeds_have_ci()
    for i, s in enumerate(tasks()):
        if i:
            body.append("\\midrule")
        zero = test_json()["tasks"][s]["rows"]["v1 zero-shot"]["test"]
        for j, (seed, e) in enumerate(seed_entries(s)):
            rep = e["representative"]
            d = e["test"] - zero
            delta = f"{spct(d)} {{\\tiny[{spct(e['ci_vs_zero'][0])}, {spct(e['ci_vs_zero'][1])}]}}" if ci else spct(d)
            vp, vh = (pct(e["val_all"][k]) for k in ("top-single", "top-stacked"))
            cells = [esc(SHORT[s]) if j == 0 else "", str(seed), f"{esc(SEED_REP[rep])} ({cand_index(e['candidate'])})",
                     f"\\textbf{{{vp}}}" if rep == "top-single" else vp, f"\\textbf{{{vh}}}" if rep == "top-stacked" else vh,
                     pct(e["test"]), delta, usd(e["omp"]), usd(e["jev_fresh"]), usd(test_cost(e["run"]))]
            body.append(" & ".join(cells) + " \\\\")
    head = ("Task & Seed & Chosen & Val P & Val P+head & Test & $\\Delta$ paper's q.\\ " + ("[95\\% CI] " if ci else "")
            + "& Opt.\\ \\$ & Jev fresh \\$ & Test \\$ \\\\")
    ci_txt_ = (f" with its paired bootstrap 95\\% CI on the same test examples ({boot()} resamples, seed 0, percentile "
               "interval, as in Table~\\ref{tab:opt-test-all})") if ci else " (no CI was computed for the seed reruns)"
    return float_table("lllrrrlrrr", head, body,
                       "The optimized questions over optimizer seeds 0, 1 and 2 (values $\\times$100). Seed 0 "
                       "is the pre-registered run of Appendix~\\ref{app:opt-v3} (its test row is the optimized-questions row of "
                       "Table~\\ref{tab:opt-test-all}). Seeds 1 and 2 are reruns under the seed protocol. \\emph{Chosen}: "
                       "the configuration picked on validation by the pre-registered rule, candidate index in parentheses. "
                       "Bold: its validation score. \\emph{Test}: the single test evaluation of the chosen program. "
                       "\\emph{$\\Delta$ paper's q.}: test minus the paper's question's test score" + ci_txt_ + ". "
                       "\\emph{Opt.\\ \\$}: optimizer LLM spend. \\emph{Jev fresh \\$}: Jev spend billed during the "
                       "validation search (answers already in the shared cache, e.g.\\ from seed 0, are not billed again). "
                       "\\emph{Test \\$}: the test evaluation.", "tab:opt-seeds-headline", pre="\\setlength{\\tabcolsep}{3pt}")


@register("opt_seeds_summary")
def opt_seeds_summary() -> str:
    body = []
    for s in tasks():
        t = seeds()["tasks"][s]
        tests = [e["test"] for _, e in seed_entries(s)]
        body.append(" & ".join([esc(SHORT[s]), f"{pct(t['val_mean'])} $\\pm$ {pct(t['val_sd'])}",
                                f"{pct(t['test_mean'])} $\\pm$ {pct(t['test_sd'])}", f"{pct(min(tests))}--{pct(max(tests))}",
                                pct(test_json()["tasks"][s]["rows"]["v1 zero-shot"]["test"])]) + " \\\\")
    head = "Task & Val & Test & Test range & Paper's q.\\ \\\\"
    return float_table("lrrrr", head, body,
                       "The optimized questions, mean $\\pm$ sample standard deviation ($n-1$) over optimizer "
                       "seeds 0--2 of the chosen program's validation and test primary metric (Table~\\ref{tab:opt-data}, "
                       "$\\times$100, $n$=3 per task), the range of the three test scores, and the paper's question's test "
                       "score on the same examples.",
                       "tab:opt-seeds-summary", star=False, pre="\\setlength{\\tabcolsep}{3pt}")


@inline("seedCodeNote")
def _scn() -> str:
    shas = {rjson(r, "environment.json")["jev_opt_git_sha"] for r in seed_reruns()}
    if len(shas) != 1:
        raise ValueError(f"seed reruns recorded several commits {shas}")
    sha, base = shas.pop(), round_commit("v3-")
    if sha == base:
        return f"as of {code_time(sha)}, the version of the seed-0 runs"
    trees = [(history()["trees"][c]["jev-opt/python"], history()["trees"][c]["jev-opt/extension"]) for c in (sha, base)]
    if trees[0] != trees[1]:
        raise ValueError("seed reruns ran different jev-opt code than seed 0")
    return (f"as of {code_time(sha)}, whose extension and worker directories are byte-identical to those of the "
            f"seed-0 runs' version of {code_time(base)}")


@inline("seedsReproduced")
def _srep() -> str:
    seeds()  # asserts the rule reproduces every seed-0 choice
    return f"all {len(tasks())}"


@inline("seedsAGap")
def _sag() -> str:
    gaps = {s: v3s()[s]["P: population-opt"]["val"] - v3s()[s]["A: seed + readout head"]["val"] for s in tasks()}
    if min(gaps.values()) * 100 < float(v3_margin()):
        raise ValueError("A came within the margin of P; the seed rule without A may not reproduce seed 0")
    return f"{100 * min(gaps.values()):.1f}"


@inline("seedsHeadChosen")
def _shc() -> str:
    reps = [e["representative"] for s in tasks() for seed, e in seed_entries(s) if seed]
    return f"{sum(r == 'top-stacked' for r in reps)} of the {len(reps)}"


@inline("seedsTestSd")
def _stsd() -> str:
    sd = {s: seeds()["tasks"][s]["test_sd"] for s in tasks()}
    lo, hi = min(sd, key=sd.get), max(sd, key=sd.get)
    return f"from {pct(sd[lo])} ({esc(SHORT[lo])}) to {pct(sd[hi])} ({esc(SHORT[hi])})"


@inline("seedsValSd")
def _svsd() -> str:
    sd = {s: seeds()["tasks"][s]["val_sd"] for s in tasks()}
    hi = max(sd, key=sd.get)
    return f"{pct(sd[hi])} points ({esc(SHORT[hi])})"


@inline("seedsOptimism")
def _sopt() -> str:
    gaps = [e["val"] - e["test"] for s in tasks() for _, e in seed_entries(s)]
    return f"{100 * sum(gaps) / len(gaps):.1f} points on average over the {len(gaps)} runs"


@inline("seedsVsAudit")
def _sva() -> str:
    """How the 24 per-seed test scores compare with the audit question (sign, and CI where computed)."""
    rows = [(e, e["test"] - test_json()["tasks"][s]["rows"]["v1 zero-shot"]["test"]) for s in tasks() for _, e in seed_entries(s)]
    above = sum(d > 0 for _, d in rows)
    n = len(rows)
    txt = (f"All {n} runs" if above == n else f"{above} of the {n} runs") + " score above the paper's question on test"
    if seeds_have_ci():
        sig = sum(e["ci_vs_zero"][0] > 0 for e, _ in rows)
        neg = sum(e["ci_vs_zero"][1] < 0 for e, _ in rows)
        txt += (f", and the 95\\% CI of the difference lies above zero for {'all of them' if sig == n else sig}"
                + (f" and below zero for {neg}" if neg else ""))
    else:
        txt += " (no CI was computed for the seed reruns)"
    return txt


@inline("seedsCost")
def _scost() -> str:
    rs = seed_reruns()
    opt, fresh, test = (sum(omp_cost(r) for r in rs), sum(fresh_cost(r) for r in rs), sum(test_cost(r) for r in rs))
    return (f"{len(rs)} reruns cost \\${opt:.2f} of optimizer LLM calls ({sum(omp_calls(r) for r in rs):,} calls), \\${fresh:.2f} "
            f"of fresh Jev spend during the validation searches and \\${test:.2f} of Jev at test, \\${opt + fresh + test:.2f} in total")


# --------------------------------------------------------------------------------------------
# Seed variance (v2)
# --------------------------------------------------------------------------------------------
def seed_spreads() -> dict[str, tuple[str, list[float], list[float]]]:
    out = {}
    for s in tasks():
        cond, keys = v2_seed_labels(s)
        base = "C: text-opt" if cond == "C" else "D: program-opt"
        tops = [v2s()[s][base]["val"]] + [v2s()[s][f"{cond} seed {i}"]["val"] for i in (1, 2)]
        ens = [v2s()[s]["E: ensemble-3 (C+D)"]["val"]] + [v2s()[s][f"{cond} seed {i} ensemble-3"]["val"] for i in (1, 2)]
        out[s] = (cond, tops, ens)
    return out


@register("opt_seeds")
def opt_seeds() -> str:
    body = []
    for s, (cond, tops, ens) in seed_spreads().items():
        cells = [esc(SHORT[s]), cond, *(pct(v) for v in tops), f"{100 * (max(tops) - min(tops)):.1f}", *(pct(v) for v in ens),
                 f"{100 * (max(ens) - min(ens)):.1f}"]
        body.append(" & ".join(cells) + " \\\\")
    head = ("& & \\multicolumn{4}{c}{top-quality} & \\multicolumn{4}{c}{ensemble-3} \\\\\n"
            "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\n"
            "Task & Cond. & s0 & s1 & s2 & range & s0$^*$ & s1 & s2 & range \\\\")
    return float_table("llrrrrrrrr", head, body,
                       "Optimizer-seed variance in GEPA with demonstrations and calibration (validation, $\\times$100): the better of C/D by validation "
                       "(\\emph{Cond.}) rerun with optimizer seeds 1 and 2. s0 is the representative of the C+D run "
                       "(\\texttt{top-jev-text-opt} or \\texttt{top-jev-program-opt}). $^*$The seed-0 ensemble is E's "
                       "ensemble-3 over C and D candidates, the seed-1/2 ensembles are over one condition. Range = max $-$ min.",
                       "tab:opt-seeds", star=False, pre="\\setlength{\\tabcolsep}{2.2pt}")


@inline("ezSpread")
def _ez() -> str:
    _, tops, _ = seed_spreads()["ezstance-subtaskA"]
    return f"{100 * (max(tops) - min(tops)):.1f}"


@inline("ezSeeds")
def _ezs() -> str:
    _, tops, _ = seed_spreads()["ezstance-subtaskA"]
    return ", ".join(pct(v) for v in tops)


@inline("maxSpreadTask")
def _mst() -> str:
    sp = {s: max(t) - min(t) for s, (_, t, _) in seed_spreads().items()}
    return esc(SHORT[max(sp, key=sp.get)])


@inline("medianSpread")
def _meds() -> str:
    sp = sorted(max(t) - min(t) for _, t, _ in seed_spreads().values())
    return f"{100 * (sp[len(sp) // 2 - 1] + sp[len(sp) // 2]) / 2:.1f}"


# --------------------------------------------------------------------------------------------
# Lever C
# --------------------------------------------------------------------------------------------
LEVERC_PREV = {"ttp_v3/tram": ("ttp/tram", "p@1"), "ttp_v3/procedures_plus": ("ttp/procedures_plus", "p@1"),
               "ttp_v3/expert": ("ttp/expert", "p@1"), "ttp_v3/expert_full": ("ttp/expert_full", "p@1"),
               "relemb_v3/wikiwd": ("relemb/wikiwd", "f1"), "made_v3": ("made", "macro_f1")}
LEVERC_TEX = {"ttp_v3/tram": "TTP TRAM", "ttp_v3/procedures_plus": "TTP Procedures+", "ttp_v3/expert": "TTP Expert",
              "ttp_v3/expert_full": "TTP Expert (full)", "relemb_v3/wikiwd": "Wiki-WD", "made_v3": "MADE"}


def lc(name: str) -> dict[str, Any]:
    return load_json(f"results/v3_dev/{name}.json")


def boot() -> str:
    """Bootstrap resamples used by every analysis script (asserted equal)."""
    vals = set()
    for rel in ("jev-opt/experiments/analyze_v2.py", "jev-opt/experiments/analyze_test_v3.py", "scripts/fit_v3.py"):
        m = re.search(r"^B = (\d+)", (ROOT / rel).read_text(), re.M)
        vals.add(int(m.group(1)))
    if len(vals) != 1:
        raise ValueError(f"bootstrap resamples differ: {vals}")
    return f"{vals.pop():,}"


register("opt.boot")(boot)


@register("opt_leverc")
def opt_leverc() -> str:
    ttp, wiki, made = lc("ttp"), lc("wikiwd"), lc("made")
    tau = f"{made['chosen_tau']:g}"
    wr = wiki["results"]
    dev = [
        ("TTP (4 test sets)", f"{ttp['n']:,} texts", "P@1", ttp["dev_v1"]["value"], ttp["dev_v3"]["value"],
         ttp["dev_delta"], f"$k$={ttp['chosen']['k']}, $w$={ttp['chosen']['w']:g}"),
        ("Wiki-WD", f"{wiki['n_sentences']:,} sent.", "micro-F1", wr["v2"]["value"], wr[wiki["chosen"]]["cross_fit_bias"]["value"],
         wr[wiki["chosen"]]["delta_vs_v2"], f"round 1 + {wiki['n_biases']} biases"),
        ("MADE", f"{made['n']:,} reports", "macro-F1", made["results"]["v2"]["value"],
         made["results"][f"tau_{tau}"]["cross_fit_thresholds"]["value"], made["results"][f"tau_{tau}"]["delta_vs_v2"],
         f"$\\tau$={tau}, {made['n_per_code_thresholds']} thresholds"),
    ]
    for d in (ttp, wiki, made):
        if not d["selected_for_test"]:
            raise ValueError("lever C configuration not registered")
    body = ["\\multicolumn{7}{l}{\\textit{Development (held out; basic bootstrap interval)}} \\\\"]
    for name, n, metric, base, v3, delta, cfg in dev:
        body.append(" & ".join([name, n, metric, pct(base), pct(v3), f"{spct(delta['delta'])} {ci_txt(delta['lo'], delta['hi'], True)}",
                                cfg]) + " \\\\")
    body += ["\\midrule", "\\multicolumn{7}{l}{\\textit{Test (single run; Zero-shot = the stored zero-shot row)}} \\\\",
             "Test set & $n$ & Metric & Zero-shot & Fitted & Paper & New Jev \\$ \\\\", "\\midrule"]
    lcj = test_json()["lever_c"]
    for task, v in lcj.items():
        prev_task, key = LEVERC_PREV[task]
        prev = load_json(f"runs/{prev_task}/scores.json")[key]
        if abs(round(prev, 3) - v["prev"]) > 5e-4:
            raise ValueError(f"{task}: prev {v['prev']} vs {prev_task} {prev}")
        scores = load_json(f"runs/{task}/scores.json")
        man = load_json(f"runs/{task}/manifest.json")
        if abs(scores[key] - v["test"]) > 1e-12 or abs(man["cost_usd_excluding_reused"] - v["cost"]) > 1e-9:
            raise ValueError(f"{task}: gepa_v3_test.json disagrees with runs/{task}")
        body.append(" & ".join([LEVERC_TEX[task], f"{scores['n']:,}", esc(key), pct(prev), f"\\textbf{{{pct(v['test'])}}}", pct(v["paper"]),
                                f"{man['cost_usd_excluding_reused']:.2f}"]) + " \\\\")
    head = "Task & Dev sample & Metric & Zero-shot & Fitted & $\\Delta$ [95\\% CI] & Configuration \\\\"
    return float_table("lllrrll", head, body,
                       "Candidate reranking and fitted thresholds on the large-label evaluated tasks. Development rows compare "
                       "the frozen zero-shot decision (\\emph{Zero-shot}) with the fitted configuration (\\emph{Fitted}) on held-out "
                       "dev data (cross-fitting). Test rows are the single registered runs, with new Jev spend excluding "
                       "responses reused from the zero-shot runs. No test CI was computed for these rows.",
                       "tab:opt-leverc")


@register("opt_leverc_dev")
def opt_leverc_dev() -> str:
    ttp, wiki, made = lc("ttp"), lc("wikiwd"), lc("made")
    t_rows = []
    for k, w in ttp["best_w_per_k"].items():
        t_rows.append(f"{k} & {pct(ttp['grid_p_at_1'][f'k{k}_w0.0'])} & {w:g} & {pct(ttp['grid_p_at_1'][f'k{k}_w{w:g}' if f'k{k}_w{w:g}' in ttp['grid_p_at_1'] else f'k{k}_w{w}'])} \\\\")
    ttp_tab = center("rrrr", "$k$ & $w$=0 & best $w$ & P@1 \\\\", t_rows, pre="\\setlength{\\tabcolsep}{4pt}")
    w_rows = []
    for name, r in wiki["results"].items():
        if name == "v2":
            continue
        d = r["delta_vs_v2"]
        mark = " (chosen)" if name == wiki["chosen"] else (" (best)" if name == wiki["best"] else "")
        w_rows.append(f"{esc(name)}{mark} & {pct(r['no_bias']['value'])} & {pct(r['cross_fit_bias']['value'])} & "
                      f"{spct(d['delta'])} {ci_txt(d['lo'], d['hi'], True)} \\\\")
    wiki_tab = center("lrrl", "Variant & no bias & biases & $\\Delta$ vs zero-shot \\\\", w_rows, pre="\\setlength{\\tabcolsep}{2.5pt}")
    m_rows = []
    for key, r in made["results"].items():
        if not key.startswith("tau_"):
            continue
        d = r["delta_vs_v2"]
        tau = key[4:]
        mark = "$^\\dagger$" if float(tau) == made["chosen_tau"] else ""
        m_rows.append(f"{tau}{mark} & {r['asked_mean']:.0f} & {pct(r['accept_0.5']['value'])} & {pct(r['cross_fit_thresholds']['value'])} & "
                      f"{spct(d['delta'])} & {ci_txt(d['lo'], d['hi'], True)} & {ci_txt(d['pct_lo'], d['pct_hi'], True)} \\\\")
    made_tab = center("rrrrrll", "$\\tau$ & asked & $P{>}.5$ & thr. & $\\Delta$ & basic & percentile \\\\", m_rows,
                      pre="\\setlength{\\tabcolsep}{2pt}")
    return (f"\\noindent\\textit{{TTP rerank}} ({ttp['n']:,} dev texts, P@1 $\\times$100 by shortlist size $k$, $w$=0 is the "
            f"tie-break fix alone):\n{ttp_tab}\\noindent\\textit{{Wiki-WD}} (dev micro-F1 $\\times$100, {wiki['n_edges']:,} edges, "
            f"zero-shot decision {pct(wiki['results']['v2']['value'])}):\n{wiki_tab}\\noindent\\textit{{MADE}} (dev macro-F1 "
            f"$\\times$100 over the codes, {made['n']:,} reports, zero-shot decision {pct(made['results']['v2']['value'])}, asked = codes "
            f"per report, $^\\dagger$registered):\n{made_tab}")


@inline("leverCDevCost")
def _lcdc() -> str:
    return f"\\${sum(lc(n)['dev_run']['cost_usd'] for n in ('ttp', 'wikiwd', 'made')):.2f}"


@inline("leverCTestCost")
def _lctc() -> str:
    return f"\\${sum(v['cost'] for v in test_json()['lever_c'].values()):.2f}"


@inline("madeTauGap")
def _mtg() -> str:
    m = lc("made")["results"]
    best = f"tau_{lc('made')['best_tau']:g}"
    ch = f"tau_{lc('made')['chosen_tau']:g}"
    return f"{100 * (m[best]['cross_fit_thresholds']['value'] - m[ch]['cross_fit_thresholds']['value']):.1f}"


@inline("madeV2Pct")
def _mv2() -> str:
    v = lc("made")["results"]["v2"]
    return f"{pct(v['value'])} outside its own percentile interval {ci_txt(v['pct_lo'], v['pct_hi'])}"


@inline("madeTauChosen")
def _mtc() -> str:
    return f"{lc('made')['chosen_tau']:g}"


@inline("madeTauBest")
def _mtb() -> str:
    return f"{lc('made')['best_tau']:g}"


@inline("paramShas")
def _pshas() -> str:
    import hashlib
    parts = []
    for name, rel in (("TTP", "data/frozen/ttp/v3_params.json"), ("Wiki-WD", "data/frozen/relemb/wikiwd_v3_params.json"),
                      ("MADE", "data/frozen/made/v3_params.json")):
        parts.append(f"{name} {tt(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()[:8])}")
    return ", ".join(parts)


# --------------------------------------------------------------------------------------------
# Fitted readouts on the audit's zero-shot answers (results/heads.json)
# --------------------------------------------------------------------------------------------
# audit test set -> (display name, dev sample under runs/dev_v3/)
HEAD_SETS = {"contempo/matres": ("ConTempo MATRES", "contempo_matres"), "contempo/tbdense": ("ConTempo TB-Dense", "contempo_tbdense"),
             "smarter/hatexplain": ("SMARTER HateXplain", "smarter_hatexplain"),
             "smarter/latent_hate": ("SMARTER Latent Hate", "smarter_latent_hate"),
             "smarter/implicit_hate": ("SMARTER Implicit Hate", "smarter_implicit_hate")}
# heads.json metric -> (display, key in scores.json)
HEAD_METRIC = {"micro_f1 (VAGUE negative)": ("micro-F1 (VAGUE neg.)", "micro_f1"), "macro_f1": ("macro-F1", "macro_f1")}
FIT_V3 = "scripts/fit_v3.py"


def fit_v3_const(pattern: str) -> str:
    m = re.search(pattern, (ROOT / FIT_V3).read_text(), re.M)
    if not m:
        raise KeyError(f"{pattern} not in {FIT_V3}")
    return m.group(1)


def part_n(v: Any) -> int:
    return v if isinstance(v, int) else v["sampled"]


def sha256_file(rel: str) -> str:
    import hashlib
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


@cache
def heads() -> dict[str, dict[str, Any]]:
    """results/heads.json, checked against the dev fit summary, the dev-sample manifest, the frozen head files, the
    dev runs and the zero-shot and +head test runs."""
    rep = load_json("results/heads.json")
    dev = load_json("results/v3_dev/heads.json")
    man = load_json("data/frozen/v3_dev_manifest.json")
    if list(rep) != list(HEAD_SETS) or set(dev) != set(HEAD_SETS):
        raise ValueError(f"head sets changed: {list(rep)} / {list(dev)}")
    frozen = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "data/frozen/heads").rglob("*.json"))
    if frozen != sorted(f"data/frozen/heads/{k}.json" for k in HEAD_SETS):
        raise ValueError(f"frozen heads {frozen}")
    out = {}
    for k, r in rep.items():
        d, _, sample = dev[k], *HEAD_SETS[k]
        key = HEAD_METRIC[r["metric"]][1]
        if not d["selected_for_test"] or not d["dev_delta"]["lo"] > 0 or d["dev_delta"] != r["dev_delta"]:
            raise ValueError(f"{k}: registration does not follow the dev rule")
        cv = d["cv_fit"]
        if d["chosen"] != r["chosen"] or d["chosen"] != max(cv, key=lambda n: (cv[n], n == "zero_shot", n == "bias")):
            raise ValueError(f"{k}: chosen head is not the CV choice")
        if (r["dev_zero_shot"], r["dev_head"]) != (d["dev_zero_shot"]["value"], d["dev_head"]["value"]) \
                or d["dev_candidates"][d["chosen"]] != d["dev_head"]["value"] or d["dev_candidates"]["zero_shot"] != r["dev_zero_shot"]:
            raise ValueError(f"{k}: dev scores disagree")
        parts = {p: part_n(v) for p, v in man[k]["parts"].items()}
        if parts != {"fit": d["n_fit"], "dev": d["n_dev"]} or man[k]["n"] != d["n_fit"] + d["n_dev"]:
            raise ValueError(f"{k}: dev sample sizes disagree with the manifest")
        dm = load_json(f"runs/dev_v3/{sample}/manifest.json")
        if dm["n_items"] != man[k]["n"] or not (dm["cost_usd"] == d["dev_run"]["cost_usd"] == r["dev_cost_usd"]):
            raise ValueError(f"{k}: dev run disagrees")
        fz = f"data/frozen/heads/{k}.json"
        h = load_json(fz)["head"]
        nl = len(h["labels"])
        n_params = nl * nl + nl if h["kind"] == "lr" else nl - 1
        if sha256_file(fz) != r["head_file_sha256"] or h["kind"] != r["head"] or n_params != r["n_params"]:
            raise ValueError(f"{k}: frozen head disagrees with heads.json")
        if h["kind"] == "lr" and len(h["coef"]) * len(h["coef"][0]) + len(h["intercept"]) != n_params:
            raise ValueError(f"{k}: LR head shape")
        zs, hs = load_json(f"runs/{k}/scores.json"), load_json(f"runs/{k}+head/scores.json")
        hm = load_json(f"runs/{k}+head/manifest.json")
        if (zs[key], hs[key]) != (r["test_zero_shot"]["value"], r["test_head"]["value"]) \
                or not (zs["n"] == hs["n"] == r["n_test"]) or hs["n_changed_from_zero_shot"] != r["n_changed"] \
                or abs(r["test_delta"]["delta"] - (hs[key] - zs[key])) > 1e-12:
            raise ValueError(f"{k}: test scores disagree with heads.json")
        if hs["head"]["sha256"] != r["head_file_sha256"] or hs["head"]["file"] != fz:
            raise ValueError(f"{k}: +head run used another head")
        if hm["responses_reused_from"] != [k] or not (hm["n_responses_reused"] == hm["n_requests"] == r["test_requests"]) \
                or hm["cost_usd_excluding_reused"] != 0 or r["test_cost_usd_new"] != 0:
            raise ValueError(f"{k}: +head run made new Jev requests")
        out[k] = r | {"n_fit": d["n_fit"], "n_dev": d["n_dev"], "fit_labels": man[k]["parts"]["fit"],
                      "dev_labels": man[k]["parts"]["dev"], "test_gold": hs["n_gold"], "k_per_class": man[k].get("k_per_class")}
    return out


def head_kind_tex(r: dict[str, Any]) -> str:
    if r["head"] == "bias":
        return f"bias ({r['n_params']})"
    return f"LR $C$={float(r['chosen'].split('_C', 1)[1]):g} ({r['n_params']})"


@register("opt_heads")
def opt_heads() -> str:
    body = []
    groups: dict[str, list[str]] = {}
    for k, r in heads().items():
        groups.setdefault(r["metric"], []).append(k)
    for i, (metric, ks) in enumerate(groups.items()):
        corpus = {HEAD_SETS[k][0].split(" ", 1)[0] for k in ks}
        if len(corpus) != 1:
            raise ValueError(f"head sets of one metric span corpora {corpus}")
        if i:
            body.append("\\midrule")
        body.append(f"\\multicolumn{{11}}{{l}}{{\\textit{{{corpus.pop()}, {esc(HEAD_METRIC[metric][0])}}}}} \\\\")
        for k in ks:
            r = heads()[k]
            dd, td = r["dev_delta"], r["test_delta"]
            body.append(" & ".join([esc(HEAD_SETS[k][0].split(" ", 1)[1]), f"{r['n_fit']:,} / {r['n_dev']:,}", head_kind_tex(r),
                                    pct(r["dev_zero_shot"]), pct(r["dev_head"]),
                                    f"{spct(dd['delta'])} {{\\tiny{ci_txt(dd['lo'], dd['hi'], True)}}}", pct(r["test_zero_shot"]["value"]),
                                    f"\\textbf{{{pct(r['test_head']['value'])}}}", f"{spct(td['delta'])} {{\\tiny{ci_txt(td['lo'], td['hi'], True)}}}",
                                    f"{r['n_changed']:,} / {r['n_test']:,}", usd(r["dev_cost_usd"])]) + " \\\\")
    head = ("& & & \\multicolumn{3}{c}{Dev part} & \\multicolumn{3}{c}{Test (evaluated test set)} & & \\\\\n"
            "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\n"
            "Test set & Fit / dev & Head (params) & Zero & Head & $\\Delta$ [95\\% CI] & Zero & Head & $\\Delta$ [95\\% CI] & "
            "Changed & Dev \\$ \\\\")
    return float_table("lllrrlrrlrr", head, body,
                       "Calibration and logistic regression on the zero-shot evaluation's answers (``Jev + classifier on the answers'', values $\\times$100). "
                       "\\emph{Fit / dev}: items in the two parts of the frozen dev sample. \\emph{Head}: the classifier chosen "
                       "by cross-validation on the fit part (bias: per-class log-biases, LR: L2 multinomial logistic "
                       "regression with its $C$) and its number of fitted parameters. \\emph{Dev part}: zero-shot argmax "
                       "(\\emph{Zero}) and head on the held-out dev part, $\\Delta$ with the paired bootstrap 95\\% CI of the "
                       "registration rule. \\emph{Test}: the stored zero-shot row and the single run with the classifier "
                       "on the same test items, $\\Delta$ with a paired bootstrap 95\\% CI over test items "
                       f"({boot()} resamples, seed 0, basic interval). \\emph{{Changed}}: test items whose label the head "
                       "changed. \\emph{Dev \\$}: Jev cost of the dev run. The test runs made no new Jev request.",
                       "tab:opt-heads", pre="\\setlength{\\tabcolsep}{2.5pt}")


for _k, _f in {
    "headsN": lambda: str(len(heads())),
    "headsFloor": lambda: fit_v3_const(r"^FLOOR = ([0-9.]+)"),
    "headsLrC": lambda: "$\\{" + ", ".join(f"{float(x):g}" for x in fit_v3_const(r"^LR_CS = \(([^)]*)\)").split(",")) + "\\}$",
    "headsFolds": lambda: fit_v3_const(r"group_folds\(group\[fit\], (\d+), SEED\)"),
    "headsDevCost": lambda: f"\\${sum(r['dev_cost_usd'] for r in heads().values()):.2f}",
    "headsTestReq": lambda: f"{sum(r['test_requests'] for r in heads().values()):,}",
    "headsShas": lambda: ", ".join(f"{esc(HEAD_SETS[k][0].split(' ', 1)[1])} {tt(r['head_file_sha256'][:8])}" for k, r in heads().items()),
}.items():
    INLINE[_k] = cache(_f)
    register(f"opt.{_k}")(INLINE[_k])


@inline("headsBalanced")
def _hbal() -> str:
    if "balanced=(kind == \"macro\")" not in (ROOT / FIT_V3).read_text():
        raise ValueError("LR class weighting changed in fit_v3.py")
    return "with balanced class weights for the macro-F1 sets"


@inline("headsDevSizes")
def _hds() -> str:
    return "; ".join(f"{esc(HEAD_SETS[k][0])} {r['n_fit']:,} / {r['n_dev']:,}" for k, r in heads().items())


@inline("headsWeakest")
def _hw() -> str:
    k = min(heads(), key=lambda k: heads()[k]["dev_delta"]["lo"])
    return f"{esc(HEAD_SETS[k][0])}'s dev $\\Delta$ lower bound was {spct(heads()[k]['dev_delta']['lo'])}"


@inline("headsTestSummary")
def _hts() -> str:
    hs = heads()
    above = sum(r["test_delta"]["lo"] > 0 for r in hs.values())
    lo = min(hs, key=lambda k: hs[k]["test_delta"]["delta"])
    hi = max(hs, key=lambda k: hs[k]["test_delta"]["delta"])
    return (f"On test the 95\\% CI of the head's difference to the zero-shot argmax lies above zero for {above} of the {len(hs)} "
            f"sets. The difference ranges from {spct(hs[lo]['test_delta']['delta'])} ({esc(HEAD_SETS[lo][0])}) to "
            f"{spct(hs[hi]['test_delta']['delta'])} ({esc(HEAD_SETS[hi][0])}) points")


@inline("headsBalance")
def _hb() -> str:
    def share(c: dict[str, int]) -> str:
        return f"{100 * max(c.values()) / sum(c.values()):.0f}"

    def labels(v: Any) -> dict[str, int]:
        return v["labels"] if isinstance(v, dict) and "labels" in v else v

    smarter, contempo = [], []
    for k, r in heads().items():
        name = esc(HEAD_SETS[k][0].split(" ", 1)[1])
        if k.startswith("smarter/"):
            t = r["test_gold"].values()
            f, dv = labels(r["fit_labels"]).values(), labels(r["dev_labels"]).values()
            if max(f) > r["k_per_class"] or max(dv) > r["k_per_class"]:
                raise ValueError(f"{k}: fit/dev part exceeds k per class")
            smarter.append(f"{name} fit {min(f):,}--{max(f):,}, dev {min(dv):,}--{max(dv):,}, test {min(t):,}--{max(t):,}")
        else:
            fl = load_json("data/frozen/v3_dev_manifest.json")[k]["labels"]
            contempo.append(f"{name} {share(fl['fit'])}/{share(fl['dev'])}/{share(r['test_gold'])}\\%")
    return (f"The SMARTER fit, dev and test samples are class-balanced samples (items per label: {'; '.join(smarter)}), so "
            f"their label priors are close. The ConTempo samples are not class-balanced (largest label's share in fit/dev/test: "
            f"{', '.join(contempo)})")


# --------------------------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------------------------
def cost_cells(run_id: str, used: float | None) -> list[str]:
    return [f"{omp_calls(run_id):,}", usd(omp_cost(run_id)), usd(fresh_cost(run_id)), usd(used), f"{wall_h(run_id):.2f}",
            usd(test_cost(run_id))]


COST_HEAD = "Task & Run & Opt.\\ calls & Opt.\\ \\$ & Jev fresh \\$ & Jev used \\$ & Wall h & Test \\$ \\\\"
COST_CAP = ("\\emph{Opt.}: optimizer LLM calls and spend (host ledger). \\emph{Jev fresh}: spend billed during the run "
            "(recorded from the third pre-registration on). \\emph{Jev used}: distinct train/validation requests of the run's response ledger at "
            "Jev's reported cost, including requests served from the shared cache. \\emph{Wall}: optimization only. "
            "\\emph{Test \\$}: the single test evaluation.")


def check_md(run_id: str, used: float, rel: str) -> None:
    rec = md_costs(rel).get(run_id)
    if rec is not None and abs(rec["Jev used $"] - used) > 0.006:
        raise ValueError(f"{run_id}: Jev used {used:.4f} vs {rel} {rec['Jev used $']}")


def round_total(runs: list[str], used: dict[str, float]) -> dict[str, float]:
    return {"calls": sum(omp_calls(r) for r in runs), "opt": sum(omp_cost(r) for r in runs),
            "fresh": sum(fresh_cost(r) or 0.0 for r in runs) if all(fresh_cost(r) is not None for r in runs) else None,
            "used": sum(used[r] for r in runs), "wall": sum(wall_h(r) for r in runs),
            "test": sum(test_cost(r) or 0.0 for r in runs)}


@cache
def used_all() -> dict[str, float]:
    out = {}
    for r in all_run_names():
        if r.startswith("v2-"):
            out[r] = v2_jev_used(r)
        else:
            out[r] = jev_used(r)
            if r.startswith("v3-"):
                check_md(r, out[r], "results/GEPA_v3_val.md")
            if r.startswith("tailored-"):
                check_md(r, out[r], "results/GEPA_tailored_val.md")
    return out


V1_LABEL = {"zero": "paper's q.", "text": "GEPA text", "program": "GEPA prog."}


@register("opt_cost")
def opt_cost() -> str:
    used = used_all()
    body = []
    for i, s in enumerate(tasks()):
        for j, c in enumerate(CONDS):
            r = f"study-{s}-{c}"
            body.append(" & ".join([esc(SHORT[s]) if j == 0 else "", V1_LABEL[c], *cost_cells(r, used[r])]) + " \\\\")
    body.append("\\midrule")
    for s in tasks():
        r = f"tailored-{s}"
        body.append(" & ".join([esc(SHORT[s]), "Jev-adapt.", *cost_cells(r, used[r])]) + " \\\\")
    body.append("\\midrule")
    body.append(" & ".join(["LifeTox", "pilot", *cost_cells("lifetox-real-1", jev_used("lifetox-real-1"))]) + " \\\\")
    body += ["\\midrule", "\\multicolumn{8}{l}{\\textit{Totals per configuration}} \\\\"]
    for name, runs in (("v1", [f"study-{s}-{c}" for s in tasks() for c in CONDS]), ("v2", list(round_runs("v2-"))),
                       ("v3", list(round_runs("v3-"))), ("v3 P seeds", list(seed_reruns())), ("tailored", list(round_runs("tailored-")))):
        t = round_total(runs, used)
        body.append(" & ".join([ROUND_TEX[name], f"{len(runs)} runs", f"{t['calls']:,}", usd(t["opt"]), usd(t["fresh"]), usd(t["used"]),
                                f"{t['wall']:.1f}", usd(t["test"])]) + " \\\\")
    sc = sum(g["optimizer_cost"] for g in seed_generations().values())
    body.append(" & ".join([ROUND_TEX["v3 seeds"], "\\texttt{/jev-seeds}", f"{sum(seed_calls().values()):,}", usd(sc), "--", "--", "--", "--"]) + " \\\\")
    dev = sum(lc(n)["dev_run"]["cost_usd"] for n in ("ttp", "wikiwd", "made"))
    body.append(" & ".join([ROUND_TEX["lever C"], "dev runs", "--", "--", usd(dev), "--", "--", usd(sum(v['cost'] for v in test_json()['lever_c'].values()))]) + " \\\\")
    hd = heads()
    body.append(" & ".join([ROUND_TEX["heads"], "dev runs", "--", "--", usd(sum(r["dev_cost_usd"] for r in hd.values())), "--", "--",
                            usd(sum(r["test_cost_usd_new"] for r in hd.values()))]) + " \\\\")
    return float_table("llrrrrrr", COST_HEAD, body,
                       "Cost per run of GEPA text/program search (\\emph{paper's q.}, \\emph{GEPA text}, \\emph{GEPA prog.}), "
                       "Jev-adapted GEPA (\\emph{Jev-adapt.}) and the LifeTox pilot, and totals per configuration (runs of GEPA "
                       "with demonstrations and calibration, of the optimized questions and of its seed reruns: "
                       "Tables~\\ref{tab:opt-cost-v2}, \\ref{tab:opt-cost-v3} and~\\ref{tab:opt-seeds-headline}). " + COST_CAP
                       + " The test \\$ of reranking and thresholds and of the classifiers on zero-shot answers is new spend "
                       "only (responses reused from the zero-shot runs excluded). Their dev runs are listed under Jev fresh.", "tab:opt-cost",
                       pos="p", pre="\\setlength{\\tabcolsep}{4pt}")


def v2_label(run_id: str, slug: str) -> str:
    return run_id[len(f"v2-{slug}-"):]


@register("opt_cost_v2")
def opt_cost_v2() -> str:
    used = used_all()
    body = []
    for i, s in enumerate(tasks()):
        if i:
            body.append("\\midrule")
        runs = [r for r in round_runs("v2-") if r.startswith(f"v2-{s}-")]
        for j, r in enumerate(sorted(runs, key=lambda x: (len(v2_label(x, s)) > 2, v2_label(x, s)))):
            body.append(" & ".join([esc(SHORT[s]) if j == 0 else "", esc(v2_label(r, s)), *cost_cells(r, used[r])]) + " \\\\")
    return float_table("llrrrrrr", COST_HEAD, body,
                       "Cost per run of GEPA with demonstrations and calibration (A, B, the C+D run, and the two seed reruns "
                       "of the better condition). " + COST_CAP + " Here \\emph{Jev used} is read from the cost table of the "
                       "validation report (Appendix~\\ref{app:opt-repro}).",
                       "tab:opt-cost-v2", pos="p", pre="\\setlength{\\tabcolsep}{4pt}")


@register("opt_cost_v3")
def opt_cost_v3() -> str:
    used = used_all()
    body = []
    for i, s in enumerate(tasks()):
        if i:
            body.append("\\midrule")
        for j, cfg in enumerate(("A", "P", "N")):
            r = f"v3-{s}-{cfg}"
            body.append(" & ".join([esc(SHORT[s]) if j == 0 else "", cfg, *cost_cells(r, used[r])]) + " \\\\")
        g = seed_generations()[s]
        body.append(" & ".join(["", "seeds", f"{seed_calls()[s]:,}", usd(g["optimizer_cost"]), "--", "--", "--", "--"]) + " \\\\")
    return float_table("llrrrrrr", COST_HEAD, body,
                       "Cost per run of question optimization (A: the paper's question + head, P: optimized "
                       "questions, N: rephrased question) and of the \\texttt{/jev-seeds} generation per task. " + COST_CAP,
                       "tab:opt-cost-v3", pos="p", pre="\\setlength{\\tabcolsep}{4pt}")


@inline("allOptCost")
def _aoc() -> str:
    runs = all_run_names()
    return f"\\${sum(omp_cost(r) for r in runs) + sum(g['optimizer_cost'] for g in seed_generations().values()):,.0f}"


@inline("allJevTestCost")
def _ajt() -> str:
    return f"\\${sum(test_cost(x['run_id']) for x in test_ledger()):.2f}"


# --------------------------------------------------------------------------------------------
# Programs
# --------------------------------------------------------------------------------------------
def readout_params(p: dict[str, Any]) -> int:
    n = 0
    for node in p["nodes"].values():
        ro = node.get("readout")
        if not ro:
            continue
        w = ro.get("W") or []
        n += sum(len(r) if isinstance(r, list) else 1 for r in w) + len(ro.get("b") or [])
        n += len(ro.get("cutoffs") or [])
    return n


def fitted_numbers(p: dict[str, Any]) -> int:
    n = 1 if p.get("temperature") not in (None, 1.0) else 0
    for node in p["nodes"].values():
        n += len(node.get("biases") or {}) + len((node.get("aggregation") or {}).get("thresholds") or {})
        if node.get("cutoffs") is not None and [round(c, 2) for c in node["cutoffs"]] != default_cutoffs(len(node["cutoffs"])):
            n += len(node["cutoffs"])
    return n


def main_nodes(p: dict[str, Any]) -> dict[str, Any]:
    views = {v for n in p["nodes"].values() for v in ((n.get("readout") or {}).get("views") or [])[1:]}
    return {k: n for k, n in p["nodes"].items() if k not in views}


@register("opt_prog_summary")
def opt_prog_summary() -> str:
    body = []
    for s in tasks():
        r = test_json()["tasks"][s]["rows"]["**v3 chosen (headline)**"]
        rel = f"{STUDY}/{r['run']}/{r['rep']}.yaml"
        p = program_of(r["run"], r["rep"])
        seed = f"{STUDY}/{r['run']}/seed.yaml"
        views = len(p["nodes"]) - len(main_nodes(p))
        demos = "yes" if p.get("demos") else "no"
        body.append(" & ".join([esc(SHORT[s]), tt(r["rep"]), kinds_of({"nodes": main_nodes(p)}), str(views), demos,
                                f"{r['calls']:.2f}", f"{r['tokens']:,.0f}", f"{readout_params(p):,}", f"{fitted_numbers(p)}",
                                f"{body_chars(seed):,}", f"{body_chars(rel):,}"]) + " \\\\")
    head = ("Task & Representative & Main nodes & Feature & Demos & Calls/ & Tok/ & LR & Other & \\multicolumn{2}{c}{Chars} \\\\\n"
            "\\cmidrule(lr){10-11}\n & & & views & & ex. & ex. & params & fitted & seed & tested \\\\")
    return float_table("llllrrrrrrr", head, body,
                       "The tested programs of question optimization. \\emph{Main nodes}: IR kinds of the decision tree. \\emph{Feature views}: "
                       "extra questions asked only as features of the logistic regression. \\emph{LR params}: entries of all "
                       "weight matrices, intercepts and ridge cutoffs. \\emph{Other fitted}: temperature, class biases, "
                       "non-default cutoffs and thresholds. \\emph{Chars}: YAML size of the paper's-question seed and of the tested "
                       "program (LR numbers included).", "tab:opt-prog-summary", pre="\\setlength{\\tabcolsep}{3pt}")


def abridge(p: Any) -> Any:
    """Deterministic abridging for print: readout W / b replaced by their shape."""
    if isinstance(p, dict):
        out = {}
        for k, v in p.items():
            if k == "W" and isinstance(v, list):
                out[k] = f"<{len(v)} x {len(v[0]) if v and isinstance(v[0], list) else 1} weights>"
            elif k == "b" and isinstance(v, list):
                out[k] = f"<{len(v)} intercepts>"
            else:
                out["answer_classifier" if k == "readout" else k] = abridge(v)
        return out
    if isinstance(p, list):
        return [abridge(x) for x in p]
    return p


YAML_WIDTH = 48  # PyYAML folds after this column, so lines stay within LINE_MAX
LINE_MAX = 60  # characters of scriptsize inconsolata that fit one ACL column


def verbatim_yaml(run_id: str, rep: str) -> str:
    p = abridge(program_of(run_id, rep))
    text = ascii_only(yaml.safe_dump(p, sort_keys=False, width=YAML_WIDTH, allow_unicode=False))
    long = [ln for ln in text.splitlines() if len(ln) > LINE_MAX]
    if long:
        raise ValueError(f"{run_id}/{rep}: YAML lines exceed {LINE_MAX} chars: {long[:2]}")
    return "\\par{\\scriptsize\n\\begin{verbatim}\n" + text.rstrip("\n") + "\n\\end{verbatim}\n}\n"


@register("opt_yaml_ez")
def opt_yaml_ez() -> str:
    r = test_json()["tasks"]["ezstance-subtaskA"]["rows"]["**v3 chosen (headline)**"]
    return verbatim_yaml(r["run"], r["rep"])


@register("opt_yaml_smarter")
def opt_yaml_smarter() -> str:
    r = test_json()["tasks"]["smarter-implicit-hate"]["rows"]["**v3 chosen (headline)**"]
    return verbatim_yaml(r["run"], r["rep"])


@inline("ezRep")
def _ezr() -> str:
    return rep_txt(test_json()["tasks"]["ezstance-subtaskA"]["rows"]["**v3 chosen (headline)**"])


@inline("smarterRep")
def _smr() -> str:
    return rep_txt(test_json()["tasks"]["smarter-implicit-hate"]["rows"]["**v3 chosen (headline)**"])


@register("opt_ensemble")
def opt_ensemble() -> str:
    """Prose description of the v2 INLI ensemble-3 (tested v2 row), generated from its YAML and pareto.json."""
    r = test_json()["tasks"]["inli"]["rows"]["v2 chosen"]
    p = program_of(r["run"], r["rep"])
    if p.get("kind") != "ensemble":
        raise ValueError("v2 INLI chosen row is no longer an ensemble")
    info = rjson(r["run"], "pareto.json")["representatives"][r["rep"]]
    members = info["members"]
    mk = [kinds_of(m) + (" + kNN demos" if m.get("demos") else "") for m in p["members"]]
    chars = [len(yaml.safe_dump(m, sort_keys=False)) for m in p["members"]]
    bias = ", ".join(f"{esc(k)} {v:+g}" for k, v in (p.get("label_bias") or {}).items())
    return (f"the tested INLI program of GEPA with demonstrations and calibration, {tt(r['rep'])}, averages the label distributions of the "
            f"{len(members)} candidates with the highest validation accuracy ({', '.join(tt(m) for m in members)}, each "
            f"{', '.join(sorted(set(mk)))}, with {min(chars):,}--{max(chars):,} YAML characters), then applies label biases fit "
            f"on the {info['fit']['n']:,}-example fit slice ({bias or 'none'}) and a temperature of {p['temperature']:g}. "
            f"It asks {r['calls']:.0f} Jev questions per test example ({r['tokens']:,.0f} input tokens)")


# --------------------------------------------------------------------------------------------
# Reproduction: pre-registration timeline, versions, models
# --------------------------------------------------------------------------------------------
@register("opt_prereg")
def opt_prereg() -> str:
    def ledger_first(rnd: str) -> datetime:
        return min(parse_time(x["consumed_at"]) for x in test_ledger() if round_of(x["run_id"]) == rnd)

    def run_first(rnd: str) -> datetime:
        return min(parse_time(rjson(r, "run.json")["started_at"]) for r in all_run_names() if round_of(r) == rnd)

    rows = []
    p1, t1 = commit("first protocol")
    rows.append(("v1", "first", p1, t1, p1, t1, run_first("v1"), ledger_first("v1")))
    p2, t2 = commit("second protocol")
    f2, ft2 = commit("second protocol, choice fixed")
    rows.append(("v2", "second", p2, t2, f2, ft2, run_first("v2"), ledger_first("v2")))
    p3, t3 = commit("third protocol")
    f3, ft3 = commit("third protocol, choice fixed")
    rows.append(("v3", "third", p3, t3, f3, ft3, run_first("v3"), ledger_first("v3")))
    ps, ts_ = commit("seed protocol")
    fs, fts = commit("seed protocol, choice fixed")  # the commit that added results/seeds.json
    fixed_sel = history()["seed_selection"]  # its representatives
    for s in tasks():
        for seed, e in seed_entries(s):
            if fixed_sel[s][str(seed)] != e["representative"]:
                raise ValueError(f"{s}/{seed}: representative changed after the selection was recorded")
    rows.append(("v3 P seeds", "seed study", ps, ts_, fs, fts, run_first("v3 P seeds"), ledger_first("v3 P seeds")))
    pt, tt_ = commit("tailored protocol")
    rows.append(("tailored", "validation-only", pt, tt_, pt, tt_, run_first("tailored"), ledger_first("tailored")))
    pc, pc_t = commit("reranking protocol")
    fixed, fixed_t = commit("reranking protocol, choice fixed")  # its last commit before the first test
    _, tc_t = commit("reranking first test")  # the commit that added the first test run
    rows.append(("lever C", "evaluated-task addendum", pc, pc_t, fixed, fixed_t, None, tc_t))
    ph, th = commit("heads protocol")
    fh, fth = commit("heads protocol, choice fixed")  # the commit that added data/frozen/heads/*.json
    for rel in (f"data/frozen/heads/{k}.json" for k in heads()):
        if history()["heads_frozen_sha256"][rel] != sha256_file(rel):
            raise ValueError(f"{rel} changed after it was frozen")
    _, dev_t = commit("heads first dev run")
    _, ht = commit("heads first test")
    if not th < ht:
        raise ValueError("heads protocol recorded after the first +head test")
    rows.append(("heads", "heads addendum", ph, th, fh, fth, dev_t, ht))
    body = []
    for rnd, f, _, t, _, ft, rstart, tfirst in rows:
        if not ft < tfirst:
            raise ValueError(f"{rnd}: selection fixed after the first test")
        if rnd in ("v1", "v3", "v3 P seeds", "tailored") and not t < rstart:
            raise ValueError(f"{rnd}: protocol recorded after the first run started")
        body.append(" & ".join([ROUND_TEX[rnd], f, utc(t), utc(ft),
                                "--" if rstart is None else utc(rstart) + ("$^\\dagger$" if rnd == "heads" else ""),
                                utc(tfirst) + ("$^\\dagger$" if rnd in ("lever C", "heads") else "")]) + " \\\\")
    head = "Configuration & Protocol & First recorded (UTC) & Choice fixed (UTC) & First run & First test \\\\"
    return float_table("p{2.2cm}p{1.6cm}llll", head, body,
                       "Pre-registration timeline from the study's git history (\\texttt{scripts/\\allowbreak appendix/\\allowbreak "
                       "opt\\_history.json}) and the run/test ledgers. \\emph{Choice fixed}: when the "
                       "chosen-configuration table was filled in (second and third pre-registration), when the "
                       "validation-only selection of the seed reruns was recorded, the last protocol change before the first "
                       "test (reranking and thresholds) or when the classifiers on zero-shot answers were frozen. The first pre-registration and "
                       "Jev-adapted GEPA fix their tested representative in the protocol's first version. $^\\dagger$Time recorded in the history (these "
                       "runs record no start time): the first test of reranking and thresholds and of the classifiers on zero-shot answers is "
                       "when its first test run was recorded, and the first run of these classifiers is when "
                       "its first dev run was recorded, which precedes the registration by design (the rule reads the dev runs).",
                       "tab:opt-prereg", pre="\\setlength{\\tabcolsep}{2.5pt}")


def env_values(key: str, runs: list[str] | None = None) -> list[str]:
    rs = runs or all_run_names()
    return sorted({json.dumps(rjson(r, "environment.json").get(key)) for r in rs})


@inline("versions")
def _versions() -> str:
    pk = env_values("packages")
    if len(pk) != 1:
        raise ValueError(f"package versions differ across runs: {pk}")
    pk = json.loads(pk[0])
    py = [json.loads(v) for v in env_values("python")]
    jo = [json.loads(v) for v in env_values("jev_opt")]
    omp = [json.loads(v) for v in env_values("omp")]
    pkgs = ", ".join(f"{esc(k)} {esc(v)}" for k, v in pk.items())
    omp_txt = "not recorded (\\texttt{omp: null})" if omp == [None] else ", ".join(esc(str(o)) for o in omp)
    return (f"Python {', '.join(esc(p) for p in py)}; \\jevopt{{}} {', '.join(esc(j) for j in jo)}; {pkgs}; OMP version "
            f"{omp_txt}")


@inline("nEnvRuns")
def _ner() -> str:
    return str(len(all_run_names()))


@inline("jevModel")
def _jm() -> str:
    models = [json.loads(v) for v in env_values("jev_model")]
    snaps = sorted({s for r in all_run_names() for s in rjson(r, "environment.json")["jev_snapshots"]})
    lc_snaps = sorted({s for t in test_json()["lever_c"] for s in load_json(f"runs/{t}/manifest.json")["model_snapshots"]})
    return (f"requested model {', '.join(tt(m) for m in models)}; served snapshot(s) recorded by the runs: "
            f"{', '.join(tt(s) for s in snaps)}; test manifests of reranking and thresholds: {', '.join(tt(s) for s in lc_snaps)}")


@register("opt_models")
def opt_models() -> str:
    """Optimizer model per round from usage.jsonl (environment.json only records the spec `current`)."""
    rounds: dict[str, Counter] = {}
    runs_by: dict[str, set[str]] = {}
    for r in all_run_names():
        rnd = round_of(r)
        for u in omp_rows(r):
            m = f"{u['provider']}/{u['model']}"
            rounds.setdefault(rnd, Counter())[m] += 1
            runs_by.setdefault(m, set()).add(r)
    seeds = Counter()
    for d in seed_dirs().values():
        for u in load_jsonl(f"{d}/usage.jsonl"):
            if u["source"] == "omp":
                seeds[f"{u['provider']}/{u['model']}"] += 1
    rounds["v3 seeds"] = seeds
    main = Counter()
    for c in rounds.values():
        main.update(c)
    major = main.most_common(1)[0][0]
    parts = []
    for rnd in ("v1", "v2", "v3", "v3 seeds", "v3 P seeds", "tailored", "LifeTox pilot"):
        c = rounds.get(rnd, Counter())
        parts.append(f"{ROUND_TEX[rnd]}: " + ", ".join(f"{tt(m)} {n:,}" for m, n in c.most_common()))
    others = sorted(m for m in main if m != major)
    flag = ""
    for m in others:
        rs = sorted(runs_by.get(m, set()))
        whole = all({model_of(u) for u in omp_rows(x)} == {m} for x in rs)
        flag += (f" All {main[m]:,} calls of {tt(m)} come from {run_desc(rs)}"
                 f"{' (every optimizer call of that run)' if whole else ''}: the OMP session model "
                 f"(\\texttt{{current}}) was a different model when that run was launched.")
    return "; ".join(parts) + "." + flag


def run_desc(runs: list[str]) -> str:
    """Runs described by configuration, task and condition (run IDs appear only in tab:opt-repro)."""
    out = []
    for r in runs:
        rnd = round_of(r)
        rest = r.split("-", 1)[1] if rnd != "LifeTox pilot" else ""
        slug = next((s for s in sorted(tasks(), key=len, reverse=True) if rest.startswith(s)), None)
        cond = rest[len(slug) + 1:] if slug else ""
        m = re.fullmatch(r"(text|program)-s(\d+)", cond)
        cond = f"{m.group(1)} search rerun with optimizer seed {m.group(2)}" if m else cond
        m = re.fullmatch(r"P-s(\d+)", cond)
        cond = f"optimizer seed {m.group(1)}" if m else cond
        out.append(f"the {esc(SHORT[slug]) if slug else ''} run of {ROUND_TEX[rnd]}" + (f" ({esc(cond)})" if cond else ""))
    return ", ".join(out)


def model_of(u: dict[str, Any]) -> str:
    return f"{u['provider']}/{u['model']}"


@inline("v2Tie")
def _v2tie() -> str:
    return f"{v2_tie():.1f}"


@inline("ensembleK")
def _ek() -> str:
    ks = round_same("v2-", "ensemble_k", [f"v2-{s}-CD" for s in tasks()])
    return " or ".join(f"top-{k}" for k in ks)


@inline("jevLimits")
def _jl() -> str:
    choice, lo, hi = code_fact(round_commit("v3-"), ARCH, "Jev limits")
    return f"Choice $\\leq$ {choice} options, Score {lo}--{hi} levels"


@inline("cutoffStep")
def _cs() -> str:
    return code_fact(round_commit("v3-"), FIT, "Score cutoff step")


@inline("featFloor")
def _ff() -> str:
    a = code_fact(round_commit("v3-"), "jev-opt/python/src/jev_opt/execute.py", "FEATURE_FLOOR")
    b = code_fact(round_commit("v3-"), "jev-opt/python/src/jev_opt/execute.py", "BIAS_FLOOR")
    if a != b:
        raise ValueError("feature and bias floors differ")
    mant, exp = a.split("e")
    return f"${mant if mant != '1' else ''}10^{{{int(exp)}}}$"


@inline("stackCands")
def _sc() -> str:
    return str(V3_MAX_STACKS)


@inline("madeMinPos")
def _mmp() -> str:
    m = re.search(r">= (\d+) dev positives", load_json("data/frozen/made/v3_params.json")["_fit"])
    return m.group(1)


@inline("wikiBiasMax")
def _wbm() -> str:
    return f"{max(abs(v) for v in load_json('data/frozen/relemb/wikiwd_v3_params.json')['bias'].values()):g}"


@inline("tailoredMargin")
def _tm() -> str:
    return TAILORED_MARGIN


@inline("lifetoxBefore")
def _ltb() -> str:
    start = parse_time(rjson("lifetox-real-1", "run.json")["started_at"])
    _, proto = commit("first protocol")
    if not start < proto:
        raise ValueError("LifeTox pilot did not start before the v1 protocol")
    pilot = max(c["summary"]["metric_calls"] for c in rjson("lifetox-real-1", "run.json")["conditions"].values())
    if pilot >= min(r["summary"]["metric_calls"] for _, c, r in all_runs() if c != "zero"):
        raise ValueError("LifeTox pilot budget not smaller than the study's")
    return "Before the first protocol was recorded"
