"""RQ3 optimization step on GPT-6 Luna answers (pre-registered, RQ3 part) -> results/LUNA_RQ3.{md,json},
paper/generated/luna_rq3.tex (macros \\luRq…) and paper/generated/app_luna_rq3.tex (Table tab:luna-rq3); with
`--model msd1` the same for Microsoft-Decision-1 (pre-registered) -> results/MSD1_RQ3.{md,json},
paper/generated/msd1_rq3.tex (\\msRq…) and app_msd1_rq3.tex, with Luna's results/LUNA_RQ3.json as context.

The pre-registered analysis plan of the Clef RQ3 step with Luna in Clef's place, through analyze_clef_rq3's model-parametric
functions (set_row: scores, paired cluster bootstrap with RQ1's units, 2,000 resamples, seed 0, jackknife-t p on
clustered sets with < 30 test clusters, gap closed, O − Jev pipeline; summarize: Holm over the sets per contrast,
pooled Δ, gap medians; jev_reference: Jev's own gap medians; compare_md). Added for Luna (pre-registered): per set
and step the refusal rate (refused or invalid answers, replaced by the uniform answer / answers), the saturation
share (largest probability exactly 1 or `noul` exactly 0 or 1 / answers not refused), rejected requests, requests,
input tokens and API cost; the frozen zero-shot reference counted on the set's test answers of runs/luna/<task>.
Clef / Clef-flash from results/CLEF_RQ3.json are shown alongside for context. Sets without both test evaluations
are reported as pending. No inference.

  uv run --project jev-opt/python python jev-opt/experiments/analyze_luna_rq3.py
  ... analyze_luna_rq3.py --report-only   # rewrite .md/.tex from results/LUNA_RQ3.json with the current spend ledger
  ... analyze_luna_rq3.py --model msd1 [--report-only]
"""

from __future__ import annotations

import gzip
import json
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_clef_rq3 as A  # noqa: E402  (imports clef_rq3, framing_features)
from clef_rq3 import SETS, cal_dir, opt_dir  # noqa: E402

from jev_opt.luna import SPECS, saturated  # noqa: E402

ROOT = HERE.parents[1]
CLEF_JSON = ROOT / "results" / "CLEF_RQ3.json"
# Per model: display name, macro prefix and the API models shown alongside as context.
MODELS = {"luna": {"title": "GPT-6 Luna", "prefix": "luRq", "api_context": ()},
          "msd1": {"title": "Microsoft-Decision-1", "prefix": "msRq", "api_context": ("luna",)}}


def configure(name: str) -> None:
    """The model of this analysis (`--model`, default luna): pin, labels and output paths."""
    global NAME, SPEC, CFG, LABEL, MODEL, SNAPSHOT, OUT_JSON, OUT_MD, OUT_TEX, OUT_APP, SPEND, FROZEN
    NAME, SPEC, CFG = name, SPECS[name], MODELS[name]
    LABEL, MODEL, SNAPSHOT = SPEC.label, SPEC.model, SPEC.snapshot
    OUT_JSON = ROOT / "results" / f"{name.upper()}_RQ3.json"
    OUT_MD = ROOT / "results" / f"{name.upper()}_RQ3.md"
    OUT_TEX = ROOT / "paper" / "generated" / f"{name}_rq3.tex"
    OUT_APP = ROOT / "paper" / "generated" / f"app_{name}_rq3.tex"
    SPEND = ROOT / "runs" / name / "spend.jsonl"
    FROZEN = ROOT / "runs" / name


configure("luna")
COUNT_KEYS = ("calls", "fresh_calls", "input_tokens", "fresh_input_tokens", "fresh_cost", "answers", "refused_answers",
              "saturated_answers", "refused_requests", "rejected_requests")
pct, ci = A.pct, A.ci


def _ratio(a: float, b: float) -> float | None:
    return a / b if b else None


def counts(usages: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum of result.json usage blocks (luna_rq3.counters deltas) with refusal rate and saturation share."""
    out: dict[str, Any] = {k: sum(u.get(k, 0) for u in usages) for k in COUNT_KEYS}
    out["fresh_cost"] = round(out["fresh_cost"], 6)
    out["refusal_rate"] = _ratio(out["refused_answers"], out["answers"])
    out["saturation_share"] = _ratio(out["saturated_answers"], out["answers"] - out["refused_answers"])
    return out


def frozen_counts(stem: str) -> dict[str, Any] | None:
    """Refusals and saturation of the frozen zero-shot reference on the set's answers (DREsS: the rubric's)."""
    _, task, rubric = SETS[stem]
    d = FROZEN / task
    if not (d / "scores.json").exists():
        return None
    n = refused = sat = rejected = refused_req = 0
    for shard in sorted(d.glob("responses*.jsonl.gz")):
        for line in gzip.open(shard, "rt", encoding="utf-8"):
            resp = json.loads(line)["response"]
            ans = resp["answers"]
            for qid in ([rubric] if rubric else list(ans)):
                n += 1
                if qid in resp.get("refused", ()):
                    refused += 1
                elif saturated(ans[qid]):
                    sat += 1
            rejected += "rejected" in resp
            refused_req += "refused_request" in resp
    return {"answers": n, "refused_answers": refused, "saturated_answers": sat, "refused_requests": refused_req,
            "rejected_requests": rejected, "refusal_rate": _ratio(refused, n), "saturation_share": _ratio(sat, n - refused)}


def answer_stats(stem: str) -> dict[str, Any] | None:
    cal, opt = _load(cal_dir(NAME, stem)), _load(opt_dir(NAME, stem))
    zero = frozen_counts(stem)
    if cal is None and opt is None and zero is None:
        return None
    out: dict[str, Any] = {"zero": zero}
    for step, rec in (("cal", cal), ("opt", opt)):
        if rec is not None:
            usage = rec["usage"]
            out[step] = {"total": counts(list(usage.values())), "substeps": {k: counts([v]) for k, v in usage.items()}}
    return out


def _load(d: Path) -> dict[str, Any] | None:
    p = d / "result.json"
    return json.loads(p.read_text()) if p.exists() else None


def spend_totals() -> dict[str, float]:
    tot: dict[str, float] = {}
    if SPEND.exists():
        for line in SPEND.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                tot[row["run"]] = tot.get(row["run"], 0.0) + float(row["cost"])
    return {k: round(v, 6) for k, v in sorted(tot.items())}


def clef_context() -> dict[str, Any]:
    """Clef and Clef-flash per set (zero-shot, calibrated, optimized, gap) and summary, from results/CLEF_RQ3.json."""
    rep = json.loads(CLEF_JSON.read_text())
    out: dict[str, Any] = {}
    for model in ("clef", "clef-flash"):
        s = rep[model]["summary"]
        out[model] = {"status": rep[model]["status"],
                      "sets": {r["set"]: {k: r[k] for k in ("zero", "calibrated", "optimized", "gap_cal", "gap_opt", "chosen")} for r in rep[model]["sets"]},
                      "summary": {"cal-zero": s.get("cal-zero"), "opt-zero": s.get("opt-zero"), "opt-cal": s.get("opt-cal"),
                                  **{k: s.get(k) for k in ("median_gap_cal", "median_gap_opt", "median_gap_opt_cfg8", "median_gap_opt_frozen5",
                                                           "n_reach_method_opt", "n_chosen_as_jev", "median_opt_minus_jev")}}}
    return out


def directions(rows: list[dict[str, Any]], s: dict[str, Any], J: dict[str, Any]) -> dict[str, Any]:
    """The pre-registered expected directions, read on the finished sets."""
    n = len(rows)
    g0 = J["median_gap_opt_seed0"]
    return {"n_sets": n,
            "cal_above_zero": f"{s['cal-zero']['n_positive']} of {n}", "opt_above_zero": f"{s['opt-zero']['n_positive']} of {n}",
            "opt_above_cal": f"{s['opt-cal']['n_positive']} of {n}",
            "median_gap_opt": s["median_gap_opt"], "jev_median_gap_opt_seed0": g0["all"],
            "median_gap_opt_cfg8": s["median_gap_opt_cfg8"], "jev_median_gap_opt_seed0_cfg8": g0["cfg8"],
            "gap_below_jev": None if s["median_gap_opt"] is None else s["median_gap_opt"] < g0["all"]}


# ------------------------------------------------------------------------------------------------ outputs


def _rate(x: float | None, digits: int = 1) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f}%"


def stats_md(rep: dict[str, Any]) -> list[str]:
    L = ["## Answers, tokens and cost per set and step", "",
         "Refusal rate = refused or invalid answers (replaced by the uniform answer) / answers; refused requests = whole "
         f"requests refused by {SPEC.provider} on three attempts (HTTP 502 from OpenRouter; every answer uniform and counted as refused, "
         "the example proceeds); saturation = answers with largest probability exactly 1 (or `noul` exactly 0 or 1) / answers "
         "not refused; rejected requests (HTTP 400/413/422) fail their example and are not answers. Zero-shot: the frozen "
         f"run's answers for the set (no RQ3 request; as in `results/{NAME.upper()}_COMPARISON.md`, a rejected request's answers are "
         "uniform and count as refused). Per step every answer the executor read is counted (cache hits within the run "
         "included, like requests); requests in parentheses were sent (rejected and refused ones included); tokens and cost "
         "are billed ones only.", ""]
    if not rep["answers"]:
        L += ["No frozen zero-shot run and no RQ3 step yet."]
    else:
        L += ["| Set | Step | Requests (fresh) | Answers | Refusal | Refused requests | Saturation | Rejected requests | Fresh input tokens | Cost |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for stem, st in rep["answers"].items():
        title = A.TITLE[stem][0]
        z = st["zero"]
        if z is not None:
            L.append(f"| {title} | zero-shot (frozen) | – | {z['answers']:,} | {_rate(z['refusal_rate'], 2)} | {z['refused_requests']} | "
                     f"{_rate(z['saturation_share'])} | {z['rejected_requests']} | – | – |")
        for step in ("cal", "opt"):
            if step not in st:
                continue
            for name, c in [*st[step]["substeps"].items(), ("total", st[step]["total"])]:
                label = f"{step} {name}" if name != "total" else f"**{step} total**"
                L.append(f"| {title} | {label} | {c['calls']:,} ({c['fresh_calls']:,}) | {c['answers']:,} | {_rate(c['refusal_rate'], 2)} | "
                         f"{c['refused_requests']} | {_rate(c['saturation_share'])} | {c['rejected_requests']} | {c['fresh_input_tokens']:,} | "
                         f"${c['fresh_cost']:.4f} |")
    t = rep["totals"]
    L += ["", f"- All steps: {t['fresh_calls']:,} fresh requests, {t['fresh_input_tokens']:,} fresh input tokens, ${t['fresh_cost']:.4f}; "
          f"refusal rate {_rate(t['refusal_rate'], 2)}, refused requests {t['refused_requests']}, saturation {_rate(t['saturation_share'])}, "
          f"rejected requests {t['rejected_requests']}.",
          f"- Spend ledger `runs/{NAME}/spend.jsonl` by run label: {json.dumps(rep['spend_ledger'])} "
          f"(caps: total ${SPEC.total_cap:g}, rq3 ${SPEC.run_caps['rq3']:g}).", ""]
    return L


def context_md(rep: dict[str, Any], rows: dict[str, dict[str, Any]]) -> list[str]:
    cx = rep["clef_context"]
    api = [(m, SPECS[m].label) for m in CFG["api_context"] if m in rep.get("api_context", {})]
    L = [f"## {LABEL}, {''.join(f'{n}, ' for _, n in api)}Clef and Clef-flash per set (context)", "",
         "Test ×100. Clef / Clef-flash from `results/CLEF_RQ3.json` (same 13 sets, programs, splits and steps)"
         + "".join(f"; {n} from `results/{m.upper()}_RQ3.json`" for m, n in api) + ".", "",
         f"| Set | {LABEL} zero / cal. / opt. | " + "".join(f"{n} zero / cal. / opt. | " for _, n in api)
         + "Clef zero / cal. / opt. | Clef-flash zero / cal. / opt. | Jev pipeline | Paper |",
         "|---|---|" + "---|" * len(api) + "---|---|---:|---:|"]
    cx = {**cx, **{m: rep["api_context"][m] for m, _ in api}}
    jev = A.jev_rows()
    for stem in SETS:
        r = rows.get(stem)
        lu = f"{pct(r['zero'])} / {pct(r['calibrated'])} / {pct(r['optimized'])}" if r else "pending"
        cells = []
        for m in (*(m for m, _ in api), "clef", "clef-flash"):
            c = cx[m]["sets"].get(stem)
            cells.append(f"{pct(c['zero'])} / {pct(c['calibrated'])} / {pct(c['optimized'])}" if c else "–")
        L.append(f"| {A.TITLE[stem][0]} | {lu} | {' | '.join(cells)} | {pct(jev[stem]['optimized'])} | {pct(jev[stem]['paper'])} |")
    L += ["", "| Model | Δ cal. pooled | Δ opt. pooled | Opt. − cal. pooled | Median gap cal. / opt. | Opt. reaches paper |",
          "|---|---|---|---|---|---:|"]
    lines = [(LABEL, rep["summary"], rep["summary"].get("n_sets"))] if rep["sets"] else []
    lines += [(n, cx[m]["summary"], cx[m]["summary"].get("n_sets", 13)) for m, n in api]
    lines += [(n, cx[m]["summary"], 13) for m, n in (("clef", "Clef"), ("clef-flash", "Clef-flash"))]
    for name, s, n in lines:
        def pool(k: str) -> str:
            x = s.get(k)
            return "–" if not x else f"{pct(x['pooled_delta'], True)} [{pct(x['pooled_ci'][0], True)}, {pct(x['pooled_ci'][1], True)}]"
        L.append(f"| {name} | {pool('cal-zero')} | {pool('opt-zero')} | {pool('opt-cal')} | {pct(s['median_gap_cal'])}% / {pct(s['median_gap_opt'])}% | "
                 f"{s['n_reach_method_opt']} of {n} |")
    return L + [""]


def write_md(rep: dict[str, Any]) -> None:
    rows, s = rep["sets"], rep["summary"]
    L = [f"# RQ3 optimization step on {CFG['title']} answers", "",
         f"Pre-registered (RQ3 part of the {LABEL} study): the analysis plan of the Clef RQ3 step, with {LABEL} in Clef's place. "
         "Generated by `jev-opt/experiments/analyze_luna_rq3.py`.",
         f"Model `{MODEL}`, pinned snapshot `{SNAPSHOT}` (provider {SPEC.provider}) through OpenRouter's Decisions endpoint.",
         f"Scores ×100 on each paper's primary metric over the gauntlet test items. Zero-shot = {LABEL}'s frozen zero-shot run",
         f"(`runs/{NAME}/<task>`); seed zero-shot = the paper's question sent as the jev-opt seed program (where its request",
         f"differs from the frozen one). Calibrated = bias + temperature fitted on {LABEL}'s fit-slice answers. Optimized = Jev's",
         f"selected questions answered by {LABEL}, readout refitted on its answers, P vs P+head on its val. Δ vs {LABEL}'s zero-shot:",
         "paired cluster bootstrap (RQ1 units, 2,000 resamples), 95% CI, Holm over the sets (jackknife-t p on TB-Dense and",
         "MATRES). Gap = share of the zero-shot-to-paper gap closed.", "",
         f"Status: {rep['status']}.", ""]
    if not rows:
        L += [f"## {LABEL}", "", "Pending: no set has both test evaluations.", "",
              "| Set | Frozen zero-shot run | Calibration step | Transfer step |", "|---|---|---|---|"]
        for stem, st in rep["state"].items():
            L.append(f"| {A.TITLE[stem][0]} | {st['frozen']} | {st['cal']} | {st['opt']} |")
        L += [""]
    else:
        L += [f"## {LABEL}", "",
              "| Set | Zero-shot | Seed z.-s. | Calibrated | Δ cal. [CI] | Optimized | Δ opt. [CI] | Chosen (val) | Paper | Jev pipeline | Gap cal. / opt. |",
              "|---|---:|---:|---:|---|---:|---|---|---:|---:|---|"]
        for r in rows:
            c = r["contrasts"]
            L.append(f"| {r['title']} | {pct(r['zero'])} | {pct(r['seed_zero'])} | {pct(r['calibrated'])} | {ci(c['cal-zero'])} | "
                     f"{pct(r['optimized'])} | {ci(c['opt-zero'])} | {r['chosen']} (Jev: {r['jev_chosen']}) | {pct(r['paper'])} | "
                     f"{pct(r['jev']['optimized'])} | {pct(r['gap_cal'])}% / {pct(r['gap_opt'])}% |")
        L += [""]
        for key, label in (("cal-zero", "Calibrated − zero-shot (primary)"), ("opt-zero", "Optimized − zero-shot (primary)"),
                           ("opt-cal", "Optimized − calibrated (secondary)"), ("cal-seed", "Calibrated − seed zero-shot (secondary)")):
            if key in s:
                x = s[key]
                L.append(f"- {label}: pooled {pct(x['pooled_delta'], True)} [{pct(x['pooled_ci'][0], True)}, {pct(x['pooled_ci'][1], True)}] "
                         f"over {x['n']} sets; Δ > 0 on {x['n_positive']}; CI above 0 on {x['n_ci_above_zero']}, below 0 on {x['n_ci_below_zero']}; "
                         f"Holm-significant positive {x['n_holm_positive']}, negative {x['n_holm_negative']}.")
        g0 = rep["jev"]["median_gap_opt_seed0"]
        L += [f"- Median gap closed: calibrated {pct(s['median_gap_cal'])}%, optimized {pct(s['median_gap_opt'])}% ({s['n_gap_opt']} sets with a gap); "
              f"optimized on the 8 configurations {pct(s['median_gap_opt_cfg8'])}%, on the 5 frozen sets {pct(s['median_gap_opt_frozen5'])}% "
              f"(Jev seed 0: {pct(g0['all'])}% all, {pct(g0['cfg8'])}% on the 8 configurations).",
              f"- Optimized reaches the paper's method on {s['n_reach_method_opt']} of {s['n_sets']} sets; selection equals Jev's on "
              f"{s['n_chosen_as_jev']}; median optimized − Jev pipeline {pct(s['median_opt_minus_jev'], True)} (descriptive).",
              f"- Expected directions (pre-registered): C − Z > 0 on {rep['directions']['cal_above_zero']}, O − Z > 0 on "
              f"{rep['directions']['opt_above_zero']}, O > C on {rep['directions']['opt_above_cal']}; median gap closed of O "
              f"{pct(s['median_gap_opt'])}% vs Jev's {pct(g0['all'])}%.", "",
              "| Set | Val seed / cal. / P / P+head | Rejected examples |", "|---|---|---:|"]
        for r in rows:
            v = r["val"]
            L.append(f"| {r['title']} | {pct(v['seed'])} / {pct(v['calibrated'])} / {pct(v.get('top-single'))} / {pct(v.get('top-stacked'))} | "
                     f"{r['rejected_examples']} |")
        L += [""]
        L += A.compare_md(LABEL, rows, s)
    L += stats_md(rep)
    L += context_md(rep, {r["set"]: r for r in rows})
    OUT_MD.write_text("\n".join(L) + "\n")


def write_tex(rep: dict[str, Any]) -> None:
    rows, s, p = rep["sets"], rep["summary"], CFG["prefix"]
    m: list[tuple[str, str]] = [(f"{p}Status", "finished" if len(rows) == len(SETS) else ("pending" if not rows else f"{len(rows)} of {len(SETS)} sets")),
                                (f"{p}Nsets", str(len(rows)))]
    t = rep["totals"]
    m += [(f"{p}FreshRequests", f"{t['fresh_calls']:,}"), (f"{p}InputTokensM", f"{t['fresh_input_tokens'] / 1e6:.1f}"),
          (f"{p}CostUsd", f"{t['fresh_cost']:.2f}"), (f"{p}RefusalRate", _rate(t["refusal_rate"], 2).replace("%", "")),
          (f"{p}SaturationShare", _rate(t["saturation_share"]).replace("%", "")), (f"{p}RejectedRequests", str(t["rejected_requests"])),
          (f"{p}RefusedRequests", str(t["refused_requests"]))]
    if rows:
        for r in rows:
            tag = A.TITLE[r["set"]][1]
            m += [(f"{p}Zero{tag}", pct(r["zero"])), (f"{p}Cal{tag}", pct(r["calibrated"])), (f"{p}Opt{tag}", pct(r["optimized"])),
                  (f"{p}Dcal{tag}", pct(r["contrasts"]["cal-zero"]["delta"], True)), (f"{p}Dopt{tag}", pct(r["contrasts"]["opt-zero"]["delta"], True))]
        for name, key in (("cal-zero", "Cal"), ("opt-zero", "Opt"), ("opt-cal", "OptCal"), ("opt-jev", "OptJev")):
            x = s[name]
            m += [(f"{p}Pool{key}", pct(x["pooled_delta"], True)), (f"{p}Pool{key}Lo", pct(x["pooled_ci"][0], True)),
                  (f"{p}Pool{key}Hi", pct(x["pooled_ci"][1], True)), (f"{p}Npos{key}", str(x["n_positive"])),
                  (f"{p}Nholm{key}", str(x["n_holm_positive"])), (f"{p}Nneg{key}", str(x["n_holm_negative"])),
                  (f"{p}Med{key}", pct(x["median_delta"], True))]
        # Whole percents, as the Clef macros (\clClefGapAll …) they are printed next to.
        m += [(f"{p}MedGapCal", A.ipct(s["median_gap_cal"])), (f"{p}MedGapOpt", A.ipct(s["median_gap_opt"])),
              (f"{p}GapCfg", A.ipct(s["median_gap_opt_cfg8"])), (f"{p}GapFrozen", A.ipct(s["median_gap_opt_frozen5"])),
              (f"{p}NreachOpt", str(s["n_reach_method_opt"])), (f"{p}NsameChoice", str(s["n_chosen_as_jev"])),
              (f"{p}MedOptMinusJev", pct(s["median_opt_minus_jev"], True))]
        enc = s["vs_encoder"]
        m += [(f"{p}EncAhead", str(enc["n_model_ahead"])), (f"{p}EncBehind", str(enc["n_baseline_ahead"])),
              (f"{p}EncUnclear", str(enc["n_unclear"])), (f"{p}EncMedDiff", pct(enc["median_diff"], True)),
              (f"{p}EncPoolNoTbd", pct(enc["no_tbdense"]["pooled_diff"], True))]
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("% generated by jev-opt/experiments/analyze_luna_rq3.py; do not edit\n"
                       + "".join(f"\\newcommand{{\\{k}}}{{{v.replace('–', '--')}}}\n" for k, v in m))


def write_app(rep: dict[str, Any]) -> None:
    if rep["sets"]:
        OUT_APP.write_text("\n".join(["% generated by jev-opt/experiments/analyze_luna_rq3.py; do not edit",
                                      *A.app_table(LABEL, f"{NAME}-rq3", rep["sets"], rep["summary"])]) + "\n")


def write_reports(rep: dict[str, Any]) -> None:
    # The spend ledger is shared with the zero-shot runs, so it is read when the reports are written, not when the
    # RQ3 analysis ran (an earlier .md carried the zero-shot spend of that moment).
    rep["spend_ledger"] = spend_totals()
    write_md(rep)
    write_tex(rep)
    write_app(rep)


def api_context() -> dict[str, Any]:
    """Other API models' RQ3 results per set and summary (results/<MODEL>_RQ3.json), as clef_context."""
    out: dict[str, Any] = {}
    for model in CFG["api_context"]:
        path = ROOT / "results" / f"{model.upper()}_RQ3.json"
        if path.exists():
            rep = json.loads(path.read_text())
            s = rep["summary"]
            out[model] = {"status": rep["status"],
                          "sets": {r["set"]: {k: r[k] for k in ("zero", "calibrated", "optimized", "gap_cal", "gap_opt", "chosen")} for r in rep["sets"]},
                          "summary": {"n_sets": s.get("n_sets"), "cal-zero": s.get("cal-zero"), "opt-zero": s.get("opt-zero"),
                                      "opt-cal": s.get("opt-cal"),
                                      **{k: s.get(k) for k in ("median_gap_cal", "median_gap_opt", "median_gap_opt_cfg8", "median_gap_opt_frozen5",
                                                               "n_reach_method_opt", "n_chosen_as_jev", "median_opt_minus_jev")}}}
    return out


def main() -> int:
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv[1:] else "luna"
    configure(model)
    if "--report-only" in sys.argv[1:]:
        rep = json.loads(OUT_JSON.read_text())
        write_reports(rep)
        OUT_JSON.write_text(json.dumps(rep, indent=1) + "\n")
        return 0
    jev = A.jev_rows()
    rows: list[dict[str, Any]] = []
    state: dict[str, dict[str, str]] = {}
    inputs: tuple[Any, Any] | None = None  # Jev's test predictions and the trained baselines, read once a set is finished
    for stem in SETS:
        done = {s: (d(NAME, stem) / "test_consumed.json").exists() for s, d in (("cal", cal_dir), ("opt", opt_dir))}
        frozen = (FROZEN / SETS[stem][1] / "scores.json").exists()
        state[stem] = {"frozen": "finished" if frozen else "pending",
                       **{s: "test consumed" if v else ("started" if (d(NAME, stem) / "result.json").exists() else "pending")
                          for (s, v), d in zip(done.items(), (cal_dir, opt_dir))}}
        if all(done.values()):
            if inputs is None:
                inputs = (json.loads((A.BB_DIR / "jev_test_preds.json").read_text()),
                          json.loads((ROOT / "results" / "BUDGET_BASELINE.json").read_text())["sets"])
            r = A.set_row(NAME, stem, jev[stem], *inputs)
            rows.append(r)
    missing = [s for s in SETS if s not in {r["set"] for r in rows}]
    summary = A.summarize(rows) if rows else {}
    summary.pop("fresh_inference_s", None)  # local GPU time: not applicable to an API model
    for r in rows:
        r.pop("fresh_inference_s", None)
        for c in r["contrasts"].values():
            c.pop("draws", None)
        r["vs_encoder"].pop("draws", None)
    stats = {stem: st for stem in SETS if (st := answer_stats(stem)) is not None}
    steps = [st[k]["total"] for st in stats.values() for k in ("cal", "opt") if k in st]
    totals = counts([{k: c[k] for k in COUNT_KEYS} for c in steps])
    rep: dict[str, Any] = {"model": MODEL, "snapshot": SNAPSHOT, "bootstrap": {"B": A.B, "seed": 0},
                           "status": "finished" if not missing else f"pending: {', '.join(missing)}",
                           "state": state, "sets": rows, "summary": summary, "answers": stats, "totals": totals,
                           "spend_ledger": spend_totals()}
    rep["jev"] = A.jev_reference(jev, rows)
    if rows:
        rep["directions"] = directions(rows, summary, rep["jev"])
    rep["clef_context"] = clef_context()
    if CFG["api_context"]:
        rep["api_context"] = api_context()
    write_reports(rep)
    OUT_JSON.write_text(json.dumps(rep, indent=1) + "\n")
    print(f"{NAME}: {len(rows)} sets finished; {rep['status']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
