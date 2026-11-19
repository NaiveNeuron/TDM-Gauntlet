# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""The paper's answer table: what each way of building a classifier needs, costs and reaches on the 13 RQ3
configurations, in one yardstick -> paper/generated/answer_table.tex (tabular), paper/generated/answer.tex
(\\at* macros), results/ANSWER_TABLE.md.

Gap closed of a configuration = (score - prompt-only) / (paper's method - prompt-only), with prompt-only = Jev's
prompt-only test score of the 46-set evaluation (results/frozen.json {orig,new}[].zero, the same value Figure 1
plots; for DREsS the rubric's own prompt-only prediction, no fitted cutoffs). Rows, all on the same 13
configurations and test items:
  + calibration          results/frozen.json calibrated (per-class biases or a temperature on 400 training labels)
  + optimized questions  results/BUDGET_BASELINE.json sets[s].jev.mean (3-seed mean)
  encoder, same labels   results/BUDGET_BASELINE.json sets[s].best_test_mean (best of 3 baselines on validation)
  LoRA-Qwen, same labels results/H100_LORA.json sets[].lora (seed 0)
Medians over all 13, the 5 rule-selected sets, the 8 configurations and the 10 configurations without DREsS; the
optimized row also gets the bootstrap-over-tasks interval of scripts/rq3_gap_interval.py (variant A, B = 10,000,
default_rng(0)). Reaches the paper's method: score >= paper - 0.005 (the 0.5-point rule).

Cost to build one classifier, median over configurations, same scope on both sides (training and validation data,
search or training grid as run; test-time inference excluded):
  calibration            Jev API $ of the paper's question on training and validation items (runs/jev-opt/v3-<t>-A,
                         8 configurations with run records)
  optimized questions    one search: Jev + optimizer-LLM $ (results/framing.json, \\frCostFqMedian, 8 configurations)
  encoder                GPU-hours of the fine-tuned-encoder grid with per-epoch validation, all seeds
                         (results/budget_baseline/ft/<set>.json seconds), 13 configurations, RTX 4090
  (GEPA rows of results/gepa_v3_test.json on the 8 configurations get the same yardstick: \\atGepa*Gap)
  LoRA-Qwen              GPU-hours of training with per-epoch validation scoring, both learning rates, seed 0
                         (runs/jev-opt/lora-<set>/lr*-s0/train_log.json), 13 configurations, H100 80 GB

    uv run scripts/paper_answer_table.py [--out DIR]   # DIR replaces paper/generated
"""

from __future__ import annotations

import gzip
import json
import statistics
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_gap_interval import DRESS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else ROOT / "paper" / "generated"
TOL = 0.005
STUDY = ["contempo-tbdense", "ezstance-subtaskA", "smarter-implicit-hate", "edit-intent", "inli",
         "dress-new-content", "dress-new-organization", "dress-new-language"]
ROWS = [("cal", "Cal"), ("opt", "Opt"), ("enc", "Enc"), ("lora", "Lora")]


def jev_cost(run_dir: Path) -> float:
    total = 0.0
    for name in json.loads((run_dir / "run.json").read_text()).get("jev_responses", {}).get("shards", []):
        with gzip.open(run_dir / name, "rt", encoding="utf-8") as fh:
            for line in fh:
                total += float((json.loads(line)["response"].get("usage") or {}).get("cost") or 0.0)
    return total


def lora_train_hours(stem: str) -> float:
    s = 0.0
    for log in (ROOT / "runs" / "jev-opt" / f"lora-{stem}").glob("lr*-s0/train_log.json"):
        lg = json.loads(log.read_text())
        s += sum(e["train_seconds"] + e["val_scoring"]["wall_seconds"] for e in lg["epoch_summaries"])
    assert s > 0, stem
    return s / 3600


def main() -> None:
    rd = lambda p: json.loads((ROOT / p).read_text())  # noqa: E731
    fz = rd("results/frozen.json")
    bb = rd("results/BUDGET_BASELINE.json")["sets"]
    lo = {r["set"]: r for r in rd("results/H100_LORA.json")["sets"]}
    fr = rd("results/framing.json")["macros"]
    rows = []
    for r in fz["orig"] + fz["new"]:
        s = r["stem"]
        b = bb[s]
        assert abs(b["paper"] - r["paper"]) < 1e-9 and abs(lo[s]["paper"] - r["paper"]) < 1e-9, s
        assert len(b["jev"]["test"]) == 3, s
        rows.append({"set": s, "group": "frozen" if r["group"] == "new" else "opt", "paper": r["paper"], "zero": r["zero"],
                     "cal": r["calibrated"], "opt": b["jev"]["mean"], "enc": b["best_test_mean"], "lora": lo[s]["lora"]})
    assert len(rows) == 13 and sorted(x["set"] for x in rows if x["group"] == "opt") == sorted(STUDY)
    gap = lambda x, k: (x[k] - x["zero"]) / (x["paper"] - x["zero"])  # noqa: E731
    pc = lambda v: f"{100 * v:.0f}"  # noqa: E731

    m: dict[str, str] = {"atN": "13", "atNFrozen": "5", "atNCfg": "8"}
    res: dict[str, dict] = {}
    for k, tag in ROWS:
        g = {x["set"]: gap(x, k) for x in rows}
        res[k] = {"all": statistics.median(g.values()),
                  "frozen": statistics.median(v for s, v in g.items() if s not in STUDY),
                  "cfg": statistics.median(v for s, v in g.items() if s in STUDY),
                  "nodress": statistics.median(v for s, v in g.items() if s not in DRESS),
                  "reach": sum(x[k] >= x["paper"] - TOL for x in rows), "per_set": g}
        m.update({f"at{tag}Gap": pc(res[k]["all"]), f"at{tag}GapFrozen": pc(res[k]["frozen"]),
                  f"at{tag}GapCfg": pc(res[k]["cfg"]), f"at{tag}GapNoDress": pc(res[k]["nodress"]),
                  f"at{tag}Reach": str(res[k]["reach"])})
        gf = {s: v for s, v in g.items() if s not in STUDY}
        m.update({f"at{tag}GapFrozenMin": pc(min(gf.values())), f"at{tag}GapFrozenMax": pc(max(gf.values()))})
    # intervals: one source of truth, results/RQ3_GAP_INTERVAL.json (scripts/rq3_gap_interval.py, variant A)
    gi = rd("results/RQ3_GAP_INTERVAL.json")
    for s, v in res["opt"]["per_set"].items():
        assert abs(gi["gaps"][s] - v) < 1e-12, s
    for grp, key in (("All", "all"), ("Cfg", "opt")):
        x = gi["A"][key]
        assert pc(x["median"]) == m[f"atOptGap{'' if grp == 'All' else 'Cfg'}"]
        m.update({f"atOptGap{grp}Lo": pc(x["lo"]), f"atOptGap{grp}Hi": pc(x["hi"])})

    # the other models' optimized step (Jev's seed-0 questions) on the same yardstick as Table 4: gap from
    # prompt-only Jev, not from each model's own prompt-only score (Table 6's other column)
    xfer = {"Jev": {r["set"]: r["jev"]["optimized"] for r in rd("results/CLEF_RQ3.json")["clef"]["sets"]},
            "Clef": {r["set"]: r["optimized"] for r in rd("results/CLEF_RQ3.json")["clef"]["sets"]},
            "Flash": {r["set"]: r["optimized"] for r in rd("results/CLEF_RQ3.json")["clef-flash"]["sets"]},
            "Luna": {r["set"]: r["optimized"] for r in rd("results/LUNA_RQ3.json")["sets"]},
            "Msd": {r["set"]: r["optimized"] for r in rd("results/MSD1_RQ3.json")["sets"]},
            "Qwen": {r["set"]: r["optimized"] for r in rd("results/H100_ROUND2.json")["base"]["sets"]}}
    for tag, sc in xfer.items():
        assert sorted(sc) == sorted(ref_all := [x["set"] for x in rows]), tag
        m[f"atXfer{tag}Gap"] = pc(statistics.median((sc[x["set"]] - x["zero"]) / (x["paper"] - x["zero"]) for x in rows))

    # off-the-shelf GEPA (text, program) and GEPA with demonstrations + calibration on the 8 configurations,
    # same yardstick (results/gepa_v3_test.json test scores; the prompt-only reference as above)
    gt = rd("results/gepa_v3_test.json")["tasks"]
    ref = {x["set"]: x for x in rows}
    for key, tag in (("v1 text-opt", "Text"), ("v1 program-opt", "Prog"), ("v2 chosen", "Demo")):
        g = [(gt[s]["rows"][key]["test"] - ref[s]["zero"]) / (ref[s]["paper"] - ref[s]["zero"]) for s in STUDY]
        m[f"atGepa{tag}Gap"] = pc(statistics.median(g))

    cal_cost = statistics.median(jev_cost(ROOT / "runs" / "jev-opt" / f"v3-{t}-A") for t in STUDY)
    search = float(fr["frCostFqMedian"])
    enc_h = statistics.median(rd(f"results/budget_baseline/ft/{x['set']}.json")["seconds"] / 3600 for x in rows)
    lora_h = statistics.median(lora_train_hours(x["set"]) for x in rows)
    m.update({"atCostCal": f"{cal_cost:.2f}", "atCostOpt": f"{search:.2f}", "atGpuEnc": f"{enc_h:.2f}",
              "atGpuLora": f"{lora_h:.2f}"})

    T = ["% Generated by scripts/paper_answer_table.py. Do not edit.",
         "\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
         "& & & \\multicolumn{2}{c}{Gap closed (\\%)} & \\\\", "\\cmidrule(lr){4-5}",
         "\\multirow{-2}{*}[0.5ex]{Classifier} & \\multicolumn{1}{c}{\\multirow{-2}{*}[0.5ex]{Labels}} "
         "& \\multicolumn{1}{c}{\\multirow{-2}{*}[0.5ex]{Build cost}} & \\multicolumn{1}{c}{all 13} "
         "& \\multicolumn{1}{c}{5 by rule} & \\multicolumn{1}{c}{\\multirow{-2}{*}[0.5ex]{Reach}} \\\\", "\\midrule",
         "\\jev{}, prompt & none & none & 0 & 0 & 0 \\\\",
         f"\\quad + calibration & 400 & \\${cal_cost:.2f} & {m['atCalGap']} & {m['atCalGapFrozen']} & {m['atCalReach']} \\\\",
         f"\\quad + opt.\\ q. & 500+val. & \\${search:.2f} & {m['atOptGap']} & {m['atOptGapFrozen']} & {m['atOptReach']} \\\\",
         f"Encoder & 500+val. & {enc_h:.2f} GPU-h & {m['atEncGap']} & {m['atEncGapFrozen']} & {m['atEncReach']} \\\\",
         f"LoRA-Qwen & 500+val. & {lora_h:.2f} GPU-h & {m['atLoraGap']} & {m['atLoraGapFrozen']} & {m['atLoraReach']} \\\\",
         "Paper's method & full & -- & 100 & 100 & 13 \\\\", "\\bottomrule", "\\end{tabular}"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "answer_table.tex").write_text("\n".join(T) + "\n")
    (OUT / "answer.tex").write_text("% Generated by scripts/paper_answer_table.py. Do not edit.\n"
                                    + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()))
    L = ["# Answer table (gap closed from prompt-only Jev, 13 RQ3 configurations)", "",
         "Generated by `scripts/paper_answer_table.py`. Gap closed = (score - prompt-only) / "
         "(paper - prompt-only), prompt-only = results/frozen.json zero. Build cost: median per configuration; API $ "
         "for calibration (8 configurations, v3-<task>-A Jev calls) and one search (8 configurations, framing.json); "
         "GPU-hours of training with per-epoch validation over the grid as run (encoder: RTX 4090, all seeds; LoRA: "
         "H100, both learning rates, seed 0).", "",
         "| Row | Median gap 13 | 5 rule-selected | 8 configurations | without DREsS (10) | reaches paper |",
         "|---|---:|---:|---:|---:|---:|"]
    for k, tag in ROWS:
        L.append(f"| {k} | {m[f'at{tag}Gap']}% | {m[f'at{tag}GapFrozen']}% ({m[f'at{tag}GapFrozenMin']}-{m[f'at{tag}GapFrozenMax']}) "
                 f"| {m[f'at{tag}GapCfg']}% | {m[f'at{tag}GapNoDress']}% | {m[f'at{tag}Reach']} |")
    L += ["", f"Optimized questions, 95% bootstrap-over-tasks interval: all 13 [{m['atOptGapAllLo']}%, {m['atOptGapAllHi']}%], "
          f"8 configurations [{m['atOptGapCfgLo']}%, {m['atOptGapCfgHi']}%].",
          f"GEPA on the 8 configurations: text {m['atGepaTextGap']}%, program {m['atGepaProgGap']}%, with demonstrations "
          f"and calibration {m['atGepaDemoGap']}%.",
          "Other models' optimized step (Jev's seed-0 questions), gap from prompt-only Jev: "
          + ", ".join(f"{k} {m[f'atXfer{k}Gap']}%" for k in xfer) + ".",
          f"Build cost: calibration ${cal_cost:.2f}, search ${search:.2f}, encoder {enc_h:.2f} GPU-h, LoRA {lora_h:.2f} GPU-h.", "",
          "Per configuration (gap closed, %):", "", "| Set | " + " | ".join(k for k, _ in ROWS) + " |", "|---|" + "---:|" * len(ROWS)]
    for x in rows:
        L.append(f"| {x['set']} | " + " | ".join(pc(res[k]["per_set"][x["set"]]) for k, _ in ROWS) + " |")
    (ROOT / "results" / "ANSWER_TABLE.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
