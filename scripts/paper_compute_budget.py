"""Total compute budget of the paper -> paper/generated/compute_budget.tex (\\cb* macros).

Sums, without recomputing any score, the spend and GPU time that the paper reports per experiment, and adds the one
figure no other script reports: the GPU time of the Clef and Clef-flash zero-shot runs.

Sources:
  paper/generated/*.tex                 the already generated macros of each experiment:
                                          \\totalCost (Jev zero-shot), \\fzCost* and \\fzsCost* (frozen configuration,
                                          seed 0 and seeds 1-2), \\cvJevCostAll (convention ablation), \\luSpend (GPT-6
                                          Luna, zero-shot and RQ3), \\msSpend (Microsoft-Decision-1, zero-shot and RQ3), \\bbGpuHours (trained baselines, RTX 4090),
                                          \\clClefGpuHours, \\clFlashGpuHours (RQ3 on Clef), \\hrBaseGpuHours (Clef's
                                          base model), \\hrNativeGpuHours (Clef on the rephrased questions), \\lrGpuHours (LoRA)
  runs/jev-opt/<run>/                   optimization study ledgers, read with scripts/paper_appendix_opt.py: optimizer
                                          LLM spend of every run and of seed generation, Jev spend of the distinct
                                          train/validation requests of every run ("Jev used", cached answers counted at
                                          list price, so an upper bound), Jev test spend, and the dev and test spend of
                                          the two evaluated-task addenda
  runs/clef*/**/manifest.json           local_compute of every Clef and Clef-flash zero-shot cohort: inference seconds of
                                          new requests per run, plus per_device cached_response_inference_seconds, the
                                          recorded inference time of responses a run reused from a stopped worker of the
                                          same cohort, tallied by the GPU each manifest names

Development and discarded runs that left no ledger are not counted.

Run from the repo root:  uv run scripts/paper_compute_budget.py [--out DIR ...]   # each DIR replaces paper/generated
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import paper_appendix_opt as P  # noqa: E402

OUTS = [Path(sys.argv[i + 1]) for i, a in enumerate(sys.argv) if a == "--out"] or [ROOT / "paper" / "generated"]
MACRO_DIR = ROOT / "paper" / "generated"
GPU_NAME = {"NVIDIA H100 PCIe": "h100", "NVIDIA GeForce RTX 4090": "rtx4090"}


def macros() -> dict[str, float]:
    out = {}
    for f in sorted(MACRO_DIR.glob("*.tex")):
        if f.name == "compute_budget.tex":
            continue
        for name, val in re.findall(r"\\newcommand\{\\(\w+)\}\{([^{}]*)\}", f.read_text()):
            v = val.replace("\\$", "").replace(",", "").strip()
            if re.fullmatch(r"-?\d+(\.\d+)?", v):
                out[name] = float(v)
    return out


def clef_zero_shot() -> dict[str, Counter]:
    """GPU seconds per model and GPU type over every saved Clef / Clef-flash zero-shot manifest."""
    out = {}
    for model in ("clef", "clef-flash"):
        sec: Counter = Counter()
        for f in sorted((ROOT / "runs" / model).glob("**/manifest.json")):
            d = json.loads(f.read_text())
            bm, lc = d["backend_metadata"], d["local_compute"]
            per = lc.get("per_device")
            if per:
                devs = bm.get("devices")
                gpus = {dev["device"]: dev["gpu"]["name"] for dev in devs} if devs else {k: bm["gpu"]["name"] for k in per}
                for dev, v in per.items():
                    sec[GPU_NAME[gpus[dev]]] += v["inference_seconds"] + v.get("cached_response_inference_seconds", 0.0)
            else:
                sec[GPU_NAME[bm["gpu"]["name"]]] += lc["inference_seconds"]
        out[model] = sec
    return out


def opt_study() -> dict[str, float]:
    runs = P.all_run_names()
    used = P.used_all()
    tests = [P.rjson(r, "test_consumed.json") for r in runs]
    hd = P.heads()
    return {
        "optimizer": sum(P.omp_cost(r) for r in runs) + sum(g["optimizer_cost"] for g in P.seed_generations().values()),
        "jev_used": sum(v for v in used.values() if v is not None),
        "jev_test": sum(float(t["metrics"]["total_cost"]) for t in tests if t is not None),
        "jev_addenda": sum(P.lc(n)["dev_run"]["cost_usd"] for n in ("ttp", "wikiwd", "made"))
        + sum(v["cost"] for v in P.test_json()["lever_c"].values())
        + sum(r["dev_cost_usd"] + r["test_cost_usd_new"] for r in hd.values()),
    }


def main() -> None:
    m = macros()
    o = opt_study()
    cz = clef_zero_shot()
    jev_opt = o["jev_used"] + o["jev_test"] + o["jev_addenda"]
    jev_frozen = m["fzCostJevSearch"] + m["fzCostJevTest"] + m["fzsCostJevSearch"] + m["fzsCostJevTest"]
    jev = m["totalCost"] + jev_opt + jev_frozen + m["cvJevCostAll"]
    optimizer = o["optimizer"] + m["fzCostOmp"] + m["fzsCostOmp"]
    api = jev + optimizer + m["luSpend"] + m["msSpend"]
    clef_h = {k: sum(v.values()) / 3600 for k, v in cz.items()}
    h100 = (sum(v["h100"] for v in cz.values()) / 3600 + m["clClefGpuHours"] + m["clFlashGpuHours"]
            + m["hrBaseGpuHours"] + m["hrNativeGpuHours"] + m["lrGpuHours"])
    rtx = m["bbGpuHours"] + sum(v["rtx4090"] for v in cz.values()) / 3600
    vals = [
        ("cbJevOpt", f"{jev_opt:,.0f}"), ("cbJevFrozen", f"{jev_frozen:.2f}"), ("cbJevTotal", f"{jev:,.0f}"),
        ("cbOptimizerTotal", f"{optimizer:,.0f}"), ("cbApiTotal", f"{api:,.0f}"),
        ("cbClefZeroGpuHours", f"{clef_h['clef']:.1f}"), ("cbFlashZeroGpuHours", f"{clef_h['clef-flash']:.1f}"),
        ("cbHundredGpuHours", f"{h100:.0f}"), ("cbRtxGpuHours", f"{rtx:.1f}"), ("cbGpuHours", f"{h100 + rtx:.0f}"),
    ]
    text = ("% Generated by scripts/paper_compute_budget.py. Do not edit.\n"
            + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in vals))
    for out in OUTS:
        out.mkdir(parents=True, exist_ok=True)
        (out / "compute_budget.tex").write_text(text)
        print(f"wrote {out / 'compute_budget.tex'}")
    print(text, json.dumps({"opt_study": o, "clef_zero_shot_s": {k: dict(v) for k, v in cz.items()}}, indent=1))


if __name__ == "__main__":
    main()
