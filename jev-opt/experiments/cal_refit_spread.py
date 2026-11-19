"""Fit-induced spread of the RQ3 calibration step (per-class bias + temperature, `fit_label_bias`) on the 13 Clef RQ3
sets, for Jev, Clef and Clef-flash. Post hoc (after the test results; not pre-registered). No inference:
fit-slice answers come from Jev's cache-only client (posting disabled by framing_features) and from the Clef steps'
recorded responses, test distributions are the frozen zero-shot answers (or the recorded seed-program test answers), and the
calibrated test score is first replayed against the recorded one.

The interval of the calibration step in the paper resamples test items with the fit held fixed. Here the fit varies,
the test items do not: (a) R refits on the fit slice resampled with replacement; (b) R refits after adding
N(0, 0.003) noise to every fit-slice answer distribution (renormalised; 0.003 is the scale of the RTX 4090 vs H100
answer differences on Clef-flash TB-Dense). Reported: sd of the calibrated test score, and of calibrated - zero-shot.

  uv run --project jev-opt/python python jev-opt/experiments/cal_refit_spread.py [--report-only] [R]
-> results/clef_rq3/cal_refit_spread.json, paper/generated/cal_refit.tex (macros), paper/generated/app_cal_refit.tex
"""

from __future__ import annotations

import gzip
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clef_rq3 as C  # noqa: E402  (imports framing_features, which disables Jev posting)
from analyze_frozen import gauntlet_zero  # noqa: E402
from confidence_readout import zero_shot as jev_zero_shot  # noqa: E402

from jev_opt.execute import combine  # noqa: E402
from jev_opt.fit import fit_label_bias  # noqa: E402
from jev_opt.ir import load_decider  # noqa: E402
from jev_opt.jev import Decision, make_client  # noqa: E402

ROOT = HERE.parents[1]
OUT_JSON = ROOT / "results" / "clef_rq3" / "cal_refit_spread.json"
CELLS = ROOT / "results" / "clef_rq3" / "cal_refit_cells"  # one JSON per (model, set), reused when R matches
OUT_TEX = ROOT / "paper" / "generated" / "cal_refit.tex"
OUT_APP = ROOT / "paper" / "generated" / "app_cal_refit.tex"
NOISE = 0.003
CFG8 = list(C.SETS)[:8]
MODELS = {"jev": "Jev", "clef": "Clef", "clef-flash": "Clef-flash"}
TITLE = {"contempo-tbdense": "TB-Dense", "ezstance-subtaskA": "EZ-STANCE A", "smarter-implicit-hate": "Implicit Hate",
         "edit-intent": "Edit intent", "inli": "INLI", "dress-new-content": "DREsS content",
         "dress-new-organization": "DREsS organization", "dress-new-language": "DREsS language", "contempo-matres": "MATRES",
         "lifetox": "LifeTox", "sciprompt-s2orc": "SciPrompt S2ORC", "smarter-hatexplain": "HateXplain",
         "smarter-latent-hate": "Latent Hatred"}


def jev_inputs(stem: str, frozen: dict[str, Any]) -> tuple[Any, list, list, list[str], float, float]:
    """Jev: fit-slice answers from Jev's cache, test = the zero-shot distributions the paper calibrates."""
    os.environ.pop("JEV_OPT_BACKEND", None)
    os.environ.pop("JEV_OPT_DEVICE", None)
    src = C.RUNS / C.SETS[stem][0]
    cfg = C.run_config(src)
    client = make_client(C.RUNS / ".jev-cache", cfg.jev.model, cfg.jev.max_rpm, cfg.jev.concurrency, [ROOT])
    try:
        env = C.Env(src, client, test=False)
        fit = env.run(load_decider((src / "seed.yaml").read_text()), env.fit, "train")
    finally:
        client.close()
    labels = env.labels
    cal = json.loads((C.cal_dir("clef", stem) / "result.json").read_text())["test"]
    zs = jev_zero_shot(stem, labels) if stem in CFG8 else gauntlet_zero(C.SETS[stem][1], labels)
    z = [zs[i] for i in cal["ids"]]
    assert [x["gold"] for x in z] == cal["gold"], stem
    test = [dict(zip(labels, x["dist"])) for x in z]
    row = frozen[stem]
    return env, fit, test, cal["gold"], row["calibrated"], row["zero"]


class Lookup:
    """Client stand-in that answers from a step's recorded Clef responses (`clef_responses*.jsonl.gz`); a miss raises."""

    concurrency = 1

    def __init__(self, run_dir: Path):
        self.table: dict[str, Any] = {}
        for shard in sorted(run_dir.glob("clef_responses*.jsonl.gz")):
            for line in gzip.open(shard, "rt", encoding="utf-8"):
                e = json.loads(line)
                self.table[json.dumps(e["request"]["state"], sort_keys=True)] = e["response"]

    def decide(self, state: Any, questions: dict[str, Any], context: dict[str, Any] | None = None) -> Any:
        return Decision(self.table[json.dumps(state, sort_keys=True)]["answers"], "lookup", 0, 0.0, 0.0, True)


def clef_inputs(model: str, stem: str) -> tuple[Any, list, list, list[str], float, float]:
    src = C.RUNS / C.SETS[stem][0]
    R = json.loads((C.cal_dir(model, stem) / "result.json").read_text())
    env = C.Env(src, Lookup(C.cal_dir(model, stem)), test=R["test"]["seed_pred"] is not None)
    seed = load_decider((src / "seed.yaml").read_text())
    fit = env.run(seed, env.fit, "train")
    assert [r.id for r in fit] == R["fit_ids"], stem
    if env.test:
        tres = env.run(seed, env.test, "test")
        assert [r.id for r in tres] == R["test"]["ids"]
        test = [r.dist for r in tres]
        zero = R["test"]["seed_primary"]
    else:
        rows = C.frozen_rows(model, stem)
        test = [C.normalise(rows[i]["probs"], env.labels) for i in R["test"]["ids"]]
        zero = env.primary(R["test"]["gold"], [rows[i]["pred"] for i in R["test"]["ids"]])
    return env, fit, test, R["test"]["gold"], R["test"]["calibrated_primary"], zero


def spread(env: Any, fit: list, test: list, gold: list[str], recorded: float, zero: float, reps: int) -> dict[str, Any]:
    labels, cfg = env.labels, env.config
    base = [C.floored(r.dist, labels) for r in fit]
    fg = [r.gold for r in fit]

    def score(dists: list, g: list[str]) -> float:
        bias, temp, _ = fit_label_bias(labels, [[d] for d in dists], g, cfg.primary_metric, cfg.negative_label)
        return env.primary(gold, [combine(labels, [C.floored(d, labels)], bias, temp)[1] for d in test])

    replayed = score(base, fg)
    assert abs(replayed - recorded) < 1e-9, (env.dir.name, replayed, recorded)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(reps):
        ix = rng.integers(0, len(base), len(base))
        boot.append(score([base[i] for i in ix], [fg[i] for i in ix]))
    noise = []
    for _ in range(reps):
        nd = []
        for d in base:
            v = np.clip(np.array([d[k] for k in labels]) + rng.normal(0, NOISE, len(labels)), 1e-4, None)
            nd.append(dict(zip(labels, (v / v.sum()).tolist())))
        noise.append(score(nd, fg))
    b, n = np.array(boot), np.array(noise)
    return {"zero": zero, "calibrated": recorded, "n_fit": len(base), "reps": reps,
            "boot_sd": float(b.std(ddof=1)), "boot_q": [float(x) for x in np.percentile(b, [2.5, 97.5])],
            "boot_share_above_zero": float(np.mean(b > zero)),
            "noise_sd": float(n.std(ddof=1)), "noise_range": [float(n.min()), float(n.max())],
            "noise_share_above_zero": float(np.mean(n > zero))}


def cell(model: str, stem: str, reps: int) -> dict[str, Any]:
    frozen = json.loads((ROOT / "results" / "frozen.json").read_text())
    frozen = {r["stem"]: r for r in frozen["new"] + frozen["orig"]}
    inp = jev_inputs(stem, frozen) if model == "jev" else clef_inputs(model, stem)
    return spread(*inp, reps=reps)


def compute(reps: int) -> dict[str, Any]:
    """One subprocess per (model, set): the refits are CPU-bound and independent."""
    import subprocess

    CELLS.mkdir(parents=True, exist_ok=True)
    jobs = {}
    for model in MODELS:
        for stem in C.SETS:
            path = CELLS / f"{model}__{stem}.json"
            if not path.exists() or json.loads(path.read_text())["reps"] != reps:
                jobs[path] = subprocess.Popen([sys.executable, __file__, "--cell", model, stem, str(reps)])
    failed = [p.name for p, j in jobs.items() if j.wait() != 0]
    if failed:
        raise SystemExit(f"cells failed: {failed}")
    out: dict[str, Any] = {"reps": reps, "noise_sd": NOISE, "sets": {}}
    for stem in C.SETS:
        out["sets"][stem] = {m: json.loads((CELLS / f"{m}__{stem}.json").read_text()) for m in MODELS}
    for model in MODELS:
        rows = [out["sets"][s][model] for s in C.SETS]
        out[model] = {"median_boot_sd": statistics.median(r["boot_sd"] for r in rows),
                      "max_boot_sd": max(r["boot_sd"] for r in rows),
                      "median_noise_sd": statistics.median(r["noise_sd"] for r in rows),
                      "max_noise_sd": max(r["noise_sd"] for r in rows)}
    return out


def pt(x: float) -> str:
    return f"{100 * x:.1f}"


def write_outputs(rep: dict[str, Any]) -> None:
    S = rep["sets"]
    tb = S["contempo-tbdense"]["clef-flash"]
    m = [("crReps", str(rep["reps"]))]
    for model, tag in (("jev", "Jev"), ("clef", "Clef"), ("clef-flash", "Flash")):
        x = rep[model]
        m += [(f"cr{tag}MedBootSd", pt(x["median_boot_sd"])), (f"cr{tag}MaxBootSd", pt(x["max_boot_sd"])),
              (f"cr{tag}MedNoiseSd", pt(x["median_noise_sd"])), (f"cr{tag}MaxNoiseSd", pt(x["max_noise_sd"])),
              # sets where the resampled refits disagree on the sign of calibrated - zero-shot (2.5-97.5% band)
              (f"cr{tag}NsignUnstable", str(sum(0.025 < S[s][model]["boot_share_above_zero"] < 0.975 for s in S)))]
    m += [("crFlashTbdBootSd", pt(tb["boot_sd"])), ("crFlashTbdNoiseSd", pt(tb["noise_sd"])),
          ("crFlashTbdNoiseLo", pt(tb["noise_range"][0])), ("crFlashTbdNoiseHi", pt(tb["noise_range"][1])),
          ("crFlashTbdBootLo", pt(tb["boot_q"][0])), ("crFlashTbdBootHi", pt(tb["boot_q"][1]))]
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("% generated by jev-opt/experiments/cal_refit_spread.py; do not edit\n"
                       + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m))
    L = ["% generated by jev-opt/experiments/cal_refit_spread.py; do not edit",
         "\\begin{table}[h]\\centering\\small\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular}{l" + "rrr" * 3 + "}\\toprule",
         " & \\multicolumn{3}{c}{\\jev{}} & \\multicolumn{3}{c}{Clef} & \\multicolumn{3}{c}{Clef-flash} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
         "Set" + " & Cal. & sd$_\\mathrm{b}$ & sd$_\\mathrm{n}$" * 3 + " \\\\\\midrule"]
    for stem in C.SETS:
        cells = "".join(f" & {pt(S[stem][mo]['calibrated'])} & {pt(S[stem][mo]['boot_sd'])} & {pt(S[stem][mo]['noise_sd'])}"
                        for mo in MODELS)
        L.append(f"{TITLE[stem]}{cells} \\\\")
    meds = "".join(f" & & {pt(rep[mo]['median_boot_sd'])} & {pt(rep[mo]['median_noise_sd'])}" for mo in MODELS)
    L += ["\\midrule", f"Median{meds} \\\\", "\\bottomrule\\end{tabular}",
          f"\\caption{{Fit-induced spread of the calibration step (post hoc, offline). Cal.: the recorded calibrated test "
          f"score. sd$_\\mathrm{{b}}$: sd of the test score over {rep['reps']} refits on the 400-example fit slice resampled "
          f"with replacement. sd$_\\mathrm{{n}}$: sd over {rep['reps']} refits after adding N(0, {NOISE}) noise to every "
          "fit-slice answer distribution, the size of the RTX~4090 vs H100 differences. Test items are held fixed. "
          "The calibration intervals in the paper hold the fit fixed and resample test items, so they omit this spread.}",
          "\\label{tab:cal-refit}\\end{table}"]
    OUT_APP.write_text("\n".join(L) + "\n")


def main() -> int:
    if sys.argv[1:2] == ["--cell"]:
        model, stem, reps = sys.argv[2], sys.argv[3], int(sys.argv[4])
        r = cell(model, stem, reps)
        (CELLS / f"{model}__{stem}.json").write_text(json.dumps(r, indent=1) + "\n")
        print(model, stem, json.dumps({k: r[k] for k in ("calibrated", "boot_sd", "noise_sd")}), flush=True)
        return 0
    args = [a for a in sys.argv[1:] if a != "--report-only"]
    if "--report-only" in sys.argv:
        rep = json.loads(OUT_JSON.read_text())
    else:
        rep = compute(int(args[0]) if args else 100)
        OUT_JSON.write_text(json.dumps(rep, indent=1) + "\n")
    write_outputs(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
