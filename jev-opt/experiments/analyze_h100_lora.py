"""H100 LoRA (pre-registered) -> results/H100_LORA.{md,json}, results/h100_lora.tex (macros \\lora…).

No inference: reads the LoRA test runs (runs/jev-opt/lora-<set>-s<seed>/result.json, the set's selection.json,
val.json, data.json and training logs), H100 round 2 Experiment 1 (h2-qwen-base-<set>-{cal,opt}), the same-label
trained baselines (results/BUDGET_BASELINE.json, results/budget_baseline/<family>/<set>.json), Jev's seed-0 pipeline
(analyze_clef_rq3.jev_preds) and results/frozen.json. Sets without every tested seed recorded are pending; with no
finished set the outputs say "pending".

  uv run --project jev-opt/python python jev-opt/experiments/analyze_h100_lora.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h100_lora as HL  # noqa: E402
from analyze_clef_rq3 import B, SMALL, TITLE, TOL, boot_p, holm, jev_preds, jev_rows  # noqa: E402
from analyze_h100_round2 import aligned, ipct, medians, pct  # noqa: E402
from clef_rq3 import RUNS, SETS, run_config  # noqa: E402
from h100_round2 import step_dir  # noqa: E402
from resample import cluster_index, jackknife_t, resamples  # noqa: E402

from jev_opt import metrics  # noqa: E402
from jev_opt.hygiene import consumed  # noqa: E402

ROOT = HERE.parents[1]
OUT_JSON = ROOT / "results" / "H100_LORA.json"
OUT_MD = ROOT / "results" / "H100_LORA.md"
OUT_TEX = ROOT / "results" / "h100_lora.tex"
BB_DIR = ROOT / "results" / "budget_baseline"
FAMILIES = ("emb", "setfit", "ft")
FAM_NAME = {"emb": "embeddings + LR", "setfit": "SetFit", "ft": "fine-tuned encoder"}
SEED_RNG = {"lora": 1, "baseline": 2}  # seed resampling generators (pre-registered)
# (name, system, reference, TeX key): Δ = system − reference. The first is primary.
CONTRASTS = [("lora-opt", "lora", "opt", "Opt"), ("lora-enc", "lora", "enc", "Enc"), ("lora-jev", "lora", "jev", "Jev"),
             ("lora-zero", "lora", "zero", "Zero"), ("lora-cal", "lora", "cal", "Cal"), ("lora-emb", "lora", "emb", "Emb"),
             ("lora-setfit", "lora", "setfit", "Setfit"), ("lora-ft", "lora", "ft", "Ft")]
LABEL = {"lora-opt": "LoRA − Qwen optimized questions + LR (O_B)", "lora-enc": "LoRA − best same-label baseline",
         "lora-jev": "LoRA − Jev pipeline (seed 0)", "lora-zero": "LoRA − Qwen zero-shot (Z_B)", "lora-cal": "LoRA − Qwen calibrated (C_B)",
         "lora-emb": "LoRA − embeddings + LR", "lora-setfit": "LoRA − SetFit", "lora-ft": "LoRA − fine-tuned encoder"}


def finished(stem: str) -> dict[str, Any] | None:
    """The set's selection if every tested seed has its test evaluation recorded."""
    sel = HL.load_json(HL.set_dir(stem) / "selection.json")
    if sel is None or not all(consumed(HL.test_dir(stem, t["seed"])) for t in sel["tested"]):
        return None
    return sel


def result(path: Path) -> dict[str, Any]:
    return json.loads((path / "result.json").read_text())


def baselines(stem: str, ids: list[str], jtp: dict[str, Any]) -> dict[str, list[list[str]]]:
    """Each trained-baseline family's per-seed test predictions in `ids` order."""
    J = jtp[stem]
    out = {}
    for fam in FAMILIES:
        rec = json.loads((BB_DIR / fam / f"{stem}.json").read_text())
        out[fam] = [aligned(ids, J["ids"], p, f"{stem}/{fam}") for p in rec["test_pred"]]
    return out


def paired(stem: str, ids: list[str], gold: list[str], score: Any, systems: dict[str, list[list[str]]],
           rngs: dict[str, int]) -> tuple[dict[str, float], dict[str, Any]]:
    """Seed-mean point scores and paired contrasts: the cluster draws of H100 round 2 (resamples(inv, B, 0), RQ1 units),
    seeds of multi-seed systems resampled per replicate (default_rng(rngs[system])); jackknife-t p on clustered sets
    with fewer than SMALL clusters (observed seed means)."""
    G = np.array(gold, dtype=object)
    P = {k: [np.array(p, dtype=object) for p in v] for k, v in systems.items()}
    point = {k: statistics.mean(score(gold, p) for p in v) for k, v in systems.items()}
    inv, unit = cluster_index(stem, ids)
    m = int(inv.max()) + 1
    per_seed = {k: np.empty((len(v), B)) for k, v in systems.items()}
    for b, ix in enumerate(resamples(inv, B, 0)):
        g = list(G[ix])
        for k, v in P.items():
            for s, p in enumerate(v):
                per_seed[k][s, b] = score(g, list(p[ix]))
    draws = {}
    for k, S in per_seed.items():
        n = S.shape[0]
        if n == 1:
            draws[k] = S[0]
        else:
            pick = np.random.default_rng(rngs[k]).integers(0, n, size=(B, n))
            draws[k] = S[pick.T, np.arange(B)].mean(axis=0)
    out = {}
    for name, a, z, _ in CONTRASTS:
        d = draws[a] - draws[z]
        c: dict[str, Any] = {"delta": point[a] - point[z], "ci": [float(x) for x in np.percentile(d, [2.5, 97.5])], "p_boot": boot_p(d)}
        if unit != "item" and m < SMALL:
            def stat(ix, a=a, z=z):
                g = [gold[i] for i in ix]
                return (statistics.mean(score(g, [p[i] for i in ix]) for p in systems[a])
                        - statistics.mean(score(g, [p[i] for i in ix]) for p in systems[z]))
            lo, hi = jackknife_t(stat, inv)
            se = (hi - lo) / (2 * student_t.ppf(0.975, m - 1))
            c["jackknife_ci"] = [lo, hi]
            c["p_jackknife"] = float(2 * student_t.sf(abs(c["delta"]) / se, m - 1)) if se > 0 else 1.0
        c["p"] = c.get("p_jackknife", c["p_boot"])
        c["draws"] = d
        out[name] = c
    return point, {"contrasts": out, "cluster_unit": unit, "n_clusters": m}


def set_row(stem: str, sel: dict[str, Any], jev: dict[str, Any], jtp: dict[str, Any], bb: dict[str, Any]) -> dict[str, Any]:
    cfg = run_config(RUNS / SETS[stem][0])
    labels = list(cfg.labels)
    score = lambda g, p: metrics.primary(cfg.primary_metric, g, p, labels, cfg.negative_label)  # noqa: E731
    lr_res = [result(HL.test_dir(stem, t["seed"])) for t in sel["tested"]]
    ids, gold = lr_res[0]["test"]["ids"], lr_res[0]["test"]["gold"]
    assert all(r["test"]["ids"] == ids and r["test"]["gold"] == gold for r in lr_res), stem
    cal, opt = result(step_dir("base", stem, "cal")), result(step_dir("base", stem, "opt"))
    fams = baselines(stem, ids, jtp)
    best = bb["best"]
    systems = {"lora": [r["test"]["pred"] for r in lr_res],
               "zero": [aligned(ids, cal["test"]["ids"], cal["test"]["zero_pred"], stem)],
               "cal": [aligned(ids, cal["test"]["ids"], cal["test"]["calibrated_pred"], stem)],
               "opt": [aligned(ids, opt["test"]["ids"], opt["test"]["pred"], stem)],
               "jev": [jev_preds(stem, ids, gold, labels, score, jev, jtp)["jev"]],
               **fams, "enc": fams[best]}
    rngs = {"lora": SEED_RNG["lora"], **{k: SEED_RNG["baseline"] for k in (*FAMILIES, "enc")}}
    point, rs = paired(stem, ids, gold, score, systems, rngs)
    zero, paper = point["zero"], jev["paper"]
    gap = (lambda x: None if paper - zero <= TOL else (x - zero) / (paper - zero))
    seeds = [r["test"]["primary"] for r in lr_res]
    data = json.loads((HL.set_dir(stem) / "data.json").read_text())
    logs = [json.loads((HL.set_dir(stem) / HL.run_name(lr, s) / "train_log.json").read_text()) for lr in HL.GRID["lrs"] for s in sel["seeds"]]
    frozen = [r["test"]["frozen_rule_primary"] for r in lr_res]
    return {"set": stem, "title": TITLE[stem][0], "metric": cfg.primary_metric, "n_test": len(ids),
            **{k: rs[k] for k in ("cluster_unit", "n_clusters")},
            "lora": point["lora"], "lora_seeds": seeds, "lora_sd": statistics.stdev(seeds) if len(seeds) > 1 else 0.0,
            "chosen_lr": sel["chosen_lr"], "epochs": [t["epoch"] for t in sel["tested"]], "val": [t["val_primary"] for t in sel["tested"]],
            "mean_best_val": sel["mean_best_val"], "val_curves": json.loads((HL.set_dir(stem) / "val.json").read_text())["runs"],
            "zero": zero, "calibrated": point["cal"], "optimized": point["opt"], "jev": point["jev"], "paper": paper,
            "encoder": {"best": best, "best_score": point["enc"], **{f: point[f] for f in FAMILIES}},
            "gap_lora": gap(point["lora"]), "gap_opt": gap(point["opt"]),
            "frozen_rule": {"lora": None if None in frozen else statistics.mean(frozen), "zero": cal["test"]["frozen_rule_zero_primary"]},
            "excluded_train_examples": len(data["train"]["excluded"]), "rejected_examples": sum(len(r["rejected"]) for r in lr_res),
            "gpu_hours": HL.measured_hours(stem),
            "peak_memory_gib": max((e["peak_memory_gib"] or 0.0) for lg in logs for e in lg["epoch_summaries"]),
            "contrasts": rs["contrasts"]}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_sets": len(rows)}
    for name, *_ in CONTRASTS:
        for p, r in zip(holm([r["contrasts"][name]["p"] for r in rows]), rows):
            r["contrasts"][name]["p_holm"] = p
        pooled = np.mean([r["contrasts"][name]["draws"] for r in rows], axis=0)
        cs = [r["contrasts"][name] for r in rows]
        out[name] = {"n": len(rows), "pooled_delta": statistics.mean(c["delta"] for c in cs),
                     "pooled_ci": [float(x) for x in np.percentile(pooled, [2.5, 97.5])], "pooled_p": boot_p(pooled),
                     "n_positive": sum(c["delta"] > 0 for c in cs), "median_delta": statistics.median(c["delta"] for c in cs),
                     "n_holm_positive": sum(c["p_holm"] < 0.05 and c["delta"] > 0 for c in cs),
                     "n_holm_negative": sum(c["p_holm"] < 0.05 and c["delta"] < 0 for c in cs)}
    out["gap"] = {"lora": medians(rows, lambda r: r["gap_lora"]), "opt": medians(rows, lambda r: r["gap_opt"])}
    out["gpu_hours"] = sum(r["gpu_hours"] or 0.0 for r in rows)
    out["peak_memory_gib"] = max(r["peak_memory_gib"] for r in rows)
    out["rejected_examples"] = sum(r["rejected_examples"] for r in rows)
    out["excluded_train_examples"] = sum(r["excluded_train_examples"] for r in rows)
    return out


def ci(c: dict[str, Any]) -> str:
    return f"{pct(c['delta'], True)} [{pct(c['ci'][0], True)}, {pct(c['ci'][1], True)}] p_Holm {c['p_holm']:.3f}"


def reading(x: dict[str, Any]) -> str:
    return (f"pooled {pct(x['pooled_delta'], True)} [{pct(x['pooled_ci'][0], True)}, {pct(x['pooled_ci'][1], True)}] over {x['n']} sets; "
            f"median {pct(x['median_delta'], True)}; Δ > 0 on {x['n_positive']}; LoRA ahead (Holm) on {x['n_holm_positive']}, "
            f"behind on {x['n_holm_negative']}, no clear difference on {x['n'] - x['n_holm_positive'] - x['n_holm_negative']}.")


def write_md(rep: dict[str, Any]) -> None:
    L = ["# H100 LoRA: Qwen3.8-27B fine-tuned on the pipeline's labels", "",
         "The study was pre-registered. Generated by `jev-opt/experiments/analyze_h100_lora.py`.",
         "Scores ×100 on each paper's primary metric over the gauntlet test items. LoRA = seed 0 only (the pre-registered reduction from three seeds). Δ: paired",
         "cluster bootstrap (RQ1 units, 2,000 resamples, seeds resampled), 95% CI, Holm over the sets (jackknife-t p on TB-Dense",
         "and MATRES).", "", f"Model `{rep['model']}`, template `{rep['template']}`. Status: {rep['status']}.", ""]
    if rep["sets"]:
        L += ["| Set | LoRA (sd) | lr | Qwen zero | Qwen cal. | Qwen opt. (O_B) | Δ LoRA − O_B [CI] | Best baseline | Δ LoRA − baseline [CI] | "
              "Jev | Δ LoRA − Jev [CI] | Paper | Gap LoRA / O_B |",
              "|---|---:|---:|---:|---:|---:|---|---:|---|---:|---|---:|---|"]
        for r in rep["sets"]:
            c, e = r["contrasts"], r["encoder"]
            L.append(f"| {r['title']} | {pct(r['lora'])} ({pct(r['lora_sd'])}) | {r['chosen_lr']:g} | {pct(r['zero'])} | {pct(r['calibrated'])} | "
                     f"{pct(r['optimized'])} | {ci(c['lora-opt'])} | {pct(e['best_score'])} ({e['best']}) | {ci(c['lora-enc'])} | {pct(r['jev'])} | "
                     f"{ci(c['lora-jev'])} | {pct(r['paper'])} | {ipct(r['gap_lora'])}% / {ipct(r['gap_opt'])}% |")
        s = rep["summary"]
        L += ["", *(f"- {LABEL[name]}: {reading(s[name])}" for name, *_ in CONTRASTS),
              f"- Median gap closed: LoRA {ipct(s['gap']['lora']['all'])}% (8 configurations {ipct(s['gap']['lora']['cfg8'])}%, 5 frozen sets "
              f"{ipct(s['gap']['lora']['frozen5'])}%); Qwen optimized questions {ipct(s['gap']['opt']['all'])}%.",
              "- DREsS under the gauntlet's frozen rule (argmax level), LoRA / Qwen zero-shot: "
              + (", ".join(f"{r['title']} {pct(r['frozen_rule']['lora'])} / {pct(r['frozen_rule']['zero'])}" for r in rep["sets"]
                           if r["frozen_rule"]["lora"] is not None) or "–") + ".",
              f"- Training examples without a target option: {s['excluded_train_examples']}; rejected test examples: {s['rejected_examples']}.",
              f"- Compute: {s['gpu_hours']:.1f} GPU-hours (training, val and test scoring, model loading); peak memory {s['peak_memory_gib']:.1f} GiB.",
              "", "Per seed (test) and validation:", "", "| Set | Seeds (test) | Best epochs | Val (best epoch) | Mean best val per lr |", "|---|---|---|---|---|"]
        for r in rep["sets"]:
            L.append(f"| {r['title']} | {', '.join(pct(x) for x in r['lora_seeds'])} | {r['epochs']} | {', '.join(pct(x) for x in r['val'])} | "
                     + ", ".join(f"{k}: {pct(v)}" for k, v in r["mean_best_val"].items()) + " |")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n")


def write_tex(rep: dict[str, Any]) -> None:
    rows = rep["sets"]
    m: list[tuple[str, str]] = [("loraStatus", "finished" if len(rows) == len(SETS) else ("pending" if not rows else f"{len(rows)} of {len(SETS)} sets"))]
    if rows:
        s = rep["summary"]
        m += [("loraNsets", str(len(rows))), ("loraNseeds", str(len(rows[0]["lora_seeds"])))]
        for r in rows:
            t = TITLE[r["set"]][1]
            m += [(f"loraScore{t}", pct(r["lora"])), (f"loraDopt{t}", pct(r["contrasts"]["lora-opt"]["delta"], True))]
        for name, _, _, key in CONTRASTS:
            x = s[name]
            m += [(f"loraPool{key}", pct(x["pooled_delta"], True)), (f"loraPool{key}Lo", pct(x["pooled_ci"][0], True)),
                  (f"loraPool{key}Hi", pct(x["pooled_ci"][1], True)), (f"loraMed{key}", pct(x["median_delta"], True)),
                  (f"loraNpos{key}", str(x["n_positive"])), (f"loraNholm{key}", str(x["n_holm_positive"])), (f"loraNneg{key}", str(x["n_holm_negative"]))]
        m += [("loraGap", ipct(s["gap"]["lora"]["all"])), ("loraGapOpt", ipct(s["gap"]["opt"]["all"])), ("loraGpuHours", f"{s['gpu_hours']:.1f}")]
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("% generated by jev-opt/experiments/analyze_h100_lora.py; do not edit\n"
                       + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m))


def main() -> int:
    done = {s: sel for s in SETS if (sel := finished(s)) is not None}
    rep: dict[str, Any] = {"model": HL.PIN, "template": HL.TEMPLATE_ID,
                           "bootstrap": {"B": B, "cluster_seed": 0, "seed_rng": SEED_RNG}, "sets": [],
                           "status": "finished" if len(done) == len(SETS) else "pending: " + ", ".join(s for s in SETS if s not in done)}
    if done:
        jev = jev_rows()
        jtp = json.loads((BB_DIR / "jev_test_preds.json").read_text())
        bb = json.loads((ROOT / "results" / "BUDGET_BASELINE.json").read_text())["sets"]
        rows = [set_row(s, sel, jev[s], jtp, bb[s]) for s, sel in done.items()]
        rep["summary"] = summarize(rows)
        for r in rows:
            for c in r["contrasts"].values():
                c.pop("draws", None)
        rep["sets"] = rows
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rep, indent=1) + "\n")
    write_md(rep)
    write_tex(rep)
    print(f"{len(done)} of {len(SETS)} sets finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
