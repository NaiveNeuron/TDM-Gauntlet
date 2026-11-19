"""Offline hybrid (Jev where confident, the paper's model elsewhere) and confident-subset coverage per pipeline step
-> results/offline_v4/hybrid.json, paper/generated/offline_hybrid_{sets,steps}.tex.

Offline only: saved runs/<task>/{predictions.jsonl, responses*.jsonl.gz}, the papers' released item-level
predictions under data/raw, and (part C) the jev-opt cache in cache-only mode through
jev-opt/experiments/confidence_readout.py (framing_features patches JevClient so that a cache miss raises). No Jev
call, no LLM call. Run from the repo root (Python 3.11):
    uv run --project jev-opt/python python jev-opt/experiments/offline_hybrid.py

DEFINITIONS (fixed before the results were computed)
=====================================================
A. Which of the 46 zero-shot test sets release item-level predictions of the paper's own model (checked: every
   scripts/prepare_*.py and every *pred*/*output*/*result* file under data/raw):
   - XSTest (Roettger et al. 2024): the paper's automated GPT-4 response judge, evaluation/automated_evaluation_labels/
     *_gpteval.csv (gpt4_label; GPT-4 non-answers = full compliance, paper fn. 10, as scripts/prepare_xstest.py);
     StrMatch (*_streval.csv) as a second reference. Item = model x prompt (xstest-<model>-<id>), aligned by
     (model, prompt id); gold asserted equal to the CSV's final_label; GPT-4-judge accuracy asserted = 0.804
     (results/paper_numbers.json["xstest"]).
   - CICC x5 (den Hengst et al. 2024): the fine-tuned BERT whose softmax the paper's conformal predictor uses,
     data/<ds>/bert/test_predictions.csv; paper-model prediction = argmax over all of BERT's classes (C150: the
     `oos` class included, so an in-scope item predicted oos is wrong). Item test-<hash>; gold asserted equal.
     The paper's reported metric ("Single") is not an accuracy; BERT accuracy is computed here.
   - LegalSeg (Nigam et al. 2025): ToInLegalBERT, LegalSeg_ToInLegalBert_Predictions (true, predicted per sentence
     in release order); the paper's headline Hier_BiLSTM-CRF predictions have 149,824 rows for 149,881 sentences
     and cannot be aligned (and do not reproduce, results/paper_numbers.json["legalseg"]); ToInLegalBERT
     reproduces its paper macro-F1 0.62 (asserted to 2 decimals). Unit = sentence (the 6 non-English sentences in
     data/frozen/legalseg/excluded_non_english.jsonl dropped), cluster = document; gold asserted equal.
   Not usable: MedReadMe (checkpoint_test_preds.json, continuous gold, no item-level correctness); ContempO
   StructTempRel outputs (an earlier model of Ning et al. 2017, not the paper's); FENICE repo outputs (AggreFact,
   other items); SMARTER llm_hate outputs (a different paper's GPT runs); CICC banking77 (B77 is not among the 46).
B. Hybrid. Per set: unit, Jev confidence (the field) and Jev correctness as scripts/confidence_zeroshot.py
   (load_choice / load_legalseg); paper correctness = the paper model's prediction equals gold.
   System prediction = Jev's where Jev's confidence >= tau, the paper model's elsewhere.
   tau by Learn-then-Test (Angelopoulos et al. 2021), delta = 0.05: coverage grid c = 0.05, 0.10, ..., 1.00 tested in
   that order (descending confidence threshold). At c, tau_c = the confidence of the ceil(c n)-th most confident
   fit-half unit (ties at tau_c all accepted); null H_c: Jev's error among accepted > target, rejected when the
   one-sided Clopper-Pearson upper bound Beta^-1(1 - delta; e + 1, m - e) <= target (m accepted, e errors).
   Fixed-sequence: stop at the first non-rejected c; tau = the last rejected tau_c (none rejected: tau = inf, the
   system is the paper model). Target = the paper model's error on the fit half. Halving: clusters split at random
   into halves A/B; tau fit on A applied to B and fit on B applied to A, pooled (every unit is scored by a threshold
   not fit on it); 50 halvings. Seeds: SeedSequence(20261001) spawned one child per set in table order -> (halvings,
   bootstrap).
   Reported per set: paper-model accuracy and Jev accuracy on the same units; mean over halvings of system accuracy,
   Jev coverage (share of units given Jev's prediction) and Jev's error among those.
   Tests (system vs paper model on the same units):
   - McNemar exact (two-sided binomial on discordant units) for halving 1; also the share of the 50 halvings with
     McNemar p < 0.05. Units are treated as independent (LegalSeg sentences are not: see the bootstrap).
   - Non-inferiority / TOST with margin m = 0.5 points (0.005 accuracy): D = mean over units of (s_i - p_i), s_i =
     unit i's system correctness averaged over the 50 halvings, p_i the paper's. Paired cluster bootstrap (2,000
     resamples, thresholds held fixed): p_NI = (1 + #{D* <= -m}) / 2001 (H0: D <= -m); p_upper = (1 + #{D* >= m}) /
     2001; p_TOST = max(p_NI, p_upper) (equivalence within +-m).
   - Holm over the sets, separately for McNemar, p_NI and p_TOST.
C. Confident-subset coverage per pipeline step, the 8 optimized configurations (jev-opt/experiments/
   confidence_readout.py, same test items and asserts): zero-shot = the paper's question (recorded prediction,
   confidence = its renormalised probability of the predicted label); calibrated = per-class log-prob bias +
   temperature on the unchanged question (jev_opt.fit.fit_label_bias on the 400 fit-slice train items, argmax
   prediction); optimized = the selected program's refitted readout (seeds 0/1/2, results/seeds.json; reported as
   the seed mean). Confidence = the method's probability of its predicted label; correct = exact label/level.
   No item-level predictions of these papers' models are released (none of the 8 is in part A), so targets are
   stated: Jev error among accepted <= 10% and <= 20% for every task, and for INLI also the paper's T5-XXL error
   1 - 0.924 (results/paper_numbers.json["inli"]). Same grid LTT, delta, halvings (test items split at random,
   SeedSequence(20261002) spawned one child per task in confidence_readout order).
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import binomtest

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))
import confidence_zeroshot as cz  # noqa: E402

OUT = ROOT / "results" / "offline_v4" / "hybrid.json"
TEX_SETS = ROOT / "paper" / "generated" / "offline_hybrid_sets.tex"
TEX_STEPS = ROOT / "paper" / "generated" / "offline_hybrid_steps.tex"
RAW = ROOT / "data" / "raw"
DELTA = 0.05
HALVINGS = 50
B = 2000
MARGIN = 0.005
GRID = tuple(round(0.05 * k, 2) for k in range(1, 21))
SEED_SETS = 20261001
SEED_STEPS = 20261002
PAPER_NUMBERS = json.loads((ROOT / "results" / "paper_numbers.json").read_text())
XSTEST_MODELS = ("llama2orig", "llama2new", "mistralinstruct", "mistralguard", "gpt4")
CICC = {"cicc/mtod": "mtod", "cicc/hwu64": "hwu64", "cicc/c150": "clinc150", "cicc/acid": "acid", "cicc/atis": "atis"}
LEGALSEG_DIGITS = ("None", "Facts", "Issue", "Arguments of Petitioner", "Arguments of Respondent", "Reasoning",
                   "Decision")  # paper label ids 0-6 (scripts/prepare_legalseg.py: AoP-3, AoR-4)
NAMES = {"xstest": "XSTest (GPT-4 judge)", "xstest_strmatch": "XSTest (StrMatch)", "cicc/mtod": "CICC MTOD",
         "cicc/hwu64": "CICC HWU64", "cicc/c150": "CICC C150", "cicc/acid": "CICC ACID", "cicc/atis": "CICC ATIS",
         "legalseg": "LegalSeg (ToInLegalBERT)"}


# ---------------------------------------------------------------------------------------------
# Learn-then-Test over a coverage grid
# ---------------------------------------------------------------------------------------------

def cp_upper(e: np.ndarray | int, m: np.ndarray | int, delta: float = DELTA) -> np.ndarray:
    """One-sided Clopper-Pearson upper bound of an error rate (e errors in m)."""
    e, m = np.asarray(e, float), np.asarray(m, float)
    return np.where(e >= m, 1.0, beta_dist.ppf(1 - delta, e + 1, np.maximum(m - e, 1)))


def ltt_grid(conf: np.ndarray, err: np.ndarray, target: float) -> float:
    """Lowest threshold of the fixed sequence over GRID (ascending coverage) whose CP upper bound <= target."""
    n = len(conf)
    s = np.sort(conf)[::-1]
    tau = float("inf")
    for c in GRID:
        t = s[math.ceil(c * n - 1e-9) - 1]
        acc = conf >= t
        if cp_upper(err[acc].sum(), acc.sum()) > target:
            break
        tau = float(t)
    return tau


def halves(rng: np.random.Generator, cl: np.ndarray) -> np.ndarray:
    n_cl = int(cl.max()) + 1
    in_a = np.zeros(n_cl, bool)
    in_a[rng.permutation(n_cl)[: n_cl // 2]] = True
    return in_a[cl]


def holm(p: dict[str, float]) -> dict[str, float]:
    return cz.holm(p)


# ---------------------------------------------------------------------------------------------
# Part A: paper-model item-level predictions aligned to our units
# ---------------------------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def xstest_units() -> tuple[cz.Units, dict[str, np.ndarray], dict[str, Any]]:
    u = cz.load_choice("xstest")
    gold = {p["item_id"]: p["gold"] for p in cz.predictions("xstest")}
    ref: dict[str, dict[str, str]] = {"gpteval": {}, "streval": {}}
    for model in XSTEST_MODELS:
        for kind, col in (("gpteval", "gpt4_label"), ("streval", "strmatch_label")):
            for r in read_csv(RAW / "xstest" / "evaluation" / "automated_evaluation_labels" / f"xstest_v2_completions_{model}_{kind}.csv"):
                iid = f"xstest-{model}-{r['id']}"
                assert r["final_label"] == gold[iid], iid
                lab = r[col]
                ref[kind][iid] = lab if lab in ("1_full_compliance", "2_full_refusal", "3_partial_refusal") else "1_full_compliance"
    ids = u.extra["item_id"]
    assert set(ids) == set(ref["gpteval"]) == set(ref["streval"]) and len(ids) == 2250
    out = {k: np.array([ref[k][i] == gold[i] for i in ids], float) for k in ref}
    want = PAPER_NUMBERS["xstest"]["paper"]["GPT-4 zero-shot response judge (App. D)"]
    assert abs(out["gpteval"].mean() - want) < 5e-4, out["gpteval"].mean()
    return u, out, {"n_aligned": len(ids), "gold_equal": True, "gpt4_judge_accuracy_check": want}


def cicc_units(task: str) -> tuple[cz.Units, np.ndarray, dict[str, Any]]:
    u = cz.load_choice(task)
    ds = CICC[task]
    d = RAW / "cicc" / "repo" / "data" / ds / "bert"
    names = [r[1] for r in list(csv.reader((d / "labels.csv").open(newline="", encoding="utf-8")))[1:]]
    rows = {f"test-{r['hash']}": r for r in read_csv(d / "test_predictions.csv")}
    gold = {p["item_id"]: p["gold"] for p in cz.predictions(task)}
    k = len([c for c in next(iter(rows.values())) if c.startswith("label_")])
    correct = []
    for iid in u.extra["item_id"]:
        r = rows[iid]
        g = names[int(r["label"])]
        assert g == gold[iid], (task, iid, g, gold[iid])
        pred = int(np.argmax([float(r[f"label_{j}"]) for j in range(k)]))
        correct.append(pred == int(r["label"]))
    return u, np.array(correct, float), {"n_aligned": len(correct), "gold_equal": True, "bert_classes": k}


def macro_f1(gold: list[int], pred: list[int], k: int) -> float:
    f = []
    for c in range(k):
        tp = sum(g == c == p for g, p in zip(gold, pred))
        fp = sum(p == c != g for g, p in zip(gold, pred))
        fn = sum(g == c != p for g, p in zip(gold, pred))
        f.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return sum(f) / k


def legalseg_units() -> tuple[cz.Units, np.ndarray, dict[str, Any]]:
    u = cz.load_legalseg("legalseg")
    rows = read_csv(RAW / "legalseg" / "predictions" / "LegalSeg_ToInLegalBert_Predictions__ToInlegalbertpredictions.csv")
    true = [int(r["true"]) for r in rows]
    pred = [int(r["predicted"]) for r in rows]
    f1 = macro_f1(true, pred, len(LEGALSEG_DIGITS))
    want = PAPER_NUMBERS["legalseg"]["paper"]["ToInLegalBERT"]
    assert round(f1, 2) == want, f1
    docs = [json.loads(x) for x in (ROOT / "data" / "frozen" / "legalseg" / "test.jsonl").read_text().splitlines()]
    assert sum(d["n_sentences"] for d in docs) == len(rows)
    excl = defaultdict(set)
    for x in (ROOT / "data" / "frozen" / "legalseg" / "excluded_non_english.jsonl").read_text().splitlines():
        e = json.loads(x)
        excl[e["doc_index"]].add(e["sentence"])
    per_doc, off = {}, 0
    for d in docs:
        assert [int(c) for c in d["gold"]] == true[off: off + d["n_sentences"]]
        per_doc[d["id"]] = (d["doc_index"], true[off: off + d["n_sentences"]], pred[off: off + d["n_sentences"]])
        off += d["n_sentences"]
    correct = []
    for p in cz.predictions("legalseg"):
        di, t, q = per_doc[p["item_id"]]
        # excluded positions: the sentence index whose removal makes the CSV gold equal ours (0- or 1-based)
        kept = None
        for base in (0, 1):
            keep = [i for i in range(len(t)) if i + base not in excl.get(di, set())]
            if [LEGALSEG_DIGITS[t[i]] for i in keep] == p["gold"]:
                kept = keep
                break
        assert kept is not None, p["item_id"]
        correct += [float(q[i] == t[i]) for i in kept]
    assert len(correct) == len(u.correct)
    return u, np.array(correct, float), {"n_aligned": len(correct), "gold_equal": True,
                                         "toinlegalbert_macro_f1_all_sentences": f1, "paper_macro_f1": want}


# ---------------------------------------------------------------------------------------------
# Part B: hybrid per set
# ---------------------------------------------------------------------------------------------

def hybrid(u: cz.Units, paper: np.ndarray, ss: np.random.SeedSequence) -> dict[str, Any]:
    conf, _, jev, cl, _ = u.arrays()
    jerr = 1 - jev
    h_ss, b_ss = ss.spawn(2)
    rng = np.random.default_rng(h_ss)
    sys_c = np.empty((HALVINGS, len(conf)))
    cov, acc, jev_err_acc, taus, targets, zero_cov = [], [], [], [], [], 0
    for h in range(HALVINGS):
        a = halves(rng, cl)
        use = np.zeros(len(conf), bool)
        for fit, score in ((a, ~a), (~a, a)):
            target = float(1 - paper[fit].mean())
            tau = ltt_grid(conf[fit], jerr[fit], target)
            use |= score & (conf >= tau)
            taus.append(tau)
            targets.append(target)
        s = np.where(use, jev, paper)
        sys_c[h] = s
        cov.append(use.mean())
        acc.append(s.mean())
        jev_err_acc.append(jerr[use].mean() if use.any() else float("nan"))
        zero_cov += not use.any()
    # McNemar on halving 1
    b = int(((sys_c[0] == 1) & (paper == 0)).sum())
    c = int(((sys_c[0] == 0) & (paper == 1)).sum())
    p_mc = float(binomtest(b, b + c).pvalue) if b + c else 1.0
    mc_all = [float(binomtest(int(((x == 1) & (paper == 0)).sum()), int((x != paper).sum())).pvalue) if (x != paper).any() else 1.0
              for x in sys_c]
    # paired cluster bootstrap of D = mean(s - p), s = halving-averaged system correctness
    s_bar = sys_c.mean(axis=0)
    diff = s_bar - paper
    n_cl = int(cl.max()) + 1
    cl_sum = np.bincount(cl, weights=diff, minlength=n_cl)
    cl_n = np.bincount(cl, minlength=n_cl).astype(float)
    brng = np.random.default_rng(b_ss)
    reps = np.empty(B)
    for i in range(B):
        w = np.bincount(brng.integers(0, n_cl, n_cl), minlength=n_cl).astype(float)
        reps[i] = (w * cl_sum).sum() / (w * cl_n).sum()
    d = float(diff.mean())
    p_ni = float((1 + (reps <= -MARGIN).sum()) / (B + 1))
    p_up = float((1 + (reps >= MARGIN).sum()) / (B + 1))
    tf = [t for t in taus if np.isfinite(t)]
    return {"n_units": len(conf), "n_clusters": n_cl, "paper_accuracy": float(paper.mean()), "jev_accuracy": float(jev.mean()),
            "system_accuracy_mean": float(np.mean(acc)), "system_accuracy_range": [float(min(acc)), float(max(acc))],
            "jev_coverage_mean": float(np.mean(cov)), "jev_coverage_range": [float(min(cov)), float(max(cov))],
            "jev_error_among_accepted_mean": float(np.nanmean(jev_err_acc)) if not np.isnan(jev_err_acc).all() else None,
            "target_error_mean": float(np.mean(targets)), "halvings_zero_coverage": zero_cov,
            "threshold_median": float(np.median(tf)) if tf else None, "fits_without_threshold": len(taus) - len(tf),
            "diff_system_minus_paper": d, "diff_ci": cz.ci(reps),
            "mcnemar_halving1": {"system_only_correct": b, "paper_only_correct": c, "p": p_mc},
            "mcnemar_share_p_below_05": float(np.mean(np.array(mc_all) < 0.05)),
            "p_noninferiority": p_ni, "p_upper": p_up, "p_tost": max(p_ni, p_up)}


# ---------------------------------------------------------------------------------------------
# Part C: confident-subset coverage per pipeline step (8 optimized configurations)
# ---------------------------------------------------------------------------------------------

def coverage_cv(conf: np.ndarray, correct: np.ndarray, target: float, rng: np.random.Generator) -> dict[str, float]:
    cl = np.arange(len(conf))
    err = 1 - correct
    cov, eacc, zero = [], [], 0
    for _ in range(HALVINGS):
        a = halves(rng, cl)
        use = np.zeros(len(conf), bool)
        for fit, score in ((a, ~a), (~a, a)):
            use |= score & (conf >= ltt_grid(conf[fit], err[fit], target))
        cov.append(use.mean())
        eacc.append(err[use].mean() if use.any() else np.nan)
        zero += not use.any()
    e = np.array(eacc)
    return {"coverage": float(np.mean(cov)), "error_accepted": float(np.nanmean(e)) if not np.isnan(e).all() else None,
            "halvings_zero_coverage": zero}


def steps() -> dict[str, Any]:
    import confidence_readout as cr  # noqa: PLC0415  (patches JevClient: cache-only)
    import framing_features as ff  # noqa: PLC0415
    from jev_opt.runs import _load_run_config  # noqa: PLC0415

    selection = json.loads((ROOT / "results" / "seeds.json").read_text())["selection"]
    inli_target = 1 - PAPER_NUMBERS["inli"]["paper"]["T5-XXL fine-tuned on INLI (4-way)"]
    out = {}
    for (slug, title), ss in zip(ff.TASKS, np.random.SeedSequence(SEED_STEPS).spawn(len(ff.TASKS))):
        sels = [selection[slug][str(s)] for s in ff.SEEDS]
        zs = cr.zero_shot(slug, list(_load_run_config(RUNS_JO / sels[0]["run"]).labels))
        runs = [cr.run_seed(slug, sel, zs) for sel in sels]
        ids, labels, gold = runs[0]["ids"], runs[0]["labels"], runs[0]["gold"]
        assert all(r["ids"] == ids for r in runs)
        z = [zs[i] for i in ids]
        ctrl = cr.calibrator_controls(slug, sels[0]["run"], labels)
        zm = cr.Method(gold, [x["pred"] for x in z], np.stack([x["dist"] for x in z]), labels)
        bm = cr.Method(gold, *cr.apply_control(z, labels, ctrl["bias"], ctrl["bias_temperature"], False), labels)
        rms = [cr.Method(gold, r["pred"], r["dist"], labels) for r in runs]
        targets = {"10": 0.10, "20": 0.20} | ({"paper": inli_target} if slug == "inli" else {})
        rng = np.random.default_rng(ss)
        res: dict[str, Any] = {"title": title, "n_test": len(ids), "accuracy": {}, "targets": targets}
        for name, ms in (("zero_shot", [zm]), ("calibrated", [bm]), ("optimized", rms)):
            res["accuracy"][name] = float(np.mean([m.correct.mean() for m in ms]))
            for tk, t in targets.items():
                per = [coverage_cv(m.conf, m.correct, t, rng) for m in ms]
                res.setdefault(tk, {})[name] = {
                    "coverage": float(np.mean([p["coverage"] for p in per])),
                    "error_accepted": (float(np.mean([p["error_accepted"] for p in per if p["error_accepted"] is not None]))
                                       if any(p["error_accepted"] is not None for p in per) else None),
                    "per_seed_coverage": [p["coverage"] for p in per]}
        out[slug] = res
        print(slug, {tk: {k: round(v["coverage"], 3) for k, v in res[tk].items()} for tk in targets}, flush=True)
    return out


RUNS_JO = ROOT / "runs" / "jev-opt"


# ---------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------

def pct(x: float | None, d: int = 1) -> str:
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{d}f}"


def pfmt(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def cname(s: str) -> str:
    return "".join(w.capitalize() for w in "".join(ch if ch.isalpha() else " " for ch in s).split())


def main() -> int:
    phase_order = ["xstest", "xstest_strmatch", "cicc/mtod", "cicc/hwu64", "cicc/c150", "cicc/acid", "cicc/atis", "legalseg"]
    streams = dict(zip(phase_order, np.random.SeedSequence(SEED_SETS).spawn(len(phase_order))))
    sets: dict[str, Any] = {}
    xu, xref, xchk = xstest_units()
    sets["xstest"] = hybrid(xu, xref["gpteval"], streams["xstest"]) | {"alignment": xchk, "paper_model": "GPT-4 response judge"}
    sets["xstest_strmatch"] = hybrid(xu, xref["streval"], streams["xstest_strmatch"]) | {"alignment": xchk, "paper_model": "StrMatch"}
    for t in CICC:
        u, p, chk = cicc_units(t)
        sets[t] = hybrid(u, p, streams[t]) | {"alignment": chk, "paper_model": "fine-tuned BERT (argmax)"}
    u, p, chk = legalseg_units()
    sets["legalseg"] = hybrid(u, p, streams["legalseg"]) | {"alignment": chk, "paper_model": "ToInLegalBERT"}
    for k in ("p_noninferiority", "p_tost"):
        adj = holm({s: v[k] for s, v in sets.items()})
        for s in sets:
            sets[s][k + "_holm"] = adj[s]
    adj = holm({s: v["mcnemar_halving1"]["p"] for s, v in sets.items()})
    for s in sets:
        sets[s]["mcnemar_halving1"]["p_holm"] = adj[s]
    for s, v in sets.items():
        print(s, f"paper {v['paper_accuracy']:.4f} jev {v['jev_accuracy']:.4f} sys {v['system_accuracy_mean']:.4f} "
              f"cov {v['jev_coverage_mean']:.3f} pNI {v['p_noninferiority_holm']:.3f} mc {v['mcnemar_halving1']['p']:.3g}", flush=True)

    st = steps()

    # macros
    macros: dict[str, str] = {"ovHyNSets": str(len(sets) - 1), "ovHyNPapers": "3", "ovHyMargin": "0.5",
                              "ovHyHalvings": str(HALVINGS), "ovHyDelta": f"{DELTA}"}
    for s, v in sets.items():
        n = cname(NAMES[s].split(" (")[0] + ("Str" if s == "xstest_strmatch" else ""))
        macros[f"ovHy{n}Paper"] = pct(v["paper_accuracy"])
        macros[f"ovHy{n}Jev"] = pct(v["jev_accuracy"])
        macros[f"ovHy{n}Sys"] = pct(v["system_accuracy_mean"])
        macros[f"ovHy{n}Cov"] = pct(v["jev_coverage_mean"])
        macros[f"ovHy{n}Diff"] = f"{100 * v['diff_system_minus_paper']:+.2f}"
        macros[f"ovHy{n}PNI"] = pfmt(v["p_noninferiority_holm"])
        macros[f"ovHy{n}PMc"] = pfmt(v["mcnemar_halving1"]["p_holm"])
    prim = {s: v for s, v in sets.items() if s != "xstest_strmatch"}
    macros["ovHyNNonInferior"] = str(sum(v["p_noninferiority_holm"] < 0.05 for v in prim.values()))
    macros["ovHyNEquivalent"] = str(sum(v["p_tost_holm"] < 0.05 for v in prim.values()))
    macros["ovHyNMcSig"] = str(sum(v["mcnemar_halving1"]["p_holm"] < 0.05 for v in prim.values()))
    macros["ovHyNPositiveCov"] = str(sum(v["jev_coverage_mean"] > 0 for v in prim.values()))
    for tk in ("10", "20"):
        w = {"10": "Ten", "20": "Twenty"}[tk]
        for name, short in (("zero_shot", "Zero"), ("calibrated", "Cal"), ("optimized", "Opt")):
            macros[f"ovHyStep{short}Cov{w}"] = pct(float(np.mean([st[t][tk][name]["coverage"] for t in st])))
            macros[f"ovHyStep{short}NPos{w}"] = str(sum(st[t][tk][name]["coverage"] > 0 for t in st))
    for name, short in (("zero_shot", "Zero"), ("calibrated", "Cal"), ("optimized", "Opt")):
        macros[f"ovHyStepInliPaper{short}"] = pct(st["inli"]["paper"][name]["coverage"])

    # tables
    t1 = ["% Generated by jev-opt/experiments/offline_hybrid.py. Do not edit.", r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
          r"Set (paper model) & Units & Paper acc. & Jev acc. & Hybrid acc. & Jev cov. & $\Delta$ [95\% CI] & $p_{\mathrm{NI}}$ (Holm) & McNemar $p$ (Holm) \\",
          r"\midrule"]
    for s, v in sets.items():
        lo, hi = v["diff_ci"]
        t1.append(f"{NAMES[s]} & {v['n_units']:,} & {pct(v['paper_accuracy'])} & {pct(v['jev_accuracy'])} & "
                  f"{pct(v['system_accuracy_mean'])} & {pct(v['jev_coverage_mean'])} & "
                  f"{100 * v['diff_system_minus_paper']:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] & "
                  f"{pfmt(v['p_noninferiority_holm'])} & {pfmt(v['mcnemar_halving1']['p_holm'])} \\\\")
    t1 += [r"\bottomrule", r"\end{tabular}", ""]
    TEX_SETS.parent.mkdir(parents=True, exist_ok=True)
    TEX_SETS.write_text("\n".join(t1))
    t2 = ["% Generated by jev-opt/experiments/offline_hybrid.py. Do not edit.", r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
          r" & & \multicolumn{3}{c}{accuracy} & \multicolumn{3}{c}{coverage, error $\le$ 10\%} & \multicolumn{2}{c}{optimized, error $\le$ 20\%} \\",
          r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
          r"Task & $n$ & Zero & Cal. & Opt. & Zero & Cal. & Opt. & Zero & Opt. \\", r"\midrule"]
    for t, v in st.items():
        a, c10, c20 = v["accuracy"], v["10"], v["20"]
        t2.append(f"{v['title']} & {v['n_test']} & {pct(a['zero_shot'])} & {pct(a['calibrated'])} & {pct(a['optimized'])} & "
                  f"{pct(c10['zero_shot']['coverage'])} & {pct(c10['calibrated']['coverage'])} & {pct(c10['optimized']['coverage'])} & "
                  f"{pct(c20['zero_shot']['coverage'])} & {pct(c20['optimized']['coverage'])} \\\\")
    t2 += [r"\bottomrule", r"\end{tabular}", ""]
    TEX_STEPS.write_text("\n".join(t2))

    md = ["## Hybrid: Jev where confident, the paper's model elsewhere", "",
          "Definitions: docstring of `jev-opt/experiments/offline_hybrid.py`. Item-level predictions of the paper's own model "
          "are released and alignable for XSTest (GPT-4 judge; StrMatch as second reference), CICC x5 (fine-tuned BERT) and "
          "LegalSeg (ToInLegalBERT; the headline Hier-BiLSTM-CRF file does not align). Threshold: grid Learn-then-Test "
          f"(coverage 0.05..1.00, delta {DELTA}, Clopper-Pearson upper bound on Jev's error among accepted <= the paper "
          f"model's error on the fit half), {HALVINGS} cross-fitted halvings. Non-inferiority margin 0.5 points (paired "
          "cluster bootstrap), McNemar on halving 1, Holm over the 7 primary sets (+ StrMatch row).", "",
          "| Set | Units | Paper acc. | Jev acc. | Hybrid acc. | Jev coverage | Jev err. accepted | Δ pts [CI] | p_NI Holm | p_TOST Holm | McNemar p Holm |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s, v in sets.items():
        lo, hi = v["diff_ci"]
        md.append(f"| {NAMES[s]} | {v['n_units']} | {pct(v['paper_accuracy'])} | {pct(v['jev_accuracy'])} | "
                  f"{pct(v['system_accuracy_mean'])} | {pct(v['jev_coverage_mean'])} | {pct(v['jev_error_among_accepted_mean'])} | "
                  f"{100 * v['diff_system_minus_paper']:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] | {pfmt(v['p_noninferiority_holm'])} | "
                  f"{pfmt(v['p_tost_holm'])} | {pfmt(v['mcnemar_halving1']['p_holm'])} |")
    md += ["", "### Confident-subset coverage per pipeline step (8 optimized configurations)", "",
           "Grid LTT as above on the test items; stated targets (no item-level paper predictions for these tasks): Jev "
           "error among accepted <= 10% / 20%; INLI also the paper's T5-XXL error (7.6%). Optimized = mean of seeds 0/1/2.", "",
           "| Task | n | Acc. zero / cal. / opt. | Cov. ≤10% zero / cal. / opt. | Cov. ≤20% zero / cal. / opt. |",
           "|---|---:|---|---|---|"]
    for t, v in st.items():
        a = v["accuracy"]
        md.append(f"| {v['title']} | {v['n_test']} | {pct(a['zero_shot'])} / {pct(a['calibrated'])} / {pct(a['optimized'])} | "
                  + " | ".join(" / ".join(pct(v[tk][m]["coverage"]) for m in ("zero_shot", "calibrated", "optimized")) for tk in ("10", "20"))
                  + " |")
    ip = st["inli"]["paper"]
    md += ["", f"INLI at the paper's error (7.6%): coverage zero {pct(ip['zero_shot']['coverage'])}%, calibrated "
           f"{pct(ip['calibrated']['coverage'])}%, optimized {pct(ip['optimized']['coverage'])}%."]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"macros": macros, "md": "\n".join(md) + "\n", "sets": sets, "steps": st,
                               "grid": list(GRID), "delta": DELTA, "halvings": HALVINGS, "margin": MARGIN,
                               "seeds": {"sets": SEED_SETS, "steps": SEED_STEPS}}, indent=1, allow_nan=True) + "\n")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
