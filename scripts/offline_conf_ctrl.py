"""Accuracy-controlled confidence analyses (offline; no Jev or LLM calls).

Inputs: results/confidence.json (per-set statistics written by scripts/confidence_zeroshot.py from saved
predictions/responses; 38 zero-shot sets with a confidence loader) and, for (c), the saved first-run responses
(runs/<task>/responses*) plus the saved zero-shot re-runs runs/jev-opt/study-<slug>-zero (same loaders and
item matching as confidence_zeroshot.rerun()).

Definitions
(a) Accuracy-controlled cause comparison. For each set s: accuracy a_s and y_s in {AUROC, ECE, top-band error
    rate (confidence band [.75, 1]), AUGRC}, exactly as in results/confidence.json (revision 3). Fit y_s = b0 + b1 a_s
    by weighted least squares on all sets where y is defined, each set weighted 1/(number of sets of its paper)
    (papers count equally). Residual r_s = y_s - fit. For a group g, the unit is a paper: its residual is the mean of
    r over its sets in g. The contrast is mean(paper residuals, conventions) - mean(paper residuals, other), other in
    {matched, unassigned}. p: exact two-sided permutation test, enumerating every relabelling of the pooled papers
    (C(10,4) = 210 and C(16,6) = 8008), p = share with |stat| >= |observed| (with a 1e-12 tolerance). Holm over
    4 metrics x 2 contrasts. The raw (unadjusted) paper-level contrast is also reported. Sensitivity: OLS of y on
    accuracy plus cause dummies (reference = matched), paper-weighted, coefficients only. R^2 is the weighted R^2 of
    the accuracy-only fit.
(b) Opaque label spaces (descriptive). For each opaque set: its residual from (a), and an accuracy-matched contrast:
    its y minus the mean y of the K = 3 non-opaque sets nearest in accuracy (ties broken by set name).
(c) Label flips. Pairs = first saved test response vs the zero-shot re-run of the identical request
    (INLI, TB-Dense, implicit hate). Flip = recorded first-run label != re-run choice (as in confidence_zeroshot).
    Margin = p1 - p2 of the first run's normalised option probabilities (0 for a tie at the top; a tie is counted
    separately). AUROC of -margin for flip (ties = 1/2), share of flips and share of requests with margin < 0.2 and
    < 0.1, AUROC of the low confidence field for comparison, and the share of byte-identical probability vectors.
"""
from __future__ import annotations

import gzip
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import confidence_zeroshot as cz  # noqa: E402

OUT = ROOT / "results" / "offline_v4" / "conf_ctrl.json"
TEX = ROOT / "paper" / "generated" / "offline_conf_ctrl_adjusted.tex"
METRICS = (("auroc", "auroc", "AUROC"), ("ece", "ece", "ECE"), ("top_rate", "top_band_error_rate", "Top-band error"),
           ("augrc", "augrc", "AUGRC"))
CONTRASTS = ("matched", "unassigned")
K_NEAR = 3


def wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    sw = np.sqrt(w)
    beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    res = y - X @ beta
    ybar = np.average(y, weights=w)
    return beta, 1 - float(np.sum(w * res**2) / np.sum(w * (y - ybar) ** 2))


def auroc(score: np.ndarray, pos: np.ndarray) -> float:
    from scipy.stats import rankdata
    r = rankdata(score)
    n1 = pos.sum()
    n0 = len(pos) - n1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def exact_perm(units: dict[str, float], a: list[str], b: list[str]) -> float:
    pool = a + b
    vals = np.array([units[p] for p in pool])
    obs = vals[: len(a)].mean() - vals[len(a):].mean()
    tot, hit = 0, 0
    for idx in combinations(range(len(pool)), len(a)):
        m = np.zeros(len(pool), bool)
        m[list(idx)] = True
        s = vals[m].mean() - vals[~m].mean()
        hit += abs(s) >= abs(obs) - 1e-12
        tot += 1
    return hit / tot


def part_a(sets: dict) -> dict:
    papers = {}
    for s, v in sets.items():
        papers.setdefault(v["paper"], []).append(s)
    out = {"per_metric": {}, "tests": {}}
    pvals = {}
    for key, field, _ in METRICS:
        names = sorted(s for s, v in sets.items() if v.get(field) is not None and np.isfinite(v[field]))
        a = np.array([sets[s]["accuracy"] for s in names])
        y = np.array([sets[s][field] for s in names])
        w = np.array([1 / sum(sets[t].get(field) is not None and np.isfinite(sets[t][field]) for t in papers[sets[s]["paper"]])
                      for s in names])
        beta, r2 = wls(np.c_[np.ones_like(a), a], y, w)
        resid = dict(zip(names, (y - (beta[0] + beta[1] * a)).tolist()))
        causes = [c for c in cz.CAUSE_ORDER if c != "matched"]
        D = np.c_[np.ones_like(a), a, *[[sets[s]["cause"] == c for s in names] for c in causes]].astype(float)
        bd, _ = wls(D, y, w)
        bu, _ = wls(D, y, np.ones_like(w))
        _, r2u = wls(np.c_[np.ones_like(a), a], y, np.ones_like(w))
        unit = {}  # (cause, paper) -> mean residual
        raw = {}
        for s in names:
            unit.setdefault((sets[s]["cause"], sets[s]["paper"]), []).append(s)
        res_u = {k: float(np.mean([resid[s] for s in v])) for k, v in unit.items()}
        raw_u = {k: float(np.mean([sets[s][field] for s in v])) for k, v in unit.items()}
        group = lambda g, d: {p: x for (c, p), x in d.items() if c == g}  # noqa: E731
        out["per_metric"][key] = {"n_sets": len(names), "intercept": float(beta[0]), "slope": float(beta[1]), "r2": r2,
                                  "cause_dummies": dict(zip(causes, bd[2:].tolist())),
                                  "r2_unweighted": r2u, "cause_dummies_unweighted": dict(zip(causes, bu[2:].tolist())), "residuals": resid,
                                  "group_resid": {g: float(np.mean(list(group(g, res_u).values()))) for g in cz.CAUSE_ORDER},
                                  "group_raw": {g: float(np.mean(list(group(g, raw_u).values()))) for g in cz.CAUSE_ORDER}}
        for other in CONTRASTS:
            cu, ou = group("conventions", res_u), group(other, res_u)
            lab = {("c", p): x for p, x in cu.items()} | {("o", p): x for p, x in ou.items()}
            p = exact_perm(lab, [k for k in lab if k[0] == "c"], [k for k in lab if k[0] == "o"])
            cr, orw = group("conventions", raw_u), group(other, raw_u)
            t = f"conventions_minus_{other}"
            out["tests"].setdefault(t, {})[key] = {
                "adjusted": float(np.mean(list(cu.values())) - np.mean(list(ou.values()))),
                "raw": float(np.mean(list(cr.values())) - np.mean(list(orw.values()))),
                "p": p, "n_papers": [len(cu), len(ou)], "n_perms": len(list(combinations(range(len(lab)), len(cu))))}
            pvals[f"{t}:{key}"] = p
    adj = cz.holm(pvals)
    for k, p in adj.items():
        t, m = k.split(":")
        out["tests"][t][m]["p_holm"] = p
    return out


def part_b(sets: dict, a_out: dict) -> dict:
    res = {}
    for key, field, _ in METRICS:
        ok = {s: v for s, v in sets.items() if v.get(field) is not None and np.isfinite(v[field])}
        others = sorted((s for s, v in ok.items() if v["cause"] != "opaque"))
        for s in sorted(t for t, v in ok.items() if v["cause"] == "opaque"):
            near = sorted(others, key=lambda t: (abs(ok[t]["accuracy"] - ok[s]["accuracy"]), t))[:K_NEAR]
            res.setdefault(s, {"name": ok[s]["name"], "paper": ok[s]["paper"], "accuracy": ok[s]["accuracy"]})[key] = {
                "value": ok[s][field], "residual": a_out["per_metric"][key]["residuals"][s],
                "matched_sets": near, "matched_mean": float(np.mean([ok[t][field] for t in near])),
                "matched_acc": float(np.mean([ok[t]["accuracy"] for t in near])),
                "diff": ok[s][field] - float(np.mean([ok[t][field] for t in near]))}
    summ = {k: {"mean_diff": float(np.mean([v[k]["diff"] for v in res.values() if k in v])),
                "mean_resid": float(np.mean([v[k]["residual"] for v in res.values() if k in v]))} for k, _, _ in METRICS}
    return {"sets": res, "summary": summ}


def part_c() -> dict:
    specs = [("inli", "relation", "inli"), ("contempo-tbdense", "relation", "contempo/tbdense"),
             ("smarter-implicit-hate", "label", "smarter/implicit_hate")]
    rows = []
    for slug, qid, task in specs:
        first = {r["item_id"]: r["response"]["answers"][qid] for r in cz.responses(task)}
        recorded = {p["item_id"]: p["prediction"]["label"] for p in cz.predictions(task)}
        ids = {}
        for line in (ROOT / "jev-opt" / "experiments" / slug.replace("-", "_") / "data" / "test.jsonl").read_text().splitlines():
            r = json.loads(line)
            ids[tuple(v for k, v in r.items() if k not in ("id", "label"))] = r["id"]
        for shard in sorted((cz.STUDY / f"study-{slug}-zero").glob("jev_responses*.jsonl.gz")):
            for line in gzip.open(shard, "rt"):
                r = json.loads(line)
                if r["split"] != "test":
                    continue
                iid = ids[tuple(r["request"]["state"].values())]
                a1, a2 = first[iid], r["response"]["answers"][qid]
                q1 = {k: max(float(v), 0.0) for k, v in a1["probabilities"].items()}
                q2 = {k: max(float(v), 0.0) for k, v in a2["probabilities"].items()}
                p = np.array(sorted(q1.values(), reverse=True) + [0.0])
                p = p / p.sum()
                p2 = sorted(q2.values(), reverse=True) + [0.0]
                rows.append({"task": task, "conf": float(a1["confidence"]), "margin": float(p[0] - p[1]),
                             "tie": bool(p[0] == p[1]), "tie_either": bool(p[0] == p[1] or p2[0] == p2[1]), "flip": recorded[iid] != a2["choice"],
                             "same_vec": all(q1.get(k, 0.0) == q2.get(k, 0.0) for k in q1.keys() | q2.keys())})
    m = np.array([r["margin"] for r in rows])
    c = np.array([r["conf"] for r in rows])
    f = np.array([r["flip"] for r in rows])
    tie = np.array([r["tie"] for r in rows])
    tie2 = np.array([r["tie_either"] for r in rows])
    return {"n": len(rows), "n_flips": int(f.sum()), "ties": int(tie.sum()), "flips_at_ties": int((f & tie).sum()),
            "ties_either_run": int(tie2.sum()), "flips_at_ties_either_run": int((f & tie2).sum()),
            "auroc_margin": auroc(-m, f), "auroc_conf": auroc(-c, f),
            "share_flips_m02": float(f[m < 0.2].sum() / f.sum()), "share_req_m02": float((m < 0.2).mean()),
            "share_flips_m01": float(f[m < 0.1].sum() / f.sum()), "share_req_m01": float((m < 0.1).mean()),
            "flips_conf_lt_06": int(f[c < 0.6].sum()), "share_req_conf_lt_06": float((c < 0.6).mean()),
            "identical_vec": float(np.mean([r["same_vec"] for r in rows]))}


def pfmt(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.2f}" if p >= 0.01 else f"{p:.3f}"


def main() -> int:
    conf = json.loads((ROOT / "results" / "confidence.json").read_text())
    sets = conf["sets"]
    a = part_a(sets)
    b = part_b(sets, a)
    c = part_c()
    pm = a["per_metric"]
    macros = {
        "ovCcRsqAuroc": f"{pm['auroc']['r2']:.2f}", "ovCcRsqEce": f"{pm['ece']['r2']:.2f}",
        "ovCcRsqTop": f"{pm['top_rate']['r2']:.2f}", "ovCcRsqAugrc": f"{pm['augrc']['r2']:.2f}",
        "ovCcNSets": str(len(sets)), "ovCcNPapers": str(len({v['paper'] for v in sets.values()})),
        "ovCcOpqAurocResid": f"{b['summary']['auroc']['mean_resid']:+.2f}",
        "ovCcOpqEceResid": f"{b['summary']['ece']['mean_resid']:+.2f}",
        "ovCcOpqAurocMatched": f"{b['summary']['auroc']['mean_diff']:+.2f}",
        "ovCcOpqEceMatched": f"{b['summary']['ece']['mean_diff']:+.2f}",
        "ovCcFlipN": str(c["n_flips"]), "ovCcFlipPairs": f"{c['n']:,}".replace(",", "{,}"),
        "ovCcFlipAurocMargin": f"{c['auroc_margin']:.2f}", "ovCcFlipAurocConf": f"{c['auroc_conf']:.2f}",
        "ovCcFlipShareMarginTwo": f"{100 * c['share_flips_m02']:.0f}\\%",
        "ovCcFlipReqMarginTwo": f"{100 * c['share_req_m02']:.0f}\\%",
        "ovCcFlipShareMarginOne": f"{100 * c['share_flips_m01']:.0f}\\%",
        "ovCcFlipReqMarginOne": f"{100 * c['share_req_m01']:.0f}\\%",
        "ovCcFlipTies": str(c["ties"]), "ovCcFlipAtTies": str(c["flips_at_ties"]),
    }
    words = {"auroc": "Auroc", "ece": "Ece", "top_rate": "Top", "augrc": "Augrc"}
    for key, w in words.items():
        for other, o in (("matched", "Mat"), ("unassigned", "Una")):
            d = a["tests"][f"conventions_minus_{other}"][key]
            macros[f"ovCcAdj{w}{o}"] = f"{d['adjusted']:+.3f}"
            macros[f"ovCcRaw{w}{o}"] = f"{d['raw']:+.3f}"
            macros[f"ovCcP{w}{o}"] = pfmt(d["p_holm"])
            macros[f"ovCcPraw{w}{o}"] = pfmt(d["p"])

    md = ["## Accuracy-controlled confidence (conf_ctrl)", "",
          f"Source: `scripts/offline_conf_ctrl.py` over `results/confidence.json` ({len(sets)} sets, "
          f"{macros['ovCcNPapers']} papers) and the saved re-runs. Paper-weighted fit of each metric on set accuracy; "
          "conventions vs other groups on paper-mean residuals, exact permutation over papers, Holm over 4 metrics x 2 contrasts.", "",
          "R² and slope from the paper-weighted fit; unweighted R²: " + ", ".join(f"{l} {pm[k]['r2_unweighted']:.2f}" for k, _, l in METRICS) + ".", "",
          "| Metric | R² (acc) | slope | conv−matched raw | adjusted | p (Holm) | conv−unassigned raw | adjusted | p (Holm) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, _, lab in METRICS:
        m_, u_ = a["tests"]["conventions_minus_matched"][key], a["tests"]["conventions_minus_unassigned"][key]
        md.append(f"| {lab} | {pm[key]['r2']:.2f} | {pm[key]['slope']:+.2f} | {m_['raw']:+.3f} | {m_['adjusted']:+.3f} | "
                  f"{m_['p']:.2f} ({m_['p_holm']:.2f}) | {u_['raw']:+.3f} | {u_['adjusted']:+.3f} | {u_['p']:.2f} ({u_['p_holm']:.2f}) |")
    md += ["", "Opaque label spaces (descriptive; residual from the accuracy fit; difference to the 3 non-opaque sets nearest in accuracy):", "",
           "| Set | Acc | AUROC | resid | vs matched-acc | ECE | resid | vs matched-acc |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s, v in b["sets"].items():
        e = v.get("ece")
        ece = f"{e['value']:.3f} | {e['residual']:+.3f} | {e['diff']:+.3f}" if e else "-- | -- | --"
        md.append(f"| {v['name']} | {v['accuracy']:.3f} | {v['auroc']['value']:.3f} | {v['auroc']['residual']:+.3f} | "
                  f"{v['auroc']['diff']:+.3f} | {ece} |")
    md += ["", f"Flips: {c['n_flips']}/{c['n']} pairs. AUROC of low margin for a flip {c['auroc_margin']:.3f} "
           f"(confidence field {c['auroc_conf']:.3f}). Margin < 0.2: {100*c['share_req_m02']:.1f}% of requests, "
           f"{100*c['share_flips_m02']:.1f}% of flips; < 0.1: {100*c['share_req_m01']:.1f}% / {100*c['share_flips_m01']:.1f}%. "
           f"Top ties: {c['ties']} requests, {c['flips_at_ties']} flips. Byte-identical probability vectors: {100*c['identical_vec']:.1f}%."]

    tex = ["\\begin{tabular}{lrrrrrrrr}", "\\toprule",
           " & & & \\multicolumn{3}{c}{Conventions $-$ matched} & \\multicolumn{3}{c}{Conventions $-$ unassigned} \\\\",
           "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}",
           "Metric & $R^2$ & Slope & Raw & Adjusted & $p_{\\mathrm{Holm}}$ & Raw & Adjusted & $p_{\\mathrm{Holm}}$ \\\\", "\\midrule"]
    for key, _, lab in METRICS:
        m_, u_ = a["tests"]["conventions_minus_matched"][key], a["tests"]["conventions_minus_unassigned"][key]
        tex.append(f"{lab} & {pm[key]['r2']:.2f} & ${pm[key]['slope']:+.2f}$ & ${m_['raw']:+.3f}$ & ${m_['adjusted']:+.3f}$ & "
                   f"{pfmt(m_['p_holm'])} & ${u_['raw']:+.3f}$ & ${u_['adjusted']:+.3f}$ & {pfmt(u_['p_holm'])} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", ""]
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text("\n".join(tex))
    otex = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
            "Set & Accuracy & AUROC & Residual & vs.\\ acc.-matched & ECE & Residual \\\\", "\\midrule"]
    for s, v in b["sets"].items():
        otex.append(f"{v['name']} & {v['accuracy']:.3f} & {v['auroc']['value']:.3f} & ${v['auroc']['residual']:+.3f}$ & "
                    f"${v['auroc']['diff']:+.3f}$ & " + (f"{v['ece']['value']:.3f} & ${v['ece']['residual']:+.3f}$" if "ece" in v else "-- & --") + " \\\\")
    otex += ["\\bottomrule", "\\end{tabular}", ""]
    (TEX.parent / "offline_conf_ctrl_opaque.tex").write_text("\n".join(otex))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"macros": macros, "md": "\n".join(md) + "\n", "accuracy_adjusted": a, "opaque": b,
                               "flips": c}, indent=1) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
