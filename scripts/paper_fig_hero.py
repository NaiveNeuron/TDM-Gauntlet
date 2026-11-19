# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pyyaml"]
# ///
"""Hero figure: per-set test scores relative to the task-specific method (paper/generated/hero_ladder{,_all}.pdf, caption macro hero_numbers.tex).

One row per test set, x = score as a fraction of the paper's task-specific method. Markers: the
paper's prompted-LLM baseline, Jev zero-shot (the zero-shot evaluation, scripts/paper_tables.py rows()), Jev with a
Jev-native question written from the task definition (zero-shot), and the pre-registered optimized
configuration. Only test numbers from saved files are used; the optimization results come from these
files (a missing run file is an error):

- v3 chosen (V3_CHOSEN, fixed on validation before any test use): the mean and sd over optimizer seeds 0-2 of the headline
  configuration (results/seeds.json; seed 0 = runs/jev-opt/<run>/test_consumed.json of the registered
  representative, cross-checked); Jev-native zero-shot: runs/jev-opt/v3-<task>-N (top-quality).
- Fitted readout on the audit's zero-shot answers (results/heads.json), for audit sets
  without another optimized marker.
- Lever C (candidate reranking and fitted thresholds): runs/{ttp_v3/*,relemb_v3/wikiwd,made_v3}/scores.json.
- v1 GEPA (results/gepa.json) only where no v3 test exists: the condition with the higher validation
  score (report.md of its top-quality representative), never chosen on test.

Files are read from disk under the repository root. Run from the repo root:

    uv run scripts/paper_fig_hero.py [-v]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import paper_figs_body as fb  # noqa: E402  (also sets the paper's rcParams)
import paper_tables as pt  # noqa: E402

OUT = ROOT / "paper" / "generated"
# Chosen configuration of question optimization (v3) per task, fixed on validation before any test use:
# slug -> (run, representative).
V3_CHOSEN = {
    "contempo-tbdense": ("v3-contempo-tbdense-P", "top-single"),
    "ezstance-subtaskA": ("v3-ezstance-subtaskA-P", "top-single"),
    "smarter-implicit-hate": ("v3-smarter-implicit-hate-P", "top-stacked"),
    "edit-intent": ("v3-edit-intent-P", "top-stacked"),
    "inli": ("v3-inli-P", "top-single"),
    "dress-new-content": ("v3-dress-new-content-P", "top-single"),
    "dress-new-organization": ("v3-dress-new-organization-P", "top-single"),
    "dress-new-language": ("v3-dress-new-language-P", "top-single"),
}

# Optimization-study slug -> (audit task, DREsS rubric index into paper_per_rubric_content_org_lang).
STUDY = {
    "contempo-tbdense": ("contempo/tbdense", None),
    "ezstance-subtaskA": ("ezstance/subtaskA", None),
    "smarter-implicit-hate": ("smarter/implicit_hate", None),
    "edit-intent": ("edit_intent", None),
    "inli": ("inli", None),
    "dress-new-content": ("dress_new", 0),
    "dress-new-organization": ("dress_new", 1),
    "dress-new-language": ("dress_new", 2),
}
RUBRICS = ("content", "organization", "language")
# Lever C (candidate reranking and fitted thresholds): audit task -> v3 gauntlet task (same test set and metric).
LEVER_C = {"ttp/tram": "ttp_v3/tram", "ttp/procedures_plus": "ttp_v3/procedures_plus", "ttp/expert": "ttp_v3/expert",
           "relemb/wikiwd": "relemb_v3/wikiwd", "made": "made_v3"}

XMIN, XMAX = -0.02, 1.3
JEV = "#1864ab"
JEV_V1 = "#74c0fc"
LLM = "#6c757d"
BAND = {"replaces": "#e8eef5", "between": "#fbf4e2", "below": "#f7ecef"}
HEAD = {"replaces": "#004488", "between": "#8a6416", "below": "#9c4155"}  # fb.COLORS, yellow and rose darkened for text
# DREsS's study baseline decodes the paper's question with Score cutoffs fit on training data, so it is not prompt
# only. The figures plot the prompt-only score (the zero-shot evaluation's levels, runs/dress_new/predictions.jsonl)
# as the zero-shot marker and the fitted-cutoff score as a separate square.
FITTED_ZERO = dict(marker="s", mfc="white", mec=JEV, mew=0.9, ms=4.0)
# Best trained baseline on the same labels; shape, not colour, separates it from Jev's markers.
TRAINED = dict(marker="X", mfc="black", mec="white", mew=0.4, ms=5.0)
# Clef answering Jev's optimized questions; a second hue and shape.
CLEF = dict(marker="^", mfc="#e8590c", mec="white", mew=0.4, ms=4.6)
DRESS_LEVELS = [f"{x / 2:.1f}" for x in range(2, 11)]


def dress_prompt_only() -> dict[str, float]:
    """Per-rubric QWK (9 levels) of the zero-shot evaluation's saved DREsS predictions; no fitted number."""
    import numpy as np

    rows = [json.loads(x) for x in (ROOT / "runs" / "dress_new" / "predictions.jsonl").read_text().splitlines() if x.strip()]
    idx = {lab: i for i, lab in enumerate(DRESS_LEVELS)}
    k = len(DRESS_LEVELS)
    out = {}
    for rub in RUBRICS:
        o = np.zeros((k, k))
        for r in rows:
            o[idx[f"{float(r['gold'][rub]):.1f}"], idx[f"{float(r['prediction'][rub]):.1f}"]] += 1
        e = np.outer(o.sum(1), o.sum(0)) / o.sum()
        w = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2
        out[rub] = float(1 - (w * o).sum() / (w * e).sum())
    return out


TITLES = {"replaces": "Zero-shot Jev matches or beats the task-specific method",
          "between": "Zero-shot Jev between the paper's prompted LLM and its task-specific method",
          "below": "Zero-shot Jev below both"}


def run_file(rel: str) -> str:
    """Content of the run file `rel` under the repository root; a missing file is an error, never skipped."""
    p = ROOT / rel
    if not p.is_file():
        raise FileNotFoundError(f"run file {rel} not found under {ROOT}: the hero figure needs the optimization-study "
                                "and lever C runs")
    return p.read_text()


def test_primary(run: str, representative: str) -> float:
    """Test primary metric of a jev-opt run, only if the consumed representative is the registered one."""
    d = json.loads(run_file(f"runs/jev-opt/{run}/test_consumed.json"))
    if d["representative"] != representative:
        raise ValueError(f"{run}: test consumed by {d['representative']}, registered {representative}")
    return float(d["metrics"]["primary"])


def v3_chosen() -> dict[str, tuple[str, str]]:
    """slug -> (run, representative) of the chosen v3 configurations."""
    return dict(V3_CHOSEN)


def v1_best(slug: str, conds: dict) -> float:
    """v1 GEPA test score of the condition with the higher validation score (never chosen on test)."""
    vals = {c: fb.report_val_test(fb.STUDY / f"study-{slug}-{c}")[0] for c in ("text", "program")}
    return conds[max(vals, key=vals.get)]["value"]


def lever_c(task: str) -> float:
    return float(json.loads(run_file(f"runs/{LEVER_C[task]}/scores.json"))[pt.PAPER[task]["metric"]])


def seed_mean(slug: str, run: str, rep: str) -> tuple[float, float]:
    """Mean and sd over optimizer seeds of the headline configuration; seed 0 must be the registered test."""
    seed0 = test_primary(run, rep)
    t = json.loads((ROOT / "results" / "seeds.json").read_text())["tasks"][slug]
    if t["seeds"]["0"]["run"] != run or abs(t["seeds"]["0"]["test"] - seed0) > 1e-9:
        raise ValueError(f"{slug}: seed 0 of seeds.json is not the registered test")
    return t["test_mean"], t["test_sd"]


def category(zero: float, paper: float, llm: float | None) -> str:
    """paper_tables.rows()'s outcome rule, for rows it does not produce (DREsS rubrics)."""
    if zero >= paper - 0.005:
        return "replaces"
    if llm is not None and zero >= llm - 0.005:
        return "between"
    if llm is None and zero / paper >= 0.8:
        return "between"
    return "below"


def ladder_rows() -> list[dict]:
    """Every comparable audit row, DREsS split into its rubrics, with all available markers (raw scores)."""
    gepa = {e["slug"]: e for e in json.loads((ROOT / "results" / "gepa.json").read_text())["rows"]}
    chosen = v3_chosen()
    heads = json.loads((ROOT / "results" / "heads.json").read_text())
    study_tasks = {task for task, _ in STUDY.values()}
    out = []
    for r in pt.rows():
        if r["paper"] is None:
            continue
        base = {"task": r["task"], "name": r["name"].replace("\\ ", " ").replace("\\", ""), "headline": r["headline"],
                "study": r["task"] in study_tasks, "zero": r["jev"], "paper": r["paper"], "llm": r["llm"], "cat": r["cat"],
                "native": None, "opt": None, "opt_sd": None, "opt_src": None, "opt_rep": None}
        slugs = [s for s, (t, _) in STUDY.items() if t == r["task"]]
        if r["task"] == "dress_new":
            # The study scores DREsS per rubric: one row per rubric. Zero-shot marker: the prompt-only levels of the
            # zero-shot evaluation; square: the study's baseline with Score cutoffs fit on training data.
            ref = pt.PAPER["dress_new"]
            per = ref["paper_per_rubric_content_org_lang"]
            llm_key = next(k for k, v in ref["paper"].items() if v == r["llm"])
            prompt_only = dress_prompt_only()
            rows = []
            for s in slugs:
                i = STUDY[s][1]
                paper, llm = per[ref["method"]][i], per[llm_key][i]
                zero = prompt_only[RUBRICS[i]]
                rows.append(base | {"name": f"DREsS {RUBRICS[i]}", "slug": s, "zero": zero, "paper": paper, "llm": llm,
                                    "cat": category(zero, paper, llm), "zero_fitted": gepa[s]["zero"]})
        else:
            rows = [base | {"slug": slugs[0] if slugs else None}]
        for row in rows:
            s = row["slug"]
            if s is not None:
                if s in chosen:
                    row["opt"], row["opt_sd"] = seed_mean(s, *chosen[s])
                    row["opt_src"], row["opt_rep"] = "v3", chosen[s][1]
                row["native"] = test_primary(f"v3-{s}-N", "top-quality")
                if row["opt"] is None and s in gepa:
                    row["opt"], row["opt_src"] = v1_best(s, gepa[s]["conds"]), "v1"
            elif row["task"] in heads:
                row["opt"], row["opt_src"] = heads[row["task"]]["test_head"]["value"], "head"
            elif row["task"] in LEVER_C:
                row["opt"], row["opt_src"] = lever_c(row["task"]), "lever C"
            out.append(row)
    return out


def ladder(rows: list[dict], path: Path, row_h: float) -> None:
    order = [c for c in fb.CAT_ORDER if any(r["cat"] == c for r in rows)]
    plan, headers, pos = [], [], 0.0
    for cat in order:
        members = sorted((r for r in rows if r["cat"] == cat), key=lambda r: -r["zero"] / r["paper"])
        headers.append((pos, cat, len(members)))
        pos += 1
        for r in members:
            plan.append((pos, r))
            pos += 1
        pos += 0.2
    fig, ax = plt.subplots(figsize=(fb.TEXTW, 0.58 + row_h * pos), layout="constrained")
    label_tf = blended_transform_factory(fig.transFigure, ax.transData)
    y0 = -0.65
    for (yh, cat, n), nxt in zip(headers, [h[0] for h in headers[1:]] + [pos]):
        ax.axhspan(yh - 0.5, nxt - 0.8, color=BAND[cat], zorder=0, lw=0)
        t = ax.text(0.004, yh, f"{TITLES[cat]} ({n})", transform=label_tf, ha="left", va="center", fontsize=6.5,
                    fontweight="bold", fontstyle="italic", color=HEAD[cat])
        t.set_in_layout(False)
    ax.axvline(1.0, color="black", lw=0.7, ls="--", zorder=1)

    def x_of(v: float, r: dict) -> tuple[float, str | None]:
        x = v / r["paper"]
        if x > XMAX:
            return XMAX, f"{x:.2f}"
        if x < XMIN:
            return XMIN, f"{x:.2f}"
        return x, None

    def put(v: float, r: dict, y: float, **style) -> float:
        x, clipped = x_of(v, r)
        if clipped:
            right = x == XMAX
            ax.plot(x, y, marker=">" if right else "<", ls="none", color=style.get("mec", style.get("color")), ms=4,
                    zorder=6, clip_on=False)
            ax.text(x - 0.014 if right else x + 0.014, y, clipped, ha="right" if right else "left", va="center", zorder=7,
                    fontsize=5.5, color=style.get("mec", style.get("color")),
                    bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"))
        else:
            ax.plot(x, y, ls="none", **style)
        return x

    ys, labels = [], []
    for y, r in plan:
        ys.append(y)
        labels.append(r["name"])
        ax.axhline(y, color="white", lw=0.4, zorder=0.5)
        if r["llm"] is not None:
            put(r["llm"], r, y, marker="D", mfc="white", mec=LLM, mew=0.8, ms=3.6, zorder=3)
        x0 = put(r["zero"], r, y, marker="o", mfc="white", mec=JEV, mew=0.9, ms=4.4, zorder=5)
        if r.get("zero_fitted") is not None:
            put(r["zero_fitted"], r, y, zorder=5, **FITTED_ZERO)
        xs = []
        if r["native"] is not None:
            xs.append(put(r["native"], r, y, marker="o", fillstyle="left", color=JEV, mfc=JEV, mfcalt="white", mec=JEV,
                          mew=0.9, ms=4.4, zorder=5))
        if r["opt"] is not None:
            fill = JEV_V1 if r["opt_src"] == "v1" else JEV
            if r["opt_sd"]:
                # +-1 sd over optimizer seeds: a light band under the arrow, so both stay readable.
                lo, hi = x_of(r["opt"] - r["opt_sd"], r)[0], x_of(r["opt"] + r["opt_sd"], r)[0]
                ax.plot([lo, hi], [y, y], color=JEV_V1, lw=3.2, alpha=0.8, solid_capstyle="round", zorder=3.9)
            xs.append(put(r["opt"], r, y, marker="o", mfc=fill, mec=JEV, mew=0.9, ms=4.4, zorder=6))
        if xs:
            best = max(xs)
            if abs(best - x0) > 0.012:
                ax.annotate("", xy=(best, y), xytext=(x0, y), zorder=4,
                            arrowprops=dict(arrowstyle="-|>", color=JEV, lw=0.8, shrinkA=2.6, shrinkB=2.8, mutation_scale=6))
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=6.3)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(pos - 0.6, y0)
    ax.set_xlim(XMIN, XMAX)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax.set_xlabel("score as a fraction of the paper's task-specific method (dashed line = 1)", fontsize=6.5)
    ax.tick_params(axis="x", labelsize=6, length=2)
    ax.grid(axis="x", color="#dee2e6", lw=0.4, zorder=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    srcs = {r["opt_src"] for _, r in plan}
    handles = [
        plt.Line2D([], [], marker="D", mfc="white", mec=LLM, mew=0.8, ms=3.6, ls="none", label="paper's prompted LLM"),
        plt.Line2D([], [], marker="o", mfc="white", mec=JEV, mew=0.9, ms=4.4, ls="none", label="Jev zero-shot"),
        *([plt.Line2D([], [], ls="none", label="DREsS, cutoffs fit", **FITTED_ZERO)] if any(r.get("zero_fitted") for _, r in plan) else []),
    ]
    if any(r["native"] is not None for _, r in plan):
        handles.append(plt.Line2D([], [], marker="o", fillstyle="left", color=JEV, mfcalt="white", mec=JEV, mew=0.9, ms=4.4,
                                  ls="none", label="Jev, rephrased question"))
    if srcs & {"v3", "head", "lever C"}:
        handles.append(plt.Line2D([], [], marker="o", mfc=JEV, mec=JEV, ms=4.4, ls="none",
                                  label="Jev + training data"))
    if any(r["opt_sd"] for _, r in plan):
        handles.append(plt.Line2D([], [], color=JEV_V1, lw=3.2, alpha=0.8, label="$\\pm$1 sd over seeds"))
    if "v1" in srcs:
        handles.append(plt.Line2D([], [], marker="o", mfc=JEV_V1, mec=JEV, mew=0.9, ms=4.4, ls="none",
                                  label="Jev optimized, GEPA search"))
    handles.append(plt.Line2D([], [], color="black", lw=0.7, ls="--", label="task-specific method"))
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False, fontsize=6.3,
               handletextpad=0.25, columnspacing=1.1)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01)
    fig.savefig(path)
    plt.close(fig)


def marker_macro(rows: list[dict]) -> str:
    """Caption sentence for what the legend cannot say: where the non-audit markers come from."""
    srcs = {r["opt_src"] for r in rows}
    parts = []
    if any(r["native"] is not None for r in rows):
        parts.append("half-filled: the rephrased question, chosen on training data")
    head_names = [r["name"] for r in rows if r["opt_src"] == "head"]
    nseeds = {len(json.loads((ROOT / "results" / "seeds.json").read_text())["tasks"][r["slug"]]["seeds"])
              for r in rows if r["opt_src"] == "v3"}
    how = [w for s, w in (("v3", "the eight optimized tasks: the optimized questions, mean over "
                                 f"{'/'.join(map(str, sorted(nseeds)))} optimizer seeds"),
                          ("head", ", ".join(head_names[:-1]) + " and " + head_names[-1]
                                   + ": calibration or a logistic regression on the unchanged zero-shot answers" if head_names else ""),
                          ("lever C", "TTP, Wiki-WD, MADE: candidate reranking and fitted thresholds")) if s in srcs]
    if how:
        parts.append(f"filled: with training data, pre-registered and tested once ({'; '.join(how)})")
    if "v1" in srcs:
        parts.append("light blue: where that test is pending, the GEPA search condition of Table~\\ref{tab:opt} better on "
                     "validation")
    text = ". ".join(p[0].upper() + p[1:] for p in parts)
    return f"\\newcommand{{\\heroMarkers}}{{{text + '.' if text else ''}}}\n"


def frozen_rows() -> list[dict]:
    """The frozen configuration's single runs on every eligible shortfall set (results/frozen.json)."""
    fz = json.loads((ROOT / "results" / "frozen.json").read_text())
    return [{"name": r["title"].replace("SMARTER ", ""), "slug": r["stem"], "zero": r["zero"], "opt": r["optimized"],
             "opt_sd": None, "paper": r["paper"], "llm": r["llm"]} for r in fz["new"]]


def page1(rows: list[dict], path: Path) -> None:
    """Column-width page-1 figure: the optimization-study tasks (3-seed mean) and the frozen configuration's
    eligible shortfall sets (one run), zero-shot -> optimized questions, the best trained baseline on the same
    labels (results/BUDGET_BASELINE.json, seed mean), Clef answering Jev's seed-0 optimized questions
    (results/CLEF_RQ3.json), plus a strip with the zero-shot outcome over all test sets (paper_tables.rows() categories)."""
    trained = {s: t["best_test_mean"] for s, t in json.loads((ROOT / "results" / "BUDGET_BASELINE.json").read_text())["sets"].items()}
    clef = {r["set"]: r["optimized"] for r in json.loads((ROOT / "results" / "CLEF_RQ3.json").read_text())["clef"]["sets"]}
    study = sorted((r for r in rows if r["opt_src"] == "v3"), key=lambda r: -r["opt"] / r["paper"])
    frozen = sorted(frozen_rows(), key=lambda r: -r["opt"] / r["paper"])
    seeds = json.loads((ROOT / "results" / "seeds.json").read_text())["tasks"]
    nseeds = "/".join(map(str, sorted({len(seeds[r["slug"]]["seeds"]) for r in study})))
    cats = [r["cat"] for r in pt.rows() if r["paper"] is not None]
    # One header row per group, then its sets.
    groups = [(f"chosen for large gaps (mean of {nseeds} seeds)", study), ("all eligible shortfall sets (one run)", frozen)]
    plan, heads, y = [], [], 0.0
    for title, members in groups:
        heads.append((y, title))
        y += 0.85
        for r in members:
            plan.append((y, r))
            y += 1
        y += 0.25
    fig, (ax, sx) = plt.subplots(2, 1, figsize=(fb.COLW, 2.7), layout="constrained",
                                 gridspec_kw={"height_ratios": [y, 1.15]})
    ax.axvline(1.0, color="black", lw=0.7, ls="--", zorder=1)
    label_tf = blended_transform_factory(fig.transFigure, ax.transData)
    for yh, title in heads:
        t = ax.text(0.004, yh, title, transform=label_tf, ha="left", va="center", fontsize=6.3, fontstyle="italic",
                    fontweight="bold", color="#495057")
        t.set_in_layout(False)
    for y_, r in plan:
        rel = {k: None if r.get(k) is None else r[k] / r["paper"] for k in ("llm", "zero", "opt", "zero_fitted")}
        ax.axhline(y_, color="#f1f3f5", lw=0.5, zorder=0.5)
        if r["opt_sd"]:
            sd = r["opt_sd"] / r["paper"]
            ax.plot([rel["opt"] - sd, rel["opt"] + sd], [y_, y_], color=JEV_V1, lw=3.4, alpha=0.8,
                    solid_capstyle="round", zorder=3)
        if rel["llm"] is not None:
            ax.plot(rel["llm"], y_, marker="D", mfc=LLM, mec="white", mew=0.4, ms=3.8, ls="none", zorder=5.5)
        ax.plot(rel["zero"], y_, marker="o", mfc="white", mec=JEV, mew=0.9, ms=4.6, ls="none", zorder=5)
        if rel["zero_fitted"] is not None:
            ax.plot(rel["zero_fitted"], y_, ls="none", zorder=5, **FITTED_ZERO)
        ax.annotate("", xy=(rel["opt"], y_), xytext=(rel["zero"], y_), zorder=4.5,
                    arrowprops=dict(arrowstyle="-|>", color=JEV, lw=0.8, shrinkA=2.8, shrinkB=2.8, mutation_scale=6))
        ax.plot(rel["opt"], y_, marker="o", mfc=JEV, mec=JEV, ms=4.6, ls="none", zorder=6)
        ax.plot(trained[r["slug"]] / r["paper"], y_, ls="none", zorder=6.5, **TRAINED)
        # Clef sits a third of a row above the line so the trained-baseline cross cannot hide it.
        ax.plot(clef[r["slug"]] / r["paper"], y_ - 0.32, ls="none", zorder=6.8, **CLEF)
    ax.set_yticks([p for p, _ in plan])
    ax.set_yticklabels([r["name"] for _, r in plan], fontsize=6.5)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylim(y - 0.6, -0.6)
    ax.set_xlim(min(0.3, min(r["zero"] / r["paper"] for _, r in plan) - 0.04),
                max(1.1, max(max(trained[r["slug"]], clef[r["slug"]]) / r["paper"] for _, r in plan) + 0.03))
    ax.set_xticks([x for x in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) if x >= ax.get_xlim()[0]])
    ax.set_xlabel("test score / paper's task-specific method", fontsize=6.5, labelpad=1)
    ax.tick_params(axis="x", labelsize=6.2, length=2, pad=1)
    ax.grid(axis="x", color="#dee2e6", lw=0.4, zorder=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    handles = [
        plt.Line2D([], [], marker="D", mfc=LLM, mec=LLM, ms=3.6, ls="none", label="paper's prompted LLM"),
        plt.Line2D([], [], marker="o", mfc="white", mec=JEV, mew=0.9, ms=4.6, ls="none", label="Jev, asked for label"),
        plt.Line2D([], [], ls="none", label="DREsS, cutoffs fit", **FITTED_ZERO),
        plt.Line2D([], [], marker="o", mfc=JEV, mec=JEV, ms=4.6, ls="none", label="optimized questions"),
        plt.Line2D([], [], color=JEV_V1, lw=3.4, alpha=0.8, label=f"$\\pm$1 sd, {nseeds} seeds"),
        plt.Line2D([], [], ls="none", label="trained on the same labels", **TRAINED),
        plt.Line2D([], [], ls="none", label="Clef, same questions", **CLEF),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False, fontsize=6.3,
               handletextpad=0.3, columnspacing=0.9, labelspacing=0.25)
    # Strip: zero-shot outcome over every comparable test set.
    labels = {"replaces": "$\\geq$ method", "between": "between", "below": "below both"}
    left = 0
    for c in fb.CAT_ORDER:
        n = cats.count(c)
        sx.barh(0, n, left=left, color=fb.COLORS[c], height=1.0, edgecolor="white", lw=0.6)
        sx.text(left + n / 2, 0, f"{labels[c]}: {n}", ha="center", va="center_baseline", fontsize=6.0,
                color="black" if c == "between" else "white", fontweight="bold")
        left += n
    sx.set_xlim(0, left)
    sx.set_ylim(-0.5, 0.5)
    sx.axis("off")
    sx.set_title(f"Jev asked for the label, all {left} test sets", fontsize=6.3, pad=1.5)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.03, hspace=0.06)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = ladder_rows()
    # Body: one headline set per paper (Table 1) plus the optimization study's other sets.
    ladder([r for r in rows if r["headline"] or r["study"]], OUT / "hero_ladder.pdf", row_h=0.092)
    ladder(rows, OUT / "hero_ladder_all.pdf", row_h=0.093)
    page1(rows, OUT / "hero_page1.pdf")
    (OUT / "hero_numbers.tex").write_text(marker_macro(rows))
    if "-v" in sys.argv:
        for r in rows:
            f = {k: (None if r[k] is None else round(r[k] / r["paper"], 3)) for k in ("llm", "zero", "native", "opt")}
            print(f"{r['name']:28s} {r['cat']:8s} H={int(r['headline'])} {f} {r['opt_src']}")
    print("wrote hero_ladder.pdf, hero_ladder_all.pdf, hero_numbers.tex;", sum(r["opt"] is not None for r in rows),
          "rows with an optimized marker")


if __name__ == "__main__":
    main()
