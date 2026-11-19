# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pyyaml"]
# ///
"""Generate LaTeX tables and numeric macros for the paper (paper/generated/).

Sources (never hand-copied): runs/<task>/scores.json (Jev), results/paper_numbers.json (paper
numbers with citations), results/gepa.json (optimization study). Run from the repo root:

    uv run scripts/paper_tables.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_appendix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "generated"
PAPER = json.loads((ROOT / "results" / "paper_numbers.json").read_text())

# (task, paper bibkey, display name, output geometry, best zero-shot/prompted LLM baseline reported
#  in the same paper or None, headline?)  The LLM baseline is the paper's own number (0-1 scale).
DATASETS: list[tuple[str, str, str, str, float | None, bool]] = [
    ("pinose/true_false", "zhang-etal-2024-transferable", "PiNose True-False", "binary", None, True),
    ("pinose/nq", "zhang-etal-2024-transferable", "PiNose NQ", "binary", None, False),
    ("pinose/triviaqa", "zhang-etal-2024-transferable", "PiNose TriviaQA", "binary", None, False),
    ("pinose/webq", "zhang-etal-2024-transferable", "PiNose WebQ", "binary", None, False),
    ("prp_graph/trec_covid", "luo-etal-2024-prp", "PRP-Graph TREC-COVID", "ranking", None, True),
    ("xstest", "rottger-etal-2024-xstest", "XSTest (judge)", "3-way", 0.804, True),
    ("factpico", "joseph-etal-2024-factpico", "FactPICO", "ordinal", 0.475, True),
    ("fenice", "scire-etal-2024-fenice", "FENICE", "binary", None, True),
    ("biorc800", "lan-etal-2024-multi", "BIORC800", "multi-label", 0.476, True),
    ("sciprompt/arxiv", "you-etal-2024-sciprompt", "SciPrompt arXiv", "53-way", 0.549, True),
    ("sciprompt/s2orc", "you-etal-2024-sciprompt", "SciPrompt S2ORC", "19-way", 0.496, False),
    ("sciprompt/sdpra", "you-etal-2024-sciprompt", "SciPrompt SDPRA", "7-way", 0.812, False),
    ("sciprompt/emerging_nlp", "you-etal-2024-sciprompt", "SciPrompt Emerging-NLP", "21-way", 0.086, False),
    ("scalar_constructs/immigration_fear/pairwise", "licht-etal-2025-measuring", "Scalar Constructs: imm.\\ fear (pair)", "pairwise", 0.74, True),
    ("scalar_constructs/ad_negativity/pairwise", "licht-etal-2025-measuring", "Scalar Constructs: ad neg.\\ (pair)", "pairwise", 0.86, False),
    ("scalar_constructs/grandstanding/pairwise", "licht-etal-2025-measuring", "Scalar Constructs: grandst.\\ (pair)", "pairwise", 0.54, False),
    ("scalar_constructs/immigration_fear/pointwise", "licht-etal-2025-measuring", "Scalar Constructs: imm.\\ fear (point)", "ordinal", 0.81, False),
    ("scalar_constructs/ad_negativity/pointwise", "licht-etal-2025-measuring", "Scalar Constructs: ad neg.\\ (point)", "ordinal", 0.91, False),
    ("scalar_constructs/grandstanding/pointwise", "licht-etal-2025-measuring", "Scalar Constructs: grandst.\\ (point)", "ordinal", 0.60, False),
    ("lifetox", "kim-etal-2024-lifetox", "LifeTox", "binary", 0.744, True),
    ("edit_intent", "ruan-etal-2024-large", "Re3-Sci edit intent", "5-way", 0.37, True),
    ("ezstance/subtaskA", "zhao-caragea-2024-ez", "EZ-STANCE A", "3-way", 0.499, True),
    ("ezstance/subtaskB", "zhao-caragea-2024-ez", "EZ-STANCE B", "3-way", 0.495, False),
    ("contempo/matres", "niu-etal-2024-contempo", "ConTempo MATRES", "4-way", 0.567, True),
    ("contempo/tbdense", "niu-etal-2024-contempo", "ConTempo TB-Dense", "6-way", 0.6758, False),
    ("mismatched", "shaik-etal-2025-mismatched", "MISMATCHED", "4-way", 0.556, True),
    ("inli", "havaldar-etal-2025-entailed", "INLI", "4-way", 0.770, True),
    ("medreadme", "jiang-xu-2024-medreadme", "MedReadMe", "ordinal", 0.721, True),
    ("nav_unknown/clinc150", "sali-toraman-2025-navigating", "Navigating the Unknown CLINC150 (ID+OOD)", "76-way", 0.837, True),
    ("nav_unknown/banking77", "sali-toraman-2025-navigating", "Navigating the Unknown BANKING77 (ID+OOD)", "39-way", 0.694, False),
    ("made", "agarwal-etal-2026-made", "MADE", "1,154 multi-label", 0.43, True),
    ("smarter/hatexplain", "nghiem-etal-2026-smarter", "SMARTER HateXplain", "3-way", 0.56, True),
    ("smarter/latent_hate", "nghiem-etal-2026-smarter", "SMARTER Latent Hate", "3-way", 0.60, False),
    ("smarter/implicit_hate", "nghiem-etal-2026-smarter", "SMARTER Implicit Hate", "6-way", 0.58, False),
    ("dress_new", "yoo-etal-2025-dress", "DREsS", "ordinal", 0.304, True),
    ("ttp/tram", "nguyen-etal-2024-noise", "TTP mapping TRAM", "737 ranking", None, True),
    ("ttp/procedures_plus", "nguyen-etal-2024-noise", "TTP mapping Procedures+", "737 ranking", None, False),
    ("ttp/expert", "nguyen-etal-2024-noise", "TTP mapping Expert", "737 ranking", None, False),
    ("legalseg", "nigam-etal-2025-legalseg", "LegalSeg", "7-way seq.", 0.08, True),
    ("relemb/wikiwd", "mtumbuka-schockaert-2024-entity", "Entity or Relation Embeddings Wiki-WD", "352-way", None, True),
    ("relemb/nytfb", "mtumbuka-schockaert-2024-entity", "Entity or Relation Embeddings NYT-FB", "56-way", None, False),
    ("cicc/mtod", "hengst-etal-2024-conformal", "CICC MTOD", "conformal set", None, True),
    ("cicc/hwu64", "hengst-etal-2024-conformal", "CICC HWU64", "conformal set", None, False),
    ("cicc/c150", "hengst-etal-2024-conformal", "CICC CLINC150", "conformal set", None, False),
    ("cicc/acid", "hengst-etal-2024-conformal", "CICC ACID", "conformal set", None, False),
    ("cicc/atis", "hengst-etal-2024-conformal", "CICC ATIS", "conformal set", None, False),
]
# Where the paper's "method" row in paper_numbers.json is not its strongest task-specific
# predictor, compare against that predictor instead: (value, label).
TASK_SPECIFIC = {
    "sciprompt/arxiv": (0.7015, "SciPromptSoft 50-shot"),
    "sciprompt/s2orc": (0.5666, "SciPrompt 50-shot"),
    "sciprompt/sdpra": (0.9071, "full fine-tuning"),
    "sciprompt/emerging_nlp": (0.8933, "SciPrompt 5-shot"),
    # Pairwise rows: the paper's task-specific model is the DeBERTa reward model (5-fold mean).
    "scalar_constructs/immigration_fear/pairwise": (0.91, "DeBERTa reward model"),
    "scalar_constructs/ad_negativity/pairwise": (0.91, "DeBERTa reward model"),
    "scalar_constructs/grandstanding/pairwise": (0.78, "DeBERTa reward model"),
}
# The paper's one label per source paper (Table 1, Table 7, the text). A test set's display name is
# this label followed by the set's own name, or the label alone for a paper with one set.
PAPER_LABEL = {
    "zhang-etal-2024-transferable": "PiNose", "scire-etal-2024-fenice": "FENICE",
    "joseph-etal-2024-factpico": "FactPICO", "rottger-etal-2024-xstest": "XSTest", "kim-etal-2024-lifetox": "LifeTox",
    "luo-etal-2024-prp": "PRP-Graph", "licht-etal-2025-measuring": "Scalar Constructs", "lan-etal-2024-multi": "BIORC800",
    "you-etal-2024-sciprompt": "SciPrompt", "ruan-etal-2024-large": "Re3-Sci edit intent",
    "zhao-caragea-2024-ez": "EZ-STANCE", "niu-etal-2024-contempo": "ConTempo", "shaik-etal-2025-mismatched": "MISMATCHED",
    "havaldar-etal-2025-entailed": "INLI", "jiang-xu-2024-medreadme": "MedReadMe",
    "sali-toraman-2025-navigating": "Navigating the Unknown", "agarwal-etal-2026-made": "MADE",
    "nghiem-etal-2026-smarter": "SMARTER", "yoo-etal-2025-dress": "DREsS", "nguyen-etal-2024-noise": "TTP mapping",
    "nigam-etal-2025-legalseg": "LegalSeg", "mtumbuka-schockaert-2024-entity": "Entity or Relation Embeddings",
    "hengst-etal-2024-conformal": "CICC", "bailleux-etal-2026-explanation": "Explanation ranking",
}
assert all(name.startswith(PAPER_LABEL[key]) for _, key, name, *_ in DATASETS)
SET_LABEL = {task: name for task, _, name, *_ in DATASETS}
# Optimization-study (RQ3) and frozen-run configurations: slug -> label. A configuration on one test set
# carries that set's label; the three DREsS rubrics are scored separately.
CONFIG_SET = {"contempo-tbdense": "contempo/tbdense", "contempo-matres": "contempo/matres",
              "ezstance-subtaskA": "ezstance/subtaskA", "smarter-implicit-hate": "smarter/implicit_hate",
              "smarter-hatexplain": "smarter/hatexplain", "smarter-latent-hate": "smarter/latent_hate",
              "edit-intent": "edit_intent", "inli": "inli", "lifetox": "lifetox", "sciprompt-s2orc": "sciprompt/s2orc"}
CONFIG_LABEL = {slug: SET_LABEL[task] for slug, task in CONFIG_SET.items()} | {
    f"dress-new-{rubric}": f"DREsS {rubric}" for rubric in ("content", "organization", "language")}
# Paper labels too long to share a line with the set name in Tables 2 and 6.
LONG_PAPER = ("Navigating the Unknown", "Entity or Relation Embeddings")


def two_lines(first: str, second: str) -> str:
    """A table cell set on two lines, aligned with the row at its first line; the second line is indented."""
    return f"\\begin{{tabular}}[t]{{@{{}}l@{{}}}}{first}\\\\\\quad {second}\\end{{tabular}}"


def name_cell(name: str, cite: str) -> str:
    """A test-set label and its citation, broken after a long paper label."""
    for p in LONG_PAPER:
        if name.startswith(p + " "):
            return two_lines(p, name[len(p) + 1:] + cite)
    return name + cite
NOT_COMPARABLE = {
    "expl_rank": "graded candidates reconstructed, and a tier-order oracle scores 0.994",
    "cicc/b77": "paper's B77 number not reproducible from released outputs",
}
METRIC_TEX = {"macro_f1": "macro-F1", "micro_f1": "micro-F1", "auc": "AUC", "accuracy": "acc.", "ndcg@10": "nDCG@10",
              "kendall_tau_b": "Kendall $\\tau_b$", "balanced_accuracy": "bal.\\ acc.", "spearman_rho": "Spearman $\\rho$",
              "pearson_mean_groups": "Pearson $r$", "all_macro_f1": "macro-F1", "qwk_total": "QWK", "p@1": "P@1",
              "f1": "micro-F1", "single": "Single", "p_at_10": "P@10"}


def scores(task: str) -> dict:
    return json.loads((ROOT / "runs" / task / "scores.json").read_text())


def jev_value(task: str) -> float:
    ref = PAPER[task]
    s = scores(task)
    if "jev_paper_protocol" in ref:
        node = s
        for part in ref["jev_paper_protocol"]["path"].split("."):
            node = node[part]
        return float(node)
    return float(s[ref["metric"]])


def rows() -> list[dict]:
    out = []
    for task, key, name, geom, llm, headline in DATASETS:
        ref = PAPER[task]
        s = scores(task)
        jev = jev_value(task)
        paper = TASK_SPECIFIC[task][0] if task in TASK_SPECIFIC else ref["paper"].get(ref["method"])
        rel = jev / paper if paper else None
        if paper is None:
            cat = "n/a"
        elif jev >= paper - 0.005:
            cat = "replaces"
        elif llm is not None and jev >= llm - 0.005:
            cat = "between"
        elif llm is None and rel is not None and rel >= 0.8:
            cat = "between"
        else:
            cat = "below"
        out.append({"task": task, "key": key, "name": name, "geom": geom, "metric": METRIC_TEX.get(ref["metric"], ref["metric"]),
                    "n": s.get("n"), "jev": jev, "paper": paper, "llm": llm, "rel": rel, "cat": cat, "headline": headline,
                    "protocol": "jev_paper_protocol" in ref})
    return out


def pct(x: float | None) -> str:
    return "--" if x is None else f"{100 * x:.1f}"


# Outcome groups, in table order. "n/a" (no task-specific number) is emitted only if present.
GROUPS = [("replaces", "Jev matches or beats the task-specific method"),
          ("between", "Jev between prompted LLM and task-specific"),
          ("below", "Jev below both"),
          ("n/a", "No task-specific number")]


def delta(r: dict) -> float | None:
    """Jev minus the task-specific method, in points (x100), rounded as displayed."""
    return None if r["paper"] is None else round(100 * r["jev"] - 100 * r["paper"], 1) + 0.0


def grouped(rs: list[dict]) -> list[tuple[str, str, list[dict]]]:
    """Rows split into outcome groups, each sorted by Delta (largest first)."""
    out = []
    for cat, title in GROUPS:
        members = sorted((r for r in rs if r["cat"] == cat), key=lambda r: -(delta(r) or 0.0))
        if members:
            out.append((cat, title, members))
    return out


def table(rs: list[dict], caption: str, label: str, star: bool, place: str = "t") -> str:
    env = "table*" if star else "table"
    lines = [f"\\begin{{{env}}}[{place}]", "\\centering\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
             # scale down only if wider than the text block, never up
             "\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%", "\\begin{tabular}{lccccccc}", "\\toprule",
             "Dataset & Output & $N$ & Metric & Jev & Task-specific & LLM & $\\Delta$ \\\\"]
    for _, title, members in grouped(rs):
        lines += ["\\midrule", f"\\multicolumn{{8}}{{l}}{{\\textbf{{\\textit{{{title}}}}}}} \\\\"]
        for r in members:
            shown = {k: None if r[k] is None else round(100 * r[k], 1) for k in ("jev", "paper", "llm")}
            best = max(v for v in shown.values() if v is not None)
            cells = {}
            for k, v in shown.items():
                if v is None:
                    cells[k] = "\\textcolor{gray}{--}"
                    continue
                s = f"{v:.1f}"
                cells[k] = f"\\textbf{{{s}}}" if v == best else s
            if r["protocol"]:
                cells["jev"] += "$^\\dagger$"
            if shown["jev"] == best:
                cells["jev"] = "\\cellcolor{green!15}" + cells["jev"]
            d = delta(r)
            dcell = f"\\textcolor{{ForestGreen}}{{{'+' if d > 0 else ''}{d:.1f}}}" if d >= 0 else f"\\textcolor{{BrickRed}}{{$-${-d:.1f}}}"
            n = f"{r['n']:,}" if isinstance(r["n"], int) else "--"
            name = name_cell(r["name"], "~\\citep{" + r["key"] + "}")
            lines.append(f"{name} & {r['geom']} & {n} & {r['metric']} & {cells['jev']} & {cells['paper']} & "
                         f"{cells['llm']} & {dcell} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", f"\\end{{{env}}}", ""]
    return "\n".join(lines)


def macros(rs: list[dict], gepa: dict | None) -> str:
    comparable = [r for r in rs if r["paper"] is not None]
    counts = {c: sum(r["cat"] == c for r in comparable) for c in ("replaces", "between", "below")}
    manifests = [m for _, m in paper_appendix.run_manifests()]
    lines = [
        f"\\newcommand{{\\nDatasets}}{{{len(comparable)}}}",
        f"\\newcommand{{\\nReplaces}}{{{counts['replaces']}}}",
        f"\\newcommand{{\\nBetween}}{{{counts['between']}}}",
        f"\\newcommand{{\\nFails}}{{{counts['below']}}}",
        f"\\newcommand{{\\nPapersScored}}{{{len({r['key'] for r in comparable})}}}",
        f"\\newcommand{{\\nRequests}}{{{sum(m['n_requests'] for m in manifests):,}}}",
        f"\\newcommand{{\\nTokensB}}{{{sum(m['input_tokens'] for m in manifests) / 1e9:.2f}}}",
        f"\\newcommand{{\\totalCost}}{{\\${sum(m['cost_usd'] for m in manifests):,.0f}}}",
    ]
    if gepa:
        done = [r for r in gepa["rows"] if "program" in r["conds"] and "text" in r["conds"]]
        lines.append(f"\\newcommand{{\\nGepaTasks}}{{{len(done)}}}")
    lines += paper_appendix.macros_tex()
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rs = rows()
    head = [r for r in rs if r["headline"]]
    (OUT / "main_table.tex").write_text(table(head, "Zero-shot Jev, prompted with the paper's label definitions, compared "
        "with each paper's method (Task-specific) and its prompted-LLM baseline (LLM) on one test set per paper. Rows are "
        "grouped by outcome as defined in \\S\\ref{sec:protocol}, and Table~\\ref{tab:full} lists all test sets. "
        "$^\\dagger$ Scored under the paper's protocol.", "tab:main", star=True))
    (OUT / "full_table.tex").write_text(table(rs, "All \\nDatasets{} test sets in the comparison, formatted as in "
        "Table~\\ref{tab:main}.", "tab:full", star=True, place="!tp"))
    gp = ROOT / "results" / "gepa.json"
    gepa = json.loads(gp.read_text()) if gp.exists() else None
    (OUT / "numbers.tex").write_text(macros(rs, gepa))
    (OUT / "not_comparable.tex").write_text("\\begin{itemize}\n" + "".join(
        f"\\item \\texttt{{{t.replace('_', chr(92) + '_')}}}: {why}.\n" for t, why in NOT_COMPARABLE.items()) + "\\end{itemize}\n")
    paper_appendix.write_all([r for _, _, members in grouped(rs) for r in members], NOT_COMPARABLE)
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
