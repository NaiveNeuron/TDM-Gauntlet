# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pyyaml"]
# ///
"""Decision guide of the paper's §6, scored against the observed outcomes (rule fixed before it is scored).

Assigns each of the 46 test sets of Table full to a row from properties recorded before any model output:
  C  hundreds of opaque codes: a fixed inventory (the same for every item) of >= 100 labels or codes
  B  labels defined by annotation conventions: the frozen prompt file's `_source` cites annotation guidelines
  A  single judgment, labels mean what they say: every other set
and scores the rows against the prompt-only RQ1 group (A right iff Jev matches or beats the paper's method, B iff
it does not, C iff Jev is below both references). For the RQ3 sets in row B it also reports how many close at least
half of the gap with the optimized questions (results/BUDGET_BASELINE.json, Jev mean over its tested seeds).

Writes results/DECISION_GUIDE.{md,json}, paper/generated/decision_guide.tex (\\dg* macros) and
paper/generated/decision_guide_table.tex (the §6 tabular). Run from the repository root:

    uv run scripts/decision_guide.py [--out DIR]   # DIR replaces paper/generated
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import paper_figs_body as fb  # noqa: E402
import paper_tables as pt  # noqa: E402

OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else ROOT / "paper" / "generated"
GUIDELINE = re.compile(r"(?i)annotation (guideline|scheme|framework|roles)|guidelines|annotators|rubric|codebook")
LARGE = 100
SUPPLIED_CANDIDATES = {"ranking"}  # Table-full geometry of tasks whose candidates come with each item (PRP-Graph)
# Table-full set -> BUDGET_BASELINE slugs (DREsS: its three rubrics, averaged)
RQ3 = {"contempo/tbdense": ["contempo-tbdense"], "ezstance/subtaskA": ["ezstance-subtaskA"],
       "smarter/implicit_hate": ["smarter-implicit-hate"], "edit_intent": ["edit-intent"], "inli": ["inli"],
       "dress_new": ["dress-new-content", "dress-new-organization", "dress-new-language"],
       "contempo/matres": ["contempo-matres"], "lifetox": ["lifetox"], "sciprompt/s2orc": ["sciprompt-s2orc"],
       "smarter/hatexplain": ["smarter-hatexplain"], "smarter/latent_hate": ["smarter-latent-hate"]}
ROWS = {"A": ("single judgment, labels mean what they say", "prompt"),
        "B": ("labels defined by annotation conventions", "about 500 labels, model not fine-tuned"),
        "C": ("hundreds of opaque codes", "fine-tune on full data")}
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


def prompt_file(task: str) -> str:
    head, _, tail = task.partition("/")
    if head in ("ezstance", "edit_intent", "nav_unknown", "made", "dress_new", "xstest", "lifetox", "inli", "fenice",
                "factpico", "medreadme", "mismatched", "legalseg", "biorc800", "ttp"):
        return head
    if head == "pinose":
        return "pinose_true_false" if tail == "true_false" else "pinose_qa"
    return task.replace("/", "_")


def assign(task: str, geom: str) -> tuple[str, int, bool]:
    k = fb.label_space(task)
    fixed = geom not in SUPPLIED_CANDIDATES
    src = json.loads((ROOT / "prompts" / f"{prompt_file(task)}.json").read_text())["_source"]
    guide = bool(GUIDELINE.search(src if isinstance(src, str) else json.dumps(src)))
    row = "C" if fixed and k >= LARGE else "B" if guide else "A"
    return row, k, guide


def right(row: str, cat: str) -> bool:
    return {"A": cat == "replaces", "B": cat != "replaces", "C": cat == "below"}[row]


def main() -> None:
    bb = json.loads((ROOT / "results" / "BUDGET_BASELINE.json").read_text())["sets"]
    sets = []
    for r in pt.rows():
        if r["paper"] is None:
            continue
        row, k, guide = assign(r["task"], r["geom"])
        s = {"task": r["task"], "name": r["name"], "row": row, "inventory": k, "guideline_source": guide,
             "observed": r["cat"], "right": right(row, r["cat"])}
        if r["task"] in RQ3:
            s["rq3_gap"] = statistics.mean(bb[x]["jev"]["gap_share"] for x in RQ3[r["task"]])
        sets.append(s)
    assert len(sets) == 46, len(sets)
    summary = {}
    for row in ROWS:
        rs = [s for s in sets if s["row"] == row]
        rq3 = [s for s in rs if "rq3_gap" in s]
        summary[row] = {"n": len(rs), "right": sum(s["right"] for s in rs),
                        "observed": {c: sum(s["observed"] == c for s in rs) for c in ("replaces", "between", "below")},
                        "wrong_sets": [s["name"] for s in rs if not s["right"]],
                        "n_rq3": len(rq3), "n_rq3_half": sum(s["rq3_gap"] >= 0.5 for s in rq3)}
    total = {"n": len(sets), "right": sum(s["right"] for s in sets)}
    rep = {"rule": "Decision guide: rule fixed before it is scored (scripts/decision_guide.py docstring)", "regex": GUIDELINE.pattern,
           "large": LARGE, "sets": sets, "rows": summary, "total": total}
    (ROOT / "results" / "DECISION_GUIDE.json").write_text(json.dumps(rep, indent=1) + "\n")

    L = ["# Decision guide scored against the observed outcomes", "",
         "Rule fixed before scoring; generated by `scripts/decision_guide.py`. Observed = the prompt-only "
         "RQ1 group of Table full. Row A is right iff Jev matches or beats the paper's method, B iff it does not, C iff "
         "Jev is below both references.", "",
         "| Row | Rule | Advice | Sets | Right | Observed (matches / between / below) | RQ3 sets closing ≥ half the gap |",
         "|---|---|---|---:|---:|---|---|"]
    for row, (rule, advice) in ROWS.items():
        s = summary[row]
        o = s["observed"]
        L.append(f"| {row} | {rule} | {advice} | {s['n']} | {s['right']} | {o['replaces']} / {o['between']} / {o['below']} | "
                 f"{s['n_rq3_half']} of {s['n_rq3']} |")
    L += ["", f"All rows: {total['right']} of {total['n']} right.", "",
          "| Set | Row | Inventory | Guideline cited | Observed | Right | RQ3 gap closed |", "|---|---|---:|---|---|---|---:|"]
    for s in sets:
        g = f"{100 * s['rq3_gap']:.0f}%" if "rq3_gap" in s else "–"
        L.append(f"| {s['name']} | {s['row']} | {s['inventory']} | {'yes' if s['guideline_source'] else 'no'} | "
                 f"{s['observed']} | {'yes' if s['right'] else 'no'} | {g} |")
    (ROOT / "results" / "DECISION_GUIDE.md").write_text("\n".join(L) + "\n")

    m = {"dgNSets": str(total["n"]), "dgNRight": str(total["right"]), "dgLarge": str(LARGE)}
    for row, tag in (("A", "Prompt"), ("B", "Labels"), ("C", "Codes")):
        s = summary[row]
        m[f"dgN{tag}"] = str(s["n"])
        m[f"dgRight{tag}"] = str(s["right"])
        m[f"dgNRqThree{tag}"] = str(s["n_rq3"])
        m[f"dgHalf{tag}"] = str(s["n_rq3_half"])
        m[f"dgWrong{tag}"] = ", ".join(s["wrong_sets"]) if s["wrong_sets"] else "none"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "decision_guide.tex").write_text("% Generated by scripts/decision_guide.py. Do not edit.\n"
                                            + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()))
    T = ["% Generated by scripts/decision_guide.py. Do not edit.",
         "\\begin{tabular}{@{}p{0.42\\columnwidth}p{0.26\\columnwidth}rrr@{}}",
         "\\toprule", "Property known before any run & Advice & Sets & Match & Right \\\\", "\\midrule"]
    props = {"A": "Neither of the two below", "B": "Label definitions cite annotation guidelines",
             "C": f"Fixed inventory of $\\geq${LARGE} codes"}
    advice = {"A": "Prompt", "B": "500 labels, frozen model", "C": "Fine-tune on full data"}
    for row in ROWS:
        s = summary[row]
        T.append(f"{props[row]} & {advice[row]} & {s['n']} & {s['observed']['replaces']} & {s['right']} \\\\")
    T += ["\\midrule", f"All & & {total['n']} & {sum(v['observed']['replaces'] for v in summary.values())} & {total['right']} \\\\",
          "\\bottomrule", "\\end{tabular}"]
    (OUT / "decision_guide_table.tex").write_text("\n".join(T) + "\n")
    print(json.dumps({"rows": {k: {x: v[x] for x in ("n", "right", "observed", "n_rq3", "n_rq3_half")} for k, v in summary.items()},
                      "total": total}, indent=1))


if __name__ == "__main__":
    main()
