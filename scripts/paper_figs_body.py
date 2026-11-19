# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pyyaml"]
# ///
"""Body-only figures, tables and macros for the paper (paper/generated/body_*).

Complements scripts/paper_tables.py (whose rows() defines the audit rows and outcome categories);
everything here is derived from saved results, never copied by hand:
runs/<task>/{scores,manifest,predictions}.json(l), prompts/*.json, results/gepa.json and
runs/jev-opt/study-*/{seed,top-quality}.yaml, report.md, usage.jsonl, jev_responses*.jsonl.gz. The
re-run consistency check also reads jev-opt/experiments/<exp>/data/test.jsonl (gitignored; written by
that experiment's prepare.py). Run from the repo root:

    uv run scripts/paper_figs_body.py
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import paper_tables as pt  # noqa: E402

OUT = ROOT / "paper" / "generated"
STUDY = ROOT / "runs" / "jev-opt"
COLW = 3.03  # ACL column width, inches
TEXTW = 6.3  # ACL text width, inches
# Outcome categories in Paul Tol's high-contrast scheme: distinct under deuteranopia, protanopia and tritanopia
# and ordered in lightness (dark blue, yellow, rose), so they also survive grayscale.
COLORS = {"replaces": "#004488", "between": "#DDAA33", "below": "#BB5566"}
CAT_ORDER = ("replaces", "between", "below")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "pdf.fonttype": 42, "axes.linewidth": 0.6,
})

# ---------------------------------------------------------------------------------------------
# Per-paper metadata for the body (domain, output, how the decision is asked, test provenance).
# Status vocabulary: "released" = paper's released test set used as is; "EN subset" = released
# test minus rows dropped by the English-only screen; "partial" = release lacks part of the
# paper's test set; "input rebuilt" = released test labels, model inputs built by us; "rebuilt" =
# test set or split reconstructed from the paper/authors' code.
# Source: data/manifests/<task>.json.
# ---------------------------------------------------------------------------------------------
PAPERS = [
    # (bibkey, domain, output, Jev interface, status); the paper's label is paper_tables.PAPER_LABEL
    ("zhang-etal-2024-transferable", "factuality", "binary", "Noul", "released"),
    ("scire-etal-2024-fenice", "factuality", "binary", "Noul per claim", "partial"),
    ("joseph-etal-2024-factpico", "medicine", "ordinal 0--4", "Score each", "released"),
    ("rottger-etal-2024-xstest", "safety", "3-way", "Choice", "released"),
    ("kim-etal-2024-lifetox", "safety", "binary", "Choice", "released"),
    ("luo-etal-2024-prp", "biomed.\\ IR", "rank 100", "Choice, tournament", "EN subset"),
    ("licht-etal-2025-measuring", "politics", "pairwise; 1--9", "Choice; Score", "released"),
    ("lan-etal-2024-multi", "biomed.", "6-label seq.", "Noul, sent.\\,$\\times$\\,label", "released"),
    ("you-etal-2024-sciprompt", "science", "7--53-way", "Choice", "EN subset"),
    ("ruan-etal-2024-large", "science", "5-way", "Choice, routed", "released"),
    ("zhao-caragea-2024-ez", "social media", "3-way", "Choice, routed", "EN subset"),
    ("niu-etal-2024-contempo", "news", "4-/6-way", "Choice", "input rebuilt"),
    ("shaik-etal-2025-mismatched", "science", "4-way", "Choice", "released"),
    ("havaldar-etal-2025-entailed", "general", "4-way", "Choice", "released"),
    ("jiang-xu-2024-medreadme", "medicine", "ordinal 1--6", "Score", "released"),
    ("sali-toraman-2025-navigating", "dialogue", "39--76-way", "Choice", "rebuilt"),
    ("agarwal-etal-2026-made", "medicine", "1,154-label set", "Noul, top-down", "released"),
    ("nghiem-etal-2026-smarter", "social media", "3-/6-way", "Choice", "rebuilt"),
    ("yoo-etal-2025-dress", "education", "3 ordinal", "Score per rubric", "rebuilt"),
    ("nguyen-etal-2024-noise", "security", "rank 737", "Noul each", "released"),
    ("nigam-etal-2025-legalseg", "law", "7-way seq.", "Choice per sent.", "EN subset"),
    ("mtumbuka-schockaert-2024-entity", "general", "56-/352-way", "Choice, split", "EN subset"),
    ("hengst-etal-2024-conformal", "dialogue", "conformal set", "Choice", "released"),
    ("bailleux-etal-2026-explanation", "NLI expl.", "rank 5", "Score each", "rebuilt"),
]

# Output-geometry groups for the 46 comparable test sets (every row of paper_tables.rows()).
GEOMETRY = [
    ("Binary judgment", ["pinose/", "fenice", "lifetox"]),
    ("3--7-way choice", ["xstest", "sciprompt/sdpra", "edit_intent", "ezstance/", "contempo/", "mismatched", "inli", "smarter/"]),
    ("19--76-way choice", ["sciprompt/arxiv", "sciprompt/s2orc", "sciprompt/emerging_nlp", "nav_unknown/", "relemb/nytfb"]),
    ("Ordinal score", ["factpico", "scalar_constructs/immigration_fear/pointwise", "scalar_constructs/ad_negativity/pointwise",
                       "scalar_constructs/grandstanding/pointwise", "medreadme", "dress_new"]),
    ("Pairwise / ranking", ["prp_graph/", "scalar_constructs/immigration_fear/pairwise", "scalar_constructs/ad_negativity/pairwise",
                            "scalar_constructs/grandstanding/pairwise"]),
    ("Multi-label sequence", ["biorc800", "legalseg"]),
    ("$>$300 outputs", ["made", "ttp/", "relemb/wikiwd"]),
    ("Conformal set", ["cicc/"]),
]
DOMAIN_OF = {key: dom for key, dom, *_ in PAPERS}
DOMAIN_GROUPS = [
    ("Factuality, safety", {"factuality", "safety"}),
    ("Biomedicine", {"medicine", "biomed.", "biomed.\\ IR"}),
    ("Science", {"science"}),
    ("Social media, politics", {"social media", "politics"}),
    ("General / news", {"general", "news"}),
    ("Dialogue intents", {"dialogue"}),
    ("Law, security, educ.", {"law", "security", "education"}),
]


def group_of(task: str, groups: list[tuple[str, list[str]]]) -> str:
    for name, prefixes in groups:
        if any(task == p or (p.endswith("/") and task.startswith(p)) for p in prefixes):
            return name
    raise KeyError(task)


def prompt(name: str) -> dict:
    return json.loads((ROOT / "prompts" / f"{name}.json").read_text())


def label_space(task: str) -> int:
    """Size of the output inventory the paper's predictor chooses from (labels, levels or candidates)."""
    fixed = {"pinose": 2, "fenice": 2, "lifetox": 2, "made": 1154, "ttp": 737, "prp_graph": 100,
             "legalseg": len(prompt("legalseg")["labels"]), "biorc800": len(prompt("biorc800")["labels"]),
             "medreadme": 6, "dress_new": len(prompt("dress_new")["levels"]), "factpico": 5,
             "relemb/wikiwd": len(prompt("relemb_wikiwd_v2")["labels"]),
             "relemb/nytfb": len(prompt("relemb_nytfb_v2")["labels"])}
    for k, v in fixed.items():
        if task == k or task.startswith(k + "/"):
            return v
    if task.startswith("scalar_constructs/"):
        return 2 if task.endswith("pairwise") else 9
    if task.startswith("nav_unknown/"):
        geom = {d[0]: d[3] for d in pt.DATASETS}[task]  # "76-way" = ID intents + OOD fallback
        return int(re.match(r"(\d+)-way", geom).group(1))
    if task.startswith("cicc/"):
        return len(prompt("cicc_" + task.split("/")[1])["question"]["criteria"])
    if task.startswith("edit_intent"):
        return len(prompt("edit_intent")["modify"]["criteria"])
    if task.startswith("ezstance/"):
        return len(prompt("ezstance")["claim"]["criteria"])
    name = {"contempo/matres": "contempo_matres", "contempo/tbdense": "contempo_tbdense"}.get(task, task.replace("/", "_"))
    return len(prompt(name)["question"]["criteria"])


def spearman(x: list[float], y: list[float]) -> float:
    def ranks(v):
        v = np.asarray(v, float)
        order = v.argsort()
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        for val in np.unique(v):  # average ties
            idx = v == val
            r[idx] = r[idx].mean()
        return r
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def manifest(task: str) -> dict:
    return json.loads((ROOT / "runs" / task / "manifest.json").read_text())


# ---------------------------------------------------------------------------------------------
# Audit analyses
# ---------------------------------------------------------------------------------------------

def audit_rows() -> list[dict]:
    rs = [r for r in pt.rows() if r["paper"] is not None]
    for r in rs:
        r["K"] = label_space(r["task"])
        m = manifest(r["task"])
        r["cost"] = m["cost_usd"]
        r["tokens"] = m["input_tokens"]
        r["requests"] = m["n_requests"]
        r["cost_per_1k"] = 1000 * m["cost_usd"] / r["n"]
        r["geometry"] = group_of(r["task"], GEOMETRY)
        dom = DOMAIN_OF[r["key"]]
        r["domain"] = next(g for g, s in DOMAIN_GROUPS if dom in s)
    return rs


def outcome_table(rs: list[dict], by: str) -> str:
    """Outcome counts by output geometry (body, tab:outcomes) or by domain (appendix, tab:outcomes-domain)."""
    title, groups, label, pos = {
        "geometry": ("By output geometry", [g for g, _ in GEOMETRY], "tab:outcomes", "t"),
        "domain": ("By domain", [g for g, _ in DOMAIN_GROUPS], "tab:outcomes-domain", "h"),
    }[by]
    lines = [f"\\begin{{table}}[{pos}]", "\\centering\\footnotesize", "\\setlength{\\tabcolsep}{2.8pt}",
             "\\begin{tabular}{lrrrrr}", "\\toprule",
             f"\\textit{{{title}}} & Sets & \\textcolor[HTML]{{2b8a3e}}{{Match}} & \\textcolor[HTML]{{c48a1e}}{{Betw.}} "
             "& \\textcolor[HTML]{c92a2a}{Below} & Ratio \\\\",
             "\\midrule"]
    for g in groups:
        sub = [r for r in rs if r[by] == g]
        if not sub:
            continue
        c = {k: sum(r["cat"] == k for r in sub) for k in CAT_ORDER}
        med = statistics.median(r["rel"] for r in sub)
        cells = " & ".join(str(c[k]) if c[k] else "\\textcolor{gray}{0}" for k in CAT_ORDER)
        lines.append(f"{g} & {len(sub)} & {cells} & {med:.2f} \\\\")
    c = {k: sum(r["cat"] == k for r in rs) for k in CAT_ORDER}
    other = ("Table~\\ref{tab:outcomes-domain} gives the same counts by domain" if by == "geometry"
             else "Table~\\ref{tab:outcomes} gives the same counts by output geometry")
    lines += ["\\midrule", f"All & {len(rs)} & {c['replaces']} & {c['between']} & {c['below']} & {statistics.median(r['rel'] for r in rs):.2f} \\\\",
              "\\bottomrule", "\\end{tabular}",
              f"\\caption{{Zero-shot outcomes on the \\nDatasets{{}} comparable test sets by {title[3:]}. "
              "Match: \\jev{} matches or beats the paper's task-specific method. Betw.: between the paper's prompted-LLM "
              "baseline and its method. Below: below both (defined in \\S\\ref{sec:protocol}). "
              f"Ratio: median of \\jev{{}}'s score divided by the task-specific method's. {other}.}}",
              f"\\label{{{label}}}", "\\end{table}", ""]
    return "\n".join(lines)


# Paper labels set on two lines in Table 1, broken after these words.
TASKS_BREAK = {"Navigating the Unknown": "Navigating the", "Entity or Relation Embeddings": "Entity or Relation",
               "Explanation ranking": "Explanation"}


def tasks_table() -> str:
    lines = ["\\begin{table}[t]", "\\centering\\scriptsize", "\\setlength{\\tabcolsep}{2.2pt}",
             "\\resizebox{\\ifdim\\width>\\columnwidth\\columnwidth\\else\\width\\fi}{!}{%",
             "\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
             "Paper & Domain & Output & \\jev{} interface & Test \\\\", "\\midrule"]
    mark = {"released": "R", "EN subset": "E", "partial": "P", "input rebuilt": "I", "rebuilt": "C"}
    for key, dom, out, iface, status in PAPERS:
        name, cite = pt.PAPER_LABEL[key], "~\\citeyearpar{" + key + "}"
        cell = pt.two_lines(TASKS_BREAK[name], name[len(TASKS_BREAK[name]) + 1:] + cite) if name in TASKS_BREAK else name + cite
        lines.append(f"{cell} & {dom} & {out} & {iface} & {mark[status]} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}",
              "\\caption{The \\specCore{} papers in our sample and how \\jev{} replaces each paper's predictor "
              "(Appendix~\\ref{app:catalogue} describes the data of each paper with an example). In the Test column, R "
              "is the released test set, E the released set without non-English rows, P a partial release, I a rebuilt "
              "input pipeline and C a rebuilt test set or split.}",
              "\\label{tab:tasks}", "\\end{table}", ""]
    return "\n".join(lines)


def cicc_stats() -> list[dict]:
    out = []
    for t in ("mtod", "hwu64", "c150", "acid", "atis", "b77"):
        labels = list(prompt(f"cicc_{t}")["question"]["criteria"])
        s = json.loads((ROOT / "runs" / "cicc" / t / "scores.json").read_text())
        cal = zero = 0
        for line in (ROOT / "runs" / "cicc" / t / "predictions.jsonl").read_text().splitlines():
            r = json.loads(line)
            if not r["item_id"].startswith("cal"):
                continue
            cal += 1
            zero += r["prediction"]["probs"][labels.index(r["gold"])] == 0
        out.append({"t": t, "K": len(labels), "alpha": 1 - s["one_minus_alpha"], "p0": zero / cal, "qhat": s["qhat"],
                    "single": s["single"], "acc": s["accuracy"], "set": s["mean_set_size"]})
    return out


def choice_calibration() -> list[dict]:
    """Top-1 confidence vs accuracy for single-question Choice test sets with full probability vectors."""
    out = []
    for task in ("xstest", "edit_intent", "ezstance/subtaskA", "ezstance/subtaskB", "contempo/matres", "contempo/tbdense",
                 "mismatched", "inli", "smarter/hatexplain", "smarter/latent_hate", "smarter/implicit_hate"):
        conf, corr, zeros, opts = [], [], 0, 0
        for line in (ROOT / "runs" / task / "predictions.jsonl").read_text().splitlines():
            r = json.loads(line)
            p = r["prediction"]["probabilities"]
            top = r["prediction"]["label"]
            conf.append(p[top])
            corr.append(top == r["gold"])
            zeros += sum(v == 0 for k, v in p.items() if k != top)
            opts += len(p) - 1
        conf_a, corr_a = np.array(conf), np.array(corr, float)
        bins = np.minimum((conf_a * 10).astype(int), 9)
        ece = sum(abs(conf_a[bins == b].mean() - corr_a[bins == b].mean()) * (bins == b).sum() for b in range(10) if (bins == b).any()) / len(conf_a)
        out.append({"task": task, "acc": corr_a.mean(), "conf": conf_a.mean(), "ece": ece, "zero_share": zeros / opts})
    return out


def rerun_consistency() -> dict:
    """Compare the audit's test responses with the optimization study's zero-seed test responses.

    For single-question tasks the jev-opt seed sends byte-identical state and question objects
    (jev-opt/experiments/README.md), but the two harnesses hash requests differently, so Jev was
    called again. Items are matched through the experiment's test file (written by its prepare.py).
    """
    specs = [("inli", "relation", "inli"), ("contempo-tbdense", "relation", "contempo/tbdense"),
             ("smarter-implicit-hate", "label", "smarter/implicit_hate")]
    n = flips = identical = 0
    diffs: list[float] = []
    for slug, qid, task in specs:
        audit = {}
        for line in (ROOT / "runs" / task / "predictions.jsonl").read_text().splitlines():
            r = json.loads(line)
            audit[r["item_id"]] = r["prediction"]
        ids = {}
        test = ROOT / "jev-opt" / "experiments" / slug.replace("-", "_") / "data" / "test.jsonl"
        for line in test.read_text().splitlines():
            r = json.loads(line)
            ids[tuple(v for k, v in r.items() if k not in ("id", "label"))] = r["id"]
        audit_hash = {}
        for shard in sorted((ROOT / "runs" / task).glob("responses*.jsonl.gz")):
            for line in gzip.open(shard, "rt"):
                r = json.loads(line)
                audit_hash[r["item_id"]] = r["request_hash"]
        for shard in sorted((STUDY / f"study-{slug}-zero").glob("jev_responses*.jsonl.gz")):
            for line in gzip.open(shard, "rt"):
                r = json.loads(line)
                if r["split"] != "test":
                    continue
                iid = ids[tuple(r["request"]["state"].values())]
                # The gauntlet's canonical request hash (src/gauntlet/jev.py) of the jev-opt request body:
                body = {"model": r["request"]["model"], "state": r["request"]["state"], "questions": r["request"]["questions"]}
                canon = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                identical += hashlib.sha256(canon.encode("utf-8")).hexdigest() == audit_hash[iid]
                a = audit[iid]
                b = r["response"]["answers"][qid]
                n += 1
                flips += a["label"] != b["choice"]
                diffs.append(max(abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["probabilities"]))
    return {"n": n, "flips": flips, "identical": identical, "med_diff": statistics.median(diffs),
            "n_changed": sum(d > 0 for d in diffs)}


def convention_stats() -> dict[str, str]:
    """Per-class evidence for failure mode (i), from runs/<task>/scores.json."""
    tbd = json.loads((ROOT / "runs" / "contempo" / "tbdense" / "scores.json").read_text())
    conf = tbd["confusion_gold_by_pred"]
    ez = json.loads((ROOT / "runs" / "ezstance" / "subtaskA" / "scores.json").read_text())["M"]
    sm = json.loads((ROOT / "runs" / "smarter" / "implicit_hate" / "scores.json").read_text())
    inli = json.loads((ROOT / "runs" / "inli" / "scores.json").read_text())
    return {
        "bodyTbdVagueGoldPct": f"{100 * tbd['n_gold']['VAGUE'] / tbd['n']:.0f}",
        "bodyTbdVaguePredPct": f"{100 * tbd['n_pred']['VAGUE'] / tbd['n']:.0f}",
        "bodyTbdVagueRecall": f"{100 * conf['VAGUE']['VAGUE'] / tbd['n_gold']['VAGUE']:.0f}",
        "bodyEzNoneF": fmt(100 * ez["per_class_f1"]["NONE"]),
        "bodyEzAgainstF": fmt(100 * ez["per_class_f1"]["AGAINST"]),
        "bodySmIncF": fmt(100 * sm["per_class_f1"]["Incitement to Violence"]),
        "bodySmIncPred": str(sm["n_pred"]["Incitement to Violence"]),
        "bodySmStereoPred": str(sm["n_pred"]["Stereotypes and Misinformation"]),
        "bodySmPerClass": str(sm["n_gold"]["Stereotypes and Misinformation"]),
        "bodyInliImpliedAcc": fmt(100 * inli["per_label_accuracy"]["implied_entailment"]),
        "bodyInliNeutralAcc": fmt(100 * inli["per_label_accuracy"]["neutral"]),
    }


# ---------------------------------------------------------------------------------------------
# Optimization-study analyses
# ---------------------------------------------------------------------------------------------

def study_runs() -> dict[str, dict[str, Path]]:
    gepa = json.loads((ROOT / "results" / "gepa.json").read_text())
    return {e["slug"]: {c: STUDY / f"study-{e['slug']}-{c}" for c in ("zero", "text", "program")} for e in gepa["rows"]}


def shape(p: dict) -> tuple:
    return tuple(sorted((v["kind"], tuple(sorted((v.get("options") or {}).keys()))) for v in p["nodes"].values()))


def body_chars(path: Path) -> int:
    return len("".join(l for l in path.read_text().splitlines(keepends=True) if not l.startswith("#")))


def report_val_test(run: Path) -> tuple[float, float]:
    """Validation primary of the top-quality representative and its (single) test primary, from report.md."""
    text = (run / "report.md").read_text()
    val = float(re.search(r"^\| top-quality \| [^|]+ \| ([0-9.]+) \|", text, re.M).group(1))
    test = float(re.search(r"by `top-quality` \([^)]*\): primary ([0-9.]+)", text).group(1))
    return val, test


def usage(run: Path) -> dict:
    u = {"refl_calls": 0, "refl_cost": 0.0, "jev_cost": 0.0, "jev_calls": 0}
    for line in (run / "usage.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("source") == "omp":
            u["refl_calls"] += 1
            u["refl_cost"] += r.get("cost_total") or 0.0
        elif r.get("source") == "jev":
            u["jev_cost"] += r.get("cost") or 0.0
            u["jev_calls"] += r.get("calls") or 0
    return u


def gepa_stats() -> dict:
    gepa = json.loads((ROOT / "results" / "gepa.json").read_text())
    runs = study_runs()
    rows = []
    for e in gepa["rows"]:
        slug, zero, paper = e["slug"], e["zero"], e["ref"]["paper"]
        r = {"slug": slug, "title": e["title"], "zero": zero, "paper": paper}
        for c in ("text", "program"):
            x = e["conds"][c]
            r[c] = x["value"]
            r[c + "_gap"] = (x["value"] - zero) / (paper - zero)
            r[c + "_sig"] = x["ci"][0] > 0 or x["ci"][1] < 0
            seed = yaml.safe_load((runs[slug][c] / "seed.yaml").read_text())
            top = yaml.safe_load((runs[slug][c] / "top-quality.yaml").read_text())
            r[c + "_topo"] = shape(seed) != shape(top)
            r[c + "_chars"] = body_chars(runs[slug][c] / "top-quality.yaml")
            r["seed_chars"] = body_chars(runs[slug][c] / "seed.yaml")
            r[c + "_val"], r[c + "_test"] = report_val_test(runs[slug][c])
            r[c + "_usage"] = usage(runs[slug][c])
        r["zero_val"], r["zero_test"] = report_val_test(runs[slug]["zero"])
        r["zero_usage"] = usage(runs[slug]["zero"])
        rows.append(r)
    return {"rows": rows}


# ---------------------------------------------------------------------------------------------
# Final optimization results (all rounds): results/gepa_v3_test.json (every tested row, CIs),
# runs/jev-opt/{v2,v3,tailored}-dev-summary.json (validation), runs/jev-opt/<run>/test_consumed.json
# (test confusion matrices) and the tested programs' YAML. Lever C (candidate reranking and fitted thresholds):
# the lever_c block of results/gepa_v3_test.json.
# ---------------------------------------------------------------------------------------------

OPT_TAG = {"contempo-tbdense": "Tbd", "ezstance-subtaskA": "Ez", "smarter-implicit-hate": "Sm", "edit-intent": "Ed",
           "inli": "Inli", "dress-new-content": "DrC", "dress-new-organization": "DrO", "dress-new-language": "DrL"}
# Row label in results/gepa_v3_test.json -> macro/column key.
OPT_ROWS = {"v1 zero-shot": "Zero", "Jev-native zero-shot (v3 N)": "Native", "v1 text-opt": "VoneText",
            "v1 program-opt": "VoneProg", "v2 chosen": "Vtwo", "**v3 chosen (headline)**": "Vthree", "tailored": "Tailored"}


def dev_summary(name: str) -> dict:
    return json.loads((STUDY / f"{name}-dev-summary.json").read_text())


def val_of(slug: str, run: str, rep: str) -> float | None:
    """Validation primary metric of a tested representative: v1 from its report.md, later rounds from the
    analysis summaries (the entry with the same run and representative)."""
    if run.startswith("study-"):
        return report_val_test(STUDY / run)[0]
    for name in ("v2", "v3", "tailored"):
        for entry in dev_summary(name).get(slug, {}).values():
            if isinstance(entry, dict) and entry.get("run") == run and entry.get("representative") == rep:
                return entry["val"]
    return None


def readout_params(program: dict) -> int:
    n = 0
    for node in program["nodes"].values():
        ro = node.get("readout")
        if ro and ro.get("W") is not None:
            n += sum(len(row) for row in ro["W"]) + len(ro.get("b") or [])
    return n


def test_consumed(run: str) -> dict:
    return json.loads((STUDY / run / "test_consumed.json").read_text())


def seed_stats(slug: str, zero: dict, seed0: dict) -> dict:
    """The headline configuration over optimizer seeds 0-2 (results/seeds.json, analyze_seeds.py): one test
    evaluation per seed; seed 0 is the pre-registered run of results/gepa_v3_test.json."""
    t = json.loads((ROOT / "results" / "seeds.json").read_text())["tasks"][slug]
    ss = [t["seeds"][k] for k in sorted(t["seeds"], key=int)]
    if ss[0]["run"] != seed0["run"] or abs(ss[0]["test"] - seed0["test"]) > 1e-9:
        raise ValueError(f"{slug}: seed 0 of seeds.json differs from the tested headline run")
    tests = [s["test"] for s in ss]
    vals = [s["val"] for s in ss]
    tc = [test_consumed(s["run"])["metrics"] for s in ss]
    out = {"n": len(ss), "test": tests, "val": vals, "mean": statistics.mean(tests), "sd": statistics.stdev(tests),
           "val_mean": statistics.mean(vals), "reps": [s["representative"] for s in ss],
           "ci0": [s.get("ci_vs_zero") for s in ss], "jk0": [s.get("ci_vs_zero_jackknife") for s in ss],
           "cluster_unit": ss[0].get("cluster_unit"), "n_clusters": ss[0].get("n_clusters"), "cost": [m["total_cost"] for m in tc],
           "tokens": [m["mean_input_tokens"] for m in tc], "calls": [m["mean_calls"] for m in tc], "metrics": tc}
    out["delta0"] = out["mean"] - zero["test"]
    return out


def opt_final() -> dict:
    data = json.loads((ROOT / "results" / "gepa_v3_test.json").read_text())
    tasks = []
    for slug, t in data["tasks"].items():
        row = {"slug": slug, "tag": OPT_TAG[slug], "short": pt.CONFIG_LABEL[slug], "paper": t["paper"], "llm": t["paper_llm"]}
        for label, key in OPT_ROWS.items():
            x = t["rows"][label]
            metrics = test_consumed(x["run"])["metrics"]
            row[key] = {"test": x["test"], "ci0": x.get("ci_vs_v1 zero-shot"), "ci2": x.get("ci_vs_v2 chosen"),
                        "cost": x["cost"], "tokens": x["tokens"], "calls": x["calls"], "run": x["run"], "rep": x["rep"],
                        "val": val_of(slug, x["run"], x["rep"]), "n": metrics["n"], "metrics": metrics}
            if key != "Zero":
                row[key]["delta0"] = x["test"] - t["rows"]["v1 zero-shot"]["test"]
        row["Seeds"] = seed_stats(slug, row["Zero"], row["Vthree"])
        tasks.append(row)
    return {"tasks": tasks, "lever_c": data["lever_c"], "discrepancies": data["discrepancies"]}


# ---------------------------------------------------------------------------------------------
# Fitted readouts on the audit's zero-shot answers (results/heads.json, scripts/report_heads.py).
# A separate "Jev + fitted readout" condition, never a zero-shot row.
# ---------------------------------------------------------------------------------------------

HEAD_SETS = {"contempo/matres": ("Matres", "MATRES"), "contempo/tbdense": ("Tbd", "TB-Dense"),
             "smarter/hatexplain": ("Hx", "HateXplain"), "smarter/latent_hate": ("Lat", "Latent hate"),
             "smarter/implicit_hate": ("Imp", "Implicit hate")}


def heads(rs: list[dict]) -> list[dict]:
    data = json.loads((ROOT / "results" / "heads.json").read_text())
    audit = {r["task"]: r for r in rs}
    out = []
    for task, (tag, name) in HEAD_SETS.items():
        h = data[task]
        a = audit[task]
        if abs(h["test_zero_shot"]["value"] - a["jev"]) > 1e-9:
            raise ValueError(f"{task}: heads.json zero-shot test differs from the audit row")
        out.append({"task": task, "tag": tag, "name": name, "params": h["n_params"],
                    "n": h["n_test"], "zero": a["jev"], "head": h["test_head"]["value"], "delta": h["test_delta"],
                    "dev_zero": h["dev_zero_shot"], "dev_head": h["dev_head"], "dev_delta": h["dev_delta"],
                    "changed": h["n_changed"], "paper": a["paper"], "llm": a["llm"], "dev_cost": h["dev_cost_usd"],
                    "test_cost": h["test_cost_usd_new"]})
    return out


def heads_table(hs: list[dict]) -> str:
    """Body table: the paper's question's own zero-shot answers decoded by a readout fit on development data."""
    lines = ["\\begin{table}[t]", "\\centering\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
             "& & Zero- & + calibration & & \\\\",
             "Test set & Par. & shot & or LR & $\\Delta$ & 95\\% CI \\\\", "\\midrule"]
    for h in hs:
        d = h["delta"]
        lines.append(f"{h['name']} & {h['params']} & {fmt(100 * h['zero'])} & {fmt(100 * h['head'])} & "
                     f"{100 * d['delta']:+.1f} & [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}] \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{\\jev{} + calibration or LR: the paper's question's unchanged zero-shot answers decoded by class biases "
              "or a logistic regression (Par.: parameters) fit on development data, pre-registered, tested once. Test "
              "$\\times$100 (micro-F1 for ConTempo MATRES and TB-Dense, macro-F1 for SMARTER). $\\Delta$: paired bootstrap. "
              "Details: Appendix~\\ref{app:opt-heads}.}",
              "\\label{tab:heads}", "\\end{table}", ""]
    return "\n".join(lines)


def heads_macros(hs: list[dict]) -> dict[str, str]:
    m = {"hdN": str(len(hs)),
         "hdDeltaMin": f"{100 * min(h['delta']['delta'] for h in hs):+.1f}",
         "hdDeltaMax": f"{100 * max(h['delta']['delta'] for h in hs):+.1f}",
         "hdParamsMin": str(min(h["params"] for h in hs)), "hdParamsMax": str(max(h["params"] for h in hs)),
         "hdNSig": str(sum(h["delta"]["lo"] > 0 for h in hs)),
         "hdDevCost": f"\\${sum(h['dev_cost'] for h in hs):.2f}",
         "hdTestCost": f"\\${sum(h['test_cost'] for h in hs):.2f}"}
    for h in hs:
        g = h["tag"]
        m[f"hd{g}Zero"] = fmt(100 * h["zero"])
        m[f"hd{g}Head"] = fmt(100 * h["head"])
        m[f"hd{g}Delta"] = f"{100 * h['delta']['delta']:+.1f}"
        m[f"hd{g}Params"] = str(h["params"])
        m[f"hd{g}Paper"] = fmt(100 * h["paper"])
    return m


def optimizer_spend() -> dict[str, tuple[int, float]]:
    """Optimizer-LLM calls and dollars per round, from each run's host usage ledger (usage.jsonl)."""
    rounds = {"study": "v1", "v2": "v2", "v3": "v3", "tailored": "tailored"}
    out = {v: [0, 0.0] for v in rounds.values()}
    for f in STUDY.glob("*/usage.jsonl"):
        pre = f.parent.name.split("-")[0]
        if pre not in rounds:
            continue
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("source") == "omp":
                out[rounds[pre]][0] += 1
                out[rounds[pre]][1] += r.get("cost_total") or 0.0
    return {k: (v[0], v[1]) for k, v in out.items()}


def hc(text: str, grey: bool = False) -> str:
    """A header cell centered over its column (keeping the grey of the optimized-question columns)."""
    spec = ">{\\columncolor{gray!10}}c" if grey else "c"
    return f"\\multicolumn{{1}}{{{spec}}}{{{text}}}"


def down(text: str, rows: int = 2, raise_: str = "") -> str:
    """A one-line header centered vertically over the header rows above it (set in the bottom row, so a cell colour
    painted later cannot cover it)."""
    return f"\\multirow{{-{rows}}}{{*}}{raise_}{{{text}}}"


def opt_table(o: dict) -> str:
    """Body Table: every tested configuration of the optimization study on the audit's test sets."""
    cols = ["Zero", "Native", "VoneText", "VoneProg", "Vtwo", "Tailored", "Vthree"]
    budget = json.loads((ROOT / "results" / "BUDGET_BASELINE.json").read_text())["sets"]  # budget_report.py
    clef_rq3 = json.loads((ROOT / "results" / "CLEF_RQ3.json").read_text())  # jev-opt/experiments/analyze_clef_rq3.py
    other = {m: {r["set"]: r for r in clef_rq3[m]["sets"]} for m in ("clef", "clef-flash")}
    lines = ["\\begin{table*}[t]", "\\centering\\small", "\\setlength{\\tabcolsep}{3.6pt}", "\\resizebox{\\textwidth}{!}{%",
             "\\begin{tabular}{@{}lccccccc>{\\columncolor{gray!10}}c>{\\columncolor{gray!10}}cccccc@{}}", "\\toprule",
             # One line per group name, a rule under each group, column names below (no vertical rules: they break
             # at the booktabs rules).
             "& & \\multicolumn{2}{c}{Zero-shot} & \\multicolumn{4}{c}{Prompt search (training data)} "
             "& \\multicolumn{2}{c}{Optimized q.\\ (ours)} & \\multicolumn{2}{c}{Other model} & \\multicolumn{1}{c}{Same labels} "
             "& \\multicolumn{2}{c}{Paper} \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-8}\\cmidrule(lr){9-10}\\cmidrule(lr){11-12}\\cmidrule(lr){13-13}"
             "\\cmidrule(l){14-15}",
             "& & & & " + " & ".join(hc(x) for x in ("GEPA", "GEPA", "GEPA", "Jev-")) + " & & & \\multicolumn{2}{c}{same q.} & & & \\\\",
             "\\multirow{-3}{*}[0.5ex]{Configuration} & " + hc(down("$N$", 3, "[0.5ex]")) + " & " + hc(down("Paper's q.")) + " & "
             + hc(down("Rephr. q.")) + " & " + " & ".join(hc(x) for x in ("text", "prog.", "+d/c", "adapted")) + " & "
             + hc(down("mean"), grey=True) + " & " + hc(down("$\\Delta$"), grey=True) + " & " + hc("Clef") + " & " + hc("Flash")
             + " & " + " & ".join(hc(down(x)) for x in ("Encoder", "Method", "LLM")) + " \\\\",
             "\\midrule"]

    def seed_sig(t: dict) -> bool | None:
        cis = t["Seeds"]["ci0"]
        return None if any(c is None for c in cis) else all(c[0] > 0 for c in cis)

    for t in o["tasks"]:
        score = {c: (t["Seeds"]["mean"] if c == "Vthree" else t[c]["test"]) for c in cols}
        jev_best = max(score.values())
        cells = []
        for c in cols:
            x = t[c]
            v = f"{100 * score[c]:.1f}"
            if score[c] == jev_best:
                v = f"\\underline{{{v}}}"
            if c == "Vthree":
                s = t["Seeds"]
                mark = "$^\\circ$" if seed_sig(t) is False else ""
                v = f"\\textbf{{{v}}}{mark}{{\\tiny\\,$\\pm${100 * s['sd']:.1f}}} & \\tiny{{{100 * s['delta0']:+.1f}}}"
            elif c != "Zero" and x["ci0"] and x["ci0"][0] <= 0 <= x["ci0"][1]:
                v += "$^\\circ$"
            elif c == "Zero" and t["slug"].startswith("dress-"):
                v += "$^\\dagger$"  # Score cutoffs fit on training data: not prompt only
            cells.append(v)
        b = budget[t["slug"]]
        bsd = f"{{\\tiny\\,$\\pm${100 * b['best_test_sd']:.1f}}}" if len(b["families"][b["best"]]["test"]) > 1 else ""
        bmark = "$^\\ast$" if b["best_vs_jev"]["p_holm"] < 0.05 else ""
        oc = []
        for m in ("clef", "clef-flash"):
            r = other[m][t["slug"]]
            ci0 = r["contrasts"]["opt-zero"]["ci"]
            oc.append(f"{100 * r['optimized']:.1f}" + ("$^\\circ$" if ci0[0] <= 0 <= ci0[1] else ""))
        lines.append(f"{t['short']} & {t['Zero']['n']:,} & " + " & ".join(cells) + " & " + " & ".join(oc)
                     + f" & {100 * b['best_test_mean']:.1f}{bmark}{bsd} & {100 * t['paper']:.1f} & {100 * t['llm']:.1f} \\\\")
    nseeds = min(t["Seeds"]["n"] for t in o["tasks"])
    lines += ["\\bottomrule", "\\end{tabular}}",
              "\\caption{Test scores ($\\times$100) on the \\optNTasks{} configurations chosen for large "
              f"convention-driven gaps. The optimized questions (bold) are a mean $\\pm$ sd over {nseeds} seeds, "
              "$\\Delta$ is their gain over the paper's question, and underline marks the best \\jev{} score in the row. "
              "Jev-adapted is GEPA text search changed for \\jev{}, which fits class biases or Score cutoffs but no "
              "logistic regression before scoring. $^\\dagger$ marks Score cutoffs fit on training data, $^\\circ$ a "
              "95\\% CI of the difference to the paper's question that includes 0, and $^\\ast$ Holm $p<0.05$ against the "
              "optimized questions.}",
              "\\label{tab:opt}", "\\end{table*}", ""]
    return "\n".join(lines)


def tex_quote(s: str) -> str:
    """Verbatim program text as LaTeX (special characters escaped, `field` names in typewriter)."""
    s = " ".join(s.split())
    rep = {"\\": "\\textbackslash{}", "{": "\\{", "}": "\\}", "$": "\\$", "&": "\\&", "%": "\\%", "#": "\\#",
           "_": "\\_", "^": "\\^{}", "~": "\\~{}", "<": "\\textless{}", ">": "\\textgreater{}"}
    s = "".join(rep.get(ch, ch) for ch in s)
    return re.sub(r"`([^`]+)`", r"\\texttt{\1}", s)


def program_figure(o: dict) -> str:
    """Figure: the TB-Dense paper's question next to the pre-registered v3 program (quoted from the YAML files)."""
    t = next(x for x in o["tasks"] if x["slug"] == "contempo-tbdense")
    seed = yaml.safe_load((STUDY / t["Zero"]["run"] / "top-quality.yaml").read_text())["nodes"]["relation"]
    v3p = yaml.safe_load((STUDY / t["Vthree"]["run"] / f"{t['Vthree']['rep']}.yaml").read_text())
    v3 = v3p["nodes"]["relation"]
    sents = re.split(r"(?<=\.)\s+", " ".join(seed["instructions"].split()))
    rule = next(s for s in sents if "80% confidence rule" in s)
    seed_instr = f"``{tex_quote(sents[0])} [\\ldots] {tex_quote(rule)} [\\ldots]''"

    def crit(node: dict, opt: str) -> str:
        c = node["options"][opt]["criterion"]
        return f"\\texttt{{{tex_quote(opt)}}}: ``{tex_quote(c.split(': ', 1)[1] if c.startswith(opt + ':') else c)}''"
    fit_n = v3["readout"]["fit"]["n"]
    body = [
        "\\begin{figure}[t]", "\\centering", "\\fbox{\\begin{minipage}{0.96\\columnwidth}\\scriptsize\\raggedright",
        f"\\textbf{{The paper's question}} (test {100 * t['Zero']['test']:.1f}, {t['Zero']['tokens']:,.0f} input tokens per pair). "
        f"One \\textsc{{Choice}} with instructions of {len(' '.join(seed['instructions'].split())):,} characters from the "
        f"annotation guidelines: {seed_instr}\\\\[1pt]",
        crit(seed, "VAGUE"),
        "\\par\\smallskip\\hrule\\smallskip",
        f"\\textbf{{Optimized questions}}, first seed (test {100 * t['Vthree']['test']:.1f}, {t['Vthree']['tokens']:,.0f} tokens). "
        f"One \\textsc{{Choice}}, decoded by a logistic regression on its answers ({readout_params(v3p)} parameters fit on {fit_n} "
        f"training pairs). Instructions: ``{tex_quote(v3['instructions'])}''\\\\[1pt]",
        "\\\\".join(crit(v3, k) for k in v3["options"]),
        "\\end{minipage}}",
        "\\caption{The TB-Dense question before and after optimization (seed 0).}",
        "\\label{fig:program}", "\\end{figure}", ""]
    return "\n".join(body)


def opt_macros(o: dict) -> dict[str, str]:
    m: dict[str, str] = {}
    ts = o["tasks"]
    # Headline summaries use the mean over optimizer seeds (results/seeds.json); seed 0 alone is in the appendix.
    hs = [t["Seeds"] for t in ts]
    m["optNTasks"] = str(len(ts))
    m["optNSeeds"] = str(min(s["n"] for s in hs))
    m["optDeltaMin"] = f"{100 * min(s['delta0'] for s in hs):+.1f}"
    m["optDeltaMax"] = f"{100 * max(s['delta0'] for s in hs):+.1f}"
    if all(c is not None for s in hs for c in s["ci0"]):
        m["optNSig"] = str(sum(all(c[0] > 0 for c in s["ci0"]) for s in hs))
        m["optSeedDeltaMin"] = f"{100 * min(t['Seeds']['test'][i] - t['Zero']['test'] for t in ts for i in range(t['Seeds']['n'])):+.1f}"
    m["optNBeatLLM"] = str(sum(t["Seeds"]["mean"] > t["llm"] for t in ts))
    m["optNBeatPaper"] = str(sum(t["Seeds"]["mean"] >= t["paper"] for t in ts))
    m["optSeedSdMin"] = fmt(100 * min(s["sd"] for s in hs))
    m["optSeedSdMax"] = fmt(100 * max(s["sd"] for s in hs))
    m["optSeedRangeMax"] = fmt(100 * max(max(s["test"]) - min(s["test"]) for s in hs))
    m["optNVtwoInRange"] = str(sum(min(t["Seeds"]["test"]) <= t["Vtwo"]["test"] <= max(t["Seeds"]["test"]) for t in ts))
    m["optNVtwoBelowRange"] = str(sum(t["Vtwo"]["test"] < min(t["Seeds"]["test"]) for t in ts))
    gaps = [(t["Seeds"]["mean"] - t["Zero"]["test"]) / (t["paper"] - t["Zero"]["test"]) for t in ts]
    m["optGapMin"] = f"{100 * min(gaps):.0f}"
    m["optGapMax"] = f"{100 * max(gaps):.0f}"
    m["optGapMedian"] = f"{100 * statistics.median(gaps):.0f}"
    within = [t for t in ts if t["paper"] - t["Seeds"]["mean"] <= 0.015]
    m["optNWithinPaper"] = str(len(within))
    vt = [s["val_mean"] - s["mean"] for s in hs]
    pos = [d for d in vt if d > 0]
    m["optNValAbove"] = str(len(pos))
    m["optValGapMin"] = fmt(100 * min(pos))
    m["optValGapMax"] = fmt(100 * max(pos))
    m["optValBelowMax"] = fmt(-100 * min(vt))
    nat = [t["Native"] for t in ts]
    m["optNNativeBetter"] = str(sum(x["ci0"][0] > 0 for x in nat))
    m["optNNativeWorse"] = str(sum(x["ci0"][1] < 0 for x in nat))
    m["optNTailoredBest"] = str(sum(t["Tailored"]["test"] > t["Seeds"]["mean"] for t in ts))
    costs = [c for s in hs for c in s["cost"]]
    m["optTestCostMin"] = f"\\${min(costs):.2f}"
    m["optTestCostMax"] = f"\\${max(costs):.2f}"
    # Jev API cost of the optimized questions per 1,000 test items (each seed's test evaluation, all questions asked)
    per_k = [1000 * mt["total_cost"] / mt["n"] for s in hs for mt in s["metrics"]]
    m["optTestCostPerKMedian"] = f"\\${statistics.median(per_k):.2f}"
    m["optTestCostPerKMin"] = f"\\${min(per_k):.2f}"
    m["optTestCostPerKMax"] = f"\\${max(per_k):.2f}"
    for t in ts:
        g, s = t["tag"], t["Seeds"]
        m[f"opt{g}Mean"] = fmt(100 * s["mean"])
        m[f"opt{g}Sd"] = fmt(100 * s["sd"])
        m[f"opt{g}MeanDelta"] = f"{100 * s['delta0']:+.1f}"
        m[f"opt{g}SeedMin"] = fmt(100 * min(s["test"]))
        m[f"opt{g}SeedMax"] = fmt(100 * max(s["test"]))
        m[f"opt{g}SeedTokensMin"] = f"{min(s['tokens']):,.0f}"
        m[f"opt{g}SeedTokensMax"] = f"{max(s['tokens']):,.0f}"
    for t in ts:
        g = t["tag"]
        for key in OPT_ROWS.values():
            x = t[key]
            m[f"opt{g}{key}"] = fmt(100 * x["test"])
            if x["val"] is not None:
                m[f"opt{g}{key}Val"] = fmt(100 * x["val"])
            if key != "Zero":
                m[f"opt{g}{key}Delta"] = f"{100 * x['delta0']:+.1f}"
                m[f"opt{g}{key}Lo"] = f"{100 * x['ci0'][0]:+.1f}"
                m[f"opt{g}{key}Hi"] = f"{100 * x['ci0'][1]:+.1f}"
            m[f"opt{g}{key}Tokens"] = f"{x['tokens']:,.0f}"
        m[f"opt{g}Paper"] = fmt(100 * t["paper"])
        m[f"opt{g}LLM"] = fmt(100 * t["llm"])
        m[f"opt{g}VthreeVtwoDelta"] = f"{100 * (t['Vthree']['test'] - t['Vtwo']['test']):+.1f}"
    # Validation-only rows of v3 (as written; seed-dependent search) from the analysis summary.
    v3s = dev_summary("v3")
    for slug, g in OPT_TAG.items():
        s = v3s[slug]
        for name, key in (("zero: paper seed", "ValPaperAsWritten"), ("zero: jev-native seed", "ValNativeAsWritten"),
                          ("search from paper", "ValSearchPaper"), ("search from jev-native", "ValSearchNative"),
                          ("A: seed + readout head", "ValHeadOnly")):
            m[f"opt{g}{key}"] = fmt(100 * s[name]["val"])
    # TB-Dense: how much of the gain is a shift towards VAGUE (test confusion matrices).
    tbd = next(t for t in ts if t["slug"] == "contempo-tbdense")

    def vague(x: dict) -> tuple[float, float, float]:
        c = x["metrics"]["confusion"]
        i = c["labels"].index("VAGUE")
        mat = np.array(c["matrix"])
        return mat[i].sum() / mat.sum(), mat[:, i].sum() / mat.sum(), mat[i, i] / mat[i].sum()
    gold, _, _ = vague(tbd["Zero"])
    m["optTbdVagueGold"] = f"{100 * gold:.0f}"
    for key in ("Zero", "Native", "Vthree"):
        _, pred, rec = vague(tbd[key])
        m[f"optTbdVaguePred{key}"] = f"{100 * pred:.0f}"
        m[f"optTbdVagueRecall{key}"] = f"{100 * rec:.0f}"
        m[f"optTbdBalAcc{key}"] = fmt(100 * tbd[key]["metrics"]["balanced_accuracy"])
        m[f"optTbdMacroF{key}"] = fmt(100 * tbd[key]["metrics"]["macro_f1"])
    # The same for every seed of the headline configuration (range of the VAGUE share, mean balanced accuracy).
    seed_metrics = tbd["Seeds"]["metrics"]
    preds = [vague({"metrics": x})[1] for x in seed_metrics]
    m["optTbdVaguePredSeedMin"] = f"{100 * min(preds):.0f}"
    m["optTbdVaguePredSeedMax"] = f"{100 * max(preds):.0f}"
    m["optTbdBalAccSeedMean"] = fmt(100 * statistics.mean(x["balanced_accuracy"] for x in seed_metrics))
    v3prog = yaml.safe_load((STUDY / tbd["Vthree"]["run"] / f"{tbd['Vthree']['rep']}.yaml").read_text())
    m["optTbdReadoutParams"] = str(readout_params(v3prog))
    m["optFitN"] = str(v3prog["nodes"]["relation"]["readout"]["fit"]["n"])
    # Lever C (runs/<task>_v3).
    for task, g in (("ttp_v3/tram", "TtpTram"), ("ttp_v3/procedures_plus", "TtpProc"), ("ttp_v3/expert", "TtpExpert"),
                    ("ttp_v3/expert_full", "TtpExpertFull"), ("relemb_v3/wikiwd", "Wiki"), ("made_v3", "Made")):
        x = o["lever_c"][task]
        m[f"lev{g}"] = fmt(100 * x["test"])
        m[f"lev{g}Prev"] = fmt(100 * x["prev"])
        m[f"lev{g}Paper"] = fmt(100 * x["paper"])
    m["levNewCost"] = f"\\${sum(x['cost'] for x in o['lever_c'].values()):.2f}"
    for rnd, (calls, cost) in optimizer_spend().items():
        key = {"v1": "Vone", "v2": "Vtwo", "v3": "Vthree", "tailored": "Tailored"}[rnd]
        m[f"optSpend{key}"] = f"\\${cost:.0f}" if cost >= 10 else f"\\${cost:.2f}"
        m[f"optCalls{key}"] = f"{calls:,}"
    m["optDiscrepancies"] = str(len(o["discrepancies"]))
    return m


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------

def _box(ax, x, y, w, h, title, body, fc="#f4f6f8", ec="#495057", tsize=7, bsize=6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.03", fc=fc, ec=ec, lw=0.6))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=tsize, fontweight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.05, body, ha="center", va="center", fontsize=bsize, linespacing=1.15)


def _arrow(ax, a, b, color="#343a40", style="-|>", ls="-", rad=0.0, lw=0.8):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=7, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


def fig_overview() -> None:
    fig = plt.figure(figsize=(COLW, 1.72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 3.03)
    ax.set_ylim(0, 1.72)
    ax.axis("off")
    w, h = 0.95, 0.68
    xs = [0.02, 1.04, 2.06]
    top, bot = 1.0, 0.03
    _box(ax, xs[0], top, w, h, "ACL paper", "trained predictor;\nreported scores of the\nmethod and an LLM", fc="#e9ecef")
    _box(ax, xs[1], top, w, h, "Bounded decision", "label set, scale or\ncandidates, at the\npredictor's output", fc="#e9ecef")
    _box(ax, xs[2], top, w, h, "Frozen question", "paper's own label\ndefinitions, verbatim;\nzero-shot, fixed", fc="#fff3bf", ec="#e8a33d")
    _box(ax, xs[2], bot, w, h, "Decision model", "typed Choice, Score\nor Noul; a choice and\na probability per option", fc="#d0ebff", ec="#1c7ed6")
    _box(ax, xs[1], bot, w, h, "Paper's metric", "paper's English test\nset, split and\nprimary metric", fc="#e9ecef")
    _box(ax, xs[0], bot, w, h, "Comparison", "vs. the paper's task-\nspecific method and\nits prompted LLM", fc="#d3f9d8", ec="#2b8a3e")
    ym_t, ym_b = top + h / 2, bot + h / 2
    _arrow(ax, (xs[0] + w, ym_t), (xs[1], ym_t))
    _arrow(ax, (xs[1] + w, ym_t), (xs[2], ym_t))
    _arrow(ax, (xs[2] + w / 2, top), (xs[2] + w / 2, bot + h))
    _arrow(ax, (xs[2], ym_b), (xs[1] + w, ym_b))
    _arrow(ax, (xs[1], ym_b), (xs[0] + w, ym_b))
    ax.text(xs[2] + w / 2 - 0.05, (top + bot + h) / 2, "every test item; raw responses stored", fontsize=6,
            va="center", ha="right", color="#1c7ed6")
    fig.savefig(OUT / "body_overview.pdf")
    plt.close(fig)


def fig_architecture() -> None:
    fig = plt.figure(figsize=(COLW, 1.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 3.03)
    ax.set_ylim(0, 1.75)
    ax.axis("off")
    _box(ax, 0.04, 0.92, 1.12, 0.78, "Agent harness (OMP)", "model registry and\ncredentials, reasoning\nlevel, usage ledger;\nanswers lm.complete", fc="#fff3bf", ec="#e8a33d")
    _box(ax, 0.04, 0.05, 1.12, 0.62, "Optimizer LLM", "any model the harness\ncan call (hosted or local)", fc="#f8f9fa")
    _box(ax, 1.42, 0.62, 1.57, 1.08, "jev-opt worker (Python)",
         "GEPA: Pareto selection, reflective\nmutation, merge, stopping\n\njev-opt: program IR, structural\noperators, numeric fitting, metrics,\nresponse cache, split hygiene\n(no LLM client installed)",
         fc="#e9ecef")
    _box(ax, 1.42, 0.05, 0.72, 0.42, "Jev", "Decisions API", fc="#d0ebff", ec="#1c7ed6")
    _box(ax, 2.27, 0.05, 0.72, 0.42, "Data", "train, val;\ntest once", fc="#d3f9d8", ec="#2b8a3e")
    _arrow(ax, (1.16, 1.36), (1.42, 1.36), style="<|-|>")
    ax.text(1.29, 1.40, "RPC", fontsize=6, ha="center", va="bottom")
    _arrow(ax, (0.6, 0.92), (0.6, 0.67), style="<|-|>")
    _arrow(ax, (1.78, 0.62), (1.78, 0.47), style="<|-|>")
    _arrow(ax, (2.63, 0.47), (2.63, 0.62))
    fig.savefig(OUT / "body_architecture.pdf")
    plt.close(fig)


def fig_factors(rs: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 1.65), sharey=True)
    ylab = "Jev / task-specific"
    for r in rs:
        c = COLORS[r["cat"]]
        if r["llm"] is not None:
            axes[0].scatter(r["llm"] / r["paper"], r["rel"], s=9, color=c, lw=0, alpha=0.9)
        axes[1].scatter(r["K"], r["rel"], s=9, color=c, lw=0, alpha=0.9)
        axes[2].scatter(r["cost_per_1k"], r["rel"], s=9, color=c, lw=0, alpha=0.9)
    lo, hi = 0, 1.2
    axes[0].plot([lo, hi], [lo, hi], color="#868e96", lw=0.6, ls=":")
    axes[0].set_xlim(lo, hi)
    axes[0].set_xlabel("paper's prompted LLM / task-specific")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("output inventory size (labels, levels, candidates)")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Jev cost per 1,000 test units (USD)")
    titles = ("(a) vs. the paper's prompted LLM", "(b) vs. output inventory size", "(c) vs. Jev cost")
    for ax, title in zip(axes, titles):
        ax.axhline(1.0, color="black", lw=0.6, ls="--")
        ax.set_ylim(-0.05, 1.3)
        ax.set_title(title, loc="left", fontsize=7, pad=3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(ylab)
    handles = [axes[2].scatter([], [], s=9, color=COLORS[k]) for k in CAT_ORDER]
    fig.legend(handles, ["matches or beats the task-specific method", "between the prompted LLM and the method", "below both"],
               loc="upper center", ncol=3, frameon=False, handletextpad=0.2, columnspacing=1.5, fontsize=6.5)
    fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0, 1, 0.91))
    fig.savefig(OUT / "body_factors.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Macros
# ---------------------------------------------------------------------------------------------

def fmt(x: float, d: int = 1) -> str:
    return f"{x:.{d}f}"


def macros(rs: list[dict], cicc: list[dict], cal: list[dict], g: dict, rerun: dict, o: dict, hs: list[dict]) -> str:
    m: dict[str, str] = {}
    with_llm = [r for r in rs if r["llm"] is not None]
    m["bodyNWithLLM"] = str(len(with_llm))
    m["bodyRhoLLM"] = fmt(spearman([r["llm"] / r["paper"] for r in with_llm], [r["rel"] for r in with_llm]), 2)
    m["bodyNJevGeLLM"] = str(sum(r["jev"] >= r["llm"] - 0.005 for r in with_llm))
    m["bodyRhoK"] = fmt(spearman([np.log(r["K"]) for r in rs], [r["rel"] for r in rs]), 2)
    small = [r for r in rs if r["K"] <= 10]
    large = [r for r in rs if r["K"] >= 100]
    m["bodyNSmallK"] = str(len(small))
    m["bodyNLargeK"] = str(len(large))
    m["bodyMedRelSmallK"] = fmt(statistics.median(r["rel"] for r in small), 2)
    m["bodyMedRelLargeK"] = fmt(statistics.median(r["rel"] for r in large), 2)
    m["bodyNLargeKBelow"] = str(sum(r["cat"] == "below" for r in large))
    m["bodyMedRelAll"] = fmt(statistics.median(r["rel"] for r in rs), 2)
    m["bodyMedRelAllPct"] = f"{100 * statistics.median(r['rel'] for r in rs):.0f}"  # the same median, in percent
    cpk = sorted(r["cost_per_1k"] for r in rs)
    m["bodyCostPerKMedian"] = f"\\${statistics.median(cpk):.2f}"
    m["bodyCostPerKMin"] = f"\\${cpk[0]:.3f}"
    m["bodyCostPerKMax"] = f"\\${cpk[-1]:,.0f}"
    m["bodyRhoCost"] = fmt(spearman([r["cost_per_1k"] for r in rs], [r["rel"] for r in rs]), 2)
    rep = [r for r in rs if r["cat"] == "replaces"]
    m["bodyCostPerKMedReplaces"] = f"\\${statistics.median(r['cost_per_1k'] for r in rep):.2f}"
    # Same manifest set as paper_tables.macros() (\totalCost): every zero-shot audit run.
    tot_cost = sum(x["cost_usd"] for _, x in pt.paper_appendix.run_manifests())
    big = ("relemb/wikiwd", "made", "ttp/tram", "ttp/procedures_plus", "ttp/expert", "ttp/expert_full")  # >300 labels
    m["bodyLargeLabelCostShare"] = f"{100 * sum(manifest(t)['cost_usd'] for t in big) / tot_cost:.0f}"
    m["bodyWikiCost"] = f"\\${manifest('relemb/wikiwd')['cost_usd']:.0f}"
    m["bodyWikiTokensB"] = fmt(manifest("relemb/wikiwd")["input_tokens"] / 1e9, 2)
    # CICC
    for c in cicc:
        t = {"mtod": "Mtod", "hwu64": "Hwu", "c150": "Clinc", "acid": "Acid", "atis": "Atis", "b77": "Bank"}[c["t"]]
        m[f"bodyCicc{t}Zero"] = fmt(100 * c["p0"], 1)
        m[f"bodyCicc{t}Alpha"] = fmt(100 * c["alpha"], 0)
        m[f"bodyCicc{t}Acc"] = fmt(100 * c["acc"], 1)
    compared = [c for c in cicc if c["t"] != "b77"]  # BANKING77 is not compared (App. repro: other coverage level)
    full = sorted(c["t"] for c in compared if c["qhat"] >= 1.0)
    if full != ["acid", "atis", "c150"]:  # results.tex names these sets
        raise ValueError(f"CICC sets with only full-label prediction sets changed: {full}")
    m["bodyCiccNFull"] = str(len(full))
    m["bodyCiccNSets"] = str(len(compared))
    # Calibration of single-question Choice tasks
    m["bodyCalN"] = str(len(cal))
    m["bodyCalNOver"] = str(sum(c["conf"] > c["acc"] for c in cal))
    m["bodyCalEceMin"] = fmt(min(c["ece"] for c in cal), 3)
    m["bodyCalEceMax"] = fmt(max(c["ece"] for c in cal), 3)
    m["bodyCalZeroMed"] = fmt(100 * statistics.median(c["zero_share"] for c in cal), 0)
    # Optimization study
    rows = g["rows"]
    best = [max(r["text_gap"], r["program_gap"]) for r in rows]
    m["bodyGapMin"] = f"{100 * min(best):.0f}"
    m["bodyGapMax"] = f"{100 * max(best):.0f}"
    m["bodyGapMedian"] = f"{100 * statistics.median(best):.0f}"
    m["bodyNTopoChanged"] = str(sum(r["program_topo"] for r in rows))
    m["bodyNProgramSig"] = str(sum(r["program_sig"] and r["program"] > r["zero"] for r in rows))
    m["bodyNTextSig"] = str(sum(r["text_sig"] and r["text"] > r["zero"] for r in rows))
    ratio_text = [r["text_chars"] / r["seed_chars"] for r in rows]
    ratio_prog = [r["program_chars"] / r["seed_chars"] for r in rows]
    m["bodyGrowthText"] = fmt(statistics.median(ratio_text), 1)
    m["bodyGrowthProgram"] = fmt(statistics.median(ratio_prog), 1)
    for c in ("text", "program"):
        m[f"bodyReflCalls{c.capitalize()}"] = f"{sum(r[c + '_usage']['refl_calls'] for r in rows):,}"
    m["bodyReflCallsAll"] = f"{sum(r[c + '_usage']['refl_calls'] for r in rows for c in ('text', 'program')):,}"
    m["bodyReflCostAll"] = f"\\${sum(r[c + '_usage']['refl_cost'] for r in rows for c in ('text', 'program')):.0f}"
    m["bodyJevCostAll"] = f"\\${sum(r[c + '_usage']['jev_cost'] for r in rows for c in ('zero', 'text', 'program')):.2f}"
    for r in rows:
        tag = {"contempo-tbdense": "Tbd", "ezstance-subtaskA": "Ez", "smarter-implicit-hate": "Sm", "edit-intent": "Ed", "inli": "Inli",
               "dress-new-content": "DrC", "dress-new-organization": "DrO", "dress-new-language": "DrL"}[r["slug"]]
        for c in ("text", "program"):
            C = c.capitalize()
            m[f"body{tag}Gap{C}"] = f"{100 * r[c + '_gap']:.0f}"
            m[f"body{tag}Val{C}"] = fmt(100 * r[c + "_val"])
            m[f"body{tag}Test{C}"] = fmt(100 * r[c + "_test"])
            m[f"body{tag}Chars{C}"] = f"{r[c + '_chars']:,}"
        m[f"body{tag}ValZero"] = fmt(100 * r["zero_val"])
        m[f"body{tag}TestZero"] = fmt(100 * r["zero_test"])
        m[f"body{tag}CharsSeed"] = f"{r['seed_chars']:,}"
    m.update(convention_stats())
    # Per-test-set audit scores and inventory sizes for the body text: \aud<Task>{Jev,Paper,LLM,K}
    # (digits spelled out, since LaTeX command names are letters only: cicc/c150 -> CiccCOneFiveZero).
    digits = str.maketrans({d: w for d, w in zip("0123456789", ("Zero", "One", "Two", "Three", "Four", "Five",
                                                               "Six", "Seven", "Eight", "Nine"))})
    for r in rs:
        name = "aud" + "".join(p[:1].upper() + p[1:] for p in re.split(r"[/_]", r["task"])).translate(digits)
        for key, suffix in (("jev", "Jev"), ("paper", "Paper"), ("llm", "LLM")):
            if r[key] is not None:
                m[name + suffix] = fmt(100 * r[key])
        m[name + "K"] = f"{r['K']:,}"
    m.update(opt_macros(o))
    m.update(heads_macros(hs))
    tbd_head = json.loads((ROOT / "runs" / "contempo" / "tbdense+head" / "scores.json").read_text())
    m["hdTbdVaguePred"] = f"{100 * tbd_head['n_pred']['VAGUE'] / tbd_head['n']:.0f}"
    m["bodyRerunN"] = f"{rerun['n']:,}"
    m["bodyRerunFlips"] = str(rerun["flips"])
    m["bodyRerunIdentical"] = f"{rerun['identical']:,}"
    m["bodyRerunFlipPct"] = fmt(100 * rerun["flips"] / rerun["n"], 1)
    m["bodyRerunChangedPct"] = f"{100 * rerun['n_changed'] / rerun['n']:.0f}"
    m["bodyRerunMedDiff"] = f"{rerun['med_diff']:.2f}"
    # Outcome counts per output geometry (Table tab:outcomes): \bodyGeo<Tag>{N,Match,Below}
    geo_tag = {"Binary judgment": "Binary", "3--7-way choice": "Few", "19--76-way choice": "Many", "Ordinal score": "Ordinal",
               "Pairwise / ranking": "Pair", "Multi-label sequence": "Seq", "$>$300 outputs": "Large", "Conformal set": "Conformal"}
    for gname, tag in geo_tag.items():
        sub = [r for r in rs if r["geometry"] == gname]
        m[f"bodyGeo{tag}N"] = str(len(sub))
        m[f"bodyGeo{tag}Match"] = str(sum(r["cat"] == "replaces" for r in sub))
        m[f"bodyGeo{tag}Below"] = str(sum(r["cat"] == "below" for r in sub))
    return "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rs = audit_rows()
    cicc = cicc_stats()
    cal = choice_calibration()
    g = gepa_stats()
    (OUT / "body_outcomes.tex").write_text(outcome_table(rs, "geometry"))
    (OUT / "app_outcomes_domain.tex").write_text(outcome_table(rs, "domain"))
    (OUT / "body_tasks.tex").write_text(tasks_table())
    rerun = rerun_consistency()
    o = opt_final()
    hs = heads(rs)
    (OUT / "body_opt.tex").write_text(opt_table(o))
    (OUT / "body_heads.tex").write_text(heads_table(hs))
    (OUT / "body_program.tex").write_text(program_figure(o))
    (OUT / "body_numbers.tex").write_text(macros(rs, cicc, cal, g, rerun, o, hs))
    fig_overview()
    fig_architecture()
    fig_factors(rs)
    if "-v" in sys.argv:
        for r in sorted(rs, key=lambda r: r["rel"]):
            print(f"{r['task']:45s} K={r['K']:5d} rel={r['rel']:.2f} cat={r['cat']:8s} $/1k={r['cost_per_1k']:9.3f} {r['geometry']} | {r['domain']}")
        for c in cicc:
            print(c)
        for c in cal:
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()})
        for r in g["rows"]:
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if not k.endswith("usage")})
        print(rerun)
    print("wrote", sorted(p.name for p in OUT.iterdir() if p.name.startswith("body_")))


if __name__ == "__main__":
    main()
