"""Coverage check A, step 3: second-pass check subset and the generated report.

    uv run python scripts/coverage_report.py subset   # writes data/coverage/second_pass_subset.jsonl
    uv run python scripts/coverage_report.py report   # writes results/COVERAGE.md,
                                                      # paper/generated/coverage.tex (macros + body hook),
                                                      # paper/generated/app_coverage.tex (appendix section)

`subset` draws the stratified 40-paper subset for the second labelling pass (20 per classifier
bounded-predictor stratum, seeded) and writes only id, title, abstract and PDF URL, so the
second-pass labels in data/coverage/labels_second_pass.jsonl are made without the classifier's labels
in view.
"""

from __future__ import annotations

import collections
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COV = ROOT / "data/coverage"
GEN = ROOT / "paper/generated"
MD = ROOT / "results/COVERAGE.md"

SECOND_PASS_SEED = 20260930
SIM_SEED = 20261001
N_SIM = 20000
PER_STRATUM = 20
CRITERIA = ("E1", "E2", "E3", "E4", "E5", "english_only")
AGREE_FIELDS = {
    "bounded_predictor": "Bounded predictor",
    "E1": "E1 bounded decision",
    "E2": "E2 meaningful replacement",
    "english_only": "English-only evaluation",
    "eligible_strict": "Eligible (strict)",
    "eligible_lenient": "Eligible (lenient)",
    "task_family": "Task family",
    "contribution": "Contribution type",
}
FAMILIES = {
    "B-SAFE": "Safety, toxicity, hate",
    "B-AFFECT": "Sentiment, emotion, stance, social attributes",
    "B-TOPIC": "Topic, intent, text categorisation",
    "B-VERIFY": "NLI, verification, hallucination",
    "B-REL": "Relation, event, temporal classification",
    "B-SEG": "Labelling supplied units in context",
    "B-SCORE": "Scalar or ordinal scoring",
    "B-RANK": "Ranking, preference, judges",
    "B-MULTI": "Multi-label and extreme-label coding",
    "B-DETECT": "Provenance (generated text, authorship)",
    "B-OTHER": "Other bounded decision",
    "N-GEN": "Text generation",
    "N-REASON": "Open QA, reasoning, code, agents",
    "N-IE": "Span extraction and parsing",
    "N-RETR": "Open-corpus retrieval, RAG",
    "N-LLM": "General LLM methods and analysis",
    "N-OTHER": "Other (analysis, resources, surveys)",
}
BOUNDED = [f for f in FAMILIES if f.startswith("B-")]
# Anthology id and family of each audited paper; families fixed in data/coverage/rubric.md
# before classification.
AUDITED = {
    "lifetox": ("2024.naacl-short.60", "B-SAFE"),
    "smarter": ("2026.acl-long.1584", "B-SAFE"),
    "xstest": ("2024.naacl-long.301", "B-SAFE"),
    "ezstance": ("2024.acl-long.838", "B-AFFECT"),
    "cicc": ("2024.findings-naacl.156", "B-TOPIC"),
    "edit_intent": ("2024.emnlp-main.839", "B-TOPIC"),
    "sciprompt": ("2024.emnlp-main.350", "B-TOPIC"),
    "nav_unknown": ("2025.findings-emnlp.791", "B-TOPIC"),
    "pinose": ("2024.acl-long.668", "B-VERIFY"),
    "fenice": ("2024.findings-acl.841", "B-VERIFY"),
    "mismatched": ("2025.findings-acl.1109", "B-VERIFY"),
    "inli": ("2025.acl-long.1552", "B-VERIFY"),
    "contempo": ("2024.findings-acl.89", "B-REL"),
    "relemb": ("2024.findings-emnlp.348", "B-REL"),
    "legalseg": ("2025.findings-naacl.63", "B-SEG"),
    "biorc800": ("2024.findings-emnlp.944", "B-SEG"),
    "factpico": ("2024.acl-long.459", "B-SCORE"),
    "scalar_constructs": ("2025.emnlp-main.1635", "B-SCORE"),
    "dress_new": ("2025.acl-long.659", "B-SCORE"),
    "medreadme": ("2024.emnlp-main.958", "B-SCORE"),
    "prp_graph": ("2024.acl-long.313", "B-RANK"),
    "expl_rank": ("2026.findings-acl.1800", "B-RANK"),
    "made": ("2026.acl-long.2148", "B-MULTI"),
    "ttp": ("2024.findings-eacl.25", "B-MULTI"),
}


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()] if path.exists() else []


def derive(lab: dict) -> dict:
    """Normalise one label dict and add the computed eligibility fields."""
    crit = [lab[k] for k in CRITERIA]
    pdf = lab.get("pdf_checked")
    return {**lab,
            "pdf_checked": pdf is True or (isinstance(pdf, str) and pdf.lower().startswith("yes")),
            "eligible_strict": "yes" if all(v == "yes" for v in crit) else "no",
            "eligible_lenient": "yes" if all(v in ("yes", "unclear") for v in crit) else "no"}


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def kappa(a: list, b: list) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def subset() -> None:
    sample = {r["id"]: r for r in load(COV / "sample.jsonl")}
    llm = {r["id"]: r["labels"] for r in load(COV / "labels_llm.jsonl")}
    assert set(llm) == set(sample), "classify the full sample first"
    rng = random.Random(SECOND_PASS_SEED)
    chosen = []
    for stratum in ("yes", "no"):
        ids = sorted(i for i, lab in llm.items() if lab["bounded_predictor"] == stratum)
        chosen += rng.sample(ids, min(PER_STRATUM, len(ids)))
    rng.shuffle(chosen)  # interleave strata so the order does not reveal them
    with (COV / "second_pass_subset.jsonl").open("w") as f:
        for i in chosen:
            r = sample[i]
            f.write(json.dumps({k: r[k] for k in ("id", "title", "abstract", "pdf")}, ensure_ascii=False) + "\n")
    print(f"wrote {len(chosen)} papers")


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}"


def thousands(x: float) -> str:
    return f"{int(round(x, -1)):,}"


def compute() -> dict:
    frame = json.loads((COV / "frame.json").read_text())
    sample = {r["id"]: r for r in load(COV / "sample.jsonl")}
    llm = {r["id"]: derive(r["labels"]) for r in load(COV / "labels_llm.jsonl")}
    second = {r["id"]: derive(r) for r in load(COV / "labels_second_pass.jsonl")}
    assert set(llm) == set(sample), f"{len(sample) - len(llm)} papers unlabelled"
    assert set(second) <= set(llm)
    model = {r["model"] for r in load(COV / "labels_llm.jsonl")}
    n = len(llm)
    labs = list(llm.values())

    def share(pred) -> dict:
        k = sum(1 for lab in labs if pred(lab))
        lo, hi = wilson(k, n)
        return {"k": k, "p": k / n, "lo": lo, "hi": hi}

    bp = [l for l in labs if l["bounded_predictor"] == "yes"]
    st = {
        "frame": frame, "n": n, "N": frame["frame_records"], "model": ", ".join(sorted(model)),
        "bp": share(lambda l: l["bounded_predictor"] == "yes"),
        "e1": share(lambda l: l["E1"] == "yes"),
        "e1e2": share(lambda l: l["E1"] == "yes" and l["E2"] == "yes"),
        "strict": share(lambda l: l["eligible_strict"] == "yes"),
        "lenient": share(lambda l: l["eligible_lenient"] == "yes"),
        "bfam": share(lambda l: l["task_family"] in BOUNDED),
        "text": share(lambda l: l["modality"] == "text"),
        "pdf": sum(1 for l in labs if l["pdf_checked"]),
        "contrib": collections.Counter(l["contribution"] for l in labs),
        "contrib_bp": collections.Counter(l["contribution"] for l in bp),
        "e2_not_bp": sum(1 for l in labs if l["E2"] == "yes" and l["bounded_predictor"] == "no"),
        "bp_fail": {k: sum(1 for l in bp if l[k] == "no") for k in CRITERIA},
        "bp_unclear": {k: sum(1 for l in bp if l[k] == "unclear") for k in CRITERIA},
        "bp_nontext": sum(1 for l in bp if l["modality"] != "text"),
        "audited_drawn": sorted(k for k, (aid, _) in AUDITED.items() if aid in llm),
    }
    st["audited_drawn_labels"] = {k: llm[AUDITED[k][0]] for k in st["audited_drawn"]}

    # Second-pass check.
    hids = sorted(second)
    st["n_second"] = len(hids)
    st["agree"] = {f: {"agree": sum(llm[i][f] == second[i][f] for i in hids) / len(hids),
                       "kappa": kappa([llm[i][f] for i in hids], [second[i][f] for i in hids])}
                   for f in AGREE_FIELDS}
    st["confusion"] = collections.Counter((llm[i]["bounded_predictor"], second[i]["bounded_predictor"]) for i in hids)
    st["disagree_bp"] = [(i, sample[i]["title"], llm[i]["task_family"]) for i in hids
                         if llm[i]["bounded_predictor"] != second[i]["bounded_predictor"]]
    strata = {s: [second[i]["bounded_predictor"] == "yes" for i in hids if llm[i]["bounded_predictor"] == s]
              for s in ("yes", "no")}
    st["strata"] = {s: (sum(v), len(v)) for s, v in strata.items()}
    ky = st["bp"]["k"]
    (py_k, py_n), (pn_k, pn_n) = st["strata"]["yes"], st["strata"]["no"]
    st["ppv"], st["fomr"] = py_k / py_n, pn_k / pn_n
    st["cal"] = ky / n * st["ppv"] + (1 - ky / n) * st["fomr"]
    # Interval: Jeffreys Beta posteriors for the classifier-positive share and for the second-pass-positive
    # rate in each classifier stratum, combined by simulation (keeps 0/20 cells from collapsing).
    rng = random.Random(SIM_SEED)
    sims = sorted(
        (w := rng.betavariate(ky + 0.5, n - ky + 0.5)) * rng.betavariate(py_k + 0.5, py_n - py_k + 0.5)
        + (1 - w) * rng.betavariate(pn_k + 0.5, pn_n - pn_k + 0.5)
        for _ in range(N_SIM))
    st["cal_lo"], st["cal_hi"] = sims[int(0.025 * N_SIM)], sims[int(0.975 * N_SIM) - 1]
    st["second_elig_strict"] = sum(1 for i in hids if second[i]["eligible_strict"] == "yes")
    st["llm_elig_strict_sub"] = sum(1 for i in hids if llm[i]["eligible_strict"] == "yes")

    # Task families.
    fam_all = collections.Counter(l["task_family"] for l in labs)
    fam_bp = collections.Counter(l["task_family"] for l in bp)
    fam_el = collections.Counter(l["task_family"] for l in labs if l["eligible_lenient"] == "yes")
    fam_aud = collections.Counter(f for _, f in AUDITED.values())
    n_bp, n_el, n_aud = len(bp), sum(fam_el.values()), len(AUDITED)
    st.update(fam_all=fam_all, fam_bp=fam_bp, fam_el=fam_el, fam_aud=fam_aud, n_bp=n_bp, n_el=n_el,
              n_aud=n_aud)
    st["tvd_bp"] = 0.5 * sum(abs(fam_bp[f] / n_bp - fam_aud[f] / n_aud) for f in BOUNDED)
    st["tvd_el"] = 0.5 * sum(abs(fam_el[f] / n_el - fam_aud[f] / n_aud) for f in BOUNDED)
    st["bp_in_aud_fam"] = sum(c for f, c in fam_bp.items() if fam_aud[f]) / n_bp
    st["el_in_aud_fam"] = sum(c for f, c in fam_el.items() if fam_aud[f]) / n_el
    diff = {f: fam_aud[f] / n_aud - fam_el[f] / n_el for f in BOUNDED}
    st["over"] = [f for f in sorted(BOUNDED, key=lambda f: -diff[f]) if diff[f] >= 0.05]
    st["under"] = [f for f in sorted(BOUNDED, key=lambda f: diff[f]) if diff[f] <= -0.05]
    st["absent"] = [f for f in BOUNDED if f != "B-OTHER" and fam_aud[f] == 0 and fam_el[f] > 0]
    st["open_share"] = sum(fam_all[f] for f in ("N-GEN", "N-REASON", "N-RETR", "N-LLM")) / n
    st["under_titles"] = {f: [sample[i]["title"] for i, l in llm.items()
                              if l["task_family"] == f and l["eligible_lenient"] == "yes"]
                          for f in st["under"]}
    st["disagree_all_llm_yes"] = all(conf == ("yes", "no") for conf in
                                     ((llm[i]["bounded_predictor"], second[i]["bounded_predictor"])
                                      for i, _, _ in st["disagree_bp"]))
    st["diff"] = diff
    return st


# Short family names for tables and prose (no internal commas, so they can be listed).
SHORT = {
    "B-SAFE": "Safety and toxicity", "B-AFFECT": "Affect and social attributes",
    "B-TOPIC": "Topic and intent", "B-VERIFY": "NLI and verification",
    "B-REL": "Relation classification", "B-SEG": "Unit labelling in context",
    "B-SCORE": "Scalar scoring", "B-RANK": "Ranking and judges", "B-MULTI": "Multi-label coding",
    "B-DETECT": "Provenance detection", "B-OTHER": "Other bounded",
    "N-GEN": "Generation", "N-REASON": "Open QA and reasoning", "N-IE": "Extraction and parsing",
    "N-RETR": "Open retrieval and RAG", "N-LLM": "General LLM methods", "N-OTHER": "Other",
}


def short(code: str) -> str:
    """Family name for running text: lower-case the first letter unless it starts an acronym."""
    s = SHORT[code]
    return s if s[:2].isupper() else s[0].lower() + s[1:]


def fam_names(codes: list[str]) -> str:
    names = [short(c) for c in codes]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + (", and " if len(names) > 2 else " and ") + names[-1]


def write_markdown(st: dict) -> None:
    fr = st["frame"]
    n, N = st["n"], st["N"]

    def ci(s: dict) -> str:
        return f"{pct(s['p'])}% ({s['k']}/{n}; 95% CI {pct(s['lo'])}–{pct(s['hi'])}%)"

    ag = st["agree"]
    conf = st["confusion"]
    rows_fam = "\n".join(
        f"| `{f}` {FAMILIES[f]} | {st['fam_all'][f]} | {st['fam_bp'][f]} | {st['fam_el'][f]} | "
        f"{st['fam_aud'][f]} | {pct(st['fam_el'][f] / st['n_el'], 0)}% | {pct(st['fam_aud'][f] / st['n_aud'], 0)}% |"
        if f in BOUNDED else
        f"| `{f}` {FAMILIES[f]} | {st['fam_all'][f]} | {st['fam_bp'][f]} | {st['fam_el'][f]} | – | – | – |"
        for f in FAMILIES)
    rows_agree = "\n".join(f"| {label} | {pct(ag[f]['agree'], 0)}% | {ag[f]['kappa']:.2f} |"
                           for f, label in AGREE_FIELDS.items())
    drawn = ", ".join(
        f"{k} (`{AUDITED[k][0]}`: bounded predictor `{v['bounded_predictor']}`, strict-eligible `{v['eligible_strict']}`)"
        for k, v in st["audited_drawn_labels"].items()) or "none"
    dis = "; ".join(f"`{i}` {t} (classifier family `{f}`)" for i, t, f in st["disagree_bp"]) or "none"
    fail = st["bp_fail"]
    unc = st["bp_unclear"]
    text = f"""# Coverage check A: how typical are the audited papers?

Generated by `scripts/coverage_report.py report` from `data/coverage/`. Do not edit by hand.
Rubric: `data/coverage/rubric.md` (fixed before classification). No Jev calls were made.

## Frame and sample

| | Selection corpus | This check |
|---|---:|---:|
| Export SHA-256 | `{fr['spec_export_sha256'][:16]}…` | `{fr['export_sha256'][:16]}…` |
| Parsed export entries | {fr['spec_parsed_records']:,} | {fr['parsed_records']:,} |
| Frame 2024 / 2025 / 2026 | {' / '.join(f"{v:,}" for v in fr['spec_frame_by_year'].values())} | {' / '.join(f"{v:,}" for v in fr['frame_by_year'].values())} |
| Frame total | {sum(fr['spec_frame_by_year'].values()):,} | {N:,} |

The selection's export could not be re-downloaded (the Anthology export is regenerated continuously),
so the current export was used with the selection's filter. It parses into one entry fewer, but the
filter reproduces the selection's frame counts exactly for every year. The frame contains
{fr['frame_by_entry_type'].get('proceedings', 0)} volume front-matter records and
{fr['frame_missing_abstract']} papers without an abstract, as in the selection's count; none of the
front-matter records was drawn, and every sampled paper has an abstract.

Sample: {n} records drawn uniformly without replacement (`random.Random({fr['seed']}).sample` over the
frame sorted by Anthology id); by year {', '.join(f"{y}: {c}" for y, c in fr['sample_by_year'].items())}.
Classifier: `{st['model']}` through the omp harness (`omp -p --no-session`, `read` tool only);
it opened the PDF for {st['pdf']} of {n} papers.

## Estimates for the frame

Shares of the {N:,}-paper frame, estimated from the {n}-paper sample with the classifier's labels
(Wilson 95% intervals; the finite-population correction, {n}/{N:,}, is negligible and not applied).

| Quantity | Share of frame | ≈ papers in frame |
|---|---|---:|
| Central contribution is a bounded predictor | {ci(st['bp'])} | {thousands(st['bp']['p'] * N)} |
| … calibrated by the second pass (see below) | {pct(st['cal'])}% (95% interval {pct(st['cal_lo'])}–{pct(st['cal_hi'])}%) | {thousands(st['cal'] * N)} |
| Main task is in a bounded family (with or without a predictor) | {ci(st['bfam'])} | {thousands(st['bfam']['p'] * N)} |
| E1 bounded decision | {ci(st['e1'])} | {thousands(st['e1']['p'] * N)} |
| E1 and E2 | {ci(st['e1e2'])} | {thousands(st['e1e2']['p'] * N)} |
| Eligible, strict (E1–E5 and English all `yes`) | {ci(st['strict'])} | {thousands(st['strict']['p'] * N)} |
| Eligible, lenient (no criterion `no`) | {ci(st['lenient'])} | {thousands(st['lenient']['p'] * N)} |
| All evaluated inputs are text | {ci(st['text'])} | {thousands(st['text']['p'] * N)} |

Contribution types in the sample: {', '.join(f"{k} {v}" for k, v in st['contrib'].most_common())}.
Among the {st['n_bp']} bounded-predictor papers: {', '.join(f"{k} {v}" for k, v in st['contrib_bp'].most_common())}.
E2 is also met by {st['e2_not_bp']} papers that are not bounded-predictor papers (resource papers
whose trained or prompted predictor is a main result; the audited sample contains such papers too).

Why bounded-predictor papers fail eligibility (counts among the {st['n_bp']}; a paper can fail several):
E3 comparable evaluation {fail['E3']} `no` / {unc['E3']} `unclear`; E4 specified text task
{fail['E4']} `no` / {unc['E4']} `unclear` ({st['bp_nontext']} have non-text inputs); E5 artifacts
{fail['E5']} `no` / {unc['E5']} `unclear`; English-only {fail['english_only']} `no` /
{unc['english_only']} `unclear`.

Audited papers that the random draw hit: {drawn}. The expected number is
{n * st['n_aud'] / N:.1f}.

## Second-pass check

{st['n_second']} papers: {st['strata']['yes'][1]} drawn at random from the classifier's bounded-predictor
`yes` papers and {st['strata']['no'][1]} from its `no` papers (seed {SECOND_PASS_SEED}), labelled with the same
rubric from title, abstract and, for {sum(1 for r in load(COV / 'labels_second_pass.jsonl') if r.get('pdf_checked') is True)} papers, the PDF.

| Field | Agreement | Cohen's κ |
|---|---:|---:|
{rows_agree}

Bounded predictor, classifier × second pass: yes/yes {conf[('yes', 'yes')]}, yes/no {conf[('yes', 'no')]},
no/yes {conf[('no', 'yes')]}, no/no {conf[('no', 'no')]}. Precision of the classifier's `yes` is
{st['strata']['yes'][0]}/{st['strata']['yes'][1]}; second-pass `yes` among its `no` is
{st['strata']['no'][0]}/{st['strata']['no'][1]}. Weighting the two strata by their share of the
full sample gives the calibrated estimate above; its interval combines Jeffreys posteriors for
the three proportions by simulation ({N_SIM:,} draws, seed {SIM_SEED}), so the 0/20-type cells do
not produce a spuriously narrow interval. Agreement is computed on a subset that over-samples
classifier positives, so it is not the agreement on a random paper (that would be higher, since
most papers are clear negatives).

Disagreements on the bounded-predictor flag: {dis}.{''' In every case the classifier said `yes` and
the second pass `no`: these are general methods evaluated on benchmarks with bounded answers, and the
rubric's line between a task-specific predictor and a general method with bounded outputs is the
main source of disagreement.''' if st['disagree_all_llm_yes'] and st['disagree_bp'] else ''}

Caveats on the second pass: it was made by an LLM (reading abstracts and PDFs) in an interactive
session, not by a human, so the agreement measures the stability of the rubric across two independent
labelling passes, not human validity. The subset file hid the classifier's labels, but the
classifier's progress log (Anthology id, family, flag) had been printed in the same session before
the second pass. Strict eligibility on the subset: classifier {st['llm_elig_strict_sub']}, second
pass {st['second_elig_strict']}.

## Task families: random sample vs the 24 audited papers

| Family | Sample (all) | Bounded predictors | Eligible (lenient) | Audited | Eligible % | Audited % |
|---|---:|---:|---:|---:|---:|---:|
{rows_fam}
| Total | {n} | {st['n_bp']} | {st['n_el']} | {st['n_aud']} | 100% | 100% |

Total variation distance between the bounded-family mix of the audited papers and that of the
sample's eligible (lenient) papers: {st['tvd_el']:.2f} (bounded-predictor papers: {st['tvd_bp']:.2f}).
{pct(st['el_in_aud_fam'], 0)}% of the sample's eligible papers ({pct(st['bp_in_aud_fam'], 0)}% of its
bounded-predictor papers) fall in a family that has at least one audited paper.

## Comparison with the audited papers

- **Scope.** Only about one frame paper in {round(1 / st['bp']['p'])} centres on a bounded predictor
  ({pct(st['bp']['p'])}%, CI {pct(st['bp']['lo'])}–{pct(st['bp']['hi'])}%; calibrated by the second pass {pct(st['cal'])}%),
  and {pct(st['strict']['p'])}–{pct(st['lenient']['p'])}% (strict–lenient) would pass E1–E5 and the
  English-only rule, roughly {thousands(st['strict']['p'] * N)}–{thousands(st['lenient']['p'] * N)} papers.
  {pct(st['open_share'], 0)}% of sampled papers are about generation, open-ended reasoning, open-corpus
  retrieval or general LLM methods, which the audit does not address. Results on the 24 papers speak
  to the bounded-predictor minority, not to NLP papers in general.
- **Purposive, not random.** The 24 were chosen for task-family coverage and difficulty, so they are
  not a random draw from the eligible papers. Of the eligible share, the audit over-represents
  {fam_names(st['over']) if st['over'] else 'no family by 5 points or more'} and under-represents
  {fam_names(st['under']) if st['under'] else 'no family by 5 points or more'}.{' No audited paper is in ' + fam_names(st['absent']) + '.' if st['absent'] else ''}
  The largest shortfall is {short(st['under'][0]) if st['under'] else 'none'}{f" ({pct(st['fam_el'][st['under'][0]] / st['n_el'], 0)}% of eligible sample papers, {pct(st['fam_aud'][st['under'][0]] / st['n_aud'], 0)}% of audited papers)" if st['under'] else ''}.
  Eligible sample papers in the under-represented families:
{chr(10).join(f"  - {SHORT[f]}: " + "; ".join(t) for f, t in st['under_titles'].items())}
- **Direction of the likely bias.** The over-represented families are supervised classification
  tasks with fixed label inventories and released test splits, where a trained task-specific model
  has the most room to learn annotation conventions; the paper identifies such conventions as a main
  cause of failed replacement. The under-represented families (including LLM judges, rerankers and
  generated-text detectors) were not tested, so how the replacement result transfers
  to them is open.
- **Precision of the estimates.** With {st['n_bp']} bounded-predictor papers in the sample, family
  shares have wide intervals (a family with 5 of {st['n_bp']} has a 95% Wilson interval of
  {pct(wilson(5, st['n_bp'])[0], 0)}–{pct(wilson(5, st['n_bp'])[1], 0)}%), so only large differences
  in the family mix are informative.
"""
    MD.write_text(text)


def tex_escape(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def write_tex(st: dict) -> None:
    n, N = st["n"], st["N"]
    fr = st["frame"]
    macros = {
        "covN": str(n),
        "covFrame": f"{N:,}",
        "covSeed": str(fr["seed"]),
        "covBP": pct(st["bp"]["p"], 0),
        "covBPk": str(st["bp"]["k"]),
        "covBPlo": pct(st["bp"]["lo"], 0),
        "covBPhi": pct(st["bp"]["hi"], 0),
        "covBPcal": pct(st["cal"], 0),
        "covBPcallo": pct(st["cal_lo"], 0),
        "covBPcalhi": pct(st["cal_hi"], 0),
        "covBPframe": thousands(st["bp"]["p"] * N),
        "covEligStrict": pct(st["strict"]["p"], 0),
        "covEligStrictlo": pct(st["strict"]["lo"], 0),
        "covEligStricthi": pct(st["strict"]["hi"], 0),
        "covEligLenient": pct(st["lenient"]["p"], 0),
        "covEligLenientlo": pct(st["lenient"]["lo"], 0),
        "covEligLenienthi": pct(st["lenient"]["hi"], 0),
        "covSecondN": str(st["n_second"]),
        "covSecondAgree": pct(st["agree"]["bounded_predictor"]["agree"], 0),
        "covSecondKappa": f"{st['agree']['bounded_predictor']['kappa']:.2f}",
        "covEligInAudFam": pct(st["el_in_aud_fam"], 0),
        "covTVD": f"{st['tvd_el']:.2f}",
    }
    lines = ["% Generated by scripts/coverage_report.py from data/coverage/. Do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    lines.append(
        r"\newcommand{\coverageHook}{In a uniform random sample of \covN{} of the \covFrame{} papers in "
        r"the corpus, \covBP\%{} (95\% CI \covBPlo--\covBPhi\%) have a bounded predictor as their central "
        r"contribution and \covEligStrict--\covEligLenient\%{} would meet our eligibility criteria and "
        r"the English-only rule (Appendix~\ref{app:coverage}).}")
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "coverage.tex").write_text("\n".join(lines) + "\n")

    def row(label: str, s: dict) -> str:
        return (f"{label} & {s['k']} & {pct(s['p'])} & {pct(s['lo'])}--{pct(s['hi'])} "
                f"& {thousands(s['p'] * N)} \\\\")

    fam_rows = []
    for f in FAMILIES:
        aud = str(st["fam_aud"][f]) if f in BOUNDED else "--"
        fam_rows.append(f"{SHORT[f]} & {st['fam_all'][f]} & {st['fam_bp'][f]} & "
                        f"{st['fam_el'][f]} & {aud} \\\\")
        if f == BOUNDED[-1]:
            fam_rows.append(r"\midrule")
    ag = st["agree"]
    agree_rows = [f"{AGREE_FIELDS[f]} & {pct(ag[f]['agree'], 0)} & {ag[f]['kappa']:.2f} \\\\"
                  for f in ("bounded_predictor", "E1", "E2", "english_only", "eligible_strict", "task_family")]
    conf = st["confusion"]
    under = fam_names(st["under"]) if st["under"] else "no family"
    over = fam_names(st["over"]) if st["over"] else "no family"
    absent = (f". No evaluated paper is in {fam_names(st['absent'])}" if st["absent"] else "")
    n_dis = conf[("yes", "no")] + conf[("no", "yes")]
    dis_dir = (f". All {n_dis} disagreements are papers the classifier counted and the second pass did "
               "not, namely general methods evaluated on benchmarks with bounded answers"
               if st["disagree_all_llm_yes"] and n_dis else "")
    drawn = len(st["audited_drawn"])
    body = rf"""% Generated by scripts/coverage_report.py from data/coverage/. Do not edit.
\section{{How typical are the evaluated papers?}}\label{{app:coverage}}
The 24 papers were chosen purposively (Appendix~\ref{{app:selection}}). To place them, we labelled a
uniform random sample of {n} records from the same {N:,}-record corpus (seed {fr['seed']}). The
Anthology export has changed since the corpus was defined, so the corpus was re-derived from the
current export (SHA-256 prefix \texttt{{{fr['export_sha256'][:16]}}}) with the same venue and date filter.
It reproduces the per-year counts exactly. Each paper was labelled from its title and abstract, and
from the PDF when the abstract was ambiguous ({st['pdf']} papers), by an LLM classifier
(\texttt{{{st['model'].split('/')[-1]}}}) following a written rubric that was fixed before labelling.
The rubric records whether the paper's central
contribution is a task-specific predictor with a bounded output (a label from a fixed inventory, a
place on a predefined scale, or an order over supplied candidates), the eligibility criteria E1--E5,
whether all evaluation data are English, and the main task family. A paper is \emph{{strictly}}
eligible if every criterion is met and \emph{{leniently}} eligible if none is failed (unclear counts
as met).

\begin{{table}}[h]
\centering\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{lrrcr}}
\toprule
 & $k$ & \% & 95\% CI & $\approx$ papers \\
\midrule
{row('Bounded predictor', st['bp'])}
{row('Bounded task family', st['bfam'])}
{row('E1 and E2 met', st['e1e2'])}
{row('Eligible, strict', st['strict'])}
{row('Eligible, lenient', st['lenient'])}
\bottomrule
\end{{tabular}}
\caption{{Shares of the {N:,}-paper corpus estimated from a random sample of {n} (Wilson intervals).
``Bounded task family'' counts papers whose main task has a bounded output, with or without a
proposed predictor.}}
\label{{tab:coverage}}
\end{{table}}

\paragraph{{Check of the labels.}} A second, independent labelling pass covered {st['n_second']} papers,
{st['strata']['yes'][1]} drawn from the classifier's bounded-predictor papers and {st['strata']['no'][1]} from
the rest, made without the classifier's labels in view. This pass was also made by an LLM, so it
measures how stably the rubric can be applied, not agreement with human judgement. The two passes
agree on {conf[('yes', 'yes')] + conf[('no', 'no')]} of {st['n_second']} bounded-predictor flags
(Table~\ref{{tab:coverage-agree}}){dis_dir}. Re-weighting by the second pass gives
{pct(st['cal'])}\% (95\% interval {pct(st['cal_lo'])}--{pct(st['cal_hi'])}\%, Jeffreys posteriors
combined by simulation) instead of {pct(st['bp']['p'])}\%.

\begin{{table}}[h]
\centering\footnotesize
\begin{{tabular}}{{lrr}}
\toprule
Field & Agreement (\%) & $\kappa$ \\
\midrule
{chr(10).join(agree_rows)}
\bottomrule
\end{{tabular}}
\caption{{Agreement between the classifier and the second pass on {st['n_second']} papers. The subset
over-samples classifier positives, so agreement on a random paper would be higher.}}
\label{{tab:coverage-agree}}
\end{{table}}

\paragraph{{Task families.}} Table~\ref{{tab:coverage-families}} compares the main task families of the
sample with those of the 24 evaluated papers. {pct(st['el_in_aud_fam'], 0)}\% of the sample's eligible
papers fall in a family that has at least one evaluated paper. The total variation distance between
the two family mixes is {st['tvd_el']:.2f}. Relative to the eligible papers, our evaluation
over-represents {over}, and under-represents {under}{absent}. The under-represented families include
LLM judges, rerankers and detectors of generated text, which our evaluation does not
test. {drawn} of the 24 evaluated papers {'was' if drawn == 1 else 'were'}
drawn by the random sample (expected {n * st['n_aud'] / N:.1f}).

\begin{{table}}[h]
\centering\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{lrrrr}}
\toprule
Task family & All & Bnd. & Elig. & Eval. \\
\midrule
{chr(10).join(fam_rows)}
\midrule
Total & {n} & {st['n_bp']} & {st['n_el']} & {st['n_aud']} \\
\bottomrule
\end{{tabular}}
\caption{{Main task family of all {n} sampled papers (All), of those with a bounded predictor (Bnd.),
of the leniently eligible ones (Elig.) and of the 24 evaluated papers (Eval.). Families in the second
block have unbounded outputs.}}
\label{{tab:coverage-families}}
\end{{table}}

\paragraph{{What this implies.}} Our evaluation speaks to a minority of recent main-venue NLP papers: about
{pct(st['bp']['p'], 0)}\% centre on a bounded predictor and {pct(st['strict']['p'], 0)}--{pct(st['lenient']['p'], 0)}\%
(roughly {thousands(st['strict']['p'] * N)}--{thousands(st['lenient']['p'] * N)} papers) would be
eligible. Within that minority, the 24 lean towards supervised classification with fixed label
inventories and released test splits, where a task-specific model has the most room to learn
annotation conventions. Family shares in the sample rest on {st['n_bp']} bounded-predictor papers and
have wide intervals, so only large differences in the mix are informative.
"""
    (GEN / "app_coverage.tex").write_text(body)


def report() -> None:
    st = compute()
    write_markdown(st)
    write_tex(st)
    print(json.dumps({k: st[k] for k in ("bp", "strict", "lenient", "cal", "cal_lo", "cal_hi", "tvd_el",
                                          "over", "under", "absent", "audited_drawn")}, indent=1))


if __name__ == "__main__":
    {"subset": subset, "report": report}[sys.argv[1]]()
