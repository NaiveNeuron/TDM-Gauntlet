#!/usr/bin/env python3
"""Does each winning feature-question-search program state a dataset convention
absent from the paper's original question?

Offline only: reads the saved programs (runs/jev-opt/<run>/*.yaml) and the seed
questions; no Jev calls, no test data.  `--judge` runs the omp model once per
program and caches raw outputs in results/framing_conventions.json; the default
mode reassembles the JSON from those cached outputs plus the SECOND_READING
verdicts below (the second reading gives the final verdict).

The 5 frozen-configuration programs (fz-<set>-P, the representative chosen in
results/frozen.json) are classified with the same rubric,
prompt and procedure (`--judge-frozen`, then FROZEN_SECOND_READING) and reported under
"frozen"; they do not enter the 24-program count.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "framing_conventions.json"
EXP = ROOT / "jev-opt" / "experiments"

RUBRIC = (
    "A program *states a dataset convention absent from the paper's original question* iff at least "
    "one of its question texts states a rule about label boundaries, defaults, or annotation practice "
    "(e.g. which label borderline/ambiguous cases go to, what counts as a label beyond its definition, "
    "a default label when cues are absent, how annotators treated some surface pattern, a rating level's "
    "typical properties such as length/structure for essay scores) that is NOT stated or directly implied "
    "by the seed prompt's text. Rephrasings, compressions or synonyms of seed definitions do NOT count; "
    "generic task instructions (\"choose one\", \"read carefully\") do NOT count; auxiliary questions that "
    "only measure an observable property without tying it to a label do NOT count by themselves. For each "
    "positive, extract the shortest verbatim span (exact substring of the program YAML text value) that "
    "states the convention, and name the seed text it goes beyond."
)

# task slug -> (experiment dir, seed file, task label-definition file)
TASKS = {
    "contempo-tbdense": ("contempo_tbdense", "seed.yaml", "task_v3.yaml"),
    "ezstance-subtaskA": ("ezstance_subtaskA", "seed.yaml", "task_v3.yaml"),
    "smarter-implicit-hate": ("smarter_implicit_hate", "seed.yaml", "task_v3.yaml"),
    "edit-intent": ("edit_intent", "seed.yaml", "task_v3.yaml"),
    "inli": ("inli", "seed.yaml", "task_v3.yaml"),
    "dress-new-content": ("dress_new", "seed_content.yaml", "task_v3_content.yaml"),
    "dress-new-organization": ("dress_new", "seed_organization.yaml", "task_v3_organization.yaml"),
    "dress-new-language": ("dress_new", "seed_language.yaml", "task_v3_language.yaml"),
}

SKIP_KEYS = {"kind", "state", "readout", "labels", "inputs", "version", "temperature", "root"}
# `omp config list` at judge time: modelRoles.default
JUDGE_MODEL = "omp default (modelRoles.default = anthropic/claude-opus-5-5:high)"

# (task, seed) -> (final verdict, note, final quote, quote node, seed text it goes beyond).
# A second reading of every program against its seed, after the judge; its verdict is final.
SIMUL = "SIMULTANEOUS: Event 1 and Event 2 occupy the same time interval; they happen at the same time."
SECOND_READING: dict[tuple[str, int], tuple[bool, str, str, str, str]] = {
    ("contempo-tbdense", 0): (True, "Agree with judge: SIMULTANEOUS narrowed from 'same time interval' to the same happening (coreference).",
                              "is the same happening as", "relation", SIMUL),
    ("contempo-tbdense", 1): (True, "Agree with judge: SIMULTANEOUS narrowed to both triggers describing the same occurrence.",
                              "both triggers to describe the same occurrence", "relation", SIMUL),
    ("contempo-tbdense", 2): (False, "Agree: 80% rule, negated/hypothetical, generic and aspect/manner clauses are all in the seed guidelines.", "", "", ""),
    ("ezstance-subtaskA", 0): (True, "Agree: claim-target prototypes describe how targets were constructed (flipped detail = AGAINST, extrapolation = NONE).",
                               "the target is the tweet with one key detail flipped", "stance",
                               "The author is definitely against the point or message from the claim."),
    ("ezstance-subtaskA", 1): (True, "Agree: boundary rule excluding criticism of third parties; the claim node also maps altered details to AGAINST.",
                               "Criticism of third parties is not stance on the target.", "stance_np",
                               "The author of the tweet is against the target."),
    ("ezstance-subtaskA", 2): (True, "Agree: default NONE unless a citable attitude phrase exists; the seed's NONE is only 'neutral toward the target'.",
                               "Assign FAVOR or AGAINST only when you could cite a phrase showing the author's attitude toward `target`. Otherwise assign NONE.",
                               "stance_np", "The author of the tweet is neutral toward the target."),
    ("smarter-implicit-hate", 0): (True, "Agree: explicit annotator routing of surface patterns (retweets, hashtags) to Incitement; seed defines Incitement as flaunting unity/elevating known hate groups.",
                                   "Retweets of movement content, nationalist hashtags or leaders go to Incitement", "label",
                                   "Incitement to Violence: includes flaunting in group unity and power or elevating known hate groups and ideologies"),
    ("smarter-implicit-hate", 1): (False, "Disagree with judge: 'wronged by minorities, elites or policies' is a paraphrase of the seed's majority-as-victims grievance; no boundary rule.", "", "", ""),
    ("smarter-implicit-hate", 2): (True, "Agree (borderline): Incitement extended to supporting/recruiting for nationalist figures, slogans and bravado, beyond 'known hate groups'.",
                                   "supports or recruits for nationalist figures", "label_x3",
                                   "Incitement to Violence: includes flaunting in group unity and power or elevating known hate groups and ideologies"),
    ("edit-intent", 0): (True, "Agree: headings/caption stubs/formatting fragments routed to Other for added/deleted sentences.",
                         "a heading, a caption stub or a formatting fragment", "edit_intent_add_delete",
                         "Other: Revise the text in a way that is irrelevant to any other type of edit intent."),
    ("edit-intent", 1): (True, "Agree: non-sentences default to Other (also: study aims and procedure purposes go to Fact/Evidence).",
                         "if it is not a full sentence, answer Other", "edit_intent_add_delete",
                         "Other: Revise the text in a way that is irrelevant to any other type of edit intent."),
    ("edit-intent", 2): (True, "Agree: borderline Grammar-vs-Fact rule for edits that also change a reference number.",
                         "even when a reference number also changes", "edit_intent_x2",
                         "Surface, fix grammar (Grammar): Correct grammatical errors, capitalization, punctuation, tense, modality, spelling, typography, abbreviations or any errors related to grammar and/or conventions to improve the language."),
    ("inli", 0): (False, "Agree: explicit/implied restated from the seed (lexical semantics, co-reference vs pragmatics, world knowledge, figurative).", "", "", ""),
    ("inli", 1): (False, "Agree: label texts compress the seed; auxiliary questions only measure overlap/indirect replies.", "", "", ""),
    ("inli", 2): (False, "Disagree with judge: contradiction of the conveyed meaning is directly implied by the seed's 'contradicted by the passage' plus conversational pragmatics.", "", "", ""),
    ("dress-new-content", 0): (True, "Agree: low levels tied to length/structure (one paragraph, two short paragraphs), high to full structure.",
                               "Two short paragraphs", "content",
                               "Rubric explanation for content: \"Paragraph is well-developed and relevant to the argument, supported with strong reasons and examples.\""),
    ("dress-new-content", 1): (False, "Disagree with judge: bands (missing/partly developed/developed) interpolate the seed's 'how fully the essay meets this explanation'; ignoring language is implied by rubric separation.", "", "", ""),
    ("dress-new-content", 2): (True, "Agree: states the grader's score distribution (and that long essays with errors are not low).",
                               "Most essays fall in 3.0-4.5.", "content",
                               "Score only the content rubric: how fully the essay meets this explanation."),
    ("dress-new-organization", 0): (False, "Disagree with judge: level statements are graded restatements of the organization rubric (structure, development, coherence devices).", "", "", ""),
    ("dress-new-organization", 1): (True, "Agree: states the grader's score distribution.",
                                    "Typical scores are 3.0-4.5; 1.0-2.0 and 5.0 are rare.", "organization_x2",
                                    "Score only the organization rubric: how fully the essay meets this explanation."),
    ("dress-new-organization", 2): (True, "Agree: a five-part essay with thin reasons is a mid level, i.e. standard structure alone is not scored high.",
                                    "five-part shape, thin reasons, weak ending", "organization",
                                    "The argument is very effectively structured and developed, making it easy for the reader to follow the ideas"),
    ("dress-new-language", 0): (False, "Agree: comparison procedure plus an unlinked article-error measurement.", "", "", ""),
    ("dress-new-language", 1): (True, "Agree: 2.5/3.0/3.5 boundary set by the share of sentences with errors.",
                                "decide it by what share of sentences contain an error (most, about half, or a minority)", "language",
                                "The essay follows grammar and usage rules throughout the paper."),
    ("dress-new-language", 2): (True, "Agree: states the grader's leniency and score distribution.",
                                "Grades are lenient: the median is 3.5, 4.0 is as common as 3.5, and scores below 2.5 are rare.", "language",
                                "Score only the language rubric: how fully the essay meets this explanation."),
}

# Frozen-configuration programs, same procedure: a second reading against the seed after the judge.
FROZEN_SECOND_READING: dict[tuple[str, int], tuple[bool, str, str, str, str]] = {
    ("contempo-matres", 0): (True, "Agree: annotation conventions for surface patterns (mention order alone is not BEFORE; a speech act and its wording, when/as clauses and two views of one event are EQUAL; parallel or background clauses are VAGUE).",
                             "one speech act with its wording", "relation",
                             "EQUAL: Event 1 and Event 2 start at the same time. Neither start time can be before the other (A1=no, A2=no)."),
    ("lifetox", 0): (True, "Agree: the label is decided from the request alone, while the seed labels the advice in the context of the question.",
                     "From the request alone", "safety",
                     "Judge the answer in the context of the question it responds to"),
    ("sciprompt-s2orc", 0): (True, "Agree: boundary rule widening Geography, Sociology and History; the topic lists also route other fields' topics (central bank interventions, bioactive compounds, organizational behavior) to a field.",
                             "Underpredicted fields like Geography, Sociology and History are broader than they seem.", "topic",
                             "The field names are Semantic Scholar fields of study; the abstract has exactly one of them."),
    ("smarter-hatexplain", 0): (True, "Agree: states the data's typical content (Gab posts carry ideological hostility without swear words); the stacked view also routes identity slurs to Hate, which the seed rules out ('not the presence of certain words').",
                                "Gab posts often carry ideological hostility toward Jews, Muslims, Black people or immigrants even without swear words.", "label",
                                "Hate Speech: language which attacks, demeans, offends, threatens, or insults a group"),
    ("smarter-latent-hate", 0): (False, "Agree: label texts compress the seed definitions; the auxiliary questions only measure observable properties.", "", "", ""),
}


def question_texts(program: dict) -> list[tuple[str, str]]:
    """(node id, text) for every string under each question node (skips readouts/state)."""
    out: list[tuple[str, str]] = []

    def walk(node_id: str, obj) -> None:
        if isinstance(obj, str):
            out.append((node_id, obj))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if k not in SKIP_KEYS:
                    walk(node_id, v)

    for node_id, node in (program.get("nodes") or {}).items():
        walk(node_id, node)
    return out


def render(texts: list[tuple[str, str]]) -> str:
    return "\n".join(f"[{nid}] {t}" for nid, t in texts)


def programs():
    sel = json.loads((ROOT / "results" / "seeds.json").read_text())["selection"]
    for task, (exp, seed_file, _task_file) in TASKS.items():
        exp_seed = yaml.safe_load((EXP / exp / seed_file).read_text())
        for s in ("0", "1", "2"):
            e = sel[task][s]
            run_dir = ROOT / "runs" / "jev-opt" / e["run"]
            seed = yaml.safe_load((run_dir / "seed.yaml").read_text())
            assert seed == exp_seed, f"{e['run']}/seed.yaml differs from {exp}/{seed_file}"
            prog = yaml.safe_load((run_dir / f"{e['representative']}.yaml").read_text())
            yield task, int(s), e, question_texts(seed), question_texts(prog)


def frozen_programs():
    """The 5 frozen-configuration programs (seed 0 only): fz-<set>-P, representative chosen on val."""
    sel = json.loads((ROOT / "results" / "frozen.json").read_text())["selection"]
    for task, e in sel.items():
        run_dir = ROOT / "runs" / "jev-opt" / e["run"]
        seed = yaml.safe_load((run_dir / "seed.yaml").read_text())
        prog = yaml.safe_load((run_dir / f"{e['representative']}.yaml").read_text())
        yield task, 0, e, question_texts(seed), question_texts(prog)



def prompt(seed_texts, prog_texts) -> str:
    return (
        "You are auditing a classification prompt. Apply this rubric exactly.\n\nRUBRIC:\n" + RUBRIC
        + "\n\nSEED (paper's original question) TEXTS, one per line with [node id]:\n" + render(seed_texts)
        + "\n\nPROGRAM QUESTION TEXTS, one per line with [node id]:\n" + render(prog_texts)
        + "\n\nAnswer with ONLY a JSON object, no prose, no code fences: "
        '{"convention": true/false, "quote": "shortest verbatim span from a PROGRAM text or empty", '
        '"quote_node": "node id or empty", "rule": "one-line paraphrase", '
        '"not_in_seed_because": "which seed text it goes beyond and why it is not stated/implied there"}'
    )


def parse(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    return json.loads(m.group(0)) if m else {}


def run_judge(items) -> dict:
    cache = {}
    for task, s, e, st, pt in items:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            f.write(prompt(st, pt))
        r = subprocess.run(["omp", "-p", "--no-session", f"@{f.name}"], stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=900)
        cache[f"{task}/{s}"] = r.stdout.strip()
        print(task, s, parse(r.stdout).get("convention"), file=sys.stderr)
    return cache


def entries(items, raw: dict, second: dict) -> list[dict]:
    rows = []
    for task, s, e, st, pt in items:
        r = raw.get(f"{task}/{s}", "")
        j = parse(r)
        full = "\n".join(t for _, t in pt)
        entry = {"task": task, "seed": s, "run": e["run"], "representative": e["representative"],
                 "candidate": e["candidate"],
                 "judge": {"convention": bool(j.get("convention")), "quote": j.get("quote", ""),
                           "rule": j.get("rule", ""), "raw": r}}
        if (task, s) in second:
            m, note, quote, node, seed_text = second[(task, s)]
            texts = [t for nid, t in pt if nid == node]
            assert (not m and not quote) or any(quote in t for t in texts), (task, s, quote)
            assert not m or any(seed_text in t for _, t in st), (task, s, seed_text)
            entry.update(final_verdict=m, note=note, quote=quote, quote_node=node, seed_text=seed_text,
                         rule=j.get("rule", "") if m else "")
        else:
            entry.update(final_verdict=None, note="", quote="", quote_node="", seed_text="", rule="")
        if entry["judge"]["quote"]:
            entry["judge"]["quote_verbatim"] = entry["judge"]["quote"] in full
        rows.append(entry)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--judge-frozen", action="store_true")
    a = ap.parse_args()
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    if a.judge:
        raw = run_judge(programs())
    else:
        raw = {f"{p['task']}/{p['seed']}": p["judge"]["raw"] for p in old.get("programs", [])}
    if a.judge_frozen:
        raw_fz = run_judge(frozen_programs())
    else:
        raw_fz = {f"{p['task']}/{p['seed']}": p["judge"]["raw"] for p in old.get("frozen", {}).get("programs", [])}
    rows = entries(programs(), raw, SECOND_READING)
    fz_rows = entries(frozen_programs(), raw_fz, FROZEN_SECOND_READING)
    done = [r for r in rows if r["final_verdict"] is not None]
    pos = [r for r in done if r["final_verdict"]]
    out = {
        "rubric": RUBRIC,
        "judge_model": JUDGE_MODEL,
        "programs": rows,
        "count": len(pos),
        "total": len(rows),
        "tasks_with_convention": len({r["task"] for r in pos}),
        "judge_agreement": sum(r["final_verdict"] == r["judge"]["convention"] for r in done),
        "examples": [{"task": r["task"], "seed": r["seed"], "quote": r["quote"], "rule": r["rule"]}
                     for r in (next(p for p in pos if (p["task"], p["seed"]) == k) for k in EXAMPLES)],
        "frozen": {
            "programs": fz_rows,
            "count": sum(bool(r["final_verdict"]) for r in fz_rows),
            "total": len(fz_rows),
            "judge_agreement": sum(r["final_verdict"] == r["judge"]["convention"] for r in fz_rows if r["final_verdict"] is not None),
        },
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")


EXAMPLES = [("smarter-implicit-hate", 0), ("edit-intent", 2), ("dress-new-language", 2)]

if __name__ == "__main__":
    main()
