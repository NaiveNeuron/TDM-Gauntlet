#!/usr/bin/env python3
"""Convention spans and the pre-specified edits of the convention ablation -> results/conventions_spans.json.

Offline: reads the saved programs only, no Jev call.

Units: every program whose final verdict in results/framing_conventions.json is positive (the 16 of the 24
seed-0/1/2 programs, plus the frozen-configuration programs classified positive with the same rubric).

SPANS lists, per program, every verbatim substring of a question text that states a dataset convention by
the fixed rubric (scripts/framing_conventions.py RUBRIC). A span is a whole sentence or a whole option
text when all of its content is the convention; when a sentence or option text also restates the seed's
definition or gives a generic instruction, the span is the convention clause alone, cut at punctuation or
a conjunction so that the rest still reads as text. Option texts that only point back to a convention
stated elsewhere in the same program are spans too (they are deleted in cell B) but are not appended in
cell D (`d=False`): the self-contained span carries the rule there.

Field syntax: "instructions", "condition", "option:<key>", "level:<index>".
D target: (seed node, field) of the paper's question; a span that routes cases to one label goes to that
label's option definition (Score seeds: the level of that label), every other span to the instructions.

Cell edits (all mechanical, from this file):
  B   delete every span from its field, then tidy that field only (collapse runs of whitespace, drop a
      space before . , ; : ? !, strip). An emptied field stays empty (Jev still sees the option key).
  C   delete non-convention text of the same total length from the same program: units = sentences of
      instructions/conditions and option texts (split after . ? ! + whitespace) that do not overlap a
      span; Score levels are never units. Choose the subset of units whose total length is closest to the
      span length (ties: fewer units, then earliest units), first among units of the nodes that carry a
      span; if that best subset misses the span length by more than 10%, among all units. Same tidying.
  D   the paper's question (seed.yaml) with each `d=True` span appended, in program order, to its target
      field: standalone form = span stripped of surrounding whitespace and , ; : and of a leading
      "or"/"and" or step number ("3)", "Step 1:") and a trailing "with"/"but"/"and"/"or"; first letter
      upper-cased; "." added if it does not end in . ? !. Separator: " " if the field ends in . ? ! :,
      else ". ".
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "conventions_spans.json"
RUNS = ROOT / "runs" / "jev-opt"

# (task, seed) -> list of (node, field, span, d, d_node, d_field, note)
I, C = "instructions", "condition"


def O(key: str) -> str:
    return f"option:{key}"


SPANS: dict[tuple[str, int], list[tuple[str, str, str, bool, str, str, str]]] = {
    ("contempo-tbdense", 0): [
        ("relation", O("SIMULTANEOUS"), "The <e2> event is the same happening as the <e1> event.", True, "relation", O("SIMULTANEOUS"),
         "SIMULTANEOUS narrowed to coreference (same happening)"),
    ],
    ("contempo-tbdense", 1): [
        ("relation", O("SIMULTANEOUS"), "A reader would take both triggers to describe the same occurrence.", True, "relation", O("SIMULTANEOUS"),
         "SIMULTANEOUS narrowed to coreference (same occurrence)"),
    ],
    ("ezstance-subtaskA", 0): [
        ("stance", O("AGAINST"), "Prototype: the target is the tweet with one key detail flipped, such as the actor, the vote direction, the country or the claimed inevitability.",
         True, "stance", O("AGAINST"), "how claim targets were built: flipped detail = AGAINST"),
        ("stance", O("FAVOR"), "Prototype: the target is a compressed restatement of the tweet, possibly with neutral rewording.",
         True, "stance", O("FAVOR"), "restatement = FAVOR"),
        ("stance", O("NONE"), 'Prototype: the target is an embellished extrapolation, such as a future outcome, a policy proposal or a "potentially highlighting" editorial.',
         True, "stance", O("NONE"), "extrapolation = NONE"),
    ],
    ("ezstance-subtaskA", 1): [
        ("stance_np", I, "Criticism of third parties is not stance on the target.", True, "stance_np", I, "boundary rule"),
        ("stance", I, "If a detail is altered, swapped, exaggerated or invented, pick AGAINST.", True, "stance", O("AGAINST"), "altered detail = AGAINST"),
        ("stance", I, "3) If every detail is backed by the tweet and the target states its core message, pick FAVOR.", True, "stance", O("FAVOR"), "supported restatement = FAVOR"),
        ("stance", I, "4) If the target goes beyond what the tweet commits to, pick NONE. This includes added causes, generalisations, or a claim the author only hedges on.",
         True, "stance", O("NONE"), "extrapolation = NONE"),
        ("stance", O("AGAINST"), "At least one target detail contradicts or distorts the tweet.", False, "", "", "restates the AGAINST step"),
        ("stance", O("FAVOR"), "All target details are supported", False, "", "", "restates the FAVOR step"),
        ("stance", O("NONE"), "The target adds unsupported content or overstates what the tweet commits to.", False, "", "", "restates the NONE step"),
    ],
    ("ezstance-subtaskA", 2): [
        ("stance_np", I, "Assign FAVOR or AGAINST only when you could cite a phrase showing the author's attitude toward `target`. Otherwise assign NONE.",
         True, "stance_np", I, "default NONE unless a citable attitude phrase exists"),
        ("stance_np", O("AGAINST"), "A citable phrase shows ", False, "", "", "points back to the citable-phrase rule"),
        ("stance_np", O("FAVOR"), "A citable phrase shows ", False, "", "", "points back to the citable-phrase rule"),
        ("stance_np", O("NONE"), "No such phrase exists.", False, "", "", "points back to the citable-phrase rule"),
    ],
    ("smarter-implicit-hate", 0): [
        ("label", I, "Retweets of movement content, nationalist hashtags or leaders go to Incitement.", True, "label", O("Incitement to Violence"),
         "annotators routed movement retweets, hashtags and leaders to Incitement"),
        ("label", O("Incitement to Violence"), "Movement content", False, "", "", "points back to the routing rule"),
        ("label_x1", O("Incitement to Violence"), " or figures, rallies, movement news, or pro-border cheering", True, "label", O("Incitement to Violence"),
         "Incitement extended beyond known hate groups"),
    ],
    ("smarter-implicit-hate", 2): [
        ("label_x3", O("Incitement to Violence"), "The post cheers, supports or recruits for nationalist figures, groups, slogans or fighting spirit, or shows hostile bravado.",
         True, "label", O("Incitement to Violence"), "Incitement extended to nationalist figures, slogans and bravado"),
        ("label_x2", O("Incitement to Violence"), "figures or ", False, "", "", "nationalist figures as Incitement (same extension)"),
    ],
    ("edit-intent", 0): [
        ("edit_intent_add_delete", O("Other"), "Prototype: a heading, a caption stub or a formatting fragment.", True, "edit_intent_add_delete", O("Other"),
         "headings, caption stubs and formatting fragments go to Other"),
    ],
    ("edit-intent", 1): [
        ("edit_intent", I, "purpose of a procedure, ", True, "edit_intent", O("Fact/Evidence"), "procedure purposes go to Fact/Evidence"),
        ("edit_intent", I, "for example a heading trimmed or a formatting artifact, ", True, "edit_intent", O("Other"), "trimmed headings and formatting artifacts go to Other"),
        ("edit_intent_add_delete", I, "Step 1: if it is not a full sentence, answer Other.", True, "edit_intent_add_delete", O("Other"), "non-sentences default to Other"),
        ("edit_intent_add_delete", I, "a study aim or ", True, "edit_intent_add_delete", O("Fact/Evidence"), "study aims go to Fact/Evidence (the seed puts aims under Claim)"),
    ],
    ("edit-intent", 2): [
        ("edit_intent_x2", I, ", even when a reference number also changes", True, "edit_intent", O("Grammar"), "Grammar even when a reference number changes"),
    ],
    ("dress-new-content", 0): [
        ("content", O("1.0"), "One paragraph", True, "content", "level:0", "1.0 tied to length/structure"),
        ("content", O("1.5"), "One paragraph with ", True, "content", "level:1", "1.5 tied to length/structure"),
        ("content", O("2.0"), "Two short paragraphs", True, "content", "level:2", "2.0 tied to length/structure"),
        ("content", O("2.5"), "Several paragraphs but ", True, "content", "level:3", "2.5 tied to length/structure"),
        ("content", O("3.0"), "Intro", True, "content", "level:4", "3.0 tied to structure"),
        ("content", O("3.5"), "Full structure; ", True, "content", "level:5", "high levels tied to full structure"),
        ("content", O("4.0"), "Full structure; ", True, "content", "level:6", "high levels tied to full structure"),
        ("content", O("4.5"), "Full structure; ", True, "content", "level:7", "high levels tied to full structure"),
        ("content", O("5.0"), "Full structure; ", True, "content", "level:8", "high levels tied to full structure"),
    ],
    ("dress-new-content", 2): [
        ("content", I, "Most essays fall in 3.0-4.5.", True, "content", I, "grader's score distribution"),
        ("content", I, "Long essays with errors are not low.", True, "content", I, "long essays with errors are not scored low"),
    ],
    ("dress-new-organization", 1): [
        ("organization_x1", I, "Typical essays land in the middle-upper range; use extremes rarely.", True, "organization", I, "grader's score distribution"),
        ("organization_x2", I, "Typical scores are 3.0-4.5; 1.0-2.0 and 5.0 are rare.", True, "organization", I, "grader's score distribution"),
    ],
    ("dress-new-organization", 2): [
        ("organization", O("3.0"), "Prototype: five-part shape, thin reasons, weak ending.", True, "organization", "level:4",
         "standard five-part structure with thin reasons is a mid score"),
    ],
    ("dress-new-language", 1): [
        ("language", I, "The hardest decision is between 2.5, 3.0 and 3.5; decide it by what share of sentences contain an error (most, about half, or a minority), not by the opening paragraph.",
         True, "language", I, "2.5/3.0/3.5 set by the share of sentences with errors"),
        ("language", O("2.5"), "Errors in most sentences.", True, "language", "level:3", "2.5 = errors in most sentences"),
        ("language", O("3.0"), "Errors in about half of the sentences.", True, "language", "level:4", "3.0 = errors in about half"),
        ("language", O("3.5"), "Errors in a minority of sentences.", True, "language", "level:5", "3.5 = errors in a minority"),
    ],
    ("dress-new-language", 2): [
        ("language", I, "Grades are lenient: the median is 3.5, 4.0 is as common as 3.5, and scores below 2.5 are rare.", True, "language", I,
         "grader's leniency and score distribution"),
        ("language", I, "; weigh error frequency most", True, "language", I, "error frequency dominates the score"),
        ("language_x1", I, "; the grader is lenient", True, "language", I, "grader's leniency"),
    ],
    ("contempo-matres", 0): [
        ("relation", I, "Being mentioned first does not make an event earlier.", True, "relation", I, "mention order is not temporal order"),
        ("relation", O("BEFORE"), ", a report of a past act", True, "relation", O("BEFORE"), "a reported past act starts before the report"),
        ("relation", O("BEFORE"), "Mention order alone is not enough.", True, "relation", O("BEFORE"), "mention order is not BEFORE"),
        ("relation", O("AFTER"), ", or Event 1 reporting or reacting to Event 2", True, "relation", O("AFTER"), "a report/reaction starts after what it reports"),
        ("relation", O("EQUAL"), " or the same occurrence: a when/as clause, two views of one event, or one speech act with its wording", True, "relation", O("EQUAL"),
         "when/as clauses, two views of one event and a speech act with its wording are EQUAL"),
        ("relation", O("VAGUE"), "The two events are in parallel or background clauses, and ", True, "relation", O("VAGUE"), "parallel/background clauses are VAGUE"),
    ],
    ("lifetox", 0): [
        ("safety", I, "From the request alone, does fulfilling it typically require dishonesty, unfairness, objectification, or profiting from harm?", True, "safety", I,
         "the label follows the request, not the advice"),
    ],
    ("sciprompt-s2orc", 0): [
        ("topic", I, "Underpredicted fields like Geography, Sociology and History are broader than they seem.", True, "topic", I, "boundary rule widening three fields"),
        ("topic", O("Engineering"), ", inventive design methods", True, "topic", O("Engineering"), "design-method papers go to Engineering"),
        ("topic", O("Political Science"), ", central bank interventions", True, "topic", O("Political Science"), "central-bank papers go to Political Science"),
        ("topic", O("Geography"), ", transport and communication traffic, tourism", True, "topic", O("Geography"), "traffic and tourism papers go to Geography"),
        ("topic", O("Medicine"), ", bioactive compounds", True, "topic", O("Medicine"), "bioactive-compound papers go to Medicine"),
        ("topic", O("Psychology"), ", organizational behavior", True, "topic", O("Psychology"), "organizational behavior goes to Psychology"),
        ("topic", O("Mathematics"), ", quantitative counts", True, "topic", O("Mathematics"), "quantitative counting papers go to Mathematics"),
    ],
    ("smarter-hatexplain", 0): [
        ("label", I, "Gab posts often carry ideological hostility toward Jews, Muslims, Black people or immigrants even without swear words.", True, "label", I,
         "typical content of the data: ideological hostility without swear words"),
        ("label_x1", I, " or using an identity slur", True, "label", O("Hate"), "identity slurs go to Hate"),
        ("label_x1", O("Hate"), " or slur", False, "", "", "points back to the slur rule"),
    ],
}

# Texts considered and left out of the spans (borderline under the rubric), recorded for the authors'
# check: (node, field, verbatim substring, reason). They are never deleted by the control C either.
EXCLUDED: dict[tuple[str, int], list[tuple[str, str, str, str]]] = {
    ("ezstance-subtaskA", 0): [("stance_np", I, "If the author complains about problems caused by `target` or treats it as a burden, that is AGAINST.",
                                "directly implied by the seed's 'against the target'")],
    ("ezstance-subtaskA", 2): [("stance", O("AGAINST"), "False according to the source: the source says otherwise.",
                                "contradicting the claim is directly implied by the seed's AGAINST")],
    ("smarter-implicit-hate", 0): [("label", I, "These are tweets from white nationalist and anti-immigrant accounts.", "provenance, no label rule")],
    ("edit-intent", 0): [("edit_intent", O("Other"), "For organisation", "organisational edits plausibly fall under the seed's 'irrelevant to any other type'"),
                         ("edit_intent_x1", O("Other"), "For organisation", "as above")],
    ("edit-intent", 1): [("edit_intent_add_delete", I, "Step 3: if it gives an unsupported general statement or explanation, or the authors' own stance, answer Claim.",
                          "the seed's Claim covers statements")],
    ("dress-new-content", 2): [("content_x1", I, "Low (1.0-2.5) is short, unsupported or off-topic.", "short ~ not well-developed")],
    ("dress-new-organization", 1): [("organization", I, "Match the instructor's scale, not an absolute ideal.", "no rule stated")],
    ("smarter-hatexplain", 0): [("label", O("Hate"), "Ideological or explicit hostility toward a protected group", "group attack is the seed's Hate")],
}

ENUM = re.compile(r"^(\d\)|Step \d:)\s*")


def get(node: dict, field: str) -> str:
    if field in (I, C):
        return node[field]
    kind, key = field.split(":", 1)
    return node["options"][key]["criterion"] if kind == "option" else node["levels"][int(key)]


def put(node: dict, field: str, value: str) -> None:
    if field in (I, C):
        node[field] = value
        return
    kind, key = field.split(":", 1)
    if kind == "option":
        node["options"][key]["criterion"] = value
    else:
        node["levels"][int(key)] = value


def tidy(t: str) -> str:
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\s+([.,;:?!])", r"\1", t)
    return t.strip()


def standalone(span: str) -> str:
    t = span.strip().strip(",;:").strip()
    t = re.sub(r"^(or|and)\s+", "", t)
    t = ENUM.sub("", t)
    t = re.sub(r"\s+(with|but|and|or)$", "", t).strip().strip(",;:").strip()
    t = t[0].upper() + t[1:]
    return t if t[-1] in ".?!" else t + "."


def append(text: str, add: str) -> str:
    base = text.rstrip()
    return base + (" " if base[-1:] in ".?!:" else ". ") + add


def text_fields(prog: dict) -> list[tuple[str, str]]:
    """(node, field) of every question text, in program order (Score levels last per node)."""
    out = []
    for nid, node in prog["nodes"].items():
        kind = node.get("kind")
        if kind in ("choice", "one_vs_rest", "score"):
            out.append((nid, I))
        if kind in ("choice", "one_vs_rest"):
            out += [(nid, O(k)) for k in node["options"]]
        if kind == "branch":
            out.append((nid, C))
        if kind == "score":
            out += [(nid, f"level:{i}") for i in range(len(node["levels"]))]
    return out


def locate(prog: dict, spans) -> list[tuple[str, str, int, int]]:
    """Character ranges of the spans; each span must occur exactly once in its field."""
    out = []
    for node, field, span, *_ in spans:
        text = get(prog["nodes"][node], field)
        assert text.count(span) == 1, (node, field, span, text)
        i = text.index(span)
        out.append((node, field, i, i + len(span)))
    return out


def units(prog: dict, ranges) -> list[dict]:
    out = []
    for node, field in text_fields(prog):
        if field.startswith("level:"):
            continue
        text = get(prog["nodes"][node], field)
        for m in re.finditer(r"\S.*?(?:[.?!](?=\s+\S)|$)", text, re.S):
            s, e = m.start(), m.end()
            if any(n == node and f == field and s < re_ and rs < e for n, f, rs, re_ in ranges):
                continue
            out.append({"node": node, "field": field, "text": m.group(0), "start": s, "end": e})
    return out


def closest_subset(lengths: list[int], target: int) -> list[int]:
    """Indices whose total length is closest to target (ties: fewer items, then earliest indices)."""
    best: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, ())}
    for i, n in enumerate(lengths):
        for s, (k, idx) in list(best.items()):
            cand = (k + 1, idx + (i,))
            if s + n not in best or cand < best[s + n]:
                best[s + n] = cand
    s = min(best, key=lambda s: (abs(s - target), best[s][0], best[s][1]))
    return list(best[s][1])


def delete(prog: dict, ranges) -> dict:
    q = copy.deepcopy(prog)
    by_field: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for node, field, s, e in ranges:
        by_field.setdefault((node, field), []).append((s, e))
    for (node, field), rs in by_field.items():
        text = get(q["nodes"][node], field)
        for s, e in sorted(rs, reverse=True):
            text = text[:s] + text[e:]
        put(q["nodes"][node], field, tidy(text))
    return q


def seed_field(node: dict, field: str) -> str:
    """A D target 'option:<label>' names the label; resolve it to the seed option routing to that label."""
    if not field.startswith("option:"):
        return field
    label = field.split(":", 1)[1]
    keys = [k for k, o in node["options"].items() if ((o.get("to") or {}).get("label") or k) == label]
    assert len(keys) == 1, (label, list(node["options"]))
    return f"option:{keys[0]}"


def programs() -> list[dict]:
    fc = json.loads((ROOT / "results" / "framing_conventions.json").read_text())
    rows = [dict(p, group="seeds") for p in fc["programs"]] + [dict(p, group="frozen") for p in fc["frozen"]["programs"]]
    return [p for p in rows if p["final_verdict"]]


def build() -> dict:
    out = []
    for p in programs():
        key = (p["task"], p["seed"])
        spans = SPANS[key]
        run_dir = RUNS / p["run"]
        prog = yaml.safe_load((run_dir / f"{p['representative']}.yaml").read_text())
        seed = yaml.safe_load((run_dir / "seed.yaml").read_text())
        assert p["quote"] and any(p["quote"] in s[2] or s[2] in p["quote"] for s in spans), (key, p["quote"])
        ranges = locate(prog, spans)
        length = sum(len(s[2]) for s in spans)
        span_nodes = {s[0] for s in spans}
        excluded = EXCLUDED.get(key, [])
        us = units(prog, ranges + locate(prog, excluded))
        near = [u for u in us if u["node"] in span_nodes]
        pick = [near[i] for i in closest_subset([len(u["text"]) for u in near], length)] if near else []
        pool = "span nodes"
        if abs(sum(len(u["text"]) for u in pick) - length) > 0.10 * length:
            pick = [us[i] for i in closest_subset([len(u["text"]) for u in us], length)]
            pool = "all nodes"
        b = delete(prog, ranges)
        c = delete(prog, [(u["node"], u["field"], u["start"], u["end"]) for u in pick])
        d = copy.deepcopy(seed)
        appended = []
        for node, field, span, dflag, dnode, dfield, _ in spans:
            if not dflag:
                continue
            add = standalone(span)
            dfield = seed_field(d["nodes"][dnode], dfield)
            put(d["nodes"][dnode], dfield, append(get(d["nodes"][dnode], dfield), add))
            appended.append({"node": dnode, "field": dfield, "text": add})
        out.append({
            "program": f"{p['task']}/s{p['seed']}", "task": p["task"], "seed": p["seed"], "group": p["group"], "run": p["run"],
            "representative": p["representative"], "candidate": p["candidate"],
            "spans": [{"node": n, "field": f, "span": s, "start": r[2], "end": r[3], "append_in_D": dflag,
                       "D_target": {"node": dn, "field": seed_field(seed["nodes"][dn], df)} if dflag else None, "note": note}
                      for (n, f, s, dflag, dn, df, note), r in zip(spans, ranges)],
            "span_chars": length,
            "excluded_borderline": [{"node": n, "field": f, "text": t, "reason": r} for n, f, t, r in excluded],
            "control_C": {"pool": pool, "chars": sum(len(u["text"]) for u in pick),
                          "units": [{k: u[k] for k in ("node", "field", "text")} for u in pick]},
            "D_appended": appended,
            "edited": {
                "B": [{"node": n, "field": f, "text": get(b["nodes"][n], f)} for n, f in sorted({(r[0], r[1]) for r in ranges})],
                "C": [{"node": n, "field": f, "text": get(c["nodes"][n], f)} for n, f in sorted({(u["node"], u["field"]) for u in pick})],
            },
            "programs": {"B": b, "C": c, "D": d},
        })
    return {
        "generated_by": "scripts/conventions_spans.py",
        "status": ("Spans marked by the model that applied the rubric in framing_conventions.py and checked by an author "
                   "(59 confirmed, 2 rejected, 4 unsure); the ablation uses all 65 marked spans."),
        "rubric": json.loads((ROOT / "results" / "framing_conventions.json").read_text())["rubric"],
        "n_programs": len(out), "n_spans": sum(len(p["spans"]) for p in out),
        "programs": out,
    }


def main() -> None:
    out = build()
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for p in out["programs"]:
        print(f"{p['program']:28} spans {len(p['spans'])} chars {p['span_chars']:4}  C {p['control_C']['chars']:4} ({p['control_C']['pool']}, "
              f"{len(p['control_C']['units'])} units)  D appends {len(p['D_appended'])}")
    print("programs", out["n_programs"], "spans", out["n_spans"])


if __name__ == "__main__":
    main()
