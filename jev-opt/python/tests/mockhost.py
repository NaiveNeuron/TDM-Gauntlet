"""Deterministic stand-in for the OMP host's `lm.complete` (no credentials, no network).

Text slots: seed criteria are replaced by keyword-rich wording. Program prompts: the YAML
in the first code block gets the same criteria; the structural role also switches the
root Choice to one-vs-rest Nouls.
"""

from __future__ import annotations

import re
from typing import Any

import yaml

IMPROVED = {
    "The message is a complaint.": "Complaint: the customer reports something broken, damaged, cracked, missing, late or terrible, is disappointed, wants a refund, or calls it awful or the worst.",
    "The message gives praise.": "Praise: the customer says they love it or that it is great, excellent, amazing, wonderful, fantastic or perfect, is happy or delighted, or says thanks.",
    "The message asks a question.": "Question: the customer asks how, when, where, does or which, wondering about something.",
}


def _first_block(prompt: str) -> str:
    match = re.search(r"```[^\n]*\n(.*?)```", prompt, flags=re.S)
    return match.group(1).strip() if match else ""


def _improve(program: dict[str, Any]) -> dict[str, Any]:
    for node in program["nodes"].values():
        for opt in (node.get("options") or {}).values():
            opt["criterion"] = IMPROVED.get(opt["criterion"], opt["criterion"])
    return program


def _population(prompt: str) -> str:
    """A YAML list of variants for the population proposer: keyword-rich Choice / one-vs-rest
    wordings, one with an aux cue and demonstrations, one with wrong option keys (rejected)."""
    labels = [lab.strip() for lab in re.search(r"exactly these keys: ([^\n#]+)", prompt).group(1).split(",")]
    crit = {lab: {"criterion": next((v for v in IMPROVED.values() if v.lower().startswith(lab.lower())), lab)} for lab in labels}
    variants = [
        {"rationale": "keyword criteria", "main": {"kind": "choice", "instructions": "Which kind of customer message is this?", "options": crit}},
        {"rationale": "one noul per label", "main": {"kind": "one_vs_rest", "instructions": "Judge the customer message.", "options": crit}},
        {"rationale": "cue + precedents", "demos": True, "main": {"kind": "choice", "instructions": "Classify the message.", "options": crit},
         "aux": {"asks": {"kind": "branch", "condition": "The customer asks how, when, where or which."}}},
        {"rationale": "bad keys", "main": {"kind": "choice", "instructions": "x", "options": {"a": {"criterion": "a"}, "b": {"criterion": "b"}}}},
    ]
    return "Here are the variants.\n```yaml\n" + yaml.safe_dump(variants, sort_keys=False) + "```"


def _seeds(prompt: str) -> str:
    """Seed programs for /jev-seeds on the minimal task (entry node `decide`): keyword-rich
    rewrites; decomposed adds one-vs-rest and a coarse-to-fine hierarchy, plus one invalid program."""
    crit = {lab: {"criterion": v} for lab, v in zip(("complaint", "praise", "question"), IMPROVED.values())}
    if "KIND jev-native" in prompt:
        programs = [{"rationale": f"contrastive rewrite {i}", "nodes": {"decide": {"kind": "choice", "instructions": f"What does the customer `message` do? ({i})", "state": ["message"], "options": crit}}} for i in (1, 2)]
    else:
        programs = [
            {"rationale": "one noul per label", "nodes": {"decide": {"kind": "one_vs_rest", "instructions": "Judge the customer `message`.", "state": ["message"], "options": crit}}},
            {"rationale": "question first, then tone", "nodes": {
                "decide": {"kind": "choice", "instructions": "Is the customer `message` a question or a statement?", "state": ["message"],
                           "options": {"question": crit["question"], "statement": {"criterion": "The customer states an opinion.", "to": {"node": "decide_tone"}}}},
                "decide_tone": {"kind": "choice", "instructions": "Is the statement negative or positive?", "state": ["message"], "options": {"complaint": crit["complaint"], "praise": crit["praise"]}}}},
            {"rationale": "drops a label", "nodes": {"decide": {"kind": "choice", "instructions": "x", "state": ["message"], "options": {"praise": crit["praise"], "question": crit["question"]}}}},
        ]
    return "```yaml\n" + yaml.safe_dump(programs, sort_keys=False) + "```"


def _tailored(prompt: str) -> str:
    """A joint rewrite for the jev-tailored proposer: the editable fields (last code block of the prompt)
    with seed criteria replaced by the keyword-rich wording."""
    current = yaml.safe_load(re.findall(r"```[^\n]*\n(.*?)```", prompt, flags=re.S)[-1])
    new = {field: IMPROVED.get(text.strip(), text) for field, text in current.items()}
    return "rationale: contrastive keyword criteria\n```yaml\n" + yaml.safe_dump(new, sort_keys=False) + "```"


def complete(params: dict[str, Any]) -> dict[str, Any]:
    prompt = params["prompt"] if isinstance(params["prompt"], str) else str(params["prompt"])
    current = _first_block(prompt)
    if "decision programs for Jev, a calibrated classifier" in prompt:
        text = _seeds(prompt)
    elif "I am improving the wording of one question of a Jev decision program" in prompt:
        text = _tailored(prompt)
    elif "NEW variants of node" in prompt:
        text = _population(prompt)
    elif params["role"] == "structural" or "decision program" in prompt:
        program = _improve(yaml.safe_load(current))
        rationale = "sharpen criteria with domain keywords"
        if params["role"] == "structural":
            root = program["nodes"][program["root"]]
            if root["kind"] == "choice":
                root["kind"] = "one_vs_rest"
                rationale = "one Noul per label with domain keywords"
        text = f"Rationale: {rationale}\n```yaml\n{yaml.safe_dump(program, sort_keys=False)}```"
    else:
        text = f"```\n{IMPROVED.get(current, current)}\n```"
    return {"text": text, "usage": {"input": len(prompt) // 4, "output": len(text) // 4, "cost_total": 0.001}, "provider": "mock", "model": "mock-reflector"}


class InProcessHost:
    """Requester for OMPReflectionLM in in-process tests; records every request."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def request(self, method: str, params: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        assert method == "lm.complete"
        self.requests.append(params)
        return complete(params)
