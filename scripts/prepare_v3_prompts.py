"""Write the v3 prompt files (deterministic; rerun reproduces them byte for byte).

  prompts/ttp_v3.json            round 1 = prompts/ttp.json question template verbatim (so round-1
                                 requests are byte-identical to the v1 run and served from its
                                 answers), plus the round-2 rerank Choice: options "<ID> <name>" with
                                 a short description = the release profile's first sentence
                                 (a leading "**this technique has been deprecated ...**" notice is
                                 skipped), cut to <= 200 characters at a word boundary.
  prompts/relemb_wikiwd_v3.json  prompts/relemb_wikiwd_v2.json verbatim (rounds 1-2 unchanged) plus
                                 the round-3 rerank: the v2 question over a shortlist of option names.

    uv run python scripts/prepare_v3_prompts.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gauntlet.task import FROZEN, PROMPTS, read_jsonl  # noqa: E402

SHORT_CHARS = 200


def short_description(description: str, cap: int = SHORT_CHARS) -> str:
    text = " ".join(description.split())
    text = re.sub(r"^\*\*.*?\*\*\s*", "", text)  # deprecation notice
    m = re.search(r"(?<!e\.g)(?<!i\.e)(?<!etc)\.\s", text)
    first = text[: m.start() + 1] if m else text
    if len(first) > cap:
        first = first[:cap].rsplit(" ", 1)[0] + " ..."
    return first


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ttp_v3() -> dict:
    v1 = json.loads((PROMPTS / "ttp.json").read_text())
    cands = read_jsonl(FROZEN / "ttp" / "candidates.jsonl")
    return {
        "_source": "v3 of prompts/ttp.json (Nguyen, Srndic & Neth, Findings EACL 2024). Round 1 is the v1 pointwise "
        "noul scan, question template copied verbatim. Round 2 (new, v3) reranks Jev's own round-1 top-k with one "
        "relative Choice, the retrieve-then-rerank pattern of TypeSafe's skill-suggestion cookbook "
        "(https://docs.typesafe.ai/cookbooks/skill_suggestion) and Infer-Retrieve-Rank (arXiv 2401.12178); the "
        "candidate set is still the paper's full 737-profile corpus (round 1 ranks all of it).",
        "_primitive": "round 1: noul per candidate (all 737); round 2: choice over the round-1 top-k (k, and the "
        "mixing weight with round 1, fit on the v3 dev sample: data/frozen/ttp/v3_params.json)",
        "_round1_prompt": "ttp.json",
        "_round1_sha256": sha256(PROMPTS / "ttp.json"),
        "_candidates_sha256": v1["_candidates_sha256"],
        "question_template": v1["question_template"],
        "rerank": {
            "type": "choice",
            "instructions": "TTP mapping: identifying MITRE ATT&CK attack patterns (TTPs) in cybersecurity writing. "
            "Which one of the following MITRE ATT&CK techniques (TTPs) is most clearly mentioned, explicitly or "
            "implicitly, in `text`?",
            "option_order": "ATT&CK ID order",
            "short_description_rule": f"first sentence of the release profile (leading deprecation notice skipped), "
            f"cut to <= {SHORT_CHARS} characters at a word boundary",
        },
        "rerank_options": {c["id"]: {"key": f"{c['id']} {c['name']}", "description": short_description(c["description"])} for c in cands},
    }


def wikiwd_v3() -> dict:
    v2 = json.loads((PROMPTS / "relemb_wikiwd_v2.json").read_text())
    return {
        **v2,
        "_version": "v3: rounds 1-2 exactly prompts/relemb_wikiwd_v2.json (every v2 field copied verbatim); new round 3 "
        "reranks a per-pair shortlist (the top-k properties of each round-1 half) with the same question.",
        "_round1_prompt": "relemb_wikiwd_v2.json",
        "_round1_sha256": sha256(PROMPTS / "relemb_wikiwd_v2.json"),
        "rerank": {
            "instructions": "the v2 `question` (same marked sentence, head and tail)",
            "options": "the k highest-probability properties of each round-1 half (probability > 0; ties in PID order), "
            "option names only with null descriptions as in v2, in PID order, plus the v2 `none_option` "
            "('none of the listed relations', read as P0), which the state's task statement already defines for "
            "questions that list only part of the inventory",
        },
    }


def main() -> None:
    for name, prompt in (("ttp_v3.json", ttp_v3()), ("relemb_wikiwd_v3.json", wikiwd_v3())):
        path = PROMPTS / name
        path.write_text(json.dumps(prompt, indent=1, ensure_ascii=False) + "\n")
        print(name, sha256(path))


if __name__ == "__main__":
    main()
