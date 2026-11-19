"""Freeze the graded e-SNLI explanation-ranking test set (Bailleux, Mukherjee et al.,
"Explanation Quality Assessment as Ranking with Listwise Rewards", Findings ACL 2026).

The paper evaluates on e-SNLI validation (Table 20, "10k queries") recast as a
five-candidate ranking task: the human e-SNLI explanation (gold tier) plus four
generated tiers (good, fair, poor, nonsense) from "deterministic, label-aware
templates (Table 10)", each scored by the "content-aware heuristic scorer
(Algorithm 1)" with overlapping tier ranges (Eqs. 5-9). Neither the graded data
nor the Table 10 / Algorithm 1 code is in the released repository
(github.com/Tankiit/PPO_Learning_to_rank @ 713c28b, branches ARR and Oct-2026),
so this script rebuilds the construction from the paper text. Every choice the
paper leaves open is fixed below and listed in data/manifests/expl_rank.json.

Source: official e-SNLI release, github.com/OanaMariaCamburu/e-SNLI @ 7b585a3,
dataset/esnli_dev.csv (MIT; SNLI text CC BY-SA 4.0), 9,842 rows.

    git clone https://github.com/OanaMariaCamburu/e-SNLI data/raw/expl_rank/esnli
    git -C data/raw/expl_rank/esnli checkout 7b585a3f077fdea899780eb0473940522ae44a2e
    uv run python scripts/prepare_expl_rank.py
    # optional language screen (report only; nothing is dropped automatically):
    uv run --with langid python scripts/prepare_expl_rank.py --check-english

Writes data/frozen/expl_rank/test.jsonl.gz (gzip, mtime 0): one row per e-SNLI validation pair,
file order. Candidates are listed in a hash-shuffled order (tier-blind) with ids
c0..c4; gold per candidate = tier, tier grade (gold 4 ... nonsense 0) and the
Algorithm 1 score in [0, 1].
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from gauntlet.task import write_jsonl

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "expl_rank" / "esnli" / "dataset" / "esnli_dev.csv"
RAW_SHA256 = "d7a656d196eea18fd827d6f5486b00f5d4cf469dbf13a252cc211495a186bcb4"
OUT = REPO / "data" / "frozen" / "expl_rank" / "test.jsonl.gz"
EXPECTED_N = 9842
N_PAIRWISE = 200
VERSION = "expl_rank-v1"

TIERS = ("gold", "good", "fair", "poor", "nonsense")
GRADE = {"gold": 4, "good": 3, "fair": 2, "poor": 1, "nonsense": 0}
# Paper App. B, Eqs. (5)-(9): overlapping base-score ranges.
RANGES = {
    "gold": (0.70, 1.00),
    "good": (0.50, 0.85),
    "fair": (0.30, 0.70),
    "poor": (0.10, 0.50),
    "nonsense": (0.00, 0.30),
}
# Paper Table 10, verbatim (the "[Wrong!]" marks are table annotations, not text).
# The paper gives no neutral column; the neutral row is ours, written in the same
# register (relation phrase from the authors' later arr-esnli-v1 builder:
# "does not provide enough information to establish" / "does not decide").
# The Table 10 "Gold" template is unused: for e-SNLI the gold tier is the human
# Explanation_1 (paper Sec. 4 "Graded Dataset Construction", App. D).
TEMPLATES = {
    "entailment": {
        "good": "The premise entails the hypothesis because they convey compatible information.",
        "fair": "The premise supports the hypothesis.",
        "poor": "This is a contradiction because the premise and hypothesis are different.",
        "nonsense": "The quantum mechanics of penguin migration patterns suggest umbrella distribution.",
    },
    "contradiction": {
        "good": "These statements contradict because they make incompatible claims.",
        "fair": "The statements conflict.",
        "poor": "This is neutral because they\u2019re both statements.",
        "nonsense": "Purple elephants dance backwards when triangles sing opera.",
    },
    "neutral": {
        "good": "The premise neither entails nor contradicts the hypothesis because it does not provide enough information to establish it.",
        "fair": "The premise does not decide the hypothesis.",
        "poor": "This is an entailment because the premise and hypothesis are similar.",
        "nonsense": "The quantum mechanics of penguin migration patterns suggest umbrella distribution.",
    },
}
# Algorithm 1 line 3 "y is mentioned in e": the label's stem, case-insensitive.
LABEL_STEM = {"entailment": "entail", "contradiction": "contradict", "neutral": "neutral"}
# Algorithm 1 line 6 "reasoning keywords ('because', 'therefore', etc.)": the
# authors' own logical-connector list (evaluate_model.py L455 @ 713c28b).
REASONING_KEYWORDS = frozenset({"because", "since", "therefore", "thus", "so", "hence", "as", "when", "if"})
WORD = re.compile(r"[a-z0-9]+(?:['\u2019][a-z]+)?")
VOWELS = set("aeiouy")


def sha(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def uniform(low: float, high: float, *parts: str) -> float:
    """Deterministic U(low, high) from a SHA-256 of the content (Algorithm 1 note)."""
    return low + (high - low) * int(sha("uniform", *parts), 16) / float((1 << 256) - 1)


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def is_gibberish(word: str) -> bool:
    """Algorithm 1 line 16 leaves "gibberish" undefined. Fixed rule: an alphabetic
    token of >= 5 letters with no vowel, or a letter repeated 4+ times in a row."""
    if not word.isalpha():
        return False
    return (len(word) >= 5 and not VOWELS & set(word)) or re.search(r"(.)\1{3,}", word) is not None


def heuristic_score(text: str, premise: str, hypothesis: str, label: str, tier: str, pair_id: str) -> tuple[float, dict]:
    """Paper Algorithm 1 (App. B)."""
    low, high = RANGES[tier]
    s = uniform(low, high, VERSION, pair_id, tier, text)
    w = words(text)
    ph = set(words(premise)) | set(words(hypothesis))
    flags = {
        "label_match": LABEL_STEM[label] in text.lower(),
        "reasoning": bool(REASONING_KEYWORDS & set(w)),
        "overlap": (len(set(w) & ph) / len(ph) if ph else 0.0) > 0.3,
        "too_short": tier in ("gold", "good") and len(w) < 15,
        "gibberish": tier != "nonsense" and any(is_gibberish(x) for x in w),
    }
    s += 0.05 * flags["label_match"] + 0.03 * flags["reasoning"] + 0.02 * flags["overlap"]
    s -= 0.10 * flags["too_short"] + 0.30 * flags["gibberish"]
    return round(max(0.0, min(1.0, s)), 6), flags


def build_row(src: dict) -> tuple[dict, dict]:
    pair_id = src["pairID"]
    premise, hypothesis, label = src["Sentence1"].strip(), src["Sentence2"].strip(), src["gold_label"]
    texts = {"gold": src["Explanation_1"].strip(), **TEMPLATES[label]}
    if len(set(texts.values())) != len(TIERS):
        raise SystemExit(f"{pair_id}: duplicate candidate texts")
    order = sorted(TIERS, key=lambda t: sha(VERSION, "order", pair_id, t))
    candidates, gold, flags = [], {}, {}
    for i, tier in enumerate(order):
        cid = f"c{i}"
        score, f = heuristic_score(texts[tier], premise, hypothesis, label, tier, pair_id)
        candidates.append({"id": cid, "text": texts[tier]})
        gold[cid] = {"tier": tier, "grade": GRADE[tier], "score": score}
        flags[tier] = f
    row = {
        "id": f"esnli-dev-{pair_id}",
        "pair_id": pair_id,
        "premise": premise,
        "hypothesis": hypothesis,
        "label": label,
        "candidates": candidates,
        "gold": gold,
    }
    return row, flags


def check_english(rows: list[dict]) -> None:
    import langid  # noqa: PLC0415 - optional, via `uv run --with langid`

    flagged = []
    for r in rows:
        gold_text = next(c["text"] for c in r["candidates"] if r["gold"][c["id"]]["tier"] == "gold")
        for field, text in (("premise", r["premise"]), ("hypothesis", r["hypothesis"]), ("explanation", gold_text)):
            lang, _ = langid.classify(text)
            non_ascii = sum(ord(ch) > 127 for ch in text)
            if lang != "en" or non_ascii:
                flagged.append({"id": r["id"], "field": field, "langid": lang, "non_ascii": non_ascii, "text": text})
    print(json.dumps({"n_flagged": len(flagged), "by_lang": Counter(f["langid"] for f in flagged)}, indent=1))
    for f in flagged:
        print(json.dumps(f, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-english", action="store_true")
    args = ap.parse_args()
    digest = hashlib.sha256(RAW.read_bytes()).hexdigest()
    if digest != RAW_SHA256:
        raise SystemExit(f"unexpected raw SHA-256 {digest}")
    with RAW.open(encoding="utf-8", newline="") as fh:
        src_rows = list(csv.DictReader(fh))
    if len(src_rows) != EXPECTED_N:
        raise SystemExit(f"expected {EXPECTED_N} rows, got {len(src_rows)}")
    built = [build_row(s) for s in src_rows]
    rows = [r for r, _ in built]
    # The paper's 200 held-out pairwise-agreement queries (App. D) are not
    # identified; freeze a tier-blind hash-selected 200.
    chosen = set(sorted((r["pair_id"] for r in rows), key=lambda p: sha(VERSION, "pairwise200", p))[:N_PAIRWISE])
    for r in rows:
        r["pairwise200"] = r["pair_id"] in chosen
    if args.check_english:
        check_english(rows)
        return

    write_jsonl(OUT, rows)

    # Construction statistics (paper Table 11 analogue).
    violations = 0
    for r in rows:
        by_tier = {g["tier"]: g["score"] for g in r["gold"].values()}
        seq = [by_tier[t] for t in TIERS]
        violations += any(a < b for a, b in zip(seq, seq[1:]))
    flag_rates = {
        tier: {k: round(sum(f[tier][k] for _, f in built) / len(built), 4) for k in built[0][1][tier]} for tier in TIERS
    }
    tier_means = {t: round(statistics.mean(r["gold"][c]["score"] for r in rows for c in r["gold"] if r["gold"][c]["tier"] == t), 4) for t in TIERS}
    print(f"wrote {len(rows)} rows to {OUT.relative_to(REPO)}")
    print("sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())
    print("labels", dict(Counter(r["label"] for r in rows)))
    print("pairwise200 labels", dict(Counter(r["label"] for r in rows if r["pairwise200"])))
    print("ambiguity (rows with an adjacent-tier order violation)", round(violations / len(rows), 4))
    print("mean score by tier", tier_means)
    print("Algorithm 1 flag rates", json.dumps(flag_rates))


if __name__ == "__main__":
    main()
