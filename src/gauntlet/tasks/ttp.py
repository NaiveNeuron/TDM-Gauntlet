"""Security TTP mapping (Nguyen, Srndic & Neth, Findings EACL 2024), Jev as the text/TTP matcher.

Paper: https://aclanthology.org/2024.findings-eacl.25/ ; data: HF `tumeteor/Security-TTP-Mapping`
@1af2f0e (CC BY 4.0; no model code is released). The paper recasts TTP mapping as text matching:
a matcher g(x, y) scores the input text x against the textual profile y of every TTP in the label
corpus, and the ranked list is evaluated with P@1, R@1, F1@3, MRR@3 (Tables 2-3) and F1@5, MRR@5
(Tables 6-7). Jev replaces g: for every text it answers one noul per candidate TTP ("is this
technique explicitly or implicitly mentioned in the text?", the paper's App. D.3 LLM prompt asked per
candidate), and candidates are ranked by P(yes).

Candidates = the paper's full label corpus: all 737 TTP profiles released in ttp-desc-mappings.csv
(every ATT&CK Enterprise v12.0 technique and sub-technique, including 143 deprecated/revoked ones;
TRAM gold uses two deprecated IDs, T1064 and T1043). No candidate pruning: the paper's matcher ranks
the whole corpus. The 737 questions are split into a fixed ID-ordered partition of requests (all over
the same one-text state) so each request stays well inside the 32K-token context; questions are
answered independently, so the partition does not affect answers.

Metrics follow App. D.1: per text P@k = hits/k, R@k = hits/|gold|, averaged over texts;
F1@k = 2 P@k R@k / (P@k + R@k) from the averaged P@k and R@k; MRR@k = mean reciprocal rank of the
first gold label within the top k (0 if none). Gold = the set of released labels (the release repeats
some IDs within a row; duplicates are dropped). Technique-level scores (Tables 3/7) resolve every
gold and ranked label to its parent technique; the ranked list keeps each parent's first occurrence.
Exact probability ties are broken by ATT&CK ID order.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from typing import Any

from ..metrics import mean
from ..task import FROZEN, Item, Request, Task, read_jsonl

DATA = FROZEN / "ttp"
PROMPT_FILE = "ttp.json"
# Frozen truncation rule: the text is cut to its first MAX_TEXT_CHARS characters. The longest
# frozen text is 2,744 characters, so nothing is truncated.
MAX_TEXT_CHARS = 8_000
# Fixed request partition: candidates in ATT&CK ID order are packed greedily so the serialized
# questions of one request total at most this many characters (~18K cl100k tokens for this
# lower-cased English prose at 5.3 chars/token; <30K tokens with the state even at 3.3 chars/token).
QUESTION_CHARS_PER_REQUEST = 96_000
KS = (1, 3, 5)
KEEP = 20  # ranked candidates stored per prediction
TECH_KEEP = 10  # ranked parent techniques stored per prediction

# task -> [(frozen split file, breakdown label)]
SPLITS: dict[str, list[tuple[str, str]]] = {
    "ttp/tram": [("tram_test", "tram")],
    "ttp/procedures_plus": [("procedure_test", "procedures"), ("derived_procedure_test", "derived_procedures")],
    "ttp/expert": [("expert_test", "expert_test")],
    "ttp/expert_full": [("expert_train", "expert_train"), ("expert_dev", "expert_dev"), ("expert_test", "expert_test")],
}


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parent(ttp: str) -> str:
    return ttp.split(".")[0]


def _clip(text: str) -> tuple[str, bool]:
    return (text[:MAX_TEXT_CHARS], True) if len(text) > MAX_TEXT_CHARS else (text, False)


def _question(template: dict[str, str], cand: dict[str, Any]) -> dict[str, str]:
    instructions = (
        template["instructions"].replace("{id}", cand["id"]).replace("{name}", cand["name"]).replace("{description}", cand["description"])
    )
    return {"type": template["type"], "instructions": instructions}


def _chunks(questions: dict[str, dict[str, str]]) -> list[dict[str, dict[str, str]]]:
    chunks: list[dict[str, dict[str, str]]] = [{}]
    size = 0
    for qid, q in questions.items():
        n = len(json.dumps({qid: q}, ensure_ascii=False))
        if chunks[-1] and size + n > QUESTION_CHARS_PER_REQUEST:
            chunks.append({})
            size = 0
        chunks[-1][qid] = q
        size += n
    return chunks


def technique_ranking(ranking: Sequence[str]) -> list[str]:
    """Resolve ranked (sub-)techniques to parents, keeping each parent's first occurrence."""
    seen: dict[str, None] = {}
    for ttp in ranking:
        seen.setdefault(_parent(ttp), None)
    return list(seen)


def ranking_metrics(rankings: Sequence[Sequence[str]], golds: Sequence[set[str]], ks: Sequence[int] = KS) -> dict[str, float]:
    """App. D.1 metrics over texts; F1@k from the averaged P@k and R@k."""
    out: dict[str, float] = {}
    for k in ks:
        hits = [len(set(r[:k]) & g) for r, g in zip(rankings, golds)]
        p = mean([h / k for h in hits])
        r = mean([h / len(g) for h, g in zip(hits, golds)])
        rr = []
        for ranking, gold in zip(rankings, golds):
            first = next((i for i, ttp in enumerate(ranking[:k]) if ttp in gold), None)
            rr.append(0.0 if first is None else 1.0 / (first + 1))
        out[f"p@{k}"] = p
        out[f"r@{k}"] = r
        out[f"f1@{k}"] = 2 * p * r / (p + r) if p + r > 0 else 0.0
        # Supplementary: per-text F1@k averaged over texts.
        out[f"f1@{k}_text_mean"] = mean([2 * h / (k + len(g)) for h, g in zip(hits, golds)])
        out[f"mrr@{k}"] = mean(rr)
    # Upper bound of R@1 (paper Table 2 caption quotes 0.504 for Expert).
    out["ideal_r@1"] = mean([1 / len(g) for g in golds])
    return out


class TTPMapping(Task):
    prompt_file = PROMPT_FILE

    def __init__(self, name: str, splits: list[tuple[str, str | None]] | None = None) -> None:
        """`splits` = [(frozen file stem, subset label)]; label None reads each row's `subset`."""
        self.name = name
        self.splits = SPLITS[name] if splits is None else splits
        prompt = self.prompt()
        cand_path = DATA / "candidates.jsonl"
        if _sha256(cand_path) != prompt["_candidates_sha256"]:
            raise ValueError(f"{cand_path} does not match _candidates_sha256 in prompts/{self.prompt_file}")
        self.candidates = read_jsonl(cand_path)
        self.order = {c["id"]: i for i, c in enumerate(self.candidates)}
        questions = {c["id"]: _question(prompt["question_template"], c) for c in self.candidates}
        self.chunks = _chunks(questions)

    def items(self) -> list[Item]:
        out = []
        for split, label in self.splits:
            for r in read_jsonl(DATA / f"{split}.jsonl"):
                out.append(Item(id=r["id"], gold=r["gold"], fields={"text": r["text"], "subset": label or r["subset"]}))
        return out

    def _state(self, item: Item) -> tuple[dict[str, str], bool]:
        text, cut = _clip(item.fields["text"])
        return {"text": text}, cut

    def requests(self, item: Item):
        state, _ = self._state(item)
        for j, chunk in enumerate(self.chunks):
            yield Request(item_id=item.id, key=f"candidates-{j:02d}", state=state, questions=chunk)

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        p: dict[str, float] = {}
        for j, chunk in enumerate(self.chunks):
            got = answers[f"candidates-{j:02d}"]
            for qid in chunk:
                p[qid] = float(got[qid]["noul"])
        ranking = sorted(p, key=lambda t: (-p[t], self.order[t]))
        return {
            "ranking": ranking[:KEEP],
            "p": [p[t] for t in ranking[:KEEP]],
            "technique_ranking": technique_ranking(ranking)[:TECH_KEEP],
        }

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        def block(its: list[Item]) -> dict[str, Any]:
            golds = [set(it.gold) for it in its]
            rankings = [predictions[it.id]["ranking"] for it in its]
            tech_golds = [{_parent(g) for g in gold} for gold in golds]
            tech_rankings = [predictions[it.id]["technique_ranking"] for it in its]
            return {
                "n": len(its),
                **ranking_metrics(rankings, golds),
                "technique_level": ranking_metrics(tech_rankings, tech_golds),
                "avg_gold_labels": mean([len(g) for g in golds]),
            }

        out = block(items)
        subsets = sorted({it.fields["subset"] for it in items})
        if len(subsets) > 1:
            out["by_subset"] = {s: block([it for it in items if it.fields["subset"] == s]) for s in subsets}
        out["n_candidates"] = len(self.candidates)
        out["n_truncated"] = sum(self._state(it)[1] for it in items)
        return out


def _factory(name: str) -> Callable[[], Task]:
    return lambda: TTPMapping(name)


TASKS: dict[str, Callable[[], Task]] = {name: _factory(name) for name in SPLITS}
