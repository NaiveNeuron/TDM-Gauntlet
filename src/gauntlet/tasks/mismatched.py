"""MISMATCHED scientific NLI (Shaik et al., Findings ACL 2025).

"A MISMATCHED Benchmark for Scientific Natural Language Inference",
https://aclanthology.org/2025.findings-acl.1109/. Input: a pair of sentences from a
psychology, engineering or public-health research article (any linking phrase at
the start of the second sentence removed). Gold: one of four scientific NLI
relations, entailment / reasoning / contrasting / neutral. Test: 2,400 pairs, 800 per
domain, 200 per domain x class (paper Table 2; release test.tsv).

Replaced predictor: the paper's scientific NLI classifiers trained on SciNLI /
MSciNLI (best: SciBERT fine-tuned on MSciNLI+, 78.17 overall macro-F1). Jev answers
one `choice` question per pair, with the authors' zero-shot LLM prompt as the
instruction. Metric (paper Table 4): macro-F1 over the four classes per domain and
overall, where overall is macro-F1 pooled over all 2,400 pairs (authors'
evaluation code; Table 8 per-class F1s average to it).

Data are CC BY-SA 4.0, frozen as JSONL by scripts/prepare_mismatched.py.
"""

from __future__ import annotations

from typing import Any, Iterator

from .. import metrics
from ..task import FROZEN, Item, Request, Task, read_jsonl

# Authors' class-index order (Training_and_testing_utils_transformers.py target
# names: contrasting, reasoning, entailing, neutral); also the argmax tie-break order.
LABELS = ("contrasting", "reasoning", "entailment", "neutral")
DOMAINS = ("Psychology", "Engineering", "Public Health")
KEY = "nli"
# Frozen truncation rule: each sentence is cut to its first MAX_FIELD_CHARS
# characters (two fields stay well under the 32K-token context). The longest
# released test sentence is 872 chars, so nothing is truncated.
MAX_FIELD_CHARS = 36_000

# Paper macro-F1 (%), (Psychology, Engineering, Public Health, Overall). Table 4
# (p. 21529) unless noted; means over 3 runs where the paper reports them.
PAPER_MACRO_F1: dict[str, tuple[float | None, ...]] = {
    "SciBERT_MSciNLI+": (79.18, 76.50, 78.79, 78.17),
    "RoBERTa_MSciNLI+": (77.91, 77.63, 78.79, 78.11),
    "SciBERT_MSciNLI+ +Implicit (Table 9)": (None, None, None, 79.66),
    "Llama-2-13B LoRA on SciNLI (Table 15)": (None, None, None, 82.87),
    "Phi-3 zero-shot": (55.38, 53.15, 49.31, 52.95),
    "Phi-3 few-shot MSciNLI": (58.64, 56.76, 55.68, 57.16),
    "GPT-4o zero-shot": (52.42, 50.12, 47.26, 50.26),
    "GPT-4o few-shot MSciNLI+": (63.62, 61.06, 62.96, 62.73),
    "Gemini-1.5-Pro zero-shot": (54.28, 58.49, 51.59, 55.55),
    "Gemini-1.5-Pro few-shot MSciNLI+": (63.68, 62.57, 62.51, 62.95),
}


def _clip(text: str) -> tuple[str, bool]:
    return (text[:MAX_FIELD_CHARS], True) if len(text) > MAX_FIELD_CHARS else (text, False)


def _block(gold: list[str], pred: list[str]) -> dict[str, Any]:
    labels = list(LABELS)
    return {
        "n": len(gold),
        "macro_f1": metrics.macro_f1(gold, pred, labels=labels),
        "accuracy": metrics.accuracy(gold, pred),
        "per_class_f1": metrics.per_class_f1(gold, pred, labels=labels),
        # confusion[gold][pred]
        "confusion": {g: {p: sum(a == g and b == p for a, b in zip(gold, pred)) for p in LABELS} for g in LABELS},
    }


class Mismatched(Task):
    name = "mismatched"
    prompt_file = "mismatched.json"

    def items(self) -> list[Item]:
        rows = read_jsonl(FROZEN / "mismatched" / "test.jsonl")
        return [
            Item(
                id=r["id"],
                gold=r["gold"],
                fields={"sentence1": r["sentence1"], "sentence2": r["sentence2"], "domain": r["domain"]},
            )
            for r in rows
        ]

    def _state(self, item: Item) -> tuple[dict[str, str], bool]:
        s1, cut1 = _clip(item.fields["sentence1"])
        s2, cut2 = _clip(item.fields["sentence2"])
        return {"Sentence1": s1, "Sentence2": s2}, cut1 or cut2

    def requests(self, item: Item) -> Iterator[Request]:
        state, _ = self._state(item)
        yield Request(item_id=item.id, key=KEY, state=state, questions={KEY: self.prompt()["question"]})

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        probs = answers[KEY][KEY]["probabilities"]
        p = {label: float(probs.get(label, 0.0)) for label in LABELS}
        # argmax; exact ties go to the earlier label in LABELS order.
        label = max(LABELS, key=lambda lab: (p[lab], -LABELS.index(lab)))
        return {"label": label, "probabilities": p}

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        gold = [it.gold for it in items]
        pred = [predictions[it.id]["label"] for it in items]
        out: dict[str, Any] = _block(gold, pred)
        out["by_domain"] = {}
        for domain in DOMAINS:
            idx = [i for i, it in enumerate(items) if it.fields["domain"] == domain]
            if idx:
                out["by_domain"][domain] = _block([gold[i] for i in idx], [pred[i] for i in idx])
        out["n_pred"] = {lab: pred.count(lab) for lab in LABELS}
        out["n_truncated"] = sum(self._state(it)[1] for it in items)
        out["paper_macro_f1_percent"] = {
            system: dict(zip((*DOMAINS, "Overall"), values)) for system, values in PAPER_MACRO_F1.items()
        }
        return out


TASKS = {"mismatched": Mismatched}
