"""FactPICO (Joseph et al., ACL 2024): expert-grounded factuality of plain language summaries.

Input: an RCT abstract (Evidence Inference 2.0) and one LLM-generated plain language
summary of it. Gold: medical experts' ratings of the summary on Population, Intervention,
Comparator, Outcome (4 accurate / 3 somewhat inaccurate or vague / 2 severe inaccuracies
or missing critical descriptors / 1 missing; 0 = N/A) and on each Evidence Inference result
span (4 accurate / 3 vague / 2 inaccurate / 1 not mentioned; "Results" = mean over spans),
plus "Avg. PICO-R" = mean of the five. Doubly annotated summaries carry averaged ratings.

Evaluation set: all 345 released summaries (115 abstracts x Alpaca/GPT-4/Llama-2); there is
no split and nothing is trained. Replaces the paper's factuality evaluators, best of which
are GPT-4 full-text evaluation and the GPT-4 + PICO-R extraction pipeline ("Extract").

Jev: one request per summary over {abstract, summary} with one score question per PICO
element (5 levels, index = paper rating 0-4) and one per result span (4 levels, index + 1 =
rating 1-4). Jev's overall score = mean of its five element ratings (expected values),
mirroring the paper's LLM evaluators ("the average rating across each element", Table 4).

Metrics (paper Table 4 / Table 10, reproduced exactly from the release by
scripts/prepare_factpico.py): Kendall tau-b, Spearman rho and pairwise accuracy acc_eq
(Deutsch et al. 2023, exact ties, no tie calibration) over all 345 summaries.
Primary: Kendall tau-b between Jev's overall score and Avg. PICO-R (Table 4 "Avg." row).
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
from scipy.stats import kendalltau, spearmanr

from ..task import FROZEN, Item, Request, Task, read_jsonl

KEY = "factuality"
PICO = ("population", "intervention", "comparator", "outcome")
ELEMENTS = (*PICO, "results")
HUMAN_ATTRIBUTES = (*ELEMENTS, "avg")
PICO_LEVELS = 5  # index = rating: 0 N/A, 1 missing, 2 severe, 3 somewhat, 4 accurate
RESULT_LEVELS = 4  # index + 1 = rating: 1 not mentioned, 2 inaccurate, 3 vague, 4 accurate
SPAN_PLACEHOLDER = "{span}"
# Frozen truncation rule: abstract cut to its first MAX_ABSTRACT_CHARS, summary to its first
# MAX_SUMMARY_CHARS characters (~15K tokens together, inside the 32K context with the
# <=9K-char question block). Longest released abstract 4,673 chars, summary 2,730: nothing is cut.
MAX_ABSTRACT_CHARS = 40_000
MAX_SUMMARY_CHARS = 20_000


def _clip(text: str, limit: int) -> tuple[str, bool]:
    return (text[:limit], True) if len(text) > limit else (text, False)


def result_qid(i: int) -> str:
    return f"result_{i + 1}"


def pairwise_accuracy(gold: list[float], pred: list[float]) -> float:
    """acc_eq (Deutsch et al., 2023) without tie calibration: share of all item pairs whose
    ordering (including exact ties) agrees. Reproduces the paper's Table 4 acc_eq column."""
    g = np.asarray(gold, dtype=float)
    p = np.asarray(pred, dtype=float)
    i, j = np.triu_indices(len(g), 1)
    return float(np.mean(np.sign(g[i] - g[j]) == np.sign(p[i] - p[j])))


def _correlations(gold: list[float], pred: list[float]) -> dict[str, float]:
    return {
        "kendall_tau_b": float(kendalltau(gold, pred, variant="b")[0]),
        "spearman_rho": float(spearmanr(gold, pred)[0]),
        "acc_eq": pairwise_accuracy(gold, pred),
    }


def _argmax_index(probabilities: dict[str, float], levels: int) -> int:
    """Most probable level; exact ties go to the lower level (the stricter rating)."""
    p = [float(probabilities.get(str(k), 0.0)) for k in range(levels)]
    return max(range(levels), key=lambda k: (p[k], -k))


class FactPICO(Task):
    name = "factpico"
    prompt_file = "factpico.json"

    def items(self) -> list[Item]:
        rows = read_jsonl(FROZEN / "factpico" / "test.jsonl")
        return [
            Item(
                id=r["id"],
                gold=r["gold"],
                fields={
                    "abstract": r["abstract"],
                    "summary": r["summary"],
                    "result_spans": r["result_spans"],
                    "summarizer": r["summarizer"],
                },
            )
            for r in rows
        ]

    def _state(self, item: Item) -> tuple[dict[str, str], bool]:
        abstract, cut_a = _clip(item.fields["abstract"], MAX_ABSTRACT_CHARS)
        summary, cut_s = _clip(item.fields["summary"], MAX_SUMMARY_CHARS)
        return {"abstract": abstract, "summary": summary}, cut_a or cut_s

    def questions(self, item: Item) -> dict[str, Any]:
        prompt = self.prompt()
        questions: dict[str, Any] = {el: prompt["pico"][el] for el in PICO}
        template = prompt["evidence_inference"]
        for i, span in enumerate(item.fields["result_spans"]):
            questions[result_qid(i)] = {
                "type": template["type"],
                "instructions": template["instructions"].replace(SPAN_PLACEHOLDER, span),
                "criteria": template["criteria"],
            }
        return questions

    def requests(self, item: Item) -> Iterator[Request]:
        state, _ = self._state(item)
        yield Request(item_id=item.id, key=KEY, state=state, questions=self.questions(item))

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        ans = answers[KEY]
        expected = {el: float(ans[el]["score"]) for el in PICO}
        discrete = {el: _argmax_index(ans[el]["probabilities"], PICO_LEVELS) for el in PICO}
        spans = [ans[result_qid(i)] for i in range(len(item.fields["result_spans"]))]
        span_expected = [float(a["score"]) + 1.0 for a in spans]
        span_discrete = [_argmax_index(a["probabilities"], RESULT_LEVELS) + 1 for a in spans]
        # Means are rounded to 9 decimals so equal rating profiles tie exactly (acc_eq and
        # tau-b count ties); gold is used verbatim as released, as in the paper.
        expected["results"] = round(sum(span_expected) / len(span_expected), 9)
        discrete["results"] = round(sum(span_discrete) / len(span_discrete), 9)
        expected["overall"] = round(sum(expected[el] for el in ELEMENTS) / len(ELEMENTS), 9)
        discrete["overall"] = round(sum(discrete[el] for el in ELEMENTS) / len(ELEMENTS), 9)
        return {"expected": expected, "discrete": discrete, "result_spans_expected": span_expected}

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        gold = {a: [it.gold[a] for it in items] for a in HUMAN_ATTRIBUTES}

        def block(kind: str) -> dict[str, Any]:
            pred = {k: [predictions[it.id][kind][k] for it in items] for k in (*ELEMENTS, "overall")}
            # Table 4 protocol: every human attribute vs the system's overall score.
            overall = {a: _correlations(gold[a], pred["overall"]) for a in HUMAN_ATTRIBUTES}
            # Table 10 protocol: attribute-wise (Jev element vs expert element); Avg = overall.
            attribute = {a: _correlations(gold[a], pred[a if a != "avg" else "overall"]) for a in HUMAN_ATTRIBUTES}
            return {"vs_overall": overall, "attribute_wise": attribute}

        expected = block("expected")
        by_summarizer: dict[str, dict[str, float]] = {}
        for m in sorted({it.fields["summarizer"] for it in items}):
            sub = [it for it in items if it.fields["summarizer"] == m]
            by_summarizer[m] = {
                "jev_overall_mean": float(np.mean([predictions[it.id]["expected"]["overall"] for it in sub])),
                "human_avg_mean": float(np.mean([it.gold["avg"] for it in sub])),
                "n": len(sub),
            }
        avg = expected["vs_overall"]["avg"]
        return {
            "n": len(items),
            # Primary: Table 4 "Avg." row, Jev overall (expected ratings) vs Avg. PICO-R.
            "kendall_tau_b": avg["kendall_tau_b"],
            "spearman_rho": avg["spearman_rho"],
            "acc_eq": avg["acc_eq"],
            "expected": expected,
            # Supplementary: most-probable level per question instead of expected value.
            "discrete": block("discrete"),
            # Table 5 analogue (system level): mean scores per summarizing LLM.
            "by_summarizer": by_summarizer,
            "n_result_questions": sum(len(it.fields["result_spans"]) for it in items),
            "n_truncated": sum(self._state(it)[1] for it in items),
        }


TASKS = {"factpico": FactPICO}
