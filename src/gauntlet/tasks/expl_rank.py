"""Explanation quality ranking on graded e-SNLI (Bailleux, Mukherjee et al.,
"Explanation Quality Assessment as Ranking with Listwise Rewards", Findings ACL 2026).

Input: an e-SNLI validation instance (premise, hypothesis, gold relation) and five
candidate explanations: the human e-SNLI explanation (gold tier) and four
template-generated tiers (good, fair, poor, nonsense; paper Table 10). Gold per
candidate: the paper's Algorithm 1 heuristic score in [0, 1] (overlapping tier
ranges, content adjustments), rebuilt by `scripts/prepare_expl_rank.py` because
neither the graded data nor the construction code was released. Replaces the
paper's ListNet reward model, a pointwise scorer f(q, e) trained with a listwise
loss: Jev scores each candidate independently (one `score` question per candidate,
five questions in one request over the shared query state) and the candidates are
ranked by Jev's expected level.

Primary metrics (paper Table 20, e-SNLI validation): NDCG@5, MAP, MRR, Kendall tau,
Spearman rho against the Algorithm 1 scores, computed exactly as the authors'
`src/evaluation/metrics.py` (gain 2^s - 1; MAP relevance = score above the
per-query median; MRR = rank of the top-scored candidate; per-query rank
correlations averaged, constant-prediction queries skipped). Also the App. D
pairwise agreement (fraction of the 10 candidate pairs ordered as the gold scores,
predicted ties counted wrong) on a frozen 200-query subset, standing in for the
paper's unidentified 200 held-out queries. Secondary: the same metrics against the
designed tier order (gold > good > fair > poor > nonsense).
"""

from __future__ import annotations

import math
from typing import Any, Iterator

import numpy as np
from scipy import stats

from ..task import FROZEN, Item, Request, Task, read_jsonl

KEY = "rank"
TIERS = ("gold", "good", "fair", "poor", "nonsense")
N_LEVELS = 5
# Frozen truncation rule: premise, hypothesis and each candidate are cut to their
# first MAX_FIELD_CHARS characters. The longest released field is < 500 chars, so
# nothing is truncated.
MAX_FIELD_CHARS = 20_000


def _clip(text: str) -> tuple[str, bool]:
    return (text[:MAX_FIELD_CHARS], True) if len(text) > MAX_FIELD_CHARS else (text, False)


def _expected_level(ans: dict[str, Any]) -> float:
    probs = ans.get("probabilities") or {}
    p = [float(probs.get(str(i), 0.0)) for i in range(N_LEVELS)]
    total = sum(p)
    if total > 0:
        return sum(i * pi for i, pi in enumerate(p)) / total
    return float(ans["score"])


# --- ranking metrics, ported from the authors' src/evaluation/metrics.py (713c28b) ---


def _ndcg_at_k(order: list[int], true: np.ndarray, k: int) -> float:
    discounts = np.log2(np.arange(2, k + 2))
    gains = np.power(2.0, true[order][:k]) - 1.0
    ideal = np.power(2.0, np.sort(true)[::-1][:k]) - 1.0
    idcg = float((ideal / discounts[: len(ideal)]).sum())
    return float((gains / discounts[: len(gains)]).sum()) / idcg if idcg > 0 else 1.0


def _average_precision(order: list[int], true: np.ndarray) -> float:
    relevant = (true[order] > np.median(true)).astype(float)
    if relevant.sum() == 0:
        return 1.0
    precisions = relevant.cumsum() / np.arange(1, len(relevant) + 1)
    return float((precisions * relevant).sum() / relevant.sum())


def _reciprocal_rank(order: list[int], true: np.ndarray) -> float:
    return 1.0 / (order.index(int(np.argmax(true))) + 1)


def _pairwise_agreement(pred: np.ndarray, true: np.ndarray) -> float:
    """Authors' explrank.metrics.pairwise (strict): gold-tied pairs skipped, predicted ties wrong."""
    correct = total = 0
    for i in range(len(true)):
        for j in range(i + 1, len(true)):
            if true[i] == true[j]:
                continue
            total += 1
            correct += pred[i] != pred[j] and (pred[i] > pred[j]) == (true[i] > true[j])
    return correct / total if total else 1.0


def ranking_metrics(preds: list[dict[str, Any]], trues: list[np.ndarray], cids: list[list[str]]) -> dict[str, Any]:
    ndcg = {k: [] for k in (1, 3, 5)}
    ap, rr, pw, rho, tau, sep = [], [], [], [], [], []
    skipped = 0
    for pred, true, ids in zip(preds, trues, cids):
        scores = np.array([pred["scores"][c] for c in ids])
        order = [ids.index(c) for c in pred["ranking"]]
        for k in ndcg:
            ndcg[k].append(_ndcg_at_k(order, true, k))
        ap.append(_average_precision(order, true))
        rr.append(_reciprocal_rank(order, true))
        pw.append(_pairwise_agreement(scores, true))
        if np.std(true) > 0:
            sep.append(float(np.std(scores / (N_LEVELS - 1)) / np.std(true)))
        if np.std(scores) < 1e-8 or np.std(true) < 1e-8:
            skipped += 1
            continue
        rho.append(float(stats.spearmanr(scores, true)[0]))
        tau.append(float(stats.kendalltau(scores, true)[0]))

    def m(xs: list[float]) -> float:
        xs = [x for x in xs if not math.isnan(x)]
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "ndcg@5": m(ndcg[5]),
        "map": m(ap),
        "mrr": m(rr),
        "kendall_tau": m(tau),
        "spearman": m(rho),
        "ndcg@1": m(ndcg[1]),
        "ndcg@3": m(ndcg[3]),
        "pairwise_agreement": m(pw),
        "separation_ratio": m(sep),
        "n_correlation_skipped": skipped,
    }


class ExplRank(Task):
    name = "expl_rank"
    prompt_file = "expl_rank.json"

    def items(self) -> list[Item]:
        rows = read_jsonl(FROZEN / "expl_rank" / "test.jsonl.gz")
        return [
            Item(
                id=r["id"],
                gold=r["gold"],
                fields={
                    "premise": r["premise"],
                    "hypothesis": r["hypothesis"],
                    "label": r["label"],
                    "candidates": r["candidates"],
                    "pairwise200": r["pairwise200"],
                },
            )
            for r in rows
        ]

    def _request(self, item: Item) -> tuple[Request, bool]:
        premise, cut_p = _clip(item.fields["premise"])
        hypothesis, cut_h = _clip(item.fields["hypothesis"])
        template = self.prompt()["question"]
        questions, cut = {}, cut_p or cut_h
        for cand in item.fields["candidates"]:
            text, cut_c = _clip(cand["text"])
            cut = cut or cut_c
            questions[cand["id"]] = {**template, "instructions": template["instructions"].replace("{explanation}", text)}
        state = {"premise": premise, "hypothesis": hypothesis, "relation": item.fields["label"]}
        return Request(item_id=item.id, key=KEY, state=state, questions=questions), cut

    def requests(self, item: Item) -> Iterator[Request]:
        yield self._request(item)[0]

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        ids = [c["id"] for c in item.fields["candidates"]]
        scores = {c: _expected_level(answers[KEY][c]) for c in ids}
        # Descending expected level; exact ties keep the frozen (hash-shuffled,
        # tier-blind) candidate order.
        ranking = sorted(ids, key=lambda c: (-scores[c], ids.index(c)))
        return {"scores": scores, "ranking": ranking}

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        cids = [[c["id"] for c in it.fields["candidates"]] for it in items]
        preds = [predictions[it.id] for it in items]
        true_score = [np.array([it.gold[c]["score"] for c in ids]) for it, ids in zip(items, cids)]
        true_grade = [np.array([float(it.gold[c]["grade"]) for c in ids]) for it, ids in zip(items, cids)]
        sub = [i for i, it in enumerate(items) if it.fields["pairwise200"]]

        primary = ranking_metrics(preds, true_score, cids)
        by_tier = {t: [] for t in TIERS}
        for it, p in zip(items, preds):
            for c, g in it.gold.items():
                by_tier[g["tier"]].append(p["scores"][c])
        gold_first = [it.gold[p["ranking"][0]]["tier"] == "gold" for it, p in zip(items, preds)]
        return {
            "n": len(items),
            **primary,
            "pairwise_agreement_200": ranking_metrics([preds[i] for i in sub], [true_score[i] for i in sub], [cids[i] for i in sub])[
                "pairwise_agreement"
            ],
            "n_pairwise_200": len(sub),
            "vs_tier_order": {k: v for k, v in ranking_metrics(preds, true_grade, cids).items() if k != "separation_ratio"},
            "vs_tier_order_pairwise_agreement_200": ranking_metrics(
                [preds[i] for i in sub], [true_grade[i] for i in sub], [cids[i] for i in sub]
            )["pairwise_agreement"],
            "mean_expected_level_by_tier": {t: float(np.mean(v)) if v else float("nan") for t, v in by_tier.items()},
            "human_gold_ranked_first": float(np.mean(gold_first)) if gold_first else float("nan"),
            "n_truncated": sum(self._request(it)[1] for it in items),
        }


TASKS = {"expl_rank": ExplRank}
