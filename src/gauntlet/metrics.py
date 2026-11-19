"""Metrics used by the papers' primary evaluations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score


def accuracy(gold: Sequence, pred: Sequence) -> float:
    return float(accuracy_score(gold, pred))


def macro_f1(gold: Sequence, pred: Sequence, labels: Sequence | None = None) -> float:
    return float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0))


def micro_f1(gold: Sequence, pred: Sequence, labels: Sequence | None = None) -> float:
    return float(f1_score(gold, pred, labels=labels, average="micro", zero_division=0))


def weighted_f1(gold: Sequence, pred: Sequence, labels: Sequence | None = None) -> float:
    return float(f1_score(gold, pred, labels=labels, average="weighted", zero_division=0))


def per_class_f1(gold: Sequence, pred: Sequence, labels: Sequence) -> dict[str, float]:
    scores = f1_score(gold, pred, labels=labels, average=None, zero_division=0)
    return {str(label): float(s) for label, s in zip(labels, scores)}


def roc_auc(gold_binary: Sequence[int], positive_prob: Sequence[float]) -> float:
    return float(roc_auc_score(gold_binary, positive_prob))


def qwk(gold: Sequence[int], pred: Sequence[int], labels: Sequence[int] | None = None) -> float:
    """Quadratic weighted kappa on integer-coded ordinal labels.

    Pass the full ordinal scale as `labels`; otherwise levels absent from both
    gold and predictions are dropped and the quadratic weights shrink.
    """
    return float(cohen_kappa_score(gold, pred, labels=labels, weights="quadratic"))


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int = 10) -> float:
    """trec_eval `ndcg_cut.k`: linear gain, ideal ranking over all judged docs."""
    gains = [max(qrels.get(doc, 0), 0) for doc in ranking[:k]]
    ideal = sorted((max(v, 0) for v in qrels.values()), reverse=True)[:k]
    denom = dcg(ideal)
    return dcg(gains) / denom if denom > 0 else 0.0


def mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if len(xs) else float("nan")
