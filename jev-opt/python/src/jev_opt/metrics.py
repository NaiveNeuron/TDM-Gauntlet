"""Task metrics via scikit-learn; ECE is the only hand-written metric."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    recall_score,
)

ERROR = "<error>"  # prediction placeholder for failed examples (counts as wrong)


def _preds(pred: list[str | None]) -> list[str]:
    return [p if p is not None else ERROR for p in pred]


def primary(name: str, gold: list[str], pred: list[str | None], labels: list[str], negative: str | None = None) -> float:
    """`negative`: a label excluded from micro-F1 averaging (never a true positive, never recalled)."""
    if not gold:
        return 0.0
    y = _preds(pred)
    if name == "accuracy":
        return float(accuracy_score(gold, y))
    if name == "balanced_accuracy":  # == macro recall over the classes present in gold
        present = [lab for lab in labels if lab in set(gold)]
        return float(recall_score(gold, y, labels=present, average="macro", zero_division=0))
    if name == "macro_f1":
        return float(f1_score(gold, y, labels=labels, average="macro", zero_division=0))
    if name == "micro_f1":
        positive = [lab for lab in labels if lab != negative]
        return float(f1_score(gold, y, labels=positive, average="micro", zero_division=0))
    if name == "qwk":
        index = {lab: i for i, lab in enumerate(labels)}
        # failed predictions count as the farthest level from gold
        yi = [index[p] if p in index else (0 if index[g] >= len(labels) / 2 else len(labels) - 1) for g, p in zip(gold, y)]
        return float(cohen_kappa_score([index[g] for g in gold], yi, weights="quadratic", labels=list(range(len(labels)))))
    raise ValueError(f"unknown metric {name!r}")


def example_quality(metric: str, gold: str, pred: str | None, labels: list[str]) -> float:
    """Per-example quality in [0, 1]: ordinal partial credit for qwk, else correctness."""
    if pred is None:
        return 0.0
    if metric == "qwk":
        return 1.0 - abs(labels.index(gold) - labels.index(pred)) / (len(labels) - 1)
    return float(gold == pred)


def example_brier(gold: str, dist: dict[str, float], labels: list[str]) -> float:
    """Half the squared error of one probability vector (in [0, 1]); matches sklearn's multiclass scaling."""
    return 0.5 * sum((dist.get(lab, 0.0) - (lab == gold)) ** 2 for lab in labels)


def ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of top-label confidence (equal-width bins)."""
    if len(confidence) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def compute(gold: list[str], pred: list[str | None], dists: list[dict[str, float]], labels: list[str], primary_metric: str, negative: str | None = None) -> dict[str, Any]:
    y = _preds(pred)
    n = len(gold)
    if n == 0:
        return {"n": 0}
    probs = np.array([[d.get(lab, 0.0) for lab in labels] for d in dists], dtype=float)
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    conf_idx = probs.argmax(axis=1)
    sorted_labels = sorted(labels)
    sort_idx = [labels.index(lab) for lab in sorted_labels]
    prec, rec, f1, support = precision_recall_fscore_support(gold, y, labels=labels, zero_division=0)
    out: dict[str, Any] = {
        "n": n,
        "primary_metric": primary_metric,
        "primary": primary(primary_metric, gold, pred, labels, negative),
        "accuracy": primary("accuracy", gold, pred, labels),
        "balanced_accuracy": primary("balanced_accuracy", gold, pred, labels),
        "macro_f1": primary("macro_f1", gold, pred, labels),
        "micro_f1": primary("micro_f1", gold, pred, labels),
        # sklearn assumes y_prob columns follow the SORTED label order.
        "log_loss": float(log_loss(gold, probs[:, sort_idx], labels=sorted_labels)),
        "brier": float(brier_score_loss(gold, probs[:, sort_idx], labels=sorted_labels, scale_by_half=True)),
        "ece": ece(probs[np.arange(n), conf_idx], np.array([labels[i] == g for i, g in zip(conf_idx, gold)], dtype=float)),
        "errors": sum(p is None for p in pred),
        "confusion": {"labels": labels, "matrix": confusion_matrix(gold, y, labels=labels).tolist()},
        "per_class": {
            lab: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, lab in enumerate(labels)
        },
    }
    if primary_metric == "qwk":
        out["qwk"] = out["primary"]
    return out
