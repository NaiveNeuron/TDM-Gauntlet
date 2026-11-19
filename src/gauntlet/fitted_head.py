"""Frozen decision heads over Jev's full probability vector: the "Jev + fitted head" condition.

A head reads x = log(max(p, floor)) over a task's label order (p = the zero-shot Choice
probabilities) and replaces the argmax:

- "bias": label = argmax_l x_l + b_l                  (per-class log-biases)
- "lr":   label = argmax_l (W x + c)_l                (L2 multinomial logistic regression)

Exact ties go to the earlier label in the task's label order (np.argmax), as the zero-shot
tasks do. Heads are fit only on a paper's released train/dev data (scripts/fit_v3.py) and frozen
to data/frozen/heads/<task>.json before any test run. `HeadTask` wraps the frozen zero-shot task:
it sends exactly the base task's requests, so a test run is served from the cached zero-shot
answers and adds no Jev calls. Its scores are reported as a separate condition, never mixed into
zero-shot rows.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .task import FROZEN, REPO, Item, Request, Task

HEADS = FROZEN / "heads"


@dataclass(frozen=True)
class Head:
    kind: str
    labels: tuple[str, ...]
    floor: float
    bias: np.ndarray | None = None
    coef: np.ndarray | None = None
    intercept: np.ndarray | None = None

    def features(self, P: np.ndarray) -> np.ndarray:
        return np.log(np.maximum(np.asarray(P, dtype=float), self.floor))

    def logits(self, P: np.ndarray) -> np.ndarray:
        X = self.features(P)
        if self.kind == "bias":
            return X + self.bias
        if self.kind == "lr":
            return X @ self.coef.T + self.intercept
        raise ValueError(f"unknown head kind {self.kind!r}")

    def predict(self, P: np.ndarray) -> np.ndarray:
        """Label indices; first maximum wins (earlier label on exact ties)."""
        return np.argmax(self.logits(P), axis=1)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind, "labels": list(self.labels), "floor": self.floor}
        if self.kind == "bias":
            out["bias"] = [round(float(b), 6) for b in self.bias]
        else:
            out["coef"] = [[round(float(v), 6) for v in row] for row in self.coef]
            out["intercept"] = [round(float(v), 6) for v in self.intercept]
        return out

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Head":
        arr = lambda k: None if k not in d else np.array(d[k], dtype=float)  # noqa: E731
        return cls(d["kind"], tuple(d["labels"]), float(d["floor"]), arr("bias"), arr("coef"), arr("intercept"))


def fit_bias(
    P: np.ndarray, y: np.ndarray, labels: Sequence[str], metric: Callable[[np.ndarray, np.ndarray], float],
    floor: float, grid: Sequence[float] = tuple(np.round(np.arange(-4.0, 4.01, 0.25), 2)), rounds: int = 4,
) -> Head:
    """Coordinate ascent on the task metric over per-class log-biases (first label fixed at 0).

    A coordinate moves only on a strict metric gain; among equal-metric grid values the one
    closest to the current value is kept, so uninformative biases stay at 0."""
    X = np.log(np.maximum(P, floor))
    b = np.zeros(len(labels))
    best = metric(y, np.argmax(X + b, axis=1))
    for _ in range(rounds):
        moved = False
        for j in range(1, len(labels)):
            cur = b[j]
            for g in sorted(grid, key=lambda g: abs(g - cur)):
                b[j] = g
                m = metric(y, np.argmax(X + b, axis=1))
                if m > best + 1e-12:
                    best, cur, moved = m, g, True
            b[j] = cur
        if not moved:
            break
    return Head("bias", tuple(labels), floor, bias=b)


def fit_lr(P: np.ndarray, y: np.ndarray, labels: Sequence[str], C: float, balanced: bool, floor: float) -> Head:
    """L2 multinomial logistic regression on log-probabilities (sklearn lbfgs)."""
    from sklearn.linear_model import LogisticRegression

    if len(labels) < 3 or sorted(set(y.tolist())) != list(range(len(labels))):
        raise ValueError("fit_lr needs >= 3 labels, every one present in the fit data")
    X = np.log(np.maximum(P, floor))
    model = LogisticRegression(C=C, class_weight="balanced" if balanced else None, max_iter=5000).fit(X, y)
    return Head("lr", tuple(labels), floor, coef=model.coef_.copy(), intercept=model.intercept_.copy())


def head_path(task_name: str) -> Path:
    return HEADS / f"{task_name}.json"


class HeadTask(Task):
    """`<task>+head`: the frozen zero-shot task, decoded by its frozen head."""

    def __init__(self, base: Task, path: Path) -> None:
        self.base = base
        self.name = f"{base.name}+head"
        self.prompt_file = base.prompt_file
        self.reuses_responses_of = (base.name,)  # identical requests: responses stay under runs/<task>
        self.path = path
        spec = json.loads(path.read_text())
        self.head = Head.from_json(spec["head"])
        if self.head.labels != tuple(base.labels):
            raise ValueError(f"{path}: head labels {self.head.labels} != task labels {tuple(base.labels)}")

    def items(self) -> list[Item]:
        return self.base.items()

    def next_requests(self, item: Item, answers: dict[str, dict[str, Any]]) -> list[Request]:
        return self.base.next_requests(item, answers)

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        zero = self.base.predict(item, answers)
        P = np.array([[zero["probabilities"][lab] for lab in self.head.labels]])
        return {**zero, "label": self.head.labels[int(self.head.predict(P)[0])], "zero_shot_label": zero["label"]}

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        out = self.base.score(items, predictions)
        out["condition"] = "Jev + fitted head (fit on released train/dev only; not zero-shot)"
        out["head"] = {"file": str(self.path.relative_to(REPO)), "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                       "kind": self.head.kind}
        out["n_changed_from_zero_shot"] = sum(predictions[it.id]["label"] != predictions[it.id]["zero_shot_label"] for it in items)
        return out
