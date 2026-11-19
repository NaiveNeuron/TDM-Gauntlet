"""Numerical fitting on train outputs (architecture §7): no LLM, no new Jev calls.

Thresholds / biases / cutoffs are fit by coordinate search over a grid, re-decoding the
recorded answers of an exhaustive train run (`execute.redecode`); temperature by
minimizing train log loss (scipy). Nodes inside stage decision subtrees are frozen,
because changing them would change derived state and need new Jev calls.
With `class_bias`, every Choice outside decision subtrees also gets per-option additive
log-prob biases (class calibration against systematic over/under-prediction of a label).
Readouts (§4.2) are fit first: L2 logistic regression (`lr`) or ridge regression plus cutoffs
(`ridge`) on the node's view features, hyperparameters chosen by k-fold cross-validation of the
task's primary metric on the fit examples (optionally grouped, e.g. by document).
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .execute import ExampleResult, MissingAnswer, _Run, combine, ordinal_vote, redecode, ridge_decode
from .ir import BranchNode, ChoiceNode, OneVsRestNode, Program, Readout, ScoreNode, default_cutoffs
from .metrics import primary
from .task import Example

LR_C = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]  # inverse L2 strength on standardized features
RIDGE_ALPHA = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
ABSENT_CLASS_B = -50.0  # intercept of a readout class never seen in the fit data (probability ~0)

PROB_GRID = [round(float(x), 2) for x in np.arange(0.05, 0.951, 0.05)]
BIAS_GRID = [round(float(x), 2) for x in np.arange(-2.0, 2.001, 0.25)]  # logit-space biases
CLASS_BIAS_GRID = [round(float(x), 2) for x in np.arange(-3.0, 3.001, 0.25)]  # Choice / ensemble log-prob biases

Getter = Callable[[Program], float]
Setter = Callable[[Program, float], None]


@dataclass(frozen=True)
class CVSpec:
    """Cross-validation of readouts: `folds`-fold, shuffled with `seed`, stratified by label and,
    with `group` (a regex whose first group extracts a key from the example id), grouped."""

    folds: int = 5
    seed: int = 0
    group: str | None = None

    def groups(self, examples: list[Example]) -> list[str] | None:
        if not self.group:
            return None
        pattern = re.compile(self.group)
        return [(m.group(1) if (m := pattern.search(ex.id)) else ex.id) for ex in examples]

    def splits(self, gold: list[str], examples: list[Example]) -> list[tuple[np.ndarray, np.ndarray]]:
        n = len(gold)
        k = min(self.folds, n)
        if k < 2:
            return []
        groups = self.groups(examples)
        dummy = np.zeros(n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # classes rarer than k: sklearn warns, splits anyway
            if groups is not None:
                k = min(k, len(set(groups)))
                return list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=self.seed).split(dummy, gold, groups))
            return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=self.seed).split(dummy, gold))


def readout_kind(metric: str) -> str:
    """`ridge` onto the ordinal label index for QWK tasks, else multinomial `lr`."""
    return "ridge" if metric == "qwk" else "lr"


def with_readouts(program: Program, metric: str) -> Program:
    """Copy with an unfitted single-view readout on every main-tree question node whose outcomes
    are all labels and that has none yet (routing nodes keep their argmax routing)."""
    q = program.model_copy(deep=True)
    skip = q.decision_nodes() | q.view_nodes()
    kind = readout_kind(metric)
    for nid, node in q.nodes.items():
        if nid in skip or not isinstance(node, (ChoiceNode, OneVsRestNode, ScoreNode)) or node.readout is not None:
            continue
        targets = q.targets(nid)
        if any(t.label is None for t in targets):
            continue
        outs = {t.label for t in targets} | ({node.aggregation.abstain} if isinstance(node, OneVsRestNode) and node.aggregation.abstain else set())
        classes = [lab for lab in q.labels if lab in outs]
        if kind == "ridge" and len(classes) < 2:
            continue
        node.readout = Readout(kind=kind, views=[nid], classes=classes)
        if isinstance(node, ChoiceNode):
            node.biases = None
    return q


def readout_design(program: Program, nid: str, pairs: list[tuple[Example, ExampleResult]]) -> tuple[np.ndarray, list[Example]]:
    """Feature rows of readout node `nid` for every recorded example that reached it."""
    rows: list[list[float]] = []
    kept: list[Example] = []
    for ex, res in pairs:
        if res.error is not None:
            continue
        run = _Run(program, ex, None, False, dict(res.memo), None, res.demos)
        try:
            rows.append(run.features(nid, {**ex.fields, **res.derived}))
        except (MissingAnswer, KeyError):
            continue
        kept.append(ex)
    return np.array(rows, dtype=float), kept


def _qwk(y: np.ndarray, p: np.ndarray, k: int) -> float:
    observed = np.bincount(y * k + p, minlength=k * k).reshape(k, k).astype(float)
    weights = np.subtract.outer(np.arange(k), np.arange(k)) ** 2
    expected = np.outer(observed.sum(1), observed.sum(0)) / observed.sum()
    denom = (weights * expected).sum()
    return 1.0 - (weights * observed).sum() / denom if denom > 0 else 0.0


def fit_cutoffs(yhat: np.ndarray, y: np.ndarray, k: int, rounds: int = 2) -> tuple[list[float], float]:
    """Ascending cutoffs on [0, k-1] mapping regression outputs to class indexes, by coordinate
    search maximizing QWK (the ordinal tasks' metric)."""
    grid = np.round(np.arange(0.0, k - 1 + 1e-9, 0.05), 2)
    cuts = [j + 0.5 for j in range(k - 1)]
    band = lambda c: np.searchsorted(np.array(c), yhat, side="right")  # noqa: E731  (#cutoffs <= y)
    best = _qwk(y, band(cuts), k)
    for _ in range(rounds):
        improved = False
        for i in range(k - 1):
            lo = cuts[i - 1] if i else 0.0
            hi = cuts[i + 1] if i + 1 < k - 1 else k - 1.0
            for v in grid[(grid >= lo) & (grid <= hi)]:
                trial = cuts[:i] + [float(v)] + cuts[i + 1 :]
                s = _qwk(y, band(trial), k)
                if s > best + 1e-12:
                    best, cuts, improved = s, trial, True
        if not improved:
            break
    return [round(c, 2) for c in cuts], best


def _lr_model(c: float, weight: str | None) -> Any:
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight=weight, max_iter=3000))


def _lr_proba(model: Any, x: np.ndarray, classes: list[str]) -> np.ndarray:
    """Probabilities in `classes` order (classes unseen by the model get 0)."""
    out = np.zeros((len(x), len(classes)))
    proba = model.predict_proba(x)
    for j, c in enumerate(model.classes_):
        out[:, classes.index(c)] = proba[:, j]
    return out


def _unscaled(model: Any) -> tuple[np.ndarray, np.ndarray]:
    """(coef, intercept) acting on raw features: the pipeline's standardization folded in."""
    scaler, est = model[0], model[-1]
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    coef = np.atleast_2d(est.coef_) / scale
    return coef, np.atleast_1d(est.intercept_) - coef @ scaler.mean_


def fit_readout(readout: Readout, x: np.ndarray, examples: list[Example], metric: str, labels: list[str], negative: str | None, cv: CVSpec) -> tuple[Readout, dict[str, Any], dict[str, tuple[str, dict[str, float]]]]:
    """Fit `readout` on rows `x` (gold = example labels): hyperparameters by `cv` on the primary
    metric, then a refit on all rows. Returns the fitted readout, a report and the out-of-fold
    `(pred, dist)` per example id (the cross-validated evidence population search ranks by)."""
    classes = list(readout.classes)
    keep = [i for i, ex in enumerate(examples) if ex.label in classes]
    x, examples = x[keep], [examples[i] for i in keep]
    gold = [ex.label for ex in examples]
    report: dict[str, Any] = {"kind": readout.kind, "n": len(gold), "features": int(x.shape[1]) if x.ndim == 2 else 0}
    if len(set(gold)) < 2:
        raise ValueError(f"readout needs >= 2 gold classes among its fit examples, got {sorted(set(gold))}")
    splits = cv.splits(gold, examples)
    scored = [lab for lab in labels if lab in set(gold) | set(classes)]
    ids = [ex.id for ex in examples]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if readout.kind == "lr":
            fitted, oof = _fit_lr(readout, x, gold, classes, splits, metric, scored, negative, report)
        else:
            fitted, oof = _fit_ridge(readout, x, gold, classes, splits, report)
    return fitted, report, dict(zip(ids, oof))


def _fit_lr(readout: Readout, x: np.ndarray, gold: list[str], classes: list[str], splits: list, metric: str, scored: list[str], negative: str | None, report: dict[str, Any]) -> tuple[Readout, list[tuple[str, dict[str, float]]]]:
    y = np.array([classes.index(g) for g in gold])
    best: tuple[float, float] | None = None
    choice: tuple[float, str | None] = (1.0, None)
    best_oof = np.full((len(gold), len(classes)), 1.0 / len(classes))
    for weight in (None, "balanced"):
        for c in LR_C:
            oof = np.zeros((len(gold), len(classes)))
            for tr, te in splits:
                if len(set(y[tr])) < 2:
                    oof[te] = np.bincount(y[tr], minlength=len(classes)) / len(tr)
                    continue
                oof[te] = _lr_proba(_lr_model(c, weight).fit(x[tr], [gold[i] for i in tr]), x[te], classes)
            preds = [classes[i] for i in oof.argmax(axis=1)]
            key = (primary(metric, gold, preds, scored, negative), -float(np.mean(-np.log(np.clip(oof[np.arange(len(y)), y], 1e-12, 1)))))
            if best is None or key > best:
                best, choice, best_oof = key, (c, weight), oof
    c, weight = choice
    model = _lr_model(c, weight).fit(x, gold)
    coef, intercept = _unscaled(model)
    seen = list(model.classes_)
    W, b = [], []
    for cls in classes:
        if cls not in seen:
            W.append([0.0] * x.shape[1])
            b.append(ABSENT_CLASS_B)
        elif len(seen) == 2:  # sklearn's binary model: one row, the logit of seen[1]
            W.append([0.0] * x.shape[1] if cls == seen[0] else [float(v) for v in coef[0]])
            b.append(0.0 if cls == seen[0] else float(intercept[0]))
        else:
            W.append([float(v) for v in coef[seen.index(cls)]])
            b.append(float(intercept[seen.index(cls)]))
    report.update(C=c, class_weight=weight or "none", folds=len(splits), cv=round(best[0], 5) if best else None, cv_log_loss=round(-best[1], 5) if best else None)
    fitted = readout.model_copy(update={"W": [[round(v, 6) for v in row] for row in W], "b": [round(v, 6) for v in b], "fit": {k: report[k] for k in ("C", "class_weight", "folds", "cv", "n")}})
    oof = [(classes[int(row.argmax())], {cls: float(p) for cls, p in zip(classes, row)}) for row in best_oof]
    return fitted, oof


def _fit_ridge(readout: Readout, x: np.ndarray, gold: list[str], classes: list[str], splits: list, report: dict[str, Any]) -> tuple[Readout, list[tuple[str, dict[str, float]]]]:
    y = np.array([classes.index(g) for g in gold])
    k = len(classes)
    best: tuple[float, float, list[float], np.ndarray] | None = None
    for alpha in RIDGE_ALPHA:
        yhat = np.zeros(len(y))
        for tr, te in splits:
            yhat[te] = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x[tr], y[tr]).predict(x[te])
        yhat = np.clip(yhat, 0.0, k - 1.0)
        cuts, q = fit_cutoffs(yhat, y, k)
        if best is None or q > best[1]:
            best = (alpha, q, cuts, yhat)
    assert best is not None
    alpha, q, cuts, yhat = best
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x, y)
    coef, intercept = _unscaled(model)
    sigma = round(max(float(np.std(yhat - y)), 0.05), 4)
    report.update(alpha=alpha, folds=len(splits), cv=round(q, 5), sigma=sigma)
    fitted = readout.model_copy(update={"W": [[round(float(v), 6) for v in coef[0]]], "b": [round(float(intercept[0]), 6)], "cutoffs": cuts, "sigma": sigma, "fit": {k_: report[k_] for k_ in ("alpha", "folds", "cv", "n")}})
    oof = [ridge_decode(float(v), cuts, sigma, classes)[::-1] for v in yhat]
    return fitted, [(pred, dist) for pred, dist in oof]


def readout_nodes(program: Program) -> list[str]:
    frozen = program.decision_nodes()
    return [nid for nid, n in program.nodes.items() if getattr(n, "readout", None) is not None and nid not in frozen]


def fit_readouts(program: Program, pairs: list[tuple[Example, ExampleResult]], metric: str, negative: str | None, cv: CVSpec) -> tuple[Program, dict[str, Any], dict[str, dict[str, tuple[str, dict[str, float]]]]]:
    """Fit every readout of `program` (independently; each on the examples that reached it)."""
    fitted = program.model_copy(deep=True)
    reports: dict[str, Any] = {}
    oof: dict[str, dict[str, tuple[str, dict[str, float]]]] = {}
    for nid in readout_nodes(fitted):
        node = fitted.nodes[nid]
        x, examples = readout_design(fitted, nid, pairs)
        node.readout, reports[nid], oof[nid] = fit_readout(node.readout, x, examples, metric, fitted.labels, negative, cv)
    return fitted, reports, oof


def parameters(program: Program, class_bias: bool = False) -> list[tuple[str, list[float], Getter, Setter]]:
    """(name, grid, get, set) for every fittable numeric parameter outside decision subtrees."""
    frozen = program.decision_nodes() | program.view_nodes() | set(readout_nodes(program))
    out: list[tuple[str, list[float], Getter, Setter]] = []
    for nid, node in program.nodes.items():
        if nid in frozen:
            continue
        if isinstance(node, ChoiceNode) and class_bias:
            for key in node.options:
                out.append((
                    f"{nid}.biases.{key}",
                    CLASS_BIAS_GRID,
                    lambda p, n=nid, k=key: (p.nodes[n].biases or {}).get(k, 0.0),
                    lambda p, v, n=nid, k=key: _set_bias(p, n, k, v),
                ))
        elif isinstance(node, OneVsRestNode):
            rule = node.aggregation.rule
            grid, default = (PROB_GRID, 0.5) if rule == "threshold" else (BIAS_GRID, 0.0)
            for key in node.options:
                out.append((
                    f"{nid}.thresholds.{key}",
                    grid,
                    lambda p, n=nid, k=key, d=default: p.nodes[n].aggregation.thresholds.get(k, d),
                    lambda p, v, n=nid, k=key: p.nodes[n].aggregation.thresholds.__setitem__(k, v),
                ))
        elif isinstance(node, ScoreNode) and len(node.outcomes) > 1:
            span = len(node.levels) - 1
            grid = [round(float(x), 2) for x in np.arange(0.0, span + 1e-9, 0.1)]
            for i in range(len(node.outcomes) - 1):
                out.append((f"{nid}.cutoffs.{i}", grid, lambda p, n=nid, i=i: _cutoffs(p, n)[i], lambda p, v, n=nid, i=i: _set_cutoff(p, n, i, v)))
        elif isinstance(node, BranchNode):
            out.append((f"{nid}.threshold", PROB_GRID, lambda p, n=nid: p.nodes[n].threshold, lambda p, v, n=nid: setattr(p.nodes[n], "threshold", v)))
    return out


def _set_bias(p: Program, nid: str, key: str, v: float) -> None:
    node = p.nodes[nid]
    biases = dict(node.biases or {})
    biases[key] = v
    node.biases = {k: b for k, b in biases.items() if b != 0.0} or None


def _cutoffs(p: Program, nid: str) -> list[float]:
    node = p.nodes[nid]
    if node.cutoffs is None:
        node.cutoffs = [round(c, 2) for c in default_cutoffs(node)]
    return node.cutoffs


def _set_cutoff(p: Program, nid: str, i: int, v: float) -> None:
    cuts = _cutoffs(p, nid)
    lo = cuts[i - 1] if i > 0 else 0.0
    hi = cuts[i + 1] if i + 1 < len(cuts) else float(len(p.nodes[nid].levels) - 1)
    cuts[i] = min(max(v, lo), hi)


def _decode_all(program: Program, pairs: list[tuple[Example, ExampleResult]]) -> list[ExampleResult]:
    out = []
    for ex, res in pairs:
        try:
            out.append(redecode(program, ex, res.memo, res.demos))
        except MissingAnswer:
            continue
    return out


def fit_temperature(labels: list[str], dists: list[dict[str, float]], gold: list[str]) -> float:
    """Temperature minimizing train log loss of `dists` (scaled as p ** (1 / t))."""
    probs = np.array([[max(d.get(lab, 0.0), 1e-12) for lab in labels] for d in dists])
    y = np.array([labels.index(g) for g in gold])

    def nll(t: float) -> float:
        logits = np.log(probs) / t
        logits -= logits.max(axis=1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        return float(-logp[np.arange(len(y)), y].mean())

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return round(float(res.x), 3)


def fit_program(program: Program, pairs: list[tuple[Example, ExampleResult]], metric: str, calibrate: bool, rounds: int = 2, negative: str | None = None, class_bias: bool = False, cv: CVSpec | None = None) -> tuple[Program, dict[str, Any]]:
    """Return a fitted copy of `program` and a report. `pairs` come from an exhaustive train run.
    Readouts are fit first (by `cv`); the remaining parameters by coordinate search."""
    pairs = [(ex, r) for ex, r in pairs if r.error is None]
    fitted = program.model_copy(deep=True)
    report: dict[str, Any] = {"n": len(pairs), "params": {}}
    if not pairs:
        return fitted, report
    labels = program.labels
    if readout_nodes(fitted):
        fitted, report["readouts"], _ = fit_readouts(fitted, pairs, metric, negative, cv or CVSpec())

    def score(p: Program) -> float:
        decoded = _decode_all(p, pairs)
        return primary(metric, [r.gold for r in decoded], [r.pred for r in decoded], labels, negative)

    params = parameters(fitted, class_bias)
    best = score(fitted)
    report["before"] = best
    for _ in range(rounds if params else 0):
        improved = False
        for name, grid, get, put in params:
            current = get(fitted)
            for value in grid:
                if value == current:
                    continue
                trial = fitted.model_copy(deep=True)
                put(trial, value)
                s = score(trial)
                if s > best + 1e-12:
                    best, fitted, improved = s, trial, True
            report["params"][name] = get(fitted)
        if not improved:
            break
    report["after"] = best

    if calibrate:
        fitted.temperature = 1.0
        decoded = _decode_all(fitted, pairs)
        fitted.temperature = fit_temperature(labels, [r.dist for r in decoded], [r.gold for r in decoded])
        report["temperature"] = fitted.temperature
    return fitted, report


def fit_label_bias(labels: list[str], member_dists: list[list[dict[str, float]]], gold: list[str], metric: str, negative: str | None = None, rounds: int = 3) -> tuple[dict[str, float], float, dict[str, Any]]:
    """Per-label log-prob biases (coordinate search) + output temperature for an ensemble.

    `member_dists[i]` holds the members' distributions for train example i."""
    bias: dict[str, float] = {}

    def score(b: dict[str, float]) -> float:
        preds = [combine(labels, dists, b, 1.0)[1] for dists in member_dists]
        return primary(metric, gold, preds, labels, negative)

    best = score(bias)
    report: dict[str, Any] = {"n": len(gold), "before": best}
    for _ in range(rounds):
        improved = False
        for lab in labels:
            for value in CLASS_BIAS_GRID:
                if value == bias.get(lab, 0.0):
                    continue
                trial = {**bias, lab: value}
                s = score(trial)
                if s > best + 1e-12:
                    best, bias, improved = s, trial, True
        if not improved:
            break
    bias = {k: v for k, v in bias.items() if v != 0.0}
    report["after"] = best
    decoded = [combine(labels, dists, bias, 1.0)[0] for dists in member_dists]
    temperature = fit_temperature(labels, decoded, gold) if gold else 1.0
    report["temperature"] = temperature
    return bias, temperature, report


def fit_ordinal_cutoffs(labels: list[str], member_preds: list[list[str]], gold: list[str], metric: str, rounds: int = 3) -> tuple[list[float], dict[str, Any]]:
    """Cutoffs over the members' mean predicted label index (coordinate search, grid step 0.1)."""
    span = len(labels) - 1
    grid = [round(float(x), 2) for x in np.arange(0.0, span + 1e-9, 0.1)]
    cuts = [i + 0.5 for i in range(span)]

    def score(c: list[float]) -> float:
        return primary(metric, gold, [ordinal_vote(labels, preds, c) for preds in member_preds], labels)

    best = score(cuts)
    report: dict[str, Any] = {"n": len(gold), "before": best}
    for _ in range(rounds):
        improved = False
        for i in range(span):
            lo, hi = (cuts[i - 1] if i else 0.0), (cuts[i + 1] if i + 1 < span else float(span))
            for value in grid:
                if value == cuts[i] or not lo <= value <= hi:
                    continue
                trial = [*cuts[:i], value, *cuts[i + 1 :]]
                s = score(trial)
                if s > best + 1e-12:
                    best, cuts, improved = s, trial, True
        if not improved:
            break
    report["after"] = best
    return cuts, report
