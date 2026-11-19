"""Paired bootstrap CIs of Jev's zero-shot audit scores vs the papers' published numbers.

For each of the 46 comparable audit test sets (`paper_tables.rows()` with a task-specific number),
resample the unit the paper's metric is defined over (items, queries, documents, ...) from the
saved predictions (`runs/<task>/predictions.jsonl`), recompute the paper-table metric
(`paper_tables.jev_value` path of the task's own `score()`), and classify Jev against the paper's
task-specific number and its LLM baseline (published numbers carry no uncertainty):

  above         95% percentile CI lower bound > v
  below         95% CI upper bound < v
  parity        neither, and the 90% CI lies within [v - 0.05, v + 0.05] (post hoc margin)
  inconclusive  otherwise

B = 10,000 resamples, `numpy.random.default_rng(0)` re-seeded per task; resamples are sorted unit
indices (a duplicated unit's copies are adjacent, which fixes tie-breaking order). Per task the
fast resample metric is checked against the saved value on the full sample and against the
task's `score()` (or, where `score()` cannot take resampled units, the module's metric code) on
3 independent resamples built with unique duplicate ids (`<id>#<k>`, predictions copied).

Offline only: no Jev calls (network patched to raise), no test-split consumption beyond the
saved test predictions. Run from the repo root:

    uv run python scripts/framing_audit_ci.py
"""

from __future__ import annotations

import dataclasses
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import httpx  # noqa: E402


def _no_network(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("framing_audit_ci is offline: a code path tried to reach the network / Jev API")


httpx.Client.send = _no_network
httpx.AsyncClient.send = _no_network
from gauntlet import jev as _jev  # noqa: E402

_jev.JevClient.decide = _no_network
_jev.JevClient._post_with_retry = _no_network

import paper_tables  # noqa: E402
from gauntlet.task import Item, read_jsonl  # noqa: E402
from gauntlet.tasks import REGISTRY  # noqa: E402
from gauntlet.tasks import biorc800 as m_biorc  # noqa: E402
from gauntlet.tasks import cicc as m_cicc  # noqa: E402
from gauntlet.tasks import contempo as m_contempo  # noqa: E402
from gauntlet.tasks import dress as m_dress  # noqa: E402
from gauntlet.tasks import edit_intent as m_edit  # noqa: E402
from gauntlet.tasks import ezstance as m_ez  # noqa: E402
from gauntlet.tasks import factpico as m_factpico  # noqa: E402
from gauntlet.tasks import fenice as m_fenice  # noqa: E402
from gauntlet.tasks import legalseg as m_legalseg  # noqa: E402
from gauntlet.tasks import lifetox as m_lifetox  # noqa: E402
from gauntlet.tasks import made as m_made  # noqa: E402
from gauntlet.tasks import medreadme as m_medreadme  # noqa: E402
from gauntlet.tasks import mismatched as m_mismatched  # noqa: E402
from gauntlet.tasks import prp_graph as m_prp  # noqa: E402
from gauntlet.tasks import scalar_constructs as m_scalar  # noqa: E402
from gauntlet import metrics  # noqa: E402

OUT = ROOT / "results" / "framing_audit_ci.json"
B = 10_000
SEED = 0
MARGIN = 0.05
CI = (0.95, 0.90)
N_CHECK = 3  # resamples checked against score() / module code, drawn from a separate stream
CHECK_SEED = 12345
TOL = 1e-9


@dataclass
class Spec:
    unit: str
    n_units: int
    stat: Callable[[np.ndarray], float]  # sorted unit indices (with repeats) -> metric
    reference: Callable[[np.ndarray], float]  # same metric through score() / module code
    method: str
    strata: list[np.ndarray] | None = None  # resample within these unit groups
    ordered: bool = False  # metric depends on unit order (tie-breaking / CV folds): sort draws
    ref_name: str = "score()"


def value_from(task: str, scores: dict) -> float:
    """`paper_tables.jev_value` applied to an in-memory scores dict."""
    orig = paper_tables.scores
    paper_tables.scores = lambda _t: scores
    try:
        return paper_tables.jev_value(task)
    finally:
        paper_tables.scores = orig


def load_predictions(task: str) -> dict[str, Any]:
    d = ROOT / "runs" / task
    path = d / "predictions.jsonl" if (d / "predictions.jsonl").exists() else d / "predictions.jsonl.gz"
    return {r["item_id"]: r["prediction"] for r in read_jsonl(path)}


def duplicate(units: list[Item], idx: np.ndarray, preds: dict[str, Any]) -> tuple[list[Item], dict[str, Any]]:
    """Resampled items with unique ids `<id>#<k>` and copied predictions."""
    seen: Counter[int] = Counter()
    items, out = [], {}
    for i in idx.tolist():
        it = units[i]
        new = f"{it.id}#{seen[i]}"
        seen[i] += 1
        items.append(dataclasses.replace(it, id=new))
        out[new] = preds[it.id]
    return items, out


def score_ref(name: str, task: Any, units: list[Item], preds: dict[str, Any], fixed: list[Item] = ()) -> Callable:
    fixed = list(fixed)
    fixed_preds = {it.id: preds[it.id] for it in fixed}

    def ref(idx: np.ndarray) -> float:
        items, p = duplicate(units, idx, preds)
        return value_from(name, task.score(fixed + items, {**fixed_preds, **p}))

    return ref


# ---------------------------------------------------------------- vectorised metric pieces


class Confusion:
    """Confusion counts of resampled (gold, pred) pairs over a fixed label index."""

    def __init__(self, gold: list, pred: list, labels: list, group: np.ndarray | None = None, n_groups: int = 1):
        labs = list(dict.fromkeys([*labels, *sorted(set(gold) | set(pred), key=str)]))
        self.index = {lab: i for i, lab in enumerate(labs)}
        self.k = len(labs)
        g = np.array([self.index[x] for x in gold])
        p = np.array([self.index[x] for x in pred])
        self.n_groups = n_groups
        grp = np.zeros(len(g), dtype=np.int64) if group is None else group
        self.code = grp * self.k * self.k + g * self.k + p

    def mat(self, idx: np.ndarray) -> np.ndarray:
        """(n_groups, k, k) counts; rows gold, columns pred."""
        c = np.bincount(self.code[idx], minlength=self.n_groups * self.k * self.k)
        return c.reshape(self.n_groups, self.k, self.k)

    def cols(self, labels: list) -> np.ndarray:
        return np.array([self.index[lab] for lab in labels])


def f1_labels(c: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Per-label F1 (zero division -> 0) from one (k, k) confusion matrix."""
    tp = np.diag(c)[cols].astype(float)
    den = c.sum(axis=0)[cols] + c.sum(axis=1)[cols]
    return np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)


def f1_counts(tp: float, n_pred: float, n_gold: float) -> float:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def rowwise_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    den = np.sqrt((xc * xc).sum(axis=1) * (yc * yc).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (xc * yc).sum(axis=1) / den, np.nan)


# ---------------------------------------------------------------- per-task specs


def spec_accuracy(name, task, items, preds):
    correct = np.array([preds[it.id]["label"] == it.gold for it in items], dtype=float)
    return Spec("item", len(items), lambda idx: float(correct[idx].mean()), score_ref(name, task, items, preds),
                "item bootstrap; accuracy = mean correctness")


def spec_macro(name, task, items, preds, labels):
    conf = Confusion([it.gold for it in items], [preds[it.id]["label"] for it in items], list(labels))
    cols = conf.cols(list(labels))
    return Spec("item", len(items), lambda idx: float(f1_labels(conf.mat(idx)[0], cols).mean()),
                score_ref(name, task, items, preds), f"item bootstrap; macro-F1 over the {len(labels)} task labels")


def spec_ezstance_b(name, task, items, preds):
    domains = list(m_ez.DOMAINS)
    dom = np.array([domains.index(it.fields["domain"]) for it in items])
    labels = list(m_ez.LABELS)
    conf = Confusion([it.gold for it in items], [preds[it.id]["label"] for it in items], labels, dom, len(domains))
    cols = conf.cols(labels)

    def stat(idx):
        c = conf.mat(idx)
        return float(metrics.mean([float(f1_labels(c[d], cols).mean()) for d in range(len(domains))]))

    return Spec("item", len(items), stat, score_ref(name, task, items, preds),
                "item bootstrap stratified by the 8 zero-shot domains; mean of per-domain macro-F1",
                strata=[np.flatnonzero(dom == d) for d in range(len(domains))])


def spec_contempo(name, task, items, preds):
    positive = [lab for lab in task.labels if lab != m_contempo.NEGATIVE]
    conf = Confusion([it.gold for it in items], [preds[it.id]["label"] for it in items], list(task.labels))
    cols = conf.cols(positive)

    def stat(idx):
        c = conf.mat(idx)[0]
        return f1_counts(float(np.diag(c)[cols].sum()), float(c[:, cols].sum()), float(c[cols, :].sum()))

    return Spec("item", len(items), stat, score_ref(name, task, items, preds),
                "item bootstrap; micro-F1 over non-VAGUE labels")


def spec_nav(name, task, items, preds):
    parts = []
    for seed in task.partitions:
        labels = task.labels(seed)
        conf = Confusion([task.gold(it, seed) for it in items], [preds[it.id][seed]["label"] for it in items], labels)
        parts.append((conf, conf.cols(labels)))

    def stat(idx):
        return float(metrics.mean([float(f1_labels(conf.mat(idx)[0], cols).mean()) for conf, cols in parts]))

    return Spec("item", len(items), stat, score_ref(name, task, items, preds),
                f"item bootstrap; mean over the {len(parts)} ID/OOD partitions of all-class macro-F1")


def spec_pinose(name, task, items, preds):
    gold = np.array([int(bool(it.gold)) for it in items])
    p = np.array([float(preds[it.id]["p_true"]) for it in items])
    return Spec("item", len(items), lambda idx: metrics.roc_auc(gold[idx], p[idx]), score_ref(name, task, items, preds),
                "item bootstrap; ROC-AUC of P(true) (gauntlet.metrics.roc_auc)")


def spec_fenice(name, task, items, preds):
    val = [it for it in items if it.fields["split"] == "val"]
    test = [it for it in items if it.fields["split"] == "test"]
    thr = m_fenice.choose_best_threshold([int(it.gold) for it in val], [float(preds[it.id]["score"]) for it in val])
    gold = np.array([int(it.gold) for it in test])
    pred = np.array([1 if float(preds[it.id]["score"]) > thr else 0 for it in test])
    return Spec("item", len(test), lambda idx: m_fenice.balanced_accuracy(gold[idx], pred[idx]),
                score_ref(name, task, test, preds, fixed=val),
                "test-item bootstrap; decision threshold tuned on the 40 val items held fixed; balanced accuracy")


def spec_factpico(name, task, items, preds):
    gold = np.array([it.gold["avg"] for it in items], dtype=float)
    pred = np.array([preds[it.id]["expected"]["overall"] for it in items], dtype=float)
    return Spec("item", len(items),
                lambda idx: float(m_factpico.kendalltau(gold[idx], pred[idx], variant="b")[0]),
                score_ref(name, task, items, preds), "item (summary) bootstrap; Kendall tau-b, Jev overall vs Avg. PICO-R")


def spec_medreadme(name, task, items, preds):
    groups = list(m_medreadme.GROUPS)
    grp = np.array([groups.index(m_medreadme.SOURCE_GROUP[it.fields["source"]]) for it in items])
    gold = np.array([it.gold for it in items], dtype=float)
    pred = np.array([preds[it.id]["readability"] for it in items], dtype=float)

    def stat(idx):
        g, p, k = gold[idx], pred[idx], grp[idx]
        return float(np.mean([m_medreadme._corr("pearson", g[k == j].tolist(), p[k == j].tolist()) for j in range(len(groups))]))

    return Spec("item", len(items), stat, score_ref(name, task, items, preds),
                "sentence bootstrap stratified by the 7 source groups; mean of per-group Pearson r",
                strata=[np.flatnonzero(grp == j) for j in range(len(groups))])


def spec_dress(name, task, items, preds):
    scale = range(m_dress._code(3 * m_dress.LEVELS[0]), m_dress._code(3 * m_dress.LEVELS[-1]) + 1)
    gold = np.array([it.gold["total"] for it in items], dtype=float)
    pred = np.array([preds[it.id]["total"] for it in items], dtype=float)
    return Spec("item", len(items), lambda idx: m_dress._qwk(gold[idx].tolist(), pred[idx].tolist(), scale),
                score_ref(name, task, items, preds), "essay bootstrap; QWK of total score")


def spec_ttp(name, task, items, preds):
    hit = np.array([len(set(preds[it.id]["ranking"][:1]) & set(it.gold)) for it in items], dtype=float)
    return Spec("query", len(items), lambda idx: float(hit[idx].mean()), score_ref(name, task, items, preds),
                "query (text) bootstrap; P@1 over the 737-technique ranking")


def spec_legalseg(name, task, items, preds):
    labels = list(m_legalseg.LABELS)
    gold = [g for it in items for g in it.gold]
    pred = [p for it in items for p in preds[it.id]["labels"]]
    doc = np.repeat(np.arange(len(items)), [len(it.gold) for it in items])
    conf = Confusion(gold, pred, labels)
    per_doc = np.zeros((len(items), conf.k * conf.k))
    np.add.at(per_doc, (doc, conf.code), 1)
    cols = conf.cols(labels)

    def stat(idx):
        c = per_doc[idx].sum(axis=0).reshape(conf.k, conf.k)
        return float(f1_labels(c, cols).mean())

    return Spec("document", len(items), stat, score_ref(name, task, items, preds),
                "document (judgment) bootstrap; sentence-level macro-F1 over 7 rhetorical roles pooled over resampled documents")


def spec_biorc(name, task, items, preds):
    frozen = json.loads(m_biorc.THRESHOLDS_PATH.read_text())
    thresholds = [float(frozen["thresholds"][lab]) for lab in m_biorc.LABELS]
    per_doc = []
    for it in items:
        y = m_biorc.indicator(it.gold).astype(bool)
        yhat = m_biorc.indicator(m_biorc.apply_thresholds(preds[it.id]["probs"], thresholds)).astype(bool)
        per_doc.append(((y & yhat).sum(), (~y & yhat).sum(), (y & ~yhat).sum()))
    per_doc = np.array(per_doc, dtype=float)

    def stat(idx):
        tp, fp, fn = per_doc[idx].sum(axis=0)
        return float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else 0.0

    return Spec("document", len(items), stat, score_ref(name, task, items, preds),
                "abstract bootstrap; sentence x label micro-F1 with per-label thresholds fit on dev held fixed")


def spec_made(name, task, items, preds):
    y_true, y_pred = m_made.binary_matrices([it.gold for it in items], [preds[it.id]["labels"] for it in items], task.vocab)
    n_labels = y_true.shape[1]
    parts = [np.nonzero(m) for m in (y_true & y_pred, ~y_true & y_pred, y_true & ~y_pred)]

    def stat(idx):
        c = np.bincount(idx, minlength=len(items)).astype(float)
        tp, fp, fn = (np.bincount(cols, weights=c[rows], minlength=n_labels) for rows, cols in parts)
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        den = precision + recall
        return float(np.divide(2 * precision * recall, den, out=np.zeros_like(tp), where=den > 0).mean())

    return Spec("report", len(items), stat, score_ref(name, task, items, preds),
                f"report bootstrap; macro-F1 over all {n_labels} IMDRF codes (absent codes count 0)")


def spec_wikiwd(name, task, items, preds):
    neg = task.inv.negative
    per = np.array([(sum(p == g and p != neg for g, p in zip(it.gold, preds[it.id]["labels"])),
                     sum(p != neg for p in preds[it.id]["labels"]), sum(g != neg for g in it.gold)) for it in items],
                   dtype=float)

    def stat(idx):
        tp, n_pred, n_gold = per[idx].sum(axis=0)
        return f1_counts(tp, n_pred, n_gold)

    return Spec("sentence", len(items), stat, score_ref(name, task, items, preds),
                "sentence bootstrap; micro-F1 over entity pairs, negative label excluded")


def spec_nytfb(name, task, items, preds):
    inv = task.inv
    pos = [(li, lab) for li, lab in enumerate(inv.labels) if lab != inv.negative]
    pos_li = np.array([li for li, _ in pos])
    pos_lab = np.array([lab for _, lab in pos], dtype=object)
    prob, item, row, lab, gold = [], [], [], [], []
    npos = np.zeros(len(items))
    for i, it in enumerate(items):
        p = preds[it.id]
        for r, (g, j) in enumerate(zip(it.gold, p["row_pair"])):
            prob.append(np.asarray(p["pair_probs"][j], dtype=float)[pos_li])
            gold.append(pos_lab == g)
            item.append(np.full(len(pos), i))
            row.append(np.full(len(pos), r))
            lab.append(pos_li)
            npos[i] += g != inv.negative
    prob, item, row, lab, gold = (np.concatenate(x) for x in (prob, item, row, lab, gold))
    order = np.lexsort((lab, row, item, -prob))  # score desc, then (item, row, label) = global row order
    prob, item, gold = prob[order], item[order], gold[order].astype(np.int64)
    level = int(paper_tables.PAPER[name]["metric"].rsplit("_", 1)[1]) / 100  # p_at_10 -> recall 0.1
    # Only the head of the ranking matters for P at 10% recall: keep entries up to 50% recall at c=1
    # (stat() raises if a resample's crossing ever fell outside the head).
    head = int(np.searchsorted(np.cumsum(gold), 0.5 * npos.sum())) + 1
    prob, item, gold = prob[:head], item[:head], gold[:head]
    head = len(prob)
    # Blocks of equal (score, item): a unit's copies are adjacent rows, so within such a block the
    # expanded order is the block's entry sequence repeated once per copy.
    start = np.flatnonzero(np.r_[True, (prob[1:] != prob[:-1]) | (item[1:] != item[:-1])])
    size = np.diff(np.r_[start, head])
    btp = np.add.reduceat(gold, start).astype(float)
    bitem = item[start]
    has_tp = btp > 0

    def stat(idx):
        c = np.bincount(idx, minlength=len(items)).astype(float)
        n_pos = float(c @ npos)
        w = c[bitem]
        ctp = np.cumsum(w * btp)
        reached = (ctp / n_pos >= level) & has_tp
        if not reached.any():
            raise RuntimeError(f"{name}: recall {level} not reached within the kept head; enlarge it")
        b = int(np.argmax(reached))
        tp0 = float(ctp[b - 1]) if b else 0.0
        k0 = float(np.dot(w[:b], size[:b]))
        flags = np.cumsum(gold[start[b]:start[b] + size[b]])
        for m in range(int(w[b])):
            for r in range(size[b]):
                tp = tp0 + m * btp[b] + flags[r]
                if tp / n_pos >= level:
                    return tp / (k0 + m * size[b] + r + 1)
        raise AssertionError("unreachable")

    return Spec("sentence", len(items), stat, score_ref(name, task, items, preds),
                "sentence bootstrap; precision at 10% recall over (pair, positive relation) scores, ties by row order",
                ordered=True)


def spec_cicc(name, task, items, preds):
    cal = [it for it in items if "calibration" in it.fields["roles"]]
    test = [it for it in items if "test" in it.fields["roles"]]
    qhat, _ = task._qhat(items, preds, task.alpha)
    probs, _ = task._matrix(test, preds)
    single = (m_cicc.marginal_sets(probs, qhat).sum(axis=1) == 1).astype(float)
    return Spec("item", len(test), lambda idx: float(single[idx].mean()), score_ref(name, task, test, preds, fixed=cal),
                "test-utterance bootstrap; conformal qhat from the calibration split held fixed; share of singleton sets")


def spec_prp(name, task, items, preds):
    run = paper_tables.PAPER[name]["jev_paper_protocol"]["path"].split(".")[1]
    bm = {it.id: [[c["docid"], c["bm25_score"]] for c in it.fields["candidates"]] for it in items}
    prp = {it.id: [[d, s] for d, s in preds[it.id]["runs"][run]] for it in items}
    m_prp._normalizer(bm)
    m_prp._normalizer(prp)
    n_alpha = 11
    ndcg = []  # ndcg[q][a]: query q's nDCG@10 after interpolation with alpha = 0.1 * a
    for it in items:
        bm_scores = dict(bm[it.id])
        row = []
        for a in range(n_alpha):
            alpha = a * 0.1
            rows = [[d, s * alpha + bm_scores[d] * (1 - alpha)] for d, s in prp[it.id] if d in bm_scores]
            row.append(m_prp.ndcg_at_k(m_prp.trec_order(rows), it.gold, m_prp.NDCG_K))
        ndcg.append(row)
    n = len(items)
    perm = list(range(n))
    random.Random(42).shuffle(perm)  # shuffle depends only on the list length
    part = n // 10
    folds = [perm[i * part:(i + 1) * part] for i in range(9)] + [perm[9 * part:]]

    def stat(idx):
        q = idx.tolist()
        final = [0.0] * n
        for fold in folds:
            held = set(fold)
            train = [q[j] for j in range(n) if j not in held]
            grid = [float(f"{sum(ndcg[t][a] for t in train) / n:.4f}") for a in range(n_alpha)]
            a = grid.index(max(grid))
            for j in fold:
                final[j] = ndcg[q[j]][a]
        return metrics.mean(final)

    return Spec("query", n, stat, score_ref(name, task, items, preds),
                f"query bootstrap re-running the paper protocol per resample (BM25 interpolation, alpha by 10-fold CV "
                f"over the resampled test queries, {run})", ordered=True)


def _scalar_key(name: str) -> str:
    return name.split("/")[1]


def spec_scalar_point(name, task, items, preds):
    key = _scalar_key(name)
    fz = m_scalar.frozen(key)
    union = set().union(*(fz["folds"][k]["test_item_ids"] for k in range(m_scalar.N_FOLDS)))
    units = [r["id"] for r in fz["items"] if r["id"] in union]
    uidx = {u: i for i, u in enumerate(units)}
    folds = []
    for k in range(m_scalar.N_FOLDS):
        test = set(fz["folds"][k]["test_item_ids"])
        rows = [i for i in m_scalar._item_order(fz["pairs"][k]) if i in test]
        folds.append((np.array([uidx[i] for i in rows]), np.array([fz["bt"][i] for i in rows]),
                      np.array([float(preds[i]["score"]) for i in rows])))

    def stat(idx):
        c = np.bincount(idx, minlength=len(units))
        return float(np.mean([m_scalar._spearman(np.repeat(g, c[u]), np.repeat(p, c[u])) for u, g, p in folds]))

    def ref(idx):
        """task.score() on a frozen manifest rewritten with the resampled texts (unique ids)."""
        seen: Counter[int] = Counter()
        new_ids = {}
        for i in idx.tolist():
            new_ids.setdefault(units[i], []).append(f"{units[i]}#{seen[i]}")
            seen[i] += 1
        bt = {n: fz["bt"][u] for u, ns in new_ids.items() for n in ns}
        p = {n: preds[u] for u, ns in new_ids.items() for n in ns}
        pairs, fold_sets = {}, {}
        for k in range(m_scalar.N_FOLDS):
            test = set(fz["folds"][k]["test_item_ids"])
            order = [n for i in m_scalar._item_order(fz["pairs"][k]) if i in test for n in new_ids.get(i, [])]
            pairs[k] = [{"id1": n, "id2": n, "label": 0} for n in order]
            fold_sets[k] = {"test_item_ids": order, "heldout_item_ids": order}
        fake = {"items": [{"id": n} for n in bt], "bt": bt, "pairs": pairs, "folds": fold_sets}
        orig = m_scalar.frozen
        m_scalar.frozen = lambda _key: fake
        try:
            return value_from(name, task.score(items, p))
        finally:
            m_scalar.frozen = orig

    return Spec("text", len(units), stat, ref,
                "text bootstrap over the union of the 5 folds' test texts (a text's multiplicity applies in every fold "
                "containing it); mean of per-fold Spearman rho (Table 3 protocol)",
                ref_name="score() on a rewritten frozen manifest")


def spec_scalar_pair(name, task, items, preds):
    key = _scalar_key(name)
    fz = m_scalar.frozen(key)
    fold = m_scalar.PRIMARY_FOLD
    pairs = fz["pairs"][fold]
    choices = [int(preds[str(p["pair_id"])]["choice"]) for p in pairs]
    params = m_scalar.bt_fit([(p["id1"], p["id2"], c) for p, c in zip(pairs, choices)])
    test = set(fz["folds"][fold]["test_item_ids"])
    rows = [r["id"] for r in fz["items"] if r["id"] in params and r["id"] in test]
    gold = np.array([fz["bt"][i] for i in rows])
    pred = np.array([params[i] for i in rows])
    n = len(rows)
    rng = np.random.RandomState(m_scalar.BOOTSTRAP_SEED)
    inner = np.array([rng.choice(n, n, replace=True) for _ in range(m_scalar.N_BOOTSTRAPS)])
    from scipy.stats import rankdata

    def stat(idx):
        g, p = gold[idx][inner], pred[idx][inner]
        return float(np.nanmean(rowwise_pearson(rankdata(g, axis=1), rankdata(p, axis=1))))

    def ref(idx):
        return m_scalar.scoring_metrics(gold[idx], pred[idx])["spearman_rho"]["bootstrap"]["mean"]

    return Spec("text", n, stat, ref,
                "fold-0 test-text bootstrap with the Bradley-Terry fit to Jev's pairwise choices held fixed; statistic is "
                "the paper's headline (mean Spearman rho over its 25 seeded inner resamples)",
                ordered=True, ref_name="scalar_constructs.scoring_metrics")


def build(name: str, task: Any, items: list[Item], preds: dict[str, Any]) -> Spec:
    head = name.split("/")[0]
    if name.startswith("scalar_constructs/"):
        return (spec_scalar_pair if name.endswith("/pairwise") else spec_scalar_point)(name, task, items, preds)
    if name == "ezstance/subtaskB":
        return spec_ezstance_b(name, task, items, preds)
    if name == "relemb/wikiwd":
        return spec_wikiwd(name, task, items, preds)
    if name == "relemb/nytfb":
        return spec_nytfb(name, task, items, preds)
    simple = {
        "pinose": spec_pinose, "prp_graph": spec_prp, "fenice": spec_fenice, "factpico": spec_factpico,
        "biorc800": spec_biorc, "contempo": spec_contempo, "medreadme": spec_medreadme, "nav_unknown": spec_nav,
        "made": spec_made, "dress_new": spec_dress, "ttp": spec_ttp, "legalseg": spec_legalseg, "cicc": spec_cicc,
        "xstest": spec_accuracy, "sciprompt": spec_accuracy, "inli": spec_accuracy,
    }
    if head in simple:
        return simple[head](name, task, items, preds)
    labels = {"lifetox": m_lifetox.LABELS, "edit_intent": m_edit.LABELS, "mismatched": m_mismatched.LABELS,
              "ezstance": m_ez.LABELS}.get(head)
    if head == "smarter":
        labels = task.labels
    if labels is None:
        raise KeyError(f"no bootstrap spec for {name}")
    return spec_macro(name, task, items, preds, labels)


# ---------------------------------------------------------------- driver


def draw(rng: np.random.Generator, spec: Spec, sort: bool) -> np.ndarray:
    if spec.strata is None:
        idx = rng.integers(0, spec.n_units, spec.n_units)
    else:
        idx = np.concatenate([s[rng.integers(0, len(s), len(s))] for s in spec.strata])
    return np.sort(idx) if sort else idx


def classify(ci95: list[float], ci90: list[float], v: float | None) -> str:
    if v is None:
        return "n/a"
    if ci95[0] > v:
        return "above"
    if ci95[1] < v:
        return "below"
    if v - MARGIN <= ci90[0] and ci90[1] <= v + MARGIN:
        return "parity"
    return "inconclusive"


def run_task(row: dict, discrepancies: list[dict]) -> dict:
    name = row["task"]
    ref_meta = paper_tables.PAPER[name]
    task = REGISTRY[name]()
    items = task.items()
    preds = load_predictions(name)
    recomputed = value_from(name, task.score(items, preds))
    if recomputed != row["jev"]:
        discrepancies.append({"task": name, "kind": "stored_vs_recomputed", "stored": row["jev"],
                              "recomputed": recomputed, "abs_diff": abs(recomputed - row["jev"])})
    spec = build(name, task, items, preds)

    full = spec.stat(np.arange(spec.n_units))
    diffs = [abs(full - row["jev"])]
    check = np.random.default_rng(CHECK_SEED)
    for _ in range(N_CHECK):
        idx = draw(check, spec, sort=True)
        fast, ref = spec.stat(idx), spec.reference(idx)
        diffs.append(abs(fast - ref))
    if max(diffs) > TOL or not np.isfinite(diffs).all():
        discrepancies.append({"task": name, "kind": "fast_vs_reference", "full_sample_abs_diff": diffs[0],
                              "resample_abs_diffs": diffs[1:], "reference": spec.ref_name})

    rng = np.random.default_rng(SEED)
    vals = np.array([spec.stat(draw(rng, spec, sort=spec.ordered)) for _ in range(B)])
    n_nan = int(np.isnan(vals).sum())
    if n_nan:
        discrepancies.append({"task": name, "kind": "nan_resamples", "n": n_nan})
    ci95 = [float(x) for x in np.nanpercentile(vals, [2.5, 97.5])]
    ci90 = [float(x) for x in np.nanpercentile(vals, [5, 95])]
    method = (f"{spec.method}; fast metric == {spec.ref_name} on full sample + {N_CHECK} resamples "
              f"(max |diff| {max(diffs):.1e})")
    return {
        "name": row["name"], "metric": ref_meta["metric"], "unit": spec.unit, "n_units": spec.n_units,
        "jev": row["jev"], "jev_recomputed": recomputed, "paper": row["paper"], "llm": row["llm"],
        "ci95": ci95, "ci90": ci90, "vs_paper": classify(ci95, ci90, row["paper"]),
        "vs_llm": classify(ci95, ci90, row["llm"]), "method": method, "status": "ok",
    }


def main() -> None:
    """`[task ...] [--B=N]`: a task subset or a smaller B is a dry run that prints but does not write."""
    global B
    t_start = time.time()
    args = sys.argv[1:]
    for a in args:
        if a.startswith("--B="):
            B = int(a.split("=", 1)[1])
    only = {a for a in args if not a.startswith("--")}
    dry = bool(only) or B != 10_000
    rows = [r for r in paper_tables.rows() if r["paper"] is not None]
    discrepancies: list[dict] = []
    tasks: dict[str, dict] = {}
    for row in rows:
        if only and row["task"] not in only:
            continue
        t0 = time.time()
        tasks[row["task"]] = res = run_task(row, discrepancies)
        print(f"{row['task']:45s} {res['unit']:8s} n={res['n_units']:6d} jev={res['jev']:.4f} "
              f"ci95=[{res['ci95'][0]:.4f},{res['ci95'][1]:.4f}] paper={res['paper']:.4f} {res['vs_paper']:12s} "
              f"llm={res['llm']} {res['vs_llm']:12s} {time.time() - t0:6.1f}s", flush=True)
    counts = {
        "vs_paper": {k: 0 for k in ("above", "below", "parity", "inconclusive", "not_bootstrappable")},
        "vs_llm": {k: 0 for k in ("above", "below", "parity", "inconclusive", "not_bootstrappable", "n/a")},
    }
    for res in tasks.values():
        for side in ("vs_paper", "vs_llm"):
            counts[side][res["status"] if res["status"] != "ok" else res[side]] += 1
    out = {"B": B, "seed": SEED, "margin": MARGIN, "ci": list(CI), "tasks": tasks, "counts": counts,
           "discrepancies": discrepancies}
    if dry:
        print(json.dumps({"counts": counts, "discrepancies": discrepancies}, indent=1))
        print(f"partial run ({len(tasks)} tasks), not written; {time.time() - t_start:.0f}s")
        return
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(counts))
    print(f"wrote {OUT.relative_to(ROOT)} ({len(tasks)} tasks, {len(discrepancies)} discrepancies) "
          f"in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
