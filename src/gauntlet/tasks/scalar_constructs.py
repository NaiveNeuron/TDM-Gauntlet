"""Scalar Constructs (Licht et al., EMNLP 2025): scalar measurement of political text.

"Measuring Scalar Constructs in Social Science with LLMs",
https://aclanthology.org/2025.emnlp-main.1635/. Release:
github.com/haukelicht/scalar_measurement @ RELEASE_COMMIT (code, data, splits,
prompts and the paper's own LLM outputs).

Three English U.S. political constructs, each a graph of human pairwise
comparisons (paper Table 1):
  immigration_fear  Carlson & Montgomery (2017) survey responses, 334 texts
  ad_negativity     Carlson & Montgomery (2017) WiscAds 2008 Senate ads, 935 texts
  grandstanding     Park (2021) House committee hearing paragraphs, 3,499 texts
Gold: `bt_scores.jsonl`, Bradley-Terry scores fit (choix ILSR) to all human
comparisons, and the human pairwise labels (1 = text1 wins, 2 = text2 wins).

Held-out evaluation (paper Section 3.3; release `prompting/evaluate_prompting.ipynb`):
fold 0 of the released graph splits. 100 high-degree vertices were held out;
the test split `0.test.jsonl` holds every comparison touching one of them. The
prompted-LLM results in Table 2 use exactly these pairs. Metrics, ported from
the notebook:
  * pairwise accuracy over all fold-0 test pairs (a predicted tie is wrong);
  * Spearman rho and RMSE against the gold BT scores over the notebook's "test
    items" (items in test pairs minus items in train pairs: 100/151/303; the
    extra 0/51/203 are train-graph vertices whose edges all went to the val
    split). RMSE is computed after min-max rescaling both score vectors to
    [0, 1] over those items;
  * every Table 2 value is the mean (+- std) over 25 bootstrap resamples
    (np.random.RandomState(42)) of the rows, in the notebook's row order.
Primary metric: Spearman rho (the paper's "primary focus is scoring
performance"); RMSE and accuracy are reported alongside, as in Table 2.

Two Jev replacements per construct, one task each:
  scalar_constructs/<construct>/pointwise  replaces the pointwise 1-9 scorer.
      One `score` request per text (9 levels). Prediction = probability-
      weighted mean level (paper Eq. 1), which Jev's score probabilities give
      directly. Pairwise choices are induced from the scores. Texts from all 5
      folds' test pairs are scored, so the fine-tuned models' 5-fold protocol
      (Table 3: mean +- std over folds of per-fold point estimates) is also
      reported as a supplement at no extra per-fold cost.
  scalar_constructs/<construct>/pairwise   replaces the pairwise judge.
      Four `choice` requests per fold-0 test pair: the paper's debiasing
      augmentations (App. A.2, Table 5; release ids `<pair>_<swap>_<reverse>`),
      label swap x order reversal. The state is the pair string the paper's
      models saw (release `0.test.pairwise.expanded.jsonl`, byte-identical).
      P(text1 wins) is the mean of the four aligned probabilities; choice 1 if
      > 0.5, 2 if < 0.5, tie (0) if equal (`debias_pairwise_locally`, margin 0).
      Item scores = BT fit (choix ILSR, alpha 1e-4, max_iter 1000) to the
      predicted choices on the test pairs.

The release carries no licence (and WiscAds storyboards have their own terms),
so texts are not redistributed: `items()` rebuilds them from data/raw (fetched
by scripts/prepare_scalar_constructs.py) and checks each text against the
frozen SHA-256 manifest in data/frozen/scalar_constructs/<construct>/.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import scipy.linalg as spl
from scipy.stats import spearmanr

from ..task import FROZEN, REPO, Item, Request, Task, read_jsonl

RELEASE_REPO = "haukelicht/scalar_measurement"
RELEASE_COMMIT = "7fec6010d3f95e9118d5907d77c65d5d8d380b11"
RAW = REPO / "data" / "raw" / "scalar_constructs" / "release"
FROZEN_DIR = FROZEN / "scalar_constructs"

N_FOLDS = 5
PRIMARY_FOLD = 0  # the only fold the paper's prompting experiments use
SCALE = tuple(range(1, 10))  # pointwise 1-9 scale; Jev score level index i -> SCALE[i]
# (swap_labels, reverse_pair_order), as in the release's augmented pair ids.
CONDITIONS = ((0, 0), (1, 0), (0, 1), (1, 1))
# Bradley-Terry: choix.ilsr_pairwise_dense(alpha=1e-4, max_iter=1000, tol=1e-8) (src/bt.py).
BT_ALPHA = 1e-4
BT_MAX_ITER = 1000
BT_TOL = 1e-8
# Bootstrap: src/icl/evaluation.py bootstrap_metric defaults.
N_BOOTSTRAPS = 25
BOOTSTRAP_SEED = 42
# Frozen truncation rule: each text is cut to its first MAX_TEXT_CHARS characters.
# The longest released text has 2,288 characters, so nothing is truncated.
MAX_TEXT_CHARS = 20_000


@dataclass(frozen=True)
class Construct:
    key: str
    dataset: str  # directory name in the release's data/processed_data/
    title: str
    # The release's Immigration Fear pair file was built by an earlier formatter that
    # wrapped the label in quotes ('"**Text 1**": ...'); the other two files do not.
    quoted_pair_labels: bool = False


CONSTRUCTS = {
    c.key: c
    for c in (
        Construct("immigration_fear", "carlson_montgomery2017_immigration", "Immigration Fear", quoted_pair_labels=True),
        Construct("ad_negativity", "carlson_montgomery2017_wiscads", "Ad-Negativity"),
        Construct("grandstanding", "park2019", "Grandstanding"),
    )
}


def raw_files(construct: Construct) -> list[str]:
    """Release paths (relative to the repo root) the adapter needs."""
    base = f"data/processed_data/{construct.dataset}"
    splits = [f"{base}/graph_splits/{k}.{s}.jsonl" for k in range(N_FOLDS) for s in ("test", "train", "val")]
    return [f"{base}/bt_scores.jsonl", *splits]


def raw_path(rel: str) -> Path:
    return RAW / rel


def frozen_dir(construct: Construct) -> Path:
    return FROZEN_DIR / construct.key


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def format_text_pair(text1: str, text2: str, swap_labels: bool, reverse_pair_order: bool, quoted: bool = False) -> str:
    """The pair string the paper's models saw, byte-identical to the `text` field of the
    release's prompting/pairwise_comparison/data/*/0.test.pairwise.expanded.jsonl.

    Logic of src/icl/pairwise_utils.format_text_pair: `swap_labels` gives the first-shown
    text the label "2", `reverse_pair_order` shows text2 first. (The Immigration Fear
    file names the two mixed conditions the other way round, `_0_1` <-> `_1_0`; the
    four presented strings are the same.)
    """
    label1, label2 = "1", "2"
    if swap_labels:
        label1, label2 = label2, label1
    if reverse_pair_order:
        text1, text2 = text2, text1
    q = '"' if quoted else ""
    return f'{q}**Text {label1}**{q}: """{text1}"""\n\n{q}**Text {label2}**{q}: """{text2}"""'


def p_text1(probabilities: dict[str, float], swap: int, reverse: int) -> float:
    """P(text1 wins) under one augmentation (release src/icl/debias.py): text1 carries
    label "1" when neither or both of swap/reverse apply, else label "2"."""
    label = "1" if swap == reverse else "2"
    return float(probabilities.get(label, 0.0))


def induce_choice(s1: float, s2: float) -> int:
    """release induce_choice_from_pairwise_preferences, margin 0: 1, 2 or tie 0."""
    return 1 if s1 > s2 else 2 if s2 > s1 else 0


# --- Bradley-Terry (port of choix 0.3.5 ilsr_pairwise_dense; choix is MIT-licensed) ---


def _statdist(generator: np.ndarray) -> np.ndarray:
    n = generator.shape[0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")  # the generator is singular by construction
        lu, _ = spl.lu_factor(generator.T, check_finite=False)
    res = spl.solve_triangular(lu[:-1, :-1], -lu[:-1, -1], check_finite=False)
    res = np.append(res, 1.0)
    return (n / res.sum()) * res


def bt_fit(comparisons: Sequence[tuple[str, str, int]]) -> dict[str, float]:
    """BT item parameters from (id1, id2, choice) rows; choice 1/2 = winner, 0 = tie (half win each)."""
    ids = sorted({i for a, b, _ in comparisons for i in (a, b)})
    index = {i: k for k, i in enumerate(ids)}
    n = len(ids)
    comp = np.zeros((n, n))
    for a, b, c in comparisons:
        if c == 1:
            comp[index[a], index[b]] += 1.0
        elif c == 2:
            comp[index[b], index[a]] += 1.0
        elif c == 0:
            comp[index[a], index[b]] += 0.5
            comp[index[b], index[a]] += 0.5
    params: np.ndarray | None = None
    prev: np.ndarray | None = None
    for _ in range(BT_MAX_ITER):
        if params is None:
            ws = np.ones(n)
        else:
            w = np.exp(params - params.mean())
            ws = (n / w.sum()) * w
        chain = BT_ALPHA * np.ones((n, n))
        denom = np.tile(ws, (n, 1))
        chain += comp.T / (denom + denom.T)
        chain -= np.diag(chain.sum(axis=1))
        logw = np.log(_statdist(chain))
        params = logw - logw.mean()
        centered = params - params.mean()
        if prev is not None and np.linalg.norm(prev - centered, ord=1) <= BT_TOL * n:
            return {i: float(p) for i, p in zip(ids, params)}
        prev = centered
    raise RuntimeError(f"Bradley-Terry ILSR did not converge after {BT_MAX_ITER} iterations")


# --- metrics (release src/icl/evaluation.py) ---


def rescale01(xs: Sequence[float]) -> np.ndarray:
    x = np.asarray(xs, dtype=float)
    return (x - x.min()) / (x.max() - x.min())


def _accuracy(gold: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(gold == pred))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(a, b).statistic)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def bootstrap(fn: Callable[[np.ndarray], float], n_rows: int) -> dict[str, float]:
    """Mean and std (ddof 0) of `fn(row_indices)` over the paper's 25 seeded resamples."""
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    vals = np.array([fn(rng.choice(n_rows, n_rows, replace=True)) for _ in range(N_BOOTSTRAPS)])
    return {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}


def pair_metrics(labels: Sequence[int], choices: Sequence[int]) -> dict[str, Any]:
    gold, pred = np.asarray(labels), np.asarray(choices)
    return {
        "point": _accuracy(gold, pred),
        "bootstrap": bootstrap(lambda ix: _accuracy(gold[ix], pred[ix]), len(gold)),
        "n_pairs": len(gold),
        "n_predicted_ties": int(np.sum(pred == 0)),
    }


def scoring_metrics(gold: Sequence[float], pred: Sequence[float]) -> dict[str, Any]:
    """Spearman rho and [0,1]-rescaled RMSE, point estimate and bootstrap, rows in given order."""
    g, p = np.asarray(gold, dtype=float), np.asarray(pred, dtype=float)
    g01, p01 = rescale01(g), rescale01(p)
    return {
        "spearman_rho": {"point": _spearman(g, p), "bootstrap": bootstrap(lambda ix: _spearman(g[ix], p[ix]), len(g))},
        "rmse": {"point": _rmse(g01, p01), "bootstrap": bootstrap(lambda ix: _rmse(g01[ix], p01[ix]), len(g))},
        "n_items": len(g),
    }


# --- frozen data ---


@cache
def frozen(construct_key: str) -> dict[str, Any]:
    """Frozen manifest: items (id, sha256, bt_score), per-fold test pairs and item sets."""
    d = frozen_dir(CONSTRUCTS[construct_key])
    items = read_jsonl(d / "items.jsonl")
    pairs = read_jsonl(d / "pairs.jsonl")
    folds = json.loads((d / "folds.json").read_text())
    return {
        "items": items,
        "bt": {r["id"]: float(r["bt_score"]) for r in items},
        "pairs": {k: [p for p in pairs if p["fold"] == k] for k in range(N_FOLDS)},
        "folds": {int(k): v for k, v in folds.items()},
    }


def load_texts(construct: Construct, raw: Path = RAW) -> dict[str, str]:
    """Item id -> text from the release's bt_scores.jsonl, verified against the frozen hashes."""
    rows = read_jsonl(raw / f"data/processed_data/{construct.dataset}/bt_scores.jsonl")
    by_id = {str(r["id"]): r["text"] for r in rows}
    texts = {}
    for entry in frozen(construct.key)["items"]:
        text = by_id.get(entry["id"])
        if text is None or text_sha256(text) != entry["sha256"]:
            raise ValueError(f"{construct.key} text {entry['id']!r} missing or changed since the freeze (sha256 mismatch)")
        texts[entry["id"]] = text
    return texts


def _clip(text: str) -> tuple[str, bool]:
    return (text[:MAX_TEXT_CHARS], True) if len(text) > MAX_TEXT_CHARS else (text, False)


def _item_order(pairs: Sequence[dict[str, Any]]) -> list[str]:
    """Items in first-appearance order over the pairs (release absolute_scoring/prepare_data.py)."""
    seen: dict[str, None] = {}
    for p in pairs:
        seen.setdefault(p["id1"])
        seen.setdefault(p["id2"])
    return list(seen)


def evaluate_fold(construct_key: str, fold: int, choices: Sequence[int], item_scores: dict[str, float],
                  score_order: Sequence[str]) -> dict[str, Any]:
    """Accuracy of `choices` (aligned with the fold's test pairs), plus rho/RMSE of
    `item_scores` against gold BT over the notebook's test items and over V_test.

    `score_order` fixes the row order for the bootstrap (the notebook's merge order).
    """
    fz = frozen(construct_key)
    pairs = fz["pairs"][fold]
    test_items = set(fz["folds"][fold]["test_item_ids"])
    heldout = set(fz["folds"][fold]["heldout_item_ids"])
    labels = [p["label"] for p in pairs]
    out: dict[str, Any] = {"accuracy": pair_metrics(labels, choices)}
    tt = [k for k, p in enumerate(pairs) if p["id1"] in test_items and p["id2"] in test_items]
    out["accuracy_test_test_pairs"] = pair_metrics([labels[k] for k in tt], [choices[k] for k in tt])
    rows = [i for i in score_order if i in test_items]
    out["scoring"] = scoring_metrics([fz["bt"][i] for i in rows], [item_scores[i] for i in rows])
    rows = [i for i in score_order if i in heldout]
    out["scoring_v_test_only"] = scoring_metrics([fz["bt"][i] for i in rows], [item_scores[i] for i in rows])
    return out


def _headline(fold_result: dict[str, Any]) -> dict[str, float]:
    """Table 2 style numbers: bootstrap means."""
    return {
        "spearman_rho": fold_result["scoring"]["spearman_rho"]["bootstrap"]["mean"],
        "rmse": fold_result["scoring"]["rmse"]["bootstrap"]["mean"],
        "accuracy": fold_result["accuracy"]["bootstrap"]["mean"],
    }


class _ScalarTask(Task):
    variant: str

    def __init__(self, construct_key: str, raw: Path = RAW) -> None:
        self.construct = CONSTRUCTS[construct_key]
        self.name = f"scalar_constructs/{construct_key}/{self.variant}"
        self.prompt_file = f"scalar_constructs_{construct_key}_{self.variant}.json"
        self.raw = raw

    def _question(self) -> dict[str, Any]:
        return self.prompt()["question"]


class PointwiseTask(_ScalarTask):
    """Jev `score` (1-9) per text; probability-weighted mean (paper Eq. 1)."""

    variant = "pointwise"
    KEY = "score"

    def items(self) -> list[Item]:
        fz = frozen(self.construct.key)
        texts = load_texts(self.construct, self.raw)
        return [Item(id=r["id"], gold=r["bt_score"], fields={"text": texts[r["id"]]}) for r in fz["items"]]

    def requests(self, item: Item) -> Iterator[Request]:
        text, _ = _clip(item.fields["text"])
        yield Request(item_id=item.id, key=self.KEY, state=text, questions={self.KEY: self._question()})

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        ans = answers[self.KEY][self.KEY]
        probs = [float(ans.get("probabilities", {}).get(str(i), 0.0)) for i in range(len(SCALE))]
        total = sum(probs)
        # Eq. 1: sum_s P(s) * s / P(scale). Jev's `score` (expected level index) is the fallback.
        score = sum(p * s for p, s in zip(probs, SCALE)) / total if total > 0 else SCALE[0] + float(ans["score"])
        # Modal level (supplementary; paper's "naive" variant). Exact ties -> lower level.
        modal = SCALE[max(range(len(SCALE)), key=lambda i: (probs[i], -i))]
        return {"score": score, "modal": modal}

    def _fold(self, fold: int, predictions: dict[str, Any], field: str) -> dict[str, Any]:
        pairs = frozen(self.construct.key)["pairs"][fold]
        s = {i: float(predictions[i][field]) for i in _item_order(pairs)}
        choices = [induce_choice(s[p["id1"]], s[p["id2"]]) for p in pairs]
        return evaluate_fold(self.construct.key, fold, choices, s, _item_order(pairs))

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        primary = self._fold(PRIMARY_FOLD, predictions, "score")
        folds = [self._fold(k, predictions, "score") for k in range(N_FOLDS)]
        per_fold = [
            {
                "spearman_rho": f["scoring"]["spearman_rho"]["point"],
                "rmse": f["scoring"]["rmse"]["point"],
                "accuracy": f["accuracy"]["point"],
            }
            for f in folds
        ]
        five_fold = {
            m: {"mean": float(np.mean([f[m] for f in per_fold])), "std": float(np.std([f[m] for f in per_fold], ddof=1))}
            for m in ("spearman_rho", "rmse", "accuracy")
        }
        return {
            "n": len(items),
            "n_truncated": sum(_clip(it.fields["text"])[1] for it in items),
            "primary_metric": "spearman_rho",
            **_headline(primary),
            "fold0": primary,
            "supplementary_modal_score_fold0": _headline(self._fold(PRIMARY_FOLD, predictions, "modal")),
            "supplementary_five_fold_table3_protocol": {"summary": five_fold, "per_fold": per_fold},
        }


class PairwiseTask(_ScalarTask):
    """Jev `choice` between Text 1 / Text 2 under the paper's four debiasing augmentations."""

    variant = "pairwise"
    QID = "preferred"

    def items(self) -> list[Item]:
        fz = frozen(self.construct.key)
        texts = load_texts(self.construct, self.raw)
        return [
            Item(
                id=str(p["pair_id"]),
                gold=p["label"],
                fields={"id1": p["id1"], "id2": p["id2"], "text1": texts[p["id1"]], "text2": texts[p["id2"]]},
            )
            for p in fz["pairs"][PRIMARY_FOLD]
        ]

    def _pair(self, item: Item) -> tuple[str, str, bool]:
        t1, c1 = _clip(item.fields["text1"])
        t2, c2 = _clip(item.fields["text2"])
        return t1, t2, c1 or c2

    def requests(self, item: Item) -> Iterator[Request]:
        t1, t2, _ = self._pair(item)
        question = {self.QID: self._question()}
        for swap, reverse in CONDITIONS:
            state = format_text_pair(t1, t2, bool(swap), bool(reverse), self.construct.quoted_pair_labels)
            yield Request(item_id=item.id, key=f"{swap}_{reverse}", state=state, questions=question)

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        by_cond = {
            f"{s}_{r}": p_text1(answers[f"{s}_{r}"][self.QID]["probabilities"], s, r) for s, r in CONDITIONS
        }
        pref1 = float(np.mean(list(by_cond.values())))
        naive = answers["0_0"][self.QID]["probabilities"]
        return {
            "p_text1": pref1,
            "choice": induce_choice(pref1, 1.0 - pref1),
            # Supplementary: the paper's non-debiased baseline (release debias.default_output):
            # original order only, choice 1 iff P("1") >= P("2").
            "choice_no_debiasing": 1 if float(naive.get("1", 0.0)) >= float(naive.get("2", 0.0)) else 2,
            "p_text1_by_condition": by_cond,
        }

    def _evaluate(self, items: list[Item], predictions: dict[str, Any], field: str) -> dict[str, Any]:
        fz = frozen(self.construct.key)
        pairs = fz["pairs"][PRIMARY_FOLD]
        choices = [int(predictions[str(p["pair_id"])][field]) for p in pairs]
        params = bt_fit([(p["id1"], p["id2"], c) for p, c in zip(pairs, choices)])
        # notebook: bt_scores.merge(predicted BT, how="inner") -> rows in bt_scores.jsonl
        # order; items.jsonl keeps that order.
        order = [r["id"] for r in fz["items"] if r["id"] in params]
        return evaluate_fold(self.construct.key, PRIMARY_FOLD, choices, params, order)

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        primary = self._evaluate(items, predictions, "choice")
        return {
            "n": len(items),
            "n_requests": len(items) * len(CONDITIONS),
            "n_truncated": sum(self._pair(it)[2] for it in items),
            "primary_metric": "spearman_rho",
            **_headline(primary),
            "fold0": primary,
            "supplementary_no_debiasing": _headline(self._evaluate(items, predictions, "choice_no_debiasing")),
        }


def _factory(cls: type[_ScalarTask], key: str) -> Callable[[], Task]:
    return lambda: cls(key)


TASKS: dict[str, Callable[[], Task]] = {
    f"scalar_constructs/{key}/{cls.variant}": _factory(cls, key)
    for key in CONSTRUCTS
    for cls in (PointwiseTask, PairwiseTask)
}
