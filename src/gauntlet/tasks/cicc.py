"""CICC: Conformal Intent Classification and Clarification (den Hengst et al., Findings NAACL 2024).

The paper wraps a fine-tuned BERT intent classifier in split conformal
prediction: nonconformity 1 - p(true intent) on the calibration split, the
MAPIE quantile, and the set of intents with p >= 1 - q-hat. Algorithm 1 then
answers directly (set size 1), asks a clarification question over the set
(2..th intents) or rejects the input as too ambiguous (> th). Jev replaces the
BERT classifier: one `choice` over the dataset's intent inventory per
utterance, whose probabilities feed the unchanged conformal procedure. The
free-form clarification wording (Vicuna-generated) is not scored.

Tasks (splits = the authors' released 60/20/20 train/calibration/test split,
repo florisdenhengst/cicc@18019a1; Jev answers calibration AND test utterances
because the conformal quantile is computed on the model's calibration outputs):
- `cicc/{acid,atis,b77,c150,hwu64,mtod}`: Table 2. The paper's 1-alpha per
  dataset (B77: the effective .98, see DATASETS), th = 7. Metrics ported from
  the authors' `src/eval.py::cq_stats`: coverage over non-rejected inputs (set
  size <= th), Single (share of size-1 sets), |CQ| (mean size over
  1 < size <= th) and Amb (share with size >= th); plus SSC, marginal
  coverage, mean set size, top-1 accuracy, the paper's heuristic baselines
  B1-B3 on Jev's scores, and `other_levels` (Appendix E 1-alpha = .90 / .99;
  B77 also .97). C150 is the in-scope set (C150-IS: `oos` test rows removed).
- `cicc/c150_oos`, `cicc/b77_oos`: Table 3 OOS detection. Conformal quantile
  from the in-scope calibration split; OOS predicted when set size > th.
  Vanilla CICC (Table 2 alpha, th 7) and CICC-OOS (alpha on a fixed grid and th
  chosen for F1 on the OOS-calibration set, then applied to the OOS-test set),
  exactly as cicc_clinc150_bert.ipynb cells 23-26 and 41. F1 and AUROC
  (score = set size / #intents). The OOS sets are the notebooks' seeded samples
  (C150: 1,200 in-scope + 600 OOS per side; B77: ATIS open-domain outliers).

No utterance text is redistributed (ACID has no licence, ATIS is LDC-derived):
`data/frozen/cicc/*.jsonl` holds row coordinates, gold intents and per-row
SHA-256; `items()` rebuilds from the pinned clone in data/raw/cicc/repo
(`uv run scripts/prepare_cicc.py`). Requests for the same utterance are
identical across the in-scope and OOS tasks, so they share cache entries.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Iterator
from functools import cache
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from ..task import FROZEN, REPO, Item, Request, Task, read_jsonl

CLONE = REPO / "data" / "raw" / "cicc" / "repo"
FROZEN_DIR = FROZEN / "cicc"
KEY = "intent"
OOS = "oos"
# Frozen truncation rule: utterances cut to their first MAX_UTTERANCE_CHARS
# characters (~9K tokens). The longest frozen utterance is far shorter.
MAX_UTTERANCE_CHARS = 36_000
# MAPIE 0.6.5 `mapie._machine_precision.EPSILON` (float64 eps), used in the set rule.
EPSILON = float(np.finfo(np.float64).eps)
TH = 7  # paper Section 4 "Parameters": th = 7 for every BERT setting
TOP_K = 5  # baselines B1/B3 (paper Section 4 "Baselines")

# task dataset -> (release dir, 1-alpha behind the paper's Table 2/3 numbers).
# B77: Tables 2/3 print 1-alpha = .97, but the numbers were produced at alpha = .02:
# the authors' `index_for_arange(.01, .01, .03)` = int(1.9999999999999998) = 1 selects
# np.arange(.01, .4, .01)[1] = .02, and the released BERT outputs reproduce Table 3's
# B77-OOS CICC row (F1 .76, AUROC .92) and results.csv's B77 row only at alpha = .02.
DATASETS: dict[str, tuple[str, float]] = {
    "acid": ("acid", 0.98),
    "atis": ("atis", 0.99),
    "b77": ("banking77", 0.98),
    "c150": ("clinc150", 0.99),
    "hwu64": ("hwu64", 0.95),
    "mtod": ("mtod", 0.99),
}
# Secondary levels reported alongside: Appendix E Tables 8/9 (unoptimised 1-alpha)
# and, for B77, the printed-but-not-effective .97.
APPENDIX_E = (0.90, 0.99)
EXTRA_LEVELS = {"b77": (0.97,)}
# CICC-OOS alpha grid start = 1/len(y_ood). C150: y_ood = 1,200 in-scope test samples +
# the 1,200-utterance OOS pool (notebook cell 3) -> 1/2400. The B77 notebook's OOS cells
# reference an undefined y_ood; 1/5871 (the ATIS pool) is our reconstruction: on the
# released BERT outputs it selects 1-alpha = .9938 (paper Table 3: .994), th 6.
ALPHA_GRID_START = {"c150": 1 / 2400, "b77": 1 / 5871}


def row_sha256(text: str, label: str) -> str:
    return hashlib.sha256(json.dumps([text, label], ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


@cache
def _raw_texts(release_dir: str, file: str) -> tuple[str, ...]:
    path = CLONE / "data" / release_dir / "bert" / file
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `uv run scripts/prepare_cicc.py`")
    with path.open(newline="", encoding="utf-8") as fh:
        return tuple(r["text"] for r in csv.DictReader(fh))


def _clip(text: str) -> tuple[str, bool]:
    return (text[:MAX_UTTERANCE_CHARS], True) if len(text) > MAX_UTTERANCE_CHARS else (text, False)


# ---------------------------------------------------------------- conformal core


def conformal_qhat(cal_scores: np.ndarray, alpha: float) -> float:
    """MAPIE 0.6.5 `compute_quantiles`: np.quantile(s, (n+1)(1-alpha)/n, method="higher").

    A level above 1 (alpha < 1/(n+1)) gives an infinite threshold (every label included).
    """
    n = len(cal_scores)
    level = (n + 1) * (1 - alpha) / n
    if level > 1:
        return math.inf
    return float(np.quantile(cal_scores, level, method="higher"))


def marginal_sets(probs: np.ndarray, qhat: float) -> np.ndarray:
    """MAPIE `method="score"`, cv="prefit": include y iff p_y - (1 - qhat) >= -EPSILON."""
    return np.greater_equal(probs - (1 - qhat), -EPSILON)


def cq_stats(sets: np.ndarray, y: np.ndarray, th: int) -> dict[str, float]:
    """Port of the authors' src/eval.py::cq_stats (paper Eqs. 7-10).

    Coverage is over inputs with size <= th and Amb counts size >= th, as in the
    released code (the two boundaries differ at size == th).
    """
    size = sets.sum(axis=1)
    covered = sets[np.arange(len(y)), y]
    kept = size <= th
    cq = (size > 1) & (size <= th)
    return {
        "coverage": float(covered[kept].mean()) if kept.any() else float("nan"),
        "single": float((size == 1).mean()),
        "cq_size": float(size[cq].mean()) if cq.any() else float("nan"),
        "amb": float((size >= th).mean()),
    }


def ssc(sets: np.ndarray, y: np.ndarray) -> float:
    """Size-stratified coverage (paper Eq. 6; src/eval.py)."""
    size = sets.sum(axis=1)
    covered = sets[np.arange(len(y)), y]
    groups = [covered[size == c].mean() for c in range(1, int(size.max()) + 1) if (size == c).any()]
    return float(min(groups)) if groups else float("nan")


def _rank(probs: np.ndarray) -> np.ndarray:
    """Label indices by descending probability; ties to the earlier inventory label."""
    return np.argsort(-probs, axis=1, kind="stable")


def topk_sets(probs: np.ndarray, k: int) -> np.ndarray:
    sets = np.zeros(probs.shape, dtype=bool)
    top = _rank(probs)[:, :k]
    sets[np.arange(len(probs))[:, None], top] = True
    return sets


def baseline_sets(probs: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    """src/heuristics.py: B1 TopKCutoffSelector, B2 HeuristicCutoffSelector, B3 TopKSelector."""
    above = probs > 1 - alpha
    top5 = topk_sets(probs, TOP_K)
    b1 = np.where(above.any(axis=1, keepdims=True), above, top5)
    b2 = above | topk_sets(probs, 1)  # always_predict=True: top-1 always included
    return {"B1": b1, "B2": b2, "B3": top5}


def set_metrics(sets: np.ndarray, y: np.ndarray, th: int) -> dict[str, float]:
    size = sets.sum(axis=1)
    return {
        **cq_stats(sets, y, th),
        "ssc": ssc(sets, y),
        "marginal_coverage": float(sets[np.arange(len(y)), y].mean()),
        "mean_set_size": float(size.mean()),
        "empty": float((size == 0).mean()),  # score-CP can return no intent at all
    }


def _auroc(truth: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(truth, score)) if 0 < truth.sum() < len(truth) else float("nan")


def _f1(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(truth, pred, zero_division=0))


# ---------------------------------------------------------------- tasks


class _CICCBase(Task):
    dataset: str
    frozen_file: str

    def __init__(self) -> None:
        release_dir, one_minus_alpha = DATASETS[self.dataset]
        self.release_dir = release_dir
        self.alpha = round(1 - one_minus_alpha, 10)
        self.prompt_file = f"cicc_{self.dataset}.json"

    @property
    def question(self) -> dict[str, Any]:
        if not hasattr(self, "_question"):
            self._question = self.prompt()["question"]
        return self._question

    @property
    def labels(self) -> tuple[str, ...]:
        if not hasattr(self, "_labels"):
            self._labels = tuple(self.question["criteria"])
        return self._labels

    def items(self) -> list[Item]:
        out = []
        for r in read_jsonl(FROZEN_DIR / self.frozen_file):
            text = _raw_texts(self.release_dir, r["file"])[r["row"]]
            if row_sha256(text, r["gold"]) != r["sha256"]:
                raise ValueError(f"{self.name} {r['id']}: raw row differs from the frozen hash")
            out.append(Item(id=r["id"], gold=r["gold"], fields={"text": text, "roles": r["roles"]}))
        return out

    def requests(self, item: Item) -> Iterator[Request]:
        text, _ = _clip(item.fields["text"])
        yield Request(item_id=item.id, key=KEY, state={"utterance": text}, questions={KEY: self.question})

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        """Probabilities over the inventory, renormalised to sum 1 (the authors' MockModel
        divides scores by their row sum); labels Jev omits get 0."""
        labels = self.labels
        raw = answers[KEY][KEY].get("probabilities") or {}
        p = [max(float(raw.get(lab, 0.0)), 0.0) for lab in labels]
        total = sum(p)
        p = [x / total for x in p] if total > 0 else [1.0 / len(labels)] * len(labels)
        top = max(range(len(labels)), key=lambda i: (p[i], -i))
        return {"top1": labels[top], "probs": p}

    def _matrix(self, items: list[Item], predictions: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        index = {lab: i for i, lab in enumerate(self.labels)}
        probs = np.array([predictions[it.id]["probs"] for it in items], dtype=np.float64)
        y = np.array([index.get(it.gold, -1) for it in items])
        return probs, y

    def _qhat(self, items: list[Item], predictions: dict[str, Any], alpha: float) -> tuple[float, int]:
        cal = [it for it in items if "calibration" in it.fields["roles"]]
        probs, y = self._matrix(cal, predictions)
        scores = 1 - probs[np.arange(len(y)), y]
        return conformal_qhat(scores, alpha), len(cal)

    def _n_truncated(self, items: list[Item]) -> int:
        return sum(_clip(it.fields["text"])[1] for it in items)


class CICC(_CICCBase):
    """Table 2 setting: in-scope intent sets on the test split."""

    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.frozen_file = f"{dataset}.jsonl"
        self.name = f"cicc/{dataset}"
        super().__init__()

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        test = [it for it in items if "test" in it.fields["roles"]]
        probs, y = self._matrix(test, predictions)
        qhat, n_cal = self._qhat(items, predictions, self.alpha)
        sets = marginal_sets(probs, qhat)
        top1 = np.array([predictions[it.id]["top1"] for it in test])
        gold = np.array([it.gold for it in test])
        cal = [it for it in items if "calibration" in it.fields["roles"]]
        other_levels = {}
        for one_minus in (*APPENDIX_E, *EXTRA_LEVELS.get(self.dataset, ())):
            a = round(1 - one_minus, 10)
            q, _ = self._qhat(items, predictions, a)
            other_levels[f"{one_minus:.2f}"] = {
                "qhat": q,
                "CICC": set_metrics(marginal_sets(probs, q), y, TH),
                **{b: set_metrics(s, y, TH) for b, s in baseline_sets(probs, a).items()},
            }
        return {
            "n": len(test),
            # Coverage is guaranteed by construction for any classifier; the classifier shows
            # in how often a single intent suffices at that guarantee (Table 2 "Single").
            "primary_metric": "single (share of test inputs answered without clarification, at coverage >= 1-alpha)",
            "n_calibration": n_cal,
            "one_minus_alpha": round(1 - self.alpha, 10),
            "th": TH,
            "qhat": qhat,
            **set_metrics(sets, y, TH),
            "accuracy": float((top1 == gold).mean()),
            "accuracy_calibration": float(np.mean([predictions[it.id]["top1"] == it.gold for it in cal])),
            "baselines": {b: set_metrics(s, y, TH) for b, s in baseline_sets(probs, self.alpha).items()},
            "other_levels": other_levels,
            "n_truncated": self._n_truncated(items),
        }


class CICCOOS(_CICCBase):
    """Table 3 setting: OOS detection by rejection (set size > th)."""

    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.frozen_file = f"{dataset}_oos.jsonl"
        self.name = f"cicc/{dataset}_oos"
        super().__init__()

    def alpha_grid(self, n_calibration: int) -> np.ndarray:
        """np.arange(a_start, .01, .001) (cicc_clinc150_bert.ipynb cell 23), keeping only
        alpha >= 1/(n+1), below which MAPIE refuses to compute the quantile."""
        grid = np.arange(ALPHA_GRID_START[self.dataset], 0.01, 0.001)
        return grid[grid >= 1 / (n_calibration + 1)]

    def _sizes(self, items: list[Item], predictions: dict[str, Any], role: str, qhat: float) -> tuple[np.ndarray, np.ndarray]:
        sel = [it for it in items if role in it.fields["roles"]]
        probs = np.array([predictions[it.id]["probs"] for it in sel], dtype=np.float64)
        truth = np.array([it.gold == OOS for it in sel])
        return marginal_sets(probs, qhat).sum(axis=1), truth

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        n_intents = len(self.labels)
        # Vanilla CICC (Table 3 rows "CICC"; notebook cell 41): paper alpha, th = 7.
        q_vanilla, n_cal = self._qhat(items, predictions, self.alpha)
        size, truth = self._sizes(items, predictions, "oos_test", q_vanilla)
        pred = size > TH
        vanilla = {
            "one_minus_alpha": round(1 - self.alpha, 10),
            "th": TH,
            "qhat": q_vanilla,
            "f1": _f1(truth, pred),
            "auroc": _auroc(truth, size / n_intents),
            "accuracy": float((pred == truth).mean()),
            "empty_sets_oos": float((size[truth] == 0).mean()),
        }
        # CICC-OOS (cells 23-26): alpha maximising the best-th F1 (th in 0..11) on the
        # OOS-calibration set, then th in 0..49 maximising F1 at that alpha; first max wins.
        cal_scores = self._cal_scores(items, predictions)
        grid = self.alpha_grid(len(cal_scores))
        cal_sizes = []
        f1s = []
        for a in grid:
            s, t = self._sizes(items, predictions, "oos_calibration", conformal_qhat(cal_scores, float(a)))
            cal_sizes.append((s, t))
            f1s.append(max(_f1(t, s > th) for th in range(12)))
        best = int(np.argmax(f1s))
        s_cal, t_cal = cal_sizes[best]
        th_star = int(np.argmax([_f1(t_cal, s_cal > th) for th in range(50)]))
        q_star = conformal_qhat(cal_scores, float(grid[best]))
        size, truth = self._sizes(items, predictions, "oos_test", q_star)
        pred = size > th_star
        tuned = {
            "one_minus_alpha": float(1 - grid[best]),
            "th": th_star,
            "qhat": q_star,
            "f1_calibration": float(f1s[best]),
            "f1": _f1(truth, pred),
            "auroc": _auroc(truth, size / n_intents),
            "accuracy": float((pred == truth).mean()),
        }
        test = [it for it in items if "oos_test" in it.fields["roles"]]
        in_scope = [it for it in test if it.gold != OOS]
        return {
            "n": len(test),
            "primary_metric": "f1 of CICC-OOS (OOS = rejected as too ambiguous), Table 3",
            "n_oos": len(test) - len(in_scope),
            "n_calibration": n_cal,
            "n_oos_calibration": sum("oos_calibration" in it.fields["roles"] for it in items),
            "f1": tuned["f1"],
            "auroc": tuned["auroc"],
            "CICC": vanilla,
            "CICC-OOS": tuned,
            "alpha_grid": [float(a) for a in grid],
            "accuracy_in_scope": float(np.mean([predictions[it.id]["top1"] == it.gold for it in in_scope])),
            "n_truncated": self._n_truncated(items),
        }

    def _cal_scores(self, items: list[Item], predictions: dict[str, Any]) -> np.ndarray:
        cal = [it for it in items if "calibration" in it.fields["roles"]]
        probs, y = self._matrix(cal, predictions)
        return 1 - probs[np.arange(len(y)), y]


TASKS = {
    **{f"cicc/{ds}": (lambda ds=ds: CICC(ds)) for ds in DATASETS},
    "cicc/c150_oos": lambda: CICCOOS("c150"),
    "cicc/b77_oos": lambda: CICCOOS("b77"),
}
