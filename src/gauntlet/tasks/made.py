"""MADE (Agarwal et al., ACL 2026): multi-label coding of FDA medical-device adverse-event reports.

"MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty
Quantification of Medical Device Adverse Events", https://aclanthology.org/2026.acl-long.2148/.

Input: one English FDA MAUDE report as released ("Device Information: ...
Event Type: ... Description of Event: ..."). Gold: the report's IMDRF adverse
event codes, each up-propagated along its own listed IMDRF paths (release
code_to_parent_codes.json; 1,154-code vocabulary over three
levels; A = Medical Device Problem, E = Health Effects - Clinical Signs and
Symptoms or Conditions). Test set: the released 10,288-report "truncated test
set" (paper Table 1), the only split the paper evaluates on.

Replaced predictor: the paper's multi-label classifiers (best: discriminatively
fine-tuned Llama-3.1-8B-Base, macro-F1 0.54). Jev answers one `noul` per IMDRF
code, P(the code should be assigned to the report). 1,154 codes do not fit one
32K-token request, so the label hierarchy decomposes the task (adaptive, via
`next_requests`): round 1 asks the 51 top-level codes; every later round asks
the not-yet-asked codes that have an accepted parent (P > 0.5), until no new
code is reachable. Parents are the nearest vocabulary codes before the code on
each of its paths in the release's code_to_parent_codes.json (prompts/made_v2.json).
The paper's own prompted inference (App. A.7.1) shows all labels at once and so
has no routing. A round's questions share one request (split only when they
exceed the per-request caps). Codes never asked are treated as P = 0 (not
assigned).

Prediction (frozen a priori, no test labels involved): accept a code iff its
Jev P > 0.5. The accepted codes that have no accepted descendant stand in for
the coder-assigned codes; the predicted set is all accepted codes plus the codes
on those deepest codes' own paths. This is how the release built the gold
(Section 2: "each code is up-propagated to include all ancestor codes"; App.
A.7.1's parent/grandparent rule). An accepted intermediate code (e.g. E1202,
listed under E03 and E12) does not pull in the other parents on its own paths:
gold sets whose coder code is E120203 hold E1202 and E12 but not E03.
Per-code confidence ("score") is Jev's P, raised to the highest P of a deepest
accepted descendant, so every predicted code scores > 0.5. Per-code
uncertainty is the binary entropy of the score (the paper's measure for
per-label probabilities, App. A.4). Per-report uncertainty for PRR is the mean
per-code uncertainty over the predicted codes, evaluate.py's default and the
paper's treatment of models that score only part of the vocabulary (reports
with no predicted code carry none and are left out of PRR, as in evaluate.py).
Secondary: the paper's full-vector protocol (convert_paper_outputs.py,
"discriminative"), summed binary entropy over all scored codes (unasked codes
P = 0, entropy 0), which here also grows with the traversal's breadth.

Metrics (paper Table 2; ports of the official evaluate.py @ 9bff3ec): macro-F1
over the 1,154 codes (primary) and per training-frequency bucket (head /
medium / tail / extreme tail, head_medium_tail.json), example-based Jaccard,
PRR (Jaccard-based rejection, 20 coverage steps, random baseline seed 42),
Spearman rho between per-code uncertainty and correctness (pooled over
predicted codes), ECE+ (mean under-confidence on gold-positive codes).

Access: the Hugging Face release is gated ("agree to share your contact
information"); each user accepts the conditions on the Hub. Only an
idx + row-SHA-256 manifest is written to data/frozen/made/; `items()` rebuilds reports and gold
from data/raw/made/hf/test.jsonl (see scripts/prepare_made.py).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from ..task import FROZEN, REPO, Item, Request, Task, read_jsonl

RAW_DIR = REPO / "data" / "raw" / "made"
HF_DIR = RAW_DIR / "hf"
FROZEN_DIR = FROZEN / "made"
HF_REPO = "ragarwal/MADE-Multilabel-Benchmark"
HF_REVISION = "2caf6fcdc07a50ccb5f0316444b187bb9310908c"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO}"
# Git-LFS SHA-256 of test.jsonl at HF_REVISION (from the Hub API file metadata).
TEST_FILE_SHA256 = "7dea78fdfa43b4f7b0e01ae1e66823afcdcb1ad4237cd1b47176b287b8c4827c"
PAPER_TEST_N = 10_288  # paper Table 1, "Truncated test set"

THRESHOLD = 0.5  # a code is accepted iff Jev P > THRESHOLD (fixed a priori)
# Frozen truncation rule: the report is cut to its first MAX_REPORT_CHARS characters,
# the paper's own limit for prompted models (the release's prompts/README.md:
# "truncated to 10,000 characters"). Paper Table 1: ~370 cl100k tokens per report on
# average.
MAX_REPORT_CHARS = 10_000
# Per-request caps for one round's questions (state <= 10K chars; together well inside
# the 32K-token context). A round that exceeds either cap is split into consecutive
# chunks in vocabulary order.
MAX_QUESTIONS_PER_REQUEST = 128
MAX_QUESTION_CHARS_PER_REQUEST = 48_000
BUCKETS = ("head", "medium", "tail", "extreme_tail")
PRR_COVERAGE_STEPS = 20  # evaluate.py defaults
PRR_SEED = 42


def row_sha256(idx: int, text: str, labels: Sequence[str]) -> str:
    canonical = json.dumps([idx, text, sorted(labels)], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_test_path(raw_dir: Path = RAW_DIR) -> Path:
    path = raw_dir / "hf" / "test.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. The MADE release is gated on Hugging Face: log in at {HF_URL}, "
            "accept the access conditions (share contact information), then run scripts/prepare_made.py "
            "stages `hierarchy` and `test` (see its docstring)."
        )
    return path


def load_raw_test(raw_dir: Path = RAW_DIR) -> list[dict[str, Any]]:
    """Rows of the released test.jsonl ({"idx", "text", "labels"}), after a file hash check."""
    path = raw_test_path(raw_dir)
    got = sha256_file(path)
    if got != TEST_FILE_SHA256:
        raise ValueError(f"{path}: sha256 {got} != pinned {TEST_FILE_SHA256} (HF revision {HF_REVISION})")
    return read_jsonl(path)


def binary_entropy(p: float, eps: float = 1e-15) -> float:
    """convert_paper_outputs.py `binary_entropy` (natural log, clipped)."""
    p = min(max(p, eps), 1 - eps)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


# ---------------------------------------------------------------------------
# Metrics: ports of raunak-agarwal/made-benchmark@9bff3ec evaluate.py
# ---------------------------------------------------------------------------


def binary_matrices(
    gold_lists: Sequence[Sequence[str]], pred_lists: Sequence[Sequence[str]], label_list: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-hot matrices over `label_list`; labels outside the vocabulary are ignored."""
    col = {label: i for i, label in enumerate(label_list)}
    y_true = np.zeros((len(gold_lists), len(label_list)), dtype=bool)
    y_pred = np.zeros((len(gold_lists), len(label_list)), dtype=bool)
    for i, (gold, pred) in enumerate(zip(gold_lists, pred_lists)):
        for label in gold:
            if label in col:
                y_true[i, col[label]] = True
        for label in pred:
            if label in col:
                y_pred[i, col[label]] = True
    return y_true, y_pred


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted mean of per-label F1; zero division yields 0 (labels absent from gold and
    predictions therefore count as 0, as in the paper)."""
    tp = (y_pred & y_true).sum(axis=0).astype(float)
    fp = (y_pred & ~y_true).sum(axis=0).astype(float)
    fn = (~y_pred & y_true).sum(axis=0).astype(float)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


def example_jaccard(gold_lists: Sequence[Sequence[str]], pred_lists: Sequence[Sequence[str]]) -> float:
    """Mean per-report Jaccard between predicted and gold sets (empty union counts 0)."""
    scores = []
    for gold, pred in zip(gold_lists, pred_lists):
        g, p = set(gold), set(pred)
        union = g | p
        scores.append(len(g & p) / len(union) if union else 0.0)
    return float(np.mean(scores))


def per_sample_jaccard(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Vectorised per-report Jaccard (empty union counts 1.0, as in the PRR code)."""
    intersection = (y_true & y_pred).sum(axis=1)
    union = (y_true | y_pred).sum(axis=1)
    return np.where(union > 0, intersection / np.maximum(union, 1), 1.0)


def _rejection_auc(quality: np.ndarray, uncertainty: np.ndarray, coverage_steps: int) -> float:
    """AUC of mean quality vs. coverage when the most uncertain reports are rejected first."""
    n = len(quality)
    order = np.argsort(uncertainty)
    quality_sorted = quality[order]
    coverages = np.linspace(1, 0, coverage_steps)
    curve = np.full(coverage_steps, np.nan)
    for i, c in enumerate(coverages):
        n_keep = int(np.ceil(c * n))
        if n_keep > 0:
            curve[i] = quality_sorted[:n_keep].mean()
    mask = ~np.isnan(curve)
    return float(np.trapezoid(curve[mask][::-1], coverages[mask][::-1]))


def prr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    coverage_steps: int = PRR_COVERAGE_STEPS,
    seed: int = PRR_SEED,
) -> float:
    """Prediction Rejection Rate: (AUC_model - AUC_random) / (AUC_oracle - AUC_random)."""
    quality = per_sample_jaccard(y_true, y_pred)
    auc_model = _rejection_auc(quality, uncertainty, coverage_steps)
    auc_oracle = _rejection_auc(quality, 1.0 - quality, coverage_steps)
    rng = np.random.RandomState(seed)
    auc_random = _rejection_auc(quality, rng.permutation(uncertainty), coverage_steps)
    if auc_oracle - auc_random == 0:
        return float("nan")
    return float((auc_model - auc_random) / (auc_oracle - auc_random))


def spearman_uncertainty(
    gold_lists: Sequence[Sequence[str]],
    pred_lists: Sequence[Sequence[str]],
    label_uncertainties: Sequence[dict[str, float]],
) -> tuple[float, int]:
    """Spearman rho between per-label uncertainty and binary correctness, pooled over all
    (report, predicted label) pairs that carry an uncertainty."""
    correct, uncertain = [], []
    for gold, pred, unc in zip(gold_lists, pred_lists, label_uncertainties):
        gold_set = set(gold)
        for label in pred:
            u = unc.get(label)
            if u is None:
                continue
            correct.append(1 if label in gold_set else 0)
            uncertain.append(float(u))
    if len(set(correct)) < 2 or len(set(uncertain)) < 2:
        return float("nan"), 0
    rho, _ = spearmanr(correct, uncertain)
    return float(rho), len(correct)


def ece_plus(
    gold_lists: Sequence[Sequence[str]], score_dicts: Sequence[dict[str, float]], label_list: Sequence[str]
) -> tuple[float, int]:
    """Positive-class ECE: mean over labels with test positives of (1 - mean confidence on
    those positives); a missing score counts as confidence 0."""
    positive_conf: dict[str, list[float]] = {label: [] for label in label_list}
    for gold, scores in zip(gold_lists, score_dicts):
        for label in gold:
            if label in positive_conf:
                positive_conf[label].append(float(scores.get(label, 0.0)))
    per_label = [1.0 - float(np.mean(confs)) for confs in positive_conf.values() if confs]
    return float(np.mean(per_label)), len(per_label)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class MADE(Task):
    name = "made"
    prompt_file = "made_v2.json"

    def __init__(self, raw_dir: Path = RAW_DIR, frozen_dir: Path = FROZEN_DIR) -> None:
        self.raw_dir = raw_dir
        self.frozen_dir = frozen_dir

    @cached_property
    def _prompt(self) -> dict[str, Any]:
        return self.prompt()

    @cached_property
    def vocab(self) -> list[str]:
        """The 1,154 codes in frozen (lexicographic) order."""
        return list(self._prompt["questions"])

    @cached_property
    def _parents(self) -> dict[str, list[str]]:
        return {c: t["parents"] for c, t in self._prompt["taxonomy"].items()}

    @cached_property
    def _ancestors(self) -> dict[str, list[str]]:
        return {c: t["ancestors"] for c, t in self._prompt["taxonomy"].items()}

    @cached_property
    def _descendants(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {c: set() for c in self.vocab}
        for code, ancestors in self._ancestors.items():
            for anc in ancestors:
                out[anc].add(code)
        return out

    @cached_property
    def _question_chars(self) -> dict[str, int]:
        return {c: len(json.dumps(q, ensure_ascii=False)) for c, q in self._prompt["questions"].items()}

    def items(self) -> list[Item]:
        by_idx = {row["idx"]: row for row in load_raw_test(self.raw_dir)}
        manifest_path = self.frozen_dir / "test_manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} is missing: finish the English screen and freeze with scripts/prepare_made.py "
                "stage `test` (see its docstring)."
            )
        manifest = read_jsonl(manifest_path)
        items = []
        for entry in manifest:
            row = by_idx.get(entry["idx"])
            if row is None or row_sha256(row["idx"], row["text"], row["labels"]) != entry["sha256"]:
                raise ValueError(f"MADE test row idx={entry['idx']} missing or changed since the freeze (sha256 mismatch)")
            items.append(Item(id=str(row["idx"]), gold=sorted(row["labels"]), fields={"text": row["text"]}))
        return items

    def _state(self, item: Item) -> tuple[dict[str, str], bool]:
        text = item.fields["text"]
        if len(text) > MAX_REPORT_CHARS:
            return {self._prompt["state_key"]: text[:MAX_REPORT_CHARS]}, True
        return {self._prompt["state_key"]: text}, False

    def _chunks(self, codes: list[str]) -> list[list[str]]:
        """Deterministic split of one round's codes (vocabulary order) under the per-request caps."""
        chunks: list[list[str]] = []
        current: list[str] = []
        chars = 0
        for code in codes:
            size = self._question_chars[code]
            if current and (len(current) >= MAX_QUESTIONS_PER_REQUEST or chars + size > MAX_QUESTION_CHARS_PER_REQUEST):
                chunks.append(current)
                current, chars = [], 0
            current.append(code)
            chars += size
        if current:
            chunks.append(current)
        return chunks

    def _traverse(
        self, item: Item, answers: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, float], list[list[str]], list[Request]]:
        """Replay the hierarchical traversal on the answers so far.

        Returns (P per asked code, codes asked per round, requests still missing). Round 1
        asks the root codes; round r+1 asks every not-yet-asked code with a parent accepted
        in rounds 1..r. The traversal stops at the first round lacking answers.
        """
        asked: dict[str, float] = {}
        rounds: list[list[str]] = []
        roots = [c for c in self.vocab if not self._parents[c]]
        frontier = roots
        while frontier:
            r = len(rounds) + 1
            chunks = self._chunks(frontier)
            keys = [f"r{r}c{i}" for i in range(len(chunks))]
            missing = [(k, ch) for k, ch in zip(keys, chunks) if k not in answers]
            if missing:
                state, _ = self._state(item)
                questions = self._prompt["questions"]
                return asked, rounds, [
                    Request(item_id=item.id, key=k, state=state, questions={c: questions[c] for c in ch})
                    for k, ch in missing
                ]
            for key, chunk in zip(keys, chunks):
                for code in chunk:
                    asked[code] = float(answers[key][code]["noul"])
            rounds.append(frontier)
            accepted = {c for c, p in asked.items() if p > THRESHOLD}
            frontier = [c for c in self.vocab if c not in asked and any(p in accepted for p in self._parents[c])]
        return asked, rounds, []

    def next_requests(self, item: Item, answers: dict[str, dict[str, Any]]) -> list[Request]:
        return self._traverse(item, answers)[2]

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        asked, rounds, missing = self._traverse(item, answers)
        if missing:
            raise ValueError(f"item {item.id}: answers missing for {[r.key for r in missing]}")
        scores = dict(asked)
        accepted = {c for c, p in asked.items() if p > THRESHOLD}
        deepest = sorted(c for c in accepted if not self._descendants[c] & accepted)
        lifted = set()
        # Hierarchy rule, as the gold was built: the deepest accepted codes imply the codes on
        # their own IMDRF paths, whose confidence is raised to the deepest code's P.
        for code in deepest:
            p = asked[code]
            for anc in self._ancestors[code]:
                if scores.get(anc, 0.0) < p:
                    scores[anc] = p
                    lifted.add(anc)
        labels = sorted(c for c, s in scores.items() if s > THRESHOLD)
        entropy = {c: binary_entropy(s) for c, s in scores.items()}
        return {
            "labels": labels,
            "scores": {c: scores[c] for c in sorted(scores)},
            "lifted": sorted(lifted),
            "deepest": deepest,
            "uncertainties": {c: entropy[c] for c in labels},
            # Per-report uncertainty for PRR (None without a predicted code, as in evaluate.py).
            "uncertainty": float(np.mean([entropy[c] for c in labels])) if labels else None,
            "uncertainty_sum_scored": float(sum(entropy.values())),
            "n_asked": len(asked),
            "n_rounds": len(rounds),
            "n_requests": sum(len(self._chunks(r)) for r in rounds),
        }

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        buckets = json.loads((self.frozen_dir / "label_buckets.json").read_text())["buckets"]
        label_list = self.vocab
        gold_lists = [it.gold for it in items]
        preds = [predictions[it.id] for it in items]
        pred_lists = [p["labels"] for p in preds]
        y_true, y_pred = binary_matrices(gold_lists, pred_lists, label_list)
        col = {label: i for i, label in enumerate(label_list)}
        f1_buckets = {}
        for name in BUCKETS:
            cols = [col[c] for c in buckets[name]]
            f1_buckets[name] = macro_f1(y_true[:, cols], y_pred[:, cols])

        rho, n_pairs = spearman_uncertainty(gold_lists, pred_lists, [p["uncertainties"] for p in preds])
        ece, ece_labels = ece_plus(gold_lists, [p["scores"] for p in preds], label_list)
        with_unc = [i for i, p in enumerate(preds) if p["uncertainty"] is not None]
        unc = np.array([preds[i]["uncertainty"] for i in with_unc], dtype=float)
        unc_sum = np.array([p["uncertainty_sum_scored"] for p in preds], dtype=float)
        return {
            "n": len(items),
            "primary": "macro_f1 (overall, 1,154 IMDRF codes)",
            "macro_f1": macro_f1(y_true, y_pred),
            "macro_f1_buckets": f1_buckets,
            "jaccard": example_jaccard(gold_lists, pred_lists),
            "prr": prr(y_true[with_unc], y_pred[with_unc], unc) if with_unc else float("nan"),
            "prr_n_samples": len(with_unc),
            "spearman": rho,
            "spearman_n_pairs": n_pairs,
            "ece_plus": ece,
            "ece_plus_n_labels": ece_labels,
            "ece_plus_n_samples": sum(bool(p["scores"]) for p in preds),
            "uq_protocol": "per-code uncertainty = binary entropy of the score; PRR per-report uncertainty = mean "
            "per-code uncertainty over predicted codes (evaluate.py default); reports without predictions excluded",
            "prr_summed_entropy_all_reports": {
                "note": "paper's full-vector protocol (convert_paper_outputs.py discriminative): summed binary "
                "entropy over scored codes, unasked codes P=0",
                "prr": prr(y_true, y_pred, unc_sum),
                "n_samples": len(items),
            },
            "labels_per_report": {
                "gold_mean": float(y_true.sum(axis=1).mean()),
                "pred_mean": float(y_pred.sum(axis=1).mean()),
                "asked_mean": float(np.mean([p["n_asked"] for p in preds])),
            },
            "n_empty_predictions": sum(not p["labels"] for p in preds),
            "n_labels_with_test_positives": int(y_true.any(axis=0).sum()),
            "requests_per_report_mean": float(np.mean([p["n_requests"] for p in preds])),
            "n_truncated": sum(self._state(it)[1] for it in items),
        }


TASKS: dict[str, Callable[[], Task]] = {"made": MADE}
