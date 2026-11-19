"""BIORC800 (Lan et al., Findings of EMNLP 2024): multi-label sequential sentence classification.

Each test item is one PubMed abstract (160 abstracts, 1,579 sentences). Every
sentence gets one or more of six rhetorical labels. The paper's LLM-SSC reads the
whole abstract as context for each sentence and emits one sigmoid probability per
label (Eq. 7), thresholded per label with an argmax fallback when no label passes
(release `predict_thresholds`).

Jev geometry: one request per abstract. The state holds the annotation guideline
and the numbered sentences; the request asks one Noul per (sentence, label), i.e.
6 x n_sentences independent questions over the same state. Abstracts whose request
would exceed MAX_REQUEST_CHARS are split into the fewest equal contiguous sentence
chunks that fit (same state; questions are answered independently, so this does not
change any answer's context). Prediction is the paper's rule with a fixed a-priori
threshold: every label with P(yes) >= 0.5, and the arg-max label if none passes
(ties -> first label in the release order).

Scoring follows the release's `evaluation()`: sklearn micro / macro F1 over the
sentence x label indicator matrix (primary: micro and macro F1, Table 3).

Secondary protocol (the paper's dynamic thresholding, Section 3.3): per-label
thresholds chosen on the released dev split by the release's
`find_best_thresholds` grid. Run `biorc800/dev` first, then
`scripts/prepare_biorc800.py fit-thresholds` freezes
`data/frozen/biorc800/dev_thresholds.json`; the test score reports that
protocol only when the file exists.

The release has no licence, so rows are not redistributed: data/frozen holds an
ID + row-SHA-256 manifest and items are rebuilt from data/raw (see
scripts/prepare_biorc800.py).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from .. import metrics
from ..task import FROZEN, REPO, Item, Request, Task, read_jsonl

RAW = REPO / "data" / "raw" / "biorc800" / "release"
REPO_URL = "https://github.com/ScienceNLP-Lab/LLM-SSC"
COMMIT = "ae0986672ce1794c6edab028607a6fd88ab9ed28"
SPLITS: dict[str, dict[str, Any]] = {
    "test": {"sha256": "3ee098dd04b1854c9a8a28d6327bda8586ab097baa4a9db27537f19d15fa3a1f", "n": 160},
    "dev": {"sha256": "eeeb6fe22bfd50829490e9a5a8db721a1e74f49b908450c752c76735168ad264", "n": 159},
}
THRESHOLDS_PATH = FROZEN / "biorc800" / "dev_thresholds.json"

# Release label order (labels_to_ids, multi_label.py line 452); argmax ties -> first.
LABELS = ("BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS", "OTHER")
KEY = "roles"
THRESHOLD = 0.5  # fixed a priori; `>=` as in the release's predict_thresholds
# Paper's dynamic-thresholding grid and fallback (find_best_thresholds, --default_threshold 0.4).
THRESHOLD_GRID = [x * 0.1 for x in range(0, 10)]
DEFAULT_THRESHOLD = 0.4
# Frozen truncation rule: each sentence is cut to its first MAX_SENTENCE_CHARS chars.
# The longest released test/dev sentence is 658 chars, so nothing is cut.
MAX_SENTENCE_CHARS = 1_500
# Frozen request-size rule (JSON chars of state + questions). The largest single test
# request would be ~112K chars (~23.7K o200k tokens) against Jev's 32K-token context;
# 90K chars keeps a margin for a less efficient tokenizer. Splits 4 of 160 test and 0 dev
# abstracts (largest request after splitting: 87,855 chars, ~18.9K o200k tokens).
MAX_REQUEST_CHARS = 90_000

# Paper numbers, 0-1 scale. Table 3 (p. 16093): task-specific tuning on BIORC800.
# Table 2 (p. 16092): LLM-SSC in-context learning with Gemma-2b.
PAPER: dict[str, dict[str, float]] = {
    "LLM-SSC with WeighCon (Table 3)": {"micro_f1": 0.908, "macro_f1": 0.912},
    "LLM-SSC with WeighCon (Table 4 / text)": {"micro_f1": 0.907, "macro_f1": 0.912},
    "LLM-SSC with HeroCon": {"micro_f1": 0.902, "macro_f1": 0.906},
    "SSC (Cohan et al., 2019)": {"micro_f1": 0.905, "macro_f1": 0.892},
    "SciBERT-HSLN (Brack et al., 2022)": {"micro_f1": 0.902, "macro_f1": 0.897},
    "HSLN (Jin and Szolovits, 2018)": {"micro_f1": 0.849, "macro_f1": 0.826},
    "LLM-SSC ICL 0-shot (Gemma-2b)": {"micro_f1": 0.476, "macro_f1": 0.322},
    "LLM-SSC ICL 1-shot (Gemma-2b)": {"micro_f1": 0.642, "macro_f1": 0.507},
    "LLM-SSC ICL 5-shot (Gemma-2b)": {"micro_f1": 0.733, "macro_f1": 0.656},
    "LLM-SSC ICL 10-shot (Gemma-2b)": {"micro_f1": 0.159, "macro_f1": 0.068},
}


def raw_path(split: str) -> Path:
    return RAW / f"{split}.jsonl"


def raw_url(split: str) -> str:
    return f"https://raw.githubusercontent.com/ScienceNLP-Lab/LLM-SSC/{COMMIT}/data_llm/biorc800/{split}.jsonl"


def frozen_manifest_path(split: str) -> Path:
    return FROZEN / "biorc800" / split / "manifest.jsonl"


def raw_rows(split: str) -> list[dict[str, Any]]:
    """Abstracts exactly as released (file order)."""
    return read_jsonl(raw_path(split))


def row_sha256(row: dict[str, Any]) -> str:
    canonical = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def indicator(label_sets: Sequence[Sequence[str]]) -> np.ndarray:
    return np.array([[int(label in labels) for label in LABELS] for labels in label_sets], dtype=int)


def apply_thresholds(probs: Sequence[Sequence[float]], thresholds: Sequence[float]) -> list[list[str]]:
    """Release `predict_thresholds`: labels with p >= threshold, else the arg-max label."""
    out = []
    for row in probs:
        chosen = [label for label, p, t in zip(LABELS, row, thresholds) if p >= t]
        out.append(chosen or [LABELS[int(np.argmax(row))]])
    return out


def fit_thresholds(probs: Sequence[Sequence[float]], gold: Sequence[Sequence[str]]) -> dict[str, float]:
    """Release `find_best_thresholds`: per-label binary F1 over the 0.0..0.9 grid, first
    maximum wins; a best threshold of 0 is replaced by the default 0.4."""
    p = np.asarray(probs, dtype=float)
    y = indicator(gold)
    thresholds = {}
    for j, label in enumerate(LABELS):
        best_f1, best_t = -1.0, DEFAULT_THRESHOLD
        for t in THRESHOLD_GRID:
            f1 = f1_score(y[:, j], (p[:, j] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, (DEFAULT_THRESHOLD if t == 0 else t)
        thresholds[label] = best_t
    return thresholds


def multilabel_scores(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]]) -> dict[str, Any]:
    """The release's `evaluation()`: sklearn P/R/F1 on the sentence x label indicator matrix."""
    y, yhat = indicator(gold), indicator(pred)
    out: dict[str, Any] = {}
    for avg in ("micro", "macro", "weighted"):
        out[f"{avg}_f1"] = float(f1_score(y, yhat, average=avg, zero_division=0))
        out[f"{avg}_precision"] = float(precision_score(y, yhat, average=avg, zero_division=0))
        out[f"{avg}_recall"] = float(recall_score(y, yhat, average=avg, zero_division=0))
    per_label = f1_score(y, yhat, average=None, zero_division=0)
    out["per_label_f1"] = {label: float(v) for label, v in zip(LABELS, per_label)}
    out["exact_match"] = metrics.mean([float((a == b).all()) for a, b in zip(y, yhat)])
    out["n_pred_labels"] = {label: int(v) for label, v in zip(LABELS, yhat.sum(axis=0))}
    out["n_pred_multilabel_sentences"] = int((yhat.sum(axis=1) > 1).sum())
    return out


def _clip(text: str) -> tuple[str, bool]:
    return (text[:MAX_SENTENCE_CHARS], True) if len(text) > MAX_SENTENCE_CHARS else (text, False)


def question_id(number: int, label: str) -> str:
    return f"s{number:02d}_{label.lower()}"


def _chunk(state: dict[str, Any], per_sentence: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Fewest contiguous, equal-as-possible sentence ranges whose requests fit MAX_REQUEST_CHARS."""
    n = len(per_sentence)
    state_chars = len(json.dumps(state, ensure_ascii=False))
    sizes = [len(json.dumps(block, ensure_ascii=False)) for block in per_sentence]
    for k in range(1, n + 1):
        bounds = [(n * j // k, n * (j + 1) // k) for j in range(k)]
        if all(state_chars + sum(sizes[a:b]) <= MAX_REQUEST_CHARS for a, b in bounds):
            return bounds
    raise ValueError("a single sentence's questions exceed MAX_REQUEST_CHARS")


class BioRC800(Task):
    def __init__(self, split: str) -> None:
        self.split = split
        self.name = "biorc800" if split == "test" else f"biorc800/{split}"
        self.prompt_file = "biorc800.json"
        self.thresholds_path = THRESHOLDS_PATH

    def items(self) -> list[Item]:
        rows = raw_rows(self.split)
        manifest = read_jsonl(frozen_manifest_path(self.split))
        if len(rows) != len(manifest):
            raise ValueError(f"{self.name}: {len(rows)} raw rows vs {len(manifest)} in frozen manifest")
        items = []
        for entry in manifest:
            row = rows[entry["row"]]
            if row_sha256(row) != entry["sha256"] or row["abstract_id"] != entry["id"]:
                raise ValueError(f"{self.name}: row {entry['row']} does not match the frozen manifest")
            items.append(
                Item(
                    id=entry["id"],
                    gold=[list(labels) for labels in row["labels"]],
                    fields={"sentences": list(row["sentences"]), "stru_unstru": row["stru_unstru"]},
                )
            )
        return items

    def _sentences(self, item: Item) -> tuple[list[str], bool]:
        clipped = [_clip(s) for s in item.fields["sentences"]]
        return [text for text, _ in clipped], any(cut for _, cut in clipped)

    def requests(self, item: Item) -> Iterator[Request]:
        prompt = self.prompt()
        sentences, _ = self._sentences(item)
        state = {
            "annotation_guideline": prompt["annotation_guideline"],
            "abstract": [{"sentence": i + 1, "text": text} for i, text in enumerate(sentences)],
        }
        template = prompt["question"]
        per_sentence = []
        for i, text in enumerate(sentences):
            block = {}
            for label in LABELS:
                fill = {"number": i + 1, "index": i, "sentence": text, "label": label}
                block[question_id(i + 1, label)] = {
                    "type": template["type"],
                    "instructions": template["instructions"].format(**fill),
                    "criteria": {k: v.format(**fill) for k, v in template["criteria"].items()},
                }
            per_sentence.append(block)
        chunks = _chunk(state, per_sentence)
        for k, (start, stop) in enumerate(chunks):
            questions = {qid: q for block in per_sentence[start:stop] for qid, q in block.items()}
            key = KEY if len(chunks) == 1 else f"{KEY}/{k}"
            yield Request(item_id=item.id, key=key, state=state, questions=questions)

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        got = {qid: ans for request_answers in answers.values() for qid, ans in request_answers.items()}
        probs = [
            [float(got[question_id(i + 1, label)]["noul"]) for label in LABELS]
            for i in range(len(item.fields["sentences"]))
        ]
        return {"labels": apply_thresholds(probs, [THRESHOLD] * len(LABELS)), "probs": probs}

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        gold = [labels for it in items for labels in it.gold]
        pred = [labels for it in items for labels in predictions[it.id]["labels"]]
        probs = [row for it in items for row in predictions[it.id]["probs"]]
        y = indicator(gold)
        out: dict[str, Any] = {
            "n": len(items),
            "n_sentences": len(gold),
            **multilabel_scores(gold, pred),
            "threshold": THRESHOLD,
            "n_gold_labels": {label: int(v) for label, v in zip(LABELS, y.sum(axis=0))},
            "n_gold_multilabel_sentences": int((y.sum(axis=1) > 1).sum()),
            "per_label_auc": {
                label: (metrics.roc_auc(y[:, j].tolist(), [row[j] for row in probs]) if 0 < y[:, j].sum() < len(y) else None)
                for j, label in enumerate(LABELS)
            },
            "n_truncated": sum(self._sentences(it)[1] for it in items),
            "n_items_split_requests": sum(len(list(self.requests(it))) > 1 for it in items),
        }
        if self.split == "test":
            if self.thresholds_path.exists():
                frozen = json.loads(self.thresholds_path.read_text())
                thresholds = [float(frozen["thresholds"][label]) for label in LABELS]
                tuned = multilabel_scores(gold, apply_thresholds(probs, thresholds))
                out["dev_tuned"] = {
                    "micro_f1": tuned["micro_f1"],
                    "macro_f1": tuned["macro_f1"],
                    "per_label_f1": tuned["per_label_f1"],
                    "thresholds": frozen["thresholds"],
                    "thresholds_sha256": hashlib.sha256(self.thresholds_path.read_bytes()).hexdigest(),
                }
            else:
                out["dev_tuned"] = None
            out["paper"] = PAPER
        else:
            out["fitted_thresholds"] = fit_thresholds(probs, gold)
        return out


def _factory(split: str) -> Callable[[], Task]:
    return lambda: BioRC800(split)


TASKS: dict[str, Callable[[], Task]] = {
    "biorc800": _factory("test"),
    "biorc800/dev": _factory("dev"),
}
