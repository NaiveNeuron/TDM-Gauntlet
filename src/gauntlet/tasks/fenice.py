"""FENICE (Scirè et al., Findings ACL 2024): long-form summarization factuality.
"FENICE: Factuality Evaluation of summarization based on Natural language Inference
and Claim Extraction", https://aclanthology.org/2024.findings-acl.841/.

Data: the paper's long-form annotations (Section 4.4), released as the HF dataset
Babelscape/story-summeval (CC BY-NC-SA 4.0): English story summaries (12 Echo-XSum
summarizers + references) with a binary factual/non-factual label and an explicit
val/test split (40/279 rows). Stories are rebuilt from Gutenberg/Wikisource by
scripts/prepare_fenice.py and hash-checked here; summaries, labels and claims are
frozen in data/frozen/fenice/{test,val}.jsonl.

Replaced predictor: FENICE's NLI claim/source alignment. Upstream claim extraction
stays fixed: claims come from the paper's released T5 extractor (the setting of
FENICE_T5_claims). For every claim Jev answers one `noul`, P(the claim is explicitly
stated or implied by the source text). All claims of a summary share one request per
source chunk. As in FENICE (Eq. 2 and Section 3.4), a claim's score is its best
support over the source (max over chunks / candidate versions) and a summary's
score is the mean of its claim scores.

Metric (paper Table 7, p. 14156): balanced accuracy on the test split, with the
decision threshold tuned on the released val split exactly as FENICE's
eval/aggrefact.py `choose_best_threshold` (val-score percentiles 0, 0.2, ..., 99.8;
factual iff score > threshold; last best wins). Val rows are items of this task
only so their Jev scores can fix that threshold; every reported number is on test.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable, Iterator

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from .. import metrics
from ..task import FROZEN, REPO, Item, Request, Task, read_jsonl

FROZEN_DIR = FROZEN / "fenice"
PREPARE = "uv run scripts/prepare_fenice.py"

# Frozen chunking rule (no truncation): a source document longer than
# MAX_CHUNK_CHARS is split into ceil(len / MAX_CHUNK_CHARS) contiguous parts at
# paragraph boundaries (line boundaries inside a paragraph longer than the cap;
# whitespace inside a longer line), each part closed once it reaches len / n_parts
# characters and never exceeding MAX_CHUNK_CHARS (~20K tokens at 3.5 chars/token,
# inside the 32K context with the claim block). Nothing is truncated.
MAX_CHUNK_CHARS = 72_000

# Paper Table 7 (p. 14156): balanced accuracy (%) on the test split.
PAPER_TABLE7 = {
    "AlignScore": 61.3,
    "MENLI": 61.7,
    "DAE": 51.4,
    "FENICE_GPT_claims": 66.2,
    "FENICE_T5_claims": 65.7,
}
# The narrative whose source translation the release does not identify (see
# scripts/prepare_fenice.py); reported separately as a sensitivity check.
AMBIGUOUS_NARRATIVE = "wikisource:Thumbelina"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def balanced_accuracy(gold: list[int], pred: list[int]) -> float:
    return float(balanced_accuracy_score(gold, pred))


def choose_best_threshold(labels: list[int], scores: list[float]) -> float:
    """Port of FENICE eval/aggrefact.py choose_best_threshold (Tang et al. 2023 protocol)."""
    best, best_thresh = 0.0, 0.0
    for p in np.arange(0, 100, 0.2):
        thresh = float(np.percentile(scores, p))
        bacc = balanced_accuracy(labels, [1 if s > thresh else 0 for s in scores])
        if bacc >= best:
            best, best_thresh = bacc, thresh
    return best_thresh


def split_units(text: str, max_chars: int) -> list[tuple[str, str]]:
    """(separator before unit, unit): paragraphs ("\n\n"); a paragraph longer than
    max_chars is split into its lines ("\n"), and a line longer than max_chars is
    cut at whitespace. Joining separators + units restores the text."""
    out: list[tuple[str, str]] = []
    for p_i, para in enumerate(text.split("\n\n")):
        pieces = [para] if len(para) <= max_chars else para.split("\n")
        for l_i, piece in enumerate(pieces):
            sep = ("\n\n" if p_i else "") if l_i == 0 else "\n"
            while len(piece) > max_chars:
                cut = piece.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                out.append((sep, piece[:cut]))
                piece, sep = piece[cut:], ""
            out.append((sep, piece))
    return out


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    n_parts = math.ceil(len(text) / max_chars)
    target = len(text) / n_parts
    chunks: list[str] = []
    current = ""
    for sep, unit in split_units(text, max_chars):
        if current and (len(current) + len(sep) + len(unit) > max_chars or (len(current) >= target and len(chunks) < n_parts - 1)):
            chunks.append(current)
            current = unit
        else:
            current += (sep if current else "") + unit
    if current:
        chunks.append(current)
    return chunks


def load_narratives() -> dict[str, dict[str, Any]]:
    """Narrative key -> {title, documents: [text, ...]}, each text hash-checked."""
    spec = json.loads((FROZEN_DIR / "narratives.json").read_text(encoding="utf-8"))
    out = {}
    for key, entry in spec.items():
        texts = []
        for doc in entry["documents"]:
            path = REPO / doc["path"]
            if not path.exists():
                raise FileNotFoundError(f"{path} missing; run `{PREPARE}`")
            text = path.read_text(encoding="utf-8")
            if sha256_text(text) != doc["sha256"]:
                raise ValueError(f"fenice: {path} does not match its frozen sha256")
            texts.append(text)
        out[key] = {"title": entry["title"], "documents": texts}
    return out


class Fenice(Task):
    name = "fenice"
    prompt_file = "fenice.json"

    def __init__(self) -> None:
        self._narratives: dict[str, dict[str, Any]] | None = None
        self._chunks: dict[str, list[tuple[str, str]]] = {}

    def narratives(self) -> dict[str, dict[str, Any]]:
        if self._narratives is None:
            self._narratives = load_narratives()
        return self._narratives

    def items(self) -> list[Item]:
        narratives = self.narratives()
        items = []
        for split in ("test", "val"):
            for row in read_jsonl(FROZEN_DIR / f"{split}.jsonl"):
                if row["narrative"] not in narratives:
                    raise ValueError(f"fenice: {row['id']} has unknown narrative {row['narrative']!r}")
                fields = {k: row[k] for k in ("split", "release_id", "system_id", "narrative", "summary", "claims") if k in row}
                fields["split"] = split
                items.append(Item(id=row["id"], gold=int(row["factual"]), fields=fields))
        return items

    def chunks(self, narrative: str) -> list[tuple[str, str]]:
        """(scope sentence, source text) per request for a narrative, in frozen order."""
        if narrative not in self._chunks:
            scope = self.prompt()["scope"]
            docs = self.narratives()[narrative]["documents"]
            out = []
            for v, doc in enumerate(docs, start=1):
                parts = chunk_text(doc)
                for p, part in enumerate(parts, start=1):
                    sentences = []
                    if len(docs) > 1:
                        sentences.append(scope["version"].format(version=v, versions=len(docs)))
                    if len(parts) > 1:
                        sentences.append(scope["part"].format(part=p, parts=len(parts)))
                    out.append((" ".join(sentences) or scope["whole"], part))
            self._chunks[narrative] = out
        return self._chunks[narrative]

    @staticmethod
    def claims(item: Item) -> list[str]:
        # FENICE falls back to the whole summary when no claims are extracted.
        return item.fields["claims"] or [item.fields["summary"]]

    def requests(self, item: Item) -> Iterator[Request]:
        question = self.prompt()["question"]
        claims = self.claims(item)
        for c, (scope, text) in enumerate(self.chunks(item.fields["narrative"])):
            questions = {
                f"claim_{j}": {
                    "type": question["type"],
                    "instructions": question["instructions"].format(scope=scope, claim=claim),
                    "criteria": question["criteria"],
                }
                for j, claim in enumerate(claims)
            }
            yield Request(item_id=item.id, key=f"chunk_{c}", state={"source_text": text}, questions=questions)

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        n_chunks = len(self.chunks(item.fields["narrative"]))
        claim_scores = []
        for j in range(len(self.claims(item))):
            claim_scores.append(max(float(answers[f"chunk_{c}"][f"claim_{j}"]["noul"]) for c in range(n_chunks)))
        return {"score": float(np.mean(claim_scores)), "claim_scores": claim_scores}

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        val = [it for it in items if it.fields["split"] == "val"]
        test = [it for it in items if it.fields["split"] == "test"]
        val_gold = [int(it.gold) for it in val]
        val_scores = [float(predictions[it.id]["score"]) for it in val]
        threshold = choose_best_threshold(val_gold, val_scores)

        def block(subset: list[Item]) -> dict[str, Any]:
            gold = [int(it.gold) for it in subset]
            scores = [float(predictions[it.id]["score"]) for it in subset]
            pred = [1 if s > threshold else 0 for s in scores]
            return {
                "n": len(subset),
                "balanced_accuracy": balanced_accuracy(gold, pred),
                "roc_auc": metrics.roc_auc(gold, scores) if 0 < sum(gold) < len(gold) else float("nan"),
                "n_gold_factual": sum(gold),
                "n_pred_factual": sum(pred),
            }

        main = block(test)
        gold = [int(it.gold) for it in test]
        scores = [float(predictions[it.id]["score"]) for it in test]
        return {
            **main,
            "threshold_from_val": threshold,
            "val_balanced_accuracy": balanced_accuracy(val_gold, [1 if s > threshold else 0 for s in val_scores]),
            "n_val": len(val),
            # Secondary: a threshold fixed a priori at 0.5 instead of the val-tuned one.
            "balanced_accuracy_at_0.5": balanced_accuracy(gold, [1 if s > 0.5 else 0 for s in scores]),
            # Sensitivity: test rows whose source text the release identifies uniquely.
            "without_ambiguous_thumbelina": block([it for it in test if it.fields["narrative"] != AMBIGUOUS_NARRATIVE]),
            "n_requests_test": sum(len(self.chunks(it.fields["narrative"])) for it in test),
            "n_claims_test": sum(len(self.claims(it)) for it in test),
            "n_truncated": 0,
            "paper_table7_balanced_accuracy_percent": PAPER_TABLE7,
        }


TASKS: dict[str, Callable[[], Task]] = {"fenice": Fenice}
