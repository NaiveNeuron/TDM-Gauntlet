"""LegalSeg: rhetorical role labelling of Indian judgments (Nigam et al., Findings NAACL 2025).

Input: one test judgment (Supreme Court of India / High Court), pre-split into
sentences by the release (spaCy sentencizer, paper Section 4.2). Gold: one of
seven rhetorical roles per sentence (Section 3): Facts, Issue, Arguments of
Petitioner, Arguments of Respondent, Reasoning, Decision, None. Replaces the
paper's document-level sequence labellers (Hierarchical BiLSTM-CRF,
ToInLegalBERT, ...). Primary metric (Section 6, Table 3): macro-averaged F1
over the seven roles, pooled over all test sentences; macro precision/recall,
accuracy and MCC are the paper's other Table 3 columns; weighted F1 is added.

One item = one judgment (712 items, 149,881 sentences), sent as one or more
requests. A request's `state` holds the verbatim role definitions once, an
excerpt note and a window of numbered sentences; its `questions` hold one
Choice per target sentence (at most MAX_QUESTIONS), each pointing at its
sentence by path (`sentences.S0042`). Frozen chunking rule (see `_plan`): the
sentences are split into the fewest contiguous, near-equal target blocks of at
most MAX_QUESTIONS sentences. If the whole judgment fits MAX_STATE_CHARS,
every block's request shows the whole judgment. Otherwise each block's window
starts at the block and grows by neighbouring context sentences (nearest
first, alternating before/after) until the budget is full (blocks are made
smaller if their own text exceeds TARGET_SHARE of the budget). No sentence
text is ever cut. `n_truncated` counts judgments not shown whole.

Data: the public, ungated `L-NLProc/LegalSeg_RhetoricLLaMA_Predictions`
release, whose Index/Text/Label columns are the full test split (712 documents,
149,881 sentences, label counts equal to paper Table 2). The canonical
`L-NLProc/LegalSeg_CSV` copy is gated on Hugging Face. Text is not
redistributed: `data/frozen/legalseg/test.jsonl` holds per-document gold codes
and a SHA-256, and `items()` rebuilds text from
`data/raw/legalseg/RhetoricLLaMA_predictions.csv`
(`scripts/prepare_legalseg.py`), verifying every hash. Sentences on the frozen
non-English list (`excluded_non_english.jsonl`; the benchmark is English-only)
are removed before numbering: they are neither shown to Jev nor scored.
"""

from __future__ import annotations

import csv
import hashlib
import json
from functools import cache
from typing import Any

from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support

from .. import metrics
from ..task import FROZEN, REPO, Item, Request, Task, read_jsonl

RAW_CSV = REPO / "data" / "raw" / "legalseg" / "RhetoricLLaMA_predictions.csv"
RAW_CSV_SHA256 = "f4a366643ab5b9edcee38e83549b5c88ff1511fa461b7e018ebaae65711ba1a1"
FROZEN_DIR = FROZEN / "legalseg"

# Paper label ids (Table 5 instruction sets: None-0, Facts-1, ..., Decision-6).
# Gold codes in the frozen file index this tuple; also the argmax tie-break order.
LABELS = (
    "None",
    "Facts",
    "Issue",
    "Arguments of Petitioner",
    "Arguments of Respondent",
    "Reasoning",
    "Decision",
)
# Frozen request budget. MAX_STATE_CHARS bounds the compact-JSON length of
# `state`; MAX_QUESTIONS bounds the Choice questions per request (~90 o200k
# tokens each). Calibrated offline (data/manifests/legalseg.json): the densest
# test judgment runs 3.98 state chars per o200k token, and Jev's reported input
# tokens on cached requests of earlier tasks are ~1.02-1.11 x o200k(state) + o200k(question)
# + ~300, so a request stays below ~25K tokens, inside the 32,000-token context.
MAX_STATE_CHARS = 72_000
MAX_QUESTIONS = 40
# A block's own sentences may use at most this share of MAX_STATE_CHARS; the
# rest is left for context sentences.
TARGET_SHARE = 0.5


def sentence_id(i: int) -> str:
    """1-based, zero-padded sentence number shown to Jev (longest test judgment: 2,100)."""
    return f"S{i + 1:04d}"


def normalize(text: str) -> str:
    """Whitespace runs -> one space, stripped (the release pads some sentences)."""
    return " ".join(text.split())


def doc_sha256(texts: list[str], labels: list[str]) -> str:
    canonical = json.dumps([texts, labels], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_raw(path) -> dict[str, tuple[list[str], list[str]]]:
    """Release CSV -> {document Index: (raw texts, labels)} in file order.

    The release writes the `None` role as an empty Label cell (pandas NaN).
    """
    csv.field_size_limit(1 << 30)
    docs: dict[str, tuple[list[str], list[str]]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            texts, labels = docs.setdefault(row["Index"], ([], []))
            texts.append(row["Text"])
            labels.append(row["Label"] or "None")
    return docs


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@cache
def _raw_docs() -> dict[str, tuple[list[str], list[str]]]:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"{RAW_CSV} missing; run `uv run python scripts/prepare_legalseg.py`")
    if _sha256_file(RAW_CSV) != RAW_CSV_SHA256:
        raise ValueError(f"{RAW_CSV} SHA-256 differs from the frozen release hash {RAW_CSV_SHA256}")
    return read_raw(RAW_CSV)


def _size(obj: Any) -> int:
    return len(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))


class LegalSeg(Task):
    name = "legalseg"
    prompt_file = "legalseg.json"

    def __init__(self) -> None:
        self._prompt = self.prompt()
        self._plans: dict[str, list[tuple[str, int, int, int, int]]] = {}

    def items(self) -> list[Item]:
        raw = _raw_docs()
        excluded: dict[str, set[int]] = {}
        for x in read_jsonl(FROZEN_DIR / "excluded_non_english.jsonl"):
            excluded.setdefault(x["doc_index"], set()).add(x["sentence"])
            text = raw[x["doc_index"]][0][x["sentence"]]
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != x["sha256"]:
                raise ValueError(f"excluded sentence {x['doc_index']}/{x['sentence']} differs from the frozen hash")
        out = []
        for r in read_jsonl(FROZEN_DIR / "test.jsonl"):
            texts, labels = raw[r["doc_index"]]
            if [LABELS[int(c)] for c in r["gold"]] != labels or doc_sha256(texts, labels) != r["sha256"]:
                raise ValueError(f"{r['id']}: raw document differs from the frozen hash")
            drop = excluded.get(r["doc_index"], set())
            keep = [i for i in range(len(texts)) if i not in drop]
            out.append(
                Item(
                    id=r["id"],
                    gold=[labels[i] for i in keep],
                    fields={
                        "doc_index": r["doc_index"],
                        "sentences": [normalize(texts[i]) for i in keep],
                        "n_excluded_non_english": len(drop),
                    },
                )
            )
        return out

    # -- request construction -------------------------------------------------

    def _question(self, i: int) -> dict[str, Any]:
        q = self._prompt["question"]
        return {"type": q["type"], "instructions": q["instructions"].format(sid=sentence_id(i)), "criteria": q["criteria"]}

    def _state(self, sentences: list[str], lo: int, hi: int) -> dict[str, Any]:
        p = self._prompt["state"]
        n = len(sentences)
        if lo == 0 and hi == n:
            excerpt = p["excerpt_full"].format(n=n)
        else:
            excerpt = p["excerpt_partial"].format(first=sentence_id(lo), last=sentence_id(hi - 1), n=n)
        return {
            "document": p["document"],
            "excerpt": excerpt,
            "role_definitions": p["role_definitions"],
            "sentences": {sentence_id(i): sentences[i] for i in range(lo, hi)},
        }

    def _plan(self, item: Item) -> list[tuple[str, int, int, int, int]]:
        """Frozen chunking rule -> [(key, target start, target end, window lo, window hi)].

        State size is the compact-JSON length of `state`, computed exactly from
        per-entry sizes (entries of `sentences` are joined by one comma).
        """
        if item.id in self._plans:
            return self._plans[item.id]
        sents = item.fields["sentences"]
        n = len(sents)
        line = [_size(sentence_id(i)) + 1 + _size(sents[i]) for i in range(n)]
        # Size with an empty `sentences` map; sentence ids are fixed-width, so the
        # partial excerpt note has the same length for every window.
        base_full = _size(self._state(sents, 0, n) | {"sentences": {}})
        base_part = _size(self._state(sents, 0, n - 1) | {"sentences": {}})

        def size(base: int, lo: int, hi: int) -> int:
            return base + sum(line[lo:hi]) + (hi - lo - 1)

        def split(k: int) -> list[tuple[int, int]]:
            bounds = [n * j // k for j in range(k + 1)]
            return list(zip(bounds, bounds[1:]))

        k = -(-n // MAX_QUESTIONS)
        if size(base_full, 0, n) <= MAX_STATE_CHARS:
            windows = [(s, e, 0, n) for s, e in split(k)]
        else:
            while not all(size(base_part, s, e) <= TARGET_SHARE * MAX_STATE_CHARS for s, e in split(k)):
                k += 1
            windows = []
            for s, e in split(k):
                lo, hi = s, e
                total = size(base_part, lo, hi)
                grow_before, grow_after = True, True
                while grow_before or grow_after:
                    if grow_before:
                        if lo > 0 and total + line[lo - 1] + 1 <= MAX_STATE_CHARS:
                            lo -= 1
                            total += line[lo] + 1
                        else:
                            grow_before = False
                    if grow_after:
                        if hi < n and total + line[hi] + 1 <= MAX_STATE_CHARS:
                            total += line[hi] + 1
                            hi += 1
                        else:
                            grow_after = False
                windows.append((s, e, lo, hi))
        plan = [(f"chunk{j:03d}", s, e, lo, hi) for j, (s, e, lo, hi) in enumerate(windows)]
        self._plans[item.id] = plan
        return plan

    def next_requests(self, item: Item, answers: dict[str, dict[str, Any]]) -> list[Request]:
        if answers:
            return []
        sents = item.fields["sentences"]
        return [
            Request(
                item_id=item.id,
                key=key,
                state=self._state(sents, lo, hi),
                questions={sentence_id(i): self._question(i) for i in range(s, e)},
            )
            for key, s, e, lo, hi in self._plan(item)
        ]

    # -- prediction and scoring -----------------------------------------------

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        labels = []
        for key, s, e, _, _ in self._plan(item):
            for i in range(s, e):
                probs = answers[key][sentence_id(i)]["probabilities"]
                p = {lab: float(probs.get(lab, 0.0)) for lab in LABELS}
                # argmax; exact ties go to the earlier label in LABELS (paper id) order.
                labels.append(max(LABELS, key=lambda lab: (p[lab], -LABELS.index(lab))))
        return {"labels": labels}

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        gold = [g for it in items for g in it.gold]
        pred = [p for it in items for p in predictions[it.id]["labels"]]
        labels = list(LABELS)
        precision, recall, f1, _ = precision_recall_fscore_support(
            gold, pred, labels=labels, average="macro", zero_division=0
        )
        truncated = [it for it in items if any((lo, hi) != (0, len(it.gold)) for *_, lo, hi in self._plan(it))]
        return {
            "n": len(gold),
            "n_docs": len(items),
            "n_excluded_non_english": sum(it.fields["n_excluded_non_english"] for it in items),
            "macro_f1": float(f1),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "accuracy": metrics.accuracy(gold, pred),
            "mcc": float(matthews_corrcoef(gold, pred)),
            "weighted_f1": metrics.weighted_f1(gold, pred, labels=labels),
            "per_class_f1": metrics.per_class_f1(gold, pred, labels=labels),
            "n_gold": {lab: gold.count(lab) for lab in LABELS},
            "n_pred": {lab: pred.count(lab) for lab in LABELS},
            "n_requests": sum(len(self._plan(it)) for it in items),
            # Judgments too long to show whole: each request sees a window of the document.
            "n_truncated": len(truncated),
            "n_sentences_in_truncated_docs": sum(len(it.gold) for it in truncated),
        }


TASKS = {"legalseg": LegalSeg}
