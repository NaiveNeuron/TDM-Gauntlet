"""SciPrompt (You et al., EMNLP 2024): scientific-abstract topic classification.

Input: one abstract. Gold: its single topic label from a fixed inventory. Replaces the
paper's knowledge-augmented prompt-tuned SciBERT (SciPrompt). Four tasks, one per
released test split (paper Table 5 / Appendix B), each scored separately:

  sciprompt/arxiv         53 arXiv categories, 5,300 abstracts (100 per category)
  sciprompt/s2orc         19 Semantic Scholar fields of study, 5,700 (300 per field)
  sciprompt/sdpra         7 SDPRA-2021 CS domains, 2,790 (the paper's 30-token
                          SciBERT filter drops 10 of the released 2,800)
  sciprompt/emerging_nlp  21 Emerging NLP topics, 420 (20 per topic)

minus a frozen list of non-English abstracts (the benchmark is English-only): scored N
5,295 / 5,660 / 2,789 / 420 (data/frozen/sciprompt/excluded.jsonl).

Primary metric (paper Tables 1-2, Figure 3): accuracy. Macro-F1 (the paper's
Limitations name it as the obvious alternative) and weighted-F1 (the SDPRA-2021
shared-task metric) are secondary.

Data: the authors' Google Drive release `sciprompt_data.zip`. SDPRA (CC BY 4.0, via
the shared-task Mendeley record) and Emerging NLP (Apache-2.0) abstracts are frozen
verbatim in data/frozen/sciprompt/; arXiv and S2ORC abstracts (no dataset licence)
are frozen as row index + SHA-256 and rebuilt from data/raw/sciprompt/ by `items()`.
Build: `scripts/prepare_sciprompt.py`.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from functools import cache
from typing import Any, Iterator

from .. import metrics
from ..task import FROZEN, REPO, Item, Request, Task, read_jsonl

RAW_ZIP = REPO / "data" / "raw" / "sciprompt" / "sciprompt_data.zip"
RAW_ZIP_SHA256 = "2b0a53d715787e42b9105f06facb0ddae9b82b3bd6bc006a62714426ca524f2d"
FROZEN_DIR = FROZEN / "sciprompt"
KEY = "topic"
# Frozen truncation rule: the abstract is cut to its first MAX_ABSTRACT_CHARS characters
# (~15K tokens; with the largest option list, ~4K tokens, well inside the 32K context).
# The longest frozen abstract is far shorter, so nothing is cut.
MAX_ABSTRACT_CHARS = 60_000

# Released test files inside the zip (paper code: arXiv_script.py, s2orc_script.py,
# sdpra_script.py; Emerging NLP: repo data/EmergingNLP/test_set.csv).
# ("lines", text member, label member) or ("csv", member) with rows (abstract, label).
SOURCES: dict[str, tuple[str, ...]] = {
    "arxiv": ("lines", "data/arXiv/sample_random_test100_dataset.txt", "data/arXiv/sample_random_label_test100_dataset.txt"),
    "s2orc": ("lines", "data/S2ORC/300_random_sample_test_dataset.txt", "data/S2ORC/300_random_sample_test_labels.txt"),
    "sdpra": ("csv", "data/SDPRA/test_output.csv"),
    "emerging_nlp": ("csv", "data/EmergingNLP/test_set.csv"),
}
DATASETS = tuple(SOURCES)


def row_sha256(abstract: str, gold: str) -> str:
    canonical = json.dumps([abstract, gold], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _lines(zf: zipfile.ZipFile, member: str) -> list[str]:
    with zf.open(member) as fh:
        return [line.strip() for line in io.TextIOWrapper(fh, encoding="utf-8")]


def read_release(zf: zipfile.ZipFile, dataset: str) -> list[tuple[str, str]]:
    """(abstract, gold) per released test row, parsed exactly as the paper's loaders do.

    Line files: text-mode line iteration (universal newlines), each line stripped
    (arXiv_script.py / s2orc_script.py). CSV: rows of `csv.reader` (sdpra_script.py).
    """
    kind, *members = SOURCES[dataset]
    if kind == "lines":
        texts, labels = _lines(zf, members[0]), _lines(zf, members[1])
        if len(texts) != len(labels):
            raise ValueError(f"{dataset}: {len(texts)} texts vs {len(labels)} labels")
        return list(zip(texts, labels))
    with zf.open(members[0]) as fh:
        rows = list(csv.reader(io.TextIOWrapper(fh, encoding="utf-8", newline="")))
    return [(body, label) for body, label in rows]


@cache
def _release(dataset: str) -> tuple[tuple[str, str], ...]:
    if not RAW_ZIP.exists():
        raise FileNotFoundError(f"{RAW_ZIP} missing; see scripts/prepare_sciprompt.py for the download")
    if _sha256_file(RAW_ZIP) != RAW_ZIP_SHA256:
        raise ValueError(f"{RAW_ZIP} SHA-256 differs from the frozen release hash {RAW_ZIP_SHA256}")
    with zipfile.ZipFile(RAW_ZIP) as zf:
        return tuple(read_release(zf, dataset))


def _clip(text: str) -> tuple[str, bool]:
    return (text[:MAX_ABSTRACT_CHARS], True) if len(text) > MAX_ABSTRACT_CHARS else (text, False)


class SciPrompt(Task):
    dataset: str

    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.name = f"sciprompt/{dataset}"
        self.prompt_file = f"sciprompt_{dataset}.json"

    @property
    def labels(self) -> tuple[str, ...]:
        """Gold labels in the paper's label-id order (label_mappings/*.json); also the tie-break order."""
        return tuple(self.prompt()["_label_map"].values())

    def items(self) -> list[Item]:
        out = []
        for r in read_jsonl(FROZEN_DIR / f"{self.dataset}.jsonl"):
            abstract = r["abstract"] if "abstract" in r else _release(self.dataset)[r["row"]][0]
            if row_sha256(abstract, r["gold"]) != r["sha256"]:
                raise ValueError(f"{r['id']}: abstract/gold differ from the frozen hash")
            out.append(Item(id=r["id"], gold=r["gold"], fields={"abstract": abstract}))
        return out

    def _state(self, item: Item) -> tuple[dict[str, str], bool]:
        abstract, cut = _clip(item.fields["abstract"])
        return {"abstract": abstract}, cut

    def requests(self, item: Item) -> Iterator[Request]:
        state, _ = self._state(item)
        yield Request(item_id=item.id, key=KEY, state=state, questions={KEY: self.prompt()["question"]})

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        to_gold = self.prompt()["_label_map"]
        probs = answers[KEY][KEY]["probabilities"]
        p = {gold: float(probs.get(option, 0.0)) for option, gold in to_gold.items()}
        labels = self.labels
        # argmax; exact ties go to the earlier label in the paper's label-id order.
        label = max(labels, key=lambda lab: (p[lab], -labels.index(lab)))
        return {"label": label, "p_label": p[label]}

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        labels = list(self.labels)
        gold = [it.gold for it in items]
        pred = [predictions[it.id]["label"] for it in items]
        excluded = [r for r in read_jsonl(FROZEN_DIR / "excluded.jsonl") if r["dataset"] == self.dataset]
        return {
            "n": len(items),
            "accuracy": metrics.accuracy(gold, pred),
            "macro_f1": metrics.macro_f1(gold, pred, labels=labels),
            "weighted_f1": metrics.weighted_f1(gold, pred, labels=labels),
            "per_class_f1": metrics.per_class_f1(gold, pred, labels=labels),
            "n_gold": {lab: gold.count(lab) for lab in labels},
            "n_pred": {lab: pred.count(lab) for lab in labels},
            "n_excluded_short": sum(r["reason"] == "short_abstract" for r in excluded),
            "n_excluded_non_english": sum(r["reason"] == "non_english" for r in excluded),
            "n_truncated": sum(self._state(it)[1] for it in items),
        }


TASKS = {f"sciprompt/{ds}": (lambda ds=ds: SciPrompt(ds)) for ds in DATASETS}
