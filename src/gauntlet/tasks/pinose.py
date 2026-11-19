"""PiNose (Zhang et al., ACL 2024): non-factual content detection.

"Transferable and Efficient Non-Factual Content Detection via Probe Training
with Offline Consistency Checking", https://aclanthology.org/2024.acl-long.668/.

Four separate English test sets of 1,000 rows each (paper Table 1, p. 12352):
  pinose/true_false  standalone statements (Azaria & Mitchell True-False), q = empty
  pinose/nq          NQ question + Llama2-7B response
  pinose/triviaqa    TriviaQA question + Llama2-7B response
  pinose/webq        WebQuestions question + Llama2-7B response
Gold: binary factuality f in {True, False} (QA sets: human annotation against
the question's ground truth; True-False: dataset labels).

Replaced predictor: the PiNose probe (MLP on Llama2-7B layer-16 last-token
state). Jev answers one `noul` question, P(the response/statement is factually
correct). Metrics (paper Table 2, p. 12353): ROC-AUC and accuracy. The paper's
accuracy uses a threshold "determined by selecting the value that yields the
highest accuracy among 100 validation instances partitioned from the test
sets"; here the threshold is fixed a priori at 0.5 (P > 0.5 -> True).

The release has no licence, so rows are not redistributed: `items()` rebuilds
them from data/raw/pinose (fetched by scripts/prepare_pinose.py) and checks each
row against the frozen ID + SHA-256 manifest in data/frozen/pinose/<subset>/.
The reference answers of the QA sets are kept on the item for provenance but
are never sent to Jev (they would reveal the label).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterator

from .. import metrics
from ..task import FROZEN, REPO, Item, Request, Task, read_jsonl

RAW = REPO / "data" / "raw" / "pinose"
ZIP_NAME = "pinose_public.zip"
ZIP_URL = "https://drive.usercontent.google.com/download?id=1soYrfWUFDw8oZNjRud-bCU4BEx1Y9m5h&export=download&confirm=t"
ZIP_SHA256 = "53e8a7c738469a32cb21424349156bb0ebb2066380acd5747215f35b24b63b1a"

SUBSETS: dict[str, dict[str, Any]] = {
    "true_false": {
        "csv": "public/TrueFalseExp/tf_split_data/test.csv",
        "csv_sha256": "b799f3968eccb909c1898709a4e02fc814bafd96d8229169c9952cea48efdf65",
        "id_prefix": "tf",
        "kind": "statement",
        "prompt_file": "pinose_true_false.json",
    },
    "nq": {
        "csv": "public/QuestionAnsweringExp/data/NQ.csv",
        "csv_sha256": "28bc68a71e79a996c8538ca8eacf82decae9eada4c2662727eb3b005c6bd386a",
        "id_prefix": "nq",
        "kind": "qa",
        "prompt_file": "pinose_qa.json",
    },
    "triviaqa": {
        "csv": "public/QuestionAnsweringExp/data/TriviaQ.csv",
        "csv_sha256": "dc03d9179a5ceb1f2a1f132858cfcfe5de268ed89c08e65c4c5231b3cae1b27c",
        "id_prefix": "triviaqa",
        "kind": "qa",
        "prompt_file": "pinose_qa.json",
    },
    "webq": {
        "csv": "public/QuestionAnsweringExp/data/WebQ.csv",
        "csv_sha256": "485006a6675d683ab5dc37f072ef938b6fe977f20c04b0d58a91607a15023809",
        "id_prefix": "webq",
        "kind": "qa",
        "prompt_file": "pinose_qa.json",
    },
}

# Release label encodings: QA CSVs use "True"/"False", True-False uses "1"/"0".
GOLD = {"True": True, "False": False, "1": True, "0": False}
KEY = "factuality"
THRESHOLD = 0.5  # fixed a priori: predict True iff P(true) > THRESHOLD
# Frozen truncation rule: each text field is cut to its first MAX_FIELD_CHARS
# characters. The longest released test field is 260 chars, so nothing is cut.
MAX_FIELD_CHARS = 20_000

# Paper Table 2 (p. 12353), AUC / ACC in percent, Llama2-7B throughout.
PAPER_TABLE2: dict[str, dict[str, dict[str, float | None]]] = {
    method: {
        subset: {"auc": auc, "acc": acc}
        for subset, (auc, acc) in zip(("true_false", "nq", "triviaqa", "webq"), values)
    }
    for method, values in {
        "RepE": ((63.3, 61.9), (62.6, 60.8), (67.5, 64.3), (65.5, 63.5)),
        "SAPLMA": ((90.2, 83.7), (69.7, 67.8), (69.3, 64.5), (75.6, 69.5)),
        "SCGPT-PT": ((None, None), (72.3, 70.3), (76.4, 69.9), (72.5, 72.5)),
        "SCGPT-NLI": ((None, None), (77.4, 71.0), (76.9, 72.1), (77.4, 72.3)),
        "PPL-AVE": ((67.1, 64.1), (61.8, 59.2), (61.7, 58.5), (63.7, 63.3)),
        "PPL-MAX": ((60.7, 61.7), (54.2, 55.0), (57.6, 57.0), (56.4, 60.0)),
        "It-is-True": ((54.8, 60.1), (63.0, 59.5), (61.7, 59.1), (75.6, 71.0)),
        "PiNose": ((86.6, 82.1), (80.4, 72.5), (83.9, 73.8), (83.3, 76.0)),
    }.items()
}


def frozen_manifest_path(subset: str) -> Path:
    return FROZEN / "pinose" / subset / "manifest.jsonl"


def raw_rows(subset: str) -> list[dict[str, str]]:
    """Test rows exactly as released (all fields as strings, file order)."""
    path = RAW / SUBSETS[subset]["csv"]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `uv run python scripts/prepare_pinose.py`")
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def row_sha256(row: dict[str, str]) -> str:
    canonical = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _clip(text: str) -> tuple[str, bool]:
    return (text[:MAX_FIELD_CHARS], True) if len(text) > MAX_FIELD_CHARS else (text, False)


class PiNose(Task):
    def __init__(self, subset: str) -> None:
        self.subset = subset
        self.spec = SUBSETS[subset]
        self.name = f"pinose/{subset}"
        self.prompt_file = self.spec["prompt_file"]

    def items(self) -> list[Item]:
        rows = raw_rows(self.subset)
        manifest = read_jsonl(frozen_manifest_path(self.subset))
        if len(rows) != len(manifest):
            raise ValueError(f"{self.name}: {len(rows)} raw rows vs {len(manifest)} in frozen manifest")
        items = []
        for entry in manifest:
            row = rows[entry["row"]]
            if row_sha256(row) != entry["sha256"]:
                raise ValueError(f"{self.name}: row {entry['row']} does not match frozen sha256")
            if self.spec["kind"] == "qa":
                fields = {"question": row["question"], "response": row["response"], "reference_answer": row["answer"]}
            else:
                fields = {"statement": row["statement"]}
            items.append(Item(id=entry["id"], gold=GOLD[row["label"]], fields=fields))
        return items

    def _state(self, item: Item) -> tuple[dict[str, str], bool]:
        names = ("question", "response") if self.spec["kind"] == "qa" else ("statement",)
        clipped = {name: _clip(item.fields[name]) for name in names}
        return {name: text for name, (text, _) in clipped.items()}, any(cut for _, cut in clipped.values())

    def requests(self, item: Item) -> Iterator[Request]:
        state, _ = self._state(item)
        yield Request(item_id=item.id, key=KEY, state=state, questions={KEY: self.prompt()["question"]})

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        p_true = float(answers[KEY][KEY]["noul"])
        # Exactly 0.5 -> False (strict inequality, as in the release's test_classifier.py).
        return {"label": p_true > THRESHOLD, "p_true": p_true}

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        gold = [bool(it.gold) for it in items]
        pred = [bool(predictions[it.id]["label"]) for it in items]
        p_true = [float(predictions[it.id]["p_true"]) for it in items]
        n_true = sum(gold)
        return {
            "n": len(items),
            "auc": metrics.roc_auc([int(g) for g in gold], p_true),
            "accuracy": metrics.accuracy(gold, pred),
            "threshold": THRESHOLD,
            "n_gold_true": n_true,
            "pct_gold_true": 100.0 * n_true / len(items),
            "n_pred_true": sum(pred),
            "confusion": {
                "tp": sum(g and p for g, p in zip(gold, pred)),
                "fp": sum(p and not g for g, p in zip(gold, pred)),
                "fn": sum(g and not p for g, p in zip(gold, pred)),
                "tn": sum(not g and not p for g, p in zip(gold, pred)),
            },
            # Context: accuracy of always predicting the majority gold class.
            "majority_class_accuracy": max(n_true, len(items) - n_true) / len(items),
            "n_truncated": sum(self._state(it)[1] for it in items),
            "paper_table2_percent": {method: rows[self.subset] for method, rows in PAPER_TABLE2.items()},
        }


def _factory(subset: str) -> Callable[[], Task]:
    return lambda: PiNose(subset)


TASKS: dict[str, Callable[[], Task]] = {f"pinose/{subset}": _factory(subset) for subset in SUBSETS}
