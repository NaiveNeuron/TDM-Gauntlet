"""DREsS_New (Yoo et al., ACL 2025): rubric-based scoring of EFL essays.

Input: essay prompt + student essay. Gold: instructor scores on three rubrics
(content, organization, language), each 1.0-5.0 in 0.5 steps; total = their sum
(paper Section 3.1). Replaces the paper's SFT scorers (BERT / Llama 3.1 8B
trained on DREsS_New + DREsS_Std. + DREsS_CASE). Primary metric (paper Tables
3-5): QWK per rubric and QWK on the total score.

Licence/access: DREsS is released behind a consent form
(https://haneul-yoo.github.io/dress/) whose terms forbid re-sharing, so no essay
text or score is stored in the repository. `scripts/prepare_dress_new.py` freezes an ID +
SHA-256 manifest of the test split to data/frozen/dress_new/test_ids.jsonl, and
`items()` rebuilds the test set from the consented download in
data/raw/dress_new/, verifying every row hash.

Test split: the paper states only "We split DREsS_New into training,
validation, and test sets in a 6:2:2 ratio with a random seed of 22"
(Appendix A) and releases no split file or code. Frozen reconstruction:
scikit-learn `train_test_split(range(n_rows), test_size=0.2, random_state=22)`
over rows in released file order (the conventional implementation of that
sentence; 2,279 rows -> 456 test rows). Not guaranteed to be the authors' split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import unicodedata
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split

from ..task import FROZEN, REPO, Item, Request, Task, read_jsonl

RAW_DIR = REPO / "data" / "raw" / "dress_new"
FROZEN_DIR = FROZEN / "dress_new"
RUBRICS = ("content", "organization", "language")
LEVELS = tuple(1.0 + 0.5 * i for i in range(9))  # score-question level index i -> score
KEY = "rubrics"
SPLIT_SEED = 22
TEST_SIZE = 0.2
# English-only rule (every row sent to the model must be English): a row is excluded
# when more than this share of its alphabetic characters (prompt + essay) are outside
# the Latin script.
MAX_NON_LATIN_SHARE = 0.05
# Frozen truncation rule: the essay prompt is cut to its first MAX_PROMPT_CHARS and
# the essay to its first MAX_ESSAY_CHARS characters (~17K tokens together, inside
# the 32K context with the ~6K-char question block). Paper: 313 words per essay.
MAX_PROMPT_CHARS = 8_000
MAX_ESSAY_CHARS = 60_000
RAW_SUFFIXES = (".tsv", ".csv", ".jsonl", ".json")
_ALIASES = {"essay_prompt": "prompt", "essay_id": "id", "text": "essay", "org": "organization", "lang": "language"}


def find_raw_file(raw_dir: Path = RAW_DIR) -> Path:
    """The single DREsS_New file of the consented download (DREsS_New.tsv etc.)."""
    hits = sorted(
        p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in RAW_SUFFIXES and "dress_new" in p.name.lower()
    )
    if len(hits) != 1:
        raise FileNotFoundError(
            f"expected exactly one DREsS_New {'/'.join(RAW_SUFFIXES)} file under {raw_dir}, found {[str(h) for h in hits]}. "
            "DREsS is gated: submit the consent form linked from https://haneul-yoo.github.io/dress/ "
            "and place the downloaded DREsS_New file in data/raw/dress_new/."
        )
    return hits[0]


def _read_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in (".tsv", ".csv"):
        with path.open(newline="", encoding="utf-8-sig") as fh:
            return list(csv.DictReader(fh, delimiter="\t" if suffix == ".tsv" else ","))
    if suffix == ".jsonl":
        return read_jsonl(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data["data"]


def _score(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    x = float(value)
    return None if math.isnan(x) else x


def load_raw(path: Path) -> list[dict[str, Any]]:
    """Rows in file order with normalised fields: id, prompt, essay, rubric scores, total."""
    rows = []
    for i, rec in enumerate(_read_records(path)):
        norm = {_ALIASES.get(k.strip().lower(), k.strip().lower()): v for k, v in rec.items() if k is not None}
        missing = [f for f in ("id", "prompt", "essay", *RUBRICS) if f not in norm]
        if missing:
            raise ValueError(f"{path.name} row {i}: missing columns {missing}; got {sorted(norm)}")
        rows.append(
            {
                "row_index": i,
                "id": str(norm["id"]).strip(),
                "prompt": norm["prompt"] or "",
                "essay": norm["essay"] or "",
                **{r: _score(norm[r]) for r in RUBRICS},
                "total": _score(norm.get("total")),
            }
        )
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path.name}: duplicate essay ids")
    return rows


def test_row_indices(n_rows: int) -> list[int]:
    """Frozen 6:2:2 reconstruction: the 20% test part of the seed-22 split, in file order."""
    _, test = train_test_split(np.arange(n_rows), test_size=TEST_SIZE, random_state=SPLIT_SEED)
    return sorted(int(i) for i in test)


def row_sha256(row: dict[str, Any]) -> str:
    payload = {k: row[k] for k in ("id", "prompt", "essay", *RUBRICS)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def non_latin_share(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    other = sum(1 for ch in letters if not unicodedata.name(ch, "").startswith("LATIN"))
    return other / len(letters)


def gold_of(row: dict[str, Any]) -> dict[str, float]:
    gold = {r: float(row[r]) for r in RUBRICS}
    gold["total"] = sum(gold[r] for r in RUBRICS)  # paper: total = sum of the three rubric scores
    return gold


def _code(score: float) -> int:
    """Integer code for QWK: half-point units (1.0 -> 2, 5.0 -> 10; total 3.0 -> 6, 15.0 -> 30)."""
    return int(round(2 * score))


def _qwk(gold: list[float], pred: list[float], scale: range) -> float:
    """QWK over the fixed ordinal scale (plus any off-scale gold codes), so an empty
    intermediate level cannot shrink the distance between its neighbours."""
    g = [_code(x) for x in gold]
    p = [_code(x) for x in pred]
    labels = sorted(set(scale) | set(g) | set(p))
    return float(cohen_kappa_score(g, p, labels=labels, weights="quadratic"))


def _clip(text: str, limit: int) -> tuple[str, bool]:
    return (text[:limit], True) if len(text) > limit else (text, False)


class DressNew(Task):
    name = "dress_new"
    prompt_file = "dress_new.json"

    def __init__(self, raw_dir: Path = RAW_DIR, frozen_dir: Path = FROZEN_DIR) -> None:
        self.raw_dir = raw_dir
        self.frozen_dir = frozen_dir

    def items(self) -> list[Item]:
        frozen = read_jsonl(self.frozen_dir / "test_ids.jsonl")
        by_id = {r["id"]: r for r in load_raw(find_raw_file(self.raw_dir))}
        items = []
        for entry in frozen:
            row = by_id.get(entry["id"])
            if row is None or row_sha256(row) != entry["sha256"]:
                raise ValueError(f"DREsS_New row {entry['id']!r} missing or changed since the freeze (sha256 mismatch)")
            items.append(Item(id=row["id"], gold=gold_of(row), fields={"prompt": row["prompt"], "essay": row["essay"]}))
        return items

    def _state(self, item: Item) -> tuple[dict[str, str], bool]:
        prompt, cut_p = _clip(item.fields["prompt"], MAX_PROMPT_CHARS)
        essay, cut_e = _clip(item.fields["essay"], MAX_ESSAY_CHARS)
        return {"essay_prompt": prompt, "essay": essay}, cut_p or cut_e

    def requests(self, item: Item) -> Iterator[Request]:
        state, _ = self._state(item)
        yield Request(item_id=item.id, key=KEY, state=state, questions=self.prompt()["questions"])

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        pred: dict[str, Any] = {}
        expected: dict[str, float] = {}
        for rubric in RUBRICS:
            ans = answers[KEY][rubric]
            probs = [float(ans["probabilities"].get(str(i), 0.0)) for i in range(len(LEVELS))]
            # Frozen rule: argmax level; exact ties go to the lower level.
            pred[rubric] = LEVELS[max(range(len(LEVELS)), key=lambda i: (probs[i], -i))]
            # Supplementary: expected level index -> score value (not rounded).
            expected[rubric] = 1.0 + 0.5 * float(ans["score"])
        pred["total"] = sum(pred[r] for r in RUBRICS)
        pred["expected"] = expected
        return pred

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        rubric_scale = range(_code(LEVELS[0]), _code(LEVELS[-1]) + 1)
        total_scale = range(_code(3 * LEVELS[0]), _code(3 * LEVELS[-1]) + 1)
        out: dict[str, Any] = {"n": len(items)}
        for rubric in RUBRICS:
            out[f"qwk_{rubric}"] = _qwk([it.gold[rubric] for it in items], [predictions[it.id][rubric] for it in items], rubric_scale)
        out["qwk_total"] = _qwk([it.gold["total"] for it in items], [predictions[it.id]["total"] for it in items], total_scale)

        # Supplementary (fixed a priori, not the primary rule): expected score rounded
        # half-up to the nearest 0.5 level instead of the argmax level.
        def nearest(x: float) -> float:
            return min(max(math.floor(2 * x + 0.5) / 2, LEVELS[0]), LEVELS[-1])

        exp = {it.id: {r: nearest(predictions[it.id]["expected"][r]) for r in RUBRICS} for it in items}
        supp = {
            f"qwk_{r}": _qwk([it.gold[r] for it in items], [exp[it.id][r] for it in items], rubric_scale) for r in RUBRICS
        }
        supp["qwk_total"] = _qwk(
            [it.gold["total"] for it in items], [sum(exp[it.id].values()) for it in items], total_scale
        )
        out["supplementary_expected_rounded"] = supp
        out["n_truncated"] = sum(self._state(it)[1] for it in items)
        return out


TASKS = {"dress_new": DressNew}
