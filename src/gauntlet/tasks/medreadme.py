"""MedReadMe (Jiang & Xu, EMNLP 2024): medical sentence readability.

"MedReadMe: A Systematic Study for Fine-grained Sentence Readability in Medical
Domain", https://aclanthology.org/2024.emnlp-main.958/.

Input: one English sentence from medical text (15 sources: Cochrane, PNAS, five
NIHR and five PLOS journals, eLife, medical Wikipedia, Merck Manuals). Gold: the
released human readability rating on the CEFR-based 1 (easiest) .. 6 (hardest)
scale; "+"/"-" ratings map to +/-0.3 and dev/test ratings are the mean of two
annotators (Section 2.2), so gold is continuous in [1, 6].

Test set: the paper's own split, i.e. the 1,140 rows with split == "test" in the
released dataset/readability.csv (paper Section 5: 2,587/784/1,140). The span
(complex-span identification) task is out of scope.

Replaced predictor: the supervised RoBERTa-large readability regressor fine-tuned
on MedReadMe (Table 7). Jev answers one 6-level `score` question per sentence
(paper Appendix H prompt, zero-shot); the prediction is 1 + the expected level
index under Jev's level probabilities.

Metric (paper Tables 4 and 7): Pearson correlation between predicted and gold
readability, computed per source group (Cochrane, PNAS, NIHR Series, eLife, PLOS
Series, Wiki, MSD; NIHR and PLOS each pool their five journals) and averaged over
the 7 groups; the paper also reports the sample standard deviation over groups.

The release has no licence, so rows are not redistributed: `items()` rebuilds
them from data/raw/medreadme/readability.csv (fetched by
scripts/prepare_medreadme.py) and checks each row against the frozen ID + SHA-256
manifest data/frozen/medreadme/test_manifest.jsonl.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy import stats

from ..task import FROZEN, REPO, Item, Request, Task, read_jsonl

RAW_DIR = REPO / "data" / "raw" / "medreadme"
RAW_CSV = RAW_DIR / "readability.csv"
REPO_URL = "https://github.com/chaojiang06/medreadme"
COMMIT = "7c52205d18f16b80e3ffccb4e9da88b1942df310"
CSV_URL = f"https://raw.githubusercontent.com/chaojiang06/medreadme/{COMMIT}/dataset/readability.csv"
CSV_SHA256 = "54082bb202d8916f296b37c9b304fd9f2c102920587ea6c346ea626be3d44ab1"
FROZEN_DIR = FROZEN / "medreadme"
MANIFEST = FROZEN_DIR / "test_manifest.jsonl"

KEY = "readability"
N_LEVELS = 6  # readability 1..6; Jev level index i <-> readability i + 1
# Frozen truncation rule: the sentence is cut to its first MAX_CHARS characters
# (~25K tokens at worst, inside the 32K context). The longest test sentence is
# 563 characters, so nothing is truncated.
MAX_CHARS = 100_000

# Table 4/7 row order. NIHR and PLOS "are aggregations of 5 sources for each"
# (Table 4 caption): their sentences are pooled into one group.
GROUPS = ("Cochrane", "PNAS", "NIHR Series", "eLife", "PLOS Series", "Wiki", "MSD")
SOURCE_GROUP = {
    "cochrane_all": "Cochrane",
    "pnas": "PNAS",
    "nhir_Efficaacy_and_Mechanism_Evaluation": "NIHR Series",
    "nhir_Health_Services_and_Delivery_Research": "NIHR Series",
    "nhir_Health_Technology_Assessment": "NIHR Series",
    "nhir_Programme_Grants_for_Applied_Research": "NIHR Series",
    "nhir_Public_Health_Research": "NIHR Series",
    "elife": "eLife",
    "plos_biology": "PLOS Series",
    "plos_computational_biology": "PLOS Series",
    "plos_genetics": "PLOS Series",
    "plos_neglected_tropical_diseases": "PLOS Series",
    "plos_pathogens": "PLOS Series",
    "wiki": "Wiki",
    "MSD": "MSD",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw(path: Path = RAW_CSV) -> list[dict[str, str]]:
    """Rows of readability.csv as released (raw strings). The unnamed first column is the row id."""
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def row_sha256(row: dict[str, str]) -> str:
    blob = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def item_id(row: dict[str, str]) -> str:
    return f"medreadme-{int(row['']):04d}"


def _clip(text: str) -> tuple[str, bool]:
    return (text[:MAX_CHARS], True) if len(text) > MAX_CHARS else (text, False)


def _corr(kind: str, x: list[float], y: list[float]) -> float:
    """Pearson/Spearman; NaN when undefined (fewer than 2 items or a constant side)."""
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan
    fn = stats.pearsonr if kind == "pearson" else stats.spearmanr
    return float(fn(x, y)[0])


def _json_num(v: float) -> float | None:
    return None if math.isnan(v) else v


class MedReadMe(Task):
    name = "medreadme"
    prompt_file = "medreadme.json"

    def items(self) -> list[Item]:
        if not RAW_CSV.exists():
            raise FileNotFoundError(f"{RAW_CSV} missing; run `uv run python scripts/prepare_medreadme.py`")
        rows = {item_id(r): r for r in load_raw(RAW_CSV)}
        out = []
        for m in read_jsonl(MANIFEST):
            row = rows.get(m["id"])
            if row is None or row_sha256(row) != m["sha256"]:
                raise ValueError(f"{m['id']}: raw row missing or SHA-256 mismatch against {MANIFEST}")
            out.append(
                Item(
                    id=m["id"],
                    gold=float(row["Readability"]),
                    fields={"sentence": row["Sentence"], "source": row["Source"], "side": row["Side"]},
                )
            )
        return out

    def _state(self, item: Item) -> tuple[dict[str, str], bool]:
        sentence, cut = _clip(item.fields["sentence"])
        return {"sentence": sentence}, cut

    def requests(self, item: Item) -> Iterator[Request]:
        state, _ = self._state(item)
        yield Request(item_id=item.id, key=KEY, state=state, questions={KEY: self.prompt()["question"]})

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        ans = answers[KEY][KEY]
        probs = np.array([float(ans["probabilities"].get(str(i), 0.0)) for i in range(N_LEVELS)])
        # Expected level from the probabilities (renormalised); falls back to Jev's
        # own `score` only if no probability mass came back.
        expected = float(probs @ np.arange(N_LEVELS) / probs.sum()) if probs.sum() > 0 else float(ans["score"])
        # argmax level, exact ties -> lower level (reported only as a secondary view).
        level = int(np.argmax(probs)) + 1
        return {"readability": 1.0 + expected, "level": level}

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        by_group: dict[str, tuple[list[float], list[float]]] = {g: ([], []) for g in GROUPS}
        by_source: dict[str, tuple[list[float], list[float]]] = {s: ([], []) for s in SOURCE_GROUP}
        gold_all, pred_all, level_all = [], [], []
        for it in items:
            g, p = it.gold, predictions[it.id]["readability"]
            by_group[SOURCE_GROUP[it.fields["source"]]][0].append(g)
            by_group[SOURCE_GROUP[it.fields["source"]]][1].append(p)
            by_source[it.fields["source"]][0].append(g)
            by_source[it.fields["source"]][1].append(p)
            gold_all.append(g)
            pred_all.append(p)
            level_all.append(float(predictions[it.id]["level"]))

        group_pearson = {g: _corr("pearson", *by_group[g]) for g in GROUPS}
        group_spearman = {g: _corr("spearman", *by_group[g]) for g in GROUPS}
        source_pearson = {s: _corr("pearson", *by_source[s]) for s in SOURCE_GROUP if by_source[s][0]}
        gp = list(group_pearson.values())
        return {
            "n": len(items),
            # Primary (Tables 4/7 "Mean"): mean over the 7 source groups of per-group Pearson.
            "pearson_mean_groups": _json_num(float(np.mean(gp))),
            "pearson_std_groups": _json_num(float(np.std(gp, ddof=1))),  # paper's "± Std" (sample SD)
            "pearson_by_group": {g: _json_num(v) for g, v in group_pearson.items()},
            "spearman_mean_groups": _json_num(float(np.mean(list(group_spearman.values())))),
            "spearman_by_group": {g: _json_num(v) for g, v in group_spearman.items()},
            # Secondary views (not reported by the paper).
            "pearson_pooled": _json_num(_corr("pearson", gold_all, pred_all)),
            "spearman_pooled": _json_num(_corr("spearman", gold_all, pred_all)),
            "pearson_pooled_argmax_level": _json_num(_corr("pearson", gold_all, level_all)),
            "pearson_mean_15_sources": _json_num(float(np.mean(list(source_pearson.values())))),
            "pearson_by_source": {s: _json_num(v) for s, v in source_pearson.items()},
            "n_by_group": {g: len(by_group[g][0]) for g in GROUPS},
            "n_truncated": sum(self._state(it)[1] for it in items),
        }


TASKS = {"medreadme": MedReadMe}
