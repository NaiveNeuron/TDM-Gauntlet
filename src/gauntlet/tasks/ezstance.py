"""EZ-STANCE: English zero-shot stance detection (Zhao & Caragea, ACL 2024).

Input: one tweet (the release's preprocessed `Text` column, which the paper's
models use) and one target, either a noun phrase or a claim. Gold: the
release's `Stance 1` label, one of AGAINST / FAVOR / NONE (the paper calls
these Against / Favor / Neutral). Replaces the paper's fine-tuned
BART-MNLI-ep classifier. Primary metric (paper Section 6): macro-averaged F1
over the three classes (the authors' `compute_f1` averages the per-class F1 of
labels 0,1,2), reported separately for mixed (M), noun-phrase (N) and claim (C)
targets as in Tables 7-8; per-class F1 as in Tables 16-19.

Tasks:
- `ezstance/subtaskA`: target-based ZSSD test set (2,663 N + 5,135 C = 7,798).
- `ezstance/subtaskB`: domain-based ZSSD, eight leave-one-domain-out test
  sets (each domain in turn is the zero-shot test domain; 46,011 pairs).
  Scored per domain, as in Table 8.

The release (Google Drive zip, no licence) is not redistributed: the frozen
files hold IDs, row coordinates, gold labels and a SHA-256 per row, and
`items()` rebuilds text and target from `data/raw/ezstance/ezstance_data.zip`
(fetched by `scripts/prepare_ezstance.py`), verifying each row hash.
A frozen list of test rows whose tweet contains non-English text is excluded
(the benchmark is English-only); see
`data/frozen/ezstance/excluded_non_english.jsonl`.

Both tasks send identical requests for identical (tweet, target, target type),
so Subtask A pairs that recur in a Subtask B test set hit the same cache entry.
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

RAW_ZIP = REPO / "data" / "raw" / "ezstance" / "ezstance_data.zip"
RAW_ZIP_SHA256 = "4683d124c8c8dde1b1282c3f534efa6c0b92e22335a8821249ef1d63cb26455b"
FROZEN_DIR = FROZEN / "ezstance"

# Release labels in the authors' label-id order (AGAINST=0, FAVOR=1, NONE=2);
# also the argmax tie-break order.
LABELS = ("AGAINST", "FAVOR", "NONE")
TARGET_TYPES = ("noun_phrase", "claim")
KEY = "stance"
# Release domain directory -> paper abbreviation (Table 3), in Table 8 column order.
DOMAINS = {
    "covid19_domain": "CE",
    "world_event_domain": "WE",
    "education_and_culture_domain": "EdC",
    "consumption_and_entertainment_domain": "EnC",
    "sports_domain": "S",
    "rights_domain": "R",
    "environmental_protection_domain": "EP",
    "politic": "P",
}
# Frozen truncation rule: tweet and target are each cut to their first
# MAX_FIELD_CHARS characters (~9K tokens each, inside the 32K context). The
# longest released test tweet is far shorter, so nothing is truncated.
MAX_FIELD_CHARS = 36_000


def row_sha256(text: str, target: str, gold: str) -> str:
    canonical = json.dumps([text, target, gold], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_member(zf: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    with zf.open(member) as fh:
        return list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", newline="")))


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@cache
def _verified_zip() -> zipfile.ZipFile:
    if not RAW_ZIP.exists():
        raise FileNotFoundError(f"{RAW_ZIP} missing; run `uv run python scripts/prepare_ezstance.py`")
    if _sha256_file(RAW_ZIP) != RAW_ZIP_SHA256:
        raise ValueError(f"{RAW_ZIP} SHA-256 differs from the frozen release hash {RAW_ZIP_SHA256}")
    return zipfile.ZipFile(RAW_ZIP)


@cache
def _raw_rows(member: str) -> tuple[dict[str, str], ...]:
    return tuple(read_member(_verified_zip(), member))


def _clip(text: str) -> tuple[str, bool]:
    return (text[:MAX_FIELD_CHARS], True) if len(text) > MAX_FIELD_CHARS else (text, False)


def _block(items: list[Item], gold: list[str], pred: list[str], idx: list[int]) -> dict[str, Any]:
    g = [gold[i] for i in idx]
    p = [pred[i] for i in idx]
    if not idx:
        return {"n": 0}
    return {
        "n": len(idx),
        "macro_f1": metrics.macro_f1(g, p, labels=list(LABELS)),
        "per_class_f1": metrics.per_class_f1(g, p, labels=list(LABELS)),
        "accuracy": metrics.accuracy(g, p),
        "n_gold": {lab: g.count(lab) for lab in LABELS},
        "n_pred": {lab: p.count(lab) for lab in LABELS},
    }


def _mnc(items: list[Item], gold: list[str], pred: list[str], idx: list[int]) -> dict[str, Any]:
    """Mixed / noun-phrase / claim breakdown over the item indices `idx`."""
    return {
        "M": _block(items, gold, pred, idx),
        "N": _block(items, gold, pred, [i for i in idx if items[i].fields["target_type"] == "noun_phrase"]),
        "C": _block(items, gold, pred, [i for i in idx if items[i].fields["target_type"] == "claim"]),
    }


class EZStance(Task):
    prompt_file = "ezstance.json"
    split: str

    def __init__(self, split: str) -> None:
        self.split = split
        self.name = f"ezstance/{split}"

    def items(self) -> list[Item]:
        out = []
        for r in read_jsonl(FROZEN_DIR / f"{self.split}.jsonl"):
            raw = _raw_rows(r["member"])[r["row"]]
            text, target, gold = raw["Text"], raw["Target 1"], raw["Stance 1"]
            if gold != r["gold"] or row_sha256(text, target, gold) != r["sha256"]:
                raise ValueError(f"{r['id']}: raw row differs from the frozen hash")
            fields = {"text": text, "target": target, "target_type": r["target_type"]}
            if "domain" in r:
                fields["domain"] = r["domain"]
            out.append(Item(id=r["id"], gold=gold, fields=fields))
        return out

    def _state(self, item: Item) -> tuple[dict[str, str], bool]:
        text, cut_text = _clip(item.fields["text"])
        target, cut_target = _clip(item.fields["target"])
        return {"tweet": text, "target": target}, cut_text or cut_target

    def requests(self, item: Item) -> Iterator[Request]:
        state, _ = self._state(item)
        question = self.prompt()[item.fields["target_type"]]
        yield Request(item_id=item.id, key=KEY, state=state, questions={KEY: question})

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        to_gold = self.prompt()["_label_map"]
        probs = answers[KEY][KEY]["probabilities"]
        p = {to_gold[choice]: float(probs.get(choice, 0.0)) for choice in to_gold}
        # argmax; exact ties go to the earlier label in LABELS order.
        label = max(LABELS, key=lambda lab: (p[lab], -LABELS.index(lab)))
        return {"label": label, "probabilities": p}

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        gold = [it.gold for it in items]
        pred = [predictions[it.id]["label"] for it in items]
        excluded = read_jsonl(FROZEN_DIR / "excluded_non_english.jsonl")
        out: dict[str, Any] = {"n": len(items)}
        if self.split == "subtaskA":
            out.update(_mnc(items, gold, pred, list(range(len(items)))))
            out["macro_f1"] = out["M"]["macro_f1"]
        else:
            by_domain = {}
            for domain, abbr in DOMAINS.items():
                idx = [i for i, it in enumerate(items) if it.fields["domain"] == domain]
                by_domain[abbr] = _mnc(items, gold, pred, idx)
            out["by_domain"] = by_domain
            # Unweighted mean over the eight zero-shot domains (derived; the paper
            # reports only the per-domain values of Table 8).
            for tt in ("M", "N", "C"):
                out[f"macro_f1_mean_over_domains_{tt}"] = metrics.mean(
                    [d[tt]["macro_f1"] for d in by_domain.values()]
                )
            out["macro_f1"] = out["macro_f1_mean_over_domains_M"]
        out["n_excluded_non_english"] = sum(1 for r in excluded if r["split"] == self.split)
        out["n_truncated"] = sum(self._state(it)[1] for it in items)
        return out


TASKS = {f"ezstance/{split}": (lambda split=split: EZStance(split)) for split in ("subtaskA", "subtaskB")}
