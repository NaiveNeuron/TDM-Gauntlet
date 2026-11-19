"""SMARTER (Nghiem & Daumé, ACL 2026): explained toxicity classification.

Replaces the paper's self-augmented (SFT + DPO) Llama-3.1-8B / COT-T5-XL
classifiers on three English tasks (paper Section 4.1, Table 4):

- `smarter/hatexplain`: HateXplain posts (Gab, Twitter), {Normal, Offensive, Hate}.
- `smarter/latent_hate`: Latent Hatred Stage-1 tweets,
  {Not Hate, Explicit Hate, Implicit Hate}.
- `smarter/implicit_hate`: Latent Hatred Stage-2 implicit-hate tweets, six
  fine-grained categories (the paper's Table 4 label space; "other" excluded).

Input: one post. Gold: the released label. Primary metric (paper Section 5.2,
Table 1): macro-F1, the unweighted mean of per-class F1, per dataset.

The authors release neither their processed data nor their test files, so the
test sets are reconstructions with the authors' own released sampling code
(`hate_dpo_public` `split_train_test`, seed 123, Table 4 split ratios; then
`k_shot_sample` of K_test per class from the test split, seed 42) — see
`scripts/prepare_smarter.py` and `data/manifests/smarter.json`.

HateXplain (MIT) is frozen with text. The Latent Hatred corpus is not
redistributed: its frozen files hold tweet IDs, row coordinates, gold labels and
a SHA-256 per row, and `items()` rebuilds the text from
`data/raw/smarter/implicit-hate-corpus.zip`, verifying each hash.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from functools import cache
from pathlib import Path
from typing import Any, Iterator

from .. import metrics
from ..task import FROZEN, REPO, Item, Request, Task, read_jsonl

RAW_DIR = REPO / "data" / "raw" / "smarter"
FROZEN_DIR = FROZEN / "smarter"
LATENT_ZIP = RAW_DIR / "implicit-hate-corpus.zip"
LATENT_ZIP_SHA256 = "140ba2af7e0e5e298caf6c587b965755c7e3e03e9d1fc85bf46dd39f2745aa29"
HATEXPLAIN_JSON = RAW_DIR / "hatexplain" / "Data" / "dataset.json"
HATEXPLAIN_JSON_SHA256 = "63bb3340fee0ec469b09690d04cb68f7c187787dd8b83807f071892c084967fb"

# Label inventories in the paper's Table 4 order (also the argmax tie-break order).
LABELS: dict[str, tuple[str, ...]] = {
    "hatexplain": ("Normal", "Offensive", "Hate"),
    "latent_hate": ("Not Hate", "Explicit Hate", "Implicit Hate"),
    "implicit_hate": (
        "White Grievance",
        "Incitement to Violence",
        "Inferiority Language",
        "Irony",
        "Stereotypes and Misinformation",
        "Threatening and Intimidation",
    ),
}
# Release label -> paper label. HateXplain labels are per-annotator (majority
# taken in the prepare script); Latent Hatred labels are the TSV `class` /
# `implicit_class` columns. The Stage-2 `other` class is outside the paper's
# label space and is dropped before splitting.
RELEASE_LABELS: dict[str, dict[str, str]] = {
    "hatexplain": {"normal": "Normal", "offensive": "Offensive", "hatespeech": "Hate"},
    "latent_hate": {"not_hate": "Not Hate", "explicit_hate": "Explicit Hate", "implicit_hate": "Implicit Hate"},
    "implicit_hate": {
        "white_grievance": "White Grievance",
        "incitement": "Incitement to Violence",
        "inferiority": "Inferiority Language",
        "irony": "Irony",
        "stereotypical": "Stereotypes and Misinformation",
        "threatening": "Threatening and Intimidation",
    },
}
# Latent Hatred zip members: (posts file with text, row-aligned file with tweet IDs).
LATENT_MEMBERS = {
    "latent_hate": ("implicit-hate-corpus/implicit_hate_v1_stg1_posts.tsv", "implicit-hate-corpus/implicit_hate_v1_stg1.tsv"),
    "implicit_hate": ("implicit-hate-corpus/implicit_hate_v1_stg2_posts.tsv", "implicit-hate-corpus/implicit_hate_v1_stg2.tsv"),
}
SUBSETS = tuple(LABELS)
KEY = "label"
# Frozen truncation rule: the post is cut to its first MAX_POST_CHARS characters
# (~9K tokens, well inside the 32K context). The longest released post is 801
# characters, so nothing is truncated.
MAX_POST_CHARS = 36_000

_HANDLE = re.compile(r"(?<![\w@])@\w+")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)


def preprocess(text: str) -> str:
    """Paper Appendix B: user handles -> '<user>', URLs removed. Nothing else."""
    return _URL.sub("", _HANDLE.sub("<user>", text))


def row_sha256(text: str, gold: str) -> str:
    canonical = json.dumps([text, gold], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_tsv(zf: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    with zf.open(member) as fh:
        return list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8", newline=""), delimiter="\t"))


@cache
def latent_rows(subset: str) -> tuple[dict[str, str], ...]:
    """Row-aligned join of a Latent Hatred posts file with its ID file.

    Each row: uid (tweet ID or SAP_ ID), post (raw text), label (release label),
    row (0-based data row). The two files share row order; alignment is checked
    through the label column they both carry.
    """
    if not LATENT_ZIP.exists():
        raise FileNotFoundError(f"{LATENT_ZIP} missing; run `scripts/prepare_smarter.py` (see its docstring)")
    if sha256_file(LATENT_ZIP) != LATENT_ZIP_SHA256:
        raise ValueError(f"{LATENT_ZIP} SHA-256 differs from the frozen release hash {LATENT_ZIP_SHA256}")
    posts_member, ids_member = LATENT_MEMBERS[subset]
    col = "class" if subset == "latent_hate" else "implicit_class"
    with zipfile.ZipFile(LATENT_ZIP) as zf:
        posts, ids = _read_tsv(zf, posts_member), _read_tsv(zf, ids_member)
    if len(posts) != len(ids):
        raise ValueError(f"{subset}: {len(posts)} posts vs {len(ids)} IDs")
    out = []
    for i, (p, r) in enumerate(zip(posts, ids)):
        if p[col] != r[col]:
            raise ValueError(f"{subset} row {i}: posts/ID files misaligned")
        out.append({"uid": r["ID"], "post": p["post"], "label": p[col], "row": i})
    return tuple(out)


def _clip(text: str) -> tuple[str, bool]:
    return (text[:MAX_POST_CHARS], True) if len(text) > MAX_POST_CHARS else (text, False)


class Smarter(Task):
    subset: str

    def __init__(self, subset: str, split: str = "test") -> None:
        self.subset = subset
        self.split = split
        # Non-test splits (v3 development samples of the authors' train/val parts) are
        # separate tasks under `<split>/smarter_<subset>` with the frozen test question.
        self.name = f"smarter/{subset}" if split == "test" else f"{split}/smarter_{subset}"
        self.prompt_file = f"smarter_{subset}.json"
        self.labels = LABELS[subset]

    def items(self) -> list[Item]:
        rows = read_jsonl(FROZEN_DIR / (f"{self.subset}.jsonl" if self.split == "test" else f"{self.subset}_{self.split}.jsonl"))
        out = []
        for r in rows:
            if self.subset == "hatexplain":
                text = r["text"]
            else:
                raw = latent_rows(self.subset)[r["row"]]
                if raw["uid"] != r["uid"]:
                    raise ValueError(f"{r['id']}: raw row {r['row']} has ID {raw['uid']}")
                text = preprocess(raw["post"])
            if row_sha256(text, r["gold"]) != r["sha256"]:
                raise ValueError(f"{r['id']}: text/gold differ from the frozen hash")
            out.append(Item(id=r["id"], gold=r["gold"], fields={"post": text}))
        return out

    def _state(self, item: Item) -> tuple[dict[str, str], bool]:
        post, cut = _clip(item.fields["post"])
        return {"post": post}, cut

    def requests(self, item: Item) -> Iterator[Request]:
        state, _ = self._state(item)
        yield Request(item_id=item.id, key=KEY, state=state, questions={KEY: self.prompt()["question"]})

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        probs = answers[KEY][KEY]["probabilities"]
        p = {lab: float(probs.get(lab, 0.0)) for lab in self.labels}
        # argmax; exact ties go to the earlier label in Table 4 order.
        label = max(self.labels, key=lambda lab: (p[lab], -self.labels.index(lab)))
        return {"label": label, "probabilities": p}

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        labels = list(self.labels)
        gold = [it.gold for it in items]
        pred = [predictions[it.id]["label"] for it in items]
        excluded = json.loads((FROZEN_DIR / "non_english.json").read_text())["excluded"]
        return {
            "n": len(items),
            "macro_f1": metrics.macro_f1(gold, pred, labels=labels),
            "per_class_f1": metrics.per_class_f1(gold, pred, labels=labels),
            "accuracy": metrics.accuracy(gold, pred),
            "n_gold": {lab: gold.count(lab) for lab in labels},
            "n_pred": {lab: pred.count(lab) for lab in labels},
            "n_excluded_non_english": sum(1 for e in excluded if e["subset"] == self.subset),
            "n_truncated": sum(self._state(it)[1] for it in items),
        }


TASKS = {
    **{f"smarter/{s}": (lambda s=s: Smarter(s)) for s in SUBSETS},
    **{f"dev_v3/smarter_{s}": (lambda s=s: Smarter(s, "dev_v3")) for s in SUBSETS},
}
