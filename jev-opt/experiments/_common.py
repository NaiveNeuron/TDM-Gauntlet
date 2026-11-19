"""Shared helpers for the experiment `prepare.py` scripts (imported, not run).

Every experiment writes gitignored `data/{train,val,test}.jsonl` + `data/manifest.json`:
train/val are drawn only from the paper's released train/dev data; test is exactly the
gauntlet's frozen English test set (`Task.items()` mapped through the adapter's own
`_state`, so field values equal what the gauntlet sends to Jev).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

# Foreign function words for the train/val language screen (the gauntlet's test screens
# were adjudicated flag by flag; train/val use this automatic approximation of them).
_FOREIGN = set(
    "der die das und nicht ist ich sie wir ein eine mit auf für auch "  # de
    "el los las una por para con como pero más está muy porque "  # es
    "les des une est pas pour avec dans sur mais très qui "  # fr
    "het een niet zijn maar voor wij ook "  # nl
    "não uma com para mas você muito "  # pt
    "il gli della che non sono per una anche questo "  # it
    .split()
)


def looks_english(text: str) -> bool:
    """False when >5% of letters are non-ASCII or the text has 2+ distinct foreign function words."""
    letters = [c for c in text if c.isalpha()]
    if letters and sum(not c.isascii() for c in letters) / len(letters) > 0.05:
        return False
    words = set(re.findall(r"[^\W\d_]+", text.lower()))
    return len(words & _FOREIGN) < 2


def stratified(rows: list[dict], n: int, seed: int, key: str = "label") -> list[dict]:
    """Deterministic label-proportional sample of `n` rows (all rows when n >= len)."""
    if n >= len(rows):
        out = list(rows)
        random.Random(seed).shuffle(out)
        return out
    rng = random.Random(seed)
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r[key], []).append(r)
    labels = sorted(by)
    quota = {lab: max(1, round(n * len(by[lab]) / len(rows))) for lab in labels}
    biggest = max(labels, key=lambda lab: len(by[lab]))
    quota[biggest] += n - sum(quota.values())
    picked = [r for lab in labels for r in rng.sample(by[lab], min(quota[lab], len(by[lab])))]
    rng.shuffle(picked)
    return picked


def gauntlet_task(name: str):
    from gauntlet.tasks import REGISTRY

    return REGISTRY[name]()


def gauntlet_n(run: str) -> int:
    return int(json.loads((REPO / "runs" / run / "scores.json").read_text())["n"])


def test_rows(task: Any, run: str, row: Callable[[Any, dict[str, str]], dict[str, Any] | None]) -> list[dict]:
    """Gauntlet test rows: `row(item, state)` maps each item (and its exact Jev state) to a row."""
    items = task.items()
    rows = [r for r in (row(it, task._state(it)[0]) for it in items) if r is not None]
    expected = gauntlet_n(run)
    assert len(items) == expected, f"{run}: items() gave {len(items)} rows, runs/{run}/scores.json n={expected}"
    return rows


def write_splits(out: Path, splits: dict[str, list[dict]], manifest: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ids = {s: {r["id"] for r in rows} for s, rows in splits.items()}
    assert not ids["train"] & ids["val"] and not (ids["train"] | ids["val"]) & ids["test"], "split overlap"
    for split, rows in splits.items():
        assert len(ids[split]) == len(rows), f"duplicate ids in {split}"
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        **manifest,
        "counts": {s: {"n": len(rows), "labels": dict(sorted(Counter(r["label"] for r in rows).items()))} for s, rows in splits.items()},
        "sha256": {s: hashlib.sha256((out / f"{s}.jsonl").read_bytes()).hexdigest() for s in splits},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest["counts"]))

