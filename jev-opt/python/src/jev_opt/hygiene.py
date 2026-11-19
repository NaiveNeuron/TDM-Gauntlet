"""Split hygiene and anti-gaming (architecture §8).

- `load_split` refuses the test split unless inside `allow_test_split()` (only `evaluate_final`).
- During optimize the test file is fingerprinted with `os.stat` only: never opened.
- `ProgramGuard` rejects programs that copy train/val text, mention example ids, or grow too large.
- The consumption ledger (`test_consumed.json` + `<runs_root>/test_ledger.jsonl`) blocks resume.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .ir import Program
from .task import Example, TaskConfig, parse_rows

LEAK_CHARS = 40  # §8: any train/val input substring of this length is a leak
MIN_ID_CHARS = 4  # shorter ids ("1", "a7") are too generic to police

_test_gate = threading.local()


class SplitAccessError(PermissionError):
    pass


@contextlib.contextmanager
def allow_test_split() -> Iterator[None]:
    _test_gate.open = True
    try:
        yield
    finally:
        _test_gate.open = False


def load_split(config: TaskConfig, split: str) -> list[Example]:
    if split == "test" and not getattr(_test_gate, "open", False):
        raise SplitAccessError("the test split is only readable inside evaluate_final")
    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split {split!r}")
    path = config.split_path(split)
    return parse_rows(path.read_text(encoding="utf-8"), config, split, str(path))


def fingerprint(path: Path) -> dict[str, Any]:
    """Identify a file without opening it (used for the test split during optimize)."""
    st = os.stat(path)
    return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_report(train: list[Example], val: list[Example]) -> dict[str, Any]:
    """Cross-split id/text overlap; id overlap is fatal, duplicated texts are reported."""
    overlap = sorted({e.id for e in train} & {e.id for e in val})
    if overlap:
        raise ValueError(f"train and val share ids: {overlap[:5]}")
    key = lambda e: json.dumps(e.fields, sort_keys=True)  # noqa: E731
    dup = len({key(e) for e in train} & {key(e) for e in val})
    return {"n_train": len(train), "n_val": len(val), "train_val_duplicate_inputs": dup}


class ProgramGuard:
    """Rejects programs that memorize data (§8). Seed text is exempt (task-author supplied)."""

    def __init__(self, examples: list[Example], seed_text: str, max_chars: int | None):
        self.corpus = "\x00".join(v.lower() for ex in examples for v in ex.fields.values())
        self.ids = sorted({ex.id for ex in examples if len(ex.id) >= MIN_ID_CHARS}, key=len, reverse=True)
        self.seed = seed_text.lower()
        self.max_chars = max_chars or max(4000, 8 * len(seed_text))

    def problems(self, text: str) -> list[str]:
        out: list[str] = []
        if len(text) > self.max_chars:
            out.append(f"program is {len(text)} chars, over the {self.max_chars}-char limit")
        low = text.lower()
        leak = self._leak(low)
        if leak:
            out.append(f"program copies example text ({leak[:60]!r}); describe patterns, never quote data")
        for rid in self.ids:
            if re.search(rf"(?<![\w-]){re.escape(rid.lower())}(?![\w-])", low):
                out.append(f"program mentions example id {rid!r}")
                break
        return out

    def check_program(self, program: Program, train_ids: set[str]) -> list[str]:
        """Text checks on the program minus its demo id list and fitted readouts (numbers, never
        text; §4.2), plus: demo ids must be TRAIN ids (§4.1)."""
        nodes = {nid: n.model_copy(update={"readout": None}) if getattr(n, "readout", None) is not None else n for nid, n in program.nodes.items()}
        program = program.model_copy(update={"nodes": nodes})
        demos = program.demos
        if demos is None or not demos.ids:
            return self.problems(program.to_yaml())
        unknown = [i for i in demos.ids if i not in train_ids]
        stripped = program.model_copy(update={"demos": demos.model_copy(update={"ids": None})})
        out = self.problems(stripped.to_yaml())
        if unknown:
            out.append(f"demo ids {unknown[:5]} are not train example ids")
        return out

    def _leak(self, low: str) -> str | None:
        # Any shared substring of length >= 2h contains an aligned h-window; screen those, then confirm.
        half = LEAK_CHARS // 2
        for start in range(0, max(len(low) - half + 1, 0), half):
            window = low[start : start + half]
            if window not in self.corpus:
                continue
            for s in range(max(0, start - half), min(start + half, len(low) - LEAK_CHARS) + 1):
                chunk = low[s : s + LEAK_CHARS]
                if chunk in self.corpus and chunk not in self.seed:
                    return chunk
        return None


# ---- test-consumption ledger ------------------------------------------------

CONSUMED = "test_consumed.json"


def consumed(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / CONSUMED
    return json.loads(path.read_text()) if path.exists() else None


def record_consumption(run_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    if consumed(run_dir):
        raise SplitAccessError(f"test split already consumed for {run_dir.name}")
    record = {**record, "consumed_at": datetime.now(UTC).isoformat()}
    (run_dir / CONSUMED).write_text(json.dumps(record, indent=2))
    with open(run_dir.parent / "test_ledger.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"run_id": run_dir.name, **{k: v for k, v in record.items() if k != "metrics"}}) + "\n")
    return record
