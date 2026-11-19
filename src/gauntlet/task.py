"""Task-adapter interface.

A task turns a frozen English test set into Jev requests, maps Jev's typed
answers to predictions, and scores them with the paper's primary metric.
One item may need several requests (e.g. pairwise ranking), so requests are
keyed by (item_id, key).
"""

from __future__ import annotations

import gzip
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "data" / "frozen"
PROMPTS = REPO / "prompts"


@dataclass(frozen=True)
class Item:
    id: str
    gold: Any
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Request:
    item_id: str
    key: str
    state: Any
    questions: dict[str, Any]


class Task(ABC):
    """Subclasses set `name` and `prompt_file`, and implement the hooks.

    Static tasks implement `requests`. Adaptive tasks, whose later requests depend on
    earlier answers, override `next_requests` instead.
    """

    name: str
    prompt_file: str

    @abstractmethod
    def items(self) -> list[Item]: ...

    def requests(self, item: Item) -> Iterator[Request]:
        raise NotImplementedError(f"{type(self).__name__} must implement requests() or next_requests()")

    def next_requests(self, item: Item, answers: dict[str, dict[str, Any]]) -> list[Request]:
        """Requests still needed for `item` given the answers so far; [] when done."""
        return [] if answers else list(self.requests(item))

    @abstractmethod
    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        """`answers` maps request key -> Jev `answers` object for that request."""

    @abstractmethod
    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]: ...

    def prompt(self) -> dict[str, Any]:
        return json.loads((PROMPTS / self.prompt_file).read_text())

    def prompt_sha256(self) -> str:
        return hashlib.sha256((PROMPTS / self.prompt_file).read_bytes()).hexdigest()


def _open(path: Path, mode: str):
    return gzip.open(path, mode + "t", encoding="utf-8") if path.suffix == ".gz" else path.open(mode, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with _open(path, "r") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Writes JSONL; gzip when the path ends in .gz (mtime pinned for reproducible bytes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            for row in rows:
                gz.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
        return
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
