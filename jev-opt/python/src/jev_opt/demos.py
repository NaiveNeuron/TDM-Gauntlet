"""Demonstrations in state (architecture §4.1): labelled TRAIN examples chosen per input.

`DemoPool` holds the train split only (hygiene: every demo id is asserted to be a train id).
`knn` ranks the pool by cosine similarity under a local sentence-embedding model
(`SentenceEmbedder`, pinned model revision, CPU, disk-cached; not an LLM call). The query
example itself is never its own demo (leave-one-out on train, so train-fitted parameters are
not fitted on self-labelled inputs).
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .ir import Demos
from .task import Example


class DemoError(ValueError):
    pass


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray:  # (n, d), rows L2-normalized
        ...


def example_text(ex: Example) -> str:
    return "\n".join(f"{k}: {v}" for k, v in ex.fields.items())


class SentenceEmbedder:
    """sentence-transformers model at a pinned revision on CPU; vectors cached per text on disk."""

    def __init__(self, cache_dir: Path, model: str, revision: str):
        self.model_name, self.revision = model, revision
        self.name = f"{model}@{revision}"
        self.cache_dir = Path(cache_dir) / hashlib.sha256(self.name.encode()).hexdigest()[:16]
        self._model: Any = None
        self._lock = threading.Lock()

    def _path(self, text: str) -> Path:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.npy"

    def encode(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            out: dict[int, np.ndarray] = {}
            missing: list[int] = []
            for i, text in enumerate(texts):
                path = self._path(text)
                if path.exists():
                    out[i] = np.load(path)
                else:
                    missing.append(i)
            if missing:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer  # lazy: torch only when knn demos run

                    self._model = SentenceTransformer(self.model_name, revision=self.revision, device="cpu")
                vecs = self._model.encode([texts[i] for i in missing], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
                for i, vec in zip(missing, vecs):
                    path = self._path(texts[i])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(f".{threading.get_ident()}.tmp.npy")
                    np.save(tmp, vec.astype(np.float32))
                    tmp.replace(path)
                    out[i] = vec.astype(np.float32)
            return np.stack([out[i] for i in range(len(texts))]) if texts else np.zeros((0, 0), dtype=np.float32)


class DemoPool:
    """TRAIN examples available as demonstrations, with nearest-neighbour selection."""

    def __init__(self, train: list[Example], embedder: Embedder | None = None):
        if any(ex.split != "train" for ex in train):
            raise DemoError("the demo pool may only contain train examples")
        self.pool = list(train)
        self.by_id = {ex.id: ex for ex in self.pool}
        self.embedder = embedder
        self._pool_vecs: np.ndarray | None = None
        self._query: dict[str, np.ndarray] = {}  # query key -> vector
        self._lock = threading.Lock()

    def ids(self) -> set[str]:
        return set(self.by_id)

    def check_ids(self, ids: list[str]) -> list[str]:
        """Problems for static demo ids: every id must be a TRAIN example id."""
        unknown = [i for i in ids if i not in self.by_id]
        return [f"demo ids {unknown[:5]} are not train example ids"] if unknown else []

    @staticmethod
    def _qkey(ex: Example) -> str:
        return f"{ex.split}\x00{ex.id}"

    def prefetch(self, spec: Demos | None, examples: list[Example]) -> None:
        """Embed the pool and every query in one batch (knn only)."""
        if spec is None or spec.mode != "knn":
            return
        if self.embedder is None:
            raise DemoError("knn demos need an embedder")
        with self._lock:
            if self._pool_vecs is None:
                self._pool_vecs = self.embedder.encode([example_text(ex) for ex in self.pool])
            todo = [ex for ex in examples if self._qkey(ex) not in self._query]
            if todo:
                for ex, vec in zip(todo, self.embedder.encode([example_text(ex) for ex in todo])):
                    self._query[self._qkey(ex)] = vec

    def vectors(self, ids: list[str]) -> dict[str, np.ndarray]:
        """Embeddings of pool (TRAIN) examples by id, e.g. for minimal pairs (§5.3); other ids are refused."""
        unknown = [i for i in ids if i not in self.by_id]
        if unknown:
            raise DemoError(f"ids {unknown[:5]} are not train example ids")
        if self.embedder is None:
            raise DemoError("embeddings need an embedder")
        with self._lock:
            if self._pool_vecs is None:
                self._pool_vecs = self.embedder.encode([example_text(ex) for ex in self.pool])
        row = {ex.id: i for i, ex in enumerate(self.pool)}
        return {i: self._pool_vecs[row[i]] for i in ids}

    def select(self, spec: Demos, ex: Example) -> list[Example]:
        if spec.mode == "static":
            problems = self.check_ids(spec.ids)
            if problems:
                raise DemoError(problems[0])
            chosen = [self.by_id[i] for i in spec.ids if i != ex.id]
        else:
            self.prefetch(spec, [ex])
            sims = self._pool_vecs @ self._query[self._qkey(ex)]
            order = [i for i in np.argsort(-sims, kind="stable") if self.pool[i].id != ex.id]
            chosen = [self.pool[i] for i in (_balanced(order, self.pool, spec.k) if spec.balanced else order[: spec.k])]
        assert all(d.id in self.by_id and d.split == "train" for d in chosen), "demo outside the train pool"
        return chosen


def _balanced(order: list[int], pool: list[Example], k: int) -> list[int]:
    """Round-robin over labels (labels ordered by their nearest member), nearest first within each."""
    by_label: dict[str, list[int]] = {}
    for i in order:
        by_label.setdefault(pool[i].label, []).append(i)
    queues = list(by_label.values())  # insertion order = label of the nearest example first
    picked: list[int] = []
    depth = 0
    while len(picked) < k and any(depth < len(q) for q in queues):
        for q in queues:
            if depth < len(q) and len(picked) < k:
                picked.append(q[depth])
        depth += 1
    rank = {i: r for r, i in enumerate(order)}
    return sorted(picked, key=rank.__getitem__)


def demo_records(demos: list[Example], fields: list[str], max_chars: int | None) -> list[dict[str, str]]:
    """State payload: each demo's `fields` (inputs only) plus its gold `label`."""
    def clip(v: str) -> str:
        return v if max_chars is None or len(v) <= max_chars else v[:max_chars] + " …"

    return [{**{f: clip(d.fields[f]) for f in fields if f in d.fields}, "label": d.label} for d in demos]
