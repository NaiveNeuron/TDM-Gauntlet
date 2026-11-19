"""Test-set resampling units shared by the RQ3 analyses (imported, not run).

The unit is the one the RQ1 statistics use for the same set (scripts/offline_rq1.py `cluster_keys`): items that
share a source unit are drawn together. ConTempo TB-Dense/MATRES: document; INLI: premise; EZ-STANCE: tweet;
LifeTox: query; edit intent: document; DREsS: writing prompt. Every other set: the item.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
REPO = EXP.parents[1]

# slug prefix -> (unit, test file relative to jev-opt/experiments, key field or None for a special rule)
_UNITS = {
    "contempo-tbdense": ("document", "contempo_tbdense/data/test.jsonl", None),
    "contempo-matres": ("document", "contempo_matres/data/test.jsonl", None),
    "inli": ("premise", "inli/data/test.jsonl", "premise"),
    "ezstance-subtaskA": ("tweet", "ezstance_subtaskA/data/test.jsonl", "tweet"),
    "lifetox": ("query", "lifetox/data/test.jsonl", "question"),
    "edit-intent": ("document", None, None),
    "dress-new-content": ("prompt", "dress_new/data/content/test.jsonl", "essay_prompt"),
    "dress-new-organization": ("prompt", "dress_new/data/organization/test.jsonl", "essay_prompt"),
    "dress-new-language": ("prompt", "dress_new/data/language/test.jsonl", "essay_prompt"),
}


def cluster_index(slug: str, ids: list[str]) -> tuple[np.ndarray, str]:
    """(cluster number per test item in `ids` order, unit name)."""
    if slug not in _UNITS:
        return np.arange(len(ids)), "item"
    unit, rel, field = _UNITS[slug]
    if slug.startswith("contempo-"):
        keys = [i.split(":")[0] for i in ids]
    elif slug == "edit-intent":
        rows = [json.loads(x) for x in (REPO / "data" / "frozen" / "edit_intent" / "test.jsonl").read_text().splitlines() if x.strip()]
        doc = {r["id"]: str(r["doc_name"]) for r in rows}
        keys = [doc[i] for i in ids]
    else:
        rows = [json.loads(x) for x in (EXP / rel).read_text().splitlines() if x.strip()]
        val = {r["id"]: str(r[field]) for r in rows}
        keys = [val[i] for i in ids]
    _, inv = np.unique(np.array(keys, dtype=object).astype(str), return_inverse=True)
    return inv, unit


def resamples(inv: np.ndarray, B: int, seed: int = 0) -> list[np.ndarray]:
    """B cluster bootstrap resamples (item indices; whole clusters drawn with replacement). With one item per
    cluster this equals default_rng(seed).integers(0, n, (B, n)) row by row."""
    m = int(inv.max()) + 1
    if m == len(inv) and (np.sort(inv) == np.arange(m)).all():
        order = np.argsort(inv)
        return list(order[np.random.default_rng(seed).integers(0, m, size=(B, m))])
    members = [np.flatnonzero(inv == c) for c in range(m)]
    rng = np.random.default_rng(seed)
    return [np.concatenate([members[c] for c in rng.integers(0, m, m)]) for _ in range(B)]


def jackknife_t(stat, inv: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Leave-one-cluster-out jackknife interval with t(m-1) critical value; `stat(idx)` maps item indices to a number."""
    from scipy.stats import t

    m = int(inv.max()) + 1
    full = stat(np.arange(len(inv)))
    loo = np.array([stat(np.flatnonzero(inv != c)) for c in range(m)])
    se = np.sqrt((m - 1) / m * ((loo - loo.mean()) ** 2).sum())
    q = t.ppf(0.5 + level / 2, m - 1)
    return float(full - q * se), float(full + q * se)
