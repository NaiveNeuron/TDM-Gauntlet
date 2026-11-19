"""Fetch the Scalar Constructs (EMNLP 2025) release files and freeze the held-out evaluation.

Source: github.com/haukelicht/scalar_measurement at the pinned RELEASE_COMMIT. The
release has no licence, so texts are NOT redistributed. This script:

  1. downloads, per construct, bt_scores.jsonl and graph_splits/{0..4}.{test,train,val}.jsonl
     into data/raw/scalar_constructs/release/<same path> (skipped if present);
  2. checks each file's SHA-256 against data/frozen/scalar_constructs/raw_sha256.json
     (written on the first run);
  3. writes data/frozen/scalar_constructs/<construct>/:
       items.jsonl  {"id", "sha256" (of the text), "bt_score"} for every text in any fold's
                    test pairs, in bt_scores.jsonl order;
       pairs.jsonl  {"fold", "pair_id", "id1", "id2", "label"} test pairs, file order per fold;
       folds.json   per fold: test_item_ids (items in test pairs minus items in train
                    pairs, the release notebook's rule) and heldout_item_ids (additionally
                    minus val-pair items: the 100 held-out vertices V_test).

Re-running is deterministic.

    uv run python scripts/prepare_scalar_constructs.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gauntlet.task import read_jsonl, write_jsonl  # noqa: E402
from gauntlet.tasks.scalar_constructs import (  # noqa: E402
    CONSTRUCTS,
    FROZEN_DIR,
    N_FOLDS,
    RELEASE_COMMIT,
    RELEASE_REPO,
    frozen_dir,
    raw_files,
    raw_path,
    text_sha256,
)

URL = f"https://raw.githubusercontent.com/{RELEASE_REPO}/{RELEASE_COMMIT}/{{path}}"
RAW_SHA256 = FROZEN_DIR / "raw_sha256.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(rel: str) -> Path:
    path = raw_path(rel)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {rel}", file=sys.stderr)
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(URL.format(path=rel)) as resp, tmp.open("wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        tmp.rename(path)
    return path


def main() -> None:
    expected = json.loads(RAW_SHA256.read_text()) if RAW_SHA256.exists() else None
    observed: dict[str, str] = {}
    for construct in CONSTRUCTS.values():
        for rel in raw_files(construct):
            observed[rel] = sha256_file(fetch(rel))
    if expected is not None and expected != observed:
        bad = sorted(k for k in observed.keys() | expected.keys() if observed.get(k) != expected.get(k))
        raise SystemExit(f"raw file SHA-256 mismatch: {bad}")
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    RAW_SHA256.write_text(json.dumps(observed, indent=2) + "\n")

    for construct in CONSTRUCTS.values():
        base = f"data/processed_data/{construct.dataset}"
        bt_rows = read_jsonl(raw_path(f"{base}/bt_scores.jsonl"))
        text_by_id = {str(r["id"]): r["text"] for r in bt_rows}
        pairs, folds, used = [], {}, set()
        for k in range(N_FOLDS):
            split = {s: read_jsonl(raw_path(f"{base}/graph_splits/{k}.{s}.jsonl")) for s in ("test", "train", "val")}
            ids = {s: {str(r[f"id{j}"]) for r in rows for j in (1, 2)} for s, rows in split.items()}
            for r in split["test"]:
                if r["label"] not in (1, 2) or r["id1"] == r["id2"]:
                    raise SystemExit(f"{construct.key} fold {k}: unexpected pair {r['pair_id']}")
                # pair texts must equal the item texts in bt_scores.jsonl (items are rebuilt from it)
                if text_by_id.get(str(r["id1"])) != r["text1"] or text_by_id.get(str(r["id2"])) != r["text2"]:
                    raise SystemExit(f"{construct.key} fold {k}: pair {r['pair_id']} text differs from bt_scores.jsonl")
                pairs.append({"fold": k, "pair_id": r["pair_id"], "id1": str(r["id1"]), "id2": str(r["id2"]), "label": r["label"]})
            used |= ids["test"]
            folds[str(k)] = {
                "n_pairs": len(split["test"]),
                "test_item_ids": sorted(ids["test"] - ids["train"], key=int),
                "heldout_item_ids": sorted(ids["test"] - ids["train"] - ids["val"], key=int),
            }
        items = [
            {"id": str(r["id"]), "sha256": text_sha256(r["text"]), "bt_score": r["label"]}
            for r in bt_rows
            if str(r["id"]) in used
        ]
        out = frozen_dir(construct)
        write_jsonl(out / "items.jsonl", items)
        write_jsonl(out / "pairs.jsonl", pairs)
        (out / "folds.json").write_text(json.dumps(folds, indent=1) + "\n")
        f0 = folds["0"]
        print(
            f"{construct.key}: items {len(items)}, fold-0 pairs {f0['n_pairs']}, "
            f"test items {len(f0['test_item_ids'])}, V_test {len(f0['heldout_item_ids'])}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
