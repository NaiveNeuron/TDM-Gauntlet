"""Freeze the LifeTox English test set from the official HF release.

Source: https://huggingface.co/datasets/mbkim/LifeTox (MIT), file LifeTox_test.jsonl
at revision 412ac6285d2047a99fe5fd17ac848a52d5da3bea.

Download first (writes data/raw/lifetox/):
    uv run --with huggingface_hub python -c "from huggingface_hub import snapshot_download; \
snapshot_download('mbkim/LifeTox', repo_type='dataset', local_dir='data/raw/lifetox', \
revision='412ac6285d2047a99fe5fd17ac848a52d5da3bea')"
Then:
    uv run python scripts/prepare_lifetox.py

Rows are kept verbatim and in file order (including the release's literal " nan"
suffix on queries whose post body was empty). Item id = zero-padded line index.
Gold label: is_safe=True -> "safe", False -> "unsafe".
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "lifetox" / "LifeTox_test.jsonl"
OUT = REPO / "data" / "frozen" / "lifetox" / "test.jsonl"
RAW_SHA256 = "d45a1495315ca34cbeb7d9dc3d3fce53e5fe28c15e5faf7b2735587b3d769d7e"
EXPECTED_N = 3000


def main() -> None:
    digest = hashlib.sha256(RAW.read_bytes()).hexdigest()
    if digest != RAW_SHA256:
        raise SystemExit(f"unexpected raw SHA-256 {digest}")
    rows = []
    with RAW.open() as fh:
        for i, line in enumerate(fh):
            src = json.loads(line)
            if not isinstance(src["is_safe"], bool):
                raise SystemExit(f"row {i}: non-boolean is_safe")
            rows.append(
                {
                    "id": f"lifetox-test-{i:04d}",
                    "query": src["query"],
                    "response": src["response"],
                    "gold": "safe" if src["is_safe"] else "unsafe",
                    "upvotes": src["score"],
                }
            )
    if len(rows) != EXPECTED_N:
        raise SystemExit(f"expected {EXPECTED_N} rows, got {len(rows)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {OUT.relative_to(REPO)}")
    print("sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
