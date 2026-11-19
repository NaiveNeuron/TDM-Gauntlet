"""Fetch the PiNose (ACL 2024) release and freeze the four English test sets.

The release (Google Drive zip linked from https://github.com/RUCKBReasoning/PiNose)
carries no licence, so test rows are NOT redistributed. This script:

  1. downloads the zip to data/raw/pinose/pinose_public.zip (skipped if present)
     and checks its SHA-256;
  2. extracts it to data/raw/pinose/public/;
  3. checks the SHA-256 of each test CSV;
  4. writes data/frozen/pinose/<subset>/manifest.jsonl: one line per test row,
     {"id", "row", "sha256"} where sha256 hashes the canonical row
     (see gauntlet.tasks.pinose.row_sha256).

The task adapter rebuilds items from data/raw and verifies every row against
the frozen manifest. Re-running is deterministic.

    uv run python scripts/prepare_pinose.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gauntlet.tasks.pinose import (  # noqa: E402
    RAW,
    SUBSETS,
    ZIP_NAME,
    ZIP_SHA256,
    ZIP_URL,
    frozen_manifest_path,
    raw_rows,
    row_sha256,
)
from gauntlet.task import write_jsonl  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    zip_path = RAW / ZIP_NAME
    if not zip_path.exists():
        print(f"downloading {ZIP_URL}", file=sys.stderr)
        tmp = zip_path.with_suffix(".part")
        with urllib.request.urlopen(ZIP_URL) as resp, tmp.open("wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        tmp.rename(zip_path)
    got = sha256_file(zip_path)
    if got != ZIP_SHA256:
        sys.exit(f"{zip_path}: sha256 {got} != expected {ZIP_SHA256}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW)

    for subset, spec in SUBSETS.items():
        csv_path = RAW / spec["csv"]
        got = sha256_file(csv_path)
        if got != spec["csv_sha256"]:
            sys.exit(f"{csv_path}: sha256 {got} != expected {spec['csv_sha256']}")
        rows = raw_rows(subset)
        manifest = [
            {"id": f"{spec['id_prefix']}-{i:04d}", "row": i, "sha256": row_sha256(row)}
            for i, row in enumerate(rows)
        ]
        write_jsonl(frozen_manifest_path(subset), manifest)
        print(f"{subset}: {len(manifest)} rows -> {frozen_manifest_path(subset).relative_to(REPO)}", file=sys.stderr)


if __name__ == "__main__":
    main()
