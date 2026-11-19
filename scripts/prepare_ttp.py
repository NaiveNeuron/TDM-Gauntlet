"""Freeze the Security TTP Mapping test sets (Nguyen et al., Findings EACL 2024).

Paper: https://aclanthology.org/2024.findings-eacl.25/
Data:  https://huggingface.co/datasets/tumeteor/Security-TTP-Mapping (CC BY 4.0) at revision
       1af2f0ef96f32ba5ff8d9b82b8cb5fc4810430bd; byte-identical to
       github.com/tumeteor/mitre-ttp-mapping@a16856a6438ca2b7888c5cadfba6d7c854f04a55.
Label names: MITRE ATT&CK Enterprise v12.0 STIX bundle,
       https://raw.githubusercontent.com/mitre/cti/ATT%26CK-v12.0/enterprise-attack/enterprise-attack.json

Download first (writes data/raw/ttp/):
    uv run --with huggingface_hub python -c "from huggingface_hub import snapshot_download; \
snapshot_download('tumeteor/Security-TTP-Mapping', repo_type='dataset', local_dir='data/raw/ttp/hf', \
revision='1af2f0ef96f32ba5ff8d9b82b8cb5fc4810430bd')"
    curl -sL -o data/raw/ttp/enterprise-attack-12.0.json \
        'https://raw.githubusercontent.com/mitre/cti/ATT%26CK-v12.0/enterprise-attack/enterprise-attack.json'
Then:
    uv run python scripts/prepare_ttp.py

Writes data/frozen/ttp/:
  <split>.jsonl      one row per released TSV row, file order: id, text (verbatim), labels_raw (as
                     released), gold (sorted unique upper-case ATT&CK IDs)
  candidates.jsonl   the 737 TTP textual profiles of ttp-desc-mappings.csv (the paper's label
                     corpus), ATT&CK ID order: id, name (v12 display name, "Parent: Sub" for
                     sub-techniques), description (released profile, verbatim), status, tactics
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "ttp"
HF = RAW / "hf" / "datasets"
STIX = RAW / "enterprise-attack-12.0.json"
OUT = REPO / "data" / "frozen" / "ttp"

RAW_SHA256 = {
    "ttp-desc-mappings.csv": "d800d4b6affe9df87c91aeeb7e0eed149e3a41473af54a89183580c916e1b2b6",
    "tram/tram_test.tsv": "2f12f2cfa909f99849f78d6101dc204bd2274877b04a643d00a452af216700a4",
    "procedures/procedure_test.tsv": "0623e0105463af4956cd63d475cb55fe04abe74530e4c1b041e702042b202723",
    "derived_procedures/derived_procedure_test.tsv": "d1e1ad8482e87910203582dfc4f5a1087dc4ef937adafa47938ecbe3d77bf0bc",
    "expert/expert_train.tsv": "e982b5464c70c257e2a0cf6ae6b4e63e5a227bc8acb1546c4c2d3d65082930fc",
    "expert/expert_dev.tsv": "7815932d419aa98a54a1bafffa0c363fc0de3ef8aa3b754096eec9101e97fedb",
    "expert/expert_test.tsv": "674cbb8149df68f4c0661ea7738705900d0682f5837adb85c8040d856e0e1378",
}
STIX_SHA256 = "a8952b72ec4104c959d0ad642541dab1b84d835f439a77f7d1322d794aeb4aaf"

# frozen split name -> (released file, id prefix, expected rows)
SPLITS = {
    "tram_test": ("tram/tram_test.tsv", "ttp-tram-test", 725),
    "procedure_test": ("procedures/procedure_test.tsv", "ttp-procedure-test", 1767),
    "derived_procedure_test": ("derived_procedures/derived_procedure_test.tsv", "ttp-derived-test", 521),
    "expert_train": ("expert/expert_train.tsv", "ttp-expert-train", 471),
    "expert_dev": ("expert/expert_dev.tsv", "ttp-expert-dev", 69),
    "expert_test": ("expert/expert_test.tsv", "ttp-expert-test", 157),
}
N_CANDIDATES = 737
TTP_ID = re.compile(r"^T\d{4}(\.\d{3})?$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def id_key(ttp: str) -> tuple[int, int]:
    """ATT&CK ID order: technique number, then sub-technique number (techniques first)."""
    tech, _, sub = ttp[1:].partition(".")
    return int(tech), int(sub or -1)


def read_tsv(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh, delimiter="\t"))


def attack_patterns() -> dict[str, dict]:
    bundle = json.loads(STIX.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for obj in bundle["objects"]:
        if obj["type"] != "attack-pattern":
            continue
        ext = [r["external_id"] for r in obj.get("external_references", []) if r.get("source_name") == "mitre-attack"]
        if not ext:
            continue
        status = "revoked" if obj.get("revoked") else "deprecated" if obj.get("x_mitre_deprecated") else "active"
        out[ext[0].upper()] = {
            "name": obj["name"],
            "status": status,
            "tactics": sorted(p["phase_name"] for p in obj.get("kill_chain_phases", [])),
        }
    return out


def build_candidates() -> list[dict]:
    stix = attack_patterns()
    rows = read_tsv(HF / "ttp-desc-mappings.csv")
    if rows[0] != ["ttp", "description"]:
        raise SystemExit(f"unexpected header {rows[0]}")
    cands = []
    for ttp, description in rows[1:]:
        tid = ttp.upper()
        if not TTP_ID.match(tid) or tid not in stix:
            raise SystemExit(f"{ttp}: not an ATT&CK v12.0 Enterprise attack-pattern")
        name = stix[tid]["name"]
        if "." in tid:
            name = f"{stix[tid.split('.')[0]]['name']}: {name}"
        cands.append({"id": tid, "name": name, "description": description, "status": stix[tid]["status"], "tactics": stix[tid]["tactics"]})
    if len(cands) != N_CANDIDATES or len({c["id"] for c in cands}) != N_CANDIDATES:
        raise SystemExit(f"expected {N_CANDIDATES} unique profiles, got {len(cands)}")
    return sorted(cands, key=lambda c: id_key(c["id"]))


def build_split(name: str, candidate_ids: set[str]) -> list[dict]:
    rel, prefix, expected = SPLITS[name]
    rows = read_tsv(HF / rel)
    if rows[0] != ["text1", "labels"]:
        raise SystemExit(f"{rel}: unexpected header {rows[0]}")
    out = []
    for i, row in enumerate(rows[1:]):
        if len(row) != 2:
            raise SystemExit(f"{rel} row {i}: {len(row)} fields")
        text, labels = row
        labels_raw = ast.literal_eval(labels)
        gold = sorted({lab.upper() for lab in labels_raw}, key=id_key)
        if not gold or not text.strip():
            raise SystemExit(f"{rel} row {i}: empty text or labels")
        unknown = [g for g in gold if g not in candidate_ids]
        if unknown:
            raise SystemExit(f"{rel} row {i}: gold labels outside the candidate corpus: {unknown}")
        out.append({"id": f"{prefix}-{i:04d}", "text": text, "labels_raw": labels_raw, "gold": gold})
    if len(out) != expected:
        raise SystemExit(f"{rel}: expected {expected} rows, got {len(out)}")
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    for rel, want in RAW_SHA256.items():
        got = sha256(HF / rel)
        if got != want:
            raise SystemExit(f"{rel}: unexpected SHA-256 {got}")
    if sha256(STIX) != STIX_SHA256:
        raise SystemExit("enterprise-attack-12.0.json: unexpected SHA-256")
    OUT.mkdir(parents=True, exist_ok=True)
    cands = build_candidates()
    write_jsonl(OUT / "candidates.jsonl", cands)
    print(f"candidates.jsonl {len(cands)} sha256 {sha256(OUT / 'candidates.jsonl')}")
    ids = {c["id"] for c in cands}
    for name in SPLITS:
        rows = build_split(name, ids)
        path = OUT / f"{name}.jsonl"
        write_jsonl(path, rows)
        print(f"{path.name} {len(rows)} sha256 {sha256(path)}")


if __name__ == "__main__":
    main()
