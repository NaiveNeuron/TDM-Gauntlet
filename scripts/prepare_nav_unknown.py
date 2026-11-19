"""Freeze the Navigating the Unknown (Sali & Toraman, Findings EMNLP 2025) test sets.

English public datasets only (the benchmark's English-only rule): CLINC150 and BANKING77. DSTC Finance and
the in-house BIT3 sets are out of scope.

Sources (download into data/raw/nav_unknown/ first):
    C=828f8093932c8fe6ca7936c3d2e52903b1c523de   # clinc/oos-eval (CC BY 3.0)
    P=57ec275d8078af65b7731c2a98be812d844a6d6b   # PolyAI-LDN/task-specific-datasets (CC BY 4.0)
    curl -sLo data/raw/nav_unknown/clinc_data_full.json \
        https://raw.githubusercontent.com/clinc/oos-eval/$C/data/data_full.json
    curl -sLo data/raw/nav_unknown/banking77_test.csv \
        https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/$P/banking_data/test.csv
Then:
    uv run python scripts/prepare_nav_unknown.py

Frozen protocol (the paper releases no code, split or seeds; reconstruction):
- Test utterances: the paper limits test utterances to 10 per intent (Section 3.2,
  Table 1: CLINC150 1,500 = 150 x 10; BANKING77 770 = 77 x 10). We take the FIRST 10
  utterances of each intent in release file order. CLINC150's separate `oos_test`
  (1,000 far-OOD queries) is not used: Table 1 counts 150 intents / 1,500 test rows.
- ID/OOD label partition: "We split the labels 50%-50% as ID and OOD" (Section 4.1).
  Results in the paper are "average of 3 runs"; we freeze three partitions,
  `train_test_split(sorted(intents), test_size=0.5, random_state=seed)` for seeds
  0, 1, 2 (train part = ID, test part = OOD). CLINC150 75 ID / 75 OOD; BANKING77
  38 ID / 39 OOD (the Table 2 baseline row satisfies ALL = (380 ID-F1 + 390 OOD-F1)/770,
  which fits 38 ID / 39 OOD intents).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "nav_unknown"
OUT = REPO / "data" / "frozen" / "nav_unknown"
RAW_SHA256 = {
    "clinc_data_full.json": "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0",
    "banking77_test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}
PER_INTENT = 10
SEEDS = (0, 1, 2)
EXPECTED = {"clinc150": (150, 1500), "banking77": (77, 770)}


def _check(name: str) -> Path:
    path = RAW / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != RAW_SHA256[name]:
        raise SystemExit(f"{name}: unexpected SHA-256 {digest}")
    return path


def _load() -> dict[str, list[tuple[int, str, str]]]:
    clinc = json.loads(_check("clinc_data_full.json").read_text())["test"]
    with _check("banking77_test.csv").open(newline="", encoding="utf-8") as fh:
        banking = [(r["text"], r["category"]) for r in csv.DictReader(fh)]
    return {
        "clinc150": [(i, text, intent) for i, (text, intent) in enumerate(clinc)],
        "banking77": [(i, text, intent) for i, (text, intent) in enumerate(banking)],
    }


def main() -> None:
    splits: dict[str, dict[str, dict[str, list[str]]]] = {}
    for dataset, rows in _load().items():
        taken: dict[str, int] = {}
        items = []
        for idx, text, intent in rows:  # release order; first PER_INTENT per intent
            if taken.get(intent, 0) >= PER_INTENT:
                continue
            taken[intent] = taken.get(intent, 0) + 1
            items.append({"id": f"{dataset}-test-{idx:04d}", "text": text, "intent": intent})
        n_intents, n_items = EXPECTED[dataset]
        if len(taken) != n_intents or len(items) != n_items or set(taken.values()) != {PER_INTENT}:
            raise SystemExit(f"{dataset}: got {len(taken)} intents / {len(items)} items")
        intents = sorted(taken)
        splits[dataset] = {}
        for seed in SEEDS:
            known, ood = train_test_split(intents, test_size=0.5, random_state=seed)
            splits[dataset][str(seed)] = {"id": sorted(known), "ood": sorted(ood)}
        path = OUT / f"{dataset}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for row in items:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{dataset}: {len(items)} items, {len(intents)} intents, sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")
        for seed, part in splits[dataset].items():
            print(f"  seed {seed}: {len(part['id'])} ID / {len(part['ood'])} OOD")
    path = OUT / "splits.json"
    path.write_text(json.dumps(splits, indent=1) + "\n", encoding="utf-8")
    print(f"splits sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
