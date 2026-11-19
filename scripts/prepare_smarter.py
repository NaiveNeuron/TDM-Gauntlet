# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas==2.2.3", "scikit-learn==1.5.2", "numpy==2.1.3", "langid==1.1.6"]
# ///
"""Freeze the SMARTER (ACL 2026) English test sets: HateXplain, Latent Hate, Implicit Hate.

The paper's repository (github.com/hnghiem-nlp/hate_dpo_public @ 3489b7e) ships
code only; its `Data/testing/{domain}_test.csv` files are not released. The test
sets are therefore rebuilt from the original public releases with the authors'
own sampling code, copied verbatim below from `Code/utils/data_util.py`:

1. `split_train_test(df, split_size, seed=123)` (the function's default seed) with
   the Table 4 ratios: 60:20:20 for HateXplain and Latent Hate, 50:20:30 for
   Implicit Hate; the third part is the test split.
2. `k_shot_sample(test, group_var='label', key_var='uid', k=K_test)` with
   K_test = 400 / 400 / 150 (Table 4) and random_state = 42, the default `--seed`
   of `Code/inference/model_inference.py`, which the evaluation job calls without
   a seed. Latent Hate's test split holds fewer than 400 Explicit Hate posts (the
   whole release has 1,089), so that class takes all of them (the authors' code
   would raise there; see the manifest).
3. Paper Appendix B preprocessing (user handles -> '<user>', URLs removed) is
   applied to every post before splitting.
4. Rows on the frozen, adjudicated non-English list
   (data/frozen/smarter/non_english.json) are removed from the sampled test sets.

Sources (fetched into data/raw/smarter/ when absent):
- HateXplain: git clone https://github.com/hate-alert/HateXplain @ 01d7422 (MIT),
  Data/dataset.json. Majority label of the three annotators; the 919 posts with no
  majority (HateXplain's "undecided") are dropped before splitting, as in the
  HateXplain release's own loader.
- Latent Hatred: the corpus zip linked from the README of
  https://github.com/SALT-NLP/implicit-hate (Dropbox p1ctnsg3xlnupwr), SHA-256
  checked. Stage-1 posts/IDs for Latent Hate; Stage-2 posts/IDs for Implicit
  Hate, with the `other` class (outside the paper's label space) dropped.

Run:
    uv run scripts/prepare_smarter.py            # freeze
    uv run scripts/prepare_smarter.py --screen   # print language-screen flags
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from gauntlet.tasks.smarter import (  # noqa: E402
    FROZEN_DIR,
    HATEXPLAIN_JSON,
    HATEXPLAIN_JSON_SHA256,
    LABELS,
    LATENT_ZIP,
    LATENT_ZIP_SHA256,
    RAW_DIR,
    RELEASE_LABELS,
    latent_rows,
    preprocess,
    row_sha256,
    sha256_file,
)

HATEXPLAIN_GIT = "https://github.com/hate-alert/HateXplain"
HATEXPLAIN_COMMIT = "01d742279dac941981f53806154481c0e15ee686"
LATENT_URL = "https://www.dropbox.com/s/p1ctnsg3xlnupwr/implicit-hate-corpus.zip?dl=1"
SPLIT_SEED = 123  # split_train_test default
SAMPLE_SEED = 42  # model_inference.py --seed default
CONFIG = {  # Table 4: split ratio (train, val, test), K_test per class
    "hatexplain": ([0.6, 0.2, 0.2], 400),
    "latent_hate": ([0.6, 0.2, 0.2], 400),
    "implicit_hate": ([0.5, 0.2, 0.3], 150),
}


# --- verbatim from hate_dpo_public Code/utils/data_util.py (logging removed) ---
def split_train_test(df, split_size, shuffle=True, seed=123):
    assert sum(split_size) == 1, "The split sizes must sum to 1"
    _, val_size, test_size = split_size
    train, test = train_test_split(df, test_size=val_size + test_size, random_state=seed, shuffle=shuffle)
    val, test = train_test_split(test, test_size=test_size / (val_size + test_size), random_state=seed, shuffle=shuffle)
    return train, val, test


def deduplicate_by_variable(df, var):
    return df.drop_duplicates(subset=var, keep="first").reset_index(drop=True)


def k_shot_sample(df, group_var, key_var, k, replace_to_meet_k=False, random_state=123):
    unique_df = deduplicate_by_variable(df, key_var)
    # Only change: a class with fewer than k rows contributes all of them
    # (x.sample(n=len(x)) shuffles them), where the original raises ValueError.
    sampled_df = unique_df.groupby(group_var, group_keys=False).apply(
        lambda x: x.sample(n=min(k, len(x)), replace=False, random_state=random_state)
    )
    return sampled_df.reset_index(drop=True)
# --- end verbatim ---


def fetch_raw() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    clone = RAW_DIR / "hatexplain"
    if not clone.exists():
        subprocess.run(["git", "clone", "-q", HATEXPLAIN_GIT, str(clone)], check=True)
        subprocess.run(["git", "-C", str(clone), "checkout", "-q", HATEXPLAIN_COMMIT], check=True)
    if sha256_file(HATEXPLAIN_JSON) != HATEXPLAIN_JSON_SHA256:
        raise SystemExit(f"{HATEXPLAIN_JSON}: unexpected SHA-256")
    if not LATENT_ZIP.exists():
        urllib.request.urlretrieve(LATENT_URL, LATENT_ZIP)
    if sha256_file(LATENT_ZIP) != LATENT_ZIP_SHA256:
        raise SystemExit(f"{LATENT_ZIP}: unexpected SHA-256")


def hatexplain_frame() -> tuple[pd.DataFrame, int]:
    data = json.loads(HATEXPLAIN_JSON.read_text(encoding="utf-8"))
    rows, undecided = [], 0
    for post_id, post in data.items():
        votes = Counter(a["label"] for a in post["annotators"]).most_common()
        if votes[0][1] < 2:  # three different labels: no majority
            undecided += 1
            continue
        text = preprocess(" ".join(post["post_tokens"]))
        rows.append({"uid": post_id, "text": text, "label": RELEASE_LABELS["hatexplain"][votes[0][0]]})
    return pd.DataFrame(rows), undecided


def latent_frame(subset: str) -> tuple[pd.DataFrame, int]:
    mapping = RELEASE_LABELS[subset]
    rows, dropped = [], 0
    for r in latent_rows(subset):
        if r["label"] not in mapping:
            if r["label"] != "other":
                raise SystemExit(f"{subset}: unknown label {r['label']!r}")
            dropped += 1
            continue
        rows.append({"uid": r["uid"], "row": r["row"], "text": preprocess(r["post"]), "label": mapping[r["label"]]})
    return pd.DataFrame(rows), dropped


def build(subset: str) -> tuple[pd.DataFrame, dict]:
    df, dropped = hatexplain_frame() if subset == "hatexplain" else latent_frame(subset)
    ratios, k = CONFIG[subset]
    train, val, test = split_train_test(df, ratios, seed=SPLIT_SEED)
    sample = k_shot_sample(test, group_var="label", key_var="uid", k=k, random_state=SAMPLE_SEED)
    stats = {
        "pool": len(df),
        "pool_label_counts": dict(Counter(df["label"])),
        "dropped_before_split": dropped,
        "split_sizes": [len(train), len(val), len(test)],
        "test_split_label_counts": dict(Counter(test["label"])),
        "k_test": k,
        "sampled": len(sample),
        "sampled_label_counts": dict(Counter(sample["label"])),
    }
    return sample, stats


def screen(samples: dict[str, pd.DataFrame]) -> None:
    """Language screen: langid (all languages) + non-ASCII-letter ratio; prints flags."""
    import langid

    for subset, df in samples.items():
        for r in df.itertuples():
            lang, _ = langid.classify(r.text)
            letters = [c for c in r.text if c.isalpha()]
            non_ascii = sum(not c.isascii() for c in letters) / max(len(letters), 1)
            if lang != "en" or non_ascii > 0.05:
                print(json.dumps({"subset": subset, "uid": r.uid, "langid": lang, "non_ascii": round(non_ascii, 3), "text": r.text}, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", action="store_true")
    args = ap.parse_args()
    fetch_raw()
    samples, stats = {}, {}
    for subset in LABELS:
        samples[subset], stats[subset] = build(subset)
    if args.screen:
        screen(samples)
        return
    non_english = json.loads((FROZEN_DIR / "non_english.json").read_text())["excluded"]
    drop = {(e["subset"], e["uid"]) for e in non_english}
    for subset, df in samples.items():
        rows = []
        for r in df.itertuples():
            if (subset, r.uid) in drop:
                continue
            row = {"id": f"{subset}-{r.uid}", "uid": r.uid}
            if subset == "hatexplain":
                row["text"] = r.text
            else:
                row["row"] = r.row
            row.update({"gold": r.label, "sha256": row_sha256(r.text, r.label)})
            rows.append(row)
        found = {(subset, u) for u in df["uid"]} & drop
        stats[subset]["excluded_non_english"] = len(found)
        stats[subset]["frozen_n"] = len(rows)
        stats[subset]["frozen_label_counts"] = dict(Counter(r["gold"] for r in rows))
        out = FROZEN_DIR / f"{subset}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats[subset]["frozen_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    missing = drop - {(s, u) for s, df in samples.items() for u in df["uid"]}
    if missing:
        raise SystemExit(f"non_english.json lists rows not in the sampled test sets: {sorted(missing)}")
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
