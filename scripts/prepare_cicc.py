# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas==2.2.3", "numpy==2.1.3", "langid==1.1.6"]
# ///
"""Freeze the CICC evaluation splits (den Hengst et al., Findings NAACL 2024).

Source: the authors' repository https://github.com/florisdenhengst/cicc at
commit 18019a1 (MIT). For every public English dataset it ships the exact
train/calibration/test split the paper evaluated (the output of the
`pretrain_*.ipynb` stratified 60/20/20 `train_test_split(seed=1)`), as
`data/<dataset>/bert/{calibration,test}_predictions.csv` with columns
`text, hash, label, label_0..label_{K-1}` (utterance, row id, gold intent
index, fine-tuned BERT softmax) and `labels.csv` (index -> intent name).
The Dutch industry dataset (IND) is not in the release and is out of scope.

The underlying corpora carry mixed or missing licences (ACID: none; ATIS:
LDC-derived), so no utterance text is redistributed: the frozen files hold row
coordinates, gold intent and a per-row SHA-256, and the task adapter rebuilds
the text from the pinned clone, verifying each hash.

    uv run scripts/prepare_cicc.py

clones the repo into data/raw/cicc/repo if absent, checks the commit and
writes data/frozen/cicc/:
  <ds>.jsonl        calibration + test rows for ds in acid, atis, b77, c150,
                    hwu64, mtod (C150 = the in-scope 'C150-IS' test: rows with
                    the `oos` intent removed, as in cicc_clinc150_bert.ipynb)
  c150_oos.jsonl    C150-OOS: conformal calibration split + the OOS-calibration
                    and OOS-test sets sampled exactly as cicc_clinc150_bert.ipynb
                    cell 3 (np.random.seed(42), pandas .sample)
  b77_oos.jsonl     B77-OOS: calibration split + ATIS open-domain outliers split
                    as cicc_banking77_bert.ipynb cell 3
  language_screen.json  langid + script screen of every frozen utterance
and prompts/cicc_<ds>.json (intent inventories from labels.csv).

Item id = "<cal|test|ood|atis>-<hash column value>".
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# langid scores one short utterance at a time; multi-threaded BLAS only adds contention.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "cicc"
CLONE = RAW / "repo"
URL = "https://github.com/florisdenhengst/cicc"
COMMIT = "18019a17a5646cdee4949101c6234cead9aaac96"
OUT = REPO / "data" / "frozen" / "cicc"
PROMPTS = REPO / "prompts"

# task dataset -> (release directory, paper Table 1 name)
DATASETS = {
    "acid": "acid",
    "atis": "atis",
    "b77": "banking77",
    "c150": "clinc150",
    "hwu64": "hwu64",
    "mtod": "mtod",
}
SPLIT_FILES = {"cal": "calibration_predictions.csv", "test": "test_predictions.csv"}
# Split sizes (release; totals equal paper Table 1 sample counts).
EXPECTED = {
    "acid": (4435, 4434),
    "atis": (1175, 1174),
    "b77": (2617, 2617),
    "c150": (3605, 3605),
    "hwu64": (5143, 5143),
    "mtod": (8665, 8665),
}
C150_OOS_INDEX = 42  # labels.csv: 42 -> "oos"
B77_OOS_INDEX = 77  # atis_predictions.csv label column (one past the 77 intents)

# MTOD (Schuster et al. 2019) intents are `domain/intent` in the official
# release (download.pytorch.org/data/multilingual_task_oriented_dialog_slotfilling.zip,
# en/*.tsv column 1); the authors' convert_mtod.py keeps only the intent part.
MTOD_DOMAIN = {
    "cancel_alarm": "alarm", "modify_alarm": "alarm", "set_alarm": "alarm", "show_alarms": "alarm",
    "snooze_alarm": "alarm", "time_left_on_alarm": "alarm", "cancel_reminder": "reminder",
    "set_reminder": "reminder", "show_reminders": "reminder", "checkSunrise": "weather",
    "checkSunset": "weather", "find": "weather",
}

# Assistant context per dataset, from the dataset papers (cited in each prompt's _source).
CONTEXT = {
    "acid": (
        "a customer of an insurance company (American Family Insurance) writing to the company's chatbot",
        "Acharya & Fung (2020), ACID: customer questions to an insurance company's conversational agent, intents defined by subject matter experts",
    ),
    "atis": (
        "a user of an air travel information system asking about flights, fares and airline travel",
        "Hemphill et al. (1990), ATIS: spoken queries to an air travel information system",
    ),
    "b77": (
        "a customer of an online bank writing to the bank's customer-service assistant",
        "Casanueva et al. (2020), BANKING77: online banking customer-service queries, 77 fine-grained intents",
    ),
    "c150": (
        "a user of a general-purpose virtual assistant (task-oriented dialogue system) spanning domains such as banking, work, travel, cooking, home, auto and small talk",
        "Larson et al. (2019), CLINC150: queries to a task-oriented dialogue system, 150 intents over 10 domains",
    ),
    "hwu64": (
        "a user of a home assistant (voice assistant) across everyday domains such as alarms, music, calendar, IoT devices, email, news and general conversation",
        "Liu et al. (2021), HWU64: user commands to a home assistant robot; intent labels are scenario_intent",
    ),
    "mtod": (
        "a user of a task-oriented assistant for alarms, reminders and weather",
        "Schuster et al. (2019), multilingual task-oriented dialog (English part): alarm, reminder and weather domains",
    ),
}
PAPER_SRC = (
    "den Hengst et al., Conformal Intent Classification and Clarification for Fast and Accurate "
    "Intent Recognition, Findings NAACL 2024, https://aclanthology.org/2024.findings-naacl.156/. "
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def row_sha256(text: str, label: str) -> str:
    return sha256_bytes(json.dumps([text, label], ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def ensure_clone() -> str:
    if not CLONE.exists():
        subprocess.run(["git", "clone", "-q", URL, str(CLONE)], check=True)
    subprocess.run(["git", "-C", str(CLONE), "checkout", "-q", COMMIT], check=True)
    head = subprocess.run(["git", "-C", str(CLONE), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    return head.stdout.strip()


def labels(ds: str) -> list[str]:
    with (CLONE / "data" / DATASETS[ds] / "bert" / "labels.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))[1:]
    assert [int(r[0]) for r in rows] == list(range(len(rows)))
    return [r[1] for r in rows]


def read_split(ds: str, file: str) -> list[dict[str, str]]:
    """text/hash/label columns only, read verbatim (no NaN coercion)."""
    path = CLONE / "data" / DATASETS[ds] / "bert" / file
    with path.open(newline="", encoding="utf-8") as fh:
        rows = [{"text": r["text"], "hash": r["hash"], "label": r["label"]} for r in csv.DictReader(fh)]
    assert [int(r["hash"]) for r in rows] == list(range(len(rows))), path
    return rows


def label_name(ds: str, idx: int, names: list[str]) -> str:
    if ds == "c150" and idx == C150_OOS_INDEX:
        return "oos"
    if ds == "b77" and idx == B77_OOS_INDEX:
        return "oos"
    return names[idx]


def frozen_row(prefix: str, file: str, r: dict[str, str], gold: str, roles: list[str]) -> dict:
    return {
        "id": f"{prefix}-{r['hash']}",
        "file": file,
        "row": int(r["hash"]),
        "gold": gold,
        "roles": roles,
        "sha256": row_sha256(r["text"], gold),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def inscope_rows(ds: str) -> list[dict]:
    names = labels(ds)
    out = []
    for prefix, file in SPLIT_FILES.items():
        role = "calibration" if prefix == "cal" else "test"
        for r in read_split(ds, file):
            gold = label_name(ds, int(r["label"]), names)
            if gold == "oos":  # C150-IS: OOS rows are not part of the in-scope test set
                continue
            out.append(frozen_row(prefix, file, r, gold, [role]))
    n_cal = sum("calibration" in r["roles"] for r in out)
    n_test = sum("test" in r["roles"] for r in out)
    assert (n_cal, n_test) == EXPECTED[ds], (ds, n_cal, n_test)
    return out


def _utterance_frame(rows: list[dict[str, str]], prefix: str) -> pd.DataFrame:
    """The notebooks' frame after `reset_index().rename(index->utterance)`."""
    # The CSV row index equals the `hash` column (asserted in read_split); filtered
    # frames (C150 test without OOS) keep their original index, hence original ids.
    idx = [int(r["hash"]) for r in rows]
    df = pd.DataFrame({"label": [int(r["label"]) for r in rows]}, index=idx)
    df = df.reset_index().rename(columns={"index": "utterance"})
    df["utterance"] = df["utterance"].apply(lambda h: f"{prefix}_{h}").astype(str)
    return df


def c150_oos_rows() -> list[dict]:
    """cicc_clinc150_bert.ipynb cells 1+3, replayed on row ids only."""
    names = labels("c150")
    test_all = read_split("c150", SPLIT_FILES["test"])
    cal = read_split("c150", SPLIT_FILES["cal"])
    ood = read_split("c150", "ood_predictions.csv")
    test_is = [r for r in test_all if int(r["label"]) != C150_OOS_INDEX]
    df_test = _utterance_frame(test_is, "test")
    df_cal = _utterance_frame(cal, "cal")
    df_ood = _utterance_frame(ood, "ood")

    np.random.seed(42)
    df_ood_shuffled = df_ood.sample(frac=1)
    df_ood = pd.concat([df_test.sample(len(df_ood)), df_ood])
    split = int(len(df_ood_shuffled) / 2)
    df_ood_calib = pd.concat([df_cal.sample(int(len(df_ood) / 2)), df_ood_shuffled[split:]])
    df_ood_test = pd.concat([df_test.sample(int(len(df_ood) / 2)), df_ood_shuffled[:split]])
    # Notebook output (cell 27) prints X_ood_test: head test_1755, test_613, test_138,
    # test_3674, test_3835; tail ood_530, ood_90, ood_1161, ood_181, ood_1026; 1800 rows.
    got = list(df_ood_test.utterance)
    assert got[:5] == ["test_1755", "test_613", "test_138", "test_3674", "test_3835"], got[:5]
    assert got[-5:] == ["ood_530", "ood_90", "ood_1161", "ood_181", "ood_1026"], got[-5:]
    assert len(got) == 1800 and len(df_ood_calib) == 1800

    roles: dict[str, list[str]] = {}
    for u in df_cal.utterance:
        roles.setdefault(u, []).append("calibration")
    for u in df_ood_calib.utterance:
        roles.setdefault(u, []).append("oos_calibration")
    for u in df_ood_test.utterance:
        roles.setdefault(u, []).append("oos_test")
    src = {"test": (test_is, SPLIT_FILES["test"]), "cal": (cal, SPLIT_FILES["cal"]), "ood": (ood, "ood_predictions.csv")}
    out = []
    for prefix, (rows, file) in src.items():
        for r in rows:
            u = f"{prefix}_{r['hash']}"
            if u in roles:
                out.append(frozen_row(prefix, file, r, label_name("c150", int(r["label"]), names), roles[u]))
    return out


def b77_oos_rows() -> list[dict]:
    """cicc_banking77_bert.ipynb cells 1+3, replayed on row ids only."""
    names = labels("b77")
    test = read_split("b77", SPLIT_FILES["test"])
    cal = read_split("b77", SPLIT_FILES["cal"])
    atis = read_split("b77", "atis_predictions.csv")
    df_test = _utterance_frame(test, "test")
    df_atis = _utterance_frame(atis, "ood")

    np.random.seed(42)
    df_ood_shuffled = df_atis.sample(frac=1)
    split_test = int(len(df_test) / 2)
    split_ood = int(len(df_test) / 2)
    df_test_shuffled = df_test.sample(frac=1)
    df_ood_calib = pd.concat([df_test_shuffled[split_test:], df_ood_shuffled[split_ood:]])
    df_ood_test = pd.concat([df_test_shuffled[:split_test], df_ood_shuffled[:split_ood]])
    assert (len(df_ood_calib), len(df_ood_test)) == (1309 + 4563, 1308 + 1308)

    roles: dict[str, list[str]] = {f"cal_{i}": ["calibration"] for i in range(len(cal))}
    for u in df_ood_calib.utterance:
        roles.setdefault(u, []).append("oos_calibration")
    for u in df_ood_test.utterance:
        roles.setdefault(u, []).append("oos_test")
    src = {"test": (test, SPLIT_FILES["test"], "test"), "cal": (cal, SPLIT_FILES["cal"], "cal"), "ood": (atis, "atis_predictions.csv", "atis")}
    out = []
    for prefix, (rows, file, id_prefix) in src.items():
        for r in rows:
            u = f"{prefix}_{r['hash']}"
            if u in roles:
                out.append(frozen_row(id_prefix, file, r, label_name("b77", int(r["label"]), names), roles[u]))
    return out


def describe(ds: str, label: str) -> str:
    """Mechanical label rendering: the authors' own `label.replace('_', ' ')`
    (cicc_*.ipynb, Appendix B tables); MTOD gets its release domain prefix."""
    text = label.replace("_", " ")
    if ds == "mtod":
        text = f"{MTOD_DOMAIN[label]}/{text}"
    return text


def write_prompt(ds: str) -> None:
    inventory = [lab for lab in labels(ds) if lab != "oos"]
    who, cite = CONTEXT[ds]
    prompt = {
        "_source": PAPER_SRC
        + "The paper has no zero-shot classifier prompt and the datasets ship no intent definitions; "
        + "criteria are the released intent names (authors' labels.csv, repo commit 18019a1) rendered "
        + "with the authors' own `replace('_', ' ')` (Appendix B tables)"
        + (", prefixed with the intent's domain from the official MTOD release (domain/intent)" if ds == "mtod" else "")
        + f". Assistant context from the dataset paper: {cite}.",
        "_primitive": "choice",
        "_labels": len(inventory),
        "question": {
            "type": "choice",
            "instructions": (
                f"The state holds one utterance (\"utterance\") from {who}. "
                "Identify the user's intent: choose the single intent from the options that the utterance expresses."
            ),
            "criteria": {lab: describe(ds, lab) for lab in inventory},
        },
    }
    (PROMPTS / f"cicc_{ds}.json").write_text(json.dumps(prompt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def language_screen() -> dict:
    """langid + non-ASCII screen over every frozen utterance (flags listed for review)."""
    from langid.langid import LanguageIdentifier, model

    ident = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    files = [(ds, f) for ds in DATASETS for f in SPLIT_FILES.values()] + [("c150", "ood_predictions.csv"), ("b77", "atis_predictions.csv")]
    total, flags = 0, []
    for ds, file in files:
        for r in read_split(ds, file):
            total += 1
            text = r["text"]
            non_ascii = "".join(sorted({c for c in text if ord(c) > 127}))
            lang, p = ident.classify(text)
            if non_ascii or (lang != "en" and p > 0.99 and len(text.split()) >= 4):
                flags.append({"dataset": ds, "file": file, "row": int(r["hash"]), "langid": lang, "p": round(p, 4), "non_ascii": non_ascii, "text": text})
    return {"utterances_checked": total, "method": "langid 1.1.6 (normalised) flag if non-en with p>0.99 and >=4 tokens; any non-ASCII character", "flags": flags}


def main() -> None:
    head = ensure_clone()
    print("clone HEAD", head)
    for ds in DATASETS:
        rows = inscope_rows(ds)
        write_jsonl(OUT / f"{ds}.jsonl", rows)
        write_prompt(ds)
        print(ds, "calibration/test", EXPECTED[ds])
    for name, fn in (("c150_oos", c150_oos_rows), ("b77_oos", b77_oos_rows)):
        rows = fn()
        write_jsonl(OUT / f"{name}.jsonl", rows)
        counts = {role: sum(role in r["roles"] for r in rows) for role in ("calibration", "oos_calibration", "oos_test")}
        oos = {role: sum(role in r["roles"] and r["gold"] == "oos" for r in rows) for role in ("oos_calibration", "oos_test")}
        print(name, "items", len(rows), counts, "oos", oos)
    screen = language_screen()
    (OUT / "language_screen.json").write_text(json.dumps(screen, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("language screen", screen["utterances_checked"], "flags", len(screen["flags"]))
    files = sorted({(ds, f) for ds in DATASETS for f in [*SPLIT_FILES.values(), "labels.csv"]} | {("c150", "ood_predictions.csv"), ("b77", "atis_predictions.csv")})
    for ds, f in files:
        print("sha256", f"data/{DATASETS[ds]}/bert/{f}", sha256_bytes((CLONE / "data" / DATASETS[ds] / "bert" / f).read_bytes()))


if __name__ == "__main__":
    sys.exit(main())
