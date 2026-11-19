"""Rebuild the ConTempo (Niu et al., Findings ACL 2024) test sets from data/raw/contempo.

Two tasks, label spaces kept separate:
  contempo/matres   MATRES Platinum test (Ning et al. 2018 GitHub release), 837 pairs, 20 docs
  contempo/tbdense  TimeBank-Dense test (Cassidy et al. 2014), event-event pairs, 9 docs, 1,427 pairs

Text: TempEval-3 TimeML. MATRES Platinum text = TempEval-3 `te3-platinum` (copy the
MATRES README points to: qiangning/StructTempRel-EMNLP17). TB-Dense text = TempEval-3
`TBAQ-cleaned/TimeBank` (official tbaq-2013-03.zip, Wayback copy).

Source texts are copyrighted news (TimeBank via LDC; Platinum 2013 AP/CNN/WSJ) with no
redistribution licence, so the built rows go to data/raw/contempo/built/ (gitignored)
and only an ID + SHA-256 manifest is frozen in data/frozen/contempo/.

Input construction (ConTempo's code is not public; github.com/frankniujc/contempo has
returned 404 since Aug 2024): the standard sentence-level TRE input used by the
baselines ConTempo reproduces (Ning et al. 2019 / Wen & Ji 2021 data): the sentence(s)
from the sentence containing the earlier event through the sentence containing the
later one (same sentence, or two adjacent sentences), with both event triggers marked.

Sentence segmentation: paragraph breaks (blank/new lines) in the TimeML TEXT, then NLTK
Punkt (English) inside each paragraph. Pinned: nltk==3.9.1 with punkt_tab.

Usage:
  uv run --with nltk==3.9.1 python scripts/prepare_contempo.py            # rebuild + verify against frozen manifests
  uv run --with nltk==3.9.1 python scripts/prepare_contempo.py --freeze   # (re)write frozen manifests
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from gauntlet.tasks.contempo import BUILT_DIR as BUILT
from gauntlet.tasks.contempo import FROZEN_DIR as FROZEN
from gauntlet.tasks.contempo import MATRES_LABELS, row_sha256

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "contempo"
NLTK_DATA = RAW / "nltk_data"

MATRES_PLATINUM = RAW / "MATRES" / "platinum.txt"
TE3_PLATINUM_DIR = RAW / "StructTempRel-EMNLP17" / "data" / "TempEval3" / "Evaluation" / "te3-platinum"
TIMEBANK_DIR = RAW / "te3_official" / "TBAQ-cleaned" / "TimeBank"
TBDENSE_FULL = RAW / "caevo_archive" / "TimebankDense.full.txt"

# Standard TB-Dense split (Chambers et al. 2014 CAEVO; Ning et al. 2018; Han et al. 2019
# rujunhan/EMNLP-2019 data/tbd/test_docs.txt).
TBDENSE_TEST_DOCS = (
    "APW19980227.0489",
    "APW19980227.0494",
    "APW19980308.0201",
    "APW19980418.0210",
    "CNN19980126.1600.1104",
    "CNN19980213.2130.0155",
    "NYT19980402.0453",
    "PRI19980115.2000.0186",
    "PRI19980306.2000.1675",
)
TBDENSE_CODES = {"b": "BEFORE", "a": "AFTER", "i": "INCLUDES", "ii": "IS_INCLUDED", "s": "SIMULTANEOUS", "v": "VAGUE"}
EXPECTED_N = {"matres": 837, "tbdense": 1427}

# Event trigger markers inserted into the context.
OPEN = {"e1": "<e1>", "e2": "<e2>"}
CLOSE = {"e1": "</e1>", "e2": "</e2>"}


@dataclass(frozen=True)
class Event:
    eid: str
    start: int
    end: int
    text: str


@dataclass
class Doc:
    doc_id: str
    text: str
    events: dict[str, Event]  # eid -> Event
    eiid_to_eid: dict[str, str]
    sentences: list[tuple[int, int]]


def parse_timeml(path: Path) -> tuple[str, dict[str, Event], dict[str, str]]:
    """Plain TEXT content with character offsets of every EVENT, plus eiid -> eid."""
    root = ET.parse(path).getroot()
    text_el = root.find("TEXT")
    if text_el is None:
        raise ValueError(f"{path}: no TEXT element")
    parts: list[str] = []
    events: dict[str, Event] = {}
    pos = 0

    def add(s: str | None) -> None:
        nonlocal pos
        if s:
            parts.append(s)
            pos += len(s)

    def walk(el: ET.Element) -> None:
        add(el.text)
        for child in el:
            start = pos
            walk(child)
            if child.tag == "EVENT":
                eid = child.attrib["eid"]
                surface = "".join(child.itertext())
                if eid in events:
                    raise ValueError(f"{path}: duplicate eid {eid}")
                events[eid] = Event(eid, start, pos, surface)
            add(child.tail)

    walk(text_el)
    eiid_to_eid = {mi.attrib["eiid"]: mi.attrib["eventID"] for mi in root.iter("MAKEINSTANCE")}
    return "".join(parts), events, eiid_to_eid


_punkt = None


def _punkt_tokenizer():
    global _punkt
    if _punkt is None:
        import nltk

        # Only the repo-local copy, so the model files hashed below are the ones used.
        nltk.data.path[:] = [str(NLTK_DATA)]
        try:
            nltk.data.find("tokenizers/punkt_tab/english/")
        except LookupError:
            nltk.download("punkt_tab", download_dir=str(NLTK_DATA), quiet=True)
        from nltk.tokenize.punkt import PunktTokenizer

        _punkt = PunktTokenizer("english")
    return _punkt


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Sentence spans: split on line breaks (TimeML paragraphs), then Punkt within each line."""
    tok = _punkt_tokenizer()
    spans: list[tuple[int, int]] = []
    for m in re.finditer(r"[^\n]+", text):
        line = m.group(0)
        for s, e in tok.span_tokenize(line):
            if line[s:e].strip():
                spans.append((m.start() + s, m.start() + e))
    return spans


def load_doc(path: Path) -> Doc:
    text, events, eiid_to_eid = parse_timeml(path)
    return Doc(path.stem, text, events, eiid_to_eid, split_sentences(text))


def sentence_index(doc: Doc, ev: Event) -> int:
    for i, (s, e) in enumerate(doc.sentences):
        if s <= ev.start and ev.end <= e:
            return i
    raise ValueError(f"{doc.doc_id}: event {ev.eid} {ev.text!r} spans a sentence boundary")


def build_context(doc: Doc, e1: Event, e2: Event) -> tuple[str, int]:
    """Sentences from the earlier event's sentence to the later one's, triggers marked."""
    s1, s2 = sentence_index(doc, e1), sentence_index(doc, e2)
    lo, hi = min(s1, s2), max(s1, s2)
    start, end = doc.sentences[lo][0], doc.sentences[hi][1]
    marks = sorted([(e1.start, e1.end, "e1"), (e2.start, e2.end, "e2")])
    if marks[0][1] > marks[1][0]:
        raise ValueError(f"{doc.doc_id}: overlapping events {e1.eid}/{e2.eid}")
    out, cur = [], start
    for s, e, tag in marks:
        out += [doc.text[cur:s], OPEN[tag], doc.text[s:e], CLOSE[tag]]
        cur = e
    out.append(doc.text[cur:end])
    context = re.sub(r"\s+", " ", "".join(out)).strip()
    return context, hi - lo


def build_matres() -> list[dict]:
    docs: dict[str, Doc] = {}
    rows = []
    for line in MATRES_PLATINUM.read_text().splitlines():
        if not line.strip():
            continue
        doc_id, verb1, verb2, eiid1, eiid2, label = line.split("\t")
        if label not in MATRES_LABELS:
            raise ValueError(f"unexpected MATRES label {label!r}")
        if doc_id not in docs:
            docs[doc_id] = load_doc(TE3_PLATINUM_DIR / f"{doc_id}.tml")
        doc = docs[doc_id]
        ev1 = doc.events[doc.eiid_to_eid[f"ei{eiid1}"]]
        ev2 = doc.events[doc.eiid_to_eid[f"ei{eiid2}"]]
        if (ev1.text, ev2.text) != (verb1, verb2):
            raise ValueError(f"{doc_id}: MATRES verbs {verb1}/{verb2} != TimeML {ev1.text}/{ev2.text}")
        context, sent_dist = build_context(doc, ev1, ev2)
        rows.append(
            {
                "id": f"{doc_id}:ei{eiid1}:ei{eiid2}",
                "doc_id": doc_id,
                "event1": ev1.text,
                "event2": ev2.text,
                "context": context,
                "sentence_distance": sent_dist,
                "gold": label,
            }
        )
    return rows


def build_tbdense() -> list[dict]:
    docs: dict[str, Doc] = {}
    rows = []
    for line in TBDENSE_FULL.read_text().splitlines():
        if not line.strip():
            continue
        doc_id, id1, id2, code = line.split("\t")
        if doc_id not in TBDENSE_TEST_DOCS or not (id1.startswith("e") and id2.startswith("e")):
            continue
        if doc_id not in docs:
            docs[doc_id] = load_doc(TIMEBANK_DIR / f"{doc_id}.tml")
        doc = docs[doc_id]
        # Standard TB-Dense event-event set (6,088 pairs; Cassidy et al. 2014; Han et al. 2019):
        # pairs whose two events are TimeBank (TempEval-3 TBAQ-cleaned) events. Pairs with
        # events added only in the TB-Dense annotation files are excluded.
        if id1 not in doc.events or id2 not in doc.events:
            continue
        ev1, ev2 = doc.events[id1], doc.events[id2]
        context, sent_dist = build_context(doc, ev1, ev2)
        rows.append(
            {
                "id": f"{doc_id}:{id1}:{id2}",
                "doc_id": doc_id,
                "event1": ev1.text,
                "event2": ev2.text,
                "context": context,
                "sentence_distance": sent_dist,
                "gold": TBDENSE_CODES[code],
            }
        )
    return rows


def raw_file_hashes(doc_dirs: dict[str, Path], extra: list[Path]) -> dict[str, str]:
    """SHA-256 of every raw input: annotation files, each TimeML doc used, Punkt model files."""
    paths = list(extra)
    for name, d in doc_dirs.items():
        paths += [d / f"{doc_id}.tml" for doc_id in USED_DOCS[name]]
    punkt_dir = NLTK_DATA / "tokenizers" / "punkt_tab" / "english"
    paths += sorted(p for p in punkt_dir.iterdir() if p.is_file())
    return {str(p.relative_to(RAW)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


USED_DOCS: dict[str, list[str]] = {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze", action="store_true", help="write data/frozen/contempo/ manifests")
    args = ap.parse_args()
    ok = True
    for name, build in (("matres", build_matres), ("tbdense", build_tbdense)):
        rows = build()
        USED_DOCS[name] = sorted({r["doc_id"] for r in rows})
        ids = [r["id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{name}: duplicate ids")
        if len(rows) != EXPECTED_N[name]:
            raise ValueError(f"{name}: built {len(rows)} rows, expected {EXPECTED_N[name]}")
        BUILT.mkdir(parents=True, exist_ok=True)
        with (BUILT / f"{name}_test.jsonl").open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest = [{"id": r["id"], "sha256": row_sha256(r)} for r in rows]
        frozen_path = FROZEN / f"{name}_test_ids.jsonl"
        if args.freeze:
            FROZEN.mkdir(parents=True, exist_ok=True)
            with frozen_path.open("w") as fh:
                for m in manifest:
                    fh.write(json.dumps(m) + "\n")
        else:
            frozen = [json.loads(x) for x in frozen_path.read_text().splitlines() if x.strip()]
            if frozen != manifest:
                ok = False
                print(f"{name}: MISMATCH against {frozen_path}", file=sys.stderr)
        print(
            f"{name}: n={len(rows)} docs={len(USED_DOCS[name])} "
            f"labels={dict(Counter(r['gold'] for r in rows))} "
            f"sentence_distance={dict(Counter(r['sentence_distance'] for r in rows))}"
        )
    hashes = raw_file_hashes({"matres": TE3_PLATINUM_DIR, "tbdense": TIMEBANK_DIR}, [MATRES_PLATINUM, TBDENSE_FULL])
    hashes_path = FROZEN / "raw_sha256.json"
    if args.freeze:
        hashes_path.write_text(json.dumps(hashes, indent=2) + "\n")
    elif json.loads(hashes_path.read_text()) != hashes:
        ok = False
        print(f"raw inputs differ from {hashes_path}", file=sys.stderr)
    print(f"raw inputs hashed: {len(hashes)}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
