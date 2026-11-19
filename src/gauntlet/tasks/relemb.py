"""Entity or Relation Embeddings? (Mtumbuka & Schockaert, Findings EMNLP 2024), Jev as the relation classifier.

Paper: https://aclanthology.org/2024.findings-emnlp.348/ ; code: https://github.com/fmtumbuka/RelationEmbeddings
(commit 66d3c9a; data loaders and label inventories only, no evaluation script). The paper's
predictor is a RoBERTa-large relation encoder ([H,T]+Mask, EnCore+Mask, ...) with an MLP softmax
over a fixed relation inventory (Section 3.2): "given a sentence in which two entities are
highlighted ... predict which relationship holds between these two entities, given a pre-defined
set of candidate relation labels" (Section 3). Jev replaces encoder + classifier.

Tasks (one per benchmark, Section 4.1 / Table 1):
- `relemb/wikiwd`: Sorokin & Gurevych (2017) Wikipedia-Wikidata held-out set, 352 labels = 351
  Wikidata properties + P0 (no relation). Metric (Table 3; RECON/KGPool `evaluation_utils`):
  micro P/R/F1 with P0 excluded from both precision and recall counts.
- `relemb/nytfb`: Riedel et al. (2010) NYT-Freebase test set (RECON release), 56 labels (the
  paper's `NYT_RELATION_TYPES`, NA included; 32 occur in test). Metric (Table 3): P@10 and P@30,
  i.e. precision at 10% / 30% recall of the ranked (instance, relation) scores with the NA score
  ignored ("P@K here represents precision at K percent recall", KGPool Section 5, whose numbers
  the paper copies). The paper's text says "averaged across all relation labels", so a per-label
  (macro) variant is reported alongside; the micro one is primary. Jev's choice probabilities are
  the scores.

Wiki-WD has 352 > 255 labels, above the choice primitive's option cap. Faithful decomposition:
the 351 properties, in numeric PID order, are split into two fixed halves (176 / 175); round 1
asks one choice per half, each half also offering "none of the listed relations". The winner of
a half that picks "none" drops out. Both halves "none" -> P0; exactly one real winner -> that
label; two real winners -> round 2 asks a choice between the two. This equals the flat argmax
whenever Jev's preferences are consistent across option sets (P0 wins the flat choice iff it
beats every property, i.e. wins both halves; otherwise the best property is the better of the two
half winners). Round 2 is requested adaptively (`next_requests`).

Prompts (v2, `prompts/relemb_*_v2.json`): options are label NAMES with null descriptions (Wikidata
2016 property labels, Freebase paths). The paper's classifier sees no label text
at all (labels are class indices, `datapipeline/relational_dataset.py`; Limitations: "we do not
attempt to model the relation labels"), so names are the minimum a zero-shot model needs.

Sentences carry several entity pairs (Wiki-WD edges; NYT rows of one sentence). One item = one
sentence; its pairs become independent questions over one shared state ({task statement, plain
sentence}, sent once per request), and each question's instructions carry the sentence with that
pair marked <e1>head</e1>, <e2>tail</e2>. Questions are packed greedily, in pair order, into
requests of at most MAX_REQUEST_CHARS serialized characters. Identical (head span, tail span)
pairs within a sentence share one question.

Exact probability ties go to the earlier label in inventory order (negative label first, then the
paper's label order; Wiki-WD: numeric PID order).
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache, cached_property
from typing import Any

from .. import metrics
from ..task import FROZEN, REPO, Item, Request, Task, read_jsonl

DATA = FROZEN / "relemb"
RAW = REPO / "data" / "raw" / "relemb"
NYT_ZIP = RAW / "NYT.zip"
NYT_ZIP_SHA256 = "7e0816de89f69545653e37fcd4deb59364a053ee9fbe4cd1fc181086cc6643c3"
NYT_MEMBER = "NYT/dataset_triples_test.json"
PREPARE = "uv run python scripts/prepare_relemb.py"

# Frozen request-size rule: serialized JSON characters of state + questions. v2 requests (option
# names with null descriptions) measure 3.7 chars/token (o200k and cl100k agree within 1%), so 90K
# chars is ~24K tokens, inside the 32K context with margin for a less efficient tokenizer.
MAX_REQUEST_CHARS = 90_000
# Frozen truncation rule: sentences are cut to their first MAX_SENTENCE_CHARS characters. The
# longest frozen sentence is far shorter (Wiki-WD 203 chars after joining tokens; NYT 1,233),
# so nothing is truncated.
MAX_SENTENCE_CHARS = 8_000
RECALL_LEVELS = (0.1, 0.3)

# Paper numbers, 0-1 scale.
# Table 3 (p. 6009): Wiki-WD P / R / F1 and NYT-FB P@10 / P@30, standard pre-training.
PAPER_WIKIWD = {"RECON": (0.872, 0.872, 0.872), "KGPool": (0.886, 0.886, 0.886), "[H,T]": (0.759, 0.747, 0.752),
                "[H,T]+Mask": (0.898, 0.893, 0.896), "EnCore+Mask": (0.894, 0.888, 0.891)}
PAPER_NYTFB = {"RECON": (0.875, 0.741), "KGPool": (0.923, 0.867), "[H,T]": (0.796, 0.785),
               "[H,T]+Mask": (0.947, 0.931), "EnCore+Mask": (0.949, 0.930)}


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _clip(text: str) -> tuple[str, bool]:
    return (text[:MAX_SENTENCE_CHARS], True) if len(text) > MAX_SENTENCE_CHARS else (text, False)


def mark_tokens(tokens: Sequence[str], head: Sequence[int], tail: Sequence[int]) -> str:
    """Space-joined tokens with <e1>..</e1> around each contiguous run of head positions and
    <e2>..</e2> around tail runs (opening e1 before e2, closing e2 before e1 at shared tokens)."""
    hs, ts = set(head), set(tail)
    out = []
    for i, tok in enumerate(tokens):
        pre = ("<e1>" if i in hs and i - 1 not in hs else "") + ("<e2>" if i in ts and i - 1 not in ts else "")
        post = ("</e2>" if i in ts and i + 1 not in ts else "") + ("</e1>" if i in hs and i + 1 not in hs else "")
        out.append(pre + tok + post)
    return " ".join(out)


def mark_chars(text: str, head: tuple[int, int], tail: tuple[int, int]) -> str:
    """Insert <e1>/<e2> tags at character spans [start, end)."""
    # At equal offsets: closers before openers; e2 closes before e1; e1 opens before e2.
    inserts = [(head[0], 2, "<e1>"), (tail[0], 3, "<e2>"), (tail[1], 0, "</e2>"), (head[1], 1, "</e1>")]
    out, prev = [], 0
    for pos, _, tag in sorted(inserts):
        out.append(text[prev:pos])
        out.append(tag)
        prev = pos
    out.append(text[prev:])
    return "".join(out)


def _mentions(low: str, name: str) -> list[int]:
    """Start offsets of `name` in `low`: whole-word matches if any exist, else any substring match."""
    pat = re.escape(name)
    return [m.start() for m in re.finditer(rf"(?<!\w){pat}(?!\w)", low)] or [m.start() for m in re.finditer(pat, low)]


def nyt_spans(sentence: str, sub: str, obj: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Character spans of the NYT head/tail mentions: first case-insensitive occurrence of the entity
    name (underscores -> spaces, as the paper's loader does), preferring whole-word matches; the tail
    takes its first occurrence not overlapping the head if one exists."""
    low = sentence.lower()
    h_name, t_name = sub.replace("_", " ").lower(), obj.replace("_", " ").lower()
    h_starts, t_starts = _mentions(low, h_name), _mentions(low, t_name)
    if not h_name or not t_name or not h_starts or not t_starts:
        raise ValueError(f"entity {sub!r} or {obj!r} not found in sentence")
    head = (h_starts[0], h_starts[0] + len(h_name))
    free = [t for t in t_starts if t + len(t_name) <= head[0] or t >= head[1]]
    t = (free or t_starts)[0]
    return head, (t, t + len(t_name))


def argmax_label(probs: dict[str, float], keys: Sequence[str]) -> str:
    """Key with the highest probability; exact ties go to the earlier key in `keys`."""
    best, best_p = keys[0], float(probs.get(keys[0], 0.0))
    for k in keys[1:]:
        p = float(probs.get(k, 0.0))
        if p > best_p:
            best, best_p = k, p
    return best


def micro_prf(gold: Sequence[str], pred: Sequence[str], negative: str) -> dict[str, float | int]:
    """Sorokin `evaluate_instance_based`: the negative label counts in neither
    precision's nor recall's denominator."""
    n_pred = sum(p != negative for p in pred)
    n_gold = sum(g != negative for g in gold)
    correct = sum(p == g and p != negative for g, p in zip(gold, pred))
    p = correct / n_pred if n_pred else 0.0
    r = correct / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "n_correct": correct, "n_pred_positive": n_pred, "n_gold_positive": n_gold}


def precision_at_recall(entries: list[tuple[float, int, int, bool]], n_pos: int, levels: Sequence[float]) -> dict[str, float]:
    """Precision at the first rank where recall reaches each level.

    `entries` = (score, row, label index, is_gold); ranked by score descending, ties by (row, label).
    """
    out: dict[str, float] = {}
    if n_pos == 0:
        return {f"p_at_{round(level * 100)}": float("nan") for level in levels}
    ranked = sorted(entries, key=lambda e: (-e[0], e[1], e[2]))
    pending = sorted(levels)
    tp = 0
    for k, entry in enumerate(ranked, 1):
        tp += entry[3]
        while pending and tp / n_pos >= pending[0]:
            out[f"p_at_{round(pending.pop(0) * 100)}"] = tp / k
        if not pending:
            break
    for level in pending:  # recall level never reached
        out[f"p_at_{round(level * 100)}"] = 0.0
    return out


_Q_OPEN = len('{"type": "choice", "instructions": ')
_Q_MID = len(', "criteria": ')
_R_OPEN = len('{"state": ')
_R_MID = len(', "questions": ')


def _jlen(obj: Any) -> int:
    return len(json.dumps(obj, ensure_ascii=False))


@dataclass(frozen=True)
class Criteria:
    """A choice criteria map shared by many questions, with its serialized length."""

    mapping: dict[str, None]
    chars: int

    @classmethod
    def of(cls, mapping: dict[str, None]) -> "Criteria":
        return cls(mapping, _jlen(mapping))


@dataclass(frozen=True)
class QSpec:
    qid: str
    instructions: str
    criteria: Criteria

    def question(self) -> dict[str, Any]:
        return {"type": "choice", "instructions": self.instructions, "criteria": self.criteria.mapping}

    def chars(self) -> int:
        """Length of `"qid": {question}` inside the serialized questions object."""
        return _jlen(self.qid) + 2 + _Q_OPEN + _jlen(self.instructions) + _Q_MID + self.criteria.chars + 1


def plan_chunks(state: Any, groups: list[list[QSpec]]) -> list[list[int]]:
    """Greedy, order-preserving packing of question groups into requests whose serialized
    {"state", "questions"} JSON is at most MAX_REQUEST_CHARS (a group larger than the limit
    goes alone). Returns group indices per request."""
    base = _R_OPEN + _jlen(state) + _R_MID + 2 + 1  # braces of the questions object + closing brace
    chunks: list[list[int]] = []
    cur: list[int] = []
    total = base
    for g, group in enumerate(groups):
        size = sum(q.chars() for q in group) + 2 * (len(group) - 1)
        add = size + (2 if cur else 0)
        if cur and total + add > MAX_REQUEST_CHARS:
            chunks.append(cur)
            cur, total, add = [], base, size
        cur.append(g)
        total += add
    if cur:
        chunks.append(cur)
    return chunks


def build_requests(item_id: str, prefix: str, state: Any, groups: list[list[QSpec]], skip: Any = ()) -> list[Request]:
    """Requests `<prefix><k>` for the planned chunks, omitting keys in `skip`."""
    out = []
    for k, chunk in enumerate(plan_chunks(state, groups)):
        key = f"{prefix}{k}"
        if key in skip:
            continue
        questions = {q.qid: q.question() for g in chunk for q in groups[g]}
        out.append(Request(item_id=item_id, key=key, state=state, questions=questions))
    return out


def _question_answers(answers: dict[str, dict[str, Any]], prefix: str) -> dict[str, dict[str, Any]]:
    """Merge the per-question answers of all requests whose key starts with `prefix`."""
    merged: dict[str, dict[str, Any]] = {}
    for key, ans in answers.items():
        if key.startswith(prefix):
            merged.update(ans)
    return merged


class _Inventory:
    """A prompt's label inventory: gold label <-> option name, tie-break order."""

    def __init__(self, prompt: dict[str, Any]):
        self.labels: list[str] = [lab["label"] for lab in prompt["labels"]]
        self.key: dict[str, str] = {lab["label"]: lab["key"] for lab in prompt["labels"]}
        self.label_of: dict[str, str] = {v: k for k, v in self.key.items()}
        self.negative: str = prompt["negative_label"]
        if len(set(self.key.values())) != len(self.labels):
            raise ValueError("duplicate choice keys in prompt")
        self.all = Criteria.of(self.criteria(self.labels))

    def criteria(self, labels: Sequence[str]) -> dict[str, None]:
        """Option names only; null descriptions (the paper's classifier sees no label text)."""
        return {self.key[lab]: None for lab in labels}


class _PromptTask(Task):
    """Parses the prompt file once per task instance."""

    @cached_property
    def spec(self) -> dict[str, Any]:
        return self.prompt()

    @cached_property
    def inv(self) -> _Inventory:
        return _Inventory(self.spec)

    def instructions(self, marked: str, head: str, tail: str) -> str:
        return self.spec["question"].format(marked_sentence=marked, head=head, tail=tail)

    def state(self, sentence: str) -> dict[str, str]:
        """Shared per request: the task statement once, then the plain sentence."""
        return {"task": self.spec["task"], "sentence": sentence}


# --------------------------------------------------------------------------------------------
# Wiki-WD


def _unique_pairs(pairs: Sequence[Any]) -> tuple[list[Any], list[int]]:
    """Unique pairs in first-occurrence order, and each input's index into them."""
    uniq: list[Any] = []
    index: dict[Any, int] = {}
    where = []
    for pair in pairs:
        if pair not in index:
            index[pair] = len(uniq)
            uniq.append(pair)
        where.append(index[pair])
    return uniq, where


class WikiWDTask(_PromptTask):
    name = "relemb/wikiwd"
    prompt_file = "relemb_wikiwd_v2.json"
    data_file = "wikiwd_test.jsonl.gz"

    @cached_property
    def halves(self) -> tuple[list[str], list[str]]:
        """Fixed round-1 partition of the positive labels (prompt `halves`, numeric PID order)."""
        a, b = self.spec["halves"]
        pos = [lab for lab in self.inv.labels if lab != self.inv.negative]
        if sorted(a + b) != sorted(pos) or set(a) & set(b) or max(len(a), len(b)) + 1 > 255:
            raise ValueError("prompt halves must partition the positive labels into choices of <= 255 options")
        return a, b

    @cached_property
    def half_criteria(self) -> tuple[Criteria, Criteria]:
        none = self.spec["none_option"]["key"]
        return tuple(Criteria.of({none: None, **self.inv.criteria(h)}) for h in self.halves)

    def items(self) -> list[Item]:
        out = []
        for r in read_jsonl(DATA / self.data_file):
            gold = [e[2] for e in r["edges"]]
            out.append(Item(id=r["id"], gold=gold, fields={"tokens": r["tokens"], "edges": r["edges"]}))
        return out

    @staticmethod
    def _pairs(item: Item) -> tuple[list[tuple[tuple[int, ...], tuple[int, ...]]], list[int]]:
        return _unique_pairs([(tuple(left), tuple(right)) for left, right, _ in item.fields["edges"]])

    @staticmethod
    def _tokens(item: Item) -> list[str]:
        return [html.unescape(t) for t in item.fields["tokens"]]

    def _state(self, item: Item) -> tuple[dict[str, str], bool]:
        sentence, cut = _clip(" ".join(self._tokens(item)))
        return self.state(sentence), cut

    def _pair_instructions(self, item: Item) -> tuple[list[str], bool]:
        tokens = self._tokens(item)
        out, cut = [], False
        for head, tail in self._pairs(item)[0]:
            marked, c = _clip(mark_tokens(tokens, head, tail))
            cut = cut or c
            out.append(self.instructions(marked, " ".join(tokens[i] for i in head), " ".join(tokens[i] for i in tail)))
        return out, cut

    def _round1_groups(self, item: Item) -> list[list[QSpec]]:
        crit_a, crit_b = self.half_criteria
        return [[QSpec(f"p{j}a", instr, crit_a), QSpec(f"p{j}b", instr, crit_b)]
                for j, instr in enumerate(self._pair_instructions(item)[0])]

    def _half_winner(self, q: dict[str, Any], half: Sequence[str]) -> str | None:
        """Round-1 winner label of one half, or None when "none of the listed relations" wins."""
        keys = [self.spec["none_option"]["key"]] + [self.inv.key[lab] for lab in half]
        win = argmax_label(q["probabilities"], keys)
        return None if win == keys[0] else self.inv.label_of[win]

    def _winners(self, n_pairs: int, answers: dict[str, dict[str, Any]]) -> list[tuple[str | None, str | None]]:
        r1 = _question_answers(answers, "r1.")
        a, b = self.halves
        return [(self._half_winner(r1[f"p{j}a"], a), self._half_winner(r1[f"p{j}b"], b)) for j in range(n_pairs)]

    def _round2_groups(self, item: Item, winners: list[tuple[str | None, str | None]]) -> list[list[QSpec]]:
        groups = []
        instrs = self._pair_instructions(item)[0]
        for j, (wa, wb) in enumerate(winners):
            if wa is not None and wb is not None:
                groups.append([QSpec(f"p{j}", instrs[j], Criteria.of(self.inv.criteria([wa, wb])))])
        return groups

    def next_requests(self, item: Item, answers: dict[str, dict[str, Any]]) -> list[Request]:
        state, _ = self._state(item)
        groups1 = self._round1_groups(item)
        n1 = len(plan_chunks(state, groups1))
        if any(f"r1.{k}" not in answers for k in range(n1)):
            return build_requests(item.id, "r1.", state, groups1, skip=answers)
        groups2 = self._round2_groups(item, self._winners(len(groups1), answers))
        return build_requests(item.id, "r2.", state, groups2, skip=answers) if groups2 else []

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        inv = self.inv
        pairs, edge_pair = self._pairs(item)
        winners = self._winners(len(pairs), answers)
        r2 = _question_answers(answers, "r2.")
        pair_label = []
        for j, (wa, wb) in enumerate(winners):
            if wa is None and wb is None:
                pair_label.append(inv.negative)
            elif wa is None or wb is None:
                pair_label.append(wa or wb)
            else:
                keys = [inv.key[wa], inv.key[wb]]  # half A first: lower PIDs win exact ties
                pair_label.append(inv.label_of[argmax_label(r2[f"p{j}"]["probabilities"], keys)])
        return {"labels": [pair_label[j] for j in edge_pair], "round1": [[wa, wb] for wa, wb in winners]}

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        inv = self.inv
        gold = [g for it in items for g in it.gold]
        pred = [p for it in items for p in predictions[it.id]["labels"]]
        out: dict[str, Any] = {"n": len(gold), "n_sentences": len(items), **micro_prf(gold, pred, inv.negative)}
        positives = sorted({g for g in gold if g != inv.negative})
        out["macro_f1"] = metrics.macro_f1(gold, pred, labels=positives)
        out["accuracy"] = metrics.accuracy(gold, pred)
        out["n_round2_pairs"] = sum(1 for it in items for wa, wb in predictions[it.id]["round1"] if wa and wb)
        out["paper_p_r_f1"] = PAPER_WIKIWD
        out["n_truncated"] = sum(self._state(it)[1] or self._pair_instructions(it)[1] for it in items)
        return out


# --------------------------------------------------------------------------------------------
# NYT-FB


@cache
def _nyt_rows() -> tuple[dict[str, str], ...]:
    if not NYT_ZIP.exists():
        raise FileNotFoundError(f"{NYT_ZIP} missing; run `{PREPARE}`")
    if _sha256_file(NYT_ZIP) != NYT_ZIP_SHA256:
        raise ValueError(f"{NYT_ZIP} SHA-256 differs from the frozen release hash {NYT_ZIP_SHA256}")
    with zipfile.ZipFile(NYT_ZIP) as zf, zf.open(NYT_MEMBER) as fh:
        return tuple(json.load(fh)["data"])


def nyt_row_sha256(row: dict[str, str]) -> str:
    return canonical_sha256([row["sent"], row["sub"], row["obj"], row["rel"]])


class NytFbTask(_PromptTask):
    name = "relemb/nytfb"
    prompt_file = "relemb_nytfb_v2.json"

    def items(self) -> list[Item]:
        raw = _nyt_rows()
        out = []
        for r in read_jsonl(DATA / "nytfb_test.jsonl.gz"):
            rows = [raw[i] for i in r["rows"]]
            if len({row["sent"] for row in rows}) != 1:
                raise ValueError(f"{r['id']}: rows do not share one sentence")
            if canonical_sha256([nyt_row_sha256(row) for row in rows]) != r["sha256"]:
                raise ValueError(f"{r['id']}: raw rows differ from the frozen hash")
            fields = {"sentence": rows[0]["sent"], "pairs": [[row["sub"], row["obj"]] for row in rows], "rows": r["rows"]}
            out.append(Item(id=r["id"], gold=[row["rel"] for row in rows], fields=fields))
        return out

    @staticmethod
    def _pairs(item: Item) -> tuple[list[tuple[str, str]], list[int]]:
        return _unique_pairs([(sub, obj) for sub, obj in item.fields["pairs"]])

    def _groups(self, item: Item) -> tuple[dict[str, str], list[list[QSpec]], bool]:
        sentence = item.fields["sentence"]
        state_text, cut = _clip(sentence)
        groups = []
        for j, (sub, obj) in enumerate(self._pairs(item)[0]):
            h, t = nyt_spans(sentence, sub, obj)
            marked, cut_m = _clip(mark_chars(sentence, h, t))
            cut = cut or cut_m
            instr = self.instructions(marked, sentence[h[0]:h[1]], sentence[t[0]:t[1]])
            groups.append([QSpec(f"q{j}", instr, self.inv.all)])
        return self.state(state_text), groups, cut

    def requests(self, item: Item):
        state, groups, _ = self._groups(item)
        yield from build_requests(item.id, "c", state, groups)

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        inv = self.inv
        merged = _question_answers(answers, "c")
        keys = [inv.key[lab] for lab in inv.labels]
        pairs, row_pair = self._pairs(item)
        pair_labels, pair_probs = [], []
        for j in range(len(pairs)):
            probs = merged[f"q{j}"]["probabilities"]
            pair_labels.append(inv.label_of[argmax_label(probs, keys)])
            pair_probs.append([float(probs.get(k, 0.0)) for k in keys])
        # Probabilities as lists in prompt label order (inv.labels), to keep predictions compact.
        return {"labels": [pair_labels[j] for j in row_pair], "row_pair": row_pair, "pair_labels": pair_labels, "pair_probs": pair_probs}

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        inv = self.inv
        positives = [(li, lab) for li, lab in enumerate(inv.labels) if lab != inv.negative]
        gold, pred, entries = [], [], []
        per_label: dict[str, list[tuple[float, int, int, bool]]] = {lab: [] for _, lab in positives}
        row = 0
        for it in items:
            p = predictions[it.id]
            for g, j in zip(it.gold, p["row_pair"]):
                probs = p["pair_probs"][j]
                for li, lab in positives:
                    e = (probs[li], row, li, g == lab)
                    entries.append(e)
                    per_label[lab].append(e)
                gold.append(g)
                pred.append(p["pair_labels"][j])
                row += 1
        n_pos = sum(g != inv.negative for g in gold)
        out: dict[str, Any] = {"n": len(gold), "n_sentences": len(items)}
        out.update(precision_at_recall(entries, n_pos, RECALL_LEVELS))
        macro: dict[str, list[float]] = {}
        for ents in per_label.values():
            n_lab = sum(e[3] for e in ents)
            if n_lab:
                for k, v in precision_at_recall(ents, n_lab, RECALL_LEVELS).items():
                    macro.setdefault(k, []).append(v)
        out["macro_over_labels"] = {k: metrics.mean(v) for k, v in macro.items()}
        out["macro_n_labels"] = sum(1 for ents in per_label.values() if any(e[3] for e in ents))
        out["argmax"] = {**micro_prf(gold, pred, inv.negative), "accuracy": metrics.accuracy(gold, pred)}
        out["n_gold_positive"] = n_pos
        out["paper_p_at_10_30"] = PAPER_NYTFB
        out["n_truncated"] = sum(self._groups(it)[2] for it in items)
        return out


TASKS: dict[str, Callable[[], Task]] = {
    "relemb/wikiwd": WikiWDTask,
    "relemb/nytfb": NytFbTask,
}
