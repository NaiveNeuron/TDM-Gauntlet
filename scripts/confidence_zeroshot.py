"""Confidence analyses of the zero-shot evaluation -> results/confidence.json (keys "rq1", "rq2", "human", "rerun", "sets").

Offline only: saved runs/<task>/{predictions.jsonl, responses*.jsonl.gz, scores.json}, data/frozen (Wiki-WD item
structure), data/raw/smarter/hatexplain (per-annotator votes), prompts/*.json (label orders), and, for the re-run
check and INLI's dev-fitted threshold, runs/jev-opt/study-<task>-zero/jev_responses*.jsonl.gz with
jev-opt/experiments/<exp>/data/{train,val,test}.jsonl. No Jev call, no LLM call. RQ3 is
jev-opt/experiments/confidence_readout.py; tables, macros and the figure are scripts/paper_fig_confidence.py.
Run from the repo root:  uv run python scripts/confidence_zeroshot.py

DEFINITIONS, revision 3 (fixed before the revision-3 results were computed). Revision 2 adopted Rao &
Callison-Burch (2026, arXiv:2609.29769v2); revision 3 follows prior work and adds: the paper as the top-level unit,
equal-mass ECE with a noise floor, Brier and RPS, Learn-then-Test thresholds, human-disagreement strata, top-band
error rate and AUGRC/E-AURC as headline metrics, Holm for group tests and BH for per-set counts.
==================================================================================================================
Sets and papers. The 46 comparable test sets are the rows of scripts/paper_tables.rows() with a task-specific
number. Outcome group = that row's "cat": replaces ("matches"), between, below. Paper = the row's bibkey; it is the
top-level unit of replication.

Unit, prediction, correctness. A unit is one decision the set's scoring counts; the prediction is the recorded one
in predictions.jsonl (the evaluation's frozen decoding); correct = prediction equals gold.

Per unit:
  confidence (ranking only: AUROC, bands, selective prediction) = Jev's `confidence` field of the answer that made
    the prediction (Choice, Score; not a probability, can be below 1/K); binary decisions without the field get
    (K P(predicted) - 1)/(K - 1) with K = 2 = 2 P(predicted) - 1 (= 2|P(yes) - 0.5| at a 0.5 threshold).
  probability (calibration: ECE) = the probability Jev assigns to the predicted option (the argmax option's share of
    the answer's probabilities, checked against the recorded label's own probability; Noul: P(predicted)); none
    where the prediction aggregates several answers.
  Brier (where a full probability vector over the set's labels exists) = half the squared error of the renormalised
    vector, sum_k (p_k - 1[k = gold])^2 / 2, in [0, 1]; for binary decisions (p - y)^2. NLL is not used (Jev returns
    exact zeros).
Per set:
  - Single-question Choice sets (XSTest, SciPrompt x4, LifeTox, edit intent, EZ-STANCE A/B, MATRES, TB-Dense,
    MISMATCHED, INLI, SMARTER x3, CICC x5): unit = item (CICC: test items only); confidence = field; probability =
    argmax share; Brier over the label-keyed vector (SciPrompt through the prompt's option->label map, CICC through
    the prompt's label order, LifeTox from P(unsafe)).
  - PiNose x4 (one Noul): prediction p > 0.5; probability = max(p, 1 - p); confidence = 2|p - 0.5|; Brier (p - y)^2.
  - FENICE: unit = test summary; s = mean over claims of the best chunk's Noul; prediction s > the val threshold;
    confidence = 2 P(pred) - 1 with P(pred) = s or 1 - s; no probability, no Brier.
  - Scalar constructs pairwise x3: unit = pair; prediction = the debiased choice (0 = tie, wrong); confidence =
    2|q - 0.5| for q = mean P(text 1) over 4 conditions; no probability, no Brier.
  - Navigating the Unknown x2: unit = item x partition (gold = the intent if in the partition's ID set, else
    "fallback"); field / argmax share / Brier over the partition's labels.
  - DREsS: unit = essay x rubric; prediction = the argmax level; correct = equals the gold level; field / predicted
    level's probability / Brier over the 9 levels; also RPS = (1/(K-1)) sum_{j<K} (F(j) - 1[gold <= j])^2 with F the
    cumulative level distribution, against the climatology RPS of the train split's level frequencies per rubric.
  - LegalSeg: unit = sentence; field / argmax share / Brier over the 7 labels.
  - NYT-FB: unit = labelled row; prediction = the pair's argmax label; field / argmax share / Brier over the 56 labels.
  - Wiki-WD: unit = labelled edge; confidence = the minimum field on the decision path; no probability, no Brier.
  - TTP x3: unit = text; prediction = the top-ranked technique; correct = it is gold (P@1); probability = its P(yes);
    confidence = 2 P(yes) - 1; Brier = (P(yes) - correct)^2.
  Excluded (no item-level correctness against one gold decision): PRP-Graph, BIORC800, MADE, FactPICO, MedReadMe,
  Scalar constructs pointwise x3.
  Primitive group: choice (field of a Choice answer), binary (PiNose, FENICE, pairwise scalar, TTP), score (DREsS),
  tournament (Wiki-WD).
Clusters (the bootstrap's test unit): the item; for multi-unit items the item carrying them (DREsS essay, utterance,
LegalSeg document, NYT-FB / Wiki-WD sentence).

Failure cause, from the paper's own grouping (results.tex Section 4.1, optimizing_decision_.tex): (i) conventions =
the eight optimization configurations' sets (TB-Dense, EZ-STANCE A, SMARTER implicit hate, edit intent, INLI,
DREsS); (ii) opaque label spaces = TTP x3, MADE (excluded), Wiki-WD; (iii) coarse probabilities = CICC; matched =
outcome "replaces"; every other set "unassigned" (no cause inferred). Pre-registered split: conventions-Choice
(conventions without DREsS, the only Score set) is reported next to conventions on equal footing.

Statistics per set:
  - AUROC of confidence for correctness (ties 1/2).
  - ECE (sets with a probability), primary: equal-mass top-label ECE, 15 bins cut at the n/15 quantiles of the
    probability and moved to tie boundaries (ties never split), sum_b (n_b/n)|acc_b - mean p_b|. Noise floor:
    1,000 label draws y* ~ Bernoulli(p) per unit, ECE recomputed with the same bins: floor = mean, consistency band =
    95th percentile, miscalibration p = (1 + #{null >= observed}) / 1,001. Sensitivity: 15 equal-width bins.
  - Brier (above); DREsS RPS and its climatology.
  - Error rate per confidence band [0,.25), [.25,.5), [.5,.75), [.75,1] (negative confidences in the first); a band
    with fewer than 20 units is omitted (tables, figure, pooling). Headline: top-band error rate (conditional risk in
    [.75,1]). The share of errors in the top band is kept as a secondary column only (base-rate dependent).
  - Selective risk: AUGRC = integral over coverage of the share of all units that are accepted errors (Traub et al.
    2024); E-AURC = AURC - AURC*, AURC* = r + (1 - r) ln(1 - r) at overall risk r (Geifman et al. 2019); AURC; ties
    at random in expectation throughout.
  - Bootstrap within a set: 2,000 cluster resamples, percentile 95% CI ("these sets"); bins fixed from the full set.
    One-sided bootstrap p-values (1 + #{replicates on the null side}) / 2,001 for "AUROC > 0.5" and "top-band error
    rate < 10%". Seeds: SeedSequence(0) -> one child per set in table order -> (bootstrap, halvings, floor) streams.
Per-set count claims (how many sets show a property) use Benjamini-Hochberg at q = 0.05 over all sets for which
the property is defined: AUROC > 0.5; ECE above its consistency band (miscalibration p); top-band error rate < 10%.

Group statistics (outcome, cause, primitive, conventions-Choice): each set weighted equally (paper-weighted means,
averaging sets within a paper first, as a secondary row): mean AUROC, ECE, Brier, AUGRC, E-AURC; error rate per band
pooled with unit weight 1/n_set over kept cells. Primary interval: fixed-set bootstrap (each set's own cluster
replicates; describes these sets). Groups whose sets come from 3 or more papers also get a paper-level bootstrap
(papers resampled with replacement within the group; sets and items within a drawn paper kept as they are, point
values) for "tasks like these". Groups from 1-2 papers (coarse: CICC only; opaque: TTP and RelEmb) are descriptive:
no paper-level interval, no test. Pre-specified group tests (Holm over the family): conventions - matched,
conventions - unassigned, below - replaces, each on mean AUROC, mean ECE, pooled top-band error rate and mean AUGRC
(12 tests); two-sided paper-level bootstrap p = 2 min(share <= 0, share >= 0) with papers resampled independently
per group (a paper with sets in both groups is resampled in each).

Selective prediction (RQ1) against the paper's number, only where it is an accuracy over the same units: primary
metric accuracy (XSTest, SciPrompt x4, INLI), P@1 (TTP x3), or the paper reports the method's accuracy (PiNose x4).
  - Primary: Learn-then-Test fixed-sequence threshold (Angelopoulos et al. 2021; delta = 0.05): distinct confidences
    in descending order; at each the kept units' one-sided Clopper-Pearson lower bound Beta^-1(delta; k, n - k + 1)
    is compared with the paper's number; thresholds are accepted until the first failure; the lowest accepted one is
    used (none: keep nothing). Chosen on a random half of the clusters, scored on the other half, swapped and pooled;
    50 halvings. Reported: mean coverage, mean kept accuracy, share of halvings whose pooled kept accuracy reaches
    the number. INLI also dev-fitted (threshold from the same question's saved val run, applied to test).
  - Sensitivity (appendix): naive cross-fitted threshold (lowest threshold whose kept accuracy on the fit half
    reaches the number) and the oracle c* (chosen on the test units it is scored on; optimistic upper bound).
  - All sets: kept accuracy at 50% / 80% / 100% coverage (label-free quantile thresholds), gains; AUGRC.

Human disagreement: HateXplain test posts with their three annotator votes (data/raw/smarter/hatexplain): accuracy,
AUROC and ECE (equal-mass) on unanimous (3/3) vs split (2/1) posts; AUROC of confidence for "annotators unanimous";
soft-label ECE (bins of the predicted option's probability, |mean annotator share of the predicted label - mean p|);
DistCE = mean total-variation distance between Jev's distribution and the annotator distribution (Baan et al. 2022),
against a majority-label one-hot predictor; expected agreement with a random annotator for Jev and for the majority
label. No other analysed set releases per-item annotations; DREsS had one instructor per essay (its paper:
inter-annotator agreement cannot be measured), so no human ceiling is available for DREsS.

Re-run non-determinism (data behind \\bodyRerunIdentical / \\bodyRerunFlipPct): the evaluation's test responses for
INLI, TB-Dense and SMARTER implicit hate against the optimization study's zero-seed test responses (byte-identical
requests). Flip = the chosen label differs; flip rate per first-run confidence band (bands under 20 omitted), share
of flips with first-run confidence < 0.6, and the AUROC of (1 - first-run confidence) for a flip.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import paper_tables as pt  # noqa: E402

RUNS = ROOT / "runs"
STUDY = RUNS / "jev-opt"
OUT = ROOT / "results" / "confidence.json"
B = 2000
SEED = 0
HALVINGS = 50
FLOOR_DRAWS = 1000
DELTA = 0.05
FDR_Q = 0.05
MIN_BAND = 20
MIN_PAPERS = 3
BANDS = (0.0, 0.25, 0.5, 0.75)  # lower edges; the last band is [0.75, 1]
BAND_NAMES = ("[0,.25)", "[.25,.5)", "[.5,.75)", "[.75,1]")
ECE_BINS = 15
COVERAGES = tuple(round(0.05 * k, 2) for k in range(1, 21))

CHOICE_SETS = ("xstest", "sciprompt/arxiv", "sciprompt/s2orc", "sciprompt/sdpra", "sciprompt/emerging_nlp", "lifetox",
               "edit_intent", "ezstance/subtaskA", "ezstance/subtaskB", "contempo/matres", "contempo/tbdense",
               "mismatched", "inli", "smarter/hatexplain", "smarter/latent_hate", "smarter/implicit_hate",
               "cicc/mtod", "cicc/hwu64", "cicc/c150", "cicc/acid", "cicc/atis")
FIELD_SETS = (*CHOICE_SETS, "nav_unknown/clinc150", "nav_unknown/banking77", "dress_new", "legalseg", "relemb/nytfb",
              "relemb/wikiwd")  # confidence = Jev's field
EXCLUDED = {
    "prp_graph/trec_covid": "unit is a query ranking (nDCG@10) built from pairwise Choices, with no item-level decision against a gold label",
    "biorc800": "multi-label sentences decided by one Noul per sentence x label, with no single item-level decision",
    "made": "1,154-code multi-label sets from top-down Nouls, with no single item-level decision",
    "factpico": "gold is a mean of annotator ratings (fractional), not a level the Score can match",
    "medreadme": "continuous gold readability (mean rating), so there is no item-level correctness",
    "scalar_constructs/immigration_fear/pointwise": "continuous latent gold score, so there is no item-level correctness",
    "scalar_constructs/ad_negativity/pointwise": "continuous latent gold score, so there is no item-level correctness",
    "scalar_constructs/grandstanding/pointwise": "continuous latent gold score, so there is no item-level correctness",
}
CAUSE = {
    "contempo/tbdense": "conventions", "ezstance/subtaskA": "conventions", "smarter/implicit_hate": "conventions",
    "edit_intent": "conventions", "inli": "conventions", "dress_new": "conventions",
    "ttp/tram": "opaque", "ttp/procedures_plus": "opaque", "ttp/expert": "opaque", "made": "opaque",
    "relemb/wikiwd": "opaque",
    "cicc/mtod": "coarse", "cicc/hwu64": "coarse", "cicc/c150": "coarse", "cicc/acid": "coarse", "cicc/atis": "coarse",
}
CAUSE_ORDER = ("matched", "conventions", "opaque", "coarse", "unassigned")
OUTCOME_ORDER = ("replaces", "between", "below")
PRIMITIVE_ORDER = ("choice", "binary", "score", "tournament")
TESTS = (("conventions", "matched"), ("conventions", "unassigned"), ("below", "replaces"))
TEST_STATS = ("auroc", "ece", "top_rate", "augrc")
ACC_PRIMARY = ("xstest", "sciprompt/arxiv", "sciprompt/s2orc", "sciprompt/sdpra", "sciprompt/emerging_nlp", "inli")
P_AT_1 = ("ttp/tram", "ttp/procedures_plus", "ttp/expert")
PAPER_ACCURACY = ("pinose/true_false", "pinose/nq", "pinose/triviaqa", "pinose/webq")
DEV = {"inli": ("inli", "relation")}  # set -> (study slug, question id) with a saved val run of the same question
DRESS_LEVELS = 9
DRESS_RUBRICS = ("content", "organization", "language")


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------

def predictions(task: str) -> list[dict]:
    return [json.loads(line) for line in (RUNS / task / "predictions.jsonl").read_text().splitlines()]


def responses(task: str) -> Iterator[dict]:
    for shard in sorted((RUNS / task).glob("responses*.jsonl.gz")):
        with gzip.open(shard, "rt") as fh:
            for line in fh:
                yield json.loads(line)


def scores(task: str) -> dict:
    return json.loads((RUNS / task / "scores.json").read_text())


def prompt(name: str) -> dict:
    return json.loads((ROOT / "prompts" / f"{name}.json").read_text())


def share(probs: dict[str, float], label: str) -> float:
    """The label's share of an answer's returned probabilities (options Jev omits count 0)."""
    tot = sum(max(float(v), 0.0) for v in probs.values())
    return max(float(probs.get(label, 0.0)), 0.0) / tot if tot > 0 else 0.0


def top_share(probs: dict[str, float]) -> float:
    """The argmax option's share (Jev's option keys can be display names, not label codes)."""
    vals = [max(float(v), 0.0) for v in probs.values()]
    return max(vals) / sum(vals) if sum(vals) > 0 else 0.0


def pred_prob(probs: dict[str, float], label: str) -> float:
    """Share of the predicted option, keyed by label; asserts it is the argmax share (argmax decoding)."""
    p = share(probs, label)
    if abs(p - top_share(probs)) > 1e-9:
        raise SystemExit(f"predicted option {label!r} is not the argmax of {probs}")
    return p


def brier(vec: dict[str, float], gold: Any) -> float:
    """Half the squared error of the renormalised probability vector."""
    tot = sum(max(float(v), 0.0) for v in vec.values())
    p = {k: max(float(v), 0.0) / tot for k, v in vec.items()} if tot > 0 else {k: 1.0 / len(vec) for k in vec}
    assert gold in p, gold
    return 0.5 * (sum(x * x for x in p.values()) - 2 * p[gold] + 1.0)


def binary_conf(p_pred: float) -> float:
    """(K P(predicted) - 1)/(K - 1) with K = 2."""
    return 2.0 * p_pred - 1.0


class Units:
    def __init__(self) -> None:
        self.conf: list[float] = []
        self.prob: list[float] = []
        self.brier: list[float] = []
        self.correct: list[bool] = []
        self.cluster: list[str] = []
        self.extra: dict[str, list] = defaultdict(list)
        self.checks: dict[str, Any] = {}

    def add(self, conf: float, prob: float | None, correct: bool, cluster: str, brier_: float | None = None) -> None:
        self.conf.append(float(conf))
        self.prob.append(float("nan") if prob is None else float(prob))
        self.brier.append(float("nan") if brier_ is None else float(brier_))
        self.correct.append(bool(correct))
        self.cluster.append(cluster)

    def arrays(self) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray | None]:
        _, cl = np.unique(np.array(self.cluster), return_inverse=True)
        out = []
        for v in (self.prob, self.brier):
            a = np.array(v)
            if np.isnan(a).all():
                a = None
            else:
                assert not np.isnan(a).any()
            out.append(a)
        return np.array(self.conf), out[0], np.array(self.correct, dtype=float), cl, out[1]

    def accuracy(self) -> float:
        return sum(self.correct) / len(self.correct)


def check(u: Units, name: str, recorded: float) -> None:
    got = u.accuracy()
    if abs(got - recorded) > 1e-9:
        raise SystemExit(f"{name}: accuracy {got} != recorded {recorded}")
    u.checks["accuracy_matches_scores_json"] = recorded


def label_vector(task: str, pred: dict, answer: dict) -> dict[Any, float]:
    """The answer's probabilities keyed by gold label, for the Brier score."""
    if "probabilities" in pred:
        return pred["probabilities"]
    if task.startswith("sciprompt/"):
        to_gold = prompt(task.replace("/", "_"))["_label_map"]
        return {gold: float(answer["probabilities"].get(option, 0.0)) for option, gold in to_gold.items()}
    if task.startswith("cicc/"):
        labels = list(prompt("cicc_" + task.split("/")[1])["question"]["criteria"])
        return dict(zip(labels, pred["probs"]))
    if task == "lifetox":
        return {"unsafe": float(pred["p_unsafe"]), "safe": 1.0 - float(pred["p_unsafe"])}
    raise KeyError(task)


def recorded_label_prob(pred: dict, label: Any, answer: dict) -> float | None:
    """The recorded prediction's own probability of its label, where predictions.jsonl stores one."""
    if "probabilities" in pred:
        return share(pred["probabilities"], label)
    if "p_label" in pred:
        return float(pred["p_label"]) / sum(max(float(v), 0.0) for v in answer["probabilities"].values())
    if "probs" in pred:
        return max(pred["probs"])
    if "p_unsafe" in pred:
        return float(pred["p_unsafe"]) if label == "unsafe" else 1.0 - float(pred["p_unsafe"])
    return None


def load_choice(task: str) -> Units:
    answers = {}
    for r in responses(task):
        (a,) = r["response"]["answers"].values()
        answers[r["item_id"]] = a
    u = Units()
    agree = n = 0
    for p in predictions(task):
        iid = p["item_id"]
        if task.startswith("cicc/") and iid.startswith("cal"):
            continue
        pred = p["prediction"]
        label = pred.get("label", pred.get("top1"))
        a = answers[iid]
        prob = top_share(a["probabilities"])
        rec = recorded_label_prob(pred, label, a)
        agree += rec is not None and abs(rec - prob) < 1e-6
        n += 1
        u.add(a["confidence"], prob, label == p["gold"], iid, brier(label_vector(task, pred, a), p["gold"]))
        u.extra["dist"].append(label_vector(task, pred, a))
        u.extra["item_id"].append(iid)
        u.extra["pred"].append(label)
    u.checks["recorded_label_prob_equals_top_share"] = agree / n
    if agree / n < 0.999:
        raise SystemExit(f"{task}: the recorded label's probability equals the argmax share on only {agree / n:.4f} of items")
    if "accuracy" in scores(task):
        check(u, task, scores(task)["accuracy"])
    return u


def load_pinose(task: str) -> Units:
    u = Units()
    for p in predictions(task):
        q = float(p["prediction"]["p_true"])
        assert p["prediction"]["label"] == (q > 0.5)
        pp = q if p["prediction"]["label"] else 1 - q
        u.add(binary_conf(pp), pp, p["prediction"]["label"] == bool(p["gold"]), p["item_id"], (q - float(bool(p["gold"]))) ** 2)
    check(u, task, scores(task)["accuracy"])
    return u


def load_fenice(task: str) -> Units:
    thr = scores(task)["threshold_from_val"]
    u = Units()
    gold, pred = [], []
    for p in predictions(task):
        if not p["item_id"].startswith("test"):
            continue
        s = float(p["prediction"]["score"])
        y = int(s > thr)
        u.add(binary_conf(s if y else 1 - s), None, y == int(p["gold"]), p["item_id"])
        gold.append(int(p["gold"]))
        pred.append(y)
    rec = [sum(g == y == c for g, y in zip(gold, pred)) / sum(g == c for g in gold) for c in (0, 1)]
    if abs(sum(rec) / 2 - scores(task)["balanced_accuracy"]) > 1e-9:
        raise SystemExit(f"{task}: balanced accuracy mismatch")
    u.checks["balanced_accuracy_matches_scores_json"] = scores(task)["balanced_accuracy"]
    return u


def load_pairwise(task: str) -> Units:
    u = Units()
    for p in predictions(task):
        q = float(p["prediction"]["p_text1"])
        u.add(2 * abs(q - 0.5), None, int(p["prediction"]["choice"]) == int(p["gold"]), p["item_id"])
    check(u, task, scores(task)["fold0"]["accuracy"]["point"])
    return u


def load_nav(task: str) -> Units:
    ds = task.split("/")[1]
    parts = json.loads((ROOT / "data" / "frozen" / "nav_unknown" / "splits.json").read_text())[ds]
    ans = {}
    for r in responses(task):
        for qid, a in r["response"]["answers"].items():
            ans[(r["item_id"], qid.removeprefix("partition_"))] = a
    u = Units()
    for p in predictions(task):
        for seed, d in p["prediction"].items():
            gold = p["gold"] if p["gold"] in parts[seed]["id"] else "fallback"
            a = ans[(p["item_id"], seed)]
            vec = {lab: float(a["probabilities"].get(lab, 0.0)) for lab in [*parts[seed]["id"], "fallback"]}
            u.add(a["confidence"], pred_prob(a["probabilities"], d["label"]), d["label"] == gold, p["item_id"], brier(vec, gold))
    check(u, task, statistics.mean(v["accuracy"] for v in scores(task)["per_partition"].values()))
    return u


def dress_climatology() -> dict[str, np.ndarray]:
    """Train-split level frequencies per rubric (jev-opt/experiments/dress_new/data/<rubric>/train.jsonl)."""
    out = {}
    for rubric in DRESS_RUBRICS:
        f = np.zeros(DRESS_LEVELS)
        for line in (ROOT / "jev-opt" / "experiments" / "dress_new" / "data" / rubric / "train.jsonl").read_text().splitlines():
            f[round((float(json.loads(line)["label"]) - 1.0) / 0.5)] += 1
        out[rubric] = f / f.sum()
    return out


def rps(p: np.ndarray, g: int) -> float:
    F = np.cumsum(p)[:-1]
    O = (np.arange(len(p) - 1) >= g).astype(float)
    return float(((F - O) ** 2).sum() / (len(p) - 1))


def load_dress(task: str) -> Units:
    ans = {}
    for r in responses(task):
        for rubric, a in r["response"]["answers"].items():
            ans[(r["item_id"], rubric)] = a
    clim = dress_climatology()
    u = Units()
    for p in predictions(task):
        for rubric in DRESS_RUBRICS:
            a = ans[(p["item_id"], rubric)]
            level = str(round((float(p["prediction"][rubric]) - 1.0) / 0.5))
            g = round((float(p["gold"][rubric]) - 1.0) / 0.5)
            vec = np.array([max(float(a["probabilities"].get(str(i), 0.0)), 0.0) for i in range(DRESS_LEVELS)])
            vec = vec / vec.sum()
            u.add(a["confidence"], pred_prob(a["probabilities"], level),
                  float(p["prediction"][rubric]) == float(p["gold"][rubric]), p["item_id"],
                  brier({str(i): vec[i] for i in range(DRESS_LEVELS)}, str(g)))
            u.extra["rps"].append(rps(vec, g))
            u.extra["rps_climatology"].append(rps(clim[rubric], g))
    return u


def load_legalseg(task: str) -> Units:
    ans: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in responses(task):
        for qid, a in r["response"]["answers"].items():
            ans[r["item_id"]][int(qid[1:]) - 1] = {"confidence": a["confidence"], "probabilities": a["probabilities"]}
    labels = list(prompt("legalseg")["labels"])
    u = Units()
    for p in predictions(task):
        pl = p["prediction"]["labels"]
        assert len(pl) == len(p["gold"]) == len(ans[p["item_id"]])
        for i, (y, g) in enumerate(zip(pl, p["gold"])):
            a = ans[p["item_id"]][i]
            vec = {lab: float(a["probabilities"].get(lab, 0.0)) for lab in labels}
            u.add(a["confidence"], pred_prob(a["probabilities"], y), y == g, p["item_id"], brier(vec, g))
    check(u, task, scores(task)["accuracy"])
    return u


def load_nytfb(task: str) -> Units:
    conf: dict[str, dict[int, float]] = defaultdict(dict)
    for r in responses(task):
        for qid, a in r["response"]["answers"].items():
            conf[r["item_id"]][int(qid[1:])] = float(a["confidence"])
    labels = [lab["label"] for lab in prompt("relemb_nytfb_v2")["labels"]]
    u = Units()
    for p in predictions(task):
        pr = p["prediction"]
        for g, j in zip(p["gold"], pr["row_pair"]):
            probs = pr["pair_probs"][j]
            u.add(conf[p["item_id"]][j], max(probs) / sum(probs) if sum(probs) > 0 else 0.0, pr["pair_labels"][j] == g,
                  p["item_id"], brier(dict(zip(labels, probs)), g))
    check(u, task, scores(task)["argmax"]["accuracy"])
    return u


def load_wikiwd(task: str) -> Units:
    from gauntlet.tasks import REGISTRY

    conf: dict[str, dict[str, float]] = defaultdict(dict)
    for r in responses(task):
        for qid, a in r["response"]["answers"].items():
            conf[r["item_id"]][f"{r['key'].split('.')[0]}:{qid}"] = float(a["confidence"])
    t = REGISTRY[task]()
    pair_of = {it.id: t._pairs(it)[1] for it in t.items()}
    u = Units()
    for p in predictions(task):
        iid, pr = p["item_id"], p["prediction"]
        c = conf[iid]
        edge_pair = pair_of[iid]
        assert len(edge_pair) == len(pr["labels"]) == len(p["gold"])
        for e, (y, g) in enumerate(zip(pr["labels"], p["gold"])):
            j = edge_pair[e]
            path = [c[f"r1:p{j}a"], c[f"r1:p{j}b"]]
            wa, wb = pr["round1"][j]
            if wa is not None and wb is not None:
                path.append(c[f"r2:p{j}"])
            u.add(min(path), None, y == g, iid)
    check(u, task, scores(task)["accuracy"])
    return u


def load_ttp(task: str) -> Units:
    u = Units()
    for p in predictions(task):
        pr = p["prediction"]
        ok = pr["ranking"][0] in set(p["gold"])
        u.add(binary_conf(pr["p"][0]), pr["p"][0], ok, p["item_id"], (pr["p"][0] - float(ok)) ** 2)
    check(u, task, scores(task)["p@1"])
    return u


def loader(task: str) -> Callable[[str], Units] | None:
    if task in EXCLUDED:
        return None
    if task in CHOICE_SETS:
        return load_choice
    if task.startswith("pinose/"):
        return load_pinose
    if task.startswith("ttp/"):
        return load_ttp
    if task.startswith("scalar_constructs/") and task.endswith("pairwise"):
        return load_pairwise
    if task.startswith("nav_unknown/"):
        return load_nav
    return {"fenice": load_fenice, "dress_new": load_dress, "legalseg": load_legalseg, "relemb/nytfb": load_nytfb,
            "relemb/wikiwd": load_wikiwd}[task]


def primitive(task: str) -> str:
    if task == "dress_new":
        return "score"
    if task == "relemb/wikiwd":
        return "tournament"
    return "choice" if task in FIELD_SETS else "binary"


def dev_units(slug: str, qid: str) -> tuple[np.ndarray, np.ndarray]:
    """(confidence, correct) of the same question on the released val split (optimization study's zero-shot run)."""
    ids: dict[tuple, list[str]] = defaultdict(list)
    gold = {}
    for line in (ROOT / "jev-opt" / "experiments" / slug.replace("-", "_") / "data" / "val.jsonl").read_text().splitlines():
        r = json.loads(line)
        ids[tuple(v for k, v in r.items() if k not in ("id", "label"))].append(r["id"])
        gold[r["id"]] = r["label"]
    conf, correct, seen = [], [], set()
    for shard in sorted((STUDY / f"study-{slug}-zero").glob("jev_responses*.jsonl.gz")):
        for line in gzip.open(shard, "rt"):
            r = json.loads(line)
            if r["split"] != "val":
                continue
            options = list(r["request"]["questions"][qid]["criteria"])
            a = r["response"]["answers"][qid]
            probs = {o: float(a["probabilities"].get(o, 0.0)) for o in options}
            pred = max(options, key=lambda o: (probs[o], -options.index(o)))  # argmax, ties to the earlier option
            for iid in ids[tuple(r["request"]["state"].values())]:
                if iid in seen:
                    continue
                seen.add(iid)
                conf.append(float(a["confidence"]))
                correct.append(pred == gold[iid])
    return np.array(conf), np.array(correct, dtype=float)


# ---------------------------------------------------------------------------------------------
# Statistics (weighted by bootstrap multiplicities; weights 1 give the point estimate)
# ---------------------------------------------------------------------------------------------

def equal_mass_bins(p: np.ndarray, bins: int = ECE_BINS) -> np.ndarray:
    """Bin index per unit: cuts at the n/bins quantiles, moved to tie boundaries (ties never split)."""
    uniq, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    n = len(p)
    cuts = sorted({int(np.searchsorted(cum, k * n / bins)) for k in range(1, bins)})
    group_bin = np.zeros(len(uniq), dtype=int)
    for c in cuts:
        group_bin[c + 1:] += 1
    return group_bin[inv]


def ece_binned(bins: np.ndarray, p: np.ndarray, correct: np.ndarray, w: np.ndarray) -> float:
    nb = int(bins.max()) + 1
    W = np.bincount(bins, w, nb)
    C = np.bincount(bins, w * p, nb)
    A = np.bincount(bins, w * correct, nb)
    return float(np.abs(A - C).sum() / W.sum())


class SetStats:
    """Precomputed orderings of one set; every statistic takes unit weights w."""

    def __init__(self, conf: np.ndarray, correct: np.ndarray, prob: np.ndarray | None = None, brier_: np.ndarray | None = None):
        self.conf, self.correct, self.prob, self.brier_ = conf, correct, prob, brier_
        uniq, self.g_asc = np.unique(conf, return_inverse=True)  # tie groups, ascending confidence
        self.n_groups = len(uniq)
        if prob is not None:
            self.bin_em = equal_mass_bins(prob)
            self.bin_w = np.minimum((prob * ECE_BINS).astype(int), ECE_BINS - 1)
        self.bin_conf = np.clip((conf * ECE_BINS).astype(int), 0, ECE_BINS - 1)
        self.band = np.searchsorted(np.array(BANDS[1:]), conf, side="right")

    def auroc(self, w: np.ndarray) -> float:
        P = np.bincount(self.g_asc, w * self.correct, self.n_groups)
        N = np.bincount(self.g_asc, w * (1 - self.correct), self.n_groups)
        tp, tn = P.sum(), N.sum()
        if tp == 0 or tn == 0:
            return float("nan")
        below = np.cumsum(N) - N
        return float((P * (below + 0.5 * N)).sum() / (tp * tn))

    def ece(self, w: np.ndarray) -> float:
        """Equal-mass ECE of the predicted option's probability (nan without one)."""
        return float("nan") if self.prob is None else ece_binned(self.bin_em, self.prob, self.correct, w)

    def ece_width(self, w: np.ndarray) -> float:
        return float("nan") if self.prob is None else ece_binned(self.bin_w, self.prob, self.correct, w)

    def ece_conf(self, w: np.ndarray) -> float:
        """Sensitivity only: the equal-width formula applied to the confidence field (not a probability)."""
        return ece_binned(self.bin_conf, np.clip(self.conf, 0, 1), self.correct, w)

    def brier(self, w: np.ndarray) -> float:
        return float("nan") if self.brier_ is None else float((w * self.brier_).sum() / w.sum())

    def floor(self, rng: np.random.Generator) -> dict[str, float]:
        """ECE of a perfectly calibrated model with these probabilities (labels drawn from p)."""
        null = np.array([ece_binned(self.bin_em, self.prob, (rng.random(len(self.prob)) < self.prob).astype(float),
                                    np.ones(len(self.prob))) for _ in range(FLOOR_DRAWS)])
        obs = self.ece(np.ones(len(self.prob)))
        return {"floor_mean": float(null.mean()), "floor_p95": float(np.percentile(null, 95)),
                "p_miscalibrated": float((1 + (null >= obs).sum()) / (FLOOR_DRAWS + 1)), "above_band": bool(obs > np.percentile(null, 95))}

    def bands(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(weighted units, weighted errors) per fixed band."""
        return np.bincount(self.band, w, 4), np.bincount(self.band, w * (1 - self.correct), 4)

    def _groups_desc(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        W = np.bincount(self.g_asc, w, self.n_groups)[::-1]
        C = np.bincount(self.g_asc, w * self.correct, self.n_groups)[::-1]
        keep = W > 0
        return W[keep], C[keep]

    def acc_at(self, w: np.ndarray, coverages: tuple[float, ...]) -> list[float]:
        """Expected kept accuracy at each coverage (random tie-breaking)."""
        W, C = self._groups_desc(w)
        cw, cc = np.cumsum(W), np.cumsum(C)
        out = []
        for c in coverages:
            k = c * cw[-1]
            g = int(np.searchsorted(cw, k - 1e-9 * cw[-1]))
            wb, cb = (cw[g - 1], cc[g - 1]) if g else (0.0, 0.0)
            out.append(float((cb + (k - wb) * C[g] / W[g]) / k))
        return out

    def aurc(self, w: np.ndarray) -> float:
        W, C = self._groups_desc(w)
        a = C / W
        wb = np.concatenate([[0.0], np.cumsum(W)[:-1]])
        cb = np.concatenate([[0.0], np.cumsum(C)[:-1]])
        with np.errstate(divide="ignore", invalid="ignore"):
            log = np.where(wb > 0, np.log((wb + W) / np.where(wb > 0, wb, 1.0)), 0.0)
        kept = a * W + (cb - a * wb) * log
        return float(1.0 - kept.sum() / W.sum())

    def augrc(self, w: np.ndarray) -> float:
        """Integral over coverage of the share of all units that are accepted errors."""
        W, C = self._groups_desc(w)
        E = W - C
        eb = np.concatenate([[0.0], np.cumsum(E)[:-1]])
        return float((eb * W + 0.5 * E * W).sum() / W.sum() ** 2)

    def eaurc(self, w: np.ndarray) -> float:
        r = 1.0 - float((w * self.correct).sum() / w.sum())
        star = 1.0 if r >= 1.0 else r + (1.0 - r) * np.log(1.0 - r) if r > 0 else 0.0
        return self.aurc(w) - star

    def max_coverage(self, w: np.ndarray, target: float) -> float:
        """Oracle: largest coverage whose expected kept accuracy is >= target (0 if none)."""
        W, C = self._groups_desc(w)
        a = C / W
        wb = np.concatenate([[0.0], np.cumsum(W)[:-1]])
        cb = np.concatenate([[0.0], np.cumsum(C)[:-1]])
        best = 0.0
        for Wg, ag, Wb, Cb in zip(W, a, wb, cb):
            if ag >= target:
                t = Wg if Cb + Wg * ag >= target * (Wb + Wg) else None
            else:
                slack = Cb - target * Wb
                t = min(Wg, slack / (target - ag)) if slack >= 0 and (Wb > 0 or slack > 0) else None
            if t is not None and Wb + t > 0:
                best = max(best, (Wb + t) / W.sum())
        return float(best)

    def quartiles(self) -> list[dict[str, Any]]:
        order = self.g_asc
        counts = np.bincount(order, minlength=self.n_groups)
        start = np.cumsum(counts) - counts
        avg_rank = start + (counts + 1) / 2.0
        q = np.minimum(((avg_rank[order] - 0.5) / len(order) * 4).astype(int), 3)
        out = []
        for k in range(4):
            m = q == k
            ok = m.sum() >= MIN_BAND
            out.append({"n": int(m.sum()), "error_rate": float(1 - self.correct[m].mean()) if ok else None,
                        "conf_min": float(self.conf[m].min()) if m.any() else None,
                        "conf_max": float(self.conf[m].max()) if m.any() else None})
        return out


def _desc_groups(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distinct confidences descending, with the kept count and kept correct count at each."""
    order = np.argsort(-conf, kind="stable")
    c, y = conf[order], correct[order]
    ends = np.flatnonzero(np.r_[c[1:] != c[:-1], True])
    return c[ends], ends + 1, np.cumsum(y)[ends]


def threshold_naive(conf: np.ndarray, correct: np.ndarray, target: float) -> float:
    """Lowest threshold whose kept accuracy reaches target; inf if none."""
    tau, n, k = _desc_groups(conf, correct)
    ok = np.flatnonzero(k / n >= target - 1e-12)
    return float(tau[ok[-1]]) if len(ok) else float("inf")


def threshold_ltt(conf: np.ndarray, correct: np.ndarray, target: float, delta: float = DELTA) -> float:
    """Learn-then-Test fixed sequence: accept descending thresholds while the one-sided Clopper-Pearson lower
    bound of the kept accuracy is >= target; the lowest accepted threshold (inf if the first fails)."""
    tau, n, k = _desc_groups(conf, correct)
    lb = np.where(k > 0, beta_dist.ppf(delta, np.maximum(k, 1), n - k + 1), 0.0)
    fail = np.flatnonzero(lb < target)
    last = (fail[0] - 1) if len(fail) else len(tau) - 1
    return float(tau[last]) if last >= 0 else float("inf")


def cross_fitted(conf: np.ndarray, correct: np.ndarray, cl: np.ndarray, target: float, rng: np.random.Generator,
                 chooser: Callable[[np.ndarray, np.ndarray, float], float]) -> dict[str, Any]:
    n_cl = int(cl.max()) + 1
    cov, acc, reach = [], [], []
    for _ in range(HALVINGS):
        perm = rng.permutation(n_cl)
        in_a = np.zeros(n_cl, bool)
        in_a[perm[: n_cl // 2]] = True
        a = in_a[cl]
        kept = correct_kept = 0.0
        for fit, score in ((a, ~a), (~a, a)):
            tau = chooser(conf[fit], correct[fit], target)
            k = score & (conf >= tau)
            kept += k.sum()
            correct_kept += correct[k].sum()
        cov.append(kept / len(conf))
        acc.append(correct_kept / kept if kept else float("nan"))
        reach.append(bool(kept) and correct_kept / kept >= target - 1e-12)
    acc_a = np.array(acc)
    return {"halvings": HALVINGS, "mean_coverage": float(np.mean(cov)),
            "mean_kept_accuracy": float(np.nanmean(acc_a)) if not np.isnan(acc_a).all() else None,
            "share_reaching": float(np.mean(reach)), "coverage_range": [float(min(cov)), float(max(cov))]}


def ci(x: np.ndarray) -> list[float]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if not len(x):
        return [float("nan"), float("nan")]
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def one_sided_p(reps: np.ndarray, null_side: Callable[[np.ndarray], np.ndarray]) -> float:
    reps = reps[~np.isnan(reps)]
    return float((1 + null_side(reps).sum()) / (len(reps) + 1)) if len(reps) else float("nan")


def bh(pvals: dict[str, float], q: float = FDR_Q) -> dict[str, bool]:
    """Benjamini-Hochberg rejections at level q."""
    keys = [k for k, v in pvals.items() if not np.isnan(v)]
    ps = np.array([pvals[k] for k in keys])
    order = np.argsort(ps)
    m = len(ps)
    passed = np.flatnonzero(ps[order] <= q * np.arange(1, m + 1) / m)
    cut = passed[-1] + 1 if len(passed) else 0
    rej = set(np.array(keys)[order[:cut]].tolist()) if cut else set()
    return {k: k in rej for k in pvals}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    keys = list(pvals)
    ps = np.array([pvals[k] for k in keys])
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return dict(zip(keys, adj.tolist()))


def analyse_set(units: Units, ss: np.random.SeedSequence, target: float | None, dev: tuple[np.ndarray, np.ndarray] | None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    conf, prob, correct, cl, brier_ = units.arrays()
    boot_ss, half_ss, floor_ss = ss.spawn(3)
    rng = np.random.default_rng(boot_ss)
    st = SetStats(conf, correct, prob, brier_)
    one = np.ones(len(conf))
    n_cl = int(cl.max()) + 1
    nb, eb = st.bands(one)
    keep_band = nb >= MIN_BAND
    acc = st.acc_at(one, (0.5, 0.8, 1.0))
    res: dict[str, Any] = {
        "n_units": len(conf), "n_clusters": n_cl, "accuracy": float(correct.mean()), "mean_conf": float(conf.mean()),
        "has_probability": prob is not None, "mean_prob": float(prob.mean()) if prob is not None else None,
        "has_brier": brier_ is not None,
        "auroc": st.auroc(one), "ece": st.ece(one), "ece_equal_width": st.ece_width(one),
        "ece_conf_field_sensitivity": st.ece_conf(one), "brier": st.brier(one),
        "bands": [{"band": BAND_NAMES[b], "n": int(nb[b]), "errors": int(eb[b]),
                   "error_rate": float(eb[b] / nb[b]) if keep_band[b] else None} for b in range(4)],
        "top_band_error_rate": float(eb[3] / nb[3]) if keep_band[3] else None,
        "confident_error_share": float(eb[3] / eb.sum()) if eb.sum() else None,
        "top_band_share": float(nb[3] / nb.sum()),
        "quartiles": st.quartiles(),
        "acc_cov50": acc[0], "acc_cov80": acc[1], "acc_cov100": acc[2],
        "gain_cov50": acc[0] - acc[2], "gain_cov80": acc[1] - acc[2],
        "aurc": st.aurc(one), "augrc": st.augrc(one), "eaurc": st.eaurc(one),
        "risk_coverage": {"coverage": list(COVERAGES), "accuracy": st.acc_at(one, COVERAGES)},
        "checks": units.checks,
    }
    if prob is not None:
        res["ece_floor"] = st.floor(np.random.default_rng(floor_ss))
    if units.extra.get("rps"):
        res["rps"] = float(np.mean(units.extra["rps"]))
        res["rps_climatology"] = float(np.mean(units.extra["rps_climatology"]))
        res["rps_skill"] = 1.0 - res["rps"] / res["rps_climatology"]
    if target is not None:
        hrng = np.random.default_rng(half_ss)
        res["target_accuracy"] = target
        res["ltt"] = cross_fitted(conf, correct, cl, target, hrng, threshold_ltt)
        res["naive_cross_fitted"] = cross_fitted(conf, correct, cl, target, hrng, threshold_naive)
        res["max_coverage_at_target"] = st.max_coverage(one, target)
        if dev is not None:
            dconf, dcorrect = dev
            res["dev_fitted"] = {"n_dev": len(dconf), "dev_accuracy": float(dcorrect.mean())}
            for name, chooser in (("ltt", threshold_ltt), ("naive", threshold_naive)):
                tau = chooser(dconf, dcorrect, target)
                k = conf >= tau
                res["dev_fitted"][name] = {"threshold": tau, "coverage": float(k.mean()),
                                           "kept_accuracy": float(correct[k].mean()) if k.any() else None,
                                           "reaches": bool(k.any() and correct[k].mean() >= target - 1e-12)}
    boot: dict[str, list] = defaultdict(list)
    for _ in range(B):
        w = np.bincount(rng.integers(0, n_cl, n_cl), minlength=n_cl)[cl].astype(float)
        boot["auroc"].append(st.auroc(w))
        boot["ece"].append(st.ece(w))
        boot["brier"].append(st.brier(w))
        n_b, e_b = st.bands(w)
        boot["band_units"].append(np.where(keep_band, n_b, 0.0) / w.sum())  # weight 1/n per unit (equal set weight)
        boot["band_errors"].append(np.where(keep_band, e_b, 0.0) / w.sum())
        boot["errors_all"].append(e_b.sum() / w.sum())
        boot["confident_error_share"].append(e_b[3] / e_b.sum() if e_b.sum() else np.nan)
        boot["top_band_error_rate"].append(e_b[3] / n_b[3] if keep_band[3] and n_b[3] else np.nan)
        boot["aurc"].append(st.aurc(w))
        boot["augrc"].append(st.augrc(w))
        boot["eaurc"].append(st.eaurc(w))
        if target is not None:
            boot["max_coverage"].append(st.max_coverage(w, target))
    arr = {k: np.array(v, float) for k, v in boot.items()}
    for k in ("auroc", "ece", "brier", "confident_error_share", "top_band_error_rate", "aurc", "augrc", "eaurc"):
        res[f"{k}_ci"] = ci(arr[k])
    if target is not None:
        res["max_coverage_ci"] = ci(arr["max_coverage"])
    res["p_auroc_gt_half"] = one_sided_p(arr["auroc"], lambda r: r <= 0.5)
    res["p_top_rate_lt_10pct"] = one_sided_p(arr["top_band_error_rate"], lambda r: r >= 0.10) if keep_band[3] else float("nan")
    arr["band_units_point"] = np.where(keep_band, nb, 0.0) / len(conf)
    arr["band_errors_point"] = np.where(keep_band, eb, 0.0) / len(conf)
    arr["errors_all_point"] = np.array(eb.sum() / len(conf))
    return res, arr


# ---------------------------------------------------------------------------------------------
# Group statistics
# ---------------------------------------------------------------------------------------------

def group_stat(names: list[str], sets: dict[str, dict], boots: dict[str, dict[str, np.ndarray]], reps: np.ndarray | None) -> dict[str, Any]:
    """Equal-set-weight group statistics over `names` (with multiplicity): point values (reps None) or one
    within-set cluster replicate per listed set."""
    def val(j: int, s: str, k: str) -> Any:
        if reps is None:
            return boots[s][f"{k}_point"] if f"{k}_point" in boots[s] else sets[s][k]
        return boots[s][k][reps[j]]
    au = [val(j, s, "auroc") for j, s in enumerate(names)]
    ece = [val(j, s, "ece") for j, s in enumerate(names) if sets[s]["has_probability"]]
    bri = [val(j, s, "brier") for j, s in enumerate(names) if sets[s]["has_brier"]]
    augrc = [val(j, s, "augrc") for j, s in enumerate(names)]
    eaurc = [val(j, s, "eaurc") for j, s in enumerate(names)]
    U = sum(val(j, s, "band_units") for j, s in enumerate(names))
    E = sum(val(j, s, "band_errors") for j, s in enumerate(names))
    Eall = sum(float(val(j, s, "errors_all")) for j, s in enumerate(names))
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(U > 0, E / U, np.nan)
    mean = lambda x: float(np.nanmean(x)) if len(x) else float("nan")  # noqa: E731
    return {"auroc": mean(au), "ece": mean(ece), "brier": mean(bri), "augrc": mean(augrc), "eaurc": mean(eaurc),
            "share": float(E[3] / Eall) if Eall else float("nan"), "top_rate": float(rate[3]), "band_rate": rate}


def paper_weighted(names: list[str], sets: dict[str, dict], key: str) -> float:
    by: dict[str, list[float]] = defaultdict(list)
    for s in names:
        v = sets[s].get(key)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            by[sets[s]["paper"]].append(v)
    return float(np.mean([np.mean(v) for v in by.values()])) if by else float("nan")


def fixed_set_boot(names: list[str], sets: dict, boots: dict, rng: np.random.Generator) -> dict[str, np.ndarray]:
    out: dict[str, list] = defaultdict(list)
    for _ in range(B):
        g = group_stat(names, sets, boots, rng.integers(0, B, len(names)))
        for k, v in g.items():
            out[k].append(v)
    return {k: np.array(v, float) for k, v in out.items()}


def paper_boot(names: list[str], sets: dict, boots: dict, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Papers resampled with replacement within the group; their sets kept as they are (point values)."""
    papers = sorted({sets[s]["paper"] for s in names})
    of = {p: [s for s in names if sets[s]["paper"] == p] for p in papers}
    out: dict[str, list] = defaultdict(list)
    for _ in range(B):
        drawn = [s for i in rng.integers(0, len(papers), len(papers)) for s in of[papers[i]]]
        g = group_stat(drawn, sets, boots, None)
        for k, v in g.items():
            out[k].append(v)
    return {k: np.array(v, float) for k, v in out.items()}


def aggregate(names: list[str], sets: dict[str, dict], boots: dict[str, dict[str, np.ndarray]], rng: np.random.Generator) -> dict[str, Any]:
    if not names:
        return {"n_sets": 0}
    papers = sorted({sets[s]["paper"] for s in names})
    pt_ = group_stat(names, sets, boots, None)
    fixed = fixed_set_boot(names, sets, boots, rng)
    general = len(papers) >= MIN_PAPERS
    pb = paper_boot(names, sets, boots, rng) if general else None
    Up = sum(boots[s]["band_units_point"] for s in names)
    out: dict[str, Any] = {
        "n_sets": len(names), "sets": names, "n_papers": len(papers), "papers": papers,
        "inference": "fixed sets + papers" if general else "descriptive (1-2 papers)",
        "accuracy_mean": float(np.mean([sets[s]["accuracy"] for s in names])),
        "auroc_median": float(np.nanmedian([sets[s]["auroc"] for s in names])),
        "ece_n_sets": sum(sets[s]["has_probability"] for s in names), "brier_n_sets": sum(sets[s]["has_brier"] for s in names),
        "band_error_rate": [None if np.isnan(x) else float(x) for x in pt_["band_rate"]],
        "band_error_rate_ci": [ci(fixed["band_rate"][:, b]) for b in range(4)],
        "band_unit_share": [float(Up[b] / Up.sum()) for b in range(4)],
        "paper_weighted": {k: paper_weighted(names, sets, k) for k in ("auroc", "ece", "brier", "augrc", "top_band_error_rate")},
        "_fixed": fixed, "_paper": pb, "_point": pt_,
    }
    for k, name in (("auroc", "auroc_mean"), ("ece", "ece_mean"), ("brier", "brier_mean"), ("augrc", "augrc_mean"),
                    ("eaurc", "eaurc_mean"), ("share", "confident_error_share"), ("top_rate", "top_band_error_rate")):
        out[name] = pt_[k]
        out[f"{name}_ci"] = ci(fixed[k])
        out[f"{name}_ci_papers"] = ci(pb[k]) if pb is not None else None
    if general:
        out["band_error_rate_ci_papers"] = [ci(pb["band_rate"][:, b]) for b in range(4)]
    return out


def difference(a: dict, b: dict) -> dict[str, Any]:
    out = {}
    for k in TEST_STATS:
        d = a["_paper"][k] - b["_paper"][k]
        d = d[~np.isnan(d)]
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())) if len(d) else float("nan")
        df = a["_fixed"][k] - b["_fixed"][k]
        out[k] = {"delta": a["_point"][k] - b["_point"][k], "ci_papers": ci(d), "ci_fixed_sets": ci(df), "p": float(p)}
    return out


# ---------------------------------------------------------------------------------------------
# Human disagreement (HateXplain)
# ---------------------------------------------------------------------------------------------

def hatexplain(units: Units, rng: np.random.Generator) -> dict[str, Any]:
    raw = json.loads((ROOT / "data" / "raw" / "smarter" / "hatexplain" / "Data" / "dataset.json").read_text())
    to_label = {"normal": "Normal", "offensive": "Offensive", "hatespeech": "Hate"}
    labels = list(to_label.values())
    conf, prob, correct, _, _ = units.arrays()
    unan, h_pred, h_maj, tvd, tvd_maj = [], [], [], [], []
    for iid, dist, pred in zip(units.extra["item_id"], units.extra["dist"], units.extra["pred"]):
        votes = [to_label[a["label"]] for a in raw[iid.removeprefix("hatexplain-")]["annotators"]]
        h = {lab: votes.count(lab) / len(votes) for lab in labels}
        tot = sum(max(float(v), 0.0) for v in dist.values())
        p = {lab: max(float(dist.get(lab, 0.0)), 0.0) / tot for lab in labels}
        maj = max(labels, key=lambda lab: h[lab])
        unan.append(max(h.values()) == 1.0)
        h_pred.append(h[pred])
        h_maj.append(h[maj])
        tvd.append(0.5 * sum(abs(p[lab] - h[lab]) for lab in labels))
        tvd_maj.append(0.5 * sum(abs(float(lab == maj) - h[lab]) for lab in labels))
    unan_a = np.array(unan)
    out: dict[str, Any] = {"n": len(unan), "n_unanimous": int(unan_a.sum()), "share_unanimous": float(unan_a.mean())}
    for name, m in (("unanimous", unan_a), ("split", ~unan_a)):
        st = SetStats(conf[m], correct[m], prob[m])
        one = np.ones(int(m.sum()))
        n = int(m.sum())
        reps = [st.auroc(np.bincount(rng.integers(0, n, n), minlength=n).astype(float)) for _ in range(B)]
        out[name] = {"n": n, "accuracy": float(correct[m].mean()), "auroc": st.auroc(one), "auroc_ci": ci(np.array(reps)),
                     "ece": st.ece(one), "mean_conf": float(conf[m].mean())}
    st_u = SetStats(conf, unan_a.astype(float))
    out["auroc_conf_for_unanimity"] = st_u.auroc(np.ones(len(conf)))
    hp = np.array(h_pred)
    bins = equal_mass_bins(prob)
    out["soft_label_ece"] = ece_binned(bins, prob, hp, np.ones(len(prob)))
    out["majority_label_ece"] = ece_binned(bins, prob, correct, np.ones(len(prob)))
    out["dist_ce_jev"] = float(np.mean(tvd))
    out["dist_ce_majority_onehot"] = float(np.mean(tvd_maj))
    out["agreement_with_random_annotator_jev"] = float(hp.mean())
    out["agreement_with_random_annotator_majority"] = float(np.mean(h_maj))
    return out


# ---------------------------------------------------------------------------------------------
# Re-run non-determinism
# ---------------------------------------------------------------------------------------------

def rerun() -> dict[str, Any]:
    specs = [("inli", "relation", "inli"), ("contempo-tbdense", "relation", "contempo/tbdense"),
             ("smarter-implicit-hate", "label", "smarter/implicit_hate")]
    rows = []
    for slug, qid, task in specs:
        first, first_hash = {}, {}
        for r in responses(task):
            first[r["item_id"]] = r["response"]["answers"][qid]
            first_hash[r["item_id"]] = r["request_hash"]
        recorded = {p["item_id"]: p["prediction"]["label"] for p in predictions(task)}
        ids = {}
        test = ROOT / "jev-opt" / "experiments" / slug.replace("-", "_") / "data" / "test.jsonl"
        for line in test.read_text().splitlines():
            r = json.loads(line)
            ids[tuple(v for k, v in r.items() if k not in ("id", "label"))] = r["id"]
        for shard in sorted((STUDY / f"study-{slug}-zero").glob("jev_responses*.jsonl.gz")):
            for line in gzip.open(shard, "rt"):
                r = json.loads(line)
                if r["split"] != "test":
                    continue
                iid = ids[tuple(r["request"]["state"].values())]
                body = {"model": r["request"]["model"], "state": r["request"]["state"], "questions": r["request"]["questions"]}
                canon = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                identical = hashlib.sha256(canon.encode("utf-8")).hexdigest() == first_hash[iid]
                second = r["response"]["answers"][qid]
                rows.append({"task": task, "identical": identical, "conf": float(first[iid]["confidence"]),
                             "flip": recorded[iid] != second["choice"]})
    conf = np.array([r["conf"] for r in rows])
    flip = np.array([r["flip"] for r in rows])
    band = np.searchsorted(np.array(BANDS[1:]), conf, side="right")
    return {
        "n": len(rows), "identical": sum(r["identical"] for r in rows), "flips": int(flip.sum()),
        "flip_rate": float(flip.mean()),
        "bands": [{"band": BAND_NAMES[b], "n": int((band == b).sum()), "flips": int(flip[band == b].sum()),
                   "flip_rate": float(flip[band == b].mean()) if (band == b).sum() >= MIN_BAND else None} for b in range(4)],
        "flips_conf_lt_06": int(flip[conf < 0.6].sum()),
        "share_flips_conf_lt_06": float(flip[conf < 0.6].sum() / flip.sum()),
        "share_requests_conf_lt_06": float((conf < 0.6).mean()),
        "flip_rate_conf_lt_06": float(flip[conf < 0.6].mean()),
        "flip_rate_conf_ge_06": float(flip[conf >= 0.6].mean()),
        "auroc_low_conf_for_flip": SetStats(-conf, flip.astype(float)).auroc(np.ones(len(conf))),
        "by_task": {t: {"n": sum(r["task"] == t for r in rows), "flips": sum(r["flip"] for r in rows if r["task"] == t)}
                    for t in sorted({r["task"] for r in rows})},
    }


# ---------------------------------------------------------------------------------------------

def main() -> int:
    rows = [r for r in pt.rows() if r["paper"] is not None]
    assert len(rows) == 46, len(rows)
    streams = np.random.SeedSequence(SEED).spawn(len(rows) + 2)
    sets: dict[str, dict] = {}
    boots: dict[str, dict[str, np.ndarray]] = {}
    excluded = {}
    human = None
    for r, ss in zip(rows, streams):
        task = r["task"]
        cause = "matched" if r["cat"] == "replaces" else CAUSE.get(task, "unassigned")
        load = loader(task)
        if load is None:
            excluded[task] = {"name": r["name"], "outcome": r["cat"], "cause": cause, "paper": r["key"], "reason": EXCLUDED[task]}
            continue
        target = None
        if task in ACC_PRIMARY or task in P_AT_1:
            target = r["paper"]
        elif task in PAPER_ACCURACY:
            target = pt.PAPER[task]["paper_accuracy"][pt.PAPER[task]["method"]]
        units = load(task)
        dev = dev_units(*DEV[task]) if task in DEV else None
        res, arr = analyse_set(units, ss, target, dev)
        res.update({"name": r["name"], "outcome": r["cat"], "cause": cause, "paper": r["key"], "primitive": primitive(task),
                    "metric": r["metric"], "jev_primary": r["jev"], "paper_primary": r["paper"], "rel": r["rel"],
                    "confidence_kind": "field" if task in FIELD_SETS else "binary 2P(pred)-1",
                    "rq1_comparable": target is not None,
                    "rq1_basis": ("primary accuracy" if task in ACC_PRIMARY else "P@1" if task in P_AT_1
                                  else "paper reports accuracy" if task in PAPER_ACCURACY else None)})
        sets[task], boots[task] = res, arr
        if task == "smarter/hatexplain":
            human = hatexplain(units, np.random.default_rng(streams[-2]))
        print(f"{task:48s} n={res['n_units']:>7} acc={res['accuracy']:.3f} auroc={res['auroc']:.3f} "
              f"ece={res['ece']:.3f} brier={res['brier']:.3f} augrc={res['augrc']:.3f} [{r['cat']}/{cause}]", flush=True)

    # Per-set count claims: Benjamini-Hochberg over the sets where each property is defined.
    for prop, key in (("auroc_gt_half", "p_auroc_gt_half"), ("top_rate_lt_10pct", "p_top_rate_lt_10pct")):
        rej = bh({s: v[key] for s, v in sets.items()})
        for s in sets:
            sets[s][f"bh_{prop}"] = rej[s] if not np.isnan(sets[s][key]) else None
    rej = bh({s: v["ece_floor"]["p_miscalibrated"] for s, v in sets.items() if v["has_probability"]})
    for s in sets:
        sets[s]["bh_miscalibrated"] = rej.get(s)

    grng = np.random.default_rng(streams[-1])
    groups = {
        "by_cause": {g: [s for s in sets if sets[s]["cause"] == g] for g in CAUSE_ORDER},
        "by_outcome": {g: [s for s in sets if sets[s]["outcome"] == g] for g in OUTCOME_ORDER},
        "by_primitive": {g: [s for s in sets if sets[s]["primitive"] == g] for g in PRIMITIVE_ORDER},
        "preregistered": {"conventions_choice": [s for s in sets if sets[s]["cause"] == "conventions" and sets[s]["primitive"] == "choice"]},
    }
    agg = {kind: {g: aggregate(names, sets, boots, grng) for g, names in gs.items()} for kind, gs in groups.items()}
    for kind in agg:
        for g, a in agg[kind].items():
            if a["n_sets"]:
                names = a["sets"]
                a["counts"] = {
                    "bh_auroc_gt_half": sum(bool(sets[s]["bh_auroc_gt_half"]) for s in names),
                    "bh_top_rate_lt_10pct": sum(bool(sets[s]["bh_top_rate_lt_10pct"]) for s in names),
                    "top_band_defined": sum(sets[s]["top_band_error_rate"] is not None for s in names),
                    "bh_miscalibrated": sum(bool(sets[s]["bh_miscalibrated"]) for s in names),
                    "ece_defined": sum(sets[s]["has_probability"] for s in names),
                }
    flat = {**agg["by_cause"], **agg["by_outcome"]}
    tests = {f"{a}_minus_{b}": difference(flat[a], flat[b]) for a, b in TESTS}
    adj = holm({f"{n}:{k}": d[k]["p"] for n, d in tests.items() for k in d})
    for n, d in tests.items():
        for k in d:
            d[k]["p_holm"] = adj[f"{n}:{k}"]
            d[k]["significant_holm"] = bool(adj[f"{n}:{k}"] < 0.05)
    descriptive = {}
    for a, b in (("conventions", "opaque"), ("conventions", "coarse")):
        descriptive[f"{a}_minus_{b}"] = {k: {"delta": flat[a]["_point"][k] - flat[b]["_point"][k],
                                             "ci_fixed_sets": ci(flat[a]["_fixed"][k] - flat[b]["_fixed"][k])} for k in TEST_STATS}
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}  # noqa: E731

    comparable = {s: sets[s] for s in sets if sets[s]["rq1_comparable"]}
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for k in ("rq1", "rq2", "rerun", "sets", "excluded", "human"):
        out.pop(k, None)
    out.update({
        "generated_by": "scripts/confidence_zeroshot.py",
        "definitions_revision": 3,
        "bootstrap": {"B": B, "seed": SEED, "ci": "percentile 95%", "unit": "cluster (item; essay/utterance/document/sentence for multi-unit items)",
                      "group_primary": "fixed sets (within-set cluster replicates)",
                      "group_papers": f"papers resampled within groups of >= {MIN_PAPERS} papers (point values of their sets)",
                      "holm_family": "conventions-matched, conventions-unassigned, below-replaces x {auroc, ece, top_rate, augrc}",
                      "per_set_counts": f"Benjamini-Hochberg q = {FDR_Q}"},
        "bands": list(BAND_NAMES), "min_band_units": MIN_BAND, "ece_bins": ECE_BINS, "ece_estimator": "equal-mass, ties kept together",
        "floor_draws": FLOOR_DRAWS, "halvings": HALVINGS, "ltt_delta": DELTA,
        "sets": sets, "excluded": excluded,
        "rq2": {kind: {g: strip(v) for g, v in a.items()} for kind, a in agg.items()} | {"tests": tests, "descriptive_contrasts": descriptive},
        "rq1": {
            "comparable": sorted(comparable),
            "ltt_reach_majority": sorted(s for s, v in comparable.items() if v["ltt"]["share_reaching"] > 0.5),
            "ltt_reach_never": sorted(s for s, v in comparable.items() if v["ltt"]["share_reaching"] == 0),
            "naive_reach_majority": sorted(s for s, v in comparable.items() if v["naive_cross_fitted"]["share_reaching"] > 0.5),
            "oracle_reach_full": sorted(s for s, v in comparable.items() if v["max_coverage_at_target"] >= 1.0 - 1e-12),
            "oracle_reach_partial": sorted(s for s, v in comparable.items() if 0 < v["max_coverage_at_target"] < 1.0 - 1e-12),
            "oracle_never": sorted(s for s, v in comparable.items() if v["max_coverage_at_target"] == 0),
            "median_gain_cov50": float(np.median([v["gain_cov50"] for v in sets.values()])),
            "median_gain_cov80": float(np.median([v["gain_cov80"] for v in sets.values()])),
            "by_outcome_median_gain_cov80": {g: float(np.median([v["gain_cov80"] for v in sets.values() if v["outcome"] == g]))
                                             for g in OUTCOME_ORDER},
        },
        "human": {"hatexplain": human,
                  "dress": "one instructor per essay; the DREsS paper states inter-annotator agreement cannot be measured (no human ceiling)"},
        "rerun": rerun(),
    })
    OUT.write_text(json.dumps(out, indent=1, allow_nan=True) + "\n")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
