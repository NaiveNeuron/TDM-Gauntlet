"""PRP-Graph (Luo et al., ACL 2024) on TREC-COVID, with Jev as the pairwise judge.

Paper: https://aclanthology.org/2024.acl-long.313/ ; code: https://github.com/Memelank/PRP-Graph
(commit 968e45ee8178f5d3ac44a2b8f11f091af69206d7). Jev replaces Flan-T5 inside the scoring
PRP unit (Eq. 1): for a pair (d_i, d_j) both orders u(q, d_i, d_j) and u(q, d_j, d_i) are asked,
and the probability of "Passage A" in each is the edge score. Everything else is the released
code, ported line by line:

* ranking-graph construction = `rankers/swiss_system.py` (setting "standard", BM25 order,
  S^0 = 1 - 0.01 i, nearest not-yet-compared pairing, Eq. 2 updates, odd lists get a bye);
* aggregation = `rankers/pagerank.py` (setting "standard"): edge source->target weighted by the
  target's comparison score, `networkx.pagerank` (v3.1: alpha 0.85, tol 1e-6, max_iter 100,
  uniform start) with the BM25 scores as personalization vector;
* baseline computed from the same Jev answers: graph construction only ("swiss", README);
* paper-protocol BM25 interpolation (`interpolate.py`): min-max normalisation, alpha chosen by
  10-fold cross-validation over the TEST queries. It tunes on test labels, so it is reported
  only as a secondary number comparable with Table 2, never as the primary metric.

PRP-Graph chooses its comparisons adaptively (pairing depends on earlier answers), so
`next_requests()` replays Algorithm 1 on the answers so far and asks Jev only for the next
round's pairings; the runner loops until every query has completed R=40 rounds.

Rankings are ordered as trec_eval orders a run file (score descending, ties by docid
descending), matching `pyserini.eval.trec_eval -c -m ndcg_cut.10` used by the paper.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..metrics import mean, ndcg_at_k
from ..task import FROZEN, REPO, Item, Request, Task, read_jsonl

DATA = FROZEN / "prp_graph" / "trec_covid"
BEIR_CORPUS = REPO / "data" / "raw" / "prp_graph" / "trec-covid" / "corpus.jsonl"
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip"

ROUNDS = (10, 20, 40)  # PRP-Graph-10/20/40 (Section 4.1, Table 2)
PRIMARY_ROUNDS = 40
DAMPING = 0.85  # Section 4.1
TOL = 1e-6  # Section 4.1 / networkx default
MAX_ITER = 100  # networkx default
NDCG_K = 10


def _format_content(title: str, text: str) -> str:
    """PRP-Graph retriever/pyserini_retriever.py passage formatting."""
    return " ".join(("Title: " + title + " " + "Content: " + text).split())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_passages(manifest: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Rebuild the frozen passages from BEIR corpus.jsonl and verify them against the manifest."""
    if not BEIR_CORPUS.exists():
        raise FileNotFoundError(
            f"{BEIR_CORPUS} missing: download {BEIR_URL} (SHA-256 in data/manifests/prp_graph.json) "
            "and unzip it into data/raw/prp_graph/"
        )
    passages: dict[str, str] = {}
    with BEIR_CORPUS.open() as fh:
        for line in fh:
            row = json.loads(line)
            spec = manifest.get(row["_id"])
            if spec is None:
                continue
            content = _format_content(row["title"], row["text"])
            passage = content[: spec["passage_chars"]]
            if _sha256(content) != spec["content_sha256"] or _sha256(passage) != spec["passage_sha256"]:
                raise ValueError(f"passage {row['_id']} does not match the frozen SHA-256")
            passages[row["_id"]] = passage
    missing = sorted(set(manifest) - set(passages))
    if missing:
        raise ValueError(f"{len(missing)} frozen passages not found in {BEIR_CORPUS}: {missing[:5]}")
    return passages


def trec_order(run: Sequence[Sequence[Any]]) -> list[str]:
    """Doc order trec_eval uses for a run: score descending, ties broken by docid descending."""
    by_docid = sorted(run, key=lambda x: x[0], reverse=True)
    return [d for d, _ in sorted(by_docid, key=lambda x: x[1], reverse=True)]


class _Player:
    """`Player` of rankers/swiss_system.py."""

    __slots__ = ("docid", "score", "round", "compare_record", "compare_score_record")

    def __init__(self, docid: str, weight: float) -> None:
        self.docid = docid
        self.score = weight
        self.round = 0
        self.compare_record: list[str] = []
        self.compare_score_record: list[float] = []

    def add_score(self, score: Sequence[float], player: _Player) -> None:
        self.compare_score_record.append(score[0])
        player.compare_score_record.append(score[1])
        tmp = self.score
        self.score += score[0] * player.score / player.round
        player.score += score[1] * tmp / self.round

    def add_compare_record(self, compare_docid: str) -> None:
        self.compare_record.append(compare_docid)
        self.round += 1


class NeedPairs(Exception):
    """Raised by a `before_round` hook when the round's comparisons lack Jev answers."""

    def __init__(self, pairs: list[tuple[str, str]]) -> None:
        super().__init__(f"{len(pairs)} pairs need answers")
        self.pairs = pairs


def swiss_system(
    docids: Sequence[str],
    prp_unit: Callable[[str, str], Sequence[float]],
    rounds: Sequence[int],
    before_round: Callable[[list[tuple[str, str]]], None] | None = None,
) -> dict[int, dict[str, Any]]:
    """Ranking graph construction (Algorithm 1) as released, setting "standard".

    `prp_unit(a, b)` returns [s_{b->a}, s_{a->b}] = [P("Passage A" | u(q,a,b)), P("Passage A" | u(q,b,a))].
    `before_round(pairs)` sees each round's (a, b) pairings before any is scored, so a
    caller can fetch exactly the comparisons the algorithm asks for, round by round.
    Returns, for each requested round R, the ranking with scores S^R and the per-document
    comparison records (the graph G) after R rounds.
    """
    snapshots: dict[int, dict[str, Any]] = {}
    if len(docids) == 1:
        for r in rounds:
            snapshots[r] = {"ranking": [(docids[0], 1)], "records": [(docids[0], [], [])], "n_compared_pairs": 0}
        return snapshots
    players = [_Player(doc, 1 - i * 0.01) for i, doc in enumerate(docids)]
    bye = None
    if len(players) % 2 != 0:
        bye = players[-1]
        players = players[: len(players) - 1]
    n_pairs = 0

    def snapshot(r: int) -> None:
        order = players + ([bye] if bye else [])
        snapshots[r] = {
            "ranking": [(p.docid, p.score) for p in order],
            "records": [(p.docid, list(p.compare_record), list(p.compare_score_record)) for p in players]
            + ([(bye.docid, [], [])] if bye else []),
            "n_compared_pairs": n_pairs,
        }

    group_1 = players[0::2]
    group_2 = players[1::2]
    if before_round:
        before_round([(a.docid, b.docid) for a, b in zip(group_1, group_2)])
    for a, b in zip(group_1, group_2):
        a.add_compare_record(b.docid)
        b.add_compare_record(a.docid)
        a.add_score(prp_unit(a.docid, b.docid), b)
        n_pairs += 1
    players = sorted(players, key=lambda x: x.score, reverse=True)
    if 1 in rounds:
        snapshot(1)

    for r in range(2, max(rounds) + 1):
        group_1, group_2 = [], []
        for i, player_1 in enumerate(players):
            if player_1.round == r:
                continue
            for player_2 in players[i + 1 :]:
                if player_2.docid in player_1.compare_record or player_2.round == r:
                    continue
                group_1.append(player_1)
                group_2.append(player_2)
                player_1.add_compare_record(player_2.docid)
                player_2.add_compare_record(player_1.docid)
                break
        if before_round:
            before_round([(a.docid, b.docid) for a, b in zip(group_1, group_2)])
        for a, b in zip(group_1, group_2):
            a.add_score(prp_unit(a.docid, b.docid), b)
            n_pairs += 1
        players = sorted(players, key=lambda x: x.score, reverse=True)
        if r in rounds:
            snapshot(r)
    return snapshots


def pagerank_aggregate(
    records: Sequence[tuple[str, Sequence[str], Sequence[float]]], bm25: Sequence[tuple[str, float]]
) -> list[tuple[str, float]]:
    """Ranking graph aggregation: rankers/pagerank.py ("standard") + networkx 3.1 `_pagerank_scipy`."""
    nodes: dict[str, int] = {}
    edges: dict[tuple[str, str], float] = {}
    for target, compare_docids, scores in records:
        if len(compare_docids) != len(scores):
            raise ValueError(f"comparison record of {target} is incomplete")
        for source, weight in zip(compare_docids, scores):
            nodes.setdefault(source, len(nodes))
            nodes.setdefault(target, len(nodes))
            edges[(source, target)] = weight
    for docid, _ in bm25:
        nodes.setdefault(docid, len(nodes))
    nodelist = list(nodes)
    n = len(nodelist)
    adj = np.zeros((n, n))
    for (source, target), weight in edges.items():
        adj[nodes[source], nodes[target]] = weight
    out_weight = adj.sum(axis=1)
    inv = np.zeros(n)
    inv[out_weight != 0] = 1.0 / out_weight[out_weight != 0]
    trans = inv[:, None] * adj
    personalization = dict(bm25)
    p = np.array([personalization.get(v, 0) for v in nodelist], dtype=float)
    p /= p.sum()
    dangling = np.where(out_weight == 0)[0]
    x = np.repeat(1.0 / n, n)
    for _ in range(MAX_ITER):
        xlast = x
        x = DAMPING * (x @ trans + sum(x[dangling]) * p) + (1 - DAMPING) * p
        if np.absolute(x - xlast).sum() < n * TOL:
            ranked = sorted(zip(nodelist, map(float, x)), key=lambda kv: kv[1], reverse=True)
            return ranked
    raise RuntimeError(f"PageRank did not converge in {MAX_ITER} iterations")


def _normalizer(result: dict[str, list[list[Any]]]) -> None:
    """interpolate.py `normalizer` (per-query min-max, same initial max/min values)."""
    for query in result:
        hi, lo = 0.0, 1000000.0
        if len(result[query]) == 1:
            continue
        for _, score in result[query]:
            hi = max(hi, score)
            lo = min(lo, score)
        for row in result[query]:
            row[1] = (row[1] - lo) / (hi - lo)


def paper_interpolation(items: Sequence[Item], runs: dict[str, Sequence[Sequence[Any]]]) -> dict[str, Any]:
    """interpolate.py: BM25 interpolation with alpha picked by 10-fold CV over the test queries.

    `runs` maps query id -> PRP run [(docid, score)], in the order of the paper's run files
    (items order = BM25 retrieval order). nDCG for the alpha search is trec_eval -c output
    (mean over all qrels topics, printed with 4 decimals); first maximum wins.
    """
    bm = {it.id: [[c["docid"], c["bm25_score"]] for c in it.fields["candidates"]] for it in items}
    prp = {qid: [[d, s] for d, s in runs[qid]] for qid in runs}
    gold = {it.id: it.gold for it in items}
    _normalizer(bm)
    _normalizer(prp)
    keys = list(prp)
    random.Random(42).shuffle(keys)
    part = len(keys) // 10
    folds = [keys[i * part : (i + 1) * part] for i in range(9)] + [keys[9 * part :]]
    alphas_grid = [x * 0.1 for x in range(11)]

    def ensemble(qids: Sequence[str], alpha: float) -> dict[str, list[list[Any]]]:
        out = {}
        for q in qids:
            bm_scores = dict(bm[q])
            out[q] = [[d, s * alpha + bm_scores[d] * (1 - alpha)] for d, s in prp[q] if d in bm_scores]
        return out

    def ndcgs(result: dict[str, list[list[Any]]]) -> dict[str, float]:
        return {q: ndcg_at_k(trec_order(rows), gold[q], NDCG_K) for q, rows in result.items()}

    chosen = []
    final: dict[str, float] = {}
    for fold in folds:
        held_out = set(fold)
        train = [q for q in prp if q not in held_out]
        grid = [float(f"{sum(ndcgs(ensemble(train, a)).values()) / len(items):.4f}") for a in alphas_grid]
        alpha = 0.1 * grid.index(max(grid))
        chosen.append(round(alpha, 1))
        final.update(ndcgs(ensemble(fold, alpha)))
    return {"ndcg@10": mean([final[it.id] for it in items]), "alphas_per_fold": chosen}


class PRPGraphTRECCOVID(Task):
    name = "prp_graph/trec_covid"
    prompt_file = "prp_graph_trec_covid.json"

    def items(self) -> list[Item]:
        manifest = {row["docid"]: row for row in read_jsonl(DATA / "passages.jsonl")}
        passages = _load_passages(manifest)
        items = []
        for row in read_jsonl(DATA / "queries.jsonl"):
            candidates = [
                {
                    "docid": c["docid"],
                    "rank": c["rank"],
                    "bm25_score": c["bm25_score"],
                    "passage": passages[c["docid"]],
                    "truncated": manifest[c["docid"]]["truncated"],
                }
                for c in row["candidates"]
            ]
            items.append(
                Item(
                    id=row["qid"],
                    gold=row["qrels"],
                    fields={
                        "query": row["query"],
                        "query_input": row["query_input"],
                        "candidates": candidates,
                        "excluded_non_english": row["excluded_non_english"],
                    },
                )
            )
        return items

    def _pair_requests(self, item: Item, pairs: Sequence[tuple[str, str]]) -> list[Request]:
        """Both orders of each (a, b) comparison: u(q,a,b) and u(q,b,a)."""
        prompt = self.prompt()
        keys = prompt["state_keys"]
        questions = {prompt["question_id"]: prompt["question"]}
        query = item.fields["query_input"]
        passage = {c["docid"]: c["passage"] for c in item.fields["candidates"]}
        return [
            Request(
                item_id=item.id,
                key=f"{first}|{second}",
                state={keys["query"]: query, keys["passage_a"]: passage[first], keys["passage_b"]: passage[second]},
                questions=questions,
            )
            for a, b in pairs
            for first, second in ((a, b), (b, a))
        ]

    def _prob_a(self, answers: dict[str, dict[str, Any]]) -> dict[tuple[str, str], float]:
        prompt = self.prompt()
        question_id = prompt["question_id"]
        option_a = list(prompt["question"]["criteria"])[0]  # "Passage A"
        return {
            tuple(key.split("|")): float(answer[question_id]["probabilities"][option_a])
            for key, answer in answers.items()
        }

    def _swiss(self, item: Item, prob_a: dict[tuple[str, str], float], check: bool) -> dict[int, dict[str, Any]]:
        def before_round(pairs: list[tuple[str, str]]) -> None:
            missing = [(a, b) for a, b in pairs if (a, b) not in prob_a or (b, a) not in prob_a]
            if missing:
                raise NeedPairs(missing)

        def prp_unit(a: str, b: str) -> list[float]:
            return [prob_a[(a, b)], prob_a[(b, a)]]

        docids = [c["docid"] for c in item.fields["candidates"]]
        return swiss_system(docids, prp_unit, ROUNDS, before_round if check else None)

    def next_requests(self, item: Item, answers: dict[str, dict[str, Any]]) -> list[Request]:
        """Replay the Swiss system on the answers so far; request the first round that lacks answers."""
        try:
            self._swiss(item, self._prob_a(answers), check=True)
        except NeedPairs as need:
            return self._pair_requests(item, need.pairs)
        return []

    def predict(self, item: Item, answers: dict[str, dict[str, Any]]) -> Any:
        prob_a = self._prob_a(answers)
        snapshots = self._swiss(item, prob_a, check=True)
        bm25 = [(c["docid"], c["bm25_score"]) for c in item.fields["candidates"]]
        runs: dict[str, list[tuple[str, float]]] = {}
        for r in ROUNDS:
            runs[f"prp_graph_R{r}"] = pagerank_aggregate(snapshots[r]["records"], bm25)
            runs[f"swiss_R{r}"] = snapshots[r]["ranking"]
        compared = sorted({tuple(sorted(pair)) for pair in prob_a})
        return {
            "runs": {name: [[d, float(s)] for d, s in run] for name, run in runs.items()},
            # P("Passage A") for both orders of every compared pair: [a, b, P(a first), P(b first)].
            "compared": [[a, b, round(prob_a[(a, b)], 6), round(prob_a[(b, a)], 6)] for a, b in compared],
            "n_compared_pairs": {f"R{r}": snapshots[r]["n_compared_pairs"] for r in ROUNDS},
        }

    def score(self, items: list[Item], predictions: dict[str, Any]) -> dict[str, Any]:
        run_names = [f"prp_graph_R{r}" for r in ROUNDS] + [f"swiss_R{r}" for r in ROUNDS]
        per_run = {
            name: {it.id: ndcg_at_k(trec_order(predictions[it.id]["runs"][name]), it.gold, NDCG_K) for it in items}
            for name in run_names
        }
        bm25 = {
            it.id: ndcg_at_k(trec_order([(c["docid"], c["bm25_score"]) for c in it.fields["candidates"]]), it.gold, NDCG_K)
            for it in items
        }
        oracle = {
            it.id: ndcg_at_k(
                sorted((c["docid"] for c in it.fields["candidates"]), key=lambda d: -it.gold.get(d, 0)), it.gold, NDCG_K
            )
            for it in items
        }

        # Pairwise audit over the pairs PRP-Graph compared (judged, different grades).
        both_orders = [0.0, 0]
        single_order = [0.0, 0]
        consistent = 0
        n_pairs = 0
        for it in items:
            for a, b, p_ab, p_ba in predictions[it.id]["compared"]:
                n_pairs += 1
                consistent += (p_ab > 0.5) != (p_ba > 0.5) and p_ab != 0.5 and p_ba != 0.5
                rel_a, rel_b = it.gold.get(a), it.gold.get(b)
                if rel_a is None or rel_b is None or rel_a == rel_b:
                    continue
                a_better = rel_a > rel_b
                pref = (p_ab + 1 - p_ba) / 2
                both_orders[0] += 0.5 if pref == 0.5 else float((pref > 0.5) == a_better)
                both_orders[1] += 1
                for p, first_better in ((p_ab, a_better), (p_ba, not a_better)):
                    single_order[0] += 0.5 if p == 0.5 else float((p > 0.5) == first_better)
                    single_order[1] += 1

        primary = f"prp_graph_R{PRIMARY_ROUNDS}"
        passages = {c["docid"]: c["truncated"] for it in items for c in it.fields["candidates"]}
        return {
            "n": len(items),
            "primary": f"ndcg@10 of PRP-Graph-{PRIMARY_ROUNDS} without BM25 interpolation",
            "ndcg@10": mean(list(per_run[primary].values())),
            "ndcg@10_by_method": {name: mean(list(v.values())) for name, v in per_run.items()},
            "bm25_ndcg@10": mean(list(bm25.values())),
            "oracle_candidates_ndcg@10": mean(list(oracle.values())),
            "paper_protocol_bm25_interpolated_cv_on_test": {
                "note": "interpolate.py: alpha tuned by 10-fold CV over TEST queries; comparable with the "
                "paper's Table 2 only, not a frozen-model metric",
                **{
                    name: paper_interpolation(items, {it.id: predictions[it.id]["runs"][name] for it in items})
                    for name in [f"prp_graph_R{r}" for r in ROUNDS]
                },
            },
            "pairwise": {
                "compared_pairs": n_pairs,
                "judged_pairs_with_different_grades": both_orders[1],
                "accuracy_both_orders_averaged": both_orders[0] / both_orders[1] if both_orders[1] else float("nan"),
                "accuracy_single_order": single_order[0] / single_order[1] if single_order[1] else float("nan"),
                "order_consistency_rate": consistent / n_pairs if n_pairs else float("nan"),
            },
            "mean_compared_pairs_per_query": {
                f"R{r}": mean([predictions[it.id]["n_compared_pairs"][f"R{r}"] for it in items]) for r in ROUNDS
            },
            "per_query_ndcg@10": {it.id: per_run[primary][it.id] for it in items},
            "candidates": {
                "per_query_mean": mean([len(it.fields["candidates"]) for it in items]),
                "non_english_excluded_pairs": sum(len(it.fields["excluded_non_english"]) for it in items),
            },
            "truncation": {
                "rule": "paper's: passages to 128 and queries to 32 Flan-T5 tokens (original-text prefix)",
                "passages": len(passages),
                "passages_truncated": sum(passages.values()),
                "queries_truncated": sum(it.fields["query_input"] != it.fields["query"] for it in items),
            },
        }


TASKS: dict[str, Callable[[], Task]] = {"prp_graph/trec_covid": PRPGraphTRECCOVID}
