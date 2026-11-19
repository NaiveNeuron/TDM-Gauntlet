"""Run a Program on examples via Jev (architecture §4 semantics).

Routing is hard (argmax / threshold); the label distribution is soft: mass routed to an
un-evaluated child node is spread uniformly over the outcomes reachable under it.
`exhaustive=True` evaluates every child so that numeric parameters can later be
re-decoded offline from the recorded answers (`redecode`, used by `fit`).
Demonstrations (§4.1) are selected once per example from the train `DemoPool` and sent as
state field `examples` to the program's demo nodes.
"""

from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .demos import DemoPool, demo_records
from .ir import DEMOS_FIELD, QUESTION_KINDS, BranchNode, ChoiceNode, Ensemble, OneVsRestNode, Program, Readout, ScoreNode, StageNode, SwitchNode, Target, default_cutoffs
from .jev import JevClient
from .task import Example

DEMOS_NOTE = (
    f"The state also holds `{DEMOS_FIELD}`: labelled examples from the training data that resemble the input "
    "(most similar first), each with its gold `label`. Use them as precedents; decide only for the item given "
    "by the other state fields.\n\n"
)
BIAS_FLOOR = 1e-4  # probability floor before log-prob biases, so a bias can move a 0-probability option
FEATURE_FLOOR = 1e-4  # probability floor of readout features (log-probs / logits)


@dataclass
class ExampleResult:
    id: str
    gold: str
    pred: str | None
    dist: dict[str, float]
    calls: int = 0
    input_tokens: int = 0
    cost: float = 0.0
    latency_s: float = 0.0
    trace: dict[str, Any] = field(default_factory=dict)  # node id -> compact decision record
    derived: dict[str, str] = field(default_factory=dict)  # stage outputs
    error: str | None = None
    external_score: float | None = None  # black-box evaluator score (task.evaluation.command)
    memo: dict[str, Any] = field(default_factory=dict, repr=False)  # raw answers for offline re-decoding
    demos: list[Example] = field(default_factory=list, repr=False)  # train demonstrations shown (§4.1)

    @property
    def correct(self) -> bool:
        return self.pred == self.gold

    @property
    def demo_ids(self) -> list[str]:
        return [d.id for d in self.demos]


class MissingAnswer(KeyError):
    pass


def _fields_preamble(program: Program, fields: list[str]) -> str:
    described = [(f, program.inputs.get(f, "")) for f in fields]
    if not any(desc for _, desc in described):
        return ""
    lines = "\n".join(f"- {f}: {desc}" if desc else f"- {f}" for f, desc in described)
    return f"State fields:\n{lines}\n\n"


def questions_for(program: Program, node_id: str) -> dict[str, dict[str, Any]]:
    """Jev questions for one node; one_vs_rest yields one Noul per option (`<node>__<option>`)."""
    node = program.nodes[node_id]
    pre = _fields_preamble(program, getattr(node, "state", []))
    if node_id in program.demo_nodes():
        pre = DEMOS_NOTE + pre
    if isinstance(node, ChoiceNode):
        crit = {key: opt.criterion for key, opt in node.options.items()}
        return {node_id: {"type": "choice", "instructions": pre + node.instructions, "criteria": crit}}
    if isinstance(node, OneVsRestNode):
        return {
            f"{node_id}__{key}": {"type": "noul", "instructions": f"{pre}{node.instructions}\n\n{opt.criterion}"}
            for key, opt in node.options.items()
        }
    if isinstance(node, ScoreNode):
        return {node_id: {"type": "score", "instructions": pre + node.instructions, "criteria": list(node.levels)}}
    if isinstance(node, BranchNode):
        return {node_id: {"type": "noul", "instructions": pre + node.condition}}
    raise TypeError(f"{node_id} ({node.kind}) issues no Jev question")


def _normalize(p: dict[str, float]) -> dict[str, float]:
    z = sum(max(v, 0.0) for v in p.values())
    if z <= 0:
        return {k: 1.0 / len(p) for k in p}
    return {k: max(v, 0.0) / z for k, v in p.items()}


def apply_log_bias(probs: dict[str, float], bias: dict[str, float] | None) -> dict[str, float]:
    """p_k * exp(b_k), renormalized (probabilities floored at BIAS_FLOOR first)."""
    if not bias:
        return probs
    return _normalize({k: max(p, BIAS_FLOOR) * math.exp(bias.get(k, 0.0)) for k, p in probs.items()})


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def one_vs_rest_decode(node: OneVsRestNode, p_true: dict[str, float]) -> tuple[dict[str, float], str]:
    """Outcome distribution and decided outcome for K Noul answers under the node's aggregation.

    Fitted parameters act in logit space so the distribution agrees with the decision:
    `threshold` rescales so p == t maps to 0.5; `argmax_normalized` subtracts a logit bias.
    """
    agg = node.aggregation
    targets = {key: opt.target(key).label or key for key, opt in node.options.items()}
    if agg.rule == "threshold":
        shift = {k: _logit(agg.thresholds.get(k, 0.5)) for k in p_true}
    else:
        shift = {k: agg.thresholds.get(k, 0.0) for k in p_true}
    score = {k: _logit(p) - shift[k] for k, p in p_true.items()}
    adjusted = {k: 1 / (1 + math.exp(-s)) for k, s in score.items()}
    mass: dict[str, float] = {}
    for key, p in adjusted.items():
        mass[targets[key]] = mass.get(targets[key], 0.0) + p
    if agg.abstain and agg.abstain not in mass:
        mass[agg.abstain] = math.prod(1.0 - p for p in adjusted.values())
    best = max(score, key=lambda k: score[k])
    if agg.rule == "threshold" and score[best] < 0 and agg.abstain:
        return _normalize(mass), agg.abstain
    return _normalize(mass), targets[best]


def view_features(program: Program, nid: str, ans: dict[str, Any]) -> list[float]:
    """Readout features of one question node's answers (§4.2): Choice log-probabilities, Noul
    logits (one_vs_rest, branch), Score level probabilities and their logs; floored at 1e-4."""
    node = program.nodes[nid]
    if isinstance(node, ChoiceNode):
        probs = _normalize({o: float(ans[nid].get("probabilities", {}).get(o, 0.0)) for o in node.options})
        return [math.log(max(probs[o], FEATURE_FLOOR)) for o in node.options]
    if isinstance(node, OneVsRestNode):
        return [_logit(float(ans[f"{nid}__{o}"]["noul"])) for o in node.options]
    if isinstance(node, ScoreNode):
        raw = ans[nid].get("probabilities", {})
        probs = _normalize({str(i): float(raw.get(str(i), 0.0)) for i in range(len(node.levels))})
        levels = [probs[str(i)] for i in range(len(node.levels))]
        return levels + [math.log(max(p, FEATURE_FLOOR)) for p in levels]
    if isinstance(node, BranchNode):
        return [_logit(float(ans[nid]["noul"]))]
    raise TypeError(f"{nid} ({node.kind}) has no answers")


def readout_decode(r: Readout, x: list[float]) -> tuple[dict[str, float], str]:
    """Label distribution and decision of a fitted readout on feature vector `x`."""
    if r.kind == "lr":
        z = [sum(w * v for w, v in zip(row, x)) + b for row, b in zip(r.W, r.b)]
        top = max(z)
        e = [math.exp(v - top) for v in z]
        dist = {c: v / sum(e) for c, v in zip(r.classes, e)}
        return dist, max(r.classes, key=lambda c: dist[c])  # ties: earliest class
    return ridge_decode(sum(w * v for w, v in zip(r.W[0], x)) + r.b[0], list(r.cutoffs or []), r.sigma, r.classes)


def ridge_decode(y: float, cuts: list[float], sigma: float, classes: list[str]) -> tuple[dict[str, float], str]:
    """Ordinal decision of a regression output `y` (class-index units): the band of `y` under
    `cuts`, and a normal(y, sigma) spread over the bands as the label distribution."""
    y = min(max(y, 0.0), len(classes) - 1.0)
    cdf = lambda t: 0.5 * (1.0 + math.erf((t - y) / (sigma * math.sqrt(2.0))))  # noqa: E731
    edges = [0.0] + [cdf(c) for c in cuts] + [1.0]
    dist = _normalize({c: edges[j + 1] - edges[j] for j, c in enumerate(classes)})
    return dist, classes[sum(1 for c in cuts if y >= c)]


class _Run:
    def __init__(self, program: Program, example: Example, client: JevClient | None, exhaustive: bool, memo: dict | None, context: dict[str, Any] | None = None, demos: list[Example] | None = None):
        self.p = program
        self.ex = example
        self.client = client
        self.exhaustive = exhaustive
        self.context = context  # {split, condition} tag for the run's Jev response ledger
        self.memo: dict[str, Any] = memo if memo is not None else {}
        self.demo_nodes = program.demo_nodes()
        self.families: dict[str, list[str]] = {}  # question node -> nodes of its readout family (batched together)
        for n in program.nodes.values():
            readout = getattr(n, "readout", None)
            for v in readout.views if readout is not None else []:
                fam = self.families.setdefault(v, [])
                fam += [w for w in readout.views if w not in fam]
        self.result = ExampleResult(id=example.id, gold=example.label, pred=None, dist={}, demos=list(demos or []))

    def _key(self, node_id: str, state: dict[str, Any]) -> str:
        blob = json.dumps(state, sort_keys=True, ensure_ascii=False)
        return f"{node_id}|{hashlib.sha256(blob.encode()).hexdigest()[:16]}"

    def _state(self, node_id: str, env: dict[str, str]) -> dict[str, Any]:
        node = self.p.nodes[node_id]
        state: dict[str, Any] = {f: env[f] for f in node.state}
        if node_id in self.demo_nodes:
            fields = [f for f in node.state if f in self.p.inputs]
            state[DEMOS_FIELD] = demo_records(self.result.demos, fields, self.p.demos.max_field_chars)
        return state

    def fetch(self, node_id: str, env: dict[str, str]) -> dict[str, Any]:
        state = self._state(node_id, env)
        key = self._key(node_id, state)
        if key in self.memo:
            return self.memo[key]
        if self.client is None:
            raise MissingAnswer(key)
        group = [node_id]
        # batch not-yet-answered questions over the identical state: every question node with
        # shared_state, else the node's readout family (its views are answered together)
        peers = list(self.p.nodes) if self.p.shared_state else self.families.get(node_id, [])
        for other in peers:
            n = self.p.nodes[other]
            if other != node_id and isinstance(n, QUESTION_KINDS) and all(f in env for f in n.state) and self._state(other, env) == state:
                if self._key(other, state) not in self.memo:
                    group.append(other)
        questions: dict[str, Any] = {}
        for nid in group:
            questions.update(questions_for(self.p, nid))
        decision = self.client.decide(state, questions, self.context)
        r = self.result
        r.calls += 1
        r.input_tokens += decision.input_tokens
        r.cost += decision.cost
        r.latency_s += decision.latency_s
        for nid in group:
            self.memo[self._key(nid, state)] = {q: decision.answers[q] for q in questions_for(self.p, nid)}
        return self.memo[key]

    def node(self, nid: str, env: dict[str, str]) -> tuple[dict[str, float], str]:
        node = self.p.nodes[nid]
        if isinstance(node, StageNode):
            _, value = self.node(node.decision, env)
            self.result.derived[node.as_] = value
            return self.node(node.next, {**env, node.as_: value})
        if isinstance(node, SwitchNode):
            value = env[node.field]
            target = node.cases.get(value, node.default)
            if target is None:
                raise ValueError(f"{nid}: no case for {node.field}={value!r}")
            self.result.trace[nid] = {"value": value}
            if target.label is not None:
                return {target.label: 1.0}, target.label
            return self.node(target.node, env)
        readout = getattr(node, "readout", None)
        if readout is not None:
            x = self.features(nid, env)
            if readout.fitted:
                dist, pred = readout_decode(readout, x)
            else:  # unfitted head: the run only collects the answers its fit needs
                dist, pred = {c: 1.0 / len(readout.classes) for c in readout.classes}, readout.classes[0]
            self.result.trace[nid] = {"readout": pred, "probabilities": {k: round(v, 4) for k, v in dist.items()}}
            return dist, pred
        ans = self.fetch(nid, env)
        if isinstance(node, ChoiceNode):
            raw = _normalize({o: float(ans[nid].get("probabilities", {}).get(o, 0.0)) for o in node.options})
            probs = apply_log_bias(raw, node.biases)
            best = max(probs, key=lambda o: probs[o])  # ties: earliest option
            self.result.trace[nid] = {"choice": best, "probabilities": {k: round(v, 4) for k, v in probs.items()}}
            routes = [(probs[o], opt.target(o)) for o, opt in node.options.items()]
            return self._mix(routes, node.options[best].target(best), env)
        if isinstance(node, OneVsRestNode):
            p_true = {o: float(ans[f"{nid}__{o}"]["noul"]) for o in node.options}
            dist, out = one_vs_rest_decode(node, p_true)
            self.result.trace[nid] = {"p_true": {k: round(v, 4) for k, v in p_true.items()}, "outcome": out}
            return dist, out
        if isinstance(node, ScoreNode):
            raw = ans[nid].get("probabilities", {})
            probs = _normalize({str(i): float(raw.get(str(i), 0.0)) for i in range(len(node.levels))})
            expected = sum(int(i) * p for i, p in probs.items())
            cuts = node.cutoffs if node.cutoffs is not None else default_cutoffs(node)
            band = lambda x: sum(1 for c in cuts if x >= c)  # noqa: E731
            self.result.trace[nid] = {"expected_level": round(expected, 4), "band": band(expected)}
            routes = [(p, node.outcomes[band(int(i))]) for i, p in probs.items()]
            chosen = node.outcomes[band(expected)]
            # A band may be reachable only through the expected level (no integer level maps to it):
            # the chosen band is always evaluated, and an exhaustive run records every band's node.
            extra = node.outcomes if self.exhaustive else [chosen]
            routes += [(0.0, t) for t in extra if all(t != r for _, r in routes)]
            return self._mix(routes, chosen, env)
        if isinstance(node, BranchNode):
            p = float(ans[nid]["noul"])
            chosen = node.if_true if p >= node.threshold else node.if_false
            self.result.trace[nid] = {"p_true": round(p, 4), "taken": p >= node.threshold}
            return self._mix([(p, node.if_true), (1.0 - p, node.if_false)], chosen, env)
        raise TypeError(node.kind)

    def features(self, nid: str, env: dict[str, str]) -> list[float]:
        """Concatenated view features of readout node `nid` (fetching unanswered views)."""
        out: list[float] = []
        for view in self.p.nodes[nid].readout.views:
            out += view_features(self.p, view, self.fetch(view, env))
        return out

    def _mix(self, routes: list[tuple[float, Target]], chosen: Target, env: dict[str, str]) -> tuple[dict[str, float], str]:
        dist: dict[str, float] = {}
        pred = chosen.label
        for p, t in routes:
            if t.label is not None:
                dist[t.label] = dist.get(t.label, 0.0) + p
            elif t == chosen or self.exhaustive:
                sub, sub_pred = self.node(t.node, env)
                if t == chosen:
                    pred = sub_pred
                for k, v in sub.items():
                    dist[k] = dist.get(k, 0.0) + p * v
            else:
                outs = self.p.outcomes_under(t.node)
                for k in outs:
                    dist[k] = dist.get(k, 0.0) + p / len(outs)
        assert pred is not None
        return dist, pred

    def run(self) -> ExampleResult:
        r = self.result
        try:
            dist, pred = self.node(self.p.root, dict(self.ex.fields))
            t = self.p.temperature
            scaled = {lab: max(dist.get(lab, 0.0), 1e-12) ** (1.0 / t) for lab in self.p.labels}
            r.dist, r.pred = _normalize(scaled), pred
        except MissingAnswer:
            raise
        except Exception as exc:  # per-example failure → low score + feedback (GEPA adapter contract)
            r.error = f"{type(exc).__name__}: {exc}"[:500]
            r.dist = {lab: 1.0 / len(self.p.labels) for lab in self.p.labels}
        r.memo = self.memo
        return r


def select_demos(program: Program, examples: list[Example], pool: DemoPool | None) -> list[list[Example]]:
    """Per-example demonstrations for `program` (empty lists when it has none)."""
    if program.demos is None:
        return [[] for _ in examples]
    if pool is None:
        raise ValueError("program has demos but no demo pool was given")
    pool.prefetch(program.demos, examples)
    return [pool.select(program.demos, ex) for ex in examples]


def run_examples(program: Program, examples: list[Example], client: JevClient, exhaustive: bool = False, context: dict[str, Any] | None = None, pool: DemoPool | None = None) -> list[ExampleResult]:
    demos = select_demos(program, examples, pool)

    def one(i: int) -> ExampleResult:
        return _Run(program, examples[i], client, exhaustive, None, context, demos[i]).run()

    if len(examples) <= 1:
        return [one(i) for i in range(len(examples))]
    with ThreadPoolExecutor(max_workers=min(client.concurrency, len(examples))) as pool_:
        return list(pool_.map(one, range(len(examples))))


def redecode(program: Program, example: Example, memo: dict[str, Any], demos: list[Example] | None = None) -> ExampleResult:
    """Decode offline from recorded answers (raises MissingAnswer if the path needs a new call).

    `demos`: the demonstrations of the recorded run (numeric fitting never changes them)."""
    return _Run(program, example, None, False, dict(memo), None, demos).run()


def combine(labels: list[str], dists: list[dict[str, float]], label_bias: dict[str, float] | None, temperature: float) -> tuple[dict[str, float], str]:
    """Ensemble decode: mean distribution, log-prob biases, argmax; temperature scales the output only."""
    mean = {lab: sum(d.get(lab, 0.0) for d in dists) / len(dists) for lab in labels}
    biased = apply_log_bias(_normalize(mean), label_bias)
    pred = max(labels, key=lambda lab: biased[lab])  # ties: earliest label
    scaled = _normalize({lab: max(biased[lab], 1e-12) ** (1.0 / temperature) for lab in labels})
    return scaled, pred


def ordinal_vote(labels: list[str], preds: list[str], cutoffs: list[float]) -> str:
    """Mean predicted label index, banded by `cutoffs` (ordinal ensembles)."""
    value = sum(labels.index(p) for p in preds) / len(preds)
    return labels[sum(1 for c in cutoffs if value >= c)]


def run_ensemble(ensemble: Ensemble, examples: list[Example], client: JevClient, context: dict[str, Any] | None = None, pool: DemoPool | None = None) -> list[ExampleResult]:
    per_member = [run_examples(m, examples, client, context=context, pool=pool) for m in ensemble.members]
    out: list[ExampleResult] = []
    for i, ex in enumerate(examples):
        rows = [results[i] for results in per_member]
        ok = [r for r in rows if r.error is None]
        r = ExampleResult(id=ex.id, gold=ex.label, pred=None, dist={lab: 1.0 / len(ensemble.labels) for lab in ensemble.labels})
        r.calls = sum(x.calls for x in rows)
        r.input_tokens = sum(x.input_tokens for x in rows)
        r.cost = sum(x.cost for x in rows)
        r.latency_s = sum(x.latency_s for x in rows)
        r.trace = {"members": [x.pred for x in rows]}
        r.demos = list({d.id: d for x in rows for d in x.demos}.values())
        if ok:
            r.dist, r.pred = combine(ensemble.labels, [x.dist for x in ok], ensemble.label_bias, ensemble.temperature)
            if ensemble.cutoffs is not None:
                r.pred = ordinal_vote(ensemble.labels, [x.pred for x in ok], ensemble.cutoffs)
        else:
            r.error = rows[0].error
        out.append(r)
    return out


def run_decider(decider: Program | Ensemble, examples: list[Example], client: JevClient, context: dict[str, Any] | None = None, pool: DemoPool | None = None) -> list[ExampleResult]:
    if isinstance(decider, Ensemble):
        return run_ensemble(decider, examples, client, context, pool)
    return run_examples(decider, examples, client, context=context, pool=pool)
