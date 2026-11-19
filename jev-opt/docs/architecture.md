# jev-opt architecture

jev-opt optimizes **Jev decision programs** (TypeSafe System One, `typesafe/jev-1.13` via
OpenRouter's Decisions API) against labelled development data. It is a thin specialization:
OMP owns every optimizer LLM call, GEPA owns the search, and jev-opt owns only what is
Jev-specific.

Upstream versions this design is pinned to (verified 2026-09-26):

| Upstream | Version | Evidence |
|---|---|---|
| Oh My Pi (`omp`) | 18.3.2 | `omp --version` |
| `@oh-my-pi/pi-ai` (host-bundled) | 18.3.x | imported from an extension inside the omp binary; `completeSimple` probe returned text + usage |
| `gepa` | 0.1.4 (PyPI, 2026-07-15), zero required deps | probe: `gepa.optimize` with a plain callable `reflection_lm`; `litellm`/`openai` never imported |
| Jev | `typesafe/jev-1.13` (snapshot `…-20260917`) | Decisions API responses |

## 1. Ownership

| Capability | Owner | How |
|---|---|---|
| LLM credentials | OMP | `ctx.modelRegistry.resolver(model, sessionId)` (OAuth refresh, key rotation, auth retry) |
| LLM model resolution | OMP | `ctx.models.resolve(spec)` for `@slow`, `@smol`, roles, `provider/id`, bare ids; `ctx.models.current()` for `current` |
| LLM inference | OMP | `@oh-my-pi/pi-ai` `completeSimple(model, context, {apiKey, reasoning, maxTokens, signal})` |
| Reasoning levels | OMP | `reasoning: Effort` / `disableReasoning`; jev-opt only forwards the configured value |
| Reflective optimization | GEPA | `gepa.optimize(..., reflection_lm=OMPReflectionLM)` and GEPA's `InstructionProposalSignature` |
| Pareto candidate search | GEPA | `candidate_selection_strategy="pareto"`, `frontier_type="hybrid"` with `objective_scores` |
| Candidate scheduling / sampling | GEPA | `EpochShuffledBatchSampler`, `module_selector="round_robin"`, `use_merge` |
| Stopping | GEPA | `max_metric_calls`, `max_reflection_cost`, `run_dir` file stopper (`gepa.stop`) |
| Generic persistence / resume | GEPA | `run_dir/gepa_state.bin`, `run_log.json`, `candidates.json` |
| Experiment callbacks | GEPA | `callbacks=[...]` → forwarded to OMP as progress notifications |
| Jev program representation | jev-opt | Pydantic IR (`jev_opt.ir`), YAML/JSON |
| Jev transformations | jev-opt | `jev_opt.transforms` (deterministic) + structural reflection prompt |
| Jev API execution | jev-opt | `jev_opt.jev` (Decisions API client, retries, pacing, usage, cache) |
| Threshold / aggregation fitting | jev-opt | `jev_opt.fit` (numerical; scipy/numpy, no LLM) |
| Task metrics | jev-opt (sklearn) | `jev_opt.metrics` wraps scikit-learn; ECE is the only hand-written metric |
| Task evaluation | jev-opt / GEPA adapter | `jev_opt.adapter.JevAdapter` implements GEPA's `GEPAAdapter` protocol |
| Progressive validation | jev-opt via GEPA protocol | `ProgressivePolicy` implements GEPA's `EvaluationPolicy` protocol (0.1.4 ships only full eval) |
| Test-set isolation | jev-opt | worker never loads the test split except in `evaluate_final`; consumption ledger |
| Experiment reporting | jev-opt | `runs/<id>/report.md`, representatives, `environment.json` |
| UI / commands / worker lifecycle | jev-opt (OMP extension) | `/jev-optimize`, `/jev-status`, … registered with `pi.registerCommand` |

Nothing in jev-opt implements population management, Pareto fronts, mutation scheduling,
merge, sampling or stopping for the GEPA conditions; the one exception is the batched population
search of `jev-population-opt` (§5.1, deviation §12.19). `jev-tailored` (§5.3) plugs its own
minibatch sampler, acceptance criterion and component selector into GEPA's hooks (deviation §12.23).
Nothing in jev-opt talks to an LLM provider.

## 2. Process topology

```
omp (Bun)                                         python (uv run worker.py)
┌──────────────────────────────┐   JSONL/RPC     ┌───────────────────────────────┐
│ jev-opt extension            │◄──────────────►│ jev_opt.worker                │
│  commands, run registry      │  stdin/stdout   │  gepa.optimize(adapter=Jev…) │
│  WorkerBridge (child_process)│                 │  OMPReflectionLM ──► lm.complete
│  lm.complete handler:        │                 │  JevAdapter ──► Jev client ──┼──► OpenRouter
│   ctx.models.resolve         │                 │  fit / metrics / hygiene      │    Decisions API
│   pi-ai completeSimple       │                 └───────────────────────────────┘
│   usage ledger               │
└──────────────────────────────┘
```

Jev is the *task model being optimized*, not an optimizer LLM. OMP has no Decisions-API
surface, so Jev requests go from the worker to OpenRouter using `OPENROUTER_API_KEY`
(env or `.env`). Every **LLM** call (reflection, structural proposals) goes through OMP.
The worker process has no LLM SDK installed: `gepa` is installed without extras, so
LiteLLM is absent. A test asserts that `litellm`, `openai`, `anthropic` and `dspy` are
not importable in a finished run's interpreter state (`sys.modules`).

## 3. JSONL RPC protocol (v1)

One JSON object per line on stdout (worker→host) and stdin (host→worker). Worker logs go to
stderr only. Messages:

- request: `{"id": str, "method": str, "params": {...}}`
- reply: `{"id": str, "result": {...}}` or `{"id": str, "error": {"message": str}}`
- notification: `{"method": str, "params": {...}}` (no id)

Host → worker methods:

| method | params | result |
|---|---|---|
| `hello` | `{}` | `{protocol: 1, versions: {jev_opt, gepa, python}}` |
| `optimize` | `{task_path, runs_root, run_id?, conditions?: [..], resume?: bool}` | `{run_id, run_dir, conditions: {name: summary}}` |
| `status` | `{runs_root, run_id?}` | `{runs: [...]}` |
| `show` | `{run_dir, candidate}` | `{program_yaml, scores, rationale}` |
| `pareto` | `{run_dir}` | `{representatives, front}` |
| `evaluate_final` | `{run_dir, representative?}` | `{metrics, consumed_at}` |
| `generate_seeds` | `{task_path, out_dir, kinds?, n?}` | `{task, kinds: {kind: {files, rejected}}, optimizer_cost}` (§5.2; no Jev call) |
| `cancel` | `{}` | `{ok}` (writes `gepa.stop` for running conditions) |
| `shutdown` | `{}` | `{ok}` |

Worker → host methods:

| method | params | result |
|---|---|---|
| `lm.complete` | `{role: "reflection"\|"structural", prompt: str \| [{role, content}], max_tokens?}` | `{text, usage: {input, output, cost_total}, provider, model}` |

Worker → host notifications: `progress` `{run_id, condition, event, iteration?, best?, message}`.

The host resolves `role` → model spec from the task config (`optimizer.reflection_model`,
`optimizer.structural_model`, default `optimizer.model`, default `current`) and applies
`optimizer.thinking`. The worker never sees a model name it could route itself.

## 4. Jev Decision Program IR (`jev_opt.ir`)

A program maps an example's input fields to one label of the task's label set. It is data,
never code. Pydantic v2 models; YAML/JSON round-trip is lossless.

```yaml
version: 1
labels: [safe, unsafe]            # task output space (must equal task labels)
inputs: {question: "...", answer: "..."}   # state fields (name → description)
shared_state: true                # batch independent questions over identical state into one request
root: safety                      # entry node id
nodes:
  safety:
    kind: choice                  # Choice over options; each option maps to a label or a child node
    instructions: "..."
    state: [question, answer]     # subset of inputs sent as Jev state
    options:
      safe:   {criterion: "...", to: {label: safe}}
      unsafe: {criterion: "...", to: {label: unsafe}}
```

Node kinds:

| kind | Jev primitive | semantics |
|---|---|---|
| `choice` | Choice | options → `{label: L}` or `{node: id}` (hierarchical routing, argmax) |
| `one_vs_rest` | K × Noul | one Noul per label; `aggregation: {rule: argmax_normalized \| threshold, thresholds: {L: t}, abstain: L?}` |
| `score` | Score | ordered `levels`; `cutoffs` on the expected score map to labels (ordinal tasks) |
| `branch` | Noul | `condition` Noul; `if_true` / `if_false` node ids; `threshold` (default 0.5) |
| `switch` | none | routes on the verbatim value of state field `field`: `cases: {value: target}`, optional `default` target; no Jev call (e.g. per-subset questions) |
| `stage` | any | evaluates `decision` node, writes its result into derived state field `as`, then runs `next` with the augmented state |

Validation (`Program.model_validate` + `Program.check(task)`): node ids unique; graph acyclic
and every node reachable; every label reachable; option/label names are task labels;
`state` fields ⊆ inputs ∪ derived fields; Jev limits (Choice ≤ 255 options; Score 2–10
levels); size limits and hygiene checks (§8).

Numeric parameters (`thresholds`, `cutoffs`, `threshold`, Choice `biases`, `readout`) are **not**
exposed to the LLM as text components; they are fit numerically (§7).

### 4.1 Demonstrations in state

A program may carry a top-level `demos` spec that adds labelled TRAIN examples to the Jev state
of its question nodes (all of them, or `demos.nodes`) as the reserved field `examples`: a list of
`{<input fields of that node's state>: value, label: <gold task label>}`, most similar first. The
question instructions of those nodes are prefixed with a fixed note that explains the field.

```yaml
demos: {mode: knn, k: 8, balanced: true}      # or {mode: static, ids: [train ids]}
```

- `knn`: the `k` nearest TRAIN examples to the input by cosine similarity of a local
  sentence-embedding model (`task.embedding`, default `BAAI/bge-small-en-v1.5` at a pinned revision,
  CPU, vectors cached on disk under `<runs_root>/.embed-cache`). It is not an LLM call.
  `balanced` takes nearest-first round-robin over labels (labels ordered by their nearest member).
- `static`: fixed train ids. Only structural operators edit them (§6); the LLM prompts say to leave
  `demos` unchanged, since reflection never sees example ids.
- Hygiene: the pool (`jev_opt.demos.DemoPool`) is built from the train split only and refuses
  anything else; every selected demo is asserted to be a train example; static ids must be train
  ids (checked on every candidate; val/test ids are rejected); a train example is never its own
  demonstration (leave-one-out), so parameters fit on train outputs are not fit on self-labelled
  inputs. `evaluate_final` records `demo_ids_outside_train` (0 by construction).
- The demonstrations of an example are chosen once per run of that example and kept with its
  result, so offline re-decoding (§7) sees the same state.

### 4.2 Readout heads

A `choice`, `one_vs_rest` or `score` node may carry a fitted `readout` that replaces its argmax
(and routing) by a small supervised head over the answers of one or more question nodes:

```yaml
relation:
  kind: choice
  ...
  readout: {kind: lr, views: [relation, relation_x1], classes: [BEFORE, ...], W: [[...]], b: [...]}
```

- Features (`execute.view_features`), per view in order: Choice log-probabilities, Noul logits
  (`one_vs_rest`, `branch`), Score level probabilities and their logs; probabilities floored at 1e-4.
- `lr`: multinomial logistic regression, `softmax(W x + b)` over `classes` (task labels; the node
  is a leaf decision over them). `ridge` (ordinal tasks): linear regression onto the class index,
  `cutoffs` map it to a class, the label distribution is a normal(ŷ, `sigma`) spread over the bands.
  `W`/`b` include the feature standardization.
- `views[0]` is the node itself; further views are question nodes asked only as features: they are
  never routed to, their outcomes are free-form, and they are batched into the node's request when
  their state is identical (one request per example for a stacked head).
- With `optimizer.readout: true` every leaf question node of the main tree gets a single-view
  readout (`fit.with_readouts`) before fitting: `lr`, or `ridge` when the metric is QWK. Readouts
  and `biases` are exclusive.

## 5. Experimental conditions

| condition | GEPA candidate (`dict[str,str]`) | proposer |
|---|---|---|
| `jev-zero` | seed only; no optimization (evaluated on val, recorded) | — |
| `jev-text-opt` | one component per text slot (`nodes.<id>.instructions`, `nodes.<id>.options.<o>.criterion`, `nodes.<id>.levels.<i>`, `inputs.<f>`) | GEPA default (`InstructionProposalSignature` + `OMPReflectionLM`); topology (incl. `demos`) frozen. With `optimizer.max_component_chars`, GEPA's `custom_candidate_proposer` hook runs the same default prompt plus a hard length limit (over-long replies retried once, then trimmed at a sentence end) |
| `jev-program-opt` | single component `program` (full YAML) | adapter `propose_new_texts`: deterministic structural operator or OMP structural proposal, each validated; text edits via GEPA's own `InstructionProposalSignature` on the YAML |
| `jev-population-opt` | not a GEPA run (§5.1, deviation §12.19) | one OMP call per round proposes `population.size` question variants of one node |
| `jev-tailored` | text slots, as `jev-text-opt` (§5.3) | `custom_candidate_proposer` with the Jev rulebook: one OMP call rewrites the selected fields jointly; GEPA hooks for minibatches, acceptance and component selection |

`jev-program-opt` is the one documented place where jev-opt supplies `propose_new_texts`
(GEPA would otherwise emit unvalidated YAML). It reuses GEPA's
`InstructionProposalSignature.prompt_renderer/output_extractor` with a structural template,
and retries once with the validation error before giving up.

Without `--conditions`, a run executes `jev-zero`, `jev-text-opt` and `jev-program-opt`;
`jev-population-opt` and `jev-tailored` are opt-in.

### 5.1 `jev-population-opt` (`jev_opt.population`)

Batched population search over question variants, for programs whose question nodes ("slots") are
leaf decisions reached only through `switch` nodes (no derived state; all 8 study seeds qualify).

1. A variant of a slot is a small bag of questions: a main question over the slot's labels
   (`choice`, `one_vs_rest` or, for ordinal tasks, `score`; any state field selection; optionally
   with the demonstrations of `population.demos`) plus up to `population.max_aux` feature questions
   (`branch`, `choice`, `score`), decided by a readout (§4.2). Round 0 holds the paper seed's
   question (with and without demonstrations) and the questions of every extra seed (§5.2): the
   seed's node for the slot and the nodes it routes to, routing replaced by the readout. Every
   variant records its `origin` (seed name or `r<round>`) and `root` seed kind (proposals inherit
   the kind of the best elite member shown to the proposer).
2. Per round, one OMP call (role `structural`, reply budget `population.max_tokens`) renders the
   slot's elite (best `population.elite` variants by train CV), a per-class summary and the worst
   out-of-fold TRAIN examples with GEPA's `InstructionProposalSignature.prompt_renderer`, and asks
   for `population.size` radically diverse variants (reframed instructions, rewritten criteria,
   other primitives, other state fields, aux cues) as one YAML list, parsed item by item. Each
   variant is validated against the IR and hygiene (§8); text fields are capped at
   `max_component_chars`. Slots take turns across `population.rounds` rounds.
3. All new variants of a round are asked for every fit example in one packed request per example
   and state: the "population program" stacks them as views of one unfitted readout, so their
   question ids are the view node ids (`<slot>`, `<slot>_x<i>`, `<slot>_x<i>_a_<aux>`, plus
   `__<option>` for Nouls). Each variant's readout is then fit with k-fold CV
   (`optimizer.readout_folds`, grouped by `readout_group`) from cached answers; its out-of-fold
   predictions are its evidence. Slots partition the examples, so every combination of per-slot
   variants is scored exactly offline (exhaustively up to 20,000 combinations, else coordinate
   ascent).
4. After the last round, these programs are fit like any candidate (§7), scored on val and written
   as this condition's candidates with `tags`: `single` = the `population.val_candidates` best by
   train CV; `head-<kind>` = the kind's best round-0 variant(s) (seed + readout); `search-<kind>` =
   the best program whose variants descend from that kind; `stacked` = one readout over several
   of the `single` programs' variants (per slot forward selection by train CV: always a second
   view, more while the CV improves, at most `population.stack_max`). Every seed program is also
   scored exactly as written, with no fitted number (`zero-<seed>`, and `zero-<kind>` for the
   kind's seed with the best zero-shot train score, so that row is not selected on val). Val
   answers of all candidate and seed questions are bought in one packed request per example and
   state (requests recorded by earlier runs are replayed first so they are not re-bought).
5. Representatives `top-<tag>` (highest val primary metric per tag: `top-single`, `top-stacked`,
   `top-zero-<kind>`, `top-head-<kind>`, `top-search-<kind>`, …) are added to the usual ones.
   `population.max_jev_cost` stops further rounds once the condition's fresh Jev spend reaches it.
   State (`population.json`: variants with origin/root, proposals and rejections, rounds done,
   fresh Jev spend; `seeds/`: snapshots of the extra seeds) makes the condition resumable without
   repeating LLM calls.

### 5.2 Seed programs from the task definition (`jev_opt.seedgen`, `/jev-seeds`)

The paper seeds are faithful copies of LLM-style prompts. `/jev-seeds <task.yaml> --out <dir>
[--kinds jev-native,decomposed] [--n 3]` (worker method `generate_seeds`; no Jev call) asks the
optimizer model through OMP, once per kind, for `n` alternative seed programs:

- `jev-native`: the paper's nodes, kinds, option keys/targets and Score levels count, every text
  rewritten for Jev;
- `decomposed`: per routed node, any primitive or topology (one-vs-rest Nouls, a coarse-to-fine
  hierarchy, a Score for ordinal tasks, branch Nouls).

The writer sees the task and label definitions (the paper seed program and the task's label
descriptions), the state field names and code-side routing, and a fixed guidance text distilled
from TypeSafe's authoring docs (short literal instructions, criteria that separate each option
from its nearest neighbour, observable descriptions for every Score level, evidence named in
state, no indirection), Kiln's Jev optimization report and T. Kellogg's notes. It never sees a
data example or a gold label, so each seed is itself a zero-shot program. Replies are validated
against the IR and hygiene (one retry listing rejections); Score levels that map 1:1 to labels
round the expected level. Seeds are written as complete programs `<kind>-<i>.yaml` plus
`generation.json` (rationales, rejections, writer model and cost) and written before any is
scored; `optimizer.population.seeds: [{kind, path}]` makes them round-0 starts of population search.
`optimizer.fit_parameters: false` (with `calibrate: false`, no readout) runs a seed as written, e.g.
as its own `jev-zero` run.

### 5.3 `jev-tailored` (`jev_opt.tailored`)

GEPA's reflective text search (`jev-text-opt`'s candidate: one component per text slot, topology
frozen) adapted to the Jev decision model. GEPA still selects parents (Pareto, hybrid frontier),
schedules val (`ProgressivePolicy`) and stops; jev-opt fills five of its hooks (deviations §12.23–25). Each
piece has an `optimizer.tailored` flag so it can be ablated.

- **T1 rulebook** (`rulebook`): every proposal prompt carries a fixed block of Jev authoring rules
  (`tailored.RULEBOOK`, sources cited in the code): short literal questions, no step-by-step procedures,
  each criterion states what separates it from its nearest neighbour option, observable Score levels,
  evidence present and named in state, no indirection, Jev's weak spots (counting, dates, numeric
  comparison, long irrelevant state), and a guard: annotation conventions the labels depend on are
  rewritten, never deleted.
- **T2 continuous acceptance** (`continuous_accept`, `accept_n` 64, ≤ 128, `accept_tol`): GEPA's
  `batch_sampler` is `StratifiedMinibatchSampler`: per iteration `accept_n` TRAIN examples (all of train,
  not only the reflection pool), class-balanced for macro metrics and label-proportional otherwise,
  deterministic in (seed, iteration). GEPA evaluates parent and child on it in batched requests; its
  `acceptance_criterion` is `ContinuousAcceptance`: accept iff the child's mean log P(gold)
  (probabilities floored at 1e-4; failures count as the floor) is higher and its primary metric on the
  minibatch is at most `accept_tol` lower; an invalid child is rejected. Verdicts are evaluation rows
  (`stage: accept`).
- **T3 confusion targeting** (`confusion_target`): GEPA's `module_selector` is
  `TailoredAdapter.select_components`. On the parent's fit-slice results (after bias, out of fold), each
  Choice node's option pairs get error mass (weighted share of examples whose gold option and decision
  are the two options, either way; an option routing to a subtree covers that subtree's labels) and
  pairwise AUC of p_a / (p_a + p_b) between gold-a and gold-b examples at the node. The pair with the
  largest mass × (1 − AUC) is selected (repeat visits of a parent rotate through its top 3): both criteria
  and the node's instructions are rewritten jointly in one OMP call. Programs without a confused Choice
  pair (Score/Noul-only) fall back to round-robin over question nodes (all text slots of one node, then the
  input descriptions).
- **T4 minimal pairs** (`minimal_pairs`, 2–4): the reflective dataset of a targeted pair also holds up to
  `minimal_pairs` disjoint pairs of fit-slice TRAIN examples, one with gold option a and one with gold
  option b, ranked by cosine similarity of the demonstrations' embedding model (§4.1, same cache; small
  bonus when Jev confused a member), each shown with Jev's probabilities at the node. Ids outside the
  train pool are dropped before ranking and asserted absent (`DemoPool.vectors` refuses them).
- **T6 wording vs bias**: every candidate is fit on the train fit slice as in §7 (Choice class biases,
  cutoffs, thresholds, temperature, per the task config). Fit-slice examples are then scored out of fold:
  the same fit repeated on `readout_folds` CV folds (grouped by `readout_group`), each fold decoded with
  the numbers fit on the others; other train examples and val use the full fit. So the acceptance
  minibatch and GEPA's val scores (selection) are scores after bias, and a rewrite earns nothing a bias
  could do. The raw score (program as written, same answers) is recorded next to it (`quality_raw`,
  `primary_raw`, `logp_raw` in `evaluations.jsonl`).

Starts (`tailored.starts`): `paper` = the task's seed program; any other entry is a seed kind of
`tailored.seeds` (e.g. `jev-native`, §5.2), whose seed with the best zero-shot primary metric on the
train fit slice (run as written) is used. Each start is its own GEPA search (`<condition>/gepa-<start>/`),
run in order; the run's `max_reflection_cost` (OMP) and `tailored.max_jev_cost` (fresh Jev $, checked
between iterations by a GEPA stopper) are split evenly over the starts still to run, each start's share
fixed when it begins. Candidates of all starts share `<condition>/candidates/` (contiguous indexes,
`start` and tags in the metadata). After the last start, every fully evaluated candidate is also scored
exactly as written (no fitted number; tags `tailored-zero`, `tailored-zero-<start>`) and, with
`tailored.head`, each start's best fitted candidate by val primary metric gets a readout head (§4.2) fit on
the fit slice (`tailored-head`, `tailored-head-<start>`); val answers of these rows come from the cache.
Representatives follow from the tags (§10): `top-tailored` (fitted), `top-tailored-zero` (as written),
`top-tailored-head`, the same per start, and each start's seed as written (`top-tailored-seed-<start>`) and
fitted (`top-tailored-seedfit-<start>`). `starts.json` (per start: seed choice, caps, spend, status,
candidate indexes) makes the condition resumable; an unfinished start resumes its GEPA checkpoint.

## 6. Structural operators (`jev_opt.transforms`)

Deterministic (no LLM): `choice_to_one_vs_rest`, `one_vs_rest_to_choice`,
`choice_to_score` (only when the task declares `ordinal: true`), `score_to_choice`,
`hierarchize(groups)` (flat choice → coarse choice → fine choices; groups from
`task.label_groups` or from an LLM proposal), `flatten`, `toggle_shared_state`,
`set_abstain`, `set_aggregation`, `narrow_state` / `widen_state`, demonstration operators
(`demos_knn`, `demos_off`, `demos_k`, `demos_balanced`; data-aware in the adapter:
`demos_static:failures` = misclassified train fit examples, class-balanced, `demos_add` / `demos_drop`).
With `optimizer.max_component_chars`, LLM proposals whose edited text fields exceed the limit are
rejected with the reason (one retry), then trimmed. LLM-proposed: decomposition
into stages, new intermediate derived state, new grouping. Every proposal carries a one-line
`rationale` stored with the candidate; selection uses measured scores only.

## 7. Numerical fitting (no LLM)

`jev_opt.fit` fits `one_vs_rest` thresholds/biases, score `cutoffs` and branch thresholds on
the **train** split's Jev outputs (cached), by coordinate search over the probability grid
maximizing the task's primary metric; calibration via temperature scaling on train. Fitting
runs inside `JevAdapter.evaluate` before scoring a candidate on val, keyed by the candidate
hash, so GEPA always sees the fitted program.

Class calibration (`optimizer.class_bias`): every Choice outside decision subtrees gets
per-option additive log-prob biases (`biases`, grid −3…+3 in 0.25 steps, coordinate search) that
maximize the primary metric on the train fit slice; decoding applies `p_k · exp(b_k)` (probabilities
floored at 1e-4) before the argmax and in the label distribution, then temperature is refit for log
loss. `optimizer.fit_examples` sets the slice size (a stratified prefix of train); with
`optimizer.fit_holdout` the slice is excluded from GEPA's trainset, so reflection never sees those
examples except through the aggregate summary record.

Readouts (§4.2) are fit first, each on the fit examples that reached its node, independently of
the rest: `lr` = L2 logistic regression on standardized features, `C` ∈ {0.01 … 3} × class
weights {none, balanced} chosen by the primary metric of the out-of-fold predictions (ties: lower
log loss); `ridge` = ridge regression, `alpha` ∈ {0.3 … 100}, cutoffs fit on the out-of-fold
predictions by coordinate search on QWK. Folds: `optimizer.readout_folds` (default 5), stratified,
shuffled with `optimizer.seed`; with `optimizer.readout_group` (a regex whose first group maps an
example id to a group, e.g. a TimeBank-Dense document or an INLI premise) no group spans folds.
The chosen model is refit on all fit examples; classes absent from them get probability ≈ 0.
Readout, view and stage-decision nodes are excluded from the coordinate search.

## 8. Split hygiene and anti-gaming

- `task.yaml` declares `train`, `val`, `test` files. `optimize` loads only train and val;
  the test file is hashed, never parsed.
- Reflection sees train examples, gold labels and failures; val is used for selection
  (GEPA `valset`) and never appears in reflection records. `optimizer.reflection_pool` caps the
  train examples GEPA samples minibatches from (a stratified prefix, after the fit slice when
  `fit_holdout`).
- Demonstrations come only from train (§4.1); program id checks skip the `demos.ids` list, which is
  validated against the train ids instead.
- `evaluate_final` requires a finished run, evaluates the chosen representative once on
  test, writes `test_consumed.json`; afterwards `optimize --resume` of that run is refused.
- Programs are rejected if they contain any train/val input substring ≥ 40 chars, any
  example id, label-leaking fields, or grow beyond `max_program_chars` (default 8 × seed,
  min 4,000 chars). Readout numbers are excluded from these text checks.
- Evaluator, labels, splits and metrics are worker-side code and data; candidates are text
  and cannot reach them.

## 9. Evaluation scheduling and caching

- Jev calls are cached on disk by SHA-256 of `{endpoint, model, state, questions}`; the
  response records the served snapshot. Retries, pacing (≤ 1,800 rpm) and 52x/429 handling
  mirror the gauntlet client.
- Second tier: every recorded request is indexed per question (`q/`: SHA-256 of `{endpoint,
  model, state, question}` → request hash and question id; `req/`: the request body). A request
  that is not recorded as such is assembled from recorded answers of the same (state, question)
  in any earlier request; only unanswered questions are sent, in chunks of ≤ 64 questions. Jev
  answers each question of a request independently of the others (TypeSafe "Parallel questions"
  cookbook: same answers, same run-to-run spread batched or not), so re-packing never re-buys an
  answer. Usage of an assembled request is estimated (`input_tokens ≈ 308 + 0.26·state chars +
  0.22·question chars`, fitted on 13,319 recorded jev-1.13 requests; `Decision.estimated`); fresh
  spend is always the API's own figure. The run ledger records every source request verbatim.
- GEPA `cache_evaluation=False`: 0.1.4 keys its cache by `(candidate, example_id)` without
  the split, so distinct train/val lists collide. The Jev cache makes re-evaluation free.
- `ProgressivePolicy` (GEPA `EvaluationPolicy`): new candidates are scored on a deterministic
  stratified 32-example val subset, then 128, then full val only if they are within the
  current best's bootstrap interval at each stage.

## 10. Multi-objective representatives

Per example, `objective_scores = {quality, calibration (1 − Brier), cheap (1/calls),
tokens (1/(1+input_tokens/1000))}`, all higher-is-better, with `frontier_type="hybrid"`.
Representatives from GEPA's result: `best-quality`, `best-calibrated`, `best-cheap` (best quality among the candidates with
the fewest Jev calls), `top-quality` (highest val primary metric, no simplicity preference),
`top-<condition>` (runs with several conditions: `top-quality` within one condition), `top-<tag>`
(population candidates, §5.1: `top-single`, `top-stacked`) and, per
`optimizer.ensemble_k`, `ensemble-<k>`: the k distinct fully-evaluated candidates with the highest
val primary metric (across conditions), decoded as the mean of their label distributions plus
per-label log-prob biases and a temperature fit on the train fit slice (`jev_opt.ir.Ensemble`,
evaluated on val like the others and accepted by `evaluate_final`). For ordinal tasks the ensemble
decision is instead the members' mean predicted label index banded by `cutoffs` fit on the train
slice (the argmax of a mean distribution would ignore the members' fitted Score cutoffs); the mean
distribution still supplies the probabilities. `runs.rewrite_representatives(run_dir)` recomputes a
finished, test-unconsumed run's representatives from the Jev cache only (a cache miss raises) and
rewrites `pareto.json`. Among candidates
whose val quality is statistically indistinguishable (paired bootstrap, 95%), the simplest
(fewest Jev calls, then fewest chars) is preferred.

## 11. Run layout

```
runs/<run-id>/
  config.yaml            task + optimizer config as resolved
  environment.json       versions (jev-opt git SHA, omp, gepa, python, packages), Jev model, optimizer model identity, seeds
  <condition>/gepa/      GEPA run_dir (gepa_state.bin, run_log.json, candidates.json)
  <condition>/candidates/<n>.yaml  + rationale
  evaluations.jsonl      one row per (condition, candidate, split-stage) evaluation
  usage.jsonl            optimizer LLM usage (host ledger) and Jev usage
  jev_responses.jsonl.gz, jev_responses.00001.jsonl.gz, …
                         verbatim Jev responses used by this run (see below)
  best-quality.yaml best-cheap.yaml best-calibrated.yaml top-quality.yaml pareto.json
  top-<condition>.yaml top-<tag>.yaml ensemble-<k>.yaml   (when applicable)
  jev-population-opt/population.json   population state (§5.1)
  jev-tailored/starts.json, gepa-<start>/, seeds/<start>.yaml   per-start state, GEPA run dirs, seed snapshots (§5.3)
  val_predictions.json   per-example val predictions of every representative (+ gold)
  report.md
  test_consumed.json     only after /jev-evaluate-final
```

Every Jev response a run uses is copied into its directory. That covers fresh responses and
`.jev-cache` hits, during fitting, validation stages, representative scoring and `evaluate_final`. Each
distinct request becomes one record `{request_hash, split, condition, request: {model, state,
questions}, response, latency_s}`, deduplicated by `request_hash` within the run (first use wins).
`jev_opt.jev.ResponseLedger` appends rows to `jev_responses.pending.jsonl` as they arrive. When
`optimize` ends (finished, cancelled or failed) and after `evaluate_final`, it rewrites all records
into gzip shards of 5,000 rows, sorted by `request_hash` and written with a zero gzip mtime so
they are byte-deterministic. `run.json` lists them as `jev_responses: {shards, records}`. On resume
the ledger reloads the existing shards. The shards are kept with the run; the shared cache is
not.

## 12. Deviations from "use GEPA for everything"

1. `propose_new_texts` in `jev-program-opt`: YAML must validate against the IR (§5).
2. `ProgressivePolicy`: GEPA 0.1.4 ships only `FullEvaluationPolicy`; we implement GEPA's
   own `EvaluationPolicy` protocol rather than a separate scheduler.
3. GEPA evaluation cache disabled (§9); Jev-level cache instead.
4. OMP usage accounting: direct pi-ai calls do not enter OMP session stats in 18.3.x, so the
   extension keeps a ledger (`usage.jsonl`, `pi.appendEntry("jev-opt.usage", …)`, status line).
5. Protocol additions (§3), agreed between host and worker: `lm.complete` params also carry
   `run_dir` (absolute) so the host books usage into that run's `usage.jsonl`; the worker emits
   `progress` `{event: "run_started", run_id, run_dir, optimizer: {model, reflection_model,
   structural_model, thinking}}` before any `lm.complete` of a run, because the worker alone parses
   `task.yaml`/`config.yaml` and the host must not duplicate that parsing just to route roles;
   `optimize` with `resume: true` omits `task_path` (the worker reads `<run_dir>/config.yaml`).
6. Extra command `/jev-cancel [run]`: the only interactive trigger for the bridge's abort → `cancel`
   path while a background run is active (otherwise only leaving omp stops a run).
7. One worker process per long job (`optimize`/resume/`evaluate_final`), short-lived workers for
   `status`/`show`/`pareto`/selftest: a worker never runs two optimizations, `cancel {}` is
   unambiguous, and a crash affects one run.
8. Test-split fingerprint: during `optimize` the test file is identified by `os.stat` (path,
   size, mtime) and never opened, not even to hash it; its SHA-256 is computed only inside
   `evaluate_final` and stored in `test_consumed.json` (plus `<runs_root>/test_ledger.jsonl`).
   A test asserts `optimize` never opens the file.
9. `ProgressivePolicy` promotion hook: GEPA 0.1.4 calls `get_eval_batch` without the candidate,
   so a new candidate is first scored on the stage-1 ids and promoted inside
   `get_valset_score`, which the engine calls immediately after adding it. Promotion evaluates
   the extra ids through the adapter, charges them with `state.increment_evals`, and updates
   per-id fronts with GEPA's own `GEPAState._update_pareto_front_for_val_id` (private API; pinned
   to 0.1.4) and the objective front. `get_best_program` only considers fully evaluated
   candidates. GEPA's `ValsetEvaluatedEvent` still reports the stage-1 count; jev-opt reads the
   post-promotion scores from the policy.
10. Run layout additions to §11: `run.json` (state, per-condition status/summary, requested
    conditions), `seed.yaml` (resume does not depend on the original seed file),
    `<condition>/candidates/<n>.json` (key, parents, rationale, per-id val scores, coverage),
    `<condition>/promotions.json`. `config.yaml` stores the resolved task under `task` with
    absolute data paths plus `test_fingerprint`.
11. IR details fixed during implementation: `Program.temperature` (fitted, §7) is part of the IR;
    `Option.to` defaults to the label named like the option; Score maps the expected level index
    through `cutoffs` into ordered `outcomes`; `one_vs_rest` thresholds act in logit space
    (`threshold`: p = t maps to 0.5; `argmax_normalized`: per-option logit biases) so the label
    distribution agrees with the thresholded decision; nodes inside a stage's decision subtree may
    emit free-form values (the derived field), everything else must emit task labels.
12. Fitting scope: parameters are fit on a stratified `optimizer.fit_examples` (default 64) train
    subset from one exhaustive run (every branch evaluated once, so re-decoding needs no calls);
    parameters inside stage decision subtrees are frozen because they change derived state.
    That fit subset also feeds the train-only summary record of the reflective dataset.
13. Per-example quality is class-balanced (weights with mean 1) when the primary metric is
    `macro_f1`/`balanced_accuracy`, so GEPA's averaged score tracks the task metric; GEPA's
    `skip_perfect_score` is then disabled because weighted scores are not bounded by 1.
    `micro_f1` with `negative_label` (e.g. TB-Dense `VAGUE`) averages over the other labels only
    in reports and fitting; that metric does not decompose per example, so GEPA's search score
    stays plain correctness.
14. `jev-program-opt` also passes the OMP reflection callable as `reflection_lm` (unused for
    proposals, since `adapter.propose_new_texts` owns them) so GEPA's `max_reflection_cost`
    stopper sees the shared text+structural cost ledger.
15. `evaluation.command` (black-box evaluator): the command gets `{program}` (fitted YAML) and
    `{examples}` (JSONL incl. labels) paths and prints `{score, per_example?}`; jev-opt then does
    not execute the program itself, `calibration` falls back to quality and `cheap` to the static
    request count.
16. Demonstrations (§4.1) use a local embedding model (sentence-transformers). It is a numeric
    retrieval component like `fit`, not an optimizer LLM; it runs on CPU at a pinned revision.
17. Length-limited text proposals use GEPA's `custom_candidate_proposer` hook with GEPA's own
    `InstructionProposalSignature` (default template + a limit sentence); without
    `max_component_chars` text-opt uses GEPA's default proposer unchanged.
18. `optimize` accepts `seed` (CLI `--seed`) to override `optimizer.seed`; `config.yaml` records the
    effective value, so extra seeds of one configuration share a task file.
19. `jev-population-opt` (§5.1) is not a GEPA run. GEPA 0.1.4 evaluates one candidate per
    minibatch and cannot batch-evaluate a population in one Jev request; Jev answers every
    question of a request independently, so M variants cost one request per example and state,
    and exact out-of-fold readouts make every combination of per-node variants scorable offline.
    The piece of GEPA reused is its `InstructionProposalSignature` prompt renderer/output extractor
    (with a population template); candidate files, representatives, run layout, ledger and hygiene
    are shared with the GEPA conditions.
20. Readouts (§4.2) are fitted heads with F×K parameters, fit on train only: a "Jev + fitted
    head" system, reported apart from zero-shot rows.
21. Two-tier Jev cache (§9): usage of a request assembled from per-question cache entries is an
    estimate (flagged), because no such request was ever billed; fresh spend is exact.
22. `/jev-seeds` (§5.2) writes seed programs with the optimizer model outside any GEPA run: a
    one-shot proposal from definitions only (no evaluation feedback), validated like structural
    proposals; `generate_seeds` announces its output directory with a `run_started` progress event
    so the host routes roles and books usage there.
23. `jev-tailored` (§5.3) replaces three GEPA defaults through GEPA 0.1.4's own hooks, not a separate
    search loop: `batch_sampler` (stratified `accept_n`-example TRAIN minibatches instead of
    `EpochShuffledBatchSampler` with 3 examples), `acceptance_criterion` (mean log P(gold) must rise and
    the minibatch primary metric may drop by at most `accept_tol`, instead of a strictly larger sum of
    per-example 0/1 scores) and `module_selector` (the most confused, least separable Choice option pair
    instead of round-robin over single slots); its proposals go through `custom_candidate_proposer` (one
    OMP call rewrites the selected fields jointly under the rulebook). Reason: on 3-example minibatches a
    0/1 sum mostly measures sampling noise, and single-slot edits cannot state what separates two options.
    The minibatch is drawn from all of train (the fit slice included), so `fit_holdout` /
    `reflection_pool` do not apply; the hygiene boundary is unchanged (train only, never val/test).
24. `jev-tailored` scores candidates after bias (§5.3 T6): fit-slice examples are decoded out of fold
    (the §7 fit repeated per CV fold), so GEPA's minibatch and val scores measure wording with the fitted
    numbers already applied; the raw scores are recorded but not used for acceptance or selection. Each
    candidate's fit costs `readout_folds` + 1 coordinate searches (no Jev call).
25. `jev-tailored` runs one GEPA search per start inside one condition; GEPA's candidate indexes are shifted
    so the starts share `<condition>/candidates/`, and the run's OMP and Jev budgets are split evenly over
    the starts (each GEPA search gets its own `max_reflection_cost` and a Jev-spend stopper).
