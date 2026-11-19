# jev-opt

Optimize **Jev decision programs** (TypeSafe System One, `typesafe/jev-1.13`) against labelled
development data. jev-opt is an Oh My Pi (`omp`) plugin plus a Python worker:

- **omp** owns every optimizer LLM call (model resolution, credentials, reasoning levels, inference
  via `@oh-my-pi/pi-ai`), the commands and the worker lifecycle;
- **GEPA** (`gepa==0.1.4`, no extras) owns the search: Pareto candidate selection, sampling, merge,
  stopping, persistence;
- jev-opt owns only what is Jev-specific: the program IR, transforms, Jev execution, numerical
  fitting, metrics and split hygiene.

The design contract is [`docs/architecture.md`](docs/architecture.md).

## Requirements

- `omp` ≥ 18.3 (verified with 18.3.2)
- [`uv`](https://docs.astral.sh/uv/) on `PATH` — the worker is `uv run python/worker.py` (inline
  script dependencies, installed on first use)
- `OPENROUTER_API_KEY` in the environment or in the project's `.env` (Jev Decisions API; never
  logged)
- An authenticated omp model for the optimizer LLM (whatever `/model`, `@slow`, … resolve to)

## Install

Link the plugin directory into your user plugins (a local path is always installed as a link, so
edits to these files take effect on the next omp start):

```sh
omp plugin link /absolute/path/to/TDM-Gauntlet/jev-opt
# equivalent: omp plugin install /absolute/path/to/TDM-Gauntlet/jev-opt
omp plugin list                                   # ● jev-opt@0.1.0
omp -p --no-session "/jev-opt-selftest"           # jev-opt loaded v0.1.0 · current=… · @slow=… · @smol=…
```

`package.json#omp.extensions` points omp at `extension/index.ts`; `@oh-my-pi/pi-ai` and
`@oh-my-pi/pi-coding-agent` are peer dependencies provided by the omp host — do not `npm install`
them.

To remove it: `omp plugin uninstall jev-opt`. omp 18.3.2 shells out to `bun` for uninstall; without
`bun` on `PATH`, delete the symlink `~/.omp/plugins/node_modules/jev-opt` and the `jev-opt` entry in
`~/.omp/plugins/omp-plugins.lock.json` instead.

Without installing, load it for one invocation:

```sh
omp -e ./jev-opt/extension/index.ts                                   # interactive
omp -p --no-session --no-extensions -e ./jev-opt/extension/index.ts "/jev-opt-selftest"
```

## Commands

`<run>` is a run id under `<cwd>/runs/jev-opt/` or a path to a run directory.

| Command | What it does |
|---|---|
| `/jev-optimize <task.yaml> [--conditions a,b] [--run-id x] [--seed n]` | Start an optimization run; `--conditions` restricts it to the listed conditions (`jev-zero`, `jev-text-opt`, `jev-program-opt`, and the opt-in `jev-population-opt` and `jev-tailored`; architecture §5); `--seed` overrides `optimizer.seed` (extra seeds of one configuration: same task file, new run id) |
| `/jev-resume <run>` | Resume an unfinished run from GEPA's saved state (refused after `/jev-evaluate-final`) |
| `/jev-status [run]` | List runs (or one run) plus jobs running in this session and optimizer-LLM usage |
| `/jev-seeds <task.yaml> --out <dir> [--kinds jev-native,decomposed] [--n 3]` | Write seed programs from the task definition only (architecture §5.2): `jev-native` rewrites of the paper seed and `decomposed` alternatives, validated, as `<kind>-<i>.yaml` + `generation.json`; optimizer LLM only, no Jev call |
| `/jev-show <run> <candidate\|representative>` | Print a candidate program (YAML), its scores and rationale; representatives: `best-quality`, `best-cheap`, `best-calibrated`, `top-quality`, `top-<condition>` (runs with several conditions), `top-single` / `top-stacked` (`jev-population-opt`), `top-tailored` / `top-tailored-zero` / `top-tailored-head` (`jev-tailored`), `ensemble-<k>` |
| `/jev-pareto <run>` | Print the representatives and the Pareto front |
| `/jev-evaluate-final <run> [representative] [--yes]` | Evaluate one representative (any listed by `/jev-pareto`, including `ensemble-<k>`) once on the test split; asks for confirmation (headless: requires `--yes`). Irreversible: the run can no longer be resumed |
| `/jev-cancel [run]` | Ask running jobs to stop gracefully (GEPA stop file); results so far are still reported |
| `/jev-opt-selftest [--worker]` | Print load status and resolved models without any LLM call; `--worker` also starts the Python worker and prints its versions |

**Interactive (TUI):** `/jev-optimize`, `/jev-resume` and `/jev-evaluate-final` run in the
background. Progress appears on the `jev-opt` status line (run, elapsed time, latest event,
optimizer-LLM calls/tokens/cost) with toasts for run start, finished conditions and errors; the final
summary is posted to the conversation as a `jev-opt` message. Worker logs go to
`<cwd>/runs/jev-opt/worker.log`. Leaving omp stops running workers (cancel, then shutdown, then
signals).

**Headless (`omp -p`):** commands wait for completion, print results to stdout and progress/worker
logs to stderr:

```sh
omp -p --no-session "/jev-optimize jev-opt/examples/lifetox/task.yaml --conditions jev-zero,jev-text-opt"
omp -p --no-session "/jev-evaluate-final <run> best-quality --yes"
```

`omp -p` reads a prompt from piped stdin until EOF, so from scripts or CI run it with `</dev/null`.
Interrupting it (`^C`) cancels the run and stops the worker.

## Optimizer LLM configuration

The worker never names a model; it asks the host for a completion by role (`reflection` or
`structural`). The host resolves the role from the run's `optimizer` config:

| key | meaning |
|---|---|
| `optimizer.reflection_model` | model for GEPA reflection (text proposals); falls back to `optimizer.model` |
| `optimizer.structural_model` | model for structural program proposals; falls back to `optimizer.model` |
| `optimizer.model` | default for both roles; default `current` |
| `optimizer.thinking` | `off` → reasoning disabled; `auto`/unset → provider default; `minimal`\|`low`\|`medium`\|`high`\|`xhigh`\|`max` → that effort |

Model specs are anything omp resolves: `current` (the session's live model), role aliases such as
`@slow`/`@smol`, `provider/id`, or a bare model id. Credentials, OAuth refresh and retries are omp's.

Optimizer-LLM usage is not part of omp's session totals (direct pi-ai calls, architecture §12.4), so
the extension keeps its own ledger: one `{"source": "omp", role, spec, provider, model, input,
output, cache_read, cache_write, cost_total, stop_reason, at}` row per call in
`<run>/usage.jsonl` (next to the worker's `"source": "jev"` rows), a `jev-opt.usage` session entry,
and the status-line totals.

## Development

```sh
node --test jev-opt/extension/*.test.ts        # bridge (fake worker), lm.complete, argument parsing
omp -p --no-session --no-extensions -e ./jev-opt/extension/index.ts "/jev-opt-selftest"
```

Tests need node ≥ 24 (TypeScript type stripping) and no credentials: the worker is a fake process
and `completeSimple` is injected. Extension files:

| file | role |
|---|---|
| `extension/index.ts` | extension factory: commands, jobs, status line, output |
| `extension/bridge.ts` | `WorkerBridge`: JSONL RPC over `node:child_process` (architecture §3) |
| `extension/lm.ts` | `lm.complete` handler (role → model, thinking, pi-ai call) and usage ledger |
| `extension/cli.ts` | argument parsing and plain-text rendering |
| `extension/omp-types.ts` | the small structural subset of omp/pi-ai types the extension touches |

## Python worker

`python/` is the `jev_opt` package (src layout, `pyproject.toml` + `uv.lock`) and `python/worker.py`,
a uv inline-script entry point (`uv run jev-opt/python/worker.py`, any cwd; `worker.py.lock` pins it)
that speaks the JSONL RPC of architecture §3 on stdin/stdout and logs to stderr. Dependencies:
`gepa==0.1.4` (no extras), pydantic, pyyaml, httpx, numpy, scipy, scikit-learn, and
sentence-transformers (local embedding model for kNN demonstrations, loaded only when a program uses
them). No LLM SDK is installed or imported; `environment.json` records `forbidden_modules_loaded` for
every run.

| module | owns |
|---|---|
| `ir.py` | Pydantic program IR (choice / one_vs_rest / score / branch / stage, `demos`, fitted `biases` and `readout` heads), `Ensemble`, YAML/JSON round-trip, `check(task)`, text slots |
| `demos.py` | demonstration pool over the TRAIN split: static ids or kNN (`SentenceEmbedder`, pinned revision, disk-cached vectors), class-balanced selection, leave-one-out |
| `task.py` | `task.yaml` schema, JSONL rows (`id`, input fields, `label`), canonical seed Choice |
| `jev.py` | Decisions client: SHA-256 disk cache of `{url, model, state, questions}` plus a per-(state, question) tier, ≤ `max_rpm` pacing, 429 reset / 52x retries, usage and served snapshots |
| `execute.py` | runs a program per example (hard routing, soft label distribution, derived stage state, shared-state batching, demonstrations in state, Choice biases) and ensembles |
| `metrics.py` | scikit-learn metrics (+ hand-written ECE) |
| `fit.py` | readout heads (LR / ridge, cross-validated), threshold / bias / cutoff / Choice class-bias coordinate search, ensemble label biases, temperature scaling on train (no LLM, no new calls) |
| `transforms.py` | deterministic structural operators; structural / text-edit prompts via GEPA's `InstructionProposalSignature` |
| `hygiene.py` | test-split gate, program leak/id/size guard, consumption ledger |
| `adapter.py` | `JevAdapter` (GEPA `GEPAAdapter`) and `ProgressivePolicy` (GEPA `EvaluationPolicy`) |
| `reflection.py` | `OMPReflectionLM`: GEPA `LanguageModel` → `lm.complete` |
| `rpc.py`, `worker.py` | JSONL channel and method dispatch |
| `runs.py` | run layout, conditions, representatives (paired bootstrap), `report.md`, status/show/pareto/evaluate_final |
| `population.py` | `jev-population-opt`: variant proposals, packed evaluation, train-CV readouts, exact combinations, stacks |
| `tailored.py` | `jev-tailored`: the Jev rulebook, stratified minibatch sampler and continuous acceptance, confusion-pair component selector, minimal pairs, out-of-fold bias scoring (GEPA hooks) |

A task is a `task.yaml` next to its JSONL splits (see `examples/minimal/task.yaml` for the full
schema in use and `examples/lifetox/task.yaml` for every optimizer knob). Rows are
`{"id": ..., "<input field>": ..., "label": ...}`; `optimize` reads only train and val.
`seed_program` is optional: without it the seed is a direct Choice over the labels whose criteria
are the label descriptions.

v2 levers (architecture §4.1, §5, §7, §10), all off by default:

| key | effect |
|---|---|
| `seed_demos: {mode: knn, k: 8, balanced: true}` | add demonstrations to the seed program (`mode: static` takes `ids:`); Jev sees them as state field `examples` |
| `embedding: {model, revision, cache_dir}` | kNN embedding model (default `BAAI/bge-small-en-v1.5` at a pinned revision, CPU; vectors cached under `<runs_root>/.embed-cache`) |
| `optimizer.class_bias: true` | fit per-option log-prob biases of every Choice on train to maximize the primary metric (then temperature) |
| `optimizer.fit_examples`, `optimizer.fit_holdout: true` | size of the stratified train fit slice; with `fit_holdout` that slice is excluded from GEPA's trainset (reflection) |
| `optimizer.reflection_pool: 100` | cap on the train examples GEPA samples reflection minibatches from |
| `optimizer.max_component_chars: 1500` | per text slot limit: stated in every proposal prompt, over-long proposals retried once, then trimmed; `max_program_chars` stays the total budget |
| `optimizer.ensemble_k: [3, 5]` | representatives `ensemble-<k>`: top-k distinct val candidates, mean label distribution, per-label biases fit on train |

v3 levers (architecture §4.2, §5.1, §7, §9), off by default:

| key | effect |
|---|---|
| `optimizer.readout: true` | every leaf question node gets a readout head fit on the train fit slice: L2 logistic regression over its answers (ridge + cutoffs for QWK), hyperparameters by cross-validation |
| `optimizer.readout_folds: 5`, `optimizer.readout_group: <regex>` | readout CV folds; the regex's first group maps example ids to groups kept within one fold (documents, premises) |
| `optimizer.population: {size, rounds, elite, worst_examples, max_aux, val_candidates, stack_max, demos, max_jev_cost, max_tokens, seeds}` | `jev-population-opt` (opt-in via `--conditions`): round 0 = the paper seed plus `seeds: [{kind, path}]` (e.g. from `/jev-seeds`); per round one OMP call proposes `size` question variants of one node, all asked in packed requests, scored by train CV; val scores the best programs, stacks of their variants, each seed zero-shot and per-kind seed+head / search bests (`top-single`, `top-stacked`, `top-zero-<kind>`, `top-head-<kind>`, `top-search-<kind>`) |
| `optimizer.fit_parameters: false` | no threshold / cutoff / bias search; with `calibrate: false` and no readout the program runs exactly as written (zero-shot rows of generated seeds) |
| `optimizer.tailored: {starts, seeds, rulebook, continuous_accept, accept_n, accept_tol, confusion_target, minimal_pairs, max_examples, max_jev_cost, head}` | `jev-tailored` (opt-in via `--conditions`, architecture §5.3): one GEPA text search per start (`paper`, or a seed kind whose best zero-shot train seed is used); rulebook prompts, acceptance on `accept_n` stratified train examples by mean log P(gold) with a primary-metric guard, joint rewrites of the most confused, least separable Choice option pair with minimal pairs, candidates scored after out-of-fold fitted biases; `max_reflection_cost` and `max_jev_cost` are split over the starts. Representatives `top-tailored` (fitted), `top-tailored-zero` (as written), `top-tailored-head` (+ readout head), each also per start, plus each start's seed as written / fitted (`top-tailored-seed-<start>`, `top-tailored-seedfit-<start>`) |

The Jev cache has a second, per-(state, question) tier: a question already answered for a state is
never re-bought when it is asked again inside a differently packed request.

Environment: `OPENROUTER_API_KEY` (env or the nearest `.env` above the task or cwd) for Jev;
`JEV_OPT_DECISIONS_URL` overrides the Decisions endpoint; `JEV_OPT_MOCK_JEV=1` replaces Jev with
the offline keyword-overlap simulator (`jev_opt.mockjev`, cached separately under
`.jev-cache/mock`). The Jev cache defaults to `<runs_root>/.jev-cache` (`jev.cache_dir` overrides).

Run directory (`<runs_root>/<run-id>/`, architecture §11):

```
config.yaml  seed.yaml  run.json  environment.json    resolved task, seed program, state, versions
<condition>/gepa/  <condition>/candidates/<n>.{yaml,json}  GEPA checkpoint, candidates + rationale
evaluations.jsonl  usage.jsonl                        per-evaluation rows; Jev + omp LLM usage
jev_responses.jsonl.gz  jev_responses.00001.jsonl.gz …  verbatim Jev responses (see below)
best-quality.yaml  best-cheap.yaml  best-calibrated.yaml  top-quality.yaml  pareto.json  report.md
top-<condition>.yaml  top-<tag>.yaml  ensemble-<k>.yaml  val_predictions.json   per-condition/tag tops, ensembles, val predictions
test_consumed.json                                    only after /jev-evaluate-final
```

Every Jev response a run uses, fresh or served from the shared `.jev-cache`, including fitting,
validation stages, representatives and `evaluate_final`, is kept in that run's directory as one
JSON line `{request_hash, split, condition, request: {model, state, questions}, response,
latency_s}` per distinct request (deduplicated by `request_hash`, first use wins). Rows are appended
to `jev_responses.pending.jsonl` while the run is active. When a run ends (finished, cancelled or
failed) and after `evaluate_final`, the pending rows are merged into gzip shards of 5,000 rows,
sorted by hash and byte-deterministic. `run.json` lists them under `jev_responses: {shards,
records}`. The shards are kept with the run; the shared cache is not.

Examples:

```sh
# offline Jev; reflection still goes through omp's configured model
JEV_OPT_MOCK_JEV=1 omp        # then: /jev-optimize jev-opt/examples/minimal/task.yaml
# LifeTox (real Jev; ~200 train / 200 val from LifeTox_train, stratified 300 of LifeTox_test)
uv run jev-opt/examples/lifetox/prepare.py                 # or --raw-dir data/raw/lifetox
omp                           # then: /jev-optimize jev-opt/examples/lifetox/task.yaml
```

Tests (no credentials, no network: mock Jev over `httpx.MockTransport` or a local HTTP stub,
mock OMP host answering `lm.complete` over the worker's pipes):

```sh
cd jev-opt/python && uv run pytest -q
```
