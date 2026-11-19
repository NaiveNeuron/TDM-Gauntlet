# jev-opt experiments on gauntlet tasks

Each directory measures how far jev-opt moves Jev on one gauntlet task with a large,
prompt-addressable gap. Layout per experiment:

| file | role |
|---|---|
| `prepare.py` | uv inline script; writes gitignored `data/{train,val,test}.jsonl` + `data/manifest.json` (counts, SHA-256). No Jev calls. |
| `task.yaml` | jev-opt task: labels (descriptions = the gauntlet prompt's criteria), inputs, primary metric, optimizer knobs |
| `seed.yaml` | the gauntlet's frozen prompt (`prompts/<task>.json`) as a jev-opt program, text verbatim, so `jev-zero` is the gauntlet baseline |

The five sets of the frozen-configuration round (`contempo_matres`, `lifetox`, `sciprompt_s2orc`,
`smarter_hatexplain`, `smarter_latent_hate`) have no `prepare.py` and no `task.yaml`:
`make_frozen_tasks.py` writes their data splits, `seed.yaml` and `task_v3.yaml`.

Drivers (each script's header gives its phases and options):

| script | runs |
|---|---|
| `run_all.sh` | first optimization study: three single-condition runs per task, then one test evaluation per run |
| `run_v2_dev.sh` | second study (calibration, demonstrations, GEPA search, ensembles); validation only |
| `run_v3_dev.sh` | third study (question optimization): seed programs, then the A, P and N runs; validation only |
| `run_tailored_dev.sh` | Jev-adapted GEPA; validation only |
| `run_seeds.sh` | the third study's P runs with optimizer seeds 1 and 2, then one test evaluation per run |
| `run_frozen.sh` | the third study's configuration on the five sets above, then one test evaluation per set |
| `run_frozen_seeds.sh` | the frozen-configuration runs with optimizer seeds 1 and 2, then one test evaluation per run |
| `run_budget.sh` | trained baselines with the same labels as the pipeline (`budget_baseline.py`), on two GPUs |
| `run_clef_rq3.sh` | optimization step on Clef / Clef-flash answers (`clef_rq3.py`) |
| `run_luna_rq3.sh` | optimization step on GPT-6 Luna answers (`luna_rq3.py`) |
| `run_msd1_rq3.sh` | the same on Microsoft-Decision-1 answers (`run_luna_rq3.sh` with `MODEL=msd1`) |
| `run_h100_round2.sh` | Clef's base model through option log-probabilities, and Clef on the label-free rephrased question (`h100_round2.py`) |
| `run_h100_lora.sh` | LoRA fine-tuning of Clef's base model on the pipeline's train labels (`h100_lora.py`) |

Shared rules (`_common.py`):

- **train/val come only from the paper's released train/dev data**, never from the test split;
  deterministic (seed 0) label-stratified samples. val ≈ 1,000 examples, or the whole dev split
  when that is smaller; train = 500 examples, used for reflection minibatches and numerical
  fitting.
- **test is exactly the gauntlet's frozen English test set**: `prepare.py` imports the gauntlet
  task, calls `items()` and maps each item through the adapter's own `_state()`, so the Jev state
  holds exactly what the gauntlet sends (same clipping, same keys, same key order). The script
  asserts that the row count equals `n` in `runs/<task>/scores.json`, so the task's zero-shot run
  (`uv run gauntlet run <task>`) must exist first.
- Gauntlet test screens for non-English text were adjudicated flag by flag; train/val use an automatic
  approximation (`looks_english`: >5% non-ASCII letters, or 2+ distinct German/Spanish/French/
  Dutch/Portuguese/Italian function words) wherever the gauntlet screens the test set.
- Optimizer settings everywhere: `optimizer.model: current`, `use_merge: true`,
  `progressive: [64, 256]`, `max_metric_calls` ≈ 14 full-val equivalents per optimizing condition,
  conditions `jev-zero`, `jev-text-opt`, `jev-program-opt`.

Run (after `uv run jev-opt/experiments/<exp>/prepare.py`):

```sh
omp -p --no-session "/jev-optimize jev-opt/experiments/<exp>/task.yaml" </dev/null
omp -p --no-session "/jev-evaluate-final <run> best-quality --yes" </dev/null
```

## Request equivalence with the gauntlet

Checked for every test row by running each seed through jev-opt's executor with a recording
client and comparing the Decisions request body with the gauntlet adapter's `requests()`:
state (order-sensitive) and question object (`type`, `instructions`, `criteria`) are
byte-identical for every row of every task. Remaining differences:

- **question id** for the second routed question: jev-opt's question id is the node id, and node
  ids are unique, so `ezstance` noun-phrase items use `stance_np` (gauntlet: `stance`) and
  `edit_intent` Add/Delete items use `edit_intent_add_delete` (gauntlet: `edit_intent`). The
  majority branch keeps the gauntlet id (claim: 5,048 of 7,666; Modify: 1,384 of 2,312).
- **DREsS_New**: see below (one rubric per request, expected-level decoding).
- jev-opt applies temperature scaling fitted on train to the output distribution; this never
  changes the argmax, so predictions of Choice seeds are unaffected (calibration metrics are).

## Experiments

### `contempo_tbdense/` — ConTempo TB-Dense (6 labels)

- Data: TimeBank-Dense event-event pairs built with the gauntlet's own builder
  (`scripts/prepare_contempo.py`: TimeML text, Punkt sentences, context window, `<e1>/<e2>`
  marking), standard doc lists `data/raw/contempo/EMNLP-2019/data/tbd/{train,dev}_docs.txt`.
- Splits: train 500 (stratified from 4,032 train-doc pairs), val **629 = the whole dev split**
  (5 docs), test 1,427 (= `contempo/tbdense` n).
- Metric: `micro_f1` with `negative_label: VAGUE` (jev-opt option added for this task): micro
  P/R/F1 over the non-VAGUE labels, the paper's and gauntlet's primary. The metric does not
  decompose per example, so GEPA's search score is plain per-example correctness; reports,
  representatives and fitting use the true metric.
- Seed: one Choice, question id `relation`, state `{context, event_1, event_2}`.

### `ezstance_subtaskA/` — EZ-STANCE Subtask A (Favor/Against/Neutral)

- Data: release zip `subtaskA/{noun_phrase,claim}/raw_{train,val}_all_onecol.csv`, stratified
  by (target type, stance); automatic English screen dropped 66 train / 19 val rows.
- Splits: train 500 (290 claim), val 1,000 (649 claim) from the 6,684-row val split, test 7,666.
- Metric: `macro_f1` (paper; gauntlet `M` block). The gauntlet's `C`/`N` breakdown is not
  computed by jev-opt.
- Seed: `switch` on `target_type` (no Jev call; a jev-opt IR node added for per-subset
  questions) → the gauntlet's noun-phrase or claim Choice; options `Favor/Against/Neutral` map
  to `FAVOR/AGAINST/NONE`. `target_type` is a routing field and never sent to Jev.

### `smarter_implicit_hate/` — SMARTER implicit hate (6 categories)

- Data: Latent Hatred Stage-2, `other` dropped, Appendix B preprocessing; the train (50%) and
  val (20%) parts of the authors' `split_train_test` (seed 123) — the same split whose 30% test
  part the gauntlet samples. Rows whose ID or text is in the test set are removed (1).
- Splits: train 500, val 1,000 (label-proportional, from 1,253), test 900 (150 per class).
- Metric: `macro_f1`. Seed: one Choice, question id `label`, state `{post}`.

### `edit_intent/` — Re3-Sci edit intent (5 labels)

- Data: the paper's own `train.csv` / `val.csv` (UKPLab/emnlp2024-llm-classifier), read like
  `scripts/prepare_edit_intent.py`; stratified by (Modify vs Add/Delete, label).
- Splits: train 500, val 1,000 (from 1,776), test 2,312.
- Metric: `macro_f1` (paper Table 2 reports accuracy and macro-F1; macro-F1 is the one that
  weights the Claim vs Fact/Evidence confusion).
- Seed: `switch` on `edit_action`: Modify → 5-label `edit_intent` Choice, Add/Delete → the
  3-label add/delete Choice, exactly the gauntlet's routing. State `{edit_action, old_text,
  new_text}`.

### `inli/` — INLI (4 labels)

- Data: release `INLI Data/train.csv` / `val.csv`; whole premises sampled, each expanded to its
  four premise-hypothesis pairs (as the gauntlet does for test), so every split is exactly
  label-balanced.
- Splits: train 500 (125 premises), val 1,000 (250 of 1,000 premises), test 4,000.
- Metric: `accuracy` (paper Table 6 and gauntlet primary; equals class-balanced accuracy here).
- Seed: one Choice, question id `relation`, state `{premise, hypothesis}`.

### `dress_new/` — DREsS_New, one experiment per rubric

- Fits the IR without changes: `task_{content,organization,language}.yaml` +
  `seed_{rubric}.yaml`, ordinal labels `"1.0"` … `"5.0"`, `primary_metric: qwk`, seed = one Score
  node with the gauntlet's rubric question (question id = rubric name, levels = its 9 criteria).
- Data: the consented DREsS_New download. The gauntlet's test = the 20% part of
  `train_test_split(range(n), test_size=0.2, random_state=22)`; the remaining 80% is split
  the same way with `test_size=0.25, random_state=22` into train (60%) and val (20%), with the
  gauntlet's frozen row rules (empty essay / missing score / >5% non-Latin dropped) plus rows
  with an off-grid 0.5 score. Splits: train 500 (seeded random sample), val **376 = the whole
  reconstructed val split**, test 401. Same essays in all three rubric experiments.
- Deviations from the gauntlet request/decoding, inherent to one-label-per-example tasks:
  (1) the gauntlet asks all three rubric Scores in one request; here each request carries one
  rubric question (same state, same question object), so Jev sees a different question set and
  total-score QWK is not computed; (2) the gauntlet predicts the argmax level, jev-opt's Score
  node bands the expected level (seed cutoffs 0.5, 1.5, …, 7.5 = rounding) and the cutoffs are
  fitted on train in every condition, including `jev-zero`. Compare `jev-zero` test QWK with the
  gauntlet's `supplementary_expected_rounded` QWK rather than the argmax QWK.
