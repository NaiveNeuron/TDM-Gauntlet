# TDM-Gauntlet

TDM-Gauntlet is the benchmark of the paper *Do We Still Need Fine-Tuned Classifiers? Swapping Typed Decision Models into 24 ACL-Venue Papers* ([arXiv:2611.20191](https://arxiv.org/abs/2611.20191)). It holds 46 English test sets from 24 papers published at ACL, EMNLP, NAACL and EACL in 2024–2026, together with the question asked on each test set, the paper's metric and the paper's published scores. The paper replaces each paper's predictor with a Typed Decision Model and asks how many of the papers' fine-tuned classifiers such a model can replace under each paper's own evaluation. A Typed Decision Model takes a JSON state and typed questions about it and returns a probability for each option, without any training.

This repository contains the code to rebuild the test sets, run the models, score them and produce the paper's tables and figures, together with the `jev-opt` optimizer. It does not contain the test data or the model responses.

## Contents

| Path | Contents |
|---|---|
| `src/gauntlet/` | Python package: one task adapter per test set (`tasks/`), model backends, scorers with each paper's metric, the `gauntlet` command |
| `prompts/` | The question asked on each test set |
| `data/frozen/` | Small ID, hash and exclusion lists that some prepare scripts read; the rebuilt test sets are written here |
| `data/coverage/` | The sample of ACL-venue papers and its labels, used by `scripts/coverage_*.py` for the paper's coverage study |
| `data/manifests/` | One manifest per paper: data sources, licences, test-set construction, SHA-256 hashes of the test items, and the paper's published scores |
| `scripts/` | `prepare_*.py` rebuild the test sets; `run_*.sh` run the hosted models; the other scripts compute the analyses and write the paper's tables and figures |
| `jev-opt/` | The optimizer for Jev decision programs (an Oh My Pi `omp` plugin with a Python worker) and the scripts of the optimization experiments (`jev-opt/experiments/`) |
| `results/paper_numbers.json` | The published scores on each test set, with the table and page they come from |
| `tests/` | Tests of the package |

## Models

| Backend | Model | Where it runs |
|---|---|---|
| `jev` (default) | TypeSafe Jev (`typesafe/jev-1.13`), the paper's main model | OpenRouter Decisions endpoint |
| `clef`, `clef-flash` | Cloudflare Clef and Clef-flash (`Cloudflare/clef`, `Cloudflare/clef-flash` on Hugging Face) | local GPU |
| `luna` | OpenAI GPT-6 Luna Decisions (`openai/gpt-6-luna-decisions`) | OpenRouter Decisions endpoint |
| `msd1` | Microsoft-Decision-1 (`microsoft/microsoft-decision-1`) | OpenRouter Decisions endpoint |

## Setup

The package needs Python 3.11 and [uv](https://docs.astral.sh/uv/):

```sh
uv sync                     # Jev, Luna and Microsoft-Decision-1 backends, scoring, analyses
uv sync --extra clef        # adds the local Clef and Clef-flash backends (PyTorch, CUDA 12.8)
uv sync --extra lora        # adds LoRA fine-tuning (gauntlet.lora_train)
```

The hosted models are called through the OpenRouter Decisions endpoint. Set `OPENROUTER_API_KEY` in the environment or in a `.env` file at the repository root.

## Rebuild the test sets

Each `scripts/prepare_<paper>.py` downloads its dataset from the original source, builds the test set, checks it against the SHA-256 hashes in `data/manifests/<paper>.json` and writes it to `data/frozen/<paper>/`. The docstring at the top of each script gives the exact command and any file you must download yourself. For example:

```sh
uv run python scripts/prepare_lifetox.py
```

Some sources require a consent form, a gated download or an LDC licence (for example DREsS, MADE, LegalSeg and the TimeBank texts of ConTempo). [`DATA_LICENSES.md`](DATA_LICENSES.md) lists the licence of each source and says which test sets are stored as full rows and which as item IDs and hashes only.

## Run a model

```sh
uv run gauntlet list                                  # registered tasks
uv run gauntlet run lifetox                           # Jev (default backend)
uv run gauntlet run lifetox --backend clef            # local Clef; --device cuda:0 (repeatable)
uv run gauntlet run lifetox --backend luna            # GPT-6 Luna Decisions
uv run gauntlet run lifetox --limit 20                # smoke run on 20 items, written to runs/_smoke/
uv run gauntlet status                                # which tasks have scores
```

A run writes `runs/<task>/`:

- `responses*.jsonl.gz`: one row per request, with the model's response;
- `predictions.jsonl`: item ID, gold label and prediction;
- `scores.json`: the paper's metrics;
- `manifest.json`: model snapshots, prompt hash, counts, tokens and cost.

The Luna and Microsoft-Decision-1 backends stop at the spend caps set in `src/gauntlet/luna.py`. `scripts/run_luna.sh` and `scripts/run_msd1.sh` run all tasks on these two models.

## Optimizer

`jev-opt/` searches for better Jev decision programs on a task's training and development data and evaluates the chosen program once on the test split. [`jev-opt/README.md`](jev-opt/README.md) covers installation (`omp` ≥ 18.3), the commands and the run layout; [`jev-opt/experiments/`](jev-opt/experiments/) holds the task files and the scripts of the paper's optimization experiments.

## Tables and figures

No model runs ship with the repository, so run the models first (`gauntlet run`, the `run_*.sh` scripts and the experiments in `jev-opt/experiments/`). The analysis scripts in `scripts/` and `jev-opt/experiments/` then read the runs and write Markdown and JSON reports to `results/` and LaTeX tables and figures to `paper/generated/`. The docstring of each script states its inputs and outputs.

## Tests

```sh
uv run pytest tests
(cd jev-opt/python && uv run pytest)
node --test jev-opt/extension/*.test.ts               # needs node ≥ 24
```

## Licences

The code is released under the Apache License 2.0 ([`LICENSE`](LICENSE)). Each test set keeps the licence and terms of its source; see [`DATA_LICENSES.md`](DATA_LICENSES.md).

## Citation

```bibtex
@misc{suppa2026tdmgauntlet,
  title         = {Do We Still Need Fine-Tuned Classifiers? Swapping Typed Decision Models into 24 ACL-Venue Papers},
  author        = {{\v{S}}uppa, Marek and Kop{\v{c}}an, Jaroslav},
  year          = {2026},
  eprint        = {2611.20191},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2611.20191}
}
```
