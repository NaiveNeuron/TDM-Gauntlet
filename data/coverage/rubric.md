# Coverage check A: classification rubric

The classification rules below were fixed before any paper in the sample was classified. The LLM
classifier (scripts/coverage_classify.py) receives this file verbatim plus one paper record; the second
labelling pass uses the same file.

## Unit

One ACL Anthology record from the corpus frame of the paper selection (main-conference and Findings
volumes of ACL, EMNLP, NAACL and EACL, 2024 through the 2026-09-26 cutoff). Classify the paper's
**own evaluated task(s) and central contribution** from the title and abstract. Open the PDF
(`https://aclanthology.org/<id>.pdf`) only when the abstract leaves a field below ambiguous for a
paper that could plausibly pass E1 and E2; say so in `pdf_checked`. Do not search the web.

## Fields

### `contribution` (one value)
- `method`: a new model, training scheme, prompting/decoding pipeline, metric or system.
- `resource`: a new dataset, benchmark, annotation scheme or corpus (possibly with baselines).
- `analysis`: an evaluation, probing, measurement or behavioural study of existing models or data.
- `other`: survey, position paper, theory, tool/demo-like artefact, proceedings front matter.

When a paper contributes both a resource and a method, pick what the abstract frames as the main
contribution; ties go to `method`.

### `modality`
`text` if every evaluated input is text (including code, tables serialised as text), else
`multimodal` (images, video, speech/audio, or other non-text signals).

### `task_family` (one code: the paper's main evaluated task)
Bounded families (the output is a label from a fixed inventory, a place on a predefined scale, or
an order/selection over supplied candidates):
- `B-SAFE`: safety, toxicity, hate, harmful-content, refusal or jailbreak-success classification.
- `B-AFFECT`: sentiment, emotion, stance, sarcasm, persuasion, framing, personality, other social
  or psychological attributes of a text.
- `B-TOPIC`: topic, intent, genre, domain, document or text categorisation, including OOD intent.
- `B-VERIFY`: NLI, entailment, claim or fact verification, factual consistency, hallucination or
  truthfulness detection, answerability.
- `B-REL`: relation, event, temporal, causal or entity-type classification over given mentions or pairs.
- `B-SEG`: labelling supplied units in context (sentence rhetorical roles, discourse relations,
  dialogue acts, argument components over given segments).
- `B-SCORE`: scalar or ordinal scoring (readability, essay or answer scoring, quality estimation,
  evaluation metrics and LLM judges validated against human ratings).
- `B-RANK`: ranking, reranking or selection among supplied candidates, pairwise preference, reward
  models evaluated on preference accuracy, task-specific answer selection.
- `B-MULTI`: multi-label or extreme-label coding (medical codes, TTPs, taxonomies, large tag sets).
- `B-DETECT`: provenance detection: machine-generated text, authorship, language identification,
  watermark detection.
- `B-OTHER`: any other bounded decision.

Non-bounded families:
- `N-GEN`: text generation (MT, summarisation, dialogue, simplification, style transfer, captioning,
  ASR, data-to-text).
- `N-REASON`: open-ended QA, reasoning, math, code generation, agents, tool use, planning.
- `N-IE`: span or structure extraction and parsing (NER, event or relation extraction that must find
  spans, semantic or syntactic parsing, entity linking into an open KB, keyphrase extraction).
- `N-RETR`: retrieval from an open corpus, retrieval-augmented generation, search.
- `N-LLM`: general LLM methods and science: pre-training, fine-tuning or alignment recipes,
  efficiency, interpretability, knowledge editing, and evaluations of general LLM capability or
  behaviour that are not a task-specific predictor (including general methods evaluated on
  multiple-choice benchmarks such as MMLU).
- `N-OTHER`: anything else (linguistic analysis, corpora without a predictive task, surveys,
  position papers, tools, front matter).

Assign the family of the task the paper is about, even when the paper fails E1 or E2 (for example
a benchmark of toxicity judgements that trains no predictor is still `B-SAFE`; the bounded
predictor and E2 fields carry the difference).

### `bounded_predictor` (`yes` / `no`)
`yes` only if the paper's **central contribution is a task-specific predictor** whose evaluated
output is bounded: it picks a label from a fixed inventory, places an item on a predefined scale,
or orders or selects among supplied candidates. The predictor may be a trained model, a training
scheme, a prompting or inference pipeline, a scoring metric or a judge, provided the paper
proposes it for a specific task (or task family) and evaluates its bounded outputs. Everything
else is `no`, including: resource papers whose models are only baselines; analyses; general LLM
methods evaluated on general benchmarks; generation, open QA, span extraction and open-corpus
retrieval.

### Eligibility (the selection criteria), each `yes` / `no` / `unclear`
- `E1` Bounded decision: the paper's evaluated predictor or judge selects from a fixed label set,
  predefined scale, candidate set or supplied ranking set. Span extraction, open retrieval and free
  generation fail.
- `E2` Meaningful replacement: the paper makes a substantive prediction claim about that bounded
  decision, so replacing the predictor tests the claim. `yes` for bounded-predictor papers, and also
  for resource papers that present a trained or prompted predictor for the task as a main result
  (e.g. a fine-tuned classifier on the new dataset, or a judge whose agreement with humans is
  reported). `no` when the bounded outputs only evaluate other systems (e.g. benchmarks that score
  general LLMs on multiple choice) or when the bounded step is incidental.
- `E3` Comparable evaluation: a fixed held-out test split (or fixed evaluation set) and a primary
  metric are reported, so the paper's number can be matched.
- `E4` Specified task: label inventory (or scale, or candidate set) and the input are defined well
  enough to prompt a text-only model. Non-text inputs fail.
- `E5` Feasible artifacts: evaluation text, labels and protocol are released (code/data link) or
  built from public datasets so they can be credibly reconstructed.

If `E1` or `E2` is `no`, set `E3`, `E4` and `E5` to `na`.

### `english_only` (`yes` / `no` / `unclear`)
`yes` if every evaluation example the predictor is scored on is English (a separable, named English
subset counts); `no` if any evaluated data are non-English, multilingual, code-mixed or
translation; `unclear` when neither abstract nor (if opened) PDF settles it. Standard English
benchmarks named in the abstract count as English.

### `evidence`
At most 40 words quoting or paraphrasing the decisive phrase(s).

## Derived quantities (computed, not labelled)

- **Bounded-predictor share**: `bounded_predictor = yes`.
- **Eligible (strict)**: `E1`–`E5` and `english_only` all `yes`. **Eligible (lenient)**: none of them
  `no` (`unclear` counts as passing).
- Frame shares use the Wilson 95% interval on the 300-paper uniform sample.

## Task families of the 24 audited papers (fixed with this rubric)

| Paper (repo id) | Family |
|---|---|
| LifeTox (`lifetox`), SMARTER (`smarter`), XSTest (`xstest`) | `B-SAFE` |
| EZ-STANCE (`ezstance`) | `B-AFFECT` |
| CICC (`cicc`), Edit Intent (`edit_intent`), SciPrompt (`sciprompt`), Navigating the Unknown (`nav_unknown`) | `B-TOPIC` |
| PiNose (`pinose`), FENICE (`fenice`), MISMATCHED (`mismatched`), INLI (`inli`) | `B-VERIFY` |
| ConTempo (`contempo`), Entity or Relation Embeddings (`relemb`) | `B-REL` |
| LegalSeg (`legalseg`), BIORC800 (`biorc800`) | `B-SEG` |
| FactPICO (`factpico`), Scalar Constructs (`scalar_constructs`), DREsS (`dress_new`), MedReadMe (`medreadme`) | `B-SCORE` |
| PRP-Graph (`prp_graph`), Explanation Ranking (`expl_rank`) | `B-RANK` |
| MADE (`made`), Security TTP Mapping (`ttp`) | `B-MULTI` |

## Output

Exactly one JSON object on a single line, no prose, with keys
`id, contribution, modality, task_family, bounded_predictor, E1, E2, E3, E4, E5, english_only,
pdf_checked, evidence`.

## Second labelling pass

A stratified 40-paper subset (20 drawn from papers the classifier labelled `bounded_predictor = yes`,
20 from `no`, seeded; all of a stratum if it has fewer than 20) is labelled a second time by an LLM
from the same title/abstract/PDF evidence, using a subset file that hides the classifier's labels.
Agreement is reported per field as raw agreement and Cohen's kappa, and the bounded-predictor share is
re-estimated with the second-pass labels as the reference (stratified correction of the classifier's
labels).
