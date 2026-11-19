# Licences

## Code

The code in this repository (`src/`, `scripts/`, `tests/`, `jev-opt/`) is released under the Apache License 2.0
(`LICENSE`).

## Test sets

This repository does not contain the test sets. Each `scripts/prepare_<task>.py` downloads its dataset from the
original source, checks the SHA-256 hashes recorded in `data/manifests/<task>.json` and writes the test set to
`data/frozen/<task>/`. Each dataset keeps the licence and terms of its source. Where a source requires a consent form,
a survey or accepted access conditions, you must complete it yourself before you run the script.

A few small files in `data/frozen/` ship with the repository because the scripts read them: the ID and hash
lists that `scripts/prepare_contempo.py` checks its rebuilt rows against, the raw-file hashes that
`scripts/prepare_scalar_constructs.py` checks, and the non-English exclusion lists of `scripts/prepare_ezstance.py`,
`scripts/prepare_made.py`, `scripts/prepare_smarter.py` and `scripts/prepare_relemb.py`. They hold IDs, row indices,
hashes and exclusion reasons, plus gold labels of redistributable HateXplain posts and short quotes from CC BY-SA
Wikipedia sentences.

The frozen files take one of two forms:

- **Full rows**: the source licence allows redistribution, so the prepare script writes the item text into the frozen
  file.
- **IDs and hashes**: the source states no licence or does not allow redistribution, so the frozen file holds only
  item IDs and SHA-256 hashes, plus gold labels for some tasks. The task loader rebuilds the text from the original
  download at run time and checks every hash.

Files derived from CC BY-NC or CC BY-NC-SA sources stay non-commercial, and files derived from share-alike sources stay
under the same licence. The table summarises the licence fields of the manifests; the manifests give the full record.

| Task | Source and licence | Frozen form |
|---|---|---|
| `biorc800` | LLM-SSC release: no licence stated; abstracts from the PubMed Central Open Access subset (mixed licences) | IDs and hashes |
| `cicc` | Code repository MIT. ACID: no licence file. ATIS: LDC-derived. BANKING77 CC BY 4.0, CLINC150 CC BY 3.0, HWU64 CC BY 4.0, MTOD CC BY-SA 4.0 (from the datasets' public cards) | IDs and hashes |
| `contempo` | MATRES and TB-Dense annotations: no licence stated. TimeBank 1.2 (LDC2006T08) and TempEval-3 texts: copyrighted news, no licence stated | IDs and hashes |
| `dress_new` | DREsS: MIT according to the paper's datasheet; access requires a consent form whose terms forbid sharing | IDs and hashes |
| `edit_intent` | Re3-Sci annotations and source corpora: CC BY-NC 4.0 | Full rows |
| `expl_rank` | e-SNLI: MIT; SNLI premise and hypothesis text: CC BY-SA 4.0 | Full rows |
| `ezstance` | No licence stated; tweets (Twitter/X terms) | IDs and hashes |
| `factpico` | Annotations CC BY 4.0, repository MIT; abstracts from Evidence Inference 2.0 (PubMed Central Open Access subset) | Full rows |
| `fenice` | Story-SummEval release and T5 claim extractor: CC BY-NC-SA 4.0. Stories: public-domain texts from Project Gutenberg and Wikisource, rebuilt from those sources and hash-checked | Full rows (summaries, labels, claims) |
| `inli` | CC BY-SA 4.0 | Full rows |
| `legalseg` | Apache-2.0; the canonical copy is access-gated on Hugging Face | IDs and hashes |
| `lifetox` | MIT | Full rows |
| `made` | Dataset card MIT; openFDA MAUDE reports (CC0 1.0 under the openFDA terms); access gated by a contact-sharing agreement on Hugging Face | IDs and hashes |
| `medreadme` | No licence stated; sources include CC BY journals, Wikipedia (CC BY-SA) and proprietary Merck Manuals | IDs and hashes |
| `mismatched` | CC BY-SA 4.0 | Full rows |
| `nav_unknown` | CLINC150 CC BY 3.0; BANKING77 CC BY 4.0 | Full rows |
| `pinose` | No licence stated | IDs and hashes |
| `prp_graph` | CORD-19 documents: CORD-19 Dataset License Agreement (text and data mining only, no redistribution). TREC-COVID topics and relevance judgments: CC BY-SA 4.0 (BEIR card) | Document IDs, truncation offsets and hashes; topics and judgments in full |
| `relemb/wikiwd` | Wikipedia-Wikidata relations (Sorokin and Gurevych, 2017): CC BY-SA 4.0 | Full rows |
| `relemb/nytfb` | No licence in the RECON release; sentences from the New York Times Annotated Corpus (LDC2008T19) | IDs and hashes |
| `scalar_constructs` | No licence stated; includes WiscAds 2008 storyboards under the Wisconsin Advertising Project's academic-use terms | IDs and hashes |
| `sciprompt/arxiv` | No licence stated | IDs and hashes |
| `sciprompt/s2orc` | No licence stated; Semantic Scholar API terms do not grant redistribution | IDs and hashes |
| `sciprompt/sdpra` | SDPRA-2021 shared-task data: CC BY 4.0 | Full rows |
| `sciprompt/emerging_nlp` | Apache-2.0 | Full rows |
| `smarter/hatexplain` | HateXplain: MIT | Full rows |
| `smarter/latent_hate`, `smarter/implicit_hate` | Latent Hatred corpus: CC BY 4.0, repository MIT; the authors ask users to complete a survey before download; tweets | IDs and hashes |
| `ttp` | Security-TTP-Mapping: CC BY 4.0. Technique names from MITRE ATT&CK Enterprise v12.0 | Full rows |
| `xstest` | Prompts CC BY 4.0; model completions under the licences of Meta, Mistral and OpenAI | IDs and hashes |

## Prompts

The typed questions in `prompts/` quote label definitions, annotation guidelines and prompts from the cited papers and
their dataset documentation. The `_source` field of each file names its sources. MITRE ATT&CK
technique names (in `prompts/ttp_v3.json`, and in `data/frozen/ttp/candidates.jsonl` once `scripts/prepare_ttp.py` has
built it) are © The MITRE Corporation and are reproduced under the ATT&CK Terms of Use.

## Model responses

The repository does not include the responses of Jev or of any other model.
