# TDM-Gauntlet

TDM-Gauntlet is the benchmark of the paper *Do We Still Need Fine-Tuned Classifiers? Swapping Typed Decision Models into 24 ACL-Venue Papers* ([arXiv:2611.20191](https://arxiv.org/abs/2611.20191)). It holds 46 English test sets from 24 papers published at ACL, EMNLP, NAACL and EACL in 2024–2026, together with the question asked on each test set, the paper's metric and the paper's published scores. The paper replaces each paper's predictor with a Typed Decision Model and asks how many of the papers' fine-tuned classifiers such a model can replace under each paper's own evaluation.

The code, prompts, test manifests, model responses and the `jev-opt` optimizer will follow in this repository.

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
