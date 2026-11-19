"""task.yaml schema and train/val example loading (the test split is only reachable via hygiene)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from .ir import ChoiceNode, Demos, Option, Program

Metric = Literal["accuracy", "balanced_accuracy", "macro_f1", "micro_f1", "qwk"]
Objective = Literal["quality", "calibration", "cheap", "tokens"]
Condition = Literal["jev-zero", "jev-text-opt", "jev-program-opt", "jev-population-opt", "jev-tailored"]
CONDITIONS: tuple[str, ...] = ("jev-zero", "jev-text-opt", "jev-program-opt", "jev-population-opt", "jev-tailored")
DEFAULT_CONDITIONS: tuple[str, ...] = CONDITIONS[:3]  # run when none are requested; population search and jev-tailored are opt-in


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataSplits(_Strict):
    train: str
    val: str
    test: str


class JevConfig(_Strict):
    model: str = "typesafe/jev-1.13"
    max_rpm: int = 1800
    concurrency: int = 16
    cache_dir: str | None = None  # default: <runs_root>/.jev-cache


class SeedRef(_Strict):
    """An extra seed program for population search (§5.2), e.g. written by `/jev-seeds`."""

    kind: str  # seed kind reported per row, e.g. jev-native, decomposed
    path: str  # program YAML, relative to the task file


class PopulationConfig(_Strict):
    """`jev-population-opt` (§5.1): batched population search over per-node question variants."""

    size: int = 20  # variants requested per proposal (one OMP call per round)
    rounds: int = 4  # proposal rounds; question nodes ("slots") take turns
    elite: int = 4  # best variants of the slot shown to the proposer
    worst_examples: int = 8  # worst out-of-fold TRAIN examples shown to the proposer
    max_aux: int = 2  # auxiliary feature questions per variant (answers feed the readout)
    val_candidates: int = 6  # best programs by train CV scored on val (the seed variant always included)
    stack_max: int = 4  # variants per stacked readout (forward selection by train CV)
    demos: Demos | None = None  # demonstration spec a variant may opt into (`demos: true`)
    max_jev_cost: float | None = None  # no further rounds once the run's fresh Jev spend exceeds this ($)
    max_tokens: int = 32000  # proposer reply budget (tokens)
    seeds: list[SeedRef] = Field(default_factory=list)  # extra seed programs (union with the paper seed in round 0)

    @model_validator(mode="after")
    def _ranges(self) -> PopulationConfig:
        if self.size < 1 or self.rounds < 0 or self.elite < 1 or self.val_candidates < 1 or self.stack_max < 2:
            raise ValueError("population: size, elite, val_candidates >= 1; rounds >= 0; stack_max >= 2")
        if self.demos is not None and self.demos.nodes is not None:
            raise ValueError("population.demos must not list nodes (variants opt in)")
        return self


class TailoredConfig(_Strict):
    """`jev-tailored` (§5.3): GEPA adapted to the Jev decision model. Flags switch single components off
    for ablations: T1 rulebook, T2 continuous acceptance, T3 confusion targeting, T4 minimal pairs."""

    starts: list[str] = Field(default_factory=lambda: ["paper"])  # one GEPA search each: `paper` = the task's seed; a seed kind = its seed with the best zero-shot train score
    seeds: list[SeedRef] = Field(default_factory=list)  # candidate seeds of the non-paper starts
    rulebook: bool = True  # T1: Jev rulebook in every proposal prompt
    continuous_accept: bool = True  # T2; False: GEPA's default acceptance on `reflection_minibatch_size` examples
    accept_n: int = 64  # T2: TRAIN minibatch per iteration (class-balanced for macro metrics)
    accept_tol: float = 0.02  # T2: largest allowed drop of the minibatch primary metric
    confusion_target: bool = True  # T3; False: round-robin over question nodes
    minimal_pairs: int = 4  # T4: similar TRAIN inputs with different gold labels in the targeted pair (0 = off)
    max_examples: int = 8  # minibatch example records shown to the proposer
    max_jev_cost: float | None = 2.0  # fresh Jev $ per run; no further GEPA iteration once reached
    head: bool = True  # also score the best fitted candidate of each start with a readout head (§4.2)

    @model_validator(mode="after")
    def _ranges(self) -> TailoredConfig:
        if not self.starts or len(set(self.starts)) != len(self.starts):
            raise ValueError("tailored.starts must list distinct starts")
        if not 8 <= self.accept_n <= 128:
            raise ValueError("tailored.accept_n must be in 8..128")
        if not 0.0 <= self.accept_tol <= 0.2:
            raise ValueError("tailored.accept_tol must be in 0..0.2")
        if not 0 <= self.minimal_pairs <= 8 or self.max_examples < 1:
            raise ValueError("tailored.minimal_pairs must be in 0..8 and max_examples >= 1")
        missing = [s for s in self.starts if s != "paper" and s not in {ref.kind for ref in self.seeds}]
        if missing:
            raise ValueError(f"tailored.starts {missing} have no seeds of that kind")
        return self



class OptimizerConfig(_Strict):
    model: str = "current"
    reflection_model: str | None = None
    structural_model: str | None = None
    thinking: str | None = None
    max_metric_calls: int = 300
    max_reflection_cost: float | None = None
    reflection_minibatch_size: int = 3
    use_merge: bool = True
    seed: int = 0
    progressive: list[int] = Field(default_factory=lambda: [32, 128])
    fit_examples: int = 64  # train examples used for numerical fitting (§7)
    calibrate: bool = True  # temperature scaling on train
    max_program_chars: int | None = None  # default max(4000, 8 x seed)
    class_bias: bool = False  # fit per-option log-prob biases of Choice nodes on train (§7)
    fit_holdout: bool = False  # fit examples are held out of GEPA's trainset (reflection never sees them)
    reflection_pool: int | None = None  # cap on train examples GEPA samples for reflection (stratified prefix)
    max_component_chars: int | None = None  # per text slot limit for proposals (prompted, then retried/trimmed)
    ensemble_k: list[int] = Field(default_factory=list)  # representatives `ensemble-<k>` (§10)
    fit_parameters: bool = True  # False: no threshold/cutoff/bias search (with calibrate: false the program runs as written)
    readout: bool = False  # fit a readout head on every leaf question node (§4.2): lr, or ridge for qwk
    readout_folds: int = 5  # cross-validation folds for readout hyperparameters
    readout_group: str | None = None  # regex on example ids; its first group keeps CV folds grouped (e.g. documents)
    population: PopulationConfig = Field(default_factory=PopulationConfig)
    tailored: TailoredConfig = Field(default_factory=TailoredConfig)

    @model_validator(mode="after")
    def _ranges(self) -> OptimizerConfig:
        if self.max_component_chars is not None and self.max_component_chars < 200:
            raise ValueError("max_component_chars must be >= 200")
        if any(k < 2 for k in self.ensemble_k):
            raise ValueError("ensemble_k entries must be >= 2")
        if self.reflection_pool is not None and self.reflection_pool < 1:
            raise ValueError("reflection_pool must be >= 1")
        return self


class EmbeddingConfig(_Strict):
    """Local sentence-embedding model for knn demos (§4.1); not an LLM; revision pinned."""

    model: str = "BAAI/bge-small-en-v1.5"
    revision: str = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    cache_dir: str | None = None  # default: <runs_root>/.embed-cache


class EvaluationConfig(_Strict):
    command: str | None = None  # black-box evaluator: `{program}` and `{examples}` are substituted


class TaskConfig(_Strict):
    name: str
    description: str = ""
    labels: list[str]
    label_descriptions: dict[str, str] = Field(default_factory=dict)
    ordinal: bool = False
    label_groups: dict[str, list[str]] | None = None
    inputs: dict[str, str]
    data: DataSplits
    primary_metric: Metric = "accuracy"
    negative_label: str | None = None  # micro_f1 only: label excluded from averaging (e.g. VAGUE)
    objectives: list[Objective] = Field(default_factory=lambda: ["quality", "calibration", "cheap", "tokens"])
    jev: JevConfig = Field(default_factory=JevConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    seed_program: str | None = None
    seed_demos: Demos | None = None  # demonstrations added to the seed program (§4.1); the seed file stays verbatim
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    _base_dir: Path = PrivateAttr(default=Path("."))

    @model_validator(mode="before")
    @classmethod
    def _labels_mapping(cls, data: Any) -> Any:
        # `labels: {name: description}` is shorthand for labels + label_descriptions.
        if isinstance(data, dict) and isinstance(data.get("labels"), dict):
            data = dict(data)
            mapping = data.pop("labels")
            data["labels"] = list(mapping)
            data["label_descriptions"] = {**{k: str(v) for k, v in mapping.items() if v}, **data.get("label_descriptions", {})}
        return data

    @model_validator(mode="after")
    def _consistent(self) -> TaskConfig:
        labels = self.labels
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ValueError("task needs >= 2 distinct labels")
        if "id" in self.inputs or "label" in self.inputs:
            raise ValueError("`id` and `label` are reserved row keys and cannot be inputs")
        if self.primary_metric == "qwk" and not self.ordinal:
            raise ValueError("primary_metric qwk requires ordinal: true")
        if self.negative_label is not None and (self.primary_metric != "micro_f1" or self.negative_label not in labels):
            raise ValueError("negative_label must be a task label and requires primary_metric micro_f1")
        if self.label_groups:
            flat = [lab for group in self.label_groups.values() for lab in group]
            if sorted(flat) != sorted(labels):
                raise ValueError("label_groups must partition the labels")
        if "quality" not in self.objectives:
            raise ValueError("objectives must include quality")
        return self

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def split_path(self, split: str) -> Path:
        return (self._base_dir / getattr(self.data, split)).resolve()

    def resolved(self) -> dict[str, Any]:
        """Config as run into `config.yaml`: absolute data/seed paths so the run dir is self-contained."""
        data = self.model_dump(mode="json")
        data["data"] = {s: str(self.split_path(s)) for s in ("train", "val", "test")}
        if self.seed_program:
            data["seed_program"] = str((self._base_dir / self.seed_program).resolve())
        for ref in data["optimizer"]["population"]["seeds"] + data["optimizer"]["tailored"]["seeds"]:
            ref["path"] = str((self._base_dir / ref["path"]).resolve())
        return data


def load_task(path: str | Path) -> TaskConfig:
    path = Path(path)
    config = TaskConfig.model_validate(yaml.safe_load(path.read_text()))
    config._base_dir = path.resolve().parent
    return config


def task_from_data(data: dict[str, Any], base_dir: Path) -> TaskConfig:
    config = TaskConfig.model_validate(data)
    config._base_dir = base_dir
    return config


@dataclass(frozen=True)
class Example:
    id: str
    fields: dict[str, str]
    label: str
    split: str
    weight: float = 1.0  # per-example quality weight (class balance for macro metrics)
    extra: dict[str, Any] = field(default_factory=dict)


def parse_rows(text: str, config: TaskConfig, split: str, source: str) -> list[Example]:
    labels = set(config.labels)
    out: list[Example] = []
    seen: set[str] = set()
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        where = f"{source}:{n}"
        if "id" not in row or "label" not in row:
            raise ValueError(f"{where}: rows need `id` and `label`")
        rid, label = str(row["id"]), str(row["label"])
        if rid in seen:
            raise ValueError(f"{where}: duplicate id {rid!r}")
        if label not in labels:
            raise ValueError(f"{where}: label {label!r} not in task labels")
        missing = [f for f in config.inputs if f not in row]
        if missing:
            raise ValueError(f"{where}: missing input fields {missing}")
        seen.add(rid)
        fields = {f: str(row[f]) for f in config.inputs}
        extra = {k: v for k, v in row.items() if k not in fields and k not in ("id", "label")}
        out.append(Example(id=rid, fields=fields, label=label, split=split, extra=extra))
    return with_weights(out, config)


def with_weights(examples: list[Example], config: TaskConfig) -> list[Example]:
    """Class-balanced weights (mean 1 over the split) when the primary metric is macro-averaged."""
    if config.primary_metric not in ("balanced_accuracy", "macro_f1") or not examples:
        return examples
    counts: dict[str, int] = {}
    for ex in examples:
        counts[ex.label] = counts.get(ex.label, 0) + 1
    k, n = len(counts), len(examples)
    return [
        Example(ex.id, ex.fields, ex.label, ex.split, n / (k * counts[ex.label]), ex.extra) for ex in examples
    ]


def canonical_seed(config: TaskConfig) -> Program:
    """Direct Choice over the task labels using the task's label descriptions (§5 jev-zero)."""
    labels = config.labels
    desc = config.label_descriptions
    instructions = config.description or f"Classify the state into exactly one of: {', '.join(labels)}."
    node = ChoiceNode(
        kind="choice",
        instructions=instructions,
        state=list(config.inputs),
        options={lab: Option(criterion=desc.get(lab) or lab) for lab in labels},
    )
    return Program(labels=labels, inputs=dict(config.inputs), root="decide", nodes={"decide": node})


def seed_program(config: TaskConfig) -> Program:
    seed = Program.from_yaml((config.base_dir / config.seed_program).read_text()) if config.seed_program else canonical_seed(config)
    if config.seed_demos is not None:
        if seed.demos is not None:
            raise ValueError("seed_demos given but the seed program already has demos")
        seed = seed.model_copy(update={"demos": config.seed_demos.model_copy(deep=True)})
    return seed
