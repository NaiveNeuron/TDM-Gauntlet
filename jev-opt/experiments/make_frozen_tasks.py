"""Frozen-configuration round (final optimizer settings): data splits, seed program and task file per new set.

For every set in SETS this writes, without any Jev call:
  <dir>/data/{train,val,test}.jsonl + manifest.json   (gitignored; train/val from the paper's released
        train/dev data only, test = the gauntlet's frozen test set mapped through the adapter's `_state`)
  <dir>/seed.yaml       the gauntlet's frozen prompt as a jev-opt program (one Choice, verbatim)
  <dir>/task_v3.yaml    the v3 task file of smarter_implicit_hate/task_v3.yaml with only the per-task
                        fields replaced (name, labels, inputs, metric, negative label, readout grouping,
                        max_metric_calls = 14 x |val|); the optimizer block is otherwise byte-for-byte v3
and checks that the seed program's Decisions request equals the gauntlet adapter's request for every
test row (state, order-sensitive, and question object), with jev-opt's executor.

Sampling (pre-registered): train = 500 rows (the first 400 of the optimizer's stratified
order are the fit slice), val = 1,000 rows (or the whole held-out pool when smaller), seed 0. Class mix
follows the test set's design: class-balanced samples where the test split is class-balanced by
construction (LifeTox, SciPrompt S2ORC, SMARTER), label-proportional otherwise (MATRES).

    uv run --project jev-opt/python python jev-opt/experiments/make_frozen_tasks.py [set ...]
"""

from __future__ import annotations

import json
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import REPO, gauntlet_task, looks_english, stratified, test_rows, write_splits  # noqa: E402

CODE = REPO  # the jev-opt executor used for the request check
TEMPLATE = HERE / "smarter_implicit_hate" / "task_v3.yaml"
SEED = 0
N_TRAIN, N_VAL = 500, 1000

# set id -> (experiment dir, gauntlet task, prompt file, run-name stem, class design)
SETS: dict[str, dict[str, Any]] = {
    "contempo/matres": {"dir": "contempo_matres", "stem": "contempo-matres", "prompt": "contempo_matres.json",
                        "balanced": False, "group": "^fit:([^:]+):", "metric": "micro_f1", "negative": "VAGUE"},
    "lifetox": {"dir": "lifetox", "stem": "lifetox", "prompt": "lifetox.json", "balanced": True, "metric": "macro_f1"},
    "sciprompt/s2orc": {"dir": "sciprompt_s2orc", "stem": "sciprompt-s2orc", "prompt": "sciprompt_s2orc.json",
                        "balanced": True, "metric": "accuracy"},
    "smarter/hatexplain": {"dir": "smarter_hatexplain", "stem": "smarter-hatexplain", "prompt": "smarter_hatexplain.json",
                           "balanced": True, "metric": "macro_f1"},
    "smarter/latent_hate": {"dir": "smarter_latent_hate", "stem": "smarter-latent-hate", "prompt": "smarter_latent_hate.json",
                            "balanced": True, "metric": "macro_f1"},
}


def balanced(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Deterministic class-balanced sample of `n` rows: n // k per class, the remainder to the first
    classes in sorted order; a class with too few rows gives all it has."""
    rng = random.Random(seed)
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["label"], []).append(r)
    labels = sorted(by)
    quota = {lab: n // len(labels) + (i < n % len(labels)) for i, lab in enumerate(labels)}
    picked = [r for lab in labels for r in rng.sample(by[lab], min(quota[lab], len(by[lab])))]
    rng.shuffle(picked)
    return picked


def sample(rows: list[dict], n: int, design_balanced: bool) -> list[dict]:
    return balanced(rows, n, SEED) if design_balanced else stratified(rows, n, SEED)


def state_row(task: Any, item: Any, rid: str) -> dict:
    return {"id": rid, **task._state(item)[0], "label": item.gold}


# --------------------------------------------------------------------------------------------
# train/val pools per set (released train/dev data only; never the test split)


def pools_dev_v3(name: str, cfg: dict) -> tuple[list[dict], list[dict], dict]:
    """MATRES / SMARTER: the gauntlet's frozen v3 development samples (scripts/prepare_v3_dev.py):
    `fit` part (train docs / the authors' train part) -> train pool, `dev` part -> val pool."""
    sub = name.split("/")[1]
    task = gauntlet_task(f"dev_v3/{name.split('/')[0]}_{sub}")
    fit, dev = [], []
    for it in task.items():
        part = it.id.split(":", 1)[0] if name.startswith("contempo") else it.id.split("-")[-2]
        (fit if part == "fit" else dev).append(state_row(task, it, it.id))
        assert part in ("fit", "dev"), it.id
    return fit, dev, {"source": f"gauntlet dev_v3/{name.split('/')[0]}_{sub} (scripts/prepare_v3_dev.py): fit part -> train, dev part -> val"}


def pools_lifetox(name: str, cfg: dict) -> tuple[list[dict], list[dict], dict]:
    """LifeTox: the release has no dev split; LifeTox_train.jsonl is split (seed 0, after dropping rows whose
    question+answer pair occurs in the test set) into a val pool and a disjoint train pool."""
    task = gauntlet_task("lifetox")
    from gauntlet.task import Item

    test_pairs = {(it.fields["query"], it.fields["response"]) for it in task.items()}
    rows, dropped = [], Counter()
    for i, line in enumerate((REPO / "data" / "raw" / "lifetox" / "LifeTox_train.jsonl").read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        if (r["query"], r["response"]) in test_pairs:
            dropped["in_test"] += 1
            continue
        gold = "safe" if str(r["is_safe"]).strip().lower() == "true" else "unsafe"
        rows.append(state_row(task, Item(id="", gold=gold, fields={"query": r["query"], "response": r["response"]}), f"lifetox-train-{i:05d}"))
    rng = random.Random(SEED)
    rng.shuffle(rows)
    half = len(rows) // 2  # val drawn from the first half of the shuffled file, train from the second
    return rows[half:], rows[:half], {"source": "LifeTox_train.jsonl (HF mbkim/LifeTox): shuffled (seed 0), halves -> train / val pools",
                                      "dropped": dict(dropped), "train_file_rows": len(rows) + sum(dropped.values())}


def pools_s2orc(name: str, cfg: dict) -> tuple[list[dict], list[dict], dict]:
    """SciPrompt S2ORC: the release's train_dataset.txt / train_labels.txt (parsed like the test files);
    rows whose abstract occurs in the test set, short (< 30 tokens as the test filter) or non-English
    (automatic screen) are dropped; shuffled (seed 0), halves -> train / val pools."""
    from gauntlet.task import Item
    from gauntlet.tasks import sciprompt as sp

    task = gauntlet_task("sciprompt/s2orc")
    test_texts = {it.fields["abstract"] for it in task.items()}
    with zipfile.ZipFile(sp.RAW_ZIP) as zf:
        texts, labels = sp._lines(zf, "data/S2ORC/train_dataset.txt"), sp._lines(zf, "data/S2ORC/train_labels.txt")
    assert len(texts) == len(labels)
    allowed = set(task.labels)
    rows, dropped = [], Counter()
    for i, (text, gold) in enumerate(zip(texts, labels)):
        if gold not in allowed:
            dropped["unknown_label"] += 1
        elif text in test_texts:
            dropped["in_test"] += 1
        elif len(text.split()) < 30:
            dropped["short"] += 1
        elif not looks_english(text):
            dropped["non_english"] += 1
        else:
            rows.append(state_row(task, Item(id="", gold=gold, fields={"abstract": text}), f"s2orc-train-{i:05d}"))
    rng = random.Random(SEED)
    rng.shuffle(rows)
    half = len(rows) // 2
    return rows[half:], rows[:half], {"source": "sciprompt_data.zip data/S2ORC/train_dataset.txt + train_labels.txt: shuffled (seed 0), halves -> train / val pools",
                                      "dropped": dict(dropped), "train_file_rows": len(texts)}


POOLS = {"contempo/matres": pools_dev_v3, "lifetox": pools_lifetox, "sciprompt/s2orc": pools_s2orc,
         "smarter/hatexplain": pools_dev_v3, "smarter/latent_hate": pools_dev_v3}


# --------------------------------------------------------------------------------------------
# seed program and task file


def seed_yaml(name: str, cfg: dict, task: Any) -> tuple[str, list[str], dict[str, str], list[str], str]:
    prompt = json.loads((REPO / "prompts" / cfg["prompt"]).read_text())
    q = prompt["question"]
    assert q["type"] == "choice" and set(q) == {"type", "instructions", "criteria"}, (name, set(q))
    if "_label_map" in prompt:
        assert prompt["_label_map"] == {k: k for k in q["criteria"]}, name
    item = task.items()[0]
    req = list(task.requests(item))
    assert len(req) == 1 and len(req[0].questions) == 1
    qid = req[0].key
    assert list(req[0].questions) == [qid] and req[0].questions[qid] == q
    state = list(req[0].state)
    labels = list(q["criteria"])
    program = {
        "version": 1, "labels": labels, "inputs": {f: "" for f in state}, "shared_state": False, "temperature": 1.0,
        "root": qid,
        "nodes": {qid: {"kind": "choice", "instructions": q["instructions"], "state": state,
                        "options": {lab: {"criterion": crit} for lab, crit in q["criteria"].items()}}},
    }
    head = (f"# jev-zero seed = the gauntlet's frozen prompt (prompts/{cfg['prompt']}), verbatim; generated by\n"
            f"# make_frozen_tasks.py, do not hand-edit. One Choice, question id `{qid}`, state {{{', '.join(state)}}} -- the gauntlet request.\n")
    return head + yaml.safe_dump(program, sort_keys=False, allow_unicode=True, width=100), labels, q["criteria"], state, qid


def task_yaml(name: str, cfg: dict, criteria: dict[str, str], state: list[str], n_val: int) -> str:
    t = yaml.safe_load(TEMPLATE.read_text())
    out: dict[str, Any] = {"name": cfg["stem"], "labels": dict(criteria), "inputs": {f: "" for f in state}}
    if cfg.get("negative"):
        out["negative_label"] = cfg["negative"]
    for k, v in t.items():
        if k not in ("name", "labels", "inputs", "negative_label"):
            out[k] = v
    out["primary_metric"] = cfg["metric"]
    opt = out["optimizer"]
    opt["max_metric_calls"] = 14 * n_val
    if cfg.get("group"):
        opt["readout_group"] = cfg["group"]
    else:
        opt.pop("readout_group", None)
    head = ("# Frozen-configuration round (final optimizer settings); generated by make_frozen_tasks.py, do not\n"
            "# hand-edit. The v3 task file (smarter_implicit_hate/task_v3.yaml) with only the per-task\n"
            "# fields replaced: name, labels (= the gauntlet prompt's criteria), inputs, primary metric, negative label,\n"
            "# readout CV grouping, max_metric_calls = 14 x |val|. Seeds: 3 jev-native + 3 decomposed (/jev-seeds).\n")
    return head + yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=100)


# --------------------------------------------------------------------------------------------
# request equivalence with the gauntlet


def check_requests(task_path: Path, task: Any) -> int:
    sys.path.insert(0, str(CODE / "jev-opt" / "python" / "src"))
    from jev_opt.execute import run_examples
    from jev_opt.ir import load_decider
    from jev_opt.jev import Decision
    from jev_opt.task import load_task, parse_rows

    config = load_task(task_path)
    program = load_decider((task_path.parent / "seed.yaml").read_text())
    test = parse_rows((task_path.parent / "data" / "test.jsonl").read_text(), config, "test", "test.jsonl")

    class Recorder:
        concurrency = 1

        def __init__(self) -> None:
            self.seen: list[tuple[Any, dict]] = []

        def decide(self, state: Any, questions: dict[str, Any], context: dict[str, Any] | None = None) -> Decision:
            self.seen.append((state, questions))
            answers = {q: {"probabilities": {lab: 1 / len(config.labels) for lab in config.labels}, "confidence": 0.5} for q in questions}
            return Decision(answers=answers, snapshot="recorder", input_tokens=0, cost=0.0, latency_s=0.0, cached=True)

    items = {it.id: it for it in task.items()}
    bad = 0
    for ex in test:
        rec = Recorder()
        run_examples(program, [ex], rec)
        req = list(task.requests(items[ex.id]))
        want = [(r.state, r.questions) for r in req]
        got = [(s, q) for s, q in rec.seen]
        same = len(got) == len(want) and all(
            list(gs.items()) == list(ws.items()) and json.dumps(gq, sort_keys=False) == json.dumps(wq, sort_keys=False)
            for (gs, gq), (ws, wq) in zip(got, want))
        bad += not same
    return bad


def main() -> int:
    names = sys.argv[1:] or list(SETS)
    for name in names:
        cfg = SETS[name]
        d = HERE / cfg["dir"]
        d.mkdir(exist_ok=True)
        task = gauntlet_task(name)
        test = test_rows(task, name, lambda it, st: {"id": it.id, **st, "label": it.gold})
        train_pool, val_pool, meta = POOLS[name](name, cfg)
        train = sample(train_pool, N_TRAIN, cfg["balanced"])
        val = sample(val_pool, N_VAL, cfg["balanced"])
        write_splits(d / "data", {"train": train, "val": val, "test": test},
                     {**meta, "gauntlet_task": name, "pools": {"train": len(train_pool), "val": len(val_pool)},
                      "design": "class-balanced" if cfg["balanced"] else "label-proportional", "seed": SEED})
        seed_text, labels, criteria, state, qid = seed_yaml(name, cfg, task)
        (d / "seed.yaml").write_text(seed_text)
        (d / "task_v3.yaml").write_text(task_yaml(name, cfg, criteria, state, len(val)))
        bad = check_requests(d / "task_v3.yaml", task)
        print(f"{name}: train {len(train)} val {len(val)} test {len(test)}; request mismatches {bad}/{len(test)}")
        if bad:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
