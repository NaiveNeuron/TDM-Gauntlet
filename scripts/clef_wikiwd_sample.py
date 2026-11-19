"""Optional follow-up to the Clef comparison: Clef on a fixed random sample of Wiki-WD sentences (~8,000 edges).

Not launched by default. The full Wiki-WD cohort (360,052 sentences, 739,977 edges) was stopped for Clef; this
script scores a sample instead. Unit = sentence (a gauntlet item with all of its edges), so every request is
byte-identical to the full frozen run's request (same packing, same 65,536-token guard; cached responses from
the stopped run are reused under their original hashes on the same device identity).

Sample rule (fixed before any run): numpy `default_rng(20261007).permutation` over the frozen
item order; take sentences in permuted order until the running edge count reaches 8,000; score every edge of
the chosen sentences with the task's own metric (micro P/R/F1, P0 excluded). Jev's (and Clef-flash's, when its
full run exists) scores on the same sentences are read from their recorded full-run predictions.

  uv run python scripts/clef_wikiwd_sample.py plan                      # dry run: sample file, pins; no model load
  uv run --extra clef python scripts/clef_wikiwd_sample.py run --model clef --device cuda:0 [--device cuda:1]
  uv run python scripts/clef_wikiwd_sample.py report                    # results/clef_wikiwd_sample/report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import cached_property
from pathlib import Path

import numpy as np

from gauntlet.task import REPO, Item
from gauntlet.tasks.relemb import WikiWDTask

SEED = 20261007
TARGET_EDGES = 8000
MAX_LENGTH = 65536  # Wiki-WD context guard of the zero-shot Clef runs (raised from 32,000; Wiki-WD requests need up to 50,437 tokens)
OUT = REPO / "results" / "clef_wikiwd_sample"
SAMPLE = OUT / "sample.json"
RUNS = REPO / "runs"
PINS = {"clef": "Cloudflare/clef@2f3de3dd85f379784083b0814d997ab627200f0c",
        "clef-flash": "Cloudflare/clef-flash@17f0b0ad64efb65d273590632833508766b2aae6"}


def draw(items: list[Item]) -> list[str]:
    order = np.random.default_rng(SEED).permutation(len(items))
    chosen, edges = [], 0
    for i in order:
        if edges >= TARGET_EDGES:
            break
        chosen.append(int(i))
        edges += len(items[i].gold)
    return [items[i].id for i in sorted(chosen)]


class WikiWDSample(WikiWDTask):
    """The frozen Wiki-WD task restricted to the sampled sentences (own output directory)."""

    name = "relemb/wikiwd_sample"

    @cached_property
    def sample_ids(self) -> set[str]:
        return set(json.loads(SAMPLE.read_text())["ids"])

    def items(self) -> list[Item]:
        return [it for it in super().items() if it.id in self.sample_ids]


def cmd_plan() -> None:
    items = WikiWDTask().items()
    ids = draw(items)
    by_id = {it.id: it for it in items}
    edges = sum(len(by_id[i].gold) for i in ids)
    data = {"seed": SEED, "target_edges": TARGET_EDGES, "rule": __doc__.split("Sample rule")[1].split("\n\n")[0].strip(),
            "population": {"sentences": len(items), "edges": sum(len(it.gold) for it in items)},
            "n_sentences": len(ids), "n_edges": edges,
            "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(), "ids": ids}
    if SAMPLE.exists() and json.loads(SAMPLE.read_text())["ids"] != ids:
        raise SystemExit(f"{SAMPLE} differs from the rule's sample; the sample is fixed")
    OUT.mkdir(parents=True, exist_ok=True)
    SAMPLE.write_text(json.dumps(data, indent=1) + "\n")
    print(json.dumps({k: v for k, v in data.items() if k != "ids"}, indent=1))
    for model, pin in PINS.items():
        mid, rev = pin.split("@")
        snap = REPO / "data" / "cache" / "huggingface" / f"models--{mid.replace('/', '--')}" / "snapshots" / rev
        print(f"{model}: weights {'present' if any(snap.glob('*.safetensors')) else 'missing'} at {snap}; "
              f"output runs/{model}/{WikiWDSample.name}; cache data/cache/{model}; guard {MAX_LENGTH}")


def cmd_run(model: str, devices: list[str]) -> None:
    from gauntlet.clef import MODELS, ClefClient, ClefMultiGpuClient
    from gauntlet.runner import run_task

    cache = REPO / "data" / "cache" / model
    if len(devices) == 1:
        client = ClefClient(cache, device=devices[0], max_length=MAX_LENGTH, model=MODELS[model])
        result = run_task(WikiWDSample(), client=client, run_root=RUNS / model)
    else:
        client = ClefMultiGpuClient(cache, devices=devices, max_length=MAX_LENGTH, model=MODELS[model])
        try:
            result = run_task(WikiWDSample(), client=client, run_root=RUNS / model)
        finally:
            client.close()
    print(json.dumps(result["scores"], indent=1))


def cmd_report() -> None:
    task = WikiWDSample()
    items = task.items()
    ids = {it.id for it in items}
    out = {"seed": SEED, "n_sentences": len(items), "n_edges": sum(len(it.gold) for it in items), "systems": {}}
    sources = {"jev": RUNS / "relemb" / "wikiwd", **{m: RUNS / m / "relemb" / "wikiwd" for m in PINS},
               **{f"{m} (sample run)": RUNS / m / WikiWDSample.name for m in PINS}}
    for name, d in sources.items():
        path = next((p for p in (d / "predictions.jsonl", d / "predictions.jsonl.gz") if p.exists()), None)
        if path is None:
            continue
        import gzip
        opener = gzip.open if path.suffix == ".gz" else open
        preds = {}
        with opener(path, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if r["item_id"] in ids:
                    preds[r["item_id"]] = r["prediction"]
        if len(preds) == len(ids):
            out["systems"][name] = {k: v for k, v in task.score(items, preds).items() if k != "paper_p_r_f1"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["plan", "run", "report"])
    ap.add_argument("--model", choices=sorted(PINS), default="clef")
    ap.add_argument("--device", action="append")
    a = ap.parse_args()
    if a.command == "plan":
        cmd_plan()
    elif a.command == "run":
        if not a.device:
            ap.error("run needs --device")
        cmd_run(a.model, a.device)
    else:
        cmd_report()


if __name__ == "__main__":
    main()
