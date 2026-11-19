"""Coverage check A, step 2: classify each sampled paper against data/coverage/rubric.md.

Every LLM call goes through the omp harness (`omp -p --no-session`), with the model and thinking
level pinned below and only the `read` tool enabled (for the paper PDF when the abstract is
ambiguous). Raw outputs and parsed labels are appended to data/coverage/labels_llm.jsonl; papers
already labelled are skipped, so the script can be resumed.

Usage: uv run python scripts/coverage_classify.py [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUBRIC = ROOT / "data/coverage/rubric.md"
SAMPLE = ROOT / "data/coverage/sample.jsonl"
OUT = ROOT / "data/coverage/labels_llm.jsonl"

MODEL = "anthropic/claude-opus-5-5"
THINKING = "medium"
KEYS = ["id", "contribution", "modality", "task_family", "bounded_predictor", "E1", "E2", "E3", "E4",
        "E5", "english_only", "pdf_checked", "evidence"]

PROMPT = """You are labelling one ACL Anthology paper for a coverage study. Apply the rubric below exactly.

<rubric>
{rubric}
</rubric>

<paper>
id: {id}
venue volume: {year} {volume}
title: {title}
abstract: {abstract}
pdf: {pdf}
</paper>

Reply with exactly one JSON object on a single line (keys: {keys}) and nothing else."""


def parse(text: str) -> dict | None:
    for line in reversed(text.strip().splitlines()):
        line = line.strip().removeprefix("```json").removesuffix("```").strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if all(k in obj for k in KEYS):
                return obj
    return None


def classify(rec: dict, rubric: str, cwd: str) -> dict:
    prompt = PROMPT.format(rubric=rubric, keys=", ".join(KEYS), **rec)
    cmd = ["omp", "-p", "--no-session", "--no-skills", "--no-rules", "--no-lsp", "--no-title",
           "--tools=read", f"--model={MODEL}", f"--thinking={THINKING}", "--max-time=15m", prompt]
    out = ""
    for attempt in range(3):
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=1200)
        out = p.stdout
        labels = parse(out)
        if labels is not None and labels["id"] == rec["id"]:
            return {"id": rec["id"], "labels": labels, "model": MODEL, "thinking": THINKING,
                    "attempts": attempt + 1, "raw": out.strip()[-4000:],
                    "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    raise RuntimeError(f"{rec['id']}: no valid label after 3 attempts; last output: {out[-500:]!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    rubric = RUBRIC.read_text()
    sample = [json.loads(l) for l in SAMPLE.open()]
    done = {json.loads(l)["id"] for l in OUT.open()} if OUT.exists() else set()
    todo = [r for r in sample if r["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to classify", flush=True)
    lock = threading.Lock()
    with tempfile.TemporaryDirectory() as cwd, ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(classify, r, rubric, cwd): r["id"] for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                row = fut.result()
            except Exception as exc:  # noqa: BLE001 - report and continue; rerun resumes
                print(f"FAILED {futs[fut]}: {exc}", flush=True)
                continue
            with lock, OUT.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            lab = row["labels"]
            print(f"[{i}/{len(todo)}] {row['id']} {lab['task_family']} bp={lab['bounded_predictor']}",
                  flush=True)


if __name__ == "__main__":
    main()
