"""Run a frozen task against a decision model and score it.

Outputs, per task, under the selected run root:
  responses*.jsonl.gz  one row per request (sharded by size, each <90 MB gzip): item_id, key, request_hash, response, optional execution_device;
                       rows already stored by a run the task names in `reuses_responses_of` are not repeated
  predictions.jsonl  item_id, gold, prediction
  scores.json      task metrics
  manifest.json    model snapshots, prompt hash, counts, tokens, cost, git head
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx

from .jev import JevClient, load_api_key, request_body
from .task import REPO, Task, read_jsonl, write_jsonl

RUNS = REPO / "runs"
CACHE = REPO / "data" / "cache" / "jev"
# Shards rotate on uncompressed JSON bytes (row sizes vary ~100x between single-question
# and multi-question tasks); JSON of Jev answers compresses >4x, so 160 MB raw stays far
# under the 100 MB per-file git-hosting limit. Every shard's gzip size is asserted below.
SHARD_RAW_BYTES = 160_000_000
SHARD_MAX_GZ_BYTES = 90_000_000


def _git_head() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    return out.stdout.strip()


async def _run_requests(
    client: Any, http: httpx.AsyncClient, bodies: list[dict[str, Any]], label: str
) -> list[dict[str, Any]]:
    # Bound local encoding windows; the client buckets lengths and caps padded
    # GPU tokens. Do not queue one coroutine per relation-classification item.
    if getattr(client, "local", False):
        if hasattr(client, "decide_windows"):
            return await client.decide_windows(bodies, label)
        results = []
        window = client.batch_size * 4
        for first in range(0, len(bodies), window):
            results.extend(await client.decide_many(http, bodies[first:first + window]))
            print(f"[{label}] {len(results)}/{len(bodies)}", file=sys.stderr, flush=True)
        return results
    done = 0
    total = len(bodies)
    start = time.monotonic()
    results: list[dict[str, Any] | None] = [None] * total

    async def one(i: int) -> None:
        nonlocal done
        results[i] = await client.decide(http, bodies[i])
        done += 1
        if done % 200 == 0 or done == total:
            rate = done / max(time.monotonic() - start, 1e-9)
            print(f"[{label}] {done}/{total} ({rate:.1f} req/s)", file=sys.stderr, flush=True)

    await asyncio.gather(*(one(i) for i in range(total)))
    return [r for r in results if r is not None]


def run_task(
    task: Task,
    concurrency: int = 16,
    limit: int | None = None,
    *,
    client: Any = None,
    run_root: Path = RUNS,
) -> dict[str, Any]:
    items = task.items()
    if limit is not None:
        items = items[:limit]
    if client is None:
        client = JevClient(cache_dir=CACHE, api_key=load_api_key(REPO / ".env"), concurrency=concurrency)
    started = time.monotonic()
    local = getattr(client, "local", False)

    answers: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    snapshots: Counter[str] = Counter()
    cost = 0.0
    input_tokens = 0
    n_requests = 0
    rows = []
    # Adaptive tasks (e.g. Swiss-system ranking) choose later requests from earlier
    # answers, so requests are issued in rounds until no item asks for more. All rounds
    # share one event loop: the client's semaphore and pacing lock are loop-bound.
    round_no = 0
    inference_seconds = 0.0
    n_truncated_requests = 0
    # Per-device inference_seconds/new_input_tokens count only fresh executions.
    # Saved response timings are historical evidence, never this cohort's compute.
    per_device: dict[str, dict[str, int | float]] = {}
    # API backends that substitute refused answers (gauntlet.luna) count refusals, rejections and saturation.
    audit_fn = getattr(client, "answer_audit", None)
    answer_audit: Counter[str] = Counter()

    async def run_rounds() -> None:
        nonlocal round_no, cost, input_tokens, n_requests, inference_seconds, n_truncated_requests
        async with httpx.AsyncClient() as http:
            while True:
                reqs = [r for item in items for r in task.next_requests(item, answers[item.id])]
                if not reqs:
                    return
                round_no += 1
                bodies = [request_body(r.state, r.questions, client.model) for r in reqs]
                records = await _run_requests(client, http, bodies, f"{task.name} round {round_no}")
                for req, rec in zip(reqs, records):
                    resp = rec["response"]
                    answers[req.item_id][req.key] = resp["answers"]
                    if audit_fn is not None:
                        answer_audit.update(audit_fn(resp))
                    snapshots[resp.get("model", "?")] += 1
                    usage = resp.get("usage", {})
                    cost += float(usage.get("cost", 0.0))
                    input_tokens += int(usage.get("input_tokens", 0))
                    audit = resp.get("audit", {})
                    n_truncated_requests += bool(audit.get("truncated", False))
                    cache_hit = rec.get("cache_hit", False)
                    native_seconds = float(audit.get("inference_seconds", 0.0))
                    if not cache_hit:
                        inference_seconds += native_seconds
                    execution_device = rec.get("execution_device")
                    if local:
                        device = execution_device or getattr(client, "device", None)
                        if device is not None:
                            compute = per_device.setdefault(device, {
                                "n_requests": 0,
                                "n_cached_requests": 0,
                                "n_new_requests": 0,
                                "input_tokens": 0,
                                "cached_input_tokens": 0,
                                "new_input_tokens": 0,
                                "inference_seconds": 0.0,
                                "cached_response_inference_seconds": 0.0,
                            })
                            tokens = int(usage.get("input_tokens", 0))
                            compute["n_requests"] += 1
                            compute["input_tokens"] += tokens
                            if cache_hit:
                                compute["n_cached_requests"] += 1
                                compute["cached_input_tokens"] += tokens
                                compute["cached_response_inference_seconds"] += native_seconds
                            else:
                                compute["n_new_requests"] += 1
                                compute["new_input_tokens"] += tokens
                                compute["inference_seconds"] += native_seconds
                    row = {"item_id": req.item_id, "key": req.key, "request_hash": rec["request_hash"], "response": resp}
                    if execution_device is not None:
                        row["execution_device"] = execution_device
                    rows.append(row)
                n_requests += len(reqs)

    asyncio.run(run_rounds())

    predictions = {item.id: task.predict(item, answers[item.id]) for item in items}
    scores = task.score(items, predictions)

    out = run_root / task.name if limit is None else run_root / "_smoke" / task.name
    # Every response is stored verbatim, once. Tasks that replay another run's requests
    # (v3 reranks, fitted heads) name it in `reuses_responses_of`; responses already stored
    # there are not written again (the manifest records the source and the count).
    reused_from = list(getattr(task, "reuses_responses_of", ()))
    stored: set[str] = set()
    for src in reused_from:
        for shard in sorted((run_root / src).glob("responses*.jsonl.gz")):
            stored.update(row["request_hash"] for row in read_jsonl(shard))
    n_reused = sum(row["request_hash"] in stored for row in rows)
    rows = [row for row in rows if row["request_hash"] not in stored]
    # Large runs are sharded into bounded gzip files (responses.jsonl.gz, then
    # responses.00001.jsonl.gz, ...) so each shard stays well under git hosting's per-file limits.
    for stale in out.glob("responses*.jsonl.gz"):
        stale.unlink()
    shards: list[list[dict[str, Any]]] = [[]]
    raw = 0
    for row in rows:
        size = len(json.dumps(row, ensure_ascii=False)) + 1
        if shards[-1] and raw + size > SHARD_RAW_BYTES:
            shards.append([])
            raw = 0
        shards[-1].append(row)
        raw += size
    for shard, chunk in enumerate(shards):
        path = out / ("responses.jsonl.gz" if shard == 0 else f"responses.{shard:05d}.jsonl.gz")
        write_jsonl(path, chunk)
        if path.stat().st_size > SHARD_MAX_GZ_BYTES:
            raise RuntimeError(f"{path} is {path.stat().st_size} bytes gzip; lower SHARD_RAW_BYTES")
    pred_rows = [{"item_id": it.id, "gold": it.gold, "prediction": predictions[it.id]} for it in items]
    # Large prediction files (e.g. Wiki-WD, 740K edges) are gzipped to stay under git limits.
    big = sum(len(json.dumps(r, ensure_ascii=False)) for r in pred_rows) > 40_000_000
    for stale in out.glob("predictions.jsonl*"):
        stale.unlink()
    write_jsonl(out / ("predictions.jsonl.gz" if big else "predictions.jsonl"), pred_rows)
    (out / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    manifest = {
        "task": task.name,
        "model_requested": client.model,
        "model_snapshots": dict(snapshots),
        "prompt_file": task.prompt_file,
        "prompt_sha256": task.prompt_sha256(),
        "n_items": len(items),
        "n_requests": n_requests,
        "n_request_rounds": round_no,
        "input_tokens": input_tokens,
        "cost_usd": None if local else round(cost, 6),
        "git_head": _git_head(),
        "limit": limit,
        "responses_reused_from": reused_from,
        "n_responses_reused": n_reused,
        # Usage of the responses not already stored by a source run (new spend for replays).
        "input_tokens_excluding_reused": sum(int(r["response"].get("usage", {}).get("input_tokens", 0)) for r in rows),
        "cost_usd_excluding_reused": None if local else round(sum(float(r["response"].get("usage", {}).get("cost", 0.0)) for r in rows), 6),
    }
    if audit_fn is not None:
        manifest["answer_audit"] = {key: answer_audit[key] for key in sorted(answer_audit)}
    if local:
        manifest.update({
            "backend_metadata": client.metadata,
            "n_truncated_requests": n_truncated_requests,
            "local_compute": {
                "inference_seconds": inference_seconds,
                "per_device": per_device,
                "wall_seconds": time.monotonic() - started,
                "monetary_cost": "not measured",
            },
        })
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"scores": scores, "manifest": manifest}
