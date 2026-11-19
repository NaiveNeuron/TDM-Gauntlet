"""Client for Jev via the OpenRouter Decisions API.

Every request is content-addressed: the SHA-256 of the canonical request JSON
keys an on-disk cache, so re-runs never re-bill and raw responses are kept
verbatim for audit.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
RETRY_STATUS = {408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529}


def load_api_key(env_file: Path = Path(".env")) -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "OPENROUTER_API_KEY" and value.strip():
                return value.strip().strip("\"'")
    raise RuntimeError("OPENROUTER_API_KEY not set in environment or .env")


def request_body(state: Any, questions: dict[str, Any], model: str = MODEL) -> dict[str, Any]:
    return {"model": model, "state": state, "questions": questions}


def request_hash(body: dict[str, Any]) -> str:
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class JevError(RuntimeError):
    def __init__(self, status: int, payload: str):
        super().__init__(f"Jev request failed with HTTP {status}: {payload[:500]}")
        self.status = status
        self.payload = payload


@dataclass
class JevClient:
    cache_dir: Path
    api_key: str
    model: str = MODEL
    concurrency: int = 16
    # OpenRouter caps typesafe/jev-1.13 at 2,000 requests/minute (shared across
    # processes on the key); stay under it so 429s are the exception.
    requests_per_minute: int = 1800
    max_retries: int = 20
    timeout_s: float = 120.0

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._sem = asyncio.Semaphore(self.concurrency)
        self._interval = 60.0 / self.requests_per_minute
        self._next_slot = 0.0
        self._slot_lock = asyncio.Lock()

    async def _pace(self) -> None:
        """Space request starts at least `_interval` seconds apart."""
        async with self._slot_lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self._interval
        if wait > 0:
            await asyncio.sleep(wait)

    def _cache_path(self, digest: str) -> Path:
        return self.cache_dir / digest[:2] / f"{digest}.json"

    def cached(self, body: dict[str, Any]) -> dict[str, Any] | None:
        path = self._cache_path(request_hash(body))
        return json.loads(path.read_text()) if path.exists() else None

    async def decide(self, http: httpx.AsyncClient, body: dict[str, Any]) -> dict[str, Any]:
        """Return the cached-or-fresh response record {request_hash, response}."""
        digest = request_hash(body)
        path = self._cache_path(digest)
        if path.exists():
            return json.loads(path.read_text())
        async with self._sem:
            response = await self._post_with_retry(http, body)
        record = {"request_hash": digest, "response": response}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False))
        tmp.replace(path)
        return record

    async def _post_with_retry(self, http: httpx.AsyncClient, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for attempt in range(self.max_retries + 1):
            await self._pace()
            try:
                resp = await http.post(DECISIONS_URL, json=body, headers=headers, timeout=self.timeout_s)
            except (httpx.TransportError, httpx.TimeoutException):
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(_backoff(attempt))
                continue
            if resp.status_code == 200:
                payload = resp.json()
                if "answers" in payload:
                    return payload
                # 200 with an error envelope: treat as retryable provider error.
                if attempt == self.max_retries:
                    raise JevError(200, resp.text)
            elif resp.status_code not in RETRY_STATUS or attempt == self.max_retries:
                raise JevError(resp.status_code, resp.text)
            await asyncio.sleep(_retry_delay(resp, attempt))
        raise AssertionError("unreachable")


def _retry_delay(resp: httpx.Response, attempt: int) -> float:
    """Prefer server hints: Retry-After header, then OpenRouter's X-RateLimit-Reset (epoch ms)."""
    retry_after = resp.headers.get("retry-after")
    if retry_after and retry_after.replace(".", "", 1).isdigit():
        return float(retry_after)
    try:
        reset_ms = float(resp.json()["error"]["metadata"]["headers"]["X-RateLimit-Reset"])
    except (ValueError, KeyError, TypeError):
        return _backoff(attempt)
    wait = reset_ms / 1000.0 - time.time()
    return min(max(wait, 0.0), 65.0) + random.random()


def _backoff(attempt: int) -> float:
    return min(60.0, 2.0**attempt) * (0.5 + random.random() / 2)
