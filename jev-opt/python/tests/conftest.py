from __future__ import annotations

import gzip
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from jev_opt.jev import JevClient  # noqa: E402
from jev_opt.mockjev import respond  # noqa: E402

PY_DIR = Path(__file__).resolve().parents[1]
EXAMPLE = PY_DIR.parent / "examples" / "minimal"


def read_ledger(run_dir: Path) -> list[dict[str, Any]]:
    """All records of a run's Jev response ledger, in shard order."""
    shards = json.loads((run_dir / "run.json").read_text())["jev_responses"]["shards"]
    rows: list[dict[str, Any]] = []
    for name in shards:
        with gzip.open(run_dir / name, "rt", encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh]
    return rows


class FakeJev:
    """Mock Decisions endpoint (real response shape) with request log and scripted failures."""

    def __init__(self, failures: list[httpx.Response] | None = None):
        self.bodies: list[dict[str, Any]] = []
        self.failures = list(failures or [])

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.failures:
            return self.failures.pop(0)
        return httpx.Response(200, json=respond(body))

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # A fake key guarantees no test can pick up (or bill) the real OPENROUTER_API_KEY.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    monkeypatch.delenv("JEV_OPT_DECISIONS_URL", raising=False)
    monkeypatch.delenv("JEV_OPT_MOCK_JEV", raising=False)


@pytest.fixture
def fake_jev() -> FakeJev:
    return FakeJev()


@pytest.fixture
def client(tmp_path: Path, fake_jev: FakeJev) -> JevClient:
    c = JevClient(tmp_path / "cache", "test-key", transport=fake_jev.transport(), sleep=lambda s: None)
    yield c
    c.close()


@pytest.fixture
def minimal(tmp_path: Path) -> Path:
    dest = tmp_path / "minimal"
    shutil.copytree(EXAMPLE, dest)
    return dest / "task.yaml"
