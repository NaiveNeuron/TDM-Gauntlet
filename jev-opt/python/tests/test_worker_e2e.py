"""End to end over the real protocol: worker.py subprocess, mock OMP host on its pipes,
stub Decisions API over HTTP. Bad seed -> GEPA -> reflection via host -> better val ->
test untouched until evaluate_final -> ledger refuses resume and re-evaluation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from conftest import PY_DIR, read_ledger
from mockhost import complete

from jev_opt.jev import cache_key
from jev_opt.mockjev import respond


class StubDecisions(BaseHTTPRequestHandler):
    hits = 0
    bodies: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.headers["Authorization"] == "Bearer test-key-not-real"
        type(self).hits += 1
        type(self).bodies.append(body)
        if type(self).hits == 1:  # first call is rate limited, like a busy shared key
            self._reply(429, {"error": {"message": "rate limited"}}, {"Retry-After": "0"})
        else:
            self._reply(200, respond(body))

    def _reply(self, status: int, payload: dict, headers: dict | None = None) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        for k, v in {"Content-Type": "application/json", "Content-Length": str(len(data)), **(headers or {})}.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: Any) -> None:
        pass


class Host:
    """Minimal OMP host: sends requests, answers lm.complete, checks every stdout line is JSON."""

    def __init__(self, env: dict[str, str], cwd: Path):
        self.proc = subprocess.Popen([sys.executable, str(PY_DIR / "worker.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=cwd)
        self.notifications: list[dict] = []
        self.lm_requests: list[dict] = []
        self.stderr: list[str] = []
        threading.Thread(target=lambda: self.stderr.extend(self.proc.stderr), daemon=True).start()
        self._n = 0

    def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._n += 1
        rid = f"h{self._n}"
        self.proc.stdin.write(json.dumps({"id": rid, "method": method, "params": params}) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            assert line, f"worker exited; stderr tail: {''.join(self.stderr[-20:])}"
            msg = json.loads(line)  # any non-JSON stdout line fails here
            if msg.get("method") == "lm.complete":
                self.lm_requests.append(msg["params"])
                self.proc.stdin.write(json.dumps({"id": msg["id"], "result": complete(msg["params"])}) + "\n")
                self.proc.stdin.flush()
            elif "method" in msg:
                self.notifications.append(msg["params"])
            elif msg["id"] == rid:
                return msg


@pytest.fixture
def stub_url():
    StubDecisions.hits = 0
    StubDecisions.bodies = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubDecisions)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/api/alpha/decisions"
    server.shutdown()


def request_hashes(url: str, bodies: list[dict]) -> set[str]:
    return {cache_key(url, b["model"], b["state"], b["questions"]) for b in bodies}


def test_worker_end_to_end(minimal: Path, tmp_path: Path, stub_url: str) -> None:
    env = {**os.environ, "JEV_OPT_DECISIONS_URL": stub_url, "OPENROUTER_API_KEY": "test-key-not-real"}
    host = Host(env, tmp_path)
    runs_root = tmp_path / "runs"
    test_file = minimal.parent / "test.jsonl"

    hello = host.call("hello", {})["result"]
    assert hello["protocol"] == 1 and hello["versions"]["gepa"] == "0.1.4"

    test_file.chmod(0)  # unreadable: optimize must never open the test split
    try:
        reply = host.call("optimize", {"task_path": str(minimal), "runs_root": str(runs_root), "run_id": "e2e"})
    finally:
        test_file.chmod(0o644)
    assert "result" in reply, reply
    result = reply["result"]
    run_dir = Path(result["run_dir"])
    seed = result["conditions"]["jev-zero"]["best_val_quality"]
    assert seed == pytest.approx(1 / 3)
    assert result["conditions"]["jev-text-opt"]["best_val_quality"] > seed + 0.5
    assert result["conditions"]["jev-program-opt"]["best_val_quality"] > seed + 0.5
    assert host.lm_requests and all(r["run_dir"] == str(run_dir) for r in host.lm_requests)
    assert host.notifications[0]["event"] == "run_started" and host.notifications[0]["optimizer"]["model"] == "current"
    assert StubDecisions.hits > 1  # the 429 was retried against the stub, not the internet
    assert not (run_dir / "test_consumed.json").exists()
    assert json.loads((run_dir / "environment.json").read_text())["forbidden_modules_loaded"] == []

    # One verbatim record per distinct Jev request of the run (every request went to the stub once).
    rows = read_ledger(run_dir)
    assert json.loads((run_dir / "run.json").read_text())["jev_responses"] == {"shards": ["jev_responses.jsonl.gz"], "records": len(rows)}
    assert [r["request_hash"] for r in rows] == sorted(request_hashes(stub_url, StubDecisions.bodies))
    posted = {cache_key(stub_url, b["model"], b["state"], b["questions"]): b for b in StubDecisions.bodies}
    assert all(r["request"] == posted[r["request_hash"]] and r["response"]["answers"] for r in rows)
    assert {r["split"] for r in rows} == {"train", "val"} and not (run_dir / "jev_responses.pending.jsonl").exists()

    status = host.call("status", {"runs_root": str(runs_root)})["result"]["runs"]
    assert status[0]["run_id"] == "e2e" and status[0]["state"] == "finished" and not status[0]["test_consumed"]
    shown = host.call("show", {"run_dir": str(run_dir), "candidate": "best-quality"})["result"]
    assert "kind: " in shown["program_yaml"] and shown["scores"]["primary"] == 1.0
    assert host.call("show", {"run_dir": str(run_dir), "candidate": "jev-zero/0"})["result"]["rationale"] == "seed"
    front = host.call("pareto", {"run_dir": str(run_dir)})["result"]
    assert set(front["representatives"]) == {"best-quality", "best-cheap", "best-calibrated", "top-quality", "top-jev-zero", "top-jev-text-opt", "top-jev-program-opt"} and front["front"]

    before_final = len(StubDecisions.bodies)
    final = host.call("evaluate_final", {"run_dir": str(run_dir)})["result"]
    assert final["metrics"]["n"] == 9 and final["metrics"]["accuracy"] == 1.0 and final["consumed_at"]
    assert final["metrics"]["hygiene"]["test_ids_in_train_or_val"] == 0
    ledger = [json.loads(line) for line in (runs_root / "test_ledger.jsonl").read_text().splitlines()]
    assert [row["run_id"] for row in ledger] == ["e2e"]
    test_rows = [r for r in read_ledger(run_dir) if r["split"] == "test"]
    assert {r["request_hash"] for r in test_rows} == request_hashes(stub_url, StubDecisions.bodies[before_final:])
    assert json.loads((run_dir / "run.json").read_text())["jev_responses"]["records"] == len(rows) + len(test_rows)

    again = host.call("evaluate_final", {"run_dir": str(run_dir)})
    assert "already consumed" in again["error"]["message"]
    resume = host.call("optimize", {"runs_root": str(runs_root), "run_id": "e2e", "resume": True})
    assert "resume is refused" in resume["error"]["message"]
    assert "Consumed at" in (run_dir / "report.md").read_text()

    # A second run served purely from the Jev cache still records every response it used.
    hits = StubDecisions.hits
    cached = host.call("optimize", {"task_path": str(minimal), "runs_root": str(runs_root), "run_id": "cached", "conditions": ["jev-zero"]})["result"]
    cached_rows = read_ledger(Path(cached["run_dir"]))
    assert StubDecisions.hits == hits and cached_rows
    assert {r["request_hash"] for r in cached_rows} == {r["request_hash"] for r in rows if r["condition"] == "jev-zero"}

    assert host.call("shutdown", {})["result"] == {"ok": True}
    assert host.proc.wait(timeout=30) == 0
