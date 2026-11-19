"""OMPReflectionLM over the JSONL channel: worker->host requests while the reader stays live."""

from __future__ import annotations

import json
import os
import threading

import pytest
from mockhost import complete

from jev_opt.reflection import OMPReflectionLM
from jev_opt.rpc import Channel, RpcError


class Pipes:
    def __init__(self) -> None:
        h2w_r, h2w_w = os.pipe()
        w2h_r, w2h_w = os.pipe()
        self.worker_in = os.fdopen(h2w_r, "r", encoding="utf-8")
        self.host_out = os.fdopen(h2w_w, "w", encoding="utf-8", buffering=1)
        self.worker_out = os.fdopen(w2h_w, "w", encoding="utf-8", buffering=1)
        self.host_in = os.fdopen(w2h_r, "r", encoding="utf-8")

    def send(self, msg: dict) -> None:
        self.host_out.write(json.dumps(msg) + "\n")

    def recv(self) -> dict:
        return json.loads(self.host_in.readline())


def test_reflection_lm_round_trip_and_errors() -> None:
    pipes = Pipes()
    channel = Channel(pipes.worker_in, pipes.worker_out)
    lm = OMPReflectionLM(channel, role="reflection", run_dir="/runs/r1", max_tokens=2000)

    def handler(method: str, params: dict) -> dict:
        if method == "reflect":
            return {"text": lm(params["prompt"])}
        return {"pong": True}

    server = threading.Thread(target=channel.serve, args=(handler,), daemon=True)
    server.start()

    pipes.send({"id": "h1", "method": "reflect", "params": {"prompt": "```\nThe message gives praise.\n```"}})
    ask = pipes.recv()
    assert ask["method"] == "lm.complete" and ask["id"].startswith("w")
    assert ask["params"] == {"role": "reflection", "prompt": "```\nThe message gives praise.\n```", "max_tokens": 2000, "run_dir": "/runs/r1"}
    # the reader thread must keep serving other host requests while the reflection is pending
    pipes.send({"id": "h2", "method": "ping", "params": {}})
    assert pipes.recv() == {"id": "h2", "result": {"pong": True}}
    pipes.send({"id": ask["id"], "result": complete(ask["params"])})
    reply = pipes.recv()
    assert reply["id"] == "h1" and "Praise: the customer" in reply["result"]["text"]
    assert lm.total_cost == pytest.approx(0.001) and lm.models_seen() == ["mock/mock-reflector"]

    pipes.send({"id": "h3", "method": "reflect", "params": {"prompt": "x"}})
    ask = pipes.recv()
    pipes.send({"id": ask["id"], "error": {"message": "model not resolved"}})
    assert pipes.recv() == {"id": "h3", "error": {"message": "RpcError: model not resolved"}}

    pipes.send({"id": "h4", "method": "shutdown", "params": {}})
    assert pipes.recv() == {"id": "h4", "result": {"pong": True}}
    server.join(timeout=5)
    assert not server.is_alive() and channel.closed.is_set()
    with pytest.raises(RpcError):
        lm("after close")
