"""JSONL RPC worker entry point (architecture §3). Protocol on stdout, logs on stderr."""

from __future__ import annotations

import importlib.metadata as md
import io
import os
import sys
from typing import Any

from . import __version__, runs, seedgen
from .reflection import OMPReflectionLM
from .rpc import Channel

PROTOCOL = 1


class Worker:
    def __init__(self, channel: Channel):
        self.channel = channel
        self.omp_version: str | None = None

    def lm_factory(self, run_dir: str) -> OMPReflectionLM:
        return OMPReflectionLM(self.channel, role="reflection", run_dir=run_dir)

    def handle(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "hello":
            self.omp_version = params.get("omp_version")
            return {"protocol": PROTOCOL, "versions": {"jev_opt": __version__, "gepa": md.version("gepa"), "python": sys.version.split()[0]}}
        if method == "optimize":
            return runs.optimize(params, self.channel.notify, self.lm_factory, omp_version=self.omp_version)
        if method == "status":
            return runs.status(params["runs_root"], params.get("run_id"))
        if method == "show":
            return runs.show(params["run_dir"], params["candidate"])
        if method == "pareto":
            return runs.pareto(params["run_dir"])
        if method == "generate_seeds":
            return seedgen.run_generation(params, self.channel.notify, self.lm_factory)
        if method == "evaluate_final":
            return runs.evaluate_final(params["run_dir"], params.get("representative"))
        if method == "cancel":
            return {"ok": True, "cancelled": runs.cancel_all()}
        if method == "shutdown":
            runs.cancel_all()
            return {"ok": True}
        raise ValueError(f"unknown method {method!r}")


def main() -> int:
    # Keep a private handle on the real stdout for the protocol, then point fd 1 and
    # sys.stdout at stderr so no library print (GEPA, uv, tqdm) can corrupt the stream.
    proto_out = io.TextIOWrapper(os.fdopen(os.dup(1), "wb"), encoding="utf-8", line_buffering=True)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
    channel = Channel(stdin, proto_out)
    channel.serve(Worker(channel).handle)
    proto_out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
