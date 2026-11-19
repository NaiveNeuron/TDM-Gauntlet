"""JSONL RPC over stdin/stdout (architecture §3, protocol v1).

The reader loop runs on the calling thread; each host request is handled on its own
thread so long `optimize` calls never block `cancel` or replies to worker→host requests.
"""

from __future__ import annotations

import itertools
import json
import sys
import threading
import traceback
from collections.abc import Callable
from concurrent.futures import Future
from typing import IO, Any

Handler = Callable[[str, dict[str, Any]], dict[str, Any]]


class RpcError(RuntimeError):
    pass


class Channel:
    def __init__(self, inp: IO[str], out: IO[str]):
        self._inp = inp
        self._out = out
        self._write_lock = threading.Lock()
        self._pending: dict[str, Future] = {}
        self._pending_lock = threading.Lock()
        self._ids = itertools.count(1)
        self.closed = threading.Event()

    # ---- outbound -------------------------------------------------------
    def _send(self, message: dict[str, Any]) -> None:
        line = json.dumps(message, ensure_ascii=False, default=str)
        with self._write_lock:
            self._out.write(line + "\n")
            self._out.flush()

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"method": method, "params": params})

    def request(self, method: str, params: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        """Worker→host request; blocks the calling (non-reader) thread until the reply arrives."""
        if self.closed.is_set():
            raise RpcError("RPC channel closed")
        rid = f"w{next(self._ids)}"
        fut: Future = Future()
        with self._pending_lock:
            self._pending[rid] = fut
        self._send({"id": rid, "method": method, "params": params})
        try:
            return fut.result(timeout=timeout)
        finally:
            with self._pending_lock:
                self._pending.pop(rid, None)

    # ---- inbound --------------------------------------------------------
    def serve(self, handler: Handler, stop_methods: frozenset[str] = frozenset({"shutdown"})) -> None:
        """Read until EOF or a stop method; replies are routed to pending futures."""
        try:
            for line in self._inp:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    print(f"jev-opt: ignoring non-JSON input: {line[:200]}", file=sys.stderr)
                    continue
                if "method" not in msg:  # reply to a worker→host request
                    self._resolve(msg)
                    continue
                if "id" not in msg:  # host notifications are not part of v1; ignore
                    continue
                if msg["method"] in stop_methods:
                    self._handle(handler, msg)
                    return
                threading.Thread(target=self._handle, args=(handler, msg), daemon=True).start()
        finally:
            self.closed.set()
            with self._pending_lock:
                for fut in self._pending.values():
                    if not fut.done():
                        fut.set_exception(RpcError("RPC channel closed"))

    def _resolve(self, msg: dict[str, Any]) -> None:
        with self._pending_lock:
            fut = self._pending.get(str(msg.get("id")))
        if fut is None or fut.done():
            return
        if "error" in msg:
            err = msg["error"]
            fut.set_exception(RpcError(err.get("message", str(err)) if isinstance(err, dict) else str(err)))
        else:
            fut.set_result(msg.get("result") or {})

    def _handle(self, handler: Handler, msg: dict[str, Any]) -> None:
        try:
            result = handler(msg["method"], msg.get("params") or {})
            self._send({"id": msg["id"], "result": result})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            self._send({"id": msg["id"], "error": {"message": f"{type(exc).__name__}: {exc}"}})
