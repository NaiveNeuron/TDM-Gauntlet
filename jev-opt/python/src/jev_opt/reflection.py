"""GEPA `LanguageModel` callable that routes every optimizer LLM call through the OMP host."""

from __future__ import annotations

import threading
from typing import Any, Protocol


class Requester(Protocol):
    def request(self, method: str, params: dict[str, Any], timeout: float | None = None) -> dict[str, Any]: ...


class OMPReflectionLM:
    """`prompt -> text` via `lm.complete`; the host picks model, auth and thinking level.

    Exposes `total_cost` so GEPA's `max_reflection_cost` stopper uses host-reported spend.
    """

    def __init__(self, rpc: Requester, role: str = "reflection", run_dir: str | None = None, max_tokens: int | None = None, timeout_s: float | None = 900.0):
        self.rpc = rpc
        self.role = role
        self.run_dir = run_dir
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        params: dict[str, Any] = {"role": self.role, "prompt": prompt}
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        if self.run_dir:
            params["run_dir"] = self.run_dir
        result = self.rpc.request("lm.complete", params, timeout=self.timeout_s)
        text = result.get("text")
        if not isinstance(text, str):
            raise TypeError("lm.complete result.text must be a string")
        usage = result.get("usage") or {}
        with self._lock:
            self.calls.append({
                "role": self.role,
                "provider": result.get("provider"),
                "model": result.get("model"),
                "input": int(usage.get("input") or 0),
                "output": int(usage.get("output") or 0),
                "cost_total": float(usage.get("cost_total") or 0.0),
            })
        return text

    def with_role(self, role: str, max_tokens: int | None = None) -> OMPReflectionLM:
        """Sibling callable for another role (optionally another reply budget) sharing the usage ledger."""
        sibling = OMPReflectionLM(self.rpc, role, self.run_dir, max_tokens or self.max_tokens, self.timeout_s)
        sibling.calls, sibling._lock = self.calls, self._lock
        return sibling

    @property
    def total_cost(self) -> float:
        with self._lock:
            return sum(c["cost_total"] for c in self.calls)

    def models_seen(self) -> list[str]:
        with self._lock:
            return sorted({f"{c['provider']}/{c['model']}" for c in self.calls if c["model"]})
