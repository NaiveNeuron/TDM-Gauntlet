"""Task registry: name -> zero-arg factory returning a Task.

Every module in this package that exposes `TASKS: dict[str, Callable[[], Task]]`
is registered automatically. A module that fails to import is reported in
`BROKEN` (module -> error) and on stderr, so one unfinished adapter does not
block running the others.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from typing import Callable

from ..task import Task

REGISTRY: dict[str, Callable[[], Task]] = {}
BROKEN: dict[str, str] = {}
for _info in pkgutil.iter_modules(__path__):
    try:
        _module = importlib.import_module(f"{__name__}.{_info.name}")
    except Exception as exc:  # noqa: BLE001 - surfaced, not swallowed
        BROKEN[_info.name] = f"{type(exc).__name__}: {exc}"
        print(f"[gauntlet] task module {_info.name!r} failed to import: {BROKEN[_info.name]}", file=sys.stderr)
        continue
    for _name, _factory in getattr(_module, "TASKS", {}).items():
        if _name in REGISTRY:
            raise RuntimeError(f"duplicate task name {_name!r} in module {_info.name}")
        REGISTRY[_name] = _factory
