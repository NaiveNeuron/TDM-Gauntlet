"""`<task>+head` tasks: frozen zero-shot tasks decoded by a head fit on train/dev only.

A task is registered only when its frozen head file exists (data/frozen/heads/<task>.json,
written by scripts/fit_v3.py for the tasks the v3 dev round selected).
"""

from __future__ import annotations

from collections.abc import Callable

from ..fitted_head import HeadTask, head_path
from ..task import Task
from . import contempo, smarter

BASES: dict[str, Callable[[], Task]] = {
    "contempo/matres": contempo.Matres,
    "contempo/tbdense": contempo.TBDense,
    **{f"smarter/{s}": (lambda s=s: smarter.Smarter(s)) for s in smarter.SUBSETS},
}

TASKS: dict[str, Callable[[], Task]] = {
    f"{name}+head": (lambda name=name, base=base: HeadTask(base(), head_path(name)))
    for name, base in BASES.items()
    if head_path(name).exists()
}
