from __future__ import annotations

import builtins
import io
import os
from pathlib import Path

import pytest

from jev_opt import runs
from jev_opt.hygiene import ProgramGuard, SplitAccessError, load_split, allow_test_split
from jev_opt.task import Example, load_task


def test_test_split_is_only_loadable_inside_the_evaluate_final_gate(minimal: Path) -> None:
    config = load_task(minimal)
    assert len(load_split(config, "train")) == 15
    with pytest.raises(SplitAccessError):
        load_split(config, "test")
    with allow_test_split():
        assert len(load_split(config, "test")) == 9
    with pytest.raises(SplitAccessError):
        load_split(config, "test")


def test_program_guard_blocks_copied_text_ids_and_bloat() -> None:
    train = [Example("support-ticket-17", {"message": "The espresso machine leaks water all over the counter every single morning."}, "complaint", "train")]
    seed = "criterion: The espresso machine leaks water all over the counter"  # task-author text is exempt
    guard = ProgramGuard(train, seed, max_chars=None)
    assert guard.max_chars == 4000
    assert guard.problems("criterion: complaints about leaks or spills") == []
    assert guard.problems(seed) == []
    leak = guard.problems("criterion: complaints like water all over the counter every single morning")
    assert leak and "copies example text" in leak[0]
    assert any("example id" in p for p in guard.problems("see support-ticket-17 for an example"))
    assert guard.problems("see support-ticket-170") == []
    assert any("char limit" in p for p in guard.problems("x" * 4001))


def test_optimize_never_opens_the_test_file(minimal: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client) -> None:
    test_file = (minimal.parent / "test.jsonl").resolve()
    opened: list[str] = []
    real_open, real_os_open = builtins.open, os.open

    def guard(path, *args, **kwargs):
        if isinstance(path, (str, bytes, os.PathLike)) and Path(os.fsdecode(path)).resolve() == test_file:
            opened.append(str(path))
            raise AssertionError("test split opened during optimize")
        return real_open(path, *args, **kwargs)

    def guard_os(path, *args, **kwargs):
        if Path(os.fsdecode(path)).resolve() == test_file:
            opened.append(str(path))
        return real_os_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guard)
    monkeypatch.setattr(io, "open", guard)
    monkeypatch.setattr(os, "open", guard_os)
    result = runs.optimize({"task_path": str(minimal), "runs_root": str(tmp_path / "runs"), "run_id": "iso", "conditions": ["jev-zero"]}, lambda m, p: None, lambda rd: None, client=client)
    assert result["state"] == "finished" and opened == []
    assert not (tmp_path / "runs" / "iso" / "test_consumed.json").exists()
