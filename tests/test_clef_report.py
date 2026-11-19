"""Regression coverage for refusing scientifically invalid paired deltas."""

import json

import pytest

from gauntlet import clef_report
from gauntlet.task import write_jsonl


@pytest.fixture
def audit(tmp_path, monkeypatch):
    monkeypatch.setattr(clef_report, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(clef_report, "RESULTS", tmp_path / "results")
    clef_report.RESULTS.mkdir()
    return tmp_path


def _run(name, *, clef=False, backend="clef", rows=None, scores=None, changes=None):
    directory = clef_report.RUNS / (backend if clef else "") / name
    directory.mkdir(parents=True, exist_ok=True)
    identity = clef_report.MODELS[backend] if clef else "typesafe/jev-frozen"
    manifest = {
        "task": name, "limit": None, "n_items": 2, "n_requests": 2,
        "prompt_sha256": "a" * 64, "model_requested": identity,
        "model_snapshots": {identity: 2}, "input_tokens": 100,
        "cost_usd": None if clef else 0.01,
    }
    if clef:
        manifest.update(
            backend_metadata=dict(zip(("model_id", "revision"), clef_report.MODELS[backend].split("@", 1))),
            n_truncated_requests=0,
            local_compute={"inference_seconds": 1.2, "wall_seconds": 2.3, "monetary_cost": "not measured"},
        )
    manifest.update(changes or {})
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "scores.json").write_text(json.dumps(scores or {"n": 2, "accuracy": 0.75 if clef else 0.5}))
    write_jsonl(directory / ("predictions.jsonl.gz" if clef else "predictions.jsonl"), rows if rows is not None else [
        {"item_id": "a", "gold": {"x": 1, "y": 2}, "prediction": 0},
        {"item_id": "b", "gold": "yes", "prediction": 1},
    ])


def test_reordered_gzip_cohort_is_comparable_without_item_hash(audit):
    _run("task/nested")
    _run("task/nested", clef=True, rows=[
        {"item_id": "b", "gold": "yes", "prediction": 0},
        {"item_id": "a", "gold": {"y": 2, "x": 1}, "prediction": 1},
    ])
    row = clef_report._paired_row("task/nested", {"metric": "accuracy"})
    assert row["status"] == "completed"
    assert row["delta"] == 0.25
    assert row["cohort"]["matched"] is True
    assert row["outcome"] == "win"


@pytest.mark.parametrize("case", ["gold", "ids", "duplicate", "limited", "missing_limit", "prompt", "revision", "truncated", "count", "dict_metric", "string_metric"])
def test_invalid_pair_never_gets_delta(audit, case):
    _run("task")
    changes, scores, rows = {}, None, None
    if case in {"gold", "ids", "duplicate"}:
        rows = [
            {"item_id": "a", "gold": {"x": 1, "y": 2}, "prediction": 0},
            {"item_id": "b", "gold": "yes", "prediction": 1},
        ]
        if case == "gold":
            rows[1]["gold"] = "no"
        elif case == "ids":
            rows[1]["item_id"] = "other"
        else:
            rows[1]["item_id"] = "a"
    elif case == "limited":
        changes["limit"] = 2
    elif case == "prompt":
        changes["prompt_sha256"] = "b" * 64
    elif case == "revision":
        changes["backend_metadata"] = {"model_id": clef_report.CLEF_MODEL, "revision": "unpinned"}
    elif case == "truncated":
        changes["n_truncated_requests"] = 1
    elif case == "count":
        scores = {"n": 1, "accuracy": 0.75}
    elif case in {"dict_metric", "string_metric"}:
        scores = {"n": 2, "accuracy": {"value": 0.75} if case == "dict_metric" else "0.75"}
    _run("task", clef=True, changes=changes, scores=scores, rows=rows)
    if case == "missing_limit":
        path = clef_report.RUNS / "clef/task/manifest.json"
        manifest = json.loads(path.read_text())
        del manifest["limit"]
        path.write_text(json.dumps(manifest))
    row = clef_report._paired_row("task", {"metric": "accuracy"})
    assert row["status"] == "noncomparable"
    assert row["delta"] is None
    assert row["outcome"] is None
    assert row["reasons"]


def test_cicc_uses_numeric_single_at_same_operating_point(audit):
    _run("cicc/b77", scores={"n": 1, "n_calibration": 1, "single": 0.0, "one_minus_alpha": 0.98, "th": 7})
    _run("cicc/b77", clef=True, scores={"n": 1, "n_calibration": 1, "single": 0.5, "one_minus_alpha": 0.98, "th": 7})
    row = clef_report._paired_row("cicc/b77", {"metric": "single"})
    assert row["delta"] == 0.5
    assert row["outcome"] == "win"
    _run("cicc/b77", clef=True, scores={"n": 1, "n_calibration": 1, "single": 0.5, "one_minus_alpha": 0.97, "th": 7})
    row = clef_report._paired_row("cicc/b77", {"metric": "single"})
    assert row["status"] == "noncomparable"
    assert row["delta"] is None


def test_oos_reporting_keeps_scored_total_and_subset_distinct(audit):
    scores = {"n": 2, "n_oos": 1, "n_calibration": 0, "n_oos_calibration": 0, "f1": 0.5}
    _run("cicc/b77_oos", scores=scores)
    for backend in clef_report.MODELS:
        _run("cicc/b77_oos", clef=True, backend=backend, scores=scores)
    markdown = clef_report.build_all()
    line = next(line for line in markdown.splitlines() if line.startswith("| cicc/b77_oos |"))
    cells = [cell.strip() for cell in line.split("|")[1:-1]]
    assert cells[2] == "2"
    assert cells[3] == "2 (1 OOS) / 2 (1 OOS) / 2 (1 OOS)"


def test_build_covers_pending_and_excludes_dev_from_outcomes(audit):
    _run("pending/nested")
    _run("biorc800/dev")
    _run("biorc800/dev", clef=True)
    for excluded in ("task+head", "_smoke/task", "task_v3", "dev_v3/task", "jev-opt/task", "delta_stance/task"):
        _run(excluded)
    _run("no-baseline", clef=True)
    references = {name: {"metric": "accuracy", "method": "paper method", "paper": {"paper method": 0.8}, "source": "Paper Table 1"} for name in ("pending/nested", "biorc800/dev")}
    (clef_report.RESULTS / "paper_numbers.json").write_text(json.dumps(references))
    markdown = clef_report.build()
    report = json.loads((clef_report.RESULTS / "clef_comparison.json").read_text())
    assert clef_report.zero_shot_names() == ["biorc800/dev", "pending/nested"]
    assert report["coverage"] == {"total": 2, "completed": 1, "pending": 1, "blocked": 0, "noncomparable": 0}
    assert report["test_outcomes"]["comparable"] == 0
    assert report["rows"][1]["clef_value"] is None
    assert report["rows"][1]["delta"] is None


def test_audit_failure_blocks_missing_score_but_not_valid_completed_run(audit):
    _run("blocked")
    _run("completed")
    _run("completed", clef=True)
    status = clef_report.RUNS / "clef/audit_status.json"
    status.write_text(json.dumps({"failures": {"blocked": "FileNotFoundError: frozen dataset absent", "completed": "old failure"}}))
    clef_report.build()
    report = json.loads((clef_report.RESULTS / "clef_comparison.json").read_text())
    rows = {row["task"]: row for row in report["rows"]}
    assert rows["blocked"]["status"] == "blocked"
    assert "frozen dataset absent" in rows["blocked"]["reasons"][0]
    assert rows["blocked"]["delta"] is None
    assert rows["completed"]["status"] == "completed"
    assert rows["completed"]["delta"] == 0.25
    assert report["coverage"]["blocked"] == 1


def test_flash_is_compared_separately_and_rejects_clef_identity(audit):
    _run("task")
    _run("task", clef=True)
    _run("task", clef=True, backend="clef-flash", scores={"n": 2, "accuracy": 0.25})
    clef_report.build_all()
    reports = json.loads((clef_report.RESULTS / "decision_models_comparison.json").read_text())
    assert reports["clef"]["rows"][0]["delta"] == 0.25
    assert reports["clef-flash"]["rows"][0]["delta"] == -0.25
    assert reports["clef-flash"]["test_outcomes"]["losses"] == 1
    _run("task", clef=True, backend="clef-flash", changes={"model_requested": clef_report.MODELS["clef"]})
    row = clef_report._paired_row("task", {"metric": "accuracy"}, backend="clef-flash")
    assert row["status"] == "noncomparable"
    assert row["delta"] is None
