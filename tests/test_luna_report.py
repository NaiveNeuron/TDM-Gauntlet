"""The paired Jev / GPT-6 Luna comparison: pin, rejection share and Wiki-WD exclusion."""

import json

import pytest

from gauntlet import clef_report
from gauntlet.luna import MODEL, REFUSED_MODEL, REJECTED_MODEL, SNAPSHOT
from gauntlet.task import write_jsonl


@pytest.fixture
def audit(tmp_path, monkeypatch):
    monkeypatch.setattr(clef_report, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(clef_report, "RESULTS", tmp_path / "results")
    clef_report.RESULTS.mkdir()
    return tmp_path


def _run(name, *, luna=False, requests=2, rejected=0, refused=0, snapshots=None, changes=None, accuracy=None):
    directory = clef_report.RUNS / ("luna" if luna else "") / name
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"task": name, "limit": None, "n_items": 2, "n_requests": requests, "prompt_sha256": "a" * 64,
                "input_tokens": 100, "cost_usd": 0.01}
    if luna:
        labels = {SNAPSHOT: requests - rejected - refused, REJECTED_MODEL: rejected, REFUSED_MODEL: refused}
        manifest.update(
            model_requested=MODEL,
            model_snapshots=snapshots or {k: v for k, v in labels.items() if v},
            answer_audit={"n_requests": requests, "n_rejected_requests": rejected, "n_refused_requests": refused,
                          "n_answers": requests, "n_refused_answers": rejected + refused, "n_saturated_answers": 0},
        )
    else:
        manifest.update(model_requested="typesafe/jev-frozen", model_snapshots={"typesafe/jev-frozen": requests})
    manifest.update(changes or {})
    (directory / "manifest.json").write_text(json.dumps(manifest))
    value = accuracy if accuracy is not None else 0.75 if luna else 0.5
    (directory / "scores.json").write_text(json.dumps({"n": 2, "accuracy": value}))
    write_jsonl(directory / "predictions.jsonl", [
        {"item_id": "a", "gold": "yes", "prediction": 0},
        {"item_id": "b", "gold": "no", "prediction": 1},
    ])


def _rows(report):
    return {row["task"]: row for row in report["rows"]}


def test_pinned_full_run_is_completed_with_luna_minus_jev(audit):
    _run("task")
    _run("task", luna=True, accuracy=0.25)
    report = clef_report.report("luna")
    row = _rows(report)["task"]
    assert row["status"] == "completed", row["reasons"]
    assert row["delta"] == -0.25
    assert row["outcome"] == "loss"
    assert report["clef_model"] == MODEL and report["clef_revision"] == SNAPSHOT
    assert report["test_outcomes"]["losses"] == 1


@pytest.mark.parametrize("case", ["other_snapshot", "mixed_snapshots", "model_requested", "no_audit"])
def test_unpinned_or_unaudited_luna_run_never_gets_delta(audit, case):
    _run("task")
    snapshots = {"other_snapshot": {"openai/gpt-6-luna-decisions-20261105": 2},
                 "mixed_snapshots": {SNAPSHOT: 1, "openai/gpt-6-luna-decisions-20261105": 1}}.get(case)
    changes = {"model_requested": {"model_requested": "openai/gpt-6-luna"}, "no_audit": {"answer_audit": None}}.get(case)
    _run("task", luna=True, snapshots=snapshots, changes=changes)
    row = _rows(clef_report.report("luna"))["task"]
    assert row["status"] == "noncomparable"
    assert row["delta"] is None and row["outcome"] is None
    assert row["reasons"]


@pytest.mark.parametrize(("rejected", "status"), [(1, "completed"), (2, "noncomparable")])
def test_more_than_one_percent_rejected_requests_is_noncomparable(audit, rejected, status):
    _run("task", requests=100)
    _run("task", luna=True, requests=100, rejected=rejected)
    row = _rows(clef_report.report("luna"))["task"]
    assert row["status"] == status
    assert (row["delta"] is None) == (status == "noncomparable")
    if status == "noncomparable":
        assert any("2 of 100 requests rejected" in reason for reason in row["reasons"])


def test_refused_requests_are_counted_not_a_comparability_condition(audit):
    _run("task", requests=100)
    _run("task", luna=True, requests=100, refused=5)
    row = _rows(clef_report.report("luna"))["task"]
    assert row["status"] == "completed", row["reasons"]
    assert row["delta"] == 0.25


def test_missing_wikiwd_is_excluded_for_luna_but_pending_for_clef(audit):
    _run("relemb/wikiwd")
    _run("task")
    report = clef_report.report("luna")
    rows = _rows(report)
    assert rows["relemb/wikiwd"]["status"] == "excluded"
    assert rows["relemb/wikiwd"]["delta"] is None
    assert rows["task"]["status"] == "pending"
    assert report["coverage"] == {"total": 2, "completed": 0, "pending": 1, "blocked": 0, "noncomparable": 0, "excluded": 1}
    assert _rows(clef_report.report("clef"))["relemb/wikiwd"]["status"] == "pending"
