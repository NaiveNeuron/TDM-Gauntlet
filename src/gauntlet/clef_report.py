"""Paired frozen zero-shot audit; never modifies Jev run artifacts.

``build()`` writes results/clef_comparison.{json,md}. Prediction cohorts are
streamed using task.read_jsonl's opener/blank-line conventions into a temporary
SQLite index, rather than retaining both large relation-task runs in memory.
"""

from __future__ import annotations

import json
import math
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterator

from .task import REPO, _open
from .clef import MODELS
from .luna import SPECS as API_SPECS

RUNS = REPO / "runs"
RESULTS = REPO / "results"
CLEF_MODEL = "Cloudflare/clef"
CLEF_REVISION = "2f3de3dd85f379784083b0814d997ab627200f0c"
# API backends: "model id@dated snapshot" (gauntlet.luna); local backends: "repo@revision" (gauntlet.clef).
API_BACKENDS = {name: f"{spec.model}@{spec.snapshot}" for name, spec in API_SPECS.items()}
PINS = {**MODELS, **API_BACKENDS}
REJECTED_MAX_SHARE = 0.01
# Explicit metric semantics, not an inference from a score's type or position.
HIGHER_BETTER = frozenset({
    "accuracy", "balanced_accuracy", "macro_f1", "micro_f1", "all_macro_f1",
    "f1", "auc", "auroc", "ndcg@5", "ndcg@10", "p@1", "p_at_10",
    "qwk_total", "kendall_tau_b", "pearson_mean_groups", "spearman_rho", "single",
})


def _excluded(relative: Path) -> bool:
    top = relative.parts[0]
    return (
        top in {"clef", "clef-flash", *API_SPECS, "_smoke", "dev_v3", "jev-opt", "delta_stance"}
        or top.endswith("_v3")
        or "_smoke" in relative.parts
        or any("+head" in part for part in relative.parts)
    )


def zero_shot_names() -> list[str]:
    """Existing scored baseline manifests eligible for the audit CLI."""
    return sorted({
        path.parent.relative_to(RUNS).as_posix()
        for path in RUNS.rglob("manifest.json")
        if (path.parent / "scores.json").exists() and not _excluded(path.relative_to(RUNS))
    })


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _safe(value: Any) -> Any:
    """Use standard JSON null for nonfinite supplementary scores, never a delta."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_safe(child) for child in value]
    return value


def _load_run(directory: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    run: dict[str, Any] = {"path": str(directory.relative_to(RUNS))}
    missing, invalid = [], []
    for kind in ("scores", "manifest"):
        path = directory / f"{kind}.json"
        if not path.exists():
            missing.append(f"missing {kind}.json")
            run[kind] = {}
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("expected a JSON object")
            run[kind] = value
        except (OSError, ValueError) as exc:
            invalid.append(f"invalid {kind}.json: {exc}")
            run[kind] = {}
    predictions = [p for p in (directory / "predictions.jsonl", directory / "predictions.jsonl.gz") if p.exists()]
    run["predictions_file"] = str(predictions[0].relative_to(RUNS)) if len(predictions) == 1 else None
    if not predictions:
        missing.append("missing predictions.jsonl[.gz]")
    elif len(predictions) != 1:
        invalid.append("ambiguous predictions: both plain and gzip files exist")
    return run, missing, invalid


def _cohort_rows(path: Path, side: int) -> Iterator[tuple[int, str, str]]:
    with _open(path, "r") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("item_id"), str):
                raise ValueError(f"{path.name}:{line_no}: missing/string-invalid item_id")
            if "gold" not in row or "prediction" not in row:
                raise ValueError(f"{path.name}:{line_no}: missing gold or prediction")
            gold = json.dumps(row["gold"], sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            yield side, row["item_id"], gold


def _compare_cohorts(jev: dict[str, Any], clef: dict[str, Any], model_label: str) -> tuple[dict[str, Any], list[str]]:
    """Compare exact ID/gold sets, accepting reordered rows and rejecting duplicates."""
    evidence: dict[str, Any] = {"method": "streamed SQLite item_id/gold equality", "matched": False}
    reasons = []
    with tempfile.TemporaryDirectory(prefix="gauntlet-clef-cohort-") as temporary:
        with sqlite3.connect(str(Path(temporary) / "cohort.sqlite")) as db:
            db.execute("PRAGMA cache_size = -8192")
            db.execute("CREATE TABLE cohort (side INTEGER, item_id TEXT, gold TEXT NOT NULL, PRIMARY KEY(side, item_id)) WITHOUT ROWID")
            for side, (label, run) in enumerate((("Jev", jev), (model_label, clef))):
                try:
                    db.executemany("INSERT INTO cohort VALUES (?, ?, ?)", _cohort_rows(RUNS / run["predictions_file"], side))
                except sqlite3.IntegrityError:
                    reasons.append(f"{label}: duplicate prediction item_id")
                except (OSError, EOFError, ValueError, TypeError) as exc:
                    reasons.append(f"{label}: invalid predictions: {exc}")
                count = db.execute("SELECT COUNT(*) FROM cohort WHERE side = ?", (side,)).fetchone()[0]
                evidence["jev_prediction_count" if side == 0 else "clef_prediction_count"] = count
                if count != run["manifest"].get("n_items"):
                    reasons.append(f"{label}: prediction count does not equal manifest n_items")
            if reasons:
                return evidence, reasons
            missing = db.execute("SELECT COUNT(*) FROM cohort j LEFT JOIN cohort c ON c.side = 1 AND c.item_id = j.item_id WHERE j.side = 0 AND c.item_id IS NULL").fetchone()[0]
            extra = db.execute("SELECT COUNT(*) FROM cohort c LEFT JOIN cohort j ON j.side = 0 AND j.item_id = c.item_id WHERE c.side = 1 AND j.item_id IS NULL").fetchone()[0]
            wrong_gold = db.execute("SELECT COUNT(*) FROM cohort j JOIN cohort c ON c.side = 1 AND c.item_id = j.item_id WHERE j.side = 0 AND j.gold != c.gold").fetchone()[0]
            evidence.update({"missing_clef_ids": missing, "extra_clef_ids": extra, "gold_mismatches": wrong_gold})
            if missing or extra:
                reasons.append(f"item_id sets differ ({model_label} missing {missing}, extra {extra})")
            if wrong_gold:
                reasons.append(f"gold differs for {wrong_gold} items")
            evidence["matched"] = not reasons
    return evidence, reasons


def _manifest_reasons(name: str, jev: dict[str, Any], clef: dict[str, Any], backend: str = "clef") -> list[str]:
    reasons = []
    expected_model, expected_revision = PINS[backend].split("@", 1)
    model_label = backend.capitalize()
    for label, manifest in (("Jev", jev), (model_label, clef)):
        if "limit" not in manifest or manifest["limit"] is not None:
            reasons.append(f"{label}: full run requires explicit limit=null")
        if manifest.get("task") != name:
            reasons.append(f"{label}: manifest task does not match run path")
        if not _count(manifest.get("n_items")) or manifest.get("n_items", 0) == 0:
            reasons.append(f"{label}: invalid/empty n_items")
        prompt = manifest.get("prompt_sha256")
        if not isinstance(prompt, str) or len(prompt) != 64 or any(c not in "0123456789abcdef" for c in prompt):
            reasons.append(f"{label}: missing/invalid prompt_sha256")
    if jev.get("prompt_sha256") != clef.get("prompt_sha256"):
        reasons.append("prompt_sha256 differs")
    if jev.get("n_items") != clef.get("n_items"):
        reasons.append("manifest n_items differs")
    if jev.get("item_sha256") and clef.get("item_sha256") and jev["item_sha256"] != clef["item_sha256"]:
        reasons.append("manifest item_sha256 differs")
    if backend in API_BACKENDS:
        return reasons + _api_reasons(clef, backend)
    metadata = clef.get("backend_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    model = metadata.get("model_id", metadata.get("repo_id"))
    revision = metadata.get("revision", metadata.get("model_revision"))
    if model != expected_model or revision != expected_revision:
        reasons.append(f"{backend} backend_metadata does not identify the pinned release")
    requested = clef.get("model_requested")
    snapshots = clef.get("model_snapshots")
    if requested != MODELS[backend]:
        reasons.append(f"{backend} model_requested does not match the pinned identity")
    if not isinstance(snapshots, dict) or not snapshots or any(
        identity != MODELS[backend] for identity in snapshots
    ):
        reasons.append(f"{model_label} model_snapshots missing or not uniformly pinned")
    truncated = clef.get("n_truncated_requests")
    if not _count(truncated) or truncated != 0:
        reasons.append(f"{model_label} encoder truncation accounting must explicitly report n_truncated_requests=0")
    return reasons


def _api_reasons(manifest: dict[str, Any], backend: str) -> list[str]:
    """API backend (Luna, MSD1): pinned model id and dated snapshot on every answered request (requests rejected or
    refused as a whole carry the labels `rejected` / `refused`); rejected requests (never truncated) at most
    REJECTED_MAX_SHARE of the cohort; refusals are counted, not a comparability condition."""
    from .luna import REFUSED_MODEL, REJECTED_MODEL

    model_id, snapshot = PINS[backend].split("@", 1)
    label = backend.capitalize()
    reasons = []
    if manifest.get("model_requested") != model_id:
        reasons.append(f"{backend} model_requested does not match the pinned model id")
    snapshots = manifest.get("model_snapshots")
    if not isinstance(snapshots, dict) or snapshots.get(snapshot, 0) <= 0 or any(
        identity not in (snapshot, REJECTED_MODEL, REFUSED_MODEL) for identity in snapshots
    ):
        reasons.append(f"{label} model_snapshots missing or not uniformly pinned")
    audit = manifest.get("answer_audit")
    requests = manifest.get("n_requests")
    if not isinstance(audit, dict) or not _count(audit.get("n_rejected_requests")) or not _count(requests) or requests == 0:
        reasons.append(f"{label} answer_audit (refusals/rejections) missing")
    elif audit["n_rejected_requests"] > REJECTED_MAX_SHARE * requests:
        reasons.append(f"{label}: {audit['n_rejected_requests']} of {requests} requests rejected (> {REJECTED_MAX_SHARE:.0%})")
    return reasons


def _metric(scores: dict[str, Any], reference: dict[str, Any]) -> str:
    if isinstance(reference.get("metric"), str):
        return reference["metric"]
    # No positional fallback: textual primary_metric descriptions are not scores.
    declared = scores.get("primary_metric")
    if isinstance(declared, str) and declared.split(" ", 1)[0] in HIGHER_BETTER:
        return declared.split(" ", 1)[0]
    candidates = [key for key in scores if key in HIGHER_BETTER]
    return candidates[0] if len(candidates) == 1 else "unknown"


def _paired_row(name: str, reference: dict[str, Any], blocked_reason: str | None = None, backend: str = "clef") -> dict[str, Any]:
    model_label = backend.capitalize()
    jev, jev_missing, jev_invalid = _load_run(RUNS / name)
    clef, clef_missing, clef_invalid = _load_run(RUNS / backend / name)
    metric = _metric(jev["scores"], reference)
    row: dict[str, Any] = {
        "task": name, "split": "dev" if name == "biorc800/dev" else "test",
        "n_items": jev["manifest"].get("n_items"), "primary_metric": metric,
        "direction": "higher is better" if metric in HIGHER_BETTER else "unknown",
        "jev_value": jev["scores"].get(metric), "clef_value": clef["scores"].get(metric),
        "delta": None, "outcome": None, "status": "pending", "reasons": [],
        "protocol_deviations": [],
        "paper_reference": reference, "paper_method": reference.get("method"),
        "paper_method_value": reference.get("paper", {}).get(reference.get("method")),
        "jev": jev, "clef": clef, "cohort": None,
    }
    metadata = clef["manifest"].get("backend_metadata", {})
    guard = metadata.get("max_length") if isinstance(metadata, dict) else None
    if _count(guard) and guard != 32000:
        row["protocol_deviations"].append(
            f"complete-input guard {guard} tokens instead of initial 32000; not an equal-context comparison"
        )
    invalid = [f"Jev: {reason}" for reason in jev_missing + jev_invalid] + [f"{model_label}: {reason}" for reason in clef_invalid]
    if blocked_reason and "missing scores.json" in clef_missing:
        row.update(status="blocked", reasons=[f"{model_label} audit failed: {blocked_reason}"] + invalid + [f"{model_label}: {reason}" for reason in clef_missing])
        return row
    if invalid:
        row.update(status="noncomparable", reasons=invalid + [f"{model_label}: {reason}" for reason in clef_missing])
        return row
    if clef_missing:
        row["reasons"] = [f"{model_label}: {reason}" for reason in clef_missing]
        return row
    reasons = _manifest_reasons(name, jev["manifest"], clef["manifest"], backend)
    for label, run in (("Jev", jev), (model_label, clef)):
        if not _number(run["scores"].get(metric)):
            reasons.append(f"{label}: primary metric {metric!r} is missing or not a finite numeric scalar")
    # Task score N can be a scored subset of manifest N (e.g. CICC test/calibration).
    for key in ("n", "n_calibration", "n_oos", "n_oos_calibration"):
        if key == "n" or key in jev["scores"] or key in clef["scores"]:
            a, b = jev["scores"].get(key), clef["scores"].get(key)
            if not _count(a) or not _count(b) or a != b:
                reasons.append(f"scores {key} differs or is invalid")
    if metric == "single":
        for key in ("one_minus_alpha", "th"):
            a, b = jev["scores"].get(key), clef["scores"].get(key)
            if not _number(a) or not _number(b) or a != b:
                reasons.append(f"CICC Single operating point {key} differs or is missing")
    if not reasons:
        row["cohort"], cohort_reasons = _compare_cohorts(jev, clef, model_label)
        reasons.extend(cohort_reasons)
    if reasons:
        row.update(status="noncomparable", reasons=reasons)
        return row
    row["status"] = "completed"
    row["delta"] = row["clef_value"] - row["jev_value"]
    if metric in HIGHER_BETTER:
        row["outcome"] = "win" if row["delta"] > 0 else "loss" if row["delta"] < 0 else "tie"
    return row


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if _number(value):
        return f"{value:.4f}" if isinstance(value, float) else str(value)
    return json.dumps(_safe(value), ensure_ascii=False)


def _scored_n(scores: dict[str, Any]) -> str:
    count = _fmt(scores.get("n"))
    if "n_oos" in scores:
        count += f" ({_fmt(scores['n_oos'])} OOS)"
    return count


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + [
        "| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows
    ]


def _status(row: dict[str, Any]) -> str:
    notes = row["protocol_deviations"]
    return row["status"] + (": " + "; ".join(notes) if notes else "")


def _markdown(report: dict[str, Any]) -> str:
    coverage, aggregate = report["coverage"], report["test_outcomes"]
    model_label = report["clef_model"].rsplit("/", 1)[-1].capitalize()
    lines = [
        f"# Frozen zero-shot Jev / {report['clef_model']} audit", "",
        f"Model is pinned to `{report['clef_model']}@{report['clef_revision']}`. Missing artifacts are pending unless the audit recorded a failure, in which case they are blocked; neither status has estimated scores.", "",
        f"Coverage: {coverage['total']} rows; {coverage['completed']} completed, {coverage['pending']} pending, {coverage['blocked']} blocked, {coverage['noncomparable']} noncomparable. Includes explicitly labeled dev rows.",
        f"Comparable test primary metrics: {aggregate['comparable']} directional comparisons; {model_label} wins {aggregate['wins']}, exact ties {aggregate['ties']}, losses {aggregate['losses']}. Dev rows are excluded. Completed rows with unknown direction: {aggregate['unknown_direction']}.", "",
        f"Deltas are {model_label} − Jev in native metric units and only appear for full runs with identical prompt SHA-256, item IDs/gold, and counts. Predictions are compared explicitly even when item hashes exist. No heterogeneous-metric average or leaderboard substitution is used. Exact numerical equality defines a tie; these are descriptive outcomes, not significance tests.", "",
        "Direction: accuracy, F1, AUC, nDCG, precision, agreement/correlation and CICC `single` are higher-is-better. CICC Single is a numeric share of test inputs answered without clarification, not a string/dict; its target coverage and threshold must match. Interpret Single jointly with achieved coverage in the detailed scores. Calibration/postprocessing already in the frozen adapters is unchanged; no model fitting or prompt optimization is added.", "",
        "N input is the frozen Jev manifest's total input count, including any calibration rows. N scored is each model's `scores.n`; `n_oos` is a subset of that total and is shown in parentheses, never added. Calibration rows are excluded. Scored units are task-specific (for example, sentences or relations), not always input rows.", "",
        "A changed local complete-input guard is a protocol deviation, disclosed beside the affected row and in JSON. Completed still means the full unchanged cohort passed pin, prompt, gold/ID and zero-truncation checks, not that every backend setting matches the initial protocol.", "",
        "## Paired primary metrics", "",
    ]
    lines += _table(
        ["Task", "Split", "N input", "N scored (Jev / model)", "Primary metric", "Jev", model_label, f"{model_label} − Jev", "Status"],
        [[row["task"], row["split"], _fmt(row["n_items"]), " / ".join(_scored_n(row[model]["scores"]) for model in ("jev", "clef")), row["primary_metric"], _fmt(row["jev_value"]), _fmt(row["clef_value"]), _fmt(row["delta"]), _status(row) + (": " + "; ".join(row["reasons"]) if row["reasons"] else "")] for row in report["rows"]],
    )
    lines += ["", "## Paper context (not a substitute for the paired cohort)", "", f"Paper method values and source/caveats below come verbatim from `results/paper_numbers.json`. Published systems may be fitted, few-shot, or use different evaluation protocols. Their values are contextual, not additional {model_label} comparisons. All paper-reference fields and every run's complete metric details are retained in JSON; nonfinite supplementary values serialize as null, never zero.", ""]
    lines += _table(
        ["Task", "Paper method", "Paper value", "Source / protocol caveats"],
        [[row["task"], row["paper_method"] or "—", _fmt(row["paper_method_value"]), row["paper_reference"].get("source", "No paper reference available")] for row in report["rows"]],
    )
    lines += ["", "## Snapshots and resource accounting", "", f"Jev's reported API dollars are not comparable to a local {model_label} cost of $0: local compute has no assigned dollar price. Local wall/device time, GPU identity, cache usage and encoder accounting are reported when recorded, otherwise unknown. Tokenizers differ, so input-token counts are accounting, not equal-work evidence. Full manifests (including backend metadata) are in JSON.", ""]
    resources = []
    for row in report["rows"]:
        j, c = row["jev"]["manifest"], row["clef"]["manifest"]
        resources.append([
            row["task"], _fmt(j.get("input_tokens")), _fmt(c.get("input_tokens")),
            _fmt(j.get("cost_usd")), _fmt(c.get("local_compute")),
            _fmt(j.get("model_snapshots")), _fmt(c.get("model_snapshots")),
        ])
    lines += _table(["Task", "Jev input tokens", f"{model_label} input tokens", "Jev API USD", f"{model_label} local compute (unpriced)", "Jev snapshots", f"{model_label} snapshots"], resources)
    lines += ["", "Excluded: `runs/clef` and `runs/clef-flash` from Jev discovery, smoke runs, `dev_v3`, `*_v3`, `jev-opt`, `delta_stance`, and fitted `+head` runs. `biorc800/dev` is shown only as dev and never contributes to test outcomes.", ""]
    return "\n".join(lines)


def report(backend: str = "clef") -> dict[str, Any]:
    """The paired comparison of one backend against Jev (rows, coverage, sign outcomes)."""
    model_id, revision = PINS[backend].split("@", 1)
    paper_path = RESULTS / "paper_numbers.json"
    paper = json.loads(paper_path.read_text(encoding="utf-8")) if paper_path.exists() else {}
    status_path = RUNS / backend / "audit_status.json"
    failures: dict[str, str] = {}
    status_error = None
    if status_path.exists():
        try:
            failures = json.loads(status_path.read_text(encoding="utf-8"))["failures"]
            if not isinstance(failures, dict) or any(not isinstance(value, str) for value in failures.values()):
                raise ValueError("audit failures must map task names to exception strings")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures = {}
            status_error = str(exc)
    names = sorted({path.parent.relative_to(RUNS).as_posix() for path in RUNS.rglob("scores.json") if not _excluded(path.relative_to(RUNS))})
    rows = [_paired_row(name, paper.get(name, {}), failures.get(name), backend) for name in names]
    statuses = ["completed", "pending", "blocked", "noncomparable"]
    if backend in API_BACKENDS:  # cohorts the protocol excludes are listed as excluded, never pending
        spec = API_SPECS[backend]
        for row in rows:
            if row["task"] in spec.excluded and row["status"] == "pending":
                row.update(status="excluded", reasons=[f"excluded by {spec.protocol} (cost)"])
        statuses.append("excluded")
    coverage = {"total": len(rows), **{status: sum(row["status"] == status for row in rows) for status in statuses}}
    test_rows = [row for row in rows if row["split"] == "test" and row["status"] == "completed"]
    out = {
        "audit": "frozen zero-shot paired comparison", "clef_model": model_id, "clef_revision": revision,
        "coverage": coverage,
        "audit_failures": failures, "audit_status_error": status_error,
        "test_outcomes": {
            "comparable": sum(row["outcome"] is not None for row in test_rows),
            "wins": sum(row["outcome"] == "win" for row in test_rows),
            "ties": sum(row["outcome"] == "tie" for row in test_rows),
            "losses": sum(row["outcome"] == "loss" for row in test_rows),
            "unknown_direction": sum(row["outcome"] is None for row in test_rows),
        },
        "delta_definition": f"{backend.capitalize()} minus Jev; native metric units, exact equality for ties",
        "higher_is_better_metrics": sorted(HIGHER_BETTER),
        "nonfinite_supplementary_values": "JSON null; not zero or an inferred score",
        "rows": rows,
    }
    if backend not in API_BACKENDS:
        out.update(local_compute_usd=None, local_compute_note="Unpriced local compute; not $0 and not comparable to API spend")
        out["rows"] = out.pop("rows")  # key order of the earlier Clef reports
    return out


def build(backend: str = "clef") -> str:
    data = report(backend)
    markdown = _markdown(data)
    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = backend.replace("-", "_") + "_comparison"
    (RESULTS / f"{stem}.json").write_text(json.dumps(_safe(data), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (RESULTS / f"{stem}.md").write_text(markdown, encoding="utf-8")
    return markdown


def build_all() -> str:
    """Render both frozen models against Jev without conflating model cohorts."""
    reports = {}
    for backend in MODELS:
        build(backend)
        reports[backend] = json.loads((RESULTS / (backend.replace("-", "_") + "_comparison.json")).read_text())
    by_model = {backend: {row["task"]: row for row in report["rows"]} for backend, report in reports.items()}
    names = sorted(set().union(*(rows.keys() for rows in by_model.values())))
    table = []
    for name in names:
        clef, flash = (by_model[backend][name] for backend in MODELS)
        table.append([
            name, clef["split"], _fmt(clef["n_items"]),
            " / ".join(_scored_n(run["scores"]) for run in (clef["jev"], clef["clef"], flash["clef"])),
            clef["primary_metric"],
            _fmt(clef["jev_value"]), _fmt(clef["clef_value"]), _fmt(clef["delta"]), _status(clef),
            _fmt(flash["clef_value"]), _fmt(flash["delta"]), _status(flash),
        ])
    lines = ["# Frozen zero-shot Jev / Clef / Clef-flash contrast", "",
             "Same frozen adapters, prompts and evaluation cohorts. Deltas are model − Jev in native metric units; missing or noncomparable runs have no delta. Per-model reports retain exact cohort checks, failures, paper context and unpriced local compute.", "",
             "N input is the frozen Jev manifest's total input count, including any calibration rows. N scored lists Jev / Clef / Clef-flash `scores.n`; `n_oos` is a subset of that total and is shown in parentheses, never added. Calibration rows are excluded. Scored units are task-specific (for example, sentences or relations), not always input rows.", "",
             "Changed local complete-input guards are protocol deviations, labeled beside each affected result and retained in JSON. Completed means full-cohort/pin/zero-truncation checks passed, not identical context guards.", ""]
    lines += _table(
        ["Task", "Split", "N input", "N scored (Jev / Clef / Flash)", "Metric", "Jev", "Clef", "Δ Clef", "Clef status",
         "Clef-flash", "Δ Flash", "Flash status"], table)
    for backend, report in reports.items():
        lines += ["", f"## {backend}", "", f"Pin: `{MODELS[backend]}`.",
                  f"Coverage: {json.dumps(report['coverage'])}.",
                  f"Descriptive test outcomes vs Jev: {json.dumps(report['test_outcomes'])}.",
                  f"Details: `{backend.replace('-', '_')}_comparison.md` and `.json`."]
    markdown = "\n".join(lines) + "\n"
    (RESULTS / "decision_models_comparison.md").write_text(markdown)
    (RESULTS / "decision_models_comparison.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return markdown


if __name__ == "__main__":
    print(build())
