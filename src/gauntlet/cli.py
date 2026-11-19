"""`gauntlet list` / `gauntlet status` / `gauntlet run <task> [--limit N]`."""

from __future__ import annotations

import argparse
import json

from .runner import RUNS, run_task
from .tasks import BROKEN, REGISTRY
from .task import REPO

# Every benchmark paper -> its registered task prefix. A paper is complete when every
# registered task under the prefix has runs/<task>/scores.json. Multi-dataset papers
# expand to several tasks under one prefix.
SPEC_PAPERS: dict[str, str] = {
    "LifeTox": "lifetox", "PiNose": "pinose", "FENICE": "fenice", "FactPICO": "factpico",
    "CICC": "cicc", "Edit intent": "edit_intent", "EZ-STANCE": "ezstance", "ConTempo": "contempo",
    "MISMATCHED": "mismatched", "INLI": "inli", "Entity or Relation Embeddings": "relemb",
    "PRP-Graph": "prp_graph", "Scalar Constructs": "scalar_constructs", "DREsS": "dress_new",
    "MedReadMe": "medreadme", "Explanation Quality Ranking": "expl_rank", "MADE": "made",
    "Security TTP": "ttp", "LegalSeg": "legalseg",
    "SciPrompt": "sciprompt", "Navigating the Unknown": "nav_unknown", "SMARTER": "smarter",
    "BIORC800": "biorc800", "XSTest": "xstest",
}


def _status() -> int:
    """Per-paper, per-task completion; exit 1 if anything is missing."""
    missing = 0
    for paper, prefix in SPEC_PAPERS.items():
        tasks = sorted(n for n in REGISTRY if n == prefix or n.startswith(prefix + "/"))
        if not tasks:
            state = "BROKEN" if prefix in BROKEN else "NO ADAPTER"
            print(f"{paper:32s} {state}")
            missing += 1
            continue
        for t in tasks:
            done = (RUNS / t / "scores.json").exists()
            state = "scored" if done else "PENDING"
            missing += not done
            print(f"{paper:32s} {t:36s} {state}")
    for mod, err in BROKEN.items():
        print(f"BROKEN module {mod}: {err}")
    print(f"{len(SPEC_PAPERS)} papers, {len(REGISTRY)} tasks registered, {missing} pending (not scored)")
    return 1 if missing else 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="gauntlet")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("compare-clef", help="paired frozen zero-shot Jev/Clef comparison")
    sub.add_parser("status", help="completeness over the benchmark papers (exit 1 if incomplete)")
    run = sub.add_parser("run")
    run.add_argument("tasks", nargs="*", help="task names or prefixes (e.g. pinose)")
    run.add_argument("--limit", type=int, default=None, help="smoke run on first N items (written under runs/_smoke)")
    run.add_argument("--concurrency", type=int, default=16)
    run.add_argument("--backend", choices=("jev", "clef", "clef-flash", "luna", "msd1"), default="jev")
    run.add_argument("--audit", action="store_true", help="all previously scored frozen zero-shot tasks")
    run.add_argument("--device", action="append", help="local Clef device (repeat for multiple GPUs; default cuda:0)")
    run.add_argument("--max-length", type=int, default=32000, help="Clef context; overflow fails rather than truncates")
    run.add_argument("--rpm", type=int, default=1200, help="Luna/MSD1: requests per minute")
    run.add_argument("--budget-usd", type=float, default=None, help="Luna/MSD1: total cap over the model's spend (default: the model's cap in gauntlet.luna)")
    run.add_argument("--run-budget-usd", type=float, default=None, help="Luna/MSD1: cap of the zero-shot run label (default: the model's cap in gauntlet.luna)")
    args = parser.parse_args()

    if args.cmd == "list":
        for name in sorted(REGISTRY):
            print(name)
        for mod, err in BROKEN.items():
            print(f"BROKEN {mod}: {err}")
        return
    if args.cmd == "status":
        raise SystemExit(_status())
    if args.cmd == "compare-clef":
        from .clef_report import build_all
        print(build_all())
        return

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.max_length <= 0:
        parser.error("--max-length must be positive")
    devices = args.device or ["cuda:0"]
    if len(devices) != len(set(devices)):
        parser.error("--device values must be distinct")
    if args.audit:
        if args.tasks:
            parser.error("--audit cannot be combined with task names")
        from .clef_report import zero_shot_names
        selected = zero_shot_names()
    else:
        selected = sorted(n for n in REGISTRY if any(n == t or n.startswith(t.rstrip("/") + "/") for t in args.tasks))
    if not selected:
        parser.error("no task matches; see `gauntlet list`, or use --audit")
    client = None
    run_root = RUNS
    multi_gpu_client = None
    try:
        if args.backend in ("luna", "msd1"):
            from .clef_report import zero_shot_names
            from .jev import load_api_key
            from .luna import SPECS, LunaClient, SpendLedger
            spec = SPECS[args.backend]
            eligible = set(zero_shot_names()) - set(spec.excluded)
            if args.audit:
                selected = sorted(eligible)
            excluded = sorted(set(selected) - eligible)
            if excluded:
                parser.error(f"{spec.label} contrast is frozen zero-shot only, Wiki-WD excluded ({spec.protocol}); excluded tasks: {excluded}")
            spend = SpendLedger(run="zero-shot", total_cap=args.budget_usd, run_cap=args.run_budget_usd, spec=spec)
            client = LunaClient(spec.cache, api_key=load_api_key(REPO / ".env"), concurrency=args.concurrency,
                                requests_per_minute=args.rpm, spec=spec, spend=spend)
            run_root = RUNS / spec.name
        if args.backend in ("clef", "clef-flash"):
            from .clef import ClefClient, ClefMultiGpuClient, MODELS
            from .clef_report import zero_shot_names
            eligible = set(zero_shot_names())
            excluded = sorted(set(selected) - eligible)
            if excluded:
                parser.error(f"Clef contrast is frozen zero-shot only; excluded tasks: {excluded}")
            cache_dir = REPO / "data" / "cache" / args.backend
            if len(devices) == 1:
                client = ClefClient(cache_dir, device=devices[0],
                                    max_length=args.max_length, model=MODELS[args.backend])
            else:
                client = multi_gpu_client = ClefMultiGpuClient(
                    cache_dir, devices=devices, max_length=args.max_length, model=MODELS[args.backend])
            run_root = RUNS / args.backend
        if args.backend != "jev":
            # BioRC's secondary paper protocol fits per-label thresholds on this
            # model's dev predictions; never transfer the stored Jev thresholds.
            if "biorc800" in selected and "biorc800/dev" not in selected and args.limit is None:
                selected.append("biorc800/dev")
            selected.sort(key=lambda name: (name != "biorc800/dev", name))
        failures = {}
        for name in selected:
            try:
                task = REGISTRY[name]()
                if client is not None and name in ("biorc800", "biorc800/dev"):
                    calibration_root = run_root if args.limit is None else run_root / "_smoke"
                    task.thresholds_path = calibration_root / "biorc800" / "dev_thresholds.json"
                result = run_task(task, concurrency=args.concurrency, limit=args.limit,
                                  client=client, run_root=run_root)
                if client is not None and name == "biorc800/dev" and args.limit is None:
                    thresholds = {"thresholds": result["scores"]["fitted_thresholds"],
                                  "model": client.model, "dev_manifest": result["manifest"]}
                    task.thresholds_path.parent.mkdir(parents=True, exist_ok=True)
                    task.thresholds_path.write_text(json.dumps(thresholds, indent=2) + "\n")
            except Exception as exc:
                if not args.audit:
                    raise
                failures[name] = f"{type(exc).__name__}: {exc}"
                print(json.dumps({"task": name, "error": failures[name]}), flush=True)
                continue
            print(json.dumps({"task": name, **result}, indent=2), flush=True)
        if args.backend in ("clef", "clef-flash") and args.audit:
            run_root.mkdir(parents=True, exist_ok=True)
            (run_root / "audit_status.json").write_text(json.dumps({"failures": failures}, indent=2) + "\n")
            from .clef_report import build_all
            print(build_all())
        if args.backend in ("luna", "msd1") and args.audit:
            run_root.mkdir(parents=True, exist_ok=True)
            (run_root / "audit_status.json").write_text(json.dumps({"failures": failures}, indent=2) + "\n")
        if failures:
            raise SystemExit(1)
    finally:
        if multi_gpu_client is not None:
            multi_gpu_client.close()


if __name__ == "__main__":
    main()
