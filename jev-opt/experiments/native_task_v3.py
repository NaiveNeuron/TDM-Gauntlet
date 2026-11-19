"""Write the jev-native-zero task file of one task (third optimization study), called by run_v3_dev.sh phase n.

The jev-native seed with the best zero-shot TRAIN score in the task's population run (recorded in its
run.json summary) becomes the seed program, run exactly as written: no readout, no class bias, no
cutoff/threshold search, no temperature. Usage:
  native_task_v3.py <v3-<task>-P/run.json> <task_v3*.yaml> <out task_v3_native*.yaml> <seeds dir relative to the task>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


def main() -> int:
    run_json, src, out, seeds = sys.argv[1:]
    info = json.loads(Path(run_json).read_text())["conditions"]["jev-population-opt"]["summary"]["seeds"]
    name = next(n for n, s in info.items() if s["kind"] == "jev-native" and s["chosen_for_kind"])
    task = yaml.safe_load(Path(src).read_text())
    task["seed_program"] = f"{seeds}/{name}.yaml"
    task["optimizer"].update(readout=False, class_bias=False, calibrate=False, fit_parameters=False)
    task["optimizer"].pop("population", None)
    head = (
        f"# v3 jev-native-zero (third pre-registration): jev-native seed {name}, the kind's best zero-shot TRAIN score\n"
        f"# in {Path(run_json).parent.name}, run exactly as written: no readout, bias, cutoff/threshold fit or temperature.\n"
    )
    Path(out).write_text(head + yaml.safe_dump(task, sort_keys=False, allow_unicode=True, width=100))
    print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
