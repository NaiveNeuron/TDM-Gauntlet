"""A model's secondary BioRC protocol must not use another model's thresholds."""

import json

import pytest

from gauntlet.task import Item
from gauntlet.tasks.biorc800 import BioRC800, LABELS


def test_secondary_score_uses_selected_model_calibration(tmp_path):
    legacy = tmp_path / "jev_thresholds.json"
    local = tmp_path / "clef_thresholds.json"
    legacy.write_text(json.dumps({"thresholds": {label: 0.2 for label in LABELS}}))
    local.write_text(json.dumps({"thresholds": {label: 0.35 if label == "BACKGROUND" else 0.9 for label in LABELS}}))
    task = BioRC800("test")
    item = Item("abstract", [["BACKGROUND"]], {"sentences": ["The study investigates a clinical problem."]})
    predictions = {item.id: {"labels": [["BACKGROUND"]], "probs": [[0.4, 0.3, 0.1, 0.1, 0.1, 0.1]]}}
    task.thresholds_path = legacy
    transferred = task.score([item], predictions)
    task.thresholds_path = local
    isolated = task.score([item], predictions)
    assert transferred["micro_f1"] == isolated["micro_f1"] == 1.0
    assert transferred["dev_tuned"]["micro_f1"] == pytest.approx(2 / 3)
    assert isolated["dev_tuned"]["micro_f1"] == 1.0
