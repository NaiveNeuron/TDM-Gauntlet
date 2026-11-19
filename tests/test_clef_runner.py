"""Runner dispatch, mixed-device provenance, and CLI client lifecycle (no GPUs)."""

import asyncio
import sys
from types import SimpleNamespace

import pytest

from gauntlet import cli, runner
from gauntlet.task import Item, Request, Task, read_jsonl




def test_single_device_keeps_original_windows():
    bodies = [{"index": i} for i in range(130)]
    windows = []

    async def decide_many(http, requests):
        windows.append(requests)
        return requests

    client = SimpleNamespace(local=True, batch_size=16, decide_many=decide_many)
    assert asyncio.run(runner._run_requests(client, None, bodies, "single")) == bodies
    assert windows == [bodies[:64], bodies[64:128], bodies[128:]]




class TinyTask(Task):
    name = "tiny"
    prompt_file = "unused.json"

    def items(self):
        return [Item(str(i), i) for i in range(3)]

    def requests(self, item):
        yield Request(item.id, "value", item.id, {})

    def predict(self, item, answers):
        return answers["value"]["value"]

    def score(self, items, predictions):
        return {"correct": sum(predictions[item.id] == item.gold for item in items)}

    def prompt_sha256(self):
        return "frozen-prompt"


def test_mixed_device_cohort_accounting_and_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_git_head", lambda: "test-head")
    records = [
        {"request_hash": "gpu1-cached", "execution_device": "cuda:1", "cache_hit": True,
         "response": {"model": "frozen", "answers": {"value": 0},
                      "usage": {"input_tokens": 10}, "audit": {"inference_seconds": 100}}},
        {"request_hash": "gpu0-new", "execution_device": "cuda:0", "cache_hit": False,
         "response": {"model": "frozen", "answers": {"value": 1},
                      "usage": {"input_tokens": 20}, "audit": {"inference_seconds": 2}}},
        {"request_hash": "gpu1-new", "execution_device": "cuda:1", "cache_hit": False,
         "response": {"model": "frozen", "answers": {"value": 2},
                      "usage": {"input_tokens": 30}, "audit": {"inference_seconds": 3}}},
    ]

    async def decide_windows(bodies, label):
        assert len(bodies) == 3
        assert label == "tiny round 1"
        return records

    metadata = {"devices": [{"device": "cuda:0"}, {"device": "cuda:1"}]}
    client = SimpleNamespace(local=True, model="frozen", metadata=metadata,
                             decide_windows=decide_windows)
    result = runner.run_task(TinyTask(), client=client, run_root=tmp_path)
    manifest = result["manifest"]
    assert result["scores"] == {"correct": 3}
    assert manifest["n_requests"] == 3
    assert manifest["input_tokens"] == 60
    assert manifest["cost_usd"] is None
    assert manifest["local_compute"]["inference_seconds"] == 5
    per_device = manifest["local_compute"]["per_device"]
    assert per_device["cuda:0"] == {
        "n_requests": 1, "n_cached_requests": 0, "n_new_requests": 1,
        "input_tokens": 20, "cached_input_tokens": 0, "new_input_tokens": 20,
        "inference_seconds": 2, "cached_response_inference_seconds": 0,
    }
    assert per_device["cuda:1"] == {
        "n_requests": 2, "n_cached_requests": 1, "n_new_requests": 1,
        "input_tokens": 40, "cached_input_tokens": 10, "new_input_tokens": 30,
        "inference_seconds": 3, "cached_response_inference_seconds": 100,
    }
    rows = read_jsonl(tmp_path / "tiny" / "responses.jsonl.gz")
    assert [row["execution_device"] for row in rows] == ["cuda:1", "cuda:0", "cuda:1"]


def test_single_device_cached_repetitions_do_not_charge_compute(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_git_head", lambda: "test-head")
    response = {"model": "frozen", "answers": {"value": 0},
                "usage": {"input_tokens": 10}, "audit": {"inference_seconds": 2}}

    async def decide_many(http, bodies):
        return [{"request_hash": "native-hash", "response": response, "cache_hit": i > 0}
                for i in range(len(bodies))]

    client = SimpleNamespace(local=True, device="cuda:0", model="frozen", batch_size=16,
                             metadata={"device": "cuda:0"}, decide_many=decide_many)
    result = runner.run_task(TinyTask(), client=client, run_root=tmp_path)
    assert result["manifest"]["input_tokens"] == 30
    compute = result["manifest"]["local_compute"]
    assert compute["inference_seconds"] == 2
    assert compute["per_device"]["cuda:0"]["n_new_requests"] == 1
    assert compute["per_device"]["cuda:0"]["n_cached_requests"] == 2
    assert compute["per_device"]["cuda:0"]["new_input_tokens"] == 10


def configure_cli(monkeypatch, devices, failure=None, audit=False):
    instances = []

    class FakeSingle:
        model = "frozen"

        def __init__(self, cache_dir, **kwargs):
            self.closed = False
            instances.append(self)

    class FakeMulti(FakeSingle):
        def close(self):
            self.closed = True

    fake_clef = SimpleNamespace(ClefClient=FakeSingle, ClefMultiGpuClient=FakeMulti,
                                MODELS={"clef": "frozen"})
    monkeypatch.setitem(sys.modules, "gauntlet.clef", fake_clef)
    monkeypatch.setitem(sys.modules, "gauntlet.clef_report", SimpleNamespace(
        zero_shot_names=lambda: ["tiny"], build_all=lambda: "report"))
    monkeypatch.setattr(cli, "REGISTRY", {"tiny": TinyTask})

    def run_task(task, **kwargs):
        if failure is not None:
            raise failure
        return {"scores": {}, "manifest": {}}

    monkeypatch.setattr(cli, "run_task", run_task)
    argv = ["gauntlet", "run", "--backend", "clef", "--max-length", "32000"]
    argv += ["--audit"] if audit else ["tiny"]
    for device in devices:
        argv += ["--device", device]
    monkeypatch.setattr(sys, "argv", argv)
    return instances


@pytest.mark.parametrize("failure", [None, RuntimeError("failed"), KeyboardInterrupt()])
def test_cli_multi_device_always_closes(monkeypatch, failure):
    instances = configure_cli(monkeypatch, ["cuda:0", "cuda:1"], failure=failure)
    if failure is None:
        cli.main()
    else:
        with pytest.raises(type(failure)):
            cli.main()
    assert instances[0].closed


def test_cli_rejects_duplicate_devices(monkeypatch):
    configure_cli(monkeypatch, ["cuda:0", "cuda:0"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_cli_audit_failure_still_closes(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RUNS", tmp_path)
    instances = configure_cli(
        monkeypatch, ["cuda:0", "cuda:1"], failure=RuntimeError("failed"), audit=True)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert instances[0].closed
