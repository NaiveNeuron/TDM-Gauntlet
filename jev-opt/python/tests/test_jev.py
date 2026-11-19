from __future__ import annotations

import time

import httpx
import pytest

from jev_opt.jev import JevClient, JevError, ResponseLedger, cache_key, load_api_key

Q = {"q": {"type": "noul", "instructions": "The message mentions a refund."}}


def make(tmp_path, fake_jev, model="typesafe/jev-1.13", sleeps=None) -> JevClient:
    return JevClient(tmp_path / "cache", "test-key", model=model, transport=fake_jev.transport(), sleep=(sleeps.append if sleeps is not None else lambda s: None))


def test_cache_hits_on_repeat_and_key_includes_model(tmp_path, fake_jev) -> None:
    a = make(tmp_path, fake_jev)
    first = a.decide({"message": "refund please"}, Q)
    second = a.decide({"message": "refund please"}, Q)
    assert (first.cached, second.cached) == (False, True)
    assert second.answers == first.answers and second.snapshot == "typesafe/jev-1.13-mock"
    assert len(fake_jev.bodies) == 1
    assert a.usage.calls == 2 and a.usage.fresh_calls == 1 and a.usage.fresh_cost == pytest.approx(first.cost)

    b = make(tmp_path, fake_jev, model="typesafe/jev-9")
    b.decide({"message": "refund please"}, Q)
    assert len(fake_jev.bodies) == 2 and fake_jev.bodies[-1]["model"] == "typesafe/jev-9"
    url = "https://openrouter.ai/api/alpha/decisions"
    assert cache_key(url, "m1", {}, Q) != cache_key(url, "m2", {}, Q)
    assert cache_key(url, "m1", {}, Q) != cache_key("http://127.0.0.1/decisions", "m1", {}, Q)


def test_rate_limit_reset_and_cloudflare_errors_are_retried(tmp_path, fake_jev) -> None:
    reset_ms = (time.time() + 3) * 1000
    fake_jev.failures = [
        httpx.Response(429, json={"error": {"message": "rate limited", "metadata": {"headers": {"X-RateLimit-Reset": str(reset_ms)}}}}),
        httpx.Response(520, text="cloudflare"),
        httpx.Response(200, json={"error": {"message": "provider hiccup"}}),
    ]
    sleeps: list[float] = []
    client = make(tmp_path, fake_jev, sleeps=sleeps)
    decision = client.decide({"message": "x"}, Q)
    assert "q" in decision.answers and len(fake_jev.bodies) == 4
    assert 2.0 < sleeps[0] <= 4.0  # waited for the X-RateLimit-Reset epoch, not a blind backoff


def test_non_retryable_status_raises(tmp_path, fake_jev) -> None:
    fake_jev.failures = [httpx.Response(400, json={"error": {"message": "bad"}})]
    with pytest.raises(JevError) as err:
        make(tmp_path, fake_jev).decide({"message": "x"}, Q)
    assert err.value.status == 400 and len(fake_jev.bodies) == 1


def test_uncached_request_without_key_fails_clearly(tmp_path, fake_jev) -> None:
    client = JevClient(tmp_path / "cache", None, transport=fake_jev.transport())
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        client.decide({"message": "x"}, Q)
    assert fake_jev.bodies == []


def test_decisions_url_override_and_env_file_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("JEV_OPT_DECISIONS_URL", "http://127.0.0.1:9/decisions")
    assert JevClient(tmp_path / "c", "k").url == "http://127.0.0.1:9/decisions"
    monkeypatch.delenv("OPENROUTER_API_KEY")
    (tmp_path / ".env").write_text("OTHER=1\nexport OPENROUTER_API_KEY='from-file'\n")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert load_api_key([nested]) == "from-file"


def test_response_ledger_dedups_and_shards_deterministically(tmp_path, fake_jev) -> None:
    def run(order, name: str) -> dict:
        (tmp_path / name).mkdir()
        client = make(tmp_path, fake_jev)  # shared cache: the second run is served from cache only
        client.ledger = ResponseLedger(tmp_path / name, shard_records=5)
        for i in order:
            client.decide({"message": f"m{i}"}, Q, {"split": "val", "condition": None})
        client.decide({"message": "m0"}, Q, {"split": "train"})  # repeat within the run: first record kept
        return client.ledger.finalize()

    a, b = run(range(12), "a"), run(reversed(range(12)), "b")
    assert len(fake_jev.bodies) == 12
    assert a == b == {"shards": ["jev_responses.jsonl.gz", "jev_responses.00001.jsonl.gz", "jev_responses.00002.jsonl.gz"], "records": 12}
    assert all((tmp_path / "a" / n).read_bytes() == (tmp_path / "b" / n).read_bytes() for n in a["shards"])
    assert not (tmp_path / "a" / "jev_responses.pending.jsonl").exists()
