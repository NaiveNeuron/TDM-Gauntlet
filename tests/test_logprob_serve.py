"""Base-model option scoring (gauntlet.logprob_serve) on a tiny random Qwen3.5 text model on the CPU."""

import io
import json

import pytest

torch = pytest.importorskip("torch")
qwen = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402

from gauntlet.logprob_serve import ALPHABET, MODEL, ContinuationError, LogprobScorer, Runtime, serve  # noqa: E402

V = 300
WORDS = {"Yes": 1, "No": 2}  # multi-character tokens; every other character is its own token


def encode(text: str) -> list[int]:
    ids, i = [], 0
    while i < len(text):
        word = next((w for w in WORDS if text.startswith(w, i)), None)
        if word:
            ids.append(WORDS[word])
            i += len(word)
        else:
            ids.append(3 + ord(text[i]) % (V - 3))
            i += 1
    return ids


def runtime(encoder=encode) -> Runtime:
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=16, full_attention_interval=4,
                            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                            linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
                            linear_conv_kernel_dim=4)
    return Runtime(qwen.Qwen3_5TextModel(cfg).eval(), torch.nn.Linear(64, V, bias=False), encoder, 0)


def scorer(**kw) -> LogprobScorer:
    rt = runtime(kw.pop("encoder", encode))
    return LogprobScorer(device="cpu", runtime_factory=lambda: rt, **kw)


def reference(rt: Runtime, prompt: str, continuations: list[str]) -> list[float]:
    """Unbatched, unpadded forward over the prompt alone."""
    with torch.inference_mode():
        h = rt.text_model(input_ids=torch.tensor([encode(prompt)]), use_cache=False).last_hidden_state[0, -1]
        logp = torch.log_softmax(rt.head(h).float(), -1)
    return [float(logp[encode(c)[0]]) for c in continuations]


def test_batched_right_padded_scores_equal_the_unbatched_forward():
    s = scorer(batch_size=3, batch_padded_token_limit=200)
    units = [(f"k{i}", {"prompt": "state " * (i * 3 + 1) + "</think>\n\n", "continuations": ["A", "B", "C"][: 2 + i % 2]}) for i in range(7)]
    units.append(("yn", {"prompt": "Is it?</think>\n\n", "continuations": ["Yes", "No"]}))
    out = s.compute(units)
    assert [r["key"] for r in out] == [k for k, _ in units]
    assert max(r["response"]["audit"]["batch_size"] for r in out) == 3  # several buckets, padding inside each
    for r, (_, payload) in zip(out, units):
        resp = r["response"]
        ref = reference(s.runtime(), payload["prompt"], payload["continuations"])
        assert resp["logprobs"] == pytest.approx(ref, abs=1e-5)
        z = sum(torch.tensor(ref).exp().tolist())
        assert resp["probabilities"] == pytest.approx([float(torch.tensor(x).exp()) / z for x in ref], abs=1e-5)
        assert resp["option_mass"] == pytest.approx(z, rel=1e-5)
        assert resp["usage"]["input_tokens"] == len(encode(payload["prompt"])) and resp["model"] == MODEL


def test_over_guard_and_multi_token_units_are_rejected_alone():
    s = scorer(max_length=50)
    out = s.compute([("ok", {"prompt": "short", "continuations": ["A", "B"]}),
                     ("long", {"prompt": "x" * 51, "continuations": ["A", "B"]}),
                     ("multi", {"prompt": "short", "continuations": ["A", "AB"]})])
    assert "response" in out[0]
    assert out[1]["type"] == "LogprobContextError" and "truncation is forbidden" in out[1]["error"]
    assert out[2]["type"] == "ContinuationError"


def test_a_tokenizer_that_splits_a_template_answer_is_refused_before_serving():
    s = scorer(encoder=lambda t: [3 + ord(c) % (V - 3) for c in t])  # "Yes" would be three tokens
    with pytest.raises(ContinuationError, match="Yes"):
        s.runtime()
    assert "Yes" in ALPHABET and "Z" in ALPHABET and "9" in ALPHABET


def test_serve_handshake_then_one_result_line_per_window():
    s = scorer(max_length=100)
    inp = io.StringIO(json.dumps({"bodies": [["a", {"prompt": "p", "continuations": ["A", "B"]}]]}) + "\n\n")
    out = io.StringIO()
    serve(s, inp, out)
    hello, reply = [json.loads(line) for line in out.getvalue().splitlines()]
    assert hello["ready"] and hello["model"] == MODEL and hello["configuration"]["max_length"] == 100
    assert hello["configuration"]["dtype"] == "bfloat16" and hello["configuration"]["padding"] == "right"
    assert [r["key"] for r in reply["results"]] == ["a"]
