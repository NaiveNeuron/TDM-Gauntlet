"""Line-delimited JSON server that scores option continuations with Clef's base model (jev-opt's log-prob backend).

`python -m gauntlet.logprob_serve --model qwen-base --device cuda:0` loads the pinned Qwen3.8-27B release from
the local Hugging Face cache once (it never downloads) and answers unit windows sent by `jev_opt.logprob`:

    -> {"bodies": [[key, {"prompt": text, "continuations": ["A", "B", ...]}], ...]}
    <- {"results": [{"key": key, "response": response} | {"key": key, "error": message, "type": name}, ...]}

Per unit: one forward pass over the complete prompt (BF16, SDPA, right-padded length-bucketed batches), FP32
log-softmax over the vocabulary at the last prompt position, the log-likelihood of each one-token continuation, and
the FP32 softmax over those log-likelihoods. The first output line is `{"ready": true, "model": ..., "configuration":
...}`; before it the server checks that every continuation the template can ask for is one token after the
prompt's last token. A unit over the guard or with a multi-token continuation is rejected on its own; the rest of
its window is answered. The server keeps no cache: the caller stores responses.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, TextIO

from .task import REPO

MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MODEL = f"{MODEL_ID}@{REVISION}"
MODELS = {"qwen-base": MODEL}
HF_CACHE = REPO / "data" / "cache" / "huggingface"
# Identical to jev_opt.logprob.SCORING (the client refuses a server whose settings differ).
SCORING = {
    "backend": "qwen-base-logprob",
    "dtype": "bfloat16",
    "logit_dtype": "float32",
    "attention_implementation": "sdpa",
    "padding": "right",
    "option_scoring": "single-token continuation: FP32 log-softmax over the vocabulary at the position after the prompt; "
                      "probabilities = FP32 softmax over the options' log-likelihoods",
    "truncation_policy": "reject_complete_input_overflow_before_forward",
}
# Every continuation of template qwen-base-logprob-v1 (jev_opt.logprob.ALPHABET) and the prompt's last characters.
ALPHABET = [*"ABCDEFGHIJKLMNOPQRSTUVWXYZ", *"123456789", "Yes", "No"]
PROMPT_END = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


class LogprobContextError(ValueError):
    """The complete prompt does not fit the guard (truncation is forbidden)."""


class ContinuationError(ValueError):
    """A continuation is not exactly one token."""


@dataclass
class Runtime:
    """What scoring needs: the text decoder (hidden states), the output projection, tokenization and padding."""

    text_model: Any
    head: Any
    encode: Callable[[str], list[int]]
    pad_id: int


def _version(distribution: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def load_runtime(device: str, model: str = MODEL) -> Runtime:
    """The pinned release from `data/cache/huggingface` only (`local_files_only`)."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    model_id, revision = model.split("@", 1)
    path = Path(snapshot_download(model_id, revision=revision, cache_dir=str(HF_CACHE), local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    lm = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map={"": device},
                                                         attn_implementation="sdpa", local_files_only=True)
    lm.config.use_cache = False
    lm.eval()
    text = lm.model.language_model if hasattr(lm.model, "language_model") else lm.model
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return Runtime(text, lm.get_output_embeddings(), lambda s: tokenizer(s, add_special_tokens=False).input_ids, int(pad))


@dataclass
class LogprobScorer:
    device: str = "cuda:0"
    max_length: int = 32000
    model: str = MODEL
    batch_size: int = 16
    batch_padded_token_limit: int = 16384
    runtime_factory: Callable[[], Runtime] | None = None
    _runtime: Runtime | None = field(default=None, init=False, repr=False)

    @cached_property
    def configuration(self) -> dict[str, Any]:
        import torch

        device = torch.device(self.device)
        gpu = None
        if device.type == "cuda":
            index = device.index if device.index is not None else torch.cuda.current_device()
            gpu = {"name": torch.cuda.get_device_name(index), "compute_capability": list(torch.cuda.get_device_capability(index))}
        model_id, revision = self.model.split("@", 1)
        return {
            **SCORING, "model_id": model_id, "revision": revision, "max_length": self.max_length,
            "batch_size": self.batch_size, "batch_padded_token_limit": self.batch_padded_token_limit,
            "batching": "length-bucketed within bounded unit windows", "device": str(device), "gpu": gpu,
            "python": platform.python_version(),
            "libraries": {name: _version(name) for name in ("torch", "transformers", "huggingface-hub", "safetensors", "tokenizers",
                                                            "accelerate", "flash-linear-attention", "fla-core", "triton", "causal-conv1d")},
            "cuda_runtime": torch.version.cuda,
            "numerical_configuration": {
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            },
        }

    def runtime(self) -> Runtime:
        if self._runtime is None:
            rt = (self.runtime_factory or (lambda: load_runtime(self.device, self.model)))()
            self.check_alphabet(rt)
            self._runtime = rt
        return self._runtime

    @staticmethod
    def check_alphabet(rt: Runtime) -> None:
        """Every template continuation is one token, also when appended to the prompt's last characters."""
        end = rt.encode(PROMPT_END)
        bad = [c for c in ALPHABET if len(rt.encode(c)) != 1 or rt.encode(PROMPT_END + c) != end + rt.encode(c)]
        if bad:
            raise ContinuationError(f"continuations not single tokens after the prompt: {bad}")

    def encode(self, payload: dict[str, Any]) -> tuple[list[int], list[int]]:
        rt = self.runtime()
        ids = rt.encode(payload["prompt"])
        cont = []
        for c in payload["continuations"]:
            t = rt.encode(c)
            if len(t) != 1:
                raise ContinuationError(f"continuation {c!r} is {len(t)} tokens")
            cont.append(t[0])
        if len(ids) > self.max_length:
            raise LogprobContextError(f"complete prompt requires {len(ids)} tokens (limit {self.max_length}); truncation is forbidden")
        return ids, cont

    def _forward(self, group: list[tuple[str, list[int], list[int]]]) -> list[dict[str, Any]]:
        import torch

        rt = self.runtime()
        device = next(rt.head.parameters()).device
        width = max(len(ids) for _, ids, _ in group)
        input_ids = torch.full((len(group), width), rt.pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((len(group), width), dtype=torch.long, device=device)
        for i, (_, ids, _) in enumerate(group):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            mask[i, : len(ids)] = 1
        last = torch.tensor([len(ids) - 1 for _, ids, _ in group], device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            hidden = rt.text_model(input_ids=input_ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
            logp = torch.log_softmax(rt.head(hidden[torch.arange(len(group), device=device), last]).float(), dim=-1)
            option_lp = [logp[i, torch.tensor(cont, device=device)] for i, (_, _, cont) in enumerate(group)]
            probs = [torch.softmax(lp, dim=-1).tolist() for lp in option_lp]
            mass = [float(lp.exp().sum()) for lp in option_lp]
            option_lp = [lp.tolist() for lp in option_lp]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        return [{"model": self.model, "probabilities": p, "logprobs": lp, "option_mass": m,
                 "usage": {"input_tokens": len(ids), "output_tokens": 0},
                 "audit": {"inference_seconds": seconds / len(group), "batch_inference_seconds": seconds, "batch_size": len(group),
                           "batch_padded_tokens": len(group) * width, "continuation_token_ids": cont, "truncated": False}}
                for (_, ids, cont), p, lp, m in zip(group, probs, option_lp, mass)]

    def compute(self, items: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        """Results of one window in its order; rejected units are reported on their own."""
        done: dict[str, dict[str, Any]] = {}
        pending = []
        for key, payload in items:
            try:
                ids, cont = self.encode(payload)
            except ValueError as exc:
                done[key] = {"key": key, "error": str(exc)[:2000], "type": type(exc).__name__}
                continue
            pending.append((key, ids, cont))
        pending.sort(key=lambda r: len(r[1]))
        first = 0
        while first < len(pending):
            end = first + 1
            while (end < len(pending) and end - first < self.batch_size
                   and len(pending[end][1]) * (end - first + 1) <= self.batch_padded_token_limit):
                end += 1
            group = pending[first:end]
            for (key, _, _), response in zip(group, self._forward(group)):
                done[key] = {"key": key, "response": response}
            first = end
        return [done[key] for key, _ in items]


def serve(scorer: LogprobScorer, inp: TextIO, out: TextIO) -> None:
    scorer.runtime()  # load and check before the handshake: a failed load is reported, not a hung request
    out.write(json.dumps({"ready": True, "model": scorer.model, "configuration": scorer.configuration}) + "\n")
    out.flush()
    for line in inp:
        if not line.strip():
            continue
        message = json.loads(line)
        out.write(json.dumps({"results": scorer.compute([(key, body) for key, body in message["bodies"]])}, ensure_ascii=False) + "\n")
        out.flush()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--max-length", type=int, default=32000)
    args = ap.parse_args(argv)
    # The protocol owns the original stdout; library output (loading progress, warnings) goes to stderr.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    serve(LogprobScorer(device=args.device, max_length=args.max_length, model=MODELS[args.model]), sys.stdin, protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
