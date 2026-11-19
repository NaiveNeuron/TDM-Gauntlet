"""A tiny randomly initialised Qwen3_5ForConditionalGeneration with the real Qwen3.8 tokenizer, saved to a directory
(tests of gauntlet.lora_train and jev-opt/experiments/h100_lora.py; nothing is downloaded).

    .venv/bin/python tests/lora_tiny.py <dir>     # used by the jev-opt driver tests through a subprocess

The tokenizer files are copied from the pinned snapshot in data/cache/huggingface (only the small files; the
weights are not needed). Without them the tests that need this model are skipped.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO / "data" / "cache" / "huggingface" / "models--Qwen--Qwen3.8-27B" / "snapshots" / "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja")


def available() -> bool:
    return all((SNAPSHOT / f).exists() for f in (*TOKENIZER_FILES, "config.json"))


def build(dest: Path) -> Path:
    import torch
    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration

    cfg = AutoConfig.from_pretrained(SNAPSHOT)
    t = cfg.text_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 64, 128, 4
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 32
    t.layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    t.linear_num_key_heads, t.linear_num_value_heads, t.linear_key_head_dim, t.linear_value_head_dim = 2, 4, 16, 16
    t.mtp_num_hidden_layers = 0
    rope = dict(t.rope_parameters)
    rope["mrope_section"] = [2, 1, 1]  # head_dim 32 × partial 0.25 / 2
    t.rope_parameters = rope
    v = cfg.vision_config
    v.depth, v.hidden_size, v.intermediate_size, v.num_heads, v.out_hidden_size = 1, 32, 64, 2, 64
    v.num_position_embeddings = 64
    torch.manual_seed(0)
    model = Qwen3_5ForConditionalGeneration(cfg).to(torch.bfloat16)
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest)
    for f in TOKENIZER_FILES:
        shutil.copyfile(SNAPSHOT / f, dest / f)
    (dest / "tiny.json").write_text(json.dumps({"text_layers": t.num_hidden_layers, "hidden": t.hidden_size}))
    return dest


if __name__ == "__main__":
    build(Path(sys.argv[1]))
