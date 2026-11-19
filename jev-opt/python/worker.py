# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "jev-opt",
#     "gepa==0.1.4",
#     "pydantic>=2.7,<3",
#     "pyyaml>=6,<7",
#     "httpx>=0.27,<1",
#     "numpy>=1.26,<3",
#     "scipy>=1.11,<2",
#     "scikit-learn>=1.4,<2",
#     "sentence-transformers>=3,<6",
# ]
#
# [tool.uv.sources]
# jev-opt = { path = ".", editable = true }
# ///
"""jev-opt worker: `uv run <plugin>/python/worker.py` speaks JSONL RPC on stdin/stdout."""

from jev_opt.worker import main

if __name__ == "__main__":
    raise SystemExit(main())
