"""Line-delimited JSON server around the frozen local Clef runtime (jev-opt's Clef backend).

`python -m gauntlet.clef_serve --model clef-flash --device cuda:1` loads the pinned release once
(`ClefClient` loading, encoding, BF16 arithmetic, FP32 probabilities, complete-input guard) and
answers request windows sent by `jev_opt.clef`:

    -> {"bodies": [[key, body], ...]}
    <- {"results": [{"key": key, "response": response} | {"key": key, "error": message, "type": name}, ...]}

The first output line is `{"ready": true, "model": ..., "configuration": ...}`. The server keeps no
cache: the caller stores responses. A body the encoder rejects (context overflow, invalid schema) is
reported for that body only; every other body of its window is still answered.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, TextIO

from .clef import MODELS, ClefClient


def answer(client: ClefClient, bodies: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Responses of one window in its order; encoder rejections are isolated per body."""
    try:
        done = {key: response for group in client.compute_batches(dict(bodies)) for key, response in group}
        return [{"key": key, "response": done[key]} for key, _ in bodies]
    except ValueError:  # ClefContextError included: retry body by body to name the rejected ones
        out = []
        for key, body in bodies:
            try:
                (group,) = list(client.compute_batches({key: body}))
                out.append({"key": key, "response": group[0][1]})
            except ValueError as exc:
                out.append({"key": key, "error": str(exc)[:2000], "type": type(exc).__name__})
        return out


def serve(client: ClefClient, inp: TextIO, out: TextIO) -> None:
    client._runtime()  # load before the handshake: a failed load is reported, not a hung request
    out.write(json.dumps({"ready": True, "model": client.model, "configuration": client.metadata}) + "\n")
    out.flush()
    for line in inp:
        if not line.strip():
            continue
        message = json.loads(line)
        results = answer(client, [(key, body) for key, body in message["bodies"]])
        out.write(json.dumps({"results": results}, ensure_ascii=False) + "\n")
        out.flush()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--max-length", type=int, default=32000)
    ap.add_argument("--scratch-dir", type=Path, required=True, help="ClefClient's cache directory (unused by the server)")
    args = ap.parse_args(argv)
    # The protocol owns the original stdout; library output (loading progress, warnings) goes to stderr.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    client = ClefClient(args.scratch_dir, device=args.device, max_length=args.max_length, model=MODELS[args.model])
    serve(client, sys.stdin, protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
