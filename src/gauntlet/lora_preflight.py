"""Preflight for FLA's Hopper backward through TileLang (LoRA fine-tuning of Qwen3.8-27B, `gauntlet.lora_train`).

FLA 0.5.2 refuses its Triton kernel for the gated `chunk_bwd_dqkwg` (the backward of the gated delta rule in
Qwen3.8's linear-attention layers) on Hopper with Triton < 3.7.1; Torch 2.11 pins Triton 3.6.0. FLA routes that one
kernel to TileLang when TileLang is installed and `nvcc` is usable. The `lora` extra pins TileLang; a standalone
CUDA 12.8.1 compiler is pinned under `data/cache/cuda-lora-12.8.1/` (the H100 driver used, 570.195.03, reports
CUDA 12.8 and cannot load CUDA 13.0 kernels). `pin_cuda_compiler` points CUDA_HOME/CUDA_PATH/PATH at it before FLA
or TileLang look for a compiler.

Checks (each failure is a problem string; the trainer's `preflight` command stops with status 4 on any):
  static  pinned TileLang and nvcc (release 12.8, full version 12.8.93), `has_usable_nvcc()`, backend available,
          accepting Qwen3.8-shaped `chunk_bwd_dqkwg` inputs, no TileLang implementation of a forward operation
  compile FLA's TileLang `chunk_bwd_dqkwg` kernel compiles for sm_90a (no GPU needed)
  gpu     driver CUDA version >= 12.8, Hopper, backend enabled; one gated delta rule forward + backward through FLA
          on the GPU calls the TileLang kernel and matches Transformers' pure-PyTorch reference (FP32) within 0.05
          relative L2 error on the output and every gradient
"""

from __future__ import annotations

import contextlib
import ctypes
import importlib.metadata
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

PINS = {"tilelang": "0.1.12"}
NVCC_RELEASE = "12.8"
NVCC_VERSION = "12.8.93"
CUDA_ROOT = Path(__file__).resolve().parents[2] / "data" / "cache" / "cuda-lora-12.8.1"
TILELANG_CACHE = CUDA_ROOT.parent / f"tilelang-lora-nvcc-{NVCC_VERSION}"
DRIVER_MIN = 12080  # cuDriverGetVersion of CUDA 12.8
CAPABILITY = (9, 0)  # H100 (Hopper)
TOL = 0.05  # relative L2 error, FLA (TileLang backward) vs the FP32 PyTorch reference
# FLA operations dispatched through the "common" registry that Qwen3.8's forward pass uses: TileLang must implement
# none of them, so the scoring forward stays that of the base model's option log-probability scoring.
FORWARD_OPS = ("chunk_fwd_o", "chunk_gated_delta_rule_fwd_h", "chunk_scaled_dot_kkt_fwd", "fused_beta_sigmoid_fwd")
# Qwen3.8's linear attention after Transformers repeats q/k to the value heads: 48 heads, head dimension 128.
QWEN_HEADS, QWEN_DIM = 48, 128
CHECK = {"batch": 1, "tokens": 256, "heads": 4, "dim": QWEN_DIM, "seed": 0}


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def pinned_cuda_home() -> Path | None:
    """The pinned standalone compiler root; never a system or Python-wheel fallback."""
    return CUDA_ROOT if (CUDA_ROOT / "bin" / "nvcc").is_file() else None


def pin_cuda_compiler(environ: Any = os.environ, home: Path | None = None) -> str | None:
    """Point CUDA_HOME, CUDA_PATH and the head of PATH at the pinned compiler (None if it is absent).
    Must run before FLA or TileLang is imported: both resolve the compiler once."""
    home = home if home is not None else pinned_cuda_home()
    if home is None:
        return None
    environ["CUDA_HOME"] = environ["CUDA_PATH"] = str(home)
    # TileLang's kernel/binary keys omit the compiler version; never reuse CUDA 13 cubins.
    environ["TILELANG_CACHE_DIR"] = str(TILELANG_CACHE)
    bin_dir = str(home / "bin")
    path = environ.get("PATH", "")
    if path.split(os.pathsep)[0] != bin_dir:
        environ["PATH"] = bin_dir + (os.pathsep + path if path else "")
    return str(home)


def nvcc_version(nvcc: str | None) -> str | None:
    if nvcc is None:
        return None
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=60, check=False).stdout
    except OSError:
        return None
    m = re.search(r"\bV(\d+\.\d+\.\d+)\b", out)
    return m.group(1) if m else None


# --------------------------------------------------------------------------------------------
# static checks (no GPU)


def qwen_dqkwg_inputs(device: str = "meta") -> dict[str, Any]:
    """`chunk_bwd_dqkwg` arguments with Qwen3.8's shapes and dtypes (bf16 q/k/v/h, FP32 gate), as FLA's gated delta
    rule backward passes them; on the meta device for the verifier."""
    import torch

    B, T, H, D = 1, 64, QWEN_HEADS, QWEN_DIM
    bf = {"dtype": torch.bfloat16, "device": device}
    return {"q": torch.empty(B, T, H, D, **bf), "k": torch.empty(B, T, H, D, **bf), "v": torch.empty(B, T, H, D, **bf),
            "do": torch.empty(B, T, H, D, **bf), "h": torch.empty(B, 1, H, D, D, **bf), "dh": torch.empty(B, 1, H, D, D, **bf),
            "w": torch.empty(B, T, H, D, **bf), "g": torch.empty(B, T, H, dtype=torch.float32, device=device),
            "dv": torch.empty(B, T, H, D, **bf), "scale": D ** -0.5, "chunk_size": 64}


def static_report() -> dict[str, Any]:
    from fla.ops.common.backends.tilelang import TileLangBackend
    from fla.utils import has_usable_nvcc

    nvcc = shutil.which("nvcc")
    ok, reason = TileLangBackend().chunk_bwd_dqkwg_verifier(**qwen_dqkwg_inputs())
    home = pinned_cuda_home()
    version = nvcc_version(nvcc)
    return {"versions": {k: _dist_version(k) for k in PINS}, "cuda_home": os.environ.get("CUDA_HOME"),
            "pinned_cuda_home": str(home) if home else None, "nvcc": nvcc,
            "nvcc_release": version.rsplit(".", 1)[0] if version else None, "nvcc_version": version,
            "tilelang_cache_dir": os.environ.get("TILELANG_CACHE_DIR"),
            "has_usable_nvcc": bool(has_usable_nvcc()), "tilelang_backend_available": bool(TileLangBackend.is_available()),
            "verifier": {"accepted": bool(ok), "reason": reason},
            "tilelang_forward_ops": [op for op in FORWARD_OPS if getattr(TileLangBackend, op, None) is not None]}


def static_problems(rep: dict[str, Any]) -> list[str]:
    p = []
    for name, want in PINS.items():
        if rep["versions"].get(name) != want:
            p.append(f"{name} {rep['versions'].get(name)} installed, {want} pinned (sync the lora extra)")
    if rep["pinned_cuda_home"] is None:
        p.append(f"the pinned standalone CUDA compiler is absent from {CUDA_ROOT}")
    elif rep["cuda_home"] != rep["pinned_cuda_home"]:
        p.append(f"CUDA_HOME {rep['cuda_home']} is not the pinned compiler {rep['pinned_cuda_home']}")
    if rep["nvcc"] is None or rep["pinned_cuda_home"] is None or Path(rep["nvcc"]).resolve().parent.parent != Path(rep["pinned_cuda_home"]):
        p.append(f"nvcc on PATH ({rep['nvcc']}) is not the pinned compiler")
    if rep["nvcc_release"] != NVCC_RELEASE:
        p.append(f"nvcc release {rep['nvcc_release']} (expected {NVCC_RELEASE})")
    if rep.get("nvcc_version") != NVCC_VERSION:
        p.append(f"nvcc version {rep.get('nvcc_version')} (expected {NVCC_VERSION})")
    if rep.get("tilelang_cache_dir") != str(TILELANG_CACHE):
        p.append(f"TileLang cache {rep.get('tilelang_cache_dir')} is not the compiler-specific cache {TILELANG_CACHE}")
    if not rep["has_usable_nvcc"]:
        p.append("FLA's has_usable_nvcc() is false: FLA would not use TileLang")
    if not rep["tilelang_backend_available"]:
        p.append("FLA's TileLang backend is not available (TileLang import or nvcc)")
    if not rep["verifier"]["accepted"]:
        p.append(f"FLA's TileLang verifier rejects Qwen3.8's chunk_bwd_dqkwg inputs: {rep['verifier']['reason']}")
    if rep["tilelang_forward_ops"]:
        p.append(f"TileLang implements forward operations {rep['tilelang_forward_ops']}: the scoring forward would change")
    return p


def compile_check(arch: str = "sm_90a") -> dict[str, Any]:
    """Compile FLA's TileLang `chunk_bwd_dqkwg` kernel for Qwen3.8's shapes (bf16, 48 heads, dimension 128, gated,
    with w) with the pinned nvcc, for `arch`, without a GPU."""
    import time

    import tilelang
    from fla.ops.common.backends.tilelang.chunk_bwd import _build_kernel

    t0 = time.monotonic()
    try:
        tir = _build_kernel.get_tir(1, QWEN_HEADS, QWEN_HEADS, QWEN_DIM, QWEN_DIM, 64, 64, 64, 2, QWEN_DIM, QWEN_DIM, "bfloat16",
                                    True, True, False, False, num_warps=4)
        kernel = tilelang.compile(tir, target={"kind": "cuda", "arch": arch}, execution_backend="tvm_ffi",
                                  pass_configs=dict(_build_kernel.pass_configs or {}))
        return {"arch": arch, "ok": True, "source_chars": len(kernel.get_kernel_source()), "seconds": round(time.monotonic() - t0, 3)}
    except Exception as exc:  # noqa: BLE001  (any compiler failure is a preflight problem, reported verbatim)
        return {"arch": arch, "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:2000]}", "seconds": round(time.monotonic() - t0, 3)}


def compile_problems(rep: dict[str, Any]) -> list[str]:
    return [] if rep["ok"] else [f"TileLang could not compile FLA's chunk_bwd_dqkwg for {rep['arch']}: {rep['error'][:500]}"]


# --------------------------------------------------------------------------------------------
# GPU checks


def driver_cuda_version(lib: str = "libcuda.so.1") -> int | None:
    """cuDriverGetVersion (e.g. 13000 for CUDA 13.0); None without a driver."""
    try:
        cuda = ctypes.CDLL(lib)
    except OSError:
        return None
    v = ctypes.c_int()
    return v.value if cuda.cuDriverGetVersion(ctypes.byref(v)) == 0 else None


@contextlib.contextmanager
def count_calls(backend: Any, name: str) -> Iterator[list[int]]:
    """Count calls of `backend.<name>` (an instance attribute shadows the method; FLA's dispatcher looks it up with
    getattr on the registered instance)."""
    calls = [0]
    original = getattr(backend, name)

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        return original(*args, **kwargs)

    setattr(backend, name, counted)
    try:
        yield calls
    finally:
        delattr(backend, name)


def rel_l2(a: Any, b: Any) -> float:
    a, b = a.float(), b.float()
    den = float(b.norm())
    return float((a - b).norm()) / den if den > 0 else float((a - b).norm())


def gdn_compare(fla_fn: Callable[..., Any], ref_fn: Callable[..., Any], device: str) -> dict[str, float]:
    """Gated delta rule forward + backward through `fla_fn` (bf16 inputs, as Qwen3.8 calls it) and through `ref_fn`
    in FP32 on the same values; relative L2 error of the output and of the gradients of q, k, v, g and beta."""
    import torch

    gen = torch.Generator(device="cpu").manual_seed(CHECK["seed"])
    B, T, H, D = CHECK["batch"], CHECK["tokens"], CHECK["heads"], CHECK["dim"]

    def rnd(*shape: int) -> Any:
        return torch.randn(*shape, generator=gen)

    base = {"q": rnd(B, T, H, D), "k": rnd(B, T, H, D), "v": rnd(B, T, H, D),
            "g": torch.nn.functional.logsigmoid(rnd(B, T, H)), "beta": torch.sigmoid(rnd(B, T, H))}
    do = rnd(B, T, H, D).to(torch.bfloat16).to(device)
    dtypes = {"q": torch.bfloat16, "k": torch.bfloat16, "v": torch.bfloat16, "g": torch.float32, "beta": torch.bfloat16}
    low = {k: v.to(dtypes[k]).to(device).requires_grad_() for k, v in base.items()}
    ref = {k: v.detach().float().clone().requires_grad_() for k, v in low.items()}
    out = fla_fn(low["q"], low["k"], low["v"], g=low["g"], beta=low["beta"], use_qk_l2norm_in_kernel=True)[0]
    (out.float() * do.float()).sum().backward()
    out_ref = ref_fn(ref["q"], ref["k"], ref["v"], g=ref["g"], beta=ref["beta"], use_qk_l2norm_in_kernel=True)[0]
    (out_ref * do.float()).sum().backward()
    errors = {"output": rel_l2(out.detach(), out_ref.detach())}
    errors.update({f"d{k}": rel_l2(low[k].grad, ref[k].grad) for k in low})
    return errors


def gpu_report(device: str) -> dict[str, Any]:
    import torch
    from fla.ops.backends import BackendRegistry
    from fla.ops.common.backends.tilelang import TileLangBackend
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    torch.cuda.set_device(device)
    rep: dict[str, Any] = {"device": device, "gpu": torch.cuda.get_device_name(device),
                           "capability": list(torch.cuda.get_device_capability(device)), "driver_cuda_version": driver_cuda_version(),
                           "tilelang_backend_enabled": bool(TileLangBackend.is_enabled())}
    BackendRegistry.ensure_initialized("common")
    backend = BackendRegistry._registries["common"]._backends.get("tilelang")
    if backend is None:
        rep.update(tilelang_calls=0, errors=None, error="FLA registered no TileLang backend for 'common'")
        return rep
    try:
        with count_calls(backend, "chunk_bwd_dqkwg") as calls:
            rep["errors"] = gdn_compare(chunk_gated_delta_rule, torch_chunk_gated_delta_rule, device)
        rep["tilelang_calls"] = calls[0]
    except Exception as exc:  # noqa: BLE001  (FLA's guard, a TileLang compile or launch failure: reported verbatim)
        rep.update(tilelang_calls=None, errors=None, error=f"{type(exc).__name__}: {str(exc)[:2000]}")
    return rep


def gpu_problems(rep: dict[str, Any]) -> list[str]:
    p = []
    dv = rep.get("driver_cuda_version")
    if dv is None or dv < DRIVER_MIN:
        p.append(f"driver supports CUDA {dv} (cuDriverGetVersion); the pinned CUDA {NVCC_RELEASE} compiler needs >= {DRIVER_MIN}")
    if tuple(rep.get("capability") or ()) != CAPABILITY:
        p.append(f"GPU capability {rep.get('capability')} is not Hopper {list(CAPABILITY)}")
    if not rep.get("tilelang_backend_enabled"):
        p.append("FLA's TileLang backend is not enabled on this GPU")
    if rep.get("error"):
        p.append(f"gated delta rule forward/backward failed: {rep['error'][:500]}")
    elif not rep.get("tilelang_calls"):
        p.append("the backward did not call FLA's TileLang chunk_bwd_dqkwg")
    for k, v in (rep.get("errors") or {}).items():
        if not v <= TOL:  # also catches NaN
            p.append(f"{k}: relative L2 error {v:.4g} vs the FP32 PyTorch reference exceeds {TOL}")
    return p
