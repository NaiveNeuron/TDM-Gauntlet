"""Behavioral compiler/driver boundaries and backward correctness for the LoRA preflight.

Run with GPUs hidden (CUDA_VISIBLE_DEVICES=); actual H100 kernels are exercised by both GPU preflights.
"""

import os

import pytest

from gauntlet import lora_preflight as PF

HAVE_TILELANG = PF._dist_version("tilelang") is not None and PF.pinned_cuda_home() is not None
needs_tilelang = pytest.mark.skipif(not HAVE_TILELANG, reason="TileLang and the standalone compiler are absent")


def good_static(home="/compiler"):
    return {"versions": dict(PF.PINS), "cuda_home": home, "pinned_cuda_home": home, "nvcc": f"{home}/bin/nvcc",
            "nvcc_release": "12.8", "nvcc_version": "12.8.93", "tilelang_cache_dir": str(PF.TILELANG_CACHE),
            "has_usable_nvcc": True, "tilelang_backend_available": True,
            "verifier": {"accepted": True, "reason": None}, "tilelang_forward_ops": []}


def good_gpu():
    return {"device": "cuda:0", "capability": [9, 0], "driver_cuda_version": 12080, "tilelang_backend_enabled": True,
            "tilelang_calls": 3, "errors": {"output": 0.004, "dq": 0.006, "dk": 0.006, "dv": 0.004, "dg": 0.01, "dbeta": 0.005}}


def test_pin_overrides_a_stray_compiler_without_duplicating_path(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "CUDA_HOME": "/usr/local/cuda-12.4"}
    home = tmp_path / "compiler"
    assert PF.pin_cuda_compiler(env, home) == str(home)
    assert env["CUDA_HOME"] == env["CUDA_PATH"] == str(home)
    assert env["PATH"] == f"{home / 'bin'}{os.pathsep}/usr/bin:/bin"
    PF.pin_cuda_compiler(env, home)
    assert env["PATH"].count(str(home / "bin")) == 1


def test_absent_pinned_compiler_does_not_fall_back_to_system_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(PF, "CUDA_ROOT", tmp_path / "absent")
    env = {"PATH": "/usr/local/cuda/bin:/usr/bin", "CUDA_HOME": "/usr/local/cuda"}
    original = dict(env)
    assert PF.pin_cuda_compiler(env) is None
    assert env == original


@pytest.mark.parametrize(("version", "accepted"), [("12.8.93", True), ("12.8.94", False), (None, False)])
def test_compiler_patch_version_is_not_interchangeable(version, accepted):
    rep = {**good_static(), "nvcc_version": version}
    assert (not PF.static_problems(rep)) is accepted


@pytest.mark.parametrize(("version", "accepted"), [(12079, False), (12080, True), (13000, True), (None, False)])
def test_driver_compatibility_boundary(version, accepted):
    rep = {**good_gpu(), "driver_cuda_version": version}
    assert (not PF.gpu_problems(rep)) is accepted


@needs_tilelang
def test_verifier_accepts_qwen_inputs_and_rejects_what_would_fall_back_to_triton():
    from fla.ops.common.backends.tilelang import TileLangBackend

    import torch

    be = TileLangBackend()
    assert be.chunk_bwd_dqkwg_verifier(**PF.qwen_dqkwg_inputs()) == (True, None)
    fp32_h = {**PF.qwen_dqkwg_inputs(), "h": torch.empty(1, 1, 48, 128, 128, device="meta")}
    assert be.chunk_bwd_dqkwg_verifier(**fp32_h)[0] is False


def test_comparison_harness_passes_the_reference_and_flags_a_wrong_kernel():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(4)
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule as ref

    same = PF.gdn_compare(ref, ref, "cpu")
    assert set(same) == {"output", "dq", "dk", "dv", "dg", "dbeta"} and max(same.values()) < PF.TOL

    def wrong_gate_grad(q, k, v, g, beta, **kw):
        g2 = g + (g * 0.5 - (g * 0.5).detach())
        return ref(q, k, v, g=g2, beta=beta, **kw)

    bad = PF.gdn_compare(wrong_gate_grad, ref, "cpu")
    assert bad["output"] < PF.TOL and bad["dg"] > PF.TOL
    assert PF.gpu_problems({**good_gpu(), "errors": bad})
