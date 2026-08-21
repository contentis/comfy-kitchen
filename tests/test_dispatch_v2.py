import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen import dispatch_v2
from comfy_kitchen.backends.eager import rope as eager_rope


def _inputs():
    q = torch.randn(1, 2, 3, 32)
    k = torch.randn_like(q)
    freqs = torch.randn(1, 1, 3, 16, 2, 2)
    q_scale = torch.randn(32)
    k_scale = torch.randn(32)
    return q, k, freqs, q_scale, k_scale


def test_public_rms_rope_uses_cached_v2_plan():
    q, k, freqs, q_scale, k_scale = _inputs()
    dispatch_v2.rms_rope.clear_cache()
    expected = eager_rope.rms_rope(q, k, freqs, q_scale, k_scale)
    with ck.use_backend("eager"):
        actual = ck.rms_rope(q, k, freqs, q_scale, k_scale)
        again = ck.rms_rope(q, k, freqs, q_scale, k_scale)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    torch.testing.assert_close(again[0], expected[0])
    assert dispatch_v2.rms_rope.cache_size == 1


def test_generic_operation_uses_cached_v2_plan():
    x = torch.randn(2, 3, 8)
    scale = torch.randn(2, 3, 8)
    shift = torch.randn(2, 3, 8)
    operation = dispatch_v2.operation("adaln")
    operation.clear_cache()

    with ck.use_backend("eager"):
        expected = ck.adaln(x, scale, shift, 1e-6)
        actual = ck.adaln(x, scale, shift, 1e-6)

    torch.testing.assert_close(actual, expected)
    assert operation.cache_size == 1


def test_cpu_call_declines_cuda_and_selects_eager():
    q, k, freqs, q_scale, k_scale = _inputs()
    backend, _ = dispatch_v2.rms_rope.select(
        q,
        k,
        freqs,
        q_scale,
        k_scale,
        1e-6,
    )
    assert backend == "eager"


def test_rot_dim_is_part_of_split_half_plan_identity():
    q, k, freqs, q_scale, k_scale = _inputs()
    operation = dispatch_v2.rms_rope_split_half
    operation.clear_cache()

    with ck.use_backend("eager"):
        operation.select(q, k, freqs, q_scale, k_scale, 1e-6, 0)
        operation.select(q, k, freqs, q_scale, k_scale, 1e-6, 16)

    assert operation.cache_size == 2


def test_runtime_failure_does_not_retry_another_backend(monkeypatch):
    fallback_called = False

    def broken_runner():
        raise ValueError("kernel failed")

    def fallback_runner():
        nonlocal fallback_called
        fallback_called = True

    def lowerings(_op_name):
        yield "first", lambda: broken_runner
        yield "fallback", lambda: fallback_runner

    monkeypatch.setattr(dispatch_v2.registry, "iter_lowerings", lowerings)
    operation = dispatch_v2.Operation("test")

    with pytest.raises(ValueError, match="kernel failed"):
        operation()
