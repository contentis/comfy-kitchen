from types import SimpleNamespace

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen import dispatch
from comfy_kitchen.backends.eager import rope as eager_rope
from comfy_kitchen.registry import BackendRegistry


def _inputs():
    q = torch.randn(1, 2, 3, 32)
    k = torch.randn_like(q)
    freqs = torch.randn(1, 1, 3, 16, 2, 2)
    q_scale = torch.randn(32)
    k_scale = torch.randn(32)
    return q, k, freqs, q_scale, k_scale


def test_public_rms_rope_uses_cached_implementation():
    q, k, freqs, q_scale, k_scale = _inputs()
    dispatch.operation("rms_rope").clear_cache()
    expected = eager_rope.rms_rope(q, k, freqs, q_scale, k_scale)
    with ck.use_backend("eager"):
        actual = ck.rms_rope(q, k, freqs, q_scale, k_scale)
        again = ck.rms_rope(q, k, freqs, q_scale, k_scale)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    torch.testing.assert_close(again[0], expected[0])
    assert dispatch.operation("rms_rope").cache_size == 1


def test_generic_operation_uses_cached_implementation():
    x = torch.randn(2, 3, 8)
    scale = torch.randn(2, 3, 8)
    shift = torch.randn(2, 3, 8)
    operation = dispatch.operation("adaln")
    operation.clear_cache()

    with ck.use_backend("eager"):
        expected = ck.adaln(x, scale, shift, 1e-6)
        actual = ck.adaln(x, scale, shift, 1e-6)

    torch.testing.assert_close(actual, expected)
    assert operation.cache_size == 1


def test_cpu_call_logs_rejections_and_selects_eager(caplog):
    q, k, freqs, q_scale, k_scale = _inputs()
    dispatch.operation("rms_rope").clear_cache()
    caplog.set_level(10, logger="comfy_kitchen.dispatch")

    dispatch.dispatch("rms_rope", q, k, freqs, q_scale, k_scale, 1e-6)

    assert "dispatch rms_rope -> eager (cache miss)" in caplog.text


def test_rot_dim_is_part_of_split_half_cache_identity():
    q, k, freqs, q_scale, k_scale = _inputs()
    operation = dispatch.operation("rms_rope_split_half")
    operation.clear_cache()

    with ck.use_backend("eager"):
        operation(q, k, freqs, q_scale, k_scale, 1e-6, 0)
        operation(q, k, freqs, q_scale, k_scale, 1e-6, 32)

    assert operation.cache_size == 2


def test_runtime_failure_does_not_retry_another_backend(monkeypatch):
    fallback_called = False

    def broken_runner():
        raise ValueError("kernel failed")

    def fallback_runner():
        nonlocal fallback_called
        fallback_called = True

    def implementations(_op_name):
        yield "first", broken_runner, frozenset()
        yield "fallback", fallback_runner, frozenset()

    monkeypatch.setattr(dispatch.registry, "iter_implementations", implementations)
    operation = dispatch.Operation("test")

    with pytest.raises(ValueError, match="kernel failed"):
        operation()


def test_unsupported_call_logs_reason_and_falls_through(monkeypatch, caplog):
    def rejected_runner():
        raise dispatch.UnsupportedCall("head dimension is unsupported")

    def eager_runner():
        return "eager result"

    def implementations(_op_name):
        yield "native", rejected_runner, frozenset()
        yield "eager", eager_runner, frozenset()

    monkeypatch.setattr(dispatch.registry, "iter_implementations", implementations)
    caplog.set_level(10, logger="comfy_kitchen.dispatch")

    assert dispatch.Operation("test")() == "eager result"
    assert "dispatch test rejected native: head dimension is unsupported" in caplog.text
    assert "dispatch test -> eager (cache miss)" in caplog.text


def test_scalar_types_have_distinct_cache_keys(monkeypatch, caplog):
    def native(flag):
        if not isinstance(flag, bool):
            raise dispatch.UnsupportedCall("requires bool")
        return "native result"

    def eager(flag):
        return "eager result"

    def implementations(_op_name):
        yield "native", native, frozenset()
        yield "eager", eager, frozenset()

    monkeypatch.setattr(dispatch.registry, "iter_implementations", implementations)
    caplog.set_level(10, logger="comfy_kitchen.dispatch")
    operation = dispatch.Operation("test")

    assert operation(True) == "native result"
    assert operation(1) == "eager result"
    assert "dispatch test rejected native: requires bool" in caplog.text


def test_cached_unsupported_call_reselects(monkeypatch, caplog):
    accepted = True
    native_calls = 0

    def native(flag):
        nonlocal native_calls
        native_calls += 1
        if not accepted:
            raise dispatch.UnsupportedCall("support changed")
        return "native result"

    def eager(flag):
        return "eager result"

    def implementations(_op_name):
        yield "native", native, frozenset()
        yield "eager", eager, frozenset()

    monkeypatch.setattr(dispatch.registry, "iter_implementations", implementations)
    caplog.set_level(10, logger="comfy_kitchen.dispatch")
    operation = dispatch.Operation("test")

    assert operation(True) == "native result"
    accepted = False
    assert operation(True) == "eager result"
    assert native_calls == 2
    assert "dispatch test rejected cached native: support changed" in caplog.text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_native_unsupported_dtype_falls_through_to_eager(caplog):
    x = torch.randn(8, device="cuda", dtype=torch.float64)
    rng = torch.randint(0, 256, x.shape, device="cuda", dtype=torch.uint8)
    dispatch.operation("stochastic_rounding_fp8").clear_cache()
    caplog.set_level(10, logger="comfy_kitchen.dispatch")

    result = dispatch.dispatch("stochastic_rounding_fp8", x, rng, torch.float8_e4m3fn)

    assert result.dtype == torch.float8_e4m3fn
    assert "dispatch stochastic_rounding_fp8 rejected cuda" in caplog.text
    assert "dispatch stochastic_rounding_fp8 -> eager" in caplog.text

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_fp8_normalizes_half_scale(caplog):
    x = torch.randn(32, device="cuda", dtype=torch.float32)
    scale = torch.tensor(1.0, device="cuda", dtype=torch.float16)
    dispatch.operation("quantize_per_tensor_fp8").clear_cache()

    with ck.use_backend("eager"):
        expected = dispatch.dispatch("quantize_per_tensor_fp8", x, scale)

    caplog.set_level(10, logger="comfy_kitchen.dispatch")
    actual = dispatch.dispatch("quantize_per_tensor_fp8", x, scale)

    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)
    assert "dispatch quantize_per_tensor_fp8 -> cuda" in caplog.text


def test_triton_int8_capability_rejects_pre_sm80(monkeypatch):
    quantization = pytest.importorskip("comfy_kitchen.backends.triton.quantization")
    tensor = SimpleNamespace(device=torch.device("cuda"))
    monkeypatch.setattr(quantization, "_cuda_capability", lambda _device: (7, 5))

    with pytest.raises(dispatch.UnsupportedCall, match=r"int8_linear requires compute capability 8\.0"):
        quantization._require_cuda_capability(tensor, (8, 0), "int8_linear")


def test_device_rejection_is_logged_before_calling_backend(monkeypatch, caplog):
    native_called = False

    def native_runner(x):
        nonlocal native_called
        native_called = True

    def eager_runner(x):
        return "eager result"

    def implementations(_op_name):
        yield "native", native_runner, frozenset({"cuda"})
        yield "eager", eager_runner, frozenset()

    monkeypatch.setattr(dispatch.registry, "iter_implementations", implementations)
    caplog.set_level(10, logger="comfy_kitchen.dispatch")

    assert dispatch.Operation("test")(torch.empty(1)) == "eager result"
    assert native_called is False
    assert "dispatch test rejected native: requires tensor device type cuda; got cpu" in caplog.text


def test_eager_operations_are_the_implicit_support_surface():
    def eager_impl(x):
        return x

    def private_impl(x):
        return x

    eager = SimpleNamespace(__all__=("op",), op=eager_impl)
    backend = SimpleNamespace(__all__=("op", "private"), op=eager_impl, private=private_impl)
    local_registry = BackendRegistry()
    local_registry.register("eager", eager)
    local_registry.register("test", backend)

    assert local_registry.list_backends()["test"]["capabilities"] == ["op"]


def test_late_eager_registration_reconciles_existing_backends():
    def implementation(x):
        return x

    eager = SimpleNamespace(__all__=("op",), op=implementation)
    backend = SimpleNamespace(
        __all__=("op", "private"), op=implementation, private=implementation
    )
    local_registry = BackendRegistry()
    local_registry.register("test", backend)
    local_registry.register("eager", eager)

    assert local_registry.list_backends()["test"]["capabilities"] == ["op"]


def test_late_eager_registration_validates_transactionally():
    def eager_impl(x, scale=1):
        return x * scale

    def incompatible(x, scale=2):
        return x * scale

    eager = SimpleNamespace(__all__=("op",), op=eager_impl)
    backend = SimpleNamespace(__all__=("op",), op=incompatible)
    local_registry = BackendRegistry()
    local_registry.register("test", backend)

    with pytest.raises(TypeError, match="does not match the eager signature"):
        local_registry.register("eager", eager)

    assert not local_registry.is_available("eager")
    assert local_registry.list_backends()["test"]["capabilities"] == ["op"]


def test_registration_is_atomic_when_an_implementation_is_missing():
    backend = SimpleNamespace(__all__=("missing",))
    local_registry = BackendRegistry()

    with pytest.raises(AttributeError):
        local_registry.register("test", backend)

    assert not local_registry.is_available("test")


def test_registration_rejects_incompatible_defaults_transactionally():
    def eager_impl(x, scale=1):
        return x * scale

    def incompatible(x, scale=2):
        return x * scale

    eager = SimpleNamespace(__all__=("op",), op=eager_impl)
    backend = SimpleNamespace(__all__=("op",), op=incompatible)
    local_registry = BackendRegistry()
    local_registry.register("eager", eager)

    with pytest.raises(TypeError, match="does not match the eager signature"):
        local_registry.register("test", backend)

    assert not local_registry.is_available("test")


def test_registration_rejects_incompatible_signature_transactionally():
    def eager_impl(x, scale=1):
        return x * scale

    def incompatible(x, weight=1):
        return x * weight

    eager = SimpleNamespace(__all__=("op",), op=eager_impl)
    backend = SimpleNamespace(__all__=("op",), op=incompatible)
    local_registry = BackendRegistry()
    local_registry.register("eager", eager)

    with pytest.raises(TypeError, match="does not match the eager signature"):
        local_registry.register("test", backend, {"op"})

    assert not local_registry.is_available("test")
