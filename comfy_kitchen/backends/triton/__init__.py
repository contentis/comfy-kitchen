__all__ = [
    "adaln",
    "na3d",
    "rms_adaln",
    "apply_rope",
    "apply_rope_",
    "apply_rope1",
    "apply_rope1_",
    "apply_rope_split_half",
    "apply_rope_split_half_",
    "apply_rope_split_half1",
    "apply_rope_split_half1_",
    "rms_rope",
    "rms_rope_",
    "rms_rope1",
    "rms_rope1_",
    "rms_rope_split_half",
    "rms_rope_split_half_",
    "rms_rope_split_half1",
    "rms_rope_split_half1_",
    "dequantize_nvfp4",
    "dequantize_per_tensor_fp8",
    "quantize_mxfp8",
    "quantize_nvfp4",
    "quantize_per_tensor_fp8",
    "int8_linear",
    "quantize_int8_rowwise",
    "quantize_and_rotate_rowwise",
    "w4a8_int8_linear",
]

# Try to import triton and register if available
import torch

from comfy_kitchen import dispatch_v2
from comfy_kitchen.registry import registry

_TRITON_AVAILABLE = True
_TRITON_ERROR = None

try:
    import triton  # noqa: F401
    from comfy_kitchen.backends.eager.quantization import (
        quantize_and_rotate_rowwise as _eager_quantize_and_rotate_rowwise,
    )
    from comfy_kitchen.backends.eager.quantization import (
        quantize_int8_rowwise as _eager_quantize_int8_rowwise,
    )

    from .adaln import adaln, rms_adaln
    from .na import na3d
    from .quantization import (
        dequantize_nvfp4,
        dequantize_per_tensor_fp8,
        int8_linear,
        quantize_mxfp8,
        quantize_nvfp4,
        quantize_per_tensor_fp8,
    )
    from .quantization import (
        triton_quantize_and_rotate_rowwise as _triton_quantize_and_rotate_rowwise,
    )
    from .quantization import (
        triton_quantize_rowwise as _triton_quantize_int8_rowwise,
    )
    from .rms_rope import (
        rms_rope,
        rms_rope1,
        rms_rope1_,
        rms_rope_,
        rms_rope_split_half,
        rms_rope_split_half1,
        rms_rope_split_half1_,
        rms_rope_split_half_,
    )
    from .rope import (
        apply_rope,
        apply_rope1,
        apply_rope1_,
        apply_rope_,
        apply_rope_split_half,
        apply_rope_split_half1,
        apply_rope_split_half1_,
        apply_rope_split_half_,
    )
    from .w4a8_int8 import w4a8_int8_linear
except ImportError as e:
    _TRITON_AVAILABLE = False
    _TRITON_ERROR = f"ImportError: {e!s}"


if _TRITON_AVAILABLE:

    def quantize_int8_rowwise(
        x: torch.Tensor,
        stochastic_rounding: int | None = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if stochastic_rounding is not None and stochastic_rounding > 0:
            return _eager_quantize_int8_rowwise(x, stochastic_rounding=stochastic_rounding)
        return _triton_quantize_int8_rowwise(x)

    def quantize_and_rotate_rowwise(
        x: torch.Tensor,
        h: torch.Tensor,
        group_size: int,
        stochastic_rounding: int | None = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if stochastic_rounding is not None and stochastic_rounding > 0:
            return _eager_quantize_and_rotate_rowwise(
                x, h, group_size, stochastic_rounding=stochastic_rounding
            )
        return _triton_quantize_and_rotate_rowwise(x, h, group_size)


def _triton_lowerings():
    floats = frozenset({torch.float32, torch.float16, torch.bfloat16})
    fp8 = frozenset({torch.float8_e4m3fn, torch.float8_e5m2})
    d1 = dispatch_v2.ndim(1)
    d2 = dispatch_v2.ndim(2)
    d6 = dispatch_v2.ndim(6)
    tensor = dispatch_v2.tensor_rule
    value = dispatch_v2.value_rule
    tx = frozenset({"cuda", "xpu"})
    cuda = frozenset({"cuda"})

    def plan(runner, *, devices=tx, min_cc=None, check=None, **params):
        return dispatch_v2.lowering(runner, devices=devices, min_cc=min_cc, check=check, **params)

    f, f2, f6 = tensor(floats), tensor(floats, d2), tensor(floats, d6)
    i8 = tensor({torch.int8})
    u8_2 = tensor({torch.uint8}, d2)
    f32 = tensor({torch.float32})
    out = {
        "adaln": plan(adaln, x=f, scale=f, shift=f),
        "rms_adaln": plan(rms_adaln, x=f, scale=f, shift=f),
        "quantize_per_tensor_fp8": plan(
            quantize_per_tensor_fp8, x=f, scale=f32, output_type=(fp8, None)
        ),
        "dequantize_per_tensor_fp8": plan(
            dequantize_per_tensor_fp8, x=tensor(fp8), scale=f32, output_type=(floats, None)
        ),
        "quantize_nvfp4": plan(
            quantize_nvfp4, devices=cuda, min_cc=(10, 0), x=f2, per_tensor_scale=f32
        ),
        "dequantize_nvfp4": plan(
            dequantize_nvfp4,
            devices=cuda,
            min_cc=(10, 0),
            qx=u8_2,
            per_tensor_scale=f32,
            block_scales=tensor({torch.float8_e4m3fn}),
            output_type=(floats, None),
        ),
        "quantize_mxfp8": plan(quantize_mxfp8, devices=cuda, min_cc=(10, 0), x=f2),
        "int8_linear": plan(
            int8_linear,
            min_cc=(8, 0),
            x=f,
            weight=i8,
            weight_scale=f,
            out_dtype=(floats, None),
            convrot=value(bool),
            convrot_groupsize=value(int),
            input_act=value(str, type(None)),
        ),
        "w4a8_int8_linear": plan(
            w4a8_int8_linear,
            min_cc=(8, 0),
            x=f,
            qdata=tensor({torch.int8}, d2),
            s_rel=tensor({torch.float8_e4m3fn, torch.float32}, d2),
            s_channel=tensor({torch.float32}, d1),
            codebook=tensor({torch.float32}, d1),
            correction=tensor(floats, d2),
            bias=f,
            group_size=value(int),
            convrot_groupsize=value(int),
            out_dtype=(floats, None),
        ),
        "quantize_int8_rowwise": plan(quantize_int8_rowwise, x=f, stochastic_rounding=value(int)),
        "quantize_and_rotate_rowwise": plan(
            quantize_and_rotate_rowwise,
            x=f,
            h=f,
            group_size=value(int),
            stochastic_rounding=value(int),
        ),
    }
    for name in (
        "apply_rope1",
        "apply_rope1_",
        "apply_rope_split_half1",
        "apply_rope_split_half1_",
    ):
        out[name] = plan(globals()[name], x=f, freqs_cis=f)
    for name in ("apply_rope", "apply_rope_", "apply_rope_split_half", "apply_rope_split_half_"):
        out[name] = plan(globals()[name], xq=f, xk=f, freqs_cis=f)

    def na_check(call):
        q = call["q"]
        if any(call[name].shape != q.shape or call[name].dtype != q.dtype for name in ("k", "v")):
            return "q, k, and v must share shape and dtype"
        if len(call["kernel_size"]) != 3 or any(size < 1 for size in call["kernel_size"]):
            return "kernel_size must contain three positive values"
        if call["is_causal"] is not None and len(call["is_causal"]) != 3:
            return "is_causal must contain three values"
        if q.shape[-1] > 128:
            return "head_dim > 128 is unsupported"

    out["na3d"] = plan(na3d, devices=cuda, min_cc=(8, 0), check=na_check, q=f6, k=f6, v=f6)
    for name in ("rms_rope1", "rms_rope1_", "rms_rope_split_half1", "rms_rope_split_half1_"):
        out[name] = plan(globals()[name], x=f, freqs_cis=f, scale=f)
    for name in ("rms_rope", "rms_rope_", "rms_rope_split_half", "rms_rope_split_half_"):
        out[name] = plan(globals()[name], q=f, k=f, freqs_cis=f, q_scale=f, k_scale=f)
    return out


def _register():
    if not _TRITON_AVAILABLE:
        registry.mark_unavailable("triton", _TRITON_ERROR or "Triton not available")
        return

    has_cuda = torch.cuda.is_available()
    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()

    if not has_cuda and not has_xpu:
        registry.mark_unavailable("triton", "Neither CUDA nor XPU available on this system")
        return

    registry.register(
        name="triton",
        module=__import__(__name__, fromlist=__all__),
        lowerings=_triton_lowerings(),
    )


_register()
