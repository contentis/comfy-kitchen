from __future__ import annotations

from typing import Any

import torch

from bench.ops._base import OpSpec, ceil_to, derived, elem, prequantize, register

# format / optional padding / optional packing / optional scale_mode
QDQ = {
    "quantize_per_tensor_fp8": {"formats": ("float8_e4m3fn", "float8_e5m2")},
    "dequantize_per_tensor_fp8": {"formats": ("float8_e4m3fn", "float8_e5m2")},
    "stochastic_rounding_fp8": {"formats": ("float8_e4m3fn", "float8_e5m2")},
    "quantize_nvfp4": {"format": "nvfp4_e2m1_e4m3", "pad": 16, "packing": "hi_first"},
    "dequantize_nvfp4": {"format": "nvfp4_e2m1_e4m3", "pad": 16, "packing": "hi_first"},
    "quantize_mxfp8": {"format": "mxfp8_e4m3_e8m0", "pad": 32},
    "dequantize_mxfp8": {"format": "mxfp8_e4m3_e8m0", "pad": 32},
    "quantize_int8_tensorwise": {"format": "int8", "scale": "tensorwise"},
    "quantize_int8_rowwise": {"format": "int8", "scale": "rowwise"},
    "dequantize_int8_simple": {"format": "int8", "scale": "by_role"},
}


def _gemm_views(case: dict) -> list[tuple[str, int, int]]:
    s, d = case["static"], case["dynamic"]
    return [
        ("activation", int(d.get("M", 0)), int(s.get("K", 0))),
        ("weight", int(s.get("N", 0)), int(s.get("K", 0))),
    ]


def _shape(case: dict) -> tuple[int, int]:
    return int(case["dynamic"]["rows"]), int(case["static"]["cols"])


def estimate(case: dict) -> tuple[float, float]:
    s, d = case["static"], case["dynamic"]
    e = elem(s.get("dtype"))
    rows = int(d.get("padded_rows", d["rows"]))
    cols = int(s.get("padded_cols", s["cols"]))
    n = rows * cols
    packed = n / 2 if str(s.get("format", "")).startswith("nvfp4") else n
    return 0.0, float(e * int(d["rows"]) * int(s["cols"]) + packed)


def derive(case: dict, kernel: str) -> list[dict]:
    spec, dtype = QDQ[kernel], case["static"].get("dtype", "bfloat16")
    formats = spec.get("formats") or (spec["format"],)
    pad = spec.get("pad")
    out = []
    for role, rows, cols in _gemm_views(case):
        if min(rows, cols) <= 0:
            continue
        for fmt in formats:
            static: dict[str, Any] = {"cols": cols, "format": fmt, "dtype": dtype}
            dynamic: dict[str, Any] = {"rows": rows}
            if pad:
                static["padded_cols"] = ceil_to(cols, pad)
                static["padding"] = pad
                dynamic["padded_rows"] = ceil_to(rows, pad)
            if "packing" in spec:
                static["packing"] = spec["packing"]
            if "scale" in spec:
                mode = spec["scale"]
                static["scale_mode"] = (
                    ("rowwise" if role == "activation" else "tensorwise")
                    if mode == "by_role"
                    else mode
                )
            out.append(derived(kernel, case, static=static, dynamic=dynamic, layout="row_major"))
    return out


def _fp8_dtype(case: dict) -> torch.dtype:
    return {
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2,
    }[case["static"]["format"]]


def kwargs(case: dict, dtype: torch.dtype, device: torch.device, backend: str) -> dict[str, Any]:
    op_name = case["op"]
    rows, cols = _shape(case)

    if op_name in {
        "quantize_per_tensor_fp8",
        "dequantize_per_tensor_fp8",
        "stochastic_rounding_fp8",
    }:
        fp8_dtype = _fp8_dtype(case)
        if op_name == "dequantize_per_tensor_fp8":
            return {
                "x": torch.randn((rows, cols), dtype=dtype, device=device).to(fp8_dtype),
                "scale": torch.ones((), dtype=torch.float32, device=device),
                "output_type": dtype,
            }
        x = torch.randn((rows, cols), dtype=dtype, device=device)
        if op_name == "stochastic_rounding_fp8":
            return {
                "x": x,
                "rng": torch.randint(0, 256, x.shape, dtype=torch.uint8, device=device),
                "output_type": fp8_dtype,
            }
        return {
            "x": x,
            "scale": torch.ones((), dtype=torch.float32, device=device),
            "output_type": fp8_dtype,
        }

    if op_name in {"quantize_nvfp4", "dequantize_nvfp4"}:
        source = torch.randn((rows, cols), dtype=dtype, device=device)
        tensor_scale = torch.ones((1,), dtype=torch.float32, device=device)
        quant_kwargs = {
            "x": source,
            "per_tensor_scale": tensor_scale,
            "epsilon": 0.0,
            "pad_16x": True,
            "hi_first": True,
        }
        if op_name == "quantize_nvfp4":
            return quant_kwargs
        qx, block_scales = prequantize("quantize_nvfp4", quant_kwargs, backend=backend)
        return {
            "qx": qx,
            "per_tensor_scale": tensor_scale,
            "block_scales": block_scales,
            "output_type": dtype,
            "hi_first": True,
        }

    if op_name in {"quantize_mxfp8", "dequantize_mxfp8"}:
        quant_kwargs = {
            "x": torch.randn((rows, cols), dtype=dtype, device=device),
            "pad_32x": True,
        }
        if op_name == "quantize_mxfp8":
            return quant_kwargs
        qx, block_scales = prequantize("quantize_mxfp8", quant_kwargs, backend=backend)
        return {"qx": qx, "block_scales": block_scales, "output_type": dtype}

    source = torch.randn((rows, cols), dtype=dtype, device=device)
    if op_name == "quantize_int8_tensorwise":
        return {"x": source, "scale": None, "stochastic_rounding": 0}
    if op_name == "quantize_int8_rowwise":
        return {"x": source, "stochastic_rounding": 0}
    quant_op = (
        "quantize_int8_rowwise"
        if case["static"]["scale_mode"] == "rowwise"
        else "quantize_int8_tensorwise"
    )
    quant_kwargs: dict[str, Any] = {"x": source, "stochastic_rounding": 0}
    if quant_op == "quantize_int8_tensorwise":
        quant_kwargs["scale"] = None
    q, scale = prequantize(quant_op, quant_kwargs, backend=backend)
    return {"q": q, "scale": scale}


def _register(
    ops: tuple[str, ...],
    *,
    primary_param: str,
    fixed_dtypes: tuple[torch.dtype, ...] | None = None,
) -> None:
    register(
        OpSpec(
            ops=ops,
            sources=("gemm",),
            derive=derive,
            estimate=estimate,
            kwargs=kwargs,
            primary_param=primary_param,
            fixed_dtypes=fixed_dtypes,
        )
    )


_register(("quantize_per_tensor_fp8", "stochastic_rounding_fp8"), primary_param="x")
_register(("dequantize_per_tensor_fp8",), primary_param="output_type")
_register(("quantize_nvfp4",), primary_param="x")
_register(("dequantize_nvfp4",), primary_param="output_type")
_register(("quantize_mxfp8",), primary_param="x")
_register(("dequantize_mxfp8",), primary_param="output_type")
_register(("quantize_int8_tensorwise", "quantize_int8_rowwise"), primary_param="x")
_register(
    ("dequantize_int8_simple",),
    primary_param="x",
    fixed_dtypes=(torch.float32,),
)
