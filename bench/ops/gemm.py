from __future__ import annotations

from typing import Any

import torch
from torch.utils.flop_counter import mm_flop

from bench.ops._base import OpSpec, derived, elem, register

_CONVROT_GROUP = 256


def estimate(case: dict) -> tuple[float, float]:
    s, d = case["static"], case["dynamic"]
    e, m, n, k = elem(s.get("dtype")), int(d["M"]), int(s["N"]), int(s["K"])
    bias = e * n if s.get("bias") else 0
    # Convrot adds a grouped Hadamard on activations (approx K^2/group per row).
    extra = 0.0
    if s.get("convrot"):
        g = int(s.get("convrot_groupsize", _CONVROT_GROUP))
        extra = float(2 * m * k * g)  # rough FLOP proxy for rotate
    return float(mm_flop((m, k), (k, n)) + extra), float(e * (m * k + m * n) + k * n + 4 + bias)


def derive(case: dict, kernel: str) -> list[dict] | None:
    s, d = case["static"], case["dynamic"]
    m, n, k = int(d.get("M", 0)), int(s.get("N", 0)), int(s.get("K", 0))
    if min(m, n, k) <= 0:
        return None
    base = {
        "N": n,
        "K": k,
        "bias": bool(s.get("bias", False)),
        "dtype": s.get("dtype", "bfloat16"),
        "convrot_groupsize": _CONVROT_GROUP,
    }
    out = [
        derived(
            kernel,
            case,
            static={**base, "convrot": False},
            dynamic={"M": m},
        )
    ]
    # Online rotation path needs K aligned to the Hadamard group size.
    if k % _CONVROT_GROUP == 0:
        out.append(
            derived(
                kernel,
                case,
                static={**base, "convrot": True},
                dynamic={"M": m},
            )
        )
    return out


def kwargs(case: dict, dtype: torch.dtype, device: torch.device, backend: str) -> dict[str, Any]:
    del backend
    static, dynamic = case["static"], case["dynamic"]
    m, n, k = int(dynamic["M"]), int(static["N"]), int(static["K"])
    return {
        "x": torch.randn((m, k), dtype=dtype, device=device),
        "weight": torch.randint(-127, 128, (n, k), dtype=torch.int8, device=device),
        "weight_scale": torch.ones((), dtype=torch.float32, device=device),
        "bias": (torch.randn(n, dtype=dtype, device=device) if static.get("bias") else None),
        "out_dtype": dtype,
        "convrot": bool(static.get("convrot", False)),
        "convrot_groupsize": int(static.get("convrot_groupsize", _CONVROT_GROUP)),
    }


register(
    OpSpec(
        ops=("int8_linear",),
        sources=("gemm",),
        derive=derive,
        estimate=estimate,
        kwargs=kwargs,
        primary_param="x",
    )
)
