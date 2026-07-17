from __future__ import annotations

from typing import Any

import torch

from bench.ops._base import OpSpec, derived, elem, register


def estimate(case: dict) -> tuple[float, float]:
    s, d = case["static"], case["dynamic"]
    e, seq, h, hd = elem(s.get("dtype")), int(d["seq"]), int(s["heads"]), int(s["head_dim"])
    scales = 8 * hd if case["op"].startswith("rms_") else 0
    return 0.0, float(e * 4 * seq * h * hd + e * seq * hd * 2 + scales)


def derive(case: dict, kernel: str) -> dict | None:
    s, d = case["static"], case["dynamic"]
    if "batch" in d or int(s.get("batch", 1)) != 1:
        return None
    seq = int(d.get("q_tokens", 0))
    if seq <= 0:
        return None
    return derived(
        kernel,
        case,
        static={
            "heads": s["heads"],
            "head_dim": s["head_dim"],
            "dtype": s.get("dtype", "bfloat16"),
        },
        dynamic={"seq": seq},
        layout=case.get("layout") or "BHND",
    )


def kwargs(case: dict, dtype: torch.dtype, device: torch.device, backend: str) -> dict[str, Any]:
    del backend
    static, dynamic = case["static"], case["dynamic"]
    heads = int(static["heads"])
    head_dim = int(static["head_dim"])
    seq = int(dynamic["seq"])
    layout = case.get("layout") or "BHND"
    if layout == "BHND":
        shape = (1, heads, seq, head_dim)
        freqs_shape = (1, 1, seq, head_dim // 2, 2, 2)
    elif layout == "BNHD":
        shape = (1, seq, heads, head_dim)
        freqs_shape = (1, seq, 1, head_dim // 2, 2, 2)
    else:
        raise ValueError(f"unsupported rope layout {layout!r}")

    q = torch.randn(shape, dtype=dtype, device=device)
    k = torch.randn(shape, dtype=dtype, device=device)
    freqs = torch.randn(freqs_shape, dtype=dtype, device=device)
    if case["op"].startswith("rms_rope"):
        return {
            "q": q,
            "k": k,
            "freqs_cis": freqs,
            "q_scale": torch.randn(head_dim, dtype=torch.float32, device=device),
            "k_scale": torch.randn(head_dim, dtype=torch.float32, device=device),
            "epsilon": 1e-6,
        }
    return {"xq": q, "xk": k, "freqs_cis": freqs}


register(
    OpSpec(
        ops=("rms_rope", "rms_rope_split_half"),
        sources=("sdpa",),
        derive=derive,
        estimate=estimate,
        kwargs=kwargs,
        primary_param="q",
    )
)
register(
    OpSpec(
        ops=("apply_rope", "apply_rope_split_half"),
        sources=("sdpa",),
        derive=derive,
        estimate=estimate,
        kwargs=kwargs,
        primary_param="xq",
    )
)
