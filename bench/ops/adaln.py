from __future__ import annotations

from typing import Any

import torch

from bench.ops._base import OpSpec, derived, elem, register


def estimate(case: dict) -> tuple[float, float]:
    s, d = case["static"], case["dynamic"]
    e, m, h = elem(s.get("dtype")), int(d["M"]), int(s["hidden"])
    nbytes = e * (2 * m * h + 2 * h) + (4 * h if case["op"] == "ada_rmsnorm" else 0)
    return 0.0, float(nbytes)


def derive(case: dict, kernel: str) -> dict | None:
    s, d = case["static"], case["dynamic"]
    m, k = int(d.get("M", 0)), int(s.get("K", 0))
    if m <= 1 or k < 64:
        return None
    static = {"hidden": k, "dtype": s.get("dtype", "bfloat16")}
    if kernel == "ada_rmsnorm":
        static["rms"] = True
    return derived(kernel, case, static=static, dynamic={"M": m})


def kwargs(case: dict, dtype: torch.dtype, device: torch.device, backend: str) -> dict[str, Any]:
    del backend
    m = int(case["dynamic"]["M"])
    hidden = int(case["static"]["hidden"])
    x = torch.randn((1, m, hidden), dtype=dtype, device=device)
    scale = torch.randn((1, 1, hidden), dtype=dtype, device=device)
    shift = torch.randn((1, 1, hidden), dtype=dtype, device=device)
    if case["op"] == "adaln":
        return {"x": x, "scale": scale, "shift": shift, "eps": 1e-6}
    return {
        "x": x,
        "norm_scale": torch.randn(hidden, dtype=torch.float32, device=device),
        "scale": scale,
        "shift": shift,
        "eps": 1e-6,
    }


register(
    OpSpec(
        ops=("ada_rmsnorm", "adaln"),
        sources=("gemm",),
        derive=derive,
        estimate=estimate,
        kwargs=kwargs,
        primary_param="x",
    )
)
