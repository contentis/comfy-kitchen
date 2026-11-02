"""Explicit dense hardware peaks; precision ratios vary by GPU and operation."""
from __future__ import annotations

import json
import math

PEAK_ARGUMENTS = {
    "fp16": "peak_fp16_tflops",
    "fp32": "peak_fp32_tflops",
    "fp8": "peak_fp8_tflops",
    "fp4": "peak_fp4_tflops",
    "int8": "peak_int8_tops",
    "int4": "peak_int4_tops",
}


def add_peak_arguments(parser):
    for precision, dest in PEAK_ARGUMENTS.items():
        parser.add_argument("--" + dest.replace("_", "-"), type=float, default=None,
                            help=f"explicit dense {precision.upper()} peak in trillion operations/s")
    parser.add_argument("--peak-bw-gbs", type=float, default=None,
                        help="peak device memory bandwidth in GB/s")


def compute_peaks(args):
    return {precision: getattr(args, dest, None) for precision, dest in PEAK_ARGUMENTS.items()}


def validate_peaks(peaks, bandwidth):
    for name, value in {**peaks, "bandwidth": bandwidth}.items():
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} peak must be finite and positive")
    if bandwidth is None:
        raise ValueError("--peak-bw-gbs is required for a report")


def compute_precision(row):
    static = row.get("static", {})
    if isinstance(static, str):
        static = json.loads(static or "{}")
    description = f"{row.get('op', '')} {static.get('format', '')}".lower()
    for precision, tokens in [("int4", ("int4",)), ("int8", ("int8",)),
                              ("fp4", ("fp4", "e2m1")),
                              ("fp8", ("fp8", "float8", "e4m3", "e5m2"))]:
        if any(token in description for token in tokens):
            return precision
    dtype = str(row.get("dtype", static.get("dtype", "")))
    return {"float16": "fp16", "bfloat16": "fp16", "float32": "fp32"}.get(dtype)
