from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from comfy_kitchen.registry import registry

ELEM = {"float32": 4, "float64": 8, "float16": 2, "bfloat16": 2, "int8": 1}

KwargsFn = Callable[[dict, torch.dtype, torch.device, str], dict[str, Any]]
DeriveFn = Callable[[dict, str], dict | list[dict] | None]
EstimateFn = Callable[[dict], tuple[float, float]]


@dataclass(frozen=True)
class OpSpec:
    ops: tuple[str, ...]
    sources: tuple[str, ...]
    derive: DeriveFn
    estimate: EstimateFn
    kwargs: KwargsFn
    primary_param: str = "x"
    fixed_dtypes: tuple[torch.dtype, ...] | None = None


SPECS: dict[str, OpSpec] = {}
CONSUMERS: list[tuple[tuple[str, ...], tuple[str, ...], DeriveFn]] = []
ESTIMATORS: dict[str, EstimateFn] = {}


def register(spec: OpSpec) -> OpSpec:
    CONSUMERS.append((spec.ops, spec.sources, spec.derive))
    for name in spec.ops:
        if name in SPECS:
            raise ValueError(f"duplicate OpSpec for {name!r}")
        SPECS[name] = spec
        ESTIMATORS[name] = spec.estimate
    return spec


def primary_param(op_name: str) -> str | None:
    spec = SPECS.get(op_name)
    return None if spec is None else spec.primary_param


def derived(kernel: str, src: dict, *, static: dict, dynamic: dict, layout=None) -> dict:
    return {
        "op": kernel,
        "static": static,
        "dynamic": dynamic,
        "layout": layout if layout is not None else src.get("layout"),
        "calls": sum(src.get("call_count", {}).values()),
        "from": src.get("layer"),
        "source_op": src.get("op"),
    }


def elem(dtype: Any) -> int:
    return ELEM.get(str(dtype).rsplit(".", 1)[-1].lower(), 2)


def ceil_to(v: int, m: int) -> int:
    return (v + m - 1) // m * m


def backend_supports(backend: str, op_name: str) -> bool:
    info = registry.list_backends().get(backend, {})
    if (
        not bool(info.get("available"))
        or bool(info.get("disabled"))
        or op_name not in info.get("capabilities", ())
    ):
        return False
    constraints = registry.get_constraints(backend, op_name)
    required = constraints.min_compute_capability if constraints else None
    if required is not None and torch.cuda.is_available():
        return torch.cuda.get_device_capability() >= required
    return True


def prequantize(op_name: str, kwargs: dict[str, Any], *, backend: str) -> Any:
    prep_backend = backend
    if (
        not backend_supports(prep_backend, op_name)
        or not registry.validate_backend_for_call(prep_backend, op_name, kwargs).success
    ):
        prep_backend = "eager"
    implementation = registry.get_implementation(
        op_name,
        backend=prep_backend,
        kwargs=kwargs,
    )
    return implementation(**kwargs)
