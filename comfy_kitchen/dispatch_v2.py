# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Implementation-driven dispatch for migrated operation schemas."""

import inspect
import threading
from collections.abc import Callable
from functools import cache
from typing import Any

import torch

from .exceptions import NoCapableBackendError
from .registry import UnsupportedCall, registry

Runner = Callable[..., Any]
Plan = tuple[str, Runner]
_CACHE_MISS = object()


def total(runner: Runner) -> Callable[..., Runner]:
    """Adapt an implementation that accepts every call into a lowering."""

    def lower(*_args, **_kwargs) -> Runner:
        return runner

    return lower


def _argument_key(value):
    if isinstance(value, torch.Tensor):
        return (value.shape, value.stride(), value.dtype, value.device)
    if isinstance(value, (tuple, list)):
        return tuple(_argument_key(item) for item in value)
    if isinstance(value, dict):
        return tuple((key, _argument_key(item)) for key, item in sorted(value.items()))
    return value


def tensor_rule(dtypes, shape=None):
    return frozenset(dtypes), shape


def value_rule(*types):
    return frozenset(types), None


def ndim(size):
    return lambda shape: len(shape) == size


def min_ndim(size):
    return lambda shape: len(shape) >= size


def aligned_ndim(size, axis, factor):
    return lambda shape: len(shape) == size and shape[axis] % factor == 0


def aligned(axis, factor):
    return lambda shape: shape[axis] % factor == 0


def lowering(runner, *, devices, min_cc=None, check=None, **params):
    signature = inspect.signature(runner)

    def lower(*args, **kwargs):
        call = signature.bind(*args, **kwargs)
        call.apply_defaults()
        tensors = [value for value in call.arguments.values() if isinstance(value, torch.Tensor)]
        if any(tensor.device.type not in devices for tensor in tensors):
            raise UnsupportedCall("unsupported device")
        if tensors and any(tensor.device != tensors[0].device for tensor in tensors[1:]):
            raise UnsupportedCall("operands must share one device")
        for name, (allowed, shape) in params.items():
            value = call.arguments.get(name)
            if value is None:
                continue
            observed = (
                value.dtype
                if isinstance(value, torch.Tensor)
                else value
                if isinstance(value, torch.dtype)
                else type(value)
            )
            if allowed is not None and observed not in allowed:
                raise UnsupportedCall(f"{name}: unsupported type {observed}")
            if shape is not None and not shape(value.shape):
                raise UnsupportedCall(f"{name}: unsupported shape {tuple(value.shape)}")
        if min_cc is not None and tensors and tensors[0].device.type == "cuda":
            props = torch.cuda.get_device_properties(tensors[0].device)
            if (props.major, props.minor) < min_cc:
                raise UnsupportedCall(f"requires compute capability {min_cc}")
        if check is not None:
            reason = check(call.arguments)
            if reason:
                raise UnsupportedCall(reason)
        return runner

    return lower


class Operation:
    """A declarative operation with a bounded copy-on-write plan cache."""

    __slots__ = ("_cache", "_cache_limit", "_lock", "name")

    def __init__(self, name: str, *, cache_limit: int = 256):
        self.name = name
        self._cache: dict[tuple, Plan] = {}
        self._cache_limit = cache_limit
        self._lock = threading.Lock()

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        with self._lock:
            self._cache = {}

    def select(self, *args, **kwargs) -> Plan:
        key = (
            registry.dispatch_token(),
            *(_argument_key(value) for value in args),
            *((name, _argument_key(value)) for name, value in kwargs.items()),
        )
        plan = self._cache.get(key, _CACHE_MISS)
        if plan is not _CACHE_MISS:
            return plan

        failures = {}
        for backend, lower in registry.iter_lowerings(self.name):
            try:
                candidate = (backend, lower(*args, **kwargs))
            except UnsupportedCall as error:
                failures[backend] = str(error)
                continue

            with self._lock:
                plan = self._cache.get(key, _CACHE_MISS)
                if plan is not _CACHE_MISS:
                    return plan

                updated = dict(self._cache)
                if len(updated) >= self._cache_limit:
                    del updated[next(iter(updated))]
                updated[key] = candidate
                self._cache = updated
            return candidate

        raise NoCapableBackendError(self.name, failures)

    def __call__(self, *args, **kwargs):
        _, runner = self.select(*args, **kwargs)
        return runner(*args, **kwargs)


@cache
def operation(name: str) -> Operation:
    """Return the generic cached operation for a public torch op."""
    return Operation(name)


def dispatch(name: str, *args, **kwargs):
    return operation(name)(*args, **kwargs)


rms_rope = operation("rms_rope")
rms_rope_ = operation("rms_rope_")
rms_rope1 = operation("rms_rope1")
rms_rope1_ = operation("rms_rope1_")
rms_rope_split_half = operation("rms_rope_split_half")
rms_rope_split_half_ = operation("rms_rope_split_half_")
rms_rope_split_half1 = operation("rms_rope_split_half1")
rms_rope_split_half1_ = operation("rms_rope_split_half1_")
