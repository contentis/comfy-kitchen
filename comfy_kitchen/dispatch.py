# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Implementation-driven dispatch for migrated operation schemas."""

import logging
import threading
from collections.abc import Callable
from functools import cache
from typing import Any

import torch

from .exceptions import NoCapableBackendError
from .registry import UnsupportedCall, registry

Runner = Callable[..., Any]
CachedImplementation = tuple[str, Runner]
_CACHE_MISS = object()
logger = logging.getLogger(__name__)


def _argument_key(value):
    if isinstance(value, torch.Tensor):
        return (value.shape, value.stride(), value.dtype, value.device)
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(_argument_key(item) for item in value))
    if isinstance(value, dict):
        return tuple((key, _argument_key(item)) for key, item in sorted(value.items()))
    return type(value), value


def _tensors(values):
    for value in values:
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, (tuple, list)):
            yield from _tensors(value)
        elif isinstance(value, dict):
            yield from _tensors(value.values())


def _check_devices(args, kwargs, allowed):
    if not allowed:
        return
    tensors = list(_tensors((*args, *kwargs.values())))
    unsupported = {tensor.device.type for tensor in tensors} - allowed
    if unsupported:
        expected = ", ".join(sorted(allowed))
        observed = ", ".join(sorted(unsupported))
        raise UnsupportedCall(f"requires tensor device type {expected}; got {observed}")
    devices = {tensor.device for tensor in tensors}
    if len(devices) > 1:
        observed = ", ".join(sorted(map(str, devices)))
        raise UnsupportedCall(f"operands span devices: {observed}")


class Operation:
    """A declarative operation with a bounded copy-on-write implementation cache."""

    __slots__ = ("_cache", "_cache_limit", "_lock", "name")

    def __init__(self, name: str, *, cache_limit: int = 256):
        self.name = name
        self._cache: dict[tuple, CachedImplementation] = {}
        self._cache_limit = cache_limit
        self._lock = threading.Lock()

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        with self._lock:
            self._cache = {}

    def _key(self, args, kwargs):
        return (
            registry.dispatch_token(),
            *(_argument_key(value) for value in args),
            *((name, _argument_key(value)) for name, value in kwargs.items()),
        )

    def _remember(self, key, candidate):
        with self._lock:
            existing = self._cache.get(key, _CACHE_MISS)
            if existing is not _CACHE_MISS:
                return existing
            updated = dict(self._cache)
            if len(updated) >= self._cache_limit:
                del updated[next(iter(updated))]
            updated[key] = candidate
            self._cache = updated
            return candidate

    def _forget(self, key, candidate):
        with self._lock:
            if self._cache.get(key, _CACHE_MISS) != candidate:
                return
            updated = dict(self._cache)
            del updated[key]
            self._cache = updated

    def __call__(self, *args, **kwargs):
        key = self._key(args, kwargs)
        cached = self._cache.get(key, _CACHE_MISS)
        if cached is not _CACHE_MISS:
            backend, runner = cached
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("dispatch %s -> %s (cache hit)", self.name, backend)
            try:
                return runner(*args, **kwargs)
            except UnsupportedCall as error:
                self._forget(key, cached)
                logger.debug("dispatch %s rejected cached %s: %s", self.name, backend, error)
                failures = {backend: str(error)}
        else:
            failures = {}
        for backend, runner, device_types in registry.iter_implementations(self.name):
            if backend in failures:
                continue
            try:
                _check_devices(args, kwargs, device_types)
                result = runner(*args, **kwargs)
            except UnsupportedCall as error:
                failures[backend] = str(error)
                logger.debug("dispatch %s rejected %s: %s", self.name, backend, error)
                continue

            self._remember(key, (backend, runner))
            logger.debug("dispatch %s -> %s (cache miss)", self.name, backend)
            return result

        raise NoCapableBackendError(self.name, failures)


@cache
def operation(name: str) -> Operation:
    """Return the generic cached operation for a public torch op."""
    return Operation(name)


def dispatch(name: str, *args, **kwargs):
    return operation(name)(*args, **kwargs)
