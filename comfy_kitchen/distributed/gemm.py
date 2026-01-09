# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from comfy_kitchen.tensor.base import QuantizedTensor

__all__ = [
    "DistributedTensor",
    "EvenShardingStrategy",
    "ShardingStrategy",
    "broadcast_to_devices",
    "distributed_linear",
    "get_default_devices",
    "multi_gpu_gemm",
    "resolve_devices",
    "set_default_devices",
]

# ==================== Lazy-cached imports ====================

_QuantizedTensor: type | None = None


def _get_quantized_tensor_type() -> type:
    """Lazy-load QuantizedTensor type to avoid circular imports."""
    global _QuantizedTensor
    if _QuantizedTensor is None:
        from comfy_kitchen.tensor.base import QuantizedTensor
        _QuantizedTensor = QuantizedTensor
    return _QuantizedTensor


# ==================== Device Configuration ====================

_default_devices: tuple[torch.device, ...] | None = None


def set_default_devices(devices: list[int] | None) -> None:
    """Set default devices for distributed operations.

    Args:
        devices: List of device IDs (e.g., [0, 1, 2, 3]).
                 None = auto-detect all available GPUs.
    """
    global _default_devices
    if devices is None:
        _default_devices = None
    else:
        _default_devices = tuple(torch.device(f"cuda:{d}") for d in devices)


def get_default_devices() -> tuple[torch.device, ...]:
    """Get configured default devices, or auto-detect if not set."""
    if _default_devices is not None:
        return _default_devices
    count = torch.cuda.device_count()
    if count == 0:
        raise RuntimeError("No CUDA devices available")
    return tuple(torch.device(f"cuda:{i}") for i in range(count))


def resolve_devices(devices: list[int] | None = None) -> tuple[torch.device, ...]:
    """Resolve devices from explicit argument or fall back to defaults.

    Warns if GPU SKUs differ across the resolved devices.
    """
    if devices is not None:
        resolved = tuple(torch.device(f"cuda:{d}") for d in devices)
    else:
        resolved = get_default_devices()

    if len(resolved) > 1:
        names = [torch.cuda.get_device_name(d) for d in resolved]
        if len(set(names)) > 1:
            device_ids = devices if devices is not None else list(range(len(resolved)))
            warnings.warn(
                f"GPU SKUs differ across devices: {dict(zip(device_ids, names, strict=False))}. "
                "Performance may be unbalanced.",
                stacklevel=3,
            )

    return resolved


# ==================== Sharding Strategy ====================


class ShardingStrategy(ABC):
    """Base class for tensor sharding strategies."""

    @abstractmethod
    def shard(
            self,
            tensor: torch.Tensor,
            dim: int,
            devices: tuple[torch.device, ...],
    ) -> list[torch.Tensor]:
        """Split tensor into shards and place on devices."""
        raise NotImplementedError

    @abstractmethod
    def validate(
            self,
            shape: tuple[int, ...],
            dim: int,
            devices: tuple[torch.device, ...],
    ) -> None:
        """Validate that the tensor can be sharded. Raises ValueError if not."""
        raise NotImplementedError


class EvenShardingStrategy(ShardingStrategy):
    """Require tensor dimension to be evenly divisible by number of devices."""

    def validate(
            self,
            shape: tuple[int, ...],
            dim: int,
            devices: tuple[torch.device, ...],
    ) -> None:
        size = shape[dim]
        num_devices = len(devices)
        if size % num_devices != 0:
            raise ValueError(
                f"Dimension {dim} (size={size}) must be evenly divisible by "
                f"number of devices ({num_devices})."
            )

    def shard(
            self,
            tensor: torch.Tensor,
            dim: int,
            devices: tuple[torch.device, ...],
    ) -> list[torch.Tensor]:
        chunks = tensor.chunk(len(devices), dim=dim)
        return [chunk.to(device) for chunk, device in zip(chunks, devices, strict=False)]


_default_strategy = EvenShardingStrategy()


# ==================== DistributedTensor ====================


class DistributedTensor:
    """Tensor distributed across GPUs along a single dimension. Gathered lazily."""

    __slots__ = ("_dim", "_is_quantized", "_shards")

    def __init__(
            self,
            shards: list[torch.Tensor | QuantizedTensor],
            dim: int,
    ) -> None:
        if not shards:
            raise ValueError("shards cannot be empty")

        # Validate shard shape compatibility
        base_shape = list(shards[0].shape)
        for i, shard in enumerate(shards[1:], start=1):
            shard_shape = list(shard.shape)
            for d in range(len(base_shape)):
                if d != dim and shard_shape[d] != base_shape[d]:
                    raise ValueError(
                        f"Shard {i} shape {tuple(shard_shape)} incompatible with "
                        f"shard 0 shape {tuple(base_shape)} at dimension {d}"
                    )

        self._shards = shards
        self._dim = dim
        self._is_quantized = isinstance(shards[0], _get_quantized_tensor_type())

    @property
    def shards(self) -> list[torch.Tensor | QuantizedTensor]:
        return self._shards

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def shape(self) -> tuple[int, ...]:
        base_shape = list(self._shards[0].shape)
        base_shape[self._dim] = sum(s.shape[self._dim] for s in self._shards)
        return tuple(base_shape)

    @property
    def devices(self) -> tuple[torch.device, ...]:
        return tuple(s.device for s in self._shards)

    @property
    def dtype(self) -> torch.dtype:
        shard = self._shards[0]
        if hasattr(shard, "_params") and hasattr(shard._params, "orig_dtype"):
            return shard._params.orig_dtype
        return shard.dtype

    @property
    def is_quantized(self) -> bool:
        return self._is_quantized

    def __repr__(self) -> str:
        return (
            f"DistributedTensor(shape={self.shape}, dim={self._dim}, "
            f"devices={tuple(str(d) for d in self.devices)}, "
            f"dtype={self.dtype}, quantized={self._is_quantized})"
        )

    def gather(self, device: int | str | torch.device = 0) -> torch.Tensor:
        """Gather all shards to a single device via P2P copy."""
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)

        if self._is_quantized:
            dequantized = [s.dequantize().to(device) for s in self._shards]
            return torch.cat(dequantized, dim=self._dim)

        copied = [s.to(device) for s in self._shards]
        return torch.cat(copied, dim=self._dim)

    def map(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> DistributedTensor:
        """Apply a function to each shard independently."""
        new_shards = []
        for shard in self._shards:
            with torch.cuda.device(shard.device):
                new_shards.append(fn(shard))
        return DistributedTensor(new_shards, self._dim)

    def __matmul__(self, other: DistributedTensor) -> DistributedTensor:
        return multi_gpu_gemm(self, other)

    def __rmatmul__(self, other: torch.Tensor) -> DistributedTensor:
        return multi_gpu_gemm(other, self)

    def __add__(self, other: DistributedTensor) -> DistributedTensor:
        if not isinstance(other, DistributedTensor):
            raise TypeError(f"Cannot add DistributedTensor with {type(other)}")
        if self.devices != other.devices:
            raise ValueError("Devices must match for addition")
        new_shards = []
        for s1, s2 in zip(self._shards, other._shards, strict=False):
            with torch.cuda.device(s1.device):
                new_shards.append(s1 + s2)
        return DistributedTensor(new_shards, self._dim)

    @classmethod
    def empty(
            cls,
            shape: tuple[int, ...],
            dim: int = 1,
            devices: list[int] | None = None,
            dtype: torch.dtype | None = None,
            strategy: ShardingStrategy | None = None,
    ) -> DistributedTensor:
        """Create an uninitialized distributed tensor."""
        resolved_devices = resolve_devices(devices)
        dtype = dtype or torch.float32
        strat = strategy or _default_strategy

        strat.validate(shape, dim, resolved_devices)

        shard_size = shape[dim] // len(resolved_devices)
        shard_shape = list(shape)
        shard_shape[dim] = shard_size

        shards = []
        for device in resolved_devices:
            with torch.cuda.device(device):
                shards.append(torch.empty(shard_shape, dtype=dtype, device=device))
        return cls(shards, dim)

    @classmethod
    def from_tensor(
            cls,
            tensor: torch.Tensor,
            dim: int = 1,
            devices: list[int] | None = None,
            strategy: ShardingStrategy | None = None,
    ) -> DistributedTensor:
        """Shard a tensor across devices."""
        resolved_devices = resolve_devices(devices)
        strat = strategy or _default_strategy

        strat.validate(tuple(tensor.shape), dim, resolved_devices)
        shards = strat.shard(tensor, dim, resolved_devices)
        return cls(shards, dim)

    @classmethod
    def from_quantized(
            cls,
            qtensor: QuantizedTensor,
            dim: int = 1,
            devices: list[int] | None = None,
            strategy: ShardingStrategy | None = None,
    ) -> DistributedTensor:
        """Shard a QuantizedTensor across devices."""
        qt_cls = _get_quantized_tensor_type()

        resolved_devices = resolve_devices(devices)
        strat = strategy or _default_strategy

        orig_shape = tuple(qtensor.shape)
        strat.validate(orig_shape, dim, resolved_devices)

        num_devices = len(resolved_devices)
        shard_size = orig_shape[dim] // num_devices

        qdata_chunks = qtensor._qdata.chunk(num_devices, dim=dim)

        shards = []
        for qdata_chunk, device in zip(qdata_chunks, resolved_devices, strict=False):
            shard_params = qtensor._params.clone()
            shard_shape = list(orig_shape)
            shard_shape[dim] = shard_size
            shard_params.orig_shape = tuple(shard_shape)

            shard_qdata = qdata_chunk.to(device)
            shard_params = shard_params.to_device(device)

            shard = qt_cls(shard_qdata, qtensor._layout_cls, shard_params)
            shards.append(shard)

        return cls(shards, dim)


# ==================== Operations ====================


def broadcast_to_devices(
        tensor: torch.Tensor,
        devices: tuple[torch.device, ...],
) -> list[torch.Tensor]:
    """Copy a tensor to multiple devices via P2P."""
    return [
        tensor if tensor.device == device else tensor.to(device)
        for device in devices
    ]


def _get_input_replicas(
        x: torch.Tensor | DistributedTensor,
        devices: tuple[torch.device, ...],
) -> list[torch.Tensor]:
    """Get input tensor replicated across devices."""
    if isinstance(x, DistributedTensor):
        if x.devices != devices:
            raise ValueError(
                f"Input devices {x.devices} must match target devices {devices}"
            )
        return x.shards
    return broadcast_to_devices(x, devices)


def multi_gpu_gemm(
        x: torch.Tensor | DistributedTensor,
        weight: DistributedTensor,
        bias: DistributedTensor | None = None,
) -> DistributedTensor:
    """Distributed GEMM: Y = X @ W across multiple GPUs.

    Each GPU computes its portion: Y_i = X @ W_i
    """
    devices = weight.devices
    x_replicas = _get_input_replicas(x, devices)

    output_shards = []
    for i, (x_i, w_i) in enumerate(zip(x_replicas, weight.shards, strict=False)):
        with torch.cuda.device(devices[i]):
            y_i = x_i @ w_i
            if bias is not None:
                y_i = y_i + bias.shards[i]
            output_shards.append(y_i)

    return DistributedTensor(output_shards, dim=weight.dim)


def distributed_linear(
        x: torch.Tensor | DistributedTensor,
        weight: DistributedTensor,
        bias: DistributedTensor | None = None,
) -> DistributedTensor:
    """Distributed linear: Y = X @ W.T + b (nn.Linear compatible).

    Weight shards are [out_features/num_devices, in_features].
    """
    devices = weight.devices
    x_replicas = _get_input_replicas(x, devices)

    output_shards = []
    for i, (x_i, w_i) in enumerate(zip(x_replicas, weight.shards, strict=False)):
        with torch.cuda.device(devices[i]):
            y_i = x_i @ w_i.t()
            if bias is not None:
                y_i = y_i + bias.shards[i]
            output_shards.append(y_i)

    return DistributedTensor(output_shards, dim=1)