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

import math
from pathlib import Path

import torch
import torch.nn as nn

from .gemm import (
    DistributedTensor,
    distributed_linear,
    resolve_devices,
)

__all__ = [
    "DistributedLinear",
    "convert_to_distributed",
    "load_distributed_checkpoint",
]


class DistributedLinear(nn.Module):
    """Drop-in replacement for nn.Linear with multi-GPU execution.

    Weights are sharded across GPUs along the output feature dimension.
    Forward pass returns a DistributedTensor that can be chained with
    other distributed operations without gathering.
    """

    weight: DistributedTensor
    bias: DistributedTensor | None

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            devices: list[int] | None = None,
            dtype: torch.dtype | None = None,
            _skip_init: bool = False,
    ) -> None:
        """Create a distributed linear layer.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If True, adds a learnable bias.
            devices: List of device IDs. Uses defaults if None.
            dtype: Data type for parameters.
            _skip_init: Internal flag to skip initialization (for from_linear).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        resolved_devices = resolve_devices(devices)
        self._devices = [d.index for d in resolved_devices]

        if _skip_init:
            # Will be set by from_linear
            self.weight = None  # type: ignore
            self.bias = None
            return

        # Weight shape: [out_features, in_features] in nn.Linear convention
        self.weight = DistributedTensor.empty(
            shape=(out_features, in_features),
            dim=0,
            devices=self._devices,
            dtype=dtype,
        )

        if bias:
            self.bias = DistributedTensor.empty(
                shape=(out_features,),
                dim=0,
                devices=self._devices,
                dtype=dtype,
            )
        else:
            self.bias = None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters similar to nn.Linear."""
        bound = 1 / math.sqrt(self.in_features)
        for shard in self.weight.shards:
            with torch.cuda.device(shard.device):
                shard.uniform_(-bound, bound)

        if self.bias is not None:
            for shard in self.bias.shards:
                with torch.cuda.device(shard.device):
                    shard.uniform_(-bound, bound)

    @classmethod
    def from_linear(
            cls,
            linear: nn.Linear,
            devices: list[int] | None = None,
    ) -> DistributedLinear:
        """Convert an existing nn.Linear to distributed."""
        has_bias = linear.bias is not None

        # Create without initialization
        module = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=has_bias,
            devices=devices,
            dtype=linear.weight.dtype,
            _skip_init=True,
        )

        # Directly set weights from source
        module.weight = DistributedTensor.from_tensor(
            linear.weight.data,
            dim=0,
            devices=devices,
        )

        if has_bias:
            module.bias = DistributedTensor.from_tensor(
                linear.bias.data,
                dim=0,
                devices=devices,
            )

        return module

    def forward(self, x: torch.Tensor | DistributedTensor) -> DistributedTensor:
        """Forward pass using distributed_linear."""
        return distributed_linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, devices={self._devices}"
        )


def convert_to_distributed(
        module: nn.Module,
        devices: list[int] | None = None,
        inplace: bool = False,
) -> nn.Module:
    """Replace all nn.Linear layers with DistributedLinear.

    Works with meta-device models for memory-efficient loading.
    """
    if not inplace:
        import copy
        module = copy.deepcopy(module)

    def _convert_children(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                is_meta = child.weight.device.type == "meta"

                if is_meta:
                    distributed = DistributedLinear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        bias=child.bias is not None,
                        devices=devices,
                        dtype=child.weight.dtype,
                    )
                else:
                    distributed = DistributedLinear.from_linear(child, devices=devices)

                setattr(parent, name, distributed)
            else:
                _convert_children(child)

    _convert_children(module)
    return module


def load_distributed_checkpoint(
        module: nn.Module,
        checkpoint_path: str | Path,
        devices: list[int] | None = None,
) -> None:
    """Load checkpoint directly into distributed shards.

    For use with meta-device initialized models.
    """
    checkpoint_path = Path(checkpoint_path)
    resolved_devices = resolve_devices(devices)
    device_ids = [d.index for d in resolved_devices]

    # Load state dict
    if checkpoint_path.is_file():
        if checkpoint_path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    elif checkpoint_path.is_dir():
        state_dict = {}
        for shard_file in sorted(checkpoint_path.glob("*.safetensors")):
            from safetensors.torch import load_file
            state_dict.update(load_file(shard_file))
        if not state_dict:
            for shard_file in sorted(checkpoint_path.glob("*.bin")):
                shard_dict = torch.load(shard_file, map_location="cpu", weights_only=True)
                state_dict.update(shard_dict)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not state_dict:
        raise ValueError(f"No weights found in {checkpoint_path}")

    def _load_into_distributed(parent: nn.Module, prefix: str = "") -> None:
        for name, child in parent.named_children():
            child_prefix = f"{prefix}{name}." if prefix else f"{name}."

            if isinstance(child, DistributedLinear):
                weight_key = f"{child_prefix}weight"
                if weight_key in state_dict:
                    child.weight = DistributedTensor.from_tensor(
                        state_dict[weight_key],
                        dim=0,
                        devices=device_ids,
                    )

                bias_key = f"{child_prefix}bias"
                if child.bias is not None and bias_key in state_dict:
                    child.bias = DistributedTensor.from_tensor(
                        state_dict[bias_key],
                        dim=0,
                        devices=device_ids,
                    )
            else:
                _load_into_distributed(child, child_prefix)

    _load_into_distributed(module)