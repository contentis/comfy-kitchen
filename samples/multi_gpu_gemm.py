# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Multi-GPU GEMM Sample

Demonstrates:
1. Basic DistributedTensor usage
2. Chained GEMMs with gathering between layers
3. DistributedLinear for new models
4. Converting existing models

Note: With output-feature parallelism, gathering is required between layers
since each GPU only has a portion of the output features.
"""

import torch
import torch.nn as nn
from torch.nn import functional

from comfy_kitchen.distributed import (
    DistributedLinear,
    DistributedTensor,
    convert_to_distributed,
    set_default_devices,
)


def get_devices() -> list[int]:
    """Get available GPU device IDs."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    n = torch.cuda.device_count()
    if n < 2:
        raise RuntimeError(f"Need 2+ GPUs, found {n}")
    return list(range(n))


# --- Example 1: Basic DistributedTensor ---

def example_basic():
    devices = get_devices()
    set_default_devices(devices)

    # Create and distribute a weight matrix
    weight = torch.randn(1024, 4096, device="cuda:0")
    dist_weight = DistributedTensor.from_tensor(weight, dim=1)

    # Distributed GEMM
    x = torch.randn(64, 1024, device="cuda:0")
    y = x @ dist_weight  # Returns DistributedTensor

    # Gather when needed
    output = y.gather(device=0)
    assert output.shape == (64, 4096)


# --- Example 2: Chained GEMMs ---

def example_chained():
    """Chain multiple GEMMs - gather required between layers for output-split."""
    devices = get_devices()

    w1 = torch.randn(512, 2048, device="cuda:0")
    w2 = torch.randn(2048, 512, device="cuda:0")

    dist_w1 = DistributedTensor.from_tensor(w1, dim=1, devices=devices)
    dist_w2 = DistributedTensor.from_tensor(w2, dim=1, devices=devices)

    x = torch.randn(64, 512, device="cuda:0")

    # Layer 1
    h = x @ dist_w1
    h = h.map(functional.relu)
    h = h.gather(device=0)  # Gather before next layer

    # Layer 2
    y = h @ dist_w2
    output = y.gather(device=0)

    assert output.shape == (64, 512)


# --- Example 3: New model with DistributedLinear ---

def example_new_model():
    devices = get_devices()

    class DistributedMLP(nn.Module):
        def __init__(self, d_in, d_hidden, d_out, devices):
            super().__init__()
            self.fc1 = DistributedLinear(d_in, d_hidden, devices=devices)
            self.fc2 = DistributedLinear(d_hidden, d_out, devices=devices)

        def forward(self, x):
            h = self.fc1(x)
            h = h.map(functional.relu)
            h = h.gather(device=0)  # Gather before fc2
            return self.fc2(h).gather(device=0)

    model = DistributedMLP(512, 2048, 512, devices)
    x = torch.randn(64, 512, device="cuda:0")
    y = model(x)
    assert y.shape == (64, 512)


# --- Example 4: Convert existing model ---

def example_convert():
    devices = get_devices()

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(512, 2048)
            self.fc2 = nn.Linear(2048, 512)

    model = MLP().cuda()
    model = convert_to_distributed(model, devices=devices)

    x = torch.randn(64, 512, device="cuda:0")
    h = model.fc1(x).map(functional.relu).gather(device=0)
    y = model.fc2(h).gather(device=0)
    assert y.shape == (64, 512)


if __name__ == "__main__":
    example_basic()
    example_chained()
    example_new_model()
    example_convert()
    print("All examples passed.")