"""Explicit eager boundaries for inference paths with Python control or raw pointers."""

import torch


@torch.compiler.disable
def eager_call(function, *args, **kwargs):
    # Compiling the public function directly can unwrap its disable decorator.
    return function(*args, **kwargs)
