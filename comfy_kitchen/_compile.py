"""Explicit eager boundaries for inference paths with Python control or raw pointers."""

import torch


@torch.compiler.disable
def eager_call(function, *args, **kwargs):
    # Keep the public function itself traceable: torch.compile(public_function)
    # may unwrap a disable decorator attached directly to that public function.
    return function(*args, **kwargs)
