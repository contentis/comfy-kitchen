"""Optional weightless geometry checks against installed Diffusers pipeline code."""

from types import SimpleNamespace

import pytest
import torch

pipeline_module = pytest.importorskip("diffusers.pipelines.qwenimage21.pipeline_qwenimage21")
Pipeline = pipeline_module.QwenImage21Pipeline


@pytest.mark.parametrize("pixels,rows", [(1024, 4096), (2048, 16384)])
def test_qwen21_pipeline_preparation_preserves_published_latent_geometry(pixels, rows):
    # Qwen2.1 pipeline __init__ and published VAE config specify factor 16.
    # Calling the actual preparation/packing functions prevents an extra patch
    # factor from being accidentally introduced by our catalog.
    pipeline = SimpleNamespace(vae_scale_factor=16, _pack_latents=Pipeline._pack_latents)
    packed, references = Pipeline.prepare_latents(
        pipeline,
        None,
        1,
        64,
        pixels,
        pixels,
        torch.bfloat16,
        torch.device("meta"),
        None,
    )
    assert packed.is_meta and packed.shape == (1, rows, 64)
    assert references is None
