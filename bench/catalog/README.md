# DiT shape catalog

The benchmark inputs in `shape_runs/` now come from 18 completed **weightless model-forward shape executions**, rather than a shared grid of guessed M values. They cover Qwen Image 2.1, MiniMax H3, Krea2, and LTX 2.5. The separate `models.json` is a source-derived reference of 37 representative projection groups; it is not the execution trace.

## Evidence boundary

No checkpoint weights were loaded. Models execute with meta tensors, and attention arithmetic is replaced by a shape-only adapter. Qwen's modulation-mask device is adapted for meta execution; Krea's compile decorators are unwrapped. Linear input/output shapes are observed while the configured forward and cache paths execute. These are neither checkpoint-backed inference captures nor GPU performance results. Uniform BF16 meta execution does not verify checkpoint mixed-precision dtypes.

The **input layouts are proposed scenarios**. Resulting M/N/K, module participation and call counts are observed from execution, but this does not establish production token distributions. Qwen2.1 RGB-to-token mapping is additionally checked through the actual Diffusers pipeline preparation/packing functions: 1024² gives4096 rows and2048² gives16384 rows (VAE factor16, no additional patching). These checks do not execute a VAE encoder. Video duration/FPS and prompt tokenization remain unverified input choices.

| Family | Explicit test inputs | Main sequence rows observed |
|---|---|---|
| Qwen 2.1 | Target latent grid 64×64 or 128×128; 0/1/3 same-grid references; 512 text tokens; batch 1; extract and cached phases | Extract: 4608/8704/16896 or 16896/33280/66048. Cached main blocks: 4096 or 16384 |
| Krea2 | 4096 or 16384 image rows, 151 text rows (length from the existing captured recipe), batch 1 | Joint padded rows: 4352 or 16640 |
| H3 | 17408 or 139264 video patch rows, 1024 audio and 512 text rows; batch 1 | Packed joint rows: 18944 or 140800 |
| LTX 2.5 | Latent video grids 17×32×32 or 17×64×128; 1024 audio and 512 text rows; batch 1 | Video: 17408 or 139264; audio: 1024 |

For example, Qwen **66048** specifically means four 128×128 grids (one target plus three references) and 512 text tokens: `4*16384+512`. It is a specific 2048² target-plus-three-equal-reference test, consistent with the pipeline geometry; the model officially recommends native2048² output. It is not a general editing shape or an observed deployment request. The actual forward also records separate input/text/conditioning shapes. Tiny M values may occur in those auxiliary operations; they are not used as main image/video scenarios.

Each recipe aggregates identical M/N/K/dtype/bias shapes and preserves contributing module paths, original input shape/stride observations and the observed projection count from **one forward**. `shape-executions.json` records context, versions, source hashes, adapters and full-capture hashes. Model names carry a `Weightless` suffix. There are no symbolic M bindings, so replay cannot silently substitute generic token counts. Existing captured recipes and workloads are unchanged.

## Replay and rerun

```bash
python bench/generate_configs.py --trace bench/catalog/shape_runs --workload replay
python bench/run_bench.py --recipe-dir bench/catalog/shape_runs --dry-run --backends eager --ops int8_linear
python -m pytest --confcutdir bench bench/tests -q
```

To reproduce shapes without downloading weights (install torch, diffusers, transformers, accelerate and einops):

```bash
python bench/catalog/capture_weightless.py qwen21 h3 ltx25 --out /tmp/dit-shapes-fresh
# Krea additionally needs its official mmdit.py source, not checkpoint weights:
python bench/catalog/capture_weightless.py krea2 --krea-source /path/to/krea-2/mmdit.py --out /tmp/krea-shapes-fresh
```

New captures use a fresh output directory. Model configs are included; Krea implementation code is supplied externally. LTX's config comes from the official converter's explicit 2.5 branch because the publisher checkpoint header required authentication. Backend support is not guaranteed for every observed shape. The harness omits ConvRot256 when K is not divisible by 256.

## Tracking checkpoint-backed runs

The ComfyUI **CK Op Trace** node now records each invocation of a reused linear and executes the model exactly once per requested forward. Set `one_shot=false` to capture multiple denoising steps. Use a distinct profile for each workload and fill `context_json` with model revision, resolution or latent grid, frames, batch, references and cache policy. The node also supports normal Diffusers use through `track_model_shapes`:

```python
from bench.tracer_node import track_model_shapes
with track_model_shapes(pipe.transformer, path="actual-shapes.json", profile="request-1",
                        metadata={"checkpoint": "...", "batch": 1, "references": 1}):
    pipe(...)  # your actual pipeline request
```

No tensors or tensor values are saved. Linear shapes, strides, dtypes, module paths and invocation counts are recorded. Functional/fused projections without hookable Linear-like modules are outside this observer's coverage. Custom attention implementations may bypass SDPA tracing. Model-forward exceptions propagate normally; save failures do not rerun the model.

## Evidence and capture fidelity

Keep the distinction between proposed inputs, executed weightless shapes and checkpoint-backed capture explicit. Qwen cached/extract phases and auxiliary projections must remain distinct. The trace-fidelity fixes are included because the prior observer kept only the last input of a reused module and the Comfy wrapper could execute twice or recurse.

The catalog extends the benchmark recipes without changing existing captured workloads. Contribution requirements are documented in `CONTRIBUTING.md`.

## Focus for kernel work

`focused_shapes/` is generated by `python bench/catalog/focus_shapes.py`. It retains repeated transformer-block shape groups whose call-weighted FLOP share is at least0.5% of the full recorded forward. Final output heads, input-only projections and small auxiliary groups remain in `shape_runs/` for audit but are omitted from this focused set.

The current focused catalog has63 groups across18 scenarios, retaining98.7–99.98% of recorded Linear FLOPs. **This is a FLOP-based prioritization, not measured end-to-end time.** A bandwidth-bound exception can be restored after GPU profiling. `meta.selection` lists every omitted group, module paths, call counts and estimated FLOP share.

```bash
python bench/run_bench.py --recipe-dir bench/catalog/focused_shapes --dry-run --backends eager --ops int8_linear
```

The optional geometry tests verify Qwen2.1's actual pipeline `prepare_latents` and `_pack_latents` functions with meta tensors. Qwen's official README recommends2048×2048 native output (about4.19MP): https://github.com/QwenLM/Qwen-Image-2.1#supported-aspect-ratios . The precise training-resolution distribution is not claimed.
