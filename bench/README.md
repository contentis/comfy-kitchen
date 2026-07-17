# Bench

Microbenchmarks for Comfy Kitchen kernels

```text
ComfyUI (CK Op Trace)  →  recipe JSON  →  workloads + OpSpecs  →  run_bench  →  report
```

## 1. Capture a recipe (custom node)

Install the tracer as a ComfyUI custom node (symlink or copy):

```bash
# symlink
ln -s /path/to/ckitchen/bench/tracer_node \
      /path/to/ComfyUI/custom_nodes/ck_tracer

# or copy
cp -r /path/to/ckitchen/bench/tracer_node \
      /path/to/ComfyUI/custom_nodes/ck_tracer
```

Restart ComfyUI. Wire **CK Op Trace** on the MODEL before sampling.


Output lands at `{output_dir}/{ModelClass}/{filename}.json`. Copy it into `bench/recipes/`:

```bash
cp ~/ComfyUI/output/ck_triage/Flux/bundle.json bench/recipes/my-model.json
```

The tracer records **gemm** (linears) and **sdpa** shapes. Everything else is derived later by OpSpecs.

## 2. Edit the recipe (symbolic shapes)

Captured dynamics are concrete numbers under `dynamic.*.values`. To retarget resolutions / token counts, add a `binding` — a small integer expression over workload symbols:

```json
"dynamic": {
  "M": {
    "values": { "default": 4096 },
    "binding": "img_tokens"
  },
  "q_tokens": {
    "values": { "default": 4608 },
    "binding": "txt_tokens+img_tokens"
  }
}
```

Supported ops in bindings: `+`, `-`, `*`, `//`.

- With a workload, the harness evaluates `binding` using that workload’s symbols.
- Without a workload (`replay`), it uses the captured `values`.

See existing files under `bench/recipes/` for patterns (bindings are already filled in for Flux / Z-Image / etc.).

## 3. Add workloads

Edit `bench/workloads.json`:

```json
{
  "dit_t2i_1024": {
    "models": ["Flux", "SingleStreamDiT"],
    "txt_tokens": 512,
    "img_tokens": 4096
  }
}
```

- `models` — recipe `model` field must match one of these names (diffusion class from the trace).
- Other keys become symbols available in `binding` expressions.

`run_bench` automatically expands each recipe against every matching workload. Preview without timing:

```bash
python bench/generate_configs.py --trace bench/recipes/my-model.json --workload dit_t2i_1024
```

## 4. Add ops / specs

Each family lives in `bench/ops/<name>.py` and registers an `OpSpec`:

```python
from bench.ops._base import OpSpec, derived, register

def derive(case: dict, kernel: str) -> dict | list[dict] | None:
    # case comes from a traced primary (gemm or sdpa)
    ...
    return derived(kernel, case, static={...}, dynamic={...})

def estimate(case: dict) -> tuple[float, float]:
    # (flops, bytes) — used for coverage ranking
    ...

def kwargs(case, dtype, device, backend) -> dict:
    # tensors / args passed to the CK registry op
    ...

register(OpSpec(
    ops=("my_kernel",),
    sources=("gemm",),      # or ("sdpa",)
    derive=derive,
    estimate=estimate,
    kwargs=kwargs,
))
```

Then import the module from `bench/ops/__init__.py` so it registers on load.

- `sources` — which traced ops feed this kernel (`gemm` → linears, `sdpa` → attention shapes).
- `derive` — map a primary event into one or more bench cases (or `None` to skip).
- `estimate` — cheap cost model so `--coverage` keeps the heavy hitters.
- `kwargs` — build the actual call into `comfy_kitchen`.

## 5. Run + report

Dry-run the selected matrix:

```bash
python bench/run_bench.py --dry-run --backends cuda --ops default
```

Full run with HTML report (peaks are device-specific; example numbers only):

```bash
python bench/run_bench.py --backends cuda --ops default --report \
  --peak-fp16-tflops 100 --peak-bw-gbs 1500
```

Artifacts land under `.bench/<timestamp>_ck-<ver>_<gpu>/`:

- `results.csv` — timings
- `metadata.json` — env / peaks
- `report.html` — interactive charts (if `--report`)

Or build a report later:

```bash
python bench/interactive_report.py .bench/<run_dir> \
  --peak-fp16-tflops 100 --peak-bw-gbs 1500
```
