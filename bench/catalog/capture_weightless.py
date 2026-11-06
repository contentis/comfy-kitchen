"""Execute unmodified DiT forward geometry with meta weights; no checkpoint or timings."""

import importlib.util
import json
import os
import sys
import traceback
import hashlib
from pathlib import Path
import argparse
import importlib

os.environ["TORCH_COMPILE_DISABLE"] = "1"
import torch
import diffusers

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("models", nargs="+", choices=["qwen21", "h3", "ltx25", "krea2"])
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--krea-source", type=Path)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.tracer_node.tracer import observe_events, _merge_to_disk  # noqa: E402

S = Path(__file__).with_name("model_configs")
OUT = args.out
OUT.mkdir(parents=True, exist_ok=False)
if "krea2" in args.models and (args.krea_source is None or not args.krea_source.is_file()):
    parser.error("Krea2 requires --krea-source /path/to/official/krea-2/mmdit.py")


def empty(*s):
    return torch.empty(*s, device="meta", dtype=torch.bfloat16)


def ids(n):
    return torch.arange(n, device="meta", dtype=torch.long)


# Shape adapters only: no attention numerics are evaluated or benchmarked.
def attention_shape(query, key=None, value=None, *args, **kwargs):
    return torch.empty_like(query)


torch.nn.functional.scaled_dot_product_attention = attention_shape
for module_name in ["transformer_qwenimage21", "transformer_minimax_h3", "transformer_ltx2"]:
    module = importlib.import_module("diffusers.models.transformers." + module_name)
    if hasattr(module, "dispatch_attention_fn"):
        module.dispatch_attention_fn = attention_shape
    if hasattr(module, "_select_modulation_rows"):
        original = module._select_modulation_rows

        def select_rows(value, mask, fn=original):
            return fn(value, mask.to(value.device) if mask is not None else None)

        module._select_modulation_rows = select_rows
reports = []


def run(name, model, kwargs, context):
    events = []
    try:
        with observe_events(model, profile=name, capture_sdpa=False) as events, torch.no_grad():
            model(**kwargs)
        status = "complete"
        error = None
    except Exception as e:
        status = "failed"
        error = repr(e)
        traceback.print_exc()
    doc = {
        "status": status,
        "profile": name,
        "evidence": "weightless_meta_forward_attention_shape_adapter",
        "context": context,
        "diffusers": diffusers.__version__,
        "torch": torch.__version__,
        "error": error,
        "events": events,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_source_sha256": hashlib.sha256(
            Path(sys.modules[type(model).__module__].__file__).read_bytes()
        ).hexdigest(),
        "adapters": [
            "attention output shape only; no QK/PV arithmetic",
            "Qwen modulation mask moved to meta device",
            "Krea torch.compile decorators unwrapped",
        ],
        "dtype_scope": "Uniform BF16 meta execution; checkpoint mixed-precision dtypes not verified",
    }
    (OUT / (name + ".json")).write_text(json.dumps(doc, indent=2) + "\n")
    if status == "complete":
        _merge_to_disk(
            OUT / (name + "-bundle.json"),
            events,
            profile=name,
            model=type(model).__name__,
            metadata=dict(evidence=doc["evidence"], **context),
        )
    reports.append(
        {k: v for k, v in doc.items() if k != "events"}
        | {
            "linear_calls": len(events),
            "unique_m": sorted({e["dynamic"]["M"]["values"][name] for e in events}),
        }
    )
    print(name, status, len(events), error, flush=True)


if "krea2" in args.models:
    spec = importlib.util.spec_from_file_location("krea_mmdit", args.krea_source)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    for cls in list(vars(mod).values()):
        if isinstance(cls, type) and issubclass(cls, torch.nn.Module):
            fn = getattr(cls, "forward", None)
            if hasattr(fn, "_torchdynamo_orig_callable"):
                cls.forward = fn._torchdynamo_orig_callable
    cfg = mod.SingleMMDiTConfig(
        features=6144,
        tdim=256,
        txtdim=2560,
        heads=48,
        kvheads=12,
        multiplier=4,
        layers=28,
        patch=2,
        channels=16,
        txtheads=20,
        txtkvheads=20,
        txtlayers=12,
    )
    with torch.device("meta"):
        model = mod.SingleStreamDiT(cfg).to(dtype=torch.bfloat16)
    for image in [4096, 16384]:
        text = 151
        n = image + text
        run(
            "krea2-image" + str(image),
            model,
            {
                "img": empty(1, image, 64),
                "context": empty(1, text, 12, 2560),
                "t": empty(1),
                "pos": empty(1, n, 3),
                "mask": torch.ones(1, n, device="meta", dtype=torch.bool),
            },
            {
                "image_rows": image,
                "text_rows": text,
                "source": "image rows from 1024/2048 geometry; text length from existing Krea capture",
                "weights_loaded": False,
            },
        )
if "h3" in args.models:
    cfg = json.loads((S / "h3.json").read_text())
    with torch.device("meta"):
        model = diffusers.MiniMaxH3Transformer3DModel.from_config(cfg).to(dtype=torch.bfloat16)
    for v in [17408, 139264]:
        a, t = 1024, 512
        n = v + a + t
        run(
            "h3-video" + str(v),
            model,
            {
                "hidden_states": empty(1, v, 96),
                "audio_hidden_states": empty(1, a, 32),
                "encoder_hidden_states": empty(1, t, 5120),
                "timestep": empty(1),
                "timestep_indices": torch.zeros(n, device="meta", dtype=torch.long),
                "token_tags": torch.cat(
                    [
                        torch.zeros(v, dtype=torch.long),
                        torch.full((a,), 2, dtype=torch.long),
                        torch.ones(t, dtype=torch.long),
                    ]
                ).to("meta"),
                "position_ids": torch.zeros(n, 3, device="meta"),
                "video_indices": ids(v),
                "audio_indices": ids(a) + v,
                "text_indices": ids(t) + v + a,
            },
            {
                "video_patch_rows": v,
                "audio_rows": a,
                "text_rows": t,
                "geometry_scope": "proposed patch-token counts, not an RGB pipeline capture",
                "weights_loaded": False,
            },
        )
if "qwen21" in args.models:
    with torch.device("meta"):
        model = diffusers.QwenImage21Transformer2DModel.from_config(
            json.loads((S / "qwen21.json").read_text())
        ).to(dtype=torch.bfloat16)
    for image, refs in [(image, refs) for image in [4096, 16384] for refs in [0, 1, 3]]:
        text = 512
        encoder = text + refs * image // 4
        mask = torch.zeros(1, encoder + image // 4, dtype=torch.bool)
        mask[:, text:] = True
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

        cache = QwenImage21KVCache(model.config.num_layers)
        kwargs = {
            "hidden_states": empty(1, image * (refs + 1), 64),
            "encoder_hidden_states": empty(1, encoder, 4096),
            "timestep": empty(1),
            "img_shapes": [[(1, int(image**0.5), int(image**0.5))] * (refs + 1)],
            "img_mask": mask,
            "kv_cache": cache,
        }
        for phase in ["extract", "cached"]:
            run(
                "qwen21-grid" + str(int(image**0.5)) + "-refs" + str(refs) + "-" + phase,
                model,
                dict(**kwargs, kv_cache_mode=phase),
                {
                    "target_grid": [1, int(image**0.5), int(image**0.5)],
                    "reference_images": refs,
                    "text_rows": text,
                    "phase": phase,
                    "geometry_scope": "proposed transformer input layout; VLM token values not executed",
                    "weights_loaded": False,
                },
            )

if "ltx25" in args.models:
    cfg = json.loads((S / "ltx25.json").read_text())
    with torch.device("meta"):
        model = diffusers.LTX2VideoTransformer3DModel.from_config(cfg).to(dtype=torch.bfloat16)
    for f, h, w in [(17, 32, 32), (17, 64, 128)]:
        v = f * h * w
        a = 1024
        t = 512
        run(
            "ltx25-video" + str(v),
            model,
            {
                "hidden_states": empty(1, v, 128),
                "audio_hidden_states": empty(1, a, 128),
                "encoder_hidden_states": empty(1, t, 4096),
                "audio_encoder_hidden_states": empty(1, t, 2048),
                "timestep": empty(1),
                "sigma": empty(1),
                "num_frames": f,
                "height": h,
                "width": w,
                "audio_num_frames": a,
            },
            {
                "latent_grid": [f, h, w],
                "video_rows": v,
                "audio_rows": a,
                "text_rows": t,
                "geometry_scope": "proposed latent geometry; no VAE/pipeline execution",
                "weights_loaded": False,
            },
        )
(OUT / ("summary-" + "-".join(args.models) + ".json")).write_text(
    json.dumps(reports, indent=2) + "\n"
)
