from __future__ import annotations

import json
import logging
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

logger = logging.getLogger("ck.tracer")


def _dtype(x: Any) -> str:
    s = str(x)
    return s.split(".")[-1] if "." in s else s


def _prod(shape: list[int]) -> int:
    n = 1
    for x in shape:
        n *= int(x)
    return n


def _fingerprint(ev: dict) -> str:
    return json.dumps(
        {"op": ev["op"], "static": ev["static"], "layout": ev.get("layout")},
        sort_keys=True,
    )


def _bundle_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.suffix.lower() == ".json" else path / "bundle.json"


def load_bundle(path: str | Path) -> SimpleNamespace:
    data = json.loads(_bundle_path(path).read_text(encoding="utf-8"))
    events = [
        SimpleNamespace(
            op=e["op"],
            static=dict(e.get("static") or {}),
            dynamic=dict(e.get("dynamic") or {}),
            call_count=dict(e.get("call_count") or {}),
            layout=e.get("layout"),
            layer=e.get("layer", ""),
        )
        for e in data.get("events") or []
    ]
    return SimpleNamespace(model=data.get("model", "unknown"), events=events)


def write_bundle(data: dict, path: str | Path) -> Path:
    out = _bundle_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return out


def _merge_to_disk(path: str | Path, events: list[dict], *, profile: str, model: str) -> dict:
    out = _bundle_path(path)
    if out.is_file():
        data = json.loads(out.read_text(encoding="utf-8"))
    else:
        data = {"schema": "ck.triage.bundle.v1", "model": model, "profiles": [], "events": []}
    if not data.get("model") or data["model"] == "unknown":
        data["model"] = model
    profiles = data.setdefault("profiles", [])
    if profile not in profiles:
        profiles.append(profile)

    index = {_fingerprint(e): e for e in data["events"]}
    for incoming in events:
        key = _fingerprint(incoming)
        if key not in index:
            data["events"].append(incoming)
            index[key] = incoming
            continue
        existing = index[key]
        for p, n in incoming["call_count"].items():
            existing["call_count"][p] = existing["call_count"].get(p, 0) + n
        for name, payload in incoming["dynamic"].items():
            slot = existing["dynamic"].setdefault(name, {"values": {}})
            slot.setdefault("values", {}).update(payload.get("values", {}))
    write_bundle(data, out)
    return data


def _is_linear(mod: nn.Module) -> bool:
    if isinstance(mod, nn.Linear):
        return True
    return (
        isinstance(getattr(mod, "in_features", None), int)
        and isinstance(getattr(mod, "out_features", None), int)
        and callable(getattr(mod, "forward", None))
    )


def _dyn(profile: str, **axes: Any) -> dict:
    return {k: {"values": {profile: v}} for k, v in axes.items()}


@contextmanager
def _patch_sdpa(on_call):
    original = functional.scaled_dot_product_attention

    def wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        on_call(args, kwargs)
        return out

    functional.scaled_dot_product_attention = wrapped
    try:
        yield
    finally:
        functional.scaled_dot_product_attention = original


def capture_events(
    model: nn.Module,
    args: tuple = (),
    kwargs: dict | None = None,
    *,
    profile: str = "default",
    capture_linear: bool = True,
    capture_sdpa: bool = True,
) -> list[dict]:
    kwargs = kwargs or {}
    events: list[dict] = []
    inputs: dict[str, list[int] | None] = {}
    hooks = []

    def make_hook(name: str):
        def hook(_m, inp, _out):
            t = inp[0] if inp and isinstance(inp[0], torch.Tensor) else None
            inputs[name] = [int(x) for x in t.shape] if t is not None else None

        return hook

    if capture_linear:
        for name, mod in model.named_modules():
            if _is_linear(mod):
                hooks.append(mod.register_forward_hook(make_hook(name)))

    def on_sdpa(f_args, f_kwargs):
        q, k = f_args[0], f_args[1]
        qs, ks = [int(x) for x in q.shape], [int(x) for x in k.shape]
        events.append(
            {
                "op": "sdpa",
                "layer": f"sdpa.{len(events) + 1:03d}",
                "static": {
                    "batch": qs[0],
                    "heads": qs[1],
                    "head_dim": qs[3],
                    "dtype": _dtype(q.dtype),
                    "causal": bool(f_kwargs.get("is_causal", False)),
                    "dropout_p": float(f_kwargs.get("dropout_p", 0.0)),
                    "kv_heads": ks[1],
                },
                "dynamic": _dyn(profile, q_tokens=qs[2], kv_tokens=ks[2]),
                "call_count": {profile: 1},
                "layout": "BHND",
            }
        )

    try:
        ctx = _patch_sdpa(on_sdpa) if capture_sdpa else nullcontext()
        with ctx, torch.no_grad():
            model(*args, **kwargs)

        if capture_linear:
            for name, mod in model.named_modules():
                shape = inputs.get(name)
                if shape is None or not _is_linear(mod):
                    continue
                w = getattr(mod, "weight", None)
                dtype = w.dtype if isinstance(w, torch.Tensor) else "unknown"
                events.append(
                    {
                        "op": "gemm",
                        "layer": name,
                        "static": {
                            "N": int(mod.out_features),
                            "K": int(mod.in_features),
                            "bias": getattr(mod, "bias", None) is not None,
                            "dtype": _dtype(dtype),
                        },
                        "dynamic": _dyn(profile, M=_prod(shape[:-1]) if len(shape) > 1 else 1),
                        "call_count": {profile: 1},
                        "layout": None,
                    }
                )
    finally:
        for h in hooks:
            h.remove()
    return events


def capture_and_merge_to_disk(
    model: nn.Module,
    args: tuple,
    kwargs: dict,
    *,
    path: str | Path,
    profile: str,
    model_name: str | None = None,
    capture_linear: bool = True,
    capture_sdpa: bool = True,
) -> dict:
    events = capture_events(
        model,
        args,
        kwargs,
        profile=profile,
        capture_linear=capture_linear,
        capture_sdpa=capture_sdpa,
    )
    return _merge_to_disk(path, events, profile=profile, model=model_name or type(model).__name__)


def _default_output_dir() -> str:
    try:
        import folder_paths

        return str(Path(folder_paths.get_output_directory()) / "ck_triage")
    except Exception:
        return str(Path.cwd() / "output" / "ck_triage")


def _diffusion_model(model: Any):
    if hasattr(model, "model") and hasattr(model.model, "diffusion_model"):
        return model.model, model.model.diffusion_model
    if hasattr(model, "diffusion_model"):
        return model, model.diffusion_model
    raise TypeError("Expected a ComfyUI MODEL with .model.diffusion_model")


class CKOpTrace:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "profile": ("STRING", {"default": "default"}),
                "output_dir": ("STRING", {"default": _default_output_dir()}),
                "filename": ("STRING", {"default": "bundle.json"}),
            },
            "optional": {
                "capture_linear": ("BOOLEAN", {"default": True}),
                "capture_sdpa": ("BOOLEAN", {"default": True}),
                "one_shot": ("BOOLEAN", {"default": True}),
                "reset_bundle": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "bundle_path")
    FUNCTION = "attach"
    OUTPUT_NODE = True
    CATEGORY = "comfy_kitchen/triage"

    def attach(
        self,
        model: Any,
        profile: str = "default",
        output_dir: str = "",
        filename: str = "bundle.json",
        capture_linear: bool = True,
        capture_sdpa: bool = True,
        one_shot: bool = True,
        reset_bundle: bool = False,
    ) -> tuple[Any, str]:
        base, diffusion = _diffusion_model(model)
        model_name = type(diffusion).__name__
        name = (filename or "bundle.json").strip()
        if not name.endswith(".json"):
            name = f"{name}.json"
        out = Path(output_dir or _default_output_dir()) / model_name / name
        out.parent.mkdir(parents=True, exist_ok=True)
        if reset_bundle and out.is_file():
            out.unlink()

        state = {
            "fwd": diffusion.forward,
            "base": base,
            "profile": profile or "default",
            "path": out,
            "one_shot": one_shot,
            "linear": capture_linear,
            "sdpa": capture_sdpa,
        }

        def patched(*args, **kwargs):
            if state["one_shot"]:
                state["base"].diffusion_model.forward = state["fwd"]
            try:
                data = capture_and_merge_to_disk(
                    state["base"].diffusion_model,
                    args,
                    kwargs,
                    path=state["path"],
                    profile=state["profile"],
                    model_name=model_name,
                    capture_linear=state["linear"],
                    capture_sdpa=state["sdpa"],
                )
                logger.info(
                    "ck.tracer wrote %d events → %s",
                    len(data["events"]),
                    state["path"],
                )
            except Exception:
                logger.exception("ck.tracer capture failed; continuing")
                state["base"].diffusion_model.forward = state["fwd"]
            return state["fwd"](*args, **kwargs)

        diffusion.forward = patched
        return (model, str(out))


NODE_CLASS_MAPPINGS = {"CKOpTrace": CKOpTrace}
NODE_DISPLAY_NAME_MAPPINGS = {"CKOpTrace": "CK Op Trace"}
