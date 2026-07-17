from __future__ import annotations

import argparse
import ast
import json
import operator as op
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.ops import CONSUMERS, ESTIMATORS  # noqa: E402
from bench.tracer_node import load_bundle  # noqa: E402

RECIPES_DIR = Path(__file__).parent / "recipes"
WORKLOADS_PATH = Path(__file__).parent / "workloads.json"

_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.FloorDiv: op.floordiv}


def load_workloads() -> dict:
    return json.loads(WORKLOADS_PATH.read_text())


WORKLOADS = load_workloads()


def supported_workloads(model: str) -> list[str]:
    return [n for n, v in WORKLOADS.items() if model in v.get("models", ())]


def resolve_expr(expr: str, sym: dict[str, int]) -> int:
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            return int(_BIN[type(n.op)](ev(n.left), ev(n.right)))
        if isinstance(n, ast.Name):
            return int(sym[n.id])
        raise ValueError(ast.dump(n))

    return int(ev(ast.parse(expr, mode="eval")))


def resolve_axis(payload: dict, sym: dict[str, int] | None):
    if sym is not None and payload.get("binding") is not None:
        return resolve_expr(str(payload["binding"]), sym)
    vals = payload.get("values") or {}
    return vals.get("default", next(iter(vals.values())))


def derive_cases(primaries: list[dict]) -> list[dict]:
    out: dict[str, dict] = {}
    for case in primaries:
        for ops, sources, fn in CONSUMERS:
            if case["op"] not in sources:
                continue
            for kernel in ops:
                got = fn(case, kernel)
                if not got:
                    continue
                for item in got if isinstance(got, list) else [got]:
                    key = json.dumps(
                        {
                            "op": item["op"],
                            "static": item["static"],
                            "dynamic": item["dynamic"],
                            "layout": item.get("layout"),
                        },
                        sort_keys=True,
                    )
                    if key in out:
                        out[key]["calls"] += item["calls"]
                    else:
                        out[key] = item
    return list(out.values())


def expand_trace(path: Path, workload: str | None) -> tuple[str, list[dict]]:
    bundle = load_bundle(path)
    sym = None
    if workload is not None:
        wl = WORKLOADS[workload]
        if bundle.model not in wl["models"]:
            raise SystemExit(f"{workload} not valid for {bundle.model}")
        sym = {k: int(v) for k, v in wl.items() if k != "models"}
    primaries = [
        {
            "op": ev.op,
            "static": dict(ev.static),
            "dynamic": {k: resolve_axis(p, sym) for k, p in ev.dynamic.items()},
            "layout": ev.layout,
            "call_count": dict(ev.call_count),
            "layer": ev.layer,
        }
        for ev in bundle.events
    ]
    return bundle.model, derive_cases(primaries)


def cost(case: dict) -> tuple[float, float, float]:
    flops, nbytes = ESTIMATORS[case["op"]](case)
    calls = float(case["calls"])
    return calls, calls * flops, calls * nbytes


def rank(cases: list[dict], *, sort: str, top: int | None, op: set[str] | None) -> list[dict]:
    scored = []
    for case in cases:
        if op and case["op"] not in op:
            continue
        _c, flops, bytes_ = cost(case)
        scored.append({**case, "flops": flops, "bytes": bytes_})
    scored.sort(key=lambda x: x[sort], reverse=True)
    return scored[:top] if top else scored


def main() -> None:
    p = argparse.ArgumentParser(description="Expand triage traces into ranked bench cases")
    p.add_argument("--trace", type=Path, action="append", default=None)
    p.add_argument("--workload", default=None)
    p.add_argument("--op", action="append")
    p.add_argument("--sort", choices=("flops", "bytes", "calls"), default="bytes")
    p.add_argument("--top", type=int, default=None)
    args = p.parse_args()

    traces = []
    for path in args.trace or [RECIPES_DIR]:
        traces.extend(sorted(path.glob("*.json")) if path.is_dir() else [path])
    if not traces:
        raise SystemExit(f"no traces under {RECIPES_DIR}")

    for trace in traces:
        if args.workload == "replay":
            workloads: list[str | None] = [None]
        elif args.workload:
            workloads = [args.workload]
        else:
            workloads = supported_workloads(load_bundle(trace).model) or [None]
        for wl in workloads:
            model, cases = expand_trace(trace, wl)
            cases = rank(cases, sort=args.sort, top=args.top, op=set(args.op) if args.op else None)
            print(f"\n=== model={model}  workload={wl or 'replay'} ===")
            for c in cases:
                dyn = " ".join(f"{k}={v}" for k, v in c["dynamic"].items())
                print(f"{c['op']:<22} {c['calls']:>5} {c['bytes']:>10.3g}  {dyn}")
            print(f"({len(cases)} cases)")


if __name__ == "__main__":
    main()
