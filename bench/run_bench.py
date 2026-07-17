from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.generate_configs import (  # noqa: E402
    WORKLOADS,
    cost,
    expand_trace,
    supported_workloads,
)
from bench.ops import SPECS, backend_supports, primary_param  # noqa: E402
from bench.tracer_node import load_bundle  # noqa: E402
from comfy_kitchen.registry import registry  # noqa: E402

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
DTYPE_NAMES = {v: k for k, v in DTYPES.items()}

CSV_FIELDS = [
    "op",
    "case_id",
    "backend",
    "dtype",
    "model",
    "workload",
    "trace",
    "from",
    "static",
    "dynamic",
    "layout",
    "calls",
    "importance",
    "ms",
    "flops",
    "io_bytes",
    "intensity",
    "tflops",
    "gbps",
    "ck_version",
    "gpu_name",
    "gpu_uuid",
    "compute_capability",
]
_CSV_FLOATS = ("ms", "flops", "io_bytes", "intensity", "tflops", "gbps", "calls", "importance")


def available_ops() -> set[str]:
    return set(SPECS)


def default_ops() -> set[str]:
    return {op for op in SPECS if not op.startswith("dequantize_")}


def resolve_ops(spec: str) -> set[str]:
    if spec == "default":
        return default_ops()
    if spec == "all":
        return available_ops()
    return {op.strip() for op in spec.split(",") if op.strip()}


def supported_dtypes(backend: str, op_name: str, traced: str | None = None) -> list[torch.dtype]:
    spec = SPECS.get(op_name)
    if spec is not None and spec.fixed_dtypes is not None:
        return list(spec.fixed_dtypes)
    constraints = registry.get_constraints(backend, op_name)
    param = primary_param(op_name)
    allowed = (
        constraints.params[param].dtypes
        if constraints is not None and param is not None and param in constraints.params
        else None
    )
    out = [torch.bfloat16, torch.float16, torch.float32]
    if allowed is not None:
        out = [d for d in out if d in allowed]
    traced_dtype = DTYPES.get(str(traced))
    if traced_dtype in out:
        out.remove(traced_dtype)
        out.insert(0, traced_dtype)
    return out


def make_runner(
    case: dict, *, backend: str, dtype: torch.dtype, device: torch.device
) -> Callable[[], Any]:
    op = case["op"]
    if not backend_supports(backend, op):
        raise ValueError(f"backend {backend!r} does not support {op!r}")
    spec = SPECS.get(op)
    if spec is None:
        raise ValueError(f"no OpSpec for {op!r}")
    kwargs = spec.kwargs(case, dtype, device, backend)
    impl = registry.get_implementation(op, backend=backend, kwargs=kwargs)
    return lambda: impl(**kwargs)


def warm_gpu(device: torch.device, seconds: float = 0.5) -> None:
    x = torch.randn((2048, 2048), device=device, dtype=torch.float16)
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        x = x @ x
    torch.cuda.synchronize(device)


def bench_cuda(
    fn: Callable[[], object],
    *,
    device: torch.device,
    warmup_ms: float = 20.0,
    rep_ms: float = 200.0,
    spin_ms: float = 20.0,
    flush: torch.Tensor | None = None,
) -> float:
    """Mean seconds per call with L2 flush and overlapping spin-wait."""
    if device.type != "cuda":
        raise ValueError("bench_cuda requires a CUDA device")
    if not hasattr(torch.cuda, "_sleep"):
        raise RuntimeError("This PyTorch build does not expose torch.cuda._sleep")

    # >L2 so each iter is DRAM-cold; keep modest so wall time isn't flush-dominated.
    flush = flush if flush is not None else torch.empty(96 << 20, dtype=torch.uint8, device=device)
    spin = max(1, int(spin_ms * torch.cuda.get_device_properties(device).clock_rate))
    fn()
    torch.cuda.synchronize(device)

    def timed_batch(n: int) -> list[float]:
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        torch.cuda._sleep(spin)
        for start, end in zip(starts, ends, strict=True):
            flush.add_(1)
            start.record()
            fn()
            end.record()
        torch.cuda.synchronize(device)
        return [s.elapsed_time(e) for s, e in zip(starts, ends, strict=True)]

    est = sum(timed_batch(5)) / 5
    if est <= 0:
        raise RuntimeError(f"invalid CUDA timing estimate: {est} ms")
    # Caps matter: flush cost is outside the timed region but dominates wall clock
    # when the kernel itself is <<1ms (otherwise we'd schedule thousands of flushes).
    torch.cuda._sleep(spin)
    for _ in range(max(2, min(10, math.ceil(warmup_ms / est)))):
        flush.add_(1)
        fn()
    torch.cuda.synchronize(device)
    times = timed_batch(max(10, min(100, math.ceil(rep_ms / est))))
    return sum(times) / len(times) / 1_000.0


def case_id(row: dict[str, Any]) -> str:
    payload = {k: row[k] for k in ("op", "backend", "dtype", "model", "workload", "trace")}
    payload.update(
        static=row["case"]["static"],
        dynamic=row["case"]["dynamic"],
        layout=row["case"].get("layout"),
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]


def select_coverage(rows: list[dict[str, Any]], coverage: float) -> list[dict[str, Any]]:
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    groups: dict[tuple, list] = defaultdict(list)
    for row in rows:
        groups[(row["op"], row["backend"], row["dtype"], row["model"], row["workload"])].append(row)
    selected = []
    for group in groups.values():
        for row in group:
            row["importance"] = row["flops_total"] or row["bytes_total"]
        group.sort(key=lambda r: r["importance"], reverse=True)
        total = sum(r["importance"] for r in group)
        acc = 0.0
        for row in group:
            selected.append(row)
            acc += row["importance"]
            if total == 0 or acc >= coverage * total:
                break
    return selected


def build_matrix(
    *,
    recipe_dir: Path,
    backends: list[str],
    ops: set[str],
    coverage: float,
    only_case_id: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in sorted(recipe_dir.glob("*.json")):
        model = load_bundle(trace).model
        for workload in supported_workloads(model) or [None]:
            model, cases = expand_trace(trace, workload)
            for case in cases:
                if case["op"] not in ops:
                    continue
                for backend in backends:
                    if not backend_supports(backend, case["op"]):
                        continue
                    for dtype in supported_dtypes(
                        backend, case["op"], traced=case["static"].get("dtype")
                    ):
                        typed = {**case, "static": {**case["static"], "dtype": DTYPE_NAMES[dtype]}}
                        calls, flops_total, bytes_total = cost(typed)
                        row = {
                            "trace": trace.name,
                            "model": model,
                            "workload": workload or "replay",
                            "backend": backend,
                            "dtype": DTYPE_NAMES[dtype],
                            "op": case["op"],
                            "case": typed,
                            "calls": calls,
                            "flops_total": flops_total,
                            "bytes_total": bytes_total,
                        }
                        row["case_id"] = case_id(row)
                        rows.append(row)
    if only_case_id is not None:
        matches = [r for r in rows if r["case_id"] == only_case_id]
        if not matches:
            raise ValueError(f"case_id {only_case_id!r} was not found")
        return matches
    return select_coverage(rows, coverage)


def _csv_meta(metadata: dict[str, Any]) -> dict[str, Any]:
    g = metadata["gpu"]
    return {
        "ck_version": metadata["ck_version"],
        "gpu_name": g["name"],
        "gpu_uuid": g["uuid"],
        "compute_capability": g["compute_capability"],
    }


def _fsync(handle) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def init_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=CSV_FIELDS).writeheader()
        _fsync(handle)


def append_csv_row(path: Path, row: dict[str, Any], metadata: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore").writerow(
            {**row, **_csv_meta(metadata)}
        )
        _fsync(handle)


def load_csv_results(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in _CSV_FLOATS:
            if row.get(key) not in (None, ""):
                row[key] = float(row[key])
    return rows


def completed_case_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {r["case_id"] for r in csv.DictReader(handle) if r.get("case_id")}


def run_matrix(
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    warmup_ms: float,
    rep_ms: float,
    spin_ms: float,
    csv_path: Path,
    metadata: dict[str, Any],
) -> int:
    flush = torch.empty(96 << 20, dtype=torch.uint8, device=device)
    n_new = 0
    progress = tqdm(rows, desc="Benchmark", unit="case")
    for row in progress:
        case = row["case"]
        progress.set_postfix_str(
            f"{row['op']} {row['backend']} {row['dtype']} {row['model']}/{row['workload']}"
        )
        try:
            seconds = bench_cuda(
                make_runner(
                    case, backend=row["backend"], dtype=getattr(torch, row["dtype"]), device=device
                ),
                device=device,
                warmup_ms=warmup_ms,
                rep_ms=rep_ms,
                spin_ms=spin_ms,
                flush=flush,
            )
        except (RuntimeError, ValueError, torch.OutOfMemoryError) as error:
            progress.write(f"skip {row['op']} ({row['backend']}/{row['dtype']}): {error}")
            torch.cuda.empty_cache()
            continue
        calls = max(float(row["calls"]), 1.0)
        flops = row["flops_total"] / calls
        io_bytes = row["bytes_total"] / calls
        result = {
            **{k: v for k, v in row.items() if k != "case"},
            "static": json.dumps(case["static"], sort_keys=True),
            "dynamic": json.dumps(case["dynamic"], sort_keys=True),
            "layout": case.get("layout", ""),
            "from": case.get("from", ""),
            "ms": seconds * 1_000,
            "flops": flops,
            "io_bytes": io_bytes,
            "intensity": flops / io_bytes if io_bytes else 0.0,
            "tflops": flops / seconds / 1e12 if flops else 0.0,
            "gbps": io_bytes / seconds / 1e9 if io_bytes else 0.0,
        }
        append_csv_row(csv_path, result, metadata)
        n_new += 1
    return n_new


def _slug(value: str) -> str:
    s = re.sub(r"[^a-z0-9._]+", "-", value.strip().lower())
    return re.sub(r"-{2,}", "-", s).strip("-.") or "unknown"


def run_dir_name(metadata: dict[str, Any]) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    version = _slug(str(metadata.get("ck_version", "unknown")))
    gpu = _slug(str((metadata.get("gpu") or {}).get("name", "unknown-gpu")))
    return f"{stamp}_ck-{version}_{gpu}"


def run_metadata(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    try:
        ck_version = importlib.metadata.version("comfy-kitchen")
    except importlib.metadata.PackageNotFoundError:
        ck_version = "unknown"
    return {
        "schema": "ck.benchmark.run.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "ck_version": ck_version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": {
            "cuda_device": device.index,
            "name": props.name,
            "uuid": str(props.uuid),
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_bytes": props.total_memory,
            "l2_cache_bytes": props.L2_cache_size,
            "clock_rate_khz": props.clock_rate,
        },
        "settings": {
            "backends": args.backends,
            "ops": args.ops,
            "coverage": args.coverage,
            "warmup_ms": args.warmup_ms,
            "rep_ms": args.rep_ms,
            "spin_ms": args.spin_ms,
            "case_id": args.case_id,
        },
    }


def _print_rows(rows: list[dict[str, Any]], *, timed: bool) -> None:
    if not rows:
        return
    op_w = max(len(r["op"]) for r in rows) + 2
    bd_w = 22
    if timed:
        print(
            f"\n{'op':<{op_w}}  {'backend/dtype':<{bd_w}}  {'ms':>9}  {'performance':>14}    configuration"
        )
        print("-" * (op_w + bd_w + 50))
        for r in rows:
            metric = f"{r['tflops']:.2f} TFLOP/s" if r["flops"] else f"{r['gbps']:.2f} GB/s"
            print(
                f"{r['op']:<{op_w}}  {r['backend'] + '/' + r['dtype']:<{bd_w}}  "
                f"{r['ms']:>8.4f}  {metric:>14}    {r['dynamic']} {r['static']}"
            )
        return
    mw = max(len(f"{r['model']}/{r['workload']}") for r in rows) + 2
    print(f"\n{'op':<{op_w}}  {'backend/dtype':<{bd_w}}  {'model/workload':<{mw}}    configuration")
    print("-" * (op_w + bd_w + mw + 40))
    for r in rows:
        c = r["case"]
        dyn = " ".join(f"{k}={v}" for k, v in c["dynamic"].items())
        static = " ".join(f"{k}={v}" for k, v in c["static"].items() if k != "dtype")
        print(
            f"{r['op']:<{op_w}}  {r['backend'] + '/' + r['dtype']:<{bd_w}}  "
            f"{r['model'] + '/' + r['workload']:<{mw}}    {dyn} {static}".rstrip()
        )


def _cli_epilog() -> str:
    import comfy_kitchen  # noqa: F401

    backends = registry.list_backends()
    bits = [
        f"{n} ({'available' if i.get('available') and not i.get('disabled') else 'unavailable'})"
        for n, i in sorted(backends.items())
    ]
    ops = ", ".join(sorted(SPECS))
    dq = ", ".join(sorted(op for op in SPECS if op.startswith("dequantize_")))
    return (
        f"available backends:\n  {', '.join(bits)}\n"
        f"available ops:\n  {ops}\n"
        f"default excludes dequantize_* (use --ops all to include):\n  {dq}\n"
        f"workloads (selected via traces/models):\n  {', '.join(sorted(WORKLOADS))}"
    )


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=32, width=100)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CUDA microbenchmarks from triage traces",
        formatter_class=_HelpFormatter,
        epilog=_cli_epilog(),
    )
    p.add_argument(
        "--recipe-dir", type=Path, default=ROOT / "bench/recipes", help="triage trace JSON dir"
    )
    p.add_argument(
        "--backends", nargs="+", default=["cuda"], metavar="NAME", help="backends (see epilog)"
    )
    p.add_argument(
        "--ops", default="default", metavar="OPS", help="ops list, 'default' (no DQ), or 'all'"
    )
    p.add_argument(
        "--coverage",
        type=float,
        default=0.8,
        metavar="FRAC",
        help="keep top shapes covering this share of estimated end-to-end time (0.8 = 80%%)",
    )
    p.add_argument(
        "--warmup-ms", type=float, default=20.0, metavar="MS", help="CUDA warmup per case"
    )
    p.add_argument(
        "--rep-ms", type=float, default=200.0, metavar="MS", help="CUDA measure per case"
    )
    p.add_argument(
        "--spin-ms", type=float, default=20.0, metavar="MS", help="GPU spin between launches"
    )
    p.add_argument("--device", type=int, default=0, help="CUDA device index")
    p.add_argument("--case-id", default=None, help="run one matrix case id")
    p.add_argument("--dry-run", action="store_true", help="print selected cells and exit")
    p.add_argument("--out", type=Path, default=ROOT / ".bench", help="parent dir for run artifacts")
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help="resume run dir (skip done case_ids)",
    )
    p.add_argument("--report", action="store_true", help="also write report.html")
    p.add_argument(
        "--peak-fp16-tflops",
        type=float,
        default=None,
        help="roofline peak FP16 TFLOP/s (INT8/FP8=2x, INT4/FP4=4x; with --report)",
    )
    p.add_argument(
        "--peak-bw-gbs",
        type=float,
        default=None,
        help="roofline peak bandwidth in GB/s (with --report)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.backends = [b for item in args.backends for b in item.split(",") if b]
    ops = resolve_ops(args.ops)
    if unknown := ops - available_ops():
        raise SystemExit(f"unknown ops: {sorted(unknown)}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)

    rows = build_matrix(
        recipe_dir=args.recipe_dir,
        backends=args.backends,
        ops=ops,
        coverage=args.coverage,
        only_case_id=args.case_id,
    )
    if not rows:
        raise SystemExit("No benchmark cells matched the requested ops/backends/recipes")
    print(f"Selected {len(rows)} benchmark cells at {args.coverage:.0%} coverage", flush=True)
    if args.dry_run:
        _print_rows(rows, timed=False)
        print(f"\nDry run: {len(rows)} cells (no timing)")
        return

    done: set[str] = set()
    if args.resume is not None:
        out_dir = args.resume
        if not out_dir.is_dir():
            raise SystemExit(f"resume directory not found: {out_dir}")
        meta_path = out_dir / "metadata.json"
        metadata = (
            json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.is_file()
            else run_metadata(args, device)
        )
        metadata["resumed_at"] = datetime.now().astimezone().isoformat()
        metadata["settings"] = run_metadata(args, device)["settings"]
        csv_path = out_dir / "results.csv"
        done = completed_case_ids(csv_path)
        if not csv_path.is_file():
            init_csv(csv_path)
        pending = [r for r in rows if r["case_id"] not in done]
        print(
            f"Resuming {out_dir}: {len(done)} done, {len(pending)} remaining (of {len(rows)})",
            flush=True,
        )
    else:
        metadata = run_metadata(args, device)
        out_dir = args.out / run_dir_name(metadata)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "results.csv"
        init_csv(csv_path)
        pending = rows

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    n_new = 0
    if pending:
        warm_gpu(device)
        n_new = run_matrix(
            pending,
            device=device,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
            spin_ms=args.spin_ms,
            csv_path=csv_path,
            metadata=metadata,
        )
    elif args.resume is not None:
        print("Nothing left to run.")

    results = load_csv_results(csv_path)
    _print_rows(results, timed=True)
    suffix = f" ({n_new} new this session)" if done else ""
    print(f"\nWrote {len(results)} rows to {out_dir}{suffix}")

    if args.report:
        if args.peak_fp16_tflops is None or args.peak_bw_gbs is None:
            raise SystemExit("--report requires --peak-fp16-tflops and --peak-bw-gbs")
        from bench.interactive_report import build_html, load_run

        csv_rows, meta = load_run(out_dir)
        report = out_dir / "report.html"
        report.write_text(
            build_html(
                csv_rows,
                meta,
                peak_fp16_tflops=args.peak_fp16_tflops,
                peak_bw_gbs=args.peak_bw_gbs,
            ),
            encoding="utf-8",
        )
        print(f"Report: {report}")
    else:
        print(
            f"Interactive report: python bench/interactive_report.py {out_dir} "
            "--peak-fp16-tflops <TFLOP/s> --peak-bw-gbs <GB/s>"
        )


if __name__ == "__main__":
    main()
