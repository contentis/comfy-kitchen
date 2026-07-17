from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_run(run_dir: Path) -> tuple[list[dict[str, str]], dict]:
    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        raise FileNotFoundError(results_path)
    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    return rows, metadata


def build_html(
    rows: list[dict[str, str]],
    metadata: dict,
    *,
    peak_fp16_tflops: float,
    peak_bw_gbs: float,
) -> str:
    data = json.dumps(rows, separators=(",", ":")).replace("</", "<\\/")
    meta = json.dumps(
        {
            **metadata,
            "peak_fp16_tflops": peak_fp16_tflops,
            "peak_bw_gbs": peak_bw_gbs,
        },
        separators=(",", ":"),
    ).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CK Benchmark Report</title>
<style>
:root {{ color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }}
body {{ margin: 0; background: #111318; color: #e7e9ee; }}
header {{ padding: 18px 24px; border-bottom: 1px solid #343842; }}
h1 {{ font-size: 20px; margin: 0 0 6px; }}
.meta {{ color: #9da4b3; font-size: 13px; }}
main {{ display: grid; grid-template-columns: minmax(0,1fr) 340px; min-height: calc(100vh - 76px); }}
.left {{ min-width: 0; padding: 18px 24px; }}
aside {{ border-left: 1px solid #343842; padding: 18px; overflow: auto; }}
.controls {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 12px; }}
.group {{ min-width: 130px; }}
.group-title {{ color: #9da4b3; font-size: 11px; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 6px; }}
label {{ display: block; font-size: 13px; margin: 4px 0; }}
select, button {{ color: inherit; background: #1b1e25; border: 1px solid #464b58; border-radius: 4px; padding: 6px 8px; }}
button {{ cursor: pointer; }}
#chart {{ width: 100%; min-height: 520px; }}
svg {{ width: 100%; height: auto; }}
.axis {{ stroke: #707888; stroke-width: 1; }}
.grid {{ stroke: #2b2f38; stroke-width: 1; }}
.roof {{ stroke: #9da4b3; stroke-width: 1.5; stroke-dasharray: 6 4; fill: none; }}
.tick {{ fill: #9da4b3; font-size: 11px; }}
.label {{ fill: #d7dae2; font-size: 12px; }}
.point {{ cursor: pointer; stroke: #111318; stroke-width: 1; }}
.point:hover {{ stroke: #fff; stroke-width: 2; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #181b21; border: 1px solid #343842; padding: 10px; font-size: 12px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 12px; font-size: 12px; margin-top: 6px; }}
.swatch {{ display: inline-block; width: 10px; height: 10px; margin-right: 5px; }}
@media (max-width: 900px) {{ main {{ grid-template-columns: 1fr; }} aside {{ border-left: 0; border-top: 1px solid #343842; }} }}
</style>
</head>
<body>
<header><h1>CK Benchmark Report</h1><div id="meta" class="meta"></div></header>
<main>
  <section class="left">
    <div class="controls">
      <div class="group"><div class="group-title">Operation</div><select id="op"></select></div>
      <div class="group"><div class="group-title">Backends</div><div id="backends"></div></div>
      <div class="group"><div class="group-title">Workloads</div><div id="workloads"></div></div>
      <div class="group"><div class="group-title">Dtypes</div><div id="dtypes"></div></div>
    </div>
    <div id="chart"></div><div id="legend" class="legend"></div>
  </section>
  <aside>
    <h2 style="font-size:16px;margin-top:0">Selected point</h2>
    <div id="details" class="meta">Click a point to inspect its configuration.</div>
    <pre id="command" hidden></pre>
    <button id="copy" hidden>Copy reproduction command</button>
  </aside>
</main>
<script>
const rows = {data};
const metadata = {meta};
const peakFp16Tflops = Number(metadata.peak_fp16_tflops);
const peakBwGbs = Number(metadata.peak_bw_gbs);
const colors = ["#65a7ff","#ff9f5a","#73d69b","#d68cff","#f1d45b","#f07178","#61c8d0","#b8c0cc"];
const state = {{ op: "", backends: new Set(), workloads: new Set(), dtypes: new Set() }};
const uniq = key => [...new Set(rows.map(r => r[key]))].sort();
const number = (r, key) => Number(r[key] || 0);
const esc = s => String(s).replace(/[&<>]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;"}}[c]));

/** Tensor-core peak vs FP16: INT8/FP8 → 2x, INT4/FP4 → 4x. */
function peakScale(r) {{
  let format = "";
  try {{ format = String(JSON.parse(r.static || "{{}}").format || ""); }} catch {{}}
  const blob = `${{r.op}} ${{r.dtype}} ${{format}}`.toLowerCase();
  if (
    blob.includes("nvfp4") || blob.includes("mxfp4") || blob.includes("fp4")
    || blob.includes("int4") || blob.includes("e2m1")
  ) return 4;
  if (
    blob.includes("int8") || blob.includes("float8") || blob.includes("fp8")
    || blob.includes("mxfp8")
  ) return 2;
  return 1;
}}

function checks(id, key, target) {{
  const root = document.getElementById(id);
  for (const value of uniq(key)) {{
    target.add(value);
    const label = document.createElement("label");
    label.innerHTML = `<input type="checkbox" checked value="${{esc(value)}}"> ${{esc(value)}}`;
    label.querySelector("input").addEventListener("change", e => {{
      e.target.checked ? target.add(value) : target.delete(value);
      render();
    }});
    root.appendChild(label);
  }}
}}

function visible() {{
  return rows.filter(r => r.op === state.op && state.backends.has(r.backend)
    && state.workloads.has(r.workload) && state.dtypes.has(r.dtype));
}}

function logTicks(min, max, count=5) {{
  const lo = Math.log10(Math.max(min, 1e-18));
  const hi = Math.log10(Math.max(max, min * 10));
  return Array.from({{length: count}}, (_, i) => 10 ** (lo + (hi - lo) * i / (count - 1)));
}}

function linTicks(min, max, count=5) {{
  return Array.from({{length: count}}, (_, i) => min + (max - min) * i / (count - 1));
}}

function marker(r, x, y, fill, index, title) {{
  const common = `class="point" data-i="${{index}}" fill="${{fill}}"`;
  if (r.backend === "cuda")
    return `<circle ${{common}} cx="${{x}}" cy="${{y}}" r="6"><title>${{title}}</title></circle>`;
  if (r.backend === "triton")
    return `<rect ${{common}} x="${{x-6}}" y="${{y-6}}" width="12" height="12"><title>${{title}}</title></rect>`;
  if (r.backend === "eager")
    return `<polygon ${{common}} points="${{x}},${{y-8}} ${{x+8}},${{y}} ${{x}},${{y+8}} ${{x-8}},${{y}}"><title>${{title}}</title></polygon>`;
  return `<polygon ${{common}} points="${{x}},${{y-8}} ${{x+7}},${{y+6}} ${{x-7}},${{y+6}}"><title>${{title}}</title></polygon>`;
}}

function render() {{
  const points = visible();
  const root = document.getElementById("chart");
  const legend = document.getElementById("legend");
  legend.innerHTML = "";
  if (!points.length) {{ root.innerHTML = "<p>No points match the active filters.</p>"; return; }}

  const compute = points.some(r => number(r, "flops") > 0);
  const xkey = compute ? "intensity" : "io_bytes";
  const ykey = compute ? "tflops" : "gbps";
  const xvals = points.map(r => Math.max(number(r, xkey), 1e-12));
  const yvals = points.map(r => number(r, ykey));

  const scale = Math.max(1, ...points.map(peakScale));
  const peakTflops = peakFp16Tflops * scale;
  const peakGbps = peakBwGbs;
  // ridge I* = peak_FLOP/s / peak_byte/s = (TFLOP/s * 1e12) / (GB/s * 1e9) = 1000 * TFLOP/s / GB/s
  const ridge = 1000 * peakTflops / peakGbps;
  const memBound = x => peakGbps * x / 1000; // TFLOP/s

  let xmin = Math.min(...xvals), xmax = Math.max(...xvals);
  let ymax = Math.max(...yvals, 0);
  if (compute) {{
    xmin = Math.min(xmin, ridge / 10);
    xmax = Math.max(xmax, ridge * 10);
    ymax = Math.max(ymax, peakTflops * 1.05);
  }} else {{
    ymax = Math.max(ymax, peakGbps * 1.05);
  }}
  if (!(xmax > xmin)) xmax = xmin * 10;
  if (!(ymax > 0)) ymax = 1;

  const W=1000, H=560, L=82, R=24, T=28, B=64;
  // Log X (intensity spans decades), linear Y (easier to read TFLOP/s / GB/s).
  const sx = x => L + (Math.log10(x) - Math.log10(xmin)) / (Math.log10(xmax) - Math.log10(xmin)) * (W - L - R);
  const sy = y => T + (1 - y / ymax) * (H - T - B);
  const dtypes = uniq("dtype");
  const color = dtype => colors[Math.max(0, dtypes.indexOf(dtype)) % colors.length];

  let svg = `<svg viewBox="0 0 ${{W}} ${{H}}" role="img">`;
  for (const x of logTicks(xmin, xmax))
    svg += `<line class="grid" x1="${{sx(x)}}" y1="${{T}}" x2="${{sx(x)}}" y2="${{H-B}}"/><text class="tick" x="${{sx(x)}}" y="${{H-B+18}}" text-anchor="middle">${{x.toPrecision(2)}}</text>`;
  for (const y of linTicks(0, ymax))
    svg += `<line class="grid" x1="${{L}}" y1="${{sy(y)}}" x2="${{W-R}}" y2="${{sy(y)}}"/><text class="tick" x="${{L-8}}" y="${{sy(y)+4}}" text-anchor="end">${{y.toFixed(1)}}</text>`;
  svg += `<line class="axis" x1="${{L}}" y1="${{H-B}}" x2="${{W-R}}" y2="${{H-B}}"/><line class="axis" x1="${{L}}" y1="${{T}}" x2="${{L}}" y2="${{H-B}}"/>`;
  svg += `<text class="label" x="${{(L+W-R)/2}}" y="${{H-12}}" text-anchor="middle">${{compute ? "Arithmetic intensity (FLOP/byte)" : "Working-set I/O bytes"}}</text>`;
  svg += `<text class="label" transform="translate(18 ${{H/2}}) rotate(-90)" text-anchor="middle">${{compute ? "Performance (TFLOP/s)" : "Bandwidth (GB/s)"}}</text>`;

  if (compute) {{
    const x0 = Math.max(xmin, Math.min(xmax, ridge));
    // Sample P=BW*I in data space so the roof is correct under log-X / linear-Y.
    const n = 48;
    let d = "";
    for (let i = 0; i <= n; i++) {{
      const t = i / n;
      const x = Math.exp(Math.log(xmin) * (1 - t) + Math.log(x0) * t);
      const y = Math.min(peakTflops, memBound(x));
      d += `${{i ? "L" : "M"}} ${{sx(x)}} ${{sy(y)}} `;
    }}
    d += `L ${{sx(xmax)}} ${{sy(peakTflops)}}`;
    svg += `<path class="roof" d="${{d}}"/>`;
  }} else {{
    svg += `<line class="roof" x1="${{L}}" y1="${{sy(peakGbps)}}" x2="${{W-R}}" y2="${{sy(peakGbps)}}"/>`;
  }}

  points.forEach((r, i) => {{
    const title = `${{esc(r.backend)}}/${{esc(r.dtype)}} · ${{esc(r.model)}}/${{esc(r.workload)}} · ${{number(r,"ms").toFixed(4)}} ms`;
    svg += marker(r, sx(xvals[i]), sy(Math.max(yvals[i], 0)), color(r.dtype), rows.indexOf(r), title);
  }});
  root.innerHTML = svg + "</svg>";
  root.querySelectorAll(".point").forEach(c => c.addEventListener("click", () => select(rows[Number(c.dataset.i)])));

  legend.insertAdjacentHTML("beforeend", "<strong>Backend:</strong>");
  const symbols = {{cuda:"●",triton:"■",eager:"◆"}};
  [...new Set(points.map(r => r.backend))].sort().forEach(b =>
    legend.insertAdjacentHTML("beforeend", `<span>${{symbols[b]||"▲"}} ${{esc(b)}}</span>`));
  legend.insertAdjacentHTML("beforeend", "<strong>Dtype:</strong>");
  [...new Set(points.map(r => r.dtype))].sort().forEach(d =>
    legend.insertAdjacentHTML("beforeend", `<span><i class="swatch" style="background:${{color(d)}}"></i>${{esc(d)}}</span>`));
  legend.insertAdjacentHTML("beforeend", "<strong>Roofline:</strong>");
  const roofLabel = compute
    ? `${{peakTflops.toFixed(1)}} TFLOP/s (FP16 ${{peakFp16Tflops.toFixed(1)}}${{scale>1 ? ` x${{scale}}` : ""}}) · ${{peakGbps.toFixed(0)}} GB/s`
    : `${{peakGbps.toFixed(0)}} GB/s`;
  legend.insertAdjacentHTML("beforeend", `<span>— ${{roofLabel}}</span>`);
}}

function select(r) {{
  const setup = {{
    case_id:r.case_id, op:r.op, backend:r.backend, dtype:r.dtype,
    model:r.model, workload:r.workload, source:r.from,
    static:JSON.parse(r.static), dynamic:JSON.parse(r.dynamic), layout:r.layout,
    measured:{{ms:number(r,"ms"),tflops:number(r,"tflops"),gbps:number(r,"gbps")}}
  }};
  document.getElementById("details").innerHTML = `<pre>${{esc(JSON.stringify(setup,null,2))}}</pre>`;
  const cmd = `python bench/run_bench.py --case-id ${{r.case_id}} --backends ${{r.backend}} --ops ${{r.op}}`;
  const pre=document.getElementById("command"); pre.hidden=false; pre.textContent=cmd;
  const copy=document.getElementById("copy"); copy.hidden=false; copy.onclick=()=>navigator.clipboard.writeText(cmd);
}}

const opSelect=document.getElementById("op");
for (const value of uniq("op")) opSelect.add(new Option(value,value));
state.op=opSelect.value; opSelect.onchange=e=>{{state.op=e.target.value;render();}};
checks("backends","backend",state.backends); checks("workloads","workload",state.workloads); checks("dtypes","dtype",state.dtypes);
const gpu=metadata.gpu||{{}};
document.getElementById("meta").textContent =
  `CK ${{metadata.ck_version||rows[0]?.ck_version||"unknown"}} · ${{gpu.name||rows[0]?.gpu_name||"unknown GPU"}} · CC ${{gpu.compute_capability||rows[0]?.compute_capability||"?"}} · peak FP16 ${{peakFp16Tflops.toFixed(1)}} TFLOP/s · ${{peakBwGbs.toFixed(0)}} GB/s · ${{metadata.created_at||""}}`;
render();
</script>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build interactive HTML report from a bench run")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--peak-fp16-tflops",
        type=float,
        required=True,
        help="device peak FP16 TFLOP/s (INT8/FP8 roof uses 2x, INT4/FP4 uses 4x)",
    )
    parser.add_argument(
        "--peak-bw-gbs",
        type=float,
        required=True,
        help="device peak memory bandwidth in GB/s",
    )
    args = parser.parse_args()
    rows, metadata = load_run(args.run_dir)
    if not rows:
        raise SystemExit("results.csv contains no benchmark rows")
    output = args.out or args.run_dir / "report.html"
    output.write_text(
        build_html(
            rows,
            metadata,
            peak_fp16_tflops=args.peak_fp16_tflops,
            peak_bw_gbs=args.peak_bw_gbs,
        ),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
