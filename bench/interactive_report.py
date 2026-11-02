from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.peaks import add_peak_arguments, compute_peaks, compute_precision, validate_peaks  # noqa: E402


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
    peak_fp16_tflops: float | None = None,
    peak_bw_gbs: float,
    precision_peaks: dict | None = None,
) -> str:
    peaks = dict(metadata.get("compute_peaks_tops", {}))
    if peak_fp16_tflops is not None:
        peaks["fp16"] = peak_fp16_tflops
    peaks.update({k: v for k, v in (precision_peaks or {}).items() if v is not None})
    if peak_bw_gbs is None:
        peak_bw_gbs = metadata.get("peak_bw_gbs")
    validate_peaks(peaks, peak_bw_gbs)
    failures = [r for r in rows if r.get("status", "ok") != "ok"]
    plotted = []
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        precision = compute_precision(row)
        plotted.append({**row, "compute_precision": precision, "compute_peak": peaks.get(precision)})
    failure_data = json.dumps(failures, separators=(",", ":")).replace("</", "<\\/")
    data = json.dumps(plotted, separators=(",", ":")).replace("</", "<\\/")
    meta = json.dumps(
        {
            **metadata,
            "peak_fp16_tflops": peak_fp16_tflops,
            "peak_bw_gbs": peak_bw_gbs,
            "compute_peaks_tops": peaks,
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
    <p class="meta">Bandwidth uses estimated logical I/O, not measured DRAM traffic. Compute ceilings use explicitly supplied dense peaks; missing peaks remain unknown.</p>
    <details id="failures" hidden><summary id="failure-count"></summary><pre id="failure-details"></pre></details>
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

const failures = {failure_data};
if (failures.length) {{
  document.getElementById("failures").hidden = false;
  document.getElementById("failure-count").textContent = `${{failures.length}} failed benchmark records`;
  document.getElementById("failure-details").textContent = JSON.stringify(failures, null, 2);
}}
const peakFor = r => r.compute_peak == null ? null : Number(r.compute_peak);

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

  const ceilings = [...new Map(points.filter(r => peakFor(r) > 0).map(r => [r.compute_precision, peakFor(r)])).entries()];
  const peakTflops = Math.max(0, ...ceilings.map(([precision, peak]) => peak));
  const peakGbps = peakBwGbs;
  // ridge I* = peak_FLOP/s / peak_byte/s = (TFLOP/s * 1e12) / (GB/s * 1e9) = 1000 * TFLOP/s / GB/s
  const ridge = 1000 * peakTflops / peakGbps;
  const memBound = x => peakGbps * x / 1000; // TFLOP/s

  let xmin = Math.min(...xvals), xmax = Math.max(...xvals);
  let ymax = Math.max(...yvals, 0);
  if (compute && peakTflops > 0) {{
    xmin = Math.min(xmin, ridge / 10);
    xmax = Math.max(xmax, ridge * 10);
    ymax = Math.max(ymax, peakTflops * 1.05);
  }} else if (!compute) {{
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
  svg += `<text class="label" x="${{(L+W-R)/2}}" y="${{H-12}}" text-anchor="middle">${{compute ? "Arithmetic intensity (operations/byte)" : "Working-set I/O bytes"}}</text>`;
  svg += `<text class="label" transform="translate(18 ${{H/2}}) rotate(-90)" text-anchor="middle">${{compute ? "Performance (trillion operations/s)" : "Logical I/O bandwidth (GB/s)"}}</text>`;

  if (compute) {{
    // A separate roof for each known precision; never infer tensor-core ratios.
    for (const [precision, peak] of ceilings) {{
      let d = "";
      for (let i = 0; i <= 64; i++) {{
        const t = i / 64;
        const x = Math.exp(Math.log(xmin) * (1-t) + Math.log(xmax) * t);
        d += `${{i ? "L" : "M"}} ${{sx(x)}} ${{sy(Math.min(peak, memBound(x)))}} `;
      }}
      svg += `<path class="roof" d="${{d}}"><title>${{esc(precision)}} dense peak</title></path>`;
    }}
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
    ? (ceilings.map(([precision, peak]) => `${{precision.toUpperCase()}} ${{peak.toFixed(1)}} TOp/s dense`).join(" · ") || "Compute peak not supplied")
    : `${{peakGbps.toFixed(0)}} GB/s`;
  const unknown = compute && points.some(r => !(peakFor(r) > 0));
  legend.insertAdjacentHTML("beforeend", `<span>— ${{esc(roofLabel)}}${{unknown ? " · some compute ceilings unknown" : ""}}</span>`);
}}

function select(r) {{
  const setup = {{
    case_id:r.case_id, op:r.op, backend:r.backend, dtype:r.dtype,
    model:r.model, workload:r.workload, source:r.from,
    static:JSON.parse(r.static), dynamic:JSON.parse(r.dynamic), layout:r.layout,
    measured:{{ms:number(r,"ms"),trillion_ops_per_s:number(r,"tflops"),logical_gbps:number(r,"gbps"),compute_peak:peakFor(r),compute_peak_pct:peakFor(r)>0 ? 100*number(r,"tflops")/peakFor(r) : null}}
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
  `CK ${{metadata.ck_version||rows[0]?.ck_version||"unknown"}} · ${{gpu.name||rows[0]?.gpu_name||"unknown GPU"}} · CC ${{gpu.compute_capability||rows[0]?.compute_capability||"?"}} · explicit per-precision compute peaks · ${{peakBwGbs.toFixed(0)}} GB/s · ${{metadata.created_at||""}}`;
render();
</script>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build interactive HTML report from a bench run")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    add_peak_arguments(parser)
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
            precision_peaks=compute_peaks(args),
        ),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
