import csv

import pytest
import torch

from bench import run_bench
from bench.interactive_report import build_html
from bench.ops import SPECS, adaln, qdq, rope
from bench.peaks import compute_precision, validate_peaks


def test_int8_peak_is_not_inferred_from_fp16():
    row = dict(op="int8_linear", dtype="bfloat16", static="{}", flops="100")
    assert compute_precision(row) == "int8"
    page = build_html([row], {}, peak_fp16_tflops=2250, peak_bw_gbs=7700)
    assert '"compute_peak":null' in page
    page = build_html([row], {}, peak_bw_gbs=7700, precision_peaks={"int8": 153.5})
    assert '"compute_peak":153.5' in page


@pytest.mark.parametrize("precision", ["fp4", "fp8", "int4", "int8"])
def test_quantized_precision_overrides_source_dtype(precision):
    assert compute_precision(dict(op="scaled_mm_" + precision, dtype="float16")) == precision


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_peaks_are_rejected(value):
    with pytest.raises(ValueError):
        validate_peaks({"int8": value}, 1000)


def test_failures_are_visible_and_not_plotted():
    page = build_html([dict(op="int8_linear", status="failed", error="unsupported shape")], {}, peak_bw_gbs=1000)
    assert "const rows = [];" in page
    assert "unsupported shape" in page
    assert "failed benchmark records" in page


def test_current_rms_adaln_arguments():
    assert "rms_adaln" in SPECS and "ada_rmsnorm" not in SPECS
    case = dict(op="rms_adaln", static={"hidden": 8}, dynamic={"M": 2})
    arguments = adaln.kwargs(case, torch.float32, torch.device("cpu"), "eager")
    result = run_bench.registry.get_implementation("rms_adaln", backend="eager", kwargs=arguments)(**arguments)
    assert result.shape == (1, 2, 8)


def test_rope_uses_fp32_frequencies_and_accounts_for_them():
    case = dict(op="rms_rope", static={"heads": 2, "head_dim": 8, "dtype": "bfloat16"}, dynamic={"seq": 4})
    arguments = rope.kwargs(case, torch.bfloat16, torch.device("cpu"), "eager")
    assert arguments["freqs_cis"].dtype == torch.float32
    assert rope.estimate(case) == (0, 2 * 4 * 4 * 2 * 8 + 4 * 2 * 8 * 4 + 8 * 8)


@pytest.mark.parametrize("op,expected", [("quantize_nvfp4", 4518), ("dequantize_nvfp4", 7428)])
def test_padded_nvfp4_io_includes_scales(op, expected):
    case = dict(op=op, static={"cols": 65, "padded_cols": 80, "format": "nvfp4_e2m1_e4m3", "dtype": "bfloat16"}, dynamic={"rows": 17, "padded_rows": 32})
    assert qdq.estimate(case) == (0, expected)


def test_legacy_csv_resume_preserves_success_and_retries_failure(tmp_path):
    path = tmp_path / "results.csv"
    fields = [f for f in run_bench.CSV_FIELDS if f not in {"status", "error"}]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(case_id="old", op="rms_rope", ms=1))
    run_bench.ensure_csv_schema(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["status"] == "ok" and rows[0]["ms"] == "1"
    with path.open("a", newline="") as handle:
        csv.DictWriter(handle, fieldnames=run_bench.CSV_FIELDS).writerow(dict(case_id="retry", status="failed"))
    assert run_bench.completed_case_ids(path) == {"old"}
