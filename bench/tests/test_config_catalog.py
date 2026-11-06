import json
from pathlib import Path

import pytest

from bench.generate_configs import expand_trace

CATALOG = Path(__file__).resolve().parents[1] / "catalog"
RUNS = sorted((CATALOG / "shape_runs").glob("*.json"))


@pytest.mark.parametrize("path", RUNS, ids=lambda p: p.stem)
def test_executed_shapes_expand_without_token_retargeting(path):
    bundle = json.loads(path.read_text())
    model, cases = expand_trace(path, None)
    assert model.endswith("Weightless") and cases
    assert bundle["meta"]["status"] == "complete"
    assert bundle["meta"]["context"]["weights_loaded"] is False
    assert (
        sum(sum(e["call_count"].values()) for e in bundle["events"])
        == bundle["meta"]["linear_calls"]
    )
    for event in bundle["events"]:
        assert "binding" not in event["dynamic"]["M"]
        expected_m = next(iter(event["dynamic"]["M"]["values"].values()))
        matches = [
            c
            for c in cases
            if c["op"] == "int8_linear"
            and c["static"]["N"] == event["static"]["N"]
            and c["static"]["K"] == event["static"]["K"]
            and c["dynamic"]["M"] == expected_m
        ]
        assert matches
        expected = {False, True} if event["static"]["K"] % 256 == 0 else {False}
        assert {c["static"]["convrot"] for c in matches} == expected


def rows(profile, fragment):
    bundle = json.loads((CATALOG / "shape_runs" / (profile + ".json")).read_text())
    return {
        next(iter(e["dynamic"]["M"]["values"].values()))
        for e in bundle["events"]
        if any(layer.startswith(fragment) for layer in e["tags"])
    }


def test_executed_geometry_preserves_joint_padding_and_cache_paths():
    assert rows("krea2-image4096", "blocks.0.attn.wq") == {4352}
    assert rows("krea2-image16384", "blocks.0.attn.wq") == {16640}
    assert rows("qwen21-grid128-refs3-extract", "transformer_blocks.0.attn.to_q") == {66048}
    assert rows("qwen21-grid128-refs3-cached", "transformer_blocks.0.attn.to_q") == {16384}
    assert rows("h3-video139264", "proj_in") == {139264}
    assert rows("ltx25-video139264", "transformer_blocks.0.audio_ff") == {1024}
    assert rows("ltx25-video139264", "transformer_blocks.0.ff") == {139264}


def test_focused_catalog_omits_final_heads_and_retains_block_work():
    from bench.catalog.focus_shapes import focus

    for path in RUNS:
        full = json.loads(path.read_text())
        actual = focus(full)
        stored = json.loads((CATALOG / "focused_shapes" / path.name).read_text())
        assert actual == stored
        assert actual["events"]
        assert actual["meta"]["selection"]["retained_flop_fraction"] >= 0.95
        assert all(
            any(t.startswith(("blocks.", "transformer_blocks.")) for t in e["tags"])
            for e in actual["events"]
        )
