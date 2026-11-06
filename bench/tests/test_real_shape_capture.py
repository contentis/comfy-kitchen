import json
from types import SimpleNamespace

import torch
from torch import nn

from bench.tracer_node.tracer import CKOpTrace, capture_and_merge_to_disk, capture_events


class ReusedLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 8)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.proj(x), self.proj(x[:, :2])


def test_each_invocation_preserves_real_input_shape():
    events = capture_events(ReusedLinear(), (torch.zeros(1, 5, 4),), capture_sdpa=False)
    assert [e["dynamic"]["M"]["values"]["default"] for e in events] == [5, 2]
    assert [e["observations"][0]["input_shape"] for e in events] == [[1, 5, 4], [1, 2, 4]]


def test_merge_keeps_distinct_shapes_and_counts(tmp_path):
    model = ReusedLinear()
    path = tmp_path / "bundle.json"
    for _ in range(2):
        capture_and_merge_to_disk(
            model, (torch.zeros(1, 5, 4),), {}, path=path, profile="request", capture_sdpa=False
        )
    bundle = json.loads(path.read_text())
    assert len(bundle["events"]) == 2
    assert all(e["call_count"]["request"] == 2 for e in bundle["events"])


def test_comfy_wrapper_executes_once_and_can_capture_multiple_steps(tmp_path):
    diffusion = ReusedLinear()
    model = SimpleNamespace(model=SimpleNamespace(diffusion_model=diffusion))
    CKOpTrace().attach(model, output_dir=str(tmp_path), capture_sdpa=False, one_shot=False)
    for _ in range(3):
        output = diffusion(torch.zeros(1, 5, 4))
        assert output[0].shape == (1, 5, 8)
    assert diffusion.calls == 3
    bundle = json.loads((tmp_path / "ReusedLinear" / "bundle.json").read_text())
    assert all(e["call_count"]["default"] == 3 for e in bundle["events"])


def test_one_shot_capture_does_not_execute_twice(tmp_path):
    diffusion = ReusedLinear()
    model = SimpleNamespace(model=SimpleNamespace(diffusion_model=diffusion))
    CKOpTrace().attach(model, output_dir=str(tmp_path), capture_sdpa=False, one_shot=True)
    diffusion(torch.zeros(1, 5, 4))
    assert diffusion.calls == 1


def test_pipeline_observer_records_forward_order(tmp_path):
    from bench.tracer_node import track_model_shapes

    model = ReusedLinear()
    path = tmp_path / "pipeline.json"
    with track_model_shapes(model, path=path, metadata={"references": 1}):
        model(torch.zeros(1, 5, 4))
        model(torch.zeros(1, 3, 4))
    data = json.loads(path.read_text())
    assert data["meta"]["profile_context"]["default"]["references"] == 1
    seen = {o["forward_index"] for e in data["events"] for o in e["observations"]}
    assert seen == {1, 2}
