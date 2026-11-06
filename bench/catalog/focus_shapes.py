"""Select repeated block GEMMs by call-weighted FLOP share; not measured runtime."""

import argparse
import json
from pathlib import Path


def focus(bundle, min_share=0.005):
    def work(e):
        m = next(iter(e["dynamic"]["M"]["values"].values()))
        return 2 * m * e["static"]["N"] * e["static"]["K"] * sum(e["call_count"].values())

    total = sum(work(e) for e in bundle["events"])
    kept, omitted = [], []
    for event in bundle["events"]:
        share = work(event) / total if total else 0
        # Full traces retain these projections for audit. Focus specifically on
        # expensive repeated blocks; final heads and input projections are outside.
        block = any(t.startswith(("blocks.", "transformer_blocks.")) for t in event["tags"])
        if block and share >= min_share:
            kept.append(event)
        else:
            omitted.append(
                {
                    "M": next(iter(event["dynamic"]["M"]["values"].values())),
                    "N": event["static"]["N"],
                    "K": event["static"]["K"],
                    "calls": sum(event["call_count"].values()),
                    "estimated_flop_share": share,
                    "layers": event["tags"],
                    "reason": "outside_repeated_blocks" if not block else "below_flop_share_cutoff",
                }
            )
    result = dict(bundle, events=kept)
    result["meta"] = dict(
        bundle["meta"],
        selection={
            "metric": "call-weighted FLOP share, not measured end-to-end time",
            "min_share": min_share,
            "scope": "repeated transformer blocks",
            "retained_flop_fraction": sum(work(e) for e in kept) / total,
            "kept_groups": len(kept),
            "omitted_groups": len(omitted),
            "omitted": omitted,
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-share", type=float, default=0.005)
    args = parser.parse_args()
    if not 0 <= args.min_share < 1:
        parser.error("--min-share must be in [0, 1)")
    root = Path(__file__).resolve().parent
    out = root / "focused_shapes"
    out.mkdir(exist_ok=True)
    for path in sorted((root / "shape_runs").glob("*.json")):
        result = focus(json.loads(path.read_text()), args.min_share)
        (out / path.name).write_text(json.dumps(result, indent=2) + "\n")
        print(
            path.stem,
            result["meta"]["selection"]["kept_groups"],
            result["meta"]["selection"]["retained_flop_fraction"],
        )


if __name__ == "__main__":
    main()
