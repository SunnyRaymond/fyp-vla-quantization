"""Extract only W4A8 fields from the completed mixed-bit aggregate."""

import argparse
import json
import math
from pathlib import Path


PROTOCOL_ID = "fastwam-activation-bit-screen-v1"
SOURCE_JOB = "25678987.pbs101"
TEST_EPISODES = list(range(8, 16))
ARMS = {
    "direct",
    "independent",
    "learned_a8_transfer",
    "permuted",
    "rtn",
    "rtn_standard",
    "shared0",
    "w4",
}
EXPECTED_MSE = {
    "independent": 0.0003671862886897819,
    "learned_a8_transfer": 0.0003753119066759934,
    "rtn": 0.00038525556570334624,
}


def extract(source_path):
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("protocol_id") != PROTOCOL_ID or source.get("complete") is not True:
        raise ValueError("Source is not a complete activation-bit-screen aggregate")

    summary = source["a8_summary"]
    per_trajectory = source["a8_per_trajectory"]
    if set(summary) != ARMS or set(per_trajectory) != ARMS:
        raise ValueError("A8 arm set is incomplete or contains unexpected arms")
    for arm in ARMS:
        values = per_trajectory[arm].get("total_mse", [])
        if len(values) != len(TEST_EPISODES) or not all(math.isfinite(float(x)) for x in values):
            raise ValueError(f"Expected eight finite trajectory MSEs for {arm}")
    for arm, expected in EXPECTED_MSE.items():
        actual = float(summary[arm]["total_mse"])
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"Unexpected W4A8 {arm} MSE: {actual}")

    source_result = source.get("sources", {}).get("a8")
    if not source_result or SOURCE_JOB not in source_result:
        raise ValueError("The aggregate does not identify the expected A8 source job")

    return {
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "scope": "Task0 reused fixed TEST; W4A8 action MSE; no closed-loop/native claim",
        "provenance": {
            "kind": "field extraction from local mixed-bit aggregate",
            "source_file": str(source_path.resolve()),
            "a8_source_job": SOURCE_JOB,
            "a8_source_result": source_result,
            "raw_a8_result_available_locally": False,
            "note": "This summary is copied from the local aggregate's A8 fields; it is not recomputed from the remote raw result.",
        },
        "protocol": {
            "activation_bits": 8,
            "weight_bits": 4,
            "test_episode_ids": TEST_EPISODES,
            "trajectory_count": 8,
            "observations_per_trajectory": 2,
            "sampler_seeds": [2026, 2027],
            "dither_draw_seeds": [2101, 2102, 2103, 2104],
            "draw_to_sampler": {"2101": 2026, "2102": 2027, "2103": 2026, "2104": 2027},
            "aggregation": "average observations and draws within each trajectory, then equally average eight trajectories",
            "configuration": "first_frame; 20 inference steps; sigma_shift=1; compile=false; 32-action chunk; W4 G=128; A8 row_maxabs/126 headroom grid",
        },
        "a8_summary": summary,
        "a8_per_trajectory": per_trajectory,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="completed mixed aggregate_result.json")
    parser.add_argument("--output", type=Path, required=True, help="A8-only summary.json")
    args = parser.parse_args()
    result = extract(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("A8_SUMMARY_EXTRACTED", args.output)


if __name__ == "__main__":
    main()
