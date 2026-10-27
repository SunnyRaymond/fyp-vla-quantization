"""Standard-library metadata tests for the AQLM closed-loop runtime."""
import copy
import json
from pathlib import Path
import unittest

import runtime


ROOT = Path(__file__).parent


def bound_protocol():
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    job = "25999999.pbs101"
    artifacts = "/scratch/users/ntu/aqlm-pair/artifacts/25999999.pbs101"
    protocol["sources"]["parent_pair_job"] = job
    protocol["sources"]["parent_pair_artifacts"] = artifacts
    for arm in protocol["arms"]:
        protocol["arm_sources"][arm]["source_job"] = job
    return protocol


class RuntimeMetadataTests(unittest.TestCase):
    def test_unbound_pair_is_rejected(self):
        protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            __import__("evaluate").validate_protocol(protocol)

    def test_bound_protocol_and_aqlm_metadata(self):
        protocol = bound_protocol()
        evaluate = __import__("evaluate")
        self.assertIs(evaluate.validate_protocol(protocol), protocol)
        metadata = runtime.expected_install_metadata(protocol, "reconstruction")
        self.assertEqual(metadata["execution"], "decoded_bf16_dense_aqlm")
        self.assertEqual(metadata["transform_implementation"],
                         "smooth05_h128_r0_aqlm_g8_m4_k256_v1")
        self.assertEqual(metadata["official_commit"], runtime.AQLM_COMMIT)
        self.assertEqual(metadata["source_job"], "25999999.pbs101")

    def test_metadata_requires_rotation_evidence(self):
        protocol = bound_protocol()
        metadata = runtime.expected_install_metadata(protocol, "ours")
        metadata.update(orthogonality={"action": 0.0, "video": 0.0},
                        rotation_movement={"action": 1.0, "video": 1.0})
        evaluate = __import__("evaluate")
        self.assertIs(evaluate.validate_install_metadata(metadata, protocol, "ours"), metadata)
        bad = copy.deepcopy(metadata)
        bad["native_low_bit_kernel"] = True
        with self.assertRaises(ValueError):
            evaluate.validate_install_metadata(bad, protocol, "ours")

    def test_anchor_json_pt_header_mismatch_is_rejected(self):
        header = {
            "protocol": "fastwam-aqlm-loss-pair-v1", "pbs_jobid": "25999999.pbs101",
            "arm": "reconstruction", "updates": 8, "exposures": 80,
            "target_linears": 614, "case_id": 2000, "seed_index": 0,
            "sampler_seed": 910004000, "plan_row": {"index": 2000},
            "counts": {"video_steps": 10, "action_steps": 10},
            "action_shape": [32, 7], "future_shape": [1, 48, 3, 14, 28],
        }
        saved = dict(header, action="cpu-tensor", future="cpu-tensor")
        runtime._validate_anchor_header(header, saved)
        bad = dict(saved, arm="ours")
        with self.assertRaises(RuntimeError):
            runtime._validate_anchor_header(header, bad)
        bad_geometry = dict(saved, future_shape=[1, 48, 3, 14, 27])
        with self.assertRaises(RuntimeError):
            runtime._validate_anchor_header(header, bad_geometry)


if __name__ == "__main__":
    unittest.main()
