"""Standard-library cohort and durable-record tests; no Torch/model import."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import evaluate
import runtime


ROOT = Path(__file__).parent


def bound_protocol():
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    job = "25999999.pbs101"
    protocol["sources"]["parent_pair_job"] = job
    protocol["sources"]["parent_pair_artifacts"] = "/scratch/users/ntu/aqlm-pair/artifacts/25999999.pbs101"
    protocol["sources"]["parent_protocol_file"] = protocol["sources"]["parent_pair_artifacts"]+"/protocol.json"
    for arm in protocol["arms"]:
        protocol["arm_sources"][arm]["source_job"] = job
        protocol["arm_sources"][arm]["source_protocol_file"] = protocol["sources"]["parent_protocol_file"]
    return protocol


def manifest_rows():
    rows = []
    for suite_index, suite in enumerate(("s0", "s1", "s2", "s3")):
        for dimension_index, dimension in enumerate(("d0", "d1", "d2", "d3", "d4", "d5", "d6")):
            for cell_index in range(10):
                index = len(rows)
                rows.append({
                    "index": index, "variant_id": f"v{index:03d}",
                    "task_id": 1000 + index, "state_id": 0,
                    "suite": suite, "dimension": dimension,
                })
    return rows


def metadata(protocol, arm):
    value = runtime.expected_install_metadata(protocol, arm)
    value.update(orthogonality={"action": 0.0, "video": 0.0},
                 rotation_movement={"action": 1.0, "video": 1.0})
    return value


def record(protocol, rows, arm, worker, index):
    meta = metadata(protocol, arm)
    job = "27000000.pbs101"
    evidence = dict(meta, observed_module_count=614, native_int4_gemm_calls=0,
                    kv_packed_prefills=0, kv_layer_reads=0, quantized_linear_calls=1)
    metric = {
        "query_index_in_arm": 0, "first_query_in_arm": True,
        "first_query_in_run": True, "replan": 0,
        "sampler_seed": evaluate.sampler_seed(index, 0),
        "transform_implementation": meta["transform_implementation"],
        "pbs_job_id": job,
    }
    for field in evaluate.TIMING_FIELDS:
        metric[field] = 0.0
    row = rows[index]
    return {
        "protocol": evaluate.PROTOCOL, "arm": arm, "worker_id": worker,
        "index": index, "complete": True, "variant_id": row["variant_id"],
        "suite": row["suite"], "dimension": row["dimension"],
        "task_id": row["task_id"], "state_id": 0,
        "env_seed": evaluate.environment_seed(index),
        "sampler_seed": evaluate.sampler_seed(index), "success": True,
        "teacher_queries": 0, "new_training_updates": 0,
        "transform_implementation": meta["transform_implementation"],
        "pbs_job_id": job, "vq_evidence": evidence,
        "episode_seconds": 1.0, "replans": 1, "metrics": [metric],
    }


class EvaluationContractTests(unittest.TestCase):
    def test_shards_and_paired_seeds(self):
        self.assertEqual(evaluate.worker_arm(0), "reconstruction")
        self.assertEqual(evaluate.worker_arm(2), "ours")
        self.assertEqual(evaluate.worker_indices(0), list(range(0, 280, 2)))
        self.assertEqual(evaluate.worker_indices(3), list(range(1, 280, 2)))
        self.assertEqual(evaluate.environment_seed(17), 800017000)
        self.assertEqual(evaluate.sampler_seed(17, 3), 900017003)

    def test_original_manifest_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({
                "protocol": evaluate.COHORT_PROTOCOL, "variants": manifest_rows(),
            }), encoding="utf-8")
            _, rows = evaluate.load_manifest(path)
            self.assertEqual(len(rows), 280)
            self.assertEqual(rows[0]["state_id"], 0)
            self.assertEqual(rows[-1]["index"], 279)

    def test_record_and_prefix_recovery(self):
        protocol = bound_protocol()
        rows = manifest_rows()
        arm, worker = "reconstruction", 0
        install = metadata(protocol, arm)
        first = record(protocol, rows, arm, worker, 0)
        self.assertEqual(evaluate.validate_record(first, rows, worker, protocol, install, arm), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.jsonl"
            path.write_text(json.dumps(first) + "\n", encoding="utf-8")
            found = evaluate.recover_records(path, rows, worker, protocol, install, arm)
            self.assertEqual(list(found), [0])
            bad = copy.deepcopy(first)
            bad["index"] = 2
            with self.assertRaises(ValueError):
                evaluate.validate_record(bad, rows, worker, protocol, install, arm)


if __name__ == "__main__":
    unittest.main()
