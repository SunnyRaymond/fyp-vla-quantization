"""CPU-only stdlib checks for the VQ-plus worker and record control logic."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate import (  # noqa: E402
    ARMS,
    COHORT_PROTOCOL,
    PROTOCOL,
    load_manifest,
    recover_records,
    validate_record,
    worker_arm,
    worker_indices,
)


def fixture_manifest():
    variants = []
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        for dimension in ("d0", "d1", "d2", "d3", "d4", "d5", "d6"):
            for variant in range(10):
                index = len(variants)
                variants.append({
                    "index": index,
                    "variant_id": f"variant-{index}",
                    "suite": suite,
                    "dimension": dimension,
                    "task_id": variant,
                    "state_id": 0,
                })
    return {"protocol": COHORT_PROTOCOL, "variants": variants}


def valid_record(rows, index, worker_id, query_index, bank="frozen-bank"):
    row = rows[index]
    arm = worker_arm(worker_id)
    return {
        "protocol": PROTOCOL,
        "arm": arm,
        "worker_id": worker_id,
        "complete": True,
        "index": index,
        "variant_id": row["variant_id"],
        "suite": row["suite"],
        "dimension": row["dimension"],
        "task_id": row["task_id"],
        "state_id": 0,
        "env_seed": 800_000_000 + index * 1000,
        "sampler_seed": 900_000_000 + index * 1000,
        "success": False,
        "episode_seconds": 1.0,
        "vq_evidence": {
            "arm": arm,
            "source_bank": bank,
            "module_count": 614,
            "observed_module_count": 614,
            "execution": "decoded_bf16_vq",
            "activation_bits": 4,
            "kv_cache_dtype": "bf16",
            "native_low_bit_kernel": False,
            "frozen_bank_reused": True,
            "native_int4_gemm_calls": 0,
            "kv_packed_prefills": 0,
            "kv_layer_reads": 0,
            "quantized_linear_calls": 614,
        },
        "metrics": [{
            "query_index_in_arm": query_index,
            "first_query_in_arm": query_index == 0,
            "denoising_latency_ms": 0.0,
            "full_query_wall_ms": 0.0,
            "peak_allocated_gpu_GB": 0.0,
            "baseline_allocated_gpu_GB": 0.0,
            "peak_reserved_gpu_GB": 0.0,
        }],
    }


class EvaluationControlTests(unittest.TestCase):
    def setUp(self):
        self.manifest = fixture_manifest()
        self.rows = self.manifest["variants"]

    def test_four_workers_partition_the_280_paired_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(self.manifest), encoding="utf-8")
            _, rows = load_manifest(path)

        self.assertEqual(len(rows), 280)
        self.assertEqual(ARMS, ("row_scale_vq_a4", "micro_scale_vq_a4"))
        assigned = {arm: [] for arm in ARMS}
        for worker_id in range(4):
            indices = worker_indices(worker_id)
            arm = worker_arm(worker_id)
            assigned[arm].extend(indices)
            self.assertEqual(arm, ARMS[worker_id // 2])
            self.assertEqual(len(indices), 140)
            self.assertEqual(set(indices), set(range(worker_id % 2, 280, 2)))
            parity_cells = Counter(
                (rows[i]["suite"], rows[i]["dimension"], i % 2) for i in indices
            )
            self.assertEqual(len(parity_cells), 28)
            self.assertEqual(set(parity_cells.values()), {5})
        for arm in ARMS:
            self.assertEqual(sorted(assigned[arm]), list(range(280)))

    def test_valid_decoded_vq_record_and_rejects_native_claim_or_wrong_seed(self):
        record = valid_record(self.rows, index=0, worker_id=0, query_index=0)
        self.assertEqual(validate_record(record, self.rows, 0, "frozen-bank"), 0)

        mislabeled_native = json.loads(json.dumps(record))
        mislabeled_native["vq_evidence"].update(
            execution="native_int4", native_low_bit_kernel=True,
            native_int4_gemm_calls=614,
        )
        with self.assertRaises(ValueError):
            validate_record(mislabeled_native, self.rows, 0, "frozen-bank")

        wrong_seed = json.loads(json.dumps(record))
        wrong_seed["sampler_seed"] += 1
        with self.assertRaises(ValueError):
            validate_record(wrong_seed, self.rows, 0, "frozen-bank")

    def test_resume_prefix_and_truncate_torn_jsonl_tail(self):
        indices = worker_indices(0)
        first = valid_record(self.rows, indices[0], 0, query_index=0)
        second = valid_record(self.rows, indices[1], 0, query_index=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.jsonl"
            path.write_bytes(
                (json.dumps(first) + "\n" + json.dumps(second) + "\n").encode("utf-8")
                + b'{"index":'
            )

            recovered = recover_records(path, self.rows, 0, "frozen-bank")

            self.assertEqual(list(recovered), indices[:2])
            self.assertEqual(next(i for i in indices if i not in recovered), indices[2])
            saved_lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(saved_lines), 2)
            self.assertEqual([json.loads(line)["index"] for line in saved_lines], indices[:2])


if __name__ == "__main__":
    unittest.main()
