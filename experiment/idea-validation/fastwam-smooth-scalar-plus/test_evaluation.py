"""CPU-only checks for bank metadata, parity assignment, and safe resume gates."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aggregate import load_manifest
from evaluate import (
    ARM,
    EXPECTED_EVIDENCE,
    PROTOCOL,
    recover_records,
    validate_install_metadata,
    worker_indices,
)


def fixture_rows():
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    dimensions = (
        "camera_viewpoints", "camera_poses", "lighting", "backgrounds",
        "robot_initial_states", "object_positions", "sensor_noise",
    )
    rows = []
    for suite in suites:
        for dimension in dimensions:
            for repeat in range(10):
                index = len(rows)
                rows.append({
                    "index": index, "variant_id": f"variant-{index:03d}",
                    "suite": suite, "dimension": dimension, "task_id": repeat,
                })
    return rows


def fixture_manifest():
    return {
        "protocol": "fastwam-rotation-baselines-v1",
        "variants": fixture_rows(),
        "shards": [
            {"worker_id": worker, "indices": list(range(worker * 70, (worker + 1) * 70))}
            for worker in range(4)
        ],
    }


def valid_record(row, worker_id, source_bank):
    index = row["index"]
    evidence = {
        "real_quant": True,
        "activation_bits": 4,
        "native_int4_gemm_calls": 1,
        "integer_gemm_calls": 1,
        "native_int4_tensorcore": True,
        "packed_weight_tensor_bytes": 128,
        "bf16_target_weight_absent": True,
        "kv_packed_prefills": 0,
        "kv_layer_reads": 0,
        **EXPECTED_EVIDENCE,
        "source_bank": source_bank,
    }
    return {
        "protocol": PROTOCOL, "complete": True, "arm": ARM,
        "worker_id": worker_id, "index": index,
        "variant_id": row["variant_id"], "suite": row["suite"],
        "dimension": row["dimension"], "task_id": row["task_id"],
        "state_id": 0, "env_seed": 800_000_000 + index * 1000,
        "sampler_seed": 900_000_000 + index * 1000,
        "success": True, "episode_seconds": 1.0,
        "real_quant_evidence": evidence,
        "metrics": [{
            "query_index_in_arm": 0, "first_query_in_arm": True,
            "denoising_latency_ms": 10.0, "full_query_wall_ms": 20.0,
            "peak_allocated_gpu_GB": 16.0, "baseline_allocated_gpu_GB": 15.0,
            "peak_reserved_gpu_GB": 17.0,
        }],
    }


class EvaluationTests(unittest.TestCase):
    def test_workers_partition_all_280_global_indices_by_parity(self):
        even, odd = worker_indices(0), worker_indices(1)
        self.assertEqual(len(even), 140)
        self.assertEqual(len(odd), 140)
        self.assertTrue(all(index % 2 == 0 for index in even))
        self.assertTrue(all(index % 2 == 1 for index in odd))
        self.assertEqual(sorted(even + odd), list(range(280)))

    def test_aggregate_loads_the_unchanged_frozen_cohort(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(fixture_manifest()), encoding="utf-8")
            doc, rows, shards = load_manifest(path)
            self.assertEqual(len(rows), 280)
            self.assertEqual(len(shards), 4)
            self.assertEqual(doc["protocol"], "fastwam-rotation-baselines-v1")

    def test_install_metadata_must_reuse_the_expected_bank_recipe(self):
        with tempfile.TemporaryDirectory() as temporary:
            bank = Path(temporary)
            metadata = {
                "arm": ARM, "module_count": 614, "source_bank": str(bank),
                "source_recipe": "smooth05_hadamard_scalar_a4",
                "smooth_alpha": 0.5, "transform_seed": 20261006,
                "transforms_reused": True, "weight_codes_reused": True,
            }
            checked = validate_install_metadata(metadata, bank)
            self.assertEqual(checked["source_bank"], str(bank.resolve()))
            self.assertTrue(checked["weight_codes_reused"])
            metadata["smooth_alpha"] = 0.75
            with self.assertRaises(ValueError):
                validate_install_metadata(metadata, bank)

    def test_resume_accepts_only_identity_seed_and_native_evidence_match(self):
        rows = fixture_rows()
        with tempfile.TemporaryDirectory() as temporary:
            bank = Path(temporary) / "bank"
            bank.mkdir()
            path = Path(temporary) / "episodes.jsonl"
            path.write_text(json.dumps(valid_record(rows[0], 0, str(bank.resolve()))) + "\n", encoding="utf-8")
            loaded = recover_records(path, rows, worker_indices(0), 0, str(bank.resolve()))
            self.assertEqual(set(loaded), {0})

    def test_resume_rejects_duplicate_indices_and_seed_mismatch(self):
        rows = fixture_rows()
        with tempfile.TemporaryDirectory() as temporary:
            bank = Path(temporary) / "bank"
            bank.mkdir()
            source_bank = str(bank.resolve())
            path = Path(temporary) / "episodes.jsonl"
            record = valid_record(rows[0], 0, source_bank)
            path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate completed slot"):
                recover_records(path, rows, worker_indices(0), 0, source_bank)

            record["sampler_seed"] += 1
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seed mismatch"):
                recover_records(path, rows, worker_indices(0), 0, source_bank)


if __name__ == "__main__":
    unittest.main()
