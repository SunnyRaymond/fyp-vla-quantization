"""Small standard-library checks for scheduling, durable resume, and duplicate rejection."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from cosmos_runtime import (_config_file_relative_to_repo, _episode_is_complete,
                            _defer_registration_checkpoint_downloads,
                            _finalize_termination)
from calibrate import calibration_embedding_cache

from evaluate import (ARMS, PROTOCOL, atomic_episode, build_bundles, claim_from_state,
                      DIMENSIONS, environment_seed, initial_queue, load_completed, pending_slots,
                      sampler_seed, SUITES, worker_completed_slots)

HERE = Path(__file__).resolve().parent
PROTOCOL_DOC = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))


class RuntimeBoundaryTests(unittest.TestCase):
    def test_cosmos_loader_config_path_is_repo_relative(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            config = repo / "cosmos_policy" / "config" / "config.py"
            config.parent.mkdir(parents=True)
            config.touch()
            loader_path = _config_file_relative_to_repo(repo)
        self.assertEqual(loader_path, "cosmos_policy/config/config.py")
        self.assertFalse(Path(loader_path).is_absolute())
        self.assertEqual(loader_path.replace("/", ".").removesuffix(".py"),
                         "cosmos_policy.config.config")

    def test_only_registration_default_checkpoints_are_deferred(self):
        from types import SimpleNamespace

        calls = []

        def original_resolver(uri):
            calls.append(uri)
            return f"resolved:{uri}"

        checkpoint_db = SimpleNamespace(get_checkpoint_path=original_resolver)
        deferred_resolver = None
        default_uri = "hf://nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt"
        aloha_uri = "hf://nvidia/Cosmos-Policy-ALOHA-Predict2-2B/Cosmos-Policy-ALOHA-Predict2-2B.pt"
        tokenizer_uri = "hf://nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth"
        with _defer_registration_checkpoint_downloads(checkpoint_db):
            deferred_resolver = checkpoint_db.get_checkpoint_path
            self.assertEqual(checkpoint_db.get_checkpoint_path(default_uri), default_uri)
            self.assertEqual(checkpoint_db.get_checkpoint_path(aloha_uri), aloha_uri)
            self.assertEqual(checkpoint_db.get_checkpoint_path(tokenizer_uri),
                             f"resolved:{tokenizer_uri}")
        self.assertIs(checkpoint_db.get_checkpoint_path, original_resolver)
        self.assertEqual(deferred_resolver(default_uri), f"resolved:{default_uri}")
        self.assertEqual(calls, [tokenizer_uri, default_uri])

    def test_c8_cache_does_not_require_closed_loop_or_plus_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "data" / "inputs"
            inputs.mkdir(parents=True)
            cache = root / "official.pkl"
            cache.touch()
            (inputs / "case_2000.json").write_text(json.dumps({"description": "original instruction"}))
            protocol = {"calibration": {"source_root": str(root), "case_ids": [2000]}}
            requirement = {"source_cache": str(cache), "missing": ["held-out Plus instruction"]}
            path = root / "language_requirements.json"
            path.write_text(json.dumps(requirement))
            self.assertEqual(calibration_embedding_cache(protocol, root), cache)
            self.assertFalse((root / "results" / "text_embeddings.pkl").exists())
            requirement["missing"].append("original instruction")
            path.write_text(json.dumps(requirement))
            with self.assertRaisesRegex(RuntimeError, "C8 instruction embedding is missing"):
                calibration_embedding_cache(protocol, root)

    def test_environment_failure_is_complete_but_deadline_is_not(self):
        termination = _finalize_termination(False, 'environment_terminated', 10, 400)
        self.assertEqual(termination, 'environment_terminated')
        self.assertTrue(_episode_is_complete(termination, 10, 40, 1))
        self.assertFalse(_episode_is_complete('soft_deadline', 10, 40, 1))
        self.assertFalse(_episode_is_complete(termination, 0, 30, 0))
        self.assertFalse(_episode_is_complete(termination, 10, 9, 1))


def fixture_rows():
    rows = []
    for suite in sorted(SUITES):
        for dimension in sorted(DIMENSIONS):
            for _ in range(10):
                index = len(rows)
                rows.append({"index": index, "variant_id": f"fixture-{index}",
                             "suite": suite, "dimension": dimension,
                             "task_id": index, "state_id": 0})
    return rows


def fixture_record(row, arm="bf16", worker_id=0):
    index = row["index"]
    evidence = {
        "arm": arm,
        "execution": "bf16_reference" if arm == "bf16" else "decoded_bf16_fake_a4",
        "execution_scope": "decoded_bf16_numerical_comparison",
        "native_int4_kernel_used": False,
        "weight_format": ("bf16" if arm == "bf16" else
                           "packed_int4_group128_bf16_scales" if arm == "spinquant_w4a4" else
                           "m2_k256_d4_bf16_books_b32_uint8_exponents"),
        "activation_format": "bf16" if arm == "bf16" else "dynamic_symmetric_int4_per_token_qmax7_ste",
        "target_module_count": 10,
        "target_weight_numel": 1000,
        "same_module_effective_bpw": None if arm == "bf16" else 4.0,
        "whole_dit_effective_bpw": None if arm == "bf16" else 4.5,
        "artifact_dir": None if arm == "bf16" else "/fixture/spinquant",
        "artifact_manifest": None if arm == "bf16" else "/fixture/spinquant/deployment.pt",
    }
    if arm != "bf16":
        evidence.update({"transform": "learned_shared_cayley_r128" if arm == "spinquant_w4a4"
                         else "smooth05_batched_h128_v2", "bpw_match_relative_delta": 0.01})
    return {
        "protocol": PROTOCOL, "arm": arm, "worker_id": worker_id, "complete": True,
        "index": index, "variant_id": row["variant_id"], "suite": row["suite"],
        "dimension": row["dimension"], "task_id": row["task_id"], "state_id": 0,
        "env_seed": environment_seed(index, PROTOCOL_DOC),
        "sampler_seed": sampler_seed(index, PROTOCOL_DOC), "pbs_job_id": "pbs-fixture", "success": False,
        "episode_seconds": 1.0, "replans": 1,
        "query_metrics": [{"seed": sampler_seed(index, PROTOCOL_DOC, 0),
                            "first_query": True, "query_seconds": 0.5,
                            "denoising_seconds": None, "peak_allocated_gb": 4.0,
                            "peak_reserved_gb": 5.0, "first_query_in_run": True,
                            "query_index_in_worker_arm": 0,
                            "first_query_in_worker_arm": True, "pbs_job_id": "pbs-fixture",
                            "worker_process_id": "pbs-fixture/worker_0/pid_1"}],
        "quantization_evidence": evidence,
    }


class EvaluationTests(unittest.TestCase):
    def test_resume_progress_counts_each_saved_slot_once_across_workers(self):
        completed = {
            ("bf16", 0): {"worker_id": 0, "pbs_job_id": "old"},
            ("bf16", 1): {"worker_id": 1, "pbs_job_id": "old"},
            ("spinquant_w4a4", 0): {"worker_id": 0, "pbs_job_id": "new"},
            ("action_latent_micro_vq_a4", 1): {"worker_id": 2, "pbs_job_id": "new"},
        }
        counts = [worker_completed_slots(completed, worker) for worker in range(4)]
        self.assertEqual(counts, [2, 1, 1, 0])
        self.assertEqual(sum(counts), len(completed))

    def setUp(self):
        self.rows = fixture_rows()
        self.bundles = build_bundles(self.rows, PROTOCOL_DOC, 20261008)

    def test_dynamic_claims_cover_each_cell_once_and_balance_long_suite_first(self):
        self.assertEqual(len(self.bundles), 28)
        self.assertEqual(sum(len(bundle["indices"]) for bundle in self.bundles), 280)
        self.assertEqual(sorted(index for bundle in self.bundles for index in bundle["indices"]),
                         list(range(280)))
        expected_slots = {(arm, index) for arm in ARMS for index in range(280)}
        self.assertEqual({slot for bundle in self.bundles for slot in pending_slots(bundle, set())},
                         expected_slots)
        self.assertEqual(self.bundles[0]["suite"], "libero_10")
        state = initial_queue(self.bundles, "job-test")
        claimed = []
        completed = set()
        for turn in range(len(self.bundles)):
            worker = turn % 4
            bundle, done = claim_from_state(state, self.bundles, worker, "job-test", "host",
                                            1000 + worker, completed)
            self.assertFalse(done)
            self.assertIsNotNone(bundle)
            claimed.append(bundle["bundle_id"])
            del state["claims"][bundle["bundle_id"]]
            completed.add(bundle["bundle_id"])
        self.assertEqual(len(set(claimed)), 28)
        bundle, done = claim_from_state(state, self.bundles, 0, "job-test", "host", 1000, completed)
        self.assertIsNone(bundle)
        self.assertTrue(done)

    def test_resume_loads_atomic_episode_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            row = self.rows[7]
            path = Path(directory) / "episodes" / "bf16" / "007.json"
            atomic_episode(path, fixture_record(row))
            records = load_completed(directory, self.rows, PROTOCOL_DOC)
        self.assertEqual(set(records), {("bf16", 7)})

    def test_duplicate_slot_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "episodes" / "bf16"
            root.mkdir(parents=True)
            record = fixture_record(self.rows[0])
            (root / "000.json").write_text(json.dumps(record), encoding="utf-8")
            (root / "001.json").write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate episode slot"):
                load_completed(directory, self.rows, PROTOCOL_DOC)


if __name__ == "__main__":
    unittest.main()
