import unittest
from pathlib import Path
from unittest.mock import patch

import result_extract as extract


ARMS, NEW, REUSED = list(extract.ARMS), list(extract.NEW_ARMS), list(extract.REUSED_ARMS)


def bucket(n):
    arms = {a: {"successes": n // 2, "episodes": n, "expected_episodes": n,
                "success_rate": 0.5} for a in ARMS}
    pairs = {a: {b: {"wins": 1, "losses": 0, "ties": n - 1, "episodes": n,
                    "expected_episodes": n, "success_rate_difference": 1 / n}
                for b in ARMS if b != a} for a in ARMS}
    return {"arms": arms, "paired": pairs}


def fixtures():
    protocol = {
        "protocol": "fixture-v1", "remote_root": "/root", "arms": ARMS,
        "new_arms": NEW, "reused_arms": REUSED,
        "cohort": {"episodes_per_arm": 280, "total_episodes": 1120, "reused_episodes": 560,
                   "comparison_episodes": 1680, "suites": 4, "perturbation_dimensions": 7},
        "performance": {"case_ids": list(range(40)), "seed_indices": [0, 1], "blocks": 3,
                        "repetitions": 3, "latency_scope": "frozen five-NFE scope"},
    }
    evaluation = {
        "protocol": "fixture-v1", "status": "complete", "complete": True,
        "pbs_job_id": "12345.pbs", "new_arms": NEW, "reused_arms": REUSED,
        "expected_worker_count": 8, "node_receipts_complete": True,
        "worker_exitcodes": {w: 0 for w in extract.WORKERS},
        "episode_slots": 1120, "expected_episode_slots": 1120,
        "new_episode_slots": 1120, "expected_new_episode_slots": 1120,
        "new_arms_status": "complete", "new_arm_episodes": {a: 280 for a in NEW},
        "arm_episodes": {a: 280 for a in NEW}, "missing_new_slots_by_arm": {a: [] for a in NEW},
        "missing_slots_by_arm": {a: [] for a in NEW}, "reused_arm_episodes": {a: 280 for a in REUSED},
        "missing_reused_slots_by_arm": {a: [] for a in REUSED},
        "reused_provenance": {a: {"protocols": ["old"], "pbs_job_ids": ["54321.pbs"],
                                  "episodes": 280, "expected_episodes": 280} for a in REUSED},
        "all_methods": {
            "episode_slots": 1680, "expected_episode_slots": 1680,
            "arm_episodes": {a: 280 for a in ARMS}, "missing_slots_by_arm": {a: [] for a in ARMS},
            "provenance_boundary": {"reused_records_relabelled_as_new_engine_evidence": False},
            "overall": bucket(280),
            "suites": {f"s{i}": bucket(70) for i in range(4)},
            "dimensions": {f"d{i}": bucket(40) for i in range(7)},
            "cells": {f"s{s}::d{d}": bucket(10) for s in range(4) for d in range(7)},
            "timing_and_memory": {a: {"episode_wall_ms": {"n": 280}} for a in ARMS},
            "encoding_evidence": {a: None for a in ARMS},
        },
    }
    stats = {"count": 720, "mean": 1.0, "median": 1.0, "p95": 1.1, "max": 1.2}
    performance = {
        "protocol": "fixture-v1", "status": "complete", "reference_arm": "bf16",
        "model": "fixture", "mode": "direct", "hardware": {"gpu": "fixture"},
        "timing_scope": "frozen five-NFE scope", "execution_boundary": "fixture boundary",
        "query_design": {"case_ids": list(range(40)), "seed_indices": [0, 1], "blocks": 3,
                         "repetitions": 3}, "arms": {},
    }
    for arm in ARMS:
        performance["arms"][arm] = {
            "denoising_wall_seconds": dict(stats), "complete_endpoint_query_seconds": dict(stats),
            "peak_allocated_gb_decimal": dict(stats), "packed_storage_bytes": 100,
            "packed_storage_gib_binary": 0.1, "runtime_label": {"native_int4_kernel_used": False},
            "speedup_over_bf16": 1.0,
            "coverage": {"queries": 80, "completed_queries": 80, "samples": 720,
                         "expected_samples": 720, "reused_samples": 0,
                         "by_query": [{"query": i} for i in range(80)]},
        }
    handle = {"stage": "evaluation", "job_name": "cpvq24eval", "remote_root": "/root",
              "job_id": "12345.pbs"}
    launch = {"protocol": "fixture-v1", "pbs_job_id": "12345.pbs", "expected_workers": 8,
              "nodes": [{"rank": 0, "hostname": "hosta"}, {"rank": 1, "hostname": "hostb"}]}
    receipts = [{"protocol": "fixture-v1", "pbs_job_id": "12345.pbs", "node_rank": rank,
                 "host": host, "exit_code": 0,
                 "worker_exitcodes": {f"cosmos-gpu-worker-{i}": 0 for i in range(rank * 4, rank * 4 + 4)}}
                for rank, host in enumerate(("hosta", "hostb"))]
    qstat = {"query_rc": 0, "Job_Name": "cpvq24eval", "job_state": "F", "Exit_status": "0"}
    return protocol, evaluation, performance, handle, qstat, launch, receipts


def build(v=None, marker=True, recovery=None, errors=None, root=None):
    v = v or fixtures()
    return extract.build_summary(*v[:5], marker, *v[5:], "99999.pbs", recovery, errors,
                                 root if recovery is not None else None)


def recovery_fixtures():
    v = fixtures()
    evaluation, launch, receipts, qstat = v[1], v[5], v[6], v[4]
    codes = {worker: (1 if worker.endswith(("-4", "-5", "-7")) else 0) for worker in extract.WORKERS}
    evaluation.update({"status": "failed", "complete": False, "node_receipts_complete": False,
                      "worker_exitcodes": codes})
    receipts[0]["worker_exitcodes"] = {f"cosmos-gpu-worker-{i}": 0 for i in range(4)}
    receipts[0]["exit_code"] = 0
    receipts[1]["worker_exitcodes"] = {f"cosmos-gpu-worker-{i}": codes[f"cosmos-gpu-worker-{i}"]
                                        for i in range(4, 8)}
    receipts[1]["exit_code"] = 1
    qstat["Exit_status"] = "1"
    root = __import__("pathlib").Path("/root")
    lock_error = f"TimeoutError: Timed out waiting for queue lock {root / 'results/evaluation/claim_queue.json.lock'}"
    recovery = {"source_evaluation_job": "12345.pbs", "extractor_pbs_job_id": "99999.pbs",
                "error_type": "queue_lock_timeout", "failed_workers": [4, 5, 7]}
    progress = {i: {"worker_id": i, "run_id": "12345.pbs", "status": "failed", "error": lock_error}
                for i in (4, 5, 7)}
    return v, recovery, progress, root


class ResultExtractTests(unittest.TestCase):
    def test_keeps_all_arms_and_only_ours_pairs_and_drops_query_list(self):
        v = fixtures()
        # Unselected pairs are deliberately irrelevant to the compact report.
        del v[1]["all_methods"]["overall"]["paired"]["bf16"]["ours_vqwam"]
        report = build(v)
        buckets = [report["all_methods"]["overall"], *report["all_methods"]["suites"].values(),
                   *report["all_methods"]["dimensions"].values(), *report["all_methods"]["cells"].values()]
        for item in buckets:
            self.assertEqual(set(item["arms"]), set(ARMS))
            self.assertEqual(set(item["paired"]), {"ours_vqwam"})
            self.assertEqual(set(item["paired"]["ours_vqwam"]), set(ARMS) - {"ours_vqwam"})
        self.assertEqual(report["all_methods"]["overall"]["arms"]["ours_vqwam"]["episodes"], 280)
        self.assertEqual(report["completion"]["new"]["slots"], 1120)
        self.assertEqual(report["completion"]["reused"]["slots"], 560)
        self.assertEqual(report["completion"]["all_methods"]["slots"], 1680)
        self.assertNotIn("by_query", report["performance"]["arms"]["ours_vqwam"]["coverage"])

    def test_completion_and_performance_denominators_are_required_for_every_arm(self):
        for arm in ARMS:
            v = fixtures(); v[1]["all_methods"]["arm_episodes"][arm] = 279
            with self.subTest(arm=arm), self.assertRaisesRegex(ValueError, "280 for every arm"):
                build(v)
            v = fixtures(); v[2]["arms"][arm]["coverage"]["samples"] = 719
            with self.subTest(arm=arm), self.assertRaisesRegex(ValueError, "80 queries/720 samples"):
                build(v)
        v = fixtures(); v[1]["new_episode_slots"] = 1119
        with self.assertRaisesRegex(ValueError, "not 1120"):
            build(v)
        v = fixtures(); v[1]["all_methods"]["episode_slots"] = 1679
        with self.assertRaisesRegex(ValueError, "not 1680"):
            build(v)

    def test_requires_eight_workers_two_nodes_terminal_job_and_marker(self):
        v = fixtures(); v[4]["query_rc"] = 153
        with self.assertRaisesRegex(ValueError, "terminal F"):
            build(v)
        v = fixtures(); v[6].pop()
        with self.assertRaisesRegex(ValueError, "Exactly two node receipts"):
            build(v)
        v = fixtures(); v[1]["worker_exitcodes"]["cosmos-gpu-worker-7"] = 1
        with self.assertRaisesRegex(ValueError, "worker exit codes"):
            build(v)
        v = fixtures(); v[4]["Exit_status"] = "1"
        with self.assertRaisesRegex(ValueError, "F/Exit 0"):
            build(v)
        with self.assertRaisesRegex(ValueError, "PIPELINE_COMPLETE"):
            build(fixtures(), marker=False)

    def test_recovers_only_full_records_with_the_authorized_queue_error(self):
        v, receipt, progress, root = recovery_fixtures()
        report = build(v, marker=False, recovery=receipt, errors=progress, root=root)
        self.assertEqual(report["status"], "complete")
        self.assertFalse(report["source_execution_complete"])
        self.assertEqual(report["source_exit_status"], 1)
        self.assertFalse(report["source_pipeline_complete"])
        self.assertTrue(report["data_complete"])
        self.assertEqual(report["recovery_type"], "queue_control_failure_after_full_record_validation")
        self.assertEqual(report["new_model_forwards"], 0)
        self.assertEqual(report["new_episodes"], 0)
        self.assertEqual(report["completion"]["workers"]["exitcodes"], v[1]["worker_exitcodes"])
        self.assertEqual(report["completion"]["workers"]["exitcodes"]["cosmos-gpu-worker-4"], 1)

    def test_cpu_recovery_entry_passes_raw_progress_to_the_final_gate(self):
        import evaluate
        import aggregate
        v, receipt, progress, root = recovery_fixtures()
        protocol, source, performance, handle, source_qstat, launch, nodes = v
        protocol['cohort']['manifest'] = '/fixture/manifest.json'

        def read(path):
            if path.name == 'summary.json':
                return source
            if path.name == 'evaluation_multinode_launch.json':
                return launch
            if path.name == 'node_receipt.json':
                return nodes[int(path.parent.name.split('_')[-1])]
            return progress[int(path.parent.name.split('_')[-1])]

        with patch.object(extract, 'read_json', side_effect=read), \
             patch.object(extract, 'qstat', return_value=(source_qstat, '')), \
             patch.object(Path, 'exists', return_value=False), \
             patch.object(evaluate, 'validate_protocol', side_effect=lambda p: p), \
             patch.object(evaluate, 'load_manifest', return_value=({}, [])), \
             patch.object(evaluate, 'load_completed', return_value=dict.fromkeys(range(1120))), \
             patch.object(evaluate, 'load_reused_records', return_value=dict.fromkeys(range(560))), \
             patch.object(extract, 'gripper_diagnostics', return_value={'scope': 'fixture'}), \
             patch.object(aggregate, 'summarize', return_value=source):
            report = extract.recover_evaluation(root, protocol, handle, performance, '99999.pbs', receipt)
        self.assertFalse(report['source_execution_complete'])
        self.assertEqual(set(report['source_worker_errors']), {'4', '5', '7'})

    def test_gripper_continuous_error_and_sign_commands_use_paired_first_chunks(self):
        import math
        records = {}
        base = [[0.0] * 6 + [-1.0] for _ in range(16)]
        for arm in ARMS:
            for index in range(280):
                records[arm, index] = {'variant_id': f'v{index}', 'suite': 's', 'dimension': 'd',
                    'task_id': index, 'state_id': 0, 'env_seed': index, 'sampler_seed': index,
                    'first_action_chunk': base}
        changed = [row[:] for row in base]
        changed[0][6], changed[1][6] = 1.0, 0.0
        records['ours_vqwam', 0]['first_action_chunk'] = changed
        result = extract.gripper_diagnostics(records)['arms']
        self.assertEqual(result['bf16']['mae'], 0)
        ours = result['ours_vqwam']
        self.assertEqual(ours['generated_gripper_values'], 4480)
        self.assertAlmostEqual(ours['mae'], 3 / 4480)
        self.assertAlmostEqual(ours['rmse'], math.sqrt(5 / 4480))
        self.assertEqual((ours['sign_disagreements'], ours['chunks_with_sign_disagreement']), (2, 1))
        records['ours_vqwam', 0]['env_seed'] = -1
        with self.assertRaisesRegex(ValueError, 'Unpaired first chunk'):
            extract.gripper_diagnostics(records)

    def test_recovery_rejects_missing_episode_unrelated_error_or_wrong_source(self):
        v, receipt, progress, root = recovery_fixtures()
        receipt['failed_workers'] = [4, 5]
        with self.assertRaisesRegex(ValueError, "worker set differs"):
            build(v, marker=False, recovery=receipt, errors=progress, root=root)
        v, receipt, progress, root = recovery_fixtures()
        v[1]["new_episode_slots"] = 1119
        with self.assertRaisesRegex(ValueError, "not 1120"):
            build(v, marker=False, recovery=receipt, errors=progress, root=root)
        v, receipt, progress, root = recovery_fixtures()
        progress[4]["error"] = "RuntimeError: unrelated failure"
        with self.assertRaisesRegex(ValueError, "authorized queue-lock timeout"):
            build(v, marker=False, recovery=receipt, errors=progress, root=root)
        v, receipt, progress, root = recovery_fixtures()
        v[4]["Exit_status"] = "0"
        with self.assertRaisesRegex(ValueError, "F/Exit 1"):
            build(v, marker=False, recovery=receipt, errors=progress, root=root)
        v, receipt, progress, root = recovery_fixtures()
        v[1]["pbs_job_id"] = "99999.pbs"
        with self.assertRaisesRegex(ValueError, "current handle"):
            build(v, marker=False, recovery=receipt, errors=progress, root=root)


if __name__ == "__main__":
    unittest.main()
