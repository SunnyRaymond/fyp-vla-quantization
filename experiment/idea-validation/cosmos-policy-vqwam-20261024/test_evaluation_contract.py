"""CPU-only contract tests; these do not import or run torch/model code."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import aggregate
import evaluate
import evaluation_multinode

HERE = Path(__file__).resolve().parent
PROTOCOL_DOC = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
DIMENSIONS = ("camera_viewpoints", "objects_layout", "robot_initial_states",
              "language_instructions", "light_conditions", "background_textures", "sensor_noise")


def fixture_rows():
    rows = []
    for suite in SUITES:
        for dimension in DIMENSIONS:
            for _ in range(10):
                index = len(rows)
                rows.append({"index": index, "variant_id": f"fixture-{index}",
                             "suite": suite, "dimension": dimension,
                             "task_id": index, "state_id": 0})
    return rows


class EvaluationContractTests(unittest.TestCase):
    def _profile(self, workers):
        value = json.loads(json.dumps(PROTOCOL_DOC))
        resources = value["resources"]["evaluation"]
        if workers == 4:
            resources.update({"gpus": 4, "ncpus": 64, "memory_gb": 440, "nodes": 1,
                              "gpus_per_node": 4, "ncpus_per_node": 64,
                              "memory_gb_per_node": 440, "required_execution_queue": "g3"})
            value["load_balancing"]["persistent_workers"] = 4
            value.pop("stages", None)
        else:
            resources.update({"gpus": 8, "ncpus": 128, "memory_gb": 880, "nodes": 2,
                              "gpus_per_node": 4, "ncpus_per_node": 64,
                              "memory_gb_per_node": 440, "required_execution_queue": "g4"})
            value["load_balancing"]["persistent_workers"] = 8
            value["stages"] = {"evaluation": {"job_name": "cpvq24eval", "script": "evaluation_g4.pbs"}}
        return value

    def test_frozen_protocol_has_exactly_four_new_and_two_reused_arms(self):
        value = evaluate.validate_protocol(PROTOCOL_DOC)
        self.assertEqual(value["arms"], [
            "bf16", "spinquant_w4a4", "quarot_w4a4", "svdquant_w4a4",
            "viditq_w4a4", "ours_vqwam",
        ])
        self.assertEqual(value["cohort"]["comparison_episodes"], 1680)
        changed = dict(value)
        changed["arms"] = [*value["arms"], "action_latent_A4"]
        with self.assertRaisesRegex(ValueError, "frozen four-new-arm"):
            evaluate.validate_protocol(changed)
        changed = json.loads(json.dumps(value))
        changed["cohort"]["seed_stride"] += 1
        with self.assertRaisesRegex(ValueError, "frozen four-new-arm"):
            evaluate.validate_protocol(changed)

    def test_exact_g3_four_worker_and_two_node_g4_eight_worker_profiles_only(self):
        self.assertEqual(evaluate.expected_worker_count(self._profile(4)), 4)
        self.assertEqual(evaluate.expected_worker_count(self._profile(8)), 8)
        self.assertEqual(evaluate.validate_protocol(self._profile(8))["load_balancing"]["persistent_workers"], 8)
        changed = self._profile(8)
        changed["resources"]["evaluation"]["required_execution_queue"] = "g3"
        with self.assertRaisesRegex(ValueError, "exact 4-GPU/g3 or 8-GPU/two-node g4"):
            evaluate.validate_protocol(changed)
        changed = self._profile(8)
        changed["resources"]["evaluation"]["gpus_per_node"] = 5
        with self.assertRaisesRegex(ValueError, "exact 4-GPU/g3 or 8-GPU/two-node g4"):
            evaluate.validate_protocol(changed)

    def test_visible_device_contract_requires_real_unique_allocation_devices(self):
        self.assertEqual(evaluate.visible_cuda_devices("GPU-a,GPU-b,GPU-c,GPU-d", 4),
                         ["GPU-a", "GPU-b", "GPU-c", "GPU-d"])
        with self.assertRaisesRegex(RuntimeError, "must be provided"):
            evaluate.visible_cuda_devices(None, 4)
        with self.assertRaisesRegex(RuntimeError, "Expected 4 unique"):
            evaluate.visible_cuda_devices("0,1,1,3", 4)

    def test_worker_four_id_and_query_timing_sequence_are_valid_only_for_eight_worker_profile(self):
        records = {("ours_vqwam", 0): {"worker_id": 4, "query_metrics": [{
            "query_index_in_worker_arm": 0, "first_query_in_run": True,
            "worker_process_id": "job/worker_4/pid_1",
        }]}}
        evaluate._validate_query_sequences(records, 8)
        with self.assertRaisesRegex(ValueError, "outside the 4-worker allocation"):
            evaluate._validate_query_sequences(records, 4)

    def test_frozen_manifest_helpers_and_paired_seeds_are_reused(self):
        rows = fixture_rows()
        manifest = {"protocol": PROTOCOL_DOC["cohort"]["manifest_protocol"],
                    "seed": 20261008, "variants": rows}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            _, loaded = evaluate.load_manifest(path, PROTOCOL_DOC)
        self.assertEqual(loaded, rows)
        bundles = evaluate.build_bundles(rows, PROTOCOL_DOC)
        self.assertEqual(len(bundles), 28)
        self.assertEqual(sorted(index for bundle in bundles for index in bundle["indices"]), list(range(280)))
        self.assertEqual(evaluate.environment_seed(7, PROTOCOL_DOC), 800_007_000)
        self.assertEqual(evaluate.sampler_seed(7, PROTOCOL_DOC, 3), 900_007_003)

    def test_method_specific_evidence_accepts_baseline_and_ours_adaptation_boundaries(self):
        baseline = evaluate.normalize_arm_evidence({
            "arm": "quarot_w4a4", "target_linears": 280,
            "target_weight_numel": evaluate.EXPECTED_WEIGHT_NUMEL,
            "effective_bpw": 4.30, "encoded_target_bpw": 4.30,
            "execution": "decoded_bf16_fake_a4_eager", "artifact_manifest": "/bank/manifest.json",
        }, "quarot_w4a4", "/task", PROTOCOL_DOC)
        self.assertFalse(baseline["native_int4_kernel_used"])
        self.assertEqual(baseline["artifact_dir"], "/bank")
        self.assertEqual(baseline["effective_bpw"], 4.30)

        ours = evaluate.normalize_arm_evidence({
            "arm": "ours_vqwam", "target_module_count": 280,
            "target_weight_numel": evaluate.EXPECTED_WEIGHT_NUMEL,
            "artifact_dir": "/task/results/ours/deployment",
            "execution_scope": "packed GPU resident decoded-BF16 GEMM",
            "native_int4_kernel_used": False, "packed_gpu_resident": True,
            "parity_status": "validated_full80",
            "cuda_graph": {"status": "passed", "graph_captures": 12},
            "export_storage": {"target_effective_bpw": 4.29},
        }, "ours_vqwam", "/task", PROTOCOL_DOC)
        self.assertEqual(ours["artifact_manifest"], "/task/results/ours/deployment/manifest.json")
        self.assertEqual(ours["execution"], "packed_gpu_resident_decoded_bf16_gemm")

    def test_graph_counters_are_not_treated_as_arm_encoding_differences(self):
        evidence = {"arm": "ours_vqwam", "execution": "decoded_bf16_gemm",
                    "execution_scope": "packed GPU resident", "native_int4_kernel_used": False,
                    "target_module_count": 280, "target_weight_numel": evaluate.EXPECTED_WEIGHT_NUMEL,
                    "effective_bpw": 4.29, "artifact_dir": "/bank", "artifact_manifest": "/bank/manifest.json",
                    "packed_gpu_resident": True, "cuda_graph": {"graph_replays": 1}}
        evidence['parity_status'] = 'validated_full80'
        records = {("ours_vqwam", 0): {"quantization_evidence": evidence}}
        changed = dict(evidence, cuda_graph={"graph_replays": 200})
        records[("ours_vqwam", 1)] = {"quantization_evidence": changed}
        canonical = aggregate._encoding(records, ("ours_vqwam",))["ours_vqwam"]
        self.assertNotIn("cuda_graph", canonical)
        self.assertEqual(canonical["effective_bpw"], 4.29)

    def test_six_arm_summary_keeps_full_denominators_and_reused_provenance(self):
        rows = fixture_rows()
        reused = {("bf16", 0): {"protocol": "cosmos-policy-c8-w4a4-v1",
                                 "pbs_job_id": "25836774.pbs101", "success": False,
                                 "episode_seconds": 1.0, "query_metrics": [],
                                 "quantization_evidence": {"arm": "bf16", "execution": "bf16_reference"}}}
        exits = {f"cosmos-gpu-worker-{index}": 0 for index in range(4)}
        summary = aggregate.summarize(PROTOCOL_DOC, rows, {}, reused, exits, "eval-job.pbs101")
        self.assertEqual(summary["status"], "checkpointed")
        self.assertEqual(summary["pbs_job_id"], "eval-job.pbs101")
        self.assertEqual(summary["expected_new_episode_slots"], 1120)
        self.assertEqual(summary["all_methods"]["expected_episode_slots"], 1680)
        self.assertEqual(summary["all_methods"]["overall"]["arms"]["bf16"]["expected_episodes"], 280)
        pair = summary["all_methods"]["overall"]["paired"]["quarot_w4a4"]["bf16"]
        self.assertEqual((pair["episodes"], pair["expected_episodes"]), (0, 280))
        self.assertEqual(summary["reused_provenance"]["bf16"]["pbs_job_ids"], ["25836774.pbs101"])
        self.assertFalse(summary["all_methods"]["provenance_boundary"][
            "reused_records_relabelled_as_new_engine_evidence"])

    def test_complete_requires_all_six_full_cohorts_and_four_zero_workers(self):
        protocol = self._profile(4)
        rows = fixture_rows()
        new, reused = {}, {}
        for arm in evaluate.NEW_ARMS:
            evidence = {"arm": arm, "execution_scope": "decoded BF16 boundary",
                        "native_int4_kernel_used": False}
            for row in rows:
                new[arm, row["index"]] = {"success": False, "episode_seconds": 1.0,
                    "query_metrics": [], "quantization_evidence": evidence}
        for arm in evaluate.REUSED_ARMS:
            evidence = {"arm": arm, "execution_scope": "original reused evidence"}
            for row in rows:
                reused[arm, row["index"]] = {"protocol": "cosmos-policy-c8-w4a4-v1",
                    "pbs_job_id": f"old-{arm}.pbs101", "success": False,
                    "episode_seconds": 1.0, "query_metrics": [], "quantization_evidence": evidence}
        exits = {f"cosmos-gpu-worker-{index}": 0 for index in range(4)}
        summary = aggregate.summarize(protocol, rows, new, reused, exits, "new-job.pbs101")
        self.assertEqual(summary["status"], "complete")
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["new_episode_slots"], 1120)
        self.assertEqual(summary["all_methods"]["episode_slots"], 1680)
        exits["cosmos-gpu-worker-2"] = 1
        failed = aggregate.summarize(protocol, rows, new, reused, exits, "new-job.pbs101")
        self.assertEqual(failed["status"], "failed")

    def test_eight_worker_profile_completes_same_full_cohort_and_requires_all_eight_exits(self):
        protocol = self._profile(8)
        rows = fixture_rows()
        new, reused = {}, {}
        for arm in evaluate.NEW_ARMS:
            evidence = {"arm": arm, "execution_scope": "decoded BF16 boundary",
                        "native_int4_kernel_used": False}
            for row in rows:
                new[arm, row["index"]] = {"success": False, "episode_seconds": 1.0,
                    "query_metrics": [], "worker_id": row["index"] % 8,
                    "quantization_evidence": evidence}
        for arm in evaluate.REUSED_ARMS:
            evidence = {"arm": arm, "execution_scope": "original reused evidence"}
            for row in rows:
                reused[arm, row["index"]] = {"protocol": "cosmos-policy-c8-w4a4-v1",
                    "pbs_job_id": f"old-{arm}.pbs101", "success": False,
                    "episode_seconds": 1.0, "query_metrics": [],
                    "quantization_evidence": evidence}
        exits = {f"cosmos-gpu-worker-{index}": 0 for index in range(8)}
        summary = aggregate.summarize(protocol, rows, new, reused, exits, "new-g4-job.pbs101")
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["expected_worker_count"], 8)
        self.assertEqual(summary["new_episode_slots"], 1120)
        self.assertEqual(summary["expected_new_episode_slots"], 1120)
        self.assertEqual(summary["all_methods"]["episode_slots"], 1680)
        self.assertEqual(summary["all_methods"]["expected_episode_slots"], 1680)
        self.assertEqual(set(summary["all_methods"]["arm_episodes"].values()), {280})

        missing_worker = dict(exits)
        missing_worker.pop("cosmos-gpu-worker-7")
        incomplete = aggregate.summarize(protocol, rows, new, reused, missing_worker, "new-g4-job.pbs101")
        self.assertEqual(incomplete["status"], "checkpointed")
        failing_worker = dict(exits, **{"cosmos-gpu-worker-7": 1})
        failed = aggregate.summarize(protocol, rows, new, reused, failing_worker, "new-g4-job.pbs101")
        self.assertEqual(failed["status"], "failed")

    def test_multinode_mapping_receipts_and_native_pbsdsh_command(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory)
            launch = {"protocol": evaluate.PROTOCOL, "pbs_job_id": "999.pbs101",
                      "artifacts_root": str(artifacts),
                      "nodes": [{"rank": 0, "hostname": "node-a"},
                                {"rank": 1, "hostname": "node-b"}],
                      "node_script": "/snapshot/evaluation_node.sh"}
            command = evaluation_multinode.pbsdsh_command(launch)
            self.assertEqual(command[:5], ["/opt/pbs/bin/pbsdsh", "--", "/bin/bash", "-l",
                                           "/snapshot/evaluation_node.sh"])
            self.assertNotIn("ssh", " ".join(command).lower())
            for rank, host in ((0, "node-a"), (1, "node-b")):
                workers = {f"cosmos-gpu-worker-{worker}": 0
                           for worker in range(rank * 4, rank * 4 + 4)}
                node_dir = artifacts / "nodes" / f"node_{rank}"
                node_dir.mkdir(parents=True)
                (node_dir / "node_receipt.json").write_text(json.dumps({
                    "protocol": evaluate.PROTOCOL, "pbs_job_id": "999.pbs101",
                    "node_rank": rank, "host": host, "exit_code": 0,
                    "worker_exitcodes": workers,
                }), encoding="utf-8")
            workers, receipts, complete, errors = evaluation_multinode.collect_node_receipts(launch, 0)
            self.assertTrue(complete, errors)
            self.assertEqual(len(receipts), 2)
            self.assertEqual(set(workers), {f"cosmos-gpu-worker-{i}" for i in range(8)})
            self.assertEqual(set(workers.values()), {0})
            rank1_receipt = artifacts / "nodes" / "node_1" / "node_receipt.json"
            changed = json.loads(rank1_receipt.read_text(encoding="utf-8"))
            changed["host"] = "node-a"
            rank1_receipt.write_text(json.dumps(changed), encoding="utf-8")
            _, _, complete, errors = evaluation_multinode.collect_node_receipts(launch, 0)
            self.assertFalse(complete)
            self.assertTrue(any("identity or worker IDs mismatch rank=1" in error for error in errors))
            rank1_receipt.unlink()
            _, _, complete, errors = evaluation_multinode.collect_node_receipts(launch, 0)
            self.assertFalse(complete)
            self.assertTrue(any("missing node receipt rank=1" in error for error in errors))
            self.assertEqual(evaluation_multinode._exec_vnodes(
                "(node-a:ncpus=64:mem=440gb:ngpus=4)+(node-b:ncpus=64:mem=440gb:ngpus=4)"), [
                    {"hostname": "node-a", "resources": {"ncpus": "64", "mem": "440gb", "ngpus": "4"}},
                    {"hostname": "node-b", "resources": {"ncpus": "64", "mem": "440gb", "ngpus": "4"}},
                ])

    def test_multinode_allocation_requires_two_hosts_and_four_gpus_per_chunk(self):
        protocol = self._profile(8)
        attrs = {
            "job_state": "R", "Job_Name": "cpvq24eval", "queue": "g4",
            "exec_vnode": "(node-a:ncpus=64:mem=440gb:ngpus=4:mpiprocs=1)+"
                          "(node-b:ncpus=64:mem=440gb:ngpus=4:mpiprocs=1)",
        }
        with tempfile.TemporaryDirectory() as directory:
            nodefile = Path(directory) / "PBS_NODEFILE"
            nodefile.write_text("node-a\nnode-b\n", encoding="utf-8")
            with patch.object(evaluation_multinode, "_qstat", return_value=attrs):
                hosts, _, chunks = evaluation_multinode._verify_g4_allocation(
                    protocol, "999.pbs101", nodefile)
            self.assertEqual(hosts, ["node-a", "node-b"])
            self.assertEqual([chunk["resources"]["ngpus"] for chunk in chunks], ["4", "4"])
            bad = dict(attrs, exec_vnode="(node-a:ncpus=64:ngpus=5)+(node-b:ncpus=64:ngpus=3)")
            with patch.object(evaluation_multinode, "_qstat", return_value=bad):
                with self.assertRaisesRegex(RuntimeError, "exactly 4 GPUs"):
                    evaluation_multinode._verify_g4_allocation(protocol, "999.pbs101", nodefile)

    def test_qstat_folded_exec_vnode_reassembles_without_inserting_spaces(self):
        protocol = self._profile(8)
        output = (
            "Job Id: 999.pbs101\n"
            "    job_state = R\n"
            "    Job_Name = cpvq24eval\n"
            "    queue = g4\n"
            "    exec_vnode = (node-a:ncpus=6\n"
            "\t4:mem=440gb:ngpus=4)+(node-\n"
            "        b:ncpus=64:mem=440gb:ngpus=4:mpiprocs=1)\n"
        )
        attrs = evaluation_multinode._qstat_attributes(output)
        self.assertEqual(attrs["exec_vnode"],
                         "(node-a:ncpus=64:mem=440gb:ngpus=4)+(node-b:ncpus=64:mem=440gb:ngpus=4:mpiprocs=1)")
        with tempfile.TemporaryDirectory() as directory:
            nodefile = Path(directory) / "PBS_NODEFILE"
            nodefile.write_text("node-a\nnode-b\n", encoding="utf-8")
            with patch.object(evaluation_multinode, "_qstat", return_value=attrs):
                hosts, _, chunks = evaluation_multinode._verify_g4_allocation(
                    protocol, "999.pbs101", nodefile)
            self.assertEqual(hosts, ["node-a", "node-b"])
            self.assertEqual([chunk["resources"]["ncpus"] for chunk in chunks], ["64", "64"])
            self.assertEqual([chunk["resources"]["ngpus"] for chunk in chunks], ["4", "4"])

            wrong = dict(attrs, exec_vnode=attrs["exec_vnode"].replace("ncpus=64", "ncpus=63", 1))
            with patch.object(evaluation_multinode, "_qstat", return_value=wrong):
                with self.assertRaisesRegex(RuntimeError, "exactly 4 GPUs and 64 CPUs"):
                    evaluation_multinode._verify_g4_allocation(protocol, "999.pbs101", nodefile)

    def test_multinode_claim_maps_native_host_and_refuses_duplicate_claim(self):
        protocol = self._profile(8)
        attrs = {
            "job_state": "R", "Job_Name": "cpvq24eval", "queue": "g4",
            "exec_vnode": "(node-a:ncpus=64:mem=440gb:ngpus=4)+(node-b:ncpus=64:mem=440gb:ngpus=4)",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nodefile = root / "999.pbs101"
            nodefile.write_text("node-a\nnode-b\n", encoding="utf-8")
            artifacts = root / "artifacts"
            control_sources = artifacts / "control_sources"
            control_sources.mkdir(parents=True)
            protocol_path = control_sources / "protocol.json"
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            launch = {
                "protocol": evaluate.PROTOCOL, "pbs_job_id": "999.pbs101",
                "pbs_nodefile": str(nodefile.resolve()),
                "nodes": [{"rank": 0, "hostname": "node-a"},
                          {"rank": 1, "hostname": "node-b"}],
                "root": str(root / "task"), "out": str(root / "task/results/evaluation"),
                "artifacts_root": str(artifacts), "protocol_path": str(protocol_path),
                "control_sources": str(control_sources), "engine_sources": str(artifacts / "engine_sources"),
                "legacy_sources": str(artifacts / "legacy_sources"),
                "source_helpers": str(artifacts / "source_helpers"),
                "node_script": str(control_sources / "evaluation_node.sh"),
                "deadline_epoch": 1_900_000_000.0, "soft_deadline_seconds": 85_500.0,
            }
            launch_path = artifacts / "evaluation_multinode_launch.json"
            launch_path.parent.mkdir(parents=True, exist_ok=True)
            launch_path.write_text(json.dumps(launch), encoding="utf-8")
            with patch.dict(os.environ, {"PBS_JOBID": "999.pbs101", "PBS_TASKNUM": "1",
                                         "PBS_NODEFILE": str(nodefile)}), \
                    patch.object(evaluation_multinode.socket, "gethostname", return_value="node-b.local"), \
                    patch.object(evaluation_multinode, "_qstat", return_value=attrs):
                info = evaluation_multinode.claim_node(launch_path)
                self.assertEqual(info["node_rank"], 1)
                self.assertEqual(info["host"], "node-b")
                self.assertEqual(info["nodefile_source"], "native_environment")
                self.assertEqual(info["effective_nodefile"], str(nodefile))
                with self.assertRaisesRegex(RuntimeError, "already has an allocation-time claim"):
                    evaluation_multinode.claim_node(launch_path)

    def test_multinode_rank0_uses_only_verified_parent_nodefile_when_native_value_is_missing(self):
        protocol = self._profile(8)
        attrs = {
            "job_state": "R", "Job_Name": "cpvq24eval", "queue": "g4",
            "exec_vnode": "(node-a:ncpus=64:mem=440gb:ngpus=4)+(node-b:ncpus=64:mem=440gb:ngpus=4)",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nodefile = root / "999.pbs101"
            nodefile.write_text("node-a\nnode-b\n", encoding="utf-8")
            control_sources = root / "artifacts" / "control_sources"
            control_sources.mkdir(parents=True)
            protocol_path = control_sources / "protocol.json"
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            launch = {
                "protocol": evaluate.PROTOCOL, "pbs_job_id": "999.pbs101",
                "pbs_nodefile": str(nodefile),
                "nodes": [{"rank": 0, "hostname": "node-a"},
                          {"rank": 1, "hostname": "node-b"}],
                "root": str(root / "task"), "out": str(root / "task/results/evaluation"),
                "artifacts_root": str(root / "artifacts"), "protocol_path": str(protocol_path),
                "control_sources": str(control_sources),
                "engine_sources": str(root / "artifacts" / "engine_sources"),
                "legacy_sources": str(root / "artifacts" / "legacy_sources"),
                "source_helpers": str(root / "artifacts" / "source_helpers"),
                "node_script": str(control_sources / "evaluation_node.sh"),
                "deadline_epoch": 1_900_000_000.0, "soft_deadline_seconds": 85_500.0,
            }
            launch_path = root / "artifacts" / "evaluation_multinode_launch.json"
            launch_path.parent.mkdir(parents=True, exist_ok=True)
            launch_path.write_text(json.dumps(launch), encoding="utf-8")
            env = {"PBS_JOBID": "999.pbs101", "PBS_TASKNUM": "0"}
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(evaluation_multinode.socket, "gethostname", return_value="node-a.local"), \
                    patch.object(evaluation_multinode, "_qstat", return_value=attrs):
                info = evaluation_multinode.claim_node(launch_path)
            self.assertEqual(info["node_rank"], 0)
            self.assertEqual(info["nodefile_source"], "verified_parent_native_path")
            self.assertEqual(info["effective_nodefile"], str(nodefile))
            self.assertEqual(info["native_nodefile"], "(missing)")

            bad_job = dict(env, PBS_JOBID="998.pbs101")
            with patch.dict(os.environ, bad_job, clear=True), \
                    patch.object(evaluation_multinode.socket, "gethostname", return_value="node-a.local"), \
                    patch.object(evaluation_multinode, "_qstat", return_value=attrs):
                with self.assertRaisesRegex(RuntimeError, "PBS_JOBID/PBS_TASKNUM"):
                    evaluation_multinode.claim_node(launch_path)

            bad_launch = dict(launch, pbs_nodefile=str(root / "wrong.pbs101"))
            bad_launch_path = root / "artifacts" / "bad_launch.json"
            bad_launch_path.write_text(json.dumps(bad_launch), encoding="utf-8")
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(evaluation_multinode.socket, "gethostname", return_value="node-a.local"), \
                    patch.object(evaluation_multinode, "_qstat", return_value=attrs):
                with self.assertRaisesRegex(RuntimeError, "does not match the native job ID"):
                    evaluation_multinode.claim_node(bad_launch_path)

            wrong_host = root / "other" / "999.pbs101"
            wrong_host.parent.mkdir()
            wrong_host.write_text("node-a\nnode-c\n", encoding="utf-8")
            mismatched = dict(launch, pbs_nodefile=str(wrong_host))
            mismatched_path = root / "artifacts" / "mismatched_launch.json"
            mismatched_path.write_text(json.dumps(mismatched), encoding="utf-8")
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(evaluation_multinode.socket, "gethostname", return_value="node-a.local"), \
                    patch.object(evaluation_multinode, "_qstat", return_value=attrs):
                with self.assertRaisesRegex(RuntimeError, "PBS_NODEFILE node order differs from the launch receipt"):
                    evaluation_multinode.claim_node(mismatched_path)

            with patch.dict(os.environ, dict(env, PBS_NODEFILE=str(wrong_host)), clear=True), \
                    patch.object(evaluation_multinode.socket, "gethostname", return_value="node-a.local"), \
                    patch.object(evaluation_multinode, "_qstat", return_value=attrs):
                with self.assertRaisesRegex(RuntimeError, "differs from the verified parent allocation path"):
                    evaluation_multinode.claim_node(launch_path)

            wrong_allocation = dict(attrs, exec_vnode=
                                    "(node-b:ncpus=64:ngpus=4)+(node-c:ncpus=64:ngpus=4)")
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(evaluation_multinode.socket, "gethostname", return_value="node-a.local"), \
                    patch.object(evaluation_multinode, "_qstat", return_value=wrong_allocation), \
                    patch.object(evaluation_multinode, "_nodefile_hosts",
                                 side_effect=AssertionError("nodefile opened before local qstat host validation")):
                with self.assertRaisesRegex(RuntimeError, "not present in qstat exec_vnode"):
                    evaluation_multinode.claim_node(launch_path)

            with patch.dict(os.environ, env, clear=True), \
                    patch.object(evaluation_multinode.socket, "gethostname", return_value="node-c.local"), \
                    patch.object(evaluation_multinode, "_qstat", return_value=attrs):
                with self.assertRaisesRegex(RuntimeError, "not present in qstat exec_vnode"):
                    evaluation_multinode.claim_node(launch_path)

    def test_multinode_node_wrapper_records_verified_nodefile_source_before_export(self):
        script = (HERE / "evaluation_node.sh").read_text(encoding="utf-8")
        self.assertIn("Native PBS_TASKNUM changed after node claim", script)
        self.assertIn("PBS_NODEFILE=$EFFECTIVE_NODEFILE", script)
        self.assertIn("PBS_NODEFILE_SOURCE=%s", script)
        self.assertIn("export PBS_NODEFILE COSMOS_NODEFILE_SOURCE=", script)


class SharedBundleQueueTests(unittest.TestCase):
    @staticmethod
    def bundles(count):
        return [{"bundle_id": f"bundle-{bundle}",
                 "indices": list(range(bundle * 10, bundle * 10 + 10))}
                for bundle in range(count)]

    @staticmethod
    def completed_for(bundle):
        return {(arm, index): {"complete": True}
                for arm in evaluate.NEW_ARMS for index in bundle["indices"]}

    def test_eight_workers_claim_twenty_eight_bundles_without_duplicates(self):
        bundles = self.bundles(28)
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"PBS_JOBID": "queue-test.pbs101"}):
            seen, seen_lock = set(), threading.Lock()

            def worker(worker_id):
                completed = {}
                while True:
                    bundle, drained = evaluate.claim_next(
                        directory, bundles, worker_id, "queue-test.pbs101", PROTOCOL_DOC, completed)
                    if drained:
                        return
                    with seen_lock:
                        if bundle["bundle_id"] in seen:
                            raise AssertionError(f"Duplicate claim for {bundle['bundle_id']}")
                        seen.add(bundle["bundle_id"])
                    completed.update(self.completed_for(bundle))
                    evaluate.release_claim(directory, bundle, worker_id,
                                           "queue-test.pbs101", PROTOCOL_DOC, completed)

            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(worker, range(8)))
            state = evaluate.read_json(Path(directory) / "claim_queue.json")
        self.assertEqual(seen, {bundle["bundle_id"] for bundle in bundles})
        self.assertEqual(set(state["completed_bundles"]), seen)
        self.assertEqual(state["claims"], {})

    def test_completed_ledger_prevents_reclaim_and_partial_release_reuses_bundle(self):
        bundles = self.bundles(2)
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"PBS_JOBID": "queue-test.pbs101"}):
            bundle, drained = evaluate.claim_next(directory, bundles, 0,
                "queue-test.pbs101", PROTOCOL_DOC, {})
            self.assertFalse(drained)
            partial = {(evaluate.NEW_ARMS[0], bundle["indices"][0]): {}}
            evaluate.release_claim(directory, bundle, 0, "queue-test.pbs101", PROTOCOL_DOC, partial)
            reclaimed, drained = evaluate.claim_next(directory, bundles, 1,
                "queue-test.pbs101", PROTOCOL_DOC, {})
            self.assertFalse(drained)
            self.assertEqual(reclaimed["bundle_id"], bundle["bundle_id"])

            full = self.completed_for(reclaimed)
            evaluate.release_claim(directory, reclaimed, 1,
                                   "queue-test.pbs101", PROTOCOL_DOC, full)
            next_bundle, drained = evaluate.claim_next(directory, bundles, 2,
                "queue-test.pbs101", PROTOCOL_DOC, full)
            self.assertFalse(drained)
            self.assertNotEqual(next_bundle["bundle_id"], reclaimed["bundle_id"])

    def test_new_run_discards_old_ledger_and_claims_but_merges_verified_seed(self):
        bundles = self.bundles(3)
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"PBS_JOBID": "new-run.pbs101"}):
            state = evaluate.initial_queue(bundles, "old-run.pbs101", {"bundle-1"})
            state["claims"]["bundle-0"] = {
                "worker_id": 4, "run_id": "old-run.pbs101", "pbs_job_id": "old-run.pbs101",
                "host": "old-node", "pid": 1234,
            }
            evaluate.save_json(Path(directory) / "claim_queue.json", state)
            verified = self.completed_for(bundles[0])
            bundle, drained = evaluate.claim_next(
                directory, bundles, 2, "new-run.pbs101", PROTOCOL_DOC, verified)
            self.assertFalse(drained)
            self.assertEqual(bundle["bundle_id"], "bundle-1")
            current = evaluate.read_json(Path(directory) / "claim_queue.json")
        self.assertEqual(current["run_id"], "new-run.pbs101")
        self.assertEqual(current["completed_bundles"], ["bundle-0"])
        self.assertEqual(set(current["claims"]), {"bundle-1"})
        self.assertEqual(current["claims"]["bundle-1"]["worker_id"], 2)

    def test_idle_worker_exits_when_every_remaining_bundle_is_claimed(self):
        bundles = self.bundles(2)
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"PBS_JOBID": "queue-test.pbs101"}):
            state = evaluate.initial_queue(bundles, "queue-test.pbs101", {"bundle-1"})
            state["claims"]["bundle-0"] = {
                "worker_id": 0, "run_id": "queue-test.pbs101", "pbs_job_id": "queue-test.pbs101",
                "host": "other-node", "pid": 1,
            }
            evaluate.save_json(Path(directory) / "claim_queue.json", state)
            with patch.object(evaluate.time, "sleep", side_effect=AssertionError("queue must not poll")):
                bundle, drained = evaluate.claim_next(
                    directory, bundles, 1, "queue-test.pbs101", PROTOCOL_DOC, {})
        self.assertIsNone(bundle)
        self.assertTrue(drained)

    def test_lock_owner_has_job_host_pid_token_and_release_removes_only_own_lock(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"PBS_JOBID": "queue-test.pbs101"}):
            lock = evaluate._queue_lock(Path(directory) / "claim_queue.json", timeout=1)
            lock.__enter__()
            lock_path = Path(directory) / "claim_queue.json.lock"
            owner = evaluate.read_json(lock_path)
            self.assertEqual(owner["pbs_job_id"], "queue-test.pbs101")
            self.assertTrue(owner["host"])
            self.assertEqual(owner["pid"], os.getpid())
            self.assertTrue(owner["token"])
            lock_path.write_text(json.dumps(dict(owner, token="another-owner")), encoding="utf-8")
            lock.__exit__(None, None, None)
            self.assertTrue(lock_path.exists())
            lock_path.unlink()


if __name__ == "__main__":
    unittest.main()
