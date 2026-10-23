import unittest
from pathlib import Path
from contextlib import nullcontext
from unittest.mock import Mock, patch

import control


def passing_observation():
    job_id = "25910001.pbs101"
    summary = {
        "protocol": control.PROTOCOL_ID, "status": "complete", "complete": True,
        "pbs_jobid": job_id,
        "banks": {"expected": 614, "available": 614, "receipt": True},
        "moments": {"expected": 80, "available": 80, "receipt": True},
        "gates": {"source": True, "repro": True, "frozen": True,
                  "formal_parity_max_error": 8e-7, "orthogonality_max_error": 4e-5},
        "endpoints": {
            "R0_VQ0": {"calibration": 80, "heldout": 80},
            "Seq8": {"calibration": 80, "heldout": 80, "updates": 8, "exposures": 80},
            "Joint8": {"calibration": 80, "heldout": 80, "updates": 8, "exposures": 80},
        },
        "references": {"H_VQ8": {"heldout": 80}, "R0_W16A4": {"heldout": 80}},
        "new_teacher_queries": 0, "comparison_file": "results/comparison.json",
        "comparison_complete": True,
    }
    stage_labels = {
        "moments_0": ("moments", 0), "moments_1": ("moments", 1),
        "fit_0": ("fit", 0), "fit_1": ("fit", 1),
        "common_eval_0": ("common_eval", 0), "common_eval_1": ("common_eval", 1),
        "gate_0": ("gate", 0), "finalize_0": ("finalize", 0),
    }
    stage_summaries = {
        name: {"protocol": control.PROTOCOL_ID, "stage": label, "shard": shard,
              "complete": True, "status": "complete", "pbs_jobid": job_id,
              "units": control.STAGE_UNITS[name]}
        for name, (label, shard) in stage_labels.items()
    }
    arm_summaries = {
        arm: {"protocol": control.PROTOCOL_ID, "arm": arm, "status": "complete",
              "complete": True, "pbs_jobid": job_id, "updates": 8, "exposures": 80,
              "calibration": 80, "heldout": 80, "all_parameters_frozen": True,
              "formal_infer_max_abs": 8e-7,
              "orthogonality": {"action": 4e-5, "video": 4e-5},
              "frozen_rotation_exact": arm == "Seq8"}
        for arm in ("Seq8", "Joint8")
    }
    return {
        "handle": {"job_id": job_id},
        "scheduler": {"query_rc": 0, "job_state": "F", "Exit_status": "0",
                      "queue": "g2", "exec_host": "x1000c0s0b0n0/0+x1000c0s0b0n0/1"},
        "artifact_exit_code": 0, "phase_complete": True,
        "summary": summary, "worker_summary": dict(summary),
        "progress": {"stage": "complete", "complete": True, "pbs_jobid": job_id},
        "receipts": {
            "banks/receipt.json": {"protocol": control.PROTOCOL_ID, "complete": True,
                                   "passed": True, "expected": 614, "available": 614},
            "moments/receipt.json": {"protocol": control.PROTOCOL_ID, "complete": True,
                                     "passed": True, "expected": 80, "available": 80},
            "source_receipt.json": {"protocol": control.PROTOCOL_ID, "complete": True, "passed": True},
            "repro_receipt.json": {"protocol": control.PROTOCOL_ID, "complete": True, "passed": True},
        },
        "stage_summaries": stage_summaries,
        "stage_progress": {stage: {"protocol": control.PROTOCOL_ID, "pbs_jobid": job_id,
                                    "status": "complete"} for stage in control.STAGES},
        "arm_summaries": arm_summaries,
        "arm_progress": {arm: {"protocol": control.PROTOCOL_ID, "pbs_jobid": job_id,
                               "status": "complete", "updates": 8, "exposures": 80}
                         for arm in ("Seq8", "Joint8")},
        "checkpoint_sizes": {arm: {"state.pt": 1024, "deployment.pt": 4096}
                            for arm in ("Seq8", "Joint8")},
        "comparison_size": 300000,
        "result_files": {name: True for name in (*control.RESULT_FILES, "comparison.json")},
        "result_json": {"training.json": {"protocol": control.PROTOCOL_ID}},
    }


class VerifyGateTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {"protocol": control.PROTOCOL_ID}

    def test_frozen_allocation_and_stage_contract(self):
        self.assertEqual(control.ROOT, "/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-dual-20261022")
        self.assertEqual(control.PROTOCOL_ID, "fastwam-vq-r128-c80-dual-v1")
        self.assertEqual(control.JOB_NAME, "fwvr128c80d")
        self.assertEqual(control.SELECT, "1:ncpus=32:mem=220gb:ngpus=2")
        self.assertEqual(control.WALLTIME, "08:00:00")
        self.assertEqual(control.ENDPOINTS, ("R0_VQ0", "Seq8", "Joint8"))
        self.assertEqual(control.STAGE_UNITS["fit_0"], 307)
        self.assertEqual(control.load_protocol()["protocol"], control.PROTOCOL_ID)

    def test_complete_pbs_and_full_c80_receipts_pass(self):
        self.assertEqual(control.verification_failures(passing_observation(), self.protocol), [])

    def test_source_parity_counts_and_banks_are_independent_gates(self):
        observation = passing_observation()
        observation["summary"]["banks"]["available"] = 613
        observation["summary"]["gates"]["source"] = False
        observation["summary"]["gates"]["formal_parity_max_error"] = 2e-6
        observation["receipts"]["moments/receipt.json"]["available"] = 79
        failures = control.verification_failures(observation, self.protocol)
        for expected in ("banks", "source", "formal parity", "moments/receipt.json"):
            self.assertTrue(any(expected in item for item in failures), expected)

    def test_each_stage_arm_reference_and_frozen_gate_is_checked(self):
        observation = passing_observation()
        observation["stage_summaries"]["common_eval_1"]["units"] = 79
        observation["arm_summaries"]["Seq8"]["frozen_rotation_exact"] = False
        observation["arm_summaries"]["Joint8"]["orthogonality"]["video"] = 6e-5
        observation["summary"]["references"]["H_VQ8"]["heldout"] = 79
        failures = control.verification_failures(observation, self.protocol)
        for expected in ("common_eval_1", "frozen rotation", "orthogonality", "H_VQ8"):
            self.assertTrue(any(expected in item for item in failures), expected)

    def test_terminal_artifact_queue_g2_and_marker_are_required(self):
        observation = passing_observation()
        observation["scheduler"].update(job_state="R", Exit_status="1", queue="g1",
                                         exec_host="x1000c0s0b0n0/0")
        observation["artifact_exit_code"] = 1
        observation["phase_complete"] = False
        failures = control.verification_failures(observation, self.protocol)
        for expected in ("terminal F/X", "Exit_status", "queue", "exit_code.txt", "PHASE_COMPLETE"):
            self.assertTrue(any(expected in item for item in failures), expected)

    def test_synthetic_qstat_route_uses_queue_not_compute_hostname(self):
        g2_fixture = """Job Id: 25910001.pbs101
    job_state = F
    queue = g2
    exec_host = x1000c0s0b0n0/0+x1000c0s0b0n0/1
    Exit_status = 0
"""
        g1_fixture = g2_fixture.replace("queue = g2", "queue = g1")
        g2 = control.parse_qstat_fields(g2_fixture)
        g1 = control.parse_qstat_fields(g1_fixture)
        self.assertEqual(g2["queue"], "g2")
        self.assertEqual(g2["exec_host"], "x1000c0s0b0n0/0+x1000c0s0b0n0/1")
        self.assertEqual(g1["queue"], "g1")
        self.assertNotEqual(g1["queue"], "g2")


class ResumeSafetyTests(unittest.TestCase):
    def test_repaired_code_is_uploaded_only_after_terminal_failure_is_confirmed(self):
        handle = dict(job_id='25910001.pbs101', job_name=control.JOB_NAME, attempt=0,
                      remote_root=control.ROOT, protocol=control.PROTOCOL_ID, status='confirmed')
        for status in ('F', 'R'):
            with self.subTest(status=status):
                state = {'handle': dict(handle)}
                observation = {'handle': handle,
                    'scheduler': {'job_id': handle['job_id'], 'query_rc': 0, 'job_state': status, 'Exit_status': '1'},
                    'artifact_exit_code': 1, 'phase_complete': False, 'progress': {'stage': 'moments'},
                    'summary': None, 'checkpoint_sizes': {}, 'result_json': {}}
                remote = Mock()
                remote.sftp_client.side_effect = lambda _: nullcontext(object())
                remote.command.return_value = (0, '25910002.pbs101')
                with patch.object(control, 'read_state', return_value=(state, {})), \
                     patch.object(control, 'recover_intent', return_value=None), \
                     patch.object(control, 'observe', return_value=(observation, state)), \
                     patch.object(control, 'save_submission_state'), \
                     patch.object(control, 'remote_protocol_matches'), \
                     patch.object(control, 'assert_remote_files'), \
                     patch.object(control, 'transfer_files', return_value=[{'name': 'worker.py', 'bytes': 5}]) as transfer, \
                     patch.object(control, 'named_jobs', return_value=[]):
                    if status == 'R':
                        with self.assertRaises(RuntimeError):
                            control.resume(remote, object())
                        transfer.assert_not_called()
                    else:
                        result = control.resume(remote, object())
                        self.assertTrue(result['resumed'])
                        self.assertEqual(result['from_job_id'], handle['job_id'])
                        self.assertFalse(transfer.call_args.kwargs['include_protocol'])

    def test_live_unknown_and_successful_handles_cannot_resume(self):
        base = {
            "handle": {"job_id": "25910001.pbs101"},
            "scheduler": {"query_rc": 0, "job_state": "X", "Exit_status": "75"},
            "artifact_exit_code": 75, "phase_complete": False,
            "summary": {"status": "checkpointed", "complete": False},
            "progress": {"complete": False},
        }
        self.assertEqual(control.resume_failures(base, self.protocol()), [])
        base["scheduler"]["job_state"] = "R"
        self.assertTrue(any("terminal" in item for item in control.resume_failures(base, self.protocol())))
        base["scheduler"].update(job_state="F", Exit_status="0")
        self.assertTrue(any("Exit_status is 0" in item for item in control.resume_failures(base, self.protocol())))
        base["scheduler"]["Exit_status"] = "75"
        base["phase_complete"] = True
        self.assertTrue(any("PHASE_COMPLETE" in item for item in control.resume_failures(base, self.protocol())))

    @staticmethod
    def protocol():
        return {"protocol": control.PROTOCOL_ID}

    def test_checkpointed_and_stage_replay_resumes_are_distinguished(self):
        self.assertEqual(control.checkpoint_resume_mode(
            {"complete": False}, {"Seq8": {"state.pt": 100, "deployment.pt": None}}, {}
        ), "checkpoint")
        self.assertEqual(control.checkpoint_resume_mode({"complete": False}, {}, {}), "stage-replay")
        with self.assertRaisesRegex(RuntimeError, "committed updates"):
            control.checkpoint_resume_mode(
                {"complete": False}, {"Seq8": {"deployment.pt": None}},
                {"Seq8": {"updates": 2}}
            )
        with self.assertRaisesRegex(RuntimeError, "empty"):
            control.checkpoint_resume_mode(
                {"complete": False}, {"Joint8": {"state.pt": 0}}, {}
            )


class RunScriptTests(unittest.TestCase):
    def test_allocation_gpu_stage_order_and_marker_gates_are_present(self):
        source = Path(__file__).with_name("run.pbs").read_text(encoding="utf-8")
        environment = Path(__file__).with_name("environment.sh").read_text(encoding="utf-8")
        for expected in (
            "#PBS -N fwvr128c80d", "#PBS -q normal",
            "#PBS -l select=1:ncpus=32:mem=220gb:ngpus=2", "#PBS -l walltime=08:00:00",
            'test -n "${PBS_JOBID:-}"', 'test -s "$PBS_NODEFILE"',
            "Refusing login-node execution", "Current host is not in the approved PBS allocation",
            "PBS job is not in running state", "queue[[:space:]]*=[[:space:]]*g2", "sleep 15",
            "FW_SOFT_DEADLINE_SECONDS=28500", "FW_SAVE_RESERVE_SECONDS=300",
            "run_parallel_stage moments", "run_parallel_stage fit", "run_parallel_stage common_eval",
            "--stage gate", "--stage arm --arm Seq8", "--stage arm --arm Joint8", "--stage finalize",
            "touch \"$ARTIFACTS/PHASE_COMPLETE\"", "formal_parity_max_error", "orthogonality_max_error",
        ):
            self.assertIn(expected, source)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG=:4096:8,PYTHONHASHSEED=20261022", environment)
        self.assertIn("OMP_NUM_THREADS=${FW_CPU_THREADS:-8}", environment)
        self.assertIn("FW_SAVE_RESERVE_SECONDS=${FW_SAVE_RESERVE_SECONDS:-300}", environment)

    def test_remote_control_limits_reads_and_does_not_pull_comparison(self):
        source = Path(__file__).with_name("control.py").read_text(encoding="utf-8")
        self.assertIn("LIMIT = 256 * 1024", source)
        self.assertIn('remote_size(sftp, RESULTS + "/comparison.json")', source)
        pull_body = source.split("def pull(", 1)[1]
        self.assertNotIn('RESULTS + "/comparison.json"', pull_body)


if __name__ == "__main__":
    unittest.main()
