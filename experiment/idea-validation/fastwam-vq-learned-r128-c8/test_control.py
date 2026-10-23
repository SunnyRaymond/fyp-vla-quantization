import unittest
from pathlib import Path
from unittest.mock import patch

import control


def passing_observation():
    summary = {
        "status": "complete",
        "complete": True,
        "protocol": control.PROTOCOL_ID,
        "rotation_updates": 8,
        "vq_updates": 8,
        "target_linears": 614,
        "heldout_queries": 80,
        "calibration_queries": 8,
        "pbs_jobid": "25900001.pbs101",
        "all_parameters_frozen": True,
    }
    return {
        "handle": {"job_id": "25900001.pbs101"},
        "scheduler": {"query_rc": 0, "job_state": "F", "Exit_status": "0"},
        "artifact_exit_code": 0,
        "phase_complete": True,
        "summary": summary,
    }


class VerifyGateTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {"protocol": control.PROTOCOL_ID}

    def test_complete_terminal_artifact_passes(self):
        self.assertEqual(control.verification_failures(passing_observation(), self.protocol), [])

    def test_requires_all_protocol_counts_and_frozen_parameters(self):
        observation = passing_observation()
        observation["summary"].update(
            rotation_updates=7,
            vq_updates=7,
            target_linears=613,
            heldout_queries=79,
            calibration_queries=7,
            all_parameters_frozen=False,
        )
        failures = control.verification_failures(observation, self.protocol)
        self.assertTrue(any("rotation_updates" in item for item in failures))
        self.assertTrue(any("vq_updates" in item for item in failures))
        self.assertTrue(any("target_linears" in item for item in failures))
        self.assertTrue(any("heldout_queries" in item for item in failures))
        self.assertTrue(any("calibration_queries" in item for item in failures))
        self.assertTrue(any("all parameters" in item for item in failures))

    def test_terminal_exit_artifact_and_job_binding_are_required(self):
        observation = passing_observation()
        observation["scheduler"].update(job_state="R", Exit_status="1")
        observation["artifact_exit_code"] = 1
        observation["phase_complete"] = False
        observation["summary"]["pbs_jobid"] = "25899999.pbs101"
        failures = control.verification_failures(observation, self.protocol)
        self.assertTrue(any("terminal F/X" in item for item in failures))
        self.assertTrue(any("Exit_status" in item for item in failures))
        self.assertTrue(any("exit_code.txt" in item for item in failures))
        self.assertTrue(any("PHASE_COMPLETE" in item for item in failures))
        self.assertTrue(any("pbs_jobid" in item for item in failures))

    def test_qstat_expiry_keeps_confirmed_terminal_state(self):
        prior = {
            "job_id": "25900001.pbs101", "query_rc": 0,
            "job_state": "X", "Exit_status": "0", "Job_Name": control.JOB_NAME,
        }
        current, retained = control.retain_terminal_history(
            {"job_id": "25900001.pbs101", "query_rc": 153, "job_state": None, "Exit_status": None},
            prior,
            "25900001.pbs101",
        )
        self.assertTrue(current["terminal_from_history"])
        self.assertEqual(current["job_state"], "X")
        self.assertEqual(current["Exit_status"], "0")
        self.assertEqual(current["current_query_rc"], 153)
        self.assertEqual(retained, prior)

    def test_remote_terminal_receipt_survives_stale_local_state(self):
        prior = {"job_id": "25900001.pbs101", "job_state": "F", "Exit_status": "0"}
        handle = {"job_id": "25900001.pbs101"}
        merged = control.merged_state(
            {"handle": handle, "last_terminal": prior},
            {"handle": handle, "last_terminal": None},
            handle,
        )
        self.assertEqual(merged["last_terminal"], prior)


class ResumeSafetyTests(unittest.TestCase):
    def test_attempt_names_are_distinct_and_valid(self):
        self.assertEqual(control.attempt_job_name(0), "fwlr128c8")
        self.assertEqual(control.attempt_job_name(1), "fwlr128c8-a1")
        self.assertEqual(control.attempt_number({"job_name": "fwlr128c8-a12"}), 12)
        with self.assertRaises(ValueError):
            control.attempt_job_name(-1)

    def test_only_confirmed_failed_incomplete_worker_is_resumable(self):
        observation = {
            "handle": {"job_id": "25900001.pbs101"},
            "scheduler": {"query_rc": 0, "job_id": "25900001.pbs101",
                           "job_state": "X", "Exit_status": "75"},
            "artifact_exit_code": 75,
            "phase_complete": False,
            "summary": {"status": "checkpointed", "complete": False},
        }
        self.assertEqual(control.is_incomplete_failure(observation, {"protocol": control.PROTOCOL_ID}), [])

    def test_missing_exit_is_allowed_but_success_marker_or_complete_summary_blocks_resume(self):
        observation = {
            "handle": {"job_id": "25900001.pbs101"},
            "scheduler": {"query_rc": 0, "job_id": "25900001.pbs101", "job_state": "F"},
            "artifact_exit_code": None,
            "phase_complete": False,
            "summary": {},
        }
        self.assertEqual(control.is_incomplete_failure(observation, {"protocol": control.PROTOCOL_ID}), [])
        observation["artifact_exit_code"] = 0
        self.assertTrue(control.is_incomplete_failure(observation, {"protocol": control.PROTOCOL_ID}))
        observation["artifact_exit_code"] = 1
        observation["phase_complete"] = True
        self.assertTrue(control.is_incomplete_failure(observation, {"protocol": control.PROTOCOL_ID}))
        observation["phase_complete"] = False
        observation["summary"] = {"status": "complete", "complete": True}
        self.assertTrue(control.is_incomplete_failure(observation, {"protocol": control.PROTOCOL_ID}))

    def test_missing_checkpoint_allows_fresh_attempt_only_at_zero_updates(self):
        progress = {"rotation_updates": 0, "vq_updates": 0, "status": "running"}
        self.assertEqual(control.checkpoint_resume_mode(progress, None), "fresh")
        self.assertEqual(control.checkpoint_resume_mode(None, 12), "checkpoint")
        with self.assertRaisesRegex(RuntimeError, "progress are both missing"):
            control.checkpoint_resume_mode(None, None)
        with self.assertRaisesRegex(RuntimeError, "rotation_updates"):
            control.checkpoint_resume_mode({**progress, "rotation_updates": 1}, None)
        with self.assertRaisesRegex(RuntimeError, "vq_updates"):
            control.checkpoint_resume_mode({**progress, "vq_updates": 1}, None)
        with self.assertRaisesRegex(RuntimeError, "claims completion"):
            control.checkpoint_resume_mode({**progress, "status": "complete"}, None)
        with self.assertRaisesRegex(RuntimeError, "empty"):
            control.checkpoint_resume_mode(progress, 0)

    def test_unknown_qsub_intent_never_retries_when_history_is_empty(self):
        class FakeRemote:
            def __init__(self):
                self.commands = []

            def command(self, _transport, command, timeout):
                self.commands.append(command)
                return 0, ""

        remote = FakeRemote()
        state = {
            "handle": {"job_id": "25900001.pbs101", "job_name": control.JOB_NAME, "attempt": 0},
            "intent": {"kind": "resume", "status": "intent_before_qsub", "attempt": 1,
                       "job_name": control.attempt_job_name(1)},
        }
        with self.assertRaisesRegex(RuntimeError, "Do not submit another attempt"):
            control.recover_pending_resume(remote, object(), state)
        self.assertEqual(len(remote.commands), 1)
        self.assertIn("qselect -x", remote.commands[0])

    def test_pending_qsub_recovers_exactly_one_history_match_without_submitting(self):
        class FakeRemote:
            def __init__(self):
                self.commands = []

            def command(self, _transport, command, timeout):
                self.commands.append(command)
                return 0, "25900002.pbs101\n"

        remote = FakeRemote()
        state = {
            "handle": {"job_id": "25900001.pbs101", "job_name": control.JOB_NAME, "attempt": 0},
            "intent": {"kind": "resume", "status": "intent_before_qsub", "attempt": 1,
                       "job_name": control.attempt_job_name(1)},
        }
        with patch.object(control, "save_submission_state") as save_state:
            result = control.recover_pending_resume(remote, object(), state)
        self.assertEqual(result["handle"]["job_id"], "25900002.pbs101")
        self.assertEqual(result["handle"]["attempt"], 1)
        self.assertEqual(len(remote.commands), 1)
        self.assertTrue(save_state.called)
        self.assertEqual(state["intent"]["status"], "confirmed")

    def test_pull_includes_small_comparison_report_and_selfcheck(self):
        source = Path(__file__).with_name("control.py").read_text(encoding="utf-8")
        for name in ("comparison.json", "RESULTS.zh.md", "costs.json", "COSTS.zh.md", "selfcheck.json"):
            self.assertIn(name, source)


class RunScriptTests(unittest.TestCase):
    def test_resources_allocation_gate_and_worker_entrypoint(self):
        source = (Path(__file__).parent / "run.pbs").read_text(encoding="utf-8")
        for expected in (
            "#PBS -q normal",
            "#PBS -l select=1:ncpus=16:mem=110gb:ngpus=1",
            "#PBS -l walltime=12:00:00",
            'test -n "${PBS_JOBID:-}"',
            'test -s "$PBS_NODEFILE"',
            "Refusing login-node execution",
            "PBS job is not in running state",
            "FW_SOFT_DEADLINE_SECONDS=42900",
            "FW_SAVE_RESERVE_SECONDS=300",
            'export CUDA_VISIBLE_DEVICES="$GPU_UUID"',
            'run_python "$ARTIFACTS/worker.py" --protocol "$ARTIFACTS/protocol.json"',
        ):
            self.assertIn(expected, source)
        self.assertNotIn('run_python "$ARTIFACTS/selfcheck.py"', source)


if __name__ == "__main__":
    unittest.main()
