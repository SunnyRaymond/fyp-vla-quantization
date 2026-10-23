import unittest
from pathlib import Path
from unittest.mock import patch

import control


def passing_observation():
    summary = {
        "protocol": control.PROTOCOL_ID,
        "status": "complete",
        "complete": True,
        "target_linears": 614,
        "existing_R_updates": 8,
        "joint_updates": 8,
        "diagnostic_heldout_queries_per_arm": 80,
        "joint_heldout_queries": 80,
        "calibration_arms": 7,
        "calibration_queries_per_arm": 8,
        "all_parameters_frozen": True,
        "new_teacher_queries": 0,
        "comparison_file": "results/comparison.json",
        "pbs_jobid": "25900001.pbs101",
    }
    progress = {
        "stage": "complete",
        "complete": True,
        "joint_updates": 8,
        "completed_queries": {
            **{"calibration/" + arm: 8 for arm in control.ARMS},
            **{"heldout/" + arm: 80 for arm in control.HELDOUT_ARMS},
        },
    }
    return {
        "handle": {"job_id": "25900001.pbs101"},
        "scheduler": {"query_rc": 0, "job_state": "F", "Exit_status": "0"},
        "artifact_exit_code": 0,
        "phase_complete": True,
        "summary": summary,
        "progress": progress,
        "worker_summary": dict(summary),
        "result_files": {name: True for name in control.RESULT_FILES},
        "result_json": {
            "comparison.json": {"protocol": control.PROTOCOL_ID},
            "training.json": {"protocol": control.PROTOCOL_ID, "history": [{} for _ in range(8)]},
        },
    }


class VerifyGateTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {"protocol": control.PROTOCOL_ID}

    def test_complete_terminal_receipts_and_all_counts_pass(self):
        self.assertEqual(control.verification_failures(passing_observation(), self.protocol), [])

    def test_all_required_metrics_and_each_arm_query_count_are_gated(self):
        observation = passing_observation()
        observation["summary"].update(target_linears=613, joint_updates=7, new_teacher_queries=1)
        observation["progress"]["completed_queries"]["heldout/H_W16A4"] = 79
        failures = control.verification_failures(observation, self.protocol)
        for expected in ("target_linears", "joint_updates", "new_teacher_queries", "H_W16A4"):
            self.assertTrue(any(expected in item for item in failures), expected)

    def test_scheduler_exit_artifact_and_complete_marker_are_required(self):
        observation = passing_observation()
        observation["scheduler"].update(job_state="R", Exit_status="1")
        observation["artifact_exit_code"] = 1
        observation["phase_complete"] = False
        failures = control.verification_failures(observation, self.protocol)
        for expected in ("terminal F/X", "Exit_status", "exit_code.txt", "PHASE_COMPLETE"):
            self.assertTrue(any(expected in item for item in failures), expected)


class ResumeSafetyTests(unittest.TestCase):
    def test_checkpoint_or_zero_work_progress_is_required(self):
        self.assertEqual(control.checkpoint_resume_mode({"stage": "joint8", "joint_updates": 3}, 15), "checkpoint")
        self.assertEqual(control.checkpoint_resume_mode(
            {"stage": "startup", "joint_updates": 0, "completed_queries": {}}, None
        ), "fresh")
        self.assertEqual(control.checkpoint_resume_mode(None, None), "fresh")
        with self.assertRaisesRegex(RuntimeError, "empty"):
            control.checkpoint_resume_mode({"stage": "startup"}, 0)
        with self.assertRaisesRegex(RuntimeError, "zero Joint8 updates"):
            control.checkpoint_resume_mode(
                {"stage": "joint_training", "joint_updates": 1, "completed_queries": {}}, None
            )
        with self.assertRaisesRegex(RuntimeError, "training.json"):
            control.checkpoint_resume_mode(None, None, {"history": [{"update": 1}]})

    def test_live_unknown_and_successful_states_cannot_resume(self):
        protocol = {"protocol": control.PROTOCOL_ID}
        base = {
            "handle": {"job_id": "25900001.pbs101"},
            "scheduler": {"job_id": "25900001.pbs101", "query_rc": 0, "job_state": "X", "Exit_status": "75"},
            "artifact_exit_code": 75,
            "phase_complete": False,
            "summary": {"status": "checkpointed", "complete": False},
            "progress": {"complete": False},
        }
        self.assertEqual(control.resume_failures(base, protocol), [])
        base["scheduler"]["job_state"] = "R"
        self.assertTrue(any("terminal" in item for item in control.resume_failures(base, protocol)))
        base["scheduler"].update(job_state="F", Exit_status="0")
        self.assertTrue(any("Exit_status is 0" in item for item in control.resume_failures(base, protocol)))
        base["scheduler"]["Exit_status"] = "75"
        base["phase_complete"] = True
        self.assertTrue(any("PHASE_COMPLETE" in item for item in control.resume_failures(base, protocol)))

    def test_unresolved_qsub_intent_never_retries_when_history_is_empty(self):
        class FakeRemote:
            def __init__(self):
                self.commands = []

            def command(self, _transport, command, timeout):
                self.commands.append(command)
                return 0, ""

        remote = FakeRemote()
        state = {"intent": {"kind": "resume", "status": "intent_before_qsub",
                            "job_name": control.job_name(1), "attempt": 1}}
        with self.assertRaisesRegex(RuntimeError, "Do not resubmit"):
            control.recover_intent(remote, object(), state)
        self.assertEqual(len(remote.commands), 1)
        self.assertIn("qselect -x", remote.commands[0])
        self.assertNotIn("qsub", remote.commands[0])

    def test_submit_returns_existing_handle_without_remote_calls(self):
        class NeverRemote:
            def command(self, *_args, **_kwargs):
                raise AssertionError("existing submission must not issue a remote command")

        state = {"handle": {"job_id": "25900001.pbs101"}}
        with patch.object(control, "load_protocol", return_value={"protocol": control.PROTOCOL_ID}), \
             patch.object(control, "read_state", return_value=(state, {})):
            result = control.submit(NeverRemote(), object())
        self.assertTrue(result["existing"])


class RunScriptTests(unittest.TestCase):
    def test_allocation_gpu_and_completion_guards_are_present(self):
        source = Path(__file__).with_name("run.pbs").read_text(encoding="utf-8")
        for expected in (
            "#PBS -q normal",
            "#PBS -l select=1:ncpus=16:mem=110gb:ngpus=1",
            "#PBS -l walltime=04:00:00",
            'test -n "${PBS_JOBID:-}"',
            'test -s "$PBS_NODEFILE"',
            "Refusing login-node execution",
            "Current host is not in the approved PBS allocation",
            "PBS job is not in running state",
            'export CUDA_VISIBLE_DEVICES="$GPU_UUID"',
            "sleep 15",
            'run_python "$ARTIFACTS/worker.py" --protocol "$ARTIFACTS/protocol.json"',
            'summary.get("complete") is not True',
            'summary.get("protocol") != sys.argv[2]',
            'touch "$ARTIFACTS/PHASE_COMPLETE"',
        ):
            self.assertIn(expected, source)
        self.assertIn("FW_SOFT_DEADLINE_SECONDS=14100", source)

    def test_runtime_wrapper_is_copied_from_successful_learned_run(self):
        copied = Path(__file__).with_name("environment.sh").read_bytes().replace(b"\r\n", b"\n")
        reference = Path(__file__).parents[1] / "fastwam-vq-learned-r128-c8" / "environment.sh"
        self.assertEqual(copied, reference.read_bytes().replace(b"\r\n", b"\n"))


if __name__ == "__main__":
    unittest.main()
