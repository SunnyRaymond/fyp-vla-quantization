"""Standard-library regression tests for the transport PBS controller."""
from contextlib import nullcontext
import errno
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import control


def passing_observation():
    job_id = "25920001.pbs101"
    summary = {
        "protocol": control.PROTOCOL_ID, "status": "complete", "complete": True,
        "pbs_jobid": job_id, "target_linears": 614, "new_teacher_queries": 0,
        "source_dual_binding": True, "allocation_gpus": 1,
        "execution_amendment": control.load_execution(),
        "reused_reference": {"Uncomp8": "Joint8", "job": "25912619.pbs101"},
        "encoding_scope": "decoded_weight_transport_reference",
        "probe_counts": {arm: {"calibration": 80, "heldout": 80} for arm in control.PROBE_ARMS},
        "endpoints": {arm: {"calibration": 80, "heldout": 80, "updates": 8, "exposures": 80}
                      for arm in control.ARMS},
        "gates": {"source": True, "fp32_invariance": True, "gradient_coupling": True,
                  "initial_parity": True, "frozen": True,
                  "formal_parity_max_error": 8e-7, "deployment_reload_max_error": 8e-7,
                  "orthogonality_max_error": 4e-5},
        "comparison_file": "results/comparison.json", "comparison_complete": True,
    }
    stage_summaries = {
        stage: {"protocol": control.PROTOCOL_ID, "status": "complete", "complete": True,
                "pbs_jobid": job_id, "units": units}
        for stage, units in control.STAGE_UNITS.items()
    }
    stage_progress = {
        stage: {"protocol": control.PROTOCOL_ID, "status": "complete", "pbs_jobid": job_id}
        for stage in control.STAGES
    }
    arm_summaries = {
        arm: {"protocol": control.PROTOCOL_ID, "arm": arm, "status": "complete", "complete": True,
              "pbs_jobid": job_id, "updates": 8, "exposures": 80, "calibration": 80,
              "heldout": 80, "all_parameters_frozen": True, "formal_infer_max_abs": 8e-7,
              "initial_parity_max_abs": 8e-7, "deployment_reload_max_abs": 8e-7,
              "orthogonality": {"action": 4e-5, "video": 4e-5}}
        for arm in control.ARMS
    }
    return {
        "handle": {"job_id": job_id, "job_name": control.JOB_NAME, "attempt": 0,
                   "remote_root": control.ROOT, "protocol": control.PROTOCOL_ID},
        "scheduler": {"job_id": job_id, "query_rc": 0, "job_state": "F", "Exit_status": "0",
                      "queue": "g1", "exec_host": "node/0"},
        "artifact_exit_code": 0, "phase_complete": True,
        "summary": summary, "worker_summary": dict(summary),
        "progress": {"protocol": control.PROTOCOL_ID, "stage": "complete", "status": "complete",
                     "complete": True, "pbs_jobid": job_id},
        "receipts": {name: {"complete": True, "passed": True} for name in control.RECEIPTS},
        "stage_summaries": stage_summaries, "stage_progress": stage_progress,
        "arm_summaries": arm_summaries,
        "arm_progress": {arm: {"protocol": control.PROTOCOL_ID, "pbs_jobid": job_id,
                                "status": "complete", "updates": 8, "exposures": 80}
                          for arm in control.ARMS},
        "checkpoint_sizes": {arm: {"state.pt": 1024, "deployment.pt": 2048} for arm in control.ARMS},
        "comparison_size": 4096,
        "result_files": {name: True for name in (*control.RESULT_FILES, "comparison.json")},
        "result_json": {"training.json": {"protocol": control.PROTOCOL_ID}},
        "log_tail": "completed\n",
    }


class ProtocolAndVerificationTests(unittest.TestCase):
    def test_frozen_metadata_and_allocation(self):
        protocol = control.load_protocol()
        self.assertEqual(protocol["protocol"], control.PROTOCOL_ID)
        self.assertEqual(control.ROOT, "/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-transport-20261022")
        self.assertEqual(control.JOB_NAME, "fwvr128tr")
        self.assertEqual(control.SELECT, "1:ncpus=16:mem=110gb:ngpus=1")
        self.assertEqual(control.WALLTIME, "04:00:00")
        self.assertEqual(control.ARMS, ('Comp8',))
        self.assertEqual(control.STAGE_UNITS, {"probe_0": 80, "probe_1": 80, "gate_0": 1, "finalize_0": 1})

    def test_full_terminal_transport_fixture_passes(self):
        observation = passing_observation()
        self.assertEqual(control.verification_failures(observation, control.load_protocol()), [])

    def test_wrong_protocol_handle_and_incomplete_counts_fail_verification(self):
        cases = (
            ("protocol", lambda row: row["summary"].update(protocol="wrong-protocol")),
            ("handle", lambda row: row["summary"].update(pbs_jobid="25920099.pbs101")),
            ("handle protocol", lambda row: row["handle"].update(protocol="wrong-protocol")),
            ("probe_counts", lambda row: row["summary"]["probe_counts"][control.PROBE_ARMS[0]].update(heldout=79)),
            ("endpoint counts", lambda row: row["summary"]["endpoints"]["Comp8"].update(calibration=79)),
        )
        for label, mutate in cases:
            with self.subTest(label=label):
                observation = passing_observation()
                mutate(observation)
                failures = control.verification_failures(observation, control.load_protocol())
                self.assertTrue(failures, label)

    def test_deployment_reload_error_is_required_and_bounded(self):
        cases = (
            ("root gate missing", lambda row: row["summary"]["gates"].pop("deployment_reload_max_error")),
            ("root gate over limit", lambda row: row["summary"]["gates"].update(deployment_reload_max_error=1.1e-6)),
            ("arm field missing", lambda row: row["arm_summaries"]["Comp8"].pop("deployment_reload_max_abs")),
            ("arm field over limit", lambda row: row["arm_summaries"]["Comp8"].update(deployment_reload_max_abs=1.1e-6)),
        )
        for label, mutate in cases:
            with self.subTest(label=label):
                observation = passing_observation()
                mutate(observation)
                failures = control.verification_failures(observation, control.load_protocol())
                self.assertTrue(any("deployment" in failure.lower() for failure in failures), label)

    def test_small_file_reader_rejects_oversized_remote_control_file(self):
        sftp = Mock()
        sftp.stat.return_value = SimpleNamespace(st_size=control.LIMIT)
        with self.assertRaisesRegex(RuntimeError, "Refusing a remote control/result file"):
            control.read_small_bytes(sftp, "/remote/summary.json")
        sftp.file.assert_not_called()

    def test_pull_reads_small_results_and_never_opens_checkpoints(self):
        class MissingFilesSftp:
            def __init__(self):
                self.opened = []

            def stat(self, path):
                raise OSError(errno.ENOENT, "missing", path)

            def file(self, path, mode):
                self.opened.append(path)
                raise AssertionError("unexpected read of a missing path")

        remote = Mock()
        sftp = MissingFilesSftp()
        remote.sftp_client.return_value = nullcontext(sftp)
        result = passing_observation()
        with tempfile.TemporaryDirectory() as temporary, patch.object(control, "HERE", Path(temporary)), \
                patch.object(control, "status", return_value=result):
            pulled = control.pull(remote, object())
        self.assertEqual(pulled["metadata"]["comparison_bytes"], result["comparison_size"])
        self.assertFalse(any(path.endswith(("state.pt", "deployment.pt")) for path in sftp.opened))


class ResumeSafetyTests(unittest.TestCase):
    def test_user_cancelled_job_can_resume_only_with_terminal_receipt_and_checkpoint(self):
        row = dict(handle={'job_id': '25919309.pbs101'},
            scheduler={'job_state': 'F', 'query_rc': 0, 'Exit_status': '141'},
            artifact_exit_code=None, phase_complete=False, summary={}, progress={},
            checkpoint_sizes={'Comp8': {'state.pt': 41978139}})
        self.assertEqual(control.resume_failures(row, control.load_protocol()), [])
        row['scheduler']['job_state'] = 'R'
        self.assertTrue(control.resume_failures(row, control.load_protocol()))
        row['scheduler']['job_state'] = 'F'
        row['handle']['job_id'] = '25920000.pbs101'
        self.assertTrue(control.resume_failures(row, control.load_protocol()))

    def _resume_with_state(self, job_state):
        job_id = "25920001.pbs101"
        handle = {"job_id": job_id, "job_name": control.JOB_NAME, "attempt": 0,
                  "remote_root": control.ROOT, "protocol": control.PROTOCOL_ID, "status": "confirmed"}
        state = {"handle": dict(handle)}
        observation = {
            "handle": handle,
            "scheduler": {"job_id": job_id, "query_rc": 0, "job_state": job_state,
                          "Exit_status": "1" if job_state == "F" else None},
            "artifact_exit_code": 1 if job_state == "F" else None,
            "phase_complete": False, "progress": {"stage": "probe_0", "complete": False},
            "summary": {"status": "failed", "complete": False}, "checkpoint_sizes": {},
            "arm_progress": {}, "result_json": {},
        }
        remote = Mock()
        remote.sftp_client.return_value = nullcontext(object())
        remote.command.return_value = (0, "25920002.pbs101")
        with patch.object(control, "read_state", return_value=(state, {})), \
                patch.object(control, "recover_intent", return_value=None), \
                patch.object(control, "observe", return_value=(observation, state)), \
                patch.object(control, "save_submission_state"), \
                patch.object(control, "remote_protocol_matches"), \
                patch.object(control, "assert_remote_files"), \
                patch.object(control, "transfer_files", return_value=[{"name": "worker.py", "bytes": 5}]) as transfer, \
                patch.object(control, "named_jobs", return_value=[]):
            if job_state == "R":
                with self.assertRaisesRegex(RuntimeError, "Resume refused"):
                    control.resume(remote, object())
                transfer.assert_not_called()
                return None
            result = control.resume(remote, object())
            transfer.assert_called_once()
            self.assertFalse(transfer.call_args.kwargs["include_protocol"])
            return result

    def test_live_job_refuses_resume_before_upload(self):
        self._resume_with_state("R")

    def test_confirmed_terminal_failure_allows_code_upload_and_new_attempt(self):
        result = self._resume_with_state("F")
        self.assertTrue(result["resumed"])
        self.assertEqual(result["from_job_id"], "25920001.pbs101")
        self.assertEqual(result["handle"]["job_name"], "fwvr128tr-a1")

    def test_unknown_qsub_intent_only_recovers_one_exact_history_match(self):
        state = {"intent": {"kind": "submit", "status": "qsub_outcome_unknown", "job_name": control.JOB_NAME,
                            "attempt": 0}}
        remote = Mock()
        with patch.object(control, "named_jobs", return_value=[]), patch.object(control, "save_submission_state"):
            with self.assertRaisesRegex(RuntimeError, "outcome is unresolved"):
                control.recover_intent(remote, object(), state)
        self.assertNotIn("handle", state)


if __name__ == "__main__":
    unittest.main()
