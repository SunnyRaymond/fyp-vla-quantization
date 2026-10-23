import importlib.util
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("loss_pair_control_tested", HERE / "control.py")
control = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(control)


class _Attrs:
    def __init__(self, size):
        self.st_size = size


class _Stream:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, count):
        return self.data[:count]


class _SFTP:
    def __init__(self, data):
        self.data = data
        self.reads = 0

    def stat(self, _path):
        return _Attrs(len(self.data))

    def file(self, _path, _mode):
        self.reads += 1
        return _Stream(self.data)


def summary(arm, job_id="401234.pbs101"):
    return {
        "protocol": control.PROTOCOL["protocol"], "arm": arm, "pbs_jobid": job_id,
        "complete": True, "status": "complete", "updates": 8, "exposures": 80,
        "target_linears": 614, "frozen_rotation": True,
        "initial_parity_max_abs": 0.0, "formal_parity_max_abs": 0.0,
        "final_anchor": True, "new_teacher_queries": 0,
        "beta": 0.0 if arm == "az" else 0.0125,
    }


def complete_receipt():
    job_id = "401234.pbs101"
    summaries = {arm: summary(arm, job_id) for arm in control.ARMS}
    arms = {
        arm: {"progress": {"phase": "complete", "updates": 8, "exposures": 80},
              "training": {"updates": 8, "exposures": 80}, "summary": receipt,
              "setup_checks": {"passed": True}, "failure": None}
        for arm, receipt in summaries.items()
    }
    return {
        "handle": {
            "job_id": job_id, "job_name": control.JOB_NAME,
            "gpus": 2, "walltime": "12:00:00",
        },
        "scheduler": {
            "query_rc": 0, "job_state": "F", "Exit_status": "0",
            "Job_Name": control.JOB_NAME, "queue": "g2", "Resource_List.walltime": "12:00:00",
        },
        "protocol_matches": True,
        "artifact": {
            "exit_code": "0", "PHASE_COMPLETE": True,
            "RUN_PHASE": {"protocol": control.PROTOCOL["protocol"], "pbs_jobid": job_id, "phase": "complete"},
        },
        "results": {
            "summary": {
                "protocol": control.PROTOCOL["protocol"], "pbs_jobid": job_id,
                "complete": True, "status": "complete", "new_teacher_queries": 0,
                "arms": copy.deepcopy(summaries),
            },
            "arms": arms,
        },
    }


class ControlTests(unittest.TestCase):
    def test_frozen_calibration_resources_and_job_name(self):
        protocol = control.load_protocol()
        self.assertEqual(protocol["allocation"]["execution_queue"], "g2")
        self.assertEqual(protocol["allocation"]["gpus"], 2)
        self.assertEqual(protocol["allocation"]["walltime"], "12:00:00")
        self.assertEqual(control.JOB_NAME, "r128lp0001")
        command = control._qsub_command()
        self.assertIn("-q normal", command)
        self.assertIn("select=1:ncpus=32:mem=220gb:ngpus=2", command)
        self.assertIn("walltime=12:00:00", command)

    def test_complete_gate_accepts_both_actual_summaries(self):
        value = complete_receipt()
        self.assertTrue(control.complete_gate(value))

    def test_complete_gate_rejects_summary_from_another_job(self):
        value = complete_receipt()
        value["results"]["arms"]["reader_kl"]["summary"]["pbs_jobid"] = "401235.pbs101"
        self.assertFalse(control.complete_gate(value))

    def test_complete_gate_rejects_root_summary_that_does_not_match_actual(self):
        value = complete_receipt()
        value["results"]["summary"]["arms"]["az"]["beta"] = 0.25
        self.assertFalse(control.complete_gate(value))

    def test_complete_gate_rejects_wrong_beta_and_parity(self):
        value = complete_receipt()
        value["results"]["arms"]["reader_kl"]["summary"]["beta"] = 0.0
        self.assertFalse(control.complete_gate(value))
        value = complete_receipt()
        value["results"]["arms"]["az"]["summary"]["formal_parity_max_abs"] = 1.01e-6
        self.assertFalse(control.complete_gate(value))

    def test_complete_gate_requires_real_terminal_success(self):
        value = complete_receipt()
        value["scheduler"]["job_state"] = "R"
        self.assertFalse(control.complete_gate(value))
        value = complete_receipt()
        value["scheduler"]["queue"] = "normal"
        self.assertFalse(control.complete_gate(value))

    def test_remote_read_is_bounded_before_file_transfer(self):
        oversized = _SFTP(b"x" * (control.LIMIT + 1))
        with self.assertRaisesRegex(RuntimeError, "256 KiB"):
            control._small_bytes(oversized, "/remote/large.json")
        self.assertEqual(oversized.reads, 0)

    def test_small_remote_json_is_read_and_decoded(self):
        remote = _SFTP(json.dumps({"phase": "complete"}).encode("utf-8"))
        self.assertEqual(control._json(remote, "/remote/phase.json"), {"phase": "complete"})
        self.assertEqual(remote.reads, 1)

    def test_compact_status_surfaces_reader_preparation_progress(self):
        value = {
            "handle": {"job_id": "401234.pbs101"},
            "scheduler": {"job_state": "R", "queue": "g2", "query_rc": 0},
            "status": "running", "completion_passed": False,
            "results": {
                "arms": {
                    "reader_kl": {"progress": {"phase": "initial_reader_calibration", "initial_cases": 41}},
                    "az": {"progress": {"phase": "training", "updates": 3, "exposures": 30}},
                },
                "reader_preparation": None,
            },
        }
        compact = control._compact(value)
        self.assertEqual(compact["arms"]["az"]["phase"], "training")
        self.assertEqual(compact["reader_preparation"]["initial_cases"], 41)

    def test_unresolved_submission_intent_never_reissues_qsub(self):
        state = {"intent": {"status": "qsub_outcome_unknown"}}
        with (patch.object(control, "_mkdirs"),
              patch.object(control, "_resolve", return_value=(None, state)),
              patch.object(control, "_require_uploaded") as require_uploaded,
              patch.object(control.remote, "command") as remote_command):
            with self.assertRaisesRegex(RuntimeError, "qsub was not repeated"):
                control.submit(object())
        require_uploaded.assert_not_called()
        remote_command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
