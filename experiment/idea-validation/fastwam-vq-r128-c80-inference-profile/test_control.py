import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("inference_profile_control", HERE / "control.py")
control = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(control)


def protocol():
    return {
        "protocol": control.PROTOCOL_ID,
        "remote_root": control.ROOT,
        "allocation": dict(control.ALLOCATION),
        "sources": {"checkpoint": "frozen-best-c80-r128"},
    }


def receipt():
    job_id = "401234.pbs101"
    return {
        "handle": {
            "remote_root": control.ROOT, "job_id": job_id,
            "job_name": control.JOB_NAME, "protocol": control.PROTOCOL_ID,
        },
        "scheduler": {
            "query_rc": 0, "job_state": "F", "Exit_status": "0",
            "Job_Name": control.JOB_NAME, "queue": control.EXECUTION_QUEUE,
            "Resource_List.walltime": control.WALLTIME,
        },
        "protocol_matches": True,
        "artifact": {"exit_code": "0", "pipeline_complete": True},
        "summary": {
            "protocol": control.PROTOCOL_ID, "pbs_jobid": job_id,
            "complete": True, "export_complete": True, "parity_passed": True,
            "profile_complete": True, "new_training_updates": 0,
        },
    }


class Attrs:
    def __init__(self, size):
        self.st_size = size


class Stream:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, count):
        return self.data[:count]


class SFTP:
    def __init__(self, data):
        self.data = data
        self.reads = 0

    def stat(self, _path):
        return Attrs(len(self.data))

    def file(self, _path, _mode):
        self.reads += 1
        return Stream(self.data)


class ControlTests(unittest.TestCase):
    def test_protocol_and_single_gpu_qsub_request(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "protocol.json"
            path.write_text(json.dumps(protocol()), encoding="utf-8")
            with patch.object(control, "HERE", Path(temp)):
                self.assertEqual(control.load_protocol()["allocation"], control.ALLOCATION)
        command = control._qsub_command()
        self.assertIn("-q normal", command)
        self.assertIn("select=1:ncpus=16:mem=120gb:ngpus=1", command)
        self.assertIn("walltime=04:00:00", command)
        self.assertIn(control.JOB_NAME, command)

    def test_complete_gate_requires_all_inference_only_receipts(self):
        value = receipt()
        self.assertTrue(control.complete_gate(value))
        for key, bad in (
            ("export_complete", False), ("parity_passed", False),
            ("profile_complete", False), ("new_training_updates", 1),
        ):
            candidate = receipt()
            candidate["summary"][key] = bad
            self.assertFalse(control.complete_gate(candidate), key)

    def test_complete_gate_requires_matching_terminal_job_and_marker(self):
        value = receipt()
        value["scheduler"]["job_state"] = "R"
        self.assertFalse(control.complete_gate(value))
        value = receipt()
        value["scheduler"]["queue"] = "normal"
        self.assertFalse(control.complete_gate(value))
        value = receipt()
        value["artifact"]["pipeline_complete"] = False
        self.assertFalse(control.complete_gate(value))
        value = receipt()
        value["summary"]["pbs_jobid"] = "401235.pbs101"
        self.assertFalse(control.complete_gate(value))

    def test_remote_read_checks_size_before_transfer(self):
        remote = SFTP(b"x" * (control.LIMIT + 1))
        with self.assertRaisesRegex(RuntimeError, "256 KiB"):
            control._small_bytes(remote, "/remote/large.json")
        self.assertEqual(remote.reads, 0)

    def test_small_remote_json_is_bounded_and_decoded(self):
        remote = SFTP(json.dumps({"phase": "profile"}).encode("utf-8"))
        self.assertEqual(control._json(remote, "/remote/progress.json"), {"phase": "profile"})
        self.assertEqual(remote.reads, 1)

    def test_unresolved_intent_never_reissues_qsub(self):
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, "protocol.json").write_text(json.dumps(protocol()), encoding="utf-8")
            state = {"intent": {"status": "qsub_outcome_unknown"}}
            with (patch.object(control, "HERE", Path(temp)),
                  patch.object(control, "_mkdirs"),
                  patch.object(control, "_resolve", return_value=(None, state)),
                  patch.object(control, "_require_uploaded") as require_uploaded,
                  patch.object(control.remote, "command") as command):
                with self.assertRaisesRegex(RuntimeError, "qsub was not repeated"):
                    control.submit(object())
            require_uploaded.assert_not_called()
            command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
