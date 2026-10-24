import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("qclip_control_tested", HERE / "control.py")
control = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(control)


class _Attrs:
    def __init__(self, size):
        self.st_size = size


class _Stream:
    def __init__(self, data):
        self.data = data
        self.read_count = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, count):
        self.read_count += 1
        return self.data[:count]


class _SFTP:
    def __init__(self, files=None):
        self.files = files or {}
        self.streams = []

    def stat(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return _Attrs(len(self.files[path]))

    def file(self, path, _mode):
        stream = _Stream(self.files[path])
        self.streams.append(stream)
        return stream


def _split(contexts, queries):
    metric_values = {metric: 0.25 for metric in control.METRICS}
    per_suite = {}
    for suite in control.PROTOCOL["suites"]:
        suite_contexts = contexts // 4
        per_suite[suite] = {"contexts": suite_contexts, "queries": queries // 4, **metric_values}
    return {"queries": queries, "contexts": contexts, "by_suite": per_suite,
            "overall": dict(metric_values)}


def complete_receipt():
    job_id = "401234.pbs101"
    arms = {
        arm: {"calibration": _split(80, 80), "heldout": _split(40, 80)}
        for arm in ("q0", "qclip")
    }
    summary = {
        "protocol": control.PROTOCOL["protocol"], "pbs_jobid": job_id,
        "status": "complete", "complete": True, "updates": 80, "exposures": 80,
        "target_linears": 614, "heldout_queries_per_arm": 80,
        "calibration_queries_per_arm": 80, "new_teacher_queries": 0,
        "closedloop_episodes": 0, "weights_frozen": True, "rotation_frozen": True,
        "initial_parity_max_abs": 0.0, "formal_parity_max_abs": 1e-7, "arms": arms,
    }
    return {
        "winner": {"protocol": control.PROTOCOL["protocol"], "winner_job_id": job_id,
                   "winner_job_name": control.JOB_NAMES["short"], "status": "complete"},
        "jobs": {"short": {
            "handle": {"job_id": job_id, "job_name": control.JOB_NAMES["short"],
                       "protocol": control.PROTOCOL["protocol"], "walltime": "01:59:00"},
            "scheduler": {"query_rc": 0, "job_state": "F", "Exit_status": "0",
                          "Job_Name": control.JOB_NAMES["short"], "queue": "gdev",
                          "Resource_List.walltime": "01:59:00",
                          "Resource_List.select": "1:ncpus=16:mem=110gb:ngpus=1"},
            "artifact": {"exit_code": "0", "PHASE_COMPLETE": True,
                         "RUN_PHASE": {"protocol": control.PROTOCOL["protocol"],
                                       "pbs_jobid": job_id, "phase": "complete"}},
        }},
        "summary": summary,
    }


class ControlTests(unittest.TestCase):
    def test_protocol_and_two_job_resources_are_frozen(self):
        protocol = control.load_protocol()
        self.assertEqual(protocol["allocation"]["queue"], "normal")
        self.assertEqual(protocol["allocation"]["gpus"], 1)
        self.assertEqual(control.SPECS["short"]["walltime"], "01:59:00")
        self.assertEqual(control.SPECS["long"]["walltime"], "12:00:00")
        self.assertIn("-q normal", control._qsub_command("short"))
        self.assertIn("select=1:ncpus=16:mem=110gb:ngpus=1", control._qsub_command("long"))
        self.assertIn("walltime=12:00:00", control._qsub_command("long"))

    def test_complete_gate_accepts_frozen_pair_and_four_suite_aggregates(self):
        self.assertTrue(control.complete_gate(complete_receipt()))

    def test_complete_gate_rejects_wrong_coverage_metric_or_parity(self):
        value = complete_receipt()
        value["summary"]["arms"]["qclip"]["heldout"]["queries"] = 79
        self.assertFalse(control.complete_gate(value))
        value = complete_receipt()
        value["summary"]["arms"]["q0"]["calibration"]["by_suite"][control.PROTOCOL["suites"][0]]["contexts"] = 19
        self.assertFalse(control.complete_gate(value))
        value = complete_receipt()
        value["summary"]["arms"]["qclip"]["heldout"]["overall"]["first10motor_rmse"] = float("nan")
        self.assertFalse(control.complete_gate(value))
        value = complete_receipt()
        value["summary"]["formal_parity_max_abs"] = 1.01e-6
        self.assertFalse(control.complete_gate(value))

    def test_one_gpu_route_is_validated_without_assuming_a_partition(self):
        value = complete_receipt()
        self.assertTrue(control.complete_gate(value))
        value["jobs"]["short"]["scheduler"]["queue"] = "g2"
        self.assertTrue(control.complete_gate(value))
        value["jobs"]["short"]["scheduler"]["queue"] = "unexpected"
        self.assertFalse(control.complete_gate(value))

    def test_remote_read_checks_size_before_transferring_bytes(self):
        remote = _SFTP({"/large.json": b"x" * (control.LIMIT + 1)})
        with self.assertRaisesRegex(RuntimeError, "256 KiB"):
            control._small_bytes(remote, "/large.json")
        self.assertEqual(remote.streams, [])

    def test_unknown_qsub_intent_is_recovered_without_new_qsub(self):
        state = {"jobs": {"short": {"intent": {"status": "qsub_outcome_unknown"}}}}
        with patch.object(control, "_recover", side_effect=RuntimeError("unresolved")) as recover, \
             patch.object(control.remote, "command") as command:
            with self.assertRaisesRegex(RuntimeError, "unresolved"):
                control._submit_one(object(), state, "short")
        recover.assert_called_once()
        command.assert_not_called()

    def test_qstat_failure_is_never_treated_as_terminal(self):
        row = {"scheduler": {"query_rc": 1, "job_state": "F"}}
        self.assertEqual(control._job_status(row), "query_inconclusive")

    def test_reconcile_only_cancels_the_verified_sibling(self):
        value = complete_receipt()
        long_handle = {"job_id": "401235.pbs101", "job_name": control.JOB_NAMES["long"],
                       "protocol": control.PROTOCOL["protocol"], "walltime": "12:00:00"}
        value["jobs"]["long"] = {"handle": long_handle, "status": "queued",
                                  "scheduler": {"query_rc": 0, "job_state": "Q",
                                                "Job_Name": control.JOB_NAMES["long"]}, "artifact": {}}
        with patch.object(control, "status", return_value=value), \
             patch.object(control.remote, "command", return_value=(0, "deleted")) as command:
            result = control.reconcile(object())
        command.assert_called_once()
        self.assertIn("401235.pbs101", result["cancelled"])
        value["jobs"]["long"]["scheduler"]["query_rc"] = 1
        with patch.object(control, "status", return_value=value), \
             patch.object(control.remote, "command") as command:
            result = control.reconcile(object())
        command.assert_not_called()
        self.assertEqual(result["cancelled"], [])

    def test_resume_requires_terminal_exit_75_checkpoint_receipt(self):
        value = complete_receipt()
        winner = value["winner"]
        winner.update(status="interrupted")
        value["jobs"]["short"]["scheduler"].update(job_state="F", Exit_status="75")
        value["failure"] = {"protocol": control.PROTOCOL["protocol"], "pbs_jobid":"401234.pbs101",
                            "error":"TimeoutError", "checkpointed":True}
        with patch.object(control, "_has_checkpoint_or_endpoint", return_value=True), \
             patch.object(control.remote, "sftp_client") as sftp:
            sftp.return_value.__enter__.return_value = object()
            result = control._resume_ready(object(), value)
        self.assertEqual(result["previous_label"], "short")
        value["jobs"]["short"]["scheduler"]["Exit_status"] = "0"
        with patch.object(control, "_has_checkpoint_or_endpoint", return_value=True), \
             patch.object(control.remote, "sftp_client") as sftp:
            sftp.return_value.__enter__.return_value = object()
            with self.assertRaisesRegex(RuntimeError, "Exit_status=75"):
                control._resume_ready(object(), value)

    def test_pbs_guard_and_source_staging_are_present(self):
        script = (HERE / "run.pbs").read_text(encoding="utf-8")
        self.assertLess(script.index("qstat -xf"), script.index('cp "$PARENT_RUNTIME/worker.py"'))
        self.assertIn('cp "$PARENT_RUNTIME/worker.py" "$JOB_ARTIFACTS/q0_worker.py"', script)
        self.assertIn('for file in source_worker.py rotation_bank.py', script)
        self.assertIn('cp "$PARENT_RUNTIME/protocol.json" "$JOB_ARTIFACTS/parent_protocol.json"', script)
        self.assertIn("FW_SOFT_DEADLINE_SECONDS=$((10#$wall_h", script)
        self.assertIn('sleep 15', script)


if __name__ == "__main__":
    unittest.main()
