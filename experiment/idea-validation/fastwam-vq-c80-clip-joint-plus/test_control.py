"""Standard-library gates for C80 PBS control and durable submit behavior."""
import io
import json
import re
import unittest
from contextlib import contextmanager
from unittest import mock

import control


class MemorySftp:
    def __init__(self):
        self.files = {}

    def file(self, path, mode):
        if "r" in mode:
            if path not in self.files:
                raise FileNotFoundError(2, "missing", path)
            return io.BytesIO(self.files[path])
        owner = self

        class Writer(io.BytesIO):
            def __exit__(self, exc_type, exc_value, traceback):
                if exc_type is None:
                    owner.files[path] = self.getvalue()
                self.close()
                return False

        return Writer()

    def posix_rename(self, source, target):
        self.files[target] = self.files.pop(source)

    def stat(self, path):
        if path not in self.files:
            raise FileNotFoundError(2, "missing", path)
        return object()


class ControlTests(unittest.TestCase):
    def test_calibration_does_not_freeze_unsubmitted_evaluation_sources(self):
        fs = MemorySftp()
        records = {"schema_version": 1, "units": {
            "clip_only": {"attempts": [{"attempt": 1}], "latest_attempt": 1}}}
        fs.files[control.CONTROL + "/handles.json"] = json.dumps(records).encode()
        fs.files[control.ROOT + "/protocol.json"] = json.dumps(control.PROTOCOL).encode()

        @contextmanager
        def sftp_client(_transport):
            yield fs

        with mock.patch.object(control, "_mkdirs"), \
             mock.patch.object(control.remote, "sftp_client", side_effect=sftp_client):
            self.assertEqual(control.upload(object(), ["run.pbs"]), ["run.pbs"])
            with self.assertRaisesRegex(RuntimeError, "source is frozen"):
                control.upload(object(), ["calibrate.py"])
            records["units"]["clip_only_eval"] = {
                "attempts": [{"attempt": 1}], "latest_attempt": 1}
            fs.files[control.CONTROL + "/handles.json"] = json.dumps(records).encode()
            with self.assertRaisesRegex(RuntimeError, "source is frozen"):
                control.upload(object(), ["evaluate.py"])

    def test_job_names_are_unique_and_within_pbs_limit(self):
        names = [control._job_name(unit, attempt)
                 for unit in control.UNITS for attempt in (1, 2, 80)]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(len(name) <= 15 for name in names))
        self.assertTrue(all(control.JOB_NAME.fullmatch(name) for name in names))

    def test_embedded_run_python_blocks_compile(self):
        source = (control.HERE / "run.pbs").read_text(encoding="utf-8")
        blocks = re.findall(r"<<'([A-Z_]+)'\n(.*?)\n\1", source, re.S)
        self.assertGreaterEqual(len(blocks), 4)
        for label, code in blocks:
            with self.subTest(block=label):
                compile(code, f"run.pbs:{label}", "exec")

    def test_backward_check_requires_exact_implementation_receipt(self):
        valid = {"passed": True, "implementation": "bounded_vector_chunks_v2",
                 "max_vectors_per_chunk": 1048576}
        self.assertTrue(control._backward_passed(valid))
        for field, wrong in (("passed", False), ("implementation", "other"),
                             ("max_vectors_per_chunk", 1048575)):
            changed = dict(valid, **{field: wrong})
            self.assertFalse(control._backward_passed(changed))

    def test_training_completion_requires_exact_c80_and_parity(self):
        ids = control.PROTOCOL["calibration"]["case_ids"]
        order = control.PROTOCOL["calibration"].get("update_order", ids)
        job_id, arm = "12345.pbs101", "clip_only"
        handle = {"job_id": job_id, "job_name": "c80clip0001"}
        summary = {"protocol": control.PROTOCOL["protocol"], "jobid": job_id,
                   "arm": arm, "status": "complete", "complete": True,
                   "updates": 80, "target_linears": 614,
                   "calibration_ids": ids, "update_order": order}
        progress = {"arm": arm, "status": "complete", "complete": True, "updates": 80}
        receipt = dict(summary, actual_infer_parity={"passed": True, "max_abs": 0.0})
        value = {"query_rc": 0, "job_state": "F", "Job_Name": handle["job_name"],
                 "Exit_status": "0", "exit_code.txt": "0", "TRAINING_COMPLETE": True,
                 "TRAINING_CHECKPOINTED": False, "calibration_summary.json": summary,
                 "progress.json": progress, "training_receipt.json": receipt,
                 "deployment_exists": True, "reference_action.json": {"passed": True},
                 "backward_check.json": {"passed": True,
                    "implementation": control.BACKWARD_IMPLEMENTATION,
                    "max_vectors_per_chunk": control.BACKWARD_MAX_VECTORS}}
        self.assertEqual(control._training_complete(handle, value, arm)["updates"], 80)
        changed = dict(value)
        changed["calibration_summary.json"] = dict(summary, updates=79)
        with self.assertRaises(RuntimeError):
            control._training_complete(handle, changed, arm)
        changed = dict(value)
        changed["backward_check.json"] = {"passed": False}
        with self.assertRaises(RuntimeError):
            control._training_complete(handle, changed, arm)

    def test_joint_resume_requires_rc75_and_matching_checkpointed_summaries(self):
        job_id, arm = "12345.pbs101", "joint"
        handle = {"job_id": job_id, "job_name": "c80joint0001"}
        checkpoint = control.RESULTS + "/training/joint/checkpoint.pt"
        summary = {"protocol": control.PROTOCOL["protocol"], "jobid": job_id,
                   "arm": arm, "status": "checkpointed", "complete": False,
                   "updates": 12, "target_linears": 614,
                   "calibration_ids": control.PROTOCOL["calibration"]["case_ids"],
                   "update_order": control.PROTOCOL["calibration"].get(
                       "update_order", control.PROTOCOL["calibration"]["case_ids"])}
        progress = {"arm": arm, "status": "checkpointed", "complete": False,
                    "updates": 12, "checkpoint": checkpoint}
        value = {"query_rc": 0, "job_state": "F", "Job_Name": handle["job_name"],
                 "Exit_status": "75", "exit_code.txt": "75",
                 "TRAINING_CHECKPOINTED": True, "TRAINING_COMPLETE": False,
                 "calibration_summary.json": summary, "progress.json": progress,
                 "checkpoint_exists": True}
        self.assertEqual(control._training_checkpointed(handle, value, arm)["updates"], 12)
        update80 = dict(value)
        update80["calibration_summary.json"] = dict(summary, updates=80)
        update80["progress.json"] = dict(progress, updates=80)
        self.assertEqual(control._training_checkpointed(handle, update80, arm)["updates"], 80)
        resumed_handle = dict(handle, source_job_id="12344.pbs101")
        with self.assertRaisesRegex(RuntimeError, "resume source"):
            control._training_checkpointed(resumed_handle, value, arm)
        value["Exit_status"] = "1"
        with self.assertRaises(RuntimeError):
            control._training_checkpointed(handle, value, arm)

    def test_evaluation_checkpoint_requires_exact_worker_prefixes(self):
        arm, job_id = "clip_only", "12345.pbs101"
        handle = {"job_id": job_id, "job_name": "c80eclip0001"}
        workers = {}
        counts = (1, 1, 0, 0)
        for worker, count in enumerate(counts):
            indices = list(range(worker, 280, 4))
            status = "checkpointed"
            receipt = {"protocol": control.PROTOCOL["protocol"], "arm": arm,
                       "worker_id": worker, "status": status, "complete": False,
                       "expected": 70, "episodes": count, "indices": indices,
                       "completed_indices": indices[:count], "soft_deadline_reached": True}
            workers[str(worker)] = {"progress": receipt, "summary": receipt}
        value = {"query_rc": 0, "job_state": "F", "Job_Name": handle["job_name"],
                 "Exit_status": "0", "exit_code.txt": "0",
                 "PIPELINE_CHECKPOINTED": True, "PIPELINE_COMPLETE": False,
                 "summary.json": {"protocol": control.PROTOCOL["protocol"], "arm": arm,
                    "status": "checkpointed", "complete": False, "episode_slots": 2,
                    "arm_episodes": {arm: 2}, "reference_episode_slots": 2240},
                 "workers": workers}
        self.assertEqual(control._eval_checkpointed(handle, value, arm)["episode_slots"], 2)
        value["workers"]["1"]["summary"] = dict(workers["1"]["summary"], episodes=2)
        with self.assertRaises(RuntimeError):
            control._eval_checkpointed(handle, value, arm)

    def test_qsub_intent_is_durable_and_unknown_result_cannot_resubmit(self):
        fs = MemorySftp()

        @contextmanager
        def sftp_client(_transport):
            yield fs

        records = {"schema_version": 1, "units": {}}
        intent = control._intent("clip_only", 1)
        qsub_calls = []

        def command(_transport, cmd, timeout):
            qsub_calls.append(cmd)
            durable = json.loads(fs.files[control._intent_path("clip_only")])
            self.assertEqual(durable["status"], "intent_before_qsub")
            return 1, "connection closed after request"

        with mock.patch.object(control, "_require_uploaded"), \
             mock.patch.object(control, "_query_name", return_value=[]), \
             mock.patch.object(control.remote, "sftp_client", side_effect=sftp_client), \
             mock.patch.object(control.remote, "command", side_effect=command):
            with self.assertRaisesRegex(RuntimeError, "outcome is unknown"):
                control._submit(object(), "clip_only", intent, records)
            with self.assertRaisesRegex(RuntimeError, "no duplicate is allowed"):
                control._resolve(object(), "clip_only", records, intent)
        self.assertEqual(len(qsub_calls), 1)


if __name__ == "__main__":
    unittest.main()
