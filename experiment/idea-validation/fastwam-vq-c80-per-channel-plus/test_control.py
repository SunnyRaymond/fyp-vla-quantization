"""Standard-library tests for single-job submission and completion gates."""
from __future__ import annotations

from contextlib import contextmanager
import json
import unittest
from unittest.mock import patch

import control


class MemoryFile:
    def __init__(self, store, path, mode):
        self.store, self.path, self.mode = store, path, mode
        self.buffer = bytearray() if "w" in mode else None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.buffer is not None:
            self.store[self.path] = bytes(self.buffer)

    def read(self, size=-1):
        data = self.store[self.path]
        return data if size < 0 else data[:size]

    def write(self, data):
        self.buffer.extend(data)
        return len(data)


class MemorySFTP:
    def __init__(self):
        self.files = {}

    def file(self, path, mode):
        if "r" in mode and path not in self.files:
            raise FileNotFoundError(path)
        return MemoryFile(self.files, path, mode)

    def stat(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return object()

    def posix_rename(self, source, target):
        self.files[target] = self.files.pop(source)


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.sftp = MemorySFTP()

    @contextmanager
    def sftp_client(self, _transport):
        yield self.sftp

    def test_submit_is_idempotent(self):
        commands = []

        def command(_transport, cmd, **_kwargs):
            commands.append(cmd)
            if cmd.startswith("qselect "):
                return 0, ""
            if cmd.startswith("qsub "):
                return 0, "12345.pbs101"
            self.fail("unexpected remote command: " + cmd)

        with (patch.object(control, "_mkdirs"), patch.object(control, "_require_uploaded"),
              patch.object(control.remote, "sftp_client", self.sftp_client),
              patch.object(control.remote, "command", side_effect=command)):
            first = control.submit(object())
            second = control.submit(object())
        self.assertEqual(first["job_id"], "12345.pbs101")
        self.assertEqual(second["job_id"], first["job_id"])
        self.assertEqual(sum(cmd.startswith("qsub ") for cmd in commands), 1)

    def test_unknown_qsub_is_only_recovered_by_exact_name(self):
        commands = []

        def command(_transport, cmd, **_kwargs):
            commands.append(cmd)
            if cmd.startswith("qselect "):
                return 0, ""
            if cmd.startswith("qsub "):
                return 255, "connection ended before the reply"
            self.fail("unexpected remote command: " + cmd)

        with (patch.object(control, "_mkdirs"), patch.object(control, "_require_uploaded"),
              patch.object(control.remote, "sftp_client", self.sftp_client),
              patch.object(control.remote, "command", side_effect=command)):
            with self.assertRaisesRegex(RuntimeError, "no retry was made"):
                control.submit(object())
            with self.assertRaisesRegex(RuntimeError, "unresolved"):
                control.submit(object())
        self.assertEqual(sum(cmd.startswith("qsub ") for cmd in commands), 1)
        intent = json.loads(self.sftp.files[control._intent_path()].decode("utf-8"))
        self.assertEqual(intent["status"], "unknown_or_rejected")

    def _complete_status(self):
        workers = {}
        for worker in range(4):
            indices = list(range(worker, 280, 4))
            receipt = {
                "protocol": control.PROTOCOL["protocol"], "arm": control.ARM,
                "worker_id": worker, "expected": 70, "episodes": 70,
                "indices": indices, "completed_indices": indices,
                "status": "complete", "complete": True,
            }
            workers[str(worker)] = {"progress": receipt, "summary": receipt}
        summary = {
            "protocol": control.PROTOCOL["protocol"], "arm": control.ARM,
            "status": "complete", "complete": True, "episode_slots": 280,
            "arm_episodes": {control.ARM: 280}, "reference_episode_slots": 2800,
        }
        marker = {
            "protocol": control.PROTOCOL["protocol"], "arm": control.ARM,
            "complete": True, "episode_slots": 280, "reference_episode_slots": 2800,
        }
        return {
            "protocol_matches": True,
            "handle": {
                "job_id": "12345.pbs101", "job_name": "c80pceval0001", "arm": control.ARM,
                "gpus": 4, "ncpus": 32, "memory_gb": 220, "walltime": "24:00:00",
            },
            "scheduler": {
                "query_rc": 0, "job_state": "F", "Exit_status": "0", "Job_Name": "c80pceval0001",
            },
            "artifact": {
                "exit_code": "0", "PIPELINE_COMPLETE": True, "PIPELINE_CHECKPOINTED": False,
                "RUN_PHASE": {"protocol": control.PROTOCOL["protocol"], "arm": control.ARM, "phase": "complete"},
            },
            "evaluation": {"summary": summary, "PIPELINE_COMPLETE": marker},
            "workers": workers,
        }

    def test_completion_requires_280_episodes_and_2800_references(self):
        value = self._complete_status()
        self.assertTrue(control.complete_gate(value))
        value["evaluation"]["summary"]["episode_slots"] = 279
        self.assertFalse(control.complete_gate(value))
        value = self._complete_status()
        value["evaluation"]["PIPELINE_COMPLETE"]["reference_episode_slots"] = 2799
        self.assertFalse(control.complete_gate(value))

    def test_completion_requires_four_complete_workers(self):
        value = self._complete_status()
        value["workers"]["2"]["summary"]["episodes"] = 69
        self.assertFalse(control.complete_gate(value))

    def test_resume_requires_valid_checkpoint_and_soft_deadline(self):
        value = self._complete_status()
        value["handle"]["attempt"] = 1
        value["protocol_matches"] = True
        value["artifact"].update(
            PIPELINE_COMPLETE=False, PIPELINE_CHECKPOINTED=True,
            RUN_PHASE={"protocol": control.PROTOCOL["protocol"], "arm": control.ARM, "phase": "checkpointed"},
        )
        workers = {}
        for worker in range(4):
            indices = list(range(worker, 280, 4))
            receipt = {
                "protocol": control.PROTOCOL["protocol"], "arm": control.ARM,
                "worker_id": worker, "expected": 70, "episodes": 25,
                "indices": indices, "completed_indices": indices[:25],
                "status": "checkpointed", "complete": False, "soft_deadline_reached": True,
            }
            workers[str(worker)] = {"progress": receipt, "summary": receipt}
        value["workers"] = workers
        value["evaluation"] = {
            "summary": {
                "protocol": control.PROTOCOL["protocol"], "arm": control.ARM,
                "status": "checkpointed", "complete": False, "episode_slots": 100,
                "arm_episodes": {control.ARM: 100}, "reference_episode_slots": 2800,
            },
            "PIPELINE_COMPLETE": None,
            "PIPELINE_CHECKPOINTED": {
                "protocol": control.PROTOCOL["protocol"], "arm": control.ARM,
                "complete": False, "episode_slots": 100, "reference_episode_slots": 2800,
            },
        }
        self.assertTrue(control._checkpoint_gate(value))
        value["workers"]["0"]["summary"]["soft_deadline_reached"] = False
        value["workers"]["0"]["progress"]["soft_deadline_reached"] = False
        self.assertFalse(control._checkpoint_gate(value))


if __name__ == "__main__":
    unittest.main()
