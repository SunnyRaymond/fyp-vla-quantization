"""Standard-library checks for PBS submission recovery and evidence gates."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import json
from pathlib import Path
import re
import shutil
import subprocess
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

    def test_unknown_qsub_only_recovers_by_exact_name_and_never_retries(self):
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
        self.assertTrue(any("-N dfbase0001" in cmd for cmd in commands if cmd.startswith("qselect ")))

    def _complete_status(self):
        workers = {}
        for worker in range(4):
            indices = list(range(worker % 2, 280, 2))
            arm = "svdquant_w4a4" if worker < 2 else "viditq_w4a4"
            receipt = {
                "protocol": control.PROTOCOL["protocol"], "arm": arm, "worker_id": worker,
                "expected": 140, "episodes": 140, "indices": indices,
                "completed_indices": indices, "status": "complete", "complete": True,
            }
            workers[str(worker)] = {"progress": copy.deepcopy(receipt), "summary": receipt}
        counts = {"svdquant_w4a4": 280, "viditq_w4a4": 280}
        summary = {
            "protocol": control.PROTOCOL["protocol"], "status": "complete", "complete": True,
            "episode_slots": 560, "arm_counts": counts, "reference_episode_slots": 2240,
        }
        marker = {**summary, "complete": True}
        return {
            "protocol_matches": True,
            "handle": {
                "job_id": "12345.pbs101", "job_name": "dfbase0001", "attempt": 1,
                "gpus": 4, "ncpus": 32, "memory_gb": 220, "walltime": "24:00:00",
            },
            "scheduler": {
                "query_rc": 0, "job_state": "F", "Exit_status": "0", "Job_Name": "dfbase0001",
            },
            "artifact": {
                "exit_code": "0", "PIPELINE_COMPLETE": True, "PIPELINE_CHECKPOINTED": False,
                "RUN_PHASE": {"protocol": control.PROTOCOL["protocol"], "phase": "complete"},
            },
            "evaluation": {"summary": summary, "PIPELINE_COMPLETE": marker, "PIPELINE_CHECKPOINTED": None},
            "workers": workers,
        }

    def test_completion_requires_exact_worker_arm_and_reference_counts(self):
        value = self._complete_status()
        self.assertTrue(control.complete_gate(value))
        value["evaluation"]["summary"]["episode_slots"] = 559
        self.assertFalse(control.complete_gate(value))
        value = self._complete_status()
        value["evaluation"]["PIPELINE_COMPLETE"]["reference_episode_slots"] = 2241
        self.assertFalse(control.complete_gate(value))
        value = self._complete_status()
        value["workers"]["2"]["summary"]["arm"] = "svdquant_w4a4"
        self.assertFalse(control.complete_gate(value))

    def test_checkpoint_requires_all_prefixes_and_soft_deadline(self):
        value = self._complete_status()
        value["artifact"].update(
            PIPELINE_COMPLETE=False, PIPELINE_CHECKPOINTED=True,
            RUN_PHASE={"protocol": control.PROTOCOL["protocol"], "phase": "checkpointed"},
        )
        workers = {}
        for worker in range(4):
            indices = list(range(worker % 2, 280, 2))
            arm = "svdquant_w4a4" if worker < 2 else "viditq_w4a4"
            receipt = {
                "protocol": control.PROTOCOL["protocol"], "arm": arm, "worker_id": worker,
                "expected": 140, "episodes": 50, "indices": indices,
                "completed_indices": indices[:50], "status": "checkpointed",
                "complete": False, "soft_deadline_reached": True,
            }
            workers[str(worker)] = {"progress": copy.deepcopy(receipt), "summary": receipt}
        value["workers"] = workers
        counts = {"svdquant_w4a4": 100, "viditq_w4a4": 100}
        summary = {
            "protocol": control.PROTOCOL["protocol"], "status": "checkpointed", "complete": False,
            "episode_slots": 200, "arm_counts": counts, "reference_episode_slots": 2240,
        }
        value["evaluation"] = {
            "summary": summary, "PIPELINE_COMPLETE": None,
            "PIPELINE_CHECKPOINTED": {**summary},
        }
        self.assertTrue(control._checkpoint_gate(value))
        value["workers"]["0"]["summary"]["soft_deadline_reached"] = False
        self.assertFalse(control._checkpoint_gate(value))

    def test_resume_submits_only_from_verified_checkpoint(self):
        checkpoint = self._complete_status()
        checkpoint["artifact"].update(
            PIPELINE_COMPLETE=False, PIPELINE_CHECKPOINTED=True,
            RUN_PHASE={"protocol": control.PROTOCOL["protocol"], "phase": "checkpointed"},
        )
        checkpoint["evaluation"]["PIPELINE_COMPLETE"] = None
        counts = {"svdquant_w4a4": 100, "viditq_w4a4": 100}
        for row in checkpoint["workers"].values():
            receipt = row["summary"]
            receipt.update(episodes=50, completed_indices=receipt["indices"][:50],
                           status="checkpointed", complete=False, soft_deadline_reached=True)
            row["progress"] = copy.deepcopy(receipt)
        checkpoint["evaluation"]["summary"].update(
            status="checkpointed", complete=False, episode_slots=200, arm_counts=counts)
        checkpoint["evaluation"]["PIPELINE_CHECKPOINTED"] = {
            **checkpoint["evaluation"]["summary"]
        }
        with (patch.object(control, "status", return_value=checkpoint),
              patch.object(control, "_submit_attempt", return_value={"job_id": "12346.pbs101"}) as submit):
            result = control.resume(object())
        self.assertEqual(result["job_id"], "12346.pbs101")
        submit.assert_called_once_with(unittest.mock.ANY, 2, "12345.pbs101")

    def test_run_pbs_shell_and_embedded_python_are_syntactically_valid(self):
        script = Path(__file__).with_name("run.pbs")
        bash = shutil.which("bash")
        if bash:
            subprocess.run([bash, "-n"], input=script.read_bytes(), check=True)
        source = script.read_text(encoding="utf-8")
        blocks = re.findall(r"<<'(?P<tag>PY_[A-Z_]+)'\r?\n(?P<body>.*?)\r?\n(?P=tag)", source, re.S)
        self.assertGreaterEqual(len(blocks), 3)
        for tag, body in blocks:
            compile(body, f"{script.name}:{tag}", "exec")


if __name__ == "__main__":
    unittest.main()
