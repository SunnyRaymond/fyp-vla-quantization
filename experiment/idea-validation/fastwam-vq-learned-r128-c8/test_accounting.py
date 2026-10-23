"""Pure-stdlib control tests for accounting allocation and optional walltime receipts."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from accounting import qstat_walltime, require_allocation


class AllocationReceiptTests(unittest.TestCase):
    def run_with_receipt(self, receipt_text, host="cnode01.cluster"):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        nodefile = root / "PBS_NODEFILE"
        nodefile.write_text("cnode01.cluster\n", encoding="utf-8")
        receipt = root / "allocation.txt"
        receipt.write_text(receipt_text, encoding="utf-8")
        env = {
            "PBS_JOBID": "789.server",
            "PBS_NODEFILE": str(nodefile),
            "PBS_ALLOCATION_RECEIPT": str(receipt),
        }
        return patch.dict(os.environ, env, clear=True), patch(
            "accounting.socket.gethostname", return_value=host
        )

    def test_accepts_matching_running_job(self):
        env_patch, host_patch = self.run_with_receipt(
            "Job Id: 789.server\n    job_state = R\n"
        )
        with env_patch, host_patch:
            self.assertEqual(require_allocation(), ("789.server", "cnode01"))

    def test_rejects_wrong_job_id(self):
        env_patch, host_patch = self.run_with_receipt(
            "Job Id: 999.server\n    job_state = R\n"
        )
        with env_patch, host_patch, self.assertRaisesRegex(RuntimeError, "receipt"):
            require_allocation()

    def test_rejects_non_running_receipt(self):
        env_patch, host_patch = self.run_with_receipt(
            "Job Id: 789.server\n    job_state = Q\n"
        )
        with env_patch, host_patch, self.assertRaisesRegex(RuntimeError, "receipt"):
            require_allocation()

    def test_rejects_login_host(self):
        env_patch, _ = self.run_with_receipt(
            "Job Id: 789.server\n    job_state = R\n", host="login01.cluster"
        )
        with env_patch, patch("accounting.socket.gethostname", return_value="login01.cluster"):
            with self.assertRaisesRegex(RuntimeError, "login node"):
                require_allocation()

    def test_optional_unavailable_walltime_is_nonfatal(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "walltime.txt"
            receipt.write_text("unavailable\n", encoding="utf-8")
            result = qstat_walltime("789.server", receipt)
        self.assertIsNone(result["seconds"])
        self.assertEqual(result["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
