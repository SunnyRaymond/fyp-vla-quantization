import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import startup_imports
import startup_trace


class StartupTraceTests(unittest.TestCase):
    def test_watchdog_is_cancelled_on_startup_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(startup_trace.faulthandler, "dump_traceback_later") as start, \
                 patch.object(startup_trace.faulthandler, "cancel_dump_traceback_later") as stop:
                with self.assertRaisesRegex(RuntimeError, "startup failed"):
                    with startup_trace.startup_trace(temporary):
                        raise RuntimeError("startup failed")
                start.assert_called_once()
                stop.assert_called_once_with()
            self.assertIn("END elapsed_seconds=", (Path(temporary) / "startup_traceback.log").read_text())

    def test_cpu_import_failure_preserves_the_last_module_and_no_forward(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(startup_imports, "guard") as guard, \
                 patch.dict(startup_imports.os.environ, {"CUDA_VISIBLE_DEVICES": ""}), \
                 patch.object(startup_imports.importlib, "import_module", side_effect=ImportError("diagnostic")), \
                 patch.object(startup_trace.faulthandler, "dump_traceback_later"), \
                 patch.object(startup_trace.faulthandler, "cancel_dump_traceback_later"):
                with self.assertRaises(ImportError):
                    startup_imports.run(temporary)
                guard.assert_called_once_with()
            result = json.loads((Path(temporary) / "startup_imports.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["currently_importing"], "torch")
            self.assertEqual(result["model_forwards"], 0)


if __name__ == "__main__":
    unittest.main()
