from contextlib import contextmanager
import copy
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

import diagnose_rotation_parity as diagnostic


class FakeModule:
    pass


class FakeNet:
    def __init__(self, rows):
        self.rows = rows

    def named_modules(self):
        return iter(self.rows)


class RotationParityDiagnosticTests(unittest.TestCase):
    def test_main_preserves_common_and_writes_original_startup_error(self):
        summaries = []
        fake_common = ModuleType("common")

        def fail_at_torch_startup():
            raise RuntimeError("gpu startup reached")

        fake_common.torch_module = fail_at_torch_startup
        fake_common.prepare_runtime = lambda *_args: self.fail("runtime should not be prepared")

        fake_startup_trace = ModuleType("startup_trace")

        @contextmanager
        def trace(_out):
            yield

        fake_startup_trace.startup_trace = trace
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            old_root, root, out = base / "old", base / "new", base / "new" / "logs"
            with mock.patch.object(diagnostic.perf_native, "validate_paths",
                                  return_value=(old_root, root, out)), \
                 mock.patch.object(diagnostic.perf_native, "read_json",
                                  return_value={"protocol": "frozen"}), \
                 mock.patch.object(diagnostic.perf_native, "validate_protocol"), \
                 mock.patch.object(diagnostic.perf_native, "_prepare_code_imports"), \
                 mock.patch.object(diagnostic.perf_native, "_prepare_rotation_backend_import"), \
                 mock.patch.object(diagnostic.perf_native, "_allocation_guard"), \
                 mock.patch.object(diagnostic, "_write_summary",
                                  side_effect=lambda _root, _out, row: summaries.append(copy.deepcopy(row))), \
                 mock.patch.dict(sys.modules, {"common": fake_common,
                                               "startup_trace": fake_startup_trace}):
                status = diagnostic.main(["--old-root", str(old_root),
                                          "--root", str(root), "--out", str(out)])

        self.assertEqual(status, 2)
        self.assertEqual(summaries[-1]["error"], "RuntimeError: gpu startup reached")
        self.assertIn("gpu startup reached", summaries[-1]["traceback"])
        self.assertNotIn("NoneType", summaries[-1]["error"])

    def test_uniform_row_selection_covers_endpoints_and_bounds_count(self):
        self.assertEqual(diagnostic.uniform_row_indices(1), [0])
        self.assertEqual(diagnostic.uniform_row_indices(4), [0, 1, 2, 3])
        rows = diagnostic.uniform_row_indices(101)
        self.assertEqual(len(rows), 32)
        self.assertEqual(rows[0], 0)
        self.assertEqual(rows[-1], 100)
        self.assertEqual(rows, sorted(set(rows)))

    def test_block_discovery_selects_outer_wrapper_once_per_index(self):
        root = FakeModule()
        rows = [("", root)]
        wrappers = []
        for index in range(28):
            wrapper = FakeModule()
            wrappers.append(wrapper)
            rows.append((f"net.blocks.{index}", wrapper))
            rows.append((f"net.blocks.{index}.attention", FakeModule()))
        found = diagnostic.discover_blocks(FakeNet(rows))
        self.assertEqual(len(found), 28)
        self.assertEqual([row[0] for row in found], list(range(28)))
        self.assertTrue(all(found[i][2] is wrappers[i] for i in range(28)))

    def test_block_discovery_rejects_missing_or_aliased_layers(self):
        rows = [(f"blocks.{index}", FakeModule()) for index in range(27)]
        with self.assertRaisesRegex(RuntimeError, "block 27"):
            diagnostic.discover_blocks(FakeNet(rows))
        shared = FakeModule()
        rows = [(f"blocks.{index}", shared if index < 2 else FakeModule())
                for index in range(28)]
        with self.assertRaisesRegex(RuntimeError, "multiple DiT blocks"):
            diagnostic.discover_blocks(FakeNet(rows))

    def test_each_replay_clears_query_state_then_restores_rng(self):
        events = []

        class Common:
            @staticmethod
            def begin_query(runtime):
                events.append(("begin", runtime))

            @staticmethod
            def end_query(runtime):
                events.append(("end", runtime))

        result = diagnostic._reset_replay_state(
            Common, "runtime", "same-rng", lambda state: events.append(("rng", state)),
            lambda: events.append(("forward", None)) or "output")
        self.assertEqual(result, "output")
        self.assertEqual(events, [("begin", "runtime"), ("rng", "same-rng"),
                                  ("forward", None), ("end", "runtime")])

    def test_arm_phase_order_and_shared_rng(self):
        events = []
        def install_reference(): events.append("reference")
        def snapshot_rng():
            events.append("snapshot")
            return "rng0"
        def replay(label, state):
            events.append((label, state))
            return label
        def install_native(): events.append("native")
        def restore_reference(): events.append("restore_decoded")
        def cleanup(): events.append("cleanup")

        outputs = diagnostic.run_arm_phases(
            install_reference=install_reference, snapshot_rng=snapshot_rng,
            replay=replay, install_native=install_native,
            restore_decoded_reference=restore_reference,
            cleanup_to_bf16=cleanup)
        self.assertEqual(events, ["reference", "snapshot", ("A", "rng0"),
            ("A_repeat", "rng0"), "native", ("B", "rng0"),
            "restore_decoded", ("A_after", "rng0"), "cleanup"])
        self.assertEqual(outputs, {"A": "A", "A_repeat": "A_repeat",
                                   "B": "B", "A_after": "A_after"})

    def test_cleanup_runs_when_replay_fails(self):
        events = []

        def replay(label, _state):
            events.append(label)
            if label == "B":
                raise RuntimeError("expected test failure")
            return label

        with self.assertRaisesRegex(RuntimeError, "expected test failure"):
            diagnostic.run_arm_phases(
                install_reference=lambda: events.append("reference"),
                snapshot_rng=lambda: "rng",
                replay=replay,
                install_native=lambda: events.append("native"),
                restore_decoded_reference=lambda: events.append("restore_decoded"),
                cleanup_to_bf16=lambda: events.append("cleanup"))
        self.assertEqual(events, ["reference", "A", "A_repeat", "native", "B", "cleanup"])


if __name__ == "__main__":
    unittest.main()
