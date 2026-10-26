from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_hadamard as builder


class HadamardBuilderTests(unittest.TestCase):
    def test_reads_exact_git_submodule_commit_pointer(self):
        commit = "a" * 40
        tree = {"truncated": False, "tree": [
            {"path": "third-party/fast-hadamard-transform", "mode": "160000",
             "type": "commit", "sha": commit},
        ]}
        self.assertEqual(builder._submodule_pointer(tree), commit)

    def test_rejects_non_submodule_or_ambiguous_tree_entry(self):
        entry = {"path": builder.QUAROT_SUBMODULE_PATH, "mode": "100644",
                 "type": "blob", "sha": "a" * 40}
        with self.assertRaisesRegex(RuntimeError, "not a git submodule"):
            builder._submodule_pointer({"truncated": False, "tree": [entry]})
        entry["mode"] = "160000"
        entry["type"] = "commit"
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            builder._submodule_pointer({"truncated": False, "tree": [entry, dict(entry)]})
        with self.assertRaisesRegex(RuntimeError, "truncated"):
            builder._submodule_pointer({"truncated": True, "tree": []})

    def test_source_url_must_match_official_quarot_submodule(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            (source / ".gitmodules").write_text(
                '[submodule "third-party/fast-hadamard-transform"]\n'
                '  path = third-party/fast-hadamard-transform\n'
                f'  url = {builder.FHT_REPOSITORY}\n', encoding="utf-8")
            self.assertEqual(builder._submodule_url(source), builder.FHT_REPOSITORY)
            (source / ".gitmodules").write_text(
                '[submodule "third-party/fast-hadamard-transform"]\n'
                '  path = third-party/fast-hadamard-transform\n'
                '  url = https://example.invalid/modified.git\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "expected official FHT"):
                builder._submodule_url(source)

    def test_build_compiles_only_upstream_binding_and_cuda_kernel_tus(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            (source / "csrc").mkdir()
            expected = ("fast_hadamard_transform.cpp", "fast_hadamard_transform_cuda.cu")
            for name in expected:
                (source / "csrc" / name).write_text("", encoding="utf-8")
            (source / "csrc" / "fast_hadamard_transform_12n_cuda.cu").write_text(
                "not needed as a separate TU", encoding="utf-8")
            self.assertEqual(tuple(path.name for path in builder._source_files(source)), expected)

    def test_staged_source_marker_must_match_pinned_revision(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "vendor" / "fast-hadamard-transform"
            target.mkdir(parents=True)
            (target / "UPSTREAM_COMMIT").write_text("b" * 40 + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "differs from the pinned"):
                builder._stage_source(root, "a" * 40)


if __name__ == "__main__":
    unittest.main()
