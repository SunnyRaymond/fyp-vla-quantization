from __future__ import annotations

import sys
import tempfile
import types
import unittest
import json
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import native_rotations as native
import perf_native as worker


class NativeRotationAdapterTests(unittest.TestCase):
    class _Storage:
        def __init__(self, pointer, size=16):
            self.pointer = pointer
            self.size = size

        def data_ptr(self):
            return self.pointer

        def nbytes(self):
            return self.size

    class _Tensor:
        def __init__(self, pointer, dtype="fp16", device="cuda:0", count=8, size=2):
            self.dtype = dtype
            self.device = device
            self._count = count
            self._size = size
            self._storage = NativeRotationAdapterTests._Storage(pointer, count * size)

        def numel(self):
            return self._count

        def element_size(self):
            return self._size

        def untyped_storage(self):
            return self._storage

    class _Module:
        in_features = 128
        out_features = 8
        bias = None

        def __init__(self, dtype):
            self.weight = NativeRotationAdapterTests._Tensor(1, dtype=dtype, size=2)
            self._buffers = {}
            self.forward = lambda value: value

        def register_buffer(self, name, tensor, persistent=False):
            self._buffers[name] = tensor
            setattr(self, name, tensor)

        def register_parameter(self, name, parameter):
            setattr(self, name, parameter)

    class _Controller:
        def __init__(self, arm):
            self.arm = arm
            self.set_arm_calls = []
            self.restore_calls = 0
            self._base_weights_cpu = {"target": object()}
            self._weight_requires_grad = {"target": False}
            self._forwards = {"target": lambda value: value}

        def _backup_weights(self):
            pass

        def _restore_current(self):
            self.restore_calls += 1

        def set_arm(self, arm, artifact_dir=None):
            self.set_arm_calls.append((arm, artifact_dir))
            self.arm = arm
            return {}

        def _bf16_metadata(self):
            return {"arm": "bf16"}

    def _run_fake_install(self, arm, controller_arm, p, old_root, artifact_dir=None):
        torch = SimpleNamespace(
            bfloat16="bf16", float16="fp16", int8="int8", uint8="uint8", float32="fp32",
            cuda=SimpleNamespace(is_available=lambda: True,
                                 get_device_capability=lambda _device: (8, 0)),
            nn=SimpleNamespace(Parameter=lambda value, requires_grad=False: value),
            no_grad=nullcontext)
        controller = self._Controller(controller_arm)
        module = self._Module(torch.bfloat16)
        if arm == native.SPIN_ARM:
            module._cosmos_spin_state = {
                "matrix": self._Tensor(2, dtype=torch.float32, count=128 * 128, size=4),
                "signs": self._Tensor(3, dtype=torch.float32, count=128, size=4),
                "rotate": lambda value, matrix, signs: value,
            }
        runtime = SimpleNamespace(controller=controller, target_linears={"target": module},
                                  current_arm=controller_arm)
        observed = {}

        def spin_banks(_runtime, directory, _torch):
            observed["bank_directory"] = directory
            return {"target": {"weight_groups": (self._Tensor(4),),
                                "scales": self._Tensor(5), "padded_width": 128}}

        def quarot_banks(_runtime, directory, _torch):
            observed["bank_directory"] = directory
            return {"target": {"weight_groups": (self._Tensor(4),),
                                "scales": self._Tensor(5), "signs": self._Tensor(6, dtype="int8"),
                                "padded_width": 128}}

        backend = SimpleNamespace(matmul=lambda *_args: None, __name__=native.BACKEND_MODULE)
        hadamard_receipt = {
            "source_commit": "a" * 40,
            "quarot_source_revision": native.QUAROT_SOURCE_REVISION,
            "submodule_path": native.HADAMARD_SUBMODULE_PATH,
            "pbs_job_id": "test-build.pbs", "host": "compute",
            "torch": "2.7.0+cu128", "torch_cuda": "12.8",
        }
        with patch.dict(sys.modules, {"torch": torch}), \
             patch.object(native, "EXPECTED_MODULES", 1), \
             patch.object(native, "_load_backend", return_value=backend), \
             patch.object(native, "_load_hadamard_backend",
                          return_value=(object(), hadamard_receipt, Path(old_root) / "fht.json")), \
             patch.object(native, "_spin_banks", side_effect=spin_banks), \
             patch.object(native, "_quarot_banks", side_effect=quarot_banks):
            metadata, _restore = native.install_native(
                runtime, p, old_root, arm, artifact_dir=artifact_dir, backend=backend)
        return observed["bank_directory"], controller, metadata

    def test_default_artifact_directories_and_explicit_override(self):
        old_root = Path("old-root")
        spin = Path("frozen-spin-bank")
        self.assertEqual(native._resolve_artifact_dir(
            {"reuse": {"spinquant_artifact": str(spin)}}, old_root, native.SPIN_ARM), spin)
        self.assertEqual(native._resolve_artifact_dir(
            {}, old_root, native.QUAROT_ARM), old_root / "results" / "quarot")
        explicit = Path("explicit-bank")
        self.assertEqual(native._resolve_artifact_dir(
            {}, old_root, native.SPIN_ARM, explicit), explicit)

    def test_same_arm_spin_and_quarot_still_load_the_resolved_bank_without_reinstall(self):
        old_root = Path("old-root")
        spin_dir = Path("frozen-spin-bank")
        spin_path, spin_controller, spin_metadata = self._run_fake_install(
            native.SPIN_ARM, native.SPIN_ARM,
            {"reuse": {"spinquant_artifact": str(spin_dir)}}, old_root)
        self.assertEqual(spin_path, spin_dir)
        self.assertEqual(spin_controller.set_arm_calls, [])
        self.assertEqual(spin_controller.restore_calls, 0)
        self.assertEqual(spin_metadata["artifact_dir"], str(spin_dir.resolve()))

        quarot_dir = old_root / "results" / "quarot"
        quarot_path, quarot_controller, quarot_metadata = self._run_fake_install(
            native.QUAROT_ARM, native.QUAROT_ARM, {}, old_root)
        self.assertEqual(quarot_path, quarot_dir)
        self.assertEqual(quarot_controller.set_arm_calls, [])
        self.assertEqual(quarot_controller.restore_calls, 0)
        self.assertEqual(quarot_metadata["artifact_dir"], str(quarot_dir.resolve()))

    def test_other_arm_spin_installs_the_same_explicit_artifact_used_for_native_bank(self):
        old_root = Path("old-root")
        explicit = Path("explicit-spin-bank")
        bank_path, controller, _metadata = self._run_fake_install(
            native.SPIN_ARM, "bf16", {}, old_root, artifact_dir=explicit)
        self.assertEqual(bank_path, explicit)
        self.assertEqual(controller.set_arm_calls, [(native.SPIN_ARM, explicit)])
        self.assertEqual(controller.restore_calls, 1)

    def test_g128_split_ranges_cover_padded_k_once(self):
        ranges = native._group_ranges(512)
        self.assertEqual(ranges, ((0, 0, 128), (1, 128, 256),
                                  (2, 256, 384), (3, 384, 512)))
        self.assertEqual(sum(end - start for _, start, end in ranges), 512)

    def test_offset8_nibble_rebase_is_exact_for_every_byte(self):
        for packed in range(256):
            rebased = native._offset8_to_twos_complement_byte(packed)
            for shift in (0, 4):
                offset_code = (packed >> shift) & 0xF
                signed_value = offset_code - 8
                self.assertEqual((rebased >> shift) & 0xF, signed_value & 0xF)

    def test_invalid_split_width_fails_closed(self):
        for width in (0, 127, 129):
            with self.subTest(width=width), self.assertRaises(ValueError):
                native._group_ranges(width)

    def test_runtime_import_does_not_require_torch_or_cuda(self):
        self.assertEqual(native.BACKEND_MODULE, "quarot_official_gemm")
        self.assertEqual(native.GROUP_SIZE, 128)
        self.assertEqual(native.HADAMARD_BACKEND_MODULE, "quarot_native_hadamard_cuda")
        self.assertEqual(native.ADAPTER_VERSION, "cosmos_rotation_int4_g128_v3_native_quarot_h128")

    def test_native_h128_calls_upstream_cuda_transform_with_fixed_scale(self):
        class Tensor:
            shape = (3, 128)

            def float(self):
                return self

            def contiguous(self):
                return self

        class Hadamard:
            def __init__(self):
                self.call = None
                self.output = object()

            def fast_hadamard_transform(self, value, scale):
                self.call = (value, scale)
                return self.output

        value = Tensor()
        backend = Hadamard()
        result = native._fwht_h128(value, backend)
        self.assertIs(result, backend.output)
        self.assertIs(backend.call[0], value)
        self.assertAlmostEqual(backend.call[1], 1.0 / (128 ** 0.5))

    def test_missing_native_h128_backend_fails_without_torch_fallback(self):
        class Tensor:
            shape = (1, 128)

            def float(self):
                return self

            def contiguous(self):
                return self

        with self.assertRaisesRegex(RuntimeError, "Torch fallback is disabled"):
            native._fwht_h128(Tensor(), None)

    def test_native_hadamard_receipt_requires_pinned_quarot_source_and_producer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            extension_dir = root / "native_extensions" / native.FUSED_BACKEND_MODULE
            extension_dir.mkdir(parents=True)
            module_path = extension_dir / (native.HADAMARD_BACKEND_MODULE + ".so")
            module_path.write_bytes(b"fake extension marker")
            module = types.ModuleType(native.HADAMARD_BACKEND_MODULE)
            module.__file__ = str(module_path)
            module.fast_hadamard_transform = lambda value, scale: value
            receipt = {
                "status": "complete",
                "module_name": native.HADAMARD_BACKEND_MODULE,
                "source_repository": native.HADAMARD_SOURCE_REPOSITORY,
                "submodule_path": native.HADAMARD_SUBMODULE_PATH,
                "quarot_source_revision": native.QUAROT_SOURCE_REVISION,
                "source_commit": "1" * 40,
                "pbs_job_id": "25999999.pbs101",
                "host": "x1001c3s7b0n0",
                "module": str(module_path),
            }
            self.assertIs(native._validate_hadamard_build(module, receipt,
                                                          extension_dir), receipt)
            receipt["quarot_source_revision"] = "2" * 40
            with self.assertRaisesRegex(RuntimeError, "not bound"):
                native._validate_hadamard_build(module, receipt, extension_dir)

    def test_quarot_hadamard_loader_requires_adjacent_producer_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            extension_dir = root / "native_extensions" / native.FUSED_BACKEND_MODULE
            extension_dir.mkdir(parents=True)
            module_path = extension_dir / (native.HADAMARD_BACKEND_MODULE + ".so")
            module_path.write_bytes(b"fake extension marker")
            module = types.ModuleType(native.HADAMARD_BACKEND_MODULE)
            module.__file__ = str(module_path)
            module.fast_hadamard_transform = lambda value, scale: value
            receipt = {
                "status": "complete",
                "module_name": native.HADAMARD_BACKEND_MODULE,
                "source_repository": native.HADAMARD_SOURCE_REPOSITORY,
                "submodule_path": native.HADAMARD_SUBMODULE_PATH,
                "quarot_source_revision": native.QUAROT_SOURCE_REVISION,
                "source_commit": "3" * 40,
                "pbs_job_id": "25999999.pbs101",
                "host": "x1001c3s7b0n0",
                "module": str(module_path),
            }
            receipt_path = root / "results" / "preparation" / "quarot_hadamard_build.json"
            receipt_path.parent.mkdir(parents=True)
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            gemm = types.ModuleType(native.FUSED_BACKEND_MODULE)
            gemm.__file__ = str(extension_dir / (native.FUSED_BACKEND_MODULE + ".so"))
            old_path = sys.path[:]
            try:
                with patch.dict(sys.modules, {native.HADAMARD_BACKEND_MODULE: module}):
                    loaded, loaded_receipt, loaded_path = native._load_hadamard_backend(gemm)
                self.assertIs(loaded, module)
                self.assertEqual(loaded_receipt["source_commit"], "3" * 40)
                self.assertEqual(loaded_path, receipt_path)

                receipt_path.unlink()
                with patch.dict(sys.modules, {native.HADAMARD_BACKEND_MODULE: module}):
                    with self.assertRaisesRegex(RuntimeError, "build receipt is missing"):
                        native._load_hadamard_backend(gemm)
            finally:
                sys.path[:] = old_path

    def test_fused_backend_has_priority_over_legacy_module(self):
        fused = types.ModuleType(native.FUSED_BACKEND_MODULE)
        fused.matmul = lambda *_args: None
        fused.scaled_add = lambda *_args: None
        legacy = types.ModuleType(native.BACKEND_MODULE)
        legacy.matmul = lambda *_args: None
        with patch.dict(sys.modules, {native.FUSED_BACKEND_MODULE: fused,
                                      native.BACKEND_MODULE: legacy}):
            self.assertIs(native._load_backend(), fused)

    def test_present_but_invalid_fused_backend_does_not_fallback(self):
        fused = types.ModuleType(native.FUSED_BACKEND_MODULE)
        fused.matmul = lambda *_args: None
        legacy = types.ModuleType(native.BACKEND_MODULE)
        legacy.matmul = lambda *_args: None
        with patch.dict(sys.modules, {native.FUSED_BACKEND_MODULE: fused,
                                      native.BACKEND_MODULE: legacy}):
            with self.assertRaisesRegex(RuntimeError, "must expose scaled_add"):
                native._load_backend()

    def test_fused_import_failure_does_not_fallback_to_legacy(self):
        legacy = types.ModuleType(native.BACKEND_MODULE)
        legacy.matmul = lambda *_args: None
        with patch.dict(sys.modules, {native.FUSED_BACKEND_MODULE: None,
                                      native.BACKEND_MODULE: legacy}):
            with self.assertRaisesRegex(RuntimeError, "present but failed to load"):
                native._load_backend()

    def test_fused_epilogue_receives_original_scale_tensor_and_group(self):
        class Backend:
            def __init__(self):
                self.received = None

            def scaled_add(self, *args):
                self.received = args

        output, accum, a_scales = object(), object(), object()
        stored_weight_scales, group = object(), 3
        backend = Backend()
        mode = native._accumulate_group(output, accum, a_scales,
                                        stored_weight_scales, group, backend)
        self.assertEqual(mode, "fused_native_scaled_add")
        self.assertEqual(backend.received,
                         (output, accum, a_scales, stored_weight_scales, group))

    def test_import_path_prefers_fused_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fused = root / "native_extensions" / native.FUSED_BACKEND_MODULE
            legacy = root / "native_extensions" / native.BACKEND_MODULE
            fused.mkdir(parents=True)
            legacy.mkdir(parents=True)
            old_path = sys.path[:]
            try:
                selected = worker._prepare_rotation_backend_import(root)
                native_path = str(fused.resolve())
                self.assertEqual(selected, fused)
                self.assertEqual(sys.path[0], native_path)
            finally:
                sys.path[:] = old_path


if __name__ == "__main__":
    unittest.main()
