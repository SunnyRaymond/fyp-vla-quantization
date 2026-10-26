import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_rotations_grouped as grouped
import native_rotations


class _FakeTensor:
    def __init__(self, values, dtype="bf16", finite=True):
        self.values = list(values)
        self.dtype = dtype
        self.finite = finite

    def to(self, *, dtype):
        values = self.values
        finite = self.finite
        if dtype == "fp16":
            values = [min(value, 65504.0) if value > 65504.0 else value
                      for value in values]
            finite = finite and all(abs(value) <= 65504.0 for value in self.values)
        return _FakeTensor(values, dtype=dtype, finite=finite)

    def repeat_interleave(self, count, dim):
        if dim != 1:
            raise AssertionError("test fake supports scale expansion on dimension 1 only")
        return _FakeMatrix([[value for value in row for _ in range(count)]
                            for row in self.values], self.dtype, self.finite)

    def transpose(self, dim0, dim1):
        if (dim0, dim1) != (0, 1):
            raise AssertionError("test fake supports matrix transpose only")
        return _FakeMatrix([list(row) for row in zip(*self.values)], self.dtype, self.finite)

    def contiguous(self):
        return self


class _FakeMatrix(_FakeTensor):
    pass


class _NonExactScale(_FakeTensor):
    def to(self, *, dtype):
        if dtype == "fp16":
            return _FakeTensor([1.0], dtype=dtype, finite=True)
        return _FakeTensor(self.values, dtype=dtype, finite=self.finite)


class _FakeMask:
    def __init__(self, value):
        self.value = value

    def all(self):
        return self

    def item(self):
        return self.value


class _FakeTorch:
    float16 = "fp16"
    bfloat16 = "bf16"

    @staticmethod
    def isfinite(value):
        return _FakeMask(value.finite)

    @staticmethod
    def equal(left, right):
        return left.dtype == right.dtype and left.values == right.values


class _TensorGroups:
    shape = (3, 4, 64)

    def __len__(self):
        return self.shape[0]

    def __bool__(self):
        raise AssertionError("Tensor-like groups must not be tested for truthiness")


class _MalformedTensorGroups:
    shape = (3, 4)

    def __len__(self):
        return 3


class _RegisteredBufferModule:
    """Minimal nn.Module storage shape: registered tensors live in _buffers."""
    def __init__(self, signs):
        self._buffers = {"_native_rotation_signs": signs}


class _AttributeBufferModule:
    def __init__(self, signs):
        self._native_rotation_signs = signs


class NativeRotationGroupedTests(unittest.TestCase):
    def test_checkpoint_g128_scale_repeats_into_adjacent_g64_groups(self):
        original = _FakeMatrix([[0.25, 0.5, 1.0]], dtype="fp16")
        expanded = grouped._repeat_group_scales(original, _FakeTorch)
        self.assertEqual(expanded.values, [[0.25, 0.25, 0.5, 0.5, 1.0, 1.0]])
        self.assertEqual(expanded.dtype, "fp16")
        self.assertEqual(grouped._group_repeat_count(), 2)

    def test_group_mapping_rejects_a_non_integral_group_ratio(self):
        with self.assertRaises(ValueError):
            grouped._group_repeat_count(128, 96)

    def test_group_tensor_container_uses_length_and_shape_not_truthiness(self):
        self.assertEqual(grouped._weight_group_count(_TensorGroups()), 3)

    def test_group_tensor_shape_is_validated(self):
        with self.assertRaisesRegex(RuntimeError, r"\[G,N,K/2\]"):
            grouped._weight_group_count(_MalformedTensorGroups())

    def test_quarot_rotation_reads_registered_sign_buffer(self):
        signs = _FakeTensor([1.0] * 128, dtype="fp32")
        module = _RegisteredBufferModule(signs)
        torch, hadamard = object(), object()
        with patch.object(native_rotations, "_rotate_quarot", return_value="rotated") as rotate:
            result = grouped._rotation_payload(module, native_rotations.QUAROT_ARM,
                                               "input", torch, hadamard)
        self.assertEqual(result, "rotated")
        rotate.assert_called_once_with("input", signs, torch, hadamard)
        self.assertIs(grouped._registered_buffer(module, "_native_rotation_signs"), signs)

    def test_registered_sign_lookup_keeps_plain_attribute_fallback(self):
        signs = object()
        self.assertIs(grouped._registered_buffer(
            _AttributeBufferModule(signs), "_native_rotation_signs"), signs)

    def test_quarot_scale_path_preserves_original_fp16_tensor(self):
        original = _FakeTensor([0.125, 2.0], dtype="fp16")
        result = grouped._weight_scales_fp16(original, native_rotations.QUAROT_ARM,
                                             _FakeTorch)
        self.assertIs(result, original)

    def test_spin_scale_path_requires_exact_finite_fp16_round_trip(self):
        exact = _FakeTensor([0.125, 2.0], dtype="bf16")
        result = grouped._weight_scales_fp16(exact, native_rotations.SPIN_ARM,
                                             _FakeTorch)
        self.assertEqual(result.dtype, "fp16")
        self.assertEqual(result.values, exact.values)
        out_of_range = _FakeTensor([100000.0], dtype="bf16")
        with self.assertRaisesRegex(RuntimeError, "not finite after FP16 conversion"):
            grouped._weight_scales_fp16(out_of_range, native_rotations.SPIN_ARM,
                                        _FakeTorch)
        with self.assertRaisesRegex(RuntimeError, "not exactly representable"):
            grouped._weight_scales_fp16(_NonExactScale([1.1], dtype="bf16"),
                                        native_rotations.SPIN_ARM, _FakeTorch)

    def test_nunchaku_build_is_bound_to_the_loaded_extension_and_pbs_producer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            extension_path = root / "native_extensions" / "svdquant_w4a4" / "nunchaku.so"
            extension_path.parent.mkdir(parents=True)
            extension_path.write_bytes(b"test-placeholder")
            receipt_path = root / "results" / "preparation" / "native_diffusion_build.json"
            receipt_path.parent.mkdir(parents=True)
            receipt_path.write_text(json.dumps({
                "status": "complete",
                "pbs_job_id": "12345.pbs",
                "arms": {"svdquant_w4a4": {
                    "module": str(extension_path),
                    "source_commit": grouped.NUNCHAKU_SOURCE_COMMIT,
                    "upstream_sources": ["upstream.cu"],
                    "task_local_sources": ["adapter.cu"],
                }},
            }), encoding="utf-8")
            extension = SimpleNamespace(
                __file__=str(extension_path),
                matmul=lambda *args: None,
                pack_activation=lambda *args: None,
                pack_weight=lambda *args: None,
                pack_activation_scales=lambda *args: None,
                pack_weight_scales=lambda *args: None,
            )
            actual, receipt, arm = grouped._read_nunchaku_build(root, extension)
            self.assertEqual(actual, receipt_path)
            self.assertEqual(receipt["pbs_job_id"], "12345.pbs")
            self.assertEqual(arm["source_commit"], grouped.NUNCHAKU_SOURCE_COMMIT)
            extension.__file__ = str(root / "wrong.so")
            with self.assertRaisesRegex(RuntimeError, "differs from the recorded build"):
                grouped._read_nunchaku_build(root, extension)

            extension.__file__ = str(extension_path)
            receipt["arms"]["svdquant_w4a4"]["source_commit"] = "a" * 40
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "pinned public-core source commit"):
                grouped._read_nunchaku_build(root, extension)

    def test_full_k_adapter_and_provenance_are_explicit_in_source(self):
        source = Path(grouped.__file__).read_text(encoding="utf-8")
        self.assertIn("public_nunchaku_w4a4_no_lora_full_k_grouped_g64_adapter", source)
        self.assertIn("not an official full Cosmos SpinQuant/QuaRot", source)
        self.assertIn("extension.matmul(packed_activation, context[\"packed_weight\"]", source)
        self.assertIn("native_diffusion._nunchaku_output_shape", source)
        forward = source.split("def _grouped_forward", 1)[1].split(
            "def _unique_cuda_storage_bytes", 1)[0]
        self.assertNotIn("native_rotations._quantize_activation", forward)
        self.assertNotIn("_unpack_signed_activation_groups", source)
        self.assertEqual(forward.count("extension.pack_activation("), 1)
        self.assertIn("torch.round(rows / safe).clamp(-7, 7)", forward)
        self.assertIn("module.register_parameter(\"weight\", None)",
                      Path(native_rotations.__file__).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
