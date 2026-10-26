import sys
import importlib.machinery
import json
import tempfile
import unittest
from itertools import product
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_diffusion
import build_diffusion


def _corrected_native_pass(a_signed, a_offset, a_scale, w_signed, w_offset_gn,
                           w_scale_gn, multiplier):
    """Pure-Python model of the per-pass signed GEMM and affine epilogue."""
    m_count, k = len(a_signed), len(a_signed[0])
    n_count = len(w_signed)
    groups = k // 128
    a_sum_mg = [[sum(a_signed[m][g * 128:(g + 1) * 128])
                  for g in range(groups)] for m in range(m_count)]
    w_sum_ng = [[sum(w_signed[n][g * 128:(g + 1) * 128])
                  for g in range(groups)] for n in range(n_count)]
    sum_weight_scale_n = [sum(w_sum_ng[n][g] * w_scale_gn[g][n]
                              for g in range(groups)) for n in range(n_count)]
    scaled_offset_ng = [[w_offset_gn[g][n] * w_scale_gn[g][n]
                         for g in range(groups)] for n in range(n_count)]
    sum_scaled_offset_n = [sum(scaled_offset_ng[n]) for n in range(n_count)]
    output = [[0.0 for _ in range(n_count)] for _ in range(m_count)]
    for m in range(m_count):
        for n in range(n_count):
            partial = 0.0
            for g in range(groups):
                start, end = g * 128, (g + 1) * 128
                dot = sum(a_signed[m][k_index] * w_signed[n][k_index]
                          for k_index in range(start, end))
                partial += dot * a_scale[m] * w_scale_gn[g][n]
            correction = (a_offset[m] * sum_weight_scale_n[n]
                          + sum(a_sum_mg[m][g] * scaled_offset_ng[n][g]
                                for g in range(groups))
                          + 128 * a_offset[m] * sum_scaled_offset_n[n])
            output[m][n] = (partial + correction * a_scale[m]) * multiplier
    return output


def _direct_affine_output(a_signed, a_offset, a_scale, w_signed,
                          w_offset_gn, w_scale_gn):
    m_count, k = len(a_signed), len(a_signed[0])
    n_count = len(w_signed)
    groups = k // 128
    return [[sum((a_signed[m][k_index] + a_offset[m])
                 * (w_signed[n][k_index] + w_offset_gn[k_index // 128][n])
                 * a_scale[m] * w_scale_gn[k_index // 128][n]
                 for k_index in range(k)) for n in range(n_count)]
            for m in range(m_count)]


class _FlatTensor:
    """Tiny stdlib-only tensor model for testing the adapter's view layout."""
    def __init__(self, values, shape, dtype="float16"):
        self.values = list(values)
        self.shape = tuple(shape)
        self.dtype = dtype
        size = 1
        for dim in self.shape:
            size *= dim
        if len(self.values) != size:
            raise ValueError("flat tensor data does not match shape")

    def reshape(self, *shape):
        return _FlatTensor(self.values, shape, self.dtype)

    def transpose(self, first, second):
        ndim = len(self.shape)
        first, second = first % ndim, second % ndim
        out_shape = list(self.shape)
        out_shape[first], out_shape[second] = out_shape[second], out_shape[first]
        strides = []
        for axis in range(ndim):
            stride = 1
            for dim in self.shape[axis + 1:]:
                stride *= dim
            strides.append(stride)
        output = []
        for out_index in product(*(range(dim) for dim in out_shape)):
            source_index = list(out_index)
            source_index[first], source_index[second] = source_index[second], source_index[first]
            offset = sum(index * stride for index, stride in zip(source_index, strides))
            output.append(self.values[offset])
        return _FlatTensor(output, out_shape, self.dtype)

    def unsqueeze(self, axis):
        shape = list(self.shape)
        shape.insert(axis % (len(shape) + 1), 1)
        return _FlatTensor(self.values, shape, self.dtype)

    def expand(self, *shape):
        out_shape = tuple(old if new == -1 else new for old, new in zip(self.shape, shape))
        if len(out_shape) != len(self.shape):
            raise ValueError("expand rank mismatch")
        strides = []
        for axis in range(len(self.shape)):
            stride = 1
            for dim in self.shape[axis + 1:]:
                stride *= dim
            strides.append(stride)
        output = []
        for out_index in product(*(range(dim) for dim in out_shape)):
            source_index = [0 if old == 1 else new
                            for old, new in zip(self.shape, out_index)]
            offset = sum(index * stride for index, stride in zip(source_index, strides))
            output.append(self.values[offset])
        return _FlatTensor(output, out_shape, self.dtype)

    def contiguous(self):
        return self


class _FakeTorch:
    float16 = "float16"

    @staticmethod
    def is_tensor(value):
        return isinstance(value, _FlatTensor)


class NativeDiffusionPackingTests(unittest.TestCase):
    def test_nunchaku_true_m_output_is_safe_for_padded_activation(self):
        # Official launch derives M from packed A and masks stores using out's actual M.
        self.assertEqual(native_diffusion._nunchaku_output_shape(
            batch=1764, padded_m=1792, padded_n=1920, out_features=1912),
            (1764, 1920))

    def test_nunchaku_shape_contract_rejects_inconsistent_padding(self):
        for values in ((1764, 1791, 1920, 1912),
                       (1537, 2048, 1920, 1912),
                       (1764, 1792, 1919, 1912),
                       (1764, 1792, 1920, 1792)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                native_diffusion._nunchaku_output_shape(*values)

    def test_nunchaku_bridge_checks_padded_inputs_scales_and_true_output(self):
        bridge = Path(__file__).with_name("nunchaku_minimal.cu").read_text(encoding="utf-8")
        for contract in ("padded_m % 256 == 0", "padded_n % 128 == 0",
                         "out.size(0) <= padded_m", "padded_m - out.size(0) < 256",
                         "out.size(1) <= padded_n", "ascales.size(0) == k / 64",
                         "ascales.size(1) == padded_m", "wscales.size(0) == k / 64",
                         "wscales.size(1) == padded_n"):
            with self.subTest(contract=contract):
                self.assertIn(contract, bridge)

    def test_nunchaku_extension_has_one_pybind_initializer(self):
        here = Path(__file__).parent
        bridge = (here / "nunchaku_minimal.cu").read_text(encoding="utf-8")
        repack = (here / "nunchaku_repack.cu").read_text(encoding="utf-8")
        self.assertEqual(bridge.count("PYBIND11_MODULE("), 1)
        self.assertEqual(repack.count("PYBIND11_MODULE("), 0)
        self.assertIn("void bind_repack(pybind11::module_& module)", repack)
        self.assertIn("void bind_repack(pybind11::module_& module);", bridge)
        self.assertIn("bind_repack(module);", bridge)
        for binding in ("pack_activation", "pack_weight",
                        "pack_activation_scales", "pack_weight_scales"):
            with self.subTest(binding=binding):
                self.assertIn(f'module.def("{binding}"', repack)

    def test_native_payload_counts_unique_cuda_storage_only(self):
        class Storage:
            def __init__(self, ptr, size):
                self.ptr = ptr
                self.size = size

            def data_ptr(self):
                return self.ptr

            def nbytes(self):
                return self.size

        class Tensor:
            def __init__(self, ptr, size, device="cuda:0"):
                self.device = device
                self.is_cuda = device.startswith("cuda")
                self.storage = Storage(ptr, size)

            def untyped_storage(self):
                return self.storage

        class FakeTorch:
            @staticmethod
            def is_tensor(value):
                return isinstance(value, Tensor)

        a = Tensor(100, 8)
        view = Tensor(100, 8)
        b = Tensor(200, 16)
        cpu = Tensor(300, 32, device="cpu")
        values = {"packed": a, "aliases": [view, b], "cpu_shadow": cpu}
        self.assertEqual(native_diffusion._unique_cuda_storage_bytes(values, FakeTorch), 24)

    def test_released_weight_is_recreated_from_cpu_shadow(self):
        class ShadowTensor:
            def to(self, *, device, dtype):
                self.destination = (device, dtype)
                return self

        class Parameter:
            def __init__(self, data, requires_grad):
                self.data = data
                self.requires_grad = requires_grad

        class Module:
            weight = None

            def register_parameter(self, name, parameter):
                setattr(self, name, parameter)

        shadow = ShadowTensor()
        module = Module()
        controller = SimpleNamespace(_base_weights_cpu={"layer": shadow},
                                     _weight_requires_grad={"layer": False})
        fake_torch = SimpleNamespace(nn=SimpleNamespace(Parameter=Parameter), bfloat16="bf16")
        native_diffusion._restore_bf16_parameters(
            {"layer": module}, controller, {"layer": "cuda:0"}, fake_torch)
        self.assertIsInstance(module.weight, Parameter)
        self.assertIs(module.weight.data, shadow)
        self.assertEqual(shadow.destination, ("cuda:0", "bf16"))
        self.assertFalse(module.weight.requires_grad)

    def test_viditq_official_launch_is_patched_to_current_stream(self):
        source = "#include <cuda.h>\nvoid f(\n  const size_t K_GLOBAL\n){\n"
        source += "  compute_gemm_imma<<<gridDim, blockDim, SHMEM_SZ>>>(x);\n}"
        patched = build_diffusion._viditq_current_stream_source(source)
        self.assertIn("#include <cuda_runtime.h>", patched)
        self.assertIn("const size_t K_GLOBAL,\n  cudaStream_t stream", patched)
        self.assertIn("compute_gemm_imma<<<gridDim, blockDim, SHMEM_SZ, stream>>>", patched)
        self.assertNotIn("cudaDeviceSynchronize", patched)

    def test_twos_complement_nibble_decode(self):
        self.assertEqual([native_diffusion._signed_nibble(x) for x in range(16)],
                         [0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1])

    def test_offset8_checkpoint_codes_map_to_signed4(self):
        for code in range(16):
            self.assertEqual(native_diffusion._signed_nibble(code - 8), code - 8)

    def test_w8_exactly_splits_into_two_signed4_values(self):
        for code in range(256):
            high, low = native_diffusion._split_w8_code(code)
            self.assertTrue(-8 <= high <= 7)
            self.assertTrue(-8 <= low <= 7)
            self.assertEqual(code - 128, 16 * high + low + 8)

    def test_affine_correction_restores_groupwise_zero_points(self):
        activations = [-8, -3, 0, 4, 7]
        weights = [-5, -1, 0, 2, 7]
        a_offset, w_offset = 6, -3
        native = sum(a * w for a, w in zip(activations, weights))
        corrected = native + native_diffusion._affine_group_correction(
            activations, weights, a_offset, w_offset)
        exact = sum((a + a_offset) * (w + w_offset)
                    for a, w in zip(activations, weights))
        self.assertEqual(corrected, exact)

    def test_affine_checkpoint_code_centering(self):
        for code in range(16):
            for zero in range(16):
                centered = native_diffusion._signed_nibble(code - 8)
                offset = 8 - zero
                self.assertEqual(centered + offset, code - zero)

    def test_viditq_w4_affine_correction_uses_n_by_g_terms_with_padded_k_and_n(self):
        m_count, n_count, padded_n, groups, padded_k, valid_k = 3, 5, 7, 3, 384, 250
        a_offset = [2, -1, 4]
        a_scale = [0.5, 0.25, 0.75]
        a_signed = [[((m * 3 + k) % 9) - 4 if k < valid_k else -a_offset[m]
                     for k in range(padded_k)] for m in range(m_count)]
        w_offset_gn = [[(g + n) % 5 - 2 if n < n_count else 0
                        for n in range(padded_n)] for g in range(groups)]
        w_scale_gn = [[0.5 + 0.125 * g + 0.0625 * n if n < n_count else 1.0
                       for n in range(padded_n)] for g in range(groups)]
        w_signed = [[(((n + 2 * k) % 15) - 7 if n < n_count else 0)
                     if k < valid_k else -w_offset_gn[k // 128][n]
                     for k in range(padded_k)] for n in range(padded_n)]

        native = _corrected_native_pass(a_signed, a_offset, a_scale, w_signed,
                                        w_offset_gn, w_scale_gn, 1)
        direct = _direct_affine_output(a_signed, a_offset, a_scale, w_signed,
                                       w_offset_gn, w_scale_gn)
        self.assertNotEqual(n_count, groups)
        for m in range(m_count):
            for n in range(padded_n):
                self.assertAlmostEqual(native[m][n], direct[m][n], places=8)
        self.assertTrue(all(native[m][n_count:] == [0.0, 0.0] for m in range(m_count)))

    def test_viditq_w8_two_pass_correction_matches_decomposition_with_padded_k_and_n(self):
        m_count, n_count, padded_n, groups, padded_k, valid_k = 3, 5, 7, 3, 384, 250
        a_offset = [1, 3, -2]
        a_scale = [0.25, 0.5, 0.75]
        a_signed = [[((m + 2 * k) % 11) - 5 if k < valid_k else -a_offset[m]
                     for k in range(padded_k)] for m in range(m_count)]
        zeros_ng = [[(91 + 19 * n + 7 * g) % 256 if n < n_count else 128
                     for g in range(groups)] for n in range(padded_n)]
        scales_gn = [[0.375 + 0.125 * g + 0.03125 * n if n < n_count else 1.0
                      for n in range(padded_n)] for g in range(groups)]
        raw_ng = [[(((n * 37 + k * 13) % 256) if n < n_count and k < valid_k
                    else zeros_ng[n][k // 128]) for k in range(padded_k)]
                  for n in range(padded_n)]
        high_ng = [[native_diffusion._split_w8_code(code)[0] for code in row]
                   for row in raw_ng]
        low_ng = [[native_diffusion._split_w8_code(code)[1] for code in row]
                  for row in raw_ng]
        high_offset_gn = [[0 for _ in range(padded_n)] for _ in range(groups)]
        low_offset_gn = [[136 - zeros_ng[n][g] for n in range(padded_n)]
                         for g in range(groups)]

        high = _corrected_native_pass(a_signed, a_offset, a_scale, high_ng,
                                      high_offset_gn, scales_gn, 16)
        low = _corrected_native_pass(a_signed, a_offset, a_scale, low_ng,
                                     low_offset_gn, scales_gn, 1)
        direct = [[sum((a_signed[m][k] + a_offset[m])
                       * (raw_ng[n][k] - zeros_ng[n][k // 128])
                       * a_scale[m] * scales_gn[k // 128][n]
                       for k in range(padded_k))
                    for n in range(padded_n)] for m in range(m_count)]
        self.assertNotEqual(n_count, groups)
        for m in range(m_count):
            for n in range(padded_n):
                self.assertAlmostEqual(high[m][n] + low[m][n], direct[m][n], places=8)
        self.assertTrue(all(direct[m][n_count:] == [0.0, 0.0] for m in range(m_count)))

    def test_viditq_forward_uses_precomputed_weight_terms_not_dense_w_unpack(self):
        source = (Path(__file__).with_name("native_diffusion.py").read_text(encoding="utf-8"))
        start = source.index("                def forward(value, _module=module, _out_features=out_features, _passes=passes")
        stop = source.index("\n\n            module.forward = forward", start)
        forward = source[start:stop]
        self.assertIn("_corrections=corrections", forward)
        self.assertNotIn("_unpack_int4(", forward)
        self.assertNotIn("w_signed.reshape", forward)
        self.assertIn("correction[\"scaled_offset_ng\"].transpose(0, 1)", forward)
        self.assertIn('"corrections": corrections', source)
        self.assertIn("retain_probe_offsets=False", source)

    def test_viditq_atom_activation_scales_match_official_lane_copy_order(self):
        # atom.cu's M_tail_check maps lane to (lane / 8) * 16 + lane % 8;
        # its 16-byte lane load contains that row and the row eight later.
        cases = ((128, 3, 128), (256, 2, 256), (1792, 4, 1764))
        for padded_m, groups, true_m in cases:
            with self.subTest(padded_m=padded_m, groups=groups):
                rows = [[(group + 1) * 10000 + row + 0.25
                         if row < true_m else 1.0
                         for row in range(padded_m)] for group in range(groups)]
                tensor = _FlatTensor([value for group in rows for value in group],
                                     (groups, padded_m))
                packed = native_diffusion._viditq_activation_scales(tensor, _FakeTorch)

                # Independent reader oracle: emulate each official CUDA lane's
                # two source rows and the four half copies consumed per row.
                expected = []
                official_lanes = []
                for group in range(groups):
                    lane_loaded = []
                    for tile in range(0, padded_m, 128):
                        for lane in range(64):
                            row = tile + (lane // 8) * 16 + lane % 8
                            official_lanes.append((group, tile // 128, lane, row, row + 8))
                            lane_loaded.extend([value for _ in range(4)
                                                for value in (rows[group][row],
                                                              rows[group][row + 8])])
                    expected.extend(lane_loaded)
                self.assertEqual(packed.shape, (groups, 4 * padded_m))
                self.assertEqual(packed.values, expected)
                # Explicit reg_a oracle: every lane's four 32-bit registers
                # carry the same pair (scale[r], scale[r+8]).
                for group, tile_index, lane, row, row8 in official_lanes:
                    base = group * 4 * padded_m + (tile_index * 64 + lane) * 8
                    self.assertEqual(packed.values[base:base + 8],
                                     [value for _ in range(4)
                                      for value in (rows[group][row], rows[group][row8])])
                for group in range(groups):
                    for row in range(true_m, padded_m):
                        tile_index, lane, first, second = next(
                            (tile, lane_index, first, second)
                            for g, tile, lane_index, first, second in official_lanes
                            if g == group and row in (first, second))
                        packed_row = group * 4 * padded_m + (tile_index * 64 + lane) * 8
                        pair_index = 0 if row == first else 1
                        self.assertEqual([packed.values[packed_row + repeat * 2 + pair_index]
                                          for repeat in range(4)], [1.0] * 4)

    def test_viditq_atom_activation_scales_reject_unpadded_or_non_fp16_input(self):
        valid = _FlatTensor([1.0] * (3 * 256), (3, 256))
        with self.assertRaises(ValueError):
            native_diffusion._viditq_activation_scales(
                _FlatTensor([1.0] * (3 * 129), (3, 129)), _FakeTorch)
        with self.assertRaises(TypeError):
            native_diffusion._viditq_activation_scales(
                _FlatTensor([1.0] * (3 * 256), (3, 256), dtype="float32"), _FakeTorch)
        self.assertEqual(native_diffusion._viditq_activation_scales(valid, _FakeTorch).shape,
                         (3, 1024))

    def test_viditq_general_extension_requires_matching_receipt_and_built_module(self):
        suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extension_dir = root / "native_extensions" / "viditq_w4a4_general"
            extension_dir.mkdir(parents=True)
            module = extension_dir / (native_diffusion.VIDITQ_GENERAL_MODULE + suffix)
            module.touch()
            old_extension_dir = root / "native_extensions" / "viditq_w4a4"
            old_extension_dir.mkdir(parents=True)
            (old_extension_dir / ("viditq_atom_native" + suffix)).touch()
            with self.assertRaises(FileNotFoundError):
                native_diffusion._extension("viditq_w4a4", root)
            receipt_dir = root / "results" / "preparation"
            receipt_dir.mkdir(parents=True)
            receipt_path = receipt_dir / "native_viditq_general_build.json"
            receipt = {
                "status": "complete",
                "source_commit": native_diffusion.VIDITQ_SOURCE_COMMIT,
                "module_name": native_diffusion.VIDITQ_GENERAL_MODULE,
                "module": str(module.resolve()),
                "producer": {"pbs_job_id": "25999999.pbs101", "host": "compute001"},
            }
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            build = native_diffusion._viditq_general_build(root)
            self.assertEqual(build["module"], str(module.resolve()))
            self.assertEqual(build["pbs_job_id"], "25999999.pbs101")
            self.assertEqual(build["source_commit"], native_diffusion.VIDITQ_SOURCE_COMMIT)

            for field, bad_value in (("status", "failed"),
                                     ("source_commit", "wrong"),
                                     ("module", str(root / "wrong.so")),
                                     ("module_name", "viditq_atom_native")):
                bad_receipt = dict(receipt)
                bad_receipt[field] = bad_value
                receipt_path.write_text(json.dumps(bad_receipt), encoding="utf-8")
                with self.subTest(field=field), self.assertRaises((RuntimeError, FileNotFoundError)):
                    native_diffusion._viditq_general_build(root)

            bad_receipt = dict(receipt)
            bad_receipt.pop("producer")
            receipt_path.write_text(json.dumps(bad_receipt), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                native_diffusion._viditq_general_build(root)

    def test_viditq_rotation_uses_pinned_public_h128_and_bumps_only_its_adapter(self):
        source = Path(__file__).with_name("native_diffusion.py").read_text(encoding="utf-8")
        self.assertEqual(native_diffusion.NATIVE_ADAPTER_VERSIONS["svdquant_w4a4"],
                         "cosmos_nunchaku_w4a4_g128_rank16_v1")
        self.assertEqual(native_diffusion.NATIVE_ADAPTER_VERSIONS["viditq_w4a4"],
                         "cosmos_viditq_atom_w4a4_g128_native_h128_precomputed_affine_half2_general_column_v4")
        self.assertIn("viditq_atom_per_channel_native", source)
        self.assertIn("native_viditq_general_build.json", source)
        self.assertIn("task-local general per-column FP16 half2 scale products", source)
        self.assertIn("native_rotations._load_backend()", source)
        self.assertIn("_load_hadamard_backend(rotation_backend)", source)
        self.assertIn("native_rotations._rotate_quarot(", source)
        for field in ("native_hadamard_source_commit", "native_hadamard_build_pbs_job_id",
                      "native_hadamard_build_host", "native_hadamard_calls",
                      "native_hadamard_input_rows"):
            self.assertIn(field, source)


if __name__ == "__main__":
    unittest.main()
