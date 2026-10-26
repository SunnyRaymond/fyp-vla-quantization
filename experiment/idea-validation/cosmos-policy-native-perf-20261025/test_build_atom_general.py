from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_atom_general as builder


DEQUANT = """__device__ __forceinline__ void dequant(
  int32_t *c_frag,
  int32_t *reg_a,
  int32_t *reg_b,
  float *accu
){
  for(int i = 0;i < WARP_COL_TILES; ++i){
    half2 row_scale = *(half2*)(&reg_a[i]);
    for(int j = 0;j < WARP_ROW_TILES; ++j){
      half2 col_scale = *(half2*)(&reg_b[j]);
      half2 rs_scale = __hmul2(row_scale, col_scale);
      float rs_scale_u = __half2float(rs_scale.x);
      float rs_scale_d = __half2float(rs_scale.y);
      accu[i * WARP_ROW_TILES * 4 + j * 4 + 0] +=
        (c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 0]) *
        rs_scale_u;
      accu[i * WARP_ROW_TILES * 4 + j * 4 + 1] +=
        (c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 1]) *
        rs_scale_u;
      accu[i * WARP_ROW_TILES * 4 + j * 4 + 2] +=
        (c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 2]) *
        rs_scale_d;
      accu[i * WARP_ROW_TILES * 4 + j * 4 + 3] +=
        (c_frag[i * WARP_ROW_TILES * 4 + j * 4 + 3]) *
        rs_scale_d;
    }
  }
}
"""

STREAM_LAUNCH = """#include <cuda.h>
void launch(
  const size_t K_GLOBAL
){
  compute_gemm_imma<<<gridDim, blockDim, SHMEM_SZ>>>(
}
"""
KERNEL_ANCHOR = "\n__global__ void __launch_bounds__(256) compute_gemm_imma(\n"


def atom_source(dequant: str = DEQUANT) -> str:
    return dequant + KERNEL_ANCHOR


class AtomGeneralBuilderTests(unittest.TestCase):
    def test_two_by_two_counterexample_and_correct_column_formula(self):
        activation_scales = (2.0, 3.0)
        column_scales = (5.0, 7.0)
        correct = (activation_scales[0] * column_scales[0],
                   activation_scales[0] * column_scales[1],
                   activation_scales[1] * column_scales[0],
                   activation_scales[1] * column_scales[1])
        legacy = (activation_scales[0] * column_scales[0],
                  activation_scales[0] * column_scales[0],
                  activation_scales[1] * column_scales[1],
                  activation_scales[1] * column_scales[1])
        self.assertEqual(correct, (10.0, 14.0, 15.0, 21.0))
        self.assertEqual(legacy, (10.0, 10.0, 21.0, 21.0))
        patched = builder._patch_per_column_dequant(atom_source())
        for offset, scale in enumerate(("c0", "c1", "c2", "c3")):
            self.assertIn(f"+ {offset}]) *\n        rs_scale_{scale};", patched)
        self.assertIn("__halves2half2(row_scale.x, row_scale.x)", patched)
        self.assertIn("__halves2half2(row_scale.y, row_scale.y)", patched)

    def test_only_dequant_changes_and_stream_patch_is_explicit(self):
        core = "/* official Atom MMA/load/store core remains intact */\n"
        source = "prefix\n" + atom_source() + core + STREAM_LAUNCH + "suffix\n"
        patched = builder._patched_current_stream_source(source)
        start = patched.index("__device__ __forceinline__ void dequant(")
        end = patched.index(KERNEL_ANCHOR, start)
        self.assertIn("__hmul2(upper_row_scale, col_scale)", patched[start:end])
        self.assertEqual(patched.count("#include <cuda_runtime.h>"), 1)
        self.assertIn("  cudaStream_t stream\n){", patched)
        self.assertIn("<<<gridDim, blockDim, SHMEM_SZ, stream>>>", patched)
        self.assertIn(core, patched)
        self.assertIn("prefix\n", patched)
        self.assertIn("suffix\n", patched)
        # No local edits to the kernel core or launcher beyond explicit stream wiring.
        self.assertNotIn("int32_t *c_frag", patched[:start])

    def test_missing_or_ambiguous_scale_and_stream_anchors_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "scale setup is missing or ambiguous"):
            builder._patch_per_column_dequant(atom_source(DEQUANT.replace(
                "half2 rs_scale = __hmul2(row_scale, col_scale);", "half2 changed_scale;")))
        with self.assertRaisesRegex(RuntimeError, "c1 is missing or ambiguous"):
            builder._patch_per_column_dequant(atom_source(DEQUANT.replace(
                "+ 1]) *\n        rs_scale_u;", "+ 1]) *\n        other_scale;")))
        with self.assertRaisesRegex(RuntimeError, "current-stream|anchor"):
            builder._patched_current_stream_source("#include <cuda.h>\n")


if __name__ == "__main__":
    unittest.main()
