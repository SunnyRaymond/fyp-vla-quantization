// Task-local repacker for the frozen Cosmos G128 checkpoint format.
// The W4A4 GEMM and its MMA layout come from Nunchaku; this file only maps
// existing signed INT4 codes and FP16 scales to that official layout.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "kernels/zgemm/gemm_w4a4.cuh"

namespace {
using Config = nunchaku::kernels::GEMMConfig_W4A4_FP16;

__device__ __forceinline__ uint32_t code_pack(const int8_t *matrix,
                                              int64_t row,
                                              int64_t column,
                                              int64_t stride) {
    uint32_t packed = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int8_t value = matrix[row * stride + column + j];
        packed |= (static_cast<uint32_t>(static_cast<uint8_t>(value)) & 0x0fu) << (4 * j);
    }
    return packed;
}

__device__ __forceinline__ void pack_instruction(const int8_t *matrix,
                                                  int64_t row0,
                                                  int64_t k0,
                                                  int64_t stride,
                                                  uint4 &packed,
                                                  uint32_t (&shared)[16][8]) {
    const int lane = threadIdx.x;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = i * 4 + lane / 8;
        const int column = (lane % 8) * 8;
        shared[row][lane % 8] = code_pack(matrix, row0 + row, k0 + column, stride);
    }
    __syncwarp();
    const int row = lane % 16;
    const int column = lane / 16 * 4;
    nunchaku::kernels::ldmatrix(&shared[row][column], packed);
    __syncwarp();
}

__global__ void pack_activation_kernel(const int8_t *codes, uint8_t *output,
                                       int rows, int columns) {
    __shared__ alignas(128) uint32_t shared[16][8];
    const int tile_m = blockIdx.x;
    const int group_k = blockIdx.y;
    const int lane = threadIdx.x;
    const int64_t row0 = static_cast<int64_t>(tile_m) * 16;
    const int64_t k0 = static_cast<int64_t>(group_k) * 64;
    uint4 packed;
    pack_instruction(codes, row0, k0, columns, packed, shared);
    const int block_m = tile_m / 16;
    const int warp_m = (tile_m % 16) / 2;
    const int tile16 = tile_m % 2;
    const int groups = columns / 64;
    const int64_t index = (((static_cast<int64_t>(block_m) * groups + group_k) * 8 + warp_m) * 2
                           + tile16) * 32 + lane;
    reinterpret_cast<uint4 *>(output)[index] = packed;
}

__global__ void pack_weight_kernel(const int8_t *codes, uint8_t *output,
                                   int rows, int columns) {
    __shared__ alignas(128) uint32_t shared[16][8];
    const int tile_n = blockIdx.x;
    const int group_k = blockIdx.y;
    const int lane = threadIdx.x;
    const int64_t row0 = static_cast<int64_t>(tile_n) * 16;
    const int64_t k0 = static_cast<int64_t>(group_k) * 64;
    uint4 packed;
    pack_instruction(codes, row0, k0, columns, packed, shared);
    // Nunchaku's official quantize_w4a4_wgt_kernel swaps the middle words
    // after the same ldmatrix operation because weights are column-major.
    uint32_t *words = reinterpret_cast<uint32_t *>(&packed);
    const uint32_t tmp = words[1];
    words[1] = words[2];
    words[2] = tmp;
    const int block_n = tile_n / 8;
    const int tile16 = tile_n % 8;
    const int groups = columns / 64;
    const int64_t index = ((static_cast<int64_t>(block_n) * groups + group_k) * 8 + tile16) * 32 + lane;
    reinterpret_cast<uint4 *>(output)[index] = packed;
}

__global__ void pack_activation_scales_kernel(const half *input, half *output,
                                               int rows, int groups) {
    const int lane = threadIdx.x;
    const int tile_m = blockIdx.x;
    const int group = blockIdx.y;
    const int block_m = tile_m / 8;
    const int warp_m = tile_m % 8;
    const int local = (lane / 8) * 16 + (lane % 8);
    const int64_t row0 = static_cast<int64_t>(block_m) * 256 + warp_m * 32;
    // The pinned kernel stores 16 packed half2 values per 32-row warp.
    const int64_t dst = ((static_cast<int64_t>(block_m) * groups + group) * 8 + warp_m) * 16 + lane;
    const int64_t src0 = static_cast<int64_t>(group) * rows + row0 + local;
    output[dst * 2] = input[src0];
    output[dst * 2 + 1] = input[src0 + 8];
}

__global__ void pack_weight_scales_kernel(const half *input, half *output,
                                           int rows, int groups) {
    const int lane = threadIdx.x;
    const int block_n = blockIdx.x;
    const int group = blockIdx.y;
    const int base = block_n * 128 + (lane / 4) * 16 + (lane % 4) * 2;
    const int64_t src = static_cast<int64_t>(group) * rows + base;
    const int64_t dst = (static_cast<int64_t>(block_n) * groups + group) * 128 + (lane / 4) * 16 + (lane % 4) * 4;
    output[dst] = input[src];
    output[dst + 1] = input[src + 1];
    output[dst + 2] = input[src + 8];
    output[dst + 3] = input[src + 9];
}

void check_codes(torch::Tensor codes, const char *label) {
    TORCH_CHECK(codes.is_cuda() && codes.is_contiguous() && codes.dim() == 2,
                label, " must be a contiguous CUDA int8 matrix");
    TORCH_CHECK(codes.scalar_type() == torch::kInt8, label, " must use signed int8 codes");
    TORCH_CHECK(codes.size(0) > 0 && codes.size(1) > 0 && codes.size(1) % 64 == 0,
                label, " dimensions must be positive and K divisible by 64");
}

torch::Tensor pack_codes(torch::Tensor codes, bool weight) {
    check_codes(codes, weight ? "weight codes" : "activation codes");
    TORCH_CHECK(weight ? codes.size(0) % 128 == 0 : codes.size(0) % 256 == 0,
                weight ? "weight N must be padded to 128" : "activation M must be padded to 256");
    c10::cuda::CUDAGuard guard(codes.device());
    auto packed = torch::empty({codes.size(0), codes.size(1) / 2}, codes.options().dtype(torch::kUInt8));
    const dim3 grid(weight ? codes.size(0) / 16 : codes.size(0) / 16,
                    codes.size(1) / 64);
    auto stream = at::cuda::getCurrentCUDAStream(codes.get_device()).stream();
    if (weight) {
        pack_weight_kernel<<<grid, 32, 0, stream>>>(codes.data_ptr<int8_t>(), packed.data_ptr<uint8_t>(),
                                                   codes.size(0), codes.size(1));
    } else {
        pack_activation_kernel<<<grid, 32, 0, stream>>>(codes.data_ptr<int8_t>(), packed.data_ptr<uint8_t>(),
                                                       codes.size(0), codes.size(1));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return packed;
}

torch::Tensor pack_activation_scales(torch::Tensor scales) {
    TORCH_CHECK(scales.is_cuda() && scales.is_contiguous() && scales.dim() == 2
                    && scales.scalar_type() == torch::kFloat16,
                "activation scales must be contiguous CUDA FP16 [K/64,M]");
    TORCH_CHECK(scales.size(0) > 0 && scales.size(1) > 0 && scales.size(1) % 256 == 0,
                "activation scale dimensions must be padded to M multiple 256");
    c10::cuda::CUDAGuard guard(scales.device());
    const int groups = scales.size(0);
    const int rows = scales.size(1);
    auto output = torch::empty({groups, rows}, scales.options());
    const dim3 grid(rows / 32, groups);
    auto stream = at::cuda::getCurrentCUDAStream(scales.get_device()).stream();
    pack_activation_scales_kernel<<<grid, 16, 0, stream>>>(
        reinterpret_cast<const half *>(scales.data_ptr<at::Half>()),
        reinterpret_cast<half *>(output.data_ptr<at::Half>()), rows, groups);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor pack_weight_scales(torch::Tensor scales) {
    TORCH_CHECK(scales.is_cuda() && scales.is_contiguous() && scales.dim() == 2
                    && scales.scalar_type() == torch::kFloat16,
                "weight scales must be contiguous CUDA FP16 [K/64,N]");
    TORCH_CHECK(scales.size(0) > 0 && scales.size(1) > 0 && scales.size(1) % 128 == 0,
                "weight scale dimensions must be padded to N multiple 128");
    c10::cuda::CUDAGuard guard(scales.device());
    const int groups = scales.size(0);
    const int rows = scales.size(1);
    auto output = torch::empty_like(scales);
    const dim3 grid(rows / 128, groups);
    auto stream = at::cuda::getCurrentCUDAStream(scales.get_device()).stream();
    pack_weight_scales_kernel<<<grid, 32, 0, stream>>>(
        reinterpret_cast<const half *>(scales.data_ptr<at::Half>()),
        reinterpret_cast<half *>(output.data_ptr<at::Half>()), rows, groups);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
}  // namespace

void bind_repack(pybind11::module_& module) {
    module.def("pack_activation", [](torch::Tensor codes) { return pack_codes(codes, false); });
    module.def("pack_weight", [](torch::Tensor codes) { return pack_codes(codes, true); });
    module.def("pack_activation_scales", &pack_activation_scales);
    module.def("pack_weight_scales", &pack_weight_scales);
}
