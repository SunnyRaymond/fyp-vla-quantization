// Thin bridge to the pinned Nunchaku W4A4 kernel and its official LoRA APIs.
// The task-local logic only allocates/discards the helper's extra groupwise A4
// outputs; factor packing is handled by native_svd_fused.py.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "interop/torch.h"
#include "kernels/zgemm/gemm_w4a4_launch.cuh"

void bind_repack(pybind11::module_& module);

namespace {
using Config = nunchaku::kernels::GEMMConfig_W4A4_FP16;
using Launcher = nunchaku::kernels::GEMM_W4A4_Launch<Config, false>;

void require_same_cuda_device(const torch::Tensor& reference, const torch::Tensor& other,
                              const char* message) {
    TORCH_CHECK(other.is_cuda() && other.device() == reference.device(), message);
}

torch::Tensor quantize_down(torch::Tensor input, torch::Tensor packed_down) {
    TORCH_CHECK(input.is_cuda() && packed_down.is_cuda(), "LoRA down tensors must be CUDA resident");
    TORCH_CHECK(input.device() == packed_down.device(), "LoRA down tensors must share a CUDA device");
    TORCH_CHECK(input.is_contiguous() && packed_down.is_contiguous(), "LoRA down tensors must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat16 && packed_down.scalar_type() == torch::kFloat16,
                "Nunchaku LoRA down uses FP16 activation and packed factor");
    TORCH_CHECK(input.dim() == 2 && packed_down.dim() == 2,
                "Nunchaku LoRA down expects input [M,K] and packed factor [K,rank]");
    const int64_t actual_m = input.size(0);
    const int64_t k = input.size(1);
    const int64_t rank = packed_down.size(1);
    TORCH_CHECK(actual_m > 0 && k > 0 && k % 128 == 0,
                "Nunchaku down K must be padded to a positive multiple of 128");
    TORCH_CHECK(packed_down.size(0) == k && rank == 16,
                "packed LoRA down must match padded K and frozen rank 16");

    const int64_t padded_m = ((actual_m + 255) / 256) * 256;
    auto output = torch::empty({padded_m, k / 2}, input.options().dtype(torch::kUInt8));
    auto scales = torch::empty({k / 64, padded_m}, input.options());
    auto lora_act = torch::empty({padded_m, rank}, input.options().dtype(torch::kFloat32));

    c10::cuda::CUDAGuard device_guard(input.device());
    TorchOpContext stream_context;
    Tensor none;
    Launcher::quantize_w4a4_act_fuse_lora(
        from_torch(input), from_torch(output), from_torch(scales), from_torch(packed_down),
        from_torch(lora_act), none, false, false);
    // output/scales are the sidecar's new groupwise-A4 quantization results;
    // this adapter intentionally discards them and retains only lora_act.
    return lora_act;
}

void matmul_lora_up(torch::Tensor act,
                    torch::Tensor wgt,
                    torch::Tensor out,
                    torch::Tensor ascales,
                    torch::Tensor wscales,
                    torch::Tensor lora_act,
                    torch::Tensor packed_up) {
    TORCH_CHECK(act.is_cuda() && wgt.is_cuda() && out.is_cuda(), "GEMM tensors must be CUDA resident");
    require_same_cuda_device(act, wgt, "packed A4 and W4 tensors must share a CUDA device");
    require_same_cuda_device(act, out, "GEMM output must share the activation CUDA device");
    require_same_cuda_device(act, ascales, "activation scales must share the activation CUDA device");
    require_same_cuda_device(act, wscales, "weight scales must share the activation CUDA device");
    require_same_cuda_device(act, lora_act, "LoRA activation must share the activation CUDA device");
    require_same_cuda_device(act, packed_up, "LoRA up factor must share the activation CUDA device");
    TORCH_CHECK(act.is_contiguous() && wgt.is_contiguous() && out.is_contiguous()
                    && ascales.is_contiguous() && wscales.is_contiguous()
                    && lora_act.is_contiguous() && packed_up.is_contiguous(),
                "all Nunchaku fused GEMM tensors must be contiguous");
    TORCH_CHECK(act.scalar_type() == torch::kUInt8 && wgt.scalar_type() == torch::kUInt8,
                "packed activation and residual weights must be UINT8 bytes");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16 && ascales.scalar_type() == torch::kFloat16
                    && wscales.scalar_type() == torch::kFloat16,
                "this Nunchaku extension uses FP16 output and scale tensors");
    TORCH_CHECK(lora_act.scalar_type() == torch::kFloat32 && packed_up.scalar_type() == torch::kFloat16,
                "official LoRA epilogue expects FP32 activation and packed FP16 up factor");
    TORCH_CHECK(act.dim() == 2 && wgt.dim() == 2 && out.dim() == 2
                    && ascales.dim() == 2 && wscales.dim() == 2
                    && lora_act.dim() == 2 && packed_up.dim() == 2,
                "Nunchaku fused GEMM expects matrix operands");

    const int64_t padded_m = act.size(0);
    const int64_t padded_n = wgt.size(0);
    const int64_t k = act.size(1) * 2;
    const int64_t rank = packed_up.size(1);
    TORCH_CHECK(padded_m > 0 && padded_m % 256 == 0,
                "packed activation M must be padded to a multiple of 256");
    TORCH_CHECK(padded_n > 0 && padded_n % 128 == 0,
                "packed residual and LoRA up N must be padded to a multiple of 128");
    TORCH_CHECK(k > 0 && k % 64 == 0 && wgt.size(1) == act.size(1),
                "packed activation and residual K dimensions must match and align to 64");
    TORCH_CHECK(out.size(0) > 0 && out.size(0) <= padded_m
                    && padded_m - out.size(0) < 256,
                "output rows must be the true M within the padded activation tile");
    TORCH_CHECK(out.size(1) > 0 && out.size(1) <= padded_n
                    && padded_n - out.size(1) < 128,
                "output columns must fit the padded N tile");
    TORCH_CHECK(ascales.size(0) == k / 64 && ascales.size(1) == padded_m,
                "activation scales must be padded K/64 x M");
    TORCH_CHECK(wscales.size(0) == k / 64 && wscales.size(1) == padded_n,
                "weight scales must be padded K/64 x N");
    TORCH_CHECK(rank == 16 && packed_up.size(0) == padded_n,
                "packed LoRA up must match padded N and frozen rank 16");
    TORCH_CHECK(lora_act.size(0) == padded_m && lora_act.size(1) == rank,
                "LoRA activation must match padded M and rank 16");

    c10::cuda::CUDAGuard device_guard(act.device());
    TorchOpContext stream_context;
    Tensor none;
    Launcher::gemm_w4a4(
        from_torch(act), from_torch(wgt), from_torch(out), none,
        from_torch(ascales), from_torch(wscales), none, none,
        from_torch(lora_act), from_torch(packed_up), none, none,
        none, none, none, none, none, none, none,
        false, {1.0f}, false, false, 1.0f, none, none, none, none, 0);
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("quantize_down", &quantize_down,
               "Nunchaku official SVDQuant down projection; discard its groupwise A4 side outputs");
    module.def("matmul_lora_up", &matmul_lora_up,
               "Nunchaku official FP16 INT4 GEMM with rank-16 LoRA up epilogue");
    bind_repack(module);
}
