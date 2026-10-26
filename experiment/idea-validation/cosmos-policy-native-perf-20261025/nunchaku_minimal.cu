#include <torch/extension.h>

#include "interop/torch.h"
#include "kernels/zgemm/gemm_w4a4_launch.cuh"

void bind_repack(pybind11::module_& module);

void matmul(torch::Tensor act,
            torch::Tensor wgt,
            torch::Tensor out,
            torch::Tensor ascales,
            torch::Tensor wscales) {
    TORCH_CHECK(act.is_cuda() && wgt.is_cuda() && out.is_cuda(), "all GEMM tensors must be CUDA resident");
    TORCH_CHECK(ascales.is_cuda() && wscales.is_cuda(), "all scale tensors must be CUDA resident");
    TORCH_CHECK(act.is_contiguous() && wgt.is_contiguous() && out.is_contiguous(), "GEMM tensors must be contiguous");
    TORCH_CHECK(ascales.is_contiguous() && wscales.is_contiguous(), "scale tensors must be contiguous");
    TORCH_CHECK(act.scalar_type() == torch::kUInt8 && wgt.scalar_type() == torch::kUInt8,
                "packed A4 and W4 operands must be uint8 bytes");
    TORCH_CHECK(ascales.scalar_type() == torch::kFloat16 && wscales.scalar_type() == torch::kFloat16,
                "this build uses the official FP16-scale INT4 kernel specialization");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "the native residual output must be float16");
    TORCH_CHECK(act.device() == wgt.device() && act.device() == out.device()
                    && act.device() == ascales.device() && act.device() == wscales.device(),
                "all GEMM tensors must share a CUDA device");
    TORCH_CHECK(act.dim() == 2 && wgt.dim() == 2 && out.dim() == 2
                    && ascales.dim() == 2 && wscales.dim() == 2,
                "packed GEMM operands and scales must be matrices");
    const int64_t padded_m = act.size(0);
    const int64_t padded_n = wgt.size(0);
    const int64_t k = act.size(1) * 2;
    TORCH_CHECK(padded_m > 0 && padded_m % 256 == 0,
                "packed activation M must be padded to a multiple of 256");
    TORCH_CHECK(padded_n > 0 && padded_n % 128 == 0,
                "packed weight N must be padded to a multiple of 128");
    TORCH_CHECK(k > 0 && k % 64 == 0 && wgt.size(1) == act.size(1),
                "packed activation and weight K dimensions must match and align to 64");
    TORCH_CHECK(out.size(0) > 0 && out.size(0) <= padded_m
                    && padded_m - out.size(0) < 256,
                "output M must be the true row count within the padded activation tile");
    TORCH_CHECK(out.size(1) > 0 && out.size(1) <= padded_n
                    && padded_n - out.size(1) < 128,
                "output N must fit the padded weight tile");
    TORCH_CHECK(ascales.size(0) == k / 64 && ascales.size(1) == padded_m,
                "packed activation scales must match padded K/64 x M");
    TORCH_CHECK(wscales.size(0) == k / 64 && wscales.size(1) == padded_n,
                "packed weight scales must match padded K/64 x N");

    TorchOpContext stream_context;
    Tensor none;
    using Config = nunchaku::kernels::GEMMConfig_W4A4_FP16;
    nunchaku::kernels::GEMM_W4A4_Launch<Config, false>::gemm_w4a4(
        from_torch(act), from_torch(wgt), from_torch(out), none,
        from_torch(ascales), from_torch(wscales), none, none,
        none, none, none, none, none, none, none, none, none, none, none,
        false, {}, false, false, 1.0f, none, none, none, none, 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("matmul", &matmul, "Nunchaku official INT4 W4A4 GEMM, FP16-scale specialization");
    bind_repack(module);
}
