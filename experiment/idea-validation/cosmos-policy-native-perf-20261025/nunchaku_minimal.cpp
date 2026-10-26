#include <torch/extension.h>

#include "interop/torch.h"
#include "kernels/zgemm/gemm_w4a4_launch.cuh"

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
}
