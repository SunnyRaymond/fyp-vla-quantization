#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAStream.h>

#include <cstddef>
#include <cstdint>

void DenseLayerGEMM_i4_o16(const uint8_t *A,
                           const uint8_t *B,
                           const uint8_t *A_scale,
                           const uint8_t *B_scale,
                           const uint8_t *A_keeper,
                           const uint8_t *B_keeper,
                           const uint8_t *A_keeper_scale,
                           const uint8_t *B_keeper_scale,
                           half *D,
                           const size_t M_GLOBAL,
                           const size_t N_GLOBAL,
                           const size_t K_GLOBAL,
                           cudaStream_t stream);

void matmul(torch::Tensor act,
            torch::Tensor wgt,
            torch::Tensor out,
            torch::Tensor ascales,
            torch::Tensor wscales,
            torch::Tensor act_keeper,
            torch::Tensor wgt_keeper,
            torch::Tensor act_keeper_scale,
            torch::Tensor wgt_keeper_scale) {
    TORCH_CHECK(act.is_cuda() && wgt.is_cuda() && out.is_cuda(), "all GEMM tensors must be CUDA resident");
    TORCH_CHECK(ascales.is_cuda() && wscales.is_cuda(), "all group scales must be CUDA resident");
    TORCH_CHECK(act_keeper.is_cuda() && wgt_keeper.is_cuda()
                    && act_keeper_scale.is_cuda() && wgt_keeper_scale.is_cuda(),
                "keeper tensors must be CUDA resident");
    TORCH_CHECK(act.device() == wgt.device() && act.device() == out.device()
                    && act.device() == ascales.device() && act.device() == wscales.device()
                    && act.device() == act_keeper.device() && act.device() == wgt_keeper.device()
                    && act.device() == act_keeper_scale.device() && act.device() == wgt_keeper_scale.device(),
                "all GEMM tensors must share a CUDA device");
    TORCH_CHECK(act.is_contiguous() && wgt.is_contiguous() && out.is_contiguous()
                    && ascales.is_contiguous() && wscales.is_contiguous()
                    && act_keeper.is_contiguous() && wgt_keeper.is_contiguous()
                    && act_keeper_scale.is_contiguous() && wgt_keeper_scale.is_contiguous(),
                "all GEMM tensors must be contiguous");
    TORCH_CHECK(act.scalar_type() == torch::kUInt8 && wgt.scalar_type() == torch::kUInt8,
                "packed A4 and W4 operands must be uint8 bytes");
    TORCH_CHECK(ascales.scalar_type() == torch::kFloat16 && wscales.scalar_type() == torch::kFloat16
                    && act_keeper_scale.scalar_type() == torch::kFloat16
                    && wgt_keeper_scale.scalar_type() == torch::kFloat16,
                "Atom scales must be float16");
    TORCH_CHECK(act_keeper.scalar_type() == torch::kInt8 && wgt_keeper.scalar_type() == torch::kInt8,
                "Atom keeper operands must be int8");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "Atom output must be float16");
    TORCH_CHECK(act.dim() == 2 && wgt.dim() == 2 && out.dim() == 2,
                "Atom expects flattened two-dimensional GEMM operands");

    const int64_t M = act.size(0);
    const int64_t K = act.size(1) * 2;
    const int64_t N = wgt.size(0);
    TORCH_CHECK(M >= 128 && M % 128 == 0, "Atom requires M padded to a multiple of 128");
    TORCH_CHECK(K > 0 && K % 128 == 0, "Atom requires K divisible by its fixed group size 128");
    TORCH_CHECK(N > 0 && N % 128 == 0, "Atom requires N divisible by its 128-wide output tile");
    TORCH_CHECK(wgt.size(1) * 2 == K, "A and W packed widths differ");
    TORCH_CHECK(out.size(0) == M && out.size(1) == N, "output shape differs from M x N");
    TORCH_CHECK(ascales.dim() == 2 && ascales.size(0) == K / 128 && ascales.size(1) == 4 * M,
                "A scales must use ViDiT-Q Atom's [K/128, 4*M] duplicate-row layout");
    TORCH_CHECK(wscales.dim() == 2 && wscales.size(0) == K / 128 && wscales.size(1) == N,
                "W scales must use Atom's [K/128, N] layout");
    TORCH_CHECK(act_keeper.sizes() == torch::IntArrayRef({M, 128})
                    && wgt_keeper.sizes() == torch::IntArrayRef({128, N}),
                "Atom requires its fixed 128-wide int8 keeper operands");
    TORCH_CHECK(act_keeper_scale.numel() == 4 * M && wgt_keeper_scale.numel() == N,
                "keeper scales have invalid sizes");

    const auto stream = c10::cuda::getCurrentCUDAStream(act.get_device()).stream();
    DenseLayerGEMM_i4_o16(act.data_ptr<uint8_t>(),
                          wgt.data_ptr<uint8_t>(),
                          reinterpret_cast<const uint8_t *>(ascales.data_ptr<at::Half>()),
                          reinterpret_cast<const uint8_t *>(wscales.data_ptr<at::Half>()),
                          reinterpret_cast<const uint8_t *>(act_keeper.data_ptr<int8_t>()),
                          reinterpret_cast<const uint8_t *>(wgt_keeper.data_ptr<int8_t>()),
                          reinterpret_cast<const uint8_t *>(act_keeper_scale.data_ptr<at::Half>()),
                          reinterpret_cast<const uint8_t *>(wgt_keeper_scale.data_ptr<at::Half>()),
                          reinterpret_cast<half *>(out.data_ptr<at::Half>()),
                          static_cast<size_t>(M),
                          static_cast<size_t>(N),
                          static_cast<size_t>(K),
                          stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("matmul", &matmul, "ViDiT-Q official Atom W4A4 GEMM with explicit keeper buffers");
}
