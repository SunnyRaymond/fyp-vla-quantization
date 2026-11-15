// 4-bit kernels of the integer LAQ runtime (compiled to PTX by nvcc in the job, loaded by int4_cuda.py).
//
//   plan4   refined rows of an operand -> row layout of its 4-bit operand
//   quant4  rotated FP32 rows -> packed signed 4-bit codes (two per byte, low nibble first) + one FP32 scale per row
//   gemm4   packed 4-bit operand x packed 4-bit weight on the INT4 tensor cores (mma m16n8k64, INT32 accumulation),
//           with the dequantization fused: (acc * row scale [+ acc of the second pass * its scale]) * weight row
//           scale + bias -> BF16
//
// Row layout of an operand (Mp rows, a multiple of 32): the refined tokens come first in tiles of 16 rows, the first
// pass of eight tokens in rows 0..7 and their second pass in rows 8..15 (P16 such tiles), then one row per other
// token. In an m16n8 accumulator fragment a thread holds rows g and g + 8 of the same columns, so the two passes of
// a refined token meet in one thread. DST[row] is the token the row writes to (-1: none).
#include <stdint.h>

#define BM 32
#define BN 128
#define KB 256                 // bytes of a K chunk (512 codes)
#define STRIDE (KB + 16)       // shared-memory row stride: 68 words, conflict-free over (g, t4)
#define THREADS 256

// ---------------------------------------------------------------- plan
extern "C" __global__ void plan4(const long long* IDX, int R, int T, int* SRC, int* PAS, int* DST, int Mp) {
    __shared__ int f[1024];
    const int tid = threadIdx.x;
    f[tid] = 0;
    for (int i = tid; i < Mp; i += 1024) { SRC[i] = -1; PAS[i] = 0; DST[i] = -1; }
    __syncthreads();
    if (tid < R) f[(int)IDX[tid]] = 1;
    __syncthreads();
    if (tid == 0) {
        int run = 0;
        for (int t = 0; t < T; ++t) f[t] = f[t] ? -1 : run++;
    }
    __syncthreads();
    if (tid < R) {
        const int p = (tid >> 3) * 16 + (tid & 7), token = (int)IDX[tid];
        SRC[p] = token; PAS[p] = 1; DST[p] = token;
        SRC[p + 8] = token; PAS[p + 8] = 2;
    }
    if (tid < T && f[tid] >= 0) {
        const int p = ((R + 7) >> 3) * 16 + f[tid];
        SRC[p] = tid; DST[p] = tid;
    }
}

// ---------------------------------------------------------------- quantize + pack
__device__ __forceinline__ float block_max(float m, float* red) {
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    m = red[0];
    for (int w = 1; w < THREADS / 32; ++w) m = fmaxf(m, red[w]);
    __syncthreads();
    return m;
}

__device__ __forceinline__ float code(float x, float s) {
    return fminf(fmaxf(rintf(__fdiv_rn(x, s)), -8.0f), 7.0f);
}

// PAS: 0 base pass (range clipped by `clip`), 1 first pass of a refined row, 2 its second pass (on the residual).
extern "C" __global__ void quant4(const float* X, const int* SRC, const int* PAS, uint8_t* Q, float* S, int K, float clip) {
    __shared__ float red[THREADS / 32];
    const int row = blockIdx.x, tid = threadIdx.x;
    const int src = SRC[row];
    if (src < 0) return;
    const int pass = PAS[row];
    const float* x = X + (size_t)src * K;
    float m = 0.0f;
    for (int i = tid; i < K; i += THREADS) m = fmaxf(m, fabsf(x[i]));
    m = block_max(m, red);
    if (pass == 0) m = __fmul_rn(m, clip);
    float s = m == 0.0f ? 1.0f : __fdiv_rn(m, 7.0f);
    float s1 = s;
    if (pass == 2) {
        float m2 = 0.0f;
        for (int i = tid; i < K; i += THREADS) m2 = fmaxf(m2, fabsf(__fsub_rn(x[i], __fmul_rn(code(x[i], s1), s1))));
        m2 = block_max(m2, red);
        s = m2 == 0.0f ? 1.0f : __fdiv_rn(m2, 7.0f);
    }
    uint32_t* q = (uint32_t*)(Q + (size_t)row * (K / 2));
    for (int j = tid; j < K / 8; j += THREADS) {
        uint32_t word = 0;
        for (int e = 0; e < 8; ++e) {
            float v = x[8 * j + e];
            if (pass == 2) v = __fsub_rn(v, __fmul_rn(code(v, s1), s1));
            word |= ((uint32_t)((int)code(v, s)) & 15u) << (4 * e);
        }
        q[j] = word;
    }
    if (tid == 0) S[row] = s;
}

// ---------------------------------------------------------------- GEMM + dequantization
__device__ __forceinline__ void mma_s4(int* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint16_t to_bf16(float y) {
    uint32_t u = __float_as_uint(y);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

__device__ __forceinline__ uint16_t finish(float t, float rs, float bias, int has_bias) {
    float y = __fmul_rn(t, rs);
    if (has_bias) y = __fadd_rn(y, bias);
    return to_bf16(y);
}

// A [Mp, K/2] packed operand, W [N, K/2] packed weight (both signed nibbles), S [Mp], RS [N], BIAS [N], DST [Mp],
// Y BF16 [T, N]. Mp % 32 == 0, N % 128 == 0, K % 512 == 0. Grid (N / BN, Mp / BM), 256 threads.
extern "C" __global__ void gemm4(const uint8_t* A, const uint8_t* W, const float* S, const float* RS, const float* BIAS,
                                 const int* DST, uint16_t* Y, int N, int K, int P16, int has_bias) {
    __shared__ __align__(16) uint8_t As[BM * STRIDE];
    __shared__ __align__(16) uint8_t Bs[BN * STRIDE];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const size_t kbytes = (size_t)(K / 2);
    int acc[2][2][4];
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) for (int e = 0; e < 4; ++e) acc[i][j][e] = 0;

    for (size_t kc = 0; kc < kbytes; kc += KB) {
        for (int i = 0; i < BM * (KB / 16) / THREADS; ++i) {
            const int v = tid + i * THREADS, row = v >> 4, col = (v & 15) * 16;
            *(uint4*)&As[row * STRIDE + col] = *(const uint4*)&A[(size_t)(m0 + row) * kbytes + kc + col];
        }
        for (int i = 0; i < BN * (KB / 16) / THREADS; ++i) {
            const int v = tid + i * THREADS, row = v >> 4, col = (v & 15) * 16;
            *(uint4*)&Bs[row * STRIDE + col] = *(const uint4*)&W[(size_t)(n0 + row) * kbytes + kc + col];
        }
        __syncthreads();
        #pragma unroll
        for (int ks = 0; ks < KB; ks += 32) {
            uint32_t a[2][4], b[2][2];
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const uint8_t* p = &As[(mi * 16 + g) * STRIDE + ks + t4 * 4];
                a[mi][0] = *(const uint32_t*)p;
                a[mi][1] = *(const uint32_t*)(p + 8 * STRIDE);
                a[mi][2] = *(const uint32_t*)(p + 16);
                a[mi][3] = *(const uint32_t*)(p + 8 * STRIDE + 16);
            }
            #pragma unroll
            for (int ni = 0; ni < 2; ++ni) {
                const uint8_t* p = &Bs[(warp * 16 + ni * 8 + g) * STRIDE + ks + t4 * 4];
                b[ni][0] = *(const uint32_t*)p;
                b[ni][1] = *(const uint32_t*)(p + 16);
            }
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi)
                #pragma unroll
                for (int ni = 0; ni < 2; ++ni) mma_s4(acc[mi][ni], a[mi], b[ni]);
        }
        __syncthreads();
    }

    for (int mi = 0; mi < 2; ++mi) {
        const int r0 = m0 + mi * 16 + g;
        const bool paired = (m0 / 16 + mi) < P16;
        const int d0 = DST[r0], d1 = paired ? -1 : DST[r0 + 8];
        const float s0 = S[r0], s1 = S[r0 + 8];
        for (int ni = 0; ni < 2; ++ni) {
            const int* c = acc[mi][ni];
            const int col = n0 + warp * 16 + ni * 8 + t4 * 2;
            const float rs0 = RS[col], rs1 = RS[col + 1];
            const float b0 = has_bias ? BIAS[col] : 0.0f, b1 = has_bias ? BIAS[col + 1] : 0.0f;
            if (paired) {
                if (d0 >= 0) {
                    const float u0 = __fadd_rn(__fmul_rn((float)c[0], s0), __fmul_rn((float)c[2], s1));
                    const float u1 = __fadd_rn(__fmul_rn((float)c[1], s0), __fmul_rn((float)c[3], s1));
                    uint16_t* y = Y + (size_t)d0 * N + col;
                    y[0] = finish(u0, rs0, b0, has_bias);
                    y[1] = finish(u1, rs1, b1, has_bias);
                }
            } else {
                if (d0 >= 0) {
                    uint16_t* y = Y + (size_t)d0 * N + col;
                    y[0] = finish(__fmul_rn((float)c[0], s0), rs0, b0, has_bias);
                    y[1] = finish(__fmul_rn((float)c[1], s0), rs1, b1, has_bias);
                }
                if (d1 >= 0) {
                    uint16_t* y = Y + (size_t)d1 * N + col;
                    y[0] = finish(__fmul_rn((float)c[2], s1), rs0, b0, has_bias);
                    y[1] = finish(__fmul_rn((float)c[3], s1), rs1, b1, has_bias);
                }
            }
        }
    }
}
