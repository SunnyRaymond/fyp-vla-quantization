# QTIP port — upstream, deviations, bit accounting, runtime

## Upstream

- Repository: https://github.com/Cornell-RelaxML/qtip
- Pinned commit: `e90c6688c8dfae326a3a81b5eb032db7c6680ec0` (2025-06-21, "Update README.md"; HEAD of `main` on 2026-10-28)
- Paper: Tseng, Sun, Hou, De Sa, "QTIP: Quantization with Trellises and Incoherence Processing", NeurIPS 2024 (arXiv 2406.11235)
- License: GPL-3.0. `LICENSE` is the upstream file, copied verbatim. Everything under `methods/qtip/` that contains upstream code is GPL-3.0.

## Files

| file | origin | change |
|---|---|---|
| `quant.py` | new | `quantize_layer(weight, hessian, bits, seed=0)` wrapper, padding, sign vectors, inverse incoherence, decode check, bit accounting |
| `qtip_official/bitshift.py` | `lib/codebook/bitshift.py` (whole file) | imports; scipy k-means + `/tmp` cache replaced by `kmeans_tlut` (exact replica); `@torch.compile` removed from `update`. Changed lines are marked `# [port]` |
| `qtip_official/ldlq.py` | `lib/algo/ldlq.py` (whole file) | glog/tqdm dropped, `utils.clean()` -> local `clean()` |
| `qtip_official/math_utils.py` | `lib/utils/math_utils.py`: `block_LDL`, `regularize_H` | verbatim; `flat_to_sym` (Hessian file format) not vendored |
| `qtip_official/matmul_had.py` | `lib/utils/matmul_had.py`: `get_hadK`, `matmul_hadU`, `matmul_hadUt`, `matmul_hadU_cuda`, `matmul_hadUt_cuda`, `is_pow2`, `get_had12/20/28/36` | `get_hadK`, `matmul_hadU(t)` and the four Hadamard tables verbatim; `fast_hadamard_transform` replaced by a pure-torch Sylvester FWHT inside `matmul_hadU_cuda`; `get_had52..172` are stubs that raise |
| `qtip_official/kernel_decompress.py` | `lib/utils/kernel_decompress.py` | `@torch.compile` removed |
| `qtip_official/linear.py` | `lib/algo/finetune.py::quantize_finetune_decoder_layer` (per-Linear body, `split_for_tp=False`) and `lib/utils/data_utils.py::unpack_quip` (rcp 0) | statements verbatim; Hessian file loading, `QuantizedLinear` construction, per-layer fine-tuning, `torch.save` removed; SU/SV passed in |
| `qtip_official/hyb_kmeans_raw_q9_v2_seed0.npy` | generated | raw k-means centroids of the official HYB LUT construction (see deviation 1); sha256 of its float32 LE bytes is checked on load |
| `test_qtip.py` | new | CPU checks |

Not vendored: model/data/eval code, `quantize_llama/*`, `lib/linear/quantized_linear.py` (only its buffer layout is used for bit counting), compiled `qtip-kernels`, `lib/codebook/__init__.py` (kernel registry), fine-tuning and end-to-end fine-tuning.

## Configuration (official Llama recipe, `example.sh` + `quantize_finetune_llama.py`)

| setting | value | source |
|---|---|---|
| code | HYB = `decode_mode=quantlut_sym`, L=16, Q=`tlut_bits`=9, V=2 | example.sh, README, paper Sec. 4 |
| K (bits per weight) | `bits` (2 or 3) | `--K` |
| trellis tile | `td_x = td_y = 16` | example.sh |
| incoherence | random sign vectors SU (in), SV (out) + Hadamard on both sides (`matmul_hadUt`) | finetune.py |
| Hadamard factors | `get_hadK`: 3072 = 12·256, 6144 = 12·512, 192 = 12·16, 14336 = 28·512, 18432 = 36·512, powers of two plain | matmul_had.py |
| rounding | BlockLDLQ with 16-column blocks, `for_kernel=True` trellis order (`_PERMUTE`), tail-biting Viterbi (two Viterbi passes per tile) | ldlq.py, bitshift.py; `has_kernel(...)` is True for this config |
| Hessian damping | `regularize_H(H, sigma_reg=1e-2)`: `H += 0.01 * mean(diag H) * I` | argparse default |
| weight scale | `Wscale = rms(Wr) / (rms(lut) * scale_override)`, `scale_override = 0.9` | example.sh (argparse default is -1; the paper names no value) |
| LUT / codebook dtype | the model dtype, BF16 for Fast-WAM (`cb.to(device).to(orig_dtype)`) | finetune.py |
| Viterbi input dtype | float16 (`quantize` casts to fp16) | bitshift.py |
| Viterbi batch | 256 tiles per call (`bs = 2**(24-L)`) | bitshift.py |
| fine-tuning | none (no LUT tuning, no layer or end-to-end fine-tuning) | decided with the user |

## What `quantize_layer` does

1. Pads `out`/`in` to the smallest size that is a multiple of 16 and an official Hadamard size (only Fast-WAM layers 1024×7, 7×1024, 4096×8 are padded: 7, 8 -> 16).
2. Draws SU, SV (official expression) from a CPU generator seeded with `seed`.
3. Calls the vendored official per-Linear routine: `regularize_H`; `Wr = hadUt(hadUt(W.T * SV).T * SU)`, `HRr` likewise; `Wscale`; `block_LDL(HRr, 16)`; `LDLQ` (Viterbi per 16×16 tile); `pack_trellis` (+ CUDA-kernel bit swizzle); official proxy error; saved dict.
4. Decodes the stored trellis from its bits alone (unswizzle, `unpack_trellis`, LUT lookup, inverse `_PERMUTE`) and requires it to equal LDLQ's `hatWr / Wscale` bit for bit; raises otherwise.
5. Returns `dequant = diag(SV) · Had_m · hatWr · Had_nᵀ · diag(SU)` cropped to `[out, in]` (the `BitshiftLinear.cache_hatW` form, with Wscale inside hatWr), i.e. the weight a dense Linear needs to reproduce the QTIP layer.

## Deviations from the official code, and why

1. **LUT k-means without scipy.** Official `quantlut_sym` builds the 512×2 LUT as `scipy.cluster.vq.kmeans(torch.randn(1<<20, 2), torch.randn(512, 2))` from the global torch RNG, normalizes it to std 0.9682458365518543, and caches it in `/tmp/kmeans_9_2.pt` (the first run on a machine fixes it). Here `kmeans_tlut` draws the same tensors in the same order from `torch.Generator().manual_seed(0)` (the official script calls `torch.manual_seed(args.seed=0)` and the codebook is the first RNG consumer) and runs `_scipy_kmeans_replica`, which reproduces scipy's `_kmeans`/`_vq_small_nf`/`update_cluster_means` float32 operation order. The full seed-0 problem (2²⁰ points, 512 centroids, 70 iterations) was checked bit-identical to scipy 1.17.1 locally; the result ships as `hyb_kmeans_raw_q9_v2_seed0.npy`. The test re-checks the replica against scipy on a smaller problem when scipy is installed (skipped otherwise); `QTIP_TEST_FULL_LUT=1` regenerates the full LUT (~2.5 min CPU) and compares it with the file. The released QTIP checkpoints use LUTs that were fine-tuned per layer during end-to-end fine-tuning; this port uses the untuned k-means initialization, identical for every layer.
2. **Sign vectors.** Official: `(torch.randn(n, device=cuda).sign() + 1e-5).sign()` after `torch.manual_seed(decoder_layer_idx)`. Here: the same expression on a CPU generator seeded with `seed` (SU first, then SV), so results are identical on CPU and GPU and set per Linear by the caller.
3. **Hadamard.** The quantization path uses the official pure-torch `matmul_hadU` unchanged. `fast_hadamard_transform` (used only by the official inference path `matmul_hadU_cuda`) is replaced by a pure-torch FWHT; the test checks `matmul_hadU == matmul_hadU_cuda` for n up to 3072. Only had12/20/28/36 are vendored (covers every Fast-WAM size); sizes that would need had52..had172 are padded instead.
4. **`torch.compile`.** Official decorates `bitshift_codebook.update` with `@torch.compile`. On CUDA the wrapper compiles the vendored eager `update` (`QTIP_UPDATE=auto`). If compilation fails it switches to `fast_update`, an eager rewrite that returns the same `prev_state` and `cost` bit for bit (tested on CPU with forced ties; it avoids the gather of the expanded cost tensor). On CPU the vendored `update` runs eagerly. `QTIP_UPDATE=official|compile|fast` forces a mode; `config['viterbi_update']` records the one used.
5. **Padding.** Dimensions that are not multiples of 16 or not an official Hadamard size are zero-padded. Padded input channels get the Hessian a dead channel gets in the official code (zero row/column plus `sigma_reg · mean(diag H)` on the diagonal, with `regularize_H` applied to the true n×n block so real channels get the official damping). Padded rows/columns are stored and counted. The CUDA-kernel bit swizzle needs both padded dimensions to be multiples of 32; for 1024×7 (padded 1024×16), 7×1024 (16×1024) and 4096×8 (4096×16) the trellis stays in `pack_trellis` order (same bits, same count). The `_PERMUTE` trellis order is used for every layer.
6. **LDLQ `buf_cols`.** Official uses 128 and asserts `n % 128 == 0`. The wrapper passes the largest of 128/64/32/16 dividing n. Every Fast-WAM input width except the padded 16 is a multiple of 128, so only the three padded layers use a different value; the recursion is the same, only its blocking changes.
7. **Hessian input.** Official loads `flatH` and `mu` and forms `E[x xᵀ] = Σ + μμᵀ`; the caller passes `E[x xᵀ]` directly.
8. **Failure handling the official code lacks.** `block_LDL` returns `None` on Cholesky failure and the official code then crashes. The wrapper retries with float64 W/H (the official `--use_fp64` path), then float64 with `sigma_reg` 0.1 and 1.0; `config['attempts']` records what ran. An all-zero Hessian (official: NaN from `H / mean(diag H)`) is replaced by the identity (`config['hessian_fallback']`). A non-finite Hessian raises.
9. **Decode check.** Step 4 above is an addition; it costs CPU seconds per layer and is disabled with `QTIP_VERIFY_DECODE=0`.
10. **No fine-tuning.** The official pipeline follows each Linear's quantization with per-decoder-layer fine-tuning of SU/SV (and optionally the LUT), then end-to-end fine-tuning with `--ft_train_lut`. None of that runs here. SU/SV stay ±1, Wscale is the closed-form RMS ratio, the LUT is global.

## Bit accounting (`bits_total`, `quant.storage_bits`)

The official deployable format is `QuantizedLinear` after `hfize_llama`/`unpack_quip` (rcp 0):

- `trellis`: int16 `[(m_p/16)(n_p/16), ceil(256·K/16)]` = exactly K bits per padded weight (tail-biting: L−KV state bits are not stored).
- `SU`: `[n_p]` in the model dtype = 16 bits each.
- `SV`: `[m_p]` float32 = 32 bits each, with Wscale folded in.
- `tlut`: `[512, 2]` fp16 = 16,384 bits. Stored per layer in the official format because it is tuned per layer; without tuning it is the same tensor for every layer, so it is **not** counted per layer (`config['bits']['shared_tlut_not_counted']`; 16,384 bits once per model = 3·10⁻⁶ bpw over Fast-WAM).
- `rcp`, `tp_rank` (two int64 scalars of layer metadata) are not counted.

`bits_total = K·m_p·n_p + 16·n_p + 32·m_p`.

| layer | K=2 bits | bpw | K=3 bits | bpw |
|---|---|---|---|---|
| 3072×3072 | 19,021,824 | 2.0156 | 28,459,008 | 3.0156 |
| 1024×1024 | 2,146,304 | 2.0469 | 3,194,880 | 3.0469 |
| 18432×3072 | 113,885,184 | 2.0113 | 170,508,288 | 3.0113 |
| 3072×14336 | 88,408,064 | 2.0074 | 132,448,256 | 3.0074 |
| 1024×7 (padded 1024×16) | 65,792 | 9.18 | 82,176 | 11.46 |
| 7×1024 (padded 16×1024) | 49,664 | 6.93 | 66,048 | 9.21 |
| 4096×8 (padded 4096×16) | 262,400 | 8.01 | 327,936 | 10.01 |

Over the 584 layers of the shape list given with the task (5.922 B weights): 2.0155 bpw at K=2, 3.0155 bpw at K=3.

## Tests (`python -B test_qtip.py`, `--quick` for a <1 min subset)

Self-contained: reads only this directory; scipy optional.

1. LUT file hash; normalized std; LUT entries are ±tlut entries; replica == scipy (if installed).
2. had12/20/28/36 satisfy HHᵀ = kI; `matmul_hadU == matmul_hadU_cuda` (FWHT) and `hadU(hadUt(x)) = x`.
3. Padding table (7, 8 -> 16; 100 -> 112; all Fast-WAM sizes unpadded) and Hadamard factors.
4. `fast_update == update` bit for bit, K=2/3, with forced cost ties.
5. Wrapper vs vendored official routine with the same W/H/SU/SV (64×128 K=2 kernel bit layout; 48×192 K=3 pack_trellis layout): dequant bitwise equal, same official proxy error; unswizzled trellis == Viterbi states; official `decode_compressed` == the unswizzled decode; official stored buffers (`unpack_quip` -> `BitshiftLinear.cache_hatW` with FWHT) reproduce the dequant to max rel 1.7e-7; official `BitshiftLinear.forward` (`eval`, kernel path through `decode_compressed`) matches `x @ dequantᵀ` to 2.9e-4 (it rounds activations to fp16). If the installed torch lacks a uint16/uint32 op used by `decode_compressed`, that sub-check prints SKIPPED.
6. Proxy error vs per-row min-max RTN at the same bits, correlated H with outlier channels.
7. Shapes 32×7, 64×8, 7×64, 32×192 (12·16), 32×448 (28·16), 16×80 (20·4), 16×144 (36·4), 192×64 (out 12·16); dead channel in H; rank-deficient H with 6 decades of channel scale; all-zero H; forced Cholesky failure -> fp64 retry; determinism; bit counts.

Results from local runs (torch 2.10 CPU, i9-13900H, CPU shared with other jobs). The suite in its final form has not been run end to end; earlier full runs differed only in test sizes and lacked the fp64-retry and unswizzle checks, which were run separately and pass.

| check | result |
|---|---|
| 256×512 proxy, K=2 | QTIP 0.00580 vs RTN 0.31823 |
| 256×512 proxy, K=3 | QTIP 0.00134 vs RTN 0.06300 |
| 128×512 proxy, K=2 / K=3 | QTIP 0.00593 / 0.00138 vs RTN 0.31183 / 0.06487 |
| wrapper vs official routine | bitwise equal (64×256, 64×128, 32×64 K=2; 48×192, 16×48 K=3) |
| official HF-format decode vs dequant | max rel 1.3e-7 to 1.7e-7 |
| shapes | all pass; proxy 0.002–0.03, decode verified |
| fp64 retry | proxy 0.0194 vs 0.0194 (fp32) |

## Runtime

Measured CPU (eager official update, 14 threads, other jobs running): 128×512 took 15.0 s at K=2 and 29.6 s at K=3. One Viterbi step on 256 tiles takes about 130 ms (K=2) / 180 ms (K=3) on this CPU without contention. Under heavy contention the same 128×512 K=2 layer took 342 s.

A100 estimate. Not measured; the model below has roughly 2× uncertainty. Calibrate it from `config['seconds']` of the first layers.

- Work: per 16-column block, `ceil(m/16/256)` Viterbi batches × 2 passes × 127 steps; each step updates 2¹⁶ states per tile. The 584-layer shape list gives 32.8 M step calls and 4.0·10¹⁴ state updates per bit width (about the same for K=2 and K=3).
- Eager official `update`: ~67 bytes of memory traffic per state update (unfused sub/square/sum/gather/min/expand-copy/add), giving ≈ 6.6 h per bit width on one A100.
- `fast_update` (CUDA fallback): ≈ 5 h per bit width.
- `torch.compile(update)` (official path, default on CUDA): if inductor fuses the step into 2–3 kernels, ~10 B per state update, giving ≈ 1–2 h per bit width.
- Everything else per layer (Hadamard, Cholesky/block_LDL, LDLQ matmuls, proxy trace, CPU pack/decode check, `gc.collect`) adds about 10–20 min per bit width in total.

Total for 614 layers × 2 bit widths on one A100: ≈ 3–4 h if compile works; ≈ 10–11 h if it falls back to `fast_update` (≈ 14 h with the eager official update). That exceeds 4 h on one GPU in the fallback case and is borderline with compile. Layers are independent: with `quant_worker.py --shard s --shards 4`, the LPT balancer (weight m·n + n²) spreads every shape class across shards. On 4 A100s both bit widths take ≈ 1 h (compiled) or ≈ 2.7 h (fallback), i.e. ≈ 0.5 h or ≈ 1.4 h per bit width.

Peak GPU memory (analytic, not measured). For the 3072×14336 layer (H 14336², 822 MB per fp32 copy) the peak is ~5 Hessian-sized buffers (the caller's H, the regularized copy, HR, HRr, LRr) plus Hadamard temporaries, plus ~1.3 GB of Viterbi state (`from_state` int64 128×192×4096) and ~1 GB of weight-sized buffers: about 7–8 GB. For 18432×3072: weight-sized buffers ~2 GB, Viterbi ~1.5 GB (`from_state` 1.07 GB at K=2, 256 tiles), proxy trace temporary 18432² fp32 = 1.36 GB: about 5 GB. The fp64 retry doubles the Hessian part (~13 GB). All are well under 40 GB.

## Risks

- **torch.compile on the cluster.** It needs a working Triton/C toolchain. If it fails, the run continues on `fast_update`, about 4–5× slower. The fallback emits a warning and `config['viterbi_update']` records it.
- **Official proxy error.** It is computed with the regularized H on the padded rotated problem. The worker's own proxy (unregularized H, original coordinates) is the number to compare across methods.
- **LUT identity.** The LUT is the seed-0 k-means LUT. The authors' released models used their own first-run LUT and then tuned it per layer. One-shot QTIP numbers here are therefore below the paper's fine-tuned results by construction.
- **scale_override.** 0.9 comes from `example.sh` (a 2-bit example) and is used for both 2 and 3 bits; the repo documents no 3-bit value.
- **Three padded layers** (1024×7, 7×1024, 4096×8) cost 7–11 bpw. They hold 0.04 M of 5.9 B weights, so they change the model-wide bpw by <10⁻⁴.
- **GPL-3.0.** The vendored code is GPL-3.0; distributing this directory carries GPL obligations.
