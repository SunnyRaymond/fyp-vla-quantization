# QuIP# port: upstream, deviations, bit accounting

## Upstream

- Repository: https://github.com/Cornell-RelaxML/quip-sharp
- Pinned commit: `1d8f873e9a2a8b86b12bb1064c312c5689b77d98` (2024-10-29, "Update README.md"; HEAD of `main` when cloned
  on 2026-10-28; the repo README says it is no longer under active development).
- Paper: Tseng, Chee, Sun, Kuleshov, De Sa, "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and
  Lattice Codebooks", ICML 2024.
- License: GPL-3.0. `LICENSE` in this directory is the upstream file, unchanged. The vendored files carry a header
  naming the source file and commit; every edit inside them is marked `PORT:`.

## Files

| File | Content |
|---|---|
| `quant.py` | `quantize_layer(weight, hessian, bits, seed=0) -> dict(dequant, bits_total, config)`; padding, seeding, decode of the stored format, bit count |
| `quipsharp_official.py` | vendored algorithm (below) |
| `quipsharp_hadamard.py` | vendored pure-torch Hadamard transform and Hadamard matrices |
| `test_quipsharp.py` | CPU tests (`python -B test_quipsharp.py`) |
| `LICENSE` | upstream GPL-3.0 text |

## Vendored code

| Upstream file | Vendored (verbatim unless listed under deviations) |
|---|---|
| `lib/algo/quip.py` | `RHT_H`, `RHT_W`, `incoherence_preprocess`, `incoherence_process`, `low_rank_preprocess`, `low_rank_process`, `LDLQ`, `LDLQ_buffered`, `LDLQ_buffered_lowmem`, `quantize`; body of `quantize_linear` as `quantize_linear_inmem` |
| `lib/utils/math_utils.py` | `block_LDL`, `regularize_H` |
| `lib/utils/misc.py` | `clean` |
| `lib/codebook/latticee8_padded12.py` | `get_norm12`, `get_packed_abs_grid`, `get_abs_grid`, `get_full_grid`, module-level grid tables, `E8P12_codebook` (`__init__`, `round`, `fast_quantize_part`, `quantize`, `maybe_pack_idxs`) |
| `lib/codebook/latticee8_padded12_rvq3bit.py` | `get_e81bgrid`, E81B tables, `E8P12RVQ3B_codebook` (`__init__`, `round`, `fast_quantize_part`, `quantize_e8p`, `quantize`, `maybe_pack_idxs`). Its grid functions are byte-identical copies of the 2-bit file's and are vendored once. |
| `lib/utils/matmul_had.py` | `matmul_hadU`, `matmul_hadUt`, `is_pow2`, `get_had12/20/28/36/52/60`, `get_hadK` (pruned, see below) |

Stripped: model loading, data/devset code, Hessian collection, fine-tuning (`lib/algo/finetune.py`, `lib/utils/finetune.py`),
glog, the `quiptools_cuda` inference kernels (`by_idxs`, `QuantizedE8P12Linear`, `QuantizedE8P12RVQ3BLinear`), the
`fast_hadamard_transform` path, the 4-bit codebook, and `get_had108/116/124/140/156/172`.

`test_quipsharp.py` checks this table mechanically when `QUIPSHARP_UPSTREAM` points to a clone: 36 vendored functions
are AST-identical to upstream; exactly `get_hadK`, `incoherence_preprocess` and `quantize` differ (the `PORT:` edits
below).

## Configuration (upstream defaults)

Common (argparse defaults of `quantize_llama/quantize_finetune_llama.py`): `incoh_mode='had'` (random-sign Hadamard on
both sides), `sigma_reg=1e-2`, `lora_rank=0`, `rescale_WH=False`, `quip_tune_iters=10`, `use_fp64=False` (official
automatic fp64 retry kept), buffered LDLQ (`LDLQ_buffered`, `buf_cols=128`), block LDL with block size =
codebook dimension 8. Upstream's per-weight RMS normalisation before quantization and the fusion of that RMS and
`Wscale` into `SV` are kept.

| bits | codebook | Wscale divisor | residual | stored code bits / weight |
|---|---|---|---|---|
| 2 | `E8P12` (E8 padded lattice, 2^16 codewords / 8 weights) | `scale_override=0.9` (README: "We suggest using 0.9 for E8P12"; also `quantize_llama/example.sh`) | - | 2 |
| 3 | `E8P12RVQ3B` (E8P12 + 1-bit E8 residual, 2^8 codewords / 8 weights) | codebook default `opt_scale=0.98` (README: "default scales for 3 and 4 bit models") | `opt_resid_scale=2.04` (default) | 3 |

Every value is written to `config["args"]` / `config` per layer, together with the padded shape, Hadamard factors,
fp64-fallback flag, the official log lines (`config["official_log"]`), the proxy error on the given Hessian and timing.

**No fine-tuning.** Neither the within-block fine-tuning of `quantize_finetune_llama.py` nor `finetune_e2e_llama.py`
(sign-vector/layernorm tuning) is run; this is the one-shot PTQ setting agreed for the comparison and corresponds to
upstream's `--ft_epochs 0` ("To reproduce earlier QuIP# results without fine-tuning"). Hessians come from the BF16
model (upstream also uses Hessians of the unquantized model, `hessian_offline_llama.py`).

## Pipeline (per Linear)

1. `H` = given `mean x x^T`; lower triangle mirrored (upstream stores Hessians as `sym_to_flat` lower triangles and
   rebuilds them with `flat_to_sym`). Upstream stores the covariance and adds `mu mu^T` back in `quantize_linear`, i.e.
   it quantizes with the same second-moment matrix.
2. Zero-pad to `out_pad x in_pad` (next section).
3. Vendored `quantize_linear` body: `W /= RMS(W)`, `regularize_H` (divide by mean diagonal, add `sigma_reg` to the
   diagonal), then `quantize`: random signs `SU`, `SV` (drawn after `torch.manual_seed(seed)`), RHT of `W` and `H`,
   Cholesky, block LDL (block 8), `W /= Wscale`, buffered LDLQ + 10 tune passes with the E8P / E8P-RVQ codebook,
   inverse RHT, `maybe_pack_idxs`, `SV <- fp16(SV * Wscale) * RMS(W)` in fp16.
4. `dequant`: the stored representation is decoded exactly like the official `QuantizedLinear` (packed `Qidxs` ->
   unpack -> codebook table -> `+ resid / 2.04` for 3 bit -> inverse RHT with the stored fp16 `SU`, `SV`), cropped
   to `out x in`. `x @ dequant.T` is the QuIP# layer output.

## Deviations from upstream and why

1. **Hadamard transform.** The official quantizer already uses the pure-torch `matmul_hadU` (butterfly FWHT +
   `hadK` bmm); it is vendored verbatim. Only the inference-side `fast_hadamard_transform` path is dropped. The test
   checks `matmul_hadU(X) == X @ (hadK kron Sylvester)^T / sqrt(n)`, i.e. the same transform the CUDA kernel applies
   (`view(-1, K, n/K)`, FWHT on the last axis, `hadK @`), for 8, 64, 192 = 12*16, 448 = 28*16, 576 = 36*16, 1024,
   3072 = 12*256, and that had12/20/28/36/52/60 satisfy `H H^T = K I`.
2. **`get_hadK` pruned** to K in {60, 52, 36, 28, 20, 12, 1} (same branch order); the 108-172 matrices (1.6 MB of
   literals) are not vendored. No Fast-WAM width needs them (3072 = 12*256, 6144 = 12*512, 192 = 12*16,
   14336 = 28*512, 18432 = 36*512, 256/1024/4096 = 2^k). A width that would need them is padded instead.
3. **Padding** (upstream asserts `out % 2 == 0`, `in % 4 == 0`, and `LDLQ_buffered` / the official storage layout
   `Qidxs.view(m//16, n//64, 8, 4)` need `out % 16 == 0`, `in % 64 == 0` and a `get_hadK` size on both sides). Each
   side is padded to the smallest size that is a multiple of 16 (out) / 64 (in) and a supported Hadamard size. Padded
   input channels are zero weight columns and zero Hessian rows/cols, so `regularize_H` treats them exactly like
   upstream treats dead channels; padded output rows are zero rows and are cropped. Padded entries are stored codes and
   are counted. Fast-WAM: only `1024x7` (in 7 -> 64), `4096x8` (in 8 -> 64) and `7x1024` (out 7 -> 16) are padded.
4. **`buf_cols`.** `quantize` hard-codes `buf_cols=128` (`in % 128 == 0`); the wrapper passes `128` when possible
   and `gcd(in_pad, 128)` (= 64) otherwise. Only the two padded `in = 64` layers use 64. The buffer only regroups
   the same LDLQ arithmetic (the 448-wide test case exercises it).
5. **Cholesky failure.** `torch.linalg.cholesky` raises in torch >= 1.13 instead of returning NaNs; the exception is
   mapped to upstream's "return None -> recompute in fp64" path. If upstream's fp64 retry also fails (it then does a
   bare `raise Exception`), the wrapper retries with `sigma_reg * 10` (at most 3 times, recorded as `sigma_retries`).
   Never triggered in the tests, including rank-deficient, slightly indefinite (-0.5 % of mean diagonal), dead-channel
   and 1e4-outlier Hessians.
6. **All-zero Hessian** -> identity (upstream divides by a zero mean diagonal). **All-zero weight** -> zero
   `dequant` with the normal bit count (upstream divides by `RMS(W) = 0`). Neither is expected on Fast-WAM.
7. **`utils.clean()`** keeps `torch.cuda.empty_cache()` but drops `gc.collect()`: it is called ~24 times per layer and
   cost ~40 ms per call (80 % of the time of small layers in a profile). No numerical effect; tensors are freed by
   reference counting (upstream creates no large-tensor reference cycles).
8. **glog -> `logging`**; the official messages are captured per layer into `config["official_log"]`.
9. **Seeding.** Upstream calls `torch.manual_seed(block_idx)` once per decoder block and draws `SU`/`SV` for the
   block's Linears in sequence on `H.device`. Here each call does `torch.manual_seed(seed)` inside
   `torch.random.fork_rng` (the caller's RNG state is restored) and draws on the weight's device, so the signs are
   deterministic per `(seed, device type)` but are not the signs of any upstream checkpoint. Pass a distinct seed per
   layer (e.g. its index) to keep the per-layer sign vectors independent, as upstream's are.
10. **TF32 forced off** inside the call (`set_float32_matmul_precision('highest')`, `allow_tf32=False`, restored on
    exit). Upstream relies on the torch default (fp32); a caller that enabled TF32 globally would otherwise change
    the LDL / LDLQ / codebook search numerics.
11. **Decode without `quiptools_cuda`.** `decompress_packed_e8p` / `decompress_e81b_packed` are replaced by
    `unpack_qidxs` (exact inverse of the vendored `maybe_pack_idxs`) and a lookup in upstream's own decode table
    `_E8P_GRID` (`get_full_grid`) and `e81b_grid`. Tests: official pack -> unpack is lossless for random and real
    indices; table lookup of every quantizer index reproduces the quantizer's codeword bit-exactly.
12. **`dequant` uses the stored fp16 `SV`** (`Wscale` and `RMS(W)` fused, as upstream saves it), like the deployed
    `QuantizedLinear`. Because `|SV|` is one value per layer, this differs from the quantizer's fp32 `hatW` by a single
    factor `config["sv_fp16_factor"]` (|factor - 1| = 1.0e-4 / 5.8e-5 in the test; bounded by fp16 rounding, < 1e-3).
13. **Per-Linear quantization.** Upstream's Llama driver stacks q/k/v and up/gate into one fused matrix with shared
    `SU`. Here every one of the 614 Linears is quantized on its own with its own Hessian, as the job interface
    requires; side information is counted per Linear.
14. Kept although unused: `kron` incoherence branch (broken upstream too: it calls a non-existent
    `utils.rand_ortho_butterfly_noblock`), low-rank (`lora_rank=0`), `rescale_WH` (off).
15. The grid construction is the upstream Python loop over 2^16 codes, run once at import (~10 s CPU); each codebook
    instance costs ~2 s more on first use. Not optimised, to stay verbatim.

## Bit accounting

`bits_total` is the size of every persistent buffer of the official `QuantizedLinear` state dict for this layer,
taken from the actual packed tensors (and asserted equal to the closed form):

```
bits_total = bits * out_pad * in_pad      # Qidxs: int64, 2 bit -> (out_pad, in_pad/32);
                                          #        3 bit -> concat[(out_pad, in_pad/32) E8P, (out_pad, in_pad/64) E81B]
           + 16 * in_pad                  # SU: fp16 random signs (upstream stores them as fp16)
           + 16 * out_pad                 # SV: fp16 signs * Wscale * RMS(W) (all per-layer scales fused here)
           + 32                           # Wscale buffer (fp32 scalar; 1.0 after fusion, still stored)
           + 64                           # codebook_id buffer (int64 scalar)
```

- Residual scale (2.04), `opt_scale`/`scale_override`, the E8P table (256 x int32 packed abs grid) and the E81B
  table (256 x 8) are global algorithm constants, not per-layer data, and are not counted (as the integer grid of
  scalar RTN is not); counted once per model they would add ~41 kbit (< 1e-5 bpw over 5.9 B weights).
- `had_left_T` / `had_right` are non-persistent constants in upstream and are not counted.
- Padding is counted (codes, `SU`, `SV` of the padded size).

| Layer | bits | bits_total | bpw |
|---|---|---|---|
| 3072 x 3072 | 2 | 18,972,768 | 2.01043 |
| 3072 x 3072 | 3 | 28,409,952 | 3.01043 |
| 1024 x 1024 | 2 | 2,130,016 | 2.03134 |
| 1024 x 1024 | 3 | 3,178,592 | 3.03134 |
| 1024 x 7 (padded 1024 x 64) | 2 | 148,576 | 20.73 |
| 4096 x 8 (padded 4096 x 64) | 2 | 590,944 | 18.03 |
| 7 x 1024 (padded 16 x 1024) | 2 | 49,504 | 6.91 |

Over the shape table of the task (584 listed Linears, 5.92 B weights; the three padded layers included):
2-bit 2.01026 bpw, 3-bit 3.01031 bpw.

## Tests (`python -B test_quipsharp.py`, local torch 2.10 CPU)

All pass in 84 s wall (30 s of it is the import-time grid construction, slowed by another CPU job running at the
same time). Synthetic main problem: `W` 256 x 512 (row scales 0.5-2), `H = X^T X / N` from 4096 correlated
activations with 3 x20 outlier channels and 2 dead channels.

- **Hadamard:** had12/20/28/36/52/60 are Hadamard; `hadK_supported` agrees with `get_hadK` for n = 1..4999; pure-torch
  `matmul_hadU`/`matmul_hadUt` equal explicit `hadK kron Sylvester / sqrt(n)` (rel < 1e-6; the residual is upstream's
  float32 `sqrt(n)`).
- **Codebooks / format:** E8P table 65536 distinct codewords, E8P search set 1366, E81B 256. Official
  `maybe_pack_idxs` -> port `unpack_qidxs` is lossless (2 and 3 bit, random and real indices); index -> table
  lookup reproduces the quantizer's codewords bit-exactly.
- **Wrapper == official routine:** for 2 and 3 bit, `hatW`, packed `Qidxs`, `SU`, `SV` from the wrapper are bitwise
  equal to calling the vendored `quantize_linear` body directly with the same config and `torch.manual_seed(0)`; the
  caller's RNG state is unchanged.
- **dequant == official decode:** decoding the stored codes with fp32 `SV` reproduces the official `hatW`
  (rel 1.5e-7 / 1.7e-7); the returned `dequant` (fp16 `SV`) equals `c * hatW` with `c - 1` = +1.0e-4 / -5.8e-5
  (rel 1.5e-7 / 1.7e-7); the official `hatW` mapped into the incoherent basis with explicit Hadamard matrices equals
  the decoded codewords (rel 3.8e-7 / 3.6e-7); `dequant` = `diag(SV) H_m vals H_n^T diag(SU)` with explicit matrices
  (rel 2.6e-7 / 2.7e-7). 2-bit codewords are on the E8 + 1/4 grid (4 * value odd integer).
- **Proxy error** `tr(dW H dW^T) / tr(W H W^T)` on the unregularized `H`, vs. per-row asymmetric min-max RTN at the
  same bit width:

  | bits | QuIP# | RTN | ratio |
  |---|---|---|---|
  | 2 | 0.0073 | 0.3218 | 0.023 |
  | 3 | 0.0014 | 0.0588 | 0.024 |

  3-bit error is 0.19x the 2-bit error. The same ordering holds on every other shape tested (QuIP# below RTN on all
  shapes with in >= 64).
- **Shapes:** 1024x7, 4096x8, 7x1024, 192x384 (12-type both sides), 224x448 (28-type both sides, `buf_cols` 64),
  576x128 (36-type out), 1x1, 5x3; all finite, correct shape, `bits_total` equal to the closed form. Fast-WAM widths
  3072/14336/18432/6144/192/1024/4096/256 need no padding and map to K = 12/28/36/12/12/1/1/1.
- **Robustness:** rank-deficient `H` (32 samples, n = 256), dead channels + 1e4 outlier, `H - 0.5 % mean(diag) I`,
  all-zero `H`, all-zero `W`: all succeed without the fp64 fallback or sigma retries.
- **Determinism:** same seed -> bitwise equal `dequant`; another seed -> different.
- **Verbatim check** (with `QUIPSHARP_UPSTREAM`): 36 vendored functions AST-identical; changed: `get_hadK`,
  `incoherence_preprocess`, `quantize`.

## Runtime and memory

### Measured on CPU (local, torch 2.10 CPU, 8 threads, another CPU-heavy job running concurrently)

| Layer | bits | wall time | per LDLQ block step |
|---|---|---|---|
| 256 x 512 | 2 / 3 | 1.1-3.4 s / 2.2-3.2 s | - |
| 1024 x 1024 | 2 | 7.7 s | 5.5 ms |
| 1024 x 1024 | 3 | 30.3 s | 21.5 ms |
| 3072 x 3072 | 2 | 213 s | 50 ms |
| 3072 x 14336, `quip_tune_iters=1` (memory run) | 2 | 333 s | - |
| 18432 x 3072, `quip_tune_iters=1` (memory run) | 2 | 167 s | - |

Plus a one-off ~10 s import (grid construction) and ~2 s per codebook on first use, per process.

### Estimated A100 time (not measured on a GPU)

The cost is the upstream LDLQ structure: one block step per 8 input columns, repeated for the LDLQ pass and the
10 tune passes, i.e. `11 * in/8` block steps per layer. Each block step is ~134 (2 bit) / ~146 (3 bit) aten ops with
two host syncs (`torch.where` on a data-dependent mask in `fast_quantize_part`), counted with a dispatch-mode op
counter. GPU work per step is small (largest: the `out x 1366` codeword-distance matrix + argmax, ~0.1 ms at
out = 18432; the tune pass reads the `in x out` residual once per step, <= 226 MB, ~0.15 ms), so steps are
dispatch/sync bound at roughly 0.8-1.2 ms.

- Block steps over the task's shape table: 2,442,352 per bit width (in = 3072: 1.41 M, 14336: 0.59 M, 1024: 0.26 M,
  4096: 0.18 M).
- 2.44 M x 0.8-1.2 ms = 33-49 min, plus per-layer fixed cost (two Cholesky factorizations, RHTs, upstream's
  `H.cpu()` round trips: ~1-2 s for the 30 in = 14336 layers, < 0.3 s otherwise; ~3-5 min total).
- Estimate: ~40-55 min for 2 bit, ~45-60 min for 3 bit; ~1.5-2 h for both bit widths on one A100-40GB.

### Memory

Measured as process peak commit (Windows private bytes) on CPU, 3 bit, `quip_tune_iters=0` (the tune passes add
one `in x out` residual, 0.16-0.21 GiB, for these layers), minus the commit before the call (which already holds the
caller's `W` and `H`):

| Layer | H | W | quantizer peak above caller state |
|---|---|---|---|
| 3072 x 14336 (largest H) | 0.77 GiB | 0.16 GiB | 4.04 GiB |
| 18432 x 3072 (largest W) | 0.04 GiB | 0.21 GiB | 0.98 GiB |

The peak is the `n x n` stage: the wrapper's symmetric `H`, the regularized copy handed to upstream, upstream's own
clone, and the RHT/Cholesky/block-LDL temporaries (about 5-6 live `n x n` fp32 buffers). The GPU path holds the same
tensors (upstream's `.cpu()` copies in `LDLQ_buffered` go to host memory), so the expected device peak for the
largest layer is ~5-7 GiB including caller tensors and CUDA context: well under 40 GB. Host RAM: upstream copies
`H` and `W` to the host inside `LDLQ_buffered` (< 1 GiB).

## Risks

- **Runtime is an estimate.** No GPU run was possible here; the official per-8-column loop is dispatch/sync bound.
  If the estimate is off, the 12 h `a16q1` budget shared with QTIP/GPTVQ is the constraint; `seconds` in each
  layer's config gives the real per-layer cost after the first layers.
- **Seeds.** `quantize_layer` draws `SU`/`SV` from `seed`; calling every layer with the same seed reuses the same
  sign vectors for same-width layers. Pass the layer index.
- **2-bit scale.** `scale_override=0.9` follows the README; the argparse default (-1 -> `opt_scale` 1.03) is a
  different, also "official" setting. It changes 2-bit numbers; it is recorded as `config["wscale_divisor"]`.
- **Hessian damping on outlier-heavy activations.** `sigma_reg` is relative to the mean diagonal, which massive
  activation channels dominate; this is upstream behaviour and was kept.
- **Tiny layers** (1024x7, 4096x8, 7x1024) are padded to the official layout and carry 18-21 / 6.9 bpw each; this
  is counted and moves the model average by < 1e-4 bpw.
- **fp16 `SV`.** A layer with `RMS(W)` below ~6e-5 would put `SV` in fp16 subnormals; `config["sv_fp16_factor"]`
  reports the realised scale error per layer.
- **License.** The vendored code is GPL-3.0; anything distributed together with it inherits those terms.
