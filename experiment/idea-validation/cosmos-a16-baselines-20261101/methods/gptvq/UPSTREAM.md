# GPTVQ port: provenance, settings, deviations

## Upstream

- Repository: https://github.com/qualcomm-ai-research/gptvq
- Pinned commit: `fd8a3ced81f767018a28a51a91af96777f885f9f` (2024-03-28, "fixed wrong bibtex citation in README.md"; the
  code itself is from `2db0930c50952da34f5a27aaad8b9e8f87b21258`, "code available").
- Paper: van Baalen et al., "GPTVQ: The Blessing of Dimensionality for LLM Quantization", arXiv:2402.15319 (v2 used
  for the settings tables and timings below).
- Licences: `_upstream/LICENSE` (Qualcomm, BSD-3-Clause-Clear style) and `_upstream/Notice` (Apache-2.0 text that covers
  the IST-DASLab/gptq-derived files) are copied unchanged. The copyright headers of every vendored file are kept, and
  the modified files carry a `[PORT]` notice.

## Files

| here | upstream | status |
|---|---|---|
| `_upstream/vq_quant.py` | `vq_quant.py` | vendored; 2 marked edits (relative import; zero-scale guard in `quantize_centroids`) |
| `_upstream/gptq.py` | `gptq.py` | vendored; marked edits: no `transformers`, relative imports, TF32 flags moved to the caller, `cuda.synchronize` guard, zero-loss guard on the two M-step SNR prints |
| `_upstream/quant.py` | `quant.py` | verbatim (needed by `gptq.py`'s `from quant import *`; its `Quantizer` is the RTN reference in the test) |
| `_upstream/uniform_quantizers.py` | `uniform_quantizers.py` | verbatim (imported by `vq_quant.py`) |
| `_upstream/LICENSE`, `_upstream/Notice` | same | verbatim |
| `quant.py` | `llama.py` (argument plumbing only) | new: `quantize_layer`, geometry and bit accounting, `VQQuantizerPort` |
| `test_gptvq.py` | none | new |

Not vendored: `llama.py` (model loading, calibration hooks, evaluation), `datautils.py`, `modelutils.py`, the `*.sh`
experiment scripts. `quant.make_quantizer` and `quant.fasterquant_kwargs` reproduce exactly how `llama.py` builds the
`VQQuantizer` (its `QClass` lambda and `configure(args.wbits, perchannel=True, sym=args.sym, mse=False)`) and calls
`GPTQ.fasterquant(...)`.

SHA-256 of the upstream blobs at the pinned commit (LF line endings, as stored in git):

```
d59d156724175e46d497a8ecc87d304689b21c0386ca4823de214cb8ca6de4b1  gptq.py
d2324ddc3ab4eab201296ffb10803cc789693d171a9caf4cfafbed11e2e610e1  vq_quant.py
0da8cae21ee72853ff69be13b784d01964cda2f92bdb84ddd0f059d3be1e7b16  quant.py               (vendored copy identical)
4d41c0c744d35be359d9f08f6d9cba9547cae1d22ee4fcf1245249672823948d  uniform_quantizers.py  (vendored copy identical)
bab328c16124f8a2b71844eafb97df1791d9af0e2f22596f1a8a261913accf8b  LICENSE                (vendored copy identical)
c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4  Notice                 (vendored copy identical)
```

## What runs for one layer

`quantize_layer(weight, hessian, bits, seed)` builds a weight-only `nn.Linear` around the (padded) weight, puts
`H = 2 * hessian` into `GPTQ.H`, and calls the official `GPTQ.fasterquant(use_vq=True, include_m_step=True, ...)`.
All of the following is upstream code:

1. `fasterquant`: dead inputs (`diag(H) == 0`) get `H[j,j] = 1` and `W[:, j] = 0`; damping `0.01 * mean(diag(H))`;
   `Hinv = chol(chol_inverse(chol(H)))^T`; columns processed in lazy blocks of 128 with GPTQ error feedback inside and
   between blocks; no act-order.
2. At every 256-column boundary, `VQQuantizer.find_params` on the current (error-updated) weights of that span: one
   codebook per `rows_per_group x 256` block, Mahalanobis seeding, 100 EM iterations whose E- and M-steps weight each
   coordinate by `diag(Hinv)` of its column, then int8 codebook quantization (`quantize_centroids`).
3. Every `vq_dim` columns: `vq_quantize` assigns each `vq_dim`-vector to the nearest codeword under the
   `1 / diag(Hinv)`-weighted distance, and the GPTQ update pushes the error into the remaining columns.
4. `lut_m_step`: Adam (lr 1e-3, 25 steps, lr x0.1 restart whenever the loss rises above the start) on all centroids
   of the layer, minimizing `tr((W - Q) H (W - Q)^T)` with the assignments fixed, then int8 re-quantization of the
   codebooks.

## Settings

Flags of the official experiment scripts (`test_bigtable.sh`, `test_scaling.sh`, `test_target_bitwidth.sh`) and what
is used here:

| flag | official scripts | here |
|---|---|---|
| `--columns-per-group` | 256 | 256 |
| `--kmeans-iters` | 100 | 100 |
| `--kmeans-init-method` | mahalanobis | mahalanobis |
| `--hessian-weighted-lookups` | on | on |
| `--include-m-step` | on | on |
| `--codebook-bitwidth` | 8 | 8 |
| `--quantize-per-codebook` | on | **off** (code default; see budget) |
| `--vq-scaling-blocksize` | -1 or 64/32 | -1 |
| `--svd-rank` | none (only in `test_target_bitwidth.sh` variants) | none |
| `--percdamp` | 0.01 (default) | 0.01 (escalated only if Cholesky fails, see deviations) |
| `--act-order`, `--static-groups` | off (raise NotImplementedError in this repo) | off |
| `--assignment-chunk-size` | none | memory bound, see deviations (result unchanged) |
| `fasterquant(blocksize=)` | 128 (not exposed by `llama.py`) | 128 |
| `--wbits` / `--vq-dim` / `--groupsize` | 2 / 4 / 65536; 3 / 2 / 8192 (the 0.125-bpv rows) | 2 / 4 / 131072; 3 / 2 / 16384 |

## Configurations and bit accounting

The paper's bits-per-value formula is `log2(k)/d + k*d*b_c/l` (k centroids of dimension d, codebook bit-width b_c,
l weights per codebook). It does not count the scale of the int8 codebooks; the count here does. The paper picks l so
that overhead is 0.125 bpv and finds higher dimension better at 2 bits (4D best) and 2D better than 1D at 3 bits (no
4D at 3 bits). So:

| bits | VQ dim | centroids | codebook block (rows x cols) | weights per codebook | paper's 0.125-bpv block |
|---|---|---|---|---|---|
| 2 | 4 | 256 | 512 x 256 | 131072 | 256 x 256 |
| 3 | 2 | 64 | 64 x 256 | 16384 | 32 x 256 |

The block is twice the paper's, which halves the codebook overhead to 0.0625 bpv. The codebook scales are stored once
per 256-column span (one FP32 scale shared by the `out/rows_per_group` codebooks of the span; this is the code's
default, `--quantize-per-codebook` off). With per-codebook FP32 scales (what the scripts pass) the side information
would be 0.06274 bpv at 2 bits and 0.06445 bpv at 3 bits, i.e. above the 0.0625 cap; with the shared scale it is
0.06254. Measured cost of sharing the scale on synthetic layers with log-normal row norms (sigma 0.3 and 1.0; 3-bit
512x512 and 2-bit 1024x256): the Hessian-weighted error changed by -0.3 %, +1.6 %, -0.0 % and +1.1 % relative to
per-codebook scales.

Exact stored bits for a layer `out x in` (`quant.geometry`):

```
in_pad   = ceil(in / d) * d                        # zero columns appended, treated as dead inputs
spans    = ceil(in_pad / 256)                      # column spans; the last may be narrower
G        = number of row groups (= out / rows_per_group for every Fast-WAM layer)
K        = 2 ** (bits * d)
bits_total = out * in_pad * bits                   # indices: (bits*d)-bit index per d-vector
           + spans * G * K * d * 8                 # int8 codebooks
           + spans * 32                            # one FP32 codebook scale per column span
```

Padded rows (only for `out` without a usable divisor, never for Fast-WAM) are not stored. The index bits of the padded
columns are counted because they share vectors with real columns.

| layer | bits | bits_total | index | codebooks | scales | bpw |
|---|---|---|---|---|---|---|
| 3072 x 3072 | 2 | 19,464,576 | 18,874,368 | 589,824 | 384 | 2.062541 |
| 3072 x 3072 | 3 | 28,901,760 | 28,311,552 | 589,824 | 384 | 3.062541 |
| 1024 x 1024 | 2 | 2,162,816 | 2,097,152 | 65,536 | 128 | 2.062622 |
| 1024 x 1024 | 3 | 3,211,392 | 3,145,728 | 65,536 | 128 | 3.062622 |

Side information on every Fast-WAM shape (both bit widths unless noted): 0.06251 to 0.06254 bpw for all layers with
out >= 3072 and 0.06262 for out = 1024. Small layers above the cap: 192x3072 at 2 bits 0.167 (a single 192-row group),
3-bit 0.0632; 1024x7 2.29 (+0.29/0.43 bpw index padding of the 8th column); 4096x8 2.00; 7x1024 4.59 at 2 bits and
0.59 at 3 bits. Over all 614 layers (5.922 B weights): 2.06257 bpw at 2 bits, 3.06255 bpw at 3 bits.

## Deviations from the official code and why

1. **Group size x2** (131072 / 16384 weights per codebook instead of 65536 / 8192): side-information budget.
2. **One codebook scale per column span** (`quantize_per_codebook=False`, the code's own default; the scripts pass
   `--quantize-per-codebook`): keeps the side information at 0.0625 bpv including the scales (numbers above).
3. **Any shape** (official code asserts `in % 256 == 0`, `out % rows_per_group == 0` and needs `in % d == 0`):
   - columns are zero-padded to a multiple of d; the pad has a zero Hessian, so upstream treats it as dead input;
   - `VQQuantizerPort.get_groupsize` drops only the `in % 256 == 0` assertion; the official loop then gives the last
     column span the remaining width (its slicing already truncates);
   - rows: `G = ceil(out / rows_per_group)` groups; if some `G' in [G, 2G]` divides `out`, `out / G'` rows per group,
     else `ceil(out / G)` rows and fewer than `G` zero rows appended to the last group. For every Fast-WAM layer
     `out` is a multiple of the nominal rows_per_group or smaller than it (one group), so no rows are padded.
   For official-compatible shapes these overrides call the official code path unchanged (tested bitwise).
4. **Hessian scale**: upstream `GPTQ.add_batch` builds `H = (2/n) * sum x x^T` with n = `inp.shape[0]` per call;
   `llama.py` passes `[1, 2048, hidden]`, so its H is `2 * 2048 * mean(x x^T)`. Here `H = 2 * mean(x x^T)` (every
   token one sample). The algorithm is invariant to a global H scale except through float rounding and Adam's eps;
   the test checks that `2048 * H` gives the same error to 4 significant digits.
5. **Singular covariance in the Mahalanobis seeding**: upstream `torch.linalg.inv` raises when a group's d x d
   covariance is exactly singular (all-zero or constant groups, dead coordinates, fewer vectors than d). Then
   `VQQuantizerPort.find_params` reruns the official `find_params` body with `pinv` for the singular groups only
   (`inv_ex` result kept for the others). Counted in `config['singular_init_fallbacks']`; no Fast-WAM-sized layer in
   the tests triggered it.
6. **All-zero codebook** (vendored edit in `quantize_centroids`): scale 0 gave 0/0 = NaN codewords and an endless
   "Outer loop: Restarting M-step" loop. Scale is replaced by 1 when it is 0; bitwise no-op otherwise.
7. **Zero proxy loss** (vendored edit in `lut_m_step`): `offset / orig_loss` with `orig_loss == 0` raised
   `ZeroDivisionError` (all-dead H, 1x1 layers). The SNR print uses inf there; nothing else changes.
8. **TF32 / determinism**: upstream disables TF32 at import time for the whole process; here it is disabled only
   inside `quantize_layer` and restored. On CUDA, `torch.use_deterministic_algorithms(True, warn_only=True)` is set
   for the call (and restored) because the gather backward in `lut_m_step` uses atomics otherwise. `seed` seeds a
   forked RNG; the Mahalanobis path uses no randomness.
9. **Assignment chunking**: `assignment_chunk_size = max(rows_per_group, 2^27 / (G * K * d))` bounds the distance
   temporaries to 512 MB each. Upstream `get_assignments` splits the 2-D `1 x d` weight tensor that `vq_quantize`
   passes along d instead of N, which crashes whenever the chunk is smaller than N = rows_per_group; hence the
   lower bound. Results are bitwise unchanged (tested).
10. **Damping fallback**: if the official 1 % damping leaves the damped H not positive definite (Cholesky or the
    inverse Cholesky fails), percdamp 0.1, 1, 10 are tried in turn; `config['percdamp']` records the value used.
11. Plumbing: no `transformers`; vendored modules loaded as a private package (`_gptvq_upstream_fd8a3ce`) so the
    upstream `quant.py` does not clash with this `quant.py`; upstream prints are captured and parsed into `config`
    (`gptq_error`, `gptq_pass_seconds`, `m_step_snr_db`, `m_step_restarts`); `torch.cuda.synchronize` only on CUDA;
    the call runs under `inference_mode(False)` so the M-step's autograd works for callers inside `inference_mode`.

## Upstream behaviours kept as they are

- The EM seeding/iterations weight coordinate j by `diag(Hinv)_j`, where `Hinv` is the upper Cholesky factor of
  `H^-1`, while the assignment in `vq_quantize` weights by `1 / diag(Hinv)_j`. The paper (Eq. 4-6) uses
  `1 / [H^-1]_jj` for both. The code's EM weighting therefore points the opposite way from the assignment weighting.
  Kept, because the baseline is the released code.
- Dead inputs: upstream zeroes their weights before quantization, so their dequantized values are codewords nearest
  to zero, not exactly zero (harmless while those inputs stay zero at inference).
- Hessian weighting is only applied for vq_dim > 1 (both configs here are 2D/4D).
- The M-step always runs 25 Adam steps at lr 1e-3 (with restarts) and re-quantizes the codebooks to int8 afterwards.
- `--vq-scaling-blocksize` is unused: its 4-bit scale per 64 weights alone is 0.0625 bpv. (Upstream also reuses the
  block's `W1_scaled` computed before the in-block GPTQ updates, so with scaling on, assignments ignore in-block
  error feedback.)

## Tests (`cd methods/gptvq && python -B test_gptvq.py`)

Self-contained (only `quant.py` and `_upstream/`). Last full local run (Windows laptop, torch 2.10 CPU, 14 threads,
machine otherwise ~85 % busy): all checks passed in 146 s. After that run only `_choose_percdamp` (temporaries freed
earlier), JSON-safe log parsing and the test's storage/JSON checks changed; those paths were re-checked with a
2-second subset (bitwise equality on 32x256, 1x1, 5x3, 7x16, all-dead H). The full suite is to be run on an
ASPIRE2A CPU node (torch 2.7.1).

Synthetic data: activations `x = z A + 0.1 n` with rank in/4, three outlier channels x20, one dead channel;
weights Gaussian with log-normal row scales. `rel_err = tr(dW H dW^T) / tr(W H W^T)` with `H = mean x x^T`.

| check | W2 | W3 |
|---|---|---|
| bitwise equal to official `fasterquant` driven like `llama.py`, 128x256 | yes | yes |
| same, 128x512 (W2, two spans) / 256x512 (W3, 4 row groups x 2 spans) | yes | yes |
| rel_err GPTVQ / per-row RTN, 128x256 | 0.00780 / 0.29549 (0.026x) | 0.00139 / 0.05421 (0.026x) |
| rel_err GPTVQ / per-row RTN, 128x512 / 256x512 | 0.00794 / 0.31530 (0.025x) | 0.00241 / 0.06593 (0.037x) |
| M-step SNR before -> after (dB), 128x256 | 18.83 -> 21.08 | 28.28 -> 28.56 |
| same seed twice, chunked assignment (>= 4 chunks), call inside `inference_mode` | bitwise equal | bitwise equal |
| rel_err with H vs 2048*H (32x256) | 0.00420 vs 0.00420 | 0.00199 vs 0.00199 |

Odd shapes (one dead input each; all finite, bits_total equal to the formula, at most K distinct vectors per codebook
block, configs JSON-serialisable with `allow_nan=False`):

| shape | W2 bpw (side, pad) | W3 bpw (side, pad) |
|---|---|---|
| 1024x7 | 4.576 (2.290, 0.286) | 5.719 (2.290, 0.429) |
| 7x1024 | 6.589 (4.589, 0) | 3.589 (0.589, 0) |
| 4096x8 | 4.001 (2.001, 0) | 5.001 (2.001, 0) |
| 67x301 | 2.836 (0.816, 0.020) | 3.216 (0.206, 0.010); 1 zero row padded |
| 100x300 | 2.548 (0.548, 0) | 3.139 (0.139, 0) |
| 1x1, 5x3 | run (singular-init fallback used) | run |

Degenerate inputs: all-dead H returns exact zeros (singular-init fallback, zero-codebook guard, zero-loss guard all
hit); all-zero W returns zeros; dead coordinates that make every 4D covariance singular still beat RTN; NaN input
raises `ValueError`. Global TF32 flags are unchanged after the calls.

## Runtime and memory

Measured on CPU (same busy laptop, torch 2.10, 14 threads), one 1024x1024 layer:

| bits | total | GPTQ pass incl. 4 x 100 EM iterations | M-step | rel_err |
|---|---|---|---|---|
| 3 | 83.7 s | 79.5 s | 4.2 s | 0.0034 |
| 2 | 250.4 s | 247.1 s | 3.4 s | 0.0131 |

The EM iterations dominate. Their cost is set by (number of d-vectors) x K x iterations, not by the group size, so
doubling the group does not change it. Traffic model per vector and EM iteration: `get_assignments` writes/reads the
`N x K x d` difference tensor about six times plus the `N x K` distances, the M-step builds an `N x K` one-hot and two
einsums over it: `4 * (6 K d + 5.25 K)` bytes. For the 1024x1024 layer that is 785 GB (W2) and 232 GB (W3), i.e.
3.2 and 2.9 GB/s effective on the CPU above, so the model tracks the measured W2/W3 ratio (3.1x measured, 3.4x model).

A100-40GB estimate for all 614 layers (5.922 B weights), per bit width:

| term | W2 (4D, K=256) | W3 (2D, K=64) |
|---|---|---|
| EM traffic | 4434 TB -> ~62 min at 1.2 TB/s | 1308 TB -> ~18 min |
| GPTQ column loop (444k / 888k vector-column steps, ~25 small kernels each) | ~2 min | ~4 min |
| M-step matmuls (3.45 PFLOP FP32, TF32 off, ~27 passes) | ~3 min | ~3 min |
| M-step gathers (444k / 888k per pass x ~27 passes, launch-bound, deterministic gather backward) | ~10 min | ~20 min |
| model total | ~1.3 h | ~0.75 h |
| paper-scaled (H100 -> A100 x2, x0.91 weights) | 4.6 h from Table 3 (GPTVQ 4D, Llama-v2-7B, 2.5 h) | 1.1 h from Table 15 (2D 3-bit with codebook update, 2195 s) |

Plan for 1.5-4.5 h (W2) and 0.75-1.1 h (W3) on one A100. Layers are independent given the BF16 Hessians, so the
614 layers can be sharded over several GPUs.

Peak GPU memory (analytic; not measured here, the cluster run should log `torch.cuda.max_memory_allocated`):

- 3072x14336 (14336^2 Hessian, 822 MB in FP32): caller's Hessian + padded copy `2*H` + upstream `G = H.clone()` +
  two Cholesky operands at a time ~ 4 x 0.82 GB, `Hinv` kept during the M-step, weights/Q/Losses/M-step temporaries
  ~1 GB, EM temporaries 2 x 0.54 GB (chunked): about 5-6 GB.
- 18432x3072 (2-bit: 36 codebooks per span, 32768 vectors each): weight-sized tensors (226 MB each) ~8-10 alive in
  the M-step ~2.3 GB, EM distance temporaries 2 x 0.54 GB (chunk 3640), one-hot `36 x 32768 x 256` 1.2 GB plus a
  possible einsum copy: about 5-6 GB.

Both are far below 40 GB. `ASSIGNMENT_CHUNK_ELEMS` (default 2^27) bounds the EM temporaries if needed.
