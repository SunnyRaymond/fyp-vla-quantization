# Revisiting RaBitQ and TurboQuant: A Symmetric Comparison of Methods, Theory, and Experiments

> **Reading-list role**: Critical companion to [TurboQuant](../14-turboquant/README.md) - the main citable source for the RaBitQ baseline dispute  
> **Verification**: `verified-full-text` - arXiv v2, 15 pages; accepted as a poster paper at VecDB@VLDB 2026  
> **Recommended effort**: Read pp. 1-3, 9-10, and 13 first; then audit the matching TurboQuant pages

> **Main local full text**: [arXiv v2 PDF](paper-arxiv-v2.pdf)  
> **Method-lineage companion**: [A Note on TurboQuant and the Earlier DRIVE/EDEN Line of Work](companion-drive-eden-note-arxiv-v1.pdf)  
> **Foundational reading**: [original RaBitQ + multi-bit extension](../14b-rabitq/README.md)  
> **Reproduction code**: [VectorDB-NTU/rabitq-turboquant-comparison](https://github.com/VectorDB-NTU/rabitq-turboquant-comparison)

## 1. Paper identity

| Field | Value |
|---|---|
| Authors | Jianyang Gao, Yutong Gou, Yuexuan Xu, Jifan Shi, Yongyi Yang, Shuolin Li, Raymond Chi-Wing Wong, Cheng Long |
| Version | arXiv:2604.19528v2, revised 2026-04-30 |
| Status | Accepted poster paper at the 2nd Workshop on Vector Databases, VecDB@VLDB 2026; this is a workshop paper, not a VLDB main-track research paper |
| Primary sources | [arXiv](https://arxiv.org/abs/2604.19528) · [official workshop programme](https://vecdb-ws.github.io/vldb2026/index.html) · [code](https://github.com/VectorDB-NTU/rabitq-turboquant-comparison) |
| Public context | [Jianyang Gao's open letter](https://dev.to/gaoj0017/turboquant-and-rabitq-what-the-public-story-gets-wrong-1i00) · [TurboQuant ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5c802ef38ab6e366c2ea06eee554c088-Abstract-Conference.html) |

## 2. First conclusion: what the sources do and do not say

The Gao et al. paper does **not** make a formal finding of `plagiarism`. Its defensible academic claim is narrower: TurboQuant gives an inaccurate account of the closest RaBitQ prior work, makes an unsupported characterization of RaBitQ's theory, and reports an asymmetric baseline whose runtime and recall could not be reproduced from the released artifacts.

The stronger public allegation appears in Gao's open letter, which describes a systematic omission of the shared `random rotation / Johnson-Lindenstrauss Transformation` structure and calls the experimental setup deliberately unfair. A separate group of `DRIVE` / `EDEN` authors makes the closest method-lineage allegation: `TurboQuant_mse` is an `EDEN` special case with fixed scale `S=1`, and part of TurboQuant's analysis mirrors earlier `DRIVE` / `EDEN` analysis. Therefore:

- `unfair / undisclosed baseline setup` is directly documented in the Gao technical note;
- `insufficient prior-work attribution` is argued by both the Gao open letter and the `DRIVE` / `EDEN` note;
- `plagiarism` or `洗稿` is an interpretation of the dispute, not a conclusion established by either paper or by an independent investigation.

## 3. One-sentence takeaway

Under the authors' symmetric implementation and hardware framework, RaBitQ matches or beats TurboQuant on most tested inner-product, ANN, runtime, and 2.5-bit KV-cache settings; the paper therefore rejects TurboQuant's presentation of RaBitQ as a consistently weaker baseline, while stopping short of an independent misconduct finding.

## 4. Method comparison to understand before reading results

| Component | RaBitQ | TurboQuant | Why it matters |
|---|---|---|---|
| Preprocessing | Normalize, then random rotation / JL Transformation | Normalize, then random rotation / JL Transformation | This shared first step is the central attribution dispute. |
| Codebook | Shifted uniform grid | Non-uniform Lloyd-Max centroids | This is a genuine algorithmic difference. |
| Stored state | Integer code plus one scalar | Integer code plus norm; `TurboQuant_prod` adds residual norm and 1-bit QJL code | Nominal bits per coordinate do not capture every scalar or kernel cost. |
| Inner-product estimator | Direct arithmetic on compressed integers with a precomputed scale | Codebook decoding plus QJL residual correction | The two methods should be compared using matched unbiased or matched reconstruction objectives. |
| Theory emphasized | Sub-Gaussian tail / failure-probability guarantee | Expected MSE or variance guarantee | `optimal` depends on the distortion and probability statement being compared. |

Do not reduce the paper to “both use rotation, so they are the same.” The paper itself identifies different codebooks, scaling rules, decoding paths, and guarantees. The defensible criticism is that the closest structural relation should have been stated and evaluated accurately.

## 5. Main claims and where to check them

| Claim by Gao et al. | Evidence in this paper | Locator | Evidence boundary |
|---|---|---|---|
| The methods share important structure | Side-by-side pipeline and code/storage comparison | Table 1; Sec. 2.2; PDF pp. 3-6 | Shared structure does not by itself prove derivation or misconduct. |
| RaBitQ has the stronger failure-probability guarantee | RaBitQ uses a sub-Gaussian tail with `log log(1/delta)` bit dependence; converting TurboQuant variance via Chebyshev yields `log(1/delta)` | Sec. 3; PDF pp. 6-7 | This compares tail guarantees; TurboQuant's expected-MSE near-optimality is a different statement. |
| The original runtime baseline was asymmetric | The note says TurboQuant used a translated Python RaBitQ on single-core CPU with multiprocessing disabled, while TurboQuant ran on A100 | Sec. 4.2; PDF p. 9, footnote 7 | Hardware detail relies partly on private email quoted by the authors; the complete email record is not public in the paper. |
| Symmetric timing reverses the ranking | RaBitQ GPU is faster than TurboQuant GPU at `d=200/1536/3072`; fast RaBitQ variants are faster still | Table 2; PDF pp. 9-10 | Results come from the RaBitQ authors' repository; not independently rerun in this reading packet. |
| TurboQuant recall advantage does not reproduce | Ten seeded runs show RaBitQ above both TurboQuant variants across the tested datasets and bit-widths | Figure 3; PDF pp. 10-11 | TurboQuant did not specify which variant or seed aggregation produced its original curves. |
| KV-cache results favor RaBitQ at 2.5-bit | Matched cache framework gives NIAH `0.951` vs `0.709`; LongBench-E averages `48.64` vs `47.78` on Llama-3.1-8B | Figure 4 and Table 3; PDF pp. 11-13 | Gao et al. repaired the TurboQuant code and changed the deterministic NIAH scorer; do not compare these absolute scores directly with TurboQuant's original NIAH `0.997`. |

## 6. The baseline problem in numbers

TurboQuant arXiv v1 reported the following 4-bit quantization times for 100,000 vectors:

| Method / source | `d=200` | `d=1536` | `d=3072` |
|---|---:|---:|---:|
| RaBitQ, TurboQuant arXiv v1 Table 2 | 597.25 s | 2267.59 s | 3957.19 s |
| TurboQuant, TurboQuant arXiv v1 Table 2 | 0.0007 s | 0.0013 s | 0.0021 s |
| RaBitQ CPU, Gao et al. Table 2 | 0.125 s | 1.003 s | 4.176 s |
| RaBitQ GPU, Gao et al. Table 2 | 0.009 s | 0.065 s | 0.152 s |
| TurboQuant GPU, Gao et al. Table 2 | 0.011 s | 0.114 s | 0.276 s |

This is the strongest concrete reason to reject the original speed comparison as a fair baseline. It is also version-sensitive: the current official ICLR proceedings PDF removes the old timing table entirely, while retaining qualitative language that RaBitQ lacks vectorized/GPU support and is slower on CPU.

## 7. The separate DRIVE / EDEN lineage claim

Read the local `DRIVE` / `EDEN` note only after understanding Gao et al. It argues:

1. `TurboQuant_mse` and biased/unbiased `EDEN` share the same rotate -> Lloyd-Max quantize -> inverse-rotate pipeline; the reconstruction scale is the key difference.
2. `TurboQuant_mse` fixes `S=1`, whereas `EDEN-biased` chooses the MSE-minimizing scale and `EDEN-unbiased` chooses an unbiased scale.
3. `TurboQuant_prod` spends `b-1` bits on the biased first stage and 1 bit on QJL residual correction; the authors report that direct `b`-bit `EDEN-unbiased` is more accurate.
4. The shifted-Beta coordinate distribution, Lloyd-Max codebook, and randomized Hadamard replacement were already analyzed in the earlier `DRIVE` / `EDEN` line.

The most useful page is PDF p. 4: it places all three scale choices in one pseudocode. This is stronger evidence of method overlap than a generic “both use random rotation” observation.

## 8. Counter-position from the TurboQuant authors

In their public OpenReview response, the TurboQuant authors deny deriving the method from RaBitQ and argue that random rotation is standard prior art. They acknowledge that RaBitQ's optimality can be supported by its deeper proofs and say the description will be corrected. They also argue that the disputed runtime benchmark is not material to TurboQuant's core compression-quality contribution.

That response matters because it prevents a one-sided reading. It does not, however, make the asymmetric runtime comparison fair; it changes the question to whether that table is central enough to affect the paper's main contribution.

## 9. My assessment

### Strongly supported

- TurboQuant arXiv v1 presented RaBitQ runtime numbers obtained under an unmatched setup and did not disclose the decisive CPU/GPU and parallelism differences in the paper.
- The current ICLR proceedings version removes that timing table.
- RaBitQ and TurboQuant share a closer structural relationship than the final TurboQuant description makes explicit.
- TurboQuant authors publicly accepted that RaBitQ's optimality needed more accurate credit.

### Plausible but not independently established here

- That the omissions or benchmark choices were deliberate rather than mistaken.
- That every result in Gao et al.'s reproduction will survive an independent clean-room rerun.
- That method overlap rises to `plagiarism`; this requires provenance and intent evidence beyond structural similarity.

### Important nuance

The two critique papers attack different layers. Gao et al. is strongest on `RaBitQ attribution + baseline symmetry + reproducibility`. Ben-Basat et al. is strongest on `earlier algorithmic lineage + missing EDEN scale + accuracy`. Keep them separate when presenting the controversy.

## 10. How to read it

### 20-minute route

1. PDF pp. 1-3: scope, three disputed dimensions, Table 1.
2. PDF pp. 9-10: read the full runtime disclosure and Table 2.
3. PDF pp. 10-11: inspect seed handling and recall discrepancy.
4. PDF p. 13: read Table 3 and Conclusion.
5. Return to [TurboQuant final PDF](../14-turboquant/paper-iclr-2026-final.pdf), pp. 9-10 and 15.

### 90-minute audit route

1. Gao et al. Sec. 2: rewrite both algorithms in the same notation yourself.
2. Gao et al. Sec. 3: separate `expected MSE`, `variance`, and `high-probability tail` guarantees.
3. Gao et al. Sec. 4: create a protocol sheet for hardware, code version, seeds, datasets, scorer, and model identity.
4. `DRIVE` / `EDEN` note pp. 1-7: trace the scale parameter `S` and compare `EDEN-unbiased` with `TurboQuant_prod`.
5. TurboQuant final pp. 2-7, 9-10, and 15: mark what the final version now says, omits, or still leaves ambiguous.

## 11. Reading questions

1. Is `random rotation + distribution-aware scalar codebook` the right unit of novelty, or is the real novelty the specific guarantee and application integration?
2. Which comparison is matched: `RaBitQ_prod` vs `TurboQuant_prod`, or reconstruction-oriented variants?
3. Does TurboQuant's expected-MSE guarantee answer the same reliability question as RaBitQ's tail bound?
4. What exact implementation produced TurboQuant arXiv v1 Table 2, and why is it absent from the proceedings final?
5. Can the released reproduction repository regenerate every figure from a clean environment without unpublished kernels?
6. Would a neutral third-party rerun on matched CPU and GPU implementations preserve the ranking?

## 12. Weekly meeting card

- **Paper**: *Revisiting RaBitQ and TurboQuant*.
- **Best-supported criticism**: the original RaBitQ runtime baseline was asymmetric and undisclosed.
- **Method overlap**: both normalize and random-rotate; codebook, scale, estimator, and theoretical guarantee differ.
- **Do not overclaim**: the sources document an attribution and reproducibility dispute, not an independent plagiarism verdict.
- **Next verification**: clean-room reproduction of Table 2, Figure 3, and the matched KV-cache framework.
