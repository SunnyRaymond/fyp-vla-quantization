# REPA / iREPA → VLA representation alignment reading companion

> **Evidence snapshot:** 2026-08-31  
> **Reading status:** local full text prepared; claims below were checked against the local PDFs  
> **Scope:** first understand `REPA` and `iREPA`, then determine which parts have already appeared in `VLA` / `WAM` research  
> **Boundary:** this is a companion route to [Flow Matching](../17-flow-matching/README.md), not a new advisor-assigned core paper

## Bottom line first

The broad idea is **already substantially implemented**:

- `FLARE` explicitly adapts `REPA` to a flow-matching `VLA` policy by aligning intermediate future-token states to future-observation embeddings.
- `Spatial Forcing` and `VEGA` align `VLA` visual features to frozen spatial/3D teachers.
- `AGRA` explicitly “repurposes representation alignment” inside a `WAM`; `Robust-WAM` aligns action-stream future queries to frozen `DINOv3` targets.

The narrower idea still appears open in the literature checked here:

> I did not find a verified robot-policy paper that implements the exact `iREPA` pair of **`3×3 Conv projector + spatial normalization of patch-token targets`** and evaluates it under a matched closed-loop `VLA` protocol.

Therefore, “add a cosine feature-alignment loss to a `VLA`” is no longer a defensible novelty claim. A potentially defensible project must test what `iREPA` adds beyond these prior methods: local spatial structure, target normalization, projector inductive bias, temporal alignment, and action grounding.

The evidence and exclusions behind this conclusion are in [PRIOR_ART_MAP.md](PRIOR_ART_MAP.md). File provenance, versions, page counts, and hashes are in [SOURCE_INVENTORY.md](SOURCE_INVENTORY.md).

## Files in this package

### Core reading

- [REPA — ICLR 2025 final](repa-iclr-2025-final.pdf)
- [iREPA — ICLR 2026 final](irepa-iclr-2026-final.pdf)

### Direct and partial implementations

- [FLARE — CoRL 2025 final](prior-art-flare-corl-2025-final.pdf)
- [Spatial Forcing — arXiv v2](prior-art-spatial-forcing-arxiv-v2.pdf)
- [FRAPPE — arXiv v1](prior-art-frappe-arxiv-v1.pdf)
- [FutureVLA — arXiv v1](prior-art-futurevla-arxiv-v1.pdf)
- [VEGA — arXiv v1](prior-art-vega-arxiv-v1.pdf)
- [AGRA — arXiv v1](prior-art-agra-arxiv-v1.pdf)
- [SAM3D-Guided — arXiv v1](prior-art-sam3d-vla-arxiv-v1.pdf)
- [Robust-WAM — arXiv v1](prior-art-robust-wam-arxiv-v1.pdf)
- [Mind-VLA — arXiv v2](prior-art-mind-vla-arxiv-v2.pdf)

### Adjacent, not REPA-equivalent

- [ReconVLA — local arXiv v1](adjacent-reconvla-arxiv-v1.pdf): an auxiliary reconstruction objective, not frozen-teacher feature regression

## 20-minute route

Use this route to decide whether the idea is worth a deeper pass.

1. `REPA`: read Abstract, Figure 1, Figure 3, and §3.3.
2. Write the two losses in one line: `L = L_denoise/velocity + λ L_REPA`.
3. `iREPA`: read Abstract, Figure 6, and §4 through Algorithm 1.
4. State the two changes without looking: `MLP → 3×3 Conv`; target patch tokens receive spatial normalization.
5. Open `FLARE` §3.1 and `Spatial Forcing` §2.2; identify student tokens, teacher features, alignment time, and inference-time removals.
6. Read the first two sections of [PRIOR_ART_MAP.md](PRIOR_ART_MAP.md).

At the end, you should be able to say:

> `REPA` is not a new generative objective. It adds training-time representation supervision to hidden states of a denoising transformer. `iREPA` argues that local patch structure, rather than global linear-probe quality, explains much of the benefit.

## 75-minute route

### Phase 1 — REPA mechanism (25 minutes)

- Read §2 only far enough to distinguish diffusion and flow/velocity objectives.
- Deep-read §3.2–§3.3.
- Inspect Tables 2–4 and Figure 6; record which comparisons are matched.
- Read the limitations and compute discussion before repeating the `17.5×` claim.

### Phase 2 — iREPA mechanism (20 minutes)

- Read §2–§3 for the `global information vs spatial structure` diagnosis.
- Deep-read §4, especially Figure 6, Algorithm 1, and Table 2.
- Inspect Appendix implementation details for `γ`, kernel size, and padding.

### Phase 3 — VLA transfer audit (30 minutes)

- `FLARE` §3.1: future tokens inside the action `DiT`.
- `Spatial Forcing` method: current-frame visual tokens aligned to `VGGT`.
- `VEGA` §3: visual-encoder output aligned to `DINOv2-FiT3D`.
- `AGRA` method: intermediate video `DiT` features aligned to `DINOv2` for an action decoder.
- Use the comparison matrix in [PRIOR_ART_MAP.md](PRIOR_ART_MAP.md) to state what remains different from exact `iREPA`.

## 180-minute deep route

1. Complete the 75-minute route.
2. Read `REPA` ablations on target encoder, alignment depth, objective, and model scale.
3. Read `iREPA`'s 27-encoder analysis and both component ablations.
4. Read `FLARE`, `Spatial Forcing`, `VEGA`, and `AGRA` end to end.
5. Scan `FRAPPE`, `FutureVLA`, `SAM3D-Guided`, `Robust-WAM`, and `Mind-VLA` for target choice and supervision location.
6. Draft the matched experiment at the end of [PRIOR_ART_MAP.md](PRIOR_ART_MAP.md) before choosing a paper title or claiming novelty.

## REPA: the minimal mental model

### Problem

A diffusion/flow transformer eventually learns useful visual structure, but the paper argues that this representation develops slowly and remains weaker than a strong frozen visual encoder such as `DINOv2`.

### Mechanism

For a clean image `x*`, a frozen encoder produces patch targets:

```text
y* = f(x*)
```

The trainable generative model receives a noisy latent at time `t`. At an intermediate layer, a trainable projector maps hidden patch states to the teacher dimension:

```text
ŷ_t = h_phi(h_t)
```

The training objective becomes:

```text
L_total = L_velocity_or_diffusion + λ L_REPA
L_REPA  = - mean_patch sim(y*, ŷ_t)
```

The frozen encoder and projection head are only needed during training. This means **no extra inference module**, not zero training cost: teacher feature extraction, activation storage, projector compute, and data I/O still matter.

### What the authors claim

- A `SiT-XL/2` with `REPA` matches the unguided quality of a 7M-step baseline in fewer than 400K steps, reported as more than `17.5×` faster convergence in training steps.
- With guidance settings reported in the paper, the final `FID` reaches `1.42`.

### What those claims do not establish

- `17.5×` is not a measured `17.5×` wall-clock speedup.
- It is not robot control speed, policy latency, or control frequency.
- `FID` and `IS` do not measure closed-loop task success.
- A clean-image teacher target does not prove the same target is optimal for noisy actions, future observations, or action-conditioned tokens.

## iREPA: what changed and why it matters for VLA

`iREPA` tests 27 visual encoders and reports that target usefulness correlates much more strongly with **spatial self-similarity among patch tokens** than with global `ImageNet-1K` linear-probe accuracy.

It then makes two small changes:

### 1. Spatially local projector

```text
standard REPA: patch-wise MLP projector
iREPA:        3×3 Conv projector, padding 1
```

The intended inductive bias is to preserve and transfer local spatial relationships instead of independently transforming every patch token.

### 2. Spatial normalization of teacher patch tokens

For target tokens `x` over patch dimension `p`:

```text
y = (x - γ E_p[x]) / (Std_p[x] + ε)
```

The paper uses `γ` in approximately `[0.6, 0.8]`. The goal is to reduce the global component shared by many patches and increase local contrast.

### Transfer hypothesis for VLA

The strongest implication is not simply “use a better teacher.” It is:

> Test whether the spatial relation structure of the teacher survives the projector and reaches the exact `VLA` tokens that determine actions.

That leads directly to questions about patch grids, multiple cameras, object masks, 3D correspondence, future frames, action-denoising time, and the layer where alignment is applied.

## A clean taxonomy for the VLA literature

| Family | Student representation | Teacher/target | Why it is not identical to exact iREPA |
|---|---|---|---|
| `FLARE` | future tokens inside action `DiT` | future observation embedding | future latent target; standard `MLP`-style alignment |
| `Spatial Forcing` | intermediate VLA visual tokens | frozen `VGGT` features | 3D teacher, but no verified `iREPA` Conv + target normalization pair |
| `VEGA` | VLA visual-encoder patch output | `DINOv2-FiT3D` | explicit two-layer `MLP` projector |
| `AGRA` | intermediate video `DiT` features in `WAM` | frozen `DINOv2` | aligns the world/action interface, not a standard VLA visual backbone |
| `Robust-WAM` | future query outputs in action stream | future-frame `DINOv3 CLS` | global `CLS` target rather than spatial patch structure |
| `SAM3D-Guided` / `Mind-VLA` | object/task-conditioned VLA features | dense 3D/object features | closer to object grounding; still not a tested exact iREPA recipe |

See [PRIOR_ART_MAP.md](PRIOR_ART_MAP.md) for the full evidence table and dates.

## Self-reading questions — deliberately unanswered

### REPA

1. Why is the teacher computed from the clean image while the student sees a noisy latent?
2. At which denoising times and transformer layers should alignment be easiest or most useful?
3. Does maximizing patch-wise cosine similarity force identical representations, or only compatible directions?
4. Which ablation supports the choice of alignment depth most directly?
5. Does the paper report extra teacher compute and peak-memory cost separately from training steps?
6. Which result justifies calling the gain convergence acceleration rather than better final quality?

### iREPA

7. How is spatial self-similarity defined, and why is it a different signal from linear-probe accuracy?
8. Why can a patch-wise `MLP` destroy useful spatial information even when token positions are unchanged?
9. What information is deliberately weakened by spatial normalization?
10. Do the component ablations show that `Conv projector` and spatial normalization are independently useful?
11. Which `γ` values are robust, and would they remain robust across multiple robot cameras?
12. Does stronger local structure ever reduce global semantic or instruction information?

### VLA transfer

13. Should the student be current visual tokens, future tokens, action-denoiser hidden states, or video-model hidden states?
14. Should the teacher see the current observation, a future observation, or an instruction-selected target object?
15. For a flow-matching action expert, should alignment weight vary with action noise/time `t`?
16. How would a `3×3 Conv` operate when tokens come from multiple cameras or are no longer arranged on one image grid?
17. Would `DINOv3 CLS`, patch tokens, `VGGT`, `DINOv2-FiT3D`, or `SAM3D` provide the most action-relevant target?
18. Which metric would demonstrate faster training: steps-to-threshold, wall-clock-to-threshold, or area under the learning curve?
19. How will you separate better visual representation from additional teacher compute and extra data?
20. If the projector and teacher are removed at inference, can training still change inference latency or memory through a changed backbone/checkpoint?

## Reading discipline

Keep three columns in your notes:

| Authors claim | Direct evidence | Your synthesis |
|---|---|---|
| Quote or faithful paraphrase | exact Figure/Table/Section and matched setup | transfer judgment, alternative explanation, or unresolved question |

Do not compare success rates across unmatched `VLA` backbones, training data, robot platforms, control rates, action horizons, seeds, episode counts, or termination rules. For a deployment-oriented result, separately report training compute, inference latency, control frequency, peak memory, and power/energy.

## Primary records

- [REPA — ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d9e42b4d7163931f3689d6d6fbaa11d0-Abstract-Conference.html) · [official code](https://github.com/sihyun-yu/REPA)
- [iREPA — ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3929a7785bd56f57edcff0152ab41289-Abstract-Conference.html) · [official code](https://github.com/end2end-diffusion/irepa)
- [FLARE — CoRL 2025 / PMLR](https://proceedings.mlr.press/v305/zheng25a.html)

## AI-assisted preparation disclosure

This reading companion was prepared with an AI-assisted literature workflow. Paper identity, local PDF structure, and the cited method passages were checked against primary full texts. The negative novelty statement is search-bounded as of 2026-08-31; it is not proof that no unpublished, unindexed, or later work exists.
