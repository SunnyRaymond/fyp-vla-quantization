# Local PDF inventory

> **Acquisition date:** through 2026-09-05  
> **Scope:** 24 canonical core paper full texts + 1 core-paper supplemental artifact + 1 archived core version + 28 companion papers + 2 official companion specifications  
> **Total:** 56 PDFs, 532,161,627 bytes (507.51 MiB)

## Verification rule

Each file was accepted only after:

1. the download returned successfully;
2. the first five bytes matched `%PDF-`;
3. `pypdf` reopened the file, counted its pages, and extracted first-page text matching the expected title;
4. the file was confirmed as unencrypted;
5. SHA-256 was recorded for the earlier acquisitions. The two additions on 2026-09-05 use structural / readability / identity checks only; their hash cells explicitly say not computed.

This is an acquisition/structure check, not a declaration that the user has read the paper. No `human-read` status was inferred.

## Core papers

| # | Reading | Local file | Local source version | Pages | Bytes | SHA-256 |
|---:|---|---|---|---:|---:|---|
| 1 | OpenVLA | [paper.pdf](papers/01-openvla/paper.pdf) | PMLR / CoRL proceedings final | 35 | 13,036,087 | `312adce96455c80da723f85bfe25c9bae7ccfbeb9daf739e05671eb7a303affa` |
| 2 | π₀ | [paper.pdf](papers/02-pi0/paper.pdf) | Physical Intelligence official PDF | 17 | 7,873,003 | `75667e62ba43f153805f2330b55966d0ff51abd03aa0ab76ef2d62ab201cc1fd` |
| 2A | FAST | [paper.pdf](papers/02a-fast/paper.pdf) | Physical Intelligence official PDF; RSS 2025 paper | 19 | 7,269,156 | `f3578de223076c1b494786db7c36b48e605e1ab11def309b50b299ec0a8f38bb` |
| 3 | π₀.₅ | [paper.pdf](papers/03-pi05/paper.pdf) | Physical Intelligence official PDF | 19 | 5,079,459 | `f5d0aa0b652f188e6649a83a54d704e14393299676754c01525e3964767db7c8` |
| 4 | π*₀.₆ | [paper.pdf](papers/04-pistar06/paper.pdf) | Physical Intelligence official PDF | 18 | 6,043,825 | `6b37590a1f17e415df93b456a65ad399788126c5a610e85122e69e471a509223` |
| 5 | ω-0 | [paper.pdf](papers/05-omega0/paper.pdf) | arXiv current version at acquisition | 39 | 28,999,996 | `c56d4115741a7979bb63ed4ff8e76e25c8e6ca933d6b18f2d122d8a46ec4d02c` |
| 6 | BitVLA | [paper.pdf](papers/06-bitvla/paper.pdf) | arXiv current version at acquisition | 15 | 2,751,393 | `e9bdae3f08eab2faf909656929d1130cde99d09d5ff7daf59d33ab058bcd1ca0` |
| 7 | FP8 Formats for Deep Learning | [paper.pdf](papers/07-fp8-formats/paper.pdf) | arXiv technical preprint | 9 | 281,195 | `809a9557e907765b452c5a1b7308a92e31dd31a07e668d49d14b1594b6c0cf0c` |
| 8 | Microscaling Data Formats for Deep Learning | [paper.pdf](papers/08-microscaling-formats/paper.pdf) | arXiv technical preprint | 9 | 406,488 | `3c22864ff532e69382d6d4d13e73bca3e4555c07b5a38059456ef18ef570bf2e` |
| 9 | Pretraining Large Language Models with NVFP4 | [paper.pdf](papers/09-nvfp4-pretraining/paper.pdf) | arXiv vendor-authored technical report | 22 | 2,823,374 | `8f0a2eeab9cdd71c36af759922f7e41a43c00cc6fc3cd8525e23b2c8d0d82565` |
| 10 | SmoothQuant | [paper.pdf](papers/10-smoothquant/paper.pdf) | PMLR / ICML venue final | 13 | 5,203,060 | `551636e4e0bd9dda9b01186473bc46c72c56e0c190a28fe6b7ed4fa80055372e` |
| 11 | AWQ | [paper.pdf](papers/11-awq/paper.pdf) | MLSys venue final | 14 | 26,255,277 | `cd7b88325267627b7159dfc32d3ee3fc5430718bd722afee6da25e14ff7525d6` |
| 12 | SpinQuant | [paper.pdf](papers/12-spinquant/paper.pdf) | ICLR 2025 venue final | 24 | 10,308,097 | `5f20713758d51ca000ac41564e2c61f64cb75e628c3760cb53936162328ddc89` |
| 13 | HAQ | [paper.pdf](papers/13-haq/paper.pdf) | CVF / CVPR venue final | 9 | 1,404,382 | `a5f1649873ff32968f37a0a9a95f2272c786d0b99f2028605422fb9027a415ab` |
| 14 | TurboQuant | [paper-iclr-2026-final.pdf](papers/14-turboquant/paper-iclr-2026-final.pdf) | ICLR 2026 official proceedings final | 22 | 687,259 | `546a55df7db7d218de61538a20443fdd3f5c280599eec28d6a52ecd0a888884b` |
| 15 | LoRA | [paper-arxiv-v2.pdf](papers/15-lora/paper-arxiv-v2.pdf) | arXiv:2106.09685 v2; ICLR 2022 work | 26 | 1,609,513 | `e9a0d3128767db616085dc0f4e6e455e672e89af823e8ed1282793682787395a` |
| 16 | Outlier Suppression | [paper.pdf](papers/16-outlier-suppression/paper.pdf) | NeurIPS 2022 official proceedings final | 13 | 1,489,121 | `320b3958f960cb93c5778225197fcecbe39ff69f5c3bbc6bcb7d5236bc83aef2` |
| 17 | Flow Matching for Generative Modeling | [paper-arxiv-v2.pdf](papers/17-flow-matching/paper-arxiv-v2.pdf) | arXiv:2210.02747 v2 author manuscript; ICLR 2023 work | 28 | 25,147,129 | `5eeb39ba516396924aba4787452f9d0abdee88467a4d0c264d9f66cad0c5ee14` |
| 18 | Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis | [paper-arxiv-v1.pdf](papers/18-self-flow/paper-arxiv-v1.pdf) | arXiv:2603.06507 v1 author manuscript; ICML 2026 accepted poster | 37 | 22,963,520 | `a9a955fd626b8850169d0edf1c0ac53d4bcafe01ab915959ff596a5687e387e1` |
| 20 | HBVLA | [paper-arxiv-v2.pdf](papers/20-hbvla/paper-arxiv-v2.pdf) | arXiv:2602.13710 v2 preprint | 9 | 3,881,276 | `2095e124c2146c611c474855dc826ca2fa02b331b9d736375ff706cbbd289d05` |
| 21 | WorldCache: Content-Aware Caching for Accelerated Video World Models | [paper-arxiv-v1.pdf](papers/21-worldcache-content-aware/paper-arxiv-v1.pdf) | arXiv:2603.22286 v1 author manuscript; ECCV 2026 accepted identity verified separately | 33 | 18,828,805 | `1c911e464cf80ea786330a0a4b41a82ea441c2c694668c54de94d901385ff7b0` |
| 22 | WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching | [paper-arxiv-v2.pdf](papers/22-worldcache-heterogeneous-token-caching/paper-arxiv-v2.pdf) | arXiv:2603.06331 v2 proceedings-style manuscript; ICML 2026 / PMLR 306 | 26 | 13,203,052 | `436c67f3301ad02cced74cd40aaa2d8f28fab27b57616def71e7b64dbb9b341a` |
| 23 | WaterSIC | [paper-arxiv-v2.pdf](papers/23-watersic/paper-arxiv-v2.pdf) | arXiv:2603.04956v2; ICML 2026 author manuscript | 32 | 2,478,721 | Not computed (structural checks only) |
| 24 | GRACE | [paper-arxiv-v5.pdf](papers/24-grace/paper-arxiv-v5.pdf) | arXiv:2601.22709v5; ICML 2026 author manuscript | 35 | 15,380,942 | Not computed (structural checks only) |

## Core-paper supporting artifact

This official supplemental completes the proof, quantization-node, ablation, and implementation-detail package for core `#16`. It is not counted as a separate paper identity.

| Reading | Local file | Local source version | Pages | Bytes | SHA-256 |
|---|---|---|---:|---:|---|
| Outlier Suppression supplemental | [supplemental.pdf](papers/16-outlier-suppression/supplemental.pdf) | NeurIPS 2022 official supplemental | 10 | 6,968,759 | `8c4d62b9f6a5cf6351c4aabad7c14623a2fb8ec39524656e16af31b4af854163` |

## Archived core-paper version

This file is retained only for a version-drift and controversy audit. It is not the canonical TurboQuant publication.

| Reading | Local file | Local source version | Pages | Bytes | SHA-256 |
|---|---|---|---:|---:|---|
| TurboQuant | [paper-arxiv-v1.pdf](papers/14-turboquant/paper-arxiv-v1.pdf) | arXiv v1, submitted 2025-04-28 | 25 | 861,881 | `431eb13926e10491f5fbd0bebd0813c51bd6c1e884426a1500c5db640b2997ab` |

## VLA systems companion paper

This reading follows `π₀.₅` into streaming flow-matching action decoding and asynchronous closed-loop deployment. It does not change the 22-paper active core count.

| Reading | Local file | Local source version | Pages | Bytes | SHA-256 |
|---|---|---|---:|---:|---|
| FlashVLA: Streaming Action Decoding for Fast and Asynchronous VLA Inference | [paper-arxiv-v1.pdf](papers/03a-flashvla/paper-arxiv-v1.pdf) | arXiv:2608.27384 v1, submitted 2026-08-27 | 17 | 4,875,032 | `74d35658a10c7a57fdf06a59e4bc021bb8de1bb4ea9b882f38f45e94b3abfa66` |

## Representation-alignment companion papers

These twelve papers define `REPA` / `iREPA`, trace direct and partial transfers into `VLA` / `WAM`, and retain one reconstruction-based adjacent baseline. They do not change the 22-paper active core count.

| Reading | Local file | Local source version | Pages | Bytes | SHA-256 |
|---|---|---|---:|---:|---|
| REPA | [repa-iclr-2025-final.pdf](papers/19-repa-irepa-vla-alignment/repa-iclr-2025-final.pdf) | ICLR 2025 official proceedings final | 43 | 34,693,389 | `d97efd44d9354162bc4ad9b0d5288952f09a90e99c5da344b2098204b7a8164a` |
| iREPA | [irepa-iclr-2026-final.pdf](papers/19-repa-irepa-vla-alignment/irepa-iclr-2026-final.pdf) | ICLR 2026 official proceedings final | 39 | 38,920,911 | `962345b71fae7157c39708fcf386d3fffca8ab6ebac6b1686aeb6db622adb967` |
| FLARE | [prior-art-flare-corl-2025-final.pdf](papers/19-repa-irepa-vla-alignment/prior-art-flare-corl-2025-final.pdf) | PMLR / CoRL 2025 final | 20 | 10,467,300 | `fc769418d2e1de506340ec3ac7d289c74a2fb6462a815bcc894d403a55851c6e` |
| Spatial Forcing | [prior-art-spatial-forcing-arxiv-v2.pdf](papers/19-repa-irepa-vla-alignment/prior-art-spatial-forcing-arxiv-v2.pdf) | arXiv:2510.12276 v2; ICLR 2026 work | 19 | 1,990,671 | `c7d15ca562f5639ab88c60e7de0de4c34965e3c57710e79febfa249b8bddada5` |
| FRAPPE | [prior-art-frappe-arxiv-v1.pdf](papers/19-repa-irepa-vla-alignment/prior-art-frappe-arxiv-v1.pdf) | arXiv:2602.17259 v1 | 16 | 27,324,107 | `12eac2e5cd9812e23f337f94ddd8081f8805532d9904658d366952cabc46faa1` |
| FutureVLA | [prior-art-futurevla-arxiv-v1.pdf](papers/19-repa-irepa-vla-alignment/prior-art-futurevla-arxiv-v1.pdf) | arXiv:2603.10712 v1 | 30 | 15,622,212 | `46b3cf66ea3f63be29e7afb02e0175682f9ccc07c5622f6e19da8f5c50da1017` |
| VEGA | [prior-art-vega-arxiv-v1.pdf](papers/19-repa-irepa-vla-alignment/prior-art-vega-arxiv-v1.pdf) | arXiv:2605.10485 v1 | 18 | 7,839,224 | `635569d092e7e7ccdfca9b25cd70411dadee6b8edd812ca52e9c62fe674d0e6b` |
| AGRA | [prior-art-agra-arxiv-v1.pdf](papers/19-repa-irepa-vla-alignment/prior-art-agra-arxiv-v1.pdf) | arXiv:2606.12217 v1 | 23 | 18,149,982 | `8fe740c3e5496a1174636e496410b954e5b63928181f5906f40c959123c72389` |
| SAM3D-Guided | [prior-art-sam3d-vla-arxiv-v1.pdf](papers/19-repa-irepa-vla-alignment/prior-art-sam3d-vla-arxiv-v1.pdf) | arXiv:2607.25912 v1 | 11 | 5,554,248 | `f7a131d01582af371b1dc98ce8c5980ba7870136f3f59f91b6dcafad9b169e28` |
| Robust-WAM | [prior-art-robust-wam-arxiv-v1.pdf](papers/19-repa-irepa-vla-alignment/prior-art-robust-wam-arxiv-v1.pdf) | arXiv:2608.05903 v1 | 13 | 2,960,753 | `6d5ad91cc2d4785550e525bc470fb0bd7659553d77fe965864d6734bbe6576f4` |
| Mind-VLA | [prior-art-mind-vla-arxiv-v2.pdf](papers/19-repa-irepa-vla-alignment/prior-art-mind-vla-arxiv-v2.pdf) | arXiv:2608.04633 v2 | 9 | 1,665,341 | `12a01a0f65c7331505e8c8a4c41eb356b56cb55173db9479c8dfc3ce4944bddb` |
| ReconVLA | [adjacent-reconvla-arxiv-v1.pdf](papers/19-repa-irepa-vla-alignment/adjacent-reconvla-arxiv-v1.pdf) | arXiv:2508.10333 v1; AAAI 2026 venue identity confirmed separately | 10 | 33,378,540 | `de568f5eb3e26a355160e7b79c286327eae38f3585a419a51a061c186e55f011` |

## Self-Flow mechanism companion paper

This later preprint directly tests the causal interpretation of `Dual-Timestep Scheduling`. It is a critical companion and does not change the 22-paper active core count.

| Reading | Local file | Local source version | Pages | Bytes | SHA-256 |
|---|---|---|---:|---:|---|
| From SRA to Self-Flow: Data Augmentation or Self-Supervision? | [critique-from-sra-to-self-flow-arxiv-v1.pdf](papers/18-self-flow/critique-from-sra-to-self-flow-arxiv-v1.pdf) | arXiv:2607.02508 v1, submitted 2026-07-02 | 10 | 6,791,818 | `62090784744a0ebc5f7d21ee5ffb217c8143af64a5e2b87fe6be42c9d03f18b8` |

## Robot RL systems companion paper

This reading supplies an on-policy/off-policy, replay, critic-stability, wall-clock, and sim-to-real systems baseline. It is not a VLA paper and does not change the 22-paper active core count.

| Reading | Local file | Local source version | Pages | Bytes | SHA-256 |
|---|---|---|---:|---:|---|
| FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control | [paper-arxiv-v2.pdf](papers/18a-flashsac/paper-arxiv-v2.pdf) | arXiv:2604.04539 v2, revised 2026-05-15; RSS 2026 `Outstanding Paper Award` work | 42 | 15,404,174 | `e8a1112811bc251adab68e3d217dfeb4cefd39b6236739f644cd4673b78f1120` |

## PI-series companion paper

This reading extends base `π₀.₆` with multi-scale embodied memory. It does not change the 22-paper active core count and is separate from `π*₀.₆`.

| Reading | Local file | Local source version | Pages | Bytes | SHA-256 |
|---|---|---|---:|---:|---|
| MEM | [paper.pdf](papers/04a-mem/paper.pdf) | Physical Intelligence official project PDF; canonical metadata is arXiv v2 | 15 | 9,037,715 | `63413aa32bb2b070ae0a0cbc5d5a2eb77686308725e48f15bcaa338bd3d644aa` |

## BitVLA companion papers

These four readings trace BitVLA's direct method dependencies and prior art. They do not change the 22-paper active core count.

| Reading | Local file | Local source version | Pages | Bytes | SHA-256 |
|---|---|---|---:|---:|---|
| Apprentice | [paper.pdf](papers/06a-apprentice-quantization-distillation/paper.pdf) | arXiv v1; ICLR 2018 work | 15 | 439,141 | `eecfe58207350b35644267cc46635711d8d18732910d9657d7c0b863b14539a1` |
| LLaVA / Visual Instruction Tuning | [paper.pdf](papers/06b-llava/paper.pdf) | NeurIPS 2023 venue final | 25 | 5,894,130 | `d3cf39a2675e79c5a216a6faf74601dcf30cb100ba20f24f397ce62ca5d614e3` |
| OpenVLA-OFT | [paper.pdf](papers/06c-openvla-oft/paper.pdf) | arXiv v2; accepted RSS 2025 | 24 | 26,994,977 | `b860aa1206b6cfb0ce8be177f961379dd6a133d52cc74ac346636e0f4952a596` |
| LLM.int8 / bitsandbytes | [paper.pdf](papers/06d-llm-int8-bitsandbytes/paper.pdf) | NeurIPS 2022 venue final | 15 | 601,510 | `a7de7700cb8d0a36eb1c7926add3ec10530f53be8d81c681e7de372586d58f77` |

## Quantization companion papers

These eight readings add one canonical weight-only PTQ method, one direct rotation-based predecessor to SpinQuant, two current field maps, the original and extended RaBitQ papers, and two critical TurboQuant prior-art/reproducibility companions. They do not change the 22-paper active core count.

| Reading | Local file | Local source version | Pages | Bytes | SHA-256 |
|---|---|---|---:|---:|---|
| OPTQ (commonly known as GPTQ) | [paper.pdf](papers/10a-gptq-optq/paper.pdf) | ICLR 2023 / ISTA Published Version | 16 | 437,492 | `e8956fc46acc6292cca14ad324298b8ddf0c21727a9c410bceae00ea84623287` |
| QuaRot | [paper.pdf](papers/12a-quarot/paper.pdf) | NeurIPS 2024 venue final | 28 | 757,470 | `18383749475f04c71bb5d563148c791c1f2a5db60efb28a888538ff97b44b2ef` |
| A Survey of Quantization in LLM | [paper.pdf](papers/10b-llm-quantization-survey-2026/paper.pdf) | JCST 2026 venue final; 1 cover + 18 article pages | 19 | 5,340,680 | `1e535249b37886dc337ae0ab7bf040b372b19673dfb56d585be1265190f8f329` |
| A Survey of Low-bit Large Language Models | [paper-arxiv-v3.pdf](papers/10c-low-bit-llm-survey/paper-arxiv-v3.pdf) | arXiv v3 author manuscript; Neural Networks 2025 venue record | 28 | 18,957,554 | `1b434d6270cf73d3a9b8e972dc55d10c3fc6a068b5516252501f29cf34372dc0` |
| RaBitQ | [paper-original-arxiv-v1.pdf](papers/14b-rabitq/paper-original-arxiv-v1.pdf) | arXiv v1; SIGMOD 2024 work | 22 | 2,093,673 | `0eb4a15dcad7303f57347d16b4b48f8ff96eb91f30b57193458ba536dd377432` |
| Practical and Asymptotically Optimal Quantization... | [companion-multibit-arxiv-v1.pdf](papers/14b-rabitq/companion-multibit-arxiv-v1.pdf) | arXiv v1 author full text; SIGMOD 2025 work | 16 | 1,560,220 | `451eae96a106a4f879ddfc1d912efbad6bfb885b99c6f38e47cd3a197de124fc` |
| Revisiting RaBitQ and TurboQuant | [paper-arxiv-v2.pdf](papers/14a-rabitq-turboquant-comparison/paper-arxiv-v2.pdf) | arXiv v2; accepted poster paper at VecDB@VLDB 2026 | 15 | 1,037,467 | `a6562d43e817e4a0bd356994877a17daccbf713d1abebdeb2cf09ccd224330ef` |
| A Note on TurboQuant and the Earlier DRIVE/EDEN Line of Work | [companion-drive-eden-note-arxiv-v1.pdf](papers/14a-rabitq-turboquant-comparison/companion-drive-eden-note-arxiv-v1.pdf) | arXiv v1 technical note | 10 | 760,492 | `f194a917d24ba4eb1694a72ac27d65dc98b503dbb80d17003661145eb7b43921` |

## Companion specifications

| Reading folder | Local file | Status | Pages | Bytes | SHA-256 |
|---|---|---|---:|---:|---|
| FP8 Formats | [OFP8 specification](papers/07-fp8-formats/specification.pdf) | OCP OFP8 Revision 1.0, official normative material | 16 | 564,311 | `1e1ebad11388cdc1cdb4afa7e226b78f18d4049c6f39c36ecacd747e9ca3c08b` |
| Microscaling Formats | [MX specification](papers/08-microscaling-formats/specification.pdf) | OCP MX Specification v1.0, official normative material | 16 | 812,323 | `d195d6a36dd4a0c89064af0c479bcaad5c0fe29d63f628502ea6d7c4b4279421` |

## Source URLs

| Item | Acquisition source |
|---|---|
| OpenVLA | [PMLR official asset](https://raw.githubusercontent.com/mlresearch/v270/main/assets/kim25c/kim25c.pdf) |
| π₀ | [Physical Intelligence](https://www.pi.website/download/pi0.pdf) |
| FAST | [Physical Intelligence](https://www.pi.website/download/fast.pdf) · [RSS proceedings record](https://www.roboticsproceedings.org/rss21/p012.html) |
| π₀.₅ | [Physical Intelligence](https://www.pi.website/download/pi05.pdf) |
| FlashVLA: Streaming Action Decoding... | [arXiv v1](https://arxiv.org/pdf/2608.27384v1) · [record](https://arxiv.org/abs/2608.27384) · [code](https://github.com/z-lab/flashvla) |
| REPA | [ICLR 2025 record](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d9e42b4d7163931f3689d6d6fbaa11d0-Abstract-Conference.html) · [official PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/d9e42b4d7163931f3689d6d6fbaa11d0-Paper-Conference.pdf) · [code](https://github.com/sihyun-yu/REPA) |
| iREPA | [ICLR 2026 record](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3929a7785bd56f57edcff0152ab41289-Abstract-Conference.html) · [official PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/3929a7785bd56f57edcff0152ab41289-Paper-Conference.pdf) · [code](https://github.com/end2end-diffusion/irepa) |
| FLARE | [PMLR / CoRL 2025](https://proceedings.mlr.press/v305/zheng25a.html) |
| Spatial Forcing | [arXiv v2](https://arxiv.org/abs/2510.12276v2) · [project](https://spatial-forcing.github.io/) |
| FRAPPE | [arXiv v1](https://arxiv.org/abs/2602.17259v1) · [code](https://github.com/Jbo-Wang/frappe) |
| FutureVLA | [arXiv v1](https://arxiv.org/abs/2603.10712v1) |
| VEGA | [arXiv v1](https://arxiv.org/abs/2605.10485v1) |
| AGRA | [arXiv v1](https://arxiv.org/abs/2606.12217v1) · [project](https://xpeng-robotics.github.io/agra) |
| SAM3D-Guided | [arXiv v1](https://arxiv.org/abs/2607.25912v1) |
| Robust-WAM | [arXiv v1](https://arxiv.org/abs/2608.05903v1) |
| Mind-VLA | [arXiv v2](https://arxiv.org/abs/2608.04633v2) |
| ReconVLA | [arXiv v1](https://arxiv.org/abs/2508.10333v1) · [AAAI 2026 record](https://doi.org/10.1609/aaai.v40i22.38921) |
| FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control | [arXiv v2](https://arxiv.org/pdf/2604.04539v2) · [arXiv record](https://arxiv.org/abs/2604.04539v2) · [RSS paper](https://roboticsconference.org/program/papers/99/) · [RSS award](https://roboticsconference.org/program/awards/) · [code](https://github.com/Holiday-Robot/FlashSAC) |
| π*₀.₆ | [Physical Intelligence](https://www.pi.website/download/pistar06.pdf) |
| MEM | [Physical Intelligence official PDF](https://www.pi.website/download/Mem.pdf) · [project page](https://www.pi.website/research/memory) · [arXiv v2](https://arxiv.org/abs/2603.03596) |
| ω-0 | [arXiv](https://arxiv.org/pdf/2608.06375) |
| BitVLA | [arXiv](https://arxiv.org/pdf/2506.07530) |
| FP8 Formats | [arXiv](https://arxiv.org/pdf/2209.05433) |
| OCP OFP8 | [Open Compute Project](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf) |
| Microscaling Formats | [arXiv](https://arxiv.org/pdf/2310.10537) |
| OCP MX | [Open Compute Project](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) |
| NVFP4 pretraining | [arXiv](https://arxiv.org/pdf/2509.25149) |
| SmoothQuant | [PMLR](https://proceedings.mlr.press/v202/xiao23c/xiao23c.pdf) |
| AWQ | [MLSys proceedings](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf) |
| SpinQuant | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/file/e5b1c0d4866f72393c522c8a00eed4eb-Paper-Conference.pdf) |
| HAQ | [CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.pdf) |
| TurboQuant | [ICLR proceedings final](https://proceedings.iclr.cc/paper_files/paper/2026/file/5c802ef38ab6e366c2ea06eee554c088-Paper-Conference.pdf) · [archived arXiv v1](https://arxiv.org/pdf/2504.19874) |
| LoRA | [arXiv v2](https://arxiv.org/pdf/2106.09685v2) · [OpenReview record](https://openreview.net/forum?id=nZeVKeeFYf9) · [Microsoft Research](https://www.microsoft.com/en-us/research/publication/lora-low-rank-adaptation-of-large-language-models/) · [code](https://github.com/microsoft/LoRA) |
| Outlier Suppression | [NeurIPS record](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6f6db140de9c9f111b12ef8a216320a9-Abstract-Conference.html) · [official paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/6f6db140de9c9f111b12ef8a216320a9-Paper-Conference.pdf) · [official supplemental](https://proceedings.neurips.cc/paper_files/paper/2022/file/6f6db140de9c9f111b12ef8a216320a9-Supplemental-Conference.pdf) · [arXiv](https://arxiv.org/abs/2209.13325) · [code](https://github.com/wimh966/outlier_suppression) |
| Flow Matching for Generative Modeling | [arXiv v2](https://arxiv.org/pdf/2210.02747v2) · [arXiv record](https://arxiv.org/abs/2210.02747) · [ICLR record](https://iclr.cc/virtual/2023/poster/11309) · [OpenReview record](https://openreview.net/forum?id=PqvMRDCJT9t) |
| Self-Flow | [arXiv v1](https://arxiv.org/pdf/2603.06507v1) · [arXiv record](https://arxiv.org/abs/2603.06507v1) · [ICML 2026 poster](https://icml.cc/virtual/2026/poster/65011) · [OpenReview record](https://openreview.net/forum?id=HoThWhfxiK) · [project](https://black-forest-labs.github.io/Self-Flow/) · [code](https://github.com/black-forest-labs/Self-Flow) |
| HBVLA | [arXiv v2 PDF](https://arxiv.org/pdf/2602.13710v2) · [arXiv record](https://arxiv.org/abs/2602.13710) |
| WorldCache: Content-Aware Caching for Accelerated Video World Models | [arXiv v1 PDF](https://arxiv.org/pdf/2603.22286v1) · [arXiv record](https://arxiv.org/abs/2603.22286) · [ECCV 2026 list](https://eccv.ecva.net/Conferences/2026/AcceptedPapers) · [project](https://umair1221.github.io/World-Cache/) · [code](https://github.com/umair1221/WorldCache) |
| WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching | [arXiv v2 PDF](https://arxiv.org/pdf/2603.06331v2) · [arXiv record](https://arxiv.org/abs/2603.06331) · [ICML 2026 list](https://icml.cc/Downloads/2026) · [code](https://github.com/FofGofx/WorldCache) |
| From SRA to Self-Flow | [arXiv v1](https://arxiv.org/pdf/2607.02508v1) · [arXiv record](https://arxiv.org/abs/2607.02508v1) · [code](https://github.com/vvvvvjdy/SRA/tree/main/SiT-SRA_DTS_AS) |
| RaBitQ | [arXiv v1](https://arxiv.org/pdf/2405.12497v1) · [archived original code](https://github.com/gaoj0017/RaBitQ) · [current library](https://github.com/VectorDB-NTU/RaBitQ-Library) |
| Practical and Asymptotically Optimal Quantization... | [arXiv v1](https://arxiv.org/pdf/2409.09913v1) · [archived extension code](https://github.com/VectorDB-NTU/Extended-RaBitQ) · [current library](https://github.com/VectorDB-NTU/RaBitQ-Library) |
| Revisiting RaBitQ and TurboQuant | [arXiv v2](https://arxiv.org/pdf/2604.19528v2) · [code](https://github.com/VectorDB-NTU/rabitq-turboquant-comparison) · [open letter](https://dev.to/gaoj0017/turboquant-and-rabitq-what-the-public-story-gets-wrong-1i00) · [VecDB@VLDB 2026 programme](https://vecdb-ws.github.io/vldb2026/index.html) |
| TurboQuant and DRIVE/EDEN note | [arXiv v1](https://arxiv.org/pdf/2604.18555v1) |
| Apprentice | [arXiv v1](https://arxiv.org/pdf/1711.05852v1) · [OpenReview record](https://openreview.net/forum?id=B1ae1lZRb) |
| LLaVA / Visual Instruction Tuning | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/file/6dcf277ea32ce3288914faf369fe6de0-Paper-Conference.pdf) |
| OpenVLA-OFT | [arXiv v2](https://arxiv.org/pdf/2502.19645v2) · [RSS proceedings record](https://www.roboticsproceedings.org/rss21/p017.html) |
| LLM.int8 / bitsandbytes | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/file/c3ba4962c05c49636d4c6206a97e9c8a-Paper-Conference.pdf) |
| OPTQ / GPTQ | [ISTA Published Version](https://research-explorer.ista.ac.at/download/17378/17385/2023_ICLR_Frantar.pdf) · [OpenReview record](https://openreview.net/forum?id=tcbBPnfwxS) · [arXiv](https://arxiv.org/abs/2210.17323) |
| QuaRot | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/file/b5b939436789f76f08b9d0da5e81af7c-Paper-Conference.pdf) · [arXiv v2](https://arxiv.org/abs/2404.00456) |
| A Survey of Quantization in LLM | [JCST official PDF](https://jcst.ict.ac.cn/cn/article/pdf/preview/10.1007/s11390-026-5979-1.pdf) |
| A Survey of Low-bit Large Language Models | [arXiv v3](https://arxiv.org/abs/2409.16694) · [Neural Networks record](https://www.sciencedirect.com/science/article/pii/S0893608025007361) |

## TurboQuant version boundary

The canonical publication is now local as the [ICLR 2026 official proceedings final](https://proceedings.iclr.cc/paper_files/paper/2026/file/5c802ef38ab6e366c2ea06eee554c088-Paper-Conference.pdf). The older `paper-arxiv-v1.pdf` is retained because it contains the disputed quantization-time Table 2 that is absent from the proceedings final. The README records both this removal and the LongBench Table 1 value change.

## Optional watchlist boundary

The eight papers in `RESEARCHER_WATCHLIST.md` remain optional. The original advisor-aligned core contained 15 paper/report identities because FAST was counted despite its `2A` label. The user-promoted standalone LoRA, Outlier Suppression, Flow Matching, and Self-Flow entries expanded the active core to 19. The three papers assigned by 李老师 on 2026-09-04 expand it to 22 and receive integer labels `20`, `21`, and `22`. `OPTQ/GPTQ` and `QuaRot` remain local companions; the other six watchlist papers remain watchlist-only.


## 2026-09-05 additions

WaterSIC and GRACE add standalone core #23 and #24, bringing the current core to 24. The separate VLA-idea library is not included in this inventory. Sources: [WaterSIC](https://arxiv.org/abs/2603.04956v2), [GRACE](https://arxiv.org/abs/2601.22709v5). See [delivery validation](research/SENIOR_REPORT_DELIVERY_2026-09-05.md).
