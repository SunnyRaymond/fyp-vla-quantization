# VLA Quantization Literature Review（Alternative Corpus）

這是一個獨立於既有 reading guide 的 strict-scope corpus。入選 paper 必須同時直接研究或實測 **Vision-Language-Action (VLA)** 與 **model quantization**；只把 action discretization、Vector Quantization action tokenizer 或 latent codebook 稱作「quantization」的 paper 不納入。

## 從哪裡開始

- 完整 literature review、comparison matrix、search protocol、排除理由與研究缺口：[`allinone.md`](allinone.md)
- 每篇 paper 的資料夾包含 local PDF 與中文 self-reading README；technical terms、model names、benchmarks 與 proper nouns 保留 English。
- Reading Questions 刻意不附答案，方便你先自行閱讀與作答。

## 建議 reading route

### 90-minute map

依序閱讀 `01 OpenVLA`、`02 SQIL`、`07 QVLA`、`09 QuantVLA` 的 README，再讀 `allinone.md` 的「核心 synthesis」。這條 route 先建立 baseline、QAT、action-aware PTQ 與 DiT-aware PTQ 四個支點。

### 6-hour method route

1. Foundation：`01 OpenVLA`、`02 SQIL`
2. Native low-bit / recovery：`03 BitVLA`、`04 RLRC`
3. Static VLA-aware PTQ：`07 QVLA`、`08 HB-VLA`、`09 QuantVLA`
4. Temporal / trajectory-aware PTQ：`11 DyQ-VLA`、`12 DA-PTQ`、`15 Mix-QVLA`
5. Aggressive full-stack / sub-4-bit：`13 ActQuant`、`14 HoloQ-VLA`

### Deployment, hardware, and security route

閱讀 `05 SQAP-VLA`、`06 Accessible Physical AI`、`10 LiteVLA-Edge`、`16 Embodied.cpp`、`17 VQVLA`、`18 FlashDrive`、`19 Bit-Flip Attacks`、`20 SpecVLA`。這條 route 回答 model size 之外的 latency、control frequency、peak memory、energy、hardware mapping 與 weight integrity 問題。

## Corpus inventory

| # | Paper | Role | Source status | Local files |
|---:|---|---|---|---|
| 01 | OpenVLA | Foundation / quantization baseline | CoRL final (PMLR) | [README](papers/01-openvla/README.md) · [PDF](papers/01-openvla/paper-corl-final.pdf) |
| 02 | SQIL | VLA-aware QAT | ICCV 2025 final | [README](papers/02-sqil/README.md) · [PDF](papers/02-sqil/paper-iccv-final.pdf) |
| 03 | BitVLA | Native 1.58-bit VLA | arXiv v2 | [README](papers/03-bitvla/README.md) · [PDF](papers/03-bitvla/paper-arxiv-v2.pdf) |
| 04 | RLRC | Pruning + RL recovery + optional NF4 | arXiv v2 | [README](papers/04-rlrc/README.md) · [PDF](papers/04-rlrc/paper-arxiv-v2.pdf) |
| 05 | SQAP-VLA | W4A4 + token pruning | arXiv v1 | [README](papers/05-sqap-vla/README.md) · [PDF](papers/05-sqap-vla/paper-arxiv-v1.pdf) |
| 06 | Towards Accessible Physical AI | Consumer-GPU deployment | arXiv v1 | [README](papers/06-accessible-physical-ai/README.md) · [PDF](papers/06-accessible-physical-ai/paper-arxiv-v1.pdf) |
| 07 | QVLA | Action-centric mixed-precision PTQ | ICLR 2026 final; local arXiv v1 fallback | [README](papers/07-qvla/README.md) · [PDF](papers/07-qvla/paper-arxiv-v1.pdf) |
| 08 | HB-VLA | 1-bit weight-only PTQ | arXiv v2 | [README](papers/08-hbvla/README.md) · [PDF](papers/08-hbvla/paper-arxiv-v2.pdf) |
| 09 | QuantVLA | Scale-calibrated DiT PTQ | CVPR 2026 final | [README](papers/09-quantvla/README.md) · [PDF](papers/09-quantvla/paper-cvpr-final.pdf) |
| 10 | LiteVLA-Edge | GGUF edge deployment | arXiv v1 | [README](papers/10-litevla-edge/README.md) · [PDF](papers/10-litevla-edge/paper-arxiv-v1.pdf) |
| 11 | DyQ-VLA | Runtime dynamic activation precision | arXiv v2 | [README](papers/11-dyq-vla/README.md) · [PDF](papers/11-dyq-vla/paper-arxiv-v2.pdf) |
| 12 | DA-PTQ | Drift-aware mixed-precision PTQ | arXiv v1 | [README](papers/12-da-ptq/README.md) · [PDF](papers/12-da-ptq/paper-arxiv-v1.pdf) |
| 13 | ActQuant | Action-guided sub-4-bit PTQ | arXiv v3 | [README](papers/13-actquant/README.md) · [PDF](papers/13-actquant/paper-arxiv-v3.pdf) |
| 14 | HoloQ-VLA | Uniform full-stack W4A4 PTQ | arXiv v3 | [README](papers/14-holoq-vla/README.md) · [PDF](papers/14-holoq-vla/paper-arxiv-v3.pdf) |
| 15 | Mix-QVLA | Task-evidence-aware PTQ | arXiv v1 | [README](papers/15-mix-qvla/README.md) · [PDF](papers/15-mix-qvla/paper-arxiv-v1.pdf) |
| 16 | Embodied.cpp | Portable quantized VLA runtime | arXiv v3 | [README](papers/16-embodied-cpp/README.md) · [PDF](papers/16-embodied-cpp/paper-arxiv-v3.pdf) |
| 17 | VQVLA | Motion-aware Vector Quantization accelerator | arXiv v1 | [README](papers/17-vqvla/README.md) · [PDF](papers/17-vqvla/paper-arxiv-v1.pdf) |
| 18 | FlashDrive | W4A8 full-pipeline driving acceleration | arXiv v1 | [README](papers/18-flashdrive/README.md) · [PDF](papers/18-flashdrive/paper-arxiv-v1.pdf) |
| 19 | Bit-Flip Attacks on VLA Models | Quantized-weight security | arXiv v1 | [README](papers/19-bit-flip-attacks/README.md) · [PDF](papers/19-bit-flip-attacks/paper-arxiv-v1.pdf) |
| 20 | SpecVLA | Quantized verifier + speculative control | arXiv v1 | [README](papers/20-specvla/README.md) · [PDF](papers/20-specvla/paper-arxiv-v1.pdf) |

## Evidence boundary

- `01`、`02`、`07`、`09` 有 peer-reviewed final；其餘 16 篇以 arXiv preprint 為準。這是一個快速變動且明顯偏向 2026 的 emerging literature。
- `17 VQVLA` 與 `20 SpecVLA` 的 local PDFs 雖排版為 MICRO 2026，但 ISBN/DOI 仍是 placeholder，因此本 corpus 不把它們當作已驗證的 proceedings final。
- `06`、`10`、`16`、`17`、`20` 主要提供 deployment/system evidence；不可用來替代 matched closed-loop quantization comparison。
- 跨 paper 的 success rate、latency 與 memory 數字不可直接排名，除非 model、checkpoint、hardware、benchmark version、rollout count、action interface 與 precision coverage 都一致。

## Local validation

20/20 PDFs 已通過必要檢查：PDF header 可辨識、可由 `pypdf` reopen、未加密、page count 大於零、first-page text 可提取。依照你的要求，**沒有建立 SHA-256 inventory**，也沒有做與閱讀目的無關的逐頁 render。

最後更新：2026-09-03（Asia/Singapore）。
