# rejected_attempt：STRC（保留的历史草案）

本目录只标记旧的 Signed/Temporal Residual Calibration 草案为 **rejected_attempt**，不删除也不改写原文件。历史正文仍在同级的 `TENTATIVE_MECHANISM.zh.md`、`phase2_generate/phase2_generate_output.json` 和 `STATUS.zh.md`；它们不再是当前候选。

拒绝原因：

1. `K_r(ell)=E[e_{t+ell}r_t^T]` 只能作配对轨迹的 association，不能叫 causal transfer kernel。
2. signed cumulative 在 `gamma=1` 时对 transition residual 会 telescope 到 terminal displacement；`sum ||K_ell||^2` 还丢失 lag 间符号，因此不能宣称 cancellation 机制。
3. phase-shuffled action injection 改变了 policy/trajectory，无法形成干净的归因；DINO-WM 也没有 denoising schedule，QuantWAMs schedule 对照不适用。
4. conditional swap 仍有退化为单 site scoring 的风险，且原计划 2--6 A100 GPU-days 超出本轮 bounded pilot。

后续 non-VQ 候选已改为冻结 baseline W4 residual 的 finite-difference transport 对齐；不复用 STRC 名称，也不把一般 observer/error feedback 作为贡献。
