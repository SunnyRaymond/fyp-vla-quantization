# D1 + D3：跨去噪步的 INT4 activation 量化器（固定输入 held-out）

设计依据：[W4A4 latent-aware 设计稿 §4、§9.1](../../../research/paper/2026-10-28_w4a4_latent_aware_design.zh.md)。

## 问什么

1. **D1**：同一个 Linear 在相邻去噪步的输入有多像？innovation（本步减上一步）相对本步的能量比、每 token absmax 比（决定 innovation 编码后的 A4 步长），以及相邻两步 A4 RTN 残差的相关。按 video 循环 / action 循环、按步、按 token 组（frame0 vs future，前 10 个执行动作 vs 其余）汇总。
2. **D3**：权重保持 BF16，只换 activation 量化器，跨步的 innovation 编码与误差反馈能把 per-token A4 的误差补回多少？对照 A8、g32 组 scale、dither（只打散时间相关），以及把 action 循环或 prefill 单独升到 A8。

## 设置

- 坐标与权重：冻结 Smooth(α=0.5) + signs + H128（C8 source banks），U = I。走的是历史 H_W16A4 臂同一段代码（`rotation_bank.transformed`，零旋转），所以 **A4 臂就是 H_W16A4**（历史 held-out motor 0.042605 / future 0.038381，闭环 204/280）。启动后 A4 的 case 3000 seed 0 与历史 endpoint 相对差超过 2% 直接失败；80 个 endpoint 的最大差写进结果。
- 跨步量化器（每个 Linear×输入形状一份状态，每个 query 开始清零）：
  `Δ_s = z_s − x̂_{s−1}`（innovation），`v_s = Δ_s + h·w_{s−1}`（反馈），`q_s = Q(v_s)`，`w_s = q_s − v_s`，`x̂_s = x̂_{s−1} + q_s`，GEMM 用 `x̂_s`。于是每步下游看到的误差是 `w_s + h·w_{s−1}`。
- 与上一次调用逐位相同的输入（context K/V、text embedding）直接复用上次 operand，等同部署时的 condition cache。
- Video 层每 query 调用 11 次：10 个去噪步 + 1 次 prefill（clean future，call index 10）；action 层 10 次。

## 臂（13 个）

| 臂 | 内容 |
|---|---|
| W16A16 | BF16 operand（H 坐标的数值下界） |
| A8 | per-token INT8 |
| A4 | per-token INT4 RTN（= 历史 H_W16A4） |
| A4_g32 | per-token、每 32 通道一个 scale（Q-WAM 的 activation 设定） |
| A4_dither | A4 + RPDF dither（每步误差更大但跨步不相关） |
| A4_sd | 一阶误差反馈 h = −1，无 innovation |
| A4_inno | innovation 编码（DPCM），h = 0 |
| A4_inno_sd | innovation + h = −1 |
| A4_inno_sd05 | innovation + h = −0.5 |
| A4_actA8 | video A4，action 循环 A8 |
| A4_preA8 | A4，prefill 那一次调用 A8 |
| A4g32_inno_sd | g32 + innovation + h = −1 |
| A4_inno_sd_actA8 | innovation + h = −1，action 循环 A8 |

## 预注册判据

- D1 go：两循环合并、步 1–9 的 innovation 比中位数 < 0.5，或相邻步 A4 残差相关中位数 > 0.5。
- D3 go：A4_sd / A4_inno / A4_inno_sd / A4_inno_sd05 中最好的一臂，在 first10 motor 与 future NMSE 上都补回 ≥ 50% 的 A4→A8 差距，且两项都低于 A4_g32。
- 机制：A4_dither 的 future 低于 A4，说明 RTN 误差的跨步相干累积存在。

## 运行

单作业、2 张 A100（g2，walltime 4 h）：worker 0 先跑 D1（10 个 C80 context）再跑 7 臂，worker 1 跑 6 臂，最后在同一 allocation 里跑 `analysis.py`。每 15 s 把两张卡的利用率与显存写进 `artifacts/<job>/job.log`。所有 endpoint 逐条落盘，中断后重提交可续跑（`--attempt r2`）。

```bash
python -X utf8 -B test_temporal_quant.py
```
```bash
python -X utf8 -B test_analysis.py
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-temporal-a4-d1d3-20261028/control.py upload
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-temporal-a4-d1d3-20261028/control.py submit
```
```bash
nscc-access/.venv/Scripts/python.exe -Xutf8 -B experiment/idea-validation/fastwam-temporal-a4-d1d3-20261028/control.py pull
```

## 输出

`results/RESULTS.zh.md`、`summary.json`、`d1/raw.json`、`arms/<arm>.json`、`endpoints/<arm>/<case>_<seed>.json`、`gates/h_w16a4_first_endpoint.json`、`worker_<i>/{startup,progress,summary,failure}.json`。

范围：固定输入 held-out 精度，eager fake-quant；不涉及 kernel、速度或闭环。
