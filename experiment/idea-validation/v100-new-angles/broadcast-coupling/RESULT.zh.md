# Broadcast activation coupling：scope_limited_preliminary_go，STOP

2026-09-13。GPU64838 在 TC1N04 V100 完成15s，workload13.3443s；独立 CPU64839 完成4s。全部六个state可识别，**5/6达到预注册 gain>=10% gate**，因此仅在这个单步诊断范围 preliminary go。

## 数据结果

Primary 是每个draw的 visual MSE 再取三draw均值，**没有先平均三个预测形成ensemble**。六state的 Shared/Spatial input MSE 在CPU float64复核中差值全部为0，每个patch处三个量化值的multiset也严格相同。

| Wall trajectory | Shared visual MSE | Spatial visual MSE | Gain | >=10% |
|---|---:|---:|---:|---|
| 1035 | .0006149145 | .0003873241 | 37.01% | yes |
| 1534 | .0009274751 | .0000874496 | 90.57% | yes |
| 1158 | .0044601208 | .0013081465 | 70.67% | yes |
| 203 | .0000373623 | .0000344829 | 7.71% | no |
| 1837 | .0003660500 | .0002877086 | 21.40% | yes |
| 1095 | .0015575016 | .0005138187 | 67.01% | yes |

这支持一个具体干预效应：对当前DINO-WM Wall checkpoint的重复state/action activation，改变误差的空间joint assignment能改变一步FP visual fidelity，且这次balanced assignment在5/6状态达到门槛。它不是encoder-versus-predictor模块比较，也不是减少了总输入量化误差。

## 验证与身份

原样保留所有weights与visual channels，只对20个broadcasted activation做A4 fake quantization。完整model state digest前后相同；60次predictor pre-hook精确核对实际输入；source/checkpoint/helper/manifest身份与CPU准备相符。

FP copy no-op、RTN before/after输入与输出exact equal、CPU float32 quantizer/code/dequant、三个uniform RNG与固定offset permutation、每patch三draw multiset、input MSE、全部shape/dtype/finite都通过。没有改threshold、换seed、修补partial arrays或重跑GPU。CPU replay曾在提交前修正两个实现问题（zero-row冗余mask和先平均prediction的错误），未使用任何实验结果选择修正。

本机CRLF文件与上传LF文件的hash不同已被明确区分。Remote freeze SHA `233bd7723478fb6deb99216b51d8cd15a0dd71a2776c073465f4a0f9ef1185a2`，CPU pin通过；不能因本机raw CRLF hash不同擅改远端pin或重跑实验。

## 边界与停止

六个先前冻结的observation是有意复用，未读取旧teacher scientific outputs。本案量化后每次forward的state/action副本不再完全相同，这是自变量；结果仅是一项受控activation诊断。Spatial不是独立patch噪声，而是固定三draw的balanced assignment。Generic novelty未核实；不声称一种全新quantizer。

未验证recorded-future、goal/planner、multi-step rollout、closed-loop success；RTN仅作placement负对照，不能据此宣称胜过RTN或W8。未实现native A4、端到端模型压缩、latency或memory收益。**完成初步验证即STOP，不增加draw/task/scale/horizon，不做完整验证。**

## 证据

- [冻结协议](PROTOCOL.zh.md)、[Root gate](../../../../idea/v100-new-angles/broadcast-coupling/ROOT_GATE.zh.md)、[独立复审](../new-cut-shortlist/BROADCAST_ROOT_REVIEW.zh.md)、[GPU审查](GPU_AUDIT.zh.md)。
- [GPU receipt](job-64838/engineering.json)、[GPU log](job-64838/run.log)、[独立CPU结果](job-64839/verification.json)。
- 完整raw保留于 `/tc1home/UG/yguo017/v100_newangles_ccds/artifacts/64838/raw.npz`，SHA256 `e23d17ef25fbdb1646b13fed1a51be87272e6d0fe2488092bdb1b99b9c521cb4`。
- producer LF SHA `a82cf47e7ec88a1747b01244cab386fab1cff60e95ad9344a17c2bc0d0419a77`；independent replay `6913a752fdee1765359fa76a79b3e926cca512d237fe931aba177a5bb38f8013`；CPU verifier `6f58fb372dd192cdd1785588b7898b1ab17f430af130a4ef003680bf529eb571`。
