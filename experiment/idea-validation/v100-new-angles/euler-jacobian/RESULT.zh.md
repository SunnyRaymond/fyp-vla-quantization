# Euler Jacobian：mechanism_no_go，STOP

2026-09-13。GPU64835 在 TC1N02 V100 完成，wall 1m40s、workload 95.2315s、peak allocated 2,733,601,792 bytes；独立 CPU64836 完成10s。六个固定 observation × FP/W4 全部完成。结果不支持“expert W4 在 t=.5 的共同 FP 状态破坏 first-token Euler map 局部 regularity”的假设；没有实现故障可作为 retry 理由。

## 主要结果

六个 FP map 均 eligible，六个 W4 map 均未触发严重 conditioning 或 robust orientation-reversal gate，**0/6 flagged**（预注册 go 需至少3个）。所有 determinant sign 都为 +1。

| task / episode | FP sigma_min | W4 sigma_min | FP min/max ratio | W4 min/max ratio | FP/Q velocity MSE |
|---|---:|---:|---:|---:|---:|
| 0 / 105 | .793398 | .805849 | .962368 | .962069 | .015056 |
| 1 / 52 | .792653 | .803998 | .967317 | .964663 | .015535 |
| 2 / 84 | .792449 | .808740 | .954135 | .951469 | .019267 |
| 3 / 66 | .783058 | .798952 | .962327 | .960361 | .023810 |
| 4 / 7 | .790701 | .805025 | .936297 | .939121 | .018597 |
| 5 / 8 | .790703 | .803085 | .948236 | .949985 | .023491 |

W4 velocity 确有变化，但 sigma_min 约 .799–.809，远高于 severe conditioning 的 .005 阈值。不能将普通 field drift 改名为 topology failure，也不能事后换到别的 time/noise 找 positive。

## 实现与身份核验

CPU 独立复核全部 receipt/source/checkpoint/input identity、112 expert Linear 的 W4 scale/code/dequant/actual readback/FP restore、非 expert digest、固定 CPU RNG。完整 input Jacobian 的 future-token block、tail intervention、FP grad/no-grad no-op、五步 FP path 重建、两条固定 directional-FD 均通过。24个 FD norm errors 约 .000248–.000600，预设 limits 约 .0367–.0416。无需 tolerance、epsilon、dtype 或 source 修复。

复用六个先前冻结 observation 仅为了省资产准备，新 noise2301，未读取旧实验 scientific arrays。SmolVLA LIBERO checkpoint31d453f7edd78c839a8bbc39744a292686daf0de，weights SHA9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8；LeRobot v0.4.4 actual source 四文件 hard pins通过。FP32 fake quantization，未实现 native W4。

## 边界与停止

结论仅适用于固定 checkpoint、六个 observation、共同 FP-path 的 t=.5 局部32D map。32D 包含25个未执行 padding coordinates；不作7D physical topology、global invertibility、完整 K10 trajectory 或 closed-loop success 推断。generic Jacobian-aware/diffusion PTQ novelty 已在 prior gate 中否决；此处只完成一次 checkpoint-specific 反证诊断。

**STOP：不增加 task/noise/timestep，不训练，不做完整验证。**

## 证据入口

- [冻结协议](PROTOCOL.zh.md)、[独立源码审查](SOURCE_AUDIT.zh.md)、[GPU静态审查](GPU_AUDIT.zh.md)。
- [GPU receipt](job-64835/engineering.json)、[GPU log](job-64835/run.log)、[独立CPU结果](job-64836/verification.json)。
- raw.npz SHA256 `a5d3fb44d128f54256c3ecfb73e859cd41ff0db20a08fe0643e52184841c0cf8`，保留于 `/tc1home/UG/yguo017/v100_newangles_ccds/artifacts/64835/`；同目录保留完整 quant_snapshot.pt。
- producer LF SHA `06b45d7343d9b99cfd40e9fce5b5e9c78642d733c6ee0f0e6eb90c7e2128b6a1`；independent replay `81f4129c968e33d25ed632bd79f8380f813231e6126e22c508c1692f7364c262`；CPU verifier `1c4854c30caab78b4cad9601fc6a7f15817f91268f3ac1d0584d14a42381f346`。
