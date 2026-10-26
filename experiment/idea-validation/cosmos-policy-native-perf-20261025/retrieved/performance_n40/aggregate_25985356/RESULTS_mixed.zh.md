# Cosmos Policy native performance (mixed measurement designs)

兼容硬件：`NVIDIA A100-SXM4-40GB` / SM80，Torch `2.7.0+cu128` / CUDA `12.8`；实测设备 3 台，UUID：`2c4aaa13-69aa-baac-0ffa-b526d3e2b2c9`, `2f124ae5-a6af-960a-07ba-0c5761ad1090`, `8919f8e7-ec75-715d-9b44-4b5429af2f98`。每个输入仍执行 5 NFE。各行 cohort 可能是已有 40 cases × seeds 0/1 的 80-query legacy 设计，也可能是新 40 cases × seed 0 的 40-query 设计；请按表格中的 Cohort、Seeds、Design 和 N 比较，不应把它们当作同一输入 cohort。Fidelity full80 是独立 gate，仍按原 80-query panel 判定。

Lat. 为 mean；Spd. 为同口径 BF16 mean / method mean；Peak 为实测最大 allocated GB（10^9）；Storage 为 whole-DiT unique tensor payload GiB（2^30），包含 padding/scales/transforms，排除 workspace。

## sampler

真实五次 NFE sampler wall time；Peak 包含 resident tokenizer/VAE 和 endpoint 临时内存。

| Method | Cohort q | Seeds | Design | N | Lat. ms | Spd. | p50 ms | p95 ms | Peak GB | Storage GiB |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 80 | 0,1 | 80q_x3blocks_x3repetitions | 720 | 432.566 | 1.000× | 431.903 | 435.097 | 5.7094 | 3.6441 |
| SpinQuant | 40 | 0 | 40ctx_once_v3 | 40 | 3823.556 | 0.113× | 2234.413 | 17121.419 | 3.0820 | 1.2089 |
| QuaRot | 40 | 0 | 40ctx_once_v3 | 40 | 2217.626 | 0.195× | 2217.078 | 2220.506 | 3.0826 | 1.2094 |
| SVDQuant | 40 | 0 | 40ctx_once_v3 | 40 | 859.117 | 0.504× | 826.036 | 930.139 | 3.1674 | 1.2797 |
| ViDiT-Q | 40 | 0 | 40ctx_once_v3 | 40 | 1387.795 | 0.312× | 1371.710 | 1438.484 | 3.1725 | 1.2865 |
| Ours | 40 | 0 | 40ctx_once_v3 | 40 | 435.695 | 0.993× | 433.615 | 450.014 | 3.3082 | 1.2421 |

## dit_replay

冻结 BF16 网络输入的五次 DiT CUDA Graph replay，CUDA event 计时；非 DiT 模块已 offload。

| Method | Cohort q | Seeds | Design | N | Lat. ms | Spd. | p50 ms | p95 ms | Peak GB | Storage GiB |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 80 | 0,1 | 80q_x3blocks_x3repetitions | 720 | 277.356 | 1.000× | 272.438 | 314.798 | 5.0787 | 3.6441 |
| SpinQuant | 40 | 0 | 40ctx_once_v3 | 40 | 2098.030 | 0.132× | 2093.528 | 2125.251 | 2.4526 | 1.2089 |
| QuaRot | 40 | 0 | 40ctx_once_v3 | 40 | 2089.925 | 0.133× | 2081.083 | 2119.462 | 2.4531 | 1.2094 |
| SVDQuant | 40 | 0 | 40ctx_once_v3 | 40 | 604.784 | 0.459× | 596.748 | 642.076 | 2.5439 | 1.2797 |
| ViDiT-Q | 40 | 0 | 40ctx_once_v3 | 40 | 1190.434 | 0.233× | 1189.768 | 1190.040 | 2.5490 | 1.2865 |
| Ours | 40 | 0 | 40ctx_once_v3 | 40 | 433.901 | 0.639× | 428.298 | 471.932 | 2.6453 | 1.2421 |

## Fidelity evidence (report-only)

按用户要求，四个 native baseline 的 fidelity 误差不阻断性能聚合。下表的 status、query 数和 thresholds 是已有 gate receipt 的原始证据；它们不构成本报告的性能资格判定。性能行的 cohort 和 N 均单独标出；独立 fidelity full80 仍需 80 个 gate queries，性能 timing samples 不能证明 fidelity 通过或 full80 已完成。

| Scope | Method | Phase | Raw status | Completed / expected | Recorded queries | Frozen thresholds | Receipt |
|---|---|---|---|---:|---:|---|---|
| sampler | SpinQuant | representative4 | failed | 1 / 4 | 1 | `{"future_nrmse":0.01,"normalized_motor_rmse":0.01}` | `results/native_gate/sampler/current/spinquant_w4a4/representative4.json` |
| sampler | SpinQuant | full80 | missing | unknown / 80 | unknown | `{"future_nrmse":0.01,"normalized_motor_rmse":0.01}` | `results/native_gate/sampler/current/spinquant_w4a4/full80.json` |
| sampler | QuaRot | representative4 | failed | 1 / 4 | 1 | `{"future_nrmse":0.01,"normalized_motor_rmse":0.01}` | `results/native_gate/sampler/current/quarot_w4a4/representative4.json` |
| sampler | QuaRot | full80 | missing | unknown / 80 | unknown | `{"future_nrmse":0.01,"normalized_motor_rmse":0.01}` | `results/native_gate/sampler/current/quarot_w4a4/full80.json` |
| sampler | SVDQuant | representative4 | failed | 1 / 4 | 1 | `{"future_nrmse":0.01,"normalized_motor_rmse":0.01}` | `results/native_gate/sampler/current/svdquant_w4a4/representative4.json` |
| sampler | SVDQuant | full80 | missing | unknown / 80 | unknown | `{"future_nrmse":0.01,"normalized_motor_rmse":0.01}` | `results/native_gate/sampler/current/svdquant_w4a4/full80.json` |
| sampler | ViDiT-Q | representative4 | failed | 1 / 4 | 1 | `{"future_nrmse":0.01,"normalized_motor_rmse":0.01}` | `results/native_gate/sampler/current/viditq_w4a4/representative4.json` |
| sampler | ViDiT-Q | full80 | missing | unknown / 80 | unknown | `{"future_nrmse":0.01,"normalized_motor_rmse":0.01}` | `results/native_gate/sampler/current/viditq_w4a4/full80.json` |
| dit_replay | SpinQuant | representative4 | failed | 1 / 4 | 1 | `{"dit_forward_nrmse":0.01,"linear_operator_nrmse":0.01}` | `results/native_gate/dit_replay/current/spinquant_w4a4/representative4.json` |
| dit_replay | SpinQuant | full80 | missing | unknown / 80 | unknown | `{"dit_forward_nrmse":0.01}` | `results/native_gate/dit_replay/current/spinquant_w4a4/full80.json` |
| dit_replay | QuaRot | representative4 | failed | 1 / 4 | 1 | `{"dit_forward_nrmse":0.01,"linear_operator_nrmse":0.01}` | `results/native_gate/dit_replay/current/quarot_w4a4/representative4.json` |
| dit_replay | QuaRot | full80 | missing | unknown / 80 | unknown | `{"dit_forward_nrmse":0.01}` | `results/native_gate/dit_replay/current/quarot_w4a4/full80.json` |
| dit_replay | SVDQuant | representative4 | failed | 1 / 4 | 1 | `{"dit_forward_nrmse":0.01,"linear_operator_nrmse":0.01}` | `results/native_gate/dit_replay/current/svdquant_w4a4/representative4.json` |
| dit_replay | SVDQuant | full80 | missing | unknown / 80 | unknown | `{"dit_forward_nrmse":0.01}` | `results/native_gate/dit_replay/current/svdquant_w4a4/full80.json` |
| dit_replay | ViDiT-Q | representative4 | failed | 1 / 4 | 1 | `{"dit_forward_nrmse":0.01,"linear_operator_nrmse":0.01}` | `results/native_gate/dit_replay/current/viditq_w4a4/representative4.json` |
| dit_replay | ViDiT-Q | full80 | missing | unknown / 80 | unknown | `{"dit_forward_nrmse":0.01}` | `results/native_gate/dit_replay/current/viditq_w4a4/full80.json` |

本轮 new training updates = 0；new closed-loop episodes = 0。Native backend 的闭环表现尚未重测，旧闭环结果只属于此前 reference/checkpoint 执行路径。
