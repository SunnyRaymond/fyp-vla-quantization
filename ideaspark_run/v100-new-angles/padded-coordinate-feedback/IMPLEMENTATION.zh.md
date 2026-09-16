# Analytic padded-coordinate screen implementation

本实现对应冻结协议的四臂 analytic padded-coordinate intervention：`FPnative`、`Qnative`、`FPanalytic`、`Qanalytic`。它复用 Flow 64763 的 `flow_screen.py` 进行 strict local checkpoint loading、processor、raw sample conversion、expert-only W4 quantizer 和 allocation-independent runtime identity，但不调用其 `_run`、`_load_manifest` 或旧的 12-sample summary/metrics pipeline。

`analytic_padding_gpu.sh` 只在真实 CCDS SLURM V100 allocation 中运行，15 分钟作业、runner 12 分钟 deadline；guard 在模型、样本、hash 和 raw I/O 前执行。模型仍从 `${TOP}/smolvla` 读取，manifest 为 `${TOP}/padded_feedback/manifest.json`，base identity 由 manifest 的 `base_identity_path` 和 SHA-256 绑定。CPU preparation 选择的 8 个 sample 必须保持 task/episode/frame metadata，样本缺失或 hash 不符直接 fail closed，不替换样本。

## 实际 sampler 接线

官方 LeRobot v0.4.4 `sample_actions` 从同一初始 noise 开始，固定 `num_steps=10`、`dt=-0.1`，每一步使用 `time=1+step*dt`，调用 `denoise_step(x_t=..., timestep=...)`，再执行 `x_t=x_t+dt*v_t`，最终才截取 physical action `[..., :7]`。自有 recorder 在每次调用后保存 `predicted_velocity`、当前 `x_inputs`、实际返回的 `used_velocity` 和 `times`，并恢复原方法。

每个 arm 都在自己的完整 current `x_t` 上调用自己的模型。`FPnative/Qnative` 原样返回 velocity；`FPanalytic/Qanalytic` 只将该 arm 返回的 `[...,7:32]` 替换为同一 branch 初始 noise 的 pad slice，`[...,0:7]` 原样保留。没有 FP/Q hybrid field、没有另一个模型 oracle、没有 RTC、compile、AMP 或最终 action 置零。每次 branch 都 `policy.reset()`、在显式 `torch.no_grad()` context 中运行，并 clone 相同 seed 的完整 noise。

Q arm 仅对 `flow_screen.py` 的 `expert_W4` allowlist 使用 signed `[-7,7]`、per-output-channel symmetric RTN、dequantized FP32；language/vision/backbone 和其它权重保持 FP32。每 arm 前后使用完整 allowlist snapshot、bypassed-state digest 与 exact restore 检查。

## raw schema 与工程 gate

`raw_analytic_padding.npz` 的 schema 为 `analytic-padding-raw-v1`，键和 shape 为：

- `predicted_velocity`, `used_velocity`, `x_inputs`: `(8,2,4,10,50,32)`；
- `actions`: `(8,2,4,50,7)`；
- `noise`: `(2,50,32)`；`times`: `(10,)`；
- `completed`: `(8,)`；`episode_ids`；`arm_names`。

每 episode 完成后先写 raw，再更新小型 `engineering.json`。工程记录包括 manifest/base identity、实际 LeRobot source/runtime hashes、processor identity、shape、no-op action exact equality、native used/predicted equality、analytic physical slice equality、首步 FP/Q physical equality、analytic pad/noise exact slice、`x_pad - t*z_pad` 最大误差和 final weight restore。`summary.json` 小于 64 KiB，只引用 raw 与 engineering 文件；四臂 scientific E0/S/E1/E2、binding 和 gain 由 root 的独立 CPU verifier 从 raw 计算。

本 screen 只支持 offline FP-fidelity intervention 诊断。它不证明 physical action 正确、environment success、部署收益、native W4 performance 或 analytic path 是 ground-truth；novelty 仍未认证。
