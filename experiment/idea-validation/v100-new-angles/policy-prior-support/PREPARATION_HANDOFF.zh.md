# Policy-prior-support CPU preparation handoff

日期：2026-09-13。只准备 reset-only input；本地未连接/提交 cluster，未加载 checkpoint/model，未调用 `step` 或 `render`。

## 交付

- `policy_support_prepare.py`：独立 fail-closed preparation，复用最终 TD-MPC2 CPU runtime 的 `numpy==1.26.4`、`dm-control==1.0.16`、`mujoco==3.1.2`。
- `policy_support_prepare.sh`：单节点 `UGGPU-TC1`、2 CPU、8 GB、5 min；`allocation_guard.sh` 是首个 scheduler-dependent action，随后才复制脚本/协议 alias 或执行 Python。
- shell 默认目标为 `${TOP}/policy_prior_support_ready`，输出为 `${TOP}/artifacts/${SLURM_JOB_ID}`；两个目录已存在时立即停止，不覆盖旧资产。

## 冻结输入合同

- seeds：`5217..5224`，task：`cartpole-balance`，每个环境只调用一次 `suite.load(...); reset()`。
- raw schema：`tdmpc2-cartpole-reset-input-v1`；预期 `observations` shape `[8,5]`，flatten order 为 `cart_position, pole_zz, pole_xz, cart_velocity, pole_angular_velocity`；同时保存 position/velocity 分量、seed 和 `metadata_json`。
- parent manifest：默认 `${TOP}/tdq_compatible_checkpoint/manifest.json`，必须是固定 SHA256 `9240979105b919f6051f27261d00ff33652105f962d039728638caabb042d369`。脚本只读取并 hash 这个小 manifest，不 hash 或加载 source/checkpoint payload。
- root 生成的 `${TOP}/control/policy_support_protocol.zh.md` 是必需协议 alias；shell 将它复制到 job artifact，Python 记录其 path/SHA256/size。可用 `POLICY_SUPPORT_PROTOCOL` 显式覆写 alias 路径。

## 输出与边界

远端 asset 将包含 `PREPARATION.lock`、`observations.npz`、`runtime_identity.json`、`manifest.json` 和 status；job artifact 保存脚本副本、协议副本、manifest/status/run log。manifest/status 需保持 controller 64 KiB 限制。manifest 绑定 allocation、runtime、parent/protocol hash、seed、shape 与 zero-step/zero-render/no-inference 证据，供 GPU screen 逐项校验。

root 审查通过并确认协议 alias 后再由 root 统一提交；本文件不包含提交命令，也不授权本 agent 连接集群。
