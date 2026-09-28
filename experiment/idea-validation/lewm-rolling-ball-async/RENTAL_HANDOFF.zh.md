# Rolling Ball / AutoDL 4090 接续

当前 AutoDL 主机为用户授权克隆的新 1×RTX 4090 实例：`autodl-container-59db4f87a9-11fcb48a`，driver `570.124.04`，显存 24564 MiB；root 已核实 base Torch CUDA 可用且设备数为 1。本次 campaign 使用 policy server PID `2477`，startup identity 确认加载 epoch-100 checkpoint；此处不镜像当前电源状态。系统最小安装了 `libXt6` / `libGLU`；默认 GLX Vulkan ICD 下 `vulkaninfo` 失败，使用 task-local `controls/nvidia_headless_icd.json`（NVIDIA EGL ICD）并设置 `VK_ICD_FILENAMES` 后，`vulkaninfo` exit 0 并枚举到 RTX 4090 / driver 570.124.04。

SDK empty-scene smoke 003（PID `4426`）已 PASS：10 steps、native exit 0、wrapper exit 0，ready report 为 `results/sdk-empty-scene-newhost-003/ready.json`。SDK run 002 也走完 10 steps，但 close_stage 挂住后由 root 定向停止；003 在保存结果后使用官方 `close(skip_cleanup=True)` 并正常退出。两次都是 empty-scene SDK 检查，不构成 task-camera RGB/action 证据。

Pair 001 因缺少 `isaaclab.sim.utils.prims.clone` 而 FAIL；旧 strict-v1 Pair 002 的 `'list' object has no attribute 'to'` 仍保留为早期失败记录。Pair 003–005 的 TAA RGB gate FAIL（mean 约 3.4、p99 约 17）；Pair 006–008 的 FXAA strict RGB gate 也失败，Pair 008 mean 0.3431986、p99 3，初始物理检查 PASS、same-first-action 未验证。physical-v2 Pair physical-001 因额外 dynamic task attributes FAIL。Pair physical-002 的内嵌 `pair_check` 对全部 50 seeds 和 same-first-action physical state PASS（maxdiff 0.0），并保存 10,781,767-byte bank；但其顶层 `pair.json` 因 `flush` TypeError 仍为 false/error，原始错误保留。独立 `fixture_bank_validation.json` 为 `PASS_PHYSICAL_FIXTURE_BANK_VALIDATION`。2,337,583-byte 压缩 bank 已取回到本地 `artifacts/autodl/pair-physical-50-newhost-002`。

旧实例 `autodl-container-e7e742ba1d-c91a1edb`（driver `595.71.05`）仍保留为 CPU preparation 历史来源：其 CPU install 与 policy import repair 已完成；用户曾主动切到无卡模式等待资源准备。旧 `results/cpu-packages` 未克隆到新实例，CPU preparation 的记录保留在本地 artifacts；不要尝试从新实例读取该旧路径，也不要重复安装。该历史不描述当前实例的 GPU 状态。

已完成：ASPIRE2A 正式 LeWM 100 epochs（25578999，PASS_epoch100），完整原版 CEM 离线接入（25579057，PASS_protocol_and_planner_shape）。同一 epoch100 checkpoint 用于三阶段。A100 单次 planner 4.7223 s / CEM 4.0880 s 仅是离线接入结果，不能推断 4090 的延迟或接球成功率。

紧凑迁移 bundle 已由 CPU export 25579500 导出：tar.gz 74,570,166 bytes。GPU 等价检查 25579502 已 PASS：180 goal embeddings、返回 action 和 300 candidate costs 的 max_abs_diff 均为 0。用户授权本次单包 SFTP 经本机中转后，bundle 已在 AutoDL 解压到 `/root/autodl-tmp/rolling-ball-lewm/bundle/rolling-ball-lewm-epoch100`，checkpoint 72,263,938 bytes、goal frames [180,224,224,3] 已就位。本机只做传输，没有加载模型。见 `RELAY_RESULT.json` 和 `bundle_ready.json`。

三模式评估入口已完成，stdlib 自检通过真实 `run_episode()` 的调度分支：sync/K0 等价、K2 精确 hold、等待 future 时继续推进、终止时 drain/drop，以及 client/server 时钟分开。Isaac Sim / Isaac Lab / ReflexBench / policy 独立环境已完成安装，22 个必需 USD/纹理文件和 SDK MDL 源已准备。最终 CPU import repair 为 PID10390 / exit0 / `PASS_CPU_POLICY_IMPORTS`；保留先前 datasets/pyarrow 导入失败记录。Base Python3.12 的预装 PyTorch等包未改，Python3.11 的 simulator 用 torch2.7，冻结 policy 用 torch2.8。资源就绪总表见本地 CPU preparation artifacts。SDK empty-scene smoke 003 已通过。Pair physical-002 的 scene/task/event/counter/RNG 与 same-first-action bank gate 已由内嵌检查及独立 loader 验证；strict RGB 仍 FAIL。Sync smoke 003（`results/sync-smoke-physical-001/sync.json`）完成 0/3 task successes，formal 8-condition campaign 已完成；每个 condition 为 50 episodes，成功数见下文与 addendum。

归档导出后更新的 evaluator、policy server、freezes、wrapper 和接续说明已作为小型 control overlay 放在远端 `controls`；后续从这些最新入口继续，不重新打包训练数据或模型。

SSH 信息已通过 `credentials.env` 读取，AutoDL host key 已固定；不把密码写入实验记录。用户提供的 6006 HTTPS 服务地址已保存为 `AUTODL_SERVICE_URL`。临时上传服务已停止，之后不依赖该服务传输模型。

旧实例的系统信息记录为 Ubuntu 22.04.5 / RTX 4090 / driver 595.71.05。当前克隆主机 driver 为 570.124.04；NVIDIA EGL ICD 的 Vulkan enumeration 通过，SDK empty-scene 003 也在此 driver 上完成 10 steps 并 exit 0。该结果只确认 empty-scene runtime，不证明 task-camera RGB 重复性，也不把此 driver 描述为官方 tested driver。Simulator 与 LeWM server 使用独立 Python 3.11 环境，通过本机 HTTP 通信。`run_rented.sh` 保存每 30 s 的 GPU 利用率/显存及退出记录。实际 CPU 准备历史见 `AUTODL_RUN.json`。

ASPIRE2A compute 到 AutoDL 的 SSH 和 HTTPS 均返回 No route to host，因此直连传输未成功。用户明确授权了仅本次模型包的 SFTP 本机中转例外；已完成，没有扩大后续 login node 操作范围。AutoDL 的 CPU 准备不受 PBS allocation 限制。两端仍保留 SSH host-key verification。

严格 v1 的渲染复现尝试仍保留于 `EVAL_FREEZE.json`：Pair 008 FXAA/256 的 mean 0.3431986、p99 3 未通过原门，停止 render warmup 调整。physical-v2 的设计不变，RGB 仅诊断、planner 每次都拿新 native RGB。Pair physical-002 bank 已独立验证且完整取回；其顶层 summary 的 `flush` TypeError 不被重写为 PASS。Sync smoke 001/002 分别因 Gym reset order 与 pinned Lab rerender config 属性兼容问题失败；003（`results/sync-smoke-physical-001/sync.json`）正常完成但 0/3 successes，每集 75 ticks 并由 native phase 2 结束。PID `30529` 的 `controls/run_campaign.sh` 已完成并退出：sync50 → fixed-delay 6×50 → true-async50，同一 physical-002 bank、epoch-100、原 CEM/native physics。三阶段 gate 均 true，native/runner/wrapper exit code 均为 0。每组 50 episodes，sync/K0/K1/K2/K4/K8/K16/async 成功数为 1/1/0/2/0/4/0/1。Sync 与 K0 成功 seed 不同；结果不能证明 latency/delay 单调下降或带来收益。True-async 的 1262/1262 ticks 均 wall deadline overrun，未达到 40 ms hard realtime；完整指标见 `EVAL_ADDENDUM_PHYSICAL_PAIRING_V2.json`。Summary JSON 与 CSV 已取回，formal evidence tar.gz 已取回（2,513,152 bytes）。不要重启 SDK/server/bank/campaign。电源和关机状态唯一见 `AUTODL_RUN.json` 的 `shutdown_after_completion`；root 会维护该字段中的最终状态。

实验 campaign 已完成，但 formal evidence archive 已完整取回（2,513,152 bytes）。电源/关机状态唯一记录于 `AUTODL_RUN.json` 的 `shutdown_after_completion`；root 会维护该字段中的最终状态。最终结果由 root 汇总到 [RESULT.zh.md](RESULT.zh.md)。
