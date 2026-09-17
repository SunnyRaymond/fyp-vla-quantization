# TD-MPC2 Q-coupling CPU preparation

本目录准备 stratified stochastic-rounding screen 所需的官方 TD-MPC2
`cartpole-balance` reset-only inputs。`tdq_prepare.sh` 是一次 bounded CPU
SLURM allocation：`UGGPU-TC1`、4 cores、16G、10 minutes，不申请 GPU。脚本
提交后的第一项 scheduler-dependent action 是 source
`${TOP}/control/allocation_guard.sh`；随后 Python 再执行
`allocation_guard.require_allocation()`，必须通过真实 job、hostname、owner、
partition、NodeList 检查。登录节点不执行下载、hash、解压、依赖导入或 env reset。

## 隔离目录与单写者

远端 `TOP` 为 `/tc1home/UG/<owner>/v100_newangles_ccds`，目标资产目录为
`${TOP}/tdmpc2_q_coupling`，job 小结果位于 `${TOP}/artifacts/${SLURM_JOB_ID}`。
目标目录只允许新建；如果目录已存在，脚本立即失败，不覆盖、清理或 retry。创建
后写入 `PREPARATION.lock`，记录 job ID、owner、actual hostname 和 partition。
脚本、source archive、checkpoint、identity、manifest 和 raw NPZ 均使用临时文件
后 atomic rename，避免同一目标的并发写入。

## Staging identity

CPU job 在真实 allocation 内下载并保存以下资产，不加载 checkpoint：

- TD-MPC2 source archive：commit
  `e9f59321933cbc8e11a002b842adc7d4ffae8ff1`，URL 为
  `https://github.com/nicklashansen/tdmpc2/archive/e9f59321933cbc8e11a002b842adc7d4ffae8ff1.tar.gz`。
  archive SHA-256、size、extract root、selected source files 的 size/SHA-256
  写入 `source_identity.json`；同时检查 Q modes、planner `avg`、DMControl
  wrapper 与 `num_q: 5` source markers。
- `nicklashansen/tdmpc2` HF revision
  `73a50e2719ed8258c72c7d1fefd23b781d66e35e` 的
  `dmcontrol/cartpole-balance-1.pt`，预期 size `31,344,610` bytes，LFS
  SHA-256/OID 为
  `4919e562d7f22f41a11118d1db1a0ebcb0e2b4681fc7fc594d772f0d5940869b`。
  实际 size/hash 写入 `checkpoint_identity.json`；`torch_loaded` 永远为
  `false`。

单次新增下载总预算为 500 MiB；checkpoint 仅接受 pinned revision、size 和
SHA-256 全部匹配。网络不可用、archive 不完整或 checkpoint identity 不匹配时
保留 job status 并退出，不换 URL、不使用 synthetic checkpoint。

Python 优先复用 `${TOP}/smolvla/venv/bin/python`。reset-only 与后续 Q runtime
必须 exact 的 distributions 是 `dm-control==1.0.16`、`mujoco==3.1.2`、
`tensordict==0.7.2`、`omegaconf==2.3.0`；脚本严格检查这些项。官方环境另列
`gymnasium==0.29.1`，但 reset-only 仅直接调用 `dm_control.suite`，因此实际
`gymnasium` 版本只记录为 non-exact environment dependency，不伪装成 exact。
实际 Python/Torch 版本另行写入 `runtime_identity.json`。
官方 `environment.yaml` 是 Python 3.11、Torch 2.7.1；若复用 runtime 为
Python 3.10、Torch 2.6.0，manifest 明确记录 `exact_official_environment=false`，
但不会因此伪造版本或安装 Torch。

## Reset-only contract

脚本直接使用官方 `dm_control.suite.load("cartpole", "balance", task_kwargs={"random": seed}, visualize_reward=False)`。
固定 seeds 为 `5201..5208`，每个 seed 建立 fresh env，只调用一次 `reset()`，
读取 `OrderedDict(position, velocity)`，然后立即 `close()`。绝不调用
`env.step()`、`render()`、policy、planner、rollout 或 checkpoint loader。

官方 TD-MPC2 `_obs_to_array` 的 `(5,)` flatten 顺序固定为：

`[cart_position, pole_zz, pole_xz, cart_velocity, pole_angular_velocity]`。

NPZ `observations.npz` schema 为 `tdmpc2-cartpole-reset-input-v1`，包含：

- `observations`: finite `float32[8,5]`；
- `position`: `float32[8,3]`；`velocity`: `float32[8,2]`；
- `seeds`: `int64[8]`；`metadata_json`：schema、seed order、flatten order、
  source/checkpoint identity，以及 `env_steps=0`、`render_calls=0`、
  `model_loaded=false`。

`manifest.json` 额外记录 source archive 与 selected-file hashes、checkpoint
identity、runtime/dependency identity、resolved config (`obs=state`、`num_q=5`、
`model_size=5`、`compile=false`)、官方 planner `avg` / target `min` 路径、
action dim 1、wrapper action repeat 2、outer timeout 500、NPZ hash 和每个 seed
的 shape/keys。它只证明 reset input 与 staging 可复现，不证明 Q screen、policy
性能或 closed-loop success。

## 后续使用边界

本 preparation 不提交 GPU screen。未来 screen 必须在另一个真实 V100
allocation 中读取本 manifest；FP terminal policy action、latent Q input、
candidate order 和 official random-two pair schedule 必须先冻结并跨 arms 复用。
五个 Q members 的 `return_type='all'` 输出若被记录，需按官方 `two_hot_inv`
先解码为 scalar，不能平均 raw logits；只可事务性量化 live `_Qs`，target-Q
clone 保持未改动。以上是后续工程契约，本 job 不读取任何 model output。

本轮只写脚本和说明，实际下载、source hash、checkpoint hash、dependency import
和 8 次 reset 均由提交后的 CPU job 产生；若 job 因 runtime、网络或 asset
identity 失败，结果保持 `resource_blocked` / `failed`，不在本机补算。

首个 preparation job `64780` 已通过 allocation guard，但在 staging 前发现复用
runtime 缺少 `dm-control` 与 `mujoco`，因此留下 `resource_blocked`，并已创建
`tdmpc2_q_coupling/PREPARATION.lock`。`tdq_runtime_repair.sh` 只作为一次获准的
isolated repair，将这两个 exact distributions 以 `--no-deps` 写入新的
`${TOP}/tdmpc2_q_coupling_runtime`；它不修改现有 venv、不安装 Torch，也不会
解锁或 retry `64780`。repair 结果只说明依赖是否可导入，不能替代原 preparation
的 source/checkpoint/reset manifest。

repair job `64782` 在真实 node `TC1N05`、allocation verified 的情况下完成了
这两个 wheel 的 isolated staging，但 import verification 因 venv/vendor 均缺少
`dm_env` 而以 `resource_blocked` 结束。依照本任务的一个 repair 上限，不再补装
第二层依赖；因此当前没有合法的 source/checkpoint/reset manifest，后续若继续
必须由 root 重新决定全新隔离目录与依赖策略。

随后获准的 overlay continuation job `64783` 同样在真实 `TC1N05` allocation
内执行。`pip --dry-run --report` 在任何实际安装前发现 resolver 会拉取
`torch==2.14.0`，并且当前 package index 没有满足
`omegaconf==2.3.0` 所需 `antlr4-python3-runtime==4.9.*` 的可用版本，因而
fail-closed。该 job 没有下载 Torch、没有完成 extra overlay import，也没有
调用 ready preparation；`tdmpc2_q_coupling_runtime_extra` 的失败目录、resolver
report 与 `runtime_extra.json` 均保留为证据。当前仍无合法 ready manifest，不能
把 resolver failure 当作 source/checkpoint/reset 成功。

overlay2 job `64784` 的 resolver/install gate 通过，实际新增下载估算为
`52,902,909` bytes，且没有新增 Torch/NVIDIA/Triton；其 import verification
最初因 process 未更新 overlay `sys.path` 而失败。resume job `64785` 修正了该
路径并确认全部 required imports，但 preparation 随后暴露复用 runtime 的
`gymnasium==1.3.0` 与官方环境版本差异。该差异已改为 manifest 中的明确
non-exact identity；不修改 overlay2、不重跑 resolver，继续使用同一已验证
overlay 进行 fresh reset-only preparation。
