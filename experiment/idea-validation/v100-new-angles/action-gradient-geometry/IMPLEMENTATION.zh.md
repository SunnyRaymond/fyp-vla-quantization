# Action-gradient geometry screen implementation

本实现只覆盖 `PROTOCOL.zh.md` 冻结的最小 GPU 采集，不修改协议、不连接集群、不下载模型，也不在本机加载模型。提交后由 `gradient_gpu_screen.sh` 在 CCDS 真实 SLURM allocation 中执行；作业为单 V100、20 分钟，runner workload deadline 为 1080 秒。shell 和 Python 都检查 allocation，Python guard 是进入 runtime 前的第一 workload action。

## 运行对象与 action graph

runner 复用已核验的 `cem_update_ccds/control/smoke_runner.py` 与 `screen_runner.py`，并比较 helper SHA-256。`smoke._runtime` 加载 Wall epoch65 checkpoint 和 validation dataset；runtime 在 Hydra dataset construction 前使用 seed `100000`。脚本强制检查实际 `num_hist=1`、`concat_dim=1`、`frameskip=5`、primitive dataset action dimension `2`，从而 effective action dimension 为 `2×5=10`，以及 encoder embedding `384` 和 model embedding `404`。因此本实验使用的 input identity 是 `visual 384 + proprio 10 + action 10 = 404`。

官方 action graph 是 `visual_world_model.py::VWorldModel.rollout(obs_0, act)`：`act` 的唯一可求导输入是独立的 normalized action leaf，经过 `encode_act`、predictor 和 `replace_actions_from_z`。每个 candidate 的 objective 是 `smoke._objective()` 返回的官方 `create_objective_fn(alpha=1, base=2, mode="last")`，即 terminal visual MSE 加 terminal proprio MSE；没有 success proxy 或 environment rollout。模型参数全部设为 `requires_grad=False`，使用 `torch.enable_grad()` 绕过 runtime 的 global grad-disabled 状态；不使用 STE、不求 parameter gradient。

## 固定样本、两臂与原始 schema

每个 episode 使用 dataset positions `118..123`，environment seed 为 `970000+i`，candidate seed 为 `980000+i`。subset `.indices` 只做 metadata mapping，并证明 fresh positions 对底层 episode IDs 与排除的 `0..117` 不重叠；不会读取 reserved state。目标只调用 `screen._new_target` 的 layout、`sample_random_init_goal_states` 和 `prepare` 初始化路径，随后将 target metadata `goal_H` 设为 `2`；不会调用 `smoke._make_explicit_targets`，不会做 25-step 或任何 environment rollout。

candidate pool 是每 episode 固定 seed 生成的 `standard_normal` pool，shape `(64,2,10)`；anchor 恒为 pool indices `0,1,2,3`，candidate 0 保持非零。两臂为 `FP32` 与 `predictor_W4`。W4 只对 predictor transformer 的六个 Linear groups 使用既有 `smoke._quantize_group`：signed `[-7,7]`、per-output-channel symmetric RTN、dequantized FP32；encoder unchanged。每 episode 前后都用完整 snapshot 做 exact `torch.equal` restore 检查。

`raw_gradient.npz` 的 schema 是 `action-gradient-geometry-raw-v1`，键和 shape 为：

- `actions`: `(6,64,2,10)`，candidate axis 在前，随后为 `H=2`；
- `scores`: `(6,2,64)`，arm order `FP32,predictor_W4`；
- `gradients`: `(6,2,4,2,10)`，四个固定 anchor 的 action gradients；
- `fp_step_scores`: `(6,2,4)`，两种 arm direction 都由 FP32 model 评价负步；两个方向各自使用 batch-4 forward；
- `base_scores`: `(6,4)`，独立 batch-4 FP32 anchor scores（不强行把 batch-64 的同 anchor 舍入结果当作 base）；
- `fd_values`: `(6,2,2)`，每个 arm 第一 anchor 的 raw `[plus,minus]` scores；
- `completed`, `dataset_indices`, `source_episode_ids`, `arm_names`。

负步长度固定 `0.10`，不 clipping、不 noise、不 line search。FD 使用同一 arm 的 first-anchor gradient unit direction，`epsilon=0.005`；runner 保存 plus/minus、中心差分、`dot(g,u)` 和协议容差。所有 raw 值必须 finite。GPU runner 不计算 global/local research verdict；root 的独立 CPU verifier 用 raw arrays 按协议以 FP64 计算 rank Spearman、centered score NRMSE、cosine 和 improvement gates。

## 身份、边界与验收

summary 记录 allocation、V100、checkpoint/helper/source identities、真实 model structure、metadata-only dataset mapping、target fingerprints/seeds、quantizer audit 路径、raw schema 和每 episode 工程记录；`summary.json` 保持小于 64 KiB，完整 raw 只保留在远端 artifact。当前文件只做 AST 与 shell 静态检查，尚未运行模型或数值实验。该 fake-quant screen 不支持 native W4 memory/speed/deployment 结论；六个 episode 和一个 pool 也不认证 novelty 或 generalization。若 runtime/source/helper/checkpoint identity 不匹配，应 fail closed。
