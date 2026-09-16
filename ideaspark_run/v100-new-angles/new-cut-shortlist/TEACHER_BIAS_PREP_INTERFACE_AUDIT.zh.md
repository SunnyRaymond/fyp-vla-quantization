# Teacher-bias CPU preparation：interface audit

> Root复核纠正（后于下方原审查）：下方“确定 environment adapter blocker”不成立。它混淆了本地 reproduction checkout 与实际 CCDS checkout 的文件格式。真实CCDS历史安装器 `../../world-model-quantization/experiments/cem-update-ptq-ccds/prepare_assets_ccds.py:31` 明确写入两行compact Wall注册，正是当前CPU gate的期待内容；不得改成仅本地存在的multiline表示。当前网络不可达，因此实际远端文件是否漂移仍待CPU allocation核验。下方原审查保留以便追溯，但该blocker及修正建议已撤销。Mapping fallback建议接受并已删除；固定seed42加入selection记录。引用的历史PBS config是静态参照，不是本次CCDS config的live证明。

日期：2026-09-13。范围：只读核对 `teacher-bias/prepare_teacher_bias.py/.sh`、DINO-WM Wall 的 `wall_dset.py`、`traj_dset.py`、`img_transforms.py`、`visual_world_model.py` 及 `source/conf/env/wall.yaml` 和 checkpoint config。未读取 dataset/checkpoint 内容，未加载模型、未做数值计算、未连接 cluster。目标是发现首个 sample 写出前的接口、split/mapping 和 normalization 风险。

## 判定

当前 preparation 有一个确定的 **`resource_blocked_before_first_sample`**：批准的 Wall-only environment adapter 与 preparation 的 exact text gate 不匹配。另有一个应在同一修订中收紧的 **mapping fail-open** 风险。除此之外，当前 config 的 nested `env.dataset` split、valid mapping 方向、frame/action 选择和“一次 normalization”路径与官方 source 一致；没有发现第二次 normalization 的实际调用。

## 确定 blocker：approved `env/__init__.py` 文本不匹配

`prepare_teacher_bias.py:182–196` 允许 tracked modifications 仅为 `models/dino.py` 与 `env/__init__.py`，这是正确的两-adapter边界；DINO adapter 也按 HEAD 内容只替换 `facebookresearch/dinov2` 的 pinned commit。问题在于脚本把 environment adapter 写成两行 compact string：

```text
from gym.envs.registration import register
register(id="wall", entry_point="env.wall.wall_env_wrapper:WallEnvWrapper", max_episode_steps=300, reward_threshold=1.0)
```

但当前 `reproduction/dino-wm-wall/source/env/__init__.py` 实际是带说明 comment 的多行 `register(...)`，并移除了其它环境注册。当前 source HEAD 是冻结的 `0a9492fa12044b852ae9e001cc74604b79c8bb0c`，working tree 的两个改动正是批准的 `env/__init__.py` 与 `models/dino.py`；因此这不是 source scope 越界，而是 gate 的 approved representation 写错。`_source_records` 会在建立 dataset 前抛出 `ResourceBlocked("environment registration differs...")`，不会到达首个 sample。

**最小修复：** 把 `wall_registration` 改成当前批准 adapter 的完整 canonical text（包括 comment、括号和换行），或对该文件使用预先固定的 approved-adapter SHA-256；不要用宽泛 whitespace normalization 放过其它注册、import 或 entry point。修复后保留 `allowed_patches`，并继续检查实际文件只注册 Wall、entry point 为 `env.wall.wall_env_wrapper:WallEnvWrapper`、`max_episode_steps=300`、`reward_threshold=1.0`。本 audit 不改该脚本。

## Config 与 valid split/mapping

实际执行的 checkpoint config 是 `artifacts/16170841.pbs101/checkpoint-config-original.yaml`，其中 `env.dataset` 明确为：

- `_target_ = datasets.wall_dset.load_wall_slice_train_val`；
- `n_rollout = null`，`normalize_action = true`；
- `data_path = ${oc.env:DATASET_DIR}/wall_single`；
- `split_ratio = 0.9`、`split_mode = random`。

来源树的 `source/conf/env/wall.yaml:5–14` 具有同一 nested `dataset` 结构。当前 script 已从 `model_cfg.env.dataset.split_mode/split_ratio` 读取并核对，而不是读取不存在的顶层 `model_cfg.env.split`；这一点与 root 已修订的 `env.dataset` 字段一致。shell 又把 `DATASET_DIR` 设为 `${ROOT}/data`，所以 interpolation 的目标是 `${ROOT}/data/wall_single`，与随后 `data_path == dataset_root/wall_single` 检查一致。

官方 loader `wall_dset.py:119–131` 的 random 分支先建立一个 unsliced `WallDataset`，再调用 `get_train_val_sliced`。`traj_dset.py:126–135` 使用固定默认 `random_seed=42` 做 trajectory-level random split；`TrajSubset` 保存底层 dataset 与 `indices`。loader 返回的第二个对象 `traj_dset['valid']` 是 `dset_val`（`wall_dset.py:149–154`），所以 preparation 当前取 `trajectory_datasets['valid']` 是正确的，且不经过 `TrajSlicerDataset.slices` 的随机 permutation。`prepare_teacher_bias.py:318–323` 再用 `valid.indices` 将 validation-local `124..129` 映射回 unsliced WallDataset source IDs，并拒绝重复和 locked `84..95`，方向正确。

**需收紧的最小 mapping 修复：** `_resolve_trajectory_mapping` 当前在 `valid.indices is None` 时退回 `range(len(valid))`，并在缺少 `.dataset` 时把 `valid` 自身当作 base。这会把未来 loader/interface 漂移静默解释成 identity mapping，可能错绑 source trajectory。由于当前冻结协议要求 unsliced `TrajSubset` mapping，应改为：要求 `valid` 同时具有明确的 sequence `indices` 和底层 `dataset`，两者缺一立即 `ResourceBlocked/PreparationError`；再核对 `len(indices)==len(valid)`、所有 indices 为非负整数且不重复。不要保留 identity fallback。

`len(valid) <= 129` 的 guard 能保证固定 local IDs 可用；source `get_seq_length` 返回 action trajectory 长度，script 仍在每个 sample 检查 `length > 25` 且 `length >= 25`，因此 frame 25 与 action 0..24 不会静默越界。`frameskip=5`、primitive action dim 2 下，`actions25.reshape(5,5,2).reshape(5,10)` 与协议的五个 model action blocks 一致。

建议把 `random_seed=42` 也写入 manifest 的 `loader_settings`/`selection`，因为它目前来自 `traj_dset.py` 函数默认值而不是 config 字段；这是复现证据增强，不是首个 sample 前的 blocker。

## Normalization / frame contract

`wall_dset.py:53–69` 在 `normalize_action=True` 时从 raw `actions` 与 raw `states` 计算 stats，然后只把 `actions` 与 `proprios` 写回 normalized tensor；`states` 保持 raw。`get_frames`（`wall_dset.py:81–94`）对 image 做一次 `/255`，再调用 `default_transform`；`img_transforms.py:3–10` 执行 Resize、CenterCrop 和 `[0.5]/[0.5]` Normalize。它返回 normalized `proprio`、normalized primitive actions 和 transformed visual。

`prepare_teacher_bias.py:361–376` 直接使用 `base.get_frames(...)` 返回的 visual/proprio，并从已经 normalized 的 `base.actions` 取 actions 0..24；没有再次调用 action/proprio normalization，也没有再次调用 image transform。`teacher_bias_screen.py` 对 manifest 中的 `dataset_float_chw` visual 直接转 tensor，不进入 uint8 transform branch；因此当前 producer/consumer 链路没有二次 normalization。`state_0_5_25` 仍是 raw state，且 screen 不把它当 proprio 输入；该 distinction 应保持在 manifest 说明中。

固定 frame `[0,5,25]`、action `[0..24]` 的边界检查和 shape checks（`prepare_teacher_bias.py:350–376`）与 protocol 的 H1/H5 定义相符；H5 的五个连续 5-step action blocks 是 recorded actions，不是 model-generated target。

## 可直接执行的修正清单

1. **先修 environment adapter gate：** 用当前 approved multiline `env/__init__.py` 的 canonical text 或 pinned hash；保留只允许两个 adapter paths 的限制。否则 CPU prep 在 source identity 阶段确定性停止。
2. **收紧 valid mapping：** 删除 `indices=None → range(...)` 和 `dataset` 缺失时的 self-fallback；要求 `TrajSubset.indices` 与 unsliced base dataset，记录 `random_seed=42`、local→source mapping 与 source IDs。
3. **保留现有 split/frame/normalization 路径：** 使用 checkpoint `env.dataset` 的 `random/0.9`，从 loader 返回的 unsliced `traj_dset['valid']` 映射，不读取 `TrajSlicerDataset` permutation；使用 `get_frames` 的一次 image transform 与一次 proprio/action normalization。
4. 修复后只做静态检查；真实 dataset construction、hash、NPZ 写出仍须在真实 SLURM CPU allocation 内执行。不要在本机用数据或 checkpoint“验证”上述结论。

本次没有发现其它会在首个 sample 前必然失败的 split 或 normalization 接口；若 environment adapter gate 与 mapping fail-closed 未修复，状态应保持 `resource_blocked`，不能把未生成的六条样本记作 `teacher-bias` scientific result。
