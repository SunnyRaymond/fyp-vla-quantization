# TD-MPC2 cartpole checkpoint/source compatibility audit

**日期：2026-09-13**  
**范围：只读 source、checkpoint metadata 与 64793 failure；未加载模型、未运行数值、未连接或提交 cluster。**  
**当前 gate：`resource_blocked`（seed 1 已证实与 pinned source 不兼容；seed 2/3 仅是待做 schema probe，尚未认证）。**

## 1. 64793 的事实

GPU allocation `64793` 的 `summary.json` 记录了官方
`cartpole-balance-1.pt` 在 pinned TD-MPC2 source
`e9f59321933cbc8e11a002b842adc7d4ffae8ff1` 上 strict `load_state_dict`
失败。失败同时包含三类证据：

- current model 需要 `_encoder.state.0/.1`、`_dynamics.0/.1/.2`、`_reward.0/.1/.2` 与 `_pi.0/.1/.2` 的 `NormedLinear` 风格 keys；
- checkpoint 出现 `_encoder.state.2/.4/.5`、嵌套的 `_dynamics.0.0/.1/.3/.4/.6`，以及 `_reward.3/.4/.6`、`_pi.3/.4/.6`；
- 还有无法由 key rename 修复的 shape mismatch：encoder 的 `[256,5]` 对 `[512,256]`，以及 dynamics/reward/pi 中 `[512]` 对 `[512,512]`。

因此这是 source/checkpoint 结构不兼容造成的 engineering failure，不是 Q coupling 的科学 no-go。`strict=False`、删除 unexpected keys、padding/slicing 或把旧层硬套到新层，都会改变函数而不能作为兼容性修复。

## 2. 官方 current loading path 与转换范围

官方 pinned source 的调用路径是：

- [`tdmpc2.py` at pinned commit](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/tdmpc2.py) 的 `TDMPC2.load` 解包 `model` 后调用 `api_model_conversion`，再调用 `self.model.load_state_dict`。
- [`world_model.py` at pinned commit](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/world_model.py) 用 `layers.enc`、`layers.mlp` 建立 encoder/dynamics/reward/pi，并用 `layers.Ensemble` 建立 Q ensemble。
- [`layers.py` at pinned commit](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/layers.py) 的 current `NormedLinear.forward` 顺序是 `Linear -> optional Dropout -> LayerNorm -> activation`；`mlp` 为两个 hidden `NormedLinear` 加 final linear（有指定 `act` 时 final 才是 `NormedLinear`）。

`api_model_conversion` 的源码只处理旧 Q API：对 `_Qs.*` 与
`_target_Qs.*` 做 index/name conversion，补齐 TensorDict 的 batch/device
metadata，复制 `log_std_*` 和可选 `_action_masks`，最后检查 Q keys。源码没有
encoder、dynamics、reward 或 pi 的 key/shape/graph conversion，也没有记录旧层
的 activation、LayerNorm `eps`、dropout 或 operation order。因此它不能解释
64793 的非 Q mismatch。

官方 repository 的公开 issue 还记录过同形状的 single-task checkpoint
loading error：[`tdmpc2 issue #23`](https://github.com/nicklashansen/tdmpc2/issues/23)
中的 missing/unexpected keys 与本次 64793 基本相同。issue 讨论中的 maintainer
建议说明过旧 API 的映射可以让部分 checkpoint strict-load，但不能据此声称
性能可复现；后续建议是改试同任务的其他 released checkpoint。这里不把未给出
完整 graph proof 的 mapping 当成转换器。

## 3. 同任务 seed 1/2/3 的官方 metadata

HF 官方仓库的目录和 release commit 为
[`nicklashansen/tdmpc2@73a50e2719ed8258c72c7d1fefd23b781d66e35e`](https://huggingface.co/nicklashansen/tdmpc2/commit/73a50e2719ed8258c72c7d1fefd23b781d66e35e)。该 commit 的每个 LFS pointer 都给出相同的 remote size，但 OID 不同：

| 文件 | HF 文件页 | remote size | LFS SHA256/OID | 当前可知的 version 信息 |
|---|---|---:|---|---|
| `cartpole-balance-1.pt` | [official file page](https://huggingface.co/nicklashansen/tdmpc2/blob/main/dmcontrol/cartpole-balance-1.pt) | 31,344,610 B | `4919e562d7f22f41a11118d1db1a0ebcb0e2b4681fc7fc594d772f0d5940869b` | file revision `73a50e2…`；64793 strict-load failed |
| `cartpole-balance-2.pt` | [official file page](https://huggingface.co/nicklashansen/tdmpc2/blob/main/dmcontrol/cartpole-balance-2.pt) | 31,344,610 B | `5e8e6194fc5e1502da73febd2b5af119d6c4805c97cabb5a0f17c3f235bed587` | file revision `73a50e2…`；schema untested |
| `cartpole-balance-3.pt` | [official file page](https://huggingface.co/nicklashansen/tdmpc2/blob/main/dmcontrol/cartpole-balance-3.pt) | 31,344,610 B | `0e2c0eada8f160dd65c8faaa5e4ca2f8f496104e21433e334d716b73257a52c2` | file revision `73a50e2…`；schema untested |

官方 [model card](https://huggingface.co/nicklashansen/tdmpc2) 只说明
single-task checkpoints 使用官方 implementation 的 default hyperparameters、
均为 5M parameters；官方 [README evaluation instructions](https://github.com/nicklashansen/tdmpc2#evaluation)
只说明 single-task 使用 `model_size=5`。这些信息没有给出训练 source commit、
旧 `layers.py`、每个文件的 state-dict schema，不能从“同 size、同 task、同
upload commit”推出 seed 2/3 一定能被当前 source 加载。

因此可以预知的 checkpoint version 只有 HF file revision、LFS size 和 OID；
无法预知与 `e9f593…` 的 source compatibility。seed 1 的 failure 也不能证明
seed 2/3 必然 failure 或 success。

## 4. 推荐的最小选择路径

在任何 Q score、rounding arm 或 planner metric 之前，root 可在真实 CPU
allocation 中按 seed ID 先后做两个 31 MB schema probes：

1. 对 `cartpole-balance-2.pt` 使用和 64793 完全相同的 pinned source、resolved
   state config、`model_size=5` 与 strict loader；记录 HF revision/OID、source
   commit、完整 missing/unexpected/shape report、resolved module graph 和
   `api_model_conversion` 是否只触碰 Q keys。
2. 对 `cartpole-balance-3.pt` 重复同一 probe，不改变 config 或 loader。两个
   probe 都保留 independent manifest；选通过 strict-load 且所有 key/shape
   通过的最低 seed ID。`cartpole-balance-1.pt` 永久标记为
   `incompatible_with_e9f593`，不因 seed 2/3 的结果而覆盖其记录。

“通过”必须是完整 `load_state_dict` 且无 missing/unexpected/shape mismatch；
`strict=False`、手工 key map、忽略非 Q modules、补零/截断权重均算失败。若
   seed 2/3 均失败，结论是 `resource_blocked`：当前 official release 与
   pinned source 没有可证实的 compatible asset，停止该 screen，等待官方
   matching source commit 或完整转换工具。

只有至少一个 seed strict-load 通过后，才可另行 amend Q-coupling protocol，
并在 GPU 前锁定：

- exact HF file revision、LFS OID 与 size；
- exact source commit 与实际 `layers.py/world_model.py/tdmpc2.py` hashes；
- resolved `model_size=5`、observation/action dimensions、`num_q=5` 与 Q
  return path；
- no-op output、完整 weight restore，以及非 Q module digest 不变。

即使 probe 通过，也只证明该 seed/source pair 可加载；没有额外的 operation-
order/activation/LayerNorm-eps 对照，不能把任何 mapping 写成“旧模型与 current
model exact equivalent”。

## 5. 结论与交接边界

- `cartpole-balance-1.pt` 的 64793 failure 是结构性 source/asset mismatch，
  不是 coupling hypothesis 的 empirical result；不要继续使用 64793 raw
  scores 做科学判断。
- `cartpole-balance-2.pt` 和 `-3.pt` 的 metadata 已核对：两者与 seed 1
  相同 size、同 HF release revision、不同 OID；兼容性未认证。建议按 2→3
  做 strict schema probe，选择最低通过 seed，或在两者失败时保留
  `resource_blocked`。
- 本审计未下载 checkpoint、未加载模型、未运行数值、未修改
  `tdq_prepare`/screen 文件，也未读取 credentials。
