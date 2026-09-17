# VLA / WAM asset map（只读审查）

日期：2026-09-13  
范围：本机 `reproduction/openvla-oft-vla-eval-libero`、`reproduction/fastwam-smoke` 的小型 source、README、control 与已保存 artifacts。

本审查没有连接 ASPIRE2A/CCDS，没有下载、传输、安装依赖，也没有加载模型或运行数值测试。目录检查是定向的；没有递归扫描大型 cache。下文把本机文件、历史 PBS artifact 和未验证的远端路径分开。定向搜索没有找到 CCDS 专属的 OpenVLA/FastWAM run artifact；已保存的运行证据是 ASPIRE2A PBS 记录，不能当作 CCDS 当前资产证明。

## 资产结论

| 项目 | 本机可见 | 已保存运行证据 | 当前边界 |
|---|---|---|---|
| OpenVLA-OFT Goal | 完整 4-shard checkpoint、`action_head`、`proprio_projector`、stats、model code；`source/openvla-oft` revision `e4287e94541f459edc4feabc4e181f537cd569a8` | `artifacts/libero-goal-full-summary.json`：500 episodes、491 successes、两份 2×A100 PBS job | Goal pipeline 有历史成功记录；没有本机 LIBERO SIF/benchmark data，也没有 V100 记录 |
| OpenVLA-OFT Spatial | 同样完整的 4-shard checkpoint、head、projector、stats | 没有找到与当前 Spatial smoke README 相匹配的完成 artifact；旧 README 仍写 pending | checkpoint 身份可读，Spatial environment/run 需重新在 allocation 内核实 |
| FastWAM Optional-IDM | `FastWAM` source revision `7faa71108368fbb3b6885649f112af607427a2d4`；本机没有 `checkpoints/`、`data/`、`cache/`、`containers/` | `artifacts/16180074.pbs101`：A100、OSMesa CPU render、`first_frame`、LIBERO-goal task 0，1/1 成功；`reproduction/otc-ptq-phase1/RESULT.zh.md:7-14,39-43` 另有 IDM paired replay / single-site W4 fake-quant 功能验证 | checkpoint、stats、LIBERO assets 和 Wan components 只在结果文件记录为远端 scratch 资产；没有完整 IDM success benchmark，且本机完成的成功 rollout 仍只有 `first_frame` |
| SmolVLA LIBERO | 本机没有副本，本审查不下载 | parent 提供的公共指针：`https://huggingface.co/lerobot/smolvla_libero`，main `31d453f`、约 907 MB、7-action padded 32、state 6、3-camera、chunk 50、flow 10 steps | model card 是 generic，未见 benchmark success 证据；仅作轻量备选资产，不作本地已拥有声明 |

## OpenVLA-OFT 的精确证据

本机 Goal checkpoint 目录为 `reproduction/openvla-oft-vla-eval-libero/source/checkpoints/openvla-7b-oft-finetuned-libero-goal/`，含四个 safetensors shard（总大小由 `model.safetensors.index.json` 记录为 15,082,474,368 bytes）、`action_head--50000_checkpoint.pt`、`proprio_projector--50000_checkpoint.pt`、`dataset_statistics.json` 与自包含的 `configuration_prismatic.py` / `modeling_prismatic.py`。Spatial 目录对应 `150000` head/projector。两个 stats 文件都是 7-action、8-proprio；Goal 的 `libero_goal_no_noops` 记录 52,042 transitions/428 trajectories（Goal stats line 130–131），Spatial 记录 52,970/432（Spatial stats line 130–131）。这些是 normalization statistics，不是可替代 LIBERO environment data。

模型结构和默认精度可由 Goal `config.json:3-19,3153,3164` 看到：`OpenVLAForActionPrediction`、`llama2-7b-pure`、`n_action_bins=256`，`torch_dtype=bfloat16`。本机还保留 pinned external source：`source/openvla-oft/.git` 的 HEAD 是 `e4287e94541f459edc4feabc4e181f537cd569a8`；其 `experiments/robot/openvla_utils.py:253-308` 明确加载 VLA，`action_heads.py:84-107` 实现 L1 continuous head，`openvla_utils.py:464-517` 加载 suite-specific head。

历史 Goal 成功证据是 `artifacts/libero-goal-full-summary.json:2-55`，对应 checkpoint revision `c2d0f9fbbd82674683b397ff923168a12f6a307b`、vla-eval commit `4aeb4369640e8019d46af9534ce9b957e486ad38`、500 个 unique task/episode pairs 和 500 个 nonempty MP4。`artifacts/16137211.pbs101/nvidia-smi.txt` 明确是两张 A100-SXM4-40GB；`final-qstat.txt` 记录每个 server 约 15,950 MB 最大 GPU memory。它证明历史 A100 pipeline 可用，不能外推 V100 或 BF16。

`control/README.md:7-33` 是旧 Spatial smoke 说明（1×A100、pending），因此标为 stale；完成 Goal summary 与对应 artifacts 优先级更高。完整 Goal job 的环境入口和远端布局见 `control/libero_goal_full_4gpu_16137185.pbs:11-23,33-45,81-127`：它依赖远端 `containers/libero-latest.sif`、scratch cache 和 checkpoint。该 SIF 与 LIBERO benchmark data 没有在本机 source 树中确认。

## FastWAM 的精确证据

官方 README `reproduction/fastwam-smoke/FastWAM/README.md:77-137` 说明 Optional-IDM 是同一 checkpoint 的两条推理路径：`idm` 先想象 future video，`first_frame` 直接从当前 observation 预测 action；README 的完整 LIBERO success table 是作者报告，不能当本地 reproduction。当前本机 source 的 `configs/model/fastwam_optional_idm.yaml:1-58` 固定 Wan2.2-TI2V-5B、两个 30-layer expert、action dim 来自 data processor，action scheduler shift 为 1.0。

本机 data config `configs/data/libero_2cam.yaml:9-43` 给出两路 512→224 image、7-action（EEF pose 6 + gripper 1）、8-state（EEF pose 6 + gripper 2）、33 frames、32 action positions 和 9 video frames。`RESULT.md:3-14,33-37` 与 `HANDOFF.md:66-68,95-110` 记录了远端 clean checkpoint、stats、A100/OSMesa smoke 和验证边界；本机只保留 `artifacts/16180074.pbs101/` 的日志、MP4、结果 JSON、preflight 和 PBS snapshot，不能由此声称 checkpoint/data/env 仍在 CCDS。

该 smoke 的直接控制脚本是 `artifacts/16180074.pbs101/osmesa_smoke.pbs:1-69`：它依赖远端 SIF、`$ROOT/checkpoints/...clean.VERIFIED`、`$ROOT/libero` compatibility tree 与 `$ROOT/model-cache`，并在 allocation 内验证 CUDA UUID。`eval.log:7-18` 显示模型组件和 clean checkpoint 已加载；`gpu0_task0_results.json` 是 1/1 success。历史参数是 `action_infer_mode=first_frame`、`sigma_shift=1.0`、`compile_action_infer=false`；IDM 有后述 paired replay 功能证据，但没有完成成功率 rollout artifact。

另有 `reproduction/otc-ptq-phase1/RESULT.zh.md:7-14,39-43` 的 bounded IDM paired replay：同一 Optional-IDM checkpoint 的 `idm` mode 在两个 LIBERO-Goal tasks、32 states 上完成 reference/repeat/identity 与 functional replay 检查，并包含 single-site W4 fake-RTN diagnostic。它证明 IDM action path 的有限功能/恢复一致性，不能升级为完整 IDM benchmark success，也不改变上面只有 `first_frame` 完成成功 rollout 的边界。

## V100 FP16 条件

NVIDIA 的 TensorRT Support Matrix 将 V100 列为 compute capability 7.0，支持 FP16/FP16 Tensor Cores，不支持 BF16：[NVIDIA TensorRT Support Matrix](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-843/pdf/TensorRT-Support-Matrix-Guide.pdf)。因此后续 V100 screen 必须把实际 device name、compute capability、实际 model dtype 和显存写入 artifact。

- **OpenVLA-OFT：当前代码不是 V100-ready FP16。** checkpoint config 是 BF16；更关键的是 `source/openvla-oft/experiments/robot/openvla_utils.py:282-303,492-493,760-766` 将 VLA、action head、输入显式放为 `torch.bfloat16`，而 `vla-evaluation-harness/.../oft.py:58-123` 没有 dtype override。V100 screen 需要先在 compute allocation 中做最小的 FP16 loader adaptation/compatibility probe；不能把 A100 success 或 15.95 GB memory 记录当成 V100 证明。若走 4/8-bit，也要独立记录实际 dtype/path。
- **FastWAM：代码有显式 FP16 路径。** `FastWAM/src/fastwam/runtime.py:22-40` 把 `mixed_precision=fp16` 映射为 `torch.float16`，`467-486` 将其传入 model；但 `configs/train.yaml:28` 默认是 `bf16`，所以 V100 job 必须显式覆盖为 `fp16`。第一次 screen 应保持 `compile_action_infer=false`，并在 allocation 内验证 device/dtype；历史 smoke 是 A100，未验证 V100。

## 最小可操作 action interface

| 模型 | action head / logits | 原始输出 | chunk / execution | gripper |
|---|---|---|---|---|
| OpenVLA-OFT | 默认 OFT server 加载 `L1RegressionActionHead`；`LIBERO.md:92-105` 与 `run_libero_eval.py:91-94` 明确该 OFT 配置是 L1 continuous objective。source 仍保留 discrete path：`modeling_prismatic.py:920-942` 对 LM logits 做 argmax，但 source available 不等于该 L1-only checkpoint 保留了可比较、可靠的 discrete action-token training target；当前 server API 也不返回 logits | continuous action array，LIBERO 为 8×7；`oft.py:169-181` 转成 float32 后返回 `{"actions": ...}` | `constants.py:26-30` 为 8×7；完整 job 与 server config 使用 `chunk_size=8`。`predict.py:226-257` trim/push/pop，`chunking.py:43-89` 支持 `newest`、`average`、`ema` | 最后一维；`oft.py:178-181` 把 RLDS `[0 close,1 open]` 映射到 robosuite `[-1 open,+1 close]` |
| FastWAM Optional-IDM | action DiT/action expert；`FastWAMOptionalIDM.infer_action` 支持 `idm` / `first_frame`，不暴露 logits | `fastwam.py:1157-1158` 返回 CPU FP32 `{"action": [T,D]}`；single eval `eval_libero_single.py:434-448` denormalize 后执行 | 默认 `action_horizon=data.train.num_frames-1=32`；单评测按 `replan_steps` 取前段，`ActionEnsembler` 可对重叠 chunk 做 timestamp mean（`eval_libero_single.py:478-535`、`action_ensembler.py:5-32`） | `eval_libero_single.py:442-447` 翻转并可 binarize 最后一维；data config 明确 6 pose + 1 gripper |

因此两者都能以 action-only chunk 作为最小 screen 输入/输出；FastWAM 没有 logits API，OpenVLA 的 logits 需要绕过默认 continuous head 并在外部 source 中加 hook，而且其 discrete path 的训练目标/可靠性尚未确认。

## 两个可独立提出的切面（尚未定协议、尚未实验）

1. **Optional-IDM branch-conditioned action sensitivity。** 利用同一 FastWAM Optional-IDM checkpoint 的 `idm` 与 `first_frame` 双路径，研究 future-imagination branch 是否改变 action expert 对同一输入扰动的放大方式；观察对象是 raw action chunk、首个 replan action 与 action/video branch 的差异。它使用现成 mode switch 和 7-D chunk，切面在 branch interaction，区别于普通 loss、scale 或 solver 调参。已有 paired replay 可作为功能边界；完整 benchmark success 仍需在真实 allocation 内另行验证。
2. **OpenVLA continuous-head vs discrete-logit objective mismatch。** source 同时实现 L1 continuous head 和 `language_model_output.logits` 的 discrete action-token 路径，但 OFT checkpoint 的公开配置与 fine-tuning command 是 L1-only；因此重点应先界定 discrete path 是 structural fallback 还是有训练支持的 representation，再讨论相同 observation 上的 chunk disagreement、token confidence 与 gripper 输出。默认 vla-eval server 只走 L1 head，故该切面目前是 source-available、training-objective-unverified，不能把两条路径当等质量基准，也不是新 loss 或 calibration scale 提议。

## 使用边界

- 本机已确认的是 source/checkpoint 文件身份与历史 artifact 文件；远端 scratch 记录、A100 记录和作者 README 各自只支持其所写范围。
- 本机没有 V100 FP16 runtime evidence，也没有 CCDS-specific OpenVLA/FastWAM evidence。所有未来模型加载、环境访问、重 I/O 和计算必须由真实 SLURM/PBS allocation 完成；head/login 只做轻量控制。
- 这份 map 不提出实验 protocol、不下载 SmolVLA、不宣称任何新结果。
