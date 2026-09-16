# Teacher-bias GPU implementation audit

日期：2026-09-13。范围仅为静态核对 `teacher_bias_screen.py`、`teacher_bias_gpu.sh`、冻结 `PROTOCOL.zh.md`、CPU preparation schema 和官方 `visual_world_model.py`；没有连接集群、加载模型或读取真实数组。

## 判定

当前为 **implementation_blocked_pending_fixes**。raw manifest/schema 已与当前 CPU preparation 对齐，但存在一个确定的 target-proprio 语义错误，以及 source/config identity 证据不足。修复前不应提交 GPU job。

## 阻断项

1. 先前将 `pred_proprio` 误判为 shape blocker，现予以更正：`proprio_encoder` 将 raw4 编成 model embedding10，且官方 `separate_emb()` 在 `model.proprio_dim=10、num_proprio_repeat=1` 时返回10；因此 `teacher_bias_screen.py:415,455–456` 的 `pred_proprio (6,3,2,10)` 是正确的。实际语义问题是 `:432` 把 raw normalized `sample["proprio"][1:,:]` 写进 `target_proprio`，而 prediction 是 encoded10，二者不能直接做 proprio MSE。最小修复是另存 `raw_target_proprio (6,2,4)`，并把同一次 `encoded_target["proprio"]` 的10D结果写入 `target_proprio`。

2. `teacher_bias_screen.py:232–249` 的 `_source_identity()` 从 `_checkpoint_identity()` 取得硬编码 source/DINO commit，只对当前 source 文件做记录性 hash，没有执行实际 source identity 校验，也没有把当前 source/config hash 与 CPU manifest 的记录逐项比较。若 CPU preparation 后 source 或 `hydra.yaml` 被改变，runner 仍可能把运行结果标为冻结 identity。最小修复应比较实际 Git HEAD，并沿用 preparation 已定义的两处 known tracked patch allowlist（不能粗暴要求 whole-tree clean），再比较 manifest 中列出的 source file hashes、额外 sourcefiles 和 checkpoint config SHA；checkpoint SHA 已有实际校验（:229–231）。

3. `teacher_bias_screen.py:396–398` 使用 `torch.equal` 检查 initial FP visual，而 protocol 规定 `allclose(atol=1e-6, rtol=0)`。同一 eval encoder 通常会相等，但该更严格条件可能把数值上合格的 GPU 重复编码误判为失败；应改为 protocol 的 `allclose`，并记录最大绝对差。

## 已核对通过的接口

- 当前 `INPUT_SCHEMA`/`SAMPLE_SCHEMA` 与 preparation manifest/sample metadata 一致；屏幕读取 `visual_0_5_25`、`proprio_0_5_25`、`model_actions_h5`，对 dataset-float CHW 直接转 tensor，没有第二次 preprocessor。visual target 通过同一 `model.encode_obs()` 的 frozen encoder 得到。
- `_cached_rollout()`（:275–285）与官方 `rollout()` 语义一致：action 0 已嵌入 `z0`，随后对 action 1–4 做四次 `predict/replace_actions_from_z`，再做第五次 predict；因此输出索引 1、5 正确对应 H1/H5。首条样本还比较 official/cached full embedding（:399–404）。
- `_runtime_structure()`（:194–224）要求 24 个 predictor Linear；每个 W4/W8 arm 都从 pristine snapshot 开始，writeback 后恢复并逐权重检查（:433–460）。raw visual 目标 `(6,2,P,384)`、预测 `(6,3,2,P,384)` 与 verifier 合同一致。
- GPU wrapper 的 V100、single GPU、24G、5-minute 和内部 240-second 限制明确；allocation guard 在模型工作前运行，未见 login-node 入口。

## 预算与证据边界

`:439` 在每个 sample/arm 重写包含 FP、codes、scale、dequant 的 `quant_params.pt`；这会反复写入数百 MB，可能侵蚀 240-second budget。建议只在每个 quantized arm 首次完成后保存一次，或保留 CPU verifier 所需 compact records；若暂不改，应把它作为 budget risk 记录。结果仍只能支持 fixed recorded-future encoder-feature diagnostic，不支持 physical truth、control success、泛化或 native low-bit deployment。
