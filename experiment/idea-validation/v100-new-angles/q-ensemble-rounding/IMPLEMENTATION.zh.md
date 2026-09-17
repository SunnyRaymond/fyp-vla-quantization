# TD-MPC2 Q coupling screen implementation

本交付只包含静态 runner 和 SLURM wrapper；当前 turn 未运行模型、数值实验或作业。`tdq_screen.sh` 申请单张 V100、4 CPU、24 GB、15 分钟，并先 source `$HOME/v100_newangles_ccds/control/allocation_guard.sh`。Python `main()` 的第一项工作是 `allocation_guard.require_allocation()`；随后才读取 manifest、hash source/checkpoint 或 import `numpy`/`torch`。输出固定为 `$TOP/artifacts/$SLURM_JOB_ID`，`run.log` 由该目录的 `tee` 持续记录。

runner 使用 `tdq_prepare.py` 产生的 pinned manifest，默认资产目录为 `tdmpc2_q_coupling_ready`，并以环境变量 `TDQ_ASSET_ROOT`、`TDQ_MANIFEST`、`TDQ_SOURCE_ROOT`、`TDQ_CHECKPOINT`、`TDQ_CONFIG` 覆盖路径。`PYTHONPATH` 顺序允许 `tdmpc2_q_coupling_runtime_extra2`、runtime vendor、官方 source 的 `tdmpc2` sibling package 和 control 共存；不下载或安装依赖。

模型路径直接构造官方 `common.world_model.WorldModel`，先 `.to(cuda:0)` 再 `torch.load(..., weights_only=False)`。checkpoint 经过官方 `common.layers.api_model_conversion` 后 strict 完整加载，记录 old/new conversion mode、module identities、resolved config、torch/tensordict/omegaconf 版本；不剥离未知 key，也不做 partial load。source 与 checkpoint identity 在 model load 前落盘，GPU identity 要求真实 V100、compute capability `(7,0)` 且显存至少 30 GB。

`_Qs`、`_detach_Qs_params`、`_target_Qs_params` 先按相同 suffix 建立一对一关系。只有实际三组 live Linear weight（shape `[5,out,in]`）以及对应 detach alias 进入 transaction；target 和其他 Q bias/LayerNorm 等都留在 bypass digest 中。`state_dict` 允许的非 tensor 只限 TensorDict 的 `__batch_size` / `__device` metadata，并以 canonical signature 做严格 restore/digest；alias storage 检查只对真实 tensor 执行。

先用一次 FP32 cache 生成固定 candidate actions、H3 terminal latents 和每个 reset state 的 FP policy terminal action，四个 arms 和三个 rounding seeds 全部复用。FP32 与 RTN 各跑一次后复制到三个 seed slot；independent SR 与 stratified SR 各用一个显式 CUDA `Generator` 连续处理按字典序排列的三个 Linear。所有 transaction 在 arm 边界 restore 完整 snapshot，并检查 target/bypass digest；完整 quantizer records（scale、权重 hash、uniform/permutation hash、weight MSE）同步写入 `quantizer_transactions.json`。

Q 输出固定调用官方 `Q(return_type="all")`，用官方 `two_hot_inv` 解码五个 scalar member。固定 pair seed 的 `Q(return_type="avg")` 对照会先 squeeze 官方 `[N,1]` 输出，再按 `atol=1e-5, rtol=1e-6` 对照 decoded pair mean，并保存 CUDA RNG state、pair、absolute/relative error。raw 文件保留全部 member、official API 输出、inputs、terminal cache、completed 矩阵和 seed；科学 verifier 使用 10 个 unordered pair 的 finite expectation，runner 不调用 planner、environment、`step` 或任务 success gate。

实现对实际 pinned source API 保持 fail-closed：若 `WorldModel` state key、TensorDict metadata、checkpoint conversion、Q logits shape、V100 条件或 official avg 语义与合同不一致，输出 implementation failure/inconclusive 并保留工程证据，不猜 key、不替代模型、不把静态检查当作 runtime pass。科学结论和 gate 留给独立 `verify_tdq.py`。
# Confirmed engineering correction after GPU 64797

CPU 64798 proves TensorDict 0.7.2 `state_dict()` produces separate serialized storage; actual `model._Qs.params[key]` and detached tensors do alias, target remains distinct. No Q outputs existed at failure. Binding and quantization now use actual TensorDict parameters; serialized state remains for snapshot/digest only. Each quantizer writes actual weight then checks fresh serialized readback exactly equals dequantized values. No science thresholds changed. GPU 64797 and CPU 64798 retained.
