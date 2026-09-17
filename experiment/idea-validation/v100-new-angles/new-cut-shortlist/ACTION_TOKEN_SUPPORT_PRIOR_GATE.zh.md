# ACTION_TOKEN_SUPPORT_PRIOR_GATE

审查日期：2026-09-13。本文只做 AR action-token 路径的 source/asset gate；不连接 cluster、不加载模型、不下载 checkpoint、不运行数值。结论区分“确有可诊断的边界问题”和“是否值得作为新的 quantization research cut”。

## 结论

对 vanilla discrete OpenVLA 及其 QVLA fork，**未看到 inference-time action-vocabulary mask**，而 `token → bin → continuous action` 的 decoder 确实会把越出预期 action-token 区间的 ID 先映射再 clip 到 bin-center 端点。因此问题路径是真实的；但“加一个 support mask”是直接的 constrained-decoding 工程修复，不能单独构成新的 PTQ 方法。最终状态：`source_path_confirmed / novelty_no_go / engineering_fix_no_go / GPU=0`。

BitVLA 不适合作为本案 AR 对象：已核对的 BitVLA 路径是 BitNet/SigLIP quantization 加 continuous action head/OFT，不是 OpenVLA discrete action-token decoder，故标 `structural_no_go`。

## OpenVLA：路径已确认

本地 `reproduction/qvla-code-audit-2026-09-05/QVLA/openvla/` 与官方 OpenVLA 代码一致地实现 `ActionTokenizer`：默认把最后 `n_bins=256` 个词表位置当作 action bins，`action_token_begin_idx = V-(n_bins+1)`。`__call__` 先把 action clip 到 `[-1,1]`、`digitize` 后写成 `V-i` token；`decode_token_ids_to_actions` 再计算 `V-token_id`，并把 bin index clip 到 `[0,254]`。见官方 [action_tokenizer.py](https://github.com/openvla/openvla/blob/main/prismatic/vla/action_tokenizer.py#L34-L68)。

所以按该 Llama convention，合法 action ID 是 `V-256,...,V-1`；越界 ID 不会报错，而会被送到首/末 bin center。未经核验的 `V`、special token 和 tokenizer resize 不能直接套用 `31743` 常数，运行时必须保存 tokenizer fingerprint 与实际 support interval。

`OpenVLA.predict_action` 调用普通 `GenerationMixin.generate`，只传 `max_new_tokens=action_dim`，随后截取最后 `action_dim` 个 ID 交给 decoder；checked path 没有 `logits_processor`、`prefix_allowed_tokens_fn` 或 action-ID rejection。见官方 [openvla.py](https://github.com/openvla/openvla/blob/main/prismatic/models/vlas/openvla.py#L57-L91)。因此 quantized logits 若把某个位置的 argmax 推到 action support 外，native decoder 会静默产生 boundary action。

这不是“模型输出连续值越界”：越界发生在 discrete token support，极端 continuous action 是其 clip/decode 后果。必须记录 raw token IDs、support mask、clipped bin index 和 unnormalized action，不能只报告 action MSE。

## QVLA：该路径未被屏蔽，量化是否实际触发尚未实测

锁定的 QVLA commit `26cc4821a3be4c003d09d3c7997b38db2a347982` 复用了 OpenVLA action tokenizer；其 `inject_fake_w.py` 只对 vision/language `Linear/Conv2d` 做 weight-only fake quant，显式排除 `projector`、`action_head` 和 `language_model.lm_head`，没有增加 generation support mask。见 [QVLA injector](https://github.com/AutoLab-SAI-SJTU/QVLA/blob/26cc4821a3be4c003d09d3c7997b38db2a347982/openvla/qvla/inject_fake_w.py#L8-L108) 与本地 [代码审计](../../../reproduction/qvla-code-audit-2026-09-05/README.md)。

这给出一个可识别的窄 diagnostic：FP 与 language/backbone W4/W8 在同一 prompt/image、同一 greedy decoding 下，比较越界率和 endpoint-bin saturation；再对两者同时加严格 action-ID mask。若 PTQ 只在 native clip 下产生额外 boundary actions，且 strict mask 后消失，机制路径成立。控制必须包括 FP-native clip、FP-strict mask、PTQ-native clip、PTQ-strict mask，以及 raw logits/token receipts；不能把 masked policy 的 action success 当作原模型结果。

最强负对照是：在相同 raw logits 上只切换 native decoder 与 strict-support decoder；若 FP 也同样越界，或 PTQ 的越界率不高于 FP，说明不是 quantization-induced effect。若 strict mask 后极端动作仍存在，说明问题来自 unnormalization/robot clipping 或另一条 head，立即停止本案解释。

## BitVLA：结构不匹配

本地 BitVLA 审计确认 `BitLinear` 对 backbone 的 weight/activation fake quant 已接入，但公开 OFT training/eval 使用 continuous action head 的 L1 路径；其 action wrapper 不提供本案需要的 `token ID → 256 bins → clip`。见官方 [BitVLA bitnet_utils.py](https://github.com/ustcwhy/BitVLA/blob/8afac0260b3748b14657a69ec58e3d9f0d6da3a7/openvla-oft/experiments/robot/bitnet_utils.py) 和 [BitVLA 审计](../../../reproduction/bitvla-code-audit-2026-09-05/README.md)。因此不能用 BitVLA continuous action 越界冒充 AR support 证据。

## Novelty、已有资产与运行边界

Constrained decoding 的 support mask/rejection 是该问题的直接工程修复；即使 screen 显示 PTQ 增加了越界 token，也只能证明一个 decoder contract 被量化放大，不能称为新的 quantizer 或新的 action-allocation 方法。当前 campaign 已有 OpenVLA-OFT、QVLA/BitVLA source audits；它们不能替代 AR weights。

真正所需的 checkpoint 是 vanilla discrete `openvla/openvla-7b`，或其离散 per-suite checkpoint `openvla/openvla-7b-finetuned-libero-{spatial,object,goal,10}`；不能使用本地 `moojink/openvla-7b-oft-finetuned-libero-*`，后者是 continuous OFT head。当前 workspace 没有这些 AR weights 的已核验 revision/byte manifest，故资产状态为 `resource_blocked`。官方 README 对 Prismatic-compatible base checkpoint 给出约 30 GB 下载量；本地已核验的 OFT release 为 15,939,159,216 bytes，但模型头不同，不能据此估计 AR checkpoint。

若未来资产补齐，最小 screen 可设计为 1×V100 32 GB 做短 prompt、7 action-token greedy generation 和 W4/W8 fake-quant；V100 不支持 native BF16，必须选用受支持的 FP16/FP32 路径并在 allocation 内核验 reference dtype、tokenizer support 与 peak memory；不能直接照搬 BF16/FlashAttention2 的其他 GPU 配置，再跑 6–8 条固定 input。禁止把现有 A100 OFT 运行记录外推为 V100 AR 可行性，也不做完整 LIBERO rollout。

## 停止门

以下任一条件均 `GPU=0`：找到官方 inference action-vocab mask；只能取得 continuous OpenVLA-OFT/BitVLA checkpoint；无法绑定 tokenizer support interval；strict mask 与 native clip 的差异只是已知工程修复且不提出独立数值机制；或需要新模型下载/完整 benchmark 才能观察越界率。当前不申请资产、不建 pipeline、不把该问题加入研究主结果。

证据边界：本轮 targeted source review 不是 exhaustive literature search。`source_path_confirmed` 仅表示代码路径成立，`novelty_no_go` 表示其最直接的干预是成熟 constrained-decoding 工程修复，`resource_blocked` 表示当前没有合适 AR checkpoint；三者不能混写成“已证明 PTQ 降低任务成功率”。

Root复核：source path成立不等于已观察到PTQ越界；headline已据此改为source_path_confirmed。V100执行dtype先前BF16建议已纠正。AR资产尚未建立可运行manifest，不把metadata-only source audit当作checkpoint存在证明。此案仍GPU0/STOP。
