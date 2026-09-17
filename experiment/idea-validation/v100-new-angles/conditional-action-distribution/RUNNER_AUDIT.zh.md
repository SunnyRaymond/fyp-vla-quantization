# `marginal_screen.py` runner implementation审查

日期：2026-09-13。只读核对当前 runner 与已冻结 `PROTOCOL.zh.md`；未读取科学输出、未连接集群、未运行数值，也未修改 Python。producer 后续字段不在本次审查范围。

## 结论

raw order、noise pairing、reset/clone、两臂切换和 expert-only W4 的实现逻辑静态一致；没有发现会把四臂/错误 horizon 写入 raw 的科学性 bug。GPU64778 若工程产物完整，可按 CPU verifier 继续审查。仍有一个需要保留的 extended-identity provenance 风险，以及一个小的输入 shape 防护缺口；它们不改变当前 metric，但在复现声明中不能隐去。

## 已核对的执行语义

1. `_load_input_manifest`（128–227）接受 `conditional_marginal_ready` 的 extended identity，因为其仍是 pinned identity schema；它验证 identity 文件 hash、SmolVLA/LeRobot revisions、state8/action7、两路 camera、8 个 sample、task0..3 各两个和排序。sample path 受 asset-root containment 与 SHA-256 保护，因而不应回退到 WallDataset namespace。GPU runner 后续保存该 identity path/hash 与 sample hashes。
2. `_make_noises`（274–291）在 CPU 上用 torch `Generator` 的 1901/1902 各生成 32×50×32，再拼成固定 64×50×32；`_run` 只创建一次 GPU noise tensor，并在每个 draw 用 slice `.clone()`。arm 与 condition 都复用相同 noise 顺序，raw action 的 `(condition, arm, draw, horizon, dim)` 顺序由 451、480、507–513 明确写出。
3. `_repeat_batch`（294–302）只接受 batch-one tensors，沿第 0 维 repeat；`_predict_action`（305–315）每次先 `policy.reset()`，再 clone batch 和显式 noise，输出形状必须是 `(batch,50,7)`。batch4 与四次 individual 的检查同时覆盖 FP32 和 expert_W4，并使用冻结 `1e-5` 容差。没有发现隐式 `noise=None`、跨 condition cache 或 A/B coupling 泄漏。
4. `_eligible_modules` 来自 pinned Flow helper 的 expert transformer layer allowlist；`_expert_state_names`（254–271）按真实 `Parameter` identity 映射到 `state_dict` names，并要求是精确 subset。每 arm 先 restore snapshot，再仅对 `expert_modules` 执行 symmetric per-output-channel RTN W4；bypass digest 在 arm 前、batch check 后、全部 condition 后及 final restore 后核对。FP32 不量化，expert_W4 不会误扩到 backbone。量化后 expert weight 本身没有另存独立 digest，但 module list、shape、scheme 和实际 weight mutation 路径均已记录，静态上足够支持本次 two-arm screen。

## 需要保留的风险与建议

- `_load_input_manifest` 只验证 extended identity 文件本身的 hash 和 base schema，不检查其中 `bounded_extension.parent_identity_sha256` 是否指向原始 `smolvla/identity.json`，也不核对 probe/selection parent chain。CPU64777 的 extended identity 和 manifest hash 已由准备阶段固定时，这不会改变 raw input；但最终复现报告应附带 parent-hash evidence，否则只能声称“读取并 hash-bind extended identity”，不能声称 runner 独立验证了继承链。
- `_load_raw_sample`（651–654）要求 state 是 finite float32 一维向量，却没有再次要求 `shape==(8,)`。identity 已 pin state8，且 CPU preparation 产物固定；因此当前资产下不是预期科学偏差，但若要让 runner 本身 fail closed，建议把 shape8 纳入输入合同。相同地，raw sample 的 source-file hash 只通过 sample SHA 间接绑定，runner 不逐项重验 metadata 中的 source-file map；这属于 provenance 强度而非 action metric bug。
- `_state_subset_digest`（798–808）每次遍历并 CPU-copy 所有非 expert state，arm 切换期间会重复多次。它不影响数值定义，但可能消耗 14-minute deadline 的可观时间；若超时，必须保持 `resource_blocked`，不能删掉 digest 或静默减少 draws。

## 最终判断

在不把 extended-identity parent chain 冒称为 runner 已独立验证、并确保 producer 完成 raw draw-order binding 后，runner 可执行且无需扩展 protocol。若只出现 parent-chain provenance 缺失，应降级复现工程声明；若 state shape、batch gate、W4 bypass 或完整 1024 samples 失败，应分别记 implementation/resource failure，不作科学 no-go。没有发现需要停止 GPU64778 的新的 raw-order、reset/noise 或 expert-state binding 阻断。
