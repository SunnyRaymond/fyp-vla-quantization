# `verify_gauge.py` 独立静态审查

审查对象是 `verify_gauge.py` 当前版本，并以 `FREEZE_AMENDMENT.zh.md` 优先、`PROTOCOL_DRAFT.zh.md` 为补充进行逐项对照。本次只读代码和协议；未读取 64807 的 science 输出，未运行 verifier、模型、数值实验或集群任务。

## 结论

科学计算部分已经与冻结定义对齐，可以正确重算候选的冻结指标。主要轴为 `member_q[state, arm, candidate, member] = (8,4,64,5)`；`error = q - q[:, :1]` 以 FP-original 为参照；10 个 `itertools.combinations(range(5), 2)` 全部进入 pair average；`A.mean(axis=(2,3))` 正确沿 candidate 和 pair 聚合，得到每个 state、arm 一个 `A`。因此 `A[:,2]` 是 W4-RTN-original，`A[:,3]` 是 W4-RTN-centered，符合 amendment 的 `A_original/A_centered` 定义（17–50 行）。

centered weight/bias 的定义、zero-row 的 `scale=1/code=0`、per-output-row RTN grid、pooled common-mode ratio

```text
101 * sum_member,input(mean_bin(W)^2) / sum_member,bin,input(W^2)
```

以及 `ratio > 1e-8` 联合 scale/code 变化的 binding 条件均正确（30–50、101–119 行）。No-op 只比较 FP-centered 与 FP-original 的 probabilities 和 decoded member Q，tolerance 也与冻结文件一致。最终 decision 先处理 no-op、无 gauge、degenerate，再应用 median gain、6/8 严格改善和总体 mean 下降，且不会绕过 `global_binding`（24–54 行）。

## 必须在称为“独立 verifier”前修复的证据缺口

1. **冻结协议和 helper 只有自洽 hash，没有外部 pin。** 182–186 行只检查 freeze/draft/helper 位于 job output 且等于 engineering.json 自己记录的 hash；没有与审查过的冻结文件 hash 或预先冻结的 helper hash 比较。因而 producer 可以替换协议或 helper，再让 engineering.json 与副本自洽，verifier 仍可接受。应把最终 `FREEZE_AMENDMENT`、`PROTOCOL_DRAFT` 以及 job helper 的批准 SHA256 作为 verifier 常量或受信任输入比较；只检查 self-hash 不足以证明执行了该协议。

2. **checkpoint/source identity 主要是 JSON 间接互相认证。** 147–150、165–169 行重新 hash 了 parent manifest、manifest、observations、config 和 `common_init`，但没有对 `checkpoint_identity.path` 实际 hash/size，也没有逐一对 `source.selected_files` 的实际文件 hash。`source_file_identities` 只将 producer 的 `source` 字典与 pinned parent 字典比较；producer 可以复制 parent 的记录而实际 source 文件不一致。最小修复是：在 allocation guard 通过后重新读取并核对 pinned checkpoint 的 size/SHA256；逐个 hash parent 选定 source file，并把 runtime actual import 的 path/hash 绑定到对应 selected-file entry。当前 `common_init_actual_import` 只验证“在 source root 下且等于自身记录 hash”，没有绑定到 selected-files 中的明确 `common/init` 条目（168 行）。

3. **V100 和 FP32/no-grad runtime 只检查 producer 的布尔/字符串声明。** 161、164 行没有验证 `grad_enabled_for_screen is False`、模型参数 dtype 为 FP32、load 前后 device 为 `cuda:0`，也没有检查 `gpu_identity` 的实际 V100 name、compute capability `[7,0]` 和至少 30 GB 显存。runner 本身可能已执行这些检查，但 verifier 的 engineering gate 没有复核，修改 engineering.json 即可伪造通过。应至少核对这些已由 runner 记录的字段；不能只接受 `verified_v100 is True`。同理，160 行的 strict-load 检查应要求 official `api_model_conversion`，并核对 conversion/payload 记录，而不只要求 missing/unexpected 为空。

4. **storage-alias 检查没有核对 alias 行属于目标 head。** 180 行只要求每个 transaction 有两行，且每行的两个布尔字段为 true；没有检查 suffix 必须恰为 `weight`、`bias`，也没有检查 `live_name`、`detach_name`、`target_name` 对应 `_Qs.params.2.*`、`_detach_Qs_params.2.*`、`_target_Qs_params.2.*`，或检查两行不重复。176–177 行只约束 transaction name 集合，不能补足 alias 行身份。应对每 arm 先构造这两个允许的 alias row 集合，再比较完整记录；否则一份结构错误但布尔值自洽的 evidence 可通过工程 gate。

这些问题不改变当前 science 公式，但会使最终 `decision` 不能被称为独立的 implementation/provenance 结论；在修复前，建议把结果标记为 `implementation_inconclusive`，或由外部 controller 完成上述核验后再使用 verifier decision。

## 非阻断但应记录的边界

- 93–95 行只检查 shape 和 finite，没有强制 raw arrays 的 FP32、`quant_codes` 的 int8 等 dtype。由于 `np.array_equal` 对数值相同但 dtype 不同的数组可能通过，建议按 raw contract 加 dtype gate，以保持 CPU replay 与 producer 的 FP32 算术语义。
- 158 行检查了无 model/checkpoint load、无 env step/render，但没有检查 preparation metadata 的 `reset_calls == 8`。manifest seed list 已固定为 5209–5216；补上 reset-call 检查可封闭“8 个 observations 但并非 8 次 fresh reset”的 provenance 空洞。
- `effect`（50 行）没有把 `global_binding` 放入布尔式；当前 decision 在 52 行先拦截无 binding，所以不会绕过冻结 gate，但 report 可能同时显示 `effect=true` 与 `global_binding=false`，应改名为 `raw_effect_gate` 或在输出中明确其不代表可判定的 preliminary go。
- CPU softmax/decode 是合理的独立复算；其 consistency tolerance（128、133 行）与冻结 no-op tolerance 不同，前者是 CPU/GPU replay 的 engineering tolerance，后者才是 science no-op gate。应在报告字段中明确两者用途，避免把 CPU kernel roundoff 误读为 FP-centered 差异。

## 验收判定

在数学 gate、维度和指标定义方面：**通过静态审查**。在独立 provenance/engineering 证据方面：**当前有 blocker**，首要是 protocol/helper 与 checkpoint/source 的 self-attested identity、V100/no-grad/dtype 字段未独立核对、alias row 身份未核对。若 root 的外部提交控制器在提交前固定并核验这些 identity，verifier 的 science 部分可继续使用；本文件不建议通过增加 seeds、读取结果或重跑模型来补救。
