# Rounding-persistence GPU implementation audit

日期：2026-09-13。只读核对当前 `persistence_screen.py`、
`persistence_gpu_screen.sh`、冻结 `PROTOCOL.zh.md` 以及已审查的
`flow_screen.py` helper；没有运行 runner、加载模型、读取科学输出或提交
cluster job。

## 必须先修复的执行阻断

1. **Frozen/RTN schedule 在第二步必然报错**（当前
   `persistence_screen.py:455–460`）。
   `_run_schedule` 只在 `step==0`（或 `reload_each_step=True`）写入 map，随后
   却进入 `else: raise RuntimeError`。因此 `RTN` negative control、F0/F1/F2
   frozen arms 以及 common-path calls 都无法完成。最小修复是：label 在
   `maps` 中且无需 reload 时保留当前已写入的权重；cyclic 仍每步重写，并可
   保留现有 exact readback。

2. **FP 被误当作 Frozen draw**（当前 `persistence_screen.py:719–737`）。
   `arm.startswith("F")` 同时匹配 `FP`；之后 `int(arm[1:])` 会对 `"P"`
   抛出 `ValueError`。最小修复是把 Frozen 分支限制为 `arm in DRAW_NAMES`
   （或明确排除 `FP`），并让 `actual_schedule` 使用 runner 返回的 labels。

3. **prefix cache 参照跨 state 复用**（当前 `persistence_screen.py:694,
   726–728`）。`prefix_cache_digests["FP"]` 只由 state0/noise0 建立，随后
   state1..5 的不同 observation 会拿自己的 cache 与 state0 digest 比较，
   合法输入也会失败。最小修复是按 `state_index` 建立 FP reference，并只在
   同一 state 的 arms/noises/common calls 内比较。

4. **wrapper 的 protocol 默认文件名不匹配**（`persistence_gpu_screen.sh:29,
   39–46,60`）。默认读取 `${CONTROL}/persistence_protocol.zh.md`，仓库冻结
   文件是 `PROTOCOL.zh.md`。若 control staging 没有显式 alias 或环境覆盖，作业
   会在提交前检查失败；应统一默认名并复制同名 protocol。

## 会改变可复现性判断的合同缺口

5. **Quant snapshot 没有保存原始 weight 与 dequantized map**
   （`persistence_screen.py:300–342,650–665`）。`rounding_snapshots.pt` 当前只
   保存 shape、RTN codes、F0 scales、三套 draw codes 和 map digest；没有每层
   original weight、每套 dequant weight 或实际 write/restore receipt。虽然
   `_apply_map`/`_restore_fp` 当场做 exact readback，CPU verifier 事后无法验证
   code/scale 对应的真实 weight 或历史写回。最小修复是把必要的 CPU snapshot、
   dequant digest 和四类 transaction receipt 放入 engineering artifact，保持
   summary 小于 64 KiB。

6. **GPU 只验证 base identity，不验证 conditional extension parent chain**
   （`persistence_screen.py:147–226,610–636`）。manifest 中的
   `extension_identity_path/sha256` 只是记录，未读取、hash 或核对
   `bounded_extension.parent_identity_*` 及两枚 `file-002` 记录。另，manifest
   入口没有强制 state `[8]`、action `[7]`、padded `32`；helper 只检查 state 是
   finite 一维数组。若 producer 已冻结这些字段，runner 至少应 fail-closed
   核对它们；否则只能把 extension/source 证据作为未验证 provenance，不能称
   完整 source identity 绑定。

7. **固定 helper SHA 未执行**（`persistence_screen.py:123–145`）。当前只检查
   必要函数存在，未比较 protocol pin 的
   `ddf6e02ceb6b27a86ac479b54a6b24b5351342876916f39709504034efdb45a9`；控制脚本
   也只复制文件。应在 import 后立即 hash 并 fail-closed，防止 solver/loader
   漂移。

8. **actual schedule/time receipt 目前是预先重建的**。`_run_schedule` 返回
   `labels`（当前 `498`），但 caller 在 `732–737` 按 arm 名称重新生成
   `actual_schedule`，没有使用返回值；wrapper 也没有保存并核对每步实际
   `timestep`。这不会自动构成 scientific no-go，但无法独立证明每个 Q call
   使用了冻结 `1,.9,...,.1` 与预期 schedule。最小修复是保存返回 labels 和
   timestep，并与冻结数组逐步比较。

## 官方接口与现有 helper 的窄核对

- manual FP 路径（当前 `503–537`）使用已核对的 v0.4.4 顺序：
  `prepare_images`/`prepare_state` → `embed_prefix` → `make_att_2d_masks`、
  `cumsum(prefix_pad_masks)-1` → `vlm_with_expert.forward(..., past_key_values=None,
  fill_kv_cache=True)` → 十次 `denoise_step(x_t, prefix_pad_masks,
  past_key_values, timestep)`。`timestep` 与协议的 `1,.9,...,.1` 及 `dt=-.1`
  一致；没有发现额外 state 截断或错误的 physical action 解码。
- `_load_runtime` 委托固定 Flow helper 做 LeRobot `0.4.4`、完整 checkpoint
  strict load、FP32/CUDA、`rtc_config=None`、V100 和 processor/source hash
  检查；`_module_bindings` 直接绑定实际 `module.weight`，并拒绝重复的
  state entry。当前未发现 Q 参数 `state_dict` clone 被用于 mutation 的问题。
- raw `actions` 维度为 `[6,2,8,50,7]`，denoising `x/v` 保留 padded 32；这与
  protocol 的 physical7 readout 和 CPU verifier 轴约定一致。manifest loader
  仍应显式核对 raw sample 的 state8/action7 字段（见第6项）。

## 已与协议一致的部分

- `flow._eligible_modules` 的 `expert_W4` allowlist 被要求恰为 112 个实际
  float32 `Linear`；量化直接写入 `module.weight`，没有 state-dict clone
  假设。
- noise 使用 CPU `torch.Generator` 的 seeds 2201/2202，shape `[1,50,32]`；
  K=10、`dt=-.1`，raw 轴与 verifier 的
  `raw_x/raw_v/actions/common_q_v` 约定一致。
- FP 起点、Frozen/Cyclic 三 draw、RTN reuse/reload negative control、FP
  manual no-op、每 arm 后的 non-expert digest 与最终 FP restore 结构正确；
  V100 检查由固定 helper 的 `_gpu_evidence` 执行。

在修复第 1–4 项前不可执行；第 5–8 项修复前即使作业运行，也只能报告
engineering/provenance incomplete，不能据此改变 frozen/cyclic 的机制结论。
