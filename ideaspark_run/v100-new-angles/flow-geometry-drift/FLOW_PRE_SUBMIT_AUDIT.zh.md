# Flow Geometry pre-submit audit

**结论：当前有 5 个执行/解释阻断项；未改源文件、未连接集群。** 三 arm 隔离、raw velocity 形状和 metric 算式静态上通过。

## 必须修复

1. **artifact manifest 的相对根目录错误（执行阻断）**：CPU 脚本在 `smolvla_cpu_prepare.py:697-728` 同时写 `$ASSETDIR/manifest.json` 与 `artifacts/$SLURM_JOB_ID/manifest.json`；`sample_path` 是相对 `$ASSETDIR` 的 `samples/...`。runner 在 `flow_screen.py:201-249` 却相对 `manifest.parent` 解析。因此按 `PREPARATION.zh.md:19` 使用 artifact manifest 会找不到 sample。最小修复是 GPU 只接受 `$ASSETDIR/manifest.json`，或在 manifest 明确写并校验 `asset_root`。

2. **raw 输入合同不一致（执行阻断）**：PREPARATION 允许 `uint8 [C,H,W]`（`PREPARATION.zh.md:18`；CPU `:381-389` 允许 C=1/3/4），GPU `flow_screen.py:572-577` 却硬拒绝非 3-channel。CPU 还允许 checkpoint 的任意唯一 state key（`smolvla_cpu_prepare.py:297-335`），GPU 预处理后却硬要求 `observation.state`（`flow_screen.py:598-600`）。应在 CPU 阶段冻结 C=3 与 exact key，或让 GPU 按 manifest 映射；否则有效样本会失败/错配。GPU 的 `/255` 与官方 SmolVLA image 输入 `[0,1]` 相符，不将其判为 double-normalization。

3. **processor/cache 未被 identity hash 约束（解释阻断）**：GPU 只校验 checkpoint 与 `config.json`（`flow_screen.py:378-385`）；`_processor_identity`（`:424-445`）和本地 VLM 文件只记录 hash，没有与 CPU `identity.json`（`smolvla_cpu_prepare.py:715-720`）逐文件比对。替换 normalizer/tokenizer/config 可静默改变输入。应比较 checkpoint processor 与 base-VLM metadata 的 expected SHA-256，并拒绝额外权重。

4. **RTC 没有显式关闭（解释阻断）**：protocol 要求关闭 compile/RTC；GPU 只强制 `compile_model=False`（`flow_screen.py:332-355`）。`_DenoiseRecorder`（`:717-787`）未检查 `rtc_config`，若 checkpoint 开启 RTC，记录的 velocity/solver 路径不再保证 `dt=-0.1` 的普通十步 Euler。应设 `rtc_config=None/disabled` 并写入 summary，或 fail-closed。

5. **12-episode provenance 未完全验证（误判阻断）**：CPU 选择规则在 `smolvla_cpu_prepare.py:452-484`，但 GPU `flow_screen.py:267-279` 只验证“四 task×三 episode、排序、floor(length/4)”，没有验证“前四 task/各前三 episode”，也不读取 NPZ `metadata_json`（`:557-578`）。旧 manifest 或手工重标仍可通过。应比对 `identity.selection` 的 task/episode IDs，并校验嵌入 metadata 与 manifest 一致。

## 已通过/需记录

`_eligible_modules`/`_quantize_locus` 与每 arm 前 restore（`flow_screen.py:614-690,1092-1120`）保持两个 locus 独立；recorder 的 `[10,50,32]`、action `[50,7]` 和 `_accel`/drift/Spearman（`:717-920`）符合 protocol。官方 v0.4.4 `predict_action_chunk` 返回 normalized flow result，因此当前 normalized-action MSE 语义成立；但代码仅检查 `(7,)`（`:514-517`），没有把 action_feature/unnormalizer 的物理维度顺序写入 identity，若该顺序是硬 gate，仍应补一条显式记录或收窄 protocol 表述。
## Runtime repair 64760

GPU job64760 FAILED 1:0 after2s; shell allocation guard passed, Python runner failed before loading models/data: local importlib.util import shadowed module binding, causing UnboundLocalError at guard import. Originalrun.log retained; no summary or scientific output exists. Root removed unused dynamic-path fallback; the shell already snapshots allocation_guard.py alongside runner, so one direct import is sufficient. Same frozen scientific protocol.

## Runtime repair 64761

GPU64761 FAILED1:0/19s，config解析阶段失败：直接SmolVLAConfig.from_pretrained不能消费registry discriminator type。官方PreTrainedPolicy.from_pretrained实际调用PreTrainedConfig.from_pretrained，root改为该入口并检查解析出的类型。另依官方pretrained.py核对checkpoint的save_model/load_model配对，采用safetensors.load_model(strict=True)，以正确验证共享storage的完整覆盖；不使用strict=False、不手工丢弃missing keys。失败发生在模型实例化前，无scientific输出。asset_identity_brief已导出，完整identity因>64KB保留compute端并以SHA绑定，小manifest本地已有12样本。官方来源：https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/pretrained.py 和 src/lerobot/configs/policies.py。

## Runtime repair 64762

GPU64762 FAILED1:0/40s。完整checkpoint strict load、config/source/hash、official processor实例化均已通过；raw第一样本完成preprocessing，runner whitelist却未包含官方transition converter生成的next.reward/next.done/next.truncated。root按实际返回加入这三个complementary metadata keys；仍只传observation.* tensors到policy，拒绝非空action训练target。没有产生scientific输出或更改实验条件。
