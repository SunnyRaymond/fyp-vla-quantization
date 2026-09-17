# PRR CCDS-TC1 R1 protocol v1

本目录只承载 Paired-Rollout Recovery（PRR）的 R1 engineering screen。它检验在固定 hard signed W4、相同 action-conditioned history 和相同训练预算下，`local` target 与 paired-clean `recovery` target 的差异，并区分 quantizer-only 与 rank-4 LoRA。它不是 novelty 认证、native low-bit benchmark、闭环 success 实验或 TEST。

## 冻结范围

四个 arm 的顺序固定为：

```text
q_local       quantizer-only + local teacher target
q_recovery    quantizer-only + paired-clean recovery target
l_local       quantizer + rank-4 LoRA + local teacher target
l_recovery    quantizer + rank-4 LoRA + paired-clean recovery target
```

每个 arm 使用 fit seeds `1201, 1202, 1203`，合计 12 个 final evaluators。每格 `1000` updates、batch size `2`、H=`2`、`gamma=1`；四格使用同一 CAL record schedule、Q0 initialization、hardening schedule 和数据。CAL 使用 dataset indices `72..77`，DEV 使用 `78..83`；每个 split 有 6 episodes、每 episode 2 windows，共 12 records。environment namespaces 是 CAL/DEV=`1000000/1100000`，CEM namespaces 是 CAL/DEV=`1010000/1110000`。`0..71` 是历史保留范围，`84..95` 是锁定 TEST，本轮不可读取或打开。

Quantizer 是 signed symmetric per-output-channel weight-only W4，整数范围 `[-7,7]`，四格作用于同一 24 个 predictor Linear modules。LoRA 仅在 `l_*` 中启用，rank=`4`，每个 fit seed 的 `A` 使用 seeded-random 初始化、`B` 初始化为零；禁止 A/B 同时为零。先形成 `W_eff=W+BA`，再执行相同 hard quantization，且 hardening 固定为 `clip(floor(W_eff/s)+1[sigmoid(alpha)>=.5],-7,7)`；序列化结果必须没有 LoRA、alpha 或 FP bypass。统一使用可审计 STE，LoRA 的 `B/A` 梯度必须有效。`quantizer_lr=0.01`，`lora_lr=0.001`，temperature `2.0 -> 0.1`，rounding regularization `0.01 -> 0.1`。只有 CAL smoke 可以让 root 冻结这些更新与 rates；不得看 DEV 后改动。

FP reference 必须与旧 verified identity 相同：checkpoint SHA256 `8441971becdae934fe08de5b163398390a32f6fe1fb0a8df113290e2468e142b`、epoch `65`、source commit `0a9492fa12044b852ae9e001cc74604b79c8bb0c`、DINOv2 commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`、dtype `float32`、decoder `null`。FP evaluator 的 batch size 必须为 `null`。

单次 R1/R2 预算上限为 6 V100 GPU-hours，单卡 CCDS-TC1；本轮只运行 R1，不做 R2、R3 或 TEST。任何 GPU/model/NPZ/hash 处理都必须在真实 SLURM allocation 中先通过 `allocation_guard.require_allocation()`，证明 `SLURM_JOB_ID`、`scontrol` 的 `JobState=RUNNING`、`UserId`、实际 `TC1N##` hostname 和 `NodeList` 一致，并在需要模型时核验 GPU。login/head 只做 <=64 KiB 的控制文件与状态查询。verifier 不连接 SSH、不显示或传输 credentials，也不在本机打开 raw arrays。

## 实际 artifact contract

runner 在 compute-side artifact directory 写入 `raw_final.npz` 与 `raw_final_summary.json`；CPU verifier 写 `verification.json`。raw NPZ 可大于 64 KiB，禁止下载到本地。`raw_final_summary.json` 和 `verification.json` 必须各自不超过 64 KiB，且不得包含 raw arrays、credentials、tokens 或 private keys。

`raw_final.npz` 必须包含以下 keys。`E=12`，`N=12`，`Hslot=1`，`P` 和 `D` 由 predictor 输出决定：

```text
arm_ids                    [E] string, frozen evaluator order
terminal_error             [E,N,1,P,D], final free H=2 error vs FP
clean_error                [E,N,1,P,D], one-step clean error vs FP
donor_ids                  [K] string, exactly `q0`, `clean_seed_1201`
donor_recovery_error       [E,K,N,1,P,D], fixed-donor recovery diagnostic
fp_first                   [N,1,P,D], shared FP clean anchor
fp_second                  [N,1,P,D], shared FP H=2 target
fp_donor_local_target      [K,N,1,P,D], shared local targets
fp_donor_recovery_target   [K,N,1,P,D], shared paired-clean targets
wz                         [P,D] finite nonnegative weights
action_mask                [P,D] bool
metadata_json              scalar string containing record_rows and ledger metadata
```

`terminal_error` 是 primary metric，必须来自最终 hard/reloaded map 的自身 free rollout；固定 donor 或 teacher-forced scalar 不能替代它。`clean_error` 与 `donor_recovery_error` 是 secondary diagnostics，donor 轴 `K=2` 的顺序固定为 `q0`, `clean_seed_1201`。Verifier 从 `metadata_json.record_rows` 重建 record/episode grouping，并按episode/window身份连接同job canonical records核验initial-state fingerprints；从 `wz` 和 `action_mask` 审计 action 坐标为零、其他 active 坐标为正。以完整 `P*D` denominator 重算 weighted MSE：`mean((error*wz)^2)`，不把 records、windows、latent coordinates 或 seeds 当成独立环境样本。

`raw_final_summary.json` 必须包含：

```text
schema = "prr-final-v1"
status = "complete", formal = true, scope = "R1_only"
arms, seeds, fit_updates, batch_size, horizon, gamma, cal_records, dev_records
donors, raw_shapes, fit_runs, dev_tuning, r2_opened, test_opened, execution
```

`metadata_json` 必须另外包含 runner 的 `record_rows`（每个 row 的
`episode_id`, `record_index`, `dataset_index`, `window_start`, `env_seed`,
`cem_seed` 以及 collector 提供的 initial-state/episode fingerprints）、
`donors`, `arm_order` 和 `hard_checkpoint_ledger`。CPU verifier 读取
`methods/<arm>_seed_<seed>.json` 的 12 个 fit ledgers；每个 ledger 必须有：

```text
schema = "prr-fit-v1", status = "complete"
fit_updates_completed = fit_updates_expected = 1000, batch_size = 2
batch_schedule_hash, gradient_summary, quant_grad_after_step2
lora_A_grad_after_step2, lora_B_grad_after_step2, update_norms
reload_equal, hard_ledger, checkpoint
```

每个 `hard_ledger` 的 24 个 target row 必须给出 `q_min/q_max`、integer
范围和 finite positive per-output-channel scale。checkpoint payload 的
`state_dict` 只能含 non-target FP 参数；`hard_quant[path]` 必须含 int8
`integer`（仅 `[-7,7]`）和 float32 positive `scale`，不含 target FP weight
副本、LoRA 或 alpha。Verifier 在 compute allocation 内重算 checkpoint
SHA256，并从 payload/ledger 审计 hard grid；不加载 model。四格每个 seed
的 `batch_schedule_hash` 必须一致，更新后 quantizer gradient 必须有效，
`l_*` 还必须同时有有效的 LoRA A/B gradient。所有 12 arms 必须完成，即使
q-pair gate 失败。

## CPU verifier 与 gate

verifier 先调用 allocation guard，再打开 manifest、summary 或 NPZ；`--self-test` 只做静态 manifest audit，不读取 artifact。它应尽量报告前 24 个短错误，并在输出中保留 allocation、manifest audit、每个 arm completion、metric macro 与 gate status；不得打印 runtime identity 整段。

令 `T_{arm,seed,episode}` 为 terminal weighted MSE 在该 episode 的两 windows macro mean。lower is better。pair gate 分别比较 `q_recovery` 对 `q_local` 和 `l_recovery` 对 `l_local`：

```text
global mean(T_recovery) <= 0.95 * global mean(T_local) + 1e-12
至少 2/3 seeds：seed mean(T_recovery) <= 0.95 * seed mean(T_local) + 1e-12，
且该 seed 至少 4/6 episodes 严格方向改善（recovery < local）
每个通过 seed 的 clean mean <= 1.10 * local clean mean + 1e-12
global clean mean(recovery) <= 1.10 * global clean mean(local) + 1e-12
```

门限相对差不超过 `1e-6` 记 `ambiguous`，不升级为通过。q-pair 和 l-pair 都要报告；interaction 只描述同一 seed/episode 的 `(D-C)-(B-A)`，其中 A=`q_local`、B=`q_recovery`、C=`l_local`、D=`l_recovery`，不作显著性或因果声明。clean one-step 与 donor recovery 只作 secondary diagnostics，不替代 free H2 primary gate。

状态规则：12 arms/raw contract/FP identity/硬化/梯度/fingerprint/budget 任一工程条件失败，或有效 gradient 缺失，或资源不完整，均为 `inconclusive`，绝不能写成 scientific `mechanism_no_go`。仅当工程完整且 12 arms 完成后，若 q-pair 或 l-pair 任一组通过则输出 `conditional_signal`；只有两组 pair 都失败才输出 `mechanism_no_go`。两个 family 的结果、`ambiguous` 状态和 interaction 分开报告。本轮不读取 TEST，不启动 R2/R3。

本方案的配对/分组工作流参考 Kassis T., Agarwal V., He Y., Patel D., Brueckner A.M. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*，https://arxiv.org/abs/2609.00065 。该引用只说明 workflow 来源，不是 PRR 有效性证据。
