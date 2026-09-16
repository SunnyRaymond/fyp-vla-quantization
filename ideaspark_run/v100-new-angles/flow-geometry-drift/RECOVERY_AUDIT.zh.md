# Flow CPU recovery evidence audit

日期：2026-09-13。范围为审查 GPU job 64763 的 summary overflow、CPU job 64764 recovery 及其证据边界；未新增实验，未连接集群，未修改 runner/verifier。

## 更新后的有限结论

可以把结果写成 numeric scope_limited_preliminary_go，并附 reporting provenance loss；不能写成 GPU/CPU agreement，也不能把 64763 改写为完整成功运行。backbone_W4 的 raw 数值 screen 通过，expert_W4 未通过，因此该结论只覆盖 backbone locus。

若项目要求单一 protocol-level 状态必须所有 engineering gate 都有完整 hash 与持久化记录，则整体状态应另标 inconclusive_provenance（或 implementation_failure_recovery）。这不抹去已由完整 raw 确定的数值子结果；报告必须把 scientific numeric decision 与复现工程状态分开。

## 64763 原始证据

64763 的 allocation.json 记录 SLURM job=64763、partition=UGGPU-TC1、hostname=tc1n03、NodeList=TC1N03，guard verified=true。专属 gpu_identity.csv 已确认 Tesla V100-PCIE-32GB、GPU UUID 与 32768 MiB。GPU 专属 asset_identity_brief 的 full identity SHA 与 CPU recovery 当前 identity 绑定，12 个 raw samples 的 SHA 均与 manifest 对上，raw episode_ids 顺序也通过 manifest binding。

run.log 的 traceback 固定落在 flow_screen.py L1255 的最终 _write_small_summary；L1060 报 110959 bytes 超过 64 KiB。64764 复核确认原 summary 未被覆盖，原始 fallback error 与 job provenance 保留。runner SHA 为 ddf6e02ceb6b27a86ac479b54a6b24b5351342876916f39709504034efdb45a9，并与当前已审查的 flow_screen.py 一致。

在上述 runner SHA 与 traceback 绑定成立时，达到 L1255 表明程序已越过 12 episodes、finite outputs、bypassed-state、FP restoration 和 _metrics 调用；因此失败位置是 summary 持久化。该证据仍是 control-flow inference，不是已保存的 no-op/restore 数值 payload。

## 64764 raw recovery 结果

verification.json 的 raw_vector_validation_pass=true、runtime_record_checks_pass=true、original_summary_unchanged=true、model_inference_repeated=false；raw SHA 为 eb33c5c6324be68025a7a947bfe645d243bad930d0540823d56588d66b87eece。CPU 将 raw float32 vectors 转为 float64，按原 flow_screen.py L952–1043 重算 Eq.11 acceleration、drift、q-action-norm、两 noise seed 的 episode 平均和 Spearman。该数值路径是从 raw 的确定性重算，不是与 GPU 序列化 metrics 做比对。

backbone_W4：rho(Q accel, drift)=0.6293706，rho(FP accel, drift)=0.3496503，rho(Q norm, drift)=0.1468531；两个增量分别为 0.2797203 与 0.4825175，满足 frozen gates，locus decision=preliminary_go。

expert_W4：rho(Q accel, drift)=0.1818182，rho(FP accel, drift)=0.2447552，rho(Q norm, drift)=0.2237762；两个增量为 -0.0629371 与 -0.0419580，locus decision=mechanism_no_go。按协议，两 locus 的总标签应为 scope-limited，而不是 full preliminary_go。

## 仍不可声称的事项

1. 64763 从未成功写出完整 summary；GPU-calculated numeric metrics、per-layer quantizer details、numeric bypass digest 和完整 runtime summary 均未恢复。不得报告 GPU/CPU metrics match；应使用 recovered-from-raw 或 numeric-recomputed 表述。

2. no-op 的 action_exact_equal、velocity_finite、wrapper_restored，以及量化前后具体 restore/bypass digest 没有独立持久化。只能在已绑定 runner source 和最终写入 traceback 下写 control_flow_gate_pass_inferred，不能写 no_op_evidence_saved。

3. runner hash、checkpoint/asset identity、manifest 与 GPU allocation 已绑定，但原 runtime package 的逐文件 source hash 详细值丢失。lerobot package version=0.4.4 的 fail-closed 检查通过，只能提供 version-level provenance，不能冒称精确 package hash 已恢复。这是 reporting provenance loss；若冻结 protocol 把逐文件 package hash作为硬 engineering gate，则单独保留 overall inconclusive_provenance。

## Recovery 输出边界

64764 的 recovery report 应保留 original_job_status=FAILED at final summary serialization、original_summary_unchanged=true、gpu_cpu_metric_agreement_verified=false，并分开记录 raw_recomputed_metrics、control_flow_gate_pass_inferred、provenance_loss 和未恢复字段。不要覆盖原 summary，不要借用 64762 等其他 job 的 metrics。当前不需要重复 GPU 或 CPU；后续只可在报告层明确上述双层结论。
