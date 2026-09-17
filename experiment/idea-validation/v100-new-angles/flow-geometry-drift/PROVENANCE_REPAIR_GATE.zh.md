# Flow geometry provenance-only repair gate

审查范围：64763 的小型失败记录、64764 recovery、`RESULT.zh.md`、`RECOVERY_AUDIT.zh.md`、冻结协议及当前 `flow_screen.py` 的持久化顺序。未读取 `raw_flow.npz`，未连接集群，未运行模型或统计。

## 判定

当前不具备必要的 provenance-only repair eligibility；保留已有的双层结论即可：

- scientific numeric：`backbone_W4` 为 `scope_limited_preliminary_go`，`expert_W4` 为 `mechanism_no_go`；数值来自 64764 对已保存 raw 的独立 CPU 重算。
- engineering/reproducibility：`inconclusive_provenance`。64763 必须继续记为 scheduler FAILED，不能改写为成功 GPU job。

## 为什么不能由 CPU 补齐

`flow_screen.py` 在 L1200 先完成 no-op gate，L1213–1241 完成两 locus 的 restore、bypass digest 和 arm loop，L1242–1246 完成 12 episodes、finite raw 与 `_metrics`，随后 L1255 才写 64 KiB 超限的完整 summary。故现有 traceback + runner SHA 可以支持 `control_flow_gate_pass_inferred`，但不是已保存的详细 receipt。

原始完整 raw 只包含 flow velocities/actions 及其输入身份，不能反推出当次 GPU 的 runtime package file hashes、逐层 quantizer records、numeric bypass digest 或完整 runtime summary。基于当前 checkpoint/source 重新计算这些值会生成新的记录，不能证明它们就是 64763 当次内存状态；也不能把 64764 的 CPU 重算冒称 GPU/CPU metric agreement。现有 recovery 已完成可做的 raw integrity、manifest/asset binding 和独立 metric replay。

## 是否应 exact rerun

对原 12 episodes、两 noise seeds、FP/backbone-W4/expert-W4 完全重跑一次，只是为补完整工程 receipt；它不增加科学样本，概念上属于 reproducibility rerun，而非新的 full validation。但当前协议和 campaign 的 STOP 规则要求不为报告缺口重复 inference；已有 numeric 结果也不依赖该 rerun。因此在没有新的明确授权或 publication-level provenance 硬要求前，不应解冻、不应提交 GPU/CPU repair job，也不能以 quota 余额作为理由。

若未来确有外部审计硬性要求，必须另立一个明确标注“exact provenance rerun”的 repair gate，固定原 manifest、source/checkpoint/loci/seeds、同一输出合同和失败后的小 summary 预算；其结果只能修复复现记录，不能扩大样本、改指标、改 expert 结论或重写 64763 历史。

## Root 建议

现在归档 raw-recomputed 数值和 provenance loss，明确 `gpu_cpu_metric_agreement_verified=false`、`model_inference_repeated=false`，保留 64763 原始 FAILED/error。无需 CPU 补提取，也无需 GPU 重跑。
