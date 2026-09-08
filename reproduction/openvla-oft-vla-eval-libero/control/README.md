# OpenVLA-OFT on LIBERO via vla-eval

目标：使用 released OpenVLA-OFT checkpoint，在 NSCC ASPIRE2A 上先完成一个会保存视频的 LIBERO smoke test。

## 固定范围

- Model: `moojink/openvla-7b-oft-finetuned-libero-spatial`
- Model config: `configs/model_servers/oft/libero_spatial.yaml`
- Benchmark config: `configs/benchmarks/libero/smoke_test.yaml`
- Workload: `LIBERO-Spatial`, 1 task × 1 episode, seed 7
- Compute: 1×A100 40GB, 16 CPU cores, 110GB RAM
- Rendering: CPU；A100 只供 model server 使用
- Recording: video、SQLite/result files、stdout/stderr、environment metadata 全部保留
- Episode sharding: smoke test 不启用

## 目录

- Local: `D:\Downloads\Final Year Project\openvla-oft-vla-eval-libero`
- Remote: `/scratch/users/ntu/yguo017/openvla-oft-vla-eval-libero`
- Remote job artifacts: `/scratch/users/ntu/yguo017/openvla-oft-vla-eval-libero/artifacts/<PBS_JOB_ID>`
- Local downloaded artifacts: `artifacts/<PBS_JOB_ID>`

`smoke_test.pbs` 会在 scratch 中保留 source checkout、Apptainer image、Python/uv cache 和 Hugging Face checkpoint cache，避免后续重复下载。实验产物使用文件清单记录，不做 SHA-256。

## 当前状态

- Attempt 1: `16119283.pbs101`，failed in 1 second；`module purge` 后缺少 GCC dependency / `git` PATH，日志保留
- Attempt 2: `16119292.pbs101`，failed in 3 seconds；compute node 没有 Git，日志保留
- Source: official harness commit `4aeb4369640e8019d46af9534ce9b957e486ad38`，本地与远端均保留
- Dependency locks: base 97 packages；OFT server 149 packages
- Active PBS job: 使用 node-local temporary storage 的修正版待提交
- Submitted: 2026-09-03
- Result: pending；每次 attempt 的 artifacts 独立保留
