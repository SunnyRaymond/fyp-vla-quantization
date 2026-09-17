# OpenVLA-OFT on LIBERO via vla-eval

目标：使用 released OpenVLA-OFT checkpoints，在 NSCC ASPIRE2A 上完成可核验、保留视频的 LIBERO evaluation。

## LIBERO-Goal full suite

- Model: `moojink/openvla-7b-oft-finetuned-libero-goal`
- Checkpoint: official per-suite Goal 50K checkpoint, revision `c2d0f9fbbd82674683b397ff923168a12f6a307b`（不是 joint 300K checkpoint）
- Benchmark: official `configs/benchmarks/libero/goal.yaml`（10 tasks × 50 episodes = 500 episodes）
- Jobs: `libero_goal_full.pbs` × 2；每个 job 使用 2×A100、2 个 model servers，并承担两个 global shards（每 GPU 125 episodes）
- Recording: 两个 jobs 各用独立 node-local SQLite/output，禁止跨节点共享数据库；最终在本地验证两边 union
- Completion gate: `tools/verify_libero_goal.py` 必须确认 SQLite 完整、500 个唯一 `(task, episode_id)`、每 task 恰好 0–49、500 个非空 MP4
- Harness: official `vla-eval` commit `4aeb4369640e8019d46af9534ce9b957e486ad38`
- Result: complete；Part 0 `16137211.pbs101` 为 246/250（98.4%，walltime 1:05:55），Part 1 `16137212.pbs101` 为 245/250（98.0%，walltime 1:04:33）
- Final union: 491/500（98.2%）；本地 verifier 确认 500 个唯一 episodes、10 tasks × 50、500 个非空 MP4
- Video decode: 抽检 Part 1 `task0000_ep0002_success.mp4`；H.264 High、256×256、20 FPS、6.05 s，首帧成功解码为 `artifacts/libero-goal-sample-frame.png`
- Summary: `artifacts/libero-goal-full-summary.json`

### Per-task results

| Task | Success |
|---|---:|
| open the middle drawer of the cabinet | 49/50 (98%) |
| open the top drawer and put the bowl inside | 46/50 (92%) |
| push the plate to the front of the stove | 50/50 (100%) |
| put the bowl on the plate | 50/50 (100%) |
| put the bowl on the stove | 48/50 (96%) |
| put the bowl on top of the cabinet | 48/50 (96%) |
| put the cream cheese in the bowl | 50/50 (100%) |
| put the wine bottle on the rack | 50/50 (100%) |
| put the wine bottle on top of the cabinet | 50/50 (100%) |
| turn on the stove | 50/50 (100%) |

## LIBERO-Spatial smoke 固定范围

- Model: `moojink/openvla-7b-oft-finetuned-libero-spatial`
- Model config: `configs/model_servers/oft/libero_spatial.yaml`
- Benchmark config: `configs/benchmarks/libero/smoke_test.yaml`
- Workload: `LIBERO-Spatial`, 1 task × 1 episode, seed 7
- Compute: 1×A100 40GB, 16 CPU cores, 110GB RAM
- Rendering: CPU；A100 只供 model server 使用
- Recording: video、SQLite/result files、stdout/stderr、environment metadata 全部保留
- Episode sharding: smoke test 不启用

## 目录

- Local: `D:\Downloads\Final Year Project\reproduction\openvla-oft-vla-eval-libero`
- Remote: `/scratch/users/ntu/yguo017/openvla-oft-vla-eval-libero`
- Remote job artifacts: `/scratch/users/ntu/yguo017/openvla-oft-vla-eval-libero/artifacts/<PBS_JOB_ID>`
- Local downloaded artifacts: `artifacts/<PBS_JOB_ID>`
- Local Goal artifacts: `artifacts/16137211.pbs101`、`artifacts/16137212.pbs101`
- Local Goal checkpoint: `source/checkpoints/openvla-7b-oft-finetuned-libero-goal`
- Remote Goal checkpoint: `/scratch/users/ntu/yguo017/openvla-oft-vla-eval-libero/source/checkpoints/openvla-7b-oft-finetuned-libero-goal`
- Local released checkpoint: `source/checkpoints/openvla-7b-oft-finetuned-libero-spatial`
- Remote released checkpoint: `/scratch/users/ntu/yguo017/openvla-oft-vla-eval-libero/source/checkpoints/openvla-7b-oft-finetuned-libero-spatial`

`prepare_container.pbs` 用 CPU job 从本地保留的 OCI archive 构建 SIF；`prepare_checkpoint.pbs` 用 CPU job 预取 released checkpoint；`smoke_test.pbs` 才申请 1×A100。scratch 中保留 source checkout、Apptainer image、Python/uv cache 和 Hugging Face checkpoint cache，避免后续重复下载。实验产物使用文件清单记录，不做 SHA-256。

## 当前状态

- Attempt 1: `16119283.pbs101`，failed in 1 second；`module purge` 后缺少 GCC dependency / `git` PATH，日志保留
- Attempt 2: `16119292.pbs101`，failed in 3 seconds；compute node 没有 Git，日志保留
- Source: official harness commit `4aeb4369640e8019d46af9534ce9b957e486ad38`，本地与远端均保留
- Dependency locks: base 97 packages；OFT server 149 packages
- Attempt 3: `16120685.pbs101`，远端 GHCR pull 极慢后终止，日志保留
- Container build 1: `16122442.pbs101`，OCI tar 的 `./` entry 被 Apptainer 1.5 拒绝，日志保留
- Container build 2: `16122460.pbs101`，exit 0；生成 `libero-latest.sif`（2,634,563,584 bytes）
- Checkpoint preload 1: `16122501.pbs101`，未运行即取消；4h walltime 被路由到繁忙的 `q2`
- Checkpoint preload 2: `16123013.pbs101`，一直未运行；本地 checkpoint 上传并建好 cache 后取消，取消前 qstat 保留在 remote `control/`
- Released checkpoint: Hugging Face revision `6d0231af0e48c5985f1ff86908f4674b84bc049b`，25 files、15,939,159,216 bytes；本地原文件和远端 raw directory 均保留
- Hugging Face cache: remote snapshot 由 raw directory hardlink 建立，保留已排队 GPU job 的位置且不重复占用约 16GB
- Smoke `16125597.pbs101`: exit 1 after 80-minute base sync；root cause 是 outer `vla-eval serve` 不接受 `--args.pretrained_checkpoint`，日志完整保留
- Retry hardening: 首次 run 的 node-local `uv sync` 出现长时间 network I/O wait；后续脚本改用 remote `$ROOT/cache/uv` 持久 cache，当前 running job 不受影响
- Dependency prep 1: `16127371.pbs101`（0-GPU）因 `/scratch` small-file extraction 极慢而取消；其 dependent retry `16127422.pbs101` 同时取消，材料保留
- WSL cache: `oft.py --help` 在 WSL2/Python 3.11 成功安装 147 locked packages；`source/uv-cache-linux.tar` 为 7,688,110,080 bytes，本地与远端均保留
- Scratch cache attempt: `16129037.pbs101` 与 dependent `16129038.pbs101` 取消；解包改到 GPU node-local，保留 partial 与 job records
- Cached smoke `16129298.pbs101`: 22 秒后 exit 1；证明 node-local 解包快速，失败仅因多余 base sync 要求 cache 中不存在的 `numpy 2.4.6`
- Offline check: fresh archive 排除不可迁移 `environments-v2` 后，147 packages offline rebuild + `oft.py --help` 成功，共 202 秒、install 533 ms
- Direct smoke `16129929.pbs101`: 14 秒后 exit 1；直接调用 `oft.py` 已绕过错误的 outer CLI，但 ASPIRE2A glibc 2.28 需要 cache 中没有的 `cryptography manylinux_2_28` wheel
- Target wheelhouse: 根据 `oft.py.lock` 为 CPython 3.11 / x86_64 / glibc ≤2.28 选择 143 个 wheels（3,534,912,952 bytes）；关键 `cryptography-46.0.5-...manylinux_2_28...whl` 已确认，未做 SHA-256
- Wheelhouse archive: `source/wheelhouse-manylinux_2_28.tar`（3,535,134,720 bytes）；上传 ASPIRE2A 后由 GPU job 解到 node-local storage，并通过 `UV_FIND_LINKS` 离线使用
- Target-wheel smoke `16130505.pbs101`: 依赖检查快速失败；PEP lock 的 remote URL 不会被 `find-links` 覆盖
- Local-URL smoke `16130547.pbs101`: 147 packages 离线安装、checkpoint load 与 health 全部通过；随后因 `apptainer exec` 跳过 OCI entrypoint，container 内找不到 `vla-eval`
- Container-entry smoke `16130581.pbs101`: 改用 `apptainer run` 后 benchmark 与 model server 成功连接；LIBERO 首次 import 交互询问 dataset path，在 batch stdin 上触发 `EOFError`
- Successful smoke `16130629.pbs101`: 使用非交互 `configs/libero/config.yaml`；PBS `Exit_status=0`，walltime 2:57，peak memory 14,890,088 KB
- Successful episode: `LIBERO-Spatial` task 0，1/1 success，80 steps，episode elapsed 49.504 s；aggregate mean success 1.0
- Video validation: H.264、256×256、20 FPS、81 frames、4.05 s；首帧成功解码，文件 57,398 bytes
- Result validation: `SUCCESS`、aggregate JSON、episode JSONL、SQLite、MP4 均存在；SQLite `integrity_check=ok`
- Retry script: 使用 `--args.pretrained_checkpoint="$ROOT/source/checkpoints/openvla-7b-oft-finetuned-libero-spatial"`；官方 model YAML 保持不变，`bash -n` 通过
- Scheduler: 曾报告 `scheduling=False`，之后恢复；最近三个 GPU smoke jobs 均能很快开始运行，当前失败与 allocation/queue 无关
- Monitor: thread heartbeat `openvla-oft-libero-smoke-monitor` checks every 5 minutes and stays quiet while queue state is unchanged
- Container source: immutable image digest `sha256:d0c45bc5a3720d569180e6b8dd92510da895f16c3cc509ccc76e4b4ffbb9e0f0`
- Local OCI archive: `source/images/libero-d0c45bc5a372.oci.tar`（2,743,703,552 bytes）
- Submitted: 2026-09-03
- Result: complete；成功 artifacts 已下载到 `artifacts/16130629.pbs101`，各 attempt 的小型 artifacts、PBS stdout 与 final qstat 也已下载保留
- LIBERO-Goal 4-GPU attempt: `16137185.pbs101`，因当时没有完整空闲 4-GPU node，在仍为 queued 状态时取消；原 PBS 保存在 `control/libero_goal_full_4gpu_16137185.pbs`
- LIBERO-Goal split run: Part 0 `16137211.pbs101`（global shards 0–1）与 Part 1 `16137212.pbs101`（global shards 2–3）；每个 1 node / 2×A100 / 32 CPU / 220GB RAM / 8h walltime
- Part 0 result: `Exit_status=0`，walltime 1:05:55；远端与本地 verifier 均确认 250 unique episodes、10 tasks、250 videos，mean success 98.4%
- Part 1 result: `Exit_status=0`，walltime 1:04:33；远端与本地 verifier 均确认 250 unique episodes、10 tasks、250 videos，mean success 98.0%
- LIBERO-Goal final result: union verifier 确认 500 unique episodes、10 tasks × 50、500 non-empty videos；491/500，mean success 98.2%；Part 0/1 artifacts 已完整下载
