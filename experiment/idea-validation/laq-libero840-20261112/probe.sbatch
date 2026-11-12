#!/bin/bash
# EEE probe (one GPU, 10 min): node layout and whether the Fast-WAM venv / Cosmos env run CUDA kernels on this GPU model.
# sbatch --gres=gpu:<model>:1 probe.sbatch
#SBATCH --job-name=laq840pr
#SBATCH --time=00:10:00
#SBATCH --output=/projects/yguo017/laq840/logs/%x-%j.out
set -uo pipefail
source /projects/yguo017/setup/env.sh
echo "HOST=$(hostname) CVD=${CUDA_VISIBLE_DEVICES:-} NPROC=$(nproc) SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-} GPUS=${SLURM_JOB_GPUS:-}"
nvidia-smi -L
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader
cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null
for f in /sys/fs/cgroup/slurm*/uid_*/job_${SLURM_JOB_ID}/memory.max /sys/fs/cgroup/system.slice/slurmstepd.scope/job_${SLURM_JOB_ID}/memory.max; do [[ -e $f ]] && echo "$f $(cat $f)"; done
scontrol show job "$SLURM_JOB_ID" | grep -E 'TRES|MinMemory|NumCPUs'
free -g | head -2
cat > "$TMPDIR/probe.py" <<'PY'
import os, sys, time
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "arch_list", torch.cuda.get_arch_list(), flush=True)
print("CVD", os.environ.get("CUDA_VISIBLE_DEVICES"), "count", torch.cuda.device_count(), flush=True)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print("dev", i, p.name, "GPU-" + str(p.uuid), f"{p.total_memory/2**30:.1f}GiB", "cc", p.major, p.minor, flush=True)
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); t = time.time()
for _ in range(50):
    b = a @ a
torch.cuda.synchronize(); dt = (time.time() - t) / 50
print(f"bf16 matmul 4096^3: {2*4096**3/dt/1e12:.1f} TFLOP/s", flush=True)
q = torch.randn(1, 8, 512, 128, device="cuda", dtype=torch.bfloat16)
print("sdpa", torch.nn.functional.scaled_dot_product_attention(q, q, q).float().abs().mean().item(), flush=True)
for name in sys.argv[1:]:
    try:
        m = __import__(name)
        print("import", name, getattr(m, "__version__", "ok"), flush=True)
    except Exception as e:
        print("import", name, "FAILED", type(e).__name__, e, flush=True)
if "transformer_engine" in sys.argv[1:]:
    import transformer_engine.pytorch as te
    layer = te.Linear(256, 256, params_dtype=torch.bfloat16).cuda()
    print("te.Linear", layer(torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)).float().abs().mean().item(), flush=True)
PY
echo "=== fastwam venv"
run_fastwam python "$TMPDIR/probe.py" flash_attn fastwam mujoco robosuite libero
echo "=== cosmos env"
run_cosmos python "$TMPDIR/probe.py" transformer_engine megatron.core libero
echo "=== threads in container"
run_fastwam bash -c 'nproc; python -c "import os; print(len(os.sched_getaffinity(0)))"; which nvidia-smi || echo no-nvidia-smi-in-container'
echo PROBE_DONE
