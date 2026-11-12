#!/bin/bash
# EEE GPU job of the standard-LIBERO 840 loop.
#   sbatch --gres=gpu:<model>:<n> --time=HH:MM:SS [--qos=override-limits-but-killable] \
#          --export=ALL,MODEL=fastwam|cosmos,STAGE=smoke|pilot|loop run840.sbatch
# Host side: snapshot of the code, GPU/CPU samples into job.log every 60 s, the running-job list for claim recovery,
# then one container step (stack840.sh + slots840.sh) that runs every slot. Requeue-safe: a requeued run gets a new
# restart count, so its stale claims are dropped and the episodes already saved are kept.
#SBATCH --job-name=laq840
#SBATCH --output=/projects/yguo017/laq840/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
set -euo pipefail
: "${MODEL:?}" "${STAGE:?}"
P=/projects/yguo017
L=$P/laq840
U=/scratch/users/ntu/yguo017
TAG=${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}
ART=$L/artifacts/$MODEL/$TAG
mkdir -p "$ART" "$P/.tmp/$SLURM_JOB_ID" "$P/home"
cp -r "$L/code" "$ART/code"
exec > >(tee -a "$ART/job.log") 2>&1
export TMPDIR=$P/.tmp/$SLURM_JOB_ID
LIMIT=$(squeue -h -j "$SLURM_JOB_ID" -o %l)
WALL=$(python3 -c "import sys; sys.path.insert(0, '$ART/code'); import l840; print(l840.walltime_seconds('$LIMIT'))")
START=$(date +%s)
WPG=${WORKERS_PER_GPU:-$(python3 -c "import json; print(json.load(open('$ART/code/protocol.json'))['models']['$MODEL']['workers_per_gpu'])")}
NGPU=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)
NCPU=${SLURM_CPUS_ON_NODE:-4}
THREADS=$(( NCPU / (NGPU * WPG) )); (( THREADS >= 1 )) || THREADS=1
echo "JOB=$SLURM_JOB_ID RESTART=${SLURM_RESTART_COUNT:-0} MODEL=$MODEL STAGE=$STAGE HOST=$(hostname -s) QOS=$(squeue -h -j "$SLURM_JOB_ID" -o %q) LIMIT=$LIMIT GPUS=$NGPU CPUS=$NCPU WPG=$WPG THREADS=$THREADS START=$(date -u +%FT%TZ)"
scontrol show job "$SLURM_JOB_ID" | grep -E 'TRES|QOS' || true
hostname -s > "$ART/nodefile"

running() {  # shared list of running jobs for claim recovery
  local tmp="$L/running.json.$SLURM_JOB_ID"
  if ids=$(squeue -h -u "$USER" -t R -o %A); then
    python3 -c "import json, sys, time; json.dump(dict(epoch=time.time(), jobs=sys.argv[1:]), open('$tmp', 'w'))" $ids \
      && mv -f "$tmp" "$L/running.json"
  fi
}
monitor() {  # GPU_SAMPLE per GPU every 60 s, CPU_SAMPLE every 120 s, running list every 5 min
  local n=0
  while :; do
    nvidia-smi --query-gpu=uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader \
      | sed "s/^/GPU_SAMPLE timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ) /" || echo GPU_SAMPLE_ERROR
    if (( n % 2 == 0 )); then
      top -b -n 2 -d 2 -u "$USER" 2>/dev/null | awk -v ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)" -v req="$NCPU" \
        '/^top -/ {frame++} frame == 2 && $1 ~ /^[0-9]+$/ {s += $9} END {printf "CPU_SAMPLE timestamp=%s busy_cores=%.1f allocated=%s\n", ts, s / 100, req}' \
        || echo CPU_SAMPLE_ERROR
      free -g | awk -v ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)" '/^Mem:/ {print "MEM_SAMPLE timestamp=" ts " node_used_gb=" $3 " node_total_gb=" $2}'
    fi
    (( n % 5 == 0 )) && running
    n=$(( n + 1 ))
    sleep 60
  done
}
running
monitor & MON=$!
finish() {
  local rc=$?
  kill "$MON" 2>/dev/null || true
  echo "$rc" > "$ART/exit_code.txt"
  (( rc == 0 )) && touch "$ART/PIPELINE_COMPLETE"
  echo "EXIT rc=$rc $(date -u +%FT%TZ)"
}
trap finish EXIT

export STAGE ARTIFACTS=$ART L840_CODE=$ART/code L840_PROTOCOL=$ART/code/protocol.json WORKERS_PER_GPU=$WPG
export L840_THREADS=$THREADS L840_DEADLINE_EPOCH=$(( START + WALL - 1500 )) L840_BUILD_LOCK=$ART/build.lock
export PBS_JOBID=slurm-$TAG PBS_NODEFILE=$ART/nodefile
srun --ntasks=1 --container-image="$P/libero-latest.sqsh" \
  --container-mounts="$P/scratch:$U,$P:$P,$P/home:$HOME" \
  --container-workdir="$ART" --export=ALL \
  bash "$ART/code/stack840.sh" "$MODEL" bash "$ART/code/slots840.sh"
echo PIPELINE_COMPLETE
