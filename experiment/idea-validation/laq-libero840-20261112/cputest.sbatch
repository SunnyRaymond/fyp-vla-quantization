#!/bin/bash
# EEE CPU job: host tools the GPU job needs, then test840.py under the Fast-WAM stack (writes the manifest) and the
# Cosmos stack (must rebuild it identically).
#SBATCH --job-name=laq840ct
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=/projects/yguo017/laq840/logs/%x-%j.out
set -euo pipefail
P=/projects/yguo017
L=$P/laq840
U=/scratch/users/ntu/yguo017
ART=$L/artifacts/cputest/$SLURM_JOB_ID
mkdir -p "$ART" "$P/.tmp/$SLURM_JOB_ID" "$P/home"
cp -r "$L/code" "$ART/code"
export TMPDIR=$P/.tmp/$SLURM_JOB_ID
echo "HOST=$(hostname -s) CPUS=${SLURM_CPUS_ON_NODE:-?} $(date -u +%FT%TZ)"
python3 --version
squeue -h -u "$USER" -t R -o %A
python3 -c "import sys; sys.path.insert(0, '$ART/code'); import l840; print('walltime', l840.walltime_seconds('1-00:00:00'))"
export L840_CODE=$ART/code L840_THREADS=4 SLURM_JOB_ID SLURM_RESTART_COUNT=0
for model in fastwam cosmos; do
  flag=(); [[ $model == fastwam ]] && flag=(--write-manifest)
  srun --ntasks=1 --container-image="$P/libero-latest.sqsh" \
    --container-mounts="$P/scratch:$U,$P:$P,$P/home:$HOME" --container-workdir="$ART" --export=ALL \
    bash "$ART/code/stack840.sh" "$model" python "$ART/code/test840.py" --protocol "$ART/code/protocol.json" \
    --model "$model" "${flag[@]}" 2>&1 | grep -v 'robosuite WARNING'
  test "${PIPESTATUS[0]}" -eq 0 || { echo "CPUTEST_FAIL $model"; exit 1; }
done
echo CPUTEST_PASS
