#!/bin/bash
set -euo pipefail
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
ACTUAL_HOST=$(hostname -s)
ACTUAL_HOST=${ACTUAL_HOST,,}
[[ "$ACTUAL_HOST" =~ ^tc1n[0-9]{2}$ ]]
test -n "${SLURM_JOB_NODELIST:-}"
JOB_RECORD=$(scontrol show job -o "$SLURM_JOB_ID")
[[ "$JOB_RECORD" == *"JobState=RUNNING"* ]]
[[ "$JOB_RECORD" == *"UserId=$(id -un)("* ]]
[[ " $JOB_RECORD " == *" JobId=${SLURM_JOB_ID} "* ]]
[[ " $JOB_RECORD " == *" Partition=UGGPU-TC1 "* ]]
RECORDED_NODELIST=$(awk '{for(i=1;i<=NF;i++) if($i ~ /^NodeList=/) {sub(/^NodeList=/,"",$i); print $i}}' <<< "$JOB_RECORD")
SCHED_HOSTS=$(scontrol show hostnames "$RECORDED_NODELIST" | tr '[:upper:]' '[:lower:]' | sort)
ENV_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '[:upper:]' '[:lower:]' | sort)
[[ "$SCHED_HOSTS" == "$ENV_HOSTS" ]]
grep -Fxq "$ACTUAL_HOST" <<< "$SCHED_HOSTS"
