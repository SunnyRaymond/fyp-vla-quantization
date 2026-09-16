#!/bin/bash
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]]
ACTUAL_HOST=$(hostname -s)
[[ "$ACTUAL_HOST" =~ ^TC1N[0-9]+$ ]]
test -n "${SLURM_JOB_NODELIST:-}"
scontrol show hostnames "$SLURM_JOB_NODELIST" | grep -Fxq "$ACTUAL_HOST"
JOB_RECORD=$(scontrol show job -o "$SLURM_JOB_ID")
[[ "$JOB_RECORD" == *"JobState=RUNNING"* ]]
[[ "$JOB_RECORD" == *"UserId=$(id -un)("* ]]
[[ "$JOB_RECORD" == *"NodeList=${SLURM_JOB_NODELIST} "* ]]
