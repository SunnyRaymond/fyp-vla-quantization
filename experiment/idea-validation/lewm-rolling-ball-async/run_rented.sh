#!/usr/bin/env bash
# Run an explicitly authorized rented-host command with periodic GPU telemetry.
set -euo pipefail
EXPECTED_HOST=${1:?Usage: run_rented.sh EXPECTED_HOST OUTPUT_DIR COMMAND [ARGS...]}
OUT=${2:?Output directory required}
shift 2
test "$#" -gt 0
test "$(uname -s)" = Linux
ROLLING_HOST=$(hostname)
test "$ROLLING_HOST" = "$EXPECTED_HOST" || { printf 'Unexpected rented host: %s\n' "$ROLLING_HOST" >&2; exit 64; }
case "${ROLLING_HOST,,}" in *login*|*head*|*submit*) exit 64;; esac
test ! -e "$OUT"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/job.log") 2>&1
printf 'host=%s\nstarted_utc=%s\n' "$ROLLING_HOST" "$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$OUT/gpu_info.csv"
grep -qi 'RTX 4090' "$OUT/gpu_info.csv" || { printf 'Expected RTX 4090\n' >&2; exit 64; }
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total --format=csv -l 30 >> "$OUT/job.log" 2>&1 &
GPU_MONITOR_PID=$!
cleanup() {
  rc=$?
  kill "$GPU_MONITOR_PID" 2>/dev/null || true
  wait "$GPU_MONITOR_PID" 2>/dev/null || true
  printf '%s\n' "$rc" > "$OUT/wrapper_exit_status.txt"
}
trap cleanup EXIT
set +e
"$@"
rc=$?
set -e
printf '%s\n' "$rc" > "$OUT/runner_exit_status.txt"
exit "$rc"
