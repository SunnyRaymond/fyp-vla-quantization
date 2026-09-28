#!/usr/bin/env bash
set -uo pipefail

ASSET_ROOT=/root/autodl-tmp/rolling-ball-lewm/assets/local/Assets/Isaac/5.1
RELATIVE_PATH=Isaac/Props/Mugs/texture/T_Mug_A2_Normal.png
ASSET_PATH="$ASSET_ROOT/$RELATIVE_PATH"
PART_PATH="$ASSET_PATH.part"
JOB_DIR=/root/autodl-tmp/rolling-ball-lewm/assets/jobs/mug-normal-aria2-retry-20260927
URL=https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/Mugs/texture/T_Mug_A2_Normal.png
EXPECTED_BYTES=1899700

mkdir -p "$JOB_DIR"
exec 9>"$JOB_DIR/writer.lock"
if ! flock -n 9; then
  echo "another aria2 writer owns this job directory" >&2
  exit 73
fi

if [[ -e "$ASSET_PATH" ]]; then
  echo "refusing to overwrite completed target: $ASSET_PATH" >&2
  exit 73
fi
if [[ ! -f "$PART_PATH" ]]; then
  echo "expected resumable partial is missing: $PART_PATH" >&2
  exit 73
fi

set +e
aria2c --continue=true --file-allocation=none \
  --max-connection-per-server=4 --split=4 --min-split-size=1M \
  --user-agent=Python-urllib/3.12 --check-certificate=true \
  --dir="$(dirname "$PART_PATH")" --out="$(basename "$PART_PATH")" \
  "$URL"
ARIA2_EXIT=$?
set -e

BYTES=$(stat -c '%s' "$PART_PATH" 2>/dev/null || echo 0)
STATUS=FAIL
FINAL_EXIT=$ARIA2_EXIT
if [[ "$ARIA2_EXIT" -eq 0 && "$BYTES" -eq "$EXPECTED_BYTES" && ! -e "$ASSET_PATH" ]]; then
  mv -- "$PART_PATH" "$ASSET_PATH"
  STATUS=PASS
  FINAL_EXIT=0
elif [[ "$ARIA2_EXIT" -eq 0 ]]; then
  FINAL_EXIT=2
fi

printf '{"status":"%s","bytes":%s,"expected_bytes":%s,"exit_code":%s,"url":"%s"}\n' \
  "$STATUS" "$BYTES" "$EXPECTED_BYTES" "$FINAL_EXIT" "$URL" > "$JOB_DIR/result.json.tmp"
mv -- "$JOB_DIR/result.json.tmp" "$JOB_DIR/result.json"
printf '%s\n' "$FINAL_EXIT" > "$JOB_DIR/runner_exit_status.txt"
exit "$FINAL_EXIT"
