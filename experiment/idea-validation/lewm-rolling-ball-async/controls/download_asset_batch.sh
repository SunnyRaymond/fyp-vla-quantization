#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 ]]; then
    echo "usage: download_asset_batch.sh MANIFEST ASSET_ROOT RUN_DIR" >&2
    exit 2
fi

manifest=$1
asset_root=$2
run_dir=$3
default_base=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1
base=${ASSET_BASE_URL:-$default_base}
case "$base" in
    "$default_base"|https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Isaac/5.1) ;;
    *) echo "refusing unapproved asset host: $base" >&2; exit 2 ;;
esac
mkdir -p "$run_dir"

if [[ ! -f "$manifest" ]]; then
    echo "missing manifest: $manifest" >&2
    exit 2
fi
if [[ -n "$(grep -vE '^[[:space:]]*(#|$)' "$manifest" | sort | uniq -d)" ]]; then
    echo "duplicate asset path in manifest" >&2
    exit 2
fi

printf '{"status":"RUNNING","manifest":"%s","asset_root":"%s"}\n' "$manifest" "$asset_root" > "$run_dir/result.json.tmp"
mv "$run_dir/result.json.tmp" "$run_dir/result.json"

download_one() {
    local rel=$1 target
    [[ "$rel" == Isaac/* && "$rel" != *".."* ]] || { echo "refusing path: $rel" >&2; return 2; }
    target="$asset_root/$rel"
    if [[ -s "$target" ]]; then
        echo "SKIP $rel $(stat -c %s "$target") bytes"
        return 0
    fi
    mkdir -p "$(dirname "$target")"
    curl --fail --location --show-error --continue-at - --max-time 900 --retry 2 --retry-all-errors --progress-bar \
        "$base/$rel" -o "$target.part"
    [[ -s "$target.part" ]]
    mv "$target.part" "$target"
    echo "DONE $rel $(stat -c %s "$target") bytes"
}
export -f download_one
export asset_root base

set +e
grep -vE '^[[:space:]]*(#|$)' "$manifest" | xargs -r -P 4 -I{} bash -c 'download_one "$1"' _ {}
rc=$?
set -e

status=PASS
if (( rc != 0 )); then status=FAIL; fi
printf '{"status":"%s","exit_code":%d,"manifest":"%s","asset_root":"%s"}\n' \
    "$status" "$rc" "$manifest" "$asset_root" > "$run_dir/result.json.tmp"
mv "$run_dir/result.json.tmp" "$run_dir/result.json"
printf '%s\n' "$rc" > "$run_dir/runner_exit_status.txt"
exit "$rc"
