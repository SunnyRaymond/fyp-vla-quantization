#!/usr/bin/env python3
"""Measure Plus glass-blur environment cost on frozen Sensor Noise rows only."""
from __future__ import annotations

import json
import math
import os
import random
import re
import socket
import statistics
import sys
import time
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
NOOP = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
NOISE_SUFFIX = re.compile(r"_noise_(\d+)$")


def require_allocation():
    job = os.environ.get("PBS_JOBID")
    nodefile = os.environ.get("PBS_NODEFILE")
    host = socket.gethostname().split(".")[0]
    if not job or not nodefile or "login" in host:
        raise RuntimeError("Approved PBS compute allocation required")
    nodes = {line.split(".")[0] for line in Path(nodefile).read_text().split()}
    if host not in nodes:
        raise RuntimeError(f"Host {host} is not listed in PBS_NODEFILE")
    return job, host


def stats(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "mean_seconds": 0.0, "p95_seconds": 0.0, "max_seconds": 0.0}
    return {
        "count": len(values),
        "mean_seconds": statistics.fmean(values),
        "p95_seconds": values[max(0, math.ceil(0.95 * len(values)) - 1)],
        "max_seconds": values[-1],
    }


def prepare_shard_ranges(root, out, manifest, job):
    rows = manifest['variants']
    ranges, cells, start = [], set(), 0
    while start < len(rows):
        cell = (rows[start]['suite'], rows[start]['dimension'])
        stop = start + 1
        while stop < len(rows) and (rows[stop]['suite'], rows[stop]['dimension']) == cell:
            stop += 1
        if cell in cells or stop - start != 50:
            raise RuntimeError(f'Frozen cell is not one contiguous 50-row block: {cell}')
        cells.add(cell)
        size = 1 if cell[1] == 'sensor_noise' else 4
        ranges.extend({'start': i, 'stop': min(i + size, stop), 'dimension': cell[1]}
                      for i in range(start, stop, size))
        start = stop
    if len(cells) != 28 or len({r['variant_id'] for r in rows}) != 1400:
        raise RuntimeError('Range plan does not cover the frozen 28-cell cohort')
    plan = {'protocol': 'fastwam-optional-idm-plus-pilot-v1', 'variant_count': 1400,
            'episode_count': 5600, 'ranges': ranges, 'prepared_job_id': job,
            'source_commit': manifest['source']['commit'], 'episode_counted': False}
    for target in (out / 'shard_ranges.json', root / 'shard_ranges.json'):
        if target.exists() and json.loads(target.read_text()) != plan:
            raise RuntimeError(f'Refusing to overwrite a different submitted range plan: {target}')
        temp = target.with_suffix('.json.tmp')
        temp.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
        temp.replace(target)
    print(f'SHARD_RANGES_READY job={job} children={len(ranges)} variants=1400 episodes=5600', flush=True)


def main():
    job, host = require_allocation()
    root = Path(os.environ["ROOT"])
    repo = Path(os.environ["REPO"])
    out = Path(os.environ["OUT"])
    status = json.loads((out / "env_status.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "manifest_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    native = json.loads((root / "native_status.json").read_text(encoding="utf-8"))
    if (status.get("checks", {}).get("ready") is not True
            or status.get("checks", {}).get("render_preflight_ready") is not True
            or summary.get("variant_count") != 1400
            or len(manifest.get("variants", [])) != 1400
            or summary.get("source_commit") != status.get("source_commit")
            or manifest.get("source", {}).get("commit") != status.get("source_commit")
            or native.get("ready") is not True):
        raise RuntimeError("Frozen source, manifest, native, or render-preflight gate is not ready")

    assets = Path(status["assets"]["path"])
    asset_commit = status["assets"]["commit"]
    marker = root / f"assets-{asset_commit}.complete"
    if (not marker.is_file() or marker.read_text(encoding="ascii").strip() != asset_commit
            or not (assets / "scenes" / "libero_tabletop_base_style.xml").is_file()):
        raise RuntimeError("Official Plus asset receipt or tabletop scene is missing")
    runtime_assets = Path(status["paths"]["runtime_assets"])
    if runtime_assets.resolve() != assets.resolve():
        raise RuntimeError("Runtime asset alias no longer resolves to the official Plus assets")

    if '--prepare-shards' in sys.argv[1:]:
        prepare_shard_ranges(root, out, manifest, job)
        return

    from libero_state_loader import ensure_numpy_compat
    numpy_compat = ensure_numpy_compat()
    import numpy as np

    import_started = time.perf_counter()
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.env_wrapper as noise_module
    from libero_state_loader import load_task_init_states

    source_package = (repo / "libero" / "libero").resolve()
    for label, source in (("benchmark", benchmark.__file__),
                          ("envs", __import__("libero.libero.envs", fromlist=["__file__"]).__file__),
                          ("noise_wrapper", noise_module.__file__)):
        try:
            Path(source).resolve().relative_to(source_package)
        except ValueError as exc:
            raise RuntimeError(f"{label} loaded outside frozen Plus source: {source}") from exc
    if not (source_package / "benchmark" / "task_classification.json").is_file():
        raise RuntimeError("Frozen Plus classification source is missing")
    print(f"PROBE_SOURCE_ASSET_GATES_OK source={status['source_commit']} assets={asset_commit} "
          f"imports_seconds={time.perf_counter() - import_started:.3f}", flush=True)

    if '--check-fog' in sys.argv[1:]:
        results = []
        classes = benchmark.get_benchmark_dict()
        for suite_name in SUITES:
            index, row = next((i, r) for i, r in enumerate(manifest['variants'])
                              if r['suite'] == suite_name and r['dimension'] == 'sensor_noise'
                              and (match := NOISE_SUFFIX.search(r['task_name']))
                              and 31 <= int(match.group(1)) <= 40)
            suite = classes[suite_name]()
            env = OffScreenRenderEnv(bddl_file_name=str(suite.get_task_bddl_file_path(int(row['task_id']))),
                                     camera_heights=256, camera_widths=256)
            try:
                states, state_path = load_task_init_states(suite, int(row['task_id']))
                seed = 800_000_000 + index * 1000
                env.seed(seed); np.random.seed(seed); random.seed(seed)
                env.reset()
                env.set_init_state(states[0])
                obs, _, _, _ = env.step(NOOP)
                assert tuple(obs['agentview_image'].shape) == (256, 256, 3)
                results.append({'suite': suite_name, 'variant_id': row['variant_id'], 'noise_level': int(env.noise),
                                'state_id': 0, 'initial_state_path': state_path, 'environment_seed': seed})
                print(f'FOG_COMPAT_SUITE_OK suite={suite_name} variant={row["variant_id"]}', flush=True)
            finally:
                env.close()
        receipt = {'pbs_job_id': job, 'passed': True, 'episode_counted': False, 'gpu_loaded': False,
                   'numpy_compatibility': numpy_compat, 'source_commit': status['source_commit'],
                   'asset_commit': asset_commit, 'reset_and_step_verified': results}
        (out / 'fog_compatibility.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        print('FOG_COMPAT_COMPLETE suites=4 episode_counted=false', flush=True)
        return

    selected = {}
    for index, row in enumerate(manifest["variants"]):
        if row.get("dimension") != "sensor_noise" or row.get("suite") not in SUITES:
            continue
        match = NOISE_SUFFIX.search(row.get("task_name", ""))
        if not match:
            continue
        level = int(match.group(1))
        if 41 <= level <= 50:
            candidate = (level, -int(row["task_id"]), index, row)
            if row["suite"] not in selected or candidate[:3] > selected[row["suite"]][:3]:
                selected[row["suite"]] = candidate
    missing = [suite for suite in SUITES if suite not in selected]
    if missing:
        raise RuntimeError(f"Frozen cohort has no selected glass-blur variant for suites: {missing}")

    catalog_started = time.perf_counter()
    classes = benchmark.get_benchmark_dict()
    print(f"PROBE_VARIANTS_SELECTED benchmark_catalog_seconds={time.perf_counter()-catalog_started:.3f} "
          + json.dumps({suite: {"level": selected[suite][0], "variant_id": selected[suite][3]["variant_id"],
                                "task_name": selected[suite][3]["task_name"]} for suite in SUITES}),
          flush=True)

    original_glass_blur = noise_module.glass_blur
    results = []
    try:
        for suite_name in SUITES:
            level, _, index, row = selected[suite_name]
            suite = classes[suite_name]()
            task = suite.get_task(int(row["task_id"]))
            if task.name != row["task_name"]:
                raise RuntimeError(f"Frozen task name mismatch: {row['task_name']} vs {task.name}")
            calls = []

            def timed_glass_blur(image, severity=1):
                started = time.perf_counter()
                try:
                    return original_glass_blur(image, severity=severity)
                finally:
                    calls.append(time.perf_counter() - started)

            noise_module.glass_blur = timed_glass_blur
            env = None
            try:
                started = time.perf_counter()
                env = OffScreenRenderEnv(
                    bddl_file_name=str(suite.get_task_bddl_file_path(int(row["task_id"]))),
                    camera_heights=256,
                    camera_widths=256,
                )
                create_seconds = time.perf_counter() - started
                if int(env.noise) != level:
                    raise RuntimeError(f"Noise severity mismatch for {row['variant_id']}: {env.noise} != {level}")

                states_started = time.perf_counter()
                states, state_path = load_task_init_states(suite, int(row["task_id"]))
                state_load_seconds = time.perf_counter() - states_started
                if not len(states):
                    raise RuntimeError(f"Official init states are empty for {row['variant_id']}")

                seed = 800_000_000 + index * 1000
                env.seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                before = len(calls)
                started = time.perf_counter()
                env.reset()
                reset_seconds = time.perf_counter() - started
                reset_noise_seconds = sum(calls[before:])

                started = time.perf_counter()
                env.set_init_state(states[0])
                set_state_seconds = time.perf_counter() - started

                settle_seconds = []
                settle_noise_seconds = []
                for _ in range(30):
                    before = len(calls)
                    started = time.perf_counter()
                    _, _, done, _ = env.step(NOOP)
                    settle_seconds.append(time.perf_counter() - started)
                    settle_noise_seconds.append(sum(calls[before:]))
                    if done:
                        raise RuntimeError(f"No-op settling completed {row['variant_id']}")

                step_seconds = []
                step_noise_seconds = []
                for _ in range(10):
                    before = len(calls)
                    started = time.perf_counter()
                    _, _, _, _ = env.step(NOOP)
                    step_seconds.append(time.perf_counter() - started)
                    step_noise_seconds.append(sum(calls[before:]))

                step_stats = stats(step_seconds)
                noise_stats = stats(step_noise_seconds)
                settle_stats = stats(settle_seconds)
                results.append({
                    "suite": suite_name,
                    "variant_id": row["variant_id"],
                    "task_id": row["task_id"],
                    "task_name": task.name,
                    "state_id": 0,
                    "noise_level": level,
                    "glass_blur_severity": level - 40,
                    "environment_seed": seed,
                    "initial_state_path": state_path,
                    "reset_seconds": reset_seconds,
                    "reset_glass_blur_seconds": reset_noise_seconds,
                    "create_env_seconds": create_seconds,
                    "load_init_states_seconds": state_load_seconds,
                    "set_init_state_seconds": set_state_seconds,
                    "settling_steps": 30,
                    "settling_step_seconds": settle_stats,
                    "settling_glass_blur_mean_seconds": statistics.fmean(settle_noise_seconds),
                    "measured_noop_steps": 10,
                    "noop_step_seconds": step_stats,
                    "noop_glass_blur_seconds": noise_stats,
                    "projected_700_step_env_seconds_mean": step_stats["mean_seconds"] * 700,
                    "projected_700_step_env_seconds_p95": step_stats["p95_seconds"] * 700,
                    "projected_700_step_glass_blur_seconds_mean": noise_stats["mean_seconds"] * 700,
                    "projected_runtime_env_seconds_mean": (
                        reset_seconds + set_state_seconds + 30 * settle_stats["mean_seconds"]
                        + 700 * step_stats["mean_seconds"]
                    ),
                })
                print(f"NOISE_PROBE_SUITE_OK suite={suite_name} level={level} "
                      f"step_mean_s={step_stats['mean_seconds']:.4f} "
                      f"glass_mean_s={noise_stats['mean_seconds']:.4f} "
                      f"projected_700_env_s={step_stats['mean_seconds']*700:.1f}", flush=True)
            finally:
                if env is not None:
                    env.close()
                noise_module.glass_blur = original_glass_blur
    finally:
        noise_module.glass_blur = original_glass_blur

    document = {
        "pbs_job_id": job,
        "host": host,
        "source_commit": status["source_commit"],
        "asset_commit": asset_commit,
        "episode_counted": False,
        "gpu_loaded": False,
        "measured_suites": len(results),
        "measured_noop_steps_per_suite": 10,
        "protocol_note": "Frozen manifest highest selected Sensor Noise glass-blur level per suite; 30 standard runtime settling no-ops plus 10 timed no-ops; no model or counted episode.",
        "results": results,
    }
    target = out / "noise_overhead.json"
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(target)
    print(f"NOISE_OVERHEAD_COMPLETE path={target} suites={len(results)} episode_counted=false", flush=True)


if __name__ == "__main__":
    main()
