"""Memory-bounded equivalent of the official Two-Rooms image renderer.

All invocation is guarded by PBS allocation checks. The implementation calls
the pinned upstream WallDataset rendering methods and stores their outputs as
the .npy layout supported by OfflineWallDataset.load_np().
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path


def require_compute_allocation() -> tuple[str, str]:
    job_id = os.environ.get("PBS_JOBID", "")
    nodefile = os.environ.get("PBS_NODEFILE", "")
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise SystemExit("Refusing data work without PBS_JOBID and PBS_NODEFILE")
    host = socket.gethostname().split(".")[0].lower()
    if any(word in host for word in ("login", "head", "submit")):
        raise SystemExit(f"Refusing probable login node: {host}")
    nodes = {line.split()[0].split(".")[0].lower() for line in Path(nodefile).read_text().splitlines() if line.strip()}
    if host not in nodes:
        raise SystemExit(f"Current host {host} is not listed in PBS_NODEFILE")
    return job_id, host


def main() -> int:
    job_id, host = require_compute_allocation()
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-seconds", type=float, default=7200)
    args = parser.parse_args()

    import numpy as np
    import torch
    import yaml

    sys.path.insert(0, str(args.repo))
    from pldm_envs.wall.data.wall import WallDataset, WallDatasetConfig
    from pldm_envs.wall.save_wall_ds import update_config_from_yaml

    if args.output.exists():
        metadata_path = args.output / "render_metadata.json"
        if metadata_path.is_file():
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing.get("source_commit") == args.commit and existing.get("input_name") == args.input.name:
                print(f"RENDER_REUSE {args.output}", flush=True)
                return 0
        raise SystemExit(f"Refusing to overwrite existing render directory: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage = args.output.with_name(args.output.name + f".stage-{job_id}")
    if stage.exists():
        raise SystemExit(f"Refusing to reuse incomplete render directory: {stage}")

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    with np.load(args.input, allow_pickle=True) as packed:
        required = {"actions", "locations", "terminals"}
        absent = sorted(required.difference(packed.files))
        if absent:
            raise RuntimeError(f"Offline archive misses required arrays: {absent}")
        actions = packed["actions"]
        locations = packed["locations"]
        terminals = packed["terminals"]

    terminal_indices = np.where(terminals)[0]
    lengths = np.unique(terminal_indices[1:] - terminal_indices[:-1])
    if len(lengths) != 1:
        raise RuntimeError(f"Official loader requires one trajectory length; got {lengths.tolist()}")
    trajectory_length = int(lengths[0])
    trajectory_count = int(len(terminal_indices))
    frame_count = int(len(locations))
    if trajectory_count * trajectory_length != frame_count:
        raise RuntimeError(
            f"Official reshape mismatch: {trajectory_count} trajectories x {trajectory_length} steps != {frame_count} frames"
        )
    if actions.shape[0] != frame_count or terminals.shape[0] != frame_count:
        raise RuntimeError("actions/terminals/locations have different frame counts")

    with args.config.open("r", encoding="utf-8") as source:
        yaml_config = yaml.safe_load(source)
    dataset_config = update_config_from_yaml(WallDatasetConfig, yaml_config)

    estimated_bytes = (
        frame_count * 2 * int(dataset_config.img_size) * int(dataset_config.img_size) * np.dtype(np.uint8).itemsize
        + trajectory_count * (trajectory_length - 1) * int(np.prod(actions.shape[1:])) * actions.dtype.itemsize
        + trajectory_count * (trajectory_length - 1) * int(np.prod(locations.shape[1:])) * locations.dtype.itemsize
    )
    free_bytes = shutil.disk_usage(args.output.parent).free
    print(f"render_estimated_bytes={estimated_bytes} filesystem_free_bytes={free_bytes}", flush=True)
    if estimated_bytes > free_bytes * 0.95:
        raise RuntimeError(f"Estimated memmap output {estimated_bytes} bytes exceeds 95% of filesystem free space {free_bytes}")

    stage.mkdir()
    dataset = WallDataset(dataset_config)
    wall_info = dataset.sample_walls()
    walls = dataset.render_walls(*wall_info)[0]
    if walls.ndim != 2:
        raise RuntimeError(f"Expected one rendered wall image with 2 dimensions, got {tuple(walls.shape)}")

    image_size = int(dataset_config.img_size)
    state_path = stage / "states.npy"
    state_map = np.lib.format.open_memmap(
        state_path,
        mode="w+",
        dtype=np.uint8,
        shape=(trajectory_count, trajectory_length, 2, image_size, image_size),
    )
    actions_map = np.lib.format.open_memmap(
        stage / "actions.npy",
        mode="w+",
        dtype=actions.dtype,
        shape=(trajectory_count, trajectory_length - 1, *actions.shape[1:]),
    )
    locations_map = np.lib.format.open_memmap(
        stage / "locations.npy",
        mode="w+",
        dtype=locations.dtype,
        shape=(trajectory_count, trajectory_length - 1, *locations.shape[1:]),
    )
    actions_map[...] = actions.reshape(trajectory_count, trajectory_length, *actions.shape[1:])[:, :-1]
    locations_map[...] = locations.reshape(trajectory_count, trajectory_length, *locations.shape[1:])[:, :-1]

    flat_states = state_map.reshape(frame_count, 2, image_size, image_size)
    render_started = time.monotonic()
    projection_checked = False
    for start in range(0, frame_count, args.batch_size):
        end = min(start + args.batch_size, frame_count)
        rendered = dataset.render_location(torch.from_numpy(locations[start:end]))
        if rendered.ndim != 3:
            raise RuntimeError(f"Expected rendered locations [N,H,W], got {tuple(rendered.shape)}")
        if rendered.shape[1:] != (image_size, image_size):
            raise RuntimeError(f"Rendered image shape changed: {tuple(rendered.shape)}")
        location_images = rendered.detach().cpu().numpy()
        if location_images.dtype != np.uint8 or walls.dtype != torch.uint8:
            raise RuntimeError(f"Unexpected upstream renderer dtype: locations={location_images.dtype}, walls={walls.dtype}")
        wall_image = walls.detach().cpu().numpy()
        flat_states[start:end, 0] = location_images
        flat_states[start:end, 1] = wall_image[None, :, :]
        if not projection_checked and end >= min(frame_count, args.batch_size * 8):
            elapsed = time.monotonic() - render_started
            projected_seconds = elapsed * frame_count / end
            print(
                f"render_projection_seconds={projected_seconds:.1f} available_seconds={args.max_seconds:.1f} "
                f"calibration_frames={end}/{frame_count}",
                flush=True,
            )
            projection_checked = True
            if projected_seconds > args.max_seconds * 0.9:
                raise RuntimeError(
                    f"Projected full render {projected_seconds:.1f}s exceeds the remaining CPU budget {args.max_seconds:.1f}s"
                )
        if start == 0:
            reference = dataset.render_location(torch.from_numpy(locations[: min(3, end)]))
            if not np.array_equal(flat_states[: len(reference), 0], reference.detach().cpu().numpy()):
                raise RuntimeError("Chunked rendering did not preserve official location pixels")
            if not np.array_equal(flat_states[: len(reference), 1], np.repeat(wall_image[None, :, :], len(reference), axis=0)):
                raise RuntimeError("Chunked rendering did not preserve official wall pixels")
        if start == 0 or end == frame_count or end % 100000 == 0:
            print(f"rendered_frames={end}/{frame_count}", flush=True)

    state_map.flush()
    actions_map.flush()
    locations_map.flush()
    metadata = {
        "source_commit": args.commit,
        "input_name": args.input.name,
        "input_bytes": args.input.stat().st_size,
        "render_method": "pinned upstream WallDataset.render_location/render_walls, chunked storage",
        "render_config": str(args.config),
        "trajectory_count": trajectory_count,
        "trajectory_length": trajectory_length,
        "frame_count": frame_count,
        "states_shape": list(state_map.shape),
        "states_dtype": str(state_map.dtype),
        "actions_shape": list(actions_map.shape),
        "actions_dtype": str(actions_map.dtype),
        "locations_shape": list(locations_map.shape),
        "locations_dtype": str(locations_map.dtype),
        "batch_size": args.batch_size,
        "estimated_output_bytes": estimated_bytes,
        "filesystem_free_bytes_before_render": free_bytes,
        "validation": "first up to 3 rendered location frames compared with direct upstream calls",
        "pbs_job_id": job_id,
        "host": host,
    }
    (stage / "render_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    del state_map, actions_map, locations_map, flat_states
    os.replace(stage, args.output)
    print(f"RENDER_PASS {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
