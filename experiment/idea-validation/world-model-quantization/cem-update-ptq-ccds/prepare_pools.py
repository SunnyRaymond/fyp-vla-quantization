"""Collect fresh paired FP32 candidate/reference pools on a CCDS V100 allocation."""
import argparse
import json
from pathlib import Path
import time

from allocation_guard import require_allocation


def _load_old_fingerprints(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "cem-update-ptq-old-target-fingerprints-v1":
        raise ValueError("old fingerprint input has an unsupported schema")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("old fingerprint input has no records")
    values = {str(row.get("target_fingerprint")) for row in records if isinstance(row, dict)}
    indices = {int(row.get("dataset_index", -1)) for row in records if isinstance(row, dict)}
    if len(records) != 40 or len(values) != 40 or "None" in values or indices != set(range(2, 42)):
        raise ValueError("old fingerprint input must contain the 40 legacy targets at indices 2..41")
    return values


def _v100_provenance(allocation, runtime_identity):
    gpu = str(runtime_identity.get("gpu") or "").strip()
    if "v100" not in gpu.casefold():
        raise RuntimeError(f"CCDS reference collection requires a reported V100 GPU, got {gpu!r}")
    return {
        "independent_rerun": True,
        "scheduler": "slurm",
        "cluster": "CCDS-TC1",
        "gpu_model": gpu,
        "cuda_visible_devices": runtime_identity.get("cuda_visible_devices"),
        "allocation": allocation,
        "candidate_score_pairing": "same_fresh_fp32_collection",
        "historical_arrays_reused": False,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--old-fingerprints", type=Path, required=True)
    args = p.parse_args()

    allocation = require_allocation()
    oldprints = _load_old_fingerprints(args.old_fingerprints.resolve())
    import numpy as np
    import smoke_runner as smoke
    import screen_runner as screen

    started = time.monotonic()
    runtime = smoke._runtime(args.root)
    runtime_identity = smoke._checkpoint_identity(runtime)
    provenance = _v100_provenance(allocation, runtime_identity)
    targets = args.base / "targets"
    definitions = {"newcal": (42, 600000), "newdev": (46, 700000)}
    for split, (offset, namespace) in definitions.items():
        screen.SPLIT_SIZES[split] = 4
        screen.SPLIT_NAMESPACES[split] = namespace
        screen.SPLIT_DATASET_OFFSETS[split] = offset

    groups = smoke._linear_groups(runtime["model"])
    label, metadata = screen._configure_mode(runtime, "FP32", None, groups)
    records = {}
    for split in definitions:
        manifest = dict(screen._prepare_targets(runtime, targets, split))
        manifest["dataset_index_strategy"] = "new CAL 42-45; new DEV 46-49; old 0-41 excluded"
        manifest["old_target_fingerprint_count"] = len(oldprints)
        manifest["hardware_provenance"] = provenance
        screen._atomic_json(targets / f"episode_manifest_{split}.json", manifest)
        _, _, episodes = screen._load_targets(targets, split, runtime)
        newprints = {str(row["target_fingerprint"]) for row in episodes}
        if oldprints.intersection(newprints):
            raise RuntimeError(f"{split} target fingerprints overlap the 40 legacy targets")
        pools, rows = [], []
        for episode in episodes:
            result, _, cases = screen._run_episode(
                runtime, episode, args.output / split / f"episode_{episode['local_index']:03d}",
                True, label, metadata
            )
            if result["status"] != "complete":
                raise RuntimeError(f"{split}/{episode['local_index']} did not complete: {result}")
            chosen = [case for case in cases if case["mpc_point"] == 0]
            if sorted(case["cem_iteration"] for case in chosen) != [1, 5]:
                raise RuntimeError(f"{split}/{episode['local_index']} is not the frozen CEM 1/5 pair")
            pools.extend(chosen)
            rows.append({key: result[key] for key in (
                "episode_id", "dataset_index", "env_seed", "cem_seed",
                "target_fingerprint", "success", "executed_env_steps"
            )})
            with np.load(args.output / split / f"episode_{episode['local_index']:03d}" / "trajectory.npz") as arrays:
                distance = float(np.linalg.norm(arrays["states"][-1, :2] - episode["target"]["state_g"][0, :2]))
                if abs(distance - result["goal_xy_distance"]) >= 1e-8:
                    raise RuntimeError(f"{split}/{episode['local_index']} trajectory distance mismatch")
                if bool(distance < 4.5) != result["success"]:
                    raise RuntimeError(f"{split}/{episode['local_index']} success mismatch")
        screen._atomic_pickle(
            args.base / "pools" / f"{split}.pkl",
            {
                "schema": "cem-update-ptq-ccds-workload-v1",
                "split": split,
                "cases": pools,
                "planner": screen._protocol_config(),
                "runtime_identity": runtime_identity,
                "hardware_provenance": provenance,
                "old_target_fingerprint_count": len(oldprints),
            },
        )
        records[split] = {"episode_count": len(rows), "pool_count": len(pools), "episodes": rows}
    screen._atomic_json(
        args.output / "summary.json",
        {
            "schema": "cem-update-ptq-ccds-reference-v1",
            "status": "complete",
            "allocation": allocation,
            "hardware_provenance": provenance,
            "splits": records,
            "old_target_overlap": 0,
            "elapsed_seconds": time.monotonic() - started,
            "claim": "fresh paired FP32 candidates and reference scores only; no method success evidence",
        },
    )


if __name__ == "__main__":
    main()

