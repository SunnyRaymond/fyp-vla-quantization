"""Export a compact, provenance-checked Rolling Ball policy bundle in PBS."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import subprocess
import sys
import tarfile
import urllib.request


LEWM_COMMIT = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"
STABLE_WM_COMMIT = "10c26dbd5677083fa31dba69eb738b973845e9a4"
REFLEXBENCH_COMMIT = "8bb931485093c6d98f8729774ad01bf824964e16"
DATASET_COMMIT = "9295b6e9878609a992047f0b8b65421a493299e7"
TRAIN_JOB = "25578999.pbs101"
PLANNER_JOB = "25579057.pbs101"


def require_pbs_cpu() -> tuple[str, str]:
    job_id = os.environ.get("PBS_JOBID", "").strip()
    nodefile_name = os.environ.get("PBS_NODEFILE", "").strip()
    if not job_id or not nodefile_name:
        raise RuntimeError("A genuine PBS allocation is required (PBS_JOBID/PBS_NODEFILE missing)")
    nodefile = Path(nodefile_name)
    if not nodefile.is_file():
        raise RuntimeError("PBS_NODEFILE is not readable")
    host = socket.gethostname().split(".")[0].lower()
    if any(tag in host for tag in ("login", "head", "submit")):
        raise RuntimeError(f"Refusing non-compute host {host}")
    allocated = {line.strip().split(".")[0].lower() for line in nodefile.read_text().splitlines() if line.strip()}
    if host not in allocated:
        raise RuntimeError(f"Current host {host} is absent from PBS_NODEFILE")
    return job_id, host


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def safe_extract_reflexbench(archive: Path, destination: Path) -> None:
    """Extract the pinned codeload tar without following links or escaping target."""
    destination.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        roots = {PurePosixPath(member.name).parts[0] for member in members if PurePosixPath(member.name).parts}
        if len(roots) != 1:
            raise RuntimeError("Pinned ReflexBench archive does not have one top-level directory")
        strip_root = next(iter(roots))
        for member in members:
            parts = PurePosixPath(member.name).parts
            if not parts or PurePosixPath(member.name).is_absolute() or ".." in parts:
                raise RuntimeError(f"Unsafe ReflexBench archive path: {member.name!r}")
            if parts[0] != strip_root:
                raise RuntimeError("ReflexBench archive changed its top-level prefix")
            if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                raise RuntimeError(f"Unsupported link or member in pinned source archive: {member.name!r}")
            relative = parts[1:]
            if not relative:
                continue
            target = destination.joinpath(*relative)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise RuntimeError(f"ReflexBench archive path escapes target: {member.name!r}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Could not read ReflexBench source member {member.name!r}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(member.mode & 0o755)
    if not (destination / "scripts" / "evaluation" / "eval.py").is_file():
        raise RuntimeError("Pinned ReflexBench archive is missing scripts/evaluation/eval.py")
    if not (destination / "source" / "reflexbench" / "reflexbench").is_dir():
        raise RuntimeError("Pinned ReflexBench archive is missing the extension source tree")
    write_json(destination / "PINNED.json", {
        "repository": "LxRoboticsLab/ReflexBench",
        "commit": REFLEXBENCH_COMMIT,
        "archive": f"https://codeload.github.com/LxRoboticsLab/ReflexBench/tar.gz/{REFLEXBENCH_COMMIT}",
        "extracted_in": "genuine PBS CPU allocation",
    })


def copy_source_tree(source: Path, destination: Path, repository: str, commit: str) -> None:
    pin = read_json(source / "PINNED.json")
    if pin.get("repository") != repository or pin.get("commit") != commit:
        raise RuntimeError(f"Source pin mismatch at {source}")
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".mypy_cache", "*.pyc"),
    )


def copy_if_present(source: Path, destination: Path, copied: list[str], missing: list[str]) -> None:
    if source.is_file():
        shutil.copy2(source, destination)
        copied.append(source.name)
    else:
        missing.append(source.name)


def package_versions(training_summary: dict) -> dict:
    names = (
        "torch", "torchvision", "numpy", "hydra-core", "omegaconf",
        "stable-pretraining", "stable-worldmodel", "gymnasium", "pillow", "lightning",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version.split()[0],
        "policy_environment_observed_during_export": versions,
        "training_environment_recorded_in_checkpoint_summary": training_summary.get("source", {}).get("packages", {}),
        "note": "Observed package versions aid recreation; they do not certify rented-host compatibility.",
    }


def bundle_tree_size(root: Path) -> tuple[int, int]:
    total = count = 0
    for directory, _, names in os.walk(root):
        for name in names:
            path = Path(directory) / name
            total += path.stat().st_size
            count += 1
    return total, count


def archive_bundle(bundle: Path, archive: Path) -> int:
    partial = archive.with_name(archive.name + ".partial")
    if archive.exists() or partial.exists():
        raise RuntimeError(f"Refusing to overwrite bundle archive or partial: {archive}")
    with tarfile.open(partial, "w:gz", compresslevel=6) as tar:
        for path in sorted(bundle.rglob("*")):
            tar.add(path, arcname=(Path(bundle.name) / path.relative_to(bundle)).as_posix(), recursive=False)
    partial.replace(archive)
    return archive.stat().st_size


def export(args: argparse.Namespace, job_id: str, host: str) -> dict:
    import numpy as np
    import torch

    root = args.root.resolve()
    code_root = args.code_root.resolve()
    output = args.output.resolve()
    bundles_root = (root / "bundles").resolve()
    if output == bundles_root or not output.is_relative_to(bundles_root):
        raise RuntimeError("Bundle output must be a dedicated path below root/bundles/")
    archive_path = output.with_name(output.name + ".tar.gz")
    if archive_path.exists() or archive_path.with_name(archive_path.name + ".partial").exists():
        raise RuntimeError(f"Refusing to overwrite bundle archive or partial: {archive_path}")
    if not root.is_dir() or not code_root.is_dir():
        raise FileNotFoundError("Task root and source snapshot must exist")

    train_dir = root / "runs" / TRAIN_JOB
    planner_dir = root / "runs" / PLANNER_JOB
    train_summary = read_json(train_dir / "summary.json")
    planner_summary = read_json(planner_dir / "planner_smoke.json")
    for run_dir, label in ((train_dir, "training"), (planner_dir, "planner")):
        for name in ("runner_exit_status.txt", "wrapper_exit_status.txt"):
            if (run_dir / name).read_text(encoding="utf-8").strip() != "0":
                raise RuntimeError(f"{label} gate {name} is not zero")
    if not (train_summary.get("status") == "PASS_epoch100"
            and train_summary.get("job_id") == TRAIN_JOB
            and train_summary.get("completed_epochs") == 100
            and train_summary.get("smoke_only") is False
            and train_summary.get("baseline_checkpoint") == "last.ckpt (epoch 100)"):
        raise RuntimeError("Formal epoch-100 training summary gate failed")
    if (planner_summary.get("status") != "PASS_protocol_and_planner_shape"
            or planner_summary.get("job_id") != PLANNER_JOB
            or planner_summary.get("checkpoint_epoch") != 100
            or planner_summary.get("closed_loop_evaluated") is not False):
        raise RuntimeError("The required offline CEM integration gate is missing or failed")
    checkpoint_path = train_dir / "last.ckpt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("format") != "vanilla_lewm_raw_model_state_v1" or int(checkpoint.get("epoch", -1)) != 100:
        raise RuntimeError("Checkpoint is not the formal epoch-100 vanilla LeWM artifact")
    if checkpoint.get("source", {}).get("lewm", {}).get("commit") != LEWM_COMMIT:
        raise RuntimeError("Checkpoint LeWM pin differs from the frozen source")
    if checkpoint.get("source", {}).get("stable_worldmodel", {}).get("commit") != STABLE_WM_COMMIT:
        raise RuntimeError("Checkpoint StableWorldModel pin differs from the frozen source")
    if checkpoint.get("source", {}).get("dataset") != f"cyx337/ReflexBench_dataset@{DATASET_COMMIT}":
        raise RuntimeError("Checkpoint dataset pin differs from the frozen source")

    data_dir = root / "data"
    data_summary = read_json(data_dir / "summary.json")
    split = read_json(data_dir / "split.json")
    train_ids = [int(value) for value in split["train_episode_ids"]]
    val_ids = [int(value) for value in split["validation_episode_ids"]]
    if (data_summary.get("status") != "PASS" or data_summary.get("task_index") != 3
            or len(train_ids) != 180 or len(val_ids) != 20 or set(train_ids) & set(val_ids)):
        raise RuntimeError("Prepared Rolling Ball data/split gate failed")
    ckpt_split = checkpoint.get("split", {})
    if ([int(x) for x in ckpt_split.get("train_episode_ids", [])] != train_ids
            or [int(x) for x in ckpt_split.get("validation_episode_ids", [])] != val_ids):
        raise RuntimeError("Checkpoint split does not match data/split.json")
    if checkpoint.get("source", {}).get("job_id") != train_summary.get("job_id"):
        raise RuntimeError("Checkpoint training job ID does not match its formal summary")
    frame_meta = data_summary.get("arrays", {}).get("frames", {})
    if frame_meta.get("dtype") != "uint8" or frame_meta.get("shape", [None])[1:] != [224, 224, 3]:
        raise RuntimeError("Prepared fixed-camera RGB array metadata is not 224x224 uint8")
    source = checkpoint["source"]
    if (source.get("lewm", {}).get("commit") != LEWM_COMMIT
            or source.get("stable_worldmodel", {}).get("commit") != STABLE_WM_COMMIT):
        raise RuntimeError("Checkpoint source provenance is inconsistent")

    goal_refs = checkpoint.get("train_goal_bank_terminal_frames")
    if not isinstance(goal_refs, list) or len(goal_refs) != 180:
        raise RuntimeError("Checkpoint must carry the ordered 180-frame training goal references")
    copied_train_refs = read_json(train_dir / "train_goal_bank_indices.json").get("indices")
    if copied_train_refs != goal_refs:
        raise RuntimeError("Checkpoint goal references differ from the formal training run record")
    if [int(ref["episode_id"]) for ref in goal_refs] != sorted(train_ids):
        raise RuntimeError("Checkpoint goal reference order differs from sorted train episode IDs")

    frames = np.load(data_dir / "frames.npy", mmap_mode="r", allow_pickle=False)
    episode_ids = np.load(data_dir / "episode_ids.npy", mmap_mode="r", allow_pickle=False)
    frame_indices = np.load(data_dir / "frame_indices.npy", mmap_mode="r", allow_pickle=False)
    state_summary = data_summary.get("arrays", {}).get("state", {})
    state_path = data_dir / "state.npy"
    if (frames.dtype != np.uint8 or frames.shape[1:] != (224, 224, 3)
            or episode_ids.shape != (len(frames),) or frame_indices.shape != (len(frames),)):
        raise RuntimeError("Prepared frame/provenance arrays do not align")
    if not state_path.is_file() or state_summary.get("shape") != [len(frames), 8] or state_summary.get("dtype") != "float32":
        raise RuntimeError("Compact handoff requires the aligned 8-D observation.state array")
    states = np.load(state_path, mmap_mode="r", allow_pickle=False)
    if states.shape != (len(frames), 8) or states.dtype != np.float32:
        raise RuntimeError("Prepared state.npy shape/dtype differs from its summary")

    goal_rows = []
    for ref in goal_refs:
        row = int(ref["dataset_row"])
        eid = int(ref["episode_id"])
        if not 0 <= row < len(frames) or int(episode_ids[row]) != eid or int(frame_indices[row]) != 25:
            raise RuntimeError(f"Checkpoint goal reference is not the terminal train frame: {ref}")
        goal_rows.append(row)
    val_rows = np.flatnonzero(np.isin(episode_ids, np.asarray(val_ids, dtype=episode_ids.dtype)))
    if not len(val_rows):
        raise RuntimeError("No validation frames are available for the compact probe")
    order = np.lexsort((np.asarray(frame_indices[val_rows]), np.asarray(episode_ids[val_rows])))
    validation_row = int(val_rows[order[0]])
    validation_episode = int(episode_ids[validation_row])
    validation_frame_index = int(frame_indices[validation_row])
    if validation_episode not in set(val_ids):
        raise RuntimeError("Selected validation probe is outside the held-out split")

    # One writer: lock and partial stage make retries explicit and the final rename atomic.
    lock = output.with_name(output.name + ".export-lock")
    stage = output.with_name(output.name + ".partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"Another exporter owns {lock}") from exc
    try:
        if output.exists() or stage.exists():
            raise RuntimeError("Refusing to overwrite a bundle or partial export")
        stage.mkdir()
        (stage / "data").mkdir()
        (stage / "checkpoint").mkdir()
        copy_source_tree(root / "upstream_lewm", stage / "upstream_lewm", "lucas-maes/le-wm", LEWM_COMMIT)
        copy_source_tree(root / "upstream_stablewm", stage / "upstream_stablewm", "galilai-group/stable-worldmodel", STABLE_WM_COMMIT)
        shutil.copy2(checkpoint_path, stage / "checkpoint" / "last.ckpt")
        shutil.copy2(train_dir / "summary.json", stage / "checkpoint" / "summary.json")
        shutil.copy2(planner_dir / "planner_smoke.json", stage / "offline_planner_smoke.json")
        shutil.copy2(data_dir / "summary.json", stage / "data" / "summary.json")
        shutil.copy2(data_dir / "split.json", stage / "data" / "split.json")
        shutil.copy2(train_dir / "train_goal_bank_indices.json", stage / "data" / "train_goal_bank_indices.json")
        goal_array = np.asarray(frames[goal_rows], dtype=np.uint8)
        if goal_array.shape != (180, 224, 224, 3):
            raise RuntimeError(f"Goal frame export shape is unexpected: {goal_array.shape}")
        np.save(stage / "data" / "goal_frames.npy", goal_array, allow_pickle=False)
        write_json(stage / "data" / "goal_refs.json", goal_refs)
        probe_frame = np.asarray(frames[validation_row], dtype=np.uint8)
        probe_state = np.asarray(states[validation_row], dtype=np.float32)
        np.savez(stage / "data" / "validation_probe.npz", frame=probe_frame, state=probe_state)
        validation_meta = {
            "episode_id": validation_episode,
            "dataset_row": validation_row,
            "frame_index": validation_frame_index,
            "camera": "observation.images.fixed_cam",
            "state_source": "observation.state",
            "state_shape": [8],
            "state_dtype": "float32",
            "split": "validation",
            "dataset": f"cyx337/ReflexBench_dataset@{DATASET_COMMIT}",
        }
        write_json(stage / "data" / "validation_probe.json", validation_meta)

        code_copied, code_pending = [], []
        for name in (
            "policy_server.py", "run_rented.sh", "BUNDLE_ADAPTER.md", "PLANNER_FREEZE.json",
            "TRAIN_FREEZE.json", "EVAL_FREEZE.json", "eval_rolling.py", "AUTODL_PREP.md",
            "RENTAL_HANDOFF.zh.md", "verify_bundle.py", "run_bundle_smoke.pbs",
        ):
            copy_if_present(code_root / name, stage / name, code_copied, code_pending)
        if "policy_server.py" not in code_copied or "run_rented.sh" not in code_copied:
            raise RuntimeError("Bundle requires policy_server.py and run_rented.sh")
        subprocess.run(["bash", "-n", str(stage / "run_rented.sh")], check=True)

        archive = stage / "reflexbench-source.tar.gz"
        archive_url = f"https://codeload.github.com/LxRoboticsLab/ReflexBench/tar.gz/{REFLEXBENCH_COMMIT}"
        with urllib.request.urlopen(archive_url, timeout=120) as response, archive.open("wb") as output_file:
            shutil.copyfileobj(response, output_file)
        safe_extract_reflexbench(archive, stage / "reflexbench")
        archive.unlink()

        versions = package_versions(train_summary)
        write_json(stage / "package_versions.json", versions)
        source_pins = {
            "lewm": f"lucas-maes/le-wm@{LEWM_COMMIT}",
            "stable_worldmodel": f"galilai-group/stable-worldmodel@{STABLE_WM_COMMIT}",
            "reflexbench": f"LxRoboticsLab/ReflexBench@{REFLEXBENCH_COMMIT}",
            "dataset": f"cyx337/ReflexBench_dataset@{DATASET_COMMIT}",
        }
        manifest = {
            "status": "PASS",
            "schema_version": 1,
            "task": "Rolling Ball Interception",
            "export_job_id": job_id,
            "export_host": host,
            "source": source_pins,
            "checkpoint": {"path": "checkpoint/last.ckpt", "training_job_id": train_summary["job_id"], "epoch": 100},
            "offline_planner_gate": {
                "path": "offline_planner_smoke.json",
                "job_id": planner_summary["job_id"],
                "status": planner_summary["status"],
                "closed_loop_evaluated": False,
            },
            "dataset": {
                "task_index": 3,
                "train_episode_count": 180,
                "validation_episode_count": 20,
                "summary_path": "data/summary.json",
                "split_path": "data/split.json",
            },
            "goal_bank": {
                "frames_path": "data/goal_frames.npy",
                "refs_path": "data/goal_refs.json",
                "count": 180,
                "ordered_as": "checkpoint.train_goal_bank_terminal_frames",
                "frame_semantics": "last available fixed-camera observation (episode-local frame 25); success labels unverified",
            },
            "validation_probe": {
                "path": "data/validation_probe.npz",
                "metadata_path": "data/validation_probe.json",
                "episode_id": validation_episode,
                "dataset_row": validation_row,
                "frame_index": validation_frame_index,
            },
            "policy_files_included": code_copied,
            "policy_files_pending_at_export": code_pending,
            "evaluation_controls": "complete" if "eval_rolling.py" in code_copied and "EVAL_FREEZE.json" in code_copied else "pending_if_not_included",
        }
        write_json(stage / "BUNDLE.json", manifest)
        export_summary = {
            "status": "PASS",
            "bundle_path": str(output),
            "export_job_id": job_id,
            "checkpoint": "checkpoint/last.ckpt (epoch 100)",
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "goal_frame_count": 180,
            "validation_probe": validation_meta,
            "closed_loop_evaluated": False,
            "policy_files_pending_at_export": code_pending,
        }
        summary_path = stage / "EXPORT_SUMMARY.json"
        write_json(summary_path, export_summary)
        size, count = bundle_tree_size(stage)
        export_summary["bundle_bytes"] = size
        export_summary["bundle_file_count"] = count
        write_json(summary_path, export_summary)
        os.replace(stage, output)
        archive_bytes = archive_bundle(output, archive_path)
        bundle_bytes, bundle_files = bundle_tree_size(output)
        export_summary["bundle_path"] = str(output)
        export_summary["bundle_bytes"] = bundle_bytes
        export_summary["bundle_file_count"] = bundle_files
        export_summary["archive_path"] = str(archive_path)
        export_summary["archive_bytes"] = archive_bytes
        if args.report is not None:
            write_json(args.report.resolve(), export_summary)
        return export_summary
    finally:
        try:
            lock.rmdir()
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    job_id, host = require_pbs_cpu()
    result = export(args, job_id, host)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
