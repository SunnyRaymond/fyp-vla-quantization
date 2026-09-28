"""Guarded source/data/dependency preparation for the pinned PLDM Two-Rooms run."""
from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SOURCE_SHA = "1bd7e564ecd961205bc18b23067b19e9ca24ac90"
SOURCE_URL = f"https://codeload.github.com/vladisai/PLDM/tar.gz/{SOURCE_SHA}"
DRIVE_ID = "1NwR-ui-akIgR2xcoJHYiHjOk9U5bYCa0"
DATA_URL = f"https://drive.google.com/uc?id={DRIVE_ID}"
DATA_MEMBER = "good_quality_data.npz"
DATA_RAW_MEMBER = "good_quality_data_no_images.npz"
SHARED_PYTHON = Path("/scratch/users/ntu/yguo017/lewm-pusht-iteration/venv/bin/python")
CPU_LIMIT_SECONDS = 6300


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def assert_allocation() -> tuple[str, str]:
    job_id = os.environ.get("PBS_JOBID", "")
    nodefile = os.environ.get("PBS_NODEFILE", "")
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("PBS_JOBID and a valid PBS_NODEFILE are required")
    host = socket.gethostname().split(".")[0].lower()
    if any(word in host for word in ("login", "head", "submit")):
        raise RuntimeError(f"Refusing probable login node: {host}")
    nodes = {line.split()[0].split(".")[0].lower() for line in Path(nodefile).read_text().splitlines() if line.strip()}
    if host not in nodes:
        raise RuntimeError(f"Current host {host} is not in PBS_NODEFILE")
    return job_id, host


def atomic_json(path: Path, value: dict, job_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{job_id}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def download_atomic(url: str, destination: Path, job_id: str, deadline: float) -> dict:
    if destination.is_file() and destination.stat().st_size > 0:
        return {"path": str(destination), "bytes": destination.stat().st_size, "reused": True}
    lock = destination.with_suffix(destination.suffix + ".lock")
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Another writer already owns {destination}: {lock}") from exc
    os.close(lock_fd)
    partial = destination.with_name(destination.name + f".partial-{job_id}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    headers: dict[str, str] = {}
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "PLDM-reproduction-prep/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            headers = {
                key: response.headers.get(key, "")
                for key in ("Content-Length", "Last-Modified", "ETag", "Content-Type")
                if response.headers.get(key) is not None
            }
            total = 0
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError("CPU preparation self-time limit reached during download")
                block = response.read(4 * 1024 * 1024)
                if not block:
                    break
                output.write(block)
                total += len(block)
                if total and total % (128 * 1024 * 1024) < len(block):
                    print(f"downloaded {destination.name}: {total} bytes", flush=True)
            output.flush()
            os.fsync(output.fileno())
        if partial.stat().st_size == 0:
            raise RuntimeError(f"Downloaded empty file: {destination.name}")
        os.replace(partial, destination)
        return {"path": str(destination), "bytes": destination.stat().st_size, "headers": headers, "reused": False}
    finally:
        lock.unlink(missing_ok=True)


def safe_extract(archive: Path, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=False)
    target_abs = target.resolve()
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        for member in members:
            if member.issym() or member.islnk():
                raise RuntimeError(f"Links are not expected in the pinned source archive: {member.name}")
            candidate = (target / member.name).resolve()
            if candidate != target_abs and target_abs not in candidate.parents:
                raise RuntimeError(f"Unsafe path in official source archive: {member.name}")
        tf.extractall(target)
    roots = list(target.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise RuntimeError(f"Unexpected source archive layout: {[p.name for p in roots]}")
    return roots[0]


def source_return_report(train_path: Path, config_path: Path) -> dict:
    text = train_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    line = next((i for i, value in enumerate(lines, 1) if "if step - first_step == 5:" in value), None)
    outer_line = next((i for i, value in enumerate(lines, 1) if "if self.config.quick_debug or (step % 100 == 0):" in value), None)
    config_text = config_path.read_text(encoding="utf-8")
    quick_debug_false = bool(re.search(r"(?m)^quick_debug:\s*false\s*$", config_text))
    return {
        "debug_return_line": line,
        "debug_return_context": lines[max(0, line - 2):line + 1] if line else [],
        "outer_log_guard_line": outer_line,
        "outer_guard": "if self.config.quick_debug or (step % 100 == 0)",
        "config_quick_debug_false": quick_debug_false,
        "interpretation": "Source/config context recorded without making AST structure a preparation gate.",
    }


def requirement_comparison(requirements: Path, probe_python: Path, repo: Path) -> dict:
    lines = [line.strip() for line in requirements.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    pinned = []
    for line in lines:
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^;\s]+)", line)
        if match:
            pinned.append((match.group(1), match.group(2)))
    if not probe_python.is_file():
        return {"requirements_pinned": len(pinned), "probe_python": str(probe_python), "probe_error": "shared runtime python not found"}
    code = r'''import importlib.metadata as m, json, sys
from pathlib import Path
req=Path(sys.argv[1]).read_text().splitlines()
items=[]
for line in req:
 line=line.strip()
 if not line or line.startswith("#") or "==" not in line: continue
 name, wanted=line.split("==",1)
 try: got=m.version(name)
 except m.PackageNotFoundError: got=None
 items.append({"name":name,"required":wanted,"installed":got,"match":got==wanted})
print(json.dumps(items))'''
    result = subprocess.run([str(probe_python), "-c", code, str(requirements)], cwd=repo, text=True, capture_output=True, timeout=120)
    if result.returncode != 0:
        return {"requirements_pinned": len(pinned), "probe_python": str(probe_python), "probe_error": result.stderr[-3000:]}
    items = json.loads(result.stdout)
    exact = [item["name"] for item in items if item["match"]]
    mismatch = [item for item in items if not item["match"]]
    return {"requirements_pinned": len(items), "exact_matches": len(exact), "mismatches_or_missing": mismatch, "probe_python": str(probe_python)}


def compatibility_patch_for_error(repo: Path, error: str) -> dict | None:
    """Patch only the dataclass field named by Python 3.11's import error."""
    if "mutable default" not in error or "use default_factory" not in error:
        return None
    match = re.search(r"<class '([^']+)'> for field ([A-Za-z_]\w*)", error)
    if not match:
        return None
    class_name, field_name = match.groups()
    expected_constructor = class_name.rsplit(".", 1)[-1]
    line_pattern = re.compile(
        rf"^(\s*){re.escape(field_name)}(\s*:\s*[^=\n]+?\s*=\s*)"
        rf"{re.escape(expected_constructor)}\(\)(\s*(?:#.*)?)$"
    )
    candidates = []
    for source in list((repo / "pldm").rglob("*.py")) + list((repo / "pldm_envs").rglob("*.py")):
        try:
            lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
        except (OSError, UnicodeDecodeError):
            continue
        for index, line in enumerate(lines):
            candidate = line.rstrip("\r\n")
            if line_pattern.match(candidate):
                candidates.append((source, lines, index, candidate))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Could not uniquely locate Python 3.11 dataclass field {field_name} "
            f"with default {expected_constructor}(): found {len(candidates)} candidates"
        )
    source, lines, index, original = candidates[0]
    replacement_line = line_pattern.sub(
        r"\1" + field_name + r"\2field(default_factory=" + expected_constructor + r")\3",
        original,
        count=1,
    )
    lines[index] = replacement_line + ("\r\n" if lines[index].endswith("\r\n") else "\n")
    field_import_present = any(
        re.match(r"\s*from\s+dataclasses\s+import\s+.*\bfield\b", line)
        for line in lines
    )
    if not field_import_present:
        import_line = next(
            (i for i, line in enumerate(lines) if re.match(r"\s*from\s+dataclasses\s+import\s+", line)),
            None,
        )
        if import_line is not None:
            stripped = lines[import_line].rstrip("\r\n")
            newline = "\r\n" if lines[import_line].endswith("\r\n") else "\n"
            lines[import_line] = stripped + ", field" + newline
        else:
            future_end = max(
                (i for i, line in enumerate(lines) if re.match(r"\s*from\s+__future__\s+import\s+", line)),
                default=-1,
            )
            lines.insert(future_end + 1, "from dataclasses import field\n")
    temporary = source.with_name(source.name + ".compat-tmp")
    temporary.write_text("".join(lines), encoding="utf-8", newline="")
    os.replace(temporary, source)
    return {
        "path": str(source.relative_to(repo)),
        "field": field_name,
        "default_class": class_name,
        "original_line": original,
        "replacement_line": replacement_line,
        "reason": "Python 3.11 rejects this mutable dataclass instance default; default_factory preserves the same per-instance value.",
    }


def runtime_probe(probe_python: Path, repo: Path, overlays: list[Path]) -> dict:
    if not probe_python.is_file():
        return {"python": str(probe_python), "status": "MISSING"}
    code = r'''import importlib, importlib.metadata as m, importlib.util, json, os, sys, traceback
repo=sys.argv[1]
for path in reversed([p for p in os.environ.get("PLDM_OVERLAYS", "").split(os.pathsep) if p]): sys.path.insert(0, path)
sys.path.insert(0, repo)
names=["torch","torchvision","numpy","scipy","omegaconf","tqdm","matplotlib","yaml","wandb","gdown","bs4","soupsieve","zarr","numcodecs","asciitree","fasteners","gym"]
mods={n: bool(importlib.util.find_spec(n)) for n in names}
versions={}
for n in ["torch","torchvision","numpy","scipy","omegaconf","tqdm","matplotlib","PyYAML","wandb","gdown","beautifulsoup4","soupsieve","zarr","numcodecs","asciitree","fasteners","gym","gym-notices"]:
 try: versions[n]=m.version(n)
 except m.PackageNotFoundError: versions[n]=None
result={"python":sys.executable,"python_version":sys.version,"modules":mods,"versions":versions}
for n in ["gdown","bs4","soupsieve","zarr","gym"]:
 try: importlib.import_module(n); result[n+"_import"]="PASS"
 except Exception as e: result[n+"_import"]={"error_type":type(e).__name__,"error":str(e)}
for n in ["pldm.train","pldm.evaluation.evaluator","pldm_envs.wall.render_images"]:
 try: importlib.import_module(n); result[n]="PASS"
 except Exception as e: result[n]={"error_type":type(e).__name__,"error":str(e),"traceback":traceback.format_exc()[-3500:]}
print(json.dumps(result))'''
    env = os.environ.copy()
    env["PLDM_OVERLAYS"] = os.pathsep.join(map(str, overlays))
    env["PYTHONPATH"] = os.pathsep.join(map(str, overlays + [repo]))
    result = subprocess.run([str(probe_python), "-c", code, str(repo)], cwd=repo / "pldm", env=env, text=True, capture_output=True, timeout=900)
    if result.returncode != 0:
        return {"python": str(probe_python), "probe_error": result.stderr[-4000:], "stdout": result.stdout[-1500:]}
    return json.loads(result.stdout)


def actual_training_loader_report(probe_python: Path, repo: Path, overlays: list[Path], config_path: Path, dataset_path: Path, output_root: Path) -> dict:
    code = r'''import json,sys,time
repo,config_path,output_root,dataset_path=sys.argv[1:]
sys.path.insert(0,repo)
from pldm.train import TrainConfig
from pldm.data.utils import make_dataloader
from pldm_envs.wall.data.offline_wall import OfflineWallDataset
sys.argv=["train.py","--configs",config_path,"--values",
 "output_root="+output_root,"output_dir=tworooms-seqlen90-3M-seed101",
 "run_name=tworooms-seqlen90-3M-seed101",
 "data.offline_wall_config.offline_data_path="+dataset_path,
 "data.offline_wall_config.lazy_load=true","data.offline_wall_config.device=cpu","seed=101","wandb=false"]
cfg=TrainConfig.parse_from_command_line()
t0=time.monotonic()
dataset=OfflineWallDataset(cfg.data.offline_wall_config)
loader=make_dataloader(dataset,loader_config=cfg.data)
updates_per_epoch=len(loader)
epoch_indices=list(range(0,int(cfg.epochs)+1))
batch_start=time.monotonic()
first_batch=next(iter(loader))
print(json.dumps({"dataset_samples":len(dataset),"batch_size":loader.dataloader.batch_size,
 "drop_last":loader.dataloader.drop_last,"optimizer_updates_per_epoch":updates_per_epoch,
 "configured_epochs":cfg.epochs,"actual_loop_epoch_indices":epoch_indices,
 "actual_passes":len(epoch_indices),"expected_optimizer_updates":updates_per_epoch*len(epoch_indices),
 "training_n_steps":cfg.data.offline_wall_config.n_steps,"loader_build_and_normalizer_seconds":time.monotonic()-t0,
 "first_batch_seconds":time.monotonic()-batch_start,"first_batch_state_shape":list(first_batch.states.shape),
 "loader_type":type(loader.dataloader).__name__,"lazy_load":cfg.data.offline_wall_config.lazy_load,
 "loader_config_device_override_for_cpu_preflight":cfg.data.offline_wall_config.device,
 "dataset_runtime_device":str(dataset.device)}))'''
    env = os.environ.copy()
    env["PLDM_OVERLAYS"] = os.pathsep.join(map(str, overlays))
    env["PYTHONPATH"] = os.pathsep.join(map(str, overlays + [repo]))
    config_arg = str(config_path.relative_to(repo / "pldm"))
    result = subprocess.run([str(probe_python), "-c", code, str(repo), config_arg, str(output_root), str(dataset_path)], cwd=repo / "pldm", env=env, text=True, capture_output=True, timeout=900)
    if result.returncode != 0:
        return {"status": "FAIL", "exit_code": result.returncode, "error": result.stderr[-3000:], "stdout": result.stdout[-1000:]}
    output_lines = [line for line in result.stdout.splitlines() if line.strip()]
    value = json.loads(output_lines[-1]) if output_lines else {}
    value["status"] = "PASS" if value.get("actual_passes") == 3 and value.get("expected_optimizer_updates", 0) > 0 else "FAIL"
    return value


def compose_config(probe_python: Path, repo: Path, overlays: list[Path], config_path: Path, dataset_path: Path, output_root: Path) -> dict:
    code = r'''import json,sys
repo,config_path,output_root,dataset_path=sys.argv[1:]
sys.path.insert(0,repo)
from pldm.train import TrainConfig
def compose(flag):
 sys.argv=["train.py",flag,config_path,"--values",
  "output_root="+output_root,
  "output_dir=tworooms-seqlen90-3M-seed101",
  "run_name=tworooms-seqlen90-3M-seed101",
  "data.offline_wall_config.offline_data_path="+dataset_path,
  "data.offline_wall_config.lazy_load=true","seed=101","wandb=false"]
 cfg=TrainConfig.parse_from_command_line()
 planning=cfg.eval_cfg.wall_planning
 return {"env_name":cfg.env_name,"epochs":cfg.epochs,"seed":cfg.seed,"quick_debug":cfg.quick_debug,
 "n_steps":cfg.n_steps,"optimizer_type":str(cfg.optimizer_type),"objectives":[str(x) for x in cfg.objectives_l1.objectives],
 "backbone_arch":cfg.hjepa.level1.backbone.arch,"predictor_arch":cfg.hjepa.level1.predictor.predictor_arch,
 "base_lr":cfg.base_lr,"eval_mpcs":cfg.eval_mpcs,"eval_env_count":planning.n_envs,"eval_steps":planning.n_steps,
 "eval_levels":planning.levels,"eval_seed":planning.seed,"planner_type":str(planning.level1.planner_type),
 "medium_env_count":planning.medium.n_envs,"medium_steps":planning.medium.n_steps,
 "medium_max_plan_length":planning.medium.max_plan_length,"mppi_samples":planning.level1.mppi.num_samples,
 "offline_dataset_path":cfg.data.offline_wall_config.offline_data_path,
 "offline_use_offline":cfg.data.offline_wall_config.use_offline,
 "offline_batch_size":cfg.data.offline_wall_config.batch_size,"offline_image_size":cfg.data.offline_wall_config.img_size,
 "lazy_load":cfg.data.offline_wall_config.lazy_load,"offline_device":cfg.data.offline_wall_config.device,"wandb":cfg.wandb,
 "output_root":cfg.output_root,"output_dir":cfg.output_dir,"run_name":cfg.run_name,
 "probe_encoder":cfg.eval_cfg.probing.probe_encoder,"probe_predictions":cfg.eval_cfg.probing.probe_preds,
 "probe_wall":cfg.eval_cfg.probing.probe_wall}
plural=compose("--configs")
singular=compose("--config")
print(json.dumps({"plural":plural,"singular":singular,"same_composition":plural==singular}))'''
    env = os.environ.copy()
    env["PLDM_OVERLAYS"] = os.pathsep.join(map(str, overlays))
    env["PYTHONPATH"] = os.pathsep.join(map(str, overlays + [repo]))
    config_arg = str(config_path.relative_to(repo / "pldm"))
    dataset_path_arg = str(dataset_path)
    output_root_arg = str(output_root)
    result = subprocess.run([str(probe_python), "-c", code, str(repo), config_arg, output_root_arg, dataset_path_arg], cwd=repo / "pldm", env=env, text=True, capture_output=True, timeout=120)
    if result.returncode != 0:
        return {"status": "FAIL", "exit_code": result.returncode, "stderr": result.stderr[-3000:], "stdout": result.stdout[-1000:], "parser_flags_tested": ["--configs", "--config"]}
    output_lines = [line for line in result.stdout.splitlines() if line.strip()]
    alias_report = json.loads(output_lines[-1]) if output_lines else {}
    value = alias_report.get("plural", {})
    checks = {
        "env_name": value.get("env_name") == "wall",
        "epochs": value.get("epochs") == 2,
        "n_steps": value.get("n_steps") == 16,
        "seed": value.get("seed") == 101,
        "quick_debug": value.get("quick_debug") is False,
        "optimizer_type": value.get("optimizer_type") == "OptimizerType.Adam",
        "objectives": value.get("objectives") == ["ObjectiveType.VICReg", "ObjectiveType.IDM"],
        "backbone_arch": value.get("backbone_arch") == "impala",
        "predictor_arch": value.get("predictor_arch") == "rnnV2",
        "base_lr": value.get("base_lr") == 0.0007,
        "eval_mpcs": value.get("eval_mpcs") == 20,
        "eval_env_count": value.get("eval_env_count") == 100,
        "eval_steps": value.get("eval_steps") == 200,
        "eval_levels": value.get("eval_levels") == "medium",
        "eval_seed": value.get("eval_seed") == 42,
        "planner_type": value.get("planner_type") == "PlannerType.MPPI",
        "medium_env_count": value.get("medium_env_count") == 100,
        "medium_steps": value.get("medium_steps") == 200,
        "medium_max_plan_length": value.get("medium_max_plan_length") == 96,
        "mppi_samples": value.get("mppi_samples") == 2000,
        "offline_use_offline": value.get("offline_use_offline") is True,
        "offline_batch_size": value.get("offline_batch_size") == 64,
        "offline_image_size": value.get("offline_image_size") == 65,
        "lazy_load": value.get("lazy_load") is True,
        "offline_device": value.get("offline_device") == "cuda",
        "wandb": value.get("wandb") is False,
        "offline_dataset_path": value.get("offline_dataset_path") == dataset_path_arg,
        "output_root": value.get("output_root") == output_root_arg,
        "output_dir": value.get("output_dir") == "tworooms-seqlen90-3M-seed101",
        "run_name": value.get("run_name") == "tworooms-seqlen90-3M-seed101",
        "probe_encoder": value.get("probe_encoder") is True,
        "probe_predictions": value.get("probe_predictions") is True,
        "probe_wall": value.get("probe_wall") is True,
    }
    value.update({"status": "PASS" if all(checks.values()) and alias_report.get("same_composition") else "FAIL", "checks": checks, "parser_flags_tested": ["--configs", "--config"], "singular_plural_same_composition": alias_report.get("same_composition"), "singular_composition": alias_report.get("singular")})
    return value


def main() -> int:
    job_id, host = assert_allocation()
    started = time.monotonic()
    deadline = started + CPU_LIMIT_SECONDS
    root = Path(os.environ.get("PLDM_ROOT", "/scratch/users/ntu/yguo017/pldm-reproduction"))
    out = root / "PREPARATION.json"
    report: dict = {
        "status": "RUNNING",
        "cpu_job_id": job_id,
        "pbs_host": host,
        "started_utc": utc_now(),
        "source_sha": SOURCE_SHA,
        "remote_root": str(root),
        "resource_caps": {"cpu_walltime": "02:00:00", "gpu_walltime": "02:00:00"},
        "stages": {},
    }
    atomic_json(out, report, job_id)
    upstream = root / "upstream"
    archive = root / "source" / f"PLDM-{SOURCE_SHA}.tar.gz"
    try:
        if upstream.exists():
            identity_path = upstream / "source_identity.json"
            if not identity_path.is_file() or json.loads(identity_path.read_text()).get("source_sha") != SOURCE_SHA:
                raise RuntimeError("Existing upstream checkout lacks the exact frozen source identity; refusing overwrite")
            source_identity = json.loads(identity_path.read_text(encoding="utf-8"))
            source_download = {"path": str(archive), "bytes": archive.stat().st_size if archive.exists() else None, "reused": True}
        else:
            source_download = download_atomic(SOURCE_URL, archive, job_id, deadline)
            extract_path = root / "source" / f"extract-{job_id}"
            extracted_root = safe_extract(archive, extract_path)
            os.replace(extracted_root, upstream)
            shutil.rmtree(extract_path)
            source_identity = {
                "repository": "https://github.com/vladisai/PLDM",
                "source_sha": SOURCE_SHA,
                "acquisition": "GitHub codeload pinned commit archive",
                "archive_url": SOURCE_URL,
                "archive_path": str(archive),
                "archive_bytes": archive.stat().st_size,
                "acquired_utc": utc_now(),
                "pbs_job_id": job_id,
            }
            (upstream / "source_identity.json").write_text(json.dumps(source_identity, indent=2) + "\n", encoding="utf-8")
        train_path = upstream / "pldm" / "train.py"
        config_path = upstream / "pldm" / "configs" / "wall" / "icml" / "seqlen90_3M.yaml"
        if not train_path.is_file() or not config_path.is_file():
            raise RuntimeError("Pinned source archive does not contain the frozen official training entry/config")
        report["stages"]["source"] = {"status": "PASS", "identity": source_identity, "return_guard": source_return_report(train_path, config_path), "checkpoint_files_in_source_archive": [str(p.relative_to(upstream)) for p in upstream.rglob("*.ckpt")][:20]}

        python = Path(os.environ.get("PLDM_PYTHON", str(SHARED_PYTHON)))
        staged = root / "staged"
        staged_identity_path = root / "staged_source_identity.json"
        attempt_dir = root / "reports" / job_id
        attempt_dir.mkdir(parents=True, exist_ok=True)
        if staged.exists():
            prior_stage = attempt_dir / "staged-source-before-rebuild"
            if prior_stage.exists():
                raise RuntimeError(f"Prior staged-source archive path already exists: {prior_stage}")
            os.replace(staged, prior_stage)
        if staged_identity_path.exists():
            prior_identity = attempt_dir / "staged_source_identity-before-rebuild.json"
            if prior_identity.exists():
                raise RuntimeError(f"Prior staged identity archive path already exists: {prior_identity}")
            os.replace(staged_identity_path, prior_identity)
        shutil.copytree(upstream, staged)
        staged_identity = {
                "source_sha": SOURCE_SHA,
                "source_path": str(upstream),
                "staged_path": str(staged),
                "compatibility": "Python 3.11 dataclass default_factory only; changes are added only for import errors explicitly naming mutable defaults.",
                "patches": [],
                "algorithm_or_config_changes": [],
                "official_config_values_preserved": {
                    "config": "pldm/configs/wall/icml/seqlen90_3M.yaml",
                    "epochs": 2,
                    "quick_debug": False,
                    "seed": 101,
                    "n_steps": 16,
                    "eval_level": "medium",
                    "eval_envs": 100,
                    "eval_steps": 200,
                    "mppi_samples": 2000,
                    "eval_mpcs": 20,
                },
            }
        atomic_json(staged_identity_path, staged_identity, job_id)

        gdown_overlay = root / "runtime_overlay" / "gdown"
        html_overlay = root / "runtime_overlay" / "html-parser"
        zarr_overlay = root / "runtime_overlay" / "zarr-2.14.2-pinned"
        gym_overlay = Path("/scratch/users/ntu/yguo017/experiment/reproduction/lpwm/runtime-overlay-minimal")
        arm_overlay = root / "runtime_overlay" / "arm-pytorch-utilities-0.4.3"
        pytorch_seed_overlay = root / "runtime_overlay" / "pytorch-seed-0.2.0"
        statsmodels_overlay = root / "runtime_overlay" / "statsmodels-patsy-pinned"
        if zarr_overlay.is_dir():
            report["stages"]["zarr_overlay_install"] = {"status": "REUSED", "path": str(zarr_overlay)}
        else:
            stage = root / "runtime_overlay" / f"zarr-stage-{job_id}"
            if stage.exists():
                raise RuntimeError(f"Isolated zarr install stage already exists: {stage}")
            zarr_timeout = min(900, max(30, int(deadline - time.monotonic() - 1200)))
            install = subprocess.run(
                [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-warn-script-location", "--no-deps", "--target", str(stage), "zarr==2.14.2", "numcodecs==0.12.1", "asciitree==0.3.3", "fasteners==0.18"],
                cwd=root, text=True, capture_output=True, timeout=zarr_timeout,
            )
            report["stages"]["zarr_overlay_install"] = {
                "status": "PASS" if install.returncode == 0 else "FAIL",
                "exit_code": install.returncode,
                "stdout": install.stdout[-2500:],
                "stderr": install.stderr[-2500:],
                "path": str(zarr_overlay),
                "request": "isolated target with official pinned zarr==2.14.2, numcodecs==0.12.1, asciitree==0.3.3, fasteners==0.18; numpy remains inherited unless an observed ABI error requires an isolated pin",
            }
            if install.returncode == 0:
                if zarr_overlay.exists():
                    raise RuntimeError(f"Refusing to merge/overwrite isolated zarr overlay: {zarr_overlay}")
                os.replace(stage, zarr_overlay)
        numpy_overlay = root / "runtime_overlay" / "numpy-1.26.4"
        overlays = [path for path in (numpy_overlay, gdown_overlay, html_overlay, zarr_overlay, gym_overlay, arm_overlay, pytorch_seed_overlay, statsmodels_overlay) if path.is_dir()]
        report["stages"]["official_requirements"] = requirement_comparison(upstream / "requirements.txt", python, upstream)
        runtime = runtime_probe(python, staged, overlays)
        zarr_import_error = runtime.get("zarr_import", {})
        zarr_import_message = zarr_import_error.get("error", "") if isinstance(zarr_import_error, dict) else ""
        if any(token in zarr_import_message for token in ("numpy.core.multiarray failed to import", "numpy.dtype size changed", "module compiled against API version")):
            if numpy_overlay.is_dir():
                report["stages"]["numpy_abi_overlay"] = {"status": "REUSED", "path": str(numpy_overlay)}
            else:
                numpy_stage = root / "runtime_overlay" / f"numpy-stage-{job_id}"
                if numpy_stage.exists():
                    raise RuntimeError(f"Isolated NumPy install stage already exists: {numpy_stage}")
                numpy_install = subprocess.run(
                    [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-warn-script-location", "--no-deps", "--target", str(numpy_stage), "numpy==1.26.4"],
                    cwd=root, text=True, capture_output=True, timeout=min(900, max(30, int(deadline - time.monotonic() - 1200))),
                )
                report["stages"]["numpy_abi_overlay"] = {
                    "status": "PASS" if numpy_install.returncode == 0 else "FAIL",
                    "exit_code": numpy_install.returncode,
                    "stdout": numpy_install.stdout[-2000:],
                    "stderr": numpy_install.stderr[-2000:],
                    "path": str(numpy_overlay),
                    "reason": "installed only after the observed zarr/numcodecs import reported a NumPy C-ABI error",
                }
                if numpy_install.returncode == 0:
                    if numpy_overlay.exists():
                        raise RuntimeError(f"Refusing to merge/overwrite isolated NumPy overlay: {numpy_overlay}")
                    os.replace(numpy_stage, numpy_overlay)
            if report["stages"].get("numpy_abi_overlay", {}).get("status") in ("PASS", "REUSED"):
                overlays = [numpy_overlay, *overlays]
                runtime = runtime_probe(python, staged, overlays)
        report["stages"]["gym_overlay_reuse"] = {
            "status": "REUSED" if gym_overlay.is_dir() else "MISSING",
            "path": str(gym_overlay),
            "packages": {"gym": "0.23.1", "gym-notices": "0.0.8"},
            "reason": "read-only reuse of the already validated PLDM gym dependency overlay",
        }
        compatibility_patches = []
        for _ in range(20):
            failures = [
                value for name, value in runtime.items()
                if name in ("pldm.train", "pldm.evaluation.evaluator", "pldm_envs.wall.render_images")
                and value != "PASS"
            ]
            dataclass_error = next((item.get("error", "") for item in failures if isinstance(item, dict)), "")
            if not dataclass_error:
                break
            patch_record = compatibility_patch_for_error(staged, dataclass_error)
            if patch_record is None:
                break
            compatibility_patches.append(patch_record)
            staged_identity.setdefault("patches", []).append(patch_record)
            atomic_json(staged_identity_path, staged_identity, job_id)
            runtime = runtime_probe(python, staged, overlays)
        report["stages"]["staged_source_compatibility"] = {
            "status": "PASS" if all(runtime.get(n) == "PASS" for n in ("pldm.train", "pldm.evaluation.evaluator", "pldm_envs.wall.render_images")) else "PARTIAL",
            "source_identity_path": str(staged_identity_path),
            "patches": compatibility_patches,
            "remaining_import_errors": {name: value for name, value in runtime.items() if name.startswith("pldm.") and value != "PASS"},
        }
        report["stages"]["runtime_probe_before_data"] = runtime
        report["stages"]["config_composition"] = compose_config(
            python, staged, overlays, staged / "pldm" / "configs" / "wall" / "icml" / "seqlen90_3M.yaml",
            root / "datasets" / "good_quality_data_memmap",
            root / "checkpoints",
        )

        data_archive = root / "datasets" / "wall_data.tar.gz"
        if time.monotonic() < deadline - 600:
            report["stages"]["data_archive"] = {"status": "RUNNING", "official_drive_file_id": DRIVE_ID}
            atomic_json(out, report, job_id)
            gdown_present = bool(runtime.get("modules", {}).get("gdown"))
            if not gdown_present and not gdown_overlay.is_dir():
                stage = root / "runtime_overlay" / f"gdown-stage-{job_id}"
                if not stage.exists() and time.monotonic() < deadline - 900:
                    install = subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-warn-script-location", "--no-deps", "--target", str(stage), "gdown==5.2.0"], cwd=root, text=True, capture_output=True, timeout=min(900, max(30, int(deadline-time.monotonic()-600))) if time.monotonic() < deadline-600 else 30)
                    report["stages"]["gdown_overlay_install"] = {"exit_code": install.returncode, "stdout": install.stdout[-2500:], "stderr": install.stderr[-2500:], "path": str(stage)}
                    if install.returncode == 0:
                        gdown_overlay.parent.mkdir(parents=True, exist_ok=True)
                        if gdown_overlay.exists():
                            raise RuntimeError("gdown overlay target already exists; refusing merge/overwrite")
                        os.replace(stage, gdown_overlay)
                else:
                    report["stages"]["gdown_overlay_install"] = {"status": "SKIPPED", "reason": "stage exists or CPU deadline near"}
            elif gdown_overlay.is_dir():
                report["stages"]["gdown_overlay_install"] = {"status": "REUSED", "path": str(gdown_overlay), "version": "5.2.0"}
            if not html_overlay.is_dir():
                stage = root / "runtime_overlay" / f"html-parser-stage-{job_id}"
                if not stage.exists() and time.monotonic() < deadline - 900:
                    install_timeout = min(900, max(30, int(deadline - time.monotonic() - 600)))
                    install = subprocess.run(
                        [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-warn-script-location", "--no-deps", "--target", str(stage), "beautifulsoup4==4.11.2", "soupsieve==2.4"],
                        cwd=root, text=True, capture_output=True, timeout=install_timeout,
                    )
                    report["stages"]["html_parser_overlay_install"] = {"exit_code": install.returncode, "stdout": install.stdout[-2500:], "stderr": install.stderr[-2500:], "path": str(stage), "packages": {"beautifulsoup4": "4.11.2", "soupsieve": "2.4"}}
                    if install.returncode == 0:
                        if html_overlay.exists():
                            raise RuntimeError("html-parser overlay target already exists; refusing merge/overwrite")
                        os.replace(stage, html_overlay)
                else:
                    report["stages"]["html_parser_overlay_install"] = {"status": "SKIPPED", "reason": "stage exists or CPU deadline near"}
            overlays = [path for path in (numpy_overlay, zarr_overlay, gdown_overlay, html_overlay) if path.is_dir()]
            download_env = os.environ.copy()
            download_env["PYTHONPATH"] = os.pathsep.join(map(str, overlays + [staged]))
            if data_archive.is_file() and data_archive.stat().st_size > 0:
                report["stages"]["data_archive"] = {"status": "PASS", "official_drive_file_id": DRIVE_ID, "path": str(data_archive), "bytes": data_archive.stat().st_size, "reused": True}
                data_exit_code = 0
            else:
                lock = data_archive.with_suffix(data_archive.suffix + ".lock")
                try:
                    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                except FileExistsError as exc:
                    raise RuntimeError(f"Another writer already owns dataset archive {data_archive}") from exc
                os.close(lock_fd)
                data_download_part = data_archive.with_name(data_archive.name + f".partial-{job_id}")
                try:
                    data_download = subprocess.run([str(python), "-m", "gdown", DATA_URL, "-O", str(data_download_part)], cwd=root, env=download_env, text=True, capture_output=True, timeout=min(3600, max(30, int(deadline - time.monotonic() - 300))))
                    data_exit_code = data_download.returncode
                    if data_download.returncode == 0 and data_download_part.is_file() and data_download_part.stat().st_size > 0:
                        os.replace(data_download_part, data_archive)
                finally:
                    lock.unlink(missing_ok=True)
                report["stages"]["data_archive"] = {"status": "PASS" if data_download.returncode == 0 and data_archive.is_file() else "FAIL", "official_drive_file_id": DRIVE_ID, "exit_code": data_download.returncode, "stdout_tail": data_download.stdout[-3000:], "stderr_tail": data_download.stderr[-3000:], "path": str(data_archive), "bytes": data_archive.stat().st_size if data_archive.exists() else None, "reused": False}
            atomic_json(out, report, job_id)
            if data_exit_code != 0:
                raise RuntimeError("Official Google Drive dataset archive download failed; no login-node fallback")

            if time.monotonic() < deadline - 1200:
                raw_data = root / "datasets" / DATA_RAW_MEMBER
                if not raw_data.exists():
                    part = raw_data.with_name(raw_data.name + f".partial-{job_id}")
                    with tarfile.open(data_archive, "r:gz") as tf:
                        members = tf.getmembers()
                        npz_members = [m for m in members if m.isfile() and Path(m.name).name.lower().endswith(".npz")]
                        matches = [m for m in npz_members if Path(m.name).name == DATA_RAW_MEMBER]
                        report["stages"]["dataset_archive_inventory"] = {
                            "status": "PASS" if len(matches) == 1 else "FAIL",
                            "archive": str(data_archive),
                            "configured_rendered_name": DATA_MEMBER,
                            "official_raw_name_from_render_all": DATA_RAW_MEMBER,
                            "npz_members": [m.name for m in npz_members],
                            "exact_raw_member_matches": [m.name for m in matches],
                        }
                        atomic_json(out, report, job_id)
                        if len(matches) != 1:
                            raise RuntimeError(f"Expected one official raw member {DATA_RAW_MEMBER}; found {len(matches)}")
                        source_file = tf.extractfile(matches[0])
                        if source_file is None:
                            raise RuntimeError(f"Cannot read official archive member {matches[0].name}")
                        with source_file, part.open("wb") as target:
                            shutil.copyfileobj(source_file, target, length=4 * 1024 * 1024)
                            target.flush()
                            os.fsync(target.fileno())
                    os.replace(part, raw_data)
                report["stages"]["raw_dataset"] = {"status": "PASS", "official_archive_member": DATA_RAW_MEMBER, "configured_rendered_output": DATA_MEMBER, "path": str(raw_data), "bytes": raw_data.stat().st_size, "archive_bytes": data_archive.stat().st_size}
                atomic_json(out, report, job_id)

                rendered = root / "datasets" / "good_quality_data_memmap"
                renderer = root / "render_wall_memmap.py"
                if not renderer.is_file():
                    report["stages"]["render"] = {"status": "BLOCKED", "reason": "guarded chunked renderer control file was not uploaded"}
                else:
                    render_env = os.environ.copy()
                    render_env["PYTHONPATH"] = os.pathsep.join(map(str, overlays + [staged]))
                    remaining = max(30, int(deadline - time.monotonic() - 180))
                    render_started = time.monotonic()
                    render = subprocess.run([str(python), str(renderer), "--repo", str(staged), "--input", str(raw_data), "--output", str(rendered), "--config", str(staged / "pldm_envs" / "wall" / "configs" / "good_quality_data.yaml"), "--commit", SOURCE_SHA, "--max-seconds", str(remaining)], cwd=staged, env=render_env, timeout=remaining)
                    report["stages"]["render"] = {"status": "PASS" if render.returncode == 0 else "FAIL", "exit_code": render.returncode, "elapsed_seconds": time.monotonic() - render_started, "stdout_tail": "streamed to the PBS CPU job log", "stderr_tail": "streamed to the PBS CPU job log", "path": str(rendered), "max_seconds_at_start": remaining}
                    if render.returncode == 0:
                        report["actual_training_loader"] = actual_training_loader_report(
                            python, staged, overlays,
                            staged / "pldm" / "configs" / "wall" / "icml" / "seqlen90_3M.yaml",
                            rendered, root / "checkpoints",
                        )
            else:
                report["stages"]["raw_dataset"] = {"status": "SKIPPED", "reason": "less than 20 minutes remained in CPU preparation allocation"}
        else:
            report["stages"]["data_archive"] = {"status": "SKIPPED", "reason": "less than 10 minutes remained in CPU preparation allocation"}

        report["finished_utc"] = utc_now()
        source_ok = report.get("stages", {}).get("source", {}).get("status") == "PASS"
        data_ok = report.get("stages", {}).get("render", {}).get("status") == "PASS"
        runtime_ok = all(runtime.get(name) == "PASS" for name in ("pldm.train", "pldm.evaluation.evaluator", "pldm_envs.wall.render_images")) if isinstance(runtime, dict) else False
        config_ok = report.get("stages", {}).get("config_composition", {}).get("status") == "PASS"
        loader_ok = report.get("actual_training_loader", {}).get("status") == "PASS"
        report["status"] = "PASS" if source_ok and data_ok and runtime_ok and config_ok and loader_ok else "PARTIAL"
        runtime = runtime_probe(python, staged, overlays)
        report["runtime_probe_final"] = runtime
        report["finished_utc"] = utc_now()
        atomic_json(out, report, job_id)
        atomic_json(root / "reports" / job_id / "PREPARATION.json", report, job_id)
        print(f"CPU_PREPARATION_{report['status']}", flush=True)
        return 0 if report["status"] == "PASS" else 2
    except Exception as exc:
        report["status"] = "FAIL"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["finished_utc"] = utc_now()
        atomic_json(out, report, job_id)
        atomic_json(root / "reports" / job_id / "PREPARATION.json", report, job_id)
        print(f"CPU_PREPARATION_FAIL {type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
