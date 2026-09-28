#!/usr/bin/env python3
"""Stage the pinned Fast-LeWM PushT source/checkpoint and inspect them on CPU."""

import ast
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import tarfile
import time
import textwrap
import traceback
import urllib.request

REPO_URL = "https://github.com/Yuntian-Gao/Fast-LeWorldModel.git"
SOURCE_SHA = "de3e9dac539f5bbe6ff1656a2fb00938d62a3c7d"
HF_REPO = "naiverer/fast-leworldmodel"
CHECKPOINT_REL = Path("checkpoints/Fast-lewm_pusht_object.ckpt")
OLD_ROOT = Path("/scratch/users/ntu/yguo017/lewm-pusht-iteration")
RUNTIME_PYTHON = OLD_ROOT / "venv/bin/python"


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def log(message):
    print(message, flush=True)


def run(args, *, cwd=None, timeout=300, capture=False):
    log("$ " + " ".join(map(str, args)))
    return subprocess.run(
        list(map(str, args)), cwd=cwd, timeout=timeout, check=True,
        text=True, stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def download(url, destination, timeout=90):
    request = urllib.request.Request(url, headers={"User-Agent": "fast-lewm-pusht-cpu-preparation/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response, open(destination, "xb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)


def safe_extract(archive, destination):
    base = destination.resolve()
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        for member in members:
            path = Path(member.name)
            target = (base / path).resolve()
            if path.is_absolute() or not target.is_relative_to(base):
                raise RuntimeError(f"Unsafe path in source archive: {member.name}")
            if member.issym() or member.islnk():
                link = Path(member.linkname)
                link_target = (target.parent / link).resolve()
                if link.is_absolute() or not link_target.is_relative_to(base):
                    raise RuntimeError(f"Unsafe link in source archive: {member.name}")
        tf.extractall(destination)


def candidate_size(item):
    value = item.get("size")
    if value is None and isinstance(item.get("lfs"), dict):
        value = item["lfs"].get("size")
    return value if isinstance(value, int) else None


def select_checkpoint(model_info):
    files = []
    for item in model_info.get("siblings", []):
        name = item.get("rfilename", "")
        if Path(name).suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors", ".bin"}:
            files.append({"name": name, "size": candidate_size(item)})
    if not files:
        raise RuntimeError("HF model metadata contains no checkpoint file")

    named = [x for x in files if re.search(r"push[\W_]*t|pusht", x["name"], re.I)]
    pool = named or [x for x in files if x["size"] and 68_000_000 <= x["size"] <= 80_000_000]
    if len(pool) != 1:
        raise RuntimeError("Could not select one PushT checkpoint from HF metadata: " + json.dumps(files))
    return pool[0], files


def parse_requirement(line):
    line = line.split("#", 1)[0].strip()
    if not line or line.startswith(("-", "git+", "http:", "https:")):
        return None
    match = re.match(r"^([A-Za-z0-9_.-]+)\s*(.*)$", line)
    if not match:
        return None
    return match.group(1), match.group(2).strip()


def source_requirements(source):
    found = {}
    for path in sorted(source.rglob("requirements*.txt")):
        if ".git" in path.parts:
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            item = parse_requirement(line)
            if item:
                found[item[0].lower().replace("_", "-")] = {
                    "requirement": item[1], "file": str(path.relative_to(source))
                }
    return found


def requirement_report(source):
    report = {}
    for name, details in source_requirements(source).items():
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        requirement = details["requirement"]
        compatible = None
        if requirement.startswith("=="):
            compatible = version == requirement[2:].strip() if version else False
        elif requirement.startswith(">=") and version:
            try:
                from packaging.version import Version
                compatible = Version(version) >= Version(requirement[2:].strip().split(",")[0])
            except Exception:
                compatible = None
        report[name] = {
            **details, "installed_version": version,
            "constraint_satisfied": compatible,
        }
    return report


def ast_classes(source, class_name):
    found = []
    for base in (source, source.parent / "runtime_overlay", OLD_ROOT / "stable-worldmodel"):
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if ".git" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
            except (SyntaxError, OSError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    found.append((path, node.lineno))
    return found


def module_name(source, path):
    roots = (source, source.parent / "runtime_overlay", OLD_ROOT / "stable-worldmodel")
    root = next((candidate for candidate in roots if path.is_relative_to(candidate)), None)
    if root is None:
        return None
    relative = path.relative_to(root)
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    if parts and all(part.isidentifier() for part in parts):
        return ".".join(parts)
    return None


def import_class(source, class_name):
    errors = []
    for path, line in ast_classes(source, class_name):
        name = module_name(source, path)
        try:
            module = importlib.import_module(name) if name else None
            cls = getattr(module, class_name) if module else None
            if cls is not None:
                return cls, {"file": str(path), "line": line, "module": name}, errors
            errors.append(f"{path}:{line}: imported module did not expose {class_name}")
        except Exception:
            errors.append(f"{path}:{line}: " + traceback.format_exc(limit=4))
    return None, None, errors


def method_report(cls, method_name):
    method = getattr(cls, method_name, None)
    if not callable(method):
        return None
    result = {"signature": str(inspect.signature(method))}
    try:
        source_lines = textwrap.dedent(inspect.getsource(method)).splitlines()
        result["source_tail"] = source_lines[-40:]
        tree = ast.parse("\n".join(source_lines))
        return_keys = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
                return_keys.extend(
                    key.value for key in node.value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                )
        result["return_dict_keys"] = list(dict.fromkeys(return_keys))
    except Exception:
        result["source_error"] = traceback.format_exc(limit=3)
    return result


def compose_selfcheck(source):
    configs = (source / "config", source / "configs")
    config_root = next((path for path in configs if path.is_dir()), None)
    if not config_root:
        return {"status": "FAIL", "error": "No upstream config/ or configs/ directory"}
    candidates = [
        config_root / "eval" / "pusht.yaml", config_root / "eval" / "pusht.yml",
        config_root / "eval.yaml", config_root / "eval.yml",
        config_root / "eval" / "default.yaml",
    ]
    config_file = next((path for path in candidates if path.is_file()), None)
    if not config_file:
        return {
            "status": "FAIL", "config_root": str(config_root),
            "available_yaml": [str(p.relative_to(config_root)) for p in sorted(config_root.rglob("*.yaml"))[:80]],
            "error": "No eval primary config found under upstream config root",
        }
    config_name = config_file.relative_to(config_root).with_suffix("").as_posix()
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        with initialize_config_dir(version_base=None, config_dir=str(config_root), job_name="fastlewm_cpu_prep"):
            cfg = compose(config_name=config_name)
        return {
            "status": "PASS", "config_root": str(config_root), "config_name": config_name,
            "top_level_keys": list(OmegaConf.to_container(cfg, resolve=False).keys()),
        }
    except Exception:
        return {
            "status": "FAIL", "config_root": str(config_root), "config_name": config_name,
            "error": traceback.format_exc(limit=8),
        }


def checkpoint_shapes(value, torch, prefix="", limit=100):
    rows = []
    seen = set()

    def visit(item, key, depth=0):
        if len(rows) >= limit or depth > 5 or id(item) in seen:
            return
        if isinstance(item, (dict, list, tuple)) or hasattr(item, "state_dict"):
            seen.add(id(item))
        if torch.is_tensor(item):
            rows.append({"name": key or "<root>", "shape": list(item.shape), "dtype": str(item.dtype)})
        elif hasattr(item, "state_dict"):
            try:
                for name, tensor in item.state_dict().items():
                    visit(tensor, (key + "." if key else "") + name, depth + 1)
            except Exception as exc:
                rows.append({"name": key or "<root>", "inspection_error": str(exc)})
        elif isinstance(item, dict):
            for name, child in item.items():
                visit(child, (key + "." if key else "") + str(name), depth + 1)
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item[:limit]):
                visit(child, (key + "." if key else "") + str(index), depth + 1)

    visit(value, prefix)
    return rows


def inspect_runtime(source_path, checkpoint_path, output_path):
    source = Path(source_path).resolve()
    overlay = source.parent / "runtime_overlay"
    stablewm_root = overlay if overlay.is_dir() else OLD_ROOT / "stable-worldmodel"
    checkpoint = Path(checkpoint_path).resolve()
    sys.path[:0] = [str(source), str(stablewm_root)]
    if (source / "src").is_dir():
        sys.path.insert(0, str(source / "src"))
    os.chdir(source)
    result = {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "source_path": str(source),
        "runtime_overlay": str(overlay) if overlay.is_dir() else None,
        "stablewm_root": str(stablewm_root),
        "checkpoint_path": str(checkpoint),
        "requirements": requirement_report(source),
        "world": {}, "auto_cost_model": {}, "world_model_policy": {}, "checkpoint_load": {},
        "config_compose": compose_selfcheck(source),
    }

    world, world_origin, world_errors = import_class(source, "World")
    result["world"] = {"origin": world_origin, "import_errors": world_errors}
    if world:
        for method_name in ("evaluate", "evaluate_from_dataset"):
            result["world"][method_name] = method_report(world, method_name)
        result["world"]["get_cost"] = method_report(world, "get_cost")

    cost_model, cost_origin, cost_errors = import_class(source, "AutoCostModel")
    result["auto_cost_model"] = {"origin": cost_origin, "import_errors": cost_errors}
    if cost_model:
        try:
            result["auto_cost_model"]["constructor"] = str(inspect.signature(cost_model))
            result["auto_cost_model"]["source_header"] = inspect.getsource(cost_model).splitlines()[:32]
            result["auto_cost_model"]["get_cost"] = method_report(cost_model, "get_cost")
        except Exception:
            result["auto_cost_model"]["inspection_error"] = traceback.format_exc(limit=4)

    policy, policy_origin, policy_errors = import_class(source, "WorldModelPolicy")
    result["world_model_policy"] = {"origin": policy_origin, "import_errors": policy_errors}
    if policy:
        result["world_model_policy"]["constructor"] = str(inspect.signature(policy))
        result["world_model_policy"]["get_cost"] = method_report(policy, "get_cost")

    try:
        import torch
        result["torch_version"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
        checkpoint_obj = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        result["checkpoint_load"] = {
            "status": "PASS",
            "type": f"{type(checkpoint_obj).__module__}.{type(checkpoint_obj).__qualname__}",
            "top_level_keys": list(checkpoint_obj.keys())[:100] if isinstance(checkpoint_obj, dict) else None,
            "shapes": checkpoint_shapes(checkpoint_obj, torch),
        }
    except Exception:
        result["checkpoint_load"] = {"status": "FAIL", "error": traceback.format_exc(limit=12)}

    atomic_json(output_path, result)
    return 0 if result["checkpoint_load"].get("status") == "PASS" and world \
        and result["config_compose"].get("status") == "PASS" else 1


def prepare(args):
    root = Path(args.root).resolve()
    report_path = root / "prepared.json"
    source = root / "upstream"
    checkpoint = root / CHECKPOINT_REL
    report = {
        "status": "RUNNING", "job_id": args.job_id,
        "repository": REPO_URL, "source_sha": SOURCE_SHA,
        "source_path": str(source), "runtime_python": str(RUNTIME_PYTHON),
        "hf_repository": HF_REPO, "checkpoint_path": str(checkpoint),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(report_path, report)

    def on_term(_signum, _frame):
        report.update({
            "status": "FAIL", "error": "CPU preparation stopped by timeout/signal",
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        atomic_json(report_path, report)
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, on_term)
    try:
        if not RUNTIME_PYTHON.is_file():
            raise RuntimeError(f"Expected shared runtime missing: {RUNTIME_PYTHON}")
        if source.exists() or checkpoint.exists():
            raise RuntimeError("Refusing to replace pre-existing source/checkpoint in new ROOT")

        overlay = root / "runtime_overlay"
        overlay_tmp = root / f".runtime_overlay-{args.job_id}.partial"
        try:
            run([str(RUNTIME_PYTHON), "-m", "pip", "install", "--disable-pip-version-check",
                 "--no-input", "--no-deps", "--target", str(overlay_tmp),
                 "stable-worldmodel==0.0.6"], timeout=360)
            os.replace(overlay_tmp, overlay)
            report["stablewm_overlay"] = {
                "status": "installed", "version": "0.0.6", "path": str(overlay),
                "install_mode": "isolated pip --target; no-deps; shared venv unchanged",
            }
        except Exception:
            report["stablewm_overlay"] = {"status": "unavailable", "error": traceback.format_exc(limit=6)}
            if overlay_tmp.exists():
                shutil.rmtree(overlay_tmp, ignore_errors=True)
        report["stablewm_root"] = str(overlay if overlay.is_dir() else OLD_ROOT / "stable-worldmodel")
        atomic_json(report_path, report)

        stage = root / f".upstream-{args.job_id}.partial"
        if stage.exists():
            raise RuntimeError(f"Refusing to reuse an existing staging directory: {stage}")
        stage.mkdir(parents=True)
        archive_url = f"https://github.com/Yuntian-Gao/Fast-LeWorldModel/archive/{SOURCE_SHA}.tar.gz"
        archive_path = stage / "source.tar.gz"
        unpack = stage / "unpacked"
        unpack.mkdir()
        download(archive_url, archive_path, timeout=90)
        safe_extract(archive_path, unpack)
        roots = [path for path in unpack.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise RuntimeError(f"Pinned GitHub archive had unexpected top level: {[p.name for p in roots]}")
        os.replace(roots[0], source)
        (source / ".fastlewm-revision").write_text(SOURCE_SHA + "\n", encoding="ascii")
        identity = {
            "source_sha": SOURCE_SHA, "acquisition": "github pinned archive",
            "archive_url": archive_url, "source_path": str(source),
            "identity_evidence": "pinned archive URL and .fastlewm-revision marker; no hash",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_json(root / "source_identity.json", identity)
        report["source_sha"] = SOURCE_SHA
        report["source_identity_path"] = str(root / "source_identity.json")
        atomic_json(report_path, report)
        shutil.rmtree(stage)

        api_url = f"https://huggingface.co/api/models/{HF_REPO}?blobs=true"
        request = urllib.request.Request(api_url, headers={"User-Agent": "fast-lewm-pusht-cpu-preparation/1.0"})
        with urllib.request.urlopen(request, timeout=45) as response:
            model_info = json.load(response)
        hf_revision = model_info.get("sha")
        if not hf_revision:
            raise RuntimeError("HF model API did not provide an immutable revision SHA")
        selected, files = select_checkpoint(model_info)
        report.update({"hf_revision": hf_revision, "hf_checkpoint_candidates": files,
                       "hf_checkpoint_filename": selected["name"]})
        atomic_json(report_path, report)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        part = checkpoint.with_name(checkpoint.name + f".part-{args.job_id}")
        if part.exists():
            raise RuntimeError(f"Refusing to reuse checkpoint partial path: {part}")
        url = f"https://huggingface.co/{HF_REPO}/resolve/{hf_revision}/{selected['name']}"
        download(url, part, timeout=90)
        actual_size = part.stat().st_size
        if actual_size <= 0 or (selected["size"] is not None and actual_size != selected["size"]):
            raise RuntimeError(f"Checkpoint size mismatch: API={selected['size']}, downloaded={actual_size}")
        os.replace(part, checkpoint)
        report.update({
            "hf_revision": hf_revision,
            "hf_checkpoint_filename": selected["name"],
            "hf_checkpoint_size_bytes": actual_size,
            "checkpoint_url": url,
            "hf_checkpoint_candidates": files,
        })

        inspection = root / f".inspection-{args.job_id}.json"
        inspect_proc = subprocess.run(
            [str(RUNTIME_PYTHON), str(Path(__file__).resolve()), "--inspect",
             str(source), str(checkpoint), str(inspection)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=720, check=False,
        )
        report["inspection_stdout"] = inspect_proc.stdout[-16000:]
        if inspection.exists():
            report["runtime_inspection"] = json.loads(inspection.read_text(encoding="utf-8"))
            inspection.unlink()
        else:
            report["runtime_inspection"] = {"status": "FAIL", "error": "Runtime inspection produced no report"}

        stage_dir = root / "stage"
        stage_dir.mkdir(exist_ok=True)
        stage_venv = stage_dir / "venv"
        if report["runtime_inspection"].get("checkpoint_load", {}).get("status") == "PASS" \
                and report["runtime_inspection"].get("world", {}).get("origin"):
            if not stage_venv.exists():
                stage_venv.symlink_to(OLD_ROOT / "venv", target_is_directory=True)
            report["stage_python"] = str(stage_venv / "bin/python")
            report["venv_mode"] = "symlink to verified shared runtime; shared environment left unchanged"
        else:
            report["stage_python"] = None
            report["venv_mode"] = "not staged because upstream import or checkpoint load did not pass"

        world = report["runtime_inspection"].get("world", {})
        cost_model = report["runtime_inspection"].get("auto_cost_model", {})
        if inspect_proc.returncode == 0 and world.get("origin") \
                and report["runtime_inspection"].get("config_compose", {}).get("status") == "PASS" \
                and report["runtime_inspection"].get("checkpoint_load", {}).get("status") == "PASS":
            report["status"] = "PASS"
        else:
            report["status"] = "FAIL"
            report["repairable_errors"] = {
                "runtime_return_code": inspect_proc.returncode,
                "world_import_errors": world.get("import_errors", []),
                "world_methods": {k: world.get(k) for k in ("evaluate", "evaluate_from_dataset")},
                "auto_cost_model_import_errors": cost_model.get("import_errors", []),
                "checkpoint_load": report["runtime_inspection"].get("checkpoint_load"),
                "stablewm_root": report.get("stablewm_root"),
            }
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_json(report_path, report)
        return 0 if report["status"] == "PASS" else 1
    except Exception:
        report.update({
            "status": "FAIL", "error": traceback.format_exc(limit=12),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        atomic_json(report_path, report)
        return 1


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "--inspect":
        return inspect_runtime(sys.argv[2], sys.argv[3], sys.argv[4])
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--job-id", required=True)
    return prepare(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
