"""Build the pinned QuaRot Fast Hadamard Transform CUDA extension in CPU PBS."""
from __future__ import annotations

import argparse
import configparser
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import urllib.request

from prepare_sources import extract_source, guard


QUAROT_SOURCE_REVISION = "5008669b08c1f11f9b64d52d16fddd47ca754c5a"
QUAROT_SUBMODULE_PATH = "third-party/fast-hadamard-transform"
FHT_REPOSITORY = "https://github.com/Dao-AILab/fast-hadamard-transform.git"
MODULE_NAME = "quarot_native_hadamard_cuda"
GEMM_MODULES = ("quarot_official_gemm_fused", "quarot_official_gemm")


def _github_json(url):
    request = urllib.request.Request(
        url, headers={"User-Agent": "Cosmos-native-performance-research"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def _safe_revision(value):
    return (isinstance(value, str) and len(value) == 40
            and re.fullmatch(r"[0-9a-fA-F]{40}", value) is not None)


def _submodule_url(quarot_source):
    config = configparser.ConfigParser()
    config.read(quarot_source / ".gitmodules")
    matches = [row.get("url") for section in config.sections()
               if (row := config[section]).get("path") == QUAROT_SUBMODULE_PATH]
    if matches != [FHT_REPOSITORY]:
        raise RuntimeError("Pinned QuaRot tree does not declare the expected official FHT submodule")
    return matches[0]


def _submodule_pointer(tree_payload):
    if not isinstance(tree_payload, dict) or tree_payload.get("truncated"):
        raise RuntimeError("GitHub tree API returned a missing or truncated QuaRot tree")
    entries = [entry for entry in tree_payload.get("tree", [])
               if entry.get("path") == QUAROT_SUBMODULE_PATH]
    if len(entries) != 1:
        raise RuntimeError("Pinned QuaRot tree must contain exactly one FHT submodule pointer")
    entry = entries[0]
    if entry.get("mode") != "160000" or entry.get("type") != "commit":
        raise RuntimeError("QuaRot FHT tree entry is not a git submodule commit pointer")
    revision = entry.get("sha")
    if not _safe_revision(revision):
        raise RuntimeError("QuaRot FHT submodule pointer is not a full commit SHA")
    return revision.lower()


def _pinned_source_revision():
    tree_url = ("https://api.github.com/repos/spcl/QuaRot/git/trees/"
                f"{QUAROT_SOURCE_REVISION}?recursive=1")
    return _submodule_pointer(_github_json(tree_url))


def _stage_source(root, revision):
    target = root / "vendor" / "fast-hadamard-transform"
    marker = target / "UPSTREAM_COMMIT"
    if marker.is_file():
        if marker.read_text(encoding="utf-8").strip().lower() != revision:
            raise RuntimeError("Existing FHT source differs from the pinned QuaRot submodule pointer")
        return target, False
    if target.exists() and any(target.iterdir()):
        raise RuntimeError("Unmarked FHT source directory exists; refusing to overwrite it")
    target.parent.mkdir(parents=True, exist_ok=True)
    archive = target.parent / f"fast-hadamard-transform-{revision}.tar.gz"
    if not archive.is_file():
        url = f"https://codeload.github.com/Dao-AILab/fast-hadamard-transform/tar.gz/{revision}"
        request = urllib.request.Request(url, headers={"User-Agent": "Cosmos-native-performance-research"})
        partial = archive.with_name(archive.name + f".{os.environ['PBS_JOBID']}.partial")
        with urllib.request.urlopen(request, timeout=180) as response, partial.open("xb") as output:
            shutil.copyfileobj(response, output)
        os.replace(partial, archive)
    extract_source(archive, target)
    marker.write_text(revision + "\n", encoding="utf-8")
    return target, True


def _source_files(source):
    files = (source / "csrc" / "fast_hadamard_transform.cpp",
             source / "csrc" / "fast_hadamard_transform_cuda.cu")
    if any(not path.is_file() for path in files):
        raise RuntimeError("Pinned FHT source is missing its official binding or CUDA kernel translation unit")
    return files


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gemm-module", choices=GEMM_MODULES,
                        default="quarot_official_gemm_fused")
    args = parser.parse_args(argv)

    guard()
    job_id = os.environ.get("PBS_JOBID")
    if not job_id:
        raise RuntimeError("FHT build requires the real PBS_JOBID")
    if shutil.which("ninja") is None:
        raise RuntimeError("FHT build requires the already staged ninja executable; install/download is disabled")

    root = args.root.resolve()
    quarot_source = root / "vendor" / "quarot"
    marker = quarot_source / "UPSTREAM_COMMIT"
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != QUAROT_SOURCE_REVISION:
        raise RuntimeError("The staged QuaRot source does not match the frozen parent revision")
    _submodule_url(quarot_source)
    revision = _pinned_source_revision()
    source, downloaded = _stage_source(root, revision)
    sources = _source_files(source)

    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    if not str(torch.__version__).startswith("2.7.0+cu128"):
        raise RuntimeError(f"Expected the existing Torch 2.7.0+cu128 builder environment, got {torch.__version__}")
    if CUDA_HOME is None:
        raise RuntimeError("No CUDA compiler toolkit is configured in the CPU PBS allocation")

    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    os.environ.setdefault("MAX_JOBS", "2")
    backend_dir = root / "native_extensions" / args.gemm_module
    compile_dir = backend_dir / "hadamard_build"
    backend_dir.mkdir(parents=True, exist_ok=True)
    compile_dir.mkdir(parents=True, exist_ok=True)
    includes = [source / 'csrc']
    nvidia_packages = Path(torch.__file__).resolve().parent.parent / 'nvidia'
    includes.extend(path for path in nvidia_packages.glob('*/include') if path.is_dir())
    extension = load(
        name=MODULE_NAME,
        sources=[str(path.resolve()) for path in sources],
        extra_include_paths=[str(path.resolve()) for path in includes],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3", "-lineinfo", "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__", "--expt-relaxed-constexpr",
            "--expt-extended-lambda", "--use_fast_math",
        ],
        build_directory=str(compile_dir),
        verbose=True,
    )
    if not callable(getattr(extension, "fast_hadamard_transform", None)):
        raise RuntimeError("Pinned upstream extension does not export fast_hadamard_transform")
    extension_path = Path(extension.__file__).resolve()
    destination = backend_dir / extension_path.name
    shutil.copy2(extension_path, destination)

    nvcc = Path(CUDA_HOME) / "bin" / "nvcc"
    nvcc_version = subprocess.run([str(nvcc), "--version"], check=True,
                                  capture_output=True, text=True).stdout.strip()
    output = root / "results" / "preparation"
    output.mkdir(parents=True, exist_ok=True)
    receipt = {
        "status": "complete",
        "pbs_job_id": job_id,
        "host": socket.gethostname(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "module_name": MODULE_NAME,
        "module": str(destination),
        "gemm_module": args.gemm_module,
        "source_repository": FHT_REPOSITORY,
        "source_commit": revision,
        "source_submodule_downloaded": downloaded,
        "source_files": [str(path.resolve()) for path in sources],
        "compiled_source_tus": [path.name for path in sources],
        "include_dirs": [str(path.resolve()) for path in includes],
        "submodule_path": QUAROT_SUBMODULE_PATH,
        "quarot_source_revision": QUAROT_SOURCE_REVISION,
        "submodule_pointer_source": "GitHub git trees API at the pinned QuaRot commit",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_home": str(CUDA_HOME),
        "nvcc": str(nvcc),
        "nvcc_version": nvcc_version,
        "arch": "sm_80",
        "build_backend": "torch.utils.cpp_extension.load; no pip install",
        "gpu_execution_tested": False,
        "new_training_updates": 0,
        "new_closed_loop_episodes": 0,
    }
    receipt_path = output / "quarot_hadamard_build.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "receipt": str(receipt_path),
                      "module": str(destination), "source_revision": revision}), flush=True)


if __name__ == "__main__":
    main()
