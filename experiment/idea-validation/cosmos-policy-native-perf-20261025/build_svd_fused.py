"""Build the isolated official Nunchaku rank-16 LoRA fusion candidate on PBS CPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess

from prepare_sources import guard


NUNCHAKU_COMMIT = "302e0e97024ebd68688fe890e5df83731edf7b54"
DEEPCOMPRESSOR_COMMIT = "69f3473f5e1c1504bae35cc50c7858ef900a9b17"
MODULE_NAME = "nunchaku_svdq_fused_native"
BUILD_DIR = Path("native_extensions") / "svdquant_w4a4_fused"


def _version(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value)
    if not match:
        raise RuntimeError(f"Could not parse version string: {value!r}")
    return tuple(int(part or 0) for part in match.groups())


def _check_environment():
    import torch
    from torch.utils.cpp_extension import CUDA_HOME

    if _version(torch.__version__) < (2, 7, 0):
        raise RuntimeError(f"Use the existing PyTorch >=2.7 environment; found {torch.__version__}")
    if not torch.version.cuda or _version(torch.version.cuda) < (12, 8, 0):
        raise RuntimeError(f"Use the existing Torch CUDA >=12.8 environment; found {torch.version.cuda}")
    if CUDA_HOME is None:
        raise RuntimeError("CUDA toolkit is unavailable inside this PBS CPU allocation")
    nvcc = Path(CUDA_HOME) / "bin" / "nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"nvcc is missing from the selected toolkit: {nvcc}")
    output = subprocess.check_output([str(nvcc), "--version"], text=True)
    match = re.search(r"release (\d+\.\d+)", output)
    if not match or _version(match.group(1)) < (12, 8, 0):
        raise RuntimeError("This candidate needs the existing CUDA 12.8 toolkit; no package install is attempted")
    return torch, CUDA_HOME, nvcc, output.strip()


def _required(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required {description} is missing: {path}")
    return path


def _build(root: Path, load):
    import torch

    source = root / "vendor" / "nunchaku"
    commit_path = _required(source / "UPSTREAM_COMMIT", "Nunchaku source receipt")
    commit = commit_path.read_text(encoding="utf-8").strip()
    if commit != NUNCHAKU_COMMIT:
        raise RuntimeError(f"Nunchaku source differs from the reviewed pin: {commit}")
    source_paths = [
        _required(source / "src" / "interop" / "torch.cpp", "official Nunchaku Torch interop"),
        _required(source / "src" / "kernels" / "zgemm" / "gemm_w4a4_launch_fp16_int4.cu",
                  "official Nunchaku FP16 INT4 GEMM instantiation"),
        _required(root / "nunchaku_svd_fused.cu", "task-local SVD fusion bridge"),
        _required(root / "nunchaku_repack.cu", "existing direct-code packing bridge"),
    ]
    includes = [source / "src"] + [source / "third_party" / name / "include"
                                    for name in ("cutlass", "json", "mio", "spdlog")]
    torch_root = Path(torch.__file__).resolve().parent.parent
    nvidia_packages = torch_root / "nvidia"
    if not nvidia_packages.is_dir():
        raise FileNotFoundError(f"Existing Torch CUDA header packages are missing: {nvidia_packages}")
    includes.extend(path for path in nvidia_packages.glob("*/include") if path.is_dir())
    for path in includes:
        if not path.is_dir():
            raise FileNotFoundError(f"Required Nunchaku header subtree is missing: {path}")

    output = root / BUILD_DIR
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    os.environ["MAX_JOBS"] = "2"
    extension = load(
        name=MODULE_NAME,
        sources=[str(path) for path in source_paths],
        extra_include_paths=[str(path) for path in includes],
        extra_cflags=["-O3", "-std=c++20", "-DENABLE_BF16=1", "-DBUILD_NUNCHAKU=1", "-UNDEBUG"],
        extra_cuda_cflags=[
            "-O3", "-std=c++20", "--expt-extended-lambda", "--expt-relaxed-constexpr",
            "-DENABLE_BF16=1", "-DBUILD_NUNCHAKU=1", "-UNDEBUG",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_HALF2_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        ],
        build_directory=str(output),
        verbose=True,
    )
    return {
        "module": str(Path(extension.__file__).resolve()),
        "module_name": MODULE_NAME,
        "build_dir": str(output.resolve()),
        "source_commit": commit,
        "packing_exporter_source": {
            "repository": "https://github.com/nunchux-ai/deepcompressor",
            "commit": DEEPCOMPRESSOR_COMMIT,
            "path": "deepcompressor/backend/nunchaku/utils.py",
            "function": "NunchakuWeightPacker.pack_lowrank_weight",
            "lines": "147-175",
        },
        "upstream_sources": [str(path.relative_to(source)) for path in source_paths[:2]],
        "task_local_sources": [path.name for path in source_paths[2:]],
        "include_dirs": [str(path) for path in includes],
        "native_kernel_fusion": {
            "lora_down": "official quantize_w4a4_act_fuse_lora; keep FP32 LoRA activation",
            "lora_up": "official GEMM_W4A4_Launch LoRA up epilogue, rank 16, scale 1.0",
            "sidecar_groupwise_A4_outputs_discarded": True,
            "main_activation_recipe": "existing per-token whole-K symmetric A4",
            "checkpoint_codes_scales_factors_requantized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="cosmos-policy-native-perf-20261025 experiment directory")
    args = parser.parse_args()
    guard()
    root = args.root.resolve()
    if not (root / "vendor").is_dir():
        raise FileNotFoundError(f"Pinned upstream sources are not staged under {root / 'vendor'}")
    torch, cuda_home, nvcc, nvcc_version = _check_environment()
    from torch.utils.cpp_extension import load

    arm = _build(root, load)
    receipt = {
        "status": "complete",
        "pbs_job_id": os.environ["PBS_JOBID"],
        "host": socket.gethostname(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_home": str(cuda_home),
        "nvcc": str(nvcc),
        "nvcc_version": nvcc_version,
        "arch": "sm_80",
        "max_jobs": 2,
        "arms": {"svdquant_w4a4_fused_candidate": arm},
        "gpu_execution_tested": False,
        "new_training_updates": 0,
        "new_closed_loop_episodes": 0,
    }
    destination = root / "results" / "preparation" / "native_svd_fused_build.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
