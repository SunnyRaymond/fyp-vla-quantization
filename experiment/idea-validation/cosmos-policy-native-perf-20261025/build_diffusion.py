"""Build only the official Nunchaku SVDQ and ViDiT-Q GEMM kernels on PBS CPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from prepare_sources import guard

NUNCHAKU_COMMIT = "302e0e97024ebd68688fe890e5df83731edf7b54"


def _version(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value)
    if not match:
        raise RuntimeError(f"Could not parse version string: {value!r}")
    return tuple(int(part or 0) for part in match.groups())


def _check_environment():
    import torch
    from torch.utils.cpp_extension import CUDA_HOME

    if _version(torch.__version__) < (2, 7, 0):
        raise RuntimeError(f"Nunchaku source requires existing PyTorch >=2.7; found {torch.__version__}")
    if not torch.version.cuda or _version(torch.version.cuda) < (12, 8, 0):
        raise RuntimeError(f"Use the existing CUDA >=12.8 Torch environment; found {torch.version.cuda}")
    if CUDA_HOME is None:
        raise RuntimeError("No nvcc toolkit is available inside this PBS CPU allocation")
    nvcc = Path(CUDA_HOME) / "bin" / "nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"nvcc is missing from the selected CUDA toolkit: {nvcc}")
    output = subprocess.check_output([str(nvcc), "--version"], text=True)
    match = re.search(r"release (\d+\.\d+)", output)
    if not match or _version(match.group(1)) < (12, 8, 0):
        raise RuntimeError("Nunchaku A100 build requires nvcc >=12.8; no Torch or CUDA upgrade is attempted")
    return torch, CUDA_HOME, nvcc, output.strip()


def _require_commit(source: Path, label: str) -> str:
    path = source / "UPSTREAM_COMMIT"
    if not path.is_file():
        raise FileNotFoundError(f"{label} source receipt is missing: {path}")
    return path.read_text(encoding="utf-8").strip()


def _require(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} source is missing: {path}")
    return path


def _viditq_current_stream_source(source_text: str) -> str:
    """Patch only Atom's host launch so its official kernel follows PyTorch's stream."""
    include_anchor = "#include <cuda.h>"
    signature_anchor = "  const size_t K_GLOBAL\n){"
    launch_anchor = "compute_gemm_imma<<<gridDim, blockDim, SHMEM_SZ>>>("
    for anchor, label in ((include_anchor, "CUDA runtime include"),
                          (signature_anchor, "Atom stream parameter"),
                          (launch_anchor, "Atom kernel launch")):
        if source_text.count(anchor) != 1:
            raise RuntimeError(f"Reviewed ViDiT-Q Atom source changed: expected one {label} anchor")
    source_text = source_text.replace(include_anchor,
                                      include_anchor + "\n#include <cuda_runtime.h>", 1)
    source_text = source_text.replace(signature_anchor,
                                      "  const size_t K_GLOBAL,\n  cudaStream_t stream\n){", 1)
    return source_text.replace(launch_anchor,
                              "compute_gemm_imma<<<gridDim, blockDim, SHMEM_SZ, stream>>>(", 1)


def _build_nunchaku(root: Path, load):
    import torch
    source = root / "vendor" / "nunchaku"
    commit = _require_commit(source, "Nunchaku")
    if commit != NUNCHAKU_COMMIT:
        raise RuntimeError(f"Nunchaku checkout differs from reviewed commit: {commit}")
    paths = [
        source / "src" / "interop" / "torch.cpp",
        source / "src" / "kernels" / "zgemm" / "gemm_w4a4_launch_fp16_int4.cu",
        root / "nunchaku_minimal.cu",
        root / "nunchaku_repack.cu",
    ]
    sources = [str(_require(path, "Nunchaku minimal GEMM")) for path in paths]
    includes = [source / "src"] + [source / "third_party" / name / "include"
                                    for name in ("cutlass", "json", "mio", "spdlog")]
    # The compiler-only toolkit reuses the existing Torch CUDA library headers.
    nvidia_packages = Path(torch.__file__).resolve().parent.parent / "nvidia"
    includes.extend(path for path in nvidia_packages.glob('*/include') if path.is_dir())
    for path in includes:
        if not path.is_dir():
            raise FileNotFoundError(f"Required Nunchaku header subtree is missing: {path}")
    output = root / "native_extensions" / "svdquant_w4a4"
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    os.environ.setdefault("MAX_JOBS", "4")
    extension = load(
        name="nunchaku_svdq_native",
        sources=sources,
        extra_include_paths=[str(path) for path in includes],
        extra_cflags=["-O3", "-std=c++20", "-DENABLE_BF16=1", "-DBUILD_NUNCHAKU=1", "-UNDEBUG"],
        extra_cuda_cflags=["-O3", "-std=c++20", "--expt-extended-lambda", "--expt-relaxed-constexpr",
                           "-DENABLE_BF16=1", "-DBUILD_NUNCHAKU=1", "-UNDEBUG",
                           "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                           "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_HALF2_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-U__CUDA_NO_BFLOAT162_CONVERSIONS__"],
        build_directory=str(output),
        verbose=True,
    )
    return {"module": extension.__file__, "source_commit": commit,
            "upstream_sources": [str(path.relative_to(source)) for path in paths[:2]],
            "task_local_sources": [path.name for path in paths[2:]],
            "include_dirs": [str(path) for path in includes],
            "adapter": "Official INT4 GEMM with task-local direct-code MMA repacking and G128-to-two-G64 scale duplication"}


def _build_viditq(root: Path, load):
    source = root / "vendor" / "viditq"
    commit = _require_commit(source, "ViDiT-Q")
    kernel = _require(source / "kernels" / "csrc" / "qgemm" / "w4a4" / "atom.cu",
                      "ViDiT-Q official Atom W4A4")
    binding = _require(root / "viditq_atom_minimal.cpp", "ViDiT-Q Atom bridge")
    include_dirs = [kernel.parent, source / "kernels" / "csrc",
                    source / "kernels" / "csrc" / "qgemm"]
    output = root / "native_extensions" / "viditq_w4a4"
    output.mkdir(parents=True, exist_ok=True)
    # Upstream's public wrapper launches on CUDA's default stream. Generate a
    # source-preserving local copy with only an explicit current-stream arg so
    # calls remain ordered with PyTorch and safe under graph capture.
    source_text = _viditq_current_stream_source(kernel.read_text(encoding="utf-8"))
    stream_kernel = output / "atom_current_stream.cu"
    stream_kernel.write_text(source_text, encoding="utf-8", newline="\n")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    os.environ.setdefault("MAX_JOBS", "4")
    extension = load(
        name="viditq_atom_native",
        sources=[str(binding), str(stream_kernel)],
        extra_include_paths=[str(path) for path in include_dirs],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr"],
        build_directory=str(output),
        verbose=True,
    )
    return {"module": extension.__file__, "source_commit": commit,
            "upstream_sources": [str(kernel.relative_to(source))],
            "task_local_sources": [binding.name, stream_kernel.name], "include_dirs": [str(path) for path in include_dirs],
            "adapter": "Official ViDiT-Q Atom W4A4 kernel with one task-local current-CUDA-stream launch patch, pybind bridge, and affine-zero correction"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="cosmos-policy-native-perf-20261025 experiment directory")
    args = parser.parse_args()
    guard()
    root = args.root.resolve()
    if not (root / "vendor").is_dir():
        raise FileNotFoundError(f"Upstream source staging is missing under {root / 'vendor'}")
    torch, cuda_home, nvcc, nvcc_version = _check_environment()
    from torch.utils.cpp_extension import load

    builds = {
        "svdquant_w4a4": _build_nunchaku(root, load),
        "viditq_w4a4": _build_viditq(root, load),
    }
    receipt = {
        "status": "complete",
        "pbs_job_id": os.environ["PBS_JOBID"],
        "host": __import__("socket").gethostname(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_home": str(cuda_home),
        "nvcc": str(nvcc),
        "nvcc_version": nvcc_version,
        "arch": "sm_80",
        "arms": builds,
        "gpu_execution_tested": False,
        "new_training_updates": 0,
        "new_closed_loop_episodes": 0,
    }
    output = root / "results" / "preparation" / "native_diffusion_build.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
