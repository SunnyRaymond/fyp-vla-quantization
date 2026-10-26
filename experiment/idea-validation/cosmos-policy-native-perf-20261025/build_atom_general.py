"""Build ViDiT-Q Atom with a task-local per-column scale dequant epilogue."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess

from prepare_sources import guard


SOURCE_COMMIT = "43657d11b32192ec0f50f0dac8d3a16c3d841196"
SOURCE_REPOSITORY = "https://github.com/thu-nics/ViDiT-Q.git"
MODULE_NAME = "viditq_atom_per_channel_native"
BUILD_SUBDIR = Path("native_extensions") / "viditq_w4a4_general"

_SCALE_SETUP = """      half2 rs_scale = __hmul2(row_scale, col_scale);
      float rs_scale_u = __half2float(rs_scale.x);
      float rs_scale_d = __half2float(rs_scale.y);"""
_PER_COLUMN_SETUP = """      half2 upper_row_scale = __halves2half2(row_scale.x, row_scale.x);
      half2 lower_row_scale = __halves2half2(row_scale.y, row_scale.y);
      half2 upper_scale = __hmul2(upper_row_scale, col_scale);
      half2 lower_scale = __hmul2(lower_row_scale, col_scale);
      float rs_scale_c0 = __half2float(upper_scale.x);
      float rs_scale_c1 = __half2float(upper_scale.y);
      float rs_scale_c2 = __half2float(lower_scale.x);
      float rs_scale_c3 = __half2float(lower_scale.y);"""


def _patch_per_column_dequant(source_text: str) -> str:
    """Change only Atom's dequant scale setup and map each MMA slot to its column."""
    function_anchor = "__device__ __forceinline__ void dequant("
    next_function_anchor = "\n__global__ void __launch_bounds__(256) compute_gemm_imma"
    if source_text.count(function_anchor) != 1 or source_text.count(next_function_anchor) != 1:
        raise RuntimeError("Reviewed ViDiT-Q Atom source changed: expected unique dequant/kernel anchors")
    start = source_text.index(function_anchor)
    end = source_text.index(next_function_anchor, start)
    if end <= start:
        raise RuntimeError("Reviewed ViDiT-Q Atom dequant function boundary is malformed")

    function = source_text[start:end]
    if function.count(_SCALE_SETUP) != 1:
        raise RuntimeError("Reviewed ViDiT-Q Atom dequant scale setup is missing or ambiguous")
    function = function.replace(_SCALE_SETUP, _PER_COLUMN_SETUP, 1)

    # PTX fragment order is c0=(row r,col c), c1=(r,c+1),
    # c2=(r+8,c), c3=(r+8,c+1); the source's four accumulator slots must match.
    for offset, old_scale, new_scale in (
        (0, "rs_scale_u", "rs_scale_c0"),
        (1, "rs_scale_u", "rs_scale_c1"),
        (2, "rs_scale_d", "rs_scale_c2"),
        (3, "rs_scale_d", "rs_scale_c3"),
    ):
        pattern = re.compile(
            rf"(c_frag\[[^\]\n]*\+\s*{offset}\]\)\s*\*\s*){old_scale};"
        )
        function, count = pattern.subn(rf"\g<1>{new_scale};", function)
        if count != 1:
            raise RuntimeError(
                f"Reviewed ViDiT-Q Atom dequant slot c{offset} is missing or ambiguous"
            )
    return source_text[:start] + function + source_text[end:]


def _patched_current_stream_source(source_text: str) -> str:
    """Apply the local scale adapter, then the existing explicit-stream patch."""
    from build_diffusion import _viditq_current_stream_source

    return _viditq_current_stream_source(_patch_per_column_dequant(source_text))


def _require_commit(source: Path) -> str:
    marker = source / "UPSTREAM_COMMIT"
    if not marker.is_file():
        raise FileNotFoundError(f"ViDiT-Q source receipt is missing: {marker}")
    commit = marker.read_text(encoding="utf-8").strip()
    if commit != SOURCE_COMMIT:
        raise RuntimeError(f"ViDiT-Q source differs from the reviewed pin: {commit}")
    return commit


def _require(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} source is missing: {path}")
    return path


def _gxx11() -> tuple[str, str]:
    compiler = shutil.which("g++-11") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("The existing PBS CPU environment must provide g++ 11; installation is disabled")
    version = subprocess.check_output([compiler, "--version"], text=True, stderr=subprocess.STDOUT)
    match = re.search(r"g\+\+[^\n]*?\b(\d+)\.(\d+)", version)
    if not match or int(match.group(1)) != 11:
        raise RuntimeError(f"Expected the existing g++ 11 compiler, found: {version.splitlines()[0]}")
    return str(Path(compiler).resolve()), version.splitlines()[0]


def _build(root: Path, load):
    from build_diffusion import _check_environment

    source = root / "vendor" / "viditq"
    commit = _require_commit(source)
    kernel = _require(source / "kernels" / "csrc" / "qgemm" / "w4a4" / "atom.cu",
                      "pinned ViDiT-Q Atom kernel")
    binding = _require(root / "viditq_atom_minimal.cpp", "existing minimal Atom pybind bridge")
    torch, cuda_home, nvcc, nvcc_version = _check_environment()
    compiler, compiler_version = _gxx11()

    output = root / BUILD_SUBDIR
    output.mkdir(parents=True, exist_ok=True)
    patched_kernel = output / "atom_per_column_current_stream.cu"
    patched_kernel.write_text(
        _patched_current_stream_source(kernel.read_text(encoding="utf-8")),
        encoding="utf-8", newline="\n")

    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    os.environ["MAX_JOBS"] = "2"
    os.environ["CC"] = compiler
    os.environ["CXX"] = compiler
    extension = load(
        name=MODULE_NAME,
        sources=[str(binding), str(patched_kernel)],
        extra_include_paths=[str(kernel.parent), str(source / "kernels" / "csrc"),
                             str(source / "kernels" / "csrc" / "qgemm")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr"],
        build_directory=str(output),
        verbose=True,
    )
    return {
        "status": "complete",
        "pbs_job_id": os.environ["PBS_JOBID"],
        "host": socket.gethostname(),
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": commit,
        "upstream_source": str(kernel.relative_to(source)),
        "module": str(Path(extension.__file__).resolve()),
        "module_name": MODULE_NAME,
        "build_dir": str(output.resolve()),
        "task_local_patches": [
            "per-column dequant epilogue: preserve official INT4 MMA/load/store; map c0/c1/c2/c3 to upper.x/upper.y/lower.x/lower.y",
            "explicit PyTorch current-CUDA-stream launch, reused from build_diffusion._viditq_current_stream_source",
            "existing minimal pybind bridge viditq_atom_minimal.cpp",
        ],
        "backend_scope": "Official ViDiT-Q Atom W4A4 core with a task-local per-column scale epilogue adapter; not the complete official ViDiT-Q engine",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_home": str(cuda_home),
        "nvcc": str(nvcc),
        "nvcc_version": nvcc_version,
        "host_compiler": compiler,
        "host_compiler_version": compiler_version,
        "arch": "sm_80",
        "max_jobs": 2,
        "gpu_execution_tested": False,
        "new_training_updates": 0,
        "new_closed_loop_episodes": 0,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="cosmos-policy-native-perf-20261025 experiment directory")
    args = parser.parse_args(argv)
    guard()
    root = args.root.resolve()
    from torch.utils.cpp_extension import load

    arm = _build(root, load)
    receipt_path = root / "results" / "preparation" / "native_viditq_general_build.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(arm, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(receipt_path), "module": arm["module"],
                      "source_commit": arm["source_commit"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
