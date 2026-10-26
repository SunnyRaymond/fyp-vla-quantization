"""Build the upstream QuaRot INT4 primitive inside a CPU PBS allocation."""
from __future__ import annotations
import argparse
import configparser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

from prepare_sources import guard, extract_source


def github_json(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'Cosmos-native-performance-research'})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def prepare_submodule(source, path):
    commit = (source / 'UPSTREAM_COMMIT').read_text().strip()
    cfg = configparser.ConfigParser()
    cfg.read(source / '.gitmodules')
    url = next(cfg[section]['url'] for section in cfg.sections() if cfg[section]['path'] == path)
    slug = url.removeprefix('https://github.com/').removesuffix('.git')
    parent_slug = {'quarot': 'spcl/QuaRot', 'nunchaku': 'nunchux-ai/nunchaku'}[source.name]
    child = path.rsplit('/', 1)[1]
    entry = github_json(f'https://api.github.com/repos/{parent_slug}/contents/{path}?ref={commit}')
    subcommit = entry['sha']
    target = source / path
    marker = target / 'UPSTREAM_COMMIT'
    if marker.is_file() and marker.read_text().strip() == subcommit:
        return {'url': url, 'commit': subcommit, 'path': str(target)}
    archive = source.parent / (source.name + '-' + child + '-' + subcommit + '.tar.gz')
    if not archive.is_file():
        urllib.request.urlretrieve(f'https://codeload.github.com/{slug}/tar.gz/{subcommit}', archive)
    extract_source(archive, target)
    marker.write_text(subcommit + '\n')
    return {'url': url, 'commit': subcommit, 'path': str(target)}


BINDING = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <gemm.h>
torch::Tensor matmul(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.device() == b.device(), "same CUDA device required");
  TORCH_CHECK(a.scalar_type() == at::kByte && b.scalar_type() == at::kByte, "packed uint8 required");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(1), "packed shapes differ");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "contiguous packed tensors required");
  c10::cuda::CUDAGuard device_guard(a.device());
  auto c = torch::empty({a.size(0), b.size(0)}, a.options().dtype(at::kInt));
  matmul_host(a.data_ptr<Int4Storage>(), b.data_ptr<Int4Storage>(),
              a.size(0), b.size(0), a.size(1)*2, c.data_ptr<int32_t>(),
              at::cuda::getCurrentCUDAStream(a.get_device()).stream());
  return c;
}
void scaled_add(torch::Tensor output, torch::Tensor acc, torch::Tensor ascale,
                torch::Tensor wscale, int64_t group);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul", &matmul);
  m.def("scaled_add", &scaled_add);
}
'''

EPILOGUE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
template <typename WeightScale>
__global__ void scale_add_kernel(float* output, const int32_t* acc,
                                const float* ascale, const WeightScale* wscale,
                                int64_t count, int64_t n, int64_t groups, int64_t group) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  int64_t row = index / n, col = index % n;
  float factor = __fmul_rn(ascale[row], static_cast<float>(wscale[col * groups + group]));
  output[index] = __fadd_rn(output[index], __fmul_rn(static_cast<float>(acc[index]), factor));
}
void scaled_add(torch::Tensor output, torch::Tensor acc, torch::Tensor ascale,
                torch::Tensor wscale, int64_t group) {
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == at::kFloat && output.is_contiguous(), "float CUDA output required");
  TORCH_CHECK(acc.device() == output.device() && ascale.device() == output.device() && wscale.device() == output.device(), "same CUDA device required");
  TORCH_CHECK(acc.scalar_type() == at::kInt && ascale.scalar_type() == at::kFloat, "int32 accumulator and FP32 A scales required");
  TORCH_CHECK(acc.is_contiguous() && ascale.is_contiguous() && wscale.is_contiguous(), "contiguous scales required");
  TORCH_CHECK(output.dim() == 2 && acc.sizes() == output.sizes(), "output/acc shape mismatch");
  TORCH_CHECK(wscale.dim() == 2 && wscale.size(0) == output.size(1) && ascale.numel() == output.size(0), "scale shape mismatch");
  TORCH_CHECK(group >= 0 && group < wscale.size(1), "group index outside scale tensor");
  c10::cuda::CUDAGuard guard(output.device());
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device()).stream();
  auto count = output.numel();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, wscale.scalar_type(), "scaled_add", [&] {
    scale_add_kernel<scalar_t><<<(count+255)/256,256,0,stream>>>(output.data_ptr<float>(),
      acc.data_ptr<int32_t>(), ascale.data_ptr<float>(), wscale.data_ptr<scalar_t>(),
      count, output.size(1), wscale.size(1), group);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--sources-only', action='store_true')
    parser.add_argument('--fused', action='store_true')
    args = parser.parse_args()
    guard()
    vendor = args.root / 'vendor'
    submodules = {'quarot_cutlass': prepare_submodule(vendor/'quarot', 'third-party/cutlass')}
    for name in ('cutlass', 'spdlog', 'json', 'mio'):
        submodules['nunchaku_' + name] = prepare_submodule(vendor/'nunchaku', 'third_party/' + name)
    out = args.root/'results/preparation'
    out.mkdir(parents=True, exist_ok=True)
    (out/'submodules.json').write_text(json.dumps(submodules, indent=2)+'\n')
    if args.sources_only:
        return
    dependencies = args.root / 'native_dependencies'
    if shutil.which('ninja') is None:
        (dependencies/'bin').mkdir(parents=True, exist_ok=True)
        archive = dependencies/'ninja-linux-v1.11.1.zip'
        if not archive.is_file():
            urllib.request.urlretrieve('https://github.com/ninja-build/ninja/releases/download/v1.11.1/ninja-linux.zip', archive)
        with zipfile.ZipFile(archive) as bundle:
            (dependencies/'bin/ninja').write_bytes(bundle.read('ninja'))
        (dependencies/'bin/ninja').chmod(0o755)
        os.environ['PATH'] = str(dependencies/'bin') + os.pathsep + os.environ['PATH']
    import torch
    from torch.utils.cpp_extension import load, CUDA_HOME
    if CUDA_HOME is None:
        raise RuntimeError('No CUDA compiler toolkit was found inside the CPU allocation')
    source = vendor/'quarot'
    module_name = 'quarot_official_gemm_fused' if args.fused else 'quarot_official_gemm'
    build = args.root/'native_extensions'/module_name
    build.mkdir(parents=True, exist_ok=True)
    header = (source/'quarot/kernels/include/gemm.h').read_text()
    kernel = (source/'quarot/kernels/gemm.cu').read_text()
    for text in (header, kernel):
        if text.count('int32_t *C') != 1:
            raise RuntimeError('Upstream signature changed; review the stream bridge')
    header = '#include <cuda_runtime_api.h>\n' + header.replace('int32_t *C', 'int32_t *C, cudaStream_t stream')
    kernel = kernel.replace('int32_t *C', 'int32_t *C, cudaStream_t stream')
    if kernel.count('gemmOp(arguments)') != 1:
        raise RuntimeError('Upstream GEMM launcher changed')
    kernel = kernel.replace('gemmOp(arguments)', 'gemmOp(arguments, nullptr, stream)')
    (build/'gemm.h').write_text(header)
    (build/'gemm.cu').write_text(kernel)
    (build/'binding.cpp').write_text(BINDING)
    (build/'epilogue.cu').write_text(EPILOGUE)
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'
    os.environ.setdefault('MAX_JOBS', '6')
    extension = load(name=module_name, sources=[str(build/'binding.cpp'), str(build/'gemm.cu'), str(build/'epilogue.cu')],
                     extra_include_paths=[str(build), str(source/'quarot/kernels/include'),
                                          str(source/'third-party/cutlass/include'),
                                          str(source/'third-party/cutlass/tools/util/include')],
                     extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '-lineinfo'],
                     build_directory=str(build), verbose=True)
    receipt = {'status': 'complete', 'pbs_job_id': os.environ['PBS_JOBID'], 'module': extension.__file__,
               'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'cuda_home': CUDA_HOME,
               'source_commit': (source/'UPSTREAM_COMMIT').read_text().strip(), 'submodules': submodules,
               'adapter_changes': ['explicit current CUDA stream argument', 'GEMM-only pybind bridge',
                                   'fused group scale/add epilogue with explicit FP32 round-to-nearest operations'],
               'gpu_execution_tested': False, 'new_training_updates': 0, 'new_closed_loop_episodes': 0}
    receipt_name = 'quarot_fused_build.json' if args.fused else 'quarot_build.json'
    (out/receipt_name).write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
