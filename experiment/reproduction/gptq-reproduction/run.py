import argparse
import contextlib
import io
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import torch

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'upstream'))
import opt

p = argparse.ArgumentParser()
p.add_argument('--model', choices=['opt-125m', 'opt-6.7b'], required=True)
p.add_argument('--method', choices=['fp16', 'rtn4', 'gptq4', 'gptq3'], required=True)
a = p.parse_args()
torch.set_num_threads(4)
torch.manual_seed(0)
manifest = json.loads((root / 'manifest.json').read_text())
data = torch.load(root / f'{a.model}-data.pt', map_location='cpu')
bits = 16 if a.method == 'fp16' else int(a.method[-1])
opt.args = SimpleNamespace(nsamples=128, wbits=bits, sym=False, trits=False, percdamp=.01, groupsize=-1, act_order=False, static_groups=False, nearest=a.method == 'rtn4')
started = time.monotonic()
model = opt.get_opt(manifest['models'][a.model]['path']).eval()
assert next(model.parameters()).dtype == torch.float16
torch.cuda.reset_peak_memory_stats()
quant_seconds = None  # RTN quantization is integrated into upstream opt_eval.
if a.method.startswith('gptq'):
    loader = []
    for inp in data['calibration']:
        tar = inp.clone(); tar[:, :-1] = -100
        loader.append((inp, tar))
    t = time.monotonic()
    opt.opt_sequential(model, loader, torch.device('cuda:0'))
    torch.cuda.synchronize()
    quant_seconds = time.monotonic() - t
capture = io.StringIO()
class Tee:
    def write(self, value):
        sys.__stdout__.write(value); sys.__stdout__.flush(); capture.write(value)
    def flush(self):
        sys.__stdout__.flush()
eval_started = time.monotonic()
with contextlib.redirect_stdout(Tee()):
    opt.opt_eval(model, SimpleNamespace(input_ids=data['test']), torch.device('cuda:0'))
eval_seconds = time.monotonic() - eval_started
ppl = float(capture.getvalue().strip().splitlines()[-1])
assert math.isfinite(ppl) and ppl > 0
result = {'model': a.model, 'method': a.method, 'ppl': ppl, 'quant_seconds': quant_seconds, 'eval_seconds_including_rtn_if_applicable': eval_seconds, 'total_seconds': time.monotonic() - started, 'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30, 'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30, 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'model_revision': manifest['models'][a.model]['revision'], 'upstream_commit': '2d65066eeb06a5c9ff5184d8cebdf33662c67faf', 'evaluated_tokens': (data['test'].numel() // 2048) * 2048, 'seed': 0, 'calibration_samples': 128, 'group_size': -1, 'act_order': False}
(root / 'results').mkdir(exist_ok=True)
out = root / 'results' / f'{a.model}-{a.method}.json'
partial = out.with_suffix('.json.partial')
partial.write_text(json.dumps(result, indent=2))
partial.replace(out)
print('RESULT', json.dumps(result), flush=True)
