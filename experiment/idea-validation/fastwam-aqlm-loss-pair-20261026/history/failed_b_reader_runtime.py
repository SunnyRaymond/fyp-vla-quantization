"""Park student dense weights while differentiating the immutable BF16 reader."""
from contextlib import contextmanager
import gc


@contextmanager
def reader_session(torch, runtime, banks, masters):
    from frozen_reader import BF16Reader
    first = next(runtime.model.parameters())
    device = first.device
    parked = {}
    weights = {}
    reader = None
    succeeded = False
    try:
        # Keeping the first model parameter on GPU preserves model.device.
        for name, module in banks.modules.items():
            if module.weight is not first:
                parked[name] = module.weight.detach().cpu()
                module.weight.data = parked[name]
        gc.collect()
        torch.cuda.empty_cache()
        weights = {name: value.to(device=device, copy=True) for name, value in masters.items()}
        reader = BF16Reader(runtime, weights)
        yield reader
        succeeded = True
    finally:
        if reader is not None:
            reader.original_weights.clear()
        weights.clear()
        gc.collect()
        torch.cuda.empty_cache()
        if succeeded:
            for name, value in parked.items():
                banks.modules[name].weight.data = value.to(device)


def compact_counts(counts):
    return {name: value for name, value in counts.items()
            if name not in ('action_initial_noise', 'action_conditioning_latents')}
