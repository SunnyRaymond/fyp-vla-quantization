"""All-NFE DiT CUDA graphs for the existing packed VQ/BF16-GEMM engine."""
from __future__ import annotations
from pathlib import Path


class AllNfeGraphs:
    """Recompute cross K/V inside each graph; avoid two eager DiT passes per query."""
    def __init__(self, engine, torch):
        import engine as legacy_engine
        self.engine, self.torch, self.helpers = engine, torch, legacy_engine
        self.records = {}
        self.captures = self.replays = self.calls = 0
        self.original = engine.runtime.model.net.forward
        engine.runtime.model.net.forward = self.forward

    def forward(self, *args, **kwargs):
        torch, helper = self.torch, self.helpers
        self.calls += 1
        key = helper._signature(torch, (args, kwargs))
        record = self.records.get(key)
        if record is None:
            static = helper._clone_tree(torch, (args, kwargs))
            x = kwargs['x_B_C_T_H_W']
            side = torch.cuda.Stream(device=x.device)
            side.wait_stream(torch.cuda.current_stream(x.device))
            with torch.cuda.stream(side):
                self.original(*static[0], **static[1])
            torch.cuda.current_stream(x.device).wait_stream(side)
            torch.cuda.synchronize(x.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self.original(*static[0], **static[1])
            record = {'inputs': static, 'output': output, 'graph': graph}
            self.records[key] = record
            self.captures += 1
        else:
            helper._copy_tree(torch, record['inputs'], (args, kwargs))
        record['graph'].replay()
        self.replays += 1
        return helper._clone_tree(torch, record['output'])

    def metadata(self):
        return {'boundary': 'model.net.forward from NFE 1', 'captures': self.captures,
                'replays': self.replays, 'calls': self.calls, 'graphs': len(self.records),
                'cross_kv_policy': 'recomputed within graph at every NFE',
                'single_request_single_stream': True}

    def restore(self):
        self.engine.runtime.model.net.forward = self.original
        self.records.clear()


def install(runtime, protocol, old_root, out):
    from common import torch_module
    from engine_profile import _install_arm
    torch = torch_module()
    if getattr(runtime, 'ours_vqwam_engine', None) is None:
        _install_arm(runtime, protocol, Path(old_root), Path(out), 'ours_vqwam', {})
    engine = runtime.ours_vqwam_engine
    if hasattr(runtime, 'optimized_vq_graphs'):
        raise RuntimeError('Optimized engine is already installed')
    engine.graphs.restore()
    engine.graphs.records.clear()
    engine.banks.graph_workspaces.clear()
    engine.banks.workspace = engine.banks.default_workspace
    old_cache_forward = engine.cache._forward

    def direct_kv(name, slot, value, args=(), kwargs=None):
        return slot['original'](value, *args, **({} if kwargs is None else kwargs))

    engine.cache._forward = direct_kv
    for slot in engine.cache.slots.values():
        slot.update(input=None, snapshot=None, buffer=None, buffers={}, valid=False, matches=0)
    torch.cuda.empty_cache()
    graphs = AllNfeGraphs(engine, torch)
    runtime.optimized_vq_graphs = graphs
    original_end = runtime.end_query

    def end_query():
        original_end()
        metadata = dict(runtime.install_metadata)
        metadata.update(engine_variant='all_nfe_graph_recompute_cross_kv_v1',
                        optimized_cuda_graph=graphs.metadata(),
                        parity_status='pending_new_full80_gate',
                        native_int4_kernel_used=False,
                        execution_scope='all five actual DiT evaluations graphed; packed VQ decode and BF16 GEMM; cross K/V recomputed')
        runtime.install_metadata = metadata

    runtime.end_query = end_query

    restored = False

    def restore():
        nonlocal restored
        if restored:
            return
        graphs.restore()
        engine.cache._forward = old_cache_forward
        runtime.end_query = original_end
        runtime.__dict__.pop('optimized_vq_graphs', None)
        engine.restore()
        if getattr(runtime, 'restore_optimized_engine', None) is restore:
            runtime.__dict__.pop('restore_optimized_engine', None)
        restored = True

    runtime.restore_optimized_engine = restore
    metadata = dict(runtime.install_metadata)
    metadata.update(engine_variant='all_nfe_graph_recompute_cross_kv_v1',
                    parity_status='pending_new_full80_gate', optimized_cuda_graph=graphs.metadata(),
                    native_int4_kernel_used=False,
                    execution_scope='all five actual DiT evaluations graphed; packed VQ decode and BF16 GEMM; cross K/V recomputed')
    runtime.install_metadata = metadata
    return metadata
