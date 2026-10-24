"""Capture fixed denoising regions while preserving the original kernel sequence."""
from __future__ import annotations

from contextlib import contextmanager
import time
from collections import Counter


def tree_map(function, value):
    if isinstance(value, dict):
        return {key: tree_map(function, part) for key, part in value.items()}
    if isinstance(value, tuple):
        return tuple(tree_map(function, part) for part in value)
    if isinstance(value, list):
        return [tree_map(function, part) for part in value]
    return function(value)


def copy_inputs(torch, target, source):
    if isinstance(target, dict):
        if target.keys() != source.keys():
            raise ValueError("Graph keyword set changed")
        for key in target:
            copy_inputs(torch, target[key], source[key])
    elif isinstance(target, (tuple, list)):
        if type(target) is not type(source) or len(target) != len(source):
            raise ValueError("Graph input structure changed")
        for left, right in zip(target, source):
            copy_inputs(torch, left, right)
    elif isinstance(target, torch.Tensor):
        if (tuple(target.shape), target.dtype, target.device) != (
                tuple(source.shape), source.dtype, source.device):
            raise ValueError("Frozen graph tensor layout changed")
        if target.data_ptr() != source.data_ptr():
            target.copy_(source)
    elif target != source:
        raise ValueError("Graph scalar argument changed")


class Regions:
    def __init__(self, runtime, optimization, compiled=False):
        import torch

        self.torch, self.runtime, self.optimization = torch, runtime, optimization
        self.banks, self.cache = runtime.banks, optimization.cache
        self.compiled = compiled
        self.regions = {}
        self.prepared_queries = {}
        self.logical_replayed_target_calls = 0

    @contextmanager
    def capture_safe(self):
        previous = self.banks.diagnostics
        tracking = self.banks.track
        self.banks.diagnostics = False
        if self.compiled:
            self.banks.track = False
        try:
            with self.cache.graph_capture():
                yield
        finally:
            self.banks.diagnostics = previous
            self.banks.track = tracking

    def prepare(self, expert, kwargs):
        if expert is not None and self.prepared_queries.get(expert) != self.cache._query:
            self.cache.prepare_expert(expert, kwargs["context"])
            self.prepared_queries[expert] = self.cache._query

    def wrap(self, label, original, expert=None):
        torch = self.torch

        def run(**kwargs):
            self.prepare(expert, kwargs)
            if label not in self.regions:
                begin = time.perf_counter()
                # Eager warmup also validates this region's new workspace operands.
                original(**kwargs)
                torch.cuda.synchronize()
                if self.compiled:
                    record = dict(captures=0, replays=0, capture_target_calls=0,
                                  captured_modules=[], compile_calls=0)
                    before = sum(self.banks.calls.values())
                    seen_before = set(self.banks.seen)
                    original(**kwargs)
                    record["capture_target_calls"] = sum(self.banks.calls.values()) - before
                    record["captured_modules"] = sorted(set(self.banks.seen) - seen_before)
                    function = torch.compile(original, backend="inductor", fullgraph=False,
                        dynamic=False, options={"triton.cudagraphs": False,
                                                "emulate_precision_casts": True})
                    with self.capture_safe():
                        output = function(**kwargs)
                    torch.cuda.synchronize()
                    record.update(function=function, setup_seconds=time.perf_counter() - begin,
                                  compile_calls=1)
                    self.regions[label] = record
                    self.optimization.compiled["regions"][label] = {"backend": "inductor",
                        "fullgraph": False, "dynamic": False, "cudagraphs": False,
                        "emulate_precision_casts": True, "compile_threads": 1,
                        "setup_seconds": record["setup_seconds"]}
                    return output

                def clone(value):
                    return value.clone(memory_format=torch.preserve_format) if isinstance(value, torch.Tensor) else value

                static = tree_map(clone, kwargs)
                # Prefill graph owns these outputs and keeps their addresses stable.
                for key in ("video_cache_k", "video_cache_v"):
                    if key in kwargs:
                        static[key] = kwargs[key]
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    original(**static)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                before = sum(self.banks.calls.values())
                before_modules = Counter(self.banks.calls)
                old_seen = set(self.banks.seen)
                with self.capture_safe():
                    with torch.cuda.graph(graph):
                        output = original(**static)
                captured = sum(self.banks.calls.values()) - before
                record = dict(graph=graph, inputs=static, output=output,
                    captures=1, replays=0, capture_target_calls=captured,
                    capture_calls_by_module=dict(Counter(self.banks.calls) - before_modules),
                    captured_modules=sorted(set(self.banks.seen) - old_seen),
                    setup_seconds=time.perf_counter() - begin)
                self.regions[label] = record
                copy_inputs(torch, static, kwargs)
                graph.replay()
                record["replays"] += 1
                self.logical_replayed_target_calls += captured
                return output

            record = self.regions[label]
            if self.compiled:
                with self.capture_safe():
                    output = record["function"](**kwargs)
                record["compile_calls"] += 1
                self.logical_replayed_target_calls += record["capture_target_calls"]
                return output
            copy_inputs(torch, record["inputs"], kwargs)
            record["graph"].replay()
            record["replays"] += 1
            self.logical_replayed_target_calls += record["capture_target_calls"]
            return record["output"]

        return run

    def reset_metrics(self):
        self.logical_replayed_target_calls = 0
        for record in self.regions.values():
            record["replays"] = 0
            if "compile_calls" in record:
                record["compile_calls"] = 0

    def module_call_counts(self):
        counts = Counter(self.banks.calls)
        for record in self.regions.values():
            for name, calls in record.get("capture_calls_by_module", {}).items():
                counts[name] += calls * record["replays"]
        return dict(counts)

    def report(self):
        result = dict(kind="torch_compile" if self.compiled else "cuda_graph",
                      logical_replayed_target_calls=self.logical_replayed_target_calls,
                      regions={key: {name: value for name, value in record.items()
                          if name not in ("graph", "inputs", "output", "function", "capture_calls_by_module")}
                          for key, record in self.regions.items()})
        if self.compiled:
            from torch._dynamo.utils import counters
            result["dynamo_counters"] = {str(key): {str(k): int(v) for k, v in row.items()}
                                        for key, row in counters.items()}
            result["unique_compiled_graphs"] = int(counters["stats"]["unique_graphs"])
            result["coverage_basis"] = "logical region calls times eager-observed calls per region; compiler may fuse operations"
        return result


def install(runtime, optimization, compiled=False):
    regions = Regions(runtime, optimization, compiled=compiled)
    model = runtime.model
    model._denoise_video = regions.wrap("video_step", model._denoise_video, "video")
    model.mot.prefill_video_cache_tensor = regions.wrap(
        "video_prefill", model.mot.prefill_video_cache_tensor)
    model._denoise_action_with_video_cache = regions.wrap(
        "action_step", model._denoise_action_with_video_cache, "action")
    optimization.graph = regions
    optimization.compiled["enabled"] = compiled
    return regions
