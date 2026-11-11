"""CUDA graphs of the three LAQ phase calls on Fast-WAM: one graph per video denoising step (the refined future rows
differ by step), one for the prefill with its K/V quantization, one shared by the ten action steps.

Each region is captured the first time it is met: its arguments are cloned into fixed buffers, the call runs once on a
side stream (warm-up, as the BF16 engine's graph_runtime does), the gaze accumulators are put back, and the call is
captured into a pool shared by all regions (they replay in capture order). From then on a call copies its arguments into
the buffers and replays. Tensors returned by one region (the prefill cache) are passed to later regions as they are.

Replay runs no Python, so everything the Python side of a call decides must be the same at every call:
  - the refined rows come from State.static, fixed index buffers that refresh() rewrites when the selection changes;
  - State.graphing turns off the text-site cache and the identity memo (their hit / miss decisions would be frozen);
  - the prefill's count of observed layers (hop_layers, a Python counter) is added after each replay.
"""
from __future__ import annotations

import time

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten


def make_static(leaves, persistent):
    """Fixed buffers for the tensor leaves (tensors another region returned are used as they are)."""
    return [x if torch.is_tensor(x) and persistent.get(id(x)) is x else x.clone() if torch.is_tensor(x) else x
            for x in leaves]


def feed(static, leaves):
    """Copy a call's arguments into the fixed buffers; anything that is not a tensor must equal the captured value."""
    if len(static) != len(leaves):
        raise RuntimeError("Graph argument structure changed")
    copied = 0
    for buf, new in zip(static, leaves):
        if torch.is_tensor(buf):
            if new is buf:
                continue
            if not torch.is_tensor(new) or new.shape != buf.shape or new.dtype != buf.dtype or new.device != buf.device:
                raise RuntimeError(f"Graph tensor argument changed layout: {tuple(buf.shape)} vs "
                                   f"{tuple(new.shape) if torch.is_tensor(new) else type(new)}")
            buf.copy_(new)
            copied += 1
        elif new is not buf and new != buf:
            raise RuntimeError(f"Graph scalar argument changed: {buf!r} -> {new!r}")
    return copied


class Region:
    def __init__(self, key):
        self.key, self.graph, self.static, self.spec, self.out = key, None, None, None, None
        self.hops, self.replays, self.seconds = 0, 0, 0.0


class Graphs:
    def __init__(self, st):
        self.st = st
        self.regions = {}
        self.persistent = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.order = []
        # The integer runtime computes the text sites at the first action step and reuses them, so that step differs.
        self.split_action = getattr(st, "native", None) is not None

    @staticmethod
    def key(kind, step):
        return (kind, 0 if kind == "action" else int(step))

    def install(self):
        if self.st.static is None:
            raise RuntimeError("Graph mode needs State.freeze() first")
        self.st.graphing = True
        self.st.call = self

    def uninstall(self):
        self.st.graphing = False
        self.st.call = lambda kind, step, fn, kwargs: fn(**kwargs)

    def __call__(self, kind, step, fn, kwargs):
        key = self.key(kind, step)
        if self.split_action and kind == "action" and step > 0:
            key = ("action", 1)
        region = self.regions.get(key)
        if region is None:
            region = self.regions[key] = self.capture(key, fn, kwargs)
        leaves, spec = tree_flatten(kwargs)
        if spec != region.spec:
            raise RuntimeError(f"Graph {key}: argument structure changed")
        feed(region.static, leaves)
        region.graph.replay()
        region.replays += 1
        self.st.hop_layers += region.hops
        return region.out

    def capture(self, key, fn, kwargs):
        st = self.st
        begin = time.perf_counter()
        region = Region(key)
        leaves, region.spec = tree_flatten(kwargs)
        region.static = make_static(leaves, self.persistent)
        static_kwargs = tree_unflatten(region.static, region.spec)
        saved = (st.acc_action.clone(), st.acc_hop.clone(), st.hop_layers)
        st.shared = None
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fn(**static_kwargs)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        st.acc_action.copy_(saved[0])
        st.acc_hop.copy_(saved[1])
        st.hop_layers = saved[2]
        st.shared = None
        region.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(region.graph, pool=self.pool):
            region.out = fn(**static_kwargs)
        region.hops = st.hop_layers - saved[2]
        st.hop_layers = saved[2]
        st.shared = None
        for leaf in tree_flatten(region.out)[0]:
            if torch.is_tensor(leaf):
                self.persistent[id(leaf)] = leaf
        region.seconds = time.perf_counter() - begin
        self.order.append(key)
        return region

    def report(self):
        return dict(regions={f"{k[0]}.{k[1]}": dict(replays=r.replays, hops=r.hops, capture_seconds=r.seconds)
                             for k, r in self.regions.items()},
                    capture_order=[f"{k[0]}.{k[1]}" for k in self.order])
