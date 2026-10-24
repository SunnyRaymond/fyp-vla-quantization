"""Standalone CUDA-Graph tuning for the existing decoded-BF16 cuBLAS path.

The selected decoder only materializes the same BF16 matrix as the reference;
the caller still invokes the existing Linear/cuBLAS implementation.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import statistics

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


FORMAT = "fastwam_additive_vq_microscale_v1"
EXPECTED_MODULES = 614
REFERENCE = "fused_vq._decode_kernel:block256_w4"
_REFERENCE_BLOCK = 256
_GRAPH_CAPTURE_REPLAYS = 8
_GRAPH_TIMING_REPLAYS = 12
_GRAPH_ROUNDS = 5
_PRIMARY_QUERIES = 80

# Every tile is a multiple of one B32 scale block and contains complete 4-value
# codewords, allowing indices and exponents to be reused across their lanes.
_CANDIDATES = (
    {"name": "block128_w4", "block": 128, "warps": 4},
    {"name": "block256_w4", "block": 256, "warps": 4},
    {"name": "block512_w4", "block": 512, "warps": 4},
    {"name": "block512_w8", "block": 512, "warps": 8},
    {"name": "block1024_w8", "block": 1024, "warps": 8},
)
_CONFIGS = {row["name"]: row for row in _CANDIDATES}
_KNOWN_CONFIGS = {"reference", *_CONFIGS}


@triton.jit
def _decode_vectorized(
    BOOKS, INDICES, ROW_SCALES, EXPONENTS, OUT,
    N: tl.constexpr, K: tl.constexpr,
    BS0: tl.constexpr, BS1: tl.constexpr, BS2: tl.constexpr,
    IS0: tl.constexpr, IS1: tl.constexpr, IS2: tl.constexpr,
    RS0: tl.constexpr, ES0: tl.constexpr, ES1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(1)
    scales_per_tile: tl.constexpr = BLOCK // 32
    scale_block = tl.program_id(0) * scales_per_tile + tl.arange(0, scales_per_tile)
    group_lane = tl.arange(0, 8)
    coord = tl.arange(0, 4)
    group = scale_block[:, None] * 8 + group_lane[None, :]
    valid_group = group * 4 < K

    code_ptr = INDICES + row * IS0 + group * IS1
    i0 = tl.load(code_ptr, valid_group, other=0).to(tl.int32)
    i1 = tl.load(code_ptr + IS2, valid_group, other=0).to(tl.int32)
    coord_3d = coord[None, None, :]
    code_mask = valid_group[:, :, None]
    v0 = tl.load(
        BOOKS + i0[:, :, None] * BS1 + coord_3d * BS2,
        code_mask, other=0,
    ).to(tl.float32)
    v1 = tl.load(
        BOOKS + BS0 + i1[:, :, None] * BS1 + coord_3d * BS2,
        code_mask, other=0,
    ).to(tl.float32)

    exponent = tl.load(
        EXPONENTS + row * ES0 + scale_block * ES1,
        scale_block < tl.cdiv(K, 32), other=127,
    ).to(tl.int32) - 127
    row_scale = tl.load(ROW_SCALES + row * RS0).to(tl.float32)
    scale = libdevice.ldexp(row_scale, exponent)
    # Keep the deployed order: FP32 book sum, ldexp, FP32 multiply, BF16 round.
    decoded = ((v0 + v1) * scale[:, None, None]).to(tl.bfloat16)
    col = group[:, :, None] * 4 + coord_3d
    tl.store(OUT + row * K + col, decoded, col < K)


def _choose_config(reference_ms, candidates):
    """Select only a bitwise-valid candidate that beat the measured reference."""
    if not math.isfinite(reference_ms) or reference_ms <= 0:
        raise ValueError("reference_ms must be finite and positive")
    valid = [
        row for row in candidates
        if row.get("bitwise") is True
        and math.isfinite(row.get("median_ms", float("nan")))
        and row["median_ms"] > 0
    ]
    if not valid:
        return "reference"
    best = min(valid, key=lambda row: row["median_ms"])
    return best["name"] if best["median_ms"] < reference_ms else "reference"


def _as_dict(prepared):
    return dict(
        rows=int(prepared.rows), width=int(prepared.width),
        books=prepared.books, indices=prepared.indices,
        row_scales=prepared.row_scales, exponents=prepared.exponents,
    )


def _launch_reference(prepared, out, kernel):
    values = _as_dict(prepared)
    rows, width = values["rows"], values["width"]
    view = out[: rows * width].view(rows, width)
    books, indices = values["books"], values["indices"]
    scales, exponents = values["row_scales"], values["exponents"]
    kernel[(triton.cdiv(rows * width, _REFERENCE_BLOCK),)](
        books, indices, scales, exponents, view,
        rows, width, books.stride(0), books.stride(1), books.stride(2),
        indices.stride(0), indices.stride(1), indices.stride(2),
        scales.stride(0), exponents.stride(0), exponents.stride(1), _REFERENCE_BLOCK,
        num_warps=4, enable_fp_fusion=False,
    )


def _launch_candidate(prepared, out, config):
    values = _as_dict(prepared)
    rows, width = values["rows"], values["width"]
    block = config["block"]
    view = out[: rows * width].view(rows, width)
    books, indices = values["books"], values["indices"]
    scales, exponents = values["row_scales"], values["exponents"]
    _decode_vectorized[(triton.cdiv(width, block), rows)](
        books, indices, scales, exponents, view,
        rows, width, books.stride(0), books.stride(1), books.stride(2),
        indices.stride(0), indices.stride(1), indices.stride(2),
        scales.stride(0), exponents.stride(0), exponents.stride(1), block,
        num_warps=config["warps"], enable_fp_fusion=False,
    )


def _measure_graph(call, device):
    """Return per-decode CUDA Graph timings; no eager launch is timed."""
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.device(device):
        with torch.cuda.stream(stream):
            call()  # compile/warm up before capture
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(_GRAPH_CAPTURE_REPLAYS):
                call()
        stream.synchronize()
        graph.replay()
        stream.synchronize()
        samples = []
        for _ in range(_GRAPH_ROUNDS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(_GRAPH_TIMING_REPLAYS):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(
                start.elapsed_time(end)
                / (_GRAPH_TIMING_REPLAYS * _GRAPH_CAPTURE_REPLAYS)
            )
    return {
        "median_ms": statistics.median(samples),
        "samples_ms": samples,
        "method": "CUDA Graph replay; per decode",
    }


def _bitwise_equal(torch_module, left, right, elements):
    return torch_module.equal(
        left[:elements].view(torch_module.int16),
        right[:elements].view(torch_module.int16),
    )


def _check_edge_cases(torch_module, device, kernel, selected_configs):
    """Check masks, partial codewords, B32 tails, zero scales and exponent ends."""
    rows = 3
    books = torch_module.linspace(-1.0, 1.0, 2 * 256 * 4, device=device).reshape(2, 256, 4)
    books = books.to(torch_module.bfloat16)
    row_scales = torch_module.tensor([0.0, 0.5, -0.25], device=device, dtype=torch_module.bfloat16)
    results = []
    for width in (1, 31, 32, 33, 63, 64, 65):
        groups = triton.cdiv(width, 4)
        blocks = triton.cdiv(width, 32)
        idx = torch_module.arange(rows * groups * 2, device=device, dtype=torch_module.int32)
        indices = (idx % 256).to(torch_module.uint8).view(rows, groups, 2)
        exp = torch_module.empty((rows, blocks), device=device, dtype=torch_module.uint8)
        edge_values = torch_module.tensor([0, 127, 254], device=device, dtype=torch_module.uint8)
        exp.copy_(edge_values[torch_module.arange(rows * blocks, device=device) % 3].view(rows, blocks))
        prepared = type("PreparedEdge", (), {
            "rows": rows, "width": width, "books": books, "indices": indices,
            "row_scales": row_scales, "exponents": exp,
        })()
        reference = torch_module.empty(rows * width, device=device, dtype=torch_module.bfloat16)
        output = torch_module.empty_like(reference)
        _launch_reference(prepared, reference, kernel)
        for config in selected_configs:
            _launch_candidate(prepared, output, config)
            if not _bitwise_equal(torch_module, reference, output, rows * width):
                raise RuntimeError(
                    f"Decoder edge preflight failed for width={width}, config={config['name']}"
                )
        results.append({"shape": [rows, width], "candidate_configs_checked": len(selected_configs)})
    torch_module.cuda.synchronize(device)
    return results


def smoke():
    """Compile and check tiny decoder cases before reading the large model bundle."""
    import os
    from reference_profile import guard, save, freeze_numerics
    import fused_vq

    guard()
    freeze_numerics(torch)
    if 'A100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Decoder smoke requires the frozen A100 hardware family')
    result = _check_edge_cases(torch, torch.device('cuda:0'), fused_vq._decode_kernel, _CANDIDATES)
    save(Path(os.environ['NEW_ROOT']) / 'results' / 'decoder_smoke.json',
         dict(pbs_jobid=os.environ['PBS_JOBID'], passed=True, cases=result))
    print('DECODER_SMOKE_PASSED', flush=True)


class DecoderDispatcher:
    """Callable decoder plus its serializable tuning and parity receipt."""

    def __init__(self, decisions, receipt, receipt_path, reference_kernel):
        self._decisions = decisions
        self._receipt = receipt
        self._receipt_path = Path(receipt_path)
        self._reference_kernel = reference_kernel

    def __call__(self, prepared, scratch):
        rows, width = int(prepared.rows), int(prepared.width)
        if scratch.dtype != torch.bfloat16 or not scratch.is_cuda or scratch.numel() < rows * width:
            raise ValueError("scratch must be a sufficiently large CUDA BF16 buffer")
        selected = self._decisions.get((rows, width))
        if selected is None:
            raise ValueError(f"No preflighted decoder decision for shape {(rows, width)}")
        if selected == "reference":
            _launch_reference(prepared, scratch, self._reference_kernel)
        else:
            _launch_candidate(prepared, scratch, _CONFIGS[selected])
        return scratch[: rows * width].view(rows, width)

    def report(self):
        return deepcopy(self._receipt)

    def update_frequencies(self, module_calls, *, query_count, source="candidate per-module logical calls including graph replay"):
        """Fill actual weights from the complete 80-query candidate panel."""
        if query_count != _PRIMARY_QUERIES:
            raise ValueError(f"frequency weights require the complete {_PRIMARY_QUERIES}-query panel")
        if not isinstance(module_calls, Mapping):
            raise TypeError("module_calls must map every module name to its logical call count")
        module_names = {row["name"] for row in self._receipt["modules"]}
        if set(module_calls) != module_names:
            raise ValueError("module_calls must contain exactly the 614 preflighted modules")
        calls = {}
        for name in module_names:
            value = module_calls[name]
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise ValueError(f"Invalid logical call count for {name}")
            calls[name] = int(value)
        if sum(calls.values()) <= 0:
            raise ValueError("At least one measured logical module call is required")

        shape_calls = {tuple(row["shape"]): 0 for row in self._receipt["shapes"]}
        for row in self._receipt["modules"]:
            count = calls[row["name"]]
            row["calls"] = count
            shape_calls[tuple(row["shape"])] += count
        for row in self._receipt["shapes"]:
            row["call_count"] = shape_calls[tuple(row["shape"])]

        reference_total = selected_total = 0.0
        for row in self._receipt["shapes"]:
            weight = row["call_count"]
            reference_total += weight * row["reference"]["median_ms"]
            chosen = row["reference"] if row["selected"] == "reference" else next(
                item for item in row["candidates"] if item["name"] == row["selected"]
            )
            selected_total += weight * chosen["median_ms"]
        self._receipt["call_frequency_status"] = "complete"
        self._receipt["call_frequency_source"] = source
        self._receipt["frequency_panel"] = {
            "query_count": query_count,
            "logical_decode_calls_full_panel": sum(calls.values()),
            "logical_decode_calls_per_query": sum(calls.values()) / query_count,
        }
        self._receipt["frequency_weighted"] = {
            "query_count": query_count,
            "logical_decode_calls_full_panel": sum(calls.values()),
            "reference_decode_ms_full_panel": reference_total,
            "selected_decode_ms_full_panel": selected_total,
            "reference_decode_ms_per_query": reference_total / query_count,
            "selected_decode_ms_per_query": selected_total / query_count,
            "selected_over_reference": selected_total / reference_total,
            "scope": "weighted decoder-only estimate; existing cuBLAS GEMM excluded",
        }
        self._write()
        return self.report()

    def _write(self):
        self._receipt_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._receipt_path.with_suffix(self._receipt_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self._receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self._receipt_path)


def _validate_reuse(path, modules, shape_rows, device, *, protocol, pbs_jobid,
                    checkpoint_identity, mode):
    reused = json.loads(Path(path).read_text(encoding="utf-8"))
    if reused.get("schema") != "fastwam_decoder_tuning_v1":
        raise ValueError("reuse tuning receipt has an unknown schema")
    if reused.get("protocol") != protocol or reused.get("pbs_jobid") != pbs_jobid:
        raise ValueError("reuse tuning receipt belongs to a different protocol or PBS job")
    if reused.get("mode") != "c_decoder" or mode != "c_buffer":
        raise ValueError("reuse is supported only from c_decoder to c_buffer")
    if reused.get("checkpoint_identity") != checkpoint_identity:
        raise ValueError("reuse tuning receipt belongs to a different checkpoint")
    if (reused.get("call_frequency_status") != "complete"
            or reused.get("frequency_panel", {}).get("query_count") != _PRIMARY_QUERIES
            or reused.get("frequency_weighted", {}).get("query_count") != _PRIMARY_QUERIES
            or reused.get("frequency_weighted", {}).get("logical_decode_calls_full_panel", 0) <= 0):
        raise ValueError("reuse receipt lacks complete 80-query call-frequency weights")
    parity = reused.get("parity", {})
    if (parity.get("checked_modules") != EXPECTED_MODULES
            or parity.get("bitwise_passed") != EXPECTED_MODULES
            or parity.get("candidate_failures") != []):
        raise ValueError("reuse receipt does not have a clean 614-module bitwise preflight")
    old_modules = {row["name"]: tuple(row["shape"]) for row in reused.get("modules", [])}
    module_rows = reused.get("modules", [])
    full_panel_calls = reused["frequency_panel"]["logical_decode_calls_full_panel"]
    recorded_calls = [row.get("calls") for row in module_rows]
    if (len(module_rows) != EXPECTED_MODULES
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                   for value in recorded_calls)
            or sum(recorded_calls) != full_panel_calls):
        raise ValueError("reuse receipt does not contain complete per-module call counts")
    weighted = reused["frequency_weighted"]
    for key in ("reference_decode_ms_full_panel", "selected_decode_ms_full_panel",
                "reference_decode_ms_per_query", "selected_decode_ms_per_query",
                "selected_over_reference"):
        if not isinstance(weighted.get(key), (int, float)) or not math.isfinite(weighted[key]):
            raise ValueError("reuse receipt has incomplete weighted decoder timing")
    if (weighted["reference_decode_ms_full_panel"] <= 0
            or weighted["selected_decode_ms_full_panel"] <= 0
            or abs(weighted["reference_decode_ms_per_query"]
                   - weighted["reference_decode_ms_full_panel"] / _PRIMARY_QUERIES) > 1e-9
            or abs(weighted["selected_decode_ms_per_query"]
                   - weighted["selected_decode_ms_full_panel"] / _PRIMARY_QUERIES) > 1e-9):
        raise ValueError("reuse receipt has inconsistent full-panel/per-query timings")
    current_modules = {
        name: (int(prepared.rows), int(prepared.width))
        for name, prepared in modules.items()
    }
    if old_modules != current_modules or len(old_modules) != EXPECTED_MODULES:
        raise ValueError("reuse tuning receipt does not match the current 614 module shapes")
    old_shapes = {tuple(row["shape"]): row for row in reused.get("shapes", [])}
    if set(old_shapes) != {tuple(row["shape"]) for row in shape_rows}:
        raise ValueError("reuse tuning receipt shape inventory differs")
    for row in shape_rows:
        old_row = old_shapes[tuple(row["shape"])]
        selected = old_row.get("selected")
        if selected not in _KNOWN_CONFIGS:
            raise ValueError(f"Unknown reused decoder choice: {selected!r}")
        reference = old_row.get("reference", {})
        if not isinstance(reference.get("median_ms"), (int, float)) or reference["median_ms"] <= 0:
            raise ValueError("reuse tuning receipt lacks a positive reference Graph timing")
        if selected != "reference":
            chosen = next((item for item in old_row.get("candidates", [])
                           if item.get("name") == selected), None)
            if (chosen is None or chosen.get("bitwise") is not True
                    or not isinstance(chosen.get("median_ms"), (int, float))
                    or not 0 < chosen["median_ms"] < reference["median_ms"]):
                raise ValueError("reuse tuning receipt has an invalid selected candidate")
        row.update(deepcopy(old_row))
        # Call frequencies belong to the current candidate capture, never the reused run.
        row["call_count"] = None
    receipt_device = reused.get("device", {})
    if receipt_device.get("name") != device["name"] or receipt_device.get("capability") != device["capability"]:
        raise ValueError("reuse tuning receipt was measured on a different GPU")
    return reused


def prepare(runtime, bundle, p, out, *, reuse_path=None):
    """Preflight all 614 prepared VQ weights and return a per-shape decoder.

    Call after the runtime installs its 614 `prepared` tensors and before the
    first anchor or CUDA Graph capture. `bundle` remains in the signature for
    the parent worker API; the live validated tensors are owned by runtime.banks.
    """
    del bundle
    pbs_jobid = os.environ.get("PBS_JOBID")
    if not pbs_jobid:
        raise RuntimeError("decoder tuning requires an active PBS_JOBID allocation")
    protocol = p.get("protocol") if isinstance(p, Mapping) else None
    if not protocol:
        raise ValueError("profiling protocol is required for decoder tuning")
    source = p.get("source", {})
    export_reuse = p.get("export_reuse", {})
    checkpoint_identity = {
        "source_job": source.get("source_job"),
        "source_protocol": source.get("source_protocol"),
        "source_export_job": export_reuse.get("source_export_job"),
        "bundle_path": export_reuse.get("bundle_path"),
    }
    if not checkpoint_identity["source_job"] or not checkpoint_identity["bundle_path"]:
        raise ValueError("protocol lacks source checkpoint/bundle identity")
    mode = Path(out).name
    if mode not in ("c_decoder", "c_buffer"):
        raise ValueError(f"unsupported decoder preflight output mode: {mode}")
    if p.get("profiling", {}).get("queries_per_mode") != _PRIMARY_QUERIES:
        raise ValueError(f"protocol must retain the {_PRIMARY_QUERIES}-query primary panel")
    banks = getattr(runtime, "banks", None)
    prepared_rows = getattr(banks, "prepared", None)
    modules = getattr(banks, "modules", None)
    if not isinstance(prepared_rows, Mapping) or not isinstance(modules, Mapping):
        raise ValueError("runtime must be installed with prepared VQ tensors and module metadata")
    if len(prepared_rows) != EXPECTED_MODULES or len(modules) != EXPECTED_MODULES:
        raise ValueError(f"Expected exactly {EXPECTED_MODULES} installed VQ modules")
    if set(prepared_rows) != set(modules):
        raise ValueError("prepared VQ module names do not match installed Linear modules")
    scratch = getattr(banks, "scratch", None)
    if scratch is None or scratch.dtype != torch.bfloat16 or not scratch.is_cuda:
        raise ValueError("runtime banks must provide a CUDA BF16 decode scratch buffer")

    for name, item in prepared_rows.items():
        rows, width = int(item.rows), int(item.width)
        if rows <= 0 or width <= 0:
            raise ValueError(f"Invalid VQ weight shape at {name}")
        if (tuple(item.books.shape) != (2, 256, 4) or item.books.dtype != torch.bfloat16
                or tuple(item.indices.shape) != (rows, triton.cdiv(width, 4), 2)
                or item.indices.dtype != torch.uint8
                or tuple(item.row_scales.shape) != (rows,) or item.row_scales.dtype != torch.bfloat16
                or tuple(item.exponents.shape) != (rows, triton.cdiv(width, 32))
                or item.exponents.dtype != torch.uint8):
            raise ValueError(f"Invalid additive VQ shape/dtype at {name}")
        if any(tensor.device != scratch.device for tensor in (
                item.books, item.indices, item.row_scales, item.exponents)):
            raise ValueError(f"VQ tensors and decode scratch are on different devices at {name}")
        if any(not tensor.is_cuda for tensor in (item.books, item.indices, item.row_scales, item.exponents)):
            raise ValueError(f"VQ tensors must be CUDA-resident at {name}")

    device_props = torch.cuda.get_device_properties(scratch.device)
    device = {
        "name": device_props.name,
        "capability": [int(device_props.major), int(device_props.minor)],
    }
    import fused_vq

    shapes = {}
    for name, item in prepared_rows.items():
        shape = (int(item.rows), int(item.width))
        shape_row = shapes.setdefault(shape, {"shape": list(shape), "module_count": 0, "call_count": None})
        shape_row["module_count"] += 1
    shape_rows = [shapes[key] for key in sorted(shapes)]
    module_rows = [
        {"name": name, "shape": [int(item.rows), int(item.width)], "calls": None}
        for name, item in prepared_rows.items()
    ]
    receipt = {
        "schema": "fastwam_decoder_tuning_v1",
        "protocol": protocol,
        "pbs_jobid": pbs_jobid,
        "mode": mode,
        "checkpoint_identity": checkpoint_identity,
        "format": FORMAT,
        "scope": "independent VQ decode into BF16 scratch; existing cuBLAS Linear unchanged",
        "expected_modules": EXPECTED_MODULES,
        "module_count": len(module_rows),
        "device": device,
        "reference": {"name": REFERENCE, "block": _REFERENCE_BLOCK, "warps": 4},
        "candidate_configs": [dict(row) for row in _CANDIDATES],
        "benchmark": {
            "method": "CUDA Graph replay; per decode",
            "graph_capture_replays": _GRAPH_CAPTURE_REPLAYS,
            "timing_replays_per_round": _GRAPH_TIMING_REPLAYS,
            "rounds": _GRAPH_ROUNDS,
        },
        "call_frequency_status": "pending",
        "call_frequency_source": None,
        "frequency_panel": None,
        "frequency_weighted": None,
        "reuse_source": str(reuse_path) if reuse_path is not None else None,
        "modules": module_rows,
        "shapes": shape_rows,
        "parity": {"checked_modules": 0, "bitwise_passed": 0, "candidate_failures": []},
    }
    if reuse_path is not None:
        completed_reuse = p.get('completed_candidate_reuse', {})
        expected_tuning_job = (completed_reuse['source_job']
            if 'c_decoder' in completed_reuse.get('modes', []) else pbs_jobid)
        reused = _validate_reuse(
            reuse_path, prepared_rows, shape_rows, device,
            protocol=protocol, pbs_jobid=expected_tuning_job,
            checkpoint_identity=checkpoint_identity, mode=mode,
        )
        receipt['reuse_source_pbs_jobid'] = expected_tuning_job
        receipt["benchmark"] = deepcopy(reused["benchmark"])
        receipt["reference"] = deepcopy(reused["reference"])
        receipt["tuning_method"] = "reused choice; repeated CUDA Graph tuning skipped"
    else:
        receipt["tuning_method"] = "bounded per-shape candidate tuning"

    reference_scratch = torch.empty_like(scratch)
    decisions = {}
    for shape_row in shape_rows:
        shape = tuple(shape_row["shape"])
        names = [name for name, item in prepared_rows.items()
                 if (int(item.rows), int(item.width)) == shape]
        representative = prepared_rows[names[0]]
        if reuse_path is None:
            ref_result = _measure_graph(
                lambda: _launch_reference(representative, scratch, fused_vq._decode_kernel),
                scratch.device,
            )
            candidate_rows = []
            for config in _CANDIDATES:
                _launch_reference(representative, reference_scratch, fused_vq._decode_kernel)
                _launch_candidate(representative, scratch, config)
                torch.cuda.synchronize(scratch.device)
                bitwise = _bitwise_equal(
                    torch, reference_scratch, scratch,
                    int(representative.rows) * int(representative.width),
                )
                timing = _measure_graph(
                    lambda config=config: _launch_candidate(representative, scratch, config),
                    scratch.device,
                )
                candidate_rows.append({
                    "name": config["name"], "block": config["block"], "warps": config["warps"],
                    "median_ms": timing["median_ms"], "samples_ms": timing["samples_ms"],
                    "bitwise": bool(bitwise),
                })
            selected = _choose_config(ref_result["median_ms"], candidate_rows)
            shape_row.update(
                reference=ref_result,
                candidates=candidate_rows,
                selected=selected,
                decision_reason=("fastest bitwise candidate beat reference" if selected != "reference"
                                 else "no bitwise candidate beat reference"),
            )
        else:
            selected = shape_row["selected"]
        decisions[shape] = selected

    # Check the chosen decoder on every real VQ matrix; a single mismatch sends
    # that entire shape back to the reference decoder before the callable escapes.
    for name, item in prepared_rows.items():
        shape = (int(item.rows), int(item.width))
        elements = shape[0] * shape[1]
        _launch_reference(item, reference_scratch, fused_vq._decode_kernel)
        selected = decisions[shape]
        if selected == "reference":
            _launch_reference(item, scratch, fused_vq._decode_kernel)
        else:
            _launch_candidate(item, scratch, _CONFIGS[selected])
        torch.cuda.synchronize(scratch.device)
        if not _bitwise_equal(torch, reference_scratch, scratch, elements):
            shape_row = next(row for row in shape_rows if tuple(row["shape"]) == shape)
            shape_row["candidate_failures"] = shape_row.get("candidate_failures", []) + [name]
            receipt["parity"]["candidate_failures"].append(name)
            shape_row["selected"] = "reference"
            shape_row["decision_reason"] = "selected candidate failed a real-module bitwise check"
            decisions[shape] = "reference"
            _launch_reference(item, scratch, fused_vq._decode_kernel)
            torch.cuda.synchronize(scratch.device)
            if not _bitwise_equal(torch, reference_scratch, scratch, elements):
                raise RuntimeError(f"Reference decoder bitwise self-check failed at {name}")
        receipt["parity"]["checked_modules"] += 1
        receipt["parity"]["bitwise_passed"] += 1

    receipt["edge_cases"] = _check_edge_cases(
        torch, scratch.device, fused_vq._decode_kernel, _CANDIDATES,
    )
    receipt["selected_by_shape"] = {
        "x".join(map(str, shape)): decision for shape, decision in decisions.items()
    }
    dispatcher = DecoderDispatcher(decisions, receipt, Path(out) / "tuning.json", fused_vq._decode_kernel)
    dispatcher._write()
    del reference_scratch
    torch.cuda.synchronize(scratch.device)
    return dispatcher
