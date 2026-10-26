"""One-query, five-NFE state and block-error diagnostic for native rotations.

This is a diagnostic only. It does not run the frozen 80-query gate or benchmark.
All model work requires a genuine PBS compute-node allocation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import random
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import perf_native


FORMAT = "cosmos_rotation_parity_diagnostic_v1"
QUERY = (3000, 0)
NFE_COUNT = 5
BLOCK_COUNT = 28
SAMPLED_ROWS = 32
ARMS = ("spinquant_w4a4", "quarot_w4a4")
PHASES = ("A", "A_repeat", "B", "A_after")
OUTPUT_RELATIVE_PATH = Path("results") / "rotation_parity" / "summary.json"


def uniform_row_indices(total_rows: int, limit: int = SAMPLED_ROWS) -> list[int]:
    if total_rows <= 0 or limit <= 0:
        raise ValueError("row count and sample limit must be positive")
    count = min(total_rows, limit)
    if count == 1:
        return [0]
    return [round(i * (total_rows - 1) / (count - 1)) for i in range(count)]


def discover_blocks(net, expected: int = BLOCK_COUNT) -> list[tuple[int, str, Any]]:
    """Find the shallowest named_modules entry for each blocks.<index> wrapper."""
    candidates: dict[int, list[tuple[int, str, Any]]] = {}
    for name, module in net.named_modules():
        match = re.search(r"(?:^|\.)blocks\.(\d+)(?:\.|$)", name)
        if match is None:
            continue
        index = int(match.group(1))
        if not 0 <= index < expected:
            continue
        suffix = name[match.end():]
        candidates.setdefault(index, []).append((len(suffix), name, module))

    selected = []
    seen_modules = set()
    for index in range(expected):
        options = sorted(candidates.get(index, ()), key=lambda row: (row[0], len(row[1]), row[1]))
        if not options:
            raise RuntimeError(f"Could not find DiT block {index} via named_modules() blocks.<i> paths")
        _, name, module = options[0]
        identity = id(module)
        if identity in seen_modules:
            raise RuntimeError(f"A single module was selected for multiple DiT blocks: {name}")
        seen_modules.add(identity)
        selected.append((index, name, module))
    return selected


def _reset_replay_state(common, runtime, rng_state, restore_rng: Callable[[Any], None], forward):
    """Clear per-query state first, then restore RNG immediately before replay."""
    common.begin_query(runtime)
    try:
        restore_rng(rng_state)
        return forward()
    finally:
        common.end_query(runtime)


def run_arm_phases(*, install_reference, snapshot_rng, replay, install_native,
                   restore_decoded_reference, cleanup_to_bf16):
    """Enforce reference A, repeat, native B, decoded A-after, and unconditional cleanup."""
    try:
        install_reference()
        rng_state = snapshot_rng()
        rows = {"A": replay("A", rng_state),
                "A_repeat": replay("A_repeat", rng_state)}
        install_native()
        rows["B"] = replay("B", rng_state)
        restore_decoded_reference()
        rows["A_after"] = replay("A_after", rng_state)
        return rows
    finally:
        cleanup_to_bf16()


def _snapshot_rng(torch) -> dict:
    state = {"python": random.getstate(), "torch_cpu": torch.get_rng_state().clone()}
    if torch.cuda.is_available():
        state["torch_cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    return state


def _restore_rng(state: dict, torch) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if "numpy" in state:
        import numpy as np
        np.random.set_state(state["numpy"])


def _first_tensor(value, torch, path="output"):
    if torch.is_tensor(value):
        return path, value
    if isinstance(value, dict):
        for key, child in value.items():
            found = _first_tensor(child, torch, f"{path}.{key}")
            if found is not None:
                return found
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            found = _first_tensor(child, torch, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _sample_output_tensor(value, torch):
    if value.ndim < 1:
        raise RuntimeError("DiT block output tensor has no feature dimension")
    flat = value.reshape(-1, int(value.shape[-1]))
    indices = uniform_row_indices(int(flat.shape[0]))
    index_tensor = torch.tensor(indices, device=value.device, dtype=torch.long)
    sample = flat.index_select(0, index_tensor).detach().to(device="cpu", copy=True)
    return sample, indices, int(flat.shape[0]), int(value.shape[-1])


def _training_receipt(runtime, torch) -> dict:
    net = runtime.model.net
    dropout = []
    for name, module in runtime.model.named_modules():
        if "dropout" in type(module).__name__.lower():
            dropout.append({"name": name, "type": type(module).__name__,
                            "training": bool(module.training),
                            "p": getattr(module, "p", None)})
    return {"model_training": bool(runtime.model.training),
            "net_training": bool(net.training), "dropout_modules": dropout}


def _replay_capture(common, runtime, capture, torch, device, rng_state,
                    block_modules, training_state):
    net = runtime.model.net
    block_rows: dict[int, dict] = {}
    outputs = []
    active_nfe = {"value": None}
    handles = []

    for index, name, module in block_modules:
        def after_block(_module, _inputs, output, *, _index=index, _name=name):
            nfe = active_nfe["value"]
            if nfe is None:
                raise RuntimeError("Block hook fired outside a tagged NFE")
            key = (_index, nfe)
            if key in block_rows:
                raise RuntimeError(f"DiT block ran more than once in one NFE: {_name}/NFE{nfe}")
            found = _first_tensor(output, torch)
            if found is None:
                raise RuntimeError(f"DiT block has no Tensor output to trace: {_name}")
            output_path, tensor = found
            sample, indices, total_rows, feature_width = _sample_output_tensor(tensor, torch)
            block_rows[key] = {"block": _index, "module": _name, "nfe": nfe,
                               "output_path": output_path, "rows": indices,
                               "total_rows": total_rows, "feature_width": feature_width,
                               "sample": sample}

        handles.append(module.register_forward_hook(after_block))

    def forward_five():
        with torch.inference_mode():
            for nfe, call in enumerate(capture["calls"]):
                active_nfe["value"] = nfe
                args = perf_native._device_tree(call["args"], device, torch, f"nfe{nfe}.args")
                kwargs = perf_native._device_tree(call["kwargs"], device, torch,
                                                  f"nfe{nfe}.kwargs")
                output = net(*args, **kwargs)
                outputs.append(perf_native._cpu_tree(output, torch, f"nfe{nfe}.output"))
                expected = {(index, nfe) for index, _, _ in block_modules}
                observed = {key for key in block_rows if key[1] == nfe}
                if observed != expected:
                    missing = sorted(expected - observed)
                    raise RuntimeError(f"Block trace missed expected block outputs at NFE{nfe}: {missing[:4]}")
        torch.cuda.synchronize(device)
        return outputs

    try:
        result = _reset_replay_state(common, runtime, rng_state,
                                     lambda state: _restore_rng(state, torch), forward_five)
    finally:
        for handle in handles:
            handle.remove()
    if len(result) != NFE_COUNT:
        raise RuntimeError(f"Expected exactly five raw DiT forwards, observed {len(result)}")
    if len(block_rows) != BLOCK_COUNT * NFE_COUNT:
        raise RuntimeError(f"Expected 28 blocks x five NFEs, observed {len(block_rows)} traces")
    return {"outputs": result, "blocks": block_rows,
            "training": training_state}


def _brief_metrics(reference, candidate, torch) -> dict:
    row = perf_native.compare_tensor_trees(reference, candidate, torch)
    return {key: row[key] for key in ("elements", "nrmse", "rmse", "max_abs")}


def _small_install_metadata(metadata) -> dict:
    keys = ("arm", "execution", "implementation", "transform", "rotation",
            "artifact_dir", "native_backend", "native_backend_module",
            "native_backend_module_path", "native_backend_build_host",
            "native_backend_source_revision", "native_backend_cutlass_revision",
            "native_backend_source_commit", "native_backend_build_pbs_job_id",
            "native_backend_build_receipt", "adapter_version",
            "native_hadamard_backend", "native_hadamard_module",
            "native_hadamard_source_repository", "native_hadamard_source_commit",
            "native_hadamard_build_pbs_job_id", "native_hadamard_build_receipt",
            "native_hadamard_build_host",
            "adapter_epilogue", "weight_group_size", "weight_scale_dtype",
            "activation_scale_dtype")
    return {key: metadata[key] for key in keys
            if isinstance(metadata, dict) and key in metadata
            and (metadata[key] is None or isinstance(metadata[key],
                (str, int, float, bool, list, dict)))}


def _phase_comparisons(rows, torch) -> dict:
    reference = rows["A"]
    summary = {"full_net_per_nfe": {}, "blocks": []}
    for phase in ("A_repeat", "B", "A_after"):
        candidate = rows[phase]
        summary["full_net_per_nfe"][phase] = [
            {"nfe": nfe, **_brief_metrics(reference["outputs"][nfe],
                                          candidate["outputs"][nfe], torch)}
            for nfe in range(NFE_COUNT)]

    for block in range(BLOCK_COUNT):
        base_name = rows["A"]["blocks"][(block, 0)]["module"]
        block_summary = {"block": block, "module": base_name, "per_nfe": []}
        for nfe in range(NFE_COUNT):
            base = rows["A"]["blocks"][(block, nfe)]
            per_nfe = {"nfe": nfe, "selected_rows": base["rows"],
                       "total_rows": base["total_rows"],
                       "feature_width": base["feature_width"],
                       "output_path": base["output_path"], "relative_to_A": {}}
            for phase in ("A_repeat", "B", "A_after"):
                current = rows[phase]["blocks"][(block, nfe)]
                identity_keys = ("module", "output_path", "rows", "total_rows", "feature_width")
                if any(base[key] != current[key] for key in identity_keys):
                    raise RuntimeError(
                        f"Block trace selection/shape changed: {phase}/block{block}/NFE{nfe}")
                per_nfe["relative_to_A"][phase] = _brief_metrics(
                    base["sample"], current["sample"], torch)
            block_summary["per_nfe"].append(per_nfe)
        summary["blocks"].append(block_summary)
    return summary


def _arm_diagnostic(common, runtime, protocol, old_root, root, out,
                    arm, capture, torch, device):
    selected_blocks = discover_blocks(runtime.model.net)
    training = _training_receipt(runtime, torch)
    install_state = {"native_metadata": None, "native_restore": None,
                     "reference_metadata": None}

    def install_reference():
        perf_native._restore_to_bf16(runtime, protocol, old_root,
                                     install_state.get("native_restore"))
        install_state["native_restore"] = None
        install_state["reference_metadata"] = perf_native._install_reference(
            runtime, protocol, old_root, arm)

    def install_native():
        metadata, restore = perf_native._install_native(
            runtime, protocol, old_root, out, arm, "current", {}, native_root=root)
        install_state["native_metadata"] = metadata
        install_state["native_restore"] = restore

    def replay(label, rng_state):
        return _replay_capture(common, runtime, capture, torch, device,
                               rng_state, selected_blocks, training)

    def cleanup():
        perf_native._restore_to_bf16(runtime, protocol, old_root,
                                     install_state.get("native_restore"))
        install_state["native_restore"] = None

    rows = run_arm_phases(
        install_reference=install_reference,
        snapshot_rng=lambda: _snapshot_rng(torch),
        replay=replay,
        install_native=install_native,
        restore_decoded_reference=install_reference,
        cleanup_to_bf16=cleanup)
    return {"arm": arm, "training_state": training,
            "reference_install": install_state["reference_metadata"],
            "native_install": install_state["native_metadata"],
            "replay_comparisons": _phase_comparisons(rows, torch)}


def _write_summary(root: Path, out: Path, summary: dict) -> None:
    payload = json.dumps(summary, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    if len(payload) > perf_native.MAX_SUMMARY_BYTES:
        raise ValueError(f"Diagnostic summary exceeds {perf_native.MAX_SUMMARY_BYTES} bytes")
    for path in (root / OUTPUT_RELATIVE_PATH, out / "summary.json"):
        perf_native._bounded_summary(path, summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    old_root, root, out = perf_native.validate_paths(args.old_root, args.root, args.out)
    protocol = perf_native.read_json(old_root / "protocol.json")
    perf_native.validate_protocol(protocol, ARMS[0])
    code_root = perf_native._prepare_code_imports(old_root, None)
    perf_native._prepare_rotation_backend_import(root)
    import common
    perf_native._allocation_guard(common)
    root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    receipt = {"format": FORMAT, "status": "loading_runtime", "complete": False,
               "diagnostic_only": True, "protocol": protocol["protocol"],
               "query": list(QUERY), "nfe_count": NFE_COUNT,
               "arms_requested": list(ARMS), "arms": [],
               "sampling": {"block_count": BLOCK_COUNT, "rows_per_block_output": SAMPLED_ROWS,
                            "policy": "flatten block output over all leading axes, then choose up to 32 uniformly spaced rows including endpoints; same row indices required across A, A_repeat, B, A_after",
                            "coverage_limit": "block-output trace only; not full-token, full-query-panel, or formal gate evidence"},
               "training_updates": 0, "closed_loop_episodes": 0,
               "created_utc": datetime.now(timezone.utc).isoformat()}
    _write_summary(root, out, receipt)
    runtime = torch = None
    try:
        from startup_trace import startup_trace
        with startup_trace(out):
            torch = common.torch_module()
            runtime = common.prepare_runtime(protocol, old_root, out)
        device = next(runtime.model.parameters()).device
        if device.type != "cuda":
            raise RuntimeError(f"A GPU allocation is required; model loaded on {device}")
        hardware = perf_native._gpu_receipt(torch, device)
        hardware_protocol_path = root / "native_protocol.json"
        hardware_protocol = (perf_native.read_json(hardware_protocol_path)
                             if hardware_protocol_path.is_file() else protocol)
        perf_native.validate_hardware_compatibility(hardware_protocol, hardware)
        manifest = perf_native._read_capture_manifest(root, protocol, hardware)
        capture = perf_native._load_capture(root, QUERY[0], QUERY[1], torch)
        receipt.update({"status": "running", "hardware": hardware,
                        "capture_manifest_hardware": manifest.get("hardware"),
                        "capture_query_present": True,
                        "runtime_device": str(device),
                        "arms": [], "runtime_load_completed_utc": datetime.now(timezone.utc).isoformat()})
        _write_summary(root, out, receipt)

        for arm in ARMS:
            try:
                row = _arm_diagnostic(common, runtime, protocol, old_root, root, out,
                                      arm, capture, torch, device)
                row["reference_install"] = _small_install_metadata(row["reference_install"])
                row["native_install"] = _small_install_metadata(row["native_install"])
                receipt["arms"].append({"status": "complete", **row})
            except Exception as exc:
                error_trace = traceback.format_exc()
                try:
                    if runtime is not None:
                        perf_native._restore_to_bf16(runtime, protocol, old_root,
                            getattr(runtime, "restore_native_rotation", None))
                except Exception:
                    pass
                receipt["arms"].append({"arm": arm, "status": "failed",
                                        "error": f"{type(exc).__name__}: {exc}",
                                        "traceback": error_trace})
            _write_summary(root, out, receipt)

        failed = [row["arm"] for row in receipt["arms"] if row.get("status") != "complete"]
        receipt.update({"status": "failed" if failed else "complete",
                        "complete": not failed,
                        "failed_arms": failed,
                        "finished_utc": datetime.now(timezone.utc).isoformat()})
        _write_summary(root, out, receipt)
        print(json.dumps({"status": receipt["status"], "summary": str(root / OUTPUT_RELATIVE_PATH),
                          "arms": [row.get("arm") for row in receipt["arms"]]},
                         ensure_ascii=False), flush=True)
        return 0 if not failed else 2
    except Exception as exc:
        receipt.update({"status": "failed", "complete": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                        "finished_utc": datetime.now(timezone.utc).isoformat()})
        _write_summary(root, out, receipt)
        return 2
    finally:
        if runtime is not None and common is not None:
            try:
                perf_native._restore_to_bf16(runtime, protocol, old_root,
                    getattr(runtime, "restore_native_rotation", None))
            except Exception:
                pass
        _ = code_root


if __name__ == "__main__":
    raise SystemExit(main())
