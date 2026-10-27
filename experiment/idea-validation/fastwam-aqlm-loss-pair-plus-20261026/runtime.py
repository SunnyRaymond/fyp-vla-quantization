"""AQLM final8 runtime for the frozen paired LIBERO-Plus evaluation.

The episode implementation is deliberately inherited from the previously
validated closed-loop adapter.  This file only replaces its old VQ deployment
loader with the pinned AQLM decoded-BF16 installation.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path


PROTOCOL = "fastwam-aqlm-loss-pair-plus-v1"
ARMS = ("reconstruction", "ours")
AQLM_COMMIT = "e79a896ed6656fe4ed06193d42d004e7d0bbdbb2"
ACTION_SHAPE = (32, 7)
FUTURE_SHAPE = (1, 48, 3, 14, 28)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def expected_install_metadata(protocol, arm):
    source = protocol["arm_sources"][arm]
    return {
        "protocol": protocol["protocol"],
        "arm": arm,
        "source_protocol": source["source_protocol"],
        "source_job": source["source_job"],
        "source_deployment": source["deployment"],
        "source_bank": source["source_bank"],
        "official_commit": AQLM_COMMIT,
        "weight_bits": 4,
        "activation_bits": 4,
        "execution": "decoded_bf16_dense_aqlm",
        "vq_enabled": True,
        "kv_cache_dtype": "bf16",
        "native_low_bit_kernel": False,
        "native_int4_gemm_calls": 0,
        "module_count": 614,
        "transform_implementation": "smooth05_h128_r0_aqlm_g8_m4_k256_v1",
        "smooth_alpha": 0.5,
        "hadamard_block": 128,
        "D": 1,
        "clipping_ratio": 1,
        "rotation_shared_streams": ["action", "video"],
        "rotation_shape": [128, 128],
        "rotation_updates": 8,
        "aqlm_codebooks": 4,
        "aqlm_codebook_size": 256,
        "aqlm_in_group_size": 8,
        "training_exposures": 80,
        "new_training_updates": 0,
        "all_parameters_frozen": True,
        "indices_frozen": True,
    }


def _require_bound(protocol, arm):
    sources = protocol.get("sources", {})
    parent_job = sources.get("parent_pair_job")
    parent_artifacts = sources.get("parent_pair_artifacts")
    if not isinstance(parent_job, str) or not parent_job:
        raise RuntimeError("AQLM closed-loop source pair is not bound to a completed PBS job")
    if not isinstance(parent_artifacts, str) or not parent_artifacts:
        raise RuntimeError("AQLM closed-loop source pair artifacts are not bound")
    source = protocol.get("arm_sources", {}).get(arm, {})
    if source.get("source_job") != parent_job:
        raise RuntimeError(f"{arm} source job is not bound to the completed pair")
    required = ("source_protocol_file", "source_summary", "deployment", "endpoint",
                "anchor_tensor", "source_bank")
    if any(not isinstance(source.get(key), str) or not source[key] for key in required):
        raise RuntimeError(f"{arm} source binding is incomplete")


def _load_torch(path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _validate_source_summary(protocol, arm, source, parent_protocol, summary, anchor):
    if (not isinstance(summary, dict)
            or summary.get("protocol") != parent_protocol.get("protocol")
            or summary.get("arm") != arm
            or summary.get("complete") is not True
            or summary.get("final_anchor") is not True
            or summary.get("updates") != 8
            or summary.get("exposures") != 80
            or summary.get("target_linears") != 614):
        raise RuntimeError(f"{arm} final8 source summary is incomplete or differently bound")
    if summary.get("pbs_jobid") not in (None, source["source_job"]):
        raise RuntimeError(f"{arm} final8 source summary PBS identity differs")
    if (not isinstance(anchor, dict)
            or anchor.get("protocol") != parent_protocol.get("protocol")
            or anchor.get("arm") != arm
            or anchor.get("updates") != 8
            or anchor.get("exposures") != 80
            or anchor.get("target_linears") != 614
            or anchor.get("case_id") != 2000
            or anchor.get("seed_index") != 0
            or anchor.get("counts") != {"video_steps": 10, "action_steps": 10}
            or not isinstance(anchor.get("pbs_jobid"), str)
            or not anchor["pbs_jobid"]):
        raise RuntimeError(f"{arm} final8 anchor metadata is incomplete")


def _validate_deployment(deployment, arm, source, parent_protocol):
    if (not isinstance(deployment, dict)
            or deployment.get("protocol") != parent_protocol
            or deployment.get("arm") != arm
            or deployment.get("updates") != 8
            or deployment.get("exposures") != 80
            or not isinstance(deployment.get("parameters"), dict)
            or len(deployment["parameters"]) != 614
            or deployment.get("banks") != source["source_bank"]):
        raise RuntimeError(f"{arm} final8 AQLM deployment binding differs")


def _max_abs(actual, expected):
    value = (actual.detach().float().cpu() - expected.detach().float().cpu()).abs()
    return float(value.max()) if value.numel() else 0.0


def _validate_anchor_header(anchor, saved):
    """Require the JSON and tensor-file headers to be the same final8 unit."""
    if not isinstance(anchor, dict) or not isinstance(saved, dict):
        raise RuntimeError("Final AQLM anchor metadata must be mappings")
    for key, value in anchor.items():
        if key in ("action", "future"):
            continue
        if key not in saved or saved[key] != value:
            raise RuntimeError(f"Final AQLM anchor JSON/PT header differs at {key}")
    for key, expected in (("action_shape", ACTION_SHAPE), ("future_shape", FUTURE_SHAPE)):
        if tuple(anchor.get(key, ())) != expected or tuple(saved.get(key, ())) != expected:
            raise RuntimeError(f"Final AQLM anchor geometry metadata differs at {key}")


def _runtime_type():
    import runner as pilot

    previous_path = Path(__file__).with_name("previous_vq_adapter.py")
    if not previous_path.is_file():
        raise FileNotFoundError(
            "previous_vq_adapter.py must be staged from the pinned 25858606 artifact"
        )
    spec = importlib.util.spec_from_file_location("aqlm_episode_base", previous_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the staged previous episode adapter")
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    base = previous._runtime_type()

    class Runtime(base):
        def __init__(self, out, protocol, worker_id):
            import torch
            from aqlm_banks import AQLMBanks

            pilot.guard()
            if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
                raise RuntimeError("Match the pinned deterministic CUBLAS workspace configuration")
            if type(worker_id) is not int or worker_id not in range(4):
                raise ValueError("Expected one of four fixed AQLM workers")
            arm = ARMS[worker_id // 2]
            _require_bound(protocol, arm)
            source = protocol["arm_sources"][arm]
            parent_protocol = read_json(protocol["sources"]["parent_protocol_file"])
            if parent_protocol.get("protocol") != source["source_protocol"]:
                raise RuntimeError("AQLM parent protocol identity differs from arm source")
            summary = read_json(source["source_summary"])
            anchor = read_json(source["endpoint"])
            _validate_source_summary(protocol, arm, source, parent_protocol, summary, anchor)
            deployment = _load_torch(source["deployment"])
            _validate_deployment(deployment, arm, source, parent_protocol)

            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
            import random
            import numpy as np
            seed = parent_protocol['training']['random_seed']
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # Do not call the old adapter constructor: it installs the old VQ
            # deployment.  Its episode/infer/close methods are inherited only.
            pilot.Runtime.__init__(self, out)
            self.model.eval()
            self.arm("bf16")
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            self.protocol = protocol
            self.arm_name = self.current_arm = arm
            self.out = Path(out)
            self.worker_id = worker_id
            self.run_queries = self.linear_calls = 0
            self.transform_implementation = protocol["quantization"]["transform_implementation"]

            self.banks = AQLMBanks(
                self, parent_protocol, source["source_bank"], load_quantized=True
            )
            self.banks.restore(deployment["parameters"])
            self.banks.mode = "aqlm"
            self.banks.bind_dense()
            for parameter in self.banks.parameters():
                parameter.requires_grad_(False)
            for parameter in self.banks.rotations.parameters():
                parameter.requires_grad_(False)
            if hasattr(self.banks, "_indices_are_frozen") and not self.banks._indices_are_frozen():
                raise RuntimeError("AQLM indices are not frozen")
            if any(parameter.requires_grad for parameter in self.model.parameters()):
                raise RuntimeError("Base model parameters must remain frozen")

            self.seen = self.banks.seen
            self.forwards = self.banks.forwards
            for module in self.q.modules.values():
                forward = module.forward

                def counted(value, *args, _forward=forward, **kwargs):
                    self.linear_calls += 1
                    return _forward(value, *args, **kwargs)

                module.forward = counted

            with torch.no_grad():
                orthogonality = self.banks.rotations.orthogonality()
                movement = {
                    stream: float(
                        (self.banks.rotations.matrix(stream)
                         - torch.eye(128, device="cuda")).norm()
                    )
                    for stream in ("action", "video")
                }
            if (not orthogonality or any(not _finite(v) for v in orthogonality.values())
                    or max(orthogonality.values()) > protocol["startup_checks"]["orthogonality_max_abs_tolerance"]
                    or any(not _finite(v) or v <= 0 for v in movement.values())):
                raise RuntimeError("Frozen AQLM R0 orthogonality/movement gate failed")

            self.install_metadata = {
                **expected_install_metadata(protocol, arm),
                "orthogonality": orthogonality,
                "rotation_movement": movement,
            }
            self.source_protocol = parent_protocol
            self.source_summary = summary
            self.anchor = anchor
            self.reference_row = None
            self.source_anchor_job = anchor["pbs_jobid"]
            del deployment
            torch.cuda.empty_cache()

        def prepare(self, results_root):
            import numpy as np
            import torch
            import calibration_source
            import sampler

            source = self.protocol["arm_sources"][self.current_arm]
            saved = _load_torch(source["anchor_tensor"])
            if not isinstance(saved, dict) or "action" not in saved or "future" not in saved:
                raise RuntimeError("Final AQLM anchor tensor is missing action/future")
            _validate_anchor_header(self.anchor, saved)
            action_device = getattr(saved["action"], "device", None)
            future_device = getattr(saved["future"], "device", None)
            if (not hasattr(saved["action"], "shape") or not hasattr(saved["future"], "shape")
                    or tuple(saved["action"].shape) != ACTION_SHAPE
                    or tuple(saved["future"].shape) != FUTURE_SHAPE
                    or action_device is None or future_device is None
                    or action_device.type != "cpu" or future_device.type != "cpu"):
                raise RuntimeError("Final AQLM anchor tensor geometry/device differs")
            if (not bool(torch.isfinite(saved["action"]).all())
                    or not bool(torch.isfinite(saved["future"]).all())):
                raise RuntimeError("Final AQLM anchor tensor contains nonfinite values")
            plan_path = Path(self.protocol["sources"]["calibration_plan"])
            plan = read_json(plan_path)
            rows = {row["case_id"]: row for row in plan["inputs"]}
            row = rows[2000]
            self.reference_row = row
            if self.anchor.get("plan_row") != row:
                raise RuntimeError("Final AQLM anchor uses a different calibration plan row")
            if self.anchor.get("sampler_seed") != row["sampler_seeds"][0]:
                raise RuntimeError("Final AQLM anchor sampler seed differs from the frozen plan")
            observation, metadata = calibration_source.load_observation(
                np, plan_path.parent, row
            )
            self.description = metadata["description"]
            datum, seed = self.datum(observation), row["sampler_seeds"][0]
            self.banks.seen.clear()
            self.banks.rotations.refresh()
            with torch.no_grad():
                action, counts, future = sampler.differentiable_action(
                    self, datum, seed, checkpointed=False, return_future=True
                )
            if (counts != {"video_steps": 10, "action_steps": 10}
                    or len(self.banks.seen) != 614
                    or tuple(action.shape) != ACTION_SHAPE
                    or tuple(future.shape) != FUTURE_SHAPE
                    or not bool(torch.isfinite(action).all())
                    or not bool(torch.isfinite(future).all())):
                raise RuntimeError("AQLM final8 anchor query coverage/shape gate failed")
            action_error = _max_abs(action, saved["action"])
            future_error = _max_abs(future, saved["future"])
            actual, _ = self.infer(datum, seed, measure=False)
            formal_error = _max_abs(actual, action)
            if (action_error != 0.0 or future_error != 0.0
                    or not _finite(formal_error)
                    or formal_error > self.protocol["startup_checks"]["actual_infer_action_max_abs_tolerance"]
                    or getattr(self.q, "_native_int4_gemm_calls", 0) != 0
                    or getattr(self.q, "_kv_active", False)):
                raise RuntimeError(
                    f"AQLM final8 anchor/formal infer parity failed: "
                    f"{action_error},{future_error},{formal_error}"
                )
            self.run_queries = 0
            return {
                "protocol": self.protocol["protocol"],
                "worker_id": self.worker_id,
                "arm": self.current_arm,
                "pbs_job_id": os.environ["PBS_JOBID"],
                "source_job": source["source_job"],
                "source_endpoint": source["endpoint"],
                "source_anchor_job": self.source_anchor_job,
                "passed": True,
                "observed_modules": 614,
                "case_id": 2000,
                "seed_index": 0,
                "sampler_seed": seed,
                "source_anchor_passed": True,
                "source_action_max_abs": action_error,
                "source_future_max_abs": future_error,
                "actual_infer_action_max_abs": formal_error,
                "startup_queries": 2,
                "new_training_updates": 0,
                "install_metadata": self.install_metadata,
            }

    return Runtime


class R128Runtime:
    def __new__(cls, *args, **kwargs):
        return _runtime_type()(*args, **kwargs)
