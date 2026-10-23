"""Small CPU-tensor schema and calibration checks; invoke only from guarded PBS jobs."""
from __future__ import annotations

import json
from pathlib import Path


def self_check(torch, protocol=None):
    from latent_reference import calibration_loss_scale
    from prepare_references import (PROTOCOL, _json_action_tensor, context_rows,
                                    validate_reference_payload)

    if protocol is None:
        protocol = json.loads((Path(__file__).with_name("protocol.json")).read_text(encoding="utf-8"))
    teacher_contexts = context_rows(protocol, include_test=True)
    parent_contexts = context_rows(protocol, include_test=False)
    rows = [{"case_id": case, "seed_index": 0, "split": "calibration",
             "action_mse": 2.0, "future_nmse": 4.0}
            for case in protocol["calibration_cases"]]
    normalization = calibration_loss_scale(rows, protocol["calibration_cases"], 1.0)
    if (normalization["normalization_c"] != 0.5
            or normalization["mean_cal_initial_action_mse"] != 2.0
            or normalization["mean_cal_initial_future_nmse"] != 4.0
            or normalization["calibration_ids"] != protocol["calibration_cases"]):
        raise AssertionError("Toy calibration normalization differs from the frozen schema")

    try:
        calibration_loss_scale(rows + [{"case_id": protocol["selection_cases"][0],
                                        "seed_index": 0, "split": "selection",
                                        "action_mse": 1e9, "future_nmse": 1e-9}],
                               protocol["calibration_cases"], 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Selection contexts must not affect frozen calibration normalization")

    action = torch.zeros((32, 7), dtype=torch.float32, device="cpu")
    future = torch.zeros((1, 48, 3, 14, 28), dtype=torch.float32, device="cpu")
    parsed_json_action = _json_action_tensor(
        torch, [[0.0] * 7 for _ in range(32)], "self-check cached JSON action")
    if tuple(parsed_json_action.shape) != (32, 7) or parsed_json_action.dtype != torch.float32:
        raise AssertionError("JSON action decoding must preserve the frozen float32 [32,7] contract")
    try:
        _json_action_tensor(torch, [[0.0] * 7], "self-check malformed JSON action")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Malformed JSON action shape must still fail the frozen tensor gate")

    def records(contexts):
        result = {}
        for context in contexts:
            key = f"{context['case_id']}:{context['seed_index']}"
            result[key] = {
                "case_id": context["case_id"], "seed_index": context["seed_index"],
                "sampler_seed": 1000000000 + context["case_id"] * 2 + context["seed_index"],
                "action": action, "future": future,
            }
            context["sampler_seed"] = result[key]["sampler_seed"]
        return result

    teacher, parent = records(teacher_contexts), records(parent_contexts)
    loss_normalization = {
        "policy": "normalization_c = mean_8cal_micro_initial_action_mse / mean_8cal_micro_initial_future_nmse",
        "normalization_c": normalization["normalization_c"],
        "mean_cal_initial_action_mse": normalization["mean_cal_initial_action_mse"],
        "mean_cal_initial_future_nmse": normalization["mean_cal_initial_future_nmse"],
        "calibration_ids": list(protocol["calibration_cases"]),
    }
    cache = {
        "schema_version": 2, "status": "complete", "protocol": PROTOCOL,
        "initial_arm": "micro_scale_vq_a4", "bank_root": "/bank-root", "bank_full_job": "bank-job",
        "source_root": "/source-root", "source_full_job": "source-job",
        "teacher_context_count": 60, "parent_context_count": 16,
        "calibration_context_count": 8, "selection_context_count": 8,
        "model_forward_count": 16, "teacher_forward_count": 0, "target_linears": 614,
        "loss_normalization": loss_normalization, "teacher": teacher, "parent": parent,
    }
    validate_reference_payload(torch, cache, teacher_contexts, parent_contexts)
    print("MICRO_REFERENCE_SELF_CHECK_PASSED", flush=True)
    return {"passed": True, "schema_version": 2, "teacher_context_count": 60,
            "parent_context_count": 16, "normalization_c": normalization["normalization_c"]}
