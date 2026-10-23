"""Install a completed control-training D/r checkpoint in SwitchingRuntime."""
import math


def _all_frozen(switching):
    params = list(switching.runtime.model.parameters())
    for name in ("books", "logs", "ratios", "shadows"):
        params.extend(getattr(switching.banks, name).values())
    params.extend(switching.original_weights.values())
    return all(not value.requires_grad for value in params)


def _range(spec, low_key, high_key, label):
    if not isinstance(spec, dict):
        raise ValueError(f"Invalid {label} bounds in the frozen runtime protocol")
    low, high = spec.get(low_key), spec.get(high_key)
    if (isinstance(low, bool) or isinstance(high, bool)
            or not isinstance(low, (int, float)) or not isinstance(high, (int, float))
            or not math.isfinite(float(low)) or not math.isfinite(float(high))
            or low > high):
        raise ValueError(f"Invalid {label} bounds in the frozen runtime protocol")
    return float(low), float(high)


def _validate_tensor_map(torch, saved, current, low, high, label):
    if not isinstance(saved, dict) or set(saved) != set(current):
        raise ValueError(f"{label} tensor names differ from the current bank")
    result = {}
    for name, value in saved.items():
        reference = current[name]
        if (not torch.is_tensor(value) or tuple(value.shape) != tuple(reference.shape)
                or value.dtype != reference.dtype or not bool(torch.isfinite(value).all())
                or bool((value < low).any()) or bool((value > high).any())):
            raise ValueError(f"{label} tensor is invalid for {name}")
        result[name] = value.detach().clone()
    return result


def install_training_deployment(switching, parent_protocol, training_config, arm,
                                deployment, summary, deployment_path):
    """Validate all bindings before installing ratios/scales into the runtime slot."""
    if any(not isinstance(value, dict) for value in
           (parent_protocol, training_config, deployment, summary)):
        raise ValueError("Parent, training, deployment, and summary must be mappings")
    if not isinstance(arm, str) or not arm:
        raise ValueError("Training arm must be a nonempty string")
    unit = f"train_{arm}"
    arms = training_config.get("arms")
    if (training_config.get("protocol") != "control-kl-training-v1"
            or not isinstance(arms, dict) or arm not in arms
            or training_config.get("parent_protocol") != parent_protocol.get("protocol")
            or deployment.get("protocol") != parent_protocol
            or deployment.get("training_protocol") != training_config
            or deployment.get("unit") != unit or deployment.get("arm") != arm
            or deployment.get("updates") != 80
            or deployment.get("checkpoint_arm") != "per_channel_clip"):
        raise ValueError("Training deployment protocol, arm, or update binding differs")

    order = training_config.get("update_order")
    data_config = training_config.get("data_config")
    source_closed = parent_protocol.get("source_closed_protocol")
    calibration = source_closed.get("calibration") if isinstance(source_closed, dict) else None
    original_order = calibration.get("update_order") if isinstance(calibration, dict) else None
    if (not isinstance(order, list) or len(order) != 80
            or any(not isinstance(value, int) or isinstance(value, bool) for value in order)
            or len(set(order)) != 80
            or order != original_order
            or deployment.get("update_order") != order
            or deployment.get("data_config") != data_config):
        raise ValueError("Training deployment order or data configuration differs")

    source_jobs = summary.get("source_jobs")
    config_sources = training_config.get("source_jobs")
    required_config_sources = {
        "training_teacher", "training_loss_selfcheck", "aggregate_c80_parameter_gate",
    }
    if (not isinstance(source_jobs, dict)
            or set(source_jobs) != required_config_sources | {
                "training_pairs", "training_teacher_plan", "calibration_0", "calibration_1",
                "training_runtime_selfcheck"}
            or any(not isinstance(value, str) or not value for value in source_jobs.values())
            or deployment.get("source_jobs") != source_jobs
            or not isinstance(config_sources, dict)
            or set(config_sources) != required_config_sources
            or any(not isinstance(value, str) or not value for value in config_sources.values())
            or any(source_jobs.get(key) != value
                   for key, value in config_sources.items())):
        raise ValueError("Training deployment source-job bindings differ")
    output_path = str(deployment_path)
    if deployment_path is None or not output_path:
        raise ValueError("A deployment output path is required")
    reference_path = deployment.get("reference_endpoint")
    this_job = summary.get("this_job")
    optimizer = training_config.get("optimizer")
    if (not isinstance(reference_path, str) or not reference_path
            or deployment.get("data_config") != summary.get("data_config")
            or training_config != summary.get("training_protocol")
            or summary.get("protocol") != parent_protocol.get("protocol")
            or summary.get("unit") != unit or summary.get("arm") != arm
            or summary.get("complete") is not True or summary.get("status") != "complete"
            or summary.get("completed_updates") != 80 or summary.get("training_updates") != 80
            or summary.get("deployment") != output_path
            or summary.get("reference_endpoint") != reference_path
            or not isinstance(summary.get("pbs_jobid"), str)
            or not summary.get("pbs_jobid")
            or not isinstance(this_job, dict)
            or this_job.get("pbs_jobid") != summary.get("pbs_jobid")
            or not isinstance(optimizer, dict)
            or summary.get("gate_fraction") != optimizer.get("gate_fraction")):
        raise ValueError("Completed training summary does not bind this deployment and final PBS job")
    parity = summary.get("native_cached_parity")
    parity_keys = {"action_max_abs", "future_max_abs", "initial_noise_max_abs"}
    if (not isinstance(parity, dict) or set(parity) != parity_keys
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(float(value)) or value != 0
                   for value in parity.values())):
        raise ValueError("Final cached endpoint parity must be exactly zero for all three outputs")

    banks = switching.banks
    if not _all_frozen(switching):
        raise ValueError("Model, reader, or VQ parameters are not frozen")
    closed = switching.p["source_closed_protocol"]
    ratio_min, ratio_max = _range(
        closed["clipping"], "minimum_ratio", "maximum_ratio", "ratio")
    scale_min, scale_max = _range(
        closed["scaling"], "minimum_log2", "maximum_log2", "log2 scale")
    ratios = _validate_tensor_map(
        switching.torch, deployment.get("ratios"), banks.ratios,
        ratio_min, ratio_max, "ratio")
    scales = _validate_tensor_map(
        switching.torch, deployment.get("log2_scales"), banks.shadows,
        scale_min, scale_max, "log2 scale")
    if "per_channel_clip" not in switching.deployments:
        raise ValueError("SwitchingRuntime has no per_channel_clip deployment slot")

    switching.deployments["per_channel_clip"] = {
        "ratios": ratios, "log2_scales": scales,
    }
    binding = {
        "protocol": parent_protocol["protocol"], "unit": unit, "arm": arm,
        "training_protocol": training_config, "data_config": data_config,
        "source_jobs": source_jobs, "producer_job": summary["pbs_jobid"],
        "deployment": output_path, "reference_endpoint": reference_path,
        "checkpoint_arm": "per_channel_clip", "updates": 80,
    }
    switching.deployment_bindings["per_channel_clip"] = binding
    switching.activate("per_channel_clip")
    if switching.active_arm != "per_channel_clip" or not _all_frozen(switching):
        raise RuntimeError("Training deployment activation changed the frozen runtime state")
    return binding
