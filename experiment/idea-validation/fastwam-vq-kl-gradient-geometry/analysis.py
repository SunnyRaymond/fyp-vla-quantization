"""CPU-only geometry of cached full-C80 gradients at the original MSE80 point."""
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import socket


PROTOCOL = "fastwam-vq-kl-gradient-geometry-v1"
CASES = list(range(2000, 2080))
PARAM_COUNTS = {"ratio": 614, "channel": 1806863}
ZERO_COUNTERS = ("new_model_forwards", "new_teacher_forwards", "new_backwards", "training_updates")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def close(a, b, rtol=1e-6, atol=1e-9):
    return math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol)


def allocation_guard():
    job = os.environ.get("PBS_JOBID", "").strip()
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    host = socket.getfqdn().lower().rstrip(".")
    require(job and nodefile and Path(nodefile).is_file(),
            "Refusing numerical analysis outside a real PBS allocation")
    require("login" not in host, f"Refusing to run on login host {host}")
    members = {line.strip().lower().rstrip(".") for line in Path(nodefile).read_text().splitlines() if line.strip()}
    member_short = {name.split(".", 1)[0] for name in members}
    require(host in members or host.split(".", 1)[0] in member_short,
            f"Current host {host} is not listed in PBS_NODEFILE")
    return job, host


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def load_helper():
    path = Path(__file__).with_name("direction_utils.py")
    spec = importlib.util.spec_from_file_location("direction_utils", path)
    require(spec is not None and spec.loader is not None, f"Missing helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def as_tensor(torch, value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    return torch.as_tensor(value, dtype=torch.float32).clone()


def finite_tensor(torch, value, label):
    require(torch.is_tensor(value) and bool(torch.isfinite(value).all().item()),
            f"Non-tensor or non-finite value in {label}")


def flatten64(torch, tensors):
    return torch.cat([value.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
                      for value in tensors])


def alignment_stats(torch, a, b):
    require(a.numel() == b.numel(), "Alignment vectors differ in size")
    opposed = a * b < 0
    na2, nb2 = a.square().sum(), b.square().sum()
    na, nb = float(torch.sqrt(na2).item()), float(torch.sqrt(nb2).item())
    dot = float((a * b).sum().item())
    cosine = dot / (na * nb) if na and nb else None
    count = int(opposed.sum().item())
    return {
        "coordinates": int(a.numel()), "norm_mse": na, "norm_kl": nb,
        "cosine": max(-1.0, min(1.0, cosine)) if cosine is not None else None,
        "opposed_coordinates": count,
        "opposed_coordinate_fraction": count / int(a.numel()) if a.numel() else None,
        "mse_squared_gradient_mass_on_opposed_fraction":
            float(a[opposed].square().sum().item() / na2.item()) if na2.item() else None,
        "kl_squared_gradient_mass_on_opposed_fraction":
            float(b[opposed].square().sum().item() / nb2.item()) if nb2.item() else None,
    }


def param_group_values(torch, checkpoint):
    ratios, channels = checkpoint["ratios"], checkpoint["shadow_log2_scales"]
    ratio_names, channel_names = list(ratios), list(channels)
    require(ratio_names == channel_names and len(ratio_names) == PARAM_COUNTS["ratio"],
            "Ratio and shadow parameter key order differs")
    ratio_values = [as_tensor(torch, ratios[name]) for name in ratio_names]
    channel_values = [as_tensor(torch, channels[name]) for name in channel_names]
    opt = checkpoint["optimizer"]
    saved_groups = opt.get("param_groups", [])
    require(len(saved_groups) == 2 and
            [len(group.get("params", [])) for group in saved_groups] == [614, 614],
            "Saved Adam must contain ratio then channel parameter groups")
    actual = [ratio_values, channel_values]
    expected = [614, 1806863]
    for name, values, group, count in zip(("ratio", "channel"), actual, saved_groups, expected):
        require(sum(value.numel() for value in values) == count,
                f"{name} parameter coordinates differ from the frozen shape contract")
        require(len(values) == len(group["params"]), f"{name} Adam group order/length differs")
        for index, value in enumerate(values):
            finite_tensor(torch, value, f"checkpoint/{name}/{index}")
    live_groups = []
    for name, values, saved in zip(("ratio", "channel"), actual, saved_groups):
        group = {key: copy.deepcopy(value) for key, value in saved.items() if key != "params"}
        group["params"] = values
        group["name"] = name
        live_groups.append(group)
    return ratio_names, channel_names, actual, live_groups, opt


def check_gradient(torch, gradients, parameters, name):
    require(len(gradients) == len(parameters), f"{name} gradient tensor count differs")
    checked = []
    for index, (gradient, parameter) in enumerate(zip(gradients, parameters)):
        require(torch.is_tensor(gradient) and tuple(gradient.shape) == tuple(parameter.shape),
                f"{name} gradient shape differs at parameter {index}")
        value = gradient.detach().cpu()
        finite_tensor(torch, value, f"{name}/gradient/{index}")
        checked.append(value)
    return checked


def metric_values(row, label):
    keys = ("action_mse", "latent_nmse", "action_kl", "latent_kl")
    values = {}
    for key in keys:
        require(key in row, f"{label} lacks baseline metric {key}")
        value = float(row[key])
        require(math.isfinite(value), f"{label} has non-finite {key}")
        values[key] = value
    return values


def validate_inputs(torch, p):
    require(p.get("protocol") == PROTOCOL and p.get("calibration_case_ids") == CASES
            and p.get("seed_index") == 0 and p.get("heldout_used") is False,
            "Geometry protocol or frozen C80 cohort differs")
    require(p.get("source_mse_job") == "25862598.pbs101"
            and p.get("mse_gradient_job") == "25881910.pbs101"
            and p.get("kl_gradient_job") == "25881167.pbs101"
            and p.get("kl_gradient_unit") == "mse_dual",
            "Frozen checkpoint or gradient artifact job binding differs")
    require(p.get("directions") == ["saved_adam", "fresh_adam"]
            and p.get("fractions") == [1.0, 0.5, 0.25]
            and p.get("new_model_forwards") == 0 and p.get("new_teacher_forwards") == 0
            and p.get("new_backwards") == 0 and p.get("training_updates") == 0,
            "Protocol must remain a cache-only, zero-update analysis")

    source_mse_root = Path(p["source_mse_root"])
    source_root = Path(p["source_root"])
    mse_root = Path(p["mse_gradient_root"])
    kl_root = Path(p["kl_gradient_root"])
    mse_protocol = read_json(source_mse_root / "protocol.json")
    source_protocol = read_json(source_root / "protocol.json")
    mse_grad_protocol = read_json(mse_root / "protocol.json")
    kl_grad_protocol = read_json(kl_root / "protocol.json")
    require(mse_protocol.get("remote_root") == str(source_mse_root)
            and source_protocol.get("remote_root") == str(source_root)
            and source_protocol.get("source_per_channel_root") == str(source_mse_root)
            and source_protocol.get("source_per_channel_job") == p["source_mse_job"],
            "Original MSE80 and dual-source protocol bindings differ")
    order = mse_protocol.get("calibration", {}).get("update_order", [])
    require(len(order) == 80 and set(order) == set(CASES) and len(set(order)) == 80,
            "Original MSE training update order is not the frozen C80 set")
    require(source_protocol.get("calibration", {}).get("case_ids") == CASES
            and source_protocol.get("calibration", {}).get("seed_index") == 0,
            "Original dual source does not bind the same C80 seed-0 cohort")

    require(mse_grad_protocol.get("protocol") == "fastwam-vq-kl-fullbatch-refine-v1"
            and mse_grad_protocol.get("remote_root") == str(mse_root)
            and mse_grad_protocol.get("source_root") == str(source_root)
            and mse_grad_protocol.get("source_mse_root") == str(source_mse_root)
            and mse_grad_protocol.get("source_mse_job") == p["source_mse_job"]
            and mse_grad_protocol.get("calibration_case_ids") == CASES,
            "Cached MSE-gradient protocol source/cohort binding differs")
    require(kl_grad_protocol.get("protocol") == "fastwam-vq-kl-direction-v1"
            and kl_grad_protocol.get("remote_root") == str(kl_root)
            and kl_grad_protocol.get("source_root") == str(source_root)
            and kl_grad_protocol.get("case_ids") == CASES
            and kl_grad_protocol.get("units", {}).get("mse_dual") ==
            {"checkpoint_arm": "mse_nmse", "loss": "dual_kl"},
            "Cached KL-gradient protocol source/loss binding differs")
    mse_summary = read_json(mse_root / "results" / "mse_nmse" / "summary.json")
    kl_summary = read_json(kl_root / "results" / "mse_dual" / "summary.json")
    require(mse_summary.get("complete") is True and mse_summary.get("protocol") == mse_grad_protocol["protocol"]
            and mse_summary.get("unit") == "mse_nmse" and mse_summary.get("pbs_jobid") == p["mse_gradient_job"],
            "Cached MSE gradient summary/job binding differs")
    require(kl_summary.get("complete") is True and kl_summary.get("protocol") == kl_grad_protocol["protocol"]
            and kl_summary.get("unit") == "mse_dual" and kl_summary.get("pbs_jobid") == p["kl_gradient_job"]
            and kl_summary.get("heldout_used") is False and kl_summary.get("gradient_query_count") == 80,
            "Cached KL gradient summary/job binding differs")

    receipt = read_json(source_mse_root / "results" / "train_eval" / "training_receipt.json")
    require(receipt.get("complete") is True and receipt.get("updates") == 80
            and receipt.get("pbs_jobid") == p["source_mse_job"],
            "Original MSE80 training receipt is incomplete or mismatched")
    checkpoint = torch.load(source_mse_root / "results" / "train_eval" / "checkpoint.pt",
                            map_location="cpu", weights_only=True)
    require(checkpoint.get("protocol") == mse_protocol and checkpoint.get("updates") == 80
            and [row.get("case_id") for row in checkpoint.get("history", [])] == order,
            "MSE80 checkpoint protocol/update history differs")

    mse_state = torch.load(mse_root / "results" / "mse_nmse" / "gradient_state.pt",
                           map_location="cpu", weights_only=True)
    kl_state = torch.load(kl_root / "results" / "mse_dual" / "gradient_state.pt",
                          map_location="cpu", weights_only=True)
    require(mse_state.get("protocol") == mse_grad_protocol and mse_state.get("unit") == "mse_nmse"
            and mse_state.get("case_ids") == CASES and mse_state.get("gradient_queries_materialized") == 80
            and mse_state.get("heldout_used") is False,
            "Cached MSE gradient does not contain complete heldout-free C80 state")
    require(kl_state.get("protocol") == kl_grad_protocol and kl_state.get("unit") == "mse_dual",
            "Cached KL gradient protocol/unit differs")
    mse_rows = mse_state.get("baseline_rows", [])
    mse_cases = mse_state.get("gradient_cases", [])
    kl_rows = kl_state.get("baseline", [])
    kl_parity = kl_state.get("parity", [])
    require(len(mse_rows) == len(mse_cases) == len(kl_rows) == len(kl_parity) == 80
            and [row.get("case_id") for row in mse_rows] == CASES
            and [row.get("case_id") for row in mse_cases] == CASES
            and [row.get("case_id") for row in kl_rows] == CASES,
            "Cached gradient baseline/parity coverage or order differs from C80")
    mse_parity = mse_state.get("source_parity", {})
    require(mse_parity.get("no_grad_checkpointed_action_max_abs") == 0.0
            and mse_parity.get("no_grad_checkpointed_future_max_abs") == 0.0
            and all(row.get("action_max_abs") == 0.0 and row.get("future_max_abs") == 0.0
                    for row in kl_parity),
            "Cached source action/future parity must be exactly zero")

    baseline_abs = {key: 0.0 for key in ("action_mse", "latent_nmse", "action_kl", "latent_kl")}
    for index, (mrow, krow) in enumerate(zip(mse_rows, kl_rows)):
        require(mrow["case_id"] == krow["case_id"] == CASES[index], "Baseline case alignment differs")
        mv, kv = metric_values(mrow, f"mse baseline {index}"), metric_values(krow, f"kl baseline {index}")
        for key in baseline_abs:
            diff = abs(mv[key] - kv[key])
            baseline_abs[key] = max(baseline_abs[key], diff)
            require(close(mv[key], kv[key]), f"MSE/KL same-point baseline differs for {CASES[index]}/{key}")

    ratio_names, channel_names, parameters, live_groups, optimizer = param_group_values(torch, checkpoint)
    flat_parameters = parameters[0] + parameters[1]
    mse_gradients = check_gradient(torch, mse_state.get("full_gradient", []), flat_parameters, "mse_nmse")
    kl_gradients = check_gradient(torch, kl_state.get("full_gradient", []), flat_parameters, "dual_kl")
    require(len(mse_gradients) == len(kl_gradients) == 1228,
            "Expected 614 ratio followed by 614 channel-scale gradient tensors")
    return {
        "source_mse_protocol": mse_protocol, "source_protocol": source_protocol,
        "mse_gradient_protocol": mse_grad_protocol, "kl_gradient_protocol": kl_grad_protocol,
        "checkpoint": checkpoint, "optimizer": optimizer, "parameters": parameters,
        "live_groups": live_groups, "ratio_names": ratio_names, "channel_names": channel_names,
        "mse_gradients": mse_gradients, "kl_gradients": kl_gradients,
        "baseline_abs": baseline_abs,
    }


def group_vectors(torch, values):
    ratio, channel = values[:614], values[614:]
    return {"ratio": flatten64(torch, ratio), "channel": flatten64(torch, channel),
            "overall": flatten64(torch, values)}


def project_step(torch, parameters, delta, fraction, bounds):
    realized, candidates, counts = [], [], {}
    offset = 0
    for name in ("ratio", "channel"):
        lower, upper = bounds[name]
        parts = []
        group_parameters = parameters[1 if name == "channel" else 0]
        for parameter, step in zip(group_parameters, delta[offset:offset + len(group_parameters)]):
            proposed = parameter + float(fraction) * step
            candidate = proposed.clamp(lower, upper)
            counts[name] = counts.get(name, 0) + int(((proposed < lower) | (proposed > upper)).sum().item())
            candidates.append(candidate)
            parts.append(candidate - parameter)
        realized.extend(parts)
        offset += len(parts)
    require(offset == len(delta), "Projected step tensor partition differs")
    counts["overall"] = counts["ratio"] + counts["channel"]
    return realized, counts, candidates


def step_summary(torch, step, mse_grad, kl_grad, projection_counts):
    step_groups, mse_groups, kl_groups = group_vectors(torch, step), group_vectors(torch, mse_grad), group_vectors(torch, kl_grad)
    result = {}
    for name in ("ratio", "channel", "overall"):
        delta = step_groups[name]
        norm = float(torch.linalg.vector_norm(delta).item())
        dot_mse = float((mse_groups[name] * delta).sum().item())
        dot_kl = float((kl_groups[name] * delta).sum().item())
        result[name] = {
            "coordinates": int(delta.numel()), "delta_norm": norm,
            "per_coordinate_rms": norm / math.sqrt(delta.numel()) if delta.numel() else None,
            "projected_coordinates": projection_counts[name],
            "g_dot_delta_mse": dot_mse, "first_order_predicted_loss_delta_mse": dot_mse,
            "g_dot_delta_kl": dot_kl, "first_order_predicted_loss_delta_kl": dot_kl,
        }
    return result, step_groups


def run_simulations(torch, helper, data, fractions):
    parameters = data["parameters"]
    flat_parameters = parameters[0] + parameters[1]
    opt = data["optimizer"]
    state0 = {"param_groups": [opt["param_groups"][0]],
              "state": {pid: opt.get("state", {})[pid] for pid in opt["param_groups"][0]["params"]
                        if pid in opt.get("state", {})}}
    bounds = {"ratio": (0.5, 1.0), "channel": (-1.0, 1.0)}
    gradients = {"mse_nmse": data["mse_gradients"], "dual_kl": data["kl_gradients"]}
    vectors = {name: group_vectors(torch, grad) for name, grad in gradients.items()}
    simulations, realized_steps, simulated_points = [], {}, {}
    ratio_group_assertions = 0
    for family in ("saved_adam", "fresh_adam"):
        saved = opt if family == "saved_adam" else None
        for loss_name, grad in gradients.items():
            delta = helper.adam_delta(torch, grad, data["live_groups"], saved)
            require(len(delta) == len(flat_parameters), f"{family}/{loss_name} Adam output length differs")
            for index, value in enumerate(delta):
                finite_tensor(torch, value, f"{family}/{loss_name}/adam_delta/{index}")
            ratio_delta = helper.adam_delta(torch, grad[:614], [data["live_groups"][0]],
                                            state0 if family == "saved_adam" else None)
            require(all(torch.equal(a, b) for a, b in zip(delta[:614], ratio_delta)),
                    f"{family}/{loss_name} ratio-only Adam step differs from full group-0 step")
            ratio_group_assertions += 1
            for fraction in fractions:
                realized, projected, candidate = project_step(torch, parameters, delta, fraction, bounds)
                details, vectors_by_group = step_summary(torch, realized, gradients["mse_nmse"],
                                                         gradients["dual_kl"], projected)
                simulations.append({"optimizer": family, "gradient_objective": loss_name,
                                    "fraction": fraction, "groups": details})
                realized_steps[(family, fraction, loss_name)] = vectors_by_group
                simulated_points[(family, fraction, loss_name)] = candidate
    pair_cosines = []
    for family in ("saved_adam", "fresh_adam"):
        for fraction in fractions:
            by_group = {}
            for group in ("ratio", "channel", "overall"):
                stat = alignment_stats(torch, realized_steps[(family, fraction, "mse_nmse")][group],
                                       realized_steps[(family, fraction, "dual_kl")][group])
                by_group[group] = {"cosine": stat["cosine"], "norm_mse_step": stat["norm_mse"],
                                   "norm_kl_step": stat["norm_kl"], "coordinates": stat["coordinates"]}
            pair_cosines.append({"optimizer": family, "fraction": fraction, "groups": by_group})
    return simulations, pair_cosines, ratio_group_assertions, simulated_points


def compare_candidate_checkpoints(torch, p, data, simulated_points):
    checks = (
        ("mse_fresh_adam_f0.25", "fresh_adam", 0.25, "mse_nmse",
         Path(p["mse_gradient_root"]) / "results" / "mse_nmse" / "checkpoint.pt"),
        ("dual_kl_saved_adam_f1", "saved_adam", 1.0, "dual_kl",
         Path("/scratch/users/ntu/yguo017/fastwam-vq-kl-mse-warmstart-20261020") /
         "results" / "dual_kl" / "checkpoint.pt"),
    )
    result = []
    for label, optimizer, fraction, loss_name, path in checks:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        require(checkpoint.get("updates") == 81,
                f"Candidate checkpoint update count differs: {label}")
        maximum = {"ratio": 0.0, "channel": 0.0}
        for group, field in (("ratio", "ratios"), ("channel", "shadow_log2_scales")):
            names = data["ratio_names"] if group == "ratio" else data["channel_names"]
            expected = simulated_points[(optimizer, fraction, loss_name)]
            observed = checkpoint.get(field, {})
            require(list(observed) == names, f"Candidate checkpoint {label} {field} key order differs")
            start, stop = (0, 614) if group == "ratio" else (614, 1228)
            differences = []
            for index, name in enumerate(names):
                actual = as_tensor(torch, observed[name])
                target = expected[start + index]
                require(tuple(actual.shape) == tuple(target.shape),
                        f"Candidate checkpoint {label} {field}/{name} shape differs")
                finite_tensor(torch, actual, f"candidate/{label}/{field}/{name}")
                differences.append((actual.to(torch.float64) - target.to(torch.float64)).abs().max())
            maximum[group] = float(torch.stack(differences).max().item())
        maximum["overall"] = max(maximum.values())
        result.append({"candidate": label, "path": str(path), "updates": checkpoint["updates"],
                       "max_abs_by_group": maximum, "tolerance": 1e-6,
                       "passed": maximum["overall"] <= 1e-6})
        require(result[-1]["passed"], f"Simulated candidate differs from saved checkpoint: {label}")
    return result


def gradient_reports(torch, data):
    mse, kl = data["mse_gradients"], data["kl_gradients"]
    group = {"ratio": (mse[:614], kl[:614]), "channel": (mse[614:], kl[614:]),
             "overall": (mse, kl)}
    report = {name: alignment_stats(torch, flatten64(torch, a), flatten64(torch, b))
              for name, (a, b) in group.items()}
    modules = []
    for index, name in enumerate(data["channel_names"]):
        a, b = flatten64(torch, [mse[614 + index]]), flatten64(torch, [kl[614 + index]])
        stats = alignment_stats(torch, a, b)
        modules.append({"module": str(name), "coordinates": stats["coordinates"],
                        "norm_mse": stats["norm_mse"], "norm_kl": stats["norm_kl"],
                        "cosine": stats["cosine"],
                        "opposed_coordinate_fraction": stats["opposed_coordinate_fraction"],
                        "mse_squared_gradient_mass_on_opposed_fraction":
                            stats["mse_squared_gradient_mass_on_opposed_fraction"],
                        "kl_squared_gradient_mass_on_opposed_fraction":
                            stats["kl_squared_gradient_mass_on_opposed_fraction"]})
    negative = sorted((row for row in modules if row["cosine"] is not None and row["cosine"] < 0),
                      key=lambda row: row["cosine"])[:10]
    largest = sorted(modules, key=lambda row: max(row["norm_mse"], row["norm_kl"]), reverse=True)[:10]
    return report, {"most_negative_channel_cosine_top10": negative,
                    "largest_channel_gradient_norm_top10": largest}


def make_report(summary):
    lines = [
        "# KL gradient geometry at the original MSE80 checkpoint", "",
        "本结果只读取已缓存的 C80 梯度、共同基线、原 MSE80 checkpoint/Adam 状态，并在 CPU 上做张量几何与 Adam 公式模拟。未读取 held-out，新增 model/teacher forward、backward、training update 均为 0。", "",
        "两个梯度都对应同一原 MSE final80 参数点。固定混合损失权重沿用共同 D=1、r=1 的初始 C80 标定；它们不是用 final80 指标重新估计的。saved_adam 使用 checkpoint 的 Adam moments，fresh_adam 使用同一超参数与零 moments。fraction 每次都从原参数点计算，再按 ratio `[0.5, 1]`、channel log2 scale `[-1, 1]` 投影。", "",
        "`g·Δ` 是缓存梯度对投影后参数步长的一阶预测，不是重新前向验证的实际损失变化；方向相反也不证明 finite-step loss 上升或下降。这里没有因果、泛化或 closed-loop 结论。", "",
        "## 缓存梯度对齐", "",
        "| 参数组 | MSE/NMSE 梯度范数 | dual-KL 梯度范数 | cosine | 反向坐标比例 | MSE 反向坐标平方梯度质量 | KL 反向坐标平方梯度质量 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("ratio", "channel", "overall"):
        item = summary["gradient_alignment"][name]
        lines.append(f"| {name} | {item['norm_mse']:.8g} | {item['norm_kl']:.8g} | {item['cosine']} | "
                     f"{item['opposed_coordinate_fraction']} | {item['mse_squared_gradient_mass_on_opposed_fraction']} | "
                     f"{item['kl_squared_gradient_mass_on_opposed_fraction']} |")
    lines += ["", "## Adam 投影步与一阶预测", "",
              "| Adam | gradient | fraction | 组 | ‖Δ‖₂ | 每坐标 RMS | MSE g·Δ | dual-KL g·Δ | 投影坐标 |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for condition in summary["step_simulations"]:
        for group in ("ratio", "channel", "overall"):
            item = condition["groups"][group]
            lines.append(f"| {condition['optimizer']} | {condition['gradient_objective']} | {condition['fraction']} | {group} | "
                         f"{item['delta_norm']:.8g} | {item['per_coordinate_rms']:.8g} | "
                         f"{item['g_dot_delta_mse']:.8g} | {item['g_dot_delta_kl']:.8g} | {item['projected_coordinates']} |")
    lines += ["", "## 两类 loss 更新之间的方向 cosine", "",
              "| Adam | fraction | ratio | channel | overall |", "|---|---:|---:|---:|---:|"]
    for item in summary["projected_step_alignment"]:
        lines.append(f"| {item['optimizer']} | {item['fraction']} | {item['groups']['ratio']['cosine']} | "
                     f"{item['groups']['channel']['cosine']} | {item['groups']['overall']['cosine']} |")
    lines += ["", "按 module 的 top-10 仅是描述性坐标汇总，不构成模块因果归因。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    pbs_jobid, host = allocation_guard()
    p = read_json(args.protocol)
    torch = __import__("torch")
    torch.set_num_threads(int(os.environ.get("FW_CPU_THREADS", "4")))
    helper = load_helper()
    data = validate_inputs(torch, p)
    alignment, modules = gradient_reports(torch, data)
    simulations, step_alignment, ratio_assertions, simulated_points = run_simulations(
        torch, helper, data, p["fractions"])
    candidate_checks = compare_candidate_checkpoints(torch, p, data, simulated_points)
    require(len(simulations) == 12 and len(step_alignment) == 6,
            "Expected twelve objective-specific steps and six paired step-cosine rows")
    summary = {
        "protocol": PROTOCOL, "complete": True, "pbs_jobid": pbs_jobid, "analysis_host": host,
        "source_root": p["source_root"], "source_mse_root": p["source_mse_root"],
        "source_mse_protocol": data["source_mse_protocol"]["protocol"],
        "source_mse_job": p["source_mse_job"], "source_updates": 80,
        "checkpoint_updates": data["checkpoint"]["updates"],
        "mse_gradient_root": p["mse_gradient_root"],
        "mse_gradient_protocol": data["mse_gradient_protocol"]["protocol"],
        "mse_gradient_job": p["mse_gradient_job"],
        "kl_gradient_root": p["kl_gradient_root"],
        "kl_gradient_protocol": data["kl_gradient_protocol"]["protocol"],
        "kl_gradient_unit": p["kl_gradient_unit"], "kl_gradient_job": p["kl_gradient_job"],
        "heldout_used": False,
        "new_model_forwards": 0, "new_teacher_forwards": 0, "new_backwards": 0, "training_updates": 0,
        "source_parity_passed": True, "baseline_metrics_match": True,
        "baseline_max_abs_differences": data["baseline_abs"],
        "gradient_counts": {"mse": 80, "kl": 80},
        "parameter_counts": {"ratio": 614, "channel": 1806863},
        "gradient_tensor_counts": {"ratio": 614, "channel": 614, "overall": 1228},
        "gradient_alignment": alignment, "module_alignment_extremes": modules,
        "step_simulations": simulations, "projected_step_alignment": step_alignment,
        "candidate_checkpoint_checks": candidate_checks, "candidate_checkpoints_passed": True,
        "ratio_only_adam_group_assertions_passed": ratio_assertions == 4,
        "ratio_only_adam_group_assertion_count": ratio_assertions,
        "analysis_scope": "cached full-C80 mixed-objective gradients at the original MSE final80 parameter point; projected Adam geometry only",
        "interpretation_limit": "g dot delta is first-order prediction; no actual loss reevaluation, heldout use, generalization, causal, or closed-loop claim",
    }
    output = Path(p["remote_root"]) / "results"
    output.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(summary, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    require(len(encoded) < 200 * 1024, "summary.json exceeds the 200 KiB artifact bound")
    temporary = output / "summary.json.tmp"
    temporary.write_bytes(encoded)
    temporary.replace(output / "summary.json")
    (output / "ANALYSIS.zh.md").write_text(make_report(summary), encoding="utf-8")
    print("KL_GRADIENT_GEOMETRY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
