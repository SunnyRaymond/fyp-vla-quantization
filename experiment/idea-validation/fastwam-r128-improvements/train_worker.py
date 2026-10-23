"""Train the frozen-protocol R128 optimization and context-coverage lanes."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import signal
import time


PROTOCOL = "fastwam-r128-improvements-v1"
C8 = [2000, 2019, 2021, 2025, 2051, 2057, 2066, 2070]
C80 = list(range(2000, 2080))
HELDOUT = list(range(3000, 3040))
LANES = ("optimization", "coverage_c80")
METRICS = (
    "first10motor_rmse", "first10motor_mse", "future_latent_nmse", "mixed_loss",
    "full32motor_rmse", "late22motor_rmse", "first10gripper_rmse",
)
LAMBDA = 0.04245452683169643


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def compare_metrics(actual, expected, tolerance=2e-6):
    """Validate a full metric replay against a pinned source query record."""
    deltas = {}
    for key in METRICS:
        if key not in actual or key not in expected:
            raise ValueError("Metric replay is missing " + key)
        left, right = float(actual[key]), float(expected[key])
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("Metric replay is nonfinite: " + key)
        deltas[key] = abs(left - right)
    maximum = max(deltas.values())
    if maximum > tolerance:
        raise ValueError("Metric replay differs from its pinned source by " + str(maximum))
    return {"max_abs_delta": maximum, "deltas": deltas, "tolerance": tolerance}


def validate_protocol(p, lane=None):
    if (p.get("protocol") != PROTOCOL or p.get("target_linears") != 614
            or p.get("calibration", {}).get("case_ids") != C8
            or p.get("calibration", {}).get("c80_case_ids") != C80
            or p.get("evaluation", {}).get("case_ids") != HELDOUT
            or p.get("evaluation", {}).get("seed_indices") != [0, 1]
            or p.get("suites") != ["libero_spatial", "libero_object", "libero_goal", "libero_10"]):
        raise ValueError("Frozen R128 cohort or target binding differs")
    training = p.get("training", {})
    if (not math.isclose(training.get("latent_weight", -1), LAMBDA, rel_tol=1e-14)
            or training.get("rotation_updates") != 8
            or training.get("rotation_learning_rate") != 0.0001
            or training.get("vq_updates") != 0
            or training.get("optimizer") != "Adam"
            or training.get("gradient_norm_clip") != 1.0
            or training.get("optimizer_reset_after_update1") is not False):
        raise ValueError("Frozen R128 training configuration differs")
    quantization = p.get("quantization", {})
    if (quantization.get("D") != 1 or quantization.get("clipping_ratio") != 1
            or quantization.get("native_kernel") is not False
            or "NO VQ" not in str(quantization.get("weights", ""))):
        raise ValueError("The worker only accepts the frozen no-VQ R128 protocol")

    lanes = p.get("lanes", {})
    opt = lanes.get("optimization", {})
    expected_opt = [
        ("grouped_lr1e4", 0.0001, C8),
        ("interleaved_lr1e3", 0.001, [2000, 2021, 2051, 2066, 2019, 2025, 2057, 2070]),
        ("interleaved_lr1e4", 0.0001, [2000, 2021, 2051, 2066, 2019, 2025, 2057, 2070]),
    ]
    got_opt = [(a.get("name"), a.get("learning_rate"), a.get("order"))
               for a in opt.get("arms", [])]
    if (got_opt != expected_opt or opt.get("updates") != 8
            or opt.get("contexts_per_update") != 1 or opt.get("training_exposures") != 8):
        raise ValueError("Frozen optimization arms or budgets differ")
    spec = lanes.get("coverage_c80", {})
    arms = spec.get("arms", [])
    if (len(arms) != 1 or arms[0].get("name") != "C80_once_lr1e4"
            or arms[0].get("learning_rate") != 0.0001
            or spec.get("updates") != 8 or spec.get("contexts_per_update") != 10
            or spec.get("training_exposures") != 80 or spec.get("case_ids") != C80
            or spec.get("repetitions_per_case") != 1):
        raise ValueError("Frozen C80 coverage arm or budget differs")
    if lane is not None and lane not in LANES:
        raise ValueError("Unsupported training lane: " + str(lane))


def coverage_schedule(rows, suites, case_ids, seed=0):
    """Build eight fixed 10-context updates over 80 unique C80 contexts."""
    if len(suites) != 4 or len(set(case_ids)) != len(case_ids):
        raise ValueError("C80 schedule requires four suites and unique cases")
    buckets = {suite: [] for suite in suites}
    for case in case_ids:
        if case not in rows or rows[case].get("suite") not in buckets:
            raise ValueError("Coverage case missing from frozen suite plan")
        buckets[rows[case]["suite"]].append(case)
    if any(len(values) != 20 for values in buckets.values()):
        raise ValueError("Each suite must contribute exactly 20 training exposures")

    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)
    cursor = Counter()
    schedule = []
    for update in range(8):
        promoted = {suites[update % 4], suites[(update + 1) % 4]}
        batch = []
        for suite in suites:
            count = 3 if suite in promoted else 2
            start = cursor[suite]
            batch.extend(buckets[suite][start:start + count])
            cursor[suite] += count
        rng.shuffle(batch)
        if len(batch) != 10:
            raise RuntimeError("Coverage update does not contain ten contexts")
        schedule.append(batch)
    if any(cursor[suite] != 20 for suite in suites):
        raise RuntimeError("Coverage schedule did not consume each suite exactly 20 times")
    observed = Counter(case for batch in schedule for case in batch)
    expected = Counter({case: 1 for case in case_ids})
    if observed != expected:
        raise RuntimeError("Coverage schedule changed per-case exposure counts")
    return schedule


def training_schedules(p, lane, rows):
    if lane not in LANES:
        raise ValueError("Unsupported training lane: " + str(lane))
    if lane == "optimization":
        return {
            arm["name"]: [[case] for case in arm["order"]]
            for arm in p["lanes"][lane]["arms"]
        }
    spec = p["lanes"][lane]
    schedule = coverage_schedule(rows, p["suites"], spec["case_ids"],
                                 seed=spec["schedule_seed"])
    return {spec["arms"][0]["name"]: schedule}


def validate_plan(plan, p):
    rows = {row["case_id"]: row for row in plan.get("inputs", [])}
    suites = p["suites"]
    if (plan.get("status") != "complete"
            or plan.get("calibration_cohorts", {}).get("c8") != C8
            or plan.get("calibration_cohorts", {}).get("c80") != C80
            or any(case not in rows for case in C8 + C80 + HELDOUT)
            or Counter(rows[c]["suite"] for c in C8) != Counter({s: 2 for s in suites})
            or Counter(rows[c]["suite"] for c in C80) != Counter({s: 20 for s in suites})
            or Counter(rows[c]["suite"] for c in HELDOUT) != Counter({s: 10 for s in suites})):
        raise RuntimeError("Frozen calibration/evaluation plan membership differs")
    return rows


def run(p, lane):
    # Do this before importing Torch, loading the model, or reading model-scale data.
    import sampler
    sampler._require_compute_allocation()
    validate_protocol(p, lane)

    import numpy as np
    import torch
    import runner
    import calibration_source as source
    import latent_reference
    from checkpointing import (atomic_torch_save, capture_rng, restore_rng,
                                to_cpu, optimizer_to_device)

    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(p["remote_root"])
    out = root / "results" / lane
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists() and read(out / "summary.json").get("complete"):
        raise RuntimeError("Lane already complete; refuse duplicate work")
    job = os.environ.get("PBS_JOBID")
    if not job or not os.environ.get("FW_JOB_START_EPOCH"):
        raise RuntimeError("PBS_JOBID and FW_JOB_START_EPOCH are required")
    started = time.monotonic()
    deadline = float(os.environ["FW_JOB_START_EPOCH"]) + p["allocation"]["soft_deadline_seconds"]
    reserve = int(os.environ.get("FW_SAVE_RESERVE_SECONDS", "300"))
    stopped = []
    signal.signal(signal.SIGTERM, lambda signum, frame: stopped.append(signum))

    progress_state = {
        "protocol": PROTOCOL, "lane": lane, "status": "running", "stage": "preflight",
        "pbs_jobid": job, "arms": {}, "new_teacher_queries": 0,
        "wall_seconds_this_job": 0.0,
    }

    def progress(**extra):
        progress_state.update(extra)
        progress_state["pbs_jobid"] = job
        progress_state["wall_seconds_this_job"] = time.monotonic() - started
        save(out / "progress.json", progress_state)
        compact = {"lane": lane, "stage": progress_state.get("stage"),
                   "current_arm": progress_state.get("current_arm"),
                   "arms": progress_state["arms"],
                   "new_teacher_queries": progress_state["new_teacher_queries"]}
        print("R128_PROGRESS " + json.dumps(compact, allow_nan=False), flush=True)

    def enough():
        if stopped or time.time() + reserve >= deadline:
            progress(status="checkpointed", stage="checkpointed")
            raise TimeoutError("Committed teachers, updates, and query records are resumable")

    progress()
    plan = read(p["calibration"]["plan_file"])
    rows = validate_plan(plan, p)
    schedules = training_schedules(p, lane, rows)
    reference_root = Path(p["parent_root"]) / "results" / "reference"
    reference_receipt = read(reference_root / "reference_receipt.json")
    reference_cache = torch.load(reference_root / "reference_cache.pt", map_location="cpu", weights_only=True)
    if (not reference_receipt.get("complete") or not reference_receipt.get("passed")
            or reference_cache.get("normalization", {}).get("normalization_c")
            != p["training"]["normalization_c"]
            or set(reference_cache.get("teacher", {})) != {str(case) for case in C8}
            or any(row.get("plan_row") != rows[row["case_id"]]
                   for row in reference_cache.get("rows", []))):
        raise RuntimeError("Verified C8 BF16 teacher cache binding differs")

    baseline = Path(p["baseline_root"])
    baseline_protocol = read(baseline / "protocol.json")
    teacher_root = baseline / "results" / "teacher"
    heldout_receipt = read(teacher_root / "receipt.json")
    if (not heldout_receipt.get("complete")
            or heldout_receipt.get("case_ids") != HELDOUT
            or heldout_receipt.get("seed_indices") != [0, 1]):
        raise RuntimeError("Verified heldout BF16 teacher coverage differs")

    shared_c80_teachers = {}
    is_coverage = lane == "coverage_c80"
    c80_teacher_dir = Path(p["c80_teacher_directory"]) if is_coverage else None
    if is_coverage:
        c80_receipt = read(c80_teacher_dir / "receipt.json")
        if (c80_receipt.get("complete") is not True or c80_receipt.get("queries") != 80
                or c80_receipt.get("case_ids") != C80
                or c80_receipt.get("seed_indices") != [0]):
            raise RuntimeError("Shared verified C80 teacher receipt differs")
        for case in C80:
            saved = torch.load(c80_teacher_dir / f"{case}_0.pt", map_location="cpu", weights_only=True)
            action, future = saved.get("action"), saved.get("future")
            if (saved.get("plan_row") != rows[case]
                    or saved.get("sampler_seed") != rows[case]["sampler_seeds"][0]
                    or action is None or tuple(action.shape) != (32, 7)
                    or future is None or tuple(future.shape) != (1, 48, 3, 14, 28)
                    or not bool(torch.isfinite(action).all())
                    or not bool(torch.isfinite(future).all())):
                raise RuntimeError("Shared C80 teacher tensor or identity differs: " + str(case))
            shared_c80_teachers[case] = saved
        for case in C8:
            cached = reference_cache["teacher"][str(case)]
            for name in ("action", "future"):
                error = float((cached[name].float() - shared_c80_teachers[case][name].float()).abs().max())
                if error > 1e-6:
                    raise RuntimeError("C80/C8 BF16 teacher differs for " + str(case) + ": " + name)
    new_teacher_queries = 0
    teacher_receipt = {
        "protocol": PROTOCOL, "lane": lane, "complete": True,
        "case_ids": C80 if is_coverage else C8,
        "seed_indices": [0], "new_teacher_queries": 0,
        "c8_source": str(reference_root / "reference_cache.pt"),
        "heldout_source": str(teacher_root),
        "c80_source": str(c80_teacher_dir) if is_coverage else None,
    }
    save(out / "teacher" / "receipt.json", teacher_receipt)
    progress(new_teacher_queries=0, stage="teacher_ready")

    observations = {}
    runtime = runner.Runtime(out)
    runtime.model.eval()
    runtime.arm("bf16")

    def datum(case):
        if case not in observations:
            observation, meta = source.load_observation(
                np, Path(p["calibration"]["plan_file"]).parent, rows[case])
            observations[case] = (runtime.datum(observation), meta["description"])
        return observations[case]

    from rotation_bank import RotationBanks
    from rotation_selfcheck import run as rotation_selfcheck
    if (out / "selfcheck.json").exists():
        checks = read(out / "selfcheck.json")
        if not checks.get("passed"):
            raise RuntimeError("Saved R128 allocation selfcheck did not pass")
    else:
        enough()
        checks = rotation_selfcheck()
        if not checks.get("passed"):
            raise RuntimeError("R128 GPU numerical selfcheck failed")
        save(out / "selfcheck.json", checks)

    rb = RotationBanks(runtime, p["calibration"]["bank_directory"])
    if (len(rb.modules) != 614 or set(rb.rotations.raw) != {"action", "video"}
            or any(parameter.requires_grad for parameter in runtime.model.parameters())):
        raise RuntimeError("Expected 614 frozen Linears and exactly two trainable stream rotations")

    teacher_cache = {}

    def teacher(case, seed):
        key = (case, seed)
        if key in teacher_cache:
            return teacher_cache[key]
        if case in C8:
            value = reference_cache["teacher"][str(case)]
        elif case in C80:
            if not is_coverage:
                raise RuntimeError("C80 teacher requested outside the C80 lane")
            value = shared_c80_teachers[case]
        else:
            path = teacher_root / f"{case}_{seed}.pt"
            value = torch.load(path, map_location="cpu", weights_only=True)
            if (value.get("protocol") != baseline_protocol.get("protocol")
                    or value.get("plan_row") != rows[case]
                    or value.get("seed_index") != seed
                    or value.get("sampler_seed") != rows[case]["sampler_seeds"][seed]):
                raise RuntimeError("Heldout teacher tensor identity differs")
        if not all(key in value for key in ("action", "future")):
            raise RuntimeError("Teacher record lacks action or future latent")
        teacher_cache[key] = value
        return value

    def query(case, seed, arm, grad=False):
        value, description = datum(case)
        runtime.description, runtime.current_arm = description, f"{lane}/{arm}"
        rb.seen.clear()
        rb.rotations.refresh()  # A fresh Cayley matrix graph for every context.
        with torch.set_grad_enabled(grad):
            # Keep each Cayley solve outside sampler and Linear checkpoint regions.
            for stream in rb.rotations.raw:
                rb.rotations.matrix(stream)
            action, counts, future = sampler.differentiable_action(
                runtime, value, rows[case]["sampler_seeds"][seed],
                checkpointed=grad, return_future=True)
        if len(rb.seen) != 614:
            raise RuntimeError("Full sampler did not reach all 614 Linears")
        return action, future, counts

    def components(action, future, target):
        action_target = target["action"].to("cuda").float()
        a = (action[:10, :6].float() - action_target[:10, :6]).square().mean()
        z = latent_reference.future_nmse_loss(torch, future, target["future"].to("cuda"))
        return a + LAMBDA * z, a, z

    def metric_values(action, future, target):
        total, a, z = components(action, future, target)
        delta = action.float() - target["action"].to("cuda").float()
        return {
            "first10motor_rmse": float(a.detach().sqrt()),
            "first10motor_mse": float(a.detach()),
            "future_latent_nmse": float(z.detach()),
            "mixed_loss": float(total.detach()),
            "full32motor_rmse": float(delta[:, :6].square().mean().sqrt().detach()),
            "late22motor_rmse": float(delta[10:, :6].square().mean().sqrt().detach()),
            "first10gripper_rmse": float(delta[:10, 6].square().mean().sqrt().detach()),
        }

    training_doc = {"protocol": PROTOCOL, "lane": lane, "arms": {}}
    training_path = out / "training.json"
    if training_path.exists():
        old_training = read(training_path)
        if old_training.get("protocol") != PROTOCOL or old_training.get("lane") != lane:
            raise RuntimeError("Training history binding differs")
        training_doc["arms"].update(old_training.get("arms", {}))

    def save_training_arm(arm, state, exposures):
        training_doc["arms"][arm] = {
            "updates": state["updates"], "training_exposures": exposures,
            "history": state["history"],
            "initial_u_identity_parity": state.get("initial_u_identity_parity"),
        }
        save(training_path, training_doc)

    def query_count(arm, split):
        return len(list((out / split / arm / "queries").glob("*.json")))

    def arm_progress(arm, updates, exposures, stage):
        progress_state["arms"][arm] = {
            "stage": stage, "updates": updates, "training_exposures": exposures,
            "completed_queries": {
                split: query_count(arm, split) for split in ("calibration", "heldout", "c80")
            },
        }
        progress(current_arm=arm)

    def query_record(arm, split, case, seed, reused=None, source_path=None):
        path = out / split / arm / "queries" / f"{case}_{seed}.json"
        binding = {
            "protocol": PROTOCOL, "lane": lane, "arm": arm, "split": split,
            "case_id": case, "seed_index": seed, "suite": rows[case]["suite"],
            "sampler_seed": rows[case]["sampler_seeds"][seed],
        }
        if path.exists():
            record = read(path)
            if any(record.get(key) != value for key, value in binding.items()):
                raise RuntimeError("Saved query identity differs")
            if any(not math.isfinite(record.get(key, float("nan"))) for key in METRICS):
                raise RuntimeError("Saved query metrics are nonfinite")
        else:
            if reused is None:
                enough()
                tick = time.monotonic()
                action, future, counts = query(case, seed, arm)
                values = metric_values(action, future, teacher(case, seed))
                del action, future
                reuse_flag = False
            else:
                tick = time.monotonic()
                values = {key: reused[key] for key in METRICS}
                counts = reused.get("counts")
                reuse_flag = True
            if any(not math.isfinite(values[key]) or values[key] < 0 for key in METRICS):
                raise RuntimeError("Query metric is negative or nonfinite")
            record = {
                **binding, **values, "counts": counts, "reused": reuse_flag,
                "source_path": source_path, "pbs_jobid": job,
                "seconds": time.monotonic() - tick,
            }
            save(path, record)
        arm_progress(arm, progress_state["arms"].get(arm, {}).get("updates", 8),
                     progress_state["arms"].get(arm, {}).get("training_exposures", 0),
                     "final_evaluation")
        return record

    reports = {}
    schedules_for_lane = schedules
    for arm, update_cases in schedules_for_lane.items():
        learning_rate = next(spec["learning_rate"] for spec in p["lanes"][lane]["arms"]
                             if spec["name"] == arm)
        exposures = sum(len(cases) for cases in update_cases)
        arm_dir = out / "arms" / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = arm_dir / "snapshot.pt"
        arm_summary_path = arm_dir / "summary.json"

        if arm_summary_path.exists() and read(arm_summary_path).get("complete"):
            saved_summary = read(arm_summary_path)
            if (saved_summary.get("protocol") != PROTOCOL or saved_summary.get("lane") != lane
                    or saved_summary.get("arm") != arm):
                raise RuntimeError("Completed arm summary binding differs")
            snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=True)
            if snapshot.get("protocol") != p or snapshot.get("lane") != lane or snapshot.get("arm") != arm:
                raise RuntimeError("Completed arm snapshot binding differs")
            state = snapshot["state"]
            save_training_arm(arm, state, exposures)
            reports[arm] = saved_summary["report"]
            arm_progress(arm, state["updates"], exposures, "complete")
            continue

        # Reset explicitly for every fresh arm; each arm owns its own optimizer and snapshot.
        params = list(rb.rotations.raw.values())
        for value in params:
            value.requires_grad_(True)
        optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
        if snapshot_path.exists():
            snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=True)
            if (snapshot.get("protocol") != p or snapshot.get("lane") != lane
                    or snapshot.get("arm") != arm or snapshot.get("learning_rate") != learning_rate):
                raise RuntimeError("Arm resume binding differs")
            state = snapshot["state"]
            if state.get("updates") != len(state.get("history", [])) or not 0 <= state["updates"] <= 8:
                raise RuntimeError("Arm resume cursor/history differs")
            rb.rotations.restore(snapshot["rotations"])
            optimizer.load_state_dict(snapshot["optimizer"])
            optimizer_to_device(torch, optimizer, "cuda")
            restore_rng(torch, snapshot["rng"])
        else:
            random_seed = p["training"]["random_seed"]
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            with torch.no_grad():
                for value in params:
                    value.zero_()
            rb.rotations.refresh()
            state = {"updates": 0, "history": [], "lineage": []}

        if job not in state["lineage"]:
            state["lineage"].append(job)

        def persist_arm():
            atomic_torch_save(torch, snapshot_path, {
                "protocol": p, "lane": lane, "arm": arm, "learning_rate": learning_rate,
                "state": state, "rotations": rb.rotations.snapshot(),
                "optimizer": to_cpu(optimizer.state_dict()), "rng": capture_rng(torch),
            })

        if state["updates"] > 0 and not state.get("initial_u_identity_parity"):
            raise RuntimeError("Resumed arm is missing its initial identity replay")
        if not state.get("initial_u_identity_parity"):
            if state["updates"] != 0:
                raise RuntimeError("Initial identity replay must precede every optimizer update")
            source_path = (Path(p["diagnostic_root"]) / "results" / "calibration"
                           / "H_W16A4" / "queries" / "2000_0.json")
            expected = read(source_path)
            expected_binding = {
                "protocol": p["diagnostic_protocol"], "arm": "H_W16A4",
                "split": "calibration", "case_id": 2000, "seed_index": 0,
                "suite": rows[2000]["suite"],
                "sampler_seed": rows[2000]["sampler_seeds"][0],
            }
            if any(expected.get(key) != value for key, value in expected_binding.items()):
                raise RuntimeError("Pinned H_W16A4 source query identity differs")
            action, future, _ = query(2000, 0, arm)
            identity_metrics = metric_values(action, future, teacher(2000, 0))
            del action, future
            try:
                parity = compare_metrics(identity_metrics, expected)
            except ValueError as exc:
                raise RuntimeError("Fresh H+U=I source replay failed: " + str(exc)) from exc
            state["initial_u_identity_parity"] = {
                **parity, "source_path": str(source_path),
                "metrics": identity_metrics,
            }

        persist_arm()
        save_training_arm(arm, state, exposures)
        progress(stage="training", current_arm=arm)
        while state["updates"] < 8:
            enough()
            update = state["updates"]
            cases = update_cases[update]
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            context_rows = []
            update_losses = []
            for case in cases:
                action, future, counts = query(case, 0, arm, grad=True)
                target = teacher(case, 0)
                total, a, z = components(action, future, target)
                values = metric_values(action.detach(), future.detach(), target)
                gradients = torch.autograd.grad(total / len(cases), params, allow_unused=True)
                rgrad = {}
                for stream, parameter, gradient in zip(rb.rotations.raw, params, gradients):
                    connected = gradient is not None
                    finite = connected and bool(torch.isfinite(gradient).all())
                    magnitude = float(gradient.abs().sum()) if connected else 0.0
                    rgrad[stream] = {"connected": connected, "finite": finite,
                                     "nonzero": magnitude > 0, "abs_sum": magnitude}
                    if not connected or not finite or magnitude == 0:
                        raise RuntimeError("R128 per-context gradient is disconnected, nonfinite, or zero")
                    if parameter.grad is None:
                        parameter.grad = gradient.detach().clone()
                    else:
                        parameter.grad.add_(gradient.detach())
                update_losses.append({"case_id": case, "loss": float(total.detach()),
                                      "action_mse": float(a.detach()), "future_nmse": float(z.detach())})
                context_rows.append({"case_id": case, "suite": rows[case]["suite"],
                                     "sampler_seed": rows[case]["sampler_seeds"][0],
                                     "loss": values["mixed_loss"], "metrics": values,
                                     "rgrad": rgrad, **counts})
                del action, future, total, a, z, target, gradients

            if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
                   for parameter in params):
                raise RuntimeError("Accumulated R128 gradient is missing or nonfinite")
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            rb.rotations.refresh()
            state["updates"] += 1

            post = []
            for case in cases:
                action, future, counts = query(case, 0, arm)
                post.append({"case_id": case, "metrics": metric_values(action, future, teacher(case, 0)),
                             **counts})
                del action, future
            torch.cuda.synchronize()
            row = {
                "update": state["updates"], "cases": cases,
                "loss": sum(item["loss"] for item in update_losses) / len(cases),
                "action_mse": sum(item["action_mse"] for item in update_losses) / len(cases),
                "future_nmse": sum(item["future_nmse"] for item in update_losses) / len(cases),
                "contexts": context_rows, "post_update": post,
                "gradient_norm_before_clip": float(grad_norm), "gradient_clip": 1.0,
                "seconds": time.monotonic() - tick,
            }
            state["history"].append(row)
            persist_arm()  # Commit parameters, Adam moments, and cursor together.
            save_training_arm(arm, state, exposures)
            arm_progress(arm, state["updates"], exposures, "training")

        # Freeze the final checkpoint before any endpoint evaluation. No mid-update
        # cohort evaluation is mixed into the fixed update-8 result.
        for parameter in params:
            parameter.requires_grad_(False)
        rb.rotations.refresh()
        orthogonality = rb.rotations.orthogonality()
        movement = {
            stream: float((rb.rotations.matrix(stream) - torch.eye(128, device="cuda")).norm())
            for stream in rb.rotations.raw
        }
        if max(orthogonality.values()) > 5e-5:
            raise RuntimeError("Final Cayley orthogonality failed")
        atomic_torch_save(torch, arm_dir / "deployment.pt", {
            "protocol": p, "lane": lane, "arm": arm, "updates": 8,
            "rotations": rb.rotations.snapshot(), "target_linears": 614,
        })

        parity_case = C8[0]
        action, future, _ = query(parity_case, 0, arm)
        value, _ = datum(parity_case)
        actual, _ = runtime.infer(value, rows[parity_case]["sampler_seeds"][0], measure=False)
        formal_parity = float((action.float().cpu() - actual.float().cpu()).abs().max())
        del action, future, actual
        if formal_parity > 1e-6:
            raise RuntimeError("Final sampler/formal_infer parity failed")

        progress(stage="final_evaluation", current_arm=arm)
        for case in C8:
            query_record(arm, "calibration", case, 0)
        for case in HELDOUT:
            for seed in (0, 1):
                query_record(arm, "heldout", case, seed)
        if is_coverage:
            for case in C80:
                source_path = None
                reused = None
                if case in C8:
                    source_path = out / "calibration" / arm / "queries" / f"{case}_0.json"
                    reused = read(source_path)
                query_record(arm, "c80", case, 0, reused=reused,
                             source_path=str(source_path) if source_path else None)

        counts = {split: query_count(arm, split) for split in ("calibration", "heldout", "c80")}
        expected_counts = {"calibration": 8, "heldout": 80,
                           "c80": 80 if is_coverage else 0}
        if counts != expected_counts:
            raise RuntimeError("Final endpoint denominators differ: " + str(counts))
        all_frozen = (all(not parameter.requires_grad for parameter in runtime.model.parameters())
                      and all(not parameter.requires_grad for parameter in rb.rotations.parameters()))
        if not all_frozen:
            raise RuntimeError("Final evaluation left trainable parameters enabled")
        report = {
            "updates": state["updates"], "training_exposures": exposures,
            "calibration_queries": counts["calibration"], "heldout_queries": counts["heldout"],
            "c80_queries": counts["c80"], "all_parameters_frozen": all_frozen,
            "formal_infer_max_abs": formal_parity, "orthogonality": orthogonality,
            "movement": movement,
            "source_replay_metric_max_abs": state["initial_u_identity_parity"]["max_abs_delta"],
        }
        arm_summary = {
            "protocol": PROTOCOL, "lane": lane, "arm": arm, "complete": True,
            "pbs_jobid": job, "report": report, "lineage": state["lineage"],
            "history_updates": len(state["history"]),
            "initial_u_identity_parity": state["initial_u_identity_parity"],
        }
        save(arm_summary_path, arm_summary)
        reports[arm] = report
        arm_progress(arm, state["updates"], exposures, "complete")

    progress(stage="aggregate", status="running")
    enough()
    from aggregate import write_lane_report
    write_lane_report(out, p, lane)
    summary = {
        "protocol": PROTOCOL, "status": "complete", "complete": True,
        "pbs_jobid": job, "lane": lane, "target_linears": 614,
        "arms": reports, "new_teacher_queries": new_teacher_queries,
        "lineage": sorted({lineage for arm in reports
                            for lineage in read(out / "arms" / arm / "summary.json").get("lineage", [])}),
        "scope": p["scope"],
    }
    save(out / "summary.json", summary)
    progress(status="complete", stage="complete", complete=True, arms=progress_state["arms"],
             new_teacher_queries=new_teacher_queries)
    print("R128_COMPLETE " + json.dumps(summary, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--lane", choices=LANES, required=True)
    args = parser.parse_args()
    protocol = read(args.protocol)
    try:
        run(protocol, args.lane)
    except TimeoutError:
        raise SystemExit(75)


if __name__ == "__main__":
    main()
