#!/usr/bin/env python3
"""Fit one bank-centered linear response adapter and evaluate new PushT episodes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import statistics
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PREVIOUS_RUNNER = HERE.parent / "terminal-response-loss" / "runner.py"
SCHEMA = "lewm-recurrent-student.linear-response-adapter-freeze"
DIM = 192
TRAIN_BANKS = 512
TRAIN_CANDIDATES = 64
FRESH_EPISODES = 8
EXPECTED_PREVIOUS_IDS = [10569, 7242, 13109, 2193, 3218, 2289, 8316, 1085]


def load_previous() -> Any:
    spec = importlib.util.spec_from_file_location("previous_terminal_response_runner", PREVIOUS_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("previous runner is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_sources(freeze: dict[str, Any], previous: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if freeze.get("schema") != SCHEMA or freeze.get("status") != "frozen_before_execution":
        raise ValueError("adapter freeze schema/status mismatch")
    source = freeze["source"]
    old_freeze = previous.load_json(Path(source["previous_freeze"]))
    previous.validate_freeze(old_freeze)
    if source["previous_job_id"] != "25549480.pbs101":
        raise ValueError("prior job identity changed")
    old_summary = previous.load_json(Path(source["prior_summary"]))
    old_selection = previous.load_json(Path(source["prior_selection"]))
    if old_summary.get("pbs_job_id") != source["previous_job_id"] or old_summary.get("status") != "PREDICTOR_LEVEL_COMPLETE":
        raise ValueError("prior experiment summary identity/status mismatch")
    if old_selection.get("ordered_episode_ids") != EXPECTED_PREVIOUS_IDS:
        raise ValueError("previous eight episode identities changed")
    if old_summary["fresh_selection"]["fresh_episode_ids"] != EXPECTED_PREVIOUS_IDS:
        raise ValueError("previous summary selection differs from saved selection")
    expected = f'{source["previous_dir"]}/terminal_response_step3000.pt'
    if source["student_checkpoint"] != expected or old_summary["training"]["checkpoints"]["terminal_response"] != expected:
        raise ValueError("treatment checkpoint path changed")
    fit = freeze["fit"]
    evaluation = freeze["evaluation"]
    if (fit["terminal_horizon_index"], fit["latent_dim"], fit["ridge_ratio"]) != (4, DIM, 0.1):
        raise ValueError("frozen adapter fit changed")
    if (evaluation["selection_seed"], evaluation["selection_start"], evaluation["new_episodes"], evaluation["banks"], evaluation["candidates_per_bank"]) != (20300903, 600, 8, 48, 300):
        raise ValueError("frozen evaluation design changed")
    if evaluation["action_prefix_seeds"] != [20301007, 20301008] or evaluation["arms"] != ["identity", "scalar", "linear"]:
        raise ValueError("frozen arm/seed identity changed")
    if freeze["gates"]["absolute_predictor"] != old_freeze["gates"]["absolute_predictor"]:
        raise ValueError("absolute gate must be inherited unchanged")
    if freeze["gates"]["stratum_protection"] != old_freeze["gates"]["stratum_protection"]:
        raise ValueError("stratum gate must be inherited unchanged")
    if freeze["gates"]["relative_mechanism"] != {
        key: old_freeze["gates"]["relative_mechanism"][key]
        for key in freeze["gates"]["relative_mechanism"]
    }:
        raise ValueError("relative gate must be inherited unchanged")
    return old_freeze, old_summary, old_selection


def select_new_episodes(previous: Any, old_freeze: dict[str, Any], old_selection: dict[str, Any]) -> tuple[list[int], dict[str, Any]]:
    import h5py

    source = old_freeze["source"]
    used_ids, used_meta = previous.read_used_episode_ids(source["used_episode_manifests"])
    dataset = Path(source["dataset_path"])
    prior, prior_meta = previous.select_fresh_episode_ids(
        dataset,
        Path(source["phase2_manifest_path"]),
        Path(source["used_episode_manifests"]["seeded_pilot32"]),
        used_ids,
        20300903,
    )
    if prior != old_selection["ordered_episode_ids"]:
        raise ValueError("recomputed prior fresh selection differs from saved identities")
    with h5py.File(dataset, "r") as handle:
        lengths = [int(value) for value in handle["ep_len"][:]]
    valid = [episode for episode, length in enumerate(lengths) if length >= 26]
    random.Random(20300903).shuffle(valid)
    manifest = previous.load_json(Path(source["phase2_manifest_path"]))
    splits = manifest["splits"]
    used = {int(row["episode_id"]) for row in splits["heldout"] + splits["train"]}
    for ids in used_ids.values():
        used.update(int(value) for value in ids)
    candidates = [episode for episode in valid[600:] if episode not in used]
    if candidates[:8] != prior:
        raise ValueError("prior episode prefix changed in frozen candidate stream")
    fresh = candidates[8:16]
    if len(fresh) != FRESH_EPISODES or len(set(fresh)) != FRESH_EPISODES or set(fresh) & (used | set(prior)):
        raise ValueError("new evaluation episodes overlap prior identities")
    return fresh, {
        "ordered_episode_ids": fresh,
        "previous_episode_ids": prior,
        "selection_seed": 20300903,
        "selection_start": 600,
        "candidate_stream_slice": "[8:16] after frozen exclusions",
        "previous_selection_recomputed": prior_meta,
        "used_episode_manifests": used_meta,
        "fit_episode_disjoint": True,
        "previous_diagnostic_episode_disjoint": True,
    }


def fit_adapter(reference: Any, student: Any, rows: list[Any], torch: Any) -> tuple[Any, float, dict[str, Any]]:
    if len(rows) != TRAIN_BANKS:
        raise ValueError("adapter requires exactly 512 frozen training banks")
    gram = torch.zeros((DIM, DIM), device="cuda", dtype=torch.float64)
    cross = torch.zeros_like(gram)
    floor_active = 0
    with torch.inference_mode():
        for row in rows:
            context = reference.base._tensor(row["latent_history"], dtype=torch.float32).to("cuda").expand(TRAIN_CANDIDATES, -1, -1)
            actions = reference.base._tensor(row["future_actions"], dtype=torch.float32).to("cuda")
            teacher = reference.base._tensor(row["teacher_targets"], dtype=torch.float32).to("cuda")[:, -1, :]
            student_terminal = student(context, actions)[:, -1, :]
            x = (student_terminal - student_terminal.mean(dim=0, keepdim=True)).to(torch.float64)
            y = (teacher - teacher.mean(dim=0, keepdim=True)).to(torch.float64)
            energy = y.square().mean()
            floor_active += int(bool((energy < 1e-6).item()))
            weight = 1.0 / (TRAIN_CANDIDATES * energy.clamp_min(1e-6))
            gram.add_((x.T @ x) * weight)
            cross.add_((x.T @ y) * weight)
    gram.div_(TRAIN_BANKS)
    cross.div_(TRAIN_BANKS)
    ridge = 0.1 * gram.trace() / DIM
    if not bool(torch.isfinite(gram).all().item() and torch.isfinite(cross).all().item()) or float(ridge.item()) <= 0:
        raise FloatingPointError("non-finite or zero adapter normal equations")
    eye = torch.eye(DIM, device="cuda", dtype=torch.float64)
    matrix64 = torch.linalg.solve(gram + ridge * eye, cross + ridge * eye)
    scalar = float(((cross.trace() + ridge * DIM) / (gram.trace() + ridge * DIM)).item())
    matrix = matrix64.to(torch.float32)
    if not bool(torch.isfinite(matrix).all().item()) or not math.isfinite(scalar):
        raise FloatingPointError("non-finite fitted adapter")
    meta = {
        "training_banks": TRAIN_BANKS,
        "candidates_per_bank": TRAIN_CANDIDATES,
        "teacher_energy_floor_active_banks": floor_active,
        "ridge_ratio": 0.1,
        "ridge_lambda": float(ridge.item()),
        "gram_trace": float(gram.trace().item()),
        "scalar_alpha": scalar,
        "matrix_frobenius_from_identity": float(torch.linalg.vector_norm(matrix64 - eye).item()),
        "matrix_finite": True,
        "teacher_targets_from_cached_training_rows": True,
    }
    return matrix, scalar, meta


class BankAdapter:
    def __init__(self, student: Any, matrix: Any) -> None:
        self.student = student
        self.matrix = matrix

    def __call__(self, context: Any, actions: Any) -> Any:
        import torch

        prediction = self.student(context, actions)
        terminal = prediction[:, -1, :]
        mean = terminal.mean(dim=0, keepdim=True)
        corrected = mean + (terminal - mean) @ self.matrix
        return torch.cat((prediction[:, :-1, :], corrected[:, None, :]), dim=1)


def evaluate(previous: Any, reference: Any, official_model: Any, student: Any, rows: list[Any], seeds: list[int], matrix: Any, scalar: float, torch: Any) -> dict[str, Any]:
    eye = torch.eye(DIM, device="cuda", dtype=torch.float32)
    arms = {
        "identity": student,
        "scalar": BankAdapter(student, scalar * eye),
        "linear": BankAdapter(student, matrix),
    }
    result: dict[str, Any] = {}
    with torch.inference_mode():
        for name, predictor in arms.items():
            blocks = [
                previous.evaluate_block(reference, official_model, predictor, row, block, int(seed))
                for row in rows
                for block, seed in enumerate(seeds)
            ]
            if len(blocks) != 48 or len({item["pairing_key"] for item in blocks}) != 48:
                raise ValueError(f"{name} did not cover exactly 48 unique banks")
            result[name] = {"blocks": blocks, **previous.summarize_evaluation(blocks)}
    keys = [{item["pairing_key"] for item in result[name]["blocks"]} for name in arms]
    if keys[0] != keys[1] or keys[0] != keys[2]:
        raise ValueError("adapter arm candidate banks do not pair")
    return result


def run(freeze_path: Path, output_dir: Path) -> dict[str, Any]:
    previous = load_previous()
    host = previous.require_compute_node()  # Before HDF5, model, checkpoint, or bank I/O.
    freeze = previous.load_json(freeze_path)
    old_freeze, old_summary, old_selection = read_sources(freeze, previous)
    source = old_freeze["source"]
    reference = previous.load_reference_modules()
    import torch

    checkpoint = torch.load(freeze["source"]["student_checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint.get("arm") != "terminal_response" or checkpoint.get("step") != 3000 or checkpoint.get("schema") != old_freeze["schema"] + ".checkpoint":
        raise ValueError("frozen treatment checkpoint identity mismatch")
    provenance = checkpoint.get("provenance", {})
    if provenance.get("freeze") != freeze["source"]["previous_freeze"] or provenance.get("training_rows_source_job") != "25213164.pbs101" or provenance.get("teacher_targets_reused_from_cached_rows") is not True:
        raise ValueError("frozen treatment checkpoint provenance mismatch")
    student = reference.instantiate_student("balanced_base").to("cuda")
    reference.load_full_state(student, checkpoint["state_dict"])
    student.eval()
    raw_rows = torch.load(source["balanced_rows_path"], map_location="cpu", weights_only=False)
    train_rows = previous.validate_training_rows(raw_rows, Path(source["phase2_manifest_path"]))
    matrix, scalar, fit_meta = fit_adapter(reference, student, train_rows, torch)

    # Selection is recorded before any new held-out row is opened for inference or scoring.
    fresh_ids, selection = select_new_episodes(previous, old_freeze, old_selection)
    output_dir.mkdir(parents=True, exist_ok=True)
    previous.write_json(output_dir / "selection.json", selection)
    official_model = reference.base.load_official_checkpoint(Path(source["stablewm_home"]))
    official_model.requires_grad_(False)
    rows, row_meta = previous.build_fresh_rows(
        reference, official_model, old_freeze, Path(source["dataset_path"]), fresh_ids
    )
    evaluations = evaluate(previous, reference, official_model, student, rows, freeze["evaluation"]["action_prefix_seeds"], matrix, scalar, torch)
    paired_linear = previous.paired_episode_deltas(evaluations["identity"], evaluations["linear"])
    paired_scalar = previous.paired_episode_deltas(evaluations["identity"], evaluations["scalar"])

    causality = reference.base.causality_test(student, rows[0], 20301007, 1e-6)
    latency = reference.base.predictor_latency(official_model, BankAdapter(student, matrix), rows[0], warmup=3, repeats=10)
    gates = freeze["gates"]
    integrity = {"causality": causality, "predictor_latency": latency}
    relative = previous.mechanism_gate(paired_linear, gates["relative_mechanism"], evaluations["linear"])
    absolute = previous.absolute_predictor_gate(
        evaluations["linear"], old_summary["training"]["treatment"], gates["absolute_predictor"], integrity
    )
    stratum = previous.stratum_protection_gate(evaluations["linear"], gates["stratum_protection"])
    overall = "GO" if all(item["status"] == "GO" for item in (relative, absolute, stratum)) else "NO-GO"
    result = {
        "schema": SCHEMA + ".result",
        "status": "PREDICTOR_LEVEL_COMPLETE",
        "pbs_job_id": os.environ["PBS_JOBID"],
        "compute_host": host,
        "freeze": str(freeze_path),
        "source": {
            "previous_job_id": "25549480.pbs101",
            "checkpoint": freeze["source"]["student_checkpoint"],
            "training_rows": source["balanced_rows_path"],
            "architecture_and_checkpoint_unchanged": True,
        },
        "fit": fit_meta,
        "selection": selection,
        "fresh_rows_metadata": row_meta,
        "evaluation": evaluations,
        "paired_linear_minus_identity": paired_linear,
        "paired_scalar_minus_identity": paired_scalar,
        "gates": {"relative_mechanism": relative, "absolute_predictor": absolute, "stratum_protection": stratum, "overall": overall},
        "latency": latency,
        "causality_of_unchanged_student": causality,
        "scope": freeze["scope"],
        "claim_boundary": "A result on eight new episodes tests only this bank-relative linear correction at predictor level; teacher is absent at adapter inference. It does not establish official CEM, planner, or closed-loop performance.",
    }
    path = output_dir / "linear_response_adapter_summary.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite result: {path}")
    previous.write_json(path, result)
    print(f"wrote {path}; linear gate={overall}; fresh episodes={fresh_ids}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.freeze, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
