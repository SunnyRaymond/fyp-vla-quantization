"""CPU-safe PBS self-checks for microscale decoder gradients and run bindings."""
from __future__ import annotations


def self_check(torch=None):
    from train_banks import self_check as micro_decoder_check
    from run_distill import (advance_training, evaluation_pairs, validate_run_binding)

    micro_decoder_check()
    protocol = {"protocol": "fastwam-vq-micro-latent-search-v1",
                "search": {"round1_rhos": [0.1, 0.5, 1.0]},
                "additional_round1_candidates": [
                    {"run_id": "r1-r003", "rho": 0.7},
                    {"run_id": "r1-r004", "rho": 1.5}]}
    first = validate_run_binding(protocol, "r1-r000", 0.1, 1, "search")
    assert first["candidate_run_id"] == "r1-r000"
    assert validate_run_binding(protocol, "r1-r003", 0.7, 1, "search")["rho"] == 0.7
    assert validate_run_binding(protocol, "r1-r004", 1.5, 1, "search")["rho"] == 1.5
    try:
        validate_run_binding(protocol, "r1-r003", 0.5, 1, "search")
    except ValueError:
        pass
    else:
        raise AssertionError("Round 1 must reject an unfrozen rho")

    round2 = {"protocol": protocol["protocol"], "round": 2, "selection_job_id": "chooser-2",
              "candidates": [{"run_id": "r2-r000", "rho": 0.0},
                             {"run_id": "r2-r001", "rho": 0.2},
                             {"run_id": "r2-r002", "rho": 1.5}]}
    zero = validate_run_binding(protocol, "r2-r000", 0.0, 2, "search", round2)
    assert zero["rho"] == 0.0
    accepted = validate_run_binding(protocol, "r2-r001", 0.2, 2, "search", round2)
    assert accepted["round"] == 2
    try:
        validate_run_binding(protocol, "r2-r099", 0.2, 2, "search", round2)
    except ValueError:
        pass
    else:
        raise AssertionError("Round 2 must bind each rho to a frozen run ID")
    try:
        validate_run_binding(protocol, "r2-r000", -0.1, 2, "search", round2)
    except ValueError:
        pass
    else:
        raise AssertionError("Negative rho must be rejected")

    optimal = {"protocol": protocol["protocol"], "rounds_complete": 2, "selected_run_id": "r2-r001",
               "selected_job_id": "search-2", "selected_result_dir": "/remote/results/r2-r001-search-2",
               "selected_epoch": 1, "selected_round": 2, "rho": 0.2, "latent_weight": 0.4,
               "normalization_c": 2.0,
               "selection_job_id": "chooser-2", "selection_source_jobs": ["chooser-1", "chooser-2"]}
    final = validate_run_binding(protocol, "final", 0.2, 2, "final", optimal,
                                 optimal["selected_result_dir"])
    assert (final["selected_run_id"] == "r2-r001" and final["selected_epoch"] == 1
            and final["selected_round"] == 2 and final["round"] == 2)
    try:
        validate_run_binding(protocol, "final", 0.5, 2, "final", optimal,
                             optimal["selected_result_dir"])
    except ValueError:
        pass
    else:
        raise AssertionError("Final mode must preserve the globally selected rho")
    optimal_zero = {**optimal, "selected_run_id": "r2-r000", "selected_job_id": "search-zero",
                    "selected_result_dir": "/remote/results/r2-r000-search-zero",
                    "selected_epoch": 0, "rho": 0.0, "latent_weight": 0.0}
    final_zero = validate_run_binding(protocol, "final", 0.0, 2, "final", optimal_zero,
                                      optimal_zero["selected_result_dir"])
    assert final_zero["selected_run_id"] == "r2-r000" and final_zero["selected_epoch"] == 0

    state = {"phase": "train", "epoch": 1, "next_train_index": 0,
             "updates_completed": 0, "epochs_completed": 0}
    for _ in range(8):
        advance_training(state, {"calibration_cases": list(range(8))})
    assert state == {"phase": "selection", "epoch": 1, "next_train_index": 8,
                     "updates_completed": 8, "epochs_completed": 1}
    cases, seeds = [1004, 1005], [0, 1]
    completed = [{"case_id": case, "seed_index": seed}
                 for case in cases[:1] for seed in seeds]
    assert evaluation_pairs(cases, seeds, completed) == [(1005, 0), (1005, 1)]
    try:
        evaluation_pairs(cases, seeds, completed + completed[:1])
    except RuntimeError:
        pass
    else:
        raise AssertionError("Evaluation resume must reject duplicate/out-of-order contexts")
    print("MICRO_TRAINING_CPU_SELF_CHECK_PASSED", flush=True)


if __name__ == "__main__":
    self_check()
