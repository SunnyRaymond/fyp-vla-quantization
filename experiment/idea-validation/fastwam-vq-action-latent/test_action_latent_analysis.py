"""Small assertion self-check; imported and executed only after the PBS guard."""


def self_check():
    from analyze_results import (ACTION_METRICS, case_means, exact_context_index,
                                 paired_summary, select_split_epoch, summarize_action_rows)

    rows = [
        {"case_id": 1, "seed_index": 0, "motor": 1.0},
        {"case_id": 1, "seed_index": 1, "motor": 3.0},
        {"case_id": 2, "seed_index": 0, "motor": 8.0},
        {"case_id": 2, "seed_index": 1, "motor": 10.0},
    ]
    means = case_means(rows, "motor", [1, 2], [0, 1], "self-check")
    assert means == {1: 2.0, 2: 9.0}
    assert (means[1] + means[2]) / 2 == 5.5
    values = {(row['case_id'], row['seed_index']): {metric: row['motor'] for metric in ACTION_METRICS}
              for row in rows}
    table = summarize_action_rows(values, [1, 2], [0, 1])
    assert table['first10motor_rmse']['mean'] == 5.5
    assert table['first10motor_rmse']['case_count'] == 2
    try:
        exact_context_index(rows + [rows[0]], [(1, 0), (1, 1), (2, 0), (2, 1)], "duplicate check")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Duplicate contexts must be rejected")
    try:
        exact_context_index(rows[:-1], [(1, 0), (1, 1), (2, 0), (2, 1)], "missing check")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Missing contexts must be rejected")
    comparison = paired_summary({1: 1.0, 2: 2.0}, {1: 0.5, 2: 2.0}, "before", "after")
    assert comparison["mean_paired_delta"] == -0.25
    assert comparison["improved_cases"] == 1 and comparison["unchanged_cases"] == 1
    selected = select_split_epoch([
        {"split": "selection", "epoch": 1, "case_id": 1},
        {"split": "selection", "epoch": 2, "case_id": 2},
        {"split": "test", "epoch": 2, "case_id": 3},
    ], "selection", 2)
    assert len(selected) == 1 and selected[0]["case_id"] == 2
