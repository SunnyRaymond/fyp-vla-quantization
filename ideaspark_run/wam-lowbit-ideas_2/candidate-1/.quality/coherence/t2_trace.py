import json
from statistics import mean, pstdev

# Synthetic full-model outputs are fixed inputs to the candidate's rules;
# this script does not instantiate or execute any model.
limits = {"c1": 1.0, "c2": 3.0}
reference = {"a": [0.0, 0.0, 0.0], "b": [2.9, 0.0, 0.0]}
outputs = {
    "A": {"a": [0.2, 0.0, 0.0], "b": [2.7, 0.0, 0.0]},
    "B": {"a": [0.0, 4.0, 0.0], "b": [2.9, 0.0, 0.0]},
    "C": {"a": [0.01, 0.0, 0.0], "b": [3.2, 0.0, 0.0]},
}
cal = [
    {"id": "a_cal", "signal": "a", "state": "c1", "h": 1},
    {"id": "b_cal", "signal": "b", "state": "c2", "h": 2},
]
holdout = [
    {"id": "a_hold", "signal": "a", "state": "c1", "h": 1},
    {"id": "b_hold", "signal": "b", "state": "c2", "h": 2},
]

def controller(raw, state, h):
    lim = limits[state]
    prefix = raw[:h]
    command = [min(max(x, -lim), lim) for x in prefix]
    signature = tuple(x < -lim or x > lim for x in prefix)
    return command, signature

def scales(records):
    by_dim = {}
    for r in records:
        cmd, _ = controller(reference[r["signal"]], r["state"], r["h"])
        for j, x in enumerate(cmd):
            by_dim.setdefault(j, []).append(x)
    return {j: (pstdev(xs) if pstdev(xs) > 0 else 1.0)
            for j, xs in by_dim.items()}

def evaluate(records, name, dim_scales=None):
    if not records:
        return {"error": "empty calibration set: rates and scales undefined"}
    dim_scales = scales(records) if dim_scales is None else dim_scales
    row_sq, flat_sq, row_mismatch, pooled_mismatch, total_branches = [], [], 0, 0, 0
    raw_row_sq = []
    for r in records:
        ref_cmd, ref_sig = controller(reference[r["signal"]], r["state"], r["h"])
        cmd, sig = controller(outputs[name][r["signal"]], r["state"], r["h"])
        errs = [((y - x) / dim_scales[j]) ** 2
                for j, (x, y) in enumerate(zip(ref_cmd, cmd))]
        row_sq.append(sum(errs))
        flat_sq.extend(errs)
        raw_row_sq.append(sum((y - x) ** 2 for x, y in zip(ref_cmd, cmd)))
        row_mismatch += (sig != ref_sig)
        pooled_mismatch += sum(a != b for a, b in zip(sig, ref_sig))
        total_branches += len(ref_sig)
    return {
        "record_branch_rate": row_mismatch / len(records),
        "pooled_branch_rate": pooled_mismatch / total_branches,
        "rowmean_normalized_sq": mean(row_sq),
        "pooled_normalized_sq": mean(flat_sq),
        "raw_rowmean_sq": mean(raw_row_sq),
        "scales": {str(k): v for k, v in dim_scales.items()},
    }

def chunk_mse(records, name):
    return mean(mean((y - x) ** 2 for x, y in zip(reference[r["signal"]], outputs[name][r["signal"]]))
                for r in records)

def coordinate_search(records, branch_key, error_key, start="A"):
    current = start
    history = []
    while True:
        updated = False
        for name in outputs:  # one quantizer coordinate; finite legal-candidate list
            before = evaluate(records, current)
            after = evaluate(records, name)
            before_key = (before[branch_key], before[error_key])
            after_key = (after[branch_key], after[error_key])
            if after_key < before_key:  # strict lexicographic descent only
                history.append({"from": current, "to": name, "before": before_key, "after": after_key})
                current, updated = name, True
        if not updated:
            return current, history

main_scores = {n: evaluate(cal, n) for n in outputs}
chunk_scores = {n: chunk_mse(cal, n) for n in outputs}
selected, search_history = coordinate_search(cal, "record_branch_rate", "rowmean_normalized_sq")
selected_pooled, pooled_history = coordinate_search(cal, "pooled_branch_rate", "pooled_normalized_sq")
naive = min(chunk_scores, key=chunk_scores.get)

# Negative control: one random permutation draw (swap) of controller state and prefix across calibration rows.
shuffled = [dict(r, state=cal[1-i]["state"], h=cal[1-i]["h"]) for i, r in enumerate(cal)]
shuffled_scores = {n: evaluate(shuffled, n) for n in outputs}
selected_shuffled, shuffled_history = coordinate_search(shuffled, "record_branch_rate", "rowmean_normalized_sq")
selected_shuffled_pooled, shuffled_pooled_history = coordinate_search(shuffled, "pooled_branch_rate", "pooled_normalized_sq")
holdout_before = evaluate(holdout, selected)
holdout_after = evaluate(holdout, selected_shuffled, scales(shuffled))

one = evaluate(cal[:1], "B")
identical = [{"id": "a1", "signal": "a", "state": "c1", "h": 1},
             {"id": "a2", "signal": "a", "state": "c1", "h": 1}]
identical_probe = evaluate(identical, "A")
tie_command, tie_signature = controller([1.0, 0.0, 0.0], "c1", 1)
empty_probe = evaluate([], "A")

result = {
    "illustrative_values": {"clip_limits": limits, "prefix_lengths": [1, 2], "configs": list(outputs)},
    "reference_scales": scales(cal),
    "candidate_scores": main_scores,
    "full_chunk_mse": chunk_scores,
    "coordinate_search_history": search_history,
    "selected_candidate": selected,
    "coordinate_search_history_under_pooled_reading": pooled_history,
    "selected_under_pooled_reading": selected_pooled,
    "independent_full_chunk_naive": naive,
    "naive_executed_score": main_scores[naive],
    "permuted_calibration_scores": shuffled_scores,
    "permuted_search_history": shuffled_history,
    "selected_after_permutation": selected_shuffled,
    "permuted_search_history_under_pooled_reading": shuffled_pooled_history,
    "selected_after_permutation_under_pooled_reading": selected_shuffled_pooled,
    "negative_control_holdout_before": holdout_before,
    "negative_control_holdout_after": holdout_after,
    "single_record_probe": one,
    "all_identical_reference_probe": identical_probe,
    "exact_threshold_probe": {"command": tie_command, "branch_signature": tie_signature},
    "empty_probe": empty_probe,
    "inferences": {
        "strict_descent_tie": "equal-score candidates are rejected; the finite one-coordinate configuration set cannot cycle under strict lexicographic decrease",
        "instance_scope": "all config outputs and clip limits are illustrative synthetic inputs, not learned WAM outputs or evidence of task benefit"
    }
}
assert selected == selected_pooled == "B"
assert naive == "A"
assert selected_shuffled == selected_shuffled_pooled == "C"
assert holdout_before["raw_rowmean_sq"] == 0.0
assert holdout_after["raw_rowmean_sq"] > holdout_before["raw_rowmean_sq"]
assert coordinate_search(cal, "record_branch_rate", "rowmean_normalized_sq", start="B")[1] == []
print(json.dumps(result, ensure_ascii=False, indent=2))
