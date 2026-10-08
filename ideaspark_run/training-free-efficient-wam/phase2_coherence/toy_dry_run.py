"""Stdlib-only illustrative execution of the candidate's score-and-mask rules."""
import itertools
import json
import math


GROUPS = ["A", "B", "C"]
EPSILON = 0.25  # illustrative admissible value; candidate range is [0, 1)
ALPHA = {"h1": {"A": 0.5, "B": 0.25, "C": 0.25},
         "h2": {"A": 0.5, "B": 0.25, "C": 0.25}}
# Synthetic values make alpha * V equal the listed per-head/group vectors.
VALUES = {
    "h1": {"A": [2.0, 0.0], "B": [0.0, 8.0], "C": [4.0, 0.0]},
    "h2": {"A": [4.0, 0.0], "B": [0.0, 2.0], "C": [-2.0, 0.0]},
}


def add(a, b):
    return [a[0] + b[0], a[1] + b[1]]


def scale(x, a):
    return [x[0] * a, x[1] * a]


def dense_group_vectors(groups, alpha, values):
    # The output-projection head slices are illustrative 2x2 identity maps.
    return {
        g: add(*(scale(values[h][g], alpha[h][g]) for h in alpha))
        for g in groups
    }


def contribution_scores(groups, alpha, values):
    vectors = dense_group_vectors(groups, alpha, values)
    return {g: math.hypot(*vectors[g]) for g in groups}, vectors


def choose(scores, epsilon, source_order):
    total = sum(scores.values())
    if total == 0:
        return list(source_order)
    ranked = sorted(source_order, key=lambda g: (-scores[g], source_order.index(g)))
    keep, cumulative = [], 0.0
    for g in ranked:
        keep.append(g)
        cumulative += scores[g] / total
        if cumulative >= 1.0 - epsilon:
            break
    return keep


def masked_attention_output(groups, selected, logits_by_head, values_by_head):
    # Sparse attention renormalizes native softmax over the retained keys.
    output = [0.0, 0.0]
    for head in logits_by_head:
        weights = [math.exp(logits_by_head[head][g]) for g in selected]
        z = sum(weights)
        head_out = [0.0, 0.0]
        for g, weight in zip(selected, weights):
            head_out = add(head_out, scale(values_by_head[head][g], weight / z))
        output = add(output, head_out)
    return output


def mask_from_scores(scores, epsilon, source_order):
    return choose(scores, epsilon, source_order)


LOGITS = {h: {g: math.log(ALPHA[h][g]) for g in GROUPS} for h in ALPHA}
scores, group_vectors = contribution_scores(GROUPS, ALPHA, VALUES)
total = sum(scores.values())
rho = {g: scores[g] / total for g in GROUPS}
ranked = mask_from_scores(scores, EPSILON, GROUPS)

# Source-group permutation swaps B/C scores while preserving mask cardinality.
permuted_scores = {"A": scores["A"], "B": scores["C"], "C": scores["B"]}
permuted = mask_from_scores(permuted_scores, EPSILON, GROUPS)

dense_output = masked_attention_output(GROUPS, GROUPS, LOGITS, VALUES)
ranked_output = masked_attention_output(GROUPS, ranked, LOGITS, VALUES)
permuted_output = masked_attention_output(GROUPS, permuted, LOGITS, VALUES)

# Independently constructed naive: keep the first two source groups at the same
# two-edge budget, without computing contribution scores.
constructed_naive = GROUPS[:len(ranked)]
constructed_naive_output = masked_attention_output(
    GROUPS, constructed_naive, LOGITS, VALUES)

# Candidate-declared naive: average dense attention mass, with stable source ties.
mass_scores = {g: sum(ALPHA[h][g] for h in ALPHA) / len(ALPHA) for g in GROUPS}
declared_naive = mask_from_scores(mass_scores, EPSILON, GROUPS)
declared_naive_output = masked_attention_output(
    GROUPS, declared_naive, LOGITS, VALUES)

# All-identical per-head values: pruning changes the graph but not this output.
IDENTICAL_VALUES = {
    "h1": {g: [2.0, 0.0] for g in GROUPS},
    "h2": {g: [0.0, 1.0] for g in GROUPS},
}
identical_scores, _ = contribution_scores(GROUPS, ALPHA, IDENTICAL_VALUES)
identical_mask = mask_from_scores(identical_scores, EPSILON, GROUPS)
identical_dense = masked_attention_output(
    GROUPS, GROUPS, LOGITS, IDENTICAL_VALUES)
identical_sparse = masked_attention_output(
    GROUPS, identical_mask, LOGITS, IDENTICAL_VALUES)

# Ties and degenerate branches required by the trace prompt.
tie_mask = mask_from_scores({g: 1.0 for g in GROUPS}, EPSILON, GROUPS)
single_group_mask = mask_from_scores({"A": 2.0}, EPSILON, ["A"])
zero_score_mask = mask_from_scores({g: 0.0 for g in GROUPS}, EPSILON, GROUPS)
epsilon_zero_mask = mask_from_scores(scores, 0.0, GROUPS)

# Unequal source-group sizes expose whether a distinct permuted mask can match
# the same edge count; the score/mask rule itself is unchanged.
sizes = {"A": 1, "B": 2, "C": 3}
permutation_costs = []
for assignment in itertools.permutations([scores[g] for g in GROUPS]):
    assigned = dict(zip(GROUPS, assignment))
    mask = mask_from_scores(assigned, EPSILON, GROUPS)
    permutation_costs.append({"mask": "".join(g for g in GROUPS if g in mask),
                              "edges": sum(sizes[g] for g in mask)})
base_edge_cost = sum(sizes[g] for g in ranked)
distinct_same_cost = sorted({p["mask"] for p in permutation_costs
                             if p["edges"] == base_edge_cost
                             and p["mask"] != "".join(g for g in GROUPS if g in ranked)})

print(json.dumps({
    "status": "illustrative synthetic rule execution only",
    "weights_loaded": False,
    "gpu_used": False,
    "illustrative_values": {
        "epsilon": EPSILON,
        "groups": GROUPS,
        "one_query_one_layer_two_heads": True,
        "attention_mass_per_head": ALPHA["h1"],
        "synthetic_head_values": VALUES,
        "output_projection_head_slices": "identity matrices",
        "source_group_token_counts": {g: 1 for g in GROUPS},
    },
    "group_projected_vectors": group_vectors,
    "r": scores,
    "sum_r": total,
    "rho": rho,
    "target_mass": 1.0 - EPSILON,
    "ranked_mask": ranked,
    "ranked_edges": len(ranked),
    "dense_attention_output": dense_output,
    "ranked_sparse_output": ranked_output,
    "ranked_vs_dense_l2": math.dist(ranked_output, dense_output),
    "permuted_scores": permuted_scores,
    "permuted_mask": permuted,
    "permuted_edges": len(permuted),
    "permuted_sparse_output": permuted_output,
    "control_moved_synthetic_attention_output": permuted_output != ranked_output,
    "constructed_naive": {
        "mask": constructed_naive,
        "output": constructed_naive_output,
        "bit_identical_to_mechanism_output": constructed_naive_output == ranked_output,
    },
    "candidate_declared_attention_mass_naive": {
        "mass_scores": mass_scores,
        "mask": declared_naive,
        "output": declared_naive_output,
        "bit_identical_to_mechanism_output": declared_naive_output == ranked_output,
    },
    "all_identical_value_probe": {
        "r": identical_scores,
        "mask": identical_mask,
        "edges_retained": len(identical_mask),
        "dense_output": identical_dense,
        "sparse_output": identical_sparse,
        "numeric_output_changed": identical_dense != identical_sparse,
    },
    "degenerate_probes": {
        "all_score_ties_epsilon_0_25_mask": tie_mask,
        "single_positive_group_mask": single_group_mask,
        "all_zero_r_fallback_mask": zero_score_mask,
        "epsilon_zero_mask": epsilon_zero_mask,
    },
    "unequal_group_size_control_probe": {
        "token_counts": sizes,
        "ranked_mask": ranked,
        "ranked_edge_cost": base_edge_cost,
        "permuted_masks_and_costs": permutation_costs,
        "distinct_same_cost_permuted_masks": distinct_same_cost,
    },
    "scope": "Values are illustrative only; no learned component, checkpoint, environment, task-success outcome, or latency was simulated.",
}, ensure_ascii=False, sort_keys=True, indent=2))
