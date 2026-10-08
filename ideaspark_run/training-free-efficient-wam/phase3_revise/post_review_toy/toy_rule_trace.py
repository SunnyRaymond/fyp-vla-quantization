from math import sqrt
import json

groups = {"A": ["a0", "a1"], "B": ["b0"], "C": ["c0", "c1", "c2"]}
native_groups = list(groups)
tokens = [t for g in native_groups for t in groups[g]]
queries = ["q0"]
heads = ["h0", "h1"]
alpha = {
    "h0": {"a0": 0.25, "a1": 0.25, "b0": 0.30, "c0": 0.20, "c1": 0.0, "c2": 0.0},
    "h1": {"a0": 0.10, "a1": 0.40, "b0": 0.10, "c0": 0.40, "c1": 0.0, "c2": 0.0},
}
values = {
    "h0": {"a0": (2, 0), "a1": (-1, 0), "b0": (0, 1), "c0": (0, 2), "c1": (1, 0), "c2": (0, 1)},
    "h1": {"a0": (0, 1), "a1": (0, -1), "b0": (2, 0), "c0": (1, 1), "c1": (0, 1), "c2": (1, 0)},
}
# Each head's output-projection slice is identity in this illustrative assignment.
wo = {"h0": ((1, 0), (0, 1)), "h1": ((1, 0), (0, 1))}

def norm2(x):
    return sqrt(sum(v * v for v in x))

def matvec(m, x):
    return tuple(sum(m[i][j] * x[j] for j in range(len(x))) for i in range(len(m)))

def contribution_group_scores():
    r = {}
    for g in native_groups:
        y = [0.0, 0.0]
        for h in heads:
            z = tuple(sum(alpha[h][t] * values[h][t][d] for t in groups[g]) for d in range(2))
            projected = matvec(wo[h], z)
            y = [y[d] + projected[d] for d in range(2)]
        r[g] = norm2(y)  # Q_l has one query in this trace.
    return r

def choose_groups(scores, epsilon):
    if not 0 <= epsilon < 1:
        raise ValueError("epsilon must be in [0, 1)")
    total = sum(scores.values())
    if total == 0:
        return list(native_groups)  # Candidate's stated dense fallback.
    ranked = sorted(native_groups, key=lambda g: (-scores[g], native_groups.index(g)))
    selected = []
    mass = 0.0
    for g in ranked:
        selected.append(g)
        mass += scores[g] / total
        if mass >= 1 - epsilon:
            break
    return selected

def per_token_scores():
    mass, v2, v1, capa = {}, {}, {}, {}
    for t in tokens:
        mass[t] = sum(alpha[h][t] for h in heads)
        v2[t] = sum(alpha[h][t] * norm2(values[h][t]) for h in heads)
        v1[t] = sum(alpha[h][t] * sum(abs(x) for x in values[h][t]) for h in heads)
        y = [0.0, 0.0]
        for h in heads:
            projected = matvec(wo[h], tuple(alpha[h][t] * x for x in values[h][t]))
            y = [y[d] + projected[d] for d in range(2)]
        capa[t] = norm2(y)
    return {"mass": mass, "v_l2": v2, "v_l1": v1, "capa_projected": capa}

def match_token_budget(token_score, budget):
    group_score = {g: sum(token_score[t] for t in groups[g]) for g in native_groups}
    ranked = sorted(native_groups, key=lambda g: (-group_score[g], native_groups.index(g)))
    selected = []
    for g in ranked:
        room = budget - len(selected)
        if room <= 0:
            break
        ordered = sorted(groups[g], key=lambda t: (-token_score[t], groups[g].index(t)))
        selected.extend(ordered[:room])
    return selected

r = contribution_group_scores()
total_r = sum(r.values())
rho = {g: r[g] / total_r for g in native_groups}
epsilon = 0.40
selected_groups = choose_groups(r, epsilon)
query_heads = len(queries) * len(heads)
dense_edges = query_heads * len(tokens)
budget_tokens = sum(len(groups[g]) for g in selected_groups)
budget_edges = query_heads * budget_tokens
explicit_edges = [(q, h, t) for q in queries for h in heads for g in selected_groups for t in groups[g]]
assert len(explicit_edges) == budget_edges
assert selected_groups == ["C", "A"]
assert budget_edges == 10 and dense_edges == 12

baseline_tokens = {}
for name, scores in per_token_scores().items():
    chosen = match_token_budget(scores, budget_tokens)
    baseline_tokens[name] = chosen
    assert len(chosen) == budget_tokens
    assert len(queries) * len(heads) * len(chosen) == budget_edges

assert choose_groups(r, 0.0) == ["C", "A", "B"]
assert choose_groups(r, 0.99) == ["C"]
assert choose_groups({"A": 1.0, "B": 1.0, "C": 1.0}, 0.5) == ["A", "B"]
assert choose_groups({"A": 0.0, "B": 0.0, "C": 0.0}, 0.4) == native_groups
try:
    choose_groups(r, 1.0)
except ValueError:
    invalid_epsilon_rejected = True
else:
    invalid_epsilon_rejected = False
assert invalid_epsilon_rejected

print(json.dumps({
    "illustrative_assignment": {"Q": len(queries), "H": len(heads), "group_sizes": {g: len(groups[g]) for g in native_groups}, "epsilon": epsilon},
    "r_lg": r,
    "rho_lg": rho,
    "selected_groups": selected_groups,
    "candidate_edges": budget_edges,
    "dense_edges": dense_edges,
    "same_budget_score_controls": {k: {"tokens": v, "edges": query_heads * len(v)} for k, v in baseline_tokens.items()},
    "degenerate_probes": {"epsilon_0": choose_groups(r, 0.0), "epsilon_near_1": choose_groups(r, 0.99), "stable_tie": choose_groups({"A": 1.0, "B": 1.0, "C": 1.0}, 0.5), "zero_denominator_fallback": choose_groups({"A": 0.0, "B": 0.0, "C": 0.0}, 0.4), "epsilon_1_rejected": invalid_epsilon_rejected},
    "scope": "stdlib arithmetic trace only; no model, task-success, or latency simulation"
}, ensure_ascii=False, indent=2))

boundary_scores = {"a0": 2.0, "a1": 1.0, "b0": 4.0, "c0": 9.0, "c1": 8.0, "c2": 7.0}
partial_boundary = match_token_budget(boundary_scores, budget_tokens)
assert partial_boundary == ["c0", "c1", "c2", "b0", "a0"]
assert len(partial_boundary) * query_heads == budget_edges
print(json.dumps({"boundary_prefix_probe": {"tokens": partial_boundary, "edges": len(partial_boundary) * query_heads}}, ensure_ascii=False))
