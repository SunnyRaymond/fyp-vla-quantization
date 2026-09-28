"""Episode-weighted diagnostics used by the OTC phase-1 report.

Score files contain several states per episode. Ranking and uncertainty helpers
first average states inside an episode and only then average episodes, so states
cannot masquerade as independent replicates. Missing measurements are reported
as ``incomplete`` or ``unavailable`` instead of being filled with zero.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


def midranks(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(a) != len(b) or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return None
    ra, rb = midranks(a), midranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def top_b(scores: Mapping[str, float], site_order: Sequence[str], b: int = 2) -> list[str]:
    order = {site: index for index, site in enumerate(site_order)}
    return sorted(site_order, key=lambda site: (-float(scores[site]), order[site]))[:b]


def row_episode_key(row: Mapping[str, Any]) -> tuple[int, int]:
    return int(row["task_id"]), int(row["episode"])


def valid_rows(rows: Iterable[Mapping[str, Any]], field: str | None = None) -> list[Mapping[str, Any]]:
    """Rows usable for one numeric calculation; input rows remain retained by the caller."""
    output: list[Mapping[str, Any]] = []
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        if field is not None:
            try:
                value = float(row.get(field))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value):
                continue
        output.append(row)
    return output


def complete_episode_scores(
    rows: Iterable[Mapping[str, Any]], sites: Sequence[str], field: str,
    *, expected_states: int | None = None,
) -> dict[tuple[int, int], dict[str, float]]:
    """Return state-mean scores for episodes complete at every site.

    ``expected_states`` is used by the strict phase-1 report so a site with
    only one surviving value cannot make a partially measured episode look
    complete.  It remains optional for the small synthetic weighting check.
    """
    values: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    site_set = set(sites)
    for row in rows:
        if row.get("status", "ok") != "ok" or row.get("site_id") not in site_set:
            continue
        try:
            value = float(row.get(field))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values[row_episode_key(row)][str(row["site_id"])].append(value)
    return {
        key: {site: float(np.mean(by_site[site])) for site in sites}
        for key, by_site in values.items()
        if all(by_site.get(site) and (expected_states is None or len(by_site[site]) == expected_states) for site in sites)
    }


def aggregate_episode_scores(
    episode_scores: Mapping[tuple[int, int], Mapping[str, float]],
    sites: Sequence[str],
    episodes: Sequence[tuple[int, int]] | None = None,
) -> dict[str, float] | None:
    selected = list(episodes if episodes is not None else episode_scores)
    if not selected or any(key not in episode_scores for key in selected):
        return None
    if any(any(site not in episode_scores[key] for site in sites) for key in selected):
        return None
    return {site: float(np.mean([episode_scores[key][site] for key in selected])) for site in sites}


def episode_mean(
    rows: Iterable[Mapping[str, Any]],
    sites: Sequence[str],
    field: str,
    subset: Callable[[Mapping[str, Any]], bool] | None = None,
) -> dict[str, float | None]:
    """Backward-compatible episode-weighted site means."""
    filtered = [row for row in rows if subset is None or subset(row)]
    table = complete_episode_scores(filtered, sites, field)
    return {site: None if not table else float(np.mean([value[site] for value in table.values()])) for site in sites}


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"p05": None, "p50": None, "p95": None}
    q = np.percentile(np.asarray(values, dtype=float), [5, 50, 95])
    return {"p05": float(q[0]), "p50": float(q[1]), "p95": float(q[2])}


def bootstrap_top2(
    episode_scores: Mapping[tuple[int, int], Mapping[str, float]],
    sites: Sequence[str], *, seed: int, iterations: int = 2000, b: int = 2
) -> dict[str, Any]:
    """Descriptive bootstrap over complete CAL episodes, never individual states."""
    keys = sorted(episode_scores)
    if not keys:
        return {"status": "incomplete", "n_episodes": 0, "n_bootstrap": 0}
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    rng = np.random.default_rng(seed)
    counts, pair_counts = Counter(), Counter()
    score_samples: dict[str, list[float]] = {site: [] for site in sites}
    task_groups = [[key for key in keys if key[0] == task]
                   for task in sorted({key[0] for key in keys})]
    for _ in range(int(iterations)):
        picked = [group[index] for group in task_groups
                  for index in rng.integers(0, len(group), size=len(group))]
        means = {site: float(np.mean([episode_scores[key][site] for key in picked])) for site in sites}
        selected = tuple(top_b(means, sites, b=b))
        counts.update(selected)
        pair_counts[tuple(sorted(selected))] += 1
        for site in sites:
            score_samples[site].append(means[site])
    full = aggregate_episode_scores(episode_scores, sites)
    return {
        "status": "ok",
        "unit": "complete episode sampled with replacement within each fixed task",
        "n_episodes": len(keys),
        "episodes": [list(key) for key in keys],
        "n_bootstrap": int(iterations),
        "seed": int(seed),
        "top2_inclusion_probability": {site: float(counts[site] / iterations) for site in sites},
        "score_intervals_p05_p50_p95": {site: _quantiles(score_samples[site]) for site in sites},
        "top2_pair_probability": {f"{pair[0]}|{pair[1]}": float(count / iterations) for pair, count in sorted(pair_counts.items())},
        "full_cal_top2": top_b(full, sites, b=b) if full is not None else [],
        "interpretation": "descriptive small-sample stability; no significance claim",
    }


def leave_one_episode_out_top2(
    episode_scores: Mapping[tuple[int, int], Mapping[str, float]], sites: Sequence[str], *, b: int = 2
) -> dict[str, Any]:
    """Top-2 rankings after holding out each complete CAL episode."""
    keys = sorted(episode_scores)
    full = aggregate_episode_scores(episode_scores, sites)
    if not keys or full is None:
        return {"status": "incomplete", "n_episodes": len(keys), "folds": []}
    full_top = top_b(full, sites, b=b)
    folds, inclusion = [], Counter()
    for held_out in keys:
        kept = [key for key in keys if key != held_out]
        scores = aggregate_episode_scores(episode_scores, sites, kept)
        if scores is None:
            folds.append({"held_out_episode": list(held_out), "status": "incomplete"})
            continue
        selected = top_b(scores, sites, b=b)
        inclusion.update(selected)
        folds.append({"held_out_episode": list(held_out), "status": "ok", "top2": selected,
                      "overlap_with_full_top2": len(set(selected) & set(full_top)), "scores": scores})
    n_ok = sum(fold.get("status") == "ok" for fold in folds)
    return {"status": "ok" if n_ok == len(keys) else "incomplete", "n_episodes": len(keys),
            "n_folds_ok": n_ok, "full_cal_top2": full_top, "folds": folds,
            "top2_inclusion_count": {site: int(inclusion[site]) for site in sites},
            "interpretation": "leave-one-complete-episode-out descriptive stability; no significance claim"}


def ranking_diagnostic(cal_rows, check_rows, sites):
    """Legacy field-based ranking report retained for existing consumers."""
    fields = ["local_mse", "local_nmse", "d_a", "d_o", "C"]
    output = {}
    for field in fields:
        cal_table = complete_episode_scores(cal_rows, sites, field, expected_states=4)
        check_table = complete_episode_scores(check_rows, sites, field, expected_states=4)
        cal, check = aggregate_episode_scores(cal_table, sites), aggregate_episode_scores(check_table, sites)
        if cal is None or check is None:
            output[field] = {"status": "incomplete"}
            continue
        output[field] = {"cal_scores": cal, "check_scores": check, "cal_top2": top_b(cal, sites),
                         "check_top2": top_b(check, sites),
                         "cal_check_spearman": spearman([cal[s] for s in sites], [check[s] for s in sites])}
    return output


def selected_episode_metric(
    rows: Iterable[Mapping[str, Any]], selected_sites: Sequence[str],
    field: str | Callable[[Mapping[str, Any]], float | None],
) -> dict[str, Any]:
    """Episode-weighted metric for a selected site set; missing values stay unavailable."""
    site_set = set(selected_sites)
    per_episode: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("status", "ok") != "ok" or row.get("site_id") not in site_set:
            continue
        value = row.get(field) if isinstance(field, str) else field(row)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            per_episode[row_episode_key(row)].append(value)
    episode_values = {key: float(np.mean(values)) for key, values in per_episode.items() if values}
    if not episode_values:
        return {"status": "unavailable", "value": None, "n_episodes": 0, "episode_values": {}}
    return {"status": "ok", "value": float(np.mean(list(episode_values.values()))),
            "n_episodes": len(episode_values),
            "episode_values": {f"{key[0]}/{key[1]}": value for key, value in sorted(episode_values.items())}}


def _contact_pairs(value: Any) -> frozenset[tuple[str, str]] | None:
    if not isinstance(value, Mapping) or not bool(value.get("available")):
        return None
    pairs = value.get("pairs")
    if not isinstance(pairs, (list, tuple)):
        return None
    normalized = []
    for pair in pairs:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            normalized.append(tuple(sorted((str(pair[0]), str(pair[1])))))
    return frozenset(normalized)


def contact_difference(row: Mapping[str, Any]) -> float | None:
    """Binary reference/quantized next-contact difference diagnostic."""
    reference_raw, quantized_raw = row.get("ref_next_contacts"), row.get("quant_next_contacts")
    reference, quantized = _contact_pairs(reference_raw), _contact_pairs(quantized_raw)
    if reference is not None and quantized is not None:
        return float(reference != quantized)
    # The runner emits ``contact_transition_quant=False`` when contact capture
    # is unavailable. Do not reinterpret that sentinel as a measured match.
    if isinstance(reference_raw, Mapping) or isinstance(quantized_raw, Mapping):
        return None
    fallback = row.get("contact_transition_quant")
    if isinstance(fallback, (bool, int, float)):
        return float(bool(fallback))
    return None


def independent_check_diagnostics(check_rows: Iterable[Mapping[str, Any]], selected_sites: Sequence[str]) -> dict[str, Any]:
    """Independent CHECK object/contact diagnostics for selected sites."""
    object_result = selected_episode_metric(check_rows, selected_sites, "max_object_position_delta_m")
    contact_result = selected_episode_metric(check_rows, selected_sites, contact_difference)
    if object_result["status"] != "ok":
        object_result.update(status="unavailable", reason="no finite object position values were recorded")
    if contact_result["status"] != "ok":
        contact_result.update(status="unavailable", reason="reference/quantized contact pairs were unavailable")
    return {"max_object_position_delta_m": object_result, "contact_difference_rate": contact_result,
            "selected_sites": list(selected_sites),
            "unit": "CHECK episodes, episode-weighted selected-site/state means"}


def _summarize_control(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"status": "unavailable", "n": 0, "quantiles_p05_p50_p95": _quantiles([])}
    return {"status": "ok", "n": len(values), "mean": float(np.mean(values)),
            "quantiles_p05_p50_p95": _quantiles(values)}


def permutation_reassignment_control(
    cal_scores: Mapping[str, float], check_rows: Iterable[Mapping[str, Any]], sites: Sequence[str], *,
    seed: int, iterations: int = 2000, b: int = 2,
) -> dict[str, Any]:
    """Fixed-seed null control by reassigning CAL scores to fixed site labels.

    CHECK measurements remain untouched. Each draw permutes the CAL score values
    across site IDs, selects top-2 labels, and evaluates independent CHECK
    object/contact diagnostics. This is descriptive assignment control, not a
    causal or inferential p-value.
    """
    if any(site not in cal_scores for site in sites):
        return {"status": "incomplete", "seed": int(seed), "n_permutations": 0}
    if iterations <= 0:
        raise ValueError("permutation iterations must be positive")
    check_rows = list(check_rows)
    rng = np.random.default_rng(seed)
    score_values = np.asarray([float(cal_scores[site]) for site in sites], dtype=float)
    observed_sites = top_b(cal_scores, sites, b=b)
    observed = independent_check_diagnostics(check_rows, observed_sites)
    object_null: list[float] = []
    contact_null: list[float] = []
    pair_counts = Counter()
    for _ in range(int(iterations)):
        reassigned = dict(zip(sites, rng.permutation(score_values)))
        selected = top_b(reassigned, sites, b=b)
        pair_counts[tuple(sorted(selected))] += 1
        diagnostic = independent_check_diagnostics(check_rows, selected)
        object_value = diagnostic["max_object_position_delta_m"].get("value")
        contact_value = diagnostic["contact_difference_rate"].get("value")
        if object_value is not None:
            object_null.append(float(object_value))
        if contact_value is not None:
            contact_null.append(float(contact_value))

    def percentile(value: Any, values: Sequence[float]) -> float | None:
        if value is None or not values:
            return None
        return float((np.sum(np.asarray(values) <= float(value)) + 1) / (len(values) + 1))

    return {
        "status": "ok",
        "control": "CAL score-value permutation/reassignment across fixed site labels",
        "check_measurements_untouched": True,
        "seed": int(seed), "n_permutations": int(iterations), "observed_top2": observed_sites,
        "observed_check_diagnostics": observed,
        "null_diagnostics": {"max_object_position_delta_m": _summarize_control(object_null),
                              "contact_difference_rate": _summarize_control(contact_null)},
        "observed_percentile_in_control": {
            "max_object_position_delta_m": percentile(observed["max_object_position_delta_m"].get("value"), object_null),
            "contact_difference_rate": percentile(observed["contact_difference_rate"].get("value"), contact_null)},
        "top2_pair_probability": {f"{pair[0]}|{pair[1]}": float(count / iterations) for pair, count in sorted(pair_counts.items())},
        "interpretation": "descriptive assignment control; not a p-value and not a joint-protection test",
    }


def _complete_episode_callable(rows: Iterable[Mapping[str, Any]], sites: Sequence[str], value_fn: Callable[[Mapping[str, Any]], float], *, expected_states: int | None = None) -> dict[tuple[int, int], dict[str, float]]:
    values: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    site_set = set(sites)
    for row in rows:
        if row.get("site_id") not in site_set:
            continue
        try:
            value = float(value_fn(row))
        except (TypeError, ValueError, KeyError):
            continue
        if np.isfinite(value):
            values[row_episode_key(row)][str(row["site_id"])].append(value)
    return {key: {site: float(np.mean(by_site[site])) for site in sites} for key, by_site in values.items()
            if all(by_site.get(site) and (expected_states is None or len(by_site[site]) == expected_states) for site in sites)}


def _ranking_record(cal, check, cal_table, check_table, sites) -> dict[str, Any]:
    if cal is None or check is None:
        return {"status": "incomplete", "cal_complete_episodes": [list(key) for key in sorted(cal_table)],
                "check_complete_episodes": [list(key) for key in sorted(check_table)]}
    return {"status": "ok", "cal_scores": dict(cal), "check_scores": dict(check),
            "cal_top2": top_b(cal, sites), "check_top2": top_b(check, sites),
            "cal_check_spearman": spearman([cal[s] for s in sites], [check[s] for s in sites]),
            "cal_complete_episodes": [list(key) for key in sorted(cal_table)],
            "check_complete_episodes": [list(key) for key in sorted(check_table)],
            "cal_episode_scores": {f"{key[0]}/{key[1]}": dict(value) for key, value in sorted(cal_table.items())},
            "check_episode_scores": {f"{key[0]}/{key[1]}": dict(value) for key, value in sorted(check_table.items())}}


def rank_methods(
    cal_rows: Iterable[Mapping[str, Any]],
    check_rows: Iterable[Mapping[str, Any]],
    sites: Sequence[str],
    *,
    lambda_values: Sequence[float] = (0.5, 1.0, 2.0),
    lambda_o_values: Sequence[float] = (1.0,),
) -> dict[str, dict[str, Any]]:
    """CAL-only rankings for local raw/normalized, action, observation, and OTC scores.

    Phase 1 preregisters the three action weights with ``lambda_o=1``.  The
    second parameter is explicit so an accidental 3x3 grid cannot enter the
    report without a deliberate caller change.
    """
    cal_rows, check_rows = list(cal_rows), list(check_rows)
    fields = {"local_raw": "local_mse", "local_normalized": "local_nmse", "action_only": "d_a", "observation_only": "d_o", "first_action_only": "first_action_mse"}
    output: dict[str, dict[str, Any]] = {}
    for name, field in fields.items():
        cal_table = complete_episode_scores(cal_rows, sites, field, expected_states=4)
        check_table = complete_episode_scores(check_rows, sites, field, expected_states=4)
        output[name] = _ranking_record(aggregate_episode_scores(cal_table, sites), aggregate_episode_scores(check_table, sites), cal_table, check_table, sites)
    for lambda_a in lambda_values:
        for lambda_o in lambda_o_values:
            name = f"otc_lambda_a{float(lambda_a):g}_lambda_o{float(lambda_o):g}"
            fn = lambda row, la=float(lambda_a), lo=float(lambda_o): la * float(row["d_a"]) + lo * float(row["d_o"])
            cal_table = _complete_episode_callable(valid_rows(cal_rows), sites, fn, expected_states=4)
            check_table = _complete_episode_callable(valid_rows(check_rows), sites, fn, expected_states=4)
            output[name] = _ranking_record(aggregate_episode_scores(cal_table, sites), aggregate_episode_scores(check_table, sites), cal_table, check_table, sites)
            output[name]["lambda_a"], output[name]["lambda_o"] = float(lambda_a), float(lambda_o)
    # Keep the preregistered lambda=(1,1) OTC name alongside the full
    # sensitivity grid, so downstream reports can retain the base comparator.
    if "otc_lambda_a1_lambda_o1" in output:
        output["otc"] = dict(output["otc_lambda_a1_lambda_o1"])
        output["otc"]["lambda_a"], output["otc"]["lambda_o"] = 1.0, 1.0
    return output


def method_episode_scores(rows: Iterable[Mapping[str, Any]], sites: Sequence[str], method: str) -> dict[tuple[int, int], dict[str, float]]:
    """Return complete-episode scores for one name emitted by :func:`rank_methods`."""
    rows = list(rows)
    fields = {"local_raw": "local_mse", "local_normalized": "local_nmse",
              "action_only": "d_a", "observation_only": "d_o", "first_action_only": "first_action_mse"}
    if method == "otc":
        method = "otc_lambda_a1_lambda_o1"
    if method in fields:
        return complete_episode_scores(rows, sites, fields[method], expected_states=4)
    prefix = "otc_lambda_a"
    if not method.startswith(prefix) or "_lambda_o" not in method:
        raise ValueError(f"unknown ranking method: {method}")
    tail = method[len(prefix):]
    lambda_a_text, lambda_o_text = tail.split("_lambda_o", 1)
    lambda_a, lambda_o = float(lambda_a_text), float(lambda_o_text)
    value_fn = lambda row: lambda_a * float(row["d_a"]) + lambda_o * float(row["d_o"])
    return _complete_episode_callable(valid_rows(rows), sites, value_fn, expected_states=4)


def synthetic_self_check(seed: int = 20260909) -> dict[str, Any]:
    """Deterministic CPU check of episode weighting, controls, and unavailable values.

    Synthetic values do not validate MuJoCo restore, IDM inference, quantization,
    or physical contact semantics; they only test analysis bookkeeping.
    """
    sites = ["s0", "s1", "s2", "s3"]
    rows: list[dict[str, Any]] = []
    for episode, n_states in enumerate((1, 2, 3, 4)):
        for state in range(n_states):
            for index, site in enumerate(sites):
                base = float(index + (episode == 3) * (3 - index))
                rows.append({"status": "ok", "task_id": 0, "episode": episode, "state_index": state, "site_id": site,
                             "score": base, "d_a": base, "d_o": base / 2,
                             "max_object_position_delta_m": None,
                             "ref_next_contacts": {"available": True, "pairs": []},
                             "quant_next_contacts": {"available": True, "pairs": [["a", "b"]]} if site == "s0" else {"available": True, "pairs": []}})
    table = complete_episode_scores(rows, sites, "score")
    aggregate = aggregate_episode_scores(table, sites)
    # s0 has episode means [0, 0, 0, 3], hence 0.75 when episodes are weighted equally.
    if aggregate is None or abs(aggregate["s0"] - 0.75) > 1e-12:
        raise AssertionError("synthetic episode weighting check failed")
    first = permutation_reassignment_control(aggregate, rows, sites, seed=seed, iterations=64)
    second = permutation_reassignment_control(aggregate, rows, sites, seed=seed, iterations=64)
    if first != second:
        raise AssertionError("synthetic permutation reproducibility check failed")
    diagnostics = independent_check_diagnostics(rows, ["s0", "s1"])
    if diagnostics["max_object_position_delta_m"]["status"] != "unavailable":
        raise AssertionError("synthetic unavailable object diagnostic check failed")
    return {"status": "passed", "seed": int(seed), "checks": [
        "complete-episode weighting differs from frame weighting",
        "fixed-seed permutation/reassignment is reproducible",
        "missing object values remain explicitly unavailable"],
        "limitations": ["synthetic rows do not exercise MuJoCo state restore or IDM inference",
                        "synthetic contact pairs only test bookkeeping, not physical contact semantics"]}
