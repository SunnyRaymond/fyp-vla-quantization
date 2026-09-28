"""Aggregate the eight compact OTC phase-1 score JSON files.

The report ranks sites using CAL episodes only, then evaluates those fixed
choices on independent CHECK episodes.  State rows are retained for audit, but
all ranking, bootstrap, and selected-site summaries are complete-episode
weighted.  Bootstrap and permutation outputs are descriptive controls for a
small four-episode CAL/four-episode CHECK design; they are not significance
tests or evidence of joint quantization protection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from analysis_core import (
    bootstrap_top2,
    complete_episode_scores,
    independent_check_diagnostics,
    leave_one_episode_out_top2,
    method_episode_scores,
    permutation_reassignment_control,
    rank_methods,
    selected_episode_metric,
    synthetic_self_check,
    spearman,
)


TASKS = (0, 1)
EPISODES = (0, 1, 2, 3)
CAL_EPISODES = (0, 1)
CHECK_EPISODES = (2, 3)
EXPECTED_EPISODES = {(task, episode) for task in TASKS for episode in EPISODES}
EXPECTED_SCORE_FILES = 8
EXPECTED_ROWS = 256
SITES_PER_STATE = 8
DEFAULT_SEED = 20260909
DEFAULT_BOOTSTRAP_ITERS = 2000
DEFAULT_PERMUTATION_ITERS = 2000


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")


def _input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("score_task*_episode*.json"))


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json_default)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _l2_or_none(lhs: Any, rhs: Any) -> float | None:
    if lhs is None or rhs is None:
        return None
    try:
        left, right = np.asarray(lhs, dtype=float), np.asarray(rhs, dtype=float)
        if left.shape != right.shape or not (np.isfinite(left).all() and np.isfinite(right).all()):
            return None
        return float(np.linalg.norm(left - right))
    except (TypeError, ValueError):
        return None


def _load_rows(input_path: Path, sites_hint: Iterable[str] | None = None) -> dict[str, Any]:
    files = _input_files(input_path)
    if len(files) != EXPECTED_SCORE_FILES:
        raise ValueError(f"expected exactly {EXPECTED_SCORE_FILES} score JSON files, found {len(files)}")
    seen_episodes: set[tuple[int, int]] = set()
    rows: list[dict[str, Any]] = []
    registries: list[Any] = []
    normalizers: list[tuple[float, float]] = []
    file_summaries: list[dict[str, Any]] = []
    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        key = (int(data["task_id"]), int(data["episode"]))
        if key in seen_episodes:
            raise ValueError(("duplicate episode result", key))
        seen_episodes.add(key)
        if key not in EXPECTED_EPISODES:
            raise ValueError(("unexpected task/episode", key))
        split = str(data.get("split", "CAL" if key[1] in CAL_EPISODES else "CHECK"))
        expected_split = "CAL" if key[1] in CAL_EPISODES else "CHECK"
        if split != expected_split:
            raise ValueError(("episode split mismatch", key, split, expected_split))
        registries.append(data.get("registry"))
        file_registry = data.get("registry") or {}
        file_sites = [str(site["site_id"]) for site in file_registry.get("sites", [])]
        if len(file_sites) != SITES_PER_STATE or len(set(file_sites)) != SITES_PER_STATE:
            raise ValueError(("file registry must contain exactly eight unique sites", key))
        file_rows = data.get("rows", [])
        if len(file_rows) != 4 * SITES_PER_STATE:
            raise ValueError(("each score file must contain exactly four states x eight sites", key, len(file_rows)))
        state_sites: dict[int, list[str]] = {}
        for source_row in file_rows:
            if "status" not in source_row:
                raise ValueError(("score row lacks explicit status", key, source_row.get("site_id")))
            if "task_id" in source_row and int(source_row["task_id"]) != key[0]:
                raise ValueError(("row task_id disagrees with score file", key, source_row.get("task_id")))
            if "episode" in source_row and int(source_row["episode"]) != key[1]:
                raise ValueError(("row episode disagrees with score file", key, source_row.get("episode")))
            if "split" in source_row and str(source_row["split"]) != expected_split:
                raise ValueError(("row split disagrees with score file", key, source_row.get("split")))
            if "state_index" not in source_row or int(source_row["state_index"]) not in range(4):
                raise ValueError(("row state_index must be one of 0..3", key, source_row.get("state_index")))
            site_id = str(source_row.get("site_id"))
            if site_id not in file_sites:
                raise ValueError(("row site_id absent from file registry", key, site_id))
            state_sites.setdefault(int(source_row["state_index"]), []).append(site_id)
        if set(state_sites) != set(range(4)) or any(sorted(values) != sorted(file_sites) for values in state_sites.values()):
            raise ValueError(("each file state must contain exactly the eight registry sites", key, state_sites))
        normal = data.get("normalization", {})
        sigma_a2, sigma_o = _finite_float(normal.get("sigma_a2")), _finite_float(normal.get("sigma_o"))
        if sigma_a2 is None or sigma_o is None:
            raise ValueError(("missing/nonfinite normalizer", key))
        normalizers.append((sigma_a2, sigma_o))
        file_summaries.append({"path": str(file), "task_id": key[0], "episode": key[1],
                               "split": split, "status": data.get("status"),
                               "n_rows": len(data.get("rows", []))})
        for source_row in file_rows:
            row = dict(source_row)
            row["task_id"], row["episode"], row["split"] = key[0], key[1], split
            row["local_mse"] = row.get("local_hook_output_mse")
            row["local_nmse"] = row.get("local_hook_output_mse_normalized")
            row["C"] = row.get("C_r")
            row["contact_present"] = row.get("contact_present_before_first_action")
            row["qpos_l2"] = _l2_or_none(row.get("quant_next_qpos"), row.get("ref_next_qpos"))
            rows.append(row)
    if seen_episodes != EXPECTED_EPISODES:
        raise ValueError(("missing/extra task episodes", sorted(EXPECTED_EPISODES - seen_episodes), sorted(seen_episodes - EXPECTED_EPISODES)))
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} score rows, found {len(rows)}")
    keys = [(int(row["task_id"]), int(row["episode"]), int(row.get("state_index", -1)), str(row["site_id"])) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate score row keys")
    registries_text = {_canonical(registry) for registry in registries}
    if len(registries_text) != 1:
        raise ValueError("registry drift across score files")
    registry = registries[0] or {}
    sites = [str(site["site_id"]) for site in registry.get("sites", [])]
    if len(sites) != SITES_PER_STATE or len(set(sites)) != SITES_PER_STATE:
        raise ValueError(f"expected registry of {SITES_PER_STATE} unique sites, found {len(sites)}")
    if sites_hint is not None and list(sites_hint) != sites:
        raise ValueError("registry site order changed")
    if any(not np.allclose(normalizers[0], normal, rtol=0.0, atol=1e-12) for normal in normalizers[1:]):
        raise ValueError("CAL normalizer drift across score files")
    status_counts: dict[str, int] = {}
    terminal_rows: list[dict[str, Any]] = []
    for row in rows:
        status = str(row.get("status", "ok"))
        status_counts[status] = status_counts.get(status, 0) + 1
        if bool(row.get("terminal")):
            terminal_rows.append({"task_id": int(row["task_id"]), "episode": int(row["episode"]),
                                  "state_index": row.get("state_index"), "site_id": row.get("site_id"),
                                  "status": status, "error": row.get("error")})
    return {"files": files, "file_summaries": file_summaries, "rows": rows, "sites": sites,
            "registry": registry, "normalizers": normalizers, "terminal_rows": terminal_rows,
            "status_counts": status_counts}


def _metric_coverage(rows: Iterable[Mapping[str, Any]], fields: Iterable[str]) -> dict[str, Any]:
    rows = list(rows)
    result = {}
    for field in fields:
        usable = sum(row.get("status", "ok") == "ok" and _finite_float(row.get(field)) is not None for row in rows)
        result[field] = {"total_rows": len(rows), "usable_ok_rows": usable,
                         "unavailable_or_nonfinite_rows": len(rows) - usable,
                         "used_only_complete_episodes": True}
    return result


def _selected_metrics(check_rows: list[dict[str, Any]], selected_sites: list[str]) -> dict[str, Any]:
    fields = {"local_raw": "local_mse", "local_normalized": "local_nmse", "action_only": "d_a",
              "observation_only": "d_o", "otc": "C"}
    return {name: selected_episode_metric(check_rows, selected_sites, field) for name, field in fields.items()}


def _strata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predicates = {
        "contact_transition_first_action": lambda row: bool(row.get("contact_transition_first_action", row.get("contact_transition", False))),
        "contact_present": lambda row: bool(row.get("contact_present", False)),
        "free": lambda row: not bool(row.get("contact_present", False)) and not bool(row.get("contact_transition_first_action", row.get("contact_transition", False))),
    }
    result: dict[str, Any] = {}
    fields = ['local_mse', 'local_nmse', 'd_a', 'first_action_mse', 'd_o', 'C']
    def diagnostic(subset):
        sites = sorted({row['site_id'] for row in subset})
        episodes = sorted({(row['task_id'], row['episode']) for row in subset})
        scores = {}
        for field in fields:
            scores[field] = {}
            for site in sites:
                means = []
                for task, episode in episodes:
                    values = [float(row[field]) for row in subset
                              if row['site_id'] == site and row['task_id'] == task
                              and row['episode'] == episode and row.get(field) is not None]
                    if values:
                        means.append(float(np.mean(values)))
                scores[field][site] = float(np.mean(means)) if means else None
        pairs = [('local_mse', 'C'), ('local_nmse', 'C'), ('d_a', 'C'),
                 ('d_a', 'd_o'), ('first_action_mse', 'd_o')]
        correlations = {}
        for first, second in pairs:
            x = [scores[first][s] for s in sites]
            y = [scores[second][s] for s in sites]
            correlations[first + '_vs_' + second] = None if any(v is None for v in x + y) else spearman(x, y)
        return {'episode_weighted_site_scores': scores, 'site_rank_correlations': correlations,
                'interpretation': 'descriptive across-site correlation, no significance or task-loss identity'}
    for name, predicate in predicates.items():
        subset = [row for row in rows if predicate(row)]
        result[name] = {"rows": len(subset), "states": len({(row["task_id"], row["episode"], row.get("state_index")) for row in subset}),
                        "episodes": sorted({f"{row['task_id']}/{row['episode']}" for row in subset}),
                        'by_split': {split: diagnostic([row for row in subset if row['split'] == split])
                                     for split in ['CAL', 'CHECK']}}
    return result


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    loaded = _load_rows(Path(args.input))
    rows = loaded["rows"]
    sites = loaded["sites"]
    input_complete = all(status == "ok" for status in loaded["status_counts"]) and all(
        str(summary.get("status")) == "ok" for summary in loaded["file_summaries"]
    )
    cal = [row for row in rows if row["split"] == "CAL"]
    check = [row for row in rows if row["split"] == "CHECK"]
    rankings = rank_methods(cal, check, sites, lambda_values=(0.5, 1.0, 2.0), lambda_o_values=(1.0,))
    per_task = {}
    for task in TASKS:
        task_cal = [row for row in cal if row['task_id'] == task]
        task_check = [row for row in check if row['task_id'] == task]
        task_ranks = rank_methods(task_cal, task_check, sites, lambda_values=(0.5, 1.0, 2.0), lambda_o_values=(1.0,))
        per_task[str(task)] = {
            'rankings': task_ranks,
            'cal_selections_on_check': {
                method: independent_check_diagnostics(task_check, ranking['cal_top2'])
                for method, ranking in task_ranks.items() if ranking.get('status') == 'ok'},
            'interpretation': 'two CAL and two CHECK episodes for this fixed task; descriptive only'}
    sensitivity: dict[str, Any] = {}
    controls: dict[str, Any] = {}
    selected_checks: dict[str, Any] = {}
    for method_index, (method, ranking) in enumerate(rankings.items()):
        if ranking.get("status") != "ok":
            sensitivity[method] = {"status": "incomplete"}
            controls[method] = {"status": "incomplete"}
            selected_checks[method] = {"status": "incomplete"}
            continue
        cal_episode_scores = method_episode_scores(cal, sites, method)
        sensitivity[method] = {
            "bootstrap": bootstrap_top2(cal_episode_scores, sites, seed=args.seed + method_index,
                                         iterations=args.bootstrap_iters),
            "leave_one_episode_out": leave_one_episode_out_top2(cal_episode_scores, sites),
        }
        selected = list(ranking["cal_top2"])
        selected_checks[method] = {
            "cal_selected_sites": selected,
            "check_metrics": _selected_metrics(check, selected),
            "independent_check_diagnostics": independent_check_diagnostics(check, selected),
            "interpretation": "fixed CAL-selected sites evaluated on CHECK; no joint-quantized outcome",
        }
        controls[method] = permutation_reassignment_control(
            ranking["cal_scores"], check, sites, seed=args.seed + 10000 + method_index,
            iterations=args.permutation_iters,
        )
    metric_fields = ["local_mse", "local_nmse", "d_a", "first_action_mse", "d_o", "C", "max_object_position_delta_m", "qpos_l2"]
    composition = {}
    for split in ['cal', 'check']:
        action_scores = rankings['action_only'].get(split + '_scores', {})
        observation_scores = rankings['observation_only'].get(split + '_scores', {})
        if action_scores and observation_scores:
            mean_a = float(np.mean(list(action_scores.values())))
            mean_o = float(np.mean(list(observation_scores.values())))
            composition[split] = {'mean_d_a': mean_a, 'mean_d_o': mean_o,
                                  'observation_share_of_mean_C': mean_o / (mean_a + mean_o) if mean_a + mean_o else None}
    validation_keys = sorted({
        str(key)
        for row in rows
        for key in (row.get("validation", {}) if isinstance(row.get("validation", {}), Mapping) else {})
    })
    replay_by_key: dict[str, Any] = {}
    for key in validation_keys:
        values = [
            value
            for row in rows
            for value in [_finite_float((row.get("validation", {}) or {}).get(key))]
            if value is not None
        ]
        replay_by_key[key] = {"status": "ok" if values else "unavailable", "max": max(values) if values else None,
                              "n_finite_values": len(values), "unit": "the validation field's native units"}
    qpos_values = [value for row in rows for value in [_finite_float(row.get("qpos_l2"))] if value is not None]
    qpos_diagnostic = {"status": "ok" if qpos_values else "unavailable", "max": max(qpos_values) if qpos_values else None,
                       "n_finite_values": len(qpos_values),
                       "unit": "mixed joint coordinates (translations, angles, quaternion components); not metres",
                       "interpretation": "quantization effect, not replay noise"}
    return {
        "schema": 3,
        "status": "ok" if input_complete else "partial_failure",
        "completion_gate": {
            "all_rows_status_ok": input_complete,
            "complete_status_requires_all_rows_and_files_ok": True,
            "message": None if input_complete else "input contains failed/partial score rows; report is diagnostic only",
        },
        "source_files": [str(file) for file in loaded["files"]],
        "episodes": len(loaded["file_summaries"]),
        "states": len({(row["task_id"], row["episode"], row.get("state_index")) for row in rows}),
        "single_site_rows": len(rows),
        "expected_shape": {"score_json_files": EXPECTED_SCORE_FILES, "rows": EXPECTED_ROWS, "sites": len(sites),
                           "tasks": list(TASKS), "CAL_episodes": list(CAL_EPISODES), "CHECK_episodes": list(CHECK_EPISODES)},
        "registry": loaded["registry"],
        "normalizers": {"sigma_a2": loaded["normalizers"][0][0], "sigma_o": loaded["normalizers"][0][1],
                        "source": "CAL task0/task1 episodes0/1 selected states only"},
        "retention": {"all_score_rows_retained": True, "status_counts": loaded["status_counts"],
                      "terminal_row_count": len(loaded["terminal_rows"]),
                      "terminal_rows": loaded["terminal_rows"],
                      "metric_coverage": _metric_coverage(rows, metric_fields),
                      "file_summaries": loaded["file_summaries"]},
        "ranking_definition": {
            "cal_only": True,
            "direction": "higher diagnostic error score ranks earlier for top2 sensitivity candidates",
            "local_raw": "episode mean local_hook_output_mse",
            "local_normalized": "episode mean local_hook_output_mse_normalized",
            "action_only": "episode mean d_a",
            "first_action_only": "secondary diagnostic: episode mean first_action_mse; tests whether full-chunk versus first-step horizon mismatch explains differences",
            "observation_only": "episode mean d_o",
            "otc": "lambda_a*d_a + lambda_o*d_o with lambda_o=1 and lambda_a in {0.5,1,2}; otc is the default (1,1) alias",
            "episode_weighting": "state means within each complete episode, then equal episode means",
        },
        "rankings": rankings,
        "per_task": per_task,
        "score_composition": composition,
        "physics_diagnostics": {"qpos_l2": qpos_diagnostic},
        "cal_ranking_sensitivity": sensitivity,
        "cal_selections_on_check": selected_checks,
        "check_site_score_permutation_controls": controls,
        "contact_strata": _strata(rows),
        "maximum_replay_noise": {
            "status": "ok" if any(item["status"] == "ok" for item in replay_by_key.values()) else "unavailable",
            "by_validation_key": replay_by_key,
            "definition": "per-key maxima; values with different units are never combined",
        },
        "synthetic_self_check": synthetic_self_check(args.seed),
        "claims_not_tested": [
            "statistical significance; four CAL and four CHECK episodes are descriptive only",
            "joint quantization protection or task-success gain",
            "rank changes or selected sensitive sites as evidence of joint protection",
            "real-kernel speed, memory, or deployment gain",
            "cross-suite generalization",
        ],
        "analysis_limitations": [
            "Permutation/reassignment controls preserve CHECK measurements and only randomize CAL score-to-site labels.",
            "max_object_position_delta_m is explicitly unavailable when runner object-body extraction produced no finite value.",
            "Contact difference is a binary next-contact-pair diagnostic, not a causal contact transition test.",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate OTC phase-1 score JSON with episode-weighted descriptive diagnostics.")
    parser.add_argument("--input", help="directory containing the eight score_task*_episode*.json files")
    parser.add_argument("--out", help="output analysis JSON path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-iters", type=int, default=DEFAULT_BOOTSTRAP_ITERS)
    parser.add_argument("--permutation-iters", type=int, default=DEFAULT_PERMUTATION_ITERS)
    parser.add_argument("--self-check", action="store_true", help="run only the deterministic CPU synthetic self-check")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.self_check:
        result = synthetic_self_check(args.seed)
        if args.out:
            _write_json(Path(args.out), result)
        print(json.dumps(result, indent=2, allow_nan=False, default=_json_default))
        return
    if not args.input or not args.out:
        parser.error("--input and --out are required unless --self-check is used")
    if args.bootstrap_iters <= 0 or args.permutation_iters <= 0:
        parser.error("bootstrap/permutation iterations must be positive")
    report = build_report(args)
    _write_json(Path(args.out), report)
    print(json.dumps({"status": "ok", "episodes": report["episodes"], "rows": report["single_site_rows"], "output": args.out}, default=_json_default))


if __name__ == "__main__":
    main()
