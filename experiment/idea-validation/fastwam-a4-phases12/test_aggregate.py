"""Synthetic stdlib-only control tests; never read real PBS campaign data."""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest

from aggregate import BITS, PLUS_DIMENSIONS, PROTOCOL, AggregationError, _action_metrics, build_report


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _label(bits):
    if bits == (4, 4, 4):
        return "all_a4"
    if bits == (8, 8, 8):
        return "all_a8"
    return f"mixed_v{bits[0]}a{bits[1]}p{bits[2]}"


def _action(case_id: int, seed_index: int, offset: float = 0.0):
    base = 0.1 + case_id * 0.001 + seed_index * 0.002
    return [[base + j * 0.0001 + offset * (1 if i < 10 and j < 6 else
                                            5 if i < 10 else
                                            3 if j < 6 else 7)
             for j in range(7)] for i in range(32)]


def _runner_action_metrics(action, reference, suffix="bf16"):
    delta = [[a - b for a, b in zip(arow, brow)] for arow, brow in zip(action, reference)]
    first10 = delta[:10]

    def rmse(items):
        return math.sqrt(sum(value * value for value in items) / len(items))

    return {
        f"motor_rmse_first10_vs_{suffix}": rmse([v for row in first10 for v in row[:6]]),
        f"gripper_rmse_first10_vs_{suffix}": rmse([row[6] for row in first10]),
        f"motor_rmse_32_vs_{suffix}": rmse([v for row in delta for v in row[:6]]),
        f"gripper_rmse_32_vs_{suffix}": rmse([row[6] for row in delta]),
        f"max_abs_first10_vs_{suffix}": max(abs(v) for row in first10 for v in row),
        f"max_abs_32_vs_{suffix}": max(abs(v) for row in delta for v in row),
    }


def make_synthetic_root(root: Path) -> None:
    plan_rows, terminal_cases = [], []
    for case_id in range(22):
        domain = "original" if case_id < 10 else "plus"
        dimension = "unperturbed" if domain == "original" else PLUS_DIMENSIONS[(case_id - 10) // 2]
        source_index = None if domain == "original" else case_id + 100
        task_id = case_id if domain == "original" else (case_id - 10) // 2
        seeds = [900000000 + case_id * 1000, 900000001 + case_id * 1000]
        base = {
            "case_id": case_id, "domain": domain, "dimension": dimension,
            "original_task": f"task_{task_id}", "task_id": task_id,
            "source_index": source_index, "environment_seed": 810000000 + case_id,
            "state_id": 0, "sampler_seeds": seeds,
            "observation_file": f"inputs/case_{case_id:02d}.npz",
            "metadata_file": f"inputs/case_{case_id:02d}.json",
        }
        plan_rows.append(base)
        _write_json(root / base["metadata_file"], {
            "case_id": case_id, "domain": domain, "dimension": dimension,
            "task_id": task_id, "environment_seed": base["environment_seed"],
            "state_id": 0, "settling_steps": 30, "evaluation_episodes": 0,
            "episode_count": 0, "statepath": "synthetic-state.json",
            "dataset_import_paths": {"source": "synthetic"},
        })
        job_id = f"toyjob{case_id:02d}"
        artifact_dir = root / "artifacts" / job_id
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "PIPELINE_COMPLETE").touch()
        (artifact_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")
        terminal_cases.append({
            "case_id": case_id, "job_id": job_id, "exit_status": 0,
            "marker_path": str(artifact_dir / "PIPELINE_COMPLETE"),
            "exit_code_path": str(artifact_dir / "exit_code.txt"),
        })

        query_records, by_seed = [], {}
        query_count = 0
        for seed_index, seed in enumerate(seeds):
            baseline = _action(case_id, seed_index)
            by_seed[seed] = {"bf16": baseline}

        for seed in seeds:
            seed_index = seeds.index(seed)
            query_count += 1
            query_records.append(_query(base, seed, seed_index, "bf16", None,
                                        by_seed[seed]["bf16"], query_count))
        for seed in seeds:
            seed_index = seeds.index(seed)
            for qtype, bits in (
                ("all_a8", (8, 8, 8)), ("all_a4", (4, 4, 4)),
                ("independent_reference", (4, 4, 4)),
                *((_label(bits), bits) for bits in BITS if bits not in ((4, 4, 4), (8, 8, 8))),
            ):
                offset = sum(1 for bit in bits if bit == 4) * 0.001
                if qtype == "independent_reference":
                    offset += 0.0005
                action = _action(case_id, seed_index, offset)
                by_seed[seed][qtype] = action
                query_count += 1
                record = _query(base, seed, seed_index, qtype, list(bits), action, query_count)
                if qtype != "bf16":
                    record["action_metrics_vs_bf16"] = _runner_action_metrics(action, by_seed[seed]["bf16"])
                if qtype == "independent_reference":
                    record["native_all_a4_vs_independent_reference"] = _runner_action_metrics(
                        by_seed[seed]["all_a4"], action, "independent_reference")
                if qtype == "all_a4":
                    trace_rel = Path("traces") / record["query_id"] / "stage_quantization_trace.json"
                    trace = _trace(case_id, seed, seed_index)
                    _write_json(root / "results" / f"case_{case_id:02d}" / trace_rel, trace)
                    record["trace_coverage"] = _trace_info(trace, trace_rel.as_posix())
                    record["observed_linear_modules"] = sorted(
                        {line["module"] for line in trace["records"]}
                    )
                    record["observed_linear_call_count"] = trace["calls"]
                    record["observed_linear_scopes"] = {
                        "video": ["video_expert.block"],
                        "action": ["action_expert.block"],
                        "proprio": ["proprio_encoder.block"],
                    }
                query_records.append(record)

        factorial = []
        for seed in seeds:
            for bits in BITS:
                factorial.append({"sampler_seed": seed, "bits": list(bits),
                                  "query_id": f"{_label(bits)}_seed{seed}", "reused_query": True})
        case_dir = root / "results" / f"case_{case_id:02d}"
        _write_json(case_dir / "case_summary.json", {
            "protocol": PROTOCOL, "status": "complete", "case_id": case_id,
            "pbs_jobid": job_id,
            "domain": domain, "dimension": dimension, "original_task": f"task_{task_id}",
            "task_id": task_id, "source_index": source_index,
            "environment_seed": base["environment_seed"], "sampler_seeds": seeds,
            "query_count": 20, "context_count": 2,
            "factorial_cell_count": 16, "factorial_cells": factorial,
            "episodes": 0, "post_query_env_steps": 0,
            "input": {"settling_steps": 30},
        })
        case_dir.mkdir(parents=True, exist_ok=True)
        with (case_dir / "queries.jsonl").open("w", encoding="utf-8") as stream:
            for record in query_records:
                stream.write(json.dumps(record, allow_nan=False) + "\n")

    _write_json(root / "plan.json", {"protocol": PROTOCOL, "inputs": plan_rows})
    _write_json(root / "terminal_evidence.json", {"cases": terminal_cases})


def _query(plan_row, seed, seed_index, qtype, bits, action, query_count):
    counters = {
        "integer_gemm_calls": 0, "native_int4_gemm_calls": 0,
        "kv_packed_prefills": 0, "kv_layer_reads": 0,
    }
    if qtype not in ("bf16", "independent_reference"):
        counters["integer_gemm_calls"] = 10
        counters["native_int4_gemm_calls"] = 1 if 4 in bits else 0
    stage_coverage = {
        stage: {"schedule_calls": 1, "step_calls": 10, "observed_steps": list(range(10))}
        for stage in ("video", "action")
    }
    qid = f"{qtype}_seed{seed}"
    scopes = {
        "video": ["video_expert.block"],
        "action": ["action_expert.block"],
        "proprio": ["proprio_encoder.block"],
    }
    return {
        "protocol": PROTOCOL, "case_id": plan_row["case_id"], "query_id": qid,
        "query_type": qtype, "domain": plan_row["domain"], "dimension": plan_row["dimension"],
        "original_task": plan_row["original_task"], "task_id": plan_row["task_id"],
        "source_index": plan_row["source_index"], "environment_seed": plan_row["environment_seed"],
        "sampler_seed": seed, "bits": bits, "query_count": query_count,
        "context_id": f"case_{plan_row['case_id']:02d}_seed{seed}",
        "raw_normalized_action_32x7": action,
        "action_metrics_vs_bf16": {},
        "native_all_a4_vs_independent_reference": None,
        "runtime_counters": {"before": {key: 0 for key in counters}, "after": counters, "delta": counters},
        "stage_scheduler_coverage": stage_coverage, "trace_coverage": None,
        "observed_linear_call_count": 3,
        "observed_linear_modules": sorted(name for names in scopes.values() for name in names),
        "observed_linear_scopes": scopes,
        "episodes": 0, "post_query_env_steps": 0,
    }


def _trace(case_id, seed, seed_index):
    names = (("video_expert.block", "video"), ("action_expert.block", "action"))
    records, coverage = [], []
    call_index = 900 + case_id * 100 + seed_index * 30
    for module, scope in names:
        stage = scope
        for step in range(10):
            record = {
                "call_index": call_index, "module": module, "scope": scope,
                "stage": stage, "step": step, "input_shape": [1, 4], "output_shape": [1, 4],
                "input_features": 4, "output_features": 4,
                "metrics": {
                    "input_absmax": 1.0, "input_rms": 0.4,
                    "a4_scale_min": 0.1, "a4_scale_mean": 0.1, "a4_scale_max": 0.1,
                    "a4_zero_code_fraction": 0.2, "a4_input_rmse": 0.01,
                    "a4_input_relative_rmse": 0.025,
                    "a8_scale_min": 0.01, "a8_scale_mean": 0.01, "a8_scale_max": 0.01,
                    "a8_zero_code_fraction": 0.1, "a8_input_rmse": 0.001,
                    "a8_input_relative_rmse": 0.0025,
                    "native_vs_a4_reference_rmse": 0.02,
                    "native_vs_a4_reference_max_abs": 0.04,
                    "native_vs_a4_reference_normalized_rmse": 0.05,
                    "a4_vs_a8_reference_rmse": 0.01,
                    "a4_vs_a8_reference_max_abs": 0.02,
                    "a4_vs_a8_reference_normalized_rmse": 0.03,
                },
            }
            records.append(record)
            coverage.append({"module": module, "scope": scope, "stage": stage, "step": step,
                             "input_shape": [1, 4], "output_shape": [1, 4], "calls": 1})
            call_index += 1
    module, scope = "proprio_encoder.block", "proprio"
    records.append({
        "call_index": call_index, "module": module, "scope": scope, "stage": "conditioning", "step": -1,
        "input_shape": [1, 4], "output_shape": [1, 4], "input_features": 4, "output_features": 4,
        "metrics": {
            "input_absmax": 1.0, "input_rms": 0.4,
            "a4_scale_min": 0.1, "a4_scale_mean": 0.1, "a4_scale_max": 0.1,
            "a4_zero_code_fraction": 0.2, "a4_input_rmse": 0.01,
            "a4_input_relative_rmse": 0.025,
            "a8_scale_min": 0.01, "a8_scale_mean": 0.01, "a8_scale_max": 0.01,
            "a8_zero_code_fraction": 0.1, "a8_input_rmse": 0.001,
            "a8_input_relative_rmse": 0.0025,
            "native_vs_a4_reference_rmse": 0.02,
            "native_vs_a4_reference_max_abs": 0.04,
            "native_vs_a4_reference_normalized_rmse": 0.05,
            "a4_vs_a8_reference_rmse": 0.01,
            "a4_vs_a8_reference_max_abs": 0.02,
            "a4_vs_a8_reference_normalized_rmse": 0.03,
        },
    })
    coverage.append({"module": module, "scope": scope, "stage": "conditioning", "step": -1,
                     "input_shape": [1, 4], "output_shape": [1, 4], "calls": 1})
    for record in records:
        record["call_index"] += 1
    prefill_record = dict(records[0])
    prefill_record.update(call_index=records[0]["call_index"] - 1,
                          stage="video_conditioning_prefill", step=-1)
    records.insert(0, prefill_record)
    prefill_coverage = dict(coverage[0])
    prefill_coverage.update(stage="video_conditioning_prefill", step=-1)
    coverage.insert(0, prefill_coverage)
    return {
        "diagnostic": "fastwam-stage-quantization-trace-v1", "completed": True,
        "context": {"case_id": case_id, "query_id": f"all_a4_seed{900000000 + case_id * 1000 + seed_index}",
                    "sampler_seed": 900000000 + case_id * 1000 + seed_index},
        "calls": len(records), "records": records, "coverage": coverage,
    }


def _trace_info(trace, path):
    modules, scopes = {}, {}
    for item in trace["coverage"]:
        modules[item["module"]] = modules.get(item["module"], 0) + item["calls"]
        scopes.setdefault(item["scope"], set()).add(item["module"])
    return {
        "coverage_records": len(trace["coverage"]), "modules": modules,
        "scopes": {scope: sorted(names) for scope, names in scopes.items()},
        "stage_steps": {stage: list(range(10)) for stage in ("video", "action")},
        "trace_path": path,
    }


class AggregateTest(unittest.TestCase):
    def test_primary_action_metric_excludes_late_steps_and_gripper(self):
        baseline = [[0.0] * 7 for _ in range(32)]
        action = [[0.0] * 7 for _ in range(32)]
        for step in range(10, 32):
            action[step][0] = 3.0
        for step in range(10):
            action[step][6] = 2.0
        measured = _action_metrics(action, baseline, "toy")
        self.assertEqual(measured["motor_rmse_first10"], 0.0)
        self.assertEqual(measured["gripper_rmse_first10"], 2.0)
        self.assertAlmostEqual(measured["rmse_32x7"], math.sqrt(238 / 224))
        for step in range(10):
            action[step][0] = 3.0
        measured = _action_metrics(action, baseline, "toy")
        self.assertAlmostEqual(measured["motor_rmse_first10"], math.sqrt(90 / 60))

    def test_synthetic_complete_campaign_and_compact_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_synthetic_root(root)
            summary, markdown = build_report(root)
            den = summary["denominators"]
            self.assertEqual((den["cases"], den["contexts"], den["actual_full_queries"]), (22, 44, 440))
            self.assertEqual((den["phase1_queries"], den["factorial_cell_entries"]), (176, 352))
            self.assertEqual((den["phase2_reused_endpoints"], den["phase2_new_mixed_queries"]), (88, 264))
            self.assertEqual(den["complete_all_a4_traces"], 44)
            self.assertEqual(summary["factorial_effects"]["contexts"], 44)
            self.assertEqual(summary["factorial_effects"]["primary_metric"], "motor_rmse_first10")
            self.assertEqual(summary["native_a4_vs_independent_reference_action"]["contexts"], 44)
            direct_primary = summary["native_a4_vs_independent_reference_action"]["metrics"]["motor_rmse_first10"]
            self.assertAlmostEqual(direct_primary["mean"], 0.0005)
            self.assertTrue(any(row["scope"] == "proprio" for row in summary["local_trace"]["summary"]))
            self.assertTrue(any(row["stage"] == "video_conditioning_prefill"
                                for row in summary["local_trace"]["summary"]))
            encoded = json.dumps(summary, allow_nan=False)
            self.assertLess(len(encoded.encode("utf-8")), 256 * 1024)
            self.assertIn("不提供成功率或 formal latency", markdown)

    def test_repeated_linear_calls_must_not_be_counted_as_unique_modules(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_synthetic_root(root)
            query_path = root / "results" / "case_00" / "queries.jsonl"
            records = [json.loads(line) for line in query_path.read_text(encoding="utf-8").splitlines()]
            record = next(row for row in records if row["query_type"] == "all_a4")
            self.assertGreater(record["observed_linear_call_count"], len(record["observed_linear_modules"]))
            record["observed_linear_call_count"] = len(record["observed_linear_modules"])
            query_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
            with self.assertRaisesRegex(AggregationError, "observed Linear call count mismatch"):
                build_report(root)

    def test_missing_marker_rejects_instead_of_reporting_partial_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_synthetic_root(root)
            (root / "artifacts" / "toyjob03" / "PIPELINE_COMPLETE").unlink()
            with self.assertRaisesRegex(AggregationError, "PIPELINE_COMPLETE"):
                build_report(root)

    def test_case_summary_job_id_must_match_terminal_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_synthetic_root(root)
            summary_path = root / "results" / "case_00" / "case_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["pbs_jobid"] = "another-job"
            _write_json(summary_path, summary)
            with self.assertRaisesRegex(AggregationError, "pbs_jobid"):
                build_report(root)


if __name__ == "__main__":
    unittest.main()
