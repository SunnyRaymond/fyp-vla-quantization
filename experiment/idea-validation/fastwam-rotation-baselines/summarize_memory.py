"""Summarize recorded query memory on an approved CPU allocation."""
import json
import math
import os
from pathlib import Path
import socket
import statistics


def main():
    if not os.environ.get("PBS_JOBID") or "login" in socket.gethostname().lower():
        raise RuntimeError("An approved PBS compute allocation is required")
    nodefile = Path(os.environ["PBS_NODEFILE"])
    nodes = {line.split(".")[0] for line in nodefile.read_text().splitlines()}
    if socket.gethostname().split(".")[0] not in nodes:
        raise RuntimeError("Current host is outside the approved allocation")
    root = Path(__file__).resolve().parent
    arms = ("bf16", "quarot_adapted_w4a4", "spinquant_adapted_w4a4")
    fields = ("baseline_allocated_gpu_GB", "peak_allocated_gpu_GB", "peak_reserved_gpu_GB")
    samples = {arm: {kind: [] for kind in ("all_queries", "first_query", "subsequent_queries")} for arm in arms}
    episodes = {arm: 0 for arm in arms}
    slots = set()
    for worker in range(4):
        with (root / "results" / ("worker_%d" % worker) / "episodes.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                arm = record["arm"]
                slot = (arm, record["index"])
                assert slot not in slots and record["complete"] is True
                slots.add(slot)
                episodes[arm] += 1
                for metric in record["metrics"]:
                    values = {key: float(metric[key]) for key in fields}
                    assert all(math.isfinite(value) and value >= 0 for value in values.values())
                    assert values["peak_allocated_gpu_GB"] >= values["baseline_allocated_gpu_GB"]
                    assert values["peak_reserved_gpu_GB"] >= values["peak_allocated_gpu_GB"]
                    values["query_increment_allocated_gpu_GB"] = values["peak_allocated_gpu_GB"] - values["baseline_allocated_gpu_GB"]
                    samples[arm]["all_queries"].append(values)
                    kind = "first_query" if metric["first_query_in_arm"] else "subsequent_queries"
                    samples[arm][kind].append(values)
    assert len(slots) == 840 and all(count == 280 for count in episodes.values())
    timing = json.loads((root / "results" / "summary.json").read_text())["timing"]
    output = {"source_evaluation_job_id": "25728285.pbs101", "aggregation_job_id": os.environ["PBS_JOBID"],
              "aggregation_host": socket.gethostname(), "units": "GB (10^9 bytes)", "episodes": episodes,
              "scope": "Per-query PyTorch memory; excludes CUDA context and external allocations. Sequential BF16 -> QuaRot -> SpinQuant retained allocator cache.", "arms": {}}
    for arm in arms:
        output["arms"][arm] = {}
        for kind, rows in samples[arm].items():
            if kind != "all_queries":
                assert len(rows) == timing[arm]["query_timing"][kind]["queries"]
            result = {"queries": len(rows)}
            for key in rows[0]:
                values = sorted(row[key] for row in rows)
                result[key] = {"min": values[0], "median": statistics.median(values),
                               "mean": statistics.mean(values), "p95": values[math.ceil(0.95 * len(values)) - 1], "max": values[-1]}
            output["arms"][arm][kind] = result
    dest = root / "results" / "memory_summary_25728285.json"
    temp = dest.with_suffix(".tmp")
    temp.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temp.replace(dest)
    print("MEMORY_SUMMARY_COMPLETE episodes=840 arms=3 path=" + str(dest), flush=True)


if __name__ == "__main__":
    main()
