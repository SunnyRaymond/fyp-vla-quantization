"""CPU-only paired280 summaries against the original fifteen frozen methods."""
import argparse
from collections import Counter
import importlib.util
import os
from pathlib import Path
import re
import sys

from control_common import guard, read, save


def summarize(rows, candidate, references):
    expected = set(range(280))
    cells = Counter((row["suite"], row["dimension"]) for row in rows)
    if (len(rows) != 280 or [row["index"] for row in rows] != list(range(280))
            or len(cells) != 28 or set(cells.values()) != {10}
            or len({suite for suite, _ in cells}) != 4
            or len({dimension for _, dimension in cells}) != 7
            or set(candidate) != expected or not references
            or any(set(records) != expected for records in references.values())):
        raise ValueError("Paired summary requires the full frozen 4 x 7 x 10 cohort")
    if any(type(record.get("success")) is not bool
           for records in [candidate, *references.values()] for record in records.values()):
        raise ValueError("Every episode must contain a boolean success outcome")

    def bucket(indices):
        successes = sum(candidate[index]["success"] for index in indices)
        comparisons = {}
        for arm, records in references.items():
            wins = sum(candidate[i]["success"] and not records[i]["success"] for i in indices)
            losses = sum(records[i]["success"] and not candidate[i]["success"] for i in indices)
            comparisons[arm] = {
                "reference_successes": sum(records[i]["success"] for i in indices),
                "wins": wins, "losses": losses, "ties": len(indices) - wins - losses,
                "episodes": len(indices), "net_successes": wins - losses,
                "success_rate_difference": (wins - losses) / len(indices),
            }
        return {"successes": successes, "episodes": len(indices),
                "success_rate": successes / len(indices), "paired_vs": comparisons}

    suites = sorted({row["suite"] for row in rows})
    dimensions = sorted({row["dimension"] for row in rows})
    return {
        "overall": bucket(list(range(280))),
        "by_suite": {suite: bucket([row["index"] for row in rows if row["suite"] == suite])
                     for suite in suites},
        "by_dimension": {dimension: bucket([row["index"] for row in rows
                                            if row["dimension"] == dimension])
                         for dimension in dimensions},
        "by_suite_dimension": {
            suite: {dimension: bucket([row["index"] for row in rows
                                       if row["suite"] == suite and row["dimension"] == dimension])
                    for dimension in dimensions} for suite in suites},
    }


def original_references(parent):
    """Reuse the successful original loaders without touching their result files."""
    source = Path(parent["source_closed_artifacts"])
    closed = parent["source_closed_protocol"]
    if read(Path(parent["source_closed_root"]) / "protocol.json") != closed:
        raise ValueError("Original closed-loop source protocol differs")
    if ((source / "exit_code.txt").read_text().strip() != "0"
            or not (source / "PIPELINE_COMPLETE").is_file()
            or read(Path(parent["source_closed_root"]) / "results/summary.json").get("complete") is not True):
        raise ValueError("Original closed-loop reference run is not complete")
    sys.path.append(str(source))
    spec = importlib.util.spec_from_file_location("original_closedloop_aggregate", source / "aggregate.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    manifest, rows = old.load_manifest(parent["frozen_evaluation"]["closedloop_manifest"])
    references, arms = old._load_references(closed, manifest, rows)
    previous, by_arm, _metadata, _setups, checkpointed = old.load_workers(
        Path(parent["source_closed_root"]) / "results", closed, manifest, rows)
    if checkpointed or len(previous) != 560:
        raise ValueError("Original softbin/relational references are incomplete")
    result = {arm: {index: record for (name, index), record in references.items() if name == arm}
              for arm in arms}
    result.update(by_arm)
    if len(result) != 15 or any(set(records) != set(range(280)) for records in result.values()):
        raise ValueError("Expected fifteen original methods, each with the same 280 episodes")
    return rows, result


def run(unit, arm):
    from aggregate import completed
    from training_closedloop import shard_indices, validate_record

    here = Path(__file__).resolve().parent
    parent, config = read(here / "protocol.json"), read(here / "training_protocol.json")
    root = Path(parent["remote_root"])
    receipts = {shard: completed(root, f"closedloop_{arm}_{shard}") for shard in (0, 1)}
    binding = receipts[0]["deployment_binding"]
    if (arm not in config["arms"] or binding.get("training_protocol") != config
            or receipts[1]["deployment_binding"] != binding):
        raise ValueError("Both closed-loop shards must bind this frozen training configuration")
    rows, references = original_references(parent)
    candidate = {}
    for shard, receipt in receipts.items():
        source_unit = f"closedloop_{arm}_{shard}"
        indices = shard_indices(shard)
        if (receipt.get("unit") != source_unit or receipt.get("arm") != arm
                or receipt.get("completed_indices") != indices
                or receipt.get("completed") != 140 or receipt.get("expected") != 140):
            raise ValueError("A closed-loop shard has incomplete or changed coverage")
        for index in indices:
            record = read(root / "results" / source_unit / "episodes" / f"{index}.json")
            validate_record(record, parent, rows[index], index, arm, binding)
            candidate[index] = record
    report = summarize(rows, candidate, references)
    summary = {
        "protocol": parent["protocol"], "unit": unit, "arm": arm,
        "pbs_jobid": os.environ["PBS_JOBID"], "complete": True,
        "deployment_binding": binding, "manifest_variants": 280,
        "episode_slots": 280, "reference_arms": list(references),
        "reference_episode_slots": 4200,
        "source_jobs": {f"closedloop_{arm}_{s}": value["pbs_jobid"]
                        for s, value in receipts.items()},
        "reference_job": parent["source_closed_job"],
        "training_updates": 0, "new_model_queries": 0, "new_teacher_queries": 0,
        "scope": "original paired280; decoded-BF16 VQ/A4/BF16-KV; success only, no native-kernel speed claim",
        **report,
    }
    details = [{"index": index, "suite": rows[index]["suite"],
                "dimension": rows[index]["dimension"], "success": candidate[index]["success"],
                "references": {name: records[index]["success"] for name, records in references.items()}}
               for index in range(280)]
    save(root / "results" / unit / "details.json", {"records": details})
    save(root / "results" / unit / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    args = parser.parse_args()
    match = re.fullmatch(r"aggregate_closedloop_([a-z][a-z0-9_]{0,25})", args.unit)
    if not match:
        parser.error("Expected aggregate_closedloop_<training_arm>")
    guard()
    run(args.unit, match[1])


if __name__ == "__main__":
    main()
