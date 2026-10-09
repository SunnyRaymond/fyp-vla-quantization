"""Run one 70-variant worker using the frozen Optional-IDM Plus protocol."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROTOCOL = "fastwam-rotation-baselines-v1"
ARMS = ("bf16", "quarot_adapted_w4a4", "spinquant_adapted_w4a4")


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest(path: Path, worker_id: int):
    doc = load_json(path)
    if doc.get("protocol") != PROTOCOL:
        raise ValueError(f"Unexpected protocol: {doc.get('protocol')}")
    rows = doc.get("variants")
    if not isinstance(rows, list) or len(rows) != 280:
        raise ValueError("Expected 280 frozen variants")
    ids = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or type(row.get("index")) is not int:
            raise ValueError(f"Invalid global variant row {index}")
        variant_id = str(row.get("variant_id", ""))
        if row.get("index") != index or not variant_id or variant_id in ids:
            raise ValueError(f"Invalid or duplicate global variant row {index}")
        ids.add(variant_id)
    cells = {}
    for row in rows:
        key = (row["suite"], row["dimension"])
        cells[key] = cells.get(key, 0) + 1
    if len({suite for suite, _ in cells}) != 4 or len({dimension for _, dimension in cells}) != 7:
        raise ValueError("Expected four suites by seven dimensions")
    if len(cells) != 28 or set(cells.values()) != {10}:
        raise ValueError("Expected 10 variants in each suite-by-dimension cell")
    shards = doc.get("shards", [])
    if len(shards) != 4:
        raise ValueError("Expected four frozen worker shards")
    shard_map = {}
    all_indices = []
    for shard in shards:
        key = shard.get("worker_id")
        indices = shard.get("indices")
        if type(key) is not int or key not in range(4) or key in shard_map or not isinstance(indices, list) or len(indices) != 70:
            raise ValueError(f"Invalid worker shard: {shard}")
        if any(type(index) is not int or not 0 <= index < len(rows) for index in indices):
            raise ValueError(f"Invalid index in worker {key} shard")
        if len(set(indices)) != 70:
            raise ValueError(f"Duplicate index in worker {key} shard")
        shard_map[key] = indices
        all_indices.extend(indices)
    if set(shard_map) != {0, 1, 2, 3} or sorted(all_indices) != list(range(len(rows))):
        raise ValueError("Worker shards must partition the global manifest")
    if worker_id not in shard_map:
        raise ValueError(f"Worker {worker_id} is absent from the manifest")
    return doc, rows, shard_map[worker_id]


def validate_prepared(results_root: Path):
    prepared = load_json(results_root / "PREPARED.json")
    if (prepared.get("protocol") != PROTOCOL or prepared.get("ready") is not True
            or prepared.get("numeric_check", {}).get("passed") is not True):
        raise RuntimeError("PREPARED.json has not passed the numerical checks")
    native = prepared.get("native_int4_check", {})
    if native.get("passed") is not True:
        authorization = prepared.get("user_authorized_native_gate_waiver", {})
        protocol = load_json(results_root / "protocol.json")
        training = prepared.get("training", {})
        if (native.get("passed") is not False or native.get("waived") is not True
                or authorization.get("approved") is not True
                or authorization.get("source_job_id") != prepared.get("pbs_jobid")
                or authorization.get("source_job_id") != "25727738.pbs101"
                or authorization.get("scope") != "four_gpu_full_evaluation"
                or protocol.get("protocol") != PROTOCOL
                or protocol.get("native_gate_waiver") != authorization
                or training.get("reused_existing_rotation") is not True
                or training.get("new_learning_steps") != 0
                or training.get("selected_step") != 200):
            raise RuntimeError("PREPARED.json has no matching user authorization for the failed native INT4 check")
        print("USER_AUTHORIZED_NATIVE_GATE_WAIVER " + json.dumps(authorization, ensure_ascii=False), flush=True)
    banks = prepared.get("banks", {})
    resolved = {}
    for arm in ARMS[1:]:
        bank_value = banks.get(arm)
        if not bank_value:
            raise RuntimeError(f"PREPARED.json has no bank path for {arm}")
        path = Path(bank_value)
        if not path.is_absolute():
            path = results_root / path
        if not path.is_dir() or not (path / "manifest.json").is_file():
            raise RuntimeError(f"Prepared bank is unavailable for {arm}: {path}")
        bank_doc = load_json(path / "manifest.json")
        if bank_doc.get("arm") != arm:
            raise RuntimeError(f"Bank arm mismatch at {path}: {bank_doc.get('arm')}")
        resolved[arm] = path
    return prepared, resolved


def recover_records(path: Path, rows, indices):
    if not path.exists():
        return {}
    payload = path.read_bytes()
    chunks = payload.splitlines(keepends=True)
    if chunks and not chunks[-1].endswith(b"\n"):
        try:
            json.loads(chunks[-1])
        except json.JSONDecodeError:
            payload = b"".join(chunks[:-1])
            path.write_bytes(payload)
            print("RESUME_DROPPED_TORN_FINAL_LINE", flush=True)
        else:
            path.write_bytes(payload + b"\n")
    completed = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            arm, index = record.get("arm"), record.get("index")
            if (record.get("protocol") != PROTOCOL or record.get("complete") is not True
                    or arm not in ARMS or type(index) is not int or index not in indices
                    or not isinstance(record.get("success"), bool)):
                raise ValueError(f"Invalid completed record at {path}:{line_number}")
            row = rows[index]
            if (str(record.get("variant_id")) != str(row["variant_id"])
                    or record.get("suite") != row["suite"]
                    or record.get("dimension") != row["dimension"]
                    or record.get("task_id") != int(row["task_id"])
                    or record.get("state_id") != 0
                    or record.get("env_seed") != 800_000_000 + index * 1000
                    or record.get("sampler_seed") != 900_000_000 + index * 1000):
                raise ValueError(f"Completed slot identity mismatch at {path}:{line_number}")
            if arm != "bf16":
                evidence = record.get("real_quant_evidence", {})
                if (evidence.get("rotation_bank") != arm or evidence.get("activation_bits") != 4
                        or evidence.get("integer_gemm_calls", 0) <= 0
                        or evidence.get("native_int4_gemm_calls", 0) <= 0
                        or evidence.get("native_int4_tensorcore") is not True
                        or evidence.get("packed_weight_tensor_bytes", 0) <= 0
                        or evidence.get("bf16_target_weight_absent") is not True):
                    raise ValueError(f"Native W4A4 evidence is incomplete at {path}:{line_number}")
            slot = (arm, index)
            if slot in completed:
                raise ValueError(f"Duplicate completed slot in resume file: {slot}")
            completed[slot] = record
    return completed


class RotationRuntime:
    """Reuse the original task, observation, inference, and episode implementation."""

    def __init__(self, out: Path, bank_paths):
        import runner
        import rotation

        runner.PROTOCOL = PROTOCOL
        self.base = runner.Runtime(out)
        self.rotation = rotation
        self.bank_paths = bank_paths

    def arm(self, arm: str):
        if arm not in ARMS:
            raise ValueError(arm)
        if arm == "bf16":
            if self.base.q.weights_converted:
                raise RuntimeError("Cannot return to BF16 after installing a packed bank")
            self.base.q.enable("bf16")
        else:
            self.base.q.disable()
            self.rotation.apply_native_bank(self.base.q, str(self.bank_paths[arm]))
            self.base.q.enable("w4a4")
        self.base.current_arm = arm

    def episode(self, row, index):
        arm = self.base.current_arm
        result = self.base.episode(row, index)
        if arm != "bf16":
            evidence = result["real_quant_evidence"]
            if (evidence.get("native_int4_gemm_calls", 0) <= 0
                    or evidence.get("native_int4_tensorcore") is not True):
                raise RuntimeError(f"{arm} episode did not execute native INT4 Tensor Core GEMM")
            evidence["rotation_bank"] = arm
        result["protocol"] = PROTOCOL
        result["arm"] = arm
        return result

    def close(self):
        base = self.base
        if base.env is not None:
            base.env.close()
            base.env = None
        if hasattr(base.q, "_rotation_native_state"):
            self.rotation.discard_native_bank(base.q)
        base.q.close()


def append_episode(path: Path, record):
    encoded = (json.dumps(record, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    from runner import guard, seed_for

    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, choices=range(4), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    _manifest, rows, indices = validate_manifest(args.manifest, args.worker_id)
    results_root = args.manifest.parent
    _prepared, bank_paths = validate_prepared(results_root)
    episodes_path = args.out / "episodes.jsonl"
    completed = recover_records(episodes_path, rows, set(indices))
    expected = len(indices) * len(ARMS)
    arm_counts = {arm: sum(slot_arm == arm for slot_arm, _ in completed) for arm in ARMS}
    if len(completed) == expected:
        if set(arm_counts.values()) != {len(indices)}:
            raise ValueError(f"Resume file has incomplete arm coverage: {arm_counts}")
        save(args.out / "summary.json", {
            "protocol": PROTOCOL, "complete": True, "worker_id": args.worker_id,
            "episodes": len(completed), "expected": expected, "indices": indices,
            "arm_episodes": arm_counts, "banks": {arm: str(bank_paths[arm]) for arm in ARMS[1:]},
            "resumed_from_episode_file": True,
        })
        save(args.out / "progress.json", {
            "protocol": PROTOCOL, "worker_id": args.worker_id, "status": "complete",
            "completed": expected, "expected": expected,
        })
        print(f"WORKER_COMPLETE worker={args.worker_id} slots={expected} resumed=true", flush=True)
        return

    runtime = None
    try:
        runtime = RotationRuntime(args.out, bank_paths)
        for arm in ARMS:
            runtime.arm(arm)
            pending = [index for index in indices if (arm, index) not in completed]
            for index in pending:
                row = rows[index]
                save(args.out / "progress.json", {
                    "protocol": PROTOCOL, "worker_id": args.worker_id, "status": "running",
                    "arm": arm, "variant_id": str(row["variant_id"]), "index": index,
                    "completed": len(completed), "expected": expected,
                })
                result = runtime.episode(row, index)
                result["worker_id"] = args.worker_id
                result["sampler_seed"] = seed_for(index)
                append_episode(episodes_path, result)
                completed[(arm, index)] = result
                save(args.out / "progress.json", {
                    "protocol": PROTOCOL, "worker_id": args.worker_id, "status": "running",
                    "arm": arm, "variant_id": str(row["variant_id"]), "index": index,
                    "completed": len(completed), "expected": expected,
                })
                print(f"SLOT_COMPLETE worker={args.worker_id} arm={arm} index={index} "
                      f"variant={row['variant_id']} success={result['success']}", flush=True)

        if len(completed) != expected:
            raise RuntimeError(f"Worker completed {len(completed)} of {expected} slots")
        arm_counts = {arm: sum(slot_arm == arm for slot_arm, _ in completed) for arm in ARMS}
        if set(arm_counts.values()) != {len(indices)}:
            raise RuntimeError(f"Worker arm coverage mismatch: {arm_counts}")
        summary = {
            "protocol": PROTOCOL, "complete": True, "worker_id": args.worker_id,
            "episodes": len(completed), "expected": expected, "indices": indices,
            "arm_episodes": arm_counts,
            "banks": {arm: str(bank_paths[arm]) for arm in ARMS[1:]},
            "quantization_runtime": runtime.base.q.summary(),
        }
        save(args.out / "summary.json", summary)
        save(args.out / "progress.json", {
            "protocol": PROTOCOL, "worker_id": args.worker_id, "status": "complete",
            "completed": expected, "expected": expected,
        })
        print(f"WORKER_COMPLETE worker={args.worker_id} slots={expected}", flush=True)
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    main()
