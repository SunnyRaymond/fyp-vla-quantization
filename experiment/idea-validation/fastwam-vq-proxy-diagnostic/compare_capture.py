"""Compare the saved source and failed preflight captures inside a PBS allocation."""
import json
import os
from pathlib import Path

import torch


def check_allocation():
    assert os.environ.get("PBS_JOBID") and os.environ.get("PBS_NODEFILE")
    assert "login" not in os.uname().nodename
    nodes = Path(os.environ["PBS_NODEFILE"]).read_text().splitlines()
    assert os.uname().nodename.split(".")[0] in {node.split(".")[0] for node in nodes}


check_allocation()
root = Path("/scratch/users/ntu/yguo017/fastwam-vq-proxy-diagnostic-20261010")
source = Path("/scratch/users/ntu/yguo017/fastwam-vq-action-aware-20261009")
old = torch.load(source / "results/full/sensitivity/case_0.pt", map_location="cpu", weights_only=True, mmap=True)
new = torch.load(root / "results/preflight/sensitivity/case_0.pt", map_location="cpu", weights_only=True, mmap=True)
counts = {key: 0 for key in ("calls", "inputs", "teacher_y", "adjoints")}
maximum = {key: 0.0 for key in ("inputs", "teacher_y", "adjoints")}
examples = []
for name, old_calls in old["layers"].items():
    new_calls = new["layers"][name]
    assert len(old_calls) == len(new_calls), name
    for old_call, new_call in zip(old_calls, new_calls):
        counts["calls"] += 1
        pos = torch.searchsorted(new_call["row_ids"], old_call["row_ids"])
        assert torch.equal(new_call["row_ids"][pos], old_call["row_ids"])
        for key in ("inputs", "teacher_y", "adjoints"):
            a = old_call[key]
            b = new_call[key][:4, pos] if key == "adjoints" else new_call[key][pos]
            if not torch.equal(a, b):
                counts[key] += 1
                gap = float((a - b).abs().max())
                maximum[key] = max(maximum[key], gap)
                if len(examples) < 8:
                    examples.append({"module": name, "call_id": old_call["call_id"], "key": key, "max_abs": gap})
print(json.dumps({"counts": counts, "max_abs": maximum, "examples": examples}), flush=True)
