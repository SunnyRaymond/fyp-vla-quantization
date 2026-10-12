"""Small self-check for the reused frozen cohort and paired VQ worker split."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / "fastwam-rotation-baselines" / "results" / "manifest.json"
PROTOCOL = HERE / "protocol.json"
sys.path.insert(0, str(HERE))
from evaluate import load_manifest, worker_arm, worker_indices


def validate() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    manifest, variants = load_manifest(MANIFEST)

    assert protocol["protocol"] == "fastwam-vq-plus-v1"
    assert protocol["cohort"]["manifest_protocol"] == manifest["protocol"]
    assert protocol["cohort"]["variants_per_arm"] == len(variants) == 280
    assert protocol["cohort"]["env_seed_base"] == 800000000
    assert protocol["cohort"]["sampler_seed_base"] == 900000000
    assert protocol["cohort"]["seed_stride"] == 1000
    assert protocol["cohort"]["reference_manifest"].endswith("/results/manifest.json")
    assert [row["index"] for row in variants] == list(range(280))
    assert len({row["variant_id"] for row in variants}) == 280
    assert all(row["state_id"] == 0 for row in variants)

    suite_counts = Counter(row["suite"] for row in variants)
    dimension_counts = Counter(row["dimension"] for row in variants)
    cell_counts = Counter((row["suite"], row["dimension"]) for row in variants)
    assert len(suite_counts) == 4 and set(suite_counts.values()) == {70}
    assert len(dimension_counts) == 7 and set(dimension_counts.values()) == {40}
    assert len(cell_counts) == 28 and set(cell_counts.values()) == {10}

    arms = protocol["arms"]
    workers = protocol["worker_assignment"]
    assert arms == ["row_scale_vq_a4", "micro_scale_vq_a4"]
    assert workers["model_loads_total"] == 4
    assert workers["variants_per_worker"] == 140
    assert protocol["soft_deadline_seconds"] == 85500
    assert protocol["source_parity_case_id"] == 1000
    assert protocol["source_parity_seed_index"] == 0
    assert protocol["source_parity_max_abs_tolerance"] == 1e-6
    assert protocol["source_parity_plan"].endswith("/plan.json")
    assert protocol["roots"]["old_banks"].endswith("fastwam-smooth-vq-phases12-20261006")
    assert protocol["worker_assignment"]["start_stagger_seconds"] == 20
    for worker_id in range(4):
        arm = arms[worker_id // 2]
        indices = worker_indices(worker_id)
        assert worker_arm(worker_id) == arm
        assert len(indices) == 140
        assert len(set(indices)) == 140
        assert all(index % 2 == worker_id % 2 for index in indices)
        assert arm in protocol["sources"]
        per_cell_parity = Counter(
            (variants[index]["suite"], variants[index]["dimension"], index % 2)
            for index in indices
        )
        assert len(per_cell_parity) == 28
        assert set(per_cell_parity.values()) == {5}

    assert protocol["references"]["reference_episode_slots"] == 4 * 280
    assert set(protocol["references"]["arms"]) == {
        "bf16", "quarot_adapted_w4a4", "spinquant_adapted_w4a4",
        "smooth05_hadamard_scalar_w4a4",
    }
    assert protocol["quantization"]["native_int4_kernel"] is False


if __name__ == "__main__":
    validate()
    print("FROZEN_VQ_PLUS_COHORT_OK variants=280 cells=28 workers=4")
