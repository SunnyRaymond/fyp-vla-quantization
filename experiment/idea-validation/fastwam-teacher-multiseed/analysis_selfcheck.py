"""Small synthetic checks; run only inside the same guarded PBS allocation."""

from analyze import (
    _covariance_and_rank,
    _histogram,
    _independent_seed_index_shuffle,
    _kernel_matrices,
    _projected_distance,
    _relative_frobenius,
    _smooth_counts,
    require_compute_allocation,
)


def run_selfcheck() -> None:
    require_compute_allocation()
    import json
    import numpy as np
    import os
    from pathlib import Path

    p = np.array([[0.2, 0.3, 0.5]], dtype=np.float64)
    identity = _projected_distance(p, p)
    assert abs(float(identity["js"][0])) < 1e-12
    assert abs(float(identity["kl_pq"][0])) < 1e-12
    assert _relative_frobenius(p, p) == 0.0

    rng = np.random.default_rng(17)
    samples = rng.standard_normal((4, 12))
    _, rank_info = _covariance_and_rank(samples)
    _, reversed_rank_info = _covariance_and_rank(samples[::-1])
    assert np.allclose(samples.mean(axis=0), samples[::-1].mean(axis=0))
    assert rank_info["numerical_rank"] <= 3
    assert np.isclose(rank_info["effective_rank"], reversed_rank_info["effective_rank"])
    toy = np.arange(16 * 60, dtype=np.float64).reshape(16, 10, 6)
    shuffled = _independent_seed_index_shuffle(toy, np.random.default_rng(23))
    assert shuffled.shape == toy.shape
    assert np.allclose(np.sort(shuffled.reshape(16, 60), axis=0), np.sort(toy.reshape(16, 60), axis=0))

    _, full_info = _covariance_and_rank(rng.standard_normal((16, 60)))
    assert full_info["numerical_rank"] <= 15
    constant = np.ones((16, 60), dtype=np.float64)
    _, constant_info = _covariance_and_rank(constant)
    assert np.var(constant) == 0.0
    assert constant_info["effective_rank"] == 0.0
    assert constant_info["numerical_rank"] == 0

    bandwidths = {"check": np.array([0.1])}
    kernels = _kernel_matrices(bandwidths)["check"][0]
    bimodal_hist = _histogram(np.array([-1.0, -1.0, 1.0, 1.0]))
    collapsed_hist = _histogram(np.zeros(4))
    assert bimodal_hist.shape == (72,) and float(bimodal_hist.sum()) == 4.0
    assert collapsed_hist.shape == (72,) and float(collapsed_hist.sum()) == 4.0
    bimodal = _smooth_counts(bimodal_hist[None, :], kernels[None, :, :])
    collapsed = _smooth_counts(collapsed_hist[None, :], kernels[None, :, :])
    assert bimodal.shape == collapsed.shape == (1, 72)
    assert np.allclose(bimodal.sum(axis=1), 1.0) and np.allclose(collapsed.sum(axis=1), 1.0)
    assert abs(float(np.mean([-1.0, -1.0, 1.0, 1.0]))) == abs(float(np.mean(np.zeros(4))))
    assert float(_projected_distance(bimodal, collapsed)["js"][0]) > 0.0

    root = Path(os.environ.get("NEW_ROOT", Path(__file__).resolve().parent))
    out_path = root / "results" / "analysis" / "analysis_selfcheck.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "complete": True,
        "passed": True,
        "pbs_jobid": os.environ["PBS_JOBID"],
        "checks": ["identity_divergence", "order_invariant_mean_and_covariance", "covariance_rank_bounds", "constant_sample_zero_variance", "independent_shuffle_marginal_preservation", "histogram_count_shape", "equal_mean_bimodal_js_positive"],
        "data_loaded": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"analysis_selfcheck: PASS (synthetic only; wrote {out_path})")


if __name__ == "__main__":
    run_selfcheck()
