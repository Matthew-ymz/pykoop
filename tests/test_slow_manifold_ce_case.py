from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.slow_manifold_ce_case.slow_manifold_ce_case import (
    SlowManifoldConfig,
    analyze_case,
    analyze_observable_modes,
    average_reversibility,
    build_observables,
    choose_truncation_rank,
    simulate_slow_manifold_system,
)


def test_simulation_returns_expected_shapes():
    config = SlowManifoldConfig(total_steps=800, burn_in=100, seed=7)
    result = simulate_slow_manifold_system(config)

    assert result["states"].shape == (700, 2)
    assert result["u"].shape == (700,)
    assert result["v"].shape == (700,)


def test_observable_dictionary_contains_named_features():
    states = np.array(
        [
            [1.0, 2.0],
            [-0.5, 0.25],
        ]
    )
    features, names = build_observables(states)

    assert names == ["u", "v", "u^2", "uv", "v^2", "u^3"]
    assert features.shape == (2, 6)
    np.testing.assert_allclose(features[0], [1.0, 2.0, 1.0, 2.0, 4.0, 1.0])


def test_fourier_observable_dictionary_contains_trig_features():
    states = np.array([[0.0, np.pi / 2]])
    features, names = build_observables(states, mode="fourier")

    assert names == ["sin(u)", "cos(u)", "sin(v)", "cos(v)"]
    assert features.shape == (1, 4)
    np.testing.assert_allclose(features[0], [0.0, 1.0, 1.0, 0.0], atol=1e-7)


def test_whitened_koopman_spectrum_is_sorted_and_bounded():
    analysis = analyze_case(
        SlowManifoldConfig(total_steps=1600, burn_in=300, seed=11),
        alpha=1.0,
        lag_steps=1,
    )
    singular_values = analysis["singular_values"]

    assert singular_values.ndim == 1
    assert singular_values.size > 1
    assert np.all(singular_values[:-1] >= singular_values[1:] - 1e-10)
    assert np.all(singular_values >= -1e-10)
    assert np.all(singular_values <= 1.0 + 1e-8)


def test_rank_one_average_reversibility_exceeds_full_average():
    analysis = analyze_case(
        SlowManifoldConfig(total_steps=2400, burn_in=400, seed=13),
        alpha=1.0,
        lag_steps=1,
    )
    singular_values = analysis["singular_values"]

    rank_one = average_reversibility(singular_values, rank=1, alpha=1.0)
    full_rank = average_reversibility(singular_values, rank=singular_values.size, alpha=1.0)

    assert rank_one > full_rank


def test_mode_comparison_includes_fourier_and_reports_truncation():
    result = analyze_observable_modes(
        ["identity", "polynomial", "fourier"],
        config=SlowManifoldConfig(total_steps=1600, burn_in=300, seed=17, observable_mode="identity"),
    )

    assert list(result["comparison_table"]["观测模式"]) == ["identity", "polynomial", "fourier"]
    assert all("截断维数" in result["comparison_table"].columns for _ in [0])
    assert all(entry["truncation_rank"] >= 1 for entry in result["mode_results"].values())


def test_choose_truncation_rank_prefers_first_mode_when_only_first_channel_is_strong():
    singular_values = np.array([0.95, 0.2, 0.1])
    assert choose_truncation_rank(singular_values, alpha=1.0) == 1
