from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.toy_nonlinear.nonlinear_common_mode_analysis import (
    SingleBasinConfig,
    build_observables,
    compare_observation_modes,
    compute_gis_singular_value_spectra,
    compute_two_step_svd_coarse_graining,
    fit_global_observation_model,
    project_macro_observables,
    simulate_single_basin_system,
)


def test_single_basin_simulation_returns_expected_shapes_and_contracts():
    config = SingleBasinConfig(total_steps=80, burn_in=0, seed=5, sigma_x1=0.0, sigma_x2=0.0)
    result = simulate_single_basin_system(config, initial_state=np.array([1.0, -0.8]))

    assert result["states"].shape == (80, 2)
    assert result["x1"].shape == (80,)
    assert result["x2"].shape == (80,)
    assert np.linalg.norm(result["states"][-1]) < np.linalg.norm(result["states"][0])


def test_closure_observables_include_square_term_with_expected_values():
    states = np.array(
        [
            [2.0, -1.0],
            [-0.5, 0.25],
        ]
    )
    features, names = build_observables(states, mode="closure")

    assert names == ["x1", "x2", "x1^2"]
    assert features.shape == (2, 3)
    np.testing.assert_allclose(features[0], [2.0, -1.0, 4.0])


def test_global_observation_fit_recovers_expected_linear_proxy_in_closure_space():
    config = SingleBasinConfig(
        total_steps=600,
        burn_in=0,
        seed=9,
        sigma_x1=0.0,
        sigma_x2=0.0,
        a=0.72,
        b=0.41,
        c=0.63,
    )
    result = simulate_single_basin_system(config, initial_state=np.array([1.2, -0.4]))
    analysis = fit_global_observation_model(result["states"], observable_mode="closure")

    expected = np.array(
        [
            [config.a, 0.0, 0.0],
            [0.0, config.b, config.c],
            [0.0, 0.0, config.a**2],
        ]
    )

    assert analysis["A"].shape == (3, 3)
    np.testing.assert_allclose(analysis["A"], expected, atol=5e-3)
    np.testing.assert_allclose(analysis["Sigma"], np.zeros((3, 3)), atol=1e-8)


def test_mode_comparison_prefers_closure_and_reports_symmetric_covariance():
    config = SingleBasinConfig(total_steps=1000, burn_in=100, seed=21, sigma_x1=0.03, sigma_x2=0.04)
    result = simulate_single_basin_system(config, initial_state=np.array([1.0, -0.6]))
    comparison = compare_observation_modes(result["states"], modes=("identity", "closure"))

    assert comparison["selected_mode"] == "closure"
    assert (
        comparison["mode_results"]["closure"]["residual_trace_per_dim"]
        < comparison["mode_results"]["identity"]["residual_trace_per_dim"]
    )

    sigma = comparison["mode_results"]["closure"]["Sigma"]
    np.testing.assert_allclose(sigma, sigma.T, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(sigma) >= -1e-10)


def test_gis_spectra_match_closed_form_for_diagonal_example():
    A = np.diag([2.0, 3.0])
    sigma = np.diag([4.0, 9.0])

    spectra = compute_gis_singular_value_spectra(A, sigma)

    np.testing.assert_allclose(spectra["determinism_singular_values"], [0.25, 1.0 / 9.0], atol=1e-12)
    np.testing.assert_allclose(spectra["nondegeneracy_singular_values"], [1.0, 1.0], atol=1e-12)


def test_closure_fit_reports_sorted_gis_spectra_and_positive_scores():
    config = SingleBasinConfig(total_steps=1000, burn_in=100, seed=17, sigma_x1=0.03, sigma_x2=0.04)
    result = simulate_single_basin_system(config, initial_state=np.array([1.0, -0.6]))
    analysis = fit_global_observation_model(result["states"], observable_mode="closure")
    spectra = compute_gis_singular_value_spectra(analysis["A"], analysis["Sigma"])

    det_sv = spectra["determinism_singular_values"]
    nondeg_sv = spectra["nondegeneracy_singular_values"]

    assert det_sv.shape == (3,)
    assert nondeg_sv.shape == (3,)
    assert np.all(det_sv[:-1] >= det_sv[1:] - 1e-12)
    assert np.all(nondeg_sv[:-1] >= nondeg_sv[1:] - 1e-12)
    assert spectra["determinism_log_pdet"] > 0.0
    assert spectra["nondegeneracy_log_pdet"] > 0.0


def test_two_step_svd_coarse_graining_recovers_single_dominant_direction_in_diagonal_case():
    A = np.diag([2.0, 1.0])
    sigma = np.diag([1.0, 4.0])

    coarse = compute_two_step_svd_coarse_graining(A, sigma, target_rank=1)

    assert coarse["W"].shape == (1, 2)
    assert coarse["effective_rank_stage1"] == 2
    assert coarse["effective_rank_stage2"] == 1
    np.testing.assert_allclose(coarse["W"] @ coarse["W"].T, np.eye(1), atol=1e-12)
    np.testing.assert_allclose(np.abs(coarse["W"][0]), np.array([1.0, 0.0]), atol=1e-12)


def test_two_step_svd_coarse_graining_finds_two_distinct_macro_directions_in_closure_space():
    config = SingleBasinConfig(total_steps=1000, burn_in=100, seed=17, sigma_x1=0.03, sigma_x2=0.04)
    result = simulate_single_basin_system(config, initial_state=np.array([1.0, -0.6]))
    analysis = fit_global_observation_model(result["states"], observable_mode="closure")

    coarse = compute_two_step_svd_coarse_graining(analysis["A"], analysis["Sigma"], target_rank=2)

    W = coarse["W"]
    assert W.shape == (2, 3)
    assert coarse["effective_rank_stage1"] == 3
    assert coarse["effective_rank_stage2"] == 2
    np.testing.assert_allclose(W @ W.T, np.eye(2), atol=1e-10)

    dominant_indices = {int(np.argmax(np.abs(row))) for row in W}
    assert dominant_indices == {0, 2}


def test_project_macro_observables_matches_linear_projection_of_closure_features():
    states = np.array(
        [
            [2.0, -1.0],
            [-0.5, 0.25],
        ]
    )
    W = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, -1.0, 0.5],
        ]
    )

    macro = project_macro_observables(states, W, observable_mode="closure")

    expected = np.array(
        [
            [10.0, 3.0],
            [0.0, -0.125],
        ]
    )
    np.testing.assert_allclose(macro, expected, atol=1e-12)
