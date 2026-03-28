from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pysindy as ps


@dataclass(frozen=True)
class SingleBasinConfig:
    total_steps: int = 1200
    burn_in: int = 100
    seed: int = 7
    a: float = 0.72
    b: float = 0.41
    c: float = 0.63
    sigma_x1: float = 0.03
    sigma_x2: float = 0.04
    x1_0: float = 1.1
    x2_0: float = -0.7


def deterministic_step_x(
    x1: np.ndarray,
    x2: np.ndarray,
    config: SingleBasinConfig,
) -> tuple[np.ndarray, np.ndarray]:
    x1_arr = np.asarray(x1, dtype=float)
    x2_arr = np.asarray(x2, dtype=float)
    x1_next = config.a * x1_arr
    x2_next = config.b * x2_arr + config.c * x1_arr ** 2
    return x1_next, x2_next


def noisy_step_x(
    x1: np.ndarray,
    x2: np.ndarray,
    config: SingleBasinConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x1_next, x2_next = deterministic_step_x(x1, x2, config)
    x1_next = x1_next + config.sigma_x1 * rng.normal(size=np.shape(x1_next))
    x2_next = x2_next + config.sigma_x2 * rng.normal(size=np.shape(x2_next))
    return x1_next, x2_next


def simulate_single_basin_system(
    config: SingleBasinConfig,
    *,
    initial_state: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    states = np.zeros((config.total_steps, 2), dtype=float)
    if initial_state is None:
        states[0] = np.array([config.x1_0, config.x2_0], dtype=float)
    else:
        init = np.asarray(initial_state, dtype=float)
        if init.shape != (2,):
            raise ValueError(f"initial_state must have shape (2,), got {init.shape}")
        states[0] = init

    for step in range(config.total_steps - 1):
        states[step + 1, 0], states[step + 1, 1] = noisy_step_x(states[step, 0], states[step, 1], config, rng)

    kept = states[config.burn_in :]
    return {
        "states": kept,
        "x1": kept[:, 0],
        "x2": kept[:, 1],
        "time": np.arange(kept.shape[0]),
    }


def simulate_single_basin_ensemble(
    config: SingleBasinConfig,
    *,
    n_trajectories: int = 8,
    x1_range: tuple[float, float] = (-1.4, 1.4),
    x2_range: tuple[float, float] = (-1.2, 1.2),
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    ensemble = np.zeros((n_trajectories, config.total_steps - config.burn_in, 2), dtype=float)
    initials = np.column_stack(
        [
            rng.uniform(*x1_range, size=n_trajectories),
            rng.uniform(*x2_range, size=n_trajectories),
        ]
    )

    for idx in range(n_trajectories):
        sample = simulate_single_basin_system(config, initial_state=initials[idx])
        ensemble[idx] = sample["states"]

    return {
        "trajectories": ensemble,
        "initial_states": initials,
    }


def build_observables(states: np.ndarray, *, mode: str = "identity") -> tuple[np.ndarray, list[str]]:
    samples = np.asarray(states, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError(f"Expected states with shape (n, 2), got {samples.shape}")

    x1 = samples[:, 0]
    x2 = samples[:, 1]

    if mode == "identity":
        features = np.column_stack([x1, x2])
        names = ["x1", "x2"]
    elif mode == "closure":
        features = np.column_stack([x1, x2, x1 ** 2])
        names = ["x1", "x2", "x1^2"]
    elif mode == "quadratic_full":
        features = np.column_stack([x1, x2, x1 ** 2, x1 * x2, x2 ** 2])
        names = ["x1", "x2", "x1^2", "x1x2", "x2^2"]
    else:
        raise ValueError(f"Unsupported observable mode: {mode}")
    return features, names


def _fit_linear_discrete_sindy(observables: np.ndarray, feature_names: list[str]) -> ps.SINDy:
    library = ps.PolynomialLibrary(
        degree=1,
        include_bias=False,
        include_interaction=False,
    )
    optimizer = ps.STLSQ(
        threshold=0.0,
        alpha=0.0,
        normalize_columns=False,
    )
    model = ps.SINDy(
        feature_library=library,
        optimizer=optimizer,
        discrete_time=True,
    )
    model.fit(observables, t=1)
    model.feature_names = list(feature_names)
    return model


def fit_global_observation_model(states: np.ndarray, *, observable_mode: str = "identity") -> dict[str, Any]:
    observables, names = build_observables(states, mode=observable_mode)
    if observables.shape[0] < 3:
        raise ValueError("Need at least three time steps to fit a discrete-time model.")

    model = _fit_linear_discrete_sindy(observables, names)
    current = observables[:-1]
    future = observables[1:]
    predicted = np.asarray(model.predict(current), dtype=float)
    residuals = future - predicted
    sigma = np.cov(residuals, rowvar=False)
    if np.ndim(sigma) == 0:
        sigma = np.array([[float(sigma)]], dtype=float)

    coefficients = np.asarray(model.coefficients(), dtype=float)
    residual_mse = float(np.mean(residuals ** 2))
    residual_trace = float(np.trace(sigma))
    residual_trace_per_dim = residual_trace / sigma.shape[0]

    return {
        "model": model,
        "feature_names": names,
        "observables": observables,
        "current": current,
        "future": future,
        "predicted": predicted,
        "residuals": residuals,
        "A": coefficients,
        "Sigma": np.asarray(sigma, dtype=float),
        "residual_mse": residual_mse,
        "residual_trace": residual_trace,
        "residual_trace_per_dim": float(residual_trace_per_dim),
    }


def compare_observation_modes(
    states: np.ndarray,
    *,
    modes: tuple[str, ...] = ("identity", "closure"),
) -> dict[str, Any]:
    mode_results: dict[str, dict[str, Any]] = {}
    for mode in modes:
        analysis = fit_global_observation_model(states, observable_mode=mode)
        mode_results[mode] = analysis

    selected_mode = min(
        modes,
        key=lambda mode: (
            mode_results[mode]["residual_trace_per_dim"],
            mode_results[mode]["residual_mse"],
        ),
    )

    return {
        "selected_mode": selected_mode,
        "mode_results": mode_results,
    }


def project_macro_observables(
    states: np.ndarray,
    W: np.ndarray,
    *,
    observable_mode: str = "closure",
) -> np.ndarray:
    observables, _ = build_observables(states, mode=observable_mode)
    projection = np.asarray(W, dtype=float)
    if projection.ndim != 2:
        raise ValueError(f"W must have shape (k, d), got {projection.shape}")
    if projection.shape[1] != observables.shape[1]:
        raise ValueError(
            f"W expects observable dimension {projection.shape[1]}, "
            f"but mode '{observable_mode}' produces {observables.shape[1]}"
        )
    return observables @ projection.T


def _sorted_eigh(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(np.asarray(matrix, dtype=float))
    order = np.argsort(values)[::-1]
    values = np.real_if_close(values[order])
    vectors = np.real_if_close(vectors[:, order])
    return values, vectors


def compute_gis_singular_value_spectra(
    A: np.ndarray,
    Sigma: np.ndarray,
    *,
    eps: float = 1e-10,
) -> dict[str, np.ndarray | float]:
    A_array = np.asarray(A, dtype=float)
    sigma_array = np.asarray(Sigma, dtype=float)

    if A_array.ndim != 2 or A_array.shape[0] != A_array.shape[1]:
        raise ValueError(f"A must be a square matrix, got shape {A_array.shape}")
    if sigma_array.ndim != 2 or sigma_array.shape[0] != sigma_array.shape[1]:
        raise ValueError(f"Sigma must be a square matrix, got shape {sigma_array.shape}")
    if sigma_array.shape != A_array.shape:
        raise ValueError(f"A and Sigma must have the same shape, got {A_array.shape} and {sigma_array.shape}")

    sigma_sym = 0.5 * (sigma_array + sigma_array.T)
    sigma_inv = np.linalg.pinv(sigma_sym, rcond=eps)
    determinism_matrix = 0.5 * (sigma_inv + sigma_inv.T)
    nondegeneracy_matrix = A_array.T @ determinism_matrix @ A_array
    nondegeneracy_matrix = 0.5 * (nondegeneracy_matrix + nondegeneracy_matrix.T)

    det_sv = np.linalg.svd(determinism_matrix, compute_uv=False)
    nondeg_sv = np.linalg.svd(nondegeneracy_matrix, compute_uv=False)
    det_sv = np.sort(np.real_if_close(det_sv))[::-1]
    nondeg_sv = np.sort(np.real_if_close(nondeg_sv))[::-1]

    det_positive = det_sv[det_sv > eps]
    nondeg_positive = nondeg_sv[nondeg_sv > eps]

    return {
        "Sigma_inv": np.real_if_close(determinism_matrix),
        "A_T_Sigma_inv_A": np.real_if_close(nondegeneracy_matrix),
        "determinism_singular_values": np.real_if_close(det_sv),
        "nondegeneracy_singular_values": np.real_if_close(nondeg_sv),
        "determinism_log_pdet": float(np.sum(np.log(det_positive))) if det_positive.size else float("-inf"),
        "nondegeneracy_log_pdet": float(np.sum(np.log(nondeg_positive))) if nondeg_positive.size else float("-inf"),
    }


def compute_two_step_svd_coarse_graining(
    A: np.ndarray,
    Sigma: np.ndarray,
    *,
    eps: float = 1e-10,
    target_rank: int | None = None,
    first_stage_rank: int | None = None,
) -> dict[str, Any]:
    A_array = np.asarray(A, dtype=float)
    sigma_array = np.asarray(Sigma, dtype=float)

    if A_array.ndim != 2 or A_array.shape[0] != A_array.shape[1]:
        raise ValueError(f"A must be a square matrix, got shape {A_array.shape}")
    if sigma_array.ndim != 2 or sigma_array.shape[0] != sigma_array.shape[1]:
        raise ValueError(f"Sigma must be a square matrix, got shape {sigma_array.shape}")
    if sigma_array.shape != A_array.shape:
        raise ValueError(f"A and Sigma must have the same shape, got {A_array.shape} and {sigma_array.shape}")

    n = A_array.shape[0]
    if target_rank is not None and not 1 <= target_rank <= n:
        raise ValueError(f"target_rank must be in [1, {n}], got {target_rank}")
    if first_stage_rank is not None and not 1 <= first_stage_rank <= 2 * n:
        raise ValueError(f"first_stage_rank must be in [1, {2 * n}], got {first_stage_rank}")

    sigma_sym = 0.5 * (sigma_array + sigma_array.T)
    sigma_inv = np.linalg.pinv(sigma_sym, rcond=eps)
    sigma_inv = 0.5 * (sigma_inv + sigma_inv.T)
    nondegeneracy = A_array.T @ sigma_inv @ A_array
    nondegeneracy = 0.5 * (nondegeneracy + nondegeneracy.T)

    s_vals, U = _sorted_eigh(nondegeneracy)
    k_vals, V = _sorted_eigh(sigma_inv)

    combined_entries: list[dict[str, Any]] = []
    for idx, value in enumerate(s_vals):
        combined_entries.append(
            {
                "source": "nondegeneracy",
                "source_index": idx,
                "singular_value": float(value),
                "vector": np.asarray(U[:, idx], dtype=float),
            }
        )
    for idx, value in enumerate(k_vals):
        combined_entries.append(
            {
                "source": "determinism",
                "source_index": idx,
                "singular_value": float(value),
                "vector": np.asarray(V[:, idx], dtype=float),
            }
        )
    combined_entries.sort(key=lambda entry: entry["singular_value"], reverse=True)

    if first_stage_rank is None:
        if target_rank is not None:
            rank_stage1 = n
        else:
            rank_stage1 = sum(entry["singular_value"] > eps for entry in combined_entries)
    else:
        rank_stage1 = first_stage_rank

    kept_stage1 = combined_entries[:rank_stage1]
    if not kept_stage1:
        raise ValueError("No singular directions were retained in the first SVD stage.")

    tilde_U1 = np.column_stack([entry["vector"] for entry in kept_stage1])
    tilde_s1 = np.array([entry["singular_value"] for entry in kept_stage1], dtype=float)
    lifted = tilde_U1 @ np.diag(tilde_s1)
    hat_U, hat_s, hat_Vt = np.linalg.svd(lifted, full_matrices=True)

    if target_rank is None:
        rank_stage2 = int(np.sum(hat_s > eps))
    else:
        rank_stage2 = target_rank

    if rank_stage2 < 1:
        raise ValueError("No coarse-graining directions were retained in the second SVD stage.")

    W = hat_U[:, :rank_stage2].T

    return {
        "W": np.real_if_close(W),
        "Sigma_inv": np.real_if_close(sigma_inv),
        "A_T_Sigma_inv_A": np.real_if_close(nondegeneracy),
        "determinism_singular_values": np.real_if_close(k_vals),
        "nondegeneracy_singular_values": np.real_if_close(s_vals),
        "combined_entries": [
            {
                "source": entry["source"],
                "source_index": entry["source_index"],
                "singular_value": entry["singular_value"],
                "vector": np.real_if_close(entry["vector"]),
            }
            for entry in kept_stage1
        ],
        "combined_singular_values": np.real_if_close(tilde_s1),
        "first_stage_vectors": np.real_if_close(tilde_U1),
        "first_stage_weighted_matrix": np.real_if_close(lifted),
        "second_stage_left_vectors": np.real_if_close(hat_U),
        "second_stage_singular_values": np.real_if_close(hat_s),
        "second_stage_right_vectors": np.real_if_close(hat_Vt.T),
        "effective_rank_stage1": int(rank_stage1),
        "effective_rank_stage2": int(rank_stage2),
    }


__all__ = [
    "SingleBasinConfig",
    "build_observables",
    "compare_observation_modes",
    "compute_gis_singular_value_spectra",
    "compute_two_step_svd_coarse_graining",
    "deterministic_step_x",
    "fit_global_observation_model",
    "noisy_step_x",
    "project_macro_observables",
    "simulate_single_basin_ensemble",
    "simulate_single_basin_system",
]
