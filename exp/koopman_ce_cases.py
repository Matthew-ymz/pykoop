from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov


Array = np.ndarray


@dataclass(frozen=True)
class CaseConfig:
    name: str
    A: Array
    Sigma: Array
    seed: int
    n_steps: int
    burn_in: int
    alpha: float = 1.0
    epsilon: float = 0.0
    lift: str = "identity"
    future_lift: str | None = None
    center: bool = True
    score_rank: int = 3


def _symmetrize(matrix: Array) -> Array:
    return 0.5 * (matrix + matrix.T)


def _orthogonal_from_seed(seed: int) -> Array:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q = q @ np.diag(signs)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def matrix_inv_sqrt_psd(matrix: Array, eps: float = 1e-10) -> tuple[Array, Array, Array]:
    evals, vecs = np.linalg.eigh(_symmetrize(matrix))
    clipped = np.clip(evals, eps, None)
    sqrt = vecs @ np.diag(np.sqrt(clipped)) @ vecs.T
    inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(clipped)) @ vecs.T
    return sqrt, inv_sqrt, clipped


def log_pdet(matrix: Array, eps: float = 1e-10) -> float:
    evals = np.linalg.eigvalsh(_symmetrize(matrix))
    positive = evals[evals > eps]
    if positive.size == 0:
        return float("-inf")
    return float(np.sum(np.log(positive)))


def split_whitened_operator(
    kbar: Array,
    eps: float = 1e-8,
) -> Dict[str, Array]:
    u, singular_values, vt = np.linalg.svd(kbar, full_matrices=False)
    rho2 = np.clip(singular_values**2, 0.0, 1.0 - eps)
    det_vals = 1.0 / (1.0 - rho2)
    nondeg_vals = rho2 / (1.0 - rho2)
    v = vt.T
    dk = v @ np.diag(det_vals) @ v.T
    nk = u @ np.diag(nondeg_vals) @ u.T
    return {
        "U": u,
        "V": v,
        "Vt": vt,
        "singular_values": singular_values,
        "rho2": rho2,
        "Dk_white": _symmetrize(dk),
        "Nk_white": _symmetrize(nk),
        "determinism_eigs": det_vals,
        "nondegeneracy_eigs": nondeg_vals,
    }


def channel_scores_from_singular_values(
    singular_values: Array,
    alpha: float = 1.0,
    rank: int | None = None,
    eps: float = 1e-8,
) -> Array:
    if rank is None:
        rank = singular_values.shape[0]
    rho2 = np.clip(np.asarray(singular_values[:rank]) ** 2, eps, 1.0 - eps)
    coeff_n = 0.5 - alpha / 4.0
    coeff_d = alpha / 4.0
    return coeff_n * np.log(rho2 / (1.0 - rho2)) + coeff_d * np.log(1.0 / (1.0 - rho2))


def stationary_covariance(A: Array, Sigma: Array) -> Array:
    return _symmetrize(solve_discrete_lyapunov(A.T, Sigma))


def canonical_correlations_from_linear_system(A: Array, Sigma: Array) -> Dict[str, Array]:
    P = stationary_covariance(A, Sigma)
    P_sqrt, P_inv_sqrt, _ = matrix_inv_sqrt_psd(P)
    Kbar = P_sqrt @ A @ P_inv_sqrt
    split = split_whitened_operator(Kbar)
    return {
        "P": P,
        "Kbar": Kbar,
        "determinism_white": split["Dk_white"],
        "nondegeneracy_white": split["Nk_white"],
        "singular_values": split["singular_values"],
    }


def liu2025_standardized_operator(A: Array, Sigma: Array, eps: float = 1e-10) -> Array:
    sigma_sqrt, sigma_inv_sqrt, _ = matrix_inv_sqrt_psd(Sigma, eps=eps)
    return sigma_inv_sqrt @ A @ sigma_sqrt


def liu2025_standardized_singular_values(A: Array, Sigma: Array, eps: float = 1e-10) -> Array:
    standardized = liu2025_standardized_operator(A, Sigma, eps=eps)
    return np.linalg.svd(standardized, compute_uv=False)


def liu2025_native_nondegeneracy_matrix(A: Array, Sigma: Array, eps: float = 1e-10) -> Array:
    sigma_inv = np.linalg.pinv(_symmetrize(Sigma), rcond=eps)
    return _symmetrize(A.T @ sigma_inv @ A)


def liu2025_native_nondegeneracy_singular_values(A: Array, Sigma: Array, eps: float = 1e-10) -> Array:
    native = liu2025_native_nondegeneracy_matrix(A, Sigma, eps=eps)
    return np.linalg.svd(native, compute_uv=False)


def liu2025_determinism_singular_values(Sigma: Array, eps: float = 1e-10) -> Array:
    sigma_inv = np.linalg.pinv(_symmetrize(Sigma), rcond=eps)
    return np.linalg.svd(_symmetrize(sigma_inv), compute_uv=False)


def liu2025_log_gamma_gis(A: Array, Sigma: Array, alpha: float = 1.0, eps: float = 1e-10) -> float:
    n = A.shape[0]
    native = liu2025_native_nondegeneracy_matrix(A, Sigma, eps=eps)
    sigma_inv = np.linalg.pinv(_symmetrize(Sigma), rcond=eps)
    coeff_n = 0.5 - alpha / 4.0
    coeff_d = alpha / 4.0
    constant = n / 2.0 * np.log(2.0 * np.pi * alpha)
    return constant + coeff_n * log_pdet(native, eps=eps) + coeff_d * log_pdet(sigma_inv, eps=eps)


def koopman_ce_total_score_from_kbar(kbar: Array, alpha: float = 1.0, eps: float = 1e-10) -> float:
    split = split_whitened_operator(kbar, eps=eps)
    coeff_n = 0.5 - alpha / 4.0
    coeff_d = alpha / 4.0
    return coeff_n * log_pdet(split["Nk_white"], eps=eps) + coeff_d * log_pdet(split["Dk_white"], eps=eps)


def koopman_ce_total_score_from_linear_system(A: Array, Sigma: Array, alpha: float = 1.0, eps: float = 1e-10) -> float:
    theory = canonical_correlations_from_linear_system(A, Sigma)
    return koopman_ce_total_score_from_kbar(theory["Kbar"], alpha=alpha, eps=eps)


def marginal_correction_for_linear_system(A: Array, Sigma: Array, alpha: float = 1.0, eps: float = 1e-10) -> float:
    P = stationary_covariance(A, Sigma)
    C11 = _symmetrize(A @ P @ A.T + Sigma)
    n = A.shape[0]
    return (
        n / 2.0 * np.log(2.0 * np.pi * alpha)
        - (0.5 - alpha / 4.0) * log_pdet(P, eps=eps)
        - (alpha / 4.0) * log_pdet(C11, eps=eps)
    )


def _quadratic_terms(x: Array) -> Array:
    x1 = x[:, [0]]
    x2 = x[:, [1]]
    x3 = x[:, [2]]
    return np.hstack(
        [
            x1**2,
            x1 * x2,
            x1 * x3,
            x2**2,
            x2 * x3,
            x3**2,
        ]
    )


def lift_observables(x: Array, lift: str = "identity") -> Array:
    if lift == "identity":
        return x.copy()
    if lift == "quadratic":
        return np.hstack([x, _quadratic_terms(x)])
    raise ValueError(f"Unsupported lift: {lift}")


def nonlinear_feature_map(x: Array) -> Array:
    x = np.clip(x, -3.0, 3.0)
    x1 = x[..., 0]
    x2 = x[..., 1]
    x3 = x[..., 2]
    return np.stack([x1**2 - x2**2, x1 * x2, x2 * x3], axis=-1)


def simulate_case(config: CaseConfig) -> Array:
    rng = np.random.default_rng(config.seed)
    total_steps = config.n_steps + config.burn_in
    P = stationary_covariance(config.A, config.Sigma)
    x = rng.multivariate_normal(np.zeros(config.A.shape[0]), P)
    xs = np.zeros((total_steps, config.A.shape[0]))
    for t in range(total_steps):
        xs[t] = x
        noise = rng.multivariate_normal(np.zeros(config.A.shape[0]), config.Sigma)
        x = x @ config.A + config.epsilon * nonlinear_feature_map(x) + noise
    return xs[config.burn_in :]


def fit_whitened_koopman(
    z_current: Array,
    z_future: Array | None = None,
    center: bool = True,
    ridge: float = 1e-10,
    eps: float = 1e-10,
) -> Dict[str, Array]:
    if z_future is None:
        z_future = z_current

    x = z_current[:-1]
    y = z_future[1:]
    if center:
        x = x - np.mean(x, axis=0, keepdims=True)
        y = y - np.mean(y, axis=0, keepdims=True)

    n_pairs = x.shape[0]
    C00 = (x.T @ x) / n_pairs
    C01 = (x.T @ y) / n_pairs
    C11 = (y.T @ y) / n_pairs

    C00_reg = C00 + ridge * np.eye(C00.shape[0])
    A = np.linalg.pinv(C00_reg) @ C01
    _, C00_inv_sqrt, _ = matrix_inv_sqrt_psd(C00, eps=eps)
    _, C11_inv_sqrt, _ = matrix_inv_sqrt_psd(C11, eps=eps)
    Kbar = C00_inv_sqrt @ C01 @ C11_inv_sqrt
    split = split_whitened_operator(Kbar)
    return {
        "A": A,
        "C00": C00,
        "C01": C01,
        "C11": C11,
        "Kbar": Kbar,
        **split,
    }


def select_state_aligned_channels(
    U: Array,
    V: Array,
    state_dim: int = 3,
    rank: int = 3,
) -> Array:
    state_dim = min(state_dim, U.shape[0], V.shape[0])
    left_energy = np.sum(U[:state_dim, :] ** 2, axis=0)
    right_energy = np.sum(V[:state_dim, :] ** 2, axis=0)
    alignment = left_energy + right_energy
    order = np.argsort(-alignment)
    chosen = np.sort(order[:rank])
    return chosen


def _base_case_b_matrices() -> tuple[Array, Array]:
    eigvals = np.array([0.97, 0.80, 0.25])
    variances = np.array([0.03**2, 0.08**2, 0.25**2])
    QA = _orthogonal_from_seed(11)
    QS = _orthogonal_from_seed(29)
    A = QA @ np.diag(eigvals) @ QA.T
    Sigma = QS @ np.diag(variances) @ QS.T
    return _symmetrize(A), _symmetrize(Sigma)


def build_case_a_gis_family(alpha: float = 1.0) -> list[CaseConfig]:
    systems: list[CaseConfig] = []

    # Family 1: diagonal A with isotropic Sigma across broad noise/amplitude settings
    diag_sets = [
        [0.98, 0.85, 0.20],
        [0.95, 0.70, 0.35],
        [0.90, 0.88, 0.15],
        [0.75, 0.55, 0.25],
        [0.99, 0.60, 0.10],
    ]
    sigma_scales = [0.03, 0.06, 0.10, 0.18]
    seed = 100
    for diag in diag_sets:
        A = np.diag(diag)
        for sigma in sigma_scales:
            systems.append(
                CaseConfig(
                    name=f"case_a_diag_sigma_{sigma:.2f}",
                    A=A,
                    Sigma=(sigma**2) * np.eye(3),
                    seed=seed,
                    n_steps=6000,
                    burn_in=1000,
                    alpha=alpha,
                )
            )
            seed += 1

    # Family 2: rotated A, isotropic noise
    rot_diag_sets = [
        [0.97, 0.82, 0.30],
        [0.92, 0.78, 0.40],
        [0.88, 0.66, 0.22],
    ]
    rot_sigmas = [0.04, 0.09, 0.16]
    for idx, diag in enumerate(rot_diag_sets):
        Q = _orthogonal_from_seed(200 + idx)
        A = _symmetrize(Q @ np.diag(diag) @ Q.T)
        for sigma in rot_sigmas:
            systems.append(
                CaseConfig(
                    name=f"case_a_rot_sigma_{sigma:.2f}_{idx}",
                    A=A,
                    Sigma=(sigma**2) * np.eye(3),
                    seed=seed,
                    n_steps=6000,
                    burn_in=1000,
                    alpha=alpha,
                )
            )
            seed += 1

    # Family 3: anisotropic Sigma, included to show native GIS scale variation explicitly
    anis_sets = [
        ([0.97, 0.80, 0.25], [0.03**2, 0.08**2, 0.20**2]),
        ([0.93, 0.75, 0.18], [0.05**2, 0.12**2, 0.18**2]),
        ([0.85, 0.62, 0.40], [0.02**2, 0.06**2, 0.15**2]),
    ]
    for idx, (diag, var_diag) in enumerate(anis_sets):
        QA = _orthogonal_from_seed(300 + idx)
        QS = _orthogonal_from_seed(330 + idx)
        A = _symmetrize(QA @ np.diag(diag) @ QA.T)
        Sigma = _symmetrize(QS @ np.diag(var_diag) @ QS.T)
        systems.append(
            CaseConfig(
                name=f"case_a_anisotropic_{idx}",
                A=A,
                Sigma=Sigma,
                seed=seed,
                n_steps=6000,
                burn_in=1000,
                alpha=alpha,
            )
        )
        seed += 1

    return systems


def make_case_a_config(
    seed: int = 0,
    n_steps: int = 20_000,
    burn_in: int = 4_000,
    alpha: float = 1.0,
) -> CaseConfig:
    A = np.diag([0.97, 0.80, 0.25])
    sigma = 0.08
    Sigma = (sigma**2) * np.eye(3)
    return CaseConfig(
        name="case_a_standardized_gis",
        A=A,
        Sigma=Sigma,
        seed=seed,
        n_steps=n_steps,
        burn_in=burn_in,
        alpha=alpha,
    )


def make_case_b_config(
    seed: int = 0,
    n_steps: int = 20_000,
    burn_in: int = 4_000,
    alpha: float = 1.0,
) -> CaseConfig:
    A, Sigma = _base_case_b_matrices()
    return CaseConfig(
        name="case_b_full_gis",
        A=A,
        Sigma=Sigma,
        seed=seed,
        n_steps=n_steps,
        burn_in=burn_in,
        alpha=alpha,
    )


def make_case_c_config(
    seed: int = 0,
    n_steps: int = 20_000,
    burn_in: int = 4_000,
    alpha: float = 1.0,
    epsilon: float = 0.1,
    lift: str = "identity",
) -> CaseConfig:
    A, Sigma = _base_case_b_matrices()
    return CaseConfig(
        name="case_c_weakly_nonlinear",
        A=A,
        Sigma=Sigma,
        seed=seed,
        n_steps=n_steps,
        burn_in=burn_in,
        alpha=alpha,
        epsilon=epsilon,
        lift=lift,
        future_lift="identity",
    )


def summarize_case(result: Dict[str, Dict[str, Array]]) -> pd.DataFrame:
    metrics = result["metrics"]
    return pd.DataFrame(
        [
            {
                "case": result["config"]["name"],
                "lift": result["config"]["lift"],
                "epsilon": result["config"]["epsilon"],
                "gamma_empirical": metrics["gamma_empirical"],
                "gamma_theory": metrics["gamma_theory"],
                "gamma_gap": metrics["gamma_gap"],
                "fro_D": metrics["fro_D_error"],
                "fro_N": metrics["fro_N_error"],
            }
        ]
    )


def build_case_a_score_comparison_dataframe(alpha: float = 1.0) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for config in build_case_a_gis_family(alpha=alpha):
        standardized_sv = liu2025_standardized_singular_values(config.A, config.Sigma)
        native_sv = liu2025_native_nondegeneracy_singular_values(config.A, config.Sigma)
        rows.append(
            {
                "name": config.name,
                "liu_log_gamma": liu2025_log_gamma_gis(config.A, config.Sigma, alpha=alpha),
                "koopman_total_score": koopman_ce_total_score_from_linear_system(config.A, config.Sigma, alpha=alpha),
                "marginal_correction": marginal_correction_for_linear_system(config.A, config.Sigma, alpha=alpha),
                "top_whitened_sv": float(np.max(standardized_sv)),
                "top_native_sv": float(np.max(native_sv)),
                "trace_sigma": float(np.trace(config.Sigma)),
                "is_isotropic_sigma": bool(np.allclose(config.Sigma, np.eye(config.Sigma.shape[0]) * config.Sigma[0, 0])),
            }
        )
    return pd.DataFrame(rows)


def plot_score_to_score_comparison(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    color_col: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    if color_col is None:
        scatter = ax.scatter(frame[x_col], frame[y_col], s=45, alpha=0.8)
    else:
        colors = frame[color_col].astype(float)
        scatter = ax.scatter(frame[x_col], frame[y_col], c=colors, cmap="viridis", s=55, alpha=0.85)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(color_col)

    x = frame[x_col].to_numpy()
    y = frame[y_col].to_numpy()
    coeff = np.polyfit(x, y, deg=1)
    line_x = np.linspace(np.min(x), np.max(x), 100)
    line_y = coeff[0] * line_x + coeff[1]
    ax.plot(line_x, line_y, "k--", linewidth=1)
    corr = np.corrcoef(x, y)[0, 1]
    ax.text(0.04, 0.96, f"corr = {corr:.4f}", transform=ax.transAxes, va="top")
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def run_case(config: CaseConfig | Dict[str, object]) -> Dict[str, Dict[str, Array]]:
    if isinstance(config, dict):
        config = CaseConfig(**config)

    trajectory = simulate_case(config)
    future_lift = config.future_lift or config.lift
    lifted = lift_observables(trajectory, lift=config.lift)
    future_observables = lift_observables(trajectory, lift=future_lift)
    empirical = fit_whitened_koopman(lifted, z_future=future_observables, center=config.center)

    theory = canonical_correlations_from_linear_system(config.A, config.Sigma)
    rank = min(config.score_rank, empirical["singular_values"].shape[0], theory["singular_values"].shape[0])
    chosen_channels = select_state_aligned_channels(
        empirical["U"],
        empirical["V"],
        state_dim=config.A.shape[0],
        rank=rank,
    )
    empirical_scores = channel_scores_from_singular_values(
        empirical["singular_values"][chosen_channels],
        alpha=config.alpha,
        rank=rank,
    )
    theory_scores = channel_scores_from_singular_values(theory["singular_values"], alpha=config.alpha, rank=rank)

    metrics = {
        "rank": rank,
        "gamma_empirical": float(np.sum(empirical_scores)),
        "gamma_theory": float(np.sum(theory_scores)),
        "gamma_gap": float(np.sum(empirical_scores) - np.sum(theory_scores)),
        "gamma_gap_abs": float(abs(np.sum(empirical_scores) - np.sum(theory_scores))),
        "fro_D_error": float(
            np.linalg.norm(empirical["Dk_white"][: theory["determinism_white"].shape[0], : theory["determinism_white"].shape[1]] - theory["determinism_white"])
        )
        if empirical["Dk_white"].shape == theory["determinism_white"].shape
        else np.nan,
        "fro_N_error": float(
            np.linalg.norm(empirical["Nk_white"][: theory["nondegeneracy_white"].shape[0], : theory["nondegeneracy_white"].shape[1]] - theory["nondegeneracy_white"])
        )
        if empirical["Nk_white"].shape == theory["nondegeneracy_white"].shape
        else np.nan,
    }

    return {
        "config": {
            "name": config.name,
            "lift": config.lift,
            "epsilon": config.epsilon,
            "alpha": config.alpha,
            "seed": config.seed,
        },
        "trajectory": trajectory,
        "lifted": lifted,
        "future_observables": future_observables,
        "empirical": {
            "Kbar": empirical["Kbar"],
            "Dk_white": empirical["Dk_white"],
            "Nk_white": empirical["Nk_white"],
            "singular_values": empirical["singular_values"],
            "selected_channels": chosen_channels,
            "channel_scores": empirical_scores,
            "C00": empirical["C00"],
            "C11": empirical["C11"],
        },
        "theory": {
            "Kbar": theory["Kbar"],
            "determinism_white": theory["determinism_white"],
            "nondegeneracy_white": theory["nondegeneracy_white"],
            "singular_values": theory["singular_values"],
            "channel_scores": theory_scores,
            "P": theory["P"],
        },
        "metrics": metrics,
    }


def run_case_c_sweep(
    eps_grid: Iterable[float],
    seed: int = 0,
    n_steps: int = 20_000,
    burn_in: int = 4_000,
) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    for lift in ["identity", "quadratic"]:
        for epsilon in eps_grid:
            result = run_case(
                make_case_c_config(
                    seed=seed,
                    n_steps=n_steps,
                    burn_in=burn_in,
                    epsilon=epsilon,
                    lift=lift,
                )
            )
            rows.append(
                {
                    "lift": lift,
                    "epsilon": epsilon,
                    "gamma_gap_abs": result["metrics"]["gamma_gap_abs"],
                    "gamma_empirical": result["metrics"]["gamma_empirical"],
                    "gamma_theory": result["metrics"]["gamma_theory"],
                }
            )
    return pd.DataFrame(rows)


def plot_case_alignment(result: Dict[str, Dict[str, Array]]) -> tuple[plt.Figure, Array]:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    panels = [
        (result["theory"]["Kbar"], result["empirical"]["Kbar"], r"$\bar K$"),
        (result["theory"]["determinism_white"], result["empirical"]["Dk_white"], r"$D_K$"),
        (result["theory"]["nondegeneracy_white"], result["empirical"]["Nk_white"], r"$N_K$"),
    ]

    for ax, (theory_matrix, empirical_matrix, label) in zip(axes, panels):
        theory_vals = np.ravel(theory_matrix)
        empirical_vals = np.ravel(empirical_matrix[: theory_matrix.shape[0], : theory_matrix.shape[1]])
        ax.scatter(theory_vals, empirical_vals, s=18, alpha=0.8)
        bounds = [
            min(theory_vals.min(), empirical_vals.min()),
            max(theory_vals.max(), empirical_vals.max()),
        ]
        ax.plot(bounds, bounds, "k--", linewidth=1)
        ax.set_xlabel(f"Theory {label}")
        ax.set_ylabel(f"Empirical {label}")

    fig.tight_layout()
    return fig, axes


def plot_koopman_vs_liu2025_spectrum(
    koopman_singular_values: Array,
    liu_singular_values: Array,
) -> tuple[plt.Figure, plt.Axes]:
    koopman_singular_values = np.asarray(koopman_singular_values)
    liu_singular_values = np.asarray(liu_singular_values)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(koopman_singular_values, liu_singular_values, s=55, alpha=0.85)
    for idx, (x, y) in enumerate(zip(koopman_singular_values, liu_singular_values), start=1):
        ax.annotate(f"{idx}", (x, y), textcoords="offset points", xytext=(5, 5))

    lo = min(koopman_singular_values.min(), liu_singular_values.min())
    hi = max(koopman_singular_values.max(), liu_singular_values.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlabel(r"Singular values of dual-whitened $\bar K$")
    ax.set_ylabel("Liu 2025 standardized GIS singular values")
    ax.set_title("Case A spectrum correspondence")
    fig.tight_layout()
    return fig, ax


def plot_spectrum_comparison(
    x_values: Array,
    y_values: Array,
    xlabel: str,
    ylabel: str,
    title: str,
) -> tuple[plt.Figure, plt.Axes]:
    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(x_values, y_values, s=55, alpha=0.85)
    for idx, (x, y) in enumerate(zip(x_values, y_values), start=1):
        ax.annotate(f"{idx}", (x, y), textcoords="offset points", xytext=(5, 5))

    lo = min(x_values.min(), y_values.min())
    hi = max(x_values.max(), y_values.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax
