from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if "MPLCONFIGDIR" not in os.environ:
    mpl_cache_dir = REPO_ROOT / ".tmp" / "matplotlib"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


@dataclass
class SlowManifoldConfig:
    total_steps: int = 6000
    burn_in: int = 1000
    seed: int = 23
    epsilon: float = 0.045
    sigma_u: float = 0.085
    sigma_v: float = 0.18
    lambda_fast: float = 0.12
    cubic_coupling: float = 0.30
    u0: float = -0.8
    v0_offset: float = 0.6
    lag_steps: int = 1
    observable_mode: str = "polynomial"


class SlowManifoldObservableLibrary:
    def __init__(self, center: bool = True, mode: str = "polynomial", cubic_coupling: float = 0.30) -> None:
        self.center = center
        self.mode = mode
        self.cubic_coupling = cubic_coupling
        self.mean_: np.ndarray | None = None
        self.feature_names_: list[str] = []

    def fit(self, samples: np.ndarray) -> "SlowManifoldObservableLibrary":
        features, names = build_observables(samples, mode=self.mode, cubic_coupling=self.cubic_coupling)
        if self.center:
            self.mean_ = np.mean(features, axis=0, keepdims=True)
        else:
            self.mean_ = np.zeros((1, features.shape[1]))
        self.feature_names_ = names
        return self

    def transform(self, samples: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Observable library must be fit before transform.")
        features, _ = build_observables(samples, mode=self.mode, cubic_coupling=self.cubic_coupling)
        return features - self.mean_

    def get_feature_names(self) -> list[str]:
        return list(self.feature_names_)


def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 180
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]


def manifold_function(u: np.ndarray | float, cubic_coupling: float = 0.30) -> np.ndarray:
    u_array = np.asarray(u, dtype=float)
    return u_array + cubic_coupling * u_array ** 3


def simulate_slow_manifold_system(config: SlowManifoldConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    states = np.zeros((config.total_steps, 2), dtype=float)
    states[0, 0] = config.u0
    states[0, 1] = manifold_function(config.u0, config.cubic_coupling) + config.v0_offset

    for step in range(config.total_steps - 1):
        u_t, v_t = states[step]
        drift_u = -config.epsilon * (u_t ** 3 - u_t)
        u_next = u_t + drift_u + config.sigma_u * rng.normal()

        v_target = manifold_function(u_t, config.cubic_coupling)
        v_next = v_target + config.lambda_fast * (v_t - v_target) + config.sigma_v * rng.normal()
        states[step + 1] = [u_next, v_next]

    kept_states = states[config.burn_in :]
    return {
        "states": kept_states,
        "u": kept_states[:, 0],
        "v": kept_states[:, 1],
        "time": np.arange(kept_states.shape[0]),
    }


def build_observables(
    states: np.ndarray,
    *,
    mode: str = "polynomial",
    cubic_coupling: float = 0.30,
) -> tuple[np.ndarray, list[str]]:
    samples = np.asarray(states, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError(f"Expected states with shape (n, 2), got {samples.shape}")

    u = samples[:, 0]
    v = samples[:, 1]
    residual = v - manifold_function(u, cubic_coupling)

    if mode == "identity":
        features = np.column_stack([u, v])
        names = ["u", "v"]
    elif mode == "polynomial":
        features = np.column_stack([u, v, u ** 2, u * v, v ** 2, u ** 3])
        names = ["u", "v", "u^2", "uv", "v^2", "u^3"]
    elif mode == "fourier":
        features = np.column_stack([np.sin(u), np.cos(u), np.sin(v), np.cos(v)])
        names = ["sin(u)", "cos(u)", "sin(v)", "cos(v)"]
    elif mode == "slow_residual":
        features = np.column_stack([u, residual])
        names = ["u", "r"]
    elif mode == "identity_plus_residual":
        features = np.column_stack([u, v, residual])
        names = ["u", "v", "r"]
    else:
        raise ValueError(f"Unsupported observable mode: {mode}")
    return features, names


def inverse_sqrt_psd(matrix: np.ndarray, eps: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    sym_matrix = 0.5 * (matrix + matrix.T)
    evals, evecs = np.linalg.eigh(sym_matrix)
    positive = evals > eps

    sqrt_matrix = np.zeros_like(sym_matrix)
    inv_sqrt_matrix = np.zeros_like(sym_matrix)
    if np.any(positive):
        sqrt_diag = np.sqrt(evals[positive])
        inv_sqrt_diag = 1.0 / sqrt_diag
        vecs = evecs[:, positive]
        sqrt_matrix = vecs @ np.diag(sqrt_diag) @ vecs.T
        inv_sqrt_matrix = vecs @ np.diag(inv_sqrt_diag) @ vecs.T
    return np.real_if_close(sqrt_matrix), np.real_if_close(inv_sqrt_matrix)


def fit_whitened_koopman(
    states: np.ndarray,
    library: SlowManifoldObservableLibrary,
    *,
    lag_steps: int = 1,
    ridge: float = 1e-10,
    eps: float = 1e-10,
) -> dict[str, np.ndarray]:
    if states.ndim != 2 or states.shape[0] <= lag_steps:
        raise ValueError("states must have shape (T, d) with T > lag_steps")

    current_states = states[:-lag_steps]
    future_states = states[lag_steps:]
    X = library.transform(current_states)
    Y = library.transform(future_states)

    num_pairs = X.shape[0]
    weights = np.ones(num_pairs, dtype=float) / num_pairs
    X_weighted = X * weights[:, None]
    Y_weighted = Y * weights[:, None]

    C00 = X_weighted.T @ X
    C01 = X_weighted.T @ Y
    C11 = Y_weighted.T @ Y
    A = np.linalg.pinv(C00 + ridge * np.eye(C00.shape[0])) @ C01
    C00_sqrt, C00_inv_sqrt = inverse_sqrt_psd(C00, eps=eps)
    C11_sqrt, C11_inv_sqrt = inverse_sqrt_psd(C11, eps=eps)
    K_bar = C00_inv_sqrt @ C01 @ C11_inv_sqrt
    K_bar_from_A = C00_sqrt @ A @ C11_inv_sqrt

    return {
        "X": np.real_if_close(X),
        "Y": np.real_if_close(Y),
        "weights": weights,
        "C00": np.real_if_close(C00),
        "C01": np.real_if_close(C01),
        "C11": np.real_if_close(C11),
        "A": np.real_if_close(A),
        "C00_sqrt": C00_sqrt,
        "C00_inv_sqrt": C00_inv_sqrt,
        "C11_sqrt": C11_sqrt,
        "C11_inv_sqrt": C11_inv_sqrt,
        "K_bar": np.real_if_close(K_bar),
        "K_bar_from_A": np.real_if_close(K_bar_from_A),
    }


def average_reversibility(singular_values: np.ndarray, rank: int | None = None, alpha: float = 1.0) -> float:
    sigmas = np.asarray(singular_values, dtype=float)
    if sigmas.ndim != 1 or sigmas.size == 0:
        raise ValueError("singular_values must be a non-empty one-dimensional array")
    if rank is None:
        rank = sigmas.size
    rank = int(np.clip(rank, 1, sigmas.size))
    return float(np.mean(np.power(np.clip(sigmas[:rank], 0.0, None), alpha)))


def average_reversibility_curve(singular_values: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    sigmas = np.asarray(singular_values, dtype=float)
    return np.array([average_reversibility(sigmas, rank=r, alpha=alpha) for r in range(1, sigmas.size + 1)])


def choose_truncation_rank(singular_values: np.ndarray, alpha: float = 1.0, tolerance: float = 1e-10) -> int:
    curve = average_reversibility_curve(singular_values, alpha=alpha)
    best_value = float(np.max(curve))
    candidate_indices = np.flatnonzero(curve >= best_value - tolerance)
    return int(candidate_indices[0] + 1)


def compute_local_drift_field(states: np.ndarray, lag_steps: int = 1, bins: int = 24) -> dict[str, np.ndarray]:
    current = states[:-lag_steps]
    future = states[lag_steps:]
    deltas = future - current

    u_edges = np.linspace(np.min(current[:, 0]), np.max(current[:, 0]), bins + 1)
    v_edges = np.linspace(np.min(current[:, 1]), np.max(current[:, 1]), bins + 1)
    u_centers = 0.5 * (u_edges[:-1] + u_edges[1:])
    v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])

    uu, vv = np.meshgrid(u_centers, v_centers, indexing="xy")
    du = np.full_like(uu, np.nan, dtype=float)
    dv = np.full_like(vv, np.nan, dtype=float)
    counts = np.zeros_like(uu, dtype=int)

    u_idx = np.clip(np.digitize(current[:, 0], u_edges) - 1, 0, bins - 1)
    v_idx = np.clip(np.digitize(current[:, 1], v_edges) - 1, 0, bins - 1)

    sum_du = np.zeros_like(uu, dtype=float)
    sum_dv = np.zeros_like(vv, dtype=float)
    for idx_u, idx_v, delta in zip(u_idx, v_idx, deltas):
        counts[idx_v, idx_u] += 1
        sum_du[idx_v, idx_u] += delta[0]
        sum_dv[idx_v, idx_u] += delta[1]

    valid = counts >= 8
    du[valid] = sum_du[valid] / counts[valid]
    dv[valid] = sum_dv[valid] / counts[valid]
    return {"u_grid": uu, "v_grid": vv, "du": du, "dv": dv, "counts": counts}


def _align_leading_mode(analysis: dict[str, Any]) -> None:
    phi_values = analysis["z_current"][:, 0]
    corr = np.corrcoef(phi_values, analysis["current_states"][:, 0])[0, 1]
    if np.isnan(corr) or corr >= 0:
        return

    analysis["left_singular_vectors"][:, 0] *= -1
    analysis["right_singular_vectors"][:, 0] *= -1
    analysis["z_current"][:, 0] *= -1
    analysis["z_future"][:, 0] *= -1
    analysis["phi1_values"] *= -1
    analysis["psi1_values"] *= -1


def analyze_case(
    config: SlowManifoldConfig | None = None,
    *,
    alpha: float = 1.0,
    lag_steps: int | None = None,
    bins: int = 24,
) -> dict[str, Any]:
    if config is None:
        config = SlowManifoldConfig()
    if lag_steps is None:
        lag_steps = config.lag_steps

    simulation = simulate_slow_manifold_system(config)
    states = simulation["states"]
    library = SlowManifoldObservableLibrary(
        center=True,
        mode=config.observable_mode,
        cubic_coupling=config.cubic_coupling,
    ).fit(states)
    koop = fit_whitened_koopman(states, library=library, lag_steps=lag_steps)

    U, singular_values, Vt = np.linalg.svd(koop["K_bar"], full_matrices=False)
    V = Vt.T
    current_states = states[:-lag_steps]
    future_states = states[lag_steps:]
    z_current = koop["X"] @ koop["C00_inv_sqrt"] @ U
    z_future = koop["Y"] @ koop["C11_inv_sqrt"] @ V

    analysis: dict[str, Any] = {
        "config": config,
        "simulation": simulation,
        "states": states,
        "current_states": current_states,
        "future_states": future_states,
        "observable_names": library.get_feature_names(),
        "observable_mean": library.mean_,
        "features_current": koop["X"],
        "features_future": koop["Y"],
        "K_bar": koop["K_bar"],
        "koopman_fit": koop,
        "singular_values": singular_values,
        "left_singular_vectors": U,
        "right_singular_vectors": V,
        "z_current": z_current,
        "z_future": z_future,
        "phi1_values": z_current[:, 0].copy(),
        "psi1_values": z_future[:, 0].copy(),
        "average_reversibility_curve": average_reversibility_curve(singular_values, alpha=alpha),
        "micro_average_reversibility": average_reversibility(singular_values, rank=singular_values.size, alpha=alpha),
        "macro_average_reversibility_rank1": average_reversibility(singular_values, rank=1, alpha=alpha),
        "local_drift": compute_local_drift_field(states, lag_steps=lag_steps, bins=bins),
        "alpha": alpha,
        "lag_steps": lag_steps,
        "library": library,
    }
    analysis["truncation_rank"] = choose_truncation_rank(singular_values, alpha=alpha)
    analysis["truncated_average_reversibility"] = average_reversibility(
        singular_values,
        rank=analysis["truncation_rank"],
        alpha=alpha,
    )
    _align_leading_mode(analysis)
    return analysis


def analyze_observable_modes(
    observable_modes: list[str],
    *,
    config: SlowManifoldConfig | None = None,
    alpha: float = 1.0,
    lag_steps: int | None = None,
    bins: int = 24,
) -> dict[str, Any]:
    if config is None:
        config = SlowManifoldConfig()

    mode_results: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for mode in observable_modes:
        mode_config = SlowManifoldConfig(**{**asdict(config), "observable_mode": mode})
        analysis = analyze_case(mode_config, alpha=alpha, lag_steps=lag_steps, bins=bins)
        mode_results[mode] = analysis
        singular_values = analysis["singular_values"]
        row = {
            "观测模式": mode,
            "维数": len(analysis["observable_names"]),
            "观测函数": ", ".join(analysis["observable_names"]),
            "截断维数": analysis["truncation_rank"],
            "截断平均可逆性": analysis["truncated_average_reversibility"],
            "微观全维平均可逆性": analysis["micro_average_reversibility"],
            "截断增益": analysis["truncated_average_reversibility"] - analysis["micro_average_reversibility"],
        }
        for index, sigma in enumerate(singular_values, start=1):
            row[f"sigma_{index}"] = float(sigma)
        rows.append(row)

    comparison_table = pd.DataFrame(rows)
    return {"mode_results": mode_results, "comparison_table": comparison_table}


def evaluate_singular_function(
    analysis: dict[str, Any],
    points: np.ndarray,
    *,
    side: str = "current",
    mode_index: int = 0,
) -> np.ndarray:
    samples = np.asarray(points, dtype=float)
    features = analysis["library"].transform(samples)
    if side == "current":
        basis = analysis["koopman_fit"]["C00_inv_sqrt"] @ analysis["left_singular_vectors"][:, mode_index]
    elif side == "future":
        basis = analysis["koopman_fit"]["C11_inv_sqrt"] @ analysis["right_singular_vectors"][:, mode_index]
    else:
        raise ValueError("side must be 'current' or 'future'")
    return np.real_if_close(features @ basis)


def build_phase_plane_grid(states: np.ndarray, n_points: int = 180, margin: float = 0.12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=float)
    mins = np.min(states, axis=0)
    maxs = np.max(states, axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    mins = mins - margin * spans
    maxs = maxs + margin * spans
    u_lin = np.linspace(mins[0], maxs[0], n_points)
    v_lin = np.linspace(mins[1], maxs[1], n_points)
    uu, vv = np.meshgrid(u_lin, v_lin, indexing="xy")
    points = np.column_stack([uu.ravel(), vv.ravel()])
    return uu, vv, points


def summarize_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    sigmas = analysis["singular_values"]
    return {
        "config": asdict(analysis["config"]),
        "observable_names": analysis["observable_names"],
        "singular_values": [float(x) for x in sigmas],
        "micro_average_reversibility": float(analysis["micro_average_reversibility"]),
        "macro_average_reversibility_rank1": float(analysis["macro_average_reversibility_rank1"]),
        "reversibility_gain_rank1": float(
            analysis["macro_average_reversibility_rank1"] - analysis["micro_average_reversibility"]
        ),
    }


def save_summary_json(analysis: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summarize_analysis(analysis), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def plot_phase_portrait(analysis: dict[str, Any], output_path: str | Path | None = None) -> tuple[plt.Figure, plt.Axes]:
    set_plot_style()
    states = analysis["states"]
    fig, ax = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)
    hb = ax.hexbin(states[:, 0], states[:, 1], gridsize=55, cmap="YlGnBu", mincnt=1)

    u_line = np.linspace(np.percentile(states[:, 0], 1), np.percentile(states[:, 0], 99), 300)
    ax.plot(
        u_line,
        manifold_function(u_line, analysis["config"].cubic_coupling),
        color="#c46a2f",
        linewidth=2.4,
        label="参考慢流形",
    )
    ax.set_title("二维微观相图：轨道主要贴附在一条慢流形附近")
    ax.set_xlabel("慢变量 u")
    ax.set_ylabel("从变量 v")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.colorbar(hb, ax=ax, label="样本密度")
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    return fig, ax


def plot_phase_portrait_with_drift(analysis: dict[str, Any], output_path: str | Path | None = None) -> tuple[plt.Figure, plt.Axes]:
    set_plot_style()
    states = analysis["states"]
    drift = analysis["local_drift"]
    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    sample = states[::4]
    ax.scatter(sample[:, 0], sample[:, 1], s=8, alpha=0.15, color="#35566f", label="采样轨道")

    u_line = np.linspace(np.percentile(states[:, 0], 1), np.percentile(states[:, 0], 99), 300)
    ax.plot(
        u_line,
        manifold_function(u_line, analysis["config"].cubic_coupling),
        color="#c46a2f",
        linewidth=2.8,
        label="参考慢流形",
    )

    valid = ~np.isnan(drift["du"])
    ax.quiver(
        drift["u_grid"][valid],
        drift["v_grid"][valid],
        drift["du"][valid],
        drift["dv"][valid],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.003,
        color="#22485f",
        alpha=0.85,
    )
    ax.set_title("相图与局部平均漂移：横向细节快速塌缩到慢流形")
    ax.set_xlabel("慢变量 u")
    ax.set_ylabel("从变量 v")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    return fig, ax


def plot_singular_spectrum(analysis: dict[str, Any], output_path: str | Path | None = None) -> tuple[plt.Figure, plt.Axes]:
    set_plot_style()
    sigmas = analysis["singular_values"]
    indices = np.arange(1, sigmas.size + 1)
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    ax.bar(indices, sigmas, color=["#2f698f" if i == 1 else "#b8ccd8" for i in indices], width=0.72)
    ax.plot(indices, sigmas, color="#1f4258", marker="o", linewidth=1.8)
    ax.set_xlabel("奇异值序号 i")
    ax.set_ylabel(r"$\sigma_i$")
    ax.set_title("双边白化 Koopman 的奇异值谱")
    ax.set_xticks(indices)
    ax.set_ylim(0.0, min(1.05, max(1.02, np.max(sigmas) + 0.05)))
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    return fig, ax


def plot_koopman_matrix_heatmap(analysis: dict[str, Any], output_path: str | Path | None = None) -> tuple[plt.Figure, plt.Axes]:
    set_plot_style()
    matrix = analysis["K_bar"]
    names = analysis["observable_names"]
    fig, ax = plt.subplots(figsize=(7.4, 6.2), constrained_layout=True)
    image = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticklabels(names)
    ax.set_title("双边白化 Koopman 矩阵热图")
    fig.colorbar(image, ax=ax, label=r"$\bar{K}_{ij}$")
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    return fig, ax


def plot_average_reversibility(analysis: dict[str, Any], output_path: str | Path | None = None) -> tuple[plt.Figure, plt.Axes]:
    set_plot_style()
    curve = analysis["average_reversibility_curve"]
    micro = analysis["micro_average_reversibility"]
    trunc_rank = analysis["truncation_rank"]
    trunc_value = analysis["truncated_average_reversibility"]
    ranks = np.arange(1, curve.size + 1)

    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    ax.plot(ranks, curve, marker="o", linewidth=2.0, color="#2f698f", label="宏观 rank-r 平均可逆性")
    ax.axhline(micro, linestyle="--", color="#9a4f18", linewidth=1.8, label="微观全维平均可逆性")
    ax.scatter([trunc_rank], [trunc_value], color="#c46a2f", s=70, zorder=3, label=f"截断点 r={trunc_rank}")
    ax.set_xlabel("保留的宏观维数 r")
    ax.set_ylabel(r"$\bar{\Gamma}_\alpha^K(r)$")
    ax.set_title("维度平均近似可逆性随宏观维数的变化")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    return fig, ax


def plot_mode_singular_spectra(
    mode_results: dict[str, dict[str, Any]],
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    set_plot_style()
    n_modes = len(mode_results)
    fig, axes = plt.subplots(1, n_modes, figsize=(5.4 * n_modes, 4.6), constrained_layout=True)
    if n_modes == 1:
        axes = np.array([axes])

    for ax, (mode, analysis) in zip(axes, mode_results.items()):
        sigmas = analysis["singular_values"]
        indices = np.arange(1, sigmas.size + 1)
        trunc_rank = analysis["truncation_rank"]
        colors = ["#2f698f" if i <= trunc_rank else "#cfdce4" for i in indices]
        ax.bar(indices, sigmas, color=colors, width=0.72)
        ax.plot(indices, sigmas, color="#1f4258", marker="o", linewidth=1.7)
        ax.axvline(trunc_rank + 0.5, color="#c46a2f", linestyle="--", linewidth=1.5)
        ax.set_title(f"{mode} 观测基")
        ax.set_xlabel("奇异值序号 i")
        ax.set_ylabel(r"$\sigma_i$")
        ax.set_xticks(indices)
        ax.set_ylim(0.0, min(1.05, max(1.02, np.max(sigmas) + 0.05)))
        ax.text(
            0.98,
            0.95,
            f"截断 r={trunc_rank}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#c46a2f", "alpha": 0.9},
        )
    fig.suptitle("不同观测基下的双边白化 Koopman 奇异值谱与截断位置", fontsize=14)
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    return fig, axes


def plot_phi1_on_phase_plane(analysis: dict[str, Any], output_path: str | Path | None = None) -> tuple[plt.Figure, plt.Axes]:
    set_plot_style()
    states = analysis["current_states"]
    phi1 = analysis["phi1_values"]
    uu, vv, points = build_phase_plane_grid(states)
    phi_grid = evaluate_singular_function(analysis, points, side="current", mode_index=0).reshape(uu.shape)

    fig, ax = plt.subplots(figsize=(9.0, 5.8), constrained_layout=True)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=np.nanmin(phi_grid), vmax=np.nanmax(phi_grid))
    contour = ax.contourf(uu, vv, phi_grid, levels=24, cmap="coolwarm", norm=norm)
    scatter = ax.scatter(states[:, 0], states[:, 1], c=phi1, s=8, cmap="coolwarm", norm=norm, alpha=0.35)
    del scatter
    u_line = np.linspace(np.percentile(states[:, 0], 1), np.percentile(states[:, 0], 99), 300)
    ax.plot(
        u_line,
        manifold_function(u_line, analysis["config"].cubic_coupling),
        color="black",
        linewidth=2.0,
        linestyle="--",
    )
    ax.set_title("第一奇异函数在相平面上的取值：沿慢流形近似单调变化")
    ax.set_xlabel("慢变量 u")
    ax.set_ylabel("从变量 v")
    fig.colorbar(contour, ax=ax, label=r"$\phi_1(x)$")
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    return fig, ax


def plot_macro_variable_timeseries(
    analysis: dict[str, Any],
    output_path: str | Path | None = None,
    window: tuple[int, int] = (0, 450),
) -> tuple[plt.Figure, np.ndarray]:
    set_plot_style()
    start, end = window
    end = min(end, analysis["current_states"].shape[0])
    idx = np.arange(start, end)

    u = analysis["current_states"][start:end, 0]
    v = analysis["current_states"][start:end, 1]
    z = analysis["phi1_values"][start:end]
    z_scaled = (z - np.mean(z)) / (np.std(z) + 1e-12)
    u_scaled = (u - np.mean(u)) / (np.std(u) + 1e-12)

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True, constrained_layout=True)
    axes[0].plot(idx, u_scaled, color="#2f698f", linewidth=1.8, label="标准化慢变量 u_t")
    axes[0].plot(idx, z_scaled, color="#c46a2f", linewidth=1.8, label="标准化宏观变量 z_t")
    axes[0].set_ylabel("标准化幅值")
    axes[0].set_title("序参量宏观变量与原始慢变量的时间演化对照")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    axes[1].plot(idx, v, color="#8aa7b8", linewidth=1.5, label="从变量 v_t")
    axes[1].plot(
        idx,
        manifold_function(u, analysis["config"].cubic_coupling),
        color="#9a4f18",
        linewidth=1.7,
        label="对应慢流形 g(u_t)",
    )
    axes[1].set_xlabel("时间步")
    axes[1].set_ylabel("幅值")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    return fig, axes


def plot_macro_transition_scatter(analysis: dict[str, Any], output_path: str | Path | None = None) -> tuple[plt.Figure, plt.Axes]:
    set_plot_style()
    z_t = analysis["phi1_values"]
    z_next = analysis["psi1_values"]
    slope, intercept = np.polyfit(z_t, z_next, deg=1)
    x_line = np.linspace(np.min(z_t), np.max(z_t), 300)

    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    ax.scatter(z_t, z_next, s=9, alpha=0.22, color="#2f698f", label="样本对")
    ax.plot(x_line, slope * x_line + intercept, color="#c46a2f", linewidth=2.0, label="线性拟合")
    ax.set_xlabel(r"$z_t = \phi_1(x_t)$")
    ax.set_ylabel(r"$z_{t+\tau}^{+}$")
    ax.set_title("主导宏观通道的跨时散点图")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    return fig, ax


def export_case_figures(analysis: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    figure_paths = {
        "phase_portrait": output_path / "phase_portrait.png",
        "phase_portrait_with_drift": output_path / "phase_portrait_with_drift.png",
        "koopman_matrix_heatmap": output_path / "koopman_matrix_heatmap.png",
        "singular_spectrum": output_path / "singular_spectrum.png",
        "avg_reversibility_gain": output_path / "avg_reversibility_gain.png",
        "phi1_on_phase_plane": output_path / "phi1_on_phase_plane.png",
        "macro_variable_timeseries": output_path / "macro_variable_timeseries.png",
        "zt_vs_ztau": output_path / "zt_vs_ztau.png",
        "summary_metrics": output_path / "summary_metrics.json",
    }

    plot_phase_portrait(analysis, figure_paths["phase_portrait"])
    plot_phase_portrait_with_drift(analysis, figure_paths["phase_portrait_with_drift"])
    plot_koopman_matrix_heatmap(analysis, figure_paths["koopman_matrix_heatmap"])
    plot_singular_spectrum(analysis, figure_paths["singular_spectrum"])
    plot_average_reversibility(analysis, figure_paths["avg_reversibility_gain"])
    plot_phi1_on_phase_plane(analysis, figure_paths["phi1_on_phase_plane"])
    plot_macro_variable_timeseries(analysis, figure_paths["macro_variable_timeseries"])
    plot_macro_transition_scatter(analysis, figure_paths["zt_vs_ztau"])
    save_summary_json(analysis, figure_paths["summary_metrics"])
    plt.close("all")
    return {key: str(value) for key, value in figure_paths.items()}


__all__ = [
    "SlowManifoldConfig",
    "SlowManifoldObservableLibrary",
    "analyze_case",
    "analyze_observable_modes",
    "average_reversibility",
    "average_reversibility_curve",
    "build_observables",
    "build_phase_plane_grid",
    "choose_truncation_rank",
    "compute_local_drift_field",
    "evaluate_singular_function",
    "export_case_figures",
    "manifold_function",
    "plot_average_reversibility",
    "plot_macro_transition_scatter",
    "plot_macro_variable_timeseries",
    "plot_mode_singular_spectra",
    "plot_koopman_matrix_heatmap",
    "plot_phase_portrait",
    "plot_phase_portrait_with_drift",
    "plot_phi1_on_phase_plane",
    "plot_singular_spectrum",
    "save_summary_json",
    "set_plot_style",
    "simulate_slow_manifold_system",
    "summarize_analysis",
]
