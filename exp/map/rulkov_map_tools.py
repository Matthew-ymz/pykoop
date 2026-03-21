from __future__ import annotations

import json
import sys
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = None
for candidate in [Path(__file__).resolve().parent, Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]]:
    if (candidate / 'tools' / 'tools.py').exists():
        REPO_ROOT = candidate
        break
if REPO_ROOT is None:
    raise RuntimeError('Could not locate repository root containing tools/tools.py')
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    import plotly.express  # type: ignore  # noqa: F401
    import plotly.graph_objects  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    plotly_module = types.ModuleType('plotly')
    plotly_express = types.ModuleType('plotly.express')
    plotly_graph_objects = types.ModuleType('plotly.graph_objects')
    plotly_module.express = plotly_express
    plotly_module.graph_objects = plotly_graph_objects
    sys.modules.setdefault('plotly', plotly_module)
    sys.modules.setdefault('plotly.express', plotly_express)
    sys.modules.setdefault('plotly.graph_objects', plotly_graph_objects)

from exp.koopman_ce_cases import channel_scores_from_singular_values, koopman_ce_total_score_from_kbar, liu2025_log_gamma_gis
from tools.tools import compute_entropy, fit_data_koopman_operator, get_positive_contributions, whiten_operator_matrix

V_LAG = 'vlag'


@dataclass
class RulkovSimulationConfig:
    """Two-population Rulkov-map configuration aligned with example_maps_Q logic."""

    n_a: int = 100
    n_b: int = 100
    alpha_a: float | list[float] = 4.6
    alpha_b: float | list[float] = 4.6
    sigma_a: float | list[float] = 0.225
    sigma_b: float | list[float] = 0.225
    mu: float = 0.001
    gamma: float = 0.06
    epsilon: float = 0.02
    total_steps: int = 7000
    burn_in: int = 5000
    x0_a: float | None = -1.0
    x0_b: float | None = -1.2
    y0_a: float | None = -3.5
    y0_b: float | None = -3.7
    seed: int = 103

    @property
    def dt(self) -> float:
        return 1.0


@dataclass
class ObservableConfig:
    """Observable-library configuration.

    Notes
    -----
    For the map workflow, ``identity_quadratic`` defaults to identity plus
    elementwise squares, which matches the example_maps_Q style and keeps the
    library scalable for multi-neuron trajectories.
    """

    mode: str = 'identity_quadratic'
    polynomial_degree: int = 2
    include_bias: bool = False
    fourier_frequencies: list[int] = field(default_factory=lambda: [1])
    custom_callable: Callable[[np.ndarray, list[str]], tuple[np.ndarray, list[str]]] | None = None


@dataclass
class WorkflowConfig:
    """Top-level workflow configuration."""

    simulation: RulkovSimulationConfig = field(default_factory=RulkovSimulationConfig)
    observables: ObservableConfig = field(default_factory=ObservableConfig)
    lag_steps: int = 1
    rank: int = 2
    alpha: float = 1.0
    ridge: float = 1e-10
    eps: float = 1e-10
    include_closed_form_ce: bool = True
    results_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / 'results')


def set_plot_style() -> None:
    """Apply plotting defaults with Chinese font fallbacks."""

    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_theme(style='whitegrid')
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['savefig.dpi'] = 160
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = [
        'Microsoft YaHei',
        'SimHei',
        'Noto Sans CJK SC',
        'Source Han Sans SC',
        'Arial Unicode MS',
        'DejaVu Sans',
    ]


def locate_reference_data() -> dict[str, Any]:
    """Locate example_maps_Q-style references and expose fallback metadata."""

    requested_names = ['example_maps_Q', 'examp_map_Q']
    matched_paths: list[str] = []
    for requested_name in requested_names:
        matched_paths.extend(str(path.relative_to(REPO_ROOT)) for path in REPO_ROOT.rglob(f'{requested_name}*'))
    matched_paths = sorted(set(matched_paths))
    return {
        'available': len(matched_paths) > 0,
        'requested_name': 'example_maps_Q',
        'matched_paths': matched_paths,
        'fallback_notebook': 'exp/discrete_maps/examp_rulkov_pll.ipynb',
    }


def _ensure_vector(value: float | list[float], length: int) -> np.ndarray:
    """Expand a scalar parameter into a per-neuron vector."""

    if isinstance(value, (int, float)):
        return np.full(length, float(value))
    array = np.asarray(value, dtype=float)
    if array.shape[0] != length:
        raise ValueError('Parameter length does not match neuron count')
    return array


def _fast_map(x_value: float, y_value: float, alpha_value: float) -> float:
    """Rulkov fast map used in example_maps_Q."""

    if x_value <= 0:
        return alpha_value / (1.0 - x_value) + y_value
    if 0 < x_value < alpha_value + y_value:
        return alpha_value + y_value
    return -1.0


def _slow_map(x_value: float, y_value: float, mu_value: float, sigma_value: float) -> float:
    """Slow-variable update used in example_maps_Q."""

    return y_value - mu_value * (x_value + 1.0) + mu_value * sigma_value


def generate_two_population_neuron_data(config: RulkovSimulationConfig) -> dict[str, Any]:
    """Generate two-population Rulkov data using the example_maps_Q logic.

    Returns a dictionary structure compatible with ``plot_neuron_analysis_combo``.
    """

    rng = np.random.default_rng(config.seed)
    n_a, n_b = config.n_a, config.n_b
    total_neurons = n_a + n_b
    alpha_a = _ensure_vector(config.alpha_a, n_a)
    alpha_b = _ensure_vector(config.alpha_b, n_b)
    sigma_a = _ensure_vector(config.sigma_a, n_a)
    sigma_b = _ensure_vector(config.sigma_b, n_b)
    alpha_all = np.concatenate([alpha_a, alpha_b])
    sigma_all = np.concatenate([sigma_a, sigma_b])

    x_series = np.zeros((total_neurons, config.total_steps), dtype=float)
    y_series = np.zeros((total_neurons, config.total_steps), dtype=float)

    if config.x0_a is None:
        x_series[:n_a, 0] = rng.uniform(-1.5, -0.5, n_a)
        x0_a_record: float | str = '随机值（每个神经元独立）'
    else:
        x_series[:n_a, 0] = float(config.x0_a)
        x0_a_record = float(config.x0_a)
    if config.x0_b is None:
        x_series[n_a:, 0] = rng.uniform(-1.5, -0.5, n_b)
        x0_b_record = '随机值（每个神经元独立）'
    else:
        x_series[n_a:, 0] = float(config.x0_b)
        x0_b_record = float(config.x0_b)
    if config.y0_a is None:
        y_series[:n_a, 0] = rng.uniform(-4.0, -3.0, n_a)
        y0_a_record: float | str = '随机值（每个神经元独立）'
    else:
        y_series[:n_a, 0] = float(config.y0_a)
        y0_a_record = float(config.y0_a)
    if config.y0_b is None:
        y_series[n_a:, 0] = rng.uniform(-4.0, -3.0, n_b)
        y0_b_record = '随机值（每个神经元独立）'
    else:
        y_series[n_a:, 0] = float(config.y0_b)
        y0_b_record = float(config.y0_b)

    for step in range(config.total_steps - 1):
        xbar_a = np.mean(x_series[:n_a, step])
        xbar_b = np.mean(x_series[n_a:, step])
        for i in range(n_a):
            local_part = _fast_map(x_series[i, step], y_series[i, step], alpha_all[i])
            x_series[i, step + 1] = (1.0 - config.gamma) * local_part + config.gamma * xbar_a + config.epsilon * xbar_b
            y_series[i, step + 1] = _slow_map(x_series[i, step], y_series[i, step], config.mu, sigma_all[i])
        for j in range(n_b):
            idx = n_a + j
            local_part = _fast_map(x_series[idx, step], y_series[idx, step], alpha_all[idx])
            x_series[idx, step + 1] = (1.0 - config.gamma) * local_part + config.gamma * xbar_b + config.epsilon * xbar_a
            y_series[idx, step + 1] = _slow_map(x_series[idx, step], y_series[idx, step], config.mu, sigma_all[idx])

    def compute_instantaneous_std(x_block: np.ndarray, mean_field: np.ndarray) -> np.ndarray:
        return np.sqrt(np.mean((x_block - mean_field[None, :]) ** 2, axis=0))

    xbar_a = np.mean(x_series[:n_a, :], axis=0)
    xbar_b = np.mean(x_series[n_a:, :], axis=0)
    xbar_all = np.mean(x_series, axis=0)
    r_a_t = compute_instantaneous_std(x_series[:n_a, :], xbar_a)
    r_b_t = compute_instantaneous_std(x_series[n_a:, :], xbar_b)
    r_t = compute_instantaneous_std(x_series, xbar_all)

    xbar_a_transient = xbar_a[config.burn_in:]
    xbar_b_transient = xbar_b[config.burn_in:]
    r_a_t_transient = r_a_t[config.burn_in:]
    r_b_t_transient = r_b_t[config.burn_in:]
    r_t_transient = r_t[config.burn_in:]
    effective_length = config.total_steps - config.burn_in

    r_a = float(np.mean(r_a_t_transient))
    r_b = float(np.mean(r_b_t_transient))
    r_global = float(np.mean(r_t_transient))
    r_delta = float(np.mean(np.abs(xbar_a_transient - xbar_b_transient)))

    def determine_sync_state(metric_a: float, metric_b: float, metric_delta: float, threshold: float = 1e-7) -> str:
        sync_a = metric_a < threshold
        sync_b = metric_b < threshold
        sync_ab = metric_delta < threshold
        if sync_a and sync_b and sync_ab:
            return 'Complete Synchronization 完全同步(CS)'
        if sync_a and sync_b and not sync_ab:
            return 'Generalized Synchronization 广义同步(GS)'
        if (sync_a and not sync_b) or (not sync_a and sync_b):
            return 'Chimera State 奇美拉态(Q)'
        if not sync_a and not sync_b:
            return 'Desynchronization 去同步化(D)'
        return 'Unknown State 未知'

    sync_state = determine_sync_state(r_a, r_b, r_delta)

    params = {
        '神经元总数': total_neurons,
        'a群体神经元数': n_a,
        'b群体神经元数': n_b,
        'a群体α参数': alpha_a[0] if np.allclose(alpha_a, alpha_a[0]) else alpha_a.tolist(),
        'b群体α参数': alpha_b[0] if np.allclose(alpha_b, alpha_b[0]) else alpha_b.tolist(),
        'a群体σ参数': sigma_a[0] if np.allclose(sigma_a, sigma_a[0]) else sigma_a.tolist(),
        'b群体σ参数': sigma_b[0] if np.allclose(sigma_b, sigma_b[0]) else sigma_b.tolist(),
        '慢变参数μ': config.mu,
        '群体内耦合强度γ': config.gamma,
        '群体间耦合强度ε': config.epsilon,
        '总步数': config.total_steps,
        '舍弃暂态': config.burn_in,
        '有效数据长度': effective_length,
        '随机种子': config.seed,
        'a群体x初值': x0_a_record,
        'b群体x初值': x0_b_record,
        'a群体y初值': y0_a_record,
        'b群体y初值': y0_b_record,
        '⟨σa⟩ (R_a)': r_a,
        '⟨σb⟩ (R_b)': r_b,
        '⟨σt⟩ (R_t)': r_global,
        '⟨δ⟩ (R_delta)': r_delta,
        '同步状态': sync_state,
        'a群体同步': r_a < 1e-7,
        'b群体同步': r_b < 1e-7,
        'Rt群体同步': r_global < 1e-7,
        '群体间同步': r_delta < 1e-7,
    }

    reference_info = locate_reference_data()
    source_note = (
        'reference data found: ' + ', '.join(reference_info['matched_paths'])
        if reference_info['available']
        else 'reference data unavailable; fallback to exp/discrete_maps/examp_rulkov_pll.ipynb generation logic'
    )

    data = {
        'params': params,
        'summary': {
            'seed': config.seed,
            'R_a': r_a,
            'R_b': r_b,
            'R_t': r_global,
            'R_delta': r_delta,
            'sync_state': sync_state,
        },
        'reference_info': reference_info,
        'source_note': source_note,
        '群体信息': {
            'a群体神经元数': n_a,
            'b群体神经元数': n_b,
            'a群体索引': list(range(n_a)),
            'b群体索引': list(range(n_a, n_a + n_b)),
            'a群体α参数': alpha_a,
            'b群体α参数': alpha_b,
            'a群体σ参数': sigma_a,
            'b群体σ参数': sigma_b,
        },
        '时间序列': {
            't': np.arange(effective_length),
            'Xbar_a': xbar_a,
            'Xbar_b': xbar_b,
            'Xbar_a_transient': xbar_a_transient,
            'Xbar_b_transient': xbar_b_transient,
            'r_a_t': r_a_t,
            'r_b_t': r_b_t,
            'r_t': r_t,
            'r_a_t_transient': r_a_t_transient,
            'r_b_t_transient': r_b_t_transient,
            'r_t_transient': r_t_transient,
        },
        '同步指标': {
            'R_a': r_a,
            'R_b': r_b,
            'R_t': r_global,
            'R_delta': r_delta,
            'sync_state': sync_state,
        },
    }

    for i in range(n_a):
        data[f'神经元_a_{i+1:03d}'] = {
            'x_transient': x_series[i, config.burn_in:],
            'y_transient': y_series[i, config.burn_in:],
            'x': x_series[i],
            'y': y_series[i],
            '群体': 'a',
            'α参数': float(alpha_a[i]),
            'σ参数': float(sigma_a[i]),
        }
    for j in range(n_b):
        idx = n_a + j
        data[f'神经元_b_{j+1:03d}'] = {
            'x_transient': x_series[idx, config.burn_in:],
            'y_transient': y_series[idx, config.burn_in:],
            'x': x_series[idx],
            'y': y_series[idx],
            '群体': 'b',
            'α参数': float(alpha_b[j]),
            'σ参数': float(sigma_b[j]),
        }

    state_matrix, state_names = build_state_matrix_from_population_data(data)
    data['state_matrix'] = state_matrix
    data['state_names'] = state_names
    data['state_frame'] = pd.DataFrame(state_matrix, columns=state_names)
    return data


def build_state_matrix_from_population_data(data: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    """Convert example_maps_Q-style simulation output into a trajectory matrix."""

    n_a = data['群体信息']['a群体神经元数']
    n_b = data['群体信息']['b群体神经元数']
    cols = []
    names: list[str] = []
    for idx in range(1, n_a + 1):
        key = f'神经元_a_{idx:03d}'
        cols.extend([data[key]['x_transient'], data[key]['y_transient']])
        names.extend([f'x_a{idx}', f'y_a{idx}'])
    for idx in range(1, n_b + 1):
        key = f'神经元_b_{idx:03d}'
        cols.extend([data[key]['x_transient'], data[key]['y_transient']])
        names.extend([f'x_b{idx}', f'y_b{idx}'])
    matrix = np.column_stack(cols)
    return matrix, names


def plot_neuron_analysis_combo(data: dict[str, Any], figsize: tuple[float, float] = (20, 8), time_window: tuple[int, int] | None = None, vmin: float | None = None, vmax: float | None = None, cmap: str = V_LAG) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot the combo figure used in example_maps_Q, with ``vlag`` heatmap."""

    params = data['params']
    group_info = data['群体信息']
    sync_info = data['同步指标']
    time_series = data['时间序列']
    n_a = group_info['a群体神经元数']
    n_b = group_info['b群体神经元数']
    total_neurons = n_a + n_b
    total_steps = params.get('总步数', 10000)
    burn_in = params.get('舍弃暂态', 1000)
    effective_steps = total_steps - burn_in

    x_data = np.zeros((total_neurons, effective_steps))
    for i in range(n_a):
        x_data[i, :] = data[f'神经元_a_{i+1:03d}']['x_transient']
    for j in range(n_b):
        x_data[n_a + j, :] = data[f'神经元_b_{j+1:03d}']['x_transient']

    if time_window is not None:
        start_t, end_t = time_window
        end_t = min(end_t, effective_steps)
        x_plot = x_data[:, start_t:end_t]
        t_display = np.arange(end_t - start_t)
    else:
        start_t, end_t = 0, effective_steps
        x_plot = x_data
        t_display = np.arange(effective_steps)

    if vmin is None:
        vmin = float(np.min(x_plot))
    if vmax is None:
        vmax = float(np.max(x_plot))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    im = ax1.imshow(
        x_plot,
        aspect='auto',
        cmap=cmap,
        extent=[0, len(t_display), 0, total_neurons],
        vmin=vmin,
        vmax=vmax,
        origin='lower',
        interpolation='nearest',
    )
    cbar = plt.colorbar(im, ax=ax1, pad=0.01, shrink=0.8)
    cbar.set_label('神经元状态 x 值', fontsize=12)
    ax1.set_xlabel('时间步 t', fontsize=12)
    ax1.set_ylabel('神经元索引', fontsize=12)
    ax1.set_yticks([n_a / 2, n_a + n_b / 2])
    ax1.set_yticklabels(['群体a', '群体b'], fontsize=12)
    ax1.axhline(y=n_a, color='white', linestyle='--', linewidth=2, alpha=0.8)
    title = f'神经元群体状态热力图 (同步状态: {sync_info.get("sync_state", "Unknown")})'
    if time_window is not None:
        title += f' (时间窗口: {start_t}-{end_t})'
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax1.grid(False)

    t_plot = np.arange(total_steps)
    ax2.plot(t_plot, time_series['r_a_t'], label='r_a(t)', color='blue', alpha=0.85, linewidth=1.3, linestyle=':', marker='.', markersize=4, markevery=max(1, total_steps // 80))
    ax2.plot(t_plot, time_series['r_b_t'], label='r_b(t)', color='red', alpha=0.85, linewidth=1.3, linestyle=':', marker='s', markersize=2, markevery=max(1, total_steps // 80))
    ax2.plot(t_plot, time_series['r_t'], label='r(t)', color='green', alpha=0.85, linewidth=1.4, linestyle='--')
    ax2.axvline(x=burn_in, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('时间步 t', fontsize=12)
    ax2.set_ylabel('瞬时标准差', fontsize=12)
    ax2.set_title('瞬时标准差时间序列', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(fontsize=10)
    ax2.set_facecolor('white')
    ax2.grid(False)

    plt.tight_layout()
    return fig, (ax1, ax2)


def build_observables(state_matrix: np.ndarray, state_names: list[str], config: ObservableConfig) -> tuple[np.ndarray, list[str]]:
    """Construct observables; default quadratic terms are elementwise squares."""

    blocks = []
    names: list[str] = []
    if config.include_bias:
        blocks.append(np.ones((state_matrix.shape[0], 1), dtype=float))
        names.append('1')

    if config.mode == 'identity':
        blocks.append(state_matrix.copy())
        names.extend(state_names)
    elif config.mode == 'quadratic':
        blocks.append(state_matrix ** 2)
        names.extend([f'{name}^2' for name in state_names])
    elif config.mode == 'identity_quadratic':
        blocks.append(state_matrix.copy())
        names.extend(state_names)
        blocks.append(state_matrix ** 2)
        names.extend([f'{name}^2' for name in state_names])
    elif config.mode == 'polynomial':
        feature_blocks = []
        feature_names: list[str] = []
        for degree in range(1, config.polynomial_degree + 1):
            for col_idx, name in enumerate(state_names):
                feature_blocks.append(state_matrix[:, [col_idx]] ** degree)
                feature_names.append(name if degree == 1 else f'{name}^{degree}')
        blocks.append(np.hstack(feature_blocks))
        names.extend(feature_names)
    elif config.mode == 'fourier':
        blocks.append(state_matrix.copy())
        names.extend(state_names)
        for frequency in config.fourier_frequencies:
            for state_idx, state_name in enumerate(state_names):
                blocks.append(np.sin(frequency * state_matrix[:, [state_idx]]))
                blocks.append(np.cos(frequency * state_matrix[:, [state_idx]]))
                names.extend([f'sin({frequency}*{state_name})', f'cos({frequency}*{state_name})'])
    elif config.mode == 'custom':
        if config.custom_callable is None:
            raise ValueError("custom_callable must be provided when mode='custom'")
        custom_features, custom_names = config.custom_callable(state_matrix, state_names)
        blocks.append(np.asarray(custom_features, dtype=float))
        names.extend(list(custom_names))
    else:
        raise ValueError(f'Unsupported observable mode: {config.mode}')

    return np.hstack(blocks), names


def format_equations(coefficient_matrix: np.ndarray, feature_names: list[str], target_names: list[str], threshold: float = 1e-5) -> list[str]:
    """Format a coefficient matrix as readable equations."""

    equations = []
    for col_idx, target_name in enumerate(target_names):
        terms = []
        for row_idx, feature_name in enumerate(feature_names):
            coefficient = coefficient_matrix[row_idx, col_idx]
            if abs(coefficient) > threshold:
                terms.append(f'{coefficient:.4f} * {feature_name}')
        equations.append(f"{target_name} = {(' + '.join(terms)).replace('+ -', '- ') if terms else '0'}")
    return equations


def compute_residual_covariance(x_pairs: np.ndarray, y_pairs: np.ndarray, operator_matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Estimate residual covariance for optional closed-form CE."""

    residuals = y_pairs - x_pairs @ operator_matrix
    covariance = residuals.T @ residuals / residuals.shape[0]
    covariance = 0.5 * (covariance + covariance.T)
    covariance += eps * np.eye(covariance.shape[0])
    return covariance


def _sparse_ticklabels(labels: list[str], step: int | None = None, show: bool = True) -> list[str] | bool:
    """Build sparse tick labels for large heatmaps."""

    if not show:
        return False
    if step is None or step <= 1:
        return labels
    return [label if idx % step == 0 else '' for idx, label in enumerate(labels)]


def plot_matrix_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    figsize: tuple[float, float] = (8.0, 6.0),
    center: float | None = 0.0,
    label_step: int | None = None,
    show_ticklabels: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a square matrix with ``vlag`` and optional sparse labels."""

    fig, ax = plt.subplots(figsize=figsize)
    ticklabels = _sparse_ticklabels(labels, step=label_step, show=show_ticklabels)
    sns.heatmap(matrix, cmap=V_LAG, center=center, xticklabels=ticklabels, yticklabels=ticklabels, ax=ax)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=90, labelsize=7)
    ax.tick_params(axis='y', rotation=0, labelsize=7)
    fig.tight_layout()
    return fig, ax


def plot_rectangular_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    figsize: tuple[float, float] = (10.0, 6.0),
    center: float | None = 0.0,
    row_label_step: int | None = None,
    column_label_step: int | None = None,
    show_row_labels: bool = True,
    show_column_labels: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a rectangular matrix with ``vlag`` and optional sparse labels."""

    fig, ax = plt.subplots(figsize=figsize)
    yticklabels = _sparse_ticklabels(row_labels, step=row_label_step, show=show_row_labels)
    xticklabels = _sparse_ticklabels(column_labels, step=column_label_step, show=show_column_labels)
    sns.heatmap(matrix, cmap=V_LAG, center=center, xticklabels=xticklabels, yticklabels=yticklabels, ax=ax)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=0, labelsize=7)
    ax.tick_params(axis='y', rotation=0, labelsize=7)
    fig.tight_layout()
    return fig, ax


def plot_singular_value_comparison(
    s_raw: np.ndarray,
    s_model: np.ndarray,
    s_empirical: np.ndarray,
    top_n: int | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the three singular-value curves on a single panel."""

    if top_n is not None:
        s_raw = s_raw[:top_n]
        s_model = s_model[:top_n]
        s_empirical = s_empirical[:top_n]
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    idx = np.arange(1, len(s_empirical) + 1)
    ax.plot(idx, s_raw, linestyle='--', marker='^', label='data-fitted step operator', markersize=3, linewidth=1.2, alpha=0.9)
    ax.plot(idx, s_model, linestyle='--', marker='|', label='model + correct whitening', markersize=10, linewidth=1.4, alpha=0.95)
    ax.plot(idx, s_empirical, linestyle='-', marker='o', linewidth=2.0, label='empirical whitened Koopman', markersize=3, alpha=0.85)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_xlabel('singular index')
    ax.set_ylabel('singular value')
    ax.set_title('singular value comparison' if top_n is None else f'singular value comparison (top {top_n})')
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_positive_contributions(contributions: list[float] | np.ndarray) -> tuple[plt.Figure, plt.Axes]:
    """Plot positive contributions in the singular spectrum.

    Parameters
    ----------
    contributions:
        Sequence returned by ``get_positive_contributions``.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axis for the bar plot.
    """

    values = np.asarray(contributions, dtype=float)
    fig, ax = plt.subplots(figsize=(10.0, 4.5))
    ax.bar(np.arange(1, len(values) + 1), values, color='#4C72B0', alpha=0.85)
    ax.set_xlabel('dimension')
    ax.set_ylabel('positive contribution')
    ax.set_title('positive contributions in singular spectrum')
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_macro_series(macro_series: np.ndarray, macro_names: list[str], max_points: int = 500) -> tuple[plt.Figure, np.ndarray]:
    """Plot macroscopic time series."""

    n_channels = macro_series.shape[1]
    fig, axes = plt.subplots(n_channels, 1, figsize=(10.0, max(3.0, 2.8 * n_channels)), sharex=True)
    axes = np.atleast_1d(axes)
    slc = slice(0, min(max_points, macro_series.shape[0]))
    for idx, ax in enumerate(axes):
        ax.plot(np.arange(macro_series[slc, idx].shape[0]), macro_series[slc, idx], linewidth=1.5)
        ax.set_ylabel(macro_names[idx])
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel('time index')
    fig.suptitle('Macroscopic time series', y=0.995)
    fig.tight_layout()
    return fig, axes


def plot_micro_macro_comparison(
    micro_series: np.ndarray,
    micro_names: list[str],
    macro_series: np.ndarray,
    macro_names: list[str],
    picked_indices: list[int] | None = None,
    max_points: int = 500,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a few raw micro channels against macro variables on a normalized scale.

    Parameters
    ----------
    micro_series:
        Microscopic state matrix with shape ``(time, n_micro_channels)``.
    micro_names:
        Names for microscopic channels.
    macro_series:
        Macroscopic time-series matrix with shape ``(time, n_macro_channels)``.
    macro_names:
        Names for macro channels.
    picked_indices:
        Optional indices of micro channels to display. If omitted, a small set
        spread across the channel list is selected automatically.
    max_points:
        Maximum number of time points to display from the start of the series.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axis containing the normalized comparison curves.
    """

    if micro_series.ndim != 2 or macro_series.ndim != 2:
        raise ValueError('micro_series and macro_series must be 2D arrays')

    time_points = min(max_points, micro_series.shape[0], macro_series.shape[0])
    micro_view = micro_series[:time_points]
    macro_view = macro_series[:time_points]

    if picked_indices is None:
        candidate_count = min(4, micro_view.shape[1])
        picked_indices = np.linspace(0, micro_view.shape[1] - 1, candidate_count, dtype=int).tolist()
    picked_indices = [idx for idx in picked_indices if 0 <= idx < micro_view.shape[1]]
    if not picked_indices:
        raise ValueError('No valid micro channel indices were selected')

    def normalize_column(column: np.ndarray) -> np.ndarray:
        std = float(np.std(column))
        if std < 1e-12:
            return column - float(np.mean(column))
        return (column - float(np.mean(column))) / std

    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    time_axis = np.arange(time_points)
    for idx in picked_indices:
        ax.plot(
            time_axis,
            normalize_column(micro_view[:, idx]),
            linewidth=1.0,
            alpha=0.8,
            label=f'micro: {micro_names[idx]}',
        )
    for idx, macro_name in enumerate(macro_names):
        ax.plot(
            time_axis,
            normalize_column(macro_view[:, idx]),
            linewidth=2.2,
            label=f'macro: {macro_name}',
        )
    ax.set_xlabel('time index')
    ax.set_ylabel('normalized value (z-score)')
    ax.set_title('Raw micro variables vs macro variables')
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    return fig, ax


def save_workflow_results(workflow: dict[str, Any], output_dir: Path | str | None = None) -> dict[str, str]:
    """Save tables, metrics, equations, and figures for the workflow."""

    if output_dir is None:
        output_dir = workflow['config'].results_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {}
    simulation = workflow['simulation']
    koopman = workflow['koopman']

    simulation['state_frame'].to_csv(output_dir / 'micro_state_series.csv')
    artifacts['micro_state_series'] = str(output_dir / 'micro_state_series.csv')

    pd.DataFrame(workflow['lifted_matrix'], columns=workflow['feature_names']).to_csv(output_dir / 'observable_library.csv', index_label='time_index')
    artifacts['observable_library'] = str(output_dir / 'observable_library.csv')

    pd.DataFrame(koopman['C00'], index=workflow['feature_names'], columns=workflow['feature_names']).to_csv(output_dir / 'C00.csv')
    pd.DataFrame(koopman['C01'], index=workflow['feature_names'], columns=workflow['feature_names']).to_csv(output_dir / 'C01.csv')
    pd.DataFrame(koopman['C11'], index=workflow['feature_names'], columns=workflow['feature_names']).to_csv(output_dir / 'C11.csv')
    artifacts['C00'] = str(output_dir / 'C00.csv')
    artifacts['C01'] = str(output_dir / 'C01.csv')
    artifacts['C11'] = str(output_dir / 'C11.csv')

    pd.DataFrame(workflow['operator_matrix'], index=workflow['feature_names'], columns=workflow['feature_names']).to_csv(output_dir / 'K_matrix.csv')
    pd.DataFrame(workflow['model_whitened_matrix'], index=workflow['feature_names'], columns=workflow['feature_names']).to_csv(output_dir / 'model_operator_with_correct_whitening.csv')
    pd.DataFrame(workflow['whitened_matrix'], index=workflow['feature_names'], columns=workflow['feature_names']).to_csv(output_dir / 'Kbar_matrix.csv')
    pd.DataFrame(workflow['singular_vectors_left'], index=workflow['feature_names']).to_csv(output_dir / 'left_singular_vectors.csv')
    pd.DataFrame(workflow['coarse_matrix'], index=workflow['feature_names'], columns=workflow['macro_names']).to_csv(output_dir / 'coarse_graining_matrix.csv')
    pd.DataFrame(workflow['macro_series'], columns=workflow['macro_names']).to_csv(output_dir / 'macro_time_series.csv', index_label='time_index')
    pd.DataFrame(workflow['macro_operator'], index=workflow['macro_names'], columns=workflow['macro_names']).to_csv(output_dir / 'macro_dynamics_matrix.csv')
    artifacts['K_matrix'] = str(output_dir / 'K_matrix.csv')
    artifacts['model_operator_with_correct_whitening'] = str(output_dir / 'model_operator_with_correct_whitening.csv')
    artifacts['Kbar_matrix'] = str(output_dir / 'Kbar_matrix.csv')
    artifacts['left_singular_vectors'] = str(output_dir / 'left_singular_vectors.csv')
    artifacts['coarse_graining_matrix'] = str(output_dir / 'coarse_graining_matrix.csv')
    artifacts['macro_time_series'] = str(output_dir / 'macro_time_series.csv')
    artifacts['macro_dynamics_matrix'] = str(output_dir / 'macro_dynamics_matrix.csv')

    pd.DataFrame({'index': np.arange(1, workflow['singular_values'].shape[0] + 1), 'singular_value': workflow['singular_values']}).to_csv(output_dir / 'singular_values.csv', index=False)
    pd.DataFrame({'dimension': np.arange(1, len(workflow['positive_contributions']) + 1), 'positive_contribution': workflow['positive_contributions']}).to_csv(output_dir / 'positive_contributions.csv', index=False)
    workflow['summary'].to_csv(output_dir / 'summary_metrics.csv', index=False)
    artifacts['singular_values'] = str(output_dir / 'singular_values.csv')
    artifacts['positive_contributions'] = str(output_dir / 'positive_contributions.csv')
    artifacts['summary_metrics'] = str(output_dir / 'summary_metrics.csv')

    (output_dir / 'equations.json').write_text(json.dumps({'coarse_equations': workflow['coarse_equations'], 'macro_equations': workflow['macro_equations']}, indent=2, ensure_ascii=False), encoding='utf-8')
    artifacts['equations'] = str(output_dir / 'equations.json')

    metadata = {
        'simulation_config': asdict(workflow['config'].simulation),
        'observable_config': {
            'mode': workflow['config'].observables.mode,
            'polynomial_degree': workflow['config'].observables.polynomial_degree,
            'include_bias': workflow['config'].observables.include_bias,
            'fourier_frequencies': workflow['config'].observables.fourier_frequencies,
        },
        'reference_info': workflow['simulation']['reference_info'],
        'source_note': workflow['simulation']['source_note'],
        'feature_names': workflow['feature_names'],
        'macro_names': workflow['macro_names'],
        'ec_increments': workflow['ec_increments'],
        'ce_channel_scores': workflow['ce_channel_scores'].tolist(),
        'ec_score': workflow['ec_score'],
        'ce_score_rank_sum': workflow['ce_score'],
        'ce_total_from_kbar': workflow['ce_total_from_kbar'],
        'closed_form_ce': workflow['closed_form_ce'],
        'positive_contributions': list(np.asarray(workflow['positive_contributions'], dtype=float)),
    }
    (output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    artifacts['metadata'] = str(output_dir / 'metadata.json')

    figures = {
        'micro_analysis_combo.png': plot_neuron_analysis_combo(simulation)[0],
        'C00.png': plot_matrix_heatmap(koopman['C00'], workflow['feature_names'], 'C00 covariance', figsize=(11, 10), label_step=100)[0],
        'C01.png': plot_matrix_heatmap(koopman['C01'], workflow['feature_names'], 'C01 covariance', figsize=(11, 10), label_step=100)[0],
        'C11.png': plot_matrix_heatmap(koopman['C11'], workflow['feature_names'], 'C11 covariance', figsize=(11, 10), label_step=100)[0],
        'K_matrix.png': plot_matrix_heatmap(workflow['operator_matrix'], workflow['feature_names'], 'Data-fitted step operator', figsize=(11, 10), label_step=100)[0],
        'model_operator_with_correct_whitening.png': plot_matrix_heatmap(workflow['model_whitened_matrix'], workflow['feature_names'], 'Model operator with correct whitening', figsize=(11, 10), label_step=100)[0],
        'Kbar_matrix.png': plot_matrix_heatmap(workflow['whitened_matrix'], workflow['feature_names'], 'Empirical whitened Koopman', figsize=(11, 10), label_step=100)[0],
        'left_singular_vectors.png': plot_rectangular_heatmap(workflow['singular_vectors_left'][:, : workflow['rank']], workflow['feature_names'], workflow['macro_names'], 'Left singular vectors (full)', figsize=(7, 12), show_row_labels=False)[0],
        'coarse_graining_matrix.png': plot_rectangular_heatmap(workflow['coarse_matrix'], workflow['feature_names'], workflow['macro_names'], 'Coarse-graining matrix', figsize=(7, 12), show_row_labels=False)[0],
        'macro_dynamics_matrix.png': plot_rectangular_heatmap(workflow['macro_operator'], workflow['macro_names'], workflow['macro_names'], 'Macro dynamics matrix')[0],
        'singular_values.png': plot_singular_value_comparison(workflow['raw_operator_singular_values'], workflow['model_whitened_singular_values'], workflow['singular_values'])[0],
        'singular_values_top15.png': plot_singular_value_comparison(workflow['raw_operator_singular_values'], workflow['model_whitened_singular_values'], workflow['singular_values'], top_n=15)[0],
        'positive_contributions.png': plot_positive_contributions(workflow['positive_contributions'])[0],
        'macro_time_series.png': plot_macro_series(workflow['macro_series'], workflow['macro_names'])[0],
        'micro_macro_comparison.png': plot_micro_macro_comparison(
            workflow['state_matrix'],
            workflow['state_names'],
            workflow['macro_series'],
            workflow['macro_names'],
        )[0],
    }
    for filename, figure in figures.items():
        figure.savefig(output_dir / filename, bbox_inches='tight')
        plt.close(figure)
        artifacts[filename] = str(output_dir / filename)

    return artifacts


__all__ = [
    'ObservableConfig',
    'RulkovSimulationConfig',
    'WorkflowConfig',
    'build_observables',
    'build_state_matrix_from_population_data',
    'channel_scores_from_singular_values',
    'compute_entropy',
    'compute_residual_covariance',
    'fit_data_koopman_operator',
    'format_equations',
    'generate_two_population_neuron_data',
    'get_positive_contributions',
    'liu2025_log_gamma_gis',
    'locate_reference_data',
    'plot_macro_series',
    'plot_micro_macro_comparison',
    'plot_matrix_heatmap',
    'plot_neuron_analysis_combo',
    'plot_rectangular_heatmap',
    'plot_singular_value_comparison',
    'plot_positive_contributions',
    'save_workflow_results',
    'set_plot_style',
    'whiten_operator_matrix',
]
