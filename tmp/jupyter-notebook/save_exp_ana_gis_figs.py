
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
import seaborn as sns

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 140
plt.rcParams['savefig.dpi'] = 220
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans']
HEATMAP_CMAP = 'vlag'
OUT = Path(r"E:\\code\\pykoop\\exp\\analysitic_exp\\figs")
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(path)


def sparse_labels(labels, step=1):
    if labels is None:
        return False
    if step <= 1:
        return labels
    return [label if i % step == 0 else '' for i, label in enumerate(labels)]


def standardize_for_plot(x):
    x = np.asarray(x, dtype=float)
    return (x - np.mean(x)) / (np.std(x) + 1e-12)


def plot_heatmap_to_ax(ax, matrix, title, row_labels=None, col_labels=None, center=0.0, label_step=1, cmap=HEATMAP_CMAP):
    sns.heatmap(
        np.asarray(matrix), ax=ax, cmap=cmap, center=center,
        square=np.asarray(matrix).shape[0] == np.asarray(matrix).shape[1],
        xticklabels=sparse_labels(col_labels, label_step),
        yticklabels=sparse_labels(row_labels, label_step),
    )
    ax.set_title(title)


def make_step_system_matrix(lam, mu):
    return np.array([[lam, 0.0, 0.0], [0.0, mu, lam ** 2 - mu], [0.0, 0.0, lam ** 2]], dtype=float)


def step_map(x, y, lam, mu):
    x_next = lam * x
    y_next = mu * y + (lam ** 2 - mu) * (x ** 2)
    return x_next, y_next


def observable_step(data_xy, mode='default'):
    data_xy = np.asarray(data_xy, dtype=float)
    x = data_xy[:, 0]
    y = data_xy[:, 1]
    if mode == 'default':
        return np.column_stack([x, y, x ** 2])
    raise ValueError(mode)


def observable_step2(data_xy, mode='default'):
    data_xy = np.asarray(data_xy, dtype=float)
    x = data_xy[:, 0]
    y = data_xy[:, 1]
    if mode == 'default':
        return np.column_stack([x, y, y ** 2])
    raise ValueError(mode)


def simulate_discrete_system(map_func, initial_states, steps, system_kwargs=None, dt=1.0):
    if system_kwargs is None:
        system_kwargs = {}
    initial_states = np.asarray(initial_states, dtype=float)
    if initial_states.ndim == 1:
        initial_states = initial_states[None, :]
    n_traj = initial_states.shape[0]
    trajectories = np.zeros((n_traj, steps + 1, 2), dtype=float)
    trajectories[:, 0, :] = initial_states
    for i in range(n_traj):
        x, y = trajectories[i, 0, :]
        for t in range(steps):
            x, y = map_func(x, y, **system_kwargs)
            trajectories[i, t + 1, 0] = x
            trajectories[i, t + 1, 1] = y
    time_grid = np.arange(steps + 1, dtype=float) * dt
    return {'trajectories': trajectories, 'time_grid': time_grid}


def add_gaussian_noise(data, noise_scale=1.0, cov=None, random_state=None):
    data = np.asarray(data, dtype=float)
    rng = np.random.default_rng(random_state)
    d = data.shape[1]
    if cov is None:
        cov = np.eye(d, dtype=float)
    noise = rng.multivariate_normal(np.zeros(d), cov, size=data.shape[0]) * noise_scale
    return {'noisy_data': data + noise, 'noise': noise}


def prepare_time_pairs(data, tau=1, burn_in=0, stride=1):
    data = np.asarray(data, dtype=float)
    start = int(burn_in)
    now = data[start:len(data) - tau:stride]
    nxt = data[start + tau::stride]
    usable = min(len(now), len(nxt))
    return now[:usable], nxt[:usable]


def _safe_symmetrize(M):
    M = np.asarray(M, dtype=float)
    return 0.5 * (M + M.T)


def _regularized_pinv(M, regularization=1e-10):
    M = np.asarray(M, dtype=float)
    return np.linalg.pinv(M + regularization * np.eye(M.shape[0], dtype=float))


def _pseudo_logdet_positive(M, eps=1e-10):
    eigvals = np.linalg.eigvalsh(_safe_symmetrize(M))
    positive = eigvals[eigvals > eps]
    if positive.size == 0:
        return -np.inf, positive
    return float(np.sum(np.log(positive))), positive


def estimate_covariance_from_residuals(residuals, center=True, regularization=1e-10):
    residuals = np.asarray(residuals, dtype=float)
    res = residuals - residuals.mean(axis=0, keepdims=True) if center else residuals
    cov = (res.T @ res) / max(res.shape[0] - 1, 1)
    cov = _safe_symmetrize(cov)
    cov += regularization * np.eye(cov.shape[0], dtype=float)
    return cov


def fit_linear_gis_from_pairs(X_now, X_next, fit_intercept=False, ridge=0.0, regularization=1e-10):
    X_now = np.asarray(X_now, dtype=float)
    X_next = np.asarray(X_next, dtype=float)
    n_samples, dim = X_now.shape
    x_now_center = X_now.copy()
    x_next_center = X_next.copy()
    intercept = np.zeros(dim, dtype=float)
    if fit_intercept:
        mean_now = X_now.mean(axis=0)
        mean_next = X_next.mean(axis=0)
        x_now_center = X_now - mean_now
        x_next_center = X_next - mean_next
    else:
        mean_now = np.zeros(dim, dtype=float)
        mean_next = np.zeros(dim, dtype=float)
    C00 = (x_now_center.T @ x_now_center) / n_samples
    C01 = (x_now_center.T @ x_next_center) / n_samples
    C11 = (x_next_center.T @ x_next_center) / n_samples
    C00_reg = C00 + (ridge + regularization) * np.eye(dim, dtype=float)
    K_raw = _regularized_pinv(C00_reg, regularization=regularization) @ C01
    A = K_raw.T
    if fit_intercept:
        intercept = mean_next - A @ mean_now
    predictions = (X_now @ A.T) + intercept
    residuals = X_next - predictions
    Sigma = estimate_covariance_from_residuals(residuals, center=True, regularization=regularization)
    return {'A': A, 'K_raw': K_raw, 'residuals': residuals, 'Sigma': Sigma, 'C00': C00, 'C01': C01, 'C11': C11, 'intercept': intercept}


def fit_linear_gis_from_matrix(A, Sigma=None, sigma_eps=1e-10):
    A = np.asarray(A, dtype=float)
    d = A.shape[0]
    if Sigma is None:
        Sigma = sigma_eps * np.eye(d, dtype=float)
    else:
        Sigma = _safe_symmetrize(np.asarray(Sigma, dtype=float))
    return {'A': A, 'Sigma': Sigma, 'state_dim': d}


def compute_gis_metrics(A, Sigma, alpha=1.0, eps=1e-10):
    A = np.asarray(A, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    d = A.shape[0]
    Sigma = _safe_symmetrize(Sigma) + eps * np.eye(d, dtype=float)
    Sigma_inv = _regularized_pinv(Sigma, regularization=eps)
    backward = _safe_symmetrize(A.T @ Sigma_inv @ A)
    D, _ = _pseudo_logdet_positive(Sigma_inv, eps=eps)
    N, _ = _pseudo_logdet_positive(backward, eps=eps)
    log_Gamma = (0.5 - alpha / 4.0) * N + (alpha / 4.0) * D
    Gamma = float(np.exp(log_Gamma)) if np.isfinite(log_Gamma) else 0.0
    J_alpha = log_Gamma / d
    sv_forward = np.linalg.svd(Sigma_inv, compute_uv=False)
    sv_backward = np.linalg.svd(backward, compute_uv=False)
    return {'Gamma': Gamma, 'log_Gamma': float(log_Gamma), 'J_alpha': float(J_alpha), 'D': float(D), 'N': float(N), 'Sigma_inv': Sigma_inv, 'A_t_Sigma_inv_A': backward, 'sv_forward': np.real_if_close(sv_forward), 'sv_backward': np.real_if_close(sv_backward), 'alpha': float(alpha), 'dimension': int(d)}


def predict_linear_gis(A, X0, steps=1):
    A = np.asarray(A, dtype=float)
    X0 = np.asarray(X0, dtype=float)
    if X0.ndim == 1:
        X0 = X0[None, :]
    A_power = np.linalg.matrix_power(A, steps)
    return X0 @ A_power.T


def compute_prediction_errors(A, series, tau=1, horizons=(1,)):
    series = np.asarray(series, dtype=float)
    results = {}
    for horizon in horizons:
        shift = horizon * tau
        x_now = series[:-shift]
        x_target = series[shift:]
        preds = predict_linear_gis(A, x_now, steps=horizon)
        pointwise = np.sum((x_target - preds) ** 2, axis=1)
        results[horizon] = {'predictions': preds, 'targets': x_target, 'pointwise_errors': pointwise, 'mean_error': float(np.mean(pointwise))}
    return results


def compute_ce_from_micro_macro(micro_metrics, macro_metrics):
    delta_J = macro_metrics['J_alpha'] - micro_metrics['J_alpha']
    return {'CE': float(delta_J), 'delta_J': float(delta_J), 'delta_D': float(macro_metrics['D'] - micro_metrics['D']), 'delta_N': float(macro_metrics['N'] - micro_metrics['N']), 'delta_log_Gamma': float(macro_metrics['log_Gamma'] - micro_metrics['log_Gamma'])}


def select_macro_rank(values, mode='gap', threshold=None, manual_r=None, eps=1e-10):
    values = np.asarray(values, dtype=float).ravel()
    if mode == 'manual' and manual_r is not None:
        return int(manual_r), {'mode': 'manual', 'manual_r': int(manual_r)}
    positive = np.sum(values > eps)
    return max(1, int(positive)), {'mode': mode, 'effective_rank': int(positive)}


def build_w_from_svd(A, Sigma, r=None, alpha=1.0, eps=1e-10, mode='two_stage'):
    metrics = compute_gis_metrics(A, Sigma, alpha=alpha, eps=eps)
    sigma_inv = metrics['Sigma_inv']
    backward = metrics['A_t_Sigma_inv_A']
    sv_forward = metrics['sv_forward']
    sv_backward = metrics['sv_backward']
    if r is None:
        r = select_macro_rank(sv_backward, mode='gap', eps=eps)[0]
    if mode == 'backward_only':
        U_b, _, _ = np.linalg.svd(backward, full_matrices=False)
        basis = U_b[:, :r]
    else:
        U_f, S_f, _ = np.linalg.svd(sigma_inv, full_matrices=False)
        U_b, S_b, _ = np.linalg.svd(backward, full_matrices=False)
        combined_vectors = np.concatenate([U_f, U_b], axis=1)
        combined_scores = np.concatenate([S_f, S_b], axis=0)
        keep = combined_scores > eps
        weighted_vectors = combined_vectors[:, keep] * combined_scores[keep]
        U_c, _, _ = np.linalg.svd(weighted_vectors, full_matrices=False)
        basis = U_c[:, :r]
    W = basis.T
    return {'W': np.real_if_close(W), 'r': int(r), 'sv_info': {'sv_forward': np.real_if_close(sv_forward), 'sv_backward': np.real_if_close(sv_backward)}}


def build_w_from_evd(A, r=None, mode='eig_abs'):
    A = np.asarray(A, dtype=float)
    eigvals, eigvecs = np.linalg.eig(A)
    if mode == 'eig_abs':
        order = np.argsort(-np.abs(eigvals))
    else:
        order = np.argsort(-np.real(eigvals))
    eigvals_sorted = eigvals[order]
    eigvecs_sorted = eigvecs[:, order]
    if r is None:
        r = 1
    basis = np.real_if_close(eigvecs_sorted[:, :r])
    basis = np.asarray(np.real(basis), dtype=float)
    q, _ = np.linalg.qr(basis)
    W = q[:, :r].T
    return {'W': np.real_if_close(W), 'r': int(r), 'eigvals': np.real_if_close(eigvals_sorted), 'eigvecs': np.real_if_close(eigvecs_sorted)}


def apply_coarse_graining(W, O):
    W = np.asarray(W, dtype=float)
    O = np.asarray(O, dtype=float)
    return O @ W.T


def generate_parameter_scan_results(lam_values=None, log_min=-4, log_max=4, n_points=120):
    if lam_values is None:
        lam_values = [0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0]
    ratios = 10 ** np.linspace(log_min, log_max, n_points)
    results = {}
    for lam in lam_values:
        results[lam] = {'ratios': ratios, 'sv1': [], 'sv2': [], 'sv3': []}
        for ratio in ratios:
            mu = lam / ratio
            sv = np.sort(np.linalg.svd(make_step_system_matrix(lam, mu), compute_uv=False))[::-1]
            results[lam]['sv1'].append(sv[0]); results[lam]['sv2'].append(sv[1]); results[lam]['sv3'].append(sv[2])
    return results, lam_values, ratios


def project_matrix_with_w(A, Sigma, W):
    return W @ A @ W.T, W @ Sigma @ W.T


def run_pipeline(observations, tau=1, alpha=1.0, delta=None, eps=1e-10, ridge=1e-10, manual_r=1, horizons=(1,3,5), w_method='svd', evd_mode='eig_abs'):
    X_now, X_next = prepare_time_pairs(observations, tau=tau, burn_in=0, stride=1)
    micro_fit = fit_linear_gis_from_pairs(X_now, X_next, fit_intercept=False, ridge=ridge, regularization=eps)
    micro_metrics = compute_gis_metrics(micro_fit['A'], micro_fit['Sigma'], alpha=alpha, eps=eps)
    micro_errors = compute_prediction_errors(micro_fit['A'], observations, tau=tau, horizons=horizons)
    rank_values = micro_metrics['sv_backward'] if w_method == 'svd' else np.abs(np.linalg.eigvals(micro_fit['A']))
    r, rank_meta = select_macro_rank(rank_values, mode='manual', manual_r=manual_r, eps=eps)
    w_result = build_w_from_svd(micro_fit['A'], micro_fit['Sigma'], r=r, alpha=alpha, eps=eps, mode='two_stage') if w_method == 'svd' else build_w_from_evd(micro_fit['A'], r=r, mode=evd_mode)
    W = w_result['W']
    macro_observations = apply_coarse_graining(W, observations)
    Z_now, Z_next = prepare_time_pairs(macro_observations, tau=tau, burn_in=0, stride=1)
    macro_fit = fit_linear_gis_from_pairs(Z_now, Z_next, fit_intercept=False, ridge=ridge, regularization=eps)
    macro_metrics = compute_gis_metrics(macro_fit['A'], macro_fit['Sigma'], alpha=alpha, eps=eps)
    macro_errors = compute_prediction_errors(macro_fit['A'], macro_observations, tau=tau, horizons=horizons)
    ce_result = compute_ce_from_micro_macro(micro_metrics, macro_metrics)
    return {'micro_fit': micro_fit, 'micro_metrics': micro_metrics, 'micro_errors': micro_errors, 'W': W, 'macro_observations': macro_observations, 'macro_fit': macro_fit, 'macro_metrics': macro_metrics, 'macro_errors': macro_errors, 'ce_result': ce_result, 'rank_meta': rank_meta}

# figure generation
scan_results, scan_lam_values, scan_ratios = generate_parameter_scan_results()
rows = []
for lam in scan_lam_values:
    for i, ratio in enumerate(scan_ratios):
        lbl = f"$10^{{{np.log10(ratio):.0f}}}$"
        rows += [
            {'Ratio': lbl, 'Singular Value': scan_results[lam]['sv1'][i], 'Type': '$\\sigma_1$'},
            {'Ratio': lbl, 'Singular Value': scan_results[lam]['sv2'][i], 'Type': '$\\sigma_2$'},
            {'Ratio': lbl, 'Singular Value': scan_results[lam]['sv3'][i], 'Type': '$\\sigma_3$'},
        ]
df = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(7.5, 5.2))
sns.boxplot(data=df, x='Ratio', y='Singular Value', hue='Type', palette='Set2', ax=ax)
ax.set_yscale('log')
ax.set_xlabel('$\\lambda / \\mu$ ratio')
ax.set_ylabel('Singular value')
ax.set_title('?????????????????')
save(fig, 'exp_ana_gis_part1_spectrum_boxplot.png')

phase_lam, phase_mu, phase_steps, num_samples = 0.1, 0.9, 50, 60
rng = np.random.default_rng(42)
initial_points = np.vstack([rng.uniform(-0.85, 0.85, num_samples), rng.uniform(-0.7, 0.85, num_samples)]).T
fig, ax = plt.subplots(figsize=(7, 6))
custom_cmap = ListedColormap(plt.cm.YlGnBu(np.linspace(0.25, 0.95, 8)))
for x0, y0 in initial_points:
    traj = simulate_discrete_system(step_map, [x0, y0], steps=phase_steps, system_kwargs={'lam': phase_lam, 'mu': phase_mu}, dt=1.0)['trajectories'][0]
    points = traj.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    color_idx = np.arange(len(segments)); color_idx[np.where(color_idx > 20)] = 20
    lc = LineCollection(segments, cmap=custom_cmap)
    lc.set_array(color_idx); lc.set_linewidth(1.4); lc.set_capstyle('round')
    ax.add_collection(lc)
ax.set_title(f'???????????lambda={phase_lam}, mu={phase_mu}')
ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_xlim([-0.95, 0.95]); ax.set_ylim([-0.85, 0.95]); ax.grid(False)
save(fig, 'exp_ana_gis_part1_phase.png')

feature_names = ['$x$', '$y$', '$x^2$']
A_direct = make_step_system_matrix(0.1, 0.9)
direct_fit = fit_linear_gis_from_matrix(A_direct, Sigma=None, sigma_eps=1e-10)
Sigma_direct = direct_fit['Sigma']
direct_metrics = compute_gis_metrics(A_direct, Sigma_direct, alpha=1.0, eps=1e-10)
r_direct, _ = select_macro_rank(direct_metrics['sv_backward'], mode='manual', manual_r=1, eps=1e-10)
W_direct = build_w_from_svd(A_direct, Sigma_direct, r=r_direct, alpha=1.0, eps=1e-10, mode='two_stage')['W']
fig, ax = plt.subplots(figsize=(5.8, 5.2))
plot_heatmap_to_ax(ax, A_direct, '???? A?????????', row_labels=feature_names, col_labels=feature_names, label_step=1)
save(fig, 'exp_ana_gis_part2_A_heatmap.png')
fig, ax = plt.subplots(figsize=(4.6, 3.8))
plot_heatmap_to_ax(ax, np.abs(W_direct), '????? W ???????', row_labels=[r'$z_1$'], col_labels=feature_names, center=None, cmap='Blues')
save(fig, 'exp_ana_gis_part2_W_heatmap.png')

analysis_config = {'lam': 0.1, 'mu': 0.9, 'initial_state': [5.0, 5.0], 'steps': 600, 'dt': 1.0, 'tau': 1, 'delta': None, 'alpha': 1.0, 'noise_scale': 0.05, 'noise_seed': 42, 'eps': 1e-10, 'ridge': 1e-10, 'manual_r': 1, 'horizons': (1, 3, 5)}
clean_sim = simulate_discrete_system(step_map, analysis_config['initial_state'], steps=analysis_config['steps'], system_kwargs={'lam': analysis_config['lam'], 'mu': analysis_config['mu']}, dt=analysis_config['dt'])
clean_xy = clean_sim['trajectories'][0]
time_grid = clean_sim['time_grid']
noisy_xy = add_gaussian_noise(clean_xy, noise_scale=analysis_config['noise_scale'], cov=None, random_state=analysis_config['noise_seed'])['noisy_data']
obs_part3 = observable_step(noisy_xy, mode='default')
res3 = run_pipeline(obs_part3, tau=1, alpha=1.0, delta=None, eps=1e-10, ridge=1e-10, manual_r=1, horizons=(1,3,5), w_method='svd')
macro_names_3 = [f'$z_{i+1}$' for i in range(res3['W'].shape[0])]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(time_grid[:120], clean_xy[:120,0], label='clean $x$')
ax.plot(time_grid[:120], noisy_xy[:120,0], '--', alpha=0.8, label='noisy $x$')
ax.plot(time_grid[:120], clean_xy[:120,1], label='clean $y$')
ax.plot(time_grid[:120], noisy_xy[:120,1], '--', alpha=0.8, label='noisy $y$')
ax.set_title('???????????????')
ax.set_xlabel('Time'); ax.set_ylabel('State value'); ax.legend(ncol=2)
save(fig, 'exp_ana_gis_part3_noisy_vs_clean.png')

fig = plt.figure(figsize=(11, 7))
gs = GridSpec(2, 1, figure=fig, hspace=0.35)
ax1 = fig.add_subplot(gs[0, 0])
pred1_micro, target1_micro = res3['micro_errors'][1]['predictions'], res3['micro_errors'][1]['targets']
for idx, name in enumerate(feature_names):
    ax1.plot(np.arange(len(pred1_micro[:80])), target1_micro[:80, idx], label=f'true {name}')
    ax1.plot(np.arange(len(pred1_micro[:80])), pred1_micro[:80, idx], '--', linewidth=1.5, label=f'pred {name}')
ax1.set_title('????????????????')
ax1.set_xlabel('Pair index'); ax1.set_ylabel('Value'); ax1.legend(ncol=3, fontsize=8)
ax2 = fig.add_subplot(gs[1, 0])
pred1_macro, target1_macro = res3['macro_errors'][1]['predictions'], res3['macro_errors'][1]['targets']
for idx, name in enumerate(macro_names_3):
    ax2.plot(np.arange(len(pred1_macro[:80])), target1_macro[:80, idx], label=f'true {name}')
    ax2.plot(np.arange(len(pred1_macro[:80])), pred1_macro[:80, idx], '--', linewidth=2.0, label=f'pred {name}')
ax2.set_title('????????????????')
ax2.set_xlabel('Pair index'); ax2.set_ylabel('Value'); ax2.legend(fontsize=8)
save(fig, 'exp_ana_gis_part3_prediction.png')

fig, ax = plt.subplots(figsize=(10, 4.5))
for idx, name in enumerate(feature_names):
    ax.plot(np.arange(120), standardize_for_plot(obs_part3[:120, idx]), linewidth=1.4, label=f'micro: {name}')
for idx, name in enumerate(macro_names_3):
    ax.plot(np.arange(120), standardize_for_plot(res3['macro_observations'][:120, idx]), '--', linewidth=2.2, label=f'macro: {name}')
ax.set_title('????????????')
ax.set_xlabel('Time index'); ax.set_ylabel('Standardized value'); ax.legend(ncol=2)
save(fig, 'exp_ana_gis_part3_micro_macro_curve.png')


def step2_local(x, y, a=0.8, coupling=10.0):
    return a * x + coupling * (y ** 2), a * y

step2_config = {'initial_state': [0.2, 0.45], 'steps': 220, 'dt': 1.0, 'tau': 1, 'delta': None, 'alpha': 1.0, 'noise_scale': 0.02, 'noise_seed': 7, 'eps': 1e-10, 'ridge': 1e-10, 'manual_r': 1, 'horizons': (1, 3, 5)}
step2_sim = simulate_discrete_system(step2_local, step2_config['initial_state'], steps=step2_config['steps'], system_kwargs={'a': 0.8, 'coupling': 10.0}, dt=step2_config['dt'])
step2_clean = step2_sim['trajectories'][0]
step2_noisy = add_gaussian_noise(step2_clean, noise_scale=step2_config['noise_scale'], cov=None, random_state=step2_config['noise_seed'])['noisy_data']
obs_step2 = observable_step2(step2_noisy, mode='default')
res_step2_svd = run_pipeline(obs_step2, tau=1, alpha=1.0, delta=None, eps=1e-10, ridge=1e-10, manual_r=1, horizons=(1,3,5), w_method='svd')
res_step2_evd = run_pipeline(obs_step2, tau=1, alpha=1.0, delta=None, eps=1e-10, ridge=1e-10, manual_r=1, horizons=(1,3,5), w_method='evd', evd_mode='eig_abs')
step2_feature_names = ['$x$', '$y$', '$y^2$']
step2_macro_names = [f'$z_{i+1}$' for i in range(res_step2_svd['W'].shape[0])]

fig, ax = plt.subplots(figsize=(7, 6))
rng = np.random.default_rng(42)
initial_points = np.vstack([rng.uniform(-0.3, 0.3, 60), rng.uniform(-0.5, 0.5, 60)]).T
custom_cmap = ListedColormap(plt.cm.YlGnBu(np.linspace(0.3, 0.95, 6)))
for x0, y0 in initial_points:
    traj = simulate_discrete_system(step2_local, [x0, y0], steps=120, system_kwargs={'a': 0.8, 'coupling': 10.0}, dt=1.0)['trajectories'][0]
    points = traj.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    color_idx = np.arange(len(segments)); color_idx[np.where(color_idx > 40)] = 40
    lc = LineCollection(segments, cmap=custom_cmap)
    lc.set_array(color_idx); lc.set_linewidth(1.6); lc.set_capstyle('round'); ax.add_collection(lc)
ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_title('step2 ??????')
ax.set_xlim([-0.5, 3.5]); ax.set_ylim([-0.6, 0.6]); ax.grid(False)
save(fig, 'exp_ana_gis_part4_step2_phase.png')

A_step2_micro = res_step2_svd['micro_fit']['A']
sv_step2 = np.linalg.svd(A_step2_micro, compute_uv=False)
eig_abs_step2 = np.sort(np.abs(np.linalg.eigvals(A_step2_micro)))[::-1]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.arange(1, len(sv_step2)+1), sv_step2, marker='o', linewidth=1.8, label='????', color='tab:red')
ax.plot(np.arange(1, len(eig_abs_step2)+1), eig_abs_step2, marker='s', linewidth=1.4, linestyle='--', label='????????', color='tab:blue')
ax.set_title('step2????????????')
ax.set_xlabel('????'); ax.set_ylabel('????'); ax.legend()
save(fig, 'exp_ana_gis_part4_svd_vs_evd_spectrum.png')

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
plot_heatmap_to_ax(axes[0], np.abs(res_step2_svd['W']), 'SVD ??????? |W|', row_labels=step2_macro_names, col_labels=step2_feature_names, center=None, cmap='Blues')
plot_heatmap_to_ax(axes[1], np.abs(res_step2_evd['W']), 'EVD ??????? |W|', row_labels=step2_macro_names, col_labels=step2_feature_names, center=None, cmap='Blues')
save(fig, 'exp_ana_gis_part4_W_compare.png')

fig, ax = plt.subplots(figsize=(10, 4.5))
for idx, name in enumerate(step2_feature_names):
    ax.plot(np.arange(120), standardize_for_plot(obs_step2[:120, idx]), linewidth=1.4, label=f'micro: {name}')
for idx in range(res_step2_svd['macro_observations'].shape[1]):
    ax.plot(np.arange(120), standardize_for_plot(res_step2_svd['macro_observations'][:120, idx]), '--', linewidth=2.0, color=f'C{idx+3}', label=f'SVD macro: $z_{idx+1}$')
for idx in range(res_step2_evd['macro_observations'].shape[1]):
    ax.plot(np.arange(120), standardize_for_plot(res_step2_evd['macro_observations'][:120, idx]), '-.', linewidth=2.0, color=f'C{idx+6}', label=f'EVD macro: $z_{idx+1}$')
ax.set_title('step2?SVD / EVD ???????????')
ax.set_xlabel('Time index'); ax.set_ylabel('Standardized value'); ax.legend(ncol=2)
save(fig, 'exp_ana_gis_part4_curve_compare.png')

print('DONE')
