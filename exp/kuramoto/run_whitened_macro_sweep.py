#!/opt/anaconda3/envs/py311/bin/python
import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pysindy as ps
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from data_generators.data_func import generate_kuramoto_cluster_data_sin_cos
from tools.tools import fit_data_koopman_operator


def build_library(kind, fourier_n_frequencies=1, poly_degree=2):
    ide = ps.IdentityLibrary()
    fourier = ps.FourierLibrary(n_frequencies=fourier_n_frequencies)
    poly = ps.PolynomialLibrary(degree=poly_degree, include_bias=False)

    if kind == "identity":
        return ide
    if kind == "fourier":
        return fourier
    if kind == "identity+fourier":
        return ide + fourier
    if kind == "identity+polynomial":
        return ide + poly
    raise ValueError(f"Unsupported library kind: {kind}")


def parse_feature_base_index(name):
    if "x" not in name:
        return None
    try:
        return int(name.split("x")[-1].rstrip(")"))
    except ValueError:
        return None


def standardize_signal(x):
    x = np.asarray(x)
    return (x - np.mean(x)) / (np.std(x) + 1e-12)


def dominant_frequency(signal, dt):
    signal = standardize_signal(signal)
    n_fft = len(signal)
    xf = np.fft.fftfreq(n_fft, dt)
    yf = np.fft.fft(signal)
    positive = xf > 0
    freqs = xf[positive]
    amps = np.abs(yf[positive]) / n_fft * 2
    if len(freqs) == 0:
        return 0.0
    return float(freqs[np.argmax(amps)])


def best_assignment_score(matrix, maximize=True):
    n_rows, n_cols = matrix.shape
    size = min(n_rows, n_cols)
    best_score = None
    best_perm = None
    for cols in itertools.permutations(range(n_cols), size):
        score = float(sum(matrix[i, cols[i]] for i in range(size)))
        if best_score is None:
            best_score = score
            best_perm = cols
            continue
        if maximize and score > best_score:
            best_score = score
            best_perm = cols
        if not maximize and score < best_score:
            best_score = score
            best_perm = cols
    return best_score, best_perm


def compute_cluster_signals(x_eval, n_clusters):
    n_total = x_eval.shape[1]
    n_osc = n_total // 2
    cluster_size = n_osc // n_clusters

    signals = []
    for c in range(n_clusters):
        start = c * cluster_size
        end = n_osc if c == n_clusters - 1 else (c + 1) * cluster_size
        cluster_cos = np.mean(x_eval[:, start:end], axis=1)
        signals.append(standardize_signal(cluster_cos))
    return np.column_stack(signals)


def compute_macro_cluster_alignment(macro_data, cluster_signals):
    macro_dim = macro_data.shape[1]
    n_clusters = cluster_signals.shape[1]
    corr = np.zeros((macro_dim, n_clusters))
    freq_diff = np.zeros((macro_dim, n_clusters))

    for i in range(macro_dim):
        macro_sig = standardize_signal(macro_data[:, i])
        macro_freq = dominant_frequency(macro_sig, dt=1.0)  # overwritten below
        for j in range(n_clusters):
            cluster_sig = standardize_signal(cluster_signals[:, j])
            corr[i, j] = abs(np.corrcoef(macro_sig, cluster_sig)[0, 1])
            freq_diff[i, j] = macro_freq  # placeholder
    return corr, freq_diff


def compute_macro_metrics(
    params,
    x_eval,
    x_eval_lift,
    names,
    koop_fit,
    singular_values,
    rank_target,
):
    U, _, _ = np.linalg.svd(koop_fit["K_bar"])
    coarse_grain_coff = koop_fit["C00_inv_sqrt"] @ U[:, :rank_target]
    macro_data = x_eval_lift @ coarse_grain_coff
    macro_data = np.real_if_close(macro_data)

    cluster_signals = compute_cluster_signals(x_eval, params["n_clusters"])
    macro_freqs = [dominant_frequency(macro_data[:, i], params["dt_effective"]) for i in range(rank_target)]
    cluster_freqs = [dominant_frequency(cluster_signals[:, j], params["dt_effective"]) for j in range(params["n_clusters"])]

    freq_diff_matrix = np.zeros((rank_target, params["n_clusters"]))
    corr_matrix = np.zeros((rank_target, params["n_clusters"]))
    for i in range(rank_target):
        macro_sig = standardize_signal(macro_data[:, i])
        for j in range(params["n_clusters"]):
            cluster_sig = standardize_signal(cluster_signals[:, j])
            freq_diff_matrix[i, j] = abs(macro_freqs[i] - cluster_freqs[j])
            corr_matrix[i, j] = abs(np.corrcoef(macro_sig, cluster_sig)[0, 1])

    best_freq_mismatch, best_freq_perm = best_assignment_score(freq_diff_matrix, maximize=False)
    best_corr, best_corr_perm = best_assignment_score(corr_matrix, maximize=True)

    cluster_size = (x_eval.shape[1] // 2) // params["n_clusters"]
    loading_matrix = np.zeros((rank_target, params["n_clusters"]))
    for feat_idx, name in enumerate(names):
        base_idx = parse_feature_base_index(name)
        if base_idx is None:
            continue
        osc_idx = base_idx % (x_eval.shape[1] // 2)
        cluster_idx = min(osc_idx // cluster_size, params["n_clusters"] - 1)
        for macro_idx in range(rank_target):
            loading_matrix[macro_idx, cluster_idx] += abs(coarse_grain_coff[feat_idx, macro_idx])

    cluster_contrast = []
    for macro_idx in range(rank_target):
        sorted_loads = np.sort(loading_matrix[macro_idx])[::-1]
        if len(sorted_loads) >= 2:
            cluster_contrast.append(float(sorted_loads[0] / (sorted_loads[1] + 1e-12)))
        else:
            cluster_contrast.append(0.0)

    sigma2 = float(singular_values[rank_target - 1])
    sigma3 = float(singular_values[rank_target]) if len(singular_values) > rank_target else 0.0
    near_one_count = int(np.sum(singular_values > 0.99))
    score = (
        4.0 * max(sigma2 - sigma3, 0.0)
        - 1.5 * float(best_freq_mismatch)
        + 0.5 * float(best_corr / rank_target)
        - 0.08 * max(near_one_count - rank_target, 0)
        + 0.05 * np.mean(cluster_contrast)
    )

    return {
        "macro_freqs": macro_freqs,
        "cluster_freqs": cluster_freqs,
        "best_freq_mismatch": float(best_freq_mismatch / rank_target),
        "best_freq_perm": list(best_freq_perm) if best_freq_perm is not None else None,
        "best_cluster_corr": float(best_corr / rank_target),
        "best_corr_perm": list(best_corr_perm) if best_corr_perm is not None else None,
        "cluster_contrast_mean": float(np.mean(cluster_contrast)),
        "loading_matrix": loading_matrix.tolist(),
        "score": float(score),
        "coarse_grain_coff": coarse_grain_coff.tolist(),
    }


def generate_trajectories(params):
    trajectories = []
    theta_histories = []
    seeds = list(range(params["n_trajectories"]))
    for seed_idx in seeds:
        x_embed, theta_hist, _, _ = generate_kuramoto_cluster_data_sin_cos(
            N=params["N"],
            n_clusters=params["n_clusters"],
            K_intra=params["K_intra"],
            K_inter=params["K_inter"],
            dt=params["dt"],
            T=params["T"],
            noise=params["noise"],
            random_seed1=0,
            random_seed2=seed_idx,
        )
        burn_in = params["burn_in"]
        stride = params["sample_stride"]
        trajectories.append(x_embed[burn_in::stride].copy())
        theta_histories.append(theta_hist[burn_in::stride].copy())
    return trajectories, theta_histories


def transform_trajectories(trajectories, library, center_features):
    all_samples = np.vstack(trajectories)
    library.fit(all_samples)
    transformed = [library.transform(traj) for traj in trajectories]
    names = library.get_feature_names()

    if center_features:
        mean_vec = np.mean(np.vstack(transformed), axis=0, keepdims=True)
        transformed = [traj - mean_vec for traj in transformed]
    return transformed, names


def preprocess_features(trajectories_lift, names, params):
    preprocess = params.get("preprocess", "none")
    if preprocess == "none":
        return trajectories_lift, names, None

    if preprocess != "pca":
        raise ValueError(f"Unsupported preprocess mode: {preprocess}")

    pca_dim = int(params["pca_dim"])
    all_samples = np.vstack(trajectories_lift)
    pca = PCA(n_components=pca_dim)
    pca.fit(all_samples)
    transformed = [pca.transform(traj) for traj in trajectories_lift]
    pca_names = [f"pc{i + 1}" for i in range(pca_dim)]
    return transformed, pca_names, pca


def run_config(config):
    params = dict(config)
    params.setdefault("poly_degree", 2)
    params.setdefault("fourier_n_frequencies", 1)
    params.setdefault("center_features", False)
    params.setdefault("n_trajectories", 1)
    params.setdefault("weights_mode", "uniform")
    params.setdefault("rank_target", params["n_clusters"])
    params.setdefault("preprocess", "none")
    params["dt_effective"] = params["dt"] * params["sample_stride"]

    trajectories, theta_histories = generate_trajectories(params)
    library = build_library(
        params["library_kind"],
        fourier_n_frequencies=params["fourier_n_frequencies"],
        poly_degree=params["poly_degree"],
    )
    trajectories_lift, names = transform_trajectories(
        trajectories,
        library=library,
        center_features=params["center_features"],
    )
    trajectories_fit, fit_names, pca = preprocess_features(trajectories_lift, names, params)

    koop_fit = fit_data_koopman_operator(
        trajectories_fit,
        weights=params["weights_mode"],
        eps=1e-10,
        ridge=1e-10,
        lag_steps=params["lag_steps"],
    )
    singular_values = np.linalg.svd(koop_fit["K_bar"], compute_uv=False)
    sigma3 = float(singular_values[params["rank_target"]]) if len(singular_values) > params["rank_target"] else 0.0
    metrics = {
        "sigma_1": float(singular_values[0]),
        "sigma_2": float(singular_values[1]) if len(singular_values) > 1 else 0.0,
        "sigma_3": sigma3,
        "gap_2": float(singular_values[params["rank_target"] - 1] - sigma3),
        "near_one_count_099": int(np.sum(singular_values > 0.99)),
        "near_one_count_095": int(np.sum(singular_values > 0.95)),
        "cond_C00": float(np.linalg.cond(koop_fit["C00"])),
        "cond_C11": float(np.linalg.cond(koop_fit["C11"])),
        "feature_dim": int(trajectories_fit[0].shape[1]),
        "n_pairs_total": int(sum(koop_fit["pair_counts"])),
        "preprocess_explained_variance_ratio": (
            [float(x) for x in pca.explained_variance_ratio_]
            if pca is not None
            else None
        ),
    }
    macro_metrics = compute_macro_metrics(
        params,
        x_eval=trajectories[0],
        x_eval_lift=trajectories_fit[0],
        names=fit_names,
        koop_fit=koop_fit,
        singular_values=singular_values,
        rank_target=params["rank_target"],
    )
    metrics.update(macro_metrics)
    metrics["top10_singular_values"] = [float(x) for x in singular_values[:10]]
    return params, metrics


def append_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--results-file", type=Path, required=True)
    parser.add_argument("--history-file", type=Path, default=Path("experiments/run_history.jsonl"))
    parser.add_argument("--objective", type=str, default="maximize rank-2 spectral gap and macro-cluster frequency alignment")
    args = parser.parse_args()

    configs = json.loads(args.config_file.read_text())
    results_records = []
    history_records = []

    for idx, config in enumerate(configs):
        run_name = config.get("run_name", f"kuramoto_whitened_{idx:04d}")
        command = (
            f"/opt/anaconda3/envs/py311/bin/python {Path(__file__).relative_to(REPO_ROOT)} "
            f"--config-file {args.config_file} --results-file {args.results_file}"
        )
        try:
            params, metrics = run_config(config)
            result = {
                "run_name": run_name,
                "status": "completed",
                "params": params,
                "metrics": metrics,
            }
            results_records.append(result)
            history_records.append(
                {
                    "run_name": run_name,
                    "status": "completed",
                    "command": command,
                    "objective": args.objective,
                    "params": params,
                    "metrics": {
                        "score": metrics["score"],
                        "gap_2": metrics["gap_2"],
                        "best_freq_mismatch": metrics["best_freq_mismatch"],
                        "best_cluster_corr": metrics["best_cluster_corr"],
                        "near_one_count_099": metrics["near_one_count_099"],
                    },
                    "artifacts": {
                        "results_file": str(args.results_file),
                    },
                    "notes": (
                        f"library={params['library_kind']}, lag={params['lag_steps']}, "
                        f"stride={params['sample_stride']}, n_traj={params['n_trajectories']}, "
                        f"preprocess={params['preprocess']}"
                    ),
                }
            )
            print(
                f"[completed] {run_name}: score={metrics['score']:.4f}, "
                f"gap_2={metrics['gap_2']:.4f}, "
                f"freq_mismatch={metrics['best_freq_mismatch']:.4f}, "
                f"near1={metrics['near_one_count_099']}"
            )
        except Exception as exc:
            history_records.append(
                {
                    "run_name": run_name,
                    "status": "failed",
                    "command": command,
                    "objective": args.objective,
                    "params": config,
                    "metrics": {},
                    "artifacts": {
                        "results_file": str(args.results_file),
                    },
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[failed] {run_name}: {type(exc).__name__}: {exc}")

    append_jsonl(args.results_file, results_records)
    append_jsonl(args.history_file, history_records)


if __name__ == "__main__":
    main()
