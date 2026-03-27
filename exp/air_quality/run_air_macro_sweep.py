import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd


def _install_plotly_stub():
    plotly_mod = types.ModuleType("plotly")
    express_mod = types.ModuleType("plotly.express")
    graph_objects_mod = types.ModuleType("plotly.graph_objects")
    plotly_mod.express = express_mod
    plotly_mod.graph_objects = graph_objects_mod
    sys.modules.setdefault("plotly", plotly_mod)
    sys.modules.setdefault("plotly.express", express_mod)
    sys.modules.setdefault("plotly.graph_objects", graph_objects_mod)


def _patch_numpy_concatenate():
    orig_concatenate = np.concatenate

    def concatenate_compat(arrays, axis=0, out=None, dtype=None, casting=None):
        result = orig_concatenate(arrays, axis=axis, out=out)
        if dtype is not None:
            result = result.astype(dtype, copy=False)
        return result

    np.concatenate = concatenate_compat


_install_plotly_stub()
_patch_numpy_concatenate()

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tools.tools import (  # noqa: E402
    analyze_kbar_metrics,
    build_air_feature_matrix,
    build_macro_from_kbar,
    compute_entropy,
    compute_transition_covariances,
    fit_data_koopman_operator,
    get_positive_contributions,
    lift_time_delay,
    load_air_data_subset,
)


def normalize_air_variable_name(name):
    mapping = {"PM2.5": "pm25", "O3": "o3"}
    name = str(name)
    if name in mapping:
        return mapping[name]
    return name.lower().replace(".", "").replace(" ", "_")


def build_normalized_feature_names(feature_names):
    normalized = []
    for feature_name in feature_names:
        text = str(feature_name)
        text = text.replace("_PM2.5", "_pm25").replace("_O3", "_o3")
        normalized.append(text)
    return normalized


def compute_air_run(config):
    start = time.perf_counter()

    air_subset = load_air_data_subset(
        dataset_path=config["dataset_path"],
        station_meta_path=config["station_meta_path"],
        subset_mode=config["subset_mode"],
        province_names=config.get("province_names"),
        city_names=config.get("city_names"),
        station_ids=None,
        variables=config["variables"],
        time_slice=None,
    )
    air_matrix = build_air_feature_matrix(air_subset)

    x_data_raw = air_matrix["x_data_raw"]
    t_data_raw = air_matrix["times"]
    station_meta_raw = air_matrix["station_meta"]
    feature_names_raw = build_normalized_feature_names(air_matrix["feature_names"])

    burn_in_steps = int(config.get("burn_in_steps", 0))
    sample_stride = int(config.get("sample_stride", 1))
    lag_steps = int(config["lag_steps"])
    n_delays = int(config["n_delays"])
    delay_interval = int(config["delay_interval"])

    x_data_fit = x_data_raw[burn_in_steps::sample_stride].copy()
    t_data_fit = t_data_raw[burn_in_steps::sample_stride].copy()

    H_list, delay_feature_names = lift_time_delay(
        x_data_fit,
        feature_names=feature_names_raw,
        n_delays=n_delays,
        delay_interval=delay_interval,
    )
    if isinstance(H_list, list):
        if len(H_list) != 1:
            raise ValueError(f"Expected one delayed trajectory, got {len(H_list)}")
        x_data_lift = np.asarray(H_list[0], dtype=float)
    else:
        x_data_lift = np.asarray(H_list, dtype=float)

    x_data_lift = x_data_lift - np.mean(x_data_lift, axis=0, keepdims=True)
    raw_names = list(delay_feature_names)

    transition_stats = compute_transition_covariances(
        [x_data_lift],
        library=None,
        weights="uniform",
        lag_steps=lag_steps,
    )
    C00 = transition_stats["C00"]
    C01 = transition_stats["C01"]
    C11 = transition_stats["C11"]
    X_pairs = transition_stats["X"]
    Y_pairs = transition_stats["Y"]

    koop_fit = fit_data_koopman_operator(
        [x_data_lift],
        library=None,
        weights="uniform",
        eps=config["eps"],
        ridge=config["ridge"],
        lag_steps=lag_steps,
    )

    K_raw = koop_fit["A"]
    K_bar = koop_fit["K_bar"]
    C00_inv_sqrt = koop_fit["C00_inv_sqrt"]
    C11_inv_sqrt = koop_fit["C11_inv_sqrt"]

    U_raw, S_raw, Vt_raw = np.linalg.svd(K_raw, full_matrices=False)
    metrics = analyze_kbar_metrics(K_bar, alpha=config["alpha"], eps=config["eps"])
    U_bar = metrics["U"]
    S_bar = metrics["S"]
    Vt_bar = metrics["Vt"]

    diff = get_positive_contributions(S_bar)
    EC = compute_entropy(diff)
    rho2 = np.clip(S_bar**2, config["eps"], 1.0 - config["eps"])
    g_i = (0.5 - config["alpha"] / 4.0) * np.log(rho2 / (1.0 - rho2)) + (
        config["alpha"] / 4.0
    ) * np.log(1.0 / (1.0 - rho2))
    rank_candidates = list(range(1, len(g_i) + 1))
    delta_g_by_r = {r: float(np.mean(g_i[:r]) - np.mean(g_i)) for r in rank_candidates}
    selected_r = int(config["rank"]) if config.get("rank") is not None else int(
        max(delta_g_by_r, key=delta_g_by_r.get)
    )

    macro = build_macro_from_kbar(
        U=U_bar,
        S=S_bar,
        Vt=Vt_bar,
        C00_inv_sqrt=C00_inv_sqrt,
        X=X_pairs,
        r=selected_r,
        feature_names=raw_names,
        Y=Y_pairs,
        C11_inv_sqrt=C11_inv_sqrt,
        center=False,
    )
    z_current = macro["z_current"]

    elapsed = time.perf_counter() - start
    return {
        "source_type": config["source_type"],
        "source_name": config["source_name"],
        "subset_mode": config["subset_mode"],
        "province_names": config.get("province_names"),
        "city_names": config.get("city_names"),
        "variables": config["variables"],
        "n_delays": n_delays,
        "delay_interval": delay_interval,
        "delay_days": n_delays,
        "lag_steps": lag_steps,
        "sample_stride": sample_stride,
        "station_count": int(air_matrix["n_stations"]),
        "raw_shape": list(map(int, x_data_raw.shape)),
        "lifted_shape": list(map(int, x_data_lift.shape)),
        "pair_count": int(X_pairs.shape[0]),
        "feature_dim": int(x_data_lift.shape[1]),
        "sigma_max_K_bar": float(metrics["sigma_max"]),
        "effective_rank": int(metrics["effective_rank"]),
        "G_alpha_K": float(metrics["G_alpha_K"]),
        "EC": float(EC),
        "selected_r": int(selected_r),
        "delta_g_selected_r": float(delta_g_by_r[selected_r]),
        "cond_C00": float(np.linalg.cond(C00)),
        "cond_C11": float(np.linalg.cond(C11)),
        "sigma_1": float(S_bar[0]) if len(S_bar) > 0 else 0.0,
        "sigma_2": float(S_bar[1]) if len(S_bar) > 1 else 0.0,
        "sigma_3": float(S_bar[2]) if len(S_bar) > 2 else 0.0,
        "top10_singular_values": [float(x) for x in S_bar[:10]],
        "macro_dim": int(z_current.shape[1]),
        "runtime_seconds": float(elapsed),
        "micro_feature_preview": raw_names[:8],
        "station_preview": station_meta_raw["station_id"].astype(str).tolist()[:8],
    }


def append_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--results-file", type=Path, required=True)
    parser.add_argument("--history-file", type=Path, required=True)
    args = parser.parse_args()

    configs = json.loads(args.config_file.read_text(encoding="utf-8"))
    result_records = []
    history_records = []

    for idx, config in enumerate(configs):
        run_name = config.get("run_name", f"air_sweep_{idx:04d}")
        try:
            metrics = compute_air_run(config)
            result_records.append(
                {
                    "run_name": run_name,
                    "status": "completed",
                    "params": config,
                    "metrics": metrics,
                }
            )
            history_records.append(
                {
                    "run_name": run_name,
                    "status": "completed",
                    "params": config,
                    "metrics": {
                        "G_alpha_K": metrics["G_alpha_K"],
                        "EC": metrics["EC"],
                        "delta_g_selected_r": metrics["delta_g_selected_r"],
                        "selected_r": metrics["selected_r"],
                        "runtime_seconds": metrics["runtime_seconds"],
                    },
                    "artifacts": {"results_file": str(args.results_file)},
                }
            )
            print(
                f"[completed] {run_name}: "
                f"CE={metrics['delta_g_selected_r']:.4f}, "
                f"G_alpha_K={metrics['G_alpha_K']:.4f}, "
                f"EC={metrics['EC']:.4f}, "
                f"stations={metrics['station_count']}, "
                f"runtime={metrics['runtime_seconds']:.2f}s"
            )
        except Exception as exc:
            history_records.append(
                {
                    "run_name": run_name,
                    "status": "failed",
                    "params": config,
                    "metrics": {},
                    "artifacts": {"results_file": str(args.results_file)},
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[failed] {run_name}: {type(exc).__name__}: {exc}")

    append_jsonl(args.results_file, result_records)
    append_jsonl(args.history_file, history_records)


if __name__ == "__main__":
    main()
