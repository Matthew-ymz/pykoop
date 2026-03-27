import argparse
import itertools
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = None
for candidate in [Path.cwd(), Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent.parent.parent]:
    if (candidate / "tools" / "tools.py").exists():
        REPO_ROOT = candidate
        break
if REPO_ROOT is None:
    raise RuntimeError("Could not locate repository root containing tools/tools.py")
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from tools.tools import (
    analyze_kbar_metrics,
    build_map_comparison_table,
    compute_entropy,
    extract_state_matrix_from_rulkov_data,
    fit_data_koopman_operator,
    generate_two_population_neuron_data,
    get_positive_contributions,
)


STATE_CONFIGS = {
    "CS": {
        "label": "Complete Synchronization (CS)",
        "n_a": 100,
        "n_b": 100,
        "alpha_a": 4.6,
        "alpha_b": 4.6,
        "sigma_a": 0.225,
        "sigma_b": 0.225,
        "mu": 0.001,
        "gamma": 0.005,
        "epsilon": 0.002,
        "T": 5000,
        "transients": 3000,
        "seed": 101,
        "x0_a": -1.0,
        "x0_b": -1.0,
        "y0_a": -3.5,
        "y0_b": -3.5,
    },
    "GS": {
        "label": "Generalized Synchronization (GS)",
        "n_a": 100,
        "n_b": 100,
        "alpha_a": 4.6,
        "alpha_b": 4.6,
        "sigma_a": 0.225,
        "sigma_b": 0.225,
        "mu": 0.001,
        "gamma": 0.06,
        "epsilon": 0.02,
        "T": 5000,
        "transients": 1000,
        "seed": 100,
        "x0_a": -1.0,
        "x0_b": -1.2,
        "y0_a": -3.5,
        "y0_b": -3.7,
    },
    "Q": {
        "label": "Chimera State (Q)",
        "n_a": 100,
        "n_b": 100,
        "alpha_a": 4.6,
        "alpha_b": 4.6,
        "sigma_a": 0.225,
        "sigma_b": 0.225,
        "mu": 0.001,
        "gamma": 0.005,
        "epsilon": 0.002,
        "T": 5000,
        "transients": 3000,
        "seed": 103,
        "x0_a": -1.0,
        "x0_b": None,
        "y0_a": -3.5,
        "y0_b": None,
    },
    "D": {
        "label": "Desynchronization (D)",
        "n_a": 100,
        "n_b": 100,
        "alpha_a": 4.6,
        "alpha_b": 4.6,
        "sigma_a": 0.225,
        "sigma_b": 0.225,
        "mu": 0.001,
        "gamma": 0.005,
        "epsilon": 0.002,
        "T": 5000,
        "transients": 3000,
        "seed": 155,
        "x0_a": None,
        "x0_b": None,
        "y0_a": None,
        "y0_b": None,
    },
}


def append_jsonl(path: Path, records):
    if not records:
        return
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_state_data(include_x=True, include_y=True, use_transient=True):
    prepared = {}
    for state_key, cfg in STATE_CONFIGS.items():
        data = generate_two_population_neuron_data(**{k: v for k, v in cfg.items() if k != "label"})
        raw_pack = extract_state_matrix_from_rulkov_data(
            data,
            include_x=include_x,
            include_y=include_y,
            use_transient=use_transient,
        )
        prepared[state_key] = {
            "config": cfg,
            "raw_data": data,
            "x_data_raw": raw_pack["x_data_raw"],
            "state_names": raw_pack["state_names"],
            "sync_metrics": raw_pack["sync_metrics"],
        }
        print(
            f"[prepared] {state_key}: "
            f"shape={raw_pack['x_data_raw'].shape}, sync_state={raw_pack['sync_metrics'].get('sync_state')}"
        )
    return prepared


def compute_state_metrics(x_data_raw, sync_metrics, params):
    sample_stride = int(params["sample_stride"])
    lag_steps = int(params["lag_steps"])
    ridge = float(params["ridge"])
    eps = float(params["eps"])
    alpha = float(params["alpha"])

    x_fit = x_data_raw[::sample_stride]
    koop = fit_data_koopman_operator(
        [x_fit],
        library=None,
        weights=None,
        eps=eps,
        ridge=ridge,
        lag_steps=lag_steps,
    )
    metrics = analyze_kbar_metrics(koop["K_bar"], alpha=alpha, eps=eps)
    singular_values = metrics["S"]
    diff = get_positive_contributions(singular_values)
    ec_value = compute_entropy(diff)
    g_i = metrics["channel_scores"]
    delta_g_by_r = {}
    for r in range(1, len(g_i) + 1):
        delta_g_by_r[r] = float(np.sum(g_i[:r]) - np.sum(g_i[r:]))
    selected_r = max(delta_g_by_r, key=delta_g_by_r.get)

    return {
        "sync_metrics": sync_metrics,
        "G_alpha_K": float(metrics["G_alpha_K"]),
        "EC": float(ec_value),
        "selected_r": int(selected_r),
        "delta_g_selected_r": float(delta_g_by_r[selected_r]),
        "effective_rank": int(metrics["effective_rank"]),
        "sigma_max_K_bar": float(np.max(singular_values)) if singular_values.size else float("nan"),
        "sigma_1": float(singular_values[0]) if singular_values.size > 0 else float("nan"),
        "sigma_2": float(singular_values[1]) if singular_values.size > 1 else float("nan"),
        "sigma_3": float(singular_values[2]) if singular_values.size > 2 else float("nan"),
        "pair_count": int(koop["X"].shape[0]),
        "feature_dim": int(koop["K_bar"].shape[0]),
        "cond_C00": float(np.linalg.cond(koop["C00"])),
        "cond_C11": float(np.linalg.cond(koop["C11"])),
        "singular_values": list(singular_values),
        "ec_spectrum": list(diff),
    }


def evaluate_combo(prepared_state_data, params):
    state_results = {}
    for state_key, prepared in prepared_state_data.items():
        state_results[state_key] = {
            "state": state_key,
            "label": prepared["config"]["label"],
            "sync_metrics": prepared["sync_metrics"],
            "metrics": compute_state_metrics(
                prepared["x_data_raw"],
                prepared["sync_metrics"],
                params,
            ),
            "singular_values": None,
            "ec_spectrum": None,
        }
        state_results[state_key]["singular_values"] = state_results[state_key]["metrics"]["singular_values"]
        state_results[state_key]["ec_spectrum"] = state_results[state_key]["metrics"]["ec_spectrum"]

    comparison = build_map_comparison_table(state_results)
    avg_ce = float(comparison["CE"].mean())
    avg_ec = float(comparison["EC"].mean())
    avg_g = float(comparison["G_alpha_K"].mean())
    std_ce = float(comparison["CE"].std(ddof=0))
    min_ce = float(comparison["CE"].min())
    score = avg_ce + 0.05 * avg_ec + 0.01 * min_ce

    return {
        "params": params,
        "comparison_table": comparison,
        "state_results": state_results,
        "aggregate": {
            "score": score,
            "avg_CE": avg_ce,
            "avg_EC": avg_ec,
            "avg_G_alpha_K": avg_g,
            "std_CE": std_ce,
            "min_CE": min_ce,
        },
    }


def save_best_plots(best_result, output_dir: Path):
    comparison = best_result["comparison_table"].copy()
    states = comparison["state"].tolist()
    ce_values = comparison["CE"].tolist()
    ec_values = comparison["EC"].tolist()

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(states, ce_values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    plt.ylabel("CE")
    plt.title("Best Map Sweep: CE by State")
    for bar, value in zip(bars, ce_values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "map_sweep_best_ce.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(states, ec_values, color=["#72B7B2", "#EECA3B", "#B279A2", "#FF9DA6"])
    plt.ylabel("EC")
    plt.title("Best Map Sweep: EC by State")
    for bar, value in zip(bars, ec_values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "map_sweep_best_ec.png", dpi=180)
    plt.close()


def build_configs():
    lag_steps = [1, 2, 3, 5, 10]
    sample_stride = [1, 2]
    ridge = [1e-10, 1e-8]
    alpha = [0.5, 1.0]
    eps = 1e-10
    configs = []
    for lag, stride, rg, a in itertools.product(lag_steps, sample_stride, ridge, alpha):
        configs.append(
            {
                "lag_steps": lag,
                "sample_stride": stride,
                "ridge": rg,
                "alpha": a,
                "eps": eps,
            }
        )
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = build_configs()
    (output_dir / "map_analysis_sweep_configs.json").write_text(
        json.dumps(configs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    prepared_state_data = prepare_state_data()

    results_file = output_dir / "map_analysis_sweep_results.jsonl"
    history_file = output_dir / "map_analysis_sweep_history.jsonl"
    if results_file.exists():
        results_file.unlink()
    if history_file.exists():
        history_file.unlink()

    combo_records = []
    history_records = []
    best_result = None
    best_score = -math.inf

    for idx, params in enumerate(configs, start=1):
        run_name = (
            f"map_lag{params['lag_steps']}_stride{params['sample_stride']}"
            f"_ridge{params['ridge']:.0e}_alpha{params['alpha']:.1f}"
        )
        try:
            result = evaluate_combo(prepared_state_data, params)
            aggregate = result["aggregate"]
            comparison = result["comparison_table"].copy()
            comparison["run_name"] = run_name
            comparison["lag_steps"] = params["lag_steps"]
            comparison["sample_stride"] = params["sample_stride"]
            comparison["ridge"] = params["ridge"]
            comparison["alpha"] = params["alpha"]
            comparison["score"] = aggregate["score"]

            combo_records.append(
                {
                    "run_name": run_name,
                    "params": params,
                    "aggregate": aggregate,
                    "comparison": comparison.to_dict(orient="records"),
                }
            )
            history_records.append(
                {
                    "run_name": run_name,
                    "status": "completed",
                    "params": params,
                    "aggregate": aggregate,
                }
            )
            append_jsonl(results_file, [combo_records[-1]])
            append_jsonl(history_file, [history_records[-1]])

            if aggregate["score"] > best_score:
                best_score = aggregate["score"]
                best_result = result
                best_result["run_name"] = run_name

            print(
                f"[{idx:02d}/{len(configs):02d}] {run_name}: "
                f"score={aggregate['score']:.4f}, avg_CE={aggregate['avg_CE']:.4f}, avg_EC={aggregate['avg_EC']:.4f}"
            )
        except Exception as exc:
            history_record = {
                "run_name": run_name,
                "status": "failed",
                "params": params,
                "notes": f"{type(exc).__name__}: {exc}",
            }
            history_records.append(history_record)
            append_jsonl(history_file, [history_record])
            print(f"[failed] {run_name}: {type(exc).__name__}: {exc}")

    rows = []
    flat_rows = []
    for record in combo_records:
        aggregate = record["aggregate"]
        row = {
            "run_name": record["run_name"],
            **record["params"],
            **aggregate,
        }
        rows.append(row)
        for state_row in record["comparison"]:
            flat_rows.append({**state_row})

    summary_df = pd.DataFrame(rows)
    state_df = pd.DataFrame(flat_rows)

    if summary_df.empty:
        raise RuntimeError("No successful sweep runs were completed.")

    summary_df = summary_df.sort_values(["score", "avg_CE", "avg_EC"], ascending=[False, False, False])

    summary_df.to_csv(output_dir / "map_analysis_sweep_summary.csv", index=False, encoding="utf-8-sig")
    summary_df.head(10).to_csv(output_dir / "map_analysis_sweep_top10.csv", index=False, encoding="utf-8-sig")
    state_df.to_csv(output_dir / "map_analysis_sweep_state_metrics.csv", index=False, encoding="utf-8-sig")

    if best_result is not None:
        best_result["comparison_table"].to_csv(
            output_dir / "map_analysis_sweep_best_comparison.csv",
            index=False,
            encoding="utf-8-sig",
        )
        save_best_plots(best_result, output_dir)

        lines = [
            "# Map Analysis Sweep Summary",
            "",
            f"总运行数: {len(summary_df)}",
            "",
            "## Best Run",
            "",
            f"- run_name: {best_result['run_name']}",
            f"- lag_steps: {best_result['params']['lag_steps']}",
            f"- sample_stride: {best_result['params']['sample_stride']}",
            f"- ridge: {best_result['params']['ridge']}",
            f"- alpha: {best_result['params']['alpha']}",
            f"- score: {best_result['aggregate']['score']:.6f}",
            f"- avg_CE: {best_result['aggregate']['avg_CE']:.6f}",
            f"- avg_EC: {best_result['aggregate']['avg_EC']:.6f}",
            f"- avg_G_alpha_K: {best_result['aggregate']['avg_G_alpha_K']:.6f}",
            "",
            "## Best Run Comparison Table",
            "",
            best_result["comparison_table"].to_csv(index=False),
            "",
            "## Top 10 Parameter Sets",
            "",
            summary_df.head(10).to_csv(index=False),
        ]
        (output_dir / "map_analysis_sweep_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
