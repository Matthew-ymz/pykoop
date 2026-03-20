# Rulkov Map 0320 Workflow

本目录实现了一个可直接运行的 `Rulkov map` 0320 版本流程，组织方式参考 `exp/kuramoto/twogroup_gram_0313.ipynb`，并保留了后续扩展到 sweep 的函数接口。

## 如何运行 notebook

1. 在仓库根目录启动 Jupyter。
2. 打开 [rulkov_map_0320.ipynb](/E:/code/pykoop/exp/map/rulkov_map_0320.ipynb)。
3. 顺序执行全部单元。
4. 执行到“结果保存”部分后，结果会写入 [results](/E:/code/pykoop/exp/map/results)。

如果只想做脚本级 smoke test，可以在仓库根目录运行：

```python
from exp.map.rulkov_map_tools import WorkflowConfig, run_rulkov_workflow, save_workflow_results

workflow = run_rulkov_workflow(WorkflowConfig())
artifacts = save_workflow_results(workflow)
print(artifacts)
```

## 参数说明

- `RulkovSimulationConfig`
  - `alpha` / `sigma`：每个神经元的 Rulkov 参数。
  - `mu`：慢变量时间尺度。
  - `beta`：快变量更新时加到慢变量中的常数输入。
  - `coupling_strength` / `beta_e` / `sigma_e`：扩散耦合设置。
  - `total_steps`：总仿真步数。
  - `burn_in`：丢弃的暂态长度。
  - `seed`：随机种子，默认固定，保证复现。
- `ObservableConfig`
  - `mode="identity_quadratic"`：默认使用 `identity + quadratic`。
  - 还预留了 `identity` / `quadratic` / `polynomial` / `fourier` / `custom` 入口。
- `WorkflowConfig`
  - `lag_steps`：快照配对滞后。
  - `rank`：宏观通道截断阶数。
  - `alpha`：CE 研究框架参数。
  - `include_closed_form_ce`：是否额外计算研究框架 CE 闭式接口。

## 输出文件说明

运行 notebook 后，默认会在 [results](/E:/code/pykoop/exp/map/results) 生成：

- `micro_state_series.csv`：微观原始时间序列。
- `observable_library.csv`：lift 后的观测库时间序列。
- `C00.csv` / `C01.csv` / `C11.csv`：协方差与跨时刻协方差矩阵。
- `K_matrix.csv`：原始拟合算子 `K`。
- `Kbar_matrix.csv`：白化矩阵 `Kbar`。
- `singular_values.csv`：奇异值谱数值。
- `left_singular_vectors.csv`：左奇异向量数值。
- `coarse_graining_matrix.csv`：粗粒化表达式对应系数矩阵。
- `macro_time_series.csv`：宏观时间序列。
- `macro_dynamics_matrix.csv`：宏观动力学方程矩阵。
- `summary_metrics.csv`：`EC`、`CE`、截断阶数等汇总指标。
- `equations.json`：粗粒化表达式和宏观动力学方程的文本表达。
- `metadata.json`：参数、特征名、fallback 信息、CE 分量等元数据。
- `*.png`：各类热力图、奇异值谱、宏观时间序列图。

所有热力图统一使用 `cmap="vlag"`。

## fallback 数据来源说明

需求指定的数据参考名是 `example_maps_Q`。当前仓库中未检测到这个精确名称，因此本实现默认 fallback 到 [exp/discrete_maps/examp_rulkov_pll.ipynb](/E:/code/pykoop/exp/discrete_maps/examp_rulkov_pll.ipynb) 里的 Rulkov 数据生成逻辑，并在 notebook 和 `metadata.json` 中显式记录该来源。

## 复用的现有工具函数

本实现没有重复造轮子，直接复用了仓库已有工具：

- `tools.tools.fit_data_koopman_operator`
- `tools.tools.compute_entropy`
- `tools.tools.get_positive_contributions`
- `exp.koopman_ce_cases.channel_scores_from_singular_values`
- `exp.koopman_ce_cases.koopman_ce_total_score_from_kbar`
- `exp.koopman_ce_cases.liu2025_log_gamma_gis`

其中：

- 白化 Koopman 主链路复用 `fit_data_koopman_operator`
- EC 指标复用 `get_positive_contributions + compute_entropy`
- CE 主流程与闭式接口分别复用 `channel_scores_from_singular_values`、`koopman_ce_total_score_from_kbar`、`liu2025_log_gamma_gis`
