import json
from pathlib import Path
from textwrap import dedent

out = Path(r'E:\code\pykoop\exp\analysitic_exp\exp_ana_gis.ipynb')

def md(text):
    text = dedent(text).strip('\n') + '\n'
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}

def code(text):
    text = dedent(text).strip('\n') + '\n'
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': text.splitlines(keepends=True)}

cells = []

cells.append(md('''
# exp_ana_gis

这个 notebook 以 `研究框架.md` 附录 C、附录 D 的 `GIS` 主线为准，重新组织 analytic step 系统、带噪数据流程以及 `step2` 系统下的 SVD / EVD 对比实验。
'''))

cells.append(md('''
## 0. 开头与公共准备

### 0.1 Notebook 目标与结构说明
本 notebook 的目标是把参考 notebook 中的 analytic step 例子改写为 `GIS` 主线流程，并把宏微观比较、`CE` 计算以及 `step2` 系统下的 SVD / EVD 对比统一到同一套分析框架下。

### 0.2 公共依赖、绘图风格与统一参数
这一块负责导入依赖、定位仓库根目录、设定绘图风格，并统一默认参数。后续默认使用：时间尺度 `tau=1`、可逆性参数 `alpha=1`、热力图配色 `vlag`，噪声默认取高斯噪声。
'''))

cells.append(code('''
import sys
from pathlib import Path
REPO_ROOT = None
for candidate in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
    if (candidate / 'tools' / 'tools.py').exists():
        REPO_ROOT = candidate
        break
if REPO_ROOT is None:
    raise RuntimeError('Could not locate repository root containing tools/tools.py')
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import seaborn as sns
from IPython.display import display

from tools.tools import (
    make_step_system_matrix, step_map, observable_step, observable_step2,
    simulate_discrete_system, add_gaussian_noise, prepare_time_pairs,
    fit_linear_gis_from_pairs, fit_linear_gis_from_matrix,
    compute_gis_metrics, compute_prediction_errors, compute_ce_from_micro_macro,
    select_macro_rank, build_w_from_svd, build_w_from_evd,
    apply_coarse_graining, summarize_pipeline_results,
)

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 160
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans']
HEATMAP_CMAP = 'vlag'
DEFAULT_CONFIG = {'tau': 1, 'delta': None, 'alpha': 1.0, 'noise_scale': 0.05, 'eps': 1e-10, 'ridge': 1e-10, 'manual_r': 1}
DEFAULT_CONFIG
'''))

cells.append(md('''
### 0.3 公共函数区（本实验特有的，不包括复用 `tools.py` 里面的函数）
这里定义本 notebook 特有的辅助函数，主要负责三件事：

1. 统一绘图风格，例如热力图与谱图；
2. 支撑第一部分的参数扫描；
3. 把附录 D 的流程打包成可重复调用的实验函数。

其中主流程包装函数 `run_gis_pipeline_from_observations` 的逻辑是：
\[
\mathbf{o}_t \rightarrow (A_o,\Sigma_o) \rightarrow W \rightarrow \mathbf{z}_t \rightarrow (A_z,\Sigma_z) \rightarrow CE.
\]
'''))

cells.append(code('''
def sparse_labels(labels, step=1):
    if labels is None:
        return False
    if step <= 1:
        return labels
    return [label if i % step == 0 else '' for i, label in enumerate(labels)]

def plot_matrix_heatmap(matrix, title, row_labels=None, col_labels=None, center=0.0, figsize=(6, 6), label_step=1, cmap=HEATMAP_CMAP):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(np.asarray(matrix), ax=ax, cmap=cmap, center=center, square=np.asarray(matrix).shape[0]==np.asarray(matrix).shape[1], xticklabels=sparse_labels(col_labels, label_step), yticklabels=sparse_labels(row_labels, label_step))
    ax.set_title(title)
    plt.tight_layout(); plt.show()

def standardize_for_plot(x):
    x = np.asarray(x, dtype=float)
    return (x - np.mean(x)) / (np.std(x) + 1e-12)

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

def plot_singular_values_boxplot(results, lam_values, ratios):
    rows = []
    for lam in lam_values:
        for i, ratio in enumerate(ratios):
            lbl = f"$10^{{{np.log10(ratio):.0f}}}$"
            rows += [
                {'Ratio': lbl, 'Singular Value': results[lam]['sv1'][i], 'Type': '$\\sigma_1$'},
                {'Ratio': lbl, 'Singular Value': results[lam]['sv2'][i], 'Type': '$\\sigma_2$'},
                {'Ratio': lbl, 'Singular Value': results[lam]['sv3'][i], 'Type': '$\\sigma_3$'},
            ]
    df = pd.DataFrame(rows)
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x='Ratio', y='Singular Value', hue='Type', palette='Set2')
    plt.yscale('log'); plt.xlabel('$\\lambda/\\mu$ ratio'); plt.ylabel('Singular value'); plt.tight_layout(); plt.show()

def project_matrix_with_w(A, Sigma, W):
    return W @ A @ W.T, W @ Sigma @ W.T

def run_gis_pipeline_from_observations(observations, tau=1, alpha=1.0, delta=None, eps=1e-10, ridge=1e-10, manual_r=1, horizons=(1,3,5), w_method='svd', evd_mode='eig_abs'):
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
    summary_dict, summary_row = summarize_pipeline_results(
        config={'tau': tau, 'delta': delta, 'alpha': alpha, 'w_method': w_method, 'manual_r': manual_r},
        micro_fit=micro_fit, macro_fit=macro_fit, micro_metrics=micro_metrics, macro_metrics=macro_metrics,
        prediction_results={'micro_errors': micro_errors, 'macro_errors': macro_errors}, ce_result=ce_result,
        extra={'rank_meta': rank_meta, 'w_result': w_result},
    )
    return {'micro_fit': micro_fit, 'micro_metrics': micro_metrics, 'micro_errors': micro_errors, 'rank_meta': rank_meta, 'w_result': w_result, 'W': W, 'macro_observations': macro_observations, 'macro_fit': macro_fit, 'macro_metrics': macro_metrics, 'macro_errors': macro_errors, 'ce_result': ce_result, 'summary_dict': summary_dict, 'summary_row': summary_row}
'''))

cells.append(md('''
## 第一部分：参数影响实验

### 1.1 第一类系统与本部分目标
本部分研究的系统为
\[
\begin{aligned}
x_{k+1} &= \lambda x_k,\\
y_{k+1} &= \mu y_k + (\lambda^2-\mu)x_k^2.
\end{aligned}
\]
在观测函数 \(g(x,y)=[x, y, x^2]^\top\) 下，其观测层解析矩阵可以直接写出。这里的目标不是走完整流程，而是先看不同参数下谱是否分离，以及固定参数下轨迹在相空间中的形态。
'''))

cells.append(code('''
scan_results, scan_lam_values, scan_ratios = generate_parameter_scan_results()
plot_singular_values_boxplot(scan_results, scan_lam_values, scan_ratios)
'''))

cells.append(md('''
### 1.2 参数扫描结果的理解
这里的箱型图并不直接给出 `CE`，而是帮助我们回答一个更基础的问题：当 \(\lambda\) 和 \(\mu\) 改变时，解析矩阵的奇异值谱是否出现明显分离。如果谱出现明显断层，那么后面的粗粒化通常更容易得到低维主方向。
'''))

cells.append(code('''
# 用一个简单表格展示某一组参数下的解析矩阵与奇异值，作为箱型图的补充参考。
example_lam, example_mu = 0.1, 0.9
example_A = make_step_system_matrix(example_lam, example_mu)
example_sv = np.sort(np.linalg.svd(example_A, compute_uv=False))[::-1]
display(pd.DataFrame({'singular_index': [1, 2, 3], 'singular_value': example_sv}))
'''))

cells.append(md('''
### 1.3 固定参数下的轨迹相图
下面固定一组代表性参数，从多个不同初值出发生成轨迹，并在相空间中画出渐变轨迹，用来直观看系统的收缩路径和组织方式。这一步主要帮助理解后面为什么某些参数下更容易出现低维宏观结构。
'''))

cells.append(code('''
phase_lam, phase_mu, phase_steps, num_samples = 0.1, 0.9, 50, 60
rng = np.random.default_rng(42)
initial_points = np.vstack([rng.uniform(-0.85, 0.85, num_samples), rng.uniform(-0.7, 0.85, num_samples)]).T
fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
custom_cmap = ListedColormap(plt.cm.YlGnBu(np.linspace(0.25, 0.95, 8)))
for x0, y0 in initial_points:
    traj = simulate_discrete_system(step_map, [x0, y0], steps=phase_steps, system_kwargs={'lam': phase_lam, 'mu': phase_mu}, dt=1.0)['trajectories'][0]
    points = traj.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    color_idx = np.arange(len(segments)); color_idx[np.where(color_idx > 20)] = 20
    lc = LineCollection(segments, cmap=custom_cmap); lc.set_array(color_idx); lc.set_linewidth(1.4); lc.set_capstyle('round'); ax.add_collection(lc)
ax.set_title(f'固定参数下的轨迹相图：lambda={phase_lam}, mu={phase_mu}')
ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_xlim([-0.95, 0.95]); ax.set_ylim([-0.85, 0.95]); ax.grid(False); plt.tight_layout(); plt.show()
'''))

cells.append(md('''
## 第二部分：已知解析矩阵 A / K_raw 的矩阵分析

### 2.1 本部分的目标与和主流程的关系
本部分不从数据出发，而是直接给定解析矩阵 \(A\)。我们把它视为观测层微观 `GIS` 的动力学矩阵，并在近零噪声条件下计算矩阵层面的指标。这样可以把“矩阵分析本身”和“数据拟合误差”分开看。
'''))

cells.append(code('''
direct_config = {'lam': 0.1, 'mu': 0.9, 'alpha': 1.0, 'eps': 1e-10, 'manual_r': 1}
direct_feature_names = ['$x$', '$y$', '$x^2$']
A_direct = make_step_system_matrix(direct_config['lam'], direct_config['mu'])
direct_fit = fit_linear_gis_from_matrix(A_direct, Sigma=None, sigma_eps=1e-10)
Sigma_direct = direct_fit['Sigma']
plot_matrix_heatmap(A_direct, '解析矩阵 A（观测层微观矩阵）', row_labels=direct_feature_names, col_labels=direct_feature_names, label_step=1)
'''))

cells.append(md('''
### 2.2 微观层 GIS 指标计算
在微观层，核心对象是
\[
\Gamma_\alpha^{\mathrm{GIS}}(A,\Sigma),\qquad
D(\Sigma)=\log \operatorname{pdet}(\Sigma^{-1}),\qquad
N(A,\Sigma)=\log \operatorname{pdet}(A^\top\Sigma^{-1}A),
\]
以及维度平均效率
\[
J_\alpha^{\mathrm{GIS}}(A,\Sigma;d)=\frac{1}{d}\log \Gamma_\alpha^{\mathrm{GIS}}(A,\Sigma).
\]
下面先在微观层上计算这些量，并画出 \(A^T\Sigma^{-1}A\) 的奇异值谱。
'''))

cells.append(code('''
direct_micro_metrics = compute_gis_metrics(A_direct, Sigma_direct, alpha=direct_config['alpha'], eps=direct_config['eps'])
display(pd.DataFrame({'metric': ['Gamma','log_Gamma','J_alpha','D','N'], 'value': [direct_micro_metrics['Gamma'], direct_micro_metrics['log_Gamma'], direct_micro_metrics['J_alpha'], direct_micro_metrics['D'], direct_micro_metrics['N']]}))
plt.figure(figsize=(7,4)); plt.plot(np.arange(1, len(direct_micro_metrics['sv_backward'])+1), direct_micro_metrics['sv_backward'], marker='o', color='tab:blue'); plt.title('微观层 $A^T\Sigma^{-1}A$ 的奇异值谱'); plt.xlabel('奇异值序号'); plt.ylabel('奇异值'); plt.tight_layout(); plt.show()
'''))

cells.append(md('''
### 2.3 宏观维度选择与粗粒化矩阵构造
这一步根据微观层的谱结构选择宏观维度 \(r\)，并构造粗粒化矩阵 \(W\)。在线性情形下，宏观变量写为
\[
\mathbf{z}_t = W \mathbf{o}_t.
\]
这里采用 SVD 路线来获得主方向，并将其组成粗粒化矩阵。
'''))

cells.append(code('''
direct_r, _ = select_macro_rank(direct_micro_metrics['sv_backward'], mode='manual', manual_r=direct_config['manual_r'], eps=direct_config['eps'])
direct_w_result = build_w_from_svd(A_direct, Sigma_direct, r=direct_r, alpha=direct_config['alpha'], eps=direct_config['eps'], mode='two_stage')
W_direct = direct_w_result['W']
print('选定的宏观维度 r =', direct_r)
plot_matrix_heatmap(np.abs(W_direct), '粗粒化矩阵 W 的绝对值热力图', row_labels=[f'$z_{i+1}$' for i in range(W_direct.shape[0])], col_labels=direct_feature_names, center=None, figsize=(5, 3.5), cmap='Blues')
'''))

cells.append(md('''
### 2.4 宏观层矩阵分析与指标
在矩阵层面，宏观矩阵与宏观协方差可写为
\[
A_z = W A W^\top,
\qquad
\Sigma_z = W \Sigma W^\top.
\]
下面先得到宏观层矩阵，再计算宏观层的 `GIS` 指标和奇异值谱。
'''))

cells.append(code('''
A_direct_macro, Sigma_direct_macro = project_matrix_with_w(A_direct, Sigma_direct, W_direct)
macro_names_direct = [f'$z_{i+1}$' for i in range(direct_r)]
plot_matrix_heatmap(A_direct_macro, '宏观矩阵 $A_z$', row_labels=macro_names_direct, col_labels=macro_names_direct, label_step=1)
direct_macro_metrics = compute_gis_metrics(A_direct_macro, Sigma_direct_macro, alpha=direct_config['alpha'], eps=direct_config['eps'])
display(pd.DataFrame({'metric': ['Gamma','log_Gamma','J_alpha','D','N'], 'value': [direct_macro_metrics['Gamma'], direct_macro_metrics['log_Gamma'], direct_macro_metrics['J_alpha'], direct_macro_metrics['D'], direct_macro_metrics['N']]}))
plt.figure(figsize=(7,4)); plt.plot(np.arange(1, len(direct_macro_metrics['sv_backward'])+1), direct_macro_metrics['sv_backward'], marker='o', color='tab:orange'); plt.title('宏观层 $A_z^T\Sigma_z^{-1}A_z$ 的奇异值谱'); plt.xlabel('奇异值序号'); plt.ylabel('奇异值'); plt.tight_layout(); plt.show()
'''))

cells.append(md('''
### 2.5 CE：矩阵层面的宏微比较
矩阵层面的 `CE` 定义为宏观层与微观层维度平均效率之差：
\[
CE = \Delta J_\alpha = J_{\alpha,z} - J_{\alpha,o}.
\]
下面直接输出宏微两层的核心指标，并计算矩阵层面的 `CE`。
'''))

cells.append(code('''
direct_ce = compute_ce_from_micro_macro(direct_micro_metrics, direct_macro_metrics)
display(pd.DataFrame({'layer': ['micro','macro'], 'J_alpha': [direct_micro_metrics['J_alpha'], direct_macro_metrics['J_alpha']], 'D': [direct_micro_metrics['D'], direct_macro_metrics['D']], 'N': [direct_micro_metrics['N'], direct_macro_metrics['N']], 'log_Gamma': [direct_micro_metrics['log_Gamma'], direct_macro_metrics['log_Gamma']]}))
print('矩阵层面的 CE =', direct_ce['CE'])
'''))

cells.append(md('''
## 第三部分：利用系统生成含噪数据，并走完整 GIS 主流程

### 3.1 本部分目标与主流程说明
本部分从数据开始，完整走一遍附录 D：数据生成、观测、微观拟合、宏观构造、宏观拟合、指标计算、预测比较和最终 `CE`。这也是后续做参数实验时最重要的主流程版本。
'''))

cells.append(code('''
analysis_config = {'experiment_name': 'exp_ana_gis_part3', 'lam': 0.1, 'mu': 0.9, 'initial_state': [5.0, 5.0], 'steps': 600, 'dt': 1.0, 'tau': 1, 'delta': None, 'alpha': 1.0, 'noise_scale': 0.05, 'noise_seed': 42, 'eps': 1e-10, 'ridge': 1e-10, 'manual_r': 1, 'horizons': (1, 3, 5)}
clean_sim = simulate_discrete_system(step_map, analysis_config['initial_state'], steps=analysis_config['steps'], system_kwargs={'lam': analysis_config['lam'], 'mu': analysis_config['mu']}, dt=analysis_config['dt'])
clean_xy = clean_sim['trajectories'][0]; time_grid = clean_sim['time_grid']
noisy_xy = add_gaussian_noise(clean_xy, noise_scale=analysis_config['noise_scale'], cov=None, random_state=analysis_config['noise_seed'])['noisy_data']
plt.figure(figsize=(10,4)); plt.plot(time_grid[:120], clean_xy[:120,0], label='clean $x$'); plt.plot(time_grid[:120], noisy_xy[:120,0], '--', alpha=0.8, label='noisy $x$'); plt.plot(time_grid[:120], clean_xy[:120,1], label='clean $y$'); plt.plot(time_grid[:120], noisy_xy[:120,1], '--', alpha=0.8, label='noisy $y$'); plt.title('第一类系统：干净轨迹与含噪轨迹'); plt.xlabel('Time'); plt.ylabel('State value'); plt.legend(ncol=2); plt.tight_layout(); plt.show()
'''))

cells.append(md('''
### 3.2 观测函数、时间尺度与样本配对
这里固定观测函数
\[
g(x,y) = [x, y, x^2]^\top,
\]
并根据时间尺度 \(\tau\) 构造观测层配对样本 \((o_t, o_{t+\tau})\)。这一层的观测变量就是后续微观 `GIS` 的输入。
'''))

cells.append(code('''
obs_part3 = observable_step(noisy_xy, mode='default')
X_now_3, X_next_3 = prepare_time_pairs(obs_part3, tau=analysis_config['tau'], burn_in=0, stride=1)
feature_names_3 = ['$x$', '$y$', '$x^2$']
print('观测层数据形状:', obs_part3.shape)
print('配对样本形状:', X_now_3.shape, X_next_3.shape)
'''))

cells.append(md('''
### 3.3 微观层 A / K_raw 与 Sigma 的拟合
在观测层上拟合线性 `GIS`：
\[
o_{t+\tau} \approx A_o o_t + \varepsilon_t^{(o)}.
\]
这里的 `A_o` 由数据回归得到，可视作经验 `K_raw`；而 `\Sigma_o` 则由一步残差协方差估计得到。
'''))

cells.append(code('''
fit3_micro = fit_linear_gis_from_pairs(X_now_3, X_next_3, fit_intercept=False, ridge=analysis_config['ridge'], regularization=analysis_config['eps'])
A3_micro = fit3_micro['A']; Sigma3_micro = fit3_micro['Sigma']
plot_matrix_heatmap(A3_micro, '第三部分：微观层 A', row_labels=feature_names_3, col_labels=feature_names_3, label_step=1)
plot_matrix_heatmap(Sigma3_micro, '第三部分：微观层 Sigma', row_labels=feature_names_3, col_labels=feature_names_3, center=None, label_step=1)
'''))

cells.append(md('''
### 3.4 微观层预测与误差
得到微观层 `A_o` 后，可做单步预测和多步滚动预测：
\[
\hat o_{t+\tau|t}=A_o o_t,
\qquad
\hat o_{t+k\tau|t}=A_o^k o_t.
\]
下面先展示误差，再画微观层真实曲线和单步预测曲线。
'''))

cells.append(code('''
errors3_micro = compute_prediction_errors(A3_micro, obs_part3, tau=analysis_config['tau'], horizons=analysis_config['horizons'])
display(pd.DataFrame({'horizon': list(errors3_micro.keys()), 'mean_error': [errors3_micro[h]['mean_error'] for h in errors3_micro.keys()]}))
pred1_micro, target1_micro = errors3_micro[1]['predictions'], errors3_micro[1]['targets']
plt.figure(figsize=(10,4))
for idx, name in enumerate(feature_names_3):
    plt.plot(np.arange(len(pred1_micro[:80])), target1_micro[:80, idx], label=f'true {name}')
    plt.plot(np.arange(len(pred1_micro[:80])), pred1_micro[:80, idx], '--', linewidth=1.6, label=f'pred {name}')
plt.title('第三部分：微观层单步预测曲线对比'); plt.xlabel('Pair index'); plt.ylabel('Value'); plt.legend(ncol=3); plt.tight_layout(); plt.show()
'''))

cells.append(md('''
### 3.5 微观层 GIS 指标
下面根据 \((A_o,\Sigma_o)\) 计算微观层的 `GIS` 指标：近似可逆性、对数近似可逆性、维度平均效率、确定性和非简并性，并画出微观层奇异值谱。
'''))

cells.append(code('''
metrics3_micro = compute_gis_metrics(A3_micro, Sigma3_micro, alpha=analysis_config['alpha'], eps=analysis_config['eps'])
display(pd.DataFrame({'metric': ['Gamma','log_Gamma','J_alpha','D','N'], 'value': [metrics3_micro['Gamma'], metrics3_micro['log_Gamma'], metrics3_micro['J_alpha'], metrics3_micro['D'], metrics3_micro['N']]}))
plt.figure(figsize=(7,4)); plt.plot(np.arange(1, len(metrics3_micro['sv_backward'])+1), metrics3_micro['sv_backward'], marker='o', color='tab:blue'); plt.title('第三部分：微观层奇异值谱'); plt.xlabel('奇异值序号'); plt.ylabel('奇异值'); plt.tight_layout(); plt.show()
'''))

cells.append(md('''
### 3.6 宏观维度选择与 W 的构造
根据微观层的谱结构选择宏观维度 \(r\)，再通过 SVD 路线构造粗粒化矩阵 \(W\)。这一阶段的结果决定了后面宏观变量的定义。
'''))

cells.append(code('''
r3, rank_meta3 = select_macro_rank(metrics3_micro['sv_backward'], mode='manual', manual_r=analysis_config['manual_r'], eps=analysis_config['eps'])
W3 = build_w_from_svd(A3_micro, Sigma3_micro, r=r3, alpha=analysis_config['alpha'], eps=analysis_config['eps'], mode='two_stage')['W']
macro_names_3 = [f'$z_{i+1}$' for i in range(r3)]
print('第三部分选定的宏观维度 r =', r3)
plot_matrix_heatmap(np.abs(W3), '第三部分：粗粒化矩阵 W 的绝对值热力图', row_labels=macro_names_3, col_labels=feature_names_3, center=None, figsize=(5, 3.5), cmap='Blues')
'''))

cells.append(md('''
### 3.7 宏观数据生成与宏观层拟合
宏观变量由
\[
z_t = W o_t
\]
给出。得到宏观数据后，再在宏观层拟合
\[
z_{t+\tau} \approx A_z z_t + \varepsilon_t^{(z)}.
\]
下面展示宏观数据形状、宏观层矩阵和宏观协方差矩阵。
'''))

cells.append(code('''
z3 = apply_coarse_graining(W3, obs_part3)
Z_now_3, Z_next_3 = prepare_time_pairs(z3, tau=analysis_config['tau'], burn_in=0, stride=1)
fit3_macro = fit_linear_gis_from_pairs(Z_now_3, Z_next_3, fit_intercept=False, ridge=analysis_config['ridge'], regularization=analysis_config['eps'])
A3_macro, Sigma3_macro = fit3_macro['A'], fit3_macro['Sigma']
print('第三部分宏观数据形状:', z3.shape)
plot_matrix_heatmap(A3_macro, '第三部分：宏观层 A', row_labels=macro_names_3, col_labels=macro_names_3, label_step=1)
plot_matrix_heatmap(Sigma3_macro, '第三部分：宏观层 Sigma', row_labels=macro_names_3, col_labels=macro_names_3, center=None, label_step=1)
'''))

cells.append(md('''
### 3.8 宏观层预测、指标与 CE
宏观层也要做预测和指标计算。最后通过
\[
CE = J_{\alpha,z} - J_{\alpha,o}
\]
比较宏观层和微观层。与此同时，我们还会把代表性的微观变量曲线和宏观变量曲线画在同一张图上，用不同线型区分可能重合的曲线。
'''))

cells.append(code('''
errors3_macro = compute_prediction_errors(A3_macro, z3, tau=analysis_config['tau'], horizons=analysis_config['horizons'])
display(pd.DataFrame({'horizon': list(errors3_macro.keys()), 'mean_error': [errors3_macro[h]['mean_error'] for h in errors3_macro.keys()]}))
pred1_macro, target1_macro = errors3_macro[1]['predictions'], errors3_macro[1]['targets']
plt.figure(figsize=(9,4))
for idx, name in enumerate(macro_names_3):
    plt.plot(np.arange(len(pred1_macro[:80])), target1_macro[:80, idx], label=f'true {name}')
    plt.plot(np.arange(len(pred1_macro[:80])), pred1_macro[:80, idx], '--', linewidth=2.0, label=f'pred {name}')
plt.title('第三部分：宏观层单步预测曲线对比'); plt.xlabel('Pair index'); plt.ylabel('Value'); plt.legend(); plt.tight_layout(); plt.show()
metrics3_macro = compute_gis_metrics(A3_macro, Sigma3_macro, alpha=analysis_config['alpha'], eps=analysis_config['eps'])
display(pd.DataFrame({'metric': ['Gamma','log_Gamma','J_alpha','D','N'], 'value': [metrics3_macro['Gamma'], metrics3_macro['log_Gamma'], metrics3_macro['J_alpha'], metrics3_macro['D'], metrics3_macro['N']]}))
plt.figure(figsize=(7,4)); plt.plot(np.arange(1, len(metrics3_macro['sv_backward'])+1), metrics3_macro['sv_backward'], marker='o', color='tab:orange'); plt.title('第三部分：宏观层奇异值谱'); plt.xlabel('奇异值序号'); plt.ylabel('奇异值'); plt.tight_layout(); plt.show()
ce3 = compute_ce_from_micro_macro(metrics3_micro, metrics3_macro)
print('第三部分 CE =', ce3['CE'])
plt.figure(figsize=(10,4.5))
for idx, name in enumerate(feature_names_3): plt.plot(np.arange(120), standardize_for_plot(obs_part3[:120, idx]), linewidth=1.4, label=f'micro: {name}')
for idx, name in enumerate(macro_names_3): plt.plot(np.arange(120), standardize_for_plot(z3[:120, idx]), '--', linewidth=2.2, label=f'macro: {name}')
plt.title('第三部分：宏微观曲线对比'); plt.xlabel('Time index'); plt.ylabel('Standardized value'); plt.legend(ncol=2); plt.tight_layout(); plt.show()
summary3_dict, summary3_row = summarize_pipeline_results(config={'experiment_name': analysis_config['experiment_name'], 'tau': analysis_config['tau'], 'delta': analysis_config['delta'], 'alpha': analysis_config['alpha'], 'noise_scale': analysis_config['noise_scale']}, micro_fit=fit3_micro, macro_fit=fit3_macro, micro_metrics=metrics3_micro, macro_metrics=metrics3_macro, prediction_results={'micro_errors': errors3_micro, 'macro_errors': errors3_macro}, ce_result=ce3, extra={'W': W3, 'rank_meta': rank_meta3})
display(pd.DataFrame([summary3_row]))
'''))

cells.append(md('''
## 第四部分：step2 系统下 SVD 与 EVD 的对比

### 4.1 step2 系统与本部分目标
`step2` 系统存在显著的非正交混合作用，因此非常适合比较奇异值分解与特征值分解在宏观方向提取上的差异。
'''))

cells.append(code('''
def step2_local(x, y, a=0.8, coupling=10.0):
    return a * x + coupling * (y ** 2), a * y

step2_config = {'experiment_name': 'exp_ana_gis_part4_step2', 'initial_state': [0.2, 0.45], 'steps': 220, 'dt': 1.0, 'tau': 1, 'delta': None, 'alpha': 1.0, 'noise_scale': 0.02, 'noise_seed': 7, 'eps': 1e-10, 'ridge': 1e-10, 'manual_r': 1, 'horizons': (1, 3, 5)}
step2_sim = simulate_discrete_system(step2_local, step2_config['initial_state'], steps=step2_config['steps'], system_kwargs={'a': 0.8, 'coupling': 10.0}, dt=step2_config['dt'])
step2_clean, step2_time = step2_sim['trajectories'][0], step2_sim['time_grid']
step2_noisy = add_gaussian_noise(step2_clean, noise_scale=step2_config['noise_scale'], cov=None, random_state=step2_config['noise_seed'])['noisy_data']
'''))

cells.append(md('''
### 4.2 step2 系统数据轨迹与相图
这里先画 `step2` 系统的轨迹相图。这样后面的 SVD / EVD 差异就不只是数值比较，还能和系统的几何轨迹联系起来。
'''))

cells.append(code('''
fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
rng = np.random.default_rng(42)
initial_points = np.vstack([rng.uniform(-0.3, 0.3, 60), rng.uniform(-0.5, 0.5, 60)]).T
custom_cmap = ListedColormap(plt.cm.YlGnBu(np.linspace(0.3, 0.95, 6)))
for x0, y0 in initial_points:
    traj = simulate_discrete_system(step2_local, [x0, y0], steps=120, system_kwargs={'a': 0.8, 'coupling': 10.0}, dt=1.0)['trajectories'][0]
    points = traj.reshape(-1, 1, 2); segments = np.concatenate([points[:-1], points[1:]], axis=1); color_idx = np.arange(len(segments)); color_idx[np.where(color_idx > 40)] = 40
    lc = LineCollection(segments, cmap=custom_cmap); lc.set_array(color_idx); lc.set_linewidth(1.6); lc.set_capstyle('round'); ax.add_collection(lc)
ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_title('step2 系统轨迹相图'); ax.set_xlim([-0.5, 3.5]); ax.set_ylim([-0.6, 0.6]); ax.grid(False); plt.tight_layout(); plt.show()
'''))

cells.append(md('''
### 4.3 标准 SVD 路线的完整流程
下面在 `step2` 数据上走标准 SVD 版本的附录 D 主流程，得到宏观表示、宏观层矩阵、指标和 `CE`。
'''))

cells.append(code('''
obs_step2 = observable_step2(step2_noisy, mode='default')
res_step2_svd = run_gis_pipeline_from_observations(obs_step2, tau=step2_config['tau'], alpha=step2_config['alpha'], delta=step2_config['delta'], eps=step2_config['eps'], ridge=step2_config['ridge'], manual_r=step2_config['manual_r'], horizons=step2_config['horizons'], w_method='svd')
print('step2 - SVD 路线 CE =', res_step2_svd['ce_result']['CE'])
'''))

cells.append(md('''
### 4.4 EVD 路线的完整流程
这里保持同样的数据、观测函数和时间尺度，只把宏观方向提取方法从 SVD 换成 EVD。对于复特征值，谱比较时统一取模长；若特征值为负，也统一按绝对值比较。
'''))

cells.append(code('''
res_step2_evd = run_gis_pipeline_from_observations(obs_step2, tau=step2_config['tau'], alpha=step2_config['alpha'], delta=step2_config['delta'], eps=step2_config['eps'], ridge=step2_config['ridge'], manual_r=step2_config['manual_r'], horizons=step2_config['horizons'], w_method='evd', evd_mode='eig_abs')
print('step2 - EVD 路线 CE =', res_step2_evd['ce_result']['CE'])
'''))

cells.append(md('''
### 4.5 奇异值谱与特征值谱的对比
下面把 `step2` 系统微观层矩阵的奇异值谱与特征值谱放在同一张图中比较。这样可以直接看到两种分解强调的结构是否一致。
'''))

cells.append(code('''
A_step2_micro = res_step2_svd['micro_fit']['A']; sv_step2 = np.linalg.svd(A_step2_micro, compute_uv=False); eig_abs_step2 = np.sort(np.abs(np.linalg.eigvals(A_step2_micro)))[::-1]
plt.figure(figsize=(7,4)); plt.plot(np.arange(1, len(sv_step2)+1), sv_step2, marker='o', linewidth=1.8, label='奇异值谱', color='tab:red'); plt.plot(np.arange(1, len(eig_abs_step2)+1), eig_abs_step2, marker='s', linewidth=1.4, linestyle='--', label='特征值谱（模长）', color='tab:blue'); plt.title('step2：奇异值谱与特征值谱对比'); plt.xlabel('模态序号'); plt.ylabel('数值大小'); plt.legend(); plt.tight_layout(); plt.show()
'''))

cells.append(md('''
### 4.6 两种情况下的粗粒化矩阵对比
粗粒化矩阵反映了宏观变量到底是由哪些观测变量组合得到的。下面分别画出 SVD 与 EVD 路线得到的粗粒化矩阵，用热力图对比两种方法提取出来的宏观方向。
'''))

cells.append(code('''
step2_feature_names = ['$x$', '$y$', '$y^2$']; step2_macro_names = [f'$z_{i+1}$' for i in range(res_step2_svd['W'].shape[0])]
plot_matrix_heatmap(np.abs(res_step2_svd['W']), 'step2：SVD 路线粗粒化矩阵 |W|', row_labels=step2_macro_names, col_labels=step2_feature_names, center=None, figsize=(5, 3.5), cmap='Blues')
plot_matrix_heatmap(np.abs(res_step2_evd['W']), 'step2：EVD 路线粗粒化矩阵 |W|', row_labels=step2_macro_names, col_labels=step2_feature_names, center=None, figsize=(5, 3.5), cmap='Blues')
'''))

cells.append(md('''
### 4.7 两种方法下的宏微观曲线对比
这里把微观曲线、SVD 宏观曲线和 EVD 宏观曲线画在同一张图上。微观层使用实线，SVD 宏观使用虚线，EVD 宏观使用点划线，以便在曲线重合时仍能区分。
'''))

cells.append(code('''
plt.figure(figsize=(10,4.5))
for idx, name in enumerate(step2_feature_names): plt.plot(np.arange(120), standardize_for_plot(obs_step2[:120, idx]), linewidth=1.4, label=f'micro: {name}')
for idx in range(res_step2_svd['macro_observations'].shape[1]): plt.plot(np.arange(120), standardize_for_plot(res_step2_svd['macro_observations'][:120, idx]), '--', linewidth=2.0, color=f'C{idx+3}', label=f'SVD macro: $z_{idx+1}$')
for idx in range(res_step2_evd['macro_observations'].shape[1]): plt.plot(np.arange(120), standardize_for_plot(res_step2_evd['macro_observations'][:120, idx]), '-.', linewidth=2.0, color=f'C{idx+6}', label=f'EVD macro: $z_{idx+1}$')
plt.title('step2：SVD / EVD 路线下的宏微观曲线对比'); plt.xlabel('Time index'); plt.ylabel('Standardized value'); plt.legend(ncol=2); plt.tight_layout(); plt.show()
'''))

cells.append(md('''
### 4.8 核心数值比较与结论
最后汇总 `step2` 系统下 SVD 与 EVD 两条路线的核心数值，包括 `CE`、宏微观的维度平均效率和对数近似可逆性，用于总结两种方法在这个系统上的差别。
'''))

cells.append(code('''
display(pd.DataFrame([{'method': 'SVD', 'micro_J_alpha': res_step2_svd['micro_metrics']['J_alpha'], 'macro_J_alpha': res_step2_svd['macro_metrics']['J_alpha'], 'micro_log_Gamma': res_step2_svd['micro_metrics']['log_Gamma'], 'macro_log_Gamma': res_step2_svd['macro_metrics']['log_Gamma'], 'CE': res_step2_svd['ce_result']['CE']}, {'method': 'EVD', 'micro_J_alpha': res_step2_evd['micro_metrics']['J_alpha'], 'macro_J_alpha': res_step2_evd['macro_metrics']['J_alpha'], 'micro_log_Gamma': res_step2_evd['micro_metrics']['log_Gamma'], 'macro_log_Gamma': res_step2_evd['macro_metrics']['log_Gamma'], 'CE': res_step2_evd['ce_result']['CE']}]))
'''))

cells.append(md('''
## 第五部分：结尾统一摘要

### 5.1 统一结论
下面用少量核心数值把参数影响实验、解析矩阵分析、含噪数据主流程和 `step2` 系统对比四部分串起来，形成统一总结。
'''))

cells.append(code('''
final_summary = {'part2_direct_matrix_ce': direct_ce['CE'], 'part3_noisy_pipeline_ce': ce3['CE'], 'part4_step2_svd_ce': res_step2_svd['ce_result']['CE'], 'part4_step2_evd_ce': res_step2_evd['ce_result']['CE']}
print('统一摘要：')
for key, value in final_summary.items(): print(f'{key}: {value:.6f}')
display(pd.DataFrame([{'part': '解析矩阵 A', 'CE': direct_ce['CE']}, {'part': '含噪数据完整流程', 'CE': ce3['CE']}, {'part': 'step2 - SVD', 'CE': res_step2_svd['ce_result']['CE']}, {'part': 'step2 - EVD', 'CE': res_step2_evd['ce_result']['CE']}]))
'''))

nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.12'}}, 'nbformat': 4, 'nbformat_minor': 5}
out = Path(r'E:\code\pykoop\exp\analysitic_exp\exp_ana_gis.ipynb')
out.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('rewritten', out)
