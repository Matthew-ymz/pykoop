import json
from pathlib import Path

p = Path(r"E:\code\pykoop\exp\analysitic_exp\exp_ana_gis.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))

def set_src(idx, text):
    nb['cells'][idx]['source'] = [line + "\n" for line in text.strip().splitlines()]

set_src(12, r'''
### 2.3 已知确定动力学微观层奇异值分析
这一部分不再使用 `GIS(A, \Sigma)` 的流程，因为此时我们并没有从数据残差中估计噪声协方差矩阵。这里直接把解析矩阵 `A / K_{raw}` 当作一个确定性的线性动力学矩阵，对它本身做奇异值分解：
\[
A = U \Sigma_A V^\top.
\]
下面先输出解析矩阵的奇异值，并把左奇异向量 `U` 作为后续粗粒化的候选方向。
''')

set_src(13, r'''
direct_U, direct_S, direct_Vt = np.linalg.svd(A_direct, full_matrices=False)
direct_gamma = compute_gamma_ce_metrics(direct_S, alpha=direct_config['alpha'], manual_r=direct_config['manual_r'], eps=direct_config['eps'])
display(pd.DataFrame({'singular_index': np.arange(1, len(direct_S) + 1), 'singular_value': direct_S}))
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.arange(1, len(direct_S) + 1), direct_S, marker='o', color='tab:blue', linewidth=1.8)
ax.set_title('解析矩阵 $A/K_{raw}$ 的奇异值谱')
ax.set_xlabel('奇异值序号')
ax.set_ylabel('奇异值')
plt.tight_layout(); plt.show()
''')

set_src(14, r'''
### 2.4 宏观维度选择与粗粒化矩阵构造
这里直接参考 `exp_analysis_0321` 的第二部分：先根据奇异值谱指定或选择宏观维度 `r`，然后用前 `r` 个左奇异向量构造粗粒化矩阵。若
\[
A = U \Sigma_A V^\top,
\]
则线性粗粒化矩阵取为
\[
W = U_r^\top,
\qquad
U_r = U[:, :r].
\]
在当前实验里默认取 `r=1`。
''')

set_src(15, r'''
direct_r = int(direct_config['manual_r'])
direct_left_basis = direct_U[:, :direct_r]
W_direct = direct_left_basis.T
print('选定的宏观维度 r =', direct_r)
plot_matrix_heatmap(direct_U, '解析矩阵 A 的左奇异向量矩阵 U', row_labels=direct_feature_names, col_labels=[f'$u_{i+1}$' for i in range(direct_U.shape[1])], label_step=1)
plot_matrix_heatmap(np.abs(W_direct), '由左奇异向量构造的粗粒化矩阵 |W|', row_labels=[f'$z_{i+1}$' for i in range(direct_r)], col_labels=direct_feature_names, center=None, figsize=(5, 3.5), cmap='Blues')
''')

set_src(16, r'''
### 2.5 宏观层矩阵分析
在确定 `W` 之后，直接把解析矩阵投影到宏观层：
\[
A_z = W A W^\top.
\]
由于这一部分是确定性矩阵分析，所以这里关注的是投影后的宏观矩阵本身，以及它的奇异值结构是否更集中。
''')

set_src(18, r'''
### 2.6 宏观层奇异值分析
宏观层同样直接对 `A_z` 做奇异值分解。这样第二部分的比较逻辑就是：先看原始解析矩阵 `A` 的奇异值谱，再看由左奇异向量粗粒化后得到的宏观矩阵 `A_z` 的奇异值谱，从而判断主要动力学方向是否被压缩并保留下来。
''')

set_src(19, r'''
direct_macro_U, direct_macro_S, direct_macro_Vt = np.linalg.svd(A_direct_macro, full_matrices=False)
display(pd.DataFrame({'singular_index': np.arange(1, len(direct_macro_S) + 1), 'singular_value': direct_macro_S}))
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.arange(1, len(direct_macro_S) + 1), direct_macro_S, marker='o', color='tab:orange', linewidth=1.8)
ax.set_title('宏观矩阵 $A_z$ 的奇异值谱')
ax.set_xlabel('奇异值序号')
ax.set_ylabel('奇异值')
plt.tight_layout(); plt.show()
''')

set_src(20, r'''
### 2.7 CE：基于解析矩阵奇异值谱的宏观增益
这里保留 `exp_analysis_0321` 第二部分中的做法，不使用 `GIS` 的 `CE` 定义，而是直接根据解析矩阵奇异值谱计算通道平均增益偏差。也就是说，先对奇异值做
\[
\sigma_i^\alpha
\]
变换，再比较截取前 `r` 个主通道后的平均值与整体平均值之差。该量可以视作确定性解析矩阵下的“谱驱动宏观增益”。
''')

set_src(21, r'''
direct_ce = {
    'CE': direct_gamma['delta_gamma_manual_r'],
    'selected_r': direct_gamma['selected_r'],
    'delta_gamma_selected_r': direct_gamma['delta_gamma_selected_r'],
    'Gamma_alpha_K': direct_gamma['Gamma_alpha_K'],
    'gamma_alpha_K': direct_gamma['gamma_alpha_K'],
}
display(pd.DataFrame({
    'quantity': ['Gamma_alpha_K', 'gamma_alpha_K', 'selected_r', 'delta_gamma_selected_r', 'manual_r', 'direct_CE'],
    'value': [direct_gamma['Gamma_alpha_K'], direct_gamma['gamma_alpha_K'], direct_gamma['selected_r'], direct_gamma['delta_gamma_selected_r'], direct_gamma['manual_r'], direct_gamma['delta_gamma_manual_r']]
}))
print('第二部分的谱驱动 CE =', direct_ce['CE'])
''')

set_src(30, r'''
### 3.6 微观层 GIS 指标
下面根据 `(A_o, \Sigma_o)` 计算微观层的 `GIS` 指标：近似可逆性、对数近似可逆性、维度平均效率、确定性和非简并性。谱图部分不再只画后向矩阵，而是同时画：
\[
\Sigma_o^{-1}
\qquad \text{和} \qquad
A_o^\top \Sigma_o^{-1} A_o.
\]
其中前者对应前向确定性结构，后者对应后向可分辨结构。这样更便于直接比较两类谱在同一层中的差异。
''')

set_src(31, r'''
metrics3_micro = compute_gis_metrics(A3_micro, Sigma3_micro, alpha=analysis_config['alpha'], eps=analysis_config['eps'])
display(pd.DataFrame({'metric': ['Gamma','log_Gamma','J_alpha','D','N'], 'value': [metrics3_micro['Gamma'], metrics3_micro['log_Gamma'], metrics3_micro['J_alpha'], metrics3_micro['D'], metrics3_micro['N']]}))
plot_dual_gis_spectrum(metrics3_micro['sv_forward'], metrics3_micro['sv_backward'], '第三部分：微观层前向谱与后向谱对比')
''')

set_src(40, r'''
### 3.11 宏观层 GIS 指标
宏观层同样要计算 `Gamma`、`log Gamma`、`J`、`D`、`N`。在谱图上也同时展示前向矩阵 `\Sigma_z^{-1}` 和后向矩阵 `A_z^\top \Sigma_z^{-1}A_z` 的谱，这样可以和微观层保持同一可视化标准。
''')

set_src(41, r'''
metrics3_macro = compute_gis_metrics(A3_macro, Sigma3_macro, alpha=analysis_config['alpha'], eps=analysis_config['eps'])
display(pd.DataFrame({'metric': ['Gamma','log_Gamma','J_alpha','D','N'], 'value': [metrics3_macro['Gamma'], metrics3_macro['log_Gamma'], metrics3_macro['J_alpha'], metrics3_macro['D'], metrics3_macro['N']]}))
plot_dual_gis_spectrum(metrics3_macro['sv_forward'], metrics3_macro['sv_backward'], '第三部分：宏观层前向谱与后向谱对比')
''')

p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('updated', p)
