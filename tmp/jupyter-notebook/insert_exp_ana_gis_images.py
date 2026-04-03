from pathlib import Path
p = next(x for x in Path(r'E:\code\pykoop\doc').iterdir() if x.suffix == '.md')
text = p.read_text(encoding='utf-8-sig')
old = '''### 4. 实验结果图
建议按以下顺序组织结果图：

1. 第一部分先放参数扫描奇异值谱箱型图和固定参数轨迹相图。箱型图用于说明解析矩阵在不同参数下可能出现明显谱分离；轨迹相图用于说明系统在状态空间中的收缩结构。
2. 第二部分放解析矩阵 `A` 热力图、微观层奇异值谱、粗粒化矩阵 `W` 热力图、宏观矩阵 `A_z` 热力图和宏观层奇异值谱。该部分主要展示“从矩阵本身出发也能得到正的宏观效率增益”。
3. 第三部分放含噪轨迹图、微观层与宏观层预测曲线图、宏微观曲线对比图。该部分图像的重点不在于展示矩阵元素，而在于说明即使有噪声，宏观层依然能够维持更高的单位维度效率，并给出更稳定的低维表示。
4. 第四部分放 `step2` 相图、奇异值谱与特征值谱对比图、SVD/EVD 的 `W` 对比图以及两种方法下的宏微观曲线对比图。该部分的重点是对比两种分解方法在宏观方向提取上的本质差异。
'''
new = '''### 4. 实验结果图
建议按以下顺序组织结果图：

1. 第一部分先放参数扫描奇异值谱箱型图和固定参数轨迹相图。箱型图用于说明解析矩阵在不同参数下可能出现明显谱分离；轨迹相图用于说明系统在状态空间中的收缩结构。

![第一部分：参数扫描奇异值谱箱型图](E:\code\pykoop\exp\analysitic_exp\figs\exp_ana_gis_part1_spectrum_boxplot.png)

![第一部分：固定参数轨迹相图](E:\code\pykoop\exp\analysitic_exp\figs\exp_ana_gis_part1_phase.png)

2. 第二部分放解析矩阵 `A` 热力图、微观层奇异值谱、粗粒化矩阵 `W` 热力图、宏观矩阵 `A_z` 热力图和宏观层奇异值谱。该部分主要展示“从矩阵本身出发也能得到正的宏观效率增益”。

![第二部分：解析矩阵 A 热力图](E:\code\pykoop\exp\analysitic_exp\figs\exp_ana_gis_part2_A_heatmap.png)

![第二部分：粗粒化矩阵 W 热力图](E:\code\pykoop\exp\analysitic_exp\figs\exp_ana_gis_part2_W_heatmap.png)

3. 第三部分放含噪轨迹图、微观层与宏观层预测曲线图、宏微观曲线对比图。该部分图像的重点不在于展示矩阵元素，而在于说明即使有噪声，宏观层依然能够维持更高的单位维度效率，并给出更稳定的低维表示。

![第三部分：干净轨迹与含噪轨迹对比](E:\code\pykoop\exp\analysitic_exp\figs\exp_ana_gis_part3_noisy_vs_clean.png)

![第三部分：微观层与宏观层预测曲线](E:\code\pykoop\exp\analysitic_exp\figs\exp_ana_gis_part3_prediction.png)

![第三部分：宏微观曲线对比](E:\code\pykoop\exp\analysitic_exp\figs\exp_ana_gis_part3_micro_macro_curve.png)

4. 第四部分放 `step2` 相图、奇异值谱与特征值谱对比图、SVD/EVD 的 `W` 对比图以及两种方法下的宏微观曲线对比图。该部分的重点是对比两种分解方法在宏观方向提取上的本质差异。

![第四部分：step2 系统轨迹相图](E:\code\pykoop\exp\analysitic_exp\figs\exp_ana_gis_part4_step2_phase.png)

![第四部分：奇异值谱与特征值谱对比](E:\code\pykoop\exp\analysitic_exp\figs\exp_ana_gis_part4_svd_vs_evd_spectrum.png)

![第四部分：SVD 与 EVD 粗粒化矩阵对比](E:\code\pykoop\exp\analysitic_exp\figs\exp_ana_gis_part4_W_compare.png)

![第四部分：SVD 与 EVD 宏微观曲线对比](E:\code\pykoop\exp\analysitic_exp\figs\exp_ana_gis_part4_curve_compare.png)
'''
if old not in text:
    raise SystemExit('target block not found')
text = text.replace(old, new, 1)
p.write_text(text, encoding='utf-8')
print('updated', p)
