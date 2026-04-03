import json
from pathlib import Path

NOTEBOOK = Path(r"E:\code\pykoop\exp\analysitic_exp\exp_ana_gis.ipynb")


def set_markdown(nb, index, text):
    nb["cells"][index]["source"] = [(text.rstrip() + "\n")]


def replace_source(nb, index, replacements):
    text = "".join(nb["cells"][index]["source"])
    for old, new in replacements:
        text = text.replace(old, new)
    nb["cells"][index]["source"] = text.splitlines(keepends=True)


nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

set_markdown(nb, 0, """# exp_ana_gis

这个 notebook 以 `研究框架.md` 附录 C、附录 D 的 `GIS` 主线为准，重新组织 analytic step 系统、带噪数据流程以及 `step2` 系统下的 SVD / EVD 对比实验。""")
set_markdown(nb, 1, """## 0. 开头与公共准备

### 0.1 Notebook 目标与结构说明
本 notebook 的目标是把参考 notebook 中的 analytic step 例子改写为 `GIS` 主线流程，并把宏微观比较、`CE` 计算以及 `step2` 系统下的 SVD / EVD 对比统一到同一套分析框架下。

### 0.2 公共依赖、绘图风格与统一参数
下面导入依赖、定位仓库根目录、设定绘图风格并从 `tools.py` 中引入主流程函数。""")
set_markdown(nb, 3, """### 0.3 公共函数区（本实验特有的，不包括复用 `tools.py` 里面的函数）
这里定义热力图绘制、参数扫描、标准化显示、矩阵投影以及附录 D 主流程的实验包装函数。""")
set_markdown(nb, 5, """## 第一部分：参数影响实验

### 1.1 第一类系统与本部分目标
这一部分只做直观分析，不直接走完整流程。目标是观察不同参数下奇异值谱是否分离，以及固定参数下轨迹在相空间中的形态。

### 1.2 解析矩阵与参数扫描设定，绘制不同参数下奇异值谱的箱型图""")
set_markdown(nb, 7, "### 1.3 固定参数下的轨迹相图")
set_markdown(nb, 9, """## 第二部分：已知解析矩阵 A / K_raw 的矩阵分析

### 2.1 本部分的目标与和主流程的关系
本部分不从数据出发，而是直接把解析矩阵带入 `GIS` 指标体系，观察最干净条件下的矩阵分析链条。

### 2.2 给定解析矩阵与无噪声设定（矩阵 A 的热力图）""")
set_markdown(nb, 11, "### 2.3 微观层 GIS 指标计算（数值指标取值和奇异值谱绘制）")
set_markdown(nb, 13, "### 2.4 宏观维度选择与粗粒化矩阵构造（r，W 的热力图）")
set_markdown(nb, 15, "### 2.5 宏观层矩阵分析（得到宏观矩阵，并绘制矩阵热力图）")
set_markdown(nb, 17, "### 2.6 宏观层 GIS 指标计算（数值取值 + 奇异值谱）")
set_markdown(nb, 19, "### 2.7 CE，重点是矩阵层面的宏微比较")
set_markdown(nb, 21, """## 第三部分：利用系统生成含噪数据，并走完整 GIS 主流程

### 3.1 本部分目标与主流程说明
本部分从数据开始，完整走一遍附录 D：数据生成、观测、微观拟合、宏观构造、宏观拟合、指标计算、预测比较和最终 `CE`。

### 3.2 数据生成与噪声设定""")
set_markdown(nb, 23, """### 3.3 观测函数、时间尺度与样本配对
### 3.4 微观层 A / K_raw 与 Sigma 的拟合""")
set_markdown(nb, 25, """### 3.5 微观层预测与误差
### 3.6 微观层 GIS 指标""")
set_markdown(nb, 27, """### 3.7 宏观维度选择与 W 的构造
### 3.8 宏观数据生成
### 3.9 宏观层拟合""")
set_markdown(nb, 29, """### 3.10 宏观层预测、误差
### 3.11 宏观层 GIS 指标
### 3.12 CE 值 + 宏微观曲线对比
### 3.13 结果汇总""")
set_markdown(nb, 31, """## 第四部分：step2 系统下 SVD 与 EVD 的对比

### 4.1 step2 系统与本部分目标
`step2` 系统存在显著的非正交混合作用，适合比较奇异值分解与特征值分解在宏观方向提取上的差异。

### 4.2 step2 系统数据轨迹与相图""")
set_markdown(nb, 33, """### 4.3 标准 SVD 路线的完整流程
### 4.4 EVD 路线的完整流程
### 4.5 奇异值谱与特征值谱的对比
### 4.6 两种情况下的粗粒化矩阵对比
### 4.7 两种方法下的宏微观曲线对比
### 4.8 核心数值比较与结论""")
set_markdown(nb, 35, """## 第五部分：结尾统一摘要

### 5.1 统一结论
下面用少量核心数值把参数影响实验、解析矩阵分析、含噪数据主流程和 `step2` 系统对比四部分串起来，形成统一总结。""")

replace_source(nb, 8, [("ax.set_title(f'???????????lambda={phase_lam}, mu={phase_mu}')", "ax.set_title(f'固定参数下的轨迹相图：lambda={phase_lam}, mu={phase_mu}')")])
replace_source(nb, 10, [("???? A?????????", "解析矩阵 A（观测层微观矩阵）")])
replace_source(nb, 12, [("??? $A^T\\Sigma^{-1}A$ ?????", "微观层 $A^T\\Sigma^{-1}A$ 的奇异值谱"), ("?????", "奇异值序号"), ("???", "奇异值")])
replace_source(nb, 14, [("??????? r =", "选定的宏观维度 r ="), ("????? W ???????", "粗粒化矩阵 W 的绝对值热力图")])
replace_source(nb, 16, [("???? $A_z$", "宏观矩阵 $A_z$")])
replace_source(nb, 18, [("??? $A_z^T\\Sigma_z^{-1}A_z$ ?????", "宏观层 $A_z^T\\Sigma_z^{-1}A_z$ 的奇异值谱"), ("?????", "奇异值序号"), ("???", "奇异值")])
replace_source(nb, 20, [("????? CE =", "矩阵层面的 CE =")])
replace_source(nb, 22, [("???????????????", "第一类系统：干净轨迹与含噪轨迹")])
replace_source(nb, 24, [("???????? A", "第三部分：微观层 A"), ("???????? Sigma", "第三部分：微观层 Sigma")])
replace_source(nb, 26, [("????????????????", "第三部分：微观层单步预测曲线对比"), ("????????????", "第三部分：微观层奇异值谱"), ("?????", "奇异值序号"), ("???", "奇异值")])
replace_source(nb, 28, [("?????????? W ???????", "第三部分：粗粒化矩阵 W 的绝对值热力图"), ("???????? A", "第三部分：宏观层 A"), ("???????? Sigma", "第三部分：宏观层 Sigma")])
replace_source(nb, 30, [("????????????????", "第三部分：宏观层单步预测曲线对比"), ("????????????", "第三部分：宏观层奇异值谱"), ("?????", "奇异值序号"), ("???", "奇异值"), ("奇异值? CE =", "第三部分 CE ="), ("????????????", "第三部分：宏微观曲线对比")])
replace_source(nb, 32, [("step2 ??????", "step2 系统轨迹相图")])
replace_source(nb, 34, [("step2 - SVD ?? CE =", "step2 - SVD 路线 CE ="), ("step2 - EVD ?? CE =", "step2 - EVD 路线 CE ="), ("step2?SVD 奇异值谱??? |W|", "step2：SVD 路线粗粒化矩阵 |W|"), ("step2?EVD 奇异值谱??? |W|", "step2：EVD 路线粗粒化矩阵 |W|"), ("step2?SVD / EVD 奇异值谱奇异值谱???", "step2：SVD / EVD 路线下的宏微观曲线对比"), ("????????", "特征值谱（模长）"), ("????", "奇异值谱")])
replace_source(nb, 36, [("?????", "统一摘要："), ("统一摘要：???", "含噪数据完整流程"), ("???? A", "解析矩阵 A")])

NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print('fixed utf8 notebook')
