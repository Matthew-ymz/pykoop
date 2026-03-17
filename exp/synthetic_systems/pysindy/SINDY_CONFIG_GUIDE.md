# SINDy 配置参数完整指南

本文档详细说明了 PySINDy 训练程序中所有可配置的参数。

## 配置文件结构

```json
{
  "system_type": "系统类型",
  "system_params": { /* 系统参数 */ },
  "training": { /* 训练参数 */ },
  "sindy": { /* SINDy 算法参数 */ },
  "seed": 随机种子,
  "description": "配置描述"
}
```

---

## 1. 基本参数

### 1.1 system_type (必需)
**类型**: `string`  
**可选值**: `"lorenz"`, `"spring"`  
**说明**: 指定要使用的动力学系统类型

**示例**:
```json
"system_type": "lorenz"
```

### 1.2 description (可选)
**类型**: `string`  
**默认值**: `"N/A"`  
**说明**: 配置文件的描述信息

**示例**:
```json
"description": "Lorenz system with standard parameters"
```

### 1.3 seed (可选)
**类型**: `integer`  
**默认值**: `42`  
**说明**: 随机数生成器种子，用于保证结果可重复性

**示例**:
```json
"seed": 42
```

---

## 2. 系统参数 (system_params)

### 2.1 Lorenz 系统参数

```json
"system_params": {
  "sigma": 10.0,    // σ 参数（默认: 10.0）
  "rho": 28.0,      // ρ 参数（默认: 28.0）
  "beta": 2.667     // β 参数（默认: 8/3 ≈ 2.667）
}
```

**参数说明**:
- `sigma`: 普朗特数，控制对流的强度
- `rho`: 瑞利数，控制浮力驱动的对流
- `beta`: 几何参数

**标准混沌参数**: σ=10, ρ=28, β=8/3

### 2.2 Spring 系统参数

```json
"system_params": {
  "n_oscillators": 3,      // 振子数量
  "k_strong": 50.0,        // 组内弹簧常数
  "k_weak": 1.0,           // 组间弹簧常数
  "mass": 1.0,             // 振子质量
  "groups": {              // 振子分组
    "0": "a",
    "1": "a",
    "2": "b"
  }
}
```

**参数说明**:
- `n_oscillators`: 振子总数（建议: 2-6）
- `k_strong`: 同组内振子间的弹簧常数（强耦合）
- `k_weak`: 不同组振子间的弹簧常数（弱耦合）
- `mass`: 每个振子的质量
- `groups`: 振子分组，键为振子索引（0开始），值为组标签

**注意**: 
- 系统维度 = n_oscillators × 4（位置 x,y + 速度 vx,vy）
- 维度越高，计算越慢

---

## 3. 训练参数 (training)

```json
"training": {
  "t_span": [0, 10],      // 时间范围
  "n_points": 1000,       // 采样点数
  "noise_level": 0.0      // 噪声水平
}
```

### 3.1 t_span (必需)
**类型**: `array [start, end]`  
**说明**: 时间序列的起止时间

**建议值**:
- Lorenz: `[0, 50]` - 足够长以捕获混沌行为
- Spring: `[0, 10]` - 较短时间以加快计算

**示例**:
```json
"t_span": [0, 20]
```

### 3.2 n_points (必需)
**类型**: `integer`  
**说明**: 时间序列的采样点数

**建议值**:
- 最少: 500-1000 点
- 推荐: 1000-5000 点
- 更多点数提高精度但增加计算时间

**示例**:
```json
"n_points": 2000
```

### 3.3 noise_level (可选)
**类型**: `float`  
**默认值**: `0.0`  
**范围**: `[0.0, 1.0]`  
**说明**: 添加到数据中的高斯噪声水平（相对标准差）

**建议值**:
- 无噪声: `0.0`
- 低噪声: `0.01` (1%)
- 中等噪声: `0.05` (5%)
- 高噪声: `0.1` (10%)

**示例**:
```json
"noise_level": 0.02  // 2% 噪声
```

---

## 4. SINDy 算法参数 (sindy)

```json
"sindy": {
  "poly_order": 1,      // 多项式阶数
  "threshold": 0.1      // 稀疏化阈值
}
```

### 4.1 poly_order (必需)
**类型**: `integer`  
**范围**: `1-5`（通常）  
**说明**: 多项式特征库的最高阶数

**影响**:
- 阶数 1: 线性模型（最简单）
- 阶数 2: 包含二次项
- 阶数 3: 包含三次项（Lorenz 推荐）
- 阶数越高，特征数量指数增长

**特征数量估算**:
对于 d 维系统和阶数 n:
- 特征数 ≈ C(d+n, n) (组合数)
- 例如: d=3, n=3 → 20 个特征
- 例如: d=12, n=2 → 91 个特征

**建议值**:
- Lorenz 系统: `3`
- Spring 系统（3振子）: `1`
- Spring 系统（6振子）: `1` （更高阶会导致计算缓慢）

**示例**:
```json
"poly_order": 2
```

### 4.2 threshold (必需)
**类型**: `float`  
**范围**: `0.0-1.0`（通常 0.001-0.5）  
**说明**: STLSQ 优化器的稀疏化阈值

**作用**:
- 低阈值（0.001-0.01）: 保留更多项，模型复杂
- 中阈值（0.01-0.1）: 平衡稀疏性和精度
- 高阈值（0.1-0.5）: 更稀疏的模型，可能丢失重要项

**建议值**:
- 精确系统: `0.01`
- 有噪声数据: `0.05-0.1`
- 复杂系统（需要简化）: `0.1-0.2`

**示例**:
```json
"threshold": 0.05
```

---

## 5. 完整配置示例

### 5.1 Lorenz 系统（标准配置）

```json
{
  "system_type": "lorenz",
  "system_params": {
    "sigma": 10.0,
    "rho": 28.0,
    "beta": 2.6666666666666665
  },
  "training": {
    "t_span": [0, 50],
    "n_points": 5000,
    "noise_level": 0.0
  },
  "sindy": {
    "poly_order": 3,
    "threshold": 0.01
  },
  "seed": 42,
  "description": "Standard Lorenz attractor configuration"
}
```

### 5.2 Spring 系统（简化配置）

```json
{
  "system_type": "spring",
  "system_params": {
    "n_oscillators": 3,
    "k_strong": 50.0,
    "k_weak": 1.0,
    "groups": {
      "0": "a",
      "1": "a",
      "2": "b"
    },
    "mass": 1.0
  },
  "training": {
    "t_span": [0, 10],
    "n_points": 1000,
    "noise_level": 0.0
  },
  "sindy": {
    "poly_order": 1,
    "threshold": 0.1
  },
  "seed": 42,
  "description": "Simplified spring system for fast processing"
}
```

### 5.3 含噪声的 Lorenz 系统

```json
{
  "system_type": "lorenz",
  "system_params": {
    "sigma": 10.0,
    "rho": 28.0,
    "beta": 2.667
  },
  "training": {
    "t_span": [0, 50],
    "n_points": 5000,
    "noise_level": 0.05
  },
  "sindy": {
    "poly_order": 3,
    "threshold": 0.05
  },
  "seed": 42,
  "description": "Lorenz with 5% noise, higher threshold"
}
```

---

## 6. 参数调优指南

### 6.1 提高模型精度
1. **增加数据点**: `n_points` 提高到 5000-10000
2. **降低阈值**: `threshold` 降至 0.001-0.01
3. **适当提高阶数**: `poly_order` 尝试 +1

### 6.2 加快计算速度
1. **减少数据点**: `n_points` 降至 500-1000
2. **降低多项式阶数**: `poly_order` 设为 1 或 2
3. **提高阈值**: `threshold` 提高到 0.1-0.2
4. **缩短时间范围**: `t_span` 减小

### 6.3 处理噪声数据
1. **提高阈值**: `threshold` 设为 0.05-0.2
2. **增加数据点**: `n_points` 提高以平均噪声
3. **可能降低阶数**: 避免过拟合噪声

### 6.4 复杂系统优化（如 Spring）
1. **减少振子数**: `n_oscillators` 限制在 3-4
2. **使用低阶多项式**: `poly_order` = 1
3. **高稀疏化阈值**: `threshold` = 0.1-0.3
4. **短时间序列**: `t_span` = [0, 5] 或 [0, 10]

---

## 7. 常见问题

### Q1: 为什么 Spring 系统运行很慢或卡住？
**A**: Spring 系统维度高，计算量大。解决方法：
- 减少振子数量（3-4个）
- 使用 `poly_order=1`
- 提高 `threshold` 到 0.1-0.2

### Q2: 如何知道模型是否过拟合？
**A**: 检查以下指标：
- 训练集 R² 很高但测试集很低
- 非零系数数量过多（> 100）
- 建议：提高 `threshold`，降低 `poly_order`

### Q3: 模型精度不够怎么办？
**A**: 尝试：
- 增加 `n_points`
- 降低 `threshold`
- 适当提高 `poly_order`（但小心过拟合）

### Q4: 如何选择合适的参数？
**A**: 从简单开始，逐步调整：
1. 使用默认参数运行
2. 观察非零系数数量和训练时间
3. 根据结果调整：太慢→降低复杂度；精度低→提高数据量

---

## 8. 未来可扩展的参数

当前实现使用了固定的特征库（PolynomialLibrary）和优化器（STLSQ）。
PySINDy 还支持以下高级配置（可在未来版本中添加）:

### 8.1 特征库选项
- `FourierLibrary`: 傅里叶基
- `CustomLibrary`: 自定义函数
- `PDELibrary`: 偏微分方程库

### 8.2 优化器选项
- `SR3`: 松弛正则化
- `MIOSR`: 混合整数优化
- `ConstrainedSR3`: 带约束的优化

### 8.3 数值微分方法
- `FiniteDifference`: 有限差分（当前使用）
- `SmoothedFiniteDifference`: 平滑有限差分
- `SINDyDerivative`: SINDy 导数方法

---

## 9. 配置模板

可以基于以下模板创建自己的配置：

```json
{
  "system_type": "<lorenz|spring>",
  "system_params": {
    // Lorenz: sigma, rho, beta
    // Spring: n_oscillators, k_strong, k_weak, groups, mass
  },
  "training": {
    "t_span": [起始, 结束],
    "n_points": 采样点数,
    "noise_level": 噪声水平
  },
  "sindy": {
    "poly_order": 多项式阶数,
    "threshold": 稀疏化阈值
  },
  "seed": 42,
  "description": "配置描述"
}
```

---

**版本**: 1.0  
**最后更新**: 2025-12-02
