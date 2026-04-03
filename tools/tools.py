import pysindy as ps
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import math
from typing import Union
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.integrate import odeint
import seaborn as sns
from numpy.polynomial.legendre import Legendre
from scipy.special import legendre # legendre(n)用于生成n阶勒让德多项式。
from scipy.integrate import fixed_quad 
from scipy.linalg import eig
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import warnings

def compute_koopman_matrices(XH, W, YH):
    """
    Parameters:
    XH+YH
    Returns:
    G: Gram矩阵<psi_j, psi_i>
    A: 第一个Galerkin矩阵 <K psi_j, psi_i>
    L: 第二个Galerkin矩阵 <K psi_j, K psi_i>
    """
    M = XH.shape[0] # 数据点数量
    N = XH.shape[1]  # 基函数数量
    
    # 初始化矩阵
    G = np.zeros((N, N))
    A = np.zeros((N, N))
    L = np.zeros((N, N))
    
    # 计算矩阵元素
    for j in range(M):
        G += np.outer(XH[j, :] * W[j], XH[j, :])
        A += np.outer(XH[j, :] * W[j], YH[j, :])
        L += np.outer(YH[j, :] * W[j], YH[j, :])
    
    return G, A, L

def compute_edmd_eigenpairs(G, A, L):
    """
    根据算法2步骤2，求解广义特征值问题 A g = λ G g
    返回特征值和对应的特征向量（系数向量）

    Parameters:
    G: Gram矩阵 Ψ_X^* W Ψ_X
    A: 矩阵 Ψ_X^* W Ψ_Y
    L: 矩阵 Ψ_Y^* W Ψ_Y

    Returns:
    eigenvalues: 特征值数组
    eigenvectors: 特征向量矩阵，每一列是一个特征向量 g
    """
    eigenvalues, eigenvectors = eig(A, G)
    return eigenvalues, eigenvectors

def compute_residual(lambda_val, g, G, A, L):
    """
    根据公式(4.6)计算给定特征对(λ, g)的残差

    Parameters:
    lambda_val: 特征值 λ
    g: 特征向量（系数向量）
    G, A, L: ResDMD矩阵

    Returns:
    res: 残差值
    """
    g = g.reshape(-1, 1) # 确保是列向量
    gH = g.T.conj() # 行向量，g的共轭转置

    # 计算分子: g^* (L - λ A^H - \bar{λ} A + |λ|^2 G) g
    numerator_matrix = L - lambda_val * A.T.conj() - np.conj(lambda_val) * A + (np.abs(lambda_val)**2) * G
    numerator = (gH @ numerator_matrix @ g).item()
    numerator = np.real(numerator) # 确保是实数

    # 计算分母: g^* G g
    denominator = (gH @ G @ g).item()
    denominator = np.real(denominator)

    # 避免除零，计算残差
    if denominator <= 0:
        return np.inf
    else:
        res_squared = numerator / denominator
        # 由于数值误差，结果可能为负，取绝对值再开方
        return np.sqrt(np.abs(res_squared))
    
def koop_pseudo_spec(G, A, L, z_pts, **kwargs):
    """
    计算K的伪谱
    
    Parameters:
    G: Gram矩阵<psi_j, psi_i>
    A: 第一个Galerkin矩阵 <K psi_j, psi_i>
    L: 第二个Galerkin矩阵 <K psi_j, K psi_i>
    z_pts: 复数点向量，用于计算伪谱
    
    Optional:
    reg_param: G的正则化参数
    
    Returns:
    RES: z_pts处的残差
    """
    # 解析可选参数
    reg_param = kwargs.get('reg_param', 1e-14)
    
    # 确保矩阵是Hermitian的
    G = (G + G.T.conj()) / 2
    L = (L + L.T.conj()) / 2
    
    # 正则化G并计算SQ矩阵
    G_reg = G + np.linalg.norm(G) * reg_param * np.eye(G.shape[0])
    w, V = eigh(G_reg)
    
    # 避免除零和负值
    w = np.maximum(w, 1e-15)
    w_sqrt = np.sqrt(1.0 / np.abs(w))
    SQ = V @ np.diag(w_sqrt) @ V.T.conj()
    
    z_pts = z_pts.flatten()
    LL = len(z_pts)
    RES = np.zeros(LL, dtype=float)
    
    if LL > 0:
        warnings.filterwarnings('ignore', category=UserWarning)
        
        for jj in range(LL):
            z = z_pts[jj]
            try:
                # 构造该z对应的矩阵
                M_z = (L - z * A.T.conj() - np.conj(z) * A + (np.abs(z)**2) * G)
                M_transformed = SQ @ M_z @ SQ
                
                # 确保矩阵是Hermitian的以用于eigvalsh
                M_transformed = (M_transformed + M_transformed.T.conj()) / 2
                
                # 使用稠密计算找到最小特征值（更稳定）
                eigvals = np.linalg.eigvalsh(M_transformed)
                min_eigval = np.min(np.real(eigvals))
                
                # 避免由于数值误差导致的负值
                RES[jj] = np.sqrt(max(0, min_eigval))
                
            except Exception as e:
                print(f"Warning: Error at z={z}: {e}")
                RES[jj] = np.nan
    
    warnings.resetwarnings()
    
    return RES

def plot_pseudospectra(G, A, L, step=0.05, residual_threshold=0.01, padding_ratio=0.2, fixed_range=None):
    """
    绘制伪谱图，并根据残差阈值区分可靠和虚假特征值

    Parameters:
    G, A, L: Koopman矩阵
    x_range, y_range: 绘图范围
    step: 网格步长
    residual_threshold: 算法2中的残差阈值ε
    padding_ratio: 在特征值范围基础上添加的边距比例
    fixed_range: 如果提供，则使用固定的绘图范围 (x_min, x_max, y_min, y_max)
    """
    # 1. 计算EDMD特征对 (算法2步骤2)
    eigenvalues, eigenvectors = compute_edmd_eigenpairs(G, A, L)
    # 特征向量在矩阵eigenvectors的列中

    # 2. 为每个特征对计算残差 (算法2步骤3的逻辑)
    residuals = []
    reliable_indices = []
    spurious_indices = []

    for i in range(len(eigenvalues)):
        lambda_i = eigenvalues[i]
        g_i = eigenvectors[:, i]
        res_i = compute_residual(lambda_i, g_i, G, A, L)
        residuals.append(res_i)

        if res_i <= residual_threshold:
            reliable_indices.append(i)
        else:
            spurious_indices.append(i)

    print(f"总特征值数量: {len(eigenvalues)}")
    print(f"res阈值: {residual_threshold}")
    print(f"可靠特征值数量 (res <= {residual_threshold}): {len(reliable_indices),reliable_indices}")
    print(f"虚假特征值数量 (res > {residual_threshold}): {len(spurious_indices)}")

    # 3. 确定绘图范围
    if fixed_range is not None:
        # 使用固定的绘图范围
        x_range = (fixed_range[0], fixed_range[1])
        y_range = (fixed_range[2], fixed_range[3])
    else:
        # 自适应范围：基于所有特征值的分布
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)
        
        # 计算特征值的范围
        real_min, real_max = np.min(real_parts), np.max(real_parts)
        imag_min, imag_max = np.min(imag_parts), np.max(imag_parts)
        
        # 使用相同的范围，保持纵横比一致
        overall_min = min(real_min, imag_min)
        overall_max = max(real_max, imag_max)
        overall_range = overall_max - overall_min
        
        # 如果范围太小（如所有特征值都集中在一点），设置最小范围
        if overall_range < 0.1:
            overall_range = 1.0
            center = (overall_min + overall_max) / 2
            overall_min = center - 0.5
            overall_max = center + 0.5
        
        # 添加边距
        padding = overall_range * padding_ratio
        x_range = (overall_min - padding, overall_max + padding)
        y_range = (overall_min - padding, overall_max + padding)
    
    # 3. 计算伪谱网格（算法3/原有伪谱计算逻辑）
    x_pts = np.arange(x_range[0], x_range[1] + step, step)
    y_pts = np.arange(y_range[0], y_range[1] + step, step)
    X_grid, Y_grid = np.meshgrid(x_pts, y_pts)
    z_pts = X_grid + 1j * Y_grid
    z_flat = z_pts.flatten()

    RES = koop_pseudo_spec(G, A, L, z_flat)
    RES = RES.reshape(z_pts.shape)
    RES = np.nan_to_num(RES, nan=np.max(RES[~np.isnan(RES)]))
    # 计算完RES并reshape后，添加这行
    print(f"残差RES的范围：{np.min(RES):.6f} ~ {np.max(RES):.6f}")

    # 4. 绘图
    plt.figure(figsize=(8, 8))

    # 伪谱等高线
    #levels = [0.001, 0.01, 0.1, 0.3]
    levels = [0.001, 0.01, 0.1]
    contour = plt.contour(X_grid, Y_grid, np.real(RES), levels=levels,
                         colors='black', linewidths=2)
    plt.clabel(contour, inline=True, fontsize=11, fmt='%.3f')

    # 绘制特征值
    # 可靠特征值 (残差小) - 蓝色十字
    reliable_eigs = eigenvalues[reliable_indices]
    plt.plot(np.real(reliable_eigs), np.imag(reliable_eigs), 'x',
             markersize=8, color='blue', markeredgewidth=2,
             label=f'Reliable eigenvalues (res $\leq$ {residual_threshold})')

    # 虚假特征值 (残差大) - 洋红点
    spurious_eigs = eigenvalues[spurious_indices]
    plt.plot(np.real(spurious_eigs), np.imag(spurious_eigs), '.',
             markersize=10, color='magenta',
             label='Spurious eigenvalues')

    # 格式化
    plt.gca().set_aspect('equal')
    plt.xlabel('Real', fontsize=14)
    plt.ylabel('Imaginary', fontsize=14)
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.grid(True, alpha=0.3)
    #plt.legend(fontsize=12, loc='upper right', frameon=True, fancybox=True, shadow=True)
    plt.legend(fontsize=12)
    plt.title(f'Reliable num={len(reliable_indices)}, Residual Threshold={residual_threshold}', fontsize=14)
    plt.tight_layout()
    plt.show()
    return eigenvalues, residuals, reliable_indices

def plot_main(kp, X_embed):
    M = len(X_embed)-1
    W = np.ones(M) / M
    X = X_embed[:-1]
    Y = X_embed[1:]
    XH = kp.transform(X)
    YH = kp.transform(Y)
    G, A, L = compute_koopman_matrices(XH, W, YH)
    eigenvalues, residuals, reliable_indices = plot_pseudospectra(G, A, L, residual_threshold=0.01,fixed_range=(0.8,1.20,-0.15,0.15))
    return eigenvalues, residuals, reliable_indices

def matrix_l1_norm_manual(matrix):
    """
    手动计算矩阵的L1范数（不使用numpy）
    """
    
    rows = len(matrix)
    cols = len(matrix[0])
    
    # 计算每列的绝对值之和
    column_sums = []
    for j in range(cols):
        col_sum = 0
        for i in range(rows):
            col_sum += abs(matrix[i][j])
        column_sums.append(col_sum)
    
    # 返回最大的列和
    return max(column_sums)

def matrix_l0_norm_corrected(matrix, threshold=1e-10):
    """
    计算矩阵的L0范数（各列非零元素数量的最大值）
    
    参数:
    matrix: numpy数组或可以转换为numpy数组的矩阵
    threshold: 阈值，绝对值小于此值的元素视为零
    
    返回:
    l0_norm: 矩阵的L0范数（整数）
    column_norms: 各列的L0范数
    """
    matrix = np.array(matrix, dtype=float)
    
    # 应用阈值：将接近零的元素视为零
    matrix_thresholded = np.where(np.abs(matrix) < threshold, 0, matrix)
    
    # 计算每列的非零元素数量
    column_norms = []
    for col in range(matrix_thresholded.shape[1]):
        non_zero_count = np.count_nonzero(matrix_thresholded[:, col])
        column_norms.append(non_zero_count)
    
    # 矩阵的L0范数是各列L0范数的最大值
    l0_norm = max(column_norms)
    
    return l0_norm

def get_positive_contributions(sing_values):  
    ave_sig = []
    for i in range(len(sing_values)):
        ave_sig.append(np.mean(np.log(sing_values[0:i+1])))

    output = []
    for id in range(len(ave_sig)-1):
        diff = ave_sig[id] - ave_sig[id+1]
        output.append(diff)
    return output

def compute_entropy(increments):
    if not increments:
        return 0.0
    
    total = sum(increments)
    # If total is 0, there's no variation => 0.0 entropy
    if total == 0:
        return 0.0
    
    # Normalize to probabilities
    probabilities = [x / total for x in increments]

    # Compute Shannon entropy (base 2)
    entropy = 0.0
    for p in probabilities:
        # Only compute for p > 0 to avoid math domain errors
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy

def print_equations(coefficient_matrix, feature_names, target_names, threshold=1e-5):
    """
    将系数矩阵转换为数学方程并打印。
    
    Args:
        coefficient_matrix (np.array): 系数矩阵 (行数=特征数, 列数=目标变量数)
        feature_names (list): 纵轴标签列表 (对应矩阵的行, 如 ['x0', 'sin(x0)', ...])
        target_names (list): 横轴标签列表 (对应矩阵的列, 如 ['y0', 'y1', 'y2'])
        threshold (float): 忽略系数绝对值小于此阈值的项 (默认 1e-5)
    """
    
    rows, cols = coefficient_matrix.shape
    
    # 检查维度匹配
    if len(feature_names) != rows:
        print(f"错误: feature_names 长度 ({len(feature_names)}) 与矩阵行数 ({rows}) 不一致")
        return
    if len(target_names) != cols:
        print(f"错误: target_names 长度 ({len(target_names)}) 与矩阵列数 ({cols}) 不一致")
        return

    # 遍历每一列 (即每一个 y0, y1, y2...)
    for col_idx in range(cols):
        lhs = target_names[col_idx] # 等号左边
        rhs_parts = []
        
        # 遍历该列的每一行，寻找非零系数
        for row_idx in range(rows):
            coef = coefficient_matrix[row_idx, col_idx]
            
            # 如果系数绝对值大于阈值，则认为该项存在
            if abs(coef) > threshold:
                term_name = feature_names[row_idx]
                
                # 格式化系数，保留4位小数
                formatted_coef = f"{coef:.4f}"
                
                # 拼接项：例如 "0.5234 * sin(x0)"
                rhs_parts.append(f"{formatted_coef} * {term_name}")
        
        # 组装整个方程
        if not rhs_parts:
            equation = f"{lhs} = 0"
        else:
            # 用 " + " 连接所有项
            equation_str = " + ".join(rhs_parts)
            # 简单的美化：处理 "+ -" 为 "- "
            equation_str = equation_str.replace("+ -", "- ")
            equation = f"{lhs} = {equation_str}"
            
        print(equation)
        print("-" * 30) # 分隔线

def draw_fft(data: pd.DataFrame, dt: Union[str, float] = 'index', 
             remove_dc: bool = True, max_f: float = 0, normalize: bool = True) -> pd.DataFrame:
    """
    对 DataFrame 数据进行 FFT 分析并绘图
    
    Parameters:
    data: pd.DataFrame - 输入数据
    dt: Union[str, float] - 采样间隔设置
        - 'index': 使用 data.index 计算间隔（支持 datetime 类型）
        - 0: 使用 cycle=1
        - 非零实数: 使用 cycle=dt 作为采样周期
    remove_dc: bool - 是否去除直流分量（默认 True）
    max_f: float - 最大显示频率（默认 0 表示不限制）
    normalize: bool - 是否对数据进行归一化（默认 True）
        归一化方法：对每列减去均值并除以标准差
    
    Returns:
    pd.DataFrame - FFT 结果，index 为频率，columns 与输入相同
    """
    # 步骤0: 归一化数据（如果需要）
    if normalize:
        data_processed = (data - data.mean()) / data.std()
    else:
        data_processed = data.copy()
    
    # 步骤1: 去除直流分量
    if remove_dc:
        data_processed = data_processed - data_processed.mean()
    
    # 获取数据点数
    N = len(data_processed)
    
    # 步骤2: 对各列进行 FFT 变换
    fft_result = np.fft.fft(data_processed.values, axis=0)
    # 归一化并只保留正频率部分
    fft_data = np.abs(fft_result / N)[:N//2]
    
    # 步骤3: 计算频率轴
    if dt == 'index':
        # 使用 index 计算采样周期
        cycle = data.index[1] - data.index[0]
        # 如果是 datetime 类型，转换为秒
        if isinstance(cycle, pd.Timedelta):
            cycle = cycle.total_seconds()
    elif dt == 0:
        cycle = 1
    else:
        # dt 是非零实数，作为采样周期
        cycle = dt
    
    # 使用 fftfreq 生成频率轴，只取正频率部分
    freq = np.fft.fftfreq(N, cycle)[:N//2]
    
    # 步骤4: 构建结果 DataFrame
    result_df = pd.DataFrame(fft_data, index=freq, columns=data.columns)
    
    # 如果指定了最大频率，进行过滤
    if max_f > 0:
        result_df = result_df[result_df.index <= max_f]
    
    # 步骤5: 绘制频谱图
    plt.figure(figsize=(10, 6))
    for col in result_df.columns:
        plt.plot(result_df.index, result_df[col], label=col)
    
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.title('FFT Spectrum', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return result_df

def fit_sindy_sr3_robust(X, lib, feature_names,
                         penalty='l0',       
                         dt=1,
                         discrete_time=True,
                         thresholds=None,
                         nu=1.0,
                         max_iter=3000,
                         tol=1e-6,
                         test_size=0.2,
                         metric='aic'):
    """
    使用 SR3 优化器拟合 SINDy 模型，支持鲁棒的参数扫描。
    支持单个时间序列或多段时间序列列表（直接对每个序列单独拟合，不进行拼接）。
    
    参数:
    X : np.ndarray 或 list of np.ndarray
        - 如果是 np.ndarray，shape (n_samples, n_dim)：单个时间序列
        - 如果是 list，每个元素是 np.ndarray (n_samples_i, n_dim)：多段时间序列
          会对每个序列单独拟合，不进行拼接
    lib : pysindy.feature_library
        特征库对象
    feature_names : list of str
        特征名列表
    penalty : str
        正则化类型：'l0', 'l1', 'l2'
    dt : float
        时间步长
    discrete_time : bool
        是否使用离散时间
    thresholds : list or np.ndarray, optional
        参数扫描范围
    nu : float
        SR3 的松弛参数
    max_iter : int
        最大迭代次数
    tol : float
        收敛阈值
    test_size : float
        测试集比例
    metric : str
        评估指标：'aic', 'bic', 'mse'
        
    返回:
    best_model : pysindy.SINDy
        最佳模型（基于所有序列的平均评分）
    history : list of dict
        参数扫描的历史记录，每个字典包含所有序列的结果
        格式: {
            'thr': threshold,
            'lam': lambda,
            'avg_score': 平均评分,
            'avg_mse': 平均MSE,
            'avg_k': 平均复杂度,
            'sequence_results': [{...}, {...}, ...],  # 每个序列的结果
            'model': model
        }
    """
    # === 第一步：检查并转换输入 ===
    if isinstance(X, list):
        # 如果是列表，验证一致性
        if len(X) == 0:
            raise ValueError("输入列表不能为空")
        
        # 验证所有序列的维度一致
        n_dim = X[0].shape[1] if X[0].ndim == 2 else 1
        for i, seq in enumerate(X):
            if seq.ndim != 2:
                raise ValueError(f"序列 {i} 应为2维数组 (n_samples, n_dim)，实际为 {seq.ndim} 维")
            if seq.shape[1] != n_dim:
                raise ValueError(f"序列 {i} 的维度 ({seq.shape[1]}) 与第一个序列 ({n_dim}) 不匹配")
        
        X_list = X
        n_sequences = len(X)
        
        print(f"✓ 检测到 {n_sequences} 个时间序列（单独拟合，不拼接）")
        for i, seq in enumerate(X_list):
            print(f"  序列 {i+1}: {seq.shape[0]} 个样本, {seq.shape[1]} 个维度")
        
    elif isinstance(X, np.ndarray):
        # 如果是单个数组
        if X.ndim != 2:
            raise ValueError(f"数组应为2维 (n_samples, n_dim)，实际为 {X.ndim} 维")
        X_list = [X]
        n_sequences = 1
        print(f"✓ 检测到单个时间序列，长度: {X.shape[0]}")
        
    else:
        raise TypeError(f"输入类型应为 np.ndarray 或 list，实际为 {type(X)}")

    # === 第二步：针对不同范数调整默认扫描范围 ===
    if thresholds is None:
        if penalty == 'l0':
            thresholds = np.logspace(-5, -1, 20)  # L0: 物理意义的系数截断值
        elif penalty == 'l1':
            thresholds = np.logspace(-5, 1, 10)  # L1: 物理意义的软阈值截断值
        else:
            thresholds = np.logspace(-2, 4, 20)   # L2: 直接是正则化权重，范围通常较大

    best_score = float('inf')
    best_model = None
    history = []

    print(f"\n开始使用 {penalty.upper()} 范数扫描 {len(thresholds)} 个参数...")
    print(f"对 {n_sequences} 个序列分别拟合...\n")

    for thr in tqdm(thresholds, desc=f"参数扫描 ({penalty})"):
        try:
            # --- 核心修改：根据范数计算 lambda ---
            if penalty == 'l0':
                # L0: 硬阈值换算
                lam = ps.SR3.calculate_l0_weight(thr, nu)
                reg_type = "l0"
            elif penalty == 'l1':
                # L1: 软阈值换算 (lambda = nu * threshold)
                lam = thr * nu 
                reg_type = "l1"
            elif penalty == 'l2':
                # L2: 没有截断概念，thr 直接作为 lambda
                lam = thr
                reg_type = "l2"
            else:
                raise ValueError("penalty must be 'l0', 'l1', or 'l2'")

            # 配置 SR3 优化器
            opt = ps.SR3(
                reg_weight_lam=lam,
                regularizer=reg_type,
                relax_coeff_nu=nu,
                normalize_columns=True, 
                unbias=True,  # 对 L1 非常重要！
                max_iter=max_iter,
                tol=tol,
            )

            model = ps.SINDy(feature_library=lib, optimizer=opt, discrete_time=discrete_time)           

            # 拆分训练测试集
            split_idx = -2
            X_train = X_list[:split_idx]
            X_test = X_list[split_idx:]           
            
            # 拟合当前序列
            model.fit(X_train, t=dt, feature_names=feature_names)
            
            # 评估当前序列
            X_test_pred = model.predict(X_test)
            
            X_test_pred = np.vstack(X_test_pred)
            X_test = np.vstack(X_test)

            if not discrete_time:
                mse = -model.score(X_test, t=dt) 
            else:
                # 确保维度匹配
                if X_test_pred.shape[0] > X_test.shape[0] - 1:
                    mse = mean_squared_error(X_test[1:], X_test_pred[:len(X_test)-1])
                else:
                    mse = mean_squared_error(X_test[1:len(X_test_pred)+1], X_test_pred)

            coef = model.coefficients()
            k_params = np.sum(np.abs(coef) > 1e-5)

            if k_params == 0:
                score = float('inf')
            else:
                n_samples = len(X_test)
                if mse <= 0: 
                    mse = 1e-10
                log_likelihood = n_samples * np.log(mse)
                
                if metric == 'aic':
                    score = log_likelihood + 2 * k_params
                elif metric == 'bic':
                    score = log_likelihood + k_params * np.log(n_samples)
                else: 
                    score = mse
            
            history.append({
                'thr': thr, 
                'lam': lam, 
                'score': score,
                'mse': mse,
                'k': k_params,
                'model': model
            })

            if score < best_score:
                best_score = score
                best_model = model
                    
        except Exception as e:
            print(f"⚠️  参数 thr={thr:.3e} 处理失败: {str(e)}")
            continue

    # === 结果展示 ===
    if len(history) == 0:
        raise RuntimeError("所有参数配置都拟合失败，请检查输入数据和参数")
    
    history.sort(key=lambda x: x['score'])
    
    if best_model:
        top = history[0]
        print(f"\n" + "="*70)
        print(f"最佳模型 ({metric.upper()}) | Penalty: {penalty.upper()}")
        print(f"="*70)
        print(f"参数 (Threshold): {top['thr']:.3e}")
        print(f"参数 (Lambda):    {top['lam']:.3e}")
        print(f"MSE:         {top['mse']:.4e}")
        print(f"k:   {top['k']:.1f}")
        print(f"评分 ({metric}):  {top['score']:.4f}")
        print(f"\n发现的方程:")
        print("="*70)
        best_model.print()
    
    return best_model, history

def lift_time_delay(X, feature_names=None, n_delays=1, delay_interval=1):
    """
    将时间序列 X 提升到时间延迟坐标系，并自动生成对应的新变量名。
    支持单个时间序列或多段时间序列列表。
    
    参数:
    X : np.ndarray 或 list of np.ndarray
        - 如果是 np.ndarray，shape (n_samples, n_dim)：单个时间序列
        - 如果是 list，每个元素是 np.ndarray (n_samples_i, n_dim)：多段时间序列
          所有序列必须有相同的 n_dim（维度）
    feature_names : list of str, optional
        原始变量的名字，例如 ['x', 'y', 'z']。
        如果不提供，默认生成 ['x0', 'x1', ...]。
    n_delays : int
        向后看的步数 (delay count)。
    delay_interval : int
        延迟的间隔步长 (tau)。
        
    返回:
    H : np.ndarray
        提升后的 Hankel 矩阵（多个序列拼接）。
    new_feature_names : list of str
        对应的变量名列表。
        格式示例：['x', 'y', 'x_d1', 'y_d1', 'x_d2', 'y_d2']
    """
    
    # 1. 检查输入类型并统一转换
    if isinstance(X, list):
        # 如果是列表，检查所有序列
        if len(X) == 0:
            raise ValueError("输入列表不能为空")
        
        # 验证所有序列的维度一致
        n_dim = X[0].shape[1] if X[0].ndim == 2 else 1
        for i, seq in enumerate(X):
            if seq.ndim != 2:
                raise ValueError(f"序列 {i} 应为2维数组 (n_samples, n_dim)，实际为 {seq.ndim} 维")
            if seq.shape[1] != n_dim:
                raise ValueError(f"序列 {i} 的维度 ({seq.shape[1]}) 与第一个序列 ({n_dim}) 不匹配")
        
        # 转换为 list of arrays
        X_list = X
    
    elif isinstance(X, np.ndarray):
        X_list = [X]
        n_dim = X.shape[1]
    
    else:
        raise TypeError(f"输入类型应为 np.ndarray 或 list，实际为 {type(X)}")

    n_dim = X_list[0].shape[1]
    
    # 2. 处理默认变量名
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_dim)]
    
    if len(feature_names) != n_dim:
        raise ValueError(f"feature_names 长度 ({len(feature_names)}) 与数据维度 ({n_dim}) 不匹配")

    # 3. 生成新的特征名（所有序列共用）
    new_names = []
    
    # 第0层：当前时刻 t
    new_names.extend(feature_names)
    
    # 后续层：延迟时刻 t - k*tau
    for k in range(1, n_delays + 1):
        suffix = f"_d{k}" if delay_interval == 1 else f"_d{k*delay_interval}"
        current_names = [f"{name}{suffix}" for name in feature_names]
        new_names.extend(current_names)

    # 4. 处理每个序列并拼接
    all_H = []
    
    for seq_idx, X_single in enumerate(X_list):
        n_samples, _ = X_single.shape
        
        # 计算有效样本数
        window_size = n_delays * delay_interval
        n_valid_samples = n_samples - window_size
        
        if n_valid_samples <= 0:
            print(f"⚠️ 警告：序列 {seq_idx} 太短 (长度 {n_samples})，无法进行 {n_delays} 次延迟"
                  f"(需要最少 {window_size + 1} 个样本)，跳过此序列")
            continue
        
        # 构建该序列的数据矩阵
        shifted_features = []
        
        # 第0层：当前时刻 t (无后缀或标记为 t)
        shifted_features.append(X_single[window_size:])
        
        # 后续层：延迟时刻 t - k*tau
        for k in range(1, n_delays + 1):
            # 计算偏移
            offset = window_size - k * delay_interval
            
            # 数据切片
            if offset == 0:
                shifted_data = X_single[:-window_size]
            else:
                shifted_data = X_single[offset:offset + n_valid_samples]
            
            shifted_features.append(shifted_data)
        
        # 拼接该序列的矩阵
        H_single = np.column_stack(shifted_features)
        all_H.append(H_single)
    
    # 5. 检查是否有有效的序列被处理
    if len(all_H) == 0:
        raise ValueError("所有输入序列都太短，无法进行时间延迟提升")
    
    # 6. 拼接所有序列（按行拼接）
    H = np.vstack(all_H)
    
    print(f"✓ 成功处理 {len(all_H)} 个序列，提升后数据形状: {H.shape}")
    
    return all_H, new_names

'''



def fit_sindy_sr3_robust(
    x_data: np.ndarray,
    library,
    feature_names: List[str],
    penalty: str = 'l1',
    discrete_time: bool = True,
    max_iter: int = 100,
    thresholds: np.ndarray = None,
    metric: str = 'bic',
    tol: float = 1e-4,
    nu: float = 1
) -> Tuple:
    """
    使用 SR3 + STLSQ 拟合 SINDy 模型
    
    Args:
        x_data: 输入数据，形状为 (n_samples, n_features)
        library: pysindy 库对象
        feature_names: 特征名称
        penalty: 正则化类型 ('l1' 或 'l2')
        discrete_time: 是否为离散时间系统
        max_iter: 最大迭代次数
        thresholds: 阈值数组
        metric: 模型选择指标
        tol: 容差
        nu: SR3 参数
        
    Returns:
        model: 拟合的 SINDy 模型
        results: 结果字典
    """
    import pysindy as ps
    
    if thresholds is None:
        thresholds = np.array([0.01, 0.1, 1.0])
    
    # 使用 STLSQ 优化器
    optimizer = ps.STLSQ(threshold=thresholds[0], alpha=0.9)
    
    # 拟合模型
    model = ps.SINDy(feature_library=library, optimizer=optimizer, discrete_time=discrete_time)
    model.fit(x_data, feature_names=feature_names)
    
    results = {'score': model.score(x_data)}
    return model, results
'''

def plot_station(df, coarse_grain_coff, delay=0):
    if isinstance(coarse_grain_coff, np.ndarray):
        coff_df = pd.DataFrame(coarse_grain_coff)
    else:
        coff_df = coarse_grain_coff.reset_index(drop=True)

    # 2. 遍历每一列进行绘图
    for col_idx in coff_df.columns:
        # 将当前列的数据合并到原始 df 中，命名为一个临时列名，例如 'value_to_plot'
        df_plot = df.copy()
        df_plot['value_to_plot'] = coff_df[col_idx].values
        
        # 绘制图形
        fig = px.scatter(
            df_plot,
            x="lon",
            y="lat",
            text="station_id",
            color="value_to_plot",  # 核心修改：颜色映射到当前列的数值
            hover_data=["station_name", "city"],
            color_continuous_scale='Viridis', # 设置颜色条，可选 'Plasma', 'Inferno', 'Turbo' 等
            title=f"y{col_idx}_d{delay}"
        )
        
        # 调整标注位置和点的大小
        fig.update_traces(
            textposition='top center',
            marker=dict(size=10, opacity=0.8) # 稍微调大点的大小以看清颜色
        )
        
        # 优化布局：保持经纬度比例，以免地图变形
        fig.update_layout(
            yaxis_scaleanchor="x", 
            yaxis_scaleratio=1,
            height=600,
            width=800
        )
        
        fig.show()

def lift_double_osc(x):
    """
    lift x from 4 dim to 7 dim
    """
    return [
        x[0],x[1],x[2],x[3],x[0]**2,x[0]*x[1],x[1]**2
    ]


def lift_double_osc_dot(y):
    """
    lift x from 4 dim to 7 dim
    """
    w1 = 1.
    w2 = 1.618
    return [
        - y[1] * w1,
        y[0] * w1,
        - y[3] * w2 + y[4],
        y[2] * w2,
        - 2 * y[5] * w1,
        (y[4] - y[6]) * w1,
        2 * w1 * y[5]
    ]

def split_and_group_matrices(U, new_names, n_splits):
    """
    根据变量名筛选行，并将结果矩阵进行 n 等分。
    
    参数:
    U : np.ndarray
        原始大矩阵 (rows x columns)
    new_names : list of str
        对应 U 每一行的名字
    n_splits : int
        每个分类后的矩阵要被切分成几份
        
    返回:
    final_list : list of np.ndarray
        包含所有切分后小矩阵的列表
    """
    
    # 1. 找到对应的行索引
    # 使用列表推导式找到包含特定字符串的索引
    idx_pm25 = [i for i, name in enumerate(new_names) if "pm25" in name]
    idx_o3   = [i for i, name in enumerate(new_names) if "o3" in name]
    
    # 检查是否找到了数据
    if not idx_pm25:
        print("警告: 没有找到包含 'pm25' 的行")
    if not idx_o3:
        print("警告: 没有找到包含 'o3' 的行")
        
    # 2. 根据索引从 U 中提取子矩阵
    # U[list_of_indices, :] 会提取对应的行
    matrix_pm25 = U[idx_pm25, :]
    matrix_o3   = U[idx_o3, :]
    
    # 3. 检查能否被 n 整除 (这是一个常见的坑)
    # 如果不能整除，np.split 会报错，或者我们需要用 array_split
    if matrix_pm25.shape[0] % n_splits != 0:
        print(f"提示: PM2.5 矩阵行数 ({matrix_pm25.shape[0]}) 不能被 {n_splits} 整除，将进行近似均分。")
    if matrix_o3.shape[0] % n_splits != 0:
        print(f"提示: O3 矩阵行数 ({matrix_o3.shape[0]}) 不能被 {n_splits} 整除，将进行近似均分。")

    # 4. 执行拆分
    # np.array_split 比 np.split 更鲁棒，它允许不均匀拆分
    splits_pm25 = np.array_split(matrix_pm25, n_splits, axis=0)
    splits_o3   = np.array_split(matrix_o3,   n_splits, axis=0)
    
    return splits_pm25, splits_o3

def plot_macro_serie(origin_data, macro_data, n_delays, delay_interval, times, selected_indices, stations):
    fig, ax1 = plt.subplots(figsize=(15, 6))

    # -------------------------------------------------
    # 左侧 Y 轴 (ax1)：绘制各个站点的曲线
    # -------------------------------------------------
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Concentration (Stations)', fontsize=12) # 左轴标签
    ax1.grid(True, linestyle='--', alpha=0.5)
    times_final = times[n_delays*delay_interval:]
    # 循环绘制选定站点的曲线 (画在 ax1 上)
    for idx in selected_indices:
        station_name = stations[idx]
        # 提取数据 (保持原有逻辑)
        station_data = origin_data.isel(station=idx).values[n_delays*delay_interval:]
        
        # 注意这里使用的是 ax1.plot
        ax1.plot(times_final, station_data, label=f'Station: {station_name}', alpha=0.7, linewidth=1)

    # -------------------------------------------------
    # 右侧 Y 轴 (ax2)：绘制最后一条宏观数据线
    # -------------------------------------------------
    ax2 = ax1.twinx()  # 关键步骤：创建共享X轴的第二个Y轴
    ax2.set_ylabel('Macro Data Value', color='red', fontsize=12) # 右轴标签，设为红色以区分
    ax2.tick_params(axis='y', labelcolor='red') # 设置右轴刻度颜色为红色

    # 绘制最后一条线 (画在 ax2 上)
    # 注意这里使用的是 ax2.plot
    ax2.plot(times_final, macro_data, color="red", linestyle='--', alpha=0.3, label=f"y (Right Axis)")

    # -------------------------------------------------
    # 合并图例 (让两个轴的图例显示在一起)
    # -------------------------------------------------
    # 分别获取两个轴的图例句柄和标签
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    # 合并并显示
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    # 优化时间轴显示
    fig.autofmt_xdate()

    plt.show()

from typing import Callable, List, Tuple

def compute_gram_matrix_for_sindy(library, sample_points_list, weights=None):
    """
    专门为 PySINDy Library 对象优化的 Gram 矩阵计算函数
    
    参数：
    -----------
    library : ConcatLibrary 或其他 SINDy Library 对象
        观测函数库
    sample_points : np.ndarray
        采样点数据，形状为 (M, state_dim)
    weights : np.ndarray, optional
        权重，形状为 (M,)
    """
    
    all_samples = []
    trajectory_lengths = []
    
    for traj in sample_points_list:
        if isinstance(traj, np.ndarray):
            all_samples.append(traj)
            trajectory_lengths.append(traj.shape[0])
        else:
            raise TypeError(f"每个时间序列应该是 np.ndarray，得到 {type(traj)}")
    
    # 合并所有样本点
    X_all = np.vstack(all_samples)  # 形状 (总样本数 M, 状态维数 state_dim)
    M = X_all.shape[0]  # 总样本数
    num_trajectories = len(sample_points_list)
    
    print(f"检测到 {num_trajectories} 条轨迹")
    print(f"各轨迹时间步数: {trajectory_lengths}")
    print(f"总样本数 M = {M}")
    
    # ========== 步骤2：使用 library 计算观测函数值 ==========
    Phi = library.transform(X_all)  # 形状 (M, N)
    M_check, N = Phi.shape
    
    print(f"观测函数个数 N = {N}")
    assert M_check == M, f"样本数不匹配: {M_check} != {M}"
    
    # ========== 步骤3：处理权重 ==========
    if weights is None or (isinstance(weights, str) and weights == "uniform"):
        # 所有样本点等权重
        w = np.ones(M) / M
        print("使用均匀权重（所有样本等权重）")
        
    elif isinstance(weights, str) and weights == "traj":
        # 按轨迹等权重：每条轨迹的权重和为 1/num_trajectories
        w = np.zeros(M)
        idx = 0
        for traj_len in trajectory_lengths:
            # 该条轨迹内部的点等权重
            w[idx : idx + traj_len] = 1.0 / (num_trajectories * traj_len)
            idx += traj_len
        print(f"使用轨迹等权重（每条轨迹权重和 = {1/num_trajectories:.4f}）")
        
    elif isinstance(weights, np.ndarray):
        # 自定义权重
        if len(weights) != M:
            raise ValueError(f"权重长度 {len(weights)} 与总样本数 {M} 不匹配")
        w = weights / np.sum(weights)  # 归一化
        print("使用自定义权重")
        
    else:
        raise ValueError(f"不支持的权重类型: {type(weights)}")
    
    # ========== 步骤4：计算 Gram 矩阵 ==========
    # G_ij = (1/M) * Σ Phi[m, i] * conj(Phi[m, j]) 加权版本
    # 矩阵形式：G = Phi.T @ diag(w) @ Phi
    
    # 高效的向量化计算
    Phi_weighted = Phi * w[:, np.newaxis]  # 形状 (M, N)，每行乘以对应的权重
    G = Phi_weighted.T @ Phi  # 形状 (N, N)
    
    # 如果都是实数，返回实矩阵
    if np.allclose(G.imag, 0):
        G = G.real
    
    print(f"Gram 矩阵形状: {G.shape}")
        
    return G


def inverse_sqrt_psd(M, eps=1e-10):
    """
    对称半正定矩阵的稳定平方根和逆平方根。

    参数
    ----------
    M : np.ndarray
        对称半正定矩阵。
    eps : float, optional
        特征值截断阈值，避免数值不稳定。

    返回
    ----------
    M_sqrt : np.ndarray
        M 的平方根。
    M_inv_sqrt : np.ndarray
        M 的逆平方根（在截断后的支撑子空间上定义）。
    evals : np.ndarray
        原始特征值。
    """
    M_sym = 0.5 * (M + M.T)
    evals, evecs = np.linalg.eigh(M_sym)
    evals_clipped = np.clip(evals, eps, None)
    M_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals_clipped)) @ evecs.T
    M_sqrt = evecs @ np.diag(np.sqrt(evals_clipped)) @ evecs.T
    return np.real_if_close(M_sqrt), np.real_if_close(M_inv_sqrt), evals


def _stack_snapshot_pairs(sample_points_list, library=None, lag_steps=1):
    """
    将时间序列堆叠成一步转移配对样本 (X_t, X_{t+1})。
    """
    if lag_steps < 1:
        raise ValueError(f"lag_steps 必须是正整数，得到 {lag_steps}")

    x_pairs = []
    y_pairs = []
    pair_counts = []

    for traj in sample_points_list:
        if not isinstance(traj, np.ndarray):
            raise TypeError(f"每个时间序列应该是 np.ndarray，得到 {type(traj)}")
        if traj.ndim != 2:
            raise ValueError(f"每个时间序列都应为二维数组，得到 shape={traj.shape}")
        if traj.shape[0] <= lag_steps:
            raise ValueError(
                f"每条时间序列至少需要 {lag_steps + 1} 个时间点，"
                f"才能构造 lag_steps={lag_steps} 的配对样本"
            )

        x_curr = traj[:-lag_steps]
        x_next = traj[lag_steps:]
        if library is not None:
            x_curr = library.transform(x_curr)
            x_next = library.transform(x_next)

        x_pairs.append(x_curr)
        y_pairs.append(x_next)
        pair_counts.append(x_curr.shape[0])

    X_all = np.vstack(x_pairs)
    Y_all = np.vstack(y_pairs)
    return X_all, Y_all, pair_counts


def _build_pair_weights(num_pairs, pair_counts, weights=None):
    """
    为一步转移配对样本构造归一化权重。
    """
    if weights is None or (isinstance(weights, str) and weights == "uniform"):
        w = np.ones(num_pairs) / num_pairs
        weight_mode = "uniform"
    elif isinstance(weights, str) and weights == "traj":
        w = np.zeros(num_pairs)
        idx = 0
        num_traj = len(pair_counts)
        for count in pair_counts:
            w[idx: idx + count] = 1.0 / (num_traj * count)
            idx += count
        weight_mode = "traj"
    elif isinstance(weights, np.ndarray):
        if len(weights) != num_pairs:
            raise ValueError(f"权重长度 {len(weights)} 与配对样本数 {num_pairs} 不匹配")
        if np.any(weights < 0):
            raise ValueError("权重必须非负")
        total_weight = np.sum(weights)
        if total_weight <= 0:
            raise ValueError("权重和必须为正")
        w = weights / total_weight
        weight_mode = "custom"
    else:
        raise ValueError(f"不支持的权重类型: {type(weights)}")

    return w, weight_mode


def compute_transition_covariances(sample_points_list, library=None, weights=None, lag_steps=1):
    """
    根据 lagged 配对样本估计 C00, C01, C11。

    参数
    ----------
    sample_points_list : list[np.ndarray]
        每个元素是一条时间序列，shape 为 (T, d)。
    library : optional
        若提供，则先对 X_t 和 X_{t+1} 分别做 lift。
    weights : {"uniform", "traj"} or np.ndarray, optional
        配对样本权重。

    返回
    ----------
    stats : dict
        包含 C00, C01, C11, X, Y, weights, pair_counts。
    """
    X_all, Y_all, pair_counts = _stack_snapshot_pairs(
        sample_points_list,
        library=library,
        lag_steps=lag_steps,
    )
    num_pairs = X_all.shape[0]
    w, weight_mode = _build_pair_weights(num_pairs, pair_counts, weights=weights)

    X_weighted = X_all * w[:, np.newaxis]
    Y_weighted = Y_all * w[:, np.newaxis]

    C00 = X_weighted.T @ X_all
    C01 = X_weighted.T @ Y_all
    C11 = Y_weighted.T @ Y_all

    return {
        "C00": np.real_if_close(C00),
        "C01": np.real_if_close(C01),
        "C11": np.real_if_close(C11),
        "X": X_all,
        "Y": Y_all,
        "weights": w,
        "weight_mode": weight_mode,
        "pair_counts": pair_counts,
        "lag_steps": lag_steps,
    }


def whiten_operator_matrix(A, C00, C11, eps=1e-10):
    """
    对行向量约定下的离散一步算子 A 做正确的双边白化。

    这里 A 满足 Y ≈ X @ A，对应的白化矩阵为
        K_bar = C00^{1/2} A C11^{-1/2}.

    注意：只有当 A 与同一批配对样本诱导的 C00, C01 满足
        A = C00^{dagger} C01
    时，K_bar 才等于标准的
        C00^{-1/2} C01 C11^{-1/2},
    并自动具有奇异值不超过 1 的保证。
    """
    C00_sqrt, C00_inv_sqrt, evals00 = inverse_sqrt_psd(C00, eps=eps)
    C11_sqrt, C11_inv_sqrt, evals11 = inverse_sqrt_psd(C11, eps=eps)
    A_bar = C00_sqrt @ A @ C11_inv_sqrt
    return {
        "A_bar": np.real_if_close(A_bar),
        "C00_sqrt": C00_sqrt,
        "C00_inv_sqrt": C00_inv_sqrt,
        "C11_sqrt": C11_sqrt,
        "C11_inv_sqrt": C11_inv_sqrt,
        "evals00": evals00,
        "evals11": evals11,
    }


def fit_data_koopman_operator(
    sample_points_list,
    library=None,
    weights=None,
    eps=1e-10,
    ridge=0.0,
    lag_steps=1,
):
    """
    用一步配对样本直接拟合离散 Koopman 矩阵，并返回正确双边白化后的矩阵。

    对行向量约定 Y ≈ X @ A，有
        A = C00^{dagger} C01,
        K_bar = C00^{1/2} A C11^{-1/2} = C00^{-1/2} C01 C11^{-1/2}.

    返回字典里同时包含：
    - A：数据驱动拟合得到的一步离散算子
    - K_bar：标准白化后的 Koopman 矩阵
    - C00/C01/C11 以及对应的平方根、逆平方根
    """
    stats = compute_transition_covariances(
        sample_points_list,
        library=library,
        weights=weights,
        lag_steps=lag_steps,
    )
    C00 = stats["C00"]
    C01 = stats["C01"]
    C11 = stats["C11"]

    if ridge > 0:
        C00_reg = C00 + ridge * np.eye(C00.shape[0])
    else:
        C00_reg = C00

    A = np.linalg.pinv(C00_reg) @ C01

    whitening = whiten_operator_matrix(A, C00, C11, eps=eps)
    K_bar_direct = whitening["C00_inv_sqrt"] @ C01 @ whitening["C11_inv_sqrt"]

    result = dict(stats)
    result.update(whitening)
    result["A"] = np.real_if_close(A)
    result["K_bar"] = np.real_if_close(K_bar_direct)
    result["K_bar_from_A"] = whitening["A_bar"]
    return result


# 初始化 notebook/实验过程中统一使用的结果容器，便于分块保存中间结果。
def init_artifacts(config):
    return {
        "config": dict(config),
        "raw": {},
        "prep": {},
        "obs": {},
        "cov": {},
        "koopman": {},
        "spectral": {},
        "metrics": {},
        "macro": {},
        "summary": {},
    }


def _log_pdet_psd(matrix, eps=1e-10):
    matrix_sym = 0.5 * (matrix + matrix.T)
    evals = np.linalg.eigvalsh(matrix_sym)
    positive = evals[evals > eps]
    if positive.size == 0:
        return float("-inf")
    return float(np.sum(np.log(positive)))


# 从双边白化矩阵 K_bar 统一计算奇异值分解、D_K、N_K 以及主总分 G_alpha^K。
def analyze_kbar_metrics(K_bar, alpha=1.0, eps=1e-10):
    K_bar = np.real_if_close(np.asarray(K_bar, dtype=float))

    # 1. 奇异值分解：主谱结构、左右奇异向量、有效维数。
    U, singular_values, Vt = np.linalg.svd(K_bar, full_matrices=False)
    effective_rank = int(np.sum(singular_values > eps))
    rho2 = np.clip(singular_values**2, eps, 1.0 - eps)

    # 2. 未来侧残差与确定性算子 D_K。
    residual_future = np.eye(K_bar.shape[1]) - K_bar.T @ K_bar
    residual_future = 0.5 * (residual_future + residual_future.T)
    D_K = np.linalg.pinv(residual_future, rcond=eps)
    D_K = 0.5 * (D_K + D_K.T)

    # 3. 当前侧增强项与非简并性算子 N_K。
    N_K = K_bar @ D_K @ K_bar.T
    N_K = 0.5 * (N_K + N_K.T)

    # 4. 单通道得分与总分 G_alpha^K。
    coeff_n = 0.5 - alpha / 4.0
    coeff_d = alpha / 4.0
    channel_scores = (
        coeff_n * np.log(rho2 / (1.0 - rho2))
        + coeff_d * np.log(1.0 / (1.0 - rho2))
    )
    log_pdet_D_K = _log_pdet_psd(D_K, eps=eps)
    log_pdet_N_K = _log_pdet_psd(N_K, eps=eps)
    G_alpha_K = coeff_n * log_pdet_N_K + coeff_d * log_pdet_D_K

    return {
        "U": np.real_if_close(U),
        "S": np.real_if_close(singular_values),
        "Vt": np.real_if_close(Vt),
        "effective_rank": effective_rank,
        "rho2": np.real_if_close(rho2),
        "residual_future": np.real_if_close(residual_future),
        "D_K": np.real_if_close(D_K),
        "N_K": np.real_if_close(N_K),
        "channel_scores": np.real_if_close(channel_scores),
        "log_pdet_D_K": float(log_pdet_D_K),
        "log_pdet_N_K": float(log_pdet_N_K),
        "G_alpha_K": float(G_alpha_K),
        "sigma_max": float(np.max(singular_values)) if singular_values.size else 0.0,
    }


# 根据 K_bar 的前 r 个左奇异向量构造宏观变量与粗粒化系数，并可选构造未来侧宏观变量。
def build_macro_from_kbar(
    U,
    S,
    Vt,
    C00_inv_sqrt,
    X,
    r,
    feature_names=None,
    Y=None,
    C11_inv_sqrt=None,
    center=False,
    rr=None,
):
    U = np.asarray(U, dtype=float)
    S = np.asarray(S, dtype=float)
    Vt = np.asarray(Vt, dtype=float)
    C00_inv_sqrt = np.asarray(C00_inv_sqrt, dtype=float)
    X = np.asarray(X, dtype=float)

    if r < 1:
        raise ValueError(f"r 必须为正整数，得到 {r}")
    if r > U.shape[1]:
        raise ValueError(f"r={r} 超过左奇异向量列数 {U.shape[1]}")

    X_work = X - np.mean(X, axis=0, keepdims=True) if center else X
    U_r = U[:, :r]
    if rr is not None:
        start, end = rr
        U_r = U[:, start:end]
    V_r = Vt.T[:, :r]

    # 当前侧白化变量 xi_t 与宏观变量 z_t。
    xi = X_work @ C00_inv_sqrt
    z_current = xi @ U_r

    # 粗粒化函数在原观测坐标中的系数。
    coarse_grain_matrix = C00_inv_sqrt @ U_r

    dominant_features = []
    if feature_names is not None:
        feature_names = list(feature_names)
        for component_idx in range(r):
            weights = np.abs(coarse_grain_matrix[:, component_idx])
            order = np.argsort(-weights)
            top_items = [
                {
                    "feature": feature_names[idx],
                    "weight": float(coarse_grain_matrix[idx, component_idx]),
                    "abs_weight": float(weights[idx]),
                }
                for idx in order[: min(5, len(order))]
            ]
            dominant_features.append(top_items)

    macro_future = None
    eta = None
    if Y is not None and C11_inv_sqrt is not None:
        Y = np.asarray(Y, dtype=float)
        C11_inv_sqrt = np.asarray(C11_inv_sqrt, dtype=float)
        Y_work = Y - np.mean(Y, axis=0, keepdims=True) if center else Y
        eta = Y_work @ C11_inv_sqrt
        macro_future = eta @ V_r

    return {
        "r": int(r),
        "U_r": np.real_if_close(U_r),
        "V_r": np.real_if_close(V_r),
        "singular_values_r": np.real_if_close(S[:r]),
        "xi": np.real_if_close(xi),
        "z_current": np.real_if_close(z_current),
        "eta": np.real_if_close(eta) if eta is not None else None,
        "z_future": np.real_if_close(macro_future) if macro_future is not None else None,
        "coarse_grain_matrix": np.real_if_close(coarse_grain_matrix),
        "dominant_features": dominant_features,
    }


def _format_summary_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return str(tuple(value))
    if isinstance(value, list):
        return np.array2string(np.asarray(value), precision=4, separator=", ")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return np.array2string(value, precision=4, separator=", ")
    if isinstance(value, pd.DataFrame):
        return value
    return str(value)


# 统一打印最终实验摘要，既支持分段字典，也支持扁平字典。
def print_summary(summary_dict):
    if not isinstance(summary_dict, dict):
        raise TypeError("summary_dict 必须是 dict")

    print("=" * 88)
    print("Summary")
    print("=" * 88)

    for key, value in summary_dict.items():
        if isinstance(value, dict):
            print(f"\n[{key}]")
            rows = [{"item": sub_key, "value": _format_summary_value(sub_value)} for sub_key, sub_value in value.items()]
            section_df = pd.DataFrame(rows)
            print(section_df.to_string(index=False))
        elif isinstance(value, pd.DataFrame):
            print(f"\n[{key}]")
            print(value.to_string(index=False))
        else:
            print(f"{key}: {_format_summary_value(value)}")

    print("=" * 88)


def generate_two_population_neuron_data(
    n_a=400, n_b=400,              # 两个群体的神经元数量
    alpha_a=4.6,                 # α群体的α参数（可激发性）
    alpha_b=4.6,                 # β群体的α参数
    sigma_a=0.225,                 # α群体的σ参数（异质性）
    sigma_b=0.225,                 # β群体的σ参数
    mu=0.001,                    # 慢变参数（局部动力学）
    gamma=0.08,                   # 群体内耦合强度（μ）
    epsilon=0.04,                # 群体间耦合强度（ε）
    T=10000,                     # 总模拟步数
    transients=1000,             # 舍弃的暂态步数
    x0_a=-1, x0_b=-0.5,          # a/b群体x初始值（None则该群体独立随机）
    y0_a=-3.5, y0_b=-3, 
    seed=42                      # 随机种子
):
    """
    生成两个群体神经元数据，基于平均场耦合动力学。
    
    参数:
        n_a, n_b: 两个群体的神经元数量
        alpha_a, alpha_b: 两个群体的可激发性参数
        sigma_a, sigma_b: 两个群体的异质性参数
        mu: 慢变参数（局部动力学）
        gamma: 群体内耦合强度（对应图中的μ）
        epsilon: 群体间耦合强度（对应图中的ε）
        T: 总模拟步数
        transients: 舍弃的暂态步数
        seed: 随机种子
        
    返回:
        dict: 包含所有数据的字典
    """
    # 设置随机种子
    np.random.seed(seed)
    
    # 总神经元数
    N = n_a + n_b
    
    # 处理参数：如果是标量，扩展为列表
    if isinstance(alpha_a, (int, float)):
        alpha_a = np.full(n_a, alpha_a)
    else:
        alpha_a = np.array(alpha_a)
        
    if isinstance(alpha_b, (int, float)):
        alpha_b = np.full(n_b, alpha_b)
    else:
        alpha_b = np.array(alpha_b)
    
    if isinstance(sigma_a, (int, float)):
        sigma_a = np.full(n_a, sigma_a)
    else:
        sigma_a = np.array(sigma_a)
        
    if isinstance(sigma_b, (int, float)):
        sigma_b = np.full(n_b, sigma_b)
    else:
        sigma_b = np.array(sigma_b)
    # 合并所有参数
    alpha_all = np.concatenate([alpha_a, alpha_b])
    sigma_all = np.concatenate([sigma_a, sigma_b])
            
    # 初始化数组
    x_series = np.zeros((N, T))
    y_series = np.zeros((N, T))
     
    if x0_a is None:
        # None → a群体每个神经元独立随机
        x_series[:n_a, 0] = np.random.uniform(-1.5, -0.5, n_a)
        x0_a_record = "随机值（每个神经元独立）"
    else:
        # 传值 → a群体所有神经元共用该值
        x_series[:n_a, 0] = x0_a
        x0_a_record = x0_a
    
    if x0_b is None:
        # None → b群体每个神经元独立随机
        x_series[n_a:, 0] = np.random.uniform(-1.5, -0.5, n_b)
        x0_b_record = "随机值（每个神经元独立）"
    else:
        # 传值 → b群体所有神经元共用该值
        x_series[n_a:, 0] = x0_b
        x0_b_record = x0_b
    
    # 处理a群体y初始值
    if y0_a is None:
        y_series[:n_a, 0] = np.random.uniform(-4, -3, n_a)
        y0_a_record = "随机值（每个神经元独立）"
    else:
        y_series[:n_a, 0] = y0_a
        y0_a_record = y0_a
    
    # 处理b群体y初始值
    if y0_b is None:
        y_series[n_a:, 0] = np.random.uniform(-4, -3, n_b)
        y0_b_record = "随机值（每个神经元独立）"
    else:
        y_series[n_a:, 0] = y0_b
        y0_b_record = y0_b
    
    # 定义分段函数 f (局部动力学)
    def f(x_val, y_val, alpha):
        """Chialvo神经元模型的非线性函数"""
        if x_val <= 0:
            return alpha / (1 - x_val) + y_val
        elif 0 < x_val < alpha + y_val:
            return alpha + y_val
        else:
            return -1
    
    # 定义函数 g (慢变量动力学)
    def g(x_val, y_val, mu, sigma):
        """慢变量的更新函数"""
        return y_val - mu * (x_val + 1) + mu * sigma
    
    # 模拟迭代
    for t in range(T-1):
        # 计算两个群体的平均场
        # a群体的平均场
        Xbar_a = np.mean(x_series[:n_a, t])
        # b群体的平均场
        Xbar_b = np.mean(x_series[n_a:, t])
        
        # 更新a群体的神经元
        for i in range(n_a):
            # 局部动力学部分
            local_part = f(x_series[i, t], y_series[i, t], alpha_all[i])
            
            # 根据公式: x_{t+1} = (1-γ)f(...) + γXbar_a + εXbar_b
            x_series[i, t+1] = (1 - gamma) * local_part + gamma * Xbar_a + epsilon * Xbar_b
            
            # 根据公式: y_{t+1} = g(x_t, y_t)
            y_series[i, t+1] = g(x_series[i, t], y_series[i, t], mu, sigma_all[i])
        
        # 更新b群体的神经元
        for j in range(n_b):
            idx = n_a + j  # 在总数组中的索引
            
            # 局部动力学部分
            local_part = f(x_series[idx, t], y_series[idx, t], alpha_all[idx])
            
            # 根据公式: x_{t+1} = (1-γ)f(...) + γXbar_b + εXbar_a
            x_series[idx, t+1] = (1 - gamma) * local_part + gamma * Xbar_b + epsilon * Xbar_a
            
            # 根据公式: y_{t+1} = g(x_t, y_t)
            y_series[idx, t+1] = g(x_series[idx, t], y_series[idx, t], mu, sigma_all[idx])
        
    # 计算同步指标
    def compute_instantaneous_std(x_data, mean_field):
        N_neurons = x_data.shape[0]
        T_steps = x_data.shape[1]
        r_t = np.zeros(T_steps)
        
        for t in range(T_steps):
            # 计算每个时刻的标准差
            r_t[t] = np.sqrt(np.mean((x_data[:, t] - mean_field[t])**2))
        
        return r_t
    
    # 计算平均场时间序列
    Xbar_a = np.zeros(T)
    Xbar_b = np.zeros(T)
    
    for t in range(T):
        Xbar_a[t] = np.mean(x_series[:n_a, t])
        Xbar_b[t] = np.mean(x_series[n_a:, t])
    Xbar_a_transient = Xbar_a[transients:]
    Xbar_b_transient = Xbar_b[transients:]
    # 计算瞬时标准差
    r_a_t = compute_instantaneous_std(x_series[:n_a, :], Xbar_a)
    r_b_t = compute_instantaneous_std(x_series[n_a:, :], Xbar_b)
    r_a_t_transient = r_a_t[transients:]
    r_b_t_transient = r_b_t[transients:]
    
    Xbar_all = np.zeros(T)
    for t in range(T):
        Xbar_all[t] = np.mean(x_series[:, t])
        
    r_t = compute_instantaneous_std(x_series, Xbar_all)
    r_t_transient = r_t[transients:]
    
    # 计算时间平均
    T_effective = T - transients
    R_a = np.mean(r_a_t_transient) 
    R_b = np.mean(r_b_t_transient) 
    R_t = np.mean(r_t_transient) 
    R_delta = np.mean(np.abs(Xbar_a[transients:] - Xbar_b[transients:]))  
    
    # 判断同步状态 (基于图片中的定义)
    def determine_sync_state(R_a, R_b, R_delta, threshold=1e-7):
        sync_a = R_a < threshold
        sync_b = R_b < threshold
        sync_ab = R_delta < threshold
        
        if sync_a and sync_b and sync_ab:
            return "Complete Synchronization 完全同步(CS)"
        elif sync_a and sync_b and not sync_ab:
            return "Generalized Synchronization 广义同步 (GS)"
        elif (sync_a and not sync_b) or (not sync_a and sync_b):
            return "Chimera State 奇美拉态(Q)"
        elif not sync_a and not sync_b:
            return "Desynchronization 去同步化(D)"
        else:
            return "Unknown State 未知"
    
    sync_state = determine_sync_state(R_a, R_b, R_delta)
    
    # 准备参数信息
    params = {
        '神经元总数': N,
        'a群体神经元数': n_a,
        'b群体神经元数': n_b,
        'a群体α参数': alpha_a[0] if isinstance(alpha_a, np.ndarray) and len(np.unique(alpha_a)) == 1 else f"列表({len(alpha_a)}个)",
        'b群体α参数': alpha_b[0] if isinstance(alpha_b, np.ndarray) and len(np.unique(alpha_b)) == 1 else f"列表({len(alpha_b)}个)",
        'a群体σ参数': sigma_a[0] if isinstance(sigma_a, np.ndarray) and len(np.unique(sigma_a)) == 1 else f"列表({len(sigma_a)}个)",
        'b群体σ参数': sigma_b[0] if isinstance(sigma_b, np.ndarray) and len(np.unique(sigma_b)) == 1 else f"列表({len(sigma_b)}个)",
        '慢变参数μ': mu,
        '群体内耦合强度γ': gamma,
        '群体间耦合强度ε': epsilon,
        '总步数': T,
        '舍弃暂态': transients,
        '有效数据长度': T_effective,
        '随机种子': seed
    }
    
    # 动力学特征
    dynamics_stats = {
        '⟨σa⟩ (R_a)': R_a,
        '⟨σb⟩ (R_b)': R_b,
        '⟨σt⟩ (R_t)': R_t,
        '⟨δ⟩ (R_delta)': R_delta,
        '同步状态': sync_state,
        'a群体同步': R_a < 1e-7,
        'b群体同步': R_b < 1e-7,
        'Rt群体同步': R_t < 1e-7,
        '群体间同步': R_delta < 1e-7,
        'r_a_t': r_a_t,
        'r_b_t': r_b_t,
        'r_t': r_t
    }
    # 准备返回数据
    data = {
        'params': {**params, **dynamics_stats},
        '群体信息': {
            'a群体神经元数': n_a,
            'b群体神经元数': n_b,
            'a群体索引': list(range(n_a)),
            'b群体索引': list(range(n_a, n_a + n_b)),
            'a群体α参数': alpha_a,
            'b群体α参数': alpha_b,
            'a群体σ参数': sigma_a,
            'b群体σ参数': sigma_b
        },
        '时间序列': {
            't': np.arange(T_effective),
            'Xbar_a': Xbar_a,  # a群体平均场
            'Xbar_b': Xbar_b,  # b群体平均场
            'Xbar_a_transient': Xbar_a_transient,  # a群体平均场
            'Xbar_b_transient': Xbar_b_transient,  # b群体平均场
            'r_a_t': r_a_t,   # a群体瞬时标准差
            'r_b_t': r_b_t,   # b群体瞬时标准差
            'r_t': r_t,   # 全部瞬时标准差
            'r_a_t_transient': r_a_t_transient,   # a群体瞬时标准差
            'r_b_t_transient': r_b_t_transient,   # b群体瞬时标准差
            'r_t_transient': r_t_transient   # 全部瞬时标准差
        },
        '同步指标': {
            'R_a': R_a,      # ⟨σα⟩的时间平均
            'R_b': R_b,      # ⟨σβ⟩的时间平均
            'R_t': R_t,      
            'R_delta': R_delta,  # ⟨δ⟩的时间平均
            'sync_state': sync_state
        }
    }
    
    # 添加每个神经元的完整时间序列
    for i in range(n_a):
        data[f'神经元_a_{i+1:03d}'] = {
            'x_transient': x_series[i, transients:],
            'y_transient': y_series[i, transients:],
            'x': x_series[i],
            'y': y_series[i],
            '群体': 'a',
            'α参数': alpha_a[i] if isinstance(alpha_a, np.ndarray) else alpha_a,
            'σ参数': sigma_a[i] if isinstance(sigma_a, np.ndarray) else sigma_a
        }
    
    for j in range(n_b):
        idx = n_a + j
        data[f'神经元_b_{j+1:03d}'] = {
            'x_transient': x_series[idx, transients:],
            'y_transient': y_series[idx, transients:],
            'x': x_series[idx],
            'y': y_series[idx],
            '群体': 'b',
            'α参数': alpha_b[j] if isinstance(alpha_b, np.ndarray) else alpha_b,
            'σ参数': sigma_b[j] if isinstance(sigma_b, np.ndarray) else sigma_b
        }
    
    return data

def plot_neuron_analysis_combo(data, figsize=(20, 8), time_window=None, 
                               vmin=None, vmax=None, cmap='viridis'):
    """
    绘制神经元分析组合图：左边热力图，右边瞬时标准差时间序列
    
    参数:
        data: generate_two_population_neuron_data函数返回的数据字典
        figsize: 图形大小
        time_window: 时间窗口
        vmin, vmax: 颜色映射的最小值和最大值
        cmap: 颜色映射
        show_transient: 是否显示暂态部分
    """
    # 从数据中提取信息
    params = data['params']
    group_info = data['群体信息']
    sync_info = data['同步指标']
    time_series = data['时间序列']
    
    n_a = group_info['a群体神经元数']
    n_b = group_info['b群体神经元数']
    N = n_a + n_b
    T_total = params.get('总步数', 10000)
    transients = params.get('舍弃暂态', 1000)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # ========== 左图：热力图 ==========
    T_effective = T_total - transients
    # 提取所有神经元的x值
    x_data = np.zeros((N, T_effective))
    
    # 提取a群体神经元的x值
    for i in range(n_a):
        neuron_key = f'神经元_a_{i+1:03d}'
        x_data[i, :] = data[neuron_key]['x_transient']
    
    # 提取b群体神经元的x值
    for j in range(n_b):
        neuron_key = f'神经元_b_{j+1:03d}'
        idx = n_a + j
        x_data[idx, :] = data[neuron_key]['x_transient']
    
    # 如果指定了时间窗口
    if time_window is not None:
        start_t, end_t = time_window
        if end_t > T_effective:
            end_t = T_effective
        x_data = x_data[:, start_t:end_t]
        t_display = np.arange(end_t - start_t)
    else:
        t_display = np.arange(T_effective)
        start_t, end_t = 0, T_effective
    
    # 确定颜色映射范围
    if vmin is None:
        vmin = np.min(x_data)
    if vmax is None:
        vmax = np.max(x_data)
    
    # 绘制热力图
    im = ax1.imshow(x_data, aspect='auto', cmap=cmap, 
                   extent=[0, len(t_display), 0, N],
                   vmin=vmin, vmax=vmax,
                   origin='lower', interpolation='nearest')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax1, pad=0.01, shrink=0.8)
    cbar.set_label('神经元状态 x 值', fontsize=12)
    
    # 设置坐标轴
    ax1.set_xlabel('时间步 t', fontsize=12)
    ax1.set_ylabel('神经元索引', fontsize=12)
    
    # 设置y轴刻度，只显示群体标签
    ax1.set_yticks([n_a/2, n_a + n_b/2])
    ax1.set_yticklabels(['群体a', '群体b'], fontsize=12)
    
    # 在两个群体之间添加分隔线
    ax1.axhline(y=n_a, color='white', linestyle='--', linewidth=2, alpha=0.8)
    
    # 设置标题
    sync_state = sync_info.get('sync_state', 'Unknown')
    title1 = f'神经元群体状态热力图 (同步状态: {sync_state})'
    if time_window is not None:
        title1 += f' (时间窗口: {time_window[0]}-{time_window[1]})'
    ax1.set_title(title1, fontsize=14, fontweight='bold', pad=20)
    
    # ========== 右图：瞬时标准差时间序列 ==========
    # 获取瞬时标准差数据
    r_a_t = time_series['r_a_t']
    r_b_t = time_series['r_b_t']
    r_t = time_series['r_t']
    
    t_plot = np.arange(T_total)
    transient_line = transients
    
    # 绘制三条曲线
    ax2.plot(t_plot, r_a_t, label='r_a(t)', 
         color='blue', alpha=0.8, linewidth=1.5, linestyle=':', marker='.', markersize=5,markevery=50)  # 点线
    ax2.plot(t_plot, r_b_t, label='r_b(t)', 
         color='red', alpha=0.8, linewidth=1.5, linestyle=':', marker='s', markersize=2,markevery=50)  # 方框线
    ax2.plot(t_plot, r_t, label='r(t)', 
         color='green', alpha=0.8, linewidth=1.5, linestyle='--')  # 虚线
    
    ax2.axvline(x=transient_line, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    
    # 设置坐标轴
    ax2.set_xlabel('时间步 t', fontsize=12)
    ax2.set_ylabel('瞬时标准差', fontsize=12)
    ax2.set_title('瞬时标准差时间序列', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(fontsize=11)
    ax2.set_facecolor('white')  
    ax2.grid(False)  
    
    # 调整布局
    ax1.grid(False)
    plt.tight_layout()
    plt.show()

# Rulkov 两群体原始数据字典中提取统一的状态矩阵，支持同时保留 x 与 y 两类状态。
def extract_state_matrix_from_rulkov_data(
    data,
    include_x=True,
    include_y=True,
    use_transient=True,
):
    if not include_x and not include_y:
        raise ValueError("include_x 和 include_y 不能同时为 False")
    if "群体信息" not in data:
        raise KeyError("输入数据缺少 '群体信息' 键")

    group_info = data["群体信息"]
    n_a = int(group_info["a群体神经元数"])
    n_b = int(group_info["b群体神经元数"])

    cols = []
    state_names = []
    group_labels = []

    time_suffix = "transient" if use_transient else ""
    x_key = "x_transient" if use_transient else "x"
    y_key = "y_transient" if use_transient else "y"

    for idx in range(1, n_a + 1):
        neuron_key = f"神经元_a_{idx:03d}"
        neuron_data = data[neuron_key]
        if include_x:
            cols.append(np.asarray(neuron_data[x_key], dtype=float))
            state_names.append(f"x_a{idx}")
            group_labels.append("a")
        if include_y:
            cols.append(np.asarray(neuron_data[y_key], dtype=float))
            state_names.append(f"y_a{idx}")
            group_labels.append("a")

    for idx in range(1, n_b + 1):
        neuron_key = f"神经元_b_{idx:03d}"
        neuron_data = data[neuron_key]
        if include_x:
            cols.append(np.asarray(neuron_data[x_key], dtype=float))
            state_names.append(f"x_b{idx}")
            group_labels.append("b")
        if include_y:
            cols.append(np.asarray(neuron_data[y_key], dtype=float))
            state_names.append(f"y_b{idx}")
            group_labels.append("b")

    x_data_raw = np.column_stack(cols)

    sync_metrics = {}
    if "同步指标" in data:
        sync_metrics = {
            "R_a": data["同步指标"].get("R_a"),
            "R_b": data["同步指标"].get("R_b"),
            "R_t": data["同步指标"].get("R_t"),
            "R_delta": data["同步指标"].get("R_delta"),
            "sync_state": data["同步指标"].get("sync_state"),
        }

    time_data = None
    if "时间序列" in data and "t" in data["时间序列"]:
        time_data = np.asarray(data["时间序列"]["t"], dtype=float)

    return {
        "x_data_raw": np.asarray(x_data_raw, dtype=float),
        "state_names": state_names,
        "group_labels": group_labels,
        "time_data": time_data,
        "sync_metrics": sync_metrics,
        "meta": {
            "n_a": n_a,
            "n_b": n_b,
            "include_x": bool(include_x),
            "include_y": bool(include_y),
            "use_transient": bool(use_transient),
            "time_key_suffix": time_suffix,
        },
    }


# 将多种 map 状态的关键单值指标整理成统一比较表，便于最终汇总展示。
def build_map_comparison_table(state_results):
    if not isinstance(state_results, dict):
        raise TypeError("state_results 必须是 dict")

    rows = []
    for state_name, result in state_results.items():
        metrics = result.get("metrics", {})
        sync_metrics = result.get("sync_metrics", {})
        singular_values = np.asarray(result.get("singular_values", []), dtype=float)

        row = {
            "state": state_name,
            "sync_state": sync_metrics.get("sync_state"),
            "R_a": sync_metrics.get("R_a"),
            "R_b": sync_metrics.get("R_b"),
            "R_t": sync_metrics.get("R_t"),
            "R_delta": sync_metrics.get("R_delta"),
            "G_alpha_K": metrics.get("G_alpha_K"),
            "EC": metrics.get("EC"),
            "r": metrics.get("selected_r"),
            "CE": metrics.get("delta_g_selected_r"),
            "effective_rank": metrics.get("effective_rank"),
            "sigma1": float(singular_values[0]) if singular_values.size >= 1 else np.nan,
            "sigma2": float(singular_values[1]) if singular_values.size >= 2 else np.nan,
            "sigma3": float(singular_values[2]) if singular_values.size >= 3 else np.nan,
        }
        rows.append(row)

    return pd.DataFrame(rows)


# =========================
# Air-quality helpers
# =========================

def _infer_yrd_province_from_city(city_name):
    """根据长三角常见地级市名称推断所属省份。"""
    if city_name is None or (isinstance(city_name, float) and np.isnan(city_name)):
        return None
    city_name = str(city_name).strip()
    mapping = {
        "上海市": "上海市",
        "南京市": "江苏省","无锡市": "江苏省","徐州市": "江苏省","常州市": "江苏省","苏州市": "江苏省",
        "南通市": "江苏省","连云港市": "江苏省","淮安市": "江苏省","盐城市": "江苏省","扬州市": "江苏省",
        "镇江市": "江苏省","泰州市": "江苏省","宿迁市": "江苏省","杭州市": "浙江省","宁波市": "浙江省",
        "温州市": "浙江省","嘉兴市": "浙江省","湖州市": "浙江省","绍兴市": "浙江省","金华市": "浙江省",
        "衢州市": "浙江省","舟山市": "浙江省","台州市": "浙江省",
        "合肥市": "安徽省","芜湖市": "安徽省","马鞍山市": "安徽省","铜陵市": "安徽省","安庆市": "安徽省",
        "滁州市": "安徽省","池州市": "安徽省","宣城市": "安徽省",
    }
    return mapping.get(city_name, None)


def _read_station_meta_csv(station_meta_path):
    """稳健读取站点元数据表，并补充 province 列。"""
    read_errors = []
    for encoding in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            station_meta = pd.read_csv(station_meta_path, encoding=encoding)
            break
        except Exception as exc:  # pragma: no cover - fallback path
            read_errors.append((encoding, repr(exc)))
    else:  # pragma: no cover - fallback path
        raise RuntimeError(
            f"无法读取站点元数据文件: {station_meta_path}。尝试编码失败记录: {read_errors}"
        )

    station_meta = station_meta.copy()
    for required_col in ("station_id", "city", "lon", "lat"):
        if required_col not in station_meta.columns:
            raise KeyError(f"站点元数据缺少必要列: {required_col}")

    station_meta["station_id"] = station_meta["station_id"].astype(str)
    station_meta["city"] = station_meta["city"].astype(str)
    station_meta["lon"] = pd.to_numeric(station_meta["lon"], errors="coerce")
    station_meta["lat"] = pd.to_numeric(station_meta["lat"], errors="coerce")

    if "province" not in station_meta.columns:
        station_meta["province"] = station_meta["city"].map(_infer_yrd_province_from_city)

    return station_meta


def _open_air_dataset_robust(dataset_path, engine_preference=None):
    """依次尝试不同引擎打开空气数据集。"""
    import xarray as xr

    tried = []
    engines = []
    if engine_preference is not None:
        engines.append(engine_preference)
    engines.extend(["h5netcdf", "netcdf4", "scipy", None])

    seen = set()
    for engine in engines:
        if engine in seen:
            continue
        seen.add(engine)
        try:
            if engine is None:
                return xr.open_dataset(dataset_path)
            return xr.open_dataset(dataset_path, engine=engine)
        except Exception as exc:  # pragma: no cover - environment dependent
            tried.append((engine, repr(exc)))

    raise RuntimeError(
        f"无法打开空气数据文件: {dataset_path}。尝试引擎失败记录: {tried}"
    )


def _normalize_name_list(values):
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        return [str(values)]
    return [str(v) for v in values]


def load_air_data_subset(
    dataset_path,
    station_meta_path,
    subset_mode="all",
    province_names=None,
    city_names=None,
    station_ids=None,
    variables=None,
    time_slice=None,
    engine_preference=None,
):
    """
    读取空气数据并按区域/站点筛选子集。

    参数
    ----------
    dataset_path : str or Path
        NetCDF 数据文件路径。
    station_meta_path : str or Path
        站点元数据 CSV 路径。
    subset_mode : str
        可选: "all", "province", "provinces", "city", "cities",
        "station", "stations", "custom"。
    province_names, city_names, station_ids : str or list[str], optional
        用于筛选的省、市、站点列表。
    variables : list[str], optional
        需要保留的变量；若为 None，则保留全部数据变量。
    time_slice : slice or tuple or list, optional
        时间切片。支持:
        - slice(start, stop)
        - (start, stop)
        - [start, stop]
    engine_preference : str, optional
        指定 xarray 打开引擎优先项。

    返回
    ----------
    result : dict
        包含 ds_subset, station_meta_subset, selected_station_ids,
        selected_variables, available_variables, subset_mode 等。
    """
    station_meta = _read_station_meta_csv(station_meta_path)
    ds = _open_air_dataset_robust(dataset_path, engine_preference=engine_preference)

    if "station" not in ds.dims:
        raise KeyError("空气数据集中缺少 station 维度")

    available_variables = list(ds.data_vars)
    selected_variables = available_variables if variables is None else list(variables)
    missing_variables = [var for var in selected_variables if var not in available_variables]
    if missing_variables:
        raise KeyError(f"所选变量在数据集中不存在: {missing_variables}")

    province_names = _normalize_name_list(province_names)
    city_names = _normalize_name_list(city_names)
    station_ids = _normalize_name_list(station_ids)

    mask = pd.Series(True, index=station_meta.index)
    subset_mode = str(subset_mode).lower()

    if subset_mode == "all":
        pass
    elif subset_mode == "province":
        if not province_names or len(province_names) != 1:
            raise ValueError("subset_mode='province' 时需要且只允许提供 1 个 province_names")
        mask &= station_meta["province"].astype(str).isin(province_names)
    elif subset_mode == "provinces":
        if not province_names:
            raise ValueError("subset_mode='provinces' 时需要提供 province_names")
        mask &= station_meta["province"].astype(str).isin(province_names)
    elif subset_mode == "city":
        if not city_names or len(city_names) != 1:
            raise ValueError("subset_mode='city' 时需要且只允许提供 1 个 city_names")
        mask &= station_meta["city"].astype(str).isin(city_names)
    elif subset_mode == "cities":
        if not city_names:
            raise ValueError("subset_mode='cities' 时需要提供 city_names")
        mask &= station_meta["city"].astype(str).isin(city_names)
    elif subset_mode == "station":
        if not station_ids or len(station_ids) != 1:
            raise ValueError("subset_mode='station' 时需要且只允许提供 1 个 station_ids")
        mask &= station_meta["station_id"].astype(str).isin(station_ids)
    elif subset_mode == "stations":
        if not station_ids:
            raise ValueError("subset_mode='stations' 时需要提供 station_ids")
        mask &= station_meta["station_id"].astype(str).isin(station_ids)
    elif subset_mode == "custom":
        if province_names is not None:
            mask &= station_meta["province"].astype(str).isin(province_names)
        if city_names is not None:
            mask &= station_meta["city"].astype(str).isin(city_names)
        if station_ids is not None:
            mask &= station_meta["station_id"].astype(str).isin(station_ids)
    else:
        raise ValueError(f"不支持的 subset_mode: {subset_mode}")

    station_meta_subset = station_meta.loc[mask].reset_index(drop=True)
    if station_meta_subset.empty:
        raise ValueError("筛选后没有站点，请检查 subset_mode 与筛选参数")

    selected_station_ids = station_meta_subset["station_id"].astype(str).tolist()

    # 优先按 station 坐标的站点 id 对齐；若不存在则退回到顺序截取。
    if "station" in ds.coords:
        station_coord_values = [str(v) for v in np.asarray(ds["station"].values).tolist()]
        station_lookup = {sid: idx for idx, sid in enumerate(station_coord_values)}
        if all(sid in station_lookup for sid in selected_station_ids):
            station_indices = [station_lookup[sid] for sid in selected_station_ids]
        else:
            if ds.dims["station"] != len(station_meta):
                raise ValueError(
                    "数据集 station 坐标无法与站点元数据按 station_id 对齐，且 station 数量也不一致"
                )
            original_lookup = {
                sid: idx for idx, sid in enumerate(station_meta["station_id"].astype(str).tolist())
            }
            station_indices = [original_lookup[sid] for sid in selected_station_ids]
    else:
        if ds.dims["station"] != len(station_meta):
            raise ValueError("数据集缺少 station 坐标且 station 数量与元数据不一致，无法安全对齐")
        original_lookup = {
            sid: idx for idx, sid in enumerate(station_meta["station_id"].astype(str).tolist())
        }
        station_indices = [original_lookup[sid] for sid in selected_station_ids]

    ds_subset = ds.isel(station=station_indices)
    ds_subset = ds_subset[selected_variables]

    if time_slice is not None:
        if isinstance(time_slice, slice):
            ds_subset = ds_subset.isel(time=time_slice)
        elif isinstance(time_slice, (tuple, list)) and len(time_slice) == 2:
            ds_subset = ds_subset.sel(time=slice(time_slice[0], time_slice[1]))
        else:
            raise ValueError("time_slice 必须是 slice 或长度为 2 的 tuple/list")

    return {
        "ds_subset": ds_subset,
        "station_meta_subset": station_meta_subset,
        "selected_station_ids": selected_station_ids,
        "selected_variables": selected_variables,
        "available_variables": available_variables,
        "subset_mode": subset_mode,
        "dataset_path": str(dataset_path),
        "station_meta_path": str(station_meta_path),
        "station_indices": station_indices,
    }


def build_air_feature_matrix(
    air_subset,
    variables=None,
    feature_name_style="station_var",
):
    """
    从筛选后的空气数据构造微观状态矩阵与特征名。

    参数
    ----------
    air_subset : dict
        load_air_data_subset 的返回结果。
    variables : list[str], optional
        若提供，则只从 air_subset 中再次挑选这些变量。
    feature_name_style : str
        目前支持 "station_var" -> "{station_id}_{var}"。

    返回
    ----------
    result : dict
        包含 x_data_raw, feature_names, times, station_meta, variables,
        variable_slices, station_ids 等。
    """
    if not isinstance(air_subset, dict) or "ds_subset" not in air_subset:
        raise TypeError("air_subset 必须是 load_air_data_subset 返回的 dict")

    ds_subset = air_subset["ds_subset"]
    station_meta = air_subset["station_meta_subset"].reset_index(drop=True).copy()
    station_ids = station_meta["station_id"].astype(str).tolist()

    selected_variables = (
        air_subset["selected_variables"] if variables is None else list(variables)
    )
    missing_variables = [var for var in selected_variables if var not in ds_subset.data_vars]
    if missing_variables:
        raise KeyError(f"所选变量在 ds_subset 中不存在: {missing_variables}")

    matrices = []
    feature_names = []
    variable_slices = {}
    start_idx = 0

    for var in selected_variables:
        data_array = ds_subset[var]
        if "time" not in data_array.dims or "station" not in data_array.dims:
            raise ValueError(f"变量 {var} 必须同时包含 time 和 station 维度")

        data_2d = data_array.transpose("time", "station").values
        data_2d = np.asarray(data_2d, dtype=float)
        if data_2d.shape[1] != len(station_ids):
            raise ValueError(f"变量 {var} 的 station 维与站点元数据长度不一致")

        matrices.append(data_2d)
        if feature_name_style != "station_var":
            raise ValueError(f"不支持的 feature_name_style: {feature_name_style}")

        current_names = [f"{sid}_{var}" for sid in station_ids]
        feature_names.extend(current_names)
        variable_slices[var] = slice(start_idx, start_idx + len(current_names))
        start_idx += len(current_names)

    x_data_raw = np.column_stack(matrices)
    times = ds_subset["time"].values if "time" in ds_subset.coords else np.arange(x_data_raw.shape[0])

    return {
        "x_data_raw": np.asarray(x_data_raw, dtype=float),
        "feature_names": feature_names,
        "times": np.asarray(times),
        "station_meta": station_meta,
        "station_ids": station_ids,
        "variables": selected_variables,
        "variable_slices": variable_slices,
        "n_stations": len(station_ids),
        "n_features": x_data_raw.shape[1],
    }


def summarize_air_subset(air_subset):
    """
    汇总空气数据子集的基本信息，便于 notebook 中直接展示。

    返回
    ----------
    summary : dict
        包含 summary_df, province_counts, city_counts 等表。
    """
    if not isinstance(air_subset, dict) or "station_meta_subset" not in air_subset:
        raise TypeError("air_subset 必须是 load_air_data_subset 返回的 dict")

    station_meta = air_subset["station_meta_subset"].reset_index(drop=True)
    ds_subset = air_subset["ds_subset"]
    selected_variables = air_subset.get("selected_variables", [])

    province_counts = (
        station_meta.groupby("province", dropna=False)["station_id"]
        .count()
        .reset_index(name="station_count")
        .sort_values(["station_count", "province"], ascending=[False, True])
        .reset_index(drop=True)
    )
    city_counts = (
        station_meta.groupby(["province", "city"], dropna=False)["station_id"]
        .count()
        .reset_index(name="station_count")
        .sort_values(["station_count", "province", "city"], ascending=[False, True, True])
        .reset_index(drop=True)
    )

    time_values = ds_subset["time"].values if "time" in ds_subset.coords else np.array([])
    if time_values.size > 0:
        time_start = str(time_values[0])
        time_end = str(time_values[-1])
        time_length = int(time_values.size)
    else:
        time_start = None
        time_end = None
        time_length = int(ds_subset.dims.get("time", 0))

    summary_df = pd.DataFrame(
        [
            {
                "subset_mode": air_subset.get("subset_mode"),
                "station_count": int(len(station_meta)),
                "province_count": int(station_meta["province"].nunique(dropna=True)),
                "city_count": int(station_meta["city"].nunique(dropna=True)),
                "variable_count": int(len(selected_variables)),
                "variables": ", ".join(map(str, selected_variables)),
                "time_length": time_length,
                "time_start": time_start,
                "time_end": time_end,
            }
        ]
    )

    return {
        "summary_df": summary_df,
        "province_counts": province_counts,
        "city_counts": city_counts,
        "station_meta": station_meta,
    }


def _coerce_map_panels(coarse_grain_coff, delay=0, title_suffix=""):
    """
    将输入统一解析为可绘制面板列表。

    支持两类输入：
    1. ndarray / DataFrame : 每一列对应一个面板；
    2. list[dict] : 每个 dict 至少包含 data，可选 title/delay/title_suffix。
    """
    panels = []

    if isinstance(coarse_grain_coff, (list, tuple)):
        for item_idx, item in enumerate(coarse_grain_coff):
            if isinstance(item, dict):
                data = item.get("data")
                if data is None:
                    raise ValueError(f"第 {item_idx} 个面板 dict 缺少 data")
                panel_delay = item.get("delay", delay)
                panel_suffix = item.get("title_suffix", title_suffix)
                panel_title = item.get("title", None)
                panel_df = pd.DataFrame(data).reset_index(drop=True)
                for col_idx in panel_df.columns:
                    if panel_title is None:
                        title = f"y{int(col_idx) + 1 if isinstance(col_idx, (int, np.integer)) else col_idx}_d{panel_delay}{panel_suffix}"
                    else:
                        title = str(panel_title)
                    panels.append(
                        {
                            "values": np.asarray(panel_df[col_idx], dtype=float),
                            "title": title,
                            "delay": panel_delay,
                        }
                    )
            else:
                panel_df = pd.DataFrame(item).reset_index(drop=True)
                for col_idx in panel_df.columns:
                    panels.append(
                        {
                            "values": np.asarray(panel_df[col_idx], dtype=float),
                            "title": f"y{int(col_idx) + 1 if isinstance(col_idx, (int, np.integer)) else col_idx}_d{delay}{title_suffix}",
                            "delay": delay,
                        }
                    )
        return panels

    coff_df = pd.DataFrame(coarse_grain_coff).reset_index(drop=True)
    for col_idx in coff_df.columns:
        display_idx = int(col_idx) + 1 if isinstance(col_idx, (int, np.integer)) else col_idx
        panels.append(
            {
                "values": np.asarray(coff_df[col_idx], dtype=float),
                "title": f"y{display_idx}_d{delay}{title_suffix}",
                "delay": delay,
            }
        )
    return panels


def plot_station_with_map(
    df,
    coarse_grain_coff,
    delay=0,
    title_suffix="",
    vmax=None,
    vmin=None,
    ncols=None,
    figsize=None,
    dpi=300,
    cmap="viridis",
    marker_size=60,
    edge_linewidth=0.4,
    sort_points=True,
    show_colorbar=True,
    colorbar_location="right",
    colorbar_label=None,
    return_fig=False,
):
    """
    在地图上绘制站点权重分布，支持单图与网格子图。

    兼容旧接口：
    - 仍支持传入单个 ndarray/DataFrame，并通过 delay/title_suffix 生成标题；
    - 保留地图底图、海岸线、湖海、河流、重点边界、颜色映射等功能。

    扩展能力：
    - 若输入包含多列，则自动按网格子图排版；
    - 所有子图共用统一的颜色范围，便于跨延迟层/跨变量比较；
    - 默认按数值从小到大排序绘点，让较大的点盖在上层，更显眼。
    """
    import math
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    df = df.copy().reset_index(drop=True)
    for required_col in ("lon", "lat"):
        if required_col not in df.columns:
            raise KeyError(f"地图绘制数据缺少必要列: {required_col}")

    panels = _coerce_map_panels(
        coarse_grain_coff=coarse_grain_coff,
        delay=delay,
        title_suffix=title_suffix,
    )
    if len(panels) == 0:
        raise ValueError("没有可绘制的地图面板")

    expected_len = len(df)
    for panel in panels:
        if len(panel["values"]) != expected_len:
            raise ValueError(
                f"面板 {panel['title']} 的长度 ({len(panel['values'])}) 与站点数 ({expected_len}) 不一致"
            )

    all_values = np.concatenate([panel["values"] for panel in panels]).astype(float)
    global_vmin = float(np.nanmin(all_values)) if vmin is None else float(vmin)
    global_vmax = float(np.nanmax(all_values)) if vmax is None else float(vmax)
    if np.isclose(global_vmin, global_vmax):
        global_vmax = global_vmin + 1e-12

    num_panels = len(panels)
    if ncols is None:
        ncols = min(3, num_panels)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(num_panels / ncols))

    if figsize is None:
        figsize = (5.6 * ncols, 4.8 * nrows)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        dpi=dpi,
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    extent = [
        float(df["lon"].min()) - 0.3,
        float(df["lon"].max()) + 0.3,
        float(df["lat"].min()) - 0.3,
        float(df["lat"].max()) + 0.3,
    ]

    scatter_ref = None
    for ax, panel in zip(axes, panels):
        df_plot = df.copy()
        df_plot["value_to_plot"] = panel["values"]
        if sort_points:
            df_plot = df_plot.sort_values("value_to_plot")

        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#fdfcf0")
        ax.add_feature(cfeature.OCEAN, facecolor="#e3f2fd")
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor="grey")
        ax.add_feature(
            cfeature.STATES.with_scale("10m"),
            linestyle="--",
            edgecolor="grey",
            linewidth=0.7,
        )

        scatter_ref = ax.scatter(
            df_plot["lon"],
            df_plot["lat"],
            c=df_plot["value_to_plot"],
            cmap=cmap,
            vmin=global_vmin,
            vmax=global_vmax,
            s=marker_size,
            alpha=1.0,
            edgecolors="black",
            linewidth=edge_linewidth,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )

        ax.set_title(str(panel["title"]), fontsize=12)
        gl = ax.gridlines(draw_labels=True, linestyle=":", alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}

    for ax in axes[num_panels:]:
        ax.set_visible(False)

    if show_colorbar and scatter_ref is not None:
        if colorbar_location == "right":
            cbar = fig.colorbar(
                scatter_ref,
                ax=[ax for ax in axes[:num_panels]],
                location="right",
                fraction=0.022,
                pad=0.02,
                shrink=0.92,
            )
        elif colorbar_location == "bottom":
            cbar = fig.colorbar(
                scatter_ref,
                ax=[ax for ax in axes[:num_panels]],
                location="bottom",
                fraction=0.05,
                pad=0.06,
                shrink=0.92,
            )
        else:
            raise ValueError(f"不支持的 colorbar_location: {colorbar_location}")
        if colorbar_label:
            cbar.set_label(str(colorbar_label), fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    if return_fig:
        return fig, axes[:num_panels]
    plt.show()


def compute_gamma_ce_metrics(singular_values, alpha=1.0, rank_candidates=None, manual_r=None, eps=1e-10):
    """
    基于定义 12 和定义 13 计算白化 Koopman 的可逆性与 CE 指标。

    参数
    ----
    singular_values : array-like
        白化 Koopman 矩阵 K_bar 的奇异值。
    alpha : float, default=1.0
        定义 12/13 中的 alpha 参数。
    rank_candidates : iterable[int] or None
        参与搜索的候选 r。若为 None，则默认使用 1..effective_rank。
    manual_r : int or None
        用户手动指定的 r。若为 None，则不计算对应单值。
    eps : float, default=1e-10
        判定有效奇异值的阈值。

    返回
    ----
    dict
        包含：
        - reversibility_channel_scores = sigma_i^alpha
        - Gamma_alpha_K
        - gamma_alpha_K
        - effective_rank
        - rank_candidates
        - delta_gamma_by_r
        - selected_r
        - delta_gamma_selected_r
        - manual_r
        - delta_gamma_manual_r
    """
    singular_values = np.real_if_close(np.asarray(singular_values, dtype=float).ravel())
    if singular_values.ndim != 1:
        raise ValueError("singular_values 必须是一维数组")

    effective_mask = singular_values > eps
    effective_rank = int(np.sum(effective_mask))
    if effective_rank <= 0:
        raise ValueError("没有有效奇异值，无法计算 Gamma/CE 指标")

    effective_singular_values = singular_values[:effective_rank]
    reversibility_channel_scores = np.power(effective_singular_values, alpha)
    Gamma_alpha_K = float(np.sum(reversibility_channel_scores))
    gamma_alpha_K = float(np.mean(reversibility_channel_scores))

    if rank_candidates is None:
        rank_candidates = list(range(1, effective_rank + 1))
    else:
        rank_candidates = sorted(
            {
                int(r)
                for r in rank_candidates
                if 1 <= int(r) <= effective_rank
            }
        )
        if not rank_candidates:
            raise ValueError("rank_candidates 过滤后为空，请检查候选 r 范围")

    delta_gamma_by_r = {
        int(r): float(np.mean(reversibility_channel_scores[: int(r)]) - gamma_alpha_K)
        for r in rank_candidates
    }

    selected_r = int(max(delta_gamma_by_r, key=delta_gamma_by_r.get))
    delta_gamma_selected_r = float(delta_gamma_by_r[selected_r])

    manual_r_out = None if manual_r is None else int(manual_r)
    delta_gamma_manual_r = None
    if manual_r_out is not None:
        if not (1 <= manual_r_out <= effective_rank):
            raise ValueError(
                f"manual_r={manual_r_out} 超出有效维数范围 1..{effective_rank}"
            )
        delta_gamma_manual_r = float(
            np.mean(reversibility_channel_scores[:manual_r_out]) - gamma_alpha_K
        )

    return {
        "reversibility_channel_scores": np.real_if_close(reversibility_channel_scores),
        "Gamma_alpha_K": Gamma_alpha_K,
        "gamma_alpha_K": gamma_alpha_K,
        "effective_rank": effective_rank,
        "rank_candidates": rank_candidates,
        "delta_gamma_by_r": delta_gamma_by_r,
        "selected_r": selected_r,
        "delta_gamma_selected_r": delta_gamma_selected_r,
        "manual_r": manual_r_out,
        "delta_gamma_manual_r": delta_gamma_manual_r,
    }


# ============================================================================
# GIS pipeline helpers
# 新增说明：
# - 以下函数服务于 `研究框架.md` 附录 C / D 所定义的 GIS 主流程。
# - 设计原则是：只新增、不影响现有函数；参数尽量显式、可扩展，便于后续做
#   tau / noise / alpha / rank / eps 等扫描实验。
# ============================================================================

def make_step_system_matrix(lam, mu):
    """
    构造第一类 step 系统在观测坐标 [x, y, x^2] 下的解析矩阵 A。

    Parameters
    ----------
    lam : float
        x_{k+1} = lam * x_k 中的参数。
    mu : float
        y_{k+1} = mu * y_k + (lam^2 - mu) * x_k^2 中的参数。

    Returns
    -------
    np.ndarray of shape (3, 3)
        观测层解析动力学矩阵。
    """
    return np.array(
        [
            [lam, 0.0, 0.0],
            [0.0, mu, lam ** 2 - mu],
            [0.0, 0.0, lam ** 2],
        ],
        dtype=float,
    )


def step_map(x, y, lam, mu):
    """
    第一类 step 系统的一步映射。

    Parameters
    ----------
    x, y : float
        当前状态。
    lam, mu : float
        系统参数。

    Returns
    -------
    tuple[float, float]
        下一步状态 (x_next, y_next)。
    """
    x_next = lam * x
    y_next = mu * y + (lam ** 2 - mu) * (x ** 2)
    return x_next, y_next


def observable_step(data_xy, mode='default'):
    """
    第一类 step 系统的默认观测函数。

    当前默认将二维状态 [x, y] 提升到观测层 [x, y, x^2]。
    后续若要改观测函数，可通过 mode 扩展。

    Parameters
    ----------
    data_xy : array-like of shape (T, 2)
        原始二维状态序列。
    mode : str, default='default'
        观测模式。目前支持：
        - 'default' : [x, y, x^2]

    Returns
    -------
    np.ndarray of shape (T, 3)
        观测层数据。
    """
    data_xy = np.asarray(data_xy, dtype=float)
    if data_xy.ndim != 2 or data_xy.shape[1] != 2:
        raise ValueError("data_xy 必须是形状为 (T, 2) 的二维数组")

    x = data_xy[:, 0]
    y = data_xy[:, 1]

    if mode == 'default':
        return np.column_stack([x, y, x ** 2])

    raise ValueError(f"暂不支持的 observable_step mode: {mode}")


def observable_step2(data_xy, mode='default'):
    """
    step2 系统的默认观测函数。

    当前默认将二维状态 [x, y] 提升到观测层 [x, y, y^2]。

    Parameters
    ----------
    data_xy : array-like of shape (T, 2)
        原始二维状态序列。
    mode : str, default='default'
        观测模式。目前支持：
        - 'default' : [x, y, y^2]

    Returns
    -------
    np.ndarray of shape (T, 3)
        观测层数据。
    """
    data_xy = np.asarray(data_xy, dtype=float)
    if data_xy.ndim != 2 or data_xy.shape[1] != 2:
        raise ValueError("data_xy 必须是形状为 (T, 2) 的二维数组")

    x = data_xy[:, 0]
    y = data_xy[:, 1]

    if mode == 'default':
        return np.column_stack([x, y, y ** 2])

    raise ValueError(f"暂不支持的 observable_step2 mode: {mode}")


def simulate_discrete_system(map_func, initial_states, steps, system_kwargs=None, dt=1.0):
    """
    用统一接口模拟离散系统轨迹。

    Parameters
    ----------
    map_func : callable
        一步映射函数，签名应为 map_func(x, y, **system_kwargs) -> (x_next, y_next)。
    initial_states : array-like
        初始状态。支持：
        - shape (2,) : 单条轨迹初值
        - shape (N, 2) : 多条轨迹初值
    steps : int
        演化步数。
    system_kwargs : dict or None
        传给 map_func 的额外参数。
    dt : float, default=1.0
        时间步长，仅用于生成时间轴。

    Returns
    -------
    dict
        {
            "trajectories": np.ndarray, shape (N, steps + 1, 2),
            "time_grid": np.ndarray, shape (steps + 1,),
        }
    """
    if system_kwargs is None:
        system_kwargs = {}

    initial_states = np.asarray(initial_states, dtype=float)
    if initial_states.ndim == 1:
        if initial_states.shape[0] != 2:
            raise ValueError("单个初始状态必须是长度为 2 的向量")
        initial_states = initial_states[None, :]
    if initial_states.ndim != 2 or initial_states.shape[1] != 2:
        raise ValueError("initial_states 必须是形状为 (2,) 或 (N, 2) 的数组")

    n_traj = initial_states.shape[0]
    trajectories = np.zeros((n_traj, steps + 1, 2), dtype=float)
    trajectories[:, 0, :] = initial_states

    for i in range(n_traj):
        x, y = initial_states[i]
        for t in range(steps):
            x, y = map_func(x, y, **system_kwargs)
            trajectories[i, t + 1, 0] = x
            trajectories[i, t + 1, 1] = y

    time_grid = np.arange(steps + 1, dtype=float) * dt
    return {"trajectories": trajectories, "time_grid": time_grid}


def add_gaussian_noise(data, noise_scale=1.0, cov=None, random_state=None):
    """
    给数据叠加高斯噪声。

    Parameters
    ----------
    data : array-like of shape (..., d)
        原始数据，最后一维视为特征维度。
    noise_scale : float, default=1.0
        噪声强度。若 cov 为单位阵，则等价于标准正态乘以 noise_scale。
    cov : array-like of shape (d, d) or None
        噪声协方差矩阵。若为 None，则默认使用单位阵。
    random_state : int, np.random.Generator, or None
        随机种子或随机数发生器。

    Returns
    -------
    dict
        {
            "noisy_data": np.ndarray,
            "noise": np.ndarray,
            "cov": np.ndarray,
        }
    """
    data = np.asarray(data, dtype=float)
    if data.ndim < 2:
        raise ValueError("data 至少应为二维，最后一维表示特征")

    feature_dim = data.shape[-1]
    cov = np.eye(feature_dim, dtype=float) if cov is None else np.asarray(cov, dtype=float)
    if cov.shape != (feature_dim, feature_dim):
        raise ValueError(f"cov 形状必须为 ({feature_dim}, {feature_dim})")

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    flat_shape = int(np.prod(data.shape[:-1]))
    noise = rng.multivariate_normal(
        mean=np.zeros(feature_dim, dtype=float),
        cov=(noise_scale ** 2) * cov,
        size=flat_shape,
    ).reshape(data.shape)

    return {
        "noisy_data": data + noise,
        "noise": noise,
        "cov": (noise_scale ** 2) * cov,
    }


def prepare_time_pairs(data, tau=1, burn_in=0, stride=1):
    """
    根据时间尺度 tau、预热步数 burn_in 与采样步长 stride 构造时间配对样本。

    Parameters
    ----------
    data : array-like of shape (T, d)
        时间序列数据。
    tau : int, default=1
        配对间隔。
    burn_in : int, default=0
        丢弃开头若干步。
    stride : int, default=1
        抽样步长。

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (X_now, X_next)，两者形状均为 (N, d)。
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data 必须是形状为 (T, d) 的二维数组")
    if tau <= 0 or stride <= 0 or burn_in < 0:
        raise ValueError("tau、stride 必须为正，burn_in 必须非负")

    data_used = data[burn_in::stride]
    if len(data_used) <= tau:
        raise ValueError("样本太短，无法构造给定 tau 的配对数据")

    x_now = data_used[:-tau]
    x_next = data_used[tau:]
    return x_now, x_next


def _safe_symmetrize(matrix):
    """返回对称化后的矩阵，便于协方差与谱运算的数值稳定。"""
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def _regularized_pinv(matrix, regularization=1e-10):
    """
    计算带正则化的伪逆。

    Parameters
    ----------
    matrix : array-like
        输入矩阵。
    regularization : float
        对角正则项强度。

    Returns
    -------
    np.ndarray
        正则化伪逆。
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return np.linalg.pinv(matrix, rcond=regularization)
    reg_eye = regularization * np.eye(matrix.shape[0], dtype=float)
    return np.linalg.pinv(matrix + reg_eye, rcond=regularization)


def _pseudo_logdet_positive(matrix, eps=1e-10):
    """
    计算半正定矩阵的对数伪行列式。

    Parameters
    ----------
    matrix : array-like
        输入矩阵。
    eps : float
        判定有效特征值的阈值。

    Returns
    -------
    tuple[float, np.ndarray]
        (log_pdet, positive_eigenvalues)
    """
    matrix = _safe_symmetrize(matrix)
    eigvals = np.linalg.eigvalsh(matrix)
    positive = eigvals[eigvals > eps]
    if len(positive) == 0:
        return -np.inf, positive
    return float(np.sum(np.log(positive))), positive


def estimate_covariance_from_residuals(residuals, center=True, regularization=1e-10):
    """
    从残差样本估计协方差矩阵。

    Parameters
    ----------
    residuals : array-like of shape (N, d)
        残差样本。
    center : bool, default=True
        是否先减去样本均值。
    regularization : float, default=1e-10
        对角正则项，避免协方差奇异。

    Returns
    -------
    np.ndarray of shape (d, d)
        残差协方差矩阵。
    """
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 2:
        raise ValueError("residuals 必须是形状为 (N, d) 的二维数组")
    if len(residuals) == 0:
        raise ValueError("residuals 为空，无法估计协方差")

    res = residuals - residuals.mean(axis=0, keepdims=True) if center else residuals
    n_samples = res.shape[0]
    cov = (res.T @ res) / max(n_samples - 1, 1)
    cov = _safe_symmetrize(cov)
    cov += regularization * np.eye(cov.shape[0], dtype=float)
    return cov


def fit_linear_gis_from_pairs(X_now, X_next, fit_intercept=False, ridge=0.0, regularization=1e-10):
    """
    从配对样本拟合线性 GIS：
        X_next ≈ A X_now + residual

    Parameters
    ----------
    X_now, X_next : array-like of shape (N, d)
        当前时刻与未来时刻的配对数据。
    fit_intercept : bool, default=False
        是否拟合截距。当前 notebook 主链默认不拟合截距。
    ridge : float, default=0.0
        岭回归正则项强度。
    regularization : float, default=1e-10
        协方差与伪逆计算时使用的数值稳定项。

    Returns
    -------
    dict
        {
            "A": A,
            "K_raw": K_raw,
            "residuals": residuals,
            "Sigma": Sigma,
            "C00": C00,
            "C01": C01,
            "C11": C11,
            "intercept": intercept,
        }
    """
    X_now = np.asarray(X_now, dtype=float)
    X_next = np.asarray(X_next, dtype=float)
    if X_now.ndim != 2 or X_next.ndim != 2 or X_now.shape != X_next.shape:
        raise ValueError("X_now 与 X_next 必须是形状相同的二维数组")

    n_samples, dim = X_now.shape
    if n_samples < 2:
        raise ValueError("样本数过少，至少需要 2 个样本")

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
    # Keep the returned empirical operator aligned with the column-vector
    # dynamics convention used elsewhere in the repo, so K_raw and A refer
    # to the same matrix instead of a transpose pair.
    empirical_regression = _regularized_pinv(C00_reg, regularization=regularization) @ C01
    K_raw = empirical_regression.T
    A = K_raw

    if fit_intercept:
        intercept = mean_next - A @ mean_now

    predictions = (X_now @ A.T) + intercept
    residuals = X_next - predictions
    Sigma = estimate_covariance_from_residuals(residuals, center=True, regularization=regularization)

    return {
        "A": A,
        "K_raw": K_raw,
        "residuals": residuals,
        "Sigma": Sigma,
        "C00": C00,
        "C01": C01,
        "C11": C11,
        "intercept": intercept,
    }


def fit_linear_gis_from_matrix(A, state_dim=None, Sigma=None, sigma_eps=1e-10):
    """
    已知解析矩阵 A 时，将其包装为统一的 GIS 对象。

    Parameters
    ----------
    A : array-like of shape (d, d)
        已知动力学矩阵。
    state_dim : int or None
        状态维度。若为 None，则由 A 自动推断。
    Sigma : array-like of shape (d, d) or None
        已知噪声协方差矩阵。若为 None，则使用 sigma_eps * I 作为近零噪声近似。
    sigma_eps : float, default=1e-10
        当 Sigma 为 None 时使用的近零噪声强度。

    Returns
    -------
    dict
        {
            "A": A,
            "Sigma": Sigma,
            "state_dim": d,
            "meta": {...},
        }
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A 必须是方阵")

    d = A.shape[0] if state_dim is None else int(state_dim)
    if d != A.shape[0]:
        raise ValueError("state_dim 与 A 的维度不一致")

    if Sigma is None:
        Sigma = sigma_eps * np.eye(d, dtype=float)
    else:
        Sigma = np.asarray(Sigma, dtype=float)
        if Sigma.shape != (d, d):
            raise ValueError(f"Sigma 形状必须为 ({d}, {d})")
        Sigma = _safe_symmetrize(Sigma)

    return {
        "A": A,
        "Sigma": Sigma,
        "state_dim": d,
        "meta": {
            "provided_matrix": True,
            "near_zero_noise": Sigma is None,
            "sigma_eps": sigma_eps,
        },
    }


def compute_gis_metrics(A, Sigma, alpha=1.0, eps=1e-10):
    """
    计算 GIS 主链中的核心指标：
    - Gamma
    - log_Gamma
    - J_alpha
    - D
    - N
    同时返回与 SVD 相关的矩阵及谱信息。

    Parameters
    ----------
    A : array-like of shape (d, d)
        动力学矩阵。
    Sigma : array-like of shape (d, d)
        噪声协方差矩阵。
    alpha : float, default=1.0
        近似可逆性中的权重参数。
    eps : float, default=1e-10
        数值稳定阈值。

    Returns
    -------
    dict
        包含 Gamma、log_Gamma、J_alpha、D、N 及谱分解所需中间量。
    """
    A = np.asarray(A, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A 必须是方阵")
    if Sigma.ndim != 2 or Sigma.shape != A.shape:
        raise ValueError("Sigma 必须与 A 维度一致")

    d = A.shape[0]
    Sigma = _safe_symmetrize(Sigma) + eps * np.eye(d, dtype=float)
    Sigma_inv = _regularized_pinv(Sigma, regularization=eps)
    backward_matrix = _safe_symmetrize(A.T @ Sigma_inv @ A)

    log_pdet_sigma_inv, sigma_inv_pos = _pseudo_logdet_positive(Sigma_inv, eps=eps)
    log_pdet_backward, backward_pos = _pseudo_logdet_positive(backward_matrix, eps=eps)

    D = log_pdet_sigma_inv
    N = log_pdet_backward
    log_Gamma = (0.5 - alpha / 4.0) * N + (alpha / 4.0) * D
    Gamma = float(np.exp(log_Gamma)) if np.isfinite(log_Gamma) else 0.0
    J_alpha = log_Gamma / d

    sv_forward = np.linalg.svd(Sigma_inv, compute_uv=False)
    sv_backward = np.linalg.svd(backward_matrix, compute_uv=False)

    return {
        "Gamma": Gamma,
        "log_Gamma": float(log_Gamma),
        "J_alpha": float(J_alpha),
        "D": float(D),
        "N": float(N),
        "Sigma_inv": Sigma_inv,
        "A_t_Sigma_inv_A": backward_matrix,
        "sv_forward": np.real_if_close(sv_forward),
        "sv_backward": np.real_if_close(sv_backward),
        "positive_eigs_forward": np.real_if_close(sigma_inv_pos),
        "positive_eigs_backward": np.real_if_close(backward_pos),
        "alpha": float(alpha),
        "dimension": int(d),
    }


def predict_linear_gis(A, X0, steps=1):
    """
    基于线性 GIS 做一步或多步预测。

    Parameters
    ----------
    A : array-like of shape (d, d)
        动力学矩阵。
    X0 : array-like of shape (N, d) or (d,)
        初始状态。
    steps : int, default=1
        预测步数。steps=1 表示单步预测。

    Returns
    -------
    np.ndarray
        预测结果，与输入 X0 在样本维度上保持一致。
    """
    A = np.asarray(A, dtype=float)
    X0 = np.asarray(X0, dtype=float)
    if X0.ndim == 1:
        X0 = X0[None, :]
    if X0.ndim != 2 or X0.shape[1] != A.shape[0]:
        raise ValueError("X0 必须是形状为 (N, d) 或 (d,) 的数组，且维度与 A 一致")
    if steps <= 0:
        raise ValueError("steps 必须为正整数")

    A_power = np.linalg.matrix_power(A, steps)
    return X0 @ A_power.T


def compute_prediction_errors(A, series, tau=1, horizons=(1,)):
    """
    统一计算线性 GIS 的单步和多步预测误差。

    Parameters
    ----------
    A : array-like of shape (d, d)
        动力学矩阵。
    series : array-like of shape (T, d)
        时间序列。
    tau : int, default=1
        基础时间尺度。当前实现中 horizon=k 表示预测 k * tau 步。
    horizons : iterable[int], default=(1,)
        需要计算的预测步数列表。

    Returns
    -------
    dict
        {
            horizon: {
                "predictions": ...,
                "targets": ...,
                "pointwise_errors": e,
                "mean_error": E,
            }
        }
    """
    series = np.asarray(series, dtype=float)
    if series.ndim != 2:
        raise ValueError("series 必须是形状为 (T, d) 的二维数组")
    if tau <= 0:
        raise ValueError("tau 必须为正整数")

    results = {}
    for horizon in horizons:
        horizon = int(horizon)
        if horizon <= 0:
            raise ValueError("horizons 中的元素必须为正整数")
        shift = horizon * tau
        if len(series) <= shift:
            raise ValueError(f"序列长度不足，无法计算 horizon={horizon} 的预测误差")

        x_now = series[:-shift]
        x_target = series[shift:]
        preds = predict_linear_gis(A, x_now, steps=horizon)
        pointwise = np.sum((x_target - preds) ** 2, axis=1)
        mean_error = float(np.mean(pointwise))
        results[horizon] = {
            "predictions": preds,
            "targets": x_target,
            "pointwise_errors": pointwise,
            "mean_error": mean_error,
        }
    return results


def compute_ce_from_micro_macro(micro_metrics, macro_metrics):
    """
    根据宏微观 GIS 指标计算宏观效率增益（CE）。

    Parameters
    ----------
    micro_metrics : dict
        微观层指标，应至少包含 J_alpha、D、N、log_Gamma。
    macro_metrics : dict
        宏观层指标，应至少包含 J_alpha、D、N、log_Gamma。

    Returns
    -------
    dict
        {
            "CE": ...,
            "delta_J": ...,
            "delta_D": ...,
            "delta_N": ...,
            "delta_log_Gamma": ...,
        }
    """
    delta_J = float(macro_metrics["J_alpha"] - micro_metrics["J_alpha"])
    delta_D = float(macro_metrics["D"] - micro_metrics["D"])
    delta_N = float(macro_metrics["N"] - micro_metrics["N"])
    delta_log_gamma = float(macro_metrics["log_Gamma"] - micro_metrics["log_Gamma"])
    return {
        "CE": delta_J,
        "delta_J": delta_J,
        "delta_D": delta_D,
        "delta_N": delta_N,
        "delta_log_Gamma": delta_log_gamma,
    }


def select_macro_rank(values, mode='gap', threshold=None, manual_r=None, eps=1e-10):
    """
    选择宏观维度 r。

    Parameters
    ----------
    values : array-like
        用于选秩的谱值，通常为奇异值谱或特征值模长谱，默认要求已按降序排列。
    mode : str, default='gap'
        选秩模式：
        - 'manual' : 使用 manual_r
        - 'threshold' : 保留大于 threshold 的谱值数目
        - 'gap' : 根据相邻谱值比值最大的间隙选取
        - 'energy' : 用 threshold 解释为累计能量比例
    threshold : float or None
        threshold / energy 模式下的阈值。
    manual_r : int or None
        手动指定的秩。
    eps : float, default=1e-10
        有效谱值阈值。

    Returns
    -------
    tuple[int, dict]
        (r, rank_meta)
    """
    values = np.real_if_close(np.asarray(values, dtype=float).ravel())
    positive = values[values > eps]
    if len(positive) == 0:
        raise ValueError("没有有效谱值，无法选择宏观维度")

    if mode == 'manual':
        if manual_r is None:
            raise ValueError("mode='manual' 时必须提供 manual_r")
        r = int(manual_r)
    elif mode == 'threshold':
        if threshold is None:
            raise ValueError("mode='threshold' 时必须提供 threshold")
        r = int(np.sum(positive >= threshold))
        r = max(r, 1)
    elif mode == 'energy':
        if threshold is None:
            raise ValueError("mode='energy' 时必须提供 threshold（建议取 0~1）")
        energy = np.cumsum(positive) / np.sum(positive)
        r = int(np.searchsorted(energy, threshold, side='left') + 1)
    elif mode == 'gap':
        if len(positive) == 1:
            r = 1
        else:
            ratios = positive[:-1] / np.maximum(positive[1:], eps)
            r = int(np.argmax(ratios) + 1)
    else:
        raise ValueError(f"不支持的 rank selection mode: {mode}")

    r = min(max(r, 1), len(positive))
    return r, {
        "mode": mode,
        "threshold": threshold,
        "manual_r": manual_r,
        "effective_rank": int(len(positive)),
        "used_values": positive,
        "selected_r": int(r),
    }


def build_w_from_svd(A, Sigma, r=None, alpha=1.0, eps=1e-10, mode='two_stage'):
    """
    基于 SVD 路线构造粗粒化矩阵 W。

    Parameters
    ----------
    A : array-like of shape (d, d)
        动力学矩阵。
    Sigma : array-like of shape (d, d)
        协方差矩阵。
    r : int or None
        宏观维度。若为 None，则根据 backward 奇异值谱自动选取。
    alpha : float, default=1.0
        当前版本中保留该参数，便于后续按 alpha 加权扩展。
    eps : float, default=1e-10
        数值阈值。
    mode : str, default='two_stage'
        构造模式：
        - 'two_stage' : 优先结合 Sigma^{-1} 与 A^T Sigma^{-1} A 的方向信息
        - 'backward_only' : 只使用 A^T Sigma^{-1} A 的左奇异向量

    Returns
    -------
    dict
        {
            "W": W,
            "r": r,
            "sv_info": ...,
            "basis_info": ...,
        }
    """
    metrics = compute_gis_metrics(A, Sigma, alpha=alpha, eps=eps)
    sigma_inv = metrics["Sigma_inv"]
    backward = metrics["A_t_Sigma_inv_A"]
    sv_forward = metrics["sv_forward"]
    sv_backward = metrics["sv_backward"]

    if r is None:
        r, _ = select_macro_rank(sv_backward, mode='gap', eps=eps)
    r = int(r)

    if mode == 'backward_only':
        U_b, _, _ = np.linalg.svd(backward, full_matrices=False)
        basis = U_b[:, :r]
    elif mode == 'two_stage':
        U_f, S_f, _ = np.linalg.svd(sigma_inv, full_matrices=False)
        U_b, S_b, _ = np.linalg.svd(backward, full_matrices=False)

        combined_vectors = np.concatenate([U_f, U_b], axis=1)
        combined_scores = np.concatenate([S_f, S_b], axis=0)
        keep = combined_scores > eps
        combined_vectors = combined_vectors[:, keep]
        combined_scores = combined_scores[keep]
        if combined_vectors.shape[1] == 0:
            raise ValueError("SVD two_stage 模式下没有可用主方向")

        weighted_vectors = combined_vectors * combined_scores
        U_c, _, _ = np.linalg.svd(weighted_vectors, full_matrices=False)
        basis = U_c[:, :r]
    else:
        raise ValueError(f"不支持的 SVD W 构造模式: {mode}")

    W = basis.T
    return {
        "W": np.real_if_close(W),
        "r": r,
        "sv_info": {
            "sv_forward": np.real_if_close(sv_forward),
            "sv_backward": np.real_if_close(sv_backward),
        },
        "basis_info": {
            "mode": mode,
            "basis": np.real_if_close(basis),
        },
    }


def build_w_from_evd(A, r=None, mode='eig_abs'):
    """
    基于特征值分解构造对照版粗粒化矩阵 W。

    Parameters
    ----------
    A : array-like of shape (d, d)
        动力学矩阵。
    r : int or None
        宏观维度。若为 None，则按特征值模长自动选取。
    mode : str, default='eig_abs'
        特征值排序方式：
        - 'eig_abs' : 按特征值模长降序
        - 'eig_real' : 按实部降序

    Returns
    -------
    dict
        {
            "W": W,
            "r": r,
            "eigvals": eigvals_sorted,
            "eigvecs": eigvecs_sorted,
        }
    """
    A = np.asarray(A, dtype=float)
    eigvals, eigvecs = np.linalg.eig(A)

    if mode == 'eig_abs':
        order = np.argsort(-np.abs(eigvals))
        score_values = np.abs(eigvals[order])
    elif mode == 'eig_real':
        order = np.argsort(-np.real(eigvals))
        score_values = np.real(eigvals[order])
    else:
        raise ValueError(f"不支持的 EVD 排序模式: {mode}")

    eigvals_sorted = eigvals[order]
    eigvecs_sorted = eigvecs[:, order]

    if r is None:
        r, _ = select_macro_rank(np.abs(score_values), mode='gap', eps=1e-10)
    r = int(r)

    basis = np.real_if_close(eigvecs_sorted[:, :r])
    basis = np.asarray(np.real(basis), dtype=float)
    q, _ = np.linalg.qr(basis)
    W = q[:, :r].T

    return {
        "W": np.real_if_close(W),
        "r": r,
        "eigvals": np.real_if_close(eigvals_sorted),
        "eigvecs": np.real_if_close(eigvecs_sorted),
    }


def apply_coarse_graining(W, O):
    """
    根据线性粗粒化矩阵 W 生成宏观数据 Z。

    Parameters
    ----------
    W : array-like of shape (r, m)
        粗粒化矩阵。
    O : array-like of shape (T, m)
        观测层数据。

    Returns
    -------
    np.ndarray of shape (T, r)
        宏观层数据。
    """
    W = np.asarray(W, dtype=float)
    O = np.asarray(O, dtype=float)
    if O.ndim != 2 or W.ndim != 2 or O.shape[1] != W.shape[1]:
        raise ValueError("W 与 O 的维度不匹配，要求 O.shape[1] == W.shape[1]")
    return O @ W.T


def summarize_pipeline_results(
    config,
    micro_fit,
    macro_fit,
    micro_metrics,
    macro_metrics,
    prediction_results,
    ce_result,
    extra=None,
):
    """
    汇总一次 GIS 主流程实验的关键结果，便于 notebook 打印与后续 DataFrame 化。

    Parameters
    ----------
    config : dict
        实验配置。
    micro_fit, macro_fit : dict
        微观/宏观层拟合结果。
    micro_metrics, macro_metrics : dict
        微观/宏观层 GIS 指标。
    prediction_results : dict
        预测误差结果，建议包含 micro_errors / macro_errors。
    ce_result : dict
        compute_ce_from_micro_macro 的输出。
    extra : dict or None
        其他补充信息。

    Returns
    -------
    tuple[dict, dict]
        (summary_dict, summary_row)
        其中 summary_row 适合直接放进 pd.DataFrame([summary_row])。
    """
    if extra is None:
        extra = {}

    summary_dict = {
        "config": config,
        "micro_fit": micro_fit,
        "macro_fit": macro_fit,
        "micro_metrics": micro_metrics,
        "macro_metrics": macro_metrics,
        "prediction_results": prediction_results,
        "ce_result": ce_result,
        "extra": extra,
    }

    micro_errors = prediction_results.get("micro_errors", {})
    macro_errors = prediction_results.get("macro_errors", {})

    summary_row = {
        "experiment_name": config.get("experiment_name"),
        "tau": config.get("tau"),
        "alpha": config.get("alpha"),
        "delta": config.get("delta"),
        "noise_scale": config.get("noise_scale"),
        "micro_dim": micro_metrics.get("dimension"),
        "macro_dim": macro_metrics.get("dimension"),
        "micro_J_alpha": micro_metrics.get("J_alpha"),
        "macro_J_alpha": macro_metrics.get("J_alpha"),
        "micro_D": micro_metrics.get("D"),
        "macro_D": macro_metrics.get("D"),
        "micro_N": micro_metrics.get("N"),
        "macro_N": macro_metrics.get("N"),
        "micro_log_Gamma": micro_metrics.get("log_Gamma"),
        "macro_log_Gamma": macro_metrics.get("log_Gamma"),
        "CE": ce_result.get("CE"),
        "delta_D": ce_result.get("delta_D"),
        "delta_N": ce_result.get("delta_N"),
        "delta_log_Gamma": ce_result.get("delta_log_Gamma"),
        "micro_E1": micro_errors.get(1, {}).get("mean_error"),
        "macro_E1": macro_errors.get(1, {}).get("mean_error"),
    }

    return summary_dict, summary_row
