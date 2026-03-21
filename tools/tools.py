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
