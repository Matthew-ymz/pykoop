"""
使用 PySINDy 训练和测试动力学模型

这个程序使用 PySINDy 算法从时间序列数据中学习动力学系统的控制方程。
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pysindy as ps
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import sys
import argparse
import json

# Add repo root to path so the shared data_generators package is importable.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from data_generators import LorenzSystem, SpringOscillatorSystem


def generate_training_data(system, t_span=(0, 50), n_points=5000, noise_level=0.0):
    """
    生成训练数据
    
    Parameters:
    -----------
    system : DynamicalSystem
        动力学系统实例
    t_span : tuple
        时间范围
    n_points : int
        采样点数
    noise_level : float
        噪声水平
        
    Returns:
    --------
    t : ndarray
        时间数组
    x : ndarray
        状态数据
    """
    print(f"\n{'='*60}")
    print(f"生成训练数据")
    print(f"{'='*60}")
    print(f"系统: {system}")
    print(f"时间范围: {t_span}")
    print(f"采样点数: {n_points}")
    print(f"噪声水平: {noise_level}")
    print(f"\n系统方程:")
    print(system.get_equations_text())
    
    t, x = system.generate_data(
        t_span=t_span,
        n_points=n_points,
        noise_level=noise_level
    )
    
    print(f"数据形状: {x.shape}")
    print(f"时间范围: [{t[0]:.2f}, {t[-1]:.2f}]")
    print(f"数据统计:")
    for i in range(x.shape[1]):
        print(f"  维度 {i}: mean={np.mean(x[:, i]):.4f}, "
              f"std={np.std(x[:, i]):.4f}, "
              f"min={np.min(x[:, i]):.4f}, "
              f"max={np.max(x[:, i]):.4f}")
    
    return t, x


def train_sindy_model(x, t, feature_names=None, poly_order=3, threshold=0.01):
    """
    训练 SINDy 模型
    
    Parameters:
    -----------
    x : ndarray
        状态数据，shape (n_points, dim)
    t : ndarray
        时间数组，shape (n_points,)
    feature_names : list, optional
        特征名称列表
    poly_order : int
        多项式库的阶数
    threshold : float
        稀疏化阈值
        
    Returns:
    --------
    model : SINDy
        训练好的 SINDy 模型
    """
    print(f"\n{'='*60}")
    print(f"训练 SINDy 模型")
    print(f"{'='*60}")
    print(f"多项式阶数: {poly_order}")
    print(f"稀疏化阈值: {threshold}")
    
    # 计算时间步长
    dt = t[1] - t[0]
    print(f"时间步长: {dt:.6f}")
    
    # 如果未提供特征名称，自动生成
    if feature_names is None:
        dim = x.shape[1]
        feature_names = [f'x{i}' for i in range(dim)]
    
    print(f"特征名称: {feature_names}")
    
    # 创建特征库（多项式）
    feature_library = ps.PolynomialLibrary(degree=poly_order)
    
    # 创建优化器（STLSQ - Sequential Thresholded Least Squares）
    optimizer = ps.STLSQ(threshold=threshold)
    
    # 创建 SINDy 模型
    model = ps.SINDy(
        feature_library=feature_library,
        optimizer=optimizer,
        discrete_time=True
    )
    
    # 训练模型
    print("\n开始训练...")
    model.fit(x, t=dt, x_dot=None, feature_names=feature_names)
    print("训练完成！")
    
    # 打印发现的方程
    print(f"\n{'='*60}")
    print("发现的动力学方程:")
    print(f"{'='*60}")
    model.print()
    
    # 获取系数
    coefficients = model.coefficients()
    feature_names_lib = model.get_feature_names()
    
    print(f"\n特征数量: {len(feature_names_lib)}")
    print(f"非零系数数量: {np.count_nonzero(coefficients)}")
    
    return model


def test_model(model, system, t_span=(0, 50), n_points=5, 
               initial_conditions=None):
    """
    测试模型性能
    
    Parameters:
    -----------
    model : SINDy
        训练好的模型
    system : DynamicalSystem
        动力学系统（用于生成真实数据）
    t_span : tuple
        测试时间范围
    n_points : int
        测试点数
    initial_conditions : array-like
        初始条件
        
    Returns:
    --------
    t : ndarray
        时间数组
    x_true : ndarray
        真实数据
    x_pred : ndarray
        预测数据
    metrics : dict
        评估指标
    """
    print(f"\n{'='*60}")
    print(f"测试模型")
    print(f"{'='*60}")
    
    # 生成真实测试数据
    t, x_true = system.generate_data(
        t_span=t_span,
        n_points=5,
        initial_conditions=initial_conditions
    )
    
    # 使用初始条件进行预测
    if initial_conditions is None:
        initial_conditions = x_true[0]
    
    print(f"初始条件: {initial_conditions}")
    print(f"预测时间范围: {t_span}")
    
    # 使用模型进行预测
    x_pred = model.simulate(initial_conditions, 5)

    # 计算误差指标
    mse = mean_squared_error(x_true, x_pred)
    r2 = r2_score(x_true, x_pred)
    
    # 逐维度计算指标
    dim_metrics = []
    for i in range(x_true.shape[1]):
        dim_mse = mean_squared_error(x_true[:, i], x_pred[:, i])
        dim_r2 = r2_score(x_true[:, i], x_pred[:, i])
        dim_metrics.append({
            'mse': dim_mse,
            'r2': dim_r2
        })
    
    metrics = {
        'overall_mse': mse,
        'overall_r2': r2,
        'dimension_metrics': dim_metrics
    }
    
    print(f"\n性能指标:")
    print(f"  总体 MSE: {mse:.6f}")
    print(f"  总体 R²: {r2:.6f}")
    print(f"\n各维度指标:")
    for i, dm in enumerate(dim_metrics):
        print(f"  维度 {i}: MSE={dm['mse']:.6f}, R²={dm['r2']:.6f}")
    
    return t, x_true, x_pred, metrics


def visualize_results(t, x_true, x_pred, system_name="System"):
    """
    可视化结果
    
    Parameters:
    -----------
    t : ndarray
        时间数组
    x_true : ndarray
        真实数据
    x_pred : ndarray
        预测数据
    system_name : str
        系统名称
    """
    print(f"\n{'='*60}")
    print(f"生成可视化")
    print(f"{'='*60}")
    
    dim = x_true.shape[1]
    
    # 创建图形
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 3D trajectory comparison (for 3D systems)
    if dim == 3:
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.plot(x_true[:, 0], x_true[:, 1], x_true[:, 2], 
                'b-', label='True', alpha=0.6, linewidth=1)
        ax1.plot(x_pred[:, 0], x_pred[:, 1], x_pred[:, 2], 
                'r--', label='Predicted', alpha=0.6, linewidth=1)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'{system_name} - 3D Trajectory Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2-4. Time series comparison for each dimension
    for i in range(min(dim, 3)):
        ax = fig.add_subplot(2, 3, i + 2)
        ax.plot(t, x_true[:, i], 'b-', label='True', alpha=0.7, linewidth=1.5)
        ax.plot(t, x_pred[:, i], 'r--', label='Predicted', alpha=0.7, linewidth=1.5)
        ax.set_xlabel('Time')
        ax.set_ylabel(f'Dimension {i}')
        ax.set_title(f'Dimension {i} - Time Series Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 5-6. Error analysis
    errors = x_pred - x_true
    
    # 5. Error time series
    ax5 = fig.add_subplot(2, 3, 5)
    for i in range(dim):
        ax5.plot(t, errors[:, i], label=f'Dim {i}', alpha=0.7)
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Prediction Error')
    ax5.set_title('Prediction Error vs Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # 6. Error distribution histogram
    ax6 = fig.add_subplot(2, 3, 6)
    for i in range(dim):
        ax6.hist(errors[:, i], bins=50, alpha=0.5, label=f'Dim {i}')
    ax6.set_xlabel('Error')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Error Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图形
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'sindy_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图形已保存到: {output_path}")
    
    plt.show()


def save_model(model, filename='sindy_model.pkl'):
    """
    保存训练好的模型
    
    Parameters:
    -----------
    model : SINDy
        训练好的模型
    filename : str
        保存文件名
    """
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n模型已保存到: {filepath}")


def load_config(config_path):
    """
    从 JSON 文件加载配置
    
    Parameters:
    -----------
    config_path : str
        配置文件路径
        
    Returns:
    --------
    config : dict
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"\n从配置文件加载: {config_path}")
    print(f"系统类型: {config.get('system_type', 'unknown')}")
    print(f"描述: {config.get('description', 'N/A')}")
    
    return config


def create_system(system_type, **kwargs):
    """
    创建动力学系统
    
    Parameters:
    -----------
    system_type : str
        系统类型 ('lorenz' 或 'spring')
    **kwargs : dict
        系统参数
        
    Returns:
    --------
    system : DynamicalSystem
        动力学系统实例
    """
    if system_type.lower() == 'lorenz':
        return LorenzSystem(**kwargs)
    elif system_type.lower() == 'spring':
        return SpringOscillatorSystem(**kwargs)
    else:
        raise ValueError(f"Unknown system type: {system_type}")


def main(system_type='lorenz', t_span=(0, 50), n_points=5000, 
         noise_level=0.0, poly_order=3, threshold=0.01, 
         system_params=None, seed=42):
    """
    主程序
    
    Parameters:
    -----------
    system_type : str
        系统类型 ('lorenz' 或 'spring')
    t_span : tuple
        时间范围
    n_points : int
        采样点数
    noise_level : float
        噪声水平
    poly_order : int
        多项式库阶数
    threshold : float
        稀疏化阈值
    system_params : dict, optional
        系统特定参数
    seed : int
        随机种子
        
    Returns:
    --------
    model : SINDy
        训练好的模型
    metrics : dict
        性能指标
    """
    print("="*60)
    print("PySINDy 动力学模型训练与测试")
    print("="*60)
    print(f"系统类型: {system_type}")
    
    # 设置随机种子以保证可重复性
    np.random.seed(seed)
    
    # 1. 创建动力学系统
    if system_params is None:
        system_params = {}
    
    system = create_system(system_type, **system_params)
    
    # 2. 生成训练数据
    t_train, x_train = generate_training_data(
        system,
        t_span=t_span,
        n_points=n_points,
        noise_level=noise_level
    )
    
    # 3. 训练 SINDy 模型
    model = train_sindy_model(
        x_train,
        t_train,
        poly_order=poly_order,
        threshold=threshold
    )
    
    # 4. 测试模型
    t_test, x_true, x_pred, metrics = test_model(
        model,
        system,
        t_span=t_span,
        n_points=n_points
    )
    
    # 5. 可视化结果
    visualize_results(t_test, x_true, x_pred, system.name)
    
    # 6. 保存模型
    model_filename = f'sindy_model_{system_type}.pkl'
    save_model(model, filename=model_filename)
    
    print(f"\n{'='*60}")
    print("程序执行完成！")
    print(f"{'='*60}")
    
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PySINDy 动力学模型训练与测试')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径 (JSON 格式)')
    
    args = parser.parse_args()
    
    # 从配置文件加载参数
    config = load_config(args.config)
    
    # 从配置文件提取参数
    system_type = config['system_type']
    system_params = config.get('system_params', {})
    
    # 如果是 Spring 系统，转换 groups 的键从字符串到整数
    if system_type == 'spring' and 'groups' in system_params:
        groups = system_params['groups']
        system_params['groups'] = {int(k): v for k, v in groups.items()}
    
    # 训练参数
    training_config = config.get('training', {})
    t_span = tuple(training_config.get('t_span', [0, 50]))
    n_points = training_config.get('n_points', 5000)
    noise_level = training_config.get('noise_level', 0.0)
    
    # SINDy 参数
    sindy_config = config.get('sindy', {})
    poly_order = sindy_config.get('poly_order', 3)
    threshold = sindy_config.get('threshold', 0.01)
    
    # 随机种子
    seed = config.get('seed', 42)
    
    # 运行主程序
    model, metrics = main(
        system_type=system_type,
        t_span=t_span,
        n_points=n_points,
        noise_level=noise_level,
        poly_order=poly_order,
        threshold=threshold,
        system_params=system_params,
        seed=seed
    )
