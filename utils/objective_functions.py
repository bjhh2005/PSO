import numpy as np

def ackley_function(x, n_dims=20):
    """
    计算Ackley函数值
    
    参数:
    x: 输入向量 (n_dims维)
    n_dims: 维度
    
    返回:
    float: Ackley函数值
    """
    x = np.array(x)
    if len(x) != n_dims:
        raise ValueError(f"输入维度应为{n_dims}, 实际为{len(x)}")
    
    # Ackley函数计算
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / n_dims))
    term2 = -np.exp(sum2 / n_dims)
    
    return 20 + np.e + term1 + term2

def sphere_function(x):
    """球函数 (用于对比测试)"""
    return np.sum(np.array(x)**2)

def rastrigin_function(x):
    """Rastrigin函数 (用于对比测试)"""
    x = np.array(x)
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))