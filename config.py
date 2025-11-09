import numpy as np

class PSOCONFIG:
    """PSO算法配置参数"""
    # 默认参数
    DEFAULT_PARAMS = {
        'w': 0.7,           # 惯性权重
        'c1': 2.0,          # 个体学习因子
        'c2': 2.0,          # 社会学习因子
        'n_particles': 50,  # 粒子数量
        'max_iter': 1000,   # 最大迭代次数
        'n_dims': 20,       # 问题维度
        'bounds': (-32.768, 32.768),  # 搜索范围
    }
    
    # 参数测试范围
    PARAM_RANGES = {
        'w': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        'c1': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        'c2': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    }

class ACKLEYCONFIG:
    """Ackley函数配置"""
    DIMENSION = 20
    GLOBAL_OPTIMUM = 0.0
    GLOBAL_POSITION = np.zeros(DIMENSION)
    BOUNDS = (-32.768, 32.768)