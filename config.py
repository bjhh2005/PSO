import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

@dataclass
class ExperimentConfig:
    """实验配置"""
    # 基础参数
    n_dims: int = 20
    n_particles: int = 50
    max_iter: int = 1000
    bounds: Tuple[float, float] = (-32.768, 32.768)
    n_runs: int = 5  # 每个参数组合的运行次数
    
    # 自动参数搜索配置 - 增加测试规模
    param_search_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.param_search_config is None:
            self.param_search_config = {
                'w': {
                    'type': 'linear',
                    'values': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],  # 增加更多值
                    'best_value': 0.7
                },
                'c1': {
                    'type': 'linear',
                    'values': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],  # 增加更多值
                    'best_value': 2.0
                },
                'c2': {
                    'type': 'linear', 
                    'values': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],  # 增加更多值
                    'best_value': 2.0
                },
                'search_strategy': 'random',  # grid, random, adaptive
                'max_combinations': 100,  # 增加到100种参数组合
                'performance_metric': 'mean_fitness'  # mean_fitness, median_fitness, success_rate
            }

class ACKLEYCONFIG:
    """Ackley函数配置"""
    DIMENSION = 20
    GLOBAL_OPTIMUM = 0.0
    GLOBAL_POSITION = np.zeros(DIMENSION)
    BOUNDS = (-32.768, 32.768)
    SUCCESS_THRESHOLD = 1e-3  # 成功阈值