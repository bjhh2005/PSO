import itertools
import random
import numpy as np
from typing import List, Dict, Any, Generator
from config import ExperimentConfig

class ParameterGenerator:
    """参数生成器"""
    
    @staticmethod
    def grid_search(config: ExperimentConfig) -> List[Dict]:
        """网格搜索生成参数组合"""
        param_config = config.param_search_config
        param_names = ['w', 'c1', 'c2']
        param_values = [param_config[name]['values'] for name in param_names]
        
        # 生成所有可能的组合
        all_combinations = list(itertools.product(*param_values))
        
        # 限制组合数量
        if len(all_combinations) > config.param_search_config['max_combinations']:
            # 随机选择部分组合
            selected_combinations = random.sample(
                all_combinations, 
                config.param_search_config['max_combinations']
            )
        else:
            selected_combinations = all_combinations
        
        # 转换为字典格式
        param_combinations = []
        for combo in selected_combinations:
            params = {
                'w': combo[0],
                'c1': combo[1], 
                'c2': combo[2],
                'n_particles': config.n_particles,
                'max_iter': config.max_iter
            }
            param_combinations.append(params)
        
        return param_combinations
    
    @staticmethod
    def random_search(config: ExperimentConfig, n_combinations: int = None) -> List[Dict]:
        """随机搜索生成参数组合"""
        if n_combinations is None:
            n_combinations = config.param_search_config['max_combinations']
        
        param_config = config.param_search_config
        param_combinations = []
        
        for _ in range(n_combinations):
            params = {}
            for param_name in ['w', 'c1', 'c2']:
                values = param_config[param_name]['values']
                params[param_name] = random.choice(values)
            
            params.update({
                'n_particles': config.n_particles,
                'max_iter': config.max_iter
            })
            param_combinations.append(params)
        
        return param_combinations
    
    @staticmethod
    def adaptive_search(config: ExperimentConfig, previous_results: List[Dict] = None) -> List[Dict]:
        """自适应参数搜索"""
        if previous_results is None or len(previous_results) == 0:
            # 如果没有历史结果，使用随机搜索初始化
            return ParameterGenerator.random_search(config, n_combinations=10)
        
        # 基于历史结果选择表现最好的参数范围
        best_results = sorted(previous_results, 
                            key=lambda x: x['performance'])[:5]
        
        # 在最佳参数附近生成新组合
        new_combinations = []
        param_config = config.param_search_config
        
        for _ in range(config.param_search_config['max_combinations']):
            base_params = random.choice(best_results)['parameters']
            
            params = {}
            for param_name in ['w', 'c1', 'c2']:
                current_value = base_params[param_name]
                values = param_config[param_name]['values']
                
                # 在当前位置附近搜索
                idx = values.index(current_value) if current_value in values else len(values) // 2
                new_idx = max(0, min(len(values) - 1, idx + random.randint(-1, 1)))
                params[param_name] = values[new_idx]
            
            params.update({
                'n_particles': config.n_particles,
                'max_iter': config.max_iter
            })
            new_combinations.append(params)
        
        return new_combinations
    
    @staticmethod
    def generate_parameters(config: ExperimentConfig, previous_results: List[Dict] = None) -> List[Dict]:
        """生成参数组合"""
        strategy = config.param_search_config.get('search_strategy', 'grid')
        
        if strategy == 'grid':
            return ParameterGenerator.grid_search(config)
        elif strategy == 'random':
            return ParameterGenerator.random_search(config)
        elif strategy == 'adaptive':
            return ParameterGenerator.adaptive_search(config, previous_results)
        else:
            return ParameterGenerator.grid_search(config)