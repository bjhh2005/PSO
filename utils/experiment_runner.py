import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any
from algorithms.pso import ParticleSwarmOptimization
from utils.objective_functions import ackley_function
from config import ACKLEYCONFIG, ExperimentConfig

class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = []
    
    def run_single_experiment(self, params: Dict, run_id: int, seed: int = None) -> Dict:
        """运行单次实验"""
        if seed is None:
            seed = run_id
        
        try:
            pso = ParticleSwarmOptimization(
                objective_func=ackley_function,
                n_dims=self.config.n_dims,
                n_particles=params['n_particles'],
                max_iter=params['max_iter'],
                bounds=self.config.bounds,
                w=params['w'],
                c1=params['c1'],
                c2=params['c2'],
                seed=seed
            )
            
            result = pso.optimize()
            
            # 计算成功与否
            success = result['best_fitness'] <= ACKLEYCONFIG.SUCCESS_THRESHOLD
            
            experiment_result = {
                'run_id': run_id,
                'parameters': params,
                'best_fitness': result['best_fitness'],
                'best_position': result['best_position'],
                'success': success,
                'history': result['history'],
                'convergence_iteration': self._find_convergence_iteration(result['history']),
                'seed': seed
            }
            
            return experiment_result
            
        except Exception as e:
            print(f"实验运行失败: {e}")
            return None
    
    def _find_convergence_iteration(self, history: Dict) -> int:
        """找到收敛迭代次数"""
        fitness_history = history['global_best_fitness']
        
        # 检查是否收敛（连续10次迭代改进小于阈值）
        convergence_threshold = 1e-8
        window_size = 10
        
        for i in range(window_size, len(fitness_history)):
            recent_improvements = [abs(fitness_history[j] - fitness_history[j-1]) 
                                 for j in range(i-window_size+1, i+1)]
            if max(recent_improvements) < convergence_threshold:
                return i - window_size
        
        return len(fitness_history) - 1
    
    def run_parameter_sweep(self, param_combinations: List[Dict], logger = None) -> pd.DataFrame:
        """运行参数扫描"""
        all_results = []
        convergence_data = []
        
        print(f"开始参数扫描，共 {len(param_combinations)} 种参数组合")
        
        for param_idx, params in enumerate(tqdm(param_combinations, desc="参数组合")):
            param_results = []
            
            for run in range(self.config.n_runs):
                result = self.run_single_experiment(params, run_id=run, seed=param_idx * 100 + run)
                if result is not None:
                    param_results.append(result)
                    convergence_data.append(result['convergence_iteration'])
            
            if param_results:
                # 统计该参数组合的总体表现
                fitnesses = [r['best_fitness'] for r in param_results]
                successes = [r['success'] for r in param_results]
                
                param_summary = {
                    'param_set_id': param_idx,
                    'w': params['w'],
                    'c1': params['c1'],
                    'c2': params['c2'],
                    'mean_fitness': np.mean(fitnesses),
                    'std_fitness': np.std(fitnesses),
                    'median_fitness': np.median(fitnesses),
                    'min_fitness': np.min(fitnesses),
                    'max_fitness': np.max(fitnesses),
                    'success_rate': np.mean(successes),
                    'mean_convergence_iter': np.mean([r['convergence_iteration'] for r in param_results]),
                    'all_runs': param_results
                }
                
                all_results.append(param_summary)
                
                # 记录单个参数组合结果
                if logger:
                    logger.log_parameter_combination(param_idx, params, param_summary)
        
        # 计算收敛性统计
        if convergence_data:
            convergence_stats = self._calculate_convergence_stats(convergence_data)
            if logger:
                logger.log_convergence_analysis(convergence_stats)
        
        return pd.DataFrame(all_results)
    
    def _calculate_convergence_stats(self, convergence_data: List[int]) -> Dict:
        """计算收敛性统计"""
        convergence_array = np.array(convergence_data)
        
        return {
            'avg_convergence_iter': np.mean(convergence_array),
            'fastest_convergence': np.min(convergence_array),
            'slowest_convergence': np.max(convergence_array),
            'early_convergence_rate': np.mean(convergence_array < 100),
            'medium_convergence_rate': np.mean((convergence_array >= 100) & (convergence_array <= 500)),
            'late_convergence_rate': np.mean(convergence_array > 500)
        }
    
    def find_best_parameters(self, results_df: pd.DataFrame) -> Dict:
        """找到最佳参数组合"""
        metric = self.config.param_search_config['performance_metric']
        
        if metric == 'mean_fitness':
            best_idx = results_df['mean_fitness'].idxmin()
        elif metric == 'median_fitness':
            best_idx = results_df['median_fitness'].idxmin()
        elif metric == 'success_rate':
            best_idx = results_df['success_rate'].idxmax()
        else:
            best_idx = results_df['mean_fitness'].idxmin()
        
        best_result = results_df.loc[best_idx]
        
        return {
            'parameters': {
                'w': best_result['w'],
                'c1': best_result['c1'],
                'c2': best_result['c2']
            },
            'performance': {
                'mean_fitness': best_result['mean_fitness'],
                'success_rate': best_result['success_rate'],
                'std_fitness': best_result['std_fitness'],
                'convergence_iter': best_result['mean_convergence_iter']
            },
            'full_result': best_result.to_dict()
        }