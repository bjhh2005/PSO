import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple

class ResultLogger:
    """结果记录器"""
    
    def __init__(self, experiment_name: str = "pso_ackley"):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"results/{experiment_name}_{self.timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 创建日志文件
        self.log_file = open(f"{self.log_dir}/experiment_log.txt", "w", encoding="utf-8")
        self.write_header()
    
    def write_header(self):
        """写入实验头信息"""
        header = f"""
PSO Algorithm Parameter Tuning Experiment for Ackley Function
============================================================
Experiment Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Target Function: 20-dimensional Ackley Function
Global Optimum: 0.0 at x = [0, 0, ..., 0]

Experiment Configuration:
- Dimension: 20
- Search Space: [-32.768, 32.768]
- Success Threshold: 1e-3

============================================================

"""
        self.log_file.write(header)
        self.log_file.flush()
    
    def log_message(self, message: str, print_to_console: bool = True):
        """记录消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.log_file.write(formatted_message)
        self.log_file.flush()
        
        if print_to_console:
            print(message)
    
    def log_parameter_sweep_start(self, n_combinations: int, n_runs: int):
        """记录参数扫描开始"""
        message = f"""
Starting Parameter Sweep
========================
Total Parameter Combinations: {n_combinations}
Runs per Combination: {n_runs}
Total Experiments: {n_combinations * n_runs}
Estimated Completion: {n_combinations * n_runs * 2} seconds (approx)

"""
        self.log_message(message, print_to_console=True)
    
    def log_parameter_combination(self, combo_id: int, params: Dict, performance: Dict):
        """记录单个参数组合的结果"""
        message = f"Combo {combo_id:3d}: w={params['w']:.1f}, c1={params['c1']:.1f}, c2={params['c2']:.1f} -> "
        message += f"Mean Fitness: {performance['mean_fitness']:.6f}, "
        message += f"Success Rate: {performance['success_rate']:.2%}"
        
        self.log_message(message, print_to_console=False)
    
    def log_best_parameters(self, best_params: Dict):
        """记录最佳参数"""
        message = f"""
Best Parameters Found
=====================
Inertia Weight (w): {best_params['parameters']['w']:.2f}
Cognitive Parameter (c1): {best_params['parameters']['c1']:.2f}
Social Parameter (c2): {best_params['parameters']['c2']:.2f}

Performance:
- Mean Fitness: {best_params['performance']['mean_fitness']:.8f}
- Success Rate: {best_params['performance']['success_rate']:.2%}
- Standard Deviation: {best_params['performance']['std_fitness']:.8f}

"""
        self.log_message(message, print_to_console=True)
    
    def log_validation_results(self, validation_results: List[Dict]):
        """记录验证结果"""
        fitnesses = [r['best_fitness'] for r in validation_results]
        
        message = f"""
Validation Results
==================
Best Parameters Validation (5 independent runs):
"""
        
        for i, result in enumerate(validation_results):
            message += f"Run {i+1}: Fitness = {result['best_fitness']:.8f}\n"
        
        message += f"""
Statistical Summary:
- Average Fitness: {np.mean(fitnesses):.8f}
- Best Fitness: {np.min(fitnesses):.8f}
- Worst Fitness: {np.max(fitnesses):.8f}
- Standard Deviation: {np.std(fitnesses):.8f}
- Success Rate: {np.mean([f <= 1e-3 for f in fitnesses]):.2%}

"""
        self.log_message(message, print_to_console=True)
    
    def log_parameter_analysis(self, results_df: pd.DataFrame):
        """记录参数分析结果"""
        message = f"""
Parameter Sensitivity Analysis
==============================

Inertia Weight (w) Analysis:
"""
        
        w_analysis = results_df.groupby('w')['mean_fitness'].agg(['mean', 'std']).round(6)
        for w, row in w_analysis.iterrows():
            message += f"w={w:.1f}: Mean={row['mean']:.6f}, Std={row['std']:.6f}\n"
        
        message += f"""
Cognitive Parameter (c1) Analysis:
"""
        
        c1_analysis = results_df.groupby('c1')['mean_fitness'].agg(['mean', 'std']).round(6)
        for c1, row in c1_analysis.iterrows():
            message += f"c1={c1:.1f}: Mean={row['mean']:.6f}, Std={row['std']:.6f}\n"
        
        message += f"""
Social Parameter (c2) Analysis:
"""
        
        c2_analysis = results_df.groupby('c2')['mean_fitness'].agg(['mean', 'std']).round(6)
        for c2, row in c2_analysis.iterrows():
            message += f"c2={c2:.1f}: Mean={row['mean']:.6f}, Std={row['std']:.6f}\n"
        
        # 找到最佳参数范围
        best_w_range = self._find_best_range(results_df, 'w')
        best_c1_range = self._find_best_range(results_df, 'c1')
        best_c2_range = self._find_best_range(results_df, 'c2')
        
        message += f"""
Recommended Parameter Ranges:
- Inertia Weight (w): {best_w_range[0]:.1f} - {best_w_range[1]:.1f}
- Cognitive Parameter (c1): {best_c1_range[0]:.1f} - {best_c1_range[1]:.1f}
- Social Parameter (c2): {best_c2_range[0]:.1f} - {best_c2_range[1]:.1f}

"""
        self.log_message(message, print_to_console=True)
    
    def _find_best_range(self, df: pd.DataFrame, param: str, n_best: int = 5) -> Tuple[float, float]:
        """找到最佳参数范围"""
        # 修复：确保我们处理的是数值数组
        best_params_series = df.nsmallest(n_best, 'mean_fitness')[param]
        
        # 转换为numpy数组并确保是数值类型
        best_params_values = best_params_series.values
        
        # 使用Python内置的min和max函数，避免numpy的类型问题
        min_val = float(min(best_params_values))
        max_val = float(max(best_params_values))
        
        return (min_val, max_val)
    
    def log_convergence_analysis(self, convergence_data: Dict):
        """记录收敛性分析"""
        message = f"""
Convergence Analysis
====================
Average Convergence Iteration: {convergence_data['avg_convergence_iter']:.1f}
Fastest Convergence: {convergence_data['fastest_convergence']} iterations
Slowest Convergence: {convergence_data['slowest_convergence']} iterations

Convergence Patterns:
- Early Convergence (<100 iterations): {convergence_data['early_convergence_rate']:.2%}
- Medium Convergence (100-500 iterations): {convergence_data['medium_convergence_rate']:.2%}
- Late Convergence (>500 iterations): {convergence_data['late_convergence_rate']:.2%}

"""
        self.log_message(message, print_to_console=True)
    
    def save_detailed_results(self, results_df: pd.DataFrame, best_params: Dict, validation_results: List[Dict]):
        """保存详细结果到文件"""
        # 保存CSV结果
        results_df.to_csv(f"{self.log_dir}/detailed_results.csv", index=False)
        
        # 保存最佳参数
        with open(f"{self.log_dir}/best_parameters.json", "w") as f:
            json.dump(best_params, f, indent=2)
        
        # 保存验证结果
        validation_data = {
            'fitness_values': [float(r['best_fitness']) for r in validation_results],  # 确保转换为Python float
            'statistics': {
                'mean': float(np.mean([r['best_fitness'] for r in validation_results])),
                'std': float(np.std([r['best_fitness'] for r in validation_results])),
                'min': float(np.min([r['best_fitness'] for r in validation_results])),
                'max': float(np.max([r['best_fitness'] for r in validation_results]))
            }
        }
        
        with open(f"{self.log_dir}/validation_results.json", "w") as f:
            json.dump(validation_data, f, indent=2)
    
    def close(self):
        """关闭日志文件"""
        if hasattr(self, 'log_file') and not self.log_file.closed:
            self.log_file.close()
    
    def __del__(self):
        """析构函数确保文件关闭"""
        self.close()