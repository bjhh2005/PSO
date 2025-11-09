import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from config import ExperimentConfig, ACKLEYCONFIG
from utils.parameter_generator import ParameterGenerator
from utils.experiment_runner import ExperimentRunner
from utils.result_logger import ResultLogger
from utils.visualization import (plot_convergence, plot_parameter_sensitivity, 
                               plot_comparison, plot_optimization_history)
from typing import List, Dict

def ensure_directories():
    """确保结果目录存在"""
    directories = [
        'results',
        'results/convergence_plots',
        'results/parameter_analysis', 
        'results/best_solutions'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def run_large_scale_parameter_tuning():
    """大规模参数调优"""
    # 初始化结果记录器
    logger = ResultLogger("large_scale_pso_tuning")
    
    try:
        logger.log_message("=== 大规模PSO参数调优实验开始 ===")
        
        # 配置大规模实验 - 使用更好的参数
        config = ExperimentConfig(
            n_dims=20,
            n_particles=50,  # 增加粒子数
            max_iter=1000,   # 增加迭代次数
            n_runs=3,
            param_search_config={
                'w': {
                    'type': 'linear',
                    'values': [0.4, 0.5, 0.6, 0.7, 0.8],  # 缩小范围，集中在常用值
                    'best_value': 0.7
                },
                'c1': {
                    'type': 'linear',
                    'values': [1.5, 2.0, 2.5],  # 缩小范围
                    'best_value': 2.0
                },
                'c2': {
                    'type': 'linear', 
                    'values': [1.5, 2.0, 2.5],  # 缩小范围
                    'best_value': 2.0
                },
                'search_strategy': 'grid',  # 使用网格搜索确保覆盖
                'max_combinations': 45,     # 3×3×5=45种组合
                'performance_metric': 'mean_fitness'
            }
        )
        
        # 生成参数组合
        param_combinations = ParameterGenerator.generate_parameters(config)
        logger.log_parameter_sweep_start(len(param_combinations), config.n_runs)
        
        # 运行实验
        runner = ExperimentRunner(config)
        results_df = runner.run_parameter_sweep(param_combinations, logger)
        
        # 找到最佳参数
        best_params = runner.find_best_parameters(results_df)
        logger.log_best_parameters(best_params)
        
        # 参数分析
        logger.log_parameter_analysis(results_df)
        
        # 验证最佳参数
        validation_results = validate_best_parameters(best_params, logger)
        
        # 保存详细结果
        logger.save_detailed_results(results_df, best_params, validation_results)
        
        # 生成可视化图表
        generate_comprehensive_visualizations(results_df, best_params, validation_results, logger)
        
        logger.log_message("=== 实验完成 ===")
        logger.log_message(f"结果保存目录: {logger.log_dir}")
        
        return best_params, results_df, validation_results
        
    finally:
        logger.close()

def validate_best_parameters(best_params: dict, logger: ResultLogger):
    """验证最佳参数"""
    logger.log_message("开始验证最佳参数...")
    
    # 使用最佳参数进行完整运行，使用改进的PSO
    config = ExperimentConfig(
        n_particles=100,    # 增加粒子数
        max_iter=2000,      # 增加迭代次数
        n_runs=5
    )
    
    params = {
        'w': best_params['parameters']['w'],
        'c1': best_params['parameters']['c1'],
        'c2': best_params['parameters']['c2'],
        'n_particles': config.n_particles,
        'max_iter': config.max_iter
    }
    
    # 使用改进的PSO
    validation_results = []
    
    for run in range(config.n_runs):
        try:
            pso = ParticleSwarmOptimization(
                objective_func=ackley_function,
                n_dims=20,
                n_particles=params['n_particles'],
                max_iter=params['max_iter'],
                bounds=(-32.768, 32.768),
                w=params['w'],
                c1=params['c1'],
                c2=params['c2'],
                seed=run,
                use_adaptive_w=True,  # 启用自适应权重
                use_clamping=True     # 启用速度钳制
            )
            
            result = pso.optimize()
            
            validation_result = {
                'run_id': run,
                'parameters': params,
                'best_fitness': result['best_fitness'],
                'best_position': result['best_position'],
                'success': result['best_fitness'] <= 1e-3,
                'history': result['history'],
                'convergence_iteration': len(result['history']['global_best_fitness']) - 1,
                'seed': run
            }
            
            validation_results.append(validation_result)
            
        except Exception as e:
            print(f"验证运行 {run} 失败: {e}")
    
    logger.log_validation_results(validation_results)
    
    return validation_results
def generate_comprehensive_visualizations(results_df: pd.DataFrame, best_params: Dict, 
                                        validation_results: List[Dict], logger: ResultLogger):
    """生成综合可视化图表"""
    logger.log_message("生成可视化图表...")
    
    # 参数敏感性分析图
    plot_parameter_sensitivity(
        results_df,
        f'{logger.log_dir}/parameter_sensitivity_analysis.png'
    )
    
    # 最佳验证运行的收敛曲线
    best_validation_run = min(validation_results, key=lambda x: x['best_fitness'])
    plot_convergence(
        best_validation_run['history'],
        f'{logger.log_dir}/best_validation_convergence.png'
    )
    
    # 参数组合性能热力图
    plot_parameter_combinations_heatmap(results_df, f'{logger.log_dir}/parameter_combinations_heatmap.png')
    
    logger.log_message("可视化图表生成完成")

def plot_parameter_combinations_heatmap(results_df: pd.DataFrame, save_path: str):
    """绘制参数组合热力图"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 创建c1-c2热力图
    pivot_table = results_df.pivot_table(
        values='mean_fitness', 
        index='c1', 
        columns='c2', 
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='viridis_r', 
                cbar_kws={'label': 'Mean Fitness'})
    plt.title('PSO Parameter Combinations Performance\n(c1 vs c2, colored by mean fitness)')
    plt.xlabel('Social Parameter (c2)')
    plt.ylabel('Cognitive Parameter (c1)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_comprehensive_comparison():
    """综合比较不同参数策略"""
    logger = ResultLogger("pso_strategy_comparison")
    
    try:
        # 测试不同的参数策略
        strategies = [
            {'name': 'Balanced', 'w': 0.7, 'c1': 2.0, 'c2': 2.0},
            {'name': 'High Exploration', 'w': 1.2, 'c1': 2.5, 'c2': 0.5},
            {'name': 'High Exploitation', 'w': 0.4, 'c1': 0.5, 'c2': 2.5},
            {'name': 'Cognitive Focus', 'w': 0.7, 'c1': 3.0, 'c2': 1.0},
            {'name': 'Social Focus', 'w': 0.7, 'c1': 1.0, 'c2': 3.0},
        ]
        
        comparison_results = {}
        config = ExperimentConfig(n_particles=40, max_iter=500, n_runs=3)
        
        for strategy in strategies:
            logger.log_message(f"测试策略: {strategy['name']}")
            
            params = {
                'w': strategy['w'],
                'c1': strategy['c1'],
                'c2': strategy['c2'],
                'n_particles': config.n_particles,
                'max_iter': config.max_iter
            }
            
            runner = ExperimentRunner(config)
            strategy_results = []
            
            for run in range(config.n_runs):
                result = runner.run_single_experiment(params, run_id=run, seed=run)
                if result:
                    strategy_results.append(result)
            
            if strategy_results:
                fitnesses = [r['best_fitness'] for r in strategy_results]
                comparison_results[strategy['name']] = {
                    'mean_fitness': np.mean(fitnesses),
                    'std_fitness': np.std(fitnesses),
                    'success_rate': np.mean([f <= 1e-3 for f in fitnesses]),
                    'convergence': np.mean([r['convergence_iteration'] for r in strategy_results])
                }
        
        # 记录比较结果
        logger.log_message("\n策略比较结果:")
        for name, results in comparison_results.items():
            logger.log_message(f"{name:15}: Mean={results['mean_fitness']:.6f}, "
                             f"Success={results['success_rate']:.2%}, "
                             f"Convergence={results['convergence']:.1f} iterations")
        
        return comparison_results
        
    finally:
        logger.close()

if __name__ == "__main__":
    ensure_directories()
    
    print("开始大规模PSO参数调优实验...")
    
    # 1. 大规模参数调优
    best_params, tuning_results, validation_results = run_large_scale_parameter_tuning()
    
    # 2. 策略比较（可选）
    # comparison_results = run_comprehensive_comparison()
    
    print("\n=== 所有实验完成 ===")
    print("请查看 results/ 目录下的详细报告和可视化图表")
    
