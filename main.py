import numpy as np
import pandas as pd
import os
from datetime import datetime
from algorithms.pso import ParticleSwarmOptimization
from utils.objective_functions import ackley_function
from utils.visualization import (plot_convergence, plot_parameter_sensitivity, 
                               plot_comparison)
from config import PSOCONFIG, ACKLEYCONFIG

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

def single_run_experiment():
    """单次运行实验"""
    print("=== 单次PSO运行实验 ===")
    
    pso = ParticleSwarmOptimization(
        objective_func=ackley_function,
        n_dims=ACKLEYCONFIG.DIMENSION,
        n_particles=PSOCONFIG.DEFAULT_PARAMS['n_particles'],
        max_iter=PSOCONFIG.DEFAULT_PARAMS['max_iter'],
        bounds=ACKLEYCONFIG.BOUNDS,
        w=PSOCONFIG.DEFAULT_PARAMS['w'],
        c1=PSOCONFIG.DEFAULT_PARAMS['c1'],
        c2=PSOCONFIG.DEFAULT_PARAMS['c2'],
        seed=42
    )
    
    result = pso.optimize()
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 绘制收敛曲线
    plot_convergence(
        result['history'],
        f'results/convergence_plots/convergence_{timestamp}.png'
    )
    
    # 保存最优解
    with open(f'results/best_solutions/best_solution_{timestamp}.txt', 'w') as f:
        f.write(f"Best Fitness: {result['best_fitness']}\n")
        f.write(f"Best Position: {result['best_position']}\n")
        f.write(f"Parameters: {result['parameters']}\n")
    
    print(f"最优适应度: {result['best_fitness']}")
    print(f"理论最优适应度: {ACKLEYCONFIG.GLOBAL_OPTIMUM}")
    
    return result

def parameter_sensitivity_analysis():
    """参数敏感性分析"""
    print("\n=== 参数敏感性分析 ===")
    
    results = []
    
    # 测试不同惯性权重
    print("测试惯性权重...")
    for w in PSOCONFIG.PARAM_RANGES['w']:
        for run in range(3):  # 每个参数运行3次取平均
            pso = ParticleSwarmOptimization(
                objective_func=ackley_function,
                n_dims=ACKLEYCONFIG.DIMENSION,
                n_particles=30,  # 减少粒子数以加快测试
                max_iter=500,    # 减少迭代次数
                bounds=ACKLEYCONFIG.BOUNDS,
                w=w,
                c1=PSOCONFIG.DEFAULT_PARAMS['c1'],
                c2=PSOCONFIG.DEFAULT_PARAMS['c2'],
                seed=run
            )
            result = pso.optimize()
            results.append({
                'w': w, 'c1': PSOCONFIG.DEFAULT_PARAMS['c1'], 
                'c2': PSOCONFIG.DEFAULT_PARAMS['c2'],
                'best_fitness': result['best_fitness'],
                'run': run
            })
    
    # 测试不同学习因子
    print("测试学习因子c1...")
    for c1 in PSOCONFIG.PARAM_RANGES['c1']:
        for run in range(3):
            pso = ParticleSwarmOptimization(
                objective_func=ackley_function,
                n_dims=ACKLEYCONFIG.DIMENSION,
                n_particles=30,
                max_iter=500,
                bounds=ACKLEYCONFIG.BOUNDS,
                w=PSOCONFIG.DEFAULT_PARAMS['w'],
                c1=c1,
                c2=PSOCONFIG.DEFAULT_PARAMS['c2'],
                seed=run
            )
            result = pso.optimize()
            results.append({
                'w': PSOCONFIG.DEFAULT_PARAMS['w'], 'c1': c1, 
                'c2': PSOCONFIG.DEFAULT_PARAMS['c2'],
                'best_fitness': result['best_fitness'],
                'run': run
            })
    
    print("测试学习因子c2...")
    for c2 in PSOCONFIG.PARAM_RANGES['c2']:
        for run in range(3):
            pso = ParticleSwarmOptimization(
                objective_func=ackley_function,
                n_dims=ACKLEYCONFIG.DIMENSION,
                n_particles=30,
                max_iter=500,
                bounds=ACKLEYCONFIG.BOUNDS,
                w=PSOCONFIG.DEFAULT_PARAMS['w'],
                c1=PSOCONFIG.DEFAULT_PARAMS['c1'],
                c2=c2,
                seed=run
            )
            result = pso.optimize()
            results.append({
                'w': PSOCONFIG.DEFAULT_PARAMS['w'], 
                'c1': PSOCONFIG.DEFAULT_PARAMS['c1'], 'c2': c2,
                'best_fitness': result['best_fitness'],
                'run': run
            })
    
    # 转换为DataFrame并保存
    df_results = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_results.to_csv(f'results/parameter_analysis/sensitivity_{timestamp}.csv', index=False)
    
    # 绘制参数敏感性图
    plot_parameter_sensitivity(
        df_results,
        f'results/parameter_analysis/sensitivity_analysis_{timestamp}.png'
    )
    
    return df_results

def comparison_experiment():
    """不同参数设置的比较实验"""
    print("\n=== 参数设置比较实验 ===")
    
    # 定义不同的参数组合
    param_combinations = {
        'Default (w=0.7, c1=2.0, c2=2.0)': 
            {'w': 0.7, 'c1': 2.0, 'c2': 2.0},
        'High Inertia (w=1.2, c1=2.0, c2=2.0)': 
            {'w': 1.2, 'c1': 2.0, 'c2': 2.0},
        'Low Inertia (w=0.4, c1=2.0, c2=2.0)': 
            {'w': 0.4, 'c1': 2.0, 'c2': 2.0},
        'Cognitive Focus (w=0.7, c1=3.0, c2=1.0)': 
            {'w': 0.7, 'c1': 3.0, 'c2': 1.0},
        'Social Focus (w=0.7, c1=1.0, c2=3.0)': 
            {'w': 0.7, 'c1': 1.0, 'c2': 3.0},
    }
    
    comparison_results = {}
    
    for label, params in param_combinations.items():
        print(f"运行: {label}")
        pso = ParticleSwarmOptimization(
            objective_func=ackley_function,
            n_dims=ACKLEYCONFIG.DIMENSION,
            n_particles=PSOCONFIG.DEFAULT_PARAMS['n_particles'],
            max_iter=PSOCONFIG.DEFAULT_PARAMS['max_iter'],
            bounds=ACKLEYCONFIG.BOUNDS,
            w=params['w'],
            c1=params['c1'],
            c2=params['c2'],
            seed=42
        )
        result = pso.optimize()
        comparison_results[label] = result['history']
        
        print(f"  {label}: {result['best_fitness']:.6f}")
    
    # 绘制比较图
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_comparison(
        comparison_results,
        f'results/convergence_plots/comparison_{timestamp}.png'
    )
    
    return comparison_results

if __name__ == "__main__":
    ensure_directories()
    
    # 运行单次实验
    single_result = single_run_experiment()
    
    # 运行参数敏感性分析
    sensitivity_results = parameter_sensitivity_analysis()
    
    # 运行比较实验
    comparison_results = comparison_experiment()
    
    print("\n=== 实验完成 ===")
    print("结果保存在以下目录:")
    print("- results/convergence_plots/: 收敛曲线图")
    print("- results/parameter_analysis/: 参数分析结果") 
    print("- results/best_solutions/: 最优解记录")