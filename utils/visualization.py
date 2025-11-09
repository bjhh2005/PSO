import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Dict, List

def plot_convergence(history: Dict, save_path: str = None):
    """
    绘制收敛曲线
    
    参数:
    history: 包含收敛历史的字典
    save_path: 图片保存路径
    """
    plt.figure(figsize=(10, 6))
    
    iterations = range(len(history['global_best_fitness']))
    plt.semilogy(iterations, history['global_best_fitness'], 
                'b-', linewidth=2, label='Global Best')
    
    plt.xlabel('Iteration')
    plt.ylabel('Fitness (log scale)')
    plt.title('PSO Convergence Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_parameter_sensitivity(results: pd.DataFrame, save_path: str = None):
    """
    绘制参数敏感性分析图
    
    参数:
    results: 包含不同参数组合结果的DataFrame
    save_path: 图片保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 惯性权重的影响
    if 'w' in results.columns:
        w_data = results.groupby('w')['best_fitness'].agg(['mean', 'std'])
        axes[0,0].errorbar(w_data.index, w_data['mean'], yerr=w_data['std'], 
                          marker='o', capsize=5)
        axes[0,0].set_xlabel('Inertia Weight (w)')
        axes[0,0].set_ylabel('Best Fitness')
        axes[0,0].set_title('Effect of Inertia Weight')
        axes[0,0].grid(True, alpha=0.3)
    
    # 学习因子c1的影响
    if 'c1' in results.columns:
        c1_data = results.groupby('c1')['best_fitness'].agg(['mean', 'std'])
        axes[0,1].errorbar(c1_data.index, c1_data['mean'], yerr=c1_data['std'],
                          marker='s', capsize=5, color='orange')
        axes[0,1].set_xlabel('Cognitive Parameter (c1)')
        axes[0,1].set_ylabel('Best Fitness')
        axes[0,1].set_title('Effect of Cognitive Parameter')
        axes[0,1].grid(True, alpha=0.3)
    
    # 学习因子c2的影响
    if 'c2' in results.columns:
        c2_data = results.groupby('c2')['best_fitness'].agg(['mean', 'std'])
        axes[1,0].errorbar(c2_data.index, c2_data['mean'], yerr=c2_data['std'],
                          marker='^', capsize=5, color='green')
        axes[1,0].set_xlabel('Social Parameter (c2)')
        axes[1,0].set_ylabel('Best Fitness')
        axes[1,0].set_title('Effect of Social Parameter')
        axes[1,0].grid(True, alpha=0.3)
    
    # 参数组合的热力图
    if all(col in results.columns for col in ['c1', 'c2']):
        pivot_table = results.pivot_table(values='best_fitness', 
                                        index='c1', columns='c2', 
                                        aggfunc='mean')
        sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='viridis',
                   ax=axes[1,1])
        axes[1,1].set_title('Parameter Combination Heatmap')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_comparison(comparison_results: Dict, save_path: str = None):
    """
    绘制不同参数设置的比较图
    
    参数:
    comparison_results: 包含不同参数设置结果的字典
    save_path: 图片保存路径
    """
    plt.figure(figsize=(12, 8))
    
    for label, history in comparison_results.items():
        iterations = range(len(history['global_best_fitness']))
        plt.semilogy(iterations, history['global_best_fitness'], 
                    linewidth=2, label=label)
    
    plt.xlabel('Iteration')
    plt.ylabel('Fitness (log scale)')
    plt.title('Comparison of Different Parameter Settings')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()