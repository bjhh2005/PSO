import numpy as np
from typing import Tuple, Dict, Callable
import copy

class ParticleSwarmOptimization:
    """粒子群优化算法实现"""
    
    def __init__(self, objective_func: Callable, n_dims: int, n_particles: int, 
                 max_iter: int, bounds: Tuple, w: float = 0.7, 
                 c1: float = 2.0, c2: float = 2.0, seed: int = None):
        """
        初始化PSO算法
        
        参数:
        objective_func: 目标函数
        n_dims: 问题维度
        n_particles: 粒子数量
        max_iter: 最大迭代次数
        bounds: 搜索边界 (min, max)
        w: 惯性权重
        c1: 个体学习因子
        c2: 社会学习因子
        seed: 随机种子
        """
        self.objective_func = objective_func
        self.n_dims = n_dims
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.bounds = bounds
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        if seed is not None:
            np.random.seed(seed)
        
        # 初始化粒子位置和速度
        self.positions = np.random.uniform(bounds[0], bounds[1], 
                                         (n_particles, n_dims))
        self.velocities = np.random.uniform(-1, 1, (n_particles, n_dims))
        
        # 初始化个体最优
        self.personal_best_positions = self.positions.copy()
        self.personal_best_fitness = np.array([objective_func(pos) 
                                             for pos in self.positions])
        
        # 初始化全局最优
        self.global_best_index = np.argmin(self.personal_best_fitness)
        self.global_best_position = self.personal_best_positions[
            self.global_best_index].copy()
        self.global_best_fitness = self.personal_best_fitness[
            self.global_best_index]
        
        # 记录历史
        self.history = {
            'global_best_fitness': [],
            'global_best_position': [],
            'mean_fitness': [],
            'std_fitness': []
        }
    
    def update_velocity(self, particle_idx: int):
        """更新粒子速度"""
        r1, r2 = np.random.random(2)
        
        cognitive_component = (self.c1 * r1 * 
                             (self.personal_best_positions[particle_idx] - 
                              self.positions[particle_idx]))
        
        social_component = (self.c2 * r2 * 
                          (self.global_best_position - 
                           self.positions[particle_idx]))
        
        self.velocities[particle_idx] = (self.w * self.velocities[particle_idx] + 
                                        cognitive_component + social_component)
    
    def update_position(self, particle_idx: int):
        """更新粒子位置"""
        self.positions[particle_idx] += self.velocities[particle_idx]
        
        # 边界处理
        self.positions[particle_idx] = np.clip(self.positions[particle_idx],
                                             self.bounds[0], self.bounds[1])
    
    def evaluate_particle(self, particle_idx: int):
        """评估粒子并更新最优位置"""
        fitness = self.objective_func(self.positions[particle_idx])
        
        if fitness < self.personal_best_fitness[particle_idx]:
            self.personal_best_positions[particle_idx] = self.positions[
                particle_idx].copy()
            self.personal_best_fitness[particle_idx] = fitness
            
            if fitness < self.global_best_fitness:
                self.global_best_position = self.positions[particle_idx].copy()
                self.global_best_fitness = fitness
    
    def optimize(self) -> Dict:
        """执行优化过程"""
        for iteration in range(self.max_iter):
            for i in range(self.n_particles):
                self.update_velocity(i)
                self.update_position(i)
                self.evaluate_particle(i)
            
            # 记录历史
            self.history['global_best_fitness'].append(self.global_best_fitness)
            self.history['global_best_position'].append(
                self.global_best_position.copy())
            self.history['mean_fitness'].append(np.mean(self.personal_best_fitness))
            self.history['std_fitness'].append(np.std(self.personal_best_fitness))
            
            # 打印进度
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Best Fitness = {self.global_best_fitness:.6f}")
        
        return {
            'best_position': self.global_best_position,
            'best_fitness': self.global_best_fitness,
            'history': self.history,
            'parameters': {
                'w': self.w,
                'c1': self.c1,
                'c2': self.c2,
                'n_particles': self.n_particles,
                'max_iter': self.max_iter
            }
        }