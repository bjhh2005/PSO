import numpy as np
from typing import Tuple, Dict, Callable
import copy

class ParticleSwarmOptimization:
    """粒子群优化算法实现"""
    
    def __init__(self, objective_func: Callable, n_dims: int, n_particles: int, 
                 max_iter: int, bounds: Tuple, w: float = 0.7, 
                 c1: float = 2.0, c2: float = 2.0, seed: int = None,
                 use_adaptive_w: bool = True,  # 新增：自适应惯性权重
                 use_clamping: bool = True     # 新增：速度钳制
                 ):
        """
        初始化PSO算法
        """
        self.objective_func = objective_func
        self.n_dims = n_dims
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.bounds = bounds
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.use_adaptive_w = use_adaptive_w
        self.use_clamping = use_clamping
        
        if seed is not None:
            np.random.seed(seed)
        
        # 计算速度限制（搜索范围的10%）
        self.v_max = 0.1 * (bounds[1] - bounds[0])
        self.v_min = -self.v_max
        
        # 初始化粒子位置和速度
        self.positions = np.random.uniform(bounds[0], bounds[1], 
                                         (n_particles, n_dims))
        self.velocities = np.random.uniform(self.v_min, self.v_max, 
                                          (n_particles, n_dims))
        
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
    
    def get_adaptive_inertia(self, iteration: int) -> float:
        """自适应惯性权重"""
        if not self.use_adaptive_w:
            return self.w
        
        # 线性递减惯性权重
        w_max = 0.9
        w_min = 0.4
        return w_max - (w_max - w_min) * (iteration / self.max_iter)
    
    def update_velocity(self, particle_idx: int, iteration: int):
        """更新粒子速度"""
        r1, r2 = np.random.random(2)
        
        # 使用自适应惯性权重
        current_w = self.get_adaptive_inertia(iteration)
        
        cognitive_component = (self.c1 * r1 * 
                             (self.personal_best_positions[particle_idx] - 
                              self.positions[particle_idx]))
        
        social_component = (self.c2 * r2 * 
                          (self.global_best_position - 
                           self.positions[particle_idx]))
        
        self.velocities[particle_idx] = (current_w * self.velocities[particle_idx] + 
                                        cognitive_component + social_component)
        
        # 速度钳制
        if self.use_clamping:
            self.velocities[particle_idx] = np.clip(self.velocities[particle_idx],
                                                  self.v_min, self.v_max)
    
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
                self.update_velocity(i, iteration)
                self.update_position(i)
                self.evaluate_particle(i)
            
            # 记录历史
            self.history['global_best_fitness'].append(self.global_best_fitness)
            self.history['global_best_position'].append(
                self.global_best_position.copy())
            self.history['mean_fitness'].append(np.mean(self.personal_best_fitness))
            self.history['std_fitness'].append(np.std(self.personal_best_fitness))
            
            # 提前终止条件
            if self.global_best_fitness < 1e-10:
                break
            
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