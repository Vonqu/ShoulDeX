#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强损失函数模块
可以轻松集成到现有的LSTM训练脚本中

使用方法:
from enhanced_loss_module import EnhancedLossFunction

# 在训练循环中
enhanced_loss = EnhancedLossFunction(alpha=1.0, beta=0.1, gamma=0.05, delta=0.02)
loss_dict = enhanced_loss.compute_loss(pred, target, sensor_input)
total_loss = loss_dict['total_loss']
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedLossFunction:
    """
    增强损失函数类
    组合：MSE + 趋势预测损失 + 迟滞惩罚 + 平滑正则化
    """
    
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.05, delta=0.02, 
                 hysteresis_threshold=0.03):
        """
        初始化增强损失函数
        
        参数:
        - alpha: MSE主损失权重 (默认1.0)
        - beta: 趋势预测损失权重 (默认0.1)  
        - gamma: 迟滞惩罚权重 (默认0.05)
        - delta: 平滑正则化权重 (默认0.02)
        - hysteresis_threshold: 迟滞惩罚阈值 (默认0.03)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.hysteresis_threshold = hysteresis_threshold
        self.mse_loss = nn.MSELoss()
        
        print(f"初始化增强损失函数:")
        print(f"  MSE权重(α): {alpha}")
        print(f"  趋势损失权重(β): {beta}")
        print(f"  迟滞惩罚权重(γ): {gamma}")
        print(f"  平滑正则权重(δ): {delta}")
        print(f"  迟滞阈值: {hysteresis_threshold}")
        
    def trend_loss(self, pred, target):
        """
        趋势预测损失：使用一阶差分比较角度学习误差
        
        这个损失函数关注预测和真实值之间的变化趋势是否一致，
        即使绝对值有差异，如果变化趋势相同，则损失较小。
        
        公式：MSE(Δpred, Δtarget) 其中 Δ = x[t+1] - x[t]
        """
        if pred.size(0) <= 1:  # 批次大小小于等于1时直接返回0
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
            
        pred_diff = pred[1:] - pred[:-1]    # 预测的一阶差分
        target_diff = target[1:] - target[:-1]  # 真实值的一阶差分
        return self.mse_loss(pred_diff, target_diff)
    
    def hysteresis_penalty(self, pred, sensor_input):
        """
        迟滞惩罚：当输入变化很小但预测结果变化很大时进行惩罚
        
        这个损失函数防止模型在传感器输入相对稳定时产生剧烈的预测变化，
        提高了模型的稳定性和鲁棒性。
        
        逻辑：
        1. 计算传感器输入的变化幅度
        2. 计算预测输出的变化幅度  
        3. 当输入变化小而输出变化大时施加惩罚
        """
        if pred.size(0) <= 1 or sensor_input is None:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
            
        # 计算输入变化幅度（传感器数据的L2范数变化）
        input_change = torch.norm(sensor_input[1:] - sensor_input[:-1], dim=-1)
        
        # 计算预测变化幅度（角度预测的L2范数变化）
        pred_change = torch.norm(pred[1:] - pred[:-1], dim=-1)
        
        # 识别小输入变化但大预测变化的情况
        small_input_mask = input_change < self.hysteresis_threshold
        large_pred_change = pred_change > self.hysteresis_threshold * 2
        
        # 应用惩罚掩码
        penalty_mask = small_input_mask & large_pred_change
        
        if penalty_mask.any():
            penalty = torch.mean(pred_change[penalty_mask])
        else:
            penalty = torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        return penalty
    
    def smoothness_loss(self, pred):
        """
        平滑正则化：惩罚预测结果的剧烈变化
        
        这个损失函数鼓励模型产生平滑的预测序列，
        避免相邻时间步之间的剧烈跳跃。
        
        使用L1范数（绝对值）而不是L2范数，因为L1对异常值更鲁棒
        """
        if pred.size(0) <= 1:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
            
        diff = pred[1:] - pred[:-1]
        return torch.mean(torch.abs(diff))  # L1范数，也可以用 .pow(2).mean() 使用L2范数
    
    def compute_loss(self, pred, target, sensor_input=None, return_components=True):
        """
        计算组合损失函数
        
        参数:
        - pred: 模型预测值 [batch_size, output_dim]
        - target: 真实标签值 [batch_size, output_dim] 
        - sensor_input: 传感器输入 [batch_size, input_dim]，用于迟滞惩罚
        - return_components: 是否返回各组件损失详情
        
        返回:
        - 如果return_components=True: 返回字典包含所有损失组件
        - 如果return_components=False: 只返回总损失值
        """
        # 主损失：MSE拟合
        mse_loss = self.mse_loss(pred, target)
        
        # 趋势预测损失
        trend_loss_val = self.trend_loss(pred, target)
        
        # 平滑正则化
        smooth_loss_val = self.smoothness_loss(pred)
        
        # 迟滞惩罚（如果提供了传感器输入）
        hysteresis_loss_val = torch.tensor(0.0, device=pred.device, requires_grad=True)
        if sensor_input is not None:
            hysteresis_loss_val = self.hysteresis_penalty(pred, sensor_input)
        
        # 组合损失
        total_loss = (self.alpha * mse_loss + 
                     self.beta * trend_loss_val + 
                     self.gamma * hysteresis_loss_val + 
                     self.delta * smooth_loss_val)
        
        if return_components:
            return {
                'total_loss': total_loss,
                'mse_loss': mse_loss,
                'trend_loss': trend_loss_val,
                'hysteresis_loss': hysteresis_loss_val,
                'smoothness_loss': smooth_loss_val
            }
        else:
            return total_loss
    
    def update_weights(self, alpha=None, beta=None, gamma=None, delta=None):
        """
        动态更新损失函数权重
        
        这允许在训练过程中调整不同损失组件的重要性
        """
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        if delta is not None:
            self.delta = delta
            
        print(f"更新损失权重: α={self.alpha}, β={self.beta}, γ={self.gamma}, δ={self.delta}")


def get_recommended_weights(task_type="general"):
    """
    根据任务类型获得推荐的损失函数权重
    
    参数:
    - task_type: 任务类型
      - "general": 通用设置，平衡所有损失
      - "smooth": 注重平滑性，增加平滑正则化权重
      - "trend": 注重趋势跟踪，增加趋势损失权重
      - "stable": 注重稳定性，增加迟滞惩罚权重
    
    返回:
    - weights: 权重字典
    """
    weight_configs = {
        "general": {
            "alpha": 1.0,   # MSE主损失
            "beta": 0.1,    # 趋势损失
            "gamma": 0.05,  # 迟滞惩罚
            "delta": 0.02   # 平滑正则
        },
        "smooth": {
            "alpha": 1.0,
            "beta": 0.08,
            "gamma": 0.03,
            "delta": 0.15   # 增加平滑权重
        },
        "trend": {
            "alpha": 1.0,
            "beta": 0.25,   # 增加趋势权重
            "gamma": 0.05,
            "delta": 0.02
        },
        "stable": {
            "alpha": 1.0,
            "beta": 0.1,
            "gamma": 0.15,  # 增加稳定性权重
            "delta": 0.05
        }
    }
    
    if task_type not in weight_configs:
        print(f"警告：未知任务类型 '{task_type}'，使用通用设置")
        task_type = "general"
    
    weights = weight_configs[task_type]
    print(f"使用 '{task_type}' 任务的推荐权重: {weights}")
    return weights


# 示例使用方法
if __name__ == "__main__":
    # 创建示例数据
    batch_size, seq_len, input_dim, output_dim = 32, 10, 14, 27
    
    pred = torch.randn(batch_size, output_dim, requires_grad=True)
    target = torch.randn(batch_size, output_dim)
    sensor_input = torch.randn(batch_size, input_dim)
    
    # 使用通用设置
    enhanced_loss = EnhancedLossFunction()
    loss_dict = enhanced_loss.compute_loss(pred, target, sensor_input)
    
    print("\n损失组件:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.6f}")
    
    # 使用推荐权重
    weights = get_recommended_weights("general")
    enhanced_loss_smooth = EnhancedLossFunction(**weights)
    
    print("\n使用平滑任务权重:")
    loss_dict_smooth = enhanced_loss_smooth.compute_loss(pred, target, sensor_input)
    for key, value in loss_dict_smooth.items():
        print(f"  {key}: {value.item():.6f}") 