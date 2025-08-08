#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于分段的训练数据加载器 - ShoulDex项目
功能：对每个动作段分别进行滑窗处理，避免跨段数据污染

创建时间：2025年1月
版本：1.0
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import List, Tuple, Dict
import pickle
from pathlib import Path


class SegmentBasedDataset(Dataset):
    """基于分段的数据集类"""
    
    def __init__(self, 
                 data_folder: str,
                 window_size: int = 60,
                 time_step: int = 1,
                 sensor_cols_range: Tuple[int, int] = (1, 15),
                 angle_cols_range: Tuple[int, int] = (15, 42)):
        """
        初始化分段数据集
        
        Args:
            data_folder (str): 数据文件夹路径
            window_size (int): 时间窗口大小
            time_step (int): 时间步长
            sensor_cols_range (tuple): 传感器列范围 (开始列, 结束列)
            angle_cols_range (tuple): 角度列范围 (开始列, 结束列)
        """
        self.data_folder = Path(data_folder)
        self.window_size = window_size
        self.time_step = time_step
        self.sensor_cols_range = sensor_cols_range
        self.angle_cols_range = angle_cols_range
        
        # 存储所有窗口数据
        self.windows_x = []  # 传感器数据窗口
        self.windows_y = []  # 角度数据窗口
        self.segment_info = []  # 记录每个窗口属于哪个文件
        
        # 加载并处理数据
        self._load_and_process_data()
    
    def _load_and_process_data(self):
        """加载并处理数据"""
        print(f"🔍 正在扫描文件夹: {self.data_folder}")
        
        # 获取所有CSV文件
        csv_files = list(self.data_folder.glob("*.csv"))
        csv_files.sort()  # 确保顺序一致
        
        if not csv_files:
            raise ValueError(f"在 {self.data_folder} 中未找到CSV文件")
        
        print(f"📁 找到 {len(csv_files)} 个CSV文件")
        
        total_windows = 0
        
        for file_idx, csv_file in enumerate(csv_files):
            print(f"📊 处理文件 {file_idx+1}/{len(csv_files)}: {csv_file.name}")
            
            # 读取单个CSV文件
            df = pd.read_csv(csv_file)
            
            # 提取传感器和角度数据
            sensor_data = df.iloc[:, self.sensor_cols_range[0]:self.sensor_cols_range[1]].values
            angle_data = df.iloc[:, self.angle_cols_range[0]:self.angle_cols_range[1]].values
            
            # 对当前段进行滑窗处理
            segment_windows_x, segment_windows_y = self._create_windows_for_segment(
                sensor_data, angle_data
            )
            
            # 添加到总数据集
            self.windows_x.extend(segment_windows_x)
            self.windows_y.extend(segment_windows_y)
            
            # 记录段信息
            segment_windows_count = len(segment_windows_x)
            self.segment_info.extend([{
                'file_name': csv_file.name,
                'file_index': file_idx,
                'segment_length': len(sensor_data),
                'window_index_in_segment': i
            } for i in range(segment_windows_count)])
            
            total_windows += segment_windows_count
            print(f"   ✓ 生成 {segment_windows_count} 个窗口")
        
        print(f"🎯 总共生成 {total_windows} 个训练窗口")
        print(f"📏 每个窗口大小: {self.window_size}")
        print(f"⏭️ 时间步长: {self.time_step}")
    
    def _create_windows_for_segment(self, sensor_data: np.ndarray, angle_data: np.ndarray) -> Tuple[List, List]:
        """
        对单个数据段创建滑动窗口
        
        Args:
            sensor_data (np.ndarray): 传感器数据 [时间, 传感器维度]
            angle_data (np.ndarray): 角度数据 [时间, 角度维度]
            
        Returns:
            Tuple[List, List]: (传感器窗口列表, 角度窗口列表)
        """
        windows_x = []
        windows_y = []
        
        # 计算可以生成的窗口数量
        max_start_idx = len(sensor_data) - self.window_size
        
        if max_start_idx < 0:
            print(f"   ⚠️ 警告: 数据段长度 {len(sensor_data)} 小于窗口大小 {self.window_size}，跳过该段")
            return windows_x, windows_y
        
        # 生成滑动窗口
        for start_idx in range(0, max_start_idx + 1, self.time_step):
            end_idx = start_idx + self.window_size
            
            # 提取窗口数据
            window_x = sensor_data[start_idx:end_idx]  # [window_size, sensor_dim]
            window_y = angle_data[end_idx - 1]  # 使用窗口最后一个时间点的角度作为目标
            
            windows_x.append(window_x)
            windows_y.append(window_y)
        
        return windows_x, windows_y
    
    def __len__(self):
        return len(self.windows_x)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.windows_x[idx]), torch.FloatTensor(self.windows_y[idx])
    
    def get_segment_info(self, idx):
        """获取指定索引的段信息"""
        return self.segment_info[idx]
    
    def get_all_data(self):
        """获取所有数据，用于归一化"""
        all_x = np.array(self.windows_x)  # [num_windows, window_size, sensor_dim]
        all_y = np.array(self.windows_y)  # [num_windows, angle_dim]
        return all_x, all_y


class SegmentBasedDataLoader:
    """基于分段的数据加载器管理类"""
    
    def __init__(self, 
                 train_folder: str,
                 test_folder: str,
                 window_size: int = 60,
                 time_step: int = 1,
                 sensor_cols_range: Tuple[int, int] = (1, 15),
                 angle_cols_range: Tuple[int, int] = (15, 42),
                 batch_size: int = 256,
                 validation_split: float = 0.2,
                 scaler_save_dir: str = "./predict/scaler",
                 trial_id: str = "segment_based"):
        """
        初始化数据加载器
        
        Args:
            train_folder (str): 训练数据文件夹
            test_folder (str): 测试数据文件夹
            window_size (int): 时间窗口大小
            time_step (int): 时间步长
            sensor_cols_range (tuple): 传感器列范围
            angle_cols_range (tuple): 角度列范围
            batch_size (int): 批处理大小
            validation_split (float): 验证集比例
            scaler_save_dir (str): 归一化器保存目录
            trial_id (str): 试验ID，用于保存文件名
        """
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.window_size = window_size
        self.time_step = time_step
        self.sensor_cols_range = sensor_cols_range
        self.angle_cols_range = angle_cols_range
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.scaler_save_dir = Path(scaler_save_dir)
        self.trial_id = trial_id
        
        # 创建归一化器保存目录
        self.scaler_save_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据集
        self.train_dataset = None
        self.test_dataset = None
        self.sensor_scaler = None
        self.angle_scaler = None
        
        # 数据加载器
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    def load_and_prepare_data(self):
        """加载并准备数据"""
        print("🚀 开始加载基于分段的训练数据...")
        
        # 1. 创建训练和测试数据集
        print("\n📈 加载训练数据...")
        self.train_dataset = SegmentBasedDataset(
            data_folder=self.train_folder,
            window_size=self.window_size,
            time_step=self.time_step,
            sensor_cols_range=self.sensor_cols_range,
            angle_cols_range=self.angle_cols_range
        )
        
        print("\n📊 加载测试数据...")
        self.test_dataset = SegmentBasedDataset(
            data_folder=self.test_folder,
            window_size=self.window_size,
            time_step=self.time_step,
            sensor_cols_range=self.sensor_cols_range,
            angle_cols_range=self.angle_cols_range
        )
        
        # 2. 获取所有数据用于归一化
        train_x, train_y = self.train_dataset.get_all_data()
        test_x, test_y = self.test_dataset.get_all_data()
        
        print(f"\n📏 数据形状信息:")
        print(f"   训练集: {train_x.shape} -> {train_y.shape}")
        print(f"   测试集: {test_x.shape} -> {test_y.shape}")
        
        # 3. 数据归一化
        print("\n🔧 进行数据归一化...")
        self._normalize_data(train_x, train_y, test_x, test_y)
        
        # 4. 创建数据加载器
        print("\n📦 创建数据加载器...")
        self._create_data_loaders()
        
        print("\n✅ 数据加载和预处理完成!")
        self._print_dataset_summary()
    
    def _normalize_data(self, train_x, train_y, test_x, test_y):
        """数据归一化处理"""
        # 重塑数据用于归一化 (合并时间维度)
        train_x_reshaped = train_x.reshape(-1, train_x.shape[-1])  # [num_windows * window_size, sensor_dim]
        test_x_reshaped = test_x.reshape(-1, test_x.shape[-1])
        
        # 创建归一化器
        self.sensor_scaler = MinMaxScaler()
        self.angle_scaler = MinMaxScaler()
        
        # 拟合并转换传感器数据
        train_x_normalized = self.sensor_scaler.fit_transform(train_x_reshaped)
        test_x_normalized = self.sensor_scaler.transform(test_x_reshaped)
        
        # 拟合并转换角度数据
        train_y_normalized = self.angle_scaler.fit_transform(train_y)
        test_y_normalized = self.angle_scaler.transform(test_y)
        
        # 重塑回原始形状
        train_x_normalized = train_x_normalized.reshape(train_x.shape)
        test_x_normalized = test_x_normalized.reshape(test_x.shape)
        
        # 检查归一化范围
        self._check_normalization_range()
        
        # 更新数据集
        self.train_dataset.windows_x = train_x_normalized.tolist()
        self.train_dataset.windows_y = train_y_normalized.tolist()
        self.test_dataset.windows_x = test_x_normalized.tolist()
        self.test_dataset.windows_y = test_y_normalized.tolist()
        
        # 保存归一化器
        self._save_scalers()
    
    def _check_normalization_range(self):
        """检查归一化数据范围"""
        sensor_range = self.sensor_scaler.data_max_ - self.sensor_scaler.data_min_
        angle_range = self.angle_scaler.data_max_ - self.angle_scaler.data_min_
        
        if np.any(sensor_range == 0):
            zero_cols = np.where(sensor_range == 0)[0]
            print(f"   ⚠️ 警告: 传感器数据第 {zero_cols} 列取值范围为0")
        
        if np.any(angle_range == 0):
            zero_cols = np.where(angle_range == 0)[0]
            print(f"   ⚠️ 警告: 角度数据第 {zero_cols} 列取值范围为0")
    
    def _save_scalers(self):
        """保存归一化器"""
        sensor_scaler_path = self.scaler_save_dir / f"sensor_scaler_{self.trial_id}.pkl"
        angle_scaler_path = self.scaler_save_dir / f"angle_scaler_{self.trial_id}.pkl"
        
        with open(sensor_scaler_path, 'wb') as f:
            pickle.dump(self.sensor_scaler, f)
        
        with open(angle_scaler_path, 'wb') as f:
            pickle.dump(self.angle_scaler, f)
        
        print(f"   ✓ 传感器归一化器已保存: {sensor_scaler_path}")
        print(f"   ✓ 角度归一化器已保存: {angle_scaler_path}")
    
    def _create_data_loaders(self):
        """创建PyTorch数据加载器"""
        # 训练集划分（训练/验证）
        train_size = len(self.train_dataset)
        val_size = int(train_size * self.validation_split)
        train_size = train_size - val_size
        
        train_subset, val_subset = torch.utils.data.random_split(
            self.train_dataset, [train_size, val_size]
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_subset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        self.val_loader = DataLoader(
            val_subset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
    
    def _print_dataset_summary(self):
        """打印数据集摘要信息"""
        print(f"\n📋 数据集摘要:")
        print(f"   训练窗口数: {len(self.train_loader.dataset)}")
        print(f"   验证窗口数: {len(self.val_loader.dataset)}")
        print(f"   测试窗口数: {len(self.test_dataset)}")
        print(f"   批处理大小: {self.batch_size}")
        print(f"   传感器维度: {self.sensor_cols_range[1] - self.sensor_cols_range[0]}")
        print(f"   角度维度: {self.angle_cols_range[1] - self.angle_cols_range[0]}")
        print(f"   时间窗口大小: {self.window_size}")
        print(f"   时间步长: {self.time_step}")
    
    def get_data_loaders(self):
        """获取数据加载器"""
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_scalers(self):
        """获取归一化器"""
        return self.sensor_scaler, self.angle_scaler
    
    def get_datasets(self):
        """获取原始数据集"""
        return self.train_dataset, self.test_dataset


# 使用示例
def example_usage():
    """使用示例"""
    
    # 配置参数
    config = {
        'train_folder': './data/model_training/train',
        'test_folder': './data/model_training/test', 
        'window_size': 60,
        'time_step': 3,
        'batch_size': 256,
        'validation_split': 0.2,
        'trial_id': 'segment_based_example'
    }
    
    # 创建数据加载器
    data_loader = SegmentBasedDataLoader(**config)
    
    # 加载和准备数据
    data_loader.load_and_prepare_data()
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    
    # 获取归一化器
    sensor_scaler, angle_scaler = data_loader.get_scalers()
    
    print("\n🎯 可以开始训练模型了!")
    
    return train_loader, val_loader, test_loader, sensor_scaler, angle_scaler


if __name__ == "__main__":
    example_usage() 