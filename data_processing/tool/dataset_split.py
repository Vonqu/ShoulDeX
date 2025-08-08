#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集划分工具 - ShoulDex项目
功能：将包含多个被试和动作的CSV文件按指定规则划分为训练集和测试集

作者：AI助手
创建时间：2025
版本：1.0
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import random
import shutil
from typing import List, Dict, Tuple, Optional
import argparse
import json
from datetime import datetime


class DatasetSplitter:
    """数据集划分类"""
    
    def __init__(self, 
                 input_dir: str,
                 output_dir: str = "./data/model_training",
                 train_ratio: float = 0.8,
                 random_seed: int = 42):
        """
        初始化数据集划分器
        
        Args:
            input_dir (str): 输入数据目录路径
            output_dir (str): 输出目录路径
            train_ratio (float): 训练集比例，默认0.8
            random_seed (int): 随机种子，默认42
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.test_ratio = 1.0 - train_ratio
        self.random_seed = random_seed
        
        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # 创建输出目录
        self.train_dir = self.output_dir / "train"
        self.test_dir = self.output_dir / "test"
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储划分信息
        self.split_info = {
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "train_ratio": self.train_ratio,
            "test_ratio": self.test_ratio,
            "random_seed": self.random_seed,
            "split_time": datetime.now().isoformat(),
            "train_files": [],
            "test_files": [],
            "statistics": {}
        }
    
    def scan_files(self) -> List[Path]:
        """
        扫描输入目录中的所有CSV文件
        
        Returns:
            List[Path]: CSV文件路径列表
        """
        csv_files = []
        
        # 递归搜索所有CSV文件
        for pattern in ["**/*.csv", "*.csv"]:
            csv_files.extend(list(self.input_dir.glob(pattern)))
        
        print(f"📁 在 {self.input_dir} 中找到 {len(csv_files)} 个CSV文件")
        
        # 按文件名排序，确保结果可重现
        csv_files.sort()
        
        return csv_files
    
    def parse_filename(self, filepath: Path) -> Dict[str, str]:
        """
        解析文件名，提取被试ID、动作类型等信息
        
        文件名格式示例：
        - P1_AB_fps30.csv -> 被试P1，动作AB，其他fps30
        - P2_C_RH_fps30_jianjiagu.csv -> 被试P2，动作类型C_RH，其他fps30_jianjiagu
        
        Args:
            filepath (Path): 文件路径
            
        Returns:
            Dict[str, str]: 解析结果，包含subject_id, motion_type, additional_info
        """
        filename = filepath.stem  # 不包含扩展名的文件名
        parts = filename.split('_')
        
        if len(parts) < 2:
            # 如果文件名格式不符合预期，返回默认值
            return {
                "subject_id": "unknown",
                "motion_type": "unknown", 
                "additional_info": filename,
                "filename": filename
            }
        
        subject_id = parts[0]  # 被试ID，如P1, P2, P3
        
        # 识别动作类型
        motion_type = parts[1]
        additional_info = "_".join(parts[2:]) if len(parts) > 2 else ""
        
        # 如果是补偿性动作（包含C_），合并前几个部分作为动作类型
        if len(parts) > 2 and parts[1] == 'C':
            motion_type = f"{parts[1]}_{parts[2]}"
            additional_info = "_".join(parts[3:]) if len(parts) > 3 else ""
        
        return {
            "subject_id": subject_id,
            "motion_type": motion_type,
            "additional_info": additional_info,
            "filename": filename
        }
    
    def group_files_by_subject(self, csv_files: List[Path]) -> Dict[str, List[Path]]:
        """
        按被试ID分组文件
        
        Args:
            csv_files (List[Path]): CSV文件列表
            
        Returns:
            Dict[str, List[Path]]: 按被试ID分组的文件字典
        """
        subject_groups = {}
        
        for file_path in csv_files:
            file_info = self.parse_filename(file_path)
            subject_id = file_info["subject_id"]
            
            if subject_id not in subject_groups:
                subject_groups[subject_id] = []
            
            subject_groups[subject_id].append(file_path)
        
        return subject_groups
    
    def group_files_by_motion(self, csv_files: List[Path]) -> Dict[str, List[Path]]:
        """
        按动作类型分组文件
        
        Args:
            csv_files (List[Path]): CSV文件列表
            
        Returns:
            Dict[str, List[Path]]: 按动作类型分组的文件字典
        """
        motion_groups = {}
        
        for file_path in csv_files:
            file_info = self.parse_filename(file_path)
            motion_type = file_info["motion_type"]
            
            if motion_type not in motion_groups:
                motion_groups[motion_type] = []
            
            motion_groups[motion_type].append(file_path)
        
        return motion_groups
    
    def split_by_subject(self, csv_files: List[Path]) -> Tuple[List[Path], List[Path]]:
        """
        按被试划分数据集（被试级别的划分）
        确保同一被试的所有数据要么在训练集，要么在测试集
        
        Args:
            csv_files (List[Path]): 所有CSV文件
            
        Returns:
            Tuple[List[Path], List[Path]]: (训练集文件, 测试集文件)
        """
        subject_groups = self.group_files_by_subject(csv_files)
        subjects = list(subject_groups.keys())
        
        # 随机打乱被试顺序
        random.shuffle(subjects)
        
        # 计算训练集被试数量
        num_train_subjects = int(len(subjects) * self.train_ratio)
        
        train_subjects = subjects[:num_train_subjects]
        test_subjects = subjects[num_train_subjects:]
        
        print(f"👥 总计 {len(subjects)} 个被试")
        print(f"📚 训练集被试 ({len(train_subjects)}): {train_subjects}")
        print(f"🧪 测试集被试 ({len(test_subjects)}): {test_subjects}")
        
        # 收集对应的文件
        train_files = []
        test_files = []
        
        for subject in train_subjects:
            train_files.extend(subject_groups[subject])
        
        for subject in test_subjects:
            test_files.extend(subject_groups[subject])
        
        return train_files, test_files
    
    def split_by_motion_balanced(self, csv_files: List[Path]) -> Tuple[List[Path], List[Path]]:
        """
        按动作类型平衡划分数据集
        确保每种动作类型在训练集和测试集中都有合理的分布
        
        Args:
            csv_files (List[Path]): 所有CSV文件
            
        Returns:
            Tuple[List[Path], List[Path]]: (训练集文件, 测试集文件)
        """
        motion_groups = self.group_files_by_motion(csv_files)
        
        train_files = []
        test_files = []
        
        print(f"🎯 按动作类型平衡划分:")
        
        for motion_type, motion_files in motion_groups.items():
            # 随机打乱该动作类型的文件
            motion_files_shuffled = motion_files.copy()
            random.shuffle(motion_files_shuffled)
            
            # 计算该动作类型的训练集文件数量
            num_train = int(len(motion_files_shuffled) * self.train_ratio)
            
            motion_train = motion_files_shuffled[:num_train]
            motion_test = motion_files_shuffled[num_train:]
            
            train_files.extend(motion_train)
            test_files.extend(motion_test)
            
            print(f"  📋 {motion_type}: 总计{len(motion_files)} → 训练{len(motion_train)} + 测试{len(motion_test)}")
        
        return train_files, test_files
    
    def split_random(self, csv_files: List[Path]) -> Tuple[List[Path], List[Path]]:
        """
        随机划分数据集（文件级别的随机划分）
        
        Args:
            csv_files (List[Path]): 所有CSV文件
            
        Returns:
            Tuple[List[Path], List[Path]]: (训练集文件, 测试集文件)
        """
        # 随机打乱文件顺序
        csv_files_shuffled = csv_files.copy()
        random.shuffle(csv_files_shuffled)
        
        # 计算训练集文件数量
        num_train_files = int(len(csv_files_shuffled) * self.train_ratio)
        
        train_files = csv_files_shuffled[:num_train_files]
        test_files = csv_files_shuffled[num_train_files:]
        
        print(f"🎲 随机划分: 总计{len(csv_files)} → 训练{len(train_files)} + 测试{len(test_files)}")
        
        return train_files, test_files
    
    def copy_files_to_dataset(self, train_files: List[Path], test_files: List[Path]):
        """
        将划分后的文件复制到训练集和测试集目录
        
        Args:
            train_files (List[Path]): 训练集文件列表
            test_files (List[Path]): 测试集文件列表
        """
        print(f"\n📁 开始复制文件到数据集目录...")
        
        # 清空输出目录
        if self.train_dir.exists():
            shutil.rmtree(self.train_dir)
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制训练集文件
        print(f"📚 复制训练集文件 ({len(train_files)} 个)...")
        for file_path in train_files:
            dest_path = self.train_dir / file_path.name
            shutil.copy2(file_path, dest_path)
            self.split_info["train_files"].append(str(file_path))
        
        # 复制测试集文件
        print(f"🧪 复制测试集文件 ({len(test_files)} 个)...")
        for file_path in test_files:
            dest_path = self.test_dir / file_path.name
            shutil.copy2(file_path, dest_path)
            self.split_info["test_files"].append(str(file_path))
        
        print(f"✅ 文件复制完成!")
        print(f"   训练集: {self.train_dir}")
        print(f"   测试集: {self.test_dir}")
    
    def analyze_dataset_statistics(self, train_files: List[Path], test_files: List[Path]):
        """
        分析数据集统计信息
        
        Args:
            train_files (List[Path]): 训练集文件列表
            test_files (List[Path]): 测试集文件列表
        """
        def analyze_file_group(files: List[Path], group_name: str):
            """分析文件组的统计信息"""
            subject_count = {}
            motion_count = {}
            total_samples = 0
            
            for file_path in files:
                file_info = self.parse_filename(file_path)
                subject_id = file_info["subject_id"]
                motion_type = file_info["motion_type"]
                
                # 统计被试数量
                subject_count[subject_id] = subject_count.get(subject_id, 0) + 1
                
                # 统计动作类型数量
                motion_count[motion_type] = motion_count.get(motion_type, 0) + 1
                
                # 统计总样本数（读取CSV文件行数）
                try:
                    df = pd.read_csv(file_path)
                    total_samples += len(df)
                except Exception as e:
                    print(f"⚠️ 读取文件 {file_path} 时出错: {e}")
            
            return {
                "file_count": len(files),
                "subject_count": subject_count,
                "motion_count": motion_count,
                "total_samples": total_samples,
                "unique_subjects": len(subject_count),
                "unique_motions": len(motion_count)
            }
        
        # 分析训练集和测试集
        train_stats = analyze_file_group(train_files, "训练集")
        test_stats = analyze_file_group(test_files, "测试集")
        
        # 存储统计信息
        self.split_info["statistics"] = {
            "train": train_stats,
            "test": test_stats
        }
        
        # 打印统计报告
        print(f"\n📊 数据集统计报告:")
        print(f"=" * 60)
        
        def print_stats(stats: dict, name: str):
            print(f"\n{name}:")
            print(f"  📁 文件数量: {stats['file_count']}")
            print(f"  👥 被试数量: {stats['unique_subjects']} ({list(stats['subject_count'].keys())})")
            print(f"  🎯 动作类型: {stats['unique_motions']} ({list(stats['motion_count'].keys())})")
            print(f"  📊 总样本数: {stats['total_samples']:,}")
            
            print(f"  📋 被试分布:")
            for subject, count in sorted(stats['subject_count'].items()):
                print(f"    - {subject}: {count} 文件")
            
            print(f"  📋 动作分布:")
            for motion, count in sorted(stats['motion_count'].items()):
                print(f"    - {motion}: {count} 文件")
        
        print_stats(train_stats, "📚 训练集")
        print_stats(test_stats, "🧪 测试集")
        
        print(f"\n📈 总体统计:")
        total_files = train_stats['file_count'] + test_stats['file_count']
        total_samples = train_stats['total_samples'] + test_stats['total_samples']
        print(f"  总文件数: {total_files}")
        print(f"  总样本数: {total_samples:,}")
        print(f"  训练集比例: {train_stats['file_count']/total_files:.1%} ({train_stats['total_samples']/total_samples:.1%} 样本)")
        print(f"  测试集比例: {test_stats['file_count']/total_files:.1%} ({test_stats['total_samples']/total_samples:.1%} 样本)")
    
    def save_split_info(self):
        """保存划分信息到JSON文件"""
        info_file = self.output_dir / "split_info.json"
        
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(self.split_info, f, ensure_ascii=False, indent=2)
        
        print(f"💾 划分信息已保存到: {info_file}")
    
    def split_dataset(self, split_method: str = "by_subject"):
        """
        执行数据集划分
        
        Args:
            split_method (str): 划分方法，可选：
                - "by_subject": 按被试划分
                - "by_motion": 按动作类型平衡划分  
                - "random": 随机划分
        """
        print(f"🚀 开始数据集划分...")
        print(f"   方法: {split_method}")
        print(f"   输入目录: {self.input_dir}")
        print(f"   输出目录: {self.output_dir}")
        print(f"   训练集比例: {self.train_ratio:.1%}")
        print(f"   测试集比例: {self.test_ratio:.1%}")
        print(f"   随机种子: {self.random_seed}")
        
        # 1. 扫描文件
        csv_files = self.scan_files()
        if not csv_files:
            print("❌ 没有找到CSV文件，请检查输入目录")
            return
        
        # 2. 根据方法划分
        if split_method == "by_subject":
            train_files, test_files = self.split_by_subject(csv_files)
        elif split_method == "by_motion":
            train_files, test_files = self.split_by_motion_balanced(csv_files)
        elif split_method == "random":
            train_files, test_files = self.split_random(csv_files)
        else:
            raise ValueError(f"不支持的划分方法: {split_method}")
        
        # 3. 复制文件
        self.copy_files_to_dataset(train_files, test_files)
        
        # 4. 统计分析
        self.analyze_dataset_statistics(train_files, test_files)
        
        # 5. 保存划分信息
        self.save_split_info()
        
        print(f"\n🎉 数据集划分完成!")


def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(description="ShoulDex 数据集划分工具")
    
    parser.add_argument("input_dir", 
                       help="输入数据目录路径（包含CSV文件）")
    
    parser.add_argument("-o", "--output_dir", 
                       default="./data/model_training",
                       help="输出目录路径（默认: ./data/model_training）")
    
    parser.add_argument("-r", "--train_ratio", 
                       type=float, 
                       default=0.8,
                       help="训练集比例（默认: 0.8）")
    
    parser.add_argument("-m", "--method", 
                       choices=["by_subject", "by_motion", "random"],
                       default="by_subject",
                       help="划分方法（默认: by_subject）")
    
    parser.add_argument("-s", "--seed", 
                       type=int, 
                       default=42,
                       help="随机种子（默认: 42）")
    
    args = parser.parse_args()
    
    # 创建划分器并执行
    splitter = DatasetSplitter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        random_seed=args.seed
    )
    
    splitter.split_dataset(split_method=args.method)


if __name__ == "__main__":
    main()
