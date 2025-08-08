# ShoulDex v1.5 - 肩关节运动分析系统

## 项目概述

ShoulDex是一个基于深度学习的肩关节运动分析系统，通过融合光学动作捕捉数据和可穿戴传感器数据，实现肩关节角度的实时预测和运动分析。

### 主要功能
- 📊 **数据处理**: 光学动捕数据和传感器数据的同步处理
- 🧠 **深度学习**: 基于LSTM的肩关节角度预测模型
- 📈 **实时预测**: 支持实时肩关节角度预测和可视化
- 🔧 **数据工具**: 完整的数据集划分和预处理工具链
- 📊 **运动分析**: ROM(运动范围)统计分析和可视化

## 项目架构

```
ShoulDex_v1.5/
├── data/                           # 数据目录
│   ├── raw_data/                   # 原始数据
│   ├── dft_final/                  # 处理后的最终数据
│   └── model_training/             # 模型训练数据集
│       ├── train/                  # 训练集
│       └── test/                   # 测试集
├── data_processing/                # 数据处理模块
│   ├── tool/                       # 数据处理工具
│   │   ├── dataset_split.py        # 🆕 数据集划分工具
│   │   ├── load_training_data.py   # 训练数据加载
│   │   └── rename.py               # 文件重命名工具
│   ├── angle_calculation/          # 角度计算
│   ├── rom_anylsis/                # ROM分析
│   └── *.py                        # 各种数据处理脚本
├── train/                          # 模型训练模块
│   ├── model.py                    # 模型定义
│   ├── lstm_*.py                   # LSTM训练脚本
│   └── result/                     # 训练结果
├── predict/                        # 预测模块
│   ├── predict_*.py                # 预测脚本
│   ├── scaler/                     # 数据归一化器
│   └── *.html                      # 可视化界面
├── model/                          # 训练好的模型
└── log/                            # 日志文件
```

## 数据格式说明

### CSV数据文件结构
每个CSV文件包含以下列：
- `Time_angle`: 时间戳
- `s1-s14`: 14个传感器数据
- `A_*`: 27个肩关节角度数据
  - `A_humerus_l/r_thorax_X/Y/Z`: 左/右肱骨与胸廓角度
  - `A_scapula_l/r_thorax_X/Y/Z`: 左/右肩胛骨与胸廓角度
  - `A_clavicle_l/r_thorax_X/Y/Z`: 左/右锁骨与胸廓角度
  - `A_humerus_l/r_scapula_X/Y/Z`: 左/右肱骨与肩胛骨角度
  - `A_thorax_hip_X/Y/Z`: 胸廓与髋关节角度

### 文件命名规范
- 格式: `{被试ID}_{动作类型}_{其他信息}.csv`
- 示例:
  - `P1_AB_fps30.csv`: 被试P1的外展(AB)动作，30fps
  - `P2_C_RH_fps30_jianjiagu.csv`: 被试P2的补偿性到达(C_RH)动作

## 🆕 数据集划分工具

### 功能特点
- 📁 **智能文件扫描**: 自动递归扫描目录中的所有CSV文件
- 🎯 **多种划分策略**: 
  - 按被试划分(推荐): 确保同一被试的数据不会同时出现在训练集和测试集
  - 按动作平衡划分: 确保每种动作在训练集和测试集中都有合理分布
  - 随机划分: 完全随机的文件级别划分
- 📊 **详细统计分析**: 自动分析被试分布、动作分布、样本数量等
- 💾 **划分信息记录**: 保存详细的划分信息到JSON文件
- 🔄 **可重现结果**: 通过随机种子确保结果可重现

### 使用方法

#### 1. 命令行使用
```bash
# 基本用法 - 按被试划分（推荐）
python data_processing/tool/dataset_split.py data/dft_final

# 指定输出目录和训练集比例
python data_processing/tool/dataset_split.py data/dft_final -o ./data/new_split -r 0.7

# 按动作类型平衡划分
python data_processing/tool/dataset_split.py data/dft_final -m by_motion

# 随机划分
python data_processing/tool/dataset_split.py data/dft_final -m random

# 设置随机种子
python data_processing/tool/dataset_split.py data/dft_final -s 123
```

#### 2. Python脚本使用
```python
from data_processing.tool.dataset_split import DatasetSplitter

# 创建划分器
splitter = DatasetSplitter(
    input_dir="data/dft_final",
    output_dir="./data/model_training",
    train_ratio=0.8,
    random_seed=42
)

# 执行划分
splitter.split_dataset(split_method="by_subject")
```

#### 3. 参数说明
- `input_dir`: 输入数据目录（必需）
- `-o/--output_dir`: 输出目录，默认`./data/model_training`
- `-r/--train_ratio`: 训练集比例，默认0.8
- `-m/--method`: 划分方法，可选`by_subject`/`by_motion`/`random`，默认`by_subject`
- `-s/--seed`: 随机种子，默认42

### 划分策略对比

| 策略 | 优点 | 缺点 | 推荐场景 |
|------|------|------|----------|
| **按被试划分** | 避免数据泄露，更真实的泛化性能 | 可能导致某些动作在测试集中较少 | **生产环境使用（推荐）** |
| **按动作平衡** | 每种动作都有足够的测试样本 | 可能存在数据泄露风险 | 动作识别和分类任务 |
| **随机划分** | 简单直接 | 容易造成数据泄露 | 快速原型验证 |

## 模型训练

### 支持的模型配置
- **输入**: 14个传感器数据 (s1-s14)
- **输出**: 15或27个角度数据
- **模型**: LSTM神经网络
- **序列长度**: 可配置

### 🆕 改进的分段数据加载
为了避免训练数据中不同动作段之间的人工连接问题，我们提供了基于分段的数据加载器：

#### 核心优势
- 📊 **分段滑窗处理**: 对每个动作段分别进行滑动窗口处理
- 🚫 **避免跨段污染**: 确保每个训练窗口都来自同一动作段
- 🎯 **更纯净的数据**: 消除不同动作间的人工转换
- 📈 **更好的性能**: 模型学习到更真实的动作模式

#### 使用方法
```python
# 导入新的数据加载器
from data_processing.tool.load_training_data_segment_based import SegmentBasedDataLoader

# 创建分段数据加载器
data_loader = SegmentBasedDataLoader(
    train_folder='./data/model_training/train',
    test_folder='./data/model_training/test',
    window_size=60,      # 时间窗口大小
    time_step=3,         # 时间步长
    batch_size=256,
    validation_split=0.2,
    trial_id='your_trial_id'
)

# 加载和预处理数据
data_loader.load_and_prepare_data()

# 获取数据加载器
train_loader, val_loader, test_loader = data_loader.get_data_loaders()
```

### 训练脚本
```bash
# 训练右侧14传感器到15角度模型
python train/lstm_right_side_output.py

# 训练双侧27角度输出模型  
python train/lstm_output_27angle.py

# 使用改进的数据加载器训练
python train/notebook_modification_guide.py  # 查看修改指导
```

## 实时预测系统

### 启动预测服务
```bash
# 启动WebSocket服务器
python predict/websocket_server.py

# 启动可视化界面
python predict/start_dashboard.py
```

### 预测脚本
- `predict_r_side_14to15_local.py`: 右侧14传感器预测15角度
- `predict_ds_14Sto27A_local.py`: 14传感器预测27角度
- `predict_pxy_local.py`: 通用预测脚本

## 数据处理工具

### 批量数据处理
```bash
# 并行批量处理
python data_processing/batch_data_process_pxy_parallel.py

# 普通批量处理
python data_processing/batch_data_process_pxy.py
```

### ROM分析
```bash
# 运动范围分析
python data_processing/rom_anylsis/rom_cal.py
```

### 角度计算
```bash
# 坐标系角度计算
python data_processing/angle_calculation/angle_cal_coordinate.py
```

## 环境配置

### 依赖库
```bash
pip install pandas numpy scikit-learn torch matplotlib seaborn
pip install pathlib argparse json datetime
```

### Python版本要求
- Python 3.7+
- PyTorch 1.8+
- pandas 1.3+
- numpy 1.20+

## 使用流程

### 1. 数据准备
1. 将原始光学动捕数据和传感器数据放入`data/raw_data/`
2. 运行数据处理脚本生成对齐的数据文件
3. 使用数据集划分工具创建训练/测试集

### 2. 模型训练
1. 使用`dataset_split.py`划分数据集
2. 运行相应的LSTM训练脚本
3. 模型保存在`model/`目录

### 3. 预测使用
1. 启动预测服务
2. 通过Web界面或API进行实时预测
3. 查看预测结果和可视化


## 最新更新

### v1.5 新功能
- ✨ 新增智能数据集划分工具 (`dataset_split.py`)
- 📊 增强的统计分析功能
- 🔧 改进的命令行接口
- 📝 完善的文档和使用说明
- 🆕 **分段数据加载器** (`load_training_data_segment_based.py`)
  - 解决原有数据拼接导致的跨段问题
  - 每个动作段独立进行滑窗处理
  - 提供详细的修改指导文档
  - 支持灵活的参数配置

## 许可证

本项目仅供学术研究使用。

---

*最后更新: 2025年1月* 
