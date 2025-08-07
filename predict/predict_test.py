# 配置文件 config.py
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class SystemConfig:
    """系统配置类"""
    # 串口配置
    serial_port: str = 'COM3'
    baud_rate: int = 115200
    timeout: float = 1.0
    
    # 传感器配置
    num_sensors: int = 14
    sampling_rate: int = 30
    
    # 模型配置
    window_length: int = 80
    step_size: int = 1
    model_path: str = 'model.ckpt'
    input_tensor_name: str = 'input:0'
    output_tensor_name: str = 'output:0'
    
    # 数据处理配置
    data_buffer_size: int = 1000
    normalization_samples: int = 100
    
    # 输出配置
    output_file: str = 'predictions.csv'
    log_level: str = 'INFO'
    enable_visualization: bool = True
    
    @classmethod
    def from_json(cls, json_path: str):
        """从JSON文件加载配置"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self, json_path: str):
        """保存配置到JSON文件"""
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)


# 可视化监控模块 visualization.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque
import threading
import time

class RealTimeVisualizer:
    """实时数据可视化器"""
    
    def __init__(self, max_points=500, update_interval=100):
        self.max_points = max_points
        self.update_interval = update_interval
        
        # 数据存储
        self.sensor_data = deque(maxlen=max_points)
        self.prediction_data = deque(maxlen=max_points)
        self.timestamps = deque(maxlen=max_points)
        
        # 图形组件
        self.fig = None
        self.axes = None
        self.lines_sensor = []
        self.lines_prediction = []
        
        # 控制变量
        self.is_running = False
        self.lock = threading.Lock()
        
    def setup_plots(self, num_sensors, num_outputs):
        """设置绘图布局"""
        self.fig, self.axes = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle('实时传感器数据与角度预测')
        
        # 传感器数据子图
        ax_sensor = self.axes[0]
        ax_sensor.set_title('织物传感器数据')
        ax_sensor.set_ylabel('传感器值')
        ax_sensor.grid(True, alpha=0.3)
        
        # 预测数据子图
        ax_prediction = self.axes[1]
        ax_prediction.set_title('预测角度数据')
        ax_prediction.set_xlabel('时间 (s)')
        ax_prediction.set_ylabel('角度值')
        ax_prediction.grid(True, alpha=0.3)
        
        # 初始化线条（只显示部分传感器以避免图形过于复杂）
        sensor_indices = [0, 3, 6, 9, 12, 13]  # 选择6个传感器显示
        for i in sensor_indices:
            line, = ax_sensor.plot([], [], label=f'Sensor {i+1}', alpha=0.7)
            self.lines_sensor.append(line)
        
        # 预测线条
        for i in range(num_outputs):
            line, = ax_prediction.plot([], [], label=f'Angle {i+1}', linewidth=2)
            self.lines_prediction.append(line)
        
        ax_sensor.legend(loc='upper right')
        ax_prediction.legend(loc='upper right')
        
        plt.tight_layout()
    
    def update_data(self, sensor_values, prediction_values):
        """更新数据"""
        with self.lock:
            current_time = time.time()
            
            self.sensor_data.append(sensor_values)
            self.prediction_data.append(prediction_values)
            self.timestamps.append(current_time)
    
    def animate(self, frame):
        """动画更新函数"""
        with self.lock:
            if len(self.timestamps) < 2:
                return self.lines_sensor + self.lines_prediction
            
            # 转换时间戳为相对时间
            timestamps_array = np.array(list(self.timestamps))
            time_relative = timestamps_array - timestamps_array[0]
            
            # 更新传感器数据线条
            if len(self.sensor_data) > 0:
                sensor_array = np.array(list(self.sensor_data))
                sensor_indices = [0, 3, 6, 9, 12, 13]
                
                for i, line in enumerate(self.lines_sensor):
                    if sensor_indices[i] < sensor_array.shape[1]:
                        line.set_data(time_relative, sensor_array[:, sensor_indices[i]])
            
            # 更新预测数据线条
            if len(self.prediction_data) > 0:
                prediction_array = np.array(list(self.prediction_data))
                
                for i, line in enumerate(self.lines_prediction):
                    if i < prediction_array.shape[1]:
                        line.set_data(time_relative, prediction_array[:, i])
            
            # 自动调整轴范围
            if len(time_relative) > 0:
                for ax in self.axes:
                    ax.relim()
                    ax.autoscale_view()
        
        return self.lines_sensor + self.lines_prediction
    
    def start_visualization(self, num_sensors=14, num_outputs=1):
        """启动可视化"""
        self.setup_plots(num_sensors, num_outputs)
        self.is_running = True
        
        # 创建动画
        self.anim = animation.FuncAnimation(
            self.fig, self.animate, interval=self.update_interval, 
            blit=False, cache_frame_data=False
        )
        
        plt.show(block=False)
    
    def stop_visualization(self):
        """停止可视化"""
        self.is_running = False
        if hasattr(self, 'anim'):
            self.anim.event_source.stop()
        plt.close(self.fig)


# 数据记录器 data_logger.py
import csv
import os
from datetime import datetime
import threading

class DataLogger:
    """数据记录器"""
    
    def __init__(self, output_file='predictions.csv'):
        self.output_file = output_file
        self.lock = threading.Lock()
        self.is_initialized = False
        
    def initialize_file(self, num_sensors, num_outputs):
        """初始化CSV文件"""
        with self.lock:
            # 创建表头
            headers = ['timestamp']
            
            # 传感器数据列
            for i in range(num_sensors):
                headers.append(f'sensor_{i+1}')
            
            # 预测数据列
            for i in range(num_outputs):
                headers.append(f'prediction_{i+1}')
            
            # 写入表头
            with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            
            self.is_initialized = True
    
    def log_data(self, sensor_data, prediction_data):
        """记录数据"""
        if not self.is_initialized:
            return
        
        with self.lock:
            timestamp = datetime.now().isoformat()
            
            # 组合数据行
            row = [timestamp]
            row.extend(sensor_data.tolist())
            row.extend(prediction_data.tolist())
            
            # 写入CSV文件
            with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)


# 改进的主系统类
class AdvancedInferenceSystem:
    """增强版实时推理系统"""
    
    def __init__(self, config_path='config.json'):
        # 加载配置
        if os.path.exists(config_path):
            self.config = SystemConfig.from_json(config_path)
        else:
            self.config = SystemConfig()
            self.config.to_json(config_path)
        
        # 初始化组件
        self.data_reader = SerialDataReader(self.config)
        self.data_processor = DataProcessor(self.config)
        self.model_inference = ModelInference(self.config)
        
        # 可选组件
        self.visualizer = None
        self.data_logger = None
        
        if self.config.enable_visualization:
            self.visualizer = RealTimeVisualizer()
        
        if self.config.output_file:
            self.data_logger = DataLogger(self.config.output_file)
        
        # 性能统计
        self.stats = {
            'predictions': 0,
            'errors': 0,
            'start_time': None,
            'last_prediction_time': None
        }
    
    def start(self):
        """启动系统"""
        logger.info("启动增强版实时推理系统...")
        
        # 连接串口
        if not self.data_reader.connect():
            return False
        
        # 初始化数据记录器
        if self.data_logger:
            # 假设输出维度为关节数量，需要根据实际模型调整
            output_dim = 10  # 根据您的模型输出维度调整
            self.data_logger.initialize_file(self.config.num_sensors, output_dim)
        
        # 启动可视化
        if self.visualizer:
            output_dim = 10  # 根据您的模型输出维度调整
            self.visualizer.start_visualization(self.config.num_sensors, output_dim)
        
        # 开始读取数据
        if not self.data_reader.start_reading():
            return False
        
        self.stats['start_time'] = time.time()
        logger.info("增强版系统启动成功")
        return True
    
    def process_loop(self):
        """主处理循环"""
        try:
            while True:
                # 获取串口数据
                raw_data = self.data_reader.get_data()
                if raw_data is None:
                    time.sleep(0.01)
                    continue
                
                # 解析数据
                sensor_data = self.data_processor.parse_serial_data(raw_data)
                if sensor_data is None:
                    self.stats['errors'] += 1
                    continue
                
                # 添加数据点并检查是否可以进行推理
                window_data = self.data_processor.add_data_point(sensor_data)
                if window_data is not None:
                    # 执行推理
                    prediction = self.model_inference.predict(window_data)
                    if prediction is not None:
                        self.stats['predictions'] += 1
                        self.stats['last_prediction_time'] = time.time()
                        
                        # 输出预测结果
                        self._output_prediction(prediction, sensor_data)
                        
                        # 更新可视化
                        if self.visualizer:
                            self.visualizer.update_data(sensor_data, prediction)
                        
                        # 记录数据
                        if self.data_logger:
                            self.data_logger.log_data(sensor_data, prediction)
                    else:
                        self.stats['errors'] += 1
                
                # 定期输出统计信息
                if self.stats['predictions'] % 100 == 0 and self.stats['predictions'] > 0:
                    self._print_statistics()
                
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在停止系统...")
        finally:
            self.stop()
    
    def _output_prediction(self, prediction, sensor_data):
        """输出预测结果"""
        # 计算推理频率
        if self.stats['start_time']:
            elapsed = time.time() - self.stats['start_time']
            frequency = self.stats['predictions'] / elapsed if elapsed > 0 else 0
        else:
            frequency = 0
        
        # 格式化输出
        pred_str = ', '.join([f"{x:.4f}" for x in prediction])
        sensor_summary = f"传感器均值: {np.mean(sensor_data):.4f}"
        
        print(f"\r预测角度: [{pred_str}] | {sensor_summary} | 频率: {frequency:.1f}Hz", end='')
        
        # 每10次预测换行一次
        if self.stats['predictions'] % 10 == 0:
            print()
    
    def _print_statistics(self):
        """打印统计信息"""
        elapsed = time.time() - self.stats['start_time']
        avg_freq = self.stats['predictions'] / elapsed
        error_rate = self.stats['errors'] / (self.stats['predictions'] + self.stats['errors'])
        
        logger.info(f"统计 - 预测: {self.stats['predictions']}, "
                   f"错误: {self.stats['errors']}, "
                   f"平均频率: {avg_freq:.2f}Hz, "
                   f"错误率: {error_rate:.2%}")
    
    def stop(self):
        """停止系统"""
        logger.info("正在停止增强版系统...")
        
        self.data_reader.stop()
        
        if self.visualizer:
            self.visualizer.stop_visualization()
        
        self._print_statistics()
        logger.info("增强版系统已停止")


# 示例配置文件内容
DEFAULT_CONFIG = {
    "serial_port": "COM3",
    "baud_rate": 115200,
    "timeout": 1.0,
    "num_sensors": 14,
    "sampling_rate": 30,
    "window_length": 80,
    "step_size": 1,
    "model_path": "your_model.ckpt",
    "input_tensor_name": "input:0",
    "output_tensor_name": "output:0",
    "data_buffer_size": 1000,
    "normalization_samples": 100,
    "output_file": "predictions.csv",
    "log_level": "INFO",
    "enable_visualization": true
}


# 性能监控器 performance_monitor.py
import time
import psutil
import threading
from collections import deque
import numpy as np

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, monitor_interval=1.0):
        self.monitor_interval = monitor_interval
        self.is_monitoring = False
        
        # 性能数据存储
        self.cpu_usage = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        self.inference_times = deque(maxlen=1000)
        self.timestamps = deque(maxlen=100)
        
        # 统计信息
        self.total_inferences = 0
        self.total_errors = 0
        self.start_time = None
        
    def start_monitoring(self):
        """开始性能监控"""
        self.is_monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 记录系统性能
                cpu_percent = psutil.cpu_percent()
                memory_info = psutil.virtual_memory()
                
                self.cpu_usage.append(cpu_percent)
                self.memory_usage.append(memory_info.percent)
                self.timestamps.append(time.time())
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                print(f"性能监控错误: {e}")
                time.sleep(self.monitor_interval)
    
    def record_inference_time(self, inference_time):
        """记录推理时间"""
        self.inference_times.append(inference_time)
        self.total_inferences += 1
    
    def record_error(self):
        """记录错误"""
        self.total_errors += 1
    
    def get_statistics(self):
        """获取性能统计"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time if self.start_time else 0
        
        stats = {
            'elapsed_time': elapsed_time,
            'total_inferences': self.total_inferences,
            'total_errors': self.total_errors,
            'inference_rate': self.total_inferences / elapsed_time if elapsed_time > 0 else 0,
            'error_rate': self.total_errors / (self.total_inferences + self.total_errors) if (self.total_inferences + self.total_errors) > 0 else 0
        }
        
        if len(self.cpu_usage) > 0:
            stats['avg_cpu_usage'] = np.mean(list(self.cpu_usage))
            stats['max_cpu_usage'] = np.max(list(self.cpu_usage))
        
        if len(self.memory_usage) > 0:
            stats['avg_memory_usage'] = np.mean(list(self.memory_usage))
            stats['max_memory_usage'] = np.max(list(self.memory_usage))
        
        if len(self.inference_times) > 0:
            stats['avg_inference_time'] = np.mean(list(self.inference_times))
            stats['max_inference_time'] = np.max(list(self.inference_times))
            stats['min_inference_time'] = np.min(list(self.inference_times))
        
        return stats
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2.0)


# 错误处理和重连机制 error_handler.py
import time
import logging
from enum import Enum

class SystemState(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    ERROR = "error"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"

class ErrorHandler:
    """错误处理器"""
    
    def __init__(self, max_retries=5, retry_delay=2.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_count = 0
        self.last_error_time = 0
        self.error_threshold = 10  # 连续错误阈值
        self.consecutive_errors = 0
        
    def handle_error(self, error, error_type="general"):
        """处理错误"""
        current_time = time.time()
        self.consecutive_errors += 1
        
        logging.error(f"错误类型: {error_type}, 错误信息: {error}")
        
        # 检查是否需要重连
        if self.consecutive_errors >= self.error_threshold:
            logging.warning(f"连续错误次数达到阈值({self.error_threshold})，建议重启系统")
            return False
        
        # 检查重试间隔
        if current_time - self.last_error_time < self.retry_delay:
            time.sleep(self.retry_delay)
        
        self.last_error_time = current_time
        self.retry_count += 1
        
        if self.retry_count >= self.max_retries:
            logging.error(f"重试次数超过最大限制({self.max_retries})")
            return False
        
        return True
    
    def reset_error_count(self):
        """重置错误计数"""
        self.consecutive_errors = 0
        self.retry_count = 0


# 完整的启动脚本 main.py
import argparse
import logging
import signal
import sys
import os

def setup_logging(log_level="INFO"):
    """设置日志"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('inference_system.log', encoding='utf-8')
        ]
    )

def signal_handler(signum, frame):
    """信号处理器"""
    print(f"\n收到信号 {signum}，正在优雅地关闭系统...")
    sys.exit(0)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='实时织物传感器角度预测系统')
    parser.add_argument('--config', '-c', default='config.json', help='配置文件路径')
    parser.add_argument('--port', '-p', help='串口端口号')
    parser.add_argument('--model', '-m', help='模型文件路径')
    parser.add_argument('--no-viz', action='store_true', help='禁用可视化')
    parser.add_argument('--output', '-o', help='输出文件路径')
    parser.add_argument('--log-level', default='INFO', help='日志级别')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # 设置信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 加载配置
        if os.path.exists(args.config):
            config = SystemConfig.from_json(args.config)
            logger.info(f"从文件加载配置: {args.config}")
        else:
            config = SystemConfig()
            config.to_json(args.config)
            logger.info(f"创建默认配置文件: {args.config}")
        
        # 命令行参数覆盖配置
        if args.port:
            config.serial_port = args.port
        if args.model:
            config.model_path = args.model
        if args.no_viz:
            config.enable_visualization = False
        if args.output:
            config.output_file = args.output
        
        # 验证模型文件是否存在
        if not os.path.exists(f"{config.model_path}.meta"):
            logger.error(f"模型文件不存在: {config.model_path}.meta")
            return 1
        
        # 创建并启动系统
        system = AdvancedInferenceSystem()
        system.config = config
        
        # 重新初始化组件
        from fabric_sensor_inference import SerialDataReader, DataProcessor, ModelInference
        system.data_reader = SerialDataReader(config)
        system.data_processor = DataProcessor(config)
        system.model_inference = ModelInference(config)
        
        if config.enable_visualization:
            system.visualizer = RealTimeVisualizer()
        
        if config.output_file:
            system.data_logger = DataLogger(config.output_file)
        
        # 添加性能监控
        performance_monitor = PerformanceMonitor()
        performance_monitor.start_monitoring()
        
        logger.info("="*50)
        logger.info("实时织物传感器角度预测系统")
        logger.info("="*50)
        logger.info(f"串口: {config.serial_port}")
        logger.info(f"波特率: {config.baud_rate}")
        logger.info(f"传感器数量: {config.num_sensors}")
        logger.info(f"窗口长度: {config.window_length}")
        logger.info(f"模型路径: {config.model_path}")
        logger.info(f"可视化: {'启用' if config.enable_visualization else '禁用'}")
        logger.info(f"输出文件: {config.output_file or '无'}")
        logger.info("="*50)
        
        # 启动系统
        if system.start():
            logger.info("系统启动成功，按 Ctrl+C 停止")
            system.process_loop()
        else:
            logger.error("系统启动失败")
            return 1
            
    except Exception as e:
        logger.error(f"系统异常: {e}", exc_info=True)
        return 1
    
    finally:
        if 'performance_monitor' in locals():
            performance_monitor.stop_monitoring()
            stats = performance_monitor.get_statistics()
            logger.info("系统性能统计:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)