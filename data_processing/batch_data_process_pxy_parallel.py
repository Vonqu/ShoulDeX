import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
from functools import partial
import time
import logging

#from angle_cal import calculate_all_angles  #angle_cal_pxy是缺少C7 的坐标系
#from angle_cal_double import calculate_all_angles #双侧18个角度
from angle_calculation.angle_cal_coordinate import calculate_all_angles #外展重点角度-14个角度
from read_opticla_pxy import read_optical_data
#from read_snesor_data import read_sensor_data
#from read_snesor_data_0406 import read_sensor_data
from read_sensor_data_0719 import read_sensor_data
from get_intersection_data_0406 import get_intersection_data #6sensor用get_intersection_data_pxy

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 线程锁，用于保护文件写入操作
file_lock = threading.Lock()

def process_single_data_group(optical_filepath, sensor_filepath, output_angle_dir, output_dft_dir):
    """
    处理单个数据组的函数，保持原有算法不变
    """
    try:
        # 读取光捕数据
        df_o = read_optical_data(optical_filepath)
        
        # 计算光捕角度
        df_o = calculate_all_angles(df_o) # 肩关节角度
        df_angle = df_o[['Frame', 'Time',
                     # Angle 1-3: 左肱骨与胸廓坐标系角度
                     'A_humerus_l_thorax_X', 'A_humerus_l_thorax_Y', 'A_humerus_l_thorax_Z',
                     # Angle 4-6: 右肱骨与胸廓坐标系角度
                     'A_humerus_r_thorax_X', 'A_humerus_r_thorax_Y', 'A_humerus_r_thorax_Z',
                     # Angle 7-9: 左肩胛骨与胸廓坐标系角度
                     'A_scapula_l_thorax_X', 'A_scapula_l_thorax_Y', 'A_scapula_l_thorax_Z',
                     # Angle 10-12: 右肩胛骨与胸廓坐标系角度
                     'A_scapula_r_thorax_X', 'A_scapula_r_thorax_Y', 'A_scapula_r_thorax_Z',
                     # Angle 13-15: 左锁骨与胸廓坐标系角度
                     'A_clavicle_l_thorax_X', 'A_clavicle_l_thorax_Y', 'A_clavicle_l_thorax_Z',
                     # Angle 16-18: 右锁骨与胸廓坐标系角度
                     'A_clavicle_r_thorax_X', 'A_clavicle_r_thorax_Y', 'A_clavicle_r_thorax_Z',
                     # Angle 19-21: 左肱骨与左肩胛骨坐标系角度
                     'A_humerus_l_scapula_X', 'A_humerus_l_scapula_Y', 'A_humerus_l_scapula_Z',
                     # Angle 22-24: 右肱骨与右肩胛骨坐标系角度
                     'A_humerus_r_scapula_X', 'A_humerus_r_scapula_Y', 'A_humerus_r_scapula_Z',
                     # Angle 25-27: 胸廓与髋关节坐标系角度
                     'A_thorax_hip_X', 'A_thorax_hip_Y', 'A_thorax_hip_Z']]
        
        # 输出角度数据（使用线程锁保护文件写入）
        angle_output_path = os.path.join(output_angle_dir, os.path.basename(optical_filepath).replace('.csv', 'angle.csv'))
        with file_lock:
            df_angle.to_csv(angle_output_path, index=False)
        
        # 读取传感器数据
        df_s, df_s_resampled = read_sensor_data(sensor_filepath)
        
        # 数据对齐（交集）
        datafinal = get_intersection_data(df_angle, df_s_resampled)
        
        # 合并数据写出（使用线程锁保护文件写入）
        final_output_path = os.path.join(output_dft_dir, os.path.basename(sensor_filepath).replace('.txt', 'dft.csv'))
        with file_lock:
            datafinal.to_csv(final_output_path, index=False)
        
        return {
            'optical_file': os.path.basename(optical_filepath),
            'sensor_file': os.path.basename(sensor_filepath),
            'status': 'success',
            'data': datafinal
        }
        
    except Exception as e:
        logger.error(f"处理文件时出错: {optical_filepath}, {sensor_filepath}, 错误: {str(e)}")
        return {
            'optical_file': os.path.basename(optical_filepath),
            'sensor_file': os.path.basename(sensor_filepath),
            'status': 'error',
            'error': str(e)
        }

def process_file_pair(args):
    """
    处理文件对的包装函数，用于多进程
    """
    optical_filepath, sensor_filepath, output_angle_dir, output_dft_dir = args
    return process_single_data_group(optical_filepath, sensor_filepath, output_angle_dir, output_dft_dir)

def batch_process_parallel(optical_dir, sensor_dir, output_angle_dir, output_dft_dir, max_workers=None):
    """
    并行批处理函数
    """
    # 确保输出目录存在
    os.makedirs(output_angle_dir, exist_ok=True)
    os.makedirs(output_dft_dir, exist_ok=True)

    if not os.path.exists(optical_dir):
        logger.error("指定的目录不存在")
        return

    optical_files = os.listdir(optical_dir)
    sensor_files = os.listdir(sensor_dir)
    
    # 如果没有指定最大工作进程数，使用CPU核心数
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(optical_files) * len(sensor_files))
    
    logger.info(f"开始并行处理，使用 {max_workers} 个工作进程")
    logger.info(f"光捕文件数量: {len(optical_files)}, 传感器文件数量: {len(sensor_files)}")
    
    # 准备任务列表
    tasks = []
    for optical_file in optical_files:
        for sensor_file in sensor_files:
            optical_filepath = os.path.join(optical_dir, optical_file)
            sensor_filepath = os.path.join(sensor_dir, sensor_file)
            tasks.append((optical_filepath, sensor_filepath, output_angle_dir, output_dft_dir))
    
    # 使用进程池执行任务
    start_time = time.time()
    successful_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(process_file_pair, task): task for task in tasks}
        
        # 处理完成的任务
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                if result['status'] == 'success':
                    successful_count += 1
                    logger.info(f"成功处理: {result['optical_file']} 和 {result['sensor_file']}")
                else:
                    error_count += 1
                    logger.error(f"处理失败: {result['optical_file']} 和 {result['sensor_file']}, 错误: {result['error']}")
            except Exception as e:
                error_count += 1
                logger.error(f"任务执行异常: {task[0]}, {task[1]}, 错误: {str(e)}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"批处理完成！")
    logger.info(f"总任务数: {len(tasks)}")
    logger.info(f"成功: {successful_count}")
    logger.info(f"失败: {error_count}")
    logger.info(f"总耗时: {total_time:.2f} 秒")
    logger.info(f"平均每个任务耗时: {total_time/len(tasks):.2f} 秒")

def batch_process_threaded(optical_dir, sensor_dir, output_angle_dir, output_dft_dir, max_workers=None):
    """
    使用线程池的批处理函数（适用于I/O密集型任务）
    """
    # 确保输出目录存在
    os.makedirs(output_angle_dir, exist_ok=True)
    os.makedirs(output_dft_dir, exist_ok=True)

    if not os.path.exists(optical_dir):
        logger.error("指定的目录不存在")
        return

    optical_files = os.listdir(optical_dir)
    sensor_files = os.listdir(sensor_dir)
    
    # 如果没有指定最大工作线程数，使用CPU核心数的2倍
    if max_workers is None:
        max_workers = min(mp.cpu_count() * 2, len(optical_files) * len(sensor_files))
    
    logger.info(f"开始线程池处理，使用 {max_workers} 个工作线程")
    logger.info(f"光捕文件数量: {len(optical_files)}, 传感器文件数量: {len(sensor_files)}")
    
    # 准备任务列表
    tasks = []
    for optical_file in optical_files:
        for sensor_file in sensor_files:
            optical_filepath = os.path.join(optical_dir, optical_file)
            sensor_filepath = os.path.join(sensor_dir, sensor_file)
            tasks.append((optical_filepath, sensor_filepath, output_angle_dir, output_dft_dir))
    
    # 使用线程池执行任务
    start_time = time.time()
    successful_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(process_file_pair, task): task for task in tasks}
        
        # 处理完成的任务
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                if result['status'] == 'success':
                    successful_count += 1
                    logger.info(f"成功处理: {result['optical_file']} 和 {result['sensor_file']}")
                else:
                    error_count += 1
                    logger.error(f"处理失败: {result['optical_file']} 和 {result['sensor_file']}, 错误: {result['error']}")
            except Exception as e:
                error_count += 1
                logger.error(f"任务执行异常: {task[0]}, {task[1]}, 错误: {str(e)}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"批处理完成！")
    logger.info(f"总任务数: {len(tasks)}")
    logger.info(f"成功: {successful_count}")
    logger.info(f"失败: {error_count}")
    logger.info(f"总耗时: {total_time:.2f} 秒")
    logger.info(f"平均每个任务耗时: {total_time/len(tasks):.2f} 秒")

def batch_process_hybrid(optical_dir, sensor_dir, output_angle_dir, output_dft_dir, process_workers=None, thread_workers=None):
    """
    混合并行处理：使用进程池处理计算密集型任务，线程池处理I/O任务
    """
    # 确保输出目录存在
    os.makedirs(output_angle_dir, exist_ok=True)
    os.makedirs(output_dft_dir, exist_ok=True)

    if not os.path.exists(optical_dir):
        logger.error("指定的目录不存在")
        return

    optical_files = os.listdir(optical_dir)
    sensor_files = os.listdir(sensor_dir)
    
    # 设置工作进程和线程数
    if process_workers is None:
        process_workers = min(mp.cpu_count(), len(optical_files) * len(sensor_files))
    if thread_workers is None:
        thread_workers = min(mp.cpu_count() * 2, len(optical_files) * len(sensor_files))
    
    logger.info(f"开始混合并行处理，使用 {process_workers} 个进程和 {thread_workers} 个线程")
    logger.info(f"光捕文件数量: {len(optical_files)}, 传感器文件数量: {len(sensor_files)}")
    
    # 准备任务列表
    tasks = []
    for optical_file in optical_files:
        for sensor_file in sensor_files:
            optical_filepath = os.path.join(optical_dir, optical_file)
            sensor_filepath = os.path.join(sensor_dir, sensor_file)
            tasks.append((optical_filepath, sensor_filepath, output_angle_dir, output_dft_dir))
    
    # 使用进程池执行任务
    start_time = time.time()
    successful_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=process_workers) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(process_file_pair, task): task for task in tasks}
        
        # 处理完成的任务
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                if result['status'] == 'success':
                    successful_count += 1
                    logger.info(f"成功处理: {result['optical_file']} 和 {result['sensor_file']}")
                else:
                    error_count += 1
                    logger.error(f"处理失败: {result['optical_file']} 和 {result['sensor_file']}, 错误: {result['error']}")
            except Exception as e:
                error_count += 1
                logger.error(f"任务执行异常: {task[0]}, {task[1]}, 错误: {str(e)}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"混合并行处理完成！")
    logger.info(f"总任务数: {len(tasks)}")
    logger.info(f"成功: {successful_count}")
    logger.info(f"失败: {error_count}")
    logger.info(f"总耗时: {total_time:.2f} 秒")
    logger.info(f"平均每个任务耗时: {total_time/len(tasks):.2f} 秒")

if __name__ == "__main__":
    # 设定文件夹路径
    participant_ids = ['qnc']
    # motion_types = ['compensatory']
    motion_types = ['normal', 'compensatory']
    
    # 选择并行处理方式
    # 1: 进程池（适合CPU密集型任务）
    # 2: 线程池（适合I/O密集型任务）
    # 3: 混合模式（推荐）
    parallel_mode = 3
    
    for participant_id in participant_ids:
        for motion_type in motion_types:
            logger.info(f"开始处理参与者: {participant_id}, 运动类型: {motion_type}")

            OPTICAL_DATA_DIR = f'./data/raw_data/optical/{participant_id}/{motion_type}'    # 光捕数据
            SENSOR_DATA_DIR = f'./data/raw_data/sensor/{participant_id}/{motion_type}'      # 传感器数据
            OUTPUT_ANGLE_DIR = f'./data/angle_data_final/{participant_id}/{motion_type}'     # 角度输出文件夹
            OUTPUT_DFT_DIR = f'./data/dft_final/{participant_id}/{motion_type}'              # 训练数据输出

            if parallel_mode == 1:
                # 使用进程池
                batch_process_parallel(OPTICAL_DATA_DIR, SENSOR_DATA_DIR, OUTPUT_ANGLE_DIR, OUTPUT_DFT_DIR)
            elif parallel_mode == 2:
                # 使用线程池
                batch_process_threaded(OPTICAL_DATA_DIR, SENSOR_DATA_DIR, OUTPUT_ANGLE_DIR, OUTPUT_DFT_DIR)
            elif parallel_mode == 3:
                # 使用混合模式
                batch_process_hybrid(OPTICAL_DATA_DIR, SENSOR_DATA_DIR, OUTPUT_ANGLE_DIR, OUTPUT_DFT_DIR)
            else:
                logger.error("无效的并行模式选择") 