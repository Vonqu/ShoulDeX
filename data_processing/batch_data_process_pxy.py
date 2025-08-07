import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
#from angle_cal import calculate_all_angles  #angle_cal_pxy是缺少C7 的坐标系
#from angle_cal_double import calculate_all_angles #双侧18个角度
from data_processing.angle_calculation.angle_cal_coordinate import calculate_all_angles #外展重点角度-14个角度
from read_opticla_pxy import read_optical_data
#from read_snesor_data import read_sensor_data
#from read_snesor_data_0406 import read_sensor_data
from read_sensor_data_0719 import read_sensor_data
from get_intersection_data_0406 import get_intersection_data #6sensor用get_intersection_data_pxy


def process_single_data_group(optical_filepath, sensor_filepath, output_angle_dir,output_dft_dir):
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
    

    # 输出角度数据
    angle_output_path = os.path.join(output_angle_dir, os.path.basename(optical_filepath).replace('.csv', 'angle.csv'))
    df_angle.to_csv(angle_output_path, index=False)
    
    # 读取传感器数据
    df_s, df_s_resampled = read_sensor_data(sensor_filepath)
    
    # 数据对齐（交集）
    datafinal = get_intersection_data(df_angle, df_s_resampled)
    
    # 合并数据写出
    final_output_path = os.path.join(output_dft_dir, os.path.basename(sensor_filepath).replace('.txt', 'dft.csv'))
    datafinal.to_csv(final_output_path, index=False)
    
    return datafinal


def batch_process(optical_dir, sensor_dir,  output_angle_dir, output_dft_dir):
    # 确保输出目录存在
    os.makedirs( output_angle_dir, exist_ok=True)
    os.makedirs( output_dft_dir, exist_ok=True)


    if not os.path.exists(optical_dir):
        print("指定的目录不存在")
        # return
    optical_files = os.listdir(optical_dir)
    sensor_files = os.listdir(sensor_dir)


    for optical_file in optical_files:
        for sensor_file in sensor_files:
            optical_filepath = os.path.join(optical_dir, optical_file)
            sensor_filepath = os.path.join(sensor_dir, sensor_file)
            datafinal = process_single_data_group(optical_filepath, sensor_filepath, output_angle_dir,output_dft_dir)
            print(f'Processed {optical_file} and {sensor_file}')


if __name__ == "__main__":
    # 设定文件夹路径
    participant_ids = ['qnc', 'lyh', 'hyd']
    motion_types = ['normal', 'compensatory']

    for participant_id in participant_ids:
        for motion_type in motion_types:
            print(f"Processing participant: {participant_id}, motion type: {motion_type}")

            OPTICAL_DATA_DIR = f'./data/raw_data/optical/{participant_id}/{motion_type}'    # 光捕数据
            SENSOR_DATA_DIR = f'./data/raw_data/sensor/{participant_id}/{motion_type}'      # 传感器数据
            OUTPUT_ANGLE_DIR = f'./data/angle_data_test/{participant_id}/{motion_type}'     # 角度输出文件夹
            OUTPUT_DFT_DIR = f'./data/dft_data/{participant_id}/{motion_type}'              # 训练数据输出

            batch_process(OPTICAL_DATA_DIR, SENSOR_DATA_DIR, OUTPUT_ANGLE_DIR, OUTPUT_DFT_DIR)