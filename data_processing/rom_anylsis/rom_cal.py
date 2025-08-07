import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import PatternFill

def calculate_rom_statistics(data_path):
    """
    计算传感器数据和角度数据的ROM统计分析
    
    Args:
        data_path (str): 包含CSV文件的目录路径
    
    Returns:
        tuple: (sensor_stats_df, angle_stats_df) 传感器和角度统计数据的DataFrame
    """
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    sensor_data = {}
    angle_data = {}
    
    for file_path in csv_files:
        print(f"正在处理: {os.path.basename(file_path)}")
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        file_name = os.path.basename(file_path).replace('.csv', '')
        
        # 排除时间列
        numeric_columns = df.columns[1:]  # 跳过Time_angle列
        numeric_data = df[numeric_columns]
        
        # 识别传感器列和角度列
        sensor_columns = [col for col in numeric_columns if col.startswith('s')]
        angle_columns = [col for col in numeric_columns if col.startswith('A_')]
        
        # 处理传感器数据
        for col in sensor_columns:
            col_data = numeric_data[col]
            
            if col not in sensor_data:
                sensor_data[col] = {}
            
            # 基础统计量
            sensor_data[col][f'{file_name}_最小值'] = col_data.min()
            sensor_data[col][f'{file_name}_最大值'] = col_data.max()
            sensor_data[col][f'{file_name}_平均值'] = col_data.mean()
            # sensor_data[col][f'{file_name}_中位数'] = col_data.median()
            # sensor_data[col][f'{file_name}_标准差'] = col_data.std()
            sensor_data[col][f'{file_name}_方差'] = col_data.var()
            sensor_data[col][f'{file_name}_峰值因子'] = col_data.max() / col_data.mean() if col_data.mean() != 0 else 0
            sensor_data[col][f'{file_name}_变异系数CV'] = col_data.std() / col_data.mean() if col_data.mean() != 0 else 0
        
        # 处理角度数据
        for col in angle_columns:
            col_data = numeric_data[col]
            
            if col not in angle_data:
                angle_data[col] = {}
            
            # 基础统计量
            angle_data[col][f'{file_name}_最小值'] = col_data.min()
            angle_data[col][f'{file_name}_最大值'] = col_data.max()
            angle_data[col][f'{file_name}_平均值'] = col_data.mean()
            # angle_data[col][f'{file_name}_中位数'] = col_data.median()
            # angle_data[col][f'{file_name}_标准差'] = col_data.std()
            angle_data[col][f'{file_name}_方差'] = col_data.var()
            
            # ROM特异性指标
            angle_data[col][f'{file_name}_运动范围ROM'] = col_data.max() - col_data.min()
            # 计算四分位数范围
            # q75, q25 = np.percentile(col_data, [75, 25])
            # angle_data[col][f'{file_name}_四分位数范围IQR'] = q75 - q25
            # angle_data[col][f'{file_name}_第25百分位'] = q25
            # angle_data[col][f'{file_name}_第75百分位'] = q75
            # angle_data[col][f'{file_name}_第95百分位'] = np.percentile(col_data, 95)
            # angle_data[col][f'{file_name}_第5百分位'] = np.percentile(col_data, 5)
    
    # 转换为DataFrame
    sensor_stats_df = pd.DataFrame(sensor_data).T  # 转置，行为传感器，列为动作_统计指标
    angle_stats_df = pd.DataFrame(angle_data).T    # 转置，行为角度，列为动作_统计指标
    
    return sensor_stats_df, angle_stats_df

def create_hierarchical_structure(df, data_type):
    """
    创建分层结构的DataFrame，便于查看
    
    Args:
        df (pd.DataFrame): 统计数据
        data_type (str): 数据类型（'sensor'或'angle'）
    
    Returns:
        pd.DataFrame: 重新组织的DataFrame
    """
    
    # 提取所有的文件名（动作名称）
    actions = set()
    for col in df.columns:
        # 从列名中提取文件名，格式为 "文件名_统计指标"
        parts = col.split('_')
        if len(parts) >= 2:
            # 重新组合除了最后一个统计指标之外的所有部分作为文件名
            action_name = '_'.join(parts[:-1])
            actions.add(action_name)
    
    actions = sorted(list(actions))
    
    # 提取统计指标类型
    if data_type == 'sensor':
        stat_types = ['最小值', '最大值', '平均值','方差', '峰值因子', '变异系数CV']
    else:  # angle
        # stat_types = ['最小值', '最大值', '平均值', '中位数', '标准差', '方差', 
        #              '运动范围ROM', '四分位数范围IQR', '第25百分位', '第75百分位', 
        #              '第95百分位', '第5百分位']
        stat_types = ['最小值', '最大值', '平均值', '方差', 
                     '运动范围ROM']
    
    # 重新组织数据
    reorganized_data = {}
    
    for stat_type in stat_types:
        for action in actions:
            col_name = f"{action}_{stat_type}"
            if col_name in df.columns:
                reorganized_data[action] = reorganized_data.get(action, {})
                for sensor_angle in df.index:
                    if sensor_angle not in reorganized_data[action]:
                        reorganized_data[action][sensor_angle] = {}
                    reorganized_data[action][sensor_angle][stat_type] = df.loc[sensor_angle, col_name]
    
    # 创建多级索引的DataFrame
    final_data = []
    multi_index = []
    
    for sensor_angle in df.index:
        for stat_type in stat_types:
            row_data = []
            for action in actions:
                value = None
                if action in reorganized_data and sensor_angle in reorganized_data[action]:
                    value = reorganized_data[action][sensor_angle].get(stat_type, None)
                row_data.append(value)
            
            final_data.append(row_data)
            multi_index.append((sensor_angle, stat_type))
    
    # 创建多级索引
    index = pd.MultiIndex.from_tuples(multi_index, names=[data_type.capitalize(), '统计指标'])
    
    result_df = pd.DataFrame(final_data, index=index, columns=actions)
    
    return result_df

def main():
    # 数据路径

    participant_id = 'hyd'
    motion_type = 'normal'

    data_path = f"F:\coding\ShoulDeX\ShoulDex_v1.0\ShoulDeX\data\dft_final\{participant_id}\{motion_type}"
    
    print("开始ROM统计分析...")
    print(f"分析路径: {data_path}")
    
    # 检查路径是否存在
    if not os.path.exists(data_path):
        print(f"错误: 路径 {data_path} 不存在")
        return
    
    # 计算统计数据
    sensor_stats_df, angle_stats_df = calculate_rom_statistics(data_path)
    
    # 创建分层结构
    sensor_hierarchical = create_hierarchical_structure(sensor_stats_df, 'sensor')
    angle_hierarchical = create_hierarchical_structure(angle_stats_df, 'angle')
    
    # 保存结果
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建Excel文件，包含两个sheet
    excel_output = os.path.join(output_dir, f"{participant_id}_{motion_type}_rom_statistics.xlsx")
    
    with pd.ExcelWriter(excel_output, engine='openpyxl') as writer:
                
        # 角度数据sheet
        angle_hierarchical.to_excel(writer, sheet_name='angle_data')
        
        # 传感器数据sheet
        sensor_hierarchical.to_excel(writer, sheet_name='sensor_data')

        # 为角度数据的ROM行添加色阶格式
        workbook = writer.book
        angle_sheet = workbook['angle_data']
        
        # 找到所有"运动范围ROM"的行
        for row_idx, (angle_name, stat_type) in enumerate(angle_hierarchical.index, start=2):  # 从第2行开始（第1行是表头）
            if stat_type == '运动范围ROM':
                # 获取数据范围（从C列开始，因为A列是角度名，B列是统计指标）
                start_col = 3  # C列
                end_col = start_col + len(angle_hierarchical.columns) - 1
                
                # 创建色阶规则：蓝色(最小值) -> 白色(中间值) -> 红色(最大值)
                color_scale = ColorScaleRule(
                    start_type='min', start_color='0066CC',  # 蓝色
                    mid_type='percentile', mid_value=50, mid_color='FFFFFF',  # 白色
                    end_type='max', end_color='FF0000'  # 红色
                )
                
                # 应用色阶到该行
                range_string = f"{chr(64 + start_col)}{row_idx}:{chr(64 + end_col)}{row_idx}"
                angle_sheet.conditional_formatting.add(range_string, color_scale)
    
    print(f"ROM统计结果已保存至: {excel_output}")
    
    # 打印基本信息
    print(f"\n=== 分析完成 ===")
    print(f"传感器数量: {len(sensor_stats_df.index)}个")
    print(f"角度数量: {len(angle_stats_df.index)}个") 
    print(f"动作数量: {len(sensor_hierarchical.columns)}个")
    
    # 显示一些关键的ROM统计信息
    print(f"\n=== 角度数据ROM分析摘要 ===")
    
    # 获取ROM相关数据
    rom_data = angle_hierarchical.xs('运动范围ROM', level='统计指标')
    
    if not rom_data.empty:
        print("前5个角度的ROM统计 (跨所有动作):")
        for i, angle_name in enumerate(rom_data.index[:27]):
            angle_data = rom_data.loc[angle_name]
            valid_data = angle_data.dropna()
            if not valid_data.empty:
                avg_rom = valid_data.mean()
                max_rom = valid_data.max()
                min_rom = valid_data.min()
                print(f"{angle_name}: 平均ROM={avg_rom:.2f}°, 最大ROM={max_rom:.2f}°, 最小ROM={min_rom:.2f}°")
    
    print(f"\n详细结果请查看Excel文件: {excel_output}")
    print("文件包含两个sheet: '传感器数据' 和 '角度数据'")
    print("每个sheet中，行为传感器/角度名称，列为动作名称，子行为统计指标")

if __name__ == "__main__":
    main()
