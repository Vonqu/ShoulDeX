from datetime import datetime, timedelta
import pandas as pd
import re

def read_sensor_data(filepath: str):
    """
    读取传感器数据，自动跳过包含中文或异常内容的行，支持 utf-8 和 gbk 编码。
    返回两个 DataFrame：原始数据和（可选）处理版本
    """
    # 自动检测编码
    encodings = ['utf-8', 'gbk']
    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc) as file:
                lines = file.readlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("❌ 文件无法用 utf-8 或 gbk 编码读取")

    # ✅ 提取起始时间（第一行注释）
    start_time_line = lines[0].strip()
    match = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+", start_time_line)
    if match:
        start_time = datetime.strptime(match.group(), "%Y-%m-%d %H:%M:%S.%f")
    else:
        raise ValueError("❌ 无法在第一行中提取有效的 start_time")

    # ✅ 跳过前两行：注释 + 表头
    data_lines = lines[2:]
    data = []

    def contains_chinese(s):
        return any('\u4e00' <= c <= '\u9fff' for c in s)

    for line in data_lines:
        if contains_chinese(line):
            continue  # 跳过包含中文的行
        parts = line.strip().split(',')  # ✅ 按逗号分隔
        if len(parts) != 17:
            continue  # 跳过格式不对的行
        try:
            time_offset = float(parts[0])
            abs_time = start_time + timedelta(seconds=time_offset)
            values = list(map(float, parts[1:]))  # s1-s16
            data.append([abs_time] + values)
        except Exception as e:
            print(f"⚠️ 跳过异常行: {line.strip()}")
            continue

    #df = pd.DataFrame(data, columns=['time', 's1', 's2', 's3', 's4', 's5', 's6'])
    df = pd.DataFrame(data, columns=['time'] + [f's{i}' for i in range(1, 17)])
    
    # ✅ 确保 'time' 是 datetime 类型
    df['time'] = pd.to_datetime(df['time'])
    # ✅ 设置时间为 index
    df.set_index('time', inplace=True)
    df = df.iloc[:, 0:14]

    return df, df

# ✅ 示例运行（你可以修改路径为你的实际数据文件）
if __name__ == "__main__":
    path = './data\\raw_data\sensor\qnc\\normal\AB_fps30.csv'  # 修改为实际路径

    import os
    if not os.path.exists(path):
        print("❌ 文件路径不存在，请检查")
    else:
        df_raw, _ = read_sensor_data(path)
        print("✅ 成功读取前 5 行数据：")
        print(df_raw.head())
        print(f"\n📊 总数据行数: {len(df_raw)}")
