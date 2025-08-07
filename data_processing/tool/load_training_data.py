
import os
import glob
import pandas as pd


motion_type = 'normal'
test_or_final = 'test'

train_folder = f'./data\model_training\\train'  # 文件夹路径
train_csv_files = glob.glob(os.path.join(train_folder, '*.csv'))
data_train_list = [pd.read_csv(f) for f in train_csv_files]
print(train_csv_files)
data_train = pd.concat(data_train_list, ignore_index=True)
print(f"训练集文件数: {len(train_csv_files)}, 合并后shape: {data_train.shape}")

test_folder = f'./data\model_training\\test'  # 文件夹路径
test_csv_files = glob.glob(os.path.join(test_folder, '*.csv'))
data_test_list = [pd.read_csv(f) for f in test_csv_files]
data_test = pd.concat(data_test_list, ignore_index=True)
print(f"测试集文件数: {len(test_csv_files)}, 合并后shape: {data_test.shape}")

# 自动获取传感器和角度列名
sensor_cols = data_train.columns[1:15]   # 第2~15列
angle_cols = data_train.columns[15:42]   # 第16~42列（27个角度）

sensor_data_train = data_train[sensor_cols].values
angle_data_train = data_train[angle_cols].values

sensor_data_test = data_test[sensor_cols].values
angle_data_test = data_test[angle_cols].values

print("传感器列名:", list(sensor_cols))
print("角度列名:", list(angle_cols))

sensor_number = sensor_data_train.shape[1]
angle_number = angle_data_train.shape[1]

print(f"Number of sensors (input size): {sensor_number}")
print(f"Number of angle (output size): {angle_number}")

# # Normalize the data
# from sklearn import MinMaxScaler
# scaler = MinMaxScaler()
# sensor_data_train = scaler.fit_transform(sensor_data_train)
# angle_data_train = scaler.fit_transform(angle_data_train)
# sensor_data_test = scaler.fit_transform(sensor_data_test)
# angle_data_test = scaler.fit_transform(angle_data_test)