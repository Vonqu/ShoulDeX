import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from predict_utilis import predict_by_batch
import pickle
import math

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 模型定义
#class LSTM(nn.Module):
#    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
#        super(LSTM, self).__init__()
#        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#        self.ln = nn.LayerNorm(hidden_size)
#        self.fc = nn.Sequential(
#            nn.Linear(hidden_size, 128),
#            nn.ReLU(),
#            nn.Dropout(dropout),
#            nn.Linear(128, output_size)
#        )

#    def forward(self, x):
#        if x.dim() == 2:
#            x = x.unsqueeze(1)
#        x, _ = self.lstm(x)
#        x = self.ln(x[:, -1, :])
#        x = self.fc(x)
#        return x

# ✅ Multi-Head LSTM
class MultiHeadLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super(MultiHeadLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.ln = nn.LayerNorm(hidden_size)
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # 每个角度一个输出头
        self.heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(output_size)])

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = self.ln(x[:, -1, :])
        x = self.shared_fc(x)
        outputs = [head(x) for head in self.heads]  # 每个输出 shape: [batch_size, 1]
        return torch.cat(outputs, dim=1)            # 拼成 [batch_size, 9]

## ✅ Multi-Head LSTM with per-head Input Projection
#class MultiHeadLSTM(nn.Module):
#    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
#        super(MultiHeadLSTM, self).__init__()
#        self.output_size = output_size
#        self.input_projections = nn.ModuleList([
#            nn.Linear(input_size, input_size) for _ in range(output_size)
#        ])

#        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#        self.ln = nn.LayerNorm(hidden_size)

#        self.shared_fc = nn.Sequential(
#            nn.Linear(hidden_size, 128),
#            nn.ReLU(),
#            nn.Dropout(dropout),
#        )

#        self.heads = nn.ModuleList([
#            nn.Linear(128, 1) for _ in range(output_size)
#        ])

#    def forward(self, x):  # x: [B, T, C]
#        outputs = []
#        for i in range(self.output_size):
#            x_proj = self.input_projections[i](x)         # 每个角度使用独立投影后的输入
#            x_lstm, _ = self.lstm(x_proj)                  # [B, T, H]
#            x_last = self.ln(x_lstm[:, -1, :])             # [B, H]
#            x_feat = self.shared_fc(x_last)                # [B, 128]
#            out = self.heads[i](x_feat)                    # [B, 1]
#            outputs.append(out)
#        return torch.cat(outputs, dim=1)  # [B, output_size]

id_try = 'double_side_27_angles'
# 加载数据
#data_folder = './motion_0407/2/all'
#data_folder = './motion_0407/2/all-clear'
#data_folder = './motion_0407/rdm/alls'
#data_folder = './motion_0407/alls'
#data_folder = './motion_0407/2/zs+mjq'
#data_folder = './motion_0407/2/zss/clear'
#data_folder = './motion_0710/alls'
data_folder = './data\model_training\\train'
# data_folder = './data\dft\qnc\mixed\\train'
csv_files = glob.glob(os.path.join(data_folder, '*.csv'))
random.shuffle(csv_files)
print(f"Found {len(csv_files)} CSV files.")

data_list = []
for file in csv_files:
    df = pd.read_csv(file)
#     #expected_sensor_cols = ['s1', 's2', 's3', 's4', 's5', 's6']
    # expected_sensor_cols = [f's{i}' for i in range(1, 15)]  # s1 到 s14
    # # expected_sensor_cols = df[]

    # # expected_angle_cols = [f'angle{i}' for i in range(1, 19)]  # angle1 到 angle14
    # expected_angle_cols = df[['Frame', 'Time',
    #                  # Angle 1-3: 左肱骨与胸廓坐标系角度
    #                  'A_humerus_l_thorax_X', 'A_humerus_l_thorax_Y', 'A_humerus_l_thorax_Z',
    #                  # Angle 4-6: 右肱骨与胸廓坐标系角度
    #                  'A_humerus_r_thorax_X', 'A_humerus_r_thorax_Y', 'A_humerus_r_thorax_Z',
    #                  # Angle 7-9: 左肩胛骨与胸廓坐标系角度
    #                  'A_scapula_l_thorax_X', 'A_scapula_l_thorax_Y', 'A_scapula_l_thorax_Z',
    #                  # Angle 10-12: 右肩胛骨与胸廓坐标系角度
    #                  'A_scapula_r_thorax_X', 'A_scapula_r_thorax_Y', 'A_scapula_r_thorax_Z',
    #                  # Angle 13-15: 左锁骨与胸廓坐标系角度
    #                  'A_clavicle_l_thorax_X', 'A_clavicle_l_thorax_Y', 'A_clavicle_l_thorax_Z',
    #                  # Angle 16-18: 右锁骨与胸廓坐标系角度
    #                  'A_clavicle_r_thorax_X', 'A_clavicle_r_thorax_Y', 'A_clavicle_r_thorax_Z',
    #                  # Angle 19-21: 左肱骨与左肩胛骨坐标系角度
    #                  'A_humerus_l_scapula_X', 'A_humerus_l_scapula_Y', 'A_humerus_l_scapula_Z',
    #                  # Angle 22-24: 右肱骨与右肩胛骨坐标系角度
    #                  'A_humerus_r_scapula_X', 'A_humerus_r_scapula_Y', 'A_humerus_r_scapula_Z',
    #                  # Angle 25-27: 胸廓与髋关节坐标系角度
    #                  'A_thorax_hip_X', 'A_thorax_hip_Y', 'A_thorax_hip_Z']]
    # missing_cols = [col for col in expected_sensor_cols + expected_angle_cols if col not in df.columns]
    # if missing_cols:
    
    # 检查数据列数
    if len(df.columns) < 42:  # 1列时间 + 14列传感器 + 27列角度 = 42列
        print(f"Warning: {file} has insufficient columns: {len(df.columns)}")
        continue
    
    # 获取传感器列（第2-15列，索引1-14）
    expected_sensor_cols = df.columns[1:15].tolist()  # s1 到 s14
    
    # 获取角度列（第16-42列，索引15-41）
    expected_angle_cols = df.columns[15:42].tolist()  # 27个角度列
    
    # 检查列名是否包含预期的关键词
    if not all('s' in col for col in expected_sensor_cols):
        print(f"Warning: {file} sensor columns may not be correct: {expected_sensor_cols}")
        continue
    
    if not all(any(keyword in col for keyword in ['A_', 'angle']) for col in expected_angle_cols):
        print(f"Warning: {file} angle columns may not be correct: {expected_angle_cols}")
        continue
    
    data_list.append(df)

data_all = pd.concat(data_list, ignore_index=True)
print(f"Combined data shape: {data_all.shape}")


sensor_data = data_all[expected_sensor_cols].values
angle_data = data_all[expected_angle_cols].values


# 检查并清除 NaN / Inf
sensor_data = np.nan_to_num(sensor_data, nan=0.0, posinf=0.0, neginf=0.0)
angle_data = np.nan_to_num(angle_data, nan=0.0, posinf=0.0, neginf=0.0)

# 滑动窗口函数
def create_dataset(X, y, window_length, time_steps):
    Xs, ys = [], []
    for i in range(int((len(X) - window_length) / time_steps)):
        Xs.append(X[i * time_steps: (i * time_steps + window_length)])
        ys.append(y[i * time_steps + window_length])
    return np.array(Xs), np.array(ys)

# 参数设置
window_length = 60
time_steps = 3
#window_length = 250
#time_steps = 5

X_all_np, y_all_np = create_dataset(sensor_data, angle_data, window_length, time_steps)
print(f"X_all shape: {X_all_np.shape}, y_all shape: {y_all_np.shape}")

# 时间划分
split_index = int(len(X_all_np) * 0.8)
X_train_np, y_train_np = X_all_np[:split_index], y_all_np[:split_index]
X_val_np, y_val_np = X_all_np[split_index:], y_all_np[split_index:]

# 标准化器仅在训练集上 fit
#scaler_sensor = StandardScaler().fit(X_train_np.reshape(-1, 6))
scaler_sensor = StandardScaler().fit(X_train_np.reshape(-1, 14))
scaler_angle = StandardScaler().fit(y_train_np)

# 检查标准差为 0 的列（会导致除以 0）
if np.any(scaler_sensor.scale_ == 0):
    raise ValueError("⚠️ 传感器数据中存在标准差为 0 的列，无法标准化！")

# 保存 scaler
with open(f'./predict/sensor_scaler{id_try}.pkl', 'wb') as f:
    pickle.dump(scaler_sensor, f)
with open(f'./predict/angle_scaler{id_try}.pkl', 'wb') as f:
    pickle.dump(scaler_angle, f)

# 标准化数据
#X_train_np = scaler_sensor.transform(X_train_np.reshape(-1, 6)).reshape(-1, window_length, 6)
#X_val_np = scaler_sensor.transform(X_val_np.reshape(-1, 6)).reshape(-1, window_length, 6)
X_train_np = scaler_sensor.transform(X_train_np.reshape(-1, 14)).reshape(-1, window_length, 14)
X_val_np = scaler_sensor.transform(X_val_np.reshape(-1, 14)).reshape(-1, window_length, 14)
y_train_np = scaler_angle.transform(y_train_np)
y_val_np = scaler_angle.transform(y_val_np)

# 转为张量
X_train = torch.from_numpy(X_train_np).float().to(device)
y_train = torch.from_numpy(y_train_np).float().to(device)
X_val = torch.from_numpy(X_val_np).float().to(device)
y_val = torch.from_numpy(y_val_np).float().to(device)

print(f"Train sequences: {X_train.shape[0]}, Validation sequences: {X_val.shape[0]}")

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256, shuffle=False)

# 模型与优化器
input_size = X_train.shape[-1] #应该是14
output_size = y_train.shape[1] #应该是18
print(f"Input size: {input_size}, Output size: {output_size}")
#model = LSTM(input_size, hidden_size=256, num_layers=3, output_size=output_size, dropout=0.1).to(device)
model = MultiHeadLSTM(input_size, hidden_size=256, num_layers=3, dropout=0.1, output_size=output_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# 训练
train_losses, val_losses = [], []
for epoch in range(70):
    model.train()
    train_loss_sum = 0
    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/10"):
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch) + 0.0003 * sum(torch.norm(p, 2) for p in model.parameters())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 🔧 梯度裁剪
        optimizer.step()
        train_loss_sum += loss.item()
    train_losses.append(train_loss_sum / len(train_loader))

    #model.eval()
    #val_loss_sum = 0
    #with torch.no_grad():
    #    for X_batch, y_batch in val_loader:
    #        preds = model(X_batch)
    #        loss = criterion(preds, y_batch)
    #        val_loss_sum += loss.item()
    #val_losses.append(val_loss_sum / len(val_loader))
    #print(f"[Epoch {epoch+1}] Train Loss: {train_losses[-1]:.6f} | Val Loss: {val_losses[-1]:.6f}")
    #scheduler.step()

        # 验证阶段
    model.eval()
    val_loss_sum = 0
    per_angle_losses = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            preds = model(X_batch)

            # 原始整体 loss
            loss = criterion(preds, y_batch)
            val_loss_sum += loss.item()

            # 👉 每个角度单独 loss
            angle_losses = [nn.functional.mse_loss(preds[:, i], y_batch[:, i]).item() for i in range(output_size)]
            per_angle_losses.append(angle_losses)

    val_losses.append(val_loss_sum / len(val_loader))

    # 👉 打印每个角度的平均验证 loss
    mean_angle_losses = np.mean(per_angle_losses, axis=0)
    print(f"[Epoch {epoch+1}] Train Loss: {train_losses[-1]:.6f} | Val Loss: {val_losses[-1]:.6f}")
    for i, l in enumerate(mean_angle_losses):
        print(f"📐 Angle {i+1} Val Loss: {l:.4f}")

    scheduler.step()

    

# RMSE 函数
def compute_rmse(y_true, y_pred):
    return [np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])) for i in range(y_true.shape[1])]

# 训练集评估
model.eval()
train_preds = predict_by_batch(model, X_train, batch_size=256)
train_true = y_train.detach().cpu().numpy()
train_preds_denorm = scaler_angle.inverse_transform(train_preds)
train_true_denorm = scaler_angle.inverse_transform(train_true)
rmse_train = compute_rmse(train_true_denorm, train_preds_denorm)
print("\n🔁 Train RMSE (degrees):", rmse_train)
print(f"🎯 Average Train RMSE: {np.mean(rmse_train):.4f}")

# 验证集评估
val_preds = predict_by_batch(model, X_val, batch_size=256)
val_true = y_val.detach().cpu().numpy()
val_preds_denorm = scaler_angle.inverse_transform(val_preds)
val_true_denorm = scaler_angle.inverse_transform(val_true)
rmse_val = compute_rmse(val_true_denorm, val_preds_denorm)
print("\n🧪 Validation RMSE (degrees):", rmse_val)
print(f"📊 Average Validation RMSE: {np.mean(rmse_val):.4f}")

# 保存模型
# torch.save(model.state_dict(), "lstm_model_db.ckpt")
model_path = f"./model/lstm_model_db{id_try}.pth"
torch.save(model.state_dict(), model_path)
print("\n✅ 模型已保存")

# 损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', linestyle='--')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.show()

# 训练集预测图
#fig, axes = plt.subplots(6, 3, figsize=(18, 12))  # 6行3列 = 18个图
num_plots = 27
# 自动确定列数（你也可以手动指定列数）
cols = 3  # 例如一行放4个图，够紧凑
rows = math.ceil(num_plots / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3))  # 动态尺寸
axes = axes.flatten()  # 转成一维方便访问

for i in range(train_true.shape[1]):
    ax = axes.flat[i]
    ax.plot(train_true_denorm[:, i], label='Actual', color='blue')
    ax.plot(train_preds_denorm[:, i], label='Predicted', color='orange')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Angle')
    ax.set_title(f'Train - Angle {i+1}')
plt.tight_layout()
plt.show()

# 验证集预测图
#fig, axes = plt.subplots(6, 3, figsize=(18, 12))  # 6 rows × 3 columns = 18 plots
num_plots = 27 # 自动确定列数（你也可以手动指定列数）
cols = 3  # 例如一行放4个图，够紧凑
rows = math.ceil(num_plots / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3))  # 动态尺寸
axes = axes.flatten()  # 转成一维方便访问

for i in range(val_true.shape[1]):
    ax = axes.flat[i]
    ax.plot(val_true_denorm[:, i], label='Actual', color='blue')
    ax.plot(val_preds_denorm[:, i], label='Predicted', color='orange')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Angle')
    ax.set_title(f'Test - Angle {i+1}')
    ax.legend()
plt.tight_layout()
plt.show()

# =============================
# ▶️ 测试集推理 & 评估
# =============================

#test_folder = './motion_0407/2/mjq/clear'
#test_folder = './motion_0407/2/all-clear'
#test_folder = './motion_0407/rdm/slla'
#test_folder = './motion_0710/slla'
test_folder = './data/model_training/test'
#test_folder = './motion_0407/rdm/slla/clear'
#test_folder = './motion_0407/2/qnc'
#test_folder = './motion_0407/2/zss/clear'
test_csv_files = glob.glob(os.path.join(test_folder, '*.csv'))
print(f"\n🧪 Found {len(test_csv_files)} test CSV files.")

# 加载并合并测试数据
test_list = []
for file in test_csv_files:
    df = pd.read_csv(file)
    
    # 检查数据列数
    if len(df.columns) < 42:  # 1列时间 + 14列传感器 + 27列角度 = 42列
        print(f"Warning: {file} has insufficient columns: {len(df.columns)}")
        continue
    
    # 获取传感器列（第2-15列，索引1-14）
    expected_sensor_cols = df.columns[1:15].tolist()  # s1 到 s14
    
    # 获取角度列（第16-42列，索引15-41）
    expected_angle_cols = df.columns[15:42].tolist()  # 27个角度列
    
    # 检查列名是否包含预期的关键词
    if not all('s' in col for col in expected_sensor_cols):
        print(f"Warning: {file} sensor columns may not be correct: {expected_sensor_cols}")
        continue
    
    if not all(any(keyword in col for keyword in ['A_', 'angle']) for col in expected_angle_cols):
        print(f"Warning: {file} angle columns may not be correct: {expected_angle_cols}")
        continue
    
    test_list.append(df)

if len(test_list) == 0:
    raise ValueError("❌ No valid test files found.")

test_data = pd.concat(test_list, ignore_index=True)
sensor_test = test_data[expected_sensor_cols].values
angle_test = test_data[expected_angle_cols].values

# 清洗 NaN / Inf
sensor_test = np.nan_to_num(sensor_test, nan=0.0, posinf=0.0, neginf=0.0)
angle_test = np.nan_to_num(angle_test, nan=0.0, posinf=0.0, neginf=0.0)

# 创建滑动窗口
X_test_np, y_test_np = create_dataset(sensor_test, angle_test, window_length, time_steps)

# 加载训练时保存的 scaler
with open(f'./predict/sensor_scaler{id_try}.pkl', 'rb') as f:
    scaler_sensor = pickle.load(f)
with open(f'./predict/angle_scaler{id_try}.pkl', 'rb') as f:
    scaler_angle = pickle.load(f)

# 标准化测试集（注意：使用训练集的 scaler）
X_test_np = scaler_sensor.transform(X_test_np.reshape(-1, 14)).reshape(-1, window_length, 14)
y_test_np = scaler_angle.transform(y_test_np)

# 转为张量
X_test = torch.from_numpy(X_test_np).float().to(device)
y_test = torch.from_numpy(y_test_np).float().to(device)

## 加载训练好的模型
#model = LSTM(input_size, hidden_size=256, num_layers=3, output_size=output_size, dropout=0.1).to(device)
#model.load_state_dict(torch.load("lstm_model_SingleAction_timeSplit.pth"))
#model.eval()

# 加载训练好的 MultiHead 模型
model = MultiHeadLSTM(input_size, hidden_size=256, num_layers=3, dropout=0.1, output_size=output_size).to(device)
model.load_state_dict(torch.load(f"./model/lstm_model_db{id_try}.pth"))
model.eval()


# 测试集预测
test_preds = predict_by_batch(model, X_test, batch_size=256)
test_true = y_test.detach().cpu().numpy()

# 反标准化
test_preds_denorm = scaler_angle.inverse_transform(test_preds)
test_true_denorm = scaler_angle.inverse_transform(test_true)

## 计算 RMSE
#rmse_test = compute_rmse(test_true_denorm, test_preds_denorm)
#avg_rmse_test = np.mean(rmse_test)

#print("\n🧪 Test RMSE (degrees):", rmse_test)
#print(f"🎯 Average Test RMSE: {avg_rmse_test:.4f}")

# 计算每个角度的 RMSE
rmse_test = compute_rmse(test_true_denorm, test_preds_denorm)
avg_rmse_test = np.mean(rmse_test)

print("\n🧪 Test RMSE (degrees):", rmse_test)
for i, rmse in enumerate(rmse_test):
    print(f"📐 Angle {i+1} Test RMSE: {rmse:.4f}")
print(f"🎯 Average Test RMSE: {avg_rmse_test:.4f}")


# 可视化测试集预测结果
#fig, axes = plt.subplots(6, 3, figsize=(18, 12))
num_plots = 27
# 自动确定列数（你也可以手动指定列数）
cols = 3  # 例如一行放4个图，够紧凑
rows = math.ceil(num_plots / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3))  # 动态尺寸
axes = axes.flatten()  # 转成一维方便访问
for i in range(test_true.shape[1]):
    ax = axes.flat[i]
    ax.plot(test_true_denorm[:, i], label='Actual', color='blue')
    ax.plot(test_preds_denorm[:, i], label='Predicted', color='orange')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Angle')
    ax.set_title(f'Test - Angle {i+1}')
    ax.legend()
plt.tight_layout()
plt.show()