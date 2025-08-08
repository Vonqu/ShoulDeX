import serial
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import torch.nn as nn
import time
import csv

# ✅ 模型定义

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout,output_size,):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 添加Layer Normalization
        self.ln = nn.LayerNorm(hidden_size)
        
        # 全连接层 - 包含一个隐藏层，增加非线性映射能力
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(hidden_size // 2, output_size)
            nn.Linear(hidden_size // 2, output_size)
            # 不再使用ReLU激活函数在最后一层，适用于回归任务
        )
        
        # 初始化LSTM权重
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM层前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 使用Layer Normalization
        out = self.ln(out[:, -1, :])
        
        # 全连接层前向传播
        out = self.fc(out)
        # out = out.view(out.size(0), 5, 10)  # Reshape output to (batch_size, 5, 10)    
        return out


trial_id = 'test_0806_1'

# ✅ 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ 1. 设置串口连接参数
SERIAL_PORT = "COM7"
BAUD_RATE = 115200
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
print(f"✅ 已连接到 {SERIAL_PORT}")

# ✅ 3. 加载 LSTM 预测模型
input_size = 14
hidden_size = 256
num_layers = 3
dropout = 0.1
output_size = 27
model = LSTM(input_size, hidden_size, num_layers, dropout, output_size).to(device)
model.load_state_dict(torch.load(f'./model/{trial_id}_lstm.pth', map_location=device))
model.eval()

# ✅ 4. 定义数据存储
window_length = 60
step_size = 5
buffer = deque(maxlen=window_length )
# 在前面初始化一个 CSV 文件
with open(f'./log/predict_log/predicted_angles_log_{trial_id}.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Time (s)'] + [f'Angle_{i+1}' for i in range(output_size)])
# 加载 scaler 和 angle_scaler
with open(f'./predict/scaler/sensor_scaler_{trial_id}.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open(f'./predict/scaler/angle_scaler_{trial_id}.pkl', 'rb') as f:
    scaler_angle = pickle.load(f)

# ✅ 5. Matplotlib实时绘制预测角度
plt.ion()
fig = plt.figure(figsize=(20, 12))

# 创建2行3列的子图布局，其中最后一个位置留空或用于其他用途
ax1 = plt.subplot(2, 3, 1)  # 第1行第1列
ax2 = plt.subplot(2, 3, 2)  # 第1行第2列
ax3 = plt.subplot(2, 3, 3)  # 第1行第3列
ax4 = plt.subplot(2, 3, 4)  # 第2行第1列
ax5 = plt.subplot(2, 3, 5)  # 第2行第2列

# 定义每个subplot要显示的角度（15个角度分成5组，每组3个）
# subplot1_angles = [0, 1, 2]      # 右肱骨与胸廓坐标系角度 XYZ
# subplot2_angles = [3, 4, 5]      # 右肩胛骨与胸廓坐标系角度 XYZ
# subplot3_angles = [6, 7, 8]      # 右锁骨与胸廓坐标系角度 XYZ
# subplot4_angles = [9, 10, 11]    # 右肱骨与右肩胛骨坐标系角度 XYZ
# subplot5_angles = [12, 13, 14]   # 胸廓与髋关节坐标系角度 XYZ

subplot1_angles = [0, 1, 2,3, 4, 5]      # 右肱骨与胸廓坐标系角度 XYZ
subplot2_angles = [6, 7, 8,9, 10, 11]      # 右肩胛骨与胸廓坐标系角度 XYZ
subplot3_angles = [12,13 ,14,15,16,17]      # 右锁骨与胸廓坐标系角度 XYZ
subplot4_angles = [18,19,20,21,22,23]    # 右肱骨与右肩胛骨坐标系角度 XYZ
subplot5_angles = [24,25,26]   # 胸廓与髋关节坐标系角度 XYZ

# 将所有角度列表合并
all_draw_indices = subplot1_angles + subplot2_angles + subplot3_angles + subplot4_angles + subplot5_angles

# 初始化历史记录（15个角度）
predicted_angle_history = [[] for _ in range(27)]  # 修改为15个角度
predicted_x = []

# 定义颜色和线型，使每个子图内的3条线容易区分
colors = ['red', 'blue', 'green']
line_styles = ['-', '--', '-.']


# 为每个subplot创建对应的线条
subplot1_lines = [ax1.plot([], [], color=colors[i], linestyle=line_styles[i], 
                          label=f'Humerus_R_Thorax_{["X", "Y", "Z"][i]}')[0] for i in range(3)]

subplot2_lines = [ax2.plot([], [], color=colors[i], linestyle=line_styles[i], 
                          label=f'Scapula_R_Thorax_{["X", "Y", "Z"][i]}')[0] for i in range(3)]
subplot3_lines = [ax3.plot([], [], color=colors[i], linestyle=line_styles[i], 
                          label=f'Clavicle_R_Thorax_{["X", "Y", "Z"][i]}')[0] for i in range(3)]
subplot4_lines = [ax4.plot([], [], color=colors[i], linestyle=line_styles[i], 
                          label=f'Humerus_R_Scapula_{["X", "Y", "Z"][i]}')[0] for i in range(3)]
subplot5_lines = [ax5.plot([], [], color=colors[i], linestyle=line_styles[i], 
                          label=f'Thorax_Hip_{["X", "Y", "Z"][i]}')[0] for i in range(3)]

# 设置每个subplot的属性
ax1.set_ylim(0, 180)
ax1.set_title("right_humerus_throax", fontsize=12)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Angle (°)")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2.set_ylim(0, 180)
ax2.set_title("right_scapula_throax", fontsize=12)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Angle (°)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

ax3.set_ylim(0, 180)
ax3.set_title("right_clavies_throax", fontsize=12)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Angle (°)")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

ax4.set_ylim(0, 180)
ax4.set_title("right_humerus_scapula", fontsize=12)
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Angle (°)")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

ax5.set_ylim(0, 180)
ax5.set_title("throax_hip", fontsize=12)
ax5.set_xlabel("Time (s)")
ax5.set_ylabel("Angle (°)")
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# 调整subplot之间的间距
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.3)

print("🚀 开始实时预测")
frame_counter = 0
sensor_check_counter = 0


# ✅ 6. 实时读取数据、预测、绘图
while True:
    try:
        line = ser.readline().decode("latin1", errors="ignore").strip()
        if not line:
            continue

        values = np.array(line.split(","), dtype=float)

        if len(values) != 14:
            print(f"⚠️ 数据格式错误: {values}")
            continue

        buffer.append(values)
        frame_counter += 1

        # 打印前100条数据检查传感器
        if sensor_check_counter < 100:
            print(f"🛠️ 传感器数据 ({sensor_check_counter+1}/100): {values}")
            sensor_check_counter += 1

        # 每10帧（1/3秒）执行一次预测
        if len(buffer) == window_length and frame_counter % step_size == 0:
            input_data = scaler.transform(np.array(buffer))
            input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
            input_tensor = input_tensor.unsqueeze(0)


            start_time = time.perf_counter()  # ⏱️ 开始计时
            with torch.no_grad():
              predicted_angles_norm = model(input_tensor).cpu().numpy()
            end_time = time.perf_counter()  # ⏱️ 结束计时

            elapsed_time = (end_time - start_time) * 1000  # 单位毫秒
            print(f"🕒 本次预测耗时: {elapsed_time:.2f} ms")


            with torch.no_grad():
                predicted_angles_norm = model(input_tensor).cpu().numpy()

            # 角度反归一化
            predicted_angles = scaler_angle.inverse_transform(predicted_angles_norm)[0]

            predicted_x.append(frame_counter // step_size)
            # 更新所有18个角度的历史记录
            for i in range(output_size):
                predicted_angle_history[i].append(predicted_angles[i])

            # 控制绘图数据长度，只保留最近300帧
            if len(predicted_x) > 200:
                predicted_x = predicted_x[-200:]
                for i in range(output_size):
                    predicted_angle_history[i] = predicted_angle_history[i][-200:]

            # 更新绘图
            for i, line in enumerate(subplot1_lines):
                line.set_xdata(predicted_x)
                line.set_ydata(predicted_angle_history[subplot1_angles[i]])
            for i, line in enumerate(subplot2_lines):
                line.set_xdata(predicted_x)
                line.set_ydata(predicted_angle_history[subplot2_angles[i]])
            for i, line in enumerate(subplot3_lines):
                line.set_xdata(predicted_x)
                line.set_ydata(predicted_angle_history[subplot3_angles[i]])
            for i, line in enumerate(subplot4_lines):
                line.set_xdata(predicted_x)
                line.set_ydata(predicted_angle_history[subplot4_angles[i]])

            # 只在数据长度变化时重设xlim
            if len(predicted_x) < 200:
                ax1.set_xlim(0, 200)
                ax2.set_xlim(0, 200)
                ax3.set_xlim(0, 200)
                ax4.set_xlim(0, 200)
            else:
                ax1.set_xlim(predicted_x[0], predicted_x[-1])
                ax2.set_xlim(predicted_x[0], predicted_x[-1])
                ax3.set_xlim(predicted_x[0], predicted_x[-1])
                ax4.set_xlim(predicted_x[0], predicted_x[-1])

            # 只在第一次或窗口变化时relim/autoscale_view
            # ax.relim()
            # ax.autoscale_view()

            plt.pause(0.005)  # 稍微大一点，防止卡顿

            current_time = round(frame_counter / 30, 2)  # 假设串口每秒30帧，可根据实际帧率调整
            print(f"🎯 第 {current_time:.2f} 秒 预测真实角度: {[predicted_angles[i] for i in all_draw_indices]}")
            #print(f"🎯 第{frame_counter // step_size}秒预测真实角度: {predicted_angles}")

            # 写入CSV日志文件**
        with open(f'./log/predict_log/predicted_angles_{trial_id}.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([current_time] + list(predicted_angles))


    except KeyboardInterrupt:
        print("❌ 程序终止")
        break
    except Exception as e:
        print(f"⚠️ 发生错误: {e}")

ser.close()