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
        self.heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(output_size)])

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = self.ln(x[:, -1, :])
        x = self.shared_fc(x)
        outputs = [head(x) for head in self.heads]
        return torch.cat(outputs, dim=1)
    

id_try = 'test_for_14angle_output'

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
output_size = 14
model = MultiHeadLSTM(input_size, hidden_size, num_layers, dropout, output_size).to(device)
model.load_state_dict(torch.load(f'./model/lstm_model_db{id_try}.pth', map_location=device))
model.eval()

# ✅ 4. 定义数据存储
window_length = 60
step_size = 3
buffer = deque(maxlen=window_length)
# 在前面初始化一个 CSV 文件
with open(f'./log/predicted_angles_log{id_try}.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Time (s)'] + [f'Angle_{i+1}' for i in range(output_size)])
# 加载 scaler 和 angle_scaler
with open(f'./predict/sensor_scaler{id_try}.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open(f'./predict/angle_scaler{id_try}.pkl', 'rb') as f:
    scaler_angle = pickle.load(f)

# ✅ 5. Matplotlib实时绘制预测角度
plt.ion()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# 定义每个subplot要显示的角度
# 你可以根据需要修改这些列表
subplot1_angles = [0, 1,4,5]      # 左上角subplot显示的角度
subplot2_angles = [2,3,6,7]      # 右上角subplot显示的角度  
subplot3_angles = [10,11]      # 左下角subplot显示的角度
subplot4_angles = [12,13]    # 右下角subplot显示的角度

# 将所有角度列表合并
all_draw_indices = subplot1_angles + subplot2_angles + subplot3_angles + subplot4_angles
predicted_angle_history = [[] for _ in range(output_size)]  # 保持18个角度的历史记录
predicted_x = []

# 为每个subplot创建对应的线条
subplot1_lines = [ax1.plot([], [], label=f'Angle {i+1}')[0] for i in subplot1_angles]
subplot2_lines = [ax2.plot([], [], label=f'Angle {i+1}')[0] for i in subplot2_angles]
subplot3_lines = [ax3.plot([], [], label=f'Angle {i+1}')[0] for i in subplot3_angles]
subplot4_lines = [ax4.plot([], [], label=f'Angle {i+1}')[0] for i in subplot4_angles]

# 设置每个subplot的属性
ax1.set_ylim(0, 180)
ax1.set_title("Left_Normal_Angles")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Angle")
ax1.legend()

ax2.set_ylim(0, 180)
ax2.set_title("Right_Normal Angles")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Angle")
ax2.legend()

ax3.set_ylim(0, 180)
ax3.set_title("Left_Compensatory_Angles")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Angle")
ax3.legend()

ax4.set_ylim(0, 180)
ax4.set_title("Right_Compensatory_Angles")
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Angle")
ax4.legend()

# 调整subplot之间的间距
plt.tight_layout()


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

            plt.pause(0.02)  # 稍微大一点，防止卡顿

            current_time = round(frame_counter / 30, 2)  # 假设串口每秒30帧，可根据实际帧率调整
            print(f"🎯 第 {current_time:.2f} 秒 预测真实角度: {[predicted_angles[i] for i in all_draw_indices]}")
            #print(f"🎯 第{frame_counter // step_size}秒预测真实角度: {predicted_angles}")

            # 写入CSV日志文件**
        with open(f'./log/predicted_angles_log_100epoch{id_try}.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([current_time] + list(predicted_angles))


    except KeyboardInterrupt:
        print("❌ 程序终止")
        break
    except Exception as e:
        print(f"⚠️ 发生错误: {e}")

ser.close()