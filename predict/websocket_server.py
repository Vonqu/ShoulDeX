#!/usr/bin/env python3
"""
WebSocket服务器 - 实时角度预测
基于predict_ds_14Sto27A.py实现的实时预测系统
"""

import asyncio
import websockets
import json
import serial
import threading
import queue
import time
import torch
import pickle
import numpy as np
from collections import deque

import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 添加Layer Normalization
        self.ln = nn.LayerNorm(hidden_size)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
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
        return out 

class AnglePredictionServer:
    def __init__(self, trial_id='test_0806_1'):
        """初始化服务器"""
        # 基本配置
        self.trial_id = trial_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 模型参数 - 与predict_ds_14Sto27A.py保持一致
        self.input_size = 14
        self.hidden_size = 256
        self.num_layers = 3
        self.dropout = 0.1
        self.output_size = 27
        
        # 数据处理参数 - 与predict_ds_14Sto27A.py保持一致
        self.window_length = 60
        self.step_size = 5
        self.buffer = deque(maxlen=self.window_length)
        
        # 串口配置 - 与predict_ds_14Sto27A.py保持一致
        self.SERIAL_PORT = "COM7"
        self.BAUD_RATE = 115200
        self.ser = None
        
        # 运行状态
        self.running = False
        self.prediction_queue = queue.Queue()
        self.frame_counter = 0
        self.sensor_check_counter = 0
        
        # WebSocket客户端管理
        self.clients = set()
        
        # 加载模型和数据处理器
        self._load_model_and_scalers()
        
    def _load_model_and_scalers(self):
        """加载模型和数据处理器"""
        try:
            # 加载LSTM模型 - 使用相对路径
            self.model = LSTM(self.input_size, self.hidden_size, self.num_layers, 
                             self.dropout, self.output_size).to(self.device)
            model_path = f'F:\coding\ShoulDeX\ShoulDex_v1.0\ShoulDeX\model/{self.trial_id}_lstm.pth'
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            # 加载scaler
            with open(f'F:\coding\ShoulDeX\ShoulDex_v1.0\ShoulDeX\predict\scaler/sensor_scaler_{self.trial_id}.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(f'F:\coding\ShoulDeX\ShoulDex_v1.0\ShoulDeX\predict\scaler/angle_scaler_{self.trial_id}.pkl', 'rb') as f:
                self.scaler_angle = pickle.load(f)
                
            print("[SUCCESS] 模型和数据处理器加载成功")
            
        except Exception as e:
            print(f"[ERROR] 模型或数据处理器加载失败: {e}")
            raise
            
    def _init_serial(self):
        """初始化串口连接"""
        try:
            self.ser = serial.Serial(self.SERIAL_PORT, self.BAUD_RATE, timeout=1)
            print(f"[SUCCESS] 串口连接成功: {self.SERIAL_PORT}")
            return True
        except Exception as e:
            print(f"[ERROR] 串口连接失败: {e}")
            return False
            
    def serial_data_thread(self):
        """串口数据读取线程 - 基于predict_ds_14Sto27A.py"""
        if not self._init_serial():
            return
            
        print("[INFO] 开始读取串口数据...")
        self.frame_counter = 0
        self.sensor_check_counter = 0
        
        while self.running:
            try:
                if not self.ser.is_open:
                    print("[ERROR] 串口已断开")
                    break
                    
                # 读取数据 - 与predict_ds_14Sto27A.py完全一致
                line = self.ser.readline().decode("latin1", errors="ignore").strip()
                if not line:
                    continue

                values = np.array(line.split(","), dtype=float)

                if len(values) != 14:
                    print(f"⚠️ 数据格式错误: {values}")
                    continue

                self.buffer.append(values)
                self.frame_counter += 1

                # 打印前100条数据检查传感器 - 与predict_ds_14Sto27A.py一致
                if self.sensor_check_counter < 100:
                    print(f"🛠️ 传感器数据 ({self.sensor_check_counter+1}/100): {values}")
                    self.sensor_check_counter += 1

                # 每step_size帧执行一次预测 - 与predict_ds_14Sto27A.py一致
                if len(self.buffer) == self.window_length and self.frame_counter % self.step_size == 0:
                    self._make_prediction()
                    
            except Exception as e:
                print(f"[ERROR] 串口读取错误: {e}")
                time.sleep(0.1)
                
        # 关闭串口
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[INFO] 串口已关闭")
            
    def _make_prediction(self):
        """进行预测 - 基于predict_ds_14Sto27A.py"""
        try:
            # 数据预处理 - 与predict_ds_14Sto27A.py完全一致
            input_data = self.scaler.transform(np.array(self.buffer))
            input_tensor = torch.tensor(input_data, dtype=torch.float32).to(self.device)
            input_tensor = input_tensor.unsqueeze(0)

            # 预测时间测量
            start_time = time.perf_counter()
            with torch.no_grad():
                predicted_angles_norm = self.model(input_tensor).cpu().numpy()
            end_time = time.perf_counter()

            elapsed_time = (end_time - start_time) * 1000  # 单位毫秒
            print(f"🕒 本次预测耗时: {elapsed_time:.2f} ms")

            # 角度反归一化 - 与predict_ds_14Sto27A.py一致
            predicted_angles = self.scaler_angle.inverse_transform(predicted_angles_norm)[0]

            # 计算当前时间 - 与predict_ds_14Sto27A.py一致
            current_time = round(self.frame_counter / 30, 2)  # 假设串口每秒30帧
            
            # 准备发送数据
            prediction_data = {
                'type': 'prediction',
                'timestamp': current_time,
                'angles': predicted_angles.tolist(),
                'prediction_time_ms': elapsed_time,
                'frame_counter': self.frame_counter
            }
            
            # 将预测结果放入队列
            self.prediction_queue.put(prediction_data)
            
            # 打印预测结果 - 与predict_ds_14Sto27A.py类似格式
            all_draw_indices = list(range(27))  # 显示所有27个角度
            print(f"🎯 第 {current_time:.2f} 秒 预测真实角度: {[predicted_angles[i] for i in all_draw_indices]}")
            
        except Exception as e:
            print(f"[ERROR] 预测错误: {e}")
            
    async def websocket_handler(self, websocket):
        """WebSocket连接处理 - 兼容新版本websockets"""
        print(f"[INFO] 客户端连接: {websocket.remote_address}")
        
        # 添加客户端到集合
        self.clients.add(websocket)
        
        try:
            # 发送连接确认
            await websocket.send(json.dumps({
                'type': 'connection',
                'message': 'Connected to Angle Prediction Server',
                'trial_id': self.trial_id,
                'device': str(self.device),
                'output_size': self.output_size
            }))
            
            # 处理客户端消息
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get('type') == 'start_prediction':
                        print("[INFO] 收到开始预测请求")
                        await self.start_prediction()
                    elif data.get('type') == 'stop_prediction':
                        print("[INFO] 收到停止预测请求")
                        await self.stop_prediction()
                except json.JSONDecodeError:
                    print(f"[WARNING] 无效的JSON消息: {message}")
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"[INFO] 客户端断开连接: {websocket.remote_address}")
        except Exception as e:
            print(f"[ERROR] WebSocket处理错误: {e}")
        finally:
            # 从客户端集合中移除
            self.clients.discard(websocket)
            
    async def broadcast_predictions(self):
        """广播预测数据到所有连接的客户端"""
        while self.running:
            try:
                # 从队列获取预测数据
                data = self.prediction_queue.get(timeout=1)
                
                # 广播到所有客户端
                if self.clients:
                    disconnected_clients = set()
                    for client in self.clients:
                        try:
                            await client.send(json.dumps(data))
                        except websockets.exceptions.ConnectionClosed:
                            disconnected_clients.add(client)
                        except Exception as e:
                            print(f"[ERROR] 发送数据到客户端失败: {e}")
                            disconnected_clients.add(client)
                    
                    # 移除断开的客户端
                    self.clients -= disconnected_clients
                    
            except queue.Empty:
                # 发送心跳包
                if self.clients:
                    heartbeat_data = {'type': 'heartbeat', 'timestamp': time.time()}
                    disconnected_clients = set()
                    for client in self.clients:
                        try:
                            await client.send(json.dumps(heartbeat_data))
                        except:
                            disconnected_clients.add(client)
                    self.clients -= disconnected_clients
                    
            except Exception as e:
                print(f"[ERROR] 广播数据错误: {e}")
            
            await asyncio.sleep(0.01)  # 短暂等待

    async def start_prediction(self):
        """开始预测"""
        if not self.running:
            print("[INFO] 启动预测服务")
            self.running = True
            # 广播开始状态给所有客户端
            if self.clients:
                message = json.dumps({
                    'type': 'status',
                    'status': 'started',
                    'message': '预测已开始'
                })
                await asyncio.gather(
                    *[client.send(message) for client in self.clients.copy()],
                    return_exceptions=True
                )
        
    async def stop_prediction(self):
        """停止预测"""
        if self.running:
            print("[INFO] 停止预测服务")
            self.running = False
            # 广播停止状态给所有客户端
            if self.clients:
                message = json.dumps({
                    'type': 'status',
                    'status': 'stopped',
                    'message': '预测已停止'
                })
                await asyncio.gather(
                    *[client.send(message) for client in self.clients.copy()],
                    return_exceptions=True
                )
            
    async def start_server(self, host="localhost", port=8765):
        """启动WebSocket服务器"""
        print(f"[INFO] 启动WebSocket服务器: ws://{host}:{port}")
        
        # 启动串口数据线程
        self.running = True
        serial_thread = threading.Thread(target=self.serial_data_thread, daemon=True)
        serial_thread.start()
        
        # 启动WebSocket服务器和广播任务
        async with websockets.serve(
            self.websocket_handler,
            host,
            port
        ):
            print(f"[SUCCESS] WebSocket服务器启动成功")
            
            # 启动广播任务
            broadcast_task = asyncio.create_task(self.broadcast_predictions())
            
            try:
                await broadcast_task
            except asyncio.CancelledError:
                pass
            
def main():
    """主函数"""
    server = AnglePredictionServer()
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        print("\n[INFO] 正在停止服务...")
        server.running = False
        if server.ser:
            server.ser.close()
        print("[SUCCESS] 服务器已停止")
    except Exception as e:
        print(f"[ERROR] 服务器启动失败: {e}")

if __name__ == "__main__":
    main() 