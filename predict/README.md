# 角度预测仪表板系统

这是一个基于LSTM模型的实时角度预测系统，包含前端可视化界面和后端预测服务。

## 系统架构

```
┌─────────────────┐    WebSocket    ┌─────────────────┐
│   前端界面      │ ←──────────────→ │   WebSocket     │
│ (HTML/CSS/JS)   │                 │    服务器       │
└─────────────────┘                 └─────────────────┘
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │   LSTM模型      │
                                    │   预测服务      │
                                    └─────────────────┘
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │   串口数据      │
                                    │   (传感器)      │
                                    └─────────────────┘
```

## 文件说明

- `angle_prediction_dashboard.html` - 前端界面，3x3网格布局显示27个角度
- `websocket_server.py` - WebSocket服务器，处理实时数据预测
- `start_dashboard.py` - 启动脚本，一键启动整个系统
- `predict_ds_14Sto27A.py` - 原始预测代码（参考）

## 功能特点

### 前端界面
- 🎨 现代化的3x3网格布局
- 📊 实时角度数据可视化
- 🎯 27个角度的实时显示
- 📈 历史数据曲线图
- 📱 响应式设计，支持移动设备
- ⚡ 实时连接状态显示
- 📊 性能统计信息

### 后端服务
- 🔌 WebSocket实时通信
- 🤖 LSTM模型预测
- 📡 串口数据读取
- 🔄 自动重连机制
- 📝 详细日志记录

## 角度分类

系统显示以下9组角度数据：

1. **左肱骨与胸廓** (Left Humerus-Thorax) - 角度 0-2
2. **右肱骨与胸廓** (Right Humerus-Thorax) - 角度 3-5
3. **左肩胛骨与胸廓** (Left Scapula-Thorax) - 角度 6-8
4. **右肩胛骨与胸廓** (Right Scapula-Thorax) - 角度 9-11
5. **左锁骨与胸廓** (Left Clavicle-Thorax) - 角度 12-14
6. **右锁骨与胸廓** (Right Clavicle-Thorax) - 角度 15-17
7. **左肱骨与肩胛骨** (Left Humerus-Scapula) - 角度 18-20
8. **右肱骨与肩胛骨** (Right Humerus-Scapula) - 角度 21-23
9. **胸廓与髋关节** (Thorax-Hip) - 角度 24-26

每组包含X、Y、Z三个轴向的角度数据。

## 快速启动

### 一键启动（推荐）

```bash
cd predict
python start_dashboard.py
```

系统将自动：
1. 启动WebSocket服务器 (端口8766)
2. 连接串口设备 (COM7)
3. 加载LSTM模型
4. 在浏览器中打开前端界面
5. 开始实时角度预测

### 手动启动

1. **安装依赖**
```bash
pip install websockets pyserial torch numpy pandas matplotlib scikit-learn
```

2. **启动WebSocket服务器**
```bash
python websocket_server.py
```

3. **打开前端界面**
```bash
# 在浏览器中打开
file:///path/to/predict/angle_prediction_dashboard.html
```

## 配置说明

### 串口配置
在 `websocket_server.py` 中修改串口设置：
```python
self.SERIAL_PORT = "COM7"  # 修改为您的串口
self.BAUD_RATE = 115200    # 修改波特率
```

### 模型配置
确保以下文件存在：
- `./model/test_0806_1_lstm.pth` - LSTM模型文件
- `./predict/scaler/sensor_scaler_test_0806_1.pkl` - 传感器数据标准化器
- `./predict/scaler/angle_scaler_test_0806_1.pkl` - 角度数据标准化器

### 网络配置
- WebSocket服务器默认运行在 `ws://localhost:8765`
- 如需外部访问，修改 `websocket_server.py` 中的host参数

## 使用说明

1. **启动系统**
   - 运行启动脚本或手动启动服务器
   - 确保串口设备已连接

2. **开始预测**
   - 点击前端界面的"开始预测"按钮
   - 系统开始读取串口数据并实时预测角度

3. **查看数据**
   - 每个小窗格显示对应角度组的实时数据
   - 曲线图显示历史变化趋势
   - 底部显示当前角度数值

4. **停止预测**
   - 点击"停止预测"按钮停止数据采集
   - 或关闭浏览器窗口

## 故障排除

### 常见问题

1. **Unicode编码错误**
   - 问题：`UnicodeEncodeError: 'gbk' codec can't encode character`
   - 解决：已修复，使用 `[SUCCESS]`、`[ERROR]` 等文本标签替代Unicode表情符号
   - 如果仍有问题，请确保Python环境支持UTF-8编码

2. **串口连接失败**
   - 检查串口设备是否正确连接
   - 确认串口号和波特率设置
   - 检查设备驱动是否安装
   - 运行 `python test_system.py` 检查串口状态

3. **WebSocket连接失败**
   - 确认服务器是否正常启动
   - 检查防火墙设置
   - 查看浏览器控制台错误信息
   - 运行 `python test_system.py` 检查WebSocket功能

4. **模型加载失败**
   - 确认模型文件路径正确
   - 检查文件权限
   - 验证模型文件完整性
   - 运行 `python test_system.py` 检查模型文件

5. **前端显示异常**
   - 刷新浏览器页面
   - 检查JavaScript控制台错误
   - 确认浏览器支持WebSocket
   - 尝试使用Chrome或Firefox浏览器

### 调试模式

启用详细日志输出：
```python
# 在websocket_server.py中添加
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 性能优化

1. **降低更新频率**
   - 修改 `step_size` 参数
   - 调整前端刷新间隔

2. **减少历史数据**
   - 修改 `angleHistory.times.length > 200` 中的数值

3. **优化模型推理**
   - 使用GPU加速（如果可用）
   - 调整批处理大小

## 扩展功能

### 添加新的角度组
1. 修改前端HTML中的角度分组
2. 更新JavaScript中的角度索引映射
3. 调整图表显示逻辑

### 数据导出
可以添加CSV导出功能：
```javascript
function exportData() {
    const csv = convertToCSV(angleHistory);
    downloadCSV(csv, 'angle_data.csv');
}
```

### 实时录制
可以添加录制功能保存预测数据：
```python
# 在websocket_server.py中添加
def save_prediction_data(self, data):
    with open('prediction_log.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([data['timestamp']] + data['angles'])
```

## 技术支持

如有问题，请检查：
1. 系统日志输出
2. 浏览器开发者工具
3. 网络连接状态
4. 硬件设备状态

## 版本信息

- 版本：1.0.0
- 更新日期：2024年
- 支持平台：Windows, Linux, macOS
- 浏览器要求：Chrome 60+, Firefox 55+, Safari 11+ 