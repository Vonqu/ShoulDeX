import matplotlib.pyplot as plt
import numpy as np

# 读取txt文件
def read_data_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    data = []
    for line in lines:
        # values = line.strip().split(',')
        # x, y, z = map(float, values)  # 假设所有数据都是数值型
        # data.append([x, y, z])

        # 假设每行只有一个数字
        value = float(line.strip())  # 转换为浮点数并去除空白字符
        data.append(value)
    
    return np.array(data)

# 从txt文件中加载数据
file_path = "Data0108/A2NB2C3_101.txt"  #文件路径
data = read_data_from_txt(file_path)
# x, y, z = data.T  # 转置数据以便分别获取每列

# # 绘制图像
# plt.figure(figsize=(10, 6))

# # 第一幅图：x vs. y
# plt.subplot(3, 1, 1)
# plt.plot(x, label='sensor1')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()

# # 第二幅图：x vs. z
# plt.subplot(3, 1, 2)
# plt.plot(y, label='sensor2')
# plt.xlabel('x')
# plt.ylabel('z')
# plt.legend()

# # 第三幅图：y vs. z
# plt.subplot(3, 1, 3)
# plt.plot(z, label='sensor3')
# plt.xlabel('y')
# plt.ylabel('z')
# plt.legend()

# plt.tight_layout()  # 自动调整子图间距
# plt.show()

# 三组数据在一起的图

plt.figure(figsize=(10, 6))
# plt.plot(x, label='sensor1',color='red')
# plt.plot(y, label='sensor2',color ='blue')
# plt.plot(z, label='sensor3',color ='green')

plt.plot(data, label='Sensor Data0108', color='purple')

plt.xlabel('frame')  
plt.ylabel('Ohm')
plt.legend()

plt.title('sensor data original')
# plt.grid(True)  # 添加网格线以增强可读性
# plt.show()
plt.savefig('Data0108/A2NB2C3_101.png')
