import serial
import datetime

port_signal = 'COM12'  # 用于接收开始信号的串口
port_data = 'COM10'     # 用于接收数据的串口
baud_rate_signal = 9600  # COM13的波特率
baud_rate_data = 115200  # COM8的波特率

ser_signal = serial.Serial(port_signal, baud_rate_signal)
ser_data = serial.Serial(port_data, baud_rate_data)

filepath = r'Data0108\A2NB2C3_5_25.txt'
file = open(filepath, 'w')

while True:
    if ser_signal.in_waiting > 0:
        data = ser_signal.readline().decode('utf-8').strip()
        print(f"Received from Arduino (Signal): {data}")

        # 如果收到“Recording Started”信号，开始记录
        if data == "Recording Started":
            print("Recording started...")
            start_time = datetime.datetime.now()
            ser_data.reset_input_buffer()

            # 开始记录数据，从 COM8 串口接收数据
            while True:
                if ser_data.in_waiting > 0:
                    data = ser_data.readline().decode('utf-8').strip()
                    sensor = data
                    file.write(f"{sensor}\n")
                    file.flush()
                    print(data)

                if ser_signal.in_waiting > 0:
                    data = ser_signal.readline().decode('utf-8').strip()
                    if data == "Motor movement completed":
                        print("Motor has finished movement.")
                        break  # 结束记录

                elapsed_time = (datetime.datetime.now() - start_time).seconds
                if elapsed_time > 10000:
                    print("Recording time exceeded.")
                    break

            print("Recording stopped.")
            break  # 结束主循环

file.close()
ser_signal.close()
ser_data.close()
