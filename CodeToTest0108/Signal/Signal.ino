void setup() {
  Serial.begin(115200);  // 初始化串口输出
  analogReadResolution(12);  // 设置 ADC 分辨率为 12 位
  pinMode(A3, INPUT);  // 配置 GPIO34 为输入引脚
  pinMode(A4, INPUT);  // 配置 GPIO35 为输入引脚
}

void loop() {
  int adcValue1 = analogRead(A3);  // 读取 GPIO34 的 ADC 值
  int adcValue2 = analogRead(A4);  // 读取 GPIO35 的 ADC 值
  
  float voltage1 = adcValue1 * (3.3 / 4095.0);  // 将 ADC 值转换为电压
  float voltage2 = adcValue2 * (3.3 / 4095.0) + 0.15;  // 将 ADC 值转换为电压
  
  float voltageDiff = voltage1 - voltage2;  // 计算电压差值

  float resistance = 100 * (voltageDiff + 1.65) / (1.65 - voltageDiff);
  // float resistance = 40 * (voltageDiff + 1.65) / (1.65 - voltageDiff);

  // 输出电压和电压差值
  // Serial.print("Voltage1,");
  // Serial.print(voltage1, 4);
  // Serial.print("\tVoltage 2,");
  // Serial.print(voltage2, 4);
  // Serial.print("\tVoltage Difference,");
  // Serial.print(voltageDiff, 4);  // 输出电压差值，保留 4 位小数
  // Serial.print("\tResistance,");
  Serial.println(resistance, 4); 

  delay(8);  // 延迟 500 毫秒
}
