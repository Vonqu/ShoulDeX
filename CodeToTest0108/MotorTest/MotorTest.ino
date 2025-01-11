#define enaPin 6
#define dirPin 4
#define pulPin 5

unsigned long startMillis;  // 用于记录开始时间
unsigned long duration = 10000;  // 默认电机运动时间（单位：毫秒）
int pulseInterval = 200;  // 设置脉冲之间的间隔（单位：微秒）

void setup() {
  Serial.begin(9600);  // 初始化串口
  pinMode(enaPin, OUTPUT); // Enable
  pinMode(pulPin, OUTPUT); // Step
  pinMode(dirPin, OUTPUT); // Dir
  digitalWrite(enaPin, HIGH); // 设置 Enable 高电平，启用电机
}

void loop() {
  // 检查串口是否有数据
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n'); // 读取串口输入的命令

    // 控制电机运动方向和持续时间
    if (command.startsWith("F") || command.startsWith("f")) {  // 向前
      duration = command.substring(1).toInt(); // 获取持续时间（命令后面是持续时间的数字）
      digitalWrite(dirPin, LOW);  // 设置电机旋转方向为前进
    } 
    else if (command.startsWith("B") || command.startsWith("b")) {  // 向后
      duration = command.substring(1).toInt(); // 获取持续时间
      digitalWrite(dirPin, HIGH);  // 设置电机旋转方向为后退
    }

    // 记录电机开始运动的时间
    startMillis = millis();

    // 电机运动，持续时间控制
    while (millis() - startMillis < duration) {
      digitalWrite(pulPin, HIGH);  // 发出脉冲高电平
      delayMicroseconds(pulseInterval);  // 控制脉冲的间隔
      digitalWrite(pulPin, LOW);  // 发出脉冲低电平
      delayMicroseconds(pulseInterval);  // 控制脉冲的间隔
    }

    // 电机停止运动
    digitalWrite(pulPin, LOW);  // 保证步进信号为低电平
    Serial.println("Motor movement completed");
  }
}
