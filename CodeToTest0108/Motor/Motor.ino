#define enaPin 6
#define dirPin 4
#define pulPin 5
#define OKPin 8

bool isRecording = false;  // 用于标记是否开始记录


unsigned long startMillis;  // 用于记录开始时间
unsigned long duration = 5000;  // 设置电机运动的时间（单位：毫秒）
int pulseInterval = 200;  // 设置脉冲之间的间隔（单位：微秒）

void setup()
{
  Serial.begin(9600);
  pinMode(enaPin, OUTPUT); // Enable
  pinMode(pulPin, OUTPUT); // Step
  pinMode(dirPin, OUTPUT); // Dir
  pinMode(OKPin, INPUT);
  digitalWrite(enaPin, HIGH); // Set Enable low
  startMillis = millis();  // 记录电机开始运动的时间

}
void loop()
{
  if (digitalRead(OKPin) == HIGH) {

      isRecording = true;
      Serial.println("Recording Started");  // 告诉 Python 开始记录

      startMillis = millis();  // 记录电机开始运动的时间

      while (1)
      {
        unsigned long currentMillis = millis();  // 获取当前时间

        if (currentMillis - startMillis >= duration) {
    // 停止电机的运动，可以通过设置方向或禁用步进信号来停止
        digitalWrite(pulPin, LOW);  // 保证步进信号为低电平
        return;  // 退出 loop 函数，停止电机
       }

        digitalWrite(dirPin, HIGH);  // 设置电机的旋转方向
        digitalWrite(pulPin, HIGH);  // 发出脉冲高电平
        delayMicroseconds(pulseInterval);  // 控制脉冲的间隔
        digitalWrite(pulPin, LOW);   // 发出脉冲低电平
        delayMicroseconds(pulseInterval);  // 控制脉冲的间隔
      }
      Serial.println("Motor movement completed");

      isRecording = false;  // 停止记录

    
  }
}
