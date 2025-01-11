#define enaPin 6
#define dirPin 4
#define pulPin 5
#define OKPin 8

bool isRecording = false;  // 用于标记是否开始记录

unsigned long startMillis;  // 用于记录开始时间
unsigned long duration = 9000;  // 设置电机运动的时间（单位：毫秒），每次运动 5 秒
int pulseInterval = 200;  // 设置脉冲之间的间隔（单位：微秒）

int cycleCount = 0;  // 用于记录往返次数
const int maxCycles = 25;  // 设置最大往返次数

void setup()
{
  Serial.begin(9600);
  pinMode(enaPin, OUTPUT); // Enable
  pinMode(pulPin, OUTPUT); // Step
  pinMode(dirPin, OUTPUT); // Dir
  pinMode(OKPin, INPUT);
  digitalWrite(enaPin, HIGH); // Set Enable high
  startMillis = millis();  // 记录电机开始运动的时间
}

void loop()
{
  if (digitalRead(OKPin) == HIGH) {  // 按钮按下时，开始记录

    isRecording = true;
    Serial.println("Recording Started");  // 告诉 Python 开始记录

    // 往返 10 次，每次持续 5 秒
    while (cycleCount < maxCycles) {
      // 向前运动
      digitalWrite(dirPin, LOW);  // 设置电机旋转方向
      Serial.println("Motor moving forward");

      startMillis = millis();  // 记录开始时间
      while (millis() - startMillis < duration) {
        digitalWrite(pulPin, HIGH);  // 发出脉冲高电平
        delayMicroseconds(pulseInterval);  // 控制脉冲的间隔
        digitalWrite(pulPin, LOW);   // 发出脉冲低电平
        delayMicroseconds(pulseInterval);  // 控制脉冲的间隔
      }

      Serial.println("Motor moved forward");

      // 向后运动
      digitalWrite(dirPin, HIGH);  // 设置电机旋转方向
      Serial.println("Motor moving backward");

      startMillis = millis();  // 记录开始时间
      while (millis() - startMillis < duration) {
        digitalWrite(pulPin, HIGH);  // 发出脉冲高电平
        delayMicroseconds(pulseInterval);  // 控制脉冲的间隔
        digitalWrite(pulPin, LOW);   // 发出脉冲低电平
        delayMicroseconds(pulseInterval);  // 控制脉冲的间隔
      }

      Serial.println("Motor moved backward");

      cycleCount++;  // 增加往返次数

      // 等待一段时间，防止过快地重复操作（例如 1 秒）
      // delay(1000);  // 延时 1 秒
    }

    // 往返完成
    Serial.println("Motor movement completed");
    isRecording = false;  // 停止记录
    cycleCount = 0;  // 重置往返计数器
  }
}
