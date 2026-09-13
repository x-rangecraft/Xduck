# GF43X40-10 / I2RT 通信与辨识工具

这个目录先解决数据采集链路的第一步：通过 USB 转 CAN 只读探测电机。

当前实机识别为 `CANable 2.5 Candlelight`，电机按新提供的 I2RT 手册通信，应优先使用
`probe_i2rt_canable.py`。脚本向 `0x7FF` 发送 DLC=4 的寄存器读取帧；失能扫描会按
四种控制模式分别尝试 `ID`、`0x100+ID`、`0x200+ID`、`0x300+ID`。

脚本只发送 I2RT 手册定义的 DLC=4 参数读取帧，不会使能电机、写参数、保存参数或
设置零点，因此不会主动让电机转动。

## 准备

确认以下硬件条件：

- 三台电机 CAN ID 互不相同；
- 三台电机并接在同一条 CAN 总线上（电源接口串接不代表 CAN ID 可以重复）；
- 总线两端各有 120 欧终端电阻，断电测量 CAN-H 与 CAN-L 应约为 60 欧；
- 手册默认 CAN 波特率为 1 Mbps；当前三台样机实测配置为 500 kbps，运行时必须显式传入；
- USB 转 CAN 使用已验证的 CANable 2.5 Candlelight / gs_usb。

创建环境并安装唯一依赖：

```bash
cd /Users/mac/xinchen/MicriDuck/DuckResource/Xduck/10-source/tools/motor/gf43x40-10-i2rt
../../../../60-tools/setup-motor-env.sh
.venv/bin/python -m pip install -r requirements.txt
```

## 运行

CANable Candlelight 先查看适配器：

```bash
.venv/bin/python probe_i2rt_canable.py --list-adapters
```

只读探测默认 ID 1、2、3：

```bash
.venv/bin/python probe_i2rt_canable.py --ids 1,2,3 --bitrate 500000
```

不知道 CAN ID 时，可以安全地只读扫描手册规定的完整范围：

```bash
.venv/bin/python probe_i2rt_canable.py --ids 1-31 --bitrate 500000
```

成功时每台电机会显示 `ONLINE`，并列出反馈 CAN ID、控制模式、MIT 映射范围和 CAN
波特率寄存器值。`no reply` 通常依次检查：适配器、所选波特率、CAN-H/CAN-L、
终端电阻、电机 ID、以及是否有重复 Master ID。

机器可读输出：

```bash
.venv/bin/python probe_i2rt_canable.py --ids 1,2,3 --bitrate 500000 --json
```

## 下一步

通信已经在经典 CAN 500 kbps 下打通，三台电机 ID 为 `13,14,15`，反馈 ID 为
`29,30,31`。`collect_i2rt.py` 会先读取每台电机的实际 `Pmax/Vmax/Tmax`，再按实际范围
打包和解析 MIT 帧。每条 CSV 记录均包含命令、反馈、独立 TX/RX 时间戳、延迟和原始帧，
旁边的 `metadata.json` 保存适配器、固件、实验条件和计数器。

### 失能静止噪声

此模式不使能电机，可同时采三台。正式实验每次 60 秒，重复三次：

```bash
cd /Users/mac/xinchen/MicriDuck/DuckResource/Xduck/10-source/tools/motor/gf43x40-10-i2rt
.venv/bin/python collect_i2rt.py disabled_static \
  --ids 13,14,15 --bitrate 500000 --duration 60 --rate 200 \
  --thermal-state cold --ambient-temperature 25 \
  --experiment-id disabled_static_r1
```

### 使能零指令

运动相关模式一次只允许一台电机，且必须显式传入 `--arm`。电机必须固定可靠、周围无人：

```bash
.venv/bin/python collect_i2rt.py enabled_zero \
  --ids 13 --duration 60 --rate 200 --arm \
  --joint-names motor_13 --experiment-id enabled_zero_m13_r1
```

### 平滑速度扫描

第一轮先从 `0.25–1.0 rad/s` 开始；脚本按正、反方向执行余弦斜坡、保持、回零和稳定段：

```bash
.venv/bin/python collect_i2rt.py velocity_scan \
  --ids 13 --rate 200 --arm --kd 0.2 \
  --speeds 0.25,0.5,0.75,1.0 --ramp 1.5 --hold 5 --settle 2 \
  --joint-names motor_13 --experiment-id velocity_low_m13_r1
```

### 小力矩脉冲

裸电机脚本硬限制为不超过 `1.0 N·m`、单脉冲 `50–300 ms`：

```bash
.venv/bin/python collect_i2rt.py torque_pulse \
  --ids 13 --rate 200 --arm --kd 0.2 \
  --torques 0.2,0.5,1.0 --pulse 0.2 --rest 2 \
  --joint-names motor_13 --experiment-id torque_pulse_m13_r1
```

任何运动实验正常结束、异常、超温、超速或电机故障时都会发送失能帧。正式采集前仍应准备
独立急停，并确保裸电机牢固固定。
