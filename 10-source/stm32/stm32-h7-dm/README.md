# H7 DM

本分支由 `/Users/xiaoyaozi/xinchen/H7_DM` 整理生成。

## 目录

- .git
- .git/hooks
- .git/info
- .git/logs
- .git/objects
- .git/refs
- CtrBoard-H7_ALL
- CtrBoard-H7_ALL/App
- CtrBoard-H7_ALL/Bsp
- CtrBoard-H7_ALL/Core
- CtrBoard-H7_ALL/Drivers
- CtrBoard-H7_ALL/Hardware
- CtrBoard-H7_ALL/MDK-ARM
- CtrBoard-H7_ALL/Middlewares
- CtrBoard-H7_ALL/USB_DEVICE


---

# H7_DM 与 H7_ZD SPI 电机通信方案

> **已停用（2026-09-07）**：RK3566 与 H7_DM 的运行时控制链路已统一为
> USB CDC。固件不再初始化 SPI1，不再创建 H7 SPI 从机任务，也不再注册
> SPI1 DMA、SPI1 或 PE15 片选边沿中断。下文仅保留为历史协议设计记录；
> `h7spi_protocol.h` 中仍被 USB 电机桥复用的数据结构不代表 SPI 外设在运行。

## 1. 讨论边界与目标

本文只讨论 `H7_ZD` 与 `H7_DM` 之间的 SPI 通信方案。`jeston/上位机 -> H7_ZD` 的网口通信暂不展开，H7_ZD 在本方案中被视为已经拿到了电机控制命令的一侧。

目标链路：

```text
H7_ZD -> SPI -> H7_DM -> CAN/FDCAN -> 电机
```

角色分工：

```text
H7_ZD: SPI Master，负责主动发起 SPI 传输
H7_DM: SPI Slave，负责接收命令、维护电机控制状态、周期性下发 CAN/FDCAN
```

H7_DM 现有电机链路保持不变：

```text
motor_task.c
  -> MotorApp_Tick()，约 1ms 一次
  -> Motor_SendAllOnce()
  -> BSP_CAN_SendStandardDataMessage()
  -> CAN/FDCAN -> 电机
```

SPI 只负责把 H7_ZD 的电机命令、管理命令、模式命令传给 H7_DM，并把 H7_DM 的状态、ACK、错误回传给 H7_ZD。

## 2. SPI 主从与硬件连接

建议：

```text
H7_ZD = SPI Master
H7_DM = SPI Slave
```

原因：

- H7_ZD 是命令入口，适合主动决定什么时候发起 SPI 事务。
- H7_DM 要保持电机实时控制节奏，SPI 收包不应打乱 1ms 电机 Tick。
- SPI Slave 不能主动发数据，所以 H7_DM 的状态由 H7_ZD 周期性 clock 出来。

推荐连接：

```text
H7_ZD -> H7_DM: SPI_CS
H7_ZD -> H7_DM: SPI_SCK
H7_ZD -> H7_DM: SPI_MOSI
H7_DM -> H7_ZD: SPI_MISO
H7_DM -> H7_ZD: DM_READY
GND 必须共地
```

`DM_READY` 建议保留：

```text
DM_READY = 1: H7_DM 已经准备好下一次 SPI DMA 收发
DM_READY = 0: H7_ZD 暂时不要拉 CS 发下一包
```

没有 READY 线也能做，但 H7_ZD 可能在 H7_DM 尚未重新挂 DMA 时发包，调试风险更高。第一版代码未使用 READY 线，靠固定 512 字节 DMA 完成回调立即续挂。

## 3. SPI 事务方式

第一版建议 SPI DMA 固定长度：

```c
#define H7SPI_TRANSFER_SIZE  512
#define H7SPI_MAX_MOTORS     24
```

理由：当前 H7_DM 中 `MOTOR_NUM = 24`。如果以后扩展到 24 个电机，512 字节仍然够用，不需要再改 SPI DMA 长度。

每次 SPI 事务固定收发 512 字节：

```text
MOSI: H7_ZD -> H7_DM，当前命令帧
MISO: H7_DM -> H7_ZD，上一轮准备好的 ACK/状态/错误帧
```

因为 SPI 是全双工，从机不能主动发，所以状态回传采用“下一帧带回上一帧结果”：

```text
第 N 次事务：
  ZD -> DM: MOTOR_CMD seq=100
  DM -> ZD: 上一次准备好的状态

第 N+1 次事务：
  ZD -> DM: MOTOR_CMD 或 HEARTBEAT seq=101
  DM -> ZD: 对 seq=100 的 ACK/状态
```

没有新控制命令时，H7_ZD 也应周期性发送 `HEARTBEAT`，用于：

- 维持 SPI 链路活性。
- clock 出 H7_DM 的状态帧。
- 让 H7_DM 区分“SPI 链路还活着”和“没有新的 MOTOR_CMD”。

## 4. 通用 SPI 帧格式

所有 SPI 包都使用固定帧头，payload 根据 `msg_type` 变化。

```c
#define H7SPI_MAGIC    0xA55A
#define H7SPI_VERSION  1

#pragma pack(push, 1)
typedef struct {
  uint16_t magic;        /* 0xA55A */
  uint8_t  version;      /* 1 */
  uint8_t  msg_type;     /* H7SPI_MSG_xxx */
  uint16_t seq;          /* SPI 帧序号 */
  uint16_t payload_len;  /* payload 有效字节数 */
  uint32_t tick_ms;      /* 发送端 tick，单位 ms */
  uint16_t flags;        /* 第一版可填 0 */
  uint16_t crc16;        /* CRC16-CCITT */
} h7spi_header_t;
#pragma pack(pop)
```

帧整体：

```text
h7spi_header_t
payload[payload_len]
padding 到 512 字节
```

CRC 计算范围：

```text
magic/version/msg_type/seq/payload_len/tick_ms/flags + payload
```

不把 `crc16` 字段本身算进去。

多字节字段统一使用 little-endian。

## 5. 消息类型

```c
#define H7SPI_MSG_MOTOR_CMD       0x01  /* ZD -> DM，电机控制 */
#define H7SPI_MSG_MOTOR_STATE     0x02  /* DM -> ZD，电机状态 */
#define H7SPI_MSG_SET_MODE        0x03  /* ZD -> DM，切换模式 */
#define H7SPI_MSG_MODE_STATE      0x04  /* DM -> ZD，模式状态 */
#define H7SPI_MSG_ADMIN_OP        0x05  /* ZD -> DM，使能/失能/标零/清错 */
#define H7SPI_MSG_ADMIN_RESULT    0x06  /* DM -> ZD，管理操作结果 */
#define H7SPI_MSG_HEARTBEAT       0x07  /* ZD -> DM，心跳/取状态 */
#define H7SPI_MSG_ACK             0x08  /* DM -> ZD，普通确认 */
#define H7SPI_MSG_ERROR           0x09  /* DM -> ZD，协议错误 */
```

第一版可以优先实现：

```text
MOTOR_CMD
MOTOR_STATE
HEARTBEAT
ACK
ERROR
```

但协议结构一次性预留 `SET_MODE`、`MODE_STATE`、`ADMIN_OP`、`ADMIN_RESULT`，避免后面改包格式。

## 6. 控制模式

```c
#define H7SPI_MODE_DISABLED       0x00
#define H7SPI_MODE_ADMIN          0x10
#define H7SPI_MODE_CONTROL_MIT    0x21
#define H7SPI_MODE_CONTROL_SPEED  0x22  /* 预留 */
#define H7SPI_MODE_CONTROL_POS    0x23  /* 预留 */
```

H7_DM 当前实际支持：

```text
ADMIN
CONTROL_MIT
DISABLED
```

`SPEED`、`POS` 可以先只保留协议值，不实现。收到暂未支持的模式时返回 `H7SPI_RESULT_BAD_MODE`。

## 7. MOTOR_CMD：电机控制命令

`msg_type = H7SPI_MSG_MOTOR_CMD`

payload：

```c
#pragma pack(push, 1)
typedef struct {
  uint8_t  motor_code;    /* 对应 H7_DM 的 MotorPoint(code) */
  uint8_t  flags;         /* 第一版可填 0 */
  int32_t  p_mrad;        /* 目标位置 rad * 1000 */
  int32_t  v_mrad_s;      /* 目标速度 rad/s * 1000 */
  int32_t  torque_mnm;    /* 目标力矩 N*m * 1000 */
  uint16_t kp_centi;      /* Kp * 100 */
  uint16_t kd_milli;      /* Kd * 1000 */
} h7spi_motor_cmd_t;

typedef struct {
  uint16_t command_seq;   /* 电机命令序号 */
  uint8_t  mode;          /* H7SPI_MODE_CONTROL_MIT 等 */
  uint8_t  motor_count;   /* 本帧有效电机数量 */
  uint32_t host_tick_ms;  /* H7_ZD 或上游时间 */
  h7spi_motor_cmd_t motors[H7SPI_MAX_MOTORS];
} h7spi_motor_cmd_frame_t;
#pragma pack(pop)
```

单个 `h7spi_motor_cmd_t` 为 20 字节，命令头为 8 字节。

若固定发满 24 个电机：

```text
payload_len = 8 + 24 * 20 = 488 字节
总帧长 = 16 + 488 = 504 字节
```

刚好放入 512 字节 SPI DMA 包。

第一版建议：

```text
H7_ZD 固定发送当前配置的全部电机命令，例如 10 个或 24 个。
H7_DM 根据 motor_count 解析有效电机数量。
```

后续如果要优化带宽，可以允许只发送部分电机，此时：

```text
payload_len = 8 + motor_count * sizeof(h7spi_motor_cmd_t)
```

## 8. MOTOR_STATE：电机状态回传

`msg_type = H7SPI_MSG_MOTOR_STATE`

payload：

```c
#define H7SPI_MOTOR_FLAG_CONFIGURED  (1U << 0)
#define H7SPI_MOTOR_FLAG_ONLINE      (1U << 1)
#define H7SPI_MOTOR_FLAG_ENABLED     (1U << 2)
#define H7SPI_MOTOR_FLAG_FAULT       (1U << 3)

#pragma pack(push, 1)
typedef struct {
  uint8_t  motor_code;
  uint8_t  state;          /* 电机反馈状态 */
  uint8_t  flags;          /* CONFIGURED/ONLINE/ENABLED/FAULT */
  uint8_t  temperature;    /* 可先放 rotor_temp，或后续拆 mos/rotor */
  int32_t  p_mrad;
  int32_t  v_mrad_s;
  int32_t  torque_mnm;
} h7spi_motor_state_t;

typedef struct {
  uint16_t state_seq;
  uint16_t ack_command_seq;   /* 最近成功接受的 command_seq */
  uint32_t dm_tick_ms;
  uint32_t fault_flags;
  uint16_t enabled_mask_low;  /* motor_code 0-15 */
  uint16_t enabled_mask_high; /* motor_code 16-31 */
  uint8_t  mode;
  uint8_t  enabled;
  uint8_t  fault;
  uint8_t  motor_count;
  h7spi_motor_state_t motors[H7SPI_MAX_MOTORS];
} h7spi_motor_state_frame_t;
#pragma pack(pop)
```

状态包要让 H7_ZD 能区分：

```text
SPI 链路问题
H7_DM 内部 fault
某个电机 CAN/FDCAN 反馈掉线
某个电机硬件故障/过温
```

## 9. SET_MODE 与 MODE_STATE

`SET_MODE` payload：

```c
#pragma pack(push, 1)
typedef struct {
  uint16_t request_seq;
  uint8_t  requested_mode;
  uint8_t  reserved;
} h7spi_set_mode_frame_t;
#pragma pack(pop)
```

`MODE_STATE` payload：

```c
#pragma pack(push, 1)
typedef struct {
  uint16_t response_seq;
  uint16_t result;
  uint32_t dm_tick_ms;
  uint32_t fault_flags;
  uint16_t enabled_mask_low;
  uint16_t enabled_mask_high;
  uint8_t  current_mode;
  uint8_t  requested_mode;
  uint8_t  busy;
  uint8_t  motor_count;
  uint8_t  failed_motor_id;
  uint8_t  failed_route_index;
  uint8_t  reserved[2];
} h7spi_mode_state_frame_t;
#pragma pack(pop)
```

## 10. ADMIN_OP 与 ADMIN_RESULT

管理操作：

```c
#define H7SPI_ADMIN_ENABLE       1
#define H7SPI_ADMIN_DISABLE      2
#define H7SPI_ADMIN_MARK_ZERO    3
#define H7SPI_ADMIN_CLEAR_ERROR  4
```

`ADMIN_OP` payload：

```c
#pragma pack(push, 1)
typedef struct {
  uint16_t request_seq;
  uint8_t  op;
  uint8_t  motor_count;
  uint8_t  motor_codes[H7SPI_MAX_MOTORS];
} h7spi_admin_op_frame_t;
#pragma pack(pop)
```

约定：

```text
motor_count = 0: 对全部已配置电机执行
motor_count > 0: 只对 motor_codes 中列出的电机执行
```

`ADMIN_RESULT` payload：

```c
#pragma pack(push, 1)
typedef struct {
  uint16_t response_seq;
  uint8_t  op;
  uint8_t  result;
  uint32_t dm_tick_ms;
  uint32_t fault_flags;
  uint32_t requested_mask;
  uint32_t succeeded_mask;
  uint8_t  busy;
  uint8_t  requested_count;
  uint8_t  success_count;
  uint8_t  failed_motor_id;
  uint8_t  failed_route_index;
  uint8_t  reserved[3];
} h7spi_admin_result_frame_t;
#pragma pack(pop)
```

## 11. ACK 与 ERROR

普通 ACK payload：

```c
#pragma pack(push, 1)
typedef struct {
  uint16_t ack_seq;       /* 确认收到的 SPI seq */
  uint16_t result;        /* H7SPI_RESULT_xxx */
  uint16_t command_seq;   /* 如果是 MOTOR_CMD，则填 command_seq */
  uint16_t error_detail;  /* 可填 motor_code / msg_type / len 等 */
} h7spi_ack_frame_t;
#pragma pack(pop)
```

结果码：

```c
#define H7SPI_RESULT_OK               0
#define H7SPI_RESULT_BAD_MAGIC        1
#define H7SPI_RESULT_BAD_VERSION      2
#define H7SPI_RESULT_BAD_CRC          3
#define H7SPI_RESULT_BAD_LENGTH       4
#define H7SPI_RESULT_BAD_TYPE         5
#define H7SPI_RESULT_BAD_MODE         6
#define H7SPI_RESULT_BUSY             7
#define H7SPI_RESULT_FAULT            8
```

## 12. 通信保护策略

### 12.1 传输保护

H7_DM 收包后按顺序检查：

```text
magic
version
payload_len
crc16
msg_type
seq
motor_count
motor_code
mode
```

规则：

```text
magic/version/len/crc/msg_type 错误：丢包，不更新电机目标
seq 跳号：记录丢帧计数，可继续处理合法包
motor_code 非法或未配置：跳过该电机目标，不影响同帧其他电机
mode 不允许：返回 BAD_MODE，不更新目标
```

CRC 错、长度错、非法电机编号的包不能刷新“最近有效 MOTOR_CMD 时间”。

### 12.2 命令超时保持策略

已经确认采用 `hold last valid command` 策略。

定义：有效 `MOTOR_CMD` 是指该 SPI 包通过 magic/version/len/crc/msg_type/motor_count/motor_code/mode 等检查，并且可以安全写入电机目标。

H7_DM 维护两个时间：

```c
last_valid_spi_tick
last_valid_motor_cmd_tick
```

策略：

```text
超过 CMD_TIMEOUT_MS 没有收到新的有效 MOTOR_CMD:
  H7_DM 不清零目标
  H7_DM 不释放力矩
  H7_DM 不主动失能
  H7_DM 继续周期性下发最后一次有效 MOTOR_CMD
  状态包上报 CMD_TIMEOUT warning/fault 标志

超过 LINK_TIMEOUT_MS 没有收到任何有效 SPI 包:
  H7_DM 仍继续保持最后一次有效 MOTOR_CMD
  状态包上报 LINK_LOST 标志
  是否失能不在第一版自动执行，由上层或后续策略决定
```

这样通信短时异常时，机构不会因为目标清零或电机失能直接倒下，而是保持最后一次正常命令对应的站立/支撑状态。

注意：如果最后一次有效命令本身是高速运动或大力矩运动，通信断开后也会继续保持这个命令。因此上层正常控制周期中应保证发给 H7_DM 的命令本身就是可保持的目标，例如站立位置、`v=0`、合适的 `kp/kd/torque`。

### 12.3 电机掉线保护

电机掉线由 H7_DM 本地处理，不依赖 H7_ZD。

当前 H7_DM 已有保护：

```text
电机 CAN/FDCAN 反馈超过 MOTOR_APP_FEEDBACK_TIMEOUT_MS 未更新
  -> MotorApp 进入 fault
  -> 记录 MOTOR_APP_FAULT_FEEDBACK_STALE
  -> DisableAllRoutes
```

当前代码中 `MOTOR_APP_FEEDBACK_TIMEOUT_MS = 40ms`。

SPI 状态包只负责把这个 fault、失败电机 ID、失败 route index、每个电机 online/fault flags 回传给 H7_ZD。

### 12.4 数值范围保护

SPI 层第一版不做额外数值范围保护。

原因：当前 `Motor.cpp` 在编码 MIT CAN 帧时已经会 clamp：

```text
Position -> 电机位置范围
Speed    -> 电机速度范围
Torque   -> 电机力矩范围
Kp       -> 0-500
Kd       -> 0-5
```

当前 H7_DM 默认电机扭矩范围：

```text
motor_code 1, 2:                  -200 到 200 N*m
motor_code 3,4,5,6,13,14,15,16:   -500 到 500 N*m
```

所以第一版 SPI 层只检查协议合法性、电机编号合法性和模式合法性，不因为目标值超范围拒包。


## 14. H7_DM 软件接入方式

推荐模块划分：

```text
SPI DMA 回调:
  只标记 rx_done / tx_done
  不做复杂解析
  不发 CAN

SPI link task:
  等待 rx_done
  校验 SPI 帧
  解析 msg_type
  更新最新有效电机命令 / mode / admin 请求
  准备下一次 MISO 返回的 ACK/STATE/ERROR

MotorApp_Tick 1ms:
  检查电机反馈保护
  检查 SPI command timeout/link lost 状态
  将最新有效命令写入 MotorPoint(code)->SetMITCommand(...)
  Motor_SendAllOnce()
```

不允许在 SPI 中断或 DMA 回调中直接发 CAN/FDCAN。

## 15. H7_ZD 软件接入方式

H7_ZD 的 SPI 任务建议：

```text
准备 tx_frame
等待 DM_READY = 1
拉 CS
SPI DMA/阻塞收发 512 bytes
释放 CS
解析 rx_frame 中上一轮 ACK/STATE/ERROR
根据 ACK/result 更新链路状态
```

H7_ZD 应做的保护：

```text
连续若干次 ACK BAD_CRC/BAD_LENGTH -> 记录错误，必要时降频或重置 SPI
超过一定时间没有 DM_READY -> 认为 H7_DM SPI 未准备好
超过一定时间没有有效 DM 状态 -> 上报 DM_LINK_LOST
```

## 16. 推荐频率与速度

初始调试：

```text
SPI: 5-10 MHz
H7_ZD -> H7_DM: 100-500 Hz
H7_DM -> 电机 CAN/FDCAN: 1 kHz Tick
```

稳定后目标：

```text
SPI: 10-20 MHz
H7_ZD -> H7_DM: 500 Hz-1 kHz
H7_DM -> 电机 CAN/FDCAN: 1 kHz Tick
```

512 字节传输时间估算：

```text
10 MHz: 512 * 8 / 10MHz = 0.4096 ms
20 MHz: 512 * 8 / 20MHz = 0.2048 ms
```

如果 SPI 目标跑 1kHz，建议 20MHz 更稳。

真正瓶颈更可能在 CAN/FDCAN 总线，而不是 SPI。

## 17. 推荐落地顺序

1. 定义公共协议头文件，例如 `h7spi_motor_protocol.h`，H7_ZD 和 H7_DM 共用。
2. H7_DM 实现 SPI Slave DMA 固定 512 字节收发，先不接电机。
3. H7_ZD 实现 SPI Master 固定 512 字节收发，配合 `DM_READY`。
4. 打通 `HEARTBEAT -> ACK/STATE`。
5. 实现 `MOTOR_CMD` 校验、ACK、错误码。
6. H7_DM 保存最新有效电机命令。
7. `MotorApp_Tick()` 每 1ms 应用最新有效命令，调用现有 `MotorPoint(code)->SetMITCommand()`。
8. 加入 command timeout/link lost 标志，但按约定保持最后一次有效命令。
9. 加入状态回传：fault_flags、mode、enabled、motor online/fault flags。
10. 再实现 `SET_MODE`、`ADMIN_OP`、`ADMIN_RESULT`。
12. 最后根据示波器和错误计数调整 SPI 时钟、READY 时序、事务频率。

---

# H7_ZD 与 H7_DM SPI 通信具体实施方案

## 1. 当前实施范围

本文只讨论 `H7_ZD` 与 `H7_DM` 之间的 SPI 通信实施。

暂不考虑：

- jeston/上位机 与 H7_ZD 的网口通信
- 上位机协议适配
- IMU/AHRS 数据透传

当前目标链路：

```text
H7_ZD
  -> SPI
  -> H7_DM
  -> CAN/FDCAN
  -> DM 电机
```

角色定义：

```text
H7_ZD: SPI Master，主动拉 CS、输出 SCK、发起每一次 SPI 事务
H7_DM: SPI Slave，被动接收命令，同时在 MISO 方向回传上一帧准备好的状态
```

H7_DM 现有电机控制链路保持不变：

```text
motor_task.c
  -> MotorApp_Tick()
  -> Motor_SendAllOnce()
  -> BSP_CAN_SendStandardDataMessage()
  -> CAN/FDCAN
  -> DM 电机
```

SPI 只负责：

- ZD 向 DM 下发电机控制命令
- ZD 向 DM 下发模式/管理命令
- DM 向 ZD 回传状态、ACK、错误信息

## 2. 硬件接线方案

当前按 Apollo V2 Wireless 作为 SPI 主机，CtrBoard U13 外接接口作为 SPI 从机接线。

| Apollo V2 Wireless | 信号 | CtrBoard U13 外接接口 |
| --- | --- | --- |
| Pin 1 GND | 地 | GND |
| Pin 5 SPI2_SCK / PB13 | SPI 时钟 | SPI1_SCK / PB03 |
| Pin 6 SPI2_MOSI / PB15 | Master 输出 | SPI1_MOSI / PD07 |
| Pin 7 SPI2_MISO / PB14 | Master 输入 | SPI1_MISO / PB04 |
| Pin 4 NRF_CS / PG10 | 片选 CS | SPI1_CS / PE15 |

注意：这里按 MCU SPI 外设功能名连接，不交叉 MOSI/MISO。

```text
主机 MOSI -> 从机 MOSI
主机 MISO <- 从机 MISO
主机 SCK  -> 从机 SCK
主机 CS   -> 从机 CS
GND 必须共地
```

也就是：

```text
Apollo PB15 / SPI2_MOSI -> CtrBoard PD07 / SPI1_MOSI
Apollo PB14 / SPI2_MISO <- CtrBoard PB04 / SPI1_MISO
Apollo PB13 / SPI2_SCK  -> CtrBoard PB03 / SPI1_SCK
Apollo PG10 / NRF_CS    -> CtrBoard PE15 / SPI1_CS
Apollo GND              -> CtrBoard GND
```

第一版先不强制增加 `DM_READY` 线。如果后续发现主机发包时从机 DMA 尚未重新挂载，导致首包/偶发包丢失，再增加一根 READY GPIO：

```text
DM_READY = 1: H7_DM 已准备好下一次 SPI DMA 收发
DM_READY = 0: H7_ZD 暂停发包
```

## 3. SPI 基本参数

建议第一版参数：

```text
模式: SPI Mode 0
CPOL: 0
CPHA: 0
数据位宽: 8 bit
字节序: 多字节字段 little-endian
CS: 低有效，由 H7_ZD 控制；H7_DM 第一版使用 SPI_NSS_SOFT，PE15 保持 GPIO 输入作为物理片选参考，DMA 固定 512 字节形成事务边界
传输方式: DMA
单次传输长度: 固定 512 字节
```

速率建议先保守：

```text
首轮联调: 1 MHz
稳定后: 5 MHz
再评估: 10 MHz 或更高
```

先保证稳定性，再提高速率。

## 4. SPI 事务模型

SPI 是全双工，但 H7_DM 是从机，不能主动发起传输。

每次 SPI 事务固定收发 512 字节：

```text
MOSI 方向: H7_ZD -> H7_DM，当前请求帧
MISO 方向: H7_DM -> H7_ZD，上一轮准备好的响应帧
```

响应采用“下一帧带回上一帧结果”：

```text
第 N 次事务:
  ZD -> DM: MOTOR_CMD seq=100
  DM -> ZD: 上一次准备好的状态或 ACK

第 N+1 次事务:
  ZD -> DM: HEARTBEAT seq=101 或 MOTOR_CMD seq=101
  DM -> ZD: 对 seq=100 的 ACK/状态
```

因此 ZD 即使没有新的电机命令，也必须周期性发送 `HEARTBEAT`，用于：

- 维持 SPI 链路活性
- 让 DM 判断主机仍在线
- 把 DM 已准备好的状态从 MISO clock 出来

## 5. 固定帧格式

第一版固定传输长度：

```c
#define H7SPI_TRANSFER_SIZE  512
#define H7SPI_MAX_MOTORS     24
#define H7SPI_MAGIC          0xA55A
#define H7SPI_VERSION        1
```

通用帧头：

```c
#pragma pack(push, 1)
typedef struct {
  uint16_t magic;        /* 0xA55A */
  uint8_t  version;      /* 1 */
  uint8_t  msg_type;     /* H7SPI_MSG_xxx */
  uint16_t seq;          /* SPI 帧序号 */
  uint16_t payload_len;  /* payload 有效字节数 */
  uint32_t tick_ms;      /* 发送端 tick，单位 ms */
  uint16_t flags;        /* 第一版填 0 */
  uint16_t crc16;        /* CRC16-CCITT */
} h7spi_header_t;
#pragma pack(pop)
```

帧整体：

```text
h7spi_header_t
payload[payload_len]
padding 到 512 字节
```

CRC 计算范围：

```text
magic/version/msg_type/seq/payload_len/tick_ms/flags + payload
```

不把 `crc16` 字段本身算进去。

## 6. 消息类型

```c
#define H7SPI_MSG_MOTOR_CMD       0x01  /* ZD -> DM，电机控制 */
#define H7SPI_MSG_MOTOR_STATE     0x02  /* DM -> ZD，电机状态 */
#define H7SPI_MSG_SET_MODE        0x03  /* ZD -> DM，切换模式 */
#define H7SPI_MSG_MODE_STATE      0x04  /* DM -> ZD，模式状态 */
#define H7SPI_MSG_ADMIN_OP        0x05  /* ZD -> DM，使能/失能/标零/清错 */
#define H7SPI_MSG_ADMIN_RESULT    0x06  /* DM -> ZD，管理操作结果 */
#define H7SPI_MSG_HEARTBEAT       0x07  /* ZD -> DM，心跳/取状态 */
#define H7SPI_MSG_ACK             0x08  /* DM -> ZD，普通确认 */
#define H7SPI_MSG_ERROR           0x09  /* DM -> ZD，协议错误 */
```

第一阶段必须实现：

```text
HEARTBEAT
ACK
ERROR
MOTOR_CMD
MOTOR_STATE
```

第二阶段再实现：

```text
SET_MODE
MODE_STATE
ADMIN_OP
ADMIN_RESULT
```

## 7. 控制模式

```c
#define H7SPI_MODE_DISABLED       0x00
#define H7SPI_MODE_ADMIN          0x10
#define H7SPI_MODE_CONTROL_MIT    0x21
#define H7SPI_MODE_CONTROL_SPEED  0x22  /* 预留 */
#define H7SPI_MODE_CONTROL_POS    0x23  /* 预留 */
```

第一版实际只支持：

```text
ADMIN
CONTROL_MIT
DISABLED
```

收到暂不支持的模式时，DM 返回 `H7SPI_RESULT_BAD_MODE`。

## 8. MOTOR_CMD 帧

`msg_type = H7SPI_MSG_MOTOR_CMD`

```c
#pragma pack(push, 1)
typedef struct {
  uint8_t  motor_code;    /* 对应 DM 侧 MotorPoint(code) */
  uint8_t  flags;         /* 第一版填 0 */
  int32_t  p_mrad;        /* 目标位置 rad * 1000 */
  int32_t  v_mrad_s;      /* 目标速度 rad/s * 1000 */
  int32_t  torque_mnm;    /* 目标力矩 N*m * 1000 */
  uint16_t kp_centi;      /* Kp * 100 */
  uint16_t kd_milli;      /* Kd * 1000 */
} h7spi_motor_cmd_t;

typedef struct {
  uint16_t command_seq;
  uint8_t  mode;
  uint8_t  motor_count;
  uint32_t host_tick_ms;
  h7spi_motor_cmd_t motors[H7SPI_MAX_MOTORS];
} h7spi_motor_cmd_frame_t;
#pragma pack(pop)
```

单个 `h7spi_motor_cmd_t` 为 20 字节，命令头 8 字节。

如果固定发满 24 个电机：

```text
payload_len = 8 + 24 * 20 = 488 字节
总帧长 = 16 + 488 = 504 字节
```

可以放入 512 字节 SPI DMA 包。

第一版建议 ZD 每帧发送当前配置电机的全部命令，DM 根据 `motor_count` 解析有效数量。

## 9. MOTOR_STATE 帧

`msg_type = H7SPI_MSG_MOTOR_STATE`

```c
#define H7SPI_MOTOR_FLAG_CONFIGURED  (1U << 0)
#define H7SPI_MOTOR_FLAG_ONLINE      (1U << 1)
#define H7SPI_MOTOR_FLAG_ENABLED     (1U << 2)
#define H7SPI_MOTOR_FLAG_FAULT       (1U << 3)

#pragma pack(push, 1)
typedef struct {
  uint8_t  motor_code;
  uint8_t  state;
  uint8_t  flags;
  uint8_t  temperature;
  int32_t  p_mrad;
  int32_t  v_mrad_s;
  int32_t  torque_mnm;
} h7spi_motor_state_t;

typedef struct {
  uint16_t state_seq;
  uint16_t ack_command_seq;
  uint32_t dm_tick_ms;
  uint32_t fault_flags;
  uint16_t enabled_mask_low;
  uint16_t enabled_mask_high;
  uint8_t  mode;
  uint8_t  enabled;
  uint8_t  fault;
  uint8_t  motor_count;
  h7spi_motor_state_t motors[H7SPI_MAX_MOTORS];
} h7spi_motor_state_frame_t;
#pragma pack(pop)
```

ZD 侧通过该帧判断：

- SPI 是否正常
- DM 是否 fault
- 当前是否已使能
- 当前控制模式
- 哪些电机在线
- 哪些电机故障或过温
- DM 最近接受到了哪个 `command_seq`

## 10. ACK 与 ERROR

ACK payload：

```c
#pragma pack(push, 1)
typedef struct {
  uint16_t ack_seq;
  uint16_t result;
  uint16_t command_seq;
  uint16_t error_detail;
} h7spi_ack_frame_t;
#pragma pack(pop)
```

结果码：

```c
#define H7SPI_RESULT_OK               0
#define H7SPI_RESULT_BAD_MAGIC        1
#define H7SPI_RESULT_BAD_VERSION      2
#define H7SPI_RESULT_BAD_CRC          3
#define H7SPI_RESULT_BAD_LENGTH       4
#define H7SPI_RESULT_BAD_TYPE         5
#define H7SPI_RESULT_BAD_MODE         6
#define H7SPI_RESULT_BUSY             7
#define H7SPI_RESULT_FAULT            8
```

ERROR 第一版可以复用 `h7spi_ack_frame_t`，只是 `msg_type = H7SPI_MSG_ERROR`。

## 11. H7_ZD 侧实施方案

### 11.1 SPI Master 初始化

ZD 侧配置 SPI2：

```text
SPI2_SCK  = PB13
SPI2_MOSI = PB15
SPI2_MISO = PB14
CS        = PG10，普通 GPIO 输出，低有效
```

建议先使用阻塞传输或 DMA 传输均可。第一版为了快速验证链路，可以先用阻塞式 `TransmitReceive`，稳定后改 DMA。

单次传输流程：

```text
1. 准备 512 字节 tx buffer
2. CS 拉低
3. SPI TransmitReceive 512 字节
4. CS 拉高
5. 解析 rx buffer
```

### 11.2 ZD 侧任务结构

建议新增任务：

```text
h7spi_master_task
```

任务职责：

```text
1. 周期构造 HEARTBEAT 或 MOTOR_CMD
2. 发起 SPI 传输
3. 解析 DM 返回帧
4. 更新 ZD 本地 DM 状态缓存
5. 统计 CRC 错、超时、seq 跳变
```

建议周期：

```text
控制状态: 当前 DM 侧已用高优先级 SPI 任务由 DMA 完成中断唤醒，可先按 2ms 一次 MOTOR_CMD 验证；有 READY 线并确认无丢包后可压到 1ms
空闲状态: 10ms 一次 HEARTBEAT；未接 READY 线时建议不要低于 2ms
管理操作等待期间: 5ms 一次 HEARTBEAT
```

### 11.3 ZD 侧状态变量

```c
static uint16_t zd_spi_seq;
static uint16_t zd_command_seq;
static uint16_t zd_last_ack_command_seq;
static uint16_t zd_last_state_seq;
static uint32_t zd_last_valid_dm_rx_tick;
static uint32_t zd_crc_error_count;
static uint32_t zd_timeout_count;
static uint8_t  zd_dm_online;
static uint8_t  zd_dm_mode;
static uint8_t  zd_dm_enabled;
static uint8_t  zd_dm_fault;
```

### 11.4 ZD 侧发送策略

无电机控制命令时：

```text
发送 HEARTBEAT
```

有电机控制命令时：

```text
发送 MOTOR_CMD
command_seq 自增
```

如果连续多次没有收到合法 DM 返回帧：

```text
zd_dm_online = 0
上报 SPI 链路异常
继续尝试 HEARTBEAT
```

第一版 ZD 不负责直接失能电机，是否失能由上层策略后续决定。

### 11.5 ZD 侧启动流程

```text
1. 初始化 SPI2 和 CS GPIO
2. 周期发送 HEARTBEAT
3. 等待收到合法 MOTOR_STATE/ACK
4. 确认 DM 在线
5. 后续进入 MOTOR_CMD 周期发送
```

由于当前暂不做 jeston 到 ZD 的网口通信，ZD 第一版可以先发送固定测试命令：

```text
p = 当前测试目标位置
v = 0
torque = 0
kp = 测试 Kp
kd = 测试 Kd
```

先验证：

- SPI 收发正确
- DM 能解析命令
- DM 能更新目标
- DM 状态能回传

## 12. H7_DM 侧实施方案

### 12.1 SPI Slave 初始化

DM 侧配置 SPI1：

```text
SPI1_SCK  = PB03
SPI1_MISO = PB04
SPI1_MOSI = PD07
SPI1_CS   = PE15
```

配置：

```text
SPI Slave
Mode 0
8 bit
DMA RX/TX
NSS 第一版使用 SPI_NSS_SOFT；PE15 作为 GPIO 输入保留物理 CS 线，事务边界由主机 CS 与 512 字节 DMA 长度共同约束
固定收发 512 字节
```

### 12.2 DM 侧双缓冲

建议 DM 侧维护：

```c
static uint8_t spi_rx_buf[H7SPI_TRANSFER_SIZE];
static uint8_t spi_tx_buf[H7SPI_TRANSFER_SIZE];
static uint8_t spi_next_tx_buf[H7SPI_TRANSFER_SIZE];
```

启动时：

```text
1. 清空 rx/tx buffer
2. 构造默认 ACK 或 MOTOR_STATE 到 spi_tx_buf
3. 启动 HAL_SPI_TransmitReceive_DMA()
```

DMA 完成回调里只做轻量操作：

```text
1. 标记 spi_rx_done = 1
2. 通知 h7spi_slave_task
3. 不在中断里解析完整协议
```

### 12.3 DM 侧任务结构

DM 侧新增高优先级 SPI 任务：

```text
h7spi_slave_task
```

任务职责：

```text
1. 由 SPI DMA 完成/错误回调通过 FreeRTOS task notify 唤醒
2. 校验 rx buffer
3. 根据 msg_type 处理 HEARTBEAT/MOTOR_CMD
4. 构造下一次要回传的 tx buffer
5. 重新启动 SPI DMA
```

电机 1ms 任务保持不变：

```text
motor_task:
  MotorApp_Tick()
  osDelay(1)
```

SPI 任务不能阻塞或长时间抢占 `MotorApp_Tick()`。

当前落地实现中，DMA 回调只置位完成/错误标志并通知 `h7spiTask`，协议解析和 `HAL_SPI_TransmitReceive_DMA()` 续挂都在任务上下文完成。这样 SPI 不再等待 1ms motor task 轮询，响应延时主要由 512 字节 SPI 事务时间、DMA 中断到高优先级任务切换时间、协议处理时间组成；电机 CAN 输出仍由 `MotorApp_Tick()` 保持 1ms 节拍。

### 12.4 DM 侧收包处理

DM 校验顺序：

```text
magic
version
payload_len
crc16
msg_type
seq
motor_count
motor_code
mode
```

处理规则：

```text
magic/version/len/crc 错:
  不更新电机目标
  下一帧返回 ERROR

msg_type 不支持:
  不更新电机目标
  下一帧返回 BAD_TYPE

motor_code 非法:
  不更新该电机
  跳过非法或未配置 motor_code

mode 非 CONTROL_MIT:
  不更新电机目标
  下一帧返回 BAD_MODE
```

### 12.5 DM 侧 MOTOR_CMD 写入

收到合法 `MOTOR_CMD` 后：

```cpp
for each command:
  Motor_t *motor = MotorPoint(motor_code);

  p = p_mrad / 1000.0f;
  v = v_mrad_s / 1000.0f;
  t = torque_mnm / 1000.0f;
  kp = kp_centi / 100.0f;
  kd = kd_milli / 1000.0f;

  motor->SetMITCommand(p, v, kp, kd, t);
```

注意：

```text
motor_code 对应 MotorPoint(code)
不是 CAN ID
```

DM 默认配置里当前有效 motor_code 由 `motor_app.cpp` 中 `kMotorAppDefaultConfig` 决定。

### 12.6 DM 侧状态回传

DM 每次处理完 SPI RX 后，准备下一帧 `MOTOR_STATE`。

状态数据来源：

```text
MotorApp_GetStatus()
MotorPoint(code)->IsConfigured()
MotorPoint(code)->IsOnline()
MotorPoint(code)->IsEnable()
MotorPoint(code)->GetState()
MotorPoint(code)->GetRotorTemp()
MotorPoint(code)->GetPosition()
MotorPoint(code)->GetSpeed()
MotorPoint(code)->GetTorque()
```

关键字段：

```text
ack_command_seq: 最近成功接受的 MOTOR_CMD command_seq
fault_flags: MotorApp fault flags
mode: MotorApp 当前模式
enabled: MotorApp 是否已使能
fault: MotorApp 是否 fault
motor_count: 默认配置电机数量
```

### 12.7 DM 侧通信保护

DM 维护：

```c
last_valid_spi_tick
last_valid_motor_cmd_tick
last_accepted_command_seq
spi_crc_error_count
spi_bad_len_count
spi_bad_motor_count
spi_seq_drop_count
```

超时策略第一版采用保持最后有效命令：

```text
超过 CMD_TIMEOUT_MS 没有新 MOTOR_CMD:
  不清零目标
  不主动失能
  继续下发最后一次有效命令
  状态帧上报 CMD_TIMEOUT 标志

超过 LINK_TIMEOUT_MS 没有任何有效 SPI 包:
  不主动失能
  继续保持最后一次有效命令
  状态帧上报 LINK_LOST 标志
```

电机 CAN 掉线仍由 DM 本地 MotorApp 保护：

```text
电机反馈超过 MOTOR_APP_FEEDBACK_TIMEOUT_MS 未更新
  -> MotorApp fault
  -> DisableAllRoutes
  -> 状态帧上报 fault_flags
```

## 13. 第一阶段联调步骤

### 13.1 不接电机，只测 SPI

目标：

```text
ZD 能发 HEARTBEAT
DM 能收到 HEARTBEAT
DM 能回 ACK 或 MOTOR_STATE
ZD 能解析 DM 返回
CRC/seq 正常
```

建议测试：

```text
1. ZD 每 10ms 发送 HEARTBEAT
2. DM 收到后计数
3. DM 返回 state_seq 自增的 MOTOR_STATE
4. ZD 打印 state_seq、dm_tick_ms
```

### 13.2 接电机但不闭环控制

目标：

```text
DM 正常 MotorApp_ConfigDefault()
DM 正常 MotorApp_EnableDefault()
ZD 能读取 DM enabled/mode/fault
```

测试：

```text
1. DM 上电自动配置并使能默认电机
2. ZD 周期 HEARTBEAT
3. ZD 检查 MOTOR_STATE:
   mode == CONTROL_MIT
   enabled == 1
   fault == 0
```

### 13.3 固定目标命令测试

目标：

```text
ZD 发送固定 MOTOR_CMD
DM 写入 MotorPoint(code)->SetMITCommand()
电机收到 CAN MIT 控制帧
状态帧 ack_command_seq 正确
```

测试命令建议保守：

```text
p = 当前附近小角度
v = 0
torque = 0
kp = 小 Kp
kd = 合理阻尼
```

### 13.4 异常测试

需要验证：

```text
CRC 错误包不会更新目标
非法 motor_code 不会写入目标
mode 错误不会写入目标
ZD 停发后 DM 保持最后有效命令
电机掉线后 DM 进入 fault 并上报
```

## 14. 建议实现文件

H7_ZD：

```text
h7spi_protocol.h/.c
h7spi_master.h/.c
h7spi_test_app.h/.c
```

H7_DM：

```text
h7spi_protocol.h/.c
h7spi_slave.h/.c
h7spi_motor_bridge.h/.cpp
```

如果两边工程结构允许，`h7spi_protocol.h/.c` 应尽量保持完全一致，避免结构体和 CRC 实现不一致。

## 15. 第一版完成标准

认为第一版 SPI 通信完成，需要满足：

```text
1. ZD 与 DM 固定 512 字节 SPI 收发稳定
2. ZD HEARTBEAT 能被 DM 正确解析
3. DM MOTOR_STATE 能被 ZD 正确解析
4. ZD MOTOR_CMD 能更新 DM 电机目标
5. DM ack_command_seq 能正确回传最近接受的 command_seq
6. CRC 错误包不会影响电机目标
7. ZD 停发后 DM 不清零、不失能，保持最后有效命令
8. 电机 CAN 掉线后 DM 本地 fault 能通过 SPI 状态回传给 ZD
```
