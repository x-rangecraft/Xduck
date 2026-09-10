# MicroDuck BLE App v1 协议

适用实现：`10-source/android/duck-ble/` 和 Rust `btd/src/mobile.rs`。
机器人仍由 robotd 唯一拥有状态、安全裁决和电机总线；btd 只做协议适配、连接会话和状态压缩。控制循环不改频率（50 Hz），STM32 自有失联保护保持不变。

## GATT 与连接顺序

- Service：`6f5d2a10-3b47-4c8e-9a1f-2d7e8c4b6019`
- RPC / write / notify：`6f5d2a11-3b47-4c8e-9a1f-2d7e8c4b6019`
- CCC：标准 `0x2902`
- 顺序：发现服务 → bond → MTU（请求 247，实际值由系统决定）→ 读 API 单字节 → CCC 订阅 → `system.authenticate {pin}` → `ble.info` → `ble.subscribe`。
- Android 14+ 可能强制协商 MTU 517；一切写入以实际回调的 MTU−3 切片。机器人回复按 20 字节保守发送。
- 默认要求 BLE 加密；`--insecure-no-pairing` 仅保留为显式开发选项，禁止用于手机控制交付。
- 链路是 Just Works 加密加应用层 PIN 校验，不是基于显示/输入的 MITM 认证，也不是自定义端到端密码协议。
- BlueZ 的同一 notify pipe 绑定首个写入设备地址，其他地址不能借用已认证会话。实际 writer 断线事件也会销毁会话；不会仅依赖通知订阅是否仍存在。

## JSON-RPC

设置、认证、动作、低频健康数据用 UTF-8 NDJSON（单行最长 8192 字节）。仅在 JSON 组帧缓冲为空时识别二进制 magic，不能把 UTF-8 续字节 0xBD/0xBE 误当作新二进制帧。不得将同一个 JSON 帧的多个片段和二进制移动写入交错。

| 方法 | 参数 | 响应 / 用途 |
| --- | --- | --- |
| `system.authenticate` | `{pin:"六位数字"}` | 复用原有 PIN 校验和三次失败限制 |
| `ble.info` | `{}` | `v=1, firmware, api, ms, control_hz=10, state_hz=2, deadman_ms=500` |
| `ble.subscribe` | `{}` | 每个连接只打开一个 `robot.subscribe {hz:2}`；返回 accepted |
| `ble.health` | `{}` | 精简 robot.health：healthy、degraded、reason（最多160字符）、battery、cpu_temp_c |
| `ble.mode` | `{}` | 转发 robot.mode，只读 |
| `ble.claim` | `{}` | 认证后申请蓝牙租约；返回随机 uint32 `s` 和会话时钟 `ms`；手柄连接时蓝牙待命，否则取得控制并清零速度 |
| `ble.release` | `{s,q,t}` | 撤销蓝牙控制权并释放租约；保留物理连接；如蓝牙正在控制则清零速度 |
| `ble.command` | `{s,q,t,m,p}` | 包装白名单 robot 指令，转发 robotd 的真实接受/拒绝结果 |
| `ble.move` | `{s,q,t,v:[整数,整数,整数]}` | 二进制移动的 JSON 等价形式，主要用于测试 |

`q` 从 1 递增，速度、动作、释放共享一个序号空间；`t` 是此控制租约的机器人单调时钟毫秒。重新接管生成新的 `s` 并从 0 开始计时，拒绝旧租约、重复/倒序序号、超过 250 ms 的历史时间和超过 30 ms 的未来时间。客户端以回复中的服务器时间锚定本地接收时刻（保守地低估服务器时间，避免非对称链路产生未来时间），接管往返超过 200 ms 就不进入操控。

移动常规包间隔至少 80 ms（容纳 10 Hz 调度抖动），零速度不受该间隔限制；动作至少间隔 200 ms。零速度立即生效，其他动作不可更新移动 deadman。500 ms 没有新的有效速度包，必须重新接管，不接受一串迟到包恢复旧租约。

动作白名单：robot.head（head_pitch ±0.4、head_yaw ±0.6 rad，neck_pitch/head_roll 为0）、robot.init、robot.relax、robot.enable、robot.do、robot.sound、robot.stop。高风险确认在 App 端；robotd 继续做原有校验。Raw robot.move 在 BLE 路由被禁止，不能绕过新会话。

## 20 字节移动帧

每次使用一次 ATT write-with-response，正好20字节，不能分片，不包含换行。所有多字节整数为 little-endian。

| 偏移 | 长度 | 字段 |
| --- | --- | --- |
| 0 | 1 | magic `0xBD` |
| 1 | 1 | version `1` |
| 2 | 4 | s uint32 |
| 6 | 4 | q uint32 |
| 10 | 4 | t uint32，机器人租约时钟毫秒 |
| 14 | 2 | vx int16，mm/s，±300 |
| 16 | 2 | vy int16，mm/s，±300，向左为正 |
| 18 | 2 | vyaw int16，mrad/s，±1000，左转为正 |

成功无逐包 JSON 回复；拒绝时发 `ble.error` 通知，App 立即释放控制。正常运动实时性由 robotd 独立 watchdog 保证，不能以收到了 ATT ACK 推断电机已经执行。

## 188 字节状态帧

状态内部全部多字节 little-endian。量化 int16 的 `-32768` 为未知（缺失、非有限或超出表示范围），不做静默饱和后冒充测量。uint16 age=65535 为未知/过期。

| 偏移 | 长度 | 含义 |
| --- | --- | --- |
| 0 | 1 | version 1 |
| 1 | 1 | policy：0未知、1 walk、2 stand、3 held |
| 2 | 1 | source：0未知、1 bluetooth、2 gamepad、3 keyboard、4 drag |
| 3 | 1 | safety flags：bit0 fallen、bit1 limp |
| 4 | 4 | robotd t ×1000，uint32（重启会归零） |
| 8 | 2 | 控制循环 Hz ×10，int16 |
| 10 | 6 | 请求速度 vx/vy/vyaw ×1000，3×int16 |
| 16 | 6 | 里程计估计实际速度 ×1000，3×int16 |
| 22 | 6 | IMU roll/pitch/yaw ×1000，3×int16 |
| 28 | 6 | 重力投影 ×1000，3×int16 |
| 34 | 154 | 14 电机，每个11字节 |

电机：flags u8、位置rad×1000 i16、速度rad/s×1000 i16、力矩Nm×1000 i16、温度°C×10 i16、反馈age毫秒u16。
flags 复用现有 DMUSB 映射：bit0 configured、bit1 online、bit2 enabled、bit3 fault、bit5 expired。App 同时检查 age≤500 ms。
顺序是 `JOINT_NAMES` 去掉 index 9 的 mouth：左髋yaw/roll/pitch、左膝、左踝、neck pitch、head pitch/yaw/roll、右髋yaw/roll/pitch、右膝、右踝。UI 行号是显示顺序，不是 CAN ID。

每个通知前放4字节片头：`[0xBE, frame_id低字节, frame_id高字节, fragment_index]`；每片最多16字节有效载荷。总计12片，最后一片12字节有效载荷（通知16字节）。从 index 0 重新组帧；缺片、乱序、长度错误、版本错误均不呈现。状态在服务端使用 latest-only watch 槽；发送队列拥塞时丢弃整帧，不能积压旧历史。

电池、主板温度、健康状态通过2 Hz精简 ble.health 补充。API/固件服务版本在连接握手显示。核心二进制状态约464 ATT payload字节/秒（188+12×4，每秒2帧），加移动200字节/秒，再加精简健康/RPC帧；射频开销和实际延迟需真机测量。

## 参考

Android 官方：[蓝牙权限](https://developer.android.com/develop/connectivity/bluetooth/bt-permissions)、[BluetoothGatt](https://developer.android.com/reference/android/bluetooth/BluetoothGatt)、[AGP 8.9](https://developer.android.com/build/releases/agp-8-9-0-release-notes)。

## 控制源仲裁

`robotd` 固定采用手柄 > 有效蓝牙租约 > 正在拖动的网页摇杆 > 正在按下的键盘。手柄重连立即取得优先权，居中仍占用控制权。蓝牙连接本身不占用控制权；`ble.claim` 申请，10 Hz 心跳续期，`ble.release`、断连或 500 ms 超时撤销。网页拖动和键盘用 500 ms 活跃租约，松开立即撤销；都空闲时显示网页拖动。

IPC `robot.move` 的 `select_source` 只能申请本来源，不能越过更高优先级；`release_source: true` 撤销软件来源租约，不改变物理连接状态。网页只允许 `drag` / `keyboard` 来源，禁止伪造手柄、蓝牙或 `source_available`。切换控制源先清零速度，旧来源的停车包也不能覆盖新来源。
