# STM32 电机命令与 CAN 反馈短时采集

采集模块在标准固件中默认编入（`H7DM_CAN_CAPTURE=ON`），可显式设为 `OFF` 构建不带采集缓冲的镜像。默认选中电机
2、7、12；电脑端 `--motors` 可调整。STM32 在 RAM 中分别保存：

- 收到并提交到异步控制入口的主机命令：16 位指令序号，原始目标位置、速度、Kp、Kd、前馈力矩。后续命令可能在应用前覆盖待处理命令；是否实际下发以 TX 记录的序号为准。
- 正常 MIT 帧成功进入 FDCAN TX FIFO 的时间、原始 8 字节数据与该电机最近已应用的指令序号。
- 有效电机反馈回调入口的时间、原始 8 字节数据与最近下发的指令序号。

目标值使用 DMUSB 的整数单位，无浮点运算和格式化打印。采集窗口由 HAL 毫秒 tick
截止，所有事件由 STM32 DWT cycle 计时；导出时统一转为相对采集起点的微秒时间。
采集期间不经 DAP 传输数据，结束后才分块读取 RAM。发送时间是 **TX FIFO 入队成功时间**，
不是 CAN 总线确认发送完成的硬件时间。`samples.csv` 取同一电机本次 TX 之后、下次 TX
之前的第一条 RX；CAN 反馈帧不带指令序号，所以这种关联是时间上的近邻，不能证明因果对应。
完整原始事件仍保存在 `events.bin`/`events.csv`，便于重新分析。

AXI SRAM 中的事件缓冲为 18,432 × 16 字节，主机目标缓冲为 768 × 24 字节，
共约 306 KiB。所选 3 台每台双向约 1 kHz 时，事件缓冲约覆盖 3.072 秒；目标缓冲
在每台 50 Hz 主机命令下约覆盖 5.12 秒。超出容量后停止存入新记录，分别累加丢失计数。
抓取时长可设 1–7000 ms，但超过缓冲容量的记录不会保留。
当前默认窗口为 3000 ms。电脑发出启动命令后，第一条选中电机的记录作为采集起点；
到时由电机任务在 1 ms 周期检查并结束采集，即使实验已停止、没有新的 CAN 帧。
完成后 RAM 中的记录保持不变；导出脚本完成读取后，下一次运行才会重新启动采集，
届时首条选中电机的记录会清零计数并开始覆盖旧记录。STM32 复位也会清除这段记录。
不会自动开启下一段。若缓冲提前写满，新记录只计入丢失计数，不覆盖旧记录。

构建命令：

```sh
cmake -S 10-source/stm32/stm32-h7-dm/CtrBoard-H7_ALL \
  -B 20-build/stm32/stm32-h7-dm/battery-can-capture-release \
  -DCMAKE_TOOLCHAIN_FILE=cmake/arm-none-eabi-gcc.cmake \
  -DCMAKE_BUILD_TYPE=Release -DH7DM_CAN_CAPTURE=ON -DH7DM_BENCH_ID1_ONLY=OFF
cmake --build 20-build/stm32/stm32-h7-dm/battery-can-capture-release
```

当前合并镜像同时包含 USART1 电池表轮询、USB 状态上报、CAN 原始数据短时采集
和已对齐的 IMU 输出；匹配的 ELF 保存在
`30-artifacts/stm32/stm32-h7-dm/imu-phase-2026-09-21/`。
采集脚本不烧录、
不复位、不暂停 MCU。脚本必须使用与板上固件完全一致的 ELF。

2026-09-18 已将该合并镜像烧录到 STM32H723，写后校验通过。机器人保持
`relaxed` 时，用电机 1、11、12 做 1.5 秒被动采样，收到 66 条 CAN 反馈，
丢失 0 条；放松状态下无控制命令与 CAN 下发，所以本次未验证 TX/命令记录。
同时 RK3566 继续收到电池电压、电量和报警位。

2026-09-21 刷写了 IMU 通道同相位及积分时间步修复版，OpenOCD 写后校验通过；
robotd 重连后 IMU 就绪，全部电机保持失能。运动中的相位改善仍需新记录验证。

烧录后抓取示例：

```sh
python3 60-tools/capture-stm32-can.py --motors 2,7,12 --duration-ms 3000 \
  --elf 30-artifacts/stm32/stm32-h7-dm/imu-phase-2026-09-21/CtrBoard-H7_ALL.elf
```

输出位于 `50-logs/test/motor/gf43x40-10-i2rt/data/can-capture/<UTC时间>/`：

| 文件 | 内容 |
|---|---|
| `commands.bin`、`commands.csv` | 原始主机目标、接收时间和指令序号 |
| `events.bin`、`events.csv` | 原始 CAN 下发、反馈数据及 STM32 时间 |
| `samples.csv` | 按电机与时间关联后的目标、实际 CAN 编码值、反馈值和延迟 |
| `metadata.json` | 采集起点、时钟、筛选、电机数量、丢失数量及关联规则 |

反馈位置/速度/力矩按当前 GF43X40-10 的 CAN 量化范围 ±12.5 rad、±10 rad/s、
±28 N·m 解码。力矩列是电机 CAN 反馈的估算力矩；当前反馈帧不含相电流原始值。
