# RK3566 电机文件任务接口 v3

电脑提前生成完整指令文件，上传并校验后单独启动。robotd 在本地 50 Hz 控制循环执行，Wi-Fi 不参与指令定时；断网继续执行既定任务，不重发、不自动重启任务。结束、停止或本地保护触发后全部失能。旧实时 acquire / command / release / feedback 接口已移除。

## 操作流程

Python 3.10+，配套 motor_experiment.py 只使用标准库。

```sh
python3 motor_experiment.py http://192.168.50.121:8080 capabilities
python3 motor_experiment.py http://192.168.50.121:8080 upload task.jsonl
# 上传只保存和校验，不使能。检查姿态、支撑条件后明确启动：
python3 motor_experiment.py http://192.168.50.121:8080 start 返回的任务ID
python3 motor_experiment.py http://192.168.50.121:8080 status 返回的任务ID
# 可提前停止；必须使用 start 响应中的 run_token
python3 motor_experiment.py http://192.168.50.121:8080 stop 返回的任务ID 返回的run_token
# 结果就绪后下载：写入电脑、校验大小/SHA-256、同步磁盘，然后确认删除设备副本
python3 motor_experiment.py http://192.168.50.121:8080 download 返回的任务ID result.tar
# --keep 可保留设备副本；之后可用本地完整文件再次校验并确认删除
python3 motor_experiment.py http://192.168.50.121:8080 acknowledge 返回的任务ID result.tar
```

上传、启动、查询和下载相互独立。电脑可在启动后退出，重连后通过 list 查找任务。`run_token` 只在成功启动响应中返回，调用者应在本次运行结束前保存在内存或受限临时文件中。启动请求的回复丢失时先查询任务；同一正在执行的任务再次 start 会被拒绝，已终止任务不能再次执行，必须重新上传。不要批量自动重试未知结果的启动操作。

## 指令文件

UTF-8 NDJSON，一行一个 JSON 对象，末行也必须有换行符。第一行为任务头，例如：

```json
{"version":1,"period_ms":20,"duration_ms":40,"motor_ids":[2],"start_tolerance_rad":0.05,"max_tracking_error_rad":0.3,"max_temperature_c":60}
```

之后每个控制周期一行，严格 at_ms=0,20,40,...,duration_ms-20，不能缺帧、重复或额外添加帧。每行包含全部选中电机，顺序不限。以下两行仅演示格式，不是现场运动建议：

```json
{"at_ms":0,"motors":[{"motor_id":2,"p":0,"v":0,"tau":0,"kp":0,"kd":0}]}
{"at_ms":20,"motors":[{"motor_id":2,"p":0,"v":0,"tau":0,"kp":0,"kd":0}]}
```

不会插值、平滑、跳帧或修改轨迹。若某次控制 tick 提前到达下一帧时间，保持上一帧参数并记录该周期，绝不提前执行下一帧。到任务结束时间后失能；不会无限保持最后一条。Linux 调度存在抖动，记录真实时间，不承诺硬实时。

| 参数 | 单位或范围 |
|---|---|
| p | 相对于保存硬件零位的 rad，范围由 capabilities 查询 |
| v | rad/s；ID 1、2 最大绝对值 15.708；ID 13、14、15 为 10 |
| tau | 前馈 N·m；ID 1、2 最大绝对值 10；ID 13、14、15 为 28 |
| kp | N·m/rad，0～500 |
| kd | N·m/(rad/s)，0～5 |
| start_tolerance_rad | 显式设置，>0 且 ≤0.5；启动姿态与首条指令允许偏差 |
| max_tracking_error_rad | 显式设置，>0 且 ≤3；执行中反馈与当前目标位置允许偏差 |
| max_temperature_c | 显式设置，1～80 ℃；所有配置电机都检查 |

位置、速度、前馈在线缆上的分辨率为 0.001，kp 为 0.01，kd 为 0.001。STM32 保留总力矩限幅，记录的请求增益不等于固件限幅后的实际输出。

当前固件仍只能统一使能全部配置电机。未选中电机发送 kp=kd=v=tau=0，处于放松状态而非硬件失能；承重关节必须由现场人员支撑，或将需要保持姿态的关节明确包含在每一帧中。所有电机在同一 STM32 命令帧提交，CAN 仍逐帧发送，不承诺硬件级同时采样。

## 校验与保护

上传校验：声明大小、SHA-256、完整文件格式、时间序列、完整电机集合、参数范围、预计存储空间。失败不进入 ready，更不会使能。

启动再次校验文件与当前标定；控制线程检查放松、全部配置电机失能、新鲜反馈、IMU、限位就绪、当前姿态与首帧匹配、温度。任务运行时其他普通控制和标定不能覆盖它，原有停止/放松按钮仍可终止任务。

本地保护：反馈/IMU异常、离线/故障、跟踪误差、过温、参数/标定异常、控制间隔超过 40 ms 或相对计划迟到超过 40 ms、预读断供、记录队列溢出或文件 I/O 失败都会终止。STM32 自身 100 ms 主机命令看门狗保留。Wi-Fi 断开不触发终止，但无线停止请求可能延迟；本地保护始终有效。

进程或设备重启后，之前正在执行的任务标记 interrupted，绝不恢复运动；保留可用的部分记录。重启前未成功落盘的末尾记录可能丢失，interrupted 不能当成完整成功实验。

## 存储与结果

输入最多 128 MiB，单行最多 4096 bytes，单任务最长 30 分钟，最多保留 8 个任务，总预算 2 GiB。为输入、原始记录和最终 tar 预留空间，另保留至少 512 MiB 空闲磁盘余量。达到限制时拒绝新任务，不自动删除未下载的结果。未完成上传超过一小时可在后续上传时清理。

输入预读和记录写盘分别最多缓存 64 帧，由独立文件线程处理。不会把完整任务或整场反馈装进内存，也不会让控制线程等待磁盘。

`result.tar` 包含：

- `input.jsonl`：原始完整任务。
- `records.jsonl`：每个执行周期的计划帧、真实时间、tick、实际发往 STM32 的完整参数、wire_seq、采集反馈及 gateway_tick_ms / ack_command_seq。
- `status.json`：任务参数、执行帧数、结束原因、是否确认失能；包的 SHA-256 在查询响应和下载头中提供，避免自引用。

同一周期先采反馈再发命令，不能把同一行理解为零延迟响应。ack_command_seq 是网关接收序号，会回绕，不是每台电机动作完成确认。取消任务没有执行周期，包中可没有 records.jsonl。

电脑先下载到 .part，校验大小和 SHA-256、同步本地磁盘，再发布为最终文件，最后提交校验确认。失败不会删除设备数据；若文件已保存而确认请求失败，用 acknowledge 重试即可。直接浏览器下载不会自动删除设备副本。

## HTTP

除下载文件外，成功响应为直接 JSON 对象；错误为非 200 和 error 字段。HTTP 是局域网接口，没有用户登录鉴权，不允许浏览器跨站控制。

| 方法和路径 | 作用 |
|---|---|
| GET /api/motor-experiment/capabilities | 能力、可启动状态、新鲜 current_motors 快照 |
| GET /api/motor-experiment/tasks | 列表与存储限制 |
| POST /api/motor-experiment/tasks/upload | 文件流；Content-Type: application/x-ndjson，Content-Length，X-Content-SHA256 |
| GET /api/motor-experiment/tasks/{id} | 状态、进度、结果大小与 SHA-256 |
| POST /api/motor-experiment/tasks/{id}/start | 单独启动，空 JSON 对象；成功响应包含本次运行的 run_token |
| POST /api/motor-experiment/tasks/{id}/stop | 停止当前运行：JSON {"run_token":"启动响应中的令牌"} |
| GET /api/motor-experiment/tasks/{id}/result | 流式 tar；Content-Length 和 X-Content-SHA256 |
| POST /api/motor-experiment/tasks/{id}/acknowledge | 删除：JSON {"sha256":"已下载并校验的结果哈希"} |

同一时刻只允许一个任务运行。任务开始后，其他脚本的启动请求会被拒绝；停止还必须携带该次启动返回的 `run_token`，因此只知道任务 ID 的脚本不能打断它。令牌丢失时任务仍按既定时长运行，现场仍可通过机器人停止入口触发安全停止。每次一个上传、最多两个结果下载。中断下载可以重新下载；不在设备额外生成下载副本。记录与任务文件的唯一所有者是 robotd，mediad 只转发。

## 本体日志

本体 systemd Journal 与实验记录分开，500 MB 上限保持不变：

```sh
python3 motor_experiment.py http://192.168.50.121:8080 logs rk3566.jsonl
```

GET /api/logs 支持 format=text/jsonl、since/until=Unix秒、unit=all 或 robotd/mediad/configd/btd/padd/tofd/updaterd/xpad-usbd/irqbalance。省略 until 取请求开始时间，省略 since 导出全部尚存历史。已轮转删除的历史不能恢复。
