# Xduck 策略实验接口 v3

同一接口支持 `inference_only`（真机反馈连续推理，策略输出不下发，电机保持默认站姿）和 `closed_loop`（策略输出经过平台安全层后下发）。默认 `initial_pose="saved_home"`：两种模式先复用机器人初始化流程，使能电机并到达保存的默认站姿。闭环模式还支持 `initial_pose="zero_torque"`：使能电机后连续发送全配置电机的零力矩 MIT 命令（`kp=0`、`kd=0`、`tau=0`、`v=0`），不执行默认站姿插值；操作者可以手动摆姿势。两条路径均须等待 `ready` 后再明确 `start`，实验时钟才开始；零力矩路径从当前真机姿势直接启动策略。

策略导入、电机实验和策略实验共享一个底层独占租约。除自由实验 ready/running 阶段外，任一操作占用期间，普通移动、姿态、策略开关、动作、声音、标定等请求均由 robotd 拒绝；只保留状态读取、当前操作自身的管理请求、放松和关机。`robot.state.control_owner` 显示 `policy_import`、`motor_experiment` 或 `policy_experiment`，`control_state` 进一步显示策略实验所处阶段。

## 自由实验

仅设置实验时长即可创建：

```json
{"mode":"free", "duration_ms":60000}
```

时长须为 20 ms 的正整数倍，最长 30 分钟。沿用 `configure → initialize → start → download`；自由模式 initialize 立即 ready，不改变当前控制、不使能、不回站姿。ready 和 running 期间，网页、手柄、蓝牙及普通控制接口均可正常操作（包括坐下、起立、放松和重新初始化）。模型导入、标定和其他实验仍互斥。`robot.state.policy_experiment_manual_control=true` 表示当前可手动控制。

自由模式不接受指令段、固定选网或零力矩初始化；使用正常控制与安全门禁。到时或带 token 的 stop 只结束记录并释放租约，机器人继续当前操作；放松可继续记录，关机结束记录。没有策略输出时同样记录保持/未使能帧；`policy_context.action="hold"`，`applied` 表示该帧总线目标写入是否成功，不等价于电机已使能。没有新鲜反馈的周期无有效采样。

以下初始化、回站姿、故障停止和普通控制锁定规则适用于原有两种受控实验模式。

## 快速操作

```sh
python3 policy_experiment.py http://192.168.50.121:8080 configure example.json
python3 policy_experiment.py http://192.168.50.121:8080 initialize 返回的ID
# 轮询 status，看到 ready 后：
python3 policy_experiment.py http://192.168.50.121:8080 start 返回的ID
# 可用 start 返回的 run_token 提前停止：
python3 policy_experiment.py http://192.168.50.121:8080 stop 返回的ID run_token
python3 policy_experiment.py http://192.168.50.121:8080 download 返回的ID result.jsonl
```

## 手动摆姿势／翻倒起身验证

配置增加以下字段，其他配置不变：

```json
{"mode":"closed_loop", "policy":"stand", "initial_pose":"zero_torque", "stop_on_fall":false}
```

该片段需合并到包含 `segments` 和 `max_temperature_c` 的完整配置。操作顺序仍是 `configure → initialize → status=ready → 手动摆姿势 → start`。`initialize` 在这条路径中只做健康检查、使能和零力矩准备，不会先拉回默认站姿。零力矩以发送的命令为准，不要求反馈测得的力矩恰好为零；只写 `tau=0` 而保留非零 `kp/kd` 会产生位置或阻尼力矩。

翻倒测试必须显式设置 `stop_on_fall=false`，否则倒地后的首个执行周期就会触发实验停止。策略实验占用期间暂停普通 `limp_fall` 恢复流程，避免它用站姿插值替代待测策略。关节限位、反馈、IMU、使能一致性和温度检查仍保留；这项接口允许测试策略，起身能力取决于选中的模型及其消费文件。

零力矩启动只支持 `closed_loop`，`inference_only` 继续保持默认站姿。零力矩 ready 仍有 240 秒期限；准备失败或 ready 超时会失能并释放控制权。主动 `stop`（ready 时不需要 token）、运行结束及普通停止仍返回默认站姿；需要直接失能时使用放松。

配置最多 256 个指令段，总时长最多 30 分钟，每段时长必须是 20 ms 的正整数倍。`policy` 为 `auto`、`walk` 或 `stand`；`auto` 保持正常的速度幅值选网逻辑，另两个值固定测试相应槽位。每段可包含 `twist`（前进、左移、偏航角速度）、`head`（4 关节目标）和 `body`（z、roll、pitch），省略的字段为零。可选的 `action` 在该段第一帧触发一次，取值为 `sit_toggle`、`ground_pick`、`kick_left`、`kick_right` 或 `roulade`。动作复用普通控制器的状态机和优先级；动作完成后恢复该实验配置的普通站立/行走选择，坐下则需在后续段再次发送 `sit_toggle` 才会站起。若动作策略未配置，创建实验时直接拒绝；若动作因另一个动作仍在运行而被控制器拒绝，实验以失败结束并返回默认站姿。

例如在一个指令段起点触发左踢：

```json
{"duration_ms": 2000, "action": "kick_left", "twist": [0.0, 0.0, 0.0]}
```

平台在接收配置时限制 vx/vy 为 ±0.3 m/s、vyaw 为 ±1.5 rad/s，body 保持在训练范围内；危险或非有限输入不会进入就绪状态。

`inference_only` 仍会更新策略内部历史，因此能连续检查有状态策略；但策略动作没有形成真实运动反馈，结果不能当作闭环轨迹。记录中的 `policy_targets` 是策略建议，`applied_targets` 是实际保持站姿时发送的目标，`applied=false` 明确表示策略建议没有下发。

结果为 NDJSON：首行是配置头及实际运行的策略文件/导入版本，中间每帧记录请求命令、传给消费文件的基础命令、`policy_context`、完整真机反馈、策略目标、实际目标及安全限制，末行是结束状态和原因。模型 Obs 与原始输出属于版本化消费文件的内部语义，不再由 robotd 假定固定形状或重复记录。客户端先校验 Content-Length 与 SHA-256，再确认删除设备副本。

## HTTP

| 方法和路径 | 作用 |
|---|---|
| GET `/api/policy-experiment/capabilities` | 能力、当前真机快照和占用者 |
| GET/POST `/api/policy-experiment/sessions` | 列出实验／创建实验配置 |
| GET `/api/policy-experiment/sessions/{id}` | 查询生命周期、帧数和结果摘要 |
| POST `.../{id}/initialize` | 申请独占、清空普通意图，按 initial_pose 准备默认站姿或使能零力矩 |
| POST `.../{id}/start` | 从 ready 开始实验，返回 run_token |
| POST `.../{id}/stop` | 使用 run_token 提前停止并返回默认站姿 |
| GET `.../{id}/result` | 下载带 SHA-256 的 NDJSON 结果 |
| POST `.../{id}/acknowledge` | 校验哈希后删除设备结果 |

反馈、IMU、电机控制、温度、推理、写总线或记录队列异常都会结束实验。准备 30 秒仍未达到健康 ready 状态会失败；到达 ready 后 240 秒未开始会自动释放。默认 `stop_on_fall=true`；到时或主动停止后先退出策略并保持默认站姿，确认回到站姿后才释放独占。放松和关机可随时抢占策略实验。

策略往返超过 10 ms 时，`robotd` Journal 记录模型槽位、帧序号、往返耗时，以及 Python worker 的前处理、输入转换、ONNX、输出处理和校验耗时。18 ms 超时时，Journal 记录超时帧并请求 worker 将 Python 堆栈写入同一 Journal，然后终止失效 worker。实验结果文件只包含成功完成的帧；定位失败帧时需同时查看 `journalctl -u robotd.service`。
