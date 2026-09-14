# Xduck 策略实验接口 v1

同一接口支持 `inference_only`（真机反馈连续推理，策略输出不下发，电机保持默认站姿）和 `closed_loop`（策略输出经过平台安全层后下发）。两种模式都会先复用机器人初始化流程，使能电机并到达保存的默认站姿；初始化达到 `ready` 后，调用方必须再明确 `start`，实验时钟才开始。

策略导入、电机实验和策略实验共享一个底层独占租约。任一操作占用期间，普通移动、姿态、策略开关、动作、声音、标定等请求均由 robotd 拒绝；只保留状态读取、当前操作自身的管理请求、放松和关机。`robot.state.control_owner` 显示 `policy_import`、`motor_experiment` 或 `policy_experiment`，`control_state` 进一步显示策略实验所处阶段。

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

配置最多 256 个指令段，总时长最多 30 分钟，每段时长必须是 20 ms 的正整数倍。`policy` 为 `auto`、`walk` 或 `stand`；`auto` 保持正常的速度幅值选网逻辑，另两个值固定测试相应槽位。每段可包含 `twist`（前进、左移、偏航角速度）、`head`（4 关节目标）和 `body`（z、roll、pitch），省略的字段为零。平台在接收配置时限制 vx/vy 为 ±0.3 m/s、vyaw 为 ±1.5 rad/s，body 保持在训练范围内；危险或非有限输入不会进入就绪状态。

`inference_only` 仍会更新策略内部历史，因此能连续检查有状态策略；但策略动作没有形成真实运动反馈，结果不能当作闭环轨迹。记录中的 `policy_targets` 是策略建议，`applied_targets` 是实际保持站姿时发送的目标，`applied=false` 明确表示策略建议没有下发。

结果为 NDJSON：首行是配置头及实际运行的策略文件/导入版本，中间每帧记录命令、完整真机反馈、标准策略的 61 维观测和 14 维原始输出、策略目标、实际目标及安全限制，末行是结束状态和原因。两文件自定义策略的预处理由隔离 worker 拥有，因此标准 `observation/raw_action` 字段为空；同一帧仍保留传给 worker 的完整传感器、命令和最终关节目标。客户端先校验 Content-Length 与 SHA-256，再确认删除设备副本。

## HTTP

| 方法和路径 | 作用 |
|---|---|
| GET `/api/policy-experiment/capabilities` | 能力、当前真机快照和占用者 |
| GET/POST `/api/policy-experiment/sessions` | 列出实验／创建实验配置 |
| GET `/api/policy-experiment/sessions/{id}` | 查询生命周期、帧数和结果摘要 |
| POST `.../{id}/initialize` | 申请独占、清空普通意图并初始化到默认站姿 |
| POST `.../{id}/start` | 从 ready 开始实验，返回 run_token |
| POST `.../{id}/stop` | 使用 run_token 提前停止并返回默认站姿 |
| GET `.../{id}/result` | 下载带 SHA-256 的 NDJSON 结果 |
| POST `.../{id}/acknowledge` | 校验哈希后删除设备结果 |

反馈、IMU、电机控制、温度、推理、写总线或记录队列异常都会结束实验。初始化 30 秒仍未达到健康站姿会失败；到达 ready 后 60 秒未开始会自动释放。默认 `stop_on_fall=true`；到时或主动停止后先退出策略并保持默认站姿，确认回到站姿后才释放独占。放松和关机可随时抢占策略实验。
