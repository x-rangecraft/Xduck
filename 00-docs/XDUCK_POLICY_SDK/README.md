# Xduck 两文件策略研发包

这个目录是一套只保存在研发电脑、可以直接复制给同事的最小完整示例；它不随机器人发布包部署。网页必须选择模型文件，策略代码可以留空：

```text
policy.py    可选；前处理、状态、后处理和逐关节 MIT 参数
model        `.pt/.pth` PPO checkpoint，或已有 `.onnx`
```

其余文件用于本机验证：

```text
validate.py          一条命令运行接口校验、reset 和三轮完整推理
requirements.txt     验证器的固定 Python 依赖
_policy_worker.py    与 robotd 编译时嵌入的同一份执行器，请勿修改
```

## 1. 五分钟开始

要求 Python 3.11 或更新版本。进入本目录后执行：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python validate.py
```

成功时会打印 `PASS`、策略周期、ONNX 输入输出和 14 个受控关节。这个命令不会连接机器人或电机。

开始自己的策略时，复制整个目录，替换 `model.onnx`，然后修改 `policy.py`。每次修改后运行：

```sh
.venv/bin/python validate.py policy.py model.onnx
```

验证通过后，在 Xduck 网页的「策略模型 · 导入与回滚」中选择模型和这份 `policy.py`。如果不选择策略代码，robotd 会根据目标槽位写入对应的内置默认消费文件；本目录示例与 Walk/Stand 默认消费文件一致。Ground Pick、左右 Kick、Roulade 和 SitStand 各有独立默认消费文件，彼此不共享命令 Obs 语义。自定义选择在导入请求成功后自动清空，下一次重新使用对应槽位的默认策略。`.pt/.pth` 在 RK3566 上转换，已有 `.onnx` 直接校验。所有模型都以双文件版本运行，不再存在单文件导入或推理分支。机器人必须先放松，并由新鲜电机反馈确认所有电机实际失能。替换成功后仍保持失能，需要操作者重新初始化和使能。

运行接口只支持 `api_version=2`，不兼容 v1。v1 消费文件会在加载和电机使能之前被明确拒绝。

默认策略不做观测归一化。robotd 从带 RSL-RL normalizer 的 checkpoint 转换时，均值和标准差变换会嵌入生成的 ONNX；自行提供的 ONNX 也应包含训练所需的归一化，或者改用自定义策略显式实现。

## 2. Policy 类

`policy.py` 必须定义可无参数构造的 `Policy`：

```python
class Policy:
    def describe(self): ...
    def reset(self, robot_info, first_frame): ...
    def preprocess(self, frame, feedback): ...
    def postprocess(self, outputs, frame): ...
```

实例拥有全部运行状态。历史帧、循环网络 hidden state、上一动作和滤波状态应写入 `self`，并在 `reset()` 中完整初始化。不要使用模块全局变量或可变类属性保存跨实例状态。

### describe()

声明策略与 ONNX 的确定接口：

```python
return {
    "api_version": 2,
    "period_us": 20_000,
    "required_sources": ["joints", "imu", "command", "policy_context"],
    "inputs": {
        "obs": {"dtype": "float32", "shape": [1, 61]},
    },
    "outputs": {
        "actions": {"dtype": "float32", "shape": [1, 14]},
    },
    "controlled_joints": [...],
}
```

- `period_us` 允许 20,000～1,000,000 微秒。平台总线仍以 20 ms 周期运行；慢策略的两个推理周期之间保持最近一次合法目标。
- 输入输出名称必须与 ONNX 完全一致。
- shape 必须写具体正整数。它可为模型中的动态维度选择实际运行值，因此 obs 不限于 61 维。
- 支持 `float32`、`float64`、`int32`、`int64`、`bool` 张量。
- 第一版必须完整控制除 `mouth` 外的 14 个关节。

### reset(robot_info, first_frame)

每次导入预热结束、版本切换、模式切换或控制器复位时都会建立新的 `Policy` 实例并调用 `reset()`。`robot_info` 提供稳定关节名；`first_frame` 与正式帧结构相同。这里应初始化状态，不要重新加载模型或安装依赖。

普通调度中，Walk、Stand 共用一个 `Policy` 实例和一个 `feedback`，两个 ONNX 常驻、每轮只选一个推理；切换不 reset，`self.last_action`、滤波和其他实例状态自然连续。上传或回滚任一侧会同步另一侧的消费文件（包含未选文件时的默认代码），但另一侧 ONNX 不变；两个模型须满足同一契约并交替预热通过后才原子更新。两侧各保留两条历史。不同代码的旧走/站组合不会被静默合并。相同代码不能验证训练时的单位、归一化与动作语义，作者仍需确认一致；自定义循环网络的 hidden state 也属于共享实例状态，不会由平台按模型切换自动清零。

其他策略在每次进入的实际首帧前 reset，同一动作连续运行不重复 reset。坐下→起身、再次触发同一动作、连翻的下一圈和被打断后的重新进入都算新一轮。从其他技能返回走/站时不继承该技能状态。显式控制器复位仍重置全部策略。

### preprocess(frame, feedback)

返回以 ONNX 输入名为键的 NumPy 数组：

```python
return {
    "obs": obs.astype(np.float32).reshape(1, 61),
}
```

多输入和不同 shape 直接增加键即可。也可返回 `{"ready": False, "reason": "等待历史帧"}`，但未就绪帧不会驱动电机。

`feedback` 首轮为 `None`；以后包含上一轮具名 ONNX 输出和上一轮后处理请求的 MIT 目标。实际电机响应必须读取新一帧 `frame`，不能把请求目标当成测量结果。

### postprocess(outputs, frame)

`outputs` 是按 ONNX 输出名称组织的 NumPy 数组。返回 14 个关节的完整 MIT 目标：

```python
return {
    "left_hip_yaw": {
        "position": 0.0,   # rad
        "velocity": 0.0,   # rad/s
        "torque_ff": 0.0,  # N·m
        "kp": 60.0,
        "kd": 4.0,
    },
    # ...其余 13 个受控关节
}
```

每个关节必须恰好包含这五个有限数值。obs 组装、归一化、动作缩放、算法裁剪、低通、步长限制和逐关节增益都在这里或 `preprocess()` 中实现。

## 3. frame 原始数据

每帧提供以下稳定字段：

| 字段 | 单位/含义 |
|---|---|
| `sequence` | 当前策略输入序号 |
| `gateway_tick_ms` | STM32 网关 tick，毫秒 |
| `dt` | 本次平台循环间隔，秒 |
| `joint_names` | 15 个稳定关节名及顺序 |
| `positions` | 关节位置，rad |
| `velocities` | 关节速度，rad/s |
| `motor_torques_nm` | 电机测得力矩，N·m |
| `motor_flags` | STM32 电机状态位 |
| `motor_temperatures_c` | 电机温度，°C |
| `motor_feedback_age_ms` | 各关节反馈年龄，ms |
| `attitude_rpy` | 融合 roll/pitch/yaw，rad |
| `imu.gyro` | 角速度，rad/s |
| `imu.gravity` | 机体坐标重力方向 |
| `imu.quat` | 姿态四元数 `[w,x,y,z]` |
| `command.twist` | 平台限幅、deadman、平滑后的基础 `[vx, vy, yaw_rate]`；未按技能改写 |
| `command.head` | 未按技能改写的 4 维头部意图 |
| `command.body` | 未按技能改写的 `[z, roll, pitch]` |
| `policy_context.action` | `walk/stand/ground_pick/kick_left/kick_right/roulade/sit/rise` |
| `policy_context.phase` | Ground Pick 的归一化周期相位；其他动作固定为 `None` |
| `policy_context.body_active` | 是否处于身体姿态控制模式 |

完整数值结构可直接查看 `validate.py` 的 `neutral_frame()`。

`policy_context` 是调度状态，不是 ONNX 输入，也不是预编码的 command Obs。消费文件自行选择是否读取这些字段，并自行决定怎样构造模型张量。默认消费者保持旧模型的编码：Walk/Stand 使用基础速度、头部和身体目标；Ground Pick 使用 `[cos(2πφ), sin(2πφ), 0]` 并清零其余命令；Kick/Roulade/Rise 全零；Sit 仅 `vx=1`。

## 4. 平台仍会拒绝的目标

算法可自由定义，但 robotd 和 STM32 仍负责硬件保护：

- 缺关节、未知关节、重复字段或 NaN/Inf；
- 超出对应电机协议范围的速度和前馈力矩；
- `kp` 不在 0～500 或 `kd` 不在 0～5；
- 位置会先经过 robotd 的执行器范围限制，STM32 继续执行已同步的标定硬限位；
- 电机反馈过期、离线、故障、未完成初始化或未授权使能；
- 单轮完整 Python + ONNX 计算超过 18 ms。

Python 在独立受限进程运行，没有网络、硬件设备或主机写权限。异常和超时不会阻塞 50 Hz 总线安全循环。结构校验通过只说明接口可执行，不说明策略在实机上稳定；新策略仍应从悬空、限力、低增益的受控实验开始。

## 5. 示例对应关系

目录中的示例重现现有 `alpha_walking.onnx` 的 61 维观测和 14 维动作路径：

- `policy.py` 组装 IMU、关节位置/速度、上一动作和命令；
- `model.onnx` 输出 `actions[1,14]`；
- `postprocess()` 应用默认姿态、动作缩放、分组低通和 MIT 增益。

要使用 96 维 obs、多输入、循环 hidden state 或不同输出含义，只需同步修改 `describe()`、`preprocess()` 和 `postprocess()`，然后导出名称与声明一致的 ONNX。平台代码无需重新编译。

## 6. 常见失败

| 报错 | 检查内容 |
|---|---|
| 输入/输出名称不一致 | Netron/导出代码中的名称与 `describe()` 是否逐字相同 |
| shape 不一致 | batch、历史帧和动态维度是否解析为声明的具体 shape |
| 缺少关节 | 是否完整返回除 `mouth` 外的 14 个稳定名称 |
| NaN/Inf | 归一化分母、未初始化历史和模型输出 |
| 超过 18 ms | 减少 Python 分配、模型规模或策略频率；在 RK3566 上重新测量 |
| 导入后不能使能 | 先放松，确认总线和全部电机的新鲜失能反馈，再重新初始化 |

更完整的平台设计和故障语义见同级的 `../POLICY_TWO_FILE_RUNTIME_PROPOSAL.md` 与仓库中的 `10-source/rk3566/microduck/docs/robot/model-import.md`。
