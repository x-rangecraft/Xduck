# Xduck 策略模型导入与回滚接口资料包

适用版本：策略消费接口 API v2  
接口方法：`robot.models`  
目标：上传、校验、切换或回滚一套不可拆分的 `policy.py + model.onnx`

本资料包不包含模型权重。模型与训练任务绑定，研发人员应提供自己的
`.pt`、`.pth` 或 `.onnx` 文件；`examples/` 中提供当前各策略槽位使用的默认消费文件。

## 1. 基本概念

一个运行版本固定由两个文件组成：

```text
policy.py     API 声明、状态初始化、Obs 构造、模型输出解释、滤波和 MIT 参数
model.onnx    robotd 持有并执行的 ONNX 模型
```

上传 `.pt/.pth` 时，robotd 在 RK3566 上校验支持的 PPO checkpoint 并转换为 ONNX；
上传 `.onnx` 时直接进入契约校验。任何候选都必须完成
`Policy.reset → preprocess → ONNX → postprocess` 的完整预热，之后才能由控制循环提交。

robotd 只向消费文件提供平台门禁和平滑后的基础命令，以及
`policy_context={action, phase, body_active}`。模型 Obs 的维度、顺序和命令编码只由
`policy.py` 决定。运行器只接受 API v2，不提供 v1 兼容分支。

## 2. 槽位与默认消费文件

实际可用槽位以 `robot.models {"action":"list"}` 返回值为准。

| 槽位 | 动作 | 默认消费文件 |
|---|---|---|
| `walk`、`stand` | 行走、站立及身体姿态控制 | `examples/default_locomotion_policy.py` |
| `ground_pick` | Ground Pick | `examples/default_ground_pick_policy.py` |
| `kick_left` | 左踢 | `examples/default_kick_left_policy.py` |
| `kick_right` | 右踢 | `examples/default_kick_right_policy.py` |
| `roulade` | 前滚翻 | `examples/default_roulade_policy.py` |
| `sitstand` | 坐下、起立 | `examples/default_sitstand_policy.py` |

Walk/Stand 是共享消费组：共用一份消费文件、一个 `Policy` 实例、一份 feedback 和滤波
历史，同时预加载两个 ONNX。上传或回滚其中一侧会同步两侧消费文件，但不会替换另一侧
ONNX；两个模型联合校验后才原子提交。Sit/Rise 共用 SitStand 消费文件，但每次 Sit、Rise
进入都按新的动作周期 reset。Ground Pick、左右 Kick 和 Roulade 各有独立实例。

默认消费者保留现有模型的 61→14 语义：

- Walk/Stand：基础速度、头部和身体目标；身体姿态模式下速度清零。
- Ground Pick：命令前三维为 `[cos(2πφ), sin(2πφ), 0]`，其余十维清零。
- Kick Left、Kick Right、Roulade、Rise：13 维命令全零。
- Sit：仅第一维 `vx=1`，其余清零。

## 3. 使用前提与限制

- 先执行“放松”，等待所有配置电机返回新鲜、在线、无故障的实际失能反馈。
- 模型文件：`.pt/.pth/.onnx`，1 字节至 64 MiB。
- 自定义消费文件：`.py`，1 字节至 1 MiB。
- 不上传消费文件时，`policy_filename` 和 `policy_size` 必须同时省略，robotd 会按槽位
  写入对应默认消费文件。
- `.pt/.pth` 支持的激活函数：`elu`、`relu`、`tanh`、`selu`、`leaky_relu`。
- `normalizer_epsilon` 范围为 `1e-12`～`1`，缺省为 `0.01`。
- 单块最多 8192 字节；`hex` 是原始字节的小写或大写十六进制文本。
- 同一文件的 `offset` 必须严格等于该文件已经确认接收的字节数。
- 导入、回滚、策略实验和电机实验共用独占控制权，不能并发执行。
- 替换或回滚成功后机器人仍保持放松，必须由操作者重新初始化和使能。

## 4. 接口操作

以下示例展示 `robot.models` 的 `params`。使用网页时这些步骤由网页自动完成；直接使用
JSON-RPC/DataChannel 的客户端应按相同顺序发送。

### 4.1 查询状态和历史

```json
{"action":"list"}
```

主要返回字段：

- `slots[]`：槽位、配置文件、活动版本及最近两个历史版本。
- `phase`：`idle/uploading/converting/validating/pending/done/error`。
- `failed_phase`：发生错误的阶段。
- `detail`：当前进度或拒绝原因。
- `editable`：当前是否已有满足时效的电机失能证明。
- `control_owner`：当前独占操作所有者。
- `token`、`received`：上传凭证和已确认字节数。

### 4.2 开始导入

使用槽位默认消费文件：

```json
{
  "action": "begin_bundle",
  "slot": "ground_pick",
  "model_filename": "ground_pick.onnx",
  "model_size": 123456
}
```

上传自定义消费文件和 checkpoint：

```json
{
  "action": "begin_bundle",
  "slot": "walk",
  "policy_filename": "policy.py",
  "policy_size": 8042,
  "model_filename": "walking.pth",
  "model_size": 3456789,
  "activation": "elu",
  "normalizer_epsilon": 0.01
}
```

成功后保存返回的 `token`。开始新上传会取消尚未完成的旧上传。

### 4.3 分块上传

自定义策略先上传 `policy`，随后上传 `model`；使用默认策略时只上传 `model`。

```json
{
  "action": "bundle_chunk",
  "token": "begin_bundle 返回的 token",
  "file": "model",
  "offset": 0,
  "hex": "001122aabbcc"
}
```

每次调用成功后再发送下一块。网络结果不确定时不要猜测偏移，应重新查询状态或取消后重传。

### 4.4 完成或取消上传

```json
{"action":"finish_bundle","token":"begin_bundle 返回的 token"}
```

`finish_bundle` 只表示所有文件已收到并开始转换/校验。继续轮询 `list`，直到 `phase` 为
`done` 或 `error`。候选处于 `pending` 时，控制循环正在等待再次确认放松、模式和新鲜电机反馈。

只可取消仍处于 `uploading` 的任务：

```json
{"action":"cancel","token":"begin_bundle 返回的 token"}
```

### 4.5 回滚

先通过 `list` 取得目标槽位 `history[]` 中的版本 `id`：

```json
{
  "action": "rollback",
  "slot": "walk",
  "version": "历史版本 id"
}
```

回滚不是直接改清单：历史版本会重新加载、校验和预热，再等待控制循环原子提交。失败时当前
活动版本和运行实例保持不变。Walk/Stand 回滚同样执行共享消费者的双模型联合校验。

## 5. `policy.py` API v2 摘要

消费文件必须定义无参数构造的 `Policy`：

```python
class Policy:
    def describe(self): ...
    def reset(self, robot_info, first_frame): ...
    def preprocess(self, frame, feedback): ...
    def postprocess(self, outputs, frame): ...
```

`describe()` 示例：

```python
return {
    "api_version": 2,
    "period_us": 20_000,
    "required_sources": ["joints", "imu", "command", "policy_context"],
    "inputs": {"obs": {"dtype": "float32", "shape": [1, 61]}},
    "outputs": {"actions": {"dtype": "float32", "shape": [1, 14]}},
    "controlled_joints": [...除 mouth 外的 14 个稳定关节名...],
}
```

输入输出名称必须与 ONNX 完全一致；shape 使用具体正整数。支持 `float32`、`float64`、
`int32`、`int64` 和 `bool`。`period_us` 范围为 20,000～1,000,000 微秒。

`postprocess()` 必须完整返回 14 个关节，每个关节恰好包含有限数值的：

```python
{"position": 0.0, "velocity": 0.0, "torque_ff": 0.0, "kp": 60.0, "kd": 4.0}
```

frame 字段、feedback、reset 生命周期、多输入输出写法和常见错误详见
`POLICY_API_V2.md`。

## 6. 本机验证

建议使用 Python 3.11 或更新版本：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r tools/requirements.txt
.venv/bin/python tools/validate.py \
  examples/default_locomotion_policy.py /path/to/your-model.onnx
```

按目标槽位替换示例消费文件路径。验证器检查 API、ONNX 具名输入输出、reset 和三轮完整
推理，但不会连接机器人或电机。电脑验证通过不等于实机运动安全；首次闭环运行仍应采用
悬空、限力、低增益和急停可达的受控实验流程。

## 7. 平台安全边界

消费进程没有网络、硬件设备或主机写权限，并受执行超时和资源限制。robotd/STM32 继续拒绝
缺失或未知关节、NaN/Inf、越界速度/前馈力矩/Kp/Kd、过期或故障反馈、未完成初始化和未授权
使能。位置还会经过 robotd 执行器范围及 STM32 已同步标定限位。接口结构校验只证明策略可
执行，不证明训练语义、关节顺序、归一化或实机稳定性正确。
