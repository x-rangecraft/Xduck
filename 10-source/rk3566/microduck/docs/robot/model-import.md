# 两文件策略导入与整体回滚

网页「策略模型 · 导入与回滚」接收一套完整策略：

```text
policy.py     Policy 类：接口声明、状态初始化、前处理、后处理
model         `.pt/.pth` PPO checkpoint，或已有 `.onnx`
```

两个文件组成一个不可拆分的版本。前处理决定模型输入和 obs 维度，后处理解释具名模型输出并产生 14 个受控关节的 MIT 目标。robotd 继续唯一拥有电机总线和硬件保护。

完整设计见 [两文件策略运行方案](../../../../../00-docs/POLICY_TWO_FILE_RUNTIME_PROPOSAL.md)。供研发同事直接复制、修改和本机校验的完整本地资料包位于 [`00-docs/XDUCK_POLICY_SDK/`](../../../../../00-docs/XDUCK_POLICY_SDK/README.md)，不进入板端发布包。

## 导入和回滚

1. 点击「放松」，等待页面确认所有配置电机有新鲜、在线、无故障的失能反馈。
2. 选择模型文件（不超过 64 MiB，支持 `.pt/.pth/.onnx`）。策略代码可选：留空时 robotd 按目标槽位选择对应的内置默认消费文件；选择不超过 1 MiB 的自定义 `policy.py` 则只覆盖本次导入，导入请求成功后网页恢复为默认策略。
3. `.pt/.pth` 延续既有导入语义，在 RK3566 上按页面选择的 PPO 激活函数和归一化 epsilon 校验、转换为 ONNX；已有 `.onnx` 不重复转换。转换器会把 checkpoint 的 RSL-RL 观测归一化写入 ONNX，策略代码不得再次归一化。
4. 点击「上传模型 → 转换/校验 → 替换」。无论策略来自内置默认还是本次上传，后端都会为版本保存实际的 `policy.py + model.onnx` 两个文件；上传事务未完成时不会产生版本。
5. 后端在独立 Python 进程中导入 Policy、校验 ONNX 契约，以中性数据 reset 并执行三轮完整的前处理→推理→后处理预热。异常、超时、形状错误或缺失关节都会拒绝候选，当前版本不变。
6. 准备成功后只排队等待控制循环提交。控制循环再次检查相同模式/槽位、Limp、策略未开启及本周期的新鲜 STM32 失能反馈，才整体切换两个文件。替换不使能电机。
7. 历史保留最近两个完整版本。回滚重新加载、校验和预热对应文件；内存状态重新初始化，不恢复旧历史帧或滤波缓存。

### 走/站共享消费文件

当前模式同时配置 Walk、Stand 时，两者组成一个共享消费组：两个 ONNX 预加载在同一隔离进程中，只有一个 `Policy` 实例和一个 `feedback`，每轮只推理选中的模型。默认消费代码的原始 `last_action` 和 `previous_targets` 低通历史因此跨走/站连续，不再保留两份暂停期间变旧的状态；不做双模型输出混合。

- 上传 Walk：更新 Walk ONNX，并让 Walk、Stand 同时引用本次消费文件；Stand ONNX 不变。上传 Stand 时反向处理。
- 不选择消费文件：沿用现有上传约定，两侧都使用本次写入的默认消费代码，而非保留上一次自定义代码。
- 两个 ONNX 都必须满足同一 `describe()`，并交替完成三轮预热。任何一侧失败都拒绝整个候选，不改变当前两侧。
- 两个活动记录在同一次清单原子提交中更新，两侧各保留两条历史；回滚所选侧的历史模型与消费代码，也同步另一侧消费代码，但不回滚另一侧 ONNX。若与另一侧当前模型不兼容，整体拒绝。
- 重启时，相同内容的旧消费文件会合并为同一路径；不同内容不会静默覆盖，而是拒绝加载共享组并报告原因。共享文件或模型只有在所有活动/历史记录都不再引用后才清理。

共享组的全部消费实例状态都会延续，包含代码自行维护的历史或 hidden state。它不等于官方原生 LSTM 的独立 hidden/cell reset；需要这种行为的自定义消费代码须另行约定。相同代码和接口校验能保证处理路径一致，不能证明模型训练时的 obs 单位、归一化、动作顺序/缩放等语义一致，也不能保证运动稳定，作者仍需确认这些约定。其他技能继续使用各自隔离的消费实例。

启动时会把旧清单中的当前 ONNX 和历史 ONNX 复制为统一的 `<id>-model.onnx`，写入配套 `<id>-policy.py`，再原子更新清单；提交成功后删除不再引用的旧 `<id>.onnx`。已由 robotd 标记为“默认”的平台消费文件会按槽位原子升级为当前 API v2 默认代码；用户上传的旧接口文件不会被改写，仍会明确拒绝。旧单文件 RPC 和旧推理分支已删除，所有版本只通过双文件运行器加载、运行和回滚。

### 内置“默认”策略

内置默认消费文件按槽位拆分：Walk/Stand 共享 locomotion 文件，Ground Pick、左右 Kick、Roulade、SitStand 分别使用自己的文件。每个文件独立组装既有 61 维观测并拥有自己的命令编码；研发包的 `policy.py` 与 locomotion 文件保持一致。它们保存上一帧原始 14 维 ONNX action，按 `HOME + 0.9 × action` 生成目标，腿部/头部低通系数为 0.7/0.5，并为全部 14 个受控关节返回 `Kp=60`、`Kd=4`、零速度和零前馈力矩。默认消费者不执行观测归一化；由 robotd 从 checkpoint 生成的 ONNX 已包含该变换。

默认命令 Obs 语义只存在于各自消费文件：Walk/Stand 使用基础速度、头部与身体命令（身体姿态模式清零速度）；Ground Pick 使用 `[cos(2πφ), sin(2πφ), 0]` 并清零其余十维；左右 Kick、Roulade、Rise 使用全零 13 维命令；Sit 仅令第一维 `vx=1`。

## `policy.py` API v2

运行器只接受 v2，不提供 v1 兼容分支。v1 文件会在候选预热或启动恢复时明确失败，不会进入电机使能流程。

文件必须定义无参数构造的 `Policy` 类：

```python
class Policy:
    def describe(self): ...
    def reset(self, robot_info, first_frame): ...
    def preprocess(self, frame, feedback): ...
    def postprocess(self, outputs, frame): ...
```

`describe()` 必须返回可 JSON 序列化且稳定不变的字典：

```python
{
    "api_version": 2,
    "period_us": 20_000,
    "required_sources": ["joints", "imu", "command", "policy_context"],
    "inputs": {"obs": {"dtype": "float32", "shape": [1, 96]}},
    "outputs": {"actions": {"dtype": "float32", "shape": [1, 28]}},
    "controlled_joints": [...除 mouth 外的全部 14 个稳定关节名...],
}
```

平台控制周期为 20 ms。`period_us` 第一版允许 20,000 至 1,000,000 微秒；较慢策略在周期之间保持最近一个经过平台校验的目标。小于 20 ms 的周期无法执行，会在导入时拒绝。

输入和输出名称必须与 ONNX 完全一致。声明 shape 必须用具体正整数解析模型中的动态维度；运行中 shape 固定。平台不拍平、补零、裁剪或猜测语义。API v2 支持 float32、float64、int32、int64 和 bool 张量；不支持 ONNX sequence/map。

`reset(robot_info, first_frame)` 初始化实例中的历史观测、滤波和循环模型状态。所有可变状态必须放在实例中，不得使用模块全局或类变量跨实例保留状态。

正常策略调度采用 reset-on-enter：除 Walk、Stand 外，其余策略每次进入时都在实际首帧推理前 reset；同一动作持续运行不重复 reset。坐下→起身（共用 SitStand）、再次触发同一动作和连翻的每一圈也视为新一轮执行。被高优先级动作打断后重新进入的技能同样 reset。Walk、Stand 在普通切换中保留同一个共享消费实例；从其他技能返回时继续该共享组自身状态，不导入不兼容技能的历史。初始化、重新使能、替换策略和显式控制器 reset 仍会重置全部策略。进入首帧失败不会消费此次进入标记，重试仍先 reset。

`preprocess(frame, feedback)` 返回按 ONNX 输入名称组织的 NumPy 数组字典。也可以返回 `{"ready": False, "reason": "..."}` 表示本轮未就绪；正式运行将按策略错误保持，第一版不会无限等待后自动恢复驱动。

frame 当前提供：帧序号、STM32 tick、实际 dt、稳定关节顺序、关节位置/速度、测得电机力矩、状态位、温度、反馈年龄、融合姿态、IMU gyro/gravity/quaternion、未经技能编码的基础 command，以及 `policy_context={action, phase, body_active}`。基础 command 已经过平台限幅、deadman 和平滑，但不会被外层改成 Ground Pick 相位、全零技能命令或 Sit/Rise 标志。`policy_context` 是调度状态而不是模型 Obs；消费文件自行决定怎样使用。没有硬件来源的测量不会伪装成原始数据。

feedback 首轮为 `None`，以后包含上一轮具名 ONNX 输出和后处理请求目标。电机实际响应始终从下一帧硬件反馈读取。

`postprocess(outputs, frame)` 必须返回以关节名为键的字典或 `{"targets": ...}`，完整覆盖除 mouth 外的 14 个关节。每个目标必须恰好包含：

```python
{
    "position": 0.0,  # rad
    "velocity": 0.0,  # rad/s
    "torque_ff": 0.0, # N·m
    "kp": 60.0,
    "kd": 4.0,
}
```

obs、归一化、动作缩放、算法裁剪、滤波、单步限制和逐关节 Kp/Kd 均由 Policy 实现。网页不再提供会与后处理重复作用的统一 Kp/Kd/缩放输入。

## 运行隔离和硬件保护

robotd 使用固定的 `/var/lib/robotd/model-python/bin/python`（或显式 `ROBOT_POLICY_PYTHON`）启动独立、持久的策略进程。运行器使用 NumPy 和 CPU ONNX Runtime。用户代码没有电机总线对象；robotd 每轮只传数据并接收目标。

策略进程由 Bubblewrap 创建独立 PID、IPC、UTS 和网络命名空间，主机根文件系统（含 ONNX Runtime 读取 CPU 拓扑所需的 `/sys`）只读，`/tmp`、`/run` 和 `/dev` 使用私有最小挂载。`setpriv` 在 Python 启动前切换到无硬件组的 `robot-policy` 用户、清空补充组、移除全部 capability 并禁止重新提权；robotd 同时限制地址空间、进程数、打开文件数、可写文件大小和 core dump。目标机缺少 `bwrap`/`setpriv` 或隔离创建失败时拒绝加载策略，不降级为直接执行。

每次策略计算限制为 18 ms。普通策略异常使该帧失败并保持目标，显式放松、初始化后可重新创建 Policy 状态；超时、进程退出或协议损坏会终止工作进程，需回滚、重新导入或重启 robotd 才能重新加载。进程隔离让 Python 阻塞不会无限阻塞 robotd，但仍需在板端验证实际性能。

后处理请求仍经过以下平台门禁：

- 全部字段有限且关节集合完整；
- 位置经过 robotd 执行器范围限制，并继续受已同步到 STM32 的标定限位保护；
- 速度、前馈力矩、Kp、Kd 满足对应电机和 DMUSB v4 范围；
- 版本、控制模式、使能、反馈新鲜度、通信故障及 STM32 独立 100 ms 主机命令超时保护保持有效；
- 初始化、放松、跌倒处理和其他非策略流程不会继承策略 MIT 参数。

## 存储与故障恢复

策略版本存放在 `/var/lib/robotd/policies/`，发布包中的原始 ONNX 不修改。清单对每个槽位记录必需的 model 路径、policy 路径和接口契约；不存在只含 ONNX 的活动版本。

候选文件先完整落盘并 fsync，再生成清单；清单使用临时文件、fsync 和 rename 提交。内存切换只由总线控制循环执行。失败前旧清单和旧实例保持不变；重启按活动引用重新创建 Policy 和 ONNX Session，保持失能并等待显式初始化/使能。

清单损坏、文件缺失、运行环境不兼容或历史版本无法重新验证时拒绝加载，不静默选择另一个策略。

## 依赖与验证

已授权部署通过 `scripts/setup-model-import.sh` 管理 Python 环境。固定依赖已包含 Python 运行所需的 NumPy 和 ONNX Runtime；不在策略导入期间联网安装任意包。

回归范围包括：两文件上传角色与分块、动态维度与多输入输出契约、reset 后状态隔离、完整 MIT 字段、非法目标拒绝、执行超时、清单原子性、重启/回滚和网页失能门禁。结构和试运行检查通过不等于策略能够稳定运动，实机验证仍须按受控实验流程进行。
