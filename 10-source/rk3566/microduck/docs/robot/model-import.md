# 两文件策略导入与整体回滚

网页「策略模型 · 导入与回滚」接收一套完整策略：

```text
policy.py     Policy 类：接口声明、状态初始化、前处理、后处理
model.onnx    ONNX 网络
```

两个文件组成一个不可拆分的版本。前处理决定模型输入和 obs 维度，后处理解释具名模型输出并产生 14 个受控关节的 MIT 目标。robotd 继续唯一拥有电机总线和硬件保护。

完整设计见 [两文件策略运行方案](../../../../../00-docs/POLICY_TWO_FILE_RUNTIME_PROPOSAL.md)。可运行接口示例见 [`policies/two_file_policy_example.py`](../../policies/two_file_policy_example.py)。

## 导入和回滚

1. 点击「放松」，等待页面确认所有配置电机有新鲜、在线、无故障的失能反馈。
2. 选择 `policy.py`（不超过 1 MiB）和单文件自包含的 `model.onnx`（不超过 64 MiB）。
3. 点击「上传两文件 → 校验预热 → 替换」。上传事务未收齐两个文件时不会产生版本。
4. 后端在独立 Python 进程中导入 Policy、校验 ONNX 契约，以中性数据 reset 并执行三轮完整的前处理→推理→后处理预热。异常、超时、形状错误或缺失关节都会拒绝候选，当前版本不变。
5. 准备成功后只排队等待控制循环提交。控制循环再次检查相同模式/槽位、Limp、策略未开启及本周期的新鲜 STM32 失能反馈，才整体切换两个文件。替换不使能电机。
6. 历史保留最近两个完整版本。回滚重新加载、校验和预热对应文件；内存状态重新初始化，不恢复旧历史帧或滤波缓存。

旧清单和旧 ONNX 历史仍可读取、运行和回滚。旧 `.pt/.pth` RPC 保留用于兼容已有客户端，但网页新入口只创建两文件版本。

## `policy.py` API v1

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
    "api_version": 1,
    "period_us": 20_000,
    "required_sources": ["joints", "imu", "command"],
    "inputs": {"obs": {"dtype": "float32", "shape": [1, 96]}},
    "outputs": {"actions": {"dtype": "float32", "shape": [1, 28]}},
    "controlled_joints": [...除 mouth 外的全部 14 个稳定关节名...],
}
```

平台控制周期为 20 ms。`period_us` 第一版允许 20,000 至 1,000,000 微秒；较慢策略在周期之间保持最近一个经过平台校验的目标。小于 20 ms 的周期无法执行，会在导入时拒绝。

输入和输出名称必须与 ONNX 完全一致。声明 shape 必须用具体正整数解析模型中的动态维度；运行中 shape 固定。平台不拍平、补零、裁剪或猜测语义。API v1 支持 float32、float64、int32、int64 和 bool 张量；不支持 ONNX sequence/map。

`reset(robot_info, first_frame)` 初始化实例中的历史观测、滤波和循环模型状态。所有可变状态必须放在实例中，不得使用模块全局或类变量跨实例保留状态。

`preprocess(frame, feedback)` 返回按 ONNX 输入名称组织的 NumPy 数组字典。也可以返回 `{"ready": False, "reason": "..."}` 表示本轮未就绪；正式运行将按策略错误保持，第一版不会无限等待后自动恢复驱动。

frame 当前提供：帧序号、STM32 tick、实际 dt、稳定关节顺序、关节位置/速度、测得电机力矩、状态位、温度、反馈年龄、融合姿态、IMU gyro/gravity/quaternion 和当前有效 command。没有硬件来源的测量不会伪装成原始数据。

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

策略版本存放在 `/var/lib/robotd/policies/`，发布包中的原始 ONNX 不修改。清单对每个槽位记录 model 路径、可选 policy 路径、契约和兼容旧版控制参数。

候选文件先完整落盘并 fsync，再生成清单；清单使用临时文件、fsync 和 rename 提交。内存切换只由总线控制循环执行。失败前旧清单和旧实例保持不变；重启按活动引用重新创建 Policy 和 ONNX Session，保持失能并等待显式初始化/使能。

清单损坏、文件缺失、运行环境不兼容或历史版本无法重新验证时拒绝加载，不静默选择另一个策略。

## 依赖与验证

已授权部署通过 `scripts/setup-model-import.sh` 管理 Python 环境。固定依赖已包含 Python 运行所需的 NumPy 和 ONNX Runtime；不在策略导入期间联网安装任意包。

回归范围包括：两文件上传角色与分块、动态维度与多输入输出契约、reset 后状态隔离、完整 MIT 字段、非法目标拒绝、执行超时、清单原子性、重启/回滚和网页失能门禁。结构和试运行检查通过不等于策略能够稳定运动，实机验证仍须按受控实验流程进行。
