# GF43X40-10 执行器与通信组件

实验性 `physics.friction_model="smooth_stribeck"` 可用于21项候选，构造器在冷路径
选择同名摩擦模式，裸轴和整机使用同一公式；不需要单轴质量、独立黏着阈值、
方向记忆或滑行过渡状态。只在显式传入候选校准包时启用，当前默认校准包保持不变。
该候选已出现裸轴低速和停稳退化，不能据软件一致性测试宣称精度不变。

**现行默认：`shared_fixed_gain_kd_band_v8_torque_proxy`。** 当前
`observation.model="mechanical_torque_proxy"`，反馈读数直接为
`0.1198991043896492 * last_torque`，只有一个数值参数 `torque_feedback_gain`。
它不再使用力矩反馈滤波、加速度/摩擦/保持/负载反馈项；4个速度反馈参数不变。
该读数仅供粗略参考，不代表高精度CAN电流反馈或真实轴端力矩。
机械/观测数值参数总数42→32。旧 `kinematic_current_proxy` 仍支持历史快照，
不在当前路径中执行。以下7→1负载反馈与精确等价清理均为历史阶段说明。

**当前默认：`shared_fixed_gain_kd_band_v8_linear_load`。** 负载反馈已由7系数模型
简化为 `0.8834199263375071 * load_proxy`，保留原反馈低通滤波。`load_proxy`
仍来自仿真机械力矩减去等效惯量乘加速度，零增益零前馈滑行时为0。
`observation.load_feedback_model="linear"` 跳过旧交互特征计算；不接受同时配置
非零旧负载交互系数。无此标记的旧快照仍按历史多项模型运行。
位置、速度、机械力矩保持原v8，反馈力矩精度按用户要求放宽；
`behavior_reference_scope="mechanics_and_velocity_only"` 明确原行为基准不再约束力矩反馈。
机械/观测数值配置在此前53→48整理之后进一步48→42，实际减少6个有效负载反馈系数。
下面的精确等价精简及7系数模型描述属于之前的历史阶段。

当前 v8 已做精确等价参数精简：删除当前观测分支不生效的 `feedback_gain`
和零值 `memory_gain`；9 个未参与运行计算的硬件寄存器移入 `hardware_reference`。
源配置将 `envelope_scale = torque_scale`、`braking_limit = low_speed_drive_limit`、
`mechanical_torque_limit = torque_scale * tmax` 作为派生关系；导出包在初始化前
物化这 3 个值以兼容现有后端，它们不是独立调参量。机械与观测数值配置从
53 项降至 48 项（包含显式指令延迟）。未重新拟合或近似任何有效系数。
`source_sha256` 跟踪精简后的源文件，`behavior_reference_sha256` 指向未改动的
原始 v8 轨迹基准；测试仍使用冻结轨迹，不因精简重新生成预期输出。

> 当前默认版本：`shared_fixed_gain_kd_band_v8`。按用户固定增益下的14电机1千赫兹阶跃，新增平滑的速度增益范围修正（`physics.kd_gain_band_slope=0.04`）；它只在Kp约75–110、Kd约3时生效，以缩短严格末端停稳时间差。v7参数已冻结供回退；54系数反馈力矩残差层继续关闭。下文v6/v7实验数字属于历史版本。

本目录提供 `GF43X40Actuator`、`GF43X40Communication` 和整机适配器 `GF43X40RobotMotor`。独立任务 `XDuckGF43X40VelocityFlat` 已接入这套组件；原 `MicroDuckDm4310VelocityFlat` 仍使用 DM4340，旧训练和 checkpoint 不变。`calibration.json` 是裸电机共性参数的快照，记录源 SHA-256，并保存 29 行说明书速度—力矩原始数据。

历史v6快照为schema 2；当前v8校准包仍为schema 2。它通过186设备的12关节、多增益、宽位置/速度数据辨识一个共同机械尺度，并新增7个负载/增益/来向反馈系数。执行器实际输出给后端的机械力矩与CAN等效反馈分开计算；CAN量程仍为±28，机械限幅读取`physics.mechanical_torque_limit`，不能再用反馈编码范围截断机械力矩。机械包络与惯量是等效仿真参数，不能当成独立测得的真实峰值能力或转子惯量。

负载代理量为`last_torque - armature * acceleration_after`，只用仿真自身状态。历史v6保留原v5的54个因果残差系数以及裸轴动力学比例；当前v8已移除该残差层，102条裸轴位置/速度/反馈最大变化约1.6e-9。Rider悬吊实验的独立增益检查中，主动关节反馈力矩RMSE为0.3452→0.2409 Nm，仍有保持偏置与高频残差。完整数据和限制见工作区`data/gf43x40_10_alignment/experiments/rider_campaign_20260917/RESULTS.md`。

单自由度、Python/C、带载便携组件及任务接入检查已完成；悬吊耦合重放以实测重力方向作为实验边界，**不代表落地策略或真实轴端力矩已验收**。v5回退包保存在上述报告目录。以下百分比与旧增益记录属于此前版本历史。

v5曾增加2/8/30 ms的因果力矩残差状态，使用54个共享正则化系数。`torque_observer.py`继续支持v1（三时间尺度）和实验性v2（另加150/500 ms），默认残差bank仍采用v1。偏置个体标定未启用。

组件保留 `shared_multichannel_v4` 的内部速度状态、增益连续修正、驱动释放动态、方向摩擦记忆、滑行过渡，以及独立速度反馈状态和驱动状态力矩观测项。每个物理子步调用 `observe` 后，以 `motor.velocity_feedback` 传给通信组件；该属性已经包含速度滤波和比例，通信层再应用观测偏置与 CAN 编码。不要用反馈速度覆写物理后端状态。

102 条历史记录的三通道开发回归已完成；激进拟合候选因局部退化未启用。当前版本位置、速度、CAN 力矩的各记录平均 RMSE 分别下降约 1.4%、16.9%、11.3%；这些数据参与过开发选择，不是新盲测。10 项软件测试覆盖冻结单轴轨迹、批量执行和部分环境通信重置。后文的 `effort_slew_limit=1000` 是前一版历史值，当前精确取值以 calibration.json 为准。

`GF43X40Actuator` 对 `(num_envs, num_motors)` 批量输入计算力矩。每 1 ms 以当前位置、速度和 `q_des, qd_des, kp, kd, tau_ff` 更新 MIT 控制、驱动/制动响应、摩擦与能力包络。物理后端积分后，再调用 `observe(qd_after, acceleration_after)` 生成**电流估算力矩反馈**。它不把反馈误差施加为真实机械力矩，也不声称轴端力矩已完成绝对标定。所有电机共用参数，各关节的 Kp/Kd 可不同。

2026-09-17 的独立 0.2 rad 高增益真机阶跃补充了共同的名义驱动力矩变化速率上限 `effort_slew_limit=1000`。它改善瞬态曲线，但没有把 Kp=80/150 的严格到达时间全部对齐；UniLab 包内的标定快照与单轴对照轨迹已同步。数值单位是模型内部名义力矩/秒，不是经独立仪器测得的真实轴端力矩变化率。

默认 `friction_mode="isolated_1d"` 是已对照的裸轴单自由度算法，调用时必须提供该轴的实际广义质量（含 armature）。整机耦合关节不能把单轴质量直接带入并声称结果一致。XDuck 任务明确选用 `smooth_coupled_approx`，它**尚未通过整机验证**。GF armature 在模型冷路径一次设置；原 XML 的 joint damping/frictionloss 同时清零，避免重复计入。

组件已经计算全部摩擦力矩。XDuck owner 在模型冷路径清零原有 joint `damping` / `frictionloss`，避免重复计算。反馈所需的物理子步后关节加速度由相邻 1 ms 关节速度差分得到；这是整机适配假设，尚未与真实负载下的加速度反馈逐点验证。env 只使用公开的 backend 关节状态接口。

`GF43X40Communication` 接收主机目标，按传入时间锁存；`advance` 要求每 1 ms 调用，`sample` 从电机后验状态按 20 ms 周期发布位置、速度、电流估算力矩。主机目标通常每 20 ms 提交一次，也可按实测时间提交以重放抖动。CAN 映射位置 16 位，其余速度、力矩与增益 12 位。序号标识已锁存指令，**不证明某条反馈由该指令产生**。额外命令/反馈延迟默认 0 的意思是尚未叠加不可辨识的单向固定延迟；这不代表真机网络零延迟。组件不模拟没有证据的随机丢包。

### 1 ms 分辨率的通信抖动

延迟抽样、待到达队列、指令保持、反馈发布和旧帧保护均在 `communication.py` 内。
`robot.py` 负责连接电机与通信，训练 YAML 仅提供参数；单独使用通信组件不需要训练配置。

XDuck 训练默认启用 `env.communication.jitter_mode=measured_20260918`。
采样实现属于 `communication.py`，分布与来源保存在同目录 `timing_profile.json`，
初始化时读取一次。1 ms 电机周期不加抖动，20 ms 是主控和反馈的标称周期。

| 额外晚到 | 命令权重（共 3,141） | 反馈权重（共 3,249） |
|---|---:|---:|
| 0 ms | 3,126（99.5224%） | 3,247（99.9384%） |
| 1 ms | 7（0.2229%） | 2（0.0616%） |
| 2 ms | 5（0.1592%） | 0 |
| 3 ms | 1（0.0318%） | 0 |
| 4 ms | 1（0.0318%） | 0 |
| 5 ms | 1（0.0318%） | 0 |

命令权重来自 `20260918T110255Z-110621Z.tar.gz` 的 3,141 个接收间隔：
间隔减去 20 ms 后取最近整数毫秒，只将正偏差计为一次晚到，非正偏差归为无额外延迟。
这是根据观测间隔推定的稀疏晚到近似，并非已辨识的单向延迟分布。
一次晚到 5 ms、下一帧正常会自然产生 25 ms/15 ms 的相邻间隔，
因此不再把补偿性的短间隔单独抽样为另一份延迟。
第一份原始包后续不再位于桌面，权重使用此前已经核对、保存的统计。

反馈权重来自 `rk-roundtrip-20260918T112252Z.tar.gz` 的 3,249 条 ACK 往返：
取整后的 19/20/21 ms 分别为 7/3,240/2 条；仅将超出既有 20 ms 周期的 2 条映射成
1 ms 额外晚到，其余为 0。这个近似不提前发布状态，也不声称还原 7 条较早的 ACK。
约 20 ms 往返不能再作为额外延迟叠加；ACK 也不保证状态是该条电机指令的因果响应。
两批工况并非同步测量，模型独立采样两个方向，不声称重现所有时间相关性。

配置示例：

```yaml
communication:
  jitter_mode: measured_20260918
  command_jitter_ms: [0, 0]
  feedback_jitter_ms: [0, 0]
  feedback_delay_ms: 0.0
  jitter_seed: 0
```

实测模式中两个均匀范围必须保持 `[0, 0]`，不会与离散分布叠加。
要关闭随机延迟，设置 `jitter_mode: uniform` 并保留两个零范围。
均匀模式仍可使用非负整数毫秒范围做额外实验。独立构造通信/整机组件时
`jitter_mode` 默认仍为 `uniform`，以保留旧接口行为；启用实测分布需显式传入模式。
固定命令延迟仍取 `env.latency.actuator_delay_ms`，固定反馈延迟仍取 `feedback_delay_ms`。

主控仍每 20 ms 生成目标，物理/电机模型仍每 1 ms 更新。抖动改变帧的可用时刻，
不改变积分步长，也不冻结电机力矩。指令未到时保持旧目标，反馈未到时保持旧反馈；
首次反馈前保持初始观测。新帧到达后，迟到的旧指令或旧反馈不会覆盖较新的数据。
USB/STM32/CAN 的影响在这里作为端到端额外延迟近似，没有分别模拟两段链路的调度，
也没有模拟主控策略计算周期本身的抖动。

一个批量帧中的全部电机和环境共享一次延迟抽样；目前不是逐环境独立的延迟随机化。
命令与反馈使用分开的随机数流，相同 seed 和调用顺序可复现；完整通信 reset 重播随机数流，
部分环境 reset 只清除相应数据，不重置时钟或其他环境的随机数流。
父级 `latency.joint_velocity_delay_ms` 仍是额外的策略观测历史延迟，
与这里的反馈通信延迟叠加，后续校准时应区分两者，避免重复计入同一段实测延迟。

### 单独分享通信组件

把以下四个文件放进同一目录即可分享：

- `communication.py`：通信模拟组件，直接导入 `GF43X40Communication`、`MITCommand`。
- `calibration.json`：默认 GF43X40-10 量程和反馈编码参数，组件从自身所在目录读取。
- `timing_profile.json`：实测晚到幅度、计数权重、来源和近似方法。
- `example_communication.py`：可直接运行的示例，已启用实测稀疏抖动。

接收方只需要 Python 和 NumPy，无需安装 UniLab、Hydra 或 MuJoCo。
在该目录运行 `uv run --with numpy example_communication.py`。
在构造函数中设置 `jitter_mode="measured_20260918"`，即可独立开启实测稀疏抖动；
固定延迟的 `command_delay_s`、`feedback_delay_s` 单位为秒。
示例使用静止状态展示通信时序，不代表电机动力学仿真。
调用者每 20 ms 提交新目标、每 1 ms 调用 `advance` 和 `sample`；
`command_period_s` 是标称信息，组件不会自动生成主控指令。

裸轴调用顺序：

```python
params = GF43X40Parameters.from_bundle()  # 环境初始化时读取一次
motor = GF43X40Actuator(num_envs, 1, dt=0.001, parameters=params)
link = GF43X40Communication(num_envs, 1)

link.submit(MITCommand(q_des, qd_des, kp, kd, tau_ff), at_s=t)  # 主机目标
command = link.advance(t)                                      # 每 1 ms
torque = motor.compute_torque(q, qd, command.q_des, command.qd_des,
    command.kp, command.kd, command.tau_ff, effective_mass=isolated_axis_mass)
# 后端用 unit-gain torque actuator 积分 torque，然后读后验状态：
estimate = motor.observe(qd_after, acceleration_after)
feedback = link.sample(t + 0.001, q_after, motor.velocity_feedback, estimate)
```

旧 MicroDuck DM4340 任务仍是 `sim_dt=0.005`；新 XDuck GF 任务使用明确的 `sim_dt=0.001`、20 ms 主机目标/反馈和 1 ms 力矩更新。对局重置会清除该环境的电机、待处理命令及反馈，保留其余环境的状态。迁移验收边界见[独立模型文档](../../../../../Model/GF43X40-10_testbench/UNILAB_PORT_CONTRACT.md)。当前的 64/1024 环境各 5 轮只证明运行链路，不证明已经学会步态或通过整机真机验证。

在 UniLab 仓库运行 `uv run --no-sync pytest tests/actuators/test_gf43x40_port.py -q`。测试包含冻结的 1000 步单轴 MuJoCo 力矩/反馈对照、批量状态重置与 20 ms/1 ms/20 ms 通信时序。
