# GF43X40-10 当前 v8 参数：中文名称与数值

力矩反馈从11个数值参数缩为1个比例参数，整个机械/观测数值配置42→32项：机械27项、速度反馈4项、粗略力矩反馈1项。运动输出保持不变，力矩反馈仅供粗略参考，不再追求高精度CAN读数对齐。

当前版本：`shared_fixed_gain_kd_band_v8_torque_proxy`；参数文件SHA-256：`25c27cf4b1a4bc391a03d5e18958255970efc594fe4a15622d533b20b8135c9e`。 [当前完整配置](parameters_current.json) · [精简前v8](baseline/parameters.json) · [验证报告](RESULTS.md)。

这些数值是等效仿真配置。`torque_scale` 和 `envelope_scale` 共享同一个放大尺度，不是“真机反馈1牛米，真实轴端就输出5.307牛米”。机械输出与CAN反馈分开计算；真实轴端力矩尚未由传感器独立测得。

## 机械施力与摩擦

| 中文名称 | 当前值 | 作用 | 代码名称 |
|---|---|---|---|
| 等效转动惯量 | 0.25429975910988406 | 决定仿真加减速所需的机械作用；随力矩尺度同步换算，不是独立实测惯量。 | `physics.armature` |
| 正向库仑摩擦 | 0.2789574700258063 | 正向运动的等效基础摩擦阻力。 | `physics.coulomb` |
| 黏性摩擦系数 | 0.22184734374066803 | 描述随速度增加的等效摩擦阻力。 | `physics.viscous` |
| 摩擦过渡速度 | 0.03280873402539427 | 使低速摩擦或方向变化平滑过渡的速度尺度，单位弧度每秒。 | `physics.friction_velocity` |
| 机械力矩尺度 | 5.307062074754017 | 把名义驱动作用换算到仿真机械作用的共用比例。 | `physics.torque_scale` |
| 驱动力矩响应时间常数 | 0.006753991462959274 | 控制驱动力矩跟随指令的快慢，单位秒。 | `physics.torque_time_constant` |
| 显式指令延迟 | 0 | 模型单独施加的指令延迟，单位秒；零不代表真实链路无延迟。 | `physics.command_delay` |
| 低速驱动及制动共用上限 | 47.232852465310756 | 同时确定低速驱动和制动力矩上限；保持原v8两者完全相同的数值。 | `physics.low_speed_drive_limit` |
| 反向库仑摩擦 | 0.26657846409584113 | 反向运动的等效基础摩擦阻力。 | `physics.coulomb_negative` |
| 制动力矩响应时间常数 | 0.0010000000065905075 | 控制制动力矩建立的快慢，单位秒。 | `physics.braking_time_constant` |
| 目标速度偏置 | -0.003890011703701447 | 指令通路的共用速度零点修正，单位弧度每秒。 | `physics.velocity_target_bias` |
| 制动过渡下沿速度 | 0.5 | 独立制动响应开始混合的速度下沿，单位弧度每秒。 | `physics.braking_transition_low` |
| 制动过渡上沿速度 | 2.0 | 独立制动响应完成混合的速度上沿，单位弧度每秒。 | `physics.braking_transition_high` |
| 摩擦模型类型 | 低速摩擦及静止黏着模型 | 使用低速摩擦变化与静止黏着、滑动相结合的模型。 | `physics.friction_model` |
| 正向静摩擦倍率 | 1.9768494886559929 | 正向静摩擦相对基础摩擦的比例。 | `physics.static_ratio` |
| 静止判定速度尺度 | 1e-05 | 静摩擦黏着判断所用的小速度阈值。 | `physics.stiction_velocity` |
| 前馈力矩增益 | 0.2445125963333848 | 调整前馈力矩指令产生的等效驱动作用。 | `physics.feedforward_gain` |
| 滑行阻力倍率 | 0.35 | 零增益、零前馈滑行时的被动阻力修正比例。 | `physics.coast_drag_scale` |
| 驱动作用变化率上限 | 990.5691290877769 | 限制名义驱动作用每秒变化的幅度，不是轴端力矩实测变化率。 | `physics.effort_slew_limit` |
| 反向静摩擦倍率 | 1.973417303124866 | 反向静摩擦相对基础摩擦的比例。 | `physics.static_ratio_negative` |
| 摩擦方向记忆强度 | 0.017171521159698763 | 描述摩擦受此前运动方向影响的程度。 | `physics.friction_memory_gain` |
| 摩擦记忆更新距离 | 0.012079130908237484 | 方向记忆随转动距离变化的尺度，单位弧度。 | `physics.friction_memory_distance` |
| 驱动释放时间倍率 | 0.9285251412884641 | 修正驱动作用减小或释放时的响应快慢。 | `physics.release_tc_ratio` |
| 内部速度滤波时间常数 | 0.00027181289353167277 | 内部控制所用速度状态的滤波快慢，单位秒。 | `physics.internal_velocity_tc` |
| 位置增益相关修正系数 | -0.003094526540601425 | 描述驱动行为随位置增益变化的共性差异。 | `physics.kp_gain_slope` |
| 速度增益相关修正系数 | -0.004168324898426792 | 描述驱动行为随速度增益变化的共性差异。 | `physics.kd_gain_slope` |
| 滑行切换时间常数 | 5.2449336094898405e-05 | 控制进入或退出滑行状态的过渡快慢，单位秒。 | `physics.coast_transition_tc` |
| 固定增益段速度阻尼修正系数 | **0.04（v8新增）** | v8新增。在位置增益约75至110、速度增益约3的范围平滑增强仿真内部等效速度阻尼；实验指令中的Kd不变，范围外逐渐回到v7。 | `physics.kd_gain_band_slope` |

## 位置、速度和反馈力矩观测

| 中文名称 | 当前值 | 作用 | 代码名称 |
|---|---|---|---|
| 速度编码方式 | 向下取整 | 向下取整的观测编码；由数据支持，不断言固件内部实现。 | `observation.velocity_encoding` |
| 速度反馈滤波时间常数 | 0.0025412807208095266 | 控制回报速度的滤波快慢，单位秒。 | `observation.velocity_filter_tc` |
| 速度反馈比例 | 0.9987451905437824 | 调整仿真回报速度的比例。 | `observation.velocity_feedback_scale` |
| 速度反馈零点偏置 | -0.0012717069082022608 | 调整仿真回报速度的共用零点，单位弧度每秒。 | `observation.velocity_feedback_bias` |
| 位置增益对速度滤波的修正 | 3.4681956441056276 | 描述速度反馈滤波随位置增益变化的程度。 | `observation.velocity_filter_kp_slope` |
| 反馈力矩模型类型 | 机械力矩单增益粗略反馈 | 反馈=torque_feedback_gain×仿真机械力矩；不参与机械计算。 | `observation.model` |
| 粗略力矩反馈比例 | 0.1198991043896492 | 将仿真施加的机械力矩乘此系数作为粗略读数，无独立力矩滤波或运动/负载修正；不是轴端力矩标定。 | `observation.torque_feedback_gain` |

## 运行协议范围

| 中文名称 | 当前值 | 作用 | 代码名称 |
|---|---|---|---|
| 位置协议范围 | 12.5 | 位置编码的正负范围，单位弧度，不是本次实验允许的机械行程。 | `registers.pmax` |
| 速度协议范围 | 10.0 | 速度编码的正负范围，单位弧度每秒。 | `registers.vmax` |
| 力矩协议范围 | 28.0 | 指令与反馈力矩的协议范围，单位牛米，不是实测轴端最大力矩。 | `registers.tmax` |

## 计算与采样时序

| 中文名称 | 当前值 | 作用 | 代码名称 |
|---|---|---|---|
| 物理仿真步长 | 0.001 | 每次物理积分推进的时间，单位秒。 | `timing.physics_step_s` |
| 指令更新周期 | 0.02 | 本次实验目标指令更新间隔，单位秒。 | `timing.command_update_s` |
| 反馈记录周期 | 0.02 | 本次实验反馈快照记录间隔，单位秒。 | `timing.feedback_sample_s` |
| 下位机周期的依据 | 用户确认的下位机循环 | 下位机循环周期来自用户确认，并非本次网关采样直接测得。 | `timing.stm32_cycle_evidence` |
| 对比时间基准 | 按实际反馈时刻与一毫秒步长仿真对照 | 按主机实际反馈快照时刻对比严格一毫秒积分的仿真。 | `timing.comparison` |

## 通信时间配置

| 中文名称 | 当前值 | 作用 | 代码名称 |
|---|---|---|---|
| 显式反馈延迟 | 0.0 | 单独附加的反馈延迟，单位秒。 | `communication.feedback_delay_s` |
| 是否扣除回报反馈龄期 | 关闭 | 关闭表示不按报文龄期额外平移反馈时间。 | `communication.use_reported_feedback_age` |
| 延迟辨识情况 | 未独立辨识单向固定延迟 | 原有响应已含观测相位；显式延迟候选损害运动回归，未采用。 | `communication.identifiability` |
| 时序抖动处理方式 | 使用实际记录的下发时刻或分块抽样 | 使用记录的实际下发时间，或分块重采样；不虚构未测丢包。 | `communication.jitter_policy` |

## 硬件参考资料（不参与运行计算）

| 中文名称 | 当前值 | 作用 | 代码名称 |
|---|---|---|---|
| 力矩常数寄存器值 | 0.0832 | 保留的电机力矩常数配置，未在本轮独立标定。 | `hardware_reference.torque_constant` |
| 减速比 | 40 | 配置中的减速比例。 | `hardware_reference.reduction_ratio` |
| 相电流限制 | 12.0 | 保留的相电流限值，单位安培。 | `hardware_reference.phase_current_limit` |
| 欠压阈值 | 12.0 | 保留的欠压保护配置，单位伏特。 | `hardware_reference.undervoltage` |
| 过压阈值 | 35.0 | 保留的过压保护配置，单位伏特。 | `hardware_reference.overvoltage` |
| 定子过温阈值 | 115.0 | 保留的定子温度保护配置，单位摄氏度。 | `hardware_reference.stator_overtemperature` |
| 功率管过温阈值 | 85.0 | 保留的功率管温度保护配置，单位摄氏度。 | `hardware_reference.mos_overtemperature` |
| 速度限制寄存器值 | 31.4 | 保留的速度限制配置，不等于本次验证过的最高速度。 | `hardware_reference.speed_limit_register` |
| 电流环带宽寄存器值 | 250.0 | 保留的电流控制带宽配置，未验证完整电流环动态。 | `hardware_reference.current_bandwidth` |

## 派生值（不再独立配置）

| 字段 | 关系 | 精确值 |
|---|---|---|
| `envelope_scale` | `physics.torque_scale` | 5.307062074754017 |
| `braking_limit` | `physics.low_speed_drive_limit` | 47.232852465310756 |
| `mechanical_torque_limit` | `physics.torque_scale * registers.tmax` | 148.5977380931125 |

删除的 `feedback_gain` 在当前观测分支被覆盖；删除的 `memory_gain` 原值为0，缺省仍为0。历史模型代码继续支持旧快照。

## 如何理解这几类数值

- **机械施力**：`torque_scale=5.307062074754017`、等效惯量、摩擦、速度包络和机械限幅共同决定仿真施加到关节的作用；包络倍率也为5.307062074754017。
- **反馈力矩**：由仿真状态另行生成CAN电流估算读数；当前反馈仅为机械力矩乘一个比例，原11参数观测模型不再执行。此前54系数反馈残差层已经移除，v8没有重新启用。
- **协议范围**：`tmax=28`是CAN名义力矩范围；`mechanical_torque_limit=148.5977380931125`是等效仿真内部截断数值，不能解释为真机真实峰值力矩。
- **时序**：`command_delay=0`和`feedback_delay_s=0`表示模型不额外施加独立延迟，不说明真实网络、STM32和CAN链路没有延迟。冻结文件中的20毫秒反馈周期是网关实验配置；单独采集的STM32高频窗口约1毫秒，见[时序结果](RESULTS.md)。
- **保护寄存器**：电流、电压、温度数值保留为配置参考，完整电气、热和持续过载动态尚未校准。

训练和验证来源、模型假设与历史对照仍可在[当前JSON](parameters_current.json)中查阅，它们属于记录元数据，不是独立执行器调节旋钮。
