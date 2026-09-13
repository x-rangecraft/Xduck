# GF43X40-10 24 V 电机 MuJoCo 建模、实机辨识与域随机化方案

## 1. 文档目的

本文定义 GF43X40-10 24 V 关节电机在机器人项目中的统一建模方法，供机械、嵌入式、控制、仿真和强化学习团队共同使用。

本文方案适用于以下实际条件：

- 当前使用 3 台电机进行重复空载实验，降低单台电机偶然性的影响；
- 最终机器人安装 14 台同型号电机；
- 数据通过 USB-CAN 或最终机器人控制器采集；
- 可用反馈为位置、速度、力矩、电机温度、MOS 温度和故障状态；
- 不依赖外部力矩传感器、测功机或相电流测量设备；
- 对电机反馈力矩采用工程假设：`CAN 力矩反馈近似等于输出轴力矩`；
- 模型目标是准确复现控制器可观察的关节输入—输出行为，而不是重建驱动器内部电磁细节。

最终模型由四部分组成：

1. 厂商标称参数和 24 V 官方性能曲线；
2. 3 台裸电机空载辨识结果；
3. 14 关节安装到机器人后的真机辨识结果；
4. 针对拟合误差和电机个体差异的域随机化。

---

## 2. 统一工程口径

项目内统一采用以下定义，未经评审不得在不同模块中使用其他数值。

| 项目 | 统一值 | 工程含义 |
|---|---:|---|
| 电机型号 | GF43X40-10 24 V | 40:1、362 g 版本 |
| 标称母线电压 | 24 V | 默认仿真电压 |
| 允许工作电压范围 | 待确认 | PDF 仅给出 24 V 额定值；性能表实测 23.834～24.085 V |
| 连续额定输出力矩 | 8.9 N·m | GF43X40-10 额定值 |
| 峰值输出力矩 | 23.5 N·m | GF43X40-10 物理输出硬上限 |
| 额定输出速度 | 43 rpm = 4.5029 rad/s | GF43X40-10 额定值 |
| 表中最高实测速度 | 55 rpm = 5.7596 rad/s | 第 9 页性能表，不是 `Vmax` 映射范围 |
| 三台真机最高瞬时速度 | 55.48 rpm = 5.8095 rad/s | 电机 15 空载速度扫描 CAN 反馈 |
| 减速比 | 40:1 | 输出轴坐标定义 |
| 额定相电流 | 2.0 A | GF43X40-10 额定值 |
| 额定母线电流 | 2.3 A | GF43X40-10 额定值 |
| 额定/峰值功率 | 40/70 W | GF43X40-10 规格 |
| 电机转矩常数 | 0.095 N·m/A | 厂商电机侧常数，不可直接当作输出轴力矩/母线电流 |
| 减速器背隙 | 15 arcmin = 0.004363 rad | 可用于回差模型 |
| 编码器 | 双编码器，位数待确认 | 该型号参数表未给出位数 |
| CAN | 500 kbps（本批实机） | 三台样机实测通信速率；部署前按批次复核 |
| MIT `Kp`范围 | 0～500 N·m/rad | 实际值必须与部署端一致 |
| MIT `Kd`范围 | 0～5 N·m·s/rad | 实际值必须与部署端一致 |
| 电机/驱动器温限 | 待对应型号验证 | GF43X40-10 页未给出具体温限 |
| 单电机质量 | 约 0.362 kg | 必须计入机器人连杆质量和惯量 |
| 外形尺寸 | 约 Ø57 mm × 56.5 mm | 不含定位销 |

### 2.1 额定值与峰值值的关系

8.9 N·m 和 4.5029 rad/s 是额定工作拐点，不是仿真的硬裁剪边界。23.5 N·m 是物理峰值力矩；厂家表最高速度为 5.7596 rad/s，三台真机空载瞬时最高值为 5.8095 rad/s。

额定点机械功率约为：

\[
P_{\mathrm{rated}}=8.9\times4.5029\approx40.1\ \mathrm{W}
\]

如果错误地同时使用峰值力矩和表中最高速度，则机械功率为：

\[
23.5\times5.8095\approx136.5\ \mathrm{W}
\]

因此电机能力必须表示为随速度、温度和过载历史变化的力矩上限：

\[
|\tau|\le\tau_{\max}(|\omega|,T,H)\le23.5\ \mathrm{N\,m}
\]

其中：

- \(\omega\)：输出轴速度；
- \(T\)：电机和 MOS 温度；
- \(H\)：短时过载累积状态。

---

## 3. 最终保留的模型参数

### 3.1 厂商标称参数

| 参数 | 符号 | 来源 | 在模型中的作用 |
|---|---|---|---|
| 标称电压 | \(V_{\mathrm{nom}}=24\ \mathrm{V}\) | 说明书 | 功率计算和电压基准 |
| 工作电压范围 | 待确认 | GF43X40-10 表仅给出 24 V 额定值 | 待补充后再扩大电压域随机化 |
| 连续力矩拐点 | \(\tau_{\mathrm{rated}}=8.9\ \mathrm{N\,m}\) | GF43X40-10 参数表 | 连续区与过载区分界 |
| 峰值力矩 | \(\tau_{\mathrm{peak}}=23.5\ \mathrm{N\,m}\) | GF43X40-10 参数表 | 电机物理输出硬上限 |
| 额定速度拐点 | \(\omega_{\mathrm{rated}}=4.5029\ \mathrm{rad/s}\) | GF43X40-10 参数表 | 额定区与高速区分界 |
| 表中最高实测速度 | \(\omega_{\mathrm{table,max}}=5.7596\ \mathrm{rad/s}\) | GF43X40-10 性能表 | 初始力矩—速度曲线边界 |
| 减速比 | \(N=40\) | 说明书 | 输入/输出轴定义和资料记录 |
| 编码器分辨率 | 待确认 | GF43X40-10 参数表未给出 | 当前仅用 CAN 帧量化步长 |
| 电机和驱动温限 | 待确认 | GF43X40-10 参数表未给出 | 不沿用其他型号温升结果 |

### 3.2 官方曲线参数

官方性能图需要数字化为 CSV，并采用分段线性或 PCHIP 单调插值。禁止使用可能在边界产生振荡、负电流或非单调结果的高阶多项式。

需要生成以下查表数据：

| 查表 | 自变量 | 输出 | 用途 |
|---|---|---|---|
| 力矩—母线电流 | \(|\tau|\) | \(I_{\mathrm{bus}}^{\mathrm{ref}}\) | 24 V 稳态电流估算 |
| 力矩—速度 | \(|\tau|\) | \(\omega^{\mathrm{curve}}\) | 初始力矩—速度能力边界 |
| 力矩—效率 | \(|\tau|\) | \(\eta^{\mathrm{ref}}\) | 输入功率和损耗估算 |
| 力矩—输入功率 | \(|\tau|\) | \(P_{\mathrm{in}}^{\mathrm{ref}}\) | 校验母线电流模型 |
| 力矩—输出功率 | \(|\tau|\) | \(P_{\mathrm{out}}^{\mathrm{ref}}\) | 校验 \(\tau\omega\) |
| 温度—时间 | \(t\) | \(T_{\mathrm{motor}}\) | 热模型初值 |

建议文件名：

```text
gf43x40_10_24v_torque_current.csv
gf43x40_10_24v_torque_speed.csv
gf43x40_10_24v_torque_efficiency.csv
gf43x40_10_24v_torque_input_power.csv
gf43x40_10_24v_temperature_rated.csv
```

官方曲线是 24 V 特定工况下的稳态截面。曲线没有覆盖的高速、高力矩、反向和制动区域由实机数据补充，并通过域随机化表达不确定性。

### 3.3 空载辨识参数

| 参数 | 符号 | 数据来源 | 用途 |
|---|---|---|---|
| 位置偏置 | \(b_q\) | 静止 CAN 数据 | 传感器视图 |
| 位置噪声 | \(\sigma_q\) | 静止 CAN 数据 | 观测随机化 |
| 速度零偏 | \(b_{\dot q}\) | 静止 CAN 数据 | 传感器视图 |
| 速度噪声 | \(\sigma_{\dot q}\) | 静止 CAN 数据 | 观测随机化 |
| 力矩反馈偏置 | \(b_\tau\) | 零指令 CAN 数据 | 力矩映射 |
| 力矩反馈噪声 | \(\sigma_\tau\) | 零指令 CAN 数据 | 力矩随机化 |
| 正/反向力矩增益 | \(a_+,a_-\) | 指令—反馈数据 | 描述方向不对称 |
| 有效库仑摩擦 | \(\tau_c^+,\tau_c^-\) | 正反低速数据 | 输出轴摩擦模型 |
| 有效黏性阻尼 | \(b^+,b^-\) | 多速度数据 | 输出轴摩擦模型 |
| 综合输出侧惯量 | \(J_{\mathrm{eff}}\) | 力矩脉冲和角加速度 | MuJoCo `armature` |
| 综合指令延迟 | \(t_d\) | 发送与反馈时间戳 | 指令 FIFO |
| 执行器时间常数 | \(T_{\mathrm{act}}\) | 力矩阶跃 | 力矩动态 |
| 延迟抖动 | \(\sigma_d\) | 多次重复测试 | 通信随机化 |
| CAN 丢包率 | \(p_{\mathrm{loss}}\) | 长时间帧统计 | 丢帧与保持模型 |
| 实际空载最高速度 | \(\omega_{\max}^{\mathrm{meas}}\) | 正反速度扫描 | 验证官方峰值速度 |
| 空载温升参数 | \(\alpha_{0},\beta_{0}\) | 多速度运行 | 热模型初值 |

### 3.4 真机带载辨识参数

| 参数 | 符号 | 数据来源 | 用途 |
|---|---|---|---|
| 每关节力矩比例修正 | \(s_j\) | 机器人悬空逆动力学 | 修正指令/反馈比例 |
| 每关节摩擦修正 | \(\tau_{c,j},b_j\) | 悬空慢速正反运动 | 描述装配和负载差异 |
| 每关节惯量修正 | \(J_j\) | 悬空阶跃或扫频 | 修正 `armature` |
| 每关节响应延迟 | \(t_{d,j}\) | 悬空动态数据 | 关节延迟模型 |
| 每关节响应时间常数 | \(T_{\mathrm{act},j}\) | 指令—反馈动态 | 关节执行器模型 |
| 力矩—速度增益曲面 | \(G_j(\tau,\omega)\) | 全工作域数据 | 高速和过载区非线性 |
| 力矩—速度能力边界 | \(\tau_{\max,j}(\omega)\) | 稳态和短时数据 | 力矩限制 |
| 电机温升参数 | \(\alpha_{\tau,j},\alpha_{\omega,j},\beta_{m,j}\) | 电机温度反馈 | 电机热状态 |
| MOS 温升参数 | \(\alpha_{d,j},\beta_{d,j}\) | MOS 温度反馈 | 驱动器热状态 |
| 过载累积和恢复参数 | \(k_{H,j},\tau_{H,j}\) | 8.9 N·m 以上短时数据 | 峰值力矩持续能力 |
| 14 电机总线延迟 | \(d_{14}\) | 最终系统同步采集 | 替换 USB-CAN 临时延迟 |
| 14 电机丢包特性 | \(p_{14}\) | 最终系统长时间统计 | 最终通信模型 |

### 3.5 当前三台空载实测参数（2026-09-02）

以下结果来自 CAN ID 13、14、15；速度动态使用第 1、2 轮，小力矩脉冲使用 3 轮，
高力矩码值扫描使用 `Kd=4`、50 ms 脉冲。完整机器可读文件为
`gf43x40-10-i2rt/data/processed/gf43x40_10_empty_fit.json`。

| 参数 | 当前标称值 | 三台范围/说明 |
|---|---:|---|
| 综合输出侧惯量 `J_eff` | 0.03047 kg·m² | 0.02246～0.03118 |
| 正向库仑摩擦 | 0.2715 N·m | 0.2397～0.2726 |
| 反向库仑摩擦 | 0.3248 N·m | 0.2977～0.3318 |
| 正向黏性阻尼 | 0.02965 N·m·s/rad | 0.02456～0.03069 |
| 反向黏性阻尼 | 0.03051 N·m·s/rad | 0.02731～0.03443 |
| CAN 力矩反馈偏置 | -0.00684 N·m | 三台静止中位数一致 |
| 正/反速度增益 | 0.9827 / 0.9819 | 稳态拟合 `R²>0.997` |
| 执行器时间常数（`Kd=0.2`） | 0.0933 s | 0.0896～0.0973 s |
| 执行器时间常数（`Kd=4`） | 0.0522 s | 0.0516～0.0713 s |
| 综合通信延迟 | 0～5 ms | 200 Hz 下为 0～1 控制周期 |
| 动态丢包概率 | 0.00094 | 三台约 0.00077～0.00096 |
| 位置噪声标准差上界 | 0.000191 rad | 静止数据未跨越量化步长 |
| 速度噪声标准差上界 | 0.002442 rad/s | 半个 CAN 量化步长 |
| 力矩噪声标准差上界 | 0.006838 N·m | 半个 CAN 量化步长 |
| MIT 速度映射上限 `Vmax` | 10 rad/s | 三台电机寄存器一致；用于指令/反馈编解码与裁剪 |
| MIT 力矩映射上限 `Tmax` | 28 N·m | 三台电机寄存器一致；用于指令/反馈编解码与裁剪 |

`Vmax/Tmax` 是当前电机 MIT 帧的映射及可发送范围，不等于已通过负载实验验证的稳态
机械能力。当前力矩指令编码先裁剪到 ±28 N·m，但 GF43X40-10 物理输出必须再限制到
±23.5 N·m；速度指令/反馈映射范围为 ±10 rad/s。

### 3.6 GF43X40-10 24 V 性能表

PDF 第 9～10 页的 29 个厂商实测点已逐行写入
`gf43x40_10_empty_fit.json.datasheet_gf43x40_10_24V.performance_table`，包含电压、母线电流、
输入功率、转矩、转速和输出功率；效率由输出功率/输入功率计算。

该表精确覆盖 `1.14～17.86 N·m`，速度从 `55.0 rpm` 降至 `30.6 rpm`。
额定附近表格点为 `8.89 N·m, 46.7 rpm, 2.232 A, 53.154 W 输入, 43.464 W 输出`。
表格未覆盖的 `17.86～23.5 N·m` 峰值区采用全表三次拟合外推。三次模型的全表
`R²=0.99920`、RMSE 为 `0.180 rpm`，优于线性和二次模型；为保证边界连续，
模型整体平移 `-0.2586 rpm`，使其严格经过最后一个实测点 `(17.86 N·m, 30.6 rpm)`：

\[
n(\tau)=
-0.00360470\tau^3
+0.05890379\tau^2
-1.34237612\tau
+56.5802102
-0.2585921
\]

| 力矩 (N·m) | 标称拟合速度 (rpm) | 标称拟合速度 (rad/s) | 标称机械功率 (W) | 二次/三次模型速度范围 (rpm) |
|---:|---:|---:|---:|---:|
| 17.86 | 30.60 | 3.204 | 57.23 | 30.60～30.60 |
| 18 | 30.22 | 3.165 | 56.97 | 30.22～30.30 |
| 19 | 27.36 | 2.865 | 54.43 | 27.36～28.08 |
| 20 | 24.20 | 2.534 | 50.68 | 24.20～25.77 |
| 21 | 20.73 | 2.170 | 45.58 | 20.73～23.37 |
| 22 | 16.92 | 1.771 | 38.97 | 16.92～20.87 |
| 23 | 12.75 | 1.335 | 30.71 | 12.75～18.27 |
| 23.5 | 10.52 | 1.102 | 25.90 | 10.52～16.94 |

该段满足速度单调下降、物理峰值力矩不超过 23.5 N·m、机械功率不超过 70 W。
它属于模型外推而非厂家实测，训练时应在表中“不确定范围”内随机化，并在将来取得
受控负载数据后直接替换。

GF43X40-10 没有在此 PDF 中提供温升表，因此不再沿用先前其他 4340 版本的
12/14 N·m 温升曲线。

位置绝对偏置仍需已知机械零位夹具；当前静止位置值不能直接解释为编码器偏置。50 ms
高力矩数据只表示瞬态命令响应，不能替代受控负载下的稳态力矩能力曲线。

---

## 4. 数据采集统一格式

所有空载和真机实验使用相同字段，避免后期无法合并数据。

```text
experiment_id
run_id
motor_serial_or_index
joint_name
test_condition

timestamp_tx
timestamp_rx

q_des
qd_des
kp
kd
tau_ff

q_feedback
qd_feedback
tau_feedback

motor_temperature
mos_temperature
fault_state

power_supply_voltage       # 电源可提供时记录
power_supply_current       # 电源可提供时记录

robot_base_pose            # 真机实验
robot_base_velocity        # 真机实验
contact_state              # 真机验证数据
```

要求：

- 保存原始 CAN 数据，不只保存滤波后数据；
- 发送时间和接收时间分别记录；
- 记录实际使用的控制频率和物理步长；
- 每次实验记录冷机/热机状态和环境温度；
- 每个实验文件必须包含电机编号、关节名称和软件版本；
- 训练集与验证集按完整实验段划分，不能随机打散单帧后划分。

---

## 5. 三台裸电机空载实验

三台电机执行完全相同的实验，每项至少重复 3 次。测试结果既用于确定标称值，也用于估计电机间差异。

### 5.1 静止噪声实验

每台电机分别执行：

1. 失能静止记录 60 s；
2. 使能、零目标速度和零前馈力矩记录 60 s；
3. 每个状态重复 3 次。

稳健统计：

\[
b_x=\operatorname{median}(x)
\]

\[
\sigma_x\approx1.4826\operatorname{MAD}(x)
\]

分别得到 \(q\)、\(\dot q\)、\(\tau\) 的偏置和噪声。

### 5.2 空载速度扫描

正反方向均测试以下目标速度：

```text
0.25, 0.5, 0.75, 1.0,
1.5, 2.0, 2.5, 3.0, 3.5,
4.0, 4.5029, 4.8,
5.2, 5.5, 5.7596, 5.8095 rad/s
```

每个速度点：

1. 使用 1～2 s 平滑加速；
2. 稳态保持 5～10 s；
3. 平滑回到零速；
4. 等待速度稳定后进入下一点；
5. 完成正向后重复反向。

该实验用于拟合：

- 目标速度—实际速度关系；
- 额定速度以下的跟踪误差；
- 额定速度以上、峰值速度以下的短时响应；
- 正反方向差异；
- 空载摩擦；
- 高速区温升。

### 5.3 小力矩脉冲实验

裸电机不能直接进行大力矩测试。初始测试使用：

```text
±0.2, ±0.5, ±1.0 N·m
```

单次脉冲 100～300 ms，并使用小幅速度阻尼限制空载加速。

力矩指令为：

\[
\tau_{\mathrm{cmd}}
=K_p(q_d-q_s)+K_d(\dot q_d-\dot q_s)+\tau_{\mathrm{ff}}
\]

正反方向分别拟合：

\[
\tau_{\mathrm{fb}}=
\begin{cases}
a_+\tau_{\mathrm{cmd}}+b_\tau,&\tau_{\mathrm{cmd}}\ge0\\
a_-\tau_{\mathrm{cmd}}+b_\tau,&\tau_{\mathrm{cmd}}<0
\end{cases}
\]

### 5.4 延迟和时间常数

对小力矩阶跃拟合离散一阶模型：

\[
\tau_{k+1}
=a\tau_k+(1-a)\tau_{\mathrm{target},k-d}
\]

其中：

\[
T_{\mathrm{act}}=-\frac{\Delta t}{\ln a}
\]

\(d\) 是延迟步数，\(T_{\mathrm{act}}\) 是执行器综合时间常数。

### 5.5 摩擦和综合惯量

采用平滑后的速度计算角加速度，禁止直接对带量化噪声的速度做简单差分。推荐 Savitzky–Golay 滤波或带正则化的局部多项式导数。

拟合：

\[
\tau_{\mathrm{fb}}
=J_{\mathrm{eff}}\ddot q
+\tau_c\tanh(\dot q/\epsilon)
+b\dot q+\tau_0
\]

得到：

- 综合输出侧惯量 \(J_{\mathrm{eff}}\)；
- 有效库仑摩擦 \(\tau_c\)；
- 有效黏性阻尼 \(b\)；
- 静态偏置 \(\tau_0\)。

正反方向差异明显时，应分别拟合 \(\tau_c^\pm\) 和 \(b^\pm\)。

### 5.6 空载温升

每台电机分别在以下速度持续运行：

```text
0.5, 1.0, 2.0, 4.5029, 4.8, 5.5 rad/s
```

记录电机和 MOS 温度。测试目的是获得正常运行区的温升斜率，不需要主动接近保护温度。

---

## 6. 机器人真机带载实验

### 6.1 辨识状态与验证状态分离

- 参数辨识：机器人主体可靠支撑，被测关节或腿悬空，不接触地面；
- 整体验证：站立、下蹲、迈步、行走和恢复动作；
- 有地面接触的数据不用于独立拟合绝对关节动力学，因为其中包含未知接触力。

### 6.2 静态姿态实验

选择多个关节角度并保持静止：

\[
\dot q=0,\qquad\ddot q=0
\]

MuJoCo 根据机器人质量和质心计算重力力矩 \(g(q)\)，拟合：

\[
g_j(q)=s_j\tau_{\mathrm{fb},j}+b_j
\]

得到每个关节的力矩比例修正和静态偏置。

### 6.3 悬空慢速扫频

每次只主动运动一个关节，其他关节保持固定：

\[
q_d(t)=q_0+A\sin(2\pi ft)
\]

建议范围：

```text
A = 0.05, 0.10, 0.15, 0.20 rad
f = 0.2, 0.5, 1.0, 1.5, 2.0 Hz
```

拟合：

\[
M(q)\ddot q+C(q,\dot q)\dot q+g(q)
=s_j\tau_{\mathrm{fb},j}
-\tau_{c,j}\tanh(\dot q_j/\epsilon)
-b_j\dot q_j
\]

用于修正每关节的惯量、摩擦、阻尼、力矩比例和延迟。

### 6.4 额定区和峰值区数据覆盖

最终目标是覆盖力矩和速度的二维工作域，而不是只测额定点。

| 力矩范围 | 0～0.5 rad/s | 0.5～2 rad/s | 2～4.5029 rad/s | 4.5029～5.2 rad/s | 5.2～5.8095 rad/s |
|---|---|---|---|---|---|
| 0～4 N·m | 必须覆盖 | 必须覆盖 | 必须覆盖 | 必须覆盖 | 必须覆盖 |
| 4～8 N·m | 必须覆盖 | 必须覆盖 | 必须覆盖 | 尽量覆盖 | 尽量覆盖 |
| 8～8.9 N·m | 必须覆盖 | 必须覆盖 | 必须覆盖 | 尽量覆盖 | 只采自然数据 |
| 8.9～16 N·m | 短时覆盖 | 短时覆盖 | 短时覆盖 | 只采自然数据 | 不主动要求 |
| 16～23.5 N·m | 短脉冲 | 短脉冲 | 只采自然数据 | 不主动测试 | 不测试 |

此表是数据覆盖目标，不是要求强制命令所有组合。高力矩主要在低速区采集，高速度主要在低力矩区采集。

### 6.5 过额定力矩采集

过额定区逐级测试：

```text
8.9, 12, 16, 20, 23.5 N·m
```

工程探索的初始脉冲上限：

| 区间 | 初始单次脉冲上限 | 约束 |
|---|---:|---|
| 8.9～16 N·m | 100～300 ms | 单关节、可靠支撑 |
| 16～20 N·m | 50～150 ms | 结构确认后逐级进入 |
| 20～23.5 N·m | 约 50 ms 起步 | 不得超过 GF43X40-10 峰值力矩 |

以上时间是保守的数据采集起点，不是厂商连续工作承诺。温度反馈存在滞后，不能因为瞬时温度不高而连续重复峰值脉冲。

### 6.6 过额定速度采集

在小角度范围内逐步覆盖：

```text
4.0, 4.5029, 4.8, 5.2, 5.5, 5.7596, 5.8095 rad/s
```

额定速度以上测试应使用较小位置幅度，避免大范围高速摆动。重点记录：

- 高速区力矩反馈是否下降；
- 跟踪误差是否明显增加；
- 响应时间是否变化；
- 驱动与制动方向是否不对称；
- 电机和 MOS 温升是否加快。

### 6.7 四象限分类

所有数据按机械功率符号分类：

\[
P_{\mathrm{mech}}=\tau\omega
\]

| 状态 | 判定 | 数据含义 |
|---|---|---|
| 正向驱动 | \(\tau>0,\omega>0\) | 正向输出功率 |
| 反向驱动 | \(\tau<0,\omega<0\) | 反向输出功率 |
| 正向制动 | \(\tau<0,\omega>0\) | 电机吸收机械能 |
| 反向制动 | \(\tau>0,\omega<0\) | 电机吸收机械能 |

驱动和制动数据不得混在同一个增益和时间常数中直接拟合。

---

## 7. 工作域拟合方法

### 7.1 数据分箱

建议速度分箱：

```text
[0, 0.5), [0.5, 1.0), [1.0, 2.0), [2.0, 3.0),
[3.0, 4.0), [4.0, 4.5029), [4.5029, 5.2), [5.2, 5.8095]
```

建议力矩分箱：

```text
[0, 2), [2, 4), [4, 6), [6, 8), [8, 8.9), [8.9, 12),
[12, 16), [16, 20), [20, 23.5]
```

每个网格单元统计：

- 样本数量；
- 力矩指令与反馈的中位数；
- 速度中位数；
- 正/反向增益；
- 时间常数；
- 延迟；
- 电机和 MOS 温度；
- 中位绝对偏差 MAD；
- 驱动或制动状态。

### 7.2 力矩响应曲面

拟合：

\[
\tau_{\mathrm{fb}}
=G(\tau_{\mathrm{cmd}},\omega,T,m)\tau_{\mathrm{cmd}}+b_\tau
\]

其中 \(m\) 表示驱动/制动和正/反方向模式。

第一版可使用二维查表：

\[
G=G(|\tau_{\mathrm{cmd}}|,|\omega|)
\]

数据充足后再增加温度和四象限维度。查表采用分段线性或单调插值，禁止在无数据区域任意外推。

### 7.3 时间常数曲面

\[
T_{\mathrm{act}}=T_{\mathrm{act}}(|\tau|,|\omega|,m)
\]

额定区、高力矩区和高速区分别拟合，不假设一个时间常数能够覆盖全部工况。

### 7.4 力矩—速度边界

厂家表和峰值区拟合首先给出 \(\omega_{\max}(\tau)\)，运行时将这条单调曲线反查为
\(\tau_{\max}^{\mathrm{curve}}(|\omega|)\)。因此同方向驱动的最终能力边界为：

\[
\tau_{\max}(\omega,T,H)
=\min\left(
23.5,
\tau_{\max}^{\mathrm{curve}}(\omega),
\tau_{\max}^{T}(T),
\tau_{\max}^{H}(H)
\right)
\]

当 \(\tau\omega<0\) 为制动工况时，不使用同方向驱动力矩—速度曲线裁剪，否则高速时会
错误削弱制动能力；但仍受 23.5 N·m 物理峰值、温度、过载和关节结构限制。查表采用分段
线性插值，\(|\omega|\ge5.8095\ \mathrm{rad/s}\) 时同方向加速力矩为零。
厂家表首行给出 `55 rpm、1.14 N·m`，模型保留该厂家点，并继续线性连接到三台真机的
空载瞬时最高端点 \((5.8095\ \mathrm{rad/s},0\ \mathrm{N\,m})\)。这里的零表示不能
再对外提供同方向负载力矩，不表示电机内部电磁力矩或克服摩擦所需力矩为零。

只有出现以下证据时，才判定数据接近能力边界：

- 力矩反馈不再随指令增加；
- 速度明显下降；
- 跟踪误差快速增加；
- 温升斜率显著上升；
- 驱动器出现限流、过载或故障状态。

没有采到某个网格单元的数据，不等于该组合不可达。该区域应标记为“稀疏/未知”，并在域随机化中扩大不确定性。

---

## 8. 电流、功率、温度和过载模型

### 8.1 母线电流和功率

官方曲线提供参考工况下的母线电流：

\[
I_{\mathrm{bus}}^{\mathrm{ref}}=f_I(|\tau|)
\]

注意该电流是 24 V 母线电流，不是电机相电流。

基础功率计算：

\[
P_{\mathrm{in}}=V_{\mathrm{bus}}I_{\mathrm{bus}}
\]

\[
P_{\mathrm{out}}=|\tau\omega|
\]

\[
P_{\mathrm{loss}}=\max(P_{\mathrm{in}}-P_{\mathrm{out}},0)
\]

如果最终电源能提供总电流，则用 14 台电机数据拟合：

\[
I_{\mathrm{supply}}
\approx I_{\mathrm{idle}}
+\sum_{j=1}^{14}f_I(\tau_j,\omega_j)
\]

如果没有同步总电流记录，则保留官方查表，并对未覆盖工况使用较宽的不确定性范围。

### 8.2 温度模型

直接使用力矩、速度和温度反馈拟合经验热模型：

\[
\dot T_{\mathrm{motor}}
=\alpha_\tau\tau^2
+\alpha_\omega\omega^2
+\alpha_P|\tau\omega|
-\beta_m(T_{\mathrm{motor}}-T_{\mathrm{ambient}})
\]

\[
\dot T_{\mathrm{MOS}}
=\gamma_\tau\tau^2
+\gamma_\omega\omega^2
-\beta_d(T_{\mathrm{MOS}}-T_{\mathrm{ambient}})
\]

离散数据可通过带非负约束的最小二乘拟合。温度导数先进行平滑，避免量化台阶放大。

### 8.3 过载累积

定义过载状态：

\[
\dot H=
\begin{cases}
 -H/\tau_H,&|\tau|\le8.9\\
k_H\left[(|\tau|/8.9)^2-1\right]-H/\tau_H,&|\tau|>8.9
\end{cases}
\]

根据过载状态限制峰值能力：

\[
\tau_{\max}^{H}
=8.9+(23.5-8.9)\operatorname{clip}(1-H,0,1)
\]

这样允许机器人短时使用超过 8.9 N·m 的力矩，但阻止策略持续依赖峰值输出。

---

## 9. 最终执行器模型

### 9.1 传感器视图

\[
q_s=q+b_q+n_q
\]

\[
\dot q_s=\dot q+b_{\dot q}+n_{\dot q}
\]

### 9.2 MIT 指令

\[
\tau_{\mathrm{cmd}}
=K_p(q_d-q_s)
+K_d(\dot q_d-\dot q_s)
+\tau_{\mathrm{ff}}
\]

### 9.3 延迟、增益和动态

延迟后的目标力矩：

\[
\tau_{\mathrm{target}}(t)
=G(\tau_{\mathrm{cmd}},\omega,T,m)
\tau_{\mathrm{cmd}}(t-t_d)+b_\tau
\]

一阶动态：

\[
\dot\tau_{\mathrm{act}}
=\frac{\tau_{\mathrm{target}}-\tau_{\mathrm{act}}}
{T_{\mathrm{act}}(|\tau|,|\omega|,m)}
\]

### 9.4 摩擦与最终输出

\[
\tau_f=
\tau_c^\pm\tanh(\dot q/\epsilon)+b^\pm\dot q
\]

\[
\boxed{
\tau_{\mathrm{joint}}
=\operatorname{clip}
(\tau_{\mathrm{act}},-\tau_{\max},\tau_{\max})
-\tau_f
}
\]

当 \(|\dot q|\ge5.8095\ \mathrm{rad/s}\) 时，禁止继续施加与速度同方向的加速力矩，但必须允许反方向制动力矩。不得直接截断 MuJoCo 的 `qvel`，否则会产生非物理的瞬时动量变化。

---

## 10. MuJoCo 实现

### 10.1 MJCF 配置

用输出轴 torque motor 替换旧 XL330 位置执行器：

```xml
<joint
    name="left_knee"
    type="hinge"
    armature="0.03047"
    damping="0"
    frictionloss="0"/>

<actuator>
  <motor
      name="left_knee_motor"
      joint="left_knee"
      gear="1"
      ctrllimited="true"
      ctrlrange="-28 28"
      forcelimited="true"
      forcerange="-23.5 23.5"/>
</actuator>
```

说明：

- `ctrlrange=±28 N·m` 对应寄存器 `Tmax=28 N·m` 的命令映射范围；
- `forcerange=±23.5 N·m` 对应 GF43X40-10 的物理峰值力矩；
- MIT 速度指令和速度反馈编解码范围为 `[-10, 10] rad/s`；
- 实际可用力矩由运行时的速度、温度和过载模型限制；
- 摩擦由自定义执行器计算时，XML 中 `damping` 和 `frictionloss` 保持为 0，避免重复计算；
- `armature` 使用空载和真机辨识得到的综合输出侧惯量；
- 每台 0.362 kg 电机的质量和惯量还必须正确分配到对应连杆 body，不能全部计入 `armature`。

### 10.2 每个物理步的计算顺序

```python
def step_gf43x40_10(action, state, params, dt):
    qd_des = clip(action.qd_des, -10.0, 10.0)
    tau_ff = clip(action.tau_ff, -28.0, 28.0)
    q_sensor = state.q + params.q_bias + sample_position_noise(params)
    qd_sensor = state.qd + params.qd_bias + sample_velocity_noise(params)

    tau_cmd = (
        params.kp * (action.q_des - q_sensor)
        + params.kd * (qd_des - qd_sensor)
        + tau_ff
    )
    tau_cmd = clip(tau_cmd, -28.0, 28.0)

    tau_delayed = state.delay_buffer.push_and_read(tau_cmd)
    mode = classify_quadrant(tau_delayed, state.qd)

    gain = lookup_gain(
        abs(tau_delayed), abs(state.qd), state.motor_temp, mode
    )
    tau_target = gain * tau_delayed + params.torque_bias

    physical_limit = min(
        23.5,
        temperature_limit(state.motor_temp, state.mos_temp),
        overload_limit(state.overload),
        params.joint_structural_limit,
    )

    if tau_target * state.qd > 0.0:  # same-direction acceleration
        speed_limit = lookup_torque_speed_limit(abs(state.qd))
        applied_limit = min(physical_limit, speed_limit)
    else:  # opposite-direction braking
        applied_limit = physical_limit

    tau_target = clip(tau_target, -applied_limit, applied_limit)

    time_constant = lookup_time_constant(
        abs(tau_target), abs(state.qd), mode
    )
    alpha = 1.0 - exp(-dt / max(time_constant, 1e-6))
    state.tau += alpha * (tau_target - state.tau)

    friction = friction_model(state.qd, params)
    tau_joint = state.tau - friction

    update_power_temperature_and_overload(state, tau_joint, dt, params)
    return tau_joint
```

### 10.3 训练与部署一致性

以下项目必须在训练端和部署端完全一致：

- 策略控制频率；
- MuJoCo 物理步长和 action decimation；
- action scale；
- 默认关节姿态；
- MIT `Kp/Kd`；
- 位置、速度和力矩裁剪；
- 延迟单位是物理步还是控制步；
- 丢帧后保持上一帧还是清零；
- 观测滤波和历史窗口；
- 正负方向和关节索引。

---

## 11. UniLab 域随机化

UniLab 将域随机化分为 init、reset 和 interval 三种生命周期。GF43X40-10 的电机参数主要在 reset 时采样，在每个物理步的 pre-step control callback 中使用。

参考：

- [UniLab Domain Randomization](https://unilabsim.github.io/UniLab-doc/en/2-user_guide/5-domain_randomization/0-index.html)
- [UniLab Domain Randomization Configuration](https://unilabsim.github.io/UniLab-doc/en/2-user_guide/5-domain_randomization/1-configuration.html)
- [UniLab Domain Randomization Contract](https://unilabsim.github.io/UniLab-doc/en/4-developer_guide/2-contracts/4-dr_contract.html)

### 11.1 随机化参数与初始范围

下表范围用于实测数据尚不完整的初始训练。获得 14 台电机数据后必须根据数据分布更新。

| 参数 | 初始范围 | 生命周期 | 相关性要求 |
|---|---:|---|---|
| 曲线力矩轴缩放 \(s_\tau\) | 0.90～1.00 | reset | 整条曲线统一缩放，不单独采样额定/峰值力矩 |
| 曲线速度轴缩放 \(s_\omega\) | 0.9773～1.00 | reset | 来自三台真机最高速度范围 |
| 峰值区形状混合 \(\xi\) | 0～1 | reset | 在三次标称与二次上界之间插值 |
| 峰值功率预算 \(P_{peak,e}\) | 63～70 W | reset | 对整条曲线施加功率双曲线约束 |
| 额定/峰值力矩和速度 | 由随机曲线派生 | reset | 禁止作为四个独立随机变量 |
| 额定区力矩增益 | 拟合值 ±10% | reset | 每关节独立 |
| 8.9～16 N·m 增益 | 拟合值 ±15% | reset | 每关节独立 |
| 16～20 N·m 增益 | 拟合值 ±20% | reset | 每关节独立 |
| 20～23.5 N·m 增益 | 拟合值 ±25% | reset | 稀疏区域更宽 |
| 驱动/制动增益差 | 实测值 ±20% | reset | 四象限分别采样 |
| 指令延迟 | 实测 P5～P95 再扩 1 个物理步 | reset | 系统公共项＋关节残差 |
| 延迟抖动 | 实测范围 | step | 小幅变化 |
| 时间常数 | 拟合值 ×0.7～1.5 | reset | 高力矩/高速区分开 |
| `armature` | 拟合值 ×0.8～1.2 | reset | 每关节独立 |
| 库仑摩擦 | 拟合值 ×0.6～1.4 | reset | 正反方向分开 |
| 黏性阻尼 | 拟合值 ×0.7～1.3 | reset | 正反方向分开 |
| `Kp` | 部署值 ×0.9～1.1 | reset | 部署稳定后可缩窄 |
| `Kd` | 部署值 ×0.9～1.1 | reset | 部署稳定后可缩窄 |
| 编码器偏置 | 实测电机间范围 | reset | 回合内固定 |
| 位置白噪声 | 实测标准差 | step | 每步采样 |
| 速度白噪声 | 实测标准差 | step | 每步采样 |
| 丢包概率/突发长度 | 实测分布 | step | 丢帧时保持上一帧 |
| 初始电机温度 | 真实启动温度范围 | reset | 每关节独立 |
| 热模型系数 | 拟合值 ×0.7～1.3 | reset | 电机/MOS 分开 |
| 过载增长/恢复参数 | 拟合值 ×0.7～1.5 | reset | 与峰值能力关联 |
| 母线电压 | 23.8～24.2 V | reset/慢变化 | 当前按性能表测试电压扩展；允许工作范围待确认 |

### 11.2 力矩和速度必须相关随机化

不得独立随机化峰值力矩和峰值速度后形成不合理矩形。每个环境随机化整条能力曲线：

\[
\tilde\tau_{\max,e}(\omega)
=s_{\tau,e}\,
\tau_{\max,\xi_e}
\left(\frac{\omega}{s_{\omega,e}}\right)
\]

\[
\tau_{\max,e}(\omega)
=\min\left[
\tilde\tau_{\max,e}(\omega),
\frac{P_{\mathrm{peak},e}}{\max(|\omega|,\epsilon)},
23.5
\right]
\]

其中：

- \(s_{\tau,e}\)：力矩能力缩放；
- \(s_{\omega,e}\)：速度能力缩放。
- \(\xi_e\)：峰值外推段在三次标称曲线与二次上界之间的形状混合系数；
- \(P_{\mathrm{peak},e}\)：本回合的峰值功率预算。

额定力矩、峰值力矩、额定速度和最高速度均从随机后的曲线派生。温度、过载和电压变化
继续作用于该曲线，而不是生成彼此矛盾的独立裁剪值。

### 11.3 随机化范围的数据化更新

当前 3 台电机阶段，对任一拟合参数 \(\theta\)：

\[
\theta_0=\operatorname{median}(\theta_1,\theta_2,\theta_3)
\]

\[
s_{\mathrm{motor}}
=1.4826\operatorname{MAD}(\theta_1,\theta_2,\theta_3)
\]

\[
\Delta\theta
=\max\left(2s_{\mathrm{motor}},2s_{\mathrm{fit}},0.15|\theta_0|\right)
\]

初始采样：

\[
\theta\sim U(\theta_0-\Delta\theta,\theta_0+\Delta\theta)
\]

获得 14 台数据后，使用：

\[
\theta_{\min}=P_5(\theta_j),\qquad
\theta_{\max}=P_{95}(\theta_j)
\]

并向两侧扩展约 10%，覆盖测量误差和未观察变化。

### 11.4 稀疏数据区域

每个力矩—速度网格单元计算样本数和拟合残差。规则如下：

- 样本充足、残差小：使用较窄随机化；
- 样本较少：放宽增益、时间常数和能力边界；
- 没有样本：不外推精确模型，使用官方硬边界和最保守随机化；
- 高力矩＋高速度区域：只允许真实任务中自然出现的范围。

### 11.5 UniLab 实现结构

任务配置示例：

```yaml
env:
  domain_rand:
    motor:
      enabled: true

      velocity_mapping_limit: 10.0
      torque_mapping_limit: 28.0
      actuator_force_range: [-23.5, 23.5]

      torque_speed_curve:
        base_lookup: torque_speed_capability
        torque_axis_scale_range: [0.9, 1.0]
        speed_axis_scale_range: [0.9773, 1.0]
        peak_region_shape_blend_range: [0.0, 1.0]
        peak_power_range: [63.0, 70.0]
        nominal_rated_torque: 8.9
        nominal_peak_torque: 23.5
        nominal_rated_velocity: 4.50294947
        nominal_maximum_velocity: 5.80952381
        # Rated/peak values are derived by scaling the complete curve.
        peak_region_torque_points: [17.86, 18, 19, 20, 21, 22, 23, 23.5]
        peak_region_speed_rpm_nominal: [30.60, 30.22, 27.36, 24.20, 20.73, 16.92, 12.75, 10.52]
        peak_region_speed_rpm_upper: [30.60, 30.30, 28.08, 25.77, 23.37, 20.87, 18.27, 16.94]

      delay_steps_range: [0, 2]  # 200 Hz 控制周期；含一周期测量扩展
      time_constant_scale_range: [0.7, 1.5]

      armature_scale_range: [0.8, 1.2]
      coulomb_scale_range: [0.6, 1.4]
      viscous_scale_range: [0.7, 1.3]

      kp_scale_range: [0.9, 1.1]
      kd_scale_range: [0.9, 1.1]

      position_bias_range: [-0.0025, 0.0025]  # 无机械零位夹具时的暂定范围
      position_noise_std: 0.000191
      velocity_noise_std: 0.002442

      voltage_range: [23.8, 24.2]
      initial_temperature_range: [32.0, 38.0]
```

实现职责：

- `dof_armature`、`Kp`、`Kd` 等后端支持字段通过 reset payload 设置；
- 力矩增益、延迟、摩擦、温度和过载状态由任务的 DR provider 采样并保存为每环境数组；
- 通过 `set_pre_step_control(...)` 注册 J4340 控制回调；
- reset 时恢复标称参数后再应用随机化，禁止参数跨回合累积；
- 温度和过载状态在回合内连续演化，reset 时按配置初始化。

---

## 12. 训练约束

为了允许短时峰值但避免策略长期依赖峰值，奖励中加入额定区和过载区不同强度的代价。

基础力矩代价：

\[
r_{\mathrm{torque}}
=-\lambda_\tau\sum_j(\tau_j/8.9)^2
\]

过额定力矩代价：

\[
r_{\mathrm{overload}}
=-\lambda_o\sum_j\max(|\tau_j|-8.9,0)^2
\]

过额定速度代价：

\[
r_{\mathrm{overspeed}}
=-\lambda_v\sum_j\max(|\dot q_j|-4.5029,0)^2
\]

温度和过载状态也应作为 critic 特权观测；是否提供给 actor 取决于部署端能否提供完全一致的信号。策略不应依赖部署端不存在的观测。

训练采用逐步扩展范围：

1. 先训练额定区：\(|\tau|\le8.9\)、\(|\omega|\le4.5029\)；
2. 加入高速低力矩区；
3. 加入低速短时高力矩区；
4. 最后加入完整随机化、温度、丢包和外部扰动。

---

## 13. 验证和验收

### 13.1 数据划分

- 三台空载电机：两台拟合、一台验证，轮换进行；
- 真机悬空数据：按完整实验段划分 70% 拟合、30% 验证；
- 站立、下蹲和行走数据：只用于最终整体验证；
- 14 台数据：采用留一关节或留一动作验证。

### 13.2 验证指标

必须至少报告：

\[
\mathrm{RMSE}_q,
\quad\mathrm{RMSE}_{\dot q},
\quad\mathrm{RMSE}_\tau,
\quad\mathrm{RMSE}_{T_{\mathrm{motor}}},
\quad\mathrm{RMSE}_{T_{\mathrm{MOS}}}
\]

同时报告：

- 阶跃上升时间和超调；
- 指令—反馈相位延迟；
- 稳态速度误差；
- 正反方向误差；
- 额定区和峰值区分别的误差；
- 驱动和制动四象限分别的误差；
- 14 电机总线延迟 P50/P95/P99；
- 长时间运行中的温度误差和轨迹漂移。

### 13.3 三组对照

必须保留以下对照实验：

1. 仅官方标称和官方曲线；
2. 官方数据＋空载/真机拟合；
3. 官方数据＋拟合＋域随机化。

只有第三组在未参与拟合的真实轨迹上稳定优于前两组，才能确认域随机化确实缩小了 sim-to-real 差距。

---

## 14. 安全要求

- 裸电机必须刚性固定，输出轴运动范围内不得有人或物体；
- 禁止用手、扳手、脚或临时夹具阻挡输出轴；
- 8.9 N·m 以上测试只能在结构确认、机器人可靠支撑和急停有效时进行；
- 每次只提高一个维度：先提高力矩或速度，不同时提高两者；
- 禁止以 GF43X40-10 主动测试超过 23.5 N·m 的物理输出；
- 温度反馈存在滞后，峰值脉冲之间必须留出恢复时间；
- 测试中出现异常振动、噪声、通信错误、过流、过载或温升异常时立即失能；
- MuJoCo 中的 28 N·m 编码上限和 23.5 N·m 电机物理上限都不能替代每关节结构安全上限；
- 真机运行时实际限制应取：

\[
\tau_{\mathrm{safe},j}
=\min(23.5,\tau_{\mathrm{motor},j},\tau_{\mathrm{structure},j})
\]

---

## 15. 实施顺序与交付物

| 阶段 | 工作 | 交付物 |
|---|---|---|
| 1 | 数字化官方性能和温升曲线 | 5 份 CSV 查表 |
| 2 | 完成三台电机静止、速度、力矩脉冲和温升测试 | 原始 CAN 数据集 |
| 3 | 拟合空载增益、噪声、延迟、时间常数、摩擦和惯量 | `gf43x40_10_empty_fit.json` |
| 4 | 实现 MuJoCo GF43X40-10 torque actuator | 执行器代码和单元测试 |
| 5 | 完成单关节仿真—空载实机对比 | 空载验证报告 |
| 6 | 机器人悬空采集 14 关节带载数据 | 真机辨识数据集 |
| 7 | 拟合每关节修正和力矩—速度曲面 | `gf43x40_10_joint_fit.json` |
| 8 | 采集 14 电机同步通信数据 | 总线延迟和丢包报告 |
| 9 | 实现 UniLab DR provider | 域随机化配置和测试 |
| 10 | 训练三组对照策略 | 官方/拟合/拟合+DR checkpoint |
| 11 | 完成站立、下蹲、行走验证 | 最终 sim-to-real 报告 |

本方案的最终产物是一套可测量、可拟合、可验证、可域随机化的 GF43X40-10 输出轴等效模型。当前 MIT 控制链使用 `Vmax=10 rad/s`、`Tmax=28 N·m` 作为编解码边界，但物理输出力矩按该型号规格限制为 23.5 N·m；温度、过载历史和关节结构安全边界继续作为能力曲线的额外约束。
