# GF43X40-10 最终版 UniLab 迁移包

版本：`shared_fixed_gain_kd_band_v8_torque_proxy`。
运行代码、JSON和冻结测试均从当前工作区原样复制；运行依赖只有NumPy，运行自检另需pytest。未包含其他机器人任务、模型网格或训练框架副本。

## 复制到另一套UniLab

将本包的 `src/unilab/actuators/gf43x40/` 整个目录复制到目标UniLab的同名路径。保留目标仓库原有的 `unilab/__init__.py` 和 `actuators/__init__.py`。若目标已有gf43x40目录，先备份，避免混用旧参数和新代码。

```bash
# 在本迁移包根目录执行，修改目标路径：
TARGET_UNILAB=/path/to/another/unilab
mkdir -p "$TARGET_UNILAB/src/unilab/actuators"
cp -R src/unilab/actuators/gf43x40 "$TARGET_UNILAB/src/unilab/actuators/"
```

运行时文件：`__init__.py`、`actuator.py`、`communication.py`、`robot.py`、`torque_observer.py`、`calibration.json`、`timing_profile.json`及可选的示例文件`example_communication.py`（前7个必需，示例不参与运行）。`torque_observer.py`虽为旧版兼容代码，仍被模块导入，不能漏拷。JSON必须与代码同目录。

## 独立自检

在迁移包根目录执行：

```bash
uv run --with numpy --with pytest python -I verify_port.py
```

已有依赖时可用 `uv run --no-sync python -I verify_port.py`。脚本核验文件哈希，并只加载本包执行器运行附带测试，不依赖原MicroDuck目录或完整UniLab安装。`tests/actuators/`可一并复制到目标仓库，随后从目标UniLab正常执行：

```bash
uv run --no-sync pytest tests/actuators/test_gf43x40_port.py tests/actuators/test_gf43x40_jitter.py -q
```

## 整机接入

```python
import numpy as np
from unilab.actuators.gf43x40 import GF43X40Parameters, GF43X40RobotMotor

params = GF43X40Parameters.from_bundle()
motor = GF43X40RobotMotor(
    num_envs, num_motors,
    kp=60.0, kd=2.0,  # 示例任务增益；也接受每关节数组
    parameters=params,
    jitter_mode="measured_20260918",
)

# 每个1 ms物理子步，q、v、q_target形状均为(num_envs, num_motors)：
tau = motor.begin_substep(q_target, q, v)
# 将tau施加给物理后端并积分一次；然后读取q_after、v_after。
motor.finish_substep(q_after, v_after)

# 供策略观测使用，保留20 ms通信保持和量化：
q_obs = motor.quantize_position_feedback(q_after)
v_obs = motor.quantize_velocity_feedback(v_after)

# 环境重置时：
motor.reset(np.asarray(env_ids, dtype=np.intp))
```

上述是嵌入已有物理循环的代码片段；q_after/v_after必须来自真实后端积分。每步严格begin→积分→finish，各执行一次。策略每20 ms更新q_target，中间保持。通用执行器支持完整MIT五项输入；这个整机适配器使用位置目标，速度目标和前馈为0。

后端冷路径设置：

- 物理步长0.001 s，策略周期0.02 s，即每策略步20个物理子步。
- 受控关节armature设为 `params.physics["armature"]`（0.25429975910988406）；浮动基座自由度不要套用电机关节惯量。
- 原关节damping/frictionloss清零，由组件统一计算摩擦。
- 推荐单位增益力矩执行器，直接施加tau；机械force/ctrl范围至少覆盖±148.5977380931125，不要沿用CAN的±28截断机械力矩。
- 若后端只能使用位置执行器，将其设为Kp=1、Kd=0，传入`q+tau`。force范围仍用机械限幅，ctrl范围还要覆盖位置项，不能把tau直接当位置目标。
- 关节状态、目标、增益、力矩的排列顺序须一致。关节数可变，代码不依赖MicroDuck的14轴资产。
- 目标任务需显式选择此执行器并注册到自身物理子步回调。复制执行器目录不会自动替换目标任务中的旧DM模型。

本包的整机适配器使用`smooth_coupled_approx`；裸轴默认`isolated_1d`，后者需要实际单轴广义质量。当前共32项标定配置；21项机械候选未包含。

## 包内文件

- `src/unilab/actuators/gf43x40/`：最终运行代码、校准包、抖动分布和通信示例。
- `tests/actuators/`：原有执行器/通信测试与6份冻结JSON数据。
- `verify_port.py`：可独立执行的迁移自检入口。
- `MANIFEST.json`：版本、源参数哈希及逐文件SHA-256。
- `MODEL_DETAILS_CN.md`：模型结构说明；其中指向原工作区的历史链接请以本包`reference/`为准。
- `reference/PARAMETERS_CN.md`、`reference/RESULTS.md`：最终版完整参数和已有验证报告；报告的原始大规模实验数据未打包。

机械尺度是等效仿真标定值；力矩反馈为机械力矩乘0.1198991043896492的粗略读数。本包迁移测试验证实现与既有轨迹，不代替目标机器人整机验证。
