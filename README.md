# XDuck simulation: UniLab 1.3.1

本分支在原有工程目录中接入公司 UniLab `main@447e5ce`（1.3.1）的 Manager-Based 任务、UniSim MuJoCo 后端和 `uni_rl` PPO。三个工程各自拥有 `pyproject.toml`、`uv.lock`、任务代码、配置、模型资产及验证入口：

| 工程 | 任务 | 当前入口 | 旧权重 |
|---|---|---|---|
| [xl330](xl330/) | XL330 官方速度行走 | `cd xl330 && uv sync --locked && uv run xl330-train --algo ppo --task microduck_xl330_velocity_flat --sim mujoco` | 尚未验收兼容 |
| [walk](walk/) | V1.1.2 GF43X40 行走 | `cd walk && uv sync --locked && uv run python -m unilab.scripts.train_rsl_rl --config-path "$PWD/conf/ppo"` | 1975 checkpoint 严格加载，物理等价未验收 |
| [stand](stand/) | V1.1.2 GF43X40 走停/站立 | `cd stand && uv sync --locked && uv run xduck-stand-train` | 599 checkpoint 严格加载，物理等价未验收 |

三个工程固定相同的公司 UniLab 提交，以及 `unisim-core==1.7.3`、`mjbatch-uni==0.2.1`、`torch==2.8.0`、`rsl-rl-lib==5.0.1`、`tensordict==0.11.0`。`stand` 以本仓库相邻的 `walk` 包为本地依赖，两者共用 GF 电机实现。

`walk/checkpoints/`、`stand/checkpoints/` 和 `policy/` 保留原先选出的模型、ONNX、运行配置及哈希；`evidence/` 保留旧仿真验收报告。旧框架完整源码仍保留在 `walk/unilab/` 和 `stand/unilab/`，当前训练入口使用上述新版工程。原 DM4310 行走、坐站、起身任务没有恢复。

**运行通过不等于旧策略等价。** 新版 BAM/GF 动作已接到物理子步，但旧版足部传感器每子步更新、接触时长、落脚奖励和部分 DR/课程语义尚未达到逐项数值对齐。XL330 还需核对 Euler 与旧 implicitfast 的差异。三个模型不能据此直接认定真机可用；具体边界见各工程 README。
