# A2.6-F703：训练与复现

本目录是 F703 阶段快照的入口说明，使用 Apple Silicon Mac / MLX PPO。包含当时训练源码、COM35模型及网格、GF43X40电机实现、完整训练配置、父权重control502、选定F703及trainer记录。源码按原训练指纹保留；无需另一个私人仓库。

F703已通过原仿真的前进/存活检查。本次真机存在偏航记录和停步跌倒保护，尚未完成真机走停验收；关键记录见 `evidence/hardware/`。

## 使用现有 UniLab 环境

进入本目录，激活同事已有的 UniLab Python 环境后运行：

```bash
uv run --active --no-sync python f703.py check
uv run --active --no-sync python f703.py evaluate
uv run --active --no-sync python f703.py train --mode continue
```

入口使用本目录冻结的源码，复用已安装依赖，不自动安装或修改现有环境。依赖版本差异只提示；实际缺少的包按需安装。环境尚未激活时，也可以用 `uv run --project /path/to/UniLab --no-sync python "$PWD/f703.py" check`。

- `train --mode reproduce`：从父候选control502，以原seed293101和原配置重跑最后201轮。
- `train --mode continue`：从选定F703继续201轮；可用 `--iterations 20` 指定轮数。
- `evaluate --video`：8环境、站5秒/前进8秒/停5秒，额外保存一个环境的视频。
- `evaluate --checkpoint <新权重.safetensors>`：检查新训练权重。默认评估已导出的F703 ONNX。

输出均在 ignored 的 `outputs/` 中。提供的是最后一段训练的可复现起点，不是从随机网络重新训练的承诺；不同芯片/依赖版本的训练结果不保证逐字节相同。

## 配置与电机模型

| 项目 | F703设置 / 文件 |
|---|---|
| 唯一策略 | `checkpoints/f703/model_200.safetensors`；SHA256以 `43564d0509d8` 开头 |
| 训练配置 | `configs/f703_reproduce.yaml`；continue只改起始权重和输出目录 |
| PPO | 128环境×24步，seed293101，学习率5e-6，5 epochs×4 minibatches，actor/critic均61维，动作14维 |
| 机械 | V1.1.2、上半身等效COM后移35mm、HOME髋/踝±0.384rad |
| 控制 | Kp30/Kd3，物理2ms、策略20ms，无新增动作EMA/低通 |
| 电机 | GF43X40扭矩—速度/功率包络、非对称平滑摩擦、等效惯量、增益/延迟/丢包/反馈误差 |
| 响应时间 | 训练随机1–10ms；默认18秒回放固定10ms |
| 完整随机化 | [DR_inventory.md](evidence/training/DR_inventory.md)及同目录JSON，含范围、关闭项和实现语义 |

电机常量和包络在 `src/unilab/actuators/gf43x.py`；控制与随机化实现在 `src/unilab/envs/locomotion/microduck_enlarged*/` 继承链；模型在 `src/unilab/assets/robots/microduck_enlarged_112/`。这些源文件已与原训练指纹核对，`check` 会检查漂移。V111资产仅保留作既有回归测试的对照。

保留一项历史实现细节：`scripts/train_mlx_ppo.py` 的 `resume_trainer_state` 实际只恢复学习率，未恢复Adam动量；本快照保持原行为，trainer文件完整保存供核查。不要将它当成已实现的完整优化器/环境状态续接。

## 建议环境与验证

参考版本为 Python3.13.13、MLX/MLX-Metal0.31.1、NumPy2.4.4、mujoco-uni3.8.0、ONNX Runtime1.24.4。完整原环境版本见 `evidence/source_identity.json`。若现有环境可用，不必重装；需要独立环境时可选：

```bash
uv sync --locked --extra mujoco
uv run --no-sync python f703.py check
uv run --no-sync pytest -q tests/envs/locomotion/microduck_enlarged_112 tests/actuators/test_gf43x.py tests/algos/test_mlx_ppo.py tests/algos/test_mlx_policy_consistency.py
```

原mujoco-uni3.8.0的PyPI索引已下架；可选安装配置使用原Python3.13/macOS arm64 wheel直链及锁文件哈希，不把安装包提交进工程。普通MuJoCo与此版的批量接口不能直接互换。

本机冷启动重跑最后201轮，产出权重SHA256与原F703完全相同；119项回归通过，18秒回放轨迹一致，现有环境续训2轮通过。详见 `evidence/validation.json`。原训练筛选证据见 `evidence/training/`；其中个人绝对路径仅为历史记录，当前运行入口不依赖它们。
