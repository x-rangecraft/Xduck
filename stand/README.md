# GF43X40 站立与走停：UniLab 1.3.1

本目录是当前有效的 Manager-Based 站立工程；相邻的 `../walk` 提供同一 GF 电机动作项与 V1.1.2 场景。`checkpoints/model_599_stand_push02.pt`、原 `run_config.json`、ONNX 和历史验收报告保持原件，`src/xduck_stand_latest/` 为新版任务实现。混合交接状态库在包内 `assets/handoff/mixed70stand30walk.npz`。

## 使用

在本目录执行（需要访问公司私有 UniLab 仓库）：

```bash
uv sync --locked
uv run xduck-stand-train algo.num_envs=2 algo.num_steps_per_env=2 \
  algo.max_iterations=1 training.no_play=true
uv run pytest -q tests
```

默认训练配置在 `conf/ppo/task/xduck_gf43x40_walk_stop_flat/mujoco.yaml`。任务使用 61 维 actor、76 维 critic、14 维动作；13 维指令恒为零。reset 从站立/行走混合状态库取机械状态，按公开实体事务写入，并将入站的上一个原始动作恢复到动作历史。站立持续 1 秒的指标另计，不与跌倒判定混为一谈。

原 599 checkpoint 已在新版 runner 下严格加载并完成一步推理，普通 Gaussian 与自己的 normalizer 保持。该结果只证明网络接口兼容。旧版与新版的接触传感器子步时序、GF 电机求解、随机化和课程仍需并列评估；不要把历史 `evidence/REPORT.md` 中的零跌倒统计当成新版训练结果。新版冒烟运行仅证明训练链路，不代表收敛。

走→站评估需要相邻 `../walk/checkpoints/model_1975_strong_bounded.pt`，切换时应保持物理、电机、通信及历史动作连续。旧版回放、原始配置与模型哈希仍保存在本目录 `evidence/`、`CHECKSUMS.sha256`。
