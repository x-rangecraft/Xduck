# 策略模型导入与 ONNX 回滚

运行状态右侧的「策略模型 · 导入与回滚」面板从 robotd 读取当前模式的已配置策略：行走、站立、坐站、拾取、左右踢、翻滚（未配置的槽位不显示；轮式模式沿用它自己的文件）。

1. 点击「放松」，等待面板确认全部配置电机失能。
2. 选择要替换的策略和本地 `.pt` / `.pth` / `.onnx` 文件，最大 64 MiB。
3. PPO 选择与训练相同的激活函数；如使用观测归一化，ε 也须与训练相同，标准 RSL-RL 默认 0.01。直接 ONNX 导入无需这些转换参数。
4. 点击「校验 → 转 ONNX → 替换」。页面显示上传、转换、校验、替换各阶段与错误详情。成功后新策略已加载，需手动初始化/开启策略。
5. 每个槽位保存最近两个旧 ONNX，在历史条目点击「回滚」。回滚会重新做目标运行时检查，并将被替换的当前版本放回历史。

## 支持范围

- **UniLab RSL-RL PPO `.pt`**：支持本地 UniLab 当前使用的 `actor_state_dict` / `mlp.*`（RSL-RL 5），以及旧版 `model_state_dict` / `actor.*`。
- **Isaac Sim / Isaac Lab 的 RSL-RL PPO `.pt`**：支持相同的新旧布局。标准确定性 MLP actor、固定动作标准差的 Gaussian policy；忽略训练专用 critic、优化器、探索方差。
- 支持标准 RSL-RL `EmpiricalNormalization`（外置 `obs_norm_state_dict` 或 actor 内的 `obs_normalizer.*` / `actor_obs_normalizer.*`），将 mean/std 和页面指定 ε 一并导出。
- 激活函数支持 ELU、ReLU、Tanh、SELU、LeakyReLU。权重本身不记录激活函数及 ε，因此页面参数必须与训练配置一致；默认值对应标准配置。
- RNN/LSTM、HIM、HORA、额外编码器、未知归一化器、其他训练后端（例如 RL-Games、SKRL、MLX）不能靠猜测还原；页面明确拒绝并提示从原框架导出完整 ONNX。框架名称相同不代表任意网络都符合本机机器人接口。
- ONNX 必须为单个自包含文件，只有一个 float32 输入 `obs`，形状 `[1,61]`（允许动态 batch），一个 float32 `[1,14]` 输出。需与本机观测语义、关节顺序、动作缩放和策略用途一致。张量/试推理检查无法证明实际运动稳定性。

## 校验与安全边界

上传采用现有 WebRTC → Unix socket JSON-RPC，每块最多 8 KiB，检查任务 token、顺序偏移、声明大小。网页不能传服务端文件路径；文件名仅作显示用途。PPO 使用 `torch.load(weights_only=True)`，不执行 checkpoint 中任意自定义 Python 类。

转换在工作线程启动的独立 Python 进程中进行，120 秒超时后终止。检查 actor 层连接、输入/输出维度、有限权重、ONNX graph，并在 9 组输入上比较 PyTorch/ONNX 输出（rtol=1e-4, atol=1e-5）。随后由 robotd 自己的 ONNX Runtime 再加载、验证并做三组有限值试推理，保证转换机器能导出不等于目标运行时能加载的问题也可见。直接 ONNX 走目标运行时校验，不要求安装 PyTorch。

开始导入、提交上传、发起回滚时后端都会检查新鲜的放松反馈。准备成功后只排队请求，由唯一拥有电机总线的控制循环提交：必须仍是相同模式/槽位、Limp、策略未开启，且本周期新鲜 STM32 样本中所有已配置电机在线、无故障、未使能、反馈不超过 500 ms。仅停止、仅关闭策略、断开连接、没有新样本、转换期间初始化/使能都不能绕过该检查。条件改变时拒绝本次替换，须重新导入/回滚，不会以后自动补执行。替换本身不发使能或运动命令。

ONNX 存放在 `/var/lib/robotd/policies/`，发布包内文件不修改。以槽位与原文件名为键持久化覆盖关系；首次替换复制原始 ONNX。先落盘候选/备份，再 fsync + rename 原子提交清单，最后交换已经预热的内存 session。失败前原清单/原 session 保持不变；重启或切换模式时读取持久覆盖。清单损坏时拒绝加载和修改并显示原因，不能静默回退。正常提交后删除退出两条历史范围的受管理文件。

## 部署与验证

该面板及转换脚本嵌入 Rust 二进制，更新 `robotd` / `mediad` 后生效。首次需要在**已授权的部署**中安装 Python 转换依赖：

```sh
sudo sh scripts/setup-model-import.sh
```

默认建立 `/var/lib/robotd/model-python/`，robotd 自动识别其 Python；也可用 `ROBOT_MODEL_PYTHON` 指向已经部署的解释器。安装脚本不重启服务，不访问电机。需要 Python 3.11+、venv/pip 和足够磁盘空间；版本固定于 `scripts/model-import-requirements.txt`。缺失依赖时页面显示模块名称和转换失败原因，原策略保持不变。

本地回归：`60-tools/test-model-import.py`（torch/onnx/onnxruntime/numpy，RSL-RL 用于对照其真实归一化实现）、`60-tools/test-web-models.cjs`。Rust `models::tests` 覆盖上传边界、清单原子性、两条历史、重启和安全拒绝；带 ONNX Runtime 的主机可设 `ORT_DYLIB_PATH` 后运行 `--include-ignored`，验证实际 ONNX session 提交/回滚，无电机访问。
