# XDuck Walk v3 API-v2 交付

主文件：

- `xc_walking_v3.onnx`：由 Ubuntu5090 上的 `model_1279.pt` 导出。
- `xc_walking_v3_policy.py`：Walk/Stand 共享的 API-v2 v3 消费文件。

## 网页上传

选择 `walk` 槽位，同时选择：

- 策略代码：`xc_walking_v3_policy.py`
- 模型：`xc_walking_v3.onnx`

Stand 侧保持现有 model_599 ONNX。robotd 会把新的共享 consumer 与当前 Stand 模型联合预热后原子替换 Walk。

## v3 契约

- ONNX 输入：`obs` float32 `[1,61]`
- ONNX 输出：`actions` float32 `[1,14]`，未缩放 raw action
- checkpoint 观测 normalizer 已嵌入 ONNX
- 参考姿态：训练 scene.xml 的 `STAND`
- 控制周期：20 ms
- 关节速度：一帧 20 ms 延迟
- 外部动作 EMA：无
- 关节方向与顺序：XDuck V1.1.2 原生训练顺序

Walk v3 使用逐关节物理尺度：

```text
[0.12, 0.0366667, 0.12, 0.12, 0.12,
 0.0966667, 0.12, 0.12, 0.12,
 0.12, 0.0366667, 0.12, 0.12, 0.12]
```

其中左右 hip_roll 为 `0.0366667 rad`，neck_pitch 为 `0.0966667 rad`，其余为 `0.12 rad`。

配对的 Stand 599 仍使用统一 `0.12 rad`。共享 consumer 在切换时保存实际物理关节偏移，并按目标模型尺度还原其 raw-action 历史，避免直接混用两种动作单位。

KP：

```text
[90,90,75,100,90,75,90,75,90,90,90,75,100,90]
```

KD：

```text
[2.5,2.5,2.5,3,3,2.5,2.5,2.5,2.5,2.5,2.5,2.5,3,3]
```

## 验证

- checkpoint 严格加载为 61D/14D Gaussian actor。
- 512 组 PyTorch/ONNX 对照结果见 `xc_walking_v3_onnx_verification.json`。
- Walk/Stand 599 联合切换、逐关节尺度、历史单位换算、命令编码、延迟、限位和增益检查见 `xc_walking_v3_api_verification.json`。
- 这些是接口与数值验证，不是实机稳定性验证。
