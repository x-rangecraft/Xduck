# XDuck 1.1.2 Walk／Stand API-v2 交付

主文件：

- `xc_walking.onnx`：Walk 槽位模型，来自 `model_1975_strong_bounded.pt`。
- `xc_standing.onnx`：Stand 槽位模型，来自 `model_399.pt`。
- `xc_policy.py`：Walk/Stand 共享的 API-v2 消费文件。

## 网页上传方式

API v2 的每次上传是一组不可拆分的 `policy.py + 一个 model.onnx`，不是把三个文件作为一个包上传。

1. 选择 `walk` 槽位，同时选择：
   - 策略代码：`xc_policy.py`
   - 模型：`xc_walking.onnx`
2. Walk 成功后选择 `stand` 槽位，同时选择：
   - 策略代码：同一个 `xc_policy.py`
   - 模型：`xc_standing.onnx`

robotd 会预加载当前 Walk/Stand 两个 ONNX，并使用一个共享 `Policy` 实例联合预热；切换时不会 reset，上一动作和速度历史保持连续。

`xc_policy.py` 不读取本地 ONNX 文件，也不创建 `InferenceSession`。模型由 robotd 管理，因此不会再寻找 `/var/lib/robotd/policies/xc_walking.onnx`。

## API 和模型接口

消费文件严格实现 API v2：

```python
class Policy:
    def describe(self): ...
    def reset(self, robot_info, first_frame): ...
    def preprocess(self, frame, feedback): ...
    def postprocess(self, outputs, frame): ...
```

两个 ONNX 均为：

- 输入：`obs`，float32 `[1,61]`
- 输出：`actions`，float32 `[1,14]`
- Walk ONNX 已嵌入 `UnitSlopeBoundedGaussianDistribution` 的不对称有界均值变换，输出为变换后的未缩放 raw action；不是裸 MLP
- Stand ONNX 输出普通 Gaussian 的 deterministic mean raw action
- checkpoint 观测 normalizer 已嵌入 ONNX

动作只在 `postprocess()` 中缩放一次：

```text
q_des = clip(REFERENCE + raw_action * 0.12, TARGET_LOW, TARGET_HIGH)
```

## 仿真对齐

- 行走 `GAIT_REFERENCE` 与停步 `STAND` 数值完全相同。
- 显式转换 robotd 平台顺序“左腿、右腿、头部、mouth”到训练顺序“左腿、头部、右腿”。
- 逐关节 KP：`[90,90,75,100,90,75,90,75,90,90,90,75,100,90]`。
- 逐关节 KD：`[2.5,2.5,2.5,3,3,2.5,2.5,2.5,2.5,2.5,2.5,2.5,3,3]`。
- 50 Hz；关节速度观测保留训练中的一帧 20 ms 延迟。
- 两份训练均未使用动作 EMA 或目标低通，因此不添加动作滤波。
- Walk 编码平台 twist/head/body 命令；Stand 强制完整 13D command 为零。
- `last_action` 保存未缩放、未裁剪 raw action，在 Walk/Stand 切换时连续。
- 1.1.2 原生关节方向；外部不得再次反号、缩放或增加参考姿态。

## 验证

- 官方资料包的 `validate.py` 已分别通过 `xc_policy.py + xc_walking.onnx` 和 `xc_policy.py + xc_standing.onnx`。
- 官方嵌入版 `_policy_worker.py` 已同时预加载两个 ONNX，并按 `walk→walk→stand→stand→walk` 使用同一实例完成联合预热。
- 两个 ONNX 各使用 512 组输入与 PyTorch actor 数值比对。
- 两个保存的仿真环境 owner 已验证参考姿态、限位、KP/KD、动作缩放及目标计算误差为 0。

详细结果见所有 `xc_*verification.json` 与 `xc_manifest.json`。
