"""Restricted tensor-only UniLab / Isaac Lab RSL-RL PPO converter.
Embedded in robotd; Python receives only server-generated paths and a known activation.
"""
import sys


def convert(source, target, activation, normalizer_epsilon="0.01"):
    import re
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    from torch import nn

    torch.set_num_threads(1)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("需要 UniLab / Isaac Lab RSL-RL PPO 权重 checkpoint")
    if "actor_state_dict" in checkpoint:
        raw = checkpoint["actor_state_dict"]
        if not isinstance(raw, dict):
            raise ValueError("actor_state_dict 不是权重字典")
        state = {}
        for key, value in raw.items():
            if key.startswith("mlp."):
                state["actor." + key[4:]] = value
            elif key in ("distribution.std_param", "distribution.log_std_param"):
                state["std"] = value
            else:
                state[key] = value  # Strict validation below rejects unknown modules.
    elif "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        raise ValueError("未找到 model_state_dict / actor_state_dict；支持 UniLab 和 Isaac Lab 的 RSL-RL PPO，其他后端请从训练框架导出 ONNX")
    if not isinstance(state, dict):
        raise ValueError("策略权重不是字典")
    normalizer = checkpoint.get("obs_norm_state_dict")
    for prefix in ("obs_normalizer.", "actor_obs_normalizer."):
        embedded = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if embedded:
            if normalizer:
                raise ValueError("同时发现多个观测归一化器，无法确定推理顺序")
            normalizer = embedded
            state = {k:v for k,v in state.items() if not k.startswith(prefix)}
    # Never discard an encoder, recurrent state, or normalizer silently.
    allowed = re.compile(r"^(actor|critic)\.\d+\.(weight|bias)$|^(std|log_std)$")
    unsupported = [k for k in state if not allowed.fullmatch(k)]
    if unsupported:
        raise ValueError("不支持的策略结构（可能含 RNN、编码器或归一化）: " + ", ".join(unsupported[:12]) + "；请使用训练框架导出完整 ONNX")
    for key in checkpoint:
        if ("norm" in key.lower()) and key not in ("obs_norm_state_dict", "critic_obs_norm_state_dict"):
            raise ValueError("无法识别的观测归一化字段: " + key + "；请从训练框架导出 ONNX")
    indices = sorted(int(k.split(".")[1]) for k in state if re.fullmatch(r"actor\.\d+\.weight", k))
    if len(indices) < 2 or indices != list(range(0, 2 * len(indices), 2)):
        raise ValueError("只支持交替 Linear/激活的 ActorCritic MLP；请从训练框架导出 ONNX")
    expected_keys = {f"actor.{i}.{suffix}" for i in indices for suffix in ("weight", "bias")}
    if {k for k in state if k.startswith("actor.")} != expected_keys:
        raise ValueError("Actor 权重/偏置不完整或存在额外层，不能安全还原网络")
    acts = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "selu": nn.SELU, "leaky_relu": nn.LeakyReLU}
    if activation not in acts:
        raise ValueError("未知激活函数；必须与训练配置 algo.policy.activation 一致")
    layers = []
    width = 61
    for i in indices:
        w, b = state[f"actor.{i}.weight"], state.get(f"actor.{i}.bias")
        if not isinstance(w, torch.Tensor) or w.ndim != 2 or w.shape[1] != width or b is None or tuple(b.shape) != (w.shape[0],):
            raise ValueError(f"actor.{i} 维度不兼容；期望输入宽度 {width}（机器人观测为 61）")
        if not torch.isfinite(w).all() or not torch.isfinite(b).all():
            raise ValueError(f"actor.{i} 权重含 NaN/Inf")
        layer = nn.Linear(width, w.shape[0])
        layer.load_state_dict({"weight": w.float(), "bias": b.float()})
        layers.append(layer)
        width = w.shape[0]
        if i != indices[-1]:
            layers.append(acts[activation]())
    if width != 14:
        raise ValueError(f"动作维度不匹配：实际 {width}，机器人要求 14")
    actor = nn.Sequential(*layers).eval()
    if normalizer:
        if not isinstance(normalizer, dict) or set(normalizer) != {"_mean", "_var", "_std", "count"}:
            raise ValueError("无法识别 RSL-RL 观测归一化结构；请从训练框架导出完整 ONNX")
        mean, std = normalizer["_mean"].float(), normalizer["_std"].float()
        if tuple(mean.shape) != (1, 61) or std.shape != mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any():
            raise ValueError("观测归一化参数维度或数值不合法")
        epsilon = float(normalizer_epsilon)
        if not 1e-12 <= epsilon <= 1:
            raise ValueError("归一化 epsilon 必须介于 1e-12 和 1")
        class NormalizedActor(nn.Module):
            def __init__(self):
                super().__init__()
                self.actor = actor
                self.register_buffer("mean", mean)
                self.register_buffer("std", std)
            def forward(self, obs):
                return self.actor((obs - self.mean) / (self.std + epsilon))
        actor = NormalizedActor().eval()
    with torch.inference_mode():
        torch.onnx.export(actor, (torch.zeros(1, 61),), target, input_names=["obs"], output_names=["actions"], opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(target))
    session = ort.InferenceSession(target, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    for obs in [np.zeros((1, 61), dtype=np.float32), *[rng.normal(size=(1, 61)).astype(np.float32) for _ in range(8)]]:
        with torch.inference_mode():
            expected = actor(torch.from_numpy(obs)).numpy()
        actual = session.run(None, {"obs": obs})[0]
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError("试推理产生 NaN/Inf")
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5, err_msg="PyTorch 与 ONNX 数值不一致")
    print("PPO 权重校验、ONNX 转换、9 组数值一致性检查通过")


if __name__ == "__main__":
    try:
        convert(*sys.argv[1:])
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
