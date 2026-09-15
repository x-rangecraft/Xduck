"""Isolated policy worker; a locomotion pair shares one instance across two ONNX sessions."""
import contextlib
import importlib.util
import json
import math
import sys


JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "mouth",
]
CONTROLLED_JOINTS = [name for name in JOINT_NAMES if name != "mouth"]


def emit(value):
    sys.stdout.write(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def fail(error):
    emit({"ok": False, "error": f"{type(error).__name__}: {error}"})


def load_policy(path):
    spec = importlib.util.spec_from_file_location("xduck_uploaded_policy", path)
    if spec is None or spec.loader is None:
        raise ValueError("无法加载 policy.py")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
    cls = getattr(module, "Policy", None)
    if cls is None:
        raise ValueError("policy.py 必须定义 Policy 类")
    return cls


def tensor_type(type_name):
    names = {
        "tensor(float)": ("float32", "float32"),
        "tensor(double)": ("float64", "float64"),
        "tensor(int64)": ("int64", "int64"),
        "tensor(int32)": ("int32", "int32"),
        "tensor(bool)": ("bool", "bool"),
    }
    if type_name not in names:
        raise ValueError(f"暂不支持 ONNX 类型 {type_name}")
    return names[type_name]


def validate_contract(contract, session):
    if not isinstance(contract, dict) or contract.get("api_version") != 2:
        raise ValueError("describe() 必须返回 api_version=2；v1 不再受支持")
    if not isinstance(contract.get("period_us"), int) or not 20_000 <= contract["period_us"] <= 1_000_000:
        raise ValueError("period_us 必须是 20000 至 1000000 的整数；平台控制周期为 20 ms")
    controlled = contract.get("controlled_joints")
    if not isinstance(controlled, list) or len(controlled) != len(set(controlled)):
        raise ValueError("controlled_joints 必须是无重复名称列表")
    if set(controlled) != set(CONTROLLED_JOINTS):
        raise ValueError("第一版策略必须完整控制除 mouth 外的 14 个关节")
    for key, ports in (("inputs", session.get_inputs()), ("outputs", session.get_outputs())):
        declared = contract.get(key)
        if not isinstance(declared, dict) or set(declared) != {port.name for port in ports}:
            raise ValueError(f"describe().{key} 名称必须与 ONNX 完全一致")
        for port in ports:
            item = declared[port.name]
            dtype, _ = tensor_type(port.type)
            if not isinstance(item, dict) or item.get("dtype") != dtype:
                raise ValueError(f"{key}.{port.name} dtype 应为 {dtype}")
            shape = item.get("shape")
            if not isinstance(shape, list) or not shape or any(type(v) is not int or v <= 0 for v in shape):
                raise ValueError(f"{key}.{port.name}.shape 必须是具体正整数列表")
            if len(shape) != len(port.shape):
                raise ValueError(f"{key}.{port.name} rank 不匹配：声明 {shape}，模型 {port.shape}")
            for actual, model_dim in zip(shape, port.shape):
                if isinstance(model_dim, int) and model_dim > 0 and model_dim != actual:
                    raise ValueError(f"{key}.{port.name} shape 不匹配：声明 {shape}，模型 {port.shape}")
    return contract


def validate_frame(frame):
    if not isinstance(frame, dict):
        raise ValueError("frame 必须是字典")
    command = frame.get("command")
    if not isinstance(command, dict) or set(command) != {"twist", "head", "body"}:
        raise ValueError("frame.command 必须恰好包含 twist/head/body 基础命令")
    for name, width in (("twist", 3), ("head", 4), ("body", 3)):
        values = command[name]
        if not isinstance(values, list) or len(values) != width:
            raise ValueError(f"frame.command.{name} 必须是 {width} 维列表")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool)
               or not math.isfinite(value) for value in values):
            raise ValueError(f"frame.command.{name} 必须只包含有限数值")
    context = frame.get("policy_context")
    if not isinstance(context, dict) or set(context) != {"action", "phase", "body_active"}:
        raise ValueError("frame.policy_context 必须恰好包含 action/phase/body_active")
    actions = {"walk", "stand", "ground_pick", "kick_left", "kick_right", "roulade", "sit", "rise"}
    if context["action"] not in actions:
        raise ValueError("frame.policy_context.action 未知")
    if not isinstance(context["body_active"], bool):
        raise ValueError("frame.policy_context.body_active 必须是布尔值")
    phase = context["phase"]
    if context["action"] == "ground_pick":
        if not isinstance(phase, (int, float)) or isinstance(phase, bool) or not math.isfinite(phase):
            raise ValueError("ground_pick 必须提供有限 phase")
    elif phase is not None:
        raise ValueError("只有 ground_pick 可以提供 phase")


def make_instance(cls, expected_contract, frame):
    validate_frame(frame)
    with contextlib.redirect_stdout(sys.stderr):
        instance = cls()
        contract = instance.describe()
    if contract != expected_contract:
        raise ValueError("新 Policy 实例的 describe() 与预加载契约不一致")
    robot_info = {"joint_names": JOINT_NAMES, "controlled_joints": CONTROLLED_JOINTS}
    with contextlib.redirect_stdout(sys.stderr):
        instance.reset(robot_info, frame)
    return instance


def as_json(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(v) for v in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("模型反馈包含 NaN/Inf")
    return value


def run_step(instance, contract, session, frame, feedback):
    import numpy as np

    validate_frame(frame)

    with contextlib.redirect_stdout(sys.stderr):
        inputs = instance.preprocess(frame, feedback)
    if isinstance(inputs, dict) and inputs.get("ready") is False:
        reason = inputs.get("reason", "策略尚未就绪")
        return {"ok": True, "ready": False, "reason": str(reason)}, feedback
    if isinstance(inputs, dict) and "inputs" in inputs and set(inputs).issubset({"ready", "inputs", "reason"}):
        inputs = inputs["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != set(contract["inputs"]):
        raise ValueError("preprocess() 返回的输入名称与 describe()/ONNX 不一致")
    feed = {}
    for port in session.get_inputs():
        _, numpy_dtype = tensor_type(port.type)
        array = np.asarray(inputs[port.name], dtype=numpy_dtype)
        wanted = tuple(contract["inputs"][port.name]["shape"])
        if array.shape != wanted:
            raise ValueError(f"输入 {port.name} 要求 {wanted}，实际 {array.shape}")
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError(f"输入 {port.name} 含 NaN/Inf")
        feed[port.name] = array
    values = session.run(None, feed)
    outputs = {}
    for port, value in zip(session.get_outputs(), values):
        wanted = tuple(contract["outputs"][port.name]["shape"])
        if value.shape != wanted:
            raise ValueError(f"输出 {port.name} 要求 {wanted}，实际 {value.shape}")
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"输出 {port.name} 含 NaN/Inf")
        outputs[port.name] = value
    with contextlib.redirect_stdout(sys.stderr):
        targets = instance.postprocess(outputs, frame)
    if isinstance(targets, dict) and "targets" in targets and set(targets).issubset({"targets"}):
        targets = targets["targets"]
    if not isinstance(targets, dict) or set(targets) != set(CONTROLLED_JOINTS):
        raise ValueError("postprocess() 必须完整返回 14 个受控关节，且不能包含 mouth/未知关节")
    clean = {}
    required = {"position", "velocity", "torque_ff", "kp", "kd"}
    for joint in CONTROLLED_JOINTS:
        target = targets[joint]
        if not isinstance(target, dict) or set(target) != required:
            raise ValueError(f"关节 {joint} 必须恰好包含 position/velocity/torque_ff/kp/kd")
        clean[joint] = {name: float(target[name]) for name in required}
        if not all(math.isfinite(value) for value in clean[joint].values()):
            raise ValueError(f"关节 {joint} 包含 NaN/Inf")
    next_feedback = {"outputs": as_json(outputs), "requested_targets": clean}
    return {"ok": True, "ready": True, "targets": clean}, next_feedback


def main():
    import onnxruntime as ort

    policy_cls = load_policy(sys.argv[1])
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    models = sys.argv[2:]
    if len(models) not in (1, 2):
        raise ValueError("worker requires one model or the walk/stand pair")
    slots = ["default"] if len(models) == 1 else ["walk", "stand"]
    sessions = {
        slot: ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
        for slot, path in zip(slots, models)
    }
    with contextlib.redirect_stdout(sys.stderr):
        probe = policy_cls()
        contract = probe.describe()
    for slot, session in sessions.items():
        try:
            validate_contract(contract, session)
        except Exception as error:
            raise ValueError(f"{slot} model is incompatible with the shared policy: {error}") from error
    json.dumps(contract, allow_nan=False)
    emit({"ok": True, "ready": True, "contract": contract})

    instance = None
    feedback = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            action = request.get("action")
            if action == "reset":
                instance = make_instance(policy_cls, contract, request["frame"])
                feedback = None
                emit({"ok": True, "reset": True})
            elif action == "step":
                if instance is None:
                    raise ValueError("策略尚未 reset")
                slot = request.get("model", "default")
                if slot not in sessions:
                    raise ValueError("未知策略模型槽位")
                if slot in ("walk", "stand") and request["frame"]["policy_context"]["action"] != slot:
                    raise ValueError("走/站模型选择必须与 policy_context.action 一致")
                session = sessions[slot]
                reply, feedback = run_step(instance, contract, session, request["frame"], feedback)
                emit(reply)
            else:
                raise ValueError("未知 worker 操作")
        except Exception as error:
            fail(error)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        fail(error)
        sys.exit(1)
