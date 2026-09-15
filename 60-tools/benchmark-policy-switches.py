#!/usr/bin/env python3
"""Benchmark deployed policy transitions through isolated workers; never opens RobotIo."""
import json
import math
import os
from pathlib import Path
import pwd
import resource
import selectors
import statistics
import subprocess
import time

MANIFEST = Path("/var/lib/robotd/policies/manifest.json")
WORKER = Path("/home/xduck1/xrange/10-source/rk3566/microduck/robotd/src/custom_policy_worker.py")
PYTHON = "/var/lib/robotd/model-python/bin/python"
SAMPLES = int(os.environ.get("XDUCK_BENCH_SAMPLES", "50"))
STATES = ["walk", "stand", "ground_pick", "kick_left", "kick_right", "roulade", "sit", "rise"]
SOURCES = os.environ.get("XDUCK_BENCH_SOURCES", ",".join(STATES)).split(",")
TARGETS = os.environ.get("XDUCK_BENCH_TARGETS", ",".join(STATES)).split(",")
SLOT = {"sit": "sitstand", "rise": "sitstand"}
MODEL = {"walk": "walk", "stand": "stand"}
COMMAND = {
    "walk": [0.3, 0.0, 0.0], "stand": [0.0, 0.0, 0.0],
    "ground_pick": [1.0, 0.0, 0.0], "kick_left": [0.0, 0.0, 0.0],
    "kick_right": [0.0, 0.0, 0.0], "roulade": [0.0, 0.0, 0.0],
    "sit": [1.0, 0.0, 0.0], "rise": [0.0, 0.0, 0.0],
}
FRAME = {
    "sequence": 0, "gateway_tick_ms": 0, "dt": 0.02,
    "joint_names": [
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
        "neck_pitch", "head_pitch", "head_yaw", "head_roll", "mouth",
    ],
    "positions": [0.0, -0.0873, -0.4579, -0.0049, 0.4530,
                  0.0, 0.0873, 0.4579, 0.0049, -0.4530,
                  0.3491, 0.3491, 0.0, 0.0, 0.0],
    "velocities": [0.0] * 15, "motor_torques_nm": [0.0] * 15,
    "motor_flags": [0] * 15, "motor_temperatures_c": [25.0] * 15,
    "motor_feedback_age_ms": [0] * 15, "attitude_rpy": [0.0] * 3,
    "imu": {"gyro": [0.0] * 3, "gravity": [0.0, 0.0, -1.0],
            "quat": [1.0, 0.0, 0.0, 0.0]},
    "command": {"twist": [0.0] * 3, "head": [0.0] * 4, "body": [0.0] * 3},
}


def limits():
    for kind, value in ((resource.RLIMIT_NOFILE, 64), (resource.RLIMIT_NPROC, 16),
                        (resource.RLIMIT_FSIZE, 128 * 1024 * 1024),
                        (resource.RLIMIT_AS, 1536 * 1024 * 1024),
                        (resource.RLIMIT_CORE, 0)):
        resource.setrlimit(kind, (value, value))


def spawn(policy, *models):
    identity = pwd.getpwnam("nobody")
    command = [
        "/usr/bin/bwrap", "--die-with-parent", "--new-session", "--unshare-net",
        "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--ro-bind", "/", "/",
        "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--tmpfs", "/run",
        "--chdir", "/", "--", "/usr/bin/setpriv", "--reuid", str(identity.pw_uid),
        "--regid", str(identity.pw_gid), "--clear-groups", "--no-new-privs",
        "--inh-caps=-all", "--ambient-caps=-all", "--bounding-set=-all",
        PYTHON, "-I", "-u", "-c", WORKER.read_text(), str(policy), *map(str, models),
    ]
    env = {"PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env, preexec_fn=limits)


def receive(process, timeout=15):
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    if not selector.select(timeout):
        raise TimeoutError("worker response timeout")
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(process.stderr.read() or "worker exited")
    reply = json.loads(line)
    if not reply.get("ok"):
        raise RuntimeError(reply.get("error", str(reply)))
    return reply


sequence = 0
def request(process, action, state):
    global sequence
    sequence += 1
    frame = dict(FRAME)
    frame["sequence"] = sequence
    frame["gateway_tick_ms"] = sequence * 20
    frame["command"] = dict(FRAME["command"], twist=COMMAND[state])
    value = {"action": action, "frame": frame}
    if action == "step" and state in MODEL:
        value["model"] = MODEL[state]
    process.stdin.write(json.dumps(value, separators=(",", ":")) + "\n")
    process.stdin.flush()
    return receive(process)


def summarize(values):
    values = sorted(values)
    percentile = lambda p: values[max(0, math.ceil(p * len(values)) - 1)]
    return {"n": len(values), "median_ms": statistics.median(values),
            "p95_ms": percentile(0.95), "p99_ms": percentile(0.99),
            "min_ms": values[0], "max_ms": values[-1]}


def python_pid(process):
    """Find sandboxed Python below bwrap, across its PID namespace boundary."""
    # /proc/<pid>/task/<pid>/children is empty at the bwrap PID-namespace
    # boundary on this target. Build the host-side parent map from /proc instead.
    children_by_parent = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            fields = stat[stat.rfind(") ") + 2:].split()
            children_by_parent.setdefault(int(fields[1]), []).append(int(entry.name))
        except (FileNotFoundError, PermissionError, IndexError, ValueError):
            continue
    pending = list(children_by_parent.get(process.pid, []))
    descendants = []
    while pending:
        pid = pending.pop()
        descendants.append(pid)
        pending.extend(children_by_parent.get(pid, []))
    for pid in reversed(descendants):
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes()
            name = Path(f"/proc/{pid}/comm").read_text().strip()
            if PYTHON.encode() in command or name.startswith("python"):
                return pid
        except (FileNotFoundError, PermissionError):
            pass
    raise RuntimeError(f"could not find sandboxed Python below PID {process.pid}")


def scheduler_snapshot(pid):
    sched = [int(value) for value in Path(f"/proc/{pid}/schedstat").read_text().split()[:3]]
    status = Path(f"/proc/{pid}/status").read_text().splitlines()
    switches = {line.split(":", 1)[0]: int(line.split()[-1]) for line in status
                if line.startswith(("voluntary_ctxt_switches:", "nonvoluntary_ctxt_switches:"))}
    stat = Path(f"/proc/{pid}/stat").read_text()
    fields = stat[stat.rfind(") ") + 2:].split()  # starts at proc(5) field 3
    return {"runtime_ns": sched[0], "runqueue_wait_ns": sched[1], "timeslices": sched[2],
            "minor_faults": int(fields[7]), "major_faults": int(fields[9]), **switches}


def scheduler_delta(before, after):
    return {key: after[key] - before[key] for key in before}


records = json.loads(MANIFEST.read_text())
active = {}
for key, record in records.items():
    slot = key.split("--", 1)[0]
    if slot in {SLOT.get(state, state) for state in STATES}:
        active[slot] = record["active"]
for slot in ("walk", "stand", "ground_pick", "kick_left", "kick_right", "roulade", "sitstand"):
    if slot not in active:
        raise RuntimeError(f"missing active slot: {slot}")
if active["walk"]["policy_path"] != active["stand"]["policy_path"]:
    raise RuntimeError("deployed walk/stand consumer paths are not shared")

workers = {}
processes = []
try:
    locomotion = spawn(active["walk"]["policy_path"], active["walk"]["path"], active["stand"]["path"])
    receive(locomotion)
    processes.append(locomotion)
    workers["walk"] = workers["stand"] = locomotion
    for state in STATES[2:]:
        slot = SLOT.get(state, state)
        if slot not in workers:
            process = spawn(active[slot]["policy_path"], active[slot]["path"])
            receive(process)
            processes.append(process)
            workers[slot] = process
        workers[state] = workers[slot]
    worker_pids = {state: python_pid(workers[state]) for state in STATES}

    # Match load-time warmup, then leave every instance reset as production does.
    request(locomotion, "reset", "stand")
    for _ in range(3):
        request(locomotion, "step", "walk")
        request(locomotion, "step", "stand")
    request(locomotion, "reset", "stand")
    warmed = set()
    for state in STATES[2:]:
        process = workers[state]
        if id(process) in warmed:
            continue
        warmed.add(id(process))
        request(process, "reset", state)
        for _ in range(3):
            request(process, "step", state)
        request(process, "reset", state)

    raw = {}
    for source in SOURCES:
        for target in TARGETS:
            if source == target:
                continue
            step_ms, reset_ms, total_ms, scheduler, client_scheduler = [], [], [], [], []
            for _ in range(SAMPLES):
                # Establish source state outside the timed region.
                if source in MODEL:
                    request(workers[source], "step", source)
                else:
                    request(workers[source], "reset", source)
                    request(workers[source], "step", source)
                sched_before = scheduler_snapshot(worker_pids[target])
                client_sched_before = scheduler_snapshot(os.getpid())
                began = time.perf_counter_ns()
                reset_finished = began
                if target not in MODEL:
                    request(workers[target], "reset", target)
                    reset_finished = time.perf_counter_ns()
                request(workers[target], "step", target)
                finished = time.perf_counter_ns()
                client_sched_after = scheduler_snapshot(os.getpid())
                sched_after = scheduler_snapshot(worker_pids[target])
                reset_ms.append((reset_finished - began) / 1e6)
                step_ms.append((finished - reset_finished) / 1e6)
                total_ms.append((finished - began) / 1e6)
                scheduler.append(scheduler_delta(sched_before, sched_after))
                client_scheduler.append(scheduler_delta(client_sched_before, client_sched_after))
            raw[f"{source}->{target}"] = {
                "reset_ms": reset_ms if target not in MODEL else None,
                "step_ms": step_ms, "total_ms": total_ms, "scheduler": scheduler,
                "client_scheduler": client_scheduler,
            }

    matrix = {}
    for source in SOURCES:
        matrix[source] = {}
        for target in TARGETS:
            if source == target:
                matrix[source][target] = None
                continue
            item = raw[f"{source}->{target}"]
            matrix[source][target] = {
                "reset": summarize(item["reset_ms"]) if item["reset_ms"] else None,
                "first_step": summarize(item["step_ms"]), "blocking_total": summarize(item["total_ms"]),
            }
    by_target = {}
    for target in TARGETS:
        items = [raw[f"{source}->{target}"] for source in SOURCES if source != target]
        totals = [v for item in items for v in item["total_ms"]]
        steps = [v for item in items for v in item["step_ms"]]
        resets = [v for item in items if item["reset_ms"] for v in item["reset_ms"]]
        by_target[target] = {"reset": summarize(resets) if resets else None,
                             "first_step": summarize(steps), "blocking_total": summarize(totals)}
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "samples_per_ordered_pair": SAMPLES,
        "states": STATES, "sources": SOURCES, "targets": TARGETS, "motor_control": False,
        "semantics": "Walk/Stand: one shared prewarmed consumer, step only. Other targets: reset request/reply plus first step, matching reset-on-enter.",
        "scope": "Wall clock includes JSON IPC, policy preprocess, CPU ONNX inference and postprocess; excludes RobotIo, motor application and Rust scheduler overhead.",
        "sandbox": "bwrap, network namespace isolated, private /dev, read-only host, nobody uid, numerical threads=1",
        "production_version": subprocess.check_output(["readlink", "-f", "/opt/robot/daemon/current"], text=True).strip(),
        "models": {slot: {"id": value["id"], "model": value["path"],
                           "policy": value["policy_path"]} for slot, value in active.items()},
        "by_target": by_target, "transition_matrix": matrix, "raw_samples": raw,
    }
    output_dir = Path("/home/xduck1/xrange/50-logs/test/rk3566/microduck")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"policy-switch-benchmark-{time.strftime('%Y%m%d-%H%M%S')}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output)
    print(json.dumps({"by_target": by_target}, ensure_ascii=False, indent=2))
finally:
    for process in processes:
        if process.poll() is None:
            process.kill()
        process.wait()
