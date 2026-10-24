use duck_control::{Command, MitTarget, Sensors, JOINT_NAMES, NUM_JOINTS, policy::Net};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    ffi::CString,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command as ProcessCommand, Stdio},
    sync::{Arc, Mutex, mpsc},
    time::{Duration, Instant},
};
use std::os::unix::process::CommandExt;

const START_TIMEOUT: Duration = Duration::from_secs(8);
const RESET_TIMEOUT: Duration = Duration::from_secs(2);
const STEP_TIMEOUT: Duration = Duration::from_millis(18);
const WORKER_NOFILE_LIMIT: libc::rlim_t = 64;
const WORKER_PROCESS_LIMIT: libc::rlim_t = 16;
const WORKER_FILE_LIMIT: libc::rlim_t = 128 * 1024 * 1024;
const WORKER_ADDRESS_SPACE_LIMIT: libc::rlim_t = 1536 * 1024 * 1024;

#[derive(Serialize)]
struct Frame<'a> {
    sequence: u64,
    gateway_tick_ms: u32,
    dt: f64,
    joint_names: &'static [&'static str; NUM_JOINTS],
    positions: &'a [f64; NUM_JOINTS],
    velocities: &'a [f64; NUM_JOINTS],
    motor_torques_nm: &'a [f64; NUM_JOINTS],
    motor_flags: &'a [u8; NUM_JOINTS],
    motor_temperatures_c: &'a [f64; NUM_JOINTS],
    motor_feedback_age_ms: &'a [u32; NUM_JOINTS],
    attitude_rpy: &'a [f64; 3],
    imu: Imu<'a>,
    command: PolicyCommand<'a>,
    policy_context: PolicyContext,
}

#[derive(Serialize)]
struct Imu<'a> {
    gyro: &'a [f64; 3],
    gravity: &'a [f64; 3],
    quat: &'a [f64; 4],
}

#[derive(Serialize)]
struct PolicyCommand<'a> {
    twist: &'a [f64; 3],
    head: &'a [f64; 4],
    body: [f64; 3],
}

/// Scheduler state supplied to policy.py without imposing any model-observation layout.
///
/// `command` remains the platform-gated base intent.  The consumer combines it with this
/// context if its model needs an action-specific encoding (for example ground-pick phase or
/// the sit/rise flag).  It is deliberately metadata, not a prebuilt command observation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct PolicyContext {
    pub action: &'static str,
    pub phase: Option<f64>,
    pub body_active: bool,
}

impl PolicyContext {
    pub const fn new(action: &'static str, phase: Option<f64>, body_active: bool) -> Self {
        Self { action, phase, body_active }
    }

    fn warmup(net: Net) -> Self {
        match net {
            Net::Walk => Self::new("walk", None, false),
            Net::Stand => Self::new("stand", None, false),
            Net::SitStand => Self::new("sit", None, false),
            Net::GroundPick => Self::new("ground_pick", Some(0.0), false),
            Net::KickLeft => Self::new("kick_left", None, false),
            Net::KickRight => Self::new("kick_right", None, false),
            Net::Roulade => Self::new("roulade", None, false),
        }
    }
}

#[derive(Deserialize)]
struct Reply {
    ok: bool,
    #[serde(default)]
    ready: bool,
    error: Option<String>,
    reason: Option<String>,
    targets: Option<BTreeMap<String, WireTarget>>,
    contract: Option<Value>,
    timing: Option<StepTiming>,
    #[serde(skip)]
    reader_received_ns: u64,
    #[serde(skip)]
    host_received_ns: u64,
}

#[derive(Deserialize)]
struct StepTiming {
    preprocess_ms: f64,
    input_ms: f64,
    onnx_ms: f64,
    output_ms: f64,
    validate_ms: f64,
    worker_total_ms: f64,
    worker_cpu_ms: f64,
    worker_received_ns: u64,
    worker_reply_start_ns: u64,
}

#[derive(Deserialize)]
struct WireTarget {
    position: f64,
    velocity: f64,
    torque_ff: f64,
    kp: f64,
    kd: f64,
}

#[derive(Clone, Copy)]
pub struct CustomOutput {
    pub positions: [f64; NUM_JOINTS],
    pub mit: [MitTarget; NUM_JOINTS],
}

pub struct CustomPolicy {
    policy_path: PathBuf,
    model_paths: Vec<(Net, PathBuf)>,
    // bwrap's --die-with-parent watches the thread that spawned it. Keep background
    // preparation and recovery threads alive until their worker is dropped.
    keepalive: Option<mpsc::Sender<()>>,
    child: Arc<Mutex<Child>>,
    input: ChildStdin,
    lines: mpsc::Receiver<(Result<String, String>, u64)>,
    contract: Value,
    sequence: u64,
    failed: bool,
    recovery: Option<mpsc::Receiver<Result<Box<CustomPolicy>, String>>>,
    recovery_error: Option<String>,
    needs_reset: bool,
    handoff_targets: Option<[MitTarget; NUM_JOINTS]>,
    period_s: f64,
    elapsed_s: f64,
    last_output: Option<CustomOutput>,
}

impl CustomPolicy {
    pub fn load(policy: &Path, model: &Path, net: Net) -> Result<Self, String> {
        Self::load_models(policy, &[(net, model)])
    }

    pub fn load_pair(policy: &Path, walk: &Path, stand: &Path) -> Result<Self, String> {
        Self::load_models(policy, &[(Net::Walk, walk), (Net::Stand, stand)])
    }

    fn load_models(policy: &Path, models: &[(Net, &Path)]) -> Result<Self, String> {
        let (worker_uid, worker_gid) = policy_identity()?;
        let python = std::env::var_os("ROBOT_POLICY_PYTHON")
            .or_else(|| std::env::var_os("ROBOT_MODEL_PYTHON"))
            .unwrap_or_else(|| {
                let installed = Path::new("/var/lib/robotd/model-python/bin/python");
                if installed.is_file() { installed.as_os_str().to_owned() } else { "/usr/bin/python3".into() }
            });
        if !Path::new(&python).is_absolute() {
            return Err("ROBOT_POLICY_PYTHON/ROBOT_MODEL_PYTHON 必须是绝对路径".into());
        }
        // The uploaded module executes arbitrary Python. Bubblewrap removes the host network and
        // hardware device tree, makes the host filesystem read-only, and gives the worker only
        // private /tmp and /run mounts. The separate uid is still useful for host files that are
        // intentionally unreadable to ordinary accounts.
        let mut child = ProcessCommand::new("/usr/bin/bwrap");
        child
            .args([
                "--die-with-parent",
                "--new-session",
                "--unshare-net",
                "--unshare-pid",
                "--unshare-ipc",
                "--unshare-uts",
                "--ro-bind",
                "/",
                "/",
                "--dev",
                "/dev",
                "--proc",
                "/proc",
                "--tmpfs",
                "/tmp",
                "--tmpfs",
                "/run",
                "--chdir",
                "/",
            ])
            .arg("--")
            // Bubblewrap must enter the mount namespaces while still privileged on this
            // Debian kernel. setpriv is the last trusted program before uploaded code: it
            // drops uid/gid privileges, supplementary groups, all capabilities and the
            // ability to gain privilege again.
            .arg("/usr/bin/setpriv")
            .arg("--reuid")
            .arg(worker_uid.to_string())
            .arg("--regid")
            .arg(worker_gid.to_string())
            .args([
                "--clear-groups",
                "--no-new-privs",
                "--inh-caps=-all",
                "--ambient-caps=-all",
                "--bounding-set=-all",
            ])
            .arg(python)
            .args(["-I", "-u", "-c", include_str!("custom_policy_worker.py")])
            .arg(policy)
            .args(models.iter().map(|(_, path)| path))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .env_clear()
            .env("PYTHONDONTWRITEBYTECODE", "1")
            // Each skill (or shared walk/stand group) owns a worker. Keep libraries single-threaded so
            // their aggregate thread count stays inside the shared robot-policy RLIMIT_NPROC.
            .env("OPENBLAS_NUM_THREADS", "1")
            .env("OMP_NUM_THREADS", "1")
            .env("MKL_NUM_THREADS", "1")
            .env("NUMEXPR_NUM_THREADS", "1")
            .current_dir("/");
        unsafe {
            child.pre_exec(|| {
                // Imports/rollbacks originate on nice +10 IPC threads. Restore priority
                // before bwrap/setpriv drops privileges, without changing the caller.
                set_policy_scheduling()?;
                if libc::setgroups(0, std::ptr::null()) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                set_limit(libc::RLIMIT_NOFILE, WORKER_NOFILE_LIMIT)?;
                set_limit(libc::RLIMIT_NPROC, WORKER_PROCESS_LIMIT)?;
                set_limit(libc::RLIMIT_FSIZE, WORKER_FILE_LIMIT)?;
                set_limit(libc::RLIMIT_AS, WORKER_ADDRESS_SPACE_LIMIT)?;
                set_limit(libc::RLIMIT_CORE, 0)?;
                Ok(())
            });
        }
        let mut child = child
            .spawn()
            .map_err(|error| format!("无法启动受限两文件策略运行器（需要 bubblewrap）：{error}"))?;
        let input = child.stdin.take().ok_or("策略运行器缺少 stdin")?;
        let output = child.stdout.take().ok_or("策略运行器缺少 stdout")?;
        let (send, lines) = mpsc::channel();
        if let Err(error) = spawn_policy_reply_reader(output, send) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("无法启动策略回复读取线程：{error}"));
        }
        let child = Arc::new(Mutex::new(child));
        let mut policy = Self {
            policy_path: policy.to_path_buf(),
            model_paths: models.iter().map(|(net, path)| (*net, path.to_path_buf())).collect(),
            keepalive: None,
            child,
            input,
            lines,
            contract: Value::Null,
            sequence: 0,
            failed: false,
            recovery: None,
            recovery_error: None,
            needs_reset: false,
            handoff_targets: None,
            period_s: 0.02,
            elapsed_s: 0.0,
            last_output: None,
        };
        let reply = policy.receive(START_TIMEOUT)?;
        policy.contract = reply.contract.ok_or("策略运行器未返回接口契约")?;
        policy.period_s = policy.contract.get("period_us").and_then(Value::as_u64)
            .ok_or("策略契约缺少 period_us")? as f64 / 1_000_000.0;
        let sensors = Sensors::default();
        let first_context = PolicyContext::warmup(models[0].0);
        policy.reset(&sensors, first_context)?;
        let warmup_dt = policy.period_s;
        let slots: Vec<_> = if models.len() == 2 {
            vec![("walk", PolicyContext::warmup(Net::Walk)),
                 ("stand", PolicyContext::warmup(Net::Stand))]
        } else if models[0].0 == Net::SitStand {
            vec![("default", PolicyContext::new("sit", None, false)),
                 ("default", PolicyContext::new("rise", None, false))]
        } else if models[0].0 == Net::GroundPick {
            vec![("default", PolicyContext::new("ground_pick", Some(0.0), false)),
                 ("default", PolicyContext::new("ground_pick", Some(0.25), false))]
        } else {
            vec![("default", first_context)]
        };
        for _ in 0..3 {
            for (slot, context) in &slots {
                policy.step_model(&sensors, &Command::default(), *context, warmup_dt, slot)?;
            }
        }
        policy.reset(&sensors, first_context)?;
        Ok(policy)
    }

    pub fn contract(&self) -> &Value { &self.contract }

    fn reset(&mut self, sensors: &Sensors, context: PolicyContext) -> Result<(), String> {
        self.sequence = 0;
        self.elapsed_s = 0.0;
        self.last_output = None;
        let command = Command::default();
        let frame = self.frame(sensors, &command, context, 0.02);
        self.transact(json!({"action":"reset", "frame":frame}), RESET_TIMEOUT)?;
        self.needs_reset = false;
        Ok(())
    }

    pub fn mark_reset(&mut self) { self.needs_reset = true; self.handoff_targets = None; }

    /// A different controller drove while this worker was suspended. Rebuild its
    /// private history and seed feedback from the last successful bus write.
    pub fn handoff(&mut self, targets: [MitTarget; NUM_JOINTS]) {
        self.mark_reset();
        self.handoff_targets = Some(targets);
    }

    pub(crate) fn keep_creator_alive(&mut self, keepalive: mpsc::Sender<()>) {
        self.keepalive = Some(keepalive);
    }

    /// Rebuild after a worker timeout without blocking the 50 Hz motor loop. Only an explicit
    /// controller reset starts recovery; ordinary failed frames continue to hold the pose.
    pub fn recover_failed(&mut self) {
        if !self.failed || self.recovery.is_some() { return; }
        self.recovery_error = None;
        let policy_path = self.policy_path.clone();
        let model_paths = self.model_paths.clone();
        let (send, receive) = mpsc::channel();
        let (keepalive, alive) = mpsc::channel();
        self.recovery = Some(receive);
        std::thread::spawn(move || {
            let models: Vec<_> = model_paths.iter().map(|(net, path)| (*net, path.as_path())).collect();
            let result = Self::load_models(&policy_path, &models).map(|mut worker| {
                worker.keep_creator_alive(keepalive);
                Box::new(worker)
            });
            let _ = send.send(result);
            // When the owner drops the worker, it drops the sender and releases this thread.
            // A failed load also drops the sender immediately.
            let _ = alive.recv();
        });
        tracing::warn!("rebuilding failed policy worker after explicit initialization");
    }

    fn finish_recovery(&mut self) -> Result<(), String> {
        let Some(receive) = self.recovery.as_ref() else {
            return self.recovery_error.as_ref().map_or(Ok(()), |error| Err(error.clone()));
        };
        match receive.try_recv() {
            Ok(Ok(mut replacement)) => {
                replacement.mark_reset();
                if let Some(targets) = self.handoff_targets { replacement.handoff(targets); }
                *self = *replacement;
                tracing::warn!("failed policy worker rebuilt and warmed");
                Ok(())
            }
            Ok(Err(error)) => {
                self.recovery = None;
                self.recovery_error = Some(format!("策略运行器重建失败：{error}"));
                Err(self.recovery_error.as_ref().unwrap().clone())
            }
            Err(mpsc::TryRecvError::Empty) => Err("策略运行器正在重建；保持当前姿态".into()),
            Err(mpsc::TryRecvError::Disconnected) => {
                self.recovery = None;
                self.recovery_error = Some("策略运行器重建线程意外退出".into());
                Err(self.recovery_error.as_ref().unwrap().clone())
            }
        }
    }

    pub fn step(&mut self, sensors: &Sensors, command: &Command, context: PolicyContext, dt: f64) -> Result<CustomOutput, String> {
        self.step_model(sensors, command, context, dt, "default")
    }

    pub fn step_model(&mut self, sensors: &Sensors, command: &Command, context: PolicyContext, dt: f64, model: &str) -> Result<CustomOutput, String> {
        self.finish_recovery()?;
        if self.needs_reset {
            self.sequence = 0;
            let frame = self.frame(sensors, command, context, dt);
            let handoff = self.handoff_targets.map(|targets| JOINT_NAMES.iter().enumerate()
                .filter(|(_, name)| **name != "mouth")
                .map(|(index, name)| {
                    let t = targets[index];
                    (*name, json!({"position":t.position,"velocity":t.velocity,
                        "torque_ff":t.torque_ff,"kp":t.kp,"kd":t.kd}))
                }).collect::<BTreeMap<_, _>>());
            self.transact(json!({"action":"reset", "frame":frame, "handoff_targets":handoff}), STEP_TIMEOUT)?;
            self.handoff_targets = None;
            self.needs_reset = false;
            self.elapsed_s = 0.0;
            self.last_output = None;
        }
        self.elapsed_s += dt.max(0.0);
        if let Some(output) = self.last_output
            && self.elapsed_s + f64::EPSILON < self.period_s
        {
            return Ok(output);
        }
        self.elapsed_s = (self.elapsed_s - self.period_s).max(0.0);
        self.sequence = self.sequence.wrapping_add(1);
        let frame = self.frame(sensors, command, context, dt);
        let reply = self.transact(json!({"action":"step", "model":model, "frame":frame}), STEP_TIMEOUT)?;
        if !reply.ready {
            return Err(reply.reason.unwrap_or_else(|| "策略尚未就绪".into()));
        }
        let targets = reply.targets.ok_or("策略未返回电机目标")?;
        let mut positions = sensors.positions;
        let mut mit = [MitTarget::default(); NUM_JOINTS];
        for (index, name) in JOINT_NAMES.iter().enumerate() {
            if *name == "mouth" {
                mit[index].position = positions[index];
                continue;
            }
            let value = targets.get(*name).ok_or_else(|| format!("策略缺少关节 {name}"))?;
            positions[index] = value.position;
            mit[index] = MitTarget {
                position: value.position,
                velocity: value.velocity,
                torque_ff: value.torque_ff,
                kp: value.kp,
                kd: value.kd,
            };
        }
        let output = CustomOutput { positions, mit };
        self.last_output = Some(output);
        Ok(output)
    }

    fn frame<'a>(&self, sensors: &'a Sensors, command: &'a Command, context: PolicyContext, dt: f64) -> Frame<'a> {
        Frame {
            sequence: self.sequence,
            gateway_tick_ms: sensors.gateway_tick_ms,
            dt,
            joint_names: &JOINT_NAMES,
            positions: &sensors.positions,
            velocities: &sensors.velocities,
            motor_torques_nm: &sensors.motor_torques_nm,
            motor_flags: &sensors.motor_flags,
            motor_temperatures_c: &sensors.motor_temperatures_c,
            motor_feedback_age_ms: &sensors.motor_feedback_age_ms,
            attitude_rpy: &sensors.attitude_rpy,
            imu: Imu { gyro: &sensors.imu.gyro, gravity: &sensors.imu.gravity, quat: &sensors.imu.quat },
            command: PolicyCommand {
                twist: &command.twist,
                head: &command.head,
                body: [command.body.z, command.body.roll, command.body.pitch],
            },
            policy_context: context,
        }
    }

    fn transact(&mut self, request: Value, timeout: Duration) -> Result<Reply, String> {
        if self.failed { return Err("策略运行器已失效，需重新初始化/使能".into()); }
        let action = request.get("action").and_then(Value::as_str).unwrap_or("unknown").to_owned();
        let model = request.get("model").and_then(Value::as_str).unwrap_or("default").to_owned();
        let sequence = request.get("frame").and_then(|frame| frame.get("sequence")).and_then(Value::as_u64).unwrap_or(0);
        let mut data = serde_json::to_vec(&request).map_err(|error| error.to_string())?;
        data.push(b'\n');
        let started = Instant::now();
        let host_started_ns = monotonic_ns();
        self.input.write_all(&data).and_then(|_| self.input.flush())
            .map_err(|error| self.stop(format!("写入策略运行器失败：{error}")))?;
        let host_written_ns = monotonic_ns();
        let reply = self.receive(timeout);
        let roundtrip_ms = started.elapsed().as_secs_f64() * 1000.0;
        match &reply {
            Ok(reply) if action == "step" && roundtrip_ms >= 10.0 => {
                if let Some(timing) = &reply.timing {
                    tracing::warn!(%model, sequence, roundtrip_ms,
                        preprocess_ms = timing.preprocess_ms,
                        input_ms = timing.input_ms,
                        onnx_ms = timing.onnx_ms,
                        output_ms = timing.output_ms,
                        validate_ms = timing.validate_ms,
                        worker_total_ms = timing.worker_total_ms,
                        worker_cpu_ms = timing.worker_cpu_ms,
                        transport_and_scheduling_ms = roundtrip_ms - timing.worker_total_ms,
                        host_write_ms = ns_ms(host_started_ns, host_written_ns),
                        worker_wakeup_ms = ns_ms(host_written_ns, timing.worker_received_ns),
                        worker_dispatch_ms = ns_ms(timing.worker_received_ns, timing.worker_reply_start_ns) - timing.worker_total_ms,
                        worker_emit_and_pipe_ms = ns_ms(timing.worker_reply_start_ns, reply.reader_received_ns),
                        reader_to_control_ms = ns_ms(reply.reader_received_ns, reply.host_received_ns),
                        "slow policy step");
                } else {
                    tracing::warn!(%model, sequence, roundtrip_ms, "slow policy step without worker timing");
                }
            }
            Err(error) if error.contains("18 ms 时限") => {
                tracing::error!(%action, %model, sequence, roundtrip_ms,
                    "policy worker timed out; SIGUSR1 Python stack dump requested");
            }
            _ => {}
        }
        reply
    }

    fn receive(&mut self, timeout: Duration) -> Result<Reply, String> {
        let (line, reader_received_ns) = match self.lines.recv_timeout(timeout) {
            Ok((Ok(line), received_ns)) => (line, received_ns),
            Ok((Err(error), _)) => return Err(self.stop(format!("读取策略运行器失败：{error}"))),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if let Ok(child) = self.child.lock() {
                    // bwrap is the direct child; with --unshare-pid it forks another bwrap
                    // process before execing Python. Signal Python, not either supervisor.
                    let parent = child.id();
                    if let Some(pid) = policy_python_pid(parent) {
                        unsafe { libc::kill(pid, libc::SIGUSR1); }
                    } else {
                        tracing::warn!(parent, "could not find policy Python PID for timeout stack dump");
                    }
                }
                std::thread::sleep(Duration::from_millis(5));
                return Err(self.stop(format!("策略运行超过 {} ms 时限", timeout.as_millis())));
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => return Err(self.stop("策略运行器已退出".into())),
        };
        let mut reply: Reply = serde_json::from_str(&line)
            .map_err(|error| self.stop(format!("策略运行器返回无效数据：{error}")))?;
        reply.reader_received_ns = reader_received_ns;
        reply.host_received_ns = monotonic_ns();
        if !reply.ok {
            // A policy exception is a failed frame, not a broken pipe. Keep the worker so
            // an explicit controller reset can rebuild the Policy instance. Protocol
            // corruption, timeout and process exit above still kill it fail-closed.
            return Err(reply.error.unwrap_or_else(|| "策略运行失败".into()));
        }
        Ok(reply)
    }

    fn stop(&mut self, message: String) -> String {
        self.failed = true;
        if let Ok(mut child) = self.child.lock() { let _ = child.kill(); }
        message
    }
}

#[cfg(target_os = "linux")]
fn set_policy_scheduling() -> std::io::Result<()> {
    let priority = libc::sched_param { sched_priority: 0 };
    if unsafe { libc::sched_setscheduler(0, libc::SCHED_OTHER, &priority) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    if unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, 0) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn set_policy_scheduling() -> std::io::Result<()> { Ok(()) }

fn spawn_policy_reply_reader(
    output: impl Read + Send + 'static,
    send: mpsc::Sender<(Result<String, String>, u64)>,
) -> std::io::Result<std::thread::JoinHandle<()>> {
    std::thread::Builder::new().name("policy-reply".into()).spawn(move || {
        // A ready Python reply must not wait behind IPC before reaching control.
        // Report setup failure through the normal startup channel; never run degraded.
        if let Err(error) = set_policy_scheduling() {
            let _ = send.send((Err(format!("无法设置策略回复线程调度：{error}")), monotonic_ns()));
            return;
        }
        for line in BufReader::new(output).lines() {
            let received_ns = monotonic_ns();
            if send.send((line.map_err(|error| error.to_string()), received_ns)).is_err() { break; }
        }
    })
}

fn monotonic_ns() -> u64 {
    let mut time = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time); }
    time.tv_sec as u64 * 1_000_000_000 + time.tv_nsec as u64
}

fn ns_ms(start: u64, end: u64) -> f64 {
    (end as i128 - start as i128) as f64 / 1_000_000.0
}

fn policy_python_pid(parent: u32) -> Option<libc::pid_t> {
    let mut pid = parent;
    for _ in 0..4 {
        let children = std::fs::read_to_string(format!("/proc/{pid}/task/{pid}/children")).ok()?;
        pid = children.split_whitespace().next()?.parse().ok()?;
        let comm = std::fs::read_to_string(format!("/proc/{pid}/comm")).ok()?;
        if comm.trim().starts_with("python") { return Some(pid as libc::pid_t); }
    }
    None
}

#[cfg(target_os = "linux")]
type LimitResource = libc::__rlimit_resource_t;
#[cfg(not(target_os = "linux"))]
type LimitResource = libc::c_int;

fn set_limit(resource: LimitResource, value: libc::rlim_t) -> std::io::Result<()> {
    let limit = libc::rlimit { rlim_cur: value, rlim_max: value };
    if unsafe { libc::setrlimit(resource, &limit) } == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

fn policy_identity() -> Result<(u32, u32), String> {
    let name = CString::new("robot-policy").unwrap();
    // getpwnam returns process-owned storage. We copy uid/gid immediately; model imports
    // are serialized by Models, so there is no concurrent lookup in this daemon.
    let entry = unsafe { libc::getpwnam(name.as_ptr()) };
    if entry.is_null() {
        return Err("缺少 robot-policy 系统用户；重新执行受管部署以安装 sysusers 配置".into());
    }
    let entry = unsafe { &*entry };
    Ok((entry.pw_uid, entry.pw_gid))
}

impl Drop for CustomPolicy {
    fn drop(&mut self) {
        if let Ok(mut child) = self.child.lock() { let _ = child.kill(); let _ = child.wait(); }
    }
}

#[cfg(test)]
mod recovery_tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "requires root/CAP_SYS_NICE; no model or motor IO"]
    fn ipc_creator_keeps_low_priority_while_child_and_reply_reader_use_normal_priority() {
        std::thread::spawn(|| {
            assert_eq!(unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, 10) }, 0);
            let mut command = ProcessCommand::new("sh");
            command.args(["-c", "ps -o ni= -p $$"]);
            unsafe { command.pre_exec(set_policy_scheduling); }
            let output = command.output().unwrap();
            assert!(output.status.success());
            assert_eq!(String::from_utf8(output.stdout).unwrap().trim(), "0");
            assert_eq!(unsafe { libc::getpriority(libc::PRIO_PROCESS, 0) }, 10);

            let (reader, mut writer) = std::os::unix::net::UnixStream::pair().unwrap();
            let (send, receive) = mpsc::channel();
            let handle = spawn_policy_reply_reader(reader, send).unwrap();
            writer.write_all(b"ready\n").unwrap();
            assert_eq!(receive.recv_timeout(Duration::from_secs(2)).unwrap().0.unwrap(), "ready");
            let readers: Vec<_> = std::fs::read_dir("/proc/self/task").unwrap()
                .filter_map(Result::ok)
                .filter(|task| std::fs::read_to_string(task.path().join("comm"))
                    .is_ok_and(|name| name.trim() == "policy-reply"))
                .collect();
            assert!(!readers.is_empty());
            for reader in readers {
                let stat = std::fs::read_to_string(reader.path().join("stat")).unwrap();
                let fields: Vec<_> = stat.rsplit_once(") ").unwrap().1.split_whitespace().collect();
                assert_eq!(fields[16], "0", "reply reader must have nice 0");
                let tid = reader.file_name().to_string_lossy().parse::<i32>().unwrap();
                assert_eq!(unsafe { libc::sched_getscheduler(tid) }, libc::SCHED_OTHER);
            }
            assert_eq!(unsafe { libc::getpriority(libc::PRIO_PROCESS, 0) }, 10);
            drop(writer);
            handle.join().unwrap();
        }).join().unwrap();
    }

    fn placeholder(failed: bool) -> CustomPolicy {
        let mut child = ProcessCommand::new("sleep")
            .arg("10")
            .stdin(Stdio::piped())
            .spawn()
            .unwrap();
        let input = child.stdin.take().unwrap();
        let (_send, lines) = mpsc::channel();
        CustomPolicy {
            policy_path: PathBuf::new(),
            model_paths: Vec::new(),
            keepalive: None,
            child: Arc::new(Mutex::new(child)),
            input,
            lines,
            contract: Value::Null,
            sequence: 0,
            failed,
            recovery: None,
            recovery_error: None,
            needs_reset: false,
            handoff_targets: None,
            period_s: 0.02,
            elapsed_s: 0.0,
            last_output: None,
        }
    }

    #[test]
    fn a_warmed_replacement_replaces_the_failed_worker_and_resets_on_first_frame() {
        let mut failed = placeholder(true);
        let (send, receive) = mpsc::channel();
        let (keepalive, alive) = mpsc::channel();
        let (released, release) = mpsc::channel();
        let owner = std::thread::spawn(move || {
            let _ = alive.recv();
            let _ = released.send(());
        });
        let mut replacement = placeholder(false);
        replacement.keepalive = Some(keepalive);
        failed.recovery = Some(receive);
        assert!(send.send(Ok(Box::new(replacement))).is_ok());
        failed.finish_recovery().unwrap();
        assert!(!failed.failed);
        assert!(failed.needs_reset);
        assert!(failed.recovery.is_none());
        assert!(release.try_recv().is_err());
        drop(failed);
        assert!(release.recv_timeout(Duration::from_secs(1)).is_ok());
        owner.join().unwrap();
    }
}
