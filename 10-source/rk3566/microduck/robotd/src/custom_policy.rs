use duck_control::{Command, MitTarget, Sensors, JOINT_NAMES, NUM_JOINTS, policy::Net};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    ffi::CString,
    io::{BufRead, BufReader, Write},
    path::Path,
    process::{Child, ChildStdin, Command as ProcessCommand, Stdio},
    sync::{Arc, Mutex, mpsc},
    time::Duration,
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
    child: Arc<Mutex<Child>>,
    input: ChildStdin,
    lines: mpsc::Receiver<Result<String, String>>,
    contract: Value,
    sequence: u64,
    failed: bool,
    needs_reset: bool,
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
        let child = Arc::new(Mutex::new(child));
        let (send, lines) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(output).lines() {
                if send.send(line.map_err(|error| error.to_string())).is_err() { break; }
            }
        });
        let mut policy = Self {
            child,
            input,
            lines,
            contract: Value::Null,
            sequence: 0,
            failed: false,
            needs_reset: false,
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

    pub fn mark_reset(&mut self) { self.needs_reset = true; }

    pub fn step(&mut self, sensors: &Sensors, command: &Command, context: PolicyContext, dt: f64) -> Result<CustomOutput, String> {
        self.step_model(sensors, command, context, dt, "default")
    }

    pub fn step_model(&mut self, sensors: &Sensors, command: &Command, context: PolicyContext, dt: f64, model: &str) -> Result<CustomOutput, String> {
        if self.needs_reset {
            self.sequence = 0;
            let frame = self.frame(sensors, command, context, dt);
            self.transact(json!({"action":"reset", "frame":frame}), STEP_TIMEOUT)?;
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
        let mut data = serde_json::to_vec(&request).map_err(|error| error.to_string())?;
        data.push(b'\n');
        self.input.write_all(&data).and_then(|_| self.input.flush())
            .map_err(|error| self.stop(format!("写入策略运行器失败：{error}")))?;
        self.receive(timeout)
    }

    fn receive(&mut self, timeout: Duration) -> Result<Reply, String> {
        let line = match self.lines.recv_timeout(timeout) {
            Ok(Ok(line)) => line,
            Ok(Err(error)) => return Err(self.stop(format!("读取策略运行器失败：{error}"))),
            Err(mpsc::RecvTimeoutError::Timeout) => return Err(self.stop(format!("策略运行超过 {} ms 时限", timeout.as_millis()))),
            Err(mpsc::RecvTimeoutError::Disconnected) => return Err(self.stop("策略运行器已退出".into())),
        };
        let reply: Reply = serde_json::from_str(&line)
            .map_err(|error| self.stop(format!("策略运行器返回无效数据：{error}")))?;
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

fn set_limit(resource: libc::__rlimit_resource_t, value: libc::rlim_t) -> std::io::Result<()> {
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
