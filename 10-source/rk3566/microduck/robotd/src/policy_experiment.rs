//! Controlled policy inference on live robot feedback, with optional motor application.
//!
//! The normal controller still performs inference. This module owns the experiment clock,
//! command schedule, lifecycle and bounded recorder; the bus-owning loop decides whether the
//! proposed targets are held for observation only or passed through the platform safety layer.
use crate::{
    control::Step,
    control_owner::{ControlOwner, Owner},
    custom_policy::PolicyContext,
    experiment_tasks::random_id,
    intents::Intents,
};
use duck_control::{Command, Sensors, model::NUM_JOINTS, obs::BodyPose, policy::Net, safety::Limit};
use duck_ipc_proto::{
    PolicyExperimentAction as Action,
    PolicyExperimentConfig as Config, PolicyExperimentMode as Mode,
    PolicyExperimentParams as Params, PolicyExperimentPolicy as Policy,
};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufRead, BufReader, Read, Write},
    path::PathBuf,
    sync::{Arc, Mutex, mpsc::{self, SyncSender, TrySendError}},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const ROOT: &str = "/var/lib/robotd/policy-experiments";
const MAX_SESSIONS: usize = 8;
const MAX_SEGMENTS: usize = 256;
const MAX_DURATION_MS: u64 = 30 * 60 * 1000;
const RECORD_BUFFER: usize = 64;
const INITIALIZE_TIMEOUT_SECS: u64 = 30;
const READY_TIMEOUT_SECS: u64 = 60;

#[derive(Clone, Serialize)]
struct Session {
    id: String,
    created_unix_s: u64,
    config: Config,
    policy_identity: Value,
    state: String,
    reason: Option<String>,
    frames: u64,
    result_bytes: Option<u64>,
    result_sha256: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase { Initializing, Ready, Running, ReturningHome }

struct Active {
    id: String,
    config: Config,
    phase: Phase,
    started: Option<Instant>,
    run_token: Option<String>,
    outcome: String,
    reason: String,
    recorder: Option<Recorder>,
    phase_at: Instant,
    action_segment: Option<usize>,
}

struct Inner {
    sessions: BTreeMap<String, Session>,
    active: Option<Active>,
    latest: Option<(Sensors, Instant)>,
}

pub struct PolicyExperiment {
    root: PathBuf,
    inner: Arc<Mutex<Inner>>,
    owner: Arc<ControlOwner>,
}

#[derive(Debug, Clone, Copy)]
pub struct Drive {
    pub command: Command,
    pub policy: Option<Net>,
    pub action: Option<Action>,
    pub inference_only: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct Availability {
    pub walk: bool,
    pub stand: bool,
    pub sitstand: bool,
    pub ground_pick: bool,
    pub kick_left: bool,
    pub kick_right: bool,
    pub roulade: bool,
}

impl Availability {
    fn action(self, action: Action) -> bool {
        match action {
            Action::SitToggle => self.sitstand,
            Action::GroundPick => self.ground_pick,
            Action::KickLeft => self.kick_left,
            Action::KickRight => self.kick_right,
            Action::Roulade => self.roulade,
        }
    }
}

#[derive(Serialize)]
struct RecordFrame {
    r#type: &'static str,
    frame_index: u64,
    elapsed_ms: f64,
    tick: u64,
    sample_monotonic_us: u64,
    gateway_tick_ms: u32,
    ack_command_seq: u16,
    requested_command: WireCommand,
    base_command: WireCommand,
    policy_context: PolicyContext,
    policy_label: String,
    policy_targets: [f64; NUM_JOINTS],
    applied_targets: [f64; NUM_JOINTS],
    applied: bool,
    limits: Vec<String>,
    positions: [f64; NUM_JOINTS],
    velocities: [f64; NUM_JOINTS],
    motor_torques_nm: [f64; NUM_JOINTS],
    motor_flags: [u8; NUM_JOINTS],
    motor_temperatures_c: [f64; NUM_JOINTS],
    motor_feedback_age_ms: [u32; NUM_JOINTS],
    attitude_rpy: [f64; 3],
    imu_gyro: [f64; 3],
    imu_gravity: [f64; 3],
    imu_quat: [f64; 4],
}

#[derive(Serialize)]
struct WireCommand {
    twist: [f64; 3],
    head: [f64; 4],
    body: [f64; 3],
}

impl From<Command> for WireCommand {
    fn from(value: Command) -> Self {
        Self {
            twist: value.twist,
            head: value.head,
            body: [value.body.z, value.body.roll, value.body.pitch],
        }
    }
}

#[derive(Default)]
struct Finish {
    outcome: String,
    reason: String,
    frames: u64,
}

struct Recorder {
    tx: SyncSender<RecordFrame>,
    finish: Arc<Mutex<Finish>>,
}

impl PolicyExperiment {
    pub fn new(owner: Arc<ControlOwner>) -> Self {
        Self::at(PathBuf::from(ROOT), owner)
    }

    fn at(root: PathBuf, owner: Arc<ControlOwner>) -> Self {
        let sessions = load_sessions(&root);
        Self {
            root,
            owner,
            inner: Arc::new(Mutex::new(Inner {
                sessions,
                active: None,
                latest: None,
            })),
        }
    }

    pub fn active(&self) -> bool { self.inner.lock().unwrap().active.is_some() }

    pub fn phase(&self) -> Option<&'static str> {
        self.inner.lock().unwrap().active.as_ref().map(|active| match active.phase {
            Phase::Initializing => "policy_experiment_initializing",
            Phase::Ready => "policy_experiment_ready",
            Phase::Running => match active.config.mode {
                Mode::InferenceOnly => "policy_experiment_inference",
                Mode::ClosedLoop => "policy_experiment_closed_loop",
            },
            Phase::ReturningHome => "policy_experiment_stopping",
        })
    }

    pub fn request(
        &self,
        params: &Params,
        intents: &Intents,
        available: Availability,
        policy_identity: Value,
    ) -> Result<Value, String> {
        match params {
            Params::Capabilities {} => {
                let control_owner = self.owner.current().map(Owner::as_str);
                let inner = self.inner.lock().unwrap();
                let latest = inner.latest.as_ref().filter(|(_, at)| at.elapsed().as_millis() < 100)
                    .map(|(sensors, _)| json!({
                        "positions": sensors.positions,
                        "velocities": sensors.velocities,
                        "motor_flags": sensors.motor_flags,
                        "motor_temperatures_c": sensors.motor_temperatures_c,
                        "attitude_rpy": sensors.attitude_rpy,
                        "imu_gravity": sensors.imu.gravity,
                    }));
                Ok(json!({
                    "version": 2,
                    "period_ms": 20,
                    "modes": ["inference_only", "closed_loop"],
                    "policies": {"auto":available.walk,"walk":available.walk,"stand":available.stand},
                    "actions": {
                        "sit_toggle": available.sitstand,
                        "ground_pick": available.ground_pick,
                        "kick_left": available.kick_left,
                        "kick_right": available.kick_right,
                        "roulade": available.roulade,
                    },
                    "policy_identity": policy_identity,
                    "max_segments": MAX_SEGMENTS,
                    "max_duration_ms": MAX_DURATION_MS,
                    "initial_pose": "saved_home",
                    "current": latest,
                    "control_owner": control_owner,
                }))
            }
            Params::List {} => {
                let control_owner = self.owner.current().map(Owner::as_str);
                let inner = self.inner.lock().unwrap();
                Ok(json!({
                    "sessions": inner.sessions.values().collect::<Vec<_>>(),
                    "limits": {"max_sessions":MAX_SESSIONS,"max_segments":MAX_SEGMENTS,"max_duration_ms":MAX_DURATION_MS},
                    "control_owner": control_owner,
                }))
            }
            Params::Configure { config } => self.configure(config.clone(), available, policy_identity),
            Params::Status { id } => self.status(id),
            Params::Initialize { id } => {
                self.initialize(id, &policy_identity)?;
                intents.stop();
                intents.set_enabled(false);
                intents.request_init();
                self.status(id)
            }
            Params::Start { id } => self.start(id),
            Params::Stop { id, run_token } => {
                self.stop(id, run_token.as_deref(), "operator requested policy experiment stop")?;
                self.status(id)
            }
            Params::Download { .. } => Err("download requires streaming handler".into()),
            Params::Delete { id, sha256 } => self.delete(id, sha256),
        }
    }

    fn configure(&self, config: Config, available: Availability, policy_identity: Value) -> Result<Value, String> {
        validate(&config)?;
        let supported = match config.policy { Policy::Auto|Policy::Walk=>available.walk, Policy::Stand=>available.stand };
        if !supported { return Err("所选策略槽位当前没有可运行的策略".into()); }
        if let Some(action) = config.segments.iter().find_map(|segment| segment.action.filter(|action| !available.action(*action))) {
            return Err(format!("动作 {action:?} 当前没有可运行的策略"));
        }
        let mut inner = self.inner.lock().unwrap();
        if inner.sessions.len() >= MAX_SESSIONS {
            return Err("最多保留 8 次策略实验；请先下载并确认删除旧结果".into());
        }
        let id = random_id()?;
        let session = Session {
            id: id.clone(),
            created_unix_s: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
            config,
            policy_identity,
            state: "configured".into(),
            reason: None,
            frames: 0,
            result_bytes: None,
            result_sha256: None,
        };
        inner.sessions.insert(id.clone(), session);
        Ok(json!({"id":id,"state":"configured"}))
    }

    fn initialize(&self, id: &str, policy_identity: &Value) -> Result<(), String> {
        let newly_acquired = self.owner.acquire(Owner::PolicyExperiment)?;
        let result = (|| {
            let mut inner = self.inner.lock().unwrap();
            if inner.active.is_some() { return Err("已有策略实验正在初始化或运行".into()); }
            let session = inner.sessions.get_mut(id).ok_or("策略实验不存在")?;
            if session.state != "configured" { return Err("仅 configured 实验可以初始化".into()); }
            if &session.policy_identity != policy_identity {
                return Err("策略模型已在配置后变更；请重新配置实验".into());
            }
            session.state = "initializing".into();
            session.reason = None;
            let config = session.config.clone();
            inner.active = Some(Active {
                id: id.into(), config, phase: Phase::Initializing, started: None,
                run_token: None, outcome: "stopped".into(), reason: "stopped".into(), recorder: None,
                phase_at: Instant::now(),
                action_segment: None,
            });
            Ok(())
        })();
        if result.is_err() && newly_acquired { self.owner.release(Owner::PolicyExperiment); }
        result
    }

    fn start(&self, id: &str) -> Result<Value, String> {
        let token = random_id()?;
        let (config, identity, path) = {
            let inner = self.inner.lock().unwrap();
            let active = inner.active.as_ref().ok_or("请先初始化策略实验")?;
            if active.id != id || active.phase != Phase::Ready { return Err("策略实验尚未就绪".into()); }
            let identity=inner.sessions.get(id).unwrap().policy_identity.clone();
            (active.config.clone(), identity, self.root.join(format!("{id}.jsonl")))
        };
        fs::create_dir_all(&self.root).map_err(|e| e.to_string())?;
        let recorder = Recorder::start(path, id.into(), config, identity, Arc::clone(&self.inner))?;
        let mut inner = self.inner.lock().unwrap();
        let active = inner.active.as_mut().ok_or("策略实验不再有效")?;
        if active.id != id || active.phase != Phase::Ready { return Err("策略实验状态已变化".into()); }
        active.phase = Phase::Running;
        active.started = Some(Instant::now());
        active.phase_at = Instant::now();
        active.run_token = Some(token.clone());
        active.recorder = Some(recorder);
        active.action_segment = None;
        inner.sessions.get_mut(id).unwrap().state = "running".into();
        Ok(json!({"id":id,"state":"running","run_token":token}))
    }

    fn stop(&self, id: &str, token: Option<&str>, reason: &str) -> Result<(), String> {
        let mut inner = self.inner.lock().unwrap();
        let active = inner.active.as_mut().ok_or("没有活动的策略实验")?;
        if active.id != id { return Err("另一个策略实验占用控制".into()); }
        if active.phase == Phase::Running && active.run_token.as_deref() != token {
            return Err("run token 不属于当前策略实验".into());
        }
        active.phase = Phase::ReturningHome;
        active.phase_at = Instant::now();
        active.outcome = "stopped".into();
        active.reason = reason.into();
        let session_id = active.id.clone();
        inner.sessions.get_mut(&session_id).unwrap().state = "returning_home".into();
        Ok(())
    }

    pub fn abort(&self, reason: &str) {
        self.finish_active("stopped", reason);
    }

    pub fn before_policy(
        &self,
        sensors: Option<&Sensors>,
        healthy: bool,
        imu_ready: bool,
        homed: bool,
        at_home: bool,
        fallen: bool,
    ) -> Option<Drive> {
        let mut finish: Option<(String, String)> = None;
        let mut result = None;
        {
            let mut inner = self.inner.lock().unwrap();
            inner.latest = sensors.map(|value| (*value, Instant::now()));
            let Some(mut active) = inner.active.take() else { return None };
            match active.phase {
                Phase::Initializing => {
                    if healthy && imu_ready && homed && at_home {
                        active.phase = Phase::Ready;
                        active.phase_at = Instant::now();
                        inner.sessions.get_mut(&active.id).unwrap().state = "ready".into();
                    } else if active.phase_at.elapsed().as_secs() >= INITIALIZE_TIMEOUT_SECS {
                        finish = Some(("failed".into(), "initialization did not reach a healthy home pose within 30 seconds".into()));
                    }
                }
                Phase::Ready => {
                    if !healthy || !homed { finish = Some(("failed".into(), "hardware became unavailable while ready".into())); }
                    else if active.phase_at.elapsed().as_secs() >= READY_TIMEOUT_SECS {
                        finish = Some(("stopped".into(), "ready state expired before start".into()));
                    }
                }
                Phase::Running => {
                    if !healthy { active.phase = Phase::ReturningHome; active.phase_at=Instant::now(); active.outcome="failed".into(); active.reason="feedback, IMU or motor control fault".into(); }
                    else if fallen && active.config.stop_on_fall { active.phase=Phase::ReturningHome; active.phase_at=Instant::now(); active.outcome="failed".into(); active.reason="fall detected".into(); }
                    else if sensors.is_some_and(|s| s.motor_temperatures_c.iter().copied().fold(0.0, f64::max) >= active.config.max_temperature_c) {
                        active.phase=Phase::ReturningHome; active.phase_at=Instant::now(); active.outcome="failed".into(); active.reason="motor temperature limit".into();
                    } else {
                        let elapsed = active.started.unwrap().elapsed().as_millis() as u64;
                        if let Some((segment_index, command, scheduled_action)) = command_at(&active.config, elapsed) {
                            let action = if active.action_segment == Some(segment_index) {
                                None
                            } else {
                                active.action_segment = Some(segment_index);
                                scheduled_action
                            };
                            result = Some(Drive {
                                command,
                                policy: match active.config.policy { Policy::Auto=>None, Policy::Walk=>Some(Net::Walk), Policy::Stand=>Some(Net::Stand) },
                                action,
                                inference_only: active.config.mode == Mode::InferenceOnly,
                            });
                        } else {
                            active.phase=Phase::ReturningHome; active.phase_at=Instant::now(); active.outcome="completed".into(); active.reason="configured duration completed".into();
                        }
                    }
                    if active.phase == Phase::ReturningHome {
                        inner.sessions.get_mut(&active.id).unwrap().state = "returning_home".into();
                    }
                }
                Phase::ReturningHome => {
                    if at_home { finish = Some((active.outcome.clone(), active.reason.clone())); }
                }
            }
            inner.active = Some(active);
        }
        if let Some((outcome, reason)) = finish {
            self.finish_active(&outcome, &reason);
        }
        result
    }

    pub fn inference_failed(&self, reason: &str) {
        self.fail_running(&format!("policy inference failed: {reason}"));
    }

    pub fn action_failed(&self, action: Action, reason: &str) {
        self.fail_running(&format!("policy action {action:?} was refused: {reason}"));
    }

    fn fail_running(&self, reason: &str) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(active) = inner.active.as_mut() {
            active.phase = Phase::ReturningHome;
            active.phase_at = Instant::now();
            active.outcome = "failed".into();
            active.reason = reason.into();
            let id = active.id.clone();
            inner.sessions.get_mut(&id).unwrap().state = "returning_home".into();
        }
    }

    pub fn record(
        &self,
        requested: Command,
        step: &Step,
        applied_targets: [f64; NUM_JOINTS],
        applied: bool,
        limits: &[Limit],
        sensors: &Sensors,
        tick: u64,
        time_us: u64,
    ) {
        let mut overflow = false;
        {
            let mut inner = self.inner.lock().unwrap();
            let Some(active) = inner.active.take() else { return };
            if active.phase != Phase::Running { inner.active=Some(active); return; }
            let elapsed_ms = active.started.unwrap().elapsed().as_secs_f64() * 1000.0;
            let Some(recorder) = active.recorder.as_ref() else { inner.active=Some(active); return };
            let frame_index = inner.sessions.get(&active.id).map_or(0, |s| s.frames);
            let frame = RecordFrame {
                r#type:"frame", frame_index, elapsed_ms, tick, sample_monotonic_us:time_us,
                gateway_tick_ms:sensors.gateway_tick_ms, ack_command_seq:sensors.ack_command_seq,
                requested_command:requested.into(), base_command:step.base_command.into(),
                policy_context:step.policy_context,
                policy_label:step.label.into(),
                policy_targets:step.targets, applied_targets, applied,
                limits:limits.iter().map(|v| format!("{v:?}").to_lowercase()).collect(),
                positions:sensors.positions, velocities:sensors.velocities,
                motor_torques_nm:sensors.motor_torques_nm, motor_flags:sensors.motor_flags,
                motor_temperatures_c:sensors.motor_temperatures_c,
                motor_feedback_age_ms:sensors.motor_feedback_age_ms, attitude_rpy:sensors.attitude_rpy,
                imu_gyro:sensors.imu.gyro, imu_gravity:sensors.imu.gravity, imu_quat:sensors.imu.quat,
            };
            match recorder.tx.try_send(frame) {
                Ok(()) => inner.sessions.get_mut(&active.id).unwrap().frames += 1,
                Err(TrySendError::Full(_)|TrySendError::Disconnected(_)) => overflow = true,
            }
            inner.active = Some(active);
        }
        if overflow { self.inference_failed("policy experiment recorder queue unavailable"); }
    }

    fn finish_active(&self, outcome: &str, reason: &str) {
        let active = {
            let mut inner = self.inner.lock().unwrap();
            let Some(active) = inner.active.take() else { return };
            if let Some(session) = inner.sessions.get_mut(&active.id) {
                session.state = "finalizing".into();
                session.reason = Some(reason.into());
            }
            active
        };
        if let Some(recorder) = active.recorder {
            let frames = self.inner.lock().unwrap().sessions.get(&active.id).map_or(0, |s| s.frames);
            *recorder.finish.lock().unwrap() = Finish { outcome:outcome.into(), reason:reason.into(), frames };
            drop(recorder);
        } else {
            let mut inner = self.inner.lock().unwrap();
            if let Some(session) = inner.sessions.get_mut(&active.id) { session.state=outcome.into(); }
        }
        self.owner.release(Owner::PolicyExperiment);
    }

    fn status(&self, id: &str) -> Result<Value, String> {
        let inner = self.inner.lock().unwrap();
        let session = inner.sessions.get(id).ok_or("策略实验不存在")?;
        serde_json::to_value(session).map_err(|e| e.to_string())
    }

    pub fn download(&self, id: &str) -> Result<(File, u64, String), String> {
        let inner = self.inner.lock().unwrap();
        let session = inner.sessions.get(id).ok_or("策略实验不存在")?;
        let bytes = session.result_bytes.ok_or("策略实验结果尚未完成")?;
        let sha = session.result_sha256.clone().ok_or("策略实验结果尚未完成")?;
        let file = File::open(self.root.join(format!("{id}.jsonl"))).map_err(|e| e.to_string())?;
        Ok((file, bytes, sha))
    }

    fn delete(&self, id: &str, sha: &str) -> Result<Value, String> {
        let mut inner = self.inner.lock().unwrap();
        if inner.active.as_ref().is_some_and(|active| active.id == id) { return Err("活动实验不能删除".into()); }
        let session = inner.sessions.get(id).ok_or("策略实验不存在")?;
        match session.result_sha256.as_deref() {
            Some(expected) if expected == sha => {
                fs::remove_file(self.root.join(format!("{id}.jsonl"))).map_err(|e| e.to_string())?;
            }
            None if sha.is_empty() => {}
            _ => return Err("结果 SHA-256 不匹配".into()),
        }
        inner.sessions.remove(id);
        Ok(json!({"id":id,"deleted":true}))
    }
}

impl Recorder {
    fn start(path: PathBuf, id: String, config: Config, policy_identity: Value, inner: Arc<Mutex<Inner>>) -> Result<Self, String> {
        let mut file = File::create(&path).map_err(|e| e.to_string())?;
        serde_json::to_writer(&mut file, &json!({"type":"header","version":1,"id":id,"config":config,"policy_identity":policy_identity})).map_err(|e| e.to_string())?;
        file.write_all(b"\n").map_err(|e| e.to_string())?;
        let (tx, rx) = mpsc::sync_channel::<RecordFrame>(RECORD_BUFFER);
        let finish = Arc::new(Mutex::new(Finish::default()));
        let worker_finish = Arc::clone(&finish);
        std::thread::spawn(move || {
            let mut write_error = None;
            for frame in rx {
                if serde_json::to_writer(&mut file, &frame).is_err()
                    || file.write_all(b"\n").is_err()
                {
                    write_error = Some("writing policy experiment record failed".to_string());
                    break;
                }
            }
            let finish = worker_finish.lock().unwrap();
            let outcome = if write_error.is_some() { "failed" }
                else if finish.outcome.is_empty() { "interrupted" }
                else { finish.outcome.as_str() };
            let reason = write_error.as_deref().unwrap_or_else(|| if finish.reason.is_empty() {
                "recorder closed without a final control-loop outcome"
            } else { &finish.reason });
            let _ = serde_json::to_writer(&mut file, &json!({"type":"summary","outcome":outcome,"reason":reason,"frames":finish.frames}));
            let _ = file.write_all(b"\n");
            let _ = file.sync_all();
            drop(file);
            let result = file_meta(&path);
            let mut inner = inner.lock().unwrap();
            if let Some(session) = inner.sessions.get_mut(&id) {
                session.state = outcome.into();
                session.reason = Some(reason.into());
                if let Ok((bytes, sha)) = result { session.result_bytes=Some(bytes); session.result_sha256=Some(sha); }
            }
        });
        Ok(Self { tx, finish })
    }
}

fn validate(config: &Config) -> Result<(), String> {
    if config.segments.is_empty() || config.segments.len() > MAX_SEGMENTS { return Err("指令段数量须为 1..256".into()); }
    let mut duration = 0u64;
    for segment in &config.segments {
        if segment.duration_ms == 0 || segment.duration_ms % 20 != 0 { return Err("每段 duration_ms 必须为正数且是 20 ms 的倍数".into()); }
        duration = duration.checked_add(segment.duration_ms).ok_or("实验时长溢出")?;
        if segment.twist.into_iter().chain(segment.head).chain(segment.body).any(|v| !v.is_finite()) {
            return Err("实验指令不能包含 NaN/Inf".into());
        }
        if segment.twist[0].abs() > 0.3 || segment.twist[1].abs() > 0.3
            || segment.twist[2].abs() > 1.5 {
            return Err("twist 超出平台范围：vx/vy ±0.3 m/s，vyaw ±1.5 rad/s".into());
        }
        if segment.head.iter().any(|value| value.abs() > 1.5) {
            return Err("head 指令绝对值不能超过 1.5 rad".into());
        }
        if !(-0.025..=0.010).contains(&segment.body[0])
            || segment.body[1].abs() > 0.26 || segment.body[2].abs() > 0.26 {
            return Err("body 超出训练范围：z -0.025..0.010 m，roll/pitch ±0.26 rad".into());
        }
    }
    if duration > MAX_DURATION_MS { return Err("策略实验最长 30 分钟".into()); }
    if !config.max_temperature_c.is_finite() || !(20.0..=80.0).contains(&config.max_temperature_c) { return Err("max_temperature_c 必须为 20..80".into()); }
    Ok(())
}

fn command_at(config: &Config, mut elapsed_ms: u64) -> Option<(usize, Command, Option<Action>)> {
    for (index, segment) in config.segments.iter().enumerate() {
        if elapsed_ms < segment.duration_ms {
            return Some((index, Command {
                twist:segment.twist,
                head:segment.head,
                body:BodyPose { z:segment.body[0], roll:segment.body[1], pitch:segment.body[2] },
            }, segment.action));
        }
        elapsed_ms -= segment.duration_ms;
    }
    None
}

fn file_meta(path: &PathBuf) -> Result<(u64, String), String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let bytes = file.metadata().map_err(|e| e.to_string())?.len();
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 16384];
    loop {
        let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 { break; }
        hash.update(&buffer[..n]);
    }
    Ok((bytes, hash.finalize().iter().map(|b| format!("{b:02x}")).collect()))
}

fn load_sessions(root: &PathBuf) -> BTreeMap<String, Session> {
    let mut sessions = BTreeMap::new();
    let Ok(entries) = fs::read_dir(root) else { return sessions };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(id) = path.file_stem().and_then(|v| v.to_str()).map(str::to_owned) else { continue };
        if path.extension().and_then(|v| v.to_str()) != Some("jsonl")
            || id.len() != 32 || !id.bytes().all(|v| v.is_ascii_hexdigit()) { continue; }
        let Ok(file) = File::open(&path) else { continue };
        let mut lines = BufReader::new(file).lines();
        let Some(Ok(first)) = lines.next() else { continue };
        let Ok(header) = serde_json::from_str::<Value>(&first) else { continue };
        let Ok(config) = serde_json::from_value::<Config>(header["config"].clone()) else { continue };
        let mut frames = 0;
        let mut outcome = "interrupted".to_string();
        let mut reason = "robotd restarted before a final summary".to_string();
        for line in lines.map_while(Result::ok) {
            let Ok(value) = serde_json::from_str::<Value>(&line) else { continue };
            match value.get("type").and_then(Value::as_str) {
                Some("frame") => frames += 1,
                Some("summary") => {
                    outcome = value.get("outcome").and_then(Value::as_str).unwrap_or("interrupted").into();
                    reason = value.get("reason").and_then(Value::as_str).unwrap_or("missing summary reason").into();
                }
                _ => {}
            }
        }
        let Ok((bytes, sha256)) = file_meta(&path) else { continue };
        let created_unix_s = entry.metadata().ok().and_then(|m| m.modified().ok())
            .and_then(|v| v.duration_since(UNIX_EPOCH).ok()).map_or(0, |v| v.as_secs());
        sessions.insert(id.clone(), Session {
            id, created_unix_s, config, policy_identity:header.get("policy_identity").cloned().unwrap_or(Value::Null), state:outcome, reason:Some(reason), frames,
            result_bytes:Some(bytes), result_sha256:Some(sha256),
        });
        if sessions.len() >= MAX_SESSIONS { break; }
    }
    sessions
}

#[cfg(test)]
mod tests {
    use super::*;
    fn available() -> Availability {
        Availability { walk:true, stand:true, sitstand:true, ground_pick:true,
            kick_left:true, kick_right:true, roulade:true }
    }
    fn config(mode: Mode) -> Config {
        Config { mode, policy:Policy::Auto, segments:vec![duck_ipc_proto::PolicyExperimentSegment { duration_ms:40, action:None, twist:[0.1,0.0,0.0], head:[0.0;4], body:[0.0;3] }], max_temperature_c:60.0, stop_on_fall:true }
    }
    #[test]
    fn schedule_has_a_hard_end() {
        assert_eq!(command_at(&config(Mode::ClosedLoop), 0).unwrap().1.twist[0], 0.1);
        assert!(command_at(&config(Mode::ClosedLoop), 40).is_none());
    }
    #[test]
    fn a_segment_action_is_emitted_only_on_its_first_control_frame() {
        let root=tempfile::tempdir().unwrap();
        let experiment=PolicyExperiment::at(root.path().into(),Arc::new(ControlOwner::default()));
        let mut configured=config(Mode::InferenceOnly);
        configured.segments[0].action=Some(Action::KickLeft);
        let id=experiment.configure(configured,available(),json!({})).unwrap()["id"].as_str().unwrap().to_owned();
        experiment.initialize(&id,&json!({})).unwrap();
        experiment.before_policy(Some(&Sensors::default()),true,true,true,true,false);
        experiment.start(&id).unwrap();
        assert_eq!(experiment.before_policy(Some(&Sensors::default()),true,true,true,true,false).unwrap().action,Some(Action::KickLeft));
        assert_eq!(experiment.before_policy(Some(&Sensors::default()),true,true,true,true,false).unwrap().action,None);
        experiment.abort("test complete");
    }
    #[test]
    fn configuration_rejects_an_unavailable_scheduled_action() {
        let root=tempfile::tempdir().unwrap();
        let experiment=PolicyExperiment::at(root.path().into(),Arc::new(ControlOwner::default()));
        let mut configured=config(Mode::ClosedLoop);
        configured.segments[0].action=Some(Action::Roulade);
        let mut capabilities=available();
        capabilities.roulade=false;
        assert!(experiment.configure(configured,capabilities,json!({})).unwrap_err().contains("Roulade"));
    }
    #[test]
    fn initialization_claims_the_shared_owner() {
        let root=tempfile::tempdir().unwrap();
        let owner=Arc::new(ControlOwner::default());
        let experiment=PolicyExperiment::at(root.path().into(), owner.clone());
        let id=experiment.configure(config(Mode::InferenceOnly),available(),json!({})).unwrap()["id"].as_str().unwrap().to_owned();
        experiment.initialize(&id,&json!({})).unwrap();
        assert_eq!(owner.current(),Some(Owner::PolicyExperiment));
        assert!(owner.acquire(Owner::MotorExperiment).is_err());
        experiment.abort("test");
        assert_eq!(owner.current(),None);
    }
    #[test]
    fn ready_start_and_tokened_stop_keep_the_lease_until_home() {
        let root=tempfile::tempdir().unwrap();
        let owner=Arc::new(ControlOwner::default());
        let experiment=PolicyExperiment::at(root.path().into(), owner.clone());
        let id=experiment.configure(config(Mode::ClosedLoop),available(),json!({})).unwrap()["id"].as_str().unwrap().to_owned();
        experiment.initialize(&id,&json!({})).unwrap();
        experiment.before_policy(Some(&Sensors::default()),true,true,true,true,false);
        assert_eq!(experiment.status(&id).unwrap()["state"],"ready");
        let started=experiment.start(&id).unwrap();
        let token=started["run_token"].as_str().unwrap();
        assert!(experiment.stop(&id,Some("wrong"),"test").is_err());
        experiment.stop(&id,Some(token),"test").unwrap();
        assert_eq!(owner.current(),Some(Owner::PolicyExperiment));
        experiment.before_policy(Some(&Sensors::default()),true,true,true,true,false);
        assert_eq!(owner.current(),None);
    }
    #[test]
    fn closed_loop_rejects_out_of_range_commands() {
        let mut value=config(Mode::ClosedLoop);
        value.segments[0].twist[0]=0.31;
        assert!(validate(&value).unwrap_err().contains("twist"));
    }
    #[test]
    fn initialization_refuses_a_policy_changed_after_configuration() {
        let root=tempfile::tempdir().unwrap();
        let owner=Arc::new(ControlOwner::default());
        let experiment=PolicyExperiment::at(root.path().into(), owner.clone());
        let id=experiment.configure(config(Mode::InferenceOnly),available(),json!({"walk":"v1"})).unwrap()["id"].as_str().unwrap().to_owned();
        assert!(experiment.initialize(&id,&json!({"walk":"v2"})).unwrap_err().contains("模型已在配置后变更"));
        assert_eq!(owner.current(),None);
        assert_eq!(experiment.status(&id).unwrap()["state"],"configured");
    }
}
