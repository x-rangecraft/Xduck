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
    PolicyExperimentInitialPose as InitialPose,
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
const READY_TIMEOUT_SECS: u64 = 240;

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
    zero_written: bool,
    zero_armed: bool,
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
    pub manual: bool,
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

    pub(crate) fn at(root: PathBuf, owner: Arc<ControlOwner>) -> Self {
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

    pub fn manual_control_allowed(&self) -> bool {
        self.inner.lock().unwrap().active.as_ref().is_some_and(|active|
            active.config.mode == Mode::Free && matches!(active.phase, Phase::Ready | Phase::Running))
    }

    /// During preparation the bus loop sends explicit zero-gain MIT commands instead of holds.
    pub fn zero_torque_preparation(&self) -> bool {
        self.inner.lock().unwrap().active.as_ref().is_some_and(|active|
            active.config.initial_pose == InitialPose::ZeroTorque
                && matches!(active.phase, Phase::Initializing | Phase::Ready))
    }

    /// Called only by the bus owner. Never use measured torque to decide this mode.
    pub fn prepare_zero_torque<I: duck_control::io::RobotIo>(
        &self,
        safety: &mut duck_control::safety::Safety<I>,
        sensors: &Sensors,
        calibration: &crate::calibration::Config,
    ) -> Result<bool, String> {
        let temperature_limit = {
            let inner = self.inner.lock().unwrap();
            let Some(active) = inner.active.as_ref() else { return Ok(false) };
            active.config.max_temperature_c
        };
        if let Some(error) = calibration.feedback_limit_error(sensors, true) { return Err(error); }
        if sensors.motor_temperatures_c.iter().any(|t| !t.is_finite() || *t >= temperature_limit) {
            return Err("zero-torque preparation motor temperature limit".into());
        }
        if safety.position_limits_ready() != Some(true) {
            return Err("zero-torque preparation requires hardware position limits".into());
        }
        if duck_control::bus::configured_motors_disabled(sensors) {
            let mut inner = self.inner.lock().unwrap();
            let active = inner.active.as_mut().ok_or("zero-torque preparation ended")?;
            if active.zero_armed { return Err("zero-torque motors lost enable feedback".into()); }
            active.zero_armed = true;
            drop(inner);
            safety.set_torque(true).map_err(|e| e.to_string())?;
        } else if !duck_control::bus::configured_motors_enabled(sensors) {
            return Err("zero-torque preparation requires consistent motor enable feedback".into());
        }
        // Wait for the enable transaction's feedback before submitting MIT commands.
        if safety.motor_torque_enabled() != Some(true) { return Ok(false); }
        let commands = zero_torque_commands(calibration);
        safety.write_motor_commands(&commands, &calibration.wire_limits()).map_err(|e| e.to_string())?;
        if let Some(active) = self.inner.lock().unwrap().active.as_mut() {
            active.zero_written = duck_control::bus::configured_motors_enabled(sensors);
            active.zero_armed = true;
        }
        Ok(true)
    }

    pub fn phase(&self) -> Option<&'static str> {
        self.inner.lock().unwrap().active.as_ref().map(|active| match active.phase {
            Phase::Initializing => "policy_experiment_initializing",
            Phase::Ready => "policy_experiment_ready",
            Phase::Running => match active.config.mode {
                Mode::Free => "policy_experiment_free",
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
                let latest = self.inner.lock().unwrap().latest;
                let latest = latest.as_ref().filter(|(_, at)| at.elapsed().as_millis() < 100)
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
                    "modes": ["free", "inference_only", "closed_loop"],
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
                    "ready_timeout_s": READY_TIMEOUT_SECS,
                    "initial_pose": "saved_home",
                    "initial_poses": ["saved_home", "zero_torque"],
                    "current": latest,
                    "control_owner": control_owner,
                }))
            }
            Params::List {} => {
                let control_owner = self.owner.current().map(Owner::as_str);
                let sessions = self.inner.lock().unwrap().sessions.values().cloned().collect::<Vec<_>>();
                Ok(json!({
                    "sessions": sessions,
                    "limits": {"max_sessions":MAX_SESSIONS,"max_segments":MAX_SEGMENTS,"max_duration_ms":MAX_DURATION_MS},
                    "control_owner": control_owner,
                }))
            }
            Params::Configure { config } => self.configure(config.clone(), available, policy_identity),
            Params::Status { id } => self.status(id),
            Params::Initialize { id } => {
                self.initialize(id, &policy_identity)?;
                if !self.manual_control_allowed() {
                    intents.stop();
                    intents.set_enabled(false);
                    if !self.zero_torque_preparation() { intents.request_init(); }
                }
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
        if config.mode != Mode::Free && !supported { return Err("所选策略槽位当前没有可运行的策略".into()); }
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
            session.state = if session.config.mode == Mode::Free { "ready" } else { "initializing" }.into();
            session.reason = None;
            let config = session.config.clone();
            inner.active = Some(Active {
                id: id.into(), phase: if config.mode == Mode::Free { Phase::Ready } else { Phase::Initializing }, config, started: None,
                run_token: None, outcome: "stopped".into(), reason: "stopped".into(), recorder: None,
                phase_at: Instant::now(),
                action_segment: None,
                zero_written: false,
                zero_armed: false,
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
            if active.config.initial_pose == InitialPose::ZeroTorque {
                let fresh = inner.latest.as_ref().is_some_and(|(s, at)|
                    at.elapsed().as_millis() < 80
                        && duck_control::bus::configured_motors_enabled(s)
                        && s.motor_temperatures_c.iter().all(|t| t.is_finite() && *t < active.config.max_temperature_c));
                if !active.zero_written || !fresh { return Err("零力矩准备状态或使能反馈已失效".into()); }
            }
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
        if active.config.mode == Mode::Free {
            drop(inner);
            self.finish_active("stopped", reason);
            return Ok(());
        }
        active.phase = Phase::ReturningHome;
        active.phase_at = Instant::now();
        active.outcome = "stopped".into();
        active.reason = reason.into();
        let session_id = active.id.clone();
        inner.sessions.get_mut(&session_id).unwrap().state = "returning_home".into();
        Ok(())
    }

    pub fn finish_zero_torque_failure(&self, reason: &str) {
        self.finish_active("failed", reason);
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
            if active.config.mode == Mode::Free {
                if active.phase == Phase::Ready {
                    if active.phase_at.elapsed().as_secs() >= READY_TIMEOUT_SECS {
                        finish = Some(("stopped".into(), "ready state expired before start".into()));
                    }
                } else if active.phase == Phase::Running {
                    let elapsed = active.started.unwrap().elapsed().as_millis() as u64;
                    if elapsed >= active.config.duration_ms.unwrap() {
                        finish = Some(("completed".into(), "configured duration completed".into()));
                    } else {
                        result = Some(Drive { command: Command::default(), policy: None, action: None, inference_only: false, manual: true });
                    }
                }
            } else { match active.phase {
                Phase::Initializing => {
                    if healthy && imu_ready && homed
                        && match active.config.initial_pose {
                            InitialPose::SavedHome => at_home,
                            InitialPose::ZeroTorque => active.zero_written,
                        } {
                        active.phase = Phase::Ready;
                        active.phase_at = Instant::now();
                        inner.sessions.get_mut(&active.id).unwrap().state = "ready".into();
                    } else if active.phase_at.elapsed().as_secs() >= INITIALIZE_TIMEOUT_SECS {
                        finish = Some(("failed".into(), "preparation did not become healthy and ready within 30 seconds".into()));
                    }
                }
                Phase::Ready => {
                    if !healthy || !imu_ready || !homed { finish = Some(("failed".into(), "hardware became unavailable while ready".into())); }
                    else if active.phase_at.elapsed().as_secs() >= READY_TIMEOUT_SECS {
                        finish = Some(("stopped".into(), "ready state expired before start".into()));
                    }
                }
                Phase::Running => {
                    if !healthy || !imu_ready || !homed { active.phase = Phase::ReturningHome; active.phase_at=Instant::now(); active.outcome="failed".into(); active.reason="feedback, IMU or motor control fault".into(); }
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
                                manual: false,
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
                    if sensors.is_some_and(duck_control::bus::configured_motors_disabled) {
                        // A relaxed robot cannot return home. Release the lease only
                        // after fresh per-motor feedback confirms physical disable.
                        finish = Some(("failed".into(), format!("{}; all motors disabled before return home", active.reason)));
                    } else if at_home { finish = Some((active.outcome.clone(), active.reason.clone())); }
                }
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
        if self.manual_control_allowed() {
            self.finish_active("failed", reason);
            return;
        }
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
        let session = self.inner.lock().unwrap().sessions.get(id).cloned().ok_or("策略实验不存在")?;
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

fn zero_torque_commands(calibration: &crate::calibration::Config) -> Vec<duck_ipc_proto::MotorCommand> {
    calibration.limits.iter().map(|limit| duck_ipc_proto::MotorCommand {
        motor_id: limit.motor_id,
        // p is irrelevant with kp=0 but must still satisfy the wire position limits.
        p: ((limit.min_rad * 1000.0).ceil() / 1000.0).max(0.0)
            .min((limit.max_rad * 1000.0).floor() / 1000.0),
        v: 0.0, tau: 0.0, kp: 0.0, kd: 0.0,
    }).collect()
}

fn validate(config: &Config) -> Result<(), String> {
    if config.mode == Mode::Free {
        if config.initial_pose != InitialPose::SavedHome || config.policy != Policy::Auto || !config.segments.is_empty() {
            return Err("free 模式只接受 duration_ms，不接受初始化姿态、指定策略或指令段".into());
        }
        return match config.duration_ms {
            Some(duration) if duration > 0 && duration <= MAX_DURATION_MS && duration % 20 == 0 => Ok(()),
            _ => Err("free duration_ms 须为 20 ms 的正整数倍，最长 30 分钟".into()),
        };
    }
    if config.duration_ms.is_some() { return Err("duration_ms 仅用于 free 模式".into()); }
    if config.initial_pose == InitialPose::ZeroTorque && config.mode != Mode::ClosedLoop {
        return Err("zero_torque 启动仅支持 closed_loop；只推理模式需要保持默认站姿".into());
    }
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
    #[test]
    fn queried_sessions_are_detached_from_control_state() {
        let root = tempfile::tempdir().unwrap();
        let experiment = PolicyExperiment::at(root.path().into(), Arc::new(ControlOwner::default()));
        let configured = experiment.configure(config(Mode::InferenceOnly), available(), json!({})).unwrap();
        let id = configured["id"].as_str().unwrap().to_owned();
        let listed = experiment.request(&Params::List {}, &Intents::new(), available(), json!({})).unwrap();
        let before = experiment.status(&id).unwrap();
        experiment.inner.lock().unwrap().sessions.get_mut(&id).unwrap().state = "running".into();
        assert_eq!(listed["sessions"][0]["state"], before["state"]);
        assert_eq!(experiment.status(&id).unwrap()["state"], "running");
    }

    #[test]
    fn free_mode_preserves_control_and_finishes_without_returning_home() {
        let root = tempfile::tempdir().unwrap();
        let owner = Arc::new(ControlOwner::default());
        let experiment = PolicyExperiment::at(root.path().into(), owner.clone());
        let config: Config = serde_json::from_value(json!({"mode":"free","duration_ms":60000})).unwrap();
        let intents = Intents::new();
        intents.set_enabled(true);
        let id = experiment.configure(config, available(), json!({})).unwrap()["id"].as_str().unwrap().to_owned();
        experiment.request(&Params::Initialize { id:id.clone() }, &intents, available(), json!({})).unwrap();
        assert!(intents.snapshot().enabled);
        assert!(experiment.manual_control_allowed());
        experiment.start(&id).unwrap();
        let drive = experiment.before_policy(Some(&Sensors::default()), false, false, false, false, true).unwrap();
        assert!(drive.manual);
        assert!(experiment.manual_control_allowed());
        let step = Step { targets:[0.2;NUM_JOINTS], label:"hold", gain:0, pd:None, mit:None,
            busy:false, base_command:Command::default(), policy_context:PolicyContext::new("hold",None,false) };
        experiment.record(Command::default(), &step, [0.2;NUM_JOINTS], true, &[], &Sensors::default(), 1, 20_000);
        assert_eq!(experiment.status(&id).unwrap()["frames"], 1);
        experiment.inner.lock().unwrap().active.as_mut().unwrap().started = Some(Instant::now()-std::time::Duration::from_secs(61));
        assert!(experiment.before_policy(None, false, false, false, false, true).is_none());
        assert!(!experiment.active());
        assert_eq!(owner.current(), None);
        assert!(intents.snapshot().enabled);
    }

    #[test]
    fn free_mode_rejects_commands_and_requires_a_bounded_duration() {
        for wire in [json!({"mode":"free"}), json!({"mode":"free","duration_ms":0}),
            json!({"mode":"free","duration_ms":21}), json!({"mode":"free","duration_ms":1800020}),
            json!({"mode":"free","duration_ms":20,"policy":"stand"}),
            json!({"mode":"free","duration_ms":20,"segments":[{"duration_ms":20}]})] {
            assert!(validate(&serde_json::from_value(wire).unwrap()).is_err());
        }
    }

    fn available() -> Availability {
        Availability { walk:true, stand:true, sitstand:true, ground_pick:true,
            kick_left:true, kick_right:true, roulade:true }
    }
    fn config(mode: Mode) -> Config {
        Config { duration_ms:None, initial_pose:InitialPose::SavedHome, mode, policy:Policy::Auto, segments:vec![duck_ipc_proto::PolicyExperimentSegment { duration_ms:40, action:None, twist:[0.1,0.0,0.0], head:[0.0;4], body:[0.0;3] }], max_temperature_c:60.0, stop_on_fall:true }
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
    #[test]
    fn old_configs_default_to_saved_home_and_zero_torque_requires_closed_loop() {
        let mut wire=serde_json::to_value(config(Mode::InferenceOnly)).unwrap();
        wire.as_object_mut().unwrap().remove("initial_pose");
        assert_eq!(serde_json::from_value::<Config>(wire).unwrap().initial_pose, InitialPose::SavedHome);
        let mut c=config(Mode::InferenceOnly);
        c.initial_pose=InitialPose::ZeroTorque;
        assert!(validate(&c).is_err());
        c.mode=Mode::ClosedLoop;
        assert!(validate(&c).is_ok());
    }

    fn enabled_sample(enabled: bool) -> Sensors {
        let mut s=Sensors::default();
        for (&id,j) in duck_control::bus::STM32_MOTOR_IDS.iter().zip(duck_control::bus::STM32_TO_CONTROL_JOINT) {
            if id != 0 { s.motor_flags[j]=if enabled { 7 } else { 3 }; }
        }
        s
    }

    #[test]
    fn stopping_waits_for_feedback_then_releases_disabled_robot_away_from_home() {
        let root = tempfile::tempdir().unwrap();
        let owner = Arc::new(ControlOwner::default());
        let experiment = PolicyExperiment::at(root.path().into(), owner.clone());
        let id = experiment.configure(config(Mode::ClosedLoop), available(), json!({})).unwrap()["id"].as_str().unwrap().to_owned();
        experiment.initialize(&id, &json!({})).unwrap();
        experiment.before_policy(Some(&enabled_sample(true)), true, true, true, true, false);
        let started = experiment.start(&id).unwrap();
        experiment.stop(&id, started["run_token"].as_str(), "test stop").unwrap();
        experiment.before_policy(None, false, false, false, false, false);
        assert_eq!(owner.current(), Some(Owner::PolicyExperiment));
        let mut partial = enabled_sample(false);
        partial.motor_flags[0] = 7;
        experiment.before_policy(Some(&partial), true, true, false, false, false);
        assert!(experiment.active());
        experiment.before_policy(Some(&enabled_sample(false)), true, true, false, false, false);
        assert!(!experiment.active());
        assert_eq!(owner.current(), None);
        let deadline = Instant::now() + std::time::Duration::from_secs(2);
        while experiment.status(&id).unwrap()["state"] == "finalizing" && Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(experiment.status(&id).unwrap()["state"], "failed");
    }

    #[derive(Default)]
    struct ZeroTrace {
        enables: Vec<bool>,
        commands: Vec<Vec<duck_ipc_proto::MotorCommand>>,
    }
    struct ZeroIo(Arc<Mutex<ZeroTrace>>);
    impl duck_control::io::RobotIo for ZeroIo {
        fn read(&mut self) -> Result<Sensors,duck_control::io::IoError> { Ok(enabled_sample(true)) }
        fn write(&mut self,_: &duck_control::io::JointTargets) -> Result<(),duck_control::io::IoError> { panic!("zero preparation must not write position holds") }
        fn set_gain(&mut self,_:u16) -> Result<(),duck_control::io::IoError> { panic!("zero preparation must not set position gains") }
        fn set_torque(&mut self,on:bool) -> Result<(),duck_control::io::IoError> { self.0.lock().unwrap().enables.push(on); Ok(()) }
        fn slow_sensors(&mut self) -> Result<duck_control::io::SlowSensors,duck_control::io::IoError> {
            Ok(duck_control::io::SlowSensors { volts:0.0,temps_c:[0.0;NUM_JOINTS] })
        }
        fn motor_torque_enabled(&self) -> Option<bool> { Some(self.0.lock().unwrap().enables.last().copied().unwrap_or(true)) }
        fn position_limits_ready(&self) -> Option<bool> { Some(true) }
        fn write_motor_commands(&mut self,c:&[duck_ipc_proto::MotorCommand]) -> Result<u16,duck_control::io::IoError> {
            self.0.lock().unwrap().commands.push(c.to_vec()); Ok(1)
        }
    }

    #[test]
    fn zero_torque_enables_without_homing_and_starts_from_a_fallen_pose() {
        let root=tempfile::tempdir().unwrap();
        let experiment=PolicyExperiment::at(root.path().into(),Arc::new(ControlOwner::default()));
        let mut c=config(Mode::ClosedLoop); c.initial_pose=InitialPose::ZeroTorque; c.stop_on_fall=false;
        let id=experiment.configure(c,available(),json!({})).unwrap()["id"].as_str().unwrap().to_owned();
        let intents=Intents::new();
        experiment.request(&Params::Initialize{id:id.clone()},&intents,available(),json!({})).unwrap();
        assert!(intents.take_power_request().is_none(),"zero mode must not request the ordinary init ramp");
        let trace=Arc::new(Mutex::new(ZeroTrace::default()));
        let mut safety=duck_control::safety::Safety::new(ZeroIo(trace.clone()),duck_control::safety::SafetyConfig::default());
        let calibration=crate::calibration::Config::default();
        assert!(experiment.prepare_zero_torque(&mut safety,&enabled_sample(false),&calibration).unwrap());
        experiment.before_policy(Some(&enabled_sample(false)),true,true,true,false,true);
        assert_eq!(experiment.status(&id).unwrap()["state"],"initializing");
        assert!(experiment.start(&id).is_err());
        let mut s=enabled_sample(true); s.positions[0]=0.4; s.motor_torques_nm[0]=0.3;
        experiment.prepare_zero_torque(&mut safety,&s,&calibration).unwrap();
        experiment.before_policy(Some(&s),true,true,true,false,true);
        assert_eq!(experiment.status(&id).unwrap()["state"],"ready");
        experiment.start(&id).unwrap();
        assert!(experiment.before_policy(Some(&s),true,true,true,false,true).is_some());
        let t=trace.lock().unwrap();
        assert_eq!(t.enables,vec![true]);
        assert_eq!(t.commands.len(),2);
        for c in t.commands.iter().flatten() { assert_eq!((c.kp,c.kd,c.tau,c.v),(0.0,0.0,0.0,0.0)); }
        assert_eq!(t.commands[0].len(),calibration.limits.len());
        experiment.abort("test complete");
    }

    #[test]
    fn zero_torque_rejects_overheat_and_lost_enable_without_reenabling() {
        let root=tempfile::tempdir().unwrap();
        let experiment=PolicyExperiment::at(root.path().into(),Arc::new(ControlOwner::default()));
        let mut c=config(Mode::ClosedLoop); c.initial_pose=InitialPose::ZeroTorque;
        let id=experiment.configure(c,available(),json!({})).unwrap()["id"].as_str().unwrap().to_owned();
        experiment.initialize(&id,&json!({})).unwrap();
        let trace=Arc::new(Mutex::new(ZeroTrace::default()));
        let mut safety=duck_control::safety::Safety::new(ZeroIo(trace.clone()),duck_control::safety::SafetyConfig::default());
        let calibration=crate::calibration::Config::default();
        let mut hot=enabled_sample(true); hot.motor_temperatures_c[0]=60.0;
        assert!(experiment.prepare_zero_torque(&mut safety,&hot,&calibration).is_err());
        assert!(trace.lock().unwrap().commands.is_empty());
        experiment.prepare_zero_torque(&mut safety,&enabled_sample(true),&calibration).unwrap();
        assert!(experiment.prepare_zero_torque(&mut safety,&enabled_sample(false),&calibration).is_err());
        assert!(trace.lock().unwrap().enables.is_empty());
        experiment.abort("test complete");
    }

    #[test]
    fn saved_home_still_requires_home_and_zero_ready_expires() {
        let root=tempfile::tempdir().unwrap();
        let owner=Arc::new(ControlOwner::default());
        let experiment=PolicyExperiment::at(root.path().into(),owner.clone());
        let id=experiment.configure(config(Mode::ClosedLoop),available(),json!({})).unwrap()["id"].as_str().unwrap().to_owned();
        experiment.initialize(&id,&json!({})).unwrap();
        experiment.before_policy(Some(&enabled_sample(true)),true,true,true,false,false);
        assert!(experiment.start(&id).is_err());
        experiment.abort("test");
        let mut c=config(Mode::ClosedLoop); c.initial_pose=InitialPose::ZeroTorque;
        let id=experiment.configure(c,available(),json!({})).unwrap()["id"].as_str().unwrap().to_owned();
        experiment.initialize(&id,&json!({})).unwrap();
        let trace=Arc::new(Mutex::new(ZeroTrace::default()));
        let mut safety=duck_control::safety::Safety::new(ZeroIo(trace),duck_control::safety::SafetyConfig::default());
        experiment.prepare_zero_torque(&mut safety,&enabled_sample(true),&crate::calibration::Config::default()).unwrap();
        experiment.before_policy(Some(&enabled_sample(true)),true,true,true,false,true);
        experiment.inner.lock().unwrap().active.as_mut().unwrap().phase_at=Instant::now()-std::time::Duration::from_secs(61);
        experiment.before_policy(Some(&enabled_sample(true)),true,true,true,false,true);
        assert_eq!(experiment.status(&id).unwrap()["state"],"ready","manual posing may take longer than one minute");
        experiment.inner.lock().unwrap().active.as_mut().unwrap().phase_at=Instant::now()-std::time::Duration::from_secs(READY_TIMEOUT_SECS);
        experiment.before_policy(Some(&enabled_sample(true)),true,true,true,false,true);
        assert!(!experiment.active());
        assert_eq!(owner.current(),None);
        assert!(experiment.start(&id).is_err());
    }

}
