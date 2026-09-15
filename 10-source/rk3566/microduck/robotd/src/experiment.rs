//! Only the bus-owning control tick executes uploaded tasks. No network heartbeat
//! participates in task timing; disconnects never restart or extend a task.
use crate::{
    calibration::Config,
    control_owner::{ControlOwner, Owner},
    experiment_tasks::{self, Run, Store},
};
use duck_control::{
    bus::{
        STM32_MOTOR_IDS, STM32_TO_CONTROL_JOINT, configured_motors_disabled, motor_command_bounds,
        validate_motor_commands,
    },
    io::{RobotIo, Sensors},
    safety::Safety,
};
use duck_ipc_proto::{ExperimentParams, ExperimentTaskParams as Task, MotorCommand};
use serde_json::{Value, json};
use std::{
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};
struct Inner {
    task: Option<Run>,
    owned: bool,
    task_id: Option<String>,
    run_token: Option<String>,
    stopping: bool,
    ready_at: Option<(Instant, NormalState)>,
    disable_attempt: Option<Instant>,
    stop_at: Option<Instant>,
    disable_on_stop: bool,
    error: Option<String>,
    outcome: String,
    running: bool,
    latest: Option<(Sensors, Instant)>,
}
pub struct Experiment {
    inner: Mutex<Inner>,
    pub hz: u32,
    store: OnceLock<Result<Arc<Store>, String>>,
    owner: Arc<ControlOwner>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalState {
    Relaxed,
    Initialized,
    Unavailable,
}
impl Default for Experiment {
    fn default() -> Self {
        Self {
            inner: Mutex::new(Inner {
                task: None,
                owned: false,
                task_id: None,
                run_token: None,
                stopping: false,
                ready_at: None,
                disable_attempt: None,
                stop_at: None,
                disable_on_stop: false,
                error: None,
                outcome: "stopped".into(),
                running: false,
                latest: None,
            }),
            hz: 50,
            store: OnceLock::new(),
            owner: Arc::new(ControlOwner::default()),
        }
    }
}
impl Experiment {
    pub fn new(hz: u32, owner: Arc<ControlOwner>) -> Self {
        Self {
            hz,
            owner,
            ..Self::default()
        }
    }
    pub fn store(&self) -> Result<Arc<Store>, String> {
        self.store
            .get_or_init(|| Store::open(experiment_tasks::ROOT.into()).map(Arc::new))
            .clone()
    }
    pub fn active(&self) -> bool {
        self.inner.lock().unwrap().owned
    }
    pub fn returning_home(&self) -> bool {
        let g = self.inner.lock().unwrap();
        g.owned && g.stopping && !g.disable_on_stop
    }
    pub fn stop(&self, reason: &str) {
        self.halt("stopped", reason, false)
    }
    pub fn abort(&self, reason: &str) {
        self.halt("stopped", reason, true)
    }
    fn halt(&self, state: &str, reason: &str, disable: bool) {
        let mut g = self.inner.lock().unwrap();
        if !g.owned {
            return;
        }
        if g.stopping {
            if disable {
                g.disable_on_stop = true;
                g.outcome = state.into();
                g.error = Some(reason.into());
            }
            return;
        }
        g.stopping = true;
        g.disable_attempt = None;
        g.stop_at = Some(Instant::now());
        g.disable_on_stop = disable;
        g.ready_at = None;
        g.error = Some(reason.into());
        g.outcome = state.into();
    }
    pub fn request(
        &self,
        p: &ExperimentParams,
        cfg: &Config,
        _normal_enabled: bool,
    ) -> Result<Value, String> {
        if !matches!(p, ExperimentParams::Capabilities {}) {
            return Err("task operation requires async IPC handler".into());
        }
        let control_owner = self.owner.current().map(Owner::as_str);
        let g = self.inner.lock().unwrap();
        let current=g.latest.as_ref().filter(|(_,t)|t.elapsed()<Duration::from_millis(100)).map(|(s,_)|STM32_MOTOR_IDS.iter().zip(STM32_TO_CONTROL_JOINT).filter(|(id,_)|**id!=0).map(|(&id,j)|json!({"motor_id":id,"p":s.positions[j],"v":s.velocities[j],"flags":s.motor_flags[j],"temperature_c":s.motor_temperatures_c[j]})).collect::<Vec<_>>());
        Ok(
            json!({"version":4,"mode":"uploaded_tasks","period_ms":20,"task_id":g.task_id,"active":g.owned,"running":g.running,"stopping":g.stopping,"last_error":g.error,
            "control_owner":control_owner,
            "ready":!g.owned&&g.ready_at.is_some_and(|(t,_)|t.elapsed()<Duration::from_millis(100)),
            "start_states":["relaxed","initialized"],
            "end_behavior":"return_to_initialized_home","fault_behavior":"disable_all",
            "disconnect_policy":"continue","enable_scope":"all_configured","unselected":"zero_gains_velocity_and_feedforward","current_motors":current,
            "motors":cfg.limits.iter().map(|l|{let (v,t)=motor_command_bounds(l.motor_id).unwrap();json!({"motor_id":l.motor_id,"p_min":(l.min_rad*1000.0).ceil()/1000.0,"p_max":(l.max_rad*1000.0).floor()/1000.0,"v_max":v,"tau_max":t,"kp_min":0,"kp_max":500,"kd_min":0,"kd_max":5})}).collect::<Vec<_>>()}),
        )
    }
    /// Called only on a blocking IPC worker, never on the control thread.
    fn runtime_status(&self, mut value: Value) -> Value {
        let control_owner = self.owner.current().map(Owner::as_str);
        let g = self.inner.lock().unwrap();
        value["control_owner"] = json!(control_owner);
        if g.owned && value.get("id").and_then(Value::as_str) == g.task_id.as_deref() {
            value["state"] = json!(if g.stopping {
                "stopping"
            } else if g.running {
                "running"
            } else {
                "preparing"
            });
            if g.error.is_some() {
                value["reason"] = json!(g.error);
            }
        }
        value
    }
    pub fn task_request(
        &self,
        p: &Task,
        cfg: &Config,
        _normal_enabled: bool,
    ) -> Result<Value, String> {
        let store = self.store()?;
        match p {
            Task::List {} => {
                let mut result = store.list()?;
                if let Some(rows) = result["tasks"].as_array_mut() {
                    for row in rows {
                        *row = self.runtime_status(row.take());
                    }
                }
                Ok(result)
            }
            Task::UploadBegin { bytes, sha256 } => store.begin(*bytes, sha256),
            Task::UploadChunk { id, offset, data } => store.chunk(id, *offset, data),
            Task::UploadCommit { id } => store.commit(id, cfg, self.hz),
            Task::UploadAbort { id } => store.abort_upload(id),
            Task::Status { id } => store.status(id).map(|v| self.runtime_status(v)),
            Task::Delete { id, sha256 } => store.delete(id, sha256),
            Task::Download { .. } => Err("download requires streaming handler".into()),
            Task::Stop { id, run_token } => {
                let owned = {
                    let g = self.inner.lock().unwrap();
                    if g.owned && g.task_id.as_deref() != Some(id) {
                        return Err("different task owns control".into());
                    }
                    if g.owned && g.run_token.as_deref() != run_token.as_deref() {
                        return Err("run token does not own the active task".into());
                    }
                    g.owned
                };
                if owned {
                    self.stop("operator requested task stop");
                    store.status(id).map(|v| self.runtime_status(v))
                } else {
                    store.cancel_ready(id)
                }
            }
            Task::Start { id } => {
                // Generate before claiming the controller: even a broken entropy source must leave
                // the robot unowned rather than wedged in a task with no usable stop token.
                let run_token = experiment_tasks::random_id()?;
                self.owner.acquire(Owner::MotorExperiment)?;
                let already_enabled = {
                    let mut g = self.inner.lock().unwrap();
                    if g.owned
                        || !g
                            .ready_at
                            .is_some_and(|(t,_)| t.elapsed() < Duration::from_millis(100))
                    {
                        self.owner.release(Owner::MotorExperiment);
                        return Err("busy or no fresh healthy relaxed/initialized feedback".into());
                    }
                    let already_enabled = g.ready_at.is_some_and(|(_, state)| state == NormalState::Initialized);
                    g.owned = true;
                    g.task_id = Some(id.clone());
                    g.run_token = Some(run_token.clone());
                    g.stopping = false;
                    g.running = false;
                    g.error = None;
                    g.ready_at = None;
                    g.disable_attempt = None;
                    g.stop_at = None;
                    g.disable_on_stop = false;
                    already_enabled
                };
                let run = match store.prepare(id, cfg, self.hz) {
                    Ok(r) => r,
                    Err(e) => {
                        let mut g = self.inner.lock().unwrap();
                        g.owned = false;
                        g.task_id = None;
                        g.run_token = None;
                        g.stopping = false;
                        self.owner.release(Owner::MotorExperiment);
                        return Err(e);
                    }
                };
                let mut g = self.inner.lock().unwrap();
                g.task = Some(run);
                Ok(json!({"id":id,"state":"preparing","enabled":already_enabled,"run_token":run_token}))
            }
        }
    }
    pub fn tick<T: RobotIo>(
        &self,
        safety: &mut Safety<T>,
        fresh: Option<&Sensors>,
        cfg: &Config,
        normal_state: NormalState,
        tick: u64,
        time_us: u64,
    ) -> bool {
        let entered = Instant::now();
        let healthy = fresh.is_some()
            && safety.motor_control_error().is_none()
            && safety.imu_ready()
            && safety.position_limits_ready() != Some(false);
        let mut run = {
            let mut g = self.inner.lock().unwrap();
            g.latest = fresh.map(|s| (*s, Instant::now()));
            if !g.owned {
                let ready = healthy && match normal_state {
                    NormalState::Relaxed => fresh.is_some_and(configured_motors_disabled),
                    NormalState::Initialized => safety.motor_torque_enabled() == Some(true),
                    NormalState::Unavailable => false,
                };
                g.ready_at = ready.then(|| (Instant::now(), normal_state));
                return false;
            }
            g.ready_at = None;
            let Some(run) = g.task.take() else {
                return true;
            };
            run
        };
        if !healthy {
            self.halt("failed", "feedback, IMU or motor control fault", true)
        }
        if !self.inner.lock().unwrap().stopping {
            if let Err(e) = self.execute(
                &mut run,
                safety,
                fresh.unwrap(),
                cfg,
                tick,
                time_us,
                entered,
            ) {
                self.halt("failed", &e, true)
            }
        }
        if self.inner.lock().unwrap().stopping {
            let (disable_on_stop, timed_out) = {
                let g = self.inner.lock().unwrap();
                (g.disable_on_stop, g.stop_at.is_some_and(|at| at.elapsed() >= Duration::from_secs(10)))
            };
            if run.armed && !disable_on_stop && !timed_out {
                if at_home(fresh.unwrap(), &cfg.home) {
                    let mut g = self.inner.lock().unwrap();
                    g.running = false;
                    run.finish(&g.outcome, g.error.as_deref().unwrap_or("stopped"), false, true);
                    g.owned = false;
                    g.task_id = None;
                    g.run_token = None;
                    g.stopping = false;
                    g.stop_at = None;
                    self.owner.release(Owner::MotorExperiment);
                } else {
                    self.inner.lock().unwrap().task = Some(run);
                }
                // The ordinary initialized hold writes the saved home pose while the lease
                // remains held. It is the same target and gain path as robot.init.
                return false;
            }
            if timed_out && !disable_on_stop {
                let mut g = self.inner.lock().unwrap();
                g.disable_on_stop = true;
                g.outcome = "failed".into();
                g.error = Some("return to initialized home pose timed out; motors disabled".into());
            }
            let due = {
                let mut g = self.inner.lock().unwrap();
                let due = g
                    .disable_attempt
                    .is_none_or(|t| t.elapsed() >= Duration::from_secs(1));
                if due {
                    g.disable_attempt = Some(Instant::now())
                }
                due
            };
            let disabled = if due {
                safety.set_torque(false)
            } else {
                Ok(())
            };
            let mut g = self.inner.lock().unwrap();
            g.running = false;
            if disabled.is_ok() && fresh.is_some_and(configured_motors_disabled) {
                run.finish(&g.outcome, g.error.as_deref().unwrap_or("stopped"), true, false);
                g.owned = false;
                g.task_id = None;
                g.run_token = None;
                g.stopping = false;
                g.stop_at = None;
                self.owner.release(Owner::MotorExperiment);
                return true;
            }
            if let Err(e) = disabled {
                g.error = Some(e.to_string())
            }
        }
        self.inner.lock().unwrap().task = Some(run);
        true
    }
    fn execute<T: RobotIo>(
        &self,
        run: &mut Run,
        safety: &mut Safety<T>,
        sensors: &Sensors,
        cfg: &Config,
        tick: u64,
        time_us: u64,
        entered: Instant,
    ) -> Result<(), String> {
        let starting = !run.armed;
        let selected: &[MotorCommand] = if starting { &run.first.motors } else { &[] };
        for (&id, j) in STM32_MOTOR_IDS.iter().zip(STM32_TO_CONTROL_JOINT) {
            if id == 0 {
                continue;
            }
            let limit = cfg
                .limits
                .iter()
                .find(|l| l.motor_id == id)
                .ok_or("missing motor limit")?;
            if !sensors.positions[j].is_finite()
                || sensors.positions[j] < limit.min_rad
                || sensors.positions[j] > limit.max_rad
            {
                return Err(format!("motor {id}: measured position outside limits"));
            }
            if !sensors.motor_temperatures_c[j].is_finite()
                || sensors.motor_temperatures_c[j] >= run.header.max_temperature_c
            {
                return Err(format!("motor {id}: temperature limit"));
            }
            if starting && let Some(c) = selected.iter().find(|c| c.motor_id == id) {
                if !sensors.positions[j].is_finite()
                    || (sensors.positions[j] - c.p).abs() > run.header.start_tolerance_rad
                {
                    return Err(format!(
                        "motor {id}: first command does not match current pose"
                    ));
                }
            }
        }
        if starting {
            // Validate the saved calibration signature before enabling, not only before writes.
            run.check_config(cfg)?;
            if configured_motors_disabled(sensors) {
                safety.set_torque(true).map_err(|e| e.to_string())?;
            } else if safety.motor_torque_enabled() != Some(true) {
                return Err("start requires consistently enabled or disabled motor feedback".into());
            }
            self.inner.lock().unwrap().running = true;
            // Enabling is a synchronous admin transaction and ends at an arbitrary point inside
            // robotd's already-running 50 Hz period. Starting the task clock here permanently
            // offsets every later 20 ms row from the control ticks. Arm now, then establish time
            // zero and send frame 0 on the next ordinary tick.
            run.armed = true;
            return Ok(());
        }
        if self.inner.lock().unwrap().stopping {
            return Ok(());
        }
        let Some((row, advance)) = run.next(cfg, safety.state_frames_pace_control())? else {
            self.halt("completed", "task duration completed", false);
            return Ok(());
        };
        for command in &row.motors {
            let j = STM32_MOTOR_IDS
                .iter()
                .position(|&id| id == command.motor_id)
                .map(|i| STM32_TO_CONTROL_JOINT[i])
                .ok_or("unknown motor")?;
            if !sensors.positions[j].is_finite()
                || (sensors.positions[j] - command.p).abs() > run.header.max_tracking_error_rad
            {
                return Err(format!("motor {}: tracking error limit", command.motor_id));
            }
        }
        let full: Vec<_> = cfg
            .limits
            .iter()
            .map(|l| {
                row.motors
                    .iter()
                    .find(|c| c.motor_id == l.motor_id)
                    .cloned()
                    .unwrap_or(MotorCommand {
                        motor_id: l.motor_id,
                        p: ((l.min_rad * 1000.0).ceil() / 1000.0)
                            .max(0.0)
                            .min((l.max_rad * 1000.0).floor() / 1000.0),
                        v: 0.0,
                        tau: 0.0,
                        kp: 0.0,
                        kd: 0.0,
                    })
            })
            .collect();
        validate_motor_commands(&full, &cfg.wire_limits()).map_err(|e| e.to_string())?;
        let wire = safety
            .write_motor_commands(&full, &cfg.wire_limits())
            .map_err(|e| e.to_string())?;
        let motors:Vec<_>=STM32_MOTOR_IDS.iter().zip(STM32_TO_CONTROL_JOINT).filter(|(id,_)|**id!=0).map(|(&id,j)|json!({"motor_id":id,"p":sensors.positions[j],"v":sensors.velocities[j],"tau":sensors.motor_torques_nm[j],"temperature_c":sensors.motor_temperatures_c[j],"flags":sensors.motor_flags[j],"feedback_age_ms":sensors.motor_feedback_age_ms[j]})).collect();
        run.record(json!({"type":"cycle","frame_index":row.at_ms/20,"at_ms":row.at_ms,"tick":tick,"sample_monotonic_us":time_us,"tx_monotonic_us":time_us+entered.elapsed().as_micros() as u64,"elapsed_ms":run.started.unwrap().elapsed().as_secs_f64()*1000.0,"wire_seq":wire,"gateway_tick_ms":sensors.gateway_tick_ms,"ack_command_seq":sensors.ack_command_seq,"commands":full,"motors":motors}),advance)
    }
}

fn at_home(sensors: &Sensors, home: &[f64; duck_control::model::NUM_JOINTS]) -> bool {
    let mut count = 0;
    for (joint, &flags) in sensors.motor_flags.iter().enumerate() {
        if flags & 1 == 0 { continue; }
        count += 1;
        if flags & 0x0f != 0x07 || sensors.motor_feedback_age_ms[joint] > 500
            || !sensors.positions[joint].is_finite()
            || (sensors.positions[joint] - home[joint]).abs() > 0.1 {
            return false;
        }
    }
    count > 0
}

#[cfg(test)]
impl Experiment {
    pub(crate) fn claim_for_test(&self, cfg: &Config) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(Store::open(dir.path().into()).unwrap());
        self.store.set(Ok(store.clone())).ok().unwrap();
        let id = experiment_tasks::fixture(&store, cfg, 3, 0.0);
        // The fake has no background 50 Hz loop during filesystem fixture setup.
        assert!(self.inner.lock().unwrap().ready_at.is_some());
        self.inner.lock().unwrap().ready_at = Some((Instant::now(), NormalState::Relaxed));
        self.task_request(&Task::Start { id }, cfg, false).unwrap();
        dir
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use duck_control::io::{FakeIo, IoError, JointTargets, SlowSensors};
    use duck_control::safety::SafetyConfig;
    #[derive(Default)]
    struct Trace {
        enabled: Vec<bool>,
        commands: usize,
    }
    struct Io(Arc<Mutex<Trace>>, FakeIo);
    impl RobotIo for Io {
        fn read(&mut self) -> Result<Sensors, IoError> {
            self.1.read()
        }
        fn write(&mut self, t: &JointTargets) -> Result<(), IoError> {
            self.1.write(t)
        }
        fn set_gain(&mut self, kp: u16) -> Result<(), IoError> {
            self.1.set_gain(kp)
        }
        fn set_torque(&mut self, on: bool) -> Result<(), IoError> {
            self.0.lock().unwrap().enabled.push(on);
            self.1.set_torque(on)
        }
        fn write_motor_commands(&mut self, _: &[MotorCommand]) -> Result<u16, IoError> {
            self.0.lock().unwrap().commands += 1;
            Ok(1)
        }
        fn slow_sensors(&mut self) -> Result<SlowSensors, IoError> {
            self.1.slow_sensors()
        }
        fn motor_torque_enabled(&self) -> Option<bool> {
            self.0.lock().unwrap().enabled.last().copied()
        }
    }
    fn sample(enabled: bool) -> Sensors {
        let mut s = Sensors::default();
        for (&id, j) in STM32_MOTOR_IDS.iter().zip(STM32_TO_CONTROL_JOINT) {
            if id != 0 {
                s.motor_flags[j] = if enabled { 7 } else { 3 }
            }
        }
        s
    }
    fn setup() -> (Experiment, Safety<Io>, Arc<Mutex<Trace>>, Config) {
        let e = Experiment::new(50, Arc::new(ControlOwner::default()));
        let trace = Arc::new(Mutex::new(Trace::default()));
        let mut safety = Safety::new(Io(trace.clone(), FakeIo::new()), SafetyConfig::default());
        let cfg = Config::default();
        e.tick(&mut safety, Some(&sample(false)), &cfg, NormalState::Relaxed, 0, 0);
        (e, safety, trace, cfg)
    }
    #[test]
    fn stop_before_first_tick_never_enables() {
        let (e, mut safety, trace, cfg) = setup();
        let _dir = e.claim_for_test(&cfg);
        let id = e.inner.lock().unwrap().task_id.clone().unwrap();
        e.stop("test");
        assert_eq!(
            e.task_request(&Task::Status { id }, &cfg, false).unwrap()["state"],
            "stopping"
        );
        e.tick(&mut safety, Some(&sample(false)), &cfg, NormalState::Relaxed, 1, 20_000);
        assert!(!e.active());
        assert_eq!(trace.lock().unwrap().enabled, vec![false]);
        assert_eq!(trace.lock().unwrap().commands, 0);
    }
    #[test]
    fn initialized_robot_is_ready_and_start_does_not_reenable() {
        let (e, mut safety, trace, cfg) = setup();
        safety.set_torque(true).unwrap();
        e.tick(
            &mut safety,
            Some(&sample(true)),
            &cfg,
            NormalState::Initialized,
            1,
            20_000,
        );
        assert_eq!(
            e.request(&ExperimentParams::Capabilities {}, &cfg, true)
                .unwrap()["ready"],
            true
        );
        let _dir = e.claim_for_test(&cfg);
        e.tick(
            &mut safety,
            Some(&sample(true)),
            &cfg,
            NormalState::Initialized,
            2,
            40_000,
        );
        assert_eq!(trace.lock().unwrap().enabled, vec![true]);
        assert!(e.inner.lock().unwrap().running);
    }
    #[test]
    fn active_run_rejects_duplicate_start_and_foreign_stop() {
        let (e, _safety, _trace, cfg) = setup();
        let _dir = e.claim_for_test(&cfg);
        let (id, run_token) = {
            let g = e.inner.lock().unwrap();
            (g.task_id.clone().unwrap(), g.run_token.clone().unwrap())
        };
        assert!(
            e.task_request(&Task::Start { id: id.clone() }, &cfg, false)
                .unwrap_err()
                .contains("busy")
        );
        assert!(
            e.task_request(
                &Task::Stop {
                    id: id.clone(),
                    run_token: Some("0".repeat(32)),
                },
                &cfg,
                false,
            )
            .unwrap_err()
            .contains("does not own")
        );
        assert!(e.active());
        let stopped = e
            .task_request(
                &Task::Stop {
                    id,
                    run_token: Some(run_token),
                },
                &cfg,
                false,
            )
            .unwrap();
        assert_eq!(stopped["state"], "stopping");
    }
    #[test]
    fn independent_task_finishes_without_network_and_returns_initialized() {
        let (e, mut safety, trace, cfg) = setup();
        let _dir = e.claim_for_test(&cfg);
        e.tick(&mut safety, Some(&sample(false)), &cfg, NormalState::Relaxed, 1, 20_000);
        assert_eq!(trace.lock().unwrap().enabled, vec![true]);
        assert_eq!(trace.lock().unwrap().commands, 0);
        for i in 2..=4 {
            std::thread::sleep(Duration::from_millis(20));
            e.tick(&mut safety, Some(&sample(true)), &cfg, NormalState::Relaxed, i, i * 20_000);
        }
        let mut away = sample(true);
        away.positions = cfg.home;
        away.positions[0] += 0.2;
        std::thread::sleep(Duration::from_millis(20));
        assert!(!e.tick(&mut safety, Some(&away), &cfg, NormalState::Initialized, 5, 100_000));
        assert!(e.active());
        assert!(e.returning_home());
        let mut homed = sample(true);
        homed.positions = cfg.home;
        assert!(!e.tick(&mut safety, Some(&homed), &cfg, NormalState::Initialized, 6, 120_000));
        assert!(!e.active());
        assert_eq!(trace.lock().unwrap().enabled, vec![true]);
        assert_eq!(trace.lock().unwrap().commands, 3);
    }
    #[test]
    fn initial_pose_mismatch_and_missing_feedback_never_enable() {
        for missing in [false, true] {
            let (e, mut safety, trace, cfg) = setup();
            let _dir = e.claim_for_test(&cfg);
            let mut s = sample(false);
            let route = STM32_MOTOR_IDS
                .iter()
                .position(|&id| id == 2)
                .expect("fixture motor ID 2 is configured");
            s.positions[STM32_TO_CONTROL_JOINT[route]] = 1.0;
            e.tick(
                &mut safety,
                if missing { None } else { Some(&s) },
                &cfg,
                NormalState::Relaxed,
                1,
                20_000,
            );
            assert!(!trace.lock().unwrap().enabled.contains(&true));
        }
    }
}
