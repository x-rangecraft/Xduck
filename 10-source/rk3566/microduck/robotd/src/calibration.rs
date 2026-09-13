//! Persistent motor coordinates. Only the control-loop thread applies queued changes.
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use duck_control::bus::{STM32_MOTOR_IDS, STM32_TO_CONTROL_JOINT, configured_motors_disabled};
use duck_control::io::{MotorPositionLimit, RobotIo, Sensors};
use duck_control::safety::Safety;
use duck_control::NUM_JOINTS;
#[cfg(test)]
use duck_control::DEFAULT_POSITION;
use duck_ipc_proto::{CalibrationLimit, CalibrationParams};
use serde::{Deserialize, Serialize};

pub const PATH: &str = "/var/lib/robotd/motor-calibration.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub version: u8,
    pub limits: Vec<CalibrationLimit>,
    pub home: [f64; NUM_JOINTS],
}

impl Default for Config {
    fn default() -> Self {
        Self {
            version: 1,
            limits: STM32_MOTOR_IDS
                .iter()
                .filter(|&&id| id != 0)
                .map(|&id| CalibrationLimit {
                    motor_id: id,
                    min_rad: -3.0,
                    max_rad: 3.0,
                })
                .collect(),
            home: [0.0; NUM_JOINTS],
        }
    }
}

impl Config {
    fn validate(&self) -> Result<(), String> {
        if self.version != 1 {
            return Err("标定文件版本不支持".into());
        }
        let ids: Vec<_> = STM32_MOTOR_IDS
            .iter()
            .copied()
            .filter(|&id| id != 0)
            .collect();
        if self.limits.len() != ids.len() {
            return Err("限位必须包含全部已配置电机".into());
        }
        let mut seen = Vec::new();
        for l in &self.limits {
            if !ids.contains(&l.motor_id) || seen.contains(&l.motor_id) {
                return Err("限位包含重复或未知电机 ID".into());
            }
            seen.push(l.motor_id);
            if !l.min_rad.is_finite()
                || !l.max_rad.is_finite()
                || l.min_rad < -12.5
                || l.max_rad > 12.5
                || (l.min_rad * 1000.0).ceil() >= (l.max_rad * 1000.0).floor()
            {
                return Err("限位应在 ±12.5 rad 内，且下限至少比上限小 0.001 rad".into());
            }
        }
        if self.home.iter().any(|x| !x.is_finite() || x.abs() > 12.5) {
            return Err("默认站姿必须是有限弧度值，范围 ±12.5 rad".into());
        }
        Ok(())
    }

    pub fn home_error(&self) -> Option<String> {
        for (&id, joint) in STM32_MOTOR_IDS.iter().zip(STM32_TO_CONTROL_JOINT) {
            if let Some(l) = self.limits.iter().find(|l| l.motor_id == id) {
                let min = (l.min_rad * 1000.0).ceil() / 1000.0;
                let max = (l.max_rad * 1000.0).floor() / 1000.0;
                if self.home[joint] < min || self.home[joint] > max {
                    return Some(format!("电机 {id} 默认站姿超出限位，请重新设置默认站姿"));
                }
            }
        }
        None
    }

    pub fn wire_limits(&self) -> Vec<MotorPositionLimit> {
        self.limits
            .iter()
            .map(|l| MotorPositionLimit {
                motor_id: l.motor_id,
                // Round inward so quantisation never widens a calibrated limit.
                min_mrad: (l.min_rad * 1000.0).ceil() as i32,
                max_mrad: (l.max_rad * 1000.0).floor() as i32,
            })
            .collect()
    }

    fn save(&self, path: &Path) -> Result<(), String> {
        use std::io::Write;
        let parent = path.parent().ok_or("无效标定文件路径")?;
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        let temp = path.with_extension("json.tmp");
        let bytes = serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?;
        let mut file = std::fs::File::create(&temp).map_err(|e| e.to_string())?;
        file.write_all(&bytes)
            .and_then(|_| file.sync_all())
            .map_err(|e| e.to_string())?;
        std::fs::rename(temp, path).map_err(|e| e.to_string())?;
        std::fs::File::open(parent)
            .and_then(|f| f.sync_all())
            .map_err(|e| e.to_string())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Status {
    /// Configured motor ID and control-joint index in ascending physical motor-ID order.
    pub motors: Vec<(u8, usize)>,
    pub config: Config,
    pub request_id: u64,
    pub busy: bool,
    pub synced: bool,
    pub editable: bool,
    pub error: Option<String>,
    pub home_error: Option<String>,
}

struct Inner {
    status: Status,
    pending: Option<CalibrationParams>,
    path: Option<PathBuf>,
    load_error: Option<String>,
    next_sync: Option<Instant>,
}

pub struct Calibration(Mutex<Inner>);

impl Default for Calibration {
    fn default() -> Self {
        Self(Mutex::new(Inner {
            status: Status {
                motors: STM32_MOTOR_IDS.iter().copied().zip(STM32_TO_CONTROL_JOINT)
                    .filter(|(id, _)| *id != 0).collect(),
                config: Config::default(),
                request_id: 0,
                busy: false,
                synced: false,
                editable: false,
                error: None,
                home_error: None,
            },
            pending: None,
            path: None,
            load_error: None,
            next_sync: None,
        }))
    }
}

impl Calibration {
    pub fn load(&self, path: PathBuf) {
        let mut inner = self.0.lock().unwrap();
        inner.path = Some(path.clone());
        let config = match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice::<Config>(&bytes).map_err(|e| e.to_string()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Config::default()),
            Err(e) => Err(e.to_string()),
        }
        .and_then(|config| {
            config.validate()?;
            Ok(config)
        });
        match config {
            Ok(config) => inner.status.config = config,
            Err(e) => {
                let error = format!("无法读取标定文件 {}：{e}", path.display());
                inner.status.error = Some(error.clone());
                inner.load_error = Some(error);
            }
        }
    }

    pub fn status(&self) -> Status {
        self.0.lock().unwrap().status.clone()
    }
    pub fn config(&self) -> Config {
        self.status().config
    }

    pub fn disconnected(&self) {
        let mut inner = self.0.lock().unwrap();
        inner.status.synced = false;
        inner.next_sync = None;
        inner.status.editable = false;
        if inner.pending.take().is_some() {
            inner.status.busy = false;
            inner.status.error = Some("USB 已断开，标定请求已取消".into());
        }
    }

    pub fn request(&self, request: &CalibrationParams) -> Result<Status, String> {
        let mut inner = self.0.lock().unwrap();
        if matches!(request, CalibrationParams::Get {}) {
            return Ok(inner.status.clone());
        }
        if let Some(error) = &inner.load_error {
            return Err(error.clone());
        }
        if inner.status.busy {
            return Err("已有标定正在处理".into());
        }
        if !inner.status.editable {
            return Err("请先放松，并等待全部已配置电机在线且失能".into());
        }
        let mut next = inner.status.config.clone();
        match request {
            CalibrationParams::SetLimits { limits } => next.limits = limits.clone(),
            CalibrationParams::SetHome { positions } => {
                next.home = positions
                    .as_slice()
                    .try_into()
                    .map_err(|_| "默认站姿需要 15 个关节值")?;
            }
            CalibrationParams::MarkZero { motor_id } => {
                if *motor_id == 0 || !STM32_MOTOR_IDS.contains(motor_id) {
                    return Err("未知电机 ID".into());
                }
            }
            CalibrationParams::Get {} => unreachable!(),
        }
        next.validate()?;
        if matches!(request, CalibrationParams::SetHome { .. })
            && let Some(error) = next.home_error()
        {
            return Err(error);
        }
        inner.status.request_id += 1;
        inner.status.busy = true;
        inner.status.error = None;
        inner.pending = Some(request.clone());
        Ok(inner.status.clone())
    }

    pub fn enable_error(&self) -> Option<String> {
        let inner = self.0.lock().unwrap();
        if let Some(error) = &inner.load_error {
            return Some(error.clone());
        }
        if inner.status.busy {
            return Some("标定尚未完成".into());
        }
        if !inner.status.synced {
            return Some("限位尚未同步到 STM32，不能使能".into());
        }
        inner.status.config.home_error()
    }

    /// Called before power requests, with actual feedback and the current bring-up state.
    pub fn tick<T: RobotIo>(&self, safety: &mut Safety<T>, sensors: Option<&Sensors>, limp: bool) {
        let (request, config, path, needs_sync) = {
            let mut inner = self.0.lock().unwrap();
            if safety.position_limits_ready() == Some(false) {
                inner.status.synced = false;
            }
            inner.status.editable = limp
                && sensors.is_some_and(|s|
                // Fakes do not report hardware flags; the real bus always reports Some.
                safety.motor_torque_enabled().is_none() || configured_motors_disabled(s));
            inner.status.home_error = inner.status.config.home_error();
            if inner.load_error.is_some() {
                inner.status.editable = false;
                return;
            }
            let request = inner.pending.take();
            if !inner.status.editable {
                if request.is_some() {
                    inner.status.busy = false;
                    inner.status.error = Some("执行时电机未确认失能，标定已拒绝".into());
                }
                return;
            }
            if request.is_none()
                && (inner.status.synced || inner.next_sync.is_some_and(|at| Instant::now() < at))
            {
                return;
            }
            inner.status.busy = true;
            (
                request,
                inner.status.config.clone(),
                inner.path.clone(),
                !inner.status.synced,
            )
        };
        let mut next = config.clone();
        let mut synced = !needs_sync;
        let result = (|| -> Result<(), String> {
            if needs_sync {
                safety
                    .set_position_limits(&config.wire_limits())
                    .map_err(|e| e.to_string())?;
                synced = true;
            }
            match &request {
                Some(CalibrationParams::SetLimits { limits }) => {
                    next.limits = limits.clone();
                    synced = false;
                    safety
                        .set_position_limits(&next.wire_limits())
                        .map_err(|e| e.to_string())?;
                    synced = true;
                }
                Some(CalibrationParams::SetHome { positions }) => {
                    next.home.copy_from_slice(positions)
                }
                Some(CalibrationParams::MarkZero { motor_id }) => {
                    safety
                        .mark_motor_zero(*motor_id)
                        .map_err(|e| e.to_string())?;
                }
                _ => {}
            }
            if matches!(
                request,
                Some(CalibrationParams::SetHome { .. } | CalibrationParams::SetLimits { .. })
            ) && let Some(path) = path.as_deref()
            {
                next.save(path)?;
            }
            Ok(())
        })();
        if result.is_err() && matches!(request, Some(CalibrationParams::SetLimits { .. })) {
            // A write or save may have failed after the firmware accepted it. Restore
            // the last persisted bounds; failed restoration prevents future enabling.
            synced = safety.set_position_limits(&config.wire_limits()).is_ok();
        }
        let mut inner = self.0.lock().unwrap();
        inner.status.synced = synced;
        inner.next_sync = (!synced).then(|| Instant::now() + Duration::from_secs(1));
        if result.is_ok() {
            inner.status.config = next;
        }
        inner.status.error = result.err();
        inner.status.home_error = inner.status.config.home_error();
        inner.status.busy = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_motor_set_and_quantized_bounds() {
        let mut config = Config::default();
        assert!(config.validate().is_ok());
        config.limits[0].max_rad = 4.0;
        assert_eq!(config.wire_limits()[0].max_mrad, 4000);
        config.limits[0].min_rad = 0.0001;
        config.limits[0].max_rad = 0.0009;
        assert!(config.validate().is_err());
        config = Config::default();
        config.limits[1].motor_id = config.limits[0].motor_id;
        assert!(config.validate().is_err());
    }
    #[test]
    fn persistence_round_trip_and_corrupt_file_blocks_enable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("calibration.json");
        let mut config = Config::default();
        config.home[0] = 0.4;
        config.save(&path).unwrap();
        let store = Calibration::default();
        store.load(path.clone());
        assert_eq!(store.config().home[0], 0.4);
        std::fs::write(&path, b"broken").unwrap();
        let store = Calibration::default();
        store.load(path);
        assert!(store.enable_error().unwrap().contains("无法读取"));
    }

    #[test]
    fn queued_edit_is_rechecked_and_default_pose_is_separate_from_zero() {
        let store = Calibration::default();
        let mut io = Safety::new(duck_control::FakeIo::new(), Default::default());
        let sensors = io.read().unwrap();
        let mut home = DEFAULT_POSITION.to_vec();
        home[0] = 0.75;
        let request = CalibrationParams::SetHome {
            positions: home.clone(),
        };
        assert!(store.request(&request).is_err()); // no disabled observation yet
        store.tick(&mut io, Some(&sensors), true);
        assert!(store.status().synced);
        assert!(store.request(&request).unwrap().busy);
        assert!(store.request(&request).is_err()); // cannot replace an in-flight change
        store.tick(&mut io, Some(&sensors), false); // enable won the race
        assert!(store.status().error.is_some());
        assert_eq!(store.config().home, [0.0; NUM_JOINTS]);
        store.tick(&mut io, Some(&sensors), true);
        store.request(&request).unwrap();
        store.tick(&mut io, Some(&sensors), true);
        assert_eq!(store.config().home[0], 0.75);
        assert!(store.status().error.is_none());
        assert_eq!(store.config().limits[0].max_rad, 3.0);
        // FakeIo refuses hardware zero; saving the home succeeded without ever zeroing it.
        store
            .request(&CalibrationParams::MarkZero { motor_id: 2 })
            .unwrap();
        store.tick(&mut io, Some(&sensors), true);
        assert!(store.status().error.is_some());
        assert_eq!(store.config().home[0], 0.75);
    }

    #[test]
    fn reduced_limits_block_init_until_home_is_valid() {
        let store = Calibration::default();
        let mut io = Safety::new(duck_control::FakeIo::new(), Default::default());
        let sensors = io.read().unwrap();
        store.tick(&mut io, Some(&sensors), true);
        let mut limits = store.config().limits;
        limits[0].min_rad = 1.0;
        limits[0].max_rad = 2.0;
        store
            .request(&CalibrationParams::SetLimits { limits })
            .unwrap();
        store.tick(&mut io, Some(&sensors), true);
        assert!(store.status().synced);
        assert!(store.enable_error().unwrap().contains("默认站姿超出限位"));
        let mut home = DEFAULT_POSITION.to_vec();
        home[0] = 1.5;
        store
            .request(&CalibrationParams::SetHome { positions: home })
            .unwrap();
        store.tick(&mut io, Some(&sensors), true);
        assert!(store.enable_error().is_none());
        store.disconnected();
        assert!(store.enable_error().is_some());
        assert!(!store.status().editable);
    }
}
