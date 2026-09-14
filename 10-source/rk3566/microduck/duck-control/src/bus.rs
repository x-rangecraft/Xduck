//! STM32 DM motor gateway over USB CDC.
//!
//! Uses the `0x4D47` v4 wire format. STM32 publishes 14 fixed motor-route slots plus a
//! fused IMU observation at 50 Hz. Mouth is joint number 15 and uses a separate servo, so it
//! has no DM motor ID and is skipped by the mapping below.

use std::io::{Read, Write};
use std::time::{Duration, Instant};

use crate::imu::ImuData;
use crate::io::{MotorPositionLimit, ImuStale, IoError, JointTargets, Result, RobotIo, Sensors, SlowSensors};
use crate::model::{JOINT_NAMES, NUM_JOINTS};

const MAGIC: u16 = 0x4D47;
const VERSION: u8 = 4;
const HEADER_LEN: usize = 18;
const MAX_PAYLOAD: usize = 1536;
const MSG_COMMAND: u8 = 1;
const MSG_STATE: u8 = 2;
const MSG_ADMIN: u8 = 12;
const MSG_ADMIN_RESULT: u8 = 13;
const MODE_MIT: u8 = 1;
const MOTOR_COMMAND_LEN: usize = 20;
const MOTOR_OBSERVATION_LEN: usize = 20;
const COMMAND_PREFIX_LEN: usize = 8;
const STATE_PREFIX_LEN: usize = 64;
const IO_TIMEOUT: Duration = Duration::from_millis(60);
const ADMIN_TIMEOUT: Duration = Duration::from_millis(250);
const IMU_SENSOR_OK: u8 = 1 << 0;
const IMU_CALIBRATED: u8 = 1 << 1;
const STM32_MOTOR_COUNT: usize = 14;
/// DM motor IDs 1–14 use the same order as control joints 1–14. Mouth is joint 15 and absent.
pub const STM32_TO_CONTROL_JOINT: [usize; STM32_MOTOR_COUNT] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
];
/// All fourteen GF43X40-10 motors participate in administration and safety checks.
const CONFIGURED_MOTOR_IDS: [u8; STM32_MOTOR_COUNT] = STM32_MOTOR_IDS;
/// Compatible STM32 firmware advertises sparse-command and enable-list support.
/// Older v4 firmware remains read-only even though motor control is now permitted.
const CONTROL_CAPABILITIES: u16 = 0x0003;
const FAULT_DIAGNOSTICS_CAPABILITY: u16 = 0x0010;
pub const STM32_MOTOR_IDS: [u8; STM32_MOTOR_COUNT] = [
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
];

#[derive(Debug)]
struct Frame { msg_type: u8, payload: Vec<u8> }

#[derive(Default)]
struct StaleImuTracker {
    last_sequence: Option<u32>,
    stale: ImuStale,
}

impl StaleImuTracker {
    fn observe(&mut self, sequence: u32) {
        if self.last_sequence.replace(sequence) == Some(sequence) {
            self.stale.total = self.stale.total.saturating_add(1);
            self.stale.run = self.stale.run.saturating_add(1);
        } else {
            self.stale.run = 0;
        }
    }
}

/// The old public name is retained so robotd's hardware abstraction stays stable.
/// This type now owns an STM32 USB CDC port rather than a Dynamixel controller.
pub struct DynamixelIo {
    port: Box<dyn serialport::SerialPort>,
    rx: Vec<u8>,
    frame_seq: u16,
    command_seq: u16,
    kp_centi: u16,
    started: Instant,
    stale_imu: StaleImuTracker,
    imu_sensor_ok: bool,
    imu_ready: bool,
    control_capabilities: u16,
    gateway_faults: u32,
    last_motor_state: Option<(Sensors, Instant)>,
    last_temps_c: [f64; NUM_JOINTS],
}

impl DynamixelIo {
    pub fn open(path: &str) -> Result<Self> {
        Self::open_with_imu(path, "")
    }

    /// `imu_path` is retained as the reserved second-IMU interface and is currently ignored.
    /// The primary BMI088 data is part of each STM32 USB state frame in DMUSB v4.
    pub fn open_with_imu(path: &str, _imu_path: &str) -> Result<Self> {
        let port = serialport::new(path, 1_000_000).timeout(IO_TIMEOUT).open()
            .map_err(|e| IoError::Port { path: path.to_owned(), source: std::io::Error::other(e) })?;
        Ok(Self::from_port(port))
    }

    fn from_port(port: Box<dyn serialport::SerialPort>) -> Self {
        Self { port, rx: Vec::with_capacity(2048), frame_seq: 0,
            command_seq: 0, kp_centi: 20_000, started: Instant::now(),
            stale_imu: StaleImuTracker::default(), imu_sensor_ok: false, imu_ready: false,
            control_capabilities: 0, gateway_faults: 0, last_motor_state: None,
            last_temps_c: [0.0; NUM_JOINTS] }

    }

    /// EEPROM and timeout setup is owned by STM32 now; this startup probe
    /// verifies that the USB state frame contains a live onboard IMU.
    pub fn check_registers(&mut self) -> Result<usize> {
        self.read_imu()?;
        if self.imu_sensor_ok { Ok(0) }
        else { Err(IoError::Bus("STM32 BMI088 is not ready".into())) }
    }

    pub fn present_positions(&mut self) -> Result<[f64; NUM_JOINTS]> {
        Ok(self.read_state()?.positions)
    }

    pub fn set_torque(&mut self, on: bool) -> Result<()> {
        if on {
            if let Some(reason) = self.motor_control_error() {
                return Err(IoError::Bus(reason.into()));
            }
        }
        let result = self.request_torque(on);
        // An accepted enable whose feedback never arrives must not leave partially
        // enabled motors behind. Disable is always allowed, including on old firmware.
        if on && result.is_err() {
            if let Err(cleanup) = self.request_torque(false) {
                return Err(IoError::Bus(format!(
                    "{result:?}; failed to disable after enable failure: {cleanup}"
                )));
            }
        }
        result
    }

    fn request_torque(&mut self, on: bool) -> Result<()> {
        let op = if on { 1 } else { 2 };
        let request_seq = self.next_seq();
        let mut payload = vec![0u8; 28];
        put_u16(&mut payload[0..2], request_seq);
        payload[2] = op;
        payload[3] = CONFIGURED_MOTOR_IDS.len() as u8;
        payload[4..4 + CONFIGURED_MOTOR_IDS.len()].copy_from_slice(&CONFIGURED_MOTOR_IDS);
        let wire = pack(MSG_ADMIN, request_seq, self.tick_us(), 0, &payload);
        self.write_all(&wire)?;
        let deadline = Instant::now() + ADMIN_TIMEOUT;
        let mut acknowledged = false;
        let mut feedback_enabled = false;
        while Instant::now() < deadline {
            let frame = self.read_frame()?;
            if frame.msg_type == MSG_STATE {
                let sensors = self.parse_and_observe_state(&frame.payload)?;
                feedback_enabled = self.motor_control_error().is_none()
                    && configured_motors_enabled(&sensors);
            } else if frame.msg_type == MSG_ADMIN_RESULT && frame.payload.len() >= 28
                && get_u16(&frame.payload[0..2]) == request_seq && frame.payload[2] == op {
                if frame.payload[3] != 0 {
                    return Err(IoError::Bus(admin_failure(&frame.payload)));
                }
                if !on { return Ok(()); }
                // Success means the firmware processed exactly the requested motors,
                // followed by a fresh enabled/online/fault-free feedback sample.
                if frame.payload[20] != 0 || frame.payload[21] as usize != CONFIGURED_MOTOR_IDS.len()
                    || frame.payload[22] as usize != CONFIGURED_MOTOR_IDS.len() {
                    return Err(IoError::Bus("STM32 未完成全部已配置电机的使能".into()));
                }
                acknowledged = true;
            }
            if acknowledged && feedback_enabled { return Ok(()); }
        }
        Err(IoError::Bus("STM32 管理命令或电机使能反馈超时".into()))
    }

    pub fn interpolate_to(&mut self, target: &[f64; NUM_JOINTS], duration: Duration,
                          step: Duration) -> Result<()> {
        let start = self.present_positions()?;
        let steps = (duration.as_secs_f64() / step.as_secs_f64()).ceil().max(1.0) as u32;
        for i in 1..=steps {
            let t = i as f64 / steps as f64;
            let mut next = [0.0; NUM_JOINTS];
            for j in 0..NUM_JOINTS { next[j] = start[j] + (target[j] - start[j]) * t; }
            self.write_targets(&JointTargets::new(next))?;
            std::thread::sleep(step);
        }
        Ok(())
    }

    fn calibration_ready(&self) -> Result<()> {
        if self.control_capabilities & 0x0004 == 0 {
            return Err(IoError::Bus("STM32 固件不支持安全标定，请先更新固件".into()));
        }
        let Some((sensors, at)) = self.last_motor_state.as_ref() else {
            return Err(IoError::Bus("尚无电机反馈".into()));
        };
        if at.elapsed() > Duration::from_millis(500) || !configured_motors_disabled(sensors) {
            return Err(IoError::Bus("标定要求全部电机在线且已确认失能".into()));
        }
        Ok(())
    }

    fn calibration_request(&mut self, msg: u8, op: u8, mut payload: Vec<u8>,
                           expected_count: usize, zero_id: Option<u8>) -> Result<()> {
        // Obtain a new state before checking enable flags.
        self.read_state()?;
        self.calibration_ready()?;
        let seq = self.next_seq();
        put_u16(&mut payload[..2], seq);
        self.write_all(&pack(msg, seq, self.tick_us(), 0, &payload))?;
        let deadline = Instant::now() + Duration::from_millis(700);
        let mut ack_tick = None;
        while Instant::now() < deadline {
            let frame = self.read_frame()?;
            if frame.msg_type == MSG_ADMIN_RESULT && frame.payload.len() >= 28
                && get_u16(&frame.payload[..2]) == seq && frame.payload[2] == op {
                if frame.payload[3] != 0 {
                    return Err(IoError::Bus(admin_failure(&frame.payload)));
                }
                if frame.payload[20] != 0 || frame.payload[21] as usize != expected_count
                    || frame.payload[22] as usize != expected_count {
                    return Err(IoError::Bus("STM32 标定结果数量不匹配".into()));
                }
                if zero_id.is_none() { return Ok(()); }
                ack_tick = Some(get_u32(&frame.payload[4..8]));
            } else if frame.msg_type == MSG_STATE {
                let sensors = self.parse_and_observe_state(&frame.payload)?;
                if let (Some(ack_tick), Some(id)) = (ack_tick, zero_id) {
                    let slot = STM32_MOTOR_IDS.iter().position(|&motor| motor == id).unwrap();
                    let joint = STM32_TO_CONTROL_JOINT[slot];
                    let feedback_offset = STATE_PREFIX_LEN + slot * MOTOR_OBSERVATION_LEN + 12;
                    let feedback_tick = get_u32(&frame.payload[feedback_offset..feedback_offset + 4]);
                    let after_ack = feedback_tick.wrapping_sub(ack_tick);
                    if after_ack > 0 && after_ack < 0x8000_0000
                        && configured_motors_disabled(&sensors)
                        && sensors.motor_feedback_age_ms[joint] <= 50
                        && sensors.positions[joint].abs() <= 0.01 {
                        return Ok(());
                    }
                }
            }
        }
        Err(IoError::Bus("标定未确认完成；请刷新位置核对，勿自动重复零位写入".into()))
    }

    fn next_seq(&mut self) -> u16 { self.frame_seq = self.frame_seq.wrapping_add(1); self.frame_seq }
    fn tick_us(&self) -> u32 { self.started.elapsed().as_micros() as u32 }

    fn write_all(&mut self, bytes: &[u8]) -> Result<()> {
        // USB CDC writes are delivered in order by the tty driver. `flush()` maps to
        // `tcdrain`, which can block inside cdc_acm for minutes when an endpoint stops
        // responding, ignoring the serial-port timeout and freezing the 50 Hz safety loop.
        // `write_all` already waits until this frame has been accepted by the driver; the
        // protocol's sequence/ack fields, state timeout and reconnect path provide the
        // end-to-end delivery checks.
        self.port
            .write_all(bytes)
            .map_err(|e| IoError::Bus(format!("USB write: {e}")))
    }

    fn read_frame(&mut self) -> Result<Frame> {
        loop {
            if let Some(frame) = take_frame(&mut self.rx)? { return Ok(frame); }
            let mut chunk = [0u8; 512];
            match self.port.read(&mut chunk) {
                Ok(0) => continue,
                Ok(n) => self.rx.extend_from_slice(&chunk[..n]),
                Err(e) if e.kind() == std::io::ErrorKind::TimedOut =>
                    return Err(IoError::Bus("STM32 state frame timed out".into())),
                Err(e) => return Err(IoError::Bus(format!("USB read: {e}"))),
            }
            if self.rx.len() > 4096 {
                self.rx.clear();
                return Err(IoError::Bus("STM32 USB receive buffer overflow".into()));
            }
        }
    }

    fn read_state(&mut self) -> Result<Sensors> {
        loop {
            let frame = self.read_frame()?;
            if frame.msg_type == MSG_STATE { return self.parse_and_observe_state(&frame.payload); }
        }
    }

    fn parse_and_observe_state(&mut self, payload: &[u8]) -> Result<Sensors> {
        let parsed = parse_state(payload)?;
        log_gateway_fault_change(self.gateway_faults, &parsed);
        self.stale_imu.observe(parsed.imu_sequence);
        self.imu_sensor_ok = parsed.imu_flags & IMU_SENSOR_OK != 0;
        self.imu_ready = parsed.imu_flags & IMU_CALIBRATED != 0;
        self.last_temps_c = parsed.temps_c;
        self.control_capabilities = parsed.control_capabilities;
        self.gateway_faults = parsed.gateway_faults;
        self.last_motor_state = Some((parsed.sensors, Instant::now()));
        Ok(parsed.sensors)
    }

    fn write_targets(&mut self, targets: &JointTargets) -> Result<()> {
        // Reading an old firmware or an entirely disabled robot must not emit motor
        // commands. No torque-on is sent implicitly by this path.
        if self.control_capabilities & CONTROL_CAPABILITIES != CONTROL_CAPABILITIES
            || !self.last_motor_state.as_ref().is_some_and(|(sensors, _)|
                sensors.motor_flags.iter().any(|flags| flags & 0x04 != 0)) {
            return Ok(());
        }
        if let Some(reason) = self.motor_control_error() {
            return Err(IoError::Bus(reason.into()));
        }
        self.command_seq = self.command_seq.wrapping_add(1);
        let p = encode_targets(targets, self.kp_centi, self.command_seq, self.tick_us())?;
        let seq = self.next_seq();
        let wire = pack(MSG_COMMAND, seq, self.tick_us(), 0, &p);
        self.write_all(&wire)
    }

    fn read_imu(&mut self) -> Result<ImuData> {
        Ok(self.read_state()?.imu)
    }
}

impl RobotIo for DynamixelIo {
    fn read(&mut self) -> Result<Sensors> {
        self.read_state()
    }
    fn state_frames_pace_control(&self) -> bool { true }
    fn write_motor_commands(&mut self, commands: &[duck_ipc_proto::MotorCommand]) -> Result<u16> {
        if let Some(reason) = self.motor_control_error() {
            return Err(IoError::Bus(reason.into()));
        }
        if self.motor_torque_enabled() != Some(true) {
            return Err(IoError::Bus("experiment motors are not enabled".into()));
        }
        self.command_seq = self.command_seq.wrapping_add(1);
        let payload = encode_motor_commands(commands, self.command_seq, self.tick_us())?;
        let seq = self.next_seq();
        self.write_all(&pack(MSG_COMMAND, seq, self.tick_us(), 0, &payload))?;
        Ok(self.command_seq)
    }
    fn write(&mut self, targets: &JointTargets) -> Result<()> { self.write_targets(targets) }
    fn set_gain(&mut self, kp: u16) -> Result<()> { self.kp_centi = kp.saturating_mul(100); Ok(()) }
    fn set_torque(&mut self, on: bool) -> Result<()> { DynamixelIo::set_torque(self, on) }
    fn slow_sensors(&mut self) -> Result<SlowSensors> {
        // DMUSB has rotor temperature but no supply-voltage field. Values are refreshed by
        // the ordinary 50 Hz state read, so this accessor adds no USB transaction.
        Ok(SlowSensors { volts: 0.0, temps_c: self.last_temps_c })
    }
    fn imu_stale(&self) -> ImuStale { self.stale_imu.stale }
    fn imu_ready(&self) -> bool { self.imu_ready }
    fn motor_torque_enabled(&self) -> Option<bool> {
        self.last_motor_state.as_ref().map(|(sensors, _)| configured_motors_enabled(sensors))
    }
    fn set_position_limits(&mut self, limits: &[MotorPositionLimit]) -> Result<()> {
        let payload = encode_position_limits(limits)?;
        self.calibration_request(14, 4, payload, CONFIGURED_MOTOR_IDS.len(), None)?;
        self.control_capabilities |= 8;
        Ok(())
    }
    fn position_limits_ready(&self) -> Option<bool> {
        (self.control_capabilities & 4 != 0).then_some(self.control_capabilities & 8 != 0)
    }
    fn mark_motor_zero(&mut self, motor_id: u8) -> Result<()> {
        if !CONFIGURED_MOTOR_IDS.contains(&motor_id) {
            return Err(IoError::Bus("未知电机 ID".into()));
        }
        let mut p = vec![0u8; 28];
        p[2] = 3; p[3] = 1; p[4] = motor_id;
        self.calibration_request(MSG_ADMIN, 3, p, 1, Some(motor_id))
    }
    fn motor_control_error(&self) -> Option<&'static str> {
        if self.position_limits_ready() == Some(false) {
            return Some("STM32 尚未装载位置限位，不能开启控制");
        }
        if !self.imu_sensor_ok || !self.imu_ready {
            return Some("STM32 IMU 未就绪或传感器异常，不能开启控制");
        }
        if self.stale_imu.stale.run >= 5 {
            return Some("STM32 IMU 数据连续 5 帧未更新，控制已停止");
        }
        motor_control_problem(self.control_capabilities, self.gateway_faults,
            self.last_motor_state.as_ref().map(|(sensors, at)| (sensors, at.elapsed())))
    }
}

pub fn configured_motors_disabled(sensors: &Sensors) -> bool {
    STM32_MOTOR_IDS.iter().zip(STM32_TO_CONTROL_JOINT).all(|(&id, joint)| {
        if id == 0 { sensors.motor_flags[joint] & 1 == 0 }
        else { sensors.motor_flags[joint] & 0x0f == 0x03 && sensors.motor_feedback_age_ms[joint] <= 500 }
    })
}

fn encode_position_limits(limits: &[MotorPositionLimit]) -> Result<Vec<u8>> {
    if limits.len() != CONFIGURED_MOTOR_IDS.len() {
        return Err(IoError::Bus("限位必须包含全部已配置电机".into()));
    }
    let mut seen = Vec::new();
    let mut p = vec![0u8; 4 + limits.len() * 12];
    p[2] = limits.len() as u8;
    for (i, limit) in limits.iter().enumerate() {
        if !CONFIGURED_MOTOR_IDS.contains(&limit.motor_id) || seen.contains(&limit.motor_id)
            || limit.min_mrad >= limit.max_mrad || limit.min_mrad < -12500 || limit.max_mrad > 12500 {
            return Err(IoError::Bus("限位的电机 ID 或范围无效".into()));
        }
        seen.push(limit.motor_id);
        let offset = 4 + i * 12;
        p[offset] = limit.motor_id;
        p[offset+4..offset+8].copy_from_slice(&limit.min_mrad.to_le_bytes());
        p[offset+8..offset+12].copy_from_slice(&limit.max_mrad.to_le_bytes());
    }
    Ok(p)
}

fn motor_control_problem(capabilities: u16, faults: u32,
                         sample: Option<(&Sensors, Duration)>) -> Option<&'static str> {
    let Some((sensors, age)) = sample else { return Some("尚未收到 STM32 电机状态"); };
    if capabilities & CONTROL_CAPABILITIES != CONTROL_CAPABILITIES {
        return Some("STM32 固件尚未支持空槽控制及电机列表使能，请先更新电机固件");
    }
    if age > Duration::from_millis(500) { return Some("STM32 电机状态已过期"); }
    // HOST_COMMAND_STALE is a recoverable stop, cleared by an explicit enable.
    if faults & 0x20 != 0 { return Some("STM32 CAN 发送失败或队列已满，电机控制已停止"); }
    if faults & 0x100 != 0 { return Some("STM32 IMU 保护已触发，全电机控制已停止"); }
    if faults & 0x02 != 0 && faults & !0x82 == 0 {
        return Some("STM32 电机反馈超时，控制已停止");
    }
    if faults & !0x80 != 0 { return Some("STM32 电机网关报告故障，请检查故障状态"); }
    for (slot, &joint) in STM32_TO_CONTROL_JOINT.iter().enumerate() {
        let flags = sensors.motor_flags[joint];
        if (flags & 1 != 0) != (STM32_MOTOR_IDS[slot] != 0) {
            return Some("STM32 电机配置与 RK3566 的槽位映射不一致");
        }
        if STM32_MOTOR_IDS[slot] == 0 { continue; }
        if flags & 0x02 == 0 || sensors.motor_feedback_age_ms[joint] > 500 {
            return Some("存在离线或反馈过期的电机，不能开启控制");
        }
        if flags & 0x08 != 0 { return Some("电机报告硬件故障，不能开启控制"); }
    }
    None
}

fn admin_failure(payload: &[u8]) -> String {
    let faults = get_u32(&payload[8..12]);
    let reason = if faults & 0x20 != 0 {
        "CAN 发送失败或队列已满"
    } else {
        match payload[3] {
            7 => "STM32 正在处理其他管理操作",
            8 => "STM32 电机网关故障",
            9 => "电机列表包含重复 ID",
            10 => "电机列表包含未知 ID",
            11 => "电机数量不匹配",
            _ => "STM32 拒绝管理命令",
        }
    };
    format!("{reason}（操作 {}，返回码 {}，故障位 0x{faults:08x}，电机 ID {}，完成 {}/{}）",
        payload[2], payload[3], payload[23], payload[22], payload[21])
}

fn configured_motors_enabled(sensors: &Sensors) -> bool {
    STM32_TO_CONTROL_JOINT.iter().enumerate().all(|(slot, &joint)|
        STM32_MOTOR_IDS[slot] == 0 || sensors.motor_flags[joint] & 0x07 == 0x07)
}

/// Bounds are the deployed STM32 command guards, not the wider CAN encoding range.
pub fn motor_command_bounds(id: u8) -> Option<(f64, f64)> {
    match id { 1..=14 => Some((10.0, 23.5)), _ => None }
}

pub fn validate_motor_commands(commands: &[duck_ipc_proto::MotorCommand], limits: &[MotorPositionLimit]) -> Result<()> {
    if commands.is_empty() || commands.len() > CONFIGURED_MOTOR_IDS.len() {
        return Err(IoError::Bus("invalid motor count".into()));
    }
    let mut seen = Vec::new();
    for c in commands {
        let (vmax, tmax) = motor_command_bounds(c.motor_id).ok_or_else(|| IoError::Bus("unknown motor ID".into()))?;
        let limit = limits.iter().find(|l| l.motor_id == c.motor_id).ok_or_else(|| IoError::Bus("missing calibrated limit".into()))?;
        if seen.contains(&c.motor_id) || [c.p,c.v,c.tau,c.kp,c.kd].iter().any(|x| !x.is_finite())
            || c.p < limit.min_mrad as f64 / 1000.0 || c.p > limit.max_mrad as f64 / 1000.0
            || c.v.abs() > vmax || c.tau.abs() > tmax || !(0.0..=500.0).contains(&c.kp) || !(0.0..=5.0).contains(&c.kd) {
            return Err(IoError::Bus(format!("motor {} has duplicate ID or out-of-range parameters", c.motor_id)));
        }
        seen.push(c.motor_id);
    }
    Ok(())
}

fn encode_motor_commands(commands: &[duck_ipc_proto::MotorCommand], sequence: u16, tick_us: u32) -> Result<Vec<u8>> {
    let limits: Vec<_> = CONFIGURED_MOTOR_IDS.iter().map(|&motor_id| MotorPositionLimit { motor_id, min_mrad: -12500, max_mrad: 12500 }).collect();
    validate_motor_commands(commands, &limits)?;
    if commands.len() != CONFIGURED_MOTOR_IDS.len() { return Err(IoError::Bus("wire frame must contain every configured motor".into())); }
    let mut p = vec![0; COMMAND_PREFIX_LEN + STM32_MOTOR_COUNT * MOTOR_COMMAND_LEN];
    put_u16(&mut p[0..2], sequence); p[2] = MODE_MIT; p[3] = STM32_MOTOR_COUNT as u8;
    put_u32(&mut p[4..8], tick_us);
    for c in commands {
        let slot = STM32_MOTOR_IDS.iter().position(|&id| id == c.motor_id).unwrap();
        let b = COMMAND_PREFIX_LEN + slot * MOTOR_COMMAND_LEN;
        put_i32(&mut p[b..b+4], (c.p*1000.0).round() as i32);
        put_i32(&mut p[b+4..b+8], (c.v*1000.0).round() as i32);
        put_i32(&mut p[b+8..b+12], (c.tau*1000.0).round() as i32);
        put_u16(&mut p[b+12..b+14], (c.kp*100.0).round() as u16);
        put_u16(&mut p[b+14..b+16], (c.kd*1000.0).round() as u16);
        put_u16(&mut p[b+16..b+18], 1);
    }
    Ok(p)
}

fn encode_targets(targets: &JointTargets, kp_centi: u16, command_seq: u16,
                  tick_us: u32) -> Result<Vec<u8>> {
    let (kp_centi, kd_milli) = if let Some([kp, kd]) = targets.pd {
        if !kp.is_finite() || !kd.is_finite() || !(0.0..=500.0).contains(&kp) || !(0.0..=5.0).contains(&kd) {
            return Err(IoError::Bus("拒绝发送无效的策略 Kp/Kd".into()));
        }
        ((kp * 100.0).round() as u16, (kd * 1000.0).round() as u16)
    } else { (kp_centi, 4_000) };
    let mut p = vec![0u8; COMMAND_PREFIX_LEN + STM32_MOTOR_COUNT * MOTOR_COMMAND_LEN];
    put_u16(&mut p[0..2], command_seq);
    p[2] = MODE_MIT;
    p[3] = STM32_MOTOR_COUNT as u8;
    put_u32(&mut p[4..8], tick_us);
    for (slot, &joint) in STM32_TO_CONTROL_JOINT.iter().enumerate() {
        if STM32_MOTOR_IDS[slot] == 0 { continue; }
        let (position, velocity, torque, joint_kp, joint_kd) = match targets.mit {
            Some(mit) => {
                let target = mit[joint];
                let (vmax, tmax) = motor_command_bounds(STM32_MOTOR_IDS[slot])
                    .ok_or_else(|| IoError::Bus("unknown motor ID".into()))?;
                if [target.position, target.velocity, target.torque_ff, target.kp, target.kd]
                    .iter().any(|value| !value.is_finite())
                    || target.velocity.abs() > vmax
                    || target.torque_ff.abs() > tmax
                    || !(0.0..=500.0).contains(&target.kp)
                    || !(0.0..=5.0).contains(&target.kd)
                {
                    return Err(IoError::Bus(format!(
                        "拒绝发送关节 {} 的无效 MIT 目标", crate::model::JOINT_NAMES[joint]
                    )));
                }
                (target.position, target.velocity, target.torque_ff,
                 (target.kp * 100.0).round() as u16,
                 (target.kd * 1000.0).round() as u16)
            }
            None => (targets.positions[joint], 0.0, 0.0, kp_centi, kd_milli),
        };
        if !position.is_finite() {
            return Err(IoError::Bus("拒绝发送非有限的电机目标位置".into()));
        }
        let b = COMMAND_PREFIX_LEN + slot * MOTOR_COMMAND_LEN;
        let pos = (position * 1000.0).round().clamp(i32::MIN as f64, i32::MAX as f64) as i32;
        put_i32(&mut p[b..b + 4], pos);
        put_i32(&mut p[b + 4..b + 8], (velocity * 1000.0).round() as i32);
        put_i32(&mut p[b + 8..b + 12], (torque * 1000.0).round() as i32);
        put_u16(&mut p[b + 12..b + 14], joint_kp);
        put_u16(&mut p[b + 14..b + 16], joint_kd);
        put_u16(&mut p[b + 16..b + 18], 1);
    }
    Ok(p)
}

struct ParsedState {
    sensors: Sensors,
    control_capabilities: u16,
    gateway_faults: u32,
    fault_motor: Option<FaultMotor>,
    imu_sequence: u32,
    imu_flags: u8,
    temps_c: [f64; NUM_JOINTS],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FaultMotor {
    id: u8,
    route: u8,
    joint: &'static str,
}

fn gateway_fault_names(faults: u32) -> String {
    const NAMES: [&str; 9] = [
        "enable_failed", "feedback_timeout", "mos_over_temperature",
        "rotor_over_temperature", "motor_hardware_fault", "can_tx_drop",
        "timeout_config_failed", "host_command_timeout", "imu_invalid",
    ];
    let mut names: Vec<&str> = NAMES.iter().enumerate()
        .filter_map(|(bit, name)| (faults & (1 << bit) != 0).then_some(*name)).collect();
    if faults & !0x1ff != 0 { names.push("unknown"); }
    names.join(",")
}

// Journal is the durable history. Log transitions once, not every 50 Hz frame;
// clearing a current fault never removes the entry describing its occurrence.
fn log_gateway_fault_change(previous: u32, state: &ParsedState) {
    let current = state.gateway_faults;
    if current == previous { return; }
    let raised = current & !previous;
    let cleared = previous & !current;
    if raised != 0 {
        tracing::warn!(
            fault_flags = format_args!("0x{current:08x}"),
            raised = format_args!("0x{raised:08x}"),
            reason = %gateway_fault_names(raised),
            stm32_tick_ms = state.sensors.gateway_tick_ms,
            last_fault_motor = ?state.fault_motor,
            motor_flags = ?state.sensors.motor_flags,
            feedback_age_ms = ?state.sensors.motor_feedback_age_ms,
            "STM32 gateway fault asserted"
        );
    }
    if cleared != 0 {
        tracing::info!(
            fault_flags = format_args!("0x{current:08x}"),
            cleared = format_args!("0x{cleared:08x}"),
            reason = %gateway_fault_names(cleared),
            stm32_tick_ms = state.sensors.gateway_tick_ms,
            last_fault_motor = ?state.fault_motor,
            motors_disabled = configured_motors_disabled(&state.sensors),
            "STM32 gateway fault cleared; this does not request motor enable"
        );
    }
}

fn parse_state(payload: &[u8]) -> Result<ParsedState> {
    if payload.len() < STATE_PREFIX_LEN {
        return Err(IoError::ShortRead { what: "STM32 state prefix", expected: STATE_PREFIX_LEN, got: payload.len() });
    }
    let count = payload[13] as usize;
    if count != STM32_MOTOR_COUNT {
        return Err(IoError::ShortRead {
            what: "STM32 motor count",
            expected: STM32_MOTOR_COUNT,
            got: count,
        });
    }
    let expected = STATE_PREFIX_LEN + count * MOTOR_OBSERVATION_LEN;
    if payload.len() != expected {
        return Err(IoError::ShortRead { what: "STM32 state payload", expected, got: payload.len() });
    }
    let mut out = Sensors::default();
    let mut temps_c = [0.0; NUM_JOINTS];
    let stm32_tick_ms = get_u32(&payload[4..8]);
    out.gateway_tick_ms = stm32_tick_ms;
    out.ack_command_seq = get_u16(&payload[2..4]);
    for axis in 0..3 {
        out.imu.gyro[axis] = get_f32(&payload[16 + axis * 4..20 + axis * 4]) as f64;
        out.imu.gravity[axis] = get_f32(&payload[28 + axis * 4..32 + axis * 4]) as f64;
    }
    let mut quat = [0.0; 4];
    for (axis, value) in quat.iter_mut().enumerate() {
        *value = get_f32(&payload[40 + axis * 4..44 + axis * 4]) as f64;
    }
    let quat_norm = quat.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !quat_norm.is_finite() || !(0.5..=1.5).contains(&quat_norm)
        || out.imu.gyro.iter().chain(out.imu.gravity.iter()).any(|value| !value.is_finite()) {
        return Err(IoError::Bus("STM32 IMU quaternion is invalid".into()));
    }
    for value in &mut quat {
        *value /= quat_norm;
    }
    out.imu.quat = quat;
    out.attitude_rpy = quaternion_to_euler_zyx(quat);
    for (route, &joint) in STM32_TO_CONTROL_JOINT.iter().enumerate() {
        let b = STATE_PREFIX_LEN + route * MOTOR_OBSERVATION_LEN;
        out.positions[joint] = get_i32(&payload[b..b + 4]) as f64 / 1000.0;
        out.velocities[joint] = get_i32(&payload[b + 4..b + 8]) as f64 / 1000.0;
        out.motor_torques_nm[joint] = get_i32(&payload[b + 8..b + 12]) as f64 / 1000.0;
        // Protocol reports torque, not phase current; do not mislabel it as mA.
        out.currents_ma[joint] = 0.0;
        let flags = payload[b + 16];
        let rx_ts_ms = get_u32(&payload[b + 12..b + 16]);
        let temperature_c = payload[b + 17] as f64;
        out.motor_flags[joint] = flags;
        out.motor_temperatures_c[joint] = temperature_c;
        out.motor_feedback_age_ms[joint] = if flags & 0x01 != 0 {
            stm32_tick_ms.wrapping_sub(rx_ts_ms)
        } else {
            0
        };
        temps_c[joint] = temperature_c;
    }
    let control_capabilities = get_u16(&payload[14..16]);
    let route = payload[62] as usize;
    // Old v4 firmware left these bytes reserved. Do not invent a fault location
    // from zeros or malformed diagnostics, and never use diagnostics to enable.
    let fault_motor = if control_capabilities & FAULT_DIAGNOSTICS_CAPABILITY != 0
        && route < STM32_MOTOR_COUNT && payload[61] != 0
        && STM32_MOTOR_IDS[route] == payload[61] {
        Some(FaultMotor { id: payload[61], route: route as u8,
            joint: JOINT_NAMES[STM32_TO_CONTROL_JOINT[route]] })
    } else { None };
    Ok(ParsedState {
        sensors: out,
        control_capabilities,
        gateway_faults: get_u32(&payload[8..12]),
        fault_motor,
        imu_sequence: get_u32(&payload[56..60]),
        imu_flags: payload[60],
        temps_c,
    })
}

fn take_frame(rx: &mut Vec<u8>) -> Result<Option<Frame>> {
    loop {
        while rx.len() >= 3 && (get_u16(&rx[0..2]) != MAGIC || rx[2] != VERSION) {
            rx.remove(0);
        }
        if rx.len() < HEADER_LEN {
            return Ok(None);
        }
        let payload_len = get_u16(&rx[6..8]) as usize;
        if payload_len > MAX_PAYLOAD {
            rx.remove(0);
            continue;
        }
        let total = HEADER_LEN + payload_len;
        if rx.len() < total {
            return Ok(None);
        }

        let received = get_u16(&rx[16..18]);
        let mut checked = rx[..total].to_vec();
        checked[16] = 0;
        checked[17] = 0;
        if received != crc16(&checked) {
            // Do not trust the corrupt frame's declared boundary. Dropping one byte and
            // searching for the next magic recovers from a bad length as well as a bad CRC.
            rx.remove(0);
            continue;
        }

        let wire: Vec<u8> = rx.drain(..total).collect();
        return Ok(Some(Frame {
            msg_type: wire[3],
            payload: wire[HEADER_LEN..].to_vec(),
        }));
    }
}

fn pack(msg_type: u8, seq: u16, tick_us: u32, flags: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; HEADER_LEN + payload.len()];
    put_u16(&mut out[0..2], MAGIC); out[2] = VERSION; out[3] = msg_type;
    put_u16(&mut out[4..6], seq); put_u16(&mut out[6..8], payload.len() as u16);
    put_u32(&mut out[8..12], tick_us); put_u32(&mut out[12..16], flags);
    out[HEADER_LEN..].copy_from_slice(payload);
    let crc = crc16(&out); put_u16(&mut out[16..18], crc); out
}

fn crc16(data: &[u8]) -> u16 {
    let mut crc = 0xFFFFu16;
    for byte in data {
        crc ^= (*byte as u16) << 8;
        for _ in 0..8 { crc = if crc & 0x8000 != 0 { (crc << 1) ^ 0x1021 } else { crc << 1 }; }
    }
    crc
}

fn get_u16(p: &[u8]) -> u16 { u16::from_le_bytes([p[0], p[1]]) }
fn get_u32(p: &[u8]) -> u32 { u32::from_le_bytes([p[0], p[1], p[2], p[3]]) }
fn get_i32(p: &[u8]) -> i32 { i32::from_le_bytes([p[0], p[1], p[2], p[3]]) }
fn get_f32(p: &[u8]) -> f32 { f32::from_le_bytes([p[0], p[1], p[2], p[3]]) }

/// Human-readable telemetry only. Control and odometry consume the v4 quaternion directly.
fn quaternion_to_euler_zyx(q: [f64; 4]) -> [f64; 3] {
    let [w, x, y, z] = q;
    [
        (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y)),
        (2.0 * (w * y - z * x)).clamp(-1.0, 1.0).asin(),
        (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z)),
    ]
}
fn put_u16(p: &mut [u8], v: u16) { p[..2].copy_from_slice(&v.to_le_bytes()); }
fn put_u32(p: &mut [u8], v: u32) { p[..4].copy_from_slice(&v.to_le_bytes()); }
fn put_i32(p: &mut [u8], v: i32) { p[..4].copy_from_slice(&v.to_le_bytes()); }

#[cfg(test)]
fn put_f32(p: &mut [u8], v: f32) { p[..4].copy_from_slice(&v.to_le_bytes()); }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_motor_ids_map_to_left_right_then_head() {
        let names = STM32_TO_CONTROL_JOINT.map(|joint| JOINT_NAMES[joint]);
        assert_eq!(names, [
            "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
            "neck_pitch", "head_pitch", "head_yaw", "head_roll",
        ]);
    }

    #[test]
    fn fault_diagnostics_are_optional_and_validate_the_motor_route() {
        let mut payload = test_state_payload(false);
        payload[61] = 11;
        payload[62] = 10;
        assert_eq!(parse_state(&payload).unwrap().fault_motor, None, "old firmware has reserved bytes");
        put_u16(&mut payload[14..16], 0x1f);
        assert_eq!(parse_state(&payload).unwrap().fault_motor,
            Some(FaultMotor { id: 11, route: 10, joint: "neck_pitch" }));
        for (id, route) in [(1, 255), (2, 5), (0, 1), (1, 14)] {
            payload[61] = id;
            payload[62] = route;
            assert_eq!(parse_state(&payload).unwrap().fault_motor, None);
        }
    }

    #[cfg(unix)]
    #[test]
    fn feedback_fault_recovery_is_logged_once_and_never_sends_enable() {
        use serialport::SerialPort;
        use std::sync::{Arc, Mutex};
        use tracing::{Event, Metadata, Subscriber, field::{Field, Visit}, span::{Attributes, Id, Record}};
        #[derive(Clone, Default)]
        struct Capture(Arc<Mutex<Vec<String>>>);
        struct Fields(String);
        impl Visit for Fields {
            fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
                use std::fmt::Write;
                write!(&mut self.0, " {}={value:?}", field.name()).unwrap();
            }
        }
        impl Subscriber for Capture {
            fn enabled(&self, _: &Metadata<'_>) -> bool { true }
            fn new_span(&self, _: &Attributes<'_>) -> Id { Id::from_u64(1) }
            fn record(&self, _: &Id, _: &Record<'_>) {}
            fn record_follows_from(&self, _: &Id, _: &Id) {}
            fn event(&self, event: &Event<'_>) {
                let mut fields = Fields(String::new());
                event.record(&mut fields);
                self.0.lock().unwrap().push(fields.0);
            }
            fn enter(&self, _: &Id) {}
            fn exit(&self, _: &Id) {}
        }
        let capture = Capture::default();
        let (firmware, host) = serialport::TTYPort::pair().unwrap();
        let mut io = DynamixelIo::from_port(Box::new(host));
        let mut payload = test_state_payload(false);
        put_u16(&mut payload[14..16], 0x1f);
        payload[61] = 11;
        payload[62] = 10;
        tracing::subscriber::with_default(capture.clone(), || {
            for (seq, fault) in [0, 2, 2, 2, 0, 0].into_iter().enumerate() {
                put_u32(&mut payload[8..12], fault);
                put_u32(&mut payload[56..60], seq as u32);
                io.parse_and_observe_state(&payload).unwrap();
                assert_eq!(io.motor_control_error().is_some(), fault != 0);
                assert_eq!(io.motor_torque_enabled(), Some(false));
                io.write_targets(&JointTargets::new([0.0; NUM_JOINTS])).unwrap();
            }
        });
        assert_eq!(firmware.bytes_to_read().unwrap(), 0, "recovery must not send commands or enable");
        let logs = capture.0.lock().unwrap();
        assert_eq!(logs.len(), 2, "unchanged fault frames must not flood Journal: {logs:?}");
        assert!(logs[0].contains("asserted") && logs[0].contains("0x00000002"));
        assert!(logs[0].contains("feedback_timeout") && logs[0].contains("neck_pitch"));
        assert!(logs[1].contains("cleared") && logs[1].contains("motors_disabled=true"));
        assert!(logs[1].contains("neck_pitch"), "recovery retains the original fault motor");
    }

    fn ready_motor_sample() -> Sensors {
        let mut sensors = Sensors::default();
        for (slot, &joint) in STM32_TO_CONTROL_JOINT.iter().enumerate() {
            if STM32_MOTOR_IDS[slot] != 0 { sensors.motor_flags[joint] = 3; }
        }
        sensors
    }

    #[test]
    fn motor_control_requires_compatible_fresh_healthy_hardware() {
        let mut sensors = ready_motor_sample();
        fn sample(s: &Sensors) -> Option<(&Sensors, Duration)> { Some((s, Duration::ZERO)) }
        assert!(motor_control_problem(0, 0, sample(&sensors)).is_some());
        assert!(motor_control_problem(1, 0, sample(&sensors)).is_some());
        assert!(motor_control_problem(3, 0, None).is_some());
        assert!(motor_control_problem(3, 0, Some((&sensors, Duration::from_millis(501)))).is_some());
        assert!(motor_control_problem(3, 1, sample(&sensors)).is_some());
        assert_eq!(motor_control_problem(3, 0, sample(&sensors)), None);
        assert_eq!(motor_control_problem(3, 0x80, sample(&sensors)), None);
        assert!(!configured_motors_enabled(&sensors));
        sensors.motor_flags[5] = 1;
        assert!(motor_control_problem(3, 0, sample(&sensors)).is_some());
        sensors.motor_flags[5] = 11;
        assert!(motor_control_problem(3, 0, sample(&sensors)).is_some());
        sensors.motor_flags[5] = 3;
        sensors.motor_feedback_age_ms[5] = 501;
        assert!(motor_control_problem(3, 0, sample(&sensors)).is_some());
        sensors.motor_feedback_age_ms[5] = 0;
        sensors.motor_flags[1] = 0; // Every one of the fourteen slots must be configured.
        assert!(motor_control_problem(3, 0, sample(&sensors)).is_some());
    }

    #[test]
    fn motor_commands_preserve_slots_and_leave_empty_records_zero() {
        let mut targets = JointTargets::new([0.0; NUM_JOINTS]);
        for (joint, position) in targets.positions.iter_mut().enumerate() {
            *position = joint as f64 / 10.0;
        }
        let payload = encode_targets(&targets, 20_000, 42, 1234).unwrap();
        assert_eq!(payload.len(), 288);
        assert_eq!(get_u16(&payload[0..2]), 42);
        assert_eq!(payload[3], 14);
        for (slot, &joint) in STM32_TO_CONTROL_JOINT.iter().enumerate() {
            let b = COMMAND_PREFIX_LEN + slot * MOTOR_COMMAND_LEN;
            if STM32_MOTOR_IDS[slot] == 0 {
                assert_eq!(&payload[b..b + MOTOR_COMMAND_LEN], &[0u8; MOTOR_COMMAND_LEN]);
            } else {
                assert_eq!(get_i32(&payload[b..b + 4]), joint as i32 * 100);
                assert_eq!(get_u16(&payload[b + 12..b + 14]), 20_000);
                assert_eq!(get_u16(&payload[b + 16..b + 18]), 1);
            }
        }
        assert_eq!(STM32_MOTOR_IDS.into_iter().filter(|id| *id != 0).collect::<Vec<_>>(),
                   CONFIGURED_MOTOR_IDS);
        targets.positions[5] = f64::NAN;
        assert!(encode_targets(&targets, 20_000, 1, 0).is_err());
    }

    #[cfg(unix)]
    fn test_state_payload(enabled: bool) -> Vec<u8> {
        let mut p = vec![0u8; STATE_PREFIX_LEN + STM32_MOTOR_COUNT * MOTOR_OBSERVATION_LEN];
        p[13] = STM32_MOTOR_COUNT as u8;
        put_u16(&mut p[14..16], CONTROL_CAPABILITIES);
        put_f32(&mut p[40..44], 1.0);
        p[60] = IMU_SENSOR_OK | IMU_CALIBRATED;
        for slot in 0..STM32_MOTOR_COUNT {
            if STM32_MOTOR_IDS[slot] != 0 {
                p[STATE_PREFIX_LEN + slot * MOTOR_OBSERVATION_LEN + 16] = if enabled { 7 } else { 3 };
            }
        }
        p
    }

    #[cfg(unix)]
    fn read_test_frame(port: &mut serialport::TTYPort) -> Frame {
        let mut header = [0u8; HEADER_LEN];
        port.read_exact(&mut header).unwrap();
        let len = get_u16(&header[6..8]) as usize;
        let mut wire = header.to_vec();
        wire.resize(HEADER_LEN + len, 0);
        port.read_exact(&mut wire[HEADER_LEN..]).unwrap();
        take_frame(&mut wire).unwrap().unwrap()
    }

    #[cfg(unix)]
    fn respond_to_test_admin(port: &mut serialport::TTYPort, request: &Frame) {
        assert_eq!(request.msg_type, MSG_ADMIN);
        assert_eq!(&request.payload[4..4 + CONFIGURED_MOTOR_IDS.len()], &CONFIGURED_MOTOR_IDS);
        let mut answer = vec![0u8; 28];
        answer[0..3].copy_from_slice(&request.payload[0..3]);
        answer[21] = CONFIGURED_MOTOR_IDS.len() as u8;
        answer[22] = CONFIGURED_MOTOR_IDS.len() as u8;
        port.write_all(&pack(MSG_ADMIN_RESULT, 1, 0, 0, &answer)).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn motor_enable_waits_for_feedback_and_disables_on_missing_confirmation() {
        use serialport::SerialPort;
        for confirm in [true, false] {
            let (mut firmware, mut host) = serialport::TTYPort::pair().unwrap();
            firmware.set_timeout(Duration::from_secs(2)).unwrap();
            host.set_timeout(Duration::from_millis(80)).unwrap();
            let mut io = DynamixelIo::from_port(Box::new(host));
            io.parse_and_observe_state(&test_state_payload(false)).unwrap();
            let peer = std::thread::spawn(move || {
                let request = read_test_frame(&mut firmware);
                assert_eq!(request.payload[2], 1);
                respond_to_test_admin(&mut firmware, &request);
                let mut state = test_state_payload(confirm);
                put_i32(&mut state[STATE_PREFIX_LEN..STATE_PREFIX_LEN + 4], 111);
                firmware.write_all(&pack(MSG_STATE, 2, 0, 0, &state)).unwrap();
                if !confirm {
                    let cleanup = read_test_frame(&mut firmware);
                    assert_eq!(cleanup.payload[2], 2, "failed enable must request disable");
                    respond_to_test_admin(&mut firmware, &cleanup);
                } else {
                    // The enable-confirmation state belongs to the synchronous admin
                    // exchange. A subsequent control read must wait for the next frame.
                    std::thread::sleep(Duration::from_millis(20));
                    put_i32(&mut state[STATE_PREFIX_LEN..STATE_PREFIX_LEN + 4], 222);
                    firmware.write_all(&pack(MSG_STATE, 3, 0, 0, &state)).unwrap();
                }
                // Keep the PTY alive until the reader has consumed the response.
                std::thread::sleep(Duration::from_millis(100));
            });
            assert_eq!(io.set_torque(true).is_ok(), confirm);
            if confirm {
                assert!((io.read_state().unwrap().positions[0] - 0.222).abs() < 1e-9);
            }
            peer.join().unwrap();
        }
    }

    #[test]
    fn calibration_limits_encode_complete_unique_ids_and_signed_milliradians() {
        let mut limits: Vec<_> = CONFIGURED_MOTOR_IDS.iter().map(|&id| MotorPositionLimit {
            motor_id: id, min_mrad: -3000, max_mrad: 4000,
        }).collect();
        let p = encode_position_limits(&limits).unwrap();
        assert_eq!(p.len(), 172);
        assert_eq!(p[2], 14);
        assert_eq!(p[4], 1);
        assert_eq!(get_i32(&p[8..12]), -3000);
        assert_eq!(get_i32(&p[12..16]), 4000);
        limits[1].motor_id = 1;
        assert!(encode_position_limits(&limits).is_err());
        assert!(encode_position_limits(&limits[..4]).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn zero_needs_disabled_feedback_and_a_post_ack_zero_sample() {
        use serialport::SerialPort;
        for enabled in [false, true] {
            let (mut firmware, mut host) = serialport::TTYPort::pair().unwrap();
            firmware.set_timeout(Duration::from_secs(2)).unwrap();
            host.set_timeout(Duration::from_millis(100)).unwrap();
            let mut io = DynamixelIo::from_port(Box::new(host));
            let mut state = test_state_payload(enabled);
            put_u16(&mut state[14..16], 7);
            let peer = std::thread::spawn(move || {
                firmware.write_all(&pack(MSG_STATE, 1, 0, 0, &state)).unwrap();
                if !enabled {
                    let request = read_test_frame(&mut firmware);
                    assert_eq!(request.msg_type, MSG_ADMIN);
                    assert_eq!(&request.payload[2..5], &[3, 1, 2]);
                    let mut ack = vec![0u8; 28];
                    ack[..3].copy_from_slice(&request.payload[..3]);
                    ack[21] = 1; ack[22] = 1;
                    put_u32(&mut ack[4..8], 100);
                    firmware.write_all(&pack(MSG_ADMIN_RESULT, 2, 0, 0, &ack)).unwrap();
                    // First buffered sample is zero but predates the request: it is not proof.
                    put_u32(&mut state[4..8], 110);
                    put_u32(&mut state[64 + MOTOR_OBSERVATION_LEN + 12..64 + MOTOR_OBSERVATION_LEN + 16], 90);
                    firmware.write_all(&pack(MSG_STATE, 3, 0, 0, &state)).unwrap();
                    std::thread::sleep(Duration::from_millis(20));
                    put_u32(&mut state[64 + MOTOR_OBSERVATION_LEN + 12..64 + MOTOR_OBSERVATION_LEN + 16], 110);
                    firmware.write_all(&pack(MSG_STATE, 4, 0, 0, &state)).unwrap();
                }
                std::thread::sleep(Duration::from_millis(120));
                if enabled { assert_eq!(firmware.bytes_to_read().unwrap(), 0, "must not send zero while enabled"); }
            });
            assert_eq!(io.mark_motor_zero(2).is_ok(), !enabled);
            if !enabled {
                assert_eq!(io.last_motor_state.as_ref().unwrap().0.motor_feedback_age_ms[1], 0);
            }
            peer.join().unwrap();
        }
    }

    #[test]
    fn protocol_round_trip_and_crc() {
        let wire = pack(MSG_COMMAND, 7, 1234, 0, &[1, 2, 3]);
        let mut rx = wire.clone();
        let frame = take_frame(&mut rx).unwrap().unwrap();
        assert_eq!(frame.msg_type, MSG_COMMAND); assert_eq!(frame.payload, [1, 2, 3]);
        let mut bad = wire;
        bad[HEADER_LEN] ^= 1;
        bad.extend_from_slice(&pack(MSG_STATE, 8, 1254, 0, &[4, 5]));
        let recovered = take_frame(&mut bad).unwrap().unwrap();
        assert_eq!(recovered.msg_type, MSG_STATE);
        assert_eq!(recovered.payload, [4, 5]);
    }

    #[cfg(unix)]
    #[test]
    fn motor_enable_reports_gateway_fault_details_after_disable_cleanup() {
        use serialport::SerialPort;
        let (mut firmware, mut host) = serialport::TTYPort::pair().unwrap();
        firmware.set_timeout(Duration::from_secs(2)).unwrap();
        host.set_timeout(Duration::from_millis(80)).unwrap();
        let mut io = DynamixelIo::from_port(Box::new(host));
        io.parse_and_observe_state(&test_state_payload(false)).unwrap();
        let peer = std::thread::spawn(move || {
            let request = read_test_frame(&mut firmware);
            let mut answer = vec![0; 28];
            answer[0..3].copy_from_slice(&request.payload[0..3]);
            answer[3] = 8;
            put_u32(&mut answer[8..12], 0x20);
            answer[21] = CONFIGURED_MOTOR_IDS.len() as u8;
            answer[22] = (CONFIGURED_MOTOR_IDS.len() - 1) as u8;
            answer[23] = 14;
            firmware.write_all(&pack(MSG_ADMIN_RESULT, 1, 0, 0x20, &answer)).unwrap();
            let cleanup = read_test_frame(&mut firmware);
            assert_eq!(cleanup.payload[2], 2);
            respond_to_test_admin(&mut firmware, &cleanup);
            std::thread::sleep(Duration::from_millis(100));
        });
        let error = io.set_torque(true).unwrap_err().to_string();
        assert!(error.contains("CAN 发送失败"), "{error}");
        assert!(error.contains("0x00000020") && error.contains("电机 ID 14") && error.contains("13/14"), "{error}");
        peer.join().unwrap();
    }
    #[test]
    fn state_order_is_fourteen_routes_with_no_mouth_slot() {
        let mut p = vec![0u8; STATE_PREFIX_LEN + STM32_MOTOR_COUNT * MOTOR_OBSERVATION_LEN];
        put_u32(&mut p[4..8], 10_000);
        p[13] = STM32_MOTOR_COUNT as u8;
        for (index, value) in [1.0f32, 2.0, 3.0, 0.0, 0.0, -1.0]
            .into_iter()
            .enumerate()
        {
            let offset = 16 + index * 4;
            put_f32(&mut p[offset..offset + 4], value);
        }
        put_f32(&mut p[40..44], 1.0);
        put_f32(&mut p[44..48], 0.0);
        put_f32(&mut p[48..52], 0.0);
        put_f32(&mut p[52..56], 0.0);
        put_u32(&mut p[56..60], 42);
        p[60] = IMU_SENSOR_OK | IMU_CALIBRATED;
        for route in 0..STM32_MOTOR_COUNT {
            let b = STATE_PREFIX_LEN + route * MOTOR_OBSERVATION_LEN;
            put_i32(&mut p[b..b + 4], route as i32 * 100);
            put_i32(&mut p[b + 8..b + 12], route as i32 * 10);
            put_u32(&mut p[b + 12..b + 16], 9_975);
            p[b + 16] = if route < 6 { 0x07 } else { 0x00 };
            p[b + 17] = 30 + route as u8;
        }
        let parsed = parse_state(&p).unwrap();
        for (route, &joint) in STM32_TO_CONTROL_JOINT.iter().enumerate() {
            assert_eq!(parsed.sensors.positions[joint], route as f64 / 10.0);
            assert_eq!(parsed.sensors.motor_torques_nm[joint], route as f64 / 100.0);
            assert_eq!(parsed.temps_c[joint], 30.0 + route as f64);
        }
        assert_eq!(parsed.sensors.positions[crate::model::MOUTH_INDEX], 0.0);
        assert_eq!(parsed.temps_c[crate::model::MOUTH_INDEX], 0.0);
        assert_eq!(parsed.sensors.motor_flags[0], 0x07);
        assert_eq!(parsed.sensors.motor_flags[5], 0x07);
        assert_eq!(parsed.sensors.motor_feedback_age_ms[0], 25);
        assert_eq!(parsed.sensors.motor_feedback_age_ms[5], 25);
        assert_eq!(parsed.sensors.motor_flags[10], 0);
        assert_eq!(parsed.sensors.motor_flags[14], 0);
        assert_eq!(parsed.sensors.motor_feedback_age_ms[14], 0);
        assert_eq!(parsed.sensors.imu.gyro, [1.0, 2.0, 3.0]);
        assert_eq!(parsed.sensors.imu.gravity, [0.0, 0.0, -1.0]);
        assert_eq!(parsed.sensors.imu.quat, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(parsed.imu_sequence, 42);
        assert_eq!(parsed.imu_flags, IMU_SENSOR_OK | IMU_CALIBRATED);
    }

    #[test]
    fn v4_quaternion_is_only_converted_to_rpy_for_telemetry() {
        let rpy = quaternion_to_euler_zyx([0.0, 0.0, 0.0, 1.0]);
        assert!(rpy[0].abs() < 1e-12);
        assert!(rpy[1].abs() < 1e-12);
        assert!((rpy[2] - std::f64::consts::PI).abs() < 1e-12);
    }
}

#[cfg(test)]
mod experiment_wire_tests {
    use super::*;
    #[test]
    fn batch_encodes_all_five_mit_fields_and_keeps_empty_slots_zero() {
        let commands:Vec<_>=CONFIGURED_MOTOR_IDS.iter().map(|&motor_id|duck_ipc_proto::MotorCommand{motor_id,p:-0.5,v:1.25,tau:-2.5,kp:12.34,kd:0.567}).collect();
        let p=encode_motor_commands(&commands,0xabcd,123456).unwrap();
        assert_eq!(get_u16(&p[0..2]),0xabcd);assert_eq!(get_u32(&p[4..8]),123456);
        for (slot,id) in STM32_MOTOR_IDS.iter().enumerate(){let b=8+slot*20;
            if *id==0{assert!(p[b..b+20].iter().all(|x|*x==0));continue}
            assert_eq!(get_i32(&p[b..b+4]),-500);assert_eq!(get_i32(&p[b+4..b+8]),1250);assert_eq!(get_i32(&p[b+8..b+12]),-2500);
            assert_eq!(get_u16(&p[b+12..b+14]),1234);assert_eq!(get_u16(&p[b+14..b+16]),567);assert_eq!(get_u16(&p[b+16..b+18]),1);
        }
        assert!(encode_motor_commands(&commands[..1],1,0).is_err());
    }
}

#[cfg(test)]
mod policy_pd_tests {
    use super::*;
    #[test]
    fn policy_pd_encodes_fractional_values_and_does_not_leak() {
        let mut targets = JointTargets::new([0.0; crate::model::NUM_JOINTS]);
        targets.pd = Some([60.25, 1.234]);
        let payload = encode_targets(&targets, 20_000, 1, 0).unwrap();
        for (slot, id) in STM32_MOTOR_IDS.iter().enumerate() {
            let b = COMMAND_PREFIX_LEN + slot * MOTOR_COMMAND_LEN;
            if *id == 0 { assert!(payload[b..b+MOTOR_COMMAND_LEN].iter().all(|v| *v==0)); }
            else { assert_eq!(get_u16(&payload[b+12..b+14]), 6025); assert_eq!(get_u16(&payload[b+14..b+16]), 1234); }
        }
        targets.pd = None;
        let payload = encode_targets(&targets, 16_000, 2, 0).unwrap();
        let slot = STM32_MOTOR_IDS.iter().position(|id| *id != 0).unwrap();
        let b = COMMAND_PREFIX_LEN + slot * MOTOR_COMMAND_LEN;
        assert_eq!(get_u16(&payload[b+12..b+14]), 16_000);
        assert_eq!(get_u16(&payload[b+14..b+16]), 4_000);
        for pd in [[f64::NAN,4.0],[60.0,f64::INFINITY],[501.0,4.0],[60.0,5.01]] {
            targets.pd=Some(pd); assert!(encode_targets(&targets,20_000,3,0).is_err());
        }
    }

    #[test]
    fn two_file_policy_encodes_per_joint_mit_fields() {
        let mut targets = JointTargets::new([0.0; crate::model::NUM_JOINTS]);
        let mut mit = [crate::io::MitTarget::default(); crate::model::NUM_JOINTS];
        for (joint, target) in mit.iter_mut().enumerate() {
            *target = crate::io::MitTarget {
                position: joint as f64 / 100.0,
                velocity: 1.25,
                torque_ff: -0.5,
                kp: 12.34 + joint as f64,
                kd: 0.567,
            };
            targets.positions[joint] = target.position;
        }
        targets.mit = Some(mit);
        let payload = encode_targets(&targets, 20_000, 7, 99).unwrap();
        for (slot, &joint) in STM32_TO_CONTROL_JOINT.iter().enumerate() {
            if STM32_MOTOR_IDS[slot] == 0 { continue; }
            let base = COMMAND_PREFIX_LEN + slot * MOTOR_COMMAND_LEN;
            assert_eq!(get_i32(&payload[base..base + 4]), (joint as i32) * 10);
            assert_eq!(get_i32(&payload[base + 4..base + 8]), 1250);
            assert_eq!(get_i32(&payload[base + 8..base + 12]), -500);
            assert_eq!(get_u16(&payload[base + 12..base + 14]), ((12.34 + joint as f64) * 100.0).round() as u16);
            assert_eq!(get_u16(&payload[base + 14..base + 16]), 567);
        }
        mit[0].velocity = 100.0;
        targets.mit = Some(mit);
        assert!(encode_targets(&targets, 20_000, 8, 100).is_err());
    }
}
