//! BLE mobile v1: transport-local session, freshness and compact telemetry.
//! Robot state and safety remain owned by robotd. No motor bus access here.
use crate::route::{Lane, Upstream};
use crate::upstream::Pool;
use duck_ipc_proto as proto;
use serde_json::{Value, json};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[cfg(test)]
static TEST_OWNER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

static OWNER: AtomicU64 = AtomicU64::new(0);

pub struct Mobile {
    session: u64,
    epoch: Instant,
    sequence: u64,
    last_move: Option<Instant>,
    last_motion: Option<Instant>,
    last_command: Option<Instant>,
    pub subscribed: bool,
    health_ids: Vec<proto::Id>,
}
impl Default for Mobile {
    fn default() -> Self {
        Self {
            session: 0,
            epoch: Instant::now(),
            sequence: 0,
            last_move: None,
            last_motion: None,
            last_command: None,
            subscribed: false,
            health_ids: Vec::new(),
        }
    }
}
impl Drop for Mobile {
    fn drop(&mut self) {
        let _ = OWNER.compare_exchange(self.session, 0, Ordering::SeqCst, Ordering::SeqCst);
    }
}
impl Mobile {
    pub fn owns(&self) -> bool {
        self.session != 0 && OWNER.load(Ordering::SeqCst) == self.session
    }
    pub fn expired(&self) -> bool {
        self.owns()
            && self
                .last_move
                .is_some_and(|t| t.elapsed() >= Duration::from_millis(500))
    }
    pub async fn stop(&mut self, pool: &mut Pool) {
        if self.owns() {
            let call = proto::Call::RobotMove(proto::MoveParams {
                source: Some(proto::ControlSource::Bluetooth),
                release_source: true,
                ..Default::default()
            });
            let line = serde_json::to_string(&proto::Request::notify(&call)).unwrap();
            if !matches!(
                tokio::time::timeout(
                    Duration::from_millis(100),
                    pool.send(Upstream::Robot, Lane::Prompt, &line)
                )
                .await,
                Ok(Ok(()))
            ) {
                pool.forget(Upstream::Robot, Lane::Prompt);
            }
            OWNER.store(0, Ordering::SeqCst);
            self.last_move = None;
        }
    }
    fn validate(&mut self, p: &Value) -> Result<(), String> {
        if !self.owns() || p["s"].as_u64() != Some(self.session) {
            return Err("请重新接管控制".into());
        }
        let seq = p["q"].as_u64().ok_or("缺少指令序号")?;
        let t = p["t"].as_u64().ok_or("缺少指令时间")?;
        let now = self.epoch.elapsed().as_millis() as u64;
        if seq <= self.sequence || t > now + 30 || now.saturating_sub(t) > 250 {
            return Err("旧指令、重复指令或延迟超过 250 ms".into());
        }
        self.sequence = seq;
        Ok(())
    }
    pub async fn handle(
        &mut self,
        req: &proto::Request,
        authenticated: bool,
        pool: &mut Pool,
    ) -> String {
        let result = self.handle_inner(req, authenticated, pool).await;
        match result {
            Ok(value) => {
                serde_json::to_string(&proto::Response::ok(req.id.clone(), &value)).unwrap()
            }
            Err(reason) => serde_json::to_string(&proto::Response::err(
                req.id.clone(),
                proto::Error::new(proto::code::PERMISSION_DENIED, reason),
            ))
            .unwrap(),
        }
    }
    async fn handle_inner(
        &mut self,
        req: &proto::Request,
        authenticated: bool,
        pool: &mut Pool,
    ) -> Result<Value, String> {
        if !authenticated {
            return Err("请先完成 PIN 认证".into());
        }
        if self.expired() {
            self.stop(pool).await;
        }
        let p = req.params.as_ref().unwrap_or(&Value::Null);
        match req.method.as_str() {
            "ble.info" => {
                return Ok(
                    json!({"v":1,"firmware":env!("CARGO_PKG_VERSION"),"api":proto::API_VERSION,"ms":self.epoch.elapsed().as_millis() as u64,"control_hz":10,"state_hz":2,"deadman_ms":500}),
                );
            }
            "ble.health" => {
                let id = req.id.clone().ok_or("health requires a request id")?;
                if self.health_ids.len() >= 4 {
                    return Err("状态请求过快".into());
                }
                self.forward(pool, proto::Call::RobotHealth, Some(id.clone()))
                    .await?;
                self.health_ids.push(id);
                return Ok(Value::Null);
            }
            "ble.mode" => {
                self.forward(pool, proto::Call::RobotMode, req.id.clone())
                    .await?;
                return Ok(Value::Null);
            }
            "ble.subscribe" => {
                if !self.subscribed {
                    let call = proto::Call::RobotSubscribe(proto::SubscribeParams { hz: Some(2) });
                    let line = serde_json::to_string(&proto::Request::call(
                        proto::Id::Text("ble-subscribe".into()),
                        &call,
                    ))
                    .unwrap();
                    pool.send(Upstream::Robot, Lane::Stream, &line)
                        .await
                        .map_err(|e| e.to_string())?;
                    self.subscribed = true;
                }
                return Ok(json!({"accepted":true}));
            }
            "ble.claim" => {
                if self.owns() {
                    return Err("已有控制会话；请先释放".into());
                }
                // Lease labels are random and renewed even on the same encrypted connection.
                use std::io::Read;
                let mut bytes = [0u8; 4];
                std::fs::File::open("/dev/urandom")
                    .and_then(|mut f| f.read_exact(&mut bytes))
                    .map_err(|e| e.to_string())?;
                self.session = u32::from_le_bytes(bytes).max(1) as u64;
                OWNER
                    .compare_exchange(0, self.session, Ordering::SeqCst, Ordering::SeqCst)
                    .map_err(|_| "机器人已由另一个手机控制")?;
                self.sequence = 0;
                self.last_motion = None;
                self.epoch = Instant::now();
                self.last_move = Some(Instant::now());
                if let Err(e) = self
                    .forward(
                        pool,
                        proto::Call::RobotMove(proto::MoveParams {
                            source: Some(proto::ControlSource::Bluetooth),
                            select_source: true,
                            source_available: Some(true),
                            ..Default::default()
                        }),
                        None,
                    )
                    .await
                {
                    self.stop(pool).await;
                    return Err(e);
                }
                return Ok(json!({"s":self.session,"ms":self.epoch.elapsed().as_millis() as u64}));
            }
            "ble.release" => {
                self.validate(p)?;
                self.stop(pool).await;
                return Ok(json!({"accepted":true}));
            }
            "ble.move" => {
                self.validate(p)?;
                let v = p["v"]
                    .as_array()
                    .filter(|v| v.len() == 3)
                    .ok_or("速度必须为三个整数")?;
                let mut values = [0.0; 3];
                for i in 0..3 {
                    let n = v[i].as_i64().ok_or("速度必须是整数")?;
                    if n.unsigned_abs() > [300, 300, 1000][i] {
                        return Err("超出首版速度上限".into());
                    }
                    values[i] = n as f64 / 1000.0;
                }
                let zero = values == [0.0; 3];
                // Stops bypass the motion rate limit. Never queue a burst of old motion.
                if !zero
                    && self
                        .last_motion
                        .is_some_and(|t| t.elapsed() < Duration::from_millis(80))
                {
                    return Err("移动指令过快".into());
                }
                self.forward(
                    pool,
                    proto::Call::RobotMove(proto::MoveParams {
                        vx: values[0],
                        vy: values[1],
                        vyaw: values[2],
                        source: Some(proto::ControlSource::Bluetooth),
                        ..Default::default()
                    }),
                    None,
                )
                .await?;
                self.last_move = Some(Instant::now());
                if !zero {
                    self.last_motion = self.last_move;
                }
                return Ok(json!({"q":self.sequence}));
            }
            "ble.command" => {
                self.validate(p)?;
                if self
                    .last_command
                    .is_some_and(|t| t.elapsed() < Duration::from_millis(200))
                {
                    return Err("动作指令过快".into());
                }
                let inner: proto::Request = serde_json::from_value(
                    json!({"jsonrpc":"2.0","id":req.id,"method":p["m"],"params":p["p"]}),
                )
                .map_err(|e| e.to_string())?;
                let call = inner.as_call().map_err(|e| e.to_string())?;
                match &call {
                    proto::Call::RobotHead(h)
                        if [h.neck_pitch, h.head_roll].iter().all(|x| *x == 0.0)
                            && h.head_pitch.abs() <= 0.4
                            && h.head_yaw.abs() <= 0.6 => {}
                    proto::Call::RobotEnable(_)
                    | proto::Call::RobotInit
                    | proto::Call::RobotRelax
                    | proto::Call::RobotDo(_)
                    | proto::Call::RobotSound(_)
                    | proto::Call::RobotStop => {}
                    _ => return Err("此指令或角度不在 BLE 首版范围内".into()),
                }
                self.last_command = Some(Instant::now());
                self.forward(pool, call, req.id.clone()).await?;
                // The upstream answers with the original id; no fabricated success here.
                return Ok(Value::Null);
            }
            _ => return Err("未知 BLE App 协议".into()),
        }
    }
    pub fn compact_reply(&mut self, line: String) -> String {
        let Ok(reply) = serde_json::from_str::<proto::Response>(&line) else {
            return line;
        };
        let Some(index) = self
            .health_ids
            .iter()
            .position(|id| Some(id) == reply.id.as_ref())
        else {
            return line;
        };
        self.health_ids.remove(index);
        if reply.error.is_some() {
            return line;
        }
        let Some(result) = reply.result else {
            return line;
        };
        let reason = result["reason"]
            .as_str()
            .map(|s| s.chars().take(160).collect::<String>());
        json!({"jsonrpc":"2.0","id":reply.id,"result":{
            "healthy":result["healthy"],"degraded":result["degraded"],"reason":reason,
            "battery":result["battery"],"cpu_temp_c":result["cpu_temp_c"]
        }})
        .to_string()
    }
    async fn forward(
        &self,
        pool: &mut Pool,
        call: proto::Call,
        id: Option<proto::Id>,
    ) -> Result<(), String> {
        let lane = call.destination().ok_or("无上游")?.1;
        let req = match id {
            Some(id) => proto::Request::call(id, &call),
            None => proto::Request::notify(&call),
        };
        let result = tokio::time::timeout(
            Duration::from_millis(100),
            pool.send(Upstream::Robot, lane, &serde_json::to_string(&req).unwrap()),
        )
        .await;
        match result {
            Ok(result) => result.map_err(|e| e.to_string()),
            Err(_) => {
                pool.forget(Upstream::Robot, lane);
                Err("机器人响应超时".into())
            }
        }
    }
}

/// Compact fixed-order integers: radians/rad-s/Nm scaled by 1000, Celsius by 10.
/// Null remains unknown. Skip mouth (index 9): DMUSB has exactly 14 motor slots.
pub fn compact(line: &str) -> Option<String> {
    let value: Value = serde_json::from_str(line).ok()?;
    if value["method"] != "robot.state" {
        return None;
    }
    let s = &value["params"];
    fn vector(v: &Value, scale: f64) -> Value {
        Value::Array(
            v.as_array()
                .map(|v| {
                    v.iter()
                        .map(|n| {
                            n.as_f64()
                                .map(|x| json!((x * scale).round() as i64))
                                .unwrap_or(Value::Null)
                        })
                        .collect()
                })
                .unwrap_or_default(),
        )
    }
    let motors: Vec<Value> = (0..15)
        .filter(|i| *i != 9)
        .map(|i| {
            json!([
                s["motor_flags"][i],
                s["joints"][i].as_f64().map(|x| (x * 1000.).round() as i64),
                s["motor_velocities"][i]
                    .as_f64()
                    .map(|x| (x * 1000.).round() as i64),
                s["motor_torques_nm"][i]
                    .as_f64()
                    .map(|x| (x * 1000.).round() as i64),
                s["motor_temperatures_c"][i]
                    .as_f64()
                    .map(|x| (x * 10.).round() as i64),
                s["motor_feedback_age_ms"][i]
            ])
        })
        .collect();
    Some(json!({"jsonrpc":"2.0","method":"ble.state","params":{
        "t":s["t"],"policy":s["policy"],"source":s["control_source"],
        "loop":s["loop"],"safety":s["safety"],"error":s["motor_control_error"],
        "req":vector(&s["move"]["requested"],1000.),"actual":vector(&s["odom"]["velocity"],1000.),
        "rpy":vector(&s["imu_rpy"],1000.),"motors":motors
    }}).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn freshness_rejects_duplicate_future_and_stale() {
        let _guard = TEST_OWNER_LOCK.lock().unwrap();
        let mut m = Mobile::default();
        m.session = 101;
        OWNER.store(m.session, Ordering::SeqCst);
        let sid = m.session;
        assert!(m.validate(&json!({"s":sid,"q":1,"t":0})).is_ok());
        assert!(m.validate(&json!({"s":sid,"q":1,"t":0})).is_err());
        assert!(m.validate(&json!({"s":sid,"q":2,"t":1000})).is_err());
        assert!(m.validate(&json!({"s":sid+1,"q":2,"t":0})).is_err());
        m.epoch = Instant::now() - Duration::from_millis(300);
        assert!(m.validate(&json!({"s":sid,"q":2,"t":0})).is_err());
        m.last_move = Some(Instant::now() - Duration::from_millis(501));
        assert!(m.expired());
    }
    #[test]
    fn telemetry_has_fourteen_slots_and_preserves_unknowns() {
        let raw=json!({"method":"robot.state","params":{"motor_flags":[1,2,3,4,5,6,7,8,9,0,11,12,13,14,15]}}).to_string();
        let result: Value = serde_json::from_str(&compact(&raw).unwrap()).unwrap();
        let motors = result["params"]["motors"].as_array().unwrap();
        assert_eq!(motors.len(), 14);
        assert_eq!(motors[9][0], 11);
        assert!(motors[0][1].is_null());
    }
}

/// One unfragmented ATT write, exactly 20 bytes. Integers are little-endian.
pub fn move_request(bytes: &[u8]) -> Option<proto::Request> {
    if bytes.len() != 20 || bytes[..2] != [0xbd, 1] {
        return None;
    }
    let u32_at = |i| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
    let i16_at = |i| i16::from_le_bytes(bytes[i..i + 2].try_into().unwrap());
    Some(proto::Request {
        jsonrpc: "2.0".into(),
        id: None,
        method: "ble.move".into(),
        params: Some(
            json!({"s":u32_at(2),"q":u32_at(6),"t":u32_at(10),"v":[i16_at(14),i16_at(16),i16_at(18)]}),
        ),
    })
}

/// Fixed 188-byte state frame, chunked with a 4-byte header into <=20-byte ATT values.
/// JSON health/mode supplies the slow optional board values independently.
pub fn state_packet(line: &str) -> Option<Vec<u8>> {
    let compact: Value = serde_json::from_str(&compact(line)?).ok()?;
    let s = &compact["params"];
    let mut out = Vec::with_capacity(188);
    out.push(1); // binary state version
    out.push(match s["policy"].as_str() {
        Some("walk") => 1,
        Some("stand") => 2,
        Some("held") => 3,
        _ => 0,
    });
    out.push(match s["source"].as_str() {
        Some("bluetooth") => 1,
        Some("gamepad") => 2,
        Some("keyboard") => 3,
        Some("drag") => 4,
        _ => 0,
    });
    out.push(
        u8::from(s["safety"]["fallen"] == true) | (u8::from(s["safety"]["limp"] == true) << 1),
    );
    out.extend_from_slice(&((s["t"].as_f64().unwrap_or(0.) * 1000.) as u32).to_le_bytes());
    fn short(out: &mut Vec<u8>, value: Option<f64>) {
        let n = value
            .filter(|x| x.is_finite())
            .map(|x| {
                if (-32767.0..=32767.0).contains(&x) {
                    x.round() as i16
                } else {
                    i16::MIN
                }
            })
            .unwrap_or(i16::MIN);
        out.extend_from_slice(&n.to_le_bytes());
    }
    short(&mut out, s["loop"]["hz"].as_f64().map(|x| x * 10.));
    for key in ["req", "actual", "rpy"] {
        for i in 0..3 {
            short(&mut out, s[key][i].as_f64());
        }
    }
    for i in 0..3 {
        short(
            &mut out,
            s["safety"]["gravity"][i].as_f64().map(|x| x * 1000.),
        );
    }
    for i in 0..14 {
        let m = &s["motors"][i];
        out.push(m[0].as_u64().unwrap_or(0) as u8);
        for j in 1..5 {
            short(&mut out, m[j].as_f64());
        }
        out.extend_from_slice(&(m[5].as_u64().unwrap_or(65535).min(65535) as u16).to_le_bytes());
    }
    // 34-byte header + 14 * 11-byte motors = 188 bytes.
    Some(out)
}

#[cfg(test)]
mod wire_tests {
    use super::*;
    #[test]
    fn speed_golden_and_malformed() {
        let frame = [
            0xbd, 1, 1, 0, 0, 0, 2, 0, 0, 0, 100, 0, 0, 0, 100, 0, 156, 255, 232, 3,
        ];
        let p = move_request(&frame).unwrap().params.unwrap();
        assert_eq!(p["v"], json!([100, -100, 1000]));
        assert_eq!(p["q"], 2);
        assert!(move_request(&frame[..19]).is_none());
        let mut bad = frame;
        bad[1] = 2;
        assert!(move_request(&bad).is_none());
    }
    #[test]
    fn fixed_state_length_and_unknown_marker() {
        let frame = state_packet(r#"{"method":"robot.state","params":{}}"#).unwrap();
        assert_eq!(frame.len(), 188);
        assert_eq!(&frame[8..10], &i16::MIN.to_le_bytes());
    }
}

#[cfg(test)]
mod safety_integration {
    use super::*;
    use tokio::io::{AsyncBufReadExt, BufReader};
    #[tokio::test]
    async fn lease_auth_replay_rate_limits_and_timeout_use_real_ipc() {
        let _guard = TEST_OWNER_LOCK.lock().unwrap();
        OWNER.store(0, Ordering::SeqCst);
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("robot.sock");
        let listener = tokio::net::UnixListener::bind(&socket).unwrap();
        let (seen, mut received) = tokio::sync::mpsc::channel::<Value>(32);
        let server = tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let seen = seen.clone();
                tokio::spawn(async move {
                    let mut lines = BufReader::new(stream).lines();
                    while let Ok(Some(line)) = lines.next_line().await {
                        seen.send(serde_json::from_str(&line).unwrap())
                            .await
                            .unwrap();
                    }
                });
            }
        });
        let (reply, _) = tokio::sync::mpsc::channel(8);
        let mut pool = Pool::new(
            crate::upstream::Sockets {
                robot: socket.clone(),
                config: socket.clone(),
                updater: socket,
            },
            reply,
        );
        fn request(method: &str, params: Value) -> proto::Request {
            proto::Request {
                jsonrpc: "2.0".into(),
                id: Some(proto::Id::Number(7)),
                method: method.into(),
                params: Some(params),
            }
        }
        let mut first = Mobile::default();
        let claim = request("ble.claim", json!({}));
        assert!(first.handle_inner(&claim, false, &mut pool).await.is_err());
        assert!(received.try_recv().is_err());
        let opened = first.handle_inner(&claim, true, &mut pool).await.unwrap();
        let sid = opened["s"].as_u64().unwrap();
        let select = received.recv().await.unwrap();
        assert_eq!(select["params"]["source"], "bluetooth");
        assert_eq!(select["params"]["select_source"], true);
        assert_eq!(select["params"]["source_available"], true);
        let mut second = Mobile::default();
        assert!(second.handle_inner(&claim, true, &mut pool).await.is_err());
        let command = |q, v| request("ble.move", json!({"s":sid,"q":q,"t":100,"v":v}));
        first.epoch = Instant::now() - Duration::from_millis(100);
        first.last_move = Some(Instant::now() - Duration::from_millis(100));
        assert!(
            first
                .handle_inner(&command(1, json!([100, -100, 200])), true, &mut pool)
                .await
                .is_ok()
        );
        let movement = received.recv().await.unwrap();
        assert_eq!(movement["params"]["vx"], 0.1);
        assert_eq!(movement["params"]["vy"], -0.1);
        assert!(
            first
                .handle_inner(&command(1, json!([100, 0, 0])), true, &mut pool)
                .await
                .is_err()
        );
        assert!(
            first
                .handle_inner(&command(2, json!([100, 0, 0])), true, &mut pool)
                .await
                .is_err()
        );
        // A release zero is valid even in the same interval as the prior movement.
        assert!(
            first
                .handle_inner(&command(3, json!([0, 0, 0])), true, &mut pool)
                .await
                .is_ok()
        );
        assert_eq!(received.recv().await.unwrap()["params"]["vx"], 0.0);
        assert!(
            first
                .handle_inner(&command(4, json!([301, 0, 0])), true, &mut pool)
                .await
                .is_err()
        );
        assert!(received.try_recv().is_err());
        // Expired commands cannot revive the lease before the next timer tick.
        first.last_move = Some(Instant::now() - Duration::from_millis(501));
        assert!(
            first
                .handle_inner(&command(5, json!([0, 0, 0])), true, &mut pool)
                .await
                .is_err()
        );
        assert_eq!(received.recv().await.unwrap()["params"]["vx"], 0.0);
        assert!(!first.owns());
        let new_session = first.handle_inner(&claim, true, &mut pool).await.unwrap()["s"]
            .as_u64()
            .unwrap();
        assert_ne!(sid, new_session);
        received.recv().await.unwrap();
        assert!(
            first
                .handle_inner(&command(6, json!([0, 0, 0])), true, &mut pool)
                .await
                .is_err()
        );
        let release = request("ble.release", json!({"s":new_session,"q":1,"t":0}));
        assert!(first.handle_inner(&release, true, &mut pool).await.is_ok());
        let released = received.recv().await.unwrap();
        assert_eq!(released["params"]["vx"], 0.0);
        assert_eq!(released["params"]["release_source"], true);
        assert!(released["params"]["source_available"].is_null(), "release must not pretend the transport disconnected");
        assert!(second.handle_inner(&claim, true, &mut pool).await.is_ok());
        second.stop(&mut pool).await;
        server.abort();
    }
}
