//! Durable precomputed tasks. Disk work is confined to IPC blocking workers and
//! one bounded reader/recorder worker; the bus-owning tick only try_recv/try_send.
use crate::calibration::Config;
use duck_control::bus::validate_motor_commands;
use duck_ipc_proto::MotorCommand;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub const ROOT: &str = "/var/lib/robotd/experiments";
pub const MAX_INPUT: u64 = 128 * 1024 * 1024;
const MAX_RESULT: u64 = 512 * 1024 * 1024;
const MAX_STORE: u64 = 2 * 1024 * 1024 * 1024;
const DISK_MARGIN: u64 = 512 * 1024 * 1024;
const MAX_TASKS: usize = 8;
const ROW_BYTES: u64 = 4096;
const RECORD_BYTES: u64 = 4096;
const BUFFER: usize = 64;

type Result<T> = std::result::Result<T, String>;
fn err(e: impl ToString) -> String {
    e.to_string()
}
fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
#[cfg(test)]
pub fn digest(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
pub fn random_id() -> Result<String> {
    let mut b = [0u8; 16];
    File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut b))
        .map_err(err)?;
    Ok(b.iter().map(|b| format!("{b:02x}")).collect())
}
fn file_hash(path: &Path) -> Result<String> {
    let mut f = File::open(path).map_err(err)?;
    let mut hash = Sha256::new();
    let mut buf = [0; 16384];
    loop {
        let n = f.read(&mut buf).map_err(err)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    Ok(hash.finalize().iter().map(|b| format!("{b:02x}")).collect())
}
fn json_save(path: &Path, value: &impl Serialize) -> Result<()> {
    let tmp = path.with_extension("tmp");
    let mut f = File::create(&tmp).map_err(err)?;
    serde_json::to_writer(&mut f, value).map_err(err)?;
    f.sync_all().map_err(err)?;
    fs::rename(tmp, path).map_err(err)?;
    File::open(path.parent().unwrap())
        .and_then(|d| d.sync_all())
        .map_err(err)
}
fn line<T: serde::de::DeserializeOwned>(reader: &mut BufReader<File>) -> Result<T> {
    let mut bytes = Vec::new();
    reader
        .take(ROW_BYTES + 1)
        .read_until(b'\n', &mut bytes)
        .map_err(err)?;
    if bytes.is_empty() || bytes.len() as u64 > ROW_BYTES || bytes.last() != Some(&b'\n') {
        return Err("missing/truncated line or line exceeds 4096 bytes".into());
    }
    serde_json::from_slice(&bytes).map_err(err)
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Header {
    pub version: u8,
    pub period_ms: u64,
    pub duration_ms: u64,
    pub motor_ids: Vec<u8>,
    pub start_tolerance_rad: f64,
    pub max_tracking_error_rad: f64,
    pub max_temperature_c: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub at_ms: u64,
    pub motors: Vec<MotorCommand>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Meta {
    pub id: String,
    pub state: String,
    pub created_unix: u64,
    pub input_bytes: u64,
    pub input_sha256: String,
    pub received_bytes: u64,
    pub header: Option<Header>,
    pub frames: u64,
    pub executed_frames: u64,
    pub reason: Option<String>,
    pub disabled_confirmed: bool,
    pub result_sha256: Option<String>,
    pub result_bytes: Option<u64>,
}
impl Meta {
    fn reserve(&self) -> u64 {
        if self.result_sha256.is_some() {
            return self.input_bytes
                + self.result_bytes.unwrap_or(0)
                + self.frames * RECORD_BYTES
                + 1024 * 1024;
        }
        self.input_bytes * 2
            + if self.header.is_some() {
                self.frames * RECORD_BYTES * 2 + 1024 * 1024
            } else {
                MAX_RESULT * 2
            }
    }
}
pub struct Store {
    root: PathBuf,
    api: Mutex<()>,
    runs: Mutex<Vec<Arc<RunShared>>>,
}
impl Store {
    pub fn open(root: PathBuf) -> Result<Self> {
        fs::create_dir_all(&root).map_err(err)?;
        let store = Self {
            root,
            api: Mutex::new(()),
            runs: Mutex::new(Vec::new()),
        };
        // Never resume motion after process restart. Keep partial data for retrieval.
        for mut m in store.metas()? {
            if matches!(
                m.state.as_str(),
                "preparing" | "running" | "stopping" | "finalizing"
            ) {
                m.state = "interrupted".into();
                m.reason = Some("robotd restarted; task never resumes automatically".into());
                m.disabled_confirmed = false;
                store.save(&m)?;
                store.package(&mut m)?;
            }
        }
        Ok(store)
    }
    fn path(&self, id: &str) -> Result<PathBuf> {
        if id.len() != 32
            || !id
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        {
            return Err("invalid task id".into());
        }
        Ok(self.root.join(id))
    }
    fn load(&self, id: &str) -> Result<Meta> {
        serde_json::from_reader(File::open(self.path(id)?.join("status.json")).map_err(err)?)
            .map_err(err)
    }
    fn save(&self, m: &Meta) -> Result<()> {
        json_save(&self.path(&m.id)?.join("status.json"), m)
    }
    fn metas(&self) -> Result<Vec<Meta>> {
        let mut out = Vec::new();
        for e in fs::read_dir(&self.root).map_err(err)? {
            let e = e.map_err(err)?;
            if !e.file_type().map_err(err)?.is_dir() {
                continue;
            }
            if let Ok(m) = self.load(&e.file_name().to_string_lossy()) {
                out.push(m)
            }
        }
        Ok(out)
    }
    fn space(&self, reserve: u64, excluding: Option<&str>) -> Result<()> {
        let all = self.metas()?;
        if all
            .iter()
            .filter(|m| Some(m.id.as_str()) != excluding)
            .map(Meta::reserve)
            .sum::<u64>()
            + reserve
            > MAX_STORE
        {
            return Err("task storage quota exceeded; download and acknowledge old results".into());
        }
        let path = std::ffi::CString::new(self.root.as_os_str().as_encoded_bytes()).map_err(err)?;
        let mut st = std::mem::MaybeUninit::<libc::statvfs>::uninit();
        if unsafe { libc::statvfs(path.as_ptr(), st.as_mut_ptr()) } != 0 {
            return Err(err(std::io::Error::last_os_error()));
        }
        let st = unsafe { st.assume_init() };
        if st.f_bavail as u64 * (st.f_frsize as u64) < reserve + DISK_MARGIN {
            return Err("insufficient free disk for task and complete result".into());
        }
        Ok(())
    }
    pub fn list(&self) -> Result<Value> {
        let _lock = self.api.lock().unwrap();
        let rows = self
            .metas()?
            .iter()
            .map(|m| self.status_unlocked(&m.id))
            .collect::<Result<Vec<_>>>()?;
        Ok(
            json!({"tasks":rows,"limits":{"max_input_bytes":MAX_INPUT,"max_result_bytes":MAX_RESULT,"max_store_bytes":MAX_STORE,"max_tasks":MAX_TASKS,"max_duration_ms":1_800_000,"period_ms":20,"buffer_frames":BUFFER},"disconnect_policy":"continue","end_behavior":"disable_all"}),
        )
    }
    fn status_unlocked(&self, id: &str) -> Result<Value> {
        if let Some(r) = self.runs.lock().unwrap().iter().find(|r| r.id == id) {
            return serde_json::to_value(&*r.meta.lock().unwrap()).map_err(err);
        }
        let mut m = self.load(id)?;
        if m.state == "uploading" {
            m.received_bytes = fs::metadata(self.path(id)?.join("input.jsonl"))
                .map_err(err)?
                .len();
        }
        serde_json::to_value(m).map_err(err)
    }
    pub fn status(&self, id: &str) -> Result<Value> {
        let _lock = self.api.lock().unwrap();
        self.status_unlocked(id)
    }
    pub fn begin(&self, bytes: u64, sha256: &str) -> Result<Value> {
        let _lock = self.api.lock().unwrap();
        if bytes == 0
            || bytes > MAX_INPUT
            || sha256.len() != 64
            || !sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        {
            return Err("invalid upload size/SHA-256".into());
        }
        // Abandoned partial uploads are disposable; ready/running/results are never evicted.
        for m in self.metas()? {
            if m.state == "uploading" && now().saturating_sub(m.created_unix) > 3600 {
                fs::remove_dir_all(self.path(&m.id)?).map_err(err)?
            }
        }
        if self.metas()?.len() >= MAX_TASKS {
            return Err("task count limit reached".into());
        }
        let id = random_id()?;
        let m = Meta {
            id: id.clone(),
            state: "uploading".into(),
            created_unix: now(),
            input_bytes: bytes,
            input_sha256: sha256.into(),
            received_bytes: 0,
            header: None,
            frames: 0,
            executed_frames: 0,
            reason: None,
            disabled_confirmed: false,
            result_sha256: None,
            result_bytes: None,
        };
        self.space(m.reserve(), None)?;
        fs::create_dir(self.path(&id)?).map_err(err)?;
        File::create(self.path(&id)?.join("input.jsonl")).map_err(err)?;
        self.save(&m)?;
        Ok(json!({"id":id,"state":"uploading"}))
    }
    pub fn chunk(&self, id: &str, offset: u64, data: &[u8]) -> Result<Value> {
        let _lock = self.api.lock().unwrap();
        let mut m = self.load(id)?;
        m.received_bytes = fs::metadata(self.path(id)?.join("input.jsonl"))
            .map_err(err)?
            .len();
        if m.state != "uploading"
            || data.is_empty()
            || data.len() > 8192
            || offset != m.received_bytes
            || offset + data.len() as u64 > m.input_bytes
        {
            return Err("invalid upload state, offset or chunk size".into());
        }
        OpenOptions::new()
            .append(true)
            .open(self.path(id)?.join("input.jsonl"))
            .and_then(|mut f| f.write_all(data))
            .map_err(err)?;
        m.received_bytes += data.len() as u64;
        Ok(json!({"received_bytes":m.received_bytes}))
    }
    pub fn abort_upload(&self, id: &str) -> Result<Value> {
        let _lock = self.api.lock().unwrap();
        let m = self.load(id)?;
        if m.state != "uploading" {
            return Err("only unfinished uploads can be aborted".into());
        }
        fs::remove_dir_all(self.path(id)?).map_err(err)?;
        Ok(json!({"deleted":true}))
    }
    pub fn cancel_ready(&self, id: &str) -> Result<Value> {
        let _lock = self.api.lock().unwrap();
        let mut m = self.load(id)?;
        if m.state == "ready" {
            m.state = "cancelled".into();
            m.reason = Some("cancelled before starting; no motor commands executed".into());
            self.save(&m)?;
            self.package(&mut m)?;
        }
        self.status_unlocked(id)
    }
    fn validate(&self, m: &Meta, cfg: &Config, hz: u32) -> Result<(Header, u64)> {
        if m.received_bytes != m.input_bytes {
            return Err("incomplete upload".into());
        }
        let path = self.path(&m.id)?.join("input.jsonl");
        if file_hash(&path)? != m.input_sha256 {
            return Err("input SHA-256 mismatch".into());
        }
        let mut reader = BufReader::new(File::open(path).map_err(err)?);
        let h: Header = line(&mut reader)?;
        if h.version != 1
            || hz != 50
            || h.period_ms != 20
            || h.duration_ms == 0
            || h.duration_ms > 1_800_000
            || h.duration_ms % 20 != 0
        {
            return Err(
                "require v1, 20 ms periods and duration 20..1800000 ms divisible by 20".into(),
            );
        }
        if h.motor_ids.is_empty()
            || h.motor_ids.len() > cfg.limits.len()
            || h.motor_ids.iter().enumerate().any(|(i, id)| {
                h.motor_ids[..i].contains(id) || !cfg.limits.iter().any(|l| l.motor_id == *id)
            })
        {
            return Err("unknown, duplicate or empty motor set".into());
        }
        if !h.start_tolerance_rad.is_finite()
            || !(0.0..=0.5).contains(&h.start_tolerance_rad)
            || h.start_tolerance_rad == 0.0
            || !h.max_tracking_error_rad.is_finite()
            || h.max_tracking_error_rad <= 0.0
            || h.max_tracking_error_rad > 3.0
            || !h.max_temperature_c.is_finite()
            || !(1.0..=80.0).contains(&h.max_temperature_c)
        {
            return Err("invalid explicit start/tracking/temperature limits".into());
        }
        let n = h.duration_ms / 20;
        for i in 0..n {
            let row: Row = line(&mut reader)?;
            if row.at_ms != i * 20 {
                return Err(format!("frame {i}: at_ms must equal {}", i * 20));
            }
            validate_motor_commands(&row.motors, &cfg.wire_limits()).map_err(err)?;
            if row.motors.len() != h.motor_ids.len()
                || row
                    .motors
                    .iter()
                    .any(|m| !h.motor_ids.contains(&m.motor_id))
            {
                return Err(format!("frame {i}: complete selected motor set required"));
            }
        }
        if !reader.fill_buf().map_err(err)?.is_empty() {
            return Err("unexpected trailing frames or bytes".into());
        }
        Ok((h, n))
    }
    pub fn commit(&self, id: &str, cfg: &Config, hz: u32) -> Result<Value> {
        let _lock = self.api.lock().unwrap();
        let mut m = self.load(id)?;
        if m.state != "uploading" {
            return self.status_unlocked(id);
        }
        m.received_bytes = fs::metadata(self.path(id)?.join("input.jsonl"))
            .map_err(err)?
            .len();
        File::open(self.path(id)?.join("input.jsonl"))
            .and_then(|f| f.sync_all())
            .map_err(err)?;
        let (h, n) = self.validate(&m, cfg, hz)?;
        m.header = Some(h);
        m.frames = n;
        self.space(m.reserve(), Some(id))?;
        m.state = "ready".into();
        self.save(&m)?;
        serde_json::to_value(m).map_err(err)
    }
    pub fn prepare(self: &Arc<Self>, id: &str, cfg: &Config, hz: u32) -> Result<Run> {
        let _lock = self.api.lock().unwrap();
        let mut m = self.load(id)?;
        if m.state != "ready" {
            return Err("task must be ready and can start only once".into());
        }
        let (header, _) = self.validate(&m, cfg, hz)?;
        self.space(m.reserve(), Some(id))?;
        let mut reader =
            BufReader::new(File::open(self.path(id)?.join("input.jsonl")).map_err(err)?);
        let _: Header = line(&mut reader)?;
        let first: Row = line(&mut reader)?;
        m.state = "preparing".into();
        self.save(&m)?;
        let shared = Arc::new(RunShared {
            id: id.into(),
            meta: Mutex::new(m),
            finish: Mutex::new(None),
            failure: Mutex::new(None),
        });
        self.runs.lock().unwrap().push(shared.clone());
        let (tx, rx) = mpsc::sync_channel(BUFFER);
        let (records, record_rx) = mpsc::sync_channel(BUFFER);
        let store = self.clone();
        let worker = shared.clone();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let first_clone = first.clone();
        std::thread::Builder::new()
            .name("experiment-files".into())
            .spawn(move || {
                let result = store.worker(&worker, reader, tx, record_rx, ready_tx);
                if let Err(e) = result {
                    *worker.failure.lock().unwrap() = Some(e.clone());
                    // Publish the failure without any disk I/O under the control-visible lock.
                    loop {
                        if worker.finish.lock().unwrap().is_some() {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    let mut meta = worker.meta.lock().unwrap().clone();
                    meta.reason = Some(e);
                    meta.state = "storage_failed".into();
                    meta.disabled_confirmed =
                        worker.finish.lock().unwrap().as_ref().is_some_and(|v| v.2);
                    let _ = store.save(&meta);
                    let _ = store.package(&mut meta);
                    *worker.meta.lock().unwrap() = meta;
                }
            })
            .map_err(err)?;
        if ready_rx.recv_timeout(Duration::from_secs(3)).is_err() {
            *shared.finish.lock().unwrap() = Some((
                "failed".into(),
                "task prefetch failed before control started".into(),
                false,
            ));
            return Err("task prefetch failed".into());
        }
        Ok(Run {
            shared,
            header,
            first: first_clone,
            pending: Some(first),
            last_row: None,
            rows: rx,
            records: Some(records),
            index: 0,
            started: None,
            last_tick: None,
            cfg_signature: serde_json::to_string(&cfg.limits).map_err(err)?,
        })
    }
    fn worker(
        &self,
        shared: &Arc<RunShared>,
        mut reader: BufReader<File>,
        tx: SyncSender<Row>,
        rx: Receiver<Value>,
        ready: SyncSender<()>,
    ) -> Result<()> {
        let path = self.path(&shared.id)?;
        let n = shared.meta.lock().unwrap().frames;
        let mut next = 1;
        let mut pending = None;
        let mut initialized = false;
        let mut writer = BufWriter::with_capacity(
            16384,
            File::create(path.join("records.jsonl")).map_err(err)?,
        );
        let mut bytes = 0;
        loop {
            while next < n {
                let row = match pending.take() {
                    Some(r) => r,
                    None => line(&mut reader)?,
                };
                match tx.try_send(row) {
                    Ok(()) => next += 1,
                    Err(TrySendError::Full(row)) => {
                        pending = Some(row);
                        break;
                    }
                    Err(TrySendError::Disconnected(_)) => {
                        next = n;
                        break;
                    }
                }
            }
            if !initialized {
                let _ = ready.send(());
                initialized = true;
            }
            match rx.recv_timeout(Duration::from_millis(5)) {
                Ok(value) => {
                    let data = serde_json::to_vec(&value).map_err(err)?;
                    if data.len() as u64 > RECORD_BYTES
                        || bytes + data.len() as u64 + 1 > MAX_RESULT
                    {
                        return Err("result recording size limit exceeded".into());
                    }
                    writer
                        .write_all(&data)
                        .and_then(|_| writer.write_all(b"\n"))
                        .map_err(err)?;
                    bytes += data.len() as u64 + 1;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        writer.flush().map_err(err)?;
        writer.get_ref().sync_all().map_err(err)?;
        let outcome = shared.finish.lock().unwrap().clone().unwrap_or((
            "failed".into(),
            "execution handle disappeared".into(),
            false,
        ));
        let mut m = shared.meta.lock().unwrap().clone();
        m.state = outcome.0;
        m.reason = Some(outcome.1);
        m.disabled_confirmed = outcome.2;
        self.save(&m)?;
        self.package(&mut m)?;
        *shared.meta.lock().unwrap() = m;
        Ok(())
    }
    fn package(&self, m: &mut Meta) -> Result<()> {
        let path = self.path(&m.id)?;
        let temp = path.join("result.tar.tmp");
        let mut tar = tar::Builder::new(File::create(&temp).map_err(err)?);
        for name in ["status.json", "input.jsonl", "records.jsonl"] {
            if path.join(name).exists() {
                tar.append_path_with_name(path.join(name), name)
                    .map_err(err)?
            }
        }
        tar.finish().map_err(err)?;
        tar.into_inner().map_err(err)?.sync_all().map_err(err)?;
        fs::rename(&temp, path.join("result.tar")).map_err(err)?;
        m.result_sha256 = Some(file_hash(&path.join("result.tar"))?);
        m.result_bytes = Some(fs::metadata(path.join("result.tar")).map_err(err)?.len());
        self.save(m)
    }
    pub fn download(&self, id: &str) -> Result<(File, Meta)> {
        let _lock = self.api.lock().unwrap();
        let m = self.load(id)?;
        if m.result_sha256.is_none() {
            return Err("result not finalized".into());
        }
        Ok((
            File::open(self.path(id)?.join("result.tar")).map_err(err)?,
            m,
        ))
    }
    pub fn delete(&self, id: &str, sha: &str) -> Result<Value> {
        let _lock = self.api.lock().unwrap();
        let m = self.load(id)?;
        if m.result_sha256.as_deref() != Some(sha) || sha.is_empty() {
            return Err("download, verify, then acknowledge the exact result SHA-256".into());
        }
        fs::remove_dir_all(self.path(id)?).map_err(err)?;
        self.runs.lock().unwrap().retain(|r| r.id != id);
        Ok(json!({"deleted":true,"id":id}))
    }
}
pub struct RunShared {
    pub id: String,
    meta: Mutex<Meta>,
    finish: Mutex<Option<(String, String, bool)>>,
    failure: Mutex<Option<String>>,
}
pub struct Run {
    shared: Arc<RunShared>,
    pub header: Header,
    pub first: Row,
    pending: Option<Row>,
    last_row: Option<Row>,
    rows: Receiver<Row>,
    records: Option<SyncSender<Value>>,
    pub index: u64,
    pub started: Option<Instant>,
    last_tick: Option<Instant>,
    cfg_signature: String,
}
impl Drop for Run {
    fn drop(&mut self) {
        let mut finish = self.shared.finish.lock().unwrap();
        if finish.is_none() {
            *finish = Some((
                "interrupted".into(),
                "control handle dropped; no automatic resume".into(),
                false,
            ))
        }
        self.records.take();
    }
}
impl Run {
    pub fn check_config(&self, cfg: &Config) -> Result<()> {
        if let Some(e) = self.shared.failure.lock().unwrap().clone() {
            return Err(e);
        }
        if self.cfg_signature != serde_json::to_string(&cfg.limits).map_err(err)? {
            return Err("calibration changed since task validation".into());
        }
        Ok(())
    }
    pub fn next(&mut self, cfg: &Config) -> Result<Option<(Row, bool)>> {
        if let Some(e) = self.shared.failure.lock().unwrap().clone() {
            return Err(e);
        }
        self.check_config(cfg)?;
        let now = Instant::now();
        if self
            .last_tick
            .is_some_and(|t| now.duration_since(t) > Duration::from_millis(40))
        {
            return Err("local control interval exceeded 40 ms".into());
        }
        if self
            .started
            .is_some_and(|t| now.duration_since(t) > Duration::from_millis(self.index * 20 + 40))
        {
            return Err("local task schedule exceeded 40 ms lateness".into());
        }
        if self
            .started
            .is_some_and(|t| t.elapsed() < Duration::from_millis(self.index * 20))
        {
            self.last_tick = Some(now);
            return Ok(self.last_row.clone().map(|r| (r, false)));
        }
        if self.index >= self.header.duration_ms / 20 {
            return Ok(None);
        }
        let row = if let Some(r) = self.pending.take() {
            r
        } else {
            match self.rows.try_recv() {
                Ok(r) => r,
                Err(TryRecvError::Empty) => return Err("command prefetch underrun".into()),
                Err(TryRecvError::Disconnected) => return Err("command reader stopped".into()),
            }
        };
        self.started.get_or_insert(now);
        self.last_tick = Some(now);
        if row.at_ms != self.index * 20 {
            return Err("command sequence changed or out of order".into());
        }
        self.last_row = Some(row.clone());
        Ok(Some((row, true)))
    }
    pub fn record(&mut self, value: Value, advance: bool) -> Result<()> {
        self.records
            .as_ref()
            .ok_or("task recorder closed")?
            .try_send(value)
            .map_err(|e| format!("task recording backpressure or failure: {e}"))?;
        if advance {
            self.index += 1;
        }
        let mut m = self.shared.meta.lock().unwrap();
        m.state = "running".into();
        m.executed_frames = self.index;
        Ok(())
    }
    pub fn finish(&mut self, state: &str, reason: &str, disabled: bool) {
        *self.shared.finish.lock().unwrap() = Some((state.into(), reason.into(), disabled));
        self.shared.meta.lock().unwrap().state = "finalizing".into();
        self.records.take();
    }
}

#[cfg(test)]
pub(crate) fn fixture(store: &Store, cfg: &Config, frames: u64, p: f64) -> String {
    let header = Header {
        version: 1,
        period_ms: 20,
        duration_ms: frames * 20,
        motor_ids: vec![2],
        start_tolerance_rad: 0.1,
        max_tracking_error_rad: 0.5,
        max_temperature_c: 60.0,
    };
    let mut bytes = serde_json::to_vec(&header).unwrap();
    bytes.push(b'\n');
    for index in 0..frames {
        let row = Row {
            at_ms: index * 20,
            motors: vec![MotorCommand {
                motor_id: 2,
                p,
                v: 0.0,
                tau: 0.0,
                kp: 0.0,
                kd: 0.0,
            }],
        };
        serde_json::to_writer(&mut bytes, &row).unwrap();
        bytes.push(b'\n');
    }
    let id = store.begin(bytes.len() as u64, &digest(&bytes)).unwrap()["id"]
        .as_str()
        .unwrap()
        .to_string();
    for (i, data) in bytes.chunks(8192).enumerate() {
        store.chunk(&id, (i * 8192) as u64, data).unwrap();
    }
    store.commit(&id, cfg, 50).unwrap();
    id
}
#[cfg(test)]
mod tests {
    use super::*;
    fn setup() -> (tempfile::TempDir, Arc<Store>, Config) {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(Store::open(dir.path().into()).unwrap());
        (dir, store, Config::default())
    }
    #[test]
    fn upload_cancel_download_hash_acknowledge_and_no_early_delete() {
        let (_dir, store, cfg) = setup();
        let id = fixture(&store, &cfg, 2, 0.0);
        assert_eq!(store.status(&id).unwrap()["state"], "ready");
        assert!(store.delete(&id, "bad").is_err());
        assert!(store.download(&id).is_err());
        store.cancel_ready(&id).unwrap();
        let (_, meta) = store.download(&id).unwrap();
        assert_eq!(meta.state, "cancelled");
        assert!(meta.result_bytes.unwrap() > 0);
        assert!(store.delete(&id, &"0".repeat(64)).is_err());
        store
            .delete(&id, meta.result_sha256.as_ref().unwrap())
            .unwrap();
        assert!(store.status(&id).is_err());
    }
    #[test]
    fn corruption_and_offsets_are_rejected() {
        let (_dir, store, cfg) = setup();
        let id = store.begin(2, &digest(b"{}")).unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(store.chunk(&id, 1, b"{}").is_err());
        store.chunk(&id, 0, b"xx").unwrap();
        assert!(store.commit(&id, &cfg, 50).unwrap_err().contains("SHA-256"));
        store.abort_upload(&id).unwrap();
    }
    #[test]
    fn schedule_and_motor_validation_precede_ready() {
        let (dir, store, cfg) = setup();
        let id = fixture(&store, &cfg, 2, 0.0);
        let path = dir.path().join(&id).join("input.jsonl");
        let data = fs::read_to_string(&path)
            .unwrap()
            .replace("\"at_ms\":20", "\"at_ms\":21");
        let upload = store
            .begin(data.len() as u64, &digest(data.as_bytes()))
            .unwrap()["id"]
            .as_str()
            .unwrap()
            .to_string();
        store.chunk(&upload, 0, data.as_bytes()).unwrap();
        assert!(
            store
                .commit(&upload, &cfg, 50)
                .unwrap_err()
                .contains("at_ms")
        );
    }
    #[test]
    fn worker_prefetch_is_bounded_and_record_overflow_is_detectable() {
        let (_dir, store, cfg) = setup();
        let id = fixture(&store, &cfg, 1000, 0.0);
        let mut run = store.prepare(&id, &cfg, 50).unwrap();
        assert_eq!(run.index, 0);
        assert!(run.next(&cfg).unwrap().is_some());
        // Replace the writer with a deliberately stalled receiver.
        let (tx, _rx) = mpsc::sync_channel(1);
        run.records = Some(tx);
        run.record(json!({"sample":1}), true).unwrap();
        assert!(run.record(json!({"sample":2}), true).is_err());
        run.finish("failed", "test overflow", true);
    }
    #[test]
    fn execution_waits_for_time_and_detects_late_local_ticks() {
        let (_dir, store, cfg) = setup();
        let id = fixture(&store, &cfg, 3, 0.0);
        let mut run = store.prepare(&id, &cfg, 50).unwrap();
        assert!(run.next(&cfg).unwrap().unwrap().1);
        run.record(json!({}), true).unwrap();
        assert!(!run.next(&cfg).unwrap().unwrap().1);
        run.last_tick = Some(Instant::now() - Duration::from_millis(50));
        assert!(run.next(&cfg).unwrap_err().contains("40 ms"));
        run.finish("failed", "test late tick", true);
    }
    #[test]
    fn restart_marks_interrupted_and_never_resumes() {
        let (dir, store, cfg) = setup();
        let id = fixture(&store, &cfg, 1, 0.0);
        let mut m = store.load(&id).unwrap();
        m.state = "running".into();
        store.save(&m).unwrap();
        drop(store);
        let reopened = Store::open(dir.path().into()).unwrap();
        let (_, m) = reopened.download(&id).unwrap();
        assert_eq!(m.state, "interrupted");
        assert!(!m.disabled_confirmed);
        assert!(reopened.status(&id).unwrap()["result_sha256"].is_string());
    }
}
