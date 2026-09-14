//! Policy files have one owner: robotd. Workers prepare; only the bus loop commits.
use duck_control::policy::{Net, PolicyPaths, PreparedNetwork};
use duck_ipc_proto::{ModelParams, ModelControl};
use crate::control_owner::{ControlOwner, Owner};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const ROOT: &str = "/var/lib/robotd/policies";
const MAX_SIZE: usize = 64 * 1024 * 1024;
#[derive(Clone, Serialize, Deserialize)]
struct Version {
    id: String,
    name: String,
    path: PathBuf,
    #[serde(default)]
    control: Option<ModelControl>,
    #[serde(default)]
    policy_path: Option<PathBuf>,
    #[serde(default)]
    contract: Option<Value>,
}
#[derive(Clone, Default, Serialize, Deserialize)]
struct Record {
    active: Option<Version>,
    history: Vec<Version>,
}
fn version_json(version: &Version) -> Value {
    json!({
        "id": version.id,
        "name": version.name,
        "control": version.control,
        "two_file": version.policy_path.is_some(),
        "contract": version.contract,
    })
}
#[derive(Clone)]
struct Slot {
    key: String,
    path: PathBuf,
    net: Net,
}
struct Upload {
    token: String,
    slot: Slot,
    name: String,
    size: usize,
    received: usize,
    activation: String,
    normalizer_epsilon: f64,
    control: ModelControl,
    path: PathBuf,
}
struct BundlePart {
    name: String,
    size: usize,
    received: usize,
    path: PathBuf,
}
struct BundleUpload {
    token: String,
    slot: Slot,
    policy: BundlePart,
    model: BundlePart,
}
enum UploadTask {
    Legacy(Upload),
    Bundle(BundleUpload),
}
impl UploadTask {
    fn token(&self) -> &str {
        match self { Self::Legacy(value) => &value.token, Self::Bundle(value) => &value.token }
    }
    fn received(&self) -> usize {
        match self {
            Self::Legacy(value) => value.received,
            Self::Bundle(value) => value.policy.received + value.model.received,
        }
    }
    fn remove_files(self) {
        match self {
            Self::Legacy(value) => { let _ = fs::remove_file(value.path); }
            Self::Bundle(value) => {
                let _ = fs::remove_file(value.policy.path);
                let _ = fs::remove_file(value.model.path);
            }
        }
    }
}
enum Prepared {
    Legacy(PreparedNetwork),
    Bundle(crate::custom_policy::CustomPolicy),
}
struct Pending {
    slot: Slot,
    version: Version,
    prepared: Prepared,
}
struct Inner {
    records: BTreeMap<String, Record>,
    slots: BTreeMap<String, Slot>,
    upload: Option<UploadTask>,
    pending: Option<Pending>,
    phase: String,
    failed_phase: Option<String>,
    detail: String,
    editable: bool,
    observed: Instant,
    load_error: Option<String>,
}
pub struct Models {
    root: PathBuf,
    inner: Mutex<Inner>,
    owner: Arc<ControlOwner>,
}
impl Default for Models {
    fn default() -> Self {
        Self::at(PathBuf::from(ROOT), Arc::new(ControlOwner::default()))
    }
}
impl Models {
    pub fn with_owner(owner: Arc<ControlOwner>) -> Self {
        Self::at(PathBuf::from(ROOT), owner)
    }

    fn at(root: PathBuf, owner: Arc<ControlOwner>) -> Self {
        let (records, load_error) = match fs::read(root.join("manifest.json")) {
            Ok(data) => match serde_json::from_slice::<BTreeMap<String, Record>>(&data) {
                Ok(records) => {
                    let error = records.values().flat_map(|r| r.active.iter().chain(r.history.iter()))
                        .filter_map(|v| v.control).find_map(|c| c.validate().err());
                    (records, error.map(|e| format!("模型控制参数清单损坏：{e}")))
                },
                Err(e) => (BTreeMap::new(), Some(format!("模型历史清单损坏：{e}"))),
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (BTreeMap::new(), None),
            Err(e) => (BTreeMap::new(), Some(format!("无法读取模型清单：{e}"))),
        };
        Self {
            root,
            owner,
            inner: Mutex::new(Inner {
                records,
                slots: BTreeMap::new(),
                upload: None,
                pending: None,
                phase: "idle".into(),
                failed_phase: None,
                detail: "选择模型文件后导入；每个模型保留两条 ONNX 历史。".into(),
                editable: false,
                observed: Instant::now(),
                load_error,
            }),
        }
    }
    pub fn resolve(&self, paths: &mut PolicyPaths) {
        let mut inner = self.inner.lock().unwrap();
        inner.slots.clear();
        let mut resolve = |id: &str, net, path: &mut PathBuf| {
            let key = format!(
                "{id}--{}",
                path.file_name().unwrap_or_default().to_string_lossy()
            );
            inner.slots.insert(
                id.into(),
                Slot {
                    key: key.clone(),
                    path: path.clone(),
                    net,
                },
            );
            if let Some(v) = inner.records.get(&key).and_then(|r| r.active.as_ref())
                && v.policy_path.is_none()
            {
                *path = v.path.clone();
            }
        };
        resolve("walk", Net::Walk, &mut paths.walk);
        for (id, net, path) in [
            ("stand", Net::Stand, &mut paths.stand),
            ("sitstand", Net::SitStand, &mut paths.sitstand),
            ("ground_pick", Net::GroundPick, &mut paths.ground_pick),
            ("kick_left", Net::KickLeft, &mut paths.kick_left),
            ("kick_right", Net::KickRight, &mut paths.kick_right),
            ("roulade", Net::Roulade, &mut paths.roulade),
        ] {
            if let Some(path) = path {
                resolve(id, net, path);
            }
        }
    }
    pub fn restore_controls(&self, controller: &mut crate::control::Controller) {
        let configured: Vec<_> = {
            let inner = self.inner.lock().unwrap();
            inner.slots.values().map(|slot| {
                (slot.clone(), inner.records.get(&slot.key).and_then(|record| record.active.clone()))
            }).collect()
        };
        for (slot, version) in configured {
            match version {
                Some(version) if version.policy_path.is_some() => {
                    match crate::custom_policy::CustomPolicy::load(
                        version.policy_path.as_ref().unwrap(), &version.path
                    ) {
                        Ok(policy) => controller.replace_custom_policy(slot.net, policy),
                        Err(error) => {
                            self.inner.lock().unwrap().load_error = Some(format!(
                                "无法恢复两文件策略 {}：{error}", version.name
                            ));
                            return;
                        }
                    }
                }
                other => controller.set_model_control(slot.net, other.and_then(|value| value.control)),
            }
        }
    }
    pub fn load_error(&self) -> Option<String> {
        self.inner.lock().unwrap().load_error.clone()
    }
    pub fn experiment_identity(&self) -> Value {
        let inner = self.inner.lock().unwrap();
        Value::Object(inner.slots.iter().filter(|(id, _)| matches!(id.as_str(), "walk" | "stand"))
            .map(|(id, slot)| {
                let active = inner.records.get(&slot.key).and_then(|record| record.active.as_ref());
                (id.clone(), json!({
                    "configured_file": slot.path.file_name().unwrap_or_default().to_string_lossy(),
                    "active": active.map(version_json),
                }))
            }).collect())
    }
    fn status(&self, inner: &Inner) -> Value {
        let slots: Vec<_> = inner.slots.iter().map(|(id, slot)| {
            let record = inner.records.get(&slot.key).cloned().unwrap_or_default();
            json!({"id": id, "file": slot.path.file_name().unwrap_or_default().to_string_lossy(), "active": record.active.as_ref().map(version_json),
                "history": record.history.iter().map(version_json).collect::<Vec<_>>()})
        }).collect();
        json!({"slots":slots,"phase":inner.phase,"failed_phase":inner.failed_phase,"detail":inner.load_error.as_ref().unwrap_or(&inner.detail),
            "control_owner":self.owner.current().map(Owner::as_str),
            "editable":inner.editable && inner.observed.elapsed() < Duration::from_millis(500) && inner.load_error.is_none(),
            "token":inner.upload.as_ref().map(UploadTask::token),"received":inner.upload.as_ref().map(UploadTask::received)})
    }
    pub fn request(self: &Arc<Self>, params: ModelParams) -> Result<Value, String> {
        let mut inner = self.inner.lock().unwrap();
        if matches!(params, ModelParams::List {}) {
            return Ok(self.status(&inner));
        }
        if let Some(error) = &inner.load_error {
            return Err(error.clone());
        }
        if !matches!(params, ModelParams::Chunk { .. } | ModelParams::BundleChunk { .. })
            && (!inner.editable || inner.observed.elapsed() >= Duration::from_millis(500))
        {
            return Err("操作已拒绝：请先放松，等待全部配置电机的新鲜失能反馈".into());
        }
        let newly_acquired = self.owner.acquire(Owner::PolicyImport)?;
        let result = (|| {
        match params {
            ModelParams::Begin {
                slot,
                filename,
                size,
                activation,
                normalizer_epsilon,
                control,
            } => {
                if matches!(
                    inner.phase.as_str(),
                    "converting" | "validating" | "pending"
                ) {
                    return Err("已有模型任务正在处理".into());
                }
                let slot = inner.slots.get(&slot).cloned().ok_or("未知模型槽位")?;
                let extension = Path::new(&filename)
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_lowercase();
                if !matches!(extension.as_str(), "pt" | "pth" | "onnx")
                    || size == 0
                    || size > MAX_SIZE
                    || filename.len() > 240
                {
                    return Err("仅支持 .pt/.pth/.onnx，文件大小须为 1 字节至 64 MiB".into());
                }
                if !["elu", "relu", "tanh", "selu", "leaky_relu"].contains(&activation.as_str()) {
                    return Err("不支持的激活函数".into());
                }
                if !normalizer_epsilon.is_finite() || !(1e-12..=1.0).contains(&normalizer_epsilon) {
                    return Err("归一化 epsilon 必须介于 1e-12 和 1".into());
                }
                control.validate()?;
                fs::create_dir_all(&self.root).map_err(|e| e.to_string())?;
                if let Some(old) = inner.upload.take() {
                    old.remove_files();
                }
                let token = unique();
                let path = self.root.join(format!("upload-{token}.{extension}"));
                fs::File::create(&path).map_err(|e| e.to_string())?;
                inner.upload = Some(UploadTask::Legacy(Upload {
                    token,
                    slot,
                    name: filename,
                    size,
                    received: 0,
                    activation,
                    normalizer_epsilon,
                    control,
                    path,
                }));
                inner.failed_phase = None;
                inner.phase = "uploading".into();
                inner.detail = "正在上传本地模型".into();
            }
            ModelParams::Chunk { token, offset, hex } => {
                if inner.phase != "uploading" {
                    return Err("没有正在上传的任务".into());
                }
                let u = inner
                    .upload
                    .as_mut()
                    .and_then(|upload| match upload {
                        UploadTask::Legacy(upload) if upload.token == token => Some(upload),
                        _ => None,
                    })
                    .ok_or("上传凭证已失效")?;
                let bytes = decode_chunk(&hex)?;
                if offset != u.received || u.received + bytes.len() > u.size {
                    return Err("上传偏移或文件大小不匹配，请重新选择文件导入".into());
                }
                fs::OpenOptions::new()
                    .append(true)
                    .open(&u.path)
                    .and_then(|mut f| f.write_all(&bytes))
                    .map_err(|e| e.to_string())?;
                u.received += bytes.len();
            }
            ModelParams::Finish { token } => {
                let u = inner
                    .upload
                    .as_ref()
                    .and_then(|upload| match upload {
                        UploadTask::Legacy(upload) if upload.token == token => Some(upload),
                        _ => None,
                    })
                    .ok_or("上传凭证已失效")?;
                if u.received != u.size
                    || fs::metadata(&u.path).map_err(|e| e.to_string())?.len() != u.size as u64
                {
                    return Err("文件未上传完整".into());
                }
                let UploadTask::Legacy(upload) = inner.upload.take().unwrap() else { unreachable!() };
                inner.phase = "converting".into();
                inner.detail = "校验 PPO 权重并转换 ONNX（最长 120 秒）".into();
                let this = self.clone();
                std::thread::spawn(move || this.prepare(upload));
            }
            ModelParams::BeginBundle { slot, policy_filename, policy_size, model_filename, model_size } => {
                if matches!(inner.phase.as_str(), "converting" | "validating" | "pending") {
                    return Err("已有模型任务正在处理".into());
                }
                if Path::new(&policy_filename).extension().and_then(|value| value.to_str()) != Some("py")
                    || Path::new(&model_filename).extension().and_then(|value| value.to_str()) != Some("onnx")
                    || policy_size == 0 || policy_size > 1024 * 1024
                    || model_size == 0 || model_size > MAX_SIZE
                    || policy_filename.len() > 240 || model_filename.len() > 240
                {
                    return Err("策略文件须为 1 字节至 1 MiB 的 .py；模型须为 1 字节至 64 MiB 的 .onnx".into());
                }
                let slot = inner.slots.get(&slot).cloned().ok_or("未知模型槽位")?;
                fs::create_dir_all(&self.root).map_err(|error| error.to_string())?;
                if let Some(old) = inner.upload.take() { old.remove_files(); }
                let token = unique();
                let policy_path = self.root.join(format!("upload-{token}-policy.py"));
                let model_path = self.root.join(format!("upload-{token}-model.onnx"));
                fs::File::create(&policy_path).map_err(|error| error.to_string())?;
                fs::File::create(&model_path).map_err(|error| error.to_string())?;
                inner.upload = Some(UploadTask::Bundle(BundleUpload {
                    token,
                    slot,
                    policy: BundlePart { name: policy_filename, size: policy_size, received: 0, path: policy_path },
                    model: BundlePart { name: model_filename, size: model_size, received: 0, path: model_path },
                }));
                inner.failed_phase = None;
                inner.phase = "uploading".into();
                inner.detail = "正在上传 policy.py 与 model.onnx".into();
            }
            ModelParams::BundleChunk { token, file, offset, hex } => {
                if inner.phase != "uploading" { return Err("没有正在上传的任务".into()); }
                let upload = inner.upload.as_mut().and_then(|upload| match upload {
                    UploadTask::Bundle(upload) if upload.token == token => Some(upload),
                    _ => None,
                }).ok_or("上传凭证已失效")?;
                let part = match file.as_str() {
                    "policy" => &mut upload.policy,
                    "model" => &mut upload.model,
                    _ => return Err("未知两文件上传角色".into()),
                };
                let bytes = decode_chunk(&hex)?;
                if offset != part.received || part.received + bytes.len() > part.size {
                    return Err("上传偏移或文件大小不匹配，请重新选择两个文件导入".into());
                }
                fs::OpenOptions::new().append(true).open(&part.path)
                    .and_then(|mut output| output.write_all(&bytes)).map_err(|error| error.to_string())?;
                part.received += bytes.len();
            }
            ModelParams::FinishBundle { token } => {
                let upload = inner.upload.as_ref().and_then(|upload| match upload {
                    UploadTask::Bundle(upload) if upload.token == token => Some(upload),
                    _ => None,
                }).ok_or("上传凭证已失效")?;
                for part in [&upload.policy, &upload.model] {
                    if part.received != part.size
                        || fs::metadata(&part.path).map_err(|error| error.to_string())?.len() != part.size as u64
                    { return Err("两个文件未上传完整".into()); }
                }
                let UploadTask::Bundle(upload) = inner.upload.take().unwrap() else { unreachable!() };
                inner.phase = "validating".into();
                inner.detail = "正在校验接口、动态张量、完整前处理→ONNX→后处理链路并预热".into();
                let this = self.clone();
                std::thread::spawn(move || this.prepare_bundle(upload));
            }
            ModelParams::Cancel { token } => {
                if inner.phase != "uploading" { return Err("只能取消尚未完成的上传".into()); }
                if inner.upload.as_ref().map(UploadTask::token) != Some(token.as_str()) {
                    return Err("上传凭证已失效".into());
                }
                let upload = inner.upload.take().unwrap();
                upload.remove_files();
                inner.phase = "idle".into();
                inner.failed_phase = None;
                inner.detail = "策略导入已取消；机器人保持放松。".into();
                self.owner.release(Owner::PolicyImport);
            }
            ModelParams::Rollback { slot, version } => {
                if matches!(
                    inner.phase.as_str(),
                    "uploading" | "converting" | "validating" | "pending"
                ) {
                    return Err("请等待当前任务完成".into());
                }
                let slot = inner.slots.get(&slot).cloned().ok_or("未知模型槽位")?;
                let version = inner
                    .records
                    .get(&slot.key)
                    .and_then(|r| r.history.iter().find(|v| v.id == version))
                    .cloned()
                    .ok_or("历史版本不存在或已过期")?;
                inner.failed_phase = None;
                inner.phase = "validating".into();
                inner.detail = "正在重新校验历史 ONNX".into();
                let this = self.clone();
                std::thread::spawn(move || {
                    let result = if let Some(policy_path) = &version.policy_path {
                        crate::custom_policy::CustomPolicy::load(policy_path, &version.path)
                            .map(|policy| Pending { slot, version, prepared: Prepared::Bundle(policy) })
                    } else {
                        PreparedNetwork::load(&version.path)
                            .map(|network| Pending { slot, version, prepared: Prepared::Legacy(network) })
                            .map_err(|error| error.to_string())
                    };
                    this.prepared(result);
                });
            }
            ModelParams::List {} => unreachable!(),
        }
        Ok(self.status(&inner))
        })();
        if result.is_err() && newly_acquired {
            self.owner.release(Owner::PolicyImport);
        }
        result
    }
    fn prepare(&self, upload: Upload) {
        let target = self.root.join(format!("{}.onnx", upload.token));
        let result = (|| {
            if upload.path.extension().is_some_and(|s| s == "onnx") {
                fs::copy(&upload.path, &target).map_err(|e| e.to_string())?;
            } else {
                convert(
                    &upload.path,
                    &target,
                    &upload.activation,
                    upload.normalizer_epsilon,
                )?;
            }
            {
                let mut inner = self.inner.lock().unwrap();
                inner.phase = "validating".into();
                inner.detail =
                    "ONNX 已就绪；检查 obs[1,61] → actions[1,14]、目标运行时和有限数值试推理"
                        .into();
            }
            let network = PreparedNetwork::load(&target).map_err(|e| e.to_string())?;
            fs::File::open(&target)
                .and_then(|f| f.sync_all())
                .map_err(|e| e.to_string())?;
            Ok(Pending {
                slot: upload.slot,
                version: Version {
                    id: upload.token,
                    name: upload.name,
                    path: target.clone(),
                    control: Some(upload.control),
                    policy_path: None,
                    contract: None,
                },
                prepared: Prepared::Legacy(network),
            })
        })();
        let _ = fs::remove_file(upload.path);
        if result.is_err() {
            let _ = fs::remove_file(target);
        }
        self.prepared(result);
    }
    fn prepare_bundle(&self, upload: BundleUpload) {
        let model = self.root.join(format!("{}-model.onnx", upload.token));
        let policy = self.root.join(format!("{}-policy.py", upload.token));
        let result = (|| {
            fs::rename(&upload.model.path, &model).map_err(|error| error.to_string())?;
            fs::rename(&upload.policy.path, &policy).map_err(|error| error.to_string())?;
            for path in [&model, &policy] {
                fs::File::open(path).and_then(|file| file.sync_all())
                    .map_err(|error| error.to_string())?;
            }
            let runtime = crate::custom_policy::CustomPolicy::load(&policy, &model)?;
            let contract = runtime.contract().clone();
            Ok(Pending {
                slot: upload.slot,
                version: Version {
                    id: upload.token,
                    name: format!("{} + {}", upload.policy.name, upload.model.name),
                    path: model.clone(),
                    control: None,
                    policy_path: Some(policy.clone()),
                    contract: Some(contract),
                },
                prepared: Prepared::Bundle(runtime),
            })
        })();
        if result.is_err() {
            let _ = fs::remove_file(model);
            let _ = fs::remove_file(policy);
            let _ = fs::remove_file(upload.model.path);
            let _ = fs::remove_file(upload.policy.path);
        }
        self.prepared(result);
    }
    fn prepared(&self, result: Result<Pending, String>) {
        let mut inner = self.inner.lock().unwrap();
        match result {
            Ok(pending) => {
                inner.pending = Some(pending);
                inner.phase = "pending".into();
                inner.detail = "校验通过；等待控制循环确认放松后替换".into();
            }
            Err(error) => {
                inner.failed_phase = Some(inner.phase.clone());
                inner.phase = "error".into();
                inner.detail = format!("校验失败，旧模型未改变：{error}");
                self.owner.release(Owner::PolicyImport);
            }
        }
    }
    /// Runs before processing any enable/init intent. Fresh hardware proof is mandatory.
    pub fn tick(&self, relaxed: bool, controller: Option<&mut crate::control::Controller>) {
        let Ok(mut inner) = self.inner.try_lock() else {
            return;
        };
        inner.editable = relaxed;
        inner.observed = Instant::now();
        let Some(pending) = inner.pending.take() else {
            return;
        };
        if !relaxed
            || controller.is_none()
            || !inner
                .slots
                .values()
                .any(|s| s.key == pending.slot.key && s.net == pending.slot.net)
        {
            inner.failed_phase = Some("pending".into());
            inner.phase = "error".into();
            inner.detail = "替换已拒绝：需要当前模式不变、策略已加载、机器人已放松且全部配置电机有新鲜失能反馈。旧模型未改变；放松后重新导入或回滚。".into();
            self.remove_version_if_unused(&inner, &pending.version);
            self.owner.release(Owner::PolicyImport);
            return;
        }
        match self.commit(&inner.records, &pending.slot, &pending.version) {
            Ok(records) => {
                let old = std::mem::replace(&mut inner.records, records);
                let controller = controller.unwrap();
                match pending.prepared {
                    Prepared::Legacy(network) => {
                        controller.replace_network(pending.slot.net, network);
                        controller.set_model_control(pending.slot.net, pending.version.control);
                    }
                    Prepared::Bundle(policy) => controller.replace_custom_policy(pending.slot.net, policy),
                }
                inner.phase = "done".into();
                inner.detail = "替换成功并已即时加载；机器人保持放松，需手动初始化/开启策略。保留最近两条 ONNX 历史。".into();
                for record in old.values() {
                    for version in record.history.iter().chain(record.active.iter()) {
                        self.remove_version_if_unused(&inner, version);
                    }
                }
                self.owner.release(Owner::PolicyImport);
            }
            Err(e) => {
                inner.failed_phase = Some("pending".into());
                inner.phase = "error".into();
                inner.detail = format!("保存失败，旧模型未改变：{e}");
                self.remove_version_if_unused(&inner, &pending.version);
                self.owner.release(Owner::PolicyImport);
            }
        }
    }
    fn remove_if_unused(&self, inner: &Inner, path: &Path) {
        if path.parent() == Some(self.root.as_path())
            && !inner.records.values().any(|r| {
                r.active
                    .iter()
                    .chain(r.history.iter())
                    .any(|v| v.path == path || v.policy_path.as_deref() == Some(path))
            })
        {
            let _ = fs::remove_file(path);
        }
    }
    fn remove_version_if_unused(&self, inner: &Inner, version: &Version) {
        self.remove_if_unused(inner, &version.path);
        if let Some(path) = &version.policy_path { self.remove_if_unused(inner, path); }
    }
    fn commit(
        &self,
        records: &BTreeMap<String, Record>,
        slot: &Slot,
        version: &Version,
    ) -> Result<BTreeMap<String, Record>, String> {
        let mut records = records.clone();
        let record = records.entry(slot.key.clone()).or_default();
        let previous = match record.active.clone() {
            Some(v) => v,
            None => {
                let id = unique();
                let path = self.root.join(format!("{id}.onnx"));
                fs::copy(&slot.path, &path).map_err(|e| e.to_string())?;
                fs::File::open(&path)
                    .and_then(|f| f.sync_all())
                    .map_err(|e| e.to_string())?;
                Version {
                    id,
                    name: slot
                        .path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned(),
                    path,
                    control: None,
                    policy_path: None,
                    contract: None,
                }
            }
        };
        record.history.retain(|v| v.id != version.id);
        record.history.insert(0, previous);
        record.history.truncate(2);
        record.active = Some(version.clone());
        let data = serde_json::to_vec(&records).map_err(|e| e.to_string())?;
        let temporary = self.root.join("manifest.tmp");
        let mut file = fs::File::create(&temporary).map_err(|e| e.to_string())?;
        file.write_all(&data)
            .and_then(|_| file.sync_all())
            .map_err(|e| e.to_string())?;
        fs::rename(temporary, self.root.join("manifest.json")).map_err(|e| e.to_string())?;
        // Rename is the commit point. Never report failure after it and leave memory stale.
        if let Ok(dir) = fs::File::open(&self.root) {
            let _ = dir.sync_all();
        }
        Ok(records)
    }
}
fn unique() -> String {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    format!(
        "{}-{}-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    )
}
fn decode_chunk(hex: &str) -> Result<Vec<u8>, String> {
    if hex.is_empty()
        || hex.len() > 16384
        || hex.len() % 2 != 0
        || !hex.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err("无效上传分块（每块最多 8 KiB）".into());
    }
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).map_err(|e| e.to_string()))
        .collect()
}
fn convert(
    source: &Path,
    target: &Path,
    activation: &str,
    normalizer_epsilon: f64,
) -> Result<(), String> {
    let error_path = source.with_extension("log");
    let log = fs::File::create(&error_path).map_err(|e| e.to_string())?;
    let result = (|| {
        let mut child = std::process::Command::new(std::env::var_os("ROBOT_MODEL_PYTHON").unwrap_or_else(|| {
            let installed = Path::new("/var/lib/robotd/model-python/bin/python");
            if installed.is_file() { installed.as_os_str().to_owned() } else { "python3".into() }
        }))
            .args(["-I", "-c", include_str!("model_import.py")]).arg(source).arg(target).arg(activation).arg(normalizer_epsilon.to_string())
            .stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(log)
            .spawn().map_err(|e| format!("无法启动转换器：{e}；配置 ROBOT_MODEL_PYTHON，并安装 torch、onnx、onnxruntime、numpy"))?;
        let start = Instant::now();
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    return if status.success() {
                        Ok(())
                    } else {
                        Err(format!(
                            "PPO 转换失败：{}",
                            fs::read_to_string(&error_path)
                                .unwrap_or_default()
                                .chars()
                                .take(6000)
                                .collect::<String>()
                        ))
                    };
                }
                Ok(None) if start.elapsed() < Duration::from_secs(120) => {
                    std::thread::sleep(Duration::from_millis(100))
                }
                outcome => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!("转换器超时或退出异常：{outcome:?}"));
                }
            }
        }
    })();
    let _ = fs::remove_file(error_path);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    fn configured(root: &Path) -> (Arc<Models>, Slot) {
        let original = root.join("original.onnx");
        fs::write(&original, b"original").unwrap();
        let store = Arc::new(Models::at(
            root.join("store"),
            Arc::new(ControlOwner::default()),
        ));
        let slot = Slot {
            key: "walk--original.onnx".into(),
            path: original,
            net: Net::Walk,
        };
        store
            .inner
            .lock()
            .unwrap()
            .slots
            .insert("walk".into(), slot.clone());
        fs::create_dir_all(&store.root).unwrap();
        store.tick(true, None);
        (store, slot)
    }
    fn version(store: &Models, id: &str) -> Version {
        let path = store.root.join(format!("{id}.onnx"));
        fs::write(&path, id).unwrap();
        Version {
            id: id.into(),
            name: format!("{id}.pt"),
            path,
            control: None,
            policy_path: None,
            contract: None,
        }
    }
    #[test]
    fn history_keeps_two_and_rollback_survives_restart() {
        let dir = tempfile::tempdir().unwrap();
        let (store, slot) = configured(dir.path());
        let a = version(&store, "a");
        let b = version(&store, "b");
        let c = version(&store, "c");
        let records = store.commit(&BTreeMap::new(), &slot, &a).unwrap();
        let original = &records[&slot.key].history[0];
        assert_eq!(fs::read(&original.path).unwrap(), b"original");
        let records = store.commit(&records, &slot, &b).unwrap();
        let records = store.commit(&records, &slot, &c).unwrap();
        assert_eq!(
            records[&slot.key]
                .history
                .iter()
                .map(|v| v.id.as_str())
                .collect::<Vec<_>>(),
            vec!["b", "a"]
        );
        let records = store.commit(&records, &slot, &a).unwrap();
        assert_eq!(
            records[&slot.key]
                .history
                .iter()
                .map(|v| v.id.as_str())
                .collect::<Vec<_>>(),
            vec!["c", "b"]
        );
        let reloaded = Models::at(store.root.clone(), Arc::new(ControlOwner::default()));
        let mut paths = PolicyPaths {
            walk: slot.path,
            ..Default::default()
        };
        reloaded.resolve(&mut paths);
        assert_eq!(paths.walk, a.path);
    }
    #[test]
    fn failed_persistence_keeps_existing_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let (store, slot) = configured(dir.path());
        let a = version(&store, "a");
        let records = store.commit(&BTreeMap::new(), &slot, &a).unwrap();
        let before = fs::read(store.root.join("manifest.json")).unwrap();
        fs::create_dir(store.root.join("manifest.tmp")).unwrap();
        assert!(
            store
                .commit(&records, &slot, &version(&store, "b"))
                .is_err()
        );
        assert_eq!(fs::read(store.root.join("manifest.json")).unwrap(), before);
    }
    #[test]
    fn invalid_controls_are_rejected_before_upload_and_on_restart() {
        let dir = tempfile::tempdir().unwrap();
        let (store, _) = configured(dir.path());
        for control in [
            ModelControl { kp: 501.0, kd: 4.0, action_scale: 1.0 },
            ModelControl { kp: 60.0, kd: -0.1, action_scale: 1.0 },
            ModelControl { kp: 60.0, kd: 4.0, action_scale: 0.0 },
            ModelControl { kp: f64::NAN, kd: 4.0, action_scale: 1.0 },
        ] {
            assert!(store.request(ModelParams::Begin { slot: "walk".into(), filename: "a.onnx".into(), size: 1, activation: "elu".into(), normalizer_epsilon: 0.01, control }).is_err());
            assert!(store.inner.lock().unwrap().upload.is_none());
        }
        fs::write(store.root.join("manifest.json"), r#"{"walk--original.onnx":{"active":{"id":"a","name":"a","path":"a.onnx","control":{"kp":999,"kd":4,"action_scale":1}},"history":[]}}"#).unwrap();
        assert!(Models::at(store.root.clone(), Arc::new(ControlOwner::default())).load_error().is_some());
        let legacy: Version = serde_json::from_str(r#"{"id":"a","name":"a","path":"a.onnx"}"#).unwrap();
        assert_eq!(legacy.control, None);
    }
    #[test]
    fn chunks_require_token_exact_offset_and_declared_size() {
        let dir = tempfile::tempdir().unwrap();
        let (store, _) = configured(dir.path());
        let result = store
            .request(ModelParams::Begin {
                slot: "walk".into(),
                filename: "../../test.pt".into(),
                size: 2,
                activation: "elu".into(),
                normalizer_epsilon: 0.01,
                control: ModelControl { kp: 60.0, kd: 4.0, action_scale: 1.0 },
            })
            .unwrap();
        let token = result["token"].as_str().unwrap().to_owned();
        let inner = store.inner.lock().unwrap();
        let Some(UploadTask::Legacy(upload)) = inner.upload.as_ref() else { panic!() };
        assert!(upload.path.starts_with(&store.root));
        drop(inner);
        let chunk = |token: String, offset, hex: &str| ModelParams::Chunk {
            token,
            offset,
            hex: hex.into(),
        };
        assert!(store.request(chunk("wrong".into(), 0, "0000")).is_err());
        assert!(store.request(chunk(token.clone(), 1, "00")).is_err());
        assert!(store.request(chunk(token.clone(), 0, "000000")).is_err());
        assert!(
            store
                .request(ModelParams::Finish {
                    token: token.clone()
                })
                .is_err()
        );
        assert!(store.request(chunk(token.clone(), 0, "00ff")).is_ok());
        assert!(store.request(chunk(token, 0, "00ff")).is_err());
        assert!(decode_chunk("éé").is_err());
        assert!(decode_chunk(&"00".repeat(8193)).is_err());
    }
    #[test]
    fn bundle_upload_keeps_two_roles_in_one_transaction() {
        let dir = tempfile::tempdir().unwrap();
        let (store, _) = configured(dir.path());
        let status = store.request(ModelParams::BeginBundle {
            slot: "walk".into(),
            policy_filename: "../../policy.py".into(),
            policy_size: 2,
            model_filename: "../../model.onnx".into(),
            model_size: 3,
        }).unwrap();
        let token = status["token"].as_str().unwrap().to_owned();
        let chunk = |file: &str, offset, hex: &str| ModelParams::BundleChunk {
            token: token.clone(), file: file.into(), offset, hex: hex.into(),
        };
        assert!(store.request(chunk("policy", 0, "7079")).is_ok());
        assert!(store.request(chunk("model", 0, "000102")).is_ok());
        assert!(store.request(chunk("model", 0, "00")).is_err());
        let inner = store.inner.lock().unwrap();
        let Some(UploadTask::Bundle(upload)) = inner.upload.as_ref() else { panic!() };
        assert!(upload.policy.path.starts_with(&store.root));
        assert!(upload.model.path.starts_with(&store.root));
        assert_ne!(upload.policy.path, upload.model.path);
    }
    #[test]
    fn status_requires_fresh_motor_proof_and_corrupt_manifest_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        let (store, _) = configured(dir.path());
        store.tick(false, None);
        assert_eq!(
            store.request(ModelParams::List {}).unwrap()["editable"],
            false
        );
        {
            let mut inner = store.inner.lock().unwrap();
            inner.editable = true;
            inner.observed = Instant::now() - Duration::from_secs(1);
        }
        assert_eq!(
            store.request(ModelParams::List {}).unwrap()["editable"],
            false
        );
        assert!(
            store
                .request(ModelParams::Begin {
                    slot: "walk".into(),
                    filename: "blocked.pt".into(),
                    size: 1,
                    activation: "elu".into(),
                    normalizer_epsilon: 0.01,
                control: ModelControl { kp: 60.0, kd: 4.0, action_scale: 1.0 },
                })
                .is_err(),
            "a forged RPC must not bypass stale feedback"
        );
        assert!(store.inner.lock().unwrap().upload.is_none());
        fs::write(store.root.join("manifest.json"), "broken").unwrap();
        let broken = Arc::new(Models::at(store.root.clone(), Arc::new(ControlOwner::default())));
        assert!(broken.load_error().is_some());
        assert!(
            broken
                .request(ModelParams::Begin {
                    slot: "walk".into(),
                    filename: "x.pt".into(),
                    size: 1,
                    activation: "elu".into(),
                    normalizer_epsilon: 0.01,
                    control: ModelControl { kp: 60.0, kd: 4.0, action_scale: 1.0 },
                })
                .is_err()
        );
    }
    #[test]
    #[ignore = "requires ORT_DYLIB_PATH; run explicitly on a host with ONNX Runtime"]
    fn a_prepared_import_cannot_commit_without_relaxation_or_controller() {
        let dir = tempfile::tempdir().unwrap();
        let (store, slot) = configured(dir.path());
        let bundled = Path::new(env!("CARGO_MANIFEST_DIR")).join("../policies/alpha_walking.onnx");
        for relaxed in [false, true] {
            let v = version(&store, if relaxed { "no-controller" } else { "enabled" });
            fs::copy(&bundled, &v.path).unwrap();
            store.prepared(Ok(Pending {
                slot: slot.clone(),
                prepared: Prepared::Legacy(PreparedNetwork::load(&v.path).unwrap()),
                version: v,
            }));
            store.tick(relaxed, None);
            let status = store.request(ModelParams::List {}).unwrap();
            assert_eq!(status["phase"], "error");
            assert!(!store.root.join("manifest.json").exists());
            assert_eq!(fs::read(&slot.path).unwrap(), b"original");
        }
    }
    #[test]
    #[ignore = "requires ORT_DYLIB_PATH; run explicitly on a host with ONNX Runtime"]
    fn import_commit_and_rollback_use_real_onnx_without_moving_hardware() {
        use duck_control::policy::{DEFAULT_STANDING_THRESHOLD, Policy};
        let dir = tempfile::tempdir().unwrap();
        let (store, slot) = configured(dir.path());
        let bundled = Path::new(env!("CARGO_MANIFEST_DIR")).join("../policies/alpha_walking.onnx");
        fs::copy(&bundled, &slot.path).unwrap();
        let policy = Policy::load(
            &PolicyPaths {
                walk: slot.path.clone(),
                ..Default::default()
            },
            DEFAULT_STANDING_THRESHOLD,
        )
        .unwrap();
        let mut controller =
            crate::control::Controller::new(policy, Default::default(), Default::default());
        let mut candidate = version(&store, "candidate");
        candidate.control = Some(ModelControl { kp: 60.25, kd: 1.234, action_scale: 1.0 });
        fs::copy(&bundled, &candidate.path).unwrap();
        store.prepared(Ok(Pending {
            slot: slot.clone(),
            prepared: Prepared::Legacy(PreparedNetwork::load(&candidate.path).unwrap()),
            version: candidate.clone(),
        }));
        store.tick(false, Some(&mut controller));
        assert_eq!(
            store.request(ModelParams::List {}).unwrap()["phase"],
            "error"
        );
        assert!(!store.root.join("manifest.json").exists());
        // The same controller can accept a newly prepared candidate only after relaxation.
        fs::copy(&bundled, &candidate.path).unwrap();
        store.prepared(Ok(Pending {
            slot: slot.clone(),
            prepared: Prepared::Legacy(PreparedNetwork::load(&candidate.path).unwrap()),
            version: candidate.clone(),
        }));
        store.tick(true, Some(&mut controller));
        assert_eq!(
            store.request(ModelParams::List {}).unwrap()["phase"],
            "done"
        );
        assert_eq!(controller.model_control(Net::Walk), candidate.control);
        let reloaded = Models::at(store.root.clone(), Arc::new(ControlOwner::default()));
        let mut paths = PolicyPaths { walk: slot.path.clone(), ..Default::default() };
        reloaded.resolve(&mut paths);
        controller.set_model_control(Net::Walk, None);
        reloaded.restore_controls(&mut controller);
        assert_eq!(controller.model_control(Net::Walk), candidate.control);
        let previous = store.inner.lock().unwrap().records[&slot.key].history[0].clone();
        store
            .request(ModelParams::Rollback {
                slot: "walk".into(),
                version: previous.id.clone(),
            })
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while store.inner.lock().unwrap().pending.is_none() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        store.tick(true, Some(&mut controller));
        let inner = store.inner.lock().unwrap();
        assert_eq!(inner.phase, "done");
        assert_eq!(
            inner.records[&slot.key].active.as_ref().unwrap().id,
            previous.id
        );
        assert_eq!(inner.records[&slot.key].history[0].id, candidate.id);
        assert_eq!(inner.records[&slot.key].history[0].control, candidate.control);
        assert_eq!(controller.model_control(Net::Walk), None, "rollback to legacy restores defaults");
    }
}
