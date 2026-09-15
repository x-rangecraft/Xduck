//! Policy files have one owner: robotd. Workers prepare; only the bus loop commits.
use duck_control::policy::{Net, PolicyPaths};
use duck_ipc_proto::ModelParams;
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
const DEFAULT_POLICY_LABEL: &str = "默认";
const DEFAULT_LOCOMOTION_POLICY: &[u8] = include_bytes!("default_locomotion_policy.py");
const DEFAULT_GROUND_PICK_POLICY: &[u8] = include_bytes!("default_ground_pick_policy.py");
const DEFAULT_KICK_LEFT_POLICY: &[u8] = include_bytes!("default_kick_left_policy.py");
const DEFAULT_KICK_RIGHT_POLICY: &[u8] = include_bytes!("default_kick_right_policy.py");
const DEFAULT_ROULADE_POLICY: &[u8] = include_bytes!("default_roulade_policy.py");
const DEFAULT_SITSTAND_POLICY: &[u8] = include_bytes!("default_sitstand_policy.py");

fn default_policy(net: Net) -> &'static [u8] {
    match net {
        Net::Walk | Net::Stand => DEFAULT_LOCOMOTION_POLICY,
        Net::SitStand => DEFAULT_SITSTAND_POLICY,
        Net::GroundPick => DEFAULT_GROUND_PICK_POLICY,
        Net::KickLeft => DEFAULT_KICK_LEFT_POLICY,
        Net::KickRight => DEFAULT_KICK_RIGHT_POLICY,
        Net::Roulade => DEFAULT_ROULADE_POLICY,
    }
}
#[derive(Clone, Serialize, Deserialize)]
struct Version {
    id: String,
    name: String,
    path: PathBuf,
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
        "two_file": true,
        "contract": version.contract,
    })
}
#[derive(Clone)]
struct Slot {
    key: String,
    path: PathBuf,
    net: Net,
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
    activation: String,
    normalizer_epsilon: f64,
}
impl BundleUpload {
    fn received(&self) -> usize { self.policy.received + self.model.received }

    fn remove_files(self) {
        let _ = fs::remove_file(self.policy.path);
        let _ = fs::remove_file(self.model.path);
    }
}
struct Pending {
    slot: Slot,
    version: Version,
    /// Only the peer's consumer changes. Its ONNX remains immutable and referenced in history.
    companion: Option<(Slot, Version, String)>,
    policy: crate::custom_policy::CustomPolicy,
}
struct Inner {
    records: BTreeMap<String, Record>,
    slots: BTreeMap<String, Slot>,
    upload: Option<BundleUpload>,
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

fn persist_records(root: &Path, records: &BTreeMap<String, Record>) -> Result<(), String> {
    let data = serde_json::to_vec(records).map_err(|error| error.to_string())?;
    let temporary = root.join("manifest.tmp");
    let mut file = fs::File::create(&temporary).map_err(|error| error.to_string())?;
    file.write_all(&data)
        .and_then(|_| file.sync_all())
        .map_err(|error| error.to_string())?;
    fs::rename(temporary, root.join("manifest.json")).map_err(|error| error.to_string())?;
    if let Ok(directory) = fs::File::open(root) {
        let _ = directory.sync_all();
    }
    Ok(())
}

fn write_default_policy(path: &Path, net: Net) -> Result<(), String> {
    if path.is_file() {
        return Ok(());
    }
    let mut file = fs::File::create(path).map_err(|error| error.to_string())?;
    file.write_all(default_policy(net))
        .and_then(|_| file.sync_all())
        .map_err(|error| error.to_string())
}

fn replace_default_policy(path: &Path, net: Net) -> Result<(), String> {
    let temporary = path.with_extension("policy.tmp");
    let mut file = fs::File::create(&temporary).map_err(|error| error.to_string())?;
    file.write_all(default_policy(net))
        .and_then(|_| file.sync_all())
        .map_err(|error| error.to_string())?;
    fs::rename(&temporary, path).map_err(|error| error.to_string())?;
    if let Some(parent) = path.parent()
        && let Ok(directory) = fs::File::open(parent)
    {
        let _ = directory.sync_all();
    }
    Ok(())
}

fn default_version(root: &Path, source: &Path, net: Net) -> Result<Version, String> {
    let id = unique();
    let model = root.join(format!("{id}-model.onnx"));
    let policy = root.join(format!("{id}-policy.py"));
    fs::copy(source, &model).map_err(|error| error.to_string())?;
    fs::File::open(&model)
        .and_then(|file| file.sync_all())
        .map_err(|error| error.to_string())?;
    if let Err(error) = write_default_policy(&policy, net) {
        let _ = fs::remove_file(&model);
        return Err(error);
    }
    Ok(Version {
        id,
        name: DEFAULT_POLICY_LABEL.into(),
        path: model,
        policy_path: Some(policy),
        contract: None,
    })
}

fn migrate_legacy_version(root: &Path, version: &mut Version, net: Net) -> Result<Option<PathBuf>, String> {
    if version.policy_path.is_some() {
        return Ok(None);
    }
    let model = root.join(format!("{}-model.onnx", version.id));
    let policy = root.join(format!("{}-policy.py", version.id));
    if version.path != model && !model.is_file() {
        fs::copy(&version.path, &model).map_err(|error| {
            format!("无法迁移旧 ONNX {}：{error}", version.path.display())
        })?;
        fs::File::open(&model)
            .and_then(|file| file.sync_all())
            .map_err(|error| error.to_string())?;
    }
    if let Err(error) = write_default_policy(&policy, net) {
        if version.path != model {
            let _ = fs::remove_file(&model);
        }
        return Err(error);
    }
    let obsolete = (version.path != model).then(|| version.path.clone());
    version.name = DEFAULT_POLICY_LABEL.into();
    version.path = model;
    version.policy_path = Some(policy);
    version.contract = None;
    Ok(obsolete)
}

fn normalize_default_name(version: &mut Version, net: Net) -> Result<bool, String> {
    let Some(policy) = version.policy_path.as_ref() else {
        return Ok(false);
    };
    let bytes = fs::read(policy).map_err(|error| error.to_string())?;
    if version.name == DEFAULT_POLICY_LABEL && bytes != default_policy(net) {
        // "默认" can only be assigned by robotd when no custom consumer was uploaded.
        // Upgrade that platform-owned artifact in place so a pre-v2 install can boot the
        // v2-only runtime. User-provided v1 consumers keep their own name and are rejected.
        replace_default_policy(policy, net)?;
        return Ok(true);
    }
    if version.name != DEFAULT_POLICY_LABEL && bytes == default_policy(net) {
        version.name = DEFAULT_POLICY_LABEL.into();
        return Ok(true);
    }
    Ok(false)
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
                Ok(records) => (records, None),
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
        let result = (|| -> Result<Vec<PathBuf>, String> {
            fs::create_dir_all(&self.root).map_err(|error| error.to_string())?;
            let mut changed = false;
            let mut obsolete = Vec::new();
            let mut resolve = |id: &str, net, path: &mut PathBuf| -> Result<(), String> {
            let configured_path = path.clone();
            let key = format!(
                "{id}--{}",
                configured_path.file_name().unwrap_or_default().to_string_lossy()
            );
            inner.slots.insert(
                id.into(),
                Slot {
                    key: key.clone(),
                    path: configured_path.clone(),
                    net,
                },
            );
            let record = inner.records.entry(key).or_default();
            if record.active.is_none() {
                record.active = Some(default_version(&self.root, &configured_path, net)?);
                changed = true;
            }
            for version in record.active.iter_mut().chain(record.history.iter_mut()) {
                if version.policy_path.is_none() {
                    if let Some(path) = migrate_legacy_version(&self.root, version, net)? {
                        obsolete.push(path);
                    }
                    changed = true;
                }
                changed |= normalize_default_name(version, net)?;
            }
            *path = record.active.as_ref().ok_or("策略槽位缺少活动版本")?.path.clone();
            Ok(())
        };
        resolve("walk", Net::Walk, &mut paths.walk)?;
        for (id, net, path) in [
            ("stand", Net::Stand, &mut paths.stand),
            ("sitstand", Net::SitStand, &mut paths.sitstand),
            ("ground_pick", Net::GroundPick, &mut paths.ground_pick),
            ("kick_left", Net::KickLeft, &mut paths.kick_left),
            ("kick_right", Net::KickRight, &mut paths.kick_right),
            ("roulade", Net::Roulade, &mut paths.roulade),
        ] {
            if let Some(path) = path {
                resolve(id, net, path)?;
            }
        }
            // Existing equal consumers can share a physical file without changing semantics.
            // Do not arbitrarily choose a winner when an older installation has different code.
            if let (Some(walk), Some(stand)) = (inner.slots.get("walk"), inner.slots.get("stand")) {
                let walk_key = walk.key.clone();
                let stand_key = stand.key.clone();
                let walk_policy = inner.records[&walk_key].active.as_ref().unwrap().policy_path.clone().unwrap();
                let stand_policy = inner.records[&stand_key].active.as_ref().unwrap().policy_path.clone().unwrap();
                if walk_policy != stand_policy
                    && fs::read(&walk_policy).map_err(|error| error.to_string())?
                        == fs::read(&stand_policy).map_err(|error| error.to_string())?
                {
                    inner.records.get_mut(&stand_key).unwrap().active.as_mut().unwrap().policy_path = Some(walk_policy);
                    obsolete.push(stand_policy);
                    changed = true;
                }
            }
            if changed {
                persist_records(&self.root, &inner.records)?;
            }
            Ok(obsolete)
        })();
        match result {
            Ok(obsolete) => {
                for path in obsolete {
                    self.remove_if_unused(&inner, &path);
                }
            }
            Err(error) => inner.load_error = Some(format!("无法统一为双文件策略：{error}")),
        }
    }
    pub fn restore_controls(&self, controller: &mut crate::control::Controller) -> Result<(), String> {
        let configured: Vec<_> = {
            let inner = self.inner.lock().unwrap();
            inner.slots.values().map(|slot| {
                (slot.clone(), inner.records.get(&slot.key).and_then(|record| record.active.clone()))
            }).collect()
        };
        let walk = configured.iter().find(|(slot, _)| slot.net == Net::Walk);
        let stand = configured.iter().find(|(slot, _)| slot.net == Net::Stand);
        if let (Some((_, walk)), Some((_, stand))) = (walk, stand) {
            let walk = walk.as_ref().ok_or("走策略缺少活动版本")?;
            let stand = stand.as_ref().ok_or("站策略缺少活动版本")?;
            let walk_policy = walk.policy_path.as_ref().ok_or("走策略缺少消费文件")?;
            let stand_policy = stand.policy_path.as_ref().ok_or("站策略缺少消费文件")?;
            if fs::read(walk_policy).map_err(|error| error.to_string())?
                != fs::read(stand_policy).map_err(|error| error.to_string())?
            {
                return Err("走/站消费文件不同，不能共享状态；需先统一两者的消费文件".into());
            }
            let policy = crate::custom_policy::CustomPolicy::load_pair(walk_policy, &walk.path, &stand.path)
                .map_err(|error| format!("无法恢复走/站共享策略：{error}"))?;
            controller.replace_locomotion_policy(policy);
        }
        let paired = stand.is_some();
        for (slot, version) in configured {
            if paired && matches!(slot.net, Net::Walk | Net::Stand) { continue; }
            let version = version.ok_or_else(|| format!("策略槽位 {} 缺少活动版本", slot.key))?;
            let policy_path = version.policy_path.as_ref().ok_or_else(|| {
                format!("策略版本 {} 尚未迁移为双文件", version.name)
            })?;
            let policy = crate::custom_policy::CustomPolicy::load(policy_path, &version.path, slot.net)
                .map_err(|error| format!("无法恢复双文件策略 {}：{error}", version.name))?;
            controller.replace_custom_policy(slot.net, policy);
        }
        Ok(())
    }
    pub fn load_error(&self) -> Option<String> {
        self.inner.lock().unwrap().load_error.clone()
    }
    pub fn set_load_error(&self, error: String) {
        self.inner.lock().unwrap().load_error = Some(error);
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
                "shared_consumer": if inner.slots.contains_key("stand") && matches!(slot.net, Net::Walk | Net::Stand) { Some("walk_stand") } else { None },
                "history": record.history.iter().map(version_json).collect::<Vec<_>>()})
        }).collect();
        json!({"slots":slots,"phase":inner.phase,"failed_phase":inner.failed_phase,"detail":inner.load_error.as_ref().unwrap_or(&inner.detail),
            "control_owner":self.owner.current().map(Owner::as_str),
            "editable":inner.editable && inner.observed.elapsed() < Duration::from_millis(500) && inner.load_error.is_none(),
            "token":inner.upload.as_ref().map(|upload| upload.token.as_str()),"received":inner.upload.as_ref().map(BundleUpload::received)})
    }
    pub fn request(self: &Arc<Self>, params: ModelParams) -> Result<Value, String> {
        let mut inner = self.inner.lock().unwrap();
        if matches!(params, ModelParams::List {}) {
            return Ok(self.status(&inner));
        }
        if let Some(error) = &inner.load_error {
            return Err(error.clone());
        }
        if !matches!(params, ModelParams::BundleChunk { .. })
            && (!inner.editable || inner.observed.elapsed() >= Duration::from_millis(500))
        {
            return Err("操作已拒绝：请先放松，等待全部配置电机的新鲜失能反馈".into());
        }
        let newly_acquired = self.owner.acquire(Owner::PolicyImport)?;
        let result = (|| {
        match params {
            ModelParams::BeginBundle {
                slot,
                policy_filename,
                policy_size,
                model_filename,
                model_size,
                activation,
                normalizer_epsilon,
            } => {
                if matches!(inner.phase.as_str(), "converting" | "validating" | "pending") {
                    return Err("已有模型任务正在处理".into());
                }
                let model_extension = Path::new(&model_filename)
                    .extension()
                    .and_then(|value| value.to_str())
                    .unwrap_or("")
                    .to_lowercase();
                if !matches!(model_extension.as_str(), "pt" | "pth" | "onnx")
                    || model_size == 0
                    || model_size > MAX_SIZE
                    || model_filename.len() > 240
                {
                    return Err("模型须为 1 字节至 64 MiB 的 .pt/.pth/.onnx".into());
                }
                let activation = activation.unwrap_or_else(|| "elu".into());
                if !["elu", "relu", "tanh", "selu", "leaky_relu"].contains(&activation.as_str()) {
                    return Err("不支持的激活函数".into());
                }
                let normalizer_epsilon = normalizer_epsilon.unwrap_or(0.01);
                if !normalizer_epsilon.is_finite() || !(1e-12..=1.0).contains(&normalizer_epsilon) {
                    return Err("归一化 epsilon 必须介于 1e-12 和 1".into());
                }
                let custom_policy = match (&policy_filename, policy_size) {
                    (None, None) => false,
                    (Some(filename), Some(size))
                        if Path::new(filename).extension().and_then(|value| value.to_str()) == Some("py")
                            && size > 0
                            && size <= 1024 * 1024
                            && filename.len() <= 240 => true,
                    (Some(_), Some(_)) => {
                        return Err("策略文件须为 1 字节至 1 MiB 的 .py".into());
                    }
                    _ => return Err("策略文件名和大小必须同时提供；两者都省略时使用内置默认策略".into()),
                };
                let slot = inner.slots.get(&slot).cloned().ok_or("未知模型槽位")?;
                fs::create_dir_all(&self.root).map_err(|error| error.to_string())?;
                if let Some(old) = inner.upload.take() { old.remove_files(); }
                let token = unique();
                let policy_path = self.root.join(format!("upload-{token}-policy.py"));
                let model_path = self.root.join(format!("upload-{token}-model.{model_extension}"));
                let (policy_name, policy_size, policy_received) = if custom_policy {
                    fs::File::create(&policy_path).map_err(|error| error.to_string())?;
                    (policy_filename.unwrap(), policy_size.unwrap(), 0)
                } else {
                    let bytes = default_policy(slot.net);
                    let mut file = fs::File::create(&policy_path).map_err(|error| error.to_string())?;
                    file.write_all(bytes)
                        .and_then(|_| file.sync_all())
                        .map_err(|error| error.to_string())?;
                    (DEFAULT_POLICY_LABEL.into(), bytes.len(), bytes.len())
                };
                fs::File::create(&model_path).map_err(|error| error.to_string())?;
                inner.upload = Some(BundleUpload {
                    token,
                    slot,
                    policy: BundlePart { name: policy_name, size: policy_size, received: policy_received, path: policy_path },
                    model: BundlePart { name: model_filename, size: model_size, received: 0, path: model_path },
                    activation,
                    normalizer_epsilon,
                });
                inner.failed_phase = None;
                inner.phase = "uploading".into();
                inner.detail = if custom_policy {
                    "正在上传策略代码与模型文件".into()
                } else {
                    "已绑定内置默认策略；正在上传模型文件".into()
                };
            }
            ModelParams::BundleChunk { token, file, offset, hex } => {
                if inner.phase != "uploading" { return Err("没有正在上传的任务".into()); }
                let upload = inner.upload.as_mut().filter(|upload| upload.token == token)
                    .ok_or("上传凭证已失效")?;
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
                let upload = inner.upload.as_ref().filter(|upload| upload.token == token)
                    .ok_or("上传凭证已失效")?;
                for part in [&upload.policy, &upload.model] {
                    if part.received != part.size
                        || fs::metadata(&part.path).map_err(|error| error.to_string())?.len() != part.size as u64
                    { return Err("两个文件未上传完整".into()); }
                }
                let upload = inner.upload.take().unwrap();
                if upload.model.path.extension().is_some_and(|value| value == "onnx") {
                    inner.phase = "validating".into();
                    inner.detail = "正在校验接口、动态张量、完整前处理→ONNX→后处理链路并预热".into();
                } else {
                    inner.phase = "converting".into();
                    inner.detail = "正在 RK3566 校验 PPO 权重并转换 ONNX（最长 120 秒）".into();
                }
                let this = self.clone();
                std::thread::spawn(move || this.prepare_bundle(upload));
            }
            ModelParams::Cancel { token } => {
                if inner.phase != "uploading" { return Err("只能取消尚未完成的上传".into()); }
                if inner.upload.as_ref().map(|upload| upload.token.as_str()) != Some(token.as_str()) {
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
                    let result = this.prepare_version(slot, version);
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
    fn prepare_bundle(&self, upload: BundleUpload) {
        let model = self.root.join(format!("{}-model.onnx", upload.token));
        let policy = self.root.join(format!("{}-policy.py", upload.token));
        let result = (|| {
            fs::rename(&upload.policy.path, &policy).map_err(|error| error.to_string())?;
            if upload.model.path.extension().is_some_and(|value| value == "onnx") {
                fs::rename(&upload.model.path, &model).map_err(|error| error.to_string())?;
            } else {
                {
                    let mut inner = self.inner.lock().unwrap();
                    inner.phase = "converting".into();
                    inner.detail = "正在 RK3566 校验 PPO 权重并转换 ONNX（最长 120 秒）".into();
                }
                convert(
                    &upload.model.path,
                    &model,
                    &upload.activation,
                    upload.normalizer_epsilon,
                )?;
                fs::remove_file(&upload.model.path).map_err(|error| error.to_string())?;
            }
            for path in [&model, &policy] {
                fs::File::open(path).and_then(|file| file.sync_all())
                    .map_err(|error| error.to_string())?;
            }
            {
                let mut inner = self.inner.lock().unwrap();
                inner.phase = "validating".into();
                inner.detail = "正在校验接口、动态张量、完整前处理→ONNX→后处理链路并预热".into();
            }
            self.prepare_version(upload.slot, Version {
                    id: upload.token,
                    name: if upload.policy.name == DEFAULT_POLICY_LABEL {
                        DEFAULT_POLICY_LABEL.into()
                    } else {
                        format!("{} + {}", upload.policy.name, upload.model.name)
                    },
                    path: model.clone(),
                    policy_path: Some(policy.clone()),
                    contract: None,
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

    fn prepare_version(&self, slot: Slot, mut version: Version) -> Result<Pending, String> {
        let peer_id = match slot.net { Net::Walk => Some("stand"), Net::Stand => Some("walk"), _ => None };
        let peer = {
            let inner = self.inner.lock().unwrap();
            peer_id.and_then(|id| inner.slots.get(id)).map(|peer| {
                let active = inner.records.get(&peer.key).and_then(|record| record.active.clone())
                    .ok_or("走/站配对槽位缺少活动模型")?;
                Ok::<_, String>((peer.clone(), active))
            }).transpose()?
        };
        let path = version.policy_path.as_ref().ok_or("策略缺少消费文件")?;
        let (runtime, companion) = if let Some((peer, current)) = peer {
            let (walk, stand) = if slot.net == Net::Walk {
                (&version.path, &current.path)
            } else { (&current.path, &version.path) };
            // One instance, alternating both models during warmup: tests the exact shared
            // preprocessing -> ONNX -> postprocessing path before either record can change.
            let runtime = crate::custom_policy::CustomPolicy::load_pair(path, walk, stand)
                .map_err(|error| format!("走/站联合校验失败，两侧均未改变：{error}"))?;
            let companion = Version {
                id: unique(),
                name: if version.name == DEFAULT_POLICY_LABEL { DEFAULT_POLICY_LABEL.into() }
                    else { "走/站共享消费文件".into() },
                path: current.path.clone(),
                policy_path: Some(path.clone()),
                contract: Some(runtime.contract().clone()),
            };
            (runtime, Some((peer, companion, current.id)))
        } else {
            (crate::custom_policy::CustomPolicy::load(path, &version.path, slot.net)?, None)
        };
        version.contract = Some(runtime.contract().clone());
        Ok(Pending { slot, version, companion, policy: runtime })
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
            || (matches!(pending.slot.net, Net::Walk | Net::Stand)
                && inner.slots.contains_key("stand") != pending.companion.is_some())
            || !inner
                .slots
                .values()
                .any(|s| s.key == pending.slot.key && s.net == pending.slot.net)
            || pending.companion.as_ref().is_some_and(|(peer, _, expected)| {
                !inner.slots.values().any(|slot| slot.key == peer.key && slot.net == peer.net)
                    || inner.records.get(&peer.key).and_then(|record| record.active.as_ref())
                        .is_none_or(|active| &active.id != expected)
            })
        {
            inner.failed_phase = Some("pending".into());
            inner.phase = "error".into();
            inner.detail = "替换已拒绝：需要当前模式不变、策略已加载、机器人已放松且全部配置电机有新鲜失能反馈。旧模型未改变；放松后重新导入或回滚。".into();
            self.remove_version_if_unused(&inner, &pending.version);
            self.owner.release(Owner::PolicyImport);
            return;
        }
        let mut updates = vec![(&pending.slot, &pending.version)];
        if let Some((slot, version, _)) = &pending.companion { updates.push((slot, version)); }
        match self.commit_updates(&inner.records, &updates) {
            Ok(records) => {
                let old = std::mem::replace(&mut inner.records, records);
                let controller = controller.unwrap();
                if pending.companion.is_some() {
                    controller.replace_locomotion_policy(pending.policy);
                } else {
                    controller.replace_custom_policy(pending.slot.net, pending.policy);
                }
                inner.phase = "done".into();
                inner.detail = if pending.companion.is_some() {
                    "走/站已原子更新并加载共享消费实例；另一侧 ONNX 未改变，消费文件已同步。机器人保持放松，需手动初始化/开启策略。两侧均保留最近两条历史。".into()
                } else {
                    "替换成功并已即时加载；机器人保持放松，需手动初始化/开启策略。保留最近两条 ONNX 历史。".into()
                };
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
    #[cfg(test)]
    fn commit(
        &self,
        records: &BTreeMap<String, Record>,
        slot: &Slot,
        version: &Version,
    ) -> Result<BTreeMap<String, Record>, String> {
        self.commit_updates(records, &[(slot, version)])
    }

    fn commit_updates(
        &self,
        records: &BTreeMap<String, Record>,
        updates: &[(&Slot, &Version)],
    ) -> Result<BTreeMap<String, Record>, String> {
        let mut records = records.clone();
        for (slot, version) in updates {
        let record = records.entry(slot.key.clone()).or_default();
        let previous = match record.active.clone() {
            Some(v) => v,
            None => default_version(&self.root, &slot.path, slot.net)?,
        };
        record.history.retain(|v| v.id != version.id);
        record.history.insert(0, previous);
        record.history.truncate(2);
        record.active = Some((*version).clone());
        }
        persist_records(&self.root, &records)?;
        // The manifest rename in persist_records is the commit point. Never report failure after
        // it and leave memory stale.
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

    fn configured_pair(root: &Path) -> (Arc<Models>, PolicyPaths) {
        let walk = root.join("walk.onnx");
        let stand = root.join("stand.onnx");
        fs::write(&walk, b"walk-model").unwrap();
        fs::write(&stand, b"stand-model").unwrap();
        let store = Arc::new(Models::at(root.join("store"), Arc::new(ControlOwner::default())));
        let mut paths = PolicyPaths { walk, stand: Some(stand), ..Default::default() };
        store.resolve(&mut paths);
        assert!(store.load_error().is_none());
        (store, paths)
    }

    #[test]
    fn equal_existing_consumers_share_one_file_on_startup() {
        let dir = tempfile::tempdir().unwrap();
        let (store, _) = configured_pair(dir.path());
        let inner = store.inner.lock().unwrap();
        let walk = inner.records[&inner.slots["walk"].key].active.as_ref().unwrap();
        let stand = inner.records[&inner.slots["stand"].key].active.as_ref().unwrap();
        assert_eq!(walk.policy_path, stand.policy_path);
        assert_ne!(walk.path, stand.path);
        assert_eq!(fs::read(&walk.path).unwrap(), b"walk-model");
        assert_eq!(fs::read(&stand.path).unwrap(), b"stand-model");
        let status = store.status(&inner);
        assert!(status["slots"].as_array().unwrap().iter()
            .all(|slot| slot["shared_consumer"] == "walk_stand"));
    }

    #[test]
    fn every_slot_materializes_its_own_default_consumer_semantics() {
        let dir = tempfile::tempdir().unwrap();
        let model = |name: &str| {
            let path = dir.path().join(name);
            fs::write(&path, name).unwrap();
            path
        };
        let store = Models::at(dir.path().join("store"), Arc::new(ControlOwner::default()));
        let mut paths = PolicyPaths {
            walk: model("walk.onnx"),
            stand: Some(model("stand.onnx")),
            sitstand: Some(model("sitstand.onnx")),
            ground_pick: Some(model("ground-pick.onnx")),
            kick_left: Some(model("kick-left.onnx")),
            kick_right: Some(model("kick-right.onnx")),
            roulade: Some(model("roulade.onnx")),
        };
        store.resolve(&mut paths);
        assert!(store.load_error().is_none());
        let inner = store.inner.lock().unwrap();
        for (id, expected) in [
            ("walk", default_policy(Net::Walk)),
            ("stand", default_policy(Net::Stand)),
            ("sitstand", default_policy(Net::SitStand)),
            ("ground_pick", default_policy(Net::GroundPick)),
            ("kick_left", default_policy(Net::KickLeft)),
            ("kick_right", default_policy(Net::KickRight)),
            ("roulade", default_policy(Net::Roulade)),
        ] {
            let slot = &inner.slots[id];
            let policy = inner.records[&slot.key].active.as_ref().unwrap().policy_path.as_ref().unwrap();
            assert_eq!(fs::read(policy).unwrap(), expected, "wrong default for {id}");
        }
        assert_eq!(default_policy(Net::Walk), default_policy(Net::Stand));
        assert_ne!(default_policy(Net::Walk), default_policy(Net::GroundPick));
        assert_ne!(default_policy(Net::SitStand), default_policy(Net::Roulade));
    }

    #[test]
    fn paired_commit_is_atomic_and_shared_files_survive_history_cleanup() {
        let dir = tempfile::tempdir().unwrap();
        let (store, _) = configured_pair(dir.path());
        let (walk, stand, records) = {
            let inner = store.inner.lock().unwrap();
            (inner.slots["walk"].clone(), inner.slots["stand"].clone(), inner.records.clone())
        };
        let incoming = version(&store, "new-walk");
        let mut companion = records[&stand.key].active.clone().unwrap();
        let old_stand_model = companion.path.clone();
        companion.id = "new-stand-consumer".into();
        companion.policy_path = incoming.policy_path.clone();
        let before = fs::read(store.root.join("manifest.json")).unwrap();
        fs::create_dir(store.root.join("manifest.tmp")).unwrap();
        assert!(store.commit_updates(&records, &[(&walk, &incoming), (&stand, &companion)]).is_err());
        assert_eq!(fs::read(store.root.join("manifest.json")).unwrap(), before);
        fs::remove_dir(store.root.join("manifest.tmp")).unwrap();
        let updated = store.commit_updates(&records, &[(&walk, &incoming), (&stand, &companion)]).unwrap();
        assert_eq!(updated[&walk.key].active.as_ref().unwrap().policy_path,
            updated[&stand.key].active.as_ref().unwrap().policy_path);
        assert_eq!(updated[&stand.key].active.as_ref().unwrap().path, old_stand_model);
        assert_eq!(updated[&walk.key].history.len(), 1);
        assert_eq!(updated[&stand.key].history.len(), 1);
        let reloaded = Models::at(store.root.clone(), Arc::new(ControlOwner::default()));
        assert_eq!(reloaded.inner.lock().unwrap().records[&stand.key].active.as_ref().unwrap().id,
            "new-stand-consumer");
        let mut inner = store.inner.lock().unwrap();
        inner.records = updated;
        // Even if a version is pruned, files referenced by its paired slot remain alive.
        store.remove_version_if_unused(&inner, &incoming);
        assert!(incoming.policy_path.as_ref().unwrap().exists());
        assert!(old_stand_model.exists());
    }

    #[test]
    #[ignore = "requires root, managed Python and XDUCK_POLICY_TEST_ROOT outside /tmp; no motor IO"]
    fn paired_preparation_commit_and_rollback_use_one_consumer() {
        use std::os::unix::fs::PermissionsExt;
        let parent = std::env::var_os("XDUCK_POLICY_TEST_ROOT").expect("set a sandbox-visible test root");
        let dir = tempfile::Builder::new().prefix("walk-stand-test-").tempdir_in(parent).unwrap();
        fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).unwrap();
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../policies");
        let walk_model = dir.path().join("walk.onnx");
        let stand_model = dir.path().join("stand.onnx");
        fs::copy(source.join("alpha_walking.onnx"), &walk_model).unwrap();
        fs::copy(source.join("alpha_stand.onnx"), &stand_model).unwrap();
        let store = Arc::new(Models::at(dir.path().join("store"), Arc::new(ControlOwner::default())));
        let mut paths = PolicyPaths { walk: walk_model, stand: Some(stand_model), ..Default::default() };
        store.resolve(&mut paths);
        assert!(store.load_error().is_none());
        let mut controller = crate::control::Controller::new(&paths, 0.05,
            crate::control::Tuning::default(), crate::control::SkillTuning::default());
        store.restore_controls(&mut controller).unwrap();
        let snapshot = || {
            let inner = store.inner.lock().unwrap();
            (inner.slots.clone(), inner.records.clone())
        };
        let candidate = |slot: &Slot, tag: &str| {
            let (slots, records) = snapshot();
            let id = unique();
            let model = store.root.join(format!("{id}-model.onnx"));
            let policy = store.root.join(format!("{id}-policy.py"));
            let active = records[&slot.key].active.as_ref().unwrap();
            fs::copy(&active.path, &model).unwrap();
            fs::write(&policy, format!("{}\n# {tag}\n",
                String::from_utf8_lossy(default_policy(Net::Walk)))).unwrap();
            assert!(slots.values().any(|s| s.key == slot.key));
            Version { id, name: tag.into(), path: model, policy_path: Some(policy), contract: None }
        };
        for (target, peer) in [("walk", "stand"), ("stand", "walk")] {
            let (slots, before) = snapshot();
            let old_peer_model = before[&slots[peer].key].active.as_ref().unwrap().path.clone();
            let version = candidate(&slots[target], target);
            let shared_path = version.policy_path.clone();
            let pending = store.prepare_version(slots[target].clone(), version).unwrap();
            assert!(pending.companion.is_some());
            store.prepared(Ok(pending));
            store.tick(true, Some(&mut controller));
            let (_, after) = snapshot();
            assert_eq!(store.inner.lock().unwrap().phase, "done");
            assert_eq!(after[&slots[target].key].active.as_ref().unwrap().policy_path, shared_path);
            assert_eq!(after[&slots[peer].key].active.as_ref().unwrap().policy_path, shared_path);
            assert_eq!(after[&slots[peer].key].active.as_ref().unwrap().path, old_peer_model);
        }
        let (slots, before) = snapshot();
        let previous_walk = before[&slots["walk"].key].history[0].clone();
        let stand_model = before[&slots["stand"].key].active.as_ref().unwrap().path.clone();
        store.prepared(store.prepare_version(slots["walk"].clone(), previous_walk.clone()));
        store.tick(true, Some(&mut controller));
        let (_, after) = snapshot();
        assert_eq!(after[&slots["walk"].key].active.as_ref().unwrap().id, previous_walk.id);
        assert_eq!(after[&slots["stand"].key].active.as_ref().unwrap().policy_path, previous_walk.policy_path);
        assert_eq!(after[&slots["stand"].key].active.as_ref().unwrap().path, stand_model);

        let manifest = fs::read(store.root.join("manifest.json")).unwrap();
        let rejected = candidate(&slots["walk"], "rejected-not-relaxed");
        store.prepared(store.prepare_version(slots["walk"].clone(), rejected));
        store.tick(false, Some(&mut controller));
        assert_eq!(fs::read(store.root.join("manifest.json")).unwrap(), manifest);
        let invalid = candidate(&slots["stand"], "incompatible");
        fs::write(&invalid.path, b"not an ONNX").unwrap();
        assert!(store.prepare_version(slots["stand"].clone(), invalid).is_err());
        assert_eq!(fs::read(store.root.join("manifest.json")).unwrap(), manifest);
        let pending = store.prepare_version(slots["walk"].clone(), candidate(&slots["walk"], "disk-failure")).unwrap();
        fs::create_dir(store.root.join("manifest.tmp")).unwrap();
        store.prepared(Ok(pending));
        store.tick(true, Some(&mut controller));
        assert_eq!(store.inner.lock().unwrap().phase, "error");
        assert_eq!(fs::read(store.root.join("manifest.json")).unwrap(), manifest);
        let result = controller.step(&duck_control::Sensors::default(), &duck_control::obs::Command::default(), false, 0.02, 1.0);
        assert!(result.is_ok(), "failed commits must keep the old shared worker usable");
    }

    fn configured(root: &Path) -> (Arc<Models>, Slot) {
        let original = root.join("original.onnx");
        fs::write(&original, b"original").unwrap();
        let store = Arc::new(Models::at(
            root.join("store"),
            Arc::new(ControlOwner::default()),
        ));
        let mut paths = PolicyPaths { walk: original, ..Default::default() };
        store.resolve(&mut paths);
        let slot = store.inner.lock().unwrap().slots["walk"].clone();
        store.tick(true, None);
        (store, slot)
    }

    fn version(store: &Models, id: &str) -> Version {
        let path = store.root.join(format!("{id}-model.onnx"));
        let policy_path = store.root.join(format!("{id}-policy.py"));
        fs::write(&path, id).unwrap();
        fs::write(&policy_path, default_policy(Net::Walk)).unwrap();
        Version {
            id: id.into(),
            name: format!("{id}.pt"),
            path,
            policy_path: Some(policy_path),
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
        let initial = store.inner.lock().unwrap().records.clone();
        let records = store.commit(&initial, &slot, &a).unwrap();
        let original = &records[&slot.key].history[0];
        assert_eq!(fs::read(&original.path).unwrap(), b"original");
        assert_eq!(fs::read(original.policy_path.as_ref().unwrap()).unwrap(), default_policy(Net::Walk));
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
    fn resolve_migrates_every_legacy_onnx_to_the_default_two_file_policy() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("store");
        fs::create_dir_all(&root).unwrap();
        let configured_model = dir.path().join("original.onnx");
        let active_model = root.join("old-active.onnx");
        let history_model = root.join("old-history.onnx");
        fs::write(&configured_model, b"configured").unwrap();
        fs::write(&active_model, b"active").unwrap();
        fs::write(&history_model, b"history").unwrap();
        fs::write(root.join("manifest.json"), serde_json::to_vec(&serde_json::json!({
            "walk--original.onnx": {
                "active": {"id":"active","name":"legacy.pt","path":active_model,"control":{"kp":20,"kd":4,"action_scale":0.5}},
                "history": [{"id":"history","name":"alpha_walking.onnx","path":history_model}]
            }
        })).unwrap()).unwrap();
        let store = Models::at(root.clone(), Arc::new(ControlOwner::default()));
        let mut paths = PolicyPaths { walk: configured_model, ..Default::default() };
        store.resolve(&mut paths);
        assert!(store.load_error().is_none());
        let inner = store.inner.lock().unwrap();
        let record = &inner.records["walk--original.onnx"];
        for version in record.active.iter().chain(record.history.iter()) {
            assert_eq!(version.name, DEFAULT_POLICY_LABEL);
            assert!(version.path.ends_with(format!("{}-model.onnx", version.id)));
            let policy = version.policy_path.as_ref().unwrap();
            assert!(policy.ends_with(format!("{}-policy.py", version.id)));
            assert_eq!(fs::read(policy).unwrap(), default_policy(Net::Walk));
        }
        assert_eq!(fs::read(&paths.walk).unwrap(), b"active");
        assert!(!active_model.exists());
        assert!(!history_model.exists());
        let persisted = fs::read_to_string(root.join("manifest.json")).unwrap();
        assert!(!persisted.contains("\"control\""));
    }

    #[test]
    fn resolve_renames_an_already_migrated_default_without_model_2000() {
        let dir = tempfile::tempdir().unwrap();
        let (store, slot) = configured(dir.path());
        {
            let mut inner = store.inner.lock().unwrap();
            inner.records.get_mut(&slot.key).unwrap().active.as_mut().unwrap().name =
                "默认 + model_2000.pt".into();
            persist_records(&store.root, &inner.records).unwrap();
        }
        let reloaded = Models::at(store.root.clone(), Arc::new(ControlOwner::default()));
        let mut paths = PolicyPaths { walk: slot.path, ..Default::default() };
        reloaded.resolve(&mut paths);
        assert!(reloaded.load_error().is_none());
        let inner = reloaded.inner.lock().unwrap();
        assert_eq!(
            inner.records[&slot.key].active.as_ref().unwrap().name,
            "默认"
        );
        assert!(
            !fs::read_to_string(store.root.join("manifest.json"))
                .unwrap()
                .contains("model_2000")
        );
    }

    #[test]
    fn resolve_upgrades_a_platform_default_consumer_to_api_v2() {
        let dir = tempfile::tempdir().unwrap();
        let (store, slot) = configured(dir.path());
        let policy = store.inner.lock().unwrap().records[&slot.key]
            .active
            .as_ref()
            .unwrap()
            .policy_path
            .clone()
            .unwrap();
        fs::write(&policy, b"# platform default from API v1\n").unwrap();

        let reloaded = Models::at(store.root.clone(), Arc::new(ControlOwner::default()));
        let mut paths = PolicyPaths {
            walk: slot.path,
            ..Default::default()
        };
        reloaded.resolve(&mut paths);

        assert!(reloaded.load_error().is_none());
        assert_eq!(fs::read(policy).unwrap(), default_policy(Net::Walk));
    }

    #[test]
    fn failed_persistence_keeps_existing_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let (store, slot) = configured(dir.path());
        let a = version(&store, "a");
        let initial = store.inner.lock().unwrap().records.clone();
        let records = store.commit(&initial, &slot, &a).unwrap();
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
    fn bundle_chunks_require_role_token_exact_offset_and_declared_size() {
        let dir = tempfile::tempdir().unwrap();
        let (store, _) = configured(dir.path());
        let result = store.request(ModelParams::BeginBundle {
            slot: "walk".into(), policy_filename: None, policy_size: None,
            model_filename: "../../test.onnx".into(), model_size: 2,
            activation: None, normalizer_epsilon: None,
        }).unwrap();
        let token = result["token"].as_str().unwrap().to_owned();
        let inner = store.inner.lock().unwrap();
        let upload = inner.upload.as_ref().unwrap();
        assert!(upload.model.path.starts_with(&store.root));
        drop(inner);
        let chunk = |token: String, file: &str, offset, hex: &str| ModelParams::BundleChunk {
            token, file: file.into(), offset, hex: hex.into(),
        };
        assert!(store.request(chunk("wrong".into(), "model", 0, "0000")).is_err());
        assert!(store.request(chunk(token.clone(), "unknown", 0, "00")).is_err());
        assert!(store.request(chunk(token.clone(), "model", 1, "00")).is_err());
        assert!(store.request(chunk(token.clone(), "model", 0, "000000")).is_err());
        assert!(store.request(ModelParams::FinishBundle { token: token.clone() }).is_err());
        assert!(store.request(chunk(token.clone(), "model", 0, "00ff")).is_ok());
        assert!(store.request(chunk(token, "model", 0, "00ff")).is_err());
        assert!(decode_chunk("éé").is_err());
        assert!(decode_chunk(&"00".repeat(8193)).is_err());
    }

    #[test]
    fn bundle_upload_keeps_two_roles_in_one_transaction() {
        let dir = tempfile::tempdir().unwrap();
        let (store, _) = configured(dir.path());
        let status = store.request(ModelParams::BeginBundle {
            slot: "walk".into(),
            policy_filename: Some("../../policy.py".into()),
            policy_size: Some(2),
            model_filename: "../../model.pt".into(),
            model_size: 3,
            activation: Some("tanh".into()),
            normalizer_epsilon: Some(1e-4),
        }).unwrap();
        let token = status["token"].as_str().unwrap().to_owned();
        let chunk = |file: &str, offset, hex: &str| ModelParams::BundleChunk {
            token: token.clone(), file: file.into(), offset, hex: hex.into(),
        };
        assert!(store.request(chunk("policy", 0, "7079")).is_ok());
        assert!(store.request(chunk("model", 0, "000102")).is_ok());
        assert!(store.request(chunk("model", 0, "00")).is_err());
        let inner = store.inner.lock().unwrap();
        let upload = inner.upload.as_ref().unwrap();
        assert!(upload.policy.path.starts_with(&store.root));
        assert!(upload.model.path.starts_with(&store.root));
        assert_ne!(upload.policy.path, upload.model.path);
        assert_eq!(upload.model.path.extension().and_then(|value| value.to_str()), Some("pt"));
        assert_eq!(upload.activation, "tanh");
        assert_eq!(upload.normalizer_epsilon, 1e-4);
    }

    #[test]
    fn bundle_upload_uses_a_fresh_default_policy_when_policy_is_omitted() {
        let dir = tempfile::tempdir().unwrap();
        let (store, _) = configured(dir.path());
        store.request(ModelParams::BeginBundle {
            slot: "walk".into(),
            policy_filename: None,
            policy_size: None,
            model_filename: "policy.pth".into(),
            model_size: 3,
            activation: None,
            normalizer_epsilon: None,
        }).unwrap();
        let inner = store.inner.lock().unwrap();
        let upload = inner.upload.as_ref().unwrap();
        assert_eq!(upload.policy.name, DEFAULT_POLICY_LABEL);
        assert_eq!(upload.policy.received, default_policy(Net::Walk).len());
        assert_eq!(upload.policy.size, default_policy(Net::Walk).len());
        assert_eq!(fs::read(&upload.policy.path).unwrap(), default_policy(Net::Walk));
        assert_eq!(upload.activation, "elu");
        assert_eq!(upload.normalizer_epsilon, 0.01);
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
                .request(ModelParams::BeginBundle {
                    slot: "walk".into(), policy_filename: None, policy_size: None,
                    model_filename: "blocked.pt".into(), model_size: 1,
                    activation: None, normalizer_epsilon: None,
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
                .request(ModelParams::BeginBundle {
                    slot: "walk".into(), policy_filename: None, policy_size: None,
                    model_filename: "x.pt".into(), model_size: 1,
                    activation: None, normalizer_epsilon: None,
                })
                .is_err()
        );
    }
}
