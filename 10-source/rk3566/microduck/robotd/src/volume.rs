//! Speaker mixer access belongs to robotd. Never run a subprocess on the control loop.
use duck_ipc_proto as proto;
use std::time::Duration;
use tokio::{process::Command, sync::Mutex};

pub struct Volume {
    card: Option<String>,
    gate: Mutex<()>,
}
impl Volume {
    pub fn new(enabled: bool, device: &str) -> Self {
        // RK809's vendor DAC control does not retain writes on this board. The
        // deployment provisions a softvol PCM; its user control is stable and read back.
        Self {
            card: (enabled && device == "xduck_speaker").then(|| "rockchiprk809".into()),
            gate: Mutex::new(()),
        }
    }
    pub async fn request(&self, percent: Option<u8>) -> Result<proto::VolumeResult, proto::Error> {
        if percent.is_some_and(|p| p > 100) {
            return Err(proto::Error::new(proto::code::INVALID_PARAMS, "volume must be 0–100"));
        }
        let card = self.card.as_ref().ok_or_else(|| failure("speaker volume is unavailable on this audio device"))?;
        let _guard = self.gate.lock().await;
        if let Some(p) = percent {
            mixer(card, Some(p)).await?;
        }
        // amixer's setter prints its local cache; only a fresh open confirms the value.
        let output = mixer(card, None).await?;
        let percent = parse_percent(&output)
            .ok_or_else(|| failure("audio mixer returned no playback volume"))?;
        Ok(proto::VolumeResult { percent })
    }
}
async fn mixer(card: &str, percent: Option<u8>) -> Result<String, proto::Error> {
    let mut command = Command::new("amixer");
    command.env("LC_ALL", "C").args(["-M", "-c", card]);
    if let Some(p) = percent {
        command.args(["sset", "Xduck", &format!("{p}%")]);
    } else {
        command.args(["sget", "Xduck"]);
    }
    command.kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_secs(2), command.output())
        .await.map_err(|_| failure("audio mixer timed out"))?
        .map_err(|e| failure(e.to_string()))?;
    if !output.status.success() {
        return Err(failure(String::from_utf8_lossy(&output.stderr).trim()));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}
fn failure(message: impl Into<String>) -> proto::Error {
    proto::Error::new(proto::code::INTERNAL_ERROR, message)
}
fn parse_percent(text: &str) -> Option<u8> {
    text.lines().filter(|l| l.contains("Playback")).filter_map(|l| {
        let before = l.split_once("%]")?.0;
        before.rsplit_once('[')?.1.parse::<u8>().ok().filter(|p| *p <= 100)
    }).max()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_playback_not_capture_and_rejects_malformed_output() {
        assert_eq!(parse_percent("Capture [100%]\nFront Left: Playback 252 [96%] [-2.23dB]\nFront Right: Playback 240 [82%]"), Some(96));
        assert_eq!(parse_percent("Playback [0%]"), Some(0));
        assert_eq!(parse_percent("Playback [101%]"), None);
        assert_eq!(parse_percent("amixer: error"), None);
    }
    #[tokio::test]
    async fn invalid_values_and_unsupported_devices_do_not_touch_hardware() {
        let volume = Volume::new(true, "plughw:unknown");
        assert_eq!(volume.request(Some(101)).await.unwrap_err().code, proto::code::INVALID_PARAMS);
        assert!(volume.request(None).await.is_err());
        assert!(Volume::new(false, "plughw:rockchiprk809").request(Some(20)).await.is_err());
    }
}
