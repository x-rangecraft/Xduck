//! Journal snapshots stream directly to the caller; child exits when download is dropped.
use axum::{
    body::{Body, Bytes},
    extract::RawQuery,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use std::{
    process::Stdio,
    sync::{Arc, LazyLock},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{io::AsyncReadExt, process::Command, sync::Semaphore};
static DOWNLOADS: LazyLock<Arc<Semaphore>> = LazyLock::new(|| Arc::new(Semaphore::new(2)));
fn options(query: Option<&str>) -> Result<(Vec<String>, &'static str), String> {
    let mut args = vec!["--no-pager".into(), "--utc".into(), "--quiet".into()];
    let mut format = "text";
    let mut since = None;
    let mut until = None;
    let mut seen = Vec::new();
    for field in query.unwrap_or("").split('&').filter(|s| !s.is_empty()) {
        let (key, value) = field.split_once('=').ok_or("expected key=value")?;
        if seen.contains(&key) {
            return Err("duplicate query parameter".into());
        }
        seen.push(key);
        match key {
            "since" | "until" => {
                let timestamp = value
                    .parse::<u64>()
                    .map_err(|_| "time must be Unix seconds")?;
                if timestamp > 253402300799 {
                    return Err("timestamp out of range".into());
                }
                if key == "since" {
                    since = Some(timestamp)
                } else {
                    until = Some(timestamp)
                }
                args.push(format!("--{key}=@{timestamp}"));
            }
            "unit" => {
                if value != "all" {
                    if ![
                        "robotd",
                        "mediad",
                        "configd",
                        "btd",
                        "padd",
                        "tofd",
                        "updaterd",
                        "xpad-usbd",
                        "irqbalance",
                    ]
                    .contains(&value)
                    {
                        return Err("unsupported unit".into());
                    }
                    args.push(format!("--unit={value}.service"));
                }
            }
            "format" => match value {
                "text" => {}
                "jsonl" => format = "jsonl",
                _ => return Err("format must be text or jsonl".into()),
            },
            _ => return Err("unknown query parameter".into()),
        }
    }
    if since.zip(until).is_some_and(|(a, b)| a > b) {
        return Err("since must not exceed until".into());
    }
    // Freeze the end of the snapshot, even while new entries arrive.
    if until.is_none() {
        args.push(format!(
            "--until=@{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        ))
    }
    args.push(
        if format == "jsonl" {
            "--output=json"
        } else {
            "--output=short-iso-precise"
        }
        .into(),
    );
    Ok((args, format))
}
pub async fn download(RawQuery(query): RawQuery) -> Response {
    let (args, format) = match options(query.as_deref()) {
        Ok(v) => v,
        Err(e) => return (StatusCode::BAD_REQUEST, e).into_response(),
    };
    let permit = match DOWNLOADS.clone().try_acquire_owned() {
        Ok(v) => v,
        Err(_) => {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                "two downloads already running",
            )
                .into_response();
        }
    };
    let mut child = match Command::new("journalctl")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
    {
        Ok(c) => c,
        Err(e) => return (StatusCode::SERVICE_UNAVAILABLE, e.to_string()).into_response(),
    };
    let stdout = child.stdout.take().unwrap();
    let stream = futures_util::stream::try_unfold(
        (child, stdout, permit),
        |(mut child, mut stdout, permit)| async move {
            let mut buffer = vec![0u8; 16384];
            let n = stdout.read(&mut buffer).await?;
            if n == 0 {
                if !child.wait().await?.success() {
                    return Err(std::io::Error::other("journalctl export failed"));
                }
                return Ok(None);
            }
            buffer.truncate(n);
            Ok(Some((Bytes::from(buffer), (child, stdout, permit))))
        },
    );
    let ext = if format == "jsonl" { "jsonl" } else { "log" };
    let epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    Response::builder()
        .header(
            "content-type",
            if format == "jsonl" {
                "application/x-ndjson"
            } else {
                "text/plain; charset=utf-8"
            },
        )
        .header(
            "content-disposition",
            format!("attachment; filename=\"rk3566-journal-{epoch}.{ext}\""),
        )
        .header("cache-control", "no-store")
        .body(Body::from_stream(stream))
        .unwrap()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn query_rejects_options_and_invalid_ranges() {
        for q in [
            "unit=--vacuum-size=1M",
            "since=now",
            "until=1&since=2",
            "format=jsonl&format=text",
            "path=/etc/passwd",
        ] {
            assert!(options(Some(q)).is_err(), "{q}")
        }
    }
    #[test]
    fn supports_full_snapshot_and_time_filtered_unit() {
        assert!(options(None).is_ok());
        let (args, format) = options(Some("unit=robotd&since=100&until=200&format=jsonl")).unwrap();
        assert_eq!(format, "jsonl");
        assert!(args.contains(&"--since=@100".into()));
        assert!(args.contains(&"--unit=robotd.service".into()));
    }
}
