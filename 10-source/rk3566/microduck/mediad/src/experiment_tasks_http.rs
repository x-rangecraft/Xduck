//! File task HTTP gateway. robotd owns every byte on disk and every task state.
use axum::{
    body::{Body, Bytes},
    extract::Path,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use duck_ipc_proto::{self as proto, ExperimentTaskParams as Task};
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::time::Duration;
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    net::UnixStream,
    sync::Semaphore,
};
static UPLOADS: Semaphore = Semaphore::const_new(1);
static DOWNLOADS: Semaphore = Semaphore::const_new(2);
const MAX_INPUT: u64 = 128 * 1024 * 1024;
fn error(status: StatusCode, message: impl ToString) -> Response {
    (
        status,
        [("content-type", "application/json")],
        json!({"error":message.to_string()}).to_string(),
    )
        .into_response()
}
fn origin(headers: &HeaderMap) -> Result<(), Response> {
    if headers
        .get("sec-fetch-site")
        .is_some_and(|v| v == "cross-site")
    {
        return Err(error(StatusCode::FORBIDDEN, "cross-site control refused"));
    }
    if let Some(origin) = headers.get("origin") {
        let host = headers
            .get("host")
            .and_then(|h| h.to_str().ok())
            .unwrap_or("");
        if origin.to_str().ok() != Some(format!("http://{host}").as_str()) {
            return Err(error(StatusCode::FORBIDDEN, "origin mismatch"));
        }
    }
    Ok(())
}
async fn connect(
    params: proto::ExperimentParams,
) -> Result<(BufReader<UnixStream>, Value), String> {
    tokio::time::timeout(Duration::from_secs(30), async {
        let mut stream = UnixStream::connect(proto::socket::ROBOT)
            .await
            .map_err(|e| e.to_string())?;
        let mut line = serde_json::to_vec(&proto::Request::call(
            proto::Id::Number(1),
            &proto::Call::RobotExperiment(params),
        ))
        .map_err(|e| e.to_string())?;
        line.push(b'\n');
        stream.write_all(&line).await.map_err(|e| e.to_string())?;
        let mut reader = BufReader::new(stream);
        let mut reply = String::new();
        reader
            .read_line(&mut reply)
            .await
            .map_err(|e| e.to_string())?;
        let value: Value = serde_json::from_str(&reply).map_err(|e| e.to_string())?;
        if let Some(e) = value.get("error") {
            return Err(e.to_string());
        }
        Ok((reader, value["result"].clone()))
    })
    .await
    .map_err(|_| "robotd task request timed out".to_string())?
}
async fn call(task: Task) -> Result<Value, String> {
    connect(proto::ExperimentParams::Task { request: task })
        .await
        .map(|(_, v)| v)
}
fn response(value: Result<Value, String>) -> Response {
    match value {
        Ok(v) => (
            [
                ("content-type", "application/json"),
                ("cache-control", "no-store"),
            ],
            v.to_string(),
        )
            .into_response(),
        Err(e) => error(StatusCode::CONFLICT, e),
    }
}
pub async fn capabilities() -> Response {
    response(
        connect(proto::ExperimentParams::Capabilities {})
            .await
            .map(|(_, v)| v),
    )
}
pub async fn list() -> Response {
    response(call(Task::List {}).await)
}
pub async fn status(Path(id): Path<String>) -> Response {
    response(call(Task::Status { id }).await)
}
pub async fn start(Path(id): Path<String>, headers: HeaderMap) -> Response {
    if let Err(e) = origin(&headers) {
        return e;
    }
    response(call(Task::Start { id }).await)
}
pub async fn stop(Path(id): Path<String>, headers: HeaderMap, body: Bytes) -> Response {
    if let Err(e) = origin(&headers) {
        return e;
    }
    let value: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let run_token = value.get("run_token").and_then(Value::as_str).map(str::to_owned);
    response(
        call(Task::Stop {
            id,
            run_token,
        })
        .await,
    )
}
pub async fn acknowledge(Path(id): Path<String>, headers: HeaderMap, body: Bytes) -> Response {
    if let Err(e) = origin(&headers) {
        return e;
    }
    let value: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let Some(sha) = value.get("sha256").and_then(Value::as_str) else {
        return error(StatusCode::BAD_REQUEST, "sha256 required");
    };
    response(
        call(Task::Delete {
            id,
            sha256: sha.into(),
        })
        .await,
    )
}
struct UploadGuard(Option<String>);
impl Drop for UploadGuard {
    fn drop(&mut self) {
        if let Some(id) = self.0.take() {
            tokio::spawn(async move {
                let _ = call(Task::UploadAbort { id }).await;
            });
        }
    }
}
pub async fn upload(headers: HeaderMap, body: Body) -> Response {
    if let Err(e) = origin(&headers) {
        return e;
    }
    let Ok(_permit) = UPLOADS.try_acquire() else {
        return error(StatusCode::TOO_MANY_REQUESTS, "one task upload at a time");
    };
    let Some(bytes) = headers
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|n| *n > 0 && *n <= MAX_INPUT)
    else {
        return error(
            StatusCode::BAD_REQUEST,
            "Content-Length must be 1..134217728",
        );
    };
    let Some(sha256) = headers
        .get("x-content-sha256")
        .and_then(|v| v.to_str().ok())
    else {
        return error(StatusCode::BAD_REQUEST, "X-Content-SHA256 required");
    };
    if headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.split(';').next())
        != Some(Some("application/x-ndjson"))
    {
        return error(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "use application/x-ndjson",
        );
    }
    let begin = match call(Task::UploadBegin {
        bytes,
        sha256: sha256.into(),
    })
    .await
    {
        Ok(v) => v,
        Err(e) => return error(StatusCode::CONFLICT, e),
    };
    let Some(id) = begin.get("id").and_then(Value::as_str).map(str::to_owned) else {
        return error(StatusCode::BAD_GATEWAY, "missing upload id");
    };
    let mut guard = UploadGuard(Some(id.clone()));
    let mut offset = 0u64;
    let mut stream = body.into_data_stream();
    loop {
        let next = match tokio::time::timeout(Duration::from_secs(30), stream.next()).await {
            Ok(v) => v,
            Err(_) => return error(StatusCode::REQUEST_TIMEOUT, "upload stalled"),
        };
        let Some(chunk) = next else { break };
        let chunk = match chunk {
            Ok(b) => b,
            Err(e) => return error(StatusCode::BAD_REQUEST, e),
        };
        if offset + chunk.len() as u64 > bytes {
            return error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "upload exceeds declared size",
            );
        }
        for part in chunk.chunks(8192) {
            if let Err(e) = call(Task::UploadChunk {
                id: id.clone(),
                offset,
                data: part.to_vec(),
            })
            .await
            {
                return error(StatusCode::CONFLICT, e);
            }
            offset += part.len() as u64;
        }
    }
    let result = call(Task::UploadCommit { id }).await;
    if result.is_ok() {
        guard.0 = None;
    }
    response(result)
}
pub async fn download(Path(id): Path<String>) -> Response {
    let Ok(permit) = DOWNLOADS.try_acquire() else {
        return error(StatusCode::TOO_MANY_REQUESTS, "at most two task downloads");
    };
    let (reader, meta) = match connect(proto::ExperimentParams::Task {
        request: Task::Download { id: id.clone() },
    })
    .await
    {
        Ok(v) => v,
        Err(e) => return error(StatusCode::CONFLICT, e),
    };
    let Some(bytes) = meta.get("bytes").and_then(Value::as_u64) else {
        return error(StatusCode::BAD_GATEWAY, "missing result size");
    };
    let stream =
        futures_util::stream::try_unfold((reader, permit), |(mut reader, permit)| async move {
            let mut buffer = vec![0u8; 16384];
            let n = tokio::time::timeout(Duration::from_secs(30), reader.read(&mut buffer))
                .await
                .map_err(|_| {
                    std::io::Error::new(std::io::ErrorKind::TimedOut, "result download stalled")
                })??;
            if n == 0 {
                return Ok::<_, std::io::Error>(None);
            }
            buffer.truncate(n);
            Ok(Some((Bytes::from(buffer), (reader, permit))))
        });
    Response::builder()
        .header("content-type", "application/x-tar")
        .header("content-length", bytes)
        .header("x-content-sha256", meta["sha256"].as_str().unwrap_or(""))
        .header(
            "content-disposition",
            format!("attachment; filename=experiment-{id}.tar"),
        )
        .header("cache-control", "no-store")
        .body(Body::from_stream(stream))
        .unwrap()
}
