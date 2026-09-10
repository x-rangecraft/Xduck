//! Same-origin, one-request control transport for the console when video is off.
//!
//! WebRTC's default signaller always creates the offered video consumer before its data channel.
//! An inactive answer therefore cannot be used as a reliable control-only session. This endpoint
//! keeps the exact same route allowlist and JSON-RPC envelopes, but opens one bounded Unix-socket
//! request at a time. `robot.subscribe` is a long poll: it returns the first state notification.

use axum::{
    body::Bytes,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use duck_ipc_proto as proto;
use serde_json::json;
use std::time::Duration;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::UnixStream,
};

fn error(status: StatusCode, id: Option<proto::Id>, message: impl ToString) -> Response {
    (
        status,
        [("content-type", "application/json"), ("cache-control", "no-store")],
        serde_json::to_string(&proto::Response::err(
            id,
            proto::Error::new(proto::code::INTERNAL_ERROR, message.to_string()),
        ))
        .unwrap_or_else(|_| json!({"error":"control gateway failed"}).to_string()),
    )
        .into_response()
}

fn same_origin(headers: &HeaderMap) -> bool {
    if headers
        .get("sec-fetch-site")
        .is_some_and(|value| value == "cross-site")
    {
        return false;
    }
    match headers.get("origin") {
        None => true,
        Some(origin) => {
            let host = headers
                .get("host")
                .and_then(|value| value.to_str().ok())
                .unwrap_or("");
            origin.to_str().ok() == Some(format!("http://{host}").as_str())
        }
    }
}

fn socket(service: proto::Service) -> &'static str {
    match service {
        proto::Service::Updater => proto::socket::UPDATER,
        proto::Service::Robot => proto::socket::ROBOT,
        proto::Service::Config => proto::socket::CONFIG,
        proto::Service::Pad => proto::socket::PAD,
        proto::Service::Tof => proto::socket::TOF,
    }
}

pub async fn call(headers: HeaderMap, body: Bytes) -> Response {
    if !same_origin(&headers) {
        return error(StatusCode::FORBIDDEN, None, "cross-site control refused");
    }
    let request: proto::Request = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(parse) => return error(StatusCode::BAD_REQUEST, None, parse),
    };
    let id = request.id.clone();
    let call = match request.as_call() {
        Ok(call) => call,
        Err(reason) => return error(StatusCode::BAD_REQUEST, id, reason.message),
    };
    let (service, _) = match crate::route::route_for(&call) {
        crate::route::Route::To(service, lane) => (service, lane),
        crate::route::Route::Refused => {
            let refusal = crate::route::refusal(&call);
            return (
                StatusCode::FORBIDDEN,
                [("content-type", "application/json"), ("cache-control", "no-store")],
                serde_json::to_string(&proto::Response::err(id, refusal))
                    .unwrap_or_else(|_| json!({"error":"control refused"}).to_string()),
            )
                .into_response();
        }
    };
    let subscription = matches!(call, proto::Call::RobotSubscribe(_));
    let operation = async {
        let mut stream = UnixStream::connect(socket(service)).await?;
        stream.write_all(&body).await?;
        stream.write_all(b"\n").await?;
        stream.flush().await?;
        let mut reader = BufReader::new(stream);
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).await? == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "service closed before answering",
                ));
            }
            if subscription {
                let value: serde_json::Value = serde_json::from_str(&line).map_err(|parse| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, parse)
                })?;
                if value.get("method").and_then(|value| value.as_str()) != Some("robot.state") {
                    continue;
                }
            }
            return Ok::<_, std::io::Error>(line);
        }
    };
    match tokio::time::timeout(Duration::from_secs(10), operation).await {
        Ok(Ok(line)) => (
            [("content-type", "application/json"), ("cache-control", "no-store")],
            line,
        )
            .into_response(),
        Ok(Err(reason)) => error(StatusCode::SERVICE_UNAVAILABLE, id, reason),
        Err(_) => error(StatusCode::GATEWAY_TIMEOUT, id, "control request timed out"),
    }
}
