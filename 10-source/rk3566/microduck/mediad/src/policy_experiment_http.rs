//! Same-origin HTTP adapter for robotd's controlled policy experiments.
use axum::{body::{Body, Bytes}, extract::Path, http::{HeaderMap, StatusCode}, response::{IntoResponse, Response}};
use duck_ipc_proto::{self as proto, PolicyExperimentParams as Params};
use serde_json::{Value, json};
use std::time::Duration;
use tokio::{io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader}, net::UnixStream, sync::Semaphore};

static DOWNLOADS: Semaphore = Semaphore::const_new(2);

fn error(status: StatusCode, message: impl ToString) -> Response {
    (status, [("content-type","application/json")], json!({"error":message.to_string()}).to_string()).into_response()
}
fn origin(headers:&HeaderMap)->Result<(),Response>{
    if headers.get("sec-fetch-site").is_some_and(|v|v=="cross-site") { return Err(error(StatusCode::FORBIDDEN,"cross-site control refused")); }
    if let Some(origin)=headers.get("origin") {
        let host=headers.get("host").and_then(|v|v.to_str().ok()).unwrap_or("");
        if origin.to_str().ok()!=Some(format!("http://{host}").as_str()) { return Err(error(StatusCode::FORBIDDEN,"origin mismatch")); }
    }
    Ok(())
}
async fn connect(params:Params)->Result<(BufReader<UnixStream>,Value),String>{
    tokio::time::timeout(Duration::from_secs(30),async{
        let mut stream=UnixStream::connect(proto::socket::ROBOT).await.map_err(|e|e.to_string())?;
        let mut line=serde_json::to_vec(&proto::Request::call(proto::Id::Number(1),&proto::Call::RobotPolicyExperiment(params))).map_err(|e|e.to_string())?;
        line.push(b'\n'); stream.write_all(&line).await.map_err(|e|e.to_string())?;
        let mut reader=BufReader::new(stream); let mut reply=String::new(); reader.read_line(&mut reply).await.map_err(|e|e.to_string())?;
        let value:Value=serde_json::from_str(&reply).map_err(|e|e.to_string())?;
        if let Some(e)=value.get("error") { return Err(e.to_string()); }
        Ok((reader,value["result"].clone()))
    }).await.map_err(|_|"robotd policy experiment request timed out".to_string())?
}
async fn call(params:Params)->Result<Value,String>{connect(params).await.map(|(_,v)|v)}
fn response(value:Result<Value,String>)->Response{match value{Ok(v)=>([("content-type","application/json"),("cache-control","no-store")],v.to_string()).into_response(),Err(e)=>error(StatusCode::CONFLICT,e)}}

pub async fn capabilities()->Response{response(call(Params::Capabilities{}).await)}
pub async fn list()->Response{response(call(Params::List{}).await)}
pub async fn status(Path(id):Path<String>)->Response{response(call(Params::Status{id}).await)}
pub async fn configure(headers:HeaderMap,body:Bytes)->Response{
    if let Err(e)=origin(&headers){return e}
    let config=match serde_json::from_slice(&body){Ok(v)=>v,Err(e)=>return error(StatusCode::BAD_REQUEST,e)};
    response(call(Params::Configure{config}).await)
}
pub async fn initialize(Path(id):Path<String>,headers:HeaderMap)->Response{
    if let Err(e)=origin(&headers){return e} response(call(Params::Initialize{id}).await)
}
pub async fn start(Path(id):Path<String>,headers:HeaderMap)->Response{
    if let Err(e)=origin(&headers){return e} response(call(Params::Start{id}).await)
}
pub async fn stop(Path(id):Path<String>,headers:HeaderMap,body:Bytes)->Response{
    if let Err(e)=origin(&headers){return e}
    let value:Value=match serde_json::from_slice(&body){Ok(v)=>v,Err(e)=>return error(StatusCode::BAD_REQUEST,e)};
    response(call(Params::Stop{id,run_token:value.get("run_token").and_then(Value::as_str).map(str::to_owned)}).await)
}
pub async fn acknowledge(Path(id):Path<String>,headers:HeaderMap,body:Bytes)->Response{
    if let Err(e)=origin(&headers){return e}
    let value:Value=match serde_json::from_slice(&body){Ok(v)=>v,Err(e)=>return error(StatusCode::BAD_REQUEST,e)};
    let Some(sha256)=value.get("sha256").and_then(Value::as_str).map(str::to_owned) else{return error(StatusCode::BAD_REQUEST,"sha256 required")};
    response(call(Params::Delete{id,sha256}).await)
}
pub async fn download(Path(id):Path<String>)->Response{
    let Ok(permit)=DOWNLOADS.try_acquire() else{return error(StatusCode::TOO_MANY_REQUESTS,"at most two policy experiment downloads")};
    let (reader,meta)=match connect(Params::Download{id:id.clone()}).await{Ok(v)=>v,Err(e)=>return error(StatusCode::CONFLICT,e)};
    let Some(bytes)=meta.get("bytes").and_then(Value::as_u64) else{return error(StatusCode::BAD_GATEWAY,"missing result size")};
    let stream=futures_util::stream::try_unfold((reader,permit),|(mut reader,permit)|async move{
        let mut buffer=vec![0u8;16384];
        let n=tokio::time::timeout(Duration::from_secs(30),reader.read(&mut buffer)).await.map_err(|_|std::io::Error::new(std::io::ErrorKind::TimedOut,"result download stalled"))??;
        if n==0{return Ok::<_,std::io::Error>(None)} buffer.truncate(n); Ok(Some((Bytes::from(buffer),(reader,permit))))
    });
    Response::builder().header("content-type","application/x-ndjson").header("content-length",bytes)
        .header("x-content-sha256",meta["sha256"].as_str().unwrap_or(""))
        .header("content-disposition",format!("attachment; filename=policy-experiment-{id}.jsonl"))
        .header("cache-control","no-store").body(Body::from_stream(stream)).unwrap()
}
