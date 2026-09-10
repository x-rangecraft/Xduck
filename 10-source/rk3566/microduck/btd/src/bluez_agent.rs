//! Headless Just Works agent with an explicit NoInputNoOutput capability.
//!
//! bluer 0.17 infers DisplayYesNo when RequestAuthorization is implemented, but
//! rejects that request when the callback is absent. Use the existing dbus
//! dependency to keep the actual IO capability while accepting pairing consent.
use dbus::blocking::Connection;
use dbus::channel::{MatchingReceiver, Sender};
use dbus::message::MatchRule;
use dbus::{Message, Path};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

const PATH: &str = "/org/microduck/pairing_agent";
const CAPABILITY: &str = "NoInputNoOutput";

pub struct HeadlessAgent(Arc<AtomicBool>);

impl HeadlessAgent {
    pub async fn register(adapter: &str) -> bluer::Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let running = stop.clone();
        let adapter = adapter.to_owned();
        let (tx, rx) = tokio::sync::oneshot::channel();
        std::thread::Builder::new()
            .name("ble-pairing".into())
            .spawn(move || {
                let connection = setup(&adapter);
                match connection {
                    Err(e) => {
                        let _ = tx.send(Err(e));
                    }
                    Ok(conn) => {
                        if tx.send(Ok(())).is_err() {
                            return;
                        }
                        while !running.load(Ordering::Relaxed) {
                            if let Err(e) = conn.process(Duration::from_millis(200)) {
                                tracing::warn!(error=%e, "pairing agent D-Bus connection lost");
                                break;
                            }
                        }
                        // Closing this private bus connection unregisters the agent.
                    }
                }
            })?;
        let handle = Self(stop);
        rx.await
            .map_err(|_| std::io::Error::other("pairing agent exited during registration"))??;
        Ok(handle)
    }
}

impl Drop for HeadlessAgent {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

fn setup(adapter: &str) -> Result<Connection, dbus::Error> {
    let conn = Connection::new_system()?;
    let bus = conn.with_proxy(
        "org.freedesktop.DBus",
        "/org/freedesktop/DBus",
        Duration::from_secs(5),
    );
    let (owner,): (String,) =
        bus.method_call("org.freedesktop.DBus", "GetNameOwner", ("org.bluez",))?;
    let prefix = format!("/org/bluez/{adapter}/dev_");
    conn.start_receive(
        MatchRule::new_method_call()
            .with_path(PATH)
            .with_interface("org.bluez.Agent1"),
        Box::new(move |message, connection| {
            // Only bluetoothd may ask this agent to authorize an incoming pairing.
            let trusted = message.sender().is_some_and(|s| s.to_string() == owner);
            let accepted = trusted && allows(&message, &prefix);
            let member = message.member().map(|m| m.to_string()).unwrap_or_default();
            tracing::info!(method=%member, accepted, "BLE pairing agent request");
            let response = if accepted {
                message.method_return()
            } else {
                message.error(
                    &"org.bluez.Error.Rejected".into(),
                    c"Unsupported pairing request",
                )
            };
            let _ = connection.send(response);
            true
        }),
    );
    let manager = conn.with_proxy("org.bluez", "/org/bluez", Duration::from_secs(5));
    let path = Path::new(PATH).expect("static agent path");
    manager.method_call::<(), _, _, _>(
        "org.bluez.AgentManager1",
        "RegisterAgent",
        (path.clone(), CAPABILITY),
    )?;
    manager.method_call::<(), _, _, _>(
        "org.bluez.AgentManager1",
        "RequestDefaultAgent",
        (path,),
    )?;
    Ok(conn)
}

fn allows(message: &Message, device_prefix: &str) -> bool {
    match message.member().map(|m| m.to_string()).as_deref() {
        Some("RequestAuthorization") => message
            .read1::<Path<'_>>()
            .is_ok_and(|device| device.starts_with(device_prefix)),
        Some("Cancel" | "Release") => true,
        // Never claim to display/input/verify a passkey. Also do not authorize
        // unrelated Bluetooth profiles; application access still requires our PIN.
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request(method: &str) -> Message {
        Message::new_method_call("org.microduck.Test", PATH, "org.bluez.Agent1", method).unwrap()
    }
    #[test]
    fn android_incoming_just_works_authorization_is_accepted() {
        let m = request("RequestAuthorization")
            .append1(Path::new("/org/bluez/hci0/dev_00_11_22_33_44_55").unwrap());
        assert!(allows(&m, "/org/bluez/hci0/dev_"));
        assert!(!allows(&m, "/org/bluez/hci1/dev_"));
        assert_eq!(CAPABILITY, "NoInputNoOutput");
    }
    #[test]
    fn malformed_and_unrelated_requests_are_rejected() {
        assert!(!allows(
            &request("RequestAuthorization"),
            "/org/bluez/hci0/dev_"
        ));
        for method in [
            "RequestConfirmation",
            "RequestPasskey",
            "RequestPinCode",
            "AuthorizeService",
        ] {
            assert!(!allows(&request(method), "/org/bluez/hci0/dev_"));
        }
        assert!(allows(&request("Cancel"), "/org/bluez/hci0/dev_"));
        assert!(allows(&request("Release"), "/org/bluez/hci0/dev_"));
    }
}
