//! One process-wide lease for operations that may replace or drive a policy or own the motors.
use serde::Serialize;
use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Owner {
    PolicyImport,
    MotorExperiment,
    PolicyExperiment,
}

impl Owner {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PolicyImport => "policy_import",
            Self::MotorExperiment => "motor_experiment",
            Self::PolicyExperiment => "policy_experiment",
        }
    }
}

#[derive(Default)]
pub struct ControlOwner(Mutex<Option<Owner>>);

impl ControlOwner {
    pub fn current(&self) -> Option<Owner> {
        *self.0.lock().unwrap()
    }

    /// Acquiring an already-owned lease is idempotent for the owning subsystem. This lets a
    /// chunked upload continue while every competing subsystem is still rejected.
    /// Returns `true` when this call created the lease and `false` for an idempotent
    /// continuation by the existing owner.
    pub fn acquire(&self, wanted: Owner) -> Result<bool, String> {
        let mut owner = self.0.lock().unwrap();
        match *owner {
            None => {
                *owner = Some(wanted);
                Ok(true)
            }
            Some(current) if current == wanted => Ok(false),
            Some(current) => Err(format!(
                "机器人当前由 {} 占用；请等待其结束或先放松",
                current.as_str()
            )),
        }
    }

    pub fn release(&self, expected: Owner) {
        let mut owner = self.0.lock().unwrap();
        if *owner == Some(expected) {
            *owner = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lease_is_exclusive_and_owner_release_is_scoped() {
        let lease = ControlOwner::default();
        assert!(lease.acquire(Owner::MotorExperiment).unwrap());
        assert!(!lease.acquire(Owner::MotorExperiment).unwrap());
        assert!(lease.acquire(Owner::PolicyExperiment).is_err());
        lease.release(Owner::PolicyImport);
        assert_eq!(lease.current(), Some(Owner::MotorExperiment));
        lease.release(Owner::MotorExperiment);
        assert_eq!(lease.current(), None);
    }
}
