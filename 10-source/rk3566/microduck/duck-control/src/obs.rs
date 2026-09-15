//! Platform command types exposed to robotd's policy scheduler.
//!
//! This crate deliberately does not build a model observation.  API-v2 policy consumers own
//! their complete tensor layout and any action-specific command encoding; the platform carries
//! only the physical command before that encoding.

/// What a client is asking the robot to do after platform input gating and smoothing, but before
/// a policy consumer maps it into model-specific observations.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Command {
    /// Forward, left, yaw-rate.
    pub twist: [f64; 3],
    /// neck_pitch, head_pitch, head_yaw, head_roll.
    pub head: [f64; 4],
    /// Body-pose offsets in the trunk frame.
    pub body: BodyPose,
}

impl Command {
    /// Magnitude used by the scheduler to choose walking versus standing.  Head and body do not
    /// participate in network selection; their model-facing meaning belongs to policy.py.
    pub fn twist_magnitude(&self) -> f64 {
        self.twist.iter().map(|value| value * value).sum::<f64>().sqrt()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BodyPose {
    pub z: f64,
    pub roll: f64,
    pub pitch: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_magnitude_uses_only_the_physical_twist() {
        let command = Command {
            twist: [3.0, 4.0, 0.0],
            head: [1.0; 4],
            body: BodyPose { z: 1.0, roll: 1.0, pitch: 1.0 },
        };
        assert_eq!(command.twist_magnitude(), 5.0);
    }
}
