//! Policy slot definitions shared with `robotd`.
//!
//! ONNX execution deliberately does not live in `duck-control`: every configured model is paired
//! with a `policy.py` and consumed through robotd's isolated two-file worker. Keeping only slot
//! identity and configured paths here makes a legacy single-file inference fallback impossible.

use std::path::PathBuf;

/// Below this velocity magnitude the standing slot takes over when one is configured.
pub const DEFAULT_STANDING_THRESHOLD: f64 = 0.05;

#[derive(Debug, thiserror::Error)]
pub enum PolicyError {
    #[error("inference failed: {0}")]
    Inference(String),
}

/// Which network drives a tick. Selection remains in robotd's skill scheduler; execution always
/// goes through the two-file worker associated with the selected slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Net {
    Walk,
    Stand,
    SitStand,
    GroundPick,
    KickLeft,
    KickRight,
    Roulade,
}

/// Configured model files. `walk` is mandatory; each `None` optional path removes that capability.
/// `robotd::models` materializes every present path into a persistent model + policy pair before
/// constructing the controller.
#[derive(Debug, Clone, Default)]
pub struct PolicyPaths {
    pub walk: PathBuf,
    pub stand: Option<PathBuf>,
    pub sitstand: Option<PathBuf>,
    pub ground_pick: Option<PathBuf>,
    pub kick_left: Option<PathBuf>,
    pub kick_right: Option<PathBuf>,
    pub roulade: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standing_threshold_remains_the_runtime_contract() {
        assert_eq!(DEFAULT_STANDING_THRESHOLD, 0.05);
    }

    #[test]
    fn optional_paths_are_capabilities() {
        let paths = PolicyPaths {
            walk: "walk.onnx".into(),
            stand: Some("stand.onnx".into()),
            ..Default::default()
        };
        assert_eq!(paths.walk, PathBuf::from("walk.onnx"));
        assert_eq!(paths.stand, Some(PathBuf::from("stand.onnx")));
        assert!(paths.sitstand.is_none());
    }
}
