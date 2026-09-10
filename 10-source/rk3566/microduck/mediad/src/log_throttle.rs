//! Bound one repetitive capture warning without hiding other camera diagnostics.
use std::time::Duration;

#[derive(Default)]
pub(crate) struct CaptureWarning {
    last: Option<Duration>,
    suppressed: u64,
}

impl CaptureWarning {
    /// None suppresses a duplicate; Some(n) emits it with n earlier duplicates counted.
    /// Call only for WARNING level. Errors must never pass through this throttle.
    pub(crate) fn record(&mut self, now: Duration, category: &str, text: &str) -> Option<u64> {
        let index = text
            .strip_prefix("newly allocated buffer ")
            .and_then(|value| value.strip_suffix(" is not free"));
        let matches = category == "v4l2bufferpool"
            && index.is_some_and(|value| {
                !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit())
            });
        if !matches {
            return Some(0);
        }
        if self
            .last
            .is_some_and(|last| now.saturating_sub(last) < Duration::from_secs(60))
        {
            self.suppressed = self.suppressed.saturating_add(1);
            return None;
        }
        self.last = Some(now);
        Some(std::mem::take(&mut self.suppressed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn first_warning_and_minute_summary_survive() {
        let mut gate = CaptureWarning::default();
        assert_eq!(
            gate.record(
                Duration::ZERO,
                "v4l2bufferpool",
                "newly allocated buffer 0 is not free"
            ),
            Some(0)
        );
        for i in 1..=59 {
            assert_eq!(
                gate.record(
                    Duration::from_secs(i),
                    "v4l2bufferpool",
                    "newly allocated buffer 3 is not free"
                ),
                None
            );
        }
        assert_eq!(
            gate.record(
                Duration::from_secs(60),
                "v4l2bufferpool",
                "newly allocated buffer 1 is not free"
            ),
            Some(59)
        );
        assert_eq!(
            gate.record(
                Duration::from_secs(120),
                "v4l2bufferpool",
                "newly allocated buffer 2 is not free"
            ),
            Some(0)
        );
    }
    #[test]
    fn other_diagnostics_are_not_throttled() {
        let mut gate = CaptureWarning::default();
        for _ in 0..100 {
            assert_eq!(
                gate.record(
                    Duration::ZERO,
                    "v4l2src",
                    "newly allocated buffer 0 is not free"
                ),
                Some(0)
            );
            assert_eq!(
                gate.record(Duration::ZERO, "v4l2bufferpool", "buffer allocation failed"),
                Some(0)
            );
            assert_eq!(
                gate.record(
                    Duration::ZERO,
                    "v4l2bufferpool",
                    "newly allocated buffer 0 is not free: fatal"
                ),
                Some(0)
            );
        }
    }
}
