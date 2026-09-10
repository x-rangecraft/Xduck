//! Per-viewer encoded-video gate: drain startup frames without blocking negotiation.
#[derive(Default)]
pub(crate) struct VideoStart {
    ready: bool,
    started: bool,
    skipped: u64,
}

pub(crate) enum Frame {
    Drop,
    First { skipped: u64 },
    Forward,
}

impl VideoStart {
    /// Returns true only on a new transport-ready transition.
    pub(crate) fn set_ready(&mut self, ready: bool) -> bool {
        let changed = ready != self.ready;
        self.ready = ready;
        if changed {
            self.started = false;
        }
        changed && ready
    }

    pub(crate) fn frame(&mut self, keyframe: bool) -> Frame {
        if !self.ready || (!self.started && !keyframe) {
            self.skipped = self.skipped.saturating_add(1);
            return Frame::Drop;
        }
        if !self.started {
            self.started = true;
            return Frame::First {
                skipped: std::mem::take(&mut self.skipped),
            };
        }
        Frame::Forward
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn drains_before_ready_and_starts_with_keyframe() {
        let mut gate = VideoStart::default();
        assert!(matches!(gate.frame(true), Frame::Drop));
        assert!(matches!(gate.frame(false), Frame::Drop));
        assert!(gate.set_ready(true));
        assert!(!gate.set_ready(true));
        assert!(matches!(gate.frame(false), Frame::Drop));
        assert!(matches!(gate.frame(true), Frame::First { skipped: 3 }));
        assert!(matches!(gate.frame(false), Frame::Forward));
    }
    #[test]
    fn reconnect_requires_a_fresh_keyframe() {
        let mut gate = VideoStart::default();
        gate.set_ready(true);
        assert!(matches!(gate.frame(true), Frame::First { skipped: 0 }));
        gate.set_ready(false);
        assert!(matches!(gate.frame(true), Frame::Drop));
        assert!(gate.set_ready(true));
        assert!(matches!(gate.frame(false), Frame::Drop));
        assert!(matches!(gate.frame(true), Frame::First { skipped: 2 }));
    }
}
