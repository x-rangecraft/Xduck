//! Turning sensors and a command into joint targets — and scheduling the skills.
//!
//! Everything here is pure computation between [`duck_control::io::RobotIo::read`] and the
//! safety layer's `apply`. It holds no IO handle — by construction it cannot command a
//! motor, only propose targets.
//!
//! The tick, in order:
//!
//! ```text
//! skill windows ← advance / expire (roulade window, kick timer, ground-pick phase, sit↔stand rise)
//! command       ← the caller's platform-gated, smoothed base command, left unchanged here
//! net/context   ← roulade > kick > ground pick > sit/rise > stand-by-magnitude > walk
//! targets       ← selected two-file policy (`policy.py` → ONNX → `policy.py`)
//! ```
//!
//! This scheduler owns only selection and lifecycle.  It supplies an action name, optional
//! ground-pick phase and body-pose-mode state; each versioned policy.py is the sole owner of
//! command-observation encoding, action scaling, filtering and MIT gains.

use duck_control::model::NUM_JOINTS;
use duck_control::obs::Command;
use duck_control::policy::{Net, PolicyError, PolicyPaths};
use crate::custom_policy::PolicyContext;

/// The ground pick hands back at this fraction of its cycle — the prototype's cutoff.
const GROUND_PICK_END_PHASE: f64 = 0.7;

/// How long the sitstand network rises (posture flag 0) before the main policy takes over.
/// 1 s is enough on the robot — velstand owns the tail of the rise fine.
const RISE_SECS: f64 = 1.0;

/// How recently a roulade request must have arrived, at the end of a roll, for another to
/// chain. The prototype chains on "X still held at the window boundary"; here the client
/// holds the button by re-sending the request every tick, so "held" is "a request landed
/// within the last few ticks". 150 ms is seven ticks — generous against a dropped packet,
/// far too short to mistake a fresh press for a hold.
const ROULADE_CHAIN_WINDOW: f64 = 0.15;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tuning {
    /// Scales raw policy output before it becomes a joint offset. The prototype's current
    /// alpha default.
    pub action_scale: f64,
    /// The standing policy is trained to be applied whole.
    pub standing_action_scale: f64,
    /// Standing runs softer, at this fraction of the running gain. `--standing-kp-ratio`.
    pub standing_gain_ratio: f64,
    pub gain: u16,
    /// First-order low-pass on the head joints. `None` is no filtering. The alpha policies
    /// are trained with 0.5 — it must match training or transfer degrades.
    pub head_lowpass: Option<f64>,
    /// Same, for the ten leg joints. Trained with 0.7.
    pub legs_lowpass: Option<f64>,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            action_scale: 0.9,
            standing_action_scale: 1.0,
            standing_gain_ratio: 0.8,
            gain: 200,
            head_lowpass: Some(0.5),
            legs_lowpass: Some(0.7),
        }
    }
}

/// The scripted-skill numbers, resolved per mode by `params`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkillTuning {
    /// One ground-pick cycle, seconds.
    pub ground_pick_period: f64,
    pub ground_pick_action_scale: f64,
    /// Gain multiplier while the pick runs.
    pub ground_pick_gain_ratio: f64,
    /// How long a kick window stays on the kick network, seconds.
    pub kick_duration: f64,
    /// One roulade — one forward roll, seconds. The prototype's measured single-roll time.
    pub roulade_duration: f64,
    /// Action scale while a roulade runs.
    pub roulade_action_scale: f64,
    /// Gain multiplier while a roulade runs.
    pub roulade_gain_ratio: f64,
}

impl Default for SkillTuning {
    fn default() -> Self {
        Self {
            ground_pick_period: 4.0,
            ground_pick_action_scale: 1.0,
            ground_pick_gain_ratio: 1.0,
            kick_duration: 0.5,
            roulade_duration: 1.0,
            roulade_action_scale: 1.0,
            roulade_gain_ratio: 1.0,
        }
    }
}

/// One tick's worth of decisions, for the caller to act on and report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Step {
    pub targets: [f64; NUM_JOINTS],
    /// Which network drove, as the wire label: `walk`, `stand`, `ground_pick`, `kick_left`,
    /// `kick_right`, `sit`, `rise`.
    pub label: &'static str,
    /// What the gain should be for this tick.
    pub gain: u16,
    pub pd: Option<[f64; 2]>,
    pub mit: Option<[duck_control::MitTarget; NUM_JOINTS]>,
    /// A scripted move is mid-flight — the robot is moving regardless of the twist, so
    /// restarting the daemon now would put it on the floor.
    pub busy: bool,
    /// Platform-gated command before any model-specific observation encoding.
    pub base_command: Command,
    /// Scheduler metadata consumed by policy.py when constructing its model inputs.
    pub policy_context: PolicyContext,
}

/// Where the robot is in the sit↔stand cycle.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Sit {
    Up,
    /// The sitstand network holds the seat (posture flag 1).
    Sitting,
    /// The sitstand network rises (posture flag 0) for the remaining seconds, then the
    /// main policy takes over.
    Rising {
        remaining: f64,
    },
}

/// Entry is committed only after a successful policy frame. Labels distinguish sit/rise,
/// while pending starts cover a new episode that reuses the same slot (including roll chains).
#[derive(Default)]
struct PolicyEntry {
    active: Option<(Net, &'static str)>,
    pending: Vec<Net>,
}

impl PolicyEntry {
    fn restart(&mut self, net: Net) {
        if !self.pending.contains(&net) {
            self.pending.push(net);
        }
    }

    fn needs_reset(&self, net: Net, label: &'static str) -> bool {
        !matches!(net, Net::Walk | Net::Stand)
            && (self.active != Some((net, label)) || self.pending.contains(&net))
    }

    fn entered(&mut self, net: Net, label: &'static str) {
        self.active = Some((net, label));
        self.pending.retain(|candidate| *candidate != net);
    }
}

pub struct Controller {
    custom_policies: Vec<(Net, crate::custom_policy::CustomPolicy)>,
    /// One consumer instance owns feedback/filter history; only its ONNX session switches.
    locomotion: Option<crate::custom_policy::CustomPolicy>,
    entry: PolicyEntry,
    has_standing: bool,
    has_sitstand: bool,
    has_ground_pick: bool,
    has_kick_left: bool,
    has_kick_right: bool,
    has_roulade: bool,
    standing_threshold: f64,
    standing_disabled: bool,
    tuning: Tuning,
    skills: SkillTuning,
    /// Ground-pick phase, 0..[`GROUND_PICK_END_PHASE`]. `None` when inactive.
    ground_pick: Option<f64>,
    /// An active kick window: which leg, and seconds remaining.
    kick: Option<(bool, f64)>,
    /// An active roulade: seconds remaining in the current roll.
    roulade: Option<f64>,
    /// Seconds since a roulade request would still count as "the button is held" — counts
    /// down every tick, refreshed by each request that arrives while a roll runs. At the
    /// end of a roll, positive means chain another.
    roulade_chain: f64,
    sit: Sit,
}

impl Controller {
    pub fn new(paths: &PolicyPaths, standing_threshold: f64, tuning: Tuning, skills: SkillTuning) -> Self {
        Self {
            custom_policies: Vec::new(),
            locomotion: None,
            entry: PolicyEntry::default(),
            has_standing: paths.stand.is_some(),
            has_sitstand: paths.sitstand.is_some(),
            has_ground_pick: paths.ground_pick.is_some(),
            has_kick_left: paths.kick_left.is_some(),
            has_kick_right: paths.kick_right.is_some(),
            has_roulade: paths.roulade.is_some(),
            standing_threshold,
            standing_disabled: false,
            tuning,
            skills,
            ground_pick: None,
            kick: None,
            roulade: None,
            roulade_chain: 0.0,
            sit: Sit::Up,
        }
    }

    pub fn replace_custom_policy(
        &mut self,
        net: Net,
        mut policy: crate::custom_policy::CustomPolicy,
    ) {
        policy.mark_reset();
        self.custom_policies.retain(|(candidate, _)| *candidate != net);
        self.custom_policies.push((net, policy));
        self.reset();
    }

    pub fn replace_locomotion_policy(&mut self, policy: crate::custom_policy::CustomPolicy) {
        self.custom_policies.retain(|(net, _)| !matches!(net, Net::Walk | Net::Stand));
        self.locomotion = Some(policy);
        self.reset();
    }

    pub fn reset(&mut self) {
        self.entry = PolicyEntry::default();
        if let Some(policy) = &mut self.locomotion { policy.mark_reset(); }
        for (_, policy) in &mut self.custom_policies { policy.mark_reset(); }
    }

    pub fn set_standing_disabled(&mut self, disabled: bool) {
        self.standing_disabled = disabled;
    }

    fn will_stand(&self, twist_magnitude: f64) -> bool {
        self.has_standing && !self.standing_disabled && twist_magnitude <= self.standing_threshold
    }

    /// Start a policy experiment from a plain locomotion state. A previously interrupted skill
    /// must not silently choose another network than the experiment requested.
    pub fn reset_for_policy_experiment(&mut self) {
        self.ground_pick = None;
        self.kick = None;
        self.roulade = None;
        self.roulade_chain = 0.0;
        self.sit = Sit::Up;
        self.reset();
    }

    pub fn has_sitstand(&self) -> bool {
        self.has_sitstand
    }

    pub fn is_sitting(&self) -> bool {
        self.sit == Sit::Sitting
    }

    pub fn is_rising(&self) -> bool {
        matches!(self.sit, Sit::Rising { .. })
    }

    pub fn cancel_rise(&mut self) {
        if self.is_rising() {
            self.sit = Sit::Up;
        }
    }

    /// A scripted move is mid-flight. Sitting itself is not busy — a seated robot is
    /// parked, not travelling.
    pub fn busy(&self) -> bool {
        self.ground_pick.is_some()
            || self.kick.is_some()
            || self.roulade.is_some()
            || matches!(self.sit, Sit::Rising { .. })
    }

    /// Start a one-shot ground pick. The prototype gates the trigger on nothing but the
    /// network existing and the move not already running — a pick can even preempt a kick's
    /// tail, and that stays as it was.
    pub fn start_ground_pick(&mut self) -> Result<(), &'static str> {
        if !self.has_ground_pick {
            return Err("no ground-pick policy loaded");
        }
        if self.ground_pick.is_some() {
            return Err("ground pick already running");
        }
        self.ground_pick = Some(0.0);
        self.entry.restart(Net::GroundPick);
        Ok(())
    }

    /// Start a kick window. Blocked while any scripted move runs, as the prototype blocks it.
    pub fn start_kick(&mut self, left: bool) -> Result<(), &'static str> {
        if !(if left { self.has_kick_left } else { self.has_kick_right }) {
            return Err("no kick policy loaded for that leg");
        }
        if self.kick.is_some() || self.ground_pick.is_some() || self.roulade.is_some() {
            return Err("a scripted move is already running");
        }
        self.kick = Some((left, self.skills.kick_duration));
        self.entry.restart(if left { Net::KickLeft } else { Net::KickRight });
        Ok(())
    }

    /// A roulade request: start a roll, or — arriving while one runs — keep the chain alive.
    ///
    /// `Ok(true)` started a roll; `Ok(false)` refreshed a running one (the caller should
    /// stay quiet: a held button lands here fifty times a second). The prototype gates the
    /// X press on nothing but the ground pick — a roulade can preempt a kick's tail or roll
    /// out of the seat, and both stay as they were.
    pub fn request_roulade(&mut self) -> Result<bool, &'static str> {
        if self.roulade.is_some() {
            self.roulade_chain = ROULADE_CHAIN_WINDOW;
            return Ok(false);
        }
        if !self.has_roulade {
            return Err("no roulade policy loaded");
        }
        if self.ground_pick.is_some() {
            return Err("a ground pick is running");
        }
        self.roulade = Some(self.skills.roulade_duration);
        self.entry.restart(Net::Roulade);
        self.roulade_chain = 0.0;
        Ok(true)
    }

    /// Sit if standing, stand if sitting. Refused mid-rise, as the prototype refuses it
    /// while a stand transition is in flight.
    pub fn sit_toggle(&mut self) -> Result<&'static str, &'static str> {
        match self.sit {
            Sit::Up => {
                if !self.has_sitstand {
                    return Err("no sitstand policy loaded");
                }
                self.sit = Sit::Sitting;
                self.entry.restart(Net::SitStand);
                Ok("sit")
            }
            Sit::Sitting => {
                self.entry.restart(Net::SitStand);
                self.sit = Sit::Rising {
                    remaining: RISE_SECS,
                };
                Ok("stand")
            }
            Sit::Rising { .. } => Err("already standing up"),
        }
    }

    /// Engage the sit for the shutdown sequence. The caller owns the timing (sit for a few
    /// seconds, then cut torque and power off); this just puts the sitstand network in
    /// charge with the posture flag at 1.
    pub fn begin_shutdown_sit(&mut self) {
        self.entry.restart(Net::SitStand);
        self.sit = Sit::Sitting;
    }

    /// Seated boot: the robot powered on already sitting, so rise via the sitstand network
    /// instead of dragging the legs through a linear ramp to the standing pose.
    pub fn begin_boot_rise(&mut self) {
        self.entry.restart(Net::SitStand);
        self.sit = Sit::Rising {
            remaining: RISE_SECS,
        };
    }

    /// Run the seated-rise network once while torque is still off.
    ///
    /// ONNX Runtime can fault model pages or finish provider setup on the first inference
    /// after a long idle period. Doing that after torque-on would leave STM32 without a host
    /// command long enough to trip its independent 100 ms watchdog. The result is discarded;
    /// reset restores the exact state from which the real rise starts.
    pub fn prepare_boot_rise(
        &mut self,
        sensors: &duck_control::Sensors,
        dt: f64,
    ) -> Result<(), PolicyError> {
        self.begin_boot_rise();
        let result = self.step(sensors, &Command::default(), false, dt, 1.0);
        self.cancel_rise();
        self.reset();
        result.map(|_| {
            self.begin_boot_rise();
        })
    }

    /// One tick.
    ///
    /// `body_active` says a client is holding the body-pose mode: the twist is zeroed and
    /// the standing network drives (by magnitude where it is selectable, forced where it is
    /// reserved), exactly as the prototype's B-button mode behaves.
    ///
    /// `scale_mult` remains in the controller API for voltage telemetry compatibility. In the
    /// unified two-file path, action scaling belongs to `policy.py`.
    pub fn step(
        &mut self,
        sensors: &duck_control::Sensors,
        command: &Command,
        body_active: bool,
        dt: f64,
        _scale_mult: f64,
    ) -> Result<Step, PolicyError> {
        self.step_selected(sensors, command, body_active, dt, _scale_mult, None)
    }

    /// Run a normal tick while optionally pinning the locomotion network. Policy experiments
    /// use this to test the walking or standing slot explicitly; ordinary control passes None.
    pub fn step_selected(
        &mut self,
        sensors: &duck_control::Sensors,
        command: &Command,
        body_active: bool,
        dt: f64,
        _scale_mult: f64,
        selected: Option<Net>,
    ) -> Result<Step, PolicyError> {
        // Expire windows first, so a tick after the deadline runs the next thing rather
        // than one more frame of a finished move — the prototype checks its timers at the
        // same point relative to inference.
        if let Some((_, remaining)) = self.kick
            && remaining <= 0.0
        {
            self.kick = None;
        }
        // The end of a roll is a fork, as the prototype forks it: the button still held
        // (a request landed within the chain window) restarts the window — the policy
        // re-initiates a roll from wherever it landed — and released hands back.
        if let Some(remaining) = self.roulade
            && remaining <= 0.0
        {
            self.roulade = if self.roulade_chain > 0.0 {
                self.entry.restart(Net::Roulade);
                Some(self.skills.roulade_duration)
            } else {
                None
            };
        }
        if let Sit::Rising { remaining } = self.sit
            && remaining <= 0.0
        {
            self.sit = Sit::Up;
        }

        // Pick the network and expose scheduler state without encoding a model observation.
        // The selected policy.py is the sole owner of how base command + context become tensors.
        // Priority remains the prototype's: roulade > kick > ground pick > sit/rise > stand > walk.
        let (net, label, context) = if self.roulade.is_some() {
            (Net::Roulade, "roulade", PolicyContext::new("roulade", None, false))
        } else if let Some((left, _)) = self.kick {
            let net = if left { Net::KickLeft } else { Net::KickRight };
            let label = if left { "kick_left" } else { "kick_right" };
            (net, label, PolicyContext::new(label, None, false))
        } else if let Some(phase) = self.ground_pick {
            (Net::GroundPick, "ground_pick",
                PolicyContext::new("ground_pick", Some(phase), false))
        } else {
            match self.sit {
                Sit::Sitting => (Net::SitStand, "sit",
                    PolicyContext::new("sit", None, false)),
                Sit::Rising { .. } => (Net::SitStand, "rise",
                    PolicyContext::new("rise", None, false)),
                Sit::Up => {
                    let (net, label) = match selected {
                        Some(Net::Walk) => (Net::Walk, "walk"),
                        Some(Net::Stand) => (Net::Stand, "stand"),
                        Some(_) => unreachable!("policy experiment only selects locomotion nets"),
                        None => {
                            let standing = self.will_stand(command.twist_magnitude())
                                || (body_active && self.has_standing);
                            if standing { (Net::Stand, "stand") }
                            else { (Net::Walk, "walk") }
                        }
                    };
                    (net, label, PolicyContext::new(label, None, body_active))
                }
            }
        };

        let output = if matches!(net, Net::Walk | Net::Stand) && self.has_standing {
            let model = if net == Net::Walk { "walk" } else { "stand" };
            self.locomotion.as_mut()
                .ok_or_else(|| PolicyError::Inference("走/站缺少共享消费运行器".into()))?
                .step_model(sensors, command, context, dt, model).map_err(PolicyError::Inference)?
        } else {
        let custom = self.custom_policies.iter_mut()
            .find(|(candidate, _)| *candidate == net)
            .map(|(_, policy)| policy)
            .ok_or_else(|| PolicyError::Inference(format!("策略槽位 {net:?} 缺少双文件运行器")))?;
        // mark_reset is consumed inside step before cached-output/rate-limit handling,
        // so the first frame can never reuse targets from the previous episode.
        if self.entry.needs_reset(net, label) {
            custom.mark_reset();
        }
        custom.step(sensors, command, context, dt).map_err(PolicyError::Inference)?
        };
        self.entry.entered(net, label);
        let (targets, gain, pd, mit) =
            (output.positions, self.tuning.gain, None, Some(output.mit));

        // Advance the windows, after the tick that used them — the prototype advances its
        // phase after the motor write.
        if let Some(phase) = self.ground_pick.as_mut() {
            *phase += dt / self.skills.ground_pick_period;
            if *phase >= GROUND_PICK_END_PHASE {
                self.ground_pick = None;
            }
        }
        if let Some((_, remaining)) = self.kick.as_mut() {
            *remaining -= dt;
        }
        if let Some(remaining) = self.roulade.as_mut() {
            *remaining -= dt;
            self.roulade_chain = (self.roulade_chain - dt).max(0.0);
        }
        if let Sit::Rising { remaining } = &mut self.sit {
            *remaining -= dt;
        }

        Ok(Step {
            targets,
            label,
            gain,
            pd,
            mit,
            busy: self.busy(),
            base_command: *command,
            policy_context: context,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locomotion_preserves_state_but_every_skill_resets_on_reentry() {
        let mut entry = PolicyEntry::default();
        for (net, label) in [(Net::Walk, "walk"), (Net::Stand, "stand"),
            (Net::SitStand, "sit"), (Net::GroundPick, "ground_pick"),
            (Net::KickLeft, "kick_left"), (Net::KickRight, "kick_right"),
            (Net::Roulade, "roulade")]
        {
            let skill = !matches!(net, Net::Walk | Net::Stand);
            assert_eq!(entry.needs_reset(net, label), skill);
            // A failed first frame does not commit entry: retry must reset again.
            assert_eq!(entry.needs_reset(net, label), skill);
            entry.entered(net, label);
            assert!(!entry.needs_reset(net, label));
            entry.entered(Net::Walk, "walk");
            assert_eq!(entry.needs_reset(net, label), skill);
            entry.entered(net, label);
        }
        assert!(!entry.needs_reset(Net::Walk, "walk"));
        assert!(!entry.needs_reset(Net::Stand, "stand"));
    }

    #[test]
    fn sit_to_rise_and_same_slot_new_episodes_reset_once() {
        let mut entry = PolicyEntry::default();
        entry.entered(Net::SitStand, "sit");
        assert!(entry.needs_reset(Net::SitStand, "rise"));
        entry.entered(Net::SitStand, "rise");
        assert!(!entry.needs_reset(Net::SitStand, "rise"));
        for net in [Net::KickLeft, Net::KickRight, Net::GroundPick, Net::Roulade] {
            entry.entered(net, "skill");
            entry.restart(net);
            assert!(entry.needs_reset(net, "skill"));
            entry.entered(net, "skill");
            assert!(!entry.needs_reset(net, "skill"));
        }
    }

    fn skill_controller() -> Controller {
        let paths = PolicyPaths {
            stand: Some("stand".into()), sitstand: Some("sitstand".into()),
            ground_pick: Some("pick".into()), kick_left: Some("left".into()),
            kick_right: Some("right".into()), roulade: Some("roll".into()),
            ..Default::default()
        };
        Controller::new(&paths, 0.05, Tuning::default(), SkillTuning::default())
    }

    #[test]
    #[ignore = "requires root, bubblewrap and managed Python on the build board; no motor IO"]
    fn locomotion_dispatch_matches_one_continuous_consumer() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let policy_path = root.join("src/default_locomotion_policy.py");
        let walk = root.join("../policies/alpha_walking.onnx");
        let stand = root.join("../policies/alpha_stand.onnx");
        let paths = PolicyPaths { walk: walk.clone(), stand: Some(stand.clone()), ..Default::default() };
        let mut controller = Controller::new(&paths, 0.05, Tuning::default(), SkillTuning::default());
        controller.replace_locomotion_policy(crate::custom_policy::CustomPolicy::load_pair(
            &policy_path, &walk, &stand).unwrap());
        let mut reference = crate::custom_policy::CustomPolicy::load_pair(&policy_path, &walk, &stand).unwrap();
        let sensors = duck_control::Sensors::default();
        for index in 0..40 {
            let walking = index % 3 != 0;
            let command = Command { twist: if walking { [0.3, 0.0, 0.0] } else { [0.0; 3] },
                ..Command::default() };
            let model = if walking { "walk" } else { "stand" };
            let context = PolicyContext::new(model, None, false);
            let expected = reference.step_model(&sensors, &command, context, 0.02, model).unwrap();
            let actual = controller.step(&sensors, &command, false, 0.02, 1.0).unwrap();
            assert_eq!(actual.label, model);
            assert_eq!(actual.targets, expected.positions, "feedback/filter diverged at switch {index}");
        }
        controller.reset();
        reference.mark_reset();
        assert_eq!(controller.step(&sensors, &Command::default(), false, 0.02, 1.0).unwrap().targets,
            reference.step_model(&sensors, &Command::default(),
                PolicyContext::new("stand", None, false), 0.02, "stand").unwrap().positions);
    }

    #[test]
    fn chained_roll_queues_entry_before_dispatch_and_failure_keeps_it_pending() {
        let mut c = skill_controller();
        c.entry.entered(Net::Roulade, "roulade");
        c.roulade = Some(0.0);
        c.roulade_chain = ROULADE_CHAIN_WINDOW;
        // No worker (and no IO handle): fail at dispatch after advancing the scheduler.
        assert!(c.step(&duck_control::Sensors::default(), &Command::default(),
            false, 0.02, 1.0).is_err());
        assert_eq!(c.roulade, Some(c.skills.roulade_duration));
        assert!(c.entry.needs_reset(Net::Roulade, "roulade"));
        assert!(c.step(&duck_control::Sensors::default(), &Command::default(),
            false, 0.02, 1.0).is_err());
        assert!(c.entry.needs_reset(Net::Roulade, "roulade"));
    }

    #[test]
    fn accepted_starts_queue_reset_but_held_or_rejected_requests_do_not() {
        let mut c = skill_controller();
        c.start_ground_pick().unwrap();
        assert!(c.entry.pending.contains(&Net::GroundPick));
        c.entry.entered(Net::GroundPick, "ground_pick");
        assert!(c.start_ground_pick().is_err());
        assert!(c.entry.pending.is_empty());
        c.ground_pick = None;
        for left in [true, false] {
            let net = if left { Net::KickLeft } else { Net::KickRight };
            c.start_kick(left).unwrap();
            assert!(c.entry.pending.contains(&net));
            c.entry.entered(net, "kick");
            assert!(c.start_kick(left).is_err());
            assert!(c.entry.pending.is_empty());
            c.kick = None;
        }
        assert!(c.request_roulade().unwrap());
        assert!(c.entry.pending.contains(&Net::Roulade));
        c.entry.entered(Net::Roulade, "roulade");
        assert!(!c.request_roulade().unwrap());
        assert!(c.entry.pending.is_empty());
        c.sit_toggle().unwrap();
        assert!(c.entry.pending.contains(&Net::SitStand));
        // A queued lower-priority skill is not consumed by another slot's execution.
        c.entry.entered(Net::Roulade, "roulade");
        assert!(c.entry.pending.contains(&Net::SitStand));
        c.entry.entered(Net::SitStand, "sit");
        c.sit_toggle().unwrap();
        assert!(c.entry.needs_reset(Net::SitStand, "rise"));
        c.entry.entered(Net::SitStand, "rise");
        assert!(c.sit_toggle().is_err());
        assert!(c.entry.pending.is_empty());
        c.begin_shutdown_sit();
        assert!(c.entry.pending.contains(&Net::SitStand));
        c.entry.entered(Net::SitStand, "sit");
        c.begin_boot_rise();
        assert!(c.entry.pending.contains(&Net::SitStand));
    }

    /// The prototype's **current alpha configuration** — its built-in defaults, which the
    /// installer deliberately passes no flags to override. The filters are ON at the values
    /// the policies are trained with; changing any of these silently changes how the robot
    /// moves relative to the thing it replaces.
    #[test]
    fn the_defaults_match_the_prototype() {
        let t = Tuning::default();
        assert_eq!(t.action_scale, 0.9);
        assert_eq!(t.standing_action_scale, 1.0);
        assert_eq!(t.standing_gain_ratio, 0.8);
        assert_eq!(
            t.head_lowpass,
            Some(0.5),
            "trained with ACTION_LOW_PASS_HEAD_ALPHA"
        );
        assert_eq!(
            t.legs_lowpass,
            Some(0.7),
            "trained with ACTION_LOW_PASS_LEG_ALPHA"
        );

        let s = SkillTuning::default();
        assert_eq!(s.ground_pick_period, 4.0);
        assert_eq!(s.ground_pick_action_scale, 1.0);
        assert_eq!(s.ground_pick_gain_ratio, 1.0);
        assert_eq!(s.kick_duration, 0.5);
        assert_eq!(s.roulade_duration, 1.0, "one roll, the measured time");
        assert_eq!(s.roulade_action_scale, 1.0);
        assert_eq!(
            s.roulade_gain_ratio, 1.0,
            "a roll runs at full walking gain"
        );
    }

    /// Standing must drop the gain. Running the standing policy at walking stiffness is a
    /// visibly different robot, and the ratio is the prototype's.
    #[test]
    fn standing_softens_the_gain() {
        let t = Tuning::default();
        let standing_gain = (t.gain as f64 * t.standing_gain_ratio).round() as u16;
        assert_eq!(standing_gain, 160);
        assert!(standing_gain < t.gain);
    }

    /// The ground pick ends at 70% of its cycle — ending at 100% replays the reach on the
    /// way out, which is the prototype bug the 0.7 cutoff fixed there.
    #[test]
    fn the_ground_pick_cutoff_is_the_prototypes() {
        assert_eq!(GROUND_PICK_END_PHASE, 0.7);
        assert_eq!(RISE_SECS, 1.0);
    }
}
