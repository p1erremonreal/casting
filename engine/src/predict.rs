//! Hand tracks: denoising + short-range prediction.
//!
//! The camera delivers ~30 noisy observations per second; the engine renders
//! at 60 Hz. Each hand is tracked across observations (matched by wrist
//! distance and handedness). Every landmark goes through a One Euro filter:
//! heavy smoothing while the hand is still (where jitter is most visible),
//! light smoothing when it moves fast (so it doesn't lag). The filter's
//! smoothed velocity drives a short, speed-gated extrapolation between frames.

use std::time::Instant;

pub type V3 = [f64; 3];

/// Landmarks we track per hand.
pub const N: usize = 9;
pub const TIPS: std::ops::Range<usize> = 0..5; // thumb, index, middle, ring, pinky
pub const WRIST: usize = 5;
pub const INDEX_MCP: usize = 6;
pub const MIDDLE_MCP: usize = 7;
pub const PINKY_MCP: usize = 8;
pub const INDEX_TIP: usize = 1;

/// Never extrapolate further than this past the last observation (seconds).
const MAX_LEAD: f64 = 0.04;
/// Below this landmark speed (frame-heights/s) there's no extrapolation at all;
/// full extrapolation above SPEED_FULL. Stops prediction from amplifying jitter.
const SPEED_GATE: f64 = 0.12;
const SPEED_FULL: f64 = 0.5;
/// Drop a track the camera hasn't confirmed for this long (seconds). Unmatched
/// tracks survive brief detection dropouts instead of being recreated.
const STALE: f64 = 0.3;

// One Euro filter parameters (x, y in normalized image units; z is
// MediaPipe's relative depth, much noisier, so it gets a lower base cutoff).
const MIN_CUTOFF_XY: f64 = 1.2; // Hz while still
const MIN_CUTOFF_Z: f64 = 0.3;
const BETA: f64 = 6.0; // how fast the cutoff opens up with speed
const D_CUTOFF: f64 = 1.5; // Hz, for the velocity estimate

fn alpha(cutoff: f64, dt: f64) -> f64 {
    let tau = 1.0 / (2.0 * std::f64::consts::PI * cutoff);
    1.0 / (1.0 + tau / dt)
}

#[derive(Clone)]
pub struct Observation {
    pub label: String,
    pub pts: [V3; N],
}

pub struct Track {
    /// Filtered landmark positions at the last observation.
    obs: [V3; N],
    /// Filtered landmark velocities (units/s).
    vel: [V3; N],
    /// Tracker timestamp of the last observation.
    t_obs: f64,
    /// Wall-clock time the last observation arrived.
    rx: Instant,
    /// Handedness votes over the track's whole life.
    votes_left: u32,
    votes_right: u32,
    /// Set once the hand has been raised; stays set while tracked.
    pub active: bool,
}

impl Track {
    fn new(o: Observation, t: f64, rx: Instant) -> Self {
        let mut tr = Self {
            obs: o.pts,
            vel: [[0.0; 3]; N],
            t_obs: t,
            rx,
            votes_left: 0,
            votes_right: 0,
            active: false,
        };
        tr.vote(&o.label);
        tr
    }

    fn vote(&mut self, label: &str) {
        match label {
            "Left" => self.votes_left += 1,
            "Right" => self.votes_right += 1,
            _ => {}
        }
    }

    /// Majority handedness. Momentary Left/Right flips (common with palms up)
    /// can't overturn a hand's history, so the palm side never flips with them.
    pub fn label(&self) -> &'static str {
        if self.votes_left >= self.votes_right { "Left" } else { "Right" }
    }

    /// Run one observation through the One Euro filters.
    fn update(&mut self, o: Observation, t: f64, rx: Instant) {
        let dt = t - self.t_obs;
        if dt > 1e-4 && dt < 0.5 {
            for j in 0..N {
                // Velocity estimate (smoothed), then a speed-dependent cutoff.
                let mut dv = [0.0; 3];
                for k in 0..3 {
                    let raw = (o.pts[j][k] - self.obs[j][k]) / dt;
                    dv[k] = self.vel[j][k] + (raw - self.vel[j][k]) * alpha(D_CUTOFF, dt);
                }
                let speed = dv[0].hypot(dv[1]);
                for k in 0..3 {
                    let base = if k == 2 { MIN_CUTOFF_Z } else { MIN_CUTOFF_XY };
                    let a = alpha(base + BETA * speed, dt);
                    self.obs[j][k] += (o.pts[j][k] - self.obs[j][k]) * a;
                }
                self.vel[j] = dv;
            }
        } else {
            // Long gap: restart from the raw observation.
            self.obs = o.pts;
            self.vel = [[0.0; 3]; N];
        }
        self.t_obs = t;
        self.rx = rx;
        self.vote(&o.label);
    }

    /// Landmarks extrapolated to `now` (only when the hand is clearly moving).
    pub fn predict(&self, now: Instant) -> [V3; N] {
        let h = now.saturating_duration_since(self.rx).as_secs_f64().min(MAX_LEAD);
        let mut out = self.obs;
        for (p, v) in out.iter_mut().zip(self.vel.iter()) {
            let speed = v[0].hypot(v[1]);
            let gate = ((speed - SPEED_GATE) / (SPEED_FULL - SPEED_GATE)).clamp(0.0, 1.0);
            for k in 0..2 {
                p[k] += v[k] * h * gate;
            }
            // No depth extrapolation: z velocity is mostly noise.
        }
        out
    }
}

#[derive(Default)]
pub struct Tracker {
    pub tracks: Vec<Track>,
}

fn dist2(a: V3, b: V3) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

impl Tracker {
    /// Feed one camera frame's hands (tracker time `t`, received at `rx`).
    pub fn observe(&mut self, t: f64, rx: Instant, hands: Vec<Observation>) {
        let mut taken = vec![false; self.tracks.len()];
        let mut fresh = Vec::new();
        for hand in hands {
            // Nearest unclaimed track by wrist, preferring the same handedness.
            let best = self
                .tracks
                .iter()
                .enumerate()
                .filter(|(i, _)| !taken[*i])
                .map(|(i, tr)| {
                    let penalty = if tr.label() == hand.label { 0.0 } else { 0.01 };
                    (i, dist2(tr.obs[WRIST], hand.pts[WRIST]) + penalty)
                })
                .filter(|&(_, d)| d < 0.2 * 0.2)
                .min_by(|a, b| a.1.total_cmp(&b.1));
            match best {
                Some((i, _)) => {
                    taken[i] = true;
                    self.tracks[i].update(hand, t, rx);
                }
                None => fresh.push(Track::new(hand, t, rx)),
            }
        }
        // Unmatched tracks are kept until they go stale (brief detection dropouts).
        self.tracks.extend(fresh);
    }

    /// Drop tracks the camera hasn't confirmed recently.
    pub fn expire(&mut self, now: Instant) {
        self.tracks
            .retain(|tr| now.saturating_duration_since(tr.rx).as_secs_f64() < STALE);
    }
}
