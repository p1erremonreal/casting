//! Tesseract engine.
//!
//! A reader thread takes hand observations from the Python tracker (one JSON
//! line per camera frame, ~30/s) off stdin. The main loop runs at a fixed
//! 60 Hz: it extrapolates every hand landmark to the current instant, derives
//! each hand's 3D orientation, and writes one JSON line per tick describing
//! the projected tesseract in normalized screen coordinates (0..1, top-left).
//!
//! The tesseract never rotates on its own:
//! - one hand: it takes on that hand's 3D orientation;
//! - two hands: it sits between them in their average orientation, and the
//!   relative rotation between the hands turns it through the 4th dimension.

mod orient;
mod predict;
mod tesseract;

use orient::{M3, V3};
use predict::{Observation, Track, Tracker};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::io::{self, BufRead, Write};
use std::sync::mpsc::{self, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};
use tesseract::Tesseract;

const TICK_HZ: f64 = 60.0;

// ---- input -------------------------------------------------------------------

#[derive(Deserialize)]
struct Hand {
    tips: [V3; 5],
    wrist: V3,
    index_mcp: V3,
    middle_mcp: V3,
    pinky_mcp: V3,
    #[serde(default)]
    handedness: String,
}

#[derive(Deserialize)]
struct Input {
    /// Seconds since tracker start (camera capture time).
    t: f64,
    /// Frame width / height.
    aspect: f64,
    hands: Vec<Hand>,
}

impl Hand {
    fn into_observation(self) -> Observation {
        let [a, b, c, d, e] = self.tips;
        Observation {
            label: self.handedness,
            pts: [a, b, c, d, e, self.wrist, self.index_mcp, self.middle_mcp, self.pinky_mcp],
        }
    }
}

// ---- output ------------------------------------------------------------------

#[derive(Serialize)]
struct OutHand {
    /// Predicted fingertips [x, y] (normalized image coords).
    tips: Vec<[f64; 2]>,
    /// Convex outline of the hand (normalized image coords).
    outline: Vec<[f64; 2]>,
    /// "palm" | "back" | "side": which side of the hand faces the camera.
    facing: &'static str,
    /// 0..1: how much this hand hides the tesseract (it sits behind the hand
    /// when the back of the hand faces the camera).
    occlude: f64,
    /// 0 (fist) .. 1 (fully open, fingers spread).
    openness: f64,
}

#[derive(Serialize)]
struct Output {
    /// 0..1 fade for the whole figure.
    alpha: f64,
    mode: &'static str,
    /// [x1, y1, x2, y2, depth] where depth is 0 (far in 4D) .. 1 (near in 4D).
    edges: Vec<[f64; 5]>,
    /// Projected vertices [x, y, depth].
    vertices: Vec<[f64; 3]>,
    /// Hands at the predicted (60 Hz) positions.
    hands: Vec<OutHand>,
    /// Placement (normalized image coords) and scale, for diagnostics.
    center: [f64; 2],
    size: f64,
}

// ---- geometry helpers ----------------------------------------------------------

/// Image point -> camera space (x right, y up, z toward viewer; height units).
fn to_space(p: V3, aspect: f64) -> V3 {
    [p[0] * aspect, -p[1], -p[2] * aspect]
}

fn len2(a: [f64; 2]) -> f64 {
    a[0].hypot(a[1])
}

fn wrap_angle(a: f64) -> f64 {
    let a = (a + PI).rem_euclid(2.0 * PI) - PI;
    if a.is_nan() {
        0.0
    } else {
        a
    }
}

/// A hand's orientation relative to the neutral pose (palm toward the camera,
/// fingers up): columns are its across-the-palm, up-the-fingers, and palm-
/// normal axes in camera space.
fn hand_rotation(pts: &[V3; predict::N], label: &str, aspect: f64) -> Option<M3> {
    let s = |i: usize| to_space(pts[i], aspect);
    let up = orient::sub(s(predict::MIDDLE_MCP), s(predict::WRIST));
    let across = orient::sub(s(predict::PINKY_MCP), s(predict::INDEX_MCP));
    // Make the palm normal point out of the palm for both hands. With our
    // mirrored frames, a palm-forward hand MediaPipe labels "Left" has its
    // pinky to the right of the index knuckle, and "Right" the reverse
    // (confirmed on the live camera).
    let across = if label == "Right" { orient::scale(across, -1.0) } else { across };
    orient::from_axes(across, up)
}

/// Index fingertip clearly above the wrist.
fn is_raised(pts: &[V3; predict::N]) -> bool {
    let w = pts[predict::WRIST];
    let m = pts[predict::MIDDLE_MCP];
    let scale = (w[0] - m[0]).hypot(w[1] - m[1]).max(1e-3);
    w[1] - pts[predict::INDEX_TIP][1] > scale * 0.8
}

/// Palm center (image coords): average of wrist and the three knuckles.
fn palm(pts: &[V3; predict::N]) -> [f64; 2] {
    let ids = [predict::WRIST, predict::INDEX_MCP, predict::MIDDLE_MCP, predict::PINKY_MCP];
    let n = ids.len() as f64;
    [
        ids.iter().map(|&i| pts[i][0]).sum::<f64>() / n,
        ids.iter().map(|&i| pts[i][1]).sum::<f64>() / n,
    ]
}

/// Hand scale in aspect space (≈ wrist-to-middle-knuckle length), from 2D
/// measurements only: MediaPipe's depth is too noisy to size things with.
/// Palm up / pointing at the camera foreshortens the length but not the
/// knuckle width; edge-on does the reverse, so take whichever is larger.
fn hand_len(pts: &[V3; predict::N], aspect: f64) -> f64 {
    let d2 = |a: usize, b: usize| len2([(pts[a][0] - pts[b][0]) * aspect, pts[a][1] - pts[b][1]]);
    // Knuckle width (index->pinky MCP) is ~1/1.45 of the palm length; slightly
    // conservative so the length wins whenever it's actually visible.
    d2(predict::WRIST, predict::MIDDLE_MCP)
        .max(d2(predict::INDEX_MCP, predict::PINKY_MCP) * 1.45)
        .max(1e-3)
}

fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// How open the hand is: 0 = fist, 1 = fingers fully extended and spread.
///
/// Mostly finger spread: the distances between neighbouring fingertips
/// (thumb-index, index-middle, middle-ring, ring-pinky). Spreading the fingers
/// widens every gap; a fist bunches the tips together. Measured on screen (2D,
/// no noisy depth) relative to the hand's own size, so it doesn't change with
/// distance from the camera. A smaller share of fingertip reach from the palm
/// separates a fist from straight fingers held together.
fn openness(pts: &[V3; predict::N], aspect: f64) -> f64 {
    let p = |i: usize| [pts[i][0] * aspect, pts[i][1]];
    let d = |a: [f64; 2], b: [f64; 2]| len2([a[0] - b[0], a[1] - b[1]]);
    let scale = hand_len(pts, aspect);

    // Finger spread: gaps index-middle, middle-ring, ring-pinky (tips are
    // thumb, index, middle, ring, pinky). ~0.55 held together, ~1.5 spread.
    let fingers = (1..4).map(|i| d(p(i), p(i + 1))).sum::<f64>() / scale;
    // Thumb spread: thumb tip to index tip. ~0.6 tucked, ~1.3 out.
    let thumb = d(p(0), p(1)) / scale;

    // Reach: fingertips' distance from the palm center (fist vs straight).
    let c = palm(pts);
    let c = [c[0] * aspect, c[1]];
    let reach = predict::TIPS.map(|i| d(p(i), c)).sum::<f64>() / 5.0 / scale;

    0.55 * smoothstep(0.55, 1.4, fingers)
        + 0.2 * smoothstep(0.6, 1.25, thumb)
        + 0.25 * smoothstep(0.7, 1.4, reach)
}

/// Convex hull (Andrew's monotone chain), counter-clockwise.
fn hull(mut p: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    p.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    p.dedup();
    if p.len() < 3 {
        return p;
    }
    let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let mut h: Vec<[f64; 2]> = Vec::with_capacity(p.len() * 2);
    for pass in 0..2 {
        let start = h.len();
        let iter: Box<dyn Iterator<Item = &[f64; 2]>> =
            if pass == 0 { Box::new(p.iter()) } else { Box::new(p.iter().rev()) };
        for &q in iter {
            while h.len() >= start + 2 && cross(h[h.len() - 2], h[h.len() - 1], q) <= 0.0 {
                h.pop();
            }
            h.push(q);
        }
        h.pop();
    }
    h
}

/// Everything the scene needs about one hand this tick.
struct HandInfo {
    pts: [V3; predict::N],
    rot: M3,
    open: f64,
}

impl HandInfo {
    fn new(pts: [V3; predict::N], label: &str, aspect: f64) -> Option<Self> {
        Some(Self { rot: hand_rotation(&pts, label, aspect)?, open: openness(&pts, aspect), pts })
    }
    /// Palm normal in camera space (points out of the palm).
    fn normal(&self) -> V3 {
        orient::col(&self.rot, 2)
    }
    /// How far the back of the hand is turned toward the camera (0..1).
    fn back_facing(&self) -> f64 {
        smoothstep(0.1, 0.45, -self.normal()[2])
    }
}

/// Tesseract scale for a hand: small for a fist, large for an open hand.
fn open_scale(open: f64) -> f64 {
    0.25 + (1.5 - 0.25) * open
}

/// Projected half-height of the tesseract per unit of `size` (measured).
const HALF_EXTENT: f64 = 1.15;

/// How much a hand is palm-up (back of the hand down): 0..1.
fn palm_up(h: &HandInfo) -> f64 {
    smoothstep(0.35, 0.75, h.normal()[1])
}

/// Screen y (aspect space) that puts the tesseract hovering clear above the
/// highest point of the given hands, with a gap.
fn hover_y(hands: &[&HandInfo], size: f64, aspect: f64) -> f64 {
    let top = hands
        .iter()
        .flat_map(|h| h.pts.iter().map(|p| p[1]))
        .fold(f64::INFINITY, f64::min);
    let hl = hands.iter().map(|h| hand_len(&h.pts, aspect)).sum::<f64>() / hands.len() as f64;
    top - size * HALF_EXTENT - hl * 0.35
}

/// Where the tesseract floats for one hand, like a spell held off the palm:
/// pushed out from the palm along the palm normal (as seen on screen), or past
/// the fingertips when the palm faces the camera, until it clears the
/// outermost point of the hand with a visible gap.
fn float_center(h: &HandInfo, rot: &M3, size: f64, aspect: f64) -> [f64; 2] {
    let p = palm(&h.pts);
    let origin = [p[0] * aspect, p[1]];
    // Screen directions (y down) of the palm normal and the fingers.
    let n = orient::col(rot, 2);
    let u = orient::col(rot, 1);
    let n2 = [n[0], -n[1]];
    let u2 = [u[0], -u[1]];
    let nl = len2(n2).min(1.0);
    let mut dir = [n2[0] + u2[0] * (1.0 - nl), n2[1] + u2[1] * (1.0 - nl)];
    let dl = len2(dir);
    if dl < 1e-6 {
        dir = [0.0, -1.0];
    } else {
        dir = [dir[0] / dl, dir[1] / dl];
    }
    // How far the hand reaches in that direction.
    let reach = h
        .pts
        .iter()
        .map(|q| (q[0] * aspect - origin[0]) * dir[0] + (q[1] - origin[1]) * dir[1])
        .fold(0.0, f64::max);
    let gap = hand_len(&h.pts, aspect) * 0.6;
    let dist = reach + gap + size * HALF_EXTENT;
    [origin[0] + dir[0] * dist, origin[1] + dir[1] * dist]
}


// ---- state ---------------------------------------------------------------------

/// Smoothed placement of the figure. Positions are in "aspect space":
/// x is multiplied by the frame aspect so distances are isotropic.
struct Scene {
    center: [f64; 2],
    size: f64,
    rot: M3,
    /// Hand-controlled 4D rotation (two-hand mode), smoothed.
    w_angles: V3,
    /// Automatic rotation, advanced only while resting on one hand:
    /// [XW, YW, ZW, yaw]. Frozen (not reset) while two hands control it.
    spin: [f64; 4],
    /// Hands' relative rotation when two-hand control began, so control
    /// starts from where the spin left off instead of jumping.
    baseline: Option<V3>,
    /// 0..1 smoothed: hovering above upturned palms (drives a gentle bob).
    hover: f64,
    /// Seconds of scene time, for the bob.
    clock: f64,
    presence: f64,
    tess: Tesseract,
}

/// Automatic spin rates (rad/s) for [XW, YW, ZW, yaw].
const SPIN_RATES: [f64; 4] = [0.65, 0.25, 0.4, 0.3];

/// Fixed viewing angle added on top of the hand's rotation, so a hand in the
/// neutral pose shows the tesseract in 3/4 view rather than face-on.
fn base_view() -> M3 {
    orient::mul(&orient::rot_x(0.35), &orient::rot_y(-0.55))
}

impl Scene {
    fn new() -> Self {
        Self {
            center: [0.5, 0.5],
            size: 0.2,
            rot: orient::IDENTITY,
            w_angles: [0.0; 3],
            spin: [0.0, 0.4, 0.0, 0.0],
            baseline: None,
            hover: 0.0,
            clock: 0.0,
            presence: 0.0,
            tess: Tesseract::new(),
        }
    }

    fn step(&mut self, dt: f64, aspect: f64, hands: &[HandInfo]) -> Output {
        // Target placement + orientation from the active hands.
        struct Target {
            center: [f64; 2],
            size: f64,
            rot: M3,
            w_angles: V3,
            /// 0..1: how much it's hovering above upturned palms.
            hover: f64,
            mode: &'static str,
        }

        let target = match hands {
            [h] => {
                // Size from how open the hand is; placed off the palm side.
                let size = hand_len(&h.pts, aspect) * open_scale(h.open);
                // Float direction from the *smoothed* orientation: the tesseract
                // sits far out along it, so raw depth jitter would swing it around.
                let (dir_rot, float_size) =
                    if self.presence > 0.02 { (self.rot, self.size) } else { (h.rot, size) };
                let center = float_center(h, &dir_rot, float_size, aspect);
                Some(Target { center, size, rot: h.rot, w_angles: [0.0; 3], hover: 1.0, mode: "one-hand" })
            }
            [ha, hb, ..] => {
                // Order hands left -> right so the 4D direction is consistent.
                let (l, r) = if palm(&ha.pts)[0] <= palm(&hb.pts)[0] { (ha, hb) } else { (hb, ha) };
                let a = palm(&l.pts);
                let b = palm(&r.pts);
                let gap = len2([(b[0] - a[0]) * aspect, b[1] - a[1]]);
                // Distance between the hands sets the room; openness scales within it.
                let open = (l.open + r.open) / 2.0;
                let size = gap * 0.42 * (0.4 + 0.6 * open);
                // Between the palms, nudged toward the side both palms face.
                let n = orient::add(l.normal(), r.normal());
                let lift = size * 0.35;
                let mut center = [
                    (a[0] + b[0]) / 2.0 * aspect + n[0] * 0.5 * lift,
                    (a[1] + b[1]) / 2.0 - n[1] * 0.5 * lift,
                ];
                // Both palms up: hover above both hands.
                let up = palm_up(l).min(palm_up(r));
                center[1] += (hover_y(&[l, r], size, aspect).min(center[1]) - center[1]) * up;
                let avg = orient::from_axes(
                    orient::add(orient::col(&l.rot, 0), orient::col(&r.rot, 0)),
                    orient::add(orient::col(&l.rot, 1), orient::col(&r.rot, 1)),
                )
                .unwrap_or(l.rot);
                // How the right hand is turned relative to the left -> 4D rotation,
                // measured from the moment two-hand control began.
                let rel = orient::mul(&orient::transpose(&l.rot), &r.rot);
                let ang = orient::angles(&rel);
                let raw = [ang[1], ang[0], ang[2]];
                let base = *self.baseline.get_or_insert(raw);
                let w_angles = [0, 1, 2].map(|i| wrap_angle(raw[i] - base[i]));
                Some(Target { center, size, rot: avg, w_angles, hover: 1.0, mode: "two-hands" })
            }
            [] => None,
        };

        let mode = match target {
            Some(t) => {
                if self.presence < 0.02 {
                    // Snap when (re)appearing instead of sliding in from the old spot.
                    self.center = t.center;
                    self.size = t.size;
                    self.rot = t.rot;
                    self.w_angles = t.w_angles;
                    self.hover = t.hover;
                } else {
                    // Easing with a soft dead zone: changes smaller than the zone
                    // are followed only slightly (jitter is ignored), larger ones
                    // at full speed. Landmarks are already One Euro filtered.
                    let ease = |rate: f64| 1.0 - (-dt * rate).exp();
                    let dead = |delta: f64, zone: f64| (delta.abs() / zone).min(1.0);

                    let k = ease(12.0);
                    let d = [t.center[0] - self.center[0], t.center[1] - self.center[1]];
                    let r = dead(len2(d), self.size * 0.05);
                    self.center[0] += d[0] * k * r;
                    self.center[1] += d[1] * k * r;
                    // Openness is measured in 2D now (no depth noise), so size can
                    // follow finger spread closely with only a small dead zone.
                    let ds = t.size - self.size;
                    self.size += ds * ease(10.0) * dead(ds, self.size * 0.04);
                    self.hover += (t.hover - self.hover) * ease(6.0);

                    let kr = ease(8.0);
                    let rel = orient::mul(&orient::transpose(&self.rot), &t.rot);
                    let cos = ((rel[0][0] + rel[1][1] + rel[2][2] - 1.0) / 2.0).clamp(-1.0, 1.0);
                    self.rot = orient::blend(&self.rot, &t.rot, kr * dead(cos.acos(), 0.05));
                    for i in 0..3 {
                        let dw = wrap_angle(t.w_angles[i] - self.w_angles[i]);
                        self.w_angles[i] += dw * kr * dead(dw, 0.04);
                    }
                }
                self.presence += (1.0 - self.presence) * (1.0 - (-dt * 8.0).exp());
                t.mode
            }
            None => {
                self.presence *= (-dt * 5.0).exp();
                "idle"
            }
        };
        if self.presence < 1e-3 {
            self.presence = 0.0;
        }

        if mode != "two-hands" && self.baseline.take().is_some() {
            // Leaving two-hand control: keep the pose the hands left it in by
            // folding the control angles into the spin, which resumes from there.
            for i in 0..3 {
                self.spin[i] += self.w_angles[i];
            }
            self.w_angles = [0.0; 3];
        }
        self.clock += dt;
        if mode == "one-hand" {
            for (a, rate) in self.spin.iter_mut().zip(SPIN_RATES) {
                *a = (*a + rate * dt) % (2.0 * PI);
            }
        }

        let rot = orient::mul(
            &self.rot,
            &orient::mul(&orient::rot_y(self.spin[3]), &base_view()),
        );
        let w = [0, 1, 2].map(|i| self.spin[i] + self.w_angles[i]);
        let projected = self.tess.project(&rot, w);

        // Gentle levitation bob while floating off the hands.
        let bob = (self.clock * 2.0 * PI * 0.45).sin() * self.size * 0.08 * self.hover;
        let to_screen = |p: [f64; 2]| -> [f64; 2] {
            // Projection space is y-up; screen is y-down.
            [(self.center[0] + p[0] * self.size) / aspect, self.center[1] + bob - p[1] * self.size]
        };
        let screen: Vec<([f64; 2], f64)> = projected.iter().map(|(p, d)| (to_screen(*p), *d)).collect();

        let edges = self
            .tess
            .edges()
            .iter()
            .map(|&(i, j)| {
                let (a, da) = screen[i];
                let (b, db) = screen[j];
                [a[0], a[1], b[0], b[1], (da + db) / 2.0]
            })
            .collect();

        Output {
            alpha: self.presence,
            mode,
            edges,
            vertices: screen.iter().map(|(p, d)| [p[0], p[1], *d]).collect(),
            hands: Vec::new(),
            center: [self.center[0] / aspect, self.center[1]],
            size: self.size,
        }
    }
}

// ---- main loop -------------------------------------------------------------------

fn main() {
    // Reader thread: parse observations as they arrive, stamped with arrival time.
    let (tx, rx) = mpsc::channel::<(Instant, Input)>();
    thread::spawn(move || {
        for line in io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<Input>(&line) {
                Ok(input) => {
                    if tx.send((Instant::now(), input)).is_err() {
                        break;
                    }
                }
                Err(e) => eprintln!("engine: bad input: {e}"),
            }
        }
    });

    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());
    let mut tracker = Tracker::default();
    let mut scene = Scene::new();
    let mut aspect = 16.0 / 9.0;

    let period = Duration::from_secs_f64(1.0 / TICK_HZ);
    let mut next = Instant::now();
    let mut last = Instant::now();

    loop {
        // Take in every observation that arrived since the last tick.
        loop {
            match rx.try_recv() {
                Ok((at, input)) => {
                    if input.aspect > 0.0 {
                        aspect = input.aspect;
                    }
                    let obs = input.hands.into_iter().map(Hand::into_observation).collect();
                    tracker.observe(input.t, at, obs);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }

        let now = Instant::now();
        let dt = now.duration_since(last).as_secs_f64().min(0.1);
        last = now;
        tracker.expire(now);

        // Predict every hand to this instant. A hand becomes active once it's
        // raised, then stays active (free to rotate any way) while tracked.
        let predicted: Vec<([V3; predict::N], usize)> = tracker
            .tracks
            .iter()
            .enumerate()
            .map(|(i, tr)| (tr.predict(now), i))
            .collect();
        for (pts, i) in &predicted {
            if is_raised(pts) {
                tracker.tracks[*i].active = true;
            }
        }
        let infos: Vec<(Option<HandInfo>, &Track, [V3; predict::N])> = predicted
            .iter()
            .map(|(pts, i)| {
                let tr = &tracker.tracks[*i];
                (HandInfo::new(*pts, tr.label(), aspect), tr, *pts)
            })
            .collect();
        let active: Vec<HandInfo> = infos
            .iter()
            .filter(|(_, tr, _)| tr.active)
            .filter_map(|(h, _, _)| h.as_ref())
            .map(|h| HandInfo { pts: h.pts, rot: h.rot, open: h.open })
            .collect();

        let mut output = scene.step(dt, aspect, &active);
        output.hands = infos
            .iter()
            .map(|(h, tr, pts)| {
                let (facing, occlude, open) = match h {
                    Some(h) => {
                        let nz = h.normal()[2];
                        let facing = if nz > 0.25 { "palm" } else if nz < -0.25 { "back" } else { "side" };
                        let occlude = if tr.active { h.back_facing() } else { 0.0 };
                        (facing, occlude, h.open)
                    }
                    None => ("side", 0.0, 0.0),
                };
                OutHand {
                    tips: pts[predict::TIPS].iter().map(|p| [p[0], p[1]]).collect(),
                    outline: hull(pts.iter().map(|p| [p[0], p[1]]).collect()),
                    facing,
                    occlude,
                    openness: open,
                }
            })
            .collect();

        if serde_json::to_writer(&mut out, &output).is_err()
            || writeln!(out).is_err()
            || out.flush().is_err()
        {
            return;
        }

        // Fixed-rate schedule; resync if we fell badly behind.
        next += period;
        let now = Instant::now();
        if next > now {
            thread::sleep(next - now);
        } else if now - next > period * 4 {
            next = now;
        }
    }
}
