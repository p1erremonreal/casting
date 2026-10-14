//! 3D rotation math for hand orientation.
//!
//! Camera space used here: x right, y up, z toward the viewer, in units of
//! frame height (x and MediaPipe's z are scaled by the aspect ratio so all
//! three axes share a unit).

pub type V3 = [f64; 3];
/// Row-major 3x3 matrix: m[row][col].
pub type M3 = [[f64; 3]; 3];

pub const IDENTITY: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
pub fn norm(a: V3) -> Option<V3> {
    let l = dot(a, a).sqrt();
    (l > 1e-9).then(|| scale(a, 1.0 / l))
}

pub fn mul(a: &M3, b: &M3) -> M3 {
    let mut m = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            m[r][c] = (0..3).map(|k| a[r][k] * b[k][c]).sum();
        }
    }
    m
}
pub fn transpose(a: &M3) -> M3 {
    let mut m = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            m[r][c] = a[c][r];
        }
    }
    m
}
pub fn apply(m: &M3, v: V3) -> V3 {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}
pub fn rot_x(a: f64) -> M3 {
    let (s, c) = a.sin_cos();
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}
pub fn rot_y(a: f64) -> M3 {
    let (s, c) = a.sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

/// Rotation whose columns are the orthonormalized `x` and `y` axes and x × y.
pub fn from_axes(x: V3, y: V3) -> Option<M3> {
    let y = norm(y)?;
    let x = norm(sub(x, scale(y, dot(x, y))))?;
    let z = cross(x, y);
    Some([[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]])
}
pub fn col(m: &M3, c: usize) -> V3 {
    [m[0][c], m[1][c], m[2][c]]
}

/// Move rotation `cur` toward `target` by fraction `k` (blend axes, re-orthonormalize).
pub fn blend(cur: &M3, target: &M3, k: f64) -> M3 {
    let x = add(scale(col(cur, 0), 1.0 - k), scale(col(target, 0), k));
    let y = add(scale(col(cur, 1), 1.0 - k), scale(col(target, 1), k));
    from_axes(x, y).unwrap_or(*target)
}

/// Rotation angles about x, y, z extracted from a rotation matrix (for small
/// relative rotations these behave like independent tilt / turn / twist).
pub fn angles(m: &M3) -> V3 {
    [
        m[2][1].atan2(m[1][1]), // about x: tilt
        m[0][2].atan2(m[2][2]), // about y: turn
        m[1][0].atan2(m[0][0]), // about z: twist
    ]
}
