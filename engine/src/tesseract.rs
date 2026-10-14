//! A 4D hypercube: 16 vertices at (±1, ±1, ±1, ±1), 32 edges joining
//! vertices that differ in exactly one coordinate. Rotated in 4D, projected
//! to 3D with a perspective divide on w, oriented in 3D, then projected to 2D
//! with a divide on z. No rotation happens on its own: all angles come from
//! the caller (i.e. from the hands).

use crate::orient::{apply, M3};

type V4 = [f64; 4];

/// Distance of the 4D "camera" from the origin along w. Must exceed the
/// vertex radius (2) so nothing crosses the projection plane.
const D4: f64 = 3.0;
/// Distance of the 3D camera along z.
const D3: f64 = 4.0;

pub struct Tesseract {
    vertices: [V4; 16],
    edges: Vec<(usize, usize)>,
}

/// Rotate coordinates `a` and `b` of `v` by `angle`.
fn rotate(v: &mut V4, a: usize, b: usize, angle: f64) {
    let (s, c) = angle.sin_cos();
    let (va, vb) = (v[a], v[b]);
    v[a] = va * c - vb * s;
    v[b] = va * s + vb * c;
}

impl Tesseract {
    pub fn new() -> Self {
        let mut vertices = [[0.0; 4]; 16];
        for (i, v) in vertices.iter_mut().enumerate() {
            for (axis, coord) in v.iter_mut().enumerate() {
                *coord = if i & (1 << axis) != 0 { 1.0 } else { -1.0 };
            }
        }
        let mut edges = Vec::with_capacity(32);
        for i in 0..16 {
            for axis in 0..4 {
                let j = i ^ (1 << axis);
                if i < j {
                    edges.push((i, j));
                }
            }
        }
        Self { vertices, edges }
    }

    pub fn edges(&self) -> &[(usize, usize)] {
        &self.edges
    }

    /// Project with 4D rotation angles `w_angles` (XW, YW, ZW planes) and 3D
    /// orientation `rot`. Returns each vertex in 2D (y-up, roughly unit
    /// scale) with a depth in 0..1, where 1 is nearest the 4D camera.
    pub fn project(&self, rot: &M3, w_angles: [f64; 3]) -> Vec<([f64; 2], f64)> {
        self.vertices
            .iter()
            .map(|&v0| {
                let mut v = v0;
                rotate(&mut v, 0, 3, w_angles[0]);
                rotate(&mut v, 1, 3, w_angles[1]);
                rotate(&mut v, 2, 3, w_angles[2]);

                // 4D -> 3D perspective.
                let w = v[3];
                let f4 = 1.0 / (D4 - w);
                let p = apply(rot, [v[0] * f4, v[1] * f4, v[2] * f4]);

                // 3D -> 2D perspective. Scale so the outer cube is ~1 unit wide.
                let f3 = D3 / (D3 - p[2]);
                let scale = 1.6;
                let depth = ((w + 2.0) / 4.0).clamp(0.0, 1.0);
                ([p[0] * f3 * scale, p[1] * f3 * scale], depth)
            })
            .collect()
    }
}
