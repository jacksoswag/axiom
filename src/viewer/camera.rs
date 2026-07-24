//! Perspective orbit and free-flight camera for the fixed 3-D torus.

use eframe::egui::{Pos2, Rect, Vec2};
type V3 = [f32; 3];
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V3, s: f32) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn unit(a: V3) -> V3 {
    let length = dot(a, a).sqrt();
    if length > 1e-9 {
        scale(a, 1.0 / length)
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Integer offsets of the 27 neighboring torus images, nearest first.
pub fn torus_images() -> [[i32; 3]; 27] {
    let mut images = [[0; 3]; 27];
    let mut index = 0;
    for x in -1i32..=1 {
        for y in -1i32..=1 {
            for z in -1i32..=1 {
                images[index] = [x, y, z];
                index += 1;
            }
        }
    }
    images.sort_by_key(|o| (o.iter().map(|&v| v.abs()).sum::<i32>(), *o));
    images
}
pub struct Projected {
    pub screen: Pos2,
    pub depth: f32,
    pub scale: f32,
}
pub struct Camera {
    pub target: V3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub focal: f32,
    pub captured: bool,
    pub look_speed: f32,
    pub move_speed: f32,
}
impl Default for Camera {
    fn default() -> Self {
        Self {
            target: [0.0; 3],
            yaw: 0.0,
            pitch: -0.25,
            distance: 1.0,
            focal: 900.0,
            captured: false,
            look_speed: 0.0032,
            move_speed: 0.9,
        }
    }
}
impl Camera {
    pub fn frame_world(&mut self, extent: f32) {
        self.target = [0.0; 3];
        self.distance = extent * 0.32;
        self.yaw = 0.6;
        self.pitch = -0.3;
    }
    pub fn basis(&self) -> (V3, V3, V3) {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let forward = unit([cp * sy, sp, cp * cy]);
        let right = unit(cross(forward, [0.0, 1.0, 0.0]));
        (right, cross(right, forward), forward)
    }
    pub fn eye(&self) -> V3 {
        let (_, _, forward) = self.basis();
        sub(self.target, scale(forward, self.distance))
    }
    pub fn project(&self, position: &[f32], extent: f32, viewport: Rect) -> Option<Projected> {
        self.project_point(
            [
                position[0] - extent * 0.5,
                position[1] - extent * 0.5,
                position[2] - extent * 0.5,
            ],
            extent,
            viewport,
        )
    }
    pub fn project_point(&self, point: V3, extent: f32, viewport: Rect) -> Option<Projected> {
        let (right, up, forward) = self.basis();
        let relative = sub(point, self.eye());
        let ahead = dot(relative, forward);
        if ahead <= extent * 0.01 {
            return None;
        }
        let perspective = self.focal / ahead;
        Some(Projected {
            screen: viewport.center()
                + Vec2::new(dot(relative, right), -dot(relative, up)) * perspective,
            depth: (ahead / (extent * 2.5)).clamp(0.0, 1.0),
            scale: perspective.clamp(0.05, 8.0),
        })
    }
    pub fn orbit(&mut self, delta: Vec2) {
        self.yaw -= delta.x * self.look_speed * 1.6;
        self.pitch = (self.pitch - delta.y * self.look_speed * 1.6).clamp(-1.5533, 1.5533);
    }
    pub fn dolly(&mut self, scroll: f32, extent: f32) {
        self.distance =
            (self.distance * (1.0 - scroll * 0.0015)).clamp(extent * 0.05, extent * 8.0);
    }
    pub fn fly(&mut self, look: Vec2, movement: V3, extent: f32, dt: f32) {
        self.yaw += look.x * self.look_speed;
        self.pitch = (self.pitch - look.y * self.look_speed).clamp(-1.5533, 1.5533);
        let (right, up, forward) = self.basis();
        let step = add(
            add(scale(right, movement[0]), scale(up, movement[1])),
            scale(forward, movement[2]),
        );
        self.target = add(self.target, scale(step, self.move_speed * extent * dt));
    }
}
