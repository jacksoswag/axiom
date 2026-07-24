//! Drawing the swarm: glow sprites, the torus frame, and the frosted pass behind the panel.
//!
//! Every particle contributes two textured quads to a single `Mesh`, so the whole swarm is
//! one draw call regardless of count.
//!
//! **Known limitation:** epaint has no additive blend mode, so overlapping halos composite
//! instead of accumulating. Against a near-black canvas the difference is small, because
//! the background contributes almost nothing to the blend. Real bloom would need a custom
//! `egui_wgpu` render pass, which would pin the render backend for a marginal gain.

use eframe::egui::{Color32, ColorImage, Context, Painter, Rect, TextureHandle, TextureOptions};
use eframe::egui::{Pos2, Stroke, Vec2};
use eframe::epaint::{Mesh, Shape};
use std::sync::Arc;

use super::camera::{self, Camera};
use super::theme;
use crate::engine::substrate::Substrate;

/// The explorer renders three-dimensional geometry, so position striding stays pinned to 3.
const DIMENSIONS: usize = 3;

const SPRITE: usize = 64;

/// Ceiling on particle-draws per frame, summed over torus images.
///
/// epaint tessellates on the CPU, so every draw is two quads and eight vertices on this
/// thread. At 100k particles even the home image is 800k vertices, and nine images would be
/// 7.2M. The budget spends what is left after the swarm itself on as many images as fit,
/// nearest first, so the endless look degrades gracefully instead of stalling the frame.
const SPRITE_BUDGET: usize = 120_000;

/// How the swarm is drawn. Everything here is user-facing.
pub struct Style {
    pub bead_size: f32,
    pub glow: f32,
    pub show_frame: bool,
}

impl Default for Style {
    fn default() -> Style {
        Style {
            bead_size: 3.0,
            glow: 1.0,
            show_frame: true,
        }
    }
}

impl Style {
    fn color(&self, trait_value: f32) -> Color32 {
        theme::trait_color(trait_value)
    }

    fn size(&self, _trait_value: f32) -> f32 {
        self.bead_size
    }
}

/// Everything needed to draw one frame of the swarm. The box comes from the substrate, so the
/// scene cannot disagree with the world it draws.
pub struct Scene<'a> {
    pub substrate: &'a Substrate,
    pub camera: &'a Camera,
    pub viewport: Rect,
    pub style: &'a Style,
}

pub struct Renderer {
    sprite: TextureHandle,
}

impl Renderer {
    pub fn new(ctx: &Context) -> Renderer {
        Renderer {
            sprite: ctx.load_texture("glow", radial_sprite(), TextureOptions::LINEAR),
        }
    }

    /// Build the swarm mesh. `bloat` scales every quad and `dim` scales every alpha, which
    /// is how the frosted pass reuses this without a second code path.
    fn mesh(
        &self,
        scene: &Scene,
        bloat: f32,
        dim: f32,
        stride: usize,
        image_budget: usize,
    ) -> Mesh {
        let Scene { substrate, camera, viewport, style } = *scene;
        let box_len = substrate.box_len;
        let mut mesh = Mesh::with_texture(self.sprite.id());
        let mut shifted = [0.0f32; DIMENSIONS];
        let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));

        // Draw the neighboring torus images so the world reads as endless rather than as a
        // box in a void. Nearest first under a budget, so a large swarm loses distant images
        // instead of tessellating millions of quads onto the CPU mesh path.
        let images = camera::torus_images();
        let affordable = (SPRITE_BUDGET / substrate.traits.len().max(1))
            .clamp(1, images.len())
            .min(image_budget);
        let cull = viewport.expand(140.0);

        for image in images.iter().take(affordable) {
            for (particle, position) in substrate
                .positions
                .chunks(DIMENSIONS)
                .enumerate()
                .step_by(stride.max(1))
            {
                if !position.iter().all(|p| p.is_finite()) {
                    continue;
                }
                for (axis, slot) in shifted.iter_mut().enumerate() {
                    *slot = position[axis] + image[axis] as f32 * box_len;
                }
                let Some(projected) = camera.project(&shifted, box_len, viewport) else {
                    continue;
                };
                if !cull.contains(projected.screen) {
                    continue;
                }

                let trait_value = substrate.traits.get(particle).copied().unwrap_or(0.5);
                let tint = style.color(trait_value);
                // Further particles dim and shrink. That is the depth cue in 3-D and up; in
                // the orthographic modes `depth` is fixed so this collapses to a constant.
                let fade = 1.0 - projected.depth * 0.55;
                let radius = style.size(trait_value) * projected.scale.min(4.0) * fade.max(0.25);

                let halo_alpha = (34.0 * style.glow * fade * dim).clamp(0.0, 255.0) as u8;
                if halo_alpha > 1 {
                    quad(
                        &mut mesh,
                        projected.screen,
                        radius * 4.2 * bloat,
                        uv,
                        alpha(tint, halo_alpha),
                    );
                }
                let core_alpha = (238.0 * fade * dim).clamp(0.0, 255.0) as u8;
                quad(
                    &mut mesh,
                    projected.screen,
                    radius * 1.15 * bloat,
                    uv,
                    alpha(tint, core_alpha),
                );
            }
        }
        mesh
    }

    pub fn draw(&self, painter: &Painter, scene: &Scene) {
        if scene.style.show_frame {
            self.frame(painter, scene.camera, scene.substrate.box_len, scene.viewport);
        }
        painter.add(Shape::Mesh(Arc::new(self.mesh(
            scene,
            1.0,
            1.0,
            1,
            usize::MAX,
        ))));
    }

    /// Sparse true-particle grain for the material path. Indices are sampled deterministically,
    /// so pausing a world cannot make the overlay shimmer independently of the simulation.
    pub fn draw_beads(&self, painter: &Painter, scene: &Scene) {
        let stride = (scene.substrate.traits.len() / 700).max(1);
        painter.add(Shape::Mesh(Arc::new(
            self.mesh(scene, 0.72, 0.72, stride, 1),
        )));
    }

    /// The blur that makes the panel read as glass rather than tinted plastic.
    ///
    /// egui has no backdrop filter, so the swarm is redrawn behind the panel at large
    /// radius and low alpha, clipped to the panel. It is a real approximation built from
    /// geometry the frame already has, not Apple's Liquid Glass, which is a platform
    /// material with no egui equivalent.
    pub fn frost(&self, painter: &Painter, scene: &Scene, clip: Rect) {
        let mesh = self.mesh(scene, 3.2, 0.16, 1, usize::MAX);
        painter
            .with_clip_rect(clip)
            .add(Shape::Mesh(Arc::new(mesh)));
    }

    /// Wireframe of the periodic box, so orientation survives flying outside it.
    fn frame(&self, painter: &Painter, camera: &Camera, box_len: f32, viewport: Rect) {
        let stroke = Stroke::new(1.0, Color32::from_white_alpha(0x16));
        let half = box_len * 0.5;

        let corner = |i: usize| -> [f32; 3] {
            [
                if i & 1 == 0 { -half } else { half },
                if i & 2 == 0 { -half } else { half },
                if i & 4 == 0 { -half } else { half },
            ]
        };
        let projected: Vec<Option<Pos2>> = (0..8)
            .map(|i| {
                camera
                    .project_point(corner(i), box_len, viewport)
                    .map(|p| p.screen)
            })
            .collect();

        for (a, b) in EDGES {
            if let (Some(from), Some(to)) = (projected[a], projected[b]) {
                painter.line_segment([from, to], stroke);
            }
        }
    }
}

/// The 12 edges of a cube, as index pairs into the 8 corners.
const EDGES: [(usize, usize); 12] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

fn alpha(color: Color32, a: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), a)
}

fn quad(mesh: &mut Mesh, centre: Pos2, radius: f32, uv: Rect, color: Color32) {
    mesh.add_rect_with_uv(
        Rect::from_center_size(centre, Vec2::splat(radius * 2.0)),
        uv,
        color,
    );
}

/// White sprite with a radial alpha falloff. The exponent controls how tight the core
/// reads: 2.5 gives a bright centre with a long, soft skirt.
fn radial_sprite() -> ColorImage {
    let mut pixels = Vec::with_capacity(SPRITE * SPRITE);
    let centre = (SPRITE as f32 - 1.0) * 0.5;
    for y in 0..SPRITE {
        for x in 0..SPRITE {
            let dx = (x as f32 - centre) / centre;
            let dy = (y as f32 - centre) / centre;
            let r = (dx * dx + dy * dy).sqrt();
            let falloff = (1.0 - r).clamp(0.0, 1.0).powf(2.5);
            pixels.push(Color32::from_white_alpha((falloff * 255.0) as u8));
        }
    }
    ColorImage::new([SPRITE, SPRITE], pixels)
}
