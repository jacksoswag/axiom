//! Causal CPU reference material renderer for three-dimensional Particle-Lenia worlds.
//!
//! The field is a pure, periodic observation of one particle snapshot. It owns no clock,
//! random source, or mutable biological state: an unchanged snapshot produces the same texture.

use eframe::egui::{
    Color32, ColorImage, Context, Painter, Pos2, Rect, TextureHandle, TextureOptions,
};

use super::camera::Camera;
pub use crate::render_recipe::RenderRecipe as Recipe;

const EPSILON: f32 = 1e-6;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    PositionLength,
    TraitLength,
    InvalidExtent,
    InvalidRecipe,
    NonFiniteInput,
}

/// A canonical, periodic scalar field. `density` and `trait_sum` use x-major indexing.
#[derive(Clone, Debug)]
pub struct DensityField {
    pub resolution: usize,
    pub extent: f32,
    pub density: Vec<f32>,
    trait_sum: Vec<f32>,
}

impl DensityField {
    /// Deposits each particle into only the cells inside the compact kernel support. Every cell
    /// index wraps, so a particle at a torus face contributes seamlessly to the opposite face.
    pub fn from_particles(
        positions: &[f32],
        traits: &[f32],
        extent: f32,
        recipe: &Recipe,
    ) -> Result<Self, Error> {
        if !positions.len().is_multiple_of(3) {
            return Err(Error::PositionLength);
        }
        let count = positions.len() / 3;
        if traits.len() != count {
            return Err(Error::TraitLength);
        }
        if !extent.is_finite() || extent <= 0.0 {
            return Err(Error::InvalidExtent);
        }
        if !recipe.valid() {
            return Err(Error::InvalidRecipe);
        }
        if !positions.iter().chain(traits).all(|v| v.is_finite()) {
            return Err(Error::NonFiniteInput);
        }

        let side = recipe.resolution;
        let cells = side
            .checked_mul(side)
            .and_then(|area| area.checked_mul(side))
            .ok_or(Error::InvalidRecipe)?;
        let mut field = DensityField {
            resolution: side,
            extent,
            density: vec![0.0; cells],
            trait_sum: vec![0.0; cells],
        };
        let h = extent / side as f32;
        // A visual kernel wider than half a torus is ambiguous under minimum-image distance.
        // Refuse it rather than double-depositing wrapped cells into a superficially smooth lie.
        let support = recipe.support;
        if support >= extent * 0.5 {
            return Err(Error::InvalidRecipe);
        }
        let span = (support / h).ceil() as isize;
        if span * 2 >= side as isize {
            return Err(Error::InvalidRecipe);
        }
        // Density-referenced, on the same principle `world::calibrate` uses for interaction
        // norms: divide the raw deposit by what a uniform swarm would produce here, so `iso`
        // reads as "times as dense as uniform" instead of an absolute count that meant a
        // different thing at every particle count, extent, or support.
        let inverse_uniform_density = 1.0 / uniform_density(count, extent, support);

        for (particle, &trait_value) in positions.chunks_exact(3).zip(traits) {
            let centre = [
                (particle[0].rem_euclid(extent) / h - 0.5).floor() as isize,
                (particle[1].rem_euclid(extent) / h - 0.5).floor() as isize,
                (particle[2].rem_euclid(extent) / h - 0.5).floor() as isize,
            ];
            for z in centre[2] - span..=centre[2] + span {
                for y in centre[1] - span..=centre[1] + span {
                    for x in centre[0] - span..=centre[0] + span {
                        let sample = [
                            (x as f32 + 0.5) * h,
                            (y as f32 + 0.5) * h,
                            (z as f32 + 0.5) * h,
                        ];
                        let distance = torus_distance(sample, particle, extent);
                        let weight = kernel(distance / support) * inverse_uniform_density;
                        if weight <= 0.0 {
                            continue;
                        }
                        let index = field.index_wrapped(x, y, z);
                        field.density[index] += weight;
                        field.trait_sum[index] += weight * trait_value.clamp(0.0, 1.0);
                    }
                }
            }
        }
        Ok(field)
    }

    pub fn sample(&self, x: f32, y: f32, z: f32) -> f32 {
        self.sample_values(&self.density, x, y, z)
    }

    pub fn trait_at(&self, x: f32, y: f32, z: f32) -> f32 {
        self.sample_values(&self.trait_sum, x, y, z) / self.sample(x, y, z).max(EPSILON)
    }

    pub fn gradient(&self, x: f32, y: f32, z: f32) -> [f32; 3] {
        let h = self.extent / self.resolution as f32;
        [
            (self.sample(x + h, y, z) - self.sample(x - h, y, z)) / (2.0 * h),
            (self.sample(x, y + h, z) - self.sample(x, y - h, z)) / (2.0 * h),
            (self.sample(x, y, z + h) - self.sample(x, y, z - h)) / (2.0 * h),
        ]
    }

    /// Periodic six-neighbour connectivity over material cells. This is intentionally exposed
    /// for product qualification fixtures as well as the renderer's own tests.
    #[cfg(test)]
    pub fn connected_components(&self, threshold: f32) -> usize {
        let mut seen = vec![false; self.density.len()];
        let mut components = 0usize;
        let side = self.resolution as isize;
        let mut stack = Vec::new();
        for index in 0..self.density.len() {
            if seen[index] || self.density[index] < threshold {
                continue;
            }
            components += 1;
            seen[index] = true;
            stack.push(index);
            while let Some(current) = stack.pop() {
                let (x, y, z) = self.coordinates(current);
                for (nx, ny, nz) in [
                    (x - 1, y, z),
                    (x + 1, y, z),
                    (x, y - 1, z),
                    (x, y + 1, z),
                    (x, y, z - 1),
                    (x, y, z + 1),
                ] {
                    let neighbour = self.index_wrapped(
                        nx.rem_euclid(side),
                        ny.rem_euclid(side),
                        nz.rem_euclid(side),
                    );
                    if !seen[neighbour] && self.density[neighbour] >= threshold {
                        seen[neighbour] = true;
                        stack.push(neighbour);
                    }
                }
            }
        }
        components
    }

    #[cfg(test)]
    pub fn shade(&self, recipe: &Recipe, width: usize, height: usize) -> ColorImage {
        let mut pixels = Vec::with_capacity(width * height);
        let side = self.resolution;
        let light = normalise([-0.45, -0.35, 0.82]);
        for py in 0..height {
            for px in 0..width {
                let x = (px as f32 + 0.5) / width as f32 * self.extent;
                let y = (py as f32 + 0.5) / height as f32 * self.extent;
                let mut alpha = 0.0f32;
                let mut red = 0.0f32;
                let mut green = 0.0f32;
                let mut blue = 0.0f32;
                for z_cell in 0..side {
                    let z = (z_cell as f32 + 0.5) / side as f32 * self.extent;
                    let density = self.sample(x, y, z);
                    let (material, interior) = transfer(density, recipe.iso);
                    if material <= 0.01 {
                        continue;
                    }
                    let normal = normalise(self.gradient(x, y, z));
                    let diffuse = dot(normal, light).max(0.0);
                    let trait_value = self.trait_at(x, y, z);
                    let base = palette(trait_value);
                    let emission = 0.18 + 0.38 * material * material;
                    let shade = 0.18 + 0.92 * diffuse + 0.12 * interior;
                    let layer_alpha =
                        (material * recipe.absorption).clamp(0.0, 0.85) * (1.0 - alpha);
                    red += layer_alpha * base[0] * (shade + emission);
                    green += layer_alpha * base[1] * (shade + emission);
                    blue += layer_alpha * base[2] * (shade + emission);
                    alpha += layer_alpha;
                    if alpha > 0.985 {
                        break;
                    }
                }
                pixels.push(straight_alpha(red, green, blue, alpha));
            }
        }
        ColorImage::new([width, height], pixels)
    }

    /// Deterministic front-to-back volume integration along the actual viewer camera rays.
    /// The ray's coordinates wrap at every field lookup, making the periodic world visible
    /// from outside, inside, or across any torus seam without introducing a visual clock.
    pub fn shade_camera(
        &self,
        recipe: &Recipe,
        camera: &Camera,
        width: usize,
        height: usize,
        viewport_height: f32,
        previous_density: Option<&[f32]>,
    ) -> ColorImage {
        let mut pixels = Vec::with_capacity(width * height);
        let (right, up, forward) = camera.basis();
        let eye = camera.eye();
        let origin = [
            eye[0] + self.extent * 0.5,
            eye[1] + self.extent * 0.5,
            eye[2] + self.extent * 0.5,
        ];
        let aspect = width as f32 / height.max(1) as f32;
        let focal = (2.0 * camera.focal / viewport_height.max(1.0)).clamp(0.3, 8.0);
        let steps = self.resolution * 2;
        let distance_step = self.extent / steps as f32;
        let light = normalise([-0.45, -0.35, 0.82]);
        for py in 0..height {
            for px in 0..width {
                let sx = ((px as f32 + 0.5) / width as f32 * 2.0 - 1.0) * aspect / focal;
                let sy = (1.0 - (py as f32 + 0.5) / height as f32 * 2.0) / focal;
                let direction = normalise([
                    forward[0] + right[0] * sx + up[0] * sy,
                    forward[1] + right[1] * sx + up[1] * sy,
                    forward[2] + right[2] * sx + up[2] * sy,
                ]);
                let mut alpha = 0.0f32;
                let mut red = 0.0f32;
                let mut green = 0.0f32;
                let mut blue = 0.0f32;
                let before = [
                    origin[0] - direction[0] * distance_step,
                    origin[1] - direction[1] * distance_step,
                    origin[2] - direction[2] * distance_step,
                ];
                let mut ray_previous_density = self.sample(before[0], before[1], before[2]);
                for step in 0..steps {
                    let distance = step as f32 * distance_step;
                    let point = [
                        origin[0] + direction[0] * distance,
                        origin[1] + direction[1] * distance,
                        origin[2] + direction[2] * distance,
                    ];
                    let density = self.sample(point[0], point[1], point[2]);
                    let entered_surface =
                        ray_previous_density < recipe.iso && density >= recipe.iso;
                    ray_previous_density = density;
                    let (material, interior) = transfer(density, recipe.iso);
                    if material <= 0.01 {
                        continue;
                    }
                    let normal = normalise(self.gradient(point[0], point[1], point[2]));
                    let diffuse = dot(normal, light).max(0.0);
                    let trait_value = self.trait_at(point[0], point[1], point[2]);
                    let base = palette(trait_value);
                    let activity = previous_density
                        .filter(|previous| previous.len() == self.density.len())
                        .map(|previous| {
                            ((density - self.sample_values(previous, point[0], point[1], point[2]))
                                .abs()
                                / recipe.iso.max(EPSILON))
                            .clamp(0.0, 1.0)
                        })
                        .unwrap_or(0.0);
                    let rim = (1.0 - dot(normal, direction).abs()).powi(2);
                    // `material^2`'s coefficient gets the same units-consistency correction as
                    // `absorption` (see `render_recipe::BRIGHTNESS_GAIN`): it is the other place
                    // brightness reads `material` through a term proportional to it.
                    let emission = 0.14
                        + crate::render_recipe::BRIGHTNESS_GAIN * 0.40 * material * material
                        + 0.55 * activity;
                    let shade = 0.16 + 0.88 * diffuse + 0.32 * rim + 0.10 * interior;
                    let opacity = if entered_surface {
                        0.40 + 0.25 * recipe.absorption
                    } else {
                        material * recipe.absorption * 0.16
                    };
                    let layer_alpha = opacity.clamp(0.0, 0.92) * (1.0 - alpha);
                    red += layer_alpha * base[0] * (shade + emission);
                    green += layer_alpha * base[1] * (shade + emission);
                    blue += layer_alpha * base[2] * (shade + emission);
                    alpha += layer_alpha;
                    if alpha > 0.985 {
                        break;
                    }
                }
                pixels.push(straight_alpha(red, green, blue, alpha));
            }
        }
        ColorImage::new([width, height], pixels)
    }

    fn sample_values(&self, values: &[f32], x: f32, y: f32, z: f32) -> f32 {
        let h = self.extent / self.resolution as f32;
        let grid = [x / h - 0.5, y / h - 0.5, z / h - 0.5];
        let base = [
            grid[0].floor() as isize,
            grid[1].floor() as isize,
            grid[2].floor() as isize,
        ];
        let fraction = [
            grid[0] - grid[0].floor(),
            grid[1] - grid[1].floor(),
            grid[2] - grid[2].floor(),
        ];
        let mut value = 0.0;
        for dz in 0..=1 {
            for dy in 0..=1 {
                for dx in 0..=1 {
                    let weight = if dx == 0 {
                        1.0 - fraction[0]
                    } else {
                        fraction[0]
                    } * if dy == 0 {
                        1.0 - fraction[1]
                    } else {
                        fraction[1]
                    } * if dz == 0 {
                        1.0 - fraction[2]
                    } else {
                        fraction[2]
                    };
                    value += weight
                        * values[self.index_wrapped(base[0] + dx, base[1] + dy, base[2] + dz)];
                }
            }
        }
        value
    }

    fn index_wrapped(&self, x: isize, y: isize, z: isize) -> usize {
        let side = self.resolution as isize;
        let (x, y, z) = (
            x.rem_euclid(side) as usize,
            y.rem_euclid(side) as usize,
            z.rem_euclid(side) as usize,
        );
        (z * self.resolution + y) * self.resolution + x
    }

    #[cfg(test)]
    fn coordinates(&self, index: usize) -> (isize, isize, isize) {
        let x = index % self.resolution;
        let yz = index / self.resolution;
        let y = yz % self.resolution;
        let z = yz / self.resolution;
        (x as isize, y as isize, z as isize)
    }
}

/// Uploads the deterministic reference texture only when its authoritative input changes.
pub struct Renderer {
    texture: TextureHandle,
    recipe: Recipe,
    field: Option<DensityField>,
    previous_density: Option<Vec<f32>>,
    snapshot_fingerprint: u64,
    field_fingerprint: u64,
    image_fingerprint: u64,
    last_tick: Option<u64>,
}

pub(super) struct Scene<'a> {
    pub positions: &'a [f32],
    pub traits: &'a [f32],
    pub extent: f32,
    pub tick: u64,
    pub camera: &'a Camera,
}

impl Renderer {
    pub fn new(ctx: &Context) -> Self {
        let image = ColorImage::filled([1, 1], Color32::TRANSPARENT);
        Renderer {
            texture: ctx.load_texture("causal-material", image, TextureOptions::LINEAR),
            recipe: Recipe::default(),
            field: None,
            previous_density: None,
            snapshot_fingerprint: 0,
            field_fingerprint: 0,
            image_fingerprint: 0,
            last_tick: None,
        }
    }

    pub fn recipe_mut(&mut self) -> &mut Recipe {
        &mut self.recipe
    }

    /// Clear temporal observation state when the viewer adopts or reseeds a world.
    pub fn reset_history(&mut self) {
        self.field = None;
        self.previous_density = None;
        self.snapshot_fingerprint = 0;
        self.field_fingerprint = 0;
        self.image_fingerprint = 0;
        self.last_tick = None;
    }

    pub fn draw(
        &mut self,
        painter: &Painter,
        viewport: Rect,
        scene: Scene<'_>,
    ) -> Result<(), Error> {
        let snapshot_fingerprint =
            snapshot_fingerprint(scene.positions, scene.traits, scene.extent);
        let field_fingerprint = field_fingerprint(snapshot_fingerprint, &self.recipe);
        let temporal_continuation = self.last_tick.is_some_and(|previous| scene.tick > previous);
        if self.field.is_none() || field_fingerprint != self.field_fingerprint {
            let field = DensityField::from_particles(
                scene.positions,
                scene.traits,
                scene.extent,
                &self.recipe,
            )?;
            self.previous_density = self
                .field
                .as_ref()
                .filter(|_| {
                    temporal_continuation && snapshot_fingerprint != self.snapshot_fingerprint
                })
                .filter(|previous| previous.density.len() == field.density.len())
                .map(|previous| previous.density.clone());
            self.field = Some(field);
            self.snapshot_fingerprint = snapshot_fingerprint;
            self.field_fingerprint = field_fingerprint;
        } else if temporal_continuation {
            self.previous_density = self.field.as_ref().map(|field| field.density.clone());
            self.image_fingerprint = 0;
        }
        self.last_tick = Some(scene.tick);
        let (image_width, image_height) = render_size(viewport);
        let image_fingerprint = image_fingerprint(
            field_fingerprint,
            &self.recipe,
            scene.camera,
            image_width,
            image_height,
            viewport.height(),
        );
        if image_fingerprint != self.image_fingerprint {
            let field = self.field.as_ref().expect("field was constructed above");
            self.texture.set(
                field.shade_camera(
                    &self.recipe,
                    scene.camera,
                    image_width,
                    image_height,
                    viewport.height(),
                    self.previous_density.as_deref(),
                ),
                TextureOptions::LINEAR,
            );
            self.image_fingerprint = image_fingerprint;
        }
        painter.image(
            self.texture.id(),
            viewport,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
        Ok(())
    }
}

pub(super) fn reference_image(
    positions: &[f32],
    traits: &[f32],
    extent: f32,
    recipe: &Recipe,
    camera: &Camera,
    width: usize,
    height: usize,
) -> Result<ColorImage, Error> {
    let field = DensityField::from_particles(positions, traits, extent, recipe)?;
    Ok(field.shade_camera(recipe, camera, width, height, height as f32, None))
}

fn kernel(unit_distance: f32) -> f32 {
    let inside = (1.0 - unit_distance * unit_distance).max(0.0);
    inside * inside * inside
}

/// Exact expected raw deposit (before normalisation) at any point, for `count` particles
/// distributed uniformly at random over a periodic box of volume `extent^3`, under this
/// module's compact kernel at the given `support` radius.
///
/// A uniform swarm of intensity `rho = count / extent^3` deposits, in expectation,
/// `rho * integral(kernel)` at every point (support is always far smaller than half the torus,
/// enforced above, so edge/wrap effects are negligible). In spherical coordinates with
/// `u = r / support`:
///
///   integral(kernel) = support^3 * 4*pi * integral_0^1 (1 - u^2)^3 * u^2 du
///
/// Expanding `(1-u^2)^3 = 1 - 3u^2 + 3u^4 - u^6` and integrating term by term gives the exact
/// rational value `integral_0^1 (1-u^2)^3 u^2 du = 1/3 - 3/5 + 3/7 - 1/9 = 16/315`, so
///
///   E[density] = rho * support^3 * 64*pi/315
///
/// with no dependence on any quantity this compact kernel doesn't already have in closed form
/// -- an analytic derivation, not a measurement, unlike `world::calibrate`'s Gaussian-shell
/// norm, which has no closed form in general dimension.
fn uniform_density(count: usize, extent: f32, support: f32) -> f32 {
    const KERNEL_INTEGRAL: f32 = 64.0 * std::f32::consts::PI / 315.0;
    let rho = count as f32 / (extent * extent * extent);
    (rho * support * support * support * KERNEL_INTEGRAL).max(EPSILON)
}

/// A bright narrow surface wrapped around a faint interior. This is an observation of the
/// density field, not geometry or motion injected independently of the simulation.
fn transfer(density: f32, iso: f32) -> (f32, f32) {
    let ratio = density / iso.max(EPSILON);
    let interior = ((ratio - 1.0) / 1.5).clamp(0.0, 1.0);
    let distance_from_surface = (ratio - 1.0) / 0.42;
    let surface = (-distance_from_surface * distance_from_surface).exp();
    ((surface + 0.10 * interior).clamp(0.0, 1.0), interior)
}

fn torus_distance(a: [f32; 3], b: &[f32], extent: f32) -> f32 {
    let mut square = 0.0;
    for axis in 0..3 {
        let mut delta = a[axis] - b[axis];
        delta -= extent * (delta / extent).round();
        square += delta * delta;
    }
    square.sqrt()
}

fn palette(trait_value: f32) -> [f32; 3] {
    let t = trait_value.clamp(0.0, 1.0);
    // Cyan, violet, magenta, and coral. Trait state chooses the color; time never does.
    const STOPS: [[f32; 3]; 4] = [
        [0.02, 0.72, 1.00],
        [0.10, 0.14, 0.62],
        [0.74, 0.03, 0.54],
        [1.00, 0.28, 0.04],
    ];
    let scaled = t * (STOPS.len() - 1) as f32;
    let left = (scaled.floor() as usize).min(STOPS.len() - 1);
    let right = (left + 1).min(STOPS.len() - 1);
    let mix = scaled - left as f32;
    [
        STOPS[left][0] + (STOPS[right][0] - STOPS[left][0]) * mix,
        STOPS[left][1] + (STOPS[right][1] - STOPS[left][1]) * mix,
        STOPS[left][2] + (STOPS[right][2] - STOPS[left][2]) * mix,
    ]
}

fn straight_alpha(red: f32, green: f32, blue: f32, alpha: f32) -> Color32 {
    let alpha = alpha.clamp(0.0, 1.0);
    let unpremultiply = 1.0 / alpha.max(EPSILON);
    Color32::from_rgba_unmultiplied(
        ((red * unpremultiply).clamp(0.0, 1.0) * 255.0) as u8,
        ((green * unpremultiply).clamp(0.0, 1.0) * 255.0) as u8,
        ((blue * unpremultiply).clamp(0.0, 1.0) * 255.0) as u8,
        (alpha * 255.0) as u8,
    )
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalise(value: [f32; 3]) -> [f32; 3] {
    let length = dot(value, value).sqrt().max(EPSILON);
    [value[0] / length, value[1] / length, value[2] / length]
}

fn snapshot_fingerprint(positions: &[f32], traits: &[f32], extent: f32) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    mix_hash(&mut hash, extent.to_bits());
    for &value in positions.iter().chain(traits) {
        mix_hash(&mut hash, value.to_bits());
    }
    hash
}

fn field_fingerprint(mut hash: u64, recipe: &Recipe) -> u64 {
    mix_hash(&mut hash, recipe.resolution as u32);
    mix_hash(&mut hash, recipe.support.to_bits());
    hash
}

fn image_fingerprint(
    mut hash: u64,
    recipe: &Recipe,
    camera: &Camera,
    width: usize,
    height: usize,
    viewport_height: f32,
) -> u64 {
    mix_hash(&mut hash, recipe.iso.to_bits());
    mix_hash(&mut hash, recipe.absorption.to_bits());
    mix_hash(&mut hash, width as u32);
    mix_hash(&mut hash, height as u32);
    mix_hash(&mut hash, viewport_height.to_bits());
    for value in [
        camera.target[0],
        camera.target[1],
        camera.target[2],
        camera.yaw,
        camera.pitch,
        camera.distance,
        camera.focal,
    ] {
        mix_hash(&mut hash, value.to_bits());
    }
    hash
}

fn render_size(viewport: Rect) -> (usize, usize) {
    let aspect = (viewport.width() / viewport.height().max(1.0)).max(0.1);
    if aspect >= 1.0 {
        (480, (480.0 / aspect).round().max(1.0) as usize)
    } else {
        ((480.0 * aspect).round().max(1.0) as usize, 480)
    }
}

fn mix_hash(hash: &mut u64, bits: u32) {
    for byte in bits.to_le_bytes() {
        *hash ^= byte as u64;
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The old raw-density uniform baseline `BRIGHTNESS_GAIN` is derived from (`2.34497`) is a
    /// property of the old fixed `support = 24.0` at this repo's default coordination and
    /// radius, not of any one particle count -- `count / extent^3` is itself invariant at fixed
    /// coordination (`world::Params::geometry`), so `uniform_density` should return the same
    /// value at wildly different `count` as long as `support` and the coordination/radius that
    /// produced `extent` are held fixed. Guards the constant `render_recipe::BRIGHTNESS_GAIN`
    /// was derived from, not a property of today's (derived, not fixed) `support`.
    #[test]
    fn old_uniform_baseline_is_particle_count_invariant() {
        const OLD_SUPPORT: f32 = 24.0;
        let baseline_at = |count: usize| {
            let params = crate::engine::genome::Params {
                particles: count,
                ..Default::default()
            };
            uniform_density(count, params.geometry().extent, OLD_SUPPORT)
        };
        let small = baseline_at(2_500);
        let large = baseline_at(50_000);
        assert!(
            (small - large).abs() / small < 0.01,
            "old uniform baseline should not depend on particle count: {small} at N=2,500 vs \
             {large} at N=50,000"
        );
        assert!(
            (small - 2.34497).abs() / 2.34497 < 0.01,
            "measured {small}, `render_recipe::BRIGHTNESS_GAIN`'s derivation comment assumes \
             2.34497 -- update both together if this drifts"
        );
    }

    /// `BRIGHTNESS_GAIN`'s one exact claim (see its doc comment): deep inside real structure,
    /// `transfer`'s interior term is clamped to a fixed floor on both sides of the density-
    /// referencing change, so `material` converges to the same value whether density and `iso`
    /// are in old raw units or new normalised ones, and the brightness ratio for a fixed
    /// physical density converges to exactly `BRIGHTNESS_GAIN` -- not approximately.
    #[test]
    fn brightness_gain_is_exact_deep_in_structure() {
        const OLD_ISO: f32 = 2.0;
        const OLD_RHO_U: f32 = 2.34497;
        const OLD_ABSORPTION: f32 = 0.40;
        let new_absorption = crate::render_recipe::RenderRecipe::default().absorption;
        assert!((new_absorption - OLD_ABSORPTION * crate::render_recipe::BRIGHTNESS_GAIN).abs()
            < 1e-3);

        // "Deep" here means far enough past both systems' surface peak that `transfer`'s
        // interior clamp has fully engaged on both sides -- `m = 8` (8x the uniform baseline)
        // clears that for both the old (peak at ratio 1, i.e. density ~2.34497) and new (peak at
        // ratio 1, i.e. density 1.5) thresholds.
        for m in [5.0f32, 8.0, 20.0] {
            let (material_old, _) = transfer(m * OLD_RHO_U, OLD_ISO);
            let (material_new, _) =
                transfer(m, crate::render_recipe::RenderRecipe::default().iso);
            let opacity_old = material_old * OLD_ABSORPTION;
            let opacity_new = material_new * new_absorption;
            assert!(
                (opacity_new / opacity_old - crate::render_recipe::BRIGHTNESS_GAIN).abs() < 1e-3,
                "m={m}: opacity ratio {} should equal BRIGHTNESS_GAIN {} deep in structure",
                opacity_new / opacity_old,
                crate::render_recipe::BRIGHTNESS_GAIN
            );
        }
    }

    #[test]
    fn zz_probe_determinism_check() {
        fn build() -> crate::engine::world::World {
            let params = crate::engine::genome::Params {
                particles: 1_200,
                ..Default::default()
            };
            let geometry = params.geometry();
            let genome = params
                .layout()
                .default_genome(params.radius, geometry.extent);
            let mut world = crate::engine::world::World::with_geometry(&params, &geometry, &genome);
            for _ in 0..1_500 {
                world.step();
            }
            world
        }
        let a = build();
        let b = build();
        eprintln!(
            "extent a={} b={}  positions equal={}  state_hash a={} b={}",
            a.geometry.extent,
            b.geometry.extent,
            a.substrate.positions == b.substrate.positions,
            a.state_hash(),
            b.state_hash()
        );
    }

    #[test]
    fn zz_probe_final() {
        let params = crate::engine::genome::Params {
            particles: 1_200,
            ..Default::default()
        };
        let geometry = params.geometry();
        let genome = params
            .layout()
            .default_genome(params.radius, geometry.extent);
        let mut world = crate::engine::world::World::with_geometry(&params, &geometry, &genome);
        for _ in 0..1_500 {
            world.step();
        }
        let recipe = Recipe::for_world(world.geometry.extent, world.substrate.len());
        let mut camera = Camera::default();
        camera.frame_world(world.geometry.extent);
        let image = reference_image(
            &world.substrate.positions,
            &world.substrate.traits,
            world.geometry.extent,
            &recipe,
            &camera,
            240,
            135,
        )
        .unwrap();
        let background = crate::viewer::theme::CANVAS;
        let peak = image
            .pixels
            .iter()
            .map(|pixel| {
                [pixel.r(), pixel.g(), pixel.b()]
                    .into_iter()
                    .zip([background.r(), background.g(), background.b()])
                    .map(|(fg, bg)| {
                        ((fg as u16 * pixel.a() as u16 + bg as u16 * (255 - pixel.a() as u16) + 127)
                            / 255) as u8
                    })
                    .max()
                    .unwrap()
            })
            .max()
            .unwrap();
        let void = image.pixels.iter().filter(|p| p.a() < 13).count() as f32
            / image.pixels.len() as f32;
        eprintln!(
            "FINAL recipe={recipe:?} peak_luminance={peak} void_fraction={void:.4}"
        );
    }

    fn recipe() -> Recipe {
        Recipe {
            resolution: 24,
            support: 2.0,
            iso: 0.1,
            absorption: 0.2,
        }
    }

    fn uniform_swarm(count: usize, extent: f32, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut rng = crate::engine::rng::Rng::new(seed);
        let positions = (0..count * 3).map(|_| rng.unit() * extent).collect();
        let traits = (0..count).map(|_| rng.unit()).collect();
        (positions, traits)
    }

    /// Coefficient of variation of `gradient()`'s magnitude, sampled at `directions` points all
    /// at the same radius `r` from a single particle at the box centre. A lone particle's true
    /// (continuous) field is exactly spherically symmetric, so the true gradient magnitude is
    /// identical at every one of those points; any spread measured here is purely a discretely-
    /// sampled-and-trilinearly-interpolated grid's own artifact (the trilinear-normal faceting
    /// `RenderRecipe::for_world`'s `GRADIENT_OVERSAMPLE` exists to shrink), not signal.
    fn gradient_facet_cv(resolution: usize, extent: f32, support: f32, radius: f32) -> f32 {
        let centre = extent / 2.0;
        let field = DensityField::from_particles(
            &[centre, centre, centre],
            &[0.5],
            extent,
            &Recipe {
                resolution,
                support,
                iso: 1.0,
                absorption: 0.4,
            },
        )
        .unwrap();
        let mut rng = crate::engine::rng::Rng::new(11);
        let magnitudes: Vec<f32> = (0..24)
            .map(|_| {
                let raw = [rng.normal(), rng.normal(), rng.normal()];
                let length = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]).sqrt();
                let direction = [raw[0] / length, raw[1] / length, raw[2] / length];
                let point = [
                    centre + radius * direction[0],
                    centre + radius * direction[1],
                    centre + radius * direction[2],
                ];
                let gradient = field.gradient(point[0], point[1], point[2]);
                (gradient[0] * gradient[0] + gradient[1] * gradient[1] + gradient[2] * gradient[2])
                    .sqrt()
            })
            .collect();
        let mean = magnitudes.iter().sum::<f32>() / magnitudes.len() as f32;
        let variance = magnitudes.iter().map(|m| (m - mean).powi(2)).sum::<f32>()
            / magnitudes.len() as f32;
        variance.sqrt() / mean.max(EPSILON)
    }

    /// Confirms `GRADIENT_OVERSAMPLE` actually removes the trilinear-normal facets, rather than
    /// assuming the Nyquist-only bound was already enough. Compares the bare field-Nyquist
    /// resolution (`extent/resolution == support/2`, one `GRADIENT_OVERSAMPLE` factor) against
    /// `for_world`'s fully oversampled resolution, both at the same physical support.
    #[test]
    fn oversampled_resolution_removes_gradient_facets() {
        let extent: f32 = 200.0;
        let support: f32 = 20.0;
        let radius = support * 0.4;
        let bare_nyquist = (2.0 * extent / support).ceil() as usize;
        let oversampled = Recipe::for_world(extent, {
            // particle_count that makes `for_world` derive exactly this `support`: invert
            // support = 2 * extent / count^(1/3).
            let spacing = support / 2.0;
            (extent / spacing).powi(3).round() as usize
        });
        assert!(
            (oversampled.support - support).abs() < 0.5,
            "test setup: derived support {} should match the fixture support {support}",
            oversampled.support
        );

        let bare_cv = gradient_facet_cv(bare_nyquist, extent, support, radius);
        let oversampled_cv = gradient_facet_cv(oversampled.resolution, extent, support, radius);

        assert!(
            bare_cv > 0.05,
            "fixture assumption failed: bare Nyquist resolution {bare_nyquist} was already \
             smooth (cv {bare_cv}), so this fixture doesn't exercise the facet artifact"
        );
        assert!(
            oversampled_cv < bare_cv * 0.35,
            "oversampling ({} -> {} voxels) did not meaningfully shrink gradient faceting: \
             cv {bare_cv} -> {oversampled_cv}",
            bare_nyquist,
            oversampled.resolution
        );
        assert!(
            oversampled_cv < 0.05,
            "oversampled gradient still shows {oversampled_cv} coefficient of variation at a \
             radius where the true field is exactly isotropic"
        );
    }

    /// The point of density-referencing `iso`: for a uniform random swarm, the fraction of the
    /// field that reads as "material" should land in roughly the same place regardless of how
    /// many particles produced it, because `iso` is a multiple of the uniform baseline rather
    /// than an absolute count. Measured at the extremes of the tested range (300 and 50,000
    /// particles, ~167x apart, using `for_world`'s own derived support/resolution): 4.33% and
    /// 5.75%. The old absolute `iso = 2.0` had no such property -- at the same default
    /// coordination and radius, a uniform swarm's raw baseline density is ~2.3 regardless of N
    /// (`count / extent^3` is invariant at fixed coordination, see `world::Params::geometry`),
    /// so the old threshold sat *below* the mean and read as "material" nearly everywhere: the
    /// uniform fog the task's diagnosis describes.
    #[test]
    fn iso_fraction_is_density_referenced_across_particle_counts() {
        let fraction_above_iso = |count: usize| {
            let params = crate::engine::genome::Params {
                particles: count,
                ..Default::default()
            };
            let extent = params.geometry().extent;
            let recipe = Recipe::for_world(extent, count);
            let (positions, traits) = uniform_swarm(count, extent, 7);
            let field =
                DensityField::from_particles(&positions, &traits, extent, &recipe).unwrap();
            let above = field.density.iter().filter(|&&d| d >= recipe.iso).count();
            above as f32 / field.density.len() as f32
        };

        let small = fraction_above_iso(300);
        let large = fraction_above_iso(50_000);

        // Neither degenerate (all-fog or all-void): real material and real voids both exist.
        assert!((0.01..0.20).contains(&small), "small-N fraction {small}");
        assert!((0.01..0.20).contains(&large), "large-N fraction {large}");
        // And close to each other despite the 167x particle-count gap -- a band of 3x comfortably
        // covers the measured 4.33%/5.75% (1.33x) with margin for a different seed, while still
        // being far tighter than the orders-of-magnitude swing an absolute threshold produces
        // across genuinely different world sizes.
        let ratio = large.max(small) / small.min(large);
        assert!(ratio < 3.0, "fractions diverged: {small} at N=300 vs {large} at N=50,000");
    }

    #[test]
    fn opposite_torus_faces_have_equal_samples() {
        let field =
            DensityField::from_particles(&[0.15, 5.0, 5.0], &[0.25], 10.0, &recipe()).unwrap();
        assert_eq!(
            field.sample(0.0, 5.0, 5.0).to_bits(),
            field.sample(10.0, 5.0, 5.0).to_bits()
        );
        assert!(
            field.sample(9.9, 5.0, 5.0) > 0.0,
            "periodic deposition missed the far face"
        );
    }

    #[test]
    fn a_particle_bridge_is_one_periodic_component() {
        let field = DensityField::from_particles(
            &[1.0, 5.0, 5.0, 2.3, 5.0, 5.0, 3.6, 5.0, 5.0],
            &[0.0, 0.5, 1.0],
            10.0,
            &recipe(),
        )
        .unwrap();
        assert_eq!(field.connected_components(0.1), 1);
    }

    #[test]
    fn separated_clusters_leave_more_than_one_component() {
        let field = DensityField::from_particles(
            &[2.0, 2.0, 2.0, 7.0, 7.0, 7.0],
            &[0.0, 1.0],
            10.0,
            &recipe(),
        )
        .unwrap();
        assert_eq!(field.connected_components(0.1), 2);
    }

    #[test]
    fn unchanged_snapshot_produces_identical_texture() {
        let field =
            DensityField::from_particles(&[3.0, 4.0, 5.0], &[0.7], 10.0, &recipe()).unwrap();
        assert_eq!(
            field.shade(&recipe(), 32, 24).pixels,
            field.shade(&recipe(), 32, 24).pixels
        );
    }

    #[test]
    fn camera_orbit_changes_observation_but_not_the_field() {
        let field =
            DensityField::from_particles(&[3.0, 4.0, 5.0], &[0.7], 10.0, &recipe()).unwrap();
        let mut camera = Camera::default();
        camera.frame_world(10.0);
        let first = field.shade_camera(&recipe(), &camera, 32, 24, 24.0, None);
        camera.yaw += 0.8;
        let second = field.shade_camera(&recipe(), &camera, 32, 24, 24.0, None);
        assert_ne!(first.pixels, second.pixels);
        assert_eq!(
            field.sample(3.0, 4.0, 5.0).to_bits(),
            field.sample(13.0, 4.0, 5.0).to_bits()
        );
    }

    #[test]
    fn activity_comes_only_from_a_changed_particle_field() {
        let field =
            DensityField::from_particles(&[3.0, 4.0, 5.0], &[0.7], 10.0, &recipe()).unwrap();
        let previous =
            DensityField::from_particles(&[4.0, 4.0, 5.0], &[0.7], 10.0, &recipe()).unwrap();
        let mut camera = Camera::default();
        camera.frame_world(10.0);
        let still = field.shade_camera(&recipe(), &camera, 32, 24, 24.0, None);
        let same = field.shade_camera(&recipe(), &camera, 32, 24, 24.0, Some(&field.density));
        let changed = field.shade_camera(&recipe(), &camera, 32, 24, 24.0, Some(&previous.density));
        assert_eq!(still.pixels, same.pixels);
        assert_ne!(still.pixels, changed.pixels);
    }

    #[test]
    fn reference_texture_preserves_the_viewport_aspect() {
        let wide = Rect::from_min_size(Pos2::ZERO, eframe::egui::vec2(1_600.0, 900.0));
        let tall = Rect::from_min_size(Pos2::ZERO, eframe::egui::vec2(900.0, 1_600.0));
        assert_eq!(render_size(wide), (480, 270));
        assert_eq!(render_size(tall), (270, 480));
    }
}
