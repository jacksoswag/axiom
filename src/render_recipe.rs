//! Versioned, simulation-independent parameters for the causal material view. Every field is
//! either derived from the world's actual geometry (resolution, support), density-referenced so
//! its value keeps one meaning at any particle count or box (iso), or genuine free visual taste
//! (absorption). Nothing here is an absolute constant tuned by eye.

use crate::tuner::genome::Caps;

pub const VERSION: u32 = 1;
pub const MAX_RESOLUTION: usize = 128;

/// support = SUPPORT_SPACING_MULTIPLE * s, where s is the mean interparticle spacing. A sphere
/// of radius 2s holds ~33 particles for a locally uniform swarm, enough for the compact deposit
/// kernel to blend nearest neighbors into a continuous surface without washing out structure.
/// Small relative to the box by construction: support shrinks as box_len / particle_count^(1/3).
const SUPPORT_SPACING_MULTIPLE: f32 = 2.0;

/// Extra resolution beyond the field-Nyquist bound (box/resolution <= support/2). The renderer
/// shades from the density gradient, and trilinear interpolation is continuous in value but not
/// in gradient, so at the bare bound the shading facets are as wide as a voxel. Oversampling by
/// k shrinks facet size to 1/k of the support radius; k = 3 puts it well below the transfer
/// band's detail, while k = 4 would clamp against MAX_RESOLUTION by 50,000 particles and
/// silently give back the oversampling it exists to provide. Cost grows as resolution^3.
const GRADIENT_OVERSAMPLE: f32 = 3.0;

/// Density-isosurface threshold in units of "times as dense as a uniform swarm", so the number
/// keeps its meaning regardless of particle count, box, or support. The exact multiple is taste.
const DEFAULT_ISO: f32 = 1.5;

/// Interior emission and absorption strength, in the density-referenced units iso establishes.
/// Visual taste, and the one value here that is. The material^2 emission coefficient in the
/// material shader is its counterpart: both scale brightness together.
const DEFAULT_ABSORPTION: f32 = 0.7034908;

#[derive(Clone, Debug, PartialEq)]
pub struct RenderRecipe {
    pub resolution: usize,
    pub support: f32, // compact density-kernel support in world coordinates
    pub iso: f32, // density isosurface, in multiples of the uniform baseline
    pub absorption: f32, // interior emission and absorption strength
}
impl RenderRecipe {
    /// Derive a recipe from the world's actual geometry instead of pinning absolute constants
    /// that mean something different at every particle count. Support follows the mean
    /// interparticle spacing; resolution is the smallest that clears the oversampled bound,
    /// clamped to MAX_RESOLUTION (past ~50,000 particles the clamp accepts some undersampling
    /// rather than growing the resolution^3 grid further). iso and absorption are size-free.
    pub fn for_world(box_len: f32, particle_count: usize) -> Self {
        let box_len = box_len.max(1e-6);
        let spacing = box_len / (particle_count.max(1) as f32).cbrt();
        let support = SUPPORT_SPACING_MULTIPLE * spacing;
        let needed = (GRADIENT_OVERSAMPLE * 2.0 * box_len / support).ceil().max(4.0) as usize;
        RenderRecipe { resolution: needed.min(MAX_RESOLUTION), support, iso: DEFAULT_ISO, absorption: DEFAULT_ABSORPTION }
    }
    pub fn valid(&self) -> bool {
        (4..=MAX_RESOLUTION).contains(&self.resolution)
            && self.support.is_finite() && self.support > 0.0
            && self.iso.is_finite() && self.iso > 0.0
            && self.absorption.is_finite() && self.absorption > 0.0
    }
}
impl Default for RenderRecipe {
    /// Derived for a reference world rather than four independent magic numbers. Runs the
    /// density probe, so default() costs milliseconds; fine at the handful of call sites.
    fn default() -> Self {
        let caps = Caps { particle_count: 320, ..Caps::default() };
        RenderRecipe::for_world(caps.probe().box_len(9.0), caps.particle_count) // 9 is the default genome's coordination
    }
}
