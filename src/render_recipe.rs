//! Versioned, simulation-independent parameters for the causal material view.
//!
//! Every field is either derived from the world's actual geometry (`resolution`, `support`),
//! density-referenced so its value carries the same meaning at any particle count, extent or
//! support (`iso`, on the same "measure the uniform baseline, express as a multiple of it"
//! principle `world::calibrate` uses for interaction norms), or genuine free visual taste
//! (`absorption`). Nothing here is an absolute constant tuned by eye.

use crate::engine::genome::Params;

pub const VERSION: u32 = 1;
pub const MAX_RESOLUTION: usize = 128;

/// `support = SUPPORT_SPACING_MULTIPLE * s`, where `s` is the mean interparticle spacing. A
/// naive (unweighted) sphere of radius `k * s` contains an expected `(4/3)*pi*k^3` particles
/// for a locally uniform swarm; at `k = 2` that is `~33.5`, enough for the compact deposition
/// kernel -- which downweights toward the boundary, so the effectively-counted neighbourhood is
/// smaller still -- to blend a cluster's nearest few neighbours into a continuous surface
/// without washing out real structure. Small relative to the box by construction: `support`
/// shrinks as `extent / particle_count^(1/3)`, not as a fixed fraction of `extent`.
const SUPPORT_SPACING_MULTIPLE: f32 = 2.0;

/// Extra resolution beyond the field-Nyquist bound (`extent/resolution <= support/2`), needed
/// because the renderer shades from `DensityField::gradient`, not from the field value.
///
/// The field-Nyquist bound only promises the sampled VALUE is representable; at that bound a
/// support radius spans ~2 voxels, sampled by a piecewise-trilinear interpolant. Trilinear
/// interpolation is continuous in value (C0) but its gradient has a slope discontinuity at every
/// cell face (not C1): the interpolant is a separate multilinear polynomial per cell, and
/// adjacent cells generally disagree on the derivative normal to the face they share. Shading
/// normals come straight from that gradient (`normalise(gradient(..))`), so at the bare Nyquist
/// resolution the facet size equals the voxel size, which is a large fraction of the support
/// radius -- the hard polygonal facets are the visible symptom.
///
/// Oversampling by `k` shrinks the voxel size (and so the facet size) to `1/k` of what it was
/// relative to the support radius, without changing what the field itself represents. At `k = 3`
/// a support radius spans ~6 voxels instead of ~2, small enough relative to `transfer`'s surface
/// band (`0.42` in ratio units, itself a fraction of one support radius in space) that the
/// per-cell gradient discontinuity is well below the shading detail the surface band can even
/// show. Cost is `resolution^3`, so this is a 27x increase in voxel count (and proportional
/// increase in `shade_camera`'s per-pixel ray-march steps, `resolution * 2`) over the bare
/// Nyquist bound -- the reason it is a separate, explicit factor rather than folded into
/// `SUPPORT_SPACING_MULTIPLE`, and why it stays at the low end of the 2x-4x range gradient-based
/// shading typically needs: `k = 4` would clamp against `MAX_RESOLUTION` by 50,000 particles
/// (the top of the tested range), silently giving back the oversampling it exists to provide.
const GRADIENT_OVERSAMPLE: f32 = 3.0;

/// Default density-isosurface threshold, in units of "times as dense as a uniform swarm" (see
/// `RenderRecipe::iso`). Free visual taste for exactly which multiple looks best, but expressed
/// this way rather than as an absolute count so the number keeps its meaning regardless of
/// particle count, extent or support.
const DEFAULT_ISO: f32 = 1.5;

/// Default interior emission and absorption strength, in the density-referenced units `iso`
/// establishes. Visual taste, not a derived quantity, and the one value here that is. The
/// `material^2` emission coefficient in `viewer::material::shade_camera` is its counterpart:
/// both scale brightness through a term proportional to `material`, so they move together.
const DEFAULT_ABSORPTION: f32 = 0.7034908;

#[derive(Clone, Debug, PartialEq)]
pub struct RenderRecipe {
    pub resolution: usize,
    /// Compact density-kernel support in world coordinates.
    pub support: f32,
    /// Density isosurface used by the material transfer function, density-referenced: `iso =
    /// 1.5` means "1.5x as dense as a uniform swarm of the same particle count, extent and
    /// support would be" at this point, regardless of what those three values actually are.
    /// `DensityField::from_particles` divides the raw deposited field by that same uniform
    /// baseline before anything is compared against `iso`.
    pub iso: f32,
    /// Interior emission and absorption strength.
    pub absorption: f32,
}

impl RenderRecipe {
    /// Derive a recipe from the world's actual geometry, instead of pinning absolute constants
    /// that mean something different at every particle count.
    ///
    /// `support` follows the mean interparticle spacing `s = extent / particle_count^(1/3)`
    /// (volume-per-particle, cube-rooted) rather than a fraction of `extent` -- see
    /// `SUPPORT_SPACING_MULTIPLE` for the neighbour-count reasoning behind the multiple.
    ///
    /// `resolution` is undersampled (the FIELD itself is not representable) unless roughly
    /// `extent / resolution <= support / 2` (a voxel no wider than half the support); shading
    /// needs more than that, see `GRADIENT_OVERSAMPLE`. This picks the smallest resolution that
    /// clears the oversampled bound, since cost grows as `resolution^3`, and clamps to
    /// `MAX_RESOLUTION`.
    ///
    /// ponytail: for particle counts far past the tested 50,000 (`support` shrinking faster
    /// than `MAX_RESOLUTION` can follow), the clamp silently accepts some undersampling rather
    /// than growing the grid further. Raise `MAX_RESOLUTION` or move to a sparse/adaptive grid
    /// if that regime becomes real.
    ///
    /// `iso` and `absorption` do not depend on world size -- `iso` because it is
    /// density-referenced (see its doc comment) and `absorption` because it is free visual
    /// taste -- so both are fixed constants here rather than derived.
    pub fn for_world(extent: f32, particle_count: usize) -> Self {
        let extent = extent.max(1e-6);
        let spacing = extent / (particle_count.max(1) as f32).cbrt();
        let support = SUPPORT_SPACING_MULTIPLE * spacing;
        let needed = (GRADIENT_OVERSAMPLE * 2.0 * extent / support)
            .ceil()
            .max(4.0) as usize;
        RenderRecipe {
            resolution: needed.min(MAX_RESOLUTION),
            support,
            iso: DEFAULT_ISO,
            absorption: DEFAULT_ABSORPTION,
        }
    }

    pub fn valid(&self) -> bool {
        (4..=MAX_RESOLUTION).contains(&self.resolution)
            && self.support.is_finite()
            && self.support > 0.0
            && self.iso.is_finite()
            && self.iso > 0.0
            && self.absorption.is_finite()
            && self.absorption > 0.0
    }
}

impl Default for RenderRecipe {
    /// Derived for a reference world rather than four independent magic numbers: this crate's
    /// rule that geometry is derived, never hand-set, applies to the render recipe too.
    ///
    /// ponytail: deriving it means this runs the coordination probe, so `default()` costs
    /// milliseconds rather than nothing. Fine at the handful of call sites that exist; cache a
    /// `DensityProbe` here if one ever lands in a loop.
    fn default() -> Self {
        let params = Params {
            particles: 320,
            ..Default::default()
        };
        RenderRecipe::for_world(crate::tuner::density::bound_len_for(&params), params.particles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `for_world` is the only non-trivial thing in this file: `resolution` must clear the
    /// oversampled Nyquist bound at every tested particle count, or the shading facets
    /// `GRADIENT_OVERSAMPLE` exists to prevent come back.
    #[test]
    fn a_derived_recipe_is_valid_and_resolves_its_own_support() {
        for count in [300usize, 1_000, 2_500, 10_000, 50_000] {
            let params = Params {
                particles: count,
                ..Default::default()
            };
            let extent = crate::tuner::density::bound_len_for(&params);
            let recipe = RenderRecipe::for_world(extent, count);
            assert!(recipe.valid(), "count {count} -> {recipe:?}");
            let voxel = extent / recipe.resolution as f32;
            assert!(
                recipe.resolution == MAX_RESOLUTION
                    || voxel <= recipe.support / (2.0 * GRADIENT_OVERSAMPLE) * 1.001,
                "count {count}: voxel {voxel} undersamples support {}",
                recipe.support
            );
        }
    }

    #[test]
    fn an_unbounded_volume_is_rejected() {
        assert!(!RenderRecipe {
            resolution: MAX_RESOLUTION + 1,
            ..RenderRecipe::default()
        }
        .valid());
        assert!(RenderRecipe::default().valid());
    }
}
