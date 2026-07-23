//! Versioned, simulation-independent parameters for the causal material view.
//!
//! Every field is either derived from the world's actual geometry (`resolution`, `support`),
//! density-referenced so its value carries the same meaning at any particle count, extent or
//! support (`iso`, on the same "measure the uniform baseline, express as a multiple of it"
//! principle `world::calibrate` uses for interaction norms), or genuine free visual taste
//! (`absorption`). Nothing here is an absolute constant tuned by eye.

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

/// Units-consistency correction, not a brightness preference: how much brighter
/// material-driven shading terms (`absorption` here, and the analogous `material^2` emission
/// coefficient in `viewer::material::shade_camera`) need to be so that the SAME physical density
/// produces the same `ratio = density / iso` -- and so the same `transfer()` output -- as it did
/// under the old absolute `iso = 2.0`.
///
/// Before density-referencing, `iso = 2.0` was an absolute count implicitly compared against
/// whatever raw density the old fixed `support = 24.0` produced. That raw uniform baseline is
/// exactly `uniform_density(count, extent, 24.0)` in today's terms (see
/// `viewer::material::uniform_density`) -- measured at this repo's default coordination and
/// radius: `2.34497`, and (because `count / extent^3` is itself invariant at fixed coordination,
/// see `world::Params::geometry`) that number does not depend on particle count, so this is a
/// property of the old recipe, not of any one measurement. In today's normalised units that
/// old absolute threshold was therefore implicitly sitting at
///
///   iso_old_relative = iso_old / rho_u_old = 2.0 / 2.34497 = 0.852889
///
/// `DEFAULT_ISO = 1.5` deliberately sits somewhere else (above the uniform baseline instead of
/// just below it, which is the entire point of the fog fix), so for the same physical density
/// `ratio = density / iso` is smaller by exactly `iso_old_relative / DEFAULT_ISO`. Restoring the
/// old ratio -- and so the old brightness, for a given density -- for terms that read `ratio`
/// only through something proportional to `material` needs the reciprocal of that:
///
///   BRIGHTNESS_GAIN = DEFAULT_ISO / iso_old_relative = 1.5 * 2.34497 / 2.0 = 1.758727
///
/// What this restores exactly: deep inside real structure, `transfer`'s interior term is
/// clamped to a fixed floor regardless of how far past threshold density is (both before and
/// after this change), so `material` itself converges to the same value either way and the
/// brightness ratio converges to exactly `BRIGHTNESS_GAIN` -- not approximately, the clamp
/// makes it exact. The uniform baseline (`ratio` near 1 either way) is restored to within
/// ~10-30%, not exactly, since `transfer` is nonlinear there.
///
/// What it does not restore: `transfer`'s bright, narrow surface peak (`ratio == 1`) now sits
/// at a different physical density than before (`DEFAULT_ISO` vs `iso_old_relative`), not just
/// a rescaled one, so a single multiplicative gain cannot also realign it -- fixing that would
/// mean reshaping `transfer`'s shape constants around the new `iso` position, which is a
/// visual-design change this fix deliberately does not make. In the density band between the
/// two thresholds (roughly `0.85 < ratio_old_equivalent < 2.5`x the uniform baseline) brightness
/// can overshoot the old value by up to ~24x rather than match it -- measured, not assumed, in
/// `viewer::material::tests::brightness_gain_is_exact_deep_in_structure`. `transfer`'s own
/// clamps (`material <= 1`) keep this from ever exceeding what an unrelated, already-bright
/// surface pixel could show, so it reads as "the surface is a little hot" rather than a new
/// class of artifact; deep interior and near-empty space, which is most of a swarm's volume,
/// are unaffected by this caveat.
pub(crate) const BRIGHTNESS_GAIN: f32 = 1.758727;

/// Default interior emission/absorption strength. Genuine visual taste is `0.40`; the
/// `BRIGHTNESS_GAIN` factor is not taste, it is restoring that taste's old effective strength
/// under the new density units (see `BRIGHTNESS_GAIN`). This repo keeps taste out of derived
/// values, so `0.40` stays a bare constant -- and a candidate for the preference-learned layer
/// (`learning.rs`) -- while the units correction next to it is not.
const DEFAULT_ABSORPTION: f32 = 0.40 * BRIGHTNESS_GAIN;

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
        let needed = (GRADIENT_OVERSAMPLE * 2.0 * extent / support).ceil().max(4.0) as usize;
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
    /// The default recipe is derived for a reference world, not four independent magic
    /// numbers -- this crate's rule that geometry is derived, never hand-set, applies to
    /// the render recipe too.
    fn default() -> Self {
        let params = crate::engine::genome::Params {
            particles: 320,
            dimensions: 3,
            coordination: 9.0,
            radius: 12.0,
            rate: 10.0,
            seed: 1,
            anchors: 4,
            shells: 3,
            bumps: 2,
            trait_logits: vec![0.0; 4],
        };
        RenderRecipe::for_world(params.geometry().extent, params.particles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_recipe_is_finite_and_positive() {
        assert!(RenderRecipe::default().valid());
    }

    #[test]
    fn unbounded_volume_allocations_are_rejected() {
        let recipe = RenderRecipe {
            resolution: MAX_RESOLUTION + 1,
            ..RenderRecipe::default()
        };
        assert!(!recipe.valid());
    }

    /// Realistic extents across the tested particle-count range, via the actual
    /// coordination-derived geometry (`world::Params::geometry`), not synthetic numbers.
    fn extents_by_count() -> Vec<(usize, f32)> {
        [300usize, 1_000, 2_500, 10_000, 50_000]
            .into_iter()
            .map(|count| {
                let params = crate::engine::genome::Params {
                    particles: count,
                    ..Default::default()
                };
                (count, params.geometry().extent)
            })
            .collect()
    }

    #[test]
    fn derived_recipe_stays_valid_from_300_to_50_000_particles() {
        for (count, extent) in extents_by_count() {
            let recipe = RenderRecipe::for_world(extent, count);
            assert!(recipe.valid(), "count {count} extent {extent} -> {recipe:?}");
        }
    }

    #[test]
    fn support_tracks_spacing_not_a_fixed_fraction_of_the_box() {
        let samples = extents_by_count();
        let (small_count, small_extent) = samples[0];
        let (large_count, large_extent) = *samples.last().unwrap();
        assert!(
            large_count / small_count >= 100,
            "range too narrow to be conclusive"
        );

        let small_fraction =
            RenderRecipe::for_world(small_extent, small_count).support / small_extent;
        let large_fraction =
            RenderRecipe::for_world(large_extent, large_count).support / large_extent;

        // At fixed coordination, extent grows as particle_count^(1/3) (world::Params::
        // geometry) while support tracks the mean spacing, which stays roughly
        // particle-count-invariant at fixed coordination -- so support's share of the box
        // should shrink well past the 100x count range above, not stay pinned the way the old
        // absolute `support = 24.0` did at every particle count.
        assert!(
            small_fraction > large_fraction * 2.0,
            "support/extent barely changed: {small_fraction} at {small_count} vs \
             {large_fraction} at {large_count}"
        );
    }

    #[test]
    fn resolution_always_resolves_the_derived_support() {
        for (count, extent) in extents_by_count() {
            let recipe = RenderRecipe::for_world(extent, count);
            let voxel = extent / recipe.resolution as f32;
            assert!(
                voxel <= recipe.support / 2.0 + 1e-4,
                "count {count}: voxel {voxel} exceeds support/2 {} (field undersampled)",
                recipe.support / 2.0
            );
        }
    }

    #[test]
    fn resolution_oversamples_the_field_bound_for_gradient_shading() {
        for (count, extent) in extents_by_count() {
            let recipe = RenderRecipe::for_world(extent, count);
            let voxel = extent / recipe.resolution as f32;
            let bound = recipe.support / (2.0 * GRADIENT_OVERSAMPLE);
            // Skip counts where the MAX_RESOLUTION clamp has engaged (documented in
            // `for_world`'s ponytail note) -- the clamp intentionally gives up the oversampling
            // bound past that ceiling, so it isn't expected to hold there.
            if recipe.resolution == MAX_RESOLUTION {
                continue;
            }
            assert!(
                voxel <= bound + 1e-4,
                "count {count}: voxel {voxel} exceeds the oversampled bound {bound} \
                 (gradient under-resolved)"
            );
        }
    }
}
