//! Binary admission and promotion gates. These clauses never become a scalar fitness.
//! Thresholds arrive as Gates rather than constants, and every threshold feeds Tuning::digest,
//! so evidence measured under different gates cannot merge silently.

use crate::tuner::metrics::Metrics;
use crate::tuner::tuning::Gates;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rejection { Dead, Dispersed, Frozen, Collapsed, Fragile }
impl Rejection {
    pub const ALL: [Self; 5] = [Self::Dead, Self::Dispersed, Self::Frozen, Self::Collapsed, Self::Fragile];
    pub fn label(self) -> &'static str {
        match self {
            Self::Dead => "dead", Self::Dispersed => "dispersed", Self::Frozen => "frozen",
            Self::Collapsed => "collapsed", Self::Fragile => "fragile",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorldRejection { Homogeneous, MaterialDisconnected, VoidDisconnected }
impl WorldRejection {
    pub const ALL: [Self; 3] = [Self::Homogeneous, Self::MaterialDisconnected, Self::VoidDisconnected];
}

/// Signed clause margins make labels useful to learning without changing any gate into a
/// score. Positive means the corresponding clause has room; negative means it failed.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GateMargins {
    pub structure_floor: f32,
    pub structure_ceiling: f32,
    pub mobility: f32,
    pub temporal: f32,
    pub repair: f32,
    pub heterogeneity: f32,
    pub material: f32,
    pub void: f32,
}

pub fn margins(metrics: &Metrics, gates: &Gates) -> GateMargins {
    let density_contrast = metrics.heterogeneity.chunks_exact(5).map(|scale| scale[0]).fold(0.0, f32::max);
    let trait_contrast = metrics.heterogeneity.chunks_exact(5).map(|scale| scale[2]).fold(0.0, f32::max);
    GateMargins {
        structure_floor: metrics.structure - gates.structure_floor,
        structure_ceiling: gates.structure_ceiling - metrics.structure,
        mobility: metrics.mobility - gates.mobility_floor,
        temporal: metrics.temporal_variance - gates.variance_floor,
        repair: metrics.robustness - gates.robustness_floor,
        heterogeneity: density_contrast.min(trait_contrast) - gates.heterogeneity_floor,
        material: metrics.connectivity.dense.into_iter().fold(1.0, f32::min) - gates.connected_fraction_floor,
        void: metrics.connectivity.void.into_iter().fold(1.0, f32::min) - gates.connected_fraction_floor,
    }
}

/// The base gate, first failing clause named. Clause order is part of the contract: a world
/// that is both dispersed and frozen reports Dispersed everywhere, always.
pub fn viable(metrics: &Metrics, gates: &Gates) -> Result<(), Rejection> {
    if !metrics.alive || metrics.descriptor.iter().any(|v| !v.is_finite()) { return Err(Rejection::Dead); }
    let margin = margins(metrics, gates);
    if margin.structure_floor <= 0.0 { return Err(Rejection::Dispersed); }
    if margin.structure_ceiling <= 0.0 { return Err(Rejection::Collapsed); }
    if margin.mobility <= 0.0 || margin.temporal <= 0.0 { return Err(Rejection::Frozen); }
    if margin.repair < 0.0 { return Err(Rejection::Fragile); } // 0 exactly passes: undamaged control counts
    Ok(())
}

/// Promotion-only clauses: persistent local regimes plus connected material and flight void.
pub fn world_qualified(metrics: &Metrics, gates: &Gates) -> Result<(), WorldRejection> {
    let margin = margins(metrics, gates);
    if margin.heterogeneity <= 0.0 { return Err(WorldRejection::Homogeneous); }
    if margin.material <= 0.0 { return Err(WorldRejection::MaterialDisconnected); }
    if margin.void <= 0.0 { return Err(WorldRejection::VoidDisconnected); }
    Ok(())
}
