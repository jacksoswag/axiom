//! Binary admission and promotion gates. These clauses never become a scalar fitness.
//!
//! Thresholds arrive as [`Gates`] rather than constants, and every threshold feeds
//! `Tuning::digest`, so evidence measured under different gates cannot merge silently.

use crate::tuner::metrics::Metrics;
use crate::tuner::tuning::Gates;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rejection {
    Dead,
    Dispersed,
    Frozen,
    Collapsed,
    Fragile,
}
impl Rejection {
    pub const ALL: [Self; 5] = [
        Self::Dead,
        Self::Dispersed,
        Self::Frozen,
        Self::Collapsed,
        Self::Fragile,
    ];
    pub fn label(self) -> &'static str {
        match self {
            Self::Dead => "dead",
            Self::Dispersed => "dispersed",
            Self::Frozen => "frozen",
            Self::Collapsed => "collapsed",
            Self::Fragile => "fragile",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorldRejection {
    Homogeneous,
    MaterialDisconnected,
    VoidDisconnected,
}
impl WorldRejection {
    pub const ALL: [Self; 3] = [
        Self::Homogeneous,
        Self::MaterialDisconnected,
        Self::VoidDisconnected,
    ];
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
    let density_contrast = metrics
        .heterogeneity
        .chunks_exact(5)
        .map(|scale| scale[0])
        .fold(0.0, f32::max);
    let trait_contrast = metrics
        .heterogeneity
        .chunks_exact(5)
        .map(|scale| scale[2])
        .fold(0.0, f32::max);
    GateMargins {
        structure_floor: metrics.structure - gates.structure_floor,
        structure_ceiling: gates.structure_ceiling - metrics.structure,
        mobility: metrics.mobility - gates.mobility_floor,
        temporal: metrics.temporal_variance - gates.variance_floor,
        repair: metrics.robustness - gates.robustness_floor,
        heterogeneity: density_contrast.min(trait_contrast) - gates.heterogeneity_floor,
        material: metrics.connectivity.dense.into_iter().fold(1.0, f32::min)
            - gates.connected_fraction_floor,
        void: metrics.connectivity.void.into_iter().fold(1.0, f32::min)
            - gates.connected_fraction_floor,
    }
}

pub fn viable(metrics: &Metrics, gates: &Gates) -> Result<(), Rejection> {
    if !metrics.alive || metrics.descriptor.iter().any(|v| !v.is_finite()) {
        return Err(Rejection::Dead);
    }
    let margin = margins(metrics, gates);
    if margin.structure_floor <= 0.0 {
        return Err(Rejection::Dispersed);
    }
    if margin.structure_ceiling <= 0.0 {
        return Err(Rejection::Collapsed);
    }
    if margin.mobility <= 0.0 || margin.temporal <= 0.0 {
        return Err(Rejection::Frozen);
    }
    if margin.repair < 0.0 {
        return Err(Rejection::Fragile);
    }
    Ok(())
}

/// Promotion-only clauses: persistent local regimes plus connected material and flight void.
pub fn world_qualified(metrics: &Metrics, gates: &Gates) -> Result<(), WorldRejection> {
    let margin = margins(metrics, gates);
    if margin.heterogeneity <= 0.0 {
        return Err(WorldRejection::Homogeneous);
    }
    if margin.material <= 0.0 {
        return Err(WorldRejection::MaterialDisconnected);
    }
    if margin.void <= 0.0 {
        return Err(WorldRejection::VoidDisconnected);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuner::metrics::{descriptor_len, Connectivity};

    fn living() -> Metrics {
        Metrics {
            mobility: 0.02,
            temporal_variance: 0.05,
            autocorrelation: [0.9, 0.7, 0.4],
            turnover: 0.1,
            robustness: 1.0,
            structure: 0.5,
            heterogeneity: [0.2; 15],
            connectivity: Connectivity {
                dense: [1.0; 2],
                void: [1.0; 2],
            },
            descriptor: vec![1.0; descriptor_len()],
            alive: true,
        }
    }

    #[test]
    fn base_gate_keeps_all_five_named_rejections() {
        let gates = Gates::default();
        let cases = [
            (
                Metrics {
                    alive: false,
                    ..living()
                },
                Rejection::Dead,
            ),
            (
                Metrics {
                    structure: 0.001,
                    ..living()
                },
                Rejection::Dispersed,
            ),
            (
                Metrics {
                    structure: 1.4,
                    ..living()
                },
                Rejection::Collapsed,
            ),
            (
                Metrics {
                    mobility: 0.0,
                    ..living()
                },
                Rejection::Frozen,
            ),
            (
                Metrics {
                    robustness: 0.0,
                    ..living()
                },
                Rejection::Fragile,
            ),
        ];
        for (metric, rejection) in cases {
            assert_eq!(viable(&metric, &gates), Err(rejection));
        }
    }

    #[test]
    fn promotion_clauses_are_named_and_not_scalar_fitness() {
        let gates = Gates::default();
        assert_eq!(
            world_qualified(
                &Metrics {
                    heterogeneity: [0.0; 15],
                    ..living()
                },
                &gates
            ),
            Err(WorldRejection::Homogeneous)
        );
        assert_eq!(
            world_qualified(
                &Metrics {
                    connectivity: Connectivity {
                        dense: [0.2; 2],
                        void: [1.0; 2]
                    },
                    ..living()
                },
                &gates
            ),
            Err(WorldRejection::MaterialDisconnected)
        );
        assert_eq!(
            world_qualified(
                &Metrics {
                    connectivity: Connectivity {
                        dense: [1.0; 2],
                        void: [0.2; 2]
                    },
                    ..living()
                },
                &gates
            ),
            Err(WorldRejection::VoidDisconnected)
        );
    }

    #[test]
    fn frozen_gas_collapse_and_fragility_remain_distinct() {
        let gates = Gates::default();
        for (metric, rejection) in [
            (
                Metrics {
                    temporal_variance: 0.0,
                    ..living()
                },
                Rejection::Frozen,
            ),
            (
                Metrics {
                    structure: 0.0,
                    ..living()
                },
                Rejection::Dispersed,
            ),
            (
                Metrics {
                    structure: 2.0,
                    ..living()
                },
                Rejection::Collapsed,
            ),
            (
                Metrics {
                    robustness: 0.3,
                    ..living()
                },
                Rejection::Fragile,
            ),
        ] {
            assert_eq!(viable(&metric, &gates), Err(rejection));
        }
    }

    #[test]
    fn a_single_uniform_blob_fails_where_a_bicontinuous_biome_passes() {
        let gates = Gates::default();
        let mut single_blob = living();
        for scale in single_blob.heterogeneity.chunks_exact_mut(5) {
            scale[0] = 1.0;
            scale[2] = 0.0;
        }
        assert_eq!(
            world_qualified(&single_blob, &gates),
            Err(WorldRejection::Homogeneous)
        );
        assert_eq!(world_qualified(&living(), &gates), Ok(()));
    }

    /// A gate is data now, so raising a floor must change the verdict on a fixed world.
    #[test]
    fn raising_a_floor_rejects_a_world_that_passed_the_default() {
        let world = living();
        assert_eq!(viable(&world, &Gates::default()), Ok(()));
        let strict = Gates {
            robustness_floor: 1.5,
            ..Gates::default()
        };
        assert_eq!(viable(&world, &strict), Err(Rejection::Fragile));
    }
}
