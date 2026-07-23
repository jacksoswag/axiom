//! Everything that measures, scores, searches, and records worlds the engine runs.
//!
//! This half treats a genome as the thing to explore. It depends on `engine` in one
//! direction only. Descriptor versioning, viability gates, and the three-dimensional pin in
//! `metrics` are contracts of this half, not facts about the physics.

pub mod archive;
pub mod campaign;
pub mod checkpoint;
pub mod learning;
pub mod metrics;
pub mod novelty;
pub mod persistence;
pub mod rollout;
pub mod search;
pub mod tuning;
pub mod viability;
