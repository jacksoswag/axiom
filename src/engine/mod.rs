//! Runtime engine: what a simulation is and how it advances, a function of a flat Vec<f32> genome.
//! Never imports from tuner or viewer, so the simulation cannot acquire a dependency on how it
//! is searched or drawn. Anchor count and dimensionality are read from the data, not a constant.

pub mod matrix;
pub mod kernel;
pub mod lenia;
pub mod params;
pub mod resolve;
pub mod sim;
pub mod substrate;
pub mod r#trait;
