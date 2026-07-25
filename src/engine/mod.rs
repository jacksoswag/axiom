//! Runtime engine: the heart of the cellular automata simulation, a function of a flat Vec<f32> genome.
//! Never imports from tuner or harness, so the simulation cannot acquire a dependency on how it's searched 
//! or drawn. Anchor count and dimensionality are read from the data, not a constant.

pub mod matrix;
pub mod gpu;
pub mod kernel;
pub mod lenia;
pub mod params;
pub mod resolve;
pub mod sim;
pub mod substrate;
pub mod r#trait;
