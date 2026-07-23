//! The runtime: what a world *is* and how it advances.
//!
//! A pure function of a flat `Vec<f32>` genome, closed under itself. Nothing here imports
//! from `tuner` or `viewer`, so the simulation cannot acquire a dependency on how it is
//! searched or drawn. Anchor count and dimensionality are read from the data, never fixed
//! by a constant.
//!
//! The dependency order is one-way: `kernel`/`rng`/`substrate`/`trait`/`grid` are leaves,
//! `genome` and `geometry` decode and derive, `interaction` builds the control net, and
//! `world` assembles all of it into the state `lenia` advances.

pub mod genome;
pub mod geometry;
pub mod grid;
pub mod interaction;
pub mod kernel;
pub mod lenia;
pub mod rng;
pub mod substrate;
pub mod r#trait;
pub mod world;
