//! AXIOM — a configurable research engine for cellular automata × graphs × ML.
//!
//! Everything downstream is a module selected and parameterized by a serializable
//! [`config::Config`]. See the crate README and the design guide for the full
//! parameter taxonomy; this slice realizes the backbone plus one measured vertical.

pub mod analysis;
pub mod config;
pub mod dynamics;
pub mod engine;
pub mod field;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod graph_ca;
pub mod growth;
pub mod hypergraph;
pub mod kernel;
pub mod loaf;
pub mod particle;
pub mod nca;
pub mod presets;
pub mod qd;
pub mod render;
pub mod rule;
pub mod substrate;

#[cfg(feature = "window")]
pub mod viz;
