//! Continuous-trait Particle-Lenia world search.
//!
//! `x_i(t + dt) = x_i(t) − dt · ∇_{x_i} E_i`, `E_i = −Σ_int w·G(U/norm)`.
//!
//! Two halves, and the dependency runs one way. [`engine`] is the runtime: a pure function
//! of a flat `Vec<f32>` genome, importing nothing from the other half. [`tuner`] measures,
//! scores, searches, and records the worlds the engine produces. Neither half depends on the
//! viewer.
//!
//! [`render_recipe`] sits between them rather than inside either: it derives how a world is
//! drawn, which is not something the engine advances nor something the tuner scores. It stays
//! outside the `viewer` feature because a headless checkpoint still records one.
//!
//! Selection is novelty and admission is a binary gate. Expensive persistence models only
//! allocate rollout effort; they cannot admit discoveries or certify worlds.

pub mod engine;
pub mod render_recipe;
pub mod tuner;
pub mod util;

#[cfg(feature = "viewer")]
pub mod viewer;
