//! The configuration tree (§2, §6) — the research surface.
//!
//! Every experiment is one serde-loadable `Config`. The tagged `rule` / `init` /
//! `analysis` enums *are* the "string → constructor registry" the design guide
//! describes, done idiomatically. `validate()` enforces capability constraints at
//! load time so invalid module combinations fail loudly, not silently (§9).

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub name: String,
    #[serde(default = "one_u32")]
    pub schema_version: u32,
    pub substrate: SubstrateConfig,
    pub rule: RuleConfig,
    #[serde(default)]
    pub init: InitConfig,
    #[serde(default)]
    pub render: RenderConfig,
    #[serde(default)]
    pub analysis: Vec<AnalysisConfig>,
    #[serde(default)]
    pub steps: Option<u64>,
    #[serde(default)]
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstrateConfig {
    /// "grid" is the only realized grid substrate; graph lives in the graph demo.
    #[serde(default = "grid_kind")]
    pub kind: String,
    pub width: usize,
    pub height: usize,
    #[serde(default = "one_usize")]
    pub channels: usize,
    /// "torus" (periodic) or "bounded".
    #[serde(default = "torus_topology")]
    pub topology: String,
}

impl SubstrateConfig {
    pub fn torus(&self) -> bool {
        self.topology != "bounded"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RuleConfig {
    Lenia(LeniaConfig),
    /// Relaxation-toward-target Lenia — smoother and more stable (§2.4).
    AsymptoticLenia(LeniaConfig),
    /// Mass-conserving Lenia via reintegration advection (§2.4).
    FlowLenia(FlowLeniaConfig),
    GrayScott(GrayScottConfig),
    /// Rule = a small neural net over a perception vector (§2.4, the ML integration).
    Nca(NcaConfig),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeniaConfig {
    #[serde(default = "dt_default")]
    pub dt: f32,
    #[serde(default = "zero_f32")]
    pub clamp_lo: f32,
    #[serde(default = "one_f32")]
    pub clamp_hi: f32,
    pub kernels: Vec<KernelConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelConfig {
    #[serde(default)]
    pub source: usize,
    #[serde(default)]
    pub target: usize,
    pub radius: usize,
    #[serde(default = "gauss_ring_core")]
    pub core: String,
    #[serde(default = "beta_default")]
    pub beta: Vec<f32>,
    #[serde(default = "half_f32")]
    pub core_mu: f32,
    #[serde(default = "core_sigma_default")]
    pub core_sigma: f32,
    #[serde(default = "one_f32")]
    pub weight: f32,
    pub growth: GrowthConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowLeniaConfig {
    #[serde(flatten)]
    pub base: LeniaConfig,
    /// Advection strength up the potential gradient.
    #[serde(default = "one_f32")]
    pub flow: f32,
    /// Concentration-gradient term that resists total collapse.
    #[serde(default = "flow_conc_default")]
    pub concentration: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NcaConfig {
    /// Hidden layer width of the per-cell MLP.
    #[serde(default = "nca_hidden")]
    pub hidden: usize,
    /// Stochastic update probability (async NCA).
    #[serde(default = "half_f32")]
    pub update_rate: f32,
    /// Seed for random weight init when `weights` is absent.
    #[serde(default)]
    pub weight_seed: u64,
    /// Trained weights (flattened), if any. Absent → random init.
    #[serde(default)]
    pub weights: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrowthConfig {
    #[serde(default = "gauss_growth")]
    pub kind: String,
    pub mu: f32,
    pub sigma: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrayScottConfig {
    #[serde(default = "gs_dt")]
    pub dt: f32,
    #[serde(default = "gs_du")]
    pub du: f32,
    #[serde(default = "gs_dv")]
    pub dv: f32,
    pub feed: f32,
    pub kill: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InitConfig {
    Zeros,
    Random { #[serde(default = "density_default")] density: f32 },
    /// The Orbium glider — the reference oracle (§7.1).
    Orbium { #[serde(default)] dx: i32, #[serde(default)] dy: i32 },
    /// `count` gaussian blobs of the given radius.
    Blobs { count: usize, #[serde(default = "blob_radius")] radius: f32 },
    /// `count` Orbium gliders on a jittered grid — persistent, moving organisms
    /// (unlike lone blobs, which dissipate) for the detection / PageRank observers.
    OrbiumSwarm { count: usize },
    /// Gray-Scott seed: a centered square of `v`.
    GrayScottSeed,
}

impl Default for InitConfig {
    fn default() -> Self {
        InitConfig::Zeros
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderConfig {
    #[serde(default = "viridis_cmap")]
    pub colormap: String,
    #[serde(default)]
    pub channel: usize,
    #[serde(default = "one_f32")]
    pub scale: f32,
}

impl Default for RenderConfig {
    fn default() -> Self {
        RenderConfig { colormap: viridis_cmap(), channel: 0, scale: 1.0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnalysisConfig {
    /// Physical descriptors: mass and circular centroid per channel.
    Descriptors,
    /// Connected-component detection above a threshold.
    Detect { #[serde(default = "detect_threshold")] threshold: f32 },
    /// PageRank centrality over the detected-organism interaction graph.
    PageRank {
        #[serde(default = "detect_threshold")] threshold: f32,
        #[serde(default = "link_radius_default")] link_radius: f32,
    },
}

impl Config {
    pub fn load(path: &Path) -> Result<Config> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let cfg: Config = match path.extension().and_then(|e| e.to_str()) {
            Some("json") => serde_json::from_str(&text).context("parsing JSON config")?,
            _ => serde_yaml::from_str(&text).context("parsing YAML config")?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn to_yaml(&self) -> Result<String> {
        Ok(serde_yaml::to_string(self)?)
    }

    /// Capability constraints — invalid module combinations fail here (§9).
    pub fn validate(&self) -> Result<()> {
        let s = &self.substrate;
        if s.width == 0 || s.height == 0 {
            bail!("substrate size must be positive");
        }
        if s.channels == 0 {
            bail!("substrate must have at least one channel");
        }
        match &self.rule {
            RuleConfig::Lenia(l) | RuleConfig::AsymptoticLenia(l) => check_lenia(l, s)?,
            RuleConfig::FlowLenia(f) => check_lenia(&f.base, s)?,
            RuleConfig::GrayScott(_) => {
                if s.channels != 2 {
                    bail!("gray_scott requires exactly 2 channels (u, v), got {}", s.channels);
                }
            }
            RuleConfig::Nca(n) => {
                if n.hidden == 0 {
                    bail!("nca hidden width must be > 0");
                }
                if s.channels < 1 {
                    bail!("nca needs at least one channel");
                }
            }
        }
        if self.render.channel >= s.channels {
            bail!("render.channel {} out of range", self.render.channel);
        }
        Ok(())
    }

    /// A stable content hash for provenance (§2.10). Not cryptographic — just a
    /// reproducibility tag over the serialized config.
    pub fn provenance_hash(&self) -> u64 {
        let text = serde_json::to_string(self).unwrap_or_default();
        let mut h: u64 = 0xcbf29ce484222325;
        for b in text.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
}

fn check_lenia(l: &LeniaConfig, s: &SubstrateConfig) -> Result<()> {
    let min_dim = s.width.min(s.height);
    if l.dt <= 0.0 {
        bail!("lenia dt must be > 0");
    }
    if l.kernels.is_empty() {
        bail!("lenia needs at least one kernel");
    }
    for (i, k) in l.kernels.iter().enumerate() {
        if k.source >= s.channels || k.target >= s.channels {
            bail!(
                "kernel {i} references channel outside 0..{} (source {}, target {})",
                s.channels, k.source, k.target
            );
        }
        if k.radius == 0 {
            bail!("kernel {i} radius must be > 0");
        }
        if 2 * k.radius + 1 > min_dim {
            bail!("kernel {i} radius {} too large for {}x{} grid", k.radius, s.width, s.height);
        }
    }
    Ok(())
}

// serde defaults ---------------------------------------------------------------
fn one_u32() -> u32 { 1 }
fn one_usize() -> usize { 1 }
fn zero_f32() -> f32 { 0.0 }
fn one_f32() -> f32 { 1.0 }
fn half_f32() -> f32 { 0.5 }
fn dt_default() -> f32 { 0.1 }
fn core_sigma_default() -> f32 { 0.15 }
fn beta_default() -> Vec<f32> { vec![1.0] }
fn grid_kind() -> String { "grid".into() }
fn torus_topology() -> String { "torus".into() }
fn gauss_ring_core() -> String { "gauss_ring".into() }
fn gauss_growth() -> String { "gauss".into() }
fn viridis_cmap() -> String { "viridis".into() }
fn density_default() -> f32 { 0.5 }
fn blob_radius() -> f32 { 12.0 }
fn detect_threshold() -> f32 { 0.15 }
fn link_radius_default() -> f32 { 60.0 }
fn gs_dt() -> f32 { 1.0 }
fn gs_du() -> f32 { 0.16 }
fn gs_dv() -> f32 { 0.08 }
fn flow_conc_default() -> f32 { 0.35 }
fn nca_hidden() -> usize { 96 }
