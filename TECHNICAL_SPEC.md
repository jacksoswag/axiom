# AXIOM: Technical Specification

## 1. System

AXIOM is a CPU-first Rust engine for cellular automata, swarms, and graph-dynamical systems,
with an optional wgpu compute path. A single serde-loadable `Config` selects and parameterizes
a trait object at every stage of the update loop. The engine (`engine.rs`) is a thin driver:
it reads the tagged config enums, instantiates the matching modules, and steps them with
ping-pong buffers.

Around 4,400 lines of Rust across 21 modules. Zero warnings across all feature combinations.

## 2. Design invariants

1. **Config is the interface.** Every experiment is one `Config` plus a seed. The provenance
   hash (`config.rs`, FNV-1a over the serialized config) tags each run.
2. **Fail at load, not at runtime.** `Config::validate()` enforces capability constraints
   (channel counts, kernel radius versus grid size, positive `dt`) before instantiation.
3. **Ping-pong state.** A rule reads the current field and writes a scratch field; the two are
   swapped. No rule reads the buffer it writes.
4. **Specialize the hot loop, generalize the trait side.** Grid Lenia is a direct rayon-parallel
   convolution; the graph path is a generic `aggregate(neighbors) → update`. They share the
   concept, not the inner loop.
5. **The CPU path is the oracle.** The GPU path is validated against it to f32 precision, and
   the Orbium glider validates the CPU path against a known attractor.

## 3. State model

`Field` (`field.rs`) is a flat `Vec<f32>` of `C · H · W`, laid out `[channel][row][col]`. Flat
storage keeps the convolution trivially rayon-parallel. Scalar Lenia is `C = 1`; Gray-Scott is
`C = 2`; the NCA uses `C` visible-plus-hidden channels.

## 4. Closed vocabularies

Each is a tagged enum resolved at config load.

| Vocabulary | Count | Values |
|---|---|---|
| Rule types | 5 | `lenia`, `asymptotic_lenia`, `flow_lenia`, `gray_scott`, `nca` |
| Kernel cores | 3 | `gauss_ring`, `poly`, `step` |
| Growth functions | 3 | `gauss`, `poly`, `exp` |
| Init patterns | 6 | `zeros`, `random`, `orbium`, `blobs`, `orbium_swarm`, `gray_scott_seed` |
| Analysis observers | 3 | `descriptors`, `detect`, `pagerank` |
| Colormaps | 4 | `viridis`, `turbo`, `magma`, `gray` |
| Substrate topologies | 2 | `torus`, `bounded` |
| Bundled presets | 7 | `orbium`, `soup`, `life`, `multiring`, `asymptotic`, `flow`, `gray_scott` |

## 5. Rules

- **Lenia** (`rule.rs`): multi-kernel, multi-channel. `A' = clip(A + dt · Σ wᵢ Gᵢ(Kᵢ * A))`.
  Classic scalar Lenia is the one-kernel, one-channel case.
- **Asymptotic Lenia**: relaxes toward a `[0,1]` target, `A' = clip(A + dt · (T(U) − A))`.
  Smoother and more stable.
- **Flow Lenia**: mass-conserving. Mass is advected along `flow · ∇U − concentration · ∇A` by
  reintegration tracking (bilinear splat, displacement clamped below one cell). Total mass is
  conserved by construction.
- **Gray-Scott**: two-channel reaction-diffusion with a 9-point Laplacian.
- **NCA** (`nca.rs`): the rule is a per-cell MLP over a 4-filter perception (identity, Sobel-x,
  Sobel-y, Laplacian) per channel, `hidden`-wide, relu. Weights come from training or random
  init. Synchronous, or stochastic (async) when `update_rate < 1`. Trainable by analytic
  backprop with Adam or by evolution strategies (§8).

## 6. Substrates and the seam

`trait Substrate` (`substrate.rs`) exposes `node_count` and `neighbors`. Realizations:

- **Grid** as a graph (4-neighborhood), for treating a lattice as a graph to analyze.
- **Small-world graph** (Watts-Strogatz ring, rewire probability `p`).
- **Particle swarm** (`particle.rs`): continuous-position particles on a torus with three modes.
  `lenia` descends the Particle-Lenia energy `E = R − G(U)` (a growth field over the kernel
  potential, minus short-range repulsion); `nca` maps per-particle features (`U`, `|∇U|`,
  neighbor count) through a small MLP to a velocity; `swarm` is cohesion-separation. All expose
  a proximity interaction graph.
- **Hypergraph** (`hypergraph.rs`): n-ary hyperedges, a hypergraph CA (nodes aggregate incident
  hyperedges), and hypergraph PageRank via the two-step random walk (node → hyperedge → node).

`mp_step` runs one generic message-passing step over any `Substrate`.

## 7. Analysis

`trait Observer` (`analysis.rs`) runs over live state.

- **Descriptors**: per-channel mass and circular-mean centroid (torus-aware).
- **Detection**: 4-connected components above a threshold, each with mass, centroid, bounding box.
- **PageRank**: power iteration with dangling-mass handling, over the organism interaction graph
  (edges between detected components within a link radius), the substrate graph, or a hypergraph.

Deeper descriptors live in `dynamics.rs`: spectral (dominant period via a direct DFT of a
time-series), dynamical (per-step activity, value-histogram entropy), topological (H0 persistent
homology by superlevel-set union-find), organism tracking (nearest-centroid matching into
persistent lineages), and an append-only experiment log with in-Rust nearest-neighbor similarity
search over descriptor vectors.

## 8. Discovery, learning, and time

- **MAP-Elites** (`qd.rs`): searches Lenia `(μ, σ)`; behavior descriptor is final mass ×
  mobility (centroid path length); fitness is field variance (structuredness). Batched
  evaluation, rayon-parallel.
- **NCA training** (`nca.rs`): the default trainer is analytic backprop with Adam, minimizing
  one-step MSE against a target rule over a batch of on-distribution states; evolution strategies
  (antithetic finite difference) are the gradient-free alternative. `rollout_error` reports
  prediction MSE as a function of horizon (the world-model metric).
- **Spacetime-loaf** (`loaf.rs`): minimizes `E = Σ ‖V[t+1] − F(V[t])‖²` over the volume with the
  endpoints fixed. `F` is a reaction-diffusion step `F(u) = u + dt·D·∇²u + dt·r·u(1−u)`. With
  `r = 0` (linear) the energy is convex and relaxation recovers the interior; with `r > 0`
  (nonlinear) the gradient uses the analytic vector-Jacobian product `J·δ = δ + dt·D·∇²δ +
  dt·r·(1−2u)·δ`, the energy is non-convex, and convergence is not guaranteed. `interior_error`
  measures reconstruction against ground truth.

## 9. GPU compute

`gpu.rs` builds a headless wgpu pipeline (no window/surface) for single-kernel, gauss-growth
Lenia. Each 16×16 workgroup cooperatively loads its cell block plus an `R`-wide halo into a
`48×48` workgroup-memory tile (radius up to 16), then every thread convolves out of that shared
tile, collapsing the redundant global reads a direct gather makes. State persists on-device
across steps (`upload` / `advance` / `read`); all dispatches record into one encoder per call.
This is the **default execution path** for eligible configs: `headless` runs it unless the config
is multi-channel, non-torus, or a non-Lenia rule, or `--cpu` is passed. Requires the `gpu` feature
(`wgpu`, `pollster`, `bytemuck`).

## 10. Module map

| Module | Lines | Responsibility |
|---|---|---|
| `main.rs` | 815 | CLI: 14 subcommands, dispatch, per-command drivers |
| `nca.rs` | 367 | learned rule, perception, Adam + ES trainers, rollout error |
| `presets.rs` | 321 | 7 presets, the Orbium cells, field seeding |
| `config.rs` | 303 | the config tree, tagged enums, `validate()`, provenance hash |
| `rule.rs` | 291 | `trait Rule`, Lenia, asymptotic, flow, Gray-Scott |
| `gpu.rs` | 283 | tiled wgpu compute pipeline, persistent stepping, CPU-diff harness |
| `dynamics.rs` | 277 | spectral / dynamical / H0 descriptors, tracking, experiment log |
| `analysis.rs` | 264 | observers, connected components, PageRank, interaction graph |
| `particle.rs` | 228 | Particle-Lenia energy, particle-NCA, swarm, proximity graph |
| `viz.rs` | 166 | live minifb window, controls, live parameter tuning |
| `substrate.rs` | 166 | `trait Substrate`, grid/graph, message passing, RNG |
| `loaf.rs` | 155 | spacetime volume, linear/nonlinear operator, VJP relaxation |
| `hypergraph.rs` | 140 | hyperedges, hypergraph CA, hypergraph PageRank |
| `kernel.rs` | 133 | kernel cores, construction, toroidal convolution |
| `render.rs` | 130 | colormaps, field → RGB / ARGB, PNG, matrix render |
| `qd.rs` | 126 | MAP-Elites archive, evaluation, insertion |
| `engine.rs` | 83 | the driver: config → modules, step, observe, rebuild |
| `field.rs` | 58 | flat multi-channel state |
| `graph_ca.rs` | 51 | graph-Lenia run, spacetime, PageRank |
| `growth.rs` | 44 | growth functions |
| `lib.rs` | 28 | module tree |

## 11. Validation methodology and results

`axiom validate` runs each subsystem and asserts a measurable property. Measured values:

| Check | Property | Result |
|---|---|---|
| Orbium oracle | alive (bounded mass) and moving (path length) | mass 75.1 → 72.8, path 234 cells |
| Gray-Scott | patterns form, not blank or saturated | mean(v) = 0.227 |
| Detection + PageRank | ≥2 organisms, valid ranking | 18 organisms, PR sum 1.000 |
| Graph seam | valid PageRank, hubs exist | PR sum 1.000, hub/leaf 2.1× |
| Flow Lenia | mass conserved | drift 1.5e-9 |
| Asymptotic Lenia | alive and bounded | mean activation 0.209 |
| GPU vs CPU | one step matches to f32 | max abs diff 2.4e-7 (Apple M3) |

Other measured figures: NCA imitation loss drops ~50× over 300 Adam epochs (~8× with
evolution strategies); linear loaf interior reconstruction error drops ~1400× and the nonlinear
(Fisher-KPP) case ~40×; tiled GPU throughput ~37 Mcell/s versus ~5.2 Mcell/s on CPU (~7×) at
512²; MAP-Elites fills ~39% of a 16×16 behavior archive from ~1900 evaluations.

## 12. Dependencies

`serde` / `serde_json` / `serde_yaml` (config), `rayon` (CPU parallelism), `image` (PNG),
`anyhow` (errors). Optional: `minifb` (`window`), `wgpu` / `pollster` / `bytemuck` (`gpu`). No
autodiff or graph-library dependency: PageRank, evolution strategies, and the RNG are
hand-written.

## 13. The ceiling

- The GPU path is tiled (~7× over the parallel CPU at 512²), but it covers single-kernel,
  single-channel, gaussian-growth Lenia only. Multi-kernel (e.g. `multiring`), multi-channel, and
  non-Lenia rules run on CPU; FFT convolution is the further unrealized speed.
- Graph, particle, and hypergraph substrates run on CPU; sparse irregular work does not map onto
  the wgpu path.
- NCA training is analytic backprop (Adam) or evolution strategies, both self-contained. Neither
  targets morphogenesis-scale nets; a differentiable-framework backend (burn/candle) and
  backprop-through-time are the path there.
- Spacetime-loaf handles a nonlinear (Fisher-KPP) operator via the analytic VJP, but the energy
  is then non-convex and convergence is not guaranteed; only the linear case is provably exact.
- The live window is minifb-based with keyed live tuning; a full egui panel, 3-D grids, non-flat
  topologies, and a scriptable rule path are unbuilt.
