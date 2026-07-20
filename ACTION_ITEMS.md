# AXIOM action items

## Built

- [x] Config + trait backbone: serde tree, tagged-enum registry, load-time capability
      validation, provenance hash.
- [x] Lenia core validated against the Orbium oracle; multi-kernel Lenia (`multiring`).
- [x] Five rule types on one trait: Lenia, asymptotic Lenia, flow Lenia (mass-conserving),
      Gray-Scott, learned NCA.
- [x] Four substrate realizations: grid, small-world graph, particle swarm, hypergraph, with
      PageRank over each.
- [x] Particle substrate modes: Particle-Lenia (energy), particle-NCA, cohesion swarm.
- [x] Analysis layer: descriptors, connected-component detection, PageRank.
- [x] Deep descriptors: spectral (DFT period), dynamical (activity, entropy), H0 persistent
      homology, organism tracking into lineages, experiment log + similarity search.
- [x] NCA training: analytic backprop (Adam, default) and evolution strategies; world-model rollout.
- [x] GPU compute: tiled wgpu pipeline, validated f32-exact, the default execution path for
      eligible Lenia configs (~7× CPU at 512²).
- [x] Spacetime-loaf: linear (convex, exact) and nonlinear Fisher-KPP (analytic VJP) relaxation.
- [x] MAP-Elites discovery over Lenia behavior space.
- [x] Live window with real-time μ / σ / dt tuning; headless PNG rendering.

## Next, ordered by leverage

### 1. Multi-kernel and multi-channel GPU
The tiled shader reuses one loaded tile, so multi-kernel is a per-kernel loop over the same
tile (add a concatenated kernel buffer + per-kernel params). Multi-channel needs a channel-major
buffer layout and per-kernel source/target indexing. Then `multiring` and multi-channel Lenia
join the GPU path. FFT convolution is the further speed step for very large kernels.

### 2. Differentiable-framework NCA backend
Backprop (Adam) and ES are self-contained and fit local rules well. For morphogenesis-scale
NCAs (many channels, growing a target over many steps), port to `burn` or `candle` and add
backprop-through-time. Prototype in JAX/CAX first if faster.

### 3. Full egui config panel
The window has keyed live tuning of μ, σ, and dt. A full `egui` + `egui-wgpu` panel would make
every config field a live control and hot-reload the rule from the edited tree.

### 4. Deepen analysis further
Higher-order persistent homology (H1 loops / voids), more dynamical descriptors (Lyapunov-ish
sensitivity, branching ratio), and a richer phylogeny (split/merge lineage graph rendered as a
tree). Swap the JSONL experiment log for SQLite if the corpus grows large.

### 5. More substrate + rule coverage
3-D grids and non-flat topologies (sphere, hyperbolic) are a substantial generalization of the
2-D core. A scriptable / WGSL custom-rule path would let a rule be specified as an expression or
shader snippet rather than a fixed enum variant.

## Housekeeping
- `serde_yaml` is deprecated-but-working; swap for `serde_yml` if it ever breaks.
- The repo is under git; keep commits Conventional and `.env`-free (none currently).
