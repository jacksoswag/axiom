# AXIOM

A configurable research engine for cellular automata × graphs × machine learning, in Rust.

Every stage of the update loop is a module selected and parameterized by one serializable
config. An experiment is a config file plus a seed, so it reproduces exactly. The engine is
a thin driver that wires configured trait objects together, and the configuration space is
the research surface.

## Quickstart

```sh
cargo run --release -- validate      # reproduce the Orbium oracle + smoke-test every subsystem
cargo run --release -- run soup      # live window: self-organization from noise
cargo run --release -- list          # bundled presets
```

Every subcommand:

| Command | What it does |
|---|---|
| `run <preset\|config>` | live window with real-time controls |
| `headless <preset\|config>` | step (GPU by default when eligible), write PNG frames + `metrics.jsonl` |
| `validate` | reproduce the Orbium oracle and smoke-test all seven subsystems |
| `learn` | train an NCA to imitate Gray-Scott (Adam by default, `--es` for evolution strategies) + rollout |
| `qd` | MAP-Elites: illuminate Lenia's behavior space |
| `analyze <preset\|config>` | spectral, dynamical, and topological descriptors + organism tracking; logs the run |
| `similar <preset\|config>` | nearest logged runs by descriptor distance |
| `gpu` | GPU compute benchmark against the CPU oracle (wgpu) |
| `graph` | graph×CA seam: small-world spacetime diagram + PageRank |
| `hyper` | hypergraph CA + hypergraph PageRank |
| `particle` | particle substrate (`--mode lenia\|nca\|swarm`) + proximity-graph PageRank |
| `loaf` | spacetime relaxation, linear or `--reaction R` nonlinear: infer occluded time |
| `list` / `dump <preset>` | list presets / print one as YAML |

Headless and every non-window subcommand build without the native windowing dependency:

```sh
cargo run --release --no-default-features -- headless flow --steps 400
```

### Features

`window` (minifb) and `gpu` (wgpu) are on by default. The whole engine, all analysis, and
headless PNG rendering build and run with `--no-default-features`; each feature is additive
and independently omittable.

### Live window controls

| Key | Action | Key | Action |
|---|---|---|---|
| `Space` | pause / resume | `1` `2` `3` `4` | load orbium / soup / life / gray_scott |
| `R` | reset / reseed | `N` | cycle colormap |
| `↑` `↓` | steps per frame | `S` | save PNG snapshot |
| `,` `.` | growth μ down / up (live) | `;` `'` | growth σ down / up (live) |
| `[` `]` | timestep dt down / up (live) | `Q` `Esc` | quit |

The tuning keys rebuild the rule in place without reseeding, so the pattern responds to a
parameter change in real time.

## What a config looks like

```yaml
name: orbium
substrate: { kind: grid, width: 128, height: 128, channels: 1, topology: torus }
rule:
  type: lenia
  dt: 0.1
  kernels:
    - { radius: 13, core: gauss_ring, beta: [1.0], core_mu: 0.5, core_sigma: 0.15,
        growth: { kind: gauss, mu: 0.15, sigma: 0.015 } }
init: { type: orbium }
render: { colormap: viridis, channel: 0 }
analysis: [ { type: descriptors } ]
steps: 400
```

Run a config file directly with `axiom run path/to/config.yaml`. The bundled presets are
dumped to [`configs/`](configs/) as editable starting points. Invalid combinations fail at
load with a reason (`gray_scott requires exactly 2 channels`, `kernel radius too large`),
not silently at runtime.

## Architecture

Each axis of the design taxonomy is a module behind a trait:

| Module | Trait / type | Realizations |
|---|---|---|
| `field` | flat multi-channel state | N-channel 2-D grid |
| `kernel` | pluggable core + convolution | gauss-ring, polynomial, step; multi-ring `β`; torus / bounded |
| `growth` | activation | gaussian, polynomial, exp |
| `rule` | `trait Rule` | Lenia, asymptotic Lenia, flow Lenia (mass-conserving), Gray-Scott |
| `nca` | `trait Rule` + trainer | learned per-cell MLP; Adam and evolution-strategies training; world-model rollout |
| `substrate` | `trait Substrate` | grid-as-graph, small-world graph, generic message passing |
| `particle` | swarm | Particle-Lenia (energy), particle-NCA, cohesion swarm; proximity graph |
| `hypergraph` | n-ary relations | hypergraph CA + random-walk PageRank |
| `analysis` | `trait Observer` | descriptors, connected-component detection, PageRank |
| `dynamics` | descriptors + log | spectral, entropy/activity, H0 persistent homology, organism tracking, experiment log |
| `qd` | MAP-Elites | illuminate behavior space over `(μ, σ)` |
| `loaf` | spacetime relaxation | infer occluded time from endpoints; linear and nonlinear operators |
| `gpu` | wgpu compute | tiled single-kernel Lenia, the default execution path for eligible configs |
| `render` | colormaps → PNG / window | viridis, turbo, magma, gray |
| `config` | serde tree + `validate()` | tagged enums select constructors; load-time capability checks |

The convolution hot loop is specialized and rayon-parallel on CPU and tiled on GPU; the graph
path is the generic `aggregate(neighbors) → update`. A convolutional CA and a graph message-passing
step are the same operation on different neighborhoods, which is the seam the engine is built for.

## The graph × CA × ML seam

One substrate abstraction has four realizations: a grid is a graph with lattice adjacency, a
particle swarm is a graph with proximity edges, a small-world graph carries explicit edges,
and a hypergraph generalizes edges to n-ary relations. PageRank is a first-class observer over
any of them, including the emergent interaction graph of detected Lenia organisms. The same
growth law that drives grid Lenia runs over a graph substrate (`axiom graph`) and a hypergraph
(`axiom hyper`), and Particle-Lenia runs the energy formulation over the particle substrate
(`axiom particle --mode lenia`).

## Machine learning

`axiom learn` fits an NCA (rule = a small per-cell MLP over a 4-filter perception) to a
Gray-Scott transition. The default trainer is gradient descent with analytic backprop and
Adam; `--es` selects gradient-free evolution strategies. The learned rule is then evaluated as
a world model: prediction error against the true simulator as a function of rollout horizon.
`axiom qd` runs MAP-Elites over Lenia growth parameters, keeping the most structured pattern in
each behavior-space bin and rendering the illuminated archive as a montage.

## Measurement

`axiom analyze` runs a config and computes spectral (dominant period), dynamical
(activity, value entropy), and topological (H0 persistent homology) descriptors, tracks detected
organisms into persistent lineages, and appends the run to an experiment log. `axiom similar`
finds the nearest logged runs by descriptor distance.

## Time modes

Forward rollout is the default. `axiom loaf` runs the spacetime-loaf mode: it treats the whole
`space × time` volume as one field and relaxes it to global consistency, so interior states are
inferred from the boundary. With a linear diffusion operator the energy is convex and relaxation
recovers the occluded middle; `--reaction R` uses a nonlinear Fisher-KPP operator, where the
energy is non-convex and relaxation still descends but convergence is not guaranteed.

## GPU compute

The Lenia convolution and growth run as a tiled wgpu compute pipeline: each workgroup caches its
cell block plus halo in workgroup memory, then convolves out of that shared tile. It is validated
f32-exact against the CPU oracle and is the default execution path for eligible single-kernel
Lenia configs (about 7× the CPU throughput at 512²). Pass `--cpu` to `headless` to force the CPU
path; non-eligible rules (multi-channel, learned, reaction-diffusion) run on CPU.

## Validation

`axiom validate` checks seven subsystems and writes artifacts to `out/validate/`:

- **Orbium oracle**: the canonical Lenia glider stays alive with bounded mass and translates.
- **Gray-Scott**: a structurally different rule forms persistent Turing patterns.
- **Detection + PageRank**: multiple organisms are detected and ranked.
- **Graph seam**: message-passing CA on a small-world graph with a valid PageRank distribution.
- **Flow Lenia**: total mass is conserved to floating-point precision.
- **Asymptotic Lenia**: the relaxation stays alive and bounded.
- **GPU vs CPU**: one Lenia step matches the CPU oracle to f32 precision.

## Scope

The realized surface is the backbone plus a broad vertical slice across the design taxonomy.
The remaining backlog (multi-channel GPU, a full egui panel, a differentiable-framework NCA
backend, 3-D grids, non-flat topologies, a scriptable rule path) lives in
[`ACTION_ITEMS.md`](ACTION_ITEMS.md). The technical reference is [`TECHNICAL_SPEC.md`](TECHNICAL_SPEC.md).
