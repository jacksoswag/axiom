# axiom

AXIOM is a Rust search system for three-dimensional continuous-trait Particle-Lenia on a periodic torus. It searches rules for persistent living material, connected voids, local regimes, and sustained change. The visible world comes from the simulated particles.

Read [TECHNICAL_SPEC.md](TECHNICAL_SPEC.md) for the implementation contract, data formats, and acceptance evidence, and [MARKET_ANALYSIS.md](MARKET_ANALYSIS.md) for the demand read. [MASTERDOC.html](MASTERDOC.html) synthesizes all three at three reading depths.

## Current verdict

Particle-Lenia is a conditional choice for this product. It remains the world engine only while equal-budget evidence supports persistent, repairing, diverse three-dimensional worlds. A Flow-Lenia replacement needs to win on persistence, recovery, bicontinuity, and diversity at equal runtime.

The renderer can reveal a connected particle state. It cannot create one. The 100k-step certification pipeline exists, but no certified world has been demonstrated.

## World model

Each particle has a position `x` on `T^3` and a fixed continuous trait `c` in `[0, 1]`. Motion follows the gradient of pair-specific Particle-Lenia interaction energy. Particle count sets the cost; coordination sets density; extent derives from them and the interaction radius.

Traits use fixed, piecewise-linear anchor memberships. At most two adjacent anchors contribute to a trait value. The genome is one bounded flat vector:

```text
[ coordination ]
[ trait-distribution logits ]
[ anchors × anchors pair blocks, source-major ]

pair block =
  shells × [kernel beta, mu, sigma]
  bumps  × [growth beta, mu, sigma]
  [directed weight]
```

Each anchor pair owns its kernel, growth curve, and directed weight. This pair-indexed control net recovers the discrete pair model exactly at anchor values while allowing continuous trait mixtures between anchors. The product boundary fixes three spatial dimensions; particle count, radius, timestep, anchor count, and the shell and bump caps are experiment settings. Coordination and initial trait-density logits remain genome genes.

## One tuning surface

Every setting a run can vary lives in one `Tuning` struct, and `tuner::campaign::run` is the only entry point. `Tuning` covers the fixed world, discovery search, the viability gates, the repair protocol, which expensive tiers run, and how the scheduling model is fit.

A registry of 43 named knobs drives command-line parsing, generated help, and the archive header from a single declaration. Ask `train` for the complete list with its defaults:

```bash
cargo run --release --no-default-features --bin train -- --help
```

Settings split into two kinds. Search scheduling knobs (`seed`, `batch`, `generations`, `capacity`, `promotion_budget`) change how much searching happens. Everything else changes what an evaluation means, and folds into a digest that `ExperimentIdentity` carries. A campaign resumed with different gates, mutation scales, lane shares, or repair protocol is refused rather than silently mixing incomparable evidence.

Descriptor geometry stays constant: the radial bins, heterogeneity grid sides, autocorrelation lags, barcode bars, sample count, and novelty neighbour count define what a descriptor axis means and keep the descriptor width a compile-time constant. Changing one creates a new descriptor version.

## Search and qualification

Search uses novelty and a binary viability gate. Novelty chooses where to explore. The gate decides archive admission. There is no scalar fitness score and no best-entry accessor.

The version 5 descriptor has 53 fixed-scale axes, all computed from true three-dimensional torus geometry:

| axes | measurement |
|---:|---|
| 18 | trait-distance-conditioned radial distribution ratios |
| 15 | shot-noise-corrected local heterogeneity over periodic grids with sides 4, 8, and 16 |
| 8 | seven H0 death-scale masses plus cutoff-separated component mass |
| 4 | dense-material and void winding fractions at two thresholds |
| 8 | dynamics, directed interaction asymmetry, and perturbation repair |

Discovery applies five base gates:

1. finite state and descriptor (`Dead`)
2. departure from a uniform gas (`Dispersed`)
3. an upper structure bound (`Collapsed`)
4. mobility and temporal variation (`Frozen`)
5. recovery after local positional shocks (`Fragile`)

Persistence and certification add three world gates: local density and trait heterogeneity, winding material, and winding void. A connected phase must occupy at least 8 percent of the descriptor grid and contain a non-contractible loop around at least one torus axis. Each gate threshold is a `Tuning` field, so a campaign can tighten or loosen what counts as living, and the evidence it produces stays separate from evidence gathered under other thresholds.

Seed counts and pass requirements belong to the tier rather than the run. A run chooses which tiers to spend on and how long each rollout is:

| tier | budget | seeds | pass condition |
|---|---:|---:|---|
| discovery | 1,500 main steps or longer | 1 shared seed | five base gates |
| persistence | 10,000 main steps or longer | 3 | at least 2 pass base and world gates |
| certification | 100,000 main steps or longer | 5 unseen seeds | at least 4 pass the aggregate base and world gates |

Every completed main rollout also runs one undamaged control and three local-shock continuations, each one quarter as long as the main rollout. Repair evaluation therefore makes a completed tier cost about twice its nominal main-step budget.

Discovery archives use format v9 and record the tuning that produced and judged them, so an archive file is self-describing. Each generation freezes its archive snapshot, evaluates offspring in parallel, applies gate results, then merges candidates with deterministic capacity pruning. Admission novelty remains provenance; current novelty drives parent choice and crowding.

Campaign state stores checksummed evaluation evidence and labels. Reusing `state=campaign.state` carries that evidence into an independent search batch. It does not resume an archive, generation counter, stall history, or search RNG. Use a new `seed=` for each new batch; an identical batch is reproducible and its duplicate discovery records are ignored.

## Learned scheduling

The persistence scheduler is a deterministic bagged logistic ensemble. It selects which candidates receive expensive continuation slots only after lineage-held-out evaluation beats uniform allocation and each outcome class has enough independent lineages. It cannot admit archive entries, choose discovery parents, or certify a preset.

## Rendering and replay

The viewer defaults to a causal material renderer. It deposits the current particle snapshot into a periodic density field, derives material, palette, emission, and activity from that field, and ray-integrates the result. Particle grain is an optional deterministic overlay for close inspection.

The renderer adds no time-based noise, autonomous animation, or unrelated visual motion. Pausing the world freezes density and emission. The reference snapshot path uses the same causal field semantics.

Archive selection in the viewer replays the selected genome from its deterministic initial seed. It does not restore a saved checkpoint. Checkpoints preserve authoritative particle positions, traits, tick, genome, measured norms, and versioned rendering metadata; cache data is rebuilt after restore. Immutable states are published with a no-replace hard link after their temporary file is flushed. The mutable manifest is replaced atomically only after the state exists.

## Run it

The viewer needs the default `viewer` feature:

```bash
cargo run --release --bin axiom
```

Run a small discovery campaign and write the archive plus durable campaign state:

```bash
cargo run --release --no-default-features --bin train -- \
  generations=25 particles=1200 anchors=4 steps=1500 \
  seed=1 evaluation_seed=1 out=archive.txt state=campaign.state
```

Queue persistence checks after discovery candidates are promoted. A tier's step count enables it; `0` turns it off:

```bash
cargo run --release --no-default-features --bin train -- \
  generations=25 promotion_budget=12 persistence_steps=10000 \
  out=archive.txt state=campaign.state
```

Enable certification as well. It writes passing checkpoint recipes to `presets` unless `preset_dir=` changes it:

```bash
cargo run --release --no-default-features --bin train -- \
  generations=25 promotion_budget=12 \
  persistence_steps=10000 certification_steps=100000 \
  out=archive.txt state=campaign.state preset_dir=presets
```

Search under stricter gates. The campaign keeps this evidence separate from evidence gathered under the defaults:

```bash
cargo run --release --no-default-features --bin train -- \
  generations=25 robustness_floor=0.8 heterogeneity_floor=0.05 \
  out=strict.txt state=strict.state
```

Render a deterministic CPU reference snapshot from the default world:

```bash
cargo run --release --example snapshot -- \
  out=axiom-snapshot.ppm steps=1500 width=960 height=540
```

Render archive entry 0. The command resolves its genome, recreates the seeded initial world, runs the requested steps, then writes a binary PPM:

```bash
cargo run --release --example snapshot -- \
  archive=archive.txt entry=0 steps=1500 out=axiom-snapshot.ppm
```

Inspect a reproducible archive recipe without opening the viewer:

```bash
cargo run --release --no-default-features --example inspect -- archive.txt 0
```

## Limits

- Traits stay fixed during a rollout. There is no trait motion, birth, death, inheritance, or evolution of trait values.
- The descriptor is hand-designed and versioned. It describes physical behavior without a projection, but it still encodes a prior.
- The viewer reads archive recipes and replays their initial seeds. Checkpoint browsing and restoration are separate work.
- Campaign files preserve evidence across independent search batches; they do not resume a half-finished evolutionary generation.
- The material renderer is a deterministic CPU reference. Interactive room-scale performance has not been established.
- No certified world has yet validated the complete 100k-step product target.
