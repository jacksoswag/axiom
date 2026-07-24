# AXIOM technical specification

AXIOM searches for three-dimensional Particle-Lenia worlds that persist, repair themselves, and contain distinct local regimes. The system simulates, evaluates, archives, promotes, renders, and checkpoints those worlds. Rendering derives visible material from simulator state and adds no autonomous animation.

Present-tense statements describe code in this repository. Items marked as future work are not implemented.

## 1. Design invariants

These constraints are structural. Changing one changes what the system is.

| Invariant | Consequence |
|---|---|
| Particle dynamics | Cost follows particle count instead of dense voxel count. |
| Three spatial dimensions | Search effort goes into worlds the product can display and inhabit. |
| Periodic torus | Worlds have no terminal boundary and structures cross every face. |
| Derived extent | Coordination controls density; extent never becomes an unrelated free scale. |
| Measured density norms | Normalized potential means the same thing across particle counts and traits. |
| Closed-form derivatives | The simulator has no autodiff or tensor dependency. |
| Novelty-led evolution | Search rewards behavioral difference instead of one global quality number. |
| Binary viability | A gate decides whether a candidate qualifies as living. It never ranks survivors. |
| Causal rendering | Geometry, color, motion, and emission derive from particle state. |
| One tuning surface | Every per-run setting lives in one struct behind one entry point. |

The dependency runs one way. `engine` is a pure function of a flat genome and imports nothing from `tuner` or `viewer`, so the simulation cannot acquire a dependency on how it is searched or drawn.

## 2. Authoritative world state

Each particle carries a position and a fixed continuous trait:

```text
x_i in T^3       position on the periodic world
c_i in [0, 1]    continuous trait
```

Position is the only dynamical particle state. The update is first-order gradient descent on the Particle-Lenia energy. Traits are fixed for the duration of a rollout; spatial self-organization can sort that continuum into niches, but trait motion, birth, death, and inheritance require a separate biological model and are outside this contract.

A run's authoritative state is reconstructible from its parameter set alone: genome genes, derived box size, timestep, and seed, plus the pair control net with its measured norms and the tick count. Density volumes, meshes, spatial indexes, and GPU buffers are caches that rebuild from that state.

## 3. Continuous trait interaction field

### 3.1 Trait basis

`M` fixed anchors divide the trait interval. Piecewise-linear hat functions interpolate between them:

```text
alpha_a(c) >= 0
sum_a alpha_a(c) = 1
```

At most two adjacent anchors have nonzero membership, and at an anchor the membership is exactly one-hot. Anchors sit evenly spaced on a circle, so a trait wraps at the seam instead of clamping. This basis is an exact bridge from a discrete species model while traits stay fixed. A smooth basis belongs in a later trait-dynamics experiment, where derivatives with respect to `c` would matter.

### 3.2 Pair-indexed control net

Every ordered anchor pair `(a, b)` owns its own kernel, growth curve, and directed weight:

```text
U_ab(i)    = sum_(j != i) alpha_a(c_j) K_ab(d_ij)
Uhat_ab(i) = U_ab(i) / norm_ab

E_i = -sum_a sum_b alpha_b(c_i) w_ab G_ab(Uhat_ab(i))

delta x_i = dt sum_a sum_b alpha_b(c_i) w_ab G'_ab(Uhat_ab(i))
            / norm_ab grad U_ab(i)

grad U_ab(i) = sum_(j != i) alpha_a(c_j) K'_ab(d_ij) delta_ij / d_ij
```

`delta_ij` uses minimum-image torus displacement with smooth distance softening. With one-hot traits at the anchors these equations reduce exactly to a discrete pair-specific implementation.

A pair's position in the net is its `(source, destination)` identity, so neither the interaction nor its callers carry those indices separately.

### 3.3 Kernel and growth mixtures

Each pair keeps bounded Gaussian mixtures:

```text
K_ab(d) = sum_m beta^K_abm exp(-(d - mu^K_abm)^2 / (2 sigma^K_abm^2))

G_ab(u) = 2 sum_m beta^G_abm exp(-(u - mu^G_abm)^2 / (2 sigma^G_abm^2)) - 1
```

Only `G'` enters movement. A zero mixture amplitude disables a component. Kernel reach stays below `0.375 * extent` and every sigma has a positive lower bound. The neighbour loop truncates each Gaussian shell at `mu + 3.5 sigma`, roughly 0.2 percent of its peak, and skips terms whose amplitude is negligible so one inert term cannot inflate the search radius. This truncation is an explicit numerical approximation: the executed force carries a small hard cutoff and is not the exact gradient of the untruncated energy at that boundary.

### 3.4 Norm calibration

`norm_ab` is the mean `U_ab` measured on a deterministic uniform reference population: a fixed calibration seed, uniform positions, and uniform anchor traits, using the same kernel, torus distance, softening, and memberships as the real rollout. Calibration runs when a simulation is constructed, so every anchor pair sees reference mass. A clumped live state never becomes the density reference.

### 3.5 Derived geometry

Extent falls out of particle count, interaction radius, dimensionality, and the genome's coordination target. A single density probe measures the nominal coordination of a uniform population at a trial extent; a genome's coordination gene rescales that reference. The probe depends only on particle count, dimensionality, and radius, so a campaign measures it once and passes it to every candidate.

## 4. Genome

The genotype is one bounded `Vec<f32>`. Its shape, anchor count and the shell and bump caps, rides beside the genes in the shared parameter set, and the pair-block stride follows from those counts.

```text
[ coordination ]
[ trait-distribution logits ]
[ M^2 pair blocks in source-major order ]

pair block =
  shells * [kernel beta, mu, sigma]
  bumps  * [growth beta, mu, sigma]
  [directed weight]
```

Anchor count and dimensionality are read from the parameter set, never fixed by a constant, so the step loop takes its shape from whatever genome it is handed.

Coordination stays a gene in a calibrated range because it changes physical density and belongs to the learned world law. Trait-distribution logits select an initial population density over trait bins; the decoder applies a small floor before normalization, then jitters traits deterministically inside each bin. Every evaluation seed receives the same trait histogram and a different reproducible spatial arrangement.

## 5. The tuning surface

Every per-run setting lives in one `Tuning` struct, and `tuner::campaign::run` is the only entry point.

| Group | Contents |
|---|---|
| `world` | particle count, dimensionality, anchors, radius, integration rate, evaluation seed, shell and bump caps |
| `discovery` | generations, steps, batch, bootstrap size, capacity, search seed, promotion budget, stall window, mutation scales, lane shares, curated parent indices |
| `gates` | the six viability thresholds, the mobility floor, the two watchdog thresholds, and the sustained-window count |
| `repair` | probe count, shake magnitude, recovery fraction, minimum effective damage |
| `tiers` | persistence and certification step budgets, each optional |
| `learning` | minimum labeled rows, model seed, continuation quotas, logistic hyperparameters |

A registry of 43 named knobs drives command-line parsing, generated help, and the archive header from one declaration. A knob cannot exist without documentation, and adding one reaches all three surfaces automatically.

### 5.1 What is not tunable

Descriptor geometry is fixed: the trait and radial bin counts, heterogeneity grid sides, autocorrelation lags, barcode bars, per-rollout sample count, mobility ceiling, and novelty neighbour count. These define what a descriptor axis means, size fixed-width arrays, and keep the descriptor width a compile-time constant. Changing one creates a new descriptor version rather than a run setting.

Seed counts, pass requirements, and whether the world gate applies belong to the evaluation tier, not the run. A run chooses which tiers to spend on and how long each rollout is; it cannot weaken what a tier means.

### 5.2 Evidence separation

Settings divide into two kinds. Search scheduling knobs, meaning seed, batch, generations, bootstrap size, capacity, promotion budget, and curated pins, change how much searching happens. Every other setting changes what an evaluation means or how it is judged, and folds into `Tuning::digest`.

`ExperimentIdentity` carries that digest alongside the simulator, descriptor, genome-layout, and feature-schema versions and the fixed world parameters. Campaign state binds to an identity on first use. A batch that differs only in scheduling shares the ledger; a batch with different gates, mutation scales, lane shares, or repair protocol is refused with an experiment mismatch before any rollout work begins.

## 6. Behavior descriptor v5

The descriptor describes a world rather than one global particle cloud. It uses true three-dimensional torus geometry and versioned fixed scaling. No feature reads a two-dimensional projection. The complete descriptor has 53 values.

### 6.1 Trait-conditioned radial structure: 18 values

Pair samples divide into three trait-distance bands and six log-spaced radial bands. Each count divides by a measured uniform baseline for the same trait distribution, so uniform matter reads approximately one in every populated bin. The global radial distribution measures interaction-scale order cheaply. It is not a biome metric.

### 6.2 Local spatial heterogeneity: 15 values

Particles deposit into periodic grids with side lengths 4, 8, and 16. Each scale contributes five values computed over local cells:

1. density overdispersion after subtracting Poisson sampling variance;
2. void probability;
3. occupancy-weighted variance of local mean trait after a finite-population shot-noise correction;
4. occupancy-weighted local trait entropy with a small-sample correction;
5. axial density autocorrelation.

These separate planted local biomes from a uniformly mixed trait field with similar global pair statistics. The shot-noise corrections matter: without them, sparse fine-grid cells make random trait mixtures look like biomes.

### 6.3 Topology: 12 values

- Seven values hold folded H0 death-scale mass from the cutoff-limited minimum spanning forest. Kruskal sees only toroidal particle pairs within the interaction cutoff.
- One value holds a log-scaled mass for components still separate at that cutoff.
- Two values hold the largest qualifying dense winding-component fraction at two density thresholds.
- Two values hold the equivalent void winding-component fractions.

Dense and void components use periodic adjacency. A phase contributes only if it occupies at least 8 percent of the grid and one of its components contains a non-contractible loop around at least one torus axis. A compact blob is therefore disconnected for product purposes even when all its occupied cells touch. The world gate asks for a material carrier and a flight space that continue through the periodic world.

### 6.4 Dynamics and interaction: 8 values

Temporal variance of the spatial descriptor, autocorrelation at three fixed lags, minimum-image particle mobility, local material turnover, directed interaction-field asymmetry, and perturbation recovery fraction.

Fixed formulas and explicit clamps map descriptor axes into a shared span. Ratios use a log transform with fixed bounds, heterogeneity uses feature-specific formulas, and H0 uses normalized forest bar mass. There is no descriptor calibration corpus, and archive statistics never rescale axes during a run.

The eight temporal windows spread across the latter half of each rollout, so autocorrelation lags are fractions of that tier's horizon rather than a fixed number of simulator steps. This suits within-tier description and must not be read as one universal physical decay constant.

## 7. Viability and world qualification

AXIOM has no scalar fitness function. It has a Boolean gate, a novelty measure, and one learned scheduling model with limited authority.

```text
V(g, s, r) = alive AND structured AND active AND bounded AND repaired

N(g) = (1 / k) sum_(j in k nearest archive neighbors) distance(z_g, z_j)
```

`V` decides admission. `N` decides exploratory priority. Neither substitutes for the other.

The base gate reports the first failing clause:

| Clause | Rejection | Meaning |
|---|---|---|
| finite state and descriptor | Dead | Integration diverged or produced invalid values. |
| departure from uniform baseline | Dispersed | The world remained a gas. |
| upper structure bound | Collapsed | Matter imploded into the shortest scale. |
| mobility and temporal variation floors | Frozen | Structure exists without ongoing change. |
| recovery after local positional shocks | Fragile | The organization does not restore itself. |

Promotion tiers add Boolean world clauses for local heterogeneity, connected material, and connected void.

Every threshold is a `Tuning` field rather than a constant, so a campaign can tighten or loosen what counts as living. Thresholds are fixed for the duration of a run: no campaign calibrates or adapts them mid-search. Signed clause margins accompany each verdict, which makes labels useful to the scheduling model without turning any gate into a score.

## 8. Evaluation ladder

Every candidate starts cheaply. Cost rises only after evidence warrants it.

| Tier | Main rollout per seed | Seeds | Decision |
|---|---:|---:|---|
| Discovery | 1,500 steps or longer | 1 shared seed | Pass the five base clauses. |
| Persistence | 10,000 steps or longer | 3 seeds | At least two seeds pass all base and world clauses. |
| Certification | 100,000 steps or longer | 5 unseen seeds | At least four seeds pass the aggregate base and world gates. |

Those numbers name the main rollout, not total force-integration work. A seed that reaches its end also runs one undamaged control and three damaged continuations, each for one quarter of the main horizon, so a completed seed costs about twice the tabled step count. Early failure reduces that cost.

A budget shorter than its tier's minimum is rejected before any rollout work. Certification requires persistence.

The evaluator retains eight late-rollout windows for every completed seed and stops a seed after non-finite state, sustained dispersion, or sustained collapse. A temporary excursion cannot trigger an early failure. Windows record evidence for later features; the current tier decision has no window-drift predicate.

Certification runs only for persistence-passing candidates. After a certification tier passes, the campaign replays one passing seed for the tier budget and snapshots it, clamping the render support radius to at most one quarter of the resolved extent so the checkpoint stays valid for small worlds. That produces a certified preset in memory and, when a preset directory is configured, saves its immutable state and manifest before the campaign returns.

Persistence labels come from persistence-tier outcomes. Certification outcomes stay in the ledger, and the persistence trainer does not use them as labels.

## 9. Evolutionary search

### 9.1 Bootstrap and offspring

The archive evaluates one deterministic default genome and a deterministic random population. The default is a reproducible starting probe, not a trusted viable seed; it passes the same gate as everything else. Iso+LineDD is the mutation and recombination operator:

```text
child = a
      + sigma_iso * gene_range * Normal(0, I)
      + sigma_line * Normal(0, 1) * (b - a)
```

Bounds clamp every child. Both scales are tuning fields.

### 9.2 Generation transaction

Each generation performs one deterministic transaction:

1. Freeze the archive descriptor snapshot.
2. Recompute current novelty for archive entries against that snapshot.
3. Choose parents by novelty tournament, goal expedition, or random restart.
4. Evaluate the batch in parallel, preserving batch order.
5. Apply the binary gate.
6. Score viable candidates against the same frozen snapshot.
7. Merge candidates, recompute crowding on the combined set, and enforce capacity without insertion-order dependence.
8. Queue a stratified subset for longer evaluation.

Admission novelty is provenance. Current novelty controls parent choice and capacity. The two carry different names in storage.

### 9.3 Search lanes

Batch allocation splits between novelty-led offspring, goal expeditions, and fresh random genomes. Random takes whatever the first two do not claim, so the three always sum to the batch at any size and a random escape route survives even in tiny test batches. The default shares are 70 percent novelty and 20 percent expedition; both are tuning fields.

Curated parent indices may supply archived parents for up to half the expedition slots. That is an explicit search input.

An expedition samples every descriptor coordinate independently and uniformly, selects the archived behavior nearest that target, and mutates its genome. It does not estimate a sparse occupied region. In 53 dimensions most uniform targets are far from the realized manifold, so the mechanism is best understood as random goal direction rather than coverage-aware illumination. It adds distant intentions without letting a language or image model invent simulation state.

`lineage_id` identifies a founder family; a random or bootstrap founder takes an ID derived from its exact initial genome and descendants inherit the primary mutation parent's. The separate `genome_hash` identifies an individual rule. That distinction keeps close relatives in one held-out learning partition.

### 9.4 Stall handling

The search declares a stall when both median current novelty and occupied descriptor neighborhood count fail to improve across the configured window. It shifts the batch toward expedition and random, and multiplies both mutation terms by the stalled spread factor. It does not inspect rejection histograms or target gene blocks. All four stalled shares and the spread factor are tuning fields.

## 10. Long-horizon scheduling

The persistence model predicts which short runs deserve expensive continuation. It never admits an archive entry, selects a mutation parent, or certifies a preset.

### 10.1 Dataset

`CampaignLedger` stores an append-only record for each imported discovery record or tier seed: identity, source and tier, requested budget, seed, final metrics, versioned feature vector, gate margins and results, early-stop code, retained windows, and the tier-wide pass result. Tier evaluations include the candidate genome and windows. Discovery imports come from search records that carry neither, so those fields stay empty rather than fabricated.

Failed and early-stopped records remain in the ledger. The trainer forms one discovery feature row per `(genome_hash, lineage_id)` and joins it only to persistence-tier labels.

### 10.2 Model and training

A deterministic bagged logistic ensemble over standardized numeric features, with no tensor runtime. Each member trains on a lineage-grouped bootstrap with regularized binary cross-entropy:

```text
p_m = sigmoid(w_m dot f + b_m)

loss = -y log(p_m) - (1-y) log(1-p_m) + lambda ||w_m||^2
```

The ensemble mean estimates survival probability and member standard deviation estimates uncertainty. Founder families assign deterministically to training, calibration, and test partitions. Calibration selects a temperature. The held-out report includes Brier score, precision-recall area, calibration error, precision at budget, and recall at budget.

Training waits until enough labeled discovery rows have joined persistence outcomes. After training, the scheduler gains authority only when the untouched test set carries enough positive and negative founder families, Brier score beats the training base-rate predictor by a margin, and recall in the top fraction clears a floor. These are code policy thresholds, not universal constants.

Continuation slots then use fixed quotas: some by predicted survival, some by ensemble uncertainty, and some spread across descriptor neighborhoods. The uniform quota keeps producing counterexamples and protects the dataset from selection bias. An unauthoritative model leaves selection to the deterministic neighborhood-uniform fallback.

## 11. Durable formats

| Collection | Contents | Authority |
|---|---|---|
| Discovery archive | Tuning header, genome, lineage, descriptor, gate result, admission and current novelty | Parent source and behavior history |
| Campaign ledger | Imported discovery or tier seed evidence, features, gates, early-stop code, windows, tier pass | Training evidence |

Discovery archive text is format v9. Its header serializes every knob through the tuning registry, so an archive file records the exact regime that produced and judged it and a reader can reproduce both. Archive capacity uses current novelty and crowding.

Campaign state is checksummed binary format v4, atomically replaced. It persists the ledger and the experiment identity. It does not persist the discovery archive, RNG state, generation number, stall history, promotion queue, or current model. A later invocation starts an independent search batch and should use a new search seed. Exact duplicate discovery records are ignored, and candidates with existing persistence or certification evidence are not rerun. This is evidence continuation, not bitwise search resumption.

Version mismatches fail closed. There is no migration path in either format: a file from a different version is rejected with a named error rather than reinterpreted.

## 12. Checkpoints

An archive entry stores a discovered genome and its behavior evidence. A checkpoint stores a place.

```text
WorldManifest
  world_id, simulator_version, genome_layout_version, descriptor_version
  resolved parameters and genome
  seed, tick, extent, softening, timestep
  render recipe
  latest checkpoint id

WorldState checkpoint
  checkpoint_id, parent_checkpoint_id
  world_id, tick
  particle positions and traits
  full genome
```

`WorldState` files are immutable. Saving creates a temporary sibling, flushes it, and publishes it without replacing an existing final name. The mutable `WorldManifest` is written atomically after its selected state file exists, so it can advance the latest checkpoint id. The optional parent checkpoint id is metadata only; no code walks a checkpoint parent graph.

Loading validates both records, reconstructs the world, restores tick and particle state, and installs the saved interaction norms directly. Restore never recalculates those norms, which avoids platform-dependent calibration drift.

Checkpoint identities are simulator v2, genome layout v2, descriptor v5, render recipe v1, manifest binary v3, and state binary v1. Manifest and state files carry independent checksums.

A world compares against a recipe by comparing the genome it was built from. Decoding is deterministic, so genome equality answers the question exactly; measured norms are excluded because checkpoints save them separately.

## 13. Rendering as a causal observation

The renderer derives periodic material density from the live particles:

```text
rho(q)      = sum_i phi(||q - x_i||_T / h)
trait(q)    = sum_i phi_i c_i / max(rho(q), epsilon)
activity(q) = |rho_t(q) - rho_(t-1)(q)|
```

`phi` is a finite-support smooth visual kernel. The render recipe's support radius is a positive world-coordinate distance independent of interaction radius.

The view combines an isosurface or narrow density band for connected material, volumetric emission and absorption for depth, normals from the density gradient for lighting, trait-weighted color with a limited cool-to-warm palette, and a sparse bead layer from deterministic strided samples of real particles.

The recipe may use density, trait, activity, camera, and fixed lighting. It has no shader-time deformation, scrolling texture, animated noise, procedural erosion, or camera-driven simulation change. Pausing the simulation freezes every biological change in the image.

| Path | Role |
|---|---|
| Particle sprites | Diagnostic ground truth and fallback. |
| Periodic density material | Product view for three-dimensional worlds. |

The particles remain the causal substrate while compact density reconstruction makes their collective state read as one luminous cavern or reef instead of unrelated balls. None of those channels invents geometry or motion. A world that is only a gas still renders as a gas, and a world without corridors cannot acquire corridors from the shader.

The reference density backend runs on the CPU at modest resolution and defines field semantics for tests. A GPU backend can replace field construction or ray integration when particle counts require it, provided sampled density and seam behavior match the reference within a declared tolerance.

## 14. Module map

| Module | Responsibility |
|---|---|
| `engine/substrate.rs` | Particle positions, traits, box geometry, and the private spatial index. |
| `engine/trait.rs` | Anchor basis, active memberships, trait seeding. |
| `engine/kernel.rs` | Gaussian shell mixtures, analytic slopes, periodic distance. |
| `engine/params.rs` | `Params`: the rollout parameter set every component shares, genome genes included. |
| `engine/matrix.rs` | The pair control net, derived from the genome and calibrated. |
| `engine/resolve.rs` | Derived box size from the measured density reference. |
| `engine/lenia.rs` | The continuous-trait force step. |
| `engine/sim.rs` | Authoritative run state, reconstructible from `Params` alone. |
| `util.rs` | Deterministic xorshift stream and the crate's one hash. |
| `tuner/tuning.rs` | `Tuning` and the knob registry. |
| `tuner/metrics.rs` | Versioned 53-value descriptor and window statistics. |
| `tuner/persistence.rs` | H0 barcode computation. |
| `tuner/viability.rs` | Named binary base and world clauses. |
| `tuner/rollout.rs` | Tiered deterministic evaluation and perturbation probes. |
| `tuner/novelty.rs` | Descriptor distance, current novelty, crowding. |
| `tuner/search.rs` | Generation transaction, search lanes, lineage, promotion queues. |
| `tuner/learning.rs` | Persistence ensemble and grouped evaluation reports. |
| `tuner/archive.rs` | Versioned discovery archive and per-evaluation records. |
| `tuner/campaign.rs` | The entry point, promotion tiers, ledger, scheduling, certification. |
| `tuner/checkpoint.rs` | Atomic world manifests and checkpoints. |
| `render_recipe.rs` | Versioned causal-material settings. |
| `viewer/material.rs` | Reference periodic density field and causal material renderer. |
| `viewer/particles.rs` | Particle diagnostic renderer and bead overlay. |
| `viewer/runs.rs` | Discovery archive browser. |
| `train.rs` | Search and promotion command line. |

## 15. Test evidence

These are code-level behaviors covered by the repository's tests. They do not establish the product target.

**Physics and traits.** One-hot traits match the discrete pair-specific step within float tolerance. Hat memberships sum to one and activate at most two anchors. Measured norms stay finite and positive for every active pair. Grid and all-pairs stepping agree on a seeded fixture. The step loop takes its anchor count from whatever genome it is handed.

**Descriptor and search.** A planted separated-biome world differs from a uniformly blended world even when their global radial distributions are close. Uniform, collapsed, frozen, fragile, single-blob, and bicontinuous fixtures fail or pass the intended named clauses. Raising a gate threshold rejects a world that passed the default. Generation results stay reproducible across thread counts. Reordering a viable batch does not change the retained archive set. Current novelty changes when archive density changes while admission novelty stays stable.

**Tuning.** Every knob round-trips through its own displayed text. Knob keys are unique. The experiment digest separates measurement regimes from search effort: batch, generations, seed, capacity, and promotion budget leave it unchanged, while gates, mutation scales, repair settings, and lane shares change it. Lane counts always sum to the batch at every size.

**Learning.** Grouped splits keep every lineage in one partition. Synthetic persistence data with a known signal produces calibrated ranking above chance. A deliberately useless model loses scheduler authority.

**Durability.** A checkpoint round-trip preserves every particle bit, trait bit, tick, and genome gene. A restored world reaches the same state hash after 10,000 further steps. Corrupt length, checksum, version, and non-finite data produce named errors. An archive round-trips its tuning header and refreshes current novelty on load.

**Rendering.** Opposite torus faces sample equal density and produce no visible crack. Pausing the world freezes the density field and emission. A particle bridge becomes connected material while separated clusters retain a void.

## 16. The honest ceiling

The largest unknown is whether Particle-Lenia can produce a bicontinuous carrier phase with several persistent local regimes. The renderer can reveal connected matter already present in the particles; search and physics must supply it.

| Decision | Evidence needed to change it |
|---|---|
| Keep Particle-Lenia as the world engine | A three-dimensional Flow-Lenia prototype wins at equal runtime on persistence, recovery, bicontinuity, and diversity. |
| Keep pair-indexed trait controls | An equal-budget search shows no meaningful loss from factorization and the one-hot fixture still passes. |
| Keep the hand-designed descriptor | A learned visual goal space improves held-out human diversity without collapsing physical diversity. |
| Keep the reference renderer on CPU | Measured field or frame cost blocks the target particle count. |

Known limits:

- Traits stay fixed during a rollout. There is no trait inheritance, mutation during life, birth, death, or resource cycle.
- The descriptor is hand-designed and versioned. It describes physical behavior without a projection, but it still encodes a prior. If v5 aliases visually different worlds, the next experiment is a distribution of local window vectors, which would create v6 after an ablation rather than a silent change mid-campaign.
- Expedition targeting in 53 dimensions is random goal direction, not coverage-aware illumination.
- The viewer reads archive recipes and replays initial seeds. Checkpoint browsing and a preset library are separate work.
- Campaign files continue evidence across batches; they do not resume a half-finished generation.
- Interactive room-scale rendering performance has not been established.
- No certified world has yet validated the complete 100,000-step product target. The recorded novelty-versus-random comparison predates descriptor and gate corrections, so its values do not count as evidence for the current search.
