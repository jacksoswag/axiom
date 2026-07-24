# AXIOM architectural audit

Audited against five goals: genuine N-dimensionality, learned-over-hand-tuned parameters,
extreme minimalism, conceptual unification, file-structure cleanliness. Every claim cites
`file:line` in the current tree. Every proposal states its cost. The best proposals were handed
to independent critics instructed to refute them; kills are reported alongside survivors.

**What was audited.** Commit `2a00622` plus ten uncommitted modified files (the `Fnv`
consolidation into `util.rs`, the `BRIGHTNESS_GAIN` deletion in `render_recipe.rs`, and fmt-style
rewraps). This report replaces an earlier same-day draft; every carried-forward claim was
re-verified against the tree, several were corrected (they are flagged where they appear), and
the measurements (tests, clippy, fmt, bench, CI status) were re-run first-hand.

One fact frames goals 1 and 2. The 3-D narrowing is a decision exactly one commit old:
`2a00622` regenerated all four doc surfaces to assert it, and the project's own notes from the
day before say "product direction should be fixed to 3-D at the search and viewer boundary."
This audit reverses that decision on the owner's instruction, so the work below is deliberate
reversal rather than cleanup of drift.

---

## The ten highest-value findings

| # | Finding | Where |
|---|---|---|
| 1 | **CI has been red on `main` since the rewrite.** The first step, `cargo fmt --all -- --check`, fails in 16 seconds before any test runs, and it enforces a formatter that rewrites the four hand-written files defining the repo's style (52 sites, 920 lines, 17 sites in those four). Delete the step. Separately, CI never compiles the viewer (`--no-default-features` only); add the default-features job, which is already the owner's own action item | ci.yml:19-21 |
| 2 | **The 3-D pin is 16 sites in one file plus one unguarded invariant.** `Caps::dimensions` is settable in code with nothing tying it to `metrics::DIMENSIONS`; a non-3 value silently corrupts 38 of 53 descriptor slots instead of failing. `dimensions` is also absent from the knob registry, `ExperimentIdentity`, and therefore the archive header. The descriptor's width is already dimension-invariant, so unpinning is bounded work, not a redesign | metrics.rs:14,117; genome.rs:188; campaign.rs:44-60 |
| 3 | **In-band structural search is a trap, and the earlier draft's plan for it is dead.** A critic proved novelty selection starves shape-grown offspring at reproduction (`pick_novelty` tournaments) and evicts them first at capacity; "neutral" initialization does not exist because anchor growth re-partitions every membership. The replacement is nearly free: sweep shape as per-campaign settings (the machinery already ships) plus one cross-shape seeding operator. Two hygiene guards land now regardless | search.rs:373-390,392-404; archive.rs:179-243 |
| 4 | **Tier step budgets sit outside the evidence guard.** Repair-probe length and the sampling window scale with `steps`, so robustness, mobility, and temporal variance change meaning with budget, and the README's own tutorial pools persistence labels across regimes while `--help` claims mixing is refused. Fix: per-tier probe/window fields under the digest, sticky identity for tier budgets | rollout.rs:284,418; campaign.rs:364-366; README.md:104-127 |
| 5 | **`learning.rs` (784 lines) is unreachable under defaults.** `promotion_budget = 0` empties the promotion queue, so no candidates, no labels, no model, ever; ~50 of its lines are dead even when enabled; and `train.rs` drops the model the campaign returns | tuning.rs:177; search.rs:49-52; campaign.rs:606; train.rs:129-148 |
| 6 | **One binary codec written twice, drifted three ways**: length bound present/absent, non-finite floats rejected/accepted, atomic-write durability split (retries without parent fsync versus parent fsync without retries). Merge into one `util` codec keeping both durability properties | campaign.rs:923-1151 vs checkpoint.rs:761-934 |
| 7 | **The checkpoint read path is built, precisely tested, and called by nothing**, while `save_world` writes on every certified run. Build the caller or delete the writer; the 10k-step exact-continuation test proves the hard part already works | checkpoint.rs:379,525,627,1169-1222 |
| 8 | **`SourceTier` is `EvaluationTier` declared twice**, dragging six hand-written enum↔u8 mapping pairs (~165 lines) and a validity check that exists only because two copies can disagree. Delete it; on-disk bytes unchanged | archive.rs:26-31; campaign.rs:1153-1350 |
| 9 | **The descriptor layout is declared five times** (doc, length, push order, novelty ranges, a test's hardcoded block starts) with nothing enforcing agreement; a silent divergence corrupts novelty hashing invisibly. One `const BLOCKS` table; prerequisite for every descriptor change here | metrics.rs:52-72,552-576; novelty.rs:14-30,96 |
| 10 | **The certification path re-simulates a sixth full-length rollout per certified world** (10% of tier compute), skips the gene clamp evaluation applies, and stores a reference-world render recipe; the viewer likewise never derives a recipe on adoption, so the material view can silently vanish for foreign worlds. `for_world` exists with one production caller | campaign.rs:776-788; mod.rs:87-115 |

Near misses: the fully redundant per-generation novelty refresh and the clone-churn around it
(search.rs:228, archive.rs:163-174); the two disagreeing trait palettes (theme.rs:34-43 vs
material.rs:551-569); three functions named `finite` with three meanings; `inspect` and
`snapshot` disagreeing on what `entry=0` names; goal sampling hardcoding a private constant's
value (search.rs:408).

### Adversarial verification scorecard

Four independent critics were given this audit's best proposals with instructions to refute
them; the two structural unifications killed in the earlier draft were re-verified analytically
rather than re-run. Corrections were adopted wholesale. The kills and corrections are the most
useful rows.

| Proposal | Verdict | Decisive reason |
|---|---|---|
| Delete the generation-start novelty refresh (search.rs:228) | **Survives, strengthened** | Dead in its entirety, generation 0 included: `merge` refreshes on both exit paths, bootstrap covers loop entry, and `Archive` has no interior mutability for the callback's `&Archive` to exploit |
| Leave-one-out novelty/crowding, deleting per-entry clones | **Survives, with a precondition** | Bit-identical only if the skip preserves iteration order (float-sum order feeds `select_nth_unstable`); never swap-remove. Cost honestly recalibrated: the default archive is bounded at 1,025 entries, so this is ~1-3% of wall-clock, a waste-shape finding |
| Stored `Shape` + structural mutation operators + local-competition protection | **KILLED at the core** | No neutral initialization exists (anchor growth re-partitions every membership and recalibrates every norm), and near-parent descriptors are exactly what `pick_novelty` starves and `merge` evicts first; archive-side protection fixes only the eviction half. Survivors: the two guards. Replacement: shape outer loop + cross-shape seeding |
| Tier-steps evidence hole | **Partially confirmed, mechanism corrected** | Per-genome relabeling is unreachable (`already_persisted` blocks re-evaluation); the live defect is population-level label pooling via the README's own resume tutorial. Naive identity inclusion would break that documented path; per-tier measurement fields plus sticky identity instead |
| Retain the certification final state instead of re-simulating | **Survives, numbers corrected** | Savings are 10%, not 17% (repair doubles per-seed cost); retention must be certification-gated and out of `CampaignRecord`. Bonus finding: `certified_preset` skips `resolve_clamped`'s clamp today |
| Derive stored and viewer recipes with `for_world` | **Survives / narrowed** | Checkpoint half is correct and currently unobservable (nothing reads recipes back). Viewer half must fire on adopt only; deriving on every rebuild would stomp the user's material sliders during drags |
| Unify the H0 union-find with the winding flood fill | **Killed** (inherited, re-verified) | They measure different objects (chaining vs level-set percolation) feeding a hard admission gate; ratio thresholds and length cutoffs are not interconvertible; nets more code |
| Replace the lattice with gliding-box windows | **Killed** (inherited, re-verified) | The shot-noise corrections are partition identities (overlap moves the neglected term from O(1/C) to O(1)) landing on absolute gate floors; `axial_correlation` has no point-cloud analogue |
| Occupancy-derived `side`, separable blur, `SourceTier` merge, codec merge, `const BLOCKS` | Survive | Re-derived directly (partition math, Künneth, byte-for-byte code comparison); unchallenged |

Claims of this audit's own earlier draft falsified during verification, kept visible on
purpose: the discovery-steps evidence hole (the identity already covers it, campaign.rs:56);
the per-genome label-AND mechanism (unreachable); the 17% certification savings (10%); the
particle-count framing of viewer recipe staleness (the driver is extent via coordination and
radius); and the novelty-pass cost digits (directionally right, decoratively precise).

---

## Goal 1: N-dimensional, genuinely

### Where generality dies

The engine keeps its own promise. `engine/mod.rs:5-6` states "Anchor count and dimensionality
are read from the data, never fixed by a constant," and it is true everywhere in `engine/`:
`Substrate::dims()` derives from the position layout ([substrate.rs:17-19](src/engine/substrate.rs)),
`Grid` walks a mixed-radix `3^dims` stencil ([grid.rs:66-80](src/engine/grid.rs)), `lenia::step`
loops `0..dims` ([lenia.rs:85,106](src/engine/lenia.rs)), `kernel::displacement` loops
`pos1.len()` ([kernel.rs:37,48](src/engine/kernel.rs)), and `GeometryScale::geometry` derives
extent with `powf(1.0 / dimensions)` ([geometry.rs:52-58](src/engine/geometry.rs)).

Generality dies in three places, and they are not equal.

**1. The descriptor: `const DIMENSIONS: usize = 3` at [metrics.rs:14](src/tuner/metrics.rs:14),
used at 16 sites in one file.** Worse than a limitation, it is an unguarded invariant.
`Caps::dimensions` is a plain `pub` field ([genome.rs:188](src/engine/genome.rs:188)) threaded
correctly through `Params`, `GeometryScale`, and every engine loop. Nothing ties it to the
metrics constant: no `dims == 3` check exists anywhere in the tree. Set it to 2 in code and
`raw_rdf` slices `positions[i * 3..(i + 1) * 3]` across particle boundaries
([metrics.rs:117,122](src/tuner/metrics.rs:117)): silent misalignment, not a panic, corrupting
38 of the 53 descriptor slots (18 RDF, 15 heterogeneity, 4 connectivity, 1 mobility). Whatever
else happens, either assert `substrate.dims() == DIMENSIONS` at the `metrics.rs` entry points or
delete the constant in favour of `substrate.dims()`.

Three more places the pin leaks past `metrics.rs`, all found by this pass:

- `dimensions` has no knob in the 43-entry registry ([tuning.rs:354-404](src/tuner/tuning.rs:354)),
  so the archive header, which is a knob dump ([archive.rs:248-250](src/tuner/archive.rs:248)),
  **cannot record dimensionality at all**. An N-D archive is not self-describing until the knob
  exists.
- `ExperimentIdentity` ([campaign.rs:44-60](src/tuner/campaign.rs:44)) carries particles,
  anchors, radius, rate, seed, shells, bumps, and discovery steps, but **not dimensions**. A
  dimension change would not even split the evidence ledger.
- `render_recipe.rs:99` uses `.cbrt()` where `geometry.rs:55` already has the general form. One
  line, outside the viewer feature.

The 16 sites split into three kinds:

- *Mechanical*: `raw_rdf` and `mobility` index by the constant where `substrate.at(i)` already
  exists and is generic (metrics.rs:117,122,481-492); `bin_range`'s `extent * 0.5 * 3.0f32.sqrt()`
  is the 3-torus half-diagonal, `sqrt(d)` in general (metrics.rs:98, and its `spacing` already
  uses `powf(1/d)` at :96).
- *Structural*: `SpatialField` is a hardcoded cube: `side.pow(3)` (metrics.rs:182), an `xyz`
  triple (:194-198), `index(x, y, z)` (:210-211), a hand-decomposed `neighbour` (:213-221), and
  `smoothed_density`'s 27-tap triple loop (:318-338). The winding flood fill uses
  `[0i32; DIMENSIONS]` const arrays (:359-361). Note the repo already contains the d-generic
  lattice idiom this needs: `Grid::cell_of` and `for_each_candidate` do mixed-radix indexing over
  `dims` ([grid.rs:66-97](src/engine/grid.rs:66)). Generalizing `SpatialField` is applying an
  idiom the engine already demonstrates.
- *Not a dimension problem*: everything else in the file.

**2. The viewer, irreducibly.** `DensityField` is a cube by construction
([material.rs:60-64,352-360](src/viewer/material.rs:352)), the trilinear sampler assumes three
axes (:314-350), `particles.rs:21` pins `DIMENSIONS = 3` with an honest comment, and the orbit
camera is built on the cross product ([camera.rs:17-23](src/viewer/camera.rs:17)), which exists
only in 3 and 7 dimensions. There is no N-D orbit camera. The honest treatment is a thin N-to-3
reduction (slice or PCA projection) in front of an unchanged `DensityField`, plus a refusal path
for high `d`. Do not make the voxel grid internally N-generic; that is a large rewrite for a
picture no one can read.

**3. The physics itself, as a research risk.** Chan's "Lenia and Expanded Universe" (ALIFE 2020)
reports, verbatim: "In higher dimensions, stable solitons are hard to find" and "in 3 or higher
dimensions, solitons become predominantly stationary." That is grid Lenia, not Particle-Lenia,
but it is the best available prior: N-D generality is a bet that interesting structure exists
above 3, not just an engineering task. Budget for `d = 4` yielding nothing certifiable.

### What a dimension-agnostic descriptor looks like

Start from the enabling fact, checked axis by axis against
[metrics.rs:61-72](src/tuner/metrics.rs:61): **the descriptor's width is already
dimension-invariant.** 18 RDF slots (3 trait bands × 6 radial bins), 15 heterogeneity (3 sides ×
5 statistics), 8 barcode, 4 connectivity, 8 dynamics. Not one count is a function of `d`.
Dimensionality changes how axes are computed, never how many there are. There is no
variable-width descriptor to design, no novelty metric to redefine, no archive column count to
make dynamic. The same is true of genome shape, which is what makes goal 2 tractable (below).

Per block:

| Block | Generalizes? | What changes with `d` |
|---|---|---|
| Trait-conditioned RDF | yes, already | Nothing structural: the normalization divides by a baseline measured on the same population ([metrics.rs:138-151](src/tuner/metrics.rs:138)), so shell-volume geometry cancels in any `d`. Only `bin_range`'s `sqrt(3)` needs `sqrt(d)` |
| Heterogeneity (5 stats) | yes, with the lattice fix below | The statistics are dimension-free reductions except `axial_correlation`, which is a lattice-adjacency statistic (correlates each cell with its `+1` neighbour per axis, metrics.rs:303-316). It already loops `0..DIMENSIONS` and averages over axes, so it yields one number in any `d`, but it requires a regular lattice: any replacement without a neighbour relation loses the axis |
| H0 barcode | yes, already | `persistence.rs` runs on the particle graph via `substrate.at()` and `displacement`; no dimension assumption exists in the file |
| Winding connectivity | yes, and the algorithm is already correct | `H1(T^d) ≅ ℤ^d` (Künneth), so a `d`-torus has exactly `d` independent non-contractible generators, one per axis. The per-axis lift tracking (metrics.rs:352-388) computes precisely that invariant in one flood fill. Replacing it with a persistent-homology library would be a cost regression (Vietoris-Rips complexes grow combinatorially in points and dimension). The fix is deleting the const-sized arrays around it, not the algorithm |
| Dynamics (temporal, mobility, turnover, asymmetry, repair) | yes | Nothing; `asymmetry` reduces the weight matrix to one scalar (metrics.rs:506-523) |

**The real ceiling is the lattice, and the fix is the side, not the grid.** Heterogeneity and
connectivity fields cost `side^d` cells. At 1,200 particles, side 16:

| `d` | cells | mean occupancy | Poisson void fraction | headroom before `void` saturates |
|---|---|---|---|---|
| 3 | 4,096 | 0.293 | 0.746 | 0.254 |
| 4 | 65,536 | 0.018 | 0.982 | 0.018 |

One added dimension at fixed side shrinks occupancy 16-fold and crushes the void axis's dynamic
range 14-fold. But the lattice already gets one thing right that any replacement must preserve:
`side` is a fixed cell *count*, so cell size is `extent / side` and auto-rescales with each
genome's derived extent (metrics.rs:195-197). The scales stay relative and comparable across
genomes.

The minimal fix keeps that property: **derive `side` from a target occupancy**,
`side = round((particles / occupancy)^(1/d))`, evaluated at the three occupancy targets the
current sides imply in 3-D (18.75, 2.34, 0.293 particles per cell), instead of the literal
`[4, 8, 16]` (metrics.rs:18). In `d = 3` at 1,200 particles this reproduces today's sides
exactly; in `d = 4` it picks side 8 and restores occupancy 0.293. It also removes the descriptor's
worst particle-count dependence (`void` is an occupancy function, metrics.rs:237), which is what
currently blocks `particles` from ever becoming a gene. One function, descriptor version 6.

Be honest about what it does not buy: constant occupancy in higher `d` means coarser cells, so
resolvable structure coarsens as `d` grows. That is the curse of dimensionality arriving where
it must; the lever that buys resolution back is more particles, which costs rollout time.

Two mechanical wins ride along:

- `smoothed_density`'s `[0.25, 0.5, 0.25]` weights are separable: three axis passes of 3 taps
  replace the `3^d` tensor loop. That is `3d` versus `3^d` work per cell, a strict win in 3-D
  too (9 versus 27), and the difference between feasible and not above `d = 5`.
- `mass()`'s eighth axis is `2·ln(components)/ln(particles)` (persistence.rs:121-124), which is
  not scale-free: ten components of 300 particles reads 0.807, a hundred of 3,000 reads 1.151,
  identical fractional fragmentation 42% apart. Replace with the component fraction `C/N`
  (log-scaled if the dynamic range needs it) in the same descriptor bump.

### What dimensionality costs elsewhere

- **Neighbour search.** The `3^d` stencil (grid.rs:68) plus the `1 << 21` cell cap (grid.rs:28)
  bound the useful range: at `d = 4` the cap allows side ≤ 38, at `d = 6` side ≤ 11, and the
  stencil itself grows 3^d. Beyond `d ≈ 5-6` the grid deactivates or stops paying, and the
  silent O(N²) fallback (`for_each_candidate`, grid.rs:59-64) carries the load. At the particle
  counts this project runs, that is acceptable and needs no code: report it, do not fight it.
- **The density probe.** `measure_coordination` (geometry.rs:68-106) already loops over `dims`,
  is capped at 1,500 probe particles, and runs once per campaign. Generalizes unchanged.
- **The renderer.** See above: reduce N to 3 in front of it, or refuse.
- **`uniform_density`** (material.rs:525-529) is closed-form 3-D (`extent³`, `support³`,
  `64π/315`). A general-`d` closed form exists (the kernel integral is a beta function times the
  `d`-sphere surface), one line of math when the viewer's reduction layer needs it.

### Should `dimensions` be searched?

**Make it a setting now; do not make it a gene.** The setting is nearly free once the descriptor
unpins: add the knob, add `dimensions` to `ExperimentIdentity`, delete the constant. As a gene it
fails on physics: dimensionality changes the regime wholesale (see Chan above), so mutating it is
a restart, not a local move, and cross-`d` descriptor values (RDF shell geometry, occupancy
statistics, winding counts) mean different things even at equal width. Run one campaign per `d`
and compare archives; the identity machinery already keeps that honest.

### Already right, one line each

- The interaction norm is measured, not computed from a dimension table
  ([interaction.rs:106-166](src/engine/interaction.rs:106)); the Particle-Lenia reference
  implementation normalizes with a hardcoded `{2: 2, 3: 4}·π` table supporting exactly two
  dimensionalities. Measuring beats the reference and is dimension-correct for free.
- Extent is derived, never a free parameter (geometry.rs:1-3, 50-63).
- The winding measure computes the mathematically right invariant cheaply (metrics.rs:342-388).

---

## Goal 2: Learned, not hand-tuned

### The complete parameter map

**Learned today (genes).** Coordination (`genome[0]`, bounds 3-20, genome.rs:21,252-253);
initial trait-density logits, one per anchor (bounds ±4, genome.rs:22,260-265); and per ordered
anchor pair: `shells × [amp, mu, sigma]`, `bumps × [amp, mu, sigma]`, and a directed weight
(bounds genome.rs:134-152). That is the entire interaction law plus the world's density scale.
It is a good set.

**Hand-set in `Caps`** ("the fixed parts of a world that no genome may vary," genome.rs:183-184):

| Field | Line | Knob? | Learnable? | What blocks it today |
|---|---|---|---|---|
| `particles` | 187 | yes | with care, later | Campaign-shared density probe (geometry.rs:24-27); `void` is an occupancy function (metrics.rs:237: 300 particles at side 16 reads ~0.93, 5,000 reads ~0.29 for identical behaviour); H0's eighth axis is `ln`-of-count scaled (persistence.rs:121-124); measured 14x cost spread 300→5,000 (0.39 → 5.48 ms/step) with no cost-aware scheduling and fully serial evaluation above discovery (rollout.rs:216-239, campaign.rs:600-638) |
| `dimensions` | 188 | **no knob** | as a setting yes; as a gene no | Everything in goal 1; absent from identity and header |
| `anchors` | 189 | yes | **yes; the decode already supports it** | No operator changes genome length (search.rs:373-390); one campaign-wide bounds vector (search.rs:194); fixed archive row width (archive.rs:304-306); no length guard in `Archive::merge` (archive.rs:179-243); `ExperimentIdentity` pins it (campaign.rs:50) |
| `radius` | 190 | yes | with care | Sets the gene bounds themselves (genome.rs:134-140), so learning it means per-genome bounds; probe depends on it (geometry.rs:35) |
| `rate` | 191 | yes | **no** | `dt = 1/rate` (genome.rs:71-73) while every dynamics metric is measured over fixed step counts (rollout.rs:284-286): a free `rate` gene is free time dilation of the mobility, variance, and autocorrelation axes and of the Frozen gate |
| `seed` | 192 | yes | **no, correctly** | An evaluation control; a seed gene lets a genome pick its own luck |
| `shells` | 193 | yes | yes | `Layout::for_genome` recovers only `anchors` from length; shells and bumps come from `Caps` (genome.rs:112-120), so the genome is not self-describing for them |
| `bumps` | 194 | yes | yes | Same |

**Hardcoded constants that bound the expressible space** (meta-parameters; a search cannot widen
its own bounds without an outer loop, and widening them is an experiment, not a defect):
`MAX_KERNEL_REACH = 0.375`, `COORDINATION_BOUNDS`, `TRAIT_LOGIT_BOUNDS` (genome.rs:20-22),
weight ±100 and bump mu/sigma bounds (genome.rs:147-149), `CUTOFF_SIGMA = 3.5` (kernel.rs:63),
`TRIAL_EXTENT_RADII = 6.0` and `PROBE_PARTICLES = 1500` (geometry.rs:9-10).

**A hand-designed prior not labelled as one.** `Layout::default_genome` (genome.rs:154-176)
builds a specific starting law (first shell amp 1.0 at `mu = radius`, `sigma = radius/2`, first
bump at 1.5/0.5, diagonal weight 40), and every search seeds with it (search.rs:199). The
coordination prior 9.0 is additionally written twice (genome.rs:58 and :284). Decide: keep it as
a labelled baseline, or drop it and let bootstrap be purely random. Either is fine; unlabelled is
not.

**Search hyperparameters.** The other 36 knobs, plus seven `LogisticConfig` fields that are not
knobs at all (learning.rs:143-165), the tier constants (seeds/passes/minimum steps,
rollout.rs:52-79), `NEIGHBOURS = 15` (novelty.rs:9-11), `SAMPLES = 8` (rollout.rs:26), and the
descriptor geometry constants (metrics.rs:16-25).

### Machinery that exists but is never exercised

The sharpest recurring pattern in the codebase:

1. **`Layout::for_genome`** (genome.rs:107-120) inverts the length quadratic, is documented as
   "what lets one search carry genomes of differing anchor counts," and the spec repeats the
   claim (TECHNICAL_SPEC.md:107). No mutation operator ever changes a length. `iso_line_dd_scaled`
   zips `a.iter().zip(b).zip(bounds)` (search.rs:382-384), so a mismatch does not even panic: it
   **silently truncates** to the shortest of the three. Harmless today only because an invariant
   two layers up guarantees equal lengths; the moment shapes vary it is silent corruption. The
   viewer, meanwhile, resizes anchors, shells, and bumps live (controls.rs:54-62), proving the
   decode side works.
2. **`Caps::dimensions`**: threaded everywhere, settable nowhere (goal 1).
3. **`curated_parent_indices`** (tuning.rs:163-165): a `ParentSource::Curated` variant, a branch
   reserving half the expedition slots (search.rs:294-315), archive serialization support, and a
   doc comment describing a viewer feature that does not exist. It is assigned only by `Default`
   (tuning.rs:181): always empty in every run that can occur. Against this goal it is worse than
   dead weight: it is wired-up surface asserting a human picks parents.
4. **The checkpoint read path** (`restore_world`, `load_world`, `load_manifest`,
   checkpoint.rs:627,525,379): fully built, precisely tested, called by nothing outside tests,
   while `save_world` runs on every certified world. The README even admits it
   (README.md:161: "Checkpoint browsing and restoration are separate work").
5. **`SchedulingAuthority` diagnostics**: `pr_auc`, `calibration_error`, `precision_at_budget`
   are computed (learning.rs:417-420) and read by nothing; the authority decision uses only
   Brier, recall, and lineage counts (learning.rs:421-424), and `train.rs` never prints any of
   it. The feedback signal needed to adapt quotas already exists and is discarded.
6. **The whole learning subsystem is unreachable under defaults.** `promotion_budget` defaults
   to 0 (tuning.rs:177), so `PromotionQueue::push_generation` returns immediately
   (search.rs:49-52), so the campaign has no candidates, so the persistence loop never runs, so
   no labels exist, so the model never fits. 784 lines (learning.rs) plus the scheduling half of
   campaign.rs have an empty behavioural surface in the default configuration, and the second
   `train_persistence` fit (campaign.rs:606) is returned to the one caller, `train.rs`, which
   drops it (train.rs:129-148 reads only `.search` and `.certifications`).

### What making structure searchable costs

The earlier draft proposed in-band structural evolution: store `Shape {anchors, shells, bumps}`
per entry (archive v10), per-genome bounds, add/remove-anchor operators with neutral
initialization, same-shape mating, and local-competition protection in the archive. **An
independent critic killed the core of that plan, and the kill is correct.** Reported in full
because the reasoning changes what should be built instead.

**The kill: novelty selection structurally suppresses fresh structure, at both ends.** A grown
genome cannot be initialized neutrally here: `membership` bins traits by `trait_value * anchors`
(trait.rs:53-58), so adding an anchor re-partitions every particle's membership and recalibrates
every pair norm (interaction.rs:108-166); the offspring is a diffuse perturbation of its parent,
not a clone. Either way its descriptor lands near the parent's, and near-parent is exactly what
the machinery punishes: low admission novelty (search.rs:462), low crowding, which is the
double-weighted ranking in `merge` (archive.rs:231-238), and near-zero probability of winning a
`pick_novelty` tournament (search.rs:392-404), which is the only path to being mutated further.
Under capacity the entry sits inert; at capacity it is evicted first. Protection in the archive
(the proposal's own mitigation) treats only the eviction half; reproduction starvation remains.
Reaching a shape six steps from the default means six compounding founder events, each starved
this way. The selection pressure the proposal relies on to explore structure is the same
pressure guaranteed to suppress it.

Two more independent hits: the persistence learner's features carry no shape covariate
(campaign.rs:1269-1334), so pooling shapes into one ledger hands a 30-row training set
(tuning.rs:206) an unmodelable confounder, and today's `ExperimentIdentity` is what correctly
prevents that; and the honest code delta is 500-850 production lines across five files (the
draft's scope list omitted `checkpoint.rs`, whose call sites assume campaign shape equals
candidate shape).

**What survives, and what replaces the rest:**

- *Hygiene now, regardless of everything else*: make `iso_line_dd_scaled` assert equal lengths
  (it silently truncates today, search.rs:382-384), and give `Archive::merge` the genome-length
  guard it lacks (archive.rs:179-243; its own tests admit length-1 genomes, archive.rs:562-574).
  A few lines each, no downside.
- *The replacement: a shape outer loop.* The 84-point shape space (anchors 2-8 × shells 1-4 ×
  bumps 1-3, the viewer's own bounds at controls.rs:54-62) is small, and the machinery for
  searching it already ships: `anchors`/`shells`/`bumps` are knobs, `ExperimentIdentity` keeps
  per-shape evidence separate, and every shape gets full-strength search instead of a starved
  minority. Sweep it; later, allocate campaigns by a bandit over shapes if the sweep shows the
  space is worth adaptive effort.
- *One new operator, at the right level*: cross-shape seeding. Embed an N-anchor champion into
  an (N+1)-anchor campaign's bootstrap by inserting one neutral logit and zero-amplitude blocks
  (a decode-level transform, ~30 lines, no archive or format change). This dissolves the
  starvation argument entirely: the embedded genome competes in the *destination shape's*
  archive, where its parent is absent and its behaviour may be genuinely novel. It is the part
  of structural search that survives contact with the selection dynamics.
- `Layout::for_genome`'s inference machinery stops being load-bearing under this plan (each
  campaign knows its shape), but it is 9 correct lines and the genome stays self-describing for
  anchor count; keep it.

This reverses the earlier draft's recommendation. The distance to "anchor count is learned" is
not a plumbing refactor away; it is one honest outer loop away, and the outer loop is nearly
free.

### The evidence-comparability line, and the hole in it

`Tuning::digest` draws the line correctly in principle: settings that change what an evaluation
means fold into the digest; settings that change how much searching happens do not
(tuning.rs:229-232). `ExperimentIdentity` carries the digest plus the world constants plus
`discovery_steps` (campaign.rs:56), and `CampaignState::bind` refuses mismatches
(campaign.rs:443-452). An earlier draft of this audit claimed discovery step budgets could
silently split evidence; that was wrong, the identity already covers them.

**The hole that remains is the tier step budgets, and an independent critic sharpened both its
mechanism and its fix.** `tiers.persistence_steps` and `certification_steps` appear in neither
the identity nor the digest (the digest body never references `self.tiers`, tuning.rs:233-297),
while step count parameterizes what gets measured: the repair probe runs `(steps/4).max(50)`
(rollout.rs:418), and the sampling interval is `steps/(2·SAMPLES)` (rollout.rs:284), so
robustness, mobility, temporal variance, and turnover, which feed the Fragile and Frozen gates
(viability.rs:100-105), all shift meaning with the budget. `budget_valid_for_source` only floors
(`steps >= tier minimum`, campaign.rs:364-366).

Three critic corrections to how this bites:

- The per-genome label-relabeling this audit first described **cannot occur**: `run()` builds
  `already_persisted` from every prior persistence record regardless of budget
  (campaign.rs:562-568), so one genome is never persistence-evaluated twice. The conservative
  AND (campaign.rs:229-235) only ever sees identical operands in production.
- The live mechanism is coarser and arguably worse: **population-level pooling**. Every resume
  with a different `persistence_steps` adds labels measured under a different regime to the one
  training set the scheduler fits (campaign.rs:218-252) and the one eligibility set
  certification reads (campaign.rs:727-767). Features are immune (they come from
  identity-locked discovery rows, campaign.rs:243); only the `survives` labels carry the noise,
  which silently degrades the model and the gate every future genome passes through.
- This is not an exotic path. The README's own tutorial resumes one `state=campaign.state`
  while switching `persistence_steps` from off to 10,000 and then adding certification
  (README.md:104-127), and both `train --help` (train.rs:17-18) and README.md:42 assert that
  differing measurement settings are "refused rather than silently mixing," which is false for
  exactly these two knobs.

The fix is also not the obvious one. Adding tier steps to `ExperimentIdentity` with plain
equality **breaks the documented golden path** (off → on trips `ExperimentMismatch` on the
tutorial's second command), and naive equality-against-current in `budget_valid_for_source`
silently discards all historical labels the moment a discovery-only resume runs. What survives
review: (1) decouple the measurements, making the repair-probe length and sampling window
explicit per-tier `Repair`/tier fields covered by the digest, so step budget stops changing
metric semantics at all (a flat probe length would wreck the deliberate tier cost ladder,
README.md:78, so per-tier it must be); and (2) track tier budgets in the identity with sticky
`Option` semantics (`None` compatible with anything, `Some(x)` must match `Some(x)`). Descriptor
comparison across tiers, for the record, never happens anywhere (the archive only ever holds
discovery-tier descriptors), so the blast radius is labels and gates, not novelty space.

**Gates must stay fixed within a campaign, and that is correct.** They are digest-covered, so
changing them forks the ledger by design. Say it in the docs as a feature; the current doc
framing celebrates the opposite ("adding one reaches all three surfaces automatically,"
TECHNICAL_SPEC.md:124).

### Which search hyperparameters could adapt

Ranked by strength of evidence, not appeal:

1. **Lane shares → a bandit.** `Lanes` is already an isolated decision point (tuning.rs:43-78).
   Multi-Emitter MAP-Elites runs exactly this bandit over emitters; Monte Carlo Elites frames
   parent selection itself as one. Best-supported adaptive change available.
2. **Tier step budgets → measured relaxation.** The rollout already computes autocorrelation at
   lags 1, 2, 4 (metrics.rs:409-432) and discards it for scheduling. A budget keyed to when a
   world stops changing is buildable from data already collected.
3. **Continuation quotas → the authority signal already computed** (finding 5 above): the loop
   is half-built; `schedule` reads one boolean (campaign.rs:697-715).
4. **Mutation scales → self-adaptation. Weakest case.** Every working precedent (NS-ES,
   NSRA-ES, CMA-ME) first collapses novelty to a scalar inside otherwise fitness-shaped
   machinery; no published method adapts step size off a raw multi-axis novelty signal. Budget
   for a scalarization layer or skip it.
5. **Promotion thresholds → calibration first.** There is no validated short-horizon surrogate
   for 100k-step survival in continuous artificial life, and the owner's own action item (the
   novelty-stratified calibration set) is the right response. Until it exists, the tier ladder's
   promotion rule is an assumption; do not tune it, measure it.

`NEIGHBOURS`, the descriptor geometry, and the axis scalings must stay fixed per descriptor
version: they define what the space means, and novelty distances across an archive are only
comparable because they never move (the file says this correctly at tuning.rs:7-10).

---

## Goal 3: Extreme minimalism

Measured against `trait.rs` and `kernel.rs`. The style rules live in
[docs/STYLE.md](docs/STYLE.md); this section is substance.

### Guards: load-bearing or ceremony

The test is which side of a parse boundary a guard sits. `checkpoint.rs` gets the read direction
right and the write direction wrong:

- **Ceremony, delete**: `write_string` (checkpoint.rs:809-819) and `write_f32s` (:821-833)
  re-check emptiness and bounds after `.validate()` proved them, with one precise exception this
  pass found: a zero-particle `WorldState` passes `validate()` (empty positions divide evenly,
  checkpoint.rs:299-311) and only dies at `write_f32s`'s emptiness check. So first make
  `validate()` reject zero particles, then delete the write-side guards.
- **Load-bearing, keep**: `Reader::string`/`f32s` (checkpoint.rs:891-923) reject the same
  conditions on untrusted bytes. Correct.
- **Genome length is validated at three layers**: `Caps::resolve` panics (genome.rs:239-246),
  `World::with_geometry` asserts (world.rs:53-57), `World::set_genome` asserts (world.rs:111-116).
  Keep one.
- **Feature rows validated twice back-to-back**: `validate_rows` (learning.rs:250 via :606-612)
  then `Standardizer::fit` re-checks `schema.accepts` per row (learning.rs:55) on the subset it
  was just handed.
- **`budget_valid_for_source` exists because two types can disagree** and is re-derived at four
  sites (campaign.rs:364-366, used :225,:230,:508,:743). Deleting `SourceTier` (goal 4) removes
  the need.
- **`train.rs::check`** (train.rs:154-180) re-implements `archive::validate_tuning`
  (archive.rs:507-522) almost clause for clause. The CLI boundary justifies *a* check; it does
  not justify a second copy of the same one. Call the existing validator.

### Error machinery with no consumer

- `material::Error`, five variants (material.rs:16-23), every consumer collapses it:
  `.is_ok()` at mod.rs:130-144, `format!("{problem:?}")` at mod.rs:363. Collapse to the boolean
  it already is, or to one message.
- `LearnError`, four variants (learning.rs:598-604), every caller `.ok()`s it
  (campaign.rs:678-682, :700-707). The granularity has no consumer.
- **Counter-example, keep**: `checkpoint::Error`'s 14 variants and the campaign errors are all
  constructed and asserted against by name in tests (checkpoint.rs:1096-1139). That machinery is
  earned. Do not flatten it in the same sweep.

### Dead second implementations and drifted twins

- `material::shade` (material.rs:174-214) is a second shading path, `#[cfg(test)]`-only, used by
  one test, and its constants have already drifted from production: emission `0.18 + 0.38·m²`
  versus `0.14 + 0.7034908·m² + 0.55·activity`, alpha clamp 0.85 versus 0.92
  (material.rs:198,201 vs :292,299). Delete; repoint the test at `shade_camera`.
- `material::torus_distance` (material.rs:541-549) reimplements `kernel::displacement` with
  three hardcoded axes.
- `dot` is declared twice, character for character (camera.rs:14-16, material.rs:582-584), and
  the two normalizers disagree on the zero vector: `camera::unit([0,0,0])` returns `[0,0,1]`,
  `material::normalise([0,0,0])` returns `[0,0,0]`, and the divergence is reachable because
  `normalise` is fed `gradient(..)`, which is genuinely near-zero in flat regions. Harmless
  today (a zero normal zeroes a shading term), but unify `dot` and pick one zero contract on
  purpose.
- The gene triple is decoded in three places: `read_shells` and `read_bumps` are byte-identical
  (trait_editor.rs:230-248), and a third copy is inlined in `Net::from_genome`
  (interaction.rs:43-51) without the `.max(1e-4)` width floor the other two apply. Production is
  safe (bounds already clamp sigma, genome.rs:137,147), but three decoders of one format have
  already drifted once.
- `finite` means three different things: replace-with-fallback (util.rs:60-66),
  reject-a-slice (checkpoint.rs:689-695), check-a-substrate (rollout.rs:20-22). Rename two.
- `search.rs:408` samples goal targets with `rng.range(0.0, 2.0)` where `2.0` is
  `metrics::AXIS_SPAN`, which is **private** (metrics.rs:24), so the literal cannot even
  reference the constant it duplicates. Changing the descriptor span silently desynchronizes
  goal sampling.
- The two trait palettes disagree: `theme::TRAIT_STOPS` is eight stops
  (theme.rs:34-43), `material::palette` is four different ones (material.rs:551-569), and both
  are visible at once with "particle grain" enabled over the material view. Converge them.

### Latent footguns where structural work will land

`goal_parent` indexes `entries[0]` unconditionally (search.rs:407) and is safe only because its
call site gates on `archive.len() >= 2` (search.rs:294); `iso_line_dd_scaled` truncates silently
(above). Both are caller-held invariants sitting exactly where goal-2 changes will land. Make the
second assert its precondition regardless of anything else in this report.

### Redundant work in hot and warm paths

- Per descriptor sample, `spatial_field` is built five times for three distinct sides:
  side 16 for turnover (rollout.rs:350), sides 4/8/16 inside `heterogeneity` (metrics.rs:390-400),
  side 8 again inside `connectivity` (metrics.rs:405-407). Two of five builds are exact
  duplicates. Pass the fields in.
- `raw_rdf` is O(N²), single-threaded, and runs ~25 times per discovery rollout (baseline
  rollout.rs:283, ~8 watchdog checks :315-319, 8 samples :348, 8 in the repair protocol via
  `recovery_signature` :532-537). At 1,200 particles that is ~18M pair evaluations against a
  rayon-parallel physics step (lenia.rs:41). Calibration, measured this pass: roughly 5% of a
  discovery rollout; it overtakes the physics somewhere past ~25k particles. This, not the
  physics, is what blocks large `particles` values; the owner's action item for a stratified
  pair estimator is the right fix and only needed at scale. (`examples/bench.rs:6` claims to
  make "the O(N²) → O(N·k) claim a measurement"; it measures the physics step only.)
- `Archive` novelty maintenance clones the world per entry: `refresh_current_novelty`
  (archive.rs:163-174), `score_combined` (:422-436), and the crowding pass inside `merge`
  (:218-230) each build, per entry, a filtered clone of every other descriptor: n·(n-1) cloned
  53-float vectors per pass, three to four passes per generation. An independent critic verified
  the fix and bounded the cost honestly: a leave-one-out variant of `novelty()`/`crowding()`
  taking a skip index is bit-identical **provided the skip preserves iteration order**
  (`enumerate().filter()`, never swap-remove, since float summation order feeds
  `select_nth_unstable`), and at defaults the archive can never exceed 1,025 entries (65
  bootstrap + 30 × 32 evaluations against capacity 2,000, so `truncate` is provably a no-op),
  putting the whole pattern at roughly 1-3% of campaign wall-clock. This is a waste-shape
  finding, not a bottleneck fix; it matters because capacity-scale archives are exactly where
  goal 2 wants to go.
- `persist_state` rewrites the entire ledger file after every persistence and certification
  candidate (campaign.rs:604,:624), and `append_search_discovery` dedupes with a linear scan per
  record (campaign.rs:143-151), quadratic across resumed campaigns. Both are fine at today's
  scale; both are the first things to notice when campaigns grow.

### The certification path wastes a rollout and stores the wrong recipe

Critic-verified, with two corrections to this audit's first framing:

- **`certified_preset` re-simulates the entire certification budget** (100,000 bare
  `world.step()` calls, campaign.rs:780-782) solely to reconstruct the final state the
  evaluation already computed for the passing seed and dropped. Retaining the final substrate
  per certification seed costs ~19 KB at 1,200 particles (~313 KB at 20,000), evaluated
  sequentially, so memory is a non-issue. The honest savings figure is **10%, not the 17% first
  claimed**: the repair protocol doubles every seed's cost (control plus three probes at
  `steps/4` each equals one extra `steps`), so five seeds cost ten rollout-equivalents and the
  re-simulation is one. Implementation constraint from review: the retained substrate must be
  certification-gated and must not thread into the durable `CampaignRecord`.
- The re-simulation is not even provably identical to what was evaluated: `certified_preset`
  decodes with bare `caps.resolve` (campaign.rs:776) and **skips the gene clamp**
  `resolve_clamped` applies before every evaluation (rollout.rs:183-195). Harmless in the
  current parameter regime (mutation already clamps into bounds computed at the smallest
  possible extent), but that is an emergent property of today's constants, not an invariant.
  Route it through the same decode path.
- The stored render recipe is `RenderRecipe::default()` with a support clamp
  (campaign.rs:787-788): a recipe derived for the 320-particle reference world, not the
  certified one. `RenderRecipe::for_world(geometry.extent, params.particles)` exists for
  exactly this (render_recipe.rs:97-110) and has exactly one production caller
  (examples/snapshot.rs:72,86). Review confirmed the fix is safe under the retained clamp and
  that the mismatch driver is the candidate's coordination (through extent), not particle
  count. Also honest: nothing reads stored recipes back today, so this matters precisely as
  much as the checkpoint read-path decision it rides on.
- The viewer has the same disease live: `rebuild` and `adopt_selected` (mod.rs:87-115) never
  touch the material recipe, `DensityField::from_particles` rejects `support >= extent/2`
  (material.rs:74-81), and the error collapses to a silent bead-renderer fallback
  (mod.rs:130-144) with no status message. Adopt a foreign run with a different radius or
  coordination and the material view can simply vanish. Review narrowed the fix: derive via
  `for_world` **on adopt only**; deriving on every `rebuild` would stomp the user's material
  sliders continuously during world-slider drags (egui fires `changed()` per frame,
  controls.rs:22-78). The bug is invisible at startup only because `App::new`'s hardcoded
  params exactly match the reference world (mod.rs:48-59).

### Over-exposed API

`campaign.rs` exposes 19 top-level `pub` items; `train.rs`, the sole consumer, uses 4 (`run`,
`validate`, `CampaignPersistence`, `CampaignState`). `checkpoint.rs` exposes 10 functions and 4
constants; campaign needs 2 and 2. Because the crate is a `[lib]`, `pub` items never trigger
`dead_code`, so the compiler is blind to every dead item in this report. Narrow visibility
first; then `-W dead_code` finds the rest mechanically.

### Over-documented constants

`render_recipe.rs` still carries a 24-line essay on `GRADIENT_OVERSAMPLE`
(render_recipe.rs:23-46), and its module header claims "Nothing here is an absolute constant
tuned by eye" (render_recipe.rs:7) three lines above two constants explicitly labelled "visual
taste" (:49-58). The bar is kernel.rs:62: one line, one number, one justification.

---

## Goal 4: Conceptual unification

Ranked by value; each deletes more than it adds.

**1. `SourceTier` and `EvaluationTier` are one enum.** Identical variants in identical order
(archive.rs:26-31, rollout.rs:42-47), identical on-disk codes (campaign.rs:1174-1206), a
derived-at-write field (`source_tier: source_tier(evaluation.budget.tier)`, campaign.rs:197),
two conversion functions (campaign.rs:1336-1350), and a validity check that exists only because
the two can disagree (campaign.rs:364-366). Delete `SourceTier`. Cost: archive.rs, one line in
search.rs:456, ~50 lines out of campaign.rs; on-disk bytes unchanged.

**2. One binary codec instead of two.** `campaign::StateReader` (campaign.rs:962-1151) and
`checkpoint::Reader` (checkpoint.rs:835-934) are the same machine written twice, and they have
drifted three ways, found by direct comparison this pass:

- `checkpoint::write_f32s` bounds length by `MAX_VALUES`; `campaign::put_f32s` writes unbounded
  (checkpoint.rs:821-833 vs campaign.rs:923-928), and campaign.rs:1005 open-codes `1 << 28`
  where checkpoint.rs:25 names it.
- `checkpoint::Reader::f32` rejects non-finite bits (checkpoint.rs:883-890);
  `campaign::StateReader::f32` accepts them (campaign.rs:1011-1013).
- The two `write_atomic`s give different durability, neither strictly better: checkpoint's
  retries 128 temp names but never fsyncs the parent directory (checkpoint.rs:761-797);
  campaign's fsyncs the parent but has one PID-keyed temp name and no retry
  (campaign.rs:936-960).

Merge into one reader/writer pair plus one `write_atomic` keeping both durability properties, in
`util.rs` (it passes the util bar: no natural owner, serves two subsystems, and the duplication
is actively drifting). Do not reach for serde: two dependencies is a feature, and both formats
depend on exact f32 bit patterns. Cost: ~150-200 lines deleted, one shared error story, the
largest single change in this section.

**3. Declare the descriptor layout once.** It is currently stated five times: the doc comment
(metrics.rs:52-60), `descriptor_len()` (:61-72), the push order in `descriptor()` (:552-576),
the hand-computed ranges in `neighborhood_key` (novelty.rs:14-30), and the hardcoded block
starts in novelty's own test (`[0, 18, 33, 41, 43, 45, 49, 51]`, novelty.rs:96). Nothing
enforces agreement; if the push order and the ranges diverge, novelty silently hashes wrong
blocks. One `const BLOCKS: &[(&str, usize)]` drives the length, the ranges, the archive width,
and the doc. Prerequisite for every descriptor change in goal 1.

**4. `feature_schema()` and `feature_values()` are two hand-synced lists** (campaign.rs:1269-1297
names, :1299-1334 values); only the width is ever checked (learning.rs:34-36). The crate already
solved this exact problem with the `KNOBS` table (tuning.rs:300-304, "a knob cannot exist
without documentation or drift out of sync"). Apply the same declare-once pattern. Local change,
no format break.

**5. Name the five heterogeneity values.** `heterogeneity()` returns a bare `[f32; 5]`
(metrics.rs:282-289) indexed positionally at three sites: `scale[0]`/`scale[2]` with a literal
`5` in `margins` (viability.rs:65-74) and a `match i % HETEROGENEITY_VALUES` scaling table
(metrics.rs:544-549). A five-field struct deletes the positional coupling and the magic 5.

**6. One checked genome-length formula.** `WorldManifest::validate` re-derives
`Layout::genome_len()`'s polynomial in 27 lines of checked arithmetic (checkpoint.rs:227-254 vs
genome.rs:122-128). Checked arithmetic at the parse boundary is right; typing the polynomial
twice is not. Add a checked variant beside the authoritative one.

**7. Six enum↔u8 mapping pairs**, 115 hand-written mirror lines (campaign.rs:1153-1267). The
crate has a macro idiom for exactly this (tuning.rs:320-352).

**8. `save_state` and `state_checksum` share 14 byte-identical serialization lines**
(checkpoint.rs:442-459 vs :670-687). Extract `encode_state`.

**9. `Params` is `Caps` plus two fields.** Eight fields repeat (genome.rs:37-49 vs :186-195);
`Params::default` already delegates to `Caps::default` field by field (:51-68). If goal 2's
shape work happens this collapses on its own; if not, it is still two declarations of one
concept.

**10. `SpatialField` should borrow `Grid`'s indexing idiom** (goal 1): the d-generic mixed-radix
walk already exists in the repo (grid.rs:66-97); metrics' cube (metrics.rs:210-221) is the same
concept hand-specialized to 3.

### Proposals killed by adversarial review

Two structural unifications from the earlier draft were killed by independent critics, and this
pass re-verified the killing arguments rather than re-running them. They are recorded because
both are attractive and both are dead ends.

**Unify the H0 particle-graph union-find with the voxel winding flood fill.** Killed.
(1) They measure different objects: single-linkage chaining on raw particles versus level-set
percolation of a smoothed field thresholded against its own mean (metrics.rs:290-300); two dense
blobs joined by one within-cutoff filament merge under chaining and separate under the blur,
same configuration, opposite verdict, and this is stock Lenia morphology. (2) The lattice
version feeds a hard gate (`MaterialDisconnected`/`VoidDisconnected` against
`connected_fraction_floor`, viability.rs:82-85,110-122), so swapping measurement changes which
genomes live. (3) The thresholds are not interconvertible: `DENSITY_THRESHOLDS` are ratios
against the configuration's own mean (self-normalizing per genome, metrics.rs:295-297); a
union-find cutoff is a length, and inventing per-genome radii reintroduces the guess-the-scale
problem `persistence.rs:10-13` exists to avoid. (4) It is more code than it deletes once the
void phase needs probe generation and a lift-carrying union-find. (5) A fixed probe count
under genome-varying extent biases a hard gate; the lattice avoids that for free because `side`
is a count.

**Replace the lattice with gliding-box window sampling.** Killed. The shot-noise corrections
are partition identities: the Poisson subtraction (metrics.rs:236) and the finite-population
trait correction (:256-261) both require cells that tile space exactly once; under overlapping
windows the neglected covariance term jumps from O(1/C) to O(1) and the `.max(0.0)` floors clamp
real structure to zero. `axial_correlation` has no analogue over an unordered point set. And the
bias lands on absolute gate floors (viability.rs:65-74), genome-dependently, so it cancels
nowhere. The diagnosis (a lattice side is not a portable scale) was right; the cure is the
occupancy-derived side under goal 1, which two critics and the partition math independently
converge on.

---

## Goal 5: File-structure cleanliness

### Target layout, committed

```
axiom/
  Cargo.toml  Cargo.lock  README.md  .gitignore
  .cargo/config.toml
  .github/workflows/ci.yml
  .claude/settings.local.json          # tool state only; keep the name
  docs/STYLE.md                        # the only doc besides README
  src/
    lib.rs  main.rs  util.rs  render_recipe.rs
    bin/train.rs
    engine/  {substrate, trait, kernel, rng, grid, genome, geometry, interaction, lenia, world}.rs
    tuner/   {archive, campaign, ledger, codec, learning, metrics, novelty, topology, rollout, search, tuning, viability}.rs
    viewer/  {mod, material, runs, particles, trait_editor, camera, theme, controls}.rs
  examples/  {bench, inspect, snapshot}.rs
```

Changes from today, each with cost:

- **`src/train.rs` → `src/bin/train.rs`.** Delete the `[[bin]]` stanza (Cargo.toml:29-31);
  autobins finds it. First-layer files go 5 → 4. Cost: zero.
- **Keep `main.rs`** (7 lines, needs explicit `[[bin]]` for `required-features` regardless, and
  `src/main.rs` is the strongest name convention in Rust).
- **Keep `util.rs` and `render_recipe.rs` at the first layer.** `render_recipe` genuinely sits
  between the halves (lib.rs:10-12): the tuner persists it into headless checkpoints
  (checkpoint.rs:16,35), so it cannot live in `viewer/`, and it is not measurement, so `tuner/`
  would be a lie. Two pillars is not drift.
- **`tuner/persistence.rs` → `tuner/topology.rs`.** The file is H0 persistent homology, not disk
  persistence, and the name collides three ways: the Persistence *tier*, `CampaignPersistence`,
  and `PersistenceEnsemble`. Cost: `git mv` plus four import sites (metrics.rs:9, novelty.rs:6,
  rollout.rs:10, archive re-exports).
- **Split `campaign.rs` (1,746 lines) into three.** Orchestration stays (~350 lines after the
  deletions in this report); the ledger and feature engineering (~270) become `tuner/ledger.rs`;
  the binary codec merges with checkpoint's into `tuner/codec.rs` or `util.rs` (goal 4, item 2).
  Halves the largest file in the repo.

### `util.rs` policy, applied honestly

The bar: no natural owner, and serves more than one subsystem. Most candidates fail it.

- **Belongs, already there**: `Fnv` (util.rs:9-59, currently uncommitted). It replaced seven
  open-coded copies of the same fold across engine, tuner, and viewer, all touching durable
  formats. Textbook case; commit it. One wart: `Fnv::resume` is `#[cfg(feature = "viewer")]`
  (util.rs:20-23), so the file's shape changes with the feature set; `#[allow(dead_code)]` is
  the smaller tool.
- **Belongs once unified**: the binary reader/writer and single `write_atomic` (goal 4, item 2).
- **Does not belong**: the archive-select logic duplicated between `examples/inspect.rs` and
  `examples/snapshot.rs`; its owner is `tuner::archive` (and moving it there fixes a real bug,
  below). `learning::mix` (one caller, learning.rs:629-635). `material::normalise` and
  `genome`'s clamp (single owners). Moving any of these is the junk-drawer failure mode.
- **Fix, not move**: the three `finite`s (goal 3); rename two.

### `examples/` earns its place

`bench`, `inspect`, and `snapshot` do three different jobs, none duplicates `train`, and all
compile under the right feature sets (Cargo.toml:33-36). Making them subcommands would need
hand-rolled dispatch or a CLI dependency in a two-dependency crate, and would not reduce
first-layer `src/` files, which was the actual instruction. Keep them. Two real defects inside:

- **`inspect` and `snapshot` disagree on what `entry=0` means**: inspect sorts by
  `current_novelty` first (inspect.rs:23-24); snapshot indexes raw stable-key order
  (snapshot.rs:64-67). The same index names different genomes. Fix with one selection function
  on `tuner::archive`, called by both.
- **`bench.rs` is documented nowhere**, including the README, and its header overclaims (it
  measures the physics step only, bench.rs:6).

### `.claude/`, and the `.agents/` question

**Keep `.claude/`, do not rename, put agent-facing prose in `docs/`.** Verified across all six
sibling repositories this pass: the only tracked references to `.claude/` anywhere are two
`.gitignore` lines (pilcrow, spectra), so nothing the owner wrote depends on the path, and the
entire dependency is Claude Code's own settings/skills/worktrees discovery, which a rename
breaks for zero gain. `spectra/.agents/` is an untracked byte-identical mirror of
`.claude/skills`, not a convention; the owner's real two-tool pattern is additive
(`jm-legacy-modeling` keeps byte-identical `CLAUDE.md` and `AGENTS.md`). Full argument at the
top of [docs/STYLE.md](docs/STYLE.md). Also: `settings.local.json` carries five dead one-off
permission grants (a `compare.py` invocation, an `exit 1`, an eframe source grep) worth
clearing.

### The worktree, already resolved

`.claude/worktrees/peaceful-montalcini-5222e3` no longer exists: `git worktree list` shows only
the main checkout, `.git/worktrees/` is absent, and `.claude/worktrees/` is an empty directory
(delete it or ignore it). Nothing was lost; the branch tip matched `main~1`. One redundant ref
remains:

```bash
git branch -d rewrite/minimal-lenia
```

### Documentation: delete three of five surfaces

**Keep `README.md` (rewritten) and `docs/STYLE.md`. Delete `TECHNICAL_SPEC.md`,
`MARKET_ANALYSIS.md`, `MASTERDOC.html` (~103 KB).** The spec restates the README at higher
resolution; its genuinely load-bearing content (format layouts, version constants) belongs in
doc comments on `checkpoint.rs`, `archive.rs`, and `tuning.rs` where an editor will see it. The
market analysis belongs in Notion, which already holds the same verdict. The masterdoc is a
synthesis of the other two for an audience that does not exist. The decisive argument is
maintenance: all three now need the same N-D and learned-parameters rewrite simultaneously, for
a codebase about to change shape again.

Stale design assertions that are now wrong, not merely dusty (report requested per-place):

| Location | Assertion | Why it is now wrong |
|---|---|---|
| README.md:3 | "search system for three-dimensional…" | 3-D baked into the identity sentence |
| README.md:5 | links to `TECHNICAL_SPEC.md` etc. | **All three links are broken**: targets live in `docs/` |
| README.md:30 | "The product boundary fixes three spatial dimensions" | The same sentence lists particle count, radius, timestep as settings; dimensions belongs in that list |
| README.md:50 | "all computed from true three-dimensional torus geometry" | Descriptor width is already d-invariant |
| TECHNICAL_SPEC.md:9,14 | "These constraints are structural" (3-D row) | Ranks dimensionality with periodic boundaries |
| TECHNICAL_SPEC.md:117 | world table lists "dimensionality" | There is no such knob; the table describes a field that cannot be set |
| TECHNICAL_SPEC.md:124 | "adding one reaches all three surfaces automatically" | Celebrates how cheap another hand-set knob is |
| TECHNICAL_SPEC.md:128-130 | seeds and passes "belong to the evaluation tier, not the run" | Argues for more permanently fixed values, against goal 2 |
| MARKET_ANALYSIS.md:35 | "AXIOM commits to 3-D particles and a product boundary" | An abandoned decision framed as differentiator |
| MASTERDOC.html:314 | "43" as a hero statistic ("named knobs behind one entry point") | "Many hand-set things" presented as the headline |
| MASTERDOC.html:392 | "three-dimensional, because the product is something you fly through" | Conflates simulation and display dimensionality; a viewer can project |
| train.rs:14-15 | "AXIOM fixes the product boundary to three dimensions." | **Ships in `--help` today** |
| train.rs:89-90 | prints "3-D, …" | Prints on every run |
| metrics.rs:1 | "Fixed, three-dimensional behaviour measurements" | Argues with its own :12-13 ("for now") |
| kernel.rs:1 | "(3D torus)" | Above functions that loop `pos1.len()` generically |

The viewer's 3-D comments (viewer/mod.rs:1, particles.rs:20-21, controls.rs:1) are different in
kind and fine: a screen really is a projection.

Also: there is no `CLAUDE.md` in this repo; the audit brief assumed one.

### CI

`.github/workflows/ci.yml` runs `cargo fmt --all -- --check` first (ci.yml:19). **CI has been
red on `main` since `2a00622`**: run 29981795119 failed in 16 seconds at that step (the three
runs before it passed in 45-66s), so nothing since the rewrite has been CI-tested at all. The
step enforces a formatter that rewrites the four files defining the repo's style (52 sites, 920
lines, 17 of the sites in the canonical four; measured this pass). Delete the step; stable
rustfmt has no configuration that preserves the style (single-line bodies and multi-statement
lines are nightly-only options). Second gap: CI builds `--no-default-features` only
(ci.yml:20-21), so the viewer, `main.rs`, and `examples/snapshot.rs` (~2,800 lines) are never
compiled in CI. Measured this pass: both feature sets currently pass tests (91) and clippy
clean, so the blind spot hides nothing today; adding the default-features job (already the
owner's own action item) keeps it that way.

---

## Straight elimination

Roughly 700-900 lines of Rust come out with no behaviour change, before the codec unification
(~150-200 more) and ~103 KB of docs. Ordered by value-to-risk; two entries are decisions, marked.

| What | Where | ~Lines | Cost to remove |
|---|---|---|---|
| Checkpoint read path (`restore_world`, `load_world`, `load_manifest`) | checkpoint.rs:627,525,379 | ~111 | **Decision**, see below |
| Dead learning diagnostics (`average_precision`, `calibration_error`, precision half of `recall_at_budget`, three authority fields) | learning.rs:438-482,417-420,196-198 | ~50 | None; nothing reads them, `train.rs` prints nothing |
| `material::shade`, drifted `cfg(test)` duplicate | material.rs:174-214 | ~41 | Repoint one test |
| `zz_probe_determinism_check`, `zz_probe_final`: assertion-free `eprintln!` probes | material.rs:648-726 | ~78 | None; they assert nothing and are most of the 70s default-feature test wall |
| `SourceTier` and its mapping functions | archive.rs:26-31, campaign.rs:1174-1189,1336-1350 | ~50 | One line at search.rs:456; on-disk bytes unchanged |
| `WindowRecord.spatial`/`.heterogeneity`/`.connectivity`/`.tick`: serialized, parsed, never consumed (`feature_values` reads only structure/mobility/turnover/len, campaign.rs:1324-1330) | rollout.rs:151-159, campaign.rs:851-867,1111-1126 | ~57 floats/window | `STATE_VERSION` 5 |
| `EvaluationTier::ALL`, `FeatureSchema::names`, `PersistenceEnsemble::temperature`, `GroupedSplit::partition` | rollout.rs:50, learning.rs:30-32,236-238,115-127 | ~20 | Zero callers (grep-verified; `partition` test-only) |
| `curated_parent_indices`, `ParentSource::Curated`, the pin branch | tuning.rs:163-165, search.rs:294-315, archive codes | ~40 | Archive format tag; never assigned anywhere |
| `current_novelty` column: parsed then unconditionally overwritten on load | archive.rs:321,401 | 1 col/entry | Keep writing it (inspect prints it) or drop at v10 |
| `Archive::tuning()`, `Archive::is_empty()` | archive.rs:133-135,149-151 | ~6 | Zero callers; mind clippy's `len_without_is_empty` |
| `PromotionRecord.source_generation` | search.rs:34,101 | 1 field | Written, never read |
| `search::run()` | search.rs:179-187 | ~9 | Test-only; repoint the test at `run_with_promotions` |
| `Archive::from_text`/`header` shared 11 lines | archive.rs:293-306,406-419 | ~11 | Extract `read_header` |
| Redundant generation-start novelty refresh | search.rs:228 | 1 line | None. Critic-verified fully dead, generation 0 included: `merge` ends with a refresh on both branches (archive.rs:205-207,241), bootstrap's absorb covers the loop entry, and `Archive` has no interior mutability for the `&Archive` callback to exploit. One full O(n²) pass per generation for free |
| `read_bumps` (byte-identical to `read_shells`) | trait_editor.rs:240-248 | ~9 | Two call sites |
| `Camera::look_speed`: written only by `Default`, no slider (sibling `move_speed` has one) | camera.rs:60,72, controls.rs:95-96 | 1 field | Inline `0.0032` or add the slider |
| `EvaluationBudget::discovery()/persistence()/certification()` | rollout.rs:101-115 | ~15 | **Keep**: nine test callers, cheap ergonomics |
| `World::state_hash` | world.rs:143-155 | ~13 | **Keep**: it is how determinism gets proved |
| Duplicate `spatial_field` builds (2 of 5 per sample) | rollout.rs:350-352 | hot path | Pass fields in |
| `material::torus_distance` | material.rs:541-549 | ~9 | Call `kernel::displacement` |
| Duplicate `dot`, divergent `normalise` | material.rs:582-589, camera.rs:14-31 | ~12 | Unify; pick one zero contract |
| Write-side ceremony guards | checkpoint.rs:809-833 | ~4 clauses | Reject zero particles in `validate` first |
| `save_state`/`state_checksum` duplication | checkpoint.rs:442-459,670-687 | ~14 | Extract `encode_state` |
| Hand-derived genome-length polynomial | checkpoint.rs:227-254 | ~25 | Checked variant beside genome.rs:126 |
| Six enum↔u8 pairs | campaign.rs:1153-1267 | ~115 → ~30 | Macro, as tuning.rs:320-352 |
| Dead one-off permission grants | .claude/settings.local.json | ~5 | None |
| Empty `.claude/worktrees/` directory, `rewrite/minimal-lenia` branch | repo | 0 | None |
| `TECHNICAL_SPEC.md`, `MARKET_ANALYSIS.md`, `MASTERDOC.html` | docs/ | ~103 KB | Fold format details into doc comments |

**The checkpoint read path is a product decision, not a code-quality call.** `save_world` writes
files on every certified run; nothing reads them back; the README documents the gap as future
work (README.md:161). Either build the caller (a `train` resume flag or a viewer "open
checkpoint" action) or delete the write side too. **Default: keep it and build the caller.** A
certified 100k-step world that cannot be reopened is not much of a deliverable, and the
10,000-step exact-continuation test (checkpoint.rs:1169-1222) is the single strongest test in
the repo, proving the hard part already works.

### Tests

91 pass (67 headless), clippy clean on both feature sets, measured this pass. Quality is better
than the average of this codebase and worth saying: checkpoint's tests corrupt exact bytes and
assert named variants (checkpoint.rs:1096-1139); topology's include a real regression guard for
a normalization that once returned constants (persistence.rs:388-422); learning verifies the
authority fail-safe engages on label-free noise (learning.rs:719-732). Three real gaps:

1. **`engine/` has zero test modules** across all ten files, while campaign has 13 tests and
   material 11. The least-tested code is what everything depends on, and the project's own notes
   record exactly the failure this invites: a `displacement` rewrite changed float accumulation
   order and "silently shifted every search result." One pinned numeric test on `displacement`
   and `bump_and_slope` catches that class. Do not add `mod tests` to the four canonical files
   (their style excludes it); put it in `tests/` or `lenia.rs`.
2. **`campaign::run`'s success path is untested**; the only call in tests asserts a rejection
   (campaign.rs:1657-1674).
3. **Nothing demonstrates the learned scheduler ever changed a real outcome**; all its evidence
   is synthetic. Given finding 5, this absence is the point.

The earlier draft proposed retiring two tests; this pass disagrees with both. The rollout
determinism test also pins descriptor length (fine, it is primarily a determinism test), and the
continuation-selection ordering test pins the deterministic fallback order, which is the
contract, not an accident. Keep both. The borderline case worth knowing about:
`every_knob_round_trips_through_its_own_text` (tuning.rs:437-446) proves self-consistency, not
correctness; a formatter that printed wrong-but-stable values would pass. Keep it, know its
limit. Within the engine gap, the sharpest hole is `Layout::for_genome`: nothing tests that it
returns `None` for a length fitting no anchor count, and that is the exact boundary structural
search will lean on.

---

## What is already right

- The engine is genuinely dimension-generic, seven files, no constant (goal 1).
- Measured interaction norms beat the reference implementation's hardcoded two-dimension table
  (interaction.rs:106-166).
- Extent is derived, never free (geometry.rs:1-3); the lattice self-normalizes across genomes
  (metrics.rs:195-197).
- The H0 barcode avoids choosing a cluster radius and documents why honestly
  (persistence.rs:1-22); its brute-force cross-check test (persistence.rs:252-278) is exactly
  how a spatial-index optimization should be proven.
- Winding connectivity computes the correct topological invariant in one pass (metrics.rs:342-388).
- Binary gates with named rejections, never a scalar fitness (viability.rs); margins make
  failures learnable without becoming scores (viability.rs:50-87).
- The `KNOBS` declare-once table (tuning.rs:300-404) is the right mechanism; goal 2 argues its
  domain should shrink, not that it is wrong.
- The digest's meaning-versus-effort split (tuning.rs:229-232) is the correct distinction,
  modulo the tier-steps hole.
- `checkpoint.rs`'s error enum has no unconstructed variants and its corruption tests are precise.
- The uncommitted `Fnv` consolidation is correctly scoped and pinned by tests against the exact
  fold it replaced (util.rs:71-108). Commit it.
- `Grid`'s design (silent brute-force fallback, callers never branch, grid.rs:56-64) and its CSR
  build are clean.
- `.cargo/config.toml` routing build output to `.cache/` does what it says in three lines.
- `runs.rs`'s refusal to offer novelty as a ranking key, with its reasoning in the module doc
  (runs.rs:11-15), is a genuinely thoughtful design decision, and tested (runs.rs:404-408).

---

## Sequenced plan

Three categories, as requested: **delete** needs no design, **redesign** needs a decision,
**research** needs evidence before code.

### Phase 0: unblock and guard (hours, no design)

1. Delete `cargo fmt --all -- --check` from ci.yml:19; add the default-features job (the
   viewer, `main.rs`, and `snapshot.rs` have never compiled in CI; both feature sets pass
   locally today, keep it that way).
2. Commit the `Fnv` consolidation already sitting in the working tree. It is finished, correct,
   and pinned by tests.
3. Land the two structural-hygiene guards: length assert in `iso_line_dd_scaled`, genome-length
   guard in `Archive::merge`. A few lines; they protect everything phase 4 touches.
4. Fix checkpoint.rs:112's error text ("must be three" for a zero check) and resolve the
   emission/absorption comment at material.rs:289-292 one way or the other (read
   `recipe.absorption`, or say "matches the default" instead of "move together").
5. `git branch -d rewrite/minimal-lenia`; clear the dead grants in `.claude/settings.local.json`;
   remove the empty `.claude/worktrees/`.

*Unblocks: every later phase gets real CI signal.*

### Phase 1: pure deletion (a day, parallelizable with everything below)

Everything in the elimination table except the checkpoint read path. Narrow the `pub` surface
first (campaign 19 → ~5, checkpoint 14 → 4), then let `-W dead_code` confirm the rest
mechanically; the lib target hides all of it from the compiler today. Includes the redundant
search.rs:228 refresh, the `zz_` probe tests, and the three stale doc surfaces plus the README
rewrite (identity sentence, broken links, the false "refused rather than silently mixing" claim
at README.md:42, document `bench`). Add the missing engine tests while in there: pinned numeric
values for `displacement` and `bump_and_slope` in `tests/` (not in the four canonical files),
and a `Layout::for_genome` returns-`None` case.

### Phase 2: unification (days; item 3 is the big one)

1. `const BLOCKS` descriptor table (goal 4 item 3). **Prerequisite for phase 3.**
2. Delete `SourceTier`; macro the enum↔u8 maps; name the heterogeneity five; checked
   genome-length variant; extract `encode_state`.
3. Merge the two binary codecs into `util` keeping both durability properties (128-name retry
   AND parent fsync AND non-finite rejection), then split `campaign.rs` into orchestration,
   `ledger.rs`, and the shared codec. Halves the biggest file.
4. `tuner/persistence.rs` → `tuner/topology.rs`; `src/train.rs` → `src/bin/train.rs`.
5. Certification path: retain the certification-tier final substrate (gated, never serialized),
   route `certified_preset` through `resolve_clamped`, store a `for_world` recipe; derive the
   viewer recipe on adopt.

### Phase 3: N-dimensional (one to two weeks, after phase 2 item 1)

0. Close the unguarded invariant first, even if nothing else ships: assert
   `substrate.dims() == DIMENSIONS` at the `metrics.rs` entry points. One line.
1. Mechanical sites: `substrate.at()` in `raw_rdf`/`mobility`, `sqrt(d)` in `bin_range`.
2. `SpatialField` to mixed-radix indexing (the idiom `Grid` already demonstrates); separable
   blur (`3d` taps, a win in 3-D too).
3. `side` from the three occupancy targets; eighth H0 axis to a scale-free component fraction.
   Descriptor v6; every archive breaks, none matter.
4. `dimensions` becomes a knob and an `ExperimentIdentity` field (the header follows from the
   knob for free); delete `const DIMENSIONS`; fix render_recipe.rs:99's `.cbrt()`.
5. Rewrite train.rs:14-15 and :89-90 and the surviving doc text.
6. Viewer last, as an N-to-3 slice/project adapter, and only after phase 5 item 2 says higher
   `d` is worth looking at.

### Phase 4: learned parameters (after phase 3)

1. The tier-steps fix first (per-tier probe/window fields under the digest; sticky
   `Option` semantics in the identity; budget-aware `already_persisted`). Until then, labels
   gathered across resumed budgets are quietly heterogeneous.
2. The shape outer loop: sweep `anchors`/`shells`/`bumps` campaigns (zero new code), add the
   cross-shape seeding operator (~30 lines, decode-level). If the sweep shows shape matters,
   graduate the outer loop to a bandit over shapes.
3. Lane shares under a bandit with change detection: the best-evidenced adaptive change
   available, on an already-isolated decision point.
4. Leave `particles`, `radius`, `rate` as settings. The phase-3 occupancy fix removes the
   `void` axis's count-dependence, which is one of two prerequisites for ever revisiting
   `particles`; cost-aware scheduling, which does not exist, is the other.

### Phase 5: research before trusting (mostly the owner's own action items)

1. Build the novelty-stratified 100k-step calibration set; measure promotion precision, recall,
   calibration. No validated short-horizon survival surrogate exists in this literature, and
   until this exists the tier ladder's promotion rule is an assumption.
2. Probe whether `d = 4` produces anything worth certifying before investing in the viewer
   adapter. Chan's higher-dimensional result (solitons "predominantly stationary" at `d >= 3`)
   is the standing warning.
3. Decide `learning.rs`'s fate on item 1's evidence: either wire its already-computed
   diagnostics into adaptive quotas, or delete down to the uniform fallback (~600 lines). Its
   current state, fully built and unreachable, is the worst of both.
4. Replace the O(N²) rollout RDF with the deterministic stratified pair estimator when particle
   counts push past ~25k (measured crossover); keep the exact path as a validation mode.

### Deferred decisions for the owner

1. **The checkpoint read path**: build the caller or delete the writer. Default: build the
   caller.
2. **The `default_genome` hand prior** seeding every search: keep as a labelled baseline or go
   fully random. Default: keep, labelled.
3. **The two trait palettes** (theme.rs:34-43 vs material.rs:551-569): converge on one. Default:
   the material palette, since the material view is the product.
4. **`dimensions` as a gene**: no. One campaign per `d`; compare archives. (Setting: yes, now.)
