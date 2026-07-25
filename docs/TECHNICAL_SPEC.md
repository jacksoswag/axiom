# AXIOM technical specification

AXIOM is a Rust crate that searches Particle-Lenia interaction laws. A law is a bounded flat `Vec<f32>`. The engine decodes one into a simulation and steps it, `harness::rollout` reduces a stepped simulation to a descriptor, the tuner gates, scores, and breeds those descriptors, and the harness puts the whole of it behind a line protocol on stdin and stdout.

This document describes the code in this repository. Where the two disagree, the code is right and this document is stale.

## 1. Design invariants

These are enforced by the code, not by convention.

| Invariant | Where it lives | Consequence |
|---|---|---|
| The engine is downstream of nothing | `engine` imports only `engine` and `util` | The simulation cannot acquire a dependency on how it is searched or drawn. |
| The tuner touches no simulation | `tuner` builds no `Sim`; `harness::rollout` is the only place one is constructed to be measured | Every criterion reads behavior the same way. |
| One trust boundary | `harness::protocol` is the only place an outside caller's numbers are checked | Past it, an out-of-range value panics rather than being re-validated at three layers. |
| Dimensionality is data | `FixedGenome::dimensions`, read by substrate, step, distance, probe | No constant fixes how many axes a particle has. |
| The measurement grid is three-axis | `field::GRID_AXES`, checked in `Search::new` | A search rejects a genome shaped any other way, at the one door into a search. |
| Periodic bounds | minimum-image distance, `rem_euclid` on every wrap | Structures cross every face and there is no terminal boundary. |
| Box size is derived | `Probe::box_len` from the coordination gene | Density is controlled; extent is never a free scale. |
| Norms are measured, never assumed | `Matrix::norm_densities` on a fixed uniform reference | Normalized potential means the same thing across particle counts and traits. |
| Closed-form derivatives | `strength_and_slope` returns value and slope together | No autodiff and no tensor dependency. |
| One shared descriptor axis | every metric clamps into `[0, 1]` | No raw statistic with a wide range dominates a distance. |
| The plan is declared, not assumed | `plan(criterion.metrics() ++ gate metrics)` | Nothing outside a criterion and the gates can move what gets measured. |
| Determinism | seeded xorshift, index tie-breaks, order-preserving parallel filter | A parallel batch judges identically to a serial one. |

## 2. Simulation state

A particle carries a position and one fixed trait:

```text
x_i in [0, box_len)^dims    position, wrapped
c_i in [0, 1]               trait
```

Position is the only dynamical state. Traits are assigned at seeding and never move. Spatial self-organization can sort the trait continuum into regions; trait motion, birth, death, and inheritance are not in this crate.

`Sim` is the authoritative state and is reconstructible from `FixedGenome`, `Genome`, and the resolved box. `Substrate` holds positions, traits, box length, softening, and dimensionality. The spatial hash grid inside it is a cache derived from those positions, and it reflects whatever they were at the last rebuild rather than authoritative state.

### 2.1 Spatial index

The grid holds one cell per interaction cutoff, stored compressed: a particle list sorted by cell, plus cell edges in list space. A neighbor walk visits the home cell and all `3^dims` surrounding cells; callers still distance-check, since the walk yields candidates rather than neighbors.

The grid deactivates below three cells per axis, where the stencil wraps onto itself and would visit a particle twice, and above `2^24` cells, where allocation stops paying. Both fall back to an all-pairs walk yielding the same candidate set. A test asserts the two walks agree on a seeded fixture.

Cell count is also capped at roughly four cells per particle. A legal small-reach genome would otherwise ask for a million cells to hold three hundred particles, where clearing and scanning the map costs more per tick than every pair it prunes. The cap stays correct because a cell wider than the cutoff is still covered by the same three-wide stencil; it only prunes a little less. That cap is what binds in practice, and the `2^24` ceiling sits above it so that a swarm approaching a million particles still indexes rather than falling back to a hundred billion pairs a step.

### 2.2 Which machine runs the step

`lenia::step` rebuilds the index, then routes. Above 16,384 particles, in three axes, with a live grid and an adapter that answered, the tick goes to the graphics card; otherwise `lenia::walk` does it on the cores. `AXIOM_GPU=0` forces the cores, and any other number sets the floor directly.

The card runs the same walk over the cell map the CPU just built. The grid rebuild deliberately stays on the cores: a prefix sum over a few million cells is a poor trade against the millisecond and a half it costs there, and the map is small enough to ride along. Threads take particles in grid order rather than index order, so neighbouring lanes stand in the same cell and read the same positions.

The two backends cannot agree bit for bit, because `exp()` on a card is its own approximation rather than the one libm ships. **Replay is exact within a backend and only close across them.** The performance report states both every time it runs: one tick apart by under `1e-6` of a box, and twenty ticks on the card repeating exactly.

## 3. Interaction law

### 3.1 Trait basis

`M` anchors sit evenly spaced on a circle. Piecewise-linear hats interpolate between them:

```text
alpha_a(c) >= 0
sum_a alpha_a(c) = 1
```

At most two adjacent anchors have nonzero membership, and at an anchor the membership is exactly one-hot. Placing anchors on a circle wraps the trait axis at the `0/1` seam instead of clamping it. With one-hot traits the whole model reduces exactly to a discrete pair-specific one, which is what makes the continuous form a generalization rather than a replacement.

Seeding matches the population's anchor distribution to the genome's logits exactly. Logits go through an exponential with a small floor, each bin's exact share is floored to an integer, and leftover particles go to the bins with the largest fractional remainder, ties by index. Each bin's quota is then filled with traits drawn uniformly from that bin's slice of `[0, 1)`.

### 3.2 Pair-indexed matrix

Every ordered anchor pair `(a, b)` owns a kernel, a growth curve, and a directed weight:

```text
U_ab(i)    = sum_(j != i) alpha_a(c_j) K_ab(d_ij)
Uhat_ab(i) = U_ab(i) / norm_ab

delta x_i = 2 dt sum_a sum_b alpha_b(c_i) w_ab B'_ab(Uhat_ab(i))
            / norm_ab grad U_ab(i)

grad U_ab(i) = sum_(j != i) alpha_a(c_j) K'_ab(d_ij) delta_ij / d_ij
```

The factor of two is the surviving term of `G = 2B - 1` under differentiation: only the slope enters movement, and the `-1` dies there.

A pair's position in the matrix is its `(source, destination)` identity, so neither the interaction nor its callers carry those indices separately. Interactions are stored flat, source-major.

The step accumulates sensed potential and its gradient only for anchor pairs where both memberships are nonzero, which is at most four pairs per neighbor. It then converts potential into motion over every possible source against the receiver's own active anchors, because a source anchor with no local mass still contributes a defined response.

`delta_ij` is minimum-image displacement. Softening is added last, after the per-axis squares, matching the accumulation order callers depend on.

### 3.3 Shells and bumps

Both mixtures are Gaussian. Shells act over distance, bumps over sensed density:

```text
K(d) = sum_m amp_m exp(-(d - peak_m)^2 / (2 width_m^2))
G(u) = 2 sum_m amp_m exp(-(u - peak_m)^2 / (2 width_m^2)) - 1
```

One function computes value and slope in the same loop, so the gradient reuses each term. A zero amplitude disables a component, which lets a search learn the active count instead of being told it.

The neighbor loop truncates each shell at `peak + 3.5 * width`, roughly 0.2 percent of its peak, and skips terms whose amplitude is below `1e-6` so one inert term cannot inflate the reach and erase the saving. This truncation is an explicit numerical approximation: the executed force carries a small hard cutoff and is not the exact gradient of the untruncated energy at that boundary.

### 3.4 Norm calibration

`norm_ab` is the mean potential a receiver anchor senses from a source anchor, per unit receiver mass, measured on a deterministic reference population: a fixed calibration seed, uniform positions, and uniform anchor logits, using the same kernel, distance, softening, and memberships as the rollout. Calibration runs when a `Sim` is constructed, so every pair sees reference mass. A measured value at or below `1e-6` falls back to `1.0`. A clumped live state never becomes the reference.

Seeding the reference with uniform logits rather than zeroed traits is load-bearing: with all traits at zero, every pair except `(0, 0)` sees no mass and silently takes the fallback. A test asserts that off-diagonal pairs get measured norms.

### 3.5 Derived box size

`Probe` measures the mean nominal potential a particle senses in a uniform population at a trial box of six radii, using a shell shaped by radius alone rather than any one genome's law, so every candidate in a search measures against the same reference. The measurement caps at 1,500 particles and rescales by the true density ratio, bounding its `O(n^2)` cost.

A genome's coordination gene then rescales that reference:

```text
box_len(coordination) = trial_box_len * (measured / coordination)^(1 / dims)
```

The probe depends only on particle count, dimensionality, and radius, none of which vary per run, so a search builds one and threads it through every candidate.

## 4. Genome

```text
[ coordination ]
[ anchor_count trait logits ]
[ anchor_count^2 pair blocks in source-major order ]

pair block =
  shells x [amp, peak, width]
  bumps  x [amp, peak, width]
  [directed weight]

pair_stride = 3 * (shells + bumps) + 1
gene_len    = 1 + anchor_count + anchor_count^2 * pair_stride
```

`FixedGenome` is the half a search holds still: particle count, dimensions, anchor count, shell count, bump count, radius, timestep, seed. `Genome` is the decoded half: coordination, the trait distribution, and the flat pair-block genes.

| gene | range |
|---|---|
| coordination | `3.0` to `20.0` neighbors |
| trait logit | `-4.0` to `4.0` |
| shell amp | `0.0` to `1.0` |
| shell peak | `0.0` to `min(2 * radius, reach / 2)` |
| shell width | `min(0.05 * radius, peak_max / 4)` to `min(radius, reach / 7)` |
| bump amp | `0.0` to `1.0` |
| bump peak | `0.0` to `3.0`, over sensed density |
| bump width | `0.05` to `1.5` |
| directed weight | `-100.0` to `100.0` |

`reach` is `0.375` of the box, the point past which a shell would wrap into itself.

Pair-block bounds depend on the box, which depends on coordination. A search mutates under bounds taken at maximum coordination, where the box and therefore the reach are widest, so every genome it can produce stays legal. A rollout re-clamps the pair genes at the box its own coordination gene resolves to, where reach is tighter. Clamping replaces any non-finite gene with its lower bound, so a single NaN cannot poison a decode.

## 5. Measurement

### 5.1 Spec and plan

A metric is a `&'static Spec` written in its own file. The spec declares:

| field | meaning |
|---|---|
| `key` | stable name, and what two handles compare on |
| `width` | slots it owns in a descriptor |
| `sides` | grid resolutions it reads |
| `depends` | metrics measured ahead of it, so its slots are already filled |
| `reduce` | how its slots collapse across samples |
| `pairs`, `graph`, `motion` | which shared blocks it needs |
| `measure` | the function, taking the blocks and the descriptor so far |

`ALL` lists all ten metrics in dependency order. `plan` takes what the criterion and the gates want, closes the set over dependencies in one backward pass, and filters `ALL`, which yields dependencies ahead of dependents without a sort.

A descriptor is a bare `Vec<f32>` and the plan is its layout, so reading one slot needs both. `slice` and `scalar` do that walk. A metric the plan skipped, and a descriptor from a simulation that died, both read empty.

### 5.2 Shared blocks

Blocks are built once per sampled tick from the union of what the plan requests, so no metric constructs its own:

| block | requested by | contents |
|---|---|---|
| pair histogram | `pairs` | unordered pair counts per trait band and log radial bin, over an evenly strided sample of at most 4,096 particles, accumulated in `f64` |
| spatial field | each entry in `sides` | periodic density, trait sum, trait square sum, and four-bin trait counts per cell |
| barcode | `graph` | H0 death-scale counts from the cutoff-limited spanning forest |
| motion | `motion` | the previous sample's positions, plus its side-16 grid |

Grid sides are deduplicated, so two metrics wanting one resolution cost one grid. Asking for a block the plan never requested panics rather than inventing a reading. The barcode's grid rebuild is the only mutation of the substrate during measurement, and it is scratch either way because the next step re-buckets.

The `f64` accumulation in the pair histogram is not decoration: a thousand-particle simulation contributes half a million pairs, and `f32` starts dropping counts.

### 5.3 Reduction

| rule | for | behavior |
|---|---|---|
| `Mean` | ordinary observations | averaged across every sampled tick |
| `Last` | anything that already read the history | the final tick's value |
| `Once` | an experiment too costly to repeat | measured on the final sample only, zero-filled before that |

`evaluate` runs a plan against one tick in order, each metric reading its dependencies out of the descriptor being built. Every output is resized to the declared width, so a miscount cannot shift the slots behind it. `fold` then collapses the per-tick readings by each metric's rule. A rollout that never sampled folds to an empty descriptor rather than a plausible row of numbers.

### 5.4 The metrics

Fifty-four slots when all ten run. Every slot is clamped into `[0, 1]`.

| slots | key | reduce | reading |
|---:|---|---|---|
| 18 | `rdf` | Mean | three trait-distance bands by six log radial bins, as ratios against the uniform start, on a symmetric log axis with a ratio cap of 20 |
| 1 | `structure` | Mean | mean absolute departure of those ratios from 1 |
| 15 | `heterogeneity` | Mean | five readings each on grids of side 4, 8, and 16 |
| 4 | `connectivity` | Mean | largest winding-component fraction, dense then void, at 0.5 and 1.5 times mean density |
| 8 | `topology` | Mean | seven folded H0 death-scale masses plus one log-scaled mass for components still separate at the cutoff |
| 1 | `mobility` | Mean | mean minimum-image travel between samples, saturating at a tenth of the box |
| 1 | `turnover` | Mean | absolute grid-mass change between samples on the side-16 grid |
| 1 | `asymmetry` | Mean | departure of the directed weight matrix from its own transpose, normalized by the weights |
| 4 | `temporal` | Last | variance of the spatial picture across samples, plus its autocorrelation at lags 1, 2, and 4 |
| 1 | `robustness` | Once | mean share of injected damage closed against an undamaged control |

**`rdf`** walks every pair among its sample, deliberately: its far bins reach the box half-diagonal, past any kernel cutoff, so the neighbor grid cannot help. Every other block is linear in the swarm and this one is quadratic, so it stands on an evenly strided sample of at most 4,096 particles and costs the same at four thousand as at four hundred thousand. Below that ceiling the swarm is walked whole and the answer is exact. The stride is taken rather than drawn: it needs no rng, and it lands on the same particles in the measured cloud and in the uniform one it is divided by, so the two contribute the same pair count and their ratio needs no rescale. Index order is trait order, since seeding lays traits down in anchor blocks and a stride crosses all of them evenly. Bins run from one mean particle spacing to the box corner. A bin the baseline left near empty reports 1.0, meaning no information rather than a division by almost nothing.

**`heterogeneity`** reads, per scale: density overdispersion after subtracting Poisson sampling variance; void probability; occupancy-weighted variance of local mean trait after a finite-population shot-noise correction, expressed as a share of the trait variance there is to explain; occupancy-weighted local trait entropy with a small-sample correction; and axial density autocorrelation. Both corrections are load-bearing. Without the Poisson subtraction a uniform swarm reads as heterogeneous merely because the grid is fine; without the shot-noise subtraction, single-particle cells on fine grids read as distinct regions. The three scales stay separate rather than averaged, because sub-structure lives at one scale and averaging hides it.

**`connectivity`** blurs density with a separable three-tap kernel before any threshold, so one stray particle cannot register as a dense phase. It then flood-fills each phase carrying an integer lift: the count of steps taken along each axis to reach a cell without wrapping. Arriving at a visited cell with a disagreeing lift means the two walks differ by a full loop, which detects a box-spanning phase using nothing but integers. A phase contributes only if it fills at least 8 percent of the grid and one of its components carries such a loop, so a compact blob is disconnected here however many of its cells touch. Dense fractions come first, then void, which is the order `criterion::structure` splits them on.

**`topology`** exploits the fact that H0 death scales are exactly the edge weights of the Euclidean minimum spanning tree, so the whole cluster hierarchy costs one spanning forest. Edges stop at the kernel cutoff: clumps no kernel can reach across are not one structure at any scale. Ties break by index so the forest, and therefore the barcode, is reproducible. Union-find uses path halving and union by size.

**`temporal`** is the only metric reading other metrics. It pulls the `rdf`, `heterogeneity`, and `connectivity` slices out of each sampled descriptor, lays them end to end as that tick's picture, and asks how far the picture moved and how much it still resembles itself a few samples later. Those axes already sit in shared units, so nothing needs rescaling. A lag correlation near 1 means it looks the same later, so a frozen crystal and a slow drift both read high. With too few samples it defaults to assuming static, which fails a variance floor rather than sneaking past one.

**`robustness`** is the only metric running an experiment. It snapshots the substrate, steps an undamaged control forward 100 steps, then for each of three probes shakes one local neighborhood by half a kernel reach, steps the damaged copy the same 100 steps, and measures how far it got back toward where the control ended. Comparing against the future control rather than the present state means ordinary drift is not billed as damage. A probe whose shake moved the signature by less than `0.01` is discarded as proving nothing; a copy that blows up closes nothing and is scored as such.

The score is a mean share of damage closed, not a count of probes clearing a threshold. Counting gave a step function with four levels that sat on zero for most genomes, so anything multiplying by it lost its gradient entirely.

Its signature is `rdf`, `heterogeneity`, and `connectivity`, read through the same plan machinery the descriptor uses, so damage and recovery land in the space the search already thinks in. `robustness` is deliberately absent from that list, which is what keeps it from recursing into itself. Its four constants are fixed rather than run settings: changing any of them changes what a robustness reading means, so two runs that disagreed about them were never comparable.

## 6. Rollout

`harness::rollout::sim` is the only place a `Sim` is built to be measured. It decides when to look, never what at.

1. Decode the flat genome, resolve the box from its coordination gene, re-clamp the pair genes at that box.
2. Build the `Sim`. The uniform start's pair histogram becomes the structure baseline.
3. Step. A non-finite coordinate ends the rollout and returns empty.
4. Sample eight times at a fixed interval across the latter half of the run.
5. From the second sample on, check the gates as a partial reading. Two consecutive failures end the rollout.
6. Fold the history into one descriptor.

The first sample has nothing behind it, so anything reading change reads zero there and is not held against the genome. An abandoned rollout returns an empty descriptor, which reads zero everywhere and therefore clears no gate carrying a floor. Patience of two rather than one is deliberate: a simulation can dip out of a band and climb back within a sample.

## 7. Gates

```text
Gate { metric, floor, ceiling }
```

A valid range on one metric, read in that metric's own descriptor units. Gates are owned by the `Search` rather than declared as constants, because a caller builds them at runtime. A partial reading, taken mid-rollout, skips any gate whose metric reduces `Once`, since that metric only lands on the final sample. A dead simulation reads zero everywhere, so any gate with a floor rejects it.

Whether a genome is worth ranking at all is the gate's job, which is why no criterion reads a threshold.

## 8. Criteria

A criterion produces one number, names the metrics it reads so a rollout measures them, and ranks nothing else. Adding one means a module and a variant, and both matches stop compiling until it is wired in.

### 8.1 Novelty

Population-relative. The score is the mean distance to the fifteen nearest behaviors already found, over the whole descriptor rather than any one metric, so it needs no plan. An empty population scores zero: an opening generation survives on capacity and the gate, never on invented novelty. The neighbor count is fixed rather than a per-run setting, so two searches compare.

Its `Vec<Metric>` is the axes it wants to spread genomes across, not a dependency list. Novelty names no metric of its own; it wants a space to be far apart in, and the gates widen that space too, since distance reads the whole plan.

### 8.2 Structure

Absolute, so unlike novelty it can be climbed toward, and unlike novelty it can be fooled: a static lattice satisfies the contrast term outright.

| clause | source |
|---|---|
| contrast | best scale's `min(density overdispersion, trait variance)` from `heterogeneity` |
| material | weakest dense winding fraction from `connectivity` |
| void | weakest void winding fraction from `connectivity` |
| repair | `robustness` |
| change | `turnover` |

Clauses multiply, so weakness anywhere costs everywhere; a sum would let beautiful contrast and no motion whatsoever place well. Contrast is read at the scale that shows it most, since sub-structure lives at one scale.

Each clause is floored at `0.02` before the product. Multiplying raw, a simulation missing one clause scored exactly zero no matter how it did on the other four, and since a tournament breaks ties toward its first draw, a population of zeroes made selection random. Floored, the conjunction still costs an order of magnitude and the other four stay rankable.

`METRICS` sits beside `score` so the two move in one edit. A metric missing from a plan reads zero and silently zeroes the product, so drift there is a bug that never announces itself.

## 9. Search

`Search` is one complete run. Six inputs: fixed genome, algorithm, criterion, gates, rollout length, seed. Three derived: the probe, the per-gene bounds, and the metric plan.

`Search::new` panics if the genome's dimensionality differs from the measurement grid's three axes. That check sits at the one door into a search, because any other shape would index cells that do not line up with the genome's own coordinates.

`Search::run` executes the algorithm and returns the surviving population sorted best score first.

`Search::run_genome` is the per-candidate path: simulate, reject an empty or non-finite descriptor, apply the gates as a final reading, score against the population, and return a `Specimen` carrying genome, descriptor, and score. Cost is the simulation.

### 9.1 The generation loop

1. Re-score the whole population. A population-relative criterion leaves every stored score stale the moment the population moves around it. Each member is scored against a set containing it, so the bias is identical for everyone and can never change an ordering.
2. Propose a batch across three lanes.
3. Evaluate in parallel with an order-preserving filter, so a parallel batch judges identically to a serial one.
4. Retain.

Generation zero proposes against an empty population, which comes back as fresh draws, so the opening sample needs no special case.

### 9.2 Lanes

| lane | parent |
|---|---|
| tournament | binary tournament on score, ties falling to the first draw |
| expedition | whichever member sits nearest a uniform random point in descriptor space |
| fresh | a new uniform draw over the bounds |

The two named shares are fractions of the batch; fresh draws take whatever they do not claim, so the three always sum to the batch regardless of rounding and there is always a way out of a converged population. Zero both named lanes and every slot comes back a fresh draw, which is the random-search baseline a real setting has to beat: if random matches evolution under some criterion, that criterion is not steering.

An expedition throws a dart at a uniform point in descriptor space and mutates whichever member landed nearest it. In dozens of dimensions that is a random goal direction rather than coverage-aware illumination, and that is the intent: distant intentions without a model inventing state.

### 9.3 Mutation

```text
child = a
      + iso * gene_range * Normal(0, I)
      + line * Normal(0, 1) * (b - a)
```

Independent noise scaled to each gene's own range, then one shared draw along the difference vector toward a second parent, so a child inherits a direction the population already contains instead of dissolving into undirected noise. The second parent is itself drawn by tournament. Bounds clamp every child.

### 9.4 Retention

An evaluated generation folds into the population that survives it. Members sort best score first, and one is kept only if it sits clear of everything already kept. The cutoff is a quarter of the population's own median nearest-neighbor spacing: a fraction rather than an absolute distance, so it holds whatever units and however many axes a descriptor turns out to have. A population of identical genomes has zero spacing and drops nothing, which is what lets an opening generation through. The top scorer in a cluster survives and the rest of that cluster does not, so a converged batch cannot fill the population with copies of one genome.

## 10. Harness

The binary is one call into `harness::run`. Commands arrive one per line on stdin, events leave one JSON line per event on stdout, flushed, because a frontend is blocked waiting on the next one. A closed pipe ends the process: there is nothing left for it to do.

### 10.1 Session loop

Read, step, report, repeat. An idle session blocks on stdin outright, so a paused world costs nothing. A session with a search running waits on the frame interval instead, which gives that core back to the batch and still answers a stop within one frame. A closed stdin with the owed work done is a finished script and returns.

| constant | value | why |
|---|---|---|
| `FRAME_INTERVAL` | 40 ms | faster than a screen refreshes spends wire on positions nobody sees |
| `FORWARD_CHUNK` | 200 ticks | a hundred-thousand-tick jump still answers commands and streams frames on the way |
| `FRAME_POINTS` | 8,192 | past this a frame decimates by an integer stride |

Speed is a rate, not a batch size: a running world owes `speed` ticks per frame interval and sleeps out the remainder. A fast-forward is exempt, since a jump is asked for rather than watched.

Edits coalesce. A slider drag arrives as dozens of `gene` lines and one pair edit costs a fresh density calibration, so edits raise a flag and the rebuild happens once per turn in `settle`. Anything that reads the world, or throws away what has been read of it, settles first.

### 10.2 Commands

| verb | arguments | effect |
|---|---|---|
| `catalog` | | re-send catalog, layout, and state, then owe a frame |
| `shape` | any of `particles`, `anchors`, `shells`, `bumps`, `radius`, `dt`, `seed`, `dimensions` | any subset of the fixed genome, range-checked here |
| `gene` | `<index>=<value>` repeated | edit genes by position, clamped into what this box allows |
| `genome` | `genes=` comma list | replace the whole flat vector, unclamped |
| `run`, `pause` | | start and stop the live world |
| `step` | `count` | owe that many ticks, taken in chunks |
| `speed` | `steps`, 1 to 1024 | ticks owed per frame interval |
| `reseed` | `seed` | rebuild the world from the shape and genes as they stand |
| `measure` | `keys`, `every`, `now` | what to watch and how often, or one full reading on the spot |
| `gates` | `<metric>=<floor>:<ceiling>` repeated | replace the whole gate set |
| `frames` | `off` | stop or resume the position stream |
| `search` | `start`, `stop`, `from`, `criterion`, `axes`, `algorithm`, `timesteps`, `seed`, and the evolve knobs | start or stop the one search a session runs at a time |
| `quit` | | return on the spot, dropping owed work |

An unknown verb is an error rather than silence. A `gene` batch is checked whole before any of it lands, because a frontend that hears an error reads it as "nothing happened". Ranges are checked in `reshape` because past that door a count of zero anchors reaches a decode that indexes a genome and panics.

### 10.3 Events

| event | carries |
|---|---|
| `catalog` | every metric with its width, reduction, dependencies, grids, and cost flags; the criteria; the evolve knobs with their ranges; the shape fields |
| `layout` | every gene with its kind, label, range at this box, and current value |
| `state` | running, tick, box length, whether the grid is in use, speed, cadence, watch list, gene count, shape |
| `frame` | tick, box length, stride, true particle count, decimated positions and traits |
| `sample` | tick, the plan's keys and widths, the descriptor, and every gate's verdict |
| `search` | `started` with the resolved plan and shape, `stopping`, `done`, or `failed` with a reason |
| `generation` | index, kept, best, median, elapsed, the four-way tally, the leader's genome and descriptor |
| `specimen` | rank, score, genome, descriptor |
| `error` | one message, with the caller's own text quoted |

The catalog is data rather than documentation, so a metric cannot exist in the crate and be invisible to a page. Positions ship as decimal text at three decimals, descriptors at six. A harvest writes up to 512 specimens in one burst rather than a write and a flush apiece.

### 10.4 Search as a job

A search is assembled inside its own thread, so a long probe measurement never blocks the session loop and the plan it settles on is reported from the one place that knows it. Cancellation is an `AtomicBool` the per-generation watch closure reads, so the tuner needs no idea a frontend exists. A dropped receiver stops it too. The join reports a panicked search as a `failed` event rather than as silence, which is why the release profile keeps `panic = "unwind"`.

Starting a search pauses the world: rayon takes every core for a batch, and a playground stuttering at two frames a second reads as a bug. Stopping lands after the generation in flight, since a batch of rollouts is already out.

### 10.5 The relay and the page

`ui/` is outside the crate. `ui/server.py` spawns the binary, fans every event line out to each open page over server-sent events, and serves the page itself. It is the only thing in the stack that touches disk: a run is exactly the event lines a search produced, so recording one is appending them to `ui/runs/*.jsonl`.

A frame is a picture of now, so a late one replaces the picture already waiting rather than queueing behind it, and a browser that fell behind catches up instead of replaying the past. Everything else is news and is kept, up to a backlog of 400 lines. The reader thread ending is the only signal that the harness is gone, so it ends that way however it ends, and the page hears about it as a `harness` event of its own.

The page holds no model of the world. Every control sends a command line, every readout came from an event, and what survives a reload is taste rather than anything the simulation owns. `ui/world.js` deposits each frame into a periodic density field on the GPU and raymarches the picture out of that field, reading density in multiples of a uniform swarm so one threshold means the same thing at any particle count, box, or decimation stride.

## 11. Module map

| Module | Responsibility |
|---|---|
| `engine/substrate.rs` | Positions, traits, box geometry, and the private spatial hash grid. |
| `engine/trait.rs` | Anchor basis, active memberships, trait seeding from logits. |
| `engine/kernel.rs` | Gaussian mixtures with analytic slopes, periodic distance, the cutoff radius. |
| `engine/params.rs` | `FixedGenome` and `Genome`: layout, decode, uniform draws, per-gene bounds, clamping. |
| `engine/matrix.rs` | The pair-indexed interaction matrix, derived from a genome and calibrated. |
| `engine/resolve.rs` | `Probe`: the measured density reference and the box a coordination gene resolves to. |
| `engine/lenia.rs` | One Particle-Lenia step: which machine runs it, and the walk on the cores. |
| `engine/gpu/mod.rs` | The same step on the graphics card: the device, its buffers, and the round trip. |
| `engine/gpu/step.wgsl` | That step as a compute shader. The one place the physics is written twice. |
| `engine/sim.rs` | Authoritative run state, reconstructible from the genome halves and the box. |
| `harness/mod.rs` | The session loop: read, step, report, repeat, and the command dispatch. |
| `harness/protocol.rs` | The frontend boundary both ways: command parsing, JSON escaping, event writing. |
| `harness/catalog.rs` | What a frontend may choose from, handed over as data rather than documented. |
| `harness/playground.rs` | The live world a frontend drives, and the live measurement that keeps reading across ticks. |
| `harness/search.rs` | A search assembled from one command line, run on its own thread, reported per generation. |
| `harness/rollout.rs` | One genome in, one descriptor out. The only place a `Sim` is built to be measured. |
| `main.rs` | The process entry point. One call into `harness::run`. |
| `util.rs` | Deterministic xorshift, Euclidean distance, the non-finite guard. |
| `tuner/metrics.rs` | `Spec`, the plan, the shared blocks, evaluation and folding. |
| `tuner/metrics/field.rs` | The periodic density and trait grid every spatial metric reads. Not a metric. |
| `tuner/metrics/rdf.rs` | The all-pairs trait-by-radial histogram, and the radial distribution. |
| `tuner/metrics/topology.rs` | H0 barcode from the cutoff-limited spanning forest. |
| `tuner/metrics/robustness.rs` | The damage-and-recover experiment. |
| `tuner/gate.rs` | Valid ranges on named metrics. |
| `tuner/criterion.rs` | Which scoring function a search uses, and what it needs measured. |
| `tuner/algorithms.rs` | Which search loop runs. |
| `tuner/algorithms/evolve.rs` | Lanes, mutation, retention, the generation loop. |
| `tuner/driver.rs` | `Search`: the assembled run and the per-candidate path. |
| `tuner/specimen.rs` | A survivor: genome, descriptor, score. |

`structure`, `heterogeneity`, `connectivity`, `mobility`, `turnover`, `asymmetry`, and `temporal` each hold one measurement and its declared spec.

## 12. Test evidence

Three suites, one folder each, one flag each, all in `tests/` outside `src/`. `tests/run.sh` runs any combination and writes a dated report per suite into `tests/reports/`. `tests/TESTS.md` fixes the layout and says so.

| suite | shape | what it holds |
|---|---|---|
| `smoke` | 69 cases, seconds, in process | one promise per case, checked against the library |
| `behavior` | 79 promises, about half a minute | the real binary over its own protocol, plus the relay over a socket |
| `performance` | timings, minutes, release, `#[ignore]` | what the machine costs, and that a replay is exact |

**smoke** takes no child process and no sleep. `smoke/fixture.rs` is the shared setup: one small shape, a genome drawn inside it, the uniform cloud and law that genome makes, and the hunt for a genome still measurable after a short run. The three case files follow the crate's own halves.

- *Engine.* Hat memberships sum to one and activate at most two adjacent anchors, across two through six anchors, and are exactly one-hot at an anchor. Trait seeding reproduces the logit histogram exactly. Grid and all-pairs neighbor walks return identical pair sets. Periodic distance uses the minimum image across the wrap seam. A seeded simulation is bit-reproducible over fifty steps. Every anchor pair calibrates a finite, positive norm, and off-diagonal pairs get measured values rather than the fallback an unseeded calibration substrate produces.
- *Tuner.* A plan pulls in what a wanted metric reads, places every dependency ahead of its dependent, and names each metric once however many callers asked for it. Descriptor width equals the sum of the plan's widths, and every metric fills exactly the slots it declared. Every metric reads onto the shared axis on a surviving rollout. A gate rejects below its floor and above its ceiling, and an empty descriptor clears no gate with a floor. A `Once` metric holds its slots at zero before the final sample without shifting the layout, and a partial check skips its clause while still judging the rest. The same genome measures identically twice. The structure criterion tells genomes apart, which guards the failure where every clause multiplied raw, nearly every genome scored zero, and the search quietly became random.
- *Harness.* A law edit keeps the world and a world edit replaces it. A reading comes due on the cadence it was given. A live reading matches the rollout it imitates. A watched search reports every generation and stops when told. A survivor carries the genome and the descriptor it was scored on.

**behavior** runs as one test on purpose, because the report is the point: an area that panics costs its own row and the rest still runs. It groups by area, from the greeting through the command boundary, editing a world, measurement, gates, search, and what the Python relay owns.

**performance** records the machine and its load beside every row, because each is a wall-clock reading that assumes it has the machine to itself.

## 13. Limits

Read as absences in the code, not as a roadmap.

- **Traits are frozen.** No trait motion, birth, death, inheritance, or resource cycle.
- **The crate serializes nothing.** A run survives only as the event lines the Python relay appended, which hold genomes and descriptors and no simulation state. A search therefore cannot be stopped and continued; loading a survivor replays it from its shape and genes.
- **A search cannot be browsed while it runs.** The generation line carries the leader and the median, and the whole population waits until the search is over.
- **One algorithm, one absolute criterion.** No comparison against another search family exists in the crate, so no claim about `Evolve` beating anything is supported here.
- **The measurement grid pins a search to three axes** even though the engine reads dimensionality from data. Lifting it means generalizing `field` and `connectivity`, not relaxing the check.
- **Explicit Euler at a fixed timestep.** Divergence is caught by a finiteness check, not prevented by the integrator.
- **The physics is written twice**, once in `lenia::walk` and once in `step.wgsl`, and nothing but a test holds the two together. A change to one is a change owed to the other.
- **The card takes one swarm at a time**, not a whole generation, so a batch queues for it. At a hundred and fifty thousand particles a generation of forty-eight is minutes rather than seconds, and the ceiling on that is the card's memory bandwidth rather than its arithmetic.
- **The shell truncation** puts a small hard cutoff in the executed force.
- **Expedition targeting** in dozens of dimensions is random goal direction, not coverage-aware illumination.
- **The metric set is hand-designed.** It reads physical behavior without a projection, and it still encodes a prior about what is worth noticing.
- **`structure` can be satisfied by a static lattice** on its contrast term alone; only the `turnover` and `robustness` clauses stop that, and both are floored rather than required.
