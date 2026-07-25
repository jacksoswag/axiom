# axiom

AXIOM searches for Particle-Lenia interaction laws. A law is a flat vector of floats. The engine turns one into a running simulation, the tuner reduces that simulation to a bounded descriptor and judges it, the search breeds the genomes whose behavior scored well, and the harness puts all of it behind one line-oriented protocol a frontend can drive.

- [docs/TECHNICAL_SPEC.md](docs/TECHNICAL_SPEC.md) is the implementation contract.
- [docs/MASTERDOC.html](docs/MASTERDOC.html) reads the same system at three depths.
- [docs/STYLE.md](docs/STYLE.md) governs anything written under `src/`.

## Shape of the crate

One library, four modules, one binary. The binary reads commands on stdin and writes JSON events on stdout, so a frontend is whatever chooses to speak to it. A caller who would rather skip the protocol assembles a `Search` in Rust and gets a ranked `Vec<Specimen>` back in process.

| module | job |
|---|---|
| `engine` | a simulation, as a pure function of a genome |
| `tuner` | measure, gate, score, search |
| `harness` | the protocol, the live world, and the rollout that turns one genome into one descriptor |
| `util` | seeded randomness, Euclidean distance, the non-finite guard |

`engine` imports nothing from `tuner` or `harness`, so the simulation cannot acquire a dependency on how it is searched or drawn. The other two lean on each other by design: `harness` drives a `Search`, and `tuner::driver` reaches back into `harness::rollout`, which is the one place a `Sim` is built to be measured. `tuner` builds no simulation of its own.

Outside the crate, `ui/` is a Python relay and a browser page. The relay spawns the binary, fans its event lines out to every open page over server-sent events, and appends the lines a search produces to `ui/runs/*.jsonl`. The page holds no model of the world: every control sends a command line and every readout came from an event.

## The simulation

Particles live in periodic bounds. Each carries a position and one fixed trait in `[0, 1]`. Positions move; traits do not.

Dimensionality is data, not a constant. `FixedGenome` names it, and the substrate, the step, the distance function, and the density probe all read it from there. The one exception is the measurement grid: it indexes a cubic three-axis lattice, so `Search::new` rejects a genome whose dimensionality disagrees with it. That constraint belongs to one metric block, not to the physics.

Traits map onto anchors evenly spaced on a circle by piecewise-linear hats. At most two adjacent anchors carry weight, a trait sitting exactly on an anchor is one-hot there, and the trait axis wraps at the `0/1` seam instead of clamping.

Every ordered anchor pair owns its own law: a mixture of Gaussian **shells** sensing over distance, a mixture of Gaussian **bumps** responding over sensed density, and one directed weight. Because the pairs are ordered, one anchor can chase another that ignores it back.

A step accumulates, for each pair, what a particle senses and the gradient of that sensing, then converts it into motion with an explicit Euler step wrapped back onto the box. Value and slope come out of one function, so the gradient reuses each Gaussian term instead of recomputing it.

Above 16,384 particles that step runs on the graphics card, and below it on the cores. Both walk the same neighbor index, which the CPU builds either way. The card is worth about four times the eight cores at a hundred and fifty thousand particles, and nothing but size decides: there is no flag to set and no build to pick. `AXIOM_GPU=0` forces the cores, which is how the two get compared. They agree to the precision `f32` has rather than exactly, because `exp()` on a card is its own approximation, so a replay is exact within a backend and close across them.

## The genome

```text
[ coordination ]
[ anchor_count trait logits ]
[ anchor_count^2 pair blocks, source-major ]

pair block =
  shells x [amp, peak, width]
  bumps  x [amp, peak, width]
  [directed weight]

pair_stride = 3 * (shells + bumps) + 1
gene_len    = 1 + anchor_count + anchor_count^2 * pair_stride
```

`FixedGenome` is the half a search holds still: particle count, dimensions, anchor count, shell and bump counts, radius, timestep, seed. `Genome` is the half it tunes, decoded out of the flat vector.

Box size is derived, never a free gene. A `Probe` measures what a particle senses in a uniform population at a trial box, once per search, and each genome's coordination gene rescales that reference by the dimensional root. Coordination runs from 3 to 20 neighbors.

Bounds are widest at the largest box, so a search mutates under bounds taken at maximum coordination and every genome it can produce stays legal. A rollout re-clamps the pair genes at the box its own gene resolves to, where kernel reach is tighter. Clamping replaces a non-finite gene with its lower bound, so one NaN cannot poison a decode.

## Measurement

Each metric owns a file and declares its own spec: a stable key, how many descriptor slots it owns, which grids it reads, which metrics it depends on, how it reduces across samples, and which shared blocks it needs. A descriptor is a bare `Vec<f32>`; the plan is its layout, so reading one slot needs both.

Nothing measures everything by default. A criterion names what it scores on, the gates name what they judge, and `plan` closes that set over dependencies. Fifty-four slots when every metric runs:

| slots | metric | reading |
|---:|---|---|
| 18 | `rdf` | three trait-distance bands by six log radial bins, as ratios against the uniform start |
| 15 | `heterogeneity` | five local readings on grids of side 4, 8, and 16 |
| 8 | `topology` | seven H0 death-scale masses plus one for components still separate at the cutoff |
| 4 | `connectivity` | dense and void winding fractions at two density thresholds |
| 4 | `temporal` | variance of the spatial picture, plus autocorrelation at lags 1, 2, and 4 |
| 1 | `structure` | mean departure of the radial ratios from uniform |
| 1 | `mobility` | minimum-image travel between samples, as a fraction of the box |
| 1 | `turnover` | grid mass that changed cells between samples |
| 1 | `asymmetry` | departure of the directed weight matrix from its own transpose |
| 1 | `robustness` | share of injected damage closed against an undamaged control |

Every slot lands in `[0, 1]`, so no raw statistic with a wide range dominates a distance.

Three metrics cost more than an observation. `rdf` walks every pair among its sample, deliberately: its far bins reach the box half-diagonal, past any cutoff the neighbor grid could exploit. Every other block is linear in the swarm and that one is quadratic, so it stands on an evenly strided sample of at most 4,096 particles, which holds its cost flat above that size and leaves it exact below. `temporal` reads three other metrics back out of the descriptor being built, which is why a plan orders dependencies ahead of dependents. `robustness` runs an experiment rather than an observation, so it reduces once per rollout instead of once per sample.

Shared work is built once per sampled tick from the union of what the plan asks for: the pair histogram, one spatial field per requested grid side, the neighbor barcode, and the previous sample's positions. Two metrics wanting one grid cost one grid. Asking for a block the plan never requested panics rather than inventing a reading.

## Rollout

`harness::rollout::sim` decides when to look, never what at.

1. Decode, resolve the box, re-clamp the pair genes at it.
2. Step. A non-finite coordinate ends the rollout on the spot.
3. Sample eight times across the latter half of the run.
4. From the second sample on, check the gates. Two consecutive failures end it.
5. Fold the history: `Mean` metrics average, `Last` and `Once` take the final sample.

An abandoned rollout returns an empty descriptor, which reads zero everywhere and so clears no gate carrying a floor. Patience of two rather than one is deliberate: a sim can dip out of a band and climb back within a sample.

## Judging and scoring

A **gate** is a floor and a ceiling on one metric, read in that metric's own units. Gates are owned by the `Search` because a caller builds them at runtime. A mid-rollout check skips any gate whose metric reduces `Once`, since that metric has not been measured yet.

A **criterion** produces one number and names what it needs measured. Two exist:

- **Novelty** scores mean distance to the fifteen nearest behaviors already found, over the whole descriptor. No target and no gradient toward one, so it dodges the trivial attractors an absolute score falls into and equally never climbs toward a goal you had in mind. Being population-relative, every stored score goes stale when the population moves, so the whole population is re-scored before anything is ranked.
- **Structure** rewards holding regions that differ, keeping open space between them, surviving damage, and still turning over. Five clauses multiply, so weakness anywhere costs everywhere, each floored at `0.02` so one zero cannot annihilate the product. Being absolute, it can be climbed toward, and it can be fooled: a static lattice satisfies the contrast term outright.

## Search

`Search` holds the whole run: fixed genome, algorithm, criterion, gates, rollout length, seed. The probe, the bounds, and the metric plan derive from those six. `Search::run` executes and returns survivors ranked best first.

`Evolve` is the only algorithm. Each generation re-scores the population, proposes a batch across three lanes, evaluates it in parallel with an order-preserving filter, and folds the survivors in:

| lane | parent |
|---|---|
| tournament | binary tournament on score |
| expedition | whoever sits nearest a uniform random point in descriptor space |
| fresh | a new uniform draw |

Fresh draws take whatever the two named lanes do not claim, so the three always sum to the batch and a converged population always has a way out. Zero both named lanes and the whole thing collapses to random search, which is the baseline any real setting has to beat.

Mutation is iso plus line: per-gene noise scaled to each gene's own range, then one shared step along the vector toward a second parent, so a child inherits a direction the population already contains instead of dissolving into noise.

Retention sorts best score first and keeps a member only if it sits clear of everything already kept, at a quarter of the population's own median nearest-neighbor spacing. A fraction rather than an absolute distance, so it holds whatever units and however many axes a descriptor turns out to have. A converged batch therefore cannot fill the population with copies of one genome.

## The harness

One command per line in, one JSON event per line out. Commands are `verb key=value`, so nothing in the crate parses JSON; events are JSON, so a browser does not have to parse anything either. This is the only place the crate prints, spawns a thread, or holds a world that is still running.

| command | effect |
|---|---|
| `catalog` | re-send everything a frontend needs before it can draw a control |
| `shape` | any subset of the fixed genome, range-checked at this boundary |
| `gene` / `genome` | edit genes by index, or replace the whole flat vector |
| `run` / `pause` / `step` / `speed` / `reseed` | drive the live world |
| `measure` | what to watch and how often, or one full reading on the spot |
| `gates` | the whole gate set at once, as `metric=floor:ceiling` |
| `frames` | stop or resume the position stream |
| `search` | start or stop the one search a session runs at a time |
| `quit` | end the session |

Events come back as `catalog`, `layout`, `state`, `frame`, `sample`, `search`, `generation`, `specimen`, and `error`. The catalog is data rather than documentation: every metric with its cost and dependencies, every criterion, every evolve knob with its range, and every shape field. A frontend builds its own controls out of it, so a metric cannot appear in the crate and be invisible to the page.

A running world is paced to a 40 ms interval and owes `speed` ticks per interval, so the knob is a rate rather than a batch size. A search takes every core, which is why starting one pauses the world.

## Run it

```bash
cargo test                    # the fast in-process suite
tests/run.sh --smk            # the same suite, with a dated report
python3 ui/server.py          # the harness behind a browser on 127.0.0.1:8731
```

Drive the protocol directly, without the browser. Closing stdin finishes whatever was still owed, so a piped script runs to the end and exits on its own; `quit` returns on the spot instead, dropping any steps not yet taken.

```bash
printf 'frames off\nmeasure keys=connectivity,turnover every=50\nstep count=400\n' \
  | cargo run --release
```

Assemble a search in code instead, skipping the protocol:

```rust
use axiom::engine::params::FixedGenome;
use axiom::tuner::algorithms::{evolve::{Evolve, Mutation}, Algorithm};
use axiom::tuner::criterion::Criterion;
use axiom::tuner::driver::Search;
use axiom::tuner::gate::Gate;
use axiom::tuner::metrics::{mobility, structure};

let search = Search::new(
    FixedGenome { particle_count: 1200, dimensions: 3, anchor_count: 4, shells: 2,
        bumps: 1, radius: 2.0, dt: 0.05, seed: 1 },
    Algorithm::Evolve(Evolve { generations: 25, batch: 64, capacity: 128,
        mutation: Mutation { tournament: 0.7, expedition: 0.2, iso: 0.01, line: 0.05 } }),
    Criterion::Structure,
    vec![Gate { metric: mobility::METRIC, floor: 0.01, ceiling: 1.0 },
         Gate { metric: structure::METRIC, floor: 0.02, ceiling: 0.9 }],
    1500,
    1,
);
// watch sees each generation as it lands and answers whether to keep going
let ranked = search.run(&mut |_generation, _population, _tally| true); // best first
```

## What the code does not contain

- Traits do not change during a rollout. No trait motion, birth, death, or inheritance.
- The crate serializes nothing. A run survives only as the event lines the Python relay appended to disk, which hold genomes and descriptors and no simulation state, so a search cannot be stopped and continued.
- One search algorithm and one absolute criterion, so no comparison against another search family exists in the crate.
- Explicit Euler at a fixed timestep. Divergence is caught by a finiteness check, not prevented by the integrator.
- Shells truncate at 3.5 widths past their peak, so the executed force carries a small hard cutoff and is not the exact gradient of the untruncated energy there.
- The measurement grid pins a search to three dimensions even though the engine does not.
