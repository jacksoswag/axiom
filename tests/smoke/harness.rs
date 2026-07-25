//! Harness behavior: what a command line turns into, what the catalog promises a frontend, what an
//! edit does to a running world, and whether a live reading means the same thing as a searched one.

use axiom::engine::params::Genome;
use axiom::engine::resolve::Probe;
use axiom::harness::catalog::{body, layout, metric_of};
use axiom::harness::playground::Playground;
use axiom::harness::protocol::{numbers, quote, Command};
use axiom::harness::rollout::{self, SAMPLES};
use axiom::harness::search;
use axiom::tuner::algorithms::evolve::{Evolve, Mutation};
use axiom::tuner::algorithms::Algorithm;
use axiom::tuner::criterion::Criterion;
use axiom::tuner::driver::Search;
use axiom::tuner::gate::Gate;
use axiom::tuner::metrics::{connectivity, fold, heterogeneity, plan, temporal, ALL};

use crate::fixture;

// the command line

#[test]
fn a_command_reads_its_verb_its_numbers_and_its_gene_edits() {
    let command = Command::parse("gene 3=0.25 12=-1.5 quiet").expect("a line with a verb parses");
    assert_eq!(command.verb, "gene");
    assert!(command.flag("quiet"), "a bare word is a flag");
    assert_eq!(command.indexed().expect("both edits are numbers"), vec![(3, 0.25), (12, -1.5)]);

    let shape = Command::parse("shape particles=400 keys=a,b").expect("parses");
    assert_eq!(shape.number("particles").expect("a number"), Some(400.0));
    assert_eq!(shape.count("particles").expect("a count"), Some(400));
    assert_eq!(shape.number("missing").expect("absent is not an error"), None);
    assert_eq!(shape.words("keys"), vec!["a", "b"]);
    assert!(shape.words("missing").is_empty(), "an absent list read as something");
    assert!(Command::parse("").is_none(), "a blank line is not a command");
    assert!(Command::parse("   ").is_none(), "a line of spaces is not a command");
    assert_eq!(Command::parse("STEP COUNT=5").expect("parses").verb, "step", "a verb arrived case sensitive");
}

/// The boundary is the one place an outside number is checked, so anything unreadable has to be caught
/// here: past it a NaN reaches a decode and a garbage count reaches an index.
#[test]
fn the_boundary_refuses_what_it_cannot_read() {
    let command = Command::parse("shape particles=lots radius=nan").expect("parses");
    assert!(command.number("particles").is_err(), "'lots' is not a number");
    assert!(command.number("radius").is_err(), "a NaN gene would poison a decode");
    assert!(Command::parse("shape dt=inf").unwrap().number("dt").is_err(), "an infinite timestep is not a number");
    assert!(Command::parse("gene 4=oops").unwrap().indexed().is_err());
    assert!(Command::parse("genome genes=0.1,bad").unwrap().floats("genes").is_err());
    assert_eq!(Command::parse("step count=-40").unwrap().count("count").expect("a count"), Some(0),
        "a negative count would wrap into a very long fast forward");
}

/// A seed is a whole number too large to survive an f32, and the one field that has to arrive
/// exactly: two runs of one seed are the same run or nothing here is reproducible.
#[test]
fn a_seed_crosses_the_boundary_intact() {
    let line = Command::parse("reseed seed=12345678901234567").expect("parses");
    assert_eq!(line.seed("seed").expect("a whole seed"), Some(12345678901234567));
    assert!(Command::parse("reseed seed=1.5").unwrap().seed("seed").is_err(), "a fractional seed was accepted");
    assert!(Command::parse("reseed seed=-1").unwrap().seed("seed").is_err(), "a negative seed was accepted");
    assert_eq!(Command::parse("reseed").unwrap().seed("seed").expect("absent is not an error"), None);
}

#[test]
fn json_leaves_here_readable_by_whoever_asked() {
    assert_eq!(numbers(&[1.0, -0.5], 2), "[1.00,-0.50]");
    assert_eq!(numbers(&[f32::NAN], 2), "[null]", "a reader gets nothing rather than a fake number");
    assert_eq!(numbers(&[f32::INFINITY], 2), "[null]");
    assert_eq!(numbers(&[], 2), "[]");
    assert_eq!(quote("say \"hi\"\n"), "\"say \\\"hi\\\"\\n\"");
    assert_eq!(quote("a\tb"), "\"a\\tb\"");
    assert_eq!(quote("\u{1}"), "\"\\u0001\"", "a control character walked into the line raw");
    assert_eq!(quote("\u{7f}"), "\"\\u007f\"");
}

// the catalog

/// A metric a frontend cannot name is a metric no search started from a frontend can ever use, so the
/// catalog has to carry every one the tuner has.
#[test]
fn the_catalog_names_every_metric_the_tuner_has() {
    let catalog = body();
    for metric in ALL {
        assert!(metric_of(metric.key).is_some(), "{} cannot be named", metric.key);
        assert!(catalog.contains(&format!("\"key\":\"{}\"", metric.key)), "{} is missing", metric.key);
        assert!(catalog.contains(&format!("\"width\":{}", metric.width)), "{} has no width", metric.key);
    }
    assert!(metric_of("invented").is_none());
    assert!(catalog.contains("\"name\":\"novelty\"") && catalog.contains("\"name\":\"structure\""));
    assert!(catalog.contains("\"name\":\"generations\""), "a knob nobody can set is a knob nobody has");
}

/// The layout is the only thing telling a frontend what a gene does. An entry that names the wrong
/// pair, or a genome with more genes than entries, is a slider wired to somebody else's law.
#[test]
fn the_layout_names_every_gene_in_the_order_the_genome_holds_them() {
    let world = Playground::new(fixture::shape(120));
    let shape = &world.shape;
    let body = layout(shape, &world.bounds(), &world.genes, world.box_len());
    assert_eq!(body.matches("{\"index\":").count(), shape.gene_len(), "the layout and the genome disagree in length");
    assert_eq!(body.matches("\"kind\":\"coordination\"").count(), 1);
    assert_eq!(body.matches("\"kind\":\"logit\"").count(), shape.anchor_count);
    assert_eq!(body.matches("\"kind\":\"pair\"").count(), shape.anchor_count * shape.anchor_count * shape.pair_stride());
    for source in 0..shape.anchor_count {
        for destination in 0..shape.anchor_count {
            assert!(body.contains(&format!("\"source\":{source},\"destination\":{destination},")),
                "no gene belongs to the block for {source} onto {destination}");
        }
    }
    // The slot names walk one pair block in the order Matrix::derive reads it: shells, bumps, weight.
    for part in ["shell 0 amp", "shell 0 peak", "shell 0 width", "bump 0 amp", "weight"] {
        assert!(body.contains(&format!("\"label\":\"{part}\"")), "no gene is called '{part}'");
    }
}

// editing a live world

/// The whole point of the two edit paths: a pair gene changes the law and leaves the particles alone,
/// which is what makes an edit judgeable against the state it was made in. A logit changes who was
/// seeded where, so the world has to start over.
#[test]
fn a_law_edit_keeps_the_world_and_a_world_edit_replaces_it() {
    let mut world = Playground::new(fixture::shape(120));
    world.advance(20);
    let before = world.frame_body().to_owned(); // the world keeps the buffer it wrote, so take a copy
    let pair_gene = 1 + world.shape.anchor_count + 1; // the first pair block's peak
    assert!(!world.shapes_world(pair_gene));
    world.edit(pair_gene, 0.5);
    world.refresh_law();
    assert_eq!(world.frame_body(), before, "a law edit moved the particles");
    assert_eq!(world.tick(), 20);

    assert!(world.shapes_world(1), "a trait logit sets who is seeded where");
    world.edit(1, 1.0);
    world.reseed();
    assert_eq!(world.tick(), 0, "a reseeded world kept its old tick");
    assert_ne!(world.frame_body(), before);
}

#[test]
fn an_edit_lands_inside_what_the_box_allows() {
    let mut world = Playground::new(fixture::shape(120));
    let bounds = world.bounds();
    let peak = 1 + world.shape.anchor_count + 1;
    world.edit(peak, 1e9);
    assert_eq!(world.genes[peak], bounds[peak].1, "a slider from a wider box was not brought in");
    world.edit(0, -5.0);
    assert_eq!(world.genes[0], bounds[0].0);
}

/// A genome arrives raw from a recorded run, so the only thing standing between a saved specimen and
/// a kernel folded onto its own image around the torus is the clamp a build does at the box that
/// genome resolved to. Decoding alone does not do it: the pair genes come through untouched.
#[test]
fn a_rollout_pulls_a_wide_genome_back_inside_its_own_box() {
    let fixed = fixture::shape(120);
    let probe = Probe::new(&fixed);
    let mut genes = vec![1e6f32; fixed.gene_len()];
    genes[0] = 6.0; // a legal coordination, so the box itself is ordinary
    let sim = rollout::build(&fixed, &genes, &probe);
    let box_len = sim.substrate.box_len;
    assert!(sim.matrix.max_reach.is_finite() && sim.matrix.max_reach > 0.0);
    assert!(sim.matrix.max_reach < box_len * 0.5,
        "a genome from off the end reached {} in a box of {box_len}", sim.matrix.max_reach);
    let unclamped = axiom::engine::matrix::Matrix::derive(&fixed, &fixed.decode(&genes));
    assert!(unclamped.max_reach > box_len, "the fixture never asked for a reach worth clamping");
}

// measuring a live world

/// A live number and a searched number have to mean the same thing, or the playground is a different
/// experiment from the search. Same genome, same schedule, same fold: the same descriptor.
#[test]
fn a_live_reading_matches_the_rollout_it_imitates() {
    let shape = fixture::shape(120);
    let assembled = plan(&[temporal::METRIC]);
    let timesteps = 60;
    let (genes, expected) = fixture::measurable(&shape, &assembled, timesteps);

    // The same schedule the rollout keeps: eight looks spread across the latter half of the run.
    let interval = (timesteps / (2 * SAMPLES)).max(1);
    let first = timesteps - interval * SAMPLES;
    let last = timesteps - interval;
    let mut world = Playground::new(shape);
    world.genes = genes;
    world.reseed();
    world.watch(&[temporal::METRIC], 1);
    let mut samples = Vec::new();
    for tick in 0..timesteps {
        world.advance(1);
        if tick >= first && (tick - first) % interval == 0 { samples.push(world.measure(tick == last)); }
    }
    assert_eq!(samples.len(), SAMPLES);
    assert_eq!(fold(&assembled, &samples), expected, "a live reading drifted from the rollout's");
}

/// A gate whose metric went unmeasured would read 0 and look failed rather than unknown, so setting
/// one has to pull it into the plan even when nobody asked to watch it.
#[test]
fn a_gate_pulls_its_own_metric_into_a_live_reading() {
    let mut world = Playground::new(fixture::shape(120));
    world.watch(&[], 0);
    assert!(world.measure(true).is_empty(), "a world watching nothing measured something anyway");
    world.set_gates(vec![Gate { metric: connectivity::METRIC, floor: 0.0, ceiling: 1.0 }]);
    let values = world.measure(true);
    assert!(!values.is_empty(), "a gate's own metric went unmeasured");
    let body = world.sample_body(&values, true);
    assert!(body.contains("\"connectivity\""), "the reading never names the metric it was gated on: {body}");
    assert!(body.contains("\"pass\":true"), "a reading inside a band of nought to one failed it: {body}");
}

#[test]
fn a_reading_comes_due_on_the_cadence_it_was_given() {
    let mut world = Playground::new(fixture::shape(120));
    assert!(!world.measurement_due(), "a world watching nothing owes a reading");
    world.watch(&[heterogeneity::METRIC], 5);
    world.advance(4);
    assert!(!world.measurement_due(), "a reading came due four ticks into a cadence of five");
    world.advance(1);
    assert!(world.measurement_due());
    world.measure(false);
    assert!(!world.measurement_due(), "a reading stayed due after it was taken");
}

// driving a search

/// Progress and cancellation are one closure. A watch that says stop keeps whatever was found, which is
/// what lets a frontend end a long search without throwing the population away.
#[test]
fn a_watched_search_reports_every_generation_and_stops_when_told() {
    let evolve = Evolve { generations: 6, batch: 6, capacity: 6,
        mutation: Mutation { tournament: 0.5, expedition: 0.25, iso: 0.05, line: 0.3 } };
    let search = Search::new(fixture::shape(120), Algorithm::Evolve(evolve), Criterion::Structure,
        Vec::new(), 40, 3);
    let mut seen = Vec::new();
    let population = search.run(&mut |generation, population, tally| {
        seen.push((generation, population.len()));
        assert_eq!(tally.evaluated, 6, "a generation judged something other than its batch");
        assert!(tally.died + tally.gated + tally.unscorable <= tally.evaluated,
            "more candidates were turned away than were ever proposed");
        generation < 1 // two generations, then the frontend changed its mind
    });
    assert_eq!(seen.iter().map(|(generation, _)| *generation).collect::<Vec<_>>(), vec![0, 1],
        "a stopped search kept going, or never reported");
    assert!(!population.is_empty(), "stopping threw away what it had found");
    assert!(population.windows(2).all(|pair| pair[0].score >= pair[1].score), "survivors came back unranked");
}

/// Refining is what joins a live world to a search: the genome on screen goes in as a starting point and
/// comes back in the population, so a search improves on it instead of starting over from noise.
#[test]
fn a_seeded_search_starts_from_what_it_was_handed() {
    let shape = fixture::shape(120);
    let assembled = plan(&[axiom::tuner::criterion::structure::METRICS[0]]);
    let (genes, _) = fixture::measurable(&shape, &assembled, 40);

    let evolve = Evolve { generations: 0, batch: 2, capacity: 8,
        mutation: Mutation { tournament: 0.0, expedition: 0.0, iso: 0.0, line: 0.0 } };
    let mut search = Search::new(shape, Algorithm::Evolve(evolve), Criterion::Structure,
        Vec::new(), 40, 5);
    search.seeds = vec![genes.clone()];
    let population = search.run(&mut |_, _, _| true);
    assert!(population.iter().any(|member| member.genome == genes),
        "the genome the search was handed is not in what came back");
}

/// A config a frontend sent is the one place a search can be asked for something that does not exist.
/// Every one of those has to come back as a complaint rather than as a search scoring everything zero.
#[test]
fn a_search_refuses_a_config_it_cannot_run() {
    let shape = fixture::shape(80);
    let started = |line: &str| search::start(&Command::parse(line).expect("parses"), &shape, Vec::new(), Vec::new());
    assert!(started("search start criterion=novelty").is_err(), "novelty with nothing to be far apart in");
    assert!(started("search start criterion=novelty axes=nosuchmetric").is_err(), "an axis nobody has");
    assert!(started("search start criterion=invented").is_err(), "a criterion nobody wrote");
    assert!(started("search start algorithm=invented").is_err(), "an algorithm nobody wrote");
    assert!(started("search start criterion=structure batch=lots").is_err(), "a knob that is not a number");

    let job = started("search start criterion=novelty axes=rdf generations=0 batch=2 capacity=2 timesteps=20 seed=3")
        .expect("a config it can run");
    let population = job.join().expect("the search thread came back");
    assert!(population.len() <= 2, "a batch of two came back with {}", population.len());
}

/// Genes travel raw so a specimen loads back as the genome it was, and the descriptor travels beside
/// them so a frontend can rank a saved run without measuring it again.
#[test]
fn a_survivor_carries_the_genome_and_the_descriptor_it_was_scored_on() {
    let shape = fixture::shape(120);
    let evolve = Evolve { generations: 0, batch: 4, capacity: 4,
        mutation: Mutation { tournament: 0.5, expedition: 0.25, iso: 0.05, line: 0.3 } };
    let search = Search::new(shape.clone(), Algorithm::Evolve(evolve), Criterion::Structure,
        Vec::new(), 40, 5);
    let population = search.run(&mut |_, _, _| true);
    assert!(!population.is_empty(), "two generations of four kept nothing at all");
    let probe = Probe::new(&shape);
    for member in &population {
        assert_eq!(member.genome.len(), shape.gene_len(), "a survivor came back the wrong length");
        assert_eq!(member.metrics.len(), search.metrics.iter().map(|metric| metric.width).sum::<usize>(),
            "a survivor's descriptor does not fit the plan it was measured under");
        assert!(member.score.is_finite() && member.score > 0.0);
        let again = rollout::sim(&shape, &search.metrics, &[], &probe, &member.genome, search.timesteps);
        assert_eq!(again, member.metrics, "a survivor measured differently the second time it was run");
    }
    let fresh = Genome::build_random(&shape.bounds(&probe), 3);
    assert_eq!(fresh.len(), shape.gene_len(), "a fresh draw is not a genome this shape can hold");
}
