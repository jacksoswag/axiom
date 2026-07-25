//! What the harness promises a page that cannot see inside it: what a fresh connection is handed, what
//! a bad line does, what an edit costs, what running and measuring and searching look like from the
//! outside, and that closing the pipe leaves nothing running.

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use axiom::harness::catalog::SHAPE_FIELDS;
use axiom::tuner::metrics::ALL;

use crate::wire::{field, genes_of, kind, list, number, text, Harness, PATIENCE};
use crate::Report;

/// A page that connects late, or reloads, has nothing of its own: what it can draw is exactly what the
/// greeting carries.
pub fn greeting(report: &mut Report) {
    let mut harness = Harness::open();
    let mut opening = Vec::new();
    while let Some(line) = harness.next(PATIENCE) {
        let done = kind(&line) == "frame";
        opening.push(kind(&line));
        if done { break; }
    }
    report.check("an unprompted harness greets with catalog, layout, state, frame",
        opening == ["catalog", "layout", "state", "frame"], format!("{opening:?}"));

    let look = harness.look();
    let catalog = look.last("catalog");
    let missing: Vec<&str> = ALL.iter().map(|metric| metric.key)
        .filter(|key| !catalog.contains(&format!("\"key\":\"{key}\""))).collect();
    report.check("the catalog names every metric the tuner has", missing.is_empty(),
        format!("{} metrics, missing {missing:?}", ALL.len()));
    let absent: Vec<&str> = SHAPE_FIELDS.iter().map(|(name, ..)| *name)
        .filter(|name| !catalog.contains(&format!("\"name\":\"{name}\""))).collect();
    report.check("the catalog names every shape field the parser reads", absent.is_empty(),
        format!("{} fields, missing {absent:?}", SHAPE_FIELDS.len()));
    report.check("asking again re-greets, so a reload needs nothing kept",
        !look.layout().is_empty() && !look.state().is_empty() && !catalog.is_empty(),
        format!("{} lines back", look.lines.len()));

    let state = look.state();
    let carried = ["running", "tick", "box_len", "gridded", "speed", "every", "watching", "gene_count", "shape"];
    let thin: Vec<&&str> = carried.iter().filter(|key| field(&state, key).is_none()).collect();
    report.check("state carries everything a frontend would otherwise remember", thin.is_empty(),
        format!("missing {thin:?}"));
    report.check("a fresh world is paused at tick zero",
        text(&state, "running") == "false" && number(&state, "tick") == 0.0,
        format!("running {}, tick {}", text(&state, "running"), number(&state, "tick")));
    harness.close();
}

/// The line protocol is the one place an outside number is checked. Past it a bad count reaches an
/// index and a NaN reaches a decode, so everything wrong has to come back as an error and change nothing.
pub fn boundary(report: &mut Report) {
    let mut harness = Harness::open();
    let opening = harness.look();
    let before = opening.state();

    harness.tell("nosuchverb");
    let look = harness.look();
    report.check("an unknown verb is an error rather than silence",
        look.errors().len() == 1 && look.complaint().contains("nosuchverb"), look.complaint());

    harness.tell("");
    harness.tell("   ");
    let look = harness.look();
    report.check("a blank line is not a command and not a complaint", look.errors().is_empty(),
        format!("{} errors", look.errors().len()));

    // A message carrying the characters JSON reserves has to come back escaped, or the frontend's own
    // parse is what breaks rather than the command.
    harness.tell("say\"it\\loud");
    let look = harness.look();
    let raw = look.errors().first().copied().unwrap_or("");
    let escaped = raw.contains("\\\"") && raw.contains("\\\\");
    report.check("an error quotes what the caller typed instead of breaking the line",
        escaped && text(raw, "message").starts_with("no command called"), raw.to_string());

    harness.tell("shape particles=lots");
    harness.tell("shape radius=nan");
    harness.tell("shape dt=inf");
    let look = harness.look();
    report.check("a number that is not a number is refused three times over", look.errors().len() == 3,
        format!("{} errors, first: {}", look.errors().len(), look.complaint()));

    let mut refused = Vec::new();
    for (name, _, low, high) in SHAPE_FIELDS {
        harness.tell(&format!("shape {name}={}", high + 1.0));
        harness.tell(&format!("shape {name}={}", low - 1.0));
        let look = harness.look();
        if look.errors().len() == 2 { refused.push(name); }
    }
    report.check("every shape field refuses both ends of its own range",
        refused.len() == SHAPE_FIELDS.len(), format!("{refused:?}"));

    harness.tell("shape dimensions=2");
    let look = harness.look();
    report.check("a shape the measurement grid cannot read is refused",
        look.errors().len() == 1, look.complaint());
    report.check("nothing that was refused changed the world", look.state() == before,
        format!("shape {}", text(&look.state(), "shape")));

    // A seed is the one field wide enough to lose its low bits through a float, so it has a reader of
    // its own and a fractional one has to be turned away rather than truncated into a different run.
    harness.tell("reseed seed=12345678901234567");
    let look = harness.look();
    report.check("a seed too wide for a float still arrives whole",
        look.errors().is_empty() && text(&look.state(), "shape").contains("12345678901234567"),
        format!("shape {}", text(&look.state(), "shape")));
    harness.tell("reseed seed=1.5");
    let look = harness.look();
    report.check("a seed that is not a whole number is refused", look.errors().len() == 1, look.complaint());
    harness.close();
}

/// A batch of gene edits is one command. A frontend that hears an error reads it as "nothing happened",
/// so a batch that applied its good half before complaining would be lying to it.
pub fn editing(report: &mut Report) {
    let mut harness = Harness::open();
    let opening = harness.look();
    let layout = opening.layout();
    let genes: Vec<f64> = genes_of(&layout).iter().map(|gene| number(gene, "value")).collect();
    let count = genes.len();
    report.check("the layout describes every gene the genome holds",
        count == number(&opening.state(), "gene_count") as usize && count > 0,
        format!("{count} genes"));

    harness.tell(&format!("gene 1=0.5 {}=9", count + 40));
    let look = harness.look();
    let after: Vec<f64> = genes_of(&look.layout()).iter().map(|gene| number(gene, "value")).collect();
    report.check("a batch naming one gene out of range applies none of itself",
        look.errors().len() == 1 && after == genes,
        format!("{}, gene 1 is {}", look.complaint(), after.get(1).copied().unwrap_or(f64::NAN)));

    // A pair gene changes the law and leaves the particles where they are, which is what makes an edit
    // judgeable against the state it was made in.
    harness.tell("step count=25");
    let moved = harness.look();
    let before_frame = moved.last("frame");
    let pair = genes_of(&layout).iter().position(|gene| text(gene, "kind") == "pair").unwrap_or(count - 1);
    harness.tell(&format!("gene {pair}=0.4"));
    let look = harness.look();
    let frame = look.last("frame");
    report.check("a law edit keeps the world and its tick",
        number(&frame, "tick") == 25.0 && field(&frame, "positions") == field(&before_frame, "positions"),
        format!("tick {} then {}", number(&before_frame, "tick"), number(&frame, "tick")));

    harness.tell("gene 0=1.5");
    let look = harness.look();
    report.check("a world gene starts the world over",
        number(&look.state(), "tick") == 0.0, format!("tick {}", number(&look.state(), "tick")));

    // The slider a frontend shows may still hold the range of a wider box, so an edit past this box's
    // ceiling lands on the ceiling rather than reaching a kernel that wraps into itself.
    harness.tell(&format!("gene {pair}=100000"));
    let look = harness.look();
    let entry = genes_of(&look.layout()).into_iter().find(|gene| number(gene, "index") == pair as f64)
        .unwrap_or_default();
    report.check("an edit past the box's reach is brought back inside it",
        number(&entry, "value") <= number(&entry, "high") && number(&entry, "value") > 0.0,
        format!("{} against a ceiling of {}", number(&entry, "value"), number(&entry, "high")));

    let carried: Vec<String> = genes_of(&look.layout()).iter()
        .map(|gene| format!("{}", number(gene, "value"))).collect();
    harness.tell(&format!("genome genes={}", carried.join(",")));
    let look = harness.look();
    let round: Vec<String> = genes_of(&look.layout()).iter()
        .map(|gene| format!("{}", number(gene, "value"))).collect();
    report.check("a genome copied out and pasted back is the same genome",
        look.errors().is_empty() && round == carried, format!("{} genes round tripped", round.len()));

    harness.tell("genome genes=0.1,0.2");
    let look = harness.look();
    report.check("a genome of the wrong length is refused whole",
        look.errors().len() == 1 && genes_of(&look.layout()).len() == count, look.complaint());

    // A wider shape is a longer genome, and the layout is the only place a frontend hears about it.
    harness.tell("shape anchors=4");
    let look = harness.look();
    let widened = genes_of(&look.layout()).len();
    report.check("a wider shape redraws the genome and says how long it now is",
        widened > count && widened == number(&look.state(), "gene_count") as usize,
        format!("{count} genes became {widened}"));
    harness.tell(&format!("genome genes={}", carried.join(",")));
    let look = harness.look();
    report.check("a genome from the shape before it is refused rather than half applied",
        look.errors().len() == 1 && genes_of(&look.layout()).len() == widened, look.complaint());
    harness.tell("shape anchors=3");
    harness.look();

    harness.tell("reseed seed=4242");
    let first = harness.look().last("frame");
    harness.tell("reseed seed=99");
    harness.look();
    harness.tell("reseed seed=4242");
    let again = harness.look().last("frame");
    report.check("the same seed opens the same world",
        field(&first, "positions") == field(&again, "positions") && !first.is_empty(),
        format!("{} positions compared", list(&first, "positions").len()));
    harness.close();
}

/// Speed is a rate. It used to be a batch size on a loop that never slept, so the slider moved nothing
/// a viewer could see and a watched world pegged a core.
pub fn running(report: &mut Report) {
    let ticks_over = |speed: usize, window: Duration| {
        let mut harness = Harness::open();
        harness.look();
        harness.tell(&format!("speed steps={speed}"));
        harness.tell("run");
        std::thread::sleep(window);
        harness.tell("pause");
        let look = harness.look();
        harness.close();
        number(&look.state(), "tick")
    };
    let window = Duration::from_millis(600);
    let slow = ticks_over(1, window);
    let fast = ticks_over(40, window);
    report.check("the slowest speed is paced rather than free running", slow > 0.0 && slow < 60.0,
        format!("{slow} ticks in 600 ms"));
    report.check("a faster speed is faster", fast > slow * 3.0, format!("{fast} ticks against {slow}"));

    let mut harness = Harness::open();
    harness.look();
    harness.tell("speed steps=100000");
    let look = harness.look();
    let held = number(&look.state(), "speed");
    report.check("a speed past what one frame interval can carry is brought back inside it",
        held >= 1.0 && held < 100000.0, format!("100000 landed on {held}"));

    harness.tell("speed steps=1");
    harness.tell("step count=2000");
    let look = harness.look();
    report.check("a step of two thousand advances exactly two thousand",
        number(&look.state(), "tick") == 2000.0, format!("tick {}", number(&look.state(), "tick")));
    // A jump runs in slices, so a viewer watches it happen instead of waiting for the far end.
    report.check("a fast forward streams frames on the way rather than one at the end",
        look.of("frame").len() >= 2, format!("{} frames", look.of("frame").len()));

    harness.tell("frames off");
    harness.tell("run");
    std::thread::sleep(Duration::from_millis(300));
    harness.tell("pause");
    let look = harness.look();
    report.check("frames off means no frames, and the world runs anyway",
        look.of("frame").is_empty() && number(&look.state(), "tick") > 2000.0,
        format!("{} frames, tick {}", look.of("frame").len(), number(&look.state(), "tick")));

    harness.tell("frames");
    let look = harness.look();
    report.check("turning frames back on owes a picture at once, paused or not",
        !look.of("frame").is_empty(), format!("{} frames", look.of("frame").len()));

    let frame = look.last("frame");
    let positions = list(&frame, "positions").len();
    let traits = list(&frame, "traits").len();
    let box_len = number(&frame, "box_len");
    let inside = list(&frame, "positions").iter().all(|&value| value >= 0.0 && value <= box_len);
    report.check("a frame carries three coordinates per trait, all inside the box",
        positions == traits * 3 && inside && traits > 0,
        format!("{traits} points, box {box_len}"));
    report.check("a frame says how much of the swarm it is drawn from",
        number(&frame, "particles") >= traits as f64 && number(&frame, "stride") >= 1.0,
        format!("{} of {} particles, stride {}", traits, number(&frame, "particles"), number(&frame, "stride")));
    harness.close();
}

/// A reading has to be the same measurement a search would take, laid out the same way, or the
/// playground is a different experiment from the one the survivors came out of.
pub fn measuring(report: &mut Report) {
    let mut harness = Harness::open();
    harness.look();
    let keys: Vec<&str> = ALL.iter().map(|metric| metric.key).collect();
    harness.tell(&format!("measure keys={} every=5", keys.join(",")));
    harness.tell("measure now");
    let look = harness.look();
    let sample = look.last("sample");
    let widths: Vec<f64> = list(&sample, "widths");
    let values = list(&sample, "values").len();
    let planned: Vec<String> = field(&sample, "keys").unwrap_or("[]").trim_matches(|c| c == '[' || c == ']')
        .split(',').map(|key| key.trim_matches('"').to_string()).collect();
    report.check("a reading names every metric asked for, dependencies included",
        keys.iter().all(|key| planned.iter().any(|named| named == key)),
        format!("{} planned for {} asked", planned.len(), keys.len()));
    report.check("the widths a reading carries add up to the values it carries",
        widths.iter().sum::<f64>() == values as f64 && values > 0,
        format!("{} slots over {} metrics", values, widths.len()));
    report.check("a reading taken on demand is the full one, repair experiment included",
        text(&sample, "full") == "true", format!("full {}", text(&sample, "full")));
    let readable = list(&sample, "values").iter().all(|value| value.is_finite() && (0.0..=1.0).contains(value));
    report.check("every slot of every metric lands on the shared axis", readable,
        format!("{} slots inside 0 to 1", values));

    harness.tell("gates nosuchmetric=0:1");
    let look = harness.look();
    report.check("a gate on a metric nobody has is refused", look.errors().len() == 1, look.complaint());
    harness.tell("gates rdf=nonsense");
    let look = harness.look();
    report.check("a gate without a floor and a ceiling is refused", look.errors().len() == 1, look.complaint());
    harness.tell("gates rdf=0:oops");
    let look = harness.look();
    report.check("a gate whose bound is not a number is refused", look.errors().len() == 1, look.complaint());

    harness.tell("measure keys= every=5");
    let look = harness.look();
    report.check("an empty watch list leaves a world watching nothing",
        text(&look.state(), "watching") == "[]" && number(&look.state(), "every") == 0.0,
        format!("watching {}, every {}", text(&look.state(), "watching"), number(&look.state(), "every")));

    // A gate's metric is measured whether or not anyone asked to watch it, or it would read zero and
    // look failed rather than unknown.
    harness.tell("gates connectivity=0:1");
    harness.tell("measure now");
    let look = harness.look();
    let sample = look.last("sample");
    let verdict = field(&sample, "gates").unwrap_or("[]");
    report.check("a gate's own metric is measured for it",
        sample.contains("\"connectivity\"") && verdict.contains("\"pass\":true"),
        verdict.to_string());

    let reading = number(verdict, "value");
    harness.tell(&format!("gates connectivity={}:1", reading + 0.5));
    harness.tell("measure now");
    let look = harness.look();
    let sample = look.last("sample");
    let verdict = field(&sample, "gates").unwrap_or("[]");
    report.check("a reading below the floor fails its gate", verdict.contains("\"pass\":false"),
        format!("{reading} against a floor of {}", reading + 0.5));

    harness.tell("measure keys=heterogeneity every=1");
    harness.tell("run");
    std::thread::sleep(Duration::from_millis(600));
    harness.tell("pause");
    let look = harness.look();
    let samples = look.of("sample").len();
    report.check("a cadence in ticks is a cadence in wall time too",
        samples > 0 && samples <= 20, format!("{samples} readings in 600 ms"));
    harness.close();
}

/// A search is the point of the instrument. It has to report as it goes, keep what it found, and hand
/// back survivors a world can be loaded from.
pub fn searching(report: &mut Report) {
    let mut harness = Harness::open();
    harness.look();
    harness.tell("shape particles=80 anchors=2 shells=1 bumps=1");
    harness.look();

    harness.tell("search start criterion=novelty");
    let look = harness.look();
    report.check("novelty without axes is refused rather than scoring everything zero",
        look.errors().len() == 1, look.complaint());
    harness.tell("search stop");
    let look = harness.look();
    report.check("stopping a search nobody started is an error", look.errors().len() == 1, look.complaint());

    // A gate tuned in the playground is the same gate the search runs under, or tuning one against a
    // live world teaches a frontend nothing about what a search will keep.
    harness.tell("gates connectivity=0:1");
    harness.look();
    harness.tell("search start criterion=structure generations=2 batch=6 capacity=6 timesteps=40 seed=5 from=world");
    let opened = Instant::now();
    let run = harness.until(|line| kind(line) == "search" && text(line, "state") == "done");
    let started = run.first("search");
    report.check("a search says what it resolved to before it runs",
        text(&started, "state") == "started" && field(&started, "plan").unwrap_or("[]") != "[]"
            && field(&started, "shape").is_some(),
        format!("plan {}", field(&started, "plan").unwrap_or("")));
    report.check("the gates a world was tuned under travel into the search it starts",
        field(&started, "gates").unwrap_or("[]").contains("connectivity"),
        format!("gates {}", field(&started, "gates").unwrap_or("")));
    report.check("a search told to refine the world starts from the genome on screen",
        number(&started, "refining") == 1.0, format!("refining {}", number(&started, "refining")));
    report.check("every generation reports as it lands", run.of("generation").len() == 3,
        format!("{} generations in {:?}", run.of("generation").len(), opened.elapsed()));
    let tally = run.first("generation");
    report.check("a generation says what happened to the candidates it judged",
        number(&tally, "died") + number(&tally, "gated") + number(&tally, "unscorable") <= number(&tally, "evaluated")
            && number(&tally, "evaluated") == 6.0,
        format!("{} judged, {} died", number(&tally, "evaluated"), number(&tally, "died")));
    let done = run.last("search");
    let specimens = run.of("specimen").len();
    report.check("the survivors come back best first, and the count agrees with them",
        specimens == number(&done, "kept") as usize && specimens > 0,
        format!("{specimens} kept"));
    let scores: Vec<f64> = run.of("specimen").iter().map(|line| number(line, "score")).collect();
    report.check("a survivor's score never rises down the list",
        scores.windows(2).all(|pair| pair[0] >= pair[1]), format!("{scores:?}"));

    // A survivor is only loadable into the shape it was drawn for, so both travel and both have to fit.
    let best = run.first("specimen");
    let genes: Vec<String> = list(&best, "genes").iter().map(|value| format!("{value}")).collect();
    let descriptor = list(&best, "descriptor");
    harness.tell(&format!("genome genes={}", genes.join(",")));
    let look = harness.look();
    report.check("a survivor loads back into the world it was searched in",
        look.errors().is_empty() && !genes.is_empty(), format!("{} genes", genes.len()));
    report.check("a survivor carries the descriptor it was scored on",
        !descriptor.is_empty() && descriptor.iter().all(|value| (0.0..=1.0).contains(value)),
        format!("{} slots", descriptor.len()));
    report.check("a finished search leaves the session idle again",
        text(&look.state(), "searching") == "false", format!("searching {}", text(&look.state(), "searching")));
    harness.close();

    // Two searches from one seed are the same search, or a survivor cannot be reproduced from what a
    // run wrote down.
    let best_of = |seed: usize| {
        let mut harness = Harness::open();
        harness.look();
        harness.tell("shape particles=80 anchors=2 shells=1 bumps=1");
        harness.look();
        harness.tell(&format!("search start criterion=structure generations=1 batch=4 capacity=4 timesteps=40 seed={seed}"));
        let run = harness.until(|line| kind(line) == "search" && text(line, "state") == "done");
        harness.close();
        run.last("generation")
    };
    let first = best_of(11);
    let again = best_of(11);
    report.check("the same seed searches the same way",
        field(&first, "genes") == field(&again, "genes") && !first.is_empty(),
        format!("best {} against {}", number(&first, "best"), number(&again, "best")));
}

/// Stopping is what makes a long search usable: a frontend that changed its mind gets what was found
/// rather than nothing, and the session comes back idle rather than half running.
pub fn stopping(report: &mut Report) {
    let mut harness = Harness::open();
    harness.look();
    harness.tell("shape particles=80 anchors=2 shells=1 bumps=1");
    harness.look();
    harness.tell("search start criterion=structure generations=200 batch=6 capacity=6 timesteps=40 seed=7");
    let opened = Instant::now();
    let opening = harness.until(|line| kind(line) == "generation");
    report.check("a long search reports its first generation without waiting for its last",
        !opening.of("generation").is_empty(), format!("first generation after {:?}", opened.elapsed()));

    harness.tell("search stop");
    let asked = Instant::now();
    let ending = harness.until(|line| kind(line) == "search" && text(line, "state") == "done");
    report.check("a stop is acknowledged rather than waited out in silence",
        ending.of("search").iter().any(|line| text(line, "state") == "stopping"),
        format!("{:?} to the last line", asked.elapsed()));
    let done = ending.last("search");
    report.check("a stopped search keeps what it had already found",
        number(&done, "kept") > 0.0 && ending.of("specimen").len() == number(&done, "kept") as usize,
        format!("{} kept after {:?}", number(&done, "kept"), asked.elapsed()));
    let generations = opening.of("generation").len() + ending.of("generation").len();
    report.check("a stopped search ends well short of the generations it was given",
        generations < 200, format!("{generations} generations of 200"));

    let look = harness.look();
    report.check("a stopped session is idle again",
        text(&look.state(), "searching") == "false", format!("searching {}", text(&look.state(), "searching")));
    harness.tell("search start criterion=structure generations=0 batch=2 capacity=2 timesteps=20");
    let again = harness.until(|line| kind(line) == "search" && text(line, "state") == "done");
    report.check("a session that stopped one search can run another",
        !again.of("generation").is_empty() && text(&again.last("search"), "state") == "done",
        format!("{} generations", again.of("generation").len()));
    harness.close();
}

/// A law that outreaches its own grid, or one too inert to reach at all, walks every pair instead of
/// the neighbors: the same picture at many times the cost. It says so rather than reading as a slow machine.
pub fn regimes(report: &mut Report) {
    let mut harness = Harness::open();
    let look = harness.look();
    report.check("an ordinary world reports that it is using its spatial index",
        text(&look.state(), "gridded") == "true", format!("gridded {}", text(&look.state(), "gridded")));

    // Every shell amplitude to zero is a legal genome that senses nothing, so nothing can reach anything.
    let amps: Vec<f64> = genes_of(&look.layout()).iter()
        .filter(|gene| text(gene, "label").starts_with("shell") && text(gene, "label").ends_with("amp"))
        .map(|gene| number(gene, "index")).collect();
    let edits: Vec<String> = amps.iter().map(|index| format!("{index}=0")).collect();
    harness.tell(&format!("gene {}", edits.join(" ")));
    let look = harness.look();
    report.check("an inert law says it has fallen back to every pair",
        text(&look.state(), "gridded") == "false" && !amps.is_empty(),
        format!("{} shell amplitudes zeroed, gridded {}", amps.len(), text(&look.state(), "gridded")));

    // An inert world is still a world: it keeps its tick, it keeps drawing, and it answers commands.
    harness.tell("step count=50");
    let look = harness.look();
    report.check("an inert world still steps and still draws",
        number(&look.state(), "tick") == 50.0 && !look.of("frame").is_empty(),
        format!("tick {}, {} frames", number(&look.state(), "tick"), look.of("frame").len()));
    harness.close();
}

/// Ending. A frontend that closes the pipe, and a script that runs out of lines, both have to leave the
/// process gone rather than stepping a world nobody is watching.
pub fn leaving(report: &mut Report) {
    let mut harness = Harness::open();
    harness.look();
    let quit = harness.close();
    report.check("quit ends the process", quit, format!("exited {quit}"));

    let mut child = Command::new(env!("CARGO_BIN_EXE_axiom"))
        .stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().expect("the harness binary runs");
    {
        let mut input = child.stdin.take().expect("stdin is piped");
        let _ = writeln!(input, "run");
        let _ = writeln!(input, "step count=50");
    } // dropping stdin is a frontend going away mid-run
    let mut gone = false;
    for _ in 0..200 {
        if let Ok(Some(_)) = child.try_wait() { gone = true; break; }
        std::thread::sleep(Duration::from_millis(50));
    }
    if !gone { let _ = child.kill(); }
    report.check("a closed pipe ends a running world rather than stepping it forever", gone,
        format!("exited {gone}"));
}
