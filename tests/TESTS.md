# The test suite

Three suites, one folder each, one flag each. This layout is fixed. Read the rule at the bottom
before changing anything in here.

```
tests/
  behavior/     the harness end to end, over its own protocol and the relay
  smoke/        the fast in-process suite: engine, tuner, harness
  performance/  what the machine costs, and that a replay is exact
  reports/      DATE_smk.md, DATE_behave.md, DATE_perf.md
  run.sh        --smk, --behave, --perf
  TESTS.md      this file
```

## Running

```
tests/run.sh --smk           seconds
tests/run.sh --behave        about half a minute, spawns the binary and python3
tests/run.sh --perf          minutes, release build, wants the machine to itself
tests/run.sh --smk --behave  any combination, run in the order written above
```

Each flag runs its suite, writes `tests/reports/DATE_{smk,behave,perf}.md`, and prints it back with
the tables lined up. The script exits non-zero if anything broke. Raw cargo output for the last run
of each suite sits in `tests/reports/.logs/`.

The suites are ordinary cargo targets, so `cargo test --test smoke` and friends work too. Only the
script names the reports, so a report written any other way is a report nobody will find.

## What goes where

**smoke** is one promise per case, checked in process against the library. No child processes, no
sleeps, no report of its own: cargo says which case failed and the script turns that into the
report. A case belongs here if it can name what it is checking and fail for that reason alone.

- `smoke/fixture.rs` is the shared setup: one small shape, a genome drawn inside it, the uniform
  cloud and law that genome makes, and the hunt for a genome still measurable after a short run.
- `smoke/engine.rs`, `smoke/tuner.rs`, `smoke/harness.rs` follow the crate's own three halves.

**behavior** drives the real binary the way a frontend does: command lines in, JSON event lines out,
and the Python relay in front of it over a socket. Every case is a promise the protocol makes to a
page that cannot see inside. It runs as one test on purpose, because the report is the point: an
area that panics costs its own row and the rest still runs.

- `behavior/main.rs` is the driver and the report.
- `behavior/wire.rs` reads a field off an event line and drives a running harness.
- `behavior/areas.rs` is the promises, grouped by area.
- `behavior/relay.rs` is what Python owns: the page, the paths, the event stream, the backlog.

**performance** is timings and determinism, `#[ignore]` so an ordinary `cargo test` never pays for
it. Every row is a wall-clock reading that assumes it has the machine, which is why it is one test
and why the report records the machine and its load.

- `performance/main.rs` times a case and writes the report.
- `performance/cases.rs` is what gets measured.

**reports** holds the dated output. One file per suite per day; a second run the same day overwrites
that day's file.

## The rule

Do not change this structure. Three folders, those three names, one report per suite named
`DATE_{smk,behave,perf}.md`, one script with those three flags. Add cases inside the suite they
belong to, add a module inside a suite if a file gets unwieldy, and leave the shape alone. In
particular: no new top-level test target, no fourth suite, no per-case report files, no renaming the
flags or the report suffixes, and no `#[cfg(test)]` module anywhere under `src/`.

If a change genuinely needs the shape to move, say so and ask first.
