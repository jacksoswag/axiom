//! Process entry point. Everything lives in `axiom::harness`, which reads commands on stdin and
//! writes events on stdout, so a frontend is whatever chooses to speak to it.

fn main() { axiom::harness::run() }
