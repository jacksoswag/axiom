//! Window entry point. Everything lives in `axiom::viewer`, behind the `viewer` feature,
//! so the headless half never pulls eframe in.

fn main() -> eframe::Result<()> {
    axiom::viewer::run()
}
