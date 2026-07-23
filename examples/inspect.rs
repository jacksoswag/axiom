//! Print reproducible archive recipes without a projection-only diagnostic.
//!
//! `cargo run --release --no-default-features --example inspect -- archive.txt 0`

use axiom::tuner::archive::Archive;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "archive.txt".into());
    let rank = std::env::args()
        .nth(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(0usize);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) => return eprintln!("could not read {path}: {error}"),
    };
    let (archive, tuning) = match Archive::from_text(&text) {
        Ok(loaded) => loaded,
        Err(error) => return eprintln!("could not parse {path}: {error}"),
    };
    let mut entries = archive.entries().to_vec();
    entries.sort_by(|left, right| right.current_novelty.total_cmp(&left.current_novelty));
    let Some(entry) = entries.get(rank) else {
        return eprintln!("archive has {} entries", entries.len());
    };
    let (params, interactions) = tuning.world.resolve(&entry.genome);
    println!("entry {rank} of {}", entries.len());
    println!(
        "lineage {} parent {:?} born generation {} via {:?}",
        entry.lineage_id, entry.parent_lineage_id, entry.birth_generation, entry.parent_source
    );
    println!(
        "admission novelty {:.5} current novelty {:.5}",
        entry.admission_novelty, entry.current_novelty
    );
    println!(
        "3-D particles {} coordination {:.3} anchors {} trait logits {:?}",
        params.particles, params.coordination, params.anchors, params.trait_logits
    );
    println!(
        "structure {:.4} mobility {:.5} turnover {:.4} robustness {:.2}",
        entry.metrics.structure,
        entry.metrics.mobility,
        entry.metrics.turnover,
        entry.metrics.robustness
    );
    println!(
        "heterogeneity {:?} dense {:?} void {:?}",
        entry.metrics.heterogeneity,
        entry.metrics.connectivity.dense,
        entry.metrics.connectivity.void
    );
    println!("{} pair-specific interactions", interactions.len());
}
