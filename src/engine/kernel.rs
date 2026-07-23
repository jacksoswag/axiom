//! Particle displacement in periodic bounds (3D torus), the radial potential particles use to
//! sense each other K(d), and the growth response to what they sense G(u).

/// One Gaussian bump, shared by K (a bump over distance, i.e. a spatial shell) and G
/// (a bump over sensed density, i.e. a preferred crowding level).
pub struct Shell {
    pub amp: f32,
    pub peak: f32, // offset of center
    pub width: f32,
}

/// Calculates Gaussian strength and its derivative (slope) from a mixture of bumps, or shells reduced to 1D.
pub fn bump_and_slope(x: f32, bumps: &[Shell]) -> (f32, f32) {
    let (mut value, mut slope) = (0.0, 0.0);
    for bump in bumps { // per-bump calculation
        let term = // this bump's contribution at particle's displacement
            (-((x - bump.peak).powi(2)     // squared gap between displacement and the shell's peak
            / (2.0 * bump.width.powi(2)))) // scale gap penalty by shell's width (wider -> lower penalty)
            .exp() * bump.amp;             // e^-x makes gap decay from 1->0, then multiply ALL by shell's amplitude
        value += term; // sum into K(d)
        slope += -(x - bump.peak) / bump.width.powi(2) * term; // derivative reuses term, uses chain rule
    }
    (value, slope)
}

/// How particles react to their own crowding. Uses Gaussian math above, normalizes to [-1, 1]: positive -> grow to this density
/// Only the slope drives motion, the squash keeps the value readable on its own.
pub fn growth(u: f32, bumps: &[Shell]) -> (f32, f32) {
    let (sum, sum_prime) = bump_and_slope(u, bumps);
    (2.0 * sum - 1.0, 2.0 * sum_prime) // normalize to [-1, 1]
}

/// Returns softened displacement between 2 particles, avoiding blowup as displacement approaches 0
/// Callers that only want the scalar never pay for the per-axis buffer `displacement_squared` fills.
pub fn displacement(pos1: &[f32], pos2: &[f32], box_len: f32, inverse_box_len: f32, s_squared: f32) -> f32 {
    let mut d_squared = 0.0;
    for k in 0..pos1.len() {
        let mut d = pos1[k] - pos2[k];
        d -= box_len * (d * inverse_box_len).round();
        d_squared += d.powi(2);
    }
    (d_squared + s_squared).sqrt() // soften last, matching the order callers already depend on
}

/// Squared-displacement triaged to separate method to optimize max-radius check (checks d^2 < r^2)
pub fn displacement_squared(pos1: &[f32], pos2: &[f32], box_len: f32, inverse_box_len: f32, out: &mut [f32]) -> f32 {
    let mut d_squared = 0.0;
    for k in 0..pos1.len() { // per-axis calculaton:
        let mut d = pos1[k] - pos2[k]; // single-axis distance
        d -= box_len * (d * inverse_box_len).round(); // optimizes for periodic bounds IF d / box_len > 0.5
        out[k] = d; // store per-axis differences for the caller (force direction)
        d_squared += d.powi(2); // add to squared displacement across all axes
    }
    d_squared
}

/// The displacement past which a shell mixture adds nothing worth computing. Keeps the loop from checking every other particle
/// A term with amplitude ~= 0 is skipped, so one inert term in the mixture can't inflate this radius and erase the saving.
pub fn displacement_cutoff(shells: &[Shell]) -> f32 {
    shells.iter().filter(|s| s.amp.abs() > 1e-6).map(|s| s.peak + CUTOFF_SIGMA * s.width.abs()).fold(0.0, f32::max)
}
/// Standard deviations from peak where clamped to 0. 3.5 is 0.2% of peak amplitude.
pub const CUTOFF_SIGMA: f32 = 3.5; 
