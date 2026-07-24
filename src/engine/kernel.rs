//! Particle distance in periodic bounds (3D torus), the radial potential particles use to
//! sense each other K(d), and the growth response to what they sense G(u).

use crate::engine::substrate::Substrate;

/// One Gaussian. K's mixtures are shells (over distance), G's are bumps (over sensed density)
pub struct Shell {
    pub amp: f32,
    pub peak: f32, // offset from center
    pub width: f32,
}

/// Calculates Gaussian strength and its derivative (slope) from a mixture of shells
pub fn strength_and_slope(x: f32, shells: &[Shell]) -> (f32, f32) {
    let (mut value, mut slope) = (0.0, 0.0);
    for shell in shells { // per-shell calculation
        let term = // this shell's contribution at particle's distance
            (-((x - shell.peak).powi(2)     // squared gap between distance and the shell's peak
            / (2.0 * shell.width.powi(2)))) // scale gap penalty by shell's width (wider -> lower penalty)
            .exp() * shell.amp;             // e^-x makes gap decay from 1->0, then multiply ALL by shell's amplitude
        value += term; // sum into K(d)
        slope += -(x - shell.peak) / shell.width.powi(2) * term; // derivative reuses term, uses chain rule
    }
    (value, slope)
}

/// Squared, softened distance between 2 particles. Fills 'out' unless empty, so some
/// callers never pay for a buffer.
pub fn distance_sq(pos1: &[f32], pos2: &[f32], sub: &Substrate, out: &mut [f32]) -> f32 {
    let inverse_box_len = 1.0 / sub.box_len;
    let want_buffer = !out.is_empty();
    let mut d_squared = 0.0;
    for k in 0..pos1.len() { // per-axis calculaton:
        let mut d = pos1[k] - pos2[k]; // single-axis distance
        d -= sub.box_len * (d * inverse_box_len).round(); // optimizes for periodic bounds IF d / box_len > 0.5
        if want_buffer { out[k] = d; } // store per-axis differences for the caller (force direction)
        d_squared += d.powi(2); // add to squared distance across all axes
    }
    d_squared + sub.softening.powi(2) // soften last, matching the order callers already depend on
}

/// The distance past which a shell mixture adds nothing worth computing. Keeps the loop from checking every other particle
/// A term with amplitude ~= 0 is skipped, so one inert term in the mixture can't inflate this radius and erase the saving.
pub fn distance_cutoff(shells: &[Shell]) -> f32 {
    shells.iter().filter(|s| s.amp.abs() > 1e-6).map(|s| s.peak + CUTOFF_SIGMA * s.width.abs()).fold(0.0, f32::max)
}
/// Standard deviations from peak where clamped to 0. 3.5 is 0.2% of peak amplitude.
pub const CUTOFF_SIGMA: f32 = 3.5; 
