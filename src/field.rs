//! `Field` — the state substrate for grid rules.
//!
//! Multi-channel 2-D state stored as a single flat `Vec<f32>` with layout
//! `[channel][row][col]`. Flat storage keeps the convolution hot loop trivial to
//! parallelise with rayon and lets us drop a heavier array dependency (§1's
//! "specialize the hot inner loops"). Classic scalar Lenia is just `C = 1`.

#[derive(Debug, Clone)]
pub struct Field {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Field {
    pub fn zeros(c: usize, h: usize, w: usize) -> Self {
        Field { c, h, w, data: vec![0.0; c * h * w] }
    }

    #[inline]
    pub fn plane(&self) -> usize {
        self.h * self.w
    }

    #[inline]
    pub fn idx(&self, ch: usize, y: usize, x: usize) -> usize {
        (ch * self.h + y) * self.w + x
    }

    #[inline]
    pub fn channel(&self, ch: usize) -> &[f32] {
        let p = self.plane();
        &self.data[ch * p..(ch + 1) * p]
    }

    #[inline]
    pub fn channel_mut(&mut self, ch: usize) -> &mut [f32] {
        let p = self.plane();
        &mut self.data[ch * p..(ch + 1) * p]
    }

    #[inline]
    pub fn get(&self, ch: usize, y: usize, x: usize) -> f32 {
        self.data[self.idx(ch, y, x)]
    }

    #[inline]
    pub fn set(&mut self, ch: usize, y: usize, x: usize, v: f32) {
        let i = self.idx(ch, y, x);
        self.data[i] = v;
    }

    /// Total mass of a channel (sum of activations).
    pub fn mass(&self, ch: usize) -> f64 {
        self.channel(ch).iter().map(|&v| v as f64).sum()
    }
}
