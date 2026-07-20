//! Spacetime-loaf time mode (§4) — the novel configurable.
//!
//! Instead of stepping forward, treat the whole `space × time` volume as one
//! field and relax it to global consistency. Information flows backward and
//! sideways in time, so interior states can be inferred from boundary states.
//!
//! Minimize `E = Σ_t ‖V[t+1] − F(V[t])‖²` over the volume with the endpoint
//! slices fixed. `F` is one reaction-diffusion step: `F(u) = u + dt·D·∇²u +
//! dt·r·u(1−u)` (Fisher-KPP). With `r = 0` it is linear diffusion and `E` is
//! convex, so relaxation recovers the true interior. With `r > 0` it is
//! nonlinear and `E` is non-convex: relaxation still descends, but convergence
//! is no longer guaranteed, which is the honest cost of the mode (§9).
//!
//! The gradient's linearized term uses the analytic vector-Jacobian product of
//! `F` at the current slice, `J·δ = δ + dt·D·∇²δ + dt·r·(1−2u)·δ`. Since `J` is
//! symmetric, the same operator serves as `Jᵀ`.

pub struct Loaf {
    pub grid: usize,
    pub t: usize,
    pub d: f32,
    pub dt: f32,
    pub react: f32,
    /// `t × grid × grid`, row-major per slice.
    pub vol: Vec<f32>,
}

#[inline]
fn laplacian(g: usize, src: &[f32], i: usize) -> f32 {
    let (y, x) = (i / g, i % g);
    let yl = if y == 0 { g - 1 } else { y - 1 };
    let yr = if y + 1 == g { 0 } else { y + 1 };
    let xl = if x == 0 { g - 1 } else { x - 1 };
    let xr = if x + 1 == g { 0 } else { x + 1 };
    src[yl * g + x] + src[yr * g + x] + src[y * g + xl] + src[y * g + xr] - 4.0 * src[i]
}

impl Loaf {
    fn plane(&self) -> usize {
        self.grid * self.grid
    }

    /// Forward step `F(u)`.
    fn forward(&self, u: &[f32], out: &mut [f32]) {
        let (g, coef, dtr) = (self.grid, self.dt * self.d, self.dt * self.react);
        for i in 0..u.len() {
            let lap = laplacian(g, u, i);
            out[i] = u[i] + coef * lap + dtr * u[i] * (1.0 - u[i]);
        }
    }

    /// Vector-Jacobian product `J_F(point) · r` (equals `Jᵀ·r`, J symmetric).
    fn jvp(&self, point: &[f32], r: &[f32], out: &mut [f32]) {
        let (g, coef, dtr) = (self.grid, self.dt * self.d, self.dt * self.react);
        for i in 0..r.len() {
            let lap = laplacian(g, r, i);
            out[i] = r[i] + coef * lap + dtr * (1.0 - 2.0 * point[i]) * r[i];
        }
    }

    fn slice(&self, t: usize) -> &[f32] {
        let p = self.plane();
        &self.vol[t * p..(t + 1) * p]
    }

    /// Build a ground-truth volume by forward-running from `init`.
    pub fn ground_truth(grid: usize, t: usize, d: f32, dt: f32, react: f32, init: Vec<f32>) -> Loaf {
        let plane = grid * grid;
        let mut vol = vec![0.0f32; t * plane];
        vol[..plane].copy_from_slice(&init);
        let loaf = Loaf { grid, t, d, dt, react, vol: Vec::new() };
        let mut v = vol;
        for k in 0..t - 1 {
            let (a, b) = v.split_at_mut((k + 1) * plane);
            loaf.forward(&a[k * plane..], &mut b[..plane]);
        }
        Loaf { vol: v, ..loaf }
    }

    /// Overwrite the interior with a linear interpolation of the endpoints.
    pub fn occlude_interior(&mut self) {
        let p = self.plane();
        let (first, last) = (self.vol[..p].to_vec(), self.vol[(self.t - 1) * p..].to_vec());
        for k in 1..self.t - 1 {
            let a = k as f32 / (self.t - 1) as f32;
            for i in 0..p {
                self.vol[k * p + i] = (1.0 - a) * first[i] + a * last[i];
            }
        }
    }

    /// Relax the interior toward global consistency (endpoints fixed). Returns the
    /// energy (residual) history.
    pub fn relax(&mut self, iters: usize, lr: f32) -> Vec<f32> {
        let p = self.plane();
        let mut fv = vec![0.0f32; self.t * p];
        let mut resid = vec![0.0f32; p];
        let mut fresid = vec![0.0f32; p];
        let mut history = Vec::with_capacity(iters);
        for _ in 0..iters {
            for t in 0..self.t {
                let src = self.slice(t).to_vec();
                self.forward(&src, &mut fv[t * p..(t + 1) * p]);
            }
            let mut energy = 0.0f64;
            for t in 0..self.t - 1 {
                for i in 0..p {
                    let r = self.vol[(t + 1) * p + i] - fv[t * p + i];
                    energy += (r * r) as f64;
                }
            }
            history.push((energy / (self.t * p) as f64) as f32);
            for t in 1..self.t - 1 {
                for i in 0..p {
                    resid[i] = self.vol[t * p + i] - fv[(t - 1) * p + i];
                }
                let next_resid: Vec<f32> = (0..p).map(|i| self.vol[(t + 1) * p + i] - fv[t * p + i]).collect();
                let point = self.slice(t).to_vec();
                self.jvp(&point, &next_resid, &mut fresid);
                for i in 0..p {
                    let grad = 2.0 * resid[i] - 2.0 * fresid[i];
                    self.vol[t * p + i] -= lr * grad;
                }
            }
        }
        history
    }

    /// Mean per-cell error of the interior against a reference volume.
    pub fn interior_error(&self, reference: &Loaf) -> f32 {
        let p = self.plane();
        let mut sum = 0.0f64;
        let mut cnt = 0usize;
        for t in 1..self.t - 1 {
            for i in 0..p {
                let d = self.vol[t * p + i] - reference.vol[t * p + i];
                sum += (d * d) as f64;
                cnt += 1;
            }
        }
        (sum / cnt.max(1) as f64).sqrt() as f32
    }

    /// An (x, t) slice through the volume at row `y`.
    pub fn xt_slice(&self, y: usize) -> Vec<f32> {
        let g = self.grid;
        let mut out = vec![0.0f32; self.t * g];
        for t in 0..self.t {
            for x in 0..g {
                out[t * g + x] = self.vol[t * self.plane() + y * g + x];
            }
        }
        out
    }
}
