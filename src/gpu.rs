//! GPU compute backend (§5.2) — the performance floor.
//!
//! A headless `wgpu` compute pipeline for single-kernel Lenia: one thread per
//! cell, ping-pong storage buffers (never read the buffer being written). Runs
//! with no window/surface, so it is verifiable in CI by diffing one GPU step
//! against the CPU `Rule` oracle. The CPU path stays the reference; this is the
//! speed path that unlocks large fields.
//!
//! ponytail: single-kernel/single-channel Lenia on GPU (covers Orbium + soup, the
//! hot cases). Multi-kernel/multi-channel and the graph path stay on CPU; porting
//! them is the documented next step.

use crate::config::LeniaConfig;
use crate::kernel::{CoreKind, Kernel, KernelParams};
use anyhow::{anyhow, bail, Result};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    w: u32,
    h: u32,
    radius: u32,
    torus: u32,
    dt: f32,
    mu: f32,
    sigma: f32,
    clamp_lo: f32,
    clamp_hi: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

/// Tiled convolution: each 16×16 workgroup cooperatively loads its cell block plus
/// an `R`-wide halo into workgroup memory, then every thread convolves out of that
/// shared tile. This collapses the redundant global reads a direct gather makes.
/// The fixed `48×48` tile supports radius up to 16 (covers every preset).
const SHADER: &str = r#"
struct Params {
    w: u32, h: u32, radius: u32, torus: u32,
    dt: f32, mu: f32, sigma: f32, clamp_lo: f32, clamp_hi: f32,
    pad0: f32, pad1: f32, pad2: f32,
};
@group(0) @binding(0) var<storage, read>       src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform>             p:   Params;
@group(0) @binding(3) var<storage, read>       kern: array<f32>;

var<workgroup> tile: array<f32, 2304>; // 48*48

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let r = i32(p.radius);
    let hh = i32(p.h);
    let ww = i32(p.w);
    let loadw = 16 + 2 * r;
    let wy0 = i32(wid.y) * 16;
    let wx0 = i32(wid.x) * 16;
    let tid = i32(lid.y) * 16 + i32(lid.x);
    // Cooperative halo load.
    var i = tid;
    loop {
        if (i >= loadw * loadw) { break; }
        let ty = i / loadw;
        let tx = i % loadw;
        var gy = wy0 - r + ty;
        var gx = wx0 - r + tx;
        var val = 0.0;
        if (p.torus == 1u) {
            gy = ((gy % hh) + hh) % hh;
            gx = ((gx % ww) + ww) % ww;
            val = src[u32(gy) * p.w + u32(gx)];
        } else if (gy >= 0 && gy < hh && gx >= 0 && gx < ww) {
            val = src[u32(gy) * p.w + u32(gx)];
        }
        tile[ty * 48 + tx] = val;
        i = i + 256;
    }
    workgroupBarrier();
    if (gid.x >= p.w || gid.y >= p.h) { return; }
    let ks = 2 * r + 1;
    let oy = i32(lid.y);
    let ox = i32(lid.x);
    var u = 0.0;
    for (var dy = -r; dy <= r; dy = dy + 1) {
        for (var dx = -r; dx <= r; dx = dx + 1) {
            let kw = kern[u32((dy + r) * ks + (dx + r))];
            u = u + tile[(oy + r + dy) * 48 + (ox + r + dx)] * kw;
        }
    }
    let g = 2.0 * exp(-0.5 * pow((u - p.mu) / p.sigma, 2.0)) - 1.0;
    let idx = gid.y * p.w + gid.x;
    dst[idx] = clamp(src[idx] + p.dt * g, p.clamp_lo, p.clamp_hi);
}
"#;

pub struct GpuLenia {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    buf_a: wgpu::Buffer,
    buf_b: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    kern_buf: wgpu::Buffer,
    readback: wgpu::Buffer,
    plane: usize,
    w: usize,
    h: usize,
    /// Which buffer holds the current state (true = `buf_a`), for persistent stepping.
    cur_a: std::cell::Cell<bool>,
    pub adapter_name: String,
}

impl GpuLenia {
    pub fn new(
        h: usize,
        w: usize,
        kernel: &Kernel,
        dt: f32,
        mu: f32,
        sigma: f32,
        clamp: (f32, f32),
        torus: bool,
    ) -> Result<GpuLenia> {
        if kernel.radius > 16 {
            bail!("GPU tiled shader supports radius <= 16, got {}", kernel.radius);
        }
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| anyhow!("no GPU adapter available"))?;
        let adapter_name = adapter.get_info().name;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("axiom-gpu"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::Performance,
        }, None))?;

        let plane = h * w;
        let bytes = (plane * std::mem::size_of::<f32>()) as u64;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lenia"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("lenia-pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let layout = pipeline.get_bind_group_layout(0);

        let mk = |label, usage| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: bytes, usage, mapped_at_creation: false })
        };
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let buf_a = mk("a", storage);
        let buf_b = mk("b", storage);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = Params {
            w: w as u32, h: h as u32, radius: kernel.radius as u32, torus: torus as u32,
            dt, mu, sigma, clamp_lo: clamp.0, clamp_hi: clamp.1, _pad0: 0.0, _pad1: 0.0, _pad2: 0.0,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let kern_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("kernel"),
            contents: bytemuck::cast_slice(&kernel.weights),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        Ok(GpuLenia {
            device, queue, pipeline, layout, buf_a, buf_b, params_buf, kern_buf, readback,
            plane, w, h, cur_a: std::cell::Cell::new(true), adapter_name,
        })
    }

    /// Upload an initial field; it becomes the current state.
    pub fn upload(&self, init: &[f32]) {
        assert_eq!(init.len(), self.plane);
        self.queue.write_buffer(&self.buf_a, 0, bytemuck::cast_slice(init));
        self.cur_a.set(true);
    }

    /// Advance the on-device state by `n` steps (state persists across calls).
    pub fn advance(&self, n: usize) {
        let (gx, gy) = (self.w.div_ceil(16) as u32, self.h.div_ceil(16) as u32);
        let bg_ab = self.bind(&self.buf_a, &self.buf_b);
        let bg_ba = self.bind(&self.buf_b, &self.buf_a);
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut a = self.cur_a.get();
        for _ in 0..n {
            let bg = if a { &bg_ab } else { &bg_ba };
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
            drop(pass);
            a = !a;
        }
        self.queue.submit(Some(enc.finish()));
        self.cur_a.set(a);
    }

    /// Read the current on-device state back to the CPU.
    pub fn read(&self) -> Vec<f32> {
        let cur = if self.cur_a.get() { &self.buf_a } else { &self.buf_b };
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(cur, 0, &self.readback, 0, (self.plane * 4) as u64);
        self.queue.submit(Some(enc.finish()));
        let slice = self.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.readback.unmap();
        out
    }

    /// Build the GPU pipeline from a single-kernel, gauss-growth Lenia config
    /// (the case the shader implements — Orbium, soup).
    pub fn from_lenia(h: usize, w: usize, cfg: &LeniaConfig, torus: bool) -> Result<GpuLenia> {
        if cfg.kernels.len() != 1 {
            bail!("GPU path supports single-kernel Lenia; got {} kernels", cfg.kernels.len());
        }
        let k = &cfg.kernels[0];
        if k.growth.kind != "gauss" {
            bail!("GPU path implements gaussian growth; got '{}'", k.growth.kind);
        }
        let kernel = Kernel::build(&KernelParams {
            radius: k.radius,
            core: CoreKind::parse(&k.core),
            beta: &k.beta,
            core_mu: k.core_mu,
            core_sigma: k.core_sigma,
        });
        GpuLenia::new(h, w, &kernel, cfg.dt, k.growth.mu, k.growth.sigma, (cfg.clamp_lo, cfg.clamp_hi), torus)
    }

    fn bind(&self, src: &wgpu::Buffer, dst: &wgpu::Buffer) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.kern_buf.as_entire_binding() },
            ],
        })
    }

    /// Upload `init`, run `steps` dispatches (one encoder, one submit; wgpu
    /// inserts the storage-buffer barriers between passes), read the result back.
    pub fn run(&self, init: &[f32], steps: usize) -> Vec<f32> {
        self.upload(init);
        self.advance(steps.max(1));
        self.read()
    }
}
