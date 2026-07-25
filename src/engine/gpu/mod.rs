//! The pair loop on the graphics card, and the decision of whether to use it. The physics lives in
//! step.wgsl beside this file; everything here is getting one substrate's worth of state onto the card
//! and one tick's worth of positions back. The grid is not built here: rebuild_grid already ran on the
//! CPU and its cell map rides along, which costs a fraction of a tick and saves a prefix sum over a
//! few million cells.
//!
//! Two things this deliberately does not do. It does not step a whole generation in one dispatch: at
//! the sizes worth a card at all, one swarm already fills it, and batching would mean every candidate
//! sharing a box size and a moment of death. And it is not bit-identical to the CPU stepper, because
//! exp() here is the card's approximation rather than the one libm ships. A run reproduces against
//! itself and against another run on the same backend, never across the two.

use std::sync::{Mutex, OnceLock};

use crate::engine::matrix::Matrix;
use crate::engine::substrate::Substrate;

/// Particles below which the card is not worth the trip. Two transfers and a dispatch cost tens of
/// microseconds whatever the size, and the CPU steps a small swarm in about that, so the crossover
/// sits where a tick starts costing more than the round trip. Measured on one machine, so it reads
/// from the environment too: AXIOM_GPU=0 turns the card off outright, which is how a comparison run
/// reaches the CPU path without a rebuild, and any number sets the floor directly.
const WORTH_IT: usize = 16_384;

/// Whether this tick should go to the card. Called after rebuild_grid, so the index it asks about is
/// the one that just got built rather than a guess at what it would be.
pub fn worth_it(substrate: &Substrate) -> bool {
    substrate.traits.len() >= threshold() && substrate.dimensions == 3
        && substrate.grid_shape().0 != 0 && open().is_some()
}
fn threshold() -> usize {
    static FLOOR: OnceLock<usize> = OnceLock::new();
    *FLOOR.get_or_init(|| match std::env::var("AXIOM_GPU").ok().as_deref() {
        Some("0") | Some("off") => usize::MAX, // never, which is what a comparison run wants
        Some(text) => text.parse().unwrap_or(WORTH_IT),
        None => WORTH_IT,
    })
}

/// One tick for the whole swarm. The caller has already rebuilt the grid, so this is upload, dispatch,
/// download: positions in, positions out, wrapped onto the torus by the shader that moved them.
pub fn step(substrate: &mut Substrate, matrix: &Matrix, dt: f32) {
    let Some(card) = open() else { return };
    let mut work = card.work.lock().unwrap_or_else(|held| held.into_inner());
    card.advance(&mut work, substrate, matrix, dt);
}

/// The one device this process opens, or None where no adapter answered. A card is a process-wide
/// resource and every sim in a batch shares it, so it opens once and is never replaced.
fn open() -> Option<&'static Card> {
    static CARD: OnceLock<Option<Card>> = OnceLock::new();
    CARD.get_or_init(Card::open).as_ref()
}

/// Which slot of the working set each buffer sits in. The shader's binding numbers are these plus one,
/// since binding zero is the parameter block.
const POSITIONS: usize = 0;
const MEMBERSHIPS: usize = 1;
const CELL_MAP: usize = 2;
const SORTED: usize = 3;
const PAIRS: usize = 4;
const SHELLS: usize = 5;
const BUMPS: usize = 6;
const NEXT: usize = 7;

struct Card {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    work: Mutex<Work>, // one working set: the card is one machine and a batch queues for it anyway
}

/// The buffers, grown to whatever the largest caller needed and never shrunk, since a campaign runs
/// one shape for its whole length. memberships carry the id of the population that filled them,
/// because they are a pure function of traits and traits do not move once seeded.
#[derive(Default)]
struct Work {
    buffers: [Option<wgpu::Buffer>; 8],
    sized: [usize; 8], // bytes each was last given, so growing is one comparison
    readback: Option<wgpu::Buffer>,
    params: Option<wgpu::Buffer>,
    seeded: u64, // whose memberships are on the card, 0 for nobody's
    staging: Vec<f32>, // memberships packed for upload, refilled only when the population changes
}

impl Card {
    fn open() -> Option<Card> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })).ok()?;
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("axiom"),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })).ok()?;
        // A shader that will not compile is a bug in this repo rather than a property of the machine,
        // so it takes the process down instead of quietly running every campaign on the CPU.
        device.on_uncaptured_error(std::sync::Arc::new(
            |problem| panic!("the card refused the step shader: {problem}")));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("step"),
            source: wgpu::ShaderSource::Wgsl(include_str!("step.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("step"), entries: &bindings() });
        let plumbing = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("step"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("step"),
            layout: Some(&plumbing),
            module: &module,
            entry_point: Some("advance"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Some(Card { device, queue, pipeline, layout, work: Mutex::new(Work::default()) })
    }

    fn advance(&self, work: &mut Work, substrate: &mut Substrate, matrix: &Matrix, dt: f32) {
        let count = substrate.traits.len();
        let (grid_len, cell_len, cells) = substrate.grid_shape();
        // Every pair's law as one four-wide row, and the two mixtures as they already sit in memory.
        // All three are a few hundred bytes at any legal shape, so they cross every tick rather than
        // being tracked for staleness.
        let law: Vec<f32> = matrix.interactions.iter()
            .flat_map(|pair| [pair.weight, pair.norm, pair.reach_sq, 0.0]).collect();
        let shells: Vec<f32> = matrix.shells.iter().flat_map(|shell| shell.packed()).collect();
        let bumps: Vec<f32> = matrix.bumps.iter().flat_map(|bump| bump.packed()).collect();
        let params: [u32; 12] = [
            count as u32, matrix.anchor_count as u32, matrix.shell_count as u32, matrix.bump_count as u32,
            grid_len as u32, 0, 0, 0,
            cell_len.to_bits(), substrate.box_len.to_bits(), substrate.softening_sq.to_bits(), dt.to_bits()];

        let held = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        self.fit(work, POSITIONS, count * 3 * 4, held);
        self.fit(work, MEMBERSHIPS, count * 4 * 4, held);
        self.fit(work, CELL_MAP, (cells + 1) * 4, held);
        self.fit(work, SORTED, count * 4, held);
        self.fit(work, PAIRS, law.len() * 4, held);
        self.fit(work, SHELLS, shells.len() * 4, held);
        self.fit(work, BUMPS, bumps.len() * 4, held);
        self.fit(work, NEXT, count * 3 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        if work.readback.is_none() {
            work.readback = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"), size: work.sized[NEXT] as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false }));
        }
        if work.params.is_none() {
            work.params = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("params"), size: 48,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false }));
        }

        self.write(work, POSITIONS, bytemuck::cast_slice(&substrate.positions));
        self.write(work, CELL_MAP, bytemuck::cast_slice(substrate.grid_map()));
        self.write(work, SORTED, bytemuck::cast_slice(substrate.grid_sorted()));
        self.write(work, PAIRS, bytemuck::cast_slice(&law));
        self.write(work, SHELLS, bytemuck::cast_slice(&shells));
        self.write(work, BUMPS, bytemuck::cast_slice(&bumps));
        self.queue.write_buffer(work.params.as_ref().unwrap(), 0, bytemuck::cast_slice(&params));
        // Memberships are a pure function of traits, and traits settle at seeding, so they only cross
        // when the population on the card is a different one from the population in hand.
        if work.seeded != substrate.id {
            work.staging.clear();
            work.staging.extend(substrate.memberships.iter().flat_map(|held| {
                let [(lower, low_weight), (upper, high_weight)] = held.entries;
                [lower as f32, low_weight, upper as f32, high_weight]
            }));
            let packed = std::mem::take(&mut work.staging); // out of the way of the borrow below
            self.write(work, MEMBERSHIPS, bytemuck::cast_slice(&packed));
            work.staging = packed;
            work.seeded = substrate.id;
        }

        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("step"), layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: work.params.as_ref().unwrap().as_entire_binding() },
                slot(work, POSITIONS), slot(work, MEMBERSHIPS), slot(work, CELL_MAP), slot(work, SORTED),
                slot(work, PAIRS), slot(work, SHELLS), slot(work, BUMPS), slot(work, NEXT)],
        });
        let mut orders = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("step") });
        {
            let mut pass = orders.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("step"), timestamp_writes: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(count.div_ceil(64) as u32, 1, 1);
        }
        let span = (count * 3 * 4) as u64;
        let readback = work.readback.as_ref().unwrap();
        orders.copy_buffer_to_buffer(work.buffers[NEXT].as_ref().unwrap(), 0, readback, 0, span);
        self.queue.submit([orders.finish()]);

        let slice = readback.slice(..span);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let landed = slice.get_mapped_range().expect("the card mapped a buffer it had just finished with");
        substrate.positions.copy_from_slice(bytemuck::cast_slice(&landed));
        drop(landed);
        readback.unmap();
    }

    /// Give a slot at least this many bytes, remaking it only when it does not already have them. A
    /// campaign runs one shape for its whole length, so this fires on the opening tick and then never.
    fn fit(&self, work: &mut Work, slot: usize, bytes: usize, usage: wgpu::BufferUsages) {
        let bytes = bytes.max(16); // a zero-length binding is not a thing the card will take
        if work.buffers[slot].is_some() && work.sized[slot] >= bytes { return; }
        work.buffers[slot] = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: bytes as u64, usage, mapped_at_creation: false }));
        work.sized[slot] = bytes;
        if slot == MEMBERSHIPS { work.seeded = 0; } // a fresh buffer holds nobody's traits
        if slot == NEXT { work.readback = None; } // and its readback has to match it
    }
    fn write(&self, work: &Work, slot: usize, raw: &[u8]) {
        self.queue.write_buffer(work.buffers[slot].as_ref().unwrap(), 0, raw);
    }
}

fn slot(work: &Work, index: usize) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding: index as u32 + 1, resource: work.buffers[index].as_ref().unwrap().as_entire_binding() }
}

/// Every binding is a plain storage buffer but the parameter block, which the shader reads as a uniform
fn bindings() -> [wgpu::BindGroupLayoutEntry; 9] {
    std::array::from_fn(|binding| wgpu::BindGroupLayoutEntry {
        binding: binding as u32,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: match binding {
                0 => wgpu::BufferBindingType::Uniform,
                8 => wgpu::BufferBindingType::Storage { read_only: false }, // where this tick's positions land
                _ => wgpu::BufferBindingType::Storage { read_only: true },
            },
            has_dynamic_offset: false, min_binding_size: None,
        },
        count: None,
    })
}

/// Enough of an executor to wait for the two calls that open a device. Both resolve on this thread on
/// native, so a whole async runtime would be a dependency for a loop that spins at most once.
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    let mut future = std::pin::pin!(future);
    loop {
        if let std::task::Poll::Ready(value) = future.as_mut().poll(&mut context) { return value; }
        std::thread::yield_now();
    }
}
