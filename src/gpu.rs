use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use bitcoin::secp256k1::{PublicKey, Secp256k1, SecretKey};
use clap::ValueEnum;
use num_bigint::BigUint;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use wgpu::util::DeviceExt;

use crate::address::{AddressFormat, AddressGenerator, GeneratedAddress};
use crate::{Pattern, ProgressCallback, ScanConfig, ScanResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum GpuBackend {
    #[default]
    Auto,
    Vulkan,
    Metal,
    Dx12,
    Gl,
}

impl GpuBackend {
    pub fn to_wgpu_backends(self) -> wgpu::Backends {
        match self {
            GpuBackend::Auto => wgpu::Backends::all(),
            GpuBackend::Vulkan => wgpu::Backends::VULKAN,
            GpuBackend::Metal => wgpu::Backends::METAL,
            GpuBackend::Dx12 => wgpu::Backends::DX12,
            GpuBackend::Gl => wgpu::Backends::GL,
        }
    }

    pub fn fallback_order() -> &'static [GpuBackend] {
        &[
            GpuBackend::Vulkan,
            GpuBackend::Metal,
            GpuBackend::Dx12,
            GpuBackend::Gl,
        ]
    }

    pub fn name(self) -> &'static str {
        match self {
            GpuBackend::Auto => "auto",
            GpuBackend::Vulkan => "Vulkan",
            GpuBackend::Metal => "Metal",
            GpuBackend::Dx12 => "DX12",
            GpuBackend::Gl => "OpenGL",
        }
    }
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

pub(crate) fn is_software_adapter(info: &wgpu::AdapterInfo) -> bool {
    if info.device_type == wgpu::DeviceType::Cpu {
        return true;
    }

    let name = info.name.to_lowercase();
    [
        "llvmpipe",
        "swiftshader",
        "lavapipe",
        "software",
        "mesa software",
    ]
    .iter()
    .any(|needle| name.contains(needle))
}

// Default batch size (can be overridden by config)
const DEFAULT_BATCH_SIZE: u32 = 1024 * 1024; // 1 Million

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BigInt256 {
    v0: [u32; 4],
    v1: [u32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Config {
    base_x: BigInt256,
    base_y: BigInt256,
    num_keys: u32,
    _pad: [u32; 3],
}

struct Frame {
    config_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    // For batch affine inversion (reserved for future optimization)
    #[allow(dead_code)]
    jacobian_buffer: wgpu::Buffer,
    // For P2TR output (32 bytes per key)
    p2tr_output_buffer: wgpu::Buffer,
    p2tr_staging_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    receiver: Mutex<Option<tokio::sync::oneshot::Receiver<Result<(), wgpu::BufferAsyncError>>>>,
    batch_start_key: Mutex<[u8; 32]>,
}

pub struct GpuRunner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    // Legacy single-pass pipeline (kept for compatibility)
    pipeline: wgpu::ComputePipeline,
    // Batch affine inversion pipelines
    compute_jacobian_pipeline: wgpu::ComputePipeline,
    batch_normalize_hash_pipeline: wgpu::ComputePipeline,
    batch_normalize_p2tr_pipeline: wgpu::ComputePipeline,
    frames: Vec<Frame>,
    pub batch_size: u32,
    pub device_name: String,
    backend: GpuBackend,
}

impl GpuRunner {
    pub fn backend(&self) -> GpuBackend {
        self.backend
    }

    pub async fn new(batch_size: u32, backend: GpuBackend) -> Result<Self> {
        fn device_type_priority(device_type: wgpu::DeviceType) -> u8 {
            match device_type {
                wgpu::DeviceType::DiscreteGpu => 0,
                wgpu::DeviceType::VirtualGpu => 1,
                wgpu::DeviceType::IntegratedGpu => 2,
                wgpu::DeviceType::Cpu => 3,
                _ => 4,
            }
        }

        let (_instance, adapter, selected_backend) = match backend {
            GpuBackend::Auto => {
                let mut selected: Option<(wgpu::Instance, wgpu::Adapter, GpuBackend)> = None;

                for &candidate in GpuBackend::fallback_order() {
                    let backends = candidate.to_wgpu_backends();
                    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                        backends,
                        ..Default::default()
                    });

                    let mut adapters = instance.enumerate_adapters(backends).await;
                    adapters.retain(|a| !is_software_adapter(&a.get_info()));
                    adapters.sort_by_key(|a| device_type_priority(a.get_info().device_type));

                    if let Some(adapter) = adapters.into_iter().next() {
                        selected = Some((instance, adapter, candidate));
                        break;
                    }
                }

                if selected.is_none() {
                    for &candidate in GpuBackend::fallback_order() {
                        let backends = candidate.to_wgpu_backends();
                        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                            backends,
                            ..Default::default()
                        });

                        let mut adapters = instance.enumerate_adapters(backends).await;
                        adapters.sort_by_key(|a| device_type_priority(a.get_info().device_type));

                        if let Some(adapter) = adapters.into_iter().next() {
                            selected = Some((instance, adapter, candidate));
                            break;
                        }
                    }
                }

                selected.context("No GPU backends available")?
            }
            _ => {
                let backends = backend.to_wgpu_backends();
                let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                    backends,
                    ..Default::default()
                });

                let mut adapters = instance.enumerate_adapters(backends).await;
                if adapters.is_empty() {
                    anyhow::bail!("Backend {} not available on this system", backend.name());
                }

                adapters.sort_by_key(|a| device_type_priority(a.get_info().device_type));
                let adapter = adapters
                    .into_iter()
                    .next()
                    .context("Failed to select GPU adapter")?;

                (instance, adapter, backend)
            }
        };

        let adapter_info = adapter.get_info();
        eprintln!(
            "Using GPU backend {} with adapter: {:?}",
            selected_backend, adapter_info
        );
        let device_name = adapter_info.name.clone();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Vanity Gen Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_storage_buffer_binding_size: 512 * 1024 * 1024,
                        ..wgpu::Limits::default()
                    },
                    memory_hints: wgpu::MemoryHints::Performance,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    trace: wgpu::Trace::Off,
                },
            )
            .await
            .context("Failed to create GPU device")?;

        eprintln!("Compiling shader...");
        let mut source = String::new();
        source.push_str(include_str!("shaders/sha256.wgsl"));
        source.push('\n');
        source.push_str(include_str!("shaders/ripemd160.wgsl"));
        source.push('\n');
        source.push_str(include_str!("shaders/generator.wgsl"));

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vanity Gen Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: jacobian_points buffer (for batch affine inversion)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 4: output_x_coords buffer (for P2TR)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        eprintln!("Creating init pipeline...");
        let init_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Init Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("init_table"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        eprintln!("Creating search pipeline...");
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Search Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Batch affine inversion pipelines
        eprintln!("Creating batch affine inversion pipelines...");
        let compute_jacobian_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Jacobian Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("compute_jacobian"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let batch_normalize_hash_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Batch Normalize Hash Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("batch_normalize_hash"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let batch_normalize_p2tr_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Batch Normalize P2TR Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("batch_normalize_p2tr"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let table_size = (batch_size as u64) * 64;
        let table_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Table Buffer"),
            size: table_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let num_frames = 2;
        let mut frames = Vec::with_capacity(num_frames);
        let output_size = (batch_size as u64) * 20; // Hash160 = 20 bytes
        let jacobian_size = (batch_size as u64) * 96; // JacobianPoint = 3 * 32 bytes
        let p2tr_output_size = (batch_size as u64) * 32; // X coordinate = 32 bytes

        let initial_config = Config {
            base_x: BigInt256 {
                v0: [0; 4],
                v1: [0; 4],
            },
            base_y: BigInt256 {
                v0: [0; 4],
                v1: [0; 4],
            },
            num_keys: batch_size,
            _pad: [0; 3],
        };

        for i in 0..num_frames {
            let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Config Buffer {}", i)),
                contents: bytemuck::bytes_of(&initial_config),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Output Buffer {}", i)),
                size: output_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Staging Buffer {}", i)),
                size: output_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Buffer for Jacobian points (batch affine inversion)
            let jacobian_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Jacobian Buffer {}", i)),
                size: jacobian_size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });

            // Buffer for P2TR X coordinate output
            let p2tr_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("P2TR Output Buffer {}", i)),
                size: p2tr_output_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let p2tr_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("P2TR Staging Buffer {}", i)),
                size: p2tr_output_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Bind Group {}", i)),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: config_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: table_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: jacobian_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: p2tr_output_buffer.as_entire_binding(),
                    },
                ],
            });

            frames.push(Frame {
                config_buffer,
                output_buffer,
                staging_buffer,
                jacobian_buffer,
                p2tr_output_buffer,
                p2tr_staging_buffer,
                bind_group,
                receiver: Mutex::new(None),
                batch_start_key: Mutex::new([0u8; 32]),
            });
        }

        eprintln!("Initializing lookup table on GPU (size: {})...", batch_size);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Init Encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Init Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&init_pipeline);
            cpass.set_bind_group(0, &frames[0].bind_group, &[]);
            let workgroups = (batch_size + 255) / 256;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        eprintln!("Initialization complete.");

        Ok(Self {
            device,
            queue,
            pipeline,
            compute_jacobian_pipeline,
            batch_normalize_hash_pipeline,
            batch_normalize_p2tr_pipeline,
            frames,
            batch_size,
            device_name,
            backend: selected_backend,
        })
    }

    pub fn dispatch(&self, start_key: [u8; 32], frame_index: usize) -> Result<()> {
        let frame = &self.frames[frame_index];

        let mut guard = frame.receiver.lock().unwrap();
        if guard.is_some() {
            frame.staging_buffer.unmap();
            *guard = None;
        }
        drop(guard);

        *frame.batch_start_key.lock().unwrap() = start_key;

        let (x_limbs, y_limbs) = key_to_affine(start_key)?;

        let config = Config {
            base_x: BigInt256 {
                v0: x_limbs[0..4].try_into()?,
                v1: x_limbs[4..8].try_into()?,
            },
            base_y: BigInt256 {
                v0: y_limbs[0..4].try_into()?,
                v1: y_limbs[4..8].try_into()?,
            },
            num_keys: self.batch_size,
            _pad: [0; 3],
        };

        self.queue
            .write_buffer(&frame.config_buffer, 0, bytemuck::bytes_of(&config));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Search Encoder"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Search Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &frame.bind_group, &[]);
            let workgroups = (self.batch_size + 255) / 256;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &frame.output_buffer,
            0,
            &frame.staging_buffer,
            0,
            frame.output_buffer.size(),
        );

        self.queue.submit(Some(encoder.finish()));

        let slice = frame.staging_buffer.slice(..);
        let (tx, rx) = tokio::sync::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = tx.send(v);
        });

        *frame.receiver.lock().unwrap() = Some(rx);

        Ok(())
    }

    pub async fn await_result(&self, frame_index: usize) -> Result<(Vec<[u8; 20]>, [u8; 32])> {
        let frame = &self.frames[frame_index];

        loop {
            self.device.poll(wgpu::PollType::Poll).ok();

            let mut guard = frame.receiver.lock().unwrap();
            if let Some(rx) = guard.as_mut() {
                match rx.try_recv() {
                    Ok(res) => {
                        res?;
                        *guard = None;
                        break;
                    }
                    Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                        drop(guard);
                        tokio::task::yield_now().await;
                        continue;
                    }
                    Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                        anyhow::bail!("Sender dropped");
                    }
                }
            } else {
                anyhow::bail!("No pending operation on frame {}", frame_index);
            }
        }

        let slice = frame.staging_buffer.slice(..);
        let data = slice.get_mapped_range();

        let mut hashes = Vec::with_capacity(self.batch_size as usize);
        for i in 0..self.batch_size as usize {
            let start = i * 20;
            let end = start + 20;
            let hash_bytes: [u8; 20] = data[start..end].try_into()?;
            hashes.push(hash_bytes);
        }

        drop(data);
        frame.staging_buffer.unmap();

        let start_key = *frame.batch_start_key.lock().unwrap();

        Ok((hashes, start_key))
    }

    pub async fn generate_batch(&self, start_key: [u8; 32]) -> Result<Vec<[u8; 20]>> {
        self.dispatch(start_key, 0)?;
        let (res, _) = self.await_result(0).await?;
        Ok(res)
    }

    /// Dispatch P2TR computation with batch affine inversion
    pub fn dispatch_p2tr(&self, start_key: [u8; 32], frame_index: usize) -> Result<()> {
        let frame = &self.frames[frame_index];

        let mut guard = frame.receiver.lock().unwrap();
        if guard.is_some() {
            frame.p2tr_staging_buffer.unmap();
            *guard = None;
        }
        drop(guard);

        *frame.batch_start_key.lock().unwrap() = start_key;

        let (x_limbs, y_limbs) = key_to_affine(start_key)?;

        let config = Config {
            base_x: BigInt256 {
                v0: x_limbs[0..4].try_into()?,
                v1: x_limbs[4..8].try_into()?,
            },
            base_y: BigInt256 {
                v0: y_limbs[0..4].try_into()?,
                v1: y_limbs[4..8].try_into()?,
            },
            num_keys: self.batch_size,
            _pad: [0; 3],
        };

        self.queue
            .write_buffer(&frame.config_buffer, 0, bytemuck::bytes_of(&config));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("P2TR Search Encoder"),
            });

        let workgroups = (self.batch_size + 255) / 256;

        // Step 1: Compute Jacobian points (no normalization)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Jacobian Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_jacobian_pipeline);
            cpass.set_bind_group(0, &frame.bind_group, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Step 2: Batch normalize with Blelloch scan, output X coordinates
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Batch Normalize P2TR Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.batch_normalize_p2tr_pipeline);
            cpass.set_bind_group(0, &frame.bind_group, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy P2TR output (32 bytes per key)
        encoder.copy_buffer_to_buffer(
            &frame.p2tr_output_buffer,
            0,
            &frame.p2tr_staging_buffer,
            0,
            frame.p2tr_output_buffer.size(),
        );

        self.queue.submit(Some(encoder.finish()));

        let slice = frame.p2tr_staging_buffer.slice(..);
        let (tx, rx) = tokio::sync::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = tx.send(v);
        });

        *frame.receiver.lock().unwrap() = Some(rx);

        Ok(())
    }

    /// Await P2TR result (X coordinates as 32-byte arrays)
    pub async fn await_result_p2tr(&self, frame_index: usize) -> Result<(Vec<[u8; 32]>, [u8; 32])> {
        let frame = &self.frames[frame_index];

        loop {
            self.device.poll(wgpu::PollType::Poll).ok();

            let mut guard = frame.receiver.lock().unwrap();
            if let Some(rx) = guard.as_mut() {
                match rx.try_recv() {
                    Ok(res) => {
                        res?;
                        *guard = None;
                        break;
                    }
                    Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                        drop(guard);
                        tokio::task::yield_now().await;
                        continue;
                    }
                    Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                        anyhow::bail!("Sender dropped");
                    }
                }
            } else {
                anyhow::bail!("No pending operation on frame {}", frame_index);
            }
        }

        let slice = frame.p2tr_staging_buffer.slice(..);
        let data = slice.get_mapped_range();

        let mut x_coords = Vec::with_capacity(self.batch_size as usize);
        for i in 0..self.batch_size as usize {
            let start = i * 32;
            let end = start + 32;
            let x_bytes: [u8; 32] = data[start..end].try_into()?;
            x_coords.push(x_bytes);
        }

        drop(data);
        frame.p2tr_staging_buffer.unmap();

        let start_key = *frame.batch_start_key.lock().unwrap();

        Ok((x_coords, start_key))
    }

    /// Dispatch batch with Blelloch optimization for Hash160 output (P2PKH/P2WPKH)
    pub fn dispatch_batch_hash(&self, start_key: [u8; 32], frame_index: usize) -> Result<()> {
        let frame = &self.frames[frame_index];

        let mut guard = frame.receiver.lock().unwrap();
        if guard.is_some() {
            frame.staging_buffer.unmap();
            *guard = None;
        }
        drop(guard);

        *frame.batch_start_key.lock().unwrap() = start_key;

        let (x_limbs, y_limbs) = key_to_affine(start_key)?;

        let config = Config {
            base_x: BigInt256 {
                v0: x_limbs[0..4].try_into()?,
                v1: x_limbs[4..8].try_into()?,
            },
            base_y: BigInt256 {
                v0: y_limbs[0..4].try_into()?,
                v1: y_limbs[4..8].try_into()?,
            },
            num_keys: self.batch_size,
            _pad: [0; 3],
        };

        self.queue
            .write_buffer(&frame.config_buffer, 0, bytemuck::bytes_of(&config));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Batch Hash Search Encoder"),
            });

        let workgroups = (self.batch_size + 255) / 256;

        // Step 1: Compute Jacobian points
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Jacobian Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_jacobian_pipeline);
            cpass.set_bind_group(0, &frame.bind_group, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Step 2: Batch normalize with Blelloch scan, output Hash160
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Batch Normalize Hash Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.batch_normalize_hash_pipeline);
            cpass.set_bind_group(0, &frame.bind_group, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &frame.output_buffer,
            0,
            &frame.staging_buffer,
            0,
            frame.output_buffer.size(),
        );

        self.queue.submit(Some(encoder.finish()));

        let slice = frame.staging_buffer.slice(..);
        let (tx, rx) = tokio::sync::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = tx.send(v);
        });

        *frame.receiver.lock().unwrap() = Some(rx);

        Ok(())
    }
}

fn key_to_affine(key: [u8; 32]) -> Result<([u32; 8], [u32; 8])> {
    let secp = Secp256k1::new();
    let sk = SecretKey::from_slice(&key)?;
    let pk = PublicKey::from_secret_key(&secp, &sk);
    let serialized = pk.serialize_uncompressed();

    fn bytes_be_to_u32_le(bytes: &[u8]) -> [u32; 8] {
        let mut limbs = [0u32; 8];
        for i in 0..8 {
            let start = 28 - i * 4;
            let chunk: [u8; 4] = bytes[start..start + 4].try_into().unwrap();
            limbs[i] = u32::from_be_bytes(chunk);
        }
        limbs
    }

    let x_limbs = bytes_be_to_u32_le(&serialized[1..33]);
    let y_limbs = bytes_be_to_u32_le(&serialized[33..65]);
    Ok((x_limbs, y_limbs))
}

pub async fn scan_gpu_with_runner(
    pattern: &Pattern,
    config: &ScanConfig,
    progress_cb: Option<ProgressCallback>,
    stop: Option<Arc<AtomicBool>>,
    runner: Arc<GpuRunner>,
) -> Result<ScanResult> {
    let start = Instant::now();

    let batch_size = runner.batch_size;

    let mut total_ops = 0u64;

    fn biguint_to_bytes32(n: &BigUint) -> [u8; 32] {
        let bytes = n.to_bytes_be();
        let mut result = [0u8; 32];
        let start = 32usize.saturating_sub(bytes.len());
        result[start..].copy_from_slice(&bytes[bytes.len().saturating_sub(32)..]);
        result
    }

    let mut current_key: [u8; 32] = if let Some(start) = &config.start {
        biguint_to_bytes32(start)
    } else {
        // Generate cryptographically secure random key with rejection sampling
        let mut rng = StdRng::from_entropy();
        loop {
            let k: [u8; 32] = rng.gen();
            // Ensure key is valid for secp256k1 (non-zero and less than curve order)
            if SecretKey::from_slice(&k).is_ok() {
                break k;
            }
        }
    };

    let end_key_bytes = config.end.as_ref().map(|e| biguint_to_bytes32(e));
    let matches_mutex: Arc<Mutex<Vec<GeneratedAddress>>> = Arc::new(Mutex::new(Vec::new()));
    let found = Arc::new(AtomicUsize::new(0));

    let increment_key = |key: [u8; 32], amount: u64| -> Option<[u8; 32]> {
        let mut k = key;
        let mut carry = amount;
        for byte_idx in (0..32).rev() {
            let sum = k[byte_idx] as u64 + (carry & 0xFF);
            k[byte_idx] = sum as u8;
            carry = (carry >> 8) + (sum >> 8);
            if carry == 0 {
                break;
            }
        }
        // Return None if overflow occurred or key is invalid
        if carry > 0 || SecretKey::from_slice(&k).is_err() {
            None
        } else {
            Some(k)
        }
    };

    let num_frames = runner.frames.len();
    let mut in_flight = 0;

    for i in 0..num_frames {
        if let Some(end) = end_key_bytes {
            if current_key > end {
                break;
            }
        }

        if let Some(s) = &stop {
            if s.load(Ordering::Relaxed) {
                break;
            }
        }
        if found.load(Ordering::Relaxed) >= config.count {
            break;
        }

        runner.dispatch(current_key, i)?;
        match increment_key(current_key, batch_size as u64) {
            Some(k) => current_key = k,
            None => break, // Key space exhausted
        }
        in_flight += 1;
    }

    let mut frame = 0;
    loop {
        if in_flight == 0 {
            break;
        }

        let (hashes, batch_start_key) = runner.await_result(frame).await?;
        in_flight -= 1;

        let mut dispatched_next = false;
        let should_continue = if let Some(s) = &stop {
            !s.load(Ordering::Relaxed)
        } else {
            true
        } && found.load(Ordering::Relaxed) < config.count;

        if should_continue {
            let in_range = if let Some(end) = end_key_bytes {
                current_key <= end
            } else {
                true
            };

            if in_range {
                runner.dispatch(current_key, frame)?;
                if let Some(k) = increment_key(current_key, batch_size as u64) {
                    current_key = k;
                }
                in_flight += 1;
                dispatched_next = true;
            }
        }

        let found_in_batch = hashes
            .par_iter()
            .enumerate()
            .filter_map(|(i, hash160)| {
                let addr_string = match config.format {
                    AddressFormat::P2pkh | AddressFormat::P2pkhUncompressed => {
                        use bitcoin::hashes::Hash;
                        use bitcoin::PubkeyHash;
                        let pkh = PubkeyHash::from_byte_array(*hash160);
                        bitcoin::Address::p2pkh(pkh, bitcoin::Network::Bitcoin).to_string()
                    }
                    AddressFormat::P2wpkh => {
                        use bitcoin::hashes::Hash;
                        use bitcoin::ScriptBuf;
                        use bitcoin::WPubkeyHash;
                        let wpkh = WPubkeyHash::from_byte_array(*hash160);
                        let script = ScriptBuf::new_p2wpkh(&wpkh);
                        bitcoin::Address::from_script(&script, bitcoin::Network::Bitcoin)
                            .expect("valid script")
                            .to_string()
                    }
                    AddressFormat::Ethereum => {
                        unreachable!("GPU path not supported for Ethereum")
                    }
                    _ => {
                        return None;
                    }
                };

                if pattern.matches(&addr_string) {
                    let k = match increment_key(batch_start_key, i as u64) {
                        Some(key) => key,
                        None => return None, // Invalid key
                    };

                    if let Some(end) = end_key_bytes {
                        if k > end {
                            return None;
                        }
                    }

                    let wif = match AddressGenerator::bytes_to_wif(&k, bitcoin::Network::Bitcoin) {
                        Some(w) => w,
                        None => return None, // Invalid key for WIF
                    };
                    Some(GeneratedAddress {
                        address: addr_string,
                        wif,
                        hex: hex::encode(k),
                        format: config.format,
                    })
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        if !found_in_batch.is_empty() {
            let mut m = matches_mutex.lock().unwrap();
            for addr in found_in_batch {
                if found.load(Ordering::Relaxed) < config.count {
                    m.push(addr);
                    found.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        total_ops += batch_size as u64;
        if let Some(cb) = &progress_cb {
            cb(total_ops);
        }

        if found.load(Ordering::Relaxed) >= config.count && !dispatched_next {
            break;
        }

        frame = (frame + 1) % num_frames;
    }

    let matches = matches_mutex.lock().unwrap().clone();

    Ok(ScanResult {
        matches,
        operations: total_ops,
        elapsed_secs: start.elapsed().as_secs_f64(),
    })
}

pub fn scan_gpu(
    pattern: &Pattern,
    config: &ScanConfig,
    progress_cb: Option<ProgressCallback>,
    stop: Option<Arc<AtomicBool>>,
) -> Result<ScanResult> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    let batch_size = config.gpu_batch_size.unwrap_or(DEFAULT_BATCH_SIZE);
    let runner = rt.block_on(GpuRunner::new(batch_size, GpuBackend::Auto))?;
    let runner = Arc::new(runner);

    rt.block_on(scan_gpu_with_runner(
        pattern,
        config,
        progress_cb,
        stop,
        runner,
    ))
}

/// Convert GPU limbs (little-endian u32 array) to big-endian bytes
fn limbs_to_bytes_be(limbs: &[u8; 32]) -> [u8; 32] {
    // GPU stores as 8 x u32 limbs in LE order
    // limbs[0..4] is LSB, limbs[28..32] is MSB
    let mut bytes = [0u8; 32];
    for i in 0..8 {
        let limb_start = i * 4;
        let out_start = (7 - i) * 4;
        // Each limb is stored as LE bytes, but represents BE value
        bytes[out_start] = limbs[limb_start + 3];
        bytes[out_start + 1] = limbs[limb_start + 2];
        bytes[out_start + 2] = limbs[limb_start + 1];
        bytes[out_start + 3] = limbs[limb_start];
    }
    bytes
}

/// GPU scan function for P2TR addresses
pub async fn scan_gpu_p2tr_with_runner(
    pattern: &Pattern,
    config: &ScanConfig,
    progress_cb: Option<ProgressCallback>,
    stop: Option<Arc<AtomicBool>>,
    runner: Arc<GpuRunner>,
) -> Result<ScanResult> {
    let start = Instant::now();
    let batch_size = runner.batch_size;
    let mut total_ops = 0u64;

    fn biguint_to_bytes32(n: &BigUint) -> [u8; 32] {
        let bytes = n.to_bytes_be();
        let mut result = [0u8; 32];
        let start = 32usize.saturating_sub(bytes.len());
        result[start..].copy_from_slice(&bytes[bytes.len().saturating_sub(32)..]);
        result
    }

    let mut current_key: [u8; 32] = if let Some(start) = &config.start {
        biguint_to_bytes32(start)
    } else {
        let mut rng = StdRng::from_entropy();
        loop {
            let k: [u8; 32] = rng.gen();
            if SecretKey::from_slice(&k).is_ok() {
                break k;
            }
        }
    };

    let end_key_bytes = config.end.as_ref().map(|e| biguint_to_bytes32(e));
    let matches_mutex: Arc<Mutex<Vec<GeneratedAddress>>> = Arc::new(Mutex::new(Vec::new()));
    let found = Arc::new(AtomicUsize::new(0));

    let increment_key = |key: [u8; 32], amount: u64| -> Option<[u8; 32]> {
        let mut k = key;
        let mut carry = amount;
        for byte_idx in (0..32).rev() {
            let sum = k[byte_idx] as u64 + (carry & 0xFF);
            k[byte_idx] = sum as u8;
            carry = (carry >> 8) + (sum >> 8);
            if carry == 0 {
                break;
            }
        }
        if carry > 0 || SecretKey::from_slice(&k).is_err() {
            None
        } else {
            Some(k)
        }
    };

    let num_frames = runner.frames.len();
    let mut in_flight = 0;

    // Initial dispatch
    for i in 0..num_frames {
        if let Some(end) = end_key_bytes {
            if current_key > end {
                break;
            }
        }
        if let Some(s) = &stop {
            if s.load(Ordering::Relaxed) {
                break;
            }
        }
        if found.load(Ordering::Relaxed) >= config.count {
            break;
        }

        runner.dispatch_p2tr(current_key, i)?;
        match increment_key(current_key, batch_size as u64) {
            Some(k) => current_key = k,
            None => break,
        }
        in_flight += 1;
    }

    let mut frame = 0;

    loop {
        if in_flight == 0 {
            break;
        }

        let (x_coords, batch_start_key) = runner.await_result_p2tr(frame).await?;
        in_flight -= 1;

        let mut dispatched_next = false;
        let should_continue = if let Some(s) = &stop {
            !s.load(Ordering::Relaxed)
        } else {
            true
        } && found.load(Ordering::Relaxed) < config.count;

        if should_continue {
            let in_range = if let Some(end) = end_key_bytes {
                current_key <= end
            } else {
                true
            };

            if in_range {
                runner.dispatch_p2tr(current_key, frame)?;
                if let Some(k) = increment_key(current_key, batch_size as u64) {
                    current_key = k;
                }
                in_flight += 1;
                dispatched_next = true;
            }
        }

        // Create secp256k1 context once outside the parallel iterator
        use bitcoin::secp256k1::{Secp256k1, XOnlyPublicKey};
        let secp = Secp256k1::verification_only();

        let found_in_batch = x_coords
            .par_iter()
            .enumerate()
            .filter_map(|(i, x_raw)| {
                // Convert GPU limbs to BE bytes - this is the internal x-only pubkey
                let x_bytes = limbs_to_bytes_be(x_raw);

                // Get the private key for this index
                let k = increment_key(batch_start_key, i as u64)?;

                // GPU returns internal X, CPU applies Taproot tweak
                let internal_key = XOnlyPublicKey::from_slice(&x_bytes).ok()?;
                let addr =
                    bitcoin::Address::p2tr(&secp, internal_key, None, bitcoin::Network::Bitcoin);
                let addr_string = addr.to_string();

                if pattern.matches(&addr_string) {
                    if let Some(end) = end_key_bytes {
                        if k > end {
                            return None;
                        }
                    }

                    let wif = AddressGenerator::bytes_to_wif(&k, bitcoin::Network::Bitcoin)?;
                    Some(GeneratedAddress {
                        address: addr_string,
                        wif,
                        hex: hex::encode(k),
                        format: AddressFormat::P2tr,
                    })
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        if !found_in_batch.is_empty() {
            let mut m = matches_mutex.lock().unwrap();
            for addr in found_in_batch {
                if found.load(Ordering::Relaxed) < config.count {
                    m.push(addr);
                    found.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        total_ops += batch_size as u64;
        if let Some(cb) = &progress_cb {
            cb(total_ops);
        }

        if found.load(Ordering::Relaxed) >= config.count && !dispatched_next {
            break;
        }

        frame = (frame + 1) % num_frames;
    }

    let matches = matches_mutex.lock().unwrap().clone();

    Ok(ScanResult {
        matches,
        operations: total_ops,
        elapsed_secs: start.elapsed().as_secs_f64(),
    })
}

/// Public entry point for P2TR GPU scan
pub fn scan_gpu_p2tr(
    pattern: &Pattern,
    config: &ScanConfig,
    progress_cb: Option<ProgressCallback>,
    stop: Option<Arc<AtomicBool>>,
) -> Result<ScanResult> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    let batch_size = config.gpu_batch_size.unwrap_or(DEFAULT_BATCH_SIZE);
    let runner = rt.block_on(GpuRunner::new(batch_size, GpuBackend::Auto))?;
    let runner = Arc::new(runner);

    rt.block_on(scan_gpu_p2tr_with_runner(
        pattern,
        config,
        progress_cb,
        stop,
        runner,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_backend_to_wgpu_backends() {
        assert_eq!(GpuBackend::Auto.to_wgpu_backends(), wgpu::Backends::all());
        assert_eq!(
            GpuBackend::Vulkan.to_wgpu_backends(),
            wgpu::Backends::VULKAN
        );
        assert_eq!(GpuBackend::Metal.to_wgpu_backends(), wgpu::Backends::METAL);
        assert_eq!(GpuBackend::Dx12.to_wgpu_backends(), wgpu::Backends::DX12);
        assert_eq!(GpuBackend::Gl.to_wgpu_backends(), wgpu::Backends::GL);
    }

    #[test]
    fn test_gpu_backend_fallback_order() {
        assert_eq!(
            GpuBackend::fallback_order(),
            &[
                GpuBackend::Vulkan,
                GpuBackend::Metal,
                GpuBackend::Dx12,
                GpuBackend::Gl
            ]
        );
    }

    #[test]
    fn test_gpu_backend_display() {
        assert_eq!(GpuBackend::Auto.to_string(), "auto");
        assert_eq!(GpuBackend::Vulkan.to_string(), "Vulkan");
        assert_eq!(GpuBackend::Metal.to_string(), "Metal");
        assert_eq!(GpuBackend::Dx12.to_string(), "DX12");
        assert_eq!(GpuBackend::Gl.to_string(), "OpenGL");
    }

    fn adapter_info(name: &str, device_type: wgpu::DeviceType) -> wgpu::AdapterInfo {
        wgpu::AdapterInfo {
            name: name.to_string(),
            vendor: 0,
            device: 0,
            device_type,
            device_pci_bus_id: String::new(),
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Vulkan,
            subgroup_min_size: 4,
            subgroup_max_size: 64,
            transient_saves_memory: false,
        }
    }

    #[test]
    fn test_is_software_adapter() {
        assert!(is_software_adapter(&adapter_info(
            "anything",
            wgpu::DeviceType::Cpu
        )));

        assert!(is_software_adapter(&adapter_info(
            "llvmpipe (LLVM 16.0.6, 256 bits)",
            wgpu::DeviceType::IntegratedGpu
        )));
        assert!(is_software_adapter(&adapter_info(
            "SwiftShader Device (Subzero)",
            wgpu::DeviceType::IntegratedGpu
        )));
        assert!(is_software_adapter(&adapter_info(
            "lavapipe",
            wgpu::DeviceType::IntegratedGpu
        )));
        assert!(is_software_adapter(&adapter_info(
            "Software Rasterizer",
            wgpu::DeviceType::IntegratedGpu
        )));
        assert!(is_software_adapter(&adapter_info(
            "Mesa Software Rasterizer",
            wgpu::DeviceType::IntegratedGpu
        )));

        assert!(!is_software_adapter(&adapter_info(
            "NVIDIA GeForce RTX 3080",
            wgpu::DeviceType::DiscreteGpu
        )));
        assert!(!is_software_adapter(&adapter_info(
            "AMD Radeon RX 6800",
            wgpu::DeviceType::DiscreteGpu
        )));
    }
}
