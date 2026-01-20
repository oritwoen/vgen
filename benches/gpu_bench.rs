use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use vgen::gpu::GpuRunner;
use vgen::{scan_gpu_with_runner, AddressFormat, Pattern, ScanConfig};

fn bench_gpu_batches(c: &mut Criterion) {
    // Fixed trivial pattern and format to isolate batch size effects.
    let pattern = Pattern::new("^1", false).expect("pattern ok");
    let config_base = ScanConfig {
        format: AddressFormat::P2pkh,
        count: 1,
        threads: None,
        gpu_batch_size: None,
        cpu_batch_size: None,
        start: None,
        end: None,
    };

    let mut group = c.benchmark_group("gpu_batch_size");
    group.sample_size(10);

    for &batch in &[262_144u32, 524_288, 1_048_576, 2_097_152] {
        group.throughput(Throughput::Elements(batch as u64));
        group.bench_with_input(
            BenchmarkId::new("batch", batch),
            &batch,
            |b, &batch_size| {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                let runner = rt.block_on(GpuRunner::new(batch_size)).unwrap();
                let runner = Arc::new(runner);
                let mut config = config_base.clone();
                config.gpu_batch_size = Some(batch_size);

                b.iter(|| {
                    rt.block_on(scan_gpu_with_runner(
                        &pattern,
                        &config,
                        None,
                        Some(Arc::new(AtomicBool::new(false))),
                        runner.clone(),
                    ))
                    .unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_gpu_batches);
criterion_main!(benches);
