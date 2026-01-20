use criterion::{criterion_group, criterion_main, Criterion};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use vgen::{AddressFormat, AddressGenerator, Pattern};

fn bench_address_generation(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let secret: [u8; 32] = rng.gen();

    let mut group = c.benchmark_group("address_generation");

    group.bench_function("p2pkh_generate", |b| {
        let gen = AddressGenerator::new(AddressFormat::P2pkh);
        b.iter(|| {
            gen.generate(std::hint::black_box(&secret)).unwrap();
        })
    });

    group.bench_function("p2wpkh_generate", |b| {
        let gen = AddressGenerator::new(AddressFormat::P2wpkh);
        b.iter(|| {
            gen.generate(std::hint::black_box(&secret)).unwrap();
        })
    });

    group.finish();
}

fn bench_pattern_matching(c: &mut Criterion) {
    let pattern = Pattern::new("^1Test", false).unwrap();
    let address_match = "1TestXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX";
    let address_no_match = "1FailXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX";

    let mut group = c.benchmark_group("pattern_matching");

    group.bench_function("match_success", |b| {
        b.iter(|| {
            pattern.matches(std::hint::black_box(address_match));
        })
    });

    group.bench_function("match_fail", |b| {
        b.iter(|| {
            pattern.matches(std::hint::black_box(address_no_match));
        })
    });

    group.finish();
}

fn bench_scan_hot_loop(c: &mut Criterion) {
    let mut group = c.benchmark_group("scan_hot_loop");

    group.bench_function("p2pkh_smallrng_regex", |b| {
        let mut rng = SmallRng::from_seed([42u8; 32]);
        let gen = AddressGenerator::new(AddressFormat::P2pkh);
        let pat = Pattern::new("^1", false).unwrap();
        let mut secret = [0u8; 32];

        b.iter(|| {
            rng.fill(&mut secret);
            let addr = gen.generate(std::hint::black_box(&secret)).unwrap();
            let _ = pat.matches(std::hint::black_box(&addr.address));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_address_generation,
    bench_pattern_matching,
    bench_scan_hot_loop
);
criterion_main!(benches);
