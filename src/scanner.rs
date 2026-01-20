//! Parallel address scanning with rayon.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use num_bigint::BigUint;
use num_traits::{One, ToPrimitive};
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::address::{AddressFormat, AddressGenerator, GeneratedAddress};
use crate::pattern::Pattern;

/// Scanner configuration.
#[derive(Debug, Clone)]
pub struct ScanConfig {
    /// Address format to generate
    pub format: AddressFormat,
    /// Number of matches to find before stopping
    pub count: usize,
    /// Number of threads (None = all CPUs)
    pub threads: Option<usize>,
    /// GPU Batch Size (default: 1M)
    pub gpu_batch_size: Option<u32>,
    /// CPU batch size per thread
    pub cpu_batch_size: Option<usize>,
    /// Start of range (if range scan)
    pub start: Option<BigUint>,
    /// End of range (if range scan)
    pub end: Option<BigUint>,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            format: AddressFormat::P2pkh,
            count: 1,
            threads: None,
            gpu_batch_size: None,
            cpu_batch_size: None,
            start: None,
            end: None,
        }
    }
}

/// Result of a scan operation.
#[derive(Debug)]
pub struct ScanResult {
    /// Found addresses matching the pattern
    pub matches: Vec<GeneratedAddress>,
    /// Total number of keys checked
    pub operations: u64,
    /// Elapsed time in seconds
    pub elapsed_secs: f64,
}

impl ScanResult {
    /// Calculate addresses per second rate.
    pub fn rate(&self) -> f64 {
        if self.elapsed_secs > 0.0 {
            self.operations as f64 / self.elapsed_secs
        } else {
            0.0
        }
    }
}

/// Callback type for progress reporting.
pub type ProgressCallback = Arc<dyn Fn(u64) + Send + Sync>;

/// Scan for addresses matching a pattern.
///
/// Returns when `count` matches are found or the stop flag is set.
pub fn scan(pattern: &Pattern, config: &ScanConfig) -> ScanResult {
    scan_with_progress(pattern, config, None, None)
}

/// Scan with optional progress callback and stop flag.
pub fn scan_with_progress(
    pattern: &Pattern,
    config: &ScanConfig,
    progress_callback: Option<ProgressCallback>,
    stop_flag: Option<Arc<AtomicBool>>,
) -> ScanResult {
    if config.start.is_some() {
        return scan_range_cpu(pattern, config, progress_callback, stop_flag);
    }

    let start = Instant::now();
    let operations = Arc::new(AtomicU64::new(0));
    let found = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let stop = stop_flag.unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
    let matches = Arc::new(std::sync::Mutex::new(Vec::new()));

    // Configure thread pool if specified
    let pool = if let Some(threads) = config.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .ok()
    } else {
        None
    };

    let batch_size = config.cpu_batch_size.unwrap_or(10_000);

    let scan_batch = || {
        // Process batches until we find enough matches or get stopped
        while !stop.load(Ordering::Relaxed) {
            // Stop if we already have enough matches
            if found.load(Ordering::Relaxed) >= config.count {
                break;
            }

            // Process a batch in parallel
            let batch_matches: Vec<GeneratedAddress> = (0..batch_size)
                .into_par_iter()
                .filter_map(|_| {
                    // Fast bail-out if we already have enough
                    if stop.load(Ordering::Relaxed) || found.load(Ordering::Relaxed) >= config.count
                    {
                        return None;
                    }

                    // Thread-local resources
                    thread_local! {
                        static GENERATOR: std::cell::RefCell<Option<AddressGenerator>> =
                            const { std::cell::RefCell::new(None) };
                        static RNG: std::cell::RefCell<Option<rand::rngs::StdRng>> =
                            const { std::cell::RefCell::new(None) };
                    }

                    GENERATOR.with(|gen| {
                        RNG.with(|rng_cell| {
                            let mut gen = gen.borrow_mut();
                            let mut rng_ref = rng_cell.borrow_mut();

                            if gen.is_none() {
                                *gen = Some(AddressGenerator::new(config.format));
                            }
                            if rng_ref.is_none() {
                                *rng_ref = Some(rand::rngs::StdRng::from_entropy());
                            }

                            let generator = gen.as_ref().unwrap();
                            let rng = rng_ref.as_mut().unwrap();

                            // Generate random key
                            let mut secret = [0u8; 32];
                            rng.fill(&mut secret);

                            // Generate address
                            let addr = generator.generate(&secret)?;

                            // Check if matches pattern
                            if pattern.matches(&addr.address) {
                                // Increment found counter; drop result if we raced past target
                                let prev = found.fetch_add(1, Ordering::Relaxed);
                                if prev < config.count {
                                    return Some(addr);
                                }
                            }
                            None
                        })
                    })
                })
                .collect();

            // Update operations counter once per batch to reduce atomic contention
            operations.fetch_add(batch_size as u64, Ordering::Relaxed);

            // Add batch matches to results
            if !batch_matches.is_empty() {
                let mut m = matches.lock().unwrap();
                for addr in batch_matches {
                    if m.len() < config.count {
                        m.push(addr);
                    }
                    if m.len() >= config.count {
                        break;
                    }
                }
            }

            // Call progress callback
            if let Some(ref cb) = progress_callback {
                cb(operations.load(Ordering::Relaxed));
            }
        }
    };

    if let Some(pool) = pool {
        pool.install(scan_batch);
    } else {
        scan_batch();
    }

    let elapsed = start.elapsed().as_secs_f64();
    let found_matches = matches.lock().unwrap().clone();

    ScanResult {
        matches: found_matches,
        operations: operations.load(Ordering::Relaxed),
        elapsed_secs: elapsed,
    }
}

fn scan_range_cpu(
    pattern: &Pattern,
    config: &ScanConfig,
    progress_callback: Option<ProgressCallback>,
    stop_flag: Option<Arc<AtomicBool>>,
) -> ScanResult {
    let start_time = Instant::now();
    let operations = Arc::new(AtomicU64::new(0));
    let stop = stop_flag.unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
    let matches = Arc::new(std::sync::Mutex::new(Vec::new()));

    let start_key = config.start.clone().unwrap_or_else(BigUint::one);
    let end_key = config
        .end
        .clone()
        .unwrap_or_else(|| &start_key + BigUint::from(u64::MAX));

    let total_keys = &end_key - &start_key + 1u32;
    let total_keys_u64 = total_keys.to_u64().unwrap_or(u64::MAX);

    // Configure thread pool if specified
    let pool = if let Some(threads) = config.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .ok()
    } else {
        None
    };

    let batch_size = 10_000u64;
    let num_batches = (total_keys_u64 + batch_size - 1) / batch_size;

    let scan_loop = || {
        (0..num_batches).into_par_iter().for_each(|batch_idx| {
            if stop.load(Ordering::Relaxed) {
                return;
            }

            // Check if found enough
            {
                let m = matches.lock().unwrap();
                if m.len() >= config.count {
                    stop.store(true, Ordering::Relaxed);
                    return;
                }
            }

            let batch_start = &start_key + batch_idx * batch_size;
            let current_batch_size = if batch_idx == num_batches - 1 {
                total_keys_u64 - batch_idx * batch_size
            } else {
                batch_size
            };

            // Thread-local generator
            thread_local! {
                static GENERATOR: std::cell::RefCell<Option<AddressGenerator>> =
                    const { std::cell::RefCell::new(None) };
            }

            let mut batch_ops = 0;
            let mut batch_found = Vec::new();

            GENERATOR.with(|gen| {
                let mut gen = gen.borrow_mut();
                if gen.is_none() {
                    *gen = Some(AddressGenerator::new(config.format));
                }
                let generator = gen.as_ref().unwrap();

                for i in 0..current_batch_size {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }

                    let key_val = &batch_start + i;
                    let key_bytes_be = key_val.to_bytes_be();
                    let mut key_bytes = [0u8; 32];
                    let start_idx = 32usize.saturating_sub(key_bytes_be.len());
                    key_bytes[start_idx..]
                        .copy_from_slice(&key_bytes_be[key_bytes_be.len().saturating_sub(32)..]);

                    if let Some(addr) = generator.generate(&key_bytes) {
                        batch_ops += 1;
                        if pattern.matches(&addr.address) {
                            batch_found.push(addr);
                        }
                    }
                }
            });

            operations.fetch_add(batch_ops, Ordering::Relaxed);

            if !batch_found.is_empty() {
                let mut m = matches.lock().unwrap();
                m.extend(batch_found);
            }

            if let Some(ref cb) = progress_callback {
                cb(operations.load(Ordering::Relaxed));
            }
        });
    };

    if let Some(pool) = pool {
        pool.install(scan_loop);
    } else {
        scan_loop();
    }

    let elapsed = start_time.elapsed().as_secs_f64();
    let found_matches = matches.lock().unwrap().clone();

    ScanResult {
        matches: found_matches,
        operations: operations.load(Ordering::Relaxed),
        elapsed_secs: elapsed,
    }
}

/// Run a quick benchmark to estimate scan rate.
pub fn benchmark(format: AddressFormat, iterations: u64) -> f64 {
    let start = Instant::now();
    let generator = AddressGenerator::new(format);
    let mut rng = rand::thread_rng();

    for _ in 0..iterations {
        let mut secret = [0u8; 32];
        rng.fill(&mut secret);
        let _ = generator.generate(&secret);
    }

    let elapsed = start.elapsed().as_secs_f64();
    iterations as f64 / elapsed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_finds_match() {
        // Pattern that matches all P2PKH addresses
        let pattern = Pattern::new("^1", false).unwrap();
        let config = ScanConfig {
            format: AddressFormat::P2pkh,
            count: 1,
            threads: Some(1),
            gpu_batch_size: None,
            cpu_batch_size: None,
            start: None,
            end: None,
        };

        let result = scan(&pattern, &config);

        assert_eq!(result.matches.len(), 1);
        assert!(result.matches[0].address.starts_with('1'));
        assert!(result.operations >= 1);
    }

    #[test]
    fn test_scan_finds_multiple() {
        let pattern = Pattern::new("^1", false).unwrap();
        let config = ScanConfig {
            format: AddressFormat::P2pkh,
            count: 3,
            threads: Some(1),
            gpu_batch_size: None,
            cpu_batch_size: None,
            start: None,
            end: None,
        };

        let result = scan(&pattern, &config);

        assert_eq!(result.matches.len(), 3);
        for m in &result.matches {
            assert!(m.address.starts_with('1'));
        }
    }

    #[test]
    fn test_scan_p2wpkh() {
        let pattern = Pattern::new("^bc1q", false).unwrap();
        let config = ScanConfig {
            format: AddressFormat::P2wpkh,
            count: 1,
            threads: Some(1),
            gpu_batch_size: None,
            cpu_batch_size: None,
            start: None,
            end: None,
        };

        let result = scan(&pattern, &config);

        assert_eq!(result.matches.len(), 1);
        assert!(result.matches[0].address.starts_with("bc1q"));
    }

    #[test]
    fn test_scan_with_stop_flag() {
        let pattern = Pattern::new("^1ZZZZZZZZZZ", false).unwrap(); // Very unlikely
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = stop.clone();

        // Stop after 100ms
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(100));
            stop_clone.store(true, Ordering::Relaxed);
        });

        let config = ScanConfig {
            format: AddressFormat::P2pkh,
            count: 1,
            threads: Some(1),
            gpu_batch_size: None,
            cpu_batch_size: None,
            start: None,
            end: None,
        };

        let result = scan_with_progress(&pattern, &config, None, Some(stop));

        // Should have stopped without finding a match
        assert!(result.matches.is_empty());
        assert!(result.operations > 0);
    }

    #[test]
    fn test_scan_result_rate() {
        let result = ScanResult {
            matches: vec![],
            operations: 1000,
            elapsed_secs: 0.5,
        };

        assert!((result.rate() - 2000.0).abs() < 0.01);
    }

    #[test]
    fn test_benchmark() {
        let rate = benchmark(AddressFormat::P2pkh, 100);
        assert!(rate > 0.0);
    }

    #[test]
    fn test_default_config() {
        let config = ScanConfig::default();
        assert_eq!(config.format, AddressFormat::P2pkh);
        assert_eq!(config.count, 1);
        assert!(config.threads.is_none());
    }
}
