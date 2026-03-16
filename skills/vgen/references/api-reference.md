# vgen Library API Reference

## Core types

### AddressFormat

```rust
pub enum AddressFormat {
    P2pkh,              // 1... (Base58, compressed)
    P2pkhUncompressed,  // 1... (Base58, uncompressed)
    P2wpkh,             // bc1q... (Bech32)
    P2shP2wpkh,         // 3... (Base58)
    P2tr,               // bc1p... (Bech32m)
    Ethereum,           // 0x... (Hex, EIP-55 checksum)
}
```

### GeneratedAddress

```rust
pub struct GeneratedAddress {
    pub address: String,       // The generated address
    pub wif: String,           // WIF private key (Bitcoin) or hex (Ethereum)
    pub hex: String,           // Hex-encoded 32-byte private key
    pub format: AddressFormat,
}
```

### Pattern

```rust
pub struct Pattern { /* ... */ }

impl Pattern {
    // Compile regex. Returns Err on empty or invalid pattern.
    pub fn new(pattern: &str, case_insensitive: bool) -> anyhow::Result<Self>;

    // Test if address matches the pattern.
    pub fn matches(&self, address: &str) -> bool;

    // Returns chars in pattern that are invalid for the given format's charset.
    pub fn validate_charset(&self, format: AddressFormat) -> Vec<char>;

    // Estimate number of addresses to check before finding a match.
    pub fn estimate_difficulty(&self, format: AddressFormat) -> u64;

    pub fn original(&self) -> &str;
    pub fn is_case_insensitive(&self) -> bool;
}
```

### ScanConfig

```rust
pub struct ScanConfig {
    pub format: AddressFormat,
    pub count: usize,                // Matches to find before stopping
    pub threads: Option<usize>,      // None = all CPUs
    pub gpu_batch_size: Option<u32>, // Default: 524288
    pub cpu_batch_size: Option<usize>,
    pub start: Option<BigUint>,      // Range scan start (None = random)
    pub end: Option<BigUint>,        // Range scan end
}
```

Default: `P2pkh`, count 1, all threads, no range.

### ScanResult

```rust
pub struct ScanResult {
    pub matches: Vec<GeneratedAddress>,
    pub operations: u64,       // Total keys checked
    pub elapsed_secs: f64,
}

impl ScanResult {
    pub fn rate(&self) -> f64; // keys/sec
}
```

## Scanner functions

```rust
// One-shot CPU scan. Blocks until `count` matches found.
pub fn scan(pattern: &Pattern, config: &ScanConfig) -> ScanResult;

// CPU scan with optional progress callback and stop flag.
pub fn scan_with_progress(
    pattern: &Pattern,
    config: &ScanConfig,
    progress_callback: Option<ProgressCallback>,
    stop_flag: Option<Arc<AtomicBool>>,
) -> ScanResult;

pub type ProgressCallback = Arc<dyn Fn(u64) + Send + Sync>;

// Benchmark raw key generation speed (no pattern matching).
pub fn benchmark(format: AddressFormat, iterations: u64) -> f64;
```

## GPU API

```rust
pub struct GpuRunner { /* ... */ }

impl GpuRunner {
    // Initialize GPU with given batch size and backend preference.
    pub async fn new(batch_size: u32, backend: GpuBackend) -> anyhow::Result<Self>;

    pub fn backend(&self) -> GpuBackend;
}

// High-level GPU scan for hash-based formats (P2PKH, P2WPKH).
pub async fn scan_gpu_with_runner(
    pattern: &Pattern,
    config: &ScanConfig,
    progress_cb: Option<ProgressCallback>,
    stop: Option<Arc<AtomicBool>>,
    runner: Arc<GpuRunner>,
) -> anyhow::Result<ScanResult>;

// High-level GPU scan for P2TR (Taproot).
pub async fn scan_gpu_p2tr_with_runner(
    pattern: &Pattern,
    config: &ScanConfig,
    progress_cb: Option<ProgressCallback>,
    stop: Option<Arc<AtomicBool>>,
    runner: Arc<GpuRunner>,
) -> anyhow::Result<ScanResult>;
```

GPU is not available for Ethereum or P2SH-P2WPKH - these fall back to CPU.

## AddressGenerator

```rust
pub struct AddressGenerator { /* Secp256k1 context + network + format */ }

impl AddressGenerator {
    pub fn new(format: AddressFormat) -> Self;
    // Generate address from a 32-byte secret key. Returns None if key is invalid.
    pub fn generate(&self, secret: &[u8; 32]) -> Option<GeneratedAddress>;
}
```

## Provider module

The `provider` module resolves data provider references (`boha:collection:id`) into address and range data. This is used internally by the CLI - library users typically construct `Pattern` and `ScanConfig` directly.
