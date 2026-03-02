# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-02
**Commit:** 63bc7a1
**Branch:** main

## OVERVIEW

Bitcoin vanity address generator with regex pattern matching. GPU acceleration via wgpu (Vulkan/Metal/DX12/OpenGL). Supports P2PKH, P2WPKH, P2TR, Ethereum formats. Data providers (boha) for Bitcoin Puzzle integration.

## STRUCTURE

```
vgen/
├── src/
│   ├── lib.rs          # CLI (clap), TUI (ratatui), orchestration (1425 lines)
│   ├── main.rs         # 3-line wrapper → lib.rs
│   ├── address.rs      # Address generation (bitcoin crate)
│   ├── pattern.rs      # Regex compilation, difficulty estimation
│   ├── scanner.rs      # CPU parallel scanning (rayon)
│   ├── gpu.rs          # wgpu pipeline, double-buffered dispatch (1408 lines)
│   ├── provider.rs     # Data providers (boha puzzle library)
│   └── shaders/        # WGSL compute shaders
│       ├── generator.wgsl  # secp256k1 EC arithmetic, 5 kernels
│       ├── sha256.wgsl     # SHA-256 implementation
│       └── ripemd160.wgsl  # RIPEMD-160 implementation
├── benches/            # Criterion benchmarks (core + GPU)
├── justfile            # Build automation + release workflow
├── cliff.toml          # Changelog generation (conventional commits)
└── PKGBUILD            # Arch Linux packaging
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add address format | `src/address.rs` | AddressFormat enum, AddressGenerator::generate() |
| Modify pattern matching | `src/pattern.rs` | Pattern struct, validate_charset(), estimate_difficulty() |
| Change CLI/TUI | `src/lib.rs` | Commands enum, run_tui(), OutputFormat enum |
| CPU scanning | `src/scanner.rs` | scan_with_progress(), scan_range_cpu() |
| GPU pipeline | `src/gpu.rs` | GpuRunner struct, dispatch(), await_result() |
| Add data provider | `src/provider.rs` | resolve(), ProviderResult, build_pattern() |
| EC point math | `src/shaders/generator.wgsl` | point_add(), point_double(), scalar_mult() |
| Add output format | `src/lib.rs` | OutputFormat enum (text/json/jsonl/csv/minimal) |
| Release new version | `justfile` | `just release X.Y.Z` → git push && git push --tags |

## CODE MAP

### Core Types

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `AddressFormat` | Enum | address.rs:9 | P2pkh, P2wpkh, P2shP2wpkh, P2tr, P2pkhUncompressed, Ethereum |
| `AddressGenerator` | Struct | address.rs:74 | Key → address generation |
| `Pattern` | Struct | pattern.rs:8 | Regex + difficulty estimation |
| `GpuRunner` | Struct | gpu.rs:116 | wgpu pipeline orchestration (12 refs) |
| `GpuBackend` | Enum | gpu.rs:18 | Auto, Vulkan, Metal, Dx12, Gl |
| `ScanConfig` | Struct | scanner.rs:15 | Search parameters (format, count, threads, batch sizes, range) |
| `ScanResult` | Struct | scanner.rs:48 | Matches + stats |
| `ProviderResult` | Struct | provider.rs:6 | Resolved puzzle data (address, format, key_range) |

### Entry Points

| Function | Location | Role |
|----------|----------|------|
| `run_from_args()` | lib.rs:264 | CLI entry |
| `run()` | lib.rs:273 | Command dispatcher |
| `run_search()` | lib.rs:596 | Unified GPU/CPU search orchestration |
| `run_tui()` | lib.rs:975 | Interactive TUI (ratatui) |
| `scan_with_progress()` | scanner.rs:80 | CPU parallel scan |
| `scan_range_cpu()` | scanner.rs:210 | Range-based CPU scan |
| `scan_gpu()` | gpu.rs:1067 | GPU batch scan (P2PKH/P2WPKH) |
| `scan_gpu_p2tr()` | gpu.rs:1293 | GPU batch scan (P2TR/Taproot) |

### Module Dependency Layers

```
Layer 0: address.rs (no internal deps)
Layer 1: pattern.rs, provider.rs (→ address)
Layer 2: scanner.rs (→ address, pattern)
Layer 3: gpu.rs (→ address, pattern, scanner)
Layer 4: lib.rs (→ all, orchestration + CLI + TUI)
```

## CONVENTIONS

- **Release profile**: opt-level=3, LTO, single codegen-unit, panic=abort, stripped
- **MSRV**: Rust 1.92, edition 2021
- **GPU backend**: Vulkan preferred → OpenGL fallback (runtime `--backend` flag)
- **Batch sizes**: GPU 1M keys default (512K safer), CPU 10K per thread
- **Entry point**: main.rs is 3-line wrapper, all logic in lib.rs
- **No custom lints**: Uses default rustfmt/clippy — no rustfmt.toml, no clippy.toml
- **Tests**: Inline `#[cfg(test)]` modules, no separate tests/ dir (~30 tests across 5 modules)
- **Shaders in src/**: WGSL files under src/shaders/ (non-standard but intentional)
- **No feature flags**: All deps unconditional, GPU always compiled in
- **No unsafe**: Zero unsafe blocks — all GPU ops via safe wgpu abstractions
- **Commits**: Conventional commits format (feat/fix/refactor/etc.)

## ANTI-PATTERNS (THIS PROJECT)

- **`--tui` flag**: Deprecated, TUI now default in terminals (warns at runtime)
- **GPU + Ethereum**: Not supported, falls back to CPU silently
- **Invalid charset in pattern**: Warns but continues — pattern will NEVER match
- **Case-insensitive + Bech32**: `-i` flag redundant for bech32 (always lowercase)
- **GPU batch 1M**: Can cause GPU timeout on some hardware — use `--gpu-batch-size 524288`
- **`#[allow(dead_code)]` on jacobian_buffer**: Reserved for future batch affine inversion optimization

## UNIQUE STYLES

### WGSL Shaders
- **BigInt256**: 8×u32 limbs, little-endian (vec4<u32> pairs)
- **EC points**: Jacobian coordinates (X, Y, Z) for efficiency
- **Compressed pubkey**: 33 bytes (0x02/0x03 prefix + X coordinate)
- **5 kernels**: `init_table` (once), `main` (legacy), `compute_jacobian` + `batch_normalize_hash` (batch pipeline), `batch_normalize_p2tr` (Taproot)
- **Blelloch scan**: Parallel batch affine inversion — O(log n) vs O(n) field inversions
- **Workgroup size**: 256 threads, shared memory for prefix/suffix products

### Rust
- **Double-buffered GPU**: 2 frames overlap compute/transfer — zero GPU idle time
- **Thread-local generators**: Avoid lock contention in rayon
- **AtomicU64 for ops**: Single counter, batch updates
- **tokio oneshot**: GPU map_async completion signaling

## COMMANDS

```bash
# Development
cargo test                    # Run all tests (~30 tests)
cargo build --release         # Optimized build (slow: LTO + 1 codegen-unit)
cargo bench                   # Criterion benchmarks (core + GPU)

# Release (updates Cargo.toml, PKGBUILD, CHANGELOG.md)
just release X.Y.Z            # Then: git push && git push --tags

# Usage
vgen generate -p "^1Cat"      # Find vanity address
vgen generate -p "^1Cat" -o jsonl  # JSON Lines output
vgen estimate -p "^1CatDog"   # Estimate difficulty
vgen range --puzzle 66        # Scan Bitcoin puzzle range
vgen range -p "boha:b1000:66" # Puzzle via data provider
vgen verify -k <WIF>          # Verify private key
vgen list-gpus                # Show available GPU backends
```

## NOTES

- **Performance**: CPU ~50-200K keys/s, GPU ~500K-2M keys/s
- **Base58 charset**: Excludes 0, O, I, l (ambiguous chars)
- **Bech32 charset**: Always lowercase, excludes 1, b, i, o
- **Security**: Experimental — don't store real funds without verification
- **CI/CD**: GitHub Actions on tag push → crates.io (`crates.yml`) + AUR (`aur.yml`)
- **Output formats**: text (default), json, jsonl, csv, minimal (WIF only)
- **Data providers**: `boha:collection:id` resolves puzzle addresses + key ranges
