# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-04
**Commit:** 3b754bf
**Branch:** main

## OVERVIEW

Bitcoin vanity address generator with regex pattern matching. GPU acceleration via wgpu (Vulkan/Metal/DX12/OpenGL). Supports P2PKH, P2WPKH, P2TR, Ethereum formats.

## STRUCTURE

```
vgen/
├── src/
│   ├── lib.rs          # CLI (clap), TUI (ratatui), orchestration
│   ├── main.rs         # Thin wrapper → lib.rs
│   ├── address.rs      # Address generation (bitcoin crate)
│   ├── pattern.rs      # Regex compilation, difficulty estimation
│   ├── scanner.rs      # CPU parallel scanning (rayon)
│   ├── gpu.rs          # wgpu pipeline, buffer management
│   └── shaders/        # WGSL compute shaders
│       ├── generator.wgsl  # secp256k1 EC arithmetic, main kernel
│       ├── sha256.wgsl     # SHA-256 implementation
│       └── ripemd160.wgsl  # RIPEMD-160 implementation
├── benches/            # Criterion benchmarks
├── justfile            # Build automation
├── cliff.toml          # Changelog generation config
└── PKGBUILD            # Arch Linux packaging
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add address format | `src/address.rs` | AddressFormat enum, AddressGenerator::generate() |
| Modify pattern matching | `src/pattern.rs` | Pattern struct, validate_charset(), estimate_difficulty() |
| Change CLI/TUI | `src/lib.rs` | Clap Commands enum, run_tui(), output formatting |
| CPU scanning | `src/scanner.rs` | scan_with_progress(), rayon parallel batches |
| GPU pipeline | `src/gpu.rs` | GpuRunner struct, dispatch(), await_result() |
| EC point math | `src/shaders/generator.wgsl` | point_add(), point_double(), scalar_mult() |
| Release new version | `justfile` | `just release X.Y.Z` |

## CODE MAP

### Core Types

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `AddressFormat` | Enum | address.rs:9 | P2pkh, P2wpkh, P2shP2wpkh, P2tr, Ethereum |
| `AddressGenerator` | Struct | address.rs:52 | Key → address generation |
| `Pattern` | Struct | pattern.rs:8 | Regex + difficulty estimation |
| `GpuRunner` | Struct | gpu.rs:45 | wgpu pipeline orchestration |
| `ScanConfig` | Struct | scanner.rs:15 | Search parameters |
| `ScanResult` | Struct | scanner.rs:48 | Matches + stats |

### Entry Points

| Function | Location | Role |
|----------|----------|------|
| `run_from_args()` | lib.rs:236 | CLI entry |
| `run()` | lib.rs:245 | Command dispatcher |
| `run_search()` | lib.rs:451 | Unified GPU/CPU search |
| `run_tui()` | lib.rs:742 | Interactive TUI |
| `scan_with_progress()` | scanner.rs:80 | CPU parallel scan |
| `scan_gpu()` | gpu.rs:636 | GPU batch scan |

## CONVENTIONS

- **Release profile**: LTO, single codegen-unit, panic=abort, stripped
- **GPU backend**: Vulkan preferred → OpenGL fallback
- **Batch sizes**: GPU 1M keys, CPU 10K per thread
- **Entry point**: main.rs is 4-line wrapper, all logic in lib.rs
- **No custom lints**: Uses default rustfmt/clippy
- **Tests**: Inline `#[cfg(test)]` modules, no separate tests/ dir
- **Shaders in src/**: WGSL files under src/shaders/ (non-standard but intentional)

## ANTI-PATTERNS (THIS PROJECT)

- **`--tui` flag**: Deprecated, TUI now default in terminals
- **GPU + Ethereum**: Not supported yet, falls back to CPU silently
- **Invalid charset in pattern**: Warns but continues (will never match)
- **Case-insensitive + Bech32**: `-i` flag redundant for bech32 (always lowercase)

## UNIQUE STYLES

### WGSL Shaders
- **BigInt256**: 8×u32 limbs, little-endian
- **EC points**: Jacobian coordinates (X, Y, Z) for efficiency
- **Compressed pubkey**: 33 bytes (0x02/0x03 prefix + X coordinate)
- **Two kernels**: `init_table` (once) + `main` (per batch)

### Rust
- **Double-buffered GPU**: 2 frames overlap compute/transfer
- **Thread-local generators**: Avoid lock contention in rayon
- **AtomicU64 for ops**: Single counter, batch updates

## COMMANDS

```bash
# Development
cargo test                    # Run tests
cargo build --release         # Optimized build
cargo bench                   # Run benchmarks

# Release (updates Cargo.toml, PKGBUILD, CHANGELOG.md)
just release X.Y.Z            # Then: git push && git push --tags

# Usage
vgen generate -p "^1Cat"      # Find vanity address
vgen estimate -p "^1CatDog"   # Estimate difficulty
vgen range --puzzle 66        # Scan Bitcoin puzzle range
vgen verify -k <WIF>          # Verify private key
```

## NOTES

- **Performance**: CPU ~50-200K keys/s, GPU ~500K-2M keys/s
- **Base58 charset**: Excludes 0, O, I, l (ambiguous chars)
- **Bech32 charset**: Always lowercase, excludes 1, b, i, o
- **Security warning**: Experimental - don't store real funds without verification
- **CI/CD**: GitHub Actions → crates.io + AUR on tag push
