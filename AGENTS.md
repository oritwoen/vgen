# vgen — Agent Playbook

Bitcoin vanity address generator. Regex patterns, GPU acceleration (wgpu), multiple address formats. Rust, ~4600 LOC, 72 tests.

## Quick Commands

```bash
cargo test                         # 72 tests, ~0.1s
cargo test pattern::tests          # run one module's tests
cargo test test_validate_charset   # run tests matching name
cargo clippy -- -D warnings        # lint (pedantic + nursery enabled)
cargo build --release              # optimized (LTO, single codegen-unit, stripped)
cargo bench                        # Criterion benchmarks (core + GPU)
just release X.Y.Z                 # bump version, changelog, tag (then git push && git push --tags)
```

## Codebase Map

```
src/
├── main.rs          # 3-line wrapper → lib.rs
├── lib.rs           # CLI (clap), TUI (ratatui), orchestration (1621 lines)
├── address.rs       # AddressFormat enum, AddressGenerator (key → address)
├── pattern.rs       # Regex compilation, charset validation, difficulty estimation
├── scanner.rs       # CPU parallel scanning (rayon), ScanConfig/ScanResult
├── gpu.rs           # wgpu pipeline, double-buffered dispatch (1460 lines)
├── provider.rs      # Data providers (boha puzzle library)
└── shaders/         # WGSL compute shaders (split into modules)
    ├── field.wgsl       # secp256k1 field arithmetic, EC point operations
    ├── init.wgsl        # Precomputation table init kernel
    ├── search.wgsl      # P2PKH/P2WPKH/P2SH-P2WPKH batch search kernels
    ├── search_p2tr.wgsl # P2TR (Taproot) batch search kernels
    ├── sha256.wgsl      # SHA-256
    └── ripemd160.wgsl   # RIPEMD-160
benches/
├── core_bench.rs    # CPU benchmarks
└── gpu_bench.rs     # GPU benchmarks
```

### Module Dependencies

```
Layer 0: address.rs (no internal deps)
Layer 1: pattern.rs, provider.rs (→ address)
Layer 2: scanner.rs (→ address, pattern)
Layer 3: gpu.rs (→ address, pattern, scanner)
Layer 4: lib.rs (→ all, orchestration + CLI + TUI)
```

### Where to Put Things

| Task | File | Key types/functions |
|------|------|---------------------|
| Add address format | `address.rs` | `AddressFormat` enum, `AddressGenerator::generate()` |
| CLI flags/subcommands | `lib.rs` | `Commands` enum, clap derive macros |
| Pattern matching | `pattern.rs` | `Pattern`, `validate_charset()`, `estimate_difficulty()` |
| CPU scanning | `scanner.rs` | `scan_with_progress()`, `scan_range_cpu()` |
| GPU pipeline | `gpu.rs` | `GpuRunner`, `dispatch()`, `await_result()` |
| Data providers | `provider.rs` | `resolve()`, `ProviderResult` |
| Output formats | `lib.rs` | `OutputFormat` enum (text/json/jsonl/csv/minimal) |
| New shader kernel | `src/shaders/` | WGSL, add module load in `gpu.rs` |

## Code Conventions

**Rust:**
- Edition 2021, MSRV 1.92
- `unsafe` is `forbid` — zero unsafe blocks, all GPU via safe wgpu abstractions
- Clippy: `all = deny`, `pedantic = warn`, `nursery = warn` (with specific allows in Cargo.toml)
- Tests: inline `#[cfg(test)]` modules, no separate `tests/` directory
- Conventional commits: `feat`, `fix`, `refactor`, `chore`, `test`, `docs`
- No feature flags — all deps unconditional, GPU always compiled in

**GPU/Shaders:**
- WGSL shaders under `src/shaders/` (non-standard location, intentional)
- Shaders split into modules: field math, init, search (P2PKH/P2WPKH/P2SH-P2WPKH), search_p2tr
- `gpu.rs` loads each shader module separately and creates pipelines per format
- Double-buffered GPU dispatch (2 frames overlap compute/transfer)
- Workgroup size: 256 threads, shared memory for Blelloch scan

**Batch sizes:**
- GPU default: 524,288 (was 1M, reduced to prevent context loss on some hardware)
- CPU default: 10,000 per thread

## Known Constraints

- **Ethereum + GPU**: not supported, falls back to CPU silently
- **`-i` (case insensitive)**: works for Base58 formats (P2PKH, P2SH-P2WPKH) and Ethereum; redundant for Bech32 (always lowercase)
- **`--tui` flag**: deprecated, TUI is now default in terminals
- **Invalid charset in pattern**: warns but continues — pattern will never match
- **`--prefix-length 0`**: rejected for provider patterns (would match everything)

## Execution Workflow

1. **Explore** — read relevant source files before changing anything
2. **Plan** — if the change touches multiple modules, sketch the approach
3. **Edit** — make focused changes, one concern per commit
4. **Verify** — `cargo test && cargo clippy -- -D warnings` after every change
5. **Don't commit unless asked** — local verification is always fine, pushing requires ask

## Git Hygiene

- Don't force-push, reset --hard, or delete branches without asking
- Don't commit secrets, .env files, or generated artifacts
- Conventional commits: scope matches the module (`fix(gpu):`, `feat(pattern):`, etc.)
- Release flow: `just release X.Y.Z` then `git push && git push --tags`

## CI/CD

- `crates.yml` — publishes to crates.io on tag push
- `aur.yml` — updates Arch Linux AUR package on tag push
- No PR CI — verify locally with `cargo test && cargo clippy -- -D warnings`

## Communication Style

- Direct and technical. Say what changed and why.
- Skip filler, don't restate the question.
- When something is uncertain, say so.
