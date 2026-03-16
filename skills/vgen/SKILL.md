---
name: vgen
description: Bitcoin vanity address generator with regex pattern matching and GPU acceleration. Use when generating Bitcoin/Ethereum vanity addresses, scanning key ranges for Bitcoin Puzzles, or verifying private keys against addresses. Supports P2PKH, P2WPKH, P2SH-P2WPKH, P2TR, and Ethereum formats.
metadata:
  author: oritwoen
  version: "0.2.0"
---

# vgen

Bitcoin vanity address generator. Finds private keys whose derived addresses match regex patterns. GPU-accelerated via wgpu (Vulkan/Metal/DX12/GL), falls back to CPU automatically.

Supports P2PKH (`1...`), P2WPKH (`bc1q...`), P2SH-P2WPKH (`3...`), P2TR (`bc1p...`), and Ethereum (`0x...`).

## Using the CLI

### Generate vanity address

```bash
# P2PKH starting with "1Cat"
vgen generate -p "^1Cat"

# Case insensitive (Base58 + Ethereum)
vgen generate -p "^1cat" -i

# Bech32 ending with "dead"
vgen generate -p "dead$" -f p2wpkh

# Ethereum
vgen generate -p "^0xdead" -f ethereum

# Find 5 matches, JSON output to file
vgen generate -p "^1Cat" -c 5 -o jsonl --file results.jsonl

# CPU only
vgen generate -p "^1Cat" --no-gpu
```

### Scan key range (Bitcoin Puzzles)

```bash
# Manual hex range
vgen range -r "2000:3FFF"

# Auto range from puzzle number
vgen range --puzzle 66

# With data provider (resolves address + range automatically)
vgen range -p "boha:b1000:66"

# Prefix matching with provider
vgen range -p "boha:b1000:66" -l 8

# Scan entire range (no early stop)
vgen range --puzzle 66 -c 0
```

### Verify private key

```bash
# From WIF
vgen verify -k "5HueCGU8rMjxEXxiPuD5BDku4MkFqeZyd4dZ1jvhTVqvbTLvyTJ"

# From hex
vgen verify -k "0c28fca386c7a227600b2fe50b7cae11ec86d3bf1fbe471be89827e19d72aa1d"

# Verify against expected address
vgen verify -k "KEY" -a "1GAehh7TsJAHuUAeKZcXf5CnwuGuGgyX2S"
```

### Estimate difficulty

```bash
vgen estimate -p "^1Cat" -f p2pkh
```

### Common flags

| Flag | Description |
|------|-------------|
| `-p, --pattern` | Regex pattern or data provider ref (`boha:collection:id`) |
| `-f, --format` | `p2pkh`, `p2wpkh`, `p2sh-p2wpkh`, `p2tr`, `ethereum` |
| `-o, --output` | `text`, `json`, `jsonl`, `csv`, `minimal` (just WIF) |
| `--file` | Write output to file instead of stdout |
| `-c, --count` | Stop after N matches (default: 1, 0 = unlimited) |
| `-t, --threads` | CPU thread count (default: all) |
| `--no-gpu` | Force CPU-only |
| `--gpu-batch-size` | GPU batch size (default: 524288) |
| `--backend` | `auto`, `vulkan`, `metal`, `dx12`, `gl` |
| `-i, --ignore-case` | Case insensitive (Base58 + Ethereum, generate/estimate) |
| `-l, --prefix-length` | Use first N chars of provider address as prefix |
| `--no-tui` | Disable interactive TUI (range only) |

## Pattern syntax

Standard Rust regex. Patterns match against the full generated address.

| Pattern | Matches |
|---------|---------|
| `^1Cat` | Address starting with `1Cat` |
| `dead$` | Address ending with `dead` |
| `^1[Cc]at` | `1Cat` or `1cat` |
| `^1.*dead$` | Starts with `1`, ends with `dead` |

Be aware of charset constraints per format - P2PKH uses Base58 (no `0`, `O`, `I`, `l`), Bech32 is lowercase only.

## Data providers

Pattern can be a provider reference instead of regex. Format: `provider:collection:id`.

Currently supported:
- `boha:collection:id` - [boha](https://github.com/oritwoen/boha) puzzle library

In `range` mode, providers auto-resolve both the target address (as regex) and the key range. With `-l N`, only the first N characters become the prefix pattern.

## Using the Library API

### Install

```bash
cargo add vgen
```

### Key types

```rust
use vgen::{
    AddressFormat,       // P2pkh, P2wpkh, P2shP2wpkh, P2tr, Ethereum
    AddressGenerator,    // Generates addresses from secret keys
    GeneratedAddress,    // Result: address + WIF + format
    Pattern,             // Compiled regex pattern
    ScanConfig,          // Scanner configuration
    ScanResult,          // Scan results with matches and stats
    scan,                // One-shot CPU scan
    scan_with_progress,  // CPU scan with progress callback
    GpuRunner,           // GPU scanner
    GpuBackend,          // Vulkan/Metal/DX12/GL/Auto
};
```

### CPU scan

```rust
use vgen::{Pattern, ScanConfig, AddressFormat, scan};

let pattern = Pattern::new("^1Cat", false)?;
let config = ScanConfig {
    format: AddressFormat::P2pkh,
    count: 1,
    ..ScanConfig::default()
};
let result = scan(&pattern, &config);
for addr in &result.matches {
    println!("{} {} {}", addr.address, addr.wif, addr.hex);
}
```

### Address generation

```rust
use vgen::{AddressFormat, AddressGenerator};

let generator = AddressGenerator::new(AddressFormat::P2pkh);
let secret = [1u8; 32]; // your 32-byte secret key
let addr = generator.generate(&secret).expect("valid key");
// addr.address, addr.wif, addr.hex, addr.format
```

For full type definitions and GPU API details, see [api-reference.md](references/api-reference.md).

## References

| When | Read |
|------|------|
| Full type definitions, GPU API, ScanConfig fields | [api-reference.md](references/api-reference.md) |

## Limitations

- GPU acceleration works for P2PKH, P2WPKH, and P2TR. Other formats fall back to CPU automatically.
- Case-insensitive matching (`-i`) works with Base58 formats (P2PKH, P2SH-P2WPKH) and Ethereum (EIP-55 mixed-case) in `generate` and `estimate` modes. Bech32 addresses are always lowercase, so `-i` is redundant for P2WPKH/P2TR.
- `--prefix-length` must be at least 1 when used.
- This is experimental software. Do not use generated keys for real funds without thorough verification.
