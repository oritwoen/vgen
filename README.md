# vgen

[![Crates.io](https://img.shields.io/crates/v/vgen?style=flat&colorA=130f40&colorB=474787)](https://crates.io/crates/vgen)
[![Downloads](https://img.shields.io/crates/d/vgen?style=flat&colorA=130f40&colorB=474787)](https://crates.io/crates/vgen)
[![License](https://img.shields.io/crates/l/vgen?style=flat&colorA=130f40&colorB=474787)](LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/oritwoen/vgen)

Bitcoin vanity address generator with regex pattern matching and GPU acceleration.

> [!WARNING]
> This project is experimental. Do not use generated keys for storing real funds without thorough verification. Always prefer hardware wallets or well-established software for production use.

## Features

- Generate Bitcoin addresses matching custom regex patterns
- Support for multiple address formats:
  - P2PKH (1...)
  - P2WPKH / Bech32 (bc1q...)
  - Ethereum (0x...)
- GPU acceleration via wgpu (Vulkan/Metal/DX12/OpenGL backends)
- Parallel CPU scanning with rayon
- Interactive TUI with real-time statistics
- Range scanning for Bitcoin Puzzle challenges
- Data providers for puzzle/bounty integration (boha)
- JSON and minimal output formats

## Installation

### From crates.io

```bash
cargo install vgen
```

### Arch Linux (AUR)

```bash
paru -S vgen
```

### From source

```bash
cargo install --path .
```

## Usage

### Generate vanity address

```bash
# Find address starting with "1Cat"
vgen generate -p "^1Cat"

# Case insensitive matching
vgen generate -p "^1cat" -i

# Bech32 address ending with "dead"
vgen generate -p "dead$" -f p2wpkh

# Ethereum address
vgen generate -p "^0xdead" -f ethereum

# CPU only (GPU is enabled by default)
vgen generate -p "^1Cat" --no-gpu

# Find multiple matches
vgen generate -p "^1Cat" -c 5
```

### Estimate difficulty

```bash
vgen estimate -p "^1CatDog"
```

### Scan key range (Bitcoin Puzzles)

```bash
# Scan puzzle #66 range
vgen range --puzzle 66 -p "."

# Custom range
vgen range -r "20000000000000000:3FFFFFFFFFFFFFFFF"
```

### Data providers

Use external data sources for pattern generation:

```bash
# Vanity with puzzle address prefix (6 characters)
vgen generate -p "boha:b1000:66" -l 6
# → Resolves to pattern: ^13zb1h

# Puzzle solving (exact address match + auto key range)
vgen range -p "boha:b1000:66"
# → Range: 2^65..2^66-1
# → Pattern: ^13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so$

# With prefix matching in range mode
vgen range -p "boha:b1000:66" -l 8
```

Supported providers:
- `boha:collection:id` - [boha](https://github.com/oritwoen/boha) puzzle library (b1000, gsmg, bitaps, etc.)

### Verify private key

```bash
# From WIF
vgen verify -k "5HueCGU8rMjxEXxiPuD5BDku4MkFqeZyd4dZ1jvhTVqvbTLvyTJ"

# From hex
vgen verify -k "0c28fca386c7a227600b2fe50b7cae11ec86d3bf1fbe471be89827e19d72aa1d"

# Verify against expected address
vgen verify -k "5HueCGU8rMjxEXxiPuD5BDku4MkFqeZyd4dZ1jvhTVqvbTLvyTJ" -a "1GAehh7TsJAHuUAeKZcXf5CnwuGuGgyX2S"
```

## Output formats

```bash
# Default text output
vgen generate -p "^1Cat" -o text

# JSON output (pretty-printed)
vgen generate -p "^1Cat" -o json

# JSON Lines (one JSON object per line, for data pipelines)
vgen generate -p "^1Cat" -o jsonl

# CSV (with header, for data catalogs/Iceberg)
vgen generate -p "^1Cat" -o csv

# Minimal (just WIF)
vgen generate -p "^1Cat" -o minimal

# Write to file
vgen generate -p "^1Cat" -o jsonl --file results.jsonl
vgen generate -p "^1Cat" -o csv --file results.csv
```

## TUI

![vgen TUI](vgen.png)

The interactive TUI is enabled by default in terminal sessions. Disable with `--no-tui`.

Features:
- Real-time hashrate display
- Performance sparkline chart
- Luck indicator
- Found matches list

## Performance

GPU acceleration is enabled by default and falls back to CPU if no compatible GPU is found.

- CPU: ~50,000-200,000 keys/sec (depends on CPU)
- GPU: ~500,000-2,000,000 keys/sec (depends on GPU)

## Pattern syntax

Patterns use Rust regex syntax:

| Pattern | Description |
|---------|-------------|
| `^1Cat` | Starts with "1Cat" |
| `dead$` | Ends with "dead" |
| `^1[Cc]at` | Starts with "1Cat" or "1cat" |
| `^1.*dead$` | Starts with "1", ends with "dead" |

## Comparison

| Project | Language | GPU | Patterns | Notes |
|---------|----------|-----|----------|-------|
| **vgen** | Rust | wgpu (Vulkan/Metal/DX12) | regex | TUI, range scanning, memory safe |
| [VanitySearch](https://github.com/JeanLucPons/VanitySearch) | C++ | CUDA | prefix | Fastest (~7 Gkeys/s), NVIDIA only |
| [vanitygen-plusplus](https://github.com/10gic/vanitygen-plusplus) | C++ | OpenCL | prefix/regex | 100+ cryptocurrencies |
| [btc-vanity](https://github.com/Emivvvvv/btc-vanity) | Rust | - | prefix/regex | BTC/ETH/SOL, CPU only |
| [nakatoshi](https://github.com/ndelvalle/nakatoshi) | Rust | - | prefix | Simple, prefix only |
| [supervanitygen](https://github.com/klynastor/supervanitygen) | C | - | prefix | ASM optimizations (AVX2/SHA-NI) |
| [vanitygen](https://github.com/samr7/vanitygen) | C | - | prefix/regex | Classic, unmaintained |

**Why vgen?**
- Cross-platform GPU via wgpu (not locked to NVIDIA CUDA)
- Memory safe Rust implementation
- Full regex pattern support
- Interactive TUI with real-time statistics
- Bitcoin Puzzle range scanning built-in

## Roadmap

- [ ] More cryptocurrencies (Litecoin, Dogecoin, Solana, etc.)
- [ ] GPU performance improvements (CUDA backend, shader optimizations)
- [ ] More data providers (mempool, blockchair, etc.)

## Security

Generated private keys are cryptographically secure random numbers. Always verify the generated key produces the expected address before use.

## License

MIT
