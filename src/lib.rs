//! Bitcoin vanity address generator CLI.

pub mod address;
pub mod gpu;
pub mod pattern;
pub mod provider;
pub mod scanner;

pub use address::{AddressFormat, AddressGenerator, GeneratedAddress};
pub use gpu::{scan_gpu_with_runner, GpuRunner};
pub use pattern::Pattern;
pub use scanner::{benchmark, scan, scan_with_progress, ProgressCallback, ScanConfig, ScanResult};

use std::fs::File;
use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use crossterm::event::{self, Event, KeyCode};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use indicatif::{ProgressBar, ProgressStyle};
use num_bigint::BigUint;
use num_traits::{Num, One};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, List, ListItem, Paragraph, Sparkline};
use ratatui::Terminal;
use serde::Serialize;


#[derive(Parser)]
#[command(name = "vgen")]
#[command(about = "Bitcoin vanity address generator with regex pattern matching")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate vanity address matching a pattern
    Generate {
        /// Regex pattern to match (e.g., "^1Cat", "^bc1q.*dead$")
        /// Or data provider reference (e.g., "boha:b1000:66")
        #[arg(short, long)]
        pattern: String,

        /// When pattern is a data provider, use first N characters of address as prefix
        #[arg(short = 'l', long)]
        prefix_length: Option<usize>,

        /// Address format: p2pkh (1...) or p2wpkh (bc1q...)
        #[arg(short, long, default_value = "p2pkh")]
        format: Format,

        /// Case insensitive matching (P2PKH only)
        #[arg(short = 'i', long)]
        ignore_case: bool,

        /// Number of threads (default: all CPUs)
        #[arg(short, long)]
        threads: Option<usize>,

        /// Disable GPU acceleration (use CPU only)
        #[arg(long)]
        no_gpu: bool,

        /// GPU batch size (default: 1,048,576)
        #[arg(long, default_value = "1048576")]
        gpu_batch_size: u32,

        /// CPU batch size per thread (default: 10,000)
        #[arg(long, default_value = "10000")]
        cpu_batch_size: u64,

        /// Enable terminal UI (deprecated, now default)
        #[arg(long)]
        tui: bool,

        /// Disable terminal UI
        #[arg(long)]
        no_tui: bool,

        /// Output format: text, json, jsonl, csv, or minimal (just WIF)
        #[arg(short, long, default_value = "text")]
        output: OutputFormat,

        /// Write output to file instead of stdout
        #[arg(long)]
        file: Option<PathBuf>,

        /// Stop after finding N matches (default: 1)
        #[arg(short, long, default_value = "1")]
        count: usize,

        /// Repeat the search this many times (non-TUI only; useful for perf testing)
        #[arg(long, default_value_t = 1)]
        repeat: usize,

        /// Quiet mode - minimal output
        #[arg(short, long)]
        quiet: bool,
    },

    /// Estimate difficulty of a pattern (dry run)
    Estimate {
        /// Regex pattern to analyze
        #[arg(short, long)]
        pattern: String,

        /// Address format
        #[arg(short, long, default_value = "p2pkh")]
        format: Format,

        /// Case insensitive
        #[arg(short = 'i', long)]
        ignore_case: bool,
    },

    /// Scan a specific range of keys (e.g., for Bitcoin Puzzles)
    Range {
        /// Start and End HEX keys separated by ':' (e.g., 2000:3FFF)
        #[arg(short, long)]
        range: Option<String>,

        /// Puzzle number (automatically sets range)
        #[arg(long)]
        puzzle: Option<u32>,

        /// Regex pattern to match (optional, matches everything by default if not provided)
        /// Or data provider reference (e.g., "boha:b1000:66")
        #[arg(short, long)]
        pattern: Option<String>,

        /// When pattern is a data provider, use first N characters of address as prefix
        #[arg(short = 'l', long)]
        prefix_length: Option<usize>,

        /// Address format: p2pkh (1...) or p2wpkh (bc1q...)
        #[arg(short, long, default_value = "p2pkh")]
        format: Format,

        /// Number of threads (default: all CPUs)
        #[arg(short, long)]
        threads: Option<usize>,

        /// Disable GPU acceleration (use CPU only)
        #[arg(long)]
        no_gpu: bool,

        /// GPU batch size (default: 1,048,576)
        #[arg(long, default_value = "1048576")]
        gpu_batch_size: u32,

        /// Stop after finding N matches (0 = scan entire range)
        #[arg(short, long, default_value_t = 1)]
        count: usize,

        /// Repeat the search this many times (non-TUI only; useful for perf testing)
        #[arg(long, default_value_t = 1)]
        repeat: usize,

        /// Disable terminal UI
        #[arg(long)]
        no_tui: bool,

        /// Output format: text, json, jsonl, csv, or minimal (just WIF)
        #[arg(short, long, default_value = "text")]
        output: OutputFormat,

        /// Write output to file instead of stdout
        #[arg(long)]
        file: Option<PathBuf>,
    },

    /// Verify a private key produces expected address
    Verify {
        /// Private key (WIF or hex)
        #[arg(short, long)]
        key: String,

        /// Expected address (optional)
        #[arg(short, long)]
        address: Option<String>,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum Format {
    P2pkh,
    P2wpkh,
    Ethereum,
}

impl From<Format> for AddressFormat {
    fn from(f: Format) -> Self {
        match f {
            Format::P2pkh => AddressFormat::P2pkh,
            Format::P2wpkh => AddressFormat::P2wpkh,
            Format::Ethereum => AddressFormat::Ethereum,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
    Jsonl,
    Csv,
    Minimal,
}

#[derive(Serialize, Clone)]
struct VanityResult {
    address: String,
    wif: String,
    private_key_hex: String,
    format: String,
    pattern: String,
    operations: u64,
    elapsed_secs: f64,
    rate: f64,
}

#[derive(Default, Clone)]
struct TuiState {
    pattern: String,
    format: String,
    difficulty: u64,
    operations: u64,
    elapsed: f64,
    rate: f64,
    matches: Vec<VanityResult>,
    done: bool,
    gpu_enabled: bool,
    tui_status_message: String,
    gpu_device_name: String,
}

pub fn run_from_args<I, S>(args: I) -> Result<()>
where
    I: IntoIterator<Item = S>,
    S: Into<std::ffi::OsString> + Clone,
{
    let cli = Cli::parse_from(args);
    run(cli)
}

pub(crate) fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Generate {
            pattern,
            prefix_length,
            format,
            ignore_case,
            threads,
            no_gpu,
            gpu_batch_size,
            cpu_batch_size,
            tui,
            no_tui,
            output,
            file,
            count,
            repeat,
            quiet,
        } => {
            let (resolved_pattern, addr_format) = resolve_pattern_and_format(
                &pattern,
                prefix_length,
                format.into(),
            )?;

            if tui {
                eprintln!("Warning: --tui is deprecated. TUI is now enabled by default in interactive terminals.");
            }

            let is_tty = std::io::stdout().is_terminal();
            let use_tui = !no_tui && is_tty;

            if ignore_case && matches!(format, Format::P2wpkh | Format::Ethereum) {
                eprintln!("Warning: Bech32/Ethereum addresses case sensitivity handling is specific. -i flag might be redundant.");
            }

            // GPU is default, but not supported for Ethereum yet
            let use_gpu = !no_gpu && !matches!(addr_format, AddressFormat::Ethereum);
            if !no_gpu && matches!(addr_format, AddressFormat::Ethereum) {
                eprintln!("Warning: GPU acceleration not yet supported for Ethereum. Falling back to CPU.");
            }

            let config = ScanConfig {
                format: addr_format,
                count,
                threads,
                gpu_batch_size: if use_gpu { Some(gpu_batch_size) } else { None },
                cpu_batch_size: Some(cpu_batch_size as usize),
                start: None,
                end: None,
            };

            let repeat = if repeat == 0 { 1 } else { repeat };

            run_search(&resolved_pattern, ignore_case, config, use_gpu, use_tui, quiet, output, file, repeat)
        }

        Commands::Estimate {
            pattern,
            format,
            ignore_case,
        } => {
            let addr_format: AddressFormat = format.into();
            let pat = Pattern::new(&pattern, ignore_case).context("Failed to compile pattern")?;

            let difficulty = pat.estimate_difficulty(addr_format);

            // Quick benchmark
            let iterations = 10_000u64;
            let rate = benchmark(addr_format, iterations);

            let expected_secs = difficulty as f64 / rate;

            println!("Pattern: {}", pattern);
            println!(
                "Format: {}",
                match format {
                    Format::P2pkh => "P2PKH (1...)",
                    Format::P2wpkh => "P2WPKH (bc1q...)",
                    Format::Ethereum => "Ethereum (0x...)",
                }
            );
            println!("Case insensitive: {}", ignore_case);
            println!();
            println!("Estimated difficulty: 1 in {}", difficulty);
            println!("Benchmark rate: {:.0} addr/sec", rate);
            println!("Expected time: {}", format_duration(expected_secs));
            Ok(())
        }

        Commands::Verify {
            key,
            address,
        } => {
            use bitcoin::key::Secp256k1;
            use bitcoin::secp256k1::SecretKey;
            use bitcoin::{Address, CompressedPublicKey, Network, PrivateKey, PublicKey};

            let secp = Secp256k1::new();

            // Try to parse as WIF first, then as hex
            let (secret_key, is_wif) = if let Ok(pk) = key.parse::<PrivateKey>() {
                (pk.inner, true)
            } else {
                let hex_key = key.trim_start_matches("0x");
                let bytes = hex::decode(hex_key).context("Invalid key format (not WIF or hex)")?;
                if bytes.len() != 32 {
                    anyhow::bail!("Hex key must be 32 bytes");
                }
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&bytes);
                (SecretKey::from_slice(&arr)?, false)
            };

            let private_key = PrivateKey::new(secret_key, Network::Bitcoin);
            let public_key = PublicKey::from_private_key(&secp, &private_key);

            let p2pkh_addr = Address::p2pkh(&public_key, Network::Bitcoin);
            let p2wpkh_addr = CompressedPublicKey::try_from(public_key)
                .ok()
                .map(|cpk| Address::p2wpkh(&cpk, Network::Bitcoin));

            let wif_str = if is_wif {
                key.clone()
            } else {
                private_key.to_wif()
            };
            println!("Private key: {}", wif_str);
            println!("Hex: {}", hex::encode(secret_key.secret_bytes()));
            println!();
            println!("P2PKH address:  {}", p2pkh_addr);
            if let Some(ref addr) = p2wpkh_addr {
                println!("P2WPKH address: {}", addr);
            }

            if let Some(expected) = address {
                let p2pkh_match = p2pkh_addr.to_string() == expected;
                let p2wpkh_match = p2wpkh_addr
                    .as_ref()
                    .map(|a| a.to_string() == expected)
                    .unwrap_or(false);

                if p2pkh_match || p2wpkh_match {
                    println!("\nMATCH!");
                } else {
                    println!("\nMISMATCH! Expected: {}", expected);
                }
            }
            Ok(())
        }

        Commands::Range {
            range,
            puzzle,
            pattern,
            prefix_length,
            format,
            threads,
            no_gpu,
            gpu_batch_size,
            count,
            repeat,
            no_tui,
            output,
            file,
        } => {
            let pattern_str = pattern.unwrap_or_else(|| ".".to_string());

            let (start_key, end_key, resolved_pattern, addr_format) =
                resolve_range_params(&pattern_str, prefix_length, format.into(), range, puzzle)?;

            let count = if count == 0 { usize::MAX } else { count };
            let repeat = if repeat == 0 { 1 } else { repeat };

            // GPU is default
            let use_gpu = !no_gpu;

            // Assume TUI if interactive
            let is_tty = std::io::stdout().is_terminal();
            let use_tui = !no_tui && is_tty;

            let config = ScanConfig {
                format: addr_format,
                count,
                threads,
                gpu_batch_size: if use_gpu { Some(gpu_batch_size) } else { None },
                cpu_batch_size: None,
                start: Some(start_key),
                end: Some(end_key),
            };

            run_search(&resolved_pattern, false, config, use_gpu, use_tui, false, output, file, repeat)
        }
    }
}

fn resolve_pattern_and_format(
    pattern: &str,
    prefix_length: Option<usize>,
    default_format: AddressFormat,
) -> Result<(String, AddressFormat)> {
    if let Some(provider_result) = provider::resolve(pattern)? {
        let prefix_len = prefix_length.ok_or_else(|| {
            anyhow::anyhow!(
                "Provider pattern '{}' requires --prefix-length (-l) to specify how many \
                 characters of address '{}' to match",
                pattern,
                provider_result.address
            )
        })?;

        let resolved = provider::build_pattern(&provider_result, prefix_len);

        eprintln!(
            "Provider: {} → {} → pattern '{}'",
            pattern, provider_result.address, resolved
        );

        Ok((resolved, provider_result.format))
    } else {
        if prefix_length.is_some() {
            eprintln!("Warning: --prefix-length is ignored for regex patterns");
        }
        Ok((pattern.to_string(), default_format))
    }
}

fn resolve_range_params(
    pattern: &str,
    prefix_length: Option<usize>,
    default_format: AddressFormat,
    range: Option<String>,
    puzzle: Option<u32>,
) -> Result<(BigUint, BigUint, String, AddressFormat)> {
    if let Some(provider_result) = provider::resolve(pattern)? {
        let resolved_pattern = if let Some(len) = prefix_length {
            let pat = provider::build_pattern(&provider_result, len);
            eprintln!(
                "Provider: {} → {} → pattern '{}'",
                pattern, provider_result.address, pat
            );
            pat
        } else {
            let pat = provider::build_exact_pattern(&provider_result);
            eprintln!(
                "Provider: {} → {} → exact match",
                pattern, provider_result.address
            );
            pat
        };

        let (start, end) = if range.is_some() || puzzle.is_some() {
            parse_explicit_range(range, puzzle)?
        } else {
            provider_result.key_range.ok_or_else(|| {
                anyhow::anyhow!(
                    "Provider '{}' has no key range. Use --range or --puzzle to specify range.",
                    pattern
                )
            })?
        };

        Ok((start, end, resolved_pattern, provider_result.format))
    } else {
        let (start, end) = parse_explicit_range(range, puzzle)?;
        let resolved_pattern = if pattern == "." {
            ".".to_string()
        } else {
            pattern.to_string()
        };
        Ok((start, end, resolved_pattern, default_format))
    }
}

fn parse_explicit_range(
    range: Option<String>,
    puzzle: Option<u32>,
) -> Result<(BigUint, BigUint)> {
    if let Some(p) = puzzle {
        if p < 1 || p > 160 {
            anyhow::bail!("Puzzle number must be between 1 and 160");
        }
        let start = BigUint::one() << (p - 1);
        let end = (BigUint::one() << p) - 1u32;
        Ok((start, end))
    } else if let Some(r) = range {
        let parts: Vec<&str> = r.split(':').collect();
        if parts.len() != 2 {
            anyhow::bail!("Range must be in format START:END");
        }
        let start = BigUint::from_str_radix(parts[0], 16).context("Invalid start hex")?;
        let end = BigUint::from_str_radix(parts[1], 16).context("Invalid end hex")?;
        Ok((start, end))
    } else {
        anyhow::bail!("Either --range, --puzzle, or a provider pattern with key range must be specified")
    }
}

// Unified search runner
fn run_search(
    pattern: &str,
    ignore_case: bool,
    config: ScanConfig,
    gpu: bool,
    use_tui: bool,
    quiet: bool,
    output: OutputFormat,
    file: Option<PathBuf>,
    repeat: usize,
) -> Result<()> {
    if use_tui && repeat > 1 {
        anyhow::bail!("TUI mode supports a single run; use --no-tui for repeated runs");
    }
    let pat = Pattern::new(pattern, ignore_case).context("Failed to compile pattern")?;

    // Validate charset - warn about impossible patterns
    let invalid_chars = pat.validate_charset(config.format);
    if !invalid_chars.is_empty() {
        let format_name = match config.format {
            AddressFormat::P2pkh | AddressFormat::P2pkhUncompressed | AddressFormat::P2shP2wpkh => "Base58",
            AddressFormat::P2wpkh | AddressFormat::P2tr => "Bech32",
            AddressFormat::Ethereum => "Hex",
        };
        let chars_str: String = invalid_chars.iter().collect();
        eprintln!(
            "Warning: Pattern contains characters not valid in {} addresses: '{}'",
            format_name, chars_str
        );
        eprintln!("  {} alphabet excludes these characters - pattern will NEVER match!",
            format_name);
        if matches!(config.format, AddressFormat::P2pkh | AddressFormat::P2pkhUncompressed | AddressFormat::P2shP2wpkh) {
            eprintln!("  Base58 excludes: 0 (zero), O (uppercase o), I (uppercase i), l (lowercase L)");
        }
        eprintln!();
    }

    // Initialize GPU runner if needed, BEFORE entering TUI loop
    let mut gpu_runner: Option<Arc<GpuRunner>> = None;
    if gpu {
        if use_tui { eprintln!("Initializing GPU..."); }
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        match rt.block_on(GpuRunner::new(config.gpu_batch_size.unwrap_or(1048576))) {
            Ok(runner) => {
                gpu_runner = Some(Arc::new(runner));
                if use_tui { eprintln!("GPU initialized."); }
            },
            Err(e) => {
                eprintln!("Failed to initialize GPU ({e:?}); falling back to CPU.");
            }
        }
    }

    // TUI path
    if use_tui {
        let tui_result = run_tui(&pattern, ignore_case, config.clone(), gpu_runner.clone());
        match tui_result {
            Ok(_) => return Ok(()),
            Err(e) => {
                eprintln!("TUI failed: ({e:?}). Falling back to console output.");
            }
        }
    }

    // --- NON-TUI PATH ---

    let pb = if !quiet {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} [{elapsed_precise}] {msg}")
                .unwrap(),
        );
        pb.set_message(format!(
            "Searching for pattern '{}'...",
            pattern
        ));
        Some(pb)
    } else {
        None
    };

    let pb_clone = pb.clone();
    let base_ops = Arc::new(AtomicU64::new(0));
    let progress_cb: Option<ProgressCallback> = pb_clone.map(|pb| {
        let base = base_ops.clone();
        Arc::new(move |ops: u64| {
            let total = base.load(Ordering::Relaxed).saturating_add(ops);
            pb.set_message(format!("Checked {} addresses", format_with_commas(total)));
        }) as ProgressCallback
    });

    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();
    ctrlc_handler(stop_clone);

    // Build a single runtime for GPU path to avoid per-iteration overhead
    let mut rt = if gpu_runner.is_some() {
        Some(
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?,
        )
    } else {
        None
    };

    let start_total = std::time::Instant::now();
    let mut all_matches = Vec::new();
    let mut total_ops: u64 = 0;

    for i in 0..repeat {
        if let Some(pb) = &pb {
            pb.set_message(format!(
                "Run {} of {} (pattern '{}')...",
                i + 1,
                repeat,
                pattern
            ));
        }

        let result = if let Some(runner) = &gpu_runner {
            let rt_ref = rt.as_mut().expect("runtime should exist");
            rt_ref.block_on(scan_gpu_with_runner(
                    &pat,
                    &config,
                    progress_cb.clone(),
                    Some(stop.clone()),
                    runner.clone(),
                ))?
        } else {
            scan_with_progress(&pat, &config, progress_cb.clone(), Some(stop.clone()))
        };

        total_ops = total_ops.saturating_add(result.operations);
        base_ops.store(total_ops, Ordering::Relaxed);
        all_matches.extend(result.matches);

        if stop.load(Ordering::Relaxed) {
            break;
        }
    }

    let result = ScanResult {
        matches: all_matches,
        operations: total_ops,
        elapsed_secs: start_total.elapsed().as_secs_f64(),
    };

    if let Some(pb) = pb {
        pb.finish_and_clear();
    }

    // Output results
    let mut writer: Box<dyn Write> = if let Some(ref path) = file {
        Box::new(File::create(path).context("Failed to create output file")?)
    } else {
        Box::new(std::io::stdout())
    };

    if matches!(output, OutputFormat::Csv) && !result.matches.is_empty() {
        writeln!(writer, "address,wif,private_key_hex,format,pattern,operations,elapsed_secs,rate")?;
    }
    for (idx, addr) in result.matches.iter().enumerate() {
            let vanity_result = VanityResult {
                address: addr.address.clone(),
                wif: addr.wif.clone(),
                private_key_hex: addr.hex.clone(),
                format: match config.format {
                    AddressFormat::P2pkh => "P2PKH".to_string(),
                    AddressFormat::P2wpkh => "P2WPKH".to_string(),
                    AddressFormat::P2pkhUncompressed => "P2PKH (Uncompressed)".to_string(),
                    AddressFormat::P2shP2wpkh => "P2SH-P2WPKH".to_string(),
                    AddressFormat::P2tr => "P2TR".to_string(),
                    AddressFormat::Ethereum => "Ethereum".to_string(),
                },
                pattern: pattern.to_string(),

            operations: result.operations,
            elapsed_secs: result.elapsed_secs,
            rate: result.rate(),
        };

        match output {
            OutputFormat::Text => {
                writeln!(
                    writer,
                    "=== Match {} of {} ===",
                    idx + 1,
                    result.matches.len()
                )?;
                writeln!(writer, "Pattern : {}", vanity_result.pattern)?;
                writeln!(writer, "Format  : {}", vanity_result.format)?;
                writeln!(writer, "Address : {}", vanity_result.address)?;
                writeln!(writer, "WIF     : {}", vanity_result.wif)?;
                writeln!(writer, "Hex     : {}", vanity_result.private_key_hex)?;
                if !quiet {
                    writeln!(
                        writer,
                        "Ops     : {} ({:.0}/sec)",
                        format_with_commas(vanity_result.operations),
                        vanity_result.rate
                    )?;
                    writeln!(writer, "Time    : {}", format_duration(vanity_result.elapsed_secs))?;
                }
                writeln!(writer)?;
            }
            OutputFormat::Json => {
                writeln!(writer, "{}", serde_json::to_string_pretty(&vanity_result)?)?;
            }
            OutputFormat::Jsonl => {
                writeln!(writer, "{}", serde_json::to_string(&vanity_result)?)?;
            }
            OutputFormat::Csv => {
                writeln!(
                    writer,
                    "{},{},{},{},{},{},{},{}",
                    vanity_result.address,
                    vanity_result.wif,
                    vanity_result.private_key_hex,
                    vanity_result.format,
                    vanity_result.pattern,
                    vanity_result.operations,
                    vanity_result.elapsed_secs,
                    vanity_result.rate
                )?;
            }
            OutputFormat::Minimal => {
                writeln!(writer, "{}", vanity_result.wif)?;
            }
        }
    }

    if file.is_some() && !result.matches.is_empty() && !quiet {
        eprintln!("Wrote {} result(s) to {:?}", result.matches.len(), file.as_ref().unwrap());
    }

    if result.matches.is_empty() && !quiet {
        eprintln!(
            "No match found after {} operations ({})",
            format_with_commas(result.operations),
            format_duration(result.elapsed_secs)
        );
    }

    Ok(())
}

fn format_duration(secs: f64) -> String {
    if secs < 1.0 {
        format!("{:.0}ms", secs * 1000.0)
    } else if secs < 60.0 {
        format!("{:.1}s", secs)
    } else if secs < 3600.0 {
        format!("{:.1}m", secs / 60.0)
    } else if secs < 86400.0 {
        format!("{:.1}h", secs / 3600.0)
    } else if secs < 31536000.0 {
        format!("{:.1}d", secs / 86400.0)
    } else {
        format!("{:.1}y", secs / 31536000.0)
    }
}

fn format_with_commas(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, &b) in bytes.iter().enumerate() {
        if i != 0 && (bytes.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(b as char);
    }
    out
}

fn ctrlc_handler(stop: Arc<AtomicBool>) {
    let _ = ctrlc::set_handler(move || {
        if stop.load(Ordering::Relaxed) {
            // Second Ctrl+C, force exit
            std::process::exit(1);
        }
        eprintln!("\nStopping... (press Ctrl+C again to force)");
        stop.store(true, Ordering::Relaxed);
    });
}

fn run_tui(
    pattern: &str,
    ignore_case: bool,
    config: ScanConfig,
    gpu_runner: Option<Arc<GpuRunner>>,
) -> Result<()> {
    let pat = Pattern::new(pattern, ignore_case).context("Failed to compile pattern")?;
    let pattern_owned = pattern.to_string();
    let format_label = match config.format {
        AddressFormat::P2pkh => "P2PKH",
        AddressFormat::P2wpkh => "P2WPKH",
        AddressFormat::P2pkhUncompressed => "P2PKH (Uncompressed)",
        AddressFormat::P2shP2wpkh => "P2SH-P2WPKH",
        AddressFormat::P2tr => "P2TR",
        AddressFormat::Ethereum => "Ethereum",
    }.to_string();

    let gpu_enabled = gpu_runner.is_some();

    let gpu_device_name = if let Some(runner) = &gpu_runner {
        runner.device_name.clone()
    } else {
        "None".to_string()
    };

    let difficulty = if config.start.is_some() {
        // Difficulty doesn't apply to range scan in the same way (it's 100% chance if in range)
        // But estimate_difficulty is for random search.
        0
    } else {
        pat.estimate_difficulty(config.format)
    };

    let state = Arc::new(std::sync::Mutex::new(TuiState {
        pattern: pattern_owned.clone(),
        format: format_label.clone(),
        difficulty,
        gpu_enabled,
        tui_status_message: "Initializing...".to_string(),
        gpu_device_name,
        ..Default::default()
    }));

    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();
    let state_for_progress = state.clone();
    let state_for_result = state.clone();
    let config_clone = config.clone();
    let runner_clone = gpu_runner.clone();

    let mut runner_thread_opt = Some(std::thread::spawn(move || -> Result<()> {
        let start = std::time::Instant::now();
        let state_cb = state_for_progress.clone();
        let progress_cb: ProgressCallback = Arc::new(move |ops: u64| {
            let mut st = state_cb.lock().unwrap();
            st.operations = ops;
            st.elapsed = start.elapsed().as_secs_f64();
            st.rate = if st.elapsed > 0.0 {
                st.operations as f64 / st.elapsed
            } else {
                0.0
            };
        });

        let res = if let Some(runner) = runner_clone {
            state_for_progress.lock().unwrap().tui_status_message = "GPU Search...".to_string();
            let rt_thread = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            match rt_thread.block_on(scan_gpu_with_runner(&pat, &config_clone, Some(progress_cb.clone()), Some(stop_clone.clone()), runner)) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("GPU path failed in TUI thread ({e:?}); falling back to CPU.");
                    state_for_progress.lock().unwrap().tui_status_message = format!("GPU failed: {}, falling back to CPU.", e).to_string();
                    scan_with_progress(&pat, &config_clone, Some(progress_cb), Some(stop_clone))
                }
            }
        } else {
            state_for_progress.lock().unwrap().tui_status_message = "CPU Search...".to_string();
            scan_with_progress(&pat, &config_clone, Some(progress_cb), Some(stop_clone))
        };

        let mut st = state_for_result.lock().unwrap();
        st.operations = res.operations;
        st.elapsed = res.elapsed_secs;
        st.rate = res.rate();
        st.matches = res.matches.iter().map(|addr| VanityResult {
            address: addr.address.clone(),
            wif: addr.wif.clone(),
            private_key_hex: addr.hex.clone(),
            format: format_label.clone(),
            pattern: pattern_owned.clone(),
            operations: res.operations,
            elapsed_secs: res.elapsed_secs,
            rate: res.rate(),
        }).collect();
        st.done = true;
        st.tui_status_message = "Search complete.".to_string();
        Ok(())
    }));

    // TUI Initialization
    match enable_raw_mode() {
        Ok(_) => {},
        Err(e) => {
            eprintln!("Failed to enable raw mode: {e:?}. Fallback to console.");
            return Err(e.into());
        }
    };
    let mut stdout = std::io::stdout();
    match crossterm::execute!(stdout, crossterm::terminal::EnterAlternateScreen) {
        Ok(_) => {},
        Err(e) => {
            eprintln!("Failed to enter alternate screen: {e:?}. Fallback to console.");
            return Err(e.into());
        }
    };
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Local state for TUI visualization
    let mut rate_history: Vec<u64> = vec![0; 100]; // Buffer for sparkline

    loop {
        let st = state.lock().unwrap().clone();

        // Update history
        rate_history.push(st.rate as u64);
        if rate_history.len() > 100 {
            rate_history.remove(0);
        }

        terminal.draw(|f| {
            let size = f.area();

            // Main Layout
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(1)
                .constraints([
                    Constraint::Length(3),  // Top Bar (Config)
                    Constraint::Length(10), // Dashboard (Stats + Chart)
                    Constraint::Min(5),     // Matches
                    Constraint::Length(1),  // Footer
                ])
                .split(size);

            // --- 1. Top Bar ---
            let title_style = Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD);
            let value_style = Style::default().fg(Color::Cyan);

            let mut top_text = vec![
                Span::styled(" Pattern: ", Style::default().fg(Color::Gray)),
                Span::styled(format!(" {} ", st.pattern), value_style),
                Span::raw(" │ "),
                Span::styled(" Format: ", Style::default().fg(Color::Gray)),
                Span::styled(format!(" {} ", st.format), value_style),
                Span::raw(" │ "),
                Span::styled(" Difficulty: ", Style::default().fg(Color::Gray)),
                Span::styled(format!(" 1 in {} ", format_with_commas(st.difficulty)), value_style),
                Span::raw(" │ "),
                Span::styled(" Mode: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    if st.gpu_enabled { " GPU ACCELERATED " } else { " CPU STANDARD " },
                    if st.gpu_enabled { Style::default().fg(Color::Green).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Yellow) }
                ),
            ];

            if st.gpu_enabled {
                top_text.push(Span::raw(" │ "));
                top_text.push(Span::styled(" Device: ", Style::default().fg(Color::Gray)));
                top_text.push(Span::styled(format!(" {} ", st.gpu_device_name), Style::default().fg(Color::Green)));
            }

            let top_block = Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(Span::styled(" VGEN ", title_style));

            let top_para = Paragraph::new(Line::from(top_text)).block(top_block).alignment(ratatui::layout::Alignment::Center);
            f.render_widget(top_para, chunks[0]);

            // --- 2. Dashboard (Split Left/Right) ---
            let dash_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
                .split(chunks[1]);

            // Left: Numeric Stats
            let rate_color = if st.rate > 500_000.0 { Color::Green } else if st.rate > 100_000.0 { Color::Yellow } else { Color::Red };

            let mut stats_text = vec![
                Line::from(vec![
                    Span::styled("Status:    ", Style::default().fg(Color::Gray)),
                    Span::styled(&st.tui_status_message, Style::default().fg(Color::White).add_modifier(Modifier::BOLD))
                ]),
                Line::from(""),
                Line::from(vec![
                    Span::styled("Hashrate:  ", Style::default().fg(Color::Gray)),
                    Span::styled(format!("{:.0} keys/s", st.rate), Style::default().fg(rate_color).add_modifier(Modifier::BOLD))
                ]),
                Line::from(vec![
                    Span::styled("Checked:   ", Style::default().fg(Color::Gray)),
                    Span::styled(format_with_commas(st.operations), Style::default().fg(Color::Cyan))
                ]),
                Line::from(vec![
                    Span::styled("Elapsed:   ", Style::default().fg(Color::Gray)),
                    Span::styled(format_duration(st.elapsed), Style::default().fg(Color::Cyan))
                ]),
            ];

            // Calculate Luck / Status relative to difficulty
            if st.difficulty > 0 {
                let factor = st.operations as f64 / st.difficulty as f64;
                let (luck_label, luck_style) = if factor < 1.0 {
                    let speedup = 1.0 / factor.max(0.0001);
                    (format!("Lucky ({:.1}x faster)", speedup), Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))
                } else {
                    (format!("Unlucky ({:.1}x slower)", factor), Style::default().fg(if factor > 3.0 { Color::Red } else { Color::Yellow }).add_modifier(Modifier::BOLD))
                };

                stats_text.push(Line::from(vec![
                    Span::styled("Luck:      ", Style::default().fg(Color::Gray)),
                    Span::styled(luck_label, luck_style)
                ]));
            } else {
                 stats_text.push(Line::from(vec![
                    Span::styled("Luck:      ", Style::default().fg(Color::Gray)),
                    Span::styled("-", Style::default().fg(Color::DarkGray))
                ]));
            }

            let stats_block = Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(" Statistics ");
            let stats_para = Paragraph::new(stats_text).block(stats_block).alignment(ratatui::layout::Alignment::Left);
            f.render_widget(stats_para, dash_chunks[0]);

            // Right: Chart
            let chart_block = Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(" Performance ");

            let sparkline = Sparkline::default()
                .block(chart_block)
                .data(&rate_history)
                .style(Style::default().fg(Color::Magenta));
            f.render_widget(sparkline, dash_chunks[1]);

            // --- 3. Matches List ---
            let items: Vec<ListItem> = if st.matches.is_empty() {
                vec![ListItem::new(Span::styled(" Waiting for matches...", Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC)))]
            } else {
                st.matches.iter().enumerate().map(|(i, m)| {
                    ListItem::new(vec![
                        Line::from(vec![
                            Span::styled(format!(" MATCH #{} ", i + 1), Style::default().bg(Color::Green).fg(Color::Black).add_modifier(Modifier::BOLD)),
                            Span::raw(" "),
                            Span::styled(&m.address, Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                        ]),
                        Line::from(vec![
                            Span::raw("    WIF: "),
                            Span::styled(&m.wif, Style::default().fg(Color::Gray)),
                        ]),
                        Line::from(""),
                    ])
                }).collect()
            };

            let matches_block = Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(" Found Matches ");
            let list = List::new(items).block(matches_block);
            f.render_widget(list, chunks[2]);

            // --- 4. Footer ---
            let footer_text = Line::from(vec![
                Span::styled(" Q ", Style::default().bg(Color::Red).fg(Color::White).add_modifier(Modifier::BOLD)),
                Span::styled(" Quit ", Style::default().fg(Color::Gray)),
            ]);
            let footer_para = Paragraph::new(footer_text).alignment(ratatui::layout::Alignment::Right);
            f.render_widget(footer_para, chunks[3]);

        })?;

        // Input Handling
        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q') || key.code == KeyCode::Esc {
                    stop.store(true, Ordering::Relaxed);
                    break;
                }
            }
        }

        // Thread management
        if let Some(handle) = runner_thread_opt.as_ref() {
            if handle.is_finished() {
                if let Err(e) = runner_thread_opt.take().unwrap().join().unwrap() {
                    let mut st_lock = state.lock().unwrap();
                    st_lock.tui_status_message = format!("Error: {:?}", e);
                }
                state.lock().unwrap().done = true;
            }
        }

        let st_lock = state.lock().unwrap();
        if st_lock.done {
            // Wait a bit so user can see "Done" before exit?
            // Or just exit. Let's wait for Q if done.
            if stop.load(Ordering::Relaxed) {
                break;
            }
        }
    }

    disable_raw_mode()?;
    crossterm::execute!(
        terminal.backend_mut(),
        crossterm::terminal::LeaveAlternateScreen
    )?;
    terminal.show_cursor()?;
    Ok(())
}
