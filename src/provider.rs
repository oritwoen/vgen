use anyhow::{anyhow, Result};
use num_bigint::BigUint;

use crate::address::AddressFormat;

pub struct ProviderResult {
    pub address: String,
    pub format: AddressFormat,
    pub key_range: Option<(BigUint, BigUint)>,
}

pub fn resolve(pattern: &str) -> Result<Option<ProviderResult>> {
    let Some((provider, path)) = pattern.split_once(':') else {
        return Ok(None);
    };

    match provider {
        "boha" => resolve_boha(path).map(Some),
        _ => Ok(None),
    }
}

fn resolve_boha(path: &str) -> Result<ProviderResult> {
    let puzzle_id = path.replace(':', "/");

    let puzzle = boha::get(&puzzle_id)
        .map_err(|e| anyhow!("Failed to get puzzle '{}': {}", puzzle_id, e))?;

    let format = match puzzle.address.kind {
        "p2pkh" => AddressFormat::P2pkh,
        "p2wpkh" => AddressFormat::P2wpkh,
        "p2tr" => AddressFormat::P2tr,
        "p2sh" => AddressFormat::P2shP2wpkh,
        other => {
            eprintln!(
                "Warning: Unknown address kind '{}', defaulting to P2PKH",
                other
            );
            AddressFormat::P2pkh
        }
    };

    let key_range = puzzle
        .key
        .as_ref()
        .and_then(|k| k.range_big())
        .map(|(start, end)| (start.clone(), end.clone()));

    Ok(ProviderResult {
        address: puzzle.address.value.to_string(),
        format,
        key_range,
    })
}

pub fn build_pattern(result: &ProviderResult, prefix_length: usize) -> String {
    let len = prefix_length.min(result.address.len());
    let prefix = &result.address[..len];
    format!("^{}", regex::escape(prefix))
}

pub fn build_exact_pattern(result: &ProviderResult) -> String {
    format!("^{}$", regex::escape(&result.address))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_non_provider_pattern() {
        let result = resolve("^1Cat").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_boha_b1000_puzzle() {
        let result = resolve("boha:b1000:1").unwrap();
        assert!(result.is_some());

        let provider_result = result.unwrap();
        assert_eq!(
            provider_result.address,
            "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH"
        );
        assert_eq!(provider_result.format, AddressFormat::P2pkh);
        assert!(provider_result.key_range.is_some());
    }

    #[test]
    fn test_resolve_boha_slash_syntax() {
        let result = resolve("boha:b1000/1").unwrap();
        assert!(result.is_some());

        let provider_result = result.unwrap();
        assert_eq!(
            provider_result.address,
            "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH"
        );
    }

    #[test]
    fn test_resolve_boha_invalid_puzzle() {
        let result = resolve("boha:invalid:999999");
        assert!(result.is_err());
    }

    #[test]
    fn test_build_pattern() {
        let result = ProviderResult {
            address: "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so".to_string(),
            format: AddressFormat::P2pkh,
            key_range: None,
        };

        assert_eq!(build_pattern(&result, 6), "^13zb1h");
        assert_eq!(build_pattern(&result, 10), "^13zb1hQbWV");
    }

    #[test]
    fn test_build_pattern_clamps_length() {
        let result = ProviderResult {
            address: "1Cat".to_string(),
            format: AddressFormat::P2pkh,
            key_range: None,
        };

        assert_eq!(build_pattern(&result, 100), "^1Cat");
    }

    #[test]
    fn test_build_exact_pattern() {
        let result = ProviderResult {
            address: "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so".to_string(),
            format: AddressFormat::P2pkh,
            key_range: None,
        };

        assert_eq!(
            build_exact_pattern(&result),
            "^13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so$"
        );
    }
}
