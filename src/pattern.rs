//! Regex pattern matching and difficulty estimation.

use anyhow::{Context, Result};
use regex::Regex;

use crate::address::AddressFormat;

/// Compiled regex pattern for address matching.
pub struct Pattern {
    regex: Regex,
    original: String,
    case_insensitive: bool,
}

impl Pattern {
    /// Create a new pattern from a regex string.
    ///
    /// # Errors
    ///
    /// Returns an error if the pattern is empty or invalid regex.
    pub fn new(pattern: &str, case_insensitive: bool) -> Result<Self> {
        if pattern.is_empty() {
            anyhow::bail!("Pattern cannot be empty");
        }

        let regex_pattern = if case_insensitive {
            format!("(?i){}", pattern)
        } else {
            pattern.to_string()
        };

        let regex = Regex::new(&regex_pattern)
            .with_context(|| format!("Invalid regex pattern: {}", pattern))?;

        Ok(Self {
            regex,
            original: pattern.to_string(),
            case_insensitive,
        })
    }

    /// Check if an address matches the pattern.
    pub fn matches(&self, address: &str) -> bool {
        self.regex.is_match(address)
    }

    /// Validate that pattern only contains characters valid for the address format.
    /// Returns a list of invalid characters found in the pattern.
    pub fn validate_charset(&self, format: AddressFormat) -> Vec<char> {
        let valid_chars: &str = match format {
            // Base58: excludes 0, O, I, l
            AddressFormat::P2pkh | AddressFormat::P2pkhUncompressed | AddressFormat::P2shP2wpkh => {
                "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
            }
            // Bech32: lowercase only, excludes 1, b, i, o
            AddressFormat::P2wpkh | AddressFormat::P2tr => "023456789acdefghjklmnpqrstuvwxyz",
            // Hex: 0-9, a-f (case insensitive but we check lowercase)
            AddressFormat::Ethereum => "0123456789abcdefABCDEFx",
        };

        let mut invalid = Vec::new();
        let mut in_class = false;
        let mut escaped = false;

        for c in self.original.chars() {
            if escaped {
                escaped = false;
                continue;
            }

            match c {
                '\\' => escaped = true,
                '[' => in_class = true,
                ']' => in_class = false,
                // Skip regex metacharacters
                '^' | '$' | '.' | '*' | '+' | '?' | '(' | ')' | '{' | '}' | '|' | '-' => {}
                _ if c.is_alphanumeric() && !in_class => {
                    // For case insensitive, check both cases
                    let char_valid = if self.case_insensitive {
                        valid_chars.contains(c.to_ascii_lowercase())
                            || valid_chars.contains(c.to_ascii_uppercase())
                    } else {
                        valid_chars.contains(c)
                    };
                    if !char_valid && !invalid.contains(&c) {
                        invalid.push(c);
                    }
                }
                _ => {}
            }
        }

        invalid
    }

    /// Estimate difficulty (1 in N addresses will match).
    ///
    /// This is a heuristic based on counting fixed alphanumeric characters.
    /// Actual difficulty depends on the pattern complexity.
    pub fn estimate_difficulty(&self, format: AddressFormat) -> u64 {
        // Alphabet sizes:
        // - Base58 (P2PKH): 58 chars (case sensitive), ~34 effective (case insensitive)
        // - Bech32 (P2WPKH/P2TR): 32 chars (always lowercase)
        // - Ethereum: 16 chars (hex)
        let alphabet_size: u64 = match format {
            AddressFormat::P2pkh | AddressFormat::P2pkhUncompressed | AddressFormat::P2shP2wpkh => {
                if self.case_insensitive {
                    34
                } else {
                    58
                }
            }
            AddressFormat::P2wpkh | AddressFormat::P2tr => 32,
            AddressFormat::Ethereum => 16,
        };

        // Count fixed alphanumeric characters in pattern (excluding regex metacharacters)
        let fixed_chars = count_fixed_chars(&self.original);

        // Only subtract prefix if pattern is anchored to start (^) and contains the prefix
        let prefix_to_subtract = if self.original.starts_with('^') {
            let pattern_after_anchor = &self.original[1..];
            match format {
                AddressFormat::P2pkh | AddressFormat::P2pkhUncompressed => {
                    if pattern_after_anchor.starts_with('1') {
                        1
                    } else {
                        0
                    }
                }
                AddressFormat::P2shP2wpkh => {
                    if pattern_after_anchor.starts_with('3') {
                        1
                    } else {
                        0
                    }
                }
                AddressFormat::P2wpkh => {
                    // bc1q prefix
                    if pattern_after_anchor.starts_with("bc1q") {
                        4
                    } else if pattern_after_anchor.starts_with("bc1") {
                        3
                    } else if pattern_after_anchor.starts_with("bc") {
                        2
                    } else if pattern_after_anchor.starts_with('b') {
                        1
                    } else {
                        0
                    }
                }
                AddressFormat::P2tr => {
                    // bc1p prefix
                    if pattern_after_anchor.starts_with("bc1p") {
                        4
                    } else if pattern_after_anchor.starts_with("bc1") {
                        3
                    } else if pattern_after_anchor.starts_with("bc") {
                        2
                    } else if pattern_after_anchor.starts_with('b') {
                        1
                    } else {
                        0
                    }
                }
                AddressFormat::Ethereum => {
                    // 0x prefix
                    if pattern_after_anchor.starts_with("0x")
                        || pattern_after_anchor.starts_with("0X")
                    {
                        2
                    } else if pattern_after_anchor.starts_with('0') {
                        1
                    } else {
                        0
                    }
                }
            }
        } else {
            0 // Not anchored to start, don't subtract any prefix
        };

        let effective_chars = fixed_chars.saturating_sub(prefix_to_subtract);

        if effective_chars == 0 {
            return 1; // Pattern will match almost anything
        }

        alphabet_size.saturating_pow(effective_chars as u32)
    }

    /// Get the original pattern string.
    pub fn original(&self) -> &str {
        &self.original
    }

    /// Check if case insensitive matching is enabled.
    pub fn is_case_insensitive(&self) -> bool {
        self.case_insensitive
    }
}

/// Count fixed alphanumeric characters in a pattern.
///
/// Excludes regex metacharacters and character classes.
fn count_fixed_chars(pattern: &str) -> usize {
    let mut count = 0;
    let mut in_class = false;
    let mut escaped = false;

    for c in pattern.chars() {
        if escaped {
            // Escaped characters don't count as fixed
            escaped = false;
            continue;
        }

        match c {
            '\\' => escaped = true,
            '[' => in_class = true,
            ']' => in_class = false,
            '^' | '$' | '.' | '*' | '+' | '?' | '(' | ')' | '{' | '}' | '|' => {
                // Regex metacharacters don't count
            }
            _ if !in_class && c.is_alphanumeric() => count += 1,
            _ => {}
        }
    }

    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_valid() {
        assert!(Pattern::new("^1Cat", false).is_ok());
        assert!(Pattern::new("^bc1q.*dead$", false).is_ok());
        assert!(Pattern::new("1[Oo]ri", false).is_ok());
    }

    #[test]
    fn test_pattern_empty() {
        let result = Pattern::new("", false);
        assert!(result.is_err());
    }

    #[test]
    fn test_pattern_invalid_regex() {
        // Unclosed bracket
        let result = Pattern::new("[invalid", false);
        assert!(result.is_err());
    }

    #[test]
    fn test_matches_simple() {
        let pat = Pattern::new("^1Cat", false).unwrap();
        assert!(pat.matches("1CatXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"));
        assert!(!pat.matches("1DogXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"));
        assert!(!pat.matches("1catXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")); // Case sensitive
    }

    #[test]
    fn test_matches_case_insensitive() {
        let pat = Pattern::new("^1cat", true).unwrap();
        assert!(pat.matches("1CatXXX"));
        assert!(pat.matches("1catXXX"));
        assert!(pat.matches("1CATXXX"));
        assert!(pat.matches("1cAtXXX"));
    }

    #[test]
    fn test_matches_suffix() {
        let pat = Pattern::new("dead$", false).unwrap();
        assert!(pat.matches("bc1qXXXXXXXXXXXXXXXXXXXXXXXXXXXdead"));
        assert!(!pat.matches("bc1qdeadXXXXXXXXXXXXXXXXXXXXXXXXXX"));
    }

    #[test]
    fn test_matches_regex() {
        let pat = Pattern::new("^1[Oo]ri", false).unwrap();
        assert!(pat.matches("1OriXXX"));
        assert!(pat.matches("1oriXXX"));
        assert!(!pat.matches("1ORIXXX")); // I is not in [Oo]
    }

    #[test]
    fn test_count_fixed_chars() {
        assert_eq!(count_fixed_chars("1Cat"), 4);
        assert_eq!(count_fixed_chars("^1Cat"), 4);
        assert_eq!(count_fixed_chars("^1Cat$"), 4);
        assert_eq!(count_fixed_chars("1[Oo]ri"), 3); // [Oo] is not fixed
        assert_eq!(count_fixed_chars("bc1q.*dead"), 8);
        assert_eq!(count_fixed_chars("^bc1q[a-z]+dead$"), 8);
    }

    #[test]
    fn test_difficulty_simple() {
        let pat = Pattern::new("^1Ab", false).unwrap();
        // 2 chars after "1", Base58 alphabet = 58
        // Expected: 58^2 = 3364
        let difficulty = pat.estimate_difficulty(AddressFormat::P2pkh);
        assert_eq!(difficulty, 58u64.pow(2));
    }

    #[test]
    fn test_difficulty_case_insensitive() {
        let pat = Pattern::new("^1Ab", true).unwrap();
        // 2 chars after "1", effective alphabet ~34 for case insensitive
        let difficulty = pat.estimate_difficulty(AddressFormat::P2pkh);
        assert_eq!(difficulty, 34u64.pow(2));
    }

    #[test]
    fn test_difficulty_bech32() {
        let pat = Pattern::new("^bc1qab", false).unwrap();
        // 2 chars after "bc1q", Bech32 alphabet = 32
        let difficulty = pat.estimate_difficulty(AddressFormat::P2wpkh);
        assert_eq!(difficulty, 32u64.pow(2));
    }

    #[test]
    fn test_difficulty_match_all() {
        let pat = Pattern::new("^1", false).unwrap();
        // Just the prefix, should match everything
        let difficulty = pat.estimate_difficulty(AddressFormat::P2pkh);
        assert_eq!(difficulty, 1);
    }

    #[test]
    fn test_difficulty_ethereum() {
        let pat = Pattern::new("^0xdead", false).unwrap();
        // 4 chars after "0x", Hex alphabet = 16
        // Expected: 16^4 = 65536
        let difficulty = pat.estimate_difficulty(AddressFormat::Ethereum);
        assert_eq!(difficulty, 16u64.pow(4));
    }

    #[test]
    fn test_original() {
        let pat = Pattern::new("^1Cat", false).unwrap();
        assert_eq!(pat.original(), "^1Cat");
    }

    #[test]
    fn test_is_case_insensitive() {
        let pat1 = Pattern::new("^1Cat", false).unwrap();
        let pat2 = Pattern::new("^1Cat", true).unwrap();
        assert!(!pat1.is_case_insensitive());
        assert!(pat2.is_case_insensitive());
    }

    #[test]
    fn test_difficulty_suffix() {
        // Suffix pattern - should NOT subtract prefix
        let pat = Pattern::new("dead$", false).unwrap();
        // 4 fixed chars, no prefix subtraction
        let difficulty = pat.estimate_difficulty(AddressFormat::P2pkh);
        assert_eq!(difficulty, 58u64.pow(4));
    }

    #[test]
    fn test_difficulty_no_anchor() {
        // Pattern without anchor - should NOT subtract prefix
        let pat = Pattern::new("Cat", false).unwrap();
        // 3 fixed chars, no prefix subtraction
        let difficulty = pat.estimate_difficulty(AddressFormat::P2pkh);
        assert_eq!(difficulty, 58u64.pow(3));
    }

    #[test]
    fn test_difficulty_anchor_without_prefix() {
        // Anchored pattern that doesn't include the address prefix
        let pat = Pattern::new("^Cat", false).unwrap();
        // 3 fixed chars, no prefix subtraction (pattern doesn't start with "1")
        let difficulty = pat.estimate_difficulty(AddressFormat::P2pkh);
        assert_eq!(difficulty, 58u64.pow(3));
    }

    #[test]
    fn test_difficulty_partial_bech32_prefix() {
        // Pattern with partial bech32 prefix
        let pat = Pattern::new("^bc1ab", false).unwrap();
        // 5 fixed chars (b,c,1,a,b), subtract 3 for "bc1", leaves 2 chars
        let difficulty = pat.estimate_difficulty(AddressFormat::P2wpkh);
        assert_eq!(difficulty, 32u64.pow(2));
    }

    #[test]
    fn test_validate_charset_base58_invalid() {
        // Base58 excludes 0, O, I, l
        let pat = Pattern::new("^1OR", false).unwrap();
        let invalid = pat.validate_charset(AddressFormat::P2pkh);
        assert_eq!(invalid, vec!['O']);
    }

    #[test]
    fn test_validate_charset_base58_valid() {
        let pat = Pattern::new("^1Cat", false).unwrap();
        let invalid = pat.validate_charset(AddressFormat::P2pkh);
        assert!(invalid.is_empty());
    }

    #[test]
    fn test_validate_charset_base58_zero() {
        let pat = Pattern::new("^10ri", false).unwrap();
        let invalid = pat.validate_charset(AddressFormat::P2pkh);
        assert_eq!(invalid, vec!['0']);
    }

    #[test]
    fn test_validate_charset_base58_multiple() {
        let pat = Pattern::new("^1OIl0", false).unwrap();
        let invalid = pat.validate_charset(AddressFormat::P2pkh);
        assert_eq!(invalid.len(), 4);
        assert!(invalid.contains(&'O'));
        assert!(invalid.contains(&'I'));
        assert!(invalid.contains(&'l'));
        assert!(invalid.contains(&'0'));
    }

    #[test]
    fn test_validate_charset_bech32() {
        // Bech32 excludes 1, b, i, o and uppercase
        let pat = Pattern::new("^bc1qAB", false).unwrap();
        let invalid = pat.validate_charset(AddressFormat::P2wpkh);
        assert!(invalid.contains(&'A'));
        assert!(invalid.contains(&'B'));
    }

    #[test]
    fn test_validate_charset_ethereum() {
        // Ethereum is hex - g, h, etc are invalid
        let pat = Pattern::new("^0xghi", false).unwrap();
        let invalid = pat.validate_charset(AddressFormat::Ethereum);
        assert!(invalid.contains(&'g'));
        assert!(invalid.contains(&'h'));
        assert!(invalid.contains(&'i'));
    }
}
