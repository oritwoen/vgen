//! Bitcoin and Ethereum address generation.

use anyhow::{anyhow, Result};
use bitcoin::key::{Keypair, Secp256k1, UntweakedPublicKey};
use bitcoin::secp256k1::{All, SecretKey};
use bitcoin::{Address, CompressedPublicKey, Network, PrivateKey, PublicKey, ScriptBuf};
use sha3::{Digest, Keccak256};

/// Address format to generate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AddressFormat {
    /// Bitcoin P2PKH - Pay to Public Key Hash (1...)
    P2pkh,
    /// Bitcoin P2WPKH - Pay to Witness Public Key Hash (bc1q...)
    P2wpkh,
    /// Bitcoin P2SH-P2WPKH - Nested SegWit (3...)
    P2shP2wpkh,
    /// Bitcoin P2TR - Pay to Taproot (bc1p...)
    P2tr,
    /// Bitcoin P2PKH Uncompressed (legacy)
    P2pkhUncompressed,
    /// Ethereum Address (0x...)
    Ethereum,
}

impl AddressFormat {
    pub fn all() -> Vec<AddressFormat> {
        vec![
            AddressFormat::P2pkh,
            AddressFormat::P2pkhUncompressed,
            AddressFormat::P2wpkh,
            AddressFormat::P2shP2wpkh,
            AddressFormat::P2tr,
            AddressFormat::Ethereum,
        ]
    }

    /// Returns the character encoding name used by this address format.
    pub fn charset_name(&self) -> &'static str {
        match self {
            AddressFormat::P2pkh | AddressFormat::P2pkhUncompressed | AddressFormat::P2shP2wpkh => "Base58",
            AddressFormat::P2wpkh | AddressFormat::P2tr => "Bech32",
            AddressFormat::Ethereum => "Hex",
        }
    }
}

impl std::fmt::Display for AddressFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AddressFormat::P2pkh => write!(f, "P2PKH"),
            AddressFormat::P2pkhUncompressed => write!(f, "P2PKH (Uncompressed)"),
            AddressFormat::P2wpkh => write!(f, "P2WPKH"),
            AddressFormat::P2shP2wpkh => write!(f, "P2SH-P2WPKH"),
            AddressFormat::P2tr => write!(f, "P2TR"),
            AddressFormat::Ethereum => write!(f, "Ethereum"),
        }
    }
}

/// A generated address with its private key.
#[derive(Debug, Clone)]
pub struct GeneratedAddress {
    /// The address string
    pub address: String,
    /// WIF-encoded private key (Bitcoin) or Hex (Ethereum)
    pub wif: String,
    /// Hex-encoded private key (32 bytes)
    pub hex: String,
    /// Format used
    pub format: AddressFormat,
}

/// Address generator with reusable Secp256k1 context.
pub struct AddressGenerator {
    secp: Secp256k1<All>,
    network: Network,
    format: AddressFormat,
}

impl AddressGenerator {
    /// Create a new address generator for the specified format.
    pub fn new(format: AddressFormat) -> Self {
        Self {
            secp: Secp256k1::new(),
            network: Network::Bitcoin,
            format,
        }
    }

    /// Generate an address from a 32-byte secret key.
    pub fn generate(&self, secret: &[u8; 32]) -> Option<GeneratedAddress> {
        let secret_key = SecretKey::from_slice(secret).ok()?;

        match self.format {
            AddressFormat::Ethereum => {
                let public_key = secret_key.public_key(&self.secp);
                let serialized = public_key.serialize_uncompressed();
                let pub_bytes = &serialized[1..]; // Drop 0x04

                let mut hasher = Keccak256::new();
                hasher.update(pub_bytes);
                let hash = hasher.finalize();

                let address_bytes = &hash[12..];
                let address_hex = hex::encode(address_bytes);
                let checksum_address = to_checksum_address(&address_hex);

                Some(GeneratedAddress {
                    address: checksum_address,
                    wif: hex::encode(secret),
                    hex: hex::encode(secret),
                    format: self.format,
                })
            }
            _ => {
                // Bitcoin formats
                let compressed = self.format != AddressFormat::P2pkhUncompressed;
                let mut private_key = PrivateKey::new(secret_key, self.network);
                private_key.compressed = compressed;

                let address = match self.format {
                    AddressFormat::P2pkh | AddressFormat::P2pkhUncompressed => {
                        let public_key = private_key.public_key(&self.secp);
                        Address::p2pkh(public_key, self.network)
                    }
                    AddressFormat::P2wpkh => {
                        let public_key =
                            CompressedPublicKey::from_private_key(&self.secp, &private_key).ok()?;
                        Address::p2wpkh(&public_key, self.network)
                    }
                    AddressFormat::P2shP2wpkh => {
                        let public_key =
                            CompressedPublicKey::from_private_key(&self.secp, &private_key).ok()?;
                        let script = ScriptBuf::new_p2wpkh(&public_key.wpubkey_hash());
                        Address::p2sh(&script, self.network).ok()?
                    }
                    AddressFormat::P2tr => {
                        let keypair = Keypair::from_secret_key(&self.secp, &secret_key);
                        let (internal_key, _) = UntweakedPublicKey::from_keypair(&keypair);
                        Address::p2tr(&self.secp, internal_key, None, self.network)
                    }
                    AddressFormat::Ethereum => unreachable!(),
                };

                Some(GeneratedAddress {
                    address: address.to_string(),
                    wif: private_key.to_wif(),
                    hex: hex::encode(secret),
                    format: self.format,
                })
            }
        }
    }

    /// Helper: Generate P2PKH from compressed public key bytes.
    pub fn p2pkh_from_public_key_bytes(bytes: &[u8; 33]) -> Result<String> {
        let pubkey = PublicKey::from_slice(bytes)?;
        Ok(Address::p2pkh(pubkey, Network::Bitcoin).to_string())
    }

    /// Helper: Generate P2WPKH from compressed public key bytes.
    pub fn p2wpkh_from_public_key_bytes(bytes: &[u8; 33]) -> Result<String> {
        let pubkey = PublicKey::from_slice(bytes)?;
        let compressed = CompressedPublicKey::try_from(pubkey)?;
        Ok(Address::p2wpkh(&compressed, Network::Bitcoin).to_string())
    }

    /// Helper: Convert secret bytes to WIF.
    /// Returns None if the secret key is invalid (zero or >= curve order).
    pub fn bytes_to_wif(secret: &[u8; 32], network: Network) -> Option<String> {
        SecretKey::from_slice(secret)
            .ok()
            .map(|sk| PrivateKey::new(sk, network).to_wif())
    }
}

/// Convert Ethereum address to checksum address (EIP-55)
fn to_checksum_address(address: &str) -> String {
    let address = address.trim_start_matches("0x").to_lowercase();
    let mut hasher = Keccak256::new();
    hasher.update(address.as_bytes());
    let hash = hex::encode(hasher.finalize());

    let mut result = String::with_capacity(42);
    result.push_str("0x");

    for (i, c) in address.chars().enumerate() {
        let hash_char = hash.chars().nth(i).unwrap();
        if let Some(d) = hash_char.to_digit(16) {
            if d >= 8 {
                result.push(c.to_ascii_uppercase());
            } else {
                result.push(c);
            }
        } else {
            result.push(c);
        }
    }
    result
}

/// Validate WIF key
pub fn validate_wif(wif: &str) -> Result<()> {
    PrivateKey::from_wif(wif)
        .map(|_| ())
        .map_err(|e| anyhow!(e))
}

/// Decode WIF to hex and compression status
pub fn decode_wif(wif: &str) -> Result<(String, bool)> {
    let pk = PrivateKey::from_wif(wif)?;
    Ok((hex::encode(pk.inner.secret_bytes()), pk.compressed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest as Sha2Digest, Sha256};

    fn sha256(data: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.finalize().into()
    }

    #[test]
    fn test_p2pkh_starts_with_1() {
        let gen = AddressGenerator::new(AddressFormat::P2pkh);
        let secret = [1u8; 32];
        let result = gen.generate(&secret).unwrap();
        assert!(result.address.starts_with('1'));
    }

    #[test]
    fn test_brainwallet_known_address() {
        let hash = sha256(b"correct horse battery staple");
        let gen = AddressGenerator::new(AddressFormat::P2pkh);
        let result = gen.generate(&hash).unwrap();
        assert_eq!(result.address, "1C7zdTfnkzmr13HfA2vNm5SJYRK6nEKyq8");
    }

    #[test]
    fn test_p2wpkh_starts_with_bc1q() {
        let gen = AddressGenerator::new(AddressFormat::P2wpkh);
        let secret = [1u8; 32];
        let result = gen.generate(&secret).unwrap();
        assert!(result.address.starts_with("bc1q"));
    }

    #[test]
    fn test_ethereum_starts_with_0x() {
        let gen = AddressGenerator::new(AddressFormat::Ethereum);
        let secret = [1u8; 32];
        let result = gen.generate(&secret).unwrap();
        assert!(result.address.starts_with("0x"));
        assert_eq!(result.address.len(), 42);
    }

    #[test]
    fn test_display_format() {
        assert_eq!(AddressFormat::P2pkh.to_string(), "P2PKH");
        assert_eq!(AddressFormat::P2wpkh.to_string(), "P2WPKH");
        assert_eq!(AddressFormat::P2shP2wpkh.to_string(), "P2SH-P2WPKH");
        assert_eq!(AddressFormat::P2tr.to_string(), "P2TR");
        assert_eq!(AddressFormat::P2pkhUncompressed.to_string(), "P2PKH (Uncompressed)");
        assert_eq!(AddressFormat::Ethereum.to_string(), "Ethereum");
    }

    #[test]
    fn test_charset_name() {
        assert_eq!(AddressFormat::P2pkh.charset_name(), "Base58");
        assert_eq!(AddressFormat::P2pkhUncompressed.charset_name(), "Base58");
        assert_eq!(AddressFormat::P2shP2wpkh.charset_name(), "Base58");
        assert_eq!(AddressFormat::P2wpkh.charset_name(), "Bech32");
        assert_eq!(AddressFormat::P2tr.charset_name(), "Bech32");
        assert_eq!(AddressFormat::Ethereum.charset_name(), "Hex");
    }
}
