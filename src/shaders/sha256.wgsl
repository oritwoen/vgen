// SHA-256 implementation in WGSL

fn rrot(x: u32, n: u32) -> u32 {
    return (x >> n) | (x << (32u - n));
}

fn ch(x: u32, y: u32, z: u32) -> u32 {
    return (x & y) ^ ((~x) & z);
}

fn maj(x: u32, y: u32, z: u32) -> u32 {
    return (x & y) ^ (x & z) ^ (y & z);
}

fn sigma0(x: u32) -> u32 {
    return rrot(x, 2u) ^ rrot(x, 13u) ^ rrot(x, 22u);
}

fn sigma1(x: u32) -> u32 {
    return rrot(x, 6u) ^ rrot(x, 11u) ^ rrot(x, 25u);
}

fn gamma0(x: u32) -> u32 {
    return rrot(x, 7u) ^ rrot(x, 18u) ^ (x >> 3u);
}

fn gamma1(x: u32) -> u32 {
    return rrot(x, 17u) ^ rrot(x, 19u) ^ (x >> 10u);
}

// Byte-swap: convert BE word to LE (for field element format)
fn bswap(x: u32) -> u32 {
    return ((x >> 24u) & 0xffu) | ((x >> 8u) & 0xff00u) |
           ((x << 8u) & 0xff0000u) | ((x << 24u) & 0xff000000u);
}

// SHA-256 compression function for a single 64-byte block
// This is specific for Bitcoin public keys:
// Input is 33 bytes: [0x02/0x03, 32 bytes X]
// Padding: 0x80, then zeros, then length (264 bits = 0x0108)
// 33 bytes data + 1 byte 0x80 + 22 bytes 0x00 + 8 bytes length = 64 bytes.
// So we always process exactly ONE 64-byte block for compressed pubkeys.
fn sha256_compressed_pubkey(parity: u32, x: array<u32, 8>) -> array<u32, 8> {
    // Initial Hash Values (H0..H7)
    var h = array<u32, 8>(
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    );

    // Prepare Message Schedule W[0..63]
    var w: array<u32, 64>;

    // Block contents (16 words):
    // Word 0: 0x02/03 | x[0].byte3 | x[0].byte2 | x[0].byte1
    // Actually input x is [u32; 8] little endian limbs of big-endian number?
    // In generator.wgsl we write p_aff.x directly.
    // field.wgsl uses u32 limbs.
    // X coordinate is 32 bytes.
    // SHA256 expects Big Endian bytes.
    // Our limbs in generator.wgsl are LE u32s representing the number.
    // But the NUMBER itself, when written as bytes, is Big Endian in Bitcoin serialization.
    // Wait. fe_add etc use u32 limbs where output[0] is the least significant limb.
    // So output[0] corresponds to the LAST 4 bytes of the Big Endian representation.
    // The first byte of the pubkey (0x02/03) is followed by MSB of X.
    // MSB of X is in output[7] (the highest limb).
    // But inside the limb, is it BE or LE?
    // WGSL u32 is just a number.
    // When we say `x[7]`, that's the most significant 32-bit integer.
    // If x[7] = 0xAABBCCDD, does it mean byte sequence AA BB CC DD?
    // Yes, if we treat it as Big Endian.
    // SHA256 works on Big Endian words.

    // Structure of 64-byte block for SHA256(compressed_pubkey):
    // Byte 0: parity (02 or 03)
    // Byte 1..32: X coordinate (Big Endian) -> x[7], x[6]... x[0]
    // Byte 33: 0x80 (Padding start)
    // Byte 34..61: 0x00
    // Byte 62..63: Length in bits (33 * 8 = 264 = 0x0108)

    // We need to pack this into 16 x u32 (Big Endian words).

    // Word 0: [Parity, X_byte0, X_byte1, X_byte2]
    // X is in limbs x[7]..x[0].
    // x[7] is MSB.
    // x[7] bytes: B3 B2 B1 B0.
    // Word 0 = (Parity << 24) | (x[7] >> 8)

    let prefix = select(0x02000000u, 0x03000000u, (parity & 1u) == 1u);

    // We need to swap bytes if the limbs are not in the order SHA256 expects?
    // SHA256 expects the "stream of bytes" grouped into u32s.
    // The stream is: [Parity, X_MSB... X_LSB, 0x80, 0...0, Len]

    // x[7] is the top 4 bytes.
    // Let's verify limb endianness from gpu.rs:
    // "Limb 0 is LSB (last 4 bytes of Big Endian output)"
    // So x[7] is the FIRST 4 bytes of X.
    // If x[7] = 0x12345678, the bytes are 12, 34, 56, 78?
    // Yes, u32 in logic.

    // Word 0: Parity | x[7].bytes[0..3] -> actually shift
    // We want: P, x7_3, x7_2, x7_1
    w[0] = prefix | (x[7] >> 8u);
    w[1] = (x[7] << 24u) | (x[6] >> 8u);
    w[2] = (x[6] << 24u) | (x[5] >> 8u);
    w[3] = (x[5] << 24u) | (x[4] >> 8u);
    w[4] = (x[4] << 24u) | (x[3] >> 8u);
    w[5] = (x[3] << 24u) | (x[2] >> 8u);
    w[6] = (x[2] << 24u) | (x[1] >> 8u);
    w[7] = (x[1] << 24u) | (x[0] >> 8u);
    // Word 8: x[0] low 8 bits, then 0x80, then 00 00
    // x[0] & 0xFF is the last byte.
    // Then 0x80.
    w[8] = (x[0] << 24u) | 0x00800000u;

    w[9] = 0u;
    w[10] = 0u;
    w[11] = 0u;
    w[12] = 0u;
    w[13] = 0u;
    w[14] = 0u;
    w[15] = 264u; // Length = 33 bytes * 8 = 264 bits.

    // Extend to 64 words
    for (var i = 16u; i < 64u; i++) {
        let s0 = gamma0(w[i-15u]);
        let s1 = gamma1(w[i-2u]);
        w[i] = w[i-16u] + s0 + w[i-7u] + s1;
    }

    // Compression
    var a = h[0]; var b = h[1]; var c = h[2]; var d = h[3];
    var e = h[4]; var f = h[5]; var g = h[6]; var h_var = h[7];

    let k = array<u32, 64>(
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
    );

    // Manual unrolling not needed for modern GPUs usually, but loop is fine
    for (var i = 0u; i < 64u; i++) {
        let t1 = h_var + sigma1(e) + ch(e, f, g) + k[i] + w[i];
        let t2 = sigma0(a) + maj(a, b, c);
        h_var = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    h[0] = h[0] + a;
    h[1] = h[1] + b;
    h[2] = h[2] + c;
    h[3] = h[3] + d;
    h[4] = h[4] + e;
    h[5] = h[5] + f;
    h[6] = h[6] + g;
    h[7] = h[7] + h_var;

    return h;
}

// BIP340 Tagged Hash for TapTweak (BIP341)
// tagged_hash("TapTweak", x) = SHA256(SHA256("TapTweak") || SHA256("TapTweak") || x)
//
// Precomputed midstate after processing first block (tag_hash || tag_hash):
// This allows computing the tagged hash with just one SHA256 compression.
fn tagged_hash_taptweak(x: array<u32, 8>) -> array<u32, 8> {
    // Midstate after SHA256 compression of (SHA256("TapTweak") || SHA256("TapTweak"))
    // Precomputed offline to save GPU cycles
    var h = array<u32, 8>(
        0xd129a2f3u, 0x701c655du, 0x6583b6c3u, 0xb9419727u,
        0x95f4e232u, 0x94fd54f4u, 0xa2ae8d85u, 0x47ca590bu
    );

    // Second block: x || 0x80 || zeros || length (768 bits = 0x300)
    // x is 32 bytes (8 words), then padding to 64 bytes
    var w: array<u32, 64>;

    // x coordinate (32 bytes = 8 words)
    w[0] = x[7]; w[1] = x[6]; w[2] = x[5]; w[3] = x[4];
    w[4] = x[3]; w[5] = x[2]; w[6] = x[1]; w[7] = x[0];

    // Padding: 0x80 followed by zeros
    w[8] = 0x80000000u;
    w[9] = 0u; w[10] = 0u; w[11] = 0u; w[12] = 0u; w[13] = 0u;

    // Length in bits: 64 + 32 = 96 bytes = 768 bits = 0x300
    w[14] = 0u;
    w[15] = 768u;

    // Extend to 64 words
    for (var i = 16u; i < 64u; i++) {
        let s0 = gamma0(w[i-15u]);
        let s1 = gamma1(w[i-2u]);
        w[i] = w[i-16u] + s0 + w[i-7u] + s1;
    }

    // Compression
    var a = h[0]; var b = h[1]; var c = h[2]; var d = h[3];
    var e = h[4]; var f = h[5]; var g = h[6]; var h_var = h[7];

    let k = array<u32, 64>(
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
    );

    for (var i = 0u; i < 64u; i++) {
        let t1 = h_var + sigma1(e) + ch(e, f, g) + k[i] + w[i];
        let t2 = sigma0(a) + maj(a, b, c);
        h_var = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    // Return hash as LE limbs (field element order: limbs[0] = LSB)
    // SHA256 output is BE words (h[0] = MSB), so we just reverse order
    // No byte-swap needed: both SHA256 words and field limbs use same internal byte order
    return array<u32, 8>(
        h[7] + h_var,
        h[6] + g,
        h[5] + f,
        h[4] + e,
        h[3] + d,
        h[2] + c,
        h[1] + b,
        h[0] + a
    );
}
