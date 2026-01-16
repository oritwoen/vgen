// =============================================================================
// Bitcoin Vanity Address Generator - GPU Kernel (Optimized)
// =============================================================================

// -----------------------------------------------------------------------------
// Field Arithmetic (same as before)
// -----------------------------------------------------------------------------

const P0: u32 = 0xFFFFFC2Fu;
const P1: u32 = 0xFFFFFFFEu;
const P2: u32 = 0xFFFFFFFFu;
const P3: u32 = 0xFFFFFFFFu;
const P4: u32 = 0xFFFFFFFFu;
const P5: u32 = 0xFFFFFFFFu;
const P6: u32 = 0xFFFFFFFFu;
const P7: u32 = 0xFFFFFFFFu;

fn fold_single(acc: ptr<function, array<u32, 17>>, h: u32, offset: u32) {
    if (h == 0u) { return; }
    let t = mul32(h, 977u);
    var sum = (*acc)[offset] + t.x;
    var carry = select(0u, 1u, sum < (*acc)[offset]);
    (*acc)[offset] = sum;
    sum = (*acc)[offset + 1u] + t.y + carry;
    carry = select(0u, 1u, sum < (*acc)[offset + 1u] || (carry == 1u && sum == (*acc)[offset + 1u]));
    (*acc)[offset + 1u] = sum;
    sum = (*acc)[offset + 1u] + h;
    carry = carry + select(0u, 1u, sum < (*acc)[offset + 1u]);
    (*acc)[offset + 1u] = sum;
    var k = offset + 2u;
    loop {
        if (carry == 0u || k >= 17u) { break; }
        sum = (*acc)[k] + carry;
        carry = select(0u, 1u, sum < (*acc)[k]);
        (*acc)[k] = sum;
        k = k + 1u;
    }
}

fn fe_cond_sub_p(val: array<u32, 8>) -> array<u32, 8> {
    var c = val;
    var tmp: array<u32, 8>;
    var borrow: u32 = 0u;
    var diff: u32;
    diff = c[0] - P0; borrow = select(0u, 1u, c[0] < P0); tmp[0] = diff;
    diff = c[1] - P1 - borrow; borrow = select(0u, 1u, c[1] < P1 + borrow || (borrow == 1u && P1 == 0xFFFFFFFFu)); tmp[1] = diff;
    diff = c[2] - P2 - borrow; borrow = select(0u, 1u, c[2] < P2 + borrow || (borrow == 1u && P2 == 0xFFFFFFFFu)); tmp[2] = diff;
    diff = c[3] - P3 - borrow; borrow = select(0u, 1u, c[3] < P3 + borrow || (borrow == 1u && P3 == 0xFFFFFFFFu)); tmp[3] = diff;
    diff = c[4] - P4 - borrow; borrow = select(0u, 1u, c[4] < P4 + borrow || (borrow == 1u && P4 == 0xFFFFFFFFu)); tmp[4] = diff;
    diff = c[5] - P5 - borrow; borrow = select(0u, 1u, c[5] < P5 + borrow || (borrow == 1u && P5 == 0xFFFFFFFFu)); tmp[5] = diff;
    diff = c[6] - P6 - borrow; borrow = select(0u, 1u, c[6] < P6 + borrow || (borrow == 1u && P6 == 0xFFFFFFFFu)); tmp[6] = diff;
    diff = c[7] - P7 - borrow; borrow = select(0u, 1u, c[7] < P7 + borrow || (borrow == 1u && P7 == 0xFFFFFFFFu)); tmp[7] = diff;
    if (borrow == 0u) { c = tmp; }
    return c;
}

fn fe_add(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    var c: array<u32, 8>;
    var carry: u32 = 0u;
    var sum: u32;
    sum = a[0] + b[0]; carry = select(0u, 1u, sum < a[0]); c[0] = sum;
    sum = a[1] + b[1] + carry; carry = select(0u, 1u, sum < a[1] || (carry == 1u && sum == a[1])); c[1] = sum;
    sum = a[2] + b[2] + carry; carry = select(0u, 1u, sum < a[2] || (carry == 1u && sum == a[2])); c[2] = sum;
    sum = a[3] + b[3] + carry; carry = select(0u, 1u, sum < a[3] || (carry == 1u && sum == a[3])); c[3] = sum;
    sum = a[4] + b[4] + carry; carry = select(0u, 1u, sum < a[4] || (carry == 1u && sum == a[4])); c[4] = sum;
    sum = a[5] + b[5] + carry; carry = select(0u, 1u, sum < a[5] || (carry == 1u && sum == a[5])); c[5] = sum;
    sum = a[6] + b[6] + carry; carry = select(0u, 1u, sum < a[6] || (carry == 1u && sum == a[6])); c[6] = sum;
    sum = a[7] + b[7] + carry; carry = select(0u, 1u, sum < a[7] || (carry == 1u && sum == a[7])); c[7] = sum;
    if (carry == 1u) {
        var old: u32;
        old = c[0]; c[0] = c[0] + 977u; carry = select(0u, 1u, c[0] < old);
        old = c[1]; c[1] = c[1] + 1u + carry; carry = select(0u, 1u, c[1] < old || (carry == 1u && c[1] == old));
        old = c[2]; c[2] = c[2] + carry; carry = select(0u, 1u, c[2] < old);
        old = c[3]; c[3] = c[3] + carry; carry = select(0u, 1u, c[3] < old);
        old = c[4]; c[4] = c[4] + carry; carry = select(0u, 1u, c[4] < old);
        old = c[5]; c[5] = c[5] + carry; carry = select(0u, 1u, c[5] < old);
        old = c[6]; c[6] = c[6] + carry; carry = select(0u, 1u, c[6] < old);
        old = c[7]; c[7] = c[7] + carry; carry = select(0u, 1u, c[7] < old);
    }
    return fe_cond_sub_p(c);
}

fn fe_sub(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    var c: array<u32, 8>;
    var borrow: u32 = 0u;
    var diff: u32;
    diff = a[0] - b[0]; borrow = select(0u, 1u, a[0] < b[0]); c[0] = diff;
    diff = a[1] - b[1] - borrow; borrow = select(0u, 1u, a[1] < b[1] + borrow || (borrow == 1u && b[1] == 0xFFFFFFFFu)); c[1] = diff;
    diff = a[2] - b[2] - borrow; borrow = select(0u, 1u, a[2] < b[2] + borrow || (borrow == 1u && b[2] == 0xFFFFFFFFu)); c[2] = diff;
    diff = a[3] - b[3] - borrow; borrow = select(0u, 1u, a[3] < b[3] + borrow || (borrow == 1u && b[3] == 0xFFFFFFFFu)); c[3] = diff;
    diff = a[4] - b[4] - borrow; borrow = select(0u, 1u, a[4] < b[4] + borrow || (borrow == 1u && b[4] == 0xFFFFFFFFu)); c[4] = diff;
    diff = a[5] - b[5] - borrow; borrow = select(0u, 1u, a[5] < b[5] + borrow || (borrow == 1u && b[5] == 0xFFFFFFFFu)); c[5] = diff;
    diff = a[6] - b[6] - borrow; borrow = select(0u, 1u, a[6] < b[6] + borrow || (borrow == 1u && b[6] == 0xFFFFFFFFu)); c[6] = diff;
    diff = a[7] - b[7] - borrow; c[7] = diff; borrow = select(0u, 1u, a[7] < b[7] + borrow || (borrow == 1u && b[7] == 0xFFFFFFFFu));
    if (borrow == 1u) {
        var carry2: u32 = 0u;
        var sum2: u32;
        sum2 = c[0] + P0; carry2 = select(0u, 1u, sum2 < c[0]); c[0] = sum2;
        sum2 = c[1] + P1 + carry2; carry2 = select(0u, 1u, sum2 < c[1] || (carry2 == 1u && sum2 == c[1])); c[1] = sum2;
        sum2 = c[2] + P2 + carry2; carry2 = select(0u, 1u, sum2 < c[2] || (carry2 == 1u && sum2 == c[2])); c[2] = sum2;
        sum2 = c[3] + P3 + carry2; carry2 = select(0u, 1u, sum2 < c[3] || (carry2 == 1u && sum2 == c[3])); c[3] = sum2;
        sum2 = c[4] + P4 + carry2; carry2 = select(0u, 1u, sum2 < c[4] || (carry2 == 1u && sum2 == c[4])); c[4] = sum2;
        sum2 = c[5] + P5 + carry2; carry2 = select(0u, 1u, sum2 < c[5] || (carry2 == 1u && sum2 == c[5])); c[5] = sum2;
        sum2 = c[6] + P6 + carry2; carry2 = select(0u, 1u, sum2 < c[6] || (carry2 == 1u && sum2 == c[6])); c[6] = sum2;
        sum2 = c[7] + P7 + carry2; c[7] = sum2;
    }
    return c;
}

fn mul32(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;
    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;
    let mid = p1 + p2;
    let mid_overflow = select(0u, 1u, mid < p1);
    let lo1 = p0 + (mid << 16u);
    let lo_carry1 = select(0u, 1u, lo1 < p0);
    let hi1 = p3 + (mid >> 16u) + (mid_overflow << 16u) + lo_carry1;
    return vec2<u32>(lo1, hi1);
}

fn fe_mul(a: array<u32, 8>, b: array<u32, 8>) -> array<u32, 8> {
    var prod: array<u32, 16>;
    for (var i = 0u; i < 16u; i++) { prod[i] = 0u; }
    for (var i = 0u; i < 8u; i++) {
        for (var j = 0u; j < 8u; j++) {
            let idx = i + j;
            let t = mul32(a[i], b[j]);
            var sum = prod[idx] + t.x;
            var carry = select(0u, 1u, sum < prod[idx]);
            prod[idx] = sum;
            sum = prod[idx + 1u] + t.y + carry;
            carry = select(0u, 1u, sum < prod[idx + 1u] || (carry == 1u && sum == prod[idx + 1u]));
            prod[idx + 1u] = sum;
            var k = idx + 2u;
            loop {
                if (carry == 0u) { break; }
                sum = prod[k] + carry;
                carry = select(0u, 1u, sum < prod[k]);
                prod[k] = sum;
                k = k + 1u;
            }
        }
    }
    var acc: array<u32, 17>;
    for (var i = 0u; i < 17u; i++) { acc[i] = 0u; }
    for (var i = 0u; i < 16u; i++) { acc[i] = prod[i]; }
    loop {
        var has_high = false;
        for (var idx = 8u; idx < 17u; idx++) {
            let h = acc[idx];
            if (h == 0u) { continue; }
            has_high = true;
            acc[idx] = 0u;
            fold_single(&acc, h, idx - 8u);
        }
        if (!has_high) { break; }
    }
    var result: array<u32, 8>;
    for (var i = 0u; i < 8u; i++) { result[i] = acc[i]; }
    return fe_cond_sub_p(fe_cond_sub_p(result));
}

fn fe_square(a: array<u32, 8>) -> array<u32, 8> { return fe_mul(a, a); }
fn fe_one() -> array<u32, 8> { return array<u32, 8>(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u); }
fn fe_zero() -> array<u32, 8> { return array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u); }
fn fe_is_zero(a: array<u32, 8>) -> bool {
    return a[0] == 0u && a[1] == 0u && a[2] == 0u && a[3] == 0u &&
           a[4] == 0u && a[5] == 0u && a[6] == 0u && a[7] == 0u;
}
fn fe_double(a: array<u32, 8>) -> array<u32, 8> { return fe_add(a, a); }

// Negate in field: -a = p - a
fn fe_neg(a: array<u32, 8>) -> array<u32, 8> {
    if (fe_is_zero(a)) { return a; }
    var result: array<u32, 8>;
    var borrow: u32 = 0u;
    // result = P - a (using safe borrow check to avoid overflow)
    var diff = P0 - a[0]; borrow = select(0u, 1u, P0 < a[0]); result[0] = diff;
    diff = P1 - a[1] - borrow; borrow = select(0u, 1u, P1 < a[1] || (borrow == 1u && a[1] == 0xFFFFFFFFu)); result[1] = diff;
    diff = P2 - a[2] - borrow; borrow = select(0u, 1u, P2 < a[2] || (borrow == 1u && a[2] == 0xFFFFFFFFu)); result[2] = diff;
    diff = P3 - a[3] - borrow; borrow = select(0u, 1u, P3 < a[3] || (borrow == 1u && a[3] == 0xFFFFFFFFu)); result[3] = diff;
    diff = P4 - a[4] - borrow; borrow = select(0u, 1u, P4 < a[4] || (borrow == 1u && a[4] == 0xFFFFFFFFu)); result[4] = diff;
    diff = P5 - a[5] - borrow; borrow = select(0u, 1u, P5 < a[5] || (borrow == 1u && a[5] == 0xFFFFFFFFu)); result[5] = diff;
    diff = P6 - a[6] - borrow; borrow = select(0u, 1u, P6 < a[6] || (borrow == 1u && a[6] == 0xFFFFFFFFu)); result[6] = diff;
    diff = P7 - a[7] - borrow; result[7] = diff;
    return result;
}

fn fe_inv(a: array<u32, 8>) -> array<u32, 8> {
    var res = fe_one();
    var base = a;
    let exp = array<u32, 8>(
        0xFFFFFC2Du, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
        0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu
    );
    for (var i = 0u; i < 8u; i++) {
        let limb = exp[i];
        for (var j = 0u; j < 32u; j++) {
            if ((limb >> j) & 1u) == 1u { res = fe_mul(res, base); }
            base = fe_square(base);
        }
    }
    return res;
}

// -----------------------------------------------------------------------------
// Curve Operations
// -----------------------------------------------------------------------------

struct JacobianPoint {
    x: array<u32, 8>,
    y: array<u32, 8>,
    z: array<u32, 8>
}

struct AffinePoint {
    x: array<u32, 8>,
    y: array<u32, 8>
}

fn jac_is_infinity(p: JacobianPoint) -> bool { return fe_is_zero(p.z); }
fn jac_infinity() -> JacobianPoint {
    var p: JacobianPoint;
    p.x = fe_one(); p.y = fe_one(); p.z = fe_zero();
    return p;
}

fn jac_double(p: JacobianPoint) -> JacobianPoint {
    if (jac_is_infinity(p)) { return p; }
    let a = fe_square(p.x);
    let b = fe_square(p.y);
    let c = fe_square(b);
    let xpb = fe_add(p.x, b);
    let xpb2 = fe_square(xpb);
    let d = fe_double(fe_sub(fe_sub(xpb2, a), c));
    let e = fe_add(fe_add(a, a), a);
    let f = fe_square(e);
    let x3 = fe_sub(f, fe_double(d));
    let y3 = fe_sub(fe_mul(e, fe_sub(d, x3)), fe_double(fe_double(fe_double(c))));
    let z3 = fe_double(fe_mul(p.y, p.z));
    var res: JacobianPoint;
    res.x = x3; res.y = y3; res.z = z3;
    return res;
}

fn jac_add(p: JacobianPoint, q: JacobianPoint) -> JacobianPoint {
    if (jac_is_infinity(p)) { return q; }
    if (jac_is_infinity(q)) { return p; }
    let z1z1 = fe_square(p.z);
    let z1z1z1 = fe_mul(z1z1, p.z);
    let z2z2 = fe_square(q.z);
    let z2z2z2 = fe_mul(z2z2, q.z);
    let u1 = fe_mul(p.x, z2z2);
    let u2 = fe_mul(q.x, z1z1);
    let s1 = fe_mul(p.y, z2z2z2);
    let s2 = fe_mul(q.y, z1z1z1);
    let h = fe_sub(u2, u1);
    let r = fe_double(fe_sub(s2, s1));
    if (fe_is_zero(h)) {
        if (fe_is_zero(r)) { return jac_double(p); }
        return jac_infinity();
    }
    let hh = fe_square(h);
    let i = fe_double(fe_double(hh));
    let j = fe_mul(h, i);
    let v = fe_mul(u1, i);
    let x3 = fe_sub(fe_sub(fe_square(r), j), fe_double(v));
    let y3 = fe_sub(fe_mul(r, fe_sub(v, x3)), fe_double(fe_mul(s1, j)));
    let z3 = fe_mul(fe_sub(fe_sub(fe_square(fe_add(p.z, q.z)), z1z1), z2z2), h);
    var res: JacobianPoint;
    res.x = x3; res.y = y3; res.z = z3;
    return res;
}

// Add Jacobian P + Affine Q (where Q.z = 1)
// Optimized to avoid multiplications by Q.z (which is 1)
fn jac_add_affine(p: JacobianPoint, q: AffinePoint) -> JacobianPoint {
    if (jac_is_infinity(p)) {
        var res: JacobianPoint;
        res.x = q.x; res.y = q.y; res.z = fe_one();
        return res;
    }
    // If Q is zero/infinity (0, 0), return P
    if (fe_is_zero(q.x) && fe_is_zero(q.y)) {
        return p;
    }

    // P + Q(x2, y2, 1)
    // Z2 = 1, Z2^2 = 1, Z2^3 = 1
    // U1 = X1 * Z2^2 = X1
    let u1 = p.x;
    // U2 = X2 * Z1^2
    let z1z1 = fe_square(p.z);
    let u2 = fe_mul(q.x, z1z1);

    // S1 = Y1 * Z2^3 = Y1
    let s1 = p.y;
    // S2 = Y2 * Z1^3
    let z1z1z1 = fe_mul(z1z1, p.z);
    let s2 = fe_mul(q.y, z1z1z1);

    let h = fe_sub(u2, u1);
    let r = fe_double(fe_sub(s2, s1));

    if (fe_is_zero(h)) {
        if (fe_is_zero(r)) { return jac_double(p); }
        return jac_infinity();
    }

    let hh = fe_square(h);
    let i = fe_double(fe_double(hh));
    let j = fe_mul(h, i);
    let v = fe_mul(u1, i);
    let x3 = fe_sub(fe_sub(fe_square(r), j), fe_double(v));
    let y3 = fe_sub(fe_mul(r, fe_sub(v, x3)), fe_double(fe_mul(s1, j)));
    // Z3 = ((Z1 + 1)^2 - Z1^2 - 1) * H = (Z1^2 + 2Z1 + 1 - Z1^2 - 1) * H = 2Z1 * H
    // Or just standard formula with Z2=1:
    // Z3 = ((Z1+1)^2 - Z1^2 - 1) * H
    let z1_plus_1 = fe_add(p.z, fe_one());
    let z3 = fe_mul(fe_sub(fe_sub(fe_square(z1_plus_1), z1z1), fe_one()), h);

    var res: JacobianPoint;
    res.x = x3; res.y = y3; res.z = z3;
    return res;
}

fn jac_normalize(p: JacobianPoint) -> AffinePoint {
    if (fe_is_zero(p.z)) { return AffinePoint(fe_zero(), fe_zero()); }
    let z_inv = fe_inv(p.z);
    let z_inv2 = fe_square(z_inv);
    let z_inv3 = fe_mul(z_inv2, z_inv);
    var res: AffinePoint;
    res.x = fe_mul(p.x, z_inv2);
    res.y = fe_mul(p.y, z_inv3);
    return res;
}

fn get_generator() -> JacobianPoint {
    var p: JacobianPoint;
    p.x = array<u32, 8>(0x16F81798u, 0x59F2815Bu, 0x2DCE28D9u, 0x029BFCDBu, 0xCE870B07u, 0x55A06295u, 0xF9DCBBACu, 0x79BE667Eu);
    p.y = array<u32, 8>(0xFB10D4B8u, 0x9C47D08Fu, 0xA6855419u, 0xFD17B448u, 0x0E1108A8u, 0x5DA4FBFCu, 0x26A3C465u, 0x483ADA77u);
    p.z = fe_one();
    return p;
}

fn scalar_mul_G(k: array<u32, 8>) -> JacobianPoint {
    var res = jac_infinity();
    var base = get_generator();
    for (var i = 0u; i < 8u; i++) {
        let limb = k[i];
        for (var j = 0u; j < 32u; j++) {
            if ((limb >> j) & 1u) == 1u { res = jac_add(res, base); }
            base = jac_double(base);
        }
    }
    return res;
}

fn scalar_to_u256(val: u32) -> array<u32, 8> {
    return array<u32, 8>(val, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
}

// -----------------------------------------------------------------------------
// Kernels
// -----------------------------------------------------------------------------

struct BigInt256 {
    v0: vec4<u32>,
    v1: vec4<u32>,
}

struct Config {
    base_x: BigInt256,
    base_y: BigInt256,
    num_keys: u32,
    _pad0: u32, _pad1: u32, _pad2: u32, // align
}

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read_write> table_rw: array<AffinePoint>;
// Output is array of RIPEMD160 hashes (5 u32s each)
@group(0) @binding(2) var<storage, read_write> output_hashes: array<array<u32, 5>>;
// Temporary storage for Jacobian points (for batch affine inversion)
@group(0) @binding(3) var<storage, read_write> jacobian_points: array<JacobianPoint>;
// Output for P2TR: X coordinates (8 u32s = 32 bytes each)
@group(0) @binding(4) var<storage, read_write> output_x_coords: array<array<u32, 8>>;

fn unpack_bigint(b: BigInt256) -> array<u32, 8> {
    return array<u32, 8>(b.v0.x, b.v0.y, b.v0.z, b.v0.w, b.v1.x, b.v1.y, b.v1.z, b.v1.w);
}

// Initialization Kernel: Precomputes table[i] = i * G
// Used once at startup.
@compute @workgroup_size(256)
fn init_table(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= config.num_keys) { return; }
    let k = scalar_to_u256(idx);
    let p_jac = scalar_mul_G(k);
    table_rw[idx] = jac_normalize(p_jac);
}

// Search Kernel
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= config.num_keys) { return; }

    // Load Base Pubkey
    var base_pub: JacobianPoint;
    base_pub.x = unpack_bigint(config.base_x);
    base_pub.y = unpack_bigint(config.base_y);
    base_pub.z = fe_one(); // Base passed as Affine, promoted to Jacobian

    // Load Precomputed Point (i * G)
    let point_i = table_rw[idx];

    // P = Base + i*G
    let p_res = jac_add_affine(base_pub, point_i);
    let p_aff = jac_normalize(p_res);

    // Serialize to Compressed Pubkey
    let parity = p_aff.y[0] & 1u; // 0x02 if even, 0x03 if odd

    // Call SHA256 (from sha256.wgsl)
    let sha_out = sha256_compressed_pubkey(parity, p_aff.x);

    // Call RIPEMD160 (from ripemd160.wgsl)
    let ripemd_out = ripemd160(sha_out);

    // Store Result (Hash160 bytes)
    output_hashes[idx] = ripemd_out;
}

// =============================================================================
// Batch Affine Inversion Kernels (Blelloch scan optimization)
// =============================================================================

// Step 1: Compute Jacobian points without normalization
@compute @workgroup_size(256)
fn compute_jacobian(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= config.num_keys) { return; }

    // Load Base Pubkey
    var base_pub: JacobianPoint;
    base_pub.x = unpack_bigint(config.base_x);
    base_pub.y = unpack_bigint(config.base_y);
    base_pub.z = fe_one();

    // Load Precomputed Point (i * G)
    let point_i = table_rw[idx];

    // P = Base + i*G (Jacobian, not normalized)
    let p_res = jac_add_affine(base_pub, point_i);

    // Store Jacobian point for batch normalization
    jacobian_points[idx] = p_res;
}

// Shared memory for fully parallel Montgomery batch inversion
// Total: 256 * 32 * 2 + 32 = ~16KB per workgroup
var<workgroup> prefix: array<array<u32, 8>, 256>;   // prefix products
var<workgroup> suffix: array<array<u32, 8>, 256>;   // suffix products
var<workgroup> inv_total_shared: array<u32, 8>;     // broadcast slot

// Step 2: Fully parallel Montgomery batch inversion, output Hash160
// Complexity: O(log n) parallel steps instead of O(n) sequential
@compute @workgroup_size(256)
fn batch_normalize_hash(@builtin(global_invocation_id) gid: vec3<u32>,
                        @builtin(local_invocation_id) lid: vec3<u32>) {
    let local_idx = lid.x;
    let global_idx = gid.x;

    // Load Z into both prefix and suffix arrays
    var z_val: array<u32, 8>;
    if (global_idx < config.num_keys) {
        z_val = jacobian_points[global_idx].z;
    } else {
        z_val = fe_one();
    }
    prefix[local_idx] = z_val;
    suffix[local_idx] = z_val;
    workgroupBarrier();

    // Phase 1a: Parallel inclusive prefix products (left to right)
    // prefix[i] = Z[0] * Z[1] * ... * Z[i]
    for (var stride = 1u; stride < 256u; stride *= 2u) {
        var val = prefix[local_idx];
        if (local_idx >= stride) {
            val = fe_mul(prefix[local_idx - stride], val);
        }
        workgroupBarrier();
        prefix[local_idx] = val;
        workgroupBarrier();
    }

    // Phase 1b: Parallel inclusive suffix products (right to left)
    // suffix[i] = Z[i] * Z[i+1] * ... * Z[255]
    for (var stride = 1u; stride < 256u; stride *= 2u) {
        var val = suffix[local_idx];
        if (local_idx + stride < 256u) {
            val = fe_mul(val, suffix[local_idx + stride]);
        }
        workgroupBarrier();
        suffix[local_idx] = val;
        workgroupBarrier();
    }

    // Phase 2: Single inversion of total product
    if (local_idx == 0u) {
        inv_total_shared = fe_inv(prefix[255]);
    }
    workgroupBarrier();
    let inv_total = inv_total_shared;

    // Phase 3: Parallel per-element inverse computation
    // Z_inv[i] = prefix[i-1] * inv_total * suffix[i+1]
    var z_inv: array<u32, 8>;
    if (local_idx == 0u) {
        // Z_inv[0] = inv_total * suffix[1]
        z_inv = fe_mul(inv_total, suffix[1]);
    } else if (local_idx == 255u) {
        // Z_inv[255] = prefix[254] * inv_total
        z_inv = fe_mul(prefix[254], inv_total);
    } else {
        // Z_inv[i] = prefix[i-1] * inv_total * suffix[i+1]
        let tmp = fe_mul(prefix[local_idx - 1u], inv_total);
        z_inv = fe_mul(tmp, suffix[local_idx + 1u]);
    }

    if (global_idx >= config.num_keys) { return; }

    let z_inv2 = fe_square(z_inv);
    let z_inv3 = fe_mul(z_inv2, z_inv);

    let p = jacobian_points[global_idx];
    var p_aff: AffinePoint;
    p_aff.x = fe_mul(p.x, z_inv2);
    p_aff.y = fe_mul(p.y, z_inv3);

    // Compute Hash160
    let parity = p_aff.y[0] & 1u;
    let sha_out = sha256_compressed_pubkey(parity, p_aff.x);
    let ripemd_out = ripemd160(sha_out);
    output_hashes[global_idx] = ripemd_out;
}

// Step 2 (P2TR variant): Full Taproot key tweaking on GPU
// 1. Batch normalize to get affine coordinates
// 2. Compute tweak: t = tagged_hash("TapTweak", x_only)
// 3. Compute t*G
// 4. Add P + t*G to get tweaked pubkey Q
// 5. Output Q.x (the tweaked x-only pubkey for P2TR address)
@compute @workgroup_size(256)
fn batch_normalize_p2tr(@builtin(global_invocation_id) gid: vec3<u32>,
                        @builtin(local_invocation_id) lid: vec3<u32>) {
    let local_idx = lid.x;
    let global_idx = gid.x;

    // Load Z into both prefix and suffix arrays for batch inversion
    var z_val: array<u32, 8>;
    if (global_idx < config.num_keys) {
        z_val = jacobian_points[global_idx].z;
    } else {
        z_val = fe_one();
    }
    prefix[local_idx] = z_val;
    suffix[local_idx] = z_val;
    workgroupBarrier();

    // Parallel inclusive prefix products
    for (var stride = 1u; stride < 256u; stride *= 2u) {
        var val = prefix[local_idx];
        if (local_idx >= stride) {
            val = fe_mul(prefix[local_idx - stride], val);
        }
        workgroupBarrier();
        prefix[local_idx] = val;
        workgroupBarrier();
    }

    // Parallel inclusive suffix products
    for (var stride = 1u; stride < 256u; stride *= 2u) {
        var val = suffix[local_idx];
        if (local_idx + stride < 256u) {
            val = fe_mul(val, suffix[local_idx + stride]);
        }
        workgroupBarrier();
        suffix[local_idx] = val;
        workgroupBarrier();
    }

    // Single inversion
    if (local_idx == 0u) {
        inv_total_shared = fe_inv(prefix[255]);
    }
    workgroupBarrier();
    let inv_total = inv_total_shared;

    // Per-element inverse
    var z_inv: array<u32, 8>;
    if (local_idx == 0u) {
        z_inv = fe_mul(inv_total, suffix[1]);
    } else if (local_idx == 255u) {
        z_inv = fe_mul(prefix[254], inv_total);
    } else {
        let tmp = fe_mul(prefix[local_idx - 1u], inv_total);
        z_inv = fe_mul(tmp, suffix[local_idx + 1u]);
    }

    if (global_idx >= config.num_keys) { return; }

    // Normalize to affine
    let p_jac = jacobian_points[global_idx];
    let z_inv2 = fe_square(z_inv);
    let z_inv3 = fe_mul(z_inv2, z_inv);
    let x_aff = fe_mul(p_jac.x, z_inv2);
    var y_aff = fe_mul(p_jac.y, z_inv3);

    // BIP340: If Y is odd, negate Y to get even-Y point for x-only representation
    // The internal key uses the point with even Y
    let y_is_odd = (y_aff[0] & 1u) == 1u;
    if (y_is_odd) {
        y_aff = fe_neg(y_aff);
    }

    // Output internal key X coordinate
    // Taproot tweak is computed on CPU (GPU instruction limit prevents full tweak on GPU)
    output_x_coords[global_idx] = x_aff;
}
