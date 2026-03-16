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
// Batch Affine Inversion Kernels
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
    // Guard against Z=0 (point at infinity) which would make prefix[255]=0
    // This is astronomically rare (requires Base = -i*G for some i in batch)
    if (local_idx == 0u) {
        let total = prefix[255];
        if (fe_is_zero(total)) {
            inv_total_shared = fe_one(); // Fallback: affected points will have invalid coords
        } else {
            inv_total_shared = fe_inv(total);
        }
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
