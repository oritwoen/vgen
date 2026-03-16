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

// Step 2 (P2TR variant): Batch normalize + output X coordinate
// 1. Batch normalize to get affine coordinates
// 2. BIP340: ensure even Y
// 3. Output x-only pubkey for P2TR address construction
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

    // Single inversion (guard against Z=0 point at infinity)
    if (local_idx == 0u) {
        let total = prefix[255];
        if (fe_is_zero(total)) {
            inv_total_shared = fe_one();
        } else {
            inv_total_shared = fe_inv(total);
        }
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
