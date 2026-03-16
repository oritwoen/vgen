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
