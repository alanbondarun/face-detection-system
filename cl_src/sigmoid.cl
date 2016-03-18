// kernels for forward/backwarding in sigmiod layers

__kernel void sigmoid_forward(__constant float* prev_a,
        __constant float* weight,
        __constant float* bias,
        __global float* cur_z,
        __global float* cur_a,
        __global float* dropout_coeffs,
        const int prev_d,
        const int cur_d,
        const int train_num)
{
    int idxr = get_global_id(0);
    int idxc = idxr % cur_d;

    // matrix-vector multiplication
    cur_z[idxr] = 0;
    for (int idxp = 0; idxp < prev_d; idxp++)
    {
        cur_z[idxr] += weight[idxc*prev_d + idxp] * prev_a[idxp];
    }

    float tmp_a = cur_z[idxr] + bias[idxc];
    tmp_a = 1.0 / (1.0 + exp(-tmp_a));
    cur_a[idxr] = tmp_a * dropout_coeffs[idxc];
}
