#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

template <typename T>
[[kernel]] void quantized_matmul_w4a16_g128(
    device const T* scales [[buffer(0)]],
    device const T* biases [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const uint32_t* b [[buffer(3)]],
    device T* out [[buffer(4)]],
    device const int &M [[buffer(5)]],
    device const int &N [[buffer(6)]],
    device const int &K [[buffer(7)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    [[maybe_unused]] threadgroup char * shmem [[threadgroup(0)]]) {
    const int bits = 4;
    const int group_size = 128;
    const int packs_per_item = 32 / bits;
    const int groups_per_row = N / group_size;
    // Each thread processes an element in the output matrix
    const int i = group_id.x * threads_per_threadgroup.x + thread_id.x;
    const int k = group_id.y * threads_per_threadgroup.y + thread_id.y;
    float sum = 0;
    int scales_biases_loc = k * groups_per_row;
    const int mask = (1 << bits) - 1;
    // A: M * N, B: K * N where N gets quantized
    if (i < M && k < K) {
        int b_loc = k * N / packs_per_item;
        int a_loc = i * N;
        for (int group_idx = 0; group_idx < groups_per_row; group_idx++) {
            const float scale = scales[scales_biases_loc];
            const float bias = biases[scales_biases_loc];
            for (int item_idx = 0; item_idx < group_size; item_idx += packs_per_item) {
                uint32_t b_val_packed = b[b_loc];
                sum += (static_cast<float>((b_val_packed >> 0) & mask) * scale + bias) * static_cast<float>(a[a_loc]);
                sum += (static_cast<float>((b_val_packed >> 4) & mask) * scale + bias) * static_cast<float>(a[a_loc + 1]);
                sum += (static_cast<float>((b_val_packed >> 8) & mask) * scale + bias) * static_cast<float>(a[a_loc + 2]);
                sum += (static_cast<float>((b_val_packed >> 12) & mask) * scale + bias) * static_cast<float>(a[a_loc + 3]);
                sum += (static_cast<float>((b_val_packed >> 16) & mask) * scale + bias) * static_cast<float>(a[a_loc + 4]);
                sum += (static_cast<float>((b_val_packed >> 20) & mask) * scale + bias) * static_cast<float>(a[a_loc + 5]);
                sum += (static_cast<float>((b_val_packed >> 24) & mask) * scale + bias) * static_cast<float>(a[a_loc + 6]);
                sum += (static_cast<float>((b_val_packed >> 28) & mask) * scale + bias) * static_cast<float>(a[a_loc + 7]);
                a_loc += packs_per_item;
                b_loc += 1;
            }
            scales_biases_loc += 1;
        }
        out[i * K + k] = static_cast<T>(sum);
    }
}

// Decode/small-batch GEMV path: one SIMD group (32 lanes) cooperates on a
// single output element. Each lane reduces a contiguous 1/32 slice of the
// contraction dimension N with coalesced quantized-weight loads, then the
// partial sums are combined with a single simd_sum. Compared to the
// one-thread-per-output kernel above this uses 32x more threads per output to
// hide memory latency and issues coalesced reads of `b` across the SIMD group.
template <typename T>
[[kernel]] void quantized_matmul_gemv_w4_g128(
    device const T* scales [[buffer(0)]],
    device const T* biases [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const uint32_t* b [[buffer(3)]],
    device T* out [[buffer(4)]],
    device const int &M [[buffer(5)]],
    device const int &N [[buffer(6)]],
    device const int &K [[buffer(7)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]]) {
    const int bits = 4;
    const int group_size = 128;
    const int packs_per_item = 32 / bits;  // 8 values per uint32
    const uint mask = (1 << bits) - 1;

    const int simdgroups_per_tg = threads_per_threadgroup.x / 32;
    // Output column (row of the quantized weight matrix) handled here.
    const int k = group_id.x * simdgroups_per_tg + simd_gid;
    // Row of the (small) activation matrix a.
    const int row_m = group_id.y;
    if (k >= K || row_m >= M) {
        return;
    }

    const int groups_per_row = N / group_size;
    const int total_packs = N / packs_per_item;
    const int packs_per_lane = total_packs / 32;

    const device T* a_row = a + row_m * N;
    const device uint32_t* b_row = b + (size_t)k * total_packs;
    const device T* scales_row = scales + (size_t)k * groups_per_row;
    const device T* biases_row = biases + (size_t)k * groups_per_row;

    // Interleaved pack assignment: lane L owns packs L, L+32, L+64, ... so that
    // within each iteration the 32 lanes read 32 consecutive uint32 words from
    // b (coalesced), and the 32*8 activation values they touch are contiguous.
    float sum = 0.0f;
    for (int p = 0; p < packs_per_lane; p++) {
        const int pack = p * 32 + simd_lid;
        const int elem = pack * packs_per_item;
        const int g = elem / group_size;
        const float scale = static_cast<float>(scales_row[g]);
        const float bias = static_cast<float>(biases_row[g]);
        const uint32_t b_val = b_row[pack];
        for (int j = 0; j < packs_per_item; j++) {
            const float w = static_cast<float>((b_val >> (j * bits)) & mask) * scale + bias;
            sum += w * static_cast<float>(a_row[elem + j]);
        }
    }

    sum = simd_sum(sum);
    if (simd_lid == 0) {
        out[row_m * K + k] = static_cast<T>(sum);
    }
}

instantiate_kernel("quantized_matmul_w4a16_g128_f16", quantized_matmul_w4a16_g128, half);
instantiate_kernel("quantized_matmul_w4a16_g128_bf16", quantized_matmul_w4a16_g128, bfloat16_t);
instantiate_kernel("quantized_matmul_gemv_w4_g128_f16", quantized_matmul_gemv_w4_g128, half);
instantiate_kernel("quantized_matmul_gemv_w4_g128_bf16", quantized_matmul_gemv_w4_g128, bfloat16_t);
