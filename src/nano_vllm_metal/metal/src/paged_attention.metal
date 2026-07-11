#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

template <typename T>
[[kernel]] void paged_attention_kv(
    device const float* q [[buffer(0)]],
    device const T* key_pages [[buffer(1)]],
    device const T* value_pages [[buffer(2)]],
    device const int* block_table [[buffer(3)]],
    device const int* context_lens [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant const int& N [[buffer(6)]],
    constant const int& L [[buffer(7)]],
    constant const int& D [[buffer(8)]],
    constant const int& page_size [[buffer(9)]],
    constant const int& max_pages [[buffer(10)]],
    constant const int& is_causal [[buffer(11)]],
    constant const int& num_kv_heads [[buffer(12)]],
    constant const int& num_heads [[buffer(13)]],
    constant const float& scale [[buffer(14)]],
    constant const int& Br [[buffer(15)]],
    constant const int& Bc [[buffer(16)]],
    [[maybe_unused]] constant const int& Tr [[buffer(17)]],
    constant const int& Tc [[buffer(18)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int n = group_id.x;
  const int i = group_id.y;
  const int a = simd_gid;
  const int b = simd_lid;
  const int row = i * Br + a;
  const bool is_i_in_range = n < N && row < L && a < Br;

  const int batch = n / num_heads;
  const int q_head = n % num_heads;
  const int q_kv_ratio = num_heads / num_kv_heads;
  const int kv_head = q_head / q_kv_ratio;
  const int context_len = context_lens[batch];
  const int causal_offset = context_len - L;

  threadgroup float q_local[32][128];
  threadgroup float o_i[32 * 128];

  if (simd_lid == 0) {
    for (int c = 0; c < D; c++) {
      q_local[a][c] = is_i_in_range ? q[n * L * D + row * D + c] : 0.0f;
      o_i[a * D + c] = 0.0f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float m_i = -INFINITY;
  float l_i = 0.0f;

  for (int j = 0; j < Tc; j++) {
    const int col = j * Bc + b;
    if (j * Bc >= context_len) {
      continue;
    }
    if (is_causal) {
      const int row_max = min((i + 1) * Br - 1, L - 1);
      if (j * Bc > row_max + causal_offset) {
        continue;
      }
    }

    const int page_idx = col / page_size;
    const int slot = col - page_idx * page_size;
    int page_id = -1;
    bool is_j_in_range = col < context_len && b < Bc && page_idx < max_pages;
    if (is_j_in_range) {
      page_id = block_table[batch * max_pages + page_idx];
      is_j_in_range = page_id >= 0;
    }

    bool visible = is_i_in_range && is_j_in_range;
    if (visible && is_causal) {
      visible = col <= row + causal_offset;
    }

    float s_a_b = -INFINITY;
    if (visible) {
      float score = 0.0f;
      for (int c = 0; c < D; c++) {
        const int k_idx =
            ((page_id * num_kv_heads + kv_head) * page_size + slot) * D + c;
        score += q_local[a][c] * static_cast<float>(key_pages[k_idx]);
      }
      s_a_b = score * scale;
    }

    const float rowmax = simd_max(s_a_b);
    const float new_max = max(m_i, rowmax);
    const float old_scale = exp(m_i - new_max);
    m_i = new_max;

    const float p_a_b = visible ? exp(s_a_b - m_i) : 0.0f;
    const float rowsum = simd_sum(p_a_b);
    l_i = old_scale * l_i + rowsum;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int c = 0; c < D; c++) {
      float partial = 0.0f;
      if (visible) {
        const int v_idx =
            ((page_id * num_kv_heads + kv_head) * page_size + slot) * D + c;
        partial = p_a_b * static_cast<float>(value_pages[v_idx]);
      }
      const float res = simd_sum(partial);
      if (simd_lid == 0 && is_i_in_range) {
        o_i[a * D + c] = old_scale * o_i[a * D + c] + res;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (simd_lid == 0) {
    for (int c = 0; c < D; c++) {
      if (is_i_in_range) {
        out[n * L * D + row * D + c] =
            l_i > 0.0f ? o_i[a * D + c] / l_i : 0.0f;
      }
    }
  }
}

instantiate_kernel("paged_attention_kv_f32", paged_attention_kv, float);
instantiate_kernel("paged_attention_kv_f16", paged_attention_kv, half);
instantiate_kernel("paged_attention_kv_bf16", paged_attention_kv, bfloat16_t);

template <typename T>
[[kernel]] void paged_attention_decode_kv(
    device const float* q [[buffer(0)]],
    device const T* key_pages [[buffer(1)]],
    device const T* value_pages [[buffer(2)]],
    device const int* block_table [[buffer(3)]],
    device const int* context_lens [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant const int& N [[buffer(6)]],
    [[maybe_unused]] constant const int& L [[buffer(7)]],
    constant const int& D [[buffer(8)]],
    constant const int& page_size [[buffer(9)]],
    constant const int& max_pages [[buffer(10)]],
    constant const int& is_causal [[buffer(11)]],
    constant const int& num_kv_heads [[buffer(12)]],
    constant const int& num_heads [[buffer(13)]],
    constant const float& scale [[buffer(14)]],
    constant const int& Tc [[buffer(15)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int n = static_cast<int>(group_id);
  const int lane = static_cast<int>(simd_lid);
  if (n >= N) {
    return;
  }

  const int batch = n / num_heads;
  const int q_head = n % num_heads;
  const int q_kv_ratio = num_heads / num_kv_heads;
  const int kv_head = q_head / q_kv_ratio;
  const int context_len = context_lens[batch];
  const int causal_limit = context_len - 1;

  threadgroup float o_i[128];
  if (lane == 0) {
    for (int c = 0; c < D; c++) {
      o_i[c] = 0.0f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float m_i = -INFINITY;
  float l_i = 0.0f;

  for (int j = 0; j < Tc; j++) {
    const bool lane_active = lane < 32;
    const int col = j * 32 + lane;
    if (j * 32 >= context_len) {
      continue;
    }

    const int page_idx = col / page_size;
    const int slot = col - page_idx * page_size;
    int page_id = -1;
    bool visible = lane_active && col < context_len && page_idx < max_pages;
    if (visible) {
      page_id = block_table[batch * max_pages + page_idx];
      visible = page_id >= 0;
    }
    if (visible && is_causal) {
      visible = col <= causal_limit;
    }

    float s_i = -INFINITY;
    if (visible) {
      float score = 0.0f;
      for (int c = 0; c < D; c++) {
        const int k_idx =
            ((page_id * num_kv_heads + kv_head) * page_size + slot) * D + c;
        score += q[n * D + c] * static_cast<float>(key_pages[k_idx]);
      }
      s_i = score * scale;
    }

    const float rowmax = simd_max(s_i);
    const float new_max = max(m_i, rowmax);
    const float old_scale = exp(m_i - new_max);
    m_i = new_max;

    const float p_i = visible ? exp(s_i - m_i) : 0.0f;
    const float rowsum = simd_sum(p_i);
    l_i = old_scale * l_i + rowsum;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int c = 0; c < D; c++) {
      float partial = 0.0f;
      if (visible) {
        const int v_idx =
            ((page_id * num_kv_heads + kv_head) * page_size + slot) * D + c;
        partial = p_i * static_cast<float>(value_pages[v_idx]);
      }
      const float res = simd_sum(partial);
      if (lane == 0) {
        o_i[c] = old_scale * o_i[c] + res;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0) {
    for (int c = 0; c < D; c++) {
      out[n * D + c] = l_i > 0.0f ? o_i[c] / l_i : 0.0f;
    }
  }
}

instantiate_kernel("paged_attention_decode_kv_f32", paged_attention_decode_kv, float);
instantiate_kernel("paged_attention_decode_kv_f16", paged_attention_decode_kv, half);
instantiate_kernel(
    "paged_attention_decode_kv_bf16", paged_attention_decode_kv, bfloat16_t);

// Coalesced 16-byte load of 4 contiguous head-dim elements into float[4].
// bfloat16_t has no valid Metal vector type, so load its raw bits as ushort4
// and reinterpret each lane; float/half have native vector types.
inline void load4(const device float* p, thread float o[4]) {
  const float4 v = *reinterpret_cast<const device float4*>(p);
  o[0] = v.x; o[1] = v.y; o[2] = v.z; o[3] = v.w;
}
inline void load4(const device half* p, thread float o[4]) {
  const half4 v = *reinterpret_cast<const device half4*>(p);
  o[0] = float(v.x); o[1] = float(v.y); o[2] = float(v.z); o[3] = float(v.w);
}
inline void load4(const device bfloat16_t* p, thread float o[4]) {
  // bfloat16 is the high 16 bits of a float32, so widening is just a bit shift.
  const ushort4 r = *reinterpret_cast<const device ushort4*>(p);
  o[0] = as_type<float>(uint(r.x) << 16);
  o[1] = as_type<float>(uint(r.y) << 16);
  o[2] = as_type<float>(uint(r.z) << 16);
  o[3] = as_type<float>(uint(r.w) << 16);
}

// Split-K / flash-decoding paged attention (decode, one query token).
//
// The single-SIMD-group decode kernel above walks the whole KV context
// serially, so its cost grows linearly with context length and, at batch=1, it
// launches only num_heads threadgroups -- leaving the GPU idle. This kernel adds
// the third parallelization axis from Flash-Decoding: the KV-length. One
// threadgroup still owns one (batch, head), but it contains `num_splits`
// SIMD-groups, each independently running online-softmax over a strided share of
// the KV chunks and writing its partial (output, running max, running sumexp) to
// threadgroup memory. A final log-sum-exp combine merges the partials. Wall-time
// per threadgroup drops from O(context) to O(context / num_splits).
template <typename T>
[[kernel]] void paged_attention_decode_split_kv(
    device const float* q [[buffer(0)]],
    device const T* key_pages [[buffer(1)]],
    device const T* value_pages [[buffer(2)]],
    device const int* block_table [[buffer(3)]],
    device const int* context_lens [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant const int& N [[buffer(6)]],
    constant const int& D [[buffer(7)]],
    constant const int& page_size [[buffer(8)]],
    constant const int& max_pages [[buffer(9)]],
    constant const int& is_causal [[buffer(10)]],
    constant const int& num_kv_heads [[buffer(11)]],
    constant const int& num_heads [[buffer(12)]],
    constant const float& scale [[buffer(13)]],
    constant const int& num_splits [[buffer(14)]],
    constant const int& Tc [[buffer(15)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int n = static_cast<int>(group_id);
  if (n >= N) {
    return;
  }
  const int s = static_cast<int>(simd_gid);  // this SIMD-group's KV split
  const int lane = static_cast<int>(simd_lid);

  const int batch = n / num_heads;
  const int q_head = n % num_heads;
  const int q_kv_ratio = num_heads / num_kv_heads;
  const int kv_head = q_head / q_kv_ratio;
  const int context_len = context_lens[batch];

  // Lane owns a contiguous slice of the head dimension: dims [lane*dpl, +dpl).
  // The QK dot is a per-lane partial reduced with one simd_sum per KV position;
  // the PV product accumulates entirely in registers (o_reg) with NO reduction
  // and no threadgroup traffic in the hot loop. Adjacent lanes touch adjacent
  // memory, so the head-dim loads are coalesced/vectorizable.
  const int dpl = D / 32;  // dims per lane (D assumed a multiple of 32, <= 256)
  const int d0 = lane * dpl;

  float q_reg[8];
  float o_reg[8];
  for (int k = 0; k < dpl; k++) {
    q_reg[k] = q[n * D + d0 + k];
    o_reg[k] = 0.0f;
  }

  float m_i = -INFINITY;
  float l_i = 0.0f;

  // Split s handles KV positions s, s+num_splits, s+2*num_splits, ...
  for (int col = s; col < context_len; col += num_splits) {
    const int page_idx = col / page_size;
    if (page_idx >= max_pages) {
      continue;
    }
    const int page_id = block_table[batch * max_pages + page_idx];
    if (page_id < 0) {
      continue;
    }
    const int slot = col - page_idx * page_size;
    const int base = ((page_id * num_kv_heads + kv_head) * page_size + slot) * D;
    const device T* kp = key_pages + base + d0;
    const device T* vp = value_pages + base + d0;

    // Load this lane's head-dim slice (16-byte coalesced load for dpl==4).
    float kloc[8];
    float vloc[8];
    if (dpl == 4) {
      load4(kp, kloc);
      load4(vp, vloc);
    } else {
      for (int k = 0; k < dpl; k++) {
        kloc[k] = static_cast<float>(kp[k]);
        vloc[k] = static_cast<float>(vp[k]);
      }
    }

    float dot = 0.0f;
    for (int k = 0; k < dpl; k++) {
      dot += q_reg[k] * kloc[k];
    }
    const float score = simd_sum(dot) * scale;

    const float new_max = max(m_i, score);
    const float old_scale = exp(m_i - new_max);
    m_i = new_max;
    const float p = exp(score - m_i);
    l_i = old_scale * l_i + p;
    for (int k = 0; k < dpl; k++) {
      o_reg[k] = old_scale * o_reg[k] + p * vloc[k];
    }
  }

  // Per-split partials (unnormalized output, running max, running sumexp).
  threadgroup float o_sh[8][128];
  threadgroup float m_sh[8];
  threadgroup float l_sh[8];
  for (int k = 0; k < dpl; k++) {
    o_sh[s][d0 + k] = o_reg[k];
  }
  if (lane == 0) {
    m_sh[s] = m_i;
    l_sh[s] = l_i;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // SIMD-group 0 combines the per-split partials via log-sum-exp rescaling.
  if (s == 0) {
    float global_max = -INFINITY;
    for (int ss = 0; ss < num_splits; ss++) {
      global_max = max(global_max, m_sh[ss]);
    }
    float l_total = 0.0f;
    for (int ss = 0; ss < num_splits; ss++) {
      if (m_sh[ss] > -INFINITY) {
        l_total += exp(m_sh[ss] - global_max) * l_sh[ss];
      }
    }
    for (int k = 0; k < dpl; k++) {
      const int c = d0 + k;
      float acc = 0.0f;
      for (int ss = 0; ss < num_splits; ss++) {
        if (m_sh[ss] > -INFINITY) {
          acc += exp(m_sh[ss] - global_max) * o_sh[ss][c];
        }
      }
      out[n * D + c] = l_total > 0.0f ? acc / l_total : 0.0f;
    }
  }
}

instantiate_kernel(
    "paged_attention_decode_split_kv_f32", paged_attention_decode_split_kv, float);
instantiate_kernel(
    "paged_attention_decode_split_kv_f16", paged_attention_decode_split_kv, half);
instantiate_kernel(
    "paged_attention_decode_split_kv_bf16", paged_attention_decode_split_kv, bfloat16_t);

[[kernel]] void paged_attention_int8_decode_kv(
    device const float* q [[buffer(0)]],
    device const int8_t* key_pages [[buffer(1)]],
    device const int8_t* value_pages [[buffer(2)]],
    device const float* key_scales [[buffer(3)]],
    device const float* value_scales [[buffer(4)]],
    device const int* block_table [[buffer(5)]],
    device const int* context_lens [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant const int& N [[buffer(8)]],
    constant const int& D [[buffer(9)]],
    constant const int& page_size [[buffer(10)]],
    constant const int& max_pages [[buffer(11)]],
    constant const int& is_causal [[buffer(12)]],
    constant const int& num_kv_heads [[buffer(13)]],
    constant const int& num_heads [[buffer(14)]],
    constant const float& scale [[buffer(15)]],
    constant const int& Tc [[buffer(16)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int n = static_cast<int>(group_id);
  const int lane = static_cast<int>(simd_lid);
  if (n >= N) {
    return;
  }

  const int batch = n / num_heads;
  const int q_head = n % num_heads;
  const int q_kv_ratio = num_heads / num_kv_heads;
  const int kv_head = q_head / q_kv_ratio;
  const int context_len = context_lens[batch];
  const int causal_limit = context_len - 1;

  threadgroup float o_i[128];
  if (lane == 0) {
    for (int c = 0; c < D; c++) {
      o_i[c] = 0.0f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float m_i = -INFINITY;
  float l_i = 0.0f;

  for (int j = 0; j < Tc; j++) {
    const bool lane_active = lane < 32;
    const int col = j * 32 + lane;
    if (j * 32 >= context_len) {
      continue;
    }

    const int page_idx = col / page_size;
    const int slot = col - page_idx * page_size;
    int page_id = -1;
    bool visible = lane_active && col < context_len && page_idx < max_pages;
    if (visible) {
      page_id = block_table[batch * max_pages + page_idx];
      visible = page_id >= 0;
    }
    if (visible && is_causal) {
      visible = col <= causal_limit;
    }

    const int scale_idx =
        (page_id * num_kv_heads + kv_head) * page_size + slot;
    float s_i = -INFINITY;
    if (visible) {
      const float key_scale = key_scales[scale_idx];
      float score = 0.0f;
      for (int c = 0; c < D; c++) {
        const int k_idx = scale_idx * D + c;
        score += q[n * D + c] *
                 (static_cast<float>(key_pages[k_idx]) * key_scale);
      }
      s_i = score * scale;
    }

    const float rowmax = simd_max(s_i);
    const float new_max = max(m_i, rowmax);
    const float old_scale = exp(m_i - new_max);
    m_i = new_max;

    const float p_i = visible ? exp(s_i - m_i) : 0.0f;
    const float rowsum = simd_sum(p_i);
    l_i = old_scale * l_i + rowsum;

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int c = 0; c < D; c++) {
      float partial = 0.0f;
      if (visible) {
        const int v_idx = scale_idx * D + c;
        partial = p_i *
                  (static_cast<float>(value_pages[v_idx]) *
                   value_scales[scale_idx]);
      }
      const float res = simd_sum(partial);
      if (lane == 0) {
        o_i[c] = old_scale * o_i[c] + res;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0) {
    for (int c = 0; c < D; c++) {
      out[n * D + c] = l_i > 0.0f ? o_i[c] / l_i : 0.0f;
    }
  }
}
