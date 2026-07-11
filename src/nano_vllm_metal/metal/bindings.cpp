// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "kernels.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
    m.doc() = "nano-vllm-metal Metal kernels for MLX";

    m.def("load_library", &nano_vllm_metal::load_library, "device"_a, "path"_a);

    m.def("quantized_matmul", &nano_vllm_metal::quantized_matmul, "scales"_a, "biases"_a, "group_size"_a, "bits"_a,
          "a"_a, "b"_a, "transpose_b"_a = false, "stream"_a = nb::none(),
          R"(
        Quantized matmul layer

        Args:
            scales (array): Scaling factors for ``a``.
            biases (array): Biases for ``a``.
            group_size (int): Group size for ``a``.
            bits (int): Number of bits for ``a``.
            a (array): Input array.
            b (array): Input array.
            transpose_b (bool): Whether to transpose ``b`` before multiplication.

        Returns:
            array: ``a * b``
      )");

    m.def("flash_attention", &nano_vllm_metal::flash_attention, "query"_a, "key"_a, "value"_a, "mask"_a, "scale"_a = 1.0,
          "is_causal"_a = false, "num_kv_heads"_a, "num_heads"_a, "stream"_a = nb::none(), R"(
        Flash attention layer

        Args:
            query (array): Query array.
            key (array): Key array.
            value (array): Value array.
            mask (array): Mask array.
            scale (float): Scaling factor.
            is_causal (bool): Enable causal-mask fast path.

        Returns:
            array: ``softmax(query @ key.T * scale) @ value``
      )");

    m.def("paged_attention", &nano_vllm_metal::paged_attention, "query"_a, "key_pages"_a, "value_pages"_a,
          "block_table"_a, "context_lens"_a, "scale"_a = 1.0, "is_causal"_a = false, "num_kv_heads"_a, "num_heads"_a,
          "stream"_a = nb::none(), R"(
        Paged attention layer

        Args:
            query (array): Query array with shape [B * H_q, L, D].
            key_pages (array): Key page storage with shape [P, H_kv, page_size, D].
            value_pages (array): Value page storage with shape [P, H_kv, page_size, D].
            block_table (array): Physical page ids with shape [B, max_pages].
            context_lens (array): Valid context length for each request.
            scale (float): Scaling factor.
            is_causal (bool): Enable causal masking.

        Returns:
            array: ``softmax(query @ paged_key.T * scale) @ paged_value``
      )");

    m.def("paged_attention_int8_decode", &nano_vllm_metal::paged_attention_int8_decode, "query"_a, "key_pages"_a,
          "value_pages"_a, "key_scales"_a, "value_scales"_a, "block_table"_a, "context_lens"_a, "scale"_a = 1.0,
          "is_causal"_a = false, "num_kv_heads"_a, "num_heads"_a, "stream"_a = nb::none(), R"(
        Decode-only paged attention over int8 KV pages.

        Args:
            query (array): Query array with shape [B * H_q, 1, D].
            key_pages (array): Int8 key page storage with shape [P, H_kv, page_size, D].
            value_pages (array): Int8 value page storage with shape [P, H_kv, page_size, D].
            key_scales (array): Per-token key scales with shape [P, H_kv, page_size, 1].
            value_scales (array): Per-token value scales with shape [P, H_kv, page_size, 1].

        Returns:
            array: ``softmax(query @ dequantized_paged_key.T * scale) @ dequantized_paged_value``
      )");
}
