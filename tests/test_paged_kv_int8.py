import mlx.core as mx

from nano_vllm_metal.layers.attention import paged_attention, scaled_dot_product_attention_grouped
from nano_vllm_metal.cache.paged import TinyKvPagedCache, TinyKvPagedPool


def test_int8_paged_attention_decode_matches_dequantized_dense_attention():
    mx.random.seed(0)
    page_size = 4
    pool = TinyKvPagedPool(page_size=page_size, kv_bits=8)
    cache = TinyKvPagedCache(pool=pool)

    first_key = mx.random.normal(shape=(1, 2, 3, 16)).astype(mx.bfloat16)
    first_value = mx.random.normal(shape=(1, 2, 3, 16)).astype(mx.bfloat16)
    second_key = mx.random.normal(shape=(1, 2, 3, 16)).astype(mx.bfloat16)
    second_value = mx.random.normal(shape=(1, 2, 3, 16)).astype(mx.bfloat16)

    cache.update_and_fetch(first_key, first_value)
    metadata = cache.update_and_fetch_paged(second_key, second_value, mask="causal")
    query = mx.random.normal(shape=(1, 4, 1, 16)).astype(mx.float32)

    dense_key, dense_value = cache.gather_dense()
    dense_output = scaled_dot_product_attention_grouped(
        query,
        dense_key,
        dense_value,
        mask="causal",
    )
    paged_output = paged_attention(
        query,
        metadata.key_pages,
        metadata.value_pages,
        metadata.block_table,
        metadata.context_lens,
        metadata.page_size,
        key_scales=metadata.key_scales,
        value_scales=metadata.value_scales,
        mask=metadata.mask,
    )

    assert metadata.key_pages.dtype == mx.int8
    assert metadata.value_pages.dtype == mx.int8
    assert metadata.key_scales is not None
    assert metadata.value_scales is not None
    assert metadata.key_scales.shape == (2, 2, page_size, 1)
    assert mx.allclose(paged_output, dense_output, rtol=1e-4, atol=1e-4).item()
