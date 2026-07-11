import mlx.core as mx
from .. import metal

from .activation import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    factor = scale if scale is not None else 1.0 / (key.shape[-1] ** 0.5)
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * factor
    if mask is not None:
        scores = scores + mask
    return mx.matmul(softmax(scores, axis=-1), value)


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        q = mx.matmul(query, self.wq.T)
        k = mx.matmul(key, self.wk.T)
        v = mx.matmul(value, self.wv.T)

        q = q.reshape(*q.shape[:-1], self.num_heads, self.head_dim).swapaxes(-3, -2)
        k = k.reshape(*k.shape[:-1], self.num_heads, self.head_dim).swapaxes(-3, -2)
        v = v.reshape(*v.shape[:-1], self.num_heads, self.head_dim).swapaxes(-3, -2)

        x = scaled_dot_product_attention_simple(q, k, v, mask=mask)
        x = x.swapaxes(-3, -2).reshape(*query.shape[:-1], self.hidden_size)
        return mx.matmul(x, self.wo.T)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    allowed = mx.tril(mx.ones((L, S)), k=S - L)
    return mx.where(allowed, mx.array(0), mx.array(-mx.inf)).astype(dtype)


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    expected_shape = query.shape
    num_query_heads, query_length, head_dim = query.shape[-3:]
    num_kv_heads, key_length, _ = key.shape[-3:]
    batch_shape = query.shape[:-3]
    assert num_query_heads % num_kv_heads == 0
    num_repeats = num_query_heads // num_kv_heads

    query = query.reshape(
        *batch_shape, 1, num_kv_heads, num_repeats, query_length, head_dim
    )
    key = key.reshape(*batch_shape, 1, num_kv_heads, 1, key_length, head_dim)
    value = value.reshape(*batch_shape, 1, num_kv_heads, 1, key_length, head_dim)

    factor = scale if scale is not None else 1.0 / (head_dim**0.5)
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * factor

    if isinstance(mask, str):
        assert mask == "causal"
        scores = scores + causal_mask(query_length, key_length, scores.dtype)
    elif mask is not None:
        mask = mx.broadcast_to(
            mask, (*batch_shape, num_query_heads, query_length, key_length)
        )
        mask = mask.reshape(
            *batch_shape, 1, num_kv_heads, num_repeats, query_length, key_length
        )
        scores = scores + mask

    output = mx.matmul(softmax(scores, axis=-1), value)
    return output.reshape(expected_shape)


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    factor = scale if scale is not None else 1.0 / (query.shape[-1] ** 0.5)
    *batch_shape, num_query_heads, query_length, head_dim = query.shape
    _, num_kv_heads, key_length, _ = key.shape
    assert num_query_heads % num_kv_heads == 0

    query = mx.contiguous(query.reshape(-1, query_length, head_dim))
    key = mx.contiguous(key.reshape(-1, key_length, head_dim))
    value = mx.contiguous(value.reshape(-1, key_length, head_dim))

    is_causal = mask == "causal"
    if isinstance(mask, str):
        assert mask == "causal"
        mask = mx.broadcast_to(
            causal_mask(query_length, key_length, mx.float32),
            (*batch_shape, num_query_heads, query_length, key_length),
        )
    elif mask is None:
        mask = mx.broadcast_to(
            mx.zeros((query_length, key_length), dtype=mx.float32),
            (*batch_shape, num_query_heads, query_length, key_length),
        )
    else:
        mask = mx.broadcast_to(
            mask,
            (*batch_shape, num_query_heads, query_length, key_length),
        )
    mask = mx.contiguous(mask.reshape(query.shape[0], query_length, key_length)).astype(
        mx.float32
    )
    result = metal.flash_attention(
        query,
        key,
        value,
        mask,
        factor,
        is_causal=is_causal,
        num_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
    )
    return mx.contiguous(
        result.reshape(*batch_shape, num_query_heads, query_length, head_dim)
    )


def paged_attention(
    query: mx.array,
    key_pages: mx.array,
    value_pages: mx.array,
    block_table: mx.array,
    context_lens: mx.array,
    page_size: int,
    key_scales: mx.array | None = None,
    value_scales: mx.array | None = None,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    if isinstance(mask, mx.array):
        raise NotImplementedError("Paged attention only supports mask=None or causal")
    if mask is not None and mask != "causal":
        raise NotImplementedError

    factor = scale if scale is not None else 1.0 / (query.shape[-1] ** 0.5)
    batch_size, num_query_heads, query_length, head_dim = query.shape
    _, num_kv_heads, _, _ = key_pages.shape
    assert num_query_heads % num_kv_heads == 0

    query = mx.contiguous(
        query.astype(mx.float32).reshape(
            batch_size * num_query_heads, query_length, head_dim
        )
    )
    key_pages = mx.contiguous(key_pages)
    value_pages = mx.contiguous(value_pages)
    block_table = mx.contiguous(block_table.astype(mx.int32))
    context_lens = mx.contiguous(context_lens.astype(mx.int32))
    if key_pages.dtype == mx.int8:
        assert key_scales is not None
        assert value_scales is not None
        key_scales = mx.contiguous(key_scales)
        value_scales = mx.contiguous(value_scales)
        if query_length == 1:
            result = metal.paged_attention_int8_decode(
                query,
                key_pages,
                value_pages,
                key_scales.astype(mx.float32),
                value_scales.astype(mx.float32),
                block_table,
                context_lens,
                float(factor),
                is_causal=mask == "causal",
                num_heads=num_query_heads,
                num_kv_heads=num_kv_heads,
            )
        else:
            result = metal.paged_attention(
                query,
                key_pages.astype(mx.float32) * key_scales.astype(mx.float32),
                value_pages.astype(mx.float32) * value_scales.astype(mx.float32),
                block_table,
                context_lens,
                float(factor),
                is_causal=mask == "causal",
                num_heads=num_query_heads,
                num_kv_heads=num_kv_heads,
            )
    else:
        result = metal.paged_attention(
            query,
            key_pages,
            value_pages,
            block_table,
            context_lens,
            float(factor),
            is_causal=mask == "causal",
            num_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
        )
    return mx.contiguous(
        result.reshape(batch_size, num_query_heads, query_length, head_dim)
    )
