from abc import ABC, abstractmethod
from typing import Optional

import mlx.core as mx
from ..layers.attention import causal_mask


class TinyKvCache(ABC):
    @abstractmethod
    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        """
        Update the key-value cache and fetch the updated key-value cache.

        Args:
            key: The key to update the cache with.
            value: The value to update the cache with.
            mask_length: The length of the mask (only used in batching mode)
            mask: The mask to use (only used in batching mode)

        Returns:
            A tuple of (updated keys, updated values, sequence length, mask).
            The sequence length and mask let the batching KV cache build the
            per-request attention mask; simple single-request callers can ignore
            them.
        """

    def release(self):
        return None

    def update_and_fetch_paged(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ):
        raise NotImplementedError("This KV cache does not support paged attention")

    def rewind(self, n: int):
        raise NotImplementedError("This KV cache does not support rewind")


class BatchingKvCache(TinyKvCache):
    def __init__(self, max_active_requests: int, max_seq_len: int):
        self.max_active_requests = max_active_requests
        self.max_seq_len = max_seq_len
        self.kv_caches: list[TinyKvCache | None] = [None] * max_active_requests
        self.HD = None

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        B, H, S, D = keys.shape
        assert keys.shape == values.shape
        assert S <= self.max_seq_len
        assert B == self.max_active_requests
        if self.HD is None:
            self.HD = (H, D)
        else:
            assert self.HD == (H, D), f"expect {self.HD} but got {(H, D)}"

        data = []
        for b in range(B):
            cache = self.kv_caches[b]
            if cache is None:
                data.append(None)
                continue
            key, value, seq_len, row_mask = cache.update_and_fetch(
                keys[b : b + 1], values[b : b + 1], mask=mask
            )
            data.append((key[0], value[0], seq_len, row_mask))

        seq_len = max((item[2] if item is not None else 0 for item in data), default=0)
        batched_keys = mx.zeros((B, H, seq_len, D), dtype=keys.dtype)
        batched_values = mx.zeros((B, H, seq_len, D), dtype=values.dtype)

        if mask_length is None:
            mask_length = S
        masks = mx.full((B, mask_length, seq_len), -mx.inf, dtype=keys.dtype)
        for b, item in enumerate(data):
            if item is None:
                continue
            key, value, row_seq_len, row_mask = item
            start = seq_len - row_seq_len
            batched_keys[b, :, start:seq_len, :] = key
            batched_values[b, :, start:seq_len, :] = value
            if row_mask is None or row_mask == "causal":
                masks[b, :, start:seq_len] = causal_mask(
                    mask_length, row_seq_len, dtype=keys.dtype
                )
            elif isinstance(row_mask, mx.array):
                masks[b, :, start:seq_len] = row_mask
            else:
                raise NotImplementedError

        return (
            batched_keys,
            batched_values,
            None,
            masks.reshape(B, 1, mask_length, seq_len),
        )

    def update_and_fetch_paged(
        self,
        keys: mx.array,
        values: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ):
        from .paged import PagedKvMetadata, TinyKvPagedCache

        B, H, S, D = keys.shape
        assert keys.shape == values.shape
        assert S <= self.max_seq_len
        assert B == self.max_active_requests
        if self.HD is None:
            self.HD = (H, D)
        else:
            assert self.HD == (H, D), f"expect {self.HD} but got {(H, D)}"

        pool = None
        context_lens = []
        max_pages = 0
        for b in range(B):
            cache = self.kv_caches[b]
            if cache is None:
                context_lens.append(0)
                continue
            if not isinstance(cache, TinyKvPagedCache):
                raise ValueError("BatchingKvCache contains a non-paged request cache")
            cache.update_and_fetch_paged(
                keys[b : b + 1],
                values[b : b + 1],
                mask_length=mask_length,
                mask=mask,
            )
            if pool is None:
                pool = cache.pool
            elif pool is not cache.pool:
                raise ValueError("Paged batch caches must share one page pool")
            context_lens.append(cache.offset)
            max_pages = max(max_pages, cache.num_pages)

        if pool is None:
            raise ValueError("Cannot build paged metadata without active requests")

        rows = []
        for cache in self.kv_caches:
            if cache is None:
                rows.append([-1] * max_pages)
            else:
                rows.append(cache.page_ids + [-1] * (max_pages - cache.num_pages))

        return PagedKvMetadata(
            key_pages=pool.key_pages,
            value_pages=pool.value_pages,
            block_table=mx.array(rows, dtype=mx.int32),
            context_lens=mx.array(context_lens, dtype=mx.int32),
            page_size=pool.page_size,
            key_scales=pool.key_scales,
            value_scales=pool.value_scales,
            mask=mask,
        )

    def add_request(self, prefilled: TinyKvCache, id: int):
        if id >= self.max_active_requests:
            raise ValueError(f"Request id {id} is out of range")
        if isinstance(prefilled, TinyKvFullCache) and prefilled.key_values is not None:
            keys, _ = prefilled.key_values
            B, H, _, D = keys.shape
            assert B == 1
            if self.HD is None:
                self.HD = (H, D)
            else:
                assert self.HD == (H, D)
        self.kv_caches[id] = prefilled

    def remove_request(self, id: int):
        if self.kv_caches[id] is None:
            raise ValueError(f"Request id {id} is not in the cache")
        self.kv_caches[id].release()
        self.kv_caches[id] = None


class TinyKvFullCache(TinyKvCache):
    def __init__(self):
        self.key_values = None
        self.offset = 0

    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        assert key.shape == value.shape
        B, H, S, D = key.shape
        if self.key_values is None:
            assert self.offset == 0
            self.key_values = (key, value)
            self.offset = S
            return key, value, self.offset, mask

        prev_key, prev_value = self.key_values
        assert prev_key.shape == (B, H, self.offset, D)
        assert prev_value.shape == (B, H, self.offset, D)
        new_key = mx.concat([prev_key, key], axis=2)
        new_value = mx.concat([prev_value, value], axis=2)
        self.key_values = (new_key, new_value)
        self.offset += S
        return new_key, new_value, self.offset, mask

    def rewind(self, n: int):
        assert 0 <= n <= self.offset
        self.offset -= n
        if self.offset == 0:
            self.key_values = None
            return
        assert self.key_values is not None
        self.key_values = (
            self.key_values[0][:, :, : self.offset],
            self.key_values[1][:, :, : self.offset],
        )
