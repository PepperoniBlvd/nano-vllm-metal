from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

from .kv_cache import TinyKvCache


@dataclass
class PagedKvMetadata:
    key_pages: mx.array
    value_pages: mx.array
    block_table: mx.array
    context_lens: mx.array
    page_size: int
    key_scales: mx.array | None = None
    value_scales: mx.array | None = None
    mask: mx.array | str | None = None


class TinyKvPagedPool:
    """Model-local physical storage for paged KV.

    The model owns one pool and passes it to every layer cache. The pool gives
    out physical page ids from one free list. Because every live page id is
    unique, the page id alone is enough to find the physical K/V page.
    """

    def __init__(self, page_size: int = 128, kv_bits: int | None = None):
        assert page_size > 0
        if kv_bits is not None and kv_bits != 8:
            raise ValueError("TinyKvPagedPool currently supports only int8 KV")
        self.page_size = page_size
        self.kv_bits = kv_bits
        self.key_pages: mx.array | None = None
        self.value_pages: mx.array | None = None
        self.key_scales: mx.array | None = None
        self.value_scales: mx.array | None = None
        self.free_page_ids: list[int] = []
        self.used_page_ids: set[int] = set()
        self.num_allocated_pages = 0
        # Per-page reference count. A page can be referenced by more than one
        # owner once prefix caching shares physical pages across requests (the
        # owning request cache and the prefix-cache index both hold a ref). A
        # page returns to the free list only when its refcount reaches zero.
        self.ref_counts: dict[int, int] = {}

    @property
    def num_pages(self) -> int:
        return self.num_allocated_pages

    @property
    def num_free_pages(self) -> int:
        return len(self.free_page_ids)

    def _check_page_chunk(self, x: mx.array) -> None:
        B, H, S, D = x.shape
        assert 0 < S <= self.page_size

    def allocate_page(self) -> int:
        # The page id is allocated from a model-wide free list. In this teaching
        # version, a layer cache owns the page until release/rewind returns it.
        if self.free_page_ids:
            page_id = self.free_page_ids.pop()
        else:
            page_id = self.num_pages
            self.num_allocated_pages += 1
        self.used_page_ids.add(page_id)
        self.ref_counts[page_id] = 1
        return page_id

    def incref(self, page_id: int) -> None:
        if page_id not in self.used_page_ids:
            raise ValueError(f"Page {page_id} is free")
        self.ref_counts[page_id] += 1

    def read_page(self, page_id: int) -> tuple[mx.array, mx.array]:
        if self.key_pages is None or self.value_pages is None:
            raise ValueError(f"Page {page_id} has no storage")
        if page_id >= self.num_pages:
            raise ValueError(f"Page {page_id} is out of range")
        if self.kv_bits == 8:
            assert self.key_scales is not None
            assert self.value_scales is not None
            return (
                self.key_pages[page_id : page_id + 1].astype(mx.float32)
                * self.key_scales[page_id : page_id + 1].astype(mx.float32),
                self.value_pages[page_id : page_id + 1].astype(mx.float32)
                * self.value_scales[page_id : page_id + 1].astype(mx.float32),
            )
        return (
            self.key_pages[page_id : page_id + 1],
            self.value_pages[page_id : page_id + 1],
        )

    def _ensure_page_storage(self, key: mx.array, value: mx.array) -> None:
        B, H, _, D = key.shape
        assert B == 1

        storage_dtype = mx.int8 if self.kv_bits == 8 else key.dtype

        cur_capacity = 0
        if self.key_pages is not None and self.value_pages is not None:
            assert self.key_pages.shape[1:] == (H, self.page_size, D)
            assert self.value_pages.shape == self.key_pages.shape
            assert self.key_pages.dtype == storage_dtype
            assert self.value_pages.dtype == storage_dtype
            cur_capacity = self.key_pages.shape[0]
            if cur_capacity >= self.num_pages:
                return

        # Amortized (geometric) growth. Reallocating + copying the whole pool on
        # every single new page is O(n^2) over a decode (the pool grows one page
        # at a time across all layers). Doubling the capacity makes reallocation
        # O(log n) total with O(n) total copying. num_pages (logical, live page
        # count) is unchanged; only the backing-store capacity grows. Tiny pools
        # stay exactly sized (reallocation there is cheap, and unit tests assert
        # the exact backing-store shape); amortization kicks in past 16 pages.
        if self.num_pages <= 16:
            new_capacity = self.num_pages
        else:
            new_capacity = max(self.num_pages, cur_capacity * 2)

        new_key_pages = mx.zeros(
            (new_capacity, H, self.page_size, D), dtype=storage_dtype
        )
        new_value_pages = mx.zeros(
            (new_capacity, H, self.page_size, D), dtype=storage_dtype
        )
        if self.key_pages is not None and self.value_pages is not None:
            old_pages = self.key_pages.shape[0]
            new_key_pages[:old_pages, :, :, :] = self.key_pages
            new_value_pages[:old_pages, :, :, :] = self.value_pages
        self.key_pages = new_key_pages
        self.value_pages = new_value_pages

        if self.kv_bits == 8:
            scale_dtype = key.dtype
            new_key_scales = mx.ones(
                (new_capacity, H, self.page_size, 1), dtype=scale_dtype
            )
            new_value_scales = mx.ones(
                (new_capacity, H, self.page_size, 1), dtype=scale_dtype
            )
            if self.key_scales is not None and self.value_scales is not None:
                old_pages = self.key_scales.shape[0]
                new_key_scales[:old_pages, :, :, :] = self.key_scales
                new_value_scales[:old_pages, :, :, :] = self.value_scales
            self.key_scales = new_key_scales
            self.value_scales = new_value_scales

    def _quantize_page_slice(self, x: mx.array) -> tuple[mx.array, mx.array]:
        max_abs = mx.max(mx.abs(x.astype(mx.float32)), axis=-1, keepdims=True)
        scale = mx.where(max_abs > 0, max_abs / 127.0, mx.ones_like(max_abs))
        quantized = mx.clip(mx.round(x.astype(mx.float32) / scale), -127, 127).astype(
            mx.int8
        )
        return quantized, scale.astype(x.dtype)

    def write_page_slice(
        self,
        page_id: int,
        start: int,
        key: mx.array,
        value: mx.array,
    ) -> None:
        assert key.shape == value.shape
        self._check_page_chunk(key)
        if page_id not in self.used_page_ids:
            raise ValueError(f"Page {page_id} is free")
        self._ensure_page_storage(key, value)
        assert self.key_pages is not None
        assert self.value_pages is not None
        H, capacity, D = self.key_pages.shape[1:]
        assert self.value_pages.shape == self.key_pages.shape
        assert capacity == self.page_size
        assert key.shape[:2] == (1, H)
        assert key.shape[3] == D
        end = start + key.shape[2]
        assert 0 <= start <= capacity
        assert end <= self.page_size

        if self.kv_bits == 8:
            assert self.key_scales is not None
            assert self.value_scales is not None
            quantized_key, key_scale = self._quantize_page_slice(key)
            quantized_value, value_scale = self._quantize_page_slice(value)
            self.key_pages[page_id, :, start:end, :] = quantized_key[0]
            self.value_pages[page_id, :, start:end, :] = quantized_value[0]
            self.key_scales[page_id, :, start:end, :] = key_scale[0]
            self.value_scales[page_id, :, start:end, :] = value_scale[0]
            return

        self.key_pages[page_id, :, start:end, :] = key[0]
        self.value_pages[page_id, :, start:end, :] = value[0]

    def free_page(self, page_id: int) -> None:
        if page_id not in self.used_page_ids:
            raise ValueError(f"Page {page_id} is already free")
        # Drop one reference. The page only returns to the free list when the
        # last owner releases it; with no prefix caching every page has a single
        # owner so this frees immediately, exactly as before.
        self.ref_counts[page_id] -= 1
        if self.ref_counts[page_id] > 0:
            return
        # Keep the page id stable. The stale K/V bytes can stay in the backing
        # tensor because block_table/page_lens decide which slots are live.
        del self.ref_counts[page_id]
        self.used_page_ids.remove(page_id)
        self.free_page_ids.append(page_id)


class PrefixCache:
    """Model-level block (prefix) cache over a shared page pool.

    Identical prompt prefixes produce identical KV, because a token's K/V depends
    only on that token and its absolute position, and a shared prefix always sits
    at positions [0, len). So a *full* page of KV can be reused across requests
    whenever the entire token prefix up to that page matches.

    We key each full page by a chained hash: hash(parent_page_hash, tokens_in_page).
    Chaining makes a page reusable only when every earlier page also matched, i.e.
    the reuse is always a true contiguous prefix from position 0. Each cache entry
    maps that hash to one physical page id per layer (all layers share one pool).
    The prefix cache holds one reference on every page it stores, so cached pages
    survive after the request that produced them releases its own reference.
    """

    _SEED = 1469598103934665603  # arbitrary non-zero chain seed

    def __init__(
        self,
        pool: "TinyKvPagedPool | list[TinyKvPagedPool]",
        num_layers: int,
        max_blocks: int | None = None,
    ):
        # Accept one pool (shared by all layers) or one pool per layer. Per-layer
        # pools are the norm (see Qwen3PagedModel); a single pool still works for
        # the unit tests. entry["pages"][layer] is a page id in self.pools[layer].
        if isinstance(pool, (list, tuple)):
            assert len(pool) == num_layers
            self.pools = list(pool)
        else:
            self.pools = [pool] * num_layers
        self.pool = self.pools[0]  # legacy alias
        self.page_size = self.pools[0].page_size
        self.num_layers = num_layers
        # Bound on the number of cached blocks. None => unbounded (grows with the
        # pool). Each block pins num_layers physical pages, so this caps the pages
        # the prefix index holds to max_blocks * num_layers.
        self.max_blocks = max_blocks
        # chained_hash -> {"pages": [page_id per layer], "tokens": tuple,
        #                  "parent": hash, "last_used": int}
        self.entries: dict[int, dict] = {}
        self._tick = 0  # monotonic recency counter for LRU
        self.evicted_blocks = 0  # stat

    @property
    def num_blocks(self) -> int:
        return len(self.entries)

    def _chain(self, parent: int, block_tokens: tuple[int, ...]) -> int:
        return hash((parent, block_tokens))

    def _touch(self, block_hash: int) -> None:
        self._tick += 1
        self.entries[block_hash]["last_used"] = self._tick

    def block_hashes(self, token_ids: list[int]) -> list[tuple[int, tuple]]:
        """Chained (hash, block_tokens) for every FULL page of token_ids."""
        ps = self.page_size
        num_full = len(token_ids) // ps
        out = []
        parent = self._SEED
        for p in range(num_full):
            block = tuple(token_ids[p * ps : (p + 1) * ps])
            h = self._chain(parent, block)
            out.append((h, block))
            parent = h
        return out

    def match_prefix(self, token_ids: list[int]) -> list[int]:
        """Longest run of leading full-page hashes present in the cache.

        Returns the list of matched chained hashes (its length is the number of
        reusable pages). Verifies stored tokens + parent chain to rule out hash
        collisions.
        """
        matched: list[int] = []
        parent = self._SEED
        for h, block in self.block_hashes(token_ids):
            entry = self.entries.get(h)
            if entry is None or entry["tokens"] != block or entry["parent"] != parent:
                break
            matched.append(h)
            parent = h
        # A lookup counts as a use for LRU: keep hot prefixes warm.
        for h in matched:
            self._touch(h)
        return matched

    def get_pages(self, block_hash: int) -> list[int]:
        return self.entries[block_hash]["pages"]

    def attach(
        self, matched_hashes: list[int], caches: list["TinyKvPagedCache"]
    ) -> int:
        """Point fresh (empty) per-layer caches at the matched shared pages.

        Bumps the refcount for each shared page (the request now owns a ref too)
        and advances each cache's offset past the reused prefix. Returns the
        number of reused prefix tokens.
        """
        ps = self.page_size
        assert all(c.offset == 0 and c.num_pages == 0 for c in caches), (
            "attach expects fresh per-layer caches"
        )
        for h in matched_hashes:
            pages = self.entries[h]["pages"]
            for layer, cache in enumerate(caches):
                pid = pages[layer]
                self.pools[layer].incref(pid)
                cache.page_ids.append(pid)
                cache.page_lens.append(ps)
        reused = len(matched_hashes) * ps
        for cache in caches:
            cache.offset = reused
        return reused

    def register(self, token_ids: list[int], caches: list["TinyKvPagedCache"]) -> int:
        """Register any not-yet-cached full pages produced for token_ids.

        caches[layer].page_ids[p] is the physical page holding block p for that
        layer. A block is cacheable once every layer has written a full page for
        it. Returns the number of newly registered blocks.
        """
        hashes = self.block_hashes(token_ids)
        added = 0
        parent = self._SEED
        for p, (h, block) in enumerate(hashes):
            if h not in self.entries:
                # Every layer must have a full page p to register the block.
                if any(
                    p >= c.num_pages or c.page_lens[p] != self.page_size for c in caches
                ):
                    break
                pages = [caches[layer].page_ids[p] for layer in range(self.num_layers)]
                for layer, pid in enumerate(pages):
                    self.pools[layer].incref(pid)  # the cache index now owns a ref
                self._tick += 1
                self.entries[h] = {
                    "pages": pages,
                    "tokens": block,
                    "parent": parent,
                    "last_used": self._tick,
                }
                added += 1
            parent = h
        self._evict_to_capacity()
        return added

    def _is_evictable(self, entry: dict) -> bool:
        # A block may be evicted only when no request is using it: every one of
        # its pages must be held solely by this index (refcount == 1). Because a
        # request attaching a prefix increfs all of blocks 0..k-1, an evictable
        # block never has an in-use descendant, so freeing it is always safe.
        return all(
            self.pools[layer].ref_counts.get(pid, 0) == 1
            for layer, pid in enumerate(entry["pages"])
        )

    def _evict_to_capacity(self) -> None:
        if self.max_blocks is None:
            return
        while len(self.entries) > self.max_blocks:
            # Least-recently-used evictable block. Scan skips in-use blocks.
            victim_h = None
            victim_used = None
            for h, entry in self.entries.items():
                if not self._is_evictable(entry):
                    continue
                lu = entry["last_used"]
                if victim_used is None or lu < victim_used:
                    victim_used, victim_h = lu, h
            if victim_h is None:
                break  # everything cached is currently in use; nothing to evict
            entry = self.entries.pop(victim_h)
            for layer, pid in enumerate(entry["pages"]):
                self.pools[layer].free_page(pid)  # refcount 1 -> 0, returns to free
            self.evicted_blocks += 1

    def clear(self) -> None:
        for entry in self.entries.values():
            for layer, pid in enumerate(entry["pages"]):
                self.pools[layer].free_page(pid)
        self.entries.clear()


class TinyKvPagedCache(TinyKvCache):
    """Layer-local K/V cache backed by a model-owned page pool.

    Each transformer layer gets its own TinyKvPagedCache and therefore its own
    `page_ids`, `page_lens`, and `offset`. The shared part is only the pool,
    which lets pages be recycled across requests and layers.
    """

    def __init__(self, pool: TinyKvPagedPool):
        self.pool = pool
        self.page_size = self.pool.page_size
        self.page_ids: list[int] = []
        self.page_lens: list[int] = []
        self.offset = 0

    @property
    def num_pages(self) -> int:
        return len(self.page_ids)

    @property
    def key_values(self) -> tuple[mx.array, mx.array] | None:
        if self.offset == 0:
            return None
        return self.gather_dense()

    def _append_chunk(self, key: mx.array, value: mx.array) -> None:
        assert key.shape == value.shape
        B, H, S, D = key.shape
        assert B == 1, "Paged request cache only supports one request at a time"
        start = 0

        # First fill the existing tail page if it has free slots.
        if self.page_ids and self.page_lens[-1] < self.page_size:
            page_id = self.page_ids[-1]
            page_start = self.page_lens[-1]
            take = min(self.page_size - page_start, S)
            self.pool.write_page_slice(
                page_id,
                page_start,
                key[:, :, :take, :],
                value[:, :, :take, :],
            )
            self.page_lens[-1] += take
            start += take

        # Then allocate fresh pages for the remaining chunk. We only write the
        # valid prefix; unused tail slots are ignored by page_lens.
        while start < S:
            end = min(start + self.page_size, S)
            page_id = self.pool.allocate_page()
            self.pool.write_page_slice(
                page_id,
                0,
                key[:, :, start:end, :],
                value[:, :, start:end, :],
            )
            self.page_ids.append(page_id)
            self.page_lens.append(end - start)
            start = end

        self.offset += S

    def gather_dense(self) -> tuple[mx.array, mx.array]:
        assert self.offset > 0
        # Dense compatibility path for tests and older callers. The paged
        # attention path uses block_table/context_lens instead of this gather.
        key_chunks = []
        value_chunks = []
        for page_id, page_len in zip(self.page_ids, self.page_lens):
            key_page, value_page = self.pool.read_page(page_id)
            assert key_page.shape[2] == self.page_size
            assert value_page.shape[2] == self.page_size
            key_chunks.append(key_page[:, :, :page_len, :])
            value_chunks.append(value_page[:, :, :page_len, :])
        if len(key_chunks) == 1:
            return key_chunks[0], value_chunks[0]
        return mx.concat(key_chunks, axis=2), mx.concat(value_chunks, axis=2)

    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        assert key.shape == value.shape
        self._append_chunk(key, value)
        # Dense-interface compatibility path. The paged attention path uses
        # update_and_fetch_paged instead so attention can read pages directly.
        dense_key, dense_value = self.gather_dense()
        return dense_key, dense_value, self.offset, mask

    def block_table(self, max_pages: int | None = None) -> mx.array:
        if max_pages is None:
            max_pages = self.num_pages
        assert max_pages >= self.num_pages
        page_ids = self.page_ids + [-1] * (max_pages - self.num_pages)
        return mx.array([page_ids], dtype=mx.int32)

    def context_lens(self) -> mx.array:
        return mx.array([self.offset], dtype=mx.int32)

    def paged_metadata(
        self,
        max_pages: int | None = None,
        mask: mx.array | str | None = None,
    ) -> PagedKvMetadata:
        assert self.pool.key_pages is not None
        assert self.pool.value_pages is not None
        return PagedKvMetadata(
            key_pages=self.pool.key_pages,
            value_pages=self.pool.value_pages,
            key_scales=self.pool.key_scales,
            value_scales=self.pool.value_scales,
            block_table=self.block_table(max_pages=max_pages),
            context_lens=self.context_lens(),
            page_size=self.page_size,
            mask=mask,
        )

    def update_and_fetch_paged(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> PagedKvMetadata:
        assert key.shape == value.shape
        self._append_chunk(key, value)
        return self.paged_metadata(mask=mask)

    def rewind(self, n: int):
        assert 0 <= n <= self.offset
        new_offset = self.offset - n
        if new_offset == self.offset:
            return
        if new_offset == 0:
            self.release()
            return

        target_num_pages = (new_offset + self.page_size - 1) // self.page_size
        while len(self.page_ids) > target_num_pages:
            # Whole pages beyond the new logical length return to the shared
            # allocator. Stale suffix slots in the final page are ignored because
            # page_lens defines the valid prefix and future writes overwrite them.
            page_id = self.page_ids.pop()
            self.page_lens.pop()
            self.pool.free_page(page_id)

        last_page_len = new_offset - self.page_size * (target_num_pages - 1)
        self.page_lens[-1] = last_page_len
        self.offset = new_offset

    def release(self):
        # Request completion returns every page owned by this layer cache to the
        # model-level allocator. Other layer caches release their own pages.
        for page_id in self.page_ids:
            self.pool.free_page(page_id)
        self.page_ids.clear()
        self.page_lens.clear()
        self.offset = 0
