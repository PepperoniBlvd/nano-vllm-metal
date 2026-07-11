import mlx.core as mx

from nano_vllm_metal.cache.paged import PrefixCache, TinyKvPagedCache, TinyKvPagedPool

H, D = 2, 8


def _run_request(pool, pc, num_layers, token_ids):
    """Simulate one request: attach any cached prefix, prefill the rest, register."""
    caches = [TinyKvPagedCache(pool) for _ in range(num_layers)]
    matched = pc.match_prefix(token_ids)
    max_reuse_blocks = max(0, (len(token_ids) - 1) // pc.page_size)
    reused = pc.attach(matched[:max_reuse_blocks], caches)
    off = reused
    while off < len(token_ids):
        n = min(pc.page_size, len(token_ids) - off)
        for c in caches:
            c.update_and_fetch_paged(mx.zeros((1, H, n, D)), mx.zeros((1, H, n, D)))
        off += n
    pc.register(token_ids, caches)
    for c in caches:
        c.release()
    return caches


def test_lru_eviction_bounds_cache_and_leaves_no_leak():
    page_size, num_layers = 4, 2
    pool = TinyKvPagedPool(page_size=page_size)
    pc = PrefixCache(pool, num_layers, max_blocks=3)

    # 5 distinct prefixes, each 2 full blocks -> 10 blocks would be cached unbounded.
    for r in range(5):
        _run_request(pool, pc, num_layers, list(range(r * 100, r * 100 + 9)))

    assert pc.num_blocks <= 3, pc.num_blocks
    assert pc.evicted_blocks > 0

    # Every used page is held by exactly the cache index (refcount 1); no leak.
    cached_pages = {pid for e in pc.entries.values() for pid in e["pages"]}
    assert pool.used_page_ids == cached_pages
    assert all(pool.ref_counts[p] == 1 for p in cached_pages)

    # clear() returns everything to the free list.
    pc.clear()
    assert pool.used_page_ids == set()
    assert pool.num_free_pages == pool.num_pages


def test_recently_used_prefix_survives_eviction():
    # Capacity must be big enough to hold `hot` plus a churning prefix at once,
    # so eviction has a real choice between the two (hot=2 blocks + churn=2).
    page_size, num_layers = 4, 2
    pool = TinyKvPagedPool(page_size=page_size)
    pc = PrefixCache(pool, num_layers, max_blocks=4)

    hot = list(range(9))  # 2 full blocks; we keep touching this prefix
    _run_request(pool, pc, num_layers, hot)
    hot_hashes = pc.match_prefix(hot)
    assert len(hot_hashes) == 2

    # Churn several distinct cold prefixes, touching `hot` right before each so
    # it stays the most-recently-used and the cold prefixes are evicted instead.
    for r in range(1, 6):
        pc.match_prefix(hot)  # keep hot warm
        _run_request(pool, pc, num_layers, list(range(r * 100, r * 100 + 9)))

    assert pc.match_prefix(hot) == hot_hashes  # hot survived
    assert pc.evicted_blocks > 0  # cold prefixes were evicted


def test_in_use_blocks_are_never_evicted():
    page_size, num_layers = 4, 1
    pool = TinyKvPagedPool(page_size=page_size)
    pc = PrefixCache(pool, num_layers, max_blocks=1)

    pinned = list(range(9))  # 2 full blocks
    _run_request(pool, pc, num_layers, pinned)

    # Attach the pinned prefix to a live request and DO NOT release it.
    live = [TinyKvPagedCache(pool) for _ in range(num_layers)]
    matched = pc.match_prefix(pinned)
    pc.attach(matched, live)

    # Registering many more prefixes cannot evict the in-use blocks.
    for r in range(1, 5):
        _run_request(pool, pc, num_layers, list(range(r * 100, r * 100 + 9)))

    assert pc.match_prefix(pinned) == matched  # still fully cached
    for c in live:
        c.release()
