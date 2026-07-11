"""Serving benchmark: paged continuous batching vs mlx_lm (sequential).

The thesis: in the concurrent-serving regime, our paged + continuous-batching +
prefix-caching stack beats mlx_lm, which has NO batching (`batch_generate` is
absent) and must serve concurrent requests one at a time.

Measures, for a batch of requests with variable prompt lengths:
  - makespan (wall time to finish all requests) and aggregate output tok/s
  - peak KV memory used (paged pool) vs the dense-padded equivalent, and the
    implied max concurrency at a fixed KV-memory budget
Optionally a shared-prefix workload (to show the prefix-caching win).

By default runs the perf backends (matmul=mlx, rope=mlx) since that's the
"serving" configuration; pass --backend custom for the from-scratch showcase.
"""

import argparse
import time
from random import Random

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate

from nano_vllm_metal import loader as models
from nano_vllm_metal.engine.scheduler import batch_generate
from nano_vllm_metal.layers import linear as q
from nano_vllm_metal.layers import rotary_embedding as pe


def build_prompts(tok, n, rng, shared_prefix=None, min_len=32, max_len=384):
    """n chat prompts of variable length. If shared_prefix, all share a long prefix."""
    filler = (
        "The Shanghai maglev, the Yangtze river delta, container ports, finance, "
        "manufacturing, research, tourism, culture, transportation, and economics. "
    )
    prompts = []
    for i in range(n):
        target = rng.randint(min_len, max_len)
        if shared_prefix:
            body = shared_prefix
        else:
            body = ""
        while len(tok.encode(body, add_special_tokens=False)) < target:
            body += filler
        q_i = f" Question {i}: summarize the above in one sentence."
        prompts.append(
            tok.apply_chat_template(
                [{"role": "user", "content": body + q_i}],
                add_generation_prompt=True,
                tokenize=False,
            )
        )
    return prompts


def live_pages(model):
    # Physical pages currently in use, summed across the per-layer pools.
    return sum(len(p.used_page_ids) for p in model.page_pools)


def page_bytes(model, page_size):
    # One page slot: page_size tokens x num_kv_heads x head_dim, for K and V (x2), bf16 (x2 bytes).
    a = model.mlx_model.args
    return page_size * a.num_key_value_heads * a.head_dim * 2 * 2


def run_ours(model, tok, prompts, batch_size, gen):
    mx.synchronize()
    t0 = time.perf_counter()
    result = batch_generate(
        model, tok, list(prompts), max_seq_len=4096, batch_size=batch_size,
        prefill_step=256, max_new_tokens=gen, verbose=False,
    )
    mx.synchronize()
    makespan = time.perf_counter() - t0
    out_tokens = sum(len(tok.encode(text, add_special_tokens=False)) for _, text in result)
    return makespan, out_tokens


def measure_concurrent_kv(model, tok, prompts, prefix_cache=None):
    """Prefill all requests concurrently (none freed) and report peak LIVE pages.

    This is the real per-N-concurrent-requests KV footprint: paged storage is the
    actual token count (no per-slot max_seq_len padding), and with prefix caching
    shared pages are counted once.
    """
    from nano_vllm_metal.cache.paged import PrefixCache
    ps = model.page_size
    caches = []
    for p in prompts:
        ids = tok.encode(p, add_special_tokens=False)
        kv = model.create_kv_cache()
        reused = 0
        if prefix_cache is not None:
            matched = prefix_cache.match_prefix(ids)
            matched = matched[: max(0, (len(ids) - 1) // ps)]
            reused = prefix_cache.attach(matched, kv)
        off = reused
        while off < len(ids):
            chunk = ids[off : off + 256]
            mx.eval(model(mx.array([chunk], dtype=mx.int32), off, kv))
            off += len(chunk)
        if prefix_cache is not None:
            prefix_cache.register(ids, kv)
        caches.append(kv)
    return live_pages(model)


def run_mlx_sequential(mlx_model, tok, prompts, gen):
    """mlx_lm has no batching -> serve requests one at a time."""
    mx.synchronize()
    t0 = time.perf_counter()
    total = 0
    latencies = []
    for p in prompts:
        r0 = time.perf_counter()
        last = None
        for r in stream_generate(mlx_model, tok, p, max_tokens=gen):
            last = r
        total += last.generation_tokens
        latencies.append(time.perf_counter() - t0)  # completion time from batch start
    mx.synchronize()
    makespan = time.perf_counter() - t0
    return makespan, total, latencies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-0.6b")
    ap.add_argument("--num-requests", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--backend", choices=["mlx", "custom"], default="mlx")
    ap.add_argument("--shared-prefix", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    q.set_quantized_matmul_backend(args.backend)
    pe.set_rope_backend(args.backend)

    name = models.shortcut_name_to_full_name(args.model)
    mlx_model, tok = load(name)
    rng = Random(args.seed)
    prefix = None
    if args.shared_prefix:
        prefix = "System context: " + ("Shanghai is a global center for finance and trade. " * 30)
    prompts = build_prompts(tok, args.num_requests, rng, shared_prefix=prefix)
    plens = [len(tok.encode(p, add_special_tokens=False)) for p in prompts]

    print(f"model={name} backend={args.backend} requests={args.num_requests} "
          f"batch_size(slots)={args.batch_size} gen={args.gen} shared_prefix={args.shared_prefix}")
    print(f"prompt lens: min={min(plens)} max={max(plens)} mean={sum(plens)//len(plens)}")

    # --- throughput: both capped at --gen tokens (fair) ---
    ours_model = models.load_model(name, mlx_model, kind="paged", page_size=args.page_size)
    o_span, o_tok = run_ours(ours_model, tok, prompts, args.batch_size, args.gen)
    m_span, m_tok, _ = run_mlx_sequential(mlx_model, tok, prompts, args.gen)

    print("\n--- throughput (serving all requests, capped at --gen tokens) ---")
    print(f"ours (paged continuous batching, {args.batch_size} slots): {o_span:6.2f}s  {o_tok/o_span:7.1f} tok/s")
    print(f"mlx_lm (sequential, no batching):                {m_span:6.2f}s  {m_tok/m_span:7.1f} tok/s")
    print(f"aggregate throughput ratio (ours/mlx): {(o_tok/o_span)/(m_tok/m_span):.2f}x")

    # --- memory: peak LIVE pages with N requests concurrently resident ---
    from nano_vllm_metal.cache.paged import PrefixCache
    mem_model = models.load_model(name, mlx_model, kind="paged", page_size=args.page_size)
    live = measure_concurrent_kv(mem_model, tok, prompts)
    kv_mb = live * page_bytes(mem_model, args.page_size) / 1e6

    print("\n--- memory / concurrency (KV cache, all requests resident) ---")
    a = mlx_model.args
    print(f"paged (actual tokens, no padding): {live} live pages -> {kv_mb:.1f} MB")
    # mlx_lm has no batching: to run N requests concurrently it needs N independent
    # caches, each padded/allocated to its own length; there is no sharing.
    per_tok = a.num_key_value_heads * a.head_dim * 2 * 2 * a.num_hidden_layers
    mlx_concurrent_mb = sum(plens) * per_tok / 1e6
    print(f"mlx_lm concurrent (N independent caches, no sharing): {mlx_concurrent_mb:.1f} MB "
          f"(and mlx_lm cannot actually batch -> serializes instead)")
    if args.shared_prefix:
        mem_model2 = models.load_model(name, mlx_model, kind="paged", page_size=args.page_size)
        pc = PrefixCache(mem_model2.page_pools, mem_model2.num_hidden_layers)
        live_pc = measure_concurrent_kv(mem_model2, tok, prompts, prefix_cache=pc)
        kv_pc_mb = live_pc * page_bytes(mem_model2, args.page_size) / 1e6
        print(f"paged + PREFIX CACHING (shared prefix computed once): {live_pc} live pages -> "
              f"{kv_pc_mb:.1f} MB  ({kv_mb/max(kv_pc_mb,1e-6):.2f}x less than no-sharing)")


if __name__ == "__main__":
    main()
