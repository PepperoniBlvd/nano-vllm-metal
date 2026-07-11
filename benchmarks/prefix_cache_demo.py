"""Prefix caching demo/benchmark on the paged paged model.

Runs a batch of prompts that share a long common prefix, one request at a time
through the paged (paged) path, and compares:
  - baseline:  every request prefills its whole prompt
  - prefix cache: the shared prefix is prefilled once and its KV pages are reused

Reports generated tokens (must be identical), prefill tokens actually computed,
and wall-clock prefill time.
"""

import argparse
import time

import mlx.core as mx
from mlx_lm import load

from nano_vllm_metal import loader as models
from nano_vllm_metal.cache.paged import PrefixCache


def argmax_last(logits: mx.array) -> int:
    return int(mx.argmax(logits[:, -1, :], axis=-1).item())


def prefill_and_first_token(model, token_ids, caches, prefill_step, prefix_cache=None):
    """Prefill token_ids into caches; return (first_token, prefill_tokens_computed).

    With prefix_cache, reuse cached leading pages and only prefill the suffix.
    """
    ps = model.page_size
    reused = 0
    if prefix_cache is not None:
        matched = prefix_cache.match_prefix(token_ids)
        # Always leave at least one token to actually run through the model.
        max_reuse_blocks = max(0, (len(token_ids) - 1) // ps)
        matched = matched[:max_reuse_blocks]
        reused = prefix_cache.attach(matched, caches)

    offset = reused
    logits = None
    computed = 0
    while offset < len(token_ids):
        chunk = token_ids[offset : offset + prefill_step]
        y = mx.array([chunk], dtype=mx.int32)
        logits = model(y, offset, caches)
        mx.eval(logits)
        computed += len(chunk)
        offset += len(chunk)

    if prefix_cache is not None:
        prefix_cache.register(token_ids, caches)

    return argmax_last(logits), computed


def generate(model, token_ids, prefill_step, max_new, prefix_cache=None):
    caches = model.create_kv_cache()
    tok, computed = prefill_and_first_token(
        model, token_ids, caches, prefill_step, prefix_cache
    )
    out = [tok]
    offset = len(token_ids)
    for _ in range(max_new - 1):
        y = mx.array([[tok]], dtype=mx.int32)
        logits = model(y, offset, caches)
        tok = argmax_last(logits)
        out.append(tok)
        offset += 1
    for c in caches:
        c.release()
    return out, computed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-0.6b")
    ap.add_argument("--page-size", type=int, default=128)
    ap.add_argument("--prefill-step", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--max-blocks", type=int, default=None,
                    help="Bound the prefix cache to this many blocks (LRU eviction).")
    args = ap.parse_args()

    model_name = models.shortcut_name_to_full_name(args.model)
    mlx_model, tok = load(model_name)

    shared = (
        "Shanghai is a direct-administered municipality and the most populous urban "
        "area in China. The city is located on the southern estuary of the Yangtze "
        "River, with the Huangpu River flowing through it. The population of the city "
        "proper is the second largest in the world after Chongqing. Shanghai is a "
        "global center for finance, business, research, science, manufacturing, "
        "transportation, tourism, and culture. The Port of Shanghai is the world's "
        "busiest container port. Based on the previous information, "
    )
    questions = [
        "where is Shanghai?",
        "what is the population of the city proper?",
        "what is Shanghai a center for?",
        "which river flows through the city?",
        "what is the busiest container port?",
        "what is the second largest city proper in the world?",
    ]
    prompts = [shared + q for q in questions]
    token_lists = [tok.encode(p, add_special_tokens=False) for p in prompts]
    shared_len = len(tok.encode(shared, add_special_tokens=False))
    print(
        f"model={model_name} page_size={args.page_size} prefill_step={args.prefill_step}"
    )
    print(
        f"{len(prompts)} prompts, shared prefix ~{shared_len} tokens, "
        f"prompt lens {[len(t) for t in token_lists]}"
    )

    with mx.stream(mx.gpu):
        # ---- baseline: no prefix cache ----
        base_model = models.load_model(model_name, mlx_model, kind="paged", page_size=args.page_size)
        mx.synchronize()
        t0 = time.perf_counter()
        base_out = []
        base_computed = 0
        for toks in token_lists:
            o, c = generate(base_model, toks, args.prefill_step, args.max_new)
            base_out.append(o)
            base_computed += c
        mx.synchronize()
        base_t = time.perf_counter() - t0

        # ---- with prefix cache ----
        pc_model = models.load_model(model_name, mlx_model, kind="paged", page_size=args.page_size)
        prefix_cache = PrefixCache(
            pc_model.page_pools, pc_model.num_hidden_layers, max_blocks=args.max_blocks
        )
        mx.synchronize()
        t0 = time.perf_counter()
        pc_out = []
        pc_computed = 0
        for toks in token_lists:
            o, c = generate(pc_model, toks, args.prefill_step, args.max_new, prefix_cache)
            pc_out.append(o)
            pc_computed += c
        mx.synchronize()
        pc_t = time.perf_counter() - t0

    identical = base_out == pc_out
    total_prompt_tokens = sum(len(t) for t in token_lists)
    print("\n--- correctness ---")
    print(f"generated tokens identical: {identical}")
    if not identical:
        for i, (a, b) in enumerate(zip(base_out, pc_out)):
            if a != b:
                print(f"  MISMATCH req {i}:\n    base={a}\n    pc  ={b}")
    print("\n--- prefill work (tokens actually run through the model) ---")
    print(f"baseline prefill tokens computed: {base_computed} (of {total_prompt_tokens})")
    print(f"prefix   prefill tokens computed: {pc_computed}")
    print(f"prefill tokens saved: {base_computed - pc_computed} "
          f"({100*(base_computed-pc_computed)/base_computed:.1f}%)")
    print("\n--- wall-clock (prefill + decode, whole batch) ---")
    print(f"baseline: {base_t*1000:.1f} ms")
    print(f"prefix  : {pc_t*1000:.1f} ms  ({base_t/pc_t:.2f}x faster)")


if __name__ == "__main__":
    main()
