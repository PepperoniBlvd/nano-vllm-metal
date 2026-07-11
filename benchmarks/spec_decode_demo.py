"""Speculative decoding benchmark: large target + small draft.

target = Qwen3-4B, draft = Qwen3-0.6B (same Qwen3 tokenizer/vocab, required).

NOTE: nano_vllm_metal's own speculative_generate is a work-in-progress (roadmap 3.4 is
marked code-in-progress and it currently crashes on a shape bug), and the
nano_vllm_metal_ref extension doesn't build on this machine. So this measures the
speculative-decoding *technique* via mlx_lm's correct built-in implementation on
the same 4B target / 0.6B draft pair -- analogous to how we measured the MLX
throughput ceiling. It quantifies what speculative decoding buys on this
hardware; it does not use nano_vllm_metal's custom kernel.
"""

import argparse
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate


def run(model, tokenizer, prompt, max_tokens, draft_model=None, num_draft=4):
    last = None
    n = 0
    for r in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        draft_model=draft_model,
        num_draft_tokens=num_draft,
    ):
        last = r
        n += 1
    return last.generation_tps, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-4B-MLX-4bit")
    ap.add_argument("--draft", default="Qwen/Qwen3-0.6B-MLX-4bit")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--prompt", default="Explain how a CPU cache works, step by step.")
    args = ap.parse_args()

    target, tok = load(args.target)
    draft, _ = load(args.draft)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # warmup
    run(target, tok, prompt, 8)
    run(target, tok, prompt, 8, draft_model=draft, num_draft=4)

    base_tps, n = run(target, tok, prompt, args.max_tokens)
    print(f"target={args.target}  draft={args.draft}  max_tokens={args.max_tokens}")
    print(f"{'method':<32}{'decode tok/s':>14}{'speedup':>10}")
    print(f"{'greedy (4B only)':<32}{base_tps:>14.1f}{1.0:>10.2f}x")
    for k in (2, 4, 6, 8):
        tps, _ = run(target, tok, prompt, args.max_tokens, draft_model=draft, num_draft=k)
        print(f"{'speculative (4B+0.6B, k=' + str(k) + ')':<32}{tps:>14.1f}{tps / base_tps:>9.2f}x")


if __name__ == "__main__":
    main()
