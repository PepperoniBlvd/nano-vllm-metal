# nano-vllm-metal

A minimal, readable **vLLM-style LLM inference engine for Apple Silicon**, built
on [MLX](https://github.com/ml-explore/mlx) with **from-scratch Metal kernels**.

It implements the pieces that make modern LLM serving fast — **paged attention,
continuous batching, and prefix caching** — on top of low-level MLX array ops and
hand-written Metal kernels (quantized matmul, paged/flash attention), rather than
high-level neural-network layers. The public API mirrors vLLM
(`LLM` + `SamplingParams`), so it reads like a tiny vLLM you can actually follow
end to end.

Target model: **Qwen3** (dense and MoE variants), using the official
`Qwen/Qwen3-*-MLX-4bit` weights.

## Quick start

Requires Apple Silicon macOS. Dependencies and tasks go through
[PDM](https://pdm-project.org/).

```bash
pdm install                 # install deps
pdm run build-ext           # build the C++/Metal kernels (nano_vllm_metal/metal)
pdm run check-installation  # sanity-check MLX (cpu + gpu)
```

```python
from nano_vllm_metal import LLM, SamplingParams

llm = LLM("qwen3-0.6b", max_num_seqs=16, enable_prefix_caching=True)

tok = llm.tokenizer
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Explain paged attention in one sentence."}],
    add_generation_prompt=True,
    tokenize=False,
)

outputs = llm.generate([prompt], SamplingParams(max_tokens=128, temperature=0.7))
print(outputs[0].text)
```

`LLM` accepts `kind="paged"` (block-paged KV — continuous batching + prefix
caching, the serving path) or `kind="dense"` (contiguous KV — best single-stream
latency), plus `backend` to pick the compute path: `"custom"` (from-scratch
Metal kernels), `"mlx"` (MLX's fused ops), or `"auto"` (custom GEMV for
single-stream, MLX for batched/prefill).

## Architecture

```
nano_vllm_metal/
  llm.py            LLM entry point (vLLM-mirroring)
  sampling_params.py, config.py
  loader.py         model-name resolution + load_model(kind='dense'|'paged')
  layers/           activation, layernorm, rotary_embedding, linear,
                    embed_head, attention, moe, sampler
  cache/            kv_cache (dense + batching), paged (pool / cache / prefix)
  models/           qwen3 (paged),  qwen3_dense
  engine/           scheduler (continuous batching),  generate
  metal/            C++/Metal kernels: quantized matmul, paged + flash attention
```

## Techniques

**Serving**

- **Paged attention** — block-paged KV cache with a from-scratch split-K
  ("flash-decoding") Metal kernel for decode.
- **Continuous batching** — a fill-the-batch scheduler: admitted requests are
  prefilled into every free slot, then decoded at full width with refill.
- **Prefix caching** — chained block hashing with refcounting and LRU eviction,
  so identical prompt prefixes reuse physical KV pages across requests.
- **Chunked prefill** — long prompts are prefilled in fixed-size steps.
- **Per-layer paged KV pools** — each layer owns its page pool so KV
  scatter-writes stay disjoint and MLX-donation-eligible (a single shared pool
  serializes every layer into one chain and is far slower on the write path).
- **Optional int8 KV cache** — quantized paged KV to cut memory.

**Kernels** (from-scratch Metal, with MLX built-ins as an opt-in backend via
`backend="custom" | "mlx" | "auto"`)

- **W4A16 quantized matmul** — a SIMD-group GEMV tuned for single-stream decode;
  MLX's `mx.quantized_matmul` handles the batched/prefill regime, where it
  amortizes weight reads better.
- **Flash attention** (prefill) and **paged attention** (decode) Metal kernels;
  MLX `mx.fast.*` equivalents are available under the `mlx` backend.

**Qwen3 model** (built on raw MLX array ops, no high-level NN layers)

- Grouped-query attention (GQA) with QK-norm, RoPE, RMSNorm, and a SwiGLU MLP.
- Mixture-of-Experts (MoE) for the sparse Qwen3 variant.

**Speculative decoding** — a draft + target decoding loop (`engine/generate.py`).
Its payoff depends on the target/draft size gap and on memory bandwidth. With a
large target and a tiny draft it speeds up generation — about **1.24×** with a
Qwen3-32B target + Qwen3-0.6B draft on an M4 Pro — because the large target's
decode is weight-bandwidth-bound and verifying several draft tokens in one
forward amortizes the weight read. With a small target or a bandwidth-limited
machine the verification pass is compute-bound and it does not beat plain
decoding.

## Performance

`mlx_lm` has no batching — it serves concurrent requests one at a time.
nano-vllm-metal serves them together with paged continuous batching, and
quantized-matmul weight reads amortize across the batch, so **`max_num_seqs`
(batch width) is the main throughput knob.**

Aggregate decode throughput, **Qwen3-4B** 4-bit on an M2 Pro (nano-vllm-metal =
paged continuous batching; `mlx_lm` = single-stream, so its rate is flat
regardless of load):

| concurrent requests | nano-vllm-metal | mlx_lm     | speedup |
| ------------------- | --------------- | ---------- | ------- |
| 8                   | ~60 tok/s       | ~60 tok/s  | 1.0×    |
| 16                  | ~87 tok/s       | ~60 tok/s  | 1.45×   |
| 32                  | ~163 tok/s      | ~60 tok/s  | 2.71×   |

**Prefix caching** adds a memory win: when requests share a prompt prefix, its
KV pages are stored once instead of per request.

| KV memory, 16 requests sharing a prefix (Qwen3-0.6B) | nano-vllm-metal | mlx_lm (no sharing) |
| ---------------------------------------------------- | --------------- | ------------------- |
| peak KV footprint                                    | ~97 MB          | ~600 MB (~6×)       |

The batching edge grows with model size — on the tiny Qwen3-0.6B, `mlx_lm`'s
single-stream path is already fast enough to be a hard baseline. Benchmarks live
in `benchmarks/` (`pdm run bench-serving`, `pdm run bench-overall`).

## Acknowledgements

Design inspiration from [vLLM](https://github.com/vllm-project/vllm),
[vllm-metal](https://github.com/vllm-project/vllm-metal), and
[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm). The C++/Metal
extension scaffolding follows the [MLX](https://github.com/ml-explore/mlx)
custom-extension example (see [`NOTICE`](NOTICE)).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
