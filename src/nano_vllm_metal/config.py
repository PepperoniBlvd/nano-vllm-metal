from dataclasses import dataclass


@dataclass
class EngineConfig:
    """Engine-level configuration for :class:`nano_vllm_metal.LLM`.

    - ``kind``: ``"paged"`` (block-paged KV, continuous batching, prefix
      caching) or ``"dense"`` (contiguous KV, best single-stream latency).
    - ``max_num_seqs``: decode batch width — how many sequences run
      concurrently per step. On quantized matmul this is the main throughput
      knob (weight-read amortization kicks in around 16-32).
    - ``backend``: matmul/RoPE backend. ``None`` keeps module defaults;
      ``"custom"`` uses the from-scratch Metal kernels, ``"mlx"`` uses MLX's
      fused ops, ``"auto"`` routes batch/prefill to MLX and single-stream to
      the custom GEMV.
    """

    model: str
    kind: str = "paged"
    max_num_seqs: int = 8
    max_model_len: int = 4096
    page_size: int = 16
    kv_bits: int | None = None
    prefill_step: int = 256
    enable_prefix_caching: bool = False
    backend: str | None = None
