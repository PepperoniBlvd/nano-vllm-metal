import os

import mlx.core as mx

# RoPE backend:
#   "custom" (default) — our from-scratch elementwise RoPE (gather cos/sin, split,
#       rotate, concat). The project's showcase implementation.
#   "mlx"              — MLX's fused `mx.fast.rope` (one Metal kernel). This is
#       what the production Apple-Silicon vLLM port (vllm-metal) uses; it's ~1
#       fused op vs our several, saving ~0.8 ms/token single-stream. It also takes
#       a VECTOR of per-request offsets, so batched decode with heterogeneous
#       offsets is a single call (cf. vllm-metal "packed RoPE", PR #496).
# Override via NVM_ROPE_BACKEND or set_rope_backend().
ROPE_BACKEND = os.environ.get("NVM_ROPE_BACKEND", "custom")
if ROPE_BACKEND not in ("custom", "mlx"):
    raise ValueError("NVM_ROPE_BACKEND must be 'custom' or 'mlx'")


def set_rope_backend(backend: str) -> None:
    """Select the RoPE backend at runtime ('custom' | 'mlx')."""
    global ROPE_BACKEND
    if backend not in ("custom", "mlx"):
        raise ValueError("backend must be 'custom' or 'mlx'")
    ROPE_BACKEND = backend


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        assert dims % 2 == 0, "dims must be even"

        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional
        self.half_dims = dims // 2

        positions = mx.arange(seq_len, dtype=mx.float32)
        dim_indices = mx.arange(self.half_dims, dtype=mx.float32)
        inv_freqs = mx.power(base, -(dim_indices / self.half_dims))
        freqs = mx.outer(positions, inv_freqs)

        self.cos_freqs = mx.cos(freqs)
        self.sin_freqs = mx.sin(freqs)

    def _mlx_rope(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        # x is [B, L, H, D]; mx.fast.rope applies rotation with the position axis
        # at -2, so transpose to [B, H, L, D], rope, then transpose back.
        if offset is None:
            off: int | mx.array = 0
        elif isinstance(offset, slice):
            off = int(offset.start)
        else:
            starts = [int(s.start) for s in offset]
            # Uniform offset -> scalar; per-request offsets -> vector (one call).
            off = starts[0] if len(set(starts)) == 1 else mx.array(starts, dtype=mx.int32)
        y = mx.fast.rope(
            x.transpose(0, 2, 1, 3),
            self.dims,
            traditional=self.traditional,
            base=self.base,
            scale=1.0,
            offset=off,
        )
        return y.transpose(0, 2, 1, 3).astype(x.dtype)

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        if ROPE_BACKEND == "mlx":
            return self._mlx_rope(x, offset)
        batch_size, seq_len, num_heads, dims = x.shape
        original_dtype = x.dtype

        if offset is None:
            cos = self.cos_freqs[:seq_len]
            sin = self.sin_freqs[:seq_len]
        elif isinstance(offset, slice):
            assert offset.start is not None and offset.stop is not None
            assert offset.stop - offset.start == seq_len
            cos = self.cos_freqs[offset]
            sin = self.sin_freqs[offset]
        else:
            assert len(offset) == batch_size
            positions = []
            for current_offset in offset:
                assert (
                    current_offset.start is not None and current_offset.stop is not None
                )
                assert current_offset.stop - current_offset.start == seq_len
                start = int(current_offset.start)
                stop = int(current_offset.stop)
                positions.append(mx.arange(start, stop))
            position_ids = mx.stack(positions)
            cos = self.cos_freqs[position_ids]
            sin = self.sin_freqs[position_ids]

        cos = cos.reshape(-1, seq_len, 1, self.half_dims)
        sin = sin.reshape(-1, seq_len, 1, self.half_dims)

        if self.traditional:
            pairs = x.reshape(batch_size, seq_len, num_heads, self.half_dims, 2)
            x1 = pairs[..., 0]
            x2 = pairs[..., 1]
            real = x1 * cos - x2 * sin
            imag = x1 * sin + x2 * cos
            y = mx.stack([real, imag], axis=-1).reshape(
                batch_size, seq_len, num_heads, dims
            )
        else:
            x1 = x[..., : self.half_dims]
            x2 = x[..., self.half_dims : self.dims]
            real = x1 * cos - x2 * sin
            imag = x1 * sin + x2 * cos
            y = mx.concatenate([real, imag], axis=-1)

        return y.astype(original_dtype)
