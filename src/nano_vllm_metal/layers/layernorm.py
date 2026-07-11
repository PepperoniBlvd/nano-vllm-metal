import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # Fused RMSNorm kernel: normalizes in fp32 and applies the weight in a
        # single launch instead of the ~5 elementwise ops below. Equivalent to
        #   normalized = x * rsqrt(mean(x^2) + eps); return normalized * weight
        return mx.fast.rms_norm(x, self.weight, self.eps)
