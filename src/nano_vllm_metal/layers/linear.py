import os
from typing import Any

import mlx.core as mx
from .. import metal

# Quantized matmul backend:
#   "custom" (default) — our from-scratch SIMD-group GEMV Metal kernel (the
#       project's showcase kernel).
#   "mlx"              — Apple's tuned `mx.quantized_matmul` (this is what the
#       production vLLM Apple-Silicon port, vllm-metal, uses for weight matmuls).
#   "auto"            — hybrid: our GEMV for small M (memory-bound decode) and
#       the MLX kernel for M >= MLX_QMATMUL_MIN_M (compute-bound prefill/batch,
#       where Apple's GEMM wins most).
# Override via NVM_QMATMUL_BACKEND / NVM_MLX_QMATMUL_MIN_M or
# set_quantized_matmul_backend().
QMATMUL_BACKEND = os.environ.get("NVM_QMATMUL_BACKEND", "custom")
# "auto" uses Apple's qmm once M (flattened rows) reaches this. 2 => any batched
# decode or multi-token prefill uses MLX (where it amortizes and wins big),
# while single-token single-stream (M=1) keeps the from-scratch GEMV.
MLX_QMATMUL_MIN_M = int(os.environ.get("NVM_MLX_QMATMUL_MIN_M", "2"))
if QMATMUL_BACKEND not in ("auto", "custom", "mlx"):
    raise ValueError("NVM_QMATMUL_BACKEND must be one of: auto, custom, mlx")


def set_quantized_matmul_backend(backend: str, mlx_min_m: int | None = None) -> None:
    """Select the quantized-matmul backend at runtime ('custom' | 'mlx' | 'auto')."""
    global QMATMUL_BACKEND, MLX_QMATMUL_MIN_M
    if backend not in ("auto", "custom", "mlx"):
        raise ValueError("backend must be one of: auto, custom, mlx")
    QMATMUL_BACKEND = backend
    if mlx_min_m is not None:
        MLX_QMATMUL_MIN_M = mlx_min_m


def dequantize_linear(mx_layer: Any) -> mx.array:
    w = mx.dequantize(
        mx_layer.weight,
        mx_layer.scales,
        mx_layer.biases,
        mx_layer.group_size,
        mx_layer.bits,
    )
    return w


class QuantizedWeights:
    def __init__(
        self,
        scales: mx.array,
        biases: mx.array,
        group_size: int,
        bits: int,
        weight: mx.array,
    ):
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.weight = weight

    @staticmethod
    def from_mlx_layer(mlx_layer: Any) -> "QuantizedWeights":
        return QuantizedWeights(
            scales=mlx_layer.scales,
            biases=mlx_layer.biases,
            group_size=mlx_layer.group_size,
            bits=mlx_layer.bits,
            weight=mlx_layer.weight,
        )


def quantized_matmul(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    *N, D = a.shape
    a = mx.contiguous(a.reshape(-1, D))
    scales = mx.contiguous(scales)
    biases = mx.contiguous(biases)
    b = mx.contiguous(b)
    return metal.quantized_matmul(
        scales, biases, group_size, bits, a, b, transpose_b
    ).reshape(*N, -1)


def quantized_linear(
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:
    # Flattened row count of the matmul: the activations are reshaped to
    # (-1, D), so M = prod of all dims except the last. For BATCHED decode the
    # input is [B, 1, hidden] -> M = B (NOT x.shape[-2], which is the sequence
    # length 1). "auto" routes M >= MLX_QMATMUL_MIN_M to Apple's qmm, which (per
    # vllm-metal + our measurements) amortizes the 4-bit weight reads across the
    # M rows; our GEMV re-reads per row and stops scaling. Single-token
    # single-stream (M=1) stays on the from-scratch GEMV showcase.
    rows = 1
    for d in x.shape[:-1]:
        rows *= d
    use_mlx = QMATMUL_BACKEND == "mlx" or (
        QMATMUL_BACKEND == "auto" and rows >= MLX_QMATMUL_MIN_M
    )
    if use_mlx:
        out = mx.quantized_matmul(
            x,
            w.weight,
            scales=w.scales,
            biases=w.biases,
            transpose=True,  # weights are stored [out, in]
            group_size=w.group_size,
            bits=w.bits,
        )
    else:
        out = quantized_matmul(
            w.scales, w.biases, w.group_size, w.bits, x, w.weight, True
        )
    if bias is not None:
        out = out + bias
    return out
