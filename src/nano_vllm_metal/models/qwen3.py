import time
from collections.abc import Callable
from typing import Any

import mlx.core as mx

from ..layers.activation import silu
from ..layers.attention import paged_attention
from ..layers.layernorm import RMSNorm
from ..layers.rotary_embedding import RoPE
from ..layers.embed_head import QuantizedEmbedding
from ..layers.linear import QuantizedWeights, quantized_linear
from ..cache.kv_cache import TinyKvCache
from ..layers.moe import Moe
from ..cache.paged import TinyKvPagedCache, TinyKvPagedPool


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert num_heads % num_kv_heads == 0, (
            f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
        )
        self.head_dim = head_dim
        self.scale = mx.rsqrt(self.head_dim)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.rope = RoPE(self.head_dim, max_seq_len, theta)
        self.q_norm = RMSNorm(self.head_dim, q_norm, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, k_norm, eps=rms_norm_eps)
        self.profile_stats: dict[str, float] | None = None
        self.profile_phase_getter: Callable[[], str] | None = None

    def _profile_phase(self) -> str | None:
        if self.profile_stats is None or self.profile_phase_getter is None:
            return None
        return self.profile_phase_getter()

    def _record_profile(
        self, phase: str | None, name: str, start: float, *values: mx.array
    ) -> None:
        if phase is None or self.profile_stats is None:
            return
        if values:
            mx.eval(*values)
        mx.synchronize()
        key = f"{phase}.{name}"
        self.profile_stats[key] = self.profile_stats.get(key, 0.0) + (
            time.perf_counter() - start
        )

    def __call__(
        self,
        x: mx.array,
        offsets: int | list[int] | mx.array,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        phase = self._profile_phase()
        B, L, _ = x.shape
        tic = time.perf_counter()
        projection_q = quantized_linear(x, self.wq).reshape(
            B, L, self.num_heads, self.head_dim
        )
        projection_k = quantized_linear(x, self.wk).reshape(
            B, L, self.num_kv_heads, self.head_dim
        )
        projection_v = quantized_linear(x, self.wv).reshape(
            B, L, self.num_kv_heads, self.head_dim
        )
        self._record_profile(
            phase,
            "attention_qkv_projection",
            tic,
            projection_q,
            projection_k,
            projection_v,
        )

        tic = time.perf_counter()
        projection_q = self.q_norm(projection_q)
        projection_k = self.k_norm(projection_k)
        if isinstance(offsets, int):
            offset_slice = [slice(int(offsets), int(offsets + L))]
        else:
            offset_slice = [slice(int(i), int(i + L)) for i in offsets]
        projection_q = self.rope(projection_q, offset=offset_slice)
        projection_k = self.rope(projection_k, offset=offset_slice)
        projection_q = projection_q.transpose(0, 2, 1, 3)
        projection_k = projection_k.transpose(0, 2, 1, 3)
        projection_v = projection_v.transpose(0, 2, 1, 3)
        self._record_profile(
            phase, "attention_norm_rope", tic, projection_q, projection_k, projection_v
        )

        is_initial_prefill = (
            isinstance(cache, TinyKvPagedCache)
            and cache.offset == 0
            and B == 1
            and L > 1
        )
        use_dense_decode = (
            isinstance(cache, TinyKvPagedCache)
            and cache.pool.kv_bits is None
            and B == 1
            and L == 1
        )
        tic = time.perf_counter()
        metadata = cache.update_and_fetch_paged(
            projection_k,
            projection_v,
            mask_length=L,
            mask=mask,
        )
        self._record_profile(phase, "attention_cache_update", tic)

        tic = time.perf_counter()
        if is_initial_prefill:
            x = mx.fast.scaled_dot_product_attention(
                projection_q.astype(mx.float32),
                projection_k.astype(mx.float32),
                projection_v.astype(mx.float32),
                scale=float(self.head_dim**-0.5),
                mask=mask,
            ).astype(x.dtype)
        elif use_dense_decode:
            dense_key, dense_value = cache.gather_dense()
            x = mx.fast.scaled_dot_product_attention(
                projection_q.astype(mx.float32),
                dense_key.astype(mx.float32),
                dense_value.astype(mx.float32),
                scale=float(self.head_dim**-0.5),
                mask=mask,
            ).astype(x.dtype)
        else:
            x = paged_attention(
                projection_q.astype(mx.float32),
                metadata.key_pages,
                metadata.value_pages,
                metadata.block_table,
                metadata.context_lens,
                metadata.page_size,
                key_scales=metadata.key_scales,
                value_scales=metadata.value_scales,
                scale=self.scale,
                mask=metadata.mask,
            ).astype(x.dtype)
        self._record_profile(phase, "attention_kernel", tic, x)

        tic = time.perf_counter()
        x = x.transpose(0, 2, 1, 3).reshape(B, L, self.num_heads * self.head_dim)
        out = quantized_linear(x, self.wo)
        self._record_profile(phase, "attention_output_projection", tic, out)
        return out


class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        return quantized_linear(
            silu(quantized_linear(x, self.w_gate)) * quantized_linear(x, self.w_up),
            self.w_down,
        )


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        rms_norm_eps: float,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        mlp: Qwen3MLP | Moe,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.mlp = mlp
        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            hidden_size, w_post_attention_layernorm, eps=rms_norm_eps
        )
        self.self_attn = Qwen3MultiHeadAttention(
            num_heads=num_attention_heads,
            hidden_size=hidden_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            q_norm=q_norm,
            k_norm=k_norm,
            max_seq_len=max_seq_len,
            theta=theta,
            rms_norm_eps=rms_norm_eps,
        )

    def __call__(
        self,
        x: mx.array,
        offset: int | list[int] | mx.array,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), offset, cache, mask)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


def is_qwen3_moe_sparse_layer(args: Any, layer_idx: int) -> bool:
    return (
        getattr(args, "num_experts", 0) > 0
        and layer_idx not in getattr(args, "mlp_only_layers", [])
        and (layer_idx + 1) % getattr(args, "decoder_sparse_step", 1) == 0
    )


class Qwen3PagedModel:
    def __init__(
        self,
        mlx_model: Any,
        page_size: int = 128,
        kv_bits: int | None = None,
    ):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.hidden_size = mlx_model.args.hidden_size
        self.vocab_size = mlx_model.args.vocab_size
        self.page_size = page_size
        self.kv_bits = kv_bits
        # One pool PER LAYER (model-level, shared across requests so prefix
        # caching can reuse pages). Crucially NOT one shared pool for all layers:
        # a single pool tensor would serialize every layer's KV scatter-write
        # into one 28-deep functional-update chain that MLX cannot donate in
        # place, making decode ~130x slower on the write path. Separate per-layer
        # tensors keep the writes disjoint and donation-eligible.
        self.page_pools = [
            TinyKvPagedPool(page_size=self.page_size, kv_bits=kv_bits)
            for _ in range(self.num_hidden_layers)
        ]
        self.page_pool = self.page_pools[0]  # legacy alias
        precision = mx.bfloat16
        self.precision = precision

        self.embedding = QuantizedEmbedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.hidden_size,
            weight=QuantizedWeights.from_mlx_layer(mlx_model.model.embed_tokens),
        )
        self.layers_inner = []

        for i in range(mlx_model.args.num_hidden_layers):
            wq = QuantizedWeights.from_mlx_layer(
                mlx_model.model.layers[i].self_attn.q_proj
            )
            wk = QuantizedWeights.from_mlx_layer(
                mlx_model.model.layers[i].self_attn.k_proj
            )
            wv = QuantizedWeights.from_mlx_layer(
                mlx_model.model.layers[i].self_attn.v_proj
            )
            wo = QuantizedWeights.from_mlx_layer(
                mlx_model.model.layers[i].self_attn.o_proj
            )
            if is_qwen3_moe_sparse_layer(mlx_model.args, i):
                mlp = Moe(
                    w_router=QuantizedWeights.from_mlx_layer(
                        mlx_model.model.layers[i].mlp.gate
                    ),
                    w_gate=QuantizedWeights.from_mlx_layer(
                        mlx_model.model.layers[i].mlp.switch_mlp.gate_proj
                    ),
                    w_up=QuantizedWeights.from_mlx_layer(
                        mlx_model.model.layers[i].mlp.switch_mlp.up_proj
                    ),
                    w_down=QuantizedWeights.from_mlx_layer(
                        mlx_model.model.layers[i].mlp.switch_mlp.down_proj
                    ),
                    num_experts_per_tok=mlx_model.args.num_experts_per_tok,
                    norm_topk_prob=mlx_model.args.norm_topk_prob,
                )
            else:
                mlp = Qwen3MLP(
                    mlx_model.args.hidden_size,
                    mlx_model.args.intermediate_size,
                    QuantizedWeights.from_mlx_layer(
                        mlx_model.model.layers[i].mlp.gate_proj
                    ),
                    QuantizedWeights.from_mlx_layer(
                        mlx_model.model.layers[i].mlp.up_proj
                    ),
                    QuantizedWeights.from_mlx_layer(
                        mlx_model.model.layers[i].mlp.down_proj
                    ),
                )

            layer = Qwen3TransformerBlock(
                num_attention_heads=mlx_model.args.num_attention_heads,
                num_kv_heads=mlx_model.args.num_key_value_heads,
                hidden_size=mlx_model.args.hidden_size,
                head_dim=mlx_model.args.head_dim,
                rms_norm_eps=mlx_model.args.rms_norm_eps,
                wq=wq,
                wk=wk,
                wv=wv,
                wo=wo,
                q_norm=mlx_model.model.layers[i].self_attn.q_norm.weight,
                k_norm=mlx_model.model.layers[i].self_attn.k_norm.weight,
                w_input_layernorm=mlx_model.model.layers[i].input_layernorm.weight,
                w_post_attention_layernorm=mlx_model.model.layers[
                    i
                ].post_attention_layernorm.weight,
                mlp=mlp,
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta,
            )
            self.layers_inner.append(layer)
        self.norm = RMSNorm(
            mlx_model.args.hidden_size,
            weight=mlx_model.model.norm.weight,
            eps=mlx_model.args.rms_norm_eps,
        )
        if not mlx_model.args.tie_word_embeddings:
            self.w_lm_head = QuantizedWeights.from_mlx_layer(mlx_model.lm_head)
        else:
            self.w_lm_head = None
        self.mlx_model = mlx_model

    def create_kv_cache(self) -> list[TinyKvCache]:
        # One request gets one cache handle per layer, each on that layer's own
        # (request-shared) pool so writes stay disjoint across layers.
        return [
            TinyKvPagedCache(pool=self.page_pools[i])
            for i in range(self.num_hidden_layers)
        ]

    def __call__(
        self,
        inputs: mx.array,
        offset: int | list[int] | mx.array,
        cache: list[TinyKvCache],
    ) -> mx.array:
        h = self.embedding(inputs)
        for layer in range(self.num_hidden_layers):
            h = self.layers_inner[layer](h, offset, cache[layer], mask="causal")
        h = self.norm(h)
        if self.w_lm_head is not None:
            return quantized_linear(h, self.w_lm_head)
        else:
            return self.embedding.as_linear(h)
