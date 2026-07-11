"""Overall performance benchmark for the nano_vllm_metal serving stack.

This is intentionally a workload matrix, not one blended number. Some
optimizations help single-stream decode, while others only matter for batched or
shared-prefix serving. The report keeps those cases separate.
"""

import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from nano_vllm_metal import loader as models
from nano_vllm_metal.engine.scheduler import batch_generate
from nano_vllm_metal.cache.kv_cache import BatchingKvCache
from nano_vllm_metal.cache.paged import PrefixCache

from spec_decode_benchmark import (
    benchmark_mlx_lm,
    chat_prompt,
    build_bench_model,
    encode_prompt,
    greedy_decode,
    normalize_model_name,
    run_mlx_generate,
    speculative_decode,
    stats_record,
    summarize_records,
)


@dataclass
class WorkloadRecord:
    workload: str
    backend: str
    model: str
    loader: str | None
    method: str
    repeat: int
    generated_tokens: int
    wall_ms: float
    tokens_per_second: float
    extra: dict


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def sample_argmax(logits: mx.array) -> mx.array:
    logits = logits[:, -1, :]
    return mx.argmax(logits, axis=-1).astype(mx.int32)


def release_cache(kv_cache) -> None:
    for layer_cache in kv_cache:
        layer_cache.release()


def prefill_first_token(model, prompt_tokens: mx.array, kv_cache):
    logits = model(prompt_tokens[None], 0, kv_cache)
    token = sample_argmax(logits).reshape(-1)
    mx.eval(token)
    return token


def _paged_model(
    target_name: str, target_mlx_model, page_size: int, kv_bits: int | None
):
    return models.load_model(
        target_name,
        target_mlx_model,
        kind="paged",
        page_size=page_size,
        kv_bits=kv_bits,
    )


def _load_bench_model(
    target_name: str,
    target_mlx_model,
    loader: str,
    page_size: int,
    kv_bits: int | None,
    enable_flash_attn: bool,
):
    if loader == "paged":
        return _paged_model(target_name, target_mlx_model, page_size, kv_bits)
    if loader == "dense_flash":
        return build_bench_model(target_name, target_mlx_model, "dense", True)
    return build_bench_model(
        target_name,
        target_mlx_model,
        loader,
        enable_flash_attn and loader == "dense",
    )


def select_auto_single_loader(prompt_tokens: int, threshold: int) -> tuple[str, str]:
    if prompt_tokens >= threshold:
        return "paged", f"prompt_tokens>={threshold}"
    return "dense", f"prompt_tokens<{threshold}"


def select_auto_batch_loader(max_prompt_tokens: int, threshold: int) -> tuple[str, str]:
    if max_prompt_tokens >= threshold:
        return "paged", f"max_prompt_tokens>={threshold}"
    return "dense", f"max_prompt_tokens<{threshold}"


def run_loader_matrix(
    target_name: str,
    target_mlx_model,
    prompt_tokens: mx.array,
    loaders: list[str],
    max_tokens: int,
    repeat: int,
    enable_flash_attn: bool,
    page_size: int,
    kv_bits: int | None,
    auto_long_context_threshold: int,
) -> list[WorkloadRecord]:
    records: list[WorkloadRecord] = []
    for loader in loaders:
        if loader == "auto":
            actual_loader, policy_reason = select_auto_single_loader(
                int(prompt_tokens.size), auto_long_context_threshold
            )
        else:
            actual_loader = loader
            policy_reason = "explicit"

        model = _load_bench_model(
            target_name,
            target_mlx_model,
            actual_loader,
            page_size,
            kv_bits if actual_loader == "paged" else None,
            enable_flash_attn,
        )
        greedy_decode(model, prompt_tokens, min(max_tokens, 8))
        for repeat_idx in range(repeat):
            tic = time.perf_counter()
            _, stats = greedy_decode(model, prompt_tokens, max_tokens)
            mx.synchronize()
            wall = time.perf_counter() - tic
            method = f"greedy {loader}"
            if loader == "auto":
                method = f"greedy auto->{actual_loader}"
            records.append(
                WorkloadRecord(
                    workload="single_stream",
                    backend="nano_vllm_metal",
                    model=target_name,
                    loader=loader,
                    method=method,
                    repeat=repeat_idx,
                    generated_tokens=stats.generated_tokens,
                    wall_ms=1000 * wall,
                    tokens_per_second=stats.generated_tokens / wall if wall else 0.0,
                    extra={
                        "comparison_scope": "dense_local_parity",
                        "selected_loader": actual_loader,
                        "policy_reason": policy_reason,
                        "auto_long_context_threshold": auto_long_context_threshold
                        if loader == "auto"
                        else None,
                        "page_size": page_size if actual_loader == "paged" else None,
                        "kv_bits": kv_bits if actual_loader == "paged" else None,
                        "decode_tps": stats.decode_tps,
                        "prefill_ms": 1000 * stats.prefill_seconds,
                        "decode_ms": 1000 * stats.decode_seconds,
                    },
                )
            )
    return records


def run_mlx_lm_single_matrix(
    target_name: str,
    target_mlx_model,
    tokenizer,
    prompt: str,
    prompt_tokens: mx.array,
    max_tokens: int,
    repeat: int,
    warmup_tokens: int,
) -> list[WorkloadRecord]:
    records: list[WorkloadRecord] = []
    if warmup_tokens > 0:
        run_mlx_generate(target_mlx_model, None, tokenizer, prompt, warmup_tokens)

    for repeat_idx in range(repeat):
        tic = time.perf_counter()
        metrics, _ = run_mlx_generate(
            target_mlx_model, None, tokenizer, prompt, max_tokens
        )
        mx.synchronize()
        wall = time.perf_counter() - tic
        generated_tokens = int(metrics["generated_tokens"])
        records.append(
            WorkloadRecord(
                workload="single_stream",
                backend="mlx_lm",
                model=target_name,
                loader="mlx_lm",
                method="greedy mlx_lm",
                repeat=repeat_idx,
                generated_tokens=generated_tokens,
                wall_ms=1000 * wall,
                tokens_per_second=generated_tokens / wall if wall else 0.0,
                extra={
                    "comparison_scope": "dense_local_parity",
                    "selected_loader": "mlx_lm",
                    "policy_reason": "external_baseline",
                    "prompt_tokens": int(prompt_tokens.size),
                    "decode_tps": metrics["decode_tps"],
                    "prefill_ms": metrics.get("prefill_ms"),
                    "decode_ms": metrics["decode_ms"],
                    "end_to_end_tps": metrics["end_to_end_tps"],
                },
            )
        )
    return records


def run_spec_matrix(
    target_name: str,
    draft_name: str,
    target_mlx_model,
    draft_mlx_model,
    tokenizer,
    prompt: str,
    prompt_tokens: mx.array,
    loader: str,
    ks: list[int],
    max_tokens: int,
    repeat: int,
    warmup_tokens: int,
    include_mlx_lm: bool,
    enable_flash_attn: bool,
) -> list[dict]:
    target_model = build_bench_model(
        target_name, target_mlx_model, loader, enable_flash_attn and loader == "dense"
    )
    draft_model = build_bench_model(
        draft_name, draft_mlx_model, loader, enable_flash_attn and loader == "dense"
    )
    if warmup_tokens > 0:
        greedy_decode(target_model, prompt_tokens, warmup_tokens)
        speculative_decode(
            target_model, draft_model, prompt_tokens, warmup_tokens, ks[0]
        )

    records = []
    for repeat_idx in range(repeat):
        greedy_output, greedy_stats = greedy_decode(
            target_model, prompt_tokens, max_tokens
        )
        baseline_tps = greedy_stats.decode_tps
        records.append(
            stats_record(
                backend="nano_vllm_metal",
                method=f"{loader} greedy",
                k=None,
                repeat=repeat_idx,
                stats=greedy_stats,
                baseline_tps=baseline_tps,
                matches_greedy=True,
            )
        )
        for k in ks:
            output, stats = speculative_decode(
                target_model, draft_model, prompt_tokens, max_tokens, k
            )
            records.append(
                stats_record(
                    backend="nano_vllm_metal",
                    method=f"{loader} spec k={k}",
                    k=k,
                    repeat=repeat_idx,
                    stats=stats,
                    baseline_tps=baseline_tps,
                    matches_greedy=output == greedy_output,
                )
            )

    if include_mlx_lm:
        records.extend(
            benchmark_mlx_lm(
                target_mlx_model,
                draft_mlx_model,
                tokenizer,
                prompt,
                max_tokens,
                ks,
                repeat,
                warmup_tokens,
            )
        )
    return records


def run_batched_decode(
    model,
    prompt_tokens_list: list[mx.array],
    max_tokens: int,
) -> tuple[int, float, float, float]:
    batch_size = len(prompt_tokens_list)
    batch_cache = [
        BatchingKvCache(max_active_requests=batch_size, max_seq_len=4096)
        for _ in range(model.num_hidden_layers)
    ]

    generated = batch_size
    prompt_lengths = [int(prompt_tokens.size) for prompt_tokens in prompt_tokens_list]
    same_prompt_length = len(set(prompt_lengths)) == 1

    prefill_tic = time.perf_counter()
    if same_prompt_length:
        request_caches = [model.create_kv_cache() for _ in prompt_tokens_list]
        for slot, request_cache in enumerate(request_caches):
            for prefill_cache, layer_batch_cache in zip(request_cache, batch_cache):
                layer_batch_cache.add_request(prefill_cache, slot)
        y = mx.stack(prompt_tokens_list, axis=0)
        logits = model(y, [0] * batch_size, batch_cache)
        token = sample_argmax(logits)
        mx.eval(token)
        current_tokens = [int(x) for x in token.tolist()]
        offsets = prompt_lengths[:]
    else:
        current_tokens = []
        offsets = []
        for slot, prompt_tokens in enumerate(prompt_tokens_list):
            request_cache = model.create_kv_cache()
            token = prefill_first_token(model, prompt_tokens, request_cache)
            for prefill_cache, layer_batch_cache in zip(request_cache, batch_cache):
                layer_batch_cache.add_request(prefill_cache, slot)
            current_tokens.append(int(token.item()))
            offsets.append(int(prompt_tokens.size))
    mx.synchronize()
    prefill_seconds = time.perf_counter() - prefill_tic

    decode_tic = time.perf_counter()
    for _ in range(max_tokens - 1):
        y = mx.array(current_tokens, dtype=mx.int32).reshape(batch_size, 1)
        logits = model(y, offsets, batch_cache)
        token = sample_argmax(logits)
        mx.eval(token)
        current_tokens = [int(x) for x in token.tolist()]
        offsets = [offset + 1 for offset in offsets]
        generated += batch_size
    mx.synchronize()
    decode_seconds = time.perf_counter() - decode_tic

    for layer_cache in batch_cache:
        for slot in range(batch_size):
            if layer_cache.kv_caches[slot] is not None:
                layer_cache.remove_request(slot)
    return generated, prefill_seconds, decode_seconds, prefill_seconds + decode_seconds


def run_batch_matrix(
    target_name: str,
    target_mlx_model,
    tokenizer,
    prompts: list[str],
    loader: str,
    page_size: int,
    max_tokens: int,
    repeat: int,
    enable_flash_attn: bool,
    kv_bits: int | None,
    auto_long_context_threshold: int,
) -> list[WorkloadRecord]:
    records: list[WorkloadRecord] = []
    prompt_tokens_list = [
        mx.array(tokenizer.encode(p, add_special_tokens=False), dtype=mx.int32)
        for p in prompts
    ]
    if loader == "auto":
        actual_loader, policy_reason = select_auto_batch_loader(
            max(int(tokens.size) for tokens in prompt_tokens_list),
            auto_long_context_threshold,
        )
    else:
        actual_loader = loader
        policy_reason = "explicit"
    model = _load_bench_model(
        target_name,
        target_mlx_model,
        actual_loader,
        page_size,
        kv_bits if actual_loader == "paged" else None,
        enable_flash_attn,
    )
    run_batched_decode(model, prompt_tokens_list, min(max_tokens, 4))
    for repeat_idx in range(repeat):
        generated, prefill_seconds, decode_seconds, wall = run_batched_decode(
            model, prompt_tokens_list, max_tokens
        )
        method = f"{loader} batch_size={len(prompts)}"
        if actual_loader == "paged":
            method = f"{loader} page={page_size} batch_size={len(prompts)}"
            if kv_bits is not None:
                method = (
                    f"{loader} page={page_size} kv{kv_bits} batch_size={len(prompts)}"
                )
        if loader == "auto":
            method = method.replace("auto", f"auto->{actual_loader}", 1)
        records.append(
            WorkloadRecord(
                workload="batched_decode",
                backend="nano_vllm_metal",
                model=target_name,
                loader=loader,
                method=method,
                repeat=repeat_idx,
                generated_tokens=generated,
                wall_ms=1000 * wall,
                tokens_per_second=generated / wall if wall else 0.0,
                extra={
                    "comparison_scope": "paged_serving_value"
                    if actual_loader == "paged"
                    else "dense_batch_baseline",
                    "selected_loader": actual_loader,
                    "policy_reason": policy_reason,
                    "auto_long_context_threshold": auto_long_context_threshold
                    if loader == "auto"
                    else None,
                    "batch_size": len(prompts),
                    "kv_bits": kv_bits if actual_loader == "paged" else None,
                    "page_size": page_size if actual_loader == "paged" else None,
                    "prefill_ms": 1000 * prefill_seconds,
                    "decode_ms": 1000 * decode_seconds,
                    "decode_tps": (generated - len(prompts)) / decode_seconds
                    if decode_seconds
                    else 0.0,
                },
            )
        )
    return records


def make_mixed_serving_prompts(
    tokenizer,
    batch_size: int,
    repeat_factors: list[int],
) -> list[str]:
    bases = [
        "Summarize the tradeoffs between dense KV cache and paged KV cache.",
        "Explain why batching requests with different sequence lengths is hard.",
        "Describe how prefix caching changes prefill cost in an LLM server.",
        "List practical bottlenecks in long-context autoregressive decoding.",
        "Compare local single-user inference with multi-request serving.",
        "Explain how request churn affects KV cache allocation.",
        "Describe why contiguous memory can be faster than page-indirected memory.",
        "Explain how shared prefixes can improve serving throughput.",
    ]
    prompts = []
    for i in range(batch_size):
        repeat = repeat_factors[i % len(repeat_factors)]
        text = " ".join([bases[i % len(bases)]] * repeat)
        prompts.append(chat_prompt(tokenizer, text, enable_thinking=False))
    return prompts


def run_serving_mixed_matrix(
    target_name: str,
    target_mlx_model,
    tokenizer,
    loader: str,
    page_size: int,
    max_new: int,
    repeat: int,
    batch_size: int,
    request_count: int,
    repeat_factors: list[int],
    prefill_step: int,
    enable_flash_attn: bool,
    kv_bits: int | None,
    auto_long_context_threshold: int,
    prompts: list[str] | None = None,
    prompt_token_counts: list[int] | None = None,
) -> list[WorkloadRecord]:
    records: list[WorkloadRecord] = []
    if prompts is None:
        prompts = make_mixed_serving_prompts(tokenizer, request_count, repeat_factors)
    if prompt_token_counts is None:
        prompt_token_counts = [
            len(tokenizer.encode(prompt, add_special_tokens=False))
            for prompt in prompts
        ]
    max_prompt_tokens = max(prompt_token_counts)
    if loader == "auto":
        actual_loader, policy_reason = select_auto_batch_loader(
            max_prompt_tokens,
            auto_long_context_threshold,
        )
    else:
        actual_loader = loader
        policy_reason = "explicit"

    model = _load_bench_model(
        target_name,
        target_mlx_model,
        actual_loader,
        page_size,
        kv_bits if actual_loader == "paged" else None,
        enable_flash_attn,
    )
    max_seq_len = max_prompt_tokens + max_new

    for repeat_idx in range(repeat):
        tic = time.perf_counter()
        _, stats = batch_generate(
            model,
            tokenizer,
            prompts.copy(),
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            prefill_step=prefill_step,
            max_new_tokens=max_new,
            return_stats=True,
            verbose=False,
        )
        mx.synchronize()
        wall = time.perf_counter() - tic
        method = f"{loader} requests={request_count} batch={batch_size}"
        if actual_loader == "paged":
            method = (
                f"{loader} page={page_size} requests={request_count} batch={batch_size}"
            )
            if kv_bits is not None:
                method = method.replace(
                    f"page={page_size}", f"page={page_size} kv{kv_bits}"
                )
        if loader == "auto":
            method = method.replace("auto", f"auto->{actual_loader}", 1)
        records.append(
            WorkloadRecord(
                workload="serving_mixed",
                backend="nano_vllm_metal",
                model=target_name,
                loader=loader,
                method=method,
                repeat=repeat_idx,
                generated_tokens=stats.generated_tokens,
                wall_ms=1000 * wall,
                tokens_per_second=stats.generated_tokens / wall if wall else 0.0,
                extra={
                    "comparison_scope": "paged_serving_value",
                    "selected_loader": actual_loader,
                    "policy_reason": policy_reason,
                    "auto_long_context_threshold": auto_long_context_threshold
                    if loader == "auto"
                    else None,
                    "batch_size": batch_size,
                    "request_count": request_count,
                    "prompt_tokens_min": min(prompt_token_counts),
                    "prompt_tokens_max": max_prompt_tokens,
                    "prompt_tokens_total": sum(prompt_token_counts),
                    "prompt_repeat_factors": repeat_factors,
                    "prefill_step": prefill_step,
                    "total_prefill_tokens": stats.total_prefill_tokens,
                    "computed_prefill_tokens": stats.computed_prefill_tokens,
                    "generated_tokens": stats.generated_tokens,
                    "decoded_tokens": stats.decoded_tokens,
                    "kv_bits": kv_bits if actual_loader == "paged" else None,
                    "page_size": page_size if actual_loader == "paged" else None,
                },
            )
        )
    return records


def run_mlx_lm_serving_mixed_matrix(
    target_name: str,
    target_mlx_model,
    tokenizer,
    prompts: list[str],
    prompt_token_counts: list[int],
    max_new: int,
    repeat: int,
    batch_size: int,
    repeat_factors: list[int],
    warmup_tokens: int,
) -> list[WorkloadRecord]:
    records: list[WorkloadRecord] = []
    if warmup_tokens > 0 and prompts:
        run_mlx_generate(target_mlx_model, None, tokenizer, prompts[0], warmup_tokens)

    for repeat_idx in range(repeat):
        generated_tokens = 0
        decode_ms = 0.0
        prompt_ms = 0.0
        tic = time.perf_counter()
        for prompt in prompts:
            metrics, _ = run_mlx_generate(
                target_mlx_model, None, tokenizer, prompt, max_new
            )
            generated_tokens += int(metrics["generated_tokens"])
            decode_ms += float(metrics.get("decode_ms") or 0.0)
            prefill_ms = metrics.get("prefill_ms")
            if prefill_ms is not None:
                prompt_ms += float(prefill_ms)
        mx.synchronize()
        wall = time.perf_counter() - tic
        records.append(
            WorkloadRecord(
                workload="serving_mixed",
                backend="mlx_lm",
                model=target_name,
                loader="mlx_lm",
                method=f"sequential mlx_lm requests={len(prompts)} batch=1",
                repeat=repeat_idx,
                generated_tokens=generated_tokens,
                wall_ms=1000 * wall,
                tokens_per_second=generated_tokens / wall if wall else 0.0,
                extra={
                    "comparison_scope": "external_dense_serving_baseline",
                    "selected_loader": "mlx_lm",
                    "policy_reason": "external_dense_sequential_baseline",
                    "batch_size": 1,
                    "request_count": len(prompts),
                    "prompt_tokens_min": min(prompt_token_counts),
                    "prompt_tokens_max": max(prompt_token_counts),
                    "prompt_tokens_total": sum(prompt_token_counts),
                    "prompt_repeat_factors": repeat_factors,
                    "generated_tokens": generated_tokens,
                    "decode_ms": decode_ms,
                    "prefill_ms": prompt_ms if prompt_ms else None,
                    "decode_tps": generated_tokens / (decode_ms / 1000.0)
                    if decode_ms
                    else 0.0,
                },
            )
        )
    return records


def run_prefix_matrix(
    model_name: str,
    mlx_model,
    tokenizer,
    loader_label: str,
    page_size: int,
    prefill_step: int,
    max_new: int,
    repeat: int,
    batch_size: int,
    kv_bits: int | None,
) -> list[WorkloadRecord]:
    shared_unit = (
        "Shanghai is a direct-administered municipality and the most populous urban "
        "area in China. The city is located on the southern estuary of the Yangtze "
        "River, with the Huangpu River flowing through it. The population of the city "
        "proper is the second largest in the world after Chongqing. Shanghai is a "
        "global center for finance, business, research, science, manufacturing, "
        "transportation, tourism, and culture. The Port of Shanghai is the world's "
        "busiest container port. Based on the previous information, "
    )
    shared = shared_unit * 4
    questions = [
        "where is Shanghai?",
        "what is the population of the city proper?",
        "what is Shanghai a center for?",
        "which river flows through the city?",
        "what is the busiest container port?",
        "what is the second largest city proper in the world?",
    ]
    token_lists = [
        tokenizer.encode(shared + question, add_special_tokens=False)
        for question in questions
    ]
    prompts = [shared + question for question in questions]
    total_prompt_tokens = sum(len(tokens) for tokens in token_lists)
    max_seq_len = max(len(tokens) for tokens in token_lists) + max(0, max_new - 1)
    records: list[WorkloadRecord] = []
    selected_loader = "paged"
    policy_reason = "prefix_cache"

    for repeat_idx in range(repeat):
        baseline_model = models.load_model(
            model_name, mlx_model, kind="paged", page_size=page_size, kv_bits=kv_bits
        )
        tic = time.perf_counter()
        baseline_out, baseline_stats = batch_generate(
            baseline_model,
            tokenizer,
            prompts.copy(),
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            prefill_step=prefill_step,
            max_new_tokens=max_new,
            return_stats=True,
            verbose=False,
        )
        mx.synchronize()
        baseline_wall = time.perf_counter() - tic

        prefix_model = models.load_model(
            model_name, mlx_model, kind="paged", page_size=page_size, kv_bits=kv_bits
        )
        prefix_cache = PrefixCache(
            prefix_model.page_pools, prefix_model.num_hidden_layers
        )
        tic = time.perf_counter()
        prefix_out, prefix_stats = batch_generate(
            prefix_model,
            tokenizer,
            prompts.copy(),
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            prefill_step=prefill_step,
            prefix_cache=prefix_cache,
            max_new_tokens=max_new,
            return_stats=True,
            verbose=False,
        )
        mx.synchronize()
        prefix_wall = time.perf_counter() - tic

        baseline_out = sorted(baseline_out)
        prefix_out = sorted(prefix_out)
        records.append(
            WorkloadRecord(
                workload="prefix_cache",
                backend="nano_vllm_metal",
                model=model_name,
                loader=loader_label,
                method=(
                    f"baseline {loader_label}->paged page={page_size}"
                    if loader_label == "auto"
                    else f"baseline page={page_size}"
                )
                if kv_bits is None
                else (
                    f"baseline {loader_label}->paged page={page_size} kv{kv_bits}"
                    if loader_label == "auto"
                    else f"baseline page={page_size} kv{kv_bits}"
                ),
                repeat=repeat_idx,
                generated_tokens=baseline_stats.generated_tokens,
                wall_ms=1000 * baseline_wall,
                tokens_per_second=baseline_stats.generated_tokens / baseline_wall
                if baseline_wall
                else 0.0,
                extra={
                    "comparison_scope": "paged_serving_value",
                    "selected_loader": selected_loader,
                    "policy_reason": policy_reason,
                    "page_size": page_size,
                    "kv_bits": kv_bits,
                    "batch_size": batch_size,
                    "prompt_tokens": total_prompt_tokens,
                    "computed_prefill_tokens": baseline_stats.computed_prefill_tokens,
                    "generated_tokens": baseline_stats.generated_tokens,
                    "decoded_tokens": baseline_stats.decoded_tokens,
                    "reused_prefix_tokens": baseline_stats.reused_prefix_tokens,
                    "prefix_cache_hits": baseline_stats.prefix_cache_hits,
                    "outputs_match": True,
                },
            )
        )
        records.append(
            WorkloadRecord(
                workload="prefix_cache",
                backend="nano_vllm_metal",
                model=model_name,
                loader=loader_label,
                method=(
                    f"prefix_cache {loader_label}->paged page={page_size}"
                    if loader_label == "auto"
                    else f"prefix_cache page={page_size}"
                )
                if kv_bits is None
                else (
                    f"prefix_cache {loader_label}->paged page={page_size} kv{kv_bits}"
                    if loader_label == "auto"
                    else f"prefix_cache page={page_size} kv{kv_bits}"
                ),
                repeat=repeat_idx,
                generated_tokens=prefix_stats.generated_tokens,
                wall_ms=1000 * prefix_wall,
                tokens_per_second=prefix_stats.generated_tokens / prefix_wall
                if prefix_wall
                else 0.0,
                extra={
                    "comparison_scope": "paged_serving_value",
                    "selected_loader": selected_loader,
                    "policy_reason": policy_reason,
                    "page_size": page_size,
                    "kv_bits": kv_bits,
                    "batch_size": batch_size,
                    "prompt_tokens": total_prompt_tokens,
                    "computed_prefill_tokens": prefix_stats.computed_prefill_tokens,
                    "generated_tokens": prefix_stats.generated_tokens,
                    "decoded_tokens": prefix_stats.decoded_tokens,
                    "reused_prefix_tokens": prefix_stats.reused_prefix_tokens,
                    "prefix_cache_hits": prefix_stats.prefix_cache_hits,
                    "prefix_cache_requests": prefix_stats.prefix_cache_requests,
                    "registered_prefix_blocks": prefix_stats.registered_prefix_blocks,
                    "prefill_tokens_saved": baseline_stats.computed_prefill_tokens
                    - prefix_stats.computed_prefill_tokens,
                    "speedup_vs_baseline": baseline_wall / prefix_wall
                    if prefix_wall
                    else 0.0,
                    "outputs_match": baseline_out == prefix_out,
                },
            )
        )
    return records


def summarize_workloads(records: list[WorkloadRecord]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[WorkloadRecord]] = {}
    for record in records:
        grouped.setdefault(
            (record.workload, record.backend, record.loader or "", record.method), []
        ).append(record)

    summaries = []
    for (workload, backend, loader, method), rows in grouped.items():
        summaries.append(
            {
                "workload": workload,
                "backend": backend,
                "loader": loader,
                "method": method,
                "runs": len(rows),
                "tokens_per_second_mean": mean([r.tokens_per_second for r in rows]),
                "tokens_per_second_std": stdev([r.tokens_per_second for r in rows]),
                "wall_ms_mean": mean([r.wall_ms for r in rows]),
            }
        )
    return summaries


def print_workload_summary(records: list[WorkloadRecord]) -> None:
    summaries = summarize_workloads(records)
    print("\n--- workload summary ---")
    print(
        f"{'workload':<16}{'backend':<18}{'loader':<12}{'method':<24}"
        f"{'runs':>6}{'tok/s':>12}{'wall ms':>12}"
    )
    for row in sorted(
        summaries, key=lambda r: (r["workload"], r["backend"], r["loader"], r["method"])
    ):
        print(
            f"{row['workload']:<16}{row['backend']:<18}{row['loader']:<12}"
            f"{row['method']:<24}{row['runs']:>6}"
            f"{row['tokens_per_second_mean']:>12.2f}{row['wall_ms_mean']:>12.1f}"
        )


def print_spec_summary(records: list[dict]) -> None:
    if not records:
        return
    summaries = summarize_records(records)
    print("\n--- speculative summary ---")
    print(
        f"{'backend':<18}{'method':<18}{'runs':>6}{'decode tok/s':>14}"
        f"{'speedup':>10}{'accepted':>10}{'matches':>10}"
    )
    for row in sorted(summaries, key=lambda r: (r["backend"], r["method"])):
        print(
            f"{row['backend']:<18}{row['method']:<18}{row['runs']:>6}"
            f"{row['decode_tps_mean']:>14.2f}{row['speedup_mean']:>9.2f}x"
            f"{row['accepted_fraction_mean']:>10.2%}"
            f"{str(row['all_match_greedy']):>10}"
        )


def write_json(path: str, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def write_csv(
    path: str, records: list[WorkloadRecord], spec_records: list[dict]
) -> None:
    rows = []
    for record in records:
        row = asdict(record)
        row.update({f"extra_{k}": v for k, v in record.extra.items()})
        del row["extra"]
        rows.append(row)
    for record in spec_records:
        row = {"workload": "speculative_decode", **record}
        rows.append(row)

    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default="qwen3-0.6b",
        help=(
            "Model for non-spec e2e stack benchmarks. Use a larger target, e.g. "
            "qwen3-8b, only when explicitly benchmarking large-model or "
            "speculative-decoding behavior."
        ),
    )
    parser.add_argument("--draft", default="qwen3-0.6b")
    parser.add_argument("--loaders", nargs="+", default=["dense", "paged"])
    parser.add_argument("--batch-loaders", nargs="+", default=None)
    parser.add_argument("--spec-loader", choices=("dense", "paged"), default="dense")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--include-mlx-lm", action="store_true")
    parser.add_argument("--enable-flash-attn", action="store_true")
    parser.add_argument("--skip-single", action="store_true")
    parser.add_argument("--skip-spec", action="store_true")
    parser.add_argument(
        "--include-spec",
        action="store_true",
        help=(
            "Include speculative decoding in this overall matrix. Disabled by "
            "default so non-spec e2e serving runs use the lightweight target."
        ),
    )
    parser.add_argument("--skip-batch", action="store_true")
    parser.add_argument("--skip-serving-mixed", action="store_true")
    parser.add_argument("--skip-prefix", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--serving-requests", type=int, default=8)
    parser.add_argument(
        "--serving-prompt-repeats",
        type=int,
        nargs="+",
        default=[1, 4, 12, 32],
    )
    parser.add_argument("--prefix-model", default=None)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--page-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--kv-bits", type=int, default=None)
    parser.add_argument(
        "--auto-long-context-threshold",
        type=int,
        default=4096,
        help=(
            "Prompt-token threshold where auto switches single/batch decode from "
            "dense dense KV to paged paged KV. Keep high by default because the "
            "paged path is currently capacity-oriented, not faster."
        ),
    )
    parser.add_argument("--prefill-step", type=int, default=128)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--prompt-repeat", type=int, default=1)
    parser.add_argument(
        "--prompt", default="Explain how a CPU cache works, step by step."
    )
    args = parser.parse_args()

    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")
    if args.repeat < 1:
        raise ValueError("--repeat must be positive")
    if any(k < 1 for k in args.ks):
        raise ValueError("--ks values must be positive")
    valid_loaders = ("auto", "dense", "dense_flash", "paged")
    if any(loader not in valid_loaders for loader in args.loaders):
        raise ValueError("--loaders only supports auto, dense, dense_flash, paged")
    batch_loaders = args.batch_loaders or args.loaders
    if any(loader not in valid_loaders for loader in batch_loaders):
        raise ValueError(
            "--batch-loaders only supports auto, dense, dense_flash, paged"
        )
    page_sizes = args.page_sizes or [args.page_size]
    if any(page_size < 1 for page_size in page_sizes):
        raise ValueError("--page-sizes values must be positive")
    if args.kv_bits is not None and args.kv_bits != 8:
        raise ValueError("--kv-bits currently supports only 8")
    if args.auto_long_context_threshold < 1:
        raise ValueError("--auto-long-context-threshold must be positive")
    if args.prompt_repeat < 1:
        raise ValueError("--prompt-repeat must be positive")
    if args.serving_requests < 1:
        raise ValueError("--serving-requests must be positive")
    if any(repeat < 1 for repeat in args.serving_prompt_repeats):
        raise ValueError("--serving-prompt-repeats values must be positive")

    target_name = normalize_model_name(args.target)
    draft_name = normalize_model_name(args.draft)
    target_mlx_model, tokenizer = load(target_name)
    draft_mlx_model, draft_tokenizer = load(draft_name)
    if tokenizer.vocab_size != draft_tokenizer.vocab_size:
        raise ValueError("Target and draft tokenizers must use the same vocabulary")

    prompt_text = " ".join([args.prompt] * args.prompt_repeat)
    prompt = chat_prompt(tokenizer, prompt_text, enable_thinking=False)
    prompt_tokens = encode_prompt(tokenizer, prompt)
    workload_records: list[WorkloadRecord] = []
    spec_records: list[dict] = []

    with mx.stream(mx.gpu):
        print(f"target={target_name}")
        print(f"draft ={draft_name}")
        print(
            f"prompt_tokens={prompt_tokens.size} max_tokens={args.max_tokens} "
            f"repeat={args.repeat}"
        )

        if not args.skip_single:
            for loader in args.loaders:
                if loader == "auto":
                    actual_loader, _ = select_auto_single_loader(
                        int(prompt_tokens.size), args.auto_long_context_threshold
                    )
                else:
                    actual_loader = loader
                single_page_sizes = (
                    page_sizes if actual_loader == "paged" else [args.page_size]
                )
                for page_size in single_page_sizes:
                    workload_records.extend(
                        run_loader_matrix(
                            target_name,
                            target_mlx_model,
                            prompt_tokens,
                            [loader],
                            args.max_tokens,
                            args.repeat,
                            args.enable_flash_attn,
                            page_size,
                            args.kv_bits,
                            args.auto_long_context_threshold,
                        )
                    )
            if args.include_mlx_lm:
                workload_records.extend(
                    run_mlx_lm_single_matrix(
                        target_name,
                        target_mlx_model,
                        tokenizer,
                        prompt,
                        prompt_tokens,
                        args.max_tokens,
                        args.repeat,
                        args.warmup_tokens,
                    )
                )

        if args.include_spec and not args.skip_spec:
            spec_records.extend(
                run_spec_matrix(
                    target_name,
                    draft_name,
                    target_mlx_model,
                    draft_mlx_model,
                    tokenizer,
                    prompt,
                    prompt_tokens,
                    args.spec_loader,
                    args.ks,
                    args.max_tokens,
                    args.repeat,
                    args.warmup_tokens,
                    args.include_mlx_lm,
                    args.enable_flash_attn,
                )
            )

        if not args.skip_batch:
            batch_prompts = [
                chat_prompt(
                    tokenizer,
                    f"Answer briefly: what is item {i} useful for?",
                    enable_thinking=False,
                )
                for i in range(args.batch_size)
            ]
            for batch_loader in batch_loaders:
                batch_page_sizes = (
                    page_sizes
                    if batch_loader in ("paged", "auto")
                    else [args.page_size]
                )
                for batch_page_size in batch_page_sizes:
                    workload_records.extend(
                        run_batch_matrix(
                            target_name,
                            target_mlx_model,
                            tokenizer,
                            batch_prompts,
                            batch_loader,
                            batch_page_size,
                            args.max_tokens,
                            args.repeat,
                            args.enable_flash_attn,
                            args.kv_bits if batch_loader in ("paged", "auto") else None,
                            args.auto_long_context_threshold,
                        )
                    )

        if not args.skip_serving_mixed:
            serving_prompts = make_mixed_serving_prompts(
                tokenizer, args.serving_requests, args.serving_prompt_repeats
            )
            serving_prompt_token_counts = [
                len(tokenizer.encode(prompt, add_special_tokens=False))
                for prompt in serving_prompts
            ]
            for batch_loader in batch_loaders:
                serving_page_sizes = (
                    page_sizes
                    if batch_loader in ("paged", "auto")
                    else [args.page_size]
                )
                for serving_page_size in serving_page_sizes:
                    workload_records.extend(
                        run_serving_mixed_matrix(
                            target_name,
                            target_mlx_model,
                            tokenizer,
                            batch_loader,
                            serving_page_size,
                            args.max_tokens,
                            args.repeat,
                            args.batch_size,
                            args.serving_requests,
                            args.serving_prompt_repeats,
                            args.prefill_step,
                            args.enable_flash_attn,
                            args.kv_bits if batch_loader in ("paged", "auto") else None,
                            args.auto_long_context_threshold,
                            prompts=serving_prompts,
                            prompt_token_counts=serving_prompt_token_counts,
                        )
                    )
            if args.include_mlx_lm:
                workload_records.extend(
                    run_mlx_lm_serving_mixed_matrix(
                        target_name,
                        target_mlx_model,
                        tokenizer,
                        serving_prompts,
                        serving_prompt_token_counts,
                        args.max_tokens,
                        args.repeat,
                        args.batch_size,
                        args.serving_prompt_repeats,
                        args.warmup_tokens,
                    )
                )

        if not args.skip_prefix:
            prefix_model_name = normalize_model_name(args.prefix_model or args.draft)
            prefix_mlx_model = (
                draft_mlx_model
                if prefix_model_name == draft_name
                else load(prefix_model_name)[0]
            )
            prefix_loader_label = "auto" if args.loaders == ["auto"] else "paged"
            for page_size in page_sizes:
                workload_records.extend(
                    run_prefix_matrix(
                        prefix_model_name,
                        prefix_mlx_model,
                        tokenizer
                        if prefix_model_name == target_name
                        else draft_tokenizer,
                        prefix_loader_label,
                        page_size,
                        args.prefill_step,
                        args.max_tokens,
                        args.repeat,
                        args.batch_size,
                        args.kv_bits,
                    )
                )

    print_workload_summary(workload_records)
    print_spec_summary(spec_records)

    payload = {
        "config": {
            "target": target_name,
            "draft": draft_name,
            "loaders": args.loaders,
            "batch_loaders": batch_loaders,
            "page_sizes": page_sizes,
            "kv_bits": args.kv_bits,
            "auto_long_context_threshold": args.auto_long_context_threshold,
            "prompt_repeat": args.prompt_repeat,
            "serving_requests": args.serving_requests,
            "serving_prompt_repeats": args.serving_prompt_repeats,
            "spec_loader": args.spec_loader,
            "ks": args.ks,
            "max_tokens": args.max_tokens,
            "repeat": args.repeat,
            "include_mlx_lm": args.include_mlx_lm,
            "include_spec": args.include_spec,
        },
        "workloads": [asdict(record) for record in workload_records],
        "workload_summaries": summarize_workloads(workload_records),
        "speculative": spec_records,
        "speculative_summaries": summarize_records(spec_records)
        if spec_records
        else [],
    }
    if args.json_output:
        write_json(args.json_output, payload)
    if args.csv_output:
        write_csv(args.csv_output, workload_records, spec_records)


if __name__ == "__main__":
    main()
