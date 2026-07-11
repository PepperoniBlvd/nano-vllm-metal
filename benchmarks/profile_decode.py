"""Profile nano_vllm_metal decode bottlenecks.

The uninstrumented path measures normal single-stream throughput. The optional
layer-instrumented path wraps each transformer block's attention and MLP calls
with synchronization barriers, so it is slower than normal generation but gives
useful cost attribution.
"""

import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate

from nano_vllm_metal import loader as models
from nano_vllm_metal.cache.kv_cache import TinyKvCache

from spec_decode_benchmark import (
    chat_prompt,
    build_bench_model,
    encode_prompt,
    normalize_model_name,
)


@dataclass
class ProfileRecord:
    backend: str
    model: str
    loader: str
    method: str
    repeat: int
    prompt_tokens: int
    generated_tokens: int
    prefill_ms: float
    decode_ms: float
    wall_ms: float
    decode_tps: float
    end_to_end_tps: float
    extra: dict[str, Any]


class TimingContext:
    def __init__(self, stats: dict[str, float]):
        self.stats = stats
        self.phase = "idle"

    def record(self, name: str, start: float, *values: mx.array) -> None:
        if values:
            mx.eval(*values)
        mx.synchronize()
        key = f"{self.phase}.{name}"
        self.stats[key] = self.stats.get(key, 0.0) + (time.perf_counter() - start)


class TimedCallable:
    def __init__(self, name: str, fn, timing_context: TimingContext):
        self.name = name
        self.fn = fn
        self.timing_context = timing_context

    def __call__(self, *args, **kwargs):
        tic = time.perf_counter()
        out = self.fn(*args, **kwargs)
        self.timing_context.record(self.name, tic, out)
        return out


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def sample_argmax(logits: mx.array) -> mx.array:
    return mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32).reshape(-1)


def release_cache(kv_cache: list[TinyKvCache]) -> None:
    for layer_cache in kv_cache:
        layer_cache.release()


def instrument_layers(model) -> tuple[dict[str, float], TimingContext]:
    stats: dict[str, float] = {}
    timing_context = TimingContext(stats)
    for layer_idx, block in enumerate(model.layers_inner):
        if hasattr(block.self_attn, "profile_stats"):
            block.self_attn.profile_stats = stats
            block.self_attn.profile_phase_getter = lambda ctx=timing_context: ctx.phase
        block.self_attn = TimedCallable(
            f"layer_{layer_idx:02d}_attention_path",
            block.self_attn,
            timing_context,
        )
        block.mlp = TimedCallable(
            f"layer_{layer_idx:02d}_mlp_path", block.mlp, timing_context
        )
    return stats, timing_context


def run_decode(
    model,
    prompt_tokens: mx.array,
    max_tokens: int,
    timing_context: TimingContext | None = None,
) -> dict:
    kv_cache = model.create_kv_cache()
    output: list[int] = []
    decode_step_ms: list[float] = []
    try:
        if timing_context is not None:
            timing_context.phase = "prefill"
        tic = time.perf_counter()
        logits = model(prompt_tokens[None], 0, kv_cache)
        sample_tic = time.perf_counter()
        token = sample_argmax(logits)
        if timing_context is not None:
            timing_context.record("sampling", sample_tic, token)
        else:
            mx.eval(token)
            mx.synchronize()
        prefill_seconds = time.perf_counter() - tic
        output.append(int(token.item()))

        offset = int(prompt_tokens.size)
        current = token
        if timing_context is not None:
            timing_context.phase = "decode"
        decode_tic = time.perf_counter()
        for _ in range(max_tokens - 1):
            step_tic = time.perf_counter()
            logits = model(current[None], offset, kv_cache)
            sample_tic = time.perf_counter()
            token = sample_argmax(logits)
            if timing_context is not None:
                timing_context.record("sampling", sample_tic, token)
            else:
                mx.eval(token)
                mx.synchronize()
            decode_step_ms.append(1000 * (time.perf_counter() - step_tic))
            output.append(int(token.item()))
            current = token
            offset += 1
        decode_seconds = time.perf_counter() - decode_tic

        return {
            "output": output,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "decode_step_ms": decode_step_ms,
        }
    finally:
        release_cache(kv_cache)


def build_model(
    model_name: str,
    mlx_model,
    loader: str,
    page_size: int,
    kv_bits: int | None,
):
    if loader == "paged":
        return models.load_model(
            model_name, mlx_model, kind="paged", page_size=page_size, kv_bits=kv_bits
        )
    if loader == "dense_flash":
        return build_bench_model(model_name, mlx_model, "dense", True)
    return build_bench_model(model_name, mlx_model, loader, False)


def sum_stats(
    stats: dict[str, float],
    *,
    phase: str | None = None,
    suffix: str | None = None,
    name: str | None = None,
) -> float:
    total = 0.0
    for key, value in stats.items():
        if phase is not None and not key.startswith(f"{phase}."):
            continue
        metric_name = key.split(".", 1)[1] if "." in key else key
        if suffix is not None and not metric_name.endswith(suffix):
            continue
        if name is not None and metric_name != name:
            continue
        total += value
    return total


def run_profile(
    *,
    model_name: str,
    mlx_model,
    prompt_tokens: mx.array,
    loader: str,
    page_size: int,
    kv_bits: int | None,
    max_tokens: int,
    repeat: int,
    instrument: bool,
) -> list[ProfileRecord]:
    records: list[ProfileRecord] = []
    method = loader
    if loader == "paged":
        method = f"{loader} page={page_size}"
        if kv_bits is not None:
            method = f"{method} kv{kv_bits}"
    if instrument:
        method = f"{method} instrumented"

    warmup_model = build_model(model_name, mlx_model, loader, page_size, kv_bits)
    run_decode(warmup_model, prompt_tokens, min(max_tokens, 8))

    for repeat_idx in range(repeat):
        model = build_model(model_name, mlx_model, loader, page_size, kv_bits)
        if instrument:
            layer_stats, timing_context = instrument_layers(model)
        else:
            layer_stats = {}
            timing_context = None
        tic = time.perf_counter()
        result = run_decode(model, prompt_tokens, max_tokens, timing_context)
        wall_seconds = time.perf_counter() - tic
        decode_seconds = result["decode_seconds"]
        generated = len(result["output"])
        step_ms = result["decode_step_ms"]

        attention_ms = 1000 * sum(
            value
            for key, value in layer_stats.items()
            if key.split(".", 1)[-1].endswith("attention_path")
        )
        mlp_ms = 1000 * sum(
            value
            for key, value in layer_stats.items()
            if key.split(".", 1)[-1].endswith("mlp_path")
        )
        decode_attention_ms = 1000 * sum_stats(
            layer_stats, phase="decode", suffix="attention_path"
        )
        decode_mlp_ms = 1000 * sum_stats(layer_stats, phase="decode", suffix="mlp_path")
        decode_qkv_ms = 1000 * sum_stats(
            layer_stats, phase="decode", name="attention_qkv_projection"
        )
        decode_norm_rope_ms = 1000 * sum_stats(
            layer_stats, phase="decode", name="attention_norm_rope"
        )
        decode_cache_ms = 1000 * sum_stats(
            layer_stats, phase="decode", name="attention_cache_update"
        )
        decode_kernel_ms = 1000 * sum_stats(
            layer_stats, phase="decode", name="attention_kernel"
        )
        decode_output_ms = 1000 * sum_stats(
            layer_stats, phase="decode", name="attention_output_projection"
        )
        decode_sampling_ms = 1000 * sum_stats(
            layer_stats, phase="decode", name="sampling"
        )
        prefill_kernel_ms = 1000 * sum_stats(
            layer_stats, phase="prefill", name="attention_kernel"
        )
        measured_ms = attention_ms + mlp_ms
        records.append(
            ProfileRecord(
                backend="nano_vllm_metal",
                model=model_name,
                loader=loader,
                method=method,
                repeat=repeat_idx,
                prompt_tokens=int(prompt_tokens.size),
                generated_tokens=generated,
                prefill_ms=1000 * result["prefill_seconds"],
                decode_ms=1000 * decode_seconds,
                wall_ms=1000 * wall_seconds,
                decode_tps=(generated - 1) / decode_seconds if decode_seconds else 0.0,
                end_to_end_tps=generated / wall_seconds if wall_seconds else 0.0,
                extra={
                    "page_size": page_size if loader == "paged" else None,
                    "kv_bits": kv_bits if loader == "paged" else None,
                    "decode_step_ms_mean": mean(step_ms),
                    "decode_step_ms_std": stdev(step_ms),
                    "decode_step_ms_p50": statistics.median(step_ms)
                    if step_ms
                    else 0.0,
                    "decode_step_ms_max": max(step_ms) if step_ms else 0.0,
                    "attention_path_ms": attention_ms if instrument else None,
                    "mlp_path_ms": mlp_ms if instrument else None,
                    "decode_attention_path_ms": decode_attention_ms
                    if instrument
                    else None,
                    "decode_mlp_path_ms": decode_mlp_ms if instrument else None,
                    "decode_qkv_projection_ms": decode_qkv_ms if instrument else None,
                    "decode_norm_rope_ms": decode_norm_rope_ms if instrument else None,
                    "decode_cache_update_ms": decode_cache_ms if instrument else None,
                    "decode_attention_kernel_ms": decode_kernel_ms
                    if instrument
                    else None,
                    "decode_output_projection_ms": decode_output_ms
                    if instrument
                    else None,
                    "decode_sampling_ms": decode_sampling_ms if instrument else None,
                    "prefill_attention_kernel_ms": prefill_kernel_ms
                    if instrument
                    else None,
                    "instrumented_remainder_ms": (1000 * wall_seconds - measured_ms)
                    if instrument
                    else None,
                    "attention_path_pct": decode_attention_ms / (1000 * decode_seconds)
                    if instrument and decode_seconds
                    else None,
                    "mlp_path_pct": decode_mlp_ms / (1000 * decode_seconds)
                    if instrument and decode_seconds
                    else None,
                    "layer_stats_ms": {
                        key: 1000 * value for key, value in layer_stats.items()
                    }
                    if instrument
                    else None,
                },
            )
        )
    return records


def run_mlx_profile(
    *,
    model_name: str,
    mlx_model,
    tokenizer,
    prompt: str,
    prompt_tokens: mx.array,
    max_tokens: int,
    repeat: int,
) -> list[ProfileRecord]:
    records: list[ProfileRecord] = []

    for _ in stream_generate(
        mlx_model, tokenizer, prompt, max_tokens=min(max_tokens, 8)
    ):
        pass

    for repeat_idx in range(repeat):
        tic = time.perf_counter()
        last = None
        for response in stream_generate(
            mlx_model, tokenizer, prompt, max_tokens=max_tokens
        ):
            last = response
        wall_seconds = time.perf_counter() - tic
        if last is None:
            raise RuntimeError("mlx_lm did not produce any tokens")
        generated = int(last.generation_tokens)
        decode_tps = float(last.generation_tps)
        decode_ms = 1000 * generated / decode_tps if decode_tps else 0.0
        prompt_tps = float(getattr(last, "prompt_tps", 0.0) or 0.0)
        prefill_ms = 1000 * int(prompt_tokens.size) / prompt_tps if prompt_tps else 0.0
        records.append(
            ProfileRecord(
                backend="mlx_lm",
                model=model_name,
                loader="mlx_lm",
                method="mlx_lm greedy",
                repeat=repeat_idx,
                prompt_tokens=int(prompt_tokens.size),
                generated_tokens=generated,
                prefill_ms=prefill_ms,
                decode_ms=decode_ms,
                wall_ms=1000 * wall_seconds,
                decode_tps=decode_tps,
                end_to_end_tps=generated / wall_seconds if wall_seconds else 0.0,
                extra={
                    "prompt_tps": prompt_tps,
                    "response_prompt_tokens": int(
                        getattr(last, "prompt_tokens", prompt_tokens.size)
                    ),
                },
            )
        )
    return records


def summarize(records: list[ProfileRecord]) -> list[dict]:
    grouped: dict[tuple[str, str], list[ProfileRecord]] = {}
    for record in records:
        grouped.setdefault((record.backend, record.method), []).append(record)

    rows = []
    for (backend, method), group in grouped.items():
        rows.append(
            {
                "backend": backend,
                "method": method,
                "runs": len(group),
                "decode_tps_mean": mean([r.decode_tps for r in group]),
                "decode_tps_std": stdev([r.decode_tps for r in group]),
                "prefill_ms_mean": mean([r.prefill_ms for r in group]),
                "decode_ms_mean": mean([r.decode_ms for r in group]),
                "wall_ms_mean": mean([r.wall_ms for r in group]),
                "step_ms_mean": mean(
                    [
                        r.extra.get("decode_step_ms_mean", 0.0)
                        for r in group
                        if r.extra.get("decode_step_ms_mean") is not None
                    ]
                ),
                "attention_pct_mean": mean(
                    [
                        r.extra.get("attention_path_pct", 0.0)
                        for r in group
                        if r.extra.get("attention_path_pct") is not None
                    ]
                ),
                "mlp_pct_mean": mean(
                    [
                        r.extra.get("mlp_path_pct", 0.0)
                        for r in group
                        if r.extra.get("mlp_path_pct") is not None
                    ]
                ),
            }
        )
    return rows


def print_summary(records: list[ProfileRecord]) -> None:
    print("\n--- decode profile summary ---")
    print(
        f"{'backend':<10}{'method':<28}{'runs':>6}{'decode tok/s':>14}"
        f"{'prefill ms':>12}{'decode ms':>12}{'step ms':>10}"
        f"{'attn %':>9}{'mlp %':>9}"
    )
    for row in sorted(summarize(records), key=lambda r: (r["backend"], r["method"])):
        print(
            f"{row['backend']:<10}{row['method']:<28}{row['runs']:>6}"
            f"{row['decode_tps_mean']:>14.2f}{row['prefill_ms_mean']:>12.1f}"
            f"{row['decode_ms_mean']:>12.1f}{row['step_ms_mean']:>10.2f}"
            f"{row['attention_pct_mean']:>8.1%}{row['mlp_pct_mean']:>8.1%}"
        )


def write_json(path: str, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def write_csv(path: str, records: list[ProfileRecord]) -> None:
    rows = []
    for record in records:
        row = asdict(record)
        extra = row.pop("extra")
        for key, value in extra.items():
            if isinstance(value, dict):
                value = json.dumps(value, sort_keys=True)
            row[f"extra_{key}"] = value
        rows.append(row)

    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="qwen3-8b")
    parser.add_argument("--loaders", nargs="+", default=["dense", "paged"])
    parser.add_argument("--page-sizes", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--kv-bits", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--include-mlx-lm", action="store_true")
    parser.add_argument("--instrument-layers", action="store_true")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--prompt-repeat", type=int, default=1)
    parser.add_argument(
        "--prompt", default="Explain how a CPU cache works, step by step."
    )
    args = parser.parse_args()

    if args.max_tokens < 2:
        raise ValueError("--max-tokens must be at least 2")
    if args.repeat < 1:
        raise ValueError("--repeat must be positive")
    if any(loader not in ("dense", "dense_flash", "paged") for loader in args.loaders):
        raise ValueError("--loaders only supports dense, dense_flash, paged")
    if any(page_size < 1 for page_size in args.page_sizes):
        raise ValueError("--page-sizes values must be positive")
    if args.kv_bits is not None and args.kv_bits != 8:
        raise ValueError("--kv-bits currently supports only 8")
    if args.prompt_repeat < 1:
        raise ValueError("--prompt-repeat must be positive")

    model_name = normalize_model_name(args.target)
    mlx_model, tokenizer = load(model_name)
    prompt_text = " ".join([args.prompt] * args.prompt_repeat)
    prompt = chat_prompt(tokenizer, prompt_text, enable_thinking=False)
    prompt_tokens = encode_prompt(tokenizer, prompt)

    records: list[ProfileRecord] = []
    with mx.stream(mx.gpu):
        print(f"target={model_name}")
        print(
            f"prompt_tokens={prompt_tokens.size} max_tokens={args.max_tokens} "
            f"repeat={args.repeat}"
        )
        for loader in args.loaders:
            page_sizes = args.page_sizes if loader == "paged" else [0]
            for page_size in page_sizes:
                records.extend(
                    run_profile(
                        model_name=model_name,
                        mlx_model=mlx_model,
                        prompt_tokens=prompt_tokens,
                        loader=loader,
                        page_size=page_size,
                        kv_bits=args.kv_bits if loader == "paged" else None,
                        max_tokens=args.max_tokens,
                        repeat=args.repeat,
                        instrument=False,
                    )
                )
                if args.instrument_layers:
                    records.extend(
                        run_profile(
                            model_name=model_name,
                            mlx_model=mlx_model,
                            prompt_tokens=prompt_tokens,
                            loader=loader,
                            page_size=page_size,
                            kv_bits=args.kv_bits if loader == "paged" else None,
                            max_tokens=args.max_tokens,
                            repeat=args.repeat,
                            instrument=True,
                        )
                    )

        if args.include_mlx_lm:
            records.extend(
                run_mlx_profile(
                    model_name=model_name,
                    mlx_model=mlx_model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    max_tokens=args.max_tokens,
                    repeat=args.repeat,
                )
            )

    print_summary(records)

    payload = {
        "config": {
            "target": model_name,
            "loaders": args.loaders,
            "page_sizes": args.page_sizes,
            "kv_bits": args.kv_bits,
            "prompt_repeat": args.prompt_repeat,
            "max_tokens": args.max_tokens,
            "repeat": args.repeat,
            "include_mlx_lm": args.include_mlx_lm,
            "instrument_layers": args.instrument_layers,
        },
        "records": [asdict(record) for record in records],
        "summaries": summarize(records),
    }
    if args.json_output:
        write_json(args.json_output, payload)
    if args.csv_output:
        write_csv(args.csv_output, records)


if __name__ == "__main__":
    main()
