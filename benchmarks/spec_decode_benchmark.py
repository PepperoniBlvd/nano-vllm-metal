"""Benchmark speculative decoding on the nano_vllm_metal serving stack.

This measures the local nano_vllm_metal dense/3 model path, not mlx_lm's generator.
It reports both throughput and the counters that determine whether speculative
decoding can win: target verification calls, target verification input tokens,
draft calls, and accepted draft tokens.
"""

import argparse
import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate

from nano_vllm_metal import loader as models
from nano_vllm_metal.engine.generate import simple_generate_with_kv_cache, speculative_generate
from nano_vllm_metal.cache.kv_cache import TinyKvCache


@dataclass
class DecodeStats:
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    generated_tokens: int = 0
    decode_tokens: int = 0
    target_calls: int = 0
    target_input_tokens: int = 0
    draft_calls: int = 0
    draft_input_tokens: int = 0
    accepted_draft_tokens: int = 0
    target_sampled_tokens: int = 0
    rejected_draft_tokens: int = 0
    iterations: int = 0
    target_seconds: float = 0.0
    draft_seconds: float = 0.0
    rewind_seconds: float = 0.0

    @property
    def decode_tps(self) -> float:
        if self.decode_seconds == 0:
            return 0.0
        return self.decode_tokens / self.decode_seconds

    @property
    def end_to_end_tps(self) -> float:
        total = self.prefill_seconds + self.decode_seconds
        if total == 0:
            return 0.0
        return self.generated_tokens / total

    @property
    def accepted_fraction(self) -> float:
        if self.decode_tokens == 0:
            return 0.0
        return self.accepted_draft_tokens / self.decode_tokens

    @property
    def mean_target_width(self) -> float:
        if self.target_calls == 0:
            return 0.0
        return self.target_input_tokens / self.target_calls

    @property
    def target_ms_per_call(self) -> float:
        if self.target_calls == 0:
            return 0.0
        return 1000 * self.target_seconds / self.target_calls

    @property
    def draft_ms_per_call(self) -> float:
        if self.draft_calls == 0:
            return 0.0
        return 1000 * self.draft_seconds / self.draft_calls

    @property
    def other_seconds(self) -> float:
        measured = self.target_seconds + self.draft_seconds + self.rewind_seconds
        return max(0.0, self.decode_seconds - measured)


def release_cache(kv_cache: list[TinyKvCache]) -> None:
    for layer_cache in kv_cache:
        layer_cache.release()


def normalize_model_name(name: str) -> str:
    return models.shortcut_name_to_full_name(name)


def chat_prompt(tokenizer, prompt: str, enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def encode_prompt(tokenizer, prompt: str) -> mx.array:
    return mx.array(tokenizer.encode(prompt, add_special_tokens=False), dtype=mx.int32)


def sample_argmax(logits: mx.array, n_tokens: int = 1) -> mx.array:
    logits = logits[:, -n_tokens:, :]
    return mx.argmax(logits, axis=-1).astype(mx.int32)


def model_step(model, tokens: mx.array, offset: int, kv_cache, n_tokens: int = 1):
    logits = model(tokens[None], offset, kv_cache)
    return sample_argmax(logits, n_tokens=n_tokens)


def prefill_first_token(model, prompt_tokens: mx.array, kv_cache):
    token = model_step(model, prompt_tokens, 0, kv_cache, n_tokens=1).reshape(-1)
    mx.eval(token)
    return token, prompt_tokens.size


def rewind_cache(kv_cache, n_tokens: int) -> None:
    if n_tokens <= 0:
        return
    for layer_cache in kv_cache:
        layer_cache.rewind(n_tokens)


def greedy_decode(model, prompt_tokens: mx.array, max_tokens: int):
    kv_cache = model.create_kv_cache()
    stats = DecodeStats(generated_tokens=max_tokens)
    output: list[int] = []
    try:
        tic = time.perf_counter()
        current, offset = prefill_first_token(model, prompt_tokens, kv_cache)
        mx.synchronize()
        stats.prefill_seconds = time.perf_counter() - tic
        output.append(int(current.item()))
        stats.target_sampled_tokens = 1

        remaining = max_tokens - 1
        if remaining <= 0:
            return output, stats

        tic = time.perf_counter()
        for _ in range(remaining):
            step_tic = time.perf_counter()
            next_token = model_step(model, current, offset, kv_cache).reshape(-1)
            mx.eval(next_token)
            stats.target_seconds += time.perf_counter() - step_tic
            stats.target_calls += 1
            stats.target_input_tokens += int(current.size)
            stats.target_sampled_tokens += 1
            output.append(int(next_token.item()))
            offset += int(current.size)
            current = next_token
        mx.synchronize()
        stats.decode_seconds = time.perf_counter() - tic
        stats.decode_tokens = remaining
        return output, stats
    finally:
        release_cache(kv_cache)


def speculative_decode(
    target_model,
    draft_model,
    prompt_tokens: mx.array,
    max_tokens: int,
    num_draft_tokens: int,
):
    target_cache = target_model.create_kv_cache()
    draft_cache = draft_model.create_kv_cache()
    stats = DecodeStats(generated_tokens=max_tokens)
    output: list[int] = []
    try:
        tic = time.perf_counter()
        current, target_offset = prefill_first_token(
            target_model, prompt_tokens, target_cache
        )
        _, draft_offset = prefill_first_token(draft_model, prompt_tokens, draft_cache)
        mx.synchronize()
        stats.prefill_seconds = time.perf_counter() - tic
        output.append(int(current.item()))
        stats.target_sampled_tokens = 1

        remaining = max_tokens - 1
        if remaining <= 0:
            return output, stats

        tic = time.perf_counter()
        while len(output) < max_tokens:
            stats.iterations += 1
            draft_count = min(num_draft_tokens, max_tokens - len(output))
            draft_tokens: list[int] = []
            draft_input = current
            for _ in range(draft_count):
                step_tic = time.perf_counter()
                token = model_step(
                    draft_model, draft_input, draft_offset, draft_cache
                ).reshape(-1)
                mx.eval(token)
                stats.draft_seconds += time.perf_counter() - step_tic
                stats.draft_calls += 1
                stats.draft_input_tokens += int(draft_input.size)
                draft_tokens.append(int(token.item()))
                draft_input = token
                draft_offset += 1

            candidates = mx.concat(
                [current, mx.array(draft_tokens, dtype=mx.int32)], axis=0
            )
            step_tic = time.perf_counter()
            verified = model_step(
                target_model,
                candidates,
                target_offset,
                target_cache,
                n_tokens=draft_count + 1,
            )
            mx.eval(verified)
            stats.target_seconds += time.perf_counter() - step_tic
            verified_tokens = verified.tolist()[0]
            stats.target_calls += 1
            stats.target_input_tokens += int(candidates.size)
            target_offset += int(candidates.size)

            accepted_all = True
            for draft_index, draft_token in enumerate(draft_tokens):
                if len(output) >= max_tokens:
                    break
                target_token = int(verified_tokens[draft_index])
                if target_token != draft_token:
                    num_remaining_candidates = len(candidates) - (draft_index + 1)
                    rewind_tic = time.perf_counter()
                    rewind_cache(draft_cache, num_remaining_candidates - 1)
                    rewind_cache(target_cache, num_remaining_candidates)
                    stats.rewind_seconds += time.perf_counter() - rewind_tic
                    draft_offset -= num_remaining_candidates - 1
                    target_offset -= num_remaining_candidates
                    current = mx.array([target_token], dtype=mx.int32)
                    output.append(target_token)
                    stats.target_sampled_tokens += 1
                    stats.rejected_draft_tokens += 1
                    accepted_all = False
                    break

                output.append(draft_token)
                stats.accepted_draft_tokens += 1
                current = mx.array([draft_token], dtype=mx.int32)

            if accepted_all and len(output) < max_tokens:
                # The target produced one bonus token after the final accepted
                # draft token. The draft cache has not processed that final
                # draft token yet, so run one draft step to align its cache.
                last_draft = mx.array([draft_tokens[-1]], dtype=mx.int32)
                step_tic = time.perf_counter()
                _ = model_step(draft_model, last_draft, draft_offset, draft_cache)
                mx.eval(_)
                stats.draft_seconds += time.perf_counter() - step_tic
                stats.draft_calls += 1
                stats.draft_input_tokens += 1
                draft_offset += 1

                bonus = int(verified_tokens[draft_count])
                current = mx.array([bonus], dtype=mx.int32)
                output.append(bonus)
                stats.target_sampled_tokens += 1

        mx.synchronize()
        stats.decode_seconds = time.perf_counter() - tic
        stats.decode_tokens = max_tokens - 1
        return output[:max_tokens], stats
    finally:
        release_cache(draft_cache)
        release_cache(target_cache)


def build_bench_model(
    model_name: str, mlx_model, loader: str, enable_flash_attn: bool
):
    kind = "paged" if loader == "paged" else "dense"
    kwargs = {}
    if kind == "dense":
        kwargs["enable_flash_attn"] = enable_flash_attn
    return models.load_model(model_name, mlx_model, kind=kind, **kwargs)


def format_row(
    method: str,
    stats: DecodeStats,
    baseline_tps: float,
    matches_greedy: bool,
) -> str:
    speedup = stats.decode_tps / baseline_tps if baseline_tps else 0.0
    return (
        f"{method:<12}"
        f"{stats.decode_tps:>12.2f}"
        f"{speedup:>9.2f}x"
        f"{stats.end_to_end_tps:>12.2f}"
        f"{stats.accepted_fraction:>10.2%}"
        f"{stats.target_calls:>9}"
        f"{stats.mean_target_width:>10.2f}"
        f"{stats.draft_calls:>9}"
        f"{str(matches_greedy):>10}"
    )


def stats_record(
    *,
    backend: str,
    method: str,
    k: int | None,
    repeat: int,
    stats: DecodeStats,
    baseline_tps: float,
    matches_greedy: bool,
) -> dict:
    speedup = stats.decode_tps / baseline_tps if baseline_tps else 0.0
    return {
        "backend": backend,
        "method": method,
        "k": k,
        "repeat": repeat,
        "matches_greedy": matches_greedy,
        "decode_tps": stats.decode_tps,
        "speedup": speedup,
        "end_to_end_tps": stats.end_to_end_tps,
        "accepted_fraction": stats.accepted_fraction,
        "prefill_ms": 1000 * stats.prefill_seconds,
        "decode_ms": 1000 * stats.decode_seconds,
        "decode_tokens": stats.decode_tokens,
        "generated_tokens": stats.generated_tokens,
        "target_calls": stats.target_calls,
        "target_input_tokens": stats.target_input_tokens,
        "mean_target_width": stats.mean_target_width,
        "target_ms_per_call": stats.target_ms_per_call,
        "draft_calls": stats.draft_calls,
        "draft_input_tokens": stats.draft_input_tokens,
        "draft_ms_per_call": stats.draft_ms_per_call,
        "accepted_draft_tokens": stats.accepted_draft_tokens,
        "target_sampled_tokens": stats.target_sampled_tokens,
        "rejected_draft_tokens": stats.rejected_draft_tokens,
        "iterations": stats.iterations,
        "rewind_ms": 1000 * stats.rewind_seconds,
        "other_ms": 1000 * stats.other_seconds,
    }


def mean(values: list[float]) -> float:
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else 0.0


def stdev(values: list[float]) -> float:
    values = [value for value in values if value is not None]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_records(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int | None], list[dict]] = {}
    for record in records:
        grouped.setdefault(
            (record["backend"], record["method"], record["k"]), []
        ).append(record)

    summaries = []
    for (backend, method, k), rows in grouped.items():
        summaries.append(
            {
                "backend": backend,
                "method": method,
                "k": k,
                "runs": len(rows),
                "decode_tps_mean": mean([r["decode_tps"] for r in rows]),
                "decode_tps_std": stdev([r["decode_tps"] for r in rows]),
                "speedup_mean": mean([r["speedup"] for r in rows]),
                "speedup_std": stdev([r["speedup"] for r in rows]),
                "end_to_end_tps_mean": mean([r["end_to_end_tps"] for r in rows]),
                "accepted_fraction_mean": mean([r["accepted_fraction"] for r in rows]),
                "target_calls_mean": mean([r["target_calls"] for r in rows]),
                "mean_target_width_mean": mean([r["mean_target_width"] for r in rows]),
                "draft_calls_mean": mean([r["draft_calls"] for r in rows]),
                "target_ms_per_call_mean": mean(
                    [r["target_ms_per_call"] for r in rows]
                ),
                "draft_ms_per_call_mean": mean([r["draft_ms_per_call"] for r in rows]),
                "rewind_ms_mean": mean([r["rewind_ms"] for r in rows]),
                "other_ms_mean": mean([r["other_ms"] for r in rows]),
                "all_match_greedy": all(r["matches_greedy"] for r in rows),
            }
        )
    return summaries


def method_sort_key(summary: dict) -> tuple[int, int]:
    backend_order = 0 if summary["backend"] == "nano_vllm_metal" else 1
    if summary["method"] == "greedy":
        return (backend_order, 0)
    return (backend_order, summary["k"] or 0)


def print_summary_table(summaries: list[dict]) -> None:
    print(
        f"{'backend':<10}"
        f"{'method':<12}"
        f"{'runs':>6}"
        f"{'decode mean':>13}"
        f"{'decode std':>12}"
        f"{'speedup':>9}"
        f"{'e2e mean':>11}"
        f"{'accepted':>10}"
        f"{'target':>9}"
        f"{'avg width':>10}"
        f"{'draft':>9}"
        f"{'matches':>10}"
    )
    for summary in sorted(summaries, key=method_sort_key):
        print(
            f"{summary['backend']:<10}"
            f"{summary['method']:<12}"
            f"{summary['runs']:>6}"
            f"{summary['decode_tps_mean']:>13.2f}"
            f"{summary['decode_tps_std']:>12.2f}"
            f"{summary['speedup_mean']:>8.2f}x"
            f"{summary['end_to_end_tps_mean']:>11.2f}"
            f"{summary['accepted_fraction_mean']:>10.2%}"
            f"{summary['target_calls_mean']:>9.1f}"
            f"{summary['mean_target_width_mean']:>10.2f}"
            f"{summary['draft_calls_mean']:>9.1f}"
            f"{str(summary['all_match_greedy']):>10}"
        )


def print_timing_summary(summaries: list[dict]) -> None:
    print("\n--- timing breakdown mean ---")
    print(
        f"{'backend':<10}"
        f"{'method':<12}"
        f"{'target ms/c':>13}"
        f"{'draft ms/c':>12}"
        f"{'rewind ms':>11}"
        f"{'other ms':>11}"
    )
    for summary in sorted(summaries, key=method_sort_key):
        print(
            f"{summary['backend']:<10}"
            f"{summary['method']:<12}"
            f"{summary['target_ms_per_call_mean']:>13.2f}"
            f"{summary['draft_ms_per_call_mean']:>12.2f}"
            f"{summary['rewind_ms_mean']:>11.2f}"
            f"{summary['other_ms_mean']:>11.2f}"
        )


def format_timing_row(method: str, stats: DecodeStats) -> str:
    return (
        f"{method:<12}"
        f"{1000 * stats.decode_seconds:>12.1f}"
        f"{stats.target_ms_per_call:>13.2f}"
        f"{stats.draft_ms_per_call:>12.2f}"
        f"{1000 * stats.rewind_seconds:>11.2f}"
        f"{1000 * stats.other_seconds:>11.2f}"
    )


def benchmark_verify_widths(
    target_model,
    prompt_tokens: mx.array,
    widths: list[int],
    repeats: int,
    warmup: int,
) -> list[dict]:
    print("\n--- target verification width benchmark ---")
    print(f"{'width':>7}{'ms/call':>12}{'effective tok/s':>18}")
    records = []
    for width in widths:
        if width < 1:
            raise ValueError("--verify-widths values must be positive")
        kv_cache = target_model.create_kv_cache()
        try:
            current, offset = prefill_first_token(target_model, prompt_tokens, kv_cache)
            candidates = mx.broadcast_to(current, (width,)).astype(mx.int32)
            for _ in range(warmup):
                verified = model_step(
                    target_model, candidates, offset, kv_cache, n_tokens=width
                )
                mx.eval(verified)
                rewind_cache(kv_cache, width)
            mx.synchronize()

            tic = time.perf_counter()
            for _ in range(repeats):
                verified = model_step(
                    target_model, candidates, offset, kv_cache, n_tokens=width
                )
                mx.eval(verified)
                rewind_cache(kv_cache, width)
            mx.synchronize()
            seconds_per_call = (time.perf_counter() - tic) / repeats
            record = {
                "width": width,
                "repeats": repeats,
                "warmup": warmup,
                "ms_per_call": 1000 * seconds_per_call,
                "effective_tps": width / seconds_per_call,
            }
            records.append(record)
            print(
                f"{width:>7}"
                f"{record['ms_per_call']:>12.2f}"
                f"{record['effective_tps']:>18.2f}"
            )
        finally:
            release_cache(kv_cache)
    return records


def check_public_generate(
    target_model,
    draft_model,
    tokenizer,
    draft_tokenizer,
    prompt: str,
    max_tokens: int,
    ks: list[int],
) -> list[dict]:
    print("\n--- public generate parity ---")
    greedy_text = simple_generate_with_kv_cache(
        target_model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        print_tokens=False,
    )
    print(f"{'method':<12}{'matches greedy':>16}")
    print(f"{'greedy':<12}{str(True):>16}")
    records = [{"method": "greedy", "k": None, "matches_greedy": True}]
    for k in ks:
        spec_text = speculative_generate(
            draft_model,
            target_model,
            draft_tokenizer,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            num_drafts=k,
            print_tokens=False,
        )
        matches = spec_text == greedy_text
        records.append({"method": f"spec k={k}", "k": k, "matches_greedy": matches})
        print(f"{'spec k=' + str(k):<12}{str(matches):>16}")
    return records


def run_mlx_generate(
    target_mlx_model,
    draft_mlx_model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    k: int | None = None,
) -> tuple[dict, str]:
    kwargs = {"max_tokens": max_tokens}
    if k is not None:
        kwargs["draft_model"] = draft_mlx_model
        kwargs["num_draft_tokens"] = k

    tic = time.perf_counter()
    text_segments = []
    last = None
    accepted = 0
    response_count = 0
    for response in stream_generate(target_mlx_model, tokenizer, prompt, **kwargs):
        last = response
        text_segments.append(response.text)
        accepted += int(response.from_draft)
        response_count += 1
    wall_seconds = time.perf_counter() - tic
    if last is None:
        raise RuntimeError("mlx_lm did not produce a generation response")

    generation_tokens = int(last.generation_tokens)
    decode_tps = float(last.generation_tps)
    record = {
        "decode_tps": decode_tps,
        "end_to_end_tps": generation_tokens / wall_seconds if wall_seconds else 0.0,
        "accepted_fraction": accepted / generation_tokens if generation_tokens else 0.0,
        "decode_ms": 1000 * generation_tokens / decode_tps if decode_tps else 0.0,
        "generated_tokens": generation_tokens,
        "decode_tokens": generation_tokens,
        "target_calls": None,
        "target_input_tokens": None,
        "mean_target_width": None,
        "target_ms_per_call": None,
        "draft_calls": None,
        "draft_input_tokens": None,
        "draft_ms_per_call": None,
        "accepted_draft_tokens": accepted,
        "target_sampled_tokens": generation_tokens - accepted,
        "rejected_draft_tokens": None,
        "iterations": None,
        "rewind_ms": None,
        "other_ms": None,
        "response_count": response_count,
        "wall_ms": 1000 * wall_seconds,
    }
    return record, "".join(text_segments)


def mlx_record(
    *,
    method: str,
    k: int | None,
    repeat: int,
    metrics: dict,
    baseline_tps: float,
    matches_greedy: bool,
) -> dict:
    speedup = metrics["decode_tps"] / baseline_tps if baseline_tps else 0.0
    return {
        "backend": "mlx_lm",
        "method": method,
        "k": k,
        "repeat": repeat,
        "matches_greedy": matches_greedy,
        "speedup": speedup,
        **metrics,
    }


def benchmark_mlx_lm(
    target_mlx_model,
    draft_mlx_model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    ks: list[int],
    repeat: int,
    warmup_tokens: int,
) -> list[dict]:
    if warmup_tokens > 0:
        run_mlx_generate(target_mlx_model, None, tokenizer, prompt, warmup_tokens)
        run_mlx_generate(
            target_mlx_model,
            draft_mlx_model,
            tokenizer,
            prompt,
            warmup_tokens,
            min(ks),
        )

    records = []
    for repeat_idx in range(repeat):
        greedy_metrics, greedy_text = run_mlx_generate(
            target_mlx_model, None, tokenizer, prompt, max_tokens
        )
        baseline_tps = greedy_metrics["decode_tps"]
        records.append(
            mlx_record(
                method="greedy",
                k=None,
                repeat=repeat_idx,
                metrics=greedy_metrics,
                baseline_tps=baseline_tps,
                matches_greedy=True,
            )
        )
        for k in ks:
            metrics, text = run_mlx_generate(
                target_mlx_model,
                draft_mlx_model,
                tokenizer,
                prompt,
                max_tokens,
                k,
            )
            records.append(
                mlx_record(
                    method=f"spec k={k}",
                    k=k,
                    repeat=repeat_idx,
                    metrics=metrics,
                    baseline_tps=baseline_tps,
                    matches_greedy=text == greedy_text,
                )
            )
    return records


def write_json(path: str, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def write_csv(path: str, records: list[dict], config: dict) -> None:
    config_fields = [f"config_{key}" for key in config]
    record_fields = sorted({key for record in records for key in record})
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=config_fields + record_fields)
        writer.writeheader()
        for record in records:
            row = {f"config_{key}": value for key, value in config.items()}
            row.update(record)
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="qwen3-4b")
    parser.add_argument("--draft", default="qwen3-0.6b")
    parser.add_argument("--loader", choices=("dense", "paged"), default="dense")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--verify-widths", type=int, nargs="*", default=[])
    parser.add_argument("--verify-repeats", type=int, default=32)
    parser.add_argument("--verify-warmup", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--enable-flash-attn", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--include-mlx-lm", action="store_true")
    parser.add_argument("--skip-public-generate-check", action="store_true")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--csv-output", default=None)
    parser.add_argument(
        "--prompt", default="Explain how a CPU cache works, step by step."
    )
    args = parser.parse_args()

    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1")
    if any(k < 1 for k in args.ks):
        raise ValueError("--ks values must be positive")
    if args.repeat < 1:
        raise ValueError("--repeat must be positive")
    if args.enable_flash_attn and args.loader != "dense":
        raise ValueError("--enable-flash-attn is only supported for --loader dense")
    if args.verify_repeats < 1:
        raise ValueError("--verify-repeats must be positive")
    if args.verify_warmup < 0:
        raise ValueError("--verify-warmup must be non-negative")

    target_name = normalize_model_name(args.target)
    draft_name = normalize_model_name(args.draft)
    target_mlx_model, tokenizer = load(target_name)
    draft_mlx_model, draft_tokenizer = load(draft_name)
    if tokenizer.vocab_size != draft_tokenizer.vocab_size:
        raise ValueError("Target and draft tokenizers must use the same vocabulary")

    prompt = chat_prompt(tokenizer, args.prompt, args.enable_thinking)
    prompt_tokens = encode_prompt(tokenizer, prompt)
    stream = mx.gpu if args.device == "gpu" else mx.cpu

    with mx.stream(stream):
        target_model = build_bench_model(
            target_name, target_mlx_model, args.loader, args.enable_flash_attn
        )
        draft_model = build_bench_model(
            draft_name, draft_mlx_model, args.loader, args.enable_flash_attn
        )

        if args.warmup_tokens > 0:
            greedy_decode(target_model, prompt_tokens, args.warmup_tokens)
            speculative_decode(
                target_model,
                draft_model,
                prompt_tokens,
                args.warmup_tokens,
                min(args.ks),
            )

        print(f"target={target_name}")
        print(f"draft ={draft_name}")
        print(
            f"loader={args.loader} device={args.device} "
            f"flash_attn={args.enable_flash_attn} prompt_tokens={prompt_tokens.size} "
            f"max_tokens={args.max_tokens} repeat={args.repeat}"
        )
        print()
        records = []
        for repeat_idx in range(args.repeat):
            greedy_output, greedy_stats = greedy_decode(
                target_model, prompt_tokens, args.max_tokens
            )
            baseline_tps = greedy_stats.decode_tps
            records.append(
                stats_record(
                    backend="nano_vllm_metal",
                    method="greedy",
                    k=None,
                    repeat=repeat_idx,
                    stats=greedy_stats,
                    baseline_tps=baseline_tps,
                    matches_greedy=True,
                )
            )
            for k in args.ks:
                output, stats = speculative_decode(
                    target_model, draft_model, prompt_tokens, args.max_tokens, k
                )
                records.append(
                    stats_record(
                        backend="nano_vllm_metal",
                        method=f"spec k={k}",
                        k=k,
                        repeat=repeat_idx,
                        stats=stats,
                        baseline_tps=baseline_tps,
                        matches_greedy=output == greedy_output,
                    )
                )

        mlx_records = []
        if args.include_mlx_lm:
            mlx_records = benchmark_mlx_lm(
                target_mlx_model,
                draft_mlx_model,
                tokenizer,
                prompt,
                args.max_tokens,
                args.ks,
                args.repeat,
                args.warmup_tokens,
            )
            records.extend(mlx_records)

        summaries = summarize_records(records)
        print_summary_table(summaries)
        print_timing_summary(summaries)

        verify_records = []
        if args.verify_widths:
            verify_records = benchmark_verify_widths(
                target_model,
                prompt_tokens,
                args.verify_widths,
                args.verify_repeats,
                args.verify_warmup,
            )

        public_generate_records = []
        if not args.skip_public_generate_check:
            public_generate_records = check_public_generate(
                target_model,
                draft_model,
                tokenizer,
                draft_tokenizer,
                prompt,
                args.max_tokens,
                args.ks,
            )

        config = {
            "target": target_name,
            "draft": draft_name,
            "loader": args.loader,
            "device": args.device,
            "enable_flash_attn": args.enable_flash_attn,
            "enable_thinking": args.enable_thinking,
            "prompt_tokens": int(prompt_tokens.size),
            "max_tokens": args.max_tokens,
            "ks": args.ks,
            "repeat": args.repeat,
            "warmup_tokens": args.warmup_tokens,
            "include_mlx_lm": args.include_mlx_lm,
        }
        payload = {
            "config": config,
            "runs": records,
            "summaries": summaries,
            "verify_widths": verify_records,
            "public_generate": public_generate_records,
        }
        if args.json_output:
            write_json(args.json_output, payload)
        if args.csv_output:
            write_csv(args.csv_output, records, config)


if __name__ == "__main__":
    main()
