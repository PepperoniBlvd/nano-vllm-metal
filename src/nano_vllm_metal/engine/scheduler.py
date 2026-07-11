import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from ..cache.kv_cache import *
from ..cache.paged import PrefixCache, TinyKvPagedCache
from ..models.qwen3_dense import Qwen3DenseModel
from dataclasses import dataclass
from datetime import datetime
import copy


def _step(model, y, offsets, kv_cache, sampler=None):
    logits = model(y, offsets, kv_cache)
    logits = logits[:, -1, :]
    if sampler is None:
        # Greedy: argmax(logits) == argmax(logprobs), skip the logsumexp.
        return mx.argmax(logits, axis=-1)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return sampler(logprobs)


@dataclass
class BatchGenerateStats:
    total_prefill_tokens: int = 0
    computed_prefill_tokens: int = 0
    generated_tokens: int = 0
    decoded_tokens: int = 0
    reused_prefix_tokens: int = 0
    prefix_cache_hits: int = 0
    prefix_cache_requests: int = 0
    registered_prefix_blocks: int = 0


class Request:
    def __init__(
        self,
        model: any,
        tokenizer: TokenizerWrapper,
        prompt: str,
        prefill_max_step: int = 128,
        prompt_idx: int = 0,
        prefix_cache: PrefixCache | None = None,
        stats: BatchGenerateStats | None = None,
        max_new_tokens: int | None = None,
        sampler=None,
    ):
        self.prompt = prompt
        self.kv_cache = model.create_kv_cache()
        self.model = model
        self.sampler = sampler
        # Reconstructing a streaming detokenizer rebuilds a ~150k-entry token map
        # from the full vocab (~50 ms/request on Qwen3). The map is identical for
        # every request, so shallow-copy the tokenizer's prebuilt detokenizer
        # (sharing the immutable map/byte-decoder) and just reset the per-request
        # streaming state — ~500x cheaper and byte-identical output.
        self.detokenizer = copy.copy(tokenizer.detokenizer)
        self.detokenizer.reset()
        self.prefill_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        self.prefill_tokens = mx.array(self.prefill_token_ids, dtype=mx.int32)
        self.prefill_max_step = prefill_max_step
        self.is_done = False
        self.is_prefill_done = False
        self.eos_token_id = tokenizer.eos_token_id
        self.next_token = None
        self.offset = 0
        self.prompt_idx = prompt_idx
        self.prefix_cache = prefix_cache
        self.stats = stats
        self.max_new_tokens = max_new_tokens
        self.generated_tokens = 0

        if self.stats is not None:
            self.stats.total_prefill_tokens += len(self.prefill_token_ids)

        if self.prefix_cache is not None:
            if not all(isinstance(cache, TinyKvPagedCache) for cache in self.kv_cache):
                raise ValueError("PrefixCache requires a paged KV cache")
            self._attach_prefix_cache()

    def _attach_prefix_cache(self):
        assert self.prefix_cache is not None
        if self.stats is not None:
            self.stats.prefix_cache_requests += 1
        matched = self.prefix_cache.match_prefix(self.prefill_token_ids)
        # Leave at least one token for the model to run, and stop reuse at a
        # prefill chunk boundary. That keeps cached and uncached prefill chunking
        # identical for the suffix, avoiding small numerical differences that can
        # flip a greedy argmax.
        max_reuse_tokens = (
            (len(self.prefill_token_ids) - 1) // self.prefill_max_step
        ) * self.prefill_max_step
        max_reuse_blocks = max(0, max_reuse_tokens // self.prefix_cache.page_size)
        reused = self.prefix_cache.attach(matched[:max_reuse_blocks], self.kv_cache)
        self.offset = reused
        if self.stats is not None and reused > 0:
            self.stats.prefix_cache_hits += 1
            self.stats.reused_prefix_tokens += reused

    def try_prefill(self):
        """
        Prefill this request up to max_step size, returns None if prefill is not done
        """
        if self.is_prefill_done:
            raise ValueError("prefill called after done")
        tokens_to_prefill = min(
            self.prefill_max_step, self.prefill_tokens.size - self.offset
        )
        token = _step(
            self.model,
            self.prefill_tokens[self.offset : self.offset + tokens_to_prefill][None],
            [self.offset],
            self.kv_cache,
            self.sampler,
        )
        self.offset += tokens_to_prefill
        if self.stats is not None:
            self.stats.computed_prefill_tokens += int(tokens_to_prefill)

        for layer_cache in self.kv_cache:
            if isinstance(layer_cache, TinyKvPagedCache):
                continue
            if layer_cache.key_values is not None:
                mx.eval(layer_cache.key_values[0])
                mx.eval(layer_cache.key_values[1])

        if self.offset == self.prefill_tokens.size:
            self.is_prefill_done = True
            mx.eval(token)
            if self.prefix_cache is not None:
                added = self.prefix_cache.register(
                    self.prefill_token_ids, self.kv_cache
                )
                if self.stats is not None:
                    self.stats.registered_prefix_blocks += added
            self.decode_done(token.item(), update_offset=False)

    def decode_done(self, token, update_offset=True):
        if self.is_done:
            raise ValueError("decode called after done")
        self.generated_tokens += 1
        if self.stats is not None:
            self.stats.generated_tokens += 1
        if token == self.eos_token_id:
            self.is_done = True
            return
        self.detokenizer.add_token(token)
        self.next_token = token
        if update_offset:
            self.offset += 1
        if (
            self.max_new_tokens is not None
            and self.generated_tokens >= self.max_new_tokens
        ):
            self.is_done = True

    def text(self):
        return self.detokenizer.text


def _print_progress(
    requests: list[Request | None],
    pending_prefill_request: Request | None,
    queue_size: int,
    progress_cnt: int,
    start_time: datetime,
):
    print(f"  --- {datetime.now() - start_time}")
    animation_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    animation_frame = animation_frames[progress_cnt % len(animation_frames)]
    for i, request in enumerate(requests):
        if request is None:
            print(f"  Decode #{i}: idle", flush=True)
        else:
            text_preview = request.text()[-80:].replace("\n", " ")
            print(
                f"{animation_frame} Decode [req {request.prompt_idx}, {request.offset}]: {text_preview}",
                flush=True,
            )
    if pending_prefill_request is not None:
        if pending_prefill_request.is_prefill_done:
            print(
                f"  Prefill [req {pending_prefill_request.prompt_idx}]: done, waiting for slot, {queue_size} requests in queue",
                flush=True,
            )
            return
        precentage = (
            pending_prefill_request.offset / pending_prefill_request.prefill_tokens.size
        ) * 100
        print(
            f"{animation_frame} Prefill [req {pending_prefill_request.prompt_idx}]: {precentage:.2f}% ({pending_prefill_request.prefill_tokens.size - pending_prefill_request.offset} remaining tokens)",
            flush=True,
        )
    else:
        print(f"  Prefill: idle, {queue_size} requests in queue", flush=True)


def batch_generate(
    model: any,
    tokenizer: TokenizerWrapper,
    prompts: list[str],
    max_seq_len=512,
    batch_size=5,
    prefill_step=128,
    prefix_cache: PrefixCache | None = None,
    max_new_tokens: int | None = None,
    sampler=None,
    return_stats: bool = False,
    verbose: bool = True,
):
    decode_requests: list[Request | None] = [None] * batch_size
    kv_cache = [
        BatchingKvCache(max_active_requests=batch_size, max_seq_len=max_seq_len)
        for _ in range(model.num_hidden_layers)
    ]
    result = []
    stats = BatchGenerateStats()
    next_request_idx = 0
    progress_cnt = 0
    start_time = datetime.now()

    while True:
        # --- Refill: prefill pending prompts into EVERY free slot before we
        # decode. This "fills the batch" so the decode step below always runs at
        # full width, instead of the V0 pattern of one-prefill-then-partial-decode
        # that leaves the batch ramping up (and draining) one row at a time. Each
        # admitted request is prefilled to completion here; the freed-slot refill
        # on later iterations keeps the decode batch topped up. ---
        for i in range(batch_size):
            if decode_requests[i] is not None or len(prompts) == 0:
                continue
            prompt = prompts.pop(0)
            req = Request(
                model,
                tokenizer,
                prompt,
                prefill_step,
                next_request_idx,
                prefix_cache=prefix_cache,
                stats=stats,
                max_new_tokens=max_new_tokens,
                sampler=sampler,
            )
            next_request_idx += 1
            while not req.is_prefill_done:
                req.try_prefill()
            if req.is_done:
                # EOS (or max_new_tokens==1) on the first token from prefill:
                # nothing to decode, just release and record.
                for layer_cache in req.kv_cache:
                    layer_cache.release()
                result.append((req.prompt_idx, req.text()))
                continue
            for prefill_cache, batch_cache in zip(req.kv_cache, kv_cache):
                batch_cache.add_request(prefill_cache, i)
            decode_requests[i] = req

        if all(req is None for req in decode_requests):
            break

        if verbose:
            _print_progress(
                decode_requests, None, len(prompts), progress_cnt, start_time
            )
            progress_cnt += 1

        # --- Decode: one full-batch step over all active slots. ---
        next_tokens = []
        offsets = []
        for req in decode_requests:
            if req is None:
                next_tokens.append(0)
                offsets.append(0)
            else:
                next_tokens.append(req.next_token)
                offsets.append(req.offset)
        next_tokens = mx.array(next_tokens)
        next_tokens = _step(
            model, next_tokens.reshape(-1, 1), offsets, kv_cache, sampler
        )
        for i in range(batch_size):
            req = decode_requests[i]
            if req is None:
                continue
            stats.decoded_tokens += 1
            req.decode_done(next_tokens[i].item())
            remove_reason = None
            if req.is_done:
                remove_reason = "EOS"
            elif req.offset >= max_seq_len:
                remove_reason = "max seq len"
            if remove_reason is not None:
                if verbose:
                    print(
                        f"Removing request {i} due to {remove_reason}",
                        flush=True,
                    )
                for layer_cache in kv_cache:
                    layer_cache.remove_request(i)
                result.append((req.prompt_idx, req.text()))
                decode_requests[i] = None
    if return_stats:
        return result, stats
    return result
