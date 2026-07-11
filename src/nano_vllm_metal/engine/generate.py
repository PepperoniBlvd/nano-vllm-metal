import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from ..cache.kv_cache import TinyKvCache
from ..models.qwen3_dense import Qwen3DenseModel
from typing import Callable


def _release_kv_cache(kv_cache: list[TinyKvCache]):
    for layer in kv_cache:
        layer.release()


def simple_generate_with_kv_cache(
    model: Qwen3DenseModel,
    tokenizer: TokenizerWrapper,
    prompt: str,
    max_tokens: int | None = None,
    print_tokens: bool = True,
) -> str:
    kv_cache = model.create_kv_cache()

    def _step(model, y, offset, kv_cache):
        logits = model(y[None], offset, kv_cache)
        logits = logits[:, -1, :]
        token = mx.argmax(logits, axis=-1)
        return token, logits.squeeze(0)

    try:
        tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
        detokenizer = tokenizer.detokenizer
        detokenizer.reset()
        if max_tokens is not None and max_tokens <= 0:
            return detokenizer.text
        offset = 0
        generated_tokens = 0
        while True:
            token, _ = _step(model, tokens, offset, kv_cache)
            mx.eval(token)
            if token.item() == tokenizer.eos_token_id:
                break
            detokenizer.add_token(token.item())
            if print_tokens:
                print(detokenizer.last_segment, end="", flush=True)
            generated_tokens += 1
            if max_tokens is not None and generated_tokens >= max_tokens:
                detokenizer.finalize()
                break
            offset += tokens.size
            tokens = token
        return detokenizer.text
    finally:
        _release_kv_cache(kv_cache)


def speculative_generate(
    draft_model: Qwen3DenseModel,
    model: Qwen3DenseModel,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
    max_tokens: int | None = None,
    num_drafts: int = 4,
    print_tokens: bool = True,
) -> str:
    draft_kv_cache = draft_model.create_kv_cache()
    kv_cache = model.create_kv_cache()

    def _step(model, y, offset, kv_cache, n_tokens=1):
        logits = model(y[None], offset, kv_cache)
        if n_tokens > 1:
            logits = logits[:, -n_tokens:, :]
        else:
            logits = logits[:, -1, :]
        token = mx.argmax(logits, axis=-1)
        return token, logits.squeeze(0)

    def _prefill(model, tokenizer, prompt, cache):
        tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
        token, _ = _step(model, tokens, 0, cache)
        mx.eval(token)
        if token.item() == tokenizer.eos_token_id:
            return None, tokens.size
        return token, tokens.size

    def _rewind_cache(cache, n_tokens):
        if n_tokens <= 0:
            return
        for layer in cache:
            layer.rewind(n_tokens)

    def _decode_one(token):
        if token.item() == tokenizer.eos_token_id:
            return False
        tokenizer.detokenizer.add_token(token.item())
        if print_tokens:
            print(tokenizer.detokenizer.last_segment, end="", flush=True)
        return True

    def _draft_generate(last_token, offset, num_drafts):
        tokens = []
        current = last_token
        current_offset = offset
        for _ in range(num_drafts):
            token, _ = _step(draft_model, current, current_offset, draft_kv_cache)
            mx.eval(token)
            tokens.append(token.item())
            current = token
            current_offset += 1
        return tokens

    try:
        if max_tokens is not None and max_tokens <= 0:
            tokenizer.detokenizer.reset()
            return tokenizer.detokenizer.text

        tokenizer.detokenizer.reset()
        draft_token, draft_offset = _prefill(
            draft_model, draft_tokenizer, prompt, draft_kv_cache
        )
        token, offset = _prefill(model, tokenizer, prompt, kv_cache)
        if token is None:
            return tokenizer.detokenizer.text
        if draft_token is None:
            return tokenizer.detokenizer.text

        generated_tokens = 0
        while True:
            draft_tokens = _draft_generate(token, draft_offset, num_drafts)
            draft_offset += num_drafts

            candidate_tokens = mx.concat(
                [token, mx.array(draft_tokens, dtype=token.dtype)]
            )
            verified_tokens, _ = _step(
                model, candidate_tokens, offset, kv_cache, num_drafts + 1
            )
            verified_tokens = verified_tokens.tolist()[0]
            offset += num_drafts + 1
            next_token_after_all_accepted = verified_tokens[-1]
            comparable_tokens = mx.array([token.item()] + verified_tokens[:-1])

            accepted_all = True
            for i in range(len(comparable_tokens)):
                if comparable_tokens[i] != candidate_tokens[i]:
                    assert i >= 1
                    rejected_suffix_len = len(candidate_tokens) - i
                    _rewind_cache(draft_kv_cache, rejected_suffix_len - 1)
                    draft_offset -= rejected_suffix_len - 1
                    _rewind_cache(kv_cache, rejected_suffix_len)
                    token = mx.array([comparable_tokens[i]])
                    offset -= rejected_suffix_len
                    accepted_all = False
                    if token.item() == tokenizer.eos_token_id:
                        return tokenizer.detokenizer.text
                    break
                if not _decode_one(comparable_tokens[i]):
                    return tokenizer.detokenizer.text
                generated_tokens += 1
                if max_tokens is not None and generated_tokens >= max_tokens:
                    tokenizer.detokenizer.finalize()
                    return tokenizer.detokenizer.text

            if accepted_all:
                _draft_generate(mx.array(candidate_tokens[-1:]), draft_offset, 1)
                token = mx.array([next_token_after_all_accepted])
                draft_offset += 1
    finally:
        _release_kv_cache(draft_kv_cache)
        _release_kv_cache(kv_cache)
