import mlx.core as mx

from nano_vllm_metal.engine.generate import simple_generate_with_kv_cache, speculative_generate


class FakeDetokenizer:
    def __init__(self):
        self.text = ""
        self.last_segment = ""

    def reset(self):
        self.text = ""
        self.last_segment = ""

    def add_token(self, token: int):
        self.last_segment = f"{token} "
        self.text += self.last_segment

    def finalize(self):
        return None


class FakeTokenizer:
    eos_token_id = 0
    vocab_size = 128

    def __init__(self, prompt_len: int = 3):
        self.prompt_len = prompt_len
        self.detokenizer = FakeDetokenizer()

    def encode(self, prompt: str, add_special_tokens: bool = False):
        return list(range(1, self.prompt_len + 1))


class FakeCache:
    def __init__(self):
        self.offset = 0

    def rewind(self, n: int):
        assert 0 <= n <= self.offset
        self.offset -= n

    def release(self):
        return None


class FakeAutoregressiveModel:
    def __init__(self, generated_tokens: list[int], prompt_len: int = 3):
        self.generated_tokens = generated_tokens
        self.prompt_len = prompt_len
        self.vocab_size = max([0, *generated_tokens]) + 2

    def create_kv_cache(self):
        return [FakeCache(), FakeCache()]

    def __call__(self, tokens: mx.array, offset: int, cache: list[FakeCache]):
        assert len(tokens.shape) == 2
        batch, seq_len = tokens.shape
        assert batch == 1
        for layer_cache in cache:
            assert layer_cache.offset == offset
            layer_cache.offset += seq_len

        rows = []
        for pos in range(seq_len):
            token_idx = offset + pos - self.prompt_len + 1
            if token_idx < 0:
                token = self.generated_tokens[0]
            elif token_idx < len(self.generated_tokens):
                token = self.generated_tokens[token_idx]
            else:
                token = 0
            row = [-1000.0] * self.vocab_size
            row[token] = 0.0
            rows.append(row)
        return mx.array([rows], dtype=mx.float32)


def test_speculative_generate_matches_greedy_when_all_drafts_are_accepted():
    tokenizer = FakeTokenizer()
    target_tokens = [10, 11, 12, 13, 14, 15, 16, 17]
    target = FakeAutoregressiveModel(target_tokens)
    draft = FakeAutoregressiveModel(target_tokens)

    greedy = simple_generate_with_kv_cache(
        target, tokenizer, "prompt", max_tokens=8, print_tokens=False
    )
    spec = speculative_generate(
        draft,
        target,
        tokenizer,
        tokenizer,
        "prompt",
        max_tokens=8,
        num_drafts=3,
        print_tokens=False,
    )

    assert spec == greedy == "10 11 12 13 14 15 16 17 "


def test_speculative_generate_matches_greedy_after_rejections_and_rewinds():
    tokenizer = FakeTokenizer()
    target_tokens = [10, 11, 12, 13, 14, 15, 16, 17]
    draft_tokens = [10, 11, 99, 13, 14, 98, 16, 17]
    target = FakeAutoregressiveModel(target_tokens)
    draft = FakeAutoregressiveModel(draft_tokens)

    greedy = simple_generate_with_kv_cache(
        target, tokenizer, "prompt", max_tokens=8, print_tokens=False
    )
    spec = speculative_generate(
        draft,
        target,
        tokenizer,
        tokenizer,
        "prompt",
        max_tokens=8,
        num_drafts=3,
        print_tokens=False,
    )

    assert spec == greedy == "10 11 12 13 14 15 16 17 "


def test_speculative_generate_supports_zero_max_tokens():
    tokenizer = FakeTokenizer()
    model = FakeAutoregressiveModel([10, 11])

    assert (
        speculative_generate(
            model,
            model,
            tokenizer,
            tokenizer,
            "prompt",
            max_tokens=0,
            print_tokens=False,
        )
        == ""
    )
