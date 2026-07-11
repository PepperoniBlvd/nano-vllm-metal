from dataclasses import dataclass

from mlx_lm import load

from .config import EngineConfig
from .sampling_params import SamplingParams
from .loader import load_model, shortcut_name_to_full_name
from .engine.scheduler import batch_generate
from .cache.paged import PrefixCache
from .layers.sampler import make_sampler
from .layers import linear as _linear
from .layers import rotary_embedding as _rope


@dataclass
class GenerationOutput:
    prompt: str
    text: str


class LLM:
    """High-level entry point, mirroring vLLM's ``LLM`` interface.

    >>> from nano_vllm_metal import LLM, SamplingParams
    >>> llm = LLM("qwen3-0.6b", max_num_seqs=16)
    >>> out = llm.generate(["Hello, world."], SamplingParams(max_tokens=64))
    >>> print(out[0].text)

    Prompts are used as-is; apply a chat template yourself if the model needs
    one (see ``tokenizer.apply_chat_template``).
    """

    def __init__(self, model: str, **kwargs):
        self.config = EngineConfig(model=model, **kwargs)
        cfg = self.config

        if cfg.backend is not None:
            _linear.set_quantized_matmul_backend(cfg.backend)
            _rope.set_rope_backend("mlx" if cfg.backend in ("mlx", "auto") else "custom")

        self.model_name = shortcut_name_to_full_name(model)
        self.mlx_model, self.tokenizer = load(self.model_name)

        if cfg.kind == "paged":
            self.model = load_model(
                self.model_name,
                self.mlx_model,
                kind="paged",
                page_size=cfg.page_size,
                kv_bits=cfg.kv_bits,
            )
        else:
            self.model = load_model(self.model_name, self.mlx_model, kind="dense")

        self.prefix_cache = None
        if cfg.enable_prefix_caching:
            if cfg.kind != "paged":
                raise ValueError("prefix caching requires kind='paged'")
            self.prefix_cache = PrefixCache(
                self.model.page_pools, self.model.num_hidden_layers
            )

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
    ) -> list[GenerationOutput]:
        if isinstance(prompts, str):
            prompts = [prompts]
        sp = sampling_params or SamplingParams()

        sampler = None
        if sp.temperature and sp.temperature > 0:
            sampler = make_sampler(sp.temperature, sp.top_p, sp.top_k)

        results = batch_generate(
            self.model,
            self.tokenizer,
            list(prompts),
            max_seq_len=self.config.max_model_len,
            batch_size=self.config.max_num_seqs,
            prefill_step=self.config.prefill_step,
            prefix_cache=self.prefix_cache,
            max_new_tokens=sp.max_tokens,
            sampler=sampler,
            verbose=False,
        )
        # batch_generate returns (prompt_index, text) in completion order.
        by_index = {idx: text for idx, text in results}
        return [
            GenerationOutput(prompt=prompt, text=by_index.get(i, ""))
            for i, prompt in enumerate(prompts)
        ]
