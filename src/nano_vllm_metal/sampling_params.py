from dataclasses import dataclass


@dataclass
class SamplingParams:
    """Decoding controls, mirroring vLLM's SamplingParams (subset).

    temperature == 0.0 selects greedy decoding (argmax); any positive value
    enables temperature/top-p/top-k sampling.
    """

    max_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 0.0
    top_k: int | None = None
