from .models.qwen3 import Qwen3PagedModel
from .models.qwen3_dense import Qwen3DenseModel


def shortcut_name_to_full_name(shortcut_name: str):
    lower_shortcut_name = shortcut_name.lower()
    if lower_shortcut_name == "qwen3-8b":
        return "Qwen/Qwen3-8B-MLX-4bit"
    elif lower_shortcut_name == "qwen3-0.6b":
        return "Qwen/Qwen3-0.6B-MLX-4bit"
    elif lower_shortcut_name == "qwen3-1.7b":
        return "Qwen/Qwen3-1.7B-MLX-4bit"
    elif lower_shortcut_name == "qwen3-4b":
        return "Qwen/Qwen3-4B-MLX-4bit"
    elif lower_shortcut_name in ("qwen3-30b-a3b", "qwen3-moe-30b-a3b"):
        return "Qwen/Qwen3-30B-A3B-MLX-4bit"
    else:
        return shortcut_name


# The two model implementations: "dense" is the plain contiguous-KV path (good
# single-stream latency); "paged" is the block-paged serving path (continuous
# batching, prefix caching). `kind` selects between them.
_MODELS = {
    "dense": Qwen3DenseModel,
    "paged": Qwen3PagedModel,
}


def load_model(model_name: str, mlx_model, kind: str = "paged", **kwargs):
    model_name = shortcut_name_to_full_name(model_name)
    if not model_name.startswith("Qwen/Qwen3"):
        raise ValueError(f"unsupported model {model_name}")
    if kind not in _MODELS:
        raise ValueError(f"unknown model kind {kind!r}; expected one of {list(_MODELS)}")
    return _MODELS[kind](mlx_model, **kwargs)
