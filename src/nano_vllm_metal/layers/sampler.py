import mlx.core as mx
import copy


def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: mx.array):
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)
        logprobs = copy.copy(logprobs)
        if top_k is not None and top_k > 0:
            mask_elements = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[
                :, top_k:
            ]
            logprobs[:, mask_elements] = -mx.inf
        if top_p is not None and top_p > 0:
            sorted_idx = mx.argsort(-logprobs, axis=-1)
            sorted_logprobs = logprobs[:, sorted_idx]
            cumsum = mx.cumsum(mx.exp(sorted_logprobs), axis=-1)
            keep = cumsum < top_p
            keep[..., 0] = True
            logprobs[:, sorted_idx] = mx.where(keep, sorted_logprobs, -mx.inf)
        return mx.random.categorical(logprobs / temp, axis=-1)

    return sample
