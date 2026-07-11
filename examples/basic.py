"""Minimal nano-vllm-metal example: batched generation with the LLM API.

    pdm run python examples/basic.py
"""

from nano_vllm_metal import LLM, SamplingParams


def main():
    # Paged serving path with prefix caching; raise max_num_seqs for more
    # concurrent throughput (batching amortizes better at 16-32 on larger models).
    llm = LLM(
        "qwen3-0.6b",
        kind="paged",
        max_num_seqs=8,
        enable_prefix_caching=True,
    )
    tok = llm.tokenizer

    questions = [
        "What is the capital of France? Answer in one word.",
        "Name three primary colors.",
        "Write a haiku about the ocean.",
    ]
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": q}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for q in questions
    ]

    outputs = llm.generate(prompts, SamplingParams(max_tokens=128, temperature=0.0))
    for q, out in zip(questions, outputs):
        print(f"\n=== {q}\n{out.text.strip()}")


if __name__ == "__main__":
    main()
