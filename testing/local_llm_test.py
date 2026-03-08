"""Test script to run local LLM"""

import os
import time

from dotenv import load_dotenv
from vllm import LLM, SamplingParams

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", None)
THINKING = False
TEMPERATURE = 0.6  # Recommended @ https://unsloth.ai/docs/models/qwen3.5


if __name__ == "__main__":
    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN not set")

    # Initialize vLLM engine — allocates KV cache via PagedAttention upfront
    llm = LLM(
        model="Qwen/Qwen3.5-9B",
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=8192,
    )

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=2048,
    )

    # Test message
    messages = [
        {
            "role": "user",
            "content": "Write a short Answer Set Program (ASP) that models and solves N-queens for N=4. Include comments to explain your reasoning.",
        }
    ]

    # Generate response — dispatches to C++/CUDA, no Python token loop
    print("Generating response...", flush=True)
    t_start = time.perf_counter()
    outputs = llm.chat(
        messages=messages,
        sampling_params=sampling_params,
        chat_template_kwargs={"enable_thinking": THINKING},
        use_tqdm=False,
    )
    t_elapsed = time.perf_counter() - t_start

    response = outputs[0].outputs[0].text
    n_tokens = len(outputs[0].outputs[0].token_ids)

    print("=== RESPONSE ===")
    print(response)
    print("\n=== STATS ===")
    print(f"{n_tokens} tokens in {t_elapsed:.1f}s ({n_tokens / t_elapsed:.1f} tok/s)")
