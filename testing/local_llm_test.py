"""Test script to run local LLM"""

import os
import time

from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from load_puzzles import data_gen, format_puzzles_vllm

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", None)
THINKING = False
TEMPERATURE = 0.6  # Recommended @ https://unsloth.ai/docs/models/qwen3.5
NUM_PUZZLES = 40


if __name__ == "__main__":
    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN not set")

    # Initialize vLLM engine — allocates KV cache via PagedAttention upfront
    llm = LLM(
        model="Qwen/Qwen3.5-4B",
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=8192,
    )

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=2048,
    )

    raw_puzzles = data_gen(dataset_name="train", num_data=NUM_PUZZLES)
    messages = format_puzzles_vllm(raw_puzzles)

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

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    print("Displaying a few responses...")
    for i in range(len(outputs)):
        if i % 10 == 0:
            print(f"=== RESPONSE {i} ===")
            print(outputs[i].outputs[0].text)

    print("\n=== STATS ===\n")
    print(
        f"Generated {total_tokens} to solve {NUM_PUZZLES} in {t_elapsed:.1f}s ({total_tokens / t_elapsed:.1f} tok/s)"
    )
