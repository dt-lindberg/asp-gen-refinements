"""Qwen3-30B-A3B-Instruct-2507 via vLLM, batched inference (GGUF Q4_K_M)."""

import json
import os
import sys
import time

from dotenv import load_dotenv

from logger import get_logger
from prompt import SYSTEM, USER

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", None)

MODEL_PATH = "/home/dlindberg/.cache/huggingface/hub/models--unsloth--Qwen3-30B-A3B-Instruct-2507-GGUF/snapshots/eea7b2be5805a5f151f8847ede8e5f9a9284bf77/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"

BATCH_SIZE = 352
MAX_TOKENS = 2000
TEMPERATURE = 1.0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: inference.py <name>", file=sys.stderr)
        sys.exit(1)
    name = sys.argv[1]
    log = get_logger(name)

    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN not set")

    from vllm import LLM, SamplingParams

    t_load_start = time.perf_counter()
    llm = LLM(
        model=MODEL_PATH,
        max_model_len=4096,
        max_num_seqs=BATCH_SIZE,
        gpu_memory_utilization=0.93,
    )
    t_load = time.perf_counter() - t_load_start
    log.info(f"Model loaded in {t_load:.2f}s")

    tokenizer = llm.get_tokenizer()
    chat = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    prompts = [prompt] * BATCH_SIZE

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    log.info(f"Generating batch_size={BATCH_SIZE}...")
    t_gen_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    t_gen = time.perf_counter() - t_gen_start

    n_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    log.info(
        f"Generated {n_tokens} tokens in {t_gen:.2f}s ({n_tokens / t_gen:.2f} tok/s)"
    )

    result = {
        "total_tokens": n_tokens,
        "generation_time": t_gen,
        "tokens_per_second": n_tokens / t_gen,
    }
    with open(f"outputs/{name}.json", "w") as f:
        json.dump(result, f, indent=2)
