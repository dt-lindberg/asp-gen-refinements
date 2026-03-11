"""Test script to run Qwen3.5-4B locally using Unsloth for fast inference."""

import os
import time

from dotenv import load_dotenv
from unsloth import FastVisionModel

from load_puzzles import data_gen, format_puzzles_vllm

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", None)
THINKING = False
TEMPERATURE = 0.6  # Recommended @ https://unsloth.ai/docs/models/qwen3.5
MAX_SEQ_LENGTH = 16384
NUM_PUZZLES = 3


def format_message_for_vision(messages: list[dict]) -> list[dict]:
    """Convert plain-string message content to the vision-compatible format
    (list of typed dicts) that the Qwen3.5 processor expects."""
    formatted = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            formatted.append(
                {"role": msg["role"], "content": [{"type": "text", "text": content}]}
            )
        else:
            formatted.append(msg)
    return formatted


if __name__ == "__main__":
    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN not set")

    # Load model with Unsloth — 4-bit quantization for reduced VRAM usage
    # Qwen3.5 is always a vision model, so we use FastVisionModel
    t_load_start = time.perf_counter()
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name="unsloth/Qwen3.5-4B",
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        token=HF_TOKEN,
    )

    # Enable Unsloth's native 2x faster inference
    FastVisionModel.for_inference(model)
    t_load = time.perf_counter() - t_load_start
    print(f"Model loaded in {t_load:.1f}s", flush=True)

    raw_puzzles = data_gen(dataset_name="train", num_data=NUM_PUZZLES)
    messages_batch = format_puzzles_vllm(raw_puzzles)

    # Generate responses one at a time (Unsloth uses HF generate, not batched vLLM)
    print(f"\nGenerating responses for {NUM_PUZZLES} puzzles...", flush=True)
    t_start = time.perf_counter()
    total_tokens = 0
    ttft_recorded = False
    ttft = 0.0
    outputs = []

    for i, messages in enumerate(messages_batch):
        if not THINKING:
            # Append /no_think instruction to disable thinking mode
            messages[-1]["content"] += "\n/no_think"

        # Convert plain-string content to vision-compatible format
        vision_messages = format_message_for_vision(messages)

        input_text = tokenizer.apply_chat_template(
            vision_messages, add_generation_prompt=True
        )
        inputs = tokenizer(
            text=input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        t_gen_start = time.perf_counter()
        generated = model.generate(
            **inputs,
            max_new_tokens=8192,
            temperature=TEMPERATURE,
            top_p=0.95,
            top_k=20,
            use_cache=True,
        )
        t_gen = time.perf_counter() - t_gen_start

        # Record time to first token (first puzzle's generation latency)
        if not ttft_recorded:
            ttft = t_gen
            ttft_recorded = True

        # Decode only the new tokens (skip the input prompt tokens)
        new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
        n_tokens = len(new_tokens)
        total_tokens += n_tokens
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        outputs.append(response_text)

        tok_per_sec = n_tokens / t_gen if t_gen > 0 else 0
        print(
            f"\n=== PUZZLE {i + 1}/{NUM_PUZZLES} === ({n_tokens} tokens in {t_gen:.1f}s, {tok_per_sec:.1f} tok/s)"
        )
        print(response_text, flush=True)

    t_elapsed = time.perf_counter() - t_start

    print("\n=== STATS ===\n")
    print(f"Model load time:    {t_load:.1f}s")
    print(f"Time to first token (puzzle 1): {ttft:.1f}s")
    print(
        f"Generated {total_tokens} tokens to solve {NUM_PUZZLES} puzzles "
        f"in {t_elapsed:.1f}s ({total_tokens / t_elapsed:.1f} tok/s)"
    )
