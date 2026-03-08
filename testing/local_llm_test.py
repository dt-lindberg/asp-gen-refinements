"""Test script to run local LLM"""

import os
from dotenv import load_dotenv
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", None)


if __name__ == "__main__":
    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN not set")

    # Initialize model
    model_name = "Qwen/Qwen3.5-9B"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    # Initialize processor
    processor = AutoProcessor.from_pretrained(model_name)

    # Test message
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Write a short Answer Set Program (ASP) that models and solves N-queens for N=4. Include comments to explain your reasoning.",
                }
            ],
        }
    ]

    # Apply chat template and prepare inputs
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    inputs = processor(
        text=[text],
        return_tensors="pt",
    ).to(model.device)

    # Generate response
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=16384,
    )

    # Parse the actual response (exclude the prompt tokens)
    output_ids = generated_ids[0][len(inputs.input_ids[0]) :].tolist()

    # Parse the reasoning (if any)
    # Get the token_id of the </think> token (denotes end of reasoning)
    thinking_end_token = processor.tokenizer.encode(
        "</think>", add_special_tokens=False
    )[0]

    # Try to find the last thinking_end_token, set index to 0 if None
    try:
        thinking_end_index = len(output_ids) - output_ids[::-1].index(
            thinking_end_token
        )
    except ValueError:
        thinking_end_index = 0

    # Extract the reasoning as text
    reasoning = processor.tokenizer.decode(
        output_ids[:thinking_end_index],
        skip_special_tokens=True,
    ).strip()

    # Extract the response as text
    response = processor.tokenizer.decode(
        output_ids[thinking_end_index:],
        skip_special_tokens=True,
    ).strip()

    print("=== REASONING ===")
    print(reasoning)
    print("=== RESPONSE ===")
    print(response)
