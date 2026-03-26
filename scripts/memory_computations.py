# Load a few inputs from reattempts to figure out how long
# inputs get; Count the number of tokens

import os
import json

FILE_PATH = os.path.join(
    "..",
    "caches",
    "vllm-qwen3-30b-refinements-v2-a6",
    "vllm_qwen3-30b-local_reattempt.json",
)

if __name__ == "__main__":
    with open(FILE_PATH, "r") as f:
        contents = f.read()
        contents = json.loads(contents)

    # Find the longest inputs
    inputs = list(contents.keys())
    inputs.sort(key=lambda _key: len(_key), reverse=True)
    TOP_K = 3
    longest_inputs = inputs[:TOP_K]

    print("Longest input:")
    print("===" * 45)
    print(longest_inputs[0])
    print("===" * 45)

    print(
        f"Top {TOP_K} longest programs (# of WORDS): {list(map(lambda x: len(x), longest_inputs))}",
    )
    print("Approximate tokens (1.3–1.7 tokens per word):")
    for inp in longest_inputs:
        print(f"{len(inp) * 1.3:.2f} - {len(inp) * 1.7:.2f}")
